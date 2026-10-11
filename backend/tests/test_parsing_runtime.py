"""Clarity runtime: availability checks, parser factory, subprocess lifecycle.

A fake ``java`` executable (a Python script) stands in for the JVM so no
replay, JDK or network is needed. On POSIX it is the script itself, made
executable; on Windows a ``java.cmd`` that hands its arguments to the script,
which makes the "JVM" a child of the process the parser started and so tests
the kill of the whole process tree on every run.
"""
from __future__ import annotations

import asyncio
import codecs
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import pytest  # noqa: E402

from backend.app import osutil, parser  # noqa: E402
from backend.app.parser import (  # noqa: E402
    ClarityParser,
    MockParser,
    ParserUnavailableError,
    ReplayParseError,
    check_clarity_available,
    get_parser,
)

REPO = Path(__file__).resolve().parents[2]

FAKE_JAVA = r"""import json, os, subprocess, sys, time
args = sys.argv[1:]
def log(line):
    with open(os.environ["FAKE_JAVA_LOG"], "a", encoding="utf-8") as f:
        f.write(line + "\n")
if args == ["-version"]:
    sys.stderr.buffer.write(bytes.fromhex(os.environ.get("FAKE_JAVA_VERSION_NOISE_HEX", "")))
    sys.stderr.flush()
    sys.stderr.write(os.environ.get("FAKE_JAVA_VERSION", 'openjdk version "17.0.9" 2023-10-17') + "\n")
    log("version")
    sys.exit(0)
log(json.dumps(args))
replay = args[-1]
mode = os.path.basename(replay).split(".")[0]
if mode.startswith("sleep"):
    # A JVM that hangs, with a child process of its own. Each holds a file lock for as long
    # as it lives (the system releases it when the process ends, however it ends): that is
    # how the test sees whether they are gone. Process ids would not do: Windows hands the
    # id of a dead process to the next one.
    sys.path.insert(0, os.environ["FAKE_JAVA_REPO"])
    from backend.app import osutil
    mine = osutil.FileLock(replay + ".jvm.lock")
    assert mine.acquire(blocking=False)
    child = subprocess.Popen([sys.executable, "-c", "import sys, time; sys.path.insert(0, sys.argv[1]); "
                              "from backend.app import osutil; lock = osutil.FileLock(sys.argv[2]); "
                              "lock.acquire(); time.sleep(60)", os.environ["FAKE_JAVA_REPO"], replay + ".child.lock"])
    probe = osutil.FileLock(replay + ".child.lock")
    while probe.acquire(blocking=False):    # free: the child has not got that far yet
        probe.release()
        time.sleep(0.02)
    with open(replay + ".running", "w", encoding="utf-8") as f:
        f.write(f"{os.getpid()} {child.pid}")
    time.sleep(60)
elif mode == "fail":
    sys.stderr.write("Exception in thread main java.lang.SECRET_STACK_TRACE\n" * 50)
    sys.exit(65)
elif mode == "oom":
    sys.exit(3)
elif mode == "garbage":
    print("not json")
elif mode.startswith("slow"):
    log(f"start {time.time()}")
    time.sleep(1.0)
    log(f"end {time.time()}")
    print(json.dumps({"coord_space": "world", "players": []}))
else:
    out = {"schema_version": 2, "coord_space": "world", "match_id": 5, "duration": 60,
           "players": [{"player_slot": 0, "hero_name": "axe", "player_name": "\u5f20\u4e09"}]}
    sys.stdout.buffer.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))
"""


def _install_fake_java(directory: Path) -> Path:
    """Write the fake JVM into ``directory``; the path to give the parser as its Java."""
    if osutil.IS_WINDOWS:
        script = directory / "fake_java.py"
        script.write_text(FAKE_JAVA, encoding="utf-8")
        java = directory / "java.cmd"
        # cmd.exe reads a batch file in the console (OEM) code page.
        java.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\nexit /b %ERRORLEVEL%\r\n',
                        encoding="oem", newline="")
        return java
    java = directory / "java"
    java.write_text(f"#!{sys.executable}\n{FAKE_JAVA}", encoding="utf-8")
    java.chmod(0o755)
    return java


@pytest.fixture
def env(tmp_path, monkeypatch):
    java = _install_fake_java(tmp_path)
    jar = tmp_path / "clarity-analyzer.jar"
    jar.write_bytes(b"PK")
    log = tmp_path / "java.log"
    monkeypatch.setenv("FAKE_JAVA_LOG", str(log))
    monkeypatch.setenv("FAKE_JAVA_REPO", str(REPO))
    monkeypatch.setattr(parser, "CLARITY_JAR_PATH", jar)
    monkeypatch.setattr(parser, "JAVA_PATH", str(java))
    monkeypatch.setattr(parser, "_clarity_slots", None)
    parser.reset_clarity_probe_cache()
    yield {"java": str(java), "jar": jar, "log": log, "dir": tmp_path}
    parser.reset_clarity_probe_cache()


@pytest.fixture(params=["asyncio", "thread"])
def subprocess_mode(request, monkeypatch):
    """Both ways the parser runs a JVM.

    An asyncio subprocess, and ``subprocess.Popen`` with a thread waiting
    for it: what is left on an event loop without subprocess support, which
    is the loop uvicorn installs on Windows with ``--reload`` (DEV=1) or
    several workers. That loop is imitated here, so the second way is tested
    on every system.
    """
    if request.param == "thread":
        async def unsupported(*args, **kwargs):
            raise NotImplementedError

        monkeypatch.setattr(asyncio, "create_subprocess_exec", unsupported)
    return request.param


def _log_lines(env) -> list[str]:
    return env["log"].read_text(encoding="utf-8").splitlines() if env["log"].exists() else []


def _replay(env, name: str) -> str:
    path = env["dir"] / f"{name}.dem"
    path.write_bytes(b"PBDEMS2\0")
    return str(path)


def _running(replay: str) -> Path:
    """Written by the fake JVM of a "sleep" replay once it and its child process are both up."""
    return Path(replay + ".running")


def _wait_for(path: Path, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() or not path.read_text(encoding="utf-8").strip():
        if time.monotonic() > deadline:
            raise AssertionError(f"{path} never appeared")
        time.sleep(0.05)


def _ended(replay: str, who: str, timeout: float = 15.0) -> bool:
    """Whether the fake JVM ("jvm") or the process it started ("child") of a "sleep" replay
    has ended: the file lock it held for life can be taken."""
    lock = osutil.FileLock(f"{replay}.{who}.lock")
    deadline = time.monotonic() + timeout
    while True:
        if lock.acquire(blocking=False):
            lock.release()
            return True
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Availability / factory
# ---------------------------------------------------------------------------

def test_get_parser_auto_raises_when_jar_missing(env):
    env["jar"].unlink()
    with pytest.raises(ParserUnavailableError):
        get_parser("auto")
    assert asyncio.run(check_clarity_available()) is False


def test_get_parser_auto_raises_when_java_missing(env, monkeypatch):
    monkeypatch.setattr(parser, "JAVA_PATH", str(env["dir"] / "no-such-java"))
    with pytest.raises(ParserUnavailableError) as info:
        get_parser("auto")
    assert not isinstance(info.value, ReplayParseError)
    assert asyncio.run(check_clarity_available()) is False


def test_get_parser_auto_rejects_old_java(env, monkeypatch):
    monkeypatch.setenv("FAKE_JAVA_VERSION", 'java version "1.8.0_292"')
    with pytest.raises(ParserUnavailableError):
        get_parser("auto")


def test_get_parser_auto_returns_clarity_and_probes_java_once(env):
    assert asyncio.run(check_clarity_available()) is True
    assert asyncio.run(check_clarity_available()) is True
    parser_obj = get_parser("auto")
    assert isinstance(parser_obj, ClarityParser) and not isinstance(parser_obj, MockParser)
    get_parser("auto")
    assert _log_lines(env).count("version") == 1


def test_java_version_output_in_another_encoding_does_not_break_the_probe(env, monkeypatch):
    """A JVM on a Chinese Windows may put text in the system code page in front of the version
    (for example the line about JAVA_TOOL_OPTIONS with a path in it): not UTF-8."""
    noise = "Picked up JAVA_TOOL_OPTIONS: -Djava.io.tmpdir=C:\\用户\\临时\n".encode("gbk")
    with pytest.raises(UnicodeDecodeError):
        noise.decode("utf-8")
    monkeypatch.setenv("FAKE_JAVA_VERSION_NOISE_HEX", noise.hex())
    assert asyncio.run(check_clarity_available()) is True
    monkeypatch.setenv("FAKE_JAVA_VERSION", 'java version "1.8.0_292"')
    parser.reset_clarity_probe_cache()
    assert asyncio.run(check_clarity_available()) is False      # ... and the version is still read


def test_get_parser_explicit_types():
    assert isinstance(get_parser("mock"), MockParser)
    assert get_parser("opendota").name() == "opendota"
    with pytest.raises(ValueError):
        get_parser("nope")


# ---------------------------------------------------------------------------
# Subprocess lifecycle
# ---------------------------------------------------------------------------

def test_parse_success_passes_jvm_options_and_decodes_utf8(env, monkeypatch, subprocess_mode):
    monkeypatch.setenv("CLARITY_JVM_XMX", "2g")
    replay = _replay(env, "ok")
    match = asyncio.run(ClarityParser().parse(replay))
    assert match.match_id == 5
    assert match.players[0].player_name == "张三"
    assert match.players[0].hero_slug == "axe"
    args = json.loads(_log_lines(env)[-1])
    assert "-Xmx2g" in args and args[-3:] == ["-jar", str(env["jar"]), replay]
    # The JVM's own text output is UTF-8 whatever the system's code page is
    # (stdout / stderr.encoding: Java 19+, file.encoding: before).
    assert {"-Dfile.encoding=UTF-8", "-Dstdout.encoding=UTF-8", "-Dstderr.encoding=UTF-8"} <= set(args[:-3])
    assert "version" not in _log_lines(env)  # no java -version per parse


def test_parse_works_on_a_selector_event_loop(env):
    """The loop uvicorn installs on Windows for --reload (DEV=1) and --workers. It cannot start
    asyncio subprocesses there (NotImplementedError), so on Windows this is the thread fallback
    on the real thing; on POSIX the selector loop can, and it is one more run of the asyncio way."""
    loop = asyncio.SelectorEventLoop()
    try:
        match = loop.run_until_complete(ClarityParser().parse(_replay(env, "ok")))
    finally:
        loop.close()
    assert match.match_id == 5 and match.players[0].player_name == "张三"
    assert not parser._live_clarity_pids


def test_invalid_xmx_falls_back_to_default(env, monkeypatch):
    monkeypatch.setenv("CLARITY_JVM_XMX", "2g -Dinjected=1")
    asyncio.run(ClarityParser().parse(_replay(env, "ok")))
    args = json.loads(_log_lines(env)[-1])
    assert "-Xmx1536m" in args and not any("injected" in a for a in args)


def test_parse_failure_is_short_and_hides_stderr(env, caplog, subprocess_mode):
    with pytest.raises(ReplayParseError) as info:
        asyncio.run(ClarityParser().parse(_replay(env, "fail")))
    message = str(info.value)
    assert "SECRET_STACK_TRACE" not in message and len(message) < 120
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "SECRET_STACK_TRACE" in logged and len(logged) < 5000  # truncated in logs


def test_paths_java_cannot_take_are_a_windows_matter(monkeypatch):
    monkeypatch.setattr(osutil, "IS_WINDOWS", False)
    assert parser._paths_java_cannot_take("/home/张三/replays/1.dem", "/srv/日本語/clarity-analyzer.jar") == []


def test_paths_outside_the_windows_code_page_are_named(monkeypatch, request):
    """java.exe gets its command line in the system's ANSI code page: a folder name outside of
    it reaches the JVM as "?" and the replay is "not found"."""
    monkeypatch.setattr(osutil, "IS_WINDOWS", True)
    plain, runes = r"C:\dota\replays\1.dem", "C:\\\u16a0\u16a2\u16a6\\tools\\clarity-analyzer.jar"
    try:
        "x".encode("mbcs")
    except LookupError:
        # Not Windows: let the code page of a Japanese Windows stand in for the system's.
        def search(name):
            return codecs.lookup("cp932") if name == "mbcs" else None

        codecs.register(search)
        request.addfinalizer(lambda: codecs.unregister(search))
        chinese, japanese = r"C:\Users\张三\dota\1.dem", r"C:\Users\山田\dota\1.dem"
        assert parser._paths_java_cannot_take(plain, chinese, japanese, runes) == [chinese, runes]
    else:
        # Windows: whatever its code page is, it has no runes (unless it is UTF-8, which has everything).
        try:
            runes.encode("mbcs")
            expected = []
        except UnicodeEncodeError:
            expected = [runes]
        assert parser._paths_java_cannot_take(plain, runes) == expected
    assert parser._paths_java_cannot_take(plain) == []


def test_failed_parse_names_a_path_java_cannot_take(env, monkeypatch, caplog):
    replay = _replay(env, "fail")
    monkeypatch.setattr(parser, "_paths_java_cannot_take", lambda *paths: [replay] if replay in paths else [])
    with caplog.at_level("WARNING", logger="backend.app.parser"):
        with pytest.raises(ReplayParseError):
            asyncio.run(ClarityParser().parse(replay))
    hints = [r.getMessage() for r in caplog.records if "system code page" in r.getMessage()]
    assert len(hints) == 1 and replay in hints[0] and "ASCII" in hints[0]
    caplog.clear()
    monkeypatch.setattr(parser, "_paths_java_cannot_take", lambda *paths: [])
    with caplog.at_level("WARNING", logger="backend.app.parser"):
        with pytest.raises(ReplayParseError):
            asyncio.run(ClarityParser().parse(replay))
    assert not [r for r in caplog.records if "system code page" in r.getMessage()]


@pytest.mark.parametrize("name,expected", [("oom", "memory"), ("garbage", "unreadable")])
def test_parse_other_failures(env, name, expected, subprocess_mode):
    with pytest.raises(ReplayParseError) as info:
        asyncio.run(ClarityParser().parse(_replay(env, name)))
    assert expected in str(info.value)


def test_missing_jar_at_parse_time(env):
    env["jar"].unlink()
    with pytest.raises(ParserUnavailableError):
        asyncio.run(ClarityParser().parse(_replay(env, "ok")))


def test_missing_java_at_parse_time(env, subprocess_mode):
    """Java removed after the availability check: not the replay's fault."""
    broken = ClarityParser(java_path=str(env["dir"] / "no-such-java"))
    with pytest.raises(ParserUnavailableError) as info:
        asyncio.run(broken.parse(_replay(env, "ok")))
    assert not isinstance(info.value, ReplayParseError)
    assert not parser._live_clarity_pids


def test_timeout_kills_the_jvm_and_its_children(env, monkeypatch, subprocess_mode):
    # Long enough for the fake JVM and its child to be up on a slow machine.
    monkeypatch.setattr(parser, "_clarity_timeout_s", lambda: 5)
    replay = _replay(env, "sleep")
    started = time.monotonic()
    with pytest.raises(ReplayParseError) as info:
        asyncio.run(ClarityParser().parse(replay))
    assert "timed out" in str(info.value)
    assert time.monotonic() - started < 40
    assert _running(replay).exists()            # both were running when the time was up ...
    assert _ended(replay, "jvm")                # ... and neither is now
    assert _ended(replay, "child")
    assert not parser._live_clarity_pids


def test_cancellation_kills_the_jvm(env, subprocess_mode):
    replay = _replay(env, "sleep2")

    async def scenario():
        task = asyncio.create_task(ClarityParser().parse(replay))
        await asyncio.to_thread(_wait_for, _running(replay))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert _ended(replay, "jvm")
    assert _ended(replay, "child")
    assert not parser._live_clarity_pids
    # the concurrency permit was released
    assert parser._get_clarity_slots().acquire(blocking=False)
    parser._get_clarity_slots().release()


def test_every_jvm_is_adopted_for_as_long_as_it_runs(env, monkeypatch, subprocess_mode):
    """On Windows osutil.adopt_child() puts the JVM into a job object: a kill then ends what
    that JVM started and nothing else (taskkill /T goes by parent ids, which Windows reuses),
    and the JVM cannot outlive the server. It must happen before anything can ask for the
    kill, and the job must be given back when the parse is over, however it ends."""
    events = []
    real_adopt, real_release, real_kill = osutil.adopt_child, osutil.release_child, osutil.kill_process_tree

    def record(name, real):
        def call(pid):
            events.append((name, pid, pid in parser._live_clarity_pids))
            return real(pid)
        return call

    monkeypatch.setattr(osutil, "adopt_child", record("adopt", real_adopt))
    monkeypatch.setattr(osutil, "release_child", record("release", real_release))
    monkeypatch.setattr(osutil, "kill_process_tree", record("kill", real_kill))

    assert asyncio.run(ClarityParser().parse(_replay(env, "ok"))).match_id == 5
    (adopt, pid, listed), (release, released_pid, still_listed) = events
    assert (adopt, release) == ("adopt", "release") and pid == released_pid
    assert listed is False and still_listed is False    # adopted before it can be killed, released after

    del events[:]
    with pytest.raises(ReplayParseError):
        asyncio.run(ClarityParser().parse(_replay(env, "fail")))
    assert [name for name, _pid, _listed in events] == ["adopt", "release"]

    del events[:]
    replay = _replay(env, "sleep4")

    async def cancelled_parse():
        task = asyncio.create_task(ClarityParser().parse(replay))
        await asyncio.to_thread(_wait_for, _running(replay))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancelled_parse())
    names = [name for name, _pid, _listed in events]
    assert names[0] == "adopt" and names[-1] == "release" and "kill" in names[1:-1]
    assert len({pid for _name, pid, _listed in events}) == 1
    assert _ended(replay, "jvm") and _ended(replay, "child")
    assert not osutil._child_jobs


def test_concurrency_limit_serializes_parses(env, monkeypatch):
    monkeypatch.setenv("CLARITY_MAX_CONCURRENCY", "1")

    async def scenario():
        await asyncio.gather(
            ClarityParser().parse(_replay(env, "slow1")),
            ClarityParser().parse(_replay(env, "slow2")),
        )

    asyncio.run(scenario())
    events = [ln.split() for ln in _log_lines(env) if ln.startswith(("start", "end"))]
    assert [e[0] for e in events] == ["start", "end", "start", "end"]
    assert float(events[2][1]) >= float(events[1][1])


def test_concurrency_limit_allows_parallel_parses(env, monkeypatch):
    monkeypatch.setenv("CLARITY_MAX_CONCURRENCY", "2")

    async def scenario():
        await asyncio.gather(
            ClarityParser().parse(_replay(env, "slow1")),
            ClarityParser().parse(_replay(env, "slow2")),
        )

    asyncio.run(scenario())
    events = [ln.split()[0] for ln in _log_lines(env) if ln.startswith(("start", "end"))]
    assert events[:2] == ["start", "start"]


def test_legacy_check_requirements_raises_unavailable(env):
    env["jar"].unlink()
    with pytest.raises(ParserUnavailableError):
        ClarityParser()._check_requirements()


# ---------------------------------------------------------------------------
# Probe cache, event-loop blocking, leftover JVMs
# ---------------------------------------------------------------------------

def test_failed_probe_is_retried_after_a_while(env, monkeypatch):
    monkeypatch.setenv("FAKE_JAVA_VERSION", 'java version "1.8.0_292"')
    assert asyncio.run(check_clarity_available()) is False
    assert asyncio.run(check_clarity_available()) is False
    assert _log_lines(env).count("version") == 1               # a failure is cached too
    monkeypatch.setenv("FAKE_JAVA_VERSION", 'openjdk version "17.0.9" 2023-10-17')
    assert asyncio.run(check_clarity_available()) is False      # ... for a while
    monkeypatch.setattr(parser, "_JAVA_PROBE_RETRY_S", 0.0)
    assert asyncio.run(check_clarity_available()) is True       # then probed again
    monkeypatch.setattr(parser, "_JAVA_PROBE_RETRY_S", 300.0)
    assert asyncio.run(check_clarity_available()) is True       # a success never expires
    assert _log_lines(env).count("version") == 2


def test_availability_check_never_waits_for_the_probe_lock(env):
    assert asyncio.run(check_clarity_available()) is True
    held, release = threading.Event(), threading.Event()

    def other_callers_probe():  # holds the lock like a java -version run in a worker thread
        with parser._java_probe_lock:
            held.set()
            release.wait(3)

    worker = threading.Thread(target=other_callers_probe)
    worker.start()
    assert held.wait(5)
    try:
        started = time.monotonic()
        assert asyncio.run(check_clarity_available()) is True
        assert get_parser("auto").name() == "clarity"
        assert time.monotonic() - started < 1.5      # answered from the cache, lock or no lock
    finally:
        release.set()
        worker.join()


def test_decode_and_transform_run_off_the_event_loop(env, monkeypatch):
    seen = {}
    original = ClarityParser._transform_data

    def spy(self, raw):
        seen["thread"] = threading.current_thread()
        return original(self, raw)

    monkeypatch.setattr(ClarityParser, "_transform_data", spy)
    match = asyncio.run(ClarityParser().parse(_replay(env, "ok")))
    assert match.match_id == 5
    assert seen["thread"] is not threading.main_thread()


def test_kill_running_parsers_stops_leftover_jvms(env, subprocess_mode):
    replay = _replay(env, "sleep3")

    async def scenario():
        task = asyncio.create_task(ClarityParser().parse(replay))
        await asyncio.to_thread(_wait_for, _running(replay))
        assert len(parser._live_clarity_pids) == 1
        assert parser.kill_running_parsers() == 1      # what the atexit hook and the server shutdown do
        # The parse ends because its JVM was killed, long before the JVM would stop by itself
        # (60 s). The replay is not at fault: a ReplayParseError would make the server answer
        # "corrupted replay" and delete the stored file.
        with pytest.raises(ParserUnavailableError) as info:
            await asyncio.wait_for(task, timeout=10)
        assert "stopped" in str(info.value)

    started = time.monotonic()
    asyncio.run(scenario())
    assert time.monotonic() - started < 40
    assert _ended(replay, "jvm")
    assert _ended(replay, "child")
    assert not parser._live_clarity_pids and parser.kill_running_parsers() == 0
    assert not parser._stopped_clarity_pids        # a later process with that pid is judged on its own


# ---------------------------------------------------------------------------
# tools/setup_parser.sh: the committed jar is used, not rebuilt
# ---------------------------------------------------------------------------

@pytest.mark.skipif(osutil.IS_WINDOWS, reason="tools/setup_parser.sh is the POSIX script (its stand-in java and "
                                              "gradlew are shell scripts); Windows has tools/setup_parser.ps1")
def test_setup_script_does_not_rebuild_an_existing_jar(tmp_path):
    """The install runbook ran it on the server: with any JDK other than the pinned one the tracked
    jar came out with other bytes, and the next `git pull` that brought a new jar was refused."""
    tools = tmp_path / "tools"
    source = tools / "clarity-analyzer"
    source.mkdir(parents=True)
    script = tools / "setup_parser.sh"
    script.write_bytes((REPO / "tools" / "setup_parser.sh").read_bytes())
    (source / "build.gradle").write_text("// stub\n", encoding="utf-8")
    # A "build" that leaves a marker and a new jar, so a rebuild cannot go unnoticed
    (source / "gradlew").write_text(
        "#!/bin/sh\ntouch built.marker\nmkdir -p build/libs\nprintf 'rebuilt' > build/libs/clarity-analyzer.jar\n",
        encoding="utf-8")
    jdk = tmp_path / "jdk" / "bin"
    jdk.mkdir(parents=True)
    (jdk / "java").write_text(
        "#!/bin/sh\nif [ \"$1\" = \"-version\" ]; then echo 'openjdk version \"17.0.9\" 2023-10-17' >&2; exit 0; fi\n"
        "echo 'clarity-analyzer 1.4.0'; exit 64\n", encoding="utf-8")
    (jdk / "javac").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    for tool in (jdk / "java", jdk / "javac"):
        tool.chmod(0o755)
    jar = tools / "clarity-analyzer.jar"
    jar.write_bytes(b"committed")

    def run(**extra):
        env = {"PATH": os.environ.get("PATH", ""), "JAVA_HOME": str(tmp_path / "jdk"), **extra}
        return subprocess.run(["bash", str(script)], capture_output=True, encoding="utf-8", errors="replace",
                              env=env, timeout=60)

    for value in ({}, {"REBUILD": "0"}, {"REBUILD": ""}):
        done = run(**value)
        assert done.returncode == 0, done.stdout + done.stderr
        assert "Using the existing JAR" in done.stdout and "analyzer 1.4.0" in done.stdout
        assert jar.read_bytes() == b"committed" and not (source / "built.marker").exists()
    # Asked for, or with no jar at all, it builds
    done = run(REBUILD="1")
    assert done.returncode == 0 and "Building from source" in done.stdout
    assert jar.read_bytes() == b"rebuilt" and (source / "built.marker").exists()
    jar.unlink()
    (source / "built.marker").unlink()
    done = run()
    assert done.returncode == 0 and jar.read_bytes() == b"rebuilt" and (source / "built.marker").exists()
    # The runbook no longer tells a server to rebuild
    readme = (REPO / "deploy" / "README.md").read_text(encoding="utf-8")
    assert "Do NOT rebuild the jar on the server" in readme and "git checkout -- tools/clarity-analyzer.jar" in readme
    assert "rebuild the jar with `bash tools/setup_parser.sh`" not in readme


# ---------------------------------------------------------------------------
# The Windows scripts: run.ps1 and tools/setup_parser.ps1
# ---------------------------------------------------------------------------

POWERSHELL_SCRIPTS = ("run.ps1", "tools/setup_parser.ps1")
# Windows PowerShell 5.1 (part of Windows) first: it is the oldest version the scripts support.
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
needs_powershell = pytest.mark.skipif(
    POWERSHELL is None, reason="needs PowerShell: part of Windows, not installed on this machine")


def _powershell(*arguments: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", *arguments],
        capture_output=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL,
        env={**os.environ, **env}, cwd=REPO, timeout=300,
    )


@pytest.mark.parametrize("script", POWERSHELL_SCRIPTS)
def test_powershell_scripts_are_ascii_with_lf_line_endings(script):
    """Windows PowerShell 5.1 reads a script without a byte order mark in the system code page
    (cp936, cp932 ...): one non-ASCII character in a UTF-8 file can end a string early there.
    LF line endings are what .gitattributes checks out on every system."""
    data = (REPO / script).read_bytes()
    assert data.isascii(), [line for line in data.splitlines() if not line.isascii()][:3]
    assert b"\r" not in data
    assert not data.startswith(b"\xef\xbb\xbf")


def test_powershell_scripts_avoid_what_windows_powershell_5_does_not_have():
    """Syntax of PowerShell 7 that the parser of 5.1 rejects (checked as text, so also where
    there is no PowerShell to parse with)."""
    for script in POWERSHELL_SCRIPTS:
        code = "\n".join(line.split("#", 1)[0] for line in (REPO / script).read_text(encoding="utf-8").splitlines())
        for token, name in ((" && ", "the && operator"), (" || ", "the || operator"), ("??", "the ?? operator"),
                            ("?.", "the ?. operator"), (" ? ", "the ternary operator")):
            assert token not in code, f"{script} uses {name}"


def test_python_that_run_ps1_runs_is_valid_and_survives_powershell(monkeypatch):
    """run.ps1 hands two pieces of Python to an interpreter with -c. Windows PowerShell 5.1 does
    not escape a double quote inside an argument, and it expands $ and the backtick in the
    string: the code must hold none of them. The second piece asks the application for its
    settings (so .env is read by the application, not by the script): run here, it must
    answer with the one line of JSON the script expects."""
    script = (REPO / "run.ps1").read_text(encoding="utf-8")
    version_probe = re.search(r'-c "(import sys; .*?)" 2>\$null', script)
    settings_probe = re.search(r'\$probe = "(import json.*)"\r?\n', script)
    assert version_probe and settings_probe
    for code in (version_probe.group(1), settings_probe.group(1)):
        assert not set(code) & set('"`$'), code
        compile(code, "run.ps1", "exec")
    done = subprocess.run([sys.executable, "-c", version_probe.group(1)], capture_output=True,
                          encoding="utf-8", errors="replace", timeout=60)
    assert done.stdout.strip() == "%d.%d" % sys.version_info[:2]

    env = {**os.environ, "HOST": "127.0.0.1", "PORT": "8123", "DEV": "1", "DEEPSEEK_API_KEY": "",
           "JAVA_PATH": "no-such-java-anywhere", "CLARITY_JAR_PATH": "tools/clarity-analyzer.jar"}
    done = subprocess.run([sys.executable, "-c", settings_probe.group(1)], capture_output=True,
                          encoding="utf-8", errors="replace", cwd=REPO, env=env, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.isascii()                    # whatever the console code page, nothing to damage
    settings = json.loads(done.stdout.strip().splitlines()[-1])
    assert settings == {"host": "127.0.0.1", "port": 8123, "dev": True, "java": False, "jar": True,
                        "jar_path": str(REPO / "tools" / "clarity-analyzer.jar")}
    # every setting the script uses afterwards is one the probe answers with
    assert set(re.findall(r"\$settings\.(\w+)", script)) == set(settings)


@needs_powershell
@pytest.mark.parametrize("script", POWERSHELL_SCRIPTS)
def test_powershell_scripts_parse_without_errors(script):
    done = _powershell(
        "-Command",
        "$errors = $null; $tokens = $null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        "$env:SCRIPT_UNDER_TEST, [ref]$tokens, [ref]$errors); "
        "foreach ($e in @($errors)) { Write-Output $e.ToString() }; exit @($errors).Count",
        SCRIPT_UNDER_TEST=str(REPO / script),
    )
    assert done.returncode == 0, done.stdout + done.stderr


@needs_powershell
def test_setup_parser_ps1_checks_java_and_the_committed_jar():
    """The script for real, with the JDK the server would use: it finds Java, reads its version
    and starts the committed jar. It must not rebuild: the jar's bytes are the same afterwards."""
    from backend.app.config import CLARITY_JAR_PATH, JAVA_PATH

    jar = REPO / "tools" / "clarity-analyzer.jar"
    if Path(CLARITY_JAR_PATH) != jar or not jar.is_file() or not parser._run_java_version(JAVA_PATH):
        pytest.skip("needs Java 17+ and the committed tools/clarity-analyzer.jar")
    if not osutil.IS_WINDOWS:
        pytest.skip("the script looks for java.exe: it is the Windows counterpart of tools/setup_parser.sh")
    before = jar.read_bytes()
    java_home = str(Path(JAVA_PATH).resolve().parent.parent) if os.sep in JAVA_PATH else os.environ.get("JAVA_HOME", "")
    done = _powershell("-File", str(REPO / "tools" / "setup_parser.ps1"), JAVA_HOME=java_home, REBUILD="")
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    assert "Using the existing JAR" in output and "Setup Complete" in output
    assert "analyzer version unknown" not in output and "(analyzer " in output
    assert 'version "' in output
    assert jar.read_bytes() == before
