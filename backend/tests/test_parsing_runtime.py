"""Clarity runtime: availability checks, parser factory, subprocess lifecycle.

A fake ``java`` executable (a Python script) stands in for the JVM so no
replay, JDK or network is needed.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import pytest  # noqa: E402

from backend.app import parser  # noqa: E402
from backend.app.parser import (  # noqa: E402
    ClarityParser,
    MockParser,
    ParserUnavailableError,
    ReplayParseError,
    check_clarity_available,
    get_parser,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="fake java script needs a POSIX shell")

FAKE_JAVA = '''#!{python}
import json, os, subprocess, sys, time
args = sys.argv[1:]
if args == ["-version"]:
    sys.stderr.write(os.environ.get("FAKE_JAVA_VERSION", 'openjdk version "17.0.9" 2023-10-17') + "\\n")
    with open(os.environ["FAKE_JAVA_LOG"], "a") as f:
        f.write("version\\n")
    sys.exit(0)
with open(os.environ["FAKE_JAVA_LOG"], "a") as f:
    f.write(json.dumps(args) + "\\n")
replay = args[-1]
mode = os.path.basename(replay).split(".")[0]
if mode.startswith("sleep"):
    child = subprocess.Popen(["sleep", "60"])
    with open(replay + ".pids", "w") as f:
        f.write(f"{{os.getpid()}} {{child.pid}}")
    time.sleep(60)
elif mode == "fail":
    sys.stderr.write("Exception in thread main java.lang.SECRET_STACK_TRACE\\n" * 50)
    sys.exit(65)
elif mode == "oom":
    sys.exit(3)
elif mode == "garbage":
    print("not json")
elif mode.startswith("slow"):
    with open(os.environ["FAKE_JAVA_LOG"], "a") as f:
        f.write(f"start {{time.time()}}\\n")
    time.sleep(0.6)
    with open(os.environ["FAKE_JAVA_LOG"], "a") as f:
        f.write(f"end {{time.time()}}\\n")
    print(json.dumps({{"coord_space": "world", "players": []}}))
else:
    out = {{"schema_version": 2, "coord_space": "world", "match_id": 5, "duration": 60,
           "players": [{{"player_slot": 0, "hero_name": "axe", "player_name": "\\u5f20\\u4e09"}}]}}
    sys.stdout.buffer.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))
'''


@pytest.fixture
def env(tmp_path, monkeypatch):
    java = tmp_path / "java"
    java.write_text(FAKE_JAVA.format(python=sys.executable))
    java.chmod(0o755)
    jar = tmp_path / "clarity-analyzer.jar"
    jar.write_bytes(b"PK")
    log = tmp_path / "java.log"
    monkeypatch.setenv("FAKE_JAVA_LOG", str(log))
    monkeypatch.setattr(parser, "CLARITY_JAR_PATH", jar)
    monkeypatch.setattr(parser, "JAVA_PATH", str(java))
    monkeypatch.setattr(parser, "_clarity_slots", None)
    parser.reset_clarity_probe_cache()
    yield {"java": str(java), "jar": jar, "log": log, "dir": tmp_path}
    parser.reset_clarity_probe_cache()


def _log_lines(env) -> list[str]:
    return env["log"].read_text().splitlines() if env["log"].exists() else []


def _replay(env, name: str) -> str:
    path = env["dir"] / f"{name}.dem"
    path.write_bytes(b"PBDEMS2\0")
    return str(path)


def _process_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    # Still in the process table. Killed but not yet reaped by init (a zombie)
    # is gone for our purposes: /proc says so on Linux, ps where there is none (macOS).
    if Path("/proc/self/status").exists():
        try:
            return "State:\tZ" in Path(f"/proc/{pid}/status").read_text()
        except OSError:
            return True
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return not state or state.startswith("Z")


def _wait_for(path: Path, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists() or not path.read_text().strip():
        if time.monotonic() > deadline:
            raise AssertionError(f"{path} never appeared")
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


def test_get_parser_explicit_types():
    assert isinstance(get_parser("mock"), MockParser)
    assert get_parser("opendota").name() == "opendota"
    with pytest.raises(ValueError):
        get_parser("nope")


# ---------------------------------------------------------------------------
# Subprocess lifecycle
# ---------------------------------------------------------------------------

def test_parse_success_passes_jvm_options_and_decodes_utf8(env, monkeypatch):
    monkeypatch.setenv("CLARITY_JVM_XMX", "2g")
    match = asyncio.run(ClarityParser().parse(_replay(env, "ok")))
    assert match.match_id == 5
    assert match.players[0].player_name == "张三"
    assert match.players[0].hero_slug == "axe"
    args = json.loads(_log_lines(env)[-1])
    assert "-Xmx2g" in args and args[-3:-1] == ["-jar", str(env["jar"])]
    assert "version" not in _log_lines(env)  # no java -version per parse


def test_invalid_xmx_falls_back_to_default(env, monkeypatch):
    monkeypatch.setenv("CLARITY_JVM_XMX", "2g -Dinjected=1")
    asyncio.run(ClarityParser().parse(_replay(env, "ok")))
    args = json.loads(_log_lines(env)[-1])
    assert "-Xmx1536m" in args and not any("injected" in a for a in args)


def test_parse_failure_is_short_and_hides_stderr(env, caplog):
    with pytest.raises(ReplayParseError) as info:
        asyncio.run(ClarityParser().parse(_replay(env, "fail")))
    message = str(info.value)
    assert "SECRET_STACK_TRACE" not in message and len(message) < 120
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "SECRET_STACK_TRACE" in logged and len(logged) < 5000  # truncated in logs


@pytest.mark.parametrize("name,expected", [("oom", "memory"), ("garbage", "unreadable")])
def test_parse_other_failures(env, name, expected):
    with pytest.raises(ReplayParseError) as info:
        asyncio.run(ClarityParser().parse(_replay(env, name)))
    assert expected in str(info.value)


def test_missing_jar_at_parse_time(env):
    env["jar"].unlink()
    with pytest.raises(ParserUnavailableError):
        asyncio.run(ClarityParser().parse(_replay(env, "ok")))


def test_timeout_kills_the_jvm_and_its_children(env, monkeypatch):
    monkeypatch.setattr(parser, "_clarity_timeout_s", lambda: 1)
    replay = _replay(env, "sleep")
    started = time.monotonic()
    with pytest.raises(ReplayParseError) as info:
        asyncio.run(ClarityParser().parse(replay))
    assert "timed out" in str(info.value)
    assert time.monotonic() - started < 15
    pid, child = (int(v) for v in Path(replay + ".pids").read_text().split())
    assert _process_gone(pid)
    deadline = time.monotonic() + 5
    while not _process_gone(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _process_gone(child)


def test_cancellation_kills_the_jvm(env):
    replay = _replay(env, "sleep2")

    async def scenario():
        task = asyncio.create_task(ClarityParser().parse(replay))
        await asyncio.to_thread(_wait_for, Path(replay + ".pids"))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    pid, _child = (int(v) for v in Path(replay + ".pids").read_text().split())
    assert _process_gone(pid)
    # the concurrency permit was released
    assert parser._get_clarity_slots().acquire(blocking=False)
    parser._get_clarity_slots().release()


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


def test_kill_running_parsers_stops_leftover_jvms(env):
    replay = _replay(env, "sleep3")

    async def scenario():
        task = asyncio.create_task(ClarityParser().parse(replay))
        await asyncio.to_thread(_wait_for, Path(replay + ".pids"))
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
    assert time.monotonic() - started < 20
    pid, child = (int(v) for v in Path(replay + ".pids").read_text().split())
    assert _process_gone(pid)
    deadline = time.monotonic() + 5
    while not _process_gone(child) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _process_gone(child)
    assert not parser._live_clarity_pids and parser.kill_running_parsers() == 0
    assert not parser._stopped_clarity_pids        # a later process with that pid is judged on its own


# ---------------------------------------------------------------------------
# tools/setup_parser.sh: the committed jar is used, not rebuilt
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.name != "posix", reason="shell script")
def test_setup_script_does_not_rebuild_an_existing_jar(tmp_path):
    """The install runbook ran it on the server: with any JDK other than the pinned one the tracked
    jar came out with other bytes, and the next `git pull` that brought a new jar was refused."""
    tools = tmp_path / "tools"
    source = tools / "clarity-analyzer"
    source.mkdir(parents=True)
    script = tools / "setup_parser.sh"
    script.write_text((Path(__file__).resolve().parents[2] / "tools" / "setup_parser.sh").read_text(encoding="utf-8"),
                      encoding="utf-8")
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
        return subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=60)

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
    readme = (Path(__file__).resolve().parents[2] / "deploy" / "README.md").read_text(encoding="utf-8")
    assert "Do NOT rebuild the jar on the server" in readme and "git checkout -- tools/clarity-analyzer.jar" in readme
    assert "rebuild the jar with `bash tools/setup_parser.sh`" not in readme
