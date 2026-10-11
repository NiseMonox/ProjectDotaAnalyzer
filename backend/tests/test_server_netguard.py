"""The test suite must not depend on the network (see conftest.py).

conftest.py blocks every non-loopback name lookup, connection and datagram
made through the socket module and fails the test that tried. These tests pin
that guard down, so it cannot be weakened unnoticed.
"""
import asyncio
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path

import pytest


def test_loopback_connections_still_work():
    server = socket.socket()
    try:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        client = socket.create_connection(("127.0.0.1", server.getsockname()[1]), timeout=5)
        conn, _addr = server.accept()
        client.sendall(b"ping")
        assert conn.recv(4) == b"ping"
        client.close()
        conn.close()
    finally:
        server.close()
    assert socket.getaddrinfo("localhost", 80) and socket.getaddrinfo(None, 80)
    assert socket.gethostbyname("127.0.0.1") == "127.0.0.1"
    assert socket.gethostbyname_ex("127.0.0.1")[2] == ["127.0.0.1"]


def test_loopback_datagrams_still_work():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(5)
        sender.sendto(b"ping", receiver.getsockname())
        sender.sendto(b"pong", 0, receiver.getsockname())      # the form with flags
        assert [receiver.recv(4), receiver.recv(4)] == [b"ping", b"pong"]
    finally:
        receiver.close()
        sender.close()


def _udp_send(*args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        return sock.sendto(b"x", *args)
    finally:
        sock.close()


@pytest.mark.parametrize("attempt, recorded", [
    (lambda: urllib.request.urlopen("https://api.opendota.com/api/constants/patch", timeout=2),
     "resolve api.opendota.com"),
    (lambda: socket.create_connection(("api.deepseek.com", 443), timeout=2), "resolve api.deepseek.com"),
    (lambda: socket.socket().connect(("192.0.2.1", 80)), "connect 192.0.2.1:80"),      # TEST-NET-1, by address
    (lambda: socket.socket().connect_ex(("192.0.2.1", 80)), "connect 192.0.2.1:80"),
    (lambda: socket.gethostbyname("api.opendota.com"), "resolve api.opendota.com"),
    (lambda: socket.gethostbyname_ex("api.opendota.com"), "resolve api.opendota.com"),
    (lambda: _udp_send(("192.0.2.1", 53)), "send to 192.0.2.1:53"),          # no connect(): UDP
    (lambda: _udp_send(0, ("192.0.2.1", 53)), "send to 192.0.2.1:53"),
])
def test_outside_hosts_are_blocked_and_recorded(_no_network, attempt, recorded):
    with pytest.raises(Exception) as info:
        attempt()
    assert "network access is blocked in tests" in str(info.value)
    # The attempt is recorded, which fails a test even when the code under test swallowed the
    # error. This test made it on purpose: take it off the record again.
    assert _no_network == [recorded]
    _no_network.clear()


def test_asyncio_connections_by_address_are_blocked_too(_no_network):
    """asyncio's default event loop on Windows (the proactor) connects through the Win32 API,
    not through socket.connect: without its own guard a literal address would get out."""
    async def attempt():
        await asyncio.wait_for(asyncio.open_connection("192.0.2.1", 80), timeout=10)

    with pytest.raises(Exception) as info:
        asyncio.run(attempt())
    assert "network access is blocked in tests" in str(info.value)
    assert _no_network == ["connect 192.0.2.1:80"]
    _no_network.clear()


def test_asyncio_loopback_connections_still_work():
    """... and the guard leaves the loop itself alone: its self-pipe is a socket pair on
    127.0.0.1 on Windows, and the tests' local servers listen there."""
    async def scenario():
        async def echo(reader, writer):
            writer.write(await reader.read(4))
            await writer.drain()
            writer.close()

        server = await asyncio.start_server(echo, "127.0.0.1", 0)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
            writer.write(b"ping")
            await writer.drain()
            answer = await asyncio.wait_for(reader.read(4), timeout=10)
            writer.close()
            await writer.wait_closed()
            return answer
        finally:
            server.close()
            await server.wait_closed()

    assert asyncio.run(scenario()) == b"ping"


def test_the_proactor_guard_checks_connect_and_sendto(_no_network):
    """The wrapper conftest.py puts around asyncio's Windows proactor, on a stand-in class:
    the real one only exists on Windows."""
    guard = next(module for module in list(sys.modules.values())
                 if getattr(module, "_blocked_attempts", None) is _no_network)
    calls = []

    class Proactor:
        def connect(self, conn, address):
            calls.append(("connect", address))
            return "future"

        def sendto(self, conn, buf, flags=0, addr=None):
            calls.append(("sendto", flags, addr))
            return "future"

    guard._guard_proactor(Proactor)
    proactor = Proactor()
    assert proactor.connect(object(), ("127.0.0.1", 8000)) == "future"
    assert proactor.connect(object(), ("::1", 8000, 0, 0)) == "future"
    assert proactor.sendto(object(), b"x", addr=("127.0.0.1", 53)) == "future"     # as asyncio calls it
    assert proactor.sendto(object(), b"x") == "future"                             # a connected socket: no address
    assert calls == [("connect", ("127.0.0.1", 8000)), ("connect", ("::1", 8000, 0, 0)),
                     ("sendto", 0, ("127.0.0.1", 53)), ("sendto", 0, None)]
    for attempt in (lambda: proactor.connect(object(), ("192.0.2.1", 443)),
                    lambda: proactor.sendto(object(), b"x", addr=("192.0.2.1", 53)),
                    lambda: proactor.sendto(object(), b"x", 0, ("2001:db8::1", 53, 0, 0))):
        with pytest.raises(guard.NetworkBlockedError):
            attempt()
    assert len(calls) == 4                                                          # nothing reached the real methods
    assert _no_network == ["connect 192.0.2.1:443", "send to 192.0.2.1:53", "send to 2001:db8::1:53"]
    _no_network.clear()


def test_a_swallowed_attempt_still_fails_the_test(tmp_path):
    """Code under test usually catches the error (a failed request is a normal
    case for it), so the guard has to fail the test by itself."""
    shutil.copy(Path(__file__).with_name("conftest.py"), tmp_path / "conftest.py")
    (tmp_path / "test_probe.py").write_text(textwrap.dedent("""
        import urllib.request


        def test_swallows_the_error():
            try:
                urllib.request.urlopen("https://api.opendota.com/api/health", timeout=2)
            except Exception:
                pass


        def test_stays_local():
            assert True
    """), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
        capture_output=True, encoding="utf-8", errors="replace", cwd=tmp_path, timeout=120,
    )
    assert result.returncode != 0, result.stdout
    assert "the test tried to reach the network: resolve api.opendota.com" in result.stdout
    # reported for the test that did it (as an error after its body), not for the other one
    assert "ERROR test_probe.py::test_swallows_the_error" in result.stdout
    assert "2 passed, 1 error" in result.stdout


def test_what_the_shell_left_behind_does_not_reach_the_suite(tmp_path):
    """`$env:DEV = "1"; .\\run.ps1` is how development mode is started on Windows, and unlike
    `DEV=1 ./run.sh` the variable stays set in that PowerShell window: six tests of the server's
    startup failed when the suite was run there afterwards. conftest.py pins the mode, and
    tells the configuration not to read .env, in a way child processes inherit."""
    shutil.copy(Path(__file__).with_name("conftest.py"), tmp_path / "conftest.py")
    (tmp_path / "test_probe.py").write_text(textwrap.dedent("""
        import os


        def test_environment():
            assert os.environ["DEV"] == "0"
            assert os.environ["DOTA_ANALYZER_SKIP_DOTENV"] == "1"
            assert os.environ["DEEPSEEK_API_KEY"] == ""
    """), encoding="utf-8")
    env = {name: value for name, value in os.environ.items()
           if name not in ("DOTA_ANALYZER_SKIP_DOTENV", "DEEPSEEK_API_KEY")}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
        capture_output=True, encoding="utf-8", errors="replace", cwd=tmp_path, timeout=120,
        env={**env, "DEV": "1"},
    )
    assert result.returncode == 0, result.stdout
    assert "1 passed" in result.stdout


def test_the_suite_does_not_read_what_this_machine_has_learned():
    """conftest.py points the learned-knowledge directory at an empty one.

    The first full learning run on a development machine (2026-10-11) made
    three tests fail: per-hero benchmarks and learned chunks from
    backend/data/learned had changed the coach's output.
    """
    from backend.app import config
    from backend.app.ai import pro_patterns
    from backend.app.ai.learning import base, pipeline, stats_learner, video_learner

    real = config.LEARNED_DATA_DIR
    for path in (base.LEARNED_DIR, base.SMOKE_DIR.parent, pipeline.LEARNED_DIR, pipeline.LAST_LEARN_FILE.parent,
                 stats_learner.LEARNED_DIR, video_learner.LEARNED_DIR, video_learner.VIDEOS_SEEN_FILE.parent):
        assert path != real and real not in path.parents
    # the coach's readers resolve the directory when they are called: the benchmark overlay
    # through this path, the knowledge base through base.load_learned_chunks (base.LEARNED_DIR)
    assert pro_patterns._learned_benchmarks_path().parent == base.LEARNED_DIR
