"""The test suite must not depend on the network (see conftest.py).

conftest.py blocks every non-loopback name lookup, connection and datagram
made through the socket module and fails the test that tried. These tests pin
that guard down, so it cannot be weakened unnoticed.
"""
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
        capture_output=True, text=True, cwd=tmp_path, timeout=120,
    )
    assert result.returncode != 0, result.stdout
    assert "the test tried to reach the network: resolve api.opendota.com" in result.stdout
    # reported for the test that did it (as an error after its body), not for the other one
    assert "ERROR test_probe.py::test_swallows_the_error" in result.stdout
    assert "2 passed, 1 error" in result.stdout


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
