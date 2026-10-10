"""Test-session environment defaults and the network guard.

Sets environment defaults (before any backend module is imported) so tests
never start background network work or call a real LLM, and blocks network
access through the ``socket`` module for the whole session: name lookups
(``getaddrinfo``, ``gethostbyname``, ``gethostbyname_ex``), connections
(``connect``, ``connect_ex``) and datagrams (``sendto``) to anything but the
loopback interface. urllib, http.client, httpx, the openai SDK and asyncio
all go through these calls, so a test that forgets to stub OpenDota, the LLM
or a download fails instead of silently depending on the network. An attempt
made by a background thread is charged to the test that is running at that
moment.

The self-learned knowledge (``backend/data/learned``, gitignored, written by
the learning pipeline) is replaced by an empty directory for the session:
what a machine has learned changes the coach's knowledge blocks and its
per-hero benchmarks, and three tests failed the first time a full learning
run had filled the real directory (2026-10-11). Tests of the learning
pipeline point it at their own directory as before.

Apart from these two no shared fixtures live here on purpose: each test
module owns its fixtures.
"""
import atexit
import ipaddress
import os
import shutil
import socket
import tempfile

import pytest

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")
# The server would build the knowledge stores in a background thread at every
# TestClient start, next to whatever the test is doing with the same data.
os.environ.setdefault("RAG_WARM_UP", "0")
# Knowledge retrieval ranks lexically unless a test asks for more: otherwise
# the result of a pipeline or server test would depend on what this machine
# happens to have (a vector cache in cache/rag, a downloaded embedding model,
# the optional packages, embedding-API settings). The RAG test modules set
# their own mode and directories.
os.environ.setdefault("RAG_EMBEDDINGS", "off")
if "RAG_INDEX_DIR" not in os.environ:
    _rag_index_dir = tempfile.mkdtemp(prefix="dota-analyzer-test-rag-")
    os.environ["RAG_INDEX_DIR"] = _rag_index_dir
    atexit.register(shutil.rmtree, _rag_index_dir, ignore_errors=True)
# Empty = the built-in default, and (as for the key below) python-dotenv then
# leaves the variable alone instead of taking a developer's value from .env.
for _name in ("RAG_MODEL", "RAG_TOP_K", "RAG_AGENT_SEARCH_LIMIT",
              "RAG_EMBED_API_BASE_URL", "RAG_EMBED_API_KEY", "RAG_EMBED_API_MODEL"):
    os.environ.setdefault(_name, "")
# An empty value also stops python-dotenv from injecting a developer's real
# key from .env (load_dotenv never overrides variables that already exist).
os.environ.setdefault("DEEPSEEK_API_KEY", "")


class NetworkBlockedError(RuntimeError):
    """A test tried to resolve or connect to a non-loopback address."""


# Attempts of the running test ("connect api.opendota.com:443"). Code under
# test often catches the error (a failed request is a normal case for it), so
# the guard also records the attempt and fails the test afterwards.
_blocked_attempts: list[str] = []


def _is_loopback(host) -> bool:
    if host is None:  # getaddrinfo(None, port): the local host
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    host = str(host).strip("[]").split("%", 1)[0]
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _check(action: str, host, port="") -> None:
    if _is_loopback(host):
        return
    target = f"{action} {host}:{port}" if port != "" else f"{action} {host}"
    _blocked_attempts.append(target)
    raise NetworkBlockedError(f"network access is blocked in tests ({target})")


def _guard_address(action: str, address) -> None:
    # AF_INET / AF_INET6 addresses are (host, port, ...) tuples; a unix
    # socket path (str / bytes) never leaves the machine.
    if isinstance(address, tuple) and address:
        _check(action, address[0], address[1] if len(address) > 1 else "")


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_sendto = socket.socket.sendto
_real_getaddrinfo = socket.getaddrinfo
_real_gethostbyname = socket.gethostbyname
_real_gethostbyname_ex = socket.gethostbyname_ex


def _guarded_connect(self, address):
    _guard_address("connect", address)
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    _guard_address("connect", address)
    return _real_connect_ex(self, address)


def _guarded_sendto(self, data, *args):
    # sendto(data, address) or sendto(data, flags, address): an unconnected
    # (UDP) socket names its destination here, not in connect().
    if args:
        _guard_address("send to", args[-1])
    return _real_sendto(self, data, *args)


def _guarded_getaddrinfo(host, port, *args, **kwargs):
    _check("resolve", host)
    return _real_getaddrinfo(host, port, *args, **kwargs)


def _guarded_gethostbyname(hostname):
    _check("resolve", hostname)
    return _real_gethostbyname(hostname)


def _guarded_gethostbyname_ex(hostname):
    _check("resolve", hostname)
    return _real_gethostbyname_ex(hostname)


# Installed at import time so module imports during collection are covered too.
socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex
socket.socket.sendto = _guarded_sendto
socket.getaddrinfo = _guarded_getaddrinfo
socket.gethostbyname = _guarded_gethostbyname
socket.gethostbyname_ex = _guarded_gethostbyname_ex


@pytest.fixture(autouse=True)
def _no_network():
    """Fail any test during which a non-loopback connection was attempted.

    Yields the list of blocked attempts of the running test.
    """
    del _blocked_attempts[:]
    yield _blocked_attempts
    if _blocked_attempts:
        attempts = ", ".join(sorted(set(_blocked_attempts)))
        del _blocked_attempts[:]
        pytest.fail(f"the test tried to reach the network: {attempts}", pytrace=False)


@pytest.fixture(scope="session", autouse=True)
def _no_local_learned_knowledge(tmp_path_factory):
    """Point every reader and writer of the learned directory at an empty one.

    ``learning.base.LEARNED_DIR`` is what the knowledge base and the
    benchmark overlay read at call time; the other names are the copies the
    learner modules took at import (their default output directory, the run
    manifest, the video registry). A module that cannot be imported has no
    copy to redirect.
    """
    import importlib

    learned = tmp_path_factory.mktemp("learned")
    targets = {
        "base": {"LEARNED_DIR": learned, "SMOKE_DIR": learned / "_smoke"},
        "pipeline": {"LEARNED_DIR": learned, "LAST_LEARN_FILE": learned / "_last_learn.json"},
        "stats_learner": {"LEARNED_DIR": learned},
        "match_miner": {"LEARNED_DIR": learned},
        "llm_distiller": {"LEARNED_DIR": learned},
        "video_learner": {"LEARNED_DIR": learned, "VIDEOS_SEEN_FILE": learned / "_videos_seen.json"},
    }
    patch = pytest.MonkeyPatch()
    for name, attributes in targets.items():
        try:
            module = importlib.import_module(f"backend.app.ai.learning.{name}")
        except Exception:
            continue
        for attribute, value in attributes.items():
            patch.setattr(module, attribute, value, raising=False)
    yield learned
    patch.undo()
