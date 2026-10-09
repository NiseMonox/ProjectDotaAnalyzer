"""Tests for backend.app.main: startup sequencing, lifespan and middleware."""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import asyncio
import sys
import time
import types

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from backend.app import config, main


@pytest.fixture
def fake_modules(monkeypatch):
    """Fake data_manager + learning pipeline recording call order."""
    calls: list[str] = []
    dm = types.ModuleType("backend.app.ai.data_manager")
    dm.meta = {"status": "no_data"}
    dm.remote = ("7.40", False)
    dm.check_error = None

    def get_meta():
        calls.append("get_meta")
        return dict(dm.meta)

    def check_patch_version():
        calls.append("check_patch_version")
        if dm.check_error:
            raise dm.check_error
        return dm.remote

    def fetch_and_save_gamedata(force=False):
        calls.append(f"fetch(force={force})")
        time.sleep(0.05)  # learning must still wait for this
        return {"patch_version": "7.40", "heroes": 1, "items": 2}

    dm.get_meta = get_meta
    dm.check_patch_version = check_patch_version
    dm.fetch_and_save_gamedata = fetch_and_save_gamedata

    pipeline = types.ModuleType("backend.app.ai.learning.pipeline")
    pipeline.due = True

    def should_learn():
        calls.append("should_learn")
        return pipeline.due

    def run_learning():
        calls.append("run_learning")
        return {"total_chunks_written": 3, "patch": "7.40"}

    pipeline.should_learn = should_learn
    pipeline.run_learning = run_learning
    monkeypatch.setitem(sys.modules, "backend.app.ai.data_manager", dm)
    monkeypatch.setitem(sys.modules, "backend.app.ai.learning.pipeline", pipeline)
    return types.SimpleNamespace(calls=calls, dm=dm, pipeline=pipeline)


def test_gamedata_runs_before_learning(fake_modules, monkeypatch):
    monkeypatch.setenv("GAMEDATA_AUTO_UPDATE", "1")
    monkeypatch.setenv("AI_COACH_AUTO_LEARN", "1")
    asyncio.run(main.run_startup_maintenance())
    assert fake_modules.calls == ["get_meta", "fetch(force=True)", "should_learn", "run_learning"]


def test_flags_disable_each_step(fake_modules, monkeypatch):
    monkeypatch.setenv("GAMEDATA_AUTO_UPDATE", "0")
    monkeypatch.setenv("AI_COACH_AUTO_LEARN", "1")
    asyncio.run(main.run_startup_maintenance())
    assert fake_modules.calls == ["should_learn", "run_learning"]

    fake_modules.calls.clear()
    monkeypatch.setenv("GAMEDATA_AUTO_UPDATE", "1")
    monkeypatch.setenv("AI_COACH_AUTO_LEARN", "0")
    asyncio.run(main.run_startup_maintenance())
    assert fake_modules.calls == ["get_meta", "fetch(force=True)"]


@pytest.mark.parametrize("meta,remote,check_error,expect_fetch", [
    ({"patch_version": "7.40", "age_days": 1}, ("7.40", False), None, False),   # up to date
    ({"patch_version": "7.39", "age_days": 1}, ("7.40", True), None, True),     # new patch
    ({"patch_version": "7.40", "age_days": 30}, ("7.40", False), None, True),   # stale
    ({"patch_version": "7.40", "age_days": 1}, None, OSError("offline"), False),
    ({"patch_version": "7.40", "age_days": 30}, None, OSError("offline"), True),
])
def test_gamedata_update_decision(fake_modules, meta, remote, check_error, expect_fetch):
    fake_modules.dm.meta = meta
    fake_modules.dm.remote = remote
    fake_modules.dm.check_error = check_error
    asyncio.run(main._check_and_update_gamedata())
    assert ("fetch(force=True)" in fake_modules.calls) is expect_fetch


def test_learning_skipped_when_fresh_and_failures_are_swallowed(fake_modules):
    fake_modules.pipeline.due = False
    asyncio.run(main._run_learning_if_due())
    assert fake_modules.calls == ["should_learn"]

    def boom():
        raise RuntimeError("network down")

    fake_modules.pipeline.should_learn = boom
    asyncio.run(main._run_learning_if_due())  # logs, never raises

    fake_modules.dm.get_meta = boom
    asyncio.run(main._check_and_update_gamedata())


def test_lifespan_does_not_wait_for_maintenance(monkeypatch):
    started = []

    async def slow_maintenance():
        started.append(True)
        await asyncio.sleep(3600)

    monkeypatch.setattr(main, "run_startup_maintenance", slow_maintenance)
    t0 = time.monotonic()
    with TestClient(main.app) as client:
        assert client.get("/api/match/none").status_code == 404
        task = main.app.state.startup_task
    assert time.monotonic() - t0 < 5
    assert started == [True]
    assert task.cancelled()


def _stream_app() -> FastAPI:
    app = FastAPI()

    @app.get("/sse")
    async def sse():
        async def gen():
            for i in range(3):
                yield f"data: {i}\n\n"
                await asyncio.sleep(0)
        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/json")
    async def big_json():
        return {"items": ["x" * 50] * 100}

    app.add_middleware(main.SSEAwareGZipMiddleware, minimum_size=500)
    return app


def test_gzip_compresses_json_but_never_event_streams():
    client = TestClient(_stream_app())
    j = client.get("/json", headers={"Accept-Encoding": "gzip"})
    assert j.headers["content-encoding"] == "gzip"
    assert j.json()["items"][0] == "x" * 50

    for accept in ("*/*", "text/event-stream"):
        s = client.get("/sse", headers={"Accept-Encoding": "gzip", "Accept": accept})
        assert "content-encoding" not in s.headers
        assert s.text == "data: 0\n\ndata: 1\n\ndata: 2\n\n"


def test_sse_chunks_are_forwarded_unbuffered():
    """Each SSE chunk must reach the transport as its own body message."""
    app = _stream_app()
    messages: list[dict] = []

    async def run():
        async def receive():
            await asyncio.sleep(10)
            return {"type": "http.disconnect"}

        async def send(msg):
            messages.append(msg)

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
            "path": "/sse", "raw_path": b"/sse", "root_path": "", "scheme": "http",
            "query_string": b"", "server": ("t", 80), "client": ("1.1.1.1", 1),
            "headers": [(b"accept-encoding", b"gzip"), (b"accept", b"*/*")],
        }
        await app(scope, receive, send)

    asyncio.run(run())
    bodies = [m["body"] for m in messages if m["type"] == "http.response.body" and m.get("body")]
    assert bodies == [b"data: 0\n\n", b"data: 1\n\n", b"data: 2\n\n"]


def test_static_and_api_headers():
    with TestClient(main.app) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert index.headers["cache-control"] == "no-cache"
        assert index.headers["x-content-type-options"] == "nosniff"
        api = client.get("/api/match/none", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in api.headers  # no CORS by default


def test_configure_logging_sets_backend_level(monkeypatch):
    import logging

    monkeypatch.setattr(config, "LOG_LEVEL", "DEBUG")
    main.configure_logging()
    assert logging.getLogger("backend").level == logging.DEBUG
    monkeypatch.setattr(config, "LOG_LEVEL", "nonsense")
    main.configure_logging()
    assert logging.getLogger("backend").level == logging.INFO
