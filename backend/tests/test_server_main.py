"""Tests for backend.app.main: startup sequencing, lifespan and middleware."""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import asyncio
import io
import re
import subprocess
import sys
import threading
import time
import types

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from backend.app import config, main, osutil


UPDATED = {"status": "updated", "changed": True, "reason": "new_source_revision", "patch_version": "7.41",
           "patch_label": "7.41f", "source": "dotaconstants_github", "heroes": 127, "items": 501}
UP_TO_DATE = {"status": "up_to_date", "patch_version": "7.41", "patch_label": "7.41f"}


@pytest.fixture
def fake_modules(monkeypatch):
    """Fake data_manager, learning pipeline and retrieval package recording call order."""
    calls: list[str] = []
    dm = types.ModuleType("backend.app.ai.data_manager")
    dm.result = dict(UPDATED)

    def refresh_gamedata_if_outdated():
        calls.append("refresh_gamedata")
        time.sleep(0.05)  # learning must still wait for this
        return dict(dm.result)

    dm.refresh_gamedata_if_outdated = refresh_gamedata_if_outdated

    pipeline = types.ModuleType("backend.app.ai.learning.pipeline")
    pipeline.due = True
    pipeline.summary = {"total_chunks_written": 3, "patch": "7.41"}

    def should_learn():
        calls.append("should_learn")
        return pipeline.due

    def run_learning():
        calls.append("run_learning")
        return dict(pipeline.summary)

    pipeline.should_learn = should_learn
    pipeline.run_learning = run_learning

    rag = types.ModuleType("backend.app.ai.rag")
    rag.warm_up = lambda: calls.append("rag.warm_up")
    rag.refresh = lambda reason: calls.append(f"rag.refresh({reason})")
    rag.shutdown = lambda: calls.append("rag.shutdown")
    rag.status = lambda: {"mode": "lexical"}

    monkeypatch.setitem(sys.modules, "backend.app.ai.data_manager", dm)
    monkeypatch.setitem(sys.modules, "backend.app.ai.learning.pipeline", pipeline)
    monkeypatch.setitem(sys.modules, "backend.app.ai.rag", rag)
    # ``from .ai import data_manager`` reads the attribute already bound on the
    # package once the real submodule has been imported, so patch that too.
    import backend.app.ai as ai_pkg
    import backend.app.ai.learning as learning_pkg
    monkeypatch.setattr(ai_pkg, "data_manager", dm, raising=False)
    monkeypatch.setattr(ai_pkg, "rag", rag, raising=False)
    monkeypatch.setattr(learning_pkg, "pipeline", pipeline, raising=False)
    return types.SimpleNamespace(calls=calls, dm=dm, pipeline=pipeline, rag=rag)


def test_gamedata_runs_before_learning(fake_modules, monkeypatch):
    monkeypatch.setenv("GAMEDATA_AUTO_UPDATE", "1")
    monkeypatch.setenv("AI_COACH_AUTO_LEARN", "1")
    asyncio.run(main.run_startup_maintenance())
    # ... and the retrieval index hears about each change of the data it is built from
    assert fake_modules.calls == [
        "refresh_gamedata", "rag.refresh(gamedata)", "should_learn", "run_learning", "rag.refresh(learning)"]


def test_flags_disable_each_step(fake_modules, monkeypatch):
    monkeypatch.setenv("GAMEDATA_AUTO_UPDATE", "0")
    monkeypatch.setenv("AI_COACH_AUTO_LEARN", "1")
    asyncio.run(main.run_startup_maintenance())
    assert fake_modules.calls == ["should_learn", "run_learning", "rag.refresh(learning)"]

    fake_modules.calls.clear()
    monkeypatch.setenv("GAMEDATA_AUTO_UPDATE", "1")
    monkeypatch.setenv("AI_COACH_AUTO_LEARN", "0")
    asyncio.run(main.run_startup_maintenance())
    assert fake_modules.calls == ["refresh_gamedata", "rag.refresh(gamedata)"]


def _meta(minutes_ago: float, **changes) -> dict:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - minutes_ago * 60))
    return {"patch_version": "7.41", "patch_label": "7.41f", "data_label": "7.41f", "data_current": True,
            "outdated_format": False, "is_stale": False, "checked_at": stamp, **changes}


def test_start_shortly_after_a_check_does_not_check_again(fake_modules, monkeypatch, caplog):
    """A start 46 s after the previous check repeated its three requests, one of them to the GitHub
    API (60 an hour without a token); with DEV=1 every saved file is such a start."""
    monkeypatch.setenv("GAMEDATA_AUTO_UPDATE", "1")
    monkeypatch.setenv("AI_COACH_AUTO_LEARN", "0")
    assert config.GAMEDATA_RECHECK_MIN_HOURS == 0.25
    fake_modules.dm.get_meta = lambda: _meta(1)
    with caplog.at_level("INFO", logger="backend.app.main"):
        asyncio.run(main.run_startup_maintenance())
    assert fake_modules.calls == [] and "Game data was checked 1 min ago" in caplog.text
    # The check stands for 15 minutes, and only while the data on disk is usable and current
    for meta in (_meta(16), _meta(1, status="no_data"), _meta(1, outdated_format=True), _meta(1, is_stale=True),
                 _meta(1, data_current=False), _meta(1, checked_at=""), _meta(1, checked_at="yesterday"),
                 _meta(-5)):
        fake_modules.calls.clear()
        fake_modules.dm.get_meta = lambda meta=meta: meta
        asyncio.run(main.run_startup_maintenance())
        assert fake_modules.calls == ["refresh_gamedata", "rag.refresh(gamedata)"], meta
    # data_current None (a source without a revision date) is not "behind"
    fake_modules.calls.clear()
    fake_modules.dm.get_meta = lambda: _meta(2, data_current=None)
    asyncio.run(main.run_startup_maintenance())
    assert fake_modules.calls == []
    # The periodic check and the admin endpoint are not concerned
    assert asyncio.run(main._check_and_update_gamedata()) is True


def test_dev_mode_starts_no_learning_run(fake_modules, monkeypatch, caplog):
    """Each reload ended the running job and the next start began it again at hero 1."""
    monkeypatch.setenv("GAMEDATA_AUTO_UPDATE", "1")
    monkeypatch.setenv("AI_COACH_AUTO_LEARN", "1")
    monkeypatch.setattr(config, "DEV", True)
    with caplog.at_level("INFO", logger="backend.app.main"):
        asyncio.run(main.run_startup_maintenance())
    assert fake_modules.calls == ["refresh_gamedata", "rag.refresh(gamedata)"]
    assert "auto-learning is off in DEV mode" in caplog.text
    monkeypatch.setattr(config, "DEV", False)
    fake_modules.calls.clear()
    asyncio.run(main.run_startup_maintenance())
    assert fake_modules.calls[-3:] == ["should_learn", "run_learning", "rag.refresh(learning)"]


def test_api_documentation_is_not_served_outside_dev_mode():
    """/docs and /openapi.json listed every route of the public server, the admin ones included."""
    assert config.DEV is False
    assert main.app.docs_url is None and main.app.redoc_url is None and main.app.openapi_url is None
    assert main._API_DOCS == {"docs_url": None, "redoc_url": None, "openapi_url": None}
    paths = {getattr(route, "path", "") for route in main.app.routes}
    assert not {"/docs", "/redoc", "/openapi.json"} & paths and "/api/capabilities" in paths


@pytest.mark.parametrize("result,changed,level,message", [
    (UPDATED, True, "INFO",
     "Game data updated (new_source_revision): version 7.41f, 127 heroes, 501 items, source dotaconstants_github"),
    # downloaded, but the same data as before: nothing downstream is stale
    ({**UPDATED, "changed": False, "reason": "forced"}, False, "INFO", "Game data updated (forced): version 7.41f"),
    (UP_TO_DATE, False, "INFO", "Game data up to date: version 7.41f"),
    ({"status": "failed", "error": "URLError: offline", "patch_version": "7.41"}, False, "WARNING",
     "Game data update failed: URLError: offline. Using local data (patch 7.41)."),
])
def test_gamedata_step_reports_what_the_data_layer_did(fake_modules, caplog, result, changed, level, message):
    """The data layer decides whether to download (it never raises); the server logs the outcome."""
    fake_modules.dm.result = result
    with caplog.at_level("INFO", logger="backend.app.main"):
        assert asyncio.run(main._check_and_update_gamedata()) is changed
    assert fake_modules.calls == ["refresh_gamedata"] + (["rag.refresh(gamedata)"] if changed else [])
    records = [r for r in caplog.records if r.name == "backend.app.main"]
    assert [(r.levelname, r.getMessage().startswith(message)) for r in records] == [(level, True)]


def test_learning_skipped_when_fresh_and_failures_are_swallowed(fake_modules):
    fake_modules.pipeline.due = False
    asyncio.run(main._run_learning_if_due())
    assert fake_modules.calls == ["should_learn"]

    def boom():
        raise RuntimeError("network down")

    fake_modules.pipeline.should_learn = boom
    asyncio.run(main._run_learning_if_due())  # logs, never raises

    fake_modules.dm.refresh_gamedata_if_outdated = boom   # "never raises" is not taken on trust
    assert asyncio.run(main._check_and_update_gamedata()) is False

    del fake_modules.dm.refresh_gamedata_if_outdated      # an older data layer: nothing to run
    assert asyncio.run(main._check_and_update_gamedata()) is False


def test_retrieval_index_is_only_refreshed_when_learning_wrote_something(fake_modules):
    fake_modules.pipeline.summary = {"total_chunks_written": 0, "patch": "7.41"}
    asyncio.run(main._run_learning_if_due())
    assert fake_modules.calls == ["should_learn", "run_learning"]


def test_maintenance_works_without_the_retrieval_package(fake_modules, monkeypatch, caplog):
    """ai.rag missing or broken: the game data and learning steps are unaffected."""
    import backend.app.ai as ai_pkg

    monkeypatch.setenv("GAMEDATA_AUTO_UPDATE", "1")
    monkeypatch.setenv("AI_COACH_AUTO_LEARN", "1")
    monkeypatch.delattr(ai_pkg, "rag")
    monkeypatch.setitem(sys.modules, "backend.app.ai.rag", None)   # import raises ImportError
    assert main.rag_package() is None
    asyncio.run(main.run_startup_maintenance())
    assert fake_modules.calls == ["refresh_gamedata", "should_learn", "run_learning"]

    def broken(reason):
        raise RuntimeError("index is corrupt")

    fake_modules.calls.clear()
    fake_modules.rag.refresh = broken
    monkeypatch.setitem(sys.modules, "backend.app.ai.rag", fake_modules.rag)
    monkeypatch.setattr(ai_pkg, "rag", fake_modules.rag, raising=False)
    asyncio.run(main.run_startup_maintenance())
    assert fake_modules.calls == ["refresh_gamedata", "should_learn", "run_learning"]

    # The failure belongs to the retrieval step: the game data step still reports new
    # data (the periodic pass runs the learning step on that) and is not logged as failed.
    with caplog.at_level("INFO", logger="backend.app.main"):
        assert asyncio.run(main._check_and_update_gamedata()) is True
    assert "Knowledge retrieval refresh failed: index is corrupt" in caplog.text
    assert "Game data update failed" not in caplog.text
    assert main._rag_call("refresh", "gamedata") is None and main._rag_call("no_such_action") is None


def test_game_version_is_checked_again_while_the_server_runs(fake_modules, monkeypatch):
    """A letter patch released after the start is picked up without a restart."""
    monkeypatch.setenv("GAMEDATA_AUTO_UPDATE", "1")
    monkeypatch.setenv("AI_COACH_AUTO_LEARN", "1")
    monkeypatch.setattr(config, "GAMEDATA_RECHECK_HOURS", 0.01 / 3600)   # every 10 ms
    results = [UP_TO_DATE, UPDATED, UP_TO_DATE]

    def refresh_gamedata_if_outdated():
        fake_modules.calls.append("refresh_gamedata")
        return dict(results.pop(0)) if results else dict(UP_TO_DATE)

    fake_modules.dm.refresh_gamedata_if_outdated = refresh_gamedata_if_outdated

    async def scenario():
        task = asyncio.create_task(main.run_periodic_maintenance())
        for _ in range(500):
            if not results:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    # Nothing new: only the check. New data: the index is told and learning runs if due.
    assert fake_modules.calls[:7] == [
        "refresh_gamedata",
        "refresh_gamedata", "rag.refresh(gamedata)", "should_learn", "run_learning", "rag.refresh(learning)",
        "refresh_gamedata",
    ]


@pytest.mark.parametrize("auto_update,hours", [("0", 24.0), ("1", 0.0)])
def test_periodic_check_can_be_switched_off(fake_modules, monkeypatch, auto_update, hours):
    monkeypatch.setenv("GAMEDATA_AUTO_UPDATE", auto_update)
    monkeypatch.setattr(config, "GAMEDATA_RECHECK_HOURS", hours)
    asyncio.run(asyncio.wait_for(main.run_periodic_maintenance(), timeout=2))   # returns at once
    assert fake_modules.calls == []


def test_periodic_check_waits_for_the_startup_pass(fake_modules, monkeypatch):
    """A startup learning run that is still going is not joined by a second game-data step."""
    monkeypatch.setenv("GAMEDATA_AUTO_UPDATE", "1")
    monkeypatch.setattr(config, "GAMEDATA_RECHECK_HOURS", 0.01 / 3600)

    async def scenario():
        startup = asyncio.create_task(asyncio.sleep(0.15))
        periodic = asyncio.create_task(main.run_periodic_maintenance(startup))
        await asyncio.sleep(0.1)
        assert fake_modules.calls == []            # ten rounds went by, all skipped
        await startup
        for _ in range(200):
            if fake_modules.calls:
                break
            await asyncio.sleep(0.01)
        periodic.cancel()
        with pytest.raises(asyncio.CancelledError):
            await periodic

    asyncio.run(scenario())
    assert fake_modules.calls[0] == "refresh_gamedata"


def test_the_real_data_layer_offers_what_the_server_calls():
    """The guarded imports above hide a renamed function: this fails instead."""
    from backend.app.ai import data_manager, rag

    assert callable(data_manager.refresh_gamedata_if_outdated)
    assert callable(data_manager.patch_label_for_time) and callable(data_manager.get_patch_info)
    assert callable(data_manager.get_meta)
    for name in ("status", "warm_up", "refresh", "shutdown"):
        assert callable(getattr(rag, name))
    assert main.rag_package() is rag


def test_blocking_maintenance_runs_in_daemon_threads(fake_modules, monkeypatch):
    """Not in the default executor: its workers are joined at exit and shared with requests."""
    seen: dict[str, threading.Thread] = {}

    def run_learning():
        seen["thread"] = threading.current_thread()
        return {"total_chunks_written": 0, "patch": "7.40"}

    fake_modules.pipeline.run_learning = run_learning

    async def scenario():
        await main._run_learning_if_due()
        return asyncio.get_running_loop()._default_executor

    assert asyncio.run(scenario()) is None        # the loop's worker pool was never touched
    assert seen["thread"].daemon is True
    assert seen["thread"].name.startswith("maintenance-")

    async def result_and_error():
        assert await main._run_in_daemon_thread(lambda a, b=0: a + b, 2, b=3) == 5
        with pytest.raises(KeyError):
            await main._run_in_daemon_thread({}.__getitem__, "missing")

    asyncio.run(result_and_error())


def test_shutdown_does_not_wait_for_a_running_learning_job(fake_modules, monkeypatch):
    """systemctl restart / DEV reload must not hang until a learning run is done."""
    monkeypatch.setenv("GAMEDATA_AUTO_UPDATE", "0")
    monkeypatch.setenv("AI_COACH_AUTO_LEARN", "1")
    started, release = threading.Event(), threading.Event()
    job: dict[str, threading.Thread] = {}

    def run_learning():
        job["thread"] = threading.current_thread()
        started.set()
        release.wait(30)
        return {"total_chunks_written": 1, "patch": "7.40"}

    stop_requests: list[bool] = []
    fake_modules.pipeline.run_learning = run_learning
    fake_modules.pipeline.request_stop = lambda: stop_requests.append(True)
    t0 = time.monotonic()
    try:
        with TestClient(main.app) as client:
            assert started.wait(5)
            assert client.get("/api/health").status_code == 200   # requests are served meanwhile
            assert stop_requests == []
            task = main.app.state.startup_task
        # Lifespan shutdown and the event loop's executor shutdown are over ...
        assert time.monotonic() - t0 < 5
        assert task.cancelled()
        # ... while the job is still running in a thread the interpreter will not join;
        # it was asked to stop at its next checkpoint.
        assert job["thread"].is_alive() and job["thread"].daemon
        assert stop_requests == [True]
    finally:
        release.set()
    job["thread"].join(5)
    assert not job["thread"].is_alive()   # finishing after the loop closed is harmless


def test_lifespan_warms_the_retrieval_index_up_without_waiting_for_it(fake_modules, monkeypatch):
    """A vector build can take a while: neither startup nor requests nor shutdown wait for it."""
    monkeypatch.setenv("RAG_WARM_UP", "1")
    started, release, stopped = threading.Event(), threading.Event(), threading.Event()
    job: dict[str, threading.Thread] = {}

    def warm_up():
        job["thread"] = threading.current_thread()
        started.set()
        release.wait(30)

    fake_modules.rag.warm_up = warm_up
    fake_modules.rag.shutdown = stopped.set
    t0 = time.monotonic()
    try:
        with TestClient(main.app) as client:
            assert started.wait(5)
            assert client.get("/api/health").status_code == 200   # requests are served meanwhile
            assert not stopped.is_set()
        assert time.monotonic() - t0 < 5
        assert stopped.wait(5)                                    # the build was asked to stop
        assert job["thread"].is_alive() and job["thread"].daemon
    finally:
        release.set()
    job["thread"].join(5)
    assert not job["thread"].is_alive()


@pytest.mark.parametrize("state,level,line", [
    ({"mode": "hybrid", "embedder": "bge-small-en-v1.5-int8", "vectors": 987, "static_coverage": 1.0, "error": None},
     "INFO", "Knowledge retrieval: hybrid ranking, 987 vectors (model bge-small-en-v1.5-int8), "
             "static coverage 1.0, error: none"),
    # e.g. an unreadable vector file: the coach quietly ranks lexically, so the log has to say it
    ({"mode": "lexical", "embedder": None, "vectors": 0, "static_coverage": 0.0, "error": "BadZipFile: not a zip"},
     "WARNING", "Knowledge retrieval: lexical ranking, 0 vectors (model none), "
                "static coverage 0.0, error: BadZipFile: not a zip"),
])
def test_startup_log_names_the_retrieval_mode(fake_modules, monkeypatch, caplog, state, level, line):
    """Otherwise only /api/admin/status tells whether the coach's knowledge is ranked with vectors."""
    monkeypatch.setenv("RAG_WARM_UP", "1")
    logged = threading.Event()

    def status():
        fake_modules.calls.append("rag.status")
        return dict(state)

    fake_modules.rag.status = status
    real_log = main.logger.log
    monkeypatch.setattr(main.logger, "log", lambda *a, **k: (real_log(*a, **k), logged.set()))
    with caplog.at_level("INFO", logger="backend.app.main"):
        with TestClient(main.app):
            assert logged.wait(5)
    assert fake_modules.calls[:2] == ["rag.warm_up", "rag.status"]     # the state after the warm-up
    records = [(r.levelname, r.getMessage()) for r in caplog.records if "Knowledge retrieval" in r.getMessage()]
    assert records == [(level, line)]

    # No retrieval package, or a status that fails: nothing to report, nothing raised.
    caplog.clear()
    fake_modules.rag.status = lambda: 1 / 0
    with caplog.at_level("INFO", logger="backend.app.main"):
        main._warm_up_retrieval()
    assert [r.getMessage() for r in caplog.records] == ["Knowledge retrieval status failed: division by zero"]


def test_lifespan_leaves_the_retrieval_index_alone_when_the_warm_up_is_off(fake_modules, monkeypatch):
    """RAG_WARM_UP=0 (the test suite's default): no build is started, so none is stopped either."""
    monkeypatch.setenv("RAG_WARM_UP", "0")
    with TestClient(main.app) as client:
        assert client.get("/api/health").status_code == 200
    time.sleep(0.05)
    assert fake_modules.calls == []


def test_shutdown_kills_running_replay_parsers(monkeypatch, caplog):
    """Analyzer JVMs lead their own process group: the server's stop signal does not reach them."""
    killed = []

    async def no_maintenance():
        return None

    monkeypatch.setattr(main, "run_startup_maintenance", no_maintenance)
    monkeypatch.setattr(main.parser, "kill_running_parsers", lambda: killed.append(True) or 2)
    with caplog.at_level("INFO", logger="backend.app.main"):
        with TestClient(main.app) as client:
            assert client.get("/api/health").status_code == 200
            assert killed == []
    assert killed == [True]
    assert "killed 2 running replay parser(s)" in caplog.text


def test_learning_run_refused_by_the_lock_is_not_logged_as_finished(fake_modules, caplog):
    """A CLI / cron run holds the learning lock: run_learning returns error already_running."""
    fake_modules.pipeline.run_learning = lambda: {
        "success": False, "error": "already_running", "levels": {}, "total_chunks_written": 0, "patch": "7.40"}
    with caplog.at_level("INFO", logger="backend.app.main"):
        asyncio.run(main._run_learning_if_due())
    assert "did not start: already_running" in caplog.text
    assert "Learning run finished" not in caplog.text


def test_lifespan_installs_the_configured_worker_pool(monkeypatch):
    monkeypatch.setattr(config, "THREADPOOL_SIZE", 7)
    seen = {}

    async def maintenance():
        executor = asyncio.get_running_loop()._default_executor
        seen["workers"] = executor._max_workers
        seen["thread"] = await asyncio.to_thread(lambda: threading.current_thread().name)

    monkeypatch.setattr(main, "run_startup_maintenance", maintenance)
    with TestClient(main.app):
        pass
    assert seen["workers"] == 7
    assert seen["thread"].startswith("worker")


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


def test_static_files_do_not_leave_the_frontend_directory():
    """.env is one directory above frontend/. A backslash is a path separator on Windows only,
    so that form of "..\\" is tried as well; .env.example stands in for .env."""
    assert (config.BASE_DIR / ".env.example").is_file() and (config.BASE_DIR / "backend" / "requirements.txt").is_file()
    with TestClient(main.app) as client:
        for url in ("/..%2F.env.example", "/..%5C.env.example", "/static/..%5C..%5C.env.example",
                    "/static/..%2F..%2F.env.example", "/static%5C..%5C..%5Cbackend%5Crequirements.txt",
                    "/..%5Cbackend%5Crequirements.txt", "/static/js/..%5C..%5C..%5C.env.example"):
            response = client.get(url)
            assert response.status_code == 404, url
            assert "DEEPSEEK_API_KEY" not in response.text and "fastapi" not in response.text


def test_page_scripts_and_styles_are_served_with_types_a_browser_runs():
    """With nosniff a browser runs a script only when its content type says JavaScript."""
    with TestClient(main.app) as client:
        script = ("text/javascript", "application/javascript")
        for url, expected in (("/static/js/app.js", script),
                              ("/static/vendor/vue-3.4.38.global.prod.js", script),
                              ("/static/css/app.css", ("text/css",)),
                              ("/static/data/heroes.json", ("application/json",)),
                              ("/static/assets/favicon.svg", ("image/svg+xml",))):
            response = client.get(url)
            assert response.status_code == 200, url
            assert response.headers["content-type"].split(";")[0] in expected, url
            assert response.headers["x-content-type-options"] == "nosniff"


def test_static_content_types_do_not_follow_the_windows_registry(monkeypatch):
    """mimetypes reads the registry on Windows, where .js is text/plain on some machines."""
    import mimetypes

    before = {suffix: mimetypes.guess_type("x" + suffix)[0] for _type, suffix in main._STATIC_MIME_TYPES}
    try:
        for _type, suffix in main._STATIC_MIME_TYPES:
            mimetypes.add_type("text/plain", suffix)            # what such a registry does
        monkeypatch.setattr(osutil, "IS_WINDOWS", False)
        main.register_static_mime_types()                       # Linux / macOS: Python's table is left alone
        assert mimetypes.guess_type("app.js")[0] == "text/plain"
        monkeypatch.setattr(osutil, "IS_WINDOWS", True)
        main.register_static_mime_types()
        assert {suffix: mimetypes.guess_type("x" + suffix)[0] for _type, suffix in main._STATIC_MIME_TYPES} == {
            ".js": "text/javascript", ".css": "text/css", ".json": "application/json",
            ".svg": "image/svg+xml", ".html": "text/html"}
    finally:
        for suffix, content_type in before.items():
            if content_type:
                mimetypes.add_type(content_type, suffix)


def test_configure_logging_sets_backend_level(monkeypatch):
    import logging

    monkeypatch.setattr(config, "LOG_LEVEL", "DEBUG")
    main.configure_logging()
    assert logging.getLogger("backend").level == logging.DEBUG
    monkeypatch.setattr(config, "LOG_LEVEL", "nonsense")
    main.configure_logging()
    assert logging.getLogger("backend").level == logging.INFO


def test_configure_logging_lets_no_console_code_page_stop_the_server(monkeypatch):
    """Output redirected to a file on a Japanese Windows is cp932: a Chinese player name in a
    log line or a print() must come out escaped, not as a UnicodeEncodeError."""
    import logging

    out = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict", newline="\n")
    err = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="backslashreplace", newline="\n")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    main.configure_logging()
    for stream in (out, err):
        handler = logging.StreamHandler(stream)
        handler.emit(logging.LogRecord("backend.test", logging.INFO, __file__, 1,
                                       "Parsed the replay of %s", ("张三",), None))
        stream.flush()
        # 三 exists in cp932, 张 does not
        assert stream.buffer.getvalue() == b"Parsed the replay of \\u5f20\x8eO\n"
    assert out.encoding == "cp932"


# ---------------------------------------------------------------------------
# config: what Windows does differently (java.exe, .env files written by its tools)
# ---------------------------------------------------------------------------

def _fake_java(home, name):
    path = home / "bin" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("exe", ["java", "java.exe"])
def test_java_is_looked_up_under_the_name_the_system_gives_it(tmp_path, monkeypatch, exe):
    """bin/java on Linux and macOS, bin\\java.exe on Windows: in tools/jdk17, then in JAVA_HOME, then on PATH."""
    other = "java.exe" if exe == "java" else "java"
    local, home, elsewhere = tmp_path / "tools" / "jdk17", tmp_path / "jdk-home", tmp_path / "elsewhere"
    monkeypatch.setattr(config, "_JAVA_EXE", exe)
    monkeypatch.setattr(config, "_LOCAL_JDK_CANDIDATES",
                        (local / "Contents" / "Home" / "bin" / exe, local / "bin" / exe))
    monkeypatch.setattr(config.shutil, "which", lambda name: str(elsewhere / name))
    monkeypatch.delenv("JAVA_PATH", raising=False)
    monkeypatch.setenv("JAVA_HOME", str(home))

    assert config._resolve_java_path() == str(elsewhere / "java")              # nothing local, nothing in JAVA_HOME
    _fake_java(home, other)                                                    # the other system's launcher: not ours
    assert config._resolve_java_path() == str(elsewhere / "java")
    in_home = _fake_java(home, exe)
    assert config._resolve_java_path() == str(in_home)
    in_tools = _fake_java(local, exe)                                          # the project's own JDK wins
    assert config._resolve_java_path() == str(in_tools)
    monkeypatch.setenv("JAVA_PATH", "java17")                                  # an explicit command name wins over all
    assert config._resolve_java_path() == "java17"
    monkeypatch.setenv("JAVA_PATH", str(in_home))
    assert config._resolve_java_path() == str(in_home)
    monkeypatch.delenv("JAVA_PATH")
    monkeypatch.delenv("JAVA_HOME")
    in_tools.unlink()
    monkeypatch.setattr(config.shutil, "which", lambda name: None)             # no Java at all: the bare command,
    assert config._resolve_java_path() == "java"                               # the parser reports it as missing


def test_the_java_launcher_name_follows_the_system():
    assert config._JAVA_EXE == ("java.exe" if osutil.IS_WINDOWS else "java")
    assert all(path.name == config._JAVA_EXE for path in config._LOCAL_JDK_CANDIDATES)
    assert config._LOCAL_JDK_CANDIDATES[-1] == config.BASE_DIR / "tools" / "jdk17" / "bin" / config._JAVA_EXE


@pytest.mark.parametrize("encoding,expected", [
    ("utf-8", "utf-8-sig"),           # as .env.example is, and what editors write
    ("utf-8-sig", "utf-8-sig"),       # Notepad's "UTF-8 with BOM"
    ("utf-16", "utf-16"),             # Windows PowerShell 5.1: echo ... > .env, Out-File, Set-Content -Encoding Unicode
    ("utf-16-be", "utf-8-sig"),       # no mark: not guessed
])
def test_env_file_written_by_windows_tools_is_read(tmp_path, monkeypatch, encoding, expected):
    """With a byte order mark python-dotenv's default reading loses the first variable of the
    file (the mark becomes part of its name) or cannot decode the file at all."""
    path = tmp_path / ".env"
    path.write_text("PROBE_ENV_FIRST=one\r\nPROBE_ENV_NAME=张三\r\nPROBE_ENV_KEPT=from-file\r\n",
                    encoding=encoding, newline="")
    assert config.dotenv_encoding(path) == expected
    if encoding == "utf-16-be":
        return
    for name in ("PROBE_ENV_FIRST", "PROBE_ENV_NAME"):
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)                    # absent now, and put back that way after the test
    monkeypatch.setenv("PROBE_ENV_KEPT", "from-environment")
    assert config.load_env_file(path) is True
    assert os.environ["PROBE_ENV_FIRST"] == "one"
    assert os.environ["PROBE_ENV_NAME"] == "张三"
    assert os.environ["PROBE_ENV_KEPT"] == "from-environment"       # the environment wins


def test_missing_env_file_is_not_an_error(tmp_path):
    assert config.dotenv_encoding(tmp_path / "no-such.env") == "utf-8-sig"
    assert config.load_env_file(tmp_path / "no-such.env") is False


_QUOTED_JAVA = r'"C:\Program Files\Eclipse Adoptium\jdk-17\bin\java.exe"'
_DAMAGED_JAVA = "C:\\Program Files\\Eclipse Adoptium\\jdk-17\x08in\\java.exe"       # \b: a backspace
_INTENDED_JAVA = r"C:\Program Files\Eclipse Adoptium\jdk-17\bin\java.exe"


def test_double_quoted_windows_path_in_the_env_file_is_repaired(tmp_path, monkeypatch, caplog):
    """python-dotenv reads \\b \\t \\n \\a \\f \\r \\v inside double quotes as control characters,
    and every JDK path has a \\bin in it: Java was "not found" at a path that looked right."""
    path = tmp_path / ".env"
    path.write_text(
        f"JAVA_PATH={_QUOTED_JAVA}\n"
        'RAG_MODEL_DIR="D:\\dota\\tools\\new\\app\\venv\\frontend\\replays"\n'      # \t \n \a \v \f \r
        "CLARITY_JAR_PATH='C:\\tools\\clarity-analyzer.jar'\n"                      # single quotes: as written
        "RAG_INDEX_DIR=C:\\cache\\rag\n"                                            # no quotes: as written
        'PROBE_ENV_NOT_A_PATH="a\\tb"\n',
        encoding="utf-8")
    names = ("JAVA_PATH", "RAG_MODEL_DIR", "CLARITY_JAR_PATH", "RAG_INDEX_DIR", "PROBE_ENV_NOT_A_PATH",
             "JAVA_HOME", "BILIBILI_COOKIES_FILE")
    for name in names:
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    assert config.load_env_file(path) is True
    assert os.environ["JAVA_PATH"] == _DAMAGED_JAVA

    monkeypatch.setattr(osutil, "IS_WINDOWS", False)            # a control character can be part of a name there
    with caplog.at_level("WARNING", logger="backend.app.config"):
        assert config.repair_path_settings() == ["JAVA_PATH", "RAG_MODEL_DIR"]
    assert os.environ["JAVA_PATH"] == _DAMAGED_JAVA
    assert "JAVA_PATH" in caplog.text and "double quotes" in caplog.text

    caplog.clear()
    monkeypatch.setattr(osutil, "IS_WINDOWS", True)
    with caplog.at_level("WARNING", logger="backend.app.config"):
        assert config.repair_path_settings() == ["JAVA_PATH", "RAG_MODEL_DIR"]
    assert os.environ["JAVA_PATH"] == _INTENDED_JAVA
    assert os.environ["RAG_MODEL_DIR"] == r"D:\dota\tools\new\app\venv\frontend\replays"
    assert os.environ["CLARITY_JAR_PATH"] == r"C:\tools\clarity-analyzer.jar"
    assert os.environ["RAG_INDEX_DIR"] == r"C:\cache\rag"
    assert os.environ["PROBE_ENV_NOT_A_PATH"] == "a\tb"         # only settings that name a path
    assert _INTENDED_JAVA in caplog.text and "double quotes" in caplog.text
    if os.name == "nt":                                         # an absolute path only there:
        assert config._resolve_java_path() == _INTENDED_JAVA    # what the parser is started with
    assert config.repair_path_settings() == []                  # nothing left to repair, nothing said twice


_ENV_FILE_PROBE = "import os; from backend.app import config; print(os.environ.get('PROBE_ENV_FROM_FILE', 'not read'))"


def test_env_file_is_skipped_on_request_also_by_a_child_process(tmp_path):
    """The suite must not see a developer's .env. An empty variable keeps load_dotenv from
    filling it in, but on Windows a variable set to "" does not exist for a child process:
    conftest.py sets a switch with a value instead, which every process a test starts inherits."""
    assert os.environ["DOTA_ANALYZER_SKIP_DOTENV"] == "1"       # conftest.py, before any backend import
    assert os.environ["DEV"] == "0" and config.DEV is False     # ... which also pins the mode

    # A project of its own (config.py finds .env next to the backend directory it is in).
    app = tmp_path / "backend" / "app"
    app.mkdir(parents=True)
    for name in ("__init__.py", "config.py", "osutil.py"):
        (app / name).write_bytes((config.BASE_DIR / "backend" / "app" / name).read_bytes())
    (tmp_path / ".env").write_text("PROBE_ENV_FROM_FILE=read\n", encoding="utf-8")

    def probe(**changes):
        env = {name: value for name, value in os.environ.items()
               if name not in ("DOTA_ANALYZER_SKIP_DOTENV", "PROBE_ENV_FROM_FILE")}
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
        done = subprocess.run([sys.executable, "-c", _ENV_FILE_PROBE], capture_output=True, encoding="utf-8",
                              errors="replace", cwd=tmp_path, env={**env, **changes}, timeout=120)
        assert done.returncode == 0, done.stderr
        return done.stdout.strip()

    assert probe() == "read"
    assert probe(DOTA_ANALYZER_SKIP_DOTENV="1") == "not read"
    assert probe(DOTA_ANALYZER_SKIP_DOTENV="0") == "read"


# A setting is read through os.environ / os.getenv or through a helper whose
# name contains "env" (config.env_int, rag.runtime._env, evaluator._env_tokens,
# whatever is added next) with the variable's name as its first argument.
_SETTING_READS = re.compile(
    r"""(?:environ\.get\(|environ\[|\b\w*env\w*\()\s*["']([A-Z][A-Z0-9_]{2,})["']""")


def test_every_setting_the_backend_reads_is_documented():
    """An environment variable without a line in .env.example fails here."""
    assert _SETTING_READS.findall(
        'os.environ.get("A_B") os.environ["C_D"] os.getenv("E_F") env_int("G_H", 1) _env("I_J") '
        '_env_tokens("K_L", 2) self._read_env_flag( "M_N" ) print("NOT_ONE") env("x")'
    ) == ["A_B", "C_D", "E_F", "G_H", "I_J", "K_L", "M_N"]

    backend = config.BASE_DIR / "backend"
    read: dict[str, str] = {}
    for path in sorted([*(backend / "app").rglob("*.py"), *(backend / "scripts").rglob("*.py")]):
        for name in _SETTING_READS.findall(path.read_text(encoding="utf-8")):
            read.setdefault(name, path.name)
    assert {"DEEPSEEK_API_KEY", "MAX_UPLOAD_MB", "GAMEDATA_RECHECK_HOURS", "RAG_WARM_UP"} <= set(read)

    example = (config.BASE_DIR / ".env.example").read_text(encoding="utf-8")
    documented = set(re.findall(r"^#? ?([A-Z][A-Z0-9_]+)=", example, flags=re.MULTILINE))
    standard = {
        "JAVA_HOME",                    # the JDK's own variable, only a fallback for JAVA_PATH
        "DOTA_ANALYZER_SKIP_DOTENV",    # decides whether .env is read: it cannot be set in that file
    }
    missing = {name: source for name, source in read.items() if name not in documented | standard}
    assert not missing, f"document these settings in .env.example: {missing}"


@pytest.mark.parametrize("raw,hours,warning", [
    ("", 24.0, None),
    ("6", 6.0, None),
    ("0", 0.0, None),                 # off
    ("-1", 0.0, None),
    ("0.25", 0.25, None),
    ("0.001", 0.25, "below the minimum of 0.25"),      # a typo must not become a request loop
    ("often", 24.0, "Invalid number"),
])
def test_recheck_interval_has_a_floor(monkeypatch, caplog, raw, hours, warning):
    """Each round asks GitHub's API, which allows 60 requests per hour without a token."""
    monkeypatch.setenv("GAMEDATA_RECHECK_HOURS", raw)
    with caplog.at_level("WARNING", logger="backend.app.config"):
        assert config._recheck_hours() == hours
    assert [r.getMessage() for r in caplog.records if warning is None or warning not in r.getMessage()] == []
    assert bool(caplog.records) is (warning is not None)
