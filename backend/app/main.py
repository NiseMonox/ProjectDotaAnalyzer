"""Dota 2 Match Analyzer - FastAPI Application.

Run with ``./run.sh`` or ``python -m backend.app.main`` (honours HOST,
PORT and DEV=1 for auto-reload). Run a single worker: rate limits, caches
and in-flight de-duplication are per process.
"""
from __future__ import annotations

import asyncio
import calendar
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipMiddleware
from starlette.types import Receive, Scope, Send

from . import config  # loads .env before any setting is read
from . import parser
from .routes import rag_package, router
from .security import BodySizeLimitMiddleware, SecurityHeadersMiddleware, install_exception_handlers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """Make backend INFO logs visible under uvicorn (LOG_LEVEL env).

    uvicorn only configures its own loggers, so without a root handler the
    app's INFO messages are dropped. Third-party loggers stay at WARNING.
    """
    level = logging.getLevelName(config.LOG_LEVEL)
    if not isinstance(level, int):
        level = logging.INFO
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    logging.getLogger("backend").setLevel(level)


configure_logging()


# ---------------------------------------------------------------------------
# GZip that never buffers Server-Sent Events
# ---------------------------------------------------------------------------

class SSEAwareGZipMiddleware(GZipMiddleware):
    """GZip for JSON/static responses, never for text/event-stream.

    A gzip stream holds small chunks back inside zlib, which would delay
    SSE events. Starlette >= 0.46 passes ``text/event-stream`` responses
    through untouched (each chunk is forwarded as it is produced); requests
    that announce an EventSource, or do not accept gzip at all, skip the
    gzip layer altogether. ``test_sse_chunks_are_forwarded_unbuffered``
    guards this on upgrades.
    """

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = Headers(scope=scope)
            wants_gzip = "gzip" in headers.get("accept-encoding", "")
            is_event_stream = "text/event-stream" in headers.get("accept", "")
            if not wants_gzip or is_event_stream:
                await self.app(scope, receive, send)
                return
        await super().__call__(scope, receive, send)


# ---------------------------------------------------------------------------
# Background maintenance (C6): game data first, then self-learning, at
# startup; the game-version check is repeated while the server runs. The
# knowledge-retrieval index (ai.rag) is warmed up next to it and told about
# every change of the data it is built from.
# ---------------------------------------------------------------------------

async def _run_in_daemon_thread(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a blocking call in its own daemon thread and await the result.

    Maintenance jobs can run for minutes (a learning run downloads and
    distils data). In the default executor they would hold a worker that
    request handling needs, and its non-daemon thread is joined when the
    process exits, so a restart or a DEV reload would hang until the job is
    done. A daemon thread does neither; cancelling the awaiting task simply
    abandons it (game data and learned files are written atomically).
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    def deliver(result: Any, error: BaseException | None) -> None:
        if future.done():  # the awaiting task was cancelled
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(result)

    def runner() -> None:
        try:
            outcome = (func(*args, **kwargs), None)
        except BaseException as exc:  # delivered to the awaiting task
            outcome = (None, exc)
        with suppress(RuntimeError):  # event loop already closed (shutdown)
            loop.call_soon_threadsafe(deliver, *outcome)

    threading.Thread(target=runner, name=f"maintenance-{getattr(func, '__name__', 'job')}", daemon=True).start()
    return await future


def _rag_call(action: str, *args: Any) -> Any:
    """Call ``ai.rag.<action>(*args)`` when the retrieval package is there.

    Blocking (``warm_up`` builds the knowledge stores and missing vectors
    in the calling thread), so it runs in a daemon thread. Never raises:
    without the package, or when it fails, retrieval stays lexical and
    None is returned.
    """
    func = getattr(rag_package(), action, None)  # None also without the package
    if func is None:
        return None
    try:
        return func(*args)
    except Exception as exc:
        logger.warning("Knowledge retrieval %s failed: %s", action, exc)
        return None


def _warm_up_retrieval() -> None:
    """Warm the retrieval index up, then log the mode the server ended up in.

    Without that line a vector file that is missing or unreadable would
    only show in ``/api/admin/status``: the coach silently ranks its
    knowledge lexically then. Blocking, for a daemon thread.
    """
    _rag_call("warm_up")
    state = _rag_call("status")
    if not isinstance(state, dict):
        return
    logger.log(
        logging.WARNING if state.get("error") else logging.INFO,
        "Knowledge retrieval: %s ranking, %s vectors (model %s), static coverage %s, error: %s",
        state.get("mode"), state.get("vectors"), state.get("embedder") or "none",
        state.get("static_coverage"), state.get("error") or "none",
    )


def _preload_llm_sdk() -> None:
    """Import the LLM SDK ahead of the first report (blocking, never raises).

    Without it the first coach request pays for the SDK's lazy imports;
    the evaluator keeps them off the event loop either way.
    """
    try:
        from .ai import evaluator

        if evaluator.is_api_available():
            evaluator.preload_sdk()
    except Exception as exc:
        logger.warning("LLM SDK preload failed: %s", exc)


async def _check_and_update_gamedata() -> bool:
    """Refresh the game data when it is missing or outdated; True when it changed.

    The data layer decides (``data_manager.refresh_gamedata_if_outdated``:
    a new game version or source revision, files in an older format, data
    older than GAMEDATA_STALE_DAYS) and downloads only then. Awaits the
    update itself so the learning run that follows sees the fresh data.
    Network failures only log a warning (local or hardcoded data keeps
    working).
    """
    try:
        from .ai.data_manager import refresh_gamedata_if_outdated
    except ImportError:
        logger.debug("data_manager not available, using hardcoded game data")
        return False

    try:
        result = await _run_in_daemon_thread(refresh_gamedata_if_outdated)
        status = result.get("status")
        if status == "failed":
            logger.warning(
                "Game data update failed: %s. Using local data (patch %s).",
                result.get("error") or "unknown error", result.get("patch_version", "?"),
            )
            return False
        if status != "updated":
            logger.info("Game data up to date: version %s",
                        result.get("patch_label") or result.get("patch_version", "?"))
            return False
        logger.info(
            "Game data updated (%s): version %s, %d heroes, %d items, source %s",
            result.get("reason") or "outdated", result.get("patch_label") or result.get("patch_version", "?"),
            result.get("heroes", 0), result.get("items", 0), result.get("source", "?"),
        )
        if not result.get("changed", True):  # the same data again: nothing downstream is stale
            return False
        await _run_in_daemon_thread(_rag_call, "refresh", "gamedata")
        return True
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Game data update failed: %s. Hardcoded fallback still available.", exc)
        return False


async def _run_learning_if_due() -> None:
    """Run the AI coach self-learning pipeline when it is due.

    When the server shuts down during a run, the pipeline is asked to stop
    at its next checkpoint (it records the unfinished levels as due again).
    """
    try:
        from .ai.learning import pipeline
    except ImportError:
        logger.debug("learning pipeline not available")
        return
    running = False
    try:
        if not await _run_in_daemon_thread(pipeline.should_learn):
            logger.info("Learned knowledge is fresh, skipping learning run")
            return
        logger.info("Learning run due, starting in background...")
        running = True
        summary = await _run_in_daemon_thread(pipeline.run_learning)
        running = False
        if summary.get("error"):
            # e.g. "already_running": a CLI / cron run holds the learning lock
            logger.info("Learning run did not start: %s", summary["error"])
            return
        logger.info(
            "Learning run finished: %d chunks written (patch %s)",
            summary.get("total_chunks_written", 0), summary.get("patch") or "unknown",
        )
        if summary.get("total_chunks_written"):
            await _run_in_daemon_thread(_rag_call, "refresh", "learning")
    except asyncio.CancelledError:
        request_stop = getattr(pipeline, "request_stop", None)
        if running and request_stop is not None:
            logger.info("Shutting down: asking the running learning job to stop")
            request_stop()
        raise
    except Exception as exc:
        logger.warning("Background learning run failed: %s", exc)


def _minutes_since_gamedata_check() -> float | None:
    """Minutes since the game version was last checked, when that check still stands.

    None when the start-up check has to run: no usable game data on disk
    (missing, written in an older format, older than GAMEDATA_STALE_DAYS,
    known to be behind the current game version), no check on record, or one
    older than ``config.GAMEDATA_RECHECK_MIN_HOURS``. Reads local files only.
    """
    try:
        from .ai.data_manager import get_meta
        meta = get_meta()
        if (meta.get("status") == "no_data" or meta.get("outdated_format") or meta.get("is_stale")
                or meta.get("data_current") is False):
            return None
        checked = calendar.timegm(time.strptime(str(meta.get("checked_at") or ""), "%Y-%m-%dT%H:%M:%SZ"))
    except Exception:  # no data layer, no check on record, an unreadable stamp
        return None
    age_s = time.time() - checked
    return age_s / 60 if 0 <= age_s < config.GAMEDATA_RECHECK_MIN_HOURS * 3600 else None


def _auto_learn_enabled() -> bool:
    """Whether this process starts learning runs by itself.

    Not with AI_COACH_AUTO_LEARN=0, and not in DEV mode: every saved file
    restarts the server there, and a restart ends the running job (a due
    level-1 run was started again with each one and spent OpenDota's daily
    request limit). ``python -m backend.scripts.learn`` runs it by hand.
    """
    if not config.env_bool("AI_COACH_AUTO_LEARN", True):
        logger.info("AI coach auto-learning disabled (AI_COACH_AUTO_LEARN=0)")
        return False
    if config.DEV:
        logger.info("AI coach auto-learning is off in DEV mode (reloads would restart it); "
                    "run `python -m backend.scripts.learn` when the knowledge should be refreshed")
        return False
    return True


async def run_startup_maintenance() -> None:
    """Background startup work; never blocks serving requests.

    GAMEDATA_AUTO_UPDATE=0 skips the game-data step, AI_COACH_AUTO_LEARN=0
    or DEV=1 skips learning. Learning only starts after the game-data step
    finished. The blocking parts run in daemon threads (see
    _run_in_daemon_thread).

    The game-version check is not repeated by a start that follows the
    previous check within ``config.GAMEDATA_RECHECK_MIN_HOURS`` (a restart,
    a crash loop, DEV reloads): its three requests include one to the
    GitHub API, which allows 60 an hour without a token.
    """
    if not config.env_bool("GAMEDATA_AUTO_UPDATE", True):
        logger.info("Game data auto-update disabled (GAMEDATA_AUTO_UPDATE=0)")
    else:
        checked = await _run_in_daemon_thread(_minutes_since_gamedata_check)
        if checked is None:
            await _check_and_update_gamedata()
        else:
            logger.info("Game data was checked %.0f min ago; not checking again at this start", checked)

    if _auto_learn_enabled():
        await _run_learning_if_due()


async def run_periodic_maintenance(startup: asyncio.Task | None = None) -> None:
    """Repeat the game-data step every GAMEDATA_RECHECK_HOURS until cancelled.

    A server that runs for weeks would otherwise keep the data of the game
    version it was started on: letter patches come every few weeks and the
    startup check only runs once. When a check brings new data, the learning
    step runs again if it is due (knowledge learned on the previous version
    is stale then). Does nothing with GAMEDATA_AUTO_UPDATE=0 or
    GAMEDATA_RECHECK_HOURS=0. A round is skipped while ``startup`` (the
    startup maintenance task) is still busy.
    """
    interval_s = config.GAMEDATA_RECHECK_HOURS * 3600
    if interval_s <= 0 or not config.env_bool("GAMEDATA_AUTO_UPDATE", True):
        return
    while True:
        await asyncio.sleep(interval_s)
        if startup is not None and not startup.done():
            continue
        if await _check_and_update_gamedata() and _auto_learn_enabled():
            await _run_learning_if_due()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Size the worker pool and start the background work.

    Four tasks: the startup maintenance, its periodic repetition, the import
    of the LLM SDK (when a key is configured) and the warm-up of the
    knowledge-retrieval index. RAG_WARM_UP=0 skips the last one:
    the first report then builds the knowledge stores, and this process
    never embeds missing texts (the index only starts builds once warmed
    up; ``python -m backend.scripts.rag build`` does it from outside).
    None of them delays startup. On shutdown they are cancelled; a job
    still running in its daemon thread is abandoned and never delays
    process exit. Replay analyzer JVMs that are still running are killed:
    each leads its own process group, so the signal that stops the server
    does not reach them.
    uvicorn runs this only after the open requests have ended or (with
    ``--timeout-graceful-shutdown``) were cancelled, so the client of an
    interrupted upload gets a closed connection or uvicorn's plain 500, not
    one of our JSON errors, and has to retry.
    """
    # asyncio.to_thread() uses the loop's default executor.
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=config.THREADPOOL_SIZE, thread_name_prefix="worker"),
    )
    task = asyncio.create_task(run_startup_maintenance(), name="startup-maintenance")
    app.state.startup_task = task
    background = [
        task,
        asyncio.create_task(run_periodic_maintenance(task), name="periodic-maintenance"),
        asyncio.create_task(_run_in_daemon_thread(_preload_llm_sdk), name="llm-sdk-preload"),
    ]
    warm_up = config.env_bool("RAG_WARM_UP", True)
    if warm_up:
        background.append(asyncio.create_task(_run_in_daemon_thread(_warm_up_retrieval), name="rag-warm-up"))
    try:
        yield
    finally:
        for job in background:
            if not job.done():
                job.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await job
        if warm_up:
            # Only a warmed-up index builds vectors. Ask it to stop at its
            # next batch, from a thread of its own: a busy index must not
            # hold the shutdown up.
            threading.Thread(target=_rag_call, args=("shutdown",), name="rag-shutdown", daemon=True).start()
        killed = parser.kill_running_parsers()
        if killed:
            logger.info("Shutting down: killed %d running replay parser(s)", killed)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

# The interactive API documentation lists every route, the admin ones
# included: served in DEV mode only (FastAPI's default is on for everyone).
_API_DOCS = {} if config.DEV else {"docs_url": None, "redoc_url": None, "openapi_url": None}

app = FastAPI(
    title="Dota 2 Match Analyzer",
    description="Analyze Dota 2 replay files for item builds, skill builds, "
                "ward placements, smoke usage, teamfights, and ability usage.",
    version="0.2.0",
    lifespan=lifespan,
    **_API_DOCS,
)

install_exception_handlers(app)

# Middleware: the last one added is the outermost.
if config.ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.ALLOWED_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept-Language", "X-Admin-Token"],
        max_age=600,
    )
app.add_middleware(SSEAwareGZipMiddleware, minimum_size=1024, compresslevel=6)
app.add_middleware(
    BodySizeLimitMiddleware,
    paths=("/api/upload",),
    max_bytes=lambda: config.MAX_UPLOAD_BYTES,
)
app.add_middleware(SecurityHeadersMiddleware)

app.include_router(router)

# Serve frontend static files
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEV,
    )
