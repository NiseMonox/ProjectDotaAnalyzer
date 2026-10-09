"""Dota 2 Match Analyzer - FastAPI Application.

Run with ``./run.sh`` or ``python -m backend.app.main`` (honours HOST,
PORT and DEV=1 for auto-reload). Run a single worker: rate limits, caches
and in-flight de-duplication are per process.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipMiddleware, GZipResponder
from starlette.types import Message, Receive, Scope, Send

from . import config  # loads .env before any setting is read
from .routes import router
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

class _SSEAwareGZipResponder(GZipResponder):
    """Starlette 0.38's streaming gzip buffers small chunks inside zlib,
    which would hold SSE events back; pass event streams through as-is."""

    async def send_with_gzip(self, message: Message) -> None:
        await super().send_with_gzip(message)
        if message["type"] == "http.response.start":
            content_type = Headers(raw=message.get("headers", [])).get("content-type", "")
            if content_type.startswith("text/event-stream"):
                # The parent forwards body chunks untouched when it believes
                # the response is already encoded.
                self.content_encoding_set = True


class SSEAwareGZipMiddleware(GZipMiddleware):
    """GZip for JSON/static responses, never for text/event-stream."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = Headers(scope=scope)
            wants_gzip = "gzip" in headers.get("accept-encoding", "")
            is_event_stream = "text/event-stream" in headers.get("accept", "")
            if wants_gzip and not is_event_stream:
                responder = _SSEAwareGZipResponder(self.app, self.minimum_size, compresslevel=self.compresslevel)
                await responder(scope, receive, send)
                return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Startup maintenance (C6): game data first, then self-learning
# ---------------------------------------------------------------------------

async def _check_and_update_gamedata() -> None:
    """Refresh game data when missing, on a new patch, or when stale.

    Awaits the update itself so the learning run that follows sees the
    fresh data. Network failures only log a warning (local or hardcoded
    data keeps working).
    """
    try:
        from .ai import data_manager
    except ImportError:
        logger.debug("data_manager not available, using hardcoded game data")
        return

    try:
        meta = await asyncio.to_thread(data_manager.get_meta)
        local_ver = meta.get("patch_version", "?")
        if meta.get("status") == "no_data":
            reason = "no local data"
        else:
            age_days = float(meta.get("age_days") or 0)
            is_stale = age_days > config.GAMEDATA_STALE_DAYS
            try:
                remote_ver, needs_update = await asyncio.to_thread(data_manager.check_patch_version)
            except Exception as exc:
                logger.warning("Patch version check failed (network?): %s. Using local data (patch %s).",
                               exc, local_ver)
                if not is_stale:
                    return
                reason = f"local data is {age_days:.1f} days old"
            else:
                if needs_update:
                    reason = f"new patch {local_ver} -> {remote_ver}"
                elif is_stale:
                    reason = f"local data is {age_days:.1f} days old"
                else:
                    logger.info(
                        "Game data up to date: patch %s, %d heroes, fetched %s",
                        local_ver, meta.get("hero_count", 0), meta.get("last_fetched", "?"),
                    )
                    return

        logger.info("Updating game data in background (%s)...", reason)
        result = await asyncio.to_thread(data_manager.fetch_and_save_gamedata, force=True)
        logger.info(
            "Game data updated: patch %s, %d heroes, %d items",
            result.get("patch_version", "?"), result.get("heroes", 0), result.get("items", 0),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Game data update failed: %s. Hardcoded fallback still available.", exc)


async def _run_learning_if_due() -> None:
    """Run the AI coach self-learning pipeline when it is due."""
    try:
        from .ai.learning.pipeline import run_learning, should_learn
    except ImportError:
        logger.debug("learning pipeline not available")
        return
    try:
        if not await asyncio.to_thread(should_learn):
            logger.info("Learned knowledge is fresh, skipping learning run")
            return
        logger.info("Learning run due, starting in background...")
        summary = await asyncio.to_thread(run_learning)
        logger.info(
            "Learning run finished: %d chunks written (patch %s)",
            summary.get("total_chunks_written", 0), summary.get("patch") or "unknown",
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Background learning run failed: %s", exc)


async def run_startup_maintenance() -> None:
    """Background startup work; never blocks serving requests.

    GAMEDATA_AUTO_UPDATE=0 skips the game-data step, AI_COACH_AUTO_LEARN=0
    skips learning. Learning only starts after the game-data step finished.
    """
    if config.env_bool("GAMEDATA_AUTO_UPDATE", True):
        await _check_and_update_gamedata()
    else:
        logger.info("Game data auto-update disabled (GAMEDATA_AUTO_UPDATE=0)")

    if config.env_bool("AI_COACH_AUTO_LEARN", True):
        await _run_learning_if_due()
    else:
        logger.info("AI coach auto-learning disabled (AI_COACH_AUTO_LEARN=0)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background maintenance; cancel it on shutdown."""
    task = asyncio.create_task(run_startup_maintenance(), name="startup-maintenance")
    app.state.startup_task = task
    try:
        yield
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Dota 2 Match Analyzer",
    description="Analyze Dota 2 replay files for item builds, skill builds, "
                "ward placements, smoke usage, teamfights, and ability usage.",
    version="0.2.0",
    lifespan=lifespan,
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
