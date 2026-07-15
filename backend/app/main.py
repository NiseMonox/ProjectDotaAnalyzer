"""Dota 2 Match Analyzer - FastAPI Application."""
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import FRONTEND_DIR
from .routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan (replaces deprecated @app.on_event('startup'))."""
    await _check_and_update_gamedata()
    await _check_and_start_learning()
    yield


app = FastAPI(
    title="Dota 2 Match Analyzer",
    description="Analyze Dota 2 replay files for item builds, skill builds, "
                "ward placements, smoke usage, teamfights, and ability usage.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


async def _check_and_update_gamedata():
    """Check game data version on startup; auto-update in background if needed.

    Triggers update when:
    1. No local data exists
    2. Remote patch version differs from local
    3. Local data is stale (older than 7 days)
    """
    import asyncio

    STALE_THRESHOLD_DAYS = 7

    try:
        from .ai.data_manager import get_meta, check_patch_version, fetch_and_save_gamedata

        meta = get_meta()

        if meta.get("status") == "no_data":
            # No local data at all — fetch in background
            logger.info("Game data not found, fetching from OpenDota in background...")
            asyncio.create_task(_bg_update_gamedata())
            return

        is_stale = meta.get("age_days", 0) > STALE_THRESHOLD_DAYS

        # Local data exists — check remote version
        try:
            remote_ver, needs_update = await asyncio.get_event_loop().run_in_executor(
                None, check_patch_version
            )
        except Exception as e:
            logger.warning("Patch version check failed (network?): %s. Using local data (patch %s).",
                           e, meta.get("patch_version", "?"))
            # Still update if stale, even if patch check failed
            if is_stale:
                logger.info(
                    "Local data is %.1f days old (threshold %d), forcing update...",
                    meta.get("age_days", 0), STALE_THRESHOLD_DAYS,
                )
                asyncio.create_task(_bg_update_gamedata())
            return

        if needs_update:
            logger.info(
                "New patch detected: %s → %s. Updating game data in background...",
                meta.get("patch_version", "?"), remote_ver,
            )
            asyncio.create_task(_bg_update_gamedata())
        elif is_stale:
            logger.info(
                "Game data is %.1f days old (threshold %d), refreshing in background...",
                meta.get("age_days", 0), STALE_THRESHOLD_DAYS,
            )
            asyncio.create_task(_bg_update_gamedata())
        else:
            logger.info(
                "Game data up to date: patch %s, %d heroes, fetched %s",
                meta.get("patch_version", "?"),
                meta.get("hero_count", 0),
                meta.get("last_fetched", "?"),
            )
    except ImportError:
        logger.debug("data_manager not available, using hardcoded game data")
    except Exception as e:
        logger.warning("Failed to check game data status: %s", e)


async def _check_and_start_learning():
    """Kick off the AI coach self-learning pipeline in the background if due.

    Controlled by AI_COACH_AUTO_LEARN (default on; set to "0" to disable).
    Never blocks startup: the run itself happens in a background task, and
    any failure here only logs a warning.
    """
    import asyncio
    import os

    if os.environ.get("AI_COACH_AUTO_LEARN", "1") == "0":
        logger.info("AI coach auto-learning disabled (AI_COACH_AUTO_LEARN=0)")
        return
    try:
        from .ai.learning.pipeline import should_learn
        if not should_learn():
            logger.info("Learned knowledge is fresh, skipping learning run")
            return
        logger.info("Learning run due, starting in background...")
        asyncio.create_task(_bg_run_learning())
    except ImportError:
        logger.debug("learning pipeline not available")
    except Exception as e:
        logger.warning("Failed to check learning status: %s", e)


async def _bg_run_learning():
    """Background task: run the self-learning pipeline without blocking requests."""
    import asyncio
    try:
        from .ai.learning.pipeline import run_learning
        summary = await asyncio.to_thread(run_learning)
        logger.info(
            "Learning run finished in background: %d chunks written (patch %s)",
            summary.get("total_chunks_written", 0),
            summary.get("patch") or "unknown",
        )
    except Exception as e:
        logger.warning("Background learning run failed: %s", e)


async def _bg_update_gamedata():
    """Background task: fetch game data from OpenDota without blocking requests."""
    import asyncio
    try:
        from .ai.data_manager import fetch_and_save_gamedata
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: fetch_and_save_gamedata(force=True)
        )
        logger.info(
            "Game data updated in background: patch %s, %d heroes, %d items",
            result.get("patch_version", "?"),
            result.get("heroes", 0),
            result.get("items", 0),
        )
    except Exception as e:
        logger.warning("Background game data update failed: %s. Hardcoded fallback still available.", e)


# Serve frontend static files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
