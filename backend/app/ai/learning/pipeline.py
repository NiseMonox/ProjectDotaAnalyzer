"""Orchestrator for the AI coach self-learning pipeline.

Runs the four learners (see ``base`` module docstring for the level
contract), records a run manifest to ``LEARNED_DIR / "_last_learn.json"``,
and decides when a new learning run is due (weekly, or on patch change).

Each learner is imported lazily inside :func:`run_learning`, so a broken or
not-yet-available learner degrades to a warning and never prevents the other
levels from running.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from .base import LEARNED_DIR, current_patch_version

logger = logging.getLogger(__name__)

# Run manifest written after every run_learning() call ("_" prefix => the
# learned-chunk loader in base.load_learned_chunks skips it).
LAST_LEARN_FILE = LEARNED_DIR / "_last_learn.json"

# A learning run is considered stale after this many days.
RELEARN_AFTER_DAYS = 7

# Coaching text is bilingual: every learner emits one chunk per language.
LANGS = ("zh", "en")

_TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"


# ---------------------------------------------------------------------------
# Per-level dispatch
# ---------------------------------------------------------------------------

def _run_level(level: int, quick: bool) -> dict:
    """Run one learner level; always returns a summary dict, never raises.

    ``quick=True`` shrinks every sample so a full pipeline smoke run stays
    cheap (a handful of OpenDota calls, tiny LLM usage, at most one video).
    """
    try:
        if level == 1:
            from . import stats_learner
            return stats_learner.learn_stats(
                langs=LANGS,
                heroes_limit=3 if quick else None,
            )
        if level == 2:
            from . import match_miner
            return match_miner.learn_from_matches(
                source="pro",
                limit=2 if quick else 20,
                langs=LANGS,
            )
        if level == 3:
            from . import llm_distiller
            return llm_distiller.distill(
                langs=LANGS,
                max_chunks=8 if quick else 40,
            )
        if level == 4:
            from . import video_learner
            return video_learner.learn_from_videos(
                langs=LANGS,
                max_videos=1 if quick else 5,
            )
        return {"chunks_written": 0, "error": f"unknown level: {level}"}
    except ImportError as exc:
        logger.warning("Level %d learner not available: %s", level, exc)
        return {"chunks_written": 0, "error": f"learner not available: {exc}"}
    except Exception as exc:  # a broken learner must never kill the run
        logger.exception("Level %d learner failed", level)
        return {"chunks_written": 0, "error": str(exc)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_learning(levels: tuple[int, ...] = (1, 2, 3, 4), quick: bool = False) -> dict:
    """Run the requested learner levels and return a summary dict.

    Afterwards writes the run manifest (timestamp + patch + summary) and
    resets the cached per-language RAG stores so the next coaching request
    picks up the fresh learned chunks.
    """
    started_at = datetime.now(timezone.utc)
    patch = current_patch_version()
    logger.info(
        "Learning run started: levels=%s quick=%s patch=%s",
        list(levels), quick, patch or "unknown",
    )

    level_results: dict[str, dict] = {}
    total_chunks = 0
    for level in levels:
        result = _run_level(level, quick)
        if not isinstance(result, dict):
            result = {"chunks_written": 0, "error": "learner returned non-dict summary"}
        level_results[str(level)] = result
        chunks = int(result.get("chunks_written") or 0)
        total_chunks += chunks
        if result.get("error"):
            logger.warning("Level %d: 0 chunks (%s)", level, result["error"])
        else:
            logger.info("Level %d: %d chunks written", level, chunks)

    summary = {
        "started_at": started_at.strftime(_TIME_FMT),
        "finished_at": datetime.now(timezone.utc).strftime(_TIME_FMT),
        "quick": quick,
        "patch": patch,
        "levels": level_results,
        "total_chunks_written": total_chunks,
    }

    try:
        manifest = {
            "timestamp": summary["finished_at"],
            "patch": patch,
            "summary": summary,
        }
        with open(LAST_LEARN_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
        logger.info("Learn manifest written: %s", LAST_LEARN_FILE.name)
    except Exception as exc:
        logger.warning("Failed to write learn manifest: %s", exc)

    # The per-language RAG stores cached in knowledge_base are now stale.
    try:
        from ..knowledge_base import reset_knowledge_stores
        reset_knowledge_stores()
    except Exception as exc:
        logger.warning("Failed to reset knowledge stores: %s", exc)

    logger.info("Learning run finished: %d total chunks written", total_chunks)
    return summary


def should_learn() -> bool:
    """Whether a new learning run is due. Never raises.

    True when the manifest is missing/malformed, older than
    :data:`RELEARN_AFTER_DAYS`, or recorded for a different patch.
    Any unexpected failure (e.g. patch lookup problems) returns False so
    startup never triggers a learn run on flaky information.
    """
    try:
        if not LAST_LEARN_FILE.exists():
            return True
        try:
            manifest = json.loads(LAST_LEARN_FILE.read_text(encoding="utf-8"))
            last_run = datetime.strptime(
                manifest.get("timestamp", ""), _TIME_FMT
            ).replace(tzinfo=timezone.utc)
        except Exception:
            logger.warning("Learn manifest unreadable, scheduling a new run")
            return True
        age_days = (datetime.now(timezone.utc) - last_run).total_seconds() / 86400
        if age_days > RELEARN_AFTER_DAYS:
            logger.info("Learned data is %.1f days old, learning run due", age_days)
            return True
        current = current_patch_version()
        if current and manifest.get("patch", "") != current:
            logger.info(
                "Patch changed (%s -> %s), learning run due",
                manifest.get("patch", "?"), current,
            )
            return True
        return False
    except Exception as exc:
        logger.warning("should_learn check failed: %s", exc)
        return False


__all__ = ["run_learning", "should_learn", "LAST_LEARN_FILE", "RELEARN_AFTER_DAYS"]
