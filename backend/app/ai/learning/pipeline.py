"""Orchestrator for the AI coach self-learning pipeline.

Runs the four learners (see ``base`` module docstring for the level
contract), records per-level results in a run manifest
(``LEARNED_DIR / "_last_learn.json"``) and decides which levels are due.

A level is due when it never succeeded, its last attempt failed (network /
missing prerequisite), its data is older than :data:`RELEARN_AFTER_DAYS`, the
patch changed since it ran, or it was skipped for a configuration reason that
has since been fixed (e.g. an API key was added). Failed levels are never
recorded as successful, so they are retried on the next run instead of being
masked for a week.

Quick / smoke runs (``quick=True``) use tiny samples, write into
``base.SMOKE_DIR`` and never touch the manifest, the full learned outputs or
the cached RAG stores.

Each learner is imported lazily inside :func:`run_learning`, so a broken or
not-yet-available learner degrades to a warning and never prevents the other
levels from running.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .base import LEARNED_DIR, current_patch_version, llm_available, output_dir_for, write_json_atomic

logger = logging.getLogger(__name__)

# Run manifest ("_" prefix => the learned-chunk loader skips it).
LAST_LEARN_FILE = LEARNED_DIR / "_last_learn.json"

# Learned data of a level is considered stale after this many days.
RELEARN_AFTER_DAYS = 7

ALL_LEVELS: tuple[int, ...] = (1, 2, 3, 4)

# Coaching text is bilingual: every learner emits one chunk per language.
LANGS = ("zh", "en")

_TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Level 2 sample size (matches). Quick runs only check the plumbing.
_FULL_MATCH_LIMIT = 100
_QUICK_MATCH_LIMIT = 2


# ---------------------------------------------------------------------------
# Per-level dispatch
# ---------------------------------------------------------------------------

def _run_level(level: int, quick: bool, output_dir: Path) -> dict:
    """Run one learner level; always returns a summary dict, never raises."""
    try:
        if level == 1:
            from . import stats_learner
            return stats_learner.learn_stats(
                langs=LANGS,
                heroes_limit=3 if quick else None,
                output_dir=output_dir,
            )
        if level == 2:
            from . import match_miner
            return match_miner.learn_from_matches(
                source="pro",
                limit=_QUICK_MATCH_LIMIT if quick else _FULL_MATCH_LIMIT,
                langs=LANGS,
                output_dir=output_dir,
            )
        if level == 3:
            from . import llm_distiller
            return llm_distiller.distill(
                langs=LANGS,
                max_chunks=8 if quick else 40,
                output_dir=output_dir,
            )
        if level == 4:
            from . import video_learner
            return video_learner.learn_from_videos(
                langs=LANGS,
                max_videos=1 if quick else 5,
                output_dir=output_dir,
            )
        return {"chunks_written": 0, "error": f"unknown_level_{level}"}
    except ImportError as exc:
        logger.warning("Level %d learner not available: %s", level, exc)
        return {"chunks_written": 0, "error": "learner_unavailable"}
    except Exception:  # a broken learner must never kill the run
        logger.exception("Level %d learner failed", level)
        return {"chunks_written": 0, "error": "learner_crashed"}


def _status(result: dict) -> str:
    if result.get("error"):
        return "failed"
    if result.get("skipped"):
        return "skipped"
    return "ok"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value) -> datetime | None:
    try:
        return datetime.strptime(str(value), _TIME_FMT).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _read_manifest() -> dict | None:
    """The manifest, or None when missing / unreadable."""
    if not LAST_LEARN_FILE.exists():
        return None
    try:
        manifest = json.loads(LAST_LEARN_FILE.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Learn manifest unreadable, treating every level as due")
        return None
    return manifest if isinstance(manifest, dict) else None


def _level_records(manifest: dict | None) -> dict[str, dict]:
    """Per-level records; legacy manifests (one timestamp) count as all-ok."""
    if not manifest:
        return {}
    levels = manifest.get("levels")
    if isinstance(levels, dict):
        return {str(k): v for k, v in levels.items() if isinstance(v, dict)}
    if manifest.get("timestamp"):
        return {
            str(level): {"status": "ok", "finished_at": manifest["timestamp"],
                         "patch": manifest.get("patch", "")}
            for level in ALL_LEVELS
        }
    return {}


def _prerequisites_met(level: int) -> bool:
    """Whether a level skipped for configuration reasons could run now."""
    if level == 3:
        return llm_available()
    if level == 4:
        return llm_available() and shutil.which("yt-dlp") is not None
    return True


def _level_due(level: int, record: dict | None, current_patch: str, now: datetime) -> bool:
    if not record:
        return True
    status = record.get("status")
    if status == "skipped":
        return _prerequisites_met(level)
    if status != "ok":
        return True
    finished = _parse_time(record.get("finished_at"))
    if finished is None or (now - finished).total_seconds() / 86400 > RELEARN_AFTER_DAYS:
        return True
    return bool(current_patch) and record.get("patch", "") != current_patch


def due_levels() -> tuple[int, ...]:
    """Levels whose learned data is missing, failed, stale or from another patch."""
    try:
        records = _level_records(_read_manifest())
        current = current_patch_version()
        now = _now()
        return tuple(
            level for level in ALL_LEVELS
            if _level_due(level, records.get(str(level)), current, now)
        )
    except Exception as exc:
        logger.warning("due_levels check failed: %s", exc)
        return ()


def should_learn() -> bool:
    """Whether a learning run is due (any level due). Never raises."""
    due = due_levels()
    if due:
        logger.info("Learning levels due: %s", list(due))
    return bool(due)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_learning(levels: tuple[int, ...] | None = None, quick: bool = False) -> dict:
    """Run learner levels and return a summary dict. Never raises.

    ``levels=None`` runs the levels that are due (all levels for a quick
    run). A full run records each level's outcome in the manifest — only
    levels that succeeded (or were skipped by configuration) are marked
    done — and resets the cached RAG stores when new chunks were written.
    """
    started_at = _now()
    patch = current_patch_version()
    if levels is None:
        levels = ALL_LEVELS if quick else due_levels()
    levels = tuple(sorted({int(level) for level in levels}))
    if not levels:
        logger.info("Learning run: no level is due, nothing to do")
        return {
            "started_at": started_at.strftime(_TIME_FMT),
            "finished_at": _now().strftime(_TIME_FMT),
            "quick": quick, "patch": patch, "levels": {},
            "total_chunks_written": 0, "success": True,
        }
    output_dir = output_dir_for(quick)
    logger.info(
        "Learning run started: levels=%s quick=%s patch=%s output=%s",
        list(levels), quick, patch or "unknown", output_dir,
    )

    level_results: dict[str, dict] = {}
    total_chunks = 0
    for level in levels:
        result = _run_level(level, quick, output_dir)
        if not isinstance(result, dict):
            result = {"chunks_written": 0, "error": "learner_returned_non_dict"}
        result = dict(result)
        result["status"] = _status(result)
        level_results[str(level)] = result
        chunks = int(result.get("chunks_written") or 0)
        total_chunks += chunks
        if result["status"] == "failed":
            logger.warning("Level %d failed: %s", level, result.get("error"))
        elif result["status"] == "skipped":
            logger.info("Level %d skipped: %s", level, result.get("skipped"))
        else:
            logger.info("Level %d: %d chunks written", level, chunks)

    finished_at = _now().strftime(_TIME_FMT)
    success = bool(level_results) and all(r["status"] != "failed" for r in level_results.values())
    summary = {
        "started_at": started_at.strftime(_TIME_FMT),
        "finished_at": finished_at,
        "quick": quick,
        "patch": patch,
        "levels": level_results,
        "total_chunks_written": total_chunks,
        "success": success,
        "output_dir": str(output_dir),
    }
    if quick:
        logger.info("Quick learning run finished: %d chunks written to %s", total_chunks, output_dir)
        return summary

    _update_manifest(level_results, finished_at, patch, summary)

    if total_chunks > 0:
        # The per-language RAG stores cached in knowledge_base are now stale.
        try:
            from ..knowledge_base import reset_knowledge_stores
            reset_knowledge_stores()
        except Exception as exc:
            logger.warning("Failed to reset knowledge stores: %s", exc)

    logger.info("Learning run finished: %d total chunks written (success=%s)", total_chunks, success)
    return summary


def _update_manifest(level_results: dict[str, dict], finished_at: str, patch: str,
                     summary: dict) -> None:
    """Record per-level outcomes; nothing is written when every level failed."""
    if not any(r["status"] != "failed" for r in level_results.values()):
        logger.warning("No learning level succeeded; manifest left unchanged so the run is retried")
        return
    manifest = _read_manifest() or {}
    records = _level_records(manifest)
    for level, result in level_results.items():
        if result["status"] == "failed":
            previous = records.get(level) or {}
            # Keep data from an earlier success, but make the level due again.
            records[level] = {**previous, "status": "failed", "error": result.get("error"),
                              "failed_at": finished_at}
        else:
            records[level] = {
                "status": result["status"],
                "finished_at": finished_at,
                "patch": patch,
                "chunks_written": int(result.get("chunks_written") or 0),
                **({"reason": result["skipped"]} if result.get("skipped") else {}),
            }
    manifest = {
        "timestamp": finished_at,
        "patch": patch,
        "levels": records,
        "summary": summary,
    }
    try:
        write_json_atomic(LAST_LEARN_FILE, manifest)
        logger.info("Learn manifest written: %s", LAST_LEARN_FILE.name)
    except Exception as exc:
        logger.warning("Failed to write learn manifest: %s", exc)


__all__ = [
    "run_learning",
    "should_learn",
    "due_levels",
    "LAST_LEARN_FILE",
    "RELEARN_AFTER_DAYS",
    "ALL_LEVELS",
]
