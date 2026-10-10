"""Orchestrator for the AI coach self-learning pipeline.

Runs the four learners (see ``base`` module docstring for the level
contract), records per-level results in a run manifest
(``LEARNED_DIR / "_last_learn.json"``) and decides which levels are due.

A level is due when it never succeeded, its last attempt failed (network /
missing prerequisite) or learned only part of its data, its data is older
than :data:`RELEARN_AFTER_DAYS`, the patch changed since it ran, or it was
skipped for a configuration reason that has since been fixed (e.g. an API
key was added). Level 3 is also due when level 2 succeeded after it (its
input changed). Failed levels are never recorded as successful, so they are
retried on the next run instead of being masked for a week.

Quick / smoke runs (``quick=True``) use tiny samples and relaxed minimum
sample sizes, write into ``base.SMOKE_DIR`` and never touch the manifest,
the full learned outputs or the cached RAG stores. Old versions wrote them
into the main directory: the first full run moves such leftovers to the
smoke directory (the chunk loader ignores them until then).

Only one run writes to an output directory at a time (an in-process lock
plus a lock file, so a CLI / cron run and the server's startup run cannot
overlap, also under different users); a second run returns
``error: "already_running"`` at once.

Each learner is imported lazily inside :func:`run_learning`, so a broken or
not-yet-available learner degrades to a warning and never prevents the other
levels from running.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .base import (
    LEARNED_DIR,
    MANIFEST_FILE,
    clear_stop,
    current_patch_version,
    llm_available,
    output_dir_for,
    request_stop,
    stop_requested,
    write_json_atomic,
)

try:  # POSIX only; elsewhere the in-process lock still applies
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

logger = logging.getLogger(__name__)

# Run manifest ("_" prefix => the learned-chunk loader skips it).
LAST_LEARN_FILE = LEARNED_DIR / MANIFEST_FILE

# Learned data of a level is considered stale after this many days.
RELEARN_AFTER_DAYS = 7

ALL_LEVELS: tuple[int, ...] = (1, 2, 3, 4)

# Coaching text is bilingual: every learner emits one chunk per language.
LANGS = ("zh", "en")

_TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Level 2 sample size (matches). Quick runs only check the plumbing: two
# matches can never reach the real minimum samples, so they render and save
# their chunks (into the smoke directory, never loaded) with minimums of 1.
_FULL_MATCH_LIMIT = 100
_QUICK_MATCH_LIMIT = 2
_QUICK_MIN_SAMPLES = {"position": 1, "hero_games": 1, "item": 1}

# Lock file of a run ("_" prefix and no .json suffix: never loaded as chunks).
LOCK_FILE_NAME = "_learn.lock"
_run_locks_guard = threading.Lock()
_run_locks: dict[str, threading.Lock] = {}
_active_runs = 0  # runs of this process holding a run lock (under _run_locks_guard)


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
                min_samples=_QUICK_MIN_SAMPLES if quick else None,
            )
        if level == 3:
            from . import llm_distiller
            return llm_distiller.distill(
                langs=LANGS,
                max_chunks=8 if quick else 40,
                output_dir=output_dir,
                require_sufficient=not quick,
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
    """Per-level records; legacy manifests (one timestamp) count as all-ok,
    except those written by an old quick run (tiny samples, every level due)."""
    if not manifest:
        return {}
    levels = manifest.get("levels")
    if isinstance(levels, dict):
        return {str(k): v for k, v in levels.items() if isinstance(v, dict)}
    summary = manifest.get("summary")
    if isinstance(summary, dict) and summary.get("quick"):
        return {}
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


def _mined_after_distilled(records: dict[str, dict]) -> bool:
    """Level 2 succeeded later than level 3: the distilled tips are built on
    older aggregates. Levels of one run share a timestamp, so a run that
    did both never makes level 3 due again."""
    mined, distilled = records.get("2") or {}, records.get("3") or {}
    if mined.get("status") != "ok" or distilled.get("status") != "ok":
        return False
    mined_at = _parse_time(mined.get("finished_at"))
    distilled_at = _parse_time(distilled.get("finished_at"))
    return mined_at is not None and distilled_at is not None and mined_at > distilled_at


def due_levels() -> tuple[int, ...]:
    """Levels whose learned data is missing, failed, stale or from another patch."""
    try:
        records = _level_records(_read_manifest())
        current = current_patch_version()
        now = _now()
        due = {
            level for level in ALL_LEVELS
            if _level_due(level, records.get(str(level)), current, now)
        }
        if 3 not in due and _mined_after_distilled(records):
            due.add(3)
        return tuple(sorted(due))
    except Exception as exc:
        logger.warning("due_levels check failed: %s", exc)
        return ()


def next_due_times() -> dict[str, str | None]:
    """When each level becomes due by age: ``{"1": "2026-10-17T21:38:33Z", ...}``.

    The time its data turns :data:`RELEARN_AFTER_DAYS` old (UTC); None for a
    level that is due now, never ran, or was skipped for a missing
    prerequisite (due as soon as that is there). A new game patch makes
    every level due earlier. Never raises.
    """
    times: dict[str, str | None] = {str(level): None for level in ALL_LEVELS}
    try:
        records = _level_records(_read_manifest())
        due = set(due_levels())
        for level in ALL_LEVELS:
            record = records.get(str(level)) or {}
            finished = _parse_time(record.get("finished_at"))
            if level in due or record.get("status") != "ok" or finished is None:
                continue
            times[str(level)] = (finished + timedelta(days=RELEARN_AFTER_DAYS)).strftime(_TIME_FMT)
    except Exception as exc:
        logger.warning("next_due_times check failed: %s", exc)
    return times


def should_learn() -> bool:
    """Whether a learning run is due (any level due). Never raises."""
    due = due_levels()
    if due:
        logger.info("Learning levels due: %s", list(due))
    return bool(due)


def learning_status() -> dict:
    """State of the learned knowledge, for status pages. Never raises.

    ``{"due_levels": [int], "next_due": {<level>: UTC time or None},
    "levels": {<level>: record}, "last_run": <timestamp of the last recorded
    run or None>, "patch": <its patch or None>}``; ``levels`` holds the
    manifest's per-level records (status, finished_at, patch, error,
    chunks_written, ...) and is empty before the first recorded run;
    ``next_due`` is :func:`next_due_times`.
    """
    status: dict = {"due_levels": list(due_levels()), "next_due": next_due_times(),
                    "levels": {}, "last_run": None, "patch": None}
    try:
        manifest = _read_manifest() or {}
        levels = manifest.get("levels")
        status["levels"] = levels if isinstance(levels, dict) else {}
        status["last_run"] = manifest.get("timestamp")
        status["patch"] = manifest.get("patch")
    except Exception as exc:
        logger.warning("Learning status unavailable: %s", exc)
    return status


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _open_lock_file(path: Path) -> int:
    """Read-only descriptor of the lock file (created 0644 when missing).

    ``flock`` needs no write access, so a lock file that belongs to another
    user (a cron run as root next to the service) can still be locked.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o644)
    except PermissionError:  # exists, but this user may not create files here
        return os.open(path, os.O_RDONLY)
    with contextlib.suppress(OSError):  # someone else's file: left as it is
        os.fchmod(fd, 0o644)  # ours: readable for the other user whatever the umask
    return fd


@contextlib.contextmanager
def _exclusive_run(output_dir: Path):
    """Yield True when this is the only learning run writing to *output_dir*.

    A thread lock covers runs in this process, ``flock`` on
    ``<output_dir>/_learn.lock`` covers other processes (the OS releases it
    when the process dies). Non-blocking: a second run yields False.

    A stop request only concerns the runs in progress: it is withdrawn when
    a run starts while no other one is in progress, and when the last one ends.
    """
    global _active_runs
    with _run_locks_guard:
        lock = _run_locks.setdefault(str(output_dir), threading.Lock())
    if not lock.acquire(blocking=False):
        yield False
        return
    fd = None
    counted = False
    try:
        if fcntl is not None:
            try:
                fd = _open_lock_file(output_dir / LOCK_FILE_NAME)
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            except OSError as exc:  # no lock file possible: in-process lock only
                logger.warning("Cannot take the learning lock file: %s", exc)
        with _run_locks_guard:
            if _active_runs == 0:
                clear_stop()
            _active_runs += 1
            counted = True
        yield True
    finally:
        if counted:
            with _run_locks_guard:
                _active_runs -= 1
                if _active_runs == 0:
                    clear_stop()
        if fd is not None:
            os.close(fd)  # closing releases the flock
        lock.release()


def _empty_summary(started_at: datetime, quick: bool, patch: str, **extra) -> dict:
    return {
        "started_at": started_at.strftime(_TIME_FMT),
        "finished_at": _now().strftime(_TIME_FMT),
        "quick": quick, "patch": patch, "levels": {},
        "total_chunks_written": 0, **extra,
    }


def run_learning(levels: tuple[int, ...] | None = None, quick: bool = False) -> dict:
    """Run learner levels and return a summary dict. Never raises.

    ``levels=None`` runs the levels that are due (all levels for a quick
    run). A full run records each level's outcome in the manifest — only
    levels that succeeded (or were skipped by configuration) are marked
    done — and resets the cached RAG stores when new chunks were written.
    When another run is already writing to the same output directory the
    summary has ``success: False`` and ``error: "already_running"``.
    """
    started_at = _now()
    patch = current_patch_version()
    if levels is None:
        levels = ALL_LEVELS if quick else due_levels()
    levels = tuple(sorted({int(level) for level in levels}))
    if not levels:
        logger.info("Learning run: no level is due, nothing to do")
        return _empty_summary(started_at, quick, patch, success=True)
    output_dir = output_dir_for(quick)
    with _exclusive_run(output_dir) as acquired:
        if not acquired:
            logger.warning("Learning run skipped: another run is writing to %s", output_dir)
            return _empty_summary(started_at, quick, patch, success=False, error="already_running")
        return _run_levels(levels, quick, output_dir, started_at, patch)


def _retire_legacy_quick_outputs(output_dir: Path) -> None:
    """Move the outputs of an old quick run out of the main directory.

    Old versions ran quick runs into the main directory and recorded them
    in the manifest (``summary.quick``). Their chunk files, raw aggregates
    and video registry describe a handful of matches; a full run must not
    build on them (a kept video would never be distilled again, kept chunks
    would pass as full-run knowledge). They go to the smoke directory under
    their own names, where quick outputs live. Never raises.

    The manifest goes last: it is the only thing that marks the unflagged
    chunk files beside it as a quick run's (``base.load_learned_chunks``),
    so until every one of them is gone — also when a move fails halfway —
    it has to stay where the loader finds it.
    """
    manifest = _read_manifest() or {}
    summary = manifest.get("summary")
    if "levels" in manifest or not (isinstance(summary, dict) and summary.get("quick") is True):
        return
    moved = []
    try:
        smoke_dir = output_dir_for(True)
        # chunk files, then registries and aggregates, then the manifest
        for path in sorted(output_dir.glob("*.json"),
                           key=lambda p: (p.name == MANIFEST_FILE, p.name.startswith("_"), p.name)):
            if path.name.startswith("_"):
                os.replace(path, smoke_dir / path.name)
                moved.append(path.name)
                continue
            try:
                meta = json.loads(path.read_text(encoding="utf-8")).get("meta") or {}
            except Exception:
                meta = {}
            if "quick" not in meta:  # a file that says quick: false was written by a full run
                os.replace(path, smoke_dir / path.name)
                moved.append(path.name)
    except OSError as exc:
        logger.warning("Could not move the outputs of an old quick run: %s", exc)
    if moved:
        logger.info("Outputs of an old quick run moved to the smoke directory: %s", ", ".join(moved))


def _run_levels(levels: tuple[int, ...], quick: bool, output_dir: Path,
                started_at: datetime, patch: str) -> dict:
    """The body of :func:`run_learning` (caller holds the run lock)."""
    logger.info(
        "Learning run started: levels=%s quick=%s patch=%s output=%s",
        list(levels), quick, patch or "unknown", output_dir,
    )
    if not quick:
        _retire_legacy_quick_outputs(output_dir)

    level_results: dict[str, dict] = {}
    total_chunks = 0
    for level in levels:
        if stop_requested():
            result = {"chunks_written": 0, "error": "cancelled"}
        else:
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
    "learning_status",
    "request_stop",
    "should_learn",
    "due_levels",
    "next_due_times",
    "LAST_LEARN_FILE",
    "RELEARN_AFTER_DAYS",
    "ALL_LEVELS",
]
