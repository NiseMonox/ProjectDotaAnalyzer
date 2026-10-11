"""CLI script to run the AI coach self-learning pipeline.

Usage:
    python -m backend.scripts.learn                    # run the levels that are due
    python -m backend.scripts.learn --levels 1,2       # only some levels (even if fresh)
    python -m backend.scripts.learn --force            # every level, even if fresh
    python -m backend.scripts.learn --quick            # tiny smoke run into learned/_smoke/
    python -m backend.scripts.learn --status           # show last run + files

Exit codes: 0 = every level succeeded or was skipped by configuration,
1 = a level failed or another run holds the lock, 2 = bad arguments.
A running server notices the new learned files by itself within about 30
seconds; no restart is needed. After a full run that wrote chunks the new
texts also get their retrieval vectors here, when this machine has the
optional embedding model (backend/requirements-rag.txt); without it they
stay retrievable through tags and keywords.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import osutil  # noqa: E402  (imports nothing from the project)


def _print_status() -> None:
    """Print the last-learn manifest, per-file chunk counts and due levels."""
    from backend.app.ai.learning.base import LEARNED_DIR, load_learned_chunks
    from backend.app.ai.learning.pipeline import (
        ALL_LEVELS, LAST_LEARN_FILE, RELEARN_AFTER_DAYS, due_levels, next_due_times,
    )

    if LAST_LEARN_FILE.exists():
        try:
            manifest = json.loads(LAST_LEARN_FILE.read_text(encoding="utf-8"))
            print("Per-level status:")
            print(json.dumps(manifest.get("levels", {}), indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"Last-learn manifest unreadable: {e}")
    else:
        print("No learning run recorded yet.")

    print("\nLearned files:")
    files = [p for p in sorted(LEARNED_DIR.glob("*.json")) if not p.name.startswith("_")]
    if not files:
        print("  (none)")
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            meta = payload.get("meta", {})
            count = meta.get("chunk_count", len(payload.get("chunks", [])))
            print(f"  {path.name}: {count} chunks (generated {meta.get('generated_at', '?')}, "
                  f"patch {meta.get('patch_label') or meta.get('patch') or '?'})")
        except Exception as e:
            print(f"  {path.name}: unreadable ({e})")
    # What the coach really gets: no chunks of quick runs or below the minimum samples.
    usable = {lang: len(load_learned_chunks(lang)) for lang in ("zh", "en")}
    print(f"  usable for coaching: {usable['zh']} zh / {usable['en']} en chunks")

    due, upcoming = due_levels(), next_due_times()
    print(f"\nLevels due: {list(due) or 'none'}")
    for level in ALL_LEVELS:
        when = upcoming.get(str(level))
        if level in due:
            print(f"  level {level}: due now")
        elif when:
            print(f"  level {level}: next due {when} ({RELEARN_AFTER_DAYS} days after its last run; "
                  "earlier when the game patch changes)")
        else:
            print(f"  level {level}: skipped for a missing prerequisite (API key / yt-dlp); due once it is there")


def _sync_vectors() -> None:
    """Embed the new learned chunks into the runtime vector cache (best effort).

    A one-shot process has no warmed-up retrieval stack that would do it in
    the background; a running server would only catch up at its next sync.
    Never fails the run: the outcome is one printed line.
    """
    try:
        from backend.app.ai.rag import runtime
        stats = runtime.sync("learning")
    except Exception as e:  # sync() does not raise; an import problem might
        stats = {"error": f"{type(e).__name__}: {e}"}
    if "not_run" in stats or "error" in stats:
        print(f"Retrieval vectors not built ({stats.get('error') or stats.get('not_run')}); "
              "the new chunks are ranked by tags and keywords.")
    else:
        print(f"Retrieval vectors: {stats.get('embedded', 0)} embedded, {stats.get('vectors', 0)} in the cache.")


def _parse_levels(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    levels = tuple(int(x) for x in value.split(",") if x.strip())
    if not levels or any(level not in (1, 2, 3, 4) for level in levels):
        raise ValueError(value)  # also "" / " " / ",": naming no level is a mistake
    return levels


def main() -> None:
    # Output redirected to a file or pipe on Windows is encoded with the system code page:
    # a character it lacks is escaped instead of ending the command.
    osutil.harden_stdio()
    parser = argparse.ArgumentParser(description="Run the AI coach self-learning pipeline")
    parser.add_argument("--levels", default=None,
                        help='Comma-separated learner levels to run, e.g. "1,2" (default: the due ones)')
    parser.add_argument("--quick", action="store_true",
                        help="Tiny samples for a fast/cheap smoke run (written to learned/_smoke/)")
    parser.add_argument("--force", action="store_true",
                        help="Run every requested level even if its data is fresh")
    parser.add_argument("--status", action="store_true",
                        help="Print the per-level status and per-file chunk counts, then exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # DEEPSEEK_API_KEY etc. come from the repo-root .env, which importing the
    # configuration loads: in the encoding the file has (Windows tools write
    # UTF-16 or a byte order mark, which a plain load_dotenv() cannot read).
    from backend.app import config  # noqa: F401

    if args.status:
        _print_status()
        return

    from backend.app.ai.learning.pipeline import ALL_LEVELS, due_levels, run_learning

    try:
        levels = _parse_levels(args.levels)
    except ValueError:
        print(f'Invalid --levels value: "{args.levels}" (expected e.g. "1,2,3,4")')
        sys.exit(2)

    if levels is None and (args.force or args.quick):
        levels = ALL_LEVELS
    if levels is None and not due_levels():
        print("Learned data is fresh - nothing to do (use --force to run anyway).")
        return

    summary = run_learning(levels=levels, quick=args.quick)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary.get("error") == "already_running":
        print("Another learning run is in progress (server startup run or another CLI run); "
              "nothing was done.")
    if not args.quick and summary.get("total_chunks_written"):
        _sync_vectors()
    # Exit 1 when a level failed so cron / CI notice it.
    sys.exit(0 if summary.get("success", True) else 1)


if __name__ == "__main__":
    main()
