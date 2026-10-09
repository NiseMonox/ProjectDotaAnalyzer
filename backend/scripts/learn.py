"""CLI script to run the AI coach self-learning pipeline.

Usage:
    python -m backend.scripts.learn                    # run the levels that are due
    python -m backend.scripts.learn --levels 1,2       # only some levels (even if fresh)
    python -m backend.scripts.learn --force            # every level, even if fresh
    python -m backend.scripts.learn --quick            # tiny smoke run into learned/_smoke/
    python -m backend.scripts.learn --status           # show last run + files
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


def _print_status() -> None:
    """Print the last-learn manifest, per-file chunk counts and due levels."""
    from backend.app.ai.learning.base import LEARNED_DIR
    from backend.app.ai.learning.pipeline import LAST_LEARN_FILE, due_levels

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
            print(f"  {path.name}: {count} chunks (generated {meta.get('generated_at', '?')})")
        except Exception as e:
            print(f"  {path.name}: unreadable ({e})")

    print(f"\nLevels due: {list(due_levels()) or 'none'}")


def _parse_levels(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    levels = tuple(int(x) for x in value.split(",") if x.strip())
    if any(level not in (1, 2, 3, 4) for level in levels):
        raise ValueError(value)
    return levels


def main() -> None:
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

    # Load DEEPSEEK_API_KEY etc. from the repo-root .env (config does this too).
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

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
        print("Learned data is fresh — nothing to do (use --force to run anyway).")
        return

    summary = run_learning(levels=levels, quick=args.quick)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # Exit 1 when a level failed so cron / CI notice it.
    sys.exit(0 if summary.get("success", True) else 1)


if __name__ == "__main__":
    main()
