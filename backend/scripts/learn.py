"""CLI script to run the AI coach self-learning pipeline.

Usage:
    python -m backend.scripts.learn                    # run all levels if due
    python -m backend.scripts.learn --levels 1,2       # only some levels
    python -m backend.scripts.learn --quick --force    # tiny smoke run
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
    """Print the last-learn manifest and per-file learned chunk counts."""
    from backend.app.ai.learning.base import LEARNED_DIR
    from backend.app.ai.learning.pipeline import LAST_LEARN_FILE, should_learn

    if LAST_LEARN_FILE.exists():
        try:
            manifest = json.loads(LAST_LEARN_FILE.read_text(encoding="utf-8"))
            print("Last learning run:")
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
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

    print(f"\nLearning run due: {should_learn()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI coach self-learning pipeline")
    parser.add_argument("--levels", default="1,2,3,4",
                        help='Comma-separated learner levels to run (default: "1,2,3,4")')
    parser.add_argument("--quick", action="store_true",
                        help="Use tiny samples for a fast/cheap smoke run")
    parser.add_argument("--force", action="store_true",
                        help="Run even if learned data is fresh (bypass should_learn)")
    parser.add_argument("--status", action="store_true",
                        help="Print last learning run and per-file chunk counts, then exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Load DEEPSEEK_API_KEY etc. from the repo-root .env
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    if args.status:
        _print_status()
        return

    from backend.app.ai.learning.pipeline import run_learning, should_learn

    if not args.force and not should_learn():
        print("Learned data is fresh — nothing to do (use --force to run anyway).")
        return

    try:
        levels = tuple(int(x) for x in args.levels.split(",") if x.strip())
    except ValueError:
        print(f'Invalid --levels value: "{args.levels}" (expected e.g. "1,2,3,4")')
        sys.exit(2)

    summary = run_learning(levels=levels, quick=args.quick)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # Exit 0 even if some levels wrote 0 chunks — learners degrade gracefully.


if __name__ == "__main__":
    main()
