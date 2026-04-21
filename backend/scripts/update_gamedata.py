"""CLI script to update game data from OpenDota API.

Usage:
    python -m backend.scripts.update_gamedata [--force] [--check-only]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Update Dota 2 game data from OpenDota API")
    parser.add_argument("--force", action="store_true", help="Force update even if patch version matches")
    parser.add_argument("--check-only", action="store_true", help="Only check if update is needed")
    args = parser.parse_args()

    from backend.app.ai.data_manager import check_patch_version, fetch_and_save_gamedata, get_meta

    if args.check_only:
        try:
            remote_ver, needs_update = check_patch_version()
            meta = get_meta()
            print(f"Remote patch version: {remote_ver}")
            print(f"Local patch version:  {meta.get('patch_version', 'none')}")
            print(f"Last fetched:         {meta.get('last_fetched', 'never')}")
            print(f"Age (days):           {meta.get('age_days', 'N/A')}")
            print(f"Needs update:         {needs_update}")
            print(f"Is stale (>14 days):  {meta.get('is_stale', 'N/A')}")
        except Exception as e:
            print(f"Error checking patch version: {e}")
            sys.exit(1)
        return

    print("Updating game data from OpenDota API...")
    try:
        result = fetch_and_save_gamedata(force=args.force)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result.get("status") == "up_to_date":
            print("\nGame data is already up to date.")
        else:
            print("\nGame data updated successfully!")
    except Exception as e:
        print(f"Error updating game data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
