"""CLI script to update the fetched game data (backend/data/gamedata).

Usage:
    python -m backend.scripts.update_gamedata [--force] [--allow-shrink] [--check-only]

The data comes from the first source that delivers a complete set: the
dotaconstants master branch on GitHub, its jsDelivr mirror, then the OpenDota
API (the npm release, which lags the game by months; see
``data_manager.CONSTANTS_SOURCES``). The game version, including letter
patches, is read from Valve's patch feed. Without ``--force`` nothing is
downloaded when the local data is current.

Stored data is never replaced by worse data: while the master branch cannot
be reached, data that came from it is kept instead of the API's older
release, and a download that lost more than a tenth of the heroes, items or
talents is refused as a broken upstream build. When such a loss is real,
``--allow-shrink`` accepts it.

A running server notices the new files by itself within about 30 seconds
(data_manager checks meta.json); no restart is needed. The committed
catalogs and the numeric fallback are a separate step
(``python -m backend.scripts.gen_static_catalogs``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import osutil  # noqa: E402  (imports nothing from the project)


def main() -> None:
    # Output redirected to a file or pipe on Windows is encoded with the system code page:
    # a character it lacks is escaped instead of ending the command.
    osutil.harden_stdio()
    parser = argparse.ArgumentParser(description="Update Dota 2 game data (dotaconstants master, mirrors, OpenDota API)")
    parser.add_argument("--force", action="store_true", help="Download even if the local data is current")
    parser.add_argument("--allow-shrink", action="store_true",
                        help="Accept data with far fewer heroes / items / talents than the stored set")
    parser.add_argument("--check-only", action="store_true", help="Only check if an update is needed")
    args = parser.parse_args()

    from backend.app.ai.data_manager import (
        check_patch_version, fetch_and_save_gamedata, get_meta, get_patch_info,
    )
    from backend.app.config import GAMEDATA_STALE_DAYS

    if args.check_only:
        try:
            remote_ver, needs_update = check_patch_version()
            meta = get_meta()
            info = get_patch_info()
            print(f"Game version (Valve):  {remote_ver}")
            print(f"Local data, major:     {meta.get('patch_version', 'none')}")
            print(f"Local data reflects:   {info['data_label'] or 'unknown'}"
                  f"{'' if info['data_current'] is not False else '  (older than the game)'}")
            print(f"Source:                {meta.get('source', 'none')}"
                  f" {meta.get('source_revision') or ''} {meta.get('source_date') or ''}".rstrip())
            print(f"Last fetched:          {meta.get('last_fetched', 'never')}")
            print(f"Age (days):            {meta.get('age_days', 'N/A')}")
            print(f"Needs update:          {needs_update}")
            print(f"Outdated file format:  {meta.get('outdated_format', 'N/A')}")
            print(f"Is stale (>{GAMEDATA_STALE_DAYS} days):    {meta.get('is_stale', 'N/A')}")
        except Exception as e:
            print(f"Error checking patch version: {e}")
            sys.exit(1)
        return

    print("Updating game data...")
    try:
        result = fetch_and_save_gamedata(force=args.force, allow_shrink=args.allow_shrink)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result.get("status") == "up_to_date":
            print("\nGame data is already up to date.")
        elif not result.get("changed", True):
            print("\nGame data fetched; it equals what was stored.")
        else:
            print("\nGame data updated successfully!")
    except Exception as e:
        print(f"Error updating game data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
