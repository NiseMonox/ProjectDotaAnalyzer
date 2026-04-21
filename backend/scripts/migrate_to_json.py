"""One-time migration: extract editorial content from hardcoded Python dicts to JSON files.

Reads ABILITY_DATA, ITEM_DATA from game_data.py and HERO_DATA, _MATCHUP_TIPS from
knowledge_base.py, then writes editorial (coaching tips, descriptions, matchup advice,
power spikes, weaknesses, laning tips) into backend/data/editorial/*.json.

These editorial JSON files are the "Layer B" that is never overwritten by auto-updates.

Usage:
    python -m backend.scripts.migrate_to_json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import EDITORIAL_DIR


def _save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Written: {path} ({len(data)} entries)")


def migrate_abilities() -> None:
    """Extract coaching_tip and description from ABILITY_DATA."""
    from backend.app.ai.game_data import _FALLBACK_ABILITY_DATA

    editorial: dict[str, dict] = {}
    for hero_name, abilities in _FALLBACK_ABILITY_DATA.items():
        for ability_name, data in abilities.items():
            key = data.get("key", f"{hero_name}:{ability_name}")
            entry: dict = {}
            if "coaching_tip" in data:
                entry["coaching_tip"] = data["coaching_tip"]
            if "description" in data:
                entry["description"] = data["description"]
            if entry:
                entry["hero"] = hero_name
                entry["ability_name"] = ability_name
                editorial[key] = entry

    _save_json(EDITORIAL_DIR / "coaching_tips_abilities.json", editorial)


def migrate_items() -> None:
    """Extract coaching_tip and description from ITEM_DATA."""
    from backend.app.ai.game_data import _FALLBACK_ITEM_DATA

    editorial: dict[str, dict] = {}
    for item_name, data in _FALLBACK_ITEM_DATA.items():
        key = data.get("key", item_name)
        entry: dict = {}
        if "coaching_tip" in data:
            entry["coaching_tip"] = data["coaching_tip"]
        if "description" in data:
            entry["description"] = data["description"]
        if "is_active" in data:
            entry["is_active"] = data["is_active"]
        if "has_active" in data:
            entry["has_active"] = data["has_active"]
        if entry:
            entry["item_name"] = item_name
            editorial[key] = entry

    _save_json(EDITORIAL_DIR / "coaching_tips_items.json", editorial)


def migrate_heroes() -> None:
    """Extract role, power_spikes, weaknesses, laning_tips from HERO_DATA."""
    from backend.app.ai.knowledge_base import _FALLBACK_HERO_DATA

    editorial: dict[str, dict] = {}
    for hero_name, data in _FALLBACK_HERO_DATA.items():
        entry: dict = {}
        for field in ("role", "attribute", "attack_type", "power_spikes", "weaknesses", "laning_tips"):
            if field in data:
                entry[field] = data[field]
        if entry:
            editorial[hero_name] = entry

    _save_json(EDITORIAL_DIR / "hero_knowledge.json", editorial)


def migrate_matchups() -> None:
    """Extract matchup tips from _MATCHUP_TIPS."""
    from backend.app.ai.knowledge_base import _FALLBACK_MATCHUP_TIPS

    editorial: dict[str, str] = {}
    for hero_name, opponents in _FALLBACK_MATCHUP_TIPS.items():
        for opp_name, tip in opponents.items():
            key = f"{hero_name}:{opp_name}"
            editorial[key] = tip

    _save_json(EDITORIAL_DIR / "matchup_tips.json", editorial)


def main() -> None:
    print("Migrating hardcoded editorial content to JSON files...")
    print(f"Output directory: {EDITORIAL_DIR}")
    print()

    migrate_abilities()
    migrate_items()
    migrate_heroes()
    migrate_matchups()

    print()
    print("Migration complete! Editorial JSON files are ready.")
    print("These files will NOT be overwritten by auto-updates.")


if __name__ == "__main__":
    main()
