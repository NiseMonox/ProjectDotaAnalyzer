"""Game data manager: fetch from OpenDota API, merge with editorial content, cache in memory.

Three data layers:
- Layer A (Raw Stats): Auto-fetched from OpenDota /api/constants/* endpoints.
  Stored as JSON in backend/data/gamedata/*.json.
- Layer B (Editorial): Hand-written coaching tips, matchup advice, power spikes.
  Stored as JSON in backend/data/editorial/*.json. Never overwritten by auto-update.
- Layer C (Structural): Laning principles, vision rules, death analysis rules.
  Lives in knowledge_base.py (unchanged by this module).

Usage:
    from .data_manager import load_abilities, load_items, load_heroes, get_patch_version
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

from ..config import GAMEDATA_DIR, EDITORIAL_DIR

logger = logging.getLogger(__name__)

OPENDOTA_BASE = "https://api.opendota.com/api/constants"

# In-memory caches (loaded once from disk per process)
_ability_cache: dict[str, dict[str, dict]] | None = None
_item_cache: dict[str, dict] | None = None
_hero_cache: dict[str, dict] | None = None
_name_map: dict[str, str] | None = None  # internal_name -> localized_name
_patch_version: str | None = None


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _api_get(endpoint: str, timeout: int = 30) -> Any:
    """GET from OpenDota API, return parsed JSON."""
    url = f"{OPENDOTA_BASE}/{endpoint}"
    logger.info("Fetching %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "DotaAnalyzer/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _save_json(path: Path, data: Any) -> None:
    """Write JSON to file with pretty formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved %s", path)


def _load_json(path: Path) -> Any | None:
    """Load JSON from file, or None if missing."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Patch version
# ---------------------------------------------------------------------------

def check_patch_version() -> tuple[str, bool]:
    """Check remote patch version against local.

    Returns (remote_version, needs_update).
    """
    patches = _api_get("patch")
    if not patches:
        raise RuntimeError("Empty patch data from OpenDota")
    # patches is a list of {name, date}, sorted ascending
    remote_ver = patches[-1]["name"]

    meta = _load_json(GAMEDATA_DIR / "meta.json")
    local_ver = meta.get("patch_version", "") if meta else ""

    return remote_ver, remote_ver != local_ver


def get_patch_version() -> str:
    """Return current patch version from local meta, or fallback."""
    global _patch_version
    if _patch_version is not None:
        return _patch_version

    meta = _load_json(GAMEDATA_DIR / "meta.json")
    if meta:
        _patch_version = meta.get("patch_version", "7.37")
    else:
        _patch_version = "7.37"
    return _patch_version


# ---------------------------------------------------------------------------
# Fetch and save game data
# ---------------------------------------------------------------------------

def _build_name_map(heroes_raw: dict) -> dict[str, str]:
    """Build internal_name -> localized_name map from heroes API data."""
    name_map = {}
    for hero_id, hero in heroes_raw.items():
        internal = hero.get("name", "").replace("npc_dota_hero_", "")
        localized = hero.get("localized_name", internal)
        if internal:
            name_map[internal] = localized
    return name_map


def _to_num_list(val: Any) -> list:
    """Convert API value to a list of numbers. Handles str, list[str], list[int], etc."""
    if val is None:
        return []
    if isinstance(val, (int, float)):
        return [val]
    if isinstance(val, str):
        # Could be "50" or "50 / 60 / 70"
        parts = [p.strip() for p in val.replace("/", " ").split()]
        nums = []
        for p in parts:
            try:
                nums.append(float(p) if "." in p else int(p))
            except ValueError:
                pass
        return nums if nums else []
    if isinstance(val, list):
        result = []
        for item in val:
            if isinstance(item, (int, float)):
                result.append(item)
            elif isinstance(item, str):
                try:
                    result.append(float(item) if "." in item else int(item))
                except ValueError:
                    pass
        return result
    return []


def _normalize_ability(raw: dict, ability_key: str) -> dict:
    """Normalize OpenDota ability fields to our schema."""
    result: dict[str, Any] = {"key": ability_key}

    # Field mappings
    if "cd" in raw:
        result["cooldown"] = _to_num_list(raw["cd"])
    if "mc" in raw:
        result["mana_cost"] = _to_num_list(raw["mc"])
    if "dmg" in raw:
        result["damage"] = _to_num_list(raw["dmg"])

    if "bkbpierce" in raw:
        val = raw["bkbpierce"]
        if isinstance(val, str):
            result["pierces_bkb"] = val == "Yes"
        elif isinstance(val, list):
            result["pierces_bkb"] = "Yes" in val
    if "dmg_type" in raw and raw["dmg_type"]:
        dtype_map = {"Magical": "magical", "Physical": "physical", "Pure": "pure"}
        dmg_type_val = raw["dmg_type"]
        if isinstance(dmg_type_val, list):
            dmg_type_val = dmg_type_val[0] if dmg_type_val else ""
        if isinstance(dmg_type_val, str):
            result["damage_type"] = dtype_map.get(dmg_type_val, dmg_type_val.lower())

    # Pass through other useful fields
    for field in ("dname", "desc", "lore", "behavior", "attrib"):
        if field in raw:
            result[field] = raw[field]

    # Mark ultimates (typically abilities with 3-level cooldowns, but heuristic)
    if "isult" in raw:
        result["is_ultimate"] = bool(raw["isult"])

    return result


def _normalize_item(raw: dict, item_key: str) -> dict:
    """Normalize OpenDota item fields to our schema."""
    result: dict[str, Any] = {"key": item_key}

    if "cost" in raw:
        result["cost"] = raw["cost"]
    if "cd" in raw:
        result["cooldown"] = raw["cd"]
    if "mc" in raw:
        result["mana_cost"] = raw["mc"]
    if "dname" in raw:
        result["name"] = raw["dname"]
    if "lore" in raw:
        result["lore"] = raw["lore"]
    if "qual" in raw:
        result["quality"] = raw["qual"]
    if "desc" in raw:
        result["description"] = raw["desc"]
    if "attrib" in raw:
        result["attributes"] = raw["attrib"]
    if "behavior" in raw:
        result["is_active"] = "Active" in (raw.get("behavior", "") or "")

    return result


def fetch_and_save_gamedata(force: bool = False) -> dict:
    """Fetch all game data from OpenDota and save to disk.

    Returns a summary dict with counts and versions.
    """
    global _patch_version, _ability_cache, _item_cache, _hero_cache, _name_map

    # Check if update needed
    if not force:
        try:
            remote_ver, needs_update = check_patch_version()
            if not needs_update:
                return {
                    "status": "up_to_date",
                    "patch_version": remote_ver,
                }
        except Exception as e:
            logger.warning("Patch check failed, proceeding with force update: %s", e)
            remote_ver = "unknown"
    else:
        try:
            patches = _api_get("patch")
            remote_ver = patches[-1]["name"] if patches else "unknown"
        except Exception:
            remote_ver = "unknown"

    # 1. Fetch heroes
    heroes_raw = _api_get("heroes")
    name_map = _build_name_map(heroes_raw)
    heroes_normalized = {}
    for hero_id, hero in heroes_raw.items():
        internal = hero.get("name", "").replace("npc_dota_hero_", "")
        localized = hero.get("localized_name", internal)
        heroes_normalized[localized] = {
            "id": hero.get("id"),
            "internal_name": internal,
            "localized_name": localized,
            "primary_attr": hero.get("primary_attr", ""),
            "attack_type": hero.get("attack_type", "").lower(),
            "roles": hero.get("roles", []),
            "legs": hero.get("legs", 2),
        }
    _save_json(GAMEDATA_DIR / "heroes.json", heroes_normalized)

    # 2. Fetch hero_abilities (mapping: hero -> ability keys)
    hero_abilities_raw = _api_get("hero_abilities")

    # 3. Fetch abilities
    abilities_raw = _api_get("abilities")
    # Build hero->abilities mapping
    abilities_by_hero: dict[str, dict[str, dict]] = {}
    for hero_internal, ability_info in hero_abilities_raw.items():
        hero_internal_clean = hero_internal.replace("npc_dota_hero_", "")
        localized = name_map.get(hero_internal_clean, hero_internal_clean)
        hero_abilities_dict = {}
        ability_keys = ability_info.get("abilities", [])
        for ability_key in ability_keys:
            # OpenDota may nest sub-abilities as lists (e.g. Monkey King transform)
            if isinstance(ability_key, list):
                for sub_key in ability_key:
                    if isinstance(sub_key, str) and sub_key in abilities_raw:
                        raw_ability = abilities_raw[sub_key]
                        display_name = raw_ability.get("dname", sub_key)
                        if display_name:
                            hero_abilities_dict[display_name] = _normalize_ability(raw_ability, sub_key)
                continue
            if not isinstance(ability_key, str):
                continue
            if ability_key in abilities_raw:
                raw_ability = abilities_raw[ability_key]
                display_name = raw_ability.get("dname", ability_key)
                if not display_name:
                    continue
                normalized = _normalize_ability(raw_ability, ability_key)
                hero_abilities_dict[display_name] = normalized
        if hero_abilities_dict:
            abilities_by_hero[localized] = hero_abilities_dict
    _save_json(GAMEDATA_DIR / "abilities.json", abilities_by_hero)

    # 4. Fetch items
    items_raw = _api_get("items")
    items_normalized = {}
    for item_key, item_data in items_raw.items():
        display_name = item_data.get("dname", "")
        if not display_name:
            continue
        items_normalized[display_name] = _normalize_item(item_data, item_key)
    _save_json(GAMEDATA_DIR / "items.json", items_normalized)

    # 5. Save meta
    meta = {
        "patch_version": remote_ver,
        "last_fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hero_count": len(heroes_normalized),
        "ability_hero_count": len(abilities_by_hero),
        "item_count": len(items_normalized),
    }
    _save_json(GAMEDATA_DIR / "meta.json", meta)

    # Clear caches so next load picks up new data
    _ability_cache = None
    _item_cache = None
    _hero_cache = None
    _name_map = None
    _patch_version = None

    return {
        "status": "updated",
        "patch_version": remote_ver,
        "heroes": len(heroes_normalized),
        "abilities": len(abilities_by_hero),
        "items": len(items_normalized),
    }


# ---------------------------------------------------------------------------
# Load and merge (disk + editorial -> memory)
# ---------------------------------------------------------------------------

def load_abilities() -> dict[str, dict[str, dict]]:
    """Load abilities: merge gamedata/abilities.json with editorial coaching tips.

    Returns dict keyed by hero_name -> ability_name -> {stats + coaching_tip}.
    Falls back to hardcoded ABILITY_DATA if JSON files don't exist.
    """
    global _ability_cache
    if _ability_cache is not None:
        return _ability_cache

    gamedata = _load_json(GAMEDATA_DIR / "abilities.json")
    if gamedata is None:
        # Fallback to hardcoded data
        try:
            from .game_data import _FALLBACK_ABILITY_DATA
            _ability_cache = _FALLBACK_ABILITY_DATA
        except ImportError:
            _ability_cache = {}
        return _ability_cache

    # Merge with editorial coaching tips
    editorial = _load_json(EDITORIAL_DIR / "coaching_tips_abilities.json") or {}

    merged: dict[str, dict[str, dict]] = {}
    for hero_name, abilities in gamedata.items():
        merged_abilities = {}
        for ability_name, ability_data in abilities.items():
            merged_ability = dict(ability_data)
            # Look up editorial tip by ability key or hero:ability_name
            ability_key = ability_data.get("key", "")
            tip_entry = editorial.get(ability_key, {})
            if not tip_entry:
                tip_entry = editorial.get(f"{hero_name}:{ability_name}", {})
            if isinstance(tip_entry, dict):
                if "coaching_tip" in tip_entry:
                    merged_ability["coaching_tip"] = tip_entry["coaching_tip"]
                if "description" in tip_entry:
                    merged_ability["description"] = tip_entry["description"]
                if "heroes" in tip_entry:
                    merged_ability["heroes"] = tip_entry["heroes"]
            elif isinstance(tip_entry, str):
                merged_ability["coaching_tip"] = tip_entry
            merged_abilities[ability_name] = merged_ability
        merged[hero_name] = merged_abilities

    _ability_cache = merged
    logger.info("Loaded abilities for %d heroes (gamedata + editorial)", len(merged))
    return _ability_cache


def load_items() -> dict[str, dict]:
    """Load items: merge gamedata/items.json with editorial coaching tips.

    Returns dict keyed by item_display_name -> {stats + coaching_tip}.
    Falls back to hardcoded ITEM_DATA if JSON files don't exist.
    """
    global _item_cache
    if _item_cache is not None:
        return _item_cache

    gamedata = _load_json(GAMEDATA_DIR / "items.json")
    if gamedata is None:
        try:
            from .game_data import _FALLBACK_ITEM_DATA
            _item_cache = _FALLBACK_ITEM_DATA
        except ImportError:
            _item_cache = {}
        return _item_cache

    editorial = _load_json(EDITORIAL_DIR / "coaching_tips_items.json") or {}

    merged: dict[str, dict] = {}
    for item_name, item_data in gamedata.items():
        merged_item = dict(item_data)
        item_key = item_data.get("key", "")
        tip_entry = editorial.get(item_key, {}) or editorial.get(item_name, {})
        if isinstance(tip_entry, dict):
            if "coaching_tip" in tip_entry:
                merged_item["coaching_tip"] = tip_entry["coaching_tip"]
            if "description" in tip_entry:
                merged_item["description"] = tip_entry["description"]
            if "has_active" in tip_entry:
                merged_item["has_active"] = tip_entry["has_active"]
            if "is_active" in tip_entry:
                merged_item["is_active"] = tip_entry["is_active"]
            if "heroes" in tip_entry:
                merged_item["heroes"] = tip_entry["heroes"]
        elif isinstance(tip_entry, str):
            merged_item["coaching_tip"] = tip_entry
        merged[item_name] = merged_item

    _item_cache = merged
    logger.info("Loaded %d items (gamedata + editorial)", len(merged))
    return _item_cache


def load_heroes() -> dict[str, dict]:
    """Load hero data: merge gamedata/heroes.json with editorial hero knowledge.

    Returns dict keyed by hero_name -> {role, attribute, attack_type, power_spikes, ...}.
    Falls back to hardcoded HERO_DATA if JSON files don't exist.
    """
    global _hero_cache
    if _hero_cache is not None:
        return _hero_cache

    gamedata = _load_json(GAMEDATA_DIR / "heroes.json")
    if gamedata is None:
        try:
            from .knowledge_base import _FALLBACK_HERO_DATA
            _hero_cache = _FALLBACK_HERO_DATA
        except (ImportError, AttributeError):
            _hero_cache = {}
        return _hero_cache

    editorial = _load_json(EDITORIAL_DIR / "hero_knowledge.json") or {}

    # Map OpenDota roles to our role system
    def _infer_role(roles: list[str]) -> str:
        if not roles:
            return "unknown"
        roles_lower = [r.lower() for r in roles]
        if "carry" in roles_lower:
            return "carry"
        if "nuker" in roles_lower and "mid" in " ".join(roles_lower):
            return "mid"
        if "initiator" in roles_lower or "durable" in roles_lower:
            return "offlane"
        if "support" in roles_lower:
            return "support_5"
        return "unknown"

    # Map primary_attr
    attr_map = {"agi": "agi", "str": "str", "int": "int", "all": "uni"}

    merged: dict[str, dict] = {}
    for hero_name, hero_data in gamedata.items():
        editorial_entry = editorial.get(hero_name, {})
        merged_hero = {
            "role": editorial_entry.get("role", _infer_role(hero_data.get("roles", []))),
            "attribute": attr_map.get(hero_data.get("primary_attr", ""), hero_data.get("primary_attr", "unknown")),
            "attack_type": hero_data.get("attack_type", "unknown"),
            "power_spikes": editorial_entry.get("power_spikes", []),
            "weaknesses": editorial_entry.get("weaknesses", []),
            "laning_tips": editorial_entry.get("laning_tips", ""),
            "internal_name": hero_data.get("internal_name", ""),
        }
        # Carry forward editorial-only fields
        for key in ("level1_strength", "level1_note", "heroes"):
            if key in editorial_entry:
                merged_hero[key] = editorial_entry[key]
        merged[hero_name] = merged_hero

    _hero_cache = merged
    logger.info("Loaded %d heroes (gamedata + editorial)", len(merged))
    return _hero_cache


def load_matchup_tips() -> dict[str, dict[str, str]]:
    """Load matchup tips from editorial data.

    Returns dict keyed by hero_name -> opponent_name -> tip_text.
    Falls back to hardcoded _MATCHUP_TIPS if JSON file doesn't exist.
    """
    editorial = _load_json(EDITORIAL_DIR / "matchup_tips.json")
    if editorial is None:
        try:
            from .knowledge_base import _FALLBACK_MATCHUP_TIPS
            return _FALLBACK_MATCHUP_TIPS
        except (ImportError, AttributeError):
            return {}

    # Convert from "hero_a:hero_b" keying to nested dict
    # Supports both old format (flat string tip) and new format (object with "tip" field)
    result: dict[str, dict[str, str]] = {}
    for key, value in editorial.items():
        if ":" in key:
            hero_a, hero_b = key.split(":", 1)
            if isinstance(value, dict):
                tip = value.get("tip", "")
            else:
                tip = value
            result.setdefault(hero_a, {})[hero_b] = tip
    return result


def get_meta() -> dict:
    """Return metadata about the current game data state."""
    meta = _load_json(GAMEDATA_DIR / "meta.json")
    if meta is None:
        return {
            "status": "no_data",
            "patch_version": "7.37",
            "source": "hardcoded_fallback",
        }
    meta["source"] = "opendota_api"
    # Check staleness
    last_fetched = meta.get("last_fetched", "")
    if last_fetched:
        try:
            fetched_ts = time.mktime(time.strptime(last_fetched, "%Y-%m-%dT%H:%M:%SZ"))
            age_days = (time.time() - fetched_ts) / 86400
            meta["age_days"] = round(age_days, 1)
            meta["is_stale"] = age_days > 14
        except (ValueError, OverflowError):
            meta["is_stale"] = True
    return meta


def clear_caches() -> None:
    """Clear all in-memory caches. Forces reload from disk on next access."""
    global _ability_cache, _item_cache, _hero_cache, _name_map, _patch_version
    _ability_cache = None
    _item_cache = None
    _hero_cache = None
    _name_map = None
    _patch_version = None
