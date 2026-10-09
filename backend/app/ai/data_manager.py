"""Game data manager: layered hero / ability / item data with editorial knowledge.

Layers (merged in this order, later layers win for the fields they provide):

- Layer A (raw stats): auto-fetched from OpenDota ``/api/constants/*`` into
  ``backend/data/gamedata/*.json`` (gitignored, refreshed at runtime). Until a
  fetch succeeds, a minimal numeric fallback (``backend/data/fallback/*.json``)
  and the committed frontend catalogs (``frontend/static/data/heroes.json`` /
  ``items.json``) stand in for it.
- Layer B (editorial): hand-written coaching knowledge in
  ``backend/data/editorial/*.json`` — the canonical source of hero, matchup,
  ability and item tips. Always merged, with or without Layer A, and never
  overwritten by auto-updates.
- Layer C (structural principles): laning / vision / death-analysis rules in
  ``knowledge_base.py``.

Hero and item identifiers are normalised here once (display names such as
"Queen of Pain", heroes.json slugs such as "queenofpain", npc names such as
"npc_dota_hero_queenofpain", item keys such as "bfury" or "item_bfury"):
see :func:`resolve_hero_name`, :func:`hero_slug`, :func:`resolve_item_name`
and :func:`item_key`.

Every loader returns a shared, cached object that callers must treat as
read-only. :func:`invalidate_all_caches` atomically drops all caches (this
module, ``game_data`` and the RAG stores in ``knowledge_base``); it runs
automatically after :func:`fetch_and_save_gamedata` writes new data.
"""
from __future__ import annotations

import calendar
import json
import logging
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .. import config

logger = logging.getLogger(__name__)

OPENDOTA_BASE = "https://api.opendota.com/api/constants"

# Patch the builtin / editorial knowledge was written against. Used whenever
# the live patch is unknown (no successful game-data fetch yet). This is the
# ONLY place a patch string may be hardcoded — everything else must call
# get_current_patch().
DEFAULT_PATCH = "7.37"

# Data directories (module attributes so tests can monkeypatch them).
GAMEDATA_DIR: Path = config.GAMEDATA_DIR
EDITORIAL_DIR: Path = config.EDITORIAL_DIR
FALLBACK_DIR: Path = config.DATA_DIR / "fallback"
STATIC_DATA_DIR: Path = config.STATIC_DIR / "data"

# Display-name renames that the squashed-name lookup cannot infer by itself.
_HERO_ALIASES: dict[str, str] = {
    "outworlddestroyer": "obsidian_destroyer",
    "outworlddevourer": "obsidian_destroyer",
}

# OpenDota primary_attr -> our attribute code
_ATTR_MAP = {"agi": "agi", "str": "str", "int": "int", "all": "uni", "uni": "uni"}


# ---------------------------------------------------------------------------
# Cache (atomic snapshot + generation counter)
# ---------------------------------------------------------------------------

_cache_lock = threading.RLock()
_cache: dict[str, Any] = {}
_generation = 0
_fetch_lock = threading.Lock()


def data_generation() -> int:
    """Monotonic counter bumped by every :func:`invalidate_all_caches` call.

    Downstream caches (game_data lookups, RAG stores) compare it to know
    when to rebuild.
    """
    return _generation


def _cached(name: str, builder: Callable[[], Any]) -> Any:
    """Return the cached value for *name*, building it once under the lock.

    Builders run while holding the (re-entrant) lock, so concurrent first
    calls never build twice and readers never observe half-built data.
    """
    value = _cache.get(name)
    if value is not None:
        return value
    with _cache_lock:
        value = _cache.get(name)
        if value is None:
            value = builder()
            _cache[name] = value
        return value


def invalidate_all_caches() -> None:
    """Drop every game-data cache so the next access reloads from disk.

    Thread-safe: the cache snapshot is swapped atomically. Also resets the
    dependent caches in ``game_data`` and the per-language RAG stores in
    ``knowledge_base`` (both additionally track :func:`data_generation`).
    """
    global _cache, _generation
    with _cache_lock:
        _cache = {}
        _generation += 1
    try:
        from .game_data import invalidate_cache as _invalidate_game_data
        _invalidate_game_data()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to invalidate game_data cache: %s", exc)
    try:
        from .knowledge_base import reset_knowledge_stores
        reset_knowledge_stores()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to reset knowledge stores: %s", exc)
    logger.info("Game data caches invalidated (generation %d)", _generation)


def clear_caches() -> None:
    """Backward-compatible alias of :func:`invalidate_all_caches`."""
    invalidate_all_caches()


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _api_get(endpoint: str, timeout: int = 30) -> Any:
    """GET from the OpenDota constants API, return parsed JSON."""
    url = f"{OPENDOTA_BASE}/{endpoint}"
    logger.info("Fetching %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "DotaAnalyzer/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _save_json(path: Path, data: Any) -> None:
    """Atomically write JSON (temp file + rename) so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    logger.info("Saved %s", path)


def _load_json(path: Path) -> Any | None:
    """Load JSON from *path*; None when missing or unreadable (logged)."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return None


def _load_dict(path: Path, key: str | None = None) -> dict:
    """Load a JSON object (optionally a nested object under *key*); {} otherwise."""
    data = _load_json(path)
    if key is not None and isinstance(data, dict):
        data = data.get(key)
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Name normalisation (heroes + items)
# ---------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _squash_hero(name: Any) -> str:
    """'Queen of Pain' / 'queenofpain' / 'npc_dota_hero_queenofpain' -> 'queenofpain'."""
    text = str(name or "").strip().lower()
    if text.startswith("npc_dota_hero_"):
        text = text[len("npc_dota_hero_"):]
    return _NON_ALNUM.sub("", text)


def _squash_item(name: Any) -> str:
    """'Battle Fury' / 'item_bfury' / 'battle_fury' -> 'battlefury' / 'bfury'."""
    text = str(name or "").strip().lower()
    if text.startswith("item_"):
        text = text[len("item_"):]
    return _NON_ALNUM.sub("", text)


def display_slug(name: Any) -> str:
    """Generic slug of a display name: lowercase, non-alphanumerics -> "_".

    'Anti-Mage' -> 'anti_mage', "Nature's Prophet" -> 'nature_s_prophet'.
    Used for identifiers that are not hero/item tags (e.g. topic slugs).
    """
    return _NON_ALNUM.sub("_", str(name or "").lower()).strip("_")


def _build_hero_index() -> dict:
    """Merge every hero catalog we have into one alias index.

    Returns ``{"by_internal": {internal: entry}, "alias": {squashed: internal}}``
    where entry is ``{"id", "name", "internal", "roles", "attack_type", "attr"}``.
    Display names come from the committed frontend catalog first (what the
    parsers emit), then fetched game data, then editorial data.
    """
    by_internal: dict[str, dict] = {}
    alias: dict[str, str] = {}

    def add(internal: str, name: str, **fields: Any) -> None:
        internal = str(internal or "").strip().lower()
        if internal.startswith("npc_dota_hero_"):
            internal = internal[len("npc_dota_hero_"):]
        if not internal:
            return
        entry = by_internal.get(internal)
        if entry is None:
            entry = {"id": None, "name": name or internal, "internal": internal,
                     "roles": [], "attack_type": "", "attr": ""}
            by_internal[internal] = entry
        for key, value in fields.items():
            if value and not entry.get(key):
                entry[key] = value
        for key in (internal, name):
            squashed = _squash_hero(key)
            if squashed:
                alias.setdefault(squashed, internal)

    static = _load_dict(STATIC_DATA_DIR / "heroes.json", "by_id")
    for hid, info in static.items():
        if not isinstance(info, dict):
            continue
        try:
            hero_id = int(info.get("id") or hid)
        except (TypeError, ValueError):
            hero_id = None
        add(info.get("slug", ""), info.get("name", ""), id=hero_id,
            roles=list(info.get("roles") or []),
            attack_type=str(info.get("attack_type") or "").lower(),
            attr=_ATTR_MAP.get(str(info.get("attr") or ""), str(info.get("attr") or "")))

    gamedata = _load_dict(GAMEDATA_DIR / "heroes.json")
    for localized, info in gamedata.items():
        if not isinstance(info, dict):
            continue
        internal = info.get("internal_name") or _squash_hero(localized)
        add(internal, localized, id=info.get("id"),
            roles=list(info.get("roles") or []),
            attack_type=str(info.get("attack_type") or "").lower(),
            attr=_ATTR_MAP.get(str(info.get("primary_attr") or ""), ""))

    for squashed, internal in _HERO_ALIASES.items():
        if internal in by_internal:
            alias.setdefault(squashed, internal)

    # Editorial-only heroes (not in any catalog) still get an entry.
    editorial = _load_dict(EDITORIAL_DIR / "hero_knowledge.json")
    for name in editorial:
        squashed = _squash_hero(name)
        if squashed and squashed not in alias:
            add(squashed, name)

    return {"by_internal": by_internal, "alias": alias}


def _hero_index() -> dict:
    return _cached("hero_index", _build_hero_index)


def _hero_entry(name: Any) -> dict | None:
    squashed = _squash_hero(name)
    if not squashed:
        return None
    index = _hero_index()
    internal = index["alias"].get(squashed)
    return index["by_internal"].get(internal) if internal else None


def resolve_hero_name(name: Any) -> str | None:
    """Canonical display name for any hero identifier (None if unknown)."""
    entry = _hero_entry(name)
    return entry["name"] if entry else None


def hero_slug(name: Any) -> str:
    """Canonical hero tag slug (internal name, e.g. "queenofpain", "furion").

    Unknown names fall back to their squashed form so producers and
    consumers still agree. ``"all"`` is preserved for universal chunks.
    """
    squashed = _squash_hero(name)
    if squashed == "all":
        return "all"
    entry = _hero_entry(name)
    return entry["internal"] if entry else squashed


def get_hero_catalog() -> list[dict]:
    """Every known hero with an id: ``[{"id", "name", "slug"}]`` sorted by id.

    Works without fetched game data (the frontend catalog is committed), so
    learners never lose their hero list to a concurrent / failed fetch.
    """
    def build() -> list[dict]:
        out = [
            {"id": int(e["id"]), "name": e["name"], "slug": e["internal"]}
            for e in _hero_index()["by_internal"].values()
            if isinstance(e.get("id"), int) and e["id"] > 0
        ]
        out.sort(key=lambda e: e["id"])
        return out
    return _cached("hero_catalog", build)


def _build_item_index() -> dict:
    """Alias index for items: ``{"by_key": {key: name}, "alias": {squashed: key}}``."""
    by_key: dict[str, str] = {}
    alias: dict[str, str] = {}

    def add(key: str, name: str) -> None:
        key = str(key or "").strip().lower()
        if key.startswith("item_"):
            key = key[len("item_"):]
        if not key:
            return
        if key not in by_key or not by_key[key]:
            by_key[key] = name or by_key.get(key, "")
        alias.setdefault(_squash_item(key), key)

    def add_name(name: str, key: str) -> None:
        squashed = _squash_item(name)
        if squashed and not str(name).lower().startswith("recipe"):
            alias.setdefault(squashed, key)

    static = _load_json(STATIC_DATA_DIR / "items.json")
    static = static if isinstance(static, dict) else {}
    by_key_static = static.get("by_key") if isinstance(static.get("by_key"), dict) else {}
    for key, info in by_key_static.items():
        if isinstance(info, dict):
            add(key, str(info.get("name") or ""))
    # Keys first (exact), then preferred display-name mapping, then other names.
    for name, key in (static.get("name_to_key") or {}).items():
        if isinstance(key, str):
            add_name(name, key)
    for key, info in by_key_static.items():
        if isinstance(info, dict) and info.get("name"):
            add_name(info["name"], key)

    for name, info in _load_dict(GAMEDATA_DIR / "items.json").items():
        if isinstance(info, dict) and info.get("key"):
            add(info["key"], name)
            add_name(name, str(info["key"]).lower())
    for data in (_load_dict(FALLBACK_DIR / "item_stats.json", "items"),
                 _load_dict(EDITORIAL_DIR / "coaching_tips_items.json")):
        for key, info in data.items():
            if isinstance(info, dict):
                add(key, str(info.get("item_name") or ""))
                add_name(str(info.get("item_name") or ""), key)
    return {"by_key": by_key, "alias": alias}


def _item_index() -> dict:
    return _cached("item_index", _build_item_index)


def resolve_item_key(name: Any) -> str | None:
    """Internal item key for a display name / key / ``item_`` name (None if unknown)."""
    squashed = _squash_item(name)
    if not squashed:
        return None
    return _item_index()["alias"].get(squashed)


def item_key(name: Any) -> str:
    """Canonical item tag slug (internal key such as "bfury"; squashed fallback)."""
    return resolve_item_key(name) or _squash_item(name)


def resolve_item_name(name: Any) -> str | None:
    """Canonical display name for an item identifier (None if unknown)."""
    key = resolve_item_key(name)
    if not key:
        return None
    return _item_index()["by_key"].get(key) or None


# ---------------------------------------------------------------------------
# Patch version
# ---------------------------------------------------------------------------

def _valid_patch(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in ("", "unknown", "none", "null") else text


def get_fetched_patch() -> str:
    """Patch recorded by the last successful game-data fetch ("" if unknown)."""
    def build() -> str:
        meta = _load_json(GAMEDATA_DIR / "meta.json")
        return _valid_patch(meta.get("patch_version")) if isinstance(meta, dict) else ""
    # Cache "" too (store as a tuple so the falsy value is not rebuilt).
    return _cached("fetched_patch", lambda: (build(),))[0]


def get_current_patch() -> str:
    """Current patch: the fetched meta patch, else :data:`DEFAULT_PATCH`."""
    return get_fetched_patch() or DEFAULT_PATCH


def get_patch_version() -> str:
    """Backward-compatible alias of :func:`get_current_patch`."""
    return get_current_patch()


def check_patch_version() -> tuple[str, bool]:
    """Check the remote patch version against the local one.

    Returns ``(remote_version, needs_update)``. Raises on network errors.
    """
    patches = _api_get("patch")
    if not isinstance(patches, list) or not patches:
        raise RuntimeError("Empty patch data from OpenDota")
    remote_ver = _valid_patch((patches[-1] or {}).get("name"))
    if not remote_ver:
        raise RuntimeError("Malformed patch data from OpenDota")
    return remote_ver, remote_ver != get_fetched_patch()


# ---------------------------------------------------------------------------
# Fetch and save game data (Layer A)
# ---------------------------------------------------------------------------

def _to_num_list(val: Any) -> list:
    """Convert API value to a list of numbers. Handles str, list[str], list[int], etc."""
    if val is None or isinstance(val, bool):
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
        return nums
    if isinstance(val, list):
        result = []
        for item in val:
            if isinstance(item, bool):
                continue
            if isinstance(item, (int, float)):
                result.append(item)
            elif isinstance(item, str):
                result.extend(_to_num_list(item))
        return result
    return []


def _behavior_text(raw: Any) -> str:
    if isinstance(raw, list):
        return " ".join(str(x) for x in raw)
    return str(raw or "")


def _normalize_ability(raw: dict, ability_key: str) -> dict:
    """Normalize OpenDota ability fields to our schema."""
    result: dict[str, Any] = {"key": ability_key}

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
    if raw.get("dmg_type"):
        dtype_map = {"Magical": "magical", "Physical": "physical", "Pure": "pure"}
        dmg_type_val = raw["dmg_type"]
        if isinstance(dmg_type_val, list):
            dmg_type_val = dmg_type_val[0] if dmg_type_val else ""
        if isinstance(dmg_type_val, str):
            result["damage_type"] = dtype_map.get(dmg_type_val, dmg_type_val.lower())

    behavior = _behavior_text(raw.get("behavior"))
    if behavior:
        if "Passive" in behavior:
            result["type"] = "passive"
        elif "Autocast" in behavior:
            result["type"] = "autocast"
        else:
            result["type"] = "active"

    for field in ("dname", "desc", "behavior", "attrib"):
        if field in raw:
            result[field] = raw[field]
    if "isult" in raw:
        result["is_ultimate"] = bool(raw["isult"])
    return result


def _normalize_item(raw: dict, item_key_: str) -> dict:
    """Normalize OpenDota item fields to our schema."""
    result: dict[str, Any] = {"key": item_key_}
    if "cost" in raw:
        result["cost"] = raw["cost"]
    if "cd" in raw and raw["cd"] not in (False, None):
        result["cooldown"] = raw["cd"]
    if "mc" in raw and raw["mc"] not in (False, None):
        result["mana_cost"] = raw["mc"]
    if raw.get("dname"):
        result["name"] = raw["dname"]
    if "qual" in raw:
        result["quality"] = raw["qual"]
    if "desc" in raw:
        result["description"] = raw["desc"]
    if "attrib" in raw:
        result["attributes"] = raw["attrib"]
    item_abilities = raw.get("abilities")
    if isinstance(item_abilities, list) and item_abilities:
        result["is_active"] = any(
            isinstance(a, dict) and a.get("type") in ("active", "toggle") for a in item_abilities
        )
    elif "behavior" in raw:
        behavior = _behavior_text(raw.get("behavior"))
        result["is_active"] = bool(behavior) and "Passive" not in behavior
    return result


def fetch_and_save_gamedata(force: bool = False) -> dict:
    """Fetch all game data from OpenDota and save it to disk.

    Everything is downloaded and validated before any file is written, and
    each file is replaced atomically, so a failed fetch never leaves a mix
    of old and new data behind. All caches are invalidated afterwards.
    Returns a summary dict; raises on network / payload errors.
    """
    with _fetch_lock:
        remote_ver = ""
        try:
            remote_ver, needs_update = check_patch_version()
            if not force and not needs_update:
                return {"status": "up_to_date", "patch_version": remote_ver}
        except Exception as exc:
            logger.warning("Patch check failed, proceeding with a full update: %s", exc)

        heroes_raw = _api_get("heroes")
        hero_abilities_raw = _api_get("hero_abilities")
        abilities_raw = _api_get("abilities")
        items_raw = _api_get("items")
        for label, payload in (("heroes", heroes_raw), ("hero_abilities", hero_abilities_raw),
                               ("abilities", abilities_raw), ("items", items_raw)):
            if not isinstance(payload, dict) or not payload:
                raise RuntimeError(f"Unexpected OpenDota '{label}' payload")

        # 1. Heroes
        name_map: dict[str, str] = {}  # internal_name -> localized_name
        heroes_normalized: dict[str, dict] = {}
        for hero in heroes_raw.values():
            if not isinstance(hero, dict):
                continue
            internal = str(hero.get("name", "")).replace("npc_dota_hero_", "")
            localized = hero.get("localized_name") or internal
            if not internal:
                continue
            name_map[internal] = localized
            heroes_normalized[localized] = {
                "id": hero.get("id"),
                "internal_name": internal,
                "localized_name": localized,
                "primary_attr": hero.get("primary_attr", ""),
                "attack_type": str(hero.get("attack_type", "")).lower(),
                "roles": hero.get("roles", []),
                "legs": hero.get("legs", 2),
            }

        # 2. Abilities grouped per hero, in the hero's ability order
        abilities_by_hero: dict[str, dict[str, dict]] = {}
        for hero_internal, ability_info in hero_abilities_raw.items():
            if not isinstance(ability_info, dict):
                continue
            hero_internal_clean = str(hero_internal).replace("npc_dota_hero_", "")
            localized = name_map.get(hero_internal_clean, hero_internal_clean)
            hero_abilities_dict: dict[str, dict] = {}
            keys: list[str] = []
            for ability_key in ability_info.get("abilities", []):
                # OpenDota may nest sub-abilities as lists (e.g. Monkey King)
                if isinstance(ability_key, list):
                    keys.extend(k for k in ability_key if isinstance(k, str))
                elif isinstance(ability_key, str):
                    keys.append(ability_key)
            for ability_key in keys:
                raw_ability = abilities_raw.get(ability_key)
                if not isinstance(raw_ability, dict):
                    continue
                display_name = raw_ability.get("dname")
                if not display_name:
                    continue
                hero_abilities_dict[display_name] = _normalize_ability(raw_ability, ability_key)
            if hero_abilities_dict:
                abilities_by_hero[localized] = hero_abilities_dict

        # 3. Items
        items_normalized: dict[str, dict] = {}
        for key, item_data in items_raw.items():
            if not isinstance(item_data, dict):
                continue
            display_name = item_data.get("dname", "")
            if not display_name or str(key).startswith("recipe_"):
                continue
            items_normalized[display_name] = _normalize_item(item_data, key)

        if not heroes_normalized or not abilities_by_hero or not items_normalized:
            raise RuntimeError("OpenDota constants payload was empty after normalisation")

        # Keep the previous patch label when the remote one is unknown.
        patch = remote_ver or get_fetched_patch()
        meta = {
            "patch_version": patch,
            "last_fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hero_count": len(heroes_normalized),
            "ability_hero_count": len(abilities_by_hero),
            "item_count": len(items_normalized),
        }
        _save_json(GAMEDATA_DIR / "heroes.json", heroes_normalized)
        _save_json(GAMEDATA_DIR / "abilities.json", abilities_by_hero)
        _save_json(GAMEDATA_DIR / "items.json", items_normalized)
        _save_json(GAMEDATA_DIR / "meta.json", meta)  # last: marks the set complete

    # Data on disk changed: every derived cache (incl. RAG stores) is stale.
    invalidate_all_caches()
    return {
        "status": "updated",
        "patch_version": patch or DEFAULT_PATCH,
        "heroes": len(heroes_normalized),
        "abilities": len(abilities_by_hero),
        "items": len(items_normalized),
    }


# ---------------------------------------------------------------------------
# Load and merge (Layer A + fallback + Layer B -> memory)
# ---------------------------------------------------------------------------

def _infer_role(roles: list[str]) -> str:
    """Map OpenDota role labels to our role system (rough, editorial wins)."""
    roles_lower = [str(r).lower() for r in roles or []]
    if not roles_lower:
        return "unknown"
    if "carry" in roles_lower:
        return "carry"
    if "support" in roles_lower:
        return "support_5"
    if "initiator" in roles_lower or "durable" in roles_lower:
        return "offlane"
    if "nuker" in roles_lower:
        return "mid"
    return "unknown"


def _build_heroes() -> dict[str, dict]:
    editorial = _load_dict(EDITORIAL_DIR / "hero_knowledge.json")
    by_display: dict[str, dict] = {}
    for entry in _hero_index()["by_internal"].values():
        by_display[entry["name"]] = {
            "role": _infer_role(entry.get("roles", [])),
            "attribute": entry.get("attr") or "unknown",
            "attack_type": entry.get("attack_type") or "unknown",
            "power_spikes": [],
            "weaknesses": [],
            "laning_tips": "",
            "internal_name": entry["internal"],
            "id": entry.get("id"),
            "roles": list(entry.get("roles", [])),
        }
    for name, info in editorial.items():
        if not isinstance(info, dict):
            continue
        display = resolve_hero_name(name) or name
        hero = by_display.setdefault(display, {
            "role": "unknown", "attribute": "unknown", "attack_type": "unknown",
            "power_spikes": [], "weaknesses": [], "laning_tips": "",
            "internal_name": hero_slug(name), "id": None, "roles": [],
        })
        for field in ("role", "attribute", "attack_type", "laning_tips",
                      "level1_strength", "level1_note"):
            if info.get(field):
                hero[field] = info[field]
        for field in ("power_spikes", "weaknesses"):
            if isinstance(info.get(field), list):
                hero[field] = [str(x) for x in info[field] if x]
        hero["has_editorial"] = True
    logger.info("Loaded %d heroes (%d with editorial knowledge)",
                len(by_display), sum(1 for h in by_display.values() if h.get("has_editorial")))
    return by_display


def load_heroes() -> dict[str, dict]:
    """Hero data keyed by display name (catalog + fetched data + editorial).

    Fields: role, attribute, attack_type, power_spikes, weaknesses,
    laning_tips, level1_strength, level1_note, internal_name, id, roles.
    Always contains editorial knowledge for every editorial hero, whether or
    not game data has been fetched.
    """
    return _cached("heroes", _build_heroes)


def _mark_ultimates(abilities: dict[str, dict], editorial_order: bool = False) -> None:
    """Flag the hero's ultimate when no source flagged it explicitly.

    Heuristic: the last ability with 3 levels of cooldown / mana cost;
    for editorial-only heroes (no numbers) the last listed ability.
    """
    if not abilities or any(a.get("is_ultimate") for a in abilities.values()):
        return
    candidates = [
        name for name, a in abilities.items()
        if len(a.get("cooldown") or []) == 3 or len(a.get("mana_cost") or []) == 3
    ]
    if candidates:
        abilities[candidates[-1]]["is_ultimate"] = True
    elif editorial_order:
        abilities[list(abilities)[-1]]["is_ultimate"] = True


def _build_abilities() -> dict[str, dict[str, dict]]:
    gamedata = _load_dict(GAMEDATA_DIR / "abilities.json")
    fallback = _load_dict(FALLBACK_DIR / "ability_stats.json", "abilities")
    editorial = _load_dict(EDITORIAL_DIR / "coaching_tips_abilities.json")

    merged: dict[str, dict[str, dict]] = {}
    heroes_with_stats: set[str] = set()
    for hero_name, abilities in gamedata.items():
        if not isinstance(abilities, dict):
            continue
        display = resolve_hero_name(hero_name) or hero_name
        merged[display] = {name: dict(data) for name, data in abilities.items()
                           if isinstance(data, dict)}
        heroes_with_stats.add(display)

    def find(display: str, key: str, ability_name: str) -> dict | None:
        abilities = merged.get(display) or {}
        for data in abilities.values():
            if data.get("key") == key:
                return data
        squashed = _NON_ALNUM.sub("", str(ability_name).lower())
        for name, data in abilities.items():
            if _NON_ALNUM.sub("", name.lower()) == squashed:
                return data
        return None

    # Numeric fallback: fills gaps in fetched data; stands in when not fetched.
    fallback_heroes: set[str] = set()
    for key, info in fallback.items():
        if not isinstance(info, dict) or not info.get("ability_name"):
            continue
        display = resolve_hero_name(info.get("hero")) or str(info.get("hero"))
        existing = find(display, key, info["ability_name"])
        stats = {k: v for k, v in info.items() if k not in ("hero", "ability_name")}
        if existing is not None:
            for field, value in stats.items():
                existing.setdefault(field, value)
        elif display not in heroes_with_stats:
            merged.setdefault(display, {})[info["ability_name"]] = {"key": key, **stats}
            fallback_heroes.add(display)

    # Editorial coaching text (canonical) — always merged.
    editorial_only: set[str] = set()
    for key, info in editorial.items():
        if not isinstance(info, dict):
            continue
        ability_name = info.get("ability_name") or key
        display = resolve_hero_name(info.get("hero")) or str(info.get("hero") or "")
        if not display:
            continue
        existing = find(display, key, ability_name)
        if existing is None:
            if display in heroes_with_stats:
                logger.debug("Editorial ability %s not in fetched data for %s, skipped", key, display)
                continue
            existing = merged.setdefault(display, {}).setdefault(ability_name, {"key": key})
            if display not in fallback_heroes:
                editorial_only.add(display)
        if info.get("coaching_tip"):
            existing["coaching_tip"] = info["coaching_tip"]
        if info.get("description"):
            existing["description"] = info["description"]

    for display, abilities in merged.items():
        _mark_ultimates(abilities, editorial_order=display in editorial_only)
    logger.info("Loaded abilities for %d heroes (%d with fetched stats)",
                len(merged), len(heroes_with_stats))
    return merged


def load_abilities() -> dict[str, dict[str, dict]]:
    """Abilities keyed by hero display name -> ability display name -> data.

    Data merges fetched stats (or the numeric fallback) with the editorial
    ``description`` / ``coaching_tip``; editorial text is merged even when
    no game data has been fetched.
    """
    return _cached("abilities", _build_abilities)


def _build_items() -> dict[str, dict]:
    gamedata = _load_dict(GAMEDATA_DIR / "items.json")
    fallback = _load_dict(FALLBACK_DIR / "item_stats.json", "items")
    editorial = _load_dict(EDITORIAL_DIR / "coaching_tips_items.json")

    merged: dict[str, dict] = {}
    by_key: dict[str, dict] = {}
    for name, data in gamedata.items():
        if not isinstance(data, dict):
            continue
        entry = dict(data)
        entry.setdefault("name", name)
        merged[name] = entry
        if entry.get("key"):
            by_key[str(entry["key"])] = entry

    def ensure(key: str, fallback_name: str) -> dict:
        entry = by_key.get(key)
        if entry is None:
            name = resolve_item_name(key) or fallback_name or key
            entry = merged.get(name)
            if entry is None:
                entry = {"key": key, "name": name}
                merged[name] = entry
            by_key[key] = entry
        return entry

    for key, info in fallback.items():
        if not isinstance(info, dict):
            continue
        entry = ensure(key, str(info.get("item_name") or ""))
        for field, value in info.items():
            if field != "item_name":
                entry.setdefault(field, value)

    for key, info in editorial.items():
        if not isinstance(info, dict):
            continue
        entry = ensure(key, str(info.get("item_name") or ""))
        for field in ("coaching_tip", "description", "heroes", "counter_tips"):
            if info.get(field):
                entry[field] = info[field]
        if isinstance(info.get("is_active"), bool):
            entry["is_active"] = info["is_active"]
    logger.info("Loaded %d items (%d with fetched stats)", len(merged), len(gamedata))
    return merged


def load_items() -> dict[str, dict]:
    """Items keyed by display name: fetched stats (or fallback) + editorial tips."""
    return _cached("items", _build_items)


def _build_matchups() -> dict[str, dict[str, str]]:
    editorial = _load_dict(EDITORIAL_DIR / "matchup_tips.json")
    result: dict[str, dict[str, str]] = {}
    for key, value in editorial.items():
        if ":" not in key:
            continue
        hero_a, hero_b = key.split(":", 1)
        tip = value.get("tip", "") if isinstance(value, dict) else value
        if not isinstance(tip, str) or not tip.strip():
            continue
        a = resolve_hero_name(hero_a) or hero_a.strip()
        b = resolve_hero_name(hero_b) or hero_b.strip()
        result.setdefault(a, {})[b] = tip.strip()
    return result


def load_matchup_tips() -> dict[str, dict[str, str]]:
    """Editorial matchup tips: ``{hero_display: {opponent_display: tip}}``."""
    return _cached("matchups", _build_matchups)


def get_meta() -> dict:
    """Return metadata about the current game data state (a fresh dict)."""
    meta = _load_json(GAMEDATA_DIR / "meta.json")
    if not isinstance(meta, dict):
        return {
            "status": "no_data",
            "patch_version": DEFAULT_PATCH,
            "source": "fallback",
        }
    meta = dict(meta)
    meta["patch_version"] = _valid_patch(meta.get("patch_version")) or DEFAULT_PATCH
    meta["source"] = "opendota_api"
    last_fetched = meta.get("last_fetched", "")
    if last_fetched:
        try:
            fetched_ts = calendar.timegm(time.strptime(last_fetched, "%Y-%m-%dT%H:%M:%SZ"))
            age_days = (time.time() - fetched_ts) / 86400
            meta["age_days"] = round(max(0.0, age_days), 1)
            meta["is_stale"] = age_days > 14
        except (ValueError, OverflowError):
            meta["is_stale"] = True
    return meta


__all__ = [
    "DEFAULT_PATCH",
    "get_current_patch",
    "get_fetched_patch",
    "get_patch_version",
    "check_patch_version",
    "fetch_and_save_gamedata",
    "load_heroes",
    "load_abilities",
    "load_items",
    "load_matchup_tips",
    "get_meta",
    "get_hero_catalog",
    "resolve_hero_name",
    "hero_slug",
    "resolve_item_key",
    "resolve_item_name",
    "item_key",
    "display_slug",
    "data_generation",
    "invalidate_all_caches",
    "clear_caches",
]
