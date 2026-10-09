"""Dota 2 ability and item data for the AI coaching system.

Thin, read-only view over :mod:`data_manager`, used to inject concrete
mechanical information (cooldowns, mana costs, damage, active items) into
LLM evaluation prompts.

Data layering lives in ``data_manager``: fetched OpenDota stats (or the
minimal numeric fallback in ``backend/data/fallback/``) merged with the
editorial coaching text in ``backend/data/editorial/``. Always go through the
accessors :func:`get_ability_table` / :func:`get_item_table` (or the helpers
below) — they return the *current* tables, which change after a game-data
update. The legacy module attributes ``ABILITY_DATA`` / ``ITEM_DATA`` /
``PATCH_VERSION`` are still served (computed on access) for backward
compatibility, but a name imported with ``from game_data import ABILITY_DATA``
goes stale after a reload.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Any

from . import data_manager

logger = logging.getLogger(__name__)
_logger = logger  # backward-compatible alias

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _squash(text: Any) -> str:
    return _NON_ALNUM.sub("", str(text or "").lower())


# ---------------------------------------------------------------------------
# Accessors (C4)
# ---------------------------------------------------------------------------

def get_ability_table() -> dict[str, dict[str, dict]]:
    """Current ability table: hero display name -> ability name -> data (read-only)."""
    return data_manager.load_abilities()


def get_item_table() -> dict[str, dict]:
    """Current item table: item display name -> data (read-only)."""
    return data_manager.load_items()


# Derived lookups, rebuilt whenever data_manager's generation changes.
_lookup_lock = threading.Lock()
_lookups: dict[str, Any] = {"generation": -1, "ability_keys": {}, "item_keys": {}}


def invalidate_cache() -> None:
    """Drop derived lookups (called by data_manager.invalidate_all_caches)."""
    global _lookups
    with _lookup_lock:
        _lookups = {"generation": -1, "ability_keys": {}, "item_keys": {}}


def _get_lookups() -> dict[str, Any]:
    global _lookups
    generation = data_manager.data_generation()
    current = _lookups
    if current["generation"] == generation:
        return current
    with _lookup_lock:
        if _lookups["generation"] == generation:
            return _lookups
        ability_keys: dict[str, tuple[str, str]] = {}
        for hero, abilities in get_ability_table().items():
            for name, data in abilities.items():
                key = data.get("key")
                if key:
                    ability_keys.setdefault(key, (hero, name))
        item_keys: dict[str, str] = {}
        for name, data in get_item_table().items():
            key = data.get("key")
            if key:
                item_keys.setdefault(str(key), name)
            item_keys.setdefault(_squash(name), name)
        _lookups = {"generation": generation, "ability_keys": ability_keys, "item_keys": item_keys}
        return _lookups


def _ensure_data_loaded() -> None:
    """Backward-compatible no-op: tables are loaded lazily by the accessors."""
    get_ability_table()
    get_item_table()


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def get_hero_abilities(hero_name: str) -> dict[str, dict]:
    """All abilities of a hero (display name / slug / npc name accepted)."""
    table = get_ability_table()
    if hero_name in table:
        return table[hero_name]
    resolved = data_manager.resolve_hero_name(hero_name)
    return table.get(resolved, {}) if resolved else {}


def get_ability_data(hero_name: str, ability_name: str) -> dict | None:
    """One ability by display name or ability key (None when unknown)."""
    abilities = get_hero_abilities(hero_name)
    if not abilities:
        return None
    if ability_name in abilities:
        return abilities[ability_name]
    squashed = _squash(ability_name)
    for name, data in abilities.items():
        if data.get("key") == ability_name or _squash(name) == squashed:
            return data
    return None


def _ability_display_name(hero_name: str, ability_name: str) -> str:
    """Table key for an ability given its display name or key."""
    abilities = get_hero_abilities(hero_name)
    if ability_name in abilities:
        return ability_name
    squashed = _squash(ability_name)
    for name, data in abilities.items():
        if data.get("key") == ability_name or _squash(name) == squashed:
            return name
    return ability_name


def get_ability_by_key(ability_key: str) -> tuple[str, str, dict] | None:
    """Reverse lookup ``ability_key -> (hero, ability_name, data)``."""
    hit = _get_lookups()["ability_keys"].get(ability_key)
    if not hit:
        return None
    hero, name = hit
    data = get_ability_table().get(hero, {}).get(name)
    return (hero, name, data) if data is not None else None


def get_item_data(item_name: str) -> dict | None:
    """One item by display name, internal key or ``item_`` name (None when unknown)."""
    table = get_item_table()
    if item_name in table:
        return table[item_name]
    lookups = _get_lookups()["item_keys"]
    key = data_manager.resolve_item_key(item_name)
    name = lookups.get(key) if key else None
    if name is None:
        name = lookups.get(_squash(item_name))
    return table.get(name) if name else None


def get_hero_ultimate(hero_name: str) -> tuple[str, dict] | None:
    """The hero's ultimate ``(name, data)`` when known."""
    for name, data in get_hero_abilities(hero_name).items():
        if data.get("is_ultimate"):
            return name, data
    return None


def get_signature_ability(hero_name: str) -> tuple[str, dict] | None:
    """The hero's most threatening basic ability (disable > damage > active)."""
    best: tuple[int, int, str, dict] | None = None
    for order, (name, data) in enumerate(get_hero_abilities(hero_name).items()):
        if data.get("is_ultimate"):
            continue
        if data.get("type") == "passive":
            rank = 0
        elif data.get("stun_duration"):
            rank = 3
        elif data.get("damage"):
            rank = 2
        else:
            rank = 1
        candidate = (rank, -order, name, data)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return (best[2], best[3]) if best else None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_FORMAT_LABELS = {
    "zh": {"cd": "CD {}秒", "mana": "魔耗{}", "stun": "眩晕{}秒", "duration": "持续{}秒",
           "damage_types": {"magical": "魔法", "physical": "物理", "pure": "纯粹"},
           "damage_fmt": "{}伤害{}", "all_levels": "各级"},
    "en": {"cd": "CD {}s", "mana": "Mana {}", "stun": "{}s stun", "duration": "Duration {}s",
           "damage_types": {"magical": "Magical", "physical": "Physical", "pure": "Pure"},
           "damage_fmt": "{} {} damage", "all_levels": "all levels"},
}


def _num(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _level_value(values: Any, level: int | None) -> str | None:
    """Value for a level (1-based), or every level joined by "/" when level is None."""
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        return _num(values)
    if isinstance(values, str):
        return values or None
    if not isinstance(values, list) or not values:
        return None
    if level is None:
        unique = []
        for v in values:
            if v not in unique:
                unique.append(v)
        return "/".join(_num(v) for v in (unique if len(unique) == 1 else values))
    idx = max(0, min(level - 1, len(values) - 1))
    return _num(values[idx])


def format_ability_at_level(
    hero_name: str, ability_name: str, level: int | None = None, lang: str = "zh"
) -> str:
    """Format one ability as localized text.

    ``level`` is 1-based; ``None`` (or <1) shows the values for every level
    ("CD 14/12/10/8s"), which suits enemy abilities whose level is unknown.
    """
    data = get_ability_data(hero_name, ability_name)
    if not data:
        return f"{ability_name}"
    if level is not None and level < 1:
        level = None
    lbl = _FORMAT_LABELS.get(lang, _FORMAT_LABELS["zh"])
    head = f"{ability_name} (Lv{level})" if level is not None else ability_name
    parts: list[str] = []
    cd = _level_value(data.get("cooldown"), level)
    if cd and cd not in ("0",):
        parts.append(lbl["cd"].format(cd))
    mana = _level_value(data.get("mana_cost"), level)
    if mana and mana not in ("0",):
        parts.append(lbl["mana"].format(mana))
    dmg = _level_value(data.get("damage"), level)
    if dmg:
        dtype = lbl["damage_types"].get(data.get("damage_type", ""), "")
        parts.append(lbl["damage_fmt"].format(dtype, dmg).strip())
    stun = _level_value(data.get("stun_duration"), level)
    if stun:
        parts.append(lbl["stun"].format(stun))
    if isinstance(data.get("duration"), list):
        dur = _level_value(data["duration"], level)
        if dur:
            parts.append(lbl["duration"].format(dur))
    return f"{head}: {', '.join(parts)}" if parts else head


def _tip_line(data: dict, lang: str) -> str | None:
    tip = str(data.get("coaching_tip") or "").strip()
    if not tip:
        return None
    return f"  - {'Tip' if lang == 'en' else '提示'}: {tip}"


def format_hero_abilities(
    hero_name: str, ability_levels: dict[str, int], lang: str = "zh", with_tips: bool = True
) -> str:
    """Format a hero's abilities at their current levels (keys: names or ability keys).

    ``with_tips=False`` omits the per-ability coaching tips.
    """
    abilities = get_hero_abilities(hero_name)
    if not abilities:
        return ""

    # When no skills have been chosen yet, show the level-1 options.
    if not ability_levels:
        if lang == "en":
            lines = [f"### {hero_name} Available Skills (not yet skilled)",
                     "Note: The hero can choose to skill any ability at any time. Below are level 1 effects for each skill."]
        else:
            lines = [f"### {hero_name} 可选技能（尚未加点）",
                     "注意：英雄在任何时刻可以选择技能加点，以下是1级各技能效果。"]
        for ability_name, data in abilities.items():
            if data.get("is_ultimate"):
                tag = "(requires level 6)" if lang == "en" else "(需要6级)"
                lines.append(f"- {ability_name}: {tag}")
            else:
                tag = "(selectable)" if lang == "en" else "(可选)"
                lines.append(f"- {tag} {format_ability_at_level(hero_name, ability_name, 1, lang=lang)}")
                tip = _tip_line(data, lang) if with_tips else None
                if tip:
                    lines.append(tip)
        return "\n".join(lines)

    levels: dict[str, int] = {}
    for name, level in ability_levels.items():
        try:
            lvl = int(level)
        except (TypeError, ValueError):
            continue
        display = _ability_display_name(hero_name, name)
        levels[display] = max(levels.get(display, 0), lvl)

    if lang == "en":
        lines = [f"### {hero_name} Ability Details"]
        not_learned_label = "Not learned"
    else:
        lines = [f"### {hero_name} 技能详情"]
        not_learned_label = "未学习"
    not_learned: list[str] = []
    for ability_name, data in abilities.items():
        level = levels.get(ability_name, 0)
        if level > 0:
            lines.append(f"- {format_ability_at_level(hero_name, ability_name, level, lang=lang)}")
            tip = _tip_line(data, lang) if with_tips else None
            if tip:
                lines.append(tip)
        else:
            not_learned.append(ability_name)
    if not_learned:
        sep = ", " if lang == "en" else "、"
        lines.append(f"- {not_learned_label}: {sep.join(not_learned)}")
    return "\n".join(lines)


def _short(text: Any, limit: int = 160) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def format_player_items(items: list[str], lang: str = "zh") -> str:
    """Format a player's items with coaching-relevant info (active items first)."""
    if not items:
        return ""
    unique: list[str] = []
    for item_name in items:
        if item_name and item_name not in unique:
            unique.append(item_name)
    lines = ["### Item Analysis" if lang == "en" else "### 装备分析"]
    active_items: list[tuple[str, dict]] = []
    passive_names: list[str] = []
    for item_name in unique:
        data = get_item_data(item_name)
        if data and data.get("is_active"):
            active_items.append((item_name, data))
        else:
            passive_names.append(item_name)
    for name, data in active_items[:4]:
        cd = data.get("cooldown")
        cd_text = _level_value(cd, None) if cd not in (None, "", False) else "?"
        desc = _short(data.get("description", ""))
        if lang == "en":
            lines.append(f"- {name} (Active, CD {cd_text}s): {desc}")
        else:
            lines.append(f"- {name}（主动, CD {cd_text}秒）: {desc}")
        tip = _tip_line(data, lang)
        if tip:
            lines.append(tip)
    passive_names.extend(name for name, _ in active_items[4:])
    if passive_names:
        shown = passive_names[:12]
        if lang == "en":
            lines.append(f"- Other items: {', '.join(shown)}")
        else:
            lines.append(f"- 其他装备: {'、'.join(shown)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backward-compatible module attributes (computed on access)
# ---------------------------------------------------------------------------

def __getattr__(name: str) -> Any:
    if name == "ABILITY_DATA":
        return get_ability_table()
    if name == "ITEM_DATA":
        return get_item_table()
    if name == "PATCH_VERSION":
        return data_manager.get_current_patch()
    if name == "ABILITY_KEY_LOOKUP":
        return dict(_get_lookups()["ability_keys"])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "get_ability_table",
    "get_item_table",
    "get_hero_abilities",
    "get_ability_data",
    "get_ability_by_key",
    "get_item_data",
    "get_hero_ultimate",
    "get_signature_ability",
    "format_ability_at_level",
    "format_hero_abilities",
    "format_player_items",
    "invalidate_cache",
]
