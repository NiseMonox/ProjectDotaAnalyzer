"""Vision and fog-of-war analysis engine for AI coaching.

Given a MatchData, a player_slot, and a time T, this module calculates:
1. What the player's team CAN see (vision from wards, towers, allied heroes)
2. What enemies are visible vs missing (and for how long)
3. How dangerous the player's position is
4. A timeline of vision changes in the 60 seconds before T (for death analysis)

All positions are Dota 2 world units (see ``backend/app/geo.py``) and every
distance threshold below is in real game units (observer ward vision 1600,
hero day vision 1800, ...). ``(0.0, 0.0)`` means "position unknown".

Tower positions come from ``MatchData.buildings`` when the parser extracted
them, otherwise from ``geo.TOWER_POSITIONS``. Text output follows ``lang``
("zh" or "en").

Performance: analyses repeatedly ask "where is hero X at second T" and
"what does team Y see at second T". A per-match index (built once, cached
weakly per MatchData object, thread-safe) memoizes those answers, so a full
report stays well under a second of CPU.
"""
from __future__ import annotations

import bisect
import logging
import math
import threading
import weakref
from typing import Any

from .. import geo
from ..models import AbilityEvent, MatchData, PlayerData, WardEvent
from .models import (
    EnemyVisibility,
    VisionAnalysis,
    VisionEvent,
    VisionState,
    VisionTimeline,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (world units / seconds)
# ---------------------------------------------------------------------------

OBS_WARD_RADIUS = 1600
SENTRY_WARD_RADIUS = 1000        # true sight only, not ground vision
OBS_WARD_DURATION = 360          # 6 minutes
SENTRY_WARD_DURATION = 420       # 7 minutes

HERO_DAY_VISION = 1800
HERO_NIGHT_VISION = 800
TOWER_DAY_VISION = 1900
TOWER_NIGHT_VISION = 800

DAY_NIGHT_PERIOD = 300           # 5 minutes per half-cycle, day first

TIMELINE_WINDOW = 60             # seconds before the decision to analyze
TIMELINE_STEP = 5                # seconds between timeline samples
LAST_SEEN_LOOKBACK = 120         # how far back "last seen" searches go

# Danger assessment thresholds
SAFE_TOWER_DISTANCE = 1500       # "near an allied tower"
MODERATE_TOWER_DISTANCE = 2500
FAR_TOWER_DISTANCE = 3500
BASE_SAFE_RADIUS = 3200          # around the own ancient counts as safe
WARD_COVERAGE_RADIUS = OBS_WARD_RADIUS

# "Nearby" hero radius: tighter while heroes are spread over three lanes.
NEARBY_RADIUS_LANING = 2000
NEARBY_RADIUS_LATER = 2500
_LANING_END = 600

# An ability cast reveals where its caster stood; older casts are stale.
FRESH_POSITION_MAX_AGE = 10
# Laning snapshots are taken every 5 s; tolerate one missing sample.
_SNAPSHOT_MAX_GAP = 10
_TEAMFIGHT_MARGIN = 5

# Backward-compatible private names (older callers imported these).
_OBS_WARD_RADIUS = OBS_WARD_RADIUS
_SENTRY_WARD_RADIUS = SENTRY_WARD_RADIUS
_OBS_WARD_DURATION = OBS_WARD_DURATION
_SENTRY_WARD_DURATION = SENTRY_WARD_DURATION
_TIMELINE_WINDOW = TIMELINE_WINDOW
_TIMELINE_STEP = TIMELINE_STEP
_SAFE_TOWER_DISTANCE = SAFE_TOWER_DISTANCE
_MODERATE_TOWER_DISTANCE = MODERATE_TOWER_DISTANCE
_FAR_TOWER_DISTANCE = FAR_TOWER_DISTANCE
_WARD_COVERAGE_RADIUS = WARD_COVERAGE_RADIUS

# Tower bitmask bit indices (LSB = index 0), OpenDota/Valve layout.
_TOWER_BIT_ORDER = [
    "top_t1", "top_t2", "top_t3",
    "mid_t1", "mid_t2", "mid_t3",
    "bot_t1", "bot_t2", "bot_t3",
    "ancient_top", "ancient_bot",
]

_UNKNOWN_POS = (0.0, 0.0)


def nearby_radius(time_s: int) -> float:
    """World-unit radius within which another hero counts as "nearby"."""
    return NEARBY_RADIUS_LANING if time_s < _LANING_END else NEARBY_RADIUS_LATER


def has_position(pos: tuple[float, float]) -> bool:
    return not (pos[0] == 0.0 and pos[1] == 0.0)


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two world-space points."""
    return math.hypot(x1 - x2, y1 - y2)


def _enemy_team(team: str) -> str:
    return "dire" if team == "radiant" else "radiant"


def ward_team(ward: WardEvent) -> str:
    """Team owning a ward ("" when unknown).

    ``WardEvent.team`` wins; otherwise it is derived from the owner slot
    (``player_slot`` is -1 when the owner is unknown).
    """
    if ward.team in ("radiant", "dire"):
        return ward.team
    if ward.player_slot is None or ward.player_slot < 0:
        return ""
    return "radiant" if ward.player_slot < 128 else "dire"


def _ward_pos_to_game(ward: WardEvent) -> tuple[float, float]:
    """Ward position in world units (kept for backward compatibility)."""
    return (ward.x, ward.y)


# ===================================================================
# Localization
# ===================================================================

_TEXT: dict[str, dict[str, str]] = {
    "zh": {
        "radiant": "天辉",
        "dire": "夜魇",
        "unknown_area": "未知区域",
        "fountain": "{side}泉水附近",
        "base": "{side}基地",
        "lane": "{side}{lane}",
        "river": "{half}河道",
        "jungle": "{side}野区（靠{half}）",
        "lane_top": "上路",
        "lane_mid": "中路",
        "lane_bot": "下路",
        "half_top": "上路",
        "half_bot": "下路",
        "night": "夜间",
        "day": "白天",
        # danger reasons
        "r_missing_long": "{n}名敌方英雄已消失超过{s}秒",
        "r_missing_many": "{n}名敌方英雄消失在迷雾中",
        "r_missing_two": "2名敌方英雄消失在迷雾中（最长{s}秒）",
        "r_missing_one": "{hero}已消失{s}秒",
        "r_far_tower": "距离最近友方防御塔约{d}码",
        "r_mod_tower": "与友方防御塔距离较远（约{d}码）",
        "r_no_ward": "当前区域没有观察守卫覆盖",
        "r_no_allies": "附近没有友方英雄支援",
        "r_night": "夜间视野缩小，更容易被突袭",
        # timeline
        "ev_night": "进入夜间，所有单位视野缩小",
        "ev_day": "进入白天，视野恢复正常",
        "ev_ward_destroyed": "{area}的观察守卫被摧毁，视野丢失",
        "ev_ward_expired": "{area}的观察守卫到期，视野丢失",
        "ev_ward_placed": "在{area}放置了观察守卫",
        "ev_enemy_missing": "{hero}从{area}消失",
        "ev_enemy_seen": "{hero}出现在{area}",
        "ev_many_missing": "{n}名敌方英雄消失超过10秒，形势危险",
        "ev_tower_lost": "友方{tower}被摧毁，失去该区域固定视野",
        "ev_entered_fog": "玩家进入无视野覆盖的{area}",
        "crit_night": "夜间开始，视野缩小增加被gank风险",
        "crit_ward_destroyed": "{area}观察守卫被摧毁后未补放新眼，失去该区域敌方动向",
        "crit_ward_expired": "{area}观察守卫到期后未补放新眼，失去该区域敌方动向",
        "crit_many_missing": "{n}名敌方英雄同时消失超过10秒，形势危险",
        "crit_tower_lost": "友方{tower}被摧毁，失去重要固定视野",
        "crit_entered_fog": "玩家进入没有视野覆盖的危险区域（{area}）",
        "sum_none": "此时间段内没有显著视野变化。",
        "sum_ward": "{n}次视野丢失（观察守卫）",
        "sum_tower": "{n}座防御塔被摧毁",
        "sum_enemy": "{n}次敌方英雄消失事件",
        "sum_prefix": "过去60秒内发生：{parts}。",
        "sum_stable": "过去60秒内视野状况相对稳定。",
        "sep": "，",
        # formatted text
        "h_vision": "## 视野分析",
        "cur_time": "当前时间: {t} ({dn})",
        "obs_count": "己方活跃观察守卫: {n}个",
        "danger": "视野危险等级: {lvl}",
        "player_pos": "玩家位置: {area}",
        "h_enemy_vis": "### 敌方英雄可见性",
        "vis_visible": "- {hero}: 可见（{area}）",
        "vis_missing": "- {hero}: 已消失{s}秒（最后出现在{area} @ {t}）",
        "vis_unknown": "- {hero}: 位置未知（无近期视野数据）",
        "vis_none": "- 无敌方英雄数据",
        "h_danger": "### 危险因素",
        "h_timeline": "### 视野时间线（事件发生前{w}秒）",
        "h_critical": "### 关键视野失误",
        "crit_line": "{t}（事件发生前{s}秒）{desc}",
        "crit_after": "此后{heroes}{suffix}仍在迷雾中，玩家未及时撤退到安全区域。",
        "crit_suffix": "等{n}名英雄",
        "name_sep": "、",
        "no_player": "## 视野分析\n无法获取玩家数据，跳过视野分析。\n",
    },
    "en": {
        "radiant": "Radiant",
        "dire": "Dire",
        "unknown_area": "unknown area",
        "fountain": "near the {side} fountain",
        "base": "{side} base",
        "lane": "{side} {lane}",
        "river": "{half} river",
        "jungle": "{side} jungle ({half} side)",
        "lane_top": "top lane",
        "lane_mid": "mid lane",
        "lane_bot": "bottom lane",
        "half_top": "top",
        "half_bot": "bottom",
        "night": "Night",
        "day": "Day",
        "r_missing_long": "{n} enemy heroes have been missing for over {s}s",
        "r_missing_many": "{n} enemy heroes are missing in the fog",
        "r_missing_two": "2 enemy heroes are missing in the fog (longest {s}s)",
        "r_missing_one": "{hero} has been missing for {s}s",
        "r_far_tower": "about {d} units from the nearest allied tower",
        "r_mod_tower": "fairly far from allied towers (about {d} units)",
        "r_no_ward": "no observer ward covers the current area",
        "r_no_allies": "no allied heroes nearby for support",
        "r_night": "night time reduces vision, making ganks easier",
        "ev_night": "Night falls; all units' vision shrinks",
        "ev_day": "Day breaks; vision returns to normal",
        "ev_ward_destroyed": "Observer ward at {area} destroyed; vision lost",
        "ev_ward_expired": "Observer ward at {area} expired; vision lost",
        "ev_ward_placed": "Observer ward placed at {area}",
        "ev_enemy_missing": "{hero} disappeared from {area}",
        "ev_enemy_seen": "{hero} spotted at {area}",
        "ev_many_missing": "{n} enemy heroes missing for over 10s — dangerous",
        "ev_tower_lost": "Allied {tower} destroyed; its permanent vision is gone",
        "ev_entered_fog": "Player entered {area} without vision coverage",
        "crit_night": "Night started; reduced vision raised the gank risk",
        "crit_ward_destroyed": "Observer ward at {area} was destroyed and not replaced, so enemy movement there went unseen",
        "crit_ward_expired": "Observer ward at {area} expired and was not replaced, so enemy movement there went unseen",
        "crit_many_missing": "{n} enemy heroes were missing for over 10s at the same time",
        "crit_tower_lost": "Allied {tower} fell, removing important permanent vision",
        "crit_entered_fog": "Player entered a dangerous area without vision ({area})",
        "sum_none": "No significant vision changes in this window.",
        "sum_ward": "{n} observer ward(s) lost",
        "sum_tower": "{n} tower(s) destroyed",
        "sum_enemy": "{n} enemy disappearance event(s)",
        "sum_prefix": "In the last 60 seconds: {parts}.",
        "sum_stable": "Vision was relatively stable in the last 60 seconds.",
        "sep": ", ",
        "h_vision": "## Vision Analysis",
        "cur_time": "Current time: {t} ({dn})",
        "obs_count": "Allied active Observer Wards: {n}",
        "danger": "Vision danger level: {lvl}",
        "player_pos": "Player position: {area}",
        "h_enemy_vis": "### Enemy Hero Visibility",
        "vis_visible": "- {hero}: visible ({area})",
        "vis_missing": "- {hero}: missing for {s}s (last seen at {area} @ {t})",
        "vis_unknown": "- {hero}: position unknown (no recent vision data)",
        "vis_none": "- No enemy hero data",
        "h_danger": "### Danger Factors",
        "h_timeline": "### Vision Timeline ({w}s before the event)",
        "h_critical": "### Critical Vision Mistake",
        "crit_line": "{t} ({s}s before the event) {desc}",
        "crit_after": "After this, {heroes}{suffix} stayed in the fog and the player did not retreat to safety in time.",
        "crit_suffix": " and {extra} more",
        "name_sep": ", ",
        "no_player": "## Vision Analysis\nUnable to retrieve player data, skipping vision analysis.\n",
    },
}

_DANGER_LABELS = {
    "zh": {"safe": "安全", "caution": "注意", "danger": "危险", "critical": "极度危险"},
    "en": {"safe": "Safe", "caution": "Caution", "danger": "Danger", "critical": "Critical"},
}

_TOWER_LANE = {"zh": {"top": "上路", "mid": "中路", "bot": "下路"},
               "en": {"top": "top", "mid": "mid", "bot": "bottom"}}
_TIER_ZH = {"t1": "一塔", "t2": "二塔", "t3": "高地塔"}


def _tx(lang: str) -> dict[str, str]:
    return _TEXT.get(lang, _TEXT["zh"])


def tower_label(name: str, lang: str = "zh") -> str:
    """Localized name for a tower key such as ``"top_t1"`` or ``"ancient_bot"``."""
    if name.startswith("ancient"):
        side = name.split("_", 1)[-1]
        if lang == "en":
            return f"tier-4 tower ({_TOWER_LANE['en'].get(side, side)})"
        return f"四塔（{_TOWER_LANE['zh'].get(side, side)}）"
    lane, _, tier = name.partition("_")
    if lang == "en":
        return f"{_TOWER_LANE['en'].get(lane, lane)} {tier.upper()} tower"
    return f"{_TOWER_LANE['zh'].get(lane, lane)}{_TIER_ZH.get(tier, tier)}"


def danger_level_label(level: str, lang: str = "zh") -> str:
    labels = _DANGER_LABELS.get(lang, _DANGER_LABELS["zh"])
    return labels.get(level, "Unknown" if lang == "en" else "未知")


def _danger_level_zh(level: str) -> str:
    return danger_level_label(level, "zh")


def _danger_level_en(level: str) -> str:
    return danger_level_label(level, "en")


# ===================================================================
# Map areas (coarse, geometry based)
# ===================================================================

def _lane_polylines() -> dict[str, list[tuple[float, float]]]:
    r = geo.TOWER_POSITIONS["radiant"]
    d = geo.TOWER_POSITIONS["dire"]
    return {
        "top": [r["top_t3"], r["top_t2"], r["top_t1"], (-6250.0, 5900.0),
                d["top_t1"], d["top_t2"], d["top_t3"]],
        "mid": [r["mid_t3"], r["mid_t2"], r["mid_t1"],
                d["mid_t1"], d["mid_t2"], d["mid_t3"]],
        "bot": [r["bot_t3"], r["bot_t2"], r["bot_t1"], (6250.0, -6100.0),
                d["bot_t1"], d["bot_t2"], d["bot_t3"]],
    }


_LANES = _lane_polylines()
_LANE_HALF_WIDTH = 1100.0
_RIVER_BAND = 1200.0
_FOUNTAIN_RADIUS = 2000.0
_MID_A = geo.TOWER_POSITIONS["radiant"]["mid_t3"]
_MID_B = geo.TOWER_POSITIONS["dire"]["mid_t3"]


def _segment_distance(px: float, py: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 0:
        return _distance(px, py, ax, ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return _distance(px, py, ax + t * dx, ay + t * dy)


def _polyline_distance(px: float, py: float, line: list[tuple[float, float]]) -> float:
    return min(_segment_distance(px, py, line[i], line[i + 1]) for i in range(len(line) - 1))


def _map_half(x: float, y: float) -> str:
    """"top" when the point lies on the top-lane side of the mid diagonal."""
    ax, ay = _MID_A
    bx, by = _MID_B
    cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
    return "top" if cross > 0 else "bot"


def area_code(x: float, y: float) -> tuple[str, str, str]:
    """Classify a world position as (kind, side, detail).

    kind is one of "unknown", "fountain", "base", "lane", "river", "jungle";
    side is "radiant"/"dire"/""; detail is the lane ("top"/"mid"/"bot") or
    the map half ("top"/"bot").
    """
    if not has_position((x, y)):
        return ("unknown", "", "")
    for team in ("radiant", "dire"):
        fx, fy = geo.FOUNTAIN_POSITIONS[team]
        if _distance(x, y, fx, fy) <= _FOUNTAIN_RADIUS:
            return ("fountain", team, "")
    d_rad = _distance(x, y, *geo.ANCIENT_POSITIONS["radiant"])
    d_dire = _distance(x, y, *geo.ANCIENT_POSITIONS["dire"])
    side = "radiant" if d_rad <= d_dire else "dire"
    if min(d_rad, d_dire) <= BASE_SAFE_RADIUS:
        return ("base", side, "")
    lane_dists = {lane: _polyline_distance(x, y, pts) for lane, pts in _LANES.items()}
    lane = min(lane_dists, key=lane_dists.get)
    if lane_dists[lane] <= _LANE_HALF_WIDTH:
        return ("lane", side, lane)
    half = _map_half(x, y)
    if abs(d_rad - d_dire) <= _RIVER_BAND:
        return ("river", "", half)
    return ("jungle", side, half)


def describe_area(x: float, y: float, lang: str = "zh") -> str:
    """Human-readable label for a world position."""
    t = _tx(lang)
    kind, side, detail = area_code(x, y)
    if kind == "unknown":
        return t["unknown_area"]
    side_label = t.get(side, "")
    if kind == "fountain":
        return t["fountain"].format(side=side_label)
    if kind == "base":
        return t["base"].format(side=side_label)
    if kind == "lane":
        return t["lane"].format(side=side_label, lane=t[f"lane_{detail}"])
    if kind == "river":
        return t["river"].format(half=t[f"half_{detail}"])
    return t["jungle"].format(side=side_label, half=t[f"half_{detail}"])


# Backward-compatible private name.
_describe_area = describe_area


# ===================================================================
# Day / Night cycle
# ===================================================================

def _is_night(time_s: int) -> bool:
    """Whether it is night: day 0:00-5:00, night 5:00-10:00, repeating."""
    if time_s < 0:
        return False
    return (time_s // DAY_NIGHT_PERIOD) % 2 == 1


def _day_night_label(time_s: int, lang: str = "zh") -> str:
    t = _tx(lang)
    return t["night"] if _is_night(time_s) else t["day"]


def _format_game_time(time_s: int) -> str:
    """Format game seconds as M:SS (negative before the horn)."""
    if time_s < 0:
        return f"-{abs(time_s) // 60}:{abs(time_s) % 60:02d}"
    return f"{time_s // 60}:{time_s % 60:02d}"


# ===================================================================
# Towers
# ===================================================================

def _parse_tower_unit_name(unit_str: str, team: str | None = None) -> str | None:
    """Map a building unit name to a tower key (e.g. "top_t1").

    Unit names look like ``npc_dota_goodguys_tower1_top`` /
    ``npc_dota_badguys_tower4``. When ``team`` is given, towers of the other
    team return None. Barracks / fort / other buildings return None.
    """
    unit = (unit_str or "").lower()
    if "tower" not in unit:
        return None
    is_radiant = "goodguys" in unit or "radiant" in unit
    is_dire = "badguys" in unit or "dire" in unit
    if not is_radiant and not is_dire:
        return None
    if team == "radiant" and not is_radiant:
        return None
    if team == "dire" and not is_dire:
        return None
    if "tower4" in unit or "ancient" in unit:
        if "bot" in unit:
            return "ancient_bot"
        return "ancient_top"
    lane = "top" if "top" in unit else "mid" if "mid" in unit else "bot" if "bot" in unit else None
    tier = None
    for n in ("1", "2", "3"):
        if f"tower{n}" in unit or f"tier{n}" in unit:
            tier = f"t{n}"
            break
    if lane and tier:
        return f"{lane}_{tier}"
    return None


def _unit_team(unit: str) -> str:
    unit = unit.lower()
    if "goodguys" in unit or "radiant" in unit:
        return "radiant"
    if "badguys" in unit or "dire" in unit:
        return "dire"
    return ""


def _destroyed_from_bitmask(match: MatchData, team: str) -> set[str]:
    """Towers missing in the end-of-game bitmask (set bit = alive)."""
    if not (match.tower_status_radiant or match.tower_status_dire):
        return set()  # both 0: the source did not provide building status
    bitmask = match.tower_status_radiant if team == "radiant" else match.tower_status_dire
    destroyed: set[str] = set()
    for i, name in enumerate(_TOWER_BIT_ORDER):
        if not (bitmask & (1 << i)):
            destroyed.add(name)
    return destroyed


def _build_towers(match: MatchData) -> dict[str, list[tuple[str, float, float, float]]]:
    """Per team: [(tower_key, x, y, destroyed_time)], inf = never destroyed."""
    inf = float("inf")
    # Destruction times from the objectives log.
    killed_at: dict[tuple[str, str], float] = {}
    for obj in match.objectives or []:
        if obj.get("type", "") not in ("building_kill", "CHAT_MESSAGE_TOWER_KILL", "tower", "tower_kill"):
            continue
        unit = str(obj.get("key") or obj.get("unit") or "")
        team = _unit_team(unit)
        name = _parse_tower_unit_name(unit, team) if team else None
        if name:
            # Both tier-4 towers may share one unsided unit name: the
            # second kill is the other tower.
            if name == "ancient_top" and "top" not in unit.lower() and (team, name) in killed_at:
                name = "ancient_bot"
            t = float(obj.get("time", 0) or 0)
            killed_at[(team, name)] = min(t, killed_at.get((team, name), inf))

    towers: dict[str, list[tuple[str, float, float, float]]] = {"radiant": [], "dire": []}
    seen: set[tuple[str, str]] = set()
    for b in match.buildings or []:
        try:
            if b.get("type") not in ("tower", None):
                continue
            unit = str(b.get("name", ""))
            team = b.get("team") or _unit_team(unit)
            if team not in towers:
                continue
            name = _parse_tower_unit_name(unit, team)
            if name == "ancient_top" and "top" not in unit.lower() and (team, name) in seen:
                name = "ancient_bot"  # second unsided tier-4 tower
            if not name or (team, name) in seen:
                continue
            x, y = float(b.get("x", 0.0)), float(b.get("y", 0.0))
            if not has_position((x, y)):
                continue
            destroyed = b.get("destroyed_time")
            dt = float(destroyed) if destroyed is not None else inf
            dt = min(dt, killed_at.get((team, name), inf))
            towers[team].append((name, x, y, dt))
            seen.add((team, name))
        except (TypeError, ValueError):
            continue

    for team in ("radiant", "dire"):
        if towers[team]:
            continue  # the replay gave us real positions for this team
        # Without any tower kill in the objectives log, the end-of-game
        # bitmask is the only information about destroyed towers.
        end_dead = _destroyed_from_bitmask(match, team) if not killed_at else set()
        for name, (x, y) in geo.TOWER_POSITIONS[team].items():
            dt = killed_at.get((team, name), inf)
            if dt == inf and name in end_dead:
                # Only the end-of-game state is known: treat as standing
                # until the very end of the match.
                dt = float(match.duration or 0)
            towers[team].append((name, x, y, dt))
    return towers


# ===================================================================
# Per-match index (memoization)
# ===================================================================

def _fingerprint(match: MatchData) -> tuple:
    return (
        len(match.players), len(match.wards), len(match.teamfights),
        len(match.ability_events), len(match.objectives or []),
        len(match.buildings or []), match.duration,
        sum(len(p.laning_snapshots) + len(p.ability_events) for p in match.players),
    )


class _MatchIndex:
    """Precomputed lookups + memo tables for one MatchData object."""

    def __init__(self, match: MatchData) -> None:
        self.fingerprint = _fingerprint(match)
        self.players: dict[int, PlayerData] = {p.player_slot: p for p in match.players}
        self.teams: dict[str, list[PlayerData]] = {
            "radiant": [p for p in match.players if p.team == "radiant"],
            "dire": [p for p in match.players if p.team == "dire"],
        }

        # Laning snapshots with a known position, sorted by time.
        self.snap_times: dict[int, list[int]] = {}
        self.snaps: dict[int, list[Any]] = {}
        for p in match.players:
            snaps = sorted(
                (s for s in p.laning_snapshots if has_position((s.x, s.y))),
                key=lambda s: s.time,
            )
            self.snaps[p.player_slot] = snaps
            self.snap_times[p.player_slot] = [s.time for s in snaps]

        # Ability casts with coordinates (match-level + player-level, deduped).
        casts: dict[int, dict[tuple, AbilityEvent]] = {}
        for p in match.players:
            for ae in p.ability_events:
                self._add_cast(casts, ae, p.player_slot)
        for ae in match.ability_events:
            self._add_cast(casts, ae, ae.player_slot)
        self.casts: dict[int, list[AbilityEvent]] = {}
        self.cast_times: dict[int, list[int]] = {}
        for slot, by_key in casts.items():
            ordered = sorted(by_key.values(), key=lambda a: a.time)
            self.casts[slot] = ordered
            self.cast_times[slot] = [a.time for a in ordered]

        # Teamfight participation per slot: (start, end, TeamfightPlayer).
        self.fights: dict[int, list[tuple[int, int, Any]]] = {}
        for tf in match.teamfights:
            for tp in tf.players:
                self.fights.setdefault(tp.player_slot, []).append((tf.start_time, tf.end_time, tp))

        self.towers = _build_towers(match)

        # Observer wards per team with their active interval [start, end).
        self.obs_wards: dict[str, list[tuple[int, WardEvent, int, int]]] = {"radiant": [], "dire": []}
        for idx, w in enumerate(match.wards):
            if w.ward_type != "observer":
                continue
            team = ward_team(w)
            if team not in self.obs_wards:
                continue
            end = w.destroyed_time if w.destroyed_time is not None else w.time + OBS_WARD_DURATION
            self.obs_wards[team].append((idx, w, w.time, end))

        self.pos_memo: dict[tuple[int, int], tuple[float, float]] = {}
        self.source_memo: dict[tuple[str, int], list[tuple[float, float, float, str]]] = {}
        self.visibility_memo: dict[tuple[str, int], list[EnemyVisibility]] = {}

    @staticmethod
    def _add_cast(casts: dict, ae: AbilityEvent, slot: int) -> None:
        if ae.x is None or ae.y is None or not has_position((ae.x, ae.y)):
            return
        key = (ae.time, ae.ability_name, round(ae.x, 1), round(ae.y, 1))
        casts.setdefault(slot, {}).setdefault(key, ae)


# Re-entrant: the weakref callback below may fire (via GC) while held.
_INDEX_LOCK = threading.RLock()
_INDEXES: dict[int, tuple[weakref.ref, _MatchIndex]] = {}
# Memo tables are dropped beyond this many entries (bounded memory for
# matches that stay cached in the server for a long time).
_MEMO_LIMIT = 60000


def _index(match: MatchData) -> _MatchIndex:
    """Return the (cached) index for ``match``; thread-safe."""
    key = id(match)
    with _INDEX_LOCK:
        entry = _INDEXES.get(key)
        if entry is not None:
            ref, idx = entry
            if ref() is match and idx.fingerprint == _fingerprint(match):
                if len(idx.pos_memo) + len(idx.source_memo) + len(idx.visibility_memo) > _MEMO_LIMIT:
                    idx.pos_memo = {}
                    idx.source_memo = {}
                    idx.visibility_memo = {}
                return idx
        idx = _MatchIndex(match)

        def _drop(_ref: weakref.ref, _key: int = key) -> None:
            with _INDEX_LOCK:
                current = _INDEXES.get(_key)
                if current is not None and current[0] is _ref:
                    del _INDEXES[_key]

        _INDEXES[key] = (weakref.ref(match, _drop), idx)
        return idx


# ===================================================================
# Player / team utilities
# ===================================================================

def _find_player(match: MatchData, player_slot: int) -> PlayerData | None:
    for p in match.players:
        if p.player_slot == player_slot:
            return p
    return None


def _same_team(slot_a: int, slot_b: int) -> bool:
    """Radiant slots are < 128, Dire slots >= 128."""
    return (slot_a < 128) == (slot_b < 128)


def _get_team_players(match: MatchData, team: str) -> list[PlayerData]:
    return [p for p in match.players if p.team == team]


def _get_enemy_players(match: MatchData, team: str) -> list[PlayerData]:
    return [p for p in match.players if p.team == _enemy_team(team)]


def _get_allied_players(match: MatchData, team: str, exclude_slot: int) -> list[PlayerData]:
    return [p for p in match.players if p.team == team and p.player_slot != exclude_slot]


# ===================================================================
# Position estimation
# ===================================================================

def _position_from_snapshots(idx: _MatchIndex, slot: int, t: int) -> tuple[float, float] | None:
    times = idx.snap_times.get(slot)
    if not times:
        return None
    snaps = idx.snaps[slot]
    i = bisect.bisect_left(times, t)
    if i < len(times) and times[i] == t:
        return (snaps[i].x, snaps[i].y)
    before = snaps[i - 1] if i > 0 else None
    after = snaps[i] if i < len(snaps) else None
    if before is not None and after is not None and after.time - before.time <= _SNAPSHOT_MAX_GAP:
        ratio = (t - before.time) / (after.time - before.time)
        return (before.x + (after.x - before.x) * ratio, before.y + (after.y - before.y) * ratio)
    for snap in (before, after):
        if snap is not None and abs(snap.time - t) <= _SNAPSHOT_MAX_GAP // 2:
            return (snap.x, snap.y)
    return None


def _position_from_fights(idx: _MatchIndex, slot: int, t: int) -> tuple[float, float] | None:
    for start, end, tp in idx.fights.get(slot, ()):
        if not (start - _TEAMFIGHT_MARGIN <= t <= end + _TEAMFIGHT_MARGIN):
            continue
        if tp.positions:
            offset = t - start
            best = min(tp.positions, key=lambda p: abs(p.time_offset - offset))
            if has_position((best.x, best.y)):
                return (best.x, best.y)
        if has_position((tp.x, tp.y)):
            return (tp.x, tp.y)
    return None


def _position_from_casts(
    idx: _MatchIndex, slot: int, t: int, max_age: int = FRESH_POSITION_MAX_AGE,
) -> tuple[float, float] | None:
    times = idx.cast_times.get(slot)
    if not times:
        return None
    casts = idx.casts[slot]
    i = bisect.bisect_left(times, t)
    best = None
    best_diff = max_age + 1
    for j in (i - 1, i):
        if 0 <= j < len(casts):
            diff = abs(casts[j].time - t)
            if diff < best_diff:
                best, best_diff = casts[j], diff
    if best is None or best_diff > max_age:
        return None
    return (float(best.x), float(best.y))


def estimate_position(match: MatchData, player_slot: int, time_s: int) -> tuple[float, float]:
    """Best-effort position of a hero at ``time_s`` (world units).

    Sources, most precise first: laning snapshots (5 s resolution, first
    10 minutes), teamfight positions, then ability casts within
    ``FRESH_POSITION_MAX_AGE`` seconds. Returns (0, 0) when unknown.
    """
    idx = _index(match)
    key = (player_slot, int(time_s))
    cached = idx.pos_memo.get(key)
    if cached is not None:
        return cached
    t = int(time_s)
    pos = (
        _position_from_snapshots(idx, player_slot, t)
        or _position_from_fights(idx, player_slot, t)
        or _position_from_casts(idx, player_slot, t)
        or _UNKNOWN_POS
    )
    idx.pos_memo[key] = pos
    return pos


def _estimate_player_position(player: PlayerData, time_s: int, match: MatchData) -> tuple[float, float]:
    """Backward-compatible wrapper around :func:`estimate_position`."""
    return estimate_position(match, player.player_slot, time_s)


def _estimate_all_hero_positions(match: MatchData, time_s: int) -> dict[int, tuple[float, float]]:
    """Positions for all heroes at ``time_s`` (slot -> (x, y))."""
    return {p.player_slot: estimate_position(match, p.player_slot, time_s) for p in match.players}


# ===================================================================
# Wards and towers over time
# ===================================================================

def _get_active_wards(match: MatchData, time_s: int) -> list[WardEvent]:
    """Wards placed and not yet expired/destroyed at ``time_s``."""
    active: list[WardEvent] = []
    for w in match.wards:
        if w.time > time_s:
            continue
        if w.destroyed_time is not None:
            if w.destroyed_time <= time_s:
                continue
        else:
            max_dur = OBS_WARD_DURATION if w.ward_type == "observer" else SENTRY_WARD_DURATION
            if time_s - w.time > max_dur:
                continue
        active.append(w)
    return active


def _active_observers(match: MatchData, team: str, time_s: int) -> list[tuple[int, WardEvent]]:
    """(ward index, ward) of a team's observer wards active at ``time_s``."""
    idx = _index(match)
    return [
        (i, w) for i, w, start, end in idx.obs_wards.get(team, ())
        if start <= time_s < end
    ]


def _get_team_observer_wards(match: MatchData, team: str, time_s: int) -> list[WardEvent]:
    return [w for _, w in _active_observers(match, team, time_s)]


def _ward_expiry_time(ward: WardEvent) -> int:
    if ward.destroyed_time is not None:
        return ward.destroyed_time
    max_dur = OBS_WARD_DURATION if ward.ward_type == "observer" else SENTRY_WARD_DURATION
    return ward.time + max_dur


def tower_positions(match: MatchData, team: str) -> list[tuple[str, float, float]]:
    """All tower positions of a team (replay buildings, else geo defaults)."""
    return [(name, x, y) for name, x, y, _dt in _index(match).towers.get(team, ())]


def _get_alive_towers(match: MatchData, team: str, time_s: int) -> list[tuple[str, float, float]]:
    """Towers of ``team`` standing at ``time_s``: [(tower_key, x, y)]."""
    return [
        (name, x, y) for name, x, y, dt in _index(match).towers.get(team, ())
        if dt > time_s
    ]


alive_towers = _get_alive_towers


def _get_destroyed_tower_names(match: MatchData, team: str, time_s: int) -> set[str]:
    return {
        name for name, _x, _y, dt in _index(match).towers.get(team, ())
        if dt <= time_s
    }


def nearest_tower_distance(
    match: MatchData, team: str, x: float, y: float, time_s: int,
) -> float:
    """Distance to the nearest standing tower of ``team`` (inf if none/unknown)."""
    if not has_position((x, y)):
        return float("inf")
    towers = _get_alive_towers(match, team, time_s)
    if not towers:
        return float("inf")
    return min(_distance(x, y, tx, ty) for _n, tx, ty in towers)


def in_own_base(team: str, x: float, y: float) -> bool:
    if not has_position((x, y)) or team not in geo.ANCIENT_POSITIONS:
        return False
    ax, ay = geo.ANCIENT_POSITIONS[team]
    return _distance(x, y, ax, ay) <= BASE_SAFE_RADIUS


# ===================================================================
# Vision sources
# ===================================================================

def _point_in_vision(px: float, py: float, sources) -> bool:
    """Whether a point lies within any (x, y, radius[, kind]) source."""
    for src in sources:
        if _distance(px, py, src[0], src[1]) <= src[2]:
            return True
    return False


def _team_sources(match: MatchData, team: str, time_s: int) -> list[tuple[float, float, float, str]]:
    """All of a team's vision sources at ``time_s`` (memoized)."""
    idx = _index(match)
    key = (team, int(time_s))
    cached = idx.source_memo.get(key)
    if cached is not None:
        return cached
    night = _is_night(time_s)
    sources: list[tuple[float, float, float, str]] = []
    for _i, w in _active_observers(match, team, time_s):
        sources.append((w.x, w.y, OBS_WARD_RADIUS, "ward"))
    tower_radius = TOWER_NIGHT_VISION if night else TOWER_DAY_VISION
    for _name, tx, ty in _get_alive_towers(match, team, time_s):
        sources.append((tx, ty, tower_radius, "tower"))
    hero_radius = HERO_NIGHT_VISION if night else HERO_DAY_VISION
    for p in idx.teams.get(team, ()):
        pos = estimate_position(match, p.player_slot, time_s)
        if has_position(pos):
            sources.append((pos[0], pos[1], hero_radius, "hero"))
    idx.source_memo[key] = sources
    return sources


def _build_vision_sources(
    match: MatchData,
    team: str,
    player_slot: int,
    time_s: int,
    include_hero_vision: bool = True,
) -> list[tuple[float, float, float]]:
    """Vision sources of a team as (x, y, radius) tuples."""
    return [
        (x, y, r) for x, y, r, kind in _team_sources(match, team, time_s)
        if include_hero_vision or kind != "hero"
    ]


def _identify_vision_source(
    x: float, y: float,
    match: MatchData, team: str, player_slot: int, time_s: int,
) -> str:
    """Kind of the first source seeing (x, y): ward > tower > hero."""
    hits = {kind for sx, sy, sr, kind in _team_sources(match, team, time_s)
            if _distance(x, y, sx, sy) <= sr}
    for kind in ("ward", "tower", "hero"):
        if kind in hits:
            return kind
    return "unknown"


# ===================================================================
# Enemy visibility tracking
# ===================================================================

def _grid_step(time_s: int) -> int:
    return 5 if time_s < _LANING_END else 10


def _find_last_seen(
    match: MatchData,
    team: str,
    player_slot: int,
    enemy: PlayerData,
    time_s: int,
    lookback: int = LAST_SEEN_LOOKBACK,
) -> tuple[int, tuple[float, float], str]:
    """When ``enemy`` was last visible to ``team`` before ``time_s``.

    Returns (last_seen_time, (x, y), source) where source is one of "ward",
    "tower", "hero", "ability", "teamfight", "unknown". last_seen_time is 0
    when the enemy was not seen within ``lookback`` seconds.
    """
    idx = _index(match)
    start = max(0, time_s - lookback)
    best_time, best_pos, best_source = 0, _UNKNOWN_POS, "unknown"

    # 1. Stepped check on a global grid (memo friendly).
    step = _grid_step(time_s)
    t = time_s - 1 - ((time_s - 1) % step)
    while t >= start and t >= 0:
        pos = estimate_position(match, enemy.player_slot, t)
        if has_position(pos):
            sources = _team_sources(match, team, t)
            if _point_in_vision(pos[0], pos[1], sources):
                best_time, best_pos = t, pos
                best_source = _identify_vision_source(pos[0], pos[1], match, team, player_slot, t)
                break
        t -= step

    # 2. Ability casts reveal the caster.
    times = idx.cast_times.get(enemy.player_slot, [])
    i = bisect.bisect_right(times, time_s) - 1
    if i >= 0:
        ae = idx.casts[enemy.player_slot][i]
        if ae.time >= start and ae.time > best_time:
            best_time, best_pos, best_source = ae.time, (float(ae.x), float(ae.y)), "ability"

    # 3. Teamfights: everyone involved was seen during the fight.
    for f_start, f_end, tp in idx.fights.get(enemy.player_slot, ()):
        seen_at = min(f_end, time_s)
        if f_start > time_s or seen_at < start or seen_at <= best_time:
            continue
        if not _involved(tp):
            continue
        pos = _position_from_fights(idx, enemy.player_slot, seen_at) or (tp.x, tp.y)
        if has_position(pos):
            best_time, best_pos, best_source = seen_at, pos, "teamfight"

    return best_time, best_pos, best_source


def _involved(tp) -> bool:
    return bool(tp.damage_dealt or tp.damage_taken or tp.kills or tp.deaths)


def _in_fight(idx: _MatchIndex, slot: int, time_s: int) -> bool:
    """Whether a hero takes part in a teamfight going on at ``time_s``.

    Heroes fighting are effectively visible to the other side.
    """
    for start, end, tp in idx.fights.get(slot, ()):
        if start <= time_s <= end and _involved(tp):
            return True
    return False


def _track_enemy_visibility(
    match: MatchData,
    team: str,
    player_slot: int,
    time_s: int,
) -> list[EnemyVisibility]:
    """Visibility status of each enemy hero at ``time_s`` (memoized).

    An enemy is visible when its estimated position lies inside the team's
    vision; enemies with no position estimate count as missing.
    """
    idx = _index(match)
    key = (team, int(time_s))  # the result does not depend on which teammate asks
    cached = idx.visibility_memo.get(key)
    if cached is not None:
        return cached

    sources = _team_sources(match, team, time_s)
    results: list[EnemyVisibility] = []
    for enemy in idx.teams.get(_enemy_team(team), ()):
        pos = estimate_position(match, enemy.player_slot, time_s)
        in_fight = _in_fight(idx, enemy.player_slot, time_s)
        if has_position(pos) and (in_fight or _point_in_vision(pos[0], pos[1], sources)):
            source = "teamfight" if in_fight else _identify_vision_source(
                pos[0], pos[1], match, team, player_slot, time_s)
            results.append(EnemyVisibility(
                hero_name=enemy.hero_name,
                player_slot=enemy.player_slot,
                is_visible=True,
                last_seen_time=time_s,
                last_seen_x=pos[0],
                last_seen_y=pos[1],
                seconds_missing=0,
                last_seen_source=source,
            ))
            continue
        seen_t, seen_pos, seen_src = _find_last_seen(match, team, player_slot, enemy, time_s)
        missing = (time_s - seen_t) if seen_t > 0 else max(0, min(time_s, LAST_SEEN_LOOKBACK))
        results.append(EnemyVisibility(
            hero_name=enemy.hero_name,
            player_slot=enemy.player_slot,
            is_visible=False,
            last_seen_time=seen_t,
            last_seen_x=seen_pos[0],
            last_seen_y=seen_pos[1],
            seconds_missing=missing,
            last_seen_source=seen_src,
        ))
    idx.visibility_memo[key] = results
    return results


# ===================================================================
# Danger assessment
# ===================================================================

def count_nearby_heroes(
    match: MatchData, team: str, x: float, y: float, time_s: int,
    exclude_slot: int | None = None, radius: float | None = None,
) -> int:
    """Number of ``team`` heroes with a known position within ``radius``."""
    if not has_position((x, y)):
        return 0
    if radius is None:
        radius = nearby_radius(time_s)
    count = 0
    for p in _index(match).teams.get(team, ()):
        if p.player_slot == exclude_slot:
            continue
        pos = estimate_position(match, p.player_slot, time_s)
        if has_position(pos) and _distance(x, y, pos[0], pos[1]) <= radius:
            count += 1
    return count


def _assess_danger(
    match: MatchData,
    team: str,
    player_slot: int,
    time_s: int,
    player_pos: tuple[float, float],
    enemy_visibility: list[EnemyVisibility],
    lang: str = "zh",
) -> tuple[str, list[str], bool, bool]:
    """Assess how dangerous the player's current position is.

    Returns (danger_level, reasons, player_in_safe_zone, has_ward_coverage)
    where danger_level is "safe", "caution", "danger" or "critical" and the
    reasons are localized sentences.
    """
    t = _tx(lang)
    reasons: list[str] = []
    known_pos = has_position(player_pos)

    missing = [e for e in enemy_visibility if not e.is_visible]
    missing_count = len(missing)
    longest_missing = max((e.seconds_missing for e in missing), default=0)

    if missing_count >= 3:
        if longest_missing > 20:
            reasons.append(t["r_missing_long"].format(n=missing_count, s=longest_missing))
        else:
            reasons.append(t["r_missing_many"].format(n=missing_count))
    elif missing_count == 2:
        reasons.append(t["r_missing_two"].format(s=longest_missing))
    elif missing_count == 1 and longest_missing > 30:
        reasons.append(t["r_missing_one"].format(hero=missing[0].hero_name, s=longest_missing))

    min_tower_dist = nearest_tower_distance(match, team, player_pos[0], player_pos[1], time_s)
    in_base = in_own_base(team, player_pos[0], player_pos[1])
    player_in_safe_zone = in_base or min_tower_dist <= SAFE_TOWER_DISTANCE
    if in_base:
        min_tower_dist = min(min_tower_dist, SAFE_TOWER_DISTANCE)

    if known_pos and min_tower_dist != float("inf"):
        if min_tower_dist > FAR_TOWER_DISTANCE:
            reasons.append(t["r_far_tower"].format(d=int(min_tower_dist)))
        elif min_tower_dist > MODERATE_TOWER_DISTANCE:
            reasons.append(t["r_mod_tower"].format(d=int(min_tower_dist)))

    has_ward_coverage = False
    if known_pos:
        for w in _get_team_observer_wards(match, team, time_s):
            if _distance(player_pos[0], player_pos[1], w.x, w.y) <= WARD_COVERAGE_RADIUS:
                has_ward_coverage = True
                break
        if not has_ward_coverage:
            reasons.append(t["r_no_ward"])

    nearby_allies = count_nearby_heroes(
        match, team, player_pos[0], player_pos[1], time_s, exclude_slot=player_slot,
    )
    if known_pos and nearby_allies == 0 and missing_count >= 2:
        reasons.append(t["r_no_allies"])

    night = _is_night(time_s)
    if night and missing_count >= 2:
        reasons.append(t["r_night"])

    danger_level = _compute_danger_level(
        missing_count=missing_count,
        longest_missing=longest_missing,
        min_tower_dist=min_tower_dist if known_pos else 0.0,
        has_ward_coverage=has_ward_coverage or not known_pos,
        nearby_allies=nearby_allies,
        is_night=night,
    )
    return danger_level, reasons, player_in_safe_zone, has_ward_coverage


def _compute_danger_level(
    missing_count: int,
    longest_missing: int,
    min_tower_dist: float,
    has_ward_coverage: bool,
    nearby_allies: int,
    is_night: bool,
) -> str:
    """Map danger factors to "safe" / "caution" / "danger" / "critical"."""
    score = 0
    if missing_count >= 4:
        score += 4
    elif missing_count >= 3:
        score += 3
    elif missing_count >= 2:
        score += 2
    elif missing_count >= 1:
        score += 1

    if longest_missing > 30:
        score += 2
    elif longest_missing > 20:
        score += 1

    if min_tower_dist > FAR_TOWER_DISTANCE:
        score += 2
    elif min_tower_dist > MODERATE_TOWER_DISTANCE:
        score += 1

    if not has_ward_coverage:
        score += 1

    if nearby_allies >= 2:
        score -= 2
    elif nearby_allies >= 1:
        score -= 1

    if is_night:
        score += 1

    if score <= 2:
        return "safe"
    if score <= 4:
        return "caution"
    if score <= 6:
        return "danger"
    return "critical"


# ===================================================================
# Vision state
# ===================================================================

def _build_vision_state(
    match: MatchData,
    team: str,
    player_slot: int,
    time_s: int,
    player_pos: tuple[float, float],
    lang: str = "zh",
) -> VisionState:
    """Complete VisionState at one moment."""
    enemy_vis = _track_enemy_visibility(match, team, player_slot, time_s)
    missing = [e for e in enemy_vis if not e.is_visible]
    danger_level, danger_reasons, in_safe_zone, ward_coverage = _assess_danger(
        match, team, player_slot, time_s, player_pos, enemy_vis, lang=lang,
    )
    return VisionState(
        time=time_s,
        is_night=_is_night(time_s),
        allied_observer_count=len(_active_observers(match, team, time_s)),
        enemy_visibility=list(enemy_vis),
        missing_enemy_count=len(missing),
        missing_enemy_names=[e.hero_name for e in missing],
        longest_missing_seconds=max((e.seconds_missing for e in missing), default=0),
        danger_level=danger_level,
        danger_reasons=danger_reasons,
        player_in_safe_zone=in_safe_zone,
        nearby_ward_coverage=ward_coverage,
    )


# ===================================================================
# Vision timeline
# ===================================================================

def _timeline_samples(start: int, end: int) -> list[int]:
    """Sample times on the global TIMELINE_STEP grid, plus ``end`` itself."""
    first = start + (-start % TIMELINE_STEP)
    samples = list(range(first, end + 1, TIMELINE_STEP))
    if not samples or samples[-1] != end:
        samples.append(end)
    return samples


def _build_vision_timeline(
    match: MatchData,
    team: str,
    player_slot: int,
    time_s: int,
    player_pos: tuple[float, float],
    lang: str = "zh",
) -> VisionTimeline:
    """Vision changes in the window before the decision.

    Samples the window and reports *changes* (wards lost/placed, enemies
    disappearing/appearing, day/night, towers lost), and picks the most
    impactful one as the critical moment.
    """
    tx = _tx(lang)
    events: list[VisionEvent] = []
    start_time = max(0, time_s - TIMELINE_WINDOW)

    pre_time = max(0, start_time - TIMELINE_STEP)
    prev_wards = {i: w for i, w in _active_observers(match, team, pre_time)}
    prev_visible = {
        e.player_slot for e in _track_enemy_visibility(match, team, player_slot, pre_time)
        if e.is_visible
    }
    prev_night = _is_night(pre_time)

    critical_moment = 0
    critical_description = ""
    critical_severity = 0
    warned_many_missing = False
    prev_t = pre_time

    def _critical(severity: int, when: int, text: str) -> None:
        nonlocal critical_moment, critical_description, critical_severity
        if severity > critical_severity:
            critical_severity, critical_moment, critical_description = severity, when, text

    for t in _timeline_samples(start_time, time_s):
        night = _is_night(t)
        if night != prev_night:
            if night:
                events.append(VisionEvent(time=t, event_type="night_start", description=tx["ev_night"]))
                if len(prev_visible) < 5:
                    _critical(2, t, tx["crit_night"])
            else:
                events.append(VisionEvent(time=t, event_type="day_start", description=tx["ev_day"]))
        prev_night = night

        current_wards = {i: w for i, w in _active_observers(match, team, t)}
        for ward_idx in prev_wards.keys() - current_wards.keys():
            ward = prev_wards[ward_idx]
            area = describe_area(ward.x, ward.y, lang)
            destroyed = (
                ward.destroyed_time is not None
                and prev_t < ward.destroyed_time <= t
                and ward.destroyed_time < ward.time + OBS_WARD_DURATION
            )
            kind = "ward_destroyed" if destroyed else "ward_expired"
            events.append(VisionEvent(
                time=t, event_type=kind, description=tx[f"ev_{kind}"].format(area=area),
            ))
            _critical(4, t, tx[f"crit_{kind}"].format(area=area))
        for ward_idx in current_wards.keys() - prev_wards.keys():
            ward = current_wards[ward_idx]
            events.append(VisionEvent(
                time=t, event_type="ward_placed",
                description=tx["ev_ward_placed"].format(area=describe_area(ward.x, ward.y, lang)),
            ))
        prev_wards = current_wards

        enemy_vis = _track_enemy_visibility(match, team, player_slot, t)
        by_slot = {e.player_slot: e for e in enemy_vis}
        current_visible = {e.player_slot for e in enemy_vis if e.is_visible}
        disappeared = prev_visible - current_visible
        for slot in sorted(disappeared):
            info = by_slot.get(slot)
            if info is not None:
                area = describe_area(info.last_seen_x, info.last_seen_y, lang)
                events.append(VisionEvent(
                    time=t, event_type="enemy_missing",
                    description=tx["ev_enemy_missing"].format(hero=info.hero_name, area=area),
                ))
        for slot in sorted(current_visible - prev_visible):
            info = by_slot.get(slot)
            if info is not None:
                area = describe_area(info.last_seen_x, info.last_seen_y, lang)
                events.append(VisionEvent(
                    time=t, event_type="enemy_seen",
                    description=tx["ev_enemy_seen"].format(hero=info.hero_name, area=area),
                ))

        long_missing = [e for e in enemy_vis if not e.is_visible and e.seconds_missing > 10]
        if len(long_missing) >= 3:
            _critical(5, t, tx["crit_many_missing"].format(n=len(long_missing)))
            if not warned_many_missing and (t == start_time or disappeared):
                events.append(VisionEvent(
                    time=t, event_type="enemy_missing",
                    description=tx["ev_many_missing"].format(n=len(long_missing)),
                ))
                warned_many_missing = True
        prev_visible = current_visible

        # Own towers falling inside this sample interval.
        for name, _x, _y, dt in _index(match).towers.get(team, ()):
            if prev_t < dt <= t:
                label = tower_label(name, lang)
                events.append(VisionEvent(
                    time=int(dt), event_type="tower_lost",
                    description=tx["ev_tower_lost"].format(tower=label),
                ))
                _critical(3, int(dt), tx["crit_tower_lost"].format(tower=label))
        prev_t = t

    if has_position(player_pos):
        sources = _build_vision_sources(match, team, player_slot, time_s, include_hero_vision=False)
        if not _point_in_vision(player_pos[0], player_pos[1], sources):
            area = describe_area(player_pos[0], player_pos[1], lang)
            events.append(VisionEvent(
                time=time_s, event_type="entered_fog",
                description=tx["ev_entered_fog"].format(area=area),
            ))
            if critical_severity < 3:
                _critical(3, time_s, tx["crit_entered_fog"].format(area=area))

    events.sort(key=lambda e: e.time)
    return VisionTimeline(
        events=events,
        critical_moment=critical_moment,
        critical_description=critical_description,
        summary=_build_timeline_summary(events, time_s, lang=lang),
    )


def _find_enemy_vis(vis_list: list[EnemyVisibility], player_slot: int) -> EnemyVisibility | None:
    for e in vis_list:
        if e.player_slot == player_slot:
            return e
    return None


def _build_timeline_summary(events: list[VisionEvent], time_s: int, lang: str = "zh") -> str:
    tx = _tx(lang)
    if not events:
        return tx["sum_none"]
    ward_losses = sum(1 for e in events if e.event_type in ("ward_expired", "ward_destroyed"))
    enemy_disappearances = sum(1 for e in events if e.event_type == "enemy_missing")
    tower_losses = sum(1 for e in events if e.event_type == "tower_lost")
    parts: list[str] = []
    if ward_losses:
        parts.append(tx["sum_ward"].format(n=ward_losses))
    if tower_losses:
        parts.append(tx["sum_tower"].format(n=tower_losses))
    if enemy_disappearances:
        parts.append(tx["sum_enemy"].format(n=enemy_disappearances))
    if parts:
        return tx["sum_prefix"].format(parts=tx["sep"].join(parts))
    return tx["sum_stable"]


# ===================================================================
# Formatted text output for the LLM prompt
# ===================================================================

_IMPORTANT_EVENTS = ("ward_expired", "ward_destroyed", "tower_lost", "entered_fog")


def _format_vision_text(
    state: VisionState,
    timeline: VisionTimeline,
    time_s: int,
    match: MatchData,
    player_slot: int,
    player_pos: tuple[float, float],
    lang: str = "zh",
) -> str:
    """Generate the formatted vision block for the LLM prompt."""
    tx = _tx(lang)
    lines: list[str] = [
        tx["h_vision"],
        tx["cur_time"].format(t=_format_game_time(time_s), dn=_day_night_label(time_s, lang)),
        tx["obs_count"].format(n=state.allied_observer_count),
        tx["danger"].format(lvl=danger_level_label(state.danger_level, lang)),
    ]
    if has_position(player_pos):
        lines.append(tx["player_pos"].format(area=describe_area(player_pos[0], player_pos[1], lang)))
    lines.append("")

    lines.append(tx["h_enemy_vis"])
    if state.enemy_visibility:
        for ev in state.enemy_visibility:
            if ev.is_visible:
                area = describe_area(ev.last_seen_x, ev.last_seen_y, lang)
                lines.append(tx["vis_visible"].format(hero=ev.hero_name, area=area))
            elif ev.last_seen_time > 0:
                area = describe_area(ev.last_seen_x, ev.last_seen_y, lang)
                lines.append(tx["vis_missing"].format(
                    hero=ev.hero_name, s=ev.seconds_missing, area=area,
                    t=_format_game_time(ev.last_seen_time),
                ))
            else:
                lines.append(tx["vis_unknown"].format(hero=ev.hero_name))
    else:
        lines.append(tx["vis_none"])
    lines.append("")

    if state.danger_reasons:
        lines.append(tx["h_danger"])
        lines.extend(f"- {reason}" for reason in state.danger_reasons)
        lines.append("")

    if timeline.events:
        lines.append(tx["h_timeline"].format(w=TIMELINE_WINDOW))
        marker = " !!!" if lang == "en" else " ⚠️"
        for evt in timeline.events:
            flag = marker if evt.event_type in _IMPORTANT_EVENTS else ""
            lines.append(f"[{_format_game_time(evt.time)}] {evt.description}{flag}")
        lines.append("")

    if timeline.critical_moment > 0 and timeline.critical_description:
        lines.append(tx["h_critical"])
        lines.append(tx["crit_line"].format(
            t=_format_game_time(timeline.critical_moment),
            s=time_s - timeline.critical_moment,
            desc=timeline.critical_description,
        ))
        missing_after = [e for e in state.enemy_visibility if not e.is_visible]
        if missing_after:
            heroes = tx["name_sep"].join(e.hero_name for e in missing_after[:3])
            suffix = (tx["crit_suffix"].format(n=len(missing_after), extra=len(missing_after) - 3)
                      if len(missing_after) > 3 else "")
            lines.append(tx["crit_after"].format(heroes=heroes, suffix=suffix))
        lines.append("")

    return "\n".join(lines)


# ===================================================================
# Public API
# ===================================================================

def analyze_vision(
    match: MatchData,
    player_slot: int,
    time: int,
    player_pos: tuple[float, float] = (0.0, 0.0),
    lang: str = "zh",
) -> VisionAnalysis:
    """Complete vision analysis at a decision point.

    Args:
        match: Full match data.
        player_slot: The player being analyzed.
        time: Game second to analyze.
        player_pos: Player's position (world units) at this time; (0, 0) to
            let the module estimate it.
        lang: Language for all text ("zh" or "en").
    """
    player = _find_player(match, player_slot)
    if player is None:
        empty_state = VisionState(time=time, is_night=_is_night(time), allied_observer_count=0)
        return VisionAnalysis(
            state_at_decision=empty_state,
            timeline=VisionTimeline(),
            formatted_text=_tx(lang)["no_player"],
        )

    team = player.team
    pos = (float(player_pos[0]), float(player_pos[1]))
    if not has_position(pos):
        pos = estimate_position(match, player_slot, time)

    state = _build_vision_state(match, team, player_slot, time, pos, lang=lang)
    timeline = _build_vision_timeline(match, team, player_slot, time, pos, lang=lang)
    formatted = _format_vision_text(state, timeline, time, match, player_slot, pos, lang=lang)
    return VisionAnalysis(state_at_decision=state, timeline=timeline, formatted_text=formatted)
