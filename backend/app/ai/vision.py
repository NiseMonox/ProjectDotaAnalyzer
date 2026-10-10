"""Vision and fog-of-war analysis engine for AI coaching.

Given a MatchData, a player_slot, and a time T, this module calculates:
1. Where every hero is and whether it is alive (``position_at`` /
   ``hero_alive_at``; a dead hero has no position and gives no vision)
2. What the player's team CAN see (vision from wards, towers, allied heroes)
3. The state of every enemy: "visible", "fog" (position known, outside the
   team's vision), "unknown" (the match data has no position for it: never
   reported as missing) or "dead"
4. How dangerous the player's position is
5. A timeline of vision changes in the 60 seconds before T (for death analysis)

All positions are Dota 2 world units (see ``backend/app/geo.py``) and every
distance threshold below is in real game units (observer ward vision 1600,
hero day vision 1800, ...). ``(0.0, 0.0)`` means "position unknown".

Game values: ward ranges and lifetimes and the Smoke of Deceit values are
read from the fetched game data (:func:`vision_rules`), each hero's day /
night vision too (``data_manager.hero_vision``); the built-in values of
``game_constants`` only stand in while no game data is available. Tower,
ancient and fountain vision are not in the fetched data and are constants
of that module as well. A value that changed with a patch is taken from the
match's own era when the old value is known (``_ERA_RULES``).

Position sources, best first: ``PlayerData.position_log`` (one sample every
5 s for the whole game; replay parser and mock), the death points of
``MatchData.kills_log``, laning snapshots, real teamfight samples and ability
casts with coordinates. OpenDota matches have no position log: outside the
moments a hero dies its position is unknown.

Tower positions come from ``MatchData.buildings`` when the parser extracted
them, otherwise from ``geo.TOWER_POSITIONS``. Text output follows ``lang``
("zh" or "en").

Performance: analyses repeatedly ask "where is hero X at second T" and
"what does team Y see at second T". A per-match index (built once, cached
weakly per MatchData object, thread-safe) holds the position log and the
deaths per hero and memoizes those answers, so a full report stays well
under a second of CPU.
"""
from __future__ import annotations

import bisect
import logging
import math
import re
import threading
import weakref
from typing import Any, NamedTuple

from .. import game_constants, geo
from ..models import AbilityEvent, MatchData, PlayerData, TeamfightEvent, WardEvent
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

# The game's own numbers are defined once, in ``game_constants`` (with the
# patch they were checked against). The item values are the offline fallback
# of :func:`vision_rules`, which reads the fetched data.
OBS_WARD_RADIUS = game_constants.OBSERVER_WARD_VISION
SENTRY_WARD_RADIUS = game_constants.SENTRY_WARD_TRUE_SIGHT    # true sight only, not ground vision
OBS_WARD_DURATION = game_constants.OBSERVER_WARD_DURATION_S
SENTRY_WARD_DURATION = game_constants.SENTRY_WARD_DURATION_S
SMOKE_DURATION = game_constants.SMOKE_DURATION_S
SMOKE_BREAK_RADIUS = game_constants.SMOKE_BREAK_RADIUS

# Default hero vision; the few heroes that differ (Night Stalker 800 / 1800,
# Slark 1800 / 1800, Sniper 1800 / 1400 ...) come from data_manager.hero_vision.
HERO_DAY_VISION = game_constants.HERO_DAY_VISION
HERO_NIGHT_VISION = game_constants.HERO_NIGHT_VISION
# Buildings (not in the fetched game data).
TOWER_DAY_VISION = game_constants.TOWER_DAY_VISION
TOWER_NIGHT_VISION = game_constants.TOWER_NIGHT_VISION        # tier 2, 3 and 4
TOWER_T1_NIGHT_VISION = game_constants.TOWER_T1_NIGHT_VISION  # tier 1
ANCIENT_VISION = game_constants.ANCIENT_VISION                # day and night
FOUNTAIN_VISION = game_constants.FOUNTAIN_VISION              # day and night
# A modelling choice, not a game value. Lane creeps are not in the data
# (they see 750 units themselves). During the laning stage the allied wave
# next to a hero keeps the lane opponents visible at night as well, so a
# hero's vision is never taken below this radius before 10:00.
LANE_CREEP_VISION = 1600

DAY_NIGHT_PERIOD = game_constants.DAY_NIGHT_PERIOD_S   # half-cycle, day first

TIMELINE_WINDOW = 60             # seconds before the decision to analyze
TIMELINE_STEP = 5                # seconds between timeline samples
LAST_SEEN_LOOKBACK = 120         # how far back "last seen" searches go
PREGAME_START = -90              # earliest game second with data (strategy time)

POSITION_LOG_STEP = 5            # PlayerData.position_log sampling interval
# Two consecutive samples further apart than this are a teleport / blink /
# respawn: the position in between is not interpolated.
_TELEPORT_DISTANCE = 3000
# A death point places the hero for the few seconds before the death.
DEATH_POINT_MAX_AGE = 5
# A hero that kills an enemy hero (or assists) was fighting it for at least
# this long before the kill.
KILL_FIGHT_SECONDS = 5
# The state "before a death" is sampled at least this long before it.
_PRE_DEATH_GAP = 2
_PRE_DEATH_LOOKBACK = 60

# An enemy hidden in the fog only threatens the player when it is close
# enough to reach him within a few seconds.
THREAT_RADIUS = 5000
# Hero deaths near a player this long before a moment belong to the fight he
# is in at that moment (a teamfight record extends it to the record's start).
ENGAGEMENT_LOOKBACK = 30

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
# With this many allies nearby, a move is the team's move, not the player's.
GROUP_ALLIES = 2

# An ability cast reveals where its caster stood; older casts are stale.
FRESH_POSITION_MAX_AGE = 10
# Laning snapshots are taken every 5 s; tolerate one missing sample.
_SNAPSHOT_MAX_GAP = 10
_TEAMFIGHT_MARGIN = 5

# Respawn time by hero level, used when the source has no position log to
# show the respawn (OpenDota).
_RESPAWN_SECONDS = game_constants.RESPAWN_SECONDS

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


class VisionRules(NamedTuple):
    """Ward and smoke values as they applied to one match (world units / seconds)."""

    observer_vision: int
    observer_duration: int
    sentry_true_sight: int
    sentry_duration: int
    smoke_duration: int
    smoke_break_radius: int


DEFAULT_RULES = VisionRules(
    OBS_WARD_RADIUS, OBS_WARD_DURATION, SENTRY_WARD_RADIUS, SENTRY_WARD_DURATION,
    SMOKE_DURATION, SMOKE_BREAK_RADIUS,
)
# VisionRules field -> name in data_manager.game_rules().
_RULE_NAMES = {
    "observer_vision": "observer_ward_vision",
    "observer_duration": "observer_ward_duration",
    "sentry_true_sight": "sentry_ward_true_sight",
    "sentry_duration": "sentry_ward_duration",
    "smoke_duration": "smoke_duration",
    "smoke_break_radius": "smoke_break_radius",
}
# Values that changed with a patch and whose old value is known: (first patch
# with today's value, field, value before it), from the history kept in
# ``game_constants``. Only these follow the era of an old match; every other
# rule is today's.
_ERA_RULES: tuple[tuple[tuple[int, int], str, int], ...] = tuple(
    (since, field, before)
    for since, source, before in game_constants.ITEM_RULE_HISTORY
    for field, name in _RULE_NAMES.items() if name == source
)
_PATCH_RE = re.compile(r"\s*(\d+)\.(\d+)")


def match_patch(match: MatchData | None) -> tuple[int, int] | None:
    """Version a match was played on as ``(7, 41)``; None when it is not known.

    ``MatchData.patch`` when the parser filled it, else the stored patch
    table by start time (never the network).
    """
    if match is None:
        return None
    label = str(getattr(match, "patch", "") or "")
    if not label:
        try:
            from . import data_manager
            start = getattr(match, "start_time", 0)
            label = data_manager.patch_label_for_time(start) or data_manager.patch_for_time(start)
        except Exception:  # pragma: no cover - the helpers never raise
            label = ""
    found = _PATCH_RE.match(label or "")
    return (int(found.group(1)), int(found.group(2))) if found else None


def vision_rules(match: MatchData | None = None) -> VisionRules:
    """Ward and smoke values for ``match`` (None: the current game).

    Read from the fetched game data; a value the data does not have is the
    7.41f value. For a match of a known older patch the values of
    ``_ERA_RULES`` are the ones of its time. The values of a match are kept
    with its index until the game data changes.
    """
    return _rules_for(None) if match is None else _index(match).rules


def _rules_for(match: MatchData | None) -> VisionRules:
    values = DEFAULT_RULES._asdict()
    try:
        from . import data_manager
        fetched = data_manager.game_rules()
        for name, source in _RULE_NAMES.items():
            value = fetched.get(source)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                values[name] = int(value)
    except Exception as exc:  # pragma: no cover - game_rules never raises
        logger.debug("Game rules unavailable, using the built-in values: %s", exc)
    patch = match_patch(match)
    if patch is not None:
        for since, name, before in _ERA_RULES:
            if patch < since:
                values[name] = before
    return VisionRules(**values)


def tower_vision(tower_key: str, night: bool) -> int:
    """Vision radius of a tower ("top_t1" ... "ancient_bot")."""
    if not night:
        return TOWER_DAY_VISION
    return TOWER_T1_NIGHT_VISION if tower_key.endswith("_t1") else TOWER_NIGHT_VISION


_data_generation_source: Any = None    # data_manager.data_generation, bound at first use


def _data_generation() -> int:
    """Counter that moves when the game data (rules, hero vision) was replaced.

    Part of every index lookup (:func:`_fingerprint`, some 3,000 a report):
    the data layer's function is looked up once, not imported per call.
    """
    global _data_generation_source
    source = _data_generation_source
    if source is None:
        try:
            from . import data_manager
            source = _data_generation_source = data_manager.data_generation
        except Exception:  # pragma: no cover - defensive
            return 0
    try:
        return source()
    except Exception:  # pragma: no cover - defensive
        return 0


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


# ===================================================================
# Localization
# ===================================================================

_TEXT: dict[str, dict[str, str]] = {
    "zh": {
        "radiant": "天辉",
        "dire": "夜魇",
        "unknown_area": "未知区域",
        "fountain": "{side}泉水",
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
        "r_missing_long": "{n}名敌方英雄藏在{r}码内的迷雾中，最长已{s}未被看到",
        "r_missing_many": "{n}名敌方英雄藏在{r}码内的迷雾中",
        "r_missing_two": "2名敌方英雄藏在{r}码内的迷雾中（最长{s}未被看到）",
        "r_missing_one": "{hero}藏在{r}码内的迷雾中，已{s}未被看到",
        "seconds": "{s}秒",
        "seconds_over": "超过{s}秒",
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
        "ev_enemy_missing": "{hero}离开己方视野（最后出现在{area}）",
        "ev_enemy_seen": "{hero}出现在{area}",
        "ev_many_missing": "{n}名敌方英雄藏在附近迷雾中超过10秒，形势危险",
        "ev_tower_lost": "友方{tower}被摧毁，失去该区域固定视野",
        "ev_entered_fog": "玩家离开守卫和防御塔的保护范围，进入{area}",
        "ev_group_move": "玩家与{n}名队友一起离开守卫和防御塔的保护范围，进入{area}",
        "crit_night": "夜间开始，视野缩小增加被gank风险",
        "crit_ward_destroyed": "{area}观察守卫被摧毁后未补放新眼，失去该区域敌方动向",
        "crit_ward_expired": "{area}观察守卫到期后未补放新眼，失去该区域敌方动向",
        "crit_many_missing": "{n}名敌方英雄同时藏在附近迷雾中超过10秒，形势危险",
        "crit_tower_lost": "友方{tower}被摧毁，失去重要固定视野",
        "crit_entered_fog": "附近有多名敌方英雄藏在迷雾中时，玩家离开了守卫和防御塔的保护范围（{area}）",
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
        "danger": "来自迷雾中敌人的威胁（推算）: {lvl}",
        "danger_unknown": "来自迷雾中敌人的威胁（推算）: 无法判断（{n}名敌方英雄没有位置数据）",
        "player_pos": "玩家位置: {area}",
        "player_pos_unknown": "玩家位置: 未知（数据中没有这一时刻的位置；下面出现的地点是眼位或防御塔的位置，不是玩家所在的位置）",
        "h_enemy_vis": "### 敌方英雄可见性",
        "vis_visible": "- {hero}: 可见（{area}）",
        "vis_visible_fight": "- {hero}: 可见（正在与玩家一方交战，数据中没有具体位置）",
        "vis_fog": "- {hero}: 在迷雾中已{s}（己方看不到）",
        "vis_smoked": "；此刻处于诡计之雾中（眼位看不到，只有贴近己方英雄或建筑才会现形）",
        "vis_seen": "；最后出现在{area} @ {t}",
        "vis_real": "；实际位置：{area}，距玩家约{d}码",
        "vis_real_nodist": "；实际位置：{area}",
        "vis_unknown": "- {hero}: 无位置数据（不计入消失）",
        "vis_dead": "- {hero}: 阵亡，等待复活",
        "vis_unknown_note": "注意：有{n}名敌方英雄没有位置数据，无法判断其是否在迷雾中。",
        "vis_no_positions": "- 其余{n}名敌方英雄: 本场数据没有位置记录，无法判断他们在哪里、己方是否看得到",
        "danger_no_positions": "来自迷雾中敌人的威胁（推算）: 无法判断（本场数据没有位置记录）",
        "vis_none": "- 无敌方英雄数据",
        "h_danger": "### 危险因素",
        "h_timeline": "### 视野时间线（事件发生前{w}秒）",
        "h_critical": "### 关键视野变化",
        "crit_line": "{t}（事件发生前{s}秒）{desc}",
        "crit_after": "此后{heroes}{suffix}仍在迷雾中，玩家未及时撤退到安全区域。",
        "crit_still": "此刻仍在迷雾中的敌方英雄：{heroes}{suffix}。",
        "crit_suffix": "等{n}名英雄",
        "name_sep": "、",
        "no_player": "## 视野分析\n无法获取玩家数据，跳过视野分析。\n",
    },
    "en": {
        "radiant": "Radiant",
        "dire": "Dire",
        "unknown_area": "unknown area",
        "fountain": "{side} fountain",
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
        "r_missing_long": "{n} enemy heroes hidden in the fog within {r} units, the longest unseen for {s}",
        "r_missing_many": "{n} enemy heroes hidden in the fog within {r} units",
        "r_missing_two": "2 enemy heroes hidden in the fog within {r} units (unseen for up to {s})",
        "r_missing_one": "{hero} hidden in the fog within {r} units, unseen for {s}",
        "seconds": "{s}s",
        "seconds_over": "over {s}s",
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
        "ev_enemy_missing": "{hero} left allied vision (last seen at {area})",
        "ev_enemy_seen": "{hero} spotted at {area}",
        "ev_many_missing": "{n} enemy heroes hidden in the fog nearby for over 10s — dangerous",
        "ev_tower_lost": "Allied {tower} destroyed; its permanent vision is gone",
        "ev_entered_fog": "Player moved away from wards and towers into {area}",
        "ev_group_move": "Player and {n} allies moved away from wards and towers together, into {area}",
        "crit_night": "Night started; reduced vision raised the gank risk",
        "crit_ward_destroyed": "Observer ward at {area} was destroyed and not replaced, so enemy movement there went unseen",
        "crit_ward_expired": "Observer ward at {area} expired and was not replaced, so enemy movement there went unseen",
        "crit_many_missing": "{n} enemy heroes were hidden in the fog nearby for over 10s at the same time",
        "crit_tower_lost": "Allied {tower} fell, removing important permanent vision",
        "crit_entered_fog": "Player moved away from wards and towers while several enemy heroes were hidden nearby ({area})",
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
        "danger": "Danger from hidden enemies (estimate): {lvl}",
        "danger_unknown": "Danger from hidden enemies (estimate): cannot be told ({n} enemy hero(es) have no position data)",
        "player_pos": "Player position: {area}",
        "player_pos_unknown": "Player position: unknown (the data has no position for this moment; the places named below are where wards or towers stood, not where the player was)",
        "h_enemy_vis": "### Enemy Hero Visibility",
        "vis_visible": "- {hero}: visible ({area})",
        "vis_visible_fight": "- {hero}: visible (fighting the player's side; the data has no exact position)",
        "vis_fog": "- {hero}: in the fog for {s} (not visible to the player's team)",
        "vis_smoked": "; under Smoke of Deceit at this moment (wards do not reveal it, only an allied hero or building close by)",
        "vis_seen": "; last seen at {area} @ {t}",
        "vis_real": "; actual position: {area}, about {d} units from the player",
        "vis_real_nodist": "; actual position: {area}",
        "vis_unknown": "- {hero}: no position data (not counted as missing)",
        "vis_dead": "- {hero}: dead, waiting to respawn",
        "vis_unknown_note": "Note: the match data has no position for {n} enemy hero(es); whether they were in the fog cannot be told.",
        "vis_no_positions": "- the other {n} enemy hero(es): this match's data has no positions, so where they were and whether the team saw them cannot be told",
        "danger_no_positions": "Danger from hidden enemies (estimate): cannot be told (this match's data has no positions)",
        "vis_none": "- No enemy hero data",
        "h_danger": "### Danger Factors",
        "h_timeline": "### Vision Timeline ({w}s before the event)",
        "h_critical": "### Key Vision Change",
        "crit_line": "{t} ({s}s before the event) {desc}",
        "crit_after": "After this, {heroes}{suffix} stayed in the fog and the player did not retreat to safety in time.",
        "crit_still": "Still in the fog at this moment: {heroes}{suffix}.",
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
_RIVER_HALF_WIDTH = 600.0
# Small enough to leave both ancients (1700 / 2100 units away) to the base.
_FOUNTAIN_RADIUS = 1200.0
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


def _river_line() -> list[tuple[float, float]]:
    """The river as a line through its three lane crossings (top, mid, bottom).

    A lane is crossed halfway between the two opposing tier-1 towers. The
    line of equal distance to both ancients is no substitute: at the side
    lanes it runs up to 2500 units behind the river, which puts the spot
    where the safe-lane creep waves meet, 1200 units in front of the own
    tower, on the enemy's side of the map.
    """
    r, d = geo.TOWER_POSITIONS["radiant"], geo.TOWER_POSITIONS["dire"]
    return [
        ((r[f"{lane}_t1"][0] + d[f"{lane}_t1"][0]) / 2, (r[f"{lane}_t1"][1] + d[f"{lane}_t1"][1]) / 2)
        for lane in ("top", "mid", "bot")
    ]


_RIVER = _river_line()


def river_offset(x: float, y: float) -> float:
    """Signed distance to the river: negative on Radiant's side, positive on Dire's."""
    top, mid, bot = _RIVER
    # Which stretch of the river the point is abeam of.
    along = (x - mid[0]) * (bot[0] - top[0]) + (y - mid[1]) * (bot[1] - top[1])
    (ax, ay), (bx, by) = (top, mid) if along < 0 else (mid, bot)
    dx, dy = bx - ax, by - ay
    return (dx * (y - ay) - dy * (x - ax)) / math.hypot(dx, dy)


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
    if min(d_rad, d_dire) <= BASE_SAFE_RADIUS:
        return ("base", "radiant" if d_rad <= d_dire else "dire", "")
    # Outside the bases a team's side of the map ends at the river.
    offset = river_offset(x, y)
    side = "radiant" if offset < 0 else "dire"
    lane_dists = {lane: _polyline_distance(x, y, pts) for lane, pts in _LANES.items()}
    lane = min(lane_dists, key=lane_dists.get)
    if lane_dists[lane] <= _LANE_HALF_WIDTH:
        return ("lane", side, lane)
    half = _map_half(x, y)
    if abs(offset) <= _RIVER_HALF_WIDTH:
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


_TIER4 = ("ancient_top", "ancient_bot")
# An unsided tier-4 kill this close to a building's own destruction time is
# that building's kill.
_TIER4_MATCH_S = 5


def _is_sided(unit: str) -> bool:
    unit = unit.lower()
    return "top" in unit or "bot" in unit


def _build_towers(match: MatchData) -> dict[str, list[tuple[str, float, float, float]]]:
    """Per team: [(tower_key, x, y, destroyed_time)], inf = never destroyed."""
    inf = float("inf")
    # Destruction times from the objectives log. A tier-4 kill may carry no
    # side ("npc_dota_goodguys_tower4"): those are kept apart and matched
    # with the buildings below.
    killed_at: dict[tuple[str, str], float] = {}
    unsided_t4: dict[str, list[float]] = {"radiant": [], "dire": []}
    for obj in match.objectives or []:
        if obj.get("type", "") not in ("building_kill", "CHAT_MESSAGE_TOWER_KILL", "tower", "tower_kill"):
            continue
        unit = str(obj.get("key") or obj.get("unit") or "")
        team = _unit_team(unit)
        name = _parse_tower_unit_name(unit, team) if team else None
        if not name:
            continue
        t = float(obj.get("time", 0) or 0)
        if name in _TIER4 and not _is_sided(unit):
            unsided_t4[team].append(t)
        else:
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
            if name == "ancient_top" and not _is_sided(unit) and (team, name) in seen:
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
        pending = sorted(unsided_t4[team])
        if towers[team]:
            # The replay gave real positions and per-building times: an
            # unsided kill only dates a tier-4 tower that has no time of its
            # own (never the other one, which may still stand).
            for _name, _x, _y, dt in towers[team]:
                if _name in _TIER4 and dt != inf:
                    near = [t for t in pending if abs(t - dt) <= _TIER4_MATCH_S]
                    if near:
                        pending.remove(near[0])
            for i, (name, x, y, dt) in enumerate(towers[team]):
                if name in _TIER4 and dt == inf and pending:
                    towers[team][i] = (name, x, y, pending.pop(0))
            continue
        # No buildings (OpenDota): unsided kills are taken in order.
        for name, t in zip(_TIER4, pending):
            if (team, name) not in killed_at:
                killed_at[(team, name)] = t
        # Without any tower kill in the objectives log, the end-of-game
        # bitmask is the only information about destroyed towers.
        end_dead = _destroyed_from_bitmask(match, team) if not (killed_at or pending) else set()
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
    """What an index was built from: a changed match (tests edit them in place) gets a new one.

    Computed on every lookup, so kept cheap; the per-player part stays in
    it (``test_changed_position_log_rebuilds_the_index``): an index that
    outlived a replaced position log would answer from the old one.
    """
    logs = 0
    for p in match.players:
        logs += len(p.laning_snapshots) + len(p.ability_events) + len(p.position_log)
    return (
        len(match.players), len(match.wards), len(match.teamfights),
        len(match.ability_events), len(match.objectives or ()),
        len(match.buildings or ()), match.duration, len(match.kills_log),
        logs,
        len(match.smokes or ()), match.patch, match.start_time,
        # The index holds ward lifetimes and hero vision of the game data.
        _data_generation(),
    )


def _ward_end(ward: WardEvent, rules: VisionRules | None = None) -> int:
    """Game second a ward stops giving vision (death, capped at its lifetime).

    ``destroyed_time`` can be a few seconds late for a natural expiry, so the
    lifetime always caps it. ``rules`` are the values of the ward's match
    (default: the current game).
    """
    rules = rules or _rules_for(None)
    max_dur = rules.observer_duration if ward.ward_type == "observer" else rules.sentry_duration
    expiry = ward.time + max_dur
    if ward.destroyed_time is not None:
        return min(ward.destroyed_time, expiry)
    return expiry


_MISS: Any = object()


class _MatchIndex:
    """Precomputed lookups + memo tables for one MatchData object."""

    def __init__(self, match: MatchData) -> None:
        self.fingerprint = _fingerprint(match)
        self.rules = _rules_for(match)
        self.players: dict[int, PlayerData] = {p.player_slot: p for p in match.players}
        # (day, night) vision of each hero, from the game data.
        self.hero_vision: dict[int, tuple[int, int]] = {
            p.player_slot: _hero_vision(p.hero_name) for p in match.players
        }
        self.teams: dict[str, list[PlayerData]] = {
            "radiant": [p for p in match.players if p.team == "radiant"],
            "dire": [p for p in match.players if p.team == "dire"],
        }
        self.duration = match.duration

        # Whole-game position log per slot: sample times and the rows
        # [time, x, y, hp_pct] (hp_pct 0 = dead, -1 = unknown).
        self.plog_times: dict[int, list[int]] = {}
        self.plog: dict[int, list[list[int]]] = {}
        for p in match.players:
            rows = sorted((r for r in p.position_log if len(r) >= 4), key=lambda r: r[0])
            if rows:
                self.plog[p.player_slot] = rows
                self.plog_times[p.player_slot] = [r[0] for r in rows]
        self.first_time = min(
            (times[0] for times in self.plog_times.values()), default=PREGAME_START)

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

        # Deaths per victim in time order: (time, x, y, certain); x/y are
        # 0.0 when the death point is unknown, certain is False when the
        # source cannot tell the death from a reincarnation. From the kill
        # log; from teamfight records only when the source has no kill log.
        deaths: dict[int, list[tuple[float, float, float, bool]]] = {}
        for k in match.kills_log:
            x, y = (float(k.x), float(k.y)) if k.x is not None and k.y is not None else _UNKNOWN_POS
            deaths.setdefault(k.victim_slot, []).append((float(k.time), x, y, bool(k.confirmed)))
        if not match.kills_log:
            for tf in match.teamfights:
                for tp in tf.players:
                    if tp.deaths > 0 and tp.death_time_offset is not None:
                        x, y = _fight_sample(tp, tp.death_time_offset) or (tp.x, tp.y)
                        deaths.setdefault(tp.player_slot, []).append(
                            (float(tf.start_time + tp.death_time_offset), x, y, True))
        self.deaths: dict[int, list[tuple[float, float, float, bool]]] = {
            slot: sorted(rows) for slot, rows in deaths.items()
        }
        self.death_times: dict[int, list[float]] = {
            slot: [r[0] for r in rows] for slot, rows in self.deaths.items()
        }
        self.buybacks: dict[int, list[float]] = {
            p.player_slot: sorted(float(b.get("time", 0) or 0) for b in p.buyback_log)
            for p in match.players if p.buyback_log
        }
        # Times of the enemy-hero kills each hero took part in (killing blow
        # or assist), sorted. Denies and unconfirmed entries are left out.
        kill_parts: dict[int, list[float]] = {}
        for k in match.kills_log:
            if k.deny or not k.confirmed:
                continue
            for part in {k.killer_slot, *k.assist_slots}:
                if part is not None and (part < 128) != (k.victim_slot < 128):
                    kill_parts.setdefault(part, []).append(float(k.time))
        self.kill_parts: dict[int, list[float]] = {
            slot: sorted(times) for slot, times in kill_parts.items()
        }

        self.towers = _build_towers(match)

        # Observer wards per team with their active interval [start, end).
        self.obs_wards: dict[str, list[tuple[int, WardEvent, int, int]]] = {"radiant": [], "dire": []}
        for idx, w in enumerate(match.wards):
            if w.ward_type != "observer":
                continue
            team = ward_team(w)
            if team not in self.obs_wards:
                continue
            self.obs_wards[team].append((idx, w, w.time, _ward_end(w, self.rules)))

        # Smoke of Deceit per hero: [start, end) while every hero of the
        # group was still smoked (``end_time`` is the first break; it can
        # never be later than the smoke's duration). A smoke without an end
        # (older data) hides nobody: how long it lasted is not known.
        self.smokes: dict[int, list[tuple[int, int]]] = {}
        for smoke in match.smokes or []:
            end = min(smoke.end_time, smoke.time + self.rules.smoke_duration)
            if end > smoke.time:
                for slot in smoke.player_slots:
                    self.smokes.setdefault(slot, []).append((smoke.time, end))

        # Running score after every counted kill, built on first use.
        self.score_times: list[float] | None = None
        self.scores: list[tuple[int, int]] = []

        self.pos_memo: dict[tuple[int, int], tuple[float, float] | None] = {}
        self.alive_memo: dict[tuple[int, int], bool | None] = {}
        self.source_memo: dict[tuple[str, int], list[tuple[float, float, float, str]]] = {}
        self.state_memo: dict[tuple[str, int, int], tuple[str, tuple[float, float] | None, str]] = {}
        self.visibility_memo: dict[tuple[str, int], list[EnemyVisibility]] = {}

    @staticmethod
    def _add_cast(casts: dict, ae: AbilityEvent, slot: int) -> None:
        if ae.x is None or ae.y is None or not has_position((ae.x, ae.y)):
            return
        key = (ae.time, ae.ability_name, round(ae.x, 1), round(ae.y, 1))
        casts.setdefault(slot, {}).setdefault(key, ae)

    def memo_size(self) -> int:
        return (len(self.pos_memo) + len(self.alive_memo) + len(self.source_memo)
                + len(self.state_memo) + len(self.visibility_memo))

    def clear_memos(self) -> None:
        self.pos_memo = {}
        self.alive_memo = {}
        self.source_memo = {}
        self.state_memo = {}
        self.visibility_memo = {}


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
                if idx.memo_size() > _MEMO_LIMIT:
                    idx.clear_memos()
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
# Alive state and position of a hero at time T
# ===================================================================

def respawn_seconds(level: int) -> int:
    """Base respawn time of a hero of ``level`` (buybacks / talents aside)."""
    return _RESPAWN_SECONDS[max(1, min(int(level), len(_RESPAWN_SECONDS))) - 1]


def _level_at(idx: _MatchIndex, slot: int, t: float) -> int:
    """Approximate hero level at ``t`` (per-minute XP, else a linear guess)."""
    player = idx.players.get(slot)
    if player is None:
        return 1
    if player.xp_t:
        from .phases import level_from_xp, series_at
        return level_from_xp(series_at(player.xp_t, int(max(0, t)) // 60))
    if idx.duration > 0:
        return max(1, round(player.level * max(0.0, t) / idx.duration))
    return max(1, player.level)


def _log_row(idx: _MatchIndex, slot: int, t: int) -> tuple[int, list[int]] | None:
    """(index, row) of the position-log sample describing second ``t``."""
    times = idx.plog_times.get(slot)
    if not times:
        return None
    i = bisect.bisect_right(times, t) - 1
    if i < 0 or t - times[i] > 2 * POSITION_LOG_STEP:
        return None  # before the first spawn / after the log ended
    return i, idx.plog[slot][i]


def _snapshot_before(idx: _MatchIndex, slot: int, t: int):
    """Laning snapshot at or just before ``t`` (None when there is none)."""
    times = idx.snap_times.get(slot)
    if not times:
        return None
    i = bisect.bisect_right(times, t) - 1
    if i < 0 or t - times[i] > _SNAPSHOT_MAX_GAP:
        return None
    return idx.snaps[slot][i]


def _alive(idx: _MatchIndex, slot: int, t: int) -> bool | None:
    """Whether a hero is alive at second ``t`` (None = the data cannot tell).

    A hero that dies during second ``t`` still counts as alive at ``t``.
    """
    key = (slot, t)
    cached = idx.alive_memo.get(key, _MISS)
    if cached is not _MISS:
        return cached
    result = _alive_uncached(idx, slot, t)
    idx.alive_memo[key] = result
    return result


def _alive_uncached(idx: _MatchIndex, slot: int, t: int) -> bool | None:
    times = idx.death_times.get(slot, ())
    j = bisect.bisect_left(times, t)  # first death at or after t
    if j < len(times) and times[j] < t + 1:
        return True  # dies within this very second
    last = idx.deaths[slot][j - 1] if j > 0 else None

    def bought_back(since: float) -> bool:
        return any(since <= b <= t for b in idx.buybacks.get(slot, ()))

    hit = _log_row(idx, slot, t)
    if hit is not None and hit[1][3] >= 0:
        row = hit[1]
        if last is not None and last[3] and last[0] >= row[0]:
            return bought_back(last[0])  # died after this sample, no newer one yet
        # A buyback after a sample that shows the corpse revives before the next one.
        return row[3] > 0 or (last is not None and bought_back(max(last[0], row[0])))

    snap = _snapshot_before(idx, slot, t)
    if snap is not None and snap.max_hp > 0 and (last is None or last[0] < snap.time):
        return snap.hp > 0
    if (last is not None and not bought_back(last[0])
            and t < last[0] + respawn_seconds(_level_at(idx, slot, last[0]))):
        return False if last[3] else None
    if snap is None and hit is None and not idx.deaths and slot not in idx.plog:
        return None  # no kill log, no death records, no samples: unknown
    return True


def _fight_sample(tp, offset: float) -> tuple[float, float] | None:
    """Real position sample of a teamfight record nearest to ``offset``."""
    if not tp.positions:
        return None
    best = min(tp.positions, key=lambda p: abs(p.time_offset - offset))
    return (best.x, best.y) if has_position((best.x, best.y)) else None


def _position_from_log(idx: _MatchIndex, slot: int, t: int) -> tuple[float, float] | None:
    """Position of a living hero from the position log (interpolated)."""
    hit = _log_row(idx, slot, t)
    if hit is None or hit[1][3] < 0:
        return None
    i, a = hit
    rows = idx.plog[slot]
    nxt = rows[i + 1] if i + 1 < len(rows) else None
    times = idx.death_times.get(slot, ())
    j = bisect.bisect_left(times, t)
    death = idx.deaths[slot][j] if j < len(times) else None  # next death at or after t
    if a[3] == 0:
        if death is not None and death[0] < t + 1 and has_position(death[1:3]):
            return death[1:3]  # the sample of the death second already shows the corpse
        # Respawned / bought back since this sample: the corpse position is
        # stale, the next sample is the first one alive.
        if nxt is not None and nxt[3] > 0 and has_position((nxt[1], nxt[2])):
            return (float(nxt[1]), float(nxt[2]))
        return None
    start = (float(a[0]), float(a[1]), float(a[2]))
    if not has_position(start[1:]):
        return None
    # Towards the next sample, or towards the death point when the hero dies
    # before it (the log then shows the corpse).
    end: tuple[float, float, float] | None = None
    if death is not None and (nxt is None or death[0] <= nxt[0]) and has_position(death[1:3]):
        end = death[:3]
    elif nxt is not None and nxt[3] > 0:
        end = (float(nxt[0]), float(nxt[1]), float(nxt[2]))
    if end is None or end[0] <= start[0] or not has_position(end[1:]):
        return start[1:]
    if _distance(start[1], start[2], end[1], end[2]) > _TELEPORT_DISTANCE:
        return start[1:] if t - start[0] <= end[0] - t else end[1:]
    ratio = (t - start[0]) / (end[0] - start[0])
    return (start[1] + (end[1] - start[1]) * ratio, start[2] + (end[2] - start[2]) * ratio)


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


def _position_from_deaths(idx: _MatchIndex, slot: int, t: int) -> tuple[float, float] | None:
    """The hero's own death point, for the few seconds before the death."""
    times = idx.death_times.get(slot, ())
    j = bisect.bisect_left(times, t)
    if j < len(times) and times[j] - t <= DEATH_POINT_MAX_AGE:
        death = idx.deaths[slot][j]
        if has_position(death[1:3]):
            return death[1:3]
    return None


def _position_from_fights(idx: _MatchIndex, slot: int, t: int) -> tuple[float, float] | None:
    """Position from a teamfight record, real samples only.

    ``TeamfightPlayer.x/y`` is the centre of the fight for a player without
    a sample of its own (OpenDota), so it is never used for a bystander.
    """
    for start, end, tp in idx.fights.get(slot, ()):
        if not (start - _TEAMFIGHT_MARGIN <= t <= end + _TEAMFIGHT_MARGIN):
            continue
        pos = _fight_sample(tp, t - start)
        if pos is not None:
            return pos
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


def _locate(idx: _MatchIndex, slot: int, t: int) -> tuple[float, float] | None:
    """Position of a hero at second ``t``; None when unknown or dead (memoized)."""
    key = (slot, t)
    cached = idx.pos_memo.get(key, _MISS)
    if cached is not _MISS:
        return cached
    pos: tuple[float, float] | None = None
    if _alive(idx, slot, t) is not False:
        if slot in idx.plog:
            pos = _position_from_log(idx, slot, t)
        else:
            pos = (
                _position_from_snapshots(idx, slot, t)
                or _position_from_deaths(idx, slot, t)
                or _position_from_fights(idx, slot, t)
                or _position_from_casts(idx, slot, t)
            )
    idx.pos_memo[key] = pos
    return pos


def hero_alive_at(match: MatchData, slot: int, t: float) -> bool | None:
    """Whether the hero in ``slot`` is alive at game second ``t``.

    Sources: the position log (hp_pct 0 = dead, which also covers Aegis and
    reincarnation deaths), else the kill log with the level-based respawn
    timer and the buyback log, else laning snapshots / teamfight death
    records. ``None`` means the data cannot tell (no death data at all, or an
    OpenDota death that may have been a reincarnation). A hero that dies
    during second ``t`` is still alive at ``t``.
    """
    return _alive(_index(match), slot, math.floor(t))


def position_at(match: MatchData, slot: int, t: float) -> tuple[float, float] | None:
    """World position of the hero in ``slot`` at game second ``t``.

    ``None`` when the position is unknown or the hero is dead (a dead hero
    has no position); never ``(0, 0)``. With a position log the 5 s samples
    are interpolated; without one only the seconds before the hero's own
    death, laning snapshots, real teamfight samples and fresh ability casts
    place it.
    """
    return _locate(_index(match), slot, math.floor(t))


def fight_participants(tf: TeamfightEvent) -> tuple[list[int], list[int]]:
    """(radiant slots, dire slots) that took part in a teamfight record.

    Taking part = dealt or took hero damage, healed, got a kill or died in
    the window. A cast alone does not count: the window is a time range
    over the whole map and heroes farming elsewhere cast spells too.
    """
    radiant: list[int] = []
    dire: list[int] = []
    for tp in tf.players:
        if _involved(tp):
            (radiant if tp.player_slot < 128 else dire).append(tp.player_slot)
    return radiant, dire


def hp_pct_at(match: MatchData, slot: int, t: float) -> int | None:
    """HP of a hero in percent (0-100) at second ``t``; None when unknown.

    The sample at or just before ``t`` is returned as it is (never
    interpolated towards a later 0-HP sample).
    """
    idx = _index(match)
    t = math.floor(t)
    hit = _log_row(idx, slot, t)
    if hit is not None:
        return int(hit[1][3]) if hit[1][3] >= 0 else None
    snap = _snapshot_before(idx, slot, t)
    if snap is not None and snap.max_hp > 0:
        return max(0, min(100, round(snap.hp / snap.max_hp * 100)))
    return None


def last_alive_time(match: MatchData, slot: int, death_time: float) -> int:
    """Second at which to sample a hero's state "before the death".

    The last position-log sample (else laning snapshot) with HP > 0 that
    lies at least ``_PRE_DEATH_GAP`` seconds before ``death_time``; sources
    without samples give the second before the death.
    """
    idx = _index(match)
    t = math.floor(death_time)
    limit = t - _PRE_DEATH_GAP
    times = idx.plog_times.get(slot)
    if times:
        rows = idx.plog[slot]
        i = bisect.bisect_right(times, limit) - 1
        while i >= 0 and t - rows[i][0] <= _PRE_DEATH_LOOKBACK:
            if rows[i][3] > 0:
                return rows[i][0]
            i -= 1
    snap_times = idx.snap_times.get(slot)
    if snap_times:
        snaps = idx.snaps[slot]
        i = bisect.bisect_right(snap_times, limit) - 1
        while i >= 0 and t - snaps[i].time <= _PRE_DEATH_LOOKBACK:
            if snaps[i].hp > 0 or snaps[i].max_hp <= 0:
                return snaps[i].time
            i -= 1
    return t - 1


def kills_before(match: MatchData, t: float) -> tuple[int, int] | None:
    """(radiant kills, dire kills) scored strictly before second ``t``.

    Follows the analyzer's kill feed (the same running score the match page
    shows): probable reincarnations, denies and deaths to neutrals do not
    score. ``None`` when the match has no kill log.
    """
    if not match.kills_log:
        return None
    idx = _index(match)
    if idx.score_times is None:
        idx.scores, idx.score_times = _running_score(match)
    i = bisect.bisect_left(idx.score_times, t)
    return idx.scores[i - 1] if i > 0 else (0, 0)


def _running_score(match: MatchData) -> tuple[list[tuple[int, int]], list[float]]:
    """(score after each kill-log entry, entry times), in time order."""
    try:
        from ..analyzer import _kill_feed, _slot_map
        feed = _kill_feed(match, _slot_map(match))
        return ([(int(e["radiant_score"]), int(e["dire_score"])) for e in feed],
                [float(e["time"]) for e in feed])
    except Exception as exc:  # the analyzer's feed is another module's internal
        logger.error("Kill feed unavailable, counting hero kills only: %s", exc)
    radiant = dire = 0
    scores: list[tuple[int, int]] = []
    times: list[float] = []
    for k in sorted(match.kills_log, key=lambda k: k.time):
        if k.confirmed and not k.deny and k.killer_slot is not None:
            if k.killer_slot < 128 <= k.victim_slot:
                radiant += 1
            elif k.victim_slot < 128 <= k.killer_slot:
                dire += 1
        scores.append((radiant, dire))
        times.append(float(k.time))
    return scores, times


def estimate_position(match: MatchData, player_slot: int, time_s: int) -> tuple[float, float]:
    """Position of a hero at ``time_s`` as (x, y); (0, 0) when unknown or dead.

    Older tuple form of :func:`position_at`.
    """
    return position_at(match, player_slot, time_s) or _UNKNOWN_POS


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
    rules = vision_rules(match)
    return [w for w in match.wards if w.time <= time_s < _ward_end(w, rules)]


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
    return _ward_end(ward)


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
        sources.append((w.x, w.y, idx.rules.observer_vision, "ward"))
    for name, tx, ty in _get_alive_towers(match, team, time_s):
        sources.append((tx, ty, tower_vision(name, night), "tower"))
    # The ancient and the fountain see their whole base (and stand until the
    # game ends). Barracks (900 / 600) stand well inside the ancient's
    # circle and are left out; activated Watchers are not in the data.
    for positions, radius in ((geo.ANCIENT_POSITIONS, ANCIENT_VISION), (geo.FOUNTAIN_POSITIONS, FOUNTAIN_VISION)):
        if team in positions:
            sources.append((positions[team][0], positions[team][1], radius, "tower"))
    for p in idx.teams.get(team, ()):
        pos = _locate(idx, p.player_slot, int(time_s))  # None for a dead hero
        if pos is None:
            continue
        day, dark = idx.hero_vision.get(p.player_slot, (HERO_DAY_VISION, HERO_NIGHT_VISION))
        hero_radius = dark if night else day
        if time_s < _LANING_END:
            hero_radius = max(hero_radius, LANE_CREEP_VISION)
        sources.append((pos[0], pos[1], hero_radius, "hero"))
    idx.source_memo[key] = sources
    return sources


def _hero_vision(hero_name: str) -> tuple[int, int]:
    """(day, night) vision of a hero from the game data (default 1800 / 800)."""
    try:
        from . import data_manager
        day, night = data_manager.hero_vision(hero_name)
        if day > 0 and night > 0:
            return int(day), int(night)
    except Exception as exc:  # pragma: no cover - hero_vision never raises
        logger.debug("Hero vision unavailable for %s: %s", hero_name, exc)
    return HERO_DAY_VISION, HERO_NIGHT_VISION


def is_smoked(match: MatchData, slot: int, time_s: int) -> bool:
    """Whether a hero is under Smoke of Deceit at ``time_s`` (as far as the data shows)."""
    return _smoked(_index(match), slot, int(time_s))


def _smoked(idx: _MatchIndex, slot: int, t: int) -> bool:
    return any(start <= t < end for start, end in idx.smokes.get(slot, ()))


def _smoke_breaker(x: float, y: float, sources, break_radius: float) -> str:
    """Source that sees a smoked hero at (x, y): a hero or building within
    its own vision and within the range that ends the smoke ("" = none).
    Wards never reveal a smoked hero."""
    for kind in ("tower", "hero"):
        if any(k == kind and _distance(x, y, sx, sy) <= min(sr, break_radius) for sx, sy, sr, k in sources):
            return kind
    return ""


def _seeing_source(x: float, y: float, sources) -> str:
    """Kind of the first source seeing (x, y): ward > tower > hero ("" = none)."""
    hits = {kind for sx, sy, sr, kind in sources if _distance(x, y, sx, sy) <= sr}
    for kind in ("ward", "tower", "hero"):
        if kind in hits:
            return kind
    return ""


# ===================================================================
# Enemy visibility tracking
# ===================================================================

# Earlier states are looked up on this global grid (the position-log step).
_STATE_STEP = POSITION_LOG_STEP


def _involved(tp) -> bool:
    return bool(tp.damage_dealt or tp.damage_taken or tp.kills or tp.deaths or tp.healing)


def _fight_record(idx: _MatchIndex, slot: int, time_s: int):
    """Teamfight record of a hero taking part in a fight going on at ``time_s``."""
    for start, end, tp in idx.fights.get(slot, ()):
        if start <= time_s <= end and _involved(tp):
            return tp
    return None


def _killing(idx: _MatchIndex, slot: int, time_s: int) -> bool:
    """Whether a hero gets a kill or an assist on an enemy hero within the
    next ``KILL_FIGHT_SECONDS`` (it is fighting that hero at ``time_s``)."""
    times = idx.kill_parts.get(slot)
    if not times:
        return False
    i = bisect.bisect_right(times, time_s - 1)  # first kill during or after this second
    return i < len(times) and times[i] <= time_s + KILL_FIGHT_SECONDS


def _enemy_state(
    match: MatchData, idx: _MatchIndex, team: str, slot: int, t: int,
) -> tuple[str, tuple[float, float] | None, str]:
    """(state, position, source) of one enemy hero as ``team`` sees it at ``t``.

    state is "visible", "fog" (position known, outside the team's vision),
    "unknown" (no position in the data) or "dead". A hero taking part in a
    teamfight, or about to kill an allied hero, is visible (source
    "teamfight") unless the data shows that no allied hero is anywhere near
    it: fighting reveals a hero, but a fight record covers the whole map and
    a kill can be a spell cast from far away. A hero under Smoke of Deceit
    is in the fog wherever it walks, until a hero or building of ``team`` is
    close enough to end the smoke.
    """
    key = (team, slot, t)
    cached = idx.state_memo.get(key)
    if cached is not None:
        return cached
    result: tuple[str, tuple[float, float] | None, str]
    if _alive(idx, slot, t) is False:
        result = ("dead", None, "")
    else:
        pos = _locate(idx, slot, t)
        fighting = _fight_record(idx, slot, t) is not None or _killing(idx, slot, t)
        if pos is None:
            result = ("visible", None, "teamfight") if fighting else ("unknown", None, "")
        else:
            sources = _team_sources(match, team, t)
            if fighting and (
                any(kind == "hero" and _distance(pos[0], pos[1], sx, sy) <= NEARBY_RADIUS_LATER
                    for sx, sy, _sr, kind in sources)
                # An ally the data cannot place may be the one it is fighting.
                or any(_alive(idx, p.player_slot, t) is not False and _locate(idx, p.player_slot, t) is None
                       for p in idx.teams.get(team, ()))
            ):
                result = ("visible", pos, "teamfight")
            else:
                kind = _seeing_source(pos[0], pos[1], sources)
                if kind and _smoked(idx, slot, t):
                    # Smoke of Deceit: invisible to wards and from afar.
                    kind = _smoke_breaker(pos[0], pos[1], sources, idx.rules.smoke_break_radius)
                result = ("visible", pos, kind) if kind else ("fog", pos, "")
    idx.state_memo[key] = result
    return result


def _look_back(
    match: MatchData, idx: _MatchIndex, team: str, slot: int, time_s: int,
    lookback: int = LAST_SEEN_LOOKBACK,
) -> tuple[int, tuple[int, tuple[float, float] | None, str] | None, bool]:
    """History of an enemy that is not visible at ``time_s``.

    Returns (fog_start, last_seen, unbroken): ``fog_start`` is the first
    second of the unbroken run of "fog" states that ends at ``time_s``;
    ``last_seen`` is (time, position, source) of the latest moment within
    ``lookback`` at which the enemy was visible, or None; ``unbroken`` says
    whether the enemy was in the fog all the way back to that moment (no
    death or gap in the data in between).
    """
    earliest = max(time_s - lookback, idx.first_time)
    fog_start = time_s
    unbroken = True
    t = time_s - 1 - ((time_s - 1) % _STATE_STEP)
    while t >= earliest:
        state, pos, source = _enemy_state(match, idx, team, slot, t)
        if state == "visible":
            return fog_start, (t, pos, source), unbroken
        if state == "fog" and unbroken:
            fog_start = t
        else:
            unbroken = False
        t -= _STATE_STEP
    return fog_start, None, unbroken


def _track_enemy_visibility(
    match: MatchData,
    team: str,
    player_slot: int,
    time_s: int,
) -> list[EnemyVisibility]:
    """Visibility of each enemy hero at ``time_s`` (memoized).

    ``EnemyVisibility.state`` separates "fog" (position known, outside the
    team's vision: the only state that counts as missing) from "unknown"
    (the data has no position) and "dead". ``seconds_missing`` is the
    measured length of the fog run, never a guess.
    """
    idx = _index(match)
    time_s = int(time_s)
    key = (team, time_s)  # the result does not depend on which teammate asks
    cached = idx.visibility_memo.get(key)
    if cached is not None:
        return cached

    results: list[EnemyVisibility] = []
    for enemy in idx.teams.get(_enemy_team(team), ()):
        slot = enemy.player_slot
        state, pos, source = _enemy_state(match, idx, team, slot, time_s)
        if state == "visible":
            # Known to be fighting but not where: no position is made up (the
            # record's point is the centre of all deaths of the fight).
            pos = pos or _UNKNOWN_POS
            results.append(EnemyVisibility(
                hero_name=enemy.hero_name, player_slot=slot, is_visible=True,
                last_seen_time=time_s, last_seen_x=pos[0], last_seen_y=pos[1],
                seconds_missing=0, last_seen_source=source, state="visible",
            ))
            continue
        if state == "dead":
            results.append(EnemyVisibility(
                hero_name=enemy.hero_name, player_slot=slot, is_visible=False,
                last_seen_time=0, seconds_missing=0, last_seen_source="", state="dead",
            ))
            continue
        fog_start, seen, unbroken = _look_back(match, idx, team, slot, time_s)
        seen_pos = (seen[1] if seen is not None else None) or _UNKNOWN_POS
        if state == "fog":
            # A sighting only counts when the hero was in the fog ever since;
            # otherwise the run is as long as the data shows it (source
            # "unknown", last_seen_time 0).
            sighted = seen is not None and unbroken
            since = seen[0] if sighted else fog_start
            if not sighted:
                seen_pos = _UNKNOWN_POS
            results.append(EnemyVisibility(
                hero_name=enemy.hero_name, player_slot=slot, is_visible=False,
                last_seen_time=since if sighted else 0,
                last_seen_x=seen_pos[0], last_seen_y=seen_pos[1],
                seconds_missing=max(0, time_s - since),
                last_seen_source=seen[2] if sighted else "unknown",
                state="fog",
            ))
        else:
            # No position data: not missing. The last sighting is kept as
            # information only.
            results.append(EnemyVisibility(
                hero_name=enemy.hero_name, player_slot=slot, is_visible=False,
                last_seen_time=seen[0] if seen is not None else 0,
                last_seen_x=seen_pos[0], last_seen_y=seen_pos[1], seconds_missing=0,
                last_seen_source=seen[2] if seen is not None else "unknown",
                state="unknown",
            ))
    idx.visibility_memo[key] = results
    return results


def _fog_seconds(seconds: int, lang: str) -> str:
    """A fog duration as text; a run as long as the lookback is "over N s"."""
    tx = _tx(lang)
    capped = seconds > LAST_SEEN_LOOKBACK - _STATE_STEP
    return tx["seconds_over" if capped else "seconds"].format(s=seconds)


def fog_threats(
    match: MatchData, enemy_visibility: list[EnemyVisibility],
    player_pos: tuple[float, float], time_s: int,
) -> list[EnemyVisibility]:
    """Enemies in the fog that are close enough to the player to matter.

    An enemy the replay shows further than ``THREAT_RADIUS`` away cannot
    take part in what happens to the player in the next seconds; it stays
    listed as missing but does not raise the danger. Without a player
    position every enemy in the fog counts. Enemies with unknown position
    are never included.
    """
    fog = [e for e in enemy_visibility if e.state == "fog"]
    if not has_position(player_pos):
        return fog
    idx = _index(match)
    threats: list[EnemyVisibility] = []
    for e in fog:
        pos = _locate(idx, e.player_slot, int(time_s))
        if pos is None or _distance(player_pos[0], player_pos[1], pos[0], pos[1]) <= THREAT_RADIUS:
            threats.append(e)
    return threats


# ===================================================================
# Danger assessment
# ===================================================================

def head_count(
    match: MatchData, slot: int, time_s: int, pos: tuple[float, float],
    tf: TeamfightEvent | None = None,
) -> tuple[str, int, int]:
    """Heroes on each side around a player: (basis, allies, enemies).

    The player is not counted. basis "nearby": living heroes with a known
    position within ``nearby_radius`` of ``pos``. When most positions are
    unknown (OpenDota) and the moment lies in teamfight ``tf``, basis is
    "participation": the heroes that took part in that fight record, which
    says nothing about how close they stood. Otherwise ("unknown", 0, 0).
    """
    idx = _index(match)
    t = int(time_s)
    player = idx.players.get(slot)
    if player is None:
        return ("unknown", 0, 0)
    known = unknown = 0
    for p in match.players:
        if p.player_slot == slot or _alive(idx, p.player_slot, t) is False:
            continue
        if _locate(idx, p.player_slot, t) is None:
            unknown += 1
        else:
            known += 1
    if has_position(pos) and known >= unknown:
        radius = nearby_radius(t)
        return (
            "nearby",
            count_nearby_heroes(match, player.team, pos[0], pos[1], t, exclude_slot=slot, radius=radius),
            count_nearby_heroes(match, _enemy_team(player.team), pos[0], pos[1], t, radius=radius),
        )
    if tf is not None:
        radiant, dire = fight_participants(tf)
        mine, theirs = (radiant, dire) if player.team == "radiant" else (dire, radiant)
        return ("participation", sum(1 for s in mine if s != slot), len(theirs))
    return ("unknown", 0, 0)


def count_nearby_heroes(
    match: MatchData, team: str, x: float, y: float, time_s: int,
    exclude_slot: int | None = None, radius: float | None = None,
) -> int:
    """Number of living ``team`` heroes with a known position within ``radius``."""
    if not has_position((x, y)):
        return 0
    if radius is None:
        radius = nearby_radius(time_s)
    idx = _index(match)
    count = 0
    for p in idx.teams.get(team, ()):
        if p.player_slot == exclude_slot:
            continue
        pos = _locate(idx, p.player_slot, int(time_s))
        if pos is not None and _distance(x, y, pos[0], pos[1]) <= radius:
            count += 1
    return count


def deaths_near(match: MatchData, slot: int, start: float, end: float) -> list[tuple[float, int]]:
    """(time, victim slot) of the other heroes that died near a player, in time order.

    Deaths with ``start < time < end`` whose death point lies within
    ``nearby_radius`` of where the player stood in that second. Needs the
    position log (the player's position while others die) and the kill log;
    entries that may be reincarnations are left out.
    """
    idx = _index(match)
    if slot not in idx.plog:
        return []
    found: list[tuple[float, int]] = []
    for victim, rows in idx.deaths.items():
        if victim == slot:
            continue
        for when, x, y, certain in rows:
            if not certain or not (start < when < end):
                continue
            second = math.floor(when)
            here = _locate(idx, slot, second)
            there = (x, y) if has_position((x, y)) else _locate(idx, victim, second)
            if here is None or there is None:
                continue
            if _distance(here[0], here[1], there[0], there[1]) <= nearby_radius(second):
                found.append((when, victim))
    found.sort()
    return found


class Engagement(NamedTuple):
    """The fight around a player up to one moment, judged from its first death.

    ``fallen_allies`` / ``fallen_enemies``: heroes that died near the player
    earlier in it. ``open_time`` is the second of the first of those deaths
    and ``open_allies`` / ``open_enemies`` the heroes near the player in that
    second (the player not counted; a hero still counts in the second it
    dies): the numbers the fight had before anyone was dead.
    ``opened_outnumbered`` says whether the player's side had fewer heroes
    then. A count that tips for a moment at the edge of the radius is not
    one, so it must also hold one position-log step earlier.
    """
    fallen_allies: int
    fallen_enemies: int
    open_time: int
    open_allies: int
    open_enemies: int
    opened_outnumbered: bool


_OPENING_CHECK_S = POSITION_LOG_STEP


def engagement(
    match: MatchData, slot: int, time_s: int, tf: TeamfightEvent | None = None,
) -> Engagement | None:
    """How the fight a player is in at ``time_s`` stood before heroes died in it.

    A head count at a player's last second alive leaves out everybody who
    died just before him: the last survivor of a 4v4 looks "outnumbered
    1v4". This looks at the hero deaths near the player
    (:func:`deaths_near`) since the start of teamfight ``tf`` or within the
    last ``ENGAGEMENT_LOOKBACK`` seconds, whichever reaches further back,
    never past the player's own previous death, and before his own death in
    second ``time_s`` (before ``time_s`` when he does not die in it).

    None when nobody died near the player in that window, or when the match
    has no position log.
    """
    idx = _index(match)
    player = idx.players.get(slot)
    if player is None or slot not in idx.plog:
        return None
    t = int(time_s)
    own = idx.death_times.get(slot, ())
    # Death times are truncated towards zero: a death at -10.3 is dated -10.
    j = bisect.bisect_right(own, t - 1)
    if j < len(own) and int(own[j]) == t:
        end = own[j]
    else:
        j = bisect.bisect_left(own, t)
        end = float(t)
    start = float(t - ENGAGEMENT_LOOKBACK)
    if tf is not None:
        start = min(start, float(tf.start_time))
    if j > 0:
        start = max(start, own[j - 1])
    fallen = deaths_near(match, slot, start, end)
    if not fallen:
        return None
    open_time = math.floor(fallen[0][0])

    def count(second: int) -> tuple[int, int] | None:
        here = _locate(idx, slot, second)
        if here is None:
            return None
        radius = nearby_radius(second)
        return (
            count_nearby_heroes(match, player.team, here[0], here[1], second, exclude_slot=slot, radius=radius),
            count_nearby_heroes(match, _enemy_team(player.team), here[0], here[1], second, radius=radius),
        )

    opening = count(open_time)
    if opening is None:  # cannot happen: deaths_near placed the player in that second
        return None
    before = count(open_time - _OPENING_CHECK_S) if open_time - _OPENING_CHECK_S > start else None
    victims = {victim for _when, victim in fallen}
    return Engagement(
        fallen_allies=sum(1 for victim in victims if _same_team(victim, slot)),
        fallen_enemies=sum(1 for victim in victims if not _same_team(victim, slot)),
        open_time=open_time,
        open_allies=opening[0],
        open_enemies=opening[1],
        opened_outnumbered=all(c[1] > c[0] + 1 for c in (opening, before) if c is not None),
    )


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
    reasons are localized sentences. Only enemies in the fog within reach of
    the player count (see :func:`fog_threats`); enemies without position
    data never do.
    """
    t = _tx(lang)
    reasons: list[str] = []
    known_pos = has_position(player_pos)

    missing = fog_threats(match, enemy_visibility, player_pos, time_s)
    missing_count = len(missing)
    longest_missing = max((e.seconds_missing for e in missing), default=0)

    unseen = _fog_seconds(longest_missing, lang)
    if missing_count >= 3:
        if longest_missing > 20:
            reasons.append(t["r_missing_long"].format(n=missing_count, s=unseen, r=THREAT_RADIUS))
        else:
            reasons.append(t["r_missing_many"].format(n=missing_count, r=THREAT_RADIUS))
    elif missing_count == 2:
        reasons.append(t["r_missing_two"].format(s=unseen, r=THREAT_RADIUS))
    elif missing_count == 1 and longest_missing > 30:
        reasons.append(t["r_missing_one"].format(hero=missing[0].hero_name, s=unseen, r=THREAT_RADIUS))

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
        coverage = vision_rules(match).observer_vision
        for w in _get_team_observer_wards(match, team, time_s):
            if _distance(player_pos[0], player_pos[1], w.x, w.y) <= coverage:
                has_ward_coverage = True
                break
        # Next to an own tower or in base a missing ward is no danger factor.
        if not has_ward_coverage and not player_in_safe_zone:
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
        has_ward_coverage=has_ward_coverage or not known_pos or player_in_safe_zone,
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
    missing = [e for e in enemy_vis if e.state == "fog"]
    unknown = [e for e in enemy_vis if e.state == "unknown"]
    threats = fog_threats(match, enemy_vis, player_pos, time_s)
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
        unknown_enemy_count=len(unknown),
        unknown_enemy_names=[e.hero_name for e in unknown],
        longest_missing_seconds=max((e.seconds_missing for e in missing), default=0),
        threat_enemy_count=len(threats),
        threat_enemy_names=[e.hero_name for e in threats],
        threat_longest_seconds=max((e.seconds_missing for e in threats), default=0),
        danger_level=danger_level,
        danger_reasons=danger_reasons,
        player_in_safe_zone=in_safe_zone,
        nearby_ward_coverage=ward_coverage,
    )


# ===================================================================
# Vision timeline
# ===================================================================

# A lost ward / tower further than this from the player is listed in the
# timeline but is not "the" vision mistake behind the decision.
_RELEVANT_VISION_RADIUS = 4000


def _timeline_samples(start: int, end: int) -> list[int]:
    """Sample times on the global TIMELINE_STEP grid, plus ``end`` itself."""
    first = start + (-start % TIMELINE_STEP)
    samples = list(range(first, end + 1, TIMELINE_STEP))
    if not samples or samples[-1] != end:
        samples.append(end)
    return samples


def _covered(
    match: MatchData, team: str, pos: tuple[float, float] | None, time_s: int,
) -> bool | None:
    """Whether a position is under own wards / towers or next to an own tower.

    None when the position is unknown.
    """
    if pos is None or not has_position(pos):
        return None
    if in_own_base(team, pos[0], pos[1]):
        return True
    if nearest_tower_distance(match, team, pos[0], pos[1], time_s) <= SAFE_TOWER_DISTANCE:
        return True
    sources = [src for src in _team_sources(match, team, time_s) if src[3] != "hero"]
    return _point_in_vision(pos[0], pos[1], sources)


def _allies_around(idx: _MatchIndex, team: str, player_slot: int, pos: tuple[float, float], t: int) -> int:
    """Living allied heroes within the "nearby" radius of ``pos`` at second ``t``."""
    radius = nearby_radius(t)
    count = 0
    for p in idx.teams.get(team, ()):
        if p.player_slot == player_slot:
            continue
        at = _locate(idx, p.player_slot, t)
        if at is not None and _distance(pos[0], pos[1], at[0], at[1]) <= radius:
            count += 1
    return count


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
    leaving/entering vision, day/night, towers lost, the player walking out
    of ward and tower vision - alone, or with ``GROUP_ALLIES`` allies as a
    "group_move"), and picks the most impactful one as the critical moment.
    Enemies without position data produce no events. The window may start
    before the horn.
    """
    tx = _tx(lang)
    idx = _index(match)
    events: list[VisionEvent] = []
    start_time = min(time_s, max(time_s - TIMELINE_WINDOW, idx.first_time))
    pre_time = start_time - TIMELINE_STEP

    def _player_at(t: int) -> tuple[float, float] | None:
        pos = _locate(idx, player_slot, t)
        if pos is None and t == time_s and has_position(player_pos):
            return player_pos
        return pos

    prev_wards = {i: w for i, w in _active_observers(match, team, pre_time)}
    prev_states = {
        e.player_slot: e.state for e in _track_enemy_visibility(match, team, player_slot, pre_time)
    }
    prev_night = _is_night(pre_time)
    # "Walked away from safety" needs both ends inside the window: covered
    # at some sample, then outside coverage and far from every own tower.
    was_covered = bool(_covered(match, team, _player_at(pre_time), pre_time))

    critical_moment = 0
    critical_description = ""
    critical_severity = 0
    warned_many_missing = False
    prev_t = pre_time

    def _critical(severity: int, when: int, text: str) -> None:
        nonlocal critical_moment, critical_description, critical_severity
        if severity > critical_severity:
            critical_severity, critical_moment, critical_description = severity, when, text

    def _relevant(x: float, y: float) -> bool:
        return (has_position(player_pos)
                and _distance(player_pos[0], player_pos[1], x, y) <= _RELEVANT_VISION_RADIUS)

    for t in _timeline_samples(start_time, time_s):
        enemy_vis = _track_enemy_visibility(match, team, player_slot, t)
        states = {e.player_slot: e.state for e in enemy_vis}
        here = _player_at(t)
        threats = fog_threats(match, enemy_vis, here or _UNKNOWN_POS, t)

        night = _is_night(t)
        if night != prev_night:
            if night:
                events.append(VisionEvent(time=t, event_type="night_start", description=tx["ev_night"]))
                if threats:
                    _critical(2, t, tx["crit_night"])
            else:
                events.append(VisionEvent(time=t, event_type="day_start", description=tx["ev_day"]))
        prev_night = night

        current_wards = {i: w for i, w in _active_observers(match, team, t)}
        for ward_idx in prev_wards.keys() - current_wards.keys():
            ward = prev_wards[ward_idx]
            area = describe_area(ward.x, ward.y, lang)
            kind = "ward_destroyed" if ward.dewarded else "ward_expired"
            replaced = any(
                _distance(ward.x, ward.y, w.x, w.y) <= vision_rules(match).observer_vision
                for _i, w in _active_observers(match, team, time_s))
            # A ward lost across the map says nothing about this decision.
            matters = _relevant(ward.x, ward.y) and not replaced
            events.append(VisionEvent(
                time=t, event_type=kind, description=tx[f"ev_{kind}"].format(area=area),
                important=matters,
            ))
            if matters:
                _critical(4, t, tx[f"crit_{kind}"].format(area=area))
        for ward_idx in current_wards.keys() - prev_wards.keys():
            ward = current_wards[ward_idx]
            events.append(VisionEvent(
                time=t, event_type="ward_placed",
                description=tx["ev_ward_placed"].format(area=describe_area(ward.x, ward.y, lang)),
            ))
        prev_wards = current_wards

        # Only a real change between "visible" and "fog" is an event: an
        # enemy whose data ends, or who dies / respawns, did not disappear.
        disappeared = False
        for info in enemy_vis:
            before = prev_states.get(info.player_slot)
            if before == "visible" and info.state == "fog":
                disappeared = True
                area = describe_area(info.last_seen_x, info.last_seen_y, lang)
                events.append(VisionEvent(
                    time=t, event_type="enemy_missing",
                    description=tx["ev_enemy_missing"].format(hero=info.hero_name, area=area),
                ))
            elif before == "fog" and info.state == "visible":
                area = describe_area(info.last_seen_x, info.last_seen_y, lang)
                events.append(VisionEvent(
                    time=t, event_type="enemy_seen",
                    description=tx["ev_enemy_seen"].format(hero=info.hero_name, area=area),
                ))

        long_missing = [e for e in threats if e.seconds_missing > 10]
        if len(long_missing) >= 3:
            _critical(5, t, tx["crit_many_missing"].format(n=len(long_missing)))
            if not warned_many_missing and (t == start_time or disappeared):
                events.append(VisionEvent(
                    time=t, event_type="enemy_missing",
                    description=tx["ev_many_missing"].format(n=len(long_missing)),
                ))
                warned_many_missing = True
        prev_states = states

        # Own towers falling inside this sample interval.
        for name, x, y, dt in idx.towers.get(team, ()):
            if prev_t < dt <= t:
                label = tower_label(name, lang)
                events.append(VisionEvent(
                    time=int(dt), event_type="tower_lost",
                    description=tx["ev_tower_lost"].format(tower=label),
                    important=_relevant(x, y),
                ))
                if _relevant(x, y):
                    _critical(3, int(dt), tx["crit_tower_lost"].format(tower=label))

        # The player walking away from wards and towers is an event at the
        # moment it happens (standing outside them all along is a state, and
        # stepping out of tower range in lane is nothing).
        covered = _covered(match, team, here, t)
        if covered:
            was_covered = True
        elif (covered is False and was_covered and nearest_tower_distance(
                match, team, here[0], here[1], t) > MODERATE_TOWER_DISTANCE):
            was_covered = False
            area = describe_area(here[0], here[1], lang)
            company = _allies_around(idx, team, player_slot, here, t)
            if company >= GROUP_ALLIES:
                # The team's move, not the player's: reported, never singled
                # out (live verdicts turned a four-hero push into "a support
                # farming deep in unwarded enemy territory").
                events.append(VisionEvent(
                    time=t, event_type="group_move",
                    description=tx["ev_group_move"].format(area=area, n=company),
                ))
            else:
                events.append(VisionEvent(
                    time=t, event_type="entered_fog",
                    description=tx["ev_entered_fog"].format(area=area), important=True,
                ))
                if len(threats) >= 2 and critical_severity < 3:
                    _critical(3, t, tx["crit_entered_fog"].format(area=area))
        prev_t = t

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

def _visibility_line(
    ev: EnemyVisibility, match: MatchData, time_s: int,
    player_pos: tuple[float, float], lang: str,
) -> str:
    """One line of the enemy visibility list."""
    tx = _tx(lang)
    seen_known = has_position((ev.last_seen_x, ev.last_seen_y))
    if ev.state == "dead":
        return tx["vis_dead"].format(hero=ev.hero_name)
    if ev.state == "visible" or (ev.state not in ("fog", "unknown") and ev.is_visible):
        if not seen_known:
            return tx["vis_visible_fight"].format(hero=ev.hero_name)
        return tx["vis_visible"].format(
            hero=ev.hero_name, area=describe_area(ev.last_seen_x, ev.last_seen_y, lang))
    seen = ""
    if seen_known and ev.last_seen_source not in ("", "unknown"):
        seen = tx["vis_seen"].format(
            area=describe_area(ev.last_seen_x, ev.last_seen_y, lang),
            t=_format_game_time(ev.last_seen_time))
    if ev.state == "unknown":
        return tx["vis_unknown"].format(hero=ev.hero_name) + seen
    line = tx["vis_fog"].format(hero=ev.hero_name, s=_fog_seconds(ev.seconds_missing, lang)) + seen
    if is_smoked(match, ev.player_slot, time_s):
        line += tx["vis_smoked"]
    # The replay knows where the hidden hero really was: say so, it tells
    # whether it could have been part of what happened next.
    real = position_at(match, ev.player_slot, time_s)
    if real is not None:
        area = describe_area(real[0], real[1], lang)
        if has_position(player_pos):
            line += tx["vis_real"].format(
                area=area, d=int(_distance(player_pos[0], player_pos[1], real[0], real[1])))
        else:
            line += tx["vis_real_nodist"].format(area=area)
    return line


def _format_vision_text(
    state: VisionState,
    timeline: VisionTimeline,
    time_s: int,
    match: MatchData,
    player_slot: int,
    player_pos: tuple[float, float],
    lang: str = "zh",
    died: bool = False,
) -> str:
    """Generate the formatted vision block for the LLM prompt.

    ``died`` says whether the decision point is the player's death; only
    then the text may state that the player failed to retreat.
    """
    tx = _tx(lang)
    # A source without position logs: the unknown enemies are not itemized
    # (four "no position data" lines read like four enemies missing).
    positionless = not _index(match).plog
    if positionless and state.unknown_enemy_count:
        danger = tx["danger_no_positions"]
    elif state.danger_level == "safe" and state.unknown_enemy_count:
        # "Safe" would be a statement about enemies nothing is known about.
        danger = tx["danger_unknown"].format(n=state.unknown_enemy_count)
    else:
        danger = tx["danger"].format(lvl=danger_level_label(state.danger_level, lang))
    lines: list[str] = [
        tx["h_vision"],
        tx["cur_time"].format(t=_format_game_time(time_s), dn=_day_night_label(time_s, lang)),
        tx["obs_count"].format(n=state.allied_observer_count),
        danger,
    ]
    if has_position(player_pos):
        lines.append(tx["player_pos"].format(area=describe_area(player_pos[0], player_pos[1], lang)))
    else:
        lines.append(tx["player_pos_unknown"])
    lines.append("")

    lines.append(tx["h_enemy_vis"])
    if state.enemy_visibility:
        for ev in state.enemy_visibility:
            if not (positionless and ev.state == "unknown"):
                lines.append(_visibility_line(ev, match, time_s, player_pos, lang))
        if state.unknown_enemy_count:
            lines.append(tx["vis_no_positions" if positionless else "vis_unknown_note"].format(
                n=state.unknown_enemy_count))
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
            flag = marker if evt.important else ""
            lines.append(f"[{_format_game_time(evt.time)}] {evt.description}{flag}")
        lines.append("")

    if timeline.critical_description:
        lines.append(tx["h_critical"])
        lines.append(tx["crit_line"].format(
            t=_format_game_time(timeline.critical_moment),
            s=time_s - timeline.critical_moment,
            desc=timeline.critical_description,
        ))
        # The enemies that mattered: hidden within reach of the player.
        in_fog = fog_threats(match, state.enemy_visibility, player_pos, time_s)
        if in_fog:
            heroes = tx["name_sep"].join(e.hero_name for e in in_fog[:3])
            suffix = (tx["crit_suffix"].format(n=len(in_fog), extra=len(in_fog) - 3)
                      if len(in_fog) > 3 else "")
            lines.append(tx["crit_after" if died else "crit_still"].format(heroes=heroes, suffix=suffix))
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
    *,
    died: bool = False,
) -> VisionAnalysis:
    """Complete vision analysis at a decision point.

    Args:
        match: Full match data.
        player_slot: The player being analyzed.
        time: Game second to analyze.
        player_pos: Player's position (world units) at this time; (0, 0) to
            let the module estimate it.
        lang: Language for all text ("zh" or "en").
        died: The decision point is the player's own death. Judgements
            about not retreating are only written for deaths.
    """
    time = int(time)
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
    formatted = _format_vision_text(state, timeline, time, match, player_slot, pos, lang=lang, died=died)
    return VisionAnalysis(state_at_decision=state, timeline=timeline, formatted_text=formatted)
