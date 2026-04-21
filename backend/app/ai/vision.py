"""Vision and fog-of-war analysis engine for AI coaching.

Given a MatchData, a player_slot, and a time T, this module calculates:
1. What the player's team CAN see (vision from wards, towers, allied heroes)
2. What enemies are visible vs missing (and for how long)
3. How dangerous the player's position is
4. A timeline of vision changes in the 60 seconds before T (for death analysis)

All text output is in Chinese (zh-CN) for direct injection into LLM prompts.
"""
from __future__ import annotations

import math

from ..models import (
    AbilityEvent,
    LaningSnapshot,
    MatchData,
    PlayerData,
    WardEvent,
)
from .models import (
    EnemyVisibility,
    VisionAnalysis,
    VisionEvent,
    VisionState,
    VisionTimeline,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Vision radii in game units
_OBS_WARD_RADIUS = 1600
_SENTRY_WARD_RADIUS = 1000       # true sight only, not ground vision
_OBS_WARD_DURATION = 360          # 6 minutes
_SENTRY_WARD_DURATION = 420       # 7 minutes

_HERO_DAY_VISION = 1800
_HERO_NIGHT_VISION = 800
_TOWER_DAY_VISION = 1900
_TOWER_NIGHT_VISION = 800

# Day/night cycle: 5 minute alternating periods, starting with day
_DAY_NIGHT_PERIOD = 300           # 5 minutes per half-cycle

# Timeline analysis window
_TIMELINE_WINDOW = 60             # seconds before decision to analyze
_TIMELINE_STEP = 5                # seconds between timeline samples

# Danger assessment thresholds
_SAFE_TOWER_DISTANCE = 1500       # "near tower" threshold
_MODERATE_TOWER_DISTANCE = 2500   # moderate distance
_FAR_TOWER_DISTANCE = 3500        # "far from tower" threshold
_NEARBY_ALLY_RADIUS = 2000        # radius to count nearby allies
_WARD_COVERAGE_RADIUS = 1600      # player is "covered" if within this of an obs ward

# Game coordinate bounds
_GAME_COORD_MIN = -4000
_GAME_COORD_MAX = 3875
_MAP_PCT_MIN = 0
_MAP_PCT_MAX = 100
_MAP_SCALE = 7937.5               # game range mapped to 0-100

# Approximate tower positions in game coordinates
TOWER_DATA: dict[str, dict[str, tuple[float, float]]] = {
    "radiant": {
        "top_t1": (-1544, 1286),
        "top_t2": (-3400, 900),
        "top_t3": (-4900, -1400),
        "mid_t1": (-1016, -926),
        "mid_t2": (-3200, -3400),
        "mid_t3": (-4600, -4400),
        "bot_t1": (4920, -3448),
        "bot_t2": (3000, -4400),
        "bot_t3": (-4800, -5500),
        "ancient_top": (-5420, -4300),
        "ancient_bot": (-4800, -5820),
    },
    "dire": {
        "top_t1": (-4856, 5806),
        "top_t2": (-3058, 4492),
        "top_t3": (4764, 3446),
        "mid_t1": (816, 706),
        "mid_t2": (3064, 3210),
        "mid_t3": (4592, 4282),
        "bot_t1": (1346, -1544),
        "bot_t2": (3414, -900),
        "bot_t3": (5050, 1628),
        "ancient_top": (4804, 5820),
        "ancient_bot": (5420, 4300),
    },
}

# Tower bitmask bit indices (LSB = index 0).
# Bit layout per team: t1_top(0), t2_top(1), t3_top(2),
#                       t1_mid(3), t2_mid(4), t3_mid(5),
#                       t1_bot(6), t2_bot(7), t3_bot(8),
#                       ancient_top(9), ancient_bot(10)
_TOWER_BIT_ORDER = [
    "top_t1", "top_t2", "top_t3",
    "mid_t1", "mid_t2", "mid_t3",
    "bot_t1", "bot_t2", "bot_t3",
    "ancient_top", "ancient_bot",
]

# Lane/area labels for approximate position description.
# Order matters: more specific areas should come first (smaller radii),
# with broader catch-all areas later.
_AREA_LABELS: list[tuple[float, float, float, str]] = [
    # (center_x, center_y, radius, label)
    # Specific landmarks
    (-3000, -5000, 1500, "Roshan坑附近"),
    # Lanes - top
    (-3500, 3500, 2000, "天辉上路"),
    (-4500, 5000, 2000, "夜魇上路"),
    (-1000, 4000, 2000, "夜魇上路河道"),
    # Lanes - mid
    (-500, -200, 1500, "中路"),
    (-1500, -800, 1500, "天辉中路"),
    (1200, 600, 1500, "夜魇中路"),
    # Lanes - bot
    (3500, -3000, 2000, "天辉下路"),
    (4500, 1500, 2000, "夜魇下路"),
    (1000, -3500, 2000, "天辉下路河道"),
    # River and rune areas
    (-2000, 1500, 1500, "上路河道"),
    (2000, -1500, 1500, "下路河道"),
    (0, 1200, 1500, "中路河道（夜魇侧）"),
    (0, -1200, 1500, "中路河道（天辉侧）"),
    # Jungle areas
    (-2500, -2500, 2000, "天辉野区"),
    (-1000, -3500, 2000, "天辉下方野区"),
    (2500, 2500, 2000, "夜魇野区"),
    (1000, 3500, 2000, "夜魇上方野区"),
    # Base areas
    (-4500, -4500, 2000, "天辉基地附近"),
    (4500, 4500, 2000, "夜魇基地附近"),
    # Broad catch-all quadrants (larger radii, checked last)
    (-2000, 2000, 3000, "地图左上方（上路方向）"),
    (2000, -2000, 3000, "地图右下方（下路方向）"),
    (-2000, -2000, 3000, "地图左下方（天辉侧）"),
    (2000, 2000, 3000, "地图右上方（夜魇侧）"),
]

_AREA_LABELS_EN: list[tuple[float, float, float, str]] = [
    (-3000, -5000, 1500, "near Roshan pit"),
    (-3500, 3500, 2000, "Radiant top lane"),
    (-4500, 5000, 2000, "Dire top lane"),
    (-1000, 4000, 2000, "Dire top river"),
    (-500, -200, 1500, "mid lane"),
    (-1500, -800, 1500, "Radiant mid lane"),
    (1200, 600, 1500, "Dire mid lane"),
    (3500, -3000, 2000, "Radiant bot lane"),
    (4500, 1500, 2000, "Dire bot lane"),
    (1000, -3500, 2000, "Radiant bot river"),
    (-2000, 1500, 1500, "top river"),
    (2000, -1500, 1500, "bot river"),
    (0, 1200, 1500, "mid river (Dire side)"),
    (0, -1200, 1500, "mid river (Radiant side)"),
    (-2500, -2500, 2000, "Radiant jungle"),
    (-1000, -3500, 2000, "Radiant lower jungle"),
    (2500, 2500, 2000, "Dire jungle"),
    (1000, 3500, 2000, "Dire upper jungle"),
    (-4500, -4500, 2000, "near Radiant base"),
    (4500, 4500, 2000, "near Dire base"),
    (-2000, 2000, 3000, "upper-left map (top lane area)"),
    (2000, -2000, 3000, "lower-right map (bot lane area)"),
    (-2000, -2000, 3000, "lower-left map (Radiant side)"),
    (2000, 2000, 3000, "upper-right map (Dire side)"),
]


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
        player_pos: Player's position in game coordinates at this time.
        lang: Language for formatted text ("zh" or "en").

    Returns:
        VisionAnalysis with state_at_decision, timeline, and formatted_text.
    """
    player = _find_player(match, player_slot)
    if player is None:
        # Cannot analyze without player data -- return empty analysis
        empty_state = VisionState(time=time, is_night=_is_night(time), allied_observer_count=0)
        if lang == "en":
            empty_text = "## Vision Analysis\nUnable to retrieve player data, skipping vision analysis.\n"
        else:
            empty_text = "## 视野分析\n无法获取玩家数据，跳过视野分析。\n"
        return VisionAnalysis(
            state_at_decision=empty_state,
            timeline=VisionTimeline(),
            formatted_text=empty_text,
        )

    team = player.team
    player_pos = _ensure_game_coords(player_pos)

    # If player_pos is origin (0,0) and we have better data, try to estimate
    if player_pos == (0.0, 0.0):
        player_pos = _estimate_player_position(player, time, match)

    # 1. Build vision state at the decision moment
    state = _build_vision_state(match, team, player_slot, time, player_pos)

    # 2. Build vision timeline for the window before the decision
    timeline = _build_vision_timeline(match, team, player_slot, time, player_pos)

    # 3. Format text for LLM prompt
    formatted = _format_vision_text(state, timeline, time, match, player_slot, player_pos, lang=lang)

    return VisionAnalysis(
        state_at_decision=state,
        timeline=timeline,
        formatted_text=formatted,
    )


# ===================================================================
# Coordinate utilities
# ===================================================================

def _is_map_percent(val: float) -> bool:
    """Check if a coordinate value is in map-percent space (0-100)."""
    return 0 <= val <= 100


def _map_pct_to_game(pct: float) -> float:
    """Convert a map-percent coordinate (0-100) to game coordinate."""
    return pct / 100.0 * _MAP_SCALE + _GAME_COORD_MIN


def _game_to_map_pct(game: float) -> float:
    """Convert a game coordinate to map-percent (0-100)."""
    return (game - _GAME_COORD_MIN) / _MAP_SCALE * 100.0


def _ensure_game_coords(pos: tuple[float, float]) -> tuple[float, float]:
    """Ensure position is in game coordinate space.

    If both x and y are in [0, 100] range, treat them as map-percent and convert.
    Otherwise assume they are already game coordinates.
    """
    x, y = pos
    if _is_map_percent(x) and _is_map_percent(y):
        # Could genuinely be game coords near origin. Use a heuristic:
        # map% values are typically not both near-zero unless it's actually origin.
        # If both are in (0,100] exclusive of a small region near 0, convert.
        if x > 5 or y > 5:
            return (_map_pct_to_game(x), _map_pct_to_game(y))
    return (x, y)


def _ward_pos_to_game(ward: WardEvent) -> tuple[float, float]:
    """Get ward position in game coordinates, converting from map% if needed."""
    return _ensure_game_coords((ward.x, ward.y))


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _describe_area(x: float, y: float, lang: str = "zh") -> str:
    """Get a human-readable label for a map position."""
    if lang == "en":
        labels = _AREA_LABELS_EN
        default = "unknown area"
    else:
        labels = _AREA_LABELS
        default = "未知区域"
    best_label = default
    best_dist = float("inf")
    for cx, cy, radius, label in labels:
        d = _distance(x, y, cx, cy)
        if d < radius and d < best_dist:
            best_dist = d
            best_label = label
    return best_label


# ===================================================================
# Day / Night cycle
# ===================================================================

def _is_night(time_s: int) -> bool:
    """Determine if it's nighttime at the given game second.

    Dota 2 simplified cycle:
    - 0:00 - 5:00 = Day (first day)
    - 5:00 - 10:00 = Night (first night)
    - Repeats every 10 minutes
    """
    if time_s < 0:
        return False
    return (time_s // _DAY_NIGHT_PERIOD) % 2 == 1


def _day_night_label(time_s: int, lang: str = "zh") -> str:
    """Return label for day/night state."""
    if _is_night(time_s):
        return "Night" if lang == "en" else "夜间"
    return "Day" if lang == "en" else "白天"


# ===================================================================
# Player / team utilities
# ===================================================================

def _find_player(match: MatchData, player_slot: int) -> PlayerData | None:
    """Find a player by slot."""
    for p in match.players:
        if p.player_slot == player_slot:
            return p
    return None


def _same_team(slot_a: int, slot_b: int) -> bool:
    """Check if two player slots are on the same team.

    Radiant slots: 0-4 (< 128), Dire slots: 128-132 (>= 128).
    """
    return (slot_a < 128) == (slot_b < 128)


def _get_team_players(match: MatchData, team: str) -> list[PlayerData]:
    """Get all players on a team."""
    return [p for p in match.players if p.team == team]


def _get_enemy_players(match: MatchData, team: str) -> list[PlayerData]:
    """Get all enemy players."""
    enemy_team = "dire" if team == "radiant" else "radiant"
    return [p for p in match.players if p.team == enemy_team]


def _get_allied_players(match: MatchData, team: str, exclude_slot: int) -> list[PlayerData]:
    """Get allied players excluding the analyzed player."""
    return [
        p for p in match.players
        if p.team == team and p.player_slot != exclude_slot
    ]


# ===================================================================
# Position estimation for heroes
# ===================================================================

def _estimate_player_position(
    player: PlayerData, time_s: int, match: MatchData,
) -> tuple[float, float]:
    """Best-effort position estimate for a player at a given time.

    Sources (in priority order):
    1. Laning snapshots (during laning phase)
    2. Teamfight positions
    3. Ability events with coordinates
    4. Default to (0, 0) if nothing available
    """
    # 1. Laning snapshots
    pos = _position_from_laning_snapshots(player.laning_snapshots, time_s)
    if pos is not None:
        return pos

    # 2. Teamfight positions
    pos = _position_from_teamfights(match, player.player_slot, time_s)
    if pos is not None:
        return pos

    # 3. Ability events (find nearest in time)
    pos = _position_from_abilities(player.ability_events, time_s)
    if pos is not None:
        return pos

    # 4. Match-level ability events
    pos = _position_from_abilities(
        [ae for ae in match.ability_events if ae.player_slot == player.player_slot],
        time_s,
    )
    if pos is not None:
        return pos

    return (0.0, 0.0)


def _position_from_laning_snapshots(
    snapshots: list[LaningSnapshot], time_s: int,
) -> tuple[float, float] | None:
    """Get position from laning snapshots if available and relevant."""
    if not snapshots:
        return None

    # Laning snapshots are for first 600s, but can still use the last one
    # if time is slightly beyond
    best: LaningSnapshot | None = None
    best_diff = float("inf")
    for snap in snapshots:
        diff = abs(snap.time - time_s)
        if diff < best_diff:
            best_diff = diff
            best = snap

    # Only use if reasonably close in time (within 30 seconds)
    if best is not None and best_diff <= 30:
        pos = _ensure_game_coords((best.x, best.y))
        if pos != (0.0, 0.0):
            return pos
    return None


def _position_from_teamfights(
    match: MatchData, player_slot: int, time_s: int,
) -> tuple[float, float] | None:
    """Get position from teamfight data if the player is in one near this time."""
    best_pos: tuple[float, float] | None = None
    best_diff = float("inf")

    for tf in match.teamfights:
        for tp in tf.players:
            if tp.player_slot != player_slot:
                continue
            # Check if teamfight overlaps or is close
            if tf.start_time <= time_s <= tf.end_time:
                offset = time_s - tf.start_time
                if tp.positions:
                    closest = min(tp.positions, key=lambda p: abs(p.time_offset - offset))
                    pos = _ensure_game_coords((closest.x, closest.y))
                    if pos != (0.0, 0.0):
                        return pos
                pos = _ensure_game_coords((tp.x, tp.y))
                if pos != (0.0, 0.0):
                    return pos

            # Also check fights near this time (within 30s)
            mid_time = (tf.start_time + tf.end_time) / 2
            diff = abs(mid_time - time_s)
            if diff < best_diff and diff <= 30:
                best_diff = diff
                pos = _ensure_game_coords((tp.x, tp.y))
                if pos != (0.0, 0.0):
                    best_pos = pos

    return best_pos


def _position_from_abilities(
    events: list[AbilityEvent], time_s: int, max_diff: int = 60,
) -> tuple[float, float] | None:
    """Get position from the nearest ability event with coordinates."""
    best_pos: tuple[float, float] | None = None
    best_diff = float("inf")

    for ae in events:
        if ae.x is None or ae.y is None:
            continue
        if ae.x == 0.0 and ae.y == 0.0:
            continue
        diff = abs(ae.time - time_s)
        if diff < best_diff and diff <= max_diff:
            best_diff = diff
            best_pos = _ensure_game_coords((ae.x, ae.y))

    return best_pos


def _estimate_all_hero_positions(
    match: MatchData, time_s: int,
) -> dict[int, tuple[float, float]]:
    """Estimate positions for all heroes at a given time.

    Returns a mapping of player_slot -> (x, y) in game coordinates.
    """
    positions: dict[int, tuple[float, float]] = {}
    for player in match.players:
        positions[player.player_slot] = _estimate_player_position(player, time_s, match)
    return positions


# ===================================================================
# Ward tracking
# ===================================================================

def _get_active_wards(match: MatchData, time_s: int) -> list[WardEvent]:
    """Get wards that are active (placed and not yet expired/destroyed) at time_s."""
    active: list[WardEvent] = []
    for w in match.wards:
        if w.time > time_s:
            continue  # not yet placed

        # Check if destroyed before this time
        if w.destroyed_time is not None and w.destroyed_time <= time_s:
            continue

        # Check natural expiry
        max_dur = _OBS_WARD_DURATION if w.ward_type == "observer" else _SENTRY_WARD_DURATION
        if w.destroyed_time is None and (time_s - w.time) > max_dur:
            continue

        active.append(w)
    return active


def _get_team_observer_wards(
    match: MatchData, team: str, time_s: int,
) -> list[WardEvent]:
    """Get active observer wards placed by the specified team."""
    wards = _get_active_wards(match, time_s)
    result: list[WardEvent] = []
    for w in wards:
        if w.ward_type != "observer":
            continue
        ward_team = "radiant" if w.player_slot < 128 else "dire"
        if ward_team == team:
            result.append(w)
    return result


def _ward_expiry_time(ward: WardEvent) -> int:
    """Calculate when a ward expires (destroyed or naturally)."""
    if ward.destroyed_time is not None:
        return ward.destroyed_time
    max_dur = _OBS_WARD_DURATION if ward.ward_type == "observer" else _SENTRY_WARD_DURATION
    return ward.time + max_dur


# ===================================================================
# Tower tracking
# ===================================================================

def _get_alive_towers(
    match: MatchData, team: str, time_s: int,
) -> list[tuple[str, float, float]]:
    """Get towers alive at a given time for a team.

    Returns list of (tower_name, x, y) tuples.

    Strategy:
    1. Start with all towers alive.
    2. Remove towers destroyed in objectives timeline before time_s.
    3. As a fallback validation, use final tower_status bitmask (but only
       if we can't determine timing from objectives).
    """
    if team not in TOWER_DATA:
        return []

    all_towers = set(TOWER_DATA[team].keys())
    destroyed_names = _get_destroyed_tower_names(match, team, time_s)
    alive_names = all_towers - destroyed_names

    result: list[tuple[str, float, float]] = []
    for name in alive_names:
        x, y = TOWER_DATA[team][name]
        result.append((name, x, y))
    return result


def _get_destroyed_tower_names(
    match: MatchData, team: str, time_s: int,
) -> set[str]:
    """Determine which towers have been destroyed before time_s.

    Parses objectives for tower kill events and maps unit names to our
    tower naming scheme.
    """
    destroyed: set[str] = set()

    for obj in match.objectives:
        obj_time = obj.get("time", 0)
        if obj_time > time_s:
            continue
        obj_type = obj.get("type", "")
        if obj_type not in ("tower", "building_kill", "CHAT_MESSAGE_TOWER_KILL"):
            continue

        unit = obj.get("unit", obj.get("key", "")).lower()
        tower_name = _parse_tower_unit_name(unit, team)
        if tower_name:
            destroyed.add(tower_name)

    # If no objectives available, fall back to final bitmask
    # (less accurate since it's end-of-game state, but better than nothing)
    if not match.objectives and time_s >= match.duration:
        destroyed.update(_destroyed_from_bitmask(match, team))

    return destroyed


def _parse_tower_unit_name(unit_str: str, team: str) -> str | None:
    """Try to map a unit string from objectives to a tower name.

    OpenDota/Clarity tower unit names vary but typically contain:
    - Team: "goodguys" (radiant), "badguys" (dire)
    - Lane: "top", "mid", "bot"
    - Tier: "tower1", "tower2", "tower3", or tier number

    Returns the tower name (e.g. "top_t1") or None if can't parse.
    """
    # Determine which team this tower belongs to
    is_radiant_tower = "goodguys" in unit_str or "radiant" in unit_str
    is_dire_tower = "badguys" in unit_str or "dire" in unit_str

    if team == "radiant" and not is_radiant_tower:
        return None
    if team == "dire" and not is_dire_tower:
        return None
    # If neither marker found, can't determine team
    if not is_radiant_tower and not is_dire_tower:
        return None

    # Determine lane
    lane = None
    if "top" in unit_str:
        lane = "top"
    elif "mid" in unit_str:
        lane = "mid"
    elif "bot" in unit_str:
        lane = "bot"

    # Determine tier
    tier = None
    if "tower1" in unit_str or "tier1" in unit_str:
        tier = "t1"
    elif "tower2" in unit_str or "tier2" in unit_str:
        tier = "t2"
    elif "tower3" in unit_str or "tier3" in unit_str:
        tier = "t3"
    elif "tower4" in unit_str or "ancient" in unit_str:
        # Ancient towers - need more context
        if "top" in unit_str:
            return "ancient_top"
        elif "bot" in unit_str:
            return "ancient_bot"
        # Default to ancient_top if can't distinguish
        return "ancient_top"

    if lane and tier:
        return f"{lane}_{tier}"
    return None


def _destroyed_from_bitmask(match: MatchData, team: str) -> set[str]:
    """Derive destroyed towers from final bitmask (end-of-game snapshot)."""
    bitmask = (
        match.tower_status_radiant if team == "radiant"
        else match.tower_status_dire
    )
    destroyed: set[str] = set()
    for i, name in enumerate(_TOWER_BIT_ORDER):
        if i >= 11:
            break
        if not (bitmask & (1 << i)):
            destroyed.add(name)
    return destroyed


# ===================================================================
# Vision source calculation
# ===================================================================

def _point_in_vision(
    px: float, py: float,
    sources: list[tuple[float, float, float]],
) -> bool:
    """Check if a point is within any vision source radius.

    Args:
        px, py: Point to check (game coordinates).
        sources: List of (x, y, radius) vision sources.

    Returns:
        True if the point is within at least one source's radius.
    """
    for sx, sy, sr in sources:
        if _distance(px, py, sx, sy) <= sr:
            return True
    return False


def _build_vision_sources(
    match: MatchData,
    team: str,
    player_slot: int,
    time_s: int,
    include_hero_vision: bool = True,
) -> list[tuple[float, float, float]]:
    """Build all vision sources for a team at a given time.

    Returns list of (x, y, radius) tuples in game coordinates.

    Vision sources:
    1. Observer wards (team's active observers)
    2. Towers (alive towers on the team)
    3. Allied heroes (if include_hero_vision is True)
    """
    night = _is_night(time_s)
    sources: list[tuple[float, float, float]] = []

    # 1. Observer wards (only observers provide ground vision)
    obs_wards = _get_team_observer_wards(match, team, time_s)
    for w in obs_wards:
        wx, wy = _ward_pos_to_game(w)
        sources.append((wx, wy, _OBS_WARD_RADIUS))

    # 2. Towers
    tower_radius = _TOWER_NIGHT_VISION if night else _TOWER_DAY_VISION
    alive_towers = _get_alive_towers(match, team, time_s)
    for _name, tx, ty in alive_towers:
        sources.append((tx, ty, tower_radius))

    # 3. Allied heroes
    if include_hero_vision:
        hero_radius = _HERO_NIGHT_VISION if night else _HERO_DAY_VISION
        hero_positions = _estimate_all_hero_positions(match, time_s)
        for p in match.players:
            if p.team != team:
                continue
            if p.player_slot == player_slot:
                continue  # skip self; we add player's own vision separately
            pos = hero_positions.get(p.player_slot, (0.0, 0.0))
            # Only use if we have a reasonable position
            if pos != (0.0, 0.0):
                sources.append((pos[0], pos[1], hero_radius))

    return sources


# ===================================================================
# Enemy visibility tracking
# ===================================================================

def _track_enemy_visibility(
    match: MatchData,
    team: str,
    player_slot: int,
    time_s: int,
) -> list[EnemyVisibility]:
    """Determine visibility status for each enemy hero at time_s.

    For each enemy, checks if their estimated position falls within any
    allied vision source. If not visible, works backwards to find when
    they were last seen.
    """
    enemies = _get_enemy_players(match, team)
    if not enemies:
        return []

    # Build vision sources for our team (including hero vision)
    # We need player_slot to exclude the analyzed player from hero sources
    # and add them separately with their actual position
    sources = _build_vision_sources(match, team, player_slot, time_s, include_hero_vision=True)

    # Add the analyzed player's own vision
    night = _is_night(time_s)
    hero_radius = _HERO_NIGHT_VISION if night else _HERO_DAY_VISION
    player = _find_player(match, player_slot)
    if player is not None:
        p_pos = _estimate_player_position(player, time_s, match)
        if p_pos != (0.0, 0.0):
            sources.append((p_pos[0], p_pos[1], hero_radius))

    hero_positions = _estimate_all_hero_positions(match, time_s)
    results: list[EnemyVisibility] = []

    for enemy in enemies:
        enemy_pos = hero_positions.get(enemy.player_slot, (0.0, 0.0))

        # If we have no position estimate for the enemy, they're "missing"
        # with unknown last-seen info
        if enemy_pos == (0.0, 0.0):
            last_seen_time, last_seen_pos, last_seen_src = _find_last_seen(
                match, team, player_slot, enemy, time_s,
            )
            missing_secs = (time_s - last_seen_time) if last_seen_time > 0 else time_s
            results.append(EnemyVisibility(
                hero_name=enemy.hero_name,
                player_slot=enemy.player_slot,
                is_visible=False,
                last_seen_time=last_seen_time,
                last_seen_x=last_seen_pos[0],
                last_seen_y=last_seen_pos[1],
                seconds_missing=missing_secs,
                last_seen_source=last_seen_src,
            ))
            continue

        # Check if enemy position is within allied vision
        visible = _point_in_vision(enemy_pos[0], enemy_pos[1], sources)

        if visible:
            # Determine which source sees them
            source_type = _identify_vision_source(
                enemy_pos[0], enemy_pos[1], match, team, player_slot, time_s,
            )
            results.append(EnemyVisibility(
                hero_name=enemy.hero_name,
                player_slot=enemy.player_slot,
                is_visible=True,
                last_seen_time=time_s,
                last_seen_x=enemy_pos[0],
                last_seen_y=enemy_pos[1],
                seconds_missing=0,
                last_seen_source=source_type,
            ))
        else:
            # Not visible -- find last seen time
            last_seen_time, last_seen_pos, last_seen_src = _find_last_seen(
                match, team, player_slot, enemy, time_s,
            )
            missing_secs = (time_s - last_seen_time) if last_seen_time > 0 else time_s
            results.append(EnemyVisibility(
                hero_name=enemy.hero_name,
                player_slot=enemy.player_slot,
                is_visible=False,
                last_seen_time=last_seen_time,
                last_seen_x=last_seen_pos[0],
                last_seen_y=last_seen_pos[1],
                seconds_missing=missing_secs,
                last_seen_source=last_seen_src,
            ))

    return results


def _find_last_seen(
    match: MatchData,
    team: str,
    player_slot: int,
    enemy: PlayerData,
    time_s: int,
    lookback: int = 120,
) -> tuple[int, tuple[float, float], str]:
    """Find when an enemy was last visible to the team before time_s.

    Works backwards from time_s checking multiple data sources.

    Returns:
        (last_seen_time, (x, y), source_type) where source_type is one of
        "ward", "tower", "hero", "ability", "laning", "teamfight", "unknown".
    """
    best_time = 0
    best_pos = (0.0, 0.0)
    best_source = "unknown"

    start_check = max(0, time_s - lookback)

    # Source 1: Laning snapshots (check if within vision at each snapshot)
    if enemy.laning_snapshots:
        for snap in reversed(enemy.laning_snapshots):
            if snap.time > time_s:
                continue
            if snap.time < start_check:
                break
            enemy_pos = _ensure_game_coords((snap.x, snap.y))
            if enemy_pos == (0.0, 0.0):
                continue
            sources = _build_vision_sources(
                match, team, player_slot, snap.time, include_hero_vision=True,
            )
            if _point_in_vision(enemy_pos[0], enemy_pos[1], sources):
                if snap.time > best_time:
                    best_time = snap.time
                    best_pos = enemy_pos
                    best_source = "laning"
                break  # laning snapshots are chronological; first hit going backwards is latest

    # Source 2: Teamfight positions
    for tf in reversed(match.teamfights):
        if tf.end_time > time_s:
            continue
        if tf.start_time < start_check:
            break
        for tp in tf.players:
            if tp.player_slot == enemy.player_slot:
                tf_pos = _ensure_game_coords((tp.x, tp.y))
                if tf_pos != (0.0, 0.0) and tf.end_time > best_time:
                    best_time = tf.end_time
                    best_pos = tf_pos
                    best_source = "teamfight"

    # Source 3: Ability events (enemy casting abilities reveals them)
    all_enemy_abilities = [
        ae for ae in match.ability_events
        if ae.player_slot == enemy.player_slot
    ]
    # Also check the player-level ability events
    all_enemy_abilities.extend(
        ae for ae in enemy.ability_events
        if ae not in all_enemy_abilities
    )
    for ae in reversed(all_enemy_abilities):
        if ae.time > time_s:
            continue
        if ae.time < start_check:
            break
        if ae.x is not None and ae.y is not None and not (ae.x == 0 and ae.y == 0):
            ae_pos = _ensure_game_coords((ae.x, ae.y))
            if ae.time > best_time:
                best_time = ae.time
                best_pos = ae_pos
                best_source = "ability"

    # Source 4: Stepped check at regular intervals (for hero/ward/tower vision)
    # Only if we didn't find anything better
    if best_time < start_check:
        step = 5 if time_s < 600 else 15
        for t in range(time_s - step, start_check - 1, -step):
            if t < 0:
                break
            enemy_pos_at_t = _estimate_player_position(enemy, t, match)
            if enemy_pos_at_t == (0.0, 0.0):
                continue
            sources = _build_vision_sources(
                match, team, player_slot, t, include_hero_vision=True,
            )
            if _point_in_vision(enemy_pos_at_t[0], enemy_pos_at_t[1], sources):
                src = _identify_vision_source(
                    enemy_pos_at_t[0], enemy_pos_at_t[1],
                    match, team, player_slot, t,
                )
                if t > best_time:
                    best_time = t
                    best_pos = enemy_pos_at_t
                    best_source = src
                break

    return (best_time, best_pos, best_source)


def _identify_vision_source(
    x: float, y: float,
    match: MatchData, team: str, player_slot: int, time_s: int,
) -> str:
    """Identify the type of vision source that can see a point.

    Priority: ward > tower > hero (returns the first matching).
    """
    night = _is_night(time_s)

    # Check observer wards first (most important)
    obs_wards = _get_team_observer_wards(match, team, time_s)
    for w in obs_wards:
        wx, wy = _ward_pos_to_game(w)
        if _distance(x, y, wx, wy) <= _OBS_WARD_RADIUS:
            return "ward"

    # Check towers
    tower_radius = _TOWER_NIGHT_VISION if night else _TOWER_DAY_VISION
    alive_towers = _get_alive_towers(match, team, time_s)
    for _name, tx, ty in alive_towers:
        if _distance(x, y, tx, ty) <= tower_radius:
            return "tower"

    # Check allied heroes
    hero_radius = _HERO_NIGHT_VISION if night else _HERO_DAY_VISION
    hero_positions = _estimate_all_hero_positions(match, time_s)
    for p in match.players:
        if p.team != team:
            continue
        pos = hero_positions.get(p.player_slot, (0.0, 0.0))
        if pos != (0.0, 0.0) and _distance(x, y, pos[0], pos[1]) <= hero_radius:
            return "hero"

    return "unknown"


# ===================================================================
# Danger assessment
# ===================================================================

def _assess_danger(
    match: MatchData,
    team: str,
    player_slot: int,
    time_s: int,
    player_pos: tuple[float, float],
    enemy_visibility: list[EnemyVisibility],
) -> tuple[str, list[str], bool, bool]:
    """Assess how dangerous the player's current position is.

    Returns:
        (danger_level, reasons, player_in_safe_zone, has_ward_coverage)
        where danger_level is "safe", "caution", "danger", or "critical"
        and reasons is a list of Chinese explanation strings.
    """
    reasons: list[str] = []

    # --- Count missing enemies ---
    missing = [e for e in enemy_visibility if not e.is_visible]
    missing_count = len(missing)
    longest_missing = max((e.seconds_missing for e in missing), default=0)

    if missing_count >= 3:
        if longest_missing > 20:
            reasons.append(
                f"{missing_count}名敌方英雄已消失超过{longest_missing}秒"
            )
        else:
            reasons.append(f"{missing_count}名敌方英雄消失在迷雾中")
    elif missing_count == 2:
        reasons.append(f"2名敌方英雄消失在迷雾中（最长{longest_missing}秒）")
    elif missing_count == 1 and longest_missing > 30:
        reasons.append(
            f"{missing[0].hero_name} 已消失{longest_missing}秒"
        )

    # --- Distance from nearest allied tower ---
    alive_towers = _get_alive_towers(match, team, time_s)
    min_tower_dist = float("inf")
    if alive_towers and player_pos != (0.0, 0.0):
        for _name, tx, ty in alive_towers:
            d = _distance(player_pos[0], player_pos[1], tx, ty)
            min_tower_dist = min(min_tower_dist, d)

    player_in_safe_zone = min_tower_dist <= _SAFE_TOWER_DISTANCE

    if min_tower_dist > _FAR_TOWER_DISTANCE and player_pos != (0.0, 0.0):
        reasons.append(f"距离最近友方防御塔约{int(min_tower_dist)}距离")
    elif min_tower_dist > _MODERATE_TOWER_DISTANCE and player_pos != (0.0, 0.0):
        reasons.append(f"与友方防御塔距离较远（约{int(min_tower_dist)}距离）")

    # --- Ward coverage ---
    has_ward_coverage = False
    if player_pos != (0.0, 0.0):
        obs_wards = _get_team_observer_wards(match, team, time_s)
        for w in obs_wards:
            wx, wy = _ward_pos_to_game(w)
            if _distance(player_pos[0], player_pos[1], wx, wy) <= _WARD_COVERAGE_RADIUS:
                has_ward_coverage = True
                break

    if not has_ward_coverage and player_pos != (0.0, 0.0):
        reasons.append("当前区域没有观察守卫覆盖")

    # --- Nearby allies ---
    nearby_allies = 0
    if player_pos != (0.0, 0.0):
        hero_positions = _estimate_all_hero_positions(match, time_s)
        for p in match.players:
            if p.team != team or p.player_slot == player_slot:
                continue
            ally_pos = hero_positions.get(p.player_slot, (0.0, 0.0))
            if ally_pos != (0.0, 0.0):
                if _distance(player_pos[0], player_pos[1], ally_pos[0], ally_pos[1]) <= _NEARBY_ALLY_RADIUS:
                    nearby_allies += 1

    if nearby_allies == 0 and missing_count >= 2:
        reasons.append("附近没有友方英雄支援")

    # --- Night penalty ---
    night = _is_night(time_s)
    if night and missing_count >= 2:
        reasons.append("夜间视野缩小，更容易被突袭")

    # --- Determine danger level ---
    danger_level = _compute_danger_level(
        missing_count=missing_count,
        longest_missing=longest_missing,
        min_tower_dist=min_tower_dist,
        has_ward_coverage=has_ward_coverage,
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
    """Compute the danger level string based on factors.

    Returns: "safe", "caution", "danger", or "critical"
    """
    score = 0

    # Missing enemies contribution
    if missing_count >= 4:
        score += 4
    elif missing_count >= 3:
        score += 3
    elif missing_count >= 2:
        score += 2
    elif missing_count >= 1:
        score += 1

    # Duration missing
    if longest_missing > 30:
        score += 2
    elif longest_missing > 20:
        score += 1

    # Tower distance
    if min_tower_dist > _FAR_TOWER_DISTANCE:
        score += 2
    elif min_tower_dist > _MODERATE_TOWER_DISTANCE:
        score += 1

    # Ward coverage
    if not has_ward_coverage:
        score += 1

    # Allies nearby (mitigating)
    if nearby_allies >= 2:
        score -= 2
    elif nearby_allies >= 1:
        score -= 1

    # Night penalty
    if is_night:
        score += 1

    # Map score to danger level
    if score <= 2:
        return "safe"
    elif score <= 4:
        return "caution"
    elif score <= 6:
        return "danger"
    else:
        return "critical"


# ===================================================================
# Vision state construction
# ===================================================================

def _build_vision_state(
    match: MatchData,
    team: str,
    player_slot: int,
    time_s: int,
    player_pos: tuple[float, float],
) -> VisionState:
    """Build a complete VisionState at a specific time."""
    night = _is_night(time_s)

    # Active observer wards for our team
    obs_wards = _get_team_observer_wards(match, team, time_s)

    # Enemy visibility
    enemy_vis = _track_enemy_visibility(match, team, player_slot, time_s)

    # Missing enemies summary
    missing = [e for e in enemy_vis if not e.is_visible]
    missing_count = len(missing)
    missing_names = [e.hero_name for e in missing]
    longest_missing = max((e.seconds_missing for e in missing), default=0)

    # Danger assessment
    danger_level, danger_reasons, in_safe_zone, ward_coverage = _assess_danger(
        match, team, player_slot, time_s, player_pos, enemy_vis,
    )

    return VisionState(
        time=time_s,
        is_night=night,
        allied_observer_count=len(obs_wards),
        enemy_visibility=enemy_vis,
        missing_enemy_count=missing_count,
        missing_enemy_names=missing_names,
        longest_missing_seconds=longest_missing,
        danger_level=danger_level,
        danger_reasons=danger_reasons,
        player_in_safe_zone=in_safe_zone,
        nearby_ward_coverage=ward_coverage,
    )


# ===================================================================
# Vision timeline
# ===================================================================

def _build_vision_timeline(
    match: MatchData,
    team: str,
    player_slot: int,
    time_s: int,
    player_pos: tuple[float, float],
) -> VisionTimeline:
    """Build a vision event timeline for the window before the decision.

    Samples every _TIMELINE_STEP seconds from (time_s - _TIMELINE_WINDOW)
    to time_s and detects significant vision changes.
    """
    events: list[VisionEvent] = []
    start_time = max(0, time_s - _TIMELINE_WINDOW)

    # Initialize tracking state from just BEFORE the window begins so that
    # we only report *changes* within the window, not pre-existing state.
    pre_window_time = max(0, start_time - _TIMELINE_STEP)
    _pre_obs = _get_team_observer_wards(match, team, pre_window_time)
    prev_ward_ids: set[int] = {w.time for w in _pre_obs}

    _pre_vis = _track_enemy_visibility(match, team, player_slot, pre_window_time)
    prev_visible_enemies: set[int] = {
        e.player_slot for e in _pre_vis if e.is_visible
    }

    prev_night: bool | None = _is_night(pre_window_time)

    # Track the most impactful vision event for critical_moment
    critical_moment = 0
    critical_description = ""
    critical_severity = 0  # higher = more critical

    for t in range(start_time, time_s + 1, _TIMELINE_STEP):
        time_label = _format_game_time(t)
        night = _is_night(t)

        # --- Day/night transition ---
        if prev_night is not None and night != prev_night:
            if night:
                evt = VisionEvent(
                    time=t,
                    event_type="night_start",
                    description=f"进入夜间，所有单位视野缩小",
                )
                events.append(evt)
                # Night transition with missing enemies is concerning
                if len(prev_visible_enemies) < 5:
                    severity = 2
                    if severity > critical_severity:
                        critical_severity = severity
                        critical_moment = t
                        critical_description = "夜间开始，视野缩小增加被gank风险"
            else:
                events.append(VisionEvent(
                    time=t,
                    event_type="day_start",
                    description=f"进入白天，视野恢复正常",
                ))
        prev_night = night

        # --- Ward changes ---
        current_obs = _get_team_observer_wards(match, team, t)
        current_ward_ids = {w.time for w in current_obs}

        # Wards that expired or were destroyed
        lost_wards = prev_ward_ids - current_ward_ids
        for ward_time_id in lost_wards:
            # Find the ward that was lost
            ward = _find_ward_by_time(match, team, ward_time_id)
            if ward is not None:
                wx, wy = _ward_pos_to_game(ward)
                area = _describe_area(wx, wy)
                was_destroyed = (
                    ward.destroyed_time is not None
                    and ward.destroyed_time <= t
                    and ward.destroyed_time > t - _TIMELINE_STEP
                )
                if was_destroyed:
                    evt = VisionEvent(
                        time=t,
                        event_type="ward_destroyed",
                        description=f"{area}的观察守卫被摧毁，视野丢失",
                    )
                else:
                    evt = VisionEvent(
                        time=t,
                        event_type="ward_expired",
                        description=f"{area}的观察守卫到期，视野丢失",
                    )
                events.append(evt)

                # Ward loss is significant
                severity = 4
                if severity > critical_severity:
                    critical_severity = severity
                    critical_moment = t
                    critical_description = (
                        f"{area}观察守卫{'被摧毁' if was_destroyed else '到期'}后"
                        f"未补放新眼，导致完全失去该区域敌方动向"
                    )

        # Wards that were newly placed
        new_wards = current_ward_ids - prev_ward_ids
        for ward_time_id in new_wards:
            ward = _find_ward_by_time(match, team, ward_time_id)
            if ward is not None:
                wx, wy = _ward_pos_to_game(ward)
                area = _describe_area(wx, wy)
                events.append(VisionEvent(
                    time=t,
                    event_type="ward_placed",
                    description=f"在{area}放置了观察守卫",
                ))

        prev_ward_ids = current_ward_ids

        # --- Enemy visibility changes ---
        enemy_vis = _track_enemy_visibility(match, team, player_slot, t)
        current_visible = {e.player_slot for e in enemy_vis if e.is_visible}

        # Enemies that disappeared
        disappeared = prev_visible_enemies - current_visible
        for slot in disappeared:
            enemy_info = _find_enemy_vis(enemy_vis, slot)
            if enemy_info is not None:
                area = _describe_area(enemy_info.last_seen_x, enemy_info.last_seen_y)
                events.append(VisionEvent(
                    time=t,
                    event_type="enemy_missing",
                    description=f"{enemy_info.hero_name} 从{area}消失",
                ))

        # Enemies that appeared
        appeared = current_visible - prev_visible_enemies
        for slot in appeared:
            enemy_info = _find_enemy_vis(enemy_vis, slot)
            if enemy_info is not None:
                area = _describe_area(enemy_info.last_seen_x, enemy_info.last_seen_y)
                events.append(VisionEvent(
                    time=t,
                    event_type="enemy_seen",
                    description=f"{enemy_info.hero_name} 出现在{area}",
                ))

        # Check for multiple enemies disappearing simultaneously
        total_missing = len([e for e in enemy_vis if not e.is_visible])
        long_missing = [e for e in enemy_vis if not e.is_visible and e.seconds_missing > 10]
        if len(long_missing) >= 3:
            severity = 5
            if severity > critical_severity:
                critical_severity = severity
                critical_moment = t
                critical_description = (
                    f"{len(long_missing)}名敌方英雄同时消失超过10秒，形势危险"
                )
            # Add a warning event (only once around this time)
            if t == start_time or len(disappeared) > 0:
                events.append(VisionEvent(
                    time=t,
                    event_type="enemy_missing",
                    description=f"⚠️ {len(long_missing)}名敌方消失超过10秒，形势危险",
                ))

        prev_visible_enemies = current_visible

        # --- Tower destruction during window ---
        for obj in match.objectives:
            obj_time = obj.get("time", 0)
            obj_type = obj.get("type", "")
            if (t - _TIMELINE_STEP < obj_time <= t
                    and obj_type in ("tower", "building_kill", "CHAT_MESSAGE_TOWER_KILL")):
                unit = obj.get("unit", obj.get("key", ""))
                obj_team = obj.get("team", "")
                # Check if it's our team's tower (vision loss)
                tower_name = _parse_tower_unit_name(unit.lower() if unit else "", team)
                if tower_name:
                    events.append(VisionEvent(
                        time=obj_time,
                        event_type="tower_lost",
                        description=f"友方防御塔被摧毁，失去该区域固定视野",
                    ))
                    severity = 3
                    if severity > critical_severity:
                        critical_severity = severity
                        critical_moment = obj_time
                        critical_description = "友方防御塔被摧毁，失去重要固定视野"

    # --- Check if player entered fog area ---
    if player_pos != (0.0, 0.0):
        sources = _build_vision_sources(match, team, player_slot, time_s, include_hero_vision=False)
        player_in_fog = not _point_in_vision(player_pos[0], player_pos[1], sources)
        if player_in_fog:
            area = _describe_area(player_pos[0], player_pos[1])
            events.append(VisionEvent(
                time=time_s,
                event_type="entered_fog",
                description=f"玩家进入无视野覆盖的{area}",
            ))
            if critical_severity < 3:
                critical_severity = 3
                critical_moment = time_s
                critical_description = f"玩家进入没有视野覆盖的危险区域（{area}）"

    # Sort events by time
    events.sort(key=lambda e: e.time)

    # Build summary
    summary = _build_timeline_summary(events, time_s)

    return VisionTimeline(
        events=events,
        critical_moment=critical_moment,
        critical_description=critical_description,
        summary=summary,
    )


def _find_ward_by_time(
    match: MatchData, team: str, ward_place_time: int,
) -> WardEvent | None:
    """Find a ward event by its placement time, filtering by team."""
    for w in match.wards:
        if w.time == ward_place_time and w.ward_type == "observer":
            ward_team = "radiant" if w.player_slot < 128 else "dire"
            if ward_team == team:
                return w
    return None


def _find_enemy_vis(
    vis_list: list[EnemyVisibility], player_slot: int,
) -> EnemyVisibility | None:
    """Find an enemy visibility entry by player_slot."""
    for e in vis_list:
        if e.player_slot == player_slot:
            return e
    return None


def _build_timeline_summary(events: list[VisionEvent], time_s: int) -> str:
    """Build a brief Chinese summary of the timeline."""
    if not events:
        return "此时间段内没有显著视野变化。"

    ward_losses = sum(1 for e in events if e.event_type in ("ward_expired", "ward_destroyed"))
    enemy_disappearances = sum(1 for e in events if e.event_type == "enemy_missing")
    tower_losses = sum(1 for e in events if e.event_type == "tower_lost")

    parts: list[str] = []
    if ward_losses > 0:
        parts.append(f"{ward_losses}次视野丢失（观察守卫）")
    if tower_losses > 0:
        parts.append(f"{tower_losses}座防御塔被摧毁")
    if enemy_disappearances > 0:
        parts.append(f"{enemy_disappearances}次敌方英雄消失事件")

    if parts:
        return f"过去60秒内发生：{'，'.join(parts)}。"
    return "过去60秒内视野状况相对稳定。"


# ===================================================================
# Formatted text output for LLM prompt
# ===================================================================

def _format_game_time(time_s: int) -> str:
    """Format game seconds as MM:SS."""
    if time_s < 0:
        return f"-{abs(time_s) // 60}:{abs(time_s) % 60:02d}"
    return f"{time_s // 60}:{time_s % 60:02d}"


def _danger_level_zh(level: str) -> str:
    """Convert danger level to Chinese."""
    return {
        "safe": "安全",
        "caution": "注意",
        "danger": "危险",
        "critical": "极度危险",
    }.get(level, "未知")


def _danger_level_en(level: str) -> str:
    """Convert danger level to English."""
    return {
        "safe": "Safe",
        "caution": "Caution",
        "danger": "Danger",
        "critical": "Critical",
    }.get(level, "Unknown")


def _format_vision_text(
    state: VisionState,
    timeline: VisionTimeline,
    time_s: int,
    match: MatchData,
    player_slot: int,
    player_pos: tuple[float, float],
    lang: str = "zh",
) -> str:
    """Generate the complete formatted text block for the LLM prompt."""
    lines: list[str] = []

    if lang == "en":
        # --- Header ---
        lines.append("## Vision Analysis")
        lines.append(
            f"Current time: {_format_game_time(time_s)} "
            f"({_day_night_label(time_s, lang='en')})"
        )
        lines.append(f"Allied active Observer Wards: {state.allied_observer_count}")
        lines.append(f"Vision danger level: {_danger_level_en(state.danger_level)}")
        if player_pos != (0.0, 0.0):
            area = _describe_area(player_pos[0], player_pos[1], lang="en")
            lines.append(f"Player position: {area}")
        lines.append("")

        # --- Enemy hero visibility ---
        lines.append("### Enemy Hero Visibility")
        if state.enemy_visibility:
            for ev in state.enemy_visibility:
                if ev.is_visible:
                    area = _describe_area(ev.last_seen_x, ev.last_seen_y, lang="en")
                    lines.append(f"- {ev.hero_name}: visible ({area})")
                else:
                    if ev.last_seen_time > 0:
                        area = _describe_area(ev.last_seen_x, ev.last_seen_y, lang="en")
                        lines.append(
                            f"- {ev.hero_name}: missing for {ev.seconds_missing}s"
                            f" (last seen at {area} @ {_format_game_time(ev.last_seen_time)})"
                        )
                    else:
                        lines.append(
                            f"- {ev.hero_name}: position unknown (no recent vision data)"
                        )
        else:
            lines.append("- No enemy hero data")
        lines.append("")

        # --- Danger factors ---
        if state.danger_reasons:
            lines.append("### Danger Factors")
            for reason in state.danger_reasons:
                lines.append(f"- {reason}")
            lines.append("")

        # --- Vision timeline ---
        if timeline.events:
            lines.append(f"### Vision Timeline ({_TIMELINE_WINDOW}s before event)")
            for evt in timeline.events:
                time_label = _format_game_time(evt.time)
                marker = " !!!" if evt.event_type in (
                    "ward_expired", "ward_destroyed", "tower_lost", "entered_fog",
                ) else ""
                lines.append(f"[{time_label}] {evt.description}{marker}")
            lines.append("")

        # --- Critical vision mistake ---
        if timeline.critical_moment > 0 and timeline.critical_description:
            lines.append("### Critical Vision Mistake")
            time_before = time_s - timeline.critical_moment
            lines.append(
                f"{_format_game_time(timeline.critical_moment)}"
                f" ({time_before}s before event) "
                f"{timeline.critical_description}"
            )

            missing_after = [
                e for e in state.enemy_visibility if not e.is_visible
            ]
            if missing_after:
                hero_names = ", ".join(e.hero_name for e in missing_after[:3])
                suffix = f" and {len(missing_after)} more heroes" if len(missing_after) > 3 else ""
                lines.append(
                    f"After this, {hero_names}{suffix} disappeared into fog, "
                    f"and the player did not retreat to safety in time."
                )
            lines.append("")
    else:
        # --- Header ---
        lines.append("## 视野分析")
        lines.append(
            f"当前时间: {_format_game_time(time_s)} "
            f"({_day_night_label(time_s)})"
        )
        lines.append(f"己方活跃观察守卫: {state.allied_observer_count}个")
        lines.append(f"视野危险等级: {_danger_level_zh(state.danger_level)}")
        if player_pos != (0.0, 0.0):
            area = _describe_area(player_pos[0], player_pos[1])
            lines.append(f"玩家位置: {area}")
        lines.append("")

        # --- Enemy hero visibility ---
        lines.append("### 敌方英雄可见性")
        if state.enemy_visibility:
            for ev in state.enemy_visibility:
                if ev.is_visible:
                    area = _describe_area(ev.last_seen_x, ev.last_seen_y)
                    lines.append(f"- {ev.hero_name}: ✅ 可见（{area}）")
                else:
                    if ev.last_seen_time > 0:
                        area = _describe_area(ev.last_seen_x, ev.last_seen_y)
                        lines.append(
                            f"- {ev.hero_name}: ❌ 已消失{ev.seconds_missing}秒"
                            f"（最后出现在{area} @ {_format_game_time(ev.last_seen_time)}）"
                        )
                    else:
                        lines.append(
                            f"- {ev.hero_name}: ❌ 位置未知（无近期视野数据）"
                        )
        else:
            lines.append("- 无敌方英雄数据")
        lines.append("")

        # --- Danger factors ---
        if state.danger_reasons:
            lines.append("### 危险因素")
            for reason in state.danger_reasons:
                lines.append(f"- {reason}")
            lines.append("")

        # --- Vision timeline ---
        if timeline.events:
            lines.append(f"### 视野时间线（事件发生前{_TIMELINE_WINDOW}秒）")
            for evt in timeline.events:
                time_label = _format_game_time(evt.time)
                marker = " ⚠️" if evt.event_type in (
                    "ward_expired", "ward_destroyed", "tower_lost", "entered_fog",
                ) else ""
                lines.append(f"[{time_label}] {evt.description}{marker}")
            lines.append("")

        # --- Critical vision mistake ---
        if timeline.critical_moment > 0 and timeline.critical_description:
            lines.append("### 关键视野失误")
            time_before = time_s - timeline.critical_moment
            lines.append(
                f"{_format_game_time(timeline.critical_moment)}"
                f"（事件发生前{time_before}秒）"
                f"{timeline.critical_description}"
            )

            missing_after = [
                e for e in state.enemy_visibility if not e.is_visible
            ]
            if missing_after:
                hero_names = "、".join(e.hero_name for e in missing_after[:3])
                suffix = f"等{len(missing_after)}名英雄" if len(missing_after) > 3 else ""
                lines.append(
                    f"此后{hero_names}{suffix}消失在迷雾中，"
                    f"玩家未及时撤退到安全区域。"
                )
            lines.append("")

    return "\n".join(lines)
