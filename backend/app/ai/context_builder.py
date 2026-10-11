"""Build DecisionContext from MatchData and DecisionPoint.

Given a match and a specific decision point (e.g. a death, a bad trade),
this module gathers all relevant context: the player's state, the allies and
enemies actually NEAR the player (world-unit radius), recent events around
the player, vision, game phase, and the match situation from the player's
team perspective. The resulting DecisionContext is ready for evaluation.

One moment, ``HeroState.state_time``, is described for every hero. For a
death it is the last sample at which the victim was alive, a few seconds
BEFORE the death (never the death tick); otherwise the decision time. A
decision point without a timestamp gets the end-of-match state of the player
and no positional context. Dead heroes have no position: they are never
"nearby" and are listed under ``unknown_position_*`` with ``is_alive`` False.

``HeroState.hp`` / ``max_hp`` / ``mana`` are only filled where the source
has absolute numbers (laning snapshots). The HP percentage, known for the
whole game from the position log, is ``HeroState.hp_pct`` (0-100; None when
unknown).

``match_context`` keys (consumed by the evaluator / agent prompts):
  team, gold_adv, xp_adv (positive = the player's team is ahead),
  radiant_gold_adv, radiant_xp_adv, team_kills, enemy_kills,
  radiant_kills, dire_kills (kills scored before the moment),
  kills_estimated (False when they come from the kill log),
  team_towers_standing, enemy_towers_standing, team_towers_lost,
  enemy_towers_lost, enemies_missing (in the fog), enemies_missing_nearby
  (of those, within vision.THREAT_RADIUS of the player), enemies_unknown (no
  position data), nearby_allies, nearby_enemies (living, known position),
  unknown_position_allies, unknown_position_enemies (living, no position),
  dead_allies, dead_enemies, headcount_basis ("nearby" / "participation" /
  "unknown") with headcount_allies / headcount_enemies, fight_allies /
  fight_enemies (participants of the teamfight the moment lies in),
  fallen_allies / fallen_enemies, opening_time, opening_allies /
  opening_enemies, opening_outnumbered (deaths with basis "nearby" only, and
  only when heroes died near the player earlier in the fight: how many, and
  the head count around the player when the first of them died, see
  ``vision.engagement``),
  state_time, seconds_before_death and pre_death_hp_pct (deaths only),
  state_is_final (time-unknown points), inventory_exact (False: the item
  lists still contain components, see ``recipes_known``), game_mode,
  recent_objectives, roshan, game_time, duration (+ player_position and
  positions, the derived position of every hero, both set by the coach),
  lane_role (positions.lane_role_of), has_positions /
  has_cast_log / has_assists (False for a source without position logs /
  cast times / assists: the prompt then says what cannot be judged),
  player_kills and player_deaths (the player's score before the decision
  second, from the kill log), aftermath (deaths only:
  kills, towers and the player's buyback in the seconds after the death),
  match_end ({seconds, won}: only for a moment in the last minute of the
  match; won is None when the source has no result).
"""
from __future__ import annotations

import bisect
import logging

from ..game_constants import ROSHAN_RESPAWN_MIN_S
from ..models import LaningSnapshot, MatchData, PlayerData, TeamfightEvent
from . import data_manager
from .models import (
    CausalChain,
    DecisionContext,
    DecisionPoint,
    EventEntry,
    HeroState,
    VisionAnalysis,
    VisionTimeline,
    is_time_unknown,
    unit_text,
)
from .phases import LANING_END, game_phase, level_from_xp, series_at
from .positions import lane_role_of
from .vision import (
    _distance,
    _get_active_wards,
    _get_alive_towers,
    _get_destroyed_tower_names,
    describe_area,
    engagement,
    fight_participants,
    has_position,
    head_count,
    hero_alive_at,
    hp_pct_at,
    kills_before,
    last_alive_time,
    nearby_radius,
    position_at,
    vision_rules,
    ward_team,
)

logger = logging.getLogger(__name__)


def _hero(name: str | None, lang: str) -> str:
    """A hero's name as text of language ``lang`` (the data keeps the English
    name: it is the key of every lookup). Anything that is not a hero of the
    catalog is returned as it is."""
    if not name or not data_manager.is_known_hero(name):
        return name or ""
    return data_manager.hero_display_name(name, lang)


def _heroes(names, lang: str) -> list[str]:
    return [_hero(name, lang) for name in names]


# How far back to look for the event timeline (seconds)
_EVENT_WINDOW = 60
# Events further than this from the player are not about this decision.
_EVENT_RADIUS = 3000
_MAX_TIMELINE_EVENTS = 40
# Casts of the killer this close to the death are part of it wherever the
# killer stood (Thundergod's Wrath from the fountain, a Sun Strike from mid).
_KILLER_CAST_WINDOW = 5
# What a death bought or cost is judged by the kills and towers right after it.
_AFTERMATH_SECONDS = 30
# A moment this close to the end of the match is told so: what happens while
# the ancient falls has no cost any more.
_MATCH_END_SECONDS = 60
# Wards within this radius of the player are reported as "nearby".
_NEARBY_WARD_RADIUS = 2000
# Six inventory slots plus three backpack slots.
_INVENTORY_SLOTS = 9

# Backward-compatible names (phase boundaries now live in phases.py).
_LANING_END = LANING_END
_NEARBY_LANING = nearby_radius(0)
_NEARBY_MID_LATE = nearby_radius(LANING_END)

# Consumables / one-shot items that only add noise to an inventory listing.
_CONSUMABLE_KEYS = {
    "tango", "tango_single", "flask", "clarity", "faerie_fire", "enchanted_mango",
    "famango", "great_famango", "greater_famango", "smoke_of_deceit", "ward_observer",
    "ward_sentry", "ward_dispenser", "dust", "tpscroll", "tome_of_knowledge",
    "blood_grenade", "bottle_doubledamage", "royal_jelly", "madstone_bundle",
    "foragers_health", "foragers_mana", "foragers_stats",
}
# Bought and consumed into a permanent buff: listed, but they take no slot.
_NO_SLOT_KEYS = {"aghanims_shard", "ultimate_scepter_2", "moon_shard"}

# Casts that say nothing about a fight: Invoker's orb switches and Invoke
# here, world interactions and cosmetic actions (lamps, outposts, Twin
# Gates ...) through the parser's own list, see _is_noop_cast.
_NOOP_CAST_KEYS = {
    "invoker_quas", "invoker_wex", "invoker_exort", "invoker_invoke",
    "ability_lamp_use", "ability_capture", "plus_high_five",
}
# The same hero repeating the same cast within this many seconds is one entry.
_REPEAT_CAST_WINDOW = 5
# Laning damage between two heroes is summed per bucket of this many seconds.
_DAMAGE_BUCKET = 5


# ===================================================================
# Public API
# ===================================================================

def build_context(
    match: MatchData,
    decision_point: DecisionPoint,
    causal_chain: CausalChain | None = None,
    lang: str = "zh",
) -> DecisionContext:
    """Build full context for a decision point.

    Parameters
    ----------
    match : MatchData
        Complete parsed match data.
    decision_point : DecisionPoint
        The moment being evaluated (death, bad trade, etc.)
    causal_chain : CausalChain | None
        Optional causal chain analysis for death events.
    lang : str
        Language for generated text ("zh" or "en").
    """
    t = decision_point.time
    slot = decision_point.player_slot
    time_unknown = is_time_unknown(decision_point)

    player = _find_player(match, slot)
    if player is None:
        return DecisionContext(
            decision_point=decision_point,
            player_state=HeroState(hero_name=decision_point.hero_name, player_slot=slot, team="unknown"),
            game_phase="" if time_unknown else game_phase(t),
        )

    # good_play points from multi-kill / kill-streak counters carry no
    # timestamp; leave the phase unknown and skip the positional context
    # instead of building a misleading 0:00 picture.
    phase = "" if time_unknown else game_phase(t)
    death = decision_point.related_death
    is_death = decision_point.point_type == "death" and death is not None and not time_unknown

    if time_unknown:
        state_time: int | None = None
        player_state = _final_hero_state(player)
    else:
        # A death is described by the last moment the victim was alive.
        state_time = last_alive_time(match, slot, t) if is_death else t
        player_state = _build_hero_state(player, state_time, match)
        if is_death:
            player_state.is_alive = True
    if not player_state.has_position and death is not None and has_position((death.x, death.y)):
        player_state.x, player_state.y = death.x, death.y

    allies: list[HeroState] = []
    enemies: list[HeroState] = []
    if not time_unknown:
        for p in match.players:
            if p.player_slot == slot:
                continue
            state = _build_hero_state(p, state_time, match)
            if player_state.has_position and state.has_position:
                state.distance_to_player = round(
                    _distance(player_state.x, player_state.y, state.x, state.y), 0)
            (allies if p.team == player.team else enemies).append(state)

    radius = nearby_radius(t)
    near_allies, far_allies, unknown_allies = split_by_distance(allies, radius)
    near_enemies, far_enemies, unknown_enemies = split_by_distance(enemies, radius)
    # The killer was at the death even when the data has no position for it.
    if is_death and (death.killer_slot is not None or death.killer_hero):
        for e in list(unknown_enemies):
            is_killer = (e.player_slot == death.killer_slot if death.killer_slot is not None
                         else e.hero_name == death.killer_hero)
            if is_killer and e.is_alive:
                unknown_enemies.remove(e)
                near_enemies.append(e)

    map_state = {} if time_unknown else _build_map_state(match, t, player_state, player.team, lang=lang)
    if phase == "pregame":
        pregame_heroes: dict[str, list[dict]] = {"radiant": [], "dire": []}
        for p in match.players:
            pregame_heroes.setdefault(p.team, []).append({
                "hero_name": p.hero_name,
                "player_slot": p.player_slot,
                "team": p.team,
                "attack_type": getattr(p, "attack_type", "unknown"),
            })
        map_state["pregame_analysis"] = pregame_heroes

    if time_unknown:
        match_context: dict = {"team": player.team, "duration": match.duration, "state_is_final": True}
    else:
        match_context = _build_match_context(match, t, player.team)
        match_context["state_time"] = state_time
        match_context["inventory_exact"] = recipes_known()
        match_context["lane_role"] = lane_role_of(player)
        # What the source recorded at all (OpenDota: neither positions nor cast times).
        match_context["has_positions"] = any(p.position_log for p in match.players)
        match_context["has_cast_log"] = bool(match.ability_events)
        if match.kills_log:
            match_context["has_assists"] = any(k.assist_slots for k in match.kills_log)
            # The player's own score before this second (the kill log is exact).
            earlier = [k for k in match.kills_log if k.confirmed and int(k.time) < t]
            match_context["player_kills"] = sum(1 for k in earlier if k.killer_slot == slot and not k.deny)
            match_context["player_deaths"] = sum(1 for k in earlier if k.victim_slot == slot)
        if 0 <= match.duration - t <= _MATCH_END_SECONDS:
            won = (player.team == "radiant") == bool(match.radiant_win) if match.winner_known else None
            match_context["match_end"] = {"seconds": int(match.duration - t), "won": won}
        if is_death:
            match_context["seconds_before_death"] = max(0, t - state_time)
            if player_state.hp_pct:
                match_context["pre_death_hp_pct"] = player_state.hp_pct
            aftermath = _aftermath(match, player, t)
            if aftermath is not None:
                match_context["aftermath"] = aftermath
        match_context["nearby_allies"] = len(near_allies)
        match_context["nearby_enemies"] = len(near_enemies)
        match_context["unknown_position_allies"] = sum(1 for a in unknown_allies if a.is_alive)
        match_context["unknown_position_enemies"] = sum(1 for e in unknown_enemies if e.is_alive)
        match_context["dead_allies"] = sum(1 for a in allies if not a.is_alive)
        match_context["dead_enemies"] = sum(1 for e in enemies if not e.is_alive)
        tf = _fight_at(match, death if is_death else None, t)
        if tf is not None:
            radiant, dire = fight_participants(tf)
            mine, theirs = (radiant, dire) if player.team == "radiant" else (dire, radiant)
            match_context["fight_allies"] = sum(1 for s in mine if s != slot)
            match_context["fight_enemies"] = len(theirs)
        basis, n_allies, n_enemies = head_count(
            match, slot, state_time, (player_state.x, player_state.y), tf)
        if basis == "nearby":  # same heroes as the lists above (incl. the killer rule)
            n_allies, n_enemies = len(near_allies), len(near_enemies)
        match_context["headcount_basis"] = basis
        match_context["headcount_allies"] = n_allies
        match_context["headcount_enemies"] = n_enemies
        # The count above is who was left at the end: a fight in which
        # heroes died around the player is judged by how it started.
        eng = engagement(match, slot, t, tf) if is_death and basis == "nearby" else None
        if eng is not None:
            match_context["fallen_allies"] = eng.fallen_allies
            match_context["fallen_enemies"] = eng.fallen_enemies
            match_context["opening_time"] = eng.open_time
            match_context["opening_allies"] = eng.open_allies
            match_context["opening_enemies"] = eng.open_enemies
            match_context["opening_outnumbered"] = eng.opened_outnumbered

    vision_analysis: VisionAnalysis | None = None
    if not time_unknown:
        try:
            from .vision import analyze_vision
            # Vision is judged at the decision second; a death at the spot
            # where the player died.
            at = (death.x, death.y) if is_death and has_position((death.x, death.y)) else (
                player_state.x, player_state.y)
            vision_analysis = analyze_vision(match, slot, t, at, lang=lang, died=is_death)
        except Exception as exc:
            logger.warning("Vision analysis failed (t=%s): %s", t, exc)

    if vision_analysis is not None:
        vs = vision_analysis.state_at_decision
        match_context["enemies_missing"] = vs.missing_enemy_count
        match_context["enemies_missing_nearby"] = vs.threat_enemy_count
        match_context["enemies_unknown"] = vs.unknown_enemy_count
        match_context["vision_danger"] = vs.danger_level
        map_state["vision_summary"] = _vision_summary(vs, lang)

    nearby_slots = {s.player_slot for s in near_allies + near_enemies}
    event_timeline = [] if time_unknown else _build_event_timeline(
        match, player, t, vision_analysis, lang=lang,
        player_pos=(player_state.x, player_state.y), nearby_slots=nearby_slots,
        killer_slot=death.killer_slot if is_death else None,
    )

    if causal_chain and causal_chain.formatted_text:
        map_state["causal_chain_text"] = causal_chain.formatted_text

    map_state["distant_allies"] = [_brief_state(s, lang) for s in far_allies]
    map_state["distant_enemies"] = [_brief_state(s, lang) for s in far_enemies]
    map_state["unknown_position_allies"] = [_brief_state(s, lang) for s in unknown_allies]
    map_state["unknown_position_enemies"] = [_brief_state(s, lang) for s in unknown_enemies]
    map_state["dead_allies"] = [s.hero_name for s in allies if not s.is_alive]
    map_state["dead_enemies"] = [s.hero_name for s in enemies if not s.is_alive]

    ctx = DecisionContext(
        decision_point=decision_point,
        player_state=player_state,
        enemy_states=near_enemies,
        ally_states=near_allies,
        event_timeline=event_timeline,
        map_state=map_state,
        game_phase=phase,
        match_context=match_context,
        vision_analysis=vision_analysis,
    )
    ctx.causal_chain = causal_chain
    return ctx


def _fight_at(match: MatchData, death, time_s: int) -> TeamfightEvent | None:
    """Teamfight a moment (or the given death) lies in.

    A death from the kill log carries its fight (``teamfight_idx``); without
    one it is a pick-off elsewhere during a fight window, not part of it.
    """
    if death is not None and 0 <= death.teamfight_idx < len(match.teamfights):
        return match.teamfights[death.teamfight_idx]
    if death is not None and death.source == "kills_log":
        return None
    for tf in match.teamfights:
        if tf.start_time <= time_s <= tf.end_time:
            return tf
    return None


def _vision_summary(vs, lang: str) -> str:
    """One line on wards and enemy visibility (no vision block in the prompt)."""
    states = [e.state for e in vs.enemy_visibility]
    visible, dead = states.count("visible"), states.count("dead")
    if lang == "en":
        parts = [f"{vs.allied_observer_count} allied observer wards", f"{visible} enemies visible"]
        if vs.missing_enemy_count:
            parts.append(f"{vs.missing_enemy_count} in the fog ({', '.join(_heroes(vs.missing_enemy_names, lang))})")
        if vs.unknown_enemy_count:
            parts.append(f"no position data for {vs.unknown_enemy_count} "
                         f"({', '.join(_heroes(vs.unknown_enemy_names, lang))})")
        if dead:
            parts.append(f"{dead} dead")
        return ", ".join(parts)
    parts = [f"己方{vs.allied_observer_count}个侦察守卫", f"敌方{visible}人可见"]
    if vs.missing_enemy_count:
        parts.append(f"{vs.missing_enemy_count}人在迷雾中（{'、'.join(_heroes(vs.missing_enemy_names, lang))}）")
    if vs.unknown_enemy_count:
        parts.append(f"{vs.unknown_enemy_count}人无位置数据（{'、'.join(_heroes(vs.unknown_enemy_names, lang))}）")
    if dead:
        parts.append(f"{dead}人阵亡")
    return "，".join(parts)


# ===================================================================
# Proximity
# ===================================================================

def split_by_distance(
    heroes: list[HeroState], radius: float,
) -> tuple[list[HeroState], list[HeroState], list[HeroState]]:
    """Split heroes into (nearby, distant, position_unknown).

    Uses ``distance_to_player`` (world units); heroes without a known
    distance are reported as position-unknown rather than assumed nearby.
    """
    nearby: list[HeroState] = []
    distant: list[HeroState] = []
    unknown: list[HeroState] = []
    for h in heroes:
        if h.distance_to_player is None:
            unknown.append(h)
        elif h.distance_to_player <= radius:
            nearby.append(h)
        else:
            distant.append(h)
    return nearby, distant, unknown


def _split_by_proximity(
    player: HeroState,
    enemies: list[HeroState],
    game_phase: str,
) -> tuple[list[HeroState], list[HeroState]]:
    """Backward-compatible (nearby, distant) split.

    Heroes without a position are conservatively kept in the nearby list.
    """
    radius = nearby_radius(LANING_END if game_phase in ("mid_game", "late_game") else 0)
    nearby: list[HeroState] = []
    distant: list[HeroState] = []
    for e in enemies:
        if not player.has_position or not e.has_position:
            nearby.append(e)
        elif _distance(player.x, player.y, e.x, e.y) <= radius:
            nearby.append(e)
        else:
            distant.append(e)
    return nearby, distant


def count_nearby_allies(
    player: HeroState,
    allies: list[HeroState],
    game_phase: str,
) -> int:
    """Count living allies within the "nearby" radius of the player.

    Allies without a known position are NOT counted: assuming the player was
    alone is the lenient direction for grading.
    """
    if not player.has_position:
        return 0
    radius = nearby_radius(LANING_END if game_phase in ("mid_game", "late_game") else 0)
    count = 0
    for a in allies:
        if not a.is_alive or not a.has_position:
            continue
        if _distance(player.x, player.y, a.x, a.y) <= radius:
            count += 1
    return count


def _brief_state(state: HeroState, lang: str) -> dict:
    """Compact description of a hero that is not near the player."""
    return {
        "hero_name": state.hero_name,
        "player_slot": state.player_slot,
        "x": state.x,
        "y": state.y,
        "is_alive": state.is_alive,
        "level": state.level,
        "hp_pct": state.hp_pct,
        "distance": state.distance_to_player,
        "area": describe_area(state.x, state.y, lang) if state.has_position else "",
    }


# ===================================================================
# Player lookup
# ===================================================================

def _find_player(match: MatchData, player_slot: int) -> PlayerData | None:
    for p in match.players:
        if p.player_slot == player_slot:
            return p
    return None


# ===================================================================
# Hero state construction
# ===================================================================

def _build_hero_state(player: PlayerData, time_s: int, match: MatchData) -> HeroState:
    """HeroState of a player at ``time_s`` (``state_time``).

    Position and alive state come from :mod:`vision` (a dead hero has no
    position). Laning snapshots (5 s resolution, first 10 minutes) give exact
    HP/mana/level/CS/net worth; later moments fall back to the per-minute
    series. ``hp_pct`` is set whenever the HP percentage is known.
    """
    slot = player.player_slot
    state = HeroState(hero_name=player.hero_name, player_slot=slot, team=player.team,
                      state_time=time_s)

    snap = _snapshot_at(player.laning_snapshots, time_s) if time_s < LANING_END else None
    if snap is not None:
        state.hp = snap.hp
        state.max_hp = snap.max_hp
        state.mana = snap.mana
        state.max_mana = snap.max_mana
        state.level = snap.level
        state.last_hits = snap.last_hits
        state.denies = snap.denies
        state.net_worth = snap.net_worth
    else:
        minute = _time_to_minute(time_s)
        # The series start at 0:00: before the horn they would show gold the
        # hero only earned afterwards (a first-blood bounty, ...).
        if player.gold_t and time_s >= 0:
            state.net_worth = series_at(player.gold_t, minute)
        # The XP series starts at 0:00 too; before the horn a hero is level 1.
        if player.xp_t and time_s >= 0:
            state.level = level_from_xp(_xp_at(player.xp_t, time_s))
        elif time_s <= 0:
            state.level = 1
        if player.lh_t:
            state.last_hits = series_at(player.lh_t, minute)
        # Denies per minute are not in every source: unknown is not zero.
        state.denies = player.dn_t[minute] if minute < len(player.dn_t) else None

    state.hp_pct = hp_pct_at(match, slot, time_s)
    state.is_alive = hero_alive_at(match, slot, time_s) is not False
    pos = position_at(match, slot, time_s)
    if pos is not None:
        state.x, state.y = pos
    state.items = _items_at_time(player, time_s)
    # A log whose first purchase lies after this moment has no starting items.
    state.items_known = bool(state.items) or any(i.time <= min(time_s, 0) for i in player.items)
    state.abilities_available = _abilities_at_time(player, time_s)
    state.ability_levels = _ability_levels_at_time(player, time_s)
    return state


def _final_hero_state(player: PlayerData) -> HeroState:
    """End-of-match state, for a decision point that has no timestamp."""
    state = HeroState(hero_name=player.hero_name, player_slot=player.player_slot, team=player.team)
    state.level = player.level
    state.last_hits = player.last_hits
    state.denies = player.denies
    state.net_worth = player.net_worth
    names = [_item_facts(key)[1] for key in list(player.final_item_keys) + [player.neutral_item] if key]
    state.items = list(dict.fromkeys(names)) or list(dict.fromkeys(player.final_items))
    end = max((s.time for s in player.skills), default=0)
    state.abilities_available = _abilities_at_time(player, end)
    state.ability_levels = _ability_levels_at_time(player, end)
    return state


def _snapshot_at(snapshots: list[LaningSnapshot], time_s: int) -> LaningSnapshot | None:
    """The laning snapshot at or just before ``time_s``.

    Values are not interpolated: the neighbour of a snapshot may be a dead
    one, and half-way to 0 HP is not what the hero had. None when every
    snapshot is later than ``time_s``.
    """
    if not snapshots:
        return None
    times = [s.time for s in snapshots]
    i = bisect.bisect_right(times, time_s) - 1
    # A moment before the first snapshot (before the horn) has none: the
    # first one describes a later state.
    return snapshots[i] if i >= 0 else None


def _time_to_minute(time_s: int) -> int:
    """Game seconds -> minute index (clamped to >= 0)."""
    return max(0, time_s // 60)


def _xp_at(xp_t: list[int], time_s: int) -> int:
    """XP at ``time_s`` from the per-minute series, interpolated within the minute.

    The value at the start of the minute would leave a hero a level behind
    for up to a minute (early on a hero gains a level in less than that),
    next to skills it has already learnt. OpenDota skill times are estimated
    from the same interpolation, so level and skills agree there.
    """
    minute = _time_to_minute(time_s)
    if minute + 1 >= len(xp_t):
        return series_at(xp_t, minute)
    start, end = xp_t[minute], xp_t[minute + 1]
    return int(start + (end - start) * (time_s - minute * 60) / 60)


def _estimate_level_from_xp(xp: int) -> int:
    """Backward-compatible alias of :func:`phases.level_from_xp`."""
    return level_from_xp(xp)


# Item facts by purchase key, valid for one data_manager generation.
_item_cache: dict = {"generation": None, "facts": {}}


def _items_cache() -> dict:
    generation = data_manager.data_generation()
    cache = _item_cache
    if cache["generation"] != generation:
        cache = {"generation": generation, "facts": {}}
        globals()["_item_cache"] = cache
    return cache


def recipes_known() -> bool:
    """Whether the game data has item recipes (component lists).

    Without them (game data written before the component table, or the
    built-in fallback) the reconstructed inventory still lists the
    components next to the items built from them.
    """
    cache = _items_cache()
    if "recipes" not in cache:
        try:
            from .game_data import get_item_table
            cache["recipes"] = any(info.get("components") for info in get_item_table().values())
        except Exception as exc:
            logger.warning("Item table unavailable: %s", exc)
            cache["recipes"] = False
    return cache["recipes"]


def _item_facts(raw_key: str) -> tuple[str, str, int, tuple[str, ...]]:
    """(canonical key, display name, cost, component keys) of a purchased item.

    Upgrade levels resolve to the base item ("dagon_3" -> "dagon"). The
    components are empty for a basic item and when the game data on disk
    predates the component table.
    """
    cache = _items_cache()
    facts = cache["facts"].get(raw_key)
    if facts is None:
        key = str(raw_key or "").lower()
        info = None
        try:
            from .game_data import get_item_data
            info = get_item_data(key)
            key = data_manager.resolve_item_key(key) or key
        except Exception as exc:  # the inventory must never break a report
            logger.warning("Item lookup failed for %r: %s", raw_key, exc)
        if info and info.get("key"):
            key = str(info["key"])
        components = tuple(
            data_manager.resolve_item_key(c) or str(c) for c in (info or {}).get("components") or ())
        cost = (info or {}).get("cost")
        facts = (key, data_manager.item_display_name(key), int(cost) if isinstance(cost, (int, float)) else 0,
                 components)
        cache["facts"][raw_key] = facts
    return facts


def _items_at_time(player: PlayerData, time_s: int) -> list[str]:
    """Estimated inventory at ``time_s`` as display names, in purchase order.

    Rebuilt from the purchase log: consumables and recipes are skipped and,
    where the game data has the recipe, the components of an item leave the
    inventory when it is built. Selling is not in the data, so when more
    items remain than a hero can carry the cheapest ones that are not in the
    end-of-game inventory are dropped. The neutral item is only known at the
    end of the game and is not listed.
    """
    held: list[tuple[str, str, int]] = []  # (key, display name, cost)
    events = sorted(player.items, key=lambda i: i.time)
    i = 0
    while i < len(events) and events[i].time <= time_s:
        j = i
        while j < len(events) and events[j].time == events[i].time:
            j += 1
        built: list[tuple[int, str, tuple[str, ...]]] = []
        for event in events[i:j]:
            raw = (event.item_key or event.item_name or "").lower()
            if not raw or raw in _CONSUMABLE_KEYS or raw.startswith("recipe"):
                continue
            key, name, cost, components = _item_facts(raw)
            if key in _CONSUMABLE_KEYS or key.startswith("recipe"):
                continue
            if key != raw and any(h[0] == key for h in held):
                continue  # an upgrade level of an item already held
            held.append((key, name, cost))
            if components:
                built.append((cost, key, components))
        # Everything bought in this second is in the inventory before the
        # items built in it take their components (cheapest first, so a
        # sub-item exists before the item made from it).
        for _cost, key, components in sorted(built):
            for component in components:
                for n, h in enumerate(held):
                    if h[0] == component and h[0] != key:
                        del held[n]
                        break
        i = j

    unique: list[tuple[str, str, int]] = []
    for h in held:
        if all(h[1] != u[1] for u in unique):
            unique.append(h)
    slots = [h for h in unique if h[0] not in _NO_SLOT_KEYS]
    if len(slots) > _INVENTORY_SLOTS:
        final = {_item_facts(k)[0] for k in list(player.final_item_keys) + list(player.backpack) if k}
        surplus = sorted((h for h in slots if h[0] not in final), key=lambda h: h[2])
        dropped = {h[0] for h in surplus[:len(slots) - _INVENTORY_SLOTS]}
        unique = [h for h in unique if h[0] not in dropped]
    return [h[1] for h in unique]


# Ability display names by key, valid for one data_manager generation.
_ability_cache: dict = {"generation": None, "names": {}}


def _ability_name(key: str, fallback: str = "", lang: str = "en") -> str:
    """Current display name of an ability key (talents included).

    Resolved from the key because the name stored with a parsed event can be
    stale. "" for a talent whose text still holds a template placeholder, for
    a talent no catalog knows (its display name would be its key, e.g.
    "Wraith King Facet 1") and for an id that resolves to nothing readable.
    """
    generation = data_manager.data_generation()
    cache = _ability_cache
    if cache["generation"] != generation:
        cache = {"generation": generation, "names": {}}
        globals()["_ability_cache"] = cache
    memo_key = (key or fallback, lang)
    name = cache["names"].get(memo_key)
    if name is None:
        name = data_manager.ability_display_name(key, lang) if key else fallback
        unnamed_talent = key.startswith("special_bonus_") and data_manager.resolve_ability_name(key) is None
        if "{" in name or name.isdigit() or unnamed_talent:
            name = ""
        cache["names"][memo_key] = name
    return name


def _skilled(player: PlayerData, time_s: int) -> list[str]:
    """Display names of the skill points spent by ``time_s`` (one per point)."""
    names: list[str] = []
    for skill_event in player.skills:
        if skill_event.time > time_s:
            continue
        key = (skill_event.ability_key or "").lower()
        if key == "special_bonus_attributes":
            continue
        name = _ability_name(key, skill_event.ability_name or "")
        if name:
            names.append(name)
    return names


def _abilities_at_time(player: PlayerData, time_s: int) -> list[str]:
    """Abilities skilled by ``time_s`` (in skill order)."""
    return list(dict.fromkeys(_skilled(player, time_s)))


def _ability_levels_at_time(player: PlayerData, time_s: int) -> dict[str, int]:
    """Skill points per ability up to ``time_s`` (display name -> level).

    The name is the current one for the event's ability key, so it matches
    the ability tables of ``game_data``.
    """
    levels: dict[str, int] = {}
    for name in _skilled(player, time_s):
        levels[name] = levels.get(name, 0) + 1
    return levels


# ===================================================================
# Event timeline
# ===================================================================

# Which entries survive the cap first (lower = kept first).
_PRIORITY_KILL = 0
_PRIORITY_MAP = 1      # smokes, vision / tower events
_PRIORITY_PLAYER = 2   # the player's casts, damage taken, nearby wards
_PRIORITY_OTHER = 3    # other heroes' casts, damage dealt


def _build_event_timeline(
    match: MatchData,
    player: PlayerData,
    time_s: int,
    vision_analysis: VisionAnalysis | None = None,
    lang: str = "zh",
    player_pos: tuple[float, float] = (0.0, 0.0),
    nearby_slots: set[int] | None = None,
    killer_slot: int | None = None,
) -> list[EventEntry]:
    """Events in the window before the decision that concern the player.

    Includes kills and deaths, buybacks, smokes, vision events, the player's
    casts, casts by nearby heroes, the last casts of ``killer_slot`` (the hero
    that killed the player at ``time_s``, however far away), laning damage
    (summed per few seconds) and ward placements near the player (none when
    his position is unknown). The window is the full ``_EVENT_WINDOW`` (also
    before the horn); when more than ``_MAX_TIMELINE_EVENTS`` entries remain,
    the less important ones are dropped first, not the oldest.
    """
    start = time_s - _EVENT_WINDOW
    end = time_s
    en = lang == "en"
    ranked: list[tuple[int, EventEntry]] = []
    nearby_slots = nearby_slots or set()
    known_pos = has_position(player_pos)

    def _close(x: float | None, y: float | None) -> bool | None:
        if not known_pos or x is None or y is None or not has_position((x, y)):
            return None
        return _distance(player_pos[0], player_pos[1], x, y) <= _EVENT_RADIUS

    casts: list[tuple[int, object]] = []
    for ae in player.ability_events:
        if start <= ae.time <= end:
            casts.append((_PRIORITY_PLAYER, ae))
    for ae in match.ability_events:
        if ae.player_slot == player.player_slot or not (start <= ae.time <= end):
            continue
        relevant = ae.target == player.hero_name or ae.player_slot in nearby_slots or (
            ae.player_slot == killer_slot and ae.time >= end - _KILLER_CAST_WINDOW)
        if not relevant:
            caster = (ae.x, ae.y) if ae.x is not None and ae.y is not None and has_position((ae.x, ae.y)) else (
                position_at(match, ae.player_slot, ae.time))
            if caster is None:
                relevant = False
            elif known_pos:
                relevant = bool(_close(*caster))
            else:
                # The player's place at the decision is unknown (he is dead, or
                # the source has no positions): where he stood when the spell
                # was cast decides. Listing every enemy cast on the map made a
                # verdict count an Ignite cast 14,000 units away among the
                # spells "that landed on you".
                me = position_at(match, player.player_slot, ae.time)
                relevant = me is not None and _distance(me[0], me[1], caster[0], caster[1]) <= _EVENT_RADIUS
        if relevant:
            on_player = ae.target == player.hero_name
            casts.append((_PRIORITY_PLAYER if on_player else _PRIORITY_OTHER, ae))
    ranked.extend(_cast_entries(casts, lang))

    if time_s < LANING_END:
        _add_laning_damage_events(ranked, match, player, start, end, lang=lang)

    for ward in match.wards:
        if not (start <= ward.time <= end):
            continue
        # Without the player's position a ward says nothing about where he
        # was (a live verdict placed a mid laner's death at "the top lane",
        # where his team had warded a minute earlier).
        if not _close(ward.x, ward.y):
            continue
        own = ward_team(ward) == player.team
        if en:
            who = "Allied" if own else "Enemy"
            what = "a Sentry Ward" if ward.ward_type == "sentry" else "an Observer Ward"
            desc = f"{who} team placed {what}"
        else:
            who = "己方" if own else "敌方"
            what = "真眼" if ward.ward_type == "sentry" else "假眼"
            desc = f"{who}放置了{what}"
        ranked.append((_PRIORITY_PLAYER, EventEntry(
            time=ward.time,
            event_type="ward_placed",
            description=desc,
            details={"ward_type": ward.ward_type, "x": ward.x, "y": ward.y,
                     "player_slot": ward.player_slot},
        )))

    for smoke in match.smokes:
        if start <= smoke.time <= end:
            own = smoke.team == player.team
            if en:
                desc = f"{'Allied' if own else 'Enemy'} team used Smoke of Deceit"
            else:
                desc = f"{'己方' if own else '敌方'}使用了诡计之雾"
            ranked.append((_PRIORITY_MAP, EventEntry(
                time=smoke.time,
                event_type="smoke_used",
                description=desc,
                details={"team": smoke.team, "participants": smoke.player_slots,
                         "x": smoke.x, "y": smoke.y},
            )))

    _add_kill_events(ranked, match, player, start, end, lang)

    for p in match.players:
        for entry in p.buyback_log:
            when = entry.get("time")
            if isinstance(when, (int, float)) and start <= when <= end:
                ranked.append((_PRIORITY_KILL, EventEntry(
                    time=int(when), event_type="buyback",
                    description=f"{p.hero_name} bought back" if en else f"{_hero(p.hero_name, lang)} 买活",
                    details={"player_slot": p.player_slot},
                )))

    if vision_analysis is not None:
        vision_events: list[EventEntry] = []
        _merge_vision_events(vision_events, vision_analysis.timeline, start, end)
        if not known_pos:
            vision_events = [e for e in vision_events if not e.event_type.startswith("ward_")]
        ranked.extend((_PRIORITY_MAP, e) for e in vision_events)

    if len(ranked) > _MAX_TIMELINE_EVENTS:
        # Most important first; within one class the most recent first.
        ranked.sort(key=lambda pe: (pe[0], -pe[1].time))
        ranked = ranked[:_MAX_TIMELINE_EVENTS]
    events = [e for _priority, e in ranked]
    # Within one second the player's own death closes the list.
    events.sort(key=lambda e: (e.time, e.event_type == "death"))
    return events


def _is_noop_cast(key: str) -> bool:
    if key in _NOOP_CAST_KEYS:
        return True
    try:
        from ..parser import is_utility_ability
    except Exception:  # pragma: no cover - the parser module is always there in the app
        return False
    return is_utility_ability(key)


def _cast_entries(casts: list[tuple[int, object]], lang: str) -> list[tuple[int, EventEntry]]:
    """Cast events without no-op casts, immediate repeats merged ("×3")."""
    en = lang == "en"
    out: list[tuple[int, EventEntry]] = []
    last: dict[tuple[int, str], EventEntry] = {}
    for priority, ae in sorted(casts, key=lambda c: c[1].time):
        key = (ae.ability_key or "").lower()
        if _is_noop_cast(key):
            continue
        ability = _ability_name(key, ae.ability_name or "", lang) or ae.ability_name
        if not ability:
            continue
        previous = last.get((ae.player_slot, ability))
        if previous is not None and ae.time - previous.details["last_time"] <= _REPEAT_CAST_WINDOW:
            previous.details["count"] += 1
            previous.details["last_time"] = ae.time
            previous.description = _cast_text(previous.details, lang)
            continue
        details = {"ability": ability, "hero": ae.hero_name, "player_slot": ae.player_slot,
                   "target": ae.target or "", "x": ae.x, "y": ae.y, "count": 1, "last_time": ae.time}
        entry = EventEntry(time=ae.time, event_type="ability_cast",
                           description=_cast_text(details, lang), details=details)
        last[(ae.player_slot, ability)] = entry
        out.append((priority, entry))
    return out


def _cast_text(details: dict, lang: str) -> str:
    hero, ability, target = details["hero"], details["ability"], details["target"]
    times = f" ×{details['count']}" if details["count"] > 1 else ""
    if lang == "en":
        return f"{hero} cast {ability}{times}" + (f" on {target}" if target else "")
    hero, target = _hero(hero, lang), _hero(target, lang)
    return f"{hero} 使用了 {ability}{times}" + (f"（目标：{target}）" if target else "")


def _add_kill_events(
    ranked: list[tuple[int, EventEntry]], match: MatchData, player: PlayerData,
    start: int, end: int, lang: str,
) -> None:
    """Hero deaths in the window: the kill log, else teamfight records."""
    en = lang == "en"
    if match.kills_log:
        by_slot = {p.player_slot: p for p in match.players}
        for k in match.kills_log:
            when = int(k.time)
            victim = by_slot.get(k.victim_slot)
            # An unconfirmed entry may be a reincarnation, not a death.
            if not (start <= when <= end) or victim is None or not k.confirmed:
                continue
            killer = by_slot.get(k.killer_slot) if k.killer_slot is not None else None
            # A unit's side: as recorded, else the side that is not the victim's.
            side = k.killer_team if k.killer_team in ("radiant", "dire") else (
                "" if k.deny else "dire" if victim.team == "radiant" else "radiant")
            unit = unit_text(k.killer_unit, lang, team=side)
            victim_name = _hero(victim.hero_name, lang)
            killer_name = _hero(killer.hero_name, lang) if killer is not None else ""
            if killer is not None and k.deny:
                desc = (f"{victim_name} was denied by {killer_name}" if en
                        else f"{victim_name} 被队友 {killer_name} 反补")
            elif killer is not None:
                desc = (f"{killer_name} killed {victim_name}" if en
                        else f"{killer_name} 击杀了 {victim_name}")
            elif unit:
                desc = (f"{victim_name} died to {unit}" if en
                        else f"{victim_name} 被{unit}击杀")
            else:
                desc = f"{victim_name} died" if en else f"{victim_name} 阵亡"
            assists = [_hero(by_slot[s].hero_name, lang) for s in k.assist_slots if s in by_slot]
            if assists:
                desc += (f" (assists: {', '.join(assists)})" if en
                         else f"（助攻：{'、'.join(assists)}）")
            if k.x is not None and k.y is not None and has_position((k.x, k.y)):
                area = describe_area(k.x, k.y, lang)
                desc += f" at {area}" if en else f"，位于{area}"
            ranked.append((_PRIORITY_KILL, EventEntry(
                time=when,
                event_type="death" if k.victim_slot == player.player_slot else "kill",
                description=desc,
                details={"victim": victim.hero_name, "victim_slot": k.victim_slot,
                         "killer": killer.hero_name if killer else "", "killer_slot": k.killer_slot,
                         "killer_unit": k.killer_unit, "assist_slots": list(k.assist_slots),
                         "deny": k.deny},
            )))
        return
    for tf in match.teamfights:
        if tf.start_time > end or tf.end_time < start:
            continue
        for tp in tf.players:
            if tp.deaths > 0 and tp.death_time_offset is not None:
                death_abs = tf.start_time + tp.death_time_offset
                if start <= death_abs <= end:
                    ranked.append((_PRIORITY_KILL, EventEntry(
                        time=int(death_abs),
                        event_type="death",
                        description=(f"{tp.hero_name} died in the teamfight" if en
                                     else f"{_hero(tp.hero_name, lang)} 在团战中阵亡"),
                        details={"hero": tp.hero_name, "player_slot": tp.player_slot,
                                 "teamfight_start": tf.start_time},
                    )))
            for kt in tp.kill_targets:
                if kt.get("time_offset") is None:
                    continue  # no time for this kill: it cannot be placed
                kill_abs = tf.start_time + kt["time_offset"]
                if start <= kill_abs <= end:
                    target = kt.get("target") or ("unknown" if en else "未知")
                    ranked.append((_PRIORITY_KILL, EventEntry(
                        time=int(kill_abs),
                        event_type="kill",
                        description=(f"{tp.hero_name} killed {target}" if en
                                     else f"{_hero(tp.hero_name, lang)} 击杀了 {_hero(target, lang)}"),
                        details={"killer": tp.hero_name, "target": kt.get("target", ""),
                                 "killer_slot": tp.player_slot},
                    )))


def _merge_vision_events(
    events: list[EventEntry],
    vision_timeline: VisionTimeline,
    start: int,
    end: int,
) -> None:
    """Merge vision timeline events into the main event timeline."""
    for ve in vision_timeline.events:
        if start <= ve.time <= end:
            events.append(EventEntry(
                time=ve.time,
                event_type=ve.event_type,
                description=ve.description,
                details={"vision_event": True},
            ))


def _add_laning_damage_events(
    ranked: list[tuple[int, EventEntry]],
    match: MatchData,
    player: PlayerData,
    start: int,
    end: int,
    lang: str = "zh",
) -> None:
    """Add laning hero-vs-hero damage (dealt and received).

    The hits between two heroes are summed per ``_DAMAGE_BUCKET`` seconds;
    damage a hero does to itself (Soul Ring, ...) is not an exchange.
    """
    player_idx = _get_player_index(match, player.player_slot)
    if player_idx is None:
        return
    en = lang == "en"

    def _bucketed(events, other_of) -> dict[tuple[int, int], dict]:
        buckets: dict[tuple[int, int], dict] = {}
        for de in events:
            if not (start <= de.time <= end):
                continue
            other = other_of(de)
            if other is None:
                continue
            b = buckets.setdefault((other, de.time // _DAMAGE_BUCKET),
                                   {"time": de.time, "damage": 0, "inflictors": []})
            b["damage"] += de.damage
            name = data_manager.ability_display_name(_inflictor_key(de.inflictor), lang)
            if name and name not in b["inflictors"]:
                b["inflictors"].append(name)
        return buckets

    def _how(bucket: dict) -> str:
        return (", " if en else "、").join(bucket["inflictors"][:3])

    dealt = _bucketed(player.laning_damage_events,
                      lambda de: de.target_idx if de.target_idx != player_idx else None)
    for (target_idx, _b), bucket in dealt.items():
        target_player = _get_player_by_index(match, target_idx)
        if en:
            target_name = target_player.hero_name if target_player else f"Player {target_idx}"
            desc = f"{player.hero_name} dealt {bucket['damage']} damage to {target_name} ({_how(bucket)})"
        else:
            target_name = target_player.hero_name if target_player else f"玩家{target_idx}"
            desc = (f"{_hero(player.hero_name, lang)} 对 {_hero(target_name, lang)} "
                    f"造成了 {bucket['damage']} 点伤害（{_how(bucket)}）")
        ranked.append((_PRIORITY_OTHER, EventEntry(
            time=bucket["time"],
            event_type="damage_dealt",
            description=desc,
            details={"source": player.hero_name, "target": target_name,
                     "damage": bucket["damage"], "inflictors": bucket["inflictors"]},
        )))

    for other in match.players:
        if other.team == player.team:
            continue
        taken = _bucketed(other.laning_damage_events,
                          lambda de: other.player_slot if de.target_idx == player_idx else None)
        for bucket in taken.values():
            if en:
                desc = (f"{player.hero_name} took {bucket['damage']} damage from "
                        f"{other.hero_name} ({_how(bucket)})")
            else:
                desc = (f"{_hero(player.hero_name, lang)} 受到了来自 {_hero(other.hero_name, lang)} 的 "
                        f"{bucket['damage']} 点伤害（{_how(bucket)}）")
            ranked.append((_PRIORITY_PLAYER, EventEntry(
                time=bucket["time"],
                event_type="damage_taken",
                description=desc,
                details={"source": other.hero_name, "target": player.hero_name,
                         "damage": bucket["damage"], "inflictors": bucket["inflictors"]},
            )))


def _inflictor_key(raw: str) -> str:
    """Damage inflictor key; older parser output marks a plain attack with
    "dota_unknown" or nothing."""
    return data_manager.ATTACK_INFLICTOR if raw in ("", "dota_unknown", "null") else raw


def _same_team(slot_a: int, slot_b: int) -> bool:
    return (slot_a < 128) == (slot_b < 128)


def _get_player_index(match: MatchData, player_slot: int) -> int | None:
    for i, p in enumerate(match.players):
        if p.player_slot == player_slot:
            return i
    return None


def _get_player_by_index(match: MatchData, index: int) -> PlayerData | None:
    if 0 <= index < len(match.players):
        return match.players[index]
    return None


# ===================================================================
# Map state
# ===================================================================

def _build_map_state(
    match: MatchData, time_s: int, player_state: HeroState | None = None,
    team: str = "", lang: str = "zh",
) -> dict:
    """Snapshot of the map at ``time_s``: wards, towers, Roshan."""
    state: dict = {}

    active_wards = _get_active_wards(match, time_s)
    state["active_wards"] = [
        {"ward_type": w.ward_type, "x": w.x, "y": w.y, "time_placed": w.time, "team": ward_team(w)}
        for w in active_wards
    ]
    # Observer wards give ground vision, sentries only true sight; both
    # ranges are the ones of the match's game version.
    rules = vision_rules(match)
    state["vision_points"] = [
        {"x": w.x, "y": w.y, "team": ward_team(w),
         "radius": rules.observer_vision if w.ward_type == "observer" else rules.sentry_true_sight}
        for w in active_wards
    ]

    if player_state is not None and player_state.has_position and team:
        counts = {"allied_observers": 0, "allied_sentries": 0, "enemy_observers": 0, "enemy_sentries": 0}
        for w in active_wards:
            if _distance(player_state.x, player_state.y, w.x, w.y) > _NEARBY_WARD_RADIUS:
                continue
            side = "allied" if ward_team(w) == team else "enemy"
            kind = "observers" if w.ward_type == "observer" else "sentries"
            counts[f"{side}_{kind}"] += 1
        state["nearby_wards"] = counts

    state["towers_destroyed"] = _get_towers_destroyed(match, time_s)
    state["towers"] = {
        side: {
            "standing": len(_get_alive_towers(match, side, time_s)),
            "lost": sorted(_get_destroyed_tower_names(match, side, time_s)),
        }
        for side in ("radiant", "dire")
    }
    state["roshan_killed"] = _roshan_killed_recently(match, time_s)
    return state


def _get_towers_destroyed(match: MatchData, time_s: int) -> list[dict]:
    """Tower kills from the objectives log up to ``time_s``."""
    destroyed = []
    for obj in match.objectives:
        obj_time = obj.get("time", 0)
        obj_type = obj.get("type", "")
        unit = str(obj.get("key") or obj.get("unit") or "")
        if obj_time > time_s:
            continue
        if obj_type in ("tower", "tower_kill", "CHAT_MESSAGE_TOWER_KILL") or (
            obj_type == "building_kill" and "tower" in unit.lower()
        ):
            destroyed.append({"time": obj_time, "type": obj_type, "unit": unit, "team": obj.get("team", "")})
    return destroyed


def _roshan_killed_recently(match: MatchData, time_s: int) -> dict | None:
    """Most recent Roshan kill he cannot have respawned from yet (8 minutes), or None."""
    last_rosh = None
    for obj in match.objectives:
        obj_time = obj.get("time", 0)
        if obj.get("type", "") in ("roshan", "CHAT_MESSAGE_ROSHAN_KILL") and obj_time <= time_s:
            if last_rosh is None or obj_time > last_rosh["time"]:
                last_rosh = {"time": obj_time, "team": obj.get("team", ""), "seconds_ago": time_s - obj_time}
    if last_rosh and last_rosh["seconds_ago"] > ROSHAN_RESPAWN_MIN_S:
        return None
    return last_rosh


# ===================================================================
# Match context
# ===================================================================

def _team_name(value) -> str:
    """Normalize objective team markers (2/3, "radiant"/"dire") to a name."""
    if value in (2, "2", "radiant"):
        return "radiant"
    if value in (3, "3", "dire"):
        return "dire"
    return ""


def _build_match_context(match: MatchData, time_s: int, team: str = "radiant") -> dict:
    """Match situation at ``time_s`` from ``team``'s perspective."""
    ctx: dict = {"team": team, "game_mode": match.game_mode}
    sign = 1 if team == "radiant" else -1
    minute = _time_to_minute(time_s)

    gold = xp = None
    if match.radiant_gold_adv:
        gold = series_at(match.radiant_gold_adv, minute)
    if match.radiant_xp_adv:
        xp = series_at(match.radiant_xp_adv, minute)
    if (gold is None or xp is None) and match.gold_xp_timeline:
        snap = _closest_timeline_snapshot(match, time_s)
        if snap is not None:
            if gold is None:
                gold = snap.radiant_gold - snap.dire_gold
            if xp is None:
                xp = snap.radiant_xp - snap.dire_xp
    if gold is not None:
        ctx["radiant_gold_adv"] = int(gold)
        ctx["gold_adv"] = int(gold) * sign
    if xp is not None:
        ctx["radiant_xp_adv"] = int(xp)
        ctx["xp_adv"] = int(xp) * sign

    # Kills so far: the scoreboard at the end of the match, the kill log
    # before that. Only a source without a kill log leaves the deaths
    # recorded in teamfights (solo kills are then missing).
    # Event times are truncated towards zero, so before the horn a kill at
    # -10.3 is dated -10: one second earlier keeps it out of its own score.
    score_time = time_s if time_s >= 0 else time_s - 1
    score = None if time_s >= match.duration > 0 else kills_before(match, score_time)
    if time_s >= match.duration > 0:
        radiant_kills, dire_kills, estimated = match.radiant_score, match.dire_score, False
    elif score is not None:
        (radiant_kills, dire_kills), estimated = score, False
    else:
        radiant_kills = dire_kills = 0
        for tf in match.teamfights:
            for tp in tf.players:
                if tp.deaths <= 0:
                    continue
                death_time = tf.start_time + (tp.death_time_offset or 0)
                if death_time <= time_s:
                    if tp.player_slot < 128:
                        dire_kills += tp.deaths
                    else:
                        radiant_kills += tp.deaths
        estimated = True
    ctx["radiant_kills"] = radiant_kills
    ctx["dire_kills"] = dire_kills
    ctx["team_kills"] = radiant_kills if team == "radiant" else dire_kills
    ctx["enemy_kills"] = dire_kills if team == "radiant" else radiant_kills
    ctx["kills_estimated"] = estimated

    enemy = "dire" if team == "radiant" else "radiant"
    ctx["team_towers_standing"] = len(_get_alive_towers(match, team, time_s))
    ctx["enemy_towers_standing"] = len(_get_alive_towers(match, enemy, time_s))
    ctx["team_towers_lost"] = sorted(_get_destroyed_tower_names(match, team, time_s))
    ctx["enemy_towers_lost"] = sorted(_get_destroyed_tower_names(match, enemy, time_s))

    recent_objectives = []
    for obj in match.objectives:
        obj_time = obj.get("time", 0)
        if time_s - 120 <= obj_time <= time_s:
            recent_objectives.append({
                "time": obj_time,
                "type": obj.get("type", ""),
                "unit": obj.get("key") or obj.get("unit") or "",
                "team": obj.get("team", ""),
            })
    ctx["recent_objectives"] = recent_objectives

    rosh = _roshan_killed_recently(match, time_s)
    if rosh is not None:
        ctx["roshan"] = {"seconds_ago": rosh["seconds_ago"], "team": _team_name(rosh.get("team"))}

    ctx["game_time"] = f"{time_s // 60}:{time_s % 60:02d}" if time_s >= 0 else f"-{abs(time_s) // 60}:{abs(time_s) % 60:02d}"
    ctx["duration"] = match.duration
    return ctx


def _aftermath(match: MatchData, player: PlayerData, time_s: int) -> dict | None:
    """Hero deaths and tower losses in the ``_AFTERMATH_SECONDS`` after a death.

    The event timeline ends at the death, so without this nothing says what
    the death was traded for (live verdicts claimed "no return kill" or "the
    team took a tower for it" either way), nor that the player bought back
    and died again. ``None`` without a kill log: then nothing is known.
    Heroes are display names, towers are tower keys; unconfirmed kill-log
    entries (possible reincarnations) are left out.
    """
    if not match.kills_log:
        return None
    by_slot = {p.player_slot: p for p in match.players}
    end = time_s + _AFTERMATH_SECONDS
    enemy_deaths: list[str] = []
    allied_deaths: list[str] = []
    own_kills: list[str] = []      # after a buyback, or with damage dealt before dying
    for k in match.kills_log:
        victim = by_slot.get(k.victim_slot)
        if victim is None or not k.confirmed or k.victim_slot == player.player_slot:
            continue
        if time_s < int(k.time) <= end:
            (allied_deaths if victim.team == player.team else enemy_deaths).append(victim.hero_name)
            if k.killer_slot == player.player_slot and victim.team != player.team:
                own_kills.append(victim.hero_name)
    enemy = "dire" if player.team == "radiant" else "radiant"

    def fallen(team: str) -> list[str]:
        return sorted(_get_destroyed_tower_names(match, team, end)
                      - _get_destroyed_tower_names(match, team, time_s))

    buybacks = [b.get("time") for b in player.buyback_log if isinstance(b.get("time"), (int, float))]
    bought_back = next((int(b - time_s) for b in sorted(buybacks) if time_s <= b <= end), None)
    died_again = next((int(k.time) - time_s for k in match.kills_log
                       if k.victim_slot == player.player_slot and k.confirmed and time_s < int(k.time) <= end), None)
    return {
        "seconds": _AFTERMATH_SECONDS,
        "enemy_deaths": enemy_deaths,
        "allied_deaths": allied_deaths,
        "own_kills": own_kills,
        "towers_taken": fallen(enemy),
        "towers_lost": fallen(player.team),
        "buyback_after": bought_back,   # seconds until the player bought back; None = did not
        "died_again_after": died_again,  # seconds until the player's next death in the window
    }


def _closest_timeline_snapshot(match: MatchData, time_s: int):
    """Closest gold/xp timeline snapshot to ``time_s``."""
    if not match.gold_xp_timeline:
        return None
    return min(match.gold_xp_timeline, key=lambda s: abs(s.time - time_s))
