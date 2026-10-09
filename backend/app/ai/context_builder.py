"""Build DecisionContext from MatchData and DecisionPoint.

Given a match and a specific decision point (e.g. a death, a bad trade),
this module gathers all relevant context: the player's state, the allies and
enemies actually NEAR the player (world-unit radius), recent events around
the player, vision, game phase, and the match situation from the player's
team perspective. The resulting DecisionContext is ready for evaluation.

``match_context`` keys (consumed by the evaluator / agent prompts):
  team, gold_adv, xp_adv (positive = the player's team is ahead),
  radiant_gold_adv, radiant_xp_adv, team_kills, enemy_kills,
  radiant_kills, dire_kills, kills_estimated, team_towers_standing,
  enemy_towers_standing, team_towers_lost, enemy_towers_lost,
  enemies_missing, nearby_allies, nearby_enemies, unknown_position_enemies,
  recent_objectives, roshan, game_time, duration (+ player_position, set by
  the coach).
"""
from __future__ import annotations

import bisect
import logging

from ..models import LaningSnapshot, MatchData, PlayerData
from .models import (
    CausalChain,
    DecisionContext,
    DecisionPoint,
    EventEntry,
    HeroState,
    VisionAnalysis,
    VisionTimeline,
    is_time_unknown,
)
from .phases import LANING_END, game_phase, level_from_xp, series_at
from .vision import (
    _distance,
    _get_active_wards,
    _get_alive_towers,
    _get_destroyed_tower_names,
    describe_area,
    estimate_position,
    has_position,
    nearby_radius,
    ward_team,
)

logger = logging.getLogger(__name__)

# How far back to look for the event timeline (seconds)
_EVENT_WINDOW = 60
# Events further than this from the player are not about this decision.
_EVENT_RADIUS = 3000
_MAX_TIMELINE_EVENTS = 40
# Wards within this radius of the player are reported as "nearby".
_NEARBY_WARD_RADIUS = 2000
_MAX_ITEMS = 9

# Backward-compatible names (phase boundaries now live in phases.py).
_LANING_END = LANING_END
_NEARBY_LANING = nearby_radius(0)
_NEARBY_MID_LATE = nearby_radius(LANING_END)

# Consumables / one-shot items that only add noise to an inventory listing.
_CONSUMABLE_KEYS = {
    "tango", "tango_single", "flask", "clarity", "faerie_fire", "enchanted_mango",
    "famango", "smoke_of_deceit", "ward_observer", "ward_sentry", "ward_dispenser",
    "dust", "tpscroll", "tome_of_knowledge", "blood_grenade", "bottle_doubledamage",
}


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
    # timestamp (time=0); leave the phase unknown and skip the positional
    # context instead of building a misleading 0:00 picture.
    phase = "" if time_unknown else game_phase(t)

    player_state = _build_hero_state(player, t, match)
    death = decision_point.related_death
    if not player_state.has_position and death is not None and has_position((death.x, death.y)):
        player_state.x, player_state.y = death.x, death.y

    allies: list[HeroState] = []
    enemies: list[HeroState] = []
    for p in match.players:
        if p.player_slot == slot:
            continue
        state = _build_hero_state(p, t, match)
        if player_state.has_position and state.has_position:
            state.distance_to_player = round(
                _distance(player_state.x, player_state.y, state.x, state.y), 0)
        (allies if p.team == player.team else enemies).append(state)

    radius = nearby_radius(t)
    if time_unknown:
        # No moment to describe: other heroes' states at 0:00 would mislead.
        near_allies, far_allies, unknown_allies = [], [], []
        near_enemies, far_enemies, unknown_enemies = [], [], []
    else:
        near_allies, far_allies, unknown_allies = split_by_distance(allies, radius)
        near_enemies, far_enemies, unknown_enemies = split_by_distance(enemies, radius)
        # The killer was at the death even when its position is unknown.
        if death is not None and death.killer_hero:
            for e in list(unknown_enemies):
                if e.hero_name == death.killer_hero:
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
        match_context: dict = {"team": player.team, "duration": match.duration}
    else:
        match_context = _build_match_context(match, t, player.team)
        match_context["nearby_allies"] = sum(1 for a in near_allies if a.is_alive)
        match_context["nearby_enemies"] = sum(1 for e in near_enemies if e.is_alive)
        match_context["unknown_position_enemies"] = len(unknown_enemies)

    vision_analysis: VisionAnalysis | None = None
    if not time_unknown:
        try:
            from .vision import analyze_vision
            vision_analysis = analyze_vision(match, slot, t, (player_state.x, player_state.y), lang=lang)
        except Exception as exc:
            logger.warning("Vision analysis failed (t=%s): %s", t, exc)

    if vision_analysis is not None:
        vs = vision_analysis.state_at_decision
        match_context["enemies_missing"] = vs.missing_enemy_count
        match_context["vision_danger"] = vs.danger_level
        if lang == "en":
            names = ", ".join(vs.missing_enemy_names)
            missing_part = f"{vs.missing_enemy_count} enemies missing" if vs.missing_enemy_count else "all enemies visible"
            if names:
                missing_part += f" ({names})"
            map_state["vision_summary"] = f"{vs.allied_observer_count} allied observer wards, {missing_part}"
        else:
            names = "、".join(vs.missing_enemy_names)
            missing_part = f"{vs.missing_enemy_count}名敌人消失" if vs.missing_enemy_count else "敌人均可见"
            if names:
                missing_part += f"（{names}）"
            map_state["vision_summary"] = f"己方{vs.allied_observer_count}个观察守卫，{missing_part}"

    nearby_slots = {s.player_slot for s in near_allies + near_enemies}
    event_timeline = [] if time_unknown else _build_event_timeline(
        match, player, t, vision_analysis, lang=lang,
        player_pos=(player_state.x, player_state.y), nearby_slots=nearby_slots,
    )

    if causal_chain and causal_chain.formatted_text:
        map_state["causal_chain_text"] = causal_chain.formatted_text

    map_state["distant_allies"] = [_brief_state(s, lang) for s in far_allies]
    map_state["distant_enemies"] = [_brief_state(s, lang) for s in far_enemies]
    map_state["unknown_position_allies"] = [_brief_state(s, lang) for s in unknown_allies]
    map_state["unknown_position_enemies"] = [_brief_state(s, lang) for s in unknown_enemies]

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


def _get_game_phase(time_seconds: int) -> str:
    """Backward-compatible alias of :func:`phases.game_phase`."""
    return game_phase(time_seconds)


# ===================================================================
# Hero state construction
# ===================================================================

def _build_hero_state(player: PlayerData, time_s: int, match: MatchData) -> HeroState:
    """HeroState snapshot of a player at ``time_s``.

    Laning snapshots (5 s resolution, first 10 minutes) give exact
    HP/mana/level/CS/net worth; later moments fall back to the per-minute
    series. Positions come from snapshots, teamfight data or fresh ability
    casts (see :func:`vision.estimate_position`).
    """
    state = HeroState(hero_name=player.hero_name, player_slot=player.player_slot, team=player.team)

    snap = None
    if time_s < LANING_END and player.laning_snapshots:
        snap = _interpolate_laning_snapshot(player.laning_snapshots, time_s)
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
        if player.gold_t:
            state.net_worth = series_at(player.gold_t, minute)
        if player.xp_t:
            state.level = level_from_xp(series_at(player.xp_t, minute))
        if player.lh_t:
            state.last_hits = series_at(player.lh_t, minute)

    state.x, state.y = estimate_position(match, player.player_slot, time_s)
    state.items = _items_at_time(player, time_s)
    state.abilities_available = _abilities_at_time(player, time_s)
    state.ability_levels = _ability_levels_at_time(player, time_s)
    state.is_alive = _is_alive_at(match, player.player_slot, time_s)
    return state


def _interpolate_laning_snapshot(
    snapshots: list[LaningSnapshot], time_s: int,
) -> LaningSnapshot | None:
    """Closest laning snapshot, linearly interpolated between neighbours."""
    if not snapshots:
        return None
    if time_s <= snapshots[0].time:
        return snapshots[0]
    if time_s >= snapshots[-1].time:
        return snapshots[-1]

    times = [s.time for s in snapshots]
    hi = bisect.bisect_left(times, time_s)
    s1 = snapshots[hi]
    if s1.time == time_s:
        return s1
    s0 = snapshots[hi - 1]
    dt = s1.time - s0.time
    if dt <= 0:
        return s0
    ratio = (time_s - s0.time) / dt
    return LaningSnapshot(
        time=time_s,
        x=round(s0.x + (s1.x - s0.x) * ratio, 1),
        y=round(s0.y + (s1.y - s0.y) * ratio, 1),
        hp=int(s0.hp + (s1.hp - s0.hp) * ratio),
        max_hp=int(s0.max_hp + (s1.max_hp - s0.max_hp) * ratio),
        mana=int(s0.mana + (s1.mana - s0.mana) * ratio),
        max_mana=int(s0.max_mana + (s1.max_mana - s0.max_mana) * ratio),
        level=s0.level,  # level doesn't interpolate smoothly
        last_hits=int(s0.last_hits + (s1.last_hits - s0.last_hits) * ratio),
        denies=int(s0.denies + (s1.denies - s0.denies) * ratio),
        net_worth=int(s0.net_worth + (s1.net_worth - s0.net_worth) * ratio),
    )


def _time_to_minute(time_s: int) -> int:
    """Game seconds -> minute index (clamped to >= 0)."""
    return max(0, time_s // 60)


def _estimate_level_from_xp(xp: int) -> int:
    """Backward-compatible alias of :func:`phases.level_from_xp`."""
    return level_from_xp(xp)


def _items_at_time(player: PlayerData, time_s: int) -> list[str]:
    """Items bought up to ``time_s`` (most recent unique, consumables dropped).

    Selling/combining is not in the data, so this is an approximation of the
    inventory capped at ``_MAX_ITEMS`` entries.
    """
    names: list[str] = []
    seen: set[str] = set()
    for item_event in sorted(player.items, key=lambda i: i.time, reverse=True):
        if item_event.time > time_s:
            continue
        key = (item_event.item_key or "").lower()
        if key in _CONSUMABLE_KEYS or key.startswith("recipe"):
            continue
        name = item_event.item_name
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= _MAX_ITEMS:
            break
    names.reverse()
    return names


def _abilities_at_time(player: PlayerData, time_s: int) -> list[str]:
    """Abilities skilled by ``time_s`` (in skill order)."""
    abilities: list[str] = []
    for skill_event in player.skills:
        if skill_event.time <= time_s:
            name = skill_event.ability_name
            if name and name not in abilities:
                abilities.append(name)
    return abilities


def _ability_levels_at_time(player: PlayerData, time_s: int) -> dict[str, int]:
    """Skill points per ability up to ``time_s`` (name -> level)."""
    levels: dict[str, int] = {}
    for skill_event in player.skills:
        if skill_event.time <= time_s:
            name = skill_event.ability_name or skill_event.ability_key
            if name:
                levels[name] = levels.get(name, 0) + 1
    return levels


def _respawn_seconds(time_s: int) -> int:
    """Rough respawn duration (scales with game time; capped at 100 s)."""
    return min(100, 6 + max(0, time_s // 60) * 2)


def _is_alive_at(match: MatchData, player_slot: int, time_s: int) -> bool:
    """Best-effort alive check from teamfight death records."""
    for tf in match.teamfights:
        for tp in tf.players:
            if tp.player_slot == player_slot and tp.deaths > 0 and tp.death_time_offset is not None:
                # Death times elsewhere are whole seconds (floored).
                death_time = int(tf.start_time + tp.death_time_offset)
                if death_time <= time_s <= death_time + _respawn_seconds(death_time):
                    return False
    return True


# ===================================================================
# Event timeline
# ===================================================================

def _build_event_timeline(
    match: MatchData,
    player: PlayerData,
    time_s: int,
    vision_analysis: VisionAnalysis | None = None,
    lang: str = "zh",
    player_pos: tuple[float, float] = (0.0, 0.0),
    nearby_slots: set[int] | None = None,
) -> list[EventEntry]:
    """Events in the window before the decision that concern the player.

    Includes the player's casts, casts targeting the player or made by nearby
    heroes / close to the player, laning damage, nearby ward placements,
    smokes, teamfight kills/deaths and vision events. Capped to the most
    recent ``_MAX_TIMELINE_EVENTS`` entries.
    """
    start = max(0, time_s - _EVENT_WINDOW)
    end = time_s
    en = lang == "en"
    events: list[EventEntry] = []
    nearby_slots = nearby_slots or set()
    known_pos = has_position(player_pos)

    def _close(x: float | None, y: float | None) -> bool | None:
        if not known_pos or x is None or y is None or not has_position((x, y)):
            return None
        return _distance(player_pos[0], player_pos[1], x, y) <= _EVENT_RADIUS

    for ae in player.ability_events:
        if start <= ae.time <= end:
            events.append(EventEntry(
                time=ae.time,
                event_type="ability_cast",
                description=(f"{player.hero_name} cast {ae.ability_name}" if en
                             else f"{player.hero_name} 使用了 {ae.ability_name}"),
                details={"ability": ae.ability_name, "target": ae.target or "", "x": ae.x, "y": ae.y},
            ))

    for ae in match.ability_events:
        if ae.player_slot == player.player_slot or not (start <= ae.time <= end):
            continue
        relevant = ae.target == player.hero_name or ae.player_slot in nearby_slots
        if not relevant:
            close = _close(ae.x, ae.y)
            if close is None:
                is_enemy = (ae.player_slot < 128) != (player.player_slot < 128)
                relevant = not known_pos and is_enemy
            else:
                relevant = close
        if not relevant:
            continue
        target = f" on {ae.target}" if (en and ae.target) else (f"（目标：{ae.target}）" if ae.target else "")
        events.append(EventEntry(
            time=ae.time,
            event_type="ability_cast",
            description=(f"{ae.hero_name} cast {ae.ability_name}{target}" if en
                         else f"{ae.hero_name} 使用了 {ae.ability_name}{target}"),
            details={"ability": ae.ability_name, "hero": ae.hero_name,
                     "player_slot": ae.player_slot, "target": ae.target or ""},
        ))

    if time_s < LANING_END:
        _add_laning_damage_events(events, match, player, start, end, lang=lang)

    for ward in match.wards:
        if not (start <= ward.time <= end):
            continue
        if _close(ward.x, ward.y) is False:
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
        events.append(EventEntry(
            time=ward.time,
            event_type="ward_placed",
            description=desc,
            details={"ward_type": ward.ward_type, "x": ward.x, "y": ward.y,
                     "player_slot": ward.player_slot},
        ))

    for smoke in match.smokes:
        if start <= smoke.time <= end:
            own = smoke.team == player.team
            if en:
                desc = f"{'Allied' if own else 'Enemy'} team used Smoke of Deceit"
            else:
                desc = f"{'己方' if own else '敌方'}使用了诡计之雾"
            events.append(EventEntry(
                time=smoke.time,
                event_type="smoke_used",
                description=desc,
                details={"team": smoke.team, "participants": smoke.player_slots,
                         "x": smoke.x, "y": smoke.y},
            ))

    for tf in match.teamfights:
        if tf.start_time > end or tf.end_time < start:
            continue
        for tp in tf.players:
            if tp.deaths > 0 and tp.death_time_offset is not None:
                death_abs = tf.start_time + tp.death_time_offset
                if start <= death_abs <= end:
                    events.append(EventEntry(
                        time=int(death_abs),
                        event_type="death",
                        description=(f"{tp.hero_name} died in the teamfight" if en
                                     else f"{tp.hero_name} 在团战中阵亡"),
                        details={"hero": tp.hero_name, "player_slot": tp.player_slot,
                                 "teamfight_start": tf.start_time},
                    ))
            for kt in tp.kill_targets:
                kill_abs = tf.start_time + (kt.get("time_offset") or 0)
                if start <= kill_abs <= end:
                    target = kt.get("target") or ("unknown" if en else "未知")
                    events.append(EventEntry(
                        time=int(kill_abs),
                        event_type="kill",
                        description=(f"{tp.hero_name} killed {target}" if en
                                     else f"{tp.hero_name} 击杀了 {target}"),
                        details={"killer": tp.hero_name, "target": kt.get("target", ""),
                                 "killer_slot": tp.player_slot},
                    ))

    if vision_analysis is not None:
        _merge_vision_events(events, vision_analysis.timeline, start, end)

    events.sort(key=lambda e: e.time)
    if len(events) > _MAX_TIMELINE_EVENTS:
        events = events[-_MAX_TIMELINE_EVENTS:]
    return events


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
    events: list[EventEntry],
    match: MatchData,
    player: PlayerData,
    start: int,
    end: int,
    lang: str = "zh",
) -> None:
    """Add laning hero-vs-hero damage events (dealt and received)."""
    player_idx = _get_player_index(match, player.player_slot)
    if player_idx is None:
        return
    en = lang == "en"

    def _inflictor(raw: str) -> str:
        if raw == "attack":
            return "auto-attack" if en else "普通攻击"
        return raw

    for de in player.laning_damage_events:
        if start <= de.time <= end:
            target_player = _get_player_by_index(match, de.target_idx)
            if en:
                target_name = target_player.hero_name if target_player else f"Player {de.target_idx}"
                desc = f"{player.hero_name} dealt {de.damage} damage to {target_name} ({_inflictor(de.inflictor)})"
            else:
                target_name = target_player.hero_name if target_player else f"玩家{de.target_idx}"
                desc = f"{player.hero_name} 对 {target_name} 造成了 {de.damage} 点伤害（{_inflictor(de.inflictor)}）"
            events.append(EventEntry(
                time=de.time,
                event_type="damage_dealt",
                description=desc,
                details={"source": player.hero_name, "target": target_name,
                         "damage": de.damage, "inflictor": de.inflictor},
            ))

    for other in match.players:
        if other.team == player.team:
            continue
        for de in other.laning_damage_events:
            if start <= de.time <= end and de.target_idx == player_idx:
                if en:
                    desc = (f"{player.hero_name} took {de.damage} damage from "
                            f"{other.hero_name} ({_inflictor(de.inflictor)})")
                else:
                    desc = (f"{player.hero_name} 受到了来自 {other.hero_name} 的 "
                            f"{de.damage} 点伤害（{_inflictor(de.inflictor)}）")
                events.append(EventEntry(
                    time=de.time,
                    event_type="damage_taken",
                    description=desc,
                    details={"source": other.hero_name, "target": player.hero_name,
                             "damage": de.damage, "inflictor": de.inflictor},
                ))


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
    # Observer wards give 1600 ground vision; sentries 1000 true sight.
    state["vision_points"] = [
        {"x": w.x, "y": w.y, "radius": 1600 if w.ward_type == "observer" else 1000, "team": ward_team(w)}
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
    """Most recent Roshan kill within the last 8 minutes, or None."""
    last_rosh = None
    for obj in match.objectives:
        obj_time = obj.get("time", 0)
        if obj.get("type", "") in ("roshan", "CHAT_MESSAGE_ROSHAN_KILL") and obj_time <= time_s:
            if last_rosh is None or obj_time > last_rosh["time"]:
                last_rosh = {"time": obj_time, "team": obj.get("team", ""), "seconds_ago": time_s - obj_time}
    if last_rosh and last_rosh["seconds_ago"] > 480:
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
    ctx: dict = {"team": team}
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

    # Kills so far. Exact at the end of the match; before that only the
    # deaths recorded in teamfights are known (solo kills are missing).
    if time_s >= match.duration > 0:
        radiant_kills, dire_kills, estimated = match.radiant_score, match.dire_score, False
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


def _closest_timeline_snapshot(match: MatchData, time_s: int):
    """Closest gold/xp timeline snapshot to ``time_s``."""
    if not match.gold_xp_timeline:
        return None
    return min(match.gold_xp_timeline, key=lambda s: abs(s.time - time_s))
