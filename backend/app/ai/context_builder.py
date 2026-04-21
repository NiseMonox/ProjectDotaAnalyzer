"""Build DecisionContext from MatchData and DecisionPoint.

Given a match and a specific decision point (e.g. a death, a bad trade),
this module gathers all relevant context: player states, enemy/ally states,
recent event timeline, map vision, game phase, and match-level context.
The resulting DecisionContext is ready for LLM evaluation.
"""
from __future__ import annotations

from ..models import (
    AbilityEvent,
    LaningSnapshot,
    MatchData,
    PlayerData,
    TeamfightEvent,
    WardEvent,
)
from .models import (
    CausalChain,
    DecisionContext,
    DecisionPoint,
    EventEntry,
    HeroState,
    VisionAnalysis,
    VisionTimeline,
)


# ---------------------------------------------------------------------------
# Game phase boundaries (seconds)
# ---------------------------------------------------------------------------
_LANING_END = 600       # 10 minutes
_MID_GAME_END = 1500    # 25 minutes

# How far back to look for event timeline (seconds)
_EVENT_WINDOW = 60


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

    Returns
    -------
    DecisionContext
        All information an LLM needs to evaluate this decision.
    """
    t = decision_point.time
    slot = decision_point.player_slot

    player = _find_player(match, slot)
    if player is None:
        # Fallback: return minimal context
        return DecisionContext(
            decision_point=decision_point,
            player_state=HeroState(
                hero_name=decision_point.hero_name,
                player_slot=slot,
                team="unknown",
            ),
            game_phase=_get_game_phase(t),
        )

    player_state = _build_hero_state(player, t, match)

    allies: list[HeroState] = []
    enemies: list[HeroState] = []
    for p in match.players:
        if p.player_slot == slot:
            continue
        state = _build_hero_state(p, t, match)
        if p.team == player.team:
            allies.append(state)
        else:
            enemies.append(state)

    # --- Proximity-based filtering for enemies ---
    # Split enemies into nearby vs distant based on game-unit distance.
    # During laning (< 10min), heroes are spread across lanes; only include
    # truly nearby enemies in the primary list.  Distant enemies get a
    # separate key so the LLM knows they exist but aren't at this location.
    nearby_enemies, distant_enemies = _split_by_proximity(
        player_state, enemies, game_phase=_get_game_phase(t),
    )

    map_state = _build_map_state(match, t)

    # --- Pregame context enrichment ---
    game_phase = _get_game_phase(t)
    if game_phase == "pregame":
        # Add all 10 heroes' basic info for team composition analysis
        pregame_heroes = {"radiant": [], "dire": []}
        for p in match.players:
            hero_info = {
                "hero_name": p.hero_name,
                "player_slot": p.player_slot,
                "team": p.team,
                "attack_type": getattr(p, "attack_type", "unknown"),
            }
            pregame_heroes[p.team].append(hero_info)
        map_state["pregame_analysis"] = pregame_heroes

    match_context = _build_match_context(match, t)

    # Vision analysis
    vision_analysis: VisionAnalysis | None = None
    try:
        from .vision import analyze_vision
        player_pos = (player_state.x, player_state.y)
        vision_analysis = analyze_vision(match, slot, t, player_pos, lang=lang)
    except ImportError:
        pass  # vision module not yet available
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Vision analysis failed: %s", exc)

    # Enrich map_state with vision summary if available
    if vision_analysis is not None:
        vs = vision_analysis.state_at_decision
        if lang == "en":
            missing_names = ", ".join(vs.missing_enemy_names) if vs.missing_enemy_names else ""
            missing_part = f"{vs.missing_enemy_count} enemies missing" if vs.missing_enemy_count else "all enemies visible"
            if missing_names:
                missing_part += f" ({missing_names})"
            map_state["vision_summary"] = (
                f"{vs.allied_observer_count} observer wards, {missing_part}"
            )
        else:
            missing_names = "、".join(vs.missing_enemy_names) if vs.missing_enemy_names else ""
            missing_part = f"{vs.missing_enemy_count}名敌人消失" if vs.missing_enemy_count else "敌人均可见"
            if missing_names:
                missing_part += f"（{missing_names}）"
            map_state["vision_summary"] = (
                f"{vs.allied_observer_count}个观察守卫，{missing_part}"
            )

    event_timeline = _build_event_timeline(match, player, t, vision_analysis, lang=lang)

    if causal_chain and causal_chain.formatted_text:
        map_state["causal_chain_text"] = causal_chain.formatted_text

    # Record distant enemies in map_state so the LLM knows they exist
    # but understands they are NOT at this location.
    if distant_enemies:
        map_state["distant_enemies"] = [
            {
                "hero_name": e.hero_name,
                "x": e.x,
                "y": e.y,
                "is_alive": e.is_alive,
                "level": e.level,
            }
            for e in distant_enemies
        ]

    ctx = DecisionContext(
        decision_point=decision_point,
        player_state=player_state,
        enemy_states=nearby_enemies,
        ally_states=allies,
        event_timeline=event_timeline,
        map_state=map_state,
        game_phase=game_phase,
        match_context=match_context,
        vision_analysis=vision_analysis,
    )
    ctx.causal_chain = causal_chain
    return ctx


# ===================================================================
# Proximity filtering
# ===================================================================

# Distance thresholds (game units) for "nearby" classification.
# During laning, heroes are spread across 3 lanes so use a tighter radius.
# Mid/late game fights are more mobile, so use a wider radius.
_NEARBY_LANING = 2500     # ~1.5 screen widths in laning phase
_NEARBY_MID_LATE = 4000   # wider radius for mid/late game


def _split_by_proximity(
    player: HeroState,
    enemies: list[HeroState],
    game_phase: str,
) -> tuple[list[HeroState], list[HeroState]]:
    """Split enemies into nearby and distant lists based on position.

    If the player or an enemy has no valid position (x=0, y=0), the enemy
    is conservatively included as nearby to avoid losing information.
    """
    threshold = _NEARBY_LANING if game_phase in ("laning", "pregame") else _NEARBY_MID_LATE
    nearby: list[HeroState] = []
    distant: list[HeroState] = []

    px, py = player.x, player.y
    has_player_pos = not (px == 0 and py == 0)

    for e in enemies:
        ex, ey = e.x, e.y
        has_enemy_pos = not (ex == 0 and ey == 0)

        if not has_player_pos or not has_enemy_pos:
            # No position data — conservatively treat as nearby
            nearby.append(e)
            continue

        dist = ((px - ex) ** 2 + (py - ey) ** 2) ** 0.5
        if dist <= threshold:
            nearby.append(e)
        else:
            distant.append(e)

    return nearby, distant


# ===================================================================
# Player lookup
# ===================================================================

def _find_player(match: MatchData, player_slot: int) -> PlayerData | None:
    """Find a player by slot in the match."""
    for p in match.players:
        if p.player_slot == player_slot:
            return p
    return None


# ===================================================================
# Game phase
# ===================================================================

def _get_game_phase(time_seconds: int) -> str:
    """Determine the game phase at a given time."""
    if time_seconds < 30:
        return "pregame"
    elif time_seconds < _LANING_END:
        return "laning"
    elif time_seconds < _MID_GAME_END:
        return "mid_game"
    else:
        return "late_game"


# ===================================================================
# Hero state construction
# ===================================================================

def _build_hero_state(player: PlayerData, time_s: int, match: MatchData) -> HeroState:
    """Build a HeroState snapshot for a player at a specific time.

    Uses the best available data source:
    - Laning snapshots (every 5s) for times < 600s
    - Teamfight player data if player is in an active teamfight
    - Per-minute arrays (gold_t, xp_t, lh_t) as fallback
    """
    state = HeroState(
        hero_name=player.hero_name,
        player_slot=player.player_slot,
        team=player.team,
    )

    # --- Try laning snapshots first (high resolution) ---
    if time_s < _LANING_END and player.laning_snapshots:
        snap = _interpolate_laning_snapshot(player.laning_snapshots, time_s)
        if snap is not None:
            state.x = snap.x
            state.y = snap.y
            state.hp = snap.hp
            state.max_hp = snap.max_hp
            state.mana = snap.mana
            state.max_mana = snap.max_mana
            state.level = snap.level
            state.last_hits = snap.last_hits
            state.denies = snap.denies
            state.net_worth = snap.net_worth

    # --- Try teamfight data for position ---
    tf_state = _state_from_teamfight(match, player.player_slot, time_s)
    if tf_state is not None:
        state.x = tf_state["x"]
        state.y = tf_state["y"]
        if tf_state.get("gold_delta"):
            state.net_worth += tf_state["gold_delta"]

    # --- Fallback: per-minute arrays ---
    if not player.laning_snapshots or time_s >= _LANING_END:
        minute = _time_to_minute(time_s)
        if player.gold_t:
            idx = min(minute, len(player.gold_t) - 1)
            state.net_worth = player.gold_t[idx]
        if player.xp_t:
            idx = min(minute, len(player.xp_t) - 1)
            # Rough level estimate from XP (Dota 2 XP table approximation)
            state.level = _estimate_level_from_xp(player.xp_t[idx])
        if player.lh_t:
            idx = min(minute, len(player.lh_t) - 1)
            state.last_hits = player.lh_t[idx]

    # --- Items owned at this time ---
    state.items = _items_at_time(player, time_s)

    # --- Abilities available at this time ---
    state.abilities_available = _abilities_at_time(player, time_s)

    # --- Ability levels at this time ---
    state.ability_levels = _ability_levels_at_time(player, time_s)

    # --- Alive status (check if died recently) ---
    state.is_alive = _is_alive_at(match, player.player_slot, time_s)

    return state


def _interpolate_laning_snapshot(
    snapshots: list[LaningSnapshot], time_s: int,
) -> LaningSnapshot | None:
    """Find the closest laning snapshot, or interpolate between two.

    Snapshots are every 5 seconds. We find the two surrounding snapshots
    and linearly interpolate numeric fields.
    """
    if not snapshots:
        return None

    # Clamp to snapshot range
    if time_s <= snapshots[0].time:
        return snapshots[0]
    if time_s >= snapshots[-1].time:
        return snapshots[-1]

    # Binary search for surrounding snapshots
    lo, hi = 0, len(snapshots) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if snapshots[mid].time <= time_s:
            lo = mid
        else:
            hi = mid

    s0 = snapshots[lo]
    s1 = snapshots[hi]
    dt = s1.time - s0.time
    if dt == 0:
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


def _state_from_teamfight(
    match: MatchData, player_slot: int, time_s: int,
) -> dict | None:
    """If the player is in a teamfight at this time, return position info."""
    for tf in match.teamfights:
        if tf.start_time <= time_s <= tf.end_time:
            for tp in tf.players:
                if tp.player_slot == player_slot:
                    # Use the closest position snapshot
                    offset = time_s - tf.start_time
                    if tp.positions:
                        best = min(tp.positions, key=lambda p: abs(p.time_offset - offset))
                        return {
                            "x": best.x,
                            "y": best.y,
                            "gold_delta": tp.gold_delta,
                        }
                    return {"x": tp.x, "y": tp.y, "gold_delta": tp.gold_delta}
    return None


def _time_to_minute(time_s: int) -> int:
    """Convert game seconds to minute index (clamped to >= 0)."""
    return max(0, time_s // 60)


def _estimate_level_from_xp(xp: int) -> int:
    """Rough Dota 2 level estimate from total XP.

    Uses simplified XP thresholds. Not perfectly accurate but sufficient
    for context building.
    """
    # Approximate cumulative XP needed per level (Dota 2, patch ~7.35+)
    thresholds = [
        0, 230, 600, 1080, 1680, 2300, 2940, 3600, 4280, 5080,       # 1-10
        5900, 6740, 7640, 8865, 10115, 11390, 12690, 14015, 15415,    # 11-19
        16905, 18505, 20405, 22605, 25105, 27905, 31005, 34405,       # 20-27
        38105, 42105, 46405,                                           # 28-30
    ]
    level = 1
    for i, threshold in enumerate(thresholds):
        if xp >= threshold:
            level = i + 1
        else:
            break
    return min(level, 30)


def _items_at_time(player: PlayerData, time_s: int) -> list[str]:
    """Get list of item names the player owns at a given time.

    Includes items purchased up to and including this time. Does not
    track item selling/consumption (that data isn't in the model).
    """
    items = []
    for item_event in player.items:
        if item_event.time <= time_s:
            items.append(item_event.item_name)
    return items


def _abilities_at_time(player: PlayerData, time_s: int) -> list[str]:
    """Get list of ability names the player has skilled by this time."""
    abilities = []
    for skill_event in player.skills:
        if skill_event.time <= time_s:
            name = skill_event.ability_name
            if name not in abilities:
                abilities.append(name)
    return abilities


def _ability_levels_at_time(player: PlayerData, time_s: int) -> dict[str, int]:
    """Count skill points in each ability from SkillEvent data up to time_s.

    Returns dict mapping ability_name -> current_level (number of skill points).
    Uses ability_name (display name) from SkillEvent.
    Falls back to ability_key if ability_name is empty.
    """
    levels: dict[str, int] = {}
    for skill_event in player.skills:
        if skill_event.time <= time_s:
            name = skill_event.ability_name or skill_event.ability_key
            if name:
                levels[name] = levels.get(name, 0) + 1
    return levels


def _is_alive_at(match: MatchData, player_slot: int, time_s: int) -> bool:
    """Estimate if a player is alive at a given time.

    Uses teamfight death data as the primary source. A hero is considered
    dead for ~30 seconds after a teamfight death (rough respawn estimate).
    Without full kill log, this is a best-effort approximation.
    """
    for tf in match.teamfights:
        for tp in tf.players:
            if tp.player_slot == player_slot and tp.deaths > 0:
                if tp.death_time_offset is not None:
                    death_time = tf.start_time + tp.death_time_offset
                    # Rough respawn time: scales with level and game time
                    respawn_time = min(100, 5 + max(0, time_s // 60) * 2)
                    if death_time <= time_s <= death_time + respawn_time:
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
) -> list[EventEntry]:
    """Collect events in the window before the decision point.

    Gathers ability casts, damage events, ward placements, smoke events,
    kills, and deaths from the 60 seconds before `time_s`.
    If vision_analysis is provided, its timeline events are merged in.
    """
    start = max(0, time_s - _EVENT_WINDOW)
    end = time_s
    events: list[EventEntry] = []

    # --- Ability events (from player's own ability_events) ---
    for ae in player.ability_events:
        if start <= ae.time <= end:
            if lang == "en":
                desc = f"{player.hero_name} cast {ae.ability_name}"
            else:
                desc = f"{player.hero_name} 使用了 {ae.ability_name}"
            events.append(EventEntry(
                time=ae.time,
                event_type="ability_cast",
                description=desc,
                details={
                    "ability": ae.ability_name,
                    "target": ae.target or "",
                    "x": ae.x,
                    "y": ae.y,
                },
            ))

    # --- All ability events nearby (enemies and allies) ---
    for ae in match.ability_events:
        if ae.player_slot == player.player_slot:
            continue  # already added above
        if start <= ae.time <= end:
            if lang == "en":
                desc = f"{ae.hero_name} cast {ae.ability_name}"
            else:
                desc = f"{ae.hero_name} 使用了 {ae.ability_name}"
            events.append(EventEntry(
                time=ae.time,
                event_type="ability_cast",
                description=desc,
                details={
                    "ability": ae.ability_name,
                    "hero": ae.hero_name,
                    "player_slot": ae.player_slot,
                    "target": ae.target or "",
                },
            ))

    # --- Laning damage events (only during laning phase) ---
    if time_s < _LANING_END:
        _add_laning_damage_events(events, match, player, start, end, lang=lang)

    # --- Ward placements nearby ---
    for ward in match.wards:
        if start <= ward.time <= end:
            if lang == "en":
                team_str = "Allied" if _same_team(ward.player_slot, player.player_slot) else "Enemy"
                ward_type_str = "Sentry Ward" if ward.ward_type == "sentry" else "Observer Ward"
                desc = f"{team_str} placed {ward_type_str}"
            else:
                team_str = "己方" if _same_team(ward.player_slot, player.player_slot) else "敌方"
                ward_type_str = "真眼" if ward.ward_type == "sentry" else "假眼"
                desc = f"{team_str}放置了{ward_type_str}"
            events.append(EventEntry(
                time=ward.time,
                event_type="ward_placed",
                description=desc,
                details={
                    "ward_type": ward.ward_type,
                    "x": ward.x,
                    "y": ward.y,
                    "player_slot": ward.player_slot,
                },
            ))

    # --- Smoke events ---
    for smoke in match.smokes:
        if start <= smoke.time <= end:
            if lang == "en":
                team_str = "Allied" if smoke.team == player.team else "Enemy"
                desc = f"{team_str} used Smoke of Deceit"
            else:
                team_str = "己方" if smoke.team == player.team else "敌方"
                desc = f"{team_str}使用了烟雾"
            events.append(EventEntry(
                time=smoke.time,
                event_type="smoke_used",
                description=desc,
                details={
                    "team": smoke.team,
                    "participants": smoke.player_slots,
                    "x": smoke.x,
                    "y": smoke.y,
                },
            ))

    # --- Teamfight kills and deaths ---
    for tf in match.teamfights:
        if tf.start_time > end or tf.end_time < start:
            continue
        for tp in tf.players:
            if tp.deaths > 0 and tp.death_time_offset is not None:
                death_abs = tf.start_time + tp.death_time_offset
                if start <= death_abs <= end:
                    if lang == "en":
                        desc = f"{tp.hero_name} died in teamfight"
                    else:
                        desc = f"{tp.hero_name} 在团战中阵亡"
                    events.append(EventEntry(
                        time=int(death_abs),
                        event_type="death",
                        description=desc,
                        details={
                            "hero": tp.hero_name,
                            "player_slot": tp.player_slot,
                            "teamfight_start": tf.start_time,
                        },
                    ))
            for kt in tp.kill_targets:
                kill_abs = tf.start_time + kt.get("time_offset", 0)
                if start <= kill_abs <= end:
                    if lang == "en":
                        desc = f"{tp.hero_name} killed {kt.get('target', 'unknown')}"
                    else:
                        desc = f"{tp.hero_name} 击杀了 {kt.get('target', '未知')}"
                    events.append(EventEntry(
                        time=int(kill_abs),
                        event_type="kill",
                        description=desc,
                        details={
                            "killer": tp.hero_name,
                            "target": kt.get("target", ""),
                            "killer_slot": tp.player_slot,
                        },
                    ))

    # --- Vision events (from vision analysis) ---
    if vision_analysis is not None:
        _merge_vision_events(events, vision_analysis.timeline, start, end)

    # Sort by time
    events.sort(key=lambda e: e.time)
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
    """Add laning damage events (hero-vs-hero) to the event list."""
    player_idx = _get_player_index(match, player.player_slot)
    if player_idx is None:
        return

    # Damage dealt by this player
    for de in player.laning_damage_events:
        if start <= de.time <= end:
            target_player = _get_player_by_index(match, de.target_idx)
            if lang == "en":
                target_name = target_player.hero_name if target_player else f"Player {de.target_idx}"
                inflictor_str = de.inflictor if de.inflictor != "attack" else "auto-attack"
                desc = f"{player.hero_name} dealt {de.damage} damage to {target_name} ({inflictor_str})"
            else:
                target_name = target_player.hero_name if target_player else f"玩家{de.target_idx}"
                inflictor_str = de.inflictor if de.inflictor != "attack" else "普通攻击"
                desc = f"{player.hero_name} 对 {target_name} 造成了 {de.damage} 点伤害（{inflictor_str}）"
            events.append(EventEntry(
                time=de.time,
                event_type="damage_dealt",
                description=desc,
                details={
                    "source": player.hero_name,
                    "target": target_name,
                    "damage": de.damage,
                    "inflictor": de.inflictor,
                },
            ))

    # Damage received by this player from enemies
    for other_player in match.players:
        if other_player.player_slot == player.player_slot:
            continue
        if other_player.team == player.team:
            continue
        for de in other_player.laning_damage_events:
            if start <= de.time <= end and de.target_idx == player_idx:
                if lang == "en":
                    inflictor_str = de.inflictor if de.inflictor != "attack" else "auto-attack"
                    desc = (
                        f"{player.hero_name} took {de.damage} damage from "
                        f"{other_player.hero_name} ({inflictor_str})"
                    )
                else:
                    inflictor_str = de.inflictor if de.inflictor != "attack" else "普通攻击"
                    desc = (
                        f"{player.hero_name} 受到了来自 {other_player.hero_name} 的 "
                        f"{de.damage} 点伤害（{inflictor_str}）"
                    )
                events.append(EventEntry(
                    time=de.time,
                    event_type="damage_taken",
                    description=desc,
                    details={
                        "source": other_player.hero_name,
                        "target": player.hero_name,
                        "damage": de.damage,
                        "inflictor": de.inflictor,
                    },
                ))


def _same_team(slot_a: int, slot_b: int) -> bool:
    """Check if two player slots are on the same team."""
    return (slot_a < 128) == (slot_b < 128)


def _get_player_index(match: MatchData, player_slot: int) -> int | None:
    """Get the index of a player in the match.players list."""
    for i, p in enumerate(match.players):
        if p.player_slot == player_slot:
            return i
    return None


def _get_player_by_index(match: MatchData, index: int) -> PlayerData | None:
    """Get a player by their index in the match.players list."""
    if 0 <= index < len(match.players):
        return match.players[index]
    return None


# ===================================================================
# Map state
# ===================================================================

def _build_map_state(match: MatchData, time_s: int) -> dict:
    """Build a snapshot of map state at a given time.

    Includes:
    - active_wards: wards that have been placed and not yet destroyed
    - towers: which towers are still standing (from objectives)
    - vision_areas: approximate areas with friendly vision
    """
    state: dict = {}

    # --- Active wards ---
    active_wards = _get_active_wards(match, time_s)
    state["active_wards"] = [
        {
            "ward_type": w.ward_type,
            "x": w.x,
            "y": w.y,
            "time_placed": w.time,
            "team": "radiant" if w.player_slot < 128 else "dire",
        }
        for w in active_wards
    ]

    # Observer ward provides ~1600 unit vision radius, sentry ~1000
    vision_points: list[dict] = []
    for w in active_wards:
        radius = 1600 if w.ward_type == "observer" else 1000
        vision_points.append({
            "x": w.x,
            "y": w.y,
            "radius": radius,
            "team": "radiant" if w.player_slot < 128 else "dire",
        })
    state["vision_points"] = vision_points

    # --- Tower status from objectives timeline ---
    towers_destroyed = _get_towers_destroyed(match, time_s)
    state["towers_destroyed"] = towers_destroyed

    # --- Roshan status ---
    state["roshan_killed"] = _roshan_killed_recently(match, time_s)

    return state


def _get_active_wards(match: MatchData, time_s: int) -> list[WardEvent]:
    """Get wards that are active (placed and not yet destroyed) at a given time.

    Observer wards last up to 360 seconds (6 min) if not destroyed.
    Sentry wards last up to 420 seconds (7 min) if not destroyed.
    """
    active: list[WardEvent] = []
    for w in match.wards:
        if w.time > time_s:
            continue  # not yet placed

        # Check if destroyed before this time
        if w.destroyed_time is not None and w.destroyed_time <= time_s:
            continue

        # Check natural expiry (if no destroyed_time recorded)
        max_duration = 360 if w.ward_type == "observer" else 420
        if w.destroyed_time is None and (time_s - w.time) > max_duration:
            continue

        active.append(w)
    return active


def _get_towers_destroyed(match: MatchData, time_s: int) -> list[dict]:
    """Get list of towers destroyed before a given time from objectives."""
    destroyed = []
    for obj in match.objectives:
        obj_time = obj.get("time", 0)
        obj_type = obj.get("type", "")
        if obj_time <= time_s and obj_type in ("tower", "building_kill", "CHAT_MESSAGE_TOWER_KILL"):
            destroyed.append({
                "time": obj_time,
                "type": obj_type,
                "unit": obj.get("unit", obj.get("key", "")),
                "team": obj.get("team", ""),
            })
    return destroyed


def _roshan_killed_recently(match: MatchData, time_s: int) -> dict | None:
    """Check if Roshan was killed recently (within last 480s = 8 min).

    Returns the most recent Roshan kill info, or None.
    """
    last_rosh = None
    for obj in match.objectives:
        obj_time = obj.get("time", 0)
        obj_type = obj.get("type", "")
        if obj_type in ("roshan", "CHAT_MESSAGE_ROSHAN_KILL") and obj_time <= time_s:
            if last_rosh is None or obj_time > last_rosh["time"]:
                last_rosh = {
                    "time": obj_time,
                    "team": obj.get("team", ""),
                    "seconds_ago": time_s - obj_time,
                }
    # Aegis expires after 300s, but info still relevant for 480s
    if last_rosh and last_rosh["seconds_ago"] > 480:
        return None
    return last_rosh


# ===================================================================
# Match context
# ===================================================================

def _build_match_context(match: MatchData, time_s: int) -> dict:
    """Build high-level match context at a given time.

    Includes gold/xp advantage, current score estimate, and recent objectives.
    """
    ctx: dict = {}
    minute = _time_to_minute(time_s)

    # --- Gold advantage ---
    if match.radiant_gold_adv:
        idx = min(minute, len(match.radiant_gold_adv) - 1)
        ctx["radiant_gold_advantage"] = match.radiant_gold_adv[idx]
    elif match.gold_xp_timeline:
        # Compute from timeline
        snap = _closest_timeline_snapshot(match, time_s)
        if snap:
            ctx["radiant_gold_advantage"] = snap.radiant_gold - snap.dire_gold

    # --- XP advantage ---
    if match.radiant_xp_adv:
        idx = min(minute, len(match.radiant_xp_adv) - 1)
        ctx["radiant_xp_advantage"] = match.radiant_xp_adv[idx]
    elif match.gold_xp_timeline:
        snap = _closest_timeline_snapshot(match, time_s)
        if snap:
            ctx["radiant_xp_advantage"] = snap.radiant_xp - snap.dire_xp

    # --- Score estimate (count deaths from teamfights up to this time) ---
    radiant_kills = 0
    dire_kills = 0
    for tf in match.teamfights:
        if tf.start_time <= time_s:
            radiant_kills += tf.dire_deaths   # radiant kills = dire deaths
            dire_kills += tf.radiant_deaths
    ctx["estimated_score"] = {
        "radiant": radiant_kills,
        "dire": dire_kills,
    }

    # --- Recent objectives (within last 120s) ---
    recent_objectives = []
    for obj in match.objectives:
        obj_time = obj.get("time", 0)
        if time_s - 120 <= obj_time <= time_s:
            recent_objectives.append({
                "time": obj_time,
                "type": obj.get("type", ""),
                "unit": obj.get("unit", obj.get("key", "")),
                "team": obj.get("team", ""),
            })
    ctx["recent_objectives"] = recent_objectives

    # --- Game time description ---
    minutes = time_s // 60
    seconds = time_s % 60
    ctx["game_time"] = f"{minutes}:{seconds:02d}"
    ctx["duration"] = match.duration

    return ctx


def _closest_timeline_snapshot(match: MatchData, time_s: int):
    """Find the closest gold/xp timeline snapshot to a given time."""
    if not match.gold_xp_timeline:
        return None
    best = None
    best_diff = float("inf")
    for snap in match.gold_xp_timeline:
        diff = abs(snap.time - time_s)
        if diff < best_diff:
            best_diff = diff
            best = snap
    return best
