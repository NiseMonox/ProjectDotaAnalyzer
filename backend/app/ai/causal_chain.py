"""Death causal chain analysis for AI coaching.

Given a death event, traces backwards through match data to reconstruct the
chain of events leading to the death and identify the root cause -- the
earliest moment where the player could have changed the outcome.

Every step is dated at or before the death and rests on data: positions
from :mod:`vision` (a hero without position data is "unknown", never
"missing"), HP from the position log / laning snapshots, head counts from
the heroes actually near the player (or, when the source has no positions,
from the heroes that took part in the fight record, worded as such). A
fight is judged by the numbers it had before heroes died in it
(:func:`vision.engagement`): the last survivor of an even fight was not
"outnumbered". A chain without evidence for a cause has no root cause.

Positions are world units; distance thresholds are real game units.
Supports Chinese ("zh") and English ("en").
"""
from __future__ import annotations

import bisect
from typing import Any

from ..models import AbilityEvent, MatchData, PlayerData, TeamfightEvent
from . import data_manager
from .models import CausalChain, CausalStep, DeathEvent, killed_by_fountain, killer_text
from .vision import (
    GROUP_ALLIES,
    PREGAME_START,
    THREAT_RADIUS,
    Engagement,
    _active_observers,
    _assess_danger,
    _distance,
    _format_game_time,
    _get_alive_towers,
    _track_enemy_visibility,
    area_code,
    count_nearby_heroes,
    deaths_near,
    describe_area,
    engagement,
    estimate_position,
    fog_threats,
    has_position,
    head_count,
    hero_alive_at,
    hp_pct_at,
    in_own_base,
    last_alive_time,
    nearby_radius,
    position_at,
    vision_rules,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_WINDOW = 120          # max seconds to look back
_STEP = 5                  # default sampling interval
_SAFE_TOWER_DIST = 2000    # world units: "close to an allied tower"
_HP_SAFE = 0.70
_MAX_STEPS = 8
_PREGAME_DEATH = 30        # deaths before 0:30 are level-1 fights (bounty runes)
_OUTNUMBERED_LOOKBACK = 30  # how far back "outnumbered since" is traced
_FARMING_PER_MINUTE = 4    # last hits in a minute that count as "farming" after laning
_MIN_REACTION_S = 5        # a step later than this before the death left no time to react
_WARNING_GAP = 30          # seconds between two "situation became dangerous" steps
_GROUP_ALLIES = GROUP_ALLIES  # with this many allies around, a move is the team's move
_LANING_END = 600          # end of the laning stage (game seconds)
_LONE_ENEMY_HIDDEN_S = 30  # a single hidden enemy counts once unseen for longer (as in vision._assess_danger)

_CTRL: dict[str, str] = {
    "ignore_warning": "high", "position_forward": "high",
    "no_retreat": "high", "position_unsafe": "high",
    "farm_under_threat": "high",
    "enemy_missing": "medium", "hidden_enemy_near": "medium", "ally_absent": "medium",
    "allies_fallen": "medium",
    "vision_loss": "low", "enemy_approach": "low",
    "cc_chain": "low", "damage_burst": "low", "death": "low",
}
_CTRL_RANK = {"high": 3, "medium": 2, "low": 1}
# What happened TO the player is a consequence, never the cause of the death.
_CONSEQUENCES = ("damage_burst", "cc_chain", "enemy_approach")

_UNKNOWN = (0.0, 0.0)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _fmt(t: int) -> str:
    return _format_game_time(t)


def _player(match: MatchData, slot: int) -> PlayerData | None:
    for p in match.players:
        if p.player_slot == slot:
            return p
    return None


def _snapshot_near(player: PlayerData, t: int, max_gap: int):
    """Laning snapshot closest to ``t`` within ``max_gap`` seconds."""
    snaps = player.laning_snapshots
    if not snaps:
        return None
    times = [s.time for s in snaps]
    i = bisect.bisect_left(times, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(snaps) and (best is None or abs(snaps[j].time - t) < abs(best.time - t)):
            best = snaps[j]
    if best is None or abs(best.time - t) > max_gap:
        return None
    return best


def _last_hits_at(player: PlayerData, t: int) -> int | None:
    """Last hits at second ``t`` from the laning snapshots (None when there is none)."""
    snap = _snapshot_near(player, t, 10)
    return snap.last_hits if snap is not None else None


def _last_hits_in_minute(player: PlayerData, t: int) -> int | None:
    """Last hits gained during the game minute ``t`` lies in (per-minute series)."""
    lh = player.lh_t
    minute = t // 60
    if t < 0 or minute + 1 >= len(lh):
        return None
    return lh[minute + 1] - lh[minute]


def _tower_distance(match: MatchData, team: str, pos: tuple[float, float], t: int) -> float:
    if not has_position(pos):
        return float("inf")
    if in_own_base(team, pos[0], pos[1]):
        return 0.0
    dists = [_distance(pos[0], pos[1], tx, ty) for _n, tx, ty in _get_alive_towers(match, team, t)]
    return min(dists) if dists else float("inf")


def _hp_fraction(match: MatchData, slot: int, t: int) -> float:
    """HP of a hero as 0..1 at ``t``; -1 when unknown."""
    pct = hp_pct_at(match, slot, t)
    return pct / 100.0 if pct is not None else -1.0


# ---------------------------------------------------------------------------
# 1. Reconstruct player position path
# ---------------------------------------------------------------------------

def _reconstruct_player_path(
    match: MatchData, player: PlayerData,
    start_time: int, end_time: int, step: int = _STEP,
) -> list[tuple[int, float, float]]:
    """Known positions of the player between two times."""
    seen: dict[int, tuple[float, float]] = {}
    for t in range(start_time, end_time + 1, step):
        pos = estimate_position(match, player.player_slot, t)
        if has_position(pos):
            seen[t] = pos
    for ae in list(player.ability_events) + [
        a for a in match.ability_events if a.player_slot == player.player_slot
    ]:
        if start_time <= ae.time <= end_time and ae.x is not None and ae.y is not None:
            if has_position((ae.x, ae.y)):
                seen[ae.time] = (float(ae.x), float(ae.y))
    return [(t, xy[0], xy[1]) for t, xy in sorted(seen.items())]


# ---------------------------------------------------------------------------
# 2. Find analysis window
# ---------------------------------------------------------------------------

def _find_analysis_window(
    match: MatchData, player_slot: int, death_time: int, walk_from: int | None = None,
) -> int:
    """Walk backwards to the last 'safe' moment (near a tower with few
    enemies hidden / healthy, or every enemy accounted for).

    ``walk_from`` (e.g. the start of the teamfight the player died in) lets
    the search skip the fight itself, where everyone is visible anyway. The
    window may reach back before the horn, never past the player's previous
    death.
    """
    player = _player(match, player_slot)
    if player is None:
        return death_time - 30
    team = player.team
    earliest = max(death_time - _MAX_WINDOW, PREGAME_START)
    first = min(death_time, walk_from if walk_from is not None else death_time) - _STEP

    for t in range(first, earliest - 1, -_STEP):
        if hero_alive_at(match, player_slot, t) is False:
            return t + _STEP  # what came before the respawn belongs to the previous death
        pos = estimate_position(match, player_slot, t)
        near_tower = _tower_distance(match, team, pos, t) < _SAFE_TOWER_DIST
        evis = _track_enemy_visibility(match, team, player_slot, t)
        hidden = len(fog_threats(match, evis, pos, t))
        hp_safe = _hp_fraction(match, player_slot, t) > _HP_SAFE
        if near_tower and (hidden <= 1 or hp_safe):
            return t
        # "Nobody hidden nearby" needs data: an enemy without a position is
        # not known to be away.
        if hidden == 0 and evis and all(e.state != "unknown" for e in evis):
            return t
    return earliest


# ---------------------------------------------------------------------------
# 3. Sample state snapshots
# ---------------------------------------------------------------------------

def _sample_states(
    match: MatchData, player: PlayerData, player_slot: int,
    start_time: int, end_time: int, step: int = _STEP,
    lang: str = "zh",
) -> list[dict[str, Any]]:
    """Gather a lightweight state snapshot at each timestep.

    ``missing_enemies`` only counts enemies hidden in the fog within reach
    of the player (:func:`vision.fog_threats`); enemies whose position the
    data does not have are counted apart and never treated as missing.
    """
    team = player.team
    observer_vision = vision_rules(match).observer_vision
    states: list[dict[str, Any]] = []
    times = list(range(start_time, end_time + 1, step))
    if not times or times[-1] != end_time:
        times.append(end_time)
    for t in times:
        pos = estimate_position(match, player_slot, t)
        known = has_position(pos)
        td = _tower_distance(match, team, pos, t)
        evis = _track_enemy_visibility(match, team, player_slot, t)
        threats = fog_threats(match, evis, pos, t) if known else []
        missing = [e.hero_name for e in threats]
        hidden_for = {e.hero_name: e.seconds_missing for e in threats}
        if known:
            dlevel, dreasons, _safe, _wc = _assess_danger(
                match, team, player_slot, t, pos, evis, lang=lang)
        else:
            dlevel, dreasons = "unknown", []
        allies = 0
        # Allied observers standing at t, and the ones that see the player.
        observers = dict(_active_observers(match, team, t))
        covering: dict[int, Any] = {}
        if known:
            allies = count_nearby_heroes(
                match, team, pos[0], pos[1], t, exclude_slot=player_slot,
                radius=nearby_radius(t),
            )
            covering = {i: w for i, w in observers.items()
                        if _distance(pos[0], pos[1], w.x, w.y) <= observer_vision}
        states.append({
            "time": t, "x": pos[0], "y": pos[1], "hp_pct": _hp_fraction(match, player_slot, t),
            "tower_dist": td, "missing_enemies": len(missing),
            "missing_names": missing, "hidden_for": hidden_for,
            "enemy_states": {e.hero_name: e.state for e in evis},
            "missing_longest": max((e.seconds_missing for e in threats), default=0),
            "unknown_enemies": sum(1 for e in evis if e.state == "unknown"),
            "danger_level": dlevel,
            "danger_reasons": dreasons,
            "nearby_allies": allies, "ward_coverage": bool(covering),
            "covering_wards": covering, "standing_wards": set(observers),
            "last_hits": _last_hits_at(player, t),
            "minute_last_hits": _last_hits_in_minute(player, t),
        })
    return states


# ---------------------------------------------------------------------------
# 4. Detect transitions
# ---------------------------------------------------------------------------

def _detect_transitions(
    states: list[dict[str, Any]], match: MatchData,
    player: PlayerData, player_slot: int,
    lang: str = "zh",
) -> list[CausalStep]:
    """Compare consecutive snapshots to detect significant transitions."""
    if len(states) < 2:
        return []
    en = lang == "en"
    steps: list[CausalStep] = []
    death_time = states[-1]["time"]
    window_start = states[0]["time"]
    incoming = _get_incoming_abilities(match, player, window_start, death_time)

    emitted_ward = emitted_unsafe = emitted_farm = False
    last_warning: int | None = None
    dorder = {"safe": 0, "caution": 1, "danger": 2, "critical": 3, "unknown": 0}
    inf = float("inf")

    for i in range(1, len(states)):
        prev, curr = states[i - 1], states[i]
        t = curr["time"]
        pos = (curr["x"], curr["y"])
        known = has_position(pos)

        # Ward loss: a ward that covered the player is gone (walking out of a
        # ward's range is the player's move, not a lost ward).
        lost = [w for i, w in prev["covering_wards"].items() if i not in curr["standing_wards"]]
        if not emitted_ward and lost and not curr["ward_coverage"] and known:
            area = describe_area(lost[0].x, lost[0].y, lang=lang)
            if lost[0].dewarded:
                desc = (f"The observer ward at {area} was destroyed: no more vision around the player" if en
                        else f"{area}的侦察守卫被摧毁，玩家周围失去视野")
            else:
                desc = (f"The observer ward at {area} expired: no more vision around the player" if en
                        else f"{area}的侦察守卫到期，玩家周围失去视野")
            steps.append(CausalStep(
                time=t, step_type="vision_loss", description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"]))
            emitted_ward = True

        # More enemies hidden in the fog near the player. One that was
        # visible at the previous sample left allied vision; one that was
        # hidden already has come within reach, or the player walked towards
        # it: two different things, two steps. One that has only just
        # respawned (or has no earlier data) did neither and is not named.
        new_miss = curr["missing_enemies"] - prev["missing_enemies"]
        names = [n for n in curr["missing_names"] if n not in prev.get("missing_names", [])]
        if new_miss >= 2 or (new_miss == 1 and curr["missing_enemies"] >= 3):
            left = [n for n in names if prev["enemy_states"].get(n) == "visible"]
            came = [n for n in names if prev["enemy_states"].get(n) == "fog"]
            n = curr["missing_enemies"]
            total = ""
            if new_miss == 1:
                total = (f"; {n} enemies are now hidden in the fog near the player" if en
                         else f"，玩家附近已有{n}名敌方藏在迷雾中")
            details = {"missing_before": prev["missing_enemies"], "missing_after": n}
            if left:
                desc = (f"{', '.join(left[:3])} moved out of allied vision close to the player" if en
                        else f"{'、'.join(_hero(n, lang) for n in left[:3])}在玩家附近脱离己方视野")
                steps.append(CausalStep(
                    time=t, step_type="enemy_missing", description=desc + total,
                    significance="contributing", position=pos,
                    danger_level=curr["danger_level"], details=dict(details)))
                total = ""
            if came:
                longest = max(curr["hidden_for"].get(name, 0) for name in came)
                if en:
                    since = "" if longest <= 0 else (
                        f" for {longest}s" if len(came) == 1 else f" (for up to {longest}s)")
                    desc = (f"{', '.join(came[:3])}, already out of sight{since}, "
                            f"{'was' if len(came) == 1 else 'were'} now within {THREAT_RADIUS} units of the player")
                else:
                    since = "" if longest <= 0 else (
                        f"（已{longest}秒）" if len(came) == 1 else f"（最长已{longest}秒）")
                    desc = (f"{'、'.join(_hero(n, lang) for n in came[:3])}此前就不在己方视野内{since}，"
                            f"此时距玩家已不足{THREAT_RADIUS}码")
                steps.append(CausalStep(
                    time=t, step_type="hidden_enemy_near", description=desc + total,
                    significance="contributing", position=pos,
                    danger_level=curr["danger_level"], details=dict(details)))

        # Danger escalation without retreating -> ignore_warning. The step
        # is about hidden enemies: one that only just left vision (the
        # danger reasons then do not even list it) does not make one.
        cv, pv = dorder.get(curr["danger_level"], 0), dorder.get(prev["danger_level"], 0)
        hidden = curr["missing_enemies"] >= 2 or (
            curr["missing_enemies"] == 1 and curr["missing_longest"] > _LONE_ENEMY_HIDDEN_S)
        if (cv >= 2 and cv > pv and known and hidden
                and (last_warning is None or t - last_warning >= _WARNING_GAP)):
            retreated = (
                curr["tower_dist"] < prev["tower_dist"] - 200
                if curr["tower_dist"] < inf and prev["tower_dist"] < inf else False
            )
            if not retreated:
                last_warning = t
                # With the team around, staying is the team's decision.
                company = curr["nearby_allies"] if curr["nearby_allies"] >= _GROUP_ALLIES else 0
                if en:
                    lvl = "very dangerous" if curr["danger_level"] == "critical" else "dangerous"
                    rs = "; ".join(curr.get("danger_reasons", [])[:2])
                    who = f"the player and the {company} allies nearby" if company else "the player"
                    desc = (f"The situation became {lvl}" + (f" ({rs})" if rs else "")
                            + f" but {who} did not move back towards safety")
                else:
                    lvl = "极度危险" if curr["danger_level"] == "critical" else "危险"
                    rs = "；".join(curr.get("danger_reasons", [])[:2])
                    who = f"玩家和身边的{company}名队友" if company else "玩家"
                    desc = f"局势变得{lvl}" + (f"（{rs}）" if rs else "") + f"，但{who}没有向安全位置后撤"
                steps.append(CausalStep(
                    time=t, step_type="ignore_warning", description=desc,
                    significance="contributing", position=pos,
                    danger_level=curr["danger_level"]))

        # Moving farther from towers while enemies are hidden nearby. On the
        # own half of the map (towers gone) that is not pushing forward, and
        # with the team around it is the team's move, not the player's.
        alone = curr["nearby_allies"] < _GROUP_ALLIES
        if (known and alone and curr["tower_dist"] < inf and prev["tower_dist"] < inf
                and curr["tower_dist"] > prev["tower_dist"] + 500
                and curr["tower_dist"] > _SAFE_TOWER_DIST
                and curr["missing_enemies"] >= 2
                and area_code(pos[0], pos[1])[1] != player.team):
            area = describe_area(pos[0], pos[1], lang=lang)
            dist = int(curr["tower_dist"])
            desc = (f"Player moved toward {area}, ~{dist} units from the nearest allied tower" if en
                    else f"玩家向{area}推进，距最近友方塔约{dist}码")
            steps.append(CausalStep(
                time=t, step_type="position_forward", description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"]))

        # Entering an unwarded area far from towers (standing in a lane
        # during the laning stage is where the hero belongs, warded or not)
        in_lane = t < _LANING_END and area_code(pos[0], pos[1])[0] == "lane"
        if (not emitted_unsafe and known and alone and not in_lane and not curr["ward_coverage"]
                and curr["tower_dist"] > _SAFE_TOWER_DIST
                and curr["missing_enemies"] >= 2):
            area = describe_area(pos[0], pos[1], lang=lang)
            n = curr["missing_enemies"]
            desc = (f"Player was at {area} without ward coverage while {n} enemies were hidden in the fog nearby" if en
                    else f"玩家身处没有眼位的{area}，附近有{n}名敌方藏在迷雾中")
            steps.append(CausalStep(
                time=t, step_type="position_unsafe", description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"]))
            emitted_unsafe = True

        # Farming in an exposed spot while the map is dark -> greedy farm.
        # Exact last hits exist for the laning stage only; later the
        # per-minute series says how much was farmed in that minute. With
        # the team around, those last hits are the creeps of a push or a
        # fight, not the player farming (live: four heroes at the enemy
        # tier-1 tower became "kept farming in the enemy jungle").
        gained = None
        if curr.get("last_hits") is not None and prev.get("last_hits") is not None:
            gained = curr["last_hits"] - prev["last_hits"]
        per_minute = curr.get("minute_last_hits") if gained is None and alone else None
        if (not emitted_farm and known
                and ((gained is not None and gained > 0) or (per_minute or 0) >= _FARMING_PER_MINUTE)
                and curr["missing_enemies"] >= 3
                and curr["tower_dist"] > _SAFE_TOWER_DIST):
            area = describe_area(pos[0], pos[1], lang=lang)
            n = curr["missing_enemies"]
            if gained:
                how = f"+{gained} CS" if en else f"+{gained}补刀"
            else:
                how = f"{per_minute} last hits in that minute" if en else f"这一分钟补了{per_minute}刀"
            desc = (f"Kept farming at {area} ({how}) while {n} enemies were hidden in the fog nearby" if en
                    else f"在{area}继续刷钱（{how}），而附近有{n}名敌方藏在迷雾中")
            steps.append(CausalStep(
                time=t, step_type="farm_under_threat", description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"]))
            emitted_farm = True

        # No ally around any more: they left or the player walked off. An
        # ally that died there did not leave the player alone by anyone's
        # choice, so that is stated and never the root cause.
        if prev["nearby_allies"] >= 1 and curr["nearby_allies"] == 0 and curr["missing_enemies"] >= 2:
            n = curr["missing_enemies"]
            died = any((victim < 128) == (player_slot < 128)
                       for _when, victim in deaths_near(match, player_slot, prev["time"], t + 1))
            if died:
                desc = (f"The ally near the player died: no allied hero around any more, with {n} enemies "
                        f"hidden in the fog nearby" if en
                        else f"玩家身边的队友阵亡，附近已经没有队友，而周围有{n}名敌方藏在迷雾中")
            else:
                desc = (f"No allied hero near the player any more, with {n} enemies hidden in the fog nearby" if en
                        else f"玩家附近已经没有队友，而周围有{n}名敌方藏在迷雾中")
            steps.append(CausalStep(
                time=t, step_type="ally_absent", description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"], details={"root_eligible": not died}))

        # Rapid HP loss (a 0% sample is the corpse of an earlier death that
        # a buyback ended, not a hit the player took and survived)
        if prev["hp_pct"] >= 0 and curr["hp_pct"] > 0 and prev["hp_pct"] - curr["hp_pct"] > 0.30:
            pb, pa = int(prev["hp_pct"] * 100), int(curr["hp_pct"] * 100)
            desc = f"HP dropped from {pb}% to {pa}%" if en else f"生命值从{pb}%骤降至{pa}%"
            steps.append(CausalStep(
                time=t, step_type="damage_burst", description=desc,
                significance="consequence", position=pos,
                danger_level=curr["danger_level"]))

    _add_ability_steps(steps, incoming, lang=lang)
    steps.sort(key=lambda s: s.time)
    return steps


def _add_ability_steps(steps: list[CausalStep], incoming: list[AbilityEvent], lang: str = "zh") -> None:
    """Detect CC chains and enemy ability casts from incoming abilities."""
    if not incoming:
        return
    en = lang == "en"
    incoming.sort(key=lambda a: a.time)
    chains: list[list[AbilityEvent]] = [[incoming[0]]]
    for j in range(1, len(incoming)):
        if incoming[j].time - incoming[j - 1].time <= 5:
            chains[-1].append(incoming[j])
        else:
            chains.append([incoming[j]])
    for chain in chains:
        first = chain[0]
        pos = _UNKNOWN
        if first.x is not None and first.y is not None and has_position((first.x, first.y)):
            pos = (float(first.x), float(first.y))
        if len(chain) >= 2:
            names = " → ".join(_ability(a, lang) for a in chain[:4])
            if len(chain) > 4:
                names += f" (+{len(chain) - 4} more)" if en else f"等{len(chain)}个技能"
            desc = f"Hit by a chain of enemy abilities: {names}" if en else f"遭到连续技能打击：{names}"
            steps.append(CausalStep(
                time=first.time, step_type="cc_chain", description=desc,
                significance="consequence", position=pos,
                details={"abilities": [_ability(a, lang) for a in chain]}))
        else:
            ability = _ability(first, lang)
            desc = (f"{first.hero_name} engaged with {ability}" if en
                    else f"{_hero(first.hero_name, lang)}使用{ability}发起攻击")
            steps.append(CausalStep(
                time=first.time, step_type="enemy_approach", description=desc,
                significance="consequence", position=pos,
                details={"ability": ability, "attacker": first.hero_name}))


def _hero(name: str, lang: str) -> str:
    """A hero's name as text of language ``lang`` (the data keeps the English name)."""
    return data_manager.hero_display_name(name, lang) or name


def _ability(ae: AbilityEvent, lang: str) -> str:
    """Display name of a cast (from its key; the stored name can be stale)."""
    if ae.ability_key:
        return data_manager.ability_display_name(ae.ability_key, lang)
    return ae.ability_name


def _get_incoming_abilities(
    match: MatchData, player: PlayerData, start: int, end: int,
) -> list[AbilityEvent]:
    """Enemy abilities targeting this player within [start, end]."""
    hero = player.hero_name
    seen: set[tuple[int, str, str]] = set()
    results: list[AbilityEvent] = []
    sources = [match.ability_events] + [p.ability_events for p in match.players if p.team != player.team]
    for src in sources:
        for ae in src:
            if ae.time < start or ae.time > end:
                continue
            if ae.target == hero and ae.player_slot != player.player_slot:
                key = (ae.time, ae.ability_name, ae.hero_name)
                if key not in seen:
                    seen.add(key)
                    results.append(ae)
    results.sort(key=lambda a: a.time)
    return results


# ---------------------------------------------------------------------------
# 5. Identify root cause
# ---------------------------------------------------------------------------

def _identify_root_cause(steps: list[CausalStep]) -> int:
    """Index of the earliest step with the highest controllability."""
    if not steps:
        return 0
    best_idx, best_ctrl = 0, 0
    for i, s in enumerate(steps):
        c = _CTRL_RANK.get(_CTRL.get(s.step_type, "low"), 1)
        if c > best_ctrl:
            best_ctrl = c
            best_idx = i
    return best_idx


def _classify_root_type(step: CausalStep) -> str:
    # "ignore_warning" is staying put while enemies are hidden nearby (that
    # is what raises the danger level), wherever on the map it happens.
    return {
        "vision_loss": "vision_loss", "enemy_missing": "ignore_missing",
        "hidden_enemy_near": "ignore_missing",
        "ignore_warning": "ignore_missing", "position_forward": "overextend",
        "position_unsafe": "bad_positioning", "no_retreat": "overextend",
        "ally_absent": "isolated", "farm_under_threat": "greedy_farm",
    }.get(step.step_type, "bad_positioning")


# ---------------------------------------------------------------------------
# 6. Format chain text
# ---------------------------------------------------------------------------

_SIG_LABEL = {"root_cause": "根因", "contributing": "诱因", "consequence": "后果", "final": "结果"}
_SIG_LABEL_EN = {"root_cause": "Root Cause", "contributing": "Contributing",
                 "consequence": "Consequence", "final": "Outcome"}


def _killer_name(death: DeathEvent, lang: str) -> str:
    """Who landed the kill, as written in a sentence ("a tower"); "" = unknown."""
    return killer_text(death, lang)


def _death_verb(death: DeathEvent, lang: str) -> str:
    """How the hero died, as a predicate ("was killed by Lina" / "died").

    A deny (``DeathEvent.denied``: the killer is an ally, or the hero itself)
    is not an enemy kill, and a death without a known killer is just a death.
    """
    killer = _killer_name(death, lang)
    denied = bool(getattr(death, "denied", False))
    own = denied and death.killer_slot == death.player_slot
    if lang == "en":
        if own:
            return "died to their own ability or item"
        if denied:
            return f"was denied by ally {killer}" if killer else "was denied by an ally"
        return f"was killed by {killer}" if killer else "died"
    if own:
        return "死于自己的技能或物品"
    if denied:
        return f"被队友{killer}反补" if killer else "被队友反补"
    return f"被{killer}击杀" if killer else "阵亡"


def _death_phrase(death: DeathEvent, lang: str) -> str:
    """The same as a step title ("Killed by Lina" / "Died")."""
    verb = _death_verb(death, lang)
    if lang == "en":
        verb = verb.removeprefix("was ")
        return verb[:1].upper() + verb[1:]
    return verb


def _format_chain_text(chain: CausalChain, lang: str = "zh") -> str:
    """Format the chain as markdown for the LLM prompt."""
    death = chain.death_event
    # A deny has no enemy killer: the death step says what happened.
    killer = "" if getattr(death, "denied", False) else killer_text(death, lang, article=False)
    if lang == "en":
        lines = ["## Death Causal Chain",
                 f"Death time: {_fmt(death.time)} | Hero: {death.hero_name}"]
        if killer:
            lines.append(f"Killer: {killer}")
    else:
        lines = ["## 死亡因果链",
                 f"死亡时间: {_fmt(death.time)} | 英雄: {_hero(death.hero_name, lang)}"]
        if killer:
            lines.append(f"击杀者: {killer}")
    lines.append("")
    sig_label = _SIG_LABEL_EN if lang == "en" else _SIG_LABEL
    for i, s in enumerate(chain.steps):
        lb = sig_label.get(s.significance, "")
        lb_s = f" [{lb}]" if lb else ""
        lines.append(f"{i + 1}. [{_fmt(s.time)}]{lb_s} {s.description}")
    lines.append("")
    if chain.root_cause_summary:
        label = "Root Cause" if lang == "en" else "根本原因"
        lines.append(f"**{label}**: {chain.root_cause_summary}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 7. Build causal chain (public API)
# ---------------------------------------------------------------------------

def _build_pregame_chain(match: MatchData, death: DeathEvent, lang: str = "zh") -> CausalChain:
    """Causal chain for a death before 0:30 (a level-1 fight; what it was about is not in the data)."""
    en = lang == "en"
    verb = _death_verb(death, lang)
    area = describe_area(death.x, death.y, lang=lang) if has_position((death.x, death.y)) else ""
    if en:
        where = f" at {area}" if area else ""
        step1_desc = f"{death.hero_name} took a fight{where} at level 1, before the creep waves met."
        step2_desc = f"{death.hero_name} {verb} in that fight."
        root_summary = "Took a fight at level 1, before the first creep wave arrived"
    else:
        where = f"在{area}" if area else ""
        hero = _hero(death.hero_name, lang)
        step1_desc = f"{hero}在兵线接触前（1级）{where}参与了战斗。"
        step2_desc = f"{hero}在这场战斗中{verb}。"
        root_summary = "在第一波兵线到达前（1级）参与了战斗"
    steps = [
        CausalStep(time=death.time - 5, step_type="rune_contest",
                   description=step1_desc, significance="root_cause"),
        CausalStep(time=death.time, step_type="death",
                   description=step2_desc, significance="final"),
    ]
    chain = CausalChain(
        death_event=death,
        steps=steps,
        root_cause_index=0,
        root_cause_type="rune_contest_unfavorable",
        root_cause_summary=root_summary,
        window_start=death.time - 10,
        window_end=death.time,
    )
    chain.formatted_text = _format_chain_text(chain, lang=lang)
    return chain


# How far back the walk into the enemy fountain is followed.
_FOUNTAIN_LOOKBACK = 30


def _fountain_entry_time(match: MatchData, player_slot: int, death: DeathEvent) -> int:
    """Second the hero entered the fountain area it died in (the death second when the data cannot tell)."""
    entered = death.time
    for back in range(1, _FOUNTAIN_LOOKBACK + 1):
        pos = position_at(match, player_slot, death.time - back)
        if pos is None or area_code(pos[0], pos[1])[0] != "fountain":
            break
        entered = death.time - back
    return entered


def _build_fountain_chain(match: MatchData, death: DeathEvent, player_slot: int, lang: str = "zh") -> CausalChain:
    """Causal chain for a death to the enemy fountain.

    Whatever fight went on around it, the hero died because it walked into
    the fountain's attack range: that walk is the root cause, and nothing
    about enemies in the fog or the head count is added to it.
    """
    en = lang == "en"
    fountain = killer_text(death, lang)
    entered = _fountain_entry_time(match, player_slot, death)
    steps = [
        CausalStep(time=entered, step_type="position_forward", significance="root_cause",
                   description=(f"Went into the attack range of {fountain}" if en
                                else f"走进了{fountain}的攻击范围"),
                   position=(death.x, death.y), danger_level="critical"),
        CausalStep(time=death.time, step_type="death", significance="final",
                   description=_death_phrase(death, lang), position=(death.x, death.y), danger_level="critical"),
    ]
    chain = CausalChain(
        death_event=death, steps=steps, root_cause_index=0, root_cause_type="overextend",
        root_cause_summary=("Went into the attack range of the enemy fountain" if en
                            else "走进了敌方泉水的攻击范围"),
        window_start=entered, window_end=death.time)
    chain.formatted_text = _format_chain_text(chain, lang=lang)
    return chain


def build_causal_chain(
    match: MatchData, death: DeathEvent, player_slot: int,
    lang: str = "zh",
) -> CausalChain:
    """Build a causal chain tracing back from a death to its root cause."""
    player = _player(match, player_slot)
    if player is None:
        if lang == "en":
            return CausalChain(
                death_event=death, root_cause_summary="Unable to retrieve player data",
                formatted_text="## Death Causal Chain\nUnable to retrieve player data, skipping analysis.\n")
        return CausalChain(
            death_event=death, root_cause_summary="无法获取玩家数据",
            formatted_text="## 死亡因果链\n无法获取玩家数据，跳过分析。\n")

    if death.time < _PREGAME_DEATH:
        return _build_pregame_chain(match, death, lang)
    if killed_by_fountain(death):
        return _build_fountain_chain(match, death, player_slot, lang)

    death_time = death.time
    tf = _find_fight(match, death)
    # The player's last moment alive: the same moment the context describes.
    state_time = last_alive_time(match, player_slot, death_time)

    w_start = _find_analysis_window(
        match, player_slot, death_time, walk_from=tf.start_time if tf is not None else None,
    )
    if death_time < 60:
        w_start = death_time - 30
    w_start = min(w_start, state_time - _STEP)
    # A buyback revives within the 5 s grid: nothing at or before the
    # previous death belongs to this one.
    previous = max((int(k.time) for k in match.kills_log
                    if k.victim_slot == player_slot and k.time < death_time - 1), default=None)
    if previous is not None and w_start <= previous < state_time:
        w_start = min(previous + 1, state_time - 1)

    si = 3 if (state_time - w_start) < 30 else _STEP
    states = _sample_states(match, player, player_slot, w_start, state_time, step=si, lang=lang)
    raw = _detect_transitions(states, match, player, player_slot, lang=lang)

    dpos = (death.x, death.y)
    if not has_position(dpos) and states:
        dpos = (states[-1]["x"], states[-1]["y"])
    ddesc = _death_phrase(death, lang)
    if has_position(dpos):
        area = describe_area(dpos[0], dpos[1], lang=lang)
        ddesc += f" at {area}" if lang == "en" else f"，位于{area}"
    raw.append(CausalStep(
        time=death_time, step_type="death", description=ddesc,
        significance="final", position=dpos, danger_level="critical"))

    grouped = False
    here = (states[-1]["x"], states[-1]["y"]) if states else _UNKNOWN
    if not has_position(here):
        here = dpos
    if tf is not None:
        raw, grouped = _enrich_teamfight(
            raw, match, player, player_slot, death, tf, state_time, here, lang=lang)
    else:
        # Allies can die around the player without a fight record.
        basis, allies, enemies = head_count(match, player_slot, state_time, here)
        eng = engagement(match, player_slot, death_time) if basis == "nearby" else None
        if eng is not None and eng.fallen_allies and not (eng.opened_outnumbered and enemies > allies + 1):
            raw.append(_fallen_step(eng, allies + 1, enemies, state_time, here, lang))

    # Causal order: nothing is dated after the death, which always comes last.
    for s in raw:
        if s.step_type != "death" and s.time > death_time:
            s.time = death_time
    raw.sort(key=lambda s: (s.time, s.step_type == "death"))
    seen: set[tuple[int, str]] = set()
    unique: list[CausalStep] = []
    for s in raw:
        k = (s.time, s.step_type)
        if k not in seen:
            seen.add(k)
            unique.append(s)
    if len(unique) > _MAX_STEPS:
        unique = _trim_steps(unique, _MAX_STEPS)

    # Only something the player did (or failed to react to) can be the root
    # cause; without such a step the chain has none. A step in the last
    # seconds before the death left no time to react, and what a player did
    # before a fight he then took with his team around him did not kill him.
    candidates = [
        s for s in unique
        if s.step_type != "death" and s.step_type not in _CONSEQUENCES
        and s.details.get("root_eligible", True)
        and (s.details.get("teamfight") or (
            not grouped and death_time - s.time >= _MIN_REACTION_S))
    ]
    rc_step: CausalStep | None = None
    rc_idx = 0
    # A teamfight death is explained by the fight itself when we know how it
    # went (numbers / who was focused); otherwise by the earlier steps.
    tf_steps = [s for s in candidates if s.details.get("teamfight")]
    if tf_steps:
        rc_step = next((s for s in tf_steps if s.step_type == "no_retreat"), tf_steps[0])
    if rc_step is None and candidates:
        rc_step = candidates[_identify_root_cause(candidates)]
    if rc_step is not None:
        rc_idx = unique.index(rc_step)

    for i, s in enumerate(unique):
        if s.step_type == "death":
            s.significance = "final"
        elif rc_step is not None and i == rc_idx:
            s.significance = "root_cause"
        elif s.step_type in _CONSEQUENCES:
            s.significance = "consequence"
        else:
            s.significance = "contributing"

    if rc_step is not None and rc_step.details.get("teamfight"):
        rc_type = "teamfight_outnumbered" if rc_step.step_type == "no_retreat" else "teamfight_positioning"
    elif rc_step is not None:
        rc_type = _classify_root_type(rc_step)
    else:
        rc_type = ""
    rc_summary = _root_summary(rc_step, unique, death, rc_type, lang=lang)

    chain = CausalChain(
        death_event=death, steps=unique, root_cause_index=rc_idx,
        root_cause_type=rc_type, root_cause_summary=rc_summary,
        window_start=w_start, window_end=death_time)
    chain.formatted_text = _format_chain_text(chain, lang=lang)
    return chain


def _find_fight(match: MatchData, death: DeathEvent) -> TeamfightEvent | None:
    """The teamfight a death belongs to.

    A death from the kill log carries its fight; without one it is a
    pick-off elsewhere during a fight window, not part of that fight.
    """
    if 0 <= death.teamfight_idx < len(match.teamfights):
        return match.teamfights[death.teamfight_idx]
    if death.source == "kills_log":
        return None
    for c in match.teamfights:
        if c.start_time - 5 <= death.time <= c.end_time + 10:
            return c
    return None


# ---------------------------------------------------------------------------
# Teamfight enrichment
# ---------------------------------------------------------------------------

def _first_allied_death(match: MatchData, tf: TeamfightEvent, death: DeathEvent, is_radiant: bool) -> bool:
    """Whether ``death`` is the first death of the player's team in the fight."""
    if match.kills_log:
        return not any(
            (k.victim_slot < 128) == is_radiant and k.victim_slot != death.player_slot
            and tf.start_time <= k.time < death.time
            for k in match.kills_log
        )
    offset = death.time - tf.start_time
    return not any(
        (tp.player_slot < 128) == is_radiant and tp.player_slot != death.player_slot
        and tp.deaths > 0 and tp.death_time_offset is not None and tp.death_time_offset < offset
        for tp in tf.players
    )


def _outnumbered_since(
    match: MatchData, player_slot: int, tf: TeamfightEvent, from_time: int,
) -> tuple[int, int, int]:
    """(second, allies incl. the player, enemies) at the start of the unbroken
    run, ending at ``from_time``, in which the player was outnumbered nearby."""
    at = estimate_position(match, player_slot, from_time)
    _b, a, e = head_count(match, player_slot, from_time, at, tf)
    since, count = from_time, (a + 1, e)
    earliest = max(tf.start_time, from_time - _OUTNUMBERED_LOOKBACK)
    t = from_time - _STEP
    while t >= earliest:
        at = estimate_position(match, player_slot, t)
        b, a, e = head_count(match, player_slot, t, at, tf)
        if b != "nearby" or e <= a + 1:
            break
        since, count = t, (a + 1, e)
        t -= _STEP
    return since, count[0], count[1]


def _fallen_step(
    eng: Engagement, ally_n: int, enemies: int, state_time: int, pos: tuple[float, float], lang: str,
) -> CausalStep:
    """Allies died around the player in a fight that is not judged outnumbered.

    The step states the deaths and the numbers the player was left with; it
    is never a root cause (being the last one standing is not a decision).
    The opening numbers are not stated: they only say that the fight did not
    clearly start outnumbered.
    """
    en = lang == "en"
    radius = int(nearby_radius(state_time))
    m = eng.fallen_allies
    if en:
        desc = f"From here on {m} {'ally' if m == 1 else 'allies'} died near the player before the player did"
        if enemies > 0:
            desc += f"; {ally_n}v{enemies} when the player died (heroes within {radius} units of the player)"
    else:
        desc = f"此后有{m}名队友先于玩家在其附近阵亡"
        if enemies > 0:
            desc += f"；玩家阵亡时为{ally_n}v{enemies}（按玩家{radius}码范围内的英雄计）"
    return CausalStep(
        time=min(eng.open_time, state_time), step_type="allies_fallen", description=desc,
        significance="contributing", position=pos, danger_level="danger",
        details={"fallen_allies": m, "allied": ally_n, "enemy": enemies, "root_eligible": False})


def _enrich_teamfight(
    steps: list[CausalStep], match: MatchData,
    player: PlayerData, player_slot: int, death: DeathEvent,
    tf: TeamfightEvent, state_time: int, pos: tuple[float, float],
    lang: str = "zh",
) -> tuple[list[CausalStep], bool]:
    """Add context for teamfight deaths (numbers, who was focused).

    Returns (steps, grouped); ``grouped`` says that the player fought with
    at least ``_GROUP_ALLIES`` allies around him.

    The head count is the one the context builder reports for the same
    moment (``state_time``, the player's last sample alive): heroes near the
    player, or the fight's participants when the source has no positions.
    When allies died near the player earlier in the fight, that count only
    shows who was left: the fight then only counts as outnumbered when it
    also was when its first hero died (:func:`vision.engagement`), and the
    "no_retreat" step states those numbers, dated at that second. Otherwise
    the allied deaths are an "allies_fallen" step, which is never a root
    cause.

    A step only qualifies as root cause (``details["teamfight"]``) when the
    evidence carries it: a count of nearby heroes, a participant count that
    differs by two or more, or being the team's first death and its most
    damaged hero. Being outnumbered inside the own base is never one:
    defending it is not a fight the player chose. Every step is dated at a
    second its numbers describe.
    """
    en = lang == "en"
    is_radiant = player.team == "radiant"

    basis, allies, enemies = head_count(match, player_slot, state_time, pos, tf)
    ally_n = allies + 1  # the player
    eng = engagement(match, player_slot, death.time, tf) if basis == "nearby" else None
    fallen = eng.fallen_allies if eng is not None else 0
    grouped = basis == "nearby" and max(allies, eng.open_allies if fallen else 0) >= _GROUP_ALLIES
    defending = in_own_base(player.team, pos[0], pos[1])
    if basis != "unknown" and enemies > ally_n and (not fallen or eng.opened_outnumbered):
        if basis == "nearby":
            if fallen:
                since, a, e = min(eng.open_time, state_time), eng.open_allies + 1, eng.open_enemies
            else:
                # Since when had the player been outnumbered there, and by how many?
                since, a, e = _outnumbered_since(match, player_slot, tf, state_time)
            radius = int(nearby_radius(since))
            if defending and en:
                desc = f"Defended the base outnumbered {a}v{e} (heroes within {radius} units of the player)"
            elif defending:
                desc = f"以{a}v{e}的人数劣势防守基地（按玩家{radius}码范围内的英雄计）"
            elif en:
                desc = f"Fought outnumbered {a}v{e} (heroes within {radius} units of the player)"
            else:
                desc = f"以{a}v{e}的人数劣势作战（按玩家{radius}码范围内的英雄计）"
            if fallen:
                desc += (f" when the first hero died; {fallen} {'ally' if fallen == 1 else 'allies'} "
                         f"died before the player" if en
                         else f"（首个英雄阵亡时），此后{fallen}名队友先于玩家阵亡")
            elif (a, e) != (ally_n, enemies):
                desc += f"; {ally_n}v{enemies} when the player died" if en else f"，玩家阵亡时为{ally_n}v{enemies}"
        else:
            a, e = ally_n, enemies
            since = min(state_time, max(tf.start_time, state_time - _OUTNUMBERED_LOOKBACK))
            if en:
                desc = (f"{enemies} enemy heroes took part in the fight against {ally_n} allied "
                        f"({ally_n}v{enemies}); where they stood is not in the data")
            else:
                desc = f"敌方{enemies}人参与了这场团战，己方{ally_n}人参战（{ally_n}v{enemies}），数据中没有站位信息"
        # One participant more somewhere on the map is not a cause.
        decisive = (basis == "nearby" or enemies >= ally_n + 2) and not defending
        steps.append(CausalStep(
            time=since, step_type="no_retreat", description=desc,
            significance="contributing", position=pos,
            danger_level="danger",
            details={"allied": a, "enemy": e, "basis": basis,
                     "teamfight": decisive, "root_eligible": decisive}))
    elif fallen:
        steps.append(_fallen_step(eng, ally_n, enemies, state_time, pos, lang))

    tfp = next((tp for tp in tf.players if tp.player_slot == player_slot), None)
    if tfp is not None and tfp.damage_taken > 0:
        ally_dmg = sorted(
            ((tp.player_slot, tp.damage_taken) for tp in tf.players
             if (tp.player_slot < 128) == is_radiant),
            key=lambda x: x[1], reverse=True,
        )
        if len(ally_dmg) > 1 and ally_dmg[0][0] == player_slot and ally_dmg[0][1] > ally_dmg[1][1]:
            first = _first_allied_death(match, tf, death, is_radiant)
            if first and en:
                desc = (f"First on the team to die in the fight, with the most damage taken "
                        f"({tfp.damage_taken}): the enemy's first target")
            elif first:
                desc = f"团战中己方第一个阵亡，且承受了全队最多的伤害（{tfp.damage_taken}），是敌方的首要集火目标"
            elif en:
                desc = f"Took the most damage on the team in the fight ({tfp.damage_taken})"
            else:
                desc = f"团战中承受己方最多伤害（{tfp.damage_taken}）"
            steps.append(CausalStep(
                time=min(tf.start_time + 2, state_time), step_type="position_forward", description=desc,
                significance="contributing", position=pos,
                danger_level="danger", details={"teamfight": first, "root_eligible": first}))
    return steps, grouped


# ---------------------------------------------------------------------------
# Step trimming
# ---------------------------------------------------------------------------

def _trim_steps(steps: list[CausalStep], n: int) -> list[CausalStep]:
    """Trim to n steps, keeping the death and high-controllability steps."""
    death = [s for s in steps if s.step_type == "death"]
    others = [s for s in steps if s.step_type != "death"]
    scored = sorted(others, key=lambda s: (-_CTRL_RANK.get(_CTRL.get(s.step_type, "low"), 1), s.time))
    kept = scored[:max(0, n - len(death))]
    kept.sort(key=lambda s: s.time)
    kept.extend(death)
    return kept


# ---------------------------------------------------------------------------
# Root cause summary
# ---------------------------------------------------------------------------

_SUMMARIES = {
    "en": {
        "vision_loss": "The ward covering the player's area was lost and the player neither replaced it nor moved back",
        "enemy_missing": "Several enemies slipped out of vision close by and the player neither checked the minimap nor retreated",
        "hidden_enemy_near": "Stayed within reach of several enemies that had been out of sight for a while; with enemies unseen that long, move back towards vision or allies",
        "ignore_warning": "Stayed where the situation had turned dangerous; move back as soon as nearby enemies drop out of vision",
        "position_forward": "Over-extended while nearby enemies were out of vision, too far from safety",
        "position_unsafe": "Was in an unwarded area while enemies were hidden nearby; only go deep with vision coverage",
        "no_retreat": "Missed the retreat window as the situation got worse",
        "ally_absent": "Isolated from teammates and caught out alone",
        "farm_under_threat": "Kept farming in an exposed spot while several enemies were hidden nearby; farm safer areas or back off when the map goes dark",
        "default": "Poor positioning led to the death",
        "tf_outnumbered": "Fought while outnumbered; wait for teammates or avoid the fight",
        "tf_outnumbered_record": "More enemy than allied heroes took part in the fight; make sure the team is together before fighting",
        "tf_positioning": "Was the enemy's first target in the fight; review whether the position or the timing of the entry exposed the player",
        "vis_factor": " (lack of vision was a key factor)",
        "iso_factor": " (no teammate support)",
    },
    "zh": {
        "vision_loss": "覆盖玩家所在区域的守卫消失后，既没有补眼也没有后撤",
        "enemy_missing": "多名敌方在附近脱离视野后未查看小地图，也未撤退到安全区域",
        "hidden_enemy_near": "多名久未露面的敌方已在可以到达玩家的范围内，玩家仍留在原地；敌人长时间不在视野内时应退向有视野或有队友的位置",
        "ignore_warning": "局势已经变得危险却仍留在原地，附近敌人脱离视野时应立即后撤",
        "position_forward": "附近敌人脱离视野时仍过度推进，远离安全区域",
        "position_unsafe": "身处没有视野的区域，而附近有敌人藏在迷雾中，应在有眼位保护时才深入",
        "no_retreat": "错过撤退时机，在形势恶化后仍未向安全区域移动",
        "ally_absent": "脱离队友孤立行动，被敌方抓单",
        "farm_under_threat": "附近有多名敌方藏在迷雾中时仍在危险区域贪刷，应换到安全区域刷钱或及时撤退",
        "default": "走位不当导致被击杀",
        "tf_outnumbered": "在人数劣势下作战，应等待队友集合或避免交战",
        "tf_outnumbered_record": "这场团战敌方参战人数多于己方，开团前应确认队友已经到位",
        "tf_positioning": "团战中成为敌方的首要集火目标，应复盘站位或进场时机是否让自己过早暴露",
        "vis_factor": "（视野缺失是重要诱因）",
        "iso_factor": "（缺乏队友支援）",
    },
}


def _root_summary(
    rc: CausalStep | None, all_steps: list[CausalStep],
    death: DeathEvent, root_type: str,
    lang: str = "zh",
) -> str:
    """One-line root-cause summary matching the chosen root-cause step."""
    text = _SUMMARIES.get(lang, _SUMMARIES["zh"])
    if rc is None:
        return ""
    if root_type == "teamfight_outnumbered":
        return text["tf_outnumbered_record" if rc.details.get("basis") == "participation"
                    else "tf_outnumbered"]
    if root_type == "teamfight_positioning":
        return text["tf_positioning"]
    s = text.get(rc.step_type, text["default"])
    has_vis = any(st.step_type == "vision_loss" for st in all_steps)
    has_iso = any(st.step_type == "ally_absent" for st in all_steps)
    if rc.step_type != "vision_loss" and has_vis:
        s += text["vis_factor"]
    elif rc.step_type != "ally_absent" and has_iso:
        s += text["iso_factor"]
    return s
