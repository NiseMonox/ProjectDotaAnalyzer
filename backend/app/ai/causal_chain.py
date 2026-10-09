"""Death causal chain analysis for AI coaching.

Given a death event, traces backwards through match data to reconstruct the
chain of events leading to the death and identify the root cause -- the
earliest moment where the player could have changed the outcome.

Positions are world units; distance thresholds are real game units.
Supports Chinese ("zh") and English ("en").
"""
from __future__ import annotations

import bisect
from typing import Any

from ..models import AbilityEvent, MatchData, PlayerData, TeamfightEvent
from .models import CausalChain, CausalStep, DeathEvent
from .vision import (
    WARD_COVERAGE_RADIUS,
    _assess_danger,
    _distance,
    _format_game_time,
    _get_alive_towers,
    _get_team_observer_wards,
    _track_enemy_visibility,
    count_nearby_heroes,
    describe_area,
    estimate_position,
    has_position,
    in_own_base,
    nearby_radius,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_WINDOW = 120          # max seconds to look back
_STEP = 5                  # default sampling interval
_SAFE_TOWER_DIST = 2000    # world units: "close to an allied tower"
_HP_SAFE = 0.70
_MAX_STEPS = 8
_PREGAME_DEATH = 30        # deaths before 0:30 are rune-contest deaths

_CTRL: dict[str, str] = {
    "ignore_warning": "high", "position_forward": "high",
    "no_retreat": "high", "position_unsafe": "high",
    "farm_under_threat": "high",
    "enemy_missing": "medium", "ally_absent": "medium",
    "vision_loss": "low", "enemy_approach": "low",
    "cc_chain": "low", "damage_burst": "low", "death": "low",
}
_CTRL_RANK = {"high": 3, "medium": 2, "low": 1}

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
    """Last hits at second ``t`` (snapshots, else per-minute interpolation)."""
    snap = _snapshot_near(player, t, 10)
    if snap is not None:
        return snap.last_hits
    lh = player.lh_t
    if not lh:
        return None
    minute = t // 60
    if minute >= len(lh) - 1:
        return lh[-1]
    frac = (t % 60) / 60.0
    return int(lh[minute] + (lh[minute + 1] - lh[minute]) * frac)


def _tower_distance(match: MatchData, team: str, pos: tuple[float, float], t: int) -> float:
    if not has_position(pos):
        return float("inf")
    if in_own_base(team, pos[0], pos[1]):
        return 0.0
    dists = [_distance(pos[0], pos[1], tx, ty) for _n, tx, ty in _get_alive_towers(match, team, t)]
    return min(dists) if dists else float("inf")


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
    enemies missing / healthy, or all enemies visible).

    ``walk_from`` (e.g. the start of the teamfight the player died in) lets
    the search skip the fight itself, where everyone is visible anyway.
    """
    player = _player(match, player_slot)
    if player is None:
        return max(0, death_time - 30)
    team = player.team
    earliest = max(0, death_time - _MAX_WINDOW)
    first = min(death_time, walk_from if walk_from is not None else death_time) - _STEP

    for t in range(first, earliest - 1, -_STEP):
        if t < 0:
            break
        pos = estimate_position(match, player_slot, t)
        near_tower = _tower_distance(match, team, pos, t) < _SAFE_TOWER_DIST
        evis = _track_enemy_visibility(match, team, player_slot, t)
        missing = sum(1 for e in evis if not e.is_visible)
        hp_safe = False
        snap = _snapshot_near(player, t, 10)
        if snap is not None and snap.max_hp > 0:
            hp_safe = snap.hp / snap.max_hp > _HP_SAFE
        if near_tower and (missing <= 1 or hp_safe):
            return t
        if missing == 0:
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
    """Gather a lightweight state snapshot at each timestep."""
    team = player.team
    states: list[dict[str, Any]] = []
    times = list(range(start_time, end_time + 1, step))
    if not times or times[-1] != end_time:
        times.append(end_time)
    for t in times:
        pos = estimate_position(match, player_slot, t)
        known = has_position(pos)
        hp_pct = -1.0
        snap = _snapshot_near(player, t, 15)
        if snap is not None and snap.max_hp > 0:
            hp_pct = snap.hp / snap.max_hp
        td = _tower_distance(match, team, pos, t)
        evis = _track_enemy_visibility(match, team, player_slot, t)
        missing = [e.hero_name for e in evis if not e.is_visible]
        if known:
            dlevel, dreasons, _safe, wc = _assess_danger(
                match, team, player_slot, t, pos, evis, lang=lang)
        else:
            dlevel, dreasons, wc = "unknown", [], False
        allies = 0
        if known:
            allies = count_nearby_heroes(
                match, team, pos[0], pos[1], t, exclude_slot=player_slot,
                radius=nearby_radius(t),
            )
            if not wc:
                for w in _get_team_observer_wards(match, team, t):
                    if _distance(pos[0], pos[1], w.x, w.y) <= WARD_COVERAGE_RADIUS:
                        wc = True
                        break
        states.append({
            "time": t, "x": pos[0], "y": pos[1], "hp_pct": hp_pct,
            "tower_dist": td, "missing_enemies": len(missing),
            "missing_names": missing, "danger_level": dlevel,
            "danger_reasons": dreasons,
            "nearby_allies": allies, "ward_coverage": wc,
            "last_hits": _last_hits_at(player, t),
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
    dorder = {"safe": 0, "caution": 1, "danger": 2, "critical": 3, "unknown": 0}
    inf = float("inf")

    for i in range(1, len(states)):
        prev, curr = states[i - 1], states[i]
        t = curr["time"]
        pos = (curr["x"], curr["y"])
        known = has_position(pos)

        # Ward loss
        if not emitted_ward and prev["ward_coverage"] and not curr["ward_coverage"] and known:
            area = describe_area(pos[0], pos[1], lang=lang)
            desc = (f"Vision lost near {area}: the ward expired or was destroyed" if en
                    else f"{area}附近视野丢失，守卫到期或被摧毁")
            steps.append(CausalStep(
                time=t, step_type="vision_loss", description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"]))
            emitted_ward = True

        # Enemies disappearing
        new_miss = curr["missing_enemies"] - prev["missing_enemies"]
        names = [n for n in curr["missing_names"] if n not in prev.get("missing_names", [])]
        if new_miss >= 2:
            if en:
                ns = ", ".join(names[:3]) if names else f"{new_miss} enemies"
                desc = f"{ns} disappeared from the map"
            else:
                ns = "、".join(names[:3]) if names else f"{new_miss}名敌人"
                desc = f"{ns}从地图上消失"
            steps.append(CausalStep(
                time=t, step_type="enemy_missing", description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"],
                details={"missing_before": prev["missing_enemies"],
                         "missing_after": curr["missing_enemies"]}))
        elif new_miss == 1 and curr["missing_enemies"] >= 3:
            n = curr["missing_enemies"]
            if en:
                desc = f"{names[0] if names else 'An enemy'} disappeared; {n} enemies are now missing"
            else:
                desc = f"{names[0] if names else '一名敌人'}消失，现有{n}名敌方去向不明"
            steps.append(CausalStep(
                time=t, step_type="enemy_missing", description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"]))

        # Danger escalation without retreating -> ignore_warning
        cv, pv = dorder.get(curr["danger_level"], 0), dorder.get(prev["danger_level"], 0)
        if cv >= 2 and cv > pv and known:
            retreated = (
                curr["tower_dist"] < prev["tower_dist"] - 200
                if curr["tower_dist"] < inf and prev["tower_dist"] < inf else False
            )
            if not retreated:
                if en:
                    lvl = {"danger": "danger", "critical": "critical danger"}.get(curr["danger_level"], "danger")
                    rs = "; ".join(curr.get("danger_reasons", [])[:2])
                    desc = f"Danger level rose to {lvl}" + (f" ({rs})" if rs else "") + " but the player did not retreat"
                else:
                    lvl = {"danger": "危险", "critical": "极度危险"}.get(curr["danger_level"], "危险")
                    rs = "；".join(curr.get("danger_reasons", [])[:2])
                    desc = f"危险等级升至{lvl}" + (f"（{rs}）" if rs else "") + "，但玩家未撤退"
                steps.append(CausalStep(
                    time=t, step_type="ignore_warning", description=desc,
                    significance="contributing", position=pos,
                    danger_level=curr["danger_level"]))

        # Moving farther from towers while enemies are missing
        if (known and curr["tower_dist"] < inf and prev["tower_dist"] < inf
                and curr["tower_dist"] > prev["tower_dist"] + 500
                and curr["tower_dist"] > _SAFE_TOWER_DIST
                and curr["missing_enemies"] >= 2):
            area = describe_area(pos[0], pos[1], lang=lang)
            dist = int(curr["tower_dist"])
            desc = (f"Player moved toward {area}, ~{dist} units from the nearest allied tower" if en
                    else f"玩家向{area}推进，距最近友方塔约{dist}码")
            steps.append(CausalStep(
                time=t, step_type="position_forward", description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"]))

        # Entering an unwarded area far from towers
        if (not emitted_unsafe and known and not curr["ward_coverage"]
                and curr["tower_dist"] > _SAFE_TOWER_DIST
                and curr["missing_enemies"] >= 2):
            area = describe_area(pos[0], pos[1], lang=lang)
            n = curr["missing_enemies"]
            desc = (f"Player entered {area} without ward coverage while {n} enemies were missing" if en
                    else f"玩家进入无视野的{area}，{n}名敌方消失")
            steps.append(CausalStep(
                time=t, step_type="position_unsafe", description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"]))
            emitted_unsafe = True

        # Farming in an exposed spot while the map is dark -> greedy farm
        gained = None
        if curr.get("last_hits") is not None and prev.get("last_hits") is not None:
            gained = curr["last_hits"] - prev["last_hits"]
        if (not emitted_farm and known and gained is not None and gained > 0
                and curr["missing_enemies"] >= 3
                and curr["tower_dist"] > _SAFE_TOWER_DIST):
            area = describe_area(pos[0], pos[1], lang=lang)
            n = curr["missing_enemies"]
            desc = (f"Kept farming at {area} (+{gained} CS) while {n} enemies were missing" if en
                    else f"在{area}继续刷钱（+{gained}补刀），而{n}名敌方英雄去向不明")
            steps.append(CausalStep(
                time=t, step_type="farm_under_threat", description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"]))
            emitted_farm = True

        # Allies leaving
        if prev["nearby_allies"] >= 1 and curr["nearby_allies"] == 0 and curr["missing_enemies"] >= 2:
            desc = ("Nearby allies left; the player became an isolated target" if en
                    else "附近友方英雄离开，玩家成为孤立目标")
            steps.append(CausalStep(
                time=t, step_type="ally_absent", description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"]))

        # Rapid HP loss
        if prev["hp_pct"] >= 0 and curr["hp_pct"] >= 0 and prev["hp_pct"] - curr["hp_pct"] > 0.30:
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
            names = " → ".join(a.ability_name for a in chain[:4])
            if len(chain) > 4:
                names += f" (+{len(chain) - 4} more)" if en else f"等{len(chain)}个技能"
            desc = f"Hit by a chain of enemy abilities: {names}" if en else f"遭到连续技能打击：{names}"
            steps.append(CausalStep(
                time=first.time, step_type="cc_chain", description=desc,
                significance="consequence", position=pos,
                details={"abilities": [a.ability_name for a in chain]}))
        else:
            desc = (f"{first.hero_name} engaged with {first.ability_name}" if en
                    else f"{first.hero_name}使用{first.ability_name}发起攻击")
            steps.append(CausalStep(
                time=first.time, step_type="enemy_approach", description=desc,
                significance="consequence", position=pos,
                details={"ability": first.ability_name, "attacker": first.hero_name}))


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
    return {
        "vision_loss": "vision_loss", "enemy_missing": "ignore_missing",
        "ignore_warning": "overextend", "position_forward": "overextend",
        "position_unsafe": "bad_positioning", "no_retreat": "overextend",
        "ally_absent": "isolated", "farm_under_threat": "greedy_farm",
    }.get(step.step_type, "bad_positioning")


# ---------------------------------------------------------------------------
# 6. Format chain text
# ---------------------------------------------------------------------------

_SIG_LABEL = {"root_cause": "根因", "contributing": "诱因", "consequence": "后果", "final": "结果"}
_SIG_LABEL_EN = {"root_cause": "Root Cause", "contributing": "Contributing",
                 "consequence": "Consequence", "final": "Outcome"}


def _format_chain_text(chain: CausalChain, lang: str = "zh") -> str:
    """Format the chain as markdown for the LLM prompt."""
    death = chain.death_event
    if lang == "en":
        lines = ["## Death Causal Chain",
                 f"Death time: {_fmt(death.time)} | Hero: {death.hero_name}"]
        if death.killer_hero:
            lines.append(f"Killer: {death.killer_hero}")
    else:
        lines = ["## 死亡因果链",
                 f"死亡时间: {_fmt(death.time)} | 英雄: {death.hero_name}"]
        if death.killer_hero:
            lines.append(f"击杀者: {death.killer_hero}")
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
    """Causal chain for a pregame death (rune contest)."""
    en = lang == "en"
    killer_info = death.killer_hero or ("an unknown hero" if en else "未知英雄")
    if en:
        step1_desc = (
            f"{death.hero_name} entered the bounty rune area before 0:30. "
            f"At level 1 every hero can skill one basic ability. "
            f"A bounty rune is worth roughly 40 gold."
        )
        step2_desc = f"{death.hero_name} was killed by {killer_info} during the rune contest."
        root_summary = "Contested the bounty rune in an unfavorable level-1 situation"
    else:
        step1_desc = (
            f"{death.hero_name}在0:30前进入了赏金符区域。"
            f"1级时所有英雄都可以学习一个基础技能。赏金符价值约40金。"
        )
        step2_desc = f"{death.hero_name}在赏金符争夺中被{killer_info}击杀。"
        root_summary = "在1级不利情况下参与赏金符争夺"
    steps = [
        CausalStep(time=max(0, death.time - 5), step_type="rune_contest",
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
        window_start=max(0, death.time - 10),
        window_end=death.time,
    )
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

    death_time = death.time
    tf = _find_fight(match, death)
    is_tf = death.is_teamfight or tf is not None

    w_start = _find_analysis_window(
        match, player_slot, death_time, walk_from=tf.start_time if tf is not None else None,
    )
    if death_time < 60:
        w_start = max(0, death_time - 30)

    si = 3 if (death_time - w_start) < 30 else _STEP
    states = _sample_states(match, player, player_slot, w_start, death_time, step=si, lang=lang)
    raw = _detect_transitions(states, match, player, player_slot, lang=lang)

    dpos = (death.x, death.y)
    if not has_position(dpos) and states:
        dpos = (states[-1]["x"], states[-1]["y"])
    if lang == "en":
        ddesc = f"Killed by {death.killer_hero}" if death.killer_hero else "Died"
        if has_position(dpos):
            ddesc += f" at {describe_area(dpos[0], dpos[1], lang=lang)}"
    else:
        ddesc = f"被{death.killer_hero}击杀" if death.killer_hero else "死亡"
        if has_position(dpos):
            ddesc += f"，位于{describe_area(dpos[0], dpos[1], lang=lang)}"
    raw.append(CausalStep(
        time=death_time, step_type="death", description=ddesc,
        significance="final", position=dpos, danger_level="critical"))

    if tf is not None:
        raw = _enrich_teamfight(raw, match, player, player_slot, death, tf, lang=lang)

    raw.sort(key=lambda s: s.time)
    seen: set[tuple[int, str]] = set()
    unique: list[CausalStep] = []
    for s in raw:
        k = (s.time, s.step_type)
        if k not in seen:
            seen.add(k)
            unique.append(s)
    if len(unique) > _MAX_STEPS:
        unique = _trim_steps(unique, _MAX_STEPS)

    candidates = [s for s in unique if s.step_type != "death"]
    rc_step: CausalStep | None = None
    rc_idx = 0
    if is_tf:
        # A teamfight death is explained by the fight itself when we know
        # how it went (numbers / who tanked); otherwise by the earlier steps.
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
        elif s.step_type in ("damage_burst", "cc_chain", "enemy_approach"):
            s.significance = "consequence"
        else:
            s.significance = "contributing"

    if rc_step is not None and rc_step.details.get("teamfight"):
        rc_type = "teamfight_outnumbered" if rc_step.step_type == "no_retreat" else "teamfight_positioning"
    elif rc_step is not None:
        rc_type = _classify_root_type(rc_step)
    elif is_tf:
        rc_type = "teamfight_positioning"
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
    if 0 <= death.teamfight_idx < len(match.teamfights):
        return match.teamfights[death.teamfight_idx]
    for c in match.teamfights:
        if c.start_time - 5 <= death.time <= c.end_time + 10:
            return c
    return None


# ---------------------------------------------------------------------------
# Teamfight enrichment
# ---------------------------------------------------------------------------

def _enrich_teamfight(
    steps: list[CausalStep], match: MatchData,
    player: PlayerData, player_slot: int, death: DeathEvent,
    tf: TeamfightEvent, lang: str = "zh",
) -> list[CausalStep]:
    """Add context for teamfight deaths (numbers, positioning)."""
    team = player.team
    is_radiant = team == "radiant"

    def _involved(tp) -> bool:
        return bool(tp.damage_dealt or tp.damage_taken or tp.kills or tp.deaths)

    rad = sum(1 for tp in tf.players if tp.player_slot < 128 and _involved(tp))
    dire = sum(1 for tp in tf.players if tp.player_slot >= 128 and _involved(tp))
    ally_n = rad if is_radiant else dire
    enemy_n = dire if is_radiant else rad

    if enemy_n > ally_n:
        if lang == "en":
            desc = f"Took a {ally_n}v{enemy_n} teamfight while outnumbered"
        else:
            desc = f"在{ally_n}v{enemy_n}的人数劣势下参与团战"
        steps.append(CausalStep(
            time=tf.start_time, step_type="no_retreat", description=desc,
            significance="contributing", position=(death.x, death.y),
            danger_level="danger", details={"allied": ally_n, "enemy": enemy_n, "teamfight": True}))

    tfp = next((tp for tp in tf.players if tp.player_slot == player_slot), None)
    if tfp is not None and tfp.damage_taken > 0:
        ally_dmg = sorted(
            ((tp.player_slot, tp.damage_taken) for tp in tf.players
             if (tp.player_slot < 128) == is_radiant),
            key=lambda x: x[1], reverse=True,
        )
        if len(ally_dmg) > 1 and ally_dmg[0][0] == player_slot:
            if lang == "en":
                desc = f"Took the most damage on the team in the fight ({tfp.damage_taken}); positioned too far forward"
            else:
                desc = f"团战中承受己方最多伤害（{tfp.damage_taken}），站位靠前"
            steps.append(CausalStep(
                time=tf.start_time + 2, step_type="position_forward", description=desc,
                significance="contributing", position=(tfp.x, tfp.y),
                danger_level="danger", details={"teamfight": True}))
    return steps


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
        "vision_loss": "Lost vision in a key area without replacing wards or retreating",
        "enemy_missing": "Multiple enemies disappeared and the player neither checked the minimap nor retreated",
        "ignore_warning": "Ignored rising danger and kept pushing; retreat as soon as danger escalates",
        "position_forward": "Over-extended while enemy positions were unknown, too far from safety",
        "position_unsafe": "Entered an unwarded dangerous area; only go deep with vision coverage",
        "no_retreat": "Missed the retreat window as the situation got worse",
        "ally_absent": "Isolated from teammates and caught out alone",
        "farm_under_threat": "Kept farming in an exposed spot while several enemies were missing; farm safer areas or back off when the map goes dark",
        "default": "Poor positioning led to the death",
        "tf_outnumbered": "Took a teamfight while outnumbered; wait for teammates or avoid the fight",
        "tf_positioning": "Poor teamfight positioning got the player focused; improve spacing and ability timing",
        "vis_factor": " (lack of vision was a key factor)",
        "iso_factor": " (no teammate support)",
    },
    "zh": {
        "vision_loss": "关键区域视野丢失后未补眼也未撤退，失去了敌方动向信息",
        "enemy_missing": "多名敌方消失后未查看小地图或未撤退到安全区域",
        "ignore_warning": "无视地图危险信号继续推进，应在危险等级升高时立即后撤",
        "position_forward": "在敌方位置不明时过度推进，远离安全区域",
        "position_unsafe": "进入无视野覆盖的危险区域，应在有眼位保护时才深入",
        "no_retreat": "错过撤退时机，在形势恶化后仍未向安全区域移动",
        "ally_absent": "脱离队友孤立行动，被敌方抓单",
        "farm_under_threat": "多名敌方消失时仍在危险区域贪刷，应换到安全区域刷钱或及时撤退",
        "default": "走位不当导致被击杀",
        "tf_outnumbered": "在人数劣势下强行参与团战，应等待队友集合或避免交战",
        "tf_positioning": "团战站位不当导致被集火击杀，应注意走位和技能释放时机",
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
    if root_type == "teamfight_outnumbered":
        return text["tf_outnumbered"]
    if root_type == "teamfight_positioning":
        return text["tf_positioning"]
    if rc is None:
        return ""
    s = text.get(rc.step_type, text["default"])
    has_vis = any(st.step_type == "vision_loss" for st in all_steps)
    has_iso = any(st.step_type == "ally_absent" for st in all_steps)
    if rc.step_type != "vision_loss" and has_vis:
        s += text["vis_factor"]
    elif rc.step_type != "ally_absent" and has_iso:
        s += text["iso_factor"]
    return s
