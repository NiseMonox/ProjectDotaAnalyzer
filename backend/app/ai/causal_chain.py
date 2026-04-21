"""Death causal chain analysis for AI coaching.

Given a death event, traces backwards through match data to reconstruct the
chain of events leading to the death and identify the root cause -- the earliest
moment where the player could have changed the outcome.

Supports Chinese (zh-CN) and English (en).
"""
from __future__ import annotations

import math
from typing import Any

from ..models import (
    AbilityEvent,
    MatchData,
    PlayerData,
    TeamfightEvent,
)
from .models import CausalChain, CausalStep, DeathEvent

# We import vision helpers lazily to avoid circular imports.
from .vision import (
    TOWER_DATA,
    _assess_danger,
    _build_vision_sources,
    _describe_area,
    _distance as _v_distance,
    _ensure_game_coords,
    _estimate_all_hero_positions,
    _estimate_player_position,
    _format_game_time,
    _get_alive_towers,
    _get_enemy_players,
    _get_team_observer_wards,
    _is_night,
    _point_in_vision,
    _track_enemy_visibility,
    _ward_pos_to_game,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_WINDOW = 120          # max seconds to look back
_STEP = 5                  # default sampling interval
_SAFE_TOWER_DIST = 2000    # game units
_ALLY_RADIUS = 2000
_WARD_RADIUS = 1600
_HP_SAFE = 0.70
_MAX_STEPS = 8

_CTRL: dict[str, str] = {
    "ignore_warning": "high", "position_forward": "high",
    "no_retreat": "high", "position_unsafe": "high",
    "enemy_missing": "medium", "ally_absent": "medium",
    "vision_loss": "low", "enemy_approach": "low",
    "cc_chain": "low", "damage_burst": "low", "death": "low",
}
_CTRL_RANK = {"high": 3, "medium": 2, "low": 1}

# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _gdist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def _fmt(t: int) -> str:
    if t < 0:
        return f"-{abs(t)//60}:{abs(t)%60:02d}"
    return f"{t//60}:{t%60:02d}"

def _player(match: MatchData, slot: int) -> PlayerData | None:
    for p in match.players:
        if p.player_slot == slot:
            return p
    return None

# ---------------------------------------------------------------------------
# 1. Reconstruct player position path
# ---------------------------------------------------------------------------

def _reconstruct_player_path(
    match: MatchData, player: PlayerData,
    start_time: int, end_time: int, step: int = _STEP,
) -> list[tuple[int, float, float]]:
    """Reconstruct position path using laning snapshots, teamfights, abilities."""
    seen: dict[int, tuple[float, float]] = {}
    for t in range(start_time, end_time + 1, step):
        pos = _estimate_player_position(player, t, match)
        if pos != (0.0, 0.0):
            seen[t] = pos
    # Add precise ability positions
    for ae in player.ability_events:
        if start_time <= ae.time <= end_time and ae.x and ae.y:
            p = _ensure_game_coords((ae.x, ae.y))
            if p != (0.0, 0.0):
                seen[ae.time] = p
    for ae in match.ability_events:
        if ae.player_slot == player.player_slot and start_time <= ae.time <= end_time:
            if ae.x and ae.y:
                p = _ensure_game_coords((ae.x, ae.y))
                if p != (0.0, 0.0):
                    seen[ae.time] = p
    return [(t, xy[0], xy[1]) for t, xy in sorted(seen.items())]

# ---------------------------------------------------------------------------
# 2. Find analysis window
# ---------------------------------------------------------------------------

def _find_analysis_window(
    match: MatchData, player_slot: int, death_time: int,
) -> int:
    """Walk backwards from death to find last 'safe' moment (near tower,
    enemies visible, or HP>70%).  Max window: _MAX_WINDOW seconds."""
    player = _player(match, player_slot)
    if player is None:
        return max(0, death_time - 30)
    team = player.team
    earliest = max(0, death_time - _MAX_WINDOW)

    for t in range(death_time - _STEP, earliest - 1, -_STEP):
        if t < 0:
            break
        pos = _estimate_player_position(player, t, match)
        # Tower proximity
        near_tower = False
        if pos != (0.0, 0.0):
            for _, tx, ty in _get_alive_towers(match, team, t):
                if _gdist(pos[0], pos[1], tx, ty) < _SAFE_TOWER_DIST:
                    near_tower = True
                    break
        # Enemy visibility
        evis = _track_enemy_visibility(match, team, player_slot, t)
        missing = sum(1 for e in evis if not e.is_visible)
        # HP check from laning snapshots
        hp_safe = False
        if player.laning_snapshots:
            snap = min(player.laning_snapshots, key=lambda s: abs(s.time - t))
            if abs(snap.time - t) <= 10 and snap.max_hp > 0:
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
) -> list[dict[str, Any]]:
    """Gather lightweight state snapshot at each timestep."""
    team = player.team
    states: list[dict[str, Any]] = []
    for t in range(start_time, end_time + 1, step):
        pos = _estimate_player_position(player, t, match)
        x, y = pos
        # HP
        hp_pct = -1.0
        if player.laning_snapshots:
            snap = min(player.laning_snapshots, key=lambda s: abs(s.time - t))
            if abs(snap.time - t) <= 15 and snap.max_hp > 0:
                hp_pct = snap.hp / snap.max_hp
        # Tower distance
        td = float("inf")
        if pos != (0.0, 0.0):
            for _, tx, ty in _get_alive_towers(match, team, t):
                td = min(td, _gdist(x, y, tx, ty))
        # Enemy visibility
        evis = _track_enemy_visibility(match, team, player_slot, t)
        miss_n = sum(1 for e in evis if not e.is_visible)
        miss_names = [e.hero_name for e in evis if not e.is_visible]
        # Danger
        if pos != (0.0, 0.0):
            dlevel, dreasons, _, wc = _assess_danger(
                match, team, player_slot, t, pos, evis)
        else:
            dlevel, dreasons, wc = "unknown", [], False
        # Nearby allies
        allies = 0
        if pos != (0.0, 0.0):
            hpos = _estimate_all_hero_positions(match, t)
            for p in match.players:
                if p.team == team and p.player_slot != player_slot:
                    ap = hpos.get(p.player_slot, (0.0, 0.0))
                    if ap != (0.0, 0.0) and _gdist(x, y, ap[0], ap[1]) <= _ALLY_RADIUS:
                        allies += 1
        # Ward coverage double-check
        if pos != (0.0, 0.0) and not wc:
            for w in _get_team_observer_wards(match, team, t):
                wx, wy = _ward_pos_to_game(w)
                if _gdist(x, y, wx, wy) <= _WARD_RADIUS:
                    wc = True
                    break
        states.append({
            "time": t, "x": x, "y": y, "hp_pct": hp_pct,
            "tower_dist": td, "missing_enemies": miss_n,
            "missing_names": miss_names, "danger_level": dlevel,
            "danger_reasons": dreasons if pos != (0.0, 0.0) else [],
            "nearby_allies": allies, "ward_coverage": wc,
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
    team = player.team
    steps: list[CausalStep] = []
    death_time = states[-1]["time"]
    window_start = states[0]["time"]
    incoming = _get_incoming_abilities(match, player, window_start, death_time)

    emitted_ward = emitted_unsafe = False
    dorder = {"safe": 0, "caution": 1, "danger": 2, "critical": 3, "unknown": 0}

    for i in range(1, len(states)):
        prev, curr = states[i - 1], states[i]
        t = curr["time"]
        pos = (curr["x"], curr["y"])

        # Ward loss
        if not emitted_ward and prev["ward_coverage"] and not curr["ward_coverage"]:
            if pos != (0.0, 0.0):
                area = _describe_area(pos[0], pos[1], lang=lang)
                if lang == "en":
                    desc = f"Vision lost near {area}, ward expired or destroyed"
                else:
                    desc = f"{area}附近视野丢失，守卫到期或被摧毁"
                steps.append(CausalStep(
                    time=t, step_type="vision_loss",
                    description=desc,
                    significance="contributing", position=pos,
                    danger_level=curr["danger_level"]))
                emitted_ward = True

        # Enemies disappearing
        new_miss = curr["missing_enemies"] - prev["missing_enemies"]
        if new_miss >= 2:
            names = [n for n in curr["missing_names"] if n not in prev.get("missing_names", [])]
            if lang == "en":
                ns = ", ".join(names[:3]) if names else f"{new_miss} enemies"
                desc = f"{ns} disappeared from the map"
            else:
                ns = "、".join(names[:3]) if names else f"{new_miss}名敌人"
                desc = f"{ns}从地图上消失"
            steps.append(CausalStep(
                time=t, step_type="enemy_missing",
                description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"],
                details={"missing_before": prev["missing_enemies"],
                         "missing_after": curr["missing_enemies"]}))
        elif new_miss == 1 and curr["missing_enemies"] >= 3:
            names = [n for n in curr["missing_names"] if n not in prev.get("missing_names", [])]
            if lang == "en":
                ns = names[0] if names else "an enemy"
                n = curr['missing_enemies']
                desc = f"{ns} disappeared, {n} enemies now missing"
            else:
                ns = names[0] if names else "一名敌人"
                desc = f"{ns}消失，现有{curr['missing_enemies']}名敌方去向不明"
            steps.append(CausalStep(
                time=t, step_type="enemy_missing",
                description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"]))

        # Danger escalation -> ignore_warning
        cv, pv = dorder.get(curr["danger_level"], 0), dorder.get(prev["danger_level"], 0)
        if cv >= 2 and cv > pv:
            retreated = (curr["tower_dist"] < prev["tower_dist"] - 200
                         if curr["tower_dist"] < float("inf") and prev["tower_dist"] < float("inf")
                         else False)
            if not retreated and pos != (0.0, 0.0):
                if lang == "en":
                    dzh = {"danger": "danger", "critical": "critical danger"}.get(curr["danger_level"], "danger")
                    rs = "; ".join(curr.get("danger_reasons", [])[:2])
                    desc = f"Danger level escalated to {dzh}"
                    if rs:
                        desc += f" ({rs})"
                    desc += " but player did not retreat"
                else:
                    dzh = {"danger": "危险", "critical": "极度危险"}.get(curr["danger_level"], "危险")
                    rs = "；".join(curr.get("danger_reasons", [])[:2])
                    desc = f"危险等级升至{dzh}"
                    if rs:
                        desc += f"（{rs}）"
                    desc += "，但玩家未撤退"
                steps.append(CausalStep(
                    time=t, step_type="ignore_warning", description=desc,
                    significance="contributing", position=pos,
                    danger_level=curr["danger_level"]))

        # Moving farther from tower
        if (curr["tower_dist"] < float("inf") and prev["tower_dist"] < float("inf")
                and curr["tower_dist"] > prev["tower_dist"] + 500
                and curr["tower_dist"] > _SAFE_TOWER_DIST
                and curr["missing_enemies"] >= 2 and pos != (0.0, 0.0)):
            area = _describe_area(pos[0], pos[1], lang=lang)
            if lang == "en":
                desc = f"Player advanced toward {area}, ~{int(curr['tower_dist'])} units from tower"
            else:
                desc = f"玩家向{area}推进，距塔约{int(curr['tower_dist'])}距离"
            steps.append(CausalStep(
                time=t, step_type="position_forward",
                description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"]))

        # Entering unwarded area
        if (not emitted_unsafe and not curr["ward_coverage"]
                and curr["tower_dist"] > _SAFE_TOWER_DIST
                and curr["missing_enemies"] >= 2 and pos != (0.0, 0.0)):
            area = _describe_area(pos[0], pos[1], lang=lang)
            n = curr['missing_enemies']
            if lang == "en":
                desc = f"Player entered unwarded {area}, {n} enemies missing"
            else:
                desc = f"玩家进入无视野的{area}，{n}名敌方消失"
            steps.append(CausalStep(
                time=t, step_type="position_unsafe",
                description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"]))
            emitted_unsafe = True

        # Allies leaving
        if prev["nearby_allies"] >= 1 and curr["nearby_allies"] == 0 and curr["missing_enemies"] >= 2:
            if lang == "en":
                desc = "Nearby allies left, player became an isolated target"
            else:
                desc = "附近友方英雄离开，玩家成为孤立目标"
            steps.append(CausalStep(
                time=t, step_type="ally_absent",
                description=desc,
                significance="contributing", position=pos,
                danger_level=curr["danger_level"]))

        # Rapid HP loss
        if prev["hp_pct"] >= 0 and curr["hp_pct"] >= 0 and prev["hp_pct"] - curr["hp_pct"] > 0.30:
            pb, pa = int(prev["hp_pct"] * 100), int(curr["hp_pct"] * 100)
            if lang == "en":
                desc = f"HP dropped from {pb}% to {pa}%"
            else:
                desc = f"生命值从{pb}%骤降至{pa}%"
            steps.append(CausalStep(
                time=t, step_type="damage_burst",
                description=desc,
                significance="consequence", position=pos,
                danger_level=curr["danger_level"]))

    # CC / ability events
    _add_ability_steps(steps, incoming, lang=lang)
    steps.sort(key=lambda s: s.time)
    return steps


def _add_ability_steps(steps: list[CausalStep], incoming: list[AbilityEvent], lang: str = "zh") -> None:
    """Detect CC chains and enemy ability casts from incoming abilities."""
    if not incoming:
        return
    incoming.sort(key=lambda a: a.time)
    # Group abilities within 5s windows
    chains: list[list[AbilityEvent]] = [[incoming[0]]]
    for j in range(1, len(incoming)):
        if incoming[j].time - incoming[j - 1].time <= 5:
            chains[-1].append(incoming[j])
        else:
            chains.append([incoming[j]])
    for chain in chains:
        pos = (0.0, 0.0)
        if chain[0].x and chain[0].y:
            pos = _ensure_game_coords((chain[0].x, chain[0].y))
        if len(chain) >= 2:
            names = " → ".join(a.ability_name for a in chain[:4])
            if len(chain) > 4:
                if lang == "en":
                    names += f" and {len(chain)} more abilities"
                else:
                    names += f"等{len(chain)}个技能"
            if lang == "en":
                desc = f"Hit by CC chain: {names}"
            else:
                desc = f"遭到连续控制链：{names}"
            steps.append(CausalStep(
                time=chain[0].time, step_type="cc_chain",
                description=desc,
                significance="consequence", position=pos,
                details={"abilities": [a.ability_name for a in chain]}))
        else:
            ae = chain[0]
            if lang == "en":
                desc = f"{ae.hero_name} initiated with {ae.ability_name}"
            else:
                desc = f"{ae.hero_name}使用{ae.ability_name}发起攻击"
            steps.append(CausalStep(
                time=ae.time, step_type="enemy_approach",
                description=desc,
                significance="consequence", position=pos,
                details={"ability": ae.ability_name, "attacker": ae.hero_name}))


def _get_incoming_abilities(
    match: MatchData, player: PlayerData,
    start: int, end: int,
) -> list[AbilityEvent]:
    """Collect enemy abilities targeting this player in a window."""
    hero = player.hero_name
    seen: set[tuple[int, str, str]] = set()
    results: list[AbilityEvent] = []
    for src in [match.ability_events] + [p.ability_events for p in match.players if p.team != player.team]:
        for ae in src:
            if ae.time < start or ae.time > end:
                continue
            if ae.target == hero and ae.player_slot != player.player_slot:
                k = (ae.time, ae.ability_name, ae.hero_name)
                if k not in seen:
                    seen.add(k)
                    results.append(ae)
    results.sort(key=lambda a: a.time)
    return results

# ---------------------------------------------------------------------------
# 5. Identify root cause
# ---------------------------------------------------------------------------

def _identify_root_cause(steps: list[CausalStep]) -> int:
    """Pick earliest highest-controllability step as root cause."""
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
        "ally_absent": "isolated",
    }.get(step.step_type, "bad_positioning")

# ---------------------------------------------------------------------------
# 6. Format chain text
# ---------------------------------------------------------------------------

_SIG_EMOJI = {"root_cause": "⚠️", "contributing": "🔶", "consequence": "🔴", "final": "💀"}
_SIG_LABEL = {"root_cause": "根因", "contributing": "诱因", "consequence": "后果", "final": "结果"}
_SIG_LABEL_EN = {"root_cause": "Root Cause", "contributing": "Contributing", "consequence": "Consequence", "final": "Outcome"}

def _format_chain_text(chain: CausalChain, lang: str = "zh") -> str:
    """Format as markdown for LLM injection."""
    if lang == "en":
        lines = ["## Death Causal Chain",
                 f"Death time: {_fmt(chain.death_event.time)} | Hero: {chain.death_event.hero_name}"]
        if chain.death_event.killer_hero:
            lines.append(f"Killer: {chain.death_event.killer_hero}")
    else:
        lines = ["## 死亡因果链",
                 f"死亡时间: {_fmt(chain.death_event.time)} | 英雄: {chain.death_event.hero_name}"]
        if chain.death_event.killer_hero:
            lines.append(f"击杀者: {chain.death_event.killer_hero}")
    lines.append("")
    sig_label = _SIG_LABEL_EN if lang == "en" else _SIG_LABEL
    for i, s in enumerate(chain.steps):
        em = _SIG_EMOJI.get(s.significance, "")
        lb = sig_label.get(s.significance, "")
        lb_s = f" [{lb}]" if lb else ""
        lines.append(f"{i+1}. [{_fmt(s.time)}] {em}{lb_s} {s.description}")
    lines.append("")
    if chain.root_cause_summary:
        if lang == "en":
            lines.append(f"**Root Cause**: {chain.root_cause_summary}")
        else:
            lines.append(f"**根本原因**: {chain.root_cause_summary}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# 7. Build causal chain (public API)
# ---------------------------------------------------------------------------

def _build_pregame_chain(
    match: MatchData, death: DeathEvent, lang: str = "zh",
) -> CausalChain:
    """Build a causal chain for pregame deaths (rune contest)."""
    player = _player(match, death.player_slot)
    killer_info = death.killer_hero or ("Unknown" if lang == "en" else "未知")

    steps = []
    # Step 1: Rune contest decision
    if lang == "en":
        step1_desc = (
            f"[Root Cause] {death.hero_name} entered the bounty rune area before 0:30. "
            f"At level 1, all heroes can skill one basic ability but haven't chosen yet. "
            f"A bounty rune is worth ~40 gold."
        )
    else:
        step1_desc = (
            f"[根因] {death.hero_name}在0:30前进入了赏金符区域。"
            f"1级时所有英雄都可以选择一个基础技能但尚未加点。"
            f"赏金符价值约40金。"
        )
    steps.append(CausalStep(
        time=max(0, death.time - 5),
        step_type="rune_contest",
        description=step1_desc,
        significance="root_cause",
    ))

    # Step 2: Death
    if lang == "en":
        step2_desc = f"{death.hero_name} was killed by {killer_info} during the rune contest."
    else:
        step2_desc = f"{death.hero_name}在赏金符争夺中被{killer_info}击杀。"
    steps.append(CausalStep(
        time=death.time,
        step_type="death",
        description=step2_desc,
        significance="final",
    ))

    # Format text
    if lang == "en":
        root_summary = "Contested bounty rune in an unfavorable level 1 situation"
        text_lines = ["## Death Causal Chain (Pregame)", ""]
        for i, s in enumerate(steps, 1):
            text_lines.append(f"{i}. [{_fmt(s.time)}] {s.description}")
        text_lines.append(f"\nRoot Cause: {root_summary}")
    else:
        root_summary = "在1级不利情况下参与赏金符争夺"
        text_lines = ["## 死亡因果链（开局阶段）", ""]
        for i, s in enumerate(steps, 1):
            text_lines.append(f"{i}. [{_fmt(s.time)}] {s.description}")
        text_lines.append(f"\n根本原因：{root_summary}")

    return CausalChain(
        death_event=death,
        steps=steps,
        root_cause_type="rune_contest_unfavorable",
        root_cause_summary=root_summary,
        window_start=max(0, death.time - 10),
        window_end=death.time,
        formatted_text="\n".join(text_lines),
    )


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
        else:
            return CausalChain(
                death_event=death, root_cause_summary="无法获取玩家数据",
                formatted_text="## 死亡因果链\n无法获取玩家数据，跳过分析。\n")

    # Pregame deaths (< 30s) get special handling
    if death.time < 30:
        return _build_pregame_chain(match, death, lang)

    death_time = death.time
    team = player.team

    # Analysis window
    w_start = _find_analysis_window(match, player_slot, death_time)
    if death_time < 60:
        w_start = max(0, death_time - 30)

    # Teamfight check
    is_tf = death.is_teamfight
    if not is_tf:
        for tf in match.teamfights:
            if tf.start_time <= death_time <= tf.end_time + 10:
                is_tf = True
                break

    # Sample & detect
    si = 3 if (death_time - w_start) < 30 else _STEP
    states = _sample_states(match, player, player_slot, w_start, death_time, step=si)
    raw = _detect_transitions(states, match, player, player_slot, lang=lang)

    # Death step
    dpos = _ensure_game_coords((death.x, death.y)) if (death.x, death.y) != (0.0, 0.0) else (
        (states[-1]["x"], states[-1]["y"]) if states else (0.0, 0.0))
    if lang == "en":
        ddesc = f"Killed by {death.killer_hero}" if death.killer_hero else "Died"
        if dpos != (0.0, 0.0):
            ddesc += f", at {_describe_area(dpos[0], dpos[1], lang=lang)}"
    else:
        ddesc = f"被{death.killer_hero}击杀" if death.killer_hero else "死亡"
        if dpos != (0.0, 0.0):
            ddesc += f"，位于{_describe_area(dpos[0], dpos[1], lang=lang)}"
    raw.append(CausalStep(
        time=death_time, step_type="death", description=ddesc,
        significance="final", position=dpos, danger_level="critical"))

    # Teamfight enrichment
    if is_tf:
        raw = _enrich_teamfight(raw, match, player, player_slot, death, lang=lang)

    # Deduplicate & trim
    raw.sort(key=lambda s: s.time)
    seen: set[tuple[int, str]] = set()
    unique = []
    for s in raw:
        k = (s.time, s.step_type)
        if k not in seen:
            seen.add(k)
            unique.append(s)
    if len(unique) > _MAX_STEPS:
        unique = _trim_steps(unique, _MAX_STEPS)

    # Root cause
    candidates = [s for s in unique if s.step_type != "death"]
    if candidates:
        rc_i = _identify_root_cause(candidates)
        rc_step = candidates[rc_i]
        rc_idx = unique.index(rc_step)
    else:
        rc_idx = 0
        rc_step = unique[0] if unique else None

    # Mark significance
    for i, s in enumerate(unique):
        if s.step_type == "death":
            s.significance = "final"
        elif i == rc_idx:
            s.significance = "root_cause"
        elif s.step_type in ("damage_burst", "cc_chain", "enemy_approach"):
            s.significance = "consequence"
        else:
            s.significance = "contributing"

    rc_type = _classify_root_type(rc_step) if rc_step else ""
    rc_summary = _root_summary(rc_step, unique, death, is_tf, lang=lang) if rc_step else ""

    chain = CausalChain(
        death_event=death, steps=unique, root_cause_index=rc_idx,
        root_cause_type=rc_type, root_cause_summary=rc_summary,
        window_start=w_start, window_end=death_time)
    chain.formatted_text = _format_chain_text(chain, lang=lang)
    return chain

# ---------------------------------------------------------------------------
# Teamfight enrichment
# ---------------------------------------------------------------------------

def _enrich_teamfight(
    steps: list[CausalStep], match: MatchData,
    player: PlayerData, player_slot: int, death: DeathEvent,
    lang: str = "zh",
) -> list[CausalStep]:
    """Add context for teamfight deaths (numbers, positioning)."""
    team = player.team
    tf: TeamfightEvent | None = None
    for c in match.teamfights:
        if c.start_time <= death.time <= c.end_time + 10:
            tf = c
            break
    if tf is None:
        return steps

    # Count participants
    rad = sum(1 for tp in tf.players if tp.player_slot < 128 and (tp.damage_dealt > 0 or tp.damage_taken > 0))
    dire = sum(1 for tp in tf.players if tp.player_slot >= 128 and (tp.damage_dealt > 0 or tp.damage_taken > 0))
    ally_n = rad if team == "radiant" else dire
    enemy_n = dire if team == "radiant" else rad

    if enemy_n > ally_n:
        if lang == "en":
            desc = f"Engaged in {ally_n}v{enemy_n} teamfight while outnumbered"
        else:
            desc = f"在{ally_n}v{enemy_n}的人数劣势下参与团战"
        steps.append(CausalStep(
            time=tf.start_time, step_type="no_retreat",
            description=desc,
            significance="contributing", position=(death.x, death.y),
            danger_level="danger", details={"allied": ally_n, "enemy": enemy_n}))

    # Check if player took most damage
    tfp = next((tp for tp in tf.players if tp.player_slot == player_slot), None)
    if tfp:
        ally_dmg = [(tp.player_slot, tp.damage_taken) for tp in tf.players
                    if (tp.player_slot < 128) == (team == "radiant")]
        ally_dmg.sort(key=lambda x: x[1], reverse=True)
        if len(ally_dmg) > 1 and ally_dmg[0][0] == player_slot and tfp.damage_taken > 0:
            if lang == "en":
                desc = f"Took most damage in teamfight ({tfp.damage_taken}), positioned too far forward"
            else:
                desc = f"团战中承受最多伤害（{tfp.damage_taken}），站位靠前"
            steps.append(CausalStep(
                time=tf.start_time + 2, step_type="position_forward",
                description=desc,
                significance="contributing", position=(tfp.x, tfp.y),
                danger_level="danger"))
    return steps

# ---------------------------------------------------------------------------
# Step trimming
# ---------------------------------------------------------------------------

def _trim_steps(steps: list[CausalStep], n: int) -> list[CausalStep]:
    """Trim to n steps, keeping death and high-controllability steps."""
    death = [s for s in steps if s.step_type == "death"]
    others = [s for s in steps if s.step_type != "death"]
    scored = sorted(others, key=lambda s: (-_CTRL_RANK.get(_CTRL.get(s.step_type, "low"), 1), s.time))
    kept = scored[:n - len(death)]
    kept.sort(key=lambda s: s.time)
    kept.extend(death)
    return kept

# ---------------------------------------------------------------------------
# Root cause summary
# ---------------------------------------------------------------------------

def _root_summary(
    rc: CausalStep, all_steps: list[CausalStep],
    death: DeathEvent, is_tf: bool,
    lang: str = "zh",
) -> str:
    """Generate a one-line root-cause summary."""
    has_vis = any(s.step_type == "vision_loss" for s in all_steps)
    has_iso = any(s.step_type == "ally_absent" for s in all_steps)

    if is_tf:
        if any(s.step_type == "no_retreat" for s in all_steps):
            if lang == "en":
                return "Engaged in teamfight while outnumbered — should have waited for teammates or avoided the fight"
            return "在人数劣势下强行参与团战，应等待队友集合或避免交战"
        if lang == "en":
            return "Poor teamfight positioning led to being focused down — improve movement and ability timing"
        return "团战站位不当导致被集火击杀，应注意走位和技能释放时机"

    if lang == "en":
        summaries_en = {
            "vision_loss": "Lost vision in a key area without replacing wards or retreating",
            "enemy_missing": "Multiple enemies disappeared without checking minimap or retreating to safety",
            "ignore_warning": "Ignored danger signals and kept pushing — should retreat when danger escalates",
            "position_forward": "Over-extended while enemy positions were unknown, too far from safety",
            "position_unsafe": "Entered an unwarded dangerous area — only push deep with vision coverage",
            "no_retreat": "Missed the retreat window — failed to move toward safety as the situation worsened",
            "ally_absent": "Isolated from teammates and caught out alone",
        }
        s = summaries_en.get(rc.step_type, "Poor positioning led to death")
        if rc.step_type != "vision_loss" and has_vis:
            s += " (lack of vision was a key factor)"
        elif rc.step_type != "ally_absent" and has_iso:
            s += " (lacked teammate support)"
        return s

    summaries = {
        "vision_loss": "关键区域视野丢失后未补眼也未撤退，导致失去敌方动向信息",
        "enemy_missing": "多名敌方消失后未查看小地图或未撤退到安全区域",
        "ignore_warning": "无视地图危险信号继续推进，应在危险等级升高时立即后撤",
        "position_forward": "在敌方位置不明时过度推进，远离安全区域",
        "position_unsafe": "进入无视野覆盖的危险区域，应在有眼位保护时才深入",
        "no_retreat": "错过撤退时机，在形势恶化后仍未向安全区域移动",
        "ally_absent": "脱离队友孤立行动，被敌方抓单",
    }
    s = summaries.get(rc.step_type, "走位不当导致被击杀")
    if rc.step_type != "vision_loss" and has_vis:
        s += "（视野缺失是重要诱因）"
    elif rc.step_type != "ally_absent" and has_iso:
        s += "（缺乏队友支援）"
    return s
