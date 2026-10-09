"""Main orchestrator for the AI coaching pipeline.

Pipeline:
1. Derive positions, detect decision points (deaths + key moments)
2. Build death causal chains and a context for each decision point
3. Retrieve knowledge for each context
4. Evaluate decisions (agent / single-prompt LLM, else rules) and the
   per-phase reviews concurrently, under one deadline
5. Compile the coaching report

CPU-heavy steps (1-3, benchmark comparison) run in worker threads so the
event loop stays responsive. LLM calls are bounded by
``AI_COACH_MAX_CONCURRENCY`` (see :func:`evaluator.llm_chat`) and by the
overall ``AI_COACH_TIMEOUT_S`` deadline: work still pending when it expires
falls back to rules and the report is marked ``degraded``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Awaitable, Callable, Optional

from ..models import MatchData, PlayerData
from .evaluator import (
    evaluate_decision,
    evaluate_decision_fallback,
    is_api_available,
    llm_deadline,
)
from .models import (
    REPORT_SCHEMA_VERSION,
    CausalChain,
    CoachingReport,
    DecisionContext,
    DecisionPoint,
    DeathEvent,
    Evaluation,
    PhaseReview,
    is_time_unknown,
    normalize_grade,
)

try:
    from .agent import AgentUnsupportedError, is_agent_enabled, run_decision_agent
    _HAS_AGENT = True
except Exception:  # pragma: no cover - agent module optional
    _HAS_AGENT = False

    class AgentUnsupportedError(RuntimeError):  # type: ignore[no-redef]
        pass

ProgressCallback = Callable[[dict], Awaitable[None]]

logger = logging.getLogger(__name__)

__all__ = ["REPORT_SCHEMA_VERSION", "generate_coaching_report"]

# Max decision points sent to the LLM per report (deaths first).
MAX_LLM_EVALS = 8
DEFAULT_TIMEOUT_S = 180.0


def _coach_timeout() -> float:
    """Overall report deadline in seconds (env AI_COACH_TIMEOUT_S)."""
    try:
        return max(1.0, float(os.environ.get("AI_COACH_TIMEOUT_S", "").strip() or DEFAULT_TIMEOUT_S))
    except ValueError:
        return DEFAULT_TIMEOUT_S


def _llm_budget(timeout: float, elapsed: float) -> float:
    """Seconds the LLM stage may use, leaving time to compile the report."""
    reserve = min(15.0, max(1.0, timeout * 0.1))
    return max(0.0, timeout - reserve - elapsed)


# ---------------------------------------------------------------------------
# Score utilities
# ---------------------------------------------------------------------------

_SCORE_TO_NUM: dict[str, float] = {"S": 5.0, "A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}
_NUM_TO_SCORE: list[tuple[float, str]] = [
    (4.5, "S"),
    (3.5, "A"),
    (2.5, "B"),
    (1.5, "C"),
    (0.0, "D"),
]


def _letter_grade(value: float) -> str:
    """Convert a numeric score to a letter grade."""
    for threshold, grade in _NUM_TO_SCORE:
        if value >= threshold:
            return grade
    return "D"


# ---------------------------------------------------------------------------
# Helper: find the player in match data
# ---------------------------------------------------------------------------

def _find_player(match: MatchData, player_slot: int) -> PlayerData | None:
    for p in match.players:
        if p.player_slot == player_slot:
            return p
    return None


# ---------------------------------------------------------------------------
# Internationalization helpers
# ---------------------------------------------------------------------------

# Keyed by RELATIVE lane role (positions.lane_role_of): 1 safe, 2 mid, 3 off,
# 4 jungle/roaming. OpenDota's "jungle" is essentially extinct in modern Dota
# (~95% are roaming supports), so it is labelled as roaming.
_LANE_NAMES: dict[str, dict[int, str]] = {
    "zh": {1: "安全路", 2: "中路", 3: "劣势路", 4: "游走"},
    "en": {1: "Safe Lane", 2: "Mid Lane", 3: "Off Lane", 4: "Roaming"},
}

_LANE_RESULT_LABELS: dict[str, dict[str, str]] = {
    "zh": {"won": "优势", "even": "均势", "lost": "劣势"},
    "en": {"won": "Won", "even": "Even", "lost": "Lost"},
}

_CS_EFFICIENCY_LABELS: dict[str, dict[str, str]] = {
    "zh": {"excellent": "优秀", "good": "良好", "average": "一般", "poor": "需要提高"},
    "en": {"excellent": "Excellent", "good": "Good", "average": "Average", "poor": "Needs Improvement"},
}

_UNKNOWN_OPPONENT: dict[str, str] = {"zh": "未知对手", "en": "Unknown opponent"}
_UNKNOWN_LANE: dict[str, str] = {"zh": "未知路", "en": "Unknown Lane"}

# Internal category keys (stable) -> display labels.
_CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "zh": {
        "map_awareness": "地图意识",
        "laning": "对线能力",
        "teamfight": "团战参与",
        "resources": "资源利用",
        "vision": "视野控制",
    },
    "en": {
        "map_awareness": "Map Awareness",
        "laning": "Laning",
        "teamfight": "Teamfight Participation",
        "resources": "Resource Efficiency",
        "vision": "Vision Control",
    },
}

_IMPROVEMENT_SUGGESTIONS: dict[str, dict[str, str]] = {
    "zh": {
        "map_awareness": "多看小地图，养成每5秒看一次的习惯。注意敌方英雄消失的信号。",
        "laning": "练习补刀，特别是塔下补刀。注意换血时机和消耗品使用。",
        "teamfight": "注意开团时机和站位。优先集火关键目标，注意保护己方核心。",
        "resources": "提高打野效率，减少游走等待时间。确保每分钟都有金币收入。",
        "vision": "养成买眼插眼的习惯，特别是团战前和烟雾时。",
    },
    "en": {
        "map_awareness": "Check the minimap more frequently — aim for every 5 seconds. Watch for missing enemy heroes.",
        "laning": "Practice last-hitting, especially under tower. Mind your trade timing and consumable usage.",
        "teamfight": "Pay attention to engagement timing and positioning. Focus fire key targets and protect your cores.",
        "resources": "Improve jungle efficiency and reduce idle time. Ensure consistent gold income every minute.",
        "vision": "Make warding a habit, especially before teamfights and smoke ganks.",
    },
}

_DEFAULT_IMPROVEMENT: dict[str, str] = {
    "zh": "持续练习和复盘，提高整体水平。",
    "en": "Keep practicing and reviewing replays to improve overall play.",
}

_GRADE_COMMENTS: dict[str, dict[str, str]] = {
    "zh": {
        "S": "整体表现极为出色，几乎没有明显失误。",
        "A": "整体表现优秀，只有少量可以改进的地方。",
        "B": "整体表现中规中矩，有一些值得注意的决策可以改进。",
        "C": "整体表现有较多需要改进的地方，建议重点关注以下问题。",
        "D": "本场比赛犯了较多严重失误，需要重点复盘和改进。",
    },
    "en": {
        "S": "Outstanding performance with virtually no notable mistakes.",
        "A": "Excellent performance with only minor room for improvement.",
        "B": "Solid performance, but there are some decisions worth revisiting.",
        "C": "Several areas need improvement — focus on the issues highlighted below.",
        "D": "Multiple serious mistakes were made — thorough review and practice is recommended.",
    },
}

_STRENGTH_DEFAULT: dict[str, str] = {
    "zh": "整体表现稳定",
    "en": "Consistent overall performance",
}

_POSITION_FALLBACK_LABEL: dict[str, str] = {"zh": "未知位置", "en": "Unknown position"}


# ---------------------------------------------------------------------------
# Laning review
# ---------------------------------------------------------------------------

_MIRROR_POSITION = {1: 3, 3: 1, 2: 2, 4: 5, 5: 4}
# Lane opponents found from snapshot positions must stand this close (world units).
_LANE_OPPONENT_RADIUS = 3000.0


def _lane_cs_10(p: PlayerData) -> int:
    """Lane-creep last hits in the first 10 minutes (best available source)."""
    cs = sum(1 for ck in p.laning_creep_kills if ck.creep_type in ("melee", "ranged", "siege"))
    if cs:
        return cs
    if p.lh_t and len(p.lh_t) > 10:
        return p.lh_t[10]
    if p.laning_snapshots:
        return p.laning_snapshots[-1].last_hits
    if p.lh_t:
        return p.lh_t[-1]
    return 0


def _lane_opponents(match: MatchData, player: PlayerData) -> list[PlayerData]:
    """Enemies that laned against ``player``.

    ``lane`` is absolute (1=bottom, 2=mid, 3=top), so lane opponents share
    the same lane number. Roamers / unknown lanes fall back to early laning
    positions.
    """
    if player.lane in (1, 2, 3):
        same = [p for p in match.players if p.team != player.team and p.lane == player.lane]
        if same:
            return same[:3]
    snaps = player.laning_snapshots[5:15]  # ~0:25-1:15
    if not snaps:
        return []
    from .vision import _distance, has_position

    mx = sum(s.x for s in snaps) / len(snaps)
    my = sum(s.y for s in snaps) / len(snaps)
    if not has_position((mx, my)):
        return []
    found: list[tuple[float, PlayerData]] = []
    for p in match.players:
        if p.team == player.team:
            continue
        e_snaps = p.laning_snapshots[5:15]
        if not e_snaps:
            continue
        ex = sum(s.x for s in e_snaps) / len(e_snaps)
        ey = sum(s.y for s in e_snaps) / len(e_snaps)
        d = _distance(mx, my, ex, ey)
        if d <= _LANE_OPPONENT_RADIUS:
            found.append((d, p))
    found.sort(key=lambda item: item[0])
    return [p for _d, p in found[:3]]


def _generate_laning_review(
    match: MatchData, player: PlayerData, evaluations: list[Evaluation],
    lang: str = "zh", positions: dict | None = None,
) -> dict[str, Any]:
    """Laning phase review for the player.

    Returns a dict with cs_10min, denies_10min, lane_matchup, lane_result
    ("won"/"even"/"lost"), key_trades, cs_efficiency, and summary.
    """
    from .positions import lane_role_of

    positions = positions or {}
    lane_creep_cs = sum(
        1 for ck in player.laning_creep_kills if ck.creep_type in ("melee", "ranged", "siege")
    )
    cs_10min = _lane_cs_10(player)
    denies_10min = player.laning_snapshots[-1].denies if player.laning_snapshots else player.denies

    lane_map = _LANE_NAMES.get(lang, _LANE_NAMES["zh"])
    player_lane_name = lane_map.get(lane_role_of(player), _UNKNOWN_LANE.get(lang, _UNKNOWN_LANE["zh"]))

    opponents = _lane_opponents(match, player)
    unknown_opp = _UNKNOWN_OPPONENT.get(lang, _UNKNOWN_OPPONENT["zh"])
    separator = ", " if lang == "en" else "、"
    lane_matchup = (
        f"{player.hero_name} ({player_lane_name}) vs "
        + (separator.join(p.hero_name for p in opponents) if opponents else unknown_opp)
    )

    # Compare against the direct counterpart (pos 1 vs 3, 2 vs 2, 4 vs 5).
    lane_result = "even"
    opp_player: PlayerData | None = None
    my_pos = positions.get(player.player_slot)
    if opponents:
        if my_pos is not None:
            mirror = _MIRROR_POSITION.get(my_pos.number)
            opp_player = next(
                (p for p in opponents
                 if positions.get(p.player_slot) is not None and positions[p.player_slot].number == mirror),
                None,
            )
        if opp_player is None:
            opp_player = opponents[0]
    if opp_player is not None:
        opp_cs = _lane_cs_10(opp_player)
        my_snap = player.laning_snapshots[-1] if player.laning_snapshots else None
        opp_snap = opp_player.laning_snapshots[-1] if opp_player.laning_snapshots else None
        advantage = 0
        cs_diff = cs_10min - opp_cs
        if abs(cs_diff) > 5:
            advantage += 1 if cs_diff > 0 else -1
        if my_snap is not None and opp_snap is not None:
            nw_diff = my_snap.net_worth - opp_snap.net_worth
            if abs(nw_diff) > 300:
                advantage += 1 if nw_diff > 0 else -1
            lvl_diff = my_snap.level - opp_snap.level
            if abs(lvl_diff) >= 1:
                advantage += 1 if lvl_diff > 0 else -1
        if advantage >= 2:
            lane_result = "won"
        elif advantage <= -2:
            lane_result = "lost"

    key_trades: list[str] = []
    if player.laning_damage_events:
        total_dmg_dealt = sum(e.damage for e in player.laning_damage_events)
        key_trades.append(
            f"Dealt {total_dmg_dealt} hero damage during the laning phase" if lang == "en"
            else f"对线阶段造成 {total_dmg_dealt} 点英雄伤害"
        )
    score_label = "Score" if lang == "en" else "评分"
    laning_evals = [
        e for e in evaluations
        if 0 <= e.decision_point.time < 600 and not is_time_unknown(e.decision_point)
    ]
    for ev in sorted(laning_evals, key=lambda e: e.decision_point.time)[:3]:
        t = ev.decision_point.time
        key_trades.append(f"[{t // 60}:{t % 60:02d}] {ev.decision_point.brief} ({score_label}: {ev.score})")

    # CS efficiency: lane creeps killed / ~82 practically killable lane creeps
    # in the first 10 minutes.
    theoretical_lane_creeps = 82
    lane_cs_for_efficiency = lane_creep_cs if lane_creep_cs > 0 else cs_10min
    efficiency_pct = lane_cs_for_efficiency / theoretical_lane_creeps * 100
    eff_labels = _CS_EFFICIENCY_LABELS.get(lang, _CS_EFFICIENCY_LABELS["zh"])
    if efficiency_pct >= 80:
        eff_key = "excellent"
    elif efficiency_pct >= 60:
        eff_key = "good"
    elif efficiency_pct >= 40:
        eff_key = "average"
    else:
        eff_key = "poor"
    cs_efficiency = f"{eff_labels[eff_key]} ({efficiency_pct:.0f}%)"

    result_labels = _LANE_RESULT_LABELS.get(lang, _LANE_RESULT_LABELS["zh"])
    lane_result_label = result_labels.get(lane_result, result_labels["even"])

    if lang == "en":
        summary_parts = [
            f"{player.hero_name} had a {lane_result_label.lower()} laning phase ({player_lane_name}). ",
            f"{cs_10min} last hits at 10 min ({efficiency_pct:.0f}% efficiency), {denies_10min} denies. ",
        ]
        if lane_result == "lost":
            summary_parts.append("CS efficiency and trading need improvement.")
        elif lane_result == "won":
            summary_parts.append("Strong laning performance — keep it up.")
        else:
            summary_parts.append("Decent laning phase with some room for improvement.")
    else:
        summary_parts = [
            f"{player.hero_name}在{player_lane_name}的对线表现为{lane_result_label}。",
            f"10分钟补刀{cs_10min}个（效率{efficiency_pct:.0f}%），反补{denies_10min}个。",
        ]
        if lane_result == "lost":
            summary_parts.append("需要改进补刀效率和对线换血。")
        elif lane_result == "won":
            summary_parts.append("对线阶段表现优秀，继续保持。")
        else:
            summary_parts.append("对线表现中规中矩，有一定的提升空间。")

    return {
        "cs_10min": cs_10min,
        "denies_10min": denies_10min,
        "lane_matchup": lane_matchup,
        "lane_result": lane_result,
        "key_trades": key_trades,
        "cs_efficiency": cs_efficiency,
        "summary": "".join(summary_parts),
    }


# ---------------------------------------------------------------------------
# Improvement areas & strengths detection
# ---------------------------------------------------------------------------

_CATEGORY_MAP: dict[str, str] = {
    "death": "map_awareness",
    "overextend": "map_awareness",
    "bad_trade": "laning",
    "missed_cs": "laning",
    "fight_engage": "teamfight",
    "good_play": "teamfight",
}
_NO_MISTAKE = {"", "无", "None", "none", "N/A"}


def _detect_improvement_areas(evaluations: list[Evaluation], lang: str = "zh") -> list[dict[str, Any]]:
    """Recurring weaknesses: categories averaging below B."""
    category_scores: dict[str, list[str]] = {}
    category_details: dict[str, list[str]] = {}
    for ev in evaluations:
        cat = _CATEGORY_MAP.get(ev.decision_point.point_type, "resources")
        category_scores.setdefault(cat, []).append(ev.score)
        if ev.mistake and ev.mistake.strip() not in _NO_MISTAKE:
            category_details.setdefault(cat, []).append(ev.mistake.strip())

    cat_labels = _CATEGORY_LABELS.get(lang, _CATEGORY_LABELS["zh"])
    suggestions = _IMPROVEMENT_SUGGESTIONS.get(lang, _IMPROVEMENT_SUGGESTIONS["zh"])
    default_suggestion = _DEFAULT_IMPROVEMENT.get(lang, _DEFAULT_IMPROVEMENT["zh"])
    detail_sep = "; " if lang == "en" else "；"

    areas: list[dict[str, Any]] = []
    for cat, scores in category_scores.items():
        avg = sum(_SCORE_TO_NUM.get(s, 2.0) for s in scores) / len(scores)
        if avg >= 3.0:
            continue
        unique_details = list(dict.fromkeys(category_details.get(cat, [])))[:3]
        suggestion = suggestions.get(cat, default_suggestion)
        detail_text = detail_sep.join(d.rstrip(".。") for d in unique_details)
        if not detail_text:
            description = suggestion
        elif lang == "en":
            description = f"{detail_text}. {suggestion}"
        else:
            description = f"{detail_text}。{suggestion}"
        areas.append({
            "category": cat_labels.get(cat, cat),
            "category_code": cat,
            "avg_score": _letter_grade(avg),
            "occurrences": len(scores),
            "description": description,
        })
    areas.sort(key=lambda a: _SCORE_TO_NUM.get(a["avg_score"], 2.0))
    return areas


def _detect_strengths(evaluations: list[Evaluation], lang: str = "zh") -> list[dict[str, Any]]:
    """Recurring strengths: categories averaging A or better."""
    category_scores: dict[str, list[str]] = {}
    category_details: dict[str, list[str]] = {}
    for ev in evaluations:
        cat = _CATEGORY_MAP.get(ev.decision_point.point_type, "resources")
        category_scores.setdefault(cat, []).append(ev.score)
        if ev.score in ("S", "A"):
            desc = ev.decision_point.brief or ev.analysis[:50]
            category_details.setdefault(cat, []).append(desc)

    cat_labels = _CATEGORY_LABELS.get(lang, _CATEGORY_LABELS["zh"])
    default_highlight = _STRENGTH_DEFAULT.get(lang, _STRENGTH_DEFAULT["zh"])
    detail_sep = "; " if lang == "en" else "；"

    strengths: list[dict[str, Any]] = []
    for cat, scores in category_scores.items():
        avg = sum(_SCORE_TO_NUM.get(s, 2.0) for s in scores) / len(scores)
        if avg < 3.5:
            continue
        unique_details = list(dict.fromkeys(category_details.get(cat, [])))[:3]
        strengths.append({
            "category": cat_labels.get(cat, cat),
            "category_code": cat,
            "avg_score": _letter_grade(avg),
            "occurrences": len(scores),
            "description": detail_sep.join(unique_details) if unique_details else default_highlight,
        })
    strengths.sort(key=lambda s: _SCORE_TO_NUM.get(s["avg_score"], 2.0), reverse=True)
    return strengths


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def _generate_summary(
    player: PlayerData,
    overall_grade: str,
    evaluations: list[Evaluation],
    improvement_areas: list[dict],
    strengths: list[dict],
    match: MatchData,
    lang: str = "zh",
) -> str:
    """Text summary of the coaching report."""
    won = (player.team == "radiant") == bool(match.radiant_win)
    kda = f"{player.kills}/{player.deaths}/{player.assists}"
    duration_min = match.duration // 60
    grade_comments = _GRADE_COMMENTS.get(lang, _GRADE_COMMENTS["zh"])
    death_count = sum(1 for e in evaluations if e.decision_point.point_type == "death")

    if lang == "en":
        parts = [
            f"{player.hero_name} {'won' if won else 'lost'} this match with a KDA of {kda} "
            f"in {duration_min} minutes. ",
            f"Overall grade {overall_grade}: {grade_comments.get(overall_grade, '')} ",
        ]
        if strengths:
            parts.append(f"Key strength: {strengths[0]['category']}. ")
        if improvement_areas:
            parts.append(
                f"Top area to improve: {improvement_areas[0]['category']} "
                f"({improvement_areas[0]['occurrences']} occurrences). "
            )
        if death_count:
            parts.append(f"{death_count} death(s) analyzed — review the cause of each death carefully.")
    else:
        parts = [
            f"本场比赛{player.hero_name}{'获胜' if won else '失败'}，KDA为{kda}，比赛时长{duration_min}分钟。",
            f"综合评分{overall_grade}：{grade_comments.get(overall_grade, '')}",
        ]
        if strengths:
            parts.append(f"主要优势是{strengths[0]['category']}方面的表现。")
        if improvement_areas:
            parts.append(
                f"最需要改进的是{improvement_areas[0]['category']}，"
                f"相关问题出现了{improvement_areas[0]['occurrences']}次。"
            )
        if death_count:
            parts.append(f"本场共分析了{death_count}次死亡，请重点关注每次死亡的原因。")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Preparation steps (run in worker threads)
# ---------------------------------------------------------------------------

def _derive_positions(match: MatchData) -> dict:
    try:
        from .positions import derive_match_positions
        return derive_match_positions(match.players)
    except Exception as exc:
        logger.warning("Position derivation failed: %s", exc)
        return {}


def _detect_step(
    match: MatchData, player_slot: int, lang: str, position: int | None,
) -> tuple[list[DeathEvent], list[DecisionPoint]]:
    try:
        from .decision_detector import detect_decisions
        return detect_decisions(match, player_slot, lang=lang, position=position)
    except Exception as exc:
        logger.error("Decision detection failed: %s", exc, exc_info=True)
        return [], []


def _causal_step(
    match: MatchData, deaths: list[DeathEvent], player_slot: int, lang: str,
) -> dict[int, CausalChain]:
    chains: dict[int, CausalChain] = {}
    try:
        from .causal_chain import build_causal_chain
    except Exception as exc:
        logger.error("Causal chain module unavailable: %s", exc)
        return chains
    for death in deaths:
        try:
            chains[death.time] = build_causal_chain(match, death, player_slot, lang=lang)
        except Exception as exc:
            logger.error("Causal chain failed (t=%s): %s", death.time, exc, exc_info=True)
    return chains


def _context_step(
    match: MatchData,
    decisions: list[DecisionPoint],
    causal_chains: dict[int, CausalChain],
    position: int | None,
    lang: str,
    knowledge_for: set[int] | None = None,
) -> list[tuple[DecisionContext, str, Evaluation]]:
    """Build (context, knowledge, rule-based evaluation) per decision point.

    Knowledge is only retrieved for the decision indices in
    ``knowledge_for`` (the ones going to the LLM); rules do not use it.
    """
    try:
        from .context_builder import build_context
    except Exception as exc:
        logger.error("Context builder unavailable: %s", exc)
        return []
    get_knowledge_for_context = None
    if knowledge_for:
        try:
            from .knowledge_base import get_knowledge_for_context
        except Exception as exc:
            logger.warning("Knowledge base unavailable: %s", exc)

    out: list[tuple[DecisionContext, str, Evaluation]] = []
    for i, dp in enumerate(decisions):
        try:
            chain = None
            if dp.point_type == "death" and dp.related_death is not None:
                chain = causal_chains.get(dp.related_death.time)
            ctx = build_context(match, dp, causal_chain=chain, lang=lang)
            if position:
                # Position 1-5 lets knowledge retrieval add a role tag.
                ctx.match_context["player_position"] = position
        except Exception as exc:
            logger.error("Context build failed (t=%s): %s", dp.time, exc, exc_info=True)
            continue
        knowledge = ""
        if get_knowledge_for_context is not None and knowledge_for and i in knowledge_for:
            try:
                knowledge = get_knowledge_for_context(ctx, lang=lang) or ""
            except Exception as exc:
                logger.warning("Knowledge retrieval failed (t=%s): %s", dp.time, exc)
        try:
            rules = evaluate_decision_fallback(ctx, lang=lang)
        except Exception as exc:
            logger.error("Rule evaluation failed (t=%s): %s", dp.time, exc, exc_info=True)
            rules = _error_evaluation(ctx, lang)
        out.append((ctx, knowledge, rules))
    return out


def _compare_step(match: MatchData, player: PlayerData, lang: str, position: int | None) -> dict:
    try:
        from .pro_patterns import compare_player, format_pro_comparison
        return format_pro_comparison(compare_player(player, match, lang=lang, position=position))
    except Exception as exc:
        logger.error("Benchmark comparison failed: %s", exc, exc_info=True)
        return {}


def _phase_stats_step(match: MatchData, player: PlayerData, deaths: list[DeathEvent]) -> list:
    try:
        from .phases import slice_player_phases
        return slice_player_phases(match, player, death_times=[d.time for d in deaths])
    except Exception as exc:
        logger.error("Phase slicing failed: %s", exc, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {"critical": 0, "major": 1, "positive": 2, "minor": 3}
_TYPE_RANK = {"death": 0, "fight_engage": 1, "overextend": 2, "bad_trade": 3, "good_play": 4, "missed_cs": 5}


def _llm_priority(dp: DecisionPoint) -> tuple:
    """Sort key for LLM slots: deaths first, then by severity and type."""
    return (
        dp.point_type != "death",
        is_time_unknown(dp),
        _SEVERITY_RANK.get(dp.severity, 4),
        _TYPE_RANK.get(dp.point_type, 6),
        dp.time,
    )


def _error_evaluation(context: DecisionContext, lang: str) -> Evaluation:
    return Evaluation(
        decision_point=context.decision_point,
        score="C",
        analysis=("An error occurred during evaluation." if lang == "en"
                  else "评估过程中出现错误，无法完成分析。"),
    )


def _rules_copy(rules: Evaluation, degraded: bool) -> Evaluation:
    ev = Evaluation(
        decision_point=rules.decision_point,
        score=normalize_grade(rules.score),
        analysis=rules.analysis,
        mistake=rules.mistake,
        better_play=rules.better_play,
        lesson=rules.lesson,
    )
    ev.degraded = degraded
    return ev


def _fill_missing_fields(ev: Evaluation, rules: Evaluation) -> Evaluation:
    """Backfill empty structured fields of an LLM verdict from the rules."""
    if not ev.mistake and rules.mistake:
        ev.mistake = rules.mistake
    if not ev.better_play and rules.better_play:
        ev.better_play = rules.better_play
    if not ev.lesson and rules.lesson:
        ev.lesson = rules.lesson
    ev.score = normalize_grade(ev.score)
    ev.used_llm = True
    ev.degraded = False
    return ev


async def _safe_evaluate(
    context: DecisionContext, knowledge: str, lang: str = "zh",
    match: MatchData | None = None, rules: Evaluation | None = None,
) -> Evaluation:
    """Evaluate one decision; never raises (except cancellation).

    Agent first (when enabled); if the endpoint does not support tool
    calling, the single-prompt evaluator; on any other failure the rule
    evaluation, marked ``degraded``.
    """
    if rules is None:
        try:
            rules = evaluate_decision_fallback(context, lang=lang)
        except Exception:
            rules = _error_evaluation(context, lang)
    if not is_api_available():
        return _rules_copy(rules, degraded=False)

    t = context.decision_point.time
    if _HAS_AGENT and match is not None and is_agent_enabled():
        try:
            ev = await run_decision_agent(match, context, knowledge, lang=lang)
            return _fill_missing_fields(ev, rules)
        except asyncio.CancelledError:
            raise
        except AgentUnsupportedError:
            logger.info("Agent unsupported by the endpoint; using single-prompt evaluation (t=%s)", t)
        except Exception as exc:
            logger.warning("Decision agent failed (t=%s), using rules: %s", t, exc)
            return _rules_copy(rules, degraded=True)

    try:
        ev = await evaluate_decision(context, knowledge, lang=lang)
        return _fill_missing_fields(ev, rules)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Single-prompt evaluation failed (t=%s), using rules: %s", t, exc)
        return _rules_copy(rules, degraded=True)


def _calculate_overall_grade(evaluations: list[Evaluation]) -> str:
    """Weighted overall grade (deaths count 1.5x); always a valid grade."""
    if not evaluations:
        return "B"
    total_weight = 0.0
    weighted_sum = 0.0
    for ev in evaluations:
        weight = 1.5 if ev.decision_point.point_type == "death" else 1.0
        weighted_sum += _SCORE_TO_NUM.get(normalize_grade(ev.score), 2.0) * weight
        total_weight += weight
    return _letter_grade(weighted_sum / total_weight if total_weight > 0 else 3.0)


def _chronological(ev: Evaluation) -> tuple:
    return (is_time_unknown(ev.decision_point), ev.decision_point.time)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def generate_coaching_report(
    match: MatchData, player_slot: int, lang: str = "zh",
    progress_callback: Optional[ProgressCallback] = None,
) -> CoachingReport:
    """Generate a complete coaching report for a player in a match.

    Parameters
    ----------
    match : MatchData
        The full parsed match data.
    player_slot : int
        The slot of the player to analyse.
    lang : str
        Language code ("zh" or "en"). Defaults to "zh".
    progress_callback : async callable, optional
        Receives ``{"stage": str, "current": int, "total": int}`` updates.

    Raises
    ------
    ValueError
        When ``player_slot`` is not a player of ``match``.
    """
    lang = lang if lang in ("zh", "en") else "zh"
    player = _find_player(match, player_slot)
    if player is None:
        raise ValueError(f"player_slot {player_slot} is not a player of match {match.match_id}")

    started = time.monotonic()
    timeout = _coach_timeout()

    async def _emit(stage: str, current: int = 0, total: int = 0) -> None:
        if progress_callback is None:
            return
        try:
            await progress_callback({"stage": stage, "current": current, "total": total})
        except Exception as exc:  # progress reporting must never kill the pipeline
            logger.debug("progress_callback raised: %s", exc)

    await _emit("starting", 0, 1)
    hero_name = player.hero_name

    # Step 0: positions (1-5) for the whole match.
    positions_map = await asyncio.to_thread(_derive_positions, match)
    my_pos = positions_map.get(player_slot)
    position_number = my_pos.number if my_pos is not None and my_pos.number else None
    position_payload: dict = {}
    position_label_text = _POSITION_FALLBACK_LABEL[lang]
    if my_pos is not None and my_pos.number:
        position_label_text = my_pos.label(lang)
        position_payload = {
            "number": my_pos.number,
            "label_zh": my_pos.label_zh,
            "label_en": my_pos.label_en,
            "confidence": my_pos.confidence,
            "method": my_pos.method,
        }

    # Step 1: decision points.
    await _emit("detect", 0, 1)
    deaths, decisions = await asyncio.to_thread(_detect_step, match, player_slot, lang, position_number)

    # Step 1.5: causal chains for deaths.
    await _emit("causal_chain", 0, max(1, len(deaths)))
    causal_chains = await asyncio.to_thread(_causal_step, match, deaths, player_slot, lang)

    # Step 2-3: contexts, knowledge (LLM-bound points only) and rule baselines.
    llm_on = is_api_available()
    order = sorted(range(len(decisions)), key=lambda i: _llm_priority(decisions[i]))
    llm_decisions = set(order[:MAX_LLM_EVALS]) if llm_on else set()
    await _emit("context", 0, max(1, len(decisions)))
    prepared = await asyncio.to_thread(
        _context_step, match, decisions, causal_chains, position_number, lang, llm_decisions,
    )
    # Contexts whose build failed are skipped, so re-derive the LLM indices.
    llm_indices = (
        sorted(range(len(prepared)), key=lambda i: _llm_priority(prepared[i][0].decision_point))[:MAX_LLM_EVALS]
        if llm_on else []
    )
    evaluations_by_index: dict[int, Evaluation] = {
        i: _rules_copy(prepared[i][2], degraded=False) for i in range(len(prepared))
    }

    phase_stats_list = await asyncio.to_thread(_phase_stats_step, match, player, deaths)
    phase_reviews: list[PhaseReview | None] = [None] * len(phase_stats_list)

    # Step 4: LLM evaluations + phase reviews, concurrently, under the deadline.
    comparison_task = asyncio.create_task(
        asyncio.to_thread(_compare_step, match, player, lang, position_number)
    )
    try:
        await _emit("evaluate", 0, max(1, len(llm_indices)))
        budget = _llm_budget(timeout, time.monotonic() - started)
        tasks = await _run_llm_stage(
            prepared, llm_indices, phase_stats_list, budget, match, hero_name,
            position_label_text, player.team, lang, _emit,
        )
    except BaseException:
        comparison_task.cancel()
        raise

    timed_out = 0
    for task, (kind, idx) in tasks.items():
        result = None
        if task.cancelled():
            timed_out += 1
        else:
            try:
                result = task.result()
            except Exception as exc:  # pragma: no cover - helpers never raise
                logger.error("AI coach task failed: %s", exc)
        if kind == "eval":
            evaluations_by_index[idx] = (
                result if isinstance(result, Evaluation) else _rules_copy(prepared[idx][2], degraded=True)
            )
        else:
            phase_reviews[idx] = result if isinstance(result, PhaseReview) else None
    if timed_out:
        logger.warning(
            "AI coach deadline (%.0fs) reached for match %s slot %s: %d LLM task(s) fell back to rules",
            timeout, match.match_id, player_slot, timed_out,
        )
    await _emit("phases", len(phase_stats_list), max(1, len(phase_stats_list)))

    final_phases: list[PhaseReview] = []
    for j, ps in enumerate(phase_stats_list):
        review = phase_reviews[j]
        if review is None:
            review = _phase_fallback(ps, lang, degraded=llm_on)
        final_phases.append(review)

    try:
        pro_comparison_data = await comparison_task
    except Exception as exc:  # pragma: no cover - _compare_step never raises
        logger.error("Benchmark comparison task failed: %s", exc)
        pro_comparison_data = {}

    evaluations = [evaluations_by_index[i] for i in range(len(prepared))]

    # Attach causal chain data for the frontend.
    for ev in evaluations:
        dp = ev.decision_point
        if dp.point_type == "death" and dp.related_death is not None:
            chain = causal_chains.get(dp.related_death.time)
            if chain is not None and chain.steps:
                ev.causal_chain_data = {
                    "root_cause_type": chain.root_cause_type,
                    "root_cause_summary": chain.root_cause_summary,
                    "steps": [
                        {"time": s.time, "type": s.step_type, "description": s.description,
                         "significance": s.significance}
                        for s in chain.steps
                    ],
                }

    # Step 5: compile.
    await _emit("summary", 0, 1)
    death_analyses = sorted((e for e in evaluations if e.decision_point.point_type == "death"), key=_chronological)
    key_moments = sorted((e for e in evaluations if e.decision_point.point_type != "death"), key=_chronological)
    overall_grade = _calculate_overall_grade(evaluations)
    laning_review = _generate_laning_review(match, player, evaluations, lang=lang, positions=positions_map)
    improvement_areas = _detect_improvement_areas(evaluations, lang=lang)
    strengths = _detect_strengths(evaluations, lang=lang)
    summary = _generate_summary(player, overall_grade, evaluations, improvement_areas, strengths, match, lang=lang)

    llm_used = any(e.used_llm for e in evaluations) or any(p.used_llm for p in final_phases)
    degraded = llm_on and (any(e.degraded for e in evaluations) or any(p.degraded for p in final_phases))

    report = CoachingReport(
        match_id=match.match_id,
        player_slot=player_slot,
        hero_name=hero_name,
        overall_grade=overall_grade,
        summary=summary,
        death_analyses=death_analyses,
        key_moments=key_moments,
        laning_review=laning_review,
        improvement_areas=improvement_areas,
        strengths=strengths,
        pro_comparison=pro_comparison_data,
        phase_reviews=final_phases,
        position=position_payload,
        llm_used=llm_used,
        degraded=degraded,
        report_version=REPORT_SCHEMA_VERSION,
    )
    logger.info(
        "AI coach report ready: match=%s slot=%s decisions=%d llm_evals=%d llm_used=%s degraded=%s (%.1fs)",
        match.match_id, player_slot, len(evaluations), len(llm_indices), llm_used, degraded,
        time.monotonic() - started,
    )
    await _emit("done", 1, 1)
    return report


async def _run_llm_stage(
    prepared: list[tuple[DecisionContext, str, Evaluation]],
    llm_indices: list[int],
    phase_stats_list: list,
    budget: float,
    match: MatchData,
    hero_name: str,
    position_label: str,
    team: str,
    lang: str,
    emit: Callable[..., Awaitable[None]],
) -> dict[asyncio.Task, tuple[str, int]]:
    """Run LLM evaluations and phase reviews until done or the deadline.

    Returns every task with its (kind, index); tasks still running at the
    deadline are cancelled (the caller substitutes rule-based results).
    """
    tasks: dict[asyncio.Task, tuple[str, int]] = {}
    with llm_deadline(budget):
        for i in llm_indices:
            ctx, know, rules = prepared[i]
            task = asyncio.create_task(_safe_evaluate(ctx, know, lang=lang, match=match, rules=rules))
            tasks[task] = ("eval", i)
        for j, ps in enumerate(phase_stats_list):
            task = asyncio.create_task(_run_phase(hero_name, position_label, team, ps, lang))
            tasks[task] = ("phase", j)

    eval_total, phase_total = len(llm_indices), len(phase_stats_list)
    eval_done = phase_done = 0
    pending = set(tasks)
    deadline = time.monotonic() + budget
    try:
        while pending:
            # LLM calls stop at the deadline on their own; the grace period
            # lets their rule fallbacks come back before cancelling.
            wait_for = max(0.0, deadline - time.monotonic()) + 2.0
            done, pending = await asyncio.wait(pending, timeout=wait_for, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                break
            for task in done:
                if tasks[task][0] == "eval":
                    eval_done += 1
                else:
                    phase_done += 1
            if eval_done < eval_total:
                await emit("evaluate", eval_done, eval_total)
            elif pending:
                await emit("phases", phase_done, max(1, phase_total))
    finally:
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    return tasks


async def _run_phase(hero_name: str, position_label: str, team: str, ps, lang: str) -> PhaseReview:
    from .evaluator import evaluate_phase

    return await evaluate_phase(
        hero_name=hero_name,
        position_label=position_label,
        team=team,
        phase_key=ps.phase,
        phase_time_range=ps.time_range,
        phase_stats_dict=ps.to_dict(),
        lang=lang,
    )


def _phase_fallback(ps, lang: str, degraded: bool) -> PhaseReview:
    from .evaluator import _phase_review_fallback

    stats = ps.to_dict()
    review = _phase_review_fallback(
        PhaseReview(phase=ps.phase, time_range=ps.time_range, stats=stats), stats, lang=lang,
    )
    review.degraded = degraded
    return review
