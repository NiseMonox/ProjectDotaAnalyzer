"""Main orchestrator for the AI coaching pipeline.

Pipeline:
1. Detect decision points (death events + key moments)
2. Build context for each decision point
3. Inject relevant knowledge
4. Evaluate each decision (LLM or fallback)
5. Compile into coaching report
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

ProgressCallback = Callable[[dict], Awaitable[None]]

from ..models import MatchData, PlayerData
from .models import (
    CausalChain,
    CoachingReport,
    DecisionContext,
    DecisionPoint,
    Evaluation,
)
from .evaluator import (
    evaluate_decision,
    evaluate_decision_fallback,
    is_api_available,
)

logger = logging.getLogger(__name__)

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
    """Find a player by slot in the match data."""
    for p in match.players:
        if p.player_slot == player_slot:
            return p
    return None


# ---------------------------------------------------------------------------
# Internationalization helpers
# ---------------------------------------------------------------------------

_LANE_NAMES: dict[str, dict[int, str]] = {
    # OpenDota's lane_role=4 ("Jungle") is essentially extinct in modern Dota
    # (post-7.20); ~95% of cases are roaming supports. Treat it as roaming.
    "zh": {0: "游走", 1: "安全路", 2: "中路", 3: "劣势路", 4: "游走"},
    "en": {0: "Roaming", 1: "Safe Lane", 2: "Mid Lane", 3: "Off Lane", 4: "Roaming"},
}

_LANE_RESULT_LABELS: dict[str, dict[str, str]] = {
    "zh": {"won": "优势", "even": "均势", "lost": "劣势"},
    "en": {"won": "Won", "even": "Even", "lost": "Lost"},
}

_CS_EFFICIENCY_LABELS: dict[str, dict[str, str]] = {
    "zh": {"excellent": "优秀", "good": "良好", "average": "一般", "poor": "需要提高"},
    "en": {"excellent": "Excellent", "good": "Good", "average": "Average", "poor": "Needs Improvement"},
}

_UNKNOWN_OPPONENT: dict[str, str] = {
    "zh": "未知对手",
    "en": "Unknown opponent",
}

_UNKNOWN_LANE: dict[str, str] = {
    "zh": "未知路",
    "en": "Unknown Lane",
}

_CATEGORY_LABELS: dict[str, dict[str, str]] = {
    "zh": {
        "地图意识": "地图意识",
        "对线能力": "对线能力",
        "团战参与": "团战参与",
        "资源利用": "资源利用",
        "视野控制": "视野控制",
    },
    "en": {
        "地图意识": "Map Awareness",
        "对线能力": "Laning",
        "团战参与": "Teamfight Participation",
        "资源利用": "Resource Efficiency",
        "视野控制": "Vision Control",
    },
}

_IMPROVEMENT_SUGGESTIONS: dict[str, dict[str, str]] = {
    "zh": {
        "地图意识": "多看小地图，养成每5秒看一次的习惯。注意敌方英雄消失的信号。",
        "对线能力": "练习补刀，特别是塔下补刀。注意换血时机和消耗品使用。",
        "团战参与": "注意开团时机和站位。优先集火关键目标，注意保护己方核心。",
        "资源利用": "提高打野效率，减少游走等待时间。确保每分钟都有金币收入。",
        "视野控制": "养成买眼插眼的习惯，特别是团战前和烟雾时。",
    },
    "en": {
        "地图意识": "Check the minimap more frequently — aim for every 5 seconds. Watch for missing enemy heroes.",
        "对线能力": "Practice last-hitting, especially under tower. Mind your trade timing and consumable usage.",
        "团战参与": "Pay attention to engagement timing and positioning. Focus fire key targets and protect your cores.",
        "资源利用": "Improve jungle efficiency and reduce idle time. Ensure consistent gold income every minute.",
        "视野控制": "Make warding a habit, especially before teamfights and smoke ganks.",
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


# ---------------------------------------------------------------------------
# Laning review
# ---------------------------------------------------------------------------

def _generate_laning_review(
    match: MatchData, player: PlayerData, evaluations: list[Evaluation],
    lang: str = "zh",
) -> dict[str, Any]:
    """Generate a laning phase review for the player.

    Returns a dict with cs_10min, denies_10min, lane_matchup,
    lane_result, key_trades, cs_efficiency, and summary.
    """
    # Calculate 10-minute CS — count only lane creeps (exclude jungle neutrals)
    lane_creep_cs = 0
    if player.laning_creep_kills:
        lane_creep_cs = sum(
            1 for ck in player.laning_creep_kills
            if ck.creep_type in ("melee", "ranged", "siege")
        )
    cs_10min = lane_creep_cs

    # Fallback: use lh_t or snapshots if no laning_creep_kills data
    if cs_10min == 0:
        if player.lh_t and len(player.lh_t) > 10:
            cs_10min = player.lh_t[10]
        elif player.laning_snapshots:
            cs_10min = player.laning_snapshots[-1].last_hits
        elif player.lh_t:
            cs_10min = player.lh_t[-1]

    denies_10min = player.denies  # fallback to total denies
    if player.laning_snapshots:
        denies_10min = player.laning_snapshots[-1].denies

    # Determine lane opponent
    lane_map = _LANE_NAMES.get(lang, _LANE_NAMES["zh"])
    player_lane_name = lane_map.get(player.lane, _UNKNOWN_LANE.get(lang, "未知路"))

    # Find lane opponent(s) - enemies in the same lane
    opponents: list[str] = []
    if player.lane > 0:
        for p in match.players:
            if p.team != player.team and p.lane == player.lane:
                opponents.append(p.hero_name)
    # If no lane info or too many matches (unreliable), try nearby from snapshots
    if len(opponents) > 3 or not opponents:
        opponents = []
        if player.laning_snapshots and len(player.laning_snapshots) > 5:
            # Find enemies whose laning snapshots are closest to ours
            from .decision_detector import _game_to_map, _dist
            my_snaps = player.laning_snapshots[5:15]  # samples from 25s-75s
            if my_snaps:
                avg_x = sum(s.x for s in my_snaps) / len(my_snaps)
                avg_y = sum(s.y for s in my_snaps) / len(my_snaps)
                mx, my = _game_to_map(avg_x, avg_y)
                for p in match.players:
                    if p.team == player.team or not p.laning_snapshots:
                        continue
                    e_snaps = p.laning_snapshots[5:15]
                    if e_snaps:
                        ex = sum(s.x for s in e_snaps) / len(e_snaps)
                        ey = sum(s.y for s in e_snaps) / len(e_snaps)
                        emx, emy = _game_to_map(ex, ey)
                        if _dist(mx, my, emx, emy) < 30:
                            opponents.append(p.hero_name)

    unknown_opp = _UNKNOWN_OPPONENT.get(lang, "未知对手")
    separator = ", " if lang == "en" else "、"
    lane_matchup = (
        f"{player.hero_name} ({player_lane_name}) vs "
        + (separator.join(opponents[:3]) if opponents else unknown_opp)
    )

    # Lane result heuristic based on CS comparison with primary lane opponent
    lane_result = "even"
    if opponents:
        opp_player = None
        for p in match.players:
            if p.hero_name == opponents[0]:
                opp_player = p
                break
        if opp_player:
            # Get opponent 10-min CS using same hierarchy as player
            opp_cs = 0
            if opp_player.laning_creep_kills:
                opp_cs = sum(
                    1 for ck in opp_player.laning_creep_kills
                    if ck.creep_type in ("melee", "ranged", "siege")
                )
            if opp_cs == 0:
                if opp_player.lh_t and len(opp_player.lh_t) > 10:
                    opp_cs = opp_player.lh_t[10]
                elif opp_player.laning_snapshots:
                    opp_cs = opp_player.laning_snapshots[-1].last_hits
                elif opp_player.lh_t:
                    opp_cs = opp_player.lh_t[-1]

            # Also consider net worth and level difference from snapshots
            my_nw = player.laning_snapshots[-1].net_worth if player.laning_snapshots else 0
            opp_nw = opp_player.laning_snapshots[-1].net_worth if opp_player.laning_snapshots else 0
            my_lvl = player.laning_snapshots[-1].level if player.laning_snapshots else 0
            opp_lvl = opp_player.laning_snapshots[-1].level if opp_player.laning_snapshots else 0

            # Score based on multiple factors
            advantage_score = 0
            cs_diff = cs_10min - opp_cs
            if abs(cs_diff) > 5:
                advantage_score += (1 if cs_diff > 0 else -1)
            nw_diff = my_nw - opp_nw
            if abs(nw_diff) > 300:
                advantage_score += (1 if nw_diff > 0 else -1)
            lvl_diff = my_lvl - opp_lvl
            if abs(lvl_diff) >= 1:
                advantage_score += (1 if lvl_diff > 0 else -1)

            if advantage_score >= 2:
                lane_result = "won"
            elif advantage_score <= -2:
                lane_result = "lost"
            else:
                lane_result = "even"

    # Key trades from laning damage events
    key_trades: list[str] = []
    if player.laning_damage_events:
        total_dmg_dealt = sum(e.damage for e in player.laning_damage_events)
        if lang == "en":
            key_trades.append(f"Dealt {total_dmg_dealt} hero damage during laning phase")
        else:
            key_trades.append(f"对线阶段造成 {total_dmg_dealt} 点英雄伤害")

    # Collect laning-related evaluations
    laning_evals = [e for e in evaluations if e.decision_point.time < 600]
    score_label = "Score" if lang == "en" else "评分"
    for ev in laning_evals[:3]:
        key_trades.append(
            f"[{ev.decision_point.time // 60}:{ev.decision_point.time % 60:02d}] "
            f"{ev.decision_point.brief} ({score_label}: {ev.score})"
        )

    # CS efficiency: lane creeps killed / lane creeps spawned
    # Dota 2: ~82 lane creeps per lane in 10 minutes
    # (20 waves: 10 waves x 4 creeps + 10 waves x 5 creeps + 4 siege ~ 94,
    #  minus creeps that die to each other ~ 82 practical max)
    theoretical_lane_creeps = 82
    lane_cs_for_efficiency = lane_creep_cs if lane_creep_cs > 0 else cs_10min
    efficiency_pct = (
        lane_cs_for_efficiency / theoretical_lane_creeps * 100
        if theoretical_lane_creeps > 0 else 0
    )
    eff_labels = _CS_EFFICIENCY_LABELS.get(lang, _CS_EFFICIENCY_LABELS["zh"])
    if efficiency_pct >= 80:
        cs_efficiency = f"{eff_labels['excellent']} ({efficiency_pct:.0f}%)"
    elif efficiency_pct >= 60:
        cs_efficiency = f"{eff_labels['good']} ({efficiency_pct:.0f}%)"
    elif efficiency_pct >= 40:
        cs_efficiency = f"{eff_labels['average']} ({efficiency_pct:.0f}%)"
    else:
        cs_efficiency = f"{eff_labels['poor']} ({efficiency_pct:.0f}%)"

    # Lane result display label
    lane_result_label = _LANE_RESULT_LABELS.get(lang, _LANE_RESULT_LABELS["zh"]).get(
        lane_result, _LANE_RESULT_LABELS.get(lang, _LANE_RESULT_LABELS["zh"])["even"]
    )

    # Summary
    if lang == "en":
        summary_parts = [
            f"{player.hero_name} had a {lane_result_label.lower()} laning phase in {player_lane_name}. "
        ]
        summary_parts.append(
            f"{cs_10min} last hits at 10 min ({efficiency_pct:.0f}% efficiency), {denies_10min} denies. "
        )
        if lane_result == "lost":
            summary_parts.append("CS efficiency and trading need improvement.")
        elif lane_result == "won":
            summary_parts.append("Strong laning performance — keep it up.")
        else:
            summary_parts.append("Decent laning phase with some room for improvement.")
    else:
        summary_parts = [
            f"{player.hero_name}在{player_lane_name}的对线表现为{lane_result_label}。"
        ]
        summary_parts.append(
            f"10分钟补刀{cs_10min}个（效率{efficiency_pct:.0f}%），反补{denies_10min}个。"
        )
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
    "death": "地图意识",
    "overextend": "地图意识",
    "no_retreat": "地图意识",
    "bad_trade": "对线能力",
    "missed_cs": "对线能力",
    "fight_engage": "团战参与",
    "good_play": "团战参与",
}


def _detect_improvement_areas(
    evaluations: list[Evaluation],
    lang: str = "zh",
) -> list[dict[str, Any]]:
    """Detect recurring weaknesses from evaluations.

    Groups evaluations by category and flags categories where the
    player consistently scored poorly.
    """
    category_scores: dict[str, list[str]] = {}
    category_details: dict[str, list[str]] = {}
    no_mistake = "None" if lang == "en" else "无"

    for ev in evaluations:
        cat = _CATEGORY_MAP.get(ev.decision_point.point_type, "资源利用")
        category_scores.setdefault(cat, []).append(ev.score)
        if ev.mistake and ev.mistake != "无" and ev.mistake != "None":
            category_details.setdefault(cat, []).append(ev.mistake)

    cat_labels = _CATEGORY_LABELS.get(lang, _CATEGORY_LABELS["zh"])
    suggestions = _IMPROVEMENT_SUGGESTIONS.get(lang, _IMPROVEMENT_SUGGESTIONS["zh"])
    default_suggestion = _DEFAULT_IMPROVEMENT.get(lang, _DEFAULT_IMPROVEMENT["zh"])
    detail_sep = "; " if lang == "en" else "；"

    areas: list[dict[str, Any]] = []
    for cat, scores in category_scores.items():
        avg = sum(_SCORE_TO_NUM.get(s, 2) for s in scores) / len(scores)
        if avg < 3.0:  # Below B average -> improvement area
            details = category_details.get(cat, [])
            # Deduplicate similar issues
            unique_details = list(dict.fromkeys(details))[:3]
            suggestion = suggestions.get(cat, default_suggestion)
            detail_text = detail_sep.join(unique_details) if unique_details else ""
            if lang == "en":
                description = f"{detail_text}. {suggestion}" if detail_text else suggestion
            else:
                description = f"{detail_text}。{suggestion}" if detail_text else suggestion
            display_cat = cat_labels.get(cat, cat)
            areas.append(
                {
                    "category": display_cat,
                    "avg_score": _letter_grade(avg),
                    "occurrences": len(scores),
                    "description": description,
                }
            )

    # Sort by severity (lowest avg score first)
    areas.sort(key=lambda a: _SCORE_TO_NUM.get(a["avg_score"], 2))
    return areas


def _improvement_suggestion(category: str, lang: str = "zh") -> str:
    """Return a general improvement suggestion for a category."""
    suggestions = _IMPROVEMENT_SUGGESTIONS.get(lang, _IMPROVEMENT_SUGGESTIONS["zh"])
    default = _DEFAULT_IMPROVEMENT.get(lang, _DEFAULT_IMPROVEMENT["zh"])
    return suggestions.get(category, default)


def _detect_strengths(
    evaluations: list[Evaluation],
    lang: str = "zh",
) -> list[dict[str, Any]]:
    """Detect recurring strengths from evaluations.

    Groups evaluations by category and flags categories where the
    player consistently scored well.
    """
    category_scores: dict[str, list[str]] = {}
    category_details: dict[str, list[str]] = {}

    for ev in evaluations:
        cat = _CATEGORY_MAP.get(ev.decision_point.point_type, "资源利用")
        category_scores.setdefault(cat, []).append(ev.score)
        if ev.score in ("S", "A"):
            desc = ev.decision_point.brief or ev.analysis[:50]
            category_details.setdefault(cat, []).append(desc)

    cat_labels = _CATEGORY_LABELS.get(lang, _CATEGORY_LABELS["zh"])
    default_highlight = _STRENGTH_DEFAULT.get(lang, _STRENGTH_DEFAULT["zh"])
    detail_sep = "; " if lang == "en" else "；"

    strengths: list[dict[str, Any]] = []
    for cat, scores in category_scores.items():
        avg = sum(_SCORE_TO_NUM.get(s, 2) for s in scores) / len(scores)
        if avg >= 3.5:  # A or above average -> strength
            details = category_details.get(cat, [])
            unique_details = list(dict.fromkeys(details))[:3]
            highlight_text = detail_sep.join(unique_details) if unique_details else default_highlight
            display_cat = cat_labels.get(cat, cat)
            strengths.append(
                {
                    "category": display_cat,
                    "avg_score": _letter_grade(avg),
                    "occurrences": len(scores),
                    "description": highlight_text,
                }
            )

    # Sort by score (highest first)
    strengths.sort(
        key=lambda s: _SCORE_TO_NUM.get(s["avg_score"], 2), reverse=True
    )
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
    """Generate a text summary of the coaching report."""
    won = (
        (player.team == "radiant" and match.radiant_win)
        or (player.team == "dire" and not match.radiant_win)
    )
    kda = f"{player.kills}/{player.deaths}/{player.assists}"
    duration_min = match.duration // 60

    grade_comments = _GRADE_COMMENTS.get(lang, _GRADE_COMMENTS["zh"])

    if lang == "en":
        result_str = "won" if won else "lost"
        parts = [
            f"{player.hero_name} {result_str} this match with a KDA of {kda} "
            f"in {duration_min} minutes. "
        ]
        parts.append(
            f"Overall grade {overall_grade}: {grade_comments.get(overall_grade, '')} "
        )
        if strengths:
            parts.append(f"Key strength: {strengths[0]['category']}. ")
        if improvement_areas:
            parts.append(
                f"Top area to improve: {improvement_areas[0]['category']} "
                f"({improvement_areas[0]['occurrences']} occurrences). "
            )
        death_evals = [
            e for e in evaluations if e.decision_point.point_type == "death"
        ]
        if death_evals:
            parts.append(
                f"{len(death_evals)} death(s) analyzed — review the cause of each death carefully."
            )
    else:
        result_str = "获胜" if won else "失败"
        parts = [
            f"本场比赛{player.hero_name}{result_str}，"
            f"KDA为{kda}，比赛时长{duration_min}分钟。"
        ]
        parts.append(f"综合评分{overall_grade}：{grade_comments.get(overall_grade, '')}")
        if strengths:
            parts.append(f"主要优势是{strengths[0]['category']}方面的表现。")
        if improvement_areas:
            parts.append(
                f"最需要改进的是{improvement_areas[0]['category']}，"
                f"相关问题出现了{improvement_areas[0]['occurrences']}次。"
            )
        death_evals = [
            e for e in evaluations if e.decision_point.point_type == "death"
        ]
        if death_evals:
            parts.append(
                f"本场共分析了{len(death_evals)}次死亡，请重点关注每次死亡的原因。"
            )

    return "".join(parts)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def generate_coaching_report(
    match: MatchData, player_slot: int, lang: str = "zh",
    progress_callback: Optional[ProgressCallback] = None,
) -> CoachingReport:
    """Generate a complete coaching report for a player in a match.

    Pipeline:
    1. Detect decision points (death events + key moments)
    2. Build context for each decision point
    3. Inject relevant knowledge
    4. Evaluate each decision (LLM or fallback)
    5. Compile into coaching report

    Parameters
    ----------
    match : MatchData
        The full parsed match data.
    player_slot : int
        The slot of the player to analyse.
    lang : str
        Language code ("zh" or "en"). Defaults to "zh".

    Returns
    -------
    CoachingReport
        The completed coaching report with evaluations, laning review,
        improvement areas, and strengths.
    """
    async def _emit(stage: str, current: int = 0, total: int = 0) -> None:
        if progress_callback is None:
            return
        try:
            await progress_callback({"stage": stage, "current": current, "total": total})
        except Exception as exc:  # never let progress reporting kill the pipeline
            logger.debug("progress_callback raised: %s", exc)

    await _emit("starting", 0, 1)

    player = _find_player(match, player_slot)
    if player is None:
        if lang == "en":
            summary_text = f"Player data not found for player_slot={player_slot}."
            hero_label = "Unknown"
        else:
            summary_text = f"未找到 player_slot={player_slot} 的玩家数据。"
            hero_label = "未知"
        return CoachingReport(
            match_id=match.match_id,
            player_slot=player_slot,
            hero_name=hero_label,
            overall_grade="C",
            summary=summary_text,
        )

    hero_name = player.hero_name

    # ------------------------------------------------------------------
    # Step 0.5: Derive role/position (1-5) for the entire match
    # ------------------------------------------------------------------
    position_payload: dict = {}
    position_label_text = "辅助" if lang == "zh" else "Support"
    try:
        from .positions import derive_match_positions

        positions_map = derive_match_positions(match.players)
        my_pos = positions_map.get(player_slot)
        if my_pos:
            position_label_text = my_pos.label(lang)
            position_payload = {
                "number": my_pos.number,
                "label_zh": my_pos.label_zh,
                "label_en": my_pos.label_en,
                "confidence": my_pos.confidence,
                "method": my_pos.method,
            }
    except Exception as exc:
        logger.warning("position derivation failed: %s", exc)

    # ------------------------------------------------------------------
    # Step 1: Detect decision points
    # ------------------------------------------------------------------
    await _emit("detect", 0, 1)
    try:
        from .decision_detector import detect_decisions

        deaths, decisions = detect_decisions(match, player_slot, lang=lang)
    except ImportError:
        logger.warning("decision_detector 模块不可用，使用空决策列表")
        deaths, decisions = [], []
    except Exception as exc:
        logger.error("detect_decisions 失败: %s", exc, exc_info=True)
        deaths, decisions = [], []

    # ------------------------------------------------------------------
    # Step 1.5: Build causal chains for deaths
    # ------------------------------------------------------------------
    await _emit("causal_chain", 0, max(1, len(deaths)))
    causal_chains: dict[int, CausalChain] = {}  # death_time -> chain
    try:
        from .causal_chain import build_causal_chain
        for death in deaths:
            try:
                chain = build_causal_chain(match, death, player_slot, lang=lang)
                causal_chains[death.time] = chain
            except Exception as exc:
                logger.error("build_causal_chain 失败 (time=%s): %s", death.time, exc)
    except ImportError:
        logger.info("causal_chain 模块不可用，跳过因果链分析")
    except Exception as exc:
        logger.error("导入 causal_chain 失败: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Step 2: Build contexts
    # ------------------------------------------------------------------
    await _emit("context", 0, max(1, len(decisions)))
    contexts: list[DecisionContext] = []
    try:
        from .context_builder import build_context

        for dp in decisions:
            try:
                # Find matching causal chain for death decisions
                chain = None
                if dp.point_type == "death" and dp.related_death:
                    chain = causal_chains.get(dp.related_death.time)
                ctx = build_context(match, dp, causal_chain=chain, lang=lang)
                contexts.append(ctx)
            except Exception as exc:
                logger.error(
                    "build_context 失败 (time=%s): %s", dp.time, exc
                )
    except ImportError:
        logger.warning("context_builder 模块不可用，跳过上下文构建")
    except Exception as exc:
        logger.error("导入 context_builder 失败: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Step 3: Inject knowledge
    # ------------------------------------------------------------------
    knowledge_strings: list[str] = []
    try:
        from .knowledge_base import get_knowledge_for_context

        for ctx in contexts:
            try:
                knowledge_strings.append(get_knowledge_for_context(ctx, lang=lang))
            except Exception:
                knowledge_strings.append("")
    except ImportError:
        logger.warning("knowledge_base 模块不可用，跳过知识注入")
        knowledge_strings = [""] * len(contexts)
    except Exception as exc:
        logger.error("导入 knowledge_base 失败: %s", exc, exc_info=True)
        knowledge_strings = [""] * len(contexts)

    # ------------------------------------------------------------------
    # Step 4: Evaluate decisions
    # ------------------------------------------------------------------
    # Prioritize: evaluate deaths first, then limit total evaluations
    # to avoid excessive API calls (rate limit friendly)
    MAX_LLM_EVALS = 8  # max decision points to send to LLM
    death_contexts = [
        (ctx, know) for ctx, know in zip(contexts, knowledge_strings)
        if ctx.decision_point.point_type == "death"
    ]
    other_contexts = [
        (ctx, know) for ctx, know in zip(contexts, knowledge_strings)
        if ctx.decision_point.point_type != "death"
    ]
    # Deaths first, then fill remaining slots with other decisions
    prioritized = death_contexts + other_contexts

    evaluations: list[Evaluation] = []
    if prioritized:
        if is_api_available():
            # Evaluate deaths with LLM, others with rules to save API budget
            llm_batch = prioritized[:MAX_LLM_EVALS]
            rule_batch = prioritized[MAX_LLM_EVALS:]

            await _emit("evaluate", 0, len(llm_batch))

            # Run all LLM evals concurrently; emit progress as each finishes.
            # Order is restored after all complete. _safe_evaluate handles
            # per-call failures by returning a rule-based fallback, so a few
            # API errors do not require pessimistic early-abort logic.
            async def _eval_with_index(idx: int, ctx: DecisionContext, know: str):
                ev, _used_api = await _safe_evaluate(ctx, know, lang=lang)
                return idx, ev

            tasks = [
                asyncio.create_task(_eval_with_index(i, ctx, know))
                for i, (ctx, know) in enumerate(llm_batch)
            ]
            done_count = 0
            indexed_results: list[tuple[int, Evaluation]] = []
            for fut in asyncio.as_completed(tasks):
                idx, ev = await fut
                indexed_results.append((idx, ev))
                done_count += 1
                await _emit("evaluate", done_count, len(llm_batch))

            # Restore original order so death/key-moment grouping stays stable.
            indexed_results.sort(key=lambda t: t[0])
            evaluations.extend(ev for _, ev in indexed_results)

            # Rule-based for remaining (cheap, synchronous).
            for ctx, _know in rule_batch:
                try:
                    evaluations.append(evaluate_decision_fallback(ctx, lang=lang))
                except Exception as exc:
                    logger.error("evaluate_decision_fallback 失败: %s", exc)
        else:
            logger.info("DeepSeek API 不可用，使用规则回退评估")
            for ctx, _know in prioritized:
                try:
                    evaluations.append(evaluate_decision_fallback(ctx, lang=lang))
                except Exception as exc:
                    logger.error("evaluate_decision_fallback 失败: %s", exc)

    # ------------------------------------------------------------------
    # Step 4.5: Attach causal chain data to evaluations for frontend
    # ------------------------------------------------------------------
    for ev in evaluations:
        if ev.decision_point.point_type == "death" and ev.decision_point.related_death:
            chain = causal_chains.get(ev.decision_point.related_death.time)
            if chain and chain.steps:
                ev.causal_chain_data = {
                    "root_cause_type": chain.root_cause_type,
                    "root_cause_summary": chain.root_cause_summary,
                    "steps": [
                        {
                            "time": s.time,
                            "type": s.step_type,
                            "description": s.description,
                            "significance": s.significance,
                        }
                        for s in chain.steps
                    ],
                }

    # ------------------------------------------------------------------
    # Step 4.7: Professional pattern comparison
    # ------------------------------------------------------------------
    await _emit("comparison", 0, 1)
    pro_comparison_data: dict = {}
    try:
        from .pro_patterns import compare_player, format_pro_comparison
        pro_report = compare_player(player, match, lang=lang)
        pro_comparison_data = format_pro_comparison(pro_report)
    except ImportError:
        logger.info("pro_patterns 模块不可用，跳过职业数据对比")
    except Exception as exc:
        logger.error("pro_patterns 对比失败: %s", exc)

    # ------------------------------------------------------------------
    # Step 4.8: Per-phase reviews (laning / early-mid / mid-late / late /
    # ultra-late). Each phase gets one LLM call, parallelized.
    # ------------------------------------------------------------------
    phase_reviews: list = []
    try:
        from .evaluator import evaluate_phase
        from .phases import slice_player_phases

        phase_stats_list = slice_player_phases(match, player)
        if phase_stats_list:
            await _emit("phases", 0, len(phase_stats_list))

            async def _run_phase(idx: int, ps):
                review = await evaluate_phase(
                    hero_name=hero_name,
                    position_label=position_label_text,
                    team=player.team,
                    phase_key=ps.phase,
                    phase_time_range=ps.time_range,
                    phase_stats_dict=ps.to_dict(),
                    lang=lang,
                )
                return idx, review

            tasks = [
                asyncio.create_task(_run_phase(i, ps))
                for i, ps in enumerate(phase_stats_list)
            ]
            indexed: list[tuple[int, object]] = []
            done = 0
            for fut in asyncio.as_completed(tasks):
                idx, review = await fut
                indexed.append((idx, review))
                done += 1
                await _emit("phases", done, len(phase_stats_list))
            indexed.sort(key=lambda t: t[0])
            phase_reviews = [r for _, r in indexed]
    except Exception as exc:
        logger.error("phase review pipeline failed: %s", exc, exc_info=True)
        phase_reviews = []

    # ------------------------------------------------------------------
    # Step 5: Compile report
    # ------------------------------------------------------------------
    await _emit("summary", 0, 1)

    # Separate death analyses from key moments
    death_analyses = [
        e for e in evaluations if e.decision_point.point_type == "death"
    ]
    key_moments = [
        e for e in evaluations if e.decision_point.point_type != "death"
    ]

    # Calculate overall grade (weight deaths 1.5x)
    overall_grade = _calculate_overall_grade(evaluations)

    # Laning review
    laning_review = _generate_laning_review(match, player, evaluations, lang=lang)

    # Improvement areas and strengths
    improvement_areas = _detect_improvement_areas(evaluations, lang=lang)
    strengths = _detect_strengths(evaluations, lang=lang)

    # Generate summary
    summary = _generate_summary(
        player, overall_grade, evaluations, improvement_areas, strengths, match,
        lang=lang,
    )

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
        phase_reviews=phase_reviews,
        position=position_payload,
    )
    await _emit("done", 1, 1)
    return report


async def _safe_evaluate(
    context: DecisionContext, knowledge: str, lang: str = "zh",
) -> tuple[Evaluation, bool]:
    """Evaluate a decision, catching any exceptions so one failure
    does not block the entire pipeline.

    Returns (Evaluation, used_api) where used_api is True if the LLM
    returned a successfully parsed evaluation.
    """
    try:
        ev = await evaluate_decision(context, knowledge, lang=lang)
        # Fill missing structured fields from rule-based fallback.
        # Claude often writes excellent free-form analysis but doesn't
        # always follow the exact structured format.
        missing_fields = (not ev.mistake) or (not ev.better_play) or (not ev.lesson)
        if missing_fields:
            fallback = evaluate_decision_fallback(context, lang=lang)
            if not ev.mistake and fallback.mistake:
                ev.mistake = fallback.mistake
            if not ev.better_play and fallback.better_play:
                ev.better_play = fallback.better_play
            if not ev.lesson and fallback.lesson:
                ev.lesson = fallback.lesson
        return ev, True
    except Exception as exc:
        logger.error(
            "evaluate_decision 异常 (time=%s): %s",
            context.decision_point.time,
            exc,
        )
        try:
            return evaluate_decision_fallback(context, lang=lang), False
        except Exception:
            error_msg = (
                "An error occurred during evaluation."
                if lang == "en"
                else "评估过程中出现错误，无法完成分析。"
            )
            return Evaluation(
                decision_point=context.decision_point,
                score="C",
                analysis=error_msg,
                mistake="",
                better_play="",
                lesson="",
            ), False


def _calculate_overall_grade(evaluations: list[Evaluation]) -> str:
    """Calculate the weighted overall grade from individual evaluations.

    Deaths are weighted 1.5x more heavily than other decision points.
    """
    if not evaluations:
        return "B"  # Default when no evaluations

    total_weight = 0.0
    weighted_sum = 0.0

    for ev in evaluations:
        score_val = _SCORE_TO_NUM.get(ev.score, 2.0)
        weight = 1.5 if ev.decision_point.point_type == "death" else 1.0
        weighted_sum += score_val * weight
        total_weight += weight

    avg = weighted_sum / total_weight if total_weight > 0 else 3.0
    return _letter_grade(avg)
