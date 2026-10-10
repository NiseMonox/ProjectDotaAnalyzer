"""Main orchestrator for the AI coaching pipeline.

Pipeline:
1. Derive positions, detect decision points (deaths + key moments)
2. Build death causal chains and a context for each decision point
3. Retrieve knowledge for each context: the full block for the points an LLM
   will judge, the best entries as ``sources`` for the others
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
    RULE_SOURCES,
    Evaluation,
    PhaseReview,
    is_time_unknown,
    normalize_grade,
    sources_payload,
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

__all__ = ["REPORT_SCHEMA_VERSION", "MIN_PARTIAL_GAME_SECONDS", "NoGameError", "generate_coaching_report",
           "has_no_game"]


class NoGameError(ValueError):
    """The match data holds no game to review (a replay cut before or just after the horn)."""


# A replay that ends this early in the game holds nothing to review: no lane
# has played out and the benchmarks have nothing to compare (a recording cut
# 4 s and one cut 133 s after the horn both got "KDA 0/0/0 after 0 minutes,
# overall grade B: ... so the grade is only a default"). routes.py refuses the
# same matches before a job is started.
MIN_PARTIAL_GAME_SECONDS = 300


def has_no_game(match: MatchData) -> bool:
    """Whether a truncated replay ends too early for a coaching report."""
    return bool(match.partial) and match.duration < MIN_PARTIAL_GAME_SECONDS


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
        "laning": "注意换血时机和消耗品使用。",
        "laning_cs": "练习补刀，特别是塔下补刀。",
        "teamfight": "注意开团时机和站位。优先集火关键目标，注意保护己方核心。",
        "resources": "提高打野效率，减少游走等待时间。确保每分钟都有金币收入。",
        "vision": "养成买眼插眼的习惯，特别是团战前和烟雾时。",
    },
    "en": {
        "map_awareness": "Check the minimap more frequently — aim for every 5 seconds. Watch for missing enemy heroes.",
        "laning": "Mind your trade timing and consumable usage.",
        "laning_cs": "Practice last-hitting, especially under tower.",
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

# Used instead of the grade comment when no decision could be analysed.
_NO_DECISIONS_COMMENT: dict[str, str] = {
    "zh": "比赛数据中没有可供分析的具体决策点，评分仅为默认值。",
    "en": "No individual decision points were found in the match data, so the grade is only a default.",
}

_POSITION_FALLBACK_LABEL: dict[str, str] = {"zh": "未知位置", "en": "Unknown position"}


def _hero_text(name: str, lang: str) -> str:
    """A hero name as written in text (never a raw code such as "hero_150")."""
    try:
        from .data_manager import hero_display_name
        return hero_display_name(name, lang) or name
    except Exception:  # pragma: no cover - the helper never raises
        return name


def _clock(seconds: int) -> str:
    """Game time as m:ss ("-0:10" before the horn)."""
    sign = "-" if seconds < 0 else ""
    seconds = abs(int(seconds))
    return f"{sign}{seconds // 60}:{seconds % 60:02d}"


# ---------------------------------------------------------------------------
# Laning review
# ---------------------------------------------------------------------------

# Lane opponents found from snapshot positions must stand this close (world units).
_LANE_OPPONENT_RADIUS = 3000.0
_TEN_MIN = 10
# The lane is won / lost when its heroes together lead / trail the enemy
# heroes of the same lane by this share of gold + XP at 10:00.
_LANE_LEAD_SHARE = 0.10
# Lane creeps a core can practically last-hit in the first 10 minutes.
_LANE_CREEPS_10MIN = 82
_LANE_CREEP_TYPES = ("melee", "ranged", "siege", "unknown")  # "unknown" is a lane creep the parser could not type


def _at_ten(series: list[int]) -> int | None:
    """Value of a per-minute series at 10:00 (None when the series ends earlier)."""
    return series[_TEN_MIN] if len(series) > _TEN_MIN else None


def _snapshot_at_ten(p: PlayerData):
    """The laning snapshot at 10:00 (within 10 s), None when there is none."""
    if not p.laning_snapshots:
        return None
    snap = min(p.laning_snapshots, key=lambda s: abs(s.time - _TEN_MIN * 60))
    return snap if abs(snap.time - _TEN_MIN * 60) <= 10 else None


def _cs_10(p: PlayerData) -> int | None:
    """Last hits at 10:00 (all creeps): the definition the benchmark comparison uses."""
    value = _at_ten(p.lh_t)
    if value is None:
        snap = _snapshot_at_ten(p)
        value = snap.last_hits if snap is not None else None
    return value


def _denies_10(p: PlayerData) -> int | None:
    value = _at_ten(p.dn_t)
    if value is None:
        snap = _snapshot_at_ten(p)
        value = snap.denies if snap is not None else None
    return value


def _lane_creep_cs_10(p: PlayerData) -> int | None:
    """Lane creeps last-hit before 10:00 (None when the source has no creep log)."""
    if not p.laning_creep_kills:
        return None
    return sum(
        1 for ck in p.laning_creep_kills
        if ck.creep_type in _LANE_CREEP_TYPES and ck.time < _TEN_MIN * 60
    )


def _lane_resources_10(players: list[PlayerData]) -> tuple[int, int] | None:
    """(gold, XP) the heroes had earned together by 10:00, None when unknown."""
    gold = xp = 0
    for p in players:
        g, x = _at_ten(p.gold_t), _at_ten(p.xp_t)
        if g is None or x is None:
            return None
        gold += g
        xp += x
    return gold, xp


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


def _lane_outcome(match: MatchData, player: PlayerData) -> tuple[str | None, int, int]:
    """(result, gold lead, XP lead) of the player's lane at 10:00.

    The lane is every hero with the player's absolute lane number; the two
    sides are compared on the gold and XP they earned together, so lane
    partners always get the same result. ``result`` is "won" / "even" /
    "lost", or None when it cannot be judged: the player had no lane
    (roaming, unknown), the two sides had different numbers of heroes in the
    lane (their totals are not comparable), or the 10-minute values are
    missing.
    """
    if player.lane not in (1, 2, 3):
        return None, 0, 0
    ours = [p for p in match.players if p.team == player.team and p.lane == player.lane]
    theirs = [p for p in match.players if p.team != player.team and p.lane == player.lane]
    if len(ours) != len(theirs):
        return None, 0, 0
    mine, enemy = _lane_resources_10(ours), _lane_resources_10(theirs)
    if mine is None or enemy is None or sum(enemy) <= 0:
        return None, 0, 0
    gold_lead, xp_lead = mine[0] - enemy[0], mine[1] - enemy[1]
    share = (gold_lead + xp_lead) / sum(enemy)
    if share >= _LANE_LEAD_SHARE:
        return "won", gold_lead, xp_lead
    if share <= -_LANE_LEAD_SHARE:
        return "lost", gold_lead, xp_lead
    return "even", gold_lead, xp_lead


def _generate_laning_review(
    match: MatchData, player: PlayerData, evaluations: list[Evaluation],
    lang: str = "zh", positions: dict | None = None,
) -> dict[str, Any]:
    """Laning phase review for the player, from 10-minute data only.

    Returns a dict with cs_10min (last hits at 10:00, the same number the
    benchmark comparison shows), denies_10min, lane_cs_10min, lane_matchup,
    lane_result ("won" / "even" / "lost" / None), lane_gold_diff,
    lane_xp_diff, key_trades, cs_efficiency and summary. A value the source
    does not have is None, never 0; without any 10-minute value (also a
    game that ended before 10:00) the review is empty ({}). A position 4 / 5
    is not judged on last hits.
    """
    from .positions import lane_role_of
    from .pro_patterns import benchmarks_apply

    positions = positions or {}
    en = lang == "en"
    cs_10min = _cs_10(player)
    denies_10min = _denies_10(player)
    if cs_10min is None or match.duration < _TEN_MIN * 60:
        return {}

    lane_map = _LANE_NAMES.get(lang, _LANE_NAMES["zh"])
    player_lane_name = lane_map.get(lane_role_of(player), _UNKNOWN_LANE.get(lang, _UNKNOWN_LANE["zh"]))

    hero = _hero_text(player.hero_name, lang)
    opponents = _lane_opponents(match, player)
    unknown_opp = _UNKNOWN_OPPONENT.get(lang, _UNKNOWN_OPPONENT["zh"])
    separator = ", " if en else "、"
    opponent_names = (separator.join(_hero_text(p.hero_name, lang) for p in opponents)
                      if opponents else unknown_opp)
    lane_matchup = f"{hero} ({player_lane_name}) vs {opponent_names}"

    lane_result, gold_lead, xp_lead = _lane_outcome(match, player)

    my_pos = positions.get(player.player_slot)
    position_number = my_pos.number if my_pos is not None else 0
    is_support = position_number in (4, 5)

    key_trades: list[str] = []
    if player.laning_damage_events:
        total_dmg_dealt = sum(e.damage for e in player.laning_damage_events)
        key_trades.append(
            f"Dealt {total_dmg_dealt} hero damage during the laning phase" if en
            else f"对线阶段造成 {total_dmg_dealt} 点英雄伤害"
        )
    score_label = "Score" if en else "评分"
    laning_evals = [
        e for e in evaluations
        if e.decision_point.time < _TEN_MIN * 60 and not is_time_unknown(e.decision_point)
    ]
    for ev in sorted(laning_evals, key=lambda e: e.decision_point.time)[:3]:
        key_trades.append(
            f"[{_clock(ev.decision_point.time)}] {ev.decision_point.brief} ({score_label}: {ev.score})")

    # CS efficiency: lane creeps last-hit / lane creeps a core can get. Only
    # for cores, only when the source logged which creeps were killed, and
    # only in a game mode the 10-minute creep count describes (not Turbo).
    lane_cs = _lane_creep_cs_10(player)
    cs_efficiency = ""
    efficiency_pct: float | None = None
    if lane_cs is not None and not is_support and benchmarks_apply(match.game_mode):
        efficiency_pct = lane_cs / _LANE_CREEPS_10MIN * 100
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

    # The direct counterpart of a core (pos 1 vs 3, 2 vs 2), for the CS line.
    counterpart: PlayerData | None = None
    if not is_support and position_number:
        mirror = {1: 3, 3: 1, 2: 2}.get(position_number)
        counterpart = next(
            (p for p in opponents
             if positions.get(p.player_slot) is not None and positions[p.player_slot].number == mirror),
            None,
        )
    counterpart_cs = _cs_10(counterpart) if counterpart is not None else None

    if en:
        parts: list[str] = []
        lead = f"{gold_lead:+,} gold and {xp_lead:+,} XP for the lane as a whole"
        if lane_result == "won":
            parts.append(f"{hero}'s lane ({player_lane_name}, against {opponent_names}) "
                         f"was ahead at 10:00: {lead}.")
        elif lane_result == "lost":
            parts.append(f"{hero}'s lane ({player_lane_name}, against {opponent_names}) "
                         f"was behind at 10:00: {lead}.")
        elif lane_result == "even":
            parts.append(f"{hero}'s lane ({player_lane_name}, against {opponent_names}) "
                         f"was about even at 10:00: {lead}.")
        else:
            parts.append(f"{hero} ({player_lane_name}): the lane result cannot be judged (no fixed lane, "
                         f"uneven numbers in the lane, or no 10-minute data).")
        numbers = f"{cs_10min} last hit{'' if cs_10min == 1 else 's'}"
        if denies_10min is not None:
            numbers += f" and {denies_10min} den{'y' if denies_10min == 1 else 'ies'}"
        numbers += " at 10:00"
        if counterpart is not None and counterpart_cs is not None:
            numbers += f" ({_hero_text(counterpart.hero_name, lang)}: {counterpart_cs})"
        if efficiency_pct is not None:
            numbers += f"; {lane_cs} of about {_LANE_CREEPS_10MIN} lane creeps ({efficiency_pct:.0f}%)"
        parts.append(numbers + ".")
        if is_support:
            parts.append(f"A position {position_number} is not measured by last hits.")
        summary = " ".join(parts)
    else:
        parts = []
        lead = f"全路合计经济{gold_lead:+,}、经验{xp_lead:+,}"
        if lane_result == "won":
            parts.append(f"{hero}所在的{player_lane_name}（对手：{opponent_names}）在10分钟时领先：{lead}。")
        elif lane_result == "lost":
            parts.append(f"{hero}所在的{player_lane_name}（对手：{opponent_names}）在10分钟时落后：{lead}。")
        elif lane_result == "even":
            parts.append(f"{hero}所在的{player_lane_name}（对手：{opponent_names}）在10分钟时基本持平：{lead}。")
        else:
            parts.append(f"{hero}（{player_lane_name}）：无法判断对线结果（没有固定对线、线上人数不对等或缺少10分钟数据）。")
        numbers = f"10分钟补刀{cs_10min}个"
        if denies_10min is not None:
            numbers += f"，反补{denies_10min}个"
        if counterpart is not None and counterpart_cs is not None:
            numbers += f"（{_hero_text(counterpart.hero_name, lang)}补刀{counterpart_cs}个）"
        if efficiency_pct is not None:
            numbers += f"；线上小兵约{_LANE_CREEPS_10MIN}个中补到{lane_cs}个（{efficiency_pct:.0f}%）"
        parts.append(numbers + "。")
        if is_support:
            parts.append(f"{position_number}号位不以补刀数衡量。")
        summary = "".join(parts)

    return {
        "cs_10min": cs_10min,
        "denies_10min": denies_10min,
        "lane_cs_10min": lane_cs,
        "lane_matchup": lane_matchup,
        "lane_result": lane_result,
        "lane_gold_diff": gold_lead if lane_result is not None else None,
        "lane_xp_diff": xp_lead if lane_result is not None else None,
        "key_trades": key_trades,
        "cs_efficiency": cs_efficiency,
        "summary": summary,
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
_MAX_EXAMPLES = 3
_MIN_OCCURRENCES = 2  # one decision is not a recurring strength or weakness


def _category_of(ev: Evaluation) -> str:
    dp = ev.decision_point
    if dp.point_type == "death" and dp.related_death is not None and dp.related_death.is_teamfight:
        return "teamfight"  # a teamfight death is not a map-awareness problem
    return _CATEGORY_MAP.get(dp.point_type, "resources")


def _example(ev: Evaluation) -> str:
    """One decision as "[m:ss] brief" (no time when it is unknown)."""
    dp = ev.decision_point
    brief = (dp.brief or "").strip().rstrip(".。")
    return brief if is_time_unknown(dp) else f"[{_clock(dp.time)}] {brief}"


# What an evaluation writes into ``mistake`` when there is none.
_NO_MISTAKE = ("none", "n/a", "无", "没有")


def _names_mistake(ev: Evaluation) -> bool:
    """Whether the evaluation says what went wrong.

    The rules leave ``mistake`` empty when the data shows none (e.g. a death
    in a match without positions); such a decision supports no diagnosis.
    """
    text = (ev.mistake or "").strip().rstrip(".。").lower()
    return bool(text) and text not in _NO_MISTAKE


def _area_description(count: int, grade: str, examples: list[str], closing: str, lang: str) -> str:
    """Full sentences for one area: how many decisions, examples, advice.

    Built from the decision briefs (short, rule-made, in the report
    language), never from the free text of the evaluations.
    """
    if lang == "en":
        noun = "decision" if count == 1 else "decisions"
        text = f"{count} {noun} in this area, average grade {grade}."
        if examples:
            text += f" Examples: {'; '.join(examples)}."
        return f"{text} {closing}".strip()
    text = f"该方面共有{count}个决策点，平均评分{grade}。"
    if examples:
        text += f"例如：{'；'.join(examples)}。"
    return text + closing


def _detect_improvement_areas(
    evaluations: list[Evaluation], lang: str = "zh", lane_result: str | None = None,
) -> list[dict[str, Any]]:
    """Recurring weaknesses: categories averaging below B.

    A weakness needs at least ``_MIN_OCCURRENCES`` decisions graded below B
    whose evaluation names a mistake: deaths nothing is known about (a match
    without positions) do not make "map awareness" a problem. Laning is not
    listed for a player whose lane was won: a few lost trades are part of
    every lane. Its advice follows what was observed: last-hitting only with
    missed-CS points (which positions 4 and 5 never get), trading only with
    lost trades.
    """
    by_category: dict[str, list[Evaluation]] = {}
    for ev in evaluations:
        by_category.setdefault(_category_of(ev), []).append(ev)

    cat_labels = _CATEGORY_LABELS.get(lang, _CATEGORY_LABELS["zh"])
    suggestions = _IMPROVEMENT_SUGGESTIONS.get(lang, _IMPROVEMENT_SUGGESTIONS["zh"])
    default_suggestion = _DEFAULT_IMPROVEMENT.get(lang, _DEFAULT_IMPROVEMENT["zh"])

    areas: list[dict[str, Any]] = []
    for cat, evs in by_category.items():
        found = [
            e for e in evs
            if _SCORE_TO_NUM.get(normalize_grade(e.score), 2.0) < 3.0 and _names_mistake(e)
        ]
        if len(found) < _MIN_OCCURRENCES or (cat == "laning" and lane_result == "won"):
            continue
        avg = sum(_SCORE_TO_NUM.get(normalize_grade(e.score), 2.0) for e in evs) / len(evs)
        if avg >= 3.0:
            continue
        weakest = sorted(
            found,
            key=lambda e: (_SCORE_TO_NUM.get(normalize_grade(e.score), 2.0), _chronological(e)),
        )[:_MAX_EXAMPLES]
        examples = [_example(e) for e in sorted(weakest, key=_chronological)]
        grade = _letter_grade(avg)
        advice = suggestions.get(cat, default_suggestion)
        if cat == "laning":
            # Advice only for what was actually observed in this lane.
            kinds = {e.decision_point.point_type for e in evs}
            advice = (" " if lang == "en" else "").join(
                suggestions[key] for key, kind in (("laning_cs", "missed_cs"), ("laning", "bad_trade"))
                if kind in kinds
            )
        areas.append({
            "category": cat_labels.get(cat, cat),
            "category_code": cat,
            "avg_score": grade,
            "occurrences": len(evs),
            "examples": examples,
            "description": _area_description(len(evs), grade, examples, advice, lang),
        })
    # Worst grade first; among equal grades the area with more decisions.
    areas.sort(key=lambda a: (_SCORE_TO_NUM.get(a["avg_score"], 2.0), -a["occurrences"]))
    return areas


def _detect_strengths(evaluations: list[Evaluation], lang: str = "zh") -> list[dict[str, Any]]:
    """Recurring strengths: categories averaging A or better."""
    by_category: dict[str, list[Evaluation]] = {}
    for ev in evaluations:
        by_category.setdefault(_category_of(ev), []).append(ev)

    cat_labels = _CATEGORY_LABELS.get(lang, _CATEGORY_LABELS["zh"])

    strengths: list[dict[str, Any]] = []
    for cat, evs in by_category.items():
        if len(evs) < _MIN_OCCURRENCES:
            continue
        avg = sum(_SCORE_TO_NUM.get(normalize_grade(e.score), 2.0) for e in evs) / len(evs)
        if avg < 3.5:
            continue
        best = [e for e in evs if normalize_grade(e.score) in ("S", "A")]
        examples = [_example(e) for e in sorted(best, key=_chronological)[:_MAX_EXAMPLES]]
        grade = _letter_grade(avg)
        strengths.append({
            "category": cat_labels.get(cat, cat),
            "category_code": cat,
            "avg_score": grade,
            "occurrences": len(evs),
            "examples": examples,
            "description": _area_description(len(evs), grade, examples, "", lang),
        })
    strengths.sort(key=lambda s: (-_SCORE_TO_NUM.get(s["avg_score"], 2.0), -s["occurrences"]))
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
    note: str = "",
) -> str:
    """Text summary of the coaching report.

    The result of the match is only stated when the source knows the winner
    (a truncated replay does not). ``note`` is an extra sentence in the
    report language (e.g. why a Turbo match has no benchmark comparison).
    """
    kda = f"{player.kills}/{player.deaths}/{player.assists}"
    hero = _hero_text(player.hero_name, lang)
    duration_min = match.duration // 60
    grade_comments = _GRADE_COMMENTS.get(lang, _GRADE_COMMENTS["zh"])
    if not evaluations:
        grade_comments = {overall_grade: _NO_DECISIONS_COMMENT.get(lang, _NO_DECISIONS_COMMENT["zh"])}
    death_count = sum(1 for e in evaluations if e.decision_point.point_type == "death")
    unlocated = max(0, player.deaths - death_count)
    won = (player.team == "radiant") == bool(match.radiant_win)

    if lang == "en":
        if match.winner_known:
            opening = (f"{hero} {'won' if won else 'lost'} this match with a KDA of {kda} "
                       f"in {duration_min} minutes. ")
        else:
            opening = (f"{hero} finished with a KDA of {kda} after {duration_min} minutes "
                       f"(the replay does not contain the result of the match). ")
        parts = [
            opening,
            f"Overall grade {overall_grade}: {grade_comments.get(overall_grade, '')} ",
        ]
        if strengths:
            parts.append(f"Key strength: {strengths[0]['category']}. ")
        if improvement_areas:
            parts.append(
                f"Top area to improve: {improvement_areas[0]['category']} "
                f"({improvement_areas[0]['occurrences']} decisions reviewed). "
            )
        if death_count:
            parts.append(f"{death_count} death{'' if death_count == 1 else 's'} analyzed — "
                         f"review the cause of each death carefully. ")
        if unlocated:
            parts.append(
                f"{unlocated} more death{' is' if unlocated == 1 else 's are'} on the scoreboard "
                f"but not in the match data, so {'it is' if unlocated == 1 else 'they are'} not analyzed. "
            )
        if note:
            parts.append(note)
    else:
        if match.winner_known:
            opening = f"本场比赛{hero}{'获胜' if won else '失败'}，KDA为{kda}，比赛时长{duration_min}分钟。"
        else:
            opening = f"本场比赛{hero}的KDA为{kda}，比赛时长{duration_min}分钟（录像未记录胜负）。"
        parts = [
            opening,
            f"综合评分{overall_grade}：{grade_comments.get(overall_grade, '')}",
        ]
        if strengths:
            parts.append(f"主要优势是{strengths[0]['category']}方面的表现。")
        if improvement_areas:
            parts.append(
                f"最需要改进的是{improvement_areas[0]['category']}"
                f"（涉及{improvement_areas[0]['occurrences']}个决策点）。"
            )
        if death_count:
            parts.append(f"本场共分析了{death_count}次死亡，请重点关注每次死亡的原因。")
        if unlocated:
            parts.append(f"另有{unlocated}次死亡只出现在计分板上、比赛数据中没有记录，因此未作分析。")
        if note:
            parts.append(note)
    return "".join(parts).strip()


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


def _positions_by_team(match: MatchData, positions: dict | None) -> dict[str, list[list]]:
    """``{"radiant": [[hero name, position number], ...], "dire": [...]}`` for the prompts.

    Only heroes whose position (1-5) could be derived, in position order.
    """
    out: dict[str, list[list]] = {"radiant": [], "dire": []}
    for p in match.players:
        pos = (positions or {}).get(p.player_slot)
        number = getattr(pos, "number", 0)
        if p.team in out and number in (1, 2, 3, 4, 5):
            out[p.team].append([p.hero_name, number])
    for team in out:
        out[team].sort(key=lambda entry: entry[1])
    return out


def _context_step(
    match: MatchData,
    decisions: list[DecisionPoint],
    causal_chains: dict[int, CausalChain],
    position: int | None,
    lang: str,
    knowledge_for: set[int] | None = None,
    positions: dict | None = None,
) -> list[tuple[DecisionContext, str, Evaluation]]:
    """Build (context, knowledge, rule-based evaluation) per decision point.

    ``positions`` (slot -> derived Position of every hero) lets a prompt name
    the role each hero really played instead of the one a hero note calls
    usual. The knowledge block is only assembled for the decision indices in
    ``knowledge_for`` (the ones going to the LLM); rules do not read it.
    Every rule evaluation still gets ``sources``: the best entries of that
    block, or for the other points the best entries the knowledge base has
    on the situation. They are what the report lists when no LLM verdict
    replaces the rule evaluation.

    The result is a list with one more attribute, ``knowledge_info``: the
    report-level ``knowledge`` object (:func:`_knowledge_info`), taken here,
    in the thread and at the moment the sources were retrieved.
    """
    out = _Prepared()
    try:
        from .context_builder import build_context
    except Exception as exc:
        logger.error("Context builder unavailable: %s", exc)
        return out
    get_knowledge_for_context = related_refs = None
    try:
        from .knowledge_base import get_knowledge_for_context, related_refs
    except Exception as exc:
        logger.warning("Knowledge base unavailable: %s", exc)
    match_positions = _positions_by_team(match, positions)

    retrieved: list = []    # what carries a ``retrieval`` record: blocks and related-entry lists
    for i, dp in enumerate(decisions):
        try:
            chain = None
            if dp.point_type == "death" and dp.related_death is not None:
                chain = causal_chains.get(dp.related_death.time)
            ctx = build_context(match, dp, causal_chain=chain, lang=lang)
            if position:
                # Position 1-5 lets knowledge retrieval add a role tag.
                ctx.match_context["player_position"] = position
            if match_positions["radiant"] or match_positions["dire"]:
                ctx.match_context["positions"] = match_positions
        except Exception as exc:
            logger.error("Context build failed (t=%s): %s", dp.time, exc, exc_info=True)
            continue
        knowledge = ""
        refs: list = []
        try:
            if knowledge_for and i in knowledge_for and get_knowledge_for_context is not None:
                # The game mode travels in ctx.match_context (a Turbo block has no benchmarks).
                knowledge = get_knowledge_for_context(ctx, lang=lang) or ""
                # Read before anything turns the block into a plain str.
                refs = list(getattr(knowledge, "refs", None) or ())
                retrieved.append(knowledge)
            elif related_refs is not None:
                related = related_refs(ctx, lang=lang, k=RULE_SOURCES)
                refs = list(related or ())
                retrieved.append(related)
        except Exception as exc:
            logger.warning("Knowledge retrieval failed (t=%s): %s", dp.time, exc)
        try:
            rules = evaluate_decision_fallback(ctx, lang=lang)
        except Exception as exc:
            logger.error("Rule evaluation failed (t=%s): %s", dp.time, exc, exc_info=True)
            rules = _error_evaluation(ctx, lang)
        try:
            rules.sources = sources_payload(refs, limit=RULE_SOURCES)
        except Exception as exc:  # a test double without refs semantics
            logger.debug("Knowledge sources unavailable (t=%s): %s", dp.time, exc)
        out.append((ctx, knowledge, rules))
    out.knowledge_info = _knowledge_info(retrieved)
    return out


class _Prepared(list):
    """Result of :func:`_context_step`: the prepared decisions plus ``knowledge_info``."""

    knowledge_info: dict | None = None


def _knowledge_info(blocks: list) -> dict:
    """The report-level ``knowledge`` object: how the sources were retrieved.

    ``mode`` is what the sources of this report were ranked with ("hybrid" =
    tags, keywords and embedding vectors; "lexical" = tags and keywords
    only), read from the ``retrieval`` records of *blocks* (knowledge blocks
    and related-entry lists); a report without any says what the retrieval
    stack would use right now. ``embedder`` and ``vectors`` describe the
    vector space and are None / 0 when it was not used; ``chunks`` is the
    size of the knowledge base.

    The numbers describe the store this thread holds (built here when no
    retrieval has built it yet): the cached stores are reset after every
    learning run and game-data update, and a report read at that moment —
    or one without decisions in a fresh process — said "lexical, 0 chunks"
    under sources that had been ranked with vectors.
    """
    info: dict[str, Any] = {"mode": "lexical", "embedder": None, "chunks": 0, "vectors": 0}
    try:
        from . import knowledge_base, rag
        stores = knowledge_base.warm_stores()
        store = stores.get("en") or next((s for s in stores.values() if s is not None), None)
        status = rag.status(store)
        info["mode"] = "hybrid" if status.get("mode") == "hybrid" else "lexical"
        info["embedder"] = status.get("embedder") or None
        info["chunks"] = int(status.get("chunks") or 0)
        info["vectors"] = int(status.get("vectors") or 0)
    except Exception as exc:
        logger.debug("Retrieval status unavailable: %s", exc)
    modes = {(getattr(block, "retrieval", None) or {}).get("mode") for block in blocks} - {None}
    if modes:
        info["mode"] = "hybrid" if "hybrid" in modes else "lexical"
    if info["mode"] != "hybrid":
        info["embedder"], info["vectors"] = None, 0
    return info


def _compare_step(match: MatchData, player: PlayerData, lang: str, position: int | None) -> dict:
    try:
        from .pro_patterns import compare_player, format_pro_comparison
        return format_pro_comparison(compare_player(player, match, lang=lang, position=position))
    except Exception as exc:
        logger.error("Benchmark comparison failed: %s", exc, exc_info=True)
        return {}


def _phase_stats_step(match: MatchData, player: PlayerData, deaths: list[DeathEvent], lang: str) -> list:
    try:
        from .phases import slice_player_phases
        return slice_player_phases(match, player, death_times=[d.time for d in deaths], lang=lang)
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


def _llm_slots(points: list[DecisionPoint]) -> list[int]:
    """Indices of the decision points that go to the LLM (at most MAX_LLM_EVALS).

    Deaths come first, then the other points by severity. A player who died
    more often than there are slots gets them spread evenly over his deaths
    (first and last included) instead of the earliest ones, so the late game
    is covered as well.
    """
    order = sorted(range(len(points)), key=lambda i: _llm_priority(points[i]))
    deaths = [i for i in order if points[i].point_type == "death"]
    if len(deaths) > MAX_LLM_EVALS > 1:
        step = (len(deaths) - 1) / (MAX_LLM_EVALS - 1)
        return [deaths[round(k * step)] for k in range(MAX_LLM_EVALS)]
    return order[:MAX_LLM_EVALS]


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
        sources=list(rules.sources),
    )
    ev.degraded = degraded
    return ev


def _llm_verdict(ev: Evaluation) -> Evaluation:
    """Mark an evaluation as an LLM verdict.

    Fields the model left empty stay empty: the rule evaluation was written
    independently and can contradict the LLM's analysis (the frontend hides
    empty fields).
    """
    ev.score = normalize_grade(ev.score)
    ev.used_llm = True
    ev.degraded = False
    return ev


def _error_text(exc: BaseException) -> str:
    """Log text for an exception (a timeout has an empty message)."""
    return str(exc) or type(exc).__name__


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
            return _llm_verdict(ev)
        except asyncio.CancelledError:
            raise
        except AgentUnsupportedError as exc:
            # Also raised when too little time is left for an investigation.
            logger.info("Agent not used (%s); using single-prompt evaluation (t=%s)", _error_text(exc), t)
        except Exception as exc:
            logger.warning("Decision agent failed (t=%s), using rules: %s", t, _error_text(exc))
            return _rules_copy(rules, degraded=True)

    try:
        ev = await evaluate_decision(context, knowledge, lang=lang)
        return _llm_verdict(ev)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Single-prompt evaluation failed (t=%s), using rules: %s", t, _error_text(exc))
        return _rules_copy(rules, degraded=True)


_DEATH_WEIGHT = 1.5


def _calculate_overall_grade(evaluations: list[Evaluation], unlocated_deaths: int = 0) -> str:
    """Weighted overall grade (deaths count 1.5x); always a valid grade.

    ``unlocated_deaths`` are deaths on the scoreboard that the match data
    does not contain. Each counts like an analysed death graded C, so a
    report with missing deaths is not graded as if they had not happened.
    """
    unlocated_deaths = max(0, unlocated_deaths)
    if not evaluations and not unlocated_deaths:
        return "B"
    total_weight = _DEATH_WEIGHT * unlocated_deaths
    weighted_sum = _SCORE_TO_NUM["C"] * total_weight
    for ev in evaluations:
        weight = _DEATH_WEIGHT if ev.decision_point.point_type == "death" else 1.0
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
    NoGameError
        (a ``ValueError``) when the replay was cut before the game started
        or within its first minutes (:func:`has_no_game`).
    """
    lang = lang if lang in ("zh", "en") else "zh"
    player = _find_player(match, player_slot)
    if player is None:
        raise ValueError(f"player_slot {player_slot} is not a player of match {match.match_id}")
    if has_no_game(match):
        # A replay cut before or just after the horn: a report on it would
        # grade a game that was never played ("KDA 0/0/0 in 0 minutes, grade B").
        raise NoGameError(f"match {match.match_id} ends {max(0, match.duration)} s into the game; nothing to review")

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
    llm_decisions = set(_llm_slots(decisions)) if llm_on else set()
    await _emit("context", 0, max(1, len(decisions)))
    prepared = await asyncio.to_thread(
        _context_step, match, decisions, causal_chains, position_number, lang, llm_decisions, positions_map,
    )
    # Contexts whose build failed are skipped, so re-derive the LLM indices.
    llm_indices = _llm_slots([item[0].decision_point for item in prepared]) if llm_on else []
    evaluations_by_index: dict[int, Evaluation] = {
        i: _rules_copy(prepared[i][2], degraded=False) for i in range(len(prepared))
    }

    phase_stats_list = await asyncio.to_thread(_phase_stats_step, match, player, deaths, lang)
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
        # The window is known here, whatever built the review.
        review.time_range, review.start_min, review.end_min = ps.time_range, ps.start_min, ps.end_min
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
    overall_grade = _calculate_overall_grade(evaluations, unlocated_deaths=player.deaths - len(death_analyses))
    laning_review = _generate_laning_review(match, player, evaluations, lang=lang, positions=positions_map)
    improvement_areas = _detect_improvement_areas(
        evaluations, lang=lang, lane_result=laning_review.get("lane_result"))
    strengths = _detect_strengths(evaluations, lang=lang)
    # A game mode without benchmarks has no comparison card, a replay that
    # ends early only part of one; say so here.
    limited = pro_comparison_data.get("applicable") is False or pro_comparison_data.get("partial") is True
    mode_note = pro_comparison_data.get("summary", "") if limited else ""
    summary = _generate_summary(
        player, overall_grade, evaluations, improvement_areas, strengths, match, lang=lang, note=mode_note)

    llm_used = any(e.used_llm for e in evaluations) or any(p.used_llm for p in final_phases)
    degraded = llm_on and (any(e.degraded for e in evaluations) or any(p.degraded for p in final_phases))
    knowledge_info = getattr(prepared, "knowledge_info", None)
    if knowledge_info is None:      # a test double of the context step
        knowledge_info = await asyncio.to_thread(_knowledge_info, [item[1] for item in prepared])

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
        knowledge=knowledge_info,
        report_version=REPORT_SCHEMA_VERSION,
    )
    logger.info(
        "AI coach report ready: match=%s slot=%s decisions=%d llm_evals=%d llm_used=%s degraded=%s "
        "knowledge=%s sources=%d cited=%d (%.1fs)",
        match.match_id, player_slot, len(evaluations), len(llm_indices), llm_used, degraded,
        knowledge_info["mode"], sum(len(e.sources) for e in evaluations),
        sum(1 for e in evaluations for src in e.sources if src.get("cited")),
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
