"""Professional-level benchmark data and pattern comparison engine.

Provides Immortal-rank (~7000+ MMR) benchmark data for CS, GPM, deaths,
damage, item timings, ward placement, and kill participation. Compares
individual player performance against these benchmarks and produces
language-aware reports.

Self-updating values: the learning pipeline (``learning/stats_learner.py``)
writes per-hero OpenDota percentiles to ``backend/data/learned/
_benchmarks_raw.json``. When that file is present, hero-specific GPM and
hero-damage-per-minute percentiles (p90/p50/p10 -> good/average/poor) overlay
the hardcoded role tables below; the hardcoded values remain the fallback
when the file or a hero entry is missing. CS, deaths, wards, kill
participation and item timing tables stay hardcoded: OpenDota's whole-game
global percentiles do not map cleanly onto those role/minute-specific curves.

Patch reference: 7.37
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import MatchData, PlayerData

from .knowledge_base import HERO_DATA

logger = logging.getLogger(__name__)

__all__ = [
    "PATCH_VERSION",
    "CS_BENCHMARKS",
    "CS_HERO_OVERRIDES",
    "ITEM_TIMING_BENCHMARKS",
    "DEATHS_PER_30MIN",
    "GPM_BENCHMARKS",
    "DAMAGE_PER_MIN",
    "KILL_PARTICIPATION_PCT",
    "WARDS_PER_30MIN",
    "PatternComparison",
    "ProPatternReport",
    "resolve_role",
    "compare_player",
    "get_benchmark_context",
    "get_cs_benchmark",
    "get_item_timing_benchmark",
    "format_pro_comparison",
]

# Dynamic patch version: prefer data_manager's version, fallback to hardcoded
def _resolve_patch_version() -> str:
    try:
        from .data_manager import get_patch_version
        return get_patch_version()
    except ImportError:
        return "7.37"

PATCH_VERSION = _resolve_patch_version()

# ===================================================================
# Learned benchmark overlay (written by learning/stats_learner.py)
# ===================================================================

# Metrics for which per-hero OpenDota percentiles genuinely map onto a
# good/average/poor table (higher is better, whole-game per-minute rates).
_LEARNED_OVERLAY_METRICS: set[str] = {"gold_per_min", "hero_damage_per_min"}

_learned_bench_cache: dict[str, dict] = {}
_learned_bench_mtime: float | None = None


def _learned_benchmarks_path() -> Path:
    try:
        from .learning.base import LEARNED_DIR
        return LEARNED_DIR / "_benchmarks_raw.json"
    except Exception:
        return Path(__file__).resolve().parents[2] / "data" / "learned" / "_benchmarks_raw.json"


def _learned_hero_slug(name: str) -> str:
    """Hero name -> slug; must match learning.stats_learner._slug."""
    return name.lower().replace(" ", "_").replace("'", "").replace("-", "_")


def _load_learned_benchmarks() -> dict[str, dict]:
    """Lazily load per-hero percentiles, re-reading when the file changes."""
    global _learned_bench_cache, _learned_bench_mtime
    path = _learned_benchmarks_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _learned_bench_cache, _learned_bench_mtime = {}, None
        return {}
    if _learned_bench_mtime == mtime:
        return _learned_bench_cache
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        heroes = payload.get("heroes", {})
        _learned_bench_cache = heroes if isinstance(heroes, dict) else {}
        logger.info(
            "Learned benchmarks loaded: %d heroes overlay hardcoded tables",
            len(_learned_bench_cache),
        )
    except Exception as exc:
        logger.warning("Failed to load learned benchmarks: %s", exc)
        _learned_bench_cache = {}
    _learned_bench_mtime = mtime
    return _learned_bench_cache


def _learned_metric_bench(hero_name: str, metric: str) -> dict | None:
    """Hero-specific {good, average, poor} from learned percentiles, or None.

    p90 -> good, p50 -> average, p10 -> poor. Returns None when the learned
    file, hero, or metric is missing so callers fall back to the hardcoded
    role tables.
    """
    if metric not in _LEARNED_OVERLAY_METRICS:
        return None
    entry = _load_learned_benchmarks().get(_learned_hero_slug(hero_name))
    if not isinstance(entry, dict):
        return None
    values = entry.get(metric)
    if not isinstance(values, dict):
        return None
    p50, p90 = values.get("p50"), values.get("p90")
    if p50 is None or p90 is None:
        return None
    p10 = values.get("p10")
    return {
        "good": int(round(p90)),
        "average": int(round(p50)),
        "poor": int(round(p10)) if p10 is not None else None,
    }


# ===================================================================
# Role label map (multilingual)
# ===================================================================

_ROLE_LABELS: dict[str, dict[str, str]] = {
    "zh": {
        "carry": "大哥（一号位）",
        "mid": "中单（二号位）",
        "offlane": "三号位（副核心）",
        "support_4": "四号位（游走辅助）",
        "support_5": "五号位（硬辅助）",
    },
    "en": {
        "carry": "Carry (Pos 1)",
        "mid": "Mid (Pos 2)",
        "offlane": "Offlane (Pos 3)",
        "support_4": "Soft Support (Pos 4)",
        "support_5": "Hard Support (Pos 5)",
    },
}

# ===================================================================
# CS (last hit) benchmarks — expected last hits at key minutes
# ===================================================================

CS_BENCHMARKS: dict[str, dict[int, int]] = {
    "carry": {5: 30, 10: 65, 15: 110, 20: 160, 25: 210, 30: 260},
    "mid": {5: 25, 10: 55, 15: 90, 20: 130, 25: 170, 30: 210},
    "offlane": {5: 15, 10: 35, 15: 55, 20: 80, 25: 110, 30: 140},
    "support_4": {5: 3, 10: 10, 15: 20, 20: 35, 25: 50, 30: 65},
    "support_5": {5: 2, 10: 5, 15: 10, 20: 18, 25: 28, 30: 38},
}

# Which CS time points to compare for each role.
# Supports shouldn't be judged heavily on CS — only show 10 min for pos 4,
# skip entirely for pos 5.
_CS_COMPARE_MINUTES: dict[str, list[int]] = {
    "carry": [10, 20, 30],
    "mid": [10, 20, 30],
    "offlane": [10, 20],
    "support_4": [10],
    "support_5": [],
}

# Roles for which ward placement is a meaningful metric
_WARD_ROLES: set[str] = {"support_4", "support_5", "offlane"}

# Heroes with notably different farming rates compared to role averages
CS_HERO_OVERRIDES: dict[str, dict[int, int]] = {
    "Anti-Mage": {5: 30, 10: 70, 15: 145, 20: 220, 25: 290, 30: 350},
    "Drow Ranger": {5: 28, 10: 65, 15: 120, 20: 180, 25: 240, 30: 300},
    "Phantom Assassin": {5: 28, 10: 60, 15: 110, 20: 170, 25: 230, 30: 280},
    "Juggernaut": {5: 28, 10: 60, 15: 100, 20: 150, 25: 200, 30: 250},
    "Invoker": {5: 22, 10: 50, 15: 85, 20: 125, 25: 165, 30: 200},
    "Zeus": {5: 20, 10: 45, 15: 80, 20: 120, 25: 160, 30: 190},
    "Sniper": {5: 25, 10: 58, 15: 100, 20: 150, 25: 200, 30: 250},
    "Enigma": {5: 18, 10: 55, 15: 95, 20: 130, 25: 165, 30: 200},
    "Axe": {5: 12, 10: 30, 15: 60, 20: 95, 25: 130, 30: 165},
    "Bristleback": {5: 15, 10: 35, 15: 65, 20: 100, 25: 140, 30: 180},
}

# ===================================================================
# Item timing benchmarks — in seconds from game start
# ===================================================================

ITEM_TIMING_BENCHMARKS: dict[str, dict[str, dict[str, int]]] = {
    "Blink Dagger": {
        "offlane": {"good": 660, "average": 840, "late": 1080},
        "support_4": {"good": 840, "average": 1080, "late": 1380},
        "mid": {"good": 600, "average": 780, "late": 1020},
    },
    "Battle Fury": {
        "carry": {"good": 780, "average": 960, "late": 1200},
    },
    "Black King Bar": {
        "carry": {"good": 1080, "average": 1380, "late": 1740},
        "mid": {"good": 960, "average": 1260, "late": 1560},
        "offlane": {"good": 1200, "average": 1500, "late": 1800},
    },
    "Manta Style": {
        "carry": {"good": 1200, "average": 1500, "late": 1800},
    },
    "Desolator": {
        "carry": {"good": 780, "average": 1020, "late": 1320},
        "mid": {"good": 720, "average": 960, "late": 1260},
    },
    "Aghanim's Scepter": {
        "mid": {"good": 900, "average": 1200, "late": 1560},
        "offlane": {"good": 1080, "average": 1380, "late": 1740},
        "support_4": {"good": 1320, "average": 1680, "late": 2040},
    },
    "Shadow Blade": {
        "mid": {"good": 720, "average": 960, "late": 1260},
        "carry": {"good": 840, "average": 1080, "late": 1380},
    },
    "Radiance": {
        "carry": {"good": 900, "average": 1140, "late": 1440},
    },
    "Assault Cuirass": {
        "carry": {"good": 1500, "average": 1800, "late": 2160},
        "offlane": {"good": 1680, "average": 2040, "late": 2400},
    },
    "Eye of Skadi": {
        "carry": {"good": 1380, "average": 1680, "late": 2040},
    },
}

# ===================================================================
# Per-role performance benchmarks (normalized to 30-minute games)
# ===================================================================

DEATHS_PER_30MIN: dict[str, dict[str, float]] = {
    "carry": {"good": 2.0, "average": 4.0, "poor": 7.0},
    "mid": {"good": 2.5, "average": 4.5, "poor": 7.5},
    "offlane": {"good": 3.0, "average": 5.0, "poor": 8.0},
    "support_4": {"good": 4.0, "average": 6.0, "poor": 9.0},
    "support_5": {"good": 5.0, "average": 7.5, "poor": 11.0},
}

GPM_BENCHMARKS: dict[str, dict[str, int]] = {
    "carry": {"good": 650, "average": 520, "poor": 400},
    "mid": {"good": 600, "average": 480, "poor": 370},
    "offlane": {"good": 450, "average": 370, "poor": 290},
    "support_4": {"good": 350, "average": 280, "poor": 210},
    "support_5": {"good": 280, "average": 220, "poor": 170},
}

DAMAGE_PER_MIN: dict[str, dict[str, int]] = {
    "carry": {"good": 800, "average": 550, "poor": 350},
    "mid": {"good": 750, "average": 500, "poor": 320},
    "offlane": {"good": 500, "average": 350, "poor": 220},
    "support_4": {"good": 400, "average": 280, "poor": 180},
    "support_5": {"good": 300, "average": 200, "poor": 120},
}

KILL_PARTICIPATION_PCT: dict[str, dict[str, int]] = {
    "carry": {"good": 65, "average": 50, "poor": 35},
    "mid": {"good": 70, "average": 55, "poor": 40},
    "offlane": {"good": 65, "average": 50, "poor": 35},
    "support_4": {"good": 75, "average": 60, "poor": 45},
    "support_5": {"good": 70, "average": 55, "poor": 40},
}

WARDS_PER_30MIN: dict[str, dict[str, int]] = {
    "support_4": {"obs_good": 8, "obs_avg": 5, "sen_good": 10, "sen_avg": 6},
    "support_5": {"obs_good": 10, "obs_avg": 7, "sen_good": 12, "sen_avg": 8},
    "offlane": {"obs_good": 3, "obs_avg": 1, "sen_good": 3, "sen_avg": 1},
    "mid": {"obs_good": 2, "obs_avg": 1, "sen_good": 2, "sen_avg": 1},
    "carry": {"obs_good": 1, "obs_avg": 0, "sen_good": 1, "sen_avg": 0},
}


# ===================================================================
# Rating labels (multilingual)
# ===================================================================

_RATING_LABELS: dict[str, dict[str, str]] = {
    "zh": {
        "excellent": "优秀",
        "good": "良好",
        "average": "一般",
        "below_average": "偏低",
        "poor": "较差",
    },
    "en": {
        "excellent": "Excellent",
        "good": "Good",
        "average": "Average",
        "below_average": "Below Average",
        "poor": "Poor",
    },
}

# Keep the old name for backward compatibility inside module
_RATING_CN = _RATING_LABELS["zh"]


# ===================================================================
# Dataclasses
# ===================================================================

@dataclass
class PatternComparison:
    """A single metric comparison against pro benchmarks."""

    metric: str           # "cs_10min", "blink_timing", "deaths", "gpm", etc.
    metric_label: str     # Display name
    actual_value: float
    benchmark_good: float
    benchmark_avg: float
    benchmark_poor: float | None = None
    rating: str = ""      # "excellent", "good", "average", "below_average", "poor"
    description: str = ""  # Explanation


@dataclass
class ProPatternReport:
    """Full pro-pattern comparison report for one player."""

    hero_name: str
    role: str
    role_label: str       # Role name
    comparisons: list[PatternComparison] = field(default_factory=list)
    summary: str = ""
    overall_rating: str = ""


# ===================================================================
# Lane role mapping
# ===================================================================

_LANE_ROLE_MAP: dict[int, str] = {
    1: "carry",
    2: "mid",
    3: "offlane",
    4: "support_4",
}


def resolve_role(player: PlayerData, hero_name: str) -> str:
    """Determine the player's role from hero data or lane_role field.

    Priority:
    1. HERO_DATA knowledge base (most accurate)
    2. player.lane_role integer mapping
    3. Fallback to ``"offlane"`` as a reasonable middle ground
    """
    # Check knowledge base first
    if hero_name in HERO_DATA:
        role = HERO_DATA[hero_name].get("role", "")
        if role:
            return role

    # Map lane_role integer
    if player.lane_role in _LANE_ROLE_MAP:
        role = _LANE_ROLE_MAP[player.lane_role]
        # Heuristic: if lane_role says carry but obs/sen placed >= 5,
        # likely a support mis-classified
        if role == "carry" and (player.obs_placed + player.sen_placed) >= 5:
            return "support_5"
        return role

    # Fallback: use lane field as secondary signal
    lane_map = {1: "carry", 2: "mid", 3: "offlane"}
    if player.lane in lane_map:
        return lane_map[player.lane]

    return "offlane"


# ===================================================================
# Rating helpers
# ===================================================================

def _rate_higher_is_better(
    actual: float, good: float, avg: float, poor: float | None
) -> str:
    """Rate a metric where higher values are better."""
    if actual >= good:
        return "excellent"
    if actual >= (good + avg) / 2:
        return "good"
    if actual >= avg:
        return "average"
    if poor is not None and actual <= poor:
        return "poor"
    return "below_average"


def _rate_lower_is_better(
    actual: float, good: float, avg: float, poor: float | None
) -> str:
    """Rate a metric where lower values are better (e.g. deaths)."""
    if actual <= good:
        return "excellent"
    if actual <= (good + avg) / 2:
        return "good"
    if actual <= avg:
        return "average"
    if poor is not None and actual >= poor:
        return "poor"
    return "below_average"


def _get_rating_label(rating: str, lang: str = "zh") -> str:
    """Return a localized rating label."""
    labels = _RATING_LABELS.get(lang, _RATING_LABELS["zh"])
    return labels.get(rating, rating)


# ===================================================================
# Metric label map (multilingual)
# ===================================================================

_METRIC_LABELS: dict[str, dict[str, str]] = {
    "zh": {
        "cs": "分钟补刀",
        "deaths": "死亡控制",
        "gpm": "每分钟金钱（GPM）",
        "damage_per_min": "每分钟英雄伤害",
        "kill_participation": "击杀参与率",
        "obs_wards": "观察守卫放置",
        "sen_wards": "岗哨守卫放置",
        "item_timing_suffix": "出装时间",
    },
    "en": {
        "cs": "min CS",
        "deaths": "Death Control",
        "gpm": "Gold Per Minute (GPM)",
        "damage_per_min": "Hero Damage Per Minute",
        "kill_participation": "Kill Participation",
        "obs_wards": "Observer Ward Placement",
        "sen_wards": "Sentry Ward Placement",
        "item_timing_suffix": "Timing",
    },
}


# ===================================================================
# Interpolation helper
# ===================================================================

def _interpolate_cs(benchmarks: dict[int, int], minute: int) -> float:
    """Linearly interpolate CS benchmarks for an arbitrary minute."""
    if minute <= 0:
        return 0.0
    keys = sorted(benchmarks.keys())
    if minute <= keys[0]:
        return benchmarks[keys[0]] * minute / keys[0]
    if minute >= keys[-1]:
        # Extrapolate linearly from last two points
        k1, k2 = keys[-2], keys[-1]
        rate = (benchmarks[k2] - benchmarks[k1]) / (k2 - k1)
        return benchmarks[k2] + rate * (minute - k2)
    # Find surrounding keys
    for i in range(len(keys) - 1):
        if keys[i] <= minute <= keys[i + 1]:
            lo, hi = keys[i], keys[i + 1]
            t = (minute - lo) / (hi - lo)
            return benchmarks[lo] + t * (benchmarks[hi] - benchmarks[lo])
    return 0.0


# ===================================================================
# Core functions
# ===================================================================

def get_cs_benchmark(hero_name: str, role: str, minute: int) -> float:
    """Return expected CS at a given minute for a hero/role.

    Uses hero-specific overrides if available, otherwise falls back
    to the role-based CS_BENCHMARKS table.
    """
    if hero_name in CS_HERO_OVERRIDES:
        return _interpolate_cs(CS_HERO_OVERRIDES[hero_name], minute)
    effective_role = role if role in CS_BENCHMARKS else "offlane"
    return _interpolate_cs(CS_BENCHMARKS[effective_role], minute)


def get_item_timing_benchmark(item_name: str, role: str) -> dict | None:
    """Return the timing benchmark dict for an item/role combo.

    Returns ``None`` if no benchmark exists for the combination.
    """
    item_data = ITEM_TIMING_BENCHMARKS.get(item_name)
    if item_data is None:
        return None
    return item_data.get(role)


def get_benchmark_context(hero_name: str, role: str, game_time: int, lang: str = "zh") -> str:
    """Return text with relevant benchmarks for a specific game time.

    Useful for injecting into LLM prompts alongside decision contexts.
    """
    minute = game_time // 60
    effective_role = role if role in CS_BENCHMARKS else "offlane"
    role_labels = _ROLE_LABELS.get(lang, _ROLE_LABELS["zh"])
    role_label = role_labels.get(effective_role, effective_role)

    if lang == "en":
        lines: list[str] = ["### Pro Benchmark Reference"]
        role_cs = _interpolate_cs(CS_BENCHMARKS[effective_role], minute)
        lines.append(f"- Expected CS at {minute} min for {role_label}: ~{int(role_cs)}")
        if hero_name in CS_HERO_OVERRIDES:
            hero_cs = _interpolate_cs(CS_HERO_OVERRIDES[hero_name], minute)
            lines.append(f"- Pro-level CS for {hero_name} at {minute} min: ~{int(hero_cs)}")
        gpm_data = _learned_metric_bench(hero_name, "gold_per_min") or GPM_BENCHMARKS.get(effective_role)
        if gpm_data:
            lines.append(f"- Expected GPM: {gpm_data['average']}+")
        deaths_data = DEATHS_PER_30MIN.get(effective_role)
        if deaths_data:
            scaled = deaths_data["average"] * minute / 30 if minute > 0 else 0
            lines.append(f"- Expected deaths by {minute} min: <={scaled:.1f}")
        dmg_data = _learned_metric_bench(hero_name, "hero_damage_per_min") or DAMAGE_PER_MIN.get(effective_role)
        if dmg_data:
            lines.append(f"- Expected damage per minute: {dmg_data['average']}+")
    else:
        lines = ["### 职业基准参考"]
        role_cs = _interpolate_cs(CS_BENCHMARKS[effective_role], minute)
        lines.append(f"- 该时间点（{minute}分钟）{role_label}预期补刀: ~{int(role_cs)}")
        if hero_name in CS_HERO_OVERRIDES:
            hero_cs = _interpolate_cs(CS_HERO_OVERRIDES[hero_name], minute)
            lines.append(f"- 职业选手该英雄{minute}分钟补刀: ~{int(hero_cs)}")
        gpm_data = _learned_metric_bench(hero_name, "gold_per_min") or GPM_BENCHMARKS.get(effective_role)
        if gpm_data:
            lines.append(f"- 预期GPM: {gpm_data['average']}+")
        deaths_data = DEATHS_PER_30MIN.get(effective_role)
        if deaths_data:
            scaled = deaths_data["average"] * minute / 30 if minute > 0 else 0
            lines.append(f"- 该时间点预期死亡次数: <={scaled:.1f}")
        dmg_data = _learned_metric_bench(hero_name, "hero_damage_per_min") or DAMAGE_PER_MIN.get(effective_role)
        if dmg_data:
            lines.append(f"- 预期每分钟伤害: {dmg_data['average']}+")

    return "\n".join(lines)


def compare_player(player: PlayerData, match: MatchData, lang: str = "zh") -> ProPatternReport:
    """Compare a player's performance against pro benchmarks.

    Evaluates CS, deaths, GPM, hero damage, item timings, ward placement,
    and kill participation. Returns a ``ProPatternReport`` with individual
    ``PatternComparison`` entries and an overall summary.
    """
    hero_name = player.hero_name
    role = resolve_role(player, hero_name)
    role_labels = _ROLE_LABELS.get(lang, _ROLE_LABELS["zh"])
    role_label = role_labels.get(role, role)
    effective_role = role if role in CS_BENCHMARKS else "offlane"
    mlabels = _METRIC_LABELS.get(lang, _METRIC_LABELS["zh"])

    report = ProPatternReport(
        hero_name=hero_name,
        role=role,
        role_label=role_label,
    )

    duration_min = max(match.duration / 60, 1.0)
    scale_30 = 30.0 / duration_min  # scale factor to normalize to 30 min

    # ------------------------------------------------------------------
    # 1. CS comparison at role-appropriate time points
    # ------------------------------------------------------------------
    cs_minutes = _CS_COMPARE_MINUTES.get(effective_role, [10, 20, 30])
    if player.lh_t and cs_minutes:
        for target_min in cs_minutes:
            idx = target_min  # lh_t is indexed by minute
            if idx < len(player.lh_t):
                actual_cs = player.lh_t[idx]
                expected = get_cs_benchmark(hero_name, role, target_min)
                good_cs = expected * 1.15  # 115% of expected = excellent
                avg_cs = expected
                poor_cs = expected * 0.6

                rating = _rate_higher_is_better(actual_cs, good_cs, avg_cs, poor_cs)
                diff = actual_cs - avg_cs
                if lang == "en":
                    if diff >= 0:
                        desc = f"{actual_cs} CS at {target_min} min, {int(diff)} above benchmark ({int(avg_cs)})"
                    else:
                        desc = f"{actual_cs} CS at {target_min} min, {int(abs(diff))} below benchmark ({int(avg_cs)})"
                    cs_label = f"{target_min}{mlabels['cs']}"
                else:
                    if diff >= 0:
                        desc = f"{target_min}分钟补刀{actual_cs}，高于基准{int(avg_cs)}约{int(diff)}刀"
                    else:
                        desc = f"{target_min}分钟补刀{actual_cs}，低于基准{int(avg_cs)}约{int(abs(diff))}刀"
                    cs_label = f"{target_min}{mlabels['cs']}"

                report.comparisons.append(PatternComparison(
                    metric=f"cs_{target_min}min",
                    metric_label=cs_label,
                    actual_value=actual_cs,
                    benchmark_good=good_cs,
                    benchmark_avg=avg_cs,
                    benchmark_poor=poor_cs,
                    rating=rating,
                    description=desc,
                ))

    # ------------------------------------------------------------------
    # 2. Deaths comparison (lower is better)
    # ------------------------------------------------------------------
    deaths_bench = DEATHS_PER_30MIN.get(effective_role)
    if deaths_bench:
        normalized_deaths = player.deaths * scale_30
        rating = _rate_lower_is_better(
            normalized_deaths,
            deaths_bench["good"],
            deaths_bench["average"],
            deaths_bench["poor"],
        )
        rating_disp = _get_rating_label(rating, lang)

        if lang == "en":
            desc_parts = [f"{player.deaths} deaths total"]
            if duration_min < 28 or duration_min > 32:
                desc_parts.append(f" (normalized to 30 min: ~{normalized_deaths:.1f})")
            desc_parts.append(f", rating: {rating_disp}")
        else:
            desc_parts = [f"全场{player.deaths}次死亡"]
            if duration_min < 28 or duration_min > 32:
                desc_parts.append(f"（换算30分钟约{normalized_deaths:.1f}次）")
            desc_parts.append(f"，评价：{rating_disp}")

        report.comparisons.append(PatternComparison(
            metric="deaths",
            metric_label=mlabels["deaths"],
            actual_value=normalized_deaths,
            benchmark_good=deaths_bench["good"],
            benchmark_avg=deaths_bench["average"],
            benchmark_poor=deaths_bench["poor"],
            rating=rating,
            description="".join(desc_parts),
        ))

    # ------------------------------------------------------------------
    # 3. GPM comparison (learned hero-specific values overlay role table)
    # ------------------------------------------------------------------
    gpm_bench = _learned_metric_bench(hero_name, "gold_per_min") or GPM_BENCHMARKS.get(effective_role)
    if gpm_bench:
        gpm = player.gold_per_min
        rating = _rate_higher_is_better(
            gpm, gpm_bench["good"], gpm_bench["average"], gpm_bench["poor"]
        )
        rating_disp = _get_rating_label(rating, lang)
        if lang == "en":
            desc = f"GPM {gpm}, {role_label} benchmark {gpm_bench['average']}, rating: {rating_disp}"
        else:
            desc = f"GPM {gpm}，{role_label}基准{gpm_bench['average']}，评价：{rating_disp}"
        report.comparisons.append(PatternComparison(
            metric="gpm",
            metric_label=mlabels["gpm"],
            actual_value=gpm,
            benchmark_good=gpm_bench["good"],
            benchmark_avg=gpm_bench["average"],
            benchmark_poor=gpm_bench["poor"],
            rating=rating,
            description=desc,
        ))

    # ------------------------------------------------------------------
    # 4. Hero damage per minute (learned hero-specific values overlay)
    # ------------------------------------------------------------------
    dmg_bench = _learned_metric_bench(hero_name, "hero_damage_per_min") or DAMAGE_PER_MIN.get(effective_role)
    if dmg_bench:
        dpm = player.hero_damage / duration_min if duration_min > 0 else 0
        rating = _rate_higher_is_better(
            dpm, dmg_bench["good"], dmg_bench["average"], dmg_bench["poor"]
        )
        rating_disp = _get_rating_label(rating, lang)
        if lang == "en":
            desc = f"Damage/min {dpm:.0f}, {role_label} benchmark {dmg_bench['average']}, rating: {rating_disp}"
        else:
            desc = f"每分钟伤害{dpm:.0f}，{role_label}基准{dmg_bench['average']}，评价：{rating_disp}"
        report.comparisons.append(PatternComparison(
            metric="damage_per_min",
            metric_label=mlabels["damage_per_min"],
            actual_value=round(dpm, 1),
            benchmark_good=dmg_bench["good"],
            benchmark_avg=dmg_bench["average"],
            benchmark_poor=dmg_bench["poor"],
            rating=rating,
            description=desc,
        ))

    # ------------------------------------------------------------------
    # 5. Kill participation
    # ------------------------------------------------------------------
    kp_bench = KILL_PARTICIPATION_PCT.get(effective_role)
    if kp_bench:
        team_kills = 0
        for p in match.players:
            if p.team == player.team:
                team_kills += p.kills
        if team_kills > 0:
            kp = (player.kills + player.assists) / team_kills * 100
        else:
            kp = 0.0
        rating = _rate_higher_is_better(
            kp, kp_bench["good"], kp_bench["average"], kp_bench["poor"]
        )
        rating_disp = _get_rating_label(rating, lang)
        if lang == "en":
            desc = f"Kill participation {kp:.1f}%, {role_label} benchmark {kp_bench['average']}%, rating: {rating_disp}"
        else:
            desc = f"击杀参与率{kp:.1f}%，{role_label}基准{kp_bench['average']}%，评价：{rating_disp}"
        report.comparisons.append(PatternComparison(
            metric="kill_participation",
            metric_label=mlabels["kill_participation"],
            actual_value=round(kp, 1),
            benchmark_good=kp_bench["good"],
            benchmark_avg=kp_bench["average"],
            benchmark_poor=kp_bench["poor"],
            rating=rating,
            description=desc,
        ))

    # ------------------------------------------------------------------
    # 6. Ward placement (only for support/offlane roles)
    # ------------------------------------------------------------------
    ward_bench = WARDS_PER_30MIN.get(effective_role) if effective_role in _WARD_ROLES else None
    if ward_bench:
        obs_scaled = player.obs_placed * scale_30
        sen_scaled = player.sen_placed * scale_30

        # Observer wards
        obs_rating = _rate_higher_is_better(
            obs_scaled,
            ward_bench["obs_good"],
            ward_bench["obs_avg"],
            None,
        )
        obs_rating_disp = _get_rating_label(obs_rating, lang)
        if lang == "en":
            obs_desc = (
                f"Placed {player.obs_placed} observer wards"
                f" (normalized to 30 min: {obs_scaled:.1f}),"
                f" benchmark {ward_bench['obs_avg']}+, rating: {obs_rating_disp}"
            )
        else:
            obs_desc = (
                f"放置{player.obs_placed}个观察守卫"
                f"（换算30分钟{obs_scaled:.1f}个），"
                f"基准{ward_bench['obs_avg']}+，评价：{obs_rating_disp}"
            )
        report.comparisons.append(PatternComparison(
            metric="obs_wards",
            metric_label=mlabels["obs_wards"],
            actual_value=round(obs_scaled, 1),
            benchmark_good=ward_bench["obs_good"],
            benchmark_avg=ward_bench["obs_avg"],
            rating=obs_rating,
            description=obs_desc,
        ))

        # Sentry wards
        sen_rating = _rate_higher_is_better(
            sen_scaled,
            ward_bench["sen_good"],
            ward_bench["sen_avg"],
            None,
        )
        sen_rating_disp = _get_rating_label(sen_rating, lang)
        if lang == "en":
            sen_desc = (
                f"Placed {player.sen_placed} sentry wards"
                f" (normalized to 30 min: {sen_scaled:.1f}),"
                f" benchmark {ward_bench['sen_avg']}+, rating: {sen_rating_disp}"
            )
        else:
            sen_desc = (
                f"放置{player.sen_placed}个岗哨守卫"
                f"（换算30分钟{sen_scaled:.1f}个），"
                f"基准{ward_bench['sen_avg']}+，评价：{sen_rating_disp}"
            )
        report.comparisons.append(PatternComparison(
            metric="sen_wards",
            metric_label=mlabels["sen_wards"],
            actual_value=round(sen_scaled, 1),
            benchmark_good=ward_bench["sen_good"],
            benchmark_avg=ward_bench["sen_avg"],
            rating=sen_rating,
            description=sen_desc,
        ))

    # ------------------------------------------------------------------
    # 7. Item timing comparisons
    # ------------------------------------------------------------------
    for item_event in player.items:
        item_name = item_event.item_name
        bench = get_item_timing_benchmark(item_name, effective_role)
        if bench is None:
            continue

        actual_time = item_event.time
        good_t = bench["good"]
        avg_t = bench["average"]
        late_t = bench["late"]

        # For item timings, lower (faster) is better
        rating = _rate_lower_is_better(actual_time, good_t, avg_t, late_t)

        actual_min = actual_time // 60
        actual_sec = actual_time % 60
        good_min = good_t // 60
        avg_min = avg_t // 60

        rating_disp = _get_rating_label(rating, lang)
        if lang == "en":
            exc_label = _get_rating_label("excellent", lang)
            avg_label = _get_rating_label("average", lang)
            desc = (
                f"{item_name} completed at {actual_min}:{actual_sec:02d}, "
                f"pro benchmark {good_min}:{good_t % 60:02d} ({exc_label})/"
                f"{avg_min}:{avg_t % 60:02d} ({avg_label}), "
                f"rating: {rating_disp}"
            )
            item_metric_label = f"{item_name} {mlabels['item_timing_suffix']}"
        else:
            desc = (
                f"{item_name}在{actual_min}:{actual_sec:02d}完成，"
                f"职业基准{good_min}:{good_t % 60:02d}（优秀）/"
                f"{avg_min}:{avg_t % 60:02d}（一般），"
                f"评价：{rating_disp}"
            )
            item_metric_label = f"{item_name}{mlabels['item_timing_suffix']}"

        report.comparisons.append(PatternComparison(
            metric=f"item_{item_name.lower().replace(' ', '_')}",
            metric_label=item_metric_label,
            actual_value=actual_time,
            benchmark_good=good_t,
            benchmark_avg=avg_t,
            benchmark_poor=late_t,
            rating=rating,
            description=desc,
        ))

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    _compute_summary(report, lang=lang)

    return report


def _compute_summary(report: ProPatternReport, lang: str = "zh") -> None:
    """Compute overall_rating and summary for a ProPatternReport."""
    if not report.comparisons:
        report.overall_rating = "average"
        if lang == "en":
            report.summary = f"{report.hero_name} ({report.role_label}): insufficient data for a comprehensive evaluation."
        else:
            report.summary = f"{report.hero_name}（{report.role_label}）的数据不足，无法给出全面评价。"
        return

    rating_scores = {
        "excellent": 5,
        "good": 4,
        "average": 3,
        "below_average": 2,
        "poor": 1,
    }

    total = 0.0
    count = 0
    for c in report.comparisons:
        score = rating_scores.get(c.rating, 3)
        total += score
        count += 1

    avg_score = total / count if count > 0 else 3.0

    if avg_score >= 4.5:
        report.overall_rating = "excellent"
    elif avg_score >= 3.75:
        report.overall_rating = "good"
    elif avg_score >= 2.75:
        report.overall_rating = "average"
    elif avg_score >= 1.75:
        report.overall_rating = "below_average"
    else:
        report.overall_rating = "poor"

    overall_disp = _get_rating_label(report.overall_rating, lang)

    # Build summary with strengths and weaknesses
    strengths: list[str] = []
    weaknesses: list[str] = []
    for c in report.comparisons:
        if c.rating in ("excellent", "good"):
            strengths.append(c.metric_label)
        elif c.rating in ("below_average", "poor"):
            weaknesses.append(c.metric_label)

    if lang == "en":
        sep = ", "
        parts = [f"{report.hero_name} ({report.role_label}) overall performance: {overall_disp}. "]
        if strengths:
            parts.append(f"Highlights: {sep.join(strengths[:3])} stood out. ")
        if weaknesses:
            parts.append(f"Needs improvement: {sep.join(weaknesses[:3])} below pro level. ")
        if not strengths and not weaknesses:
            parts.append("All metrics are close to pro averages. ")
    else:
        sep = "、"
        parts = [f"{report.hero_name}（{report.role_label}）整体表现{overall_disp}。"]
        if strengths:
            parts.append(f"亮点：{sep.join(strengths[:3])}表现出色。")
        if weaknesses:
            parts.append(f"需要改进：{sep.join(weaknesses[:3])}低于职业水准。")
        if not strengths and not weaknesses:
            parts.append("各项数据接近职业平均水平。")

    report.summary = "".join(parts)


# ===================================================================
# Serialization
# ===================================================================

def format_pro_comparison(report: ProPatternReport) -> dict:
    """Serialize a ``ProPatternReport`` to a JSON-friendly dict.

    Suitable for returning via the API to the frontend.
    """
    return {
        "hero_name": report.hero_name,
        "role": report.role,
        "role_label": report.role_label,
        "overall_rating": report.overall_rating,
        "summary": report.summary,
        "comparisons": [
            {
                "metric": c.metric,
                "metric_label": c.metric_label,
                "actual_value": c.actual_value,
                "benchmark_good": c.benchmark_good,
                "benchmark_avg": c.benchmark_avg,
                "benchmark_poor": c.benchmark_poor,
                "rating": c.rating,
                "description": c.description,
            }
            for c in report.comparisons
        ],
    }
