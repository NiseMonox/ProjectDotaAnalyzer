"""Benchmark data and pattern comparison engine.

Three kinds of reference numbers, never mixed up in a text:

* **Pro reference** (``benchmark_source`` = ``"pro_reference"``): numbers of
  professional matches. Last hits per position and kill participation are
  the means of the pro matches the learning pipeline mined
  (``backend/data/learned/_mining_aggregates.json``, ``learning/
  match_miner.py``) when that file is there and sufficient, else the tables
  below, which hold the values of the run of 2026-10-10 (100 matches of
  7.41f, 200 player-games per position). Deaths and ward placement are
  hand-written tables that agree with the same run.
* **Guideline** (``"guideline"``): hand-written numbers nothing here
  measures — the item timings. Text built from them says "guideline" and
  never "pro".
* **Population data** (``"opendota_population"``): OpenDota percentiles over
  ALL ranks. GPM and hero damage per minute are always of this kind: the
  per-hero percentiles the learning pipeline writes to ``backend/data/
  learned/_benchmarks_raw.json`` (``learning/stats_learner.py``) when the
  hero has them for the player's side of the team (cores: positions 1-3,
  supports: 4-5), else the role tables below, which are all-rank
  percentiles as well (measured 2026-10-10 on 7.41f). Text built from them
  says "OpenDota, all ranks" and never "pro".

Roles come from :mod:`positions` (derived position 1-5), the single source
of truth for "which role did this player play"; :func:`resolve_role` never
raises.

The tables describe standard 5v5 matches played to the end:
:func:`benchmarks_apply` is False for Turbo and other special modes, and
such a match gets no comparison. Of a replay that ends before the game does
(``MatchData.partial``) only the rows the recording covers are compared:
last hits at the minutes it reached.

The game patch is never hardcoded here: ``PATCH_VERSION`` resolves through
``data_manager`` at access time.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import MatchData, PlayerData

logger = logging.getLogger(__name__)

__all__ = [
    "PATCH_VERSION",
    "SOURCE_PRO",
    "SOURCE_GUIDELINE",
    "SOURCE_POPULATION",
    "CS_BENCHMARKS",
    "CS_TOP_QUARTER",
    "ITEM_TIMING_BENCHMARKS",
    "DEATHS_PER_30MIN",
    "GPM_BENCHMARKS",
    "DAMAGE_PER_MIN",
    "KILL_PARTICIPATION_PCT",
    "WARDS_PER_30MIN",
    "PatternComparison",
    "ProPatternReport",
    "resolve_role",
    "benchmarks_apply",
    "compare_player",
    "get_benchmark_context",
    "get_cs_benchmark",
    "get_item_timing_benchmark",
    "format_pro_comparison",
    "get_patch_version",
]

# Benchmark provenance codes (serialized as ``benchmark_source``).
SOURCE_PRO = "pro_reference"
SOURCE_GUIDELINE = "guideline"
SOURCE_POPULATION = "opendota_population"


def _item_metric(item_name: str) -> str:
    """Metric code of an item-timing row: "Aghanim's Scepter" -> "item_aghanims_scepter"."""
    slug = re.sub(r"[^a-z0-9]+", "_", item_name.lower().replace("'", "")).strip("_")
    return f"item_{slug}"


def _hero_text(name: str, lang: str) -> str:
    """A hero name as written in text (never a raw code such as "hero_150")."""
    try:
        from .data_manager import hero_display_name
        return hero_display_name(name, lang) or name
    except Exception:
        return name


def _item_text(name: str, lang: str) -> str:
    """An item name as written in text of language ``lang`` (the benchmark
    tables and metric codes keep the English name)."""
    try:
        from .data_manager import item_display_name
        return item_display_name(name, lang) or name
    except Exception:
        return name


def get_patch_version() -> str:
    """Current game patch from data_manager ("" when unknown)."""
    try:
        from . import data_manager
    except Exception:
        return ""
    for name in ("get_current_patch", "get_patch_version"):
        fn = getattr(data_manager, name, None)
        if not callable(fn):
            continue
        try:
            value = fn()
        except Exception as exc:
            logger.debug("data_manager.%s failed: %s", name, exc)
            continue
        if value:
            return str(value)
    return ""


def __getattr__(name: str):
    # PEP 562: ``PATCH_VERSION`` is resolved on every access so it never
    # goes stale after a game-data update.
    if name == "PATCH_VERSION":
        return get_patch_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ===================================================================
# Learned benchmark overlay (written by learning/stats_learner.py)
# ===================================================================

# Metrics for which per-hero OpenDota percentiles genuinely map onto a
# good/average/poor table (higher is better, whole-game per-minute rates).
_LEARNED_OVERLAY_METRICS: set[str] = {"gold_per_min", "hero_damage_per_min"}

_learned_bench_cache: dict[str, dict] = {}
_learned_bench_mtime: float | None = None
_learned_bench_lock = threading.Lock()


def _learned_benchmarks_path() -> Path:
    try:
        from .learning.base import LEARNED_DIR
        return LEARNED_DIR / "_benchmarks_raw.json"
    except Exception:
        return Path(__file__).resolve().parents[2] / "data" / "learned" / "_benchmarks_raw.json"


def _learned_hero_slug(name: str) -> str:
    """Hero name -> slug; must match learning.stats_learner._bench_key."""
    return name.lower().replace(" ", "_").replace("'", "").replace("-", "_")


def _load_learned_benchmarks() -> dict[str, dict]:
    """Lazily load per-hero percentiles, re-reading when the file changes.

    Thread-safe: the cache is swapped atomically under a lock.
    """
    global _learned_bench_cache, _learned_bench_mtime
    path = _learned_benchmarks_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        with _learned_bench_lock:
            _learned_bench_cache, _learned_bench_mtime = {}, None
        return {}
    if _learned_bench_mtime == mtime:
        return _learned_bench_cache
    with _learned_bench_lock:
        if _learned_bench_mtime == mtime:
            return _learned_bench_cache
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            heroes = payload.get("heroes", {}) if isinstance(payload, dict) else {}
            cache = heroes if isinstance(heroes, dict) else {}
            logger.info("Learned benchmarks loaded: %d heroes", len(cache))
        except Exception as exc:
            logger.warning("Failed to load learned benchmarks: %s", exc)
            cache = {}
        _learned_bench_cache, _learned_bench_mtime = cache, mtime
        return cache


# Roles of this module -> the half of a hero's games OpenDota benchmarks
# separately ("cores or supports, estimated from farm within each team").
_BENCH_SIDES: dict[str, str] = {
    "carry": "core", "mid": "core", "offlane": "core",
    "support_4": "support", "support_5": "support",
}


def _learned_metric_bench(hero_name: str, metric: str, role: str) -> dict | None:
    """Hero-specific {good, average, poor, side} from learned percentiles, or None.

    p90 -> good, p50 -> average, p10 -> poor, of the hero's games on the
    player's side of the team (``side``: "core" | "support"). Returns None
    when the learned file, hero, side or metric is missing so callers fall
    back to the hardcoded role tables.

    The hero's percentiles over all its games are never used: they mix
    both sides, so whoever plays the hero in its less common role is rated
    against the other one (first full learning run, real replay: a position
    5 Techies with 382 GPM was "below average" against a median of 418; the
    support half's median is 392, the core half's 499).
    """
    if metric not in _LEARNED_OVERLAY_METRICS:
        return None
    side = _BENCH_SIDES.get(role)
    entry = _load_learned_benchmarks().get(_learned_hero_slug(hero_name))
    if side is None or not isinstance(entry, dict):
        return None
    roles = entry.get("roles")
    values = roles.get(side) if isinstance(roles, dict) else None
    values = values.get(metric) if isinstance(values, dict) else None
    if not isinstance(values, dict):
        return None
    p50, p90 = values.get("p50"), values.get("p90")
    if p50 is None or p90 is None:
        return None
    p10 = values.get("p10")
    try:
        return {
            "good": int(round(float(p90))),
            "average": int(round(float(p50))),
            "poor": int(round(float(p10))) if p10 is not None else None,
            "source": SOURCE_POPULATION,
            "side": side,
        }
    except (TypeError, ValueError):
        return None


def _metric_bench(hero_name: str, metric: str, role_table: dict, role: str) -> dict | None:
    """Benchmark for GPM / damage per minute: population data either way.

    The hero's own learned percentiles on the player's side of the team
    when there are any (``scope`` "hero", with ``side``), else the all-rank
    role table (``scope`` "role").
    """
    learned = _learned_metric_bench(hero_name, metric, role)
    if learned is not None:
        return {**learned, "scope": "hero"}
    row = role_table.get(role)
    if row is None:
        return None
    return {**row, "source": SOURCE_POPULATION, "scope": "role"}


def _side_text(bench: dict, lang: str) -> str:
    """How a text names the half of a hero's games its percentiles come from."""
    support = bench.get("side") == "support"
    if lang == "en":
        return "as a support (positions 4-5)" if support else "as a core (positions 1-3)"
    return "打4/5号位时" if support else "打1/2/3号位时"


def _population_ref(bench: dict, hero_text: str, role_label: str, lang: str) -> str:
    """"<whose> OpenDota median (all ranks) N": the reference of a population row."""
    value = bench["average"]
    if bench.get("scope") == "hero":
        side = _side_text(bench, lang)
        return (f"{hero_text} {side}: OpenDota median (all ranks) {value}" if lang == "en"
                else f"{hero_text}{side}在OpenDota全分段中位数{value}")
    return (f"OpenDota median of {role_label} heroes (all ranks) {value}" if lang == "en"
            else f"{role_label}英雄在OpenDota全分段中位数{value}")


# ===================================================================
# Pro-match overlay (written by learning/match_miner.py)
# ===================================================================

# Roles of this module -> position key of the mined aggregates.
_ROLE_POSITIONS: dict[str, str] = {
    "carry": "1", "mid": "2", "offlane": "3", "support_4": "4", "support_5": "5",
}

_mined_cache: dict[str, dict] = {}
_mined_mtime: float | None = None
_mined_lock = threading.Lock()


def _mined_aggregates_path() -> Path:
    try:
        from .learning.base import LEARNED_DIR, MINING_AGGREGATES
        return LEARNED_DIR / MINING_AGGREGATES
    except Exception:
        return Path(__file__).resolve().parents[2] / "data" / "learned" / "_mining_aggregates.json"


def _min_position_sample() -> int:
    try:
        from .learning.base import MINING_MIN_SAMPLES
        return int(MINING_MIN_SAMPLES["position"])
    except Exception:
        return 30


def _load_mined_positions() -> dict[str, dict]:
    """Per-position statistics of the mined pro matches, re-read when the file changes.

    ``{"1": {...}, ...}`` as the match miner stores them; {} without the
    file, for aggregates the miner itself found too thin (``sufficient``)
    and for matches that are not professional ones. Thread-safe like
    :func:`_load_learned_benchmarks`.
    """
    global _mined_cache, _mined_mtime
    path = _mined_aggregates_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        with _mined_lock:
            _mined_cache, _mined_mtime = {}, None
        return {}
    if _mined_mtime == mtime:
        return _mined_cache
    with _mined_lock:
        if _mined_mtime == mtime:
            return _mined_cache
        cache: dict[str, dict] = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            positions = payload.get("positions") if isinstance(payload, dict) else None
            if (isinstance(positions, dict) and payload.get("sufficient") is True
                    and payload.get("source") == "pro"):
                cache = {str(pos): stats for pos, stats in positions.items() if isinstance(stats, dict)}
                logger.info("Pro-match aggregates loaded: %d positions, %s matches",
                            len(cache), payload.get("matches_mined"))
        except Exception as exc:
            logger.warning("Failed to load the pro-match aggregates: %s", exc)
        _mined_cache, _mined_mtime = cache, mtime
        return cache


def _mined_number(value, sample, minimum: int) -> float | None:
    """A mined value that rests on at least *minimum* observations, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return None
    if isinstance(sample, bool) or not isinstance(sample, (int, float)) or sample < minimum:
        return None
    return float(value)


def _mined_cs(role: str) -> dict[int, tuple[float, float | None]]:
    """``{minute: (mean, top-quarter value or None)}`` of a role's last hits in pro matches.

    Only minutes with at least the miner's minimum of player-games; {} when
    nothing was mined. Aggregates written before the miner sampled the whole
    curve hold minute 10 only.
    """
    stats = _load_mined_positions().get(_ROLE_POSITIONS.get(role, ""))
    if not stats:
        return {}
    minimum = _min_position_sample()
    rows = stats.get("cs_curve")
    if not isinstance(rows, dict):
        rows = {"10": {"sample": stats.get("cs_at_10_sample"), "mean": stats.get("cs_at_10_mean"),
                       "p75": stats.get("cs_at_10_p75")}}
    curve: dict[int, tuple[float, float | None]] = {}
    for minute, row in rows.items():
        if not isinstance(row, dict) or not str(minute).isdigit():
            continue
        mean = _mined_number(row.get("mean"), row.get("sample"), minimum)
        if mean is not None and mean > 0:
            curve[int(minute)] = (mean, _mined_number(row.get("p75"), row.get("sample"), minimum))
    return curve


def _role_cs_curve(role: str) -> dict[int, float]:
    """A role's last-hit curve: the table, with the mined means laid over it."""
    curve: dict[int, float] = dict(CS_BENCHMARKS[role])
    for minute, (mean, _top) in _mined_cs(role).items():
        curve[minute] = mean
    return curve


def _kill_participation_bench(role: str) -> dict | None:
    """``{good, average, poor}`` (percent) of a role's kill participation.

    Mean and top-quarter value of the mined pro matches when there are any,
    else the table. "Poor" keeps the table's distance below the average: no
    lower quartile of professionals is a line for an amateur game.
    """
    row = KILL_PARTICIPATION_PCT.get(role)
    if row is None:
        return None
    stats = _load_mined_positions().get(_ROLE_POSITIONS.get(role, ""))
    minimum = _min_position_sample()
    mean = _mined_number((stats or {}).get("kill_participation_mean"), (stats or {}).get("sample"), minimum)
    if mean is None or not 0 < mean <= 1:
        return dict(row)
    average = int(round(mean * 100))
    top = _mined_number(stats.get("kill_participation_p75"), stats.get("sample"), minimum)
    good = int(round(top * 100)) if top is not None and average < top * 100 <= 100 else None
    spread = row["average"] - row["poor"]
    return {"good": good if good is not None else min(100, average + (row["good"] - row["average"])),
            "average": average, "poor": max(0, average - spread)}


# ===================================================================
# Role label map (multilingual)
# ===================================================================

_ROLE_LABELS: dict[str, dict[str, str]] = {
    "zh": {
        "carry": "1号位",
        "mid": "2号位",
        "offlane": "3号位",
        "support_4": "4号位",
        "support_5": "5号位",
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

# Mean last hits per position in professional matches: the match miner's run
# of 2026-10-10 (100 matches of 7.41f, 200 player-games per position; 198
# reached minute 10, 186 minute 20 and 158 minute 30, so the later columns
# lean towards longer games). This is what the code uses while no pro
# matches were mined on this machine; :func:`_role_cs_curve` lays the means
# of the local run over it. Before that run the table was written by hand:
# its position-3 row said 35 at 10:00 (measured 55; 10 of 16 offlaners of
# eight real replays were rated "excellent" on it) and its position-1 row
# 160 at 20:00 (measured 203).
CS_BENCHMARKS: dict[str, dict[int, int]] = {
    "carry": {5: 24, 10: 65, 15: 127, 20: 203, 25: 278, 30: 349},
    "mid": {5: 25, 10: 57, 15: 96, 20: 137, 25: 183, 30: 228},
    "offlane": {5: 20, 10: 55, 15: 100, 20: 150, 25: 199, 30: 246},
    "support_4": {5: 4, 10: 9, 15: 17, 20: 26, 25: 36, 30: 45},
    "support_5": {5: 3, 10: 7, 15: 14, 20: 21, 25: 29, 30: 37},
}
# The top quarter of the same run at the minutes that are compared: the
# "excellent" line of a row (the mined value of the local run when there is
# one, 115 % of the mean for a minute without either).
CS_TOP_QUARTER: dict[str, dict[int, int]] = {
    "carry": {10: 74, 20: 231, 30: 411},
    "mid": {10: 64, 20: 151, 30: 254},
    "offlane": {10: 64, 20: 174, 30: 285},
}

# Which CS time points to compare for each role. Positions 4 and 5 are not
# judged on last hits at all.
_CS_COMPARE_MINUTES: dict[str, list[int]] = {
    "carry": [10, 20, 30],
    "mid": [10, 20, 30],
    "offlane": [10, 20],
    "support_4": [],
    "support_5": [],
}

# Roles compared on ward placement. The observer supply of a team (2 at the
# start, one more every 135 s) is used up by its two supports, so a core is
# not rated on the wards it did not place.
_WARD_ROLES: set[str] = {"support_4", "support_5"}

# No hero has a last-hit curve of its own. The ten that were written by hand
# ("Anti-Mage 220 at 20:00", "Axe 30 at 10:00", "Juggernaut 150 at 20:00")
# were adjustments of the old role rows: against the measured ones they
# rated a Juggernaut far below the mean of his position as on target, and
# nothing here measures a single hero's last hits (the mined sample is 10-30
# games per hero).

# ===================================================================
# Item timing benchmarks — in seconds from game start
# ===================================================================

# Hand-written guidelines (SOURCE_GUIDELINE), not measured timings.
# ``order`` is the place of the item among the player's major items that the
# timing assumes (1 = first major item, 2 = second, ...). A Blink Dagger
# bought third is not late because it missed the first-item timing, so a
# timing is only compared when the purchase order matches.
ITEM_TIMING_BENCHMARKS: dict[str, dict[str, dict[str, int]]] = {
    "Blink Dagger": {
        "offlane": {"good": 660, "average": 840, "late": 1080, "order": 1},
        "support_4": {"good": 840, "average": 1080, "late": 1380, "order": 1},
        "mid": {"good": 600, "average": 780, "late": 1020, "order": 1},
    },
    "Battle Fury": {
        "carry": {"good": 780, "average": 960, "late": 1200, "order": 1},
    },
    "Black King Bar": {
        "carry": {"good": 1080, "average": 1380, "late": 1740, "order": 2},
        "mid": {"good": 960, "average": 1260, "late": 1560, "order": 2},
        "offlane": {"good": 1200, "average": 1500, "late": 1800, "order": 2},
    },
    "Manta Style": {
        "carry": {"good": 1200, "average": 1500, "late": 1800, "order": 2},
    },
    "Desolator": {
        "carry": {"good": 780, "average": 1020, "late": 1320, "order": 1},
        "mid": {"good": 720, "average": 960, "late": 1260, "order": 1},
    },
    "Aghanim's Scepter": {
        "mid": {"good": 900, "average": 1200, "late": 1560, "order": 1},
        "offlane": {"good": 1080, "average": 1380, "late": 1740, "order": 2},
        "support_4": {"good": 1320, "average": 1680, "late": 2040, "order": 2},
    },
    "Shadow Blade": {
        "mid": {"good": 720, "average": 960, "late": 1260, "order": 1},
        "carry": {"good": 840, "average": 1080, "late": 1380, "order": 1},
    },
    "Radiance": {
        "carry": {"good": 900, "average": 1140, "late": 1440, "order": 1},
    },
    "Assault Cuirass": {
        "carry": {"good": 1500, "average": 1800, "late": 2160, "order": 3},
        "offlane": {"good": 1680, "average": 2040, "late": 2400, "order": 3},
    },
    "Eye of Skadi": {
        "carry": {"good": 1380, "average": 1680, "late": 2040, "order": 2},
    },
}

# A purchase of at least this much gold counts as a major item when the
# purchase order of a timed item is determined.
MAJOR_ITEM_COST = 2000

# Game modes the tables do not describe (gold/XP rates, map or rules differ).
_NON_STANDARD_MODES: frozenset[str] = frozenset({
    "turbo", "ability_draft", "one_v_one_mid", "mid_only", "all_random_death_match",
    "mutation", "event", "custom", "diretide", "greeviling", "tutorial", "intro",
    "bot_challenge",
})

_MODE_LABELS: dict[str, dict[str, str]] = {
    "zh": {
        "turbo": "加速模式", "ability_draft": "技能征召", "one_v_one_mid": "中路单挑",
        "mid_only": "仅中路", "all_random_death_match": "随机死亡竞赛", "mutation": "突变模式",
        "event": "活动模式", "custom": "自定义游戏", "diretide": "夜魇暗潮",
        "greeviling": "贪魔节", "tutorial": "教学", "intro": "教学",
        "bot_challenge": "机器人挑战",
    },
    "en": {
        "turbo": "Turbo", "ability_draft": "Ability Draft", "one_v_one_mid": "1v1 Mid",
        "mid_only": "Mid Only", "all_random_death_match": "All Random Deathmatch",
        "mutation": "Mutation", "event": "event", "custom": "custom game",
        "diretide": "Diretide", "greeviling": "Greeviling", "tutorial": "tutorial", "intro": "tutorial",
        "bot_challenge": "Bot Challenge",
    },
}


def benchmarks_apply(game_mode: str | None) -> bool:
    """Whether the benchmark tables describe matches of this game mode.

    False for Turbo (doubled gold and XP) and the other special modes; an
    unknown mode is treated as a standard match.
    """
    return str(game_mode or "").strip().lower() not in _NON_STANDARD_MODES


def _mode_note(game_mode: str | None, lang: str) -> str:
    """Sentence explaining why a match of this mode gets no comparison."""
    code = str(game_mode or "").strip().lower()
    labels = _MODE_LABELS.get(lang, _MODE_LABELS["zh"])
    label = labels.get(code, code)
    if lang == "en":   # no article before the label ("a Ability Draft match")
        return (f"This match was played in {label} mode: the benchmarks describe standard matches, "
                f"so no comparison is made.")
    return f"本场为{label}比赛：基准数据只适用于常规比赛，因此不做对比。"


# A match this many minor versions away from the benchmarks' (or of another
# major version: 6.78) was played in a different game: GPM, damage and last
# hits of 2013 are not what they are now.
_ERA_MINOR_VERSIONS = 3


def _era_note(match: MatchData, lang: str) -> str:
    """Sentence for a match from a much older (or newer) game version than
    the one the benchmarks describe; "" when the versions are close or one
    of them is unknown."""
    try:
        from . import data_manager
        played = data_manager.patch_major(match.patch) or data_manager.patch_for_time(match.start_time)
        current = data_manager.get_current_patch()
        ours, theirs = (tuple(int(part) for part in version.split(".")[:2]) for version in (current, played))
    except Exception:   # unknown or unusual version text: no claim
        return ""
    if len(ours) != 2 or len(theirs) != 2:
        return ""
    if ours[0] == theirs[0] and abs(ours[1] - theirs[1]) < _ERA_MINOR_VERSIONS:
        return ""
    if lang == "en":
        return (f"Note: this match was played on patch {played}, the benchmarks describe patch {current}: "
                f"read the ratings as a rough guide.")
    return f"注意：这场比赛的版本是{played}，而基准数据描述的是{current}版本的对局，评价仅供参考。"


def _partial_note(match: MatchData, compared: list[str], lang: str) -> str:
    """Sentence saying what of a replay that ends early is compared (*compared*: row labels)."""
    clock = _clock(max(0, match.duration))
    if lang == "en":
        if not compared:
            return (f"The replay ends at {clock}, before the game did: nothing in it can be compared "
                    f"with the benchmarks, which describe complete matches.")
        return (f"The replay ends at {clock}, before the game did: only what the recording covers is "
                f"compared ({', '.join(compared)}). Whole-game figures such as GPM, damage, deaths and "
                f"kill participation are not rated.")
    if not compared:
        return f"录像在{clock}处中断，比赛尚未结束：基准数据描述的是完整比赛，因此不做对比。"
    return (f"录像在{clock}处中断，比赛尚未结束：只对比录像覆盖到的数据（{'、'.join(compared)}），"
            f"GPM、伤害、死亡次数、击杀参与率等全场数据不做评价。")

# ===================================================================
# Per-role performance benchmarks (normalized to 30-minute games)
# ===================================================================

# Hand-written. The same pro matches average 4.3 / 4.7 / 5.3 / 7.3 / 8.3
# deaths per game for positions 1-5 (games of about 35-40 minutes): the
# "average" column is in line with them.
DEATHS_PER_30MIN: dict[str, dict[str, float]] = {
    "carry": {"good": 2.0, "average": 4.0, "poor": 7.0},
    "mid": {"good": 2.5, "average": 4.5, "poor": 7.5},
    "offlane": {"good": 3.0, "average": 5.0, "poor": 8.0},
    "support_4": {"good": 4.0, "average": 6.0, "poor": 9.0},
    "support_5": {"good": 5.0, "average": 7.5, "poor": 11.0},
}

# GPM and hero damage per minute by role: POPULATION data, not a pro
# reference. OpenDota /benchmarks on 2026-10-10 (7.41f), all ranks, standard
# modes, mean over 4-5 typical heroes of the role, rounded: good = 80th
# percentile, average = median, poor = 20th percentile. Used for a hero
# without learned percentiles of its own.
GPM_BENCHMARKS: dict[str, dict[str, int]] = {
    "carry": {"good": 750, "average": 620, "poor": 470},
    "mid": {"good": 650, "average": 520, "poor": 400},
    "offlane": {"good": 570, "average": 480, "poor": 380},
    "support_4": {"good": 450, "average": 370, "poor": 295},
    "support_5": {"good": 410, "average": 350, "poor": 280},
}

DAMAGE_PER_MIN: dict[str, dict[str, int]] = {
    "carry": {"good": 980, "average": 660, "poor": 410},
    "mid": {"good": 930, "average": 660, "poor": 440},
    "offlane": {"good": 750, "average": 560, "poor": 400},
    "support_4": {"good": 675, "average": 490, "poor": 350},
    "support_5": {"good": 550, "average": 405, "poor": 295},
}

# Kill participation in percent, from the same pro matches as CS_BENCHMARKS:
# average = mean of the position, good = its top quarter; poor is 15 points
# under the mean (:func:`_kill_participation_bench` reads the local mining
# run first). The hand-written averages before were 50 / 55 / 50 / 60 / 55.
KILL_PARTICIPATION_PCT: dict[str, dict[str, int]] = {
    "carry": {"good": 69, "average": 58, "poor": 43},
    "mid": {"good": 77, "average": 70, "poor": 55},
    "offlane": {"good": 72, "average": 63, "poor": 48},
    "support_4": {"good": 75, "average": 67, "poor": 52},
    "support_5": {"good": 75, "average": 64, "poor": 49},
}

# Wards placed per 30 minutes. A team gets 2 + 1800 / 135 = 15 observers in
# that time (pro supports place 5-9 each; mined 2026-10-10: 7.2 and 9.9 per
# game for positions 4 and 5), so the two "good" values together stay within
# the supply; 28 sentries (3 + 1800 / 70) leave more room.
WARDS_PER_30MIN: dict[str, dict[str, int]] = {
    "support_4": {"obs_good": 6, "obs_avg": 4, "sen_good": 10, "sen_avg": 6},
    "support_5": {"obs_good": 8, "obs_avg": 6, "sen_good": 12, "sen_avg": 8},
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
    benchmark_source: str = SOURCE_PRO  # "pro_reference" | "guideline" | "opendota_population"
    value_text: str = ""       # actual value, rounded and with its unit (report language)
    benchmark_text: str = ""   # the "average" benchmark in the same form


@dataclass
class ProPatternReport:
    """Full pro-pattern comparison report for one player."""

    hero_name: str
    role: str
    role_label: str       # Role name
    comparisons: list[PatternComparison] = field(default_factory=list)
    summary: str = ""
    overall_rating: str = ""
    applicable: bool = True   # False = nothing can be compared (game mode, cut-off replay; see summary)
    partial: bool = False     # the replay ends before the game: only the rows it covers, no overall rating


# ===================================================================
# Role resolution (positions.py is the single source of truth)
# ===================================================================

_LANE_ROLE_MAP: dict[int, str] = {
    1: "carry",
    2: "mid",
    3: "offlane",
    4: "support_4",
}
_DEFAULT_ROLE = "offlane"


def resolve_role(
    player: PlayerData | None,
    hero_name: str,
    position: int | None = None,
    match: MatchData | None = None,
) -> str:
    """Role tag ("carry", "mid", "offlane", "support_4", "support_5").

    Priority:
    1. ``position`` (1-5) when the caller already derived it;
    2. the position derived from the whole match (``match`` + ``player``);
    3. the player's own lane role, refined by the hero archetype;
    4. the hero archetype from the knowledge base;
    5. ``"offlane"`` as a neutral middle ground.

    Never raises (``player`` may be None).
    """
    try:
        from .positions import (
            hero_typical_position,
            lane_role_of,
            player_position,
            role_for_position,
        )
    except Exception:  # pragma: no cover - defensive
        return _DEFAULT_ROLE
    try:
        role = role_for_position(position)
        if role:
            return role
        if match is not None and player is not None:
            pos = player_position(match, player.player_slot)
            if pos is not None and pos.role:
                return pos.role
        archetype = role_for_position(hero_typical_position(hero_name)) if hero_name else ""
        if player is not None:
            lane_role = lane_role_of(player)
            supportish = archetype in ("support_4", "support_5") or (
                (player.obs_placed or 0) + (player.sen_placed or 0) >= 5
            )
            if lane_role == 2:
                return "mid"
            if lane_role == 1:
                return "support_5" if supportish else "carry"
            if lane_role == 3:
                return "support_4" if supportish else "offlane"
            if lane_role == 4:
                return archetype if archetype in ("support_4", "support_5") else "support_4"
        if archetype:
            return archetype
    except Exception as exc:
        logger.debug("resolve_role failed for %s: %s", hero_name, exc)
    return _DEFAULT_ROLE


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
        "obs_wards": "侦察守卫放置",
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
    """Return expected CS at a given minute for a role.

    The role's curve: the means of the mined pro matches where there are
    any, else the CS_BENCHMARKS table. ``hero_name`` is not read (no hero
    has a curve of its own); it stays in the signature of the callers.
    """
    effective_role = role if role in CS_BENCHMARKS else _DEFAULT_ROLE
    return _interpolate_cs(_role_cs_curve(effective_role), minute)


def get_item_timing_benchmark(item_name: str, role: str) -> dict | None:
    """Return the timing benchmark dict for an item/role combo.

    ``{"good", "average", "late"}`` in seconds plus ``"order"`` (see
    ``ITEM_TIMING_BENCHMARKS``); ``None`` if no benchmark exists for the
    combination.
    """
    item_data = ITEM_TIMING_BENCHMARKS.get(item_name)
    if item_data is None:
        return None
    return item_data.get(role)


def get_benchmark_context(
    hero_name: str, role: str, game_time: int, lang: str = "zh", game_mode: str | None = None,
) -> str:
    """Benchmarks relevant at ``game_time`` as prompt text.

    Last hits of the role and the death table are labelled as pro
    reference; GPM and damage per minute are OpenDota medians over all
    ranks (of the hero when learned, else of typical heroes of the role)
    and say so. Positions 4 and 5 get no last-hit line: they are not
    measured by last hits (a live report added a 4's denies to his last
    hits to "reach" the number).
    With a ``game_mode`` the tables do not describe (Turbo, ...) the text
    only says so.
    """
    minute = max(0, game_time // 60)
    effective_role = role if role in CS_BENCHMARKS else _DEFAULT_ROLE
    role_label = _ROLE_LABELS.get(lang, _ROLE_LABELS["zh"]).get(effective_role, effective_role)
    en = lang == "en"
    hero_text = _hero_text(hero_name, lang)
    lines: list[str] = ["### Benchmark Reference" if en else "### 基准参考"]
    if not benchmarks_apply(game_mode):
        lines.append(f"- {_mode_note(game_mode, lang)}")
        return "\n".join(lines)

    if _CS_COMPARE_MINUTES.get(effective_role):
        role_cs = _interpolate_cs(_role_cs_curve(effective_role), minute)
        lines.append(
            f"- Pro-level CS at {minute} min for {role_label}: ~{int(role_cs)}" if en
            else f"- 职业水平{role_label}在{minute}分钟的补刀: ~{int(role_cs)}"
        )

    for metric, table, en_name, zh_name in (
        ("gold_per_min", GPM_BENCHMARKS, "GPM", "GPM"),
        ("hero_damage_per_min", DAMAGE_PER_MIN, "hero damage per minute", "每分钟英雄伤害"),
    ):
        bench = _metric_bench(hero_name, metric, table, effective_role)
        if not bench:
            continue
        if bench.get("scope") == "hero":
            side = _side_text(bench, lang)
            lines.append(
                f"- {hero_text} {side}: median {en_name} on OpenDota (all ranks, not pro): {bench['average']}" if en
                else f"- {hero_text}{side}在OpenDota全分段的{zh_name}中位数（非职业数据）: {bench['average']}"
            )
        else:
            lines.append(
                f"- Median {en_name} of typical {role_label} heroes on OpenDota (all ranks, not pro): "
                f"{bench['average']}" if en
                else f"- {role_label}常见英雄在OpenDota全分段的{zh_name}中位数（非职业数据）: {bench['average']}"
            )

    deaths_data = DEATHS_PER_30MIN.get(effective_role)
    if deaths_data and minute > 0:
        scaled = deaths_data["average"] * minute / 30
        lines.append(
            f"- Typical deaths by {minute} min (pro reference): <={scaled:.1f}" if en
            else f"- 职业参考：{minute}分钟时死亡次数 <={scaled:.1f}"
        )
    return "\n".join(lines)


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def _per_30(value: float, lang: str, unit_zh: str) -> str:
    """A count normalized to a 30-minute game, with its unit."""
    return f"{value:.1f} per 30 min" if lang == "en" else f"每30分钟{value:.1f}{unit_zh}"


def _item_table() -> dict:
    """Item game data ({} when unavailable); never raises."""
    try:
        from . import game_data
        return game_data.get_item_table() or {}
    except Exception as exc:
        logger.debug("Item table unavailable: %s", exc)
        return {}


def _item_info(identifier: str) -> dict | None:
    try:
        from . import game_data
        return game_data.get_item_data(identifier)
    except Exception:
        return None


def _component_closure(key: str, seen: set[str] | None = None) -> set[str]:
    """Keys of every item that goes into ``key`` (recursively)."""
    seen = seen if seen is not None else set()
    info = _item_info(key) or {}
    for comp in info.get("components") or []:
        comp = str(comp)
        if comp not in seen:
            seen.add(comp)
            _component_closure(comp, seen)
    return seen


def _purchase_order(player: PlayerData, item_event, components_known: bool) -> int | None:
    """Place of a purchase among the player's major items (1 = first).

    Counts the other items of at least ``MAJOR_ITEM_COST`` gold bought before
    it, leaving out its own components and the components of items completed
    by then. None when the order cannot be determined: no cost data, or no
    component data and an earlier big purchase that may be a component.
    """
    info = _item_info(item_event.item_key or item_event.item_name)
    if not info or not info.get("cost"):
        return None
    own_key = str(info.get("key") or item_event.item_key)
    own_parts = _component_closure(own_key)
    earlier: dict[str, int] = {}
    completed: set[str] = set()
    for ev in player.items:
        if ev is item_event or ev.time > item_event.time:
            continue
        ev_info = _item_info(ev.item_key or ev.item_name)
        if not ev_info:
            continue
        key = str(ev_info.get("key") or ev.item_key)
        if key == own_key:
            continue
        completed.add(key)
        cost = ev_info.get("cost") or 0
        if ev.time < item_event.time and cost >= MAJOR_ITEM_COST and key not in own_parts:
            earlier[key] = cost
    for key in list(earlier):
        if any(key in _component_closure(other) for other in completed if other != key):
            del earlier[key]
    if earlier and not components_known:
        return None
    return len(earlier) + 1


def compare_player(
    player: PlayerData, match: MatchData, lang: str = "zh", position: int | None = None,
) -> ProPatternReport:
    """Compare a player's performance against benchmarks.

    Evaluates CS, deaths, GPM, hero damage, item timings, ward placement,
    and kill participation. ``position`` (1-5) is the derived position from
    :mod:`positions`; it is derived from ``match`` when omitted.

    A match in a game mode the tables do not describe (Turbo, ...) gets no
    comparisons: ``applicable`` is False and the summary says why.

    Of a replay that ends before the game does (``match.partial``) only the
    rows the recording covers are compared — last hits at the minutes it
    reached and item timings — and there is no overall rating: the other
    rows are rates over a whole game (the first 11 minutes of a mid player
    were rated "GPM 377 against a median of 520: poor"). ``partial`` is
    True and the summary says what was compared; without any such row
    ``applicable`` is False.
    """
    hero_name = player.hero_name
    role = resolve_role(player, hero_name, position=position, match=match)
    role_labels = _ROLE_LABELS.get(lang, _ROLE_LABELS["zh"])
    role_label = role_labels.get(role, role)
    effective_role = role if role in CS_BENCHMARKS else _DEFAULT_ROLE
    mlabels = _METRIC_LABELS.get(lang, _METRIC_LABELS["zh"])
    en = lang == "en"
    hero_text = _hero_text(hero_name, lang)

    report = ProPatternReport(
        hero_name=hero_name,
        role=role,
        role_label=role_label,
    )
    if not benchmarks_apply(match.game_mode):
        report.applicable = False
        report.summary = _mode_note(match.game_mode, lang)
        return report
    report.partial = bool(getattr(match, "partial", False))
    whole_game = not report.partial     # rows that are rates over a complete game

    duration_min = max(match.duration / 60, 1.0)
    scale_30 = 30.0 / duration_min  # scale factor to normalize to 30 min

    # ------------------------------------------------------------------
    # 1. CS comparison at role-appropriate time points
    # ------------------------------------------------------------------
    cs_minutes = _CS_COMPARE_MINUTES.get(effective_role, [10, 20, 30])
    if player.lh_t and cs_minutes:
        # The mean of the position in pro matches is the benchmark, the top
        # quarter of the same matches "excellent" (of the local mining run
        # when that minute was mined, else of the run the table holds).
        mined = _mined_cs(effective_role)
        for target_min in cs_minutes:
            idx = target_min  # lh_t is indexed by minute
            if idx < len(player.lh_t):
                actual_cs = player.lh_t[idx]
                expected = get_cs_benchmark(hero_name, role, target_min)
                if target_min in mined:
                    top = mined[target_min][1]
                else:
                    top = CS_TOP_QUARTER.get(effective_role, {}).get(target_min)
                # 115% of expected = excellent where no top quarter is known
                good_cs = round(top if top is not None and top > expected else expected * 1.15, 1)
                avg_cs = round(expected, 1)
                poor_cs = round(expected * 0.6, 1)

                rating = _rate_higher_is_better(actual_cs, good_cs, avg_cs, poor_cs)
                # The sentence quotes the benchmark as the table shows it
                # (``benchmark_text``) and the difference to that number:
                # truncating here gave "117 below benchmark (202)" under a
                # column that read 203.
                shown = f"{avg_cs:.0f}"
                diff = actual_cs - int(shown)
                if en:
                    if diff >= 0:
                        desc = f"{actual_cs} CS at {target_min} min, {diff} above benchmark ({shown})"
                    else:
                        desc = f"{actual_cs} CS at {target_min} min, {abs(diff)} below benchmark ({shown})"
                else:
                    if diff >= 0:
                        desc = f"{target_min}分钟补刀{actual_cs}，高于基准{shown}约{diff}刀"
                    else:
                        desc = f"{target_min}分钟补刀{actual_cs}，低于基准{shown}约{abs(diff)}刀"

                report.comparisons.append(PatternComparison(
                    metric=f"cs_{target_min}min",
                    metric_label=f"{target_min}{mlabels['cs']}",
                    actual_value=actual_cs,
                    benchmark_good=good_cs,
                    benchmark_avg=avg_cs,
                    benchmark_poor=poor_cs,
                    rating=rating,
                    description=desc,
                    value_text=str(actual_cs),
                    benchmark_text=shown,
                ))

    # ------------------------------------------------------------------
    # 2. Deaths comparison (lower is better)
    # ------------------------------------------------------------------
    deaths_bench = DEATHS_PER_30MIN.get(effective_role) if whole_game else None
    if deaths_bench:
        normalized_deaths = player.deaths * scale_30
        rating = _rate_lower_is_better(
            normalized_deaths,
            deaths_bench["good"],
            deaths_bench["average"],
            deaths_bench["poor"],
        )
        rating_disp = _get_rating_label(rating, lang)

        if en:
            desc_parts = [f"{player.deaths} death{'' if player.deaths == 1 else 's'} total"]
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
            actual_value=round(normalized_deaths, 1),
            benchmark_good=deaths_bench["good"],
            benchmark_avg=deaths_bench["average"],
            benchmark_poor=deaths_bench["poor"],
            rating=rating,
            description="".join(desc_parts),
            value_text=_per_30(normalized_deaths, lang, "次"),
            benchmark_text=_per_30(deaths_bench["average"], lang, "次"),
        ))

    # ------------------------------------------------------------------
    # 3. GPM comparison (population data: the hero's own percentiles when
    #    learned, else the all-rank role table)
    # ------------------------------------------------------------------
    gpm_bench = _metric_bench(hero_name, "gold_per_min", GPM_BENCHMARKS, effective_role) if whole_game else None
    if gpm_bench:
        gpm = player.gold_per_min
        rating = _rate_higher_is_better(
            gpm, gpm_bench["good"], gpm_bench["average"], gpm_bench["poor"]
        )
        rating_disp = _get_rating_label(rating, lang)
        ref = _population_ref(gpm_bench, hero_text, role_label, lang)
        desc = f"GPM {gpm}, {ref}, rating: {rating_disp}" if en else f"GPM {gpm}，{ref}，评价：{rating_disp}"
        report.comparisons.append(PatternComparison(
            metric="gpm",
            metric_label=mlabels["gpm"],
            actual_value=gpm,
            benchmark_good=gpm_bench["good"],
            benchmark_avg=gpm_bench["average"],
            benchmark_poor=gpm_bench["poor"],
            rating=rating,
            description=desc,
            benchmark_source=gpm_bench.get("source", SOURCE_POPULATION),
            value_text=str(gpm),
            benchmark_text=str(gpm_bench["average"]),
        ))

    # ------------------------------------------------------------------
    # 4. Hero damage per minute (population data, like GPM)
    # ------------------------------------------------------------------
    dmg_bench = (_metric_bench(hero_name, "hero_damage_per_min", DAMAGE_PER_MIN, effective_role)
                 if whole_game else None)
    if dmg_bench:
        dpm = player.hero_damage / duration_min if duration_min > 0 else 0
        rating = _rate_higher_is_better(
            dpm, dmg_bench["good"], dmg_bench["average"], dmg_bench["poor"]
        )
        rating_disp = _get_rating_label(rating, lang)
        ref = _population_ref(dmg_bench, hero_text, role_label, lang)
        desc = (f"Damage/min {dpm:.0f}, {ref}, rating: {rating_disp}" if en
                else f"每分钟伤害{dpm:.0f}，{ref}，评价：{rating_disp}")
        report.comparisons.append(PatternComparison(
            metric="damage_per_min",
            metric_label=mlabels["damage_per_min"],
            actual_value=round(dpm, 1),
            benchmark_good=dmg_bench["good"],
            benchmark_avg=dmg_bench["average"],
            benchmark_poor=dmg_bench["poor"],
            rating=rating,
            description=desc,
            benchmark_source=dmg_bench.get("source", SOURCE_POPULATION),
            value_text=f"{dpm:.0f}",
            benchmark_text=str(dmg_bench["average"]),
        ))

    # ------------------------------------------------------------------
    # 5. Kill participation
    # ------------------------------------------------------------------
    kp_bench = _kill_participation_bench(effective_role) if whole_game else None
    if kp_bench:
        team_kills = 0
        for p in match.players:
            if p.team == player.team:
                team_kills += p.kills
        if team_kills > 0:
            kp = min(100.0, (player.kills + player.assists) / team_kills * 100)
        else:
            kp = 0.0
        rating = _rate_higher_is_better(
            kp, kp_bench["good"], kp_bench["average"], kp_bench["poor"]
        )
        rating_disp = _get_rating_label(rating, lang)
        if en:
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
            value_text=f"{kp:.1f}%",
            benchmark_text=f"{kp_bench['average']}%",
        ))

    # ------------------------------------------------------------------
    # 6. Ward placement (positions 4 and 5 only)
    # ------------------------------------------------------------------
    ward_bench = WARDS_PER_30MIN.get(effective_role) if effective_role in _WARD_ROLES and whole_game else None
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
        if en:
            obs_desc = (
                f"Placed {player.obs_placed} observer wards"
                f" (normalized to 30 min: {obs_scaled:.1f}),"
                f" benchmark {ward_bench['obs_avg']}+, rating: {obs_rating_disp}"
            )
        else:
            obs_desc = (
                f"放置{player.obs_placed}个侦察守卫"
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
            value_text=_per_30(obs_scaled, lang, "个"),
            benchmark_text=_per_30(ward_bench["obs_avg"], lang, "个"),
        ))

        # Sentry wards
        sen_rating = _rate_higher_is_better(
            sen_scaled,
            ward_bench["sen_good"],
            ward_bench["sen_avg"],
            None,
        )
        sen_rating_disp = _get_rating_label(sen_rating, lang)
        if en:
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
            value_text=_per_30(sen_scaled, lang, "个"),
            benchmark_text=_per_30(ward_bench["sen_avg"], lang, "个"),
        ))

    # ------------------------------------------------------------------
    # 7. Item timing comparisons (only in the purchase order a timing assumes)
    # ------------------------------------------------------------------
    components_known = any(info.get("components") for info in _item_table().values())
    timed_items: set[str] = set()
    for item_event in sorted(player.items, key=lambda i: i.time):
        item_name = item_event.item_name
        if item_name in timed_items:
            continue  # only the first purchase counts
        bench = get_item_timing_benchmark(item_name, effective_role)
        if bench is None:
            continue
        timed_items.add(item_name)
        if _purchase_order(player, item_event, components_known) != bench.get("order", 1):
            continue  # bought earlier or later in the build than the timing assumes

        actual_time = item_event.time
        good_t = bench["good"]
        avg_t = bench["average"]
        late_t = bench["late"]

        # For item timings, lower (faster) is better
        rating = _rate_lower_is_better(actual_time, good_t, avg_t, late_t)

        rating_disp = _get_rating_label(rating, lang)
        if en:
            exc_label = _get_rating_label("excellent", lang)
            avg_label = _get_rating_label("average", lang)
            desc = (
                f"{item_name} completed at {_clock(actual_time)}, "
                f"guideline {_clock(good_t)} ({exc_label})/"
                f"{_clock(avg_t)} ({avg_label}), "
                f"rating: {rating_disp}"
            )
            item_metric_label = f"{item_name} {mlabels['item_timing_suffix']}"
        else:
            item_text = _item_text(item_name, lang)
            desc = (
                f"{item_text}在{_clock(actual_time)}完成，"
                f"参考时间{_clock(good_t)}（优秀）/"
                f"{_clock(avg_t)}（一般），"
                f"评价：{rating_disp}"
            )
            item_metric_label = f"{item_text}{mlabels['item_timing_suffix']}"

        report.comparisons.append(PatternComparison(
            metric=_item_metric(item_name),
            metric_label=item_metric_label,
            actual_value=actual_time,
            benchmark_good=good_t,
            benchmark_avg=avg_t,
            benchmark_poor=late_t,
            rating=rating,
            description=desc,
            benchmark_source=SOURCE_GUIDELINE,
            value_text=_clock(actual_time),
            benchmark_text=_clock(avg_t),
        ))

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------
    if report.partial:
        # A handful of rows of the first minutes is no overall rating.
        report.applicable = bool(report.comparisons)
        report.summary = _partial_note(match, [c.metric_label for c in report.comparisons], lang)
        return report
    _compute_summary(report, lang=lang)
    era = _era_note(match, lang)
    if era and report.comparisons:
        report.summary = f"{report.summary} {era}" if lang == "en" else report.summary + era

    return report


def _compute_summary(report: ProPatternReport, lang: str = "zh") -> None:
    """Compute overall_rating and summary for a ProPatternReport."""
    hero_text = _hero_text(report.hero_name, lang)
    if not report.comparisons:
        report.overall_rating = "average"
        if lang == "en":
            report.summary = f"{hero_text} ({report.role_label}): insufficient data for a comprehensive evaluation."
        else:
            report.summary = f"{hero_text}（{report.role_label}）的数据不足，无法给出全面评价。"
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
    strengths: list[PatternComparison] = []
    weaknesses: list[PatternComparison] = []
    for c in report.comparisons:
        if c.rating in ("excellent", "good"):
            strengths.append(c)
        elif c.rating in ("below_average", "poor"):
            weaknesses.append(c)

    # "Pro" is only claimed when every row behind the sentence is compared
    # with the pro reference; all-rank OpenDota medians are just "benchmarks".
    weak_shown = weaknesses[:3]
    pro_weak = all(c.benchmark_source == SOURCE_PRO for c in weak_shown)
    pro_all = all(c.benchmark_source == SOURCE_PRO for c in report.comparisons)
    if lang == "en":
        sep = ", "
        parts = [f"{hero_text} ({report.role_label}) overall performance: {overall_disp}. "]
        if strengths:
            parts.append(f"Highlights: {sep.join(c.metric_label for c in strengths[:3])} stood out. ")
        if weaknesses:
            level = "the pro reference" if pro_weak else "the benchmark"
            parts.append(f"Needs improvement: {sep.join(c.metric_label for c in weak_shown)} below {level}. ")
        if not strengths and not weaknesses:
            parts.append("All metrics are close to the pro reference. " if pro_all
                         else "All metrics are close to the benchmarks. ")
    else:
        sep = "、"
        parts = [f"{hero_text}（{report.role_label}）整体表现{overall_disp}。"]
        if strengths:
            parts.append(f"亮点：{sep.join(c.metric_label for c in strengths[:3])}表现出色。")
        if weaknesses:
            level = "职业参考水准" if pro_weak else "基准水平"
            parts.append(f"需要改进：{sep.join(c.metric_label for c in weak_shown)}低于{level}。")
        if not strengths and not weaknesses:
            parts.append("各项数据接近职业参考水准。" if pro_all else "各项数据接近基准水平。")

    report.summary = "".join(parts).strip()


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
        "applicable": report.applicable,
        "partial": report.partial,
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
                "benchmark_source": c.benchmark_source,
                "value_text": c.value_text,
                "benchmark_text": c.benchmark_text,
            }
            for c in report.comparisons
        ],
    }
