"""Level 1 learner: statistical self-learning from OpenDota aggregate endpoints.

Zero LLM cost. For every hero it pulls three aggregate endpoints:

- ``/benchmarks?hero_id=N``        -> per-metric percentiles (GPM, XPM, ...)
- ``/scenarios/itemTimings?hero_id=N`` -> win rate when an item is completed
  before a given time bucket
- ``/heroes/{id}/matchups``        -> per-opponent win rates

and renders them into bilingual template chunks (one LearnedChunk per
language). Raw percentile values are additionally persisted to
``LEARNED_DIR / "_benchmarks_raw.json"`` in machine-readable form so that
``pro_patterns`` can overlay live values onto its hardcoded benchmark tables.

Budget: ~3 API calls per hero; a full run over ~125 heroes is ~375 calls
(~7 minutes with the 1.1s rate floor in ``opendota_get``). Use
``heroes_limit`` for tests.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from ...config import GAMEDATA_DIR
from ..data_manager import load_heroes, load_items
from .base import (
    LEARNED_DIR,
    LearnedChunk,
    current_patch_version,
    opendota_get,
    save_learned_file,
)

logger = logging.getLogger(__name__)

__all__ = ["learn_stats", "BENCHMARKS_RAW_FILE"]

# Machine-readable percentile values consumed by pro_patterns (not chunks)
BENCHMARKS_RAW_FILE = "_benchmarks_raw.json"

_MIN_ITEM_GAMES = 30       # itemTimings rows below this sample size are noise
_MIN_MATCHUP_GAMES = 50    # matchup rows below this sample size are noise
_TOP_ITEM_INSIGHTS = 4     # item timing insights kept per hero
_TOP_MATCHUPS = 3          # worst 3 + best 3 matchups per hero
_MAX_CONSECUTIVE_ERRORS = 5  # abort early when the API looks unreachable

# Benchmark metrics: (api_key, zh label, en label, display decimals)
_BENCH_METRICS: list[tuple[str, str, str, int]] = [
    ("gold_per_min", "GPM", "GPM", 0),
    ("xp_per_min", "XPM", "XPM", 0),
    ("last_hits_per_min", "每分钟补刀", "last hits/min", 1),
    ("hero_damage_per_min", "每分钟英雄伤害", "hero damage/min", 0),
    ("kills_per_min", "每分钟击杀", "kills/min", 2),
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    """Hero name -> tag slug; must match knowledge_chunks._hero_slug."""
    return name.lower().replace(" ", "_").replace("'", "").replace("-", "_")


def _fmt(value: float, decimals: int) -> str:
    """Format a metric value with a fixed number of decimals (0 -> int)."""
    if decimals <= 0:
        return str(int(round(value)))
    return f"{value:.{decimals}f}"


def _fmt_minutes(seconds: float) -> str:
    """Seconds -> minute string: 450 -> '7.5', 900 -> '15'."""
    return f"{seconds / 60:g}"


def _percentile_value(rows: list[dict], target: float) -> float | None:
    """Value at the percentile closest to *target* (None if unavailable)."""
    best: tuple[float, float] | None = None  # (distance, value)
    for row in rows or []:
        try:
            p = float(row["percentile"])
            v = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        dist = abs(p - target)
        if best is None or dist < best[0]:
            best = (dist, v)
    # Reject matches that are not really the requested percentile
    if best is None or best[0] > 0.05:
        return None
    return best[1]


def _hero_catalog() -> list[tuple[int, str, str]]:
    """Return [(hero_id, localized_name, slug)] sorted by hero id.

    Names come from data_manager.load_heroes(); ids from the raw gamedata
    file (load_heroes drops them). Empty list when gamedata is unavailable.
    """
    try:
        heroes = load_heroes()
        with open(GAMEDATA_DIR / "heroes.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        logger.warning("Hero catalog unavailable (run data update first): %s", exc)
        return []
    catalog: list[tuple[int, str, str]] = []
    for name in heroes:
        hero_id = (raw.get(name) or {}).get("id")
        if isinstance(hero_id, int):
            catalog.append((hero_id, name, _slug(name)))
    catalog.sort(key=lambda t: t[0])
    return catalog


def _item_display_map() -> dict[str, str]:
    """internal item key (e.g. 'bfury') -> display name (e.g. 'Battle Fury')."""
    try:
        return {
            data["key"]: name
            for name, data in load_items().items()
            if data.get("key")
        }
    except Exception as exc:
        logger.warning("Item name map unavailable, using raw keys: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Chunk builders (one LearnedChunk per language per insight)
# ---------------------------------------------------------------------------

def _benchmark_chunks(
    hero_name: str,
    slug: str,
    hero_id: int,
    result: dict,
    patch: str,
    langs: tuple[str, ...],
) -> tuple[list[LearnedChunk], dict]:
    """Render /benchmarks percentiles into chunks + raw metric dict."""
    metrics: dict[str, dict[str, float]] = {}
    for key, _zh, _en, _nd in _BENCH_METRICS:
        rows = result.get(key) or []
        p10 = _percentile_value(rows, 0.1)
        p50 = _percentile_value(rows, 0.5)
        p90 = _percentile_value(rows, 0.9)
        if p50 is None or p90 is None:
            continue
        entry = {"p50": p50, "p90": p90}
        if p10 is not None:
            entry["p10"] = p10
        metrics[key] = entry
    if not metrics:
        return [], {}

    chunks: list[LearnedChunk] = []
    for lang in langs:
        parts: list[str] = []
        for key, zh_label, en_label, nd in _BENCH_METRICS:
            m = metrics.get(key)
            if not m:
                continue
            if lang == "en":
                parts.append(
                    f"median {en_label} {_fmt(m['p50'], nd)}, "
                    f"top 10% reach {_fmt(m['p90'], nd)}"
                )
            else:
                parts.append(
                    f"{zh_label}中位数 {_fmt(m['p50'], nd)}，"
                    f"前10%玩家达到 {_fmt(m['p90'], nd)}"
                )
        if lang == "en":
            content = (
                f"{hero_name} performance benchmarks (OpenDota aggregate stats): "
                + "; ".join(parts)
                + ". A metric clearly below the median is the first thing to improve."
            )
        else:
            content = (
                f"{hero_name} 数据基准（OpenDota 全量统计）："
                + "；".join(parts)
                + "。某项数据明显低于中位数时，应将其作为优先改进方向。"
            )
        chunks.append(LearnedChunk(
            chunk_id=f"learned:benchmarks:{slug}",
            content=content,
            lang=lang,
            category="economy_patterns",
            tags=[f"hero:{slug}"],
            priority=2,
            source="opendota_stats",
            patch_version=patch,
            provenance={"endpoint": f"/benchmarks?hero_id={hero_id}"},
        ))
    return chunks, metrics


def _item_timing_chunks(
    hero_name: str,
    slug: str,
    hero_id: int,
    rows: list[dict],
    item_names: dict[str, str],
    patch: str,
    langs: tuple[str, ...],
) -> list[LearnedChunk]:
    """Render /scenarios/itemTimings rows into top-N timing insight chunks."""
    # Best (win rate, sample) time bucket per item, min sample size applied
    best_by_item: dict[str, tuple[float, int, int]] = {}  # item -> (wr, games, time)
    for row in rows or []:
        try:
            item = str(row.get("item") or "")
            games = int(row.get("games") or 0)
            wins = int(row.get("wins") or 0)
            bucket = int(row.get("time") or 0)
        except (TypeError, ValueError):
            continue
        if not item or games < _MIN_ITEM_GAMES or bucket <= 0:
            continue
        winrate = wins / games * 100
        current = best_by_item.get(item)
        if current is None or (winrate, games) > (current[0], current[1]):
            best_by_item[item] = (winrate, games, bucket)

    top = sorted(
        best_by_item.items(), key=lambda kv: (kv[1][0], kv[1][1]), reverse=True
    )[:_TOP_ITEM_INSIGHTS]

    chunks: list[LearnedChunk] = []
    for item, (winrate, games, bucket) in top:
        display = item_names.get(item) or item.replace("_", " ").title()
        minutes = _fmt_minutes(bucket)
        for lang in langs:
            if lang == "en":
                content = (
                    f"{hero_name}: completing {display} before {minutes} min "
                    f"gives a {winrate:.1f}% win rate ({games} games, OpenDota). "
                    f"Plan your farming route around hitting this timing."
                )
            else:
                content = (
                    f"{hero_name}：{display} 在 {minutes} 分钟前做出时"
                    f"胜率 {winrate:.1f}%（样本 {games} 场，OpenDota 统计）。"
                    f"规划打钱路线时尽量卡住这个装备时间点。"
                )
            chunks.append(LearnedChunk(
                chunk_id=f"learned:item_timing:{slug}:{item}",
                content=content,
                lang=lang,
                category="item_coaching",
                tags=[f"hero:{slug}", f"item:{item}"],
                priority=2,
                source="opendota_stats",
                patch_version=patch,
                provenance={
                    "endpoint": f"/scenarios/itemTimings?hero_id={hero_id}",
                    "games": games,
                    "winrate": round(winrate, 1),
                    "before_seconds": bucket,
                },
            ))
    return chunks


def _matchup_chunks(
    hero_name: str,
    slug: str,
    hero_id: int,
    rows: list[dict],
    id_map: dict[int, tuple[str, str]],
    patch: str,
    langs: tuple[str, ...],
) -> list[LearnedChunk]:
    """Render /heroes/{id}/matchups into worst-3 / best-3 matchup chunks."""
    ranked: list[tuple[float, int, int]] = []  # (winrate, games, opponent_id)
    for row in rows or []:
        try:
            opponent_id = int(row.get("hero_id") or 0)
            games = int(row.get("games_played") or 0)
            wins = int(row.get("wins") or 0)
        except (TypeError, ValueError):
            continue
        if games < _MIN_MATCHUP_GAMES or opponent_id not in id_map:
            continue
        ranked.append((wins / games * 100, games, opponent_id))
    ranked.sort(key=lambda t: (t[0], t[1]))

    worst = ranked[:_TOP_MATCHUPS]
    remaining = ranked[_TOP_MATCHUPS:]
    best = list(reversed(remaining[-_TOP_MATCHUPS:]))

    chunks: list[LearnedChunk] = []
    for kind, entries in (("worst", worst), ("best", best)):
        for winrate, games, opponent_id in entries:
            opp_name, opp_slug = id_map[opponent_id]
            for lang in langs:
                if lang == "en":
                    if kind == "worst":
                        content = (
                            f"{hero_name} vs {opp_name}: only a {winrate:.1f}% win rate "
                            f"({games} games, OpenDota) — an unfavorable matchup. Avoid "
                            f"solo trades in lane, ask teammates to rotate, or consider "
                            f"a lane swap."
                        )
                    else:
                        content = (
                            f"{hero_name} vs {opp_name}: a {winrate:.1f}% win rate "
                            f"({games} games, OpenDota) — a favorable matchup. Play "
                            f"proactively and press the advantage."
                        )
                else:
                    if kind == "worst":
                        content = (
                            f"{hero_name} 对阵 {opp_name} 胜率仅 {winrate:.1f}%"
                            f"（样本 {games} 场，OpenDota 统计），属于劣势对局："
                            f"对线期避免单独硬拼，优先叫队友游走支援或考虑换线。"
                        )
                    else:
                        content = (
                            f"{hero_name} 对阵 {opp_name} 胜率达 {winrate:.1f}%"
                            f"（样本 {games} 场，OpenDota 统计），属于优势对局："
                            f"可以更主动地压制对方、扩大优势。"
                        )
                chunks.append(LearnedChunk(
                    chunk_id=f"learned:matchup:{slug}:{opp_slug}",
                    content=content,
                    lang=lang,
                    category="hero_matchup",
                    tags=[
                        f"hero:{slug}",
                        f"hero:{opp_slug}",
                        f"matchup:{slug}:{opp_slug}",
                    ],
                    priority=2,
                    source="opendota_stats",
                    patch_version=patch,
                    provenance={
                        "endpoint": f"/heroes/{hero_id}/matchups",
                        "games": games,
                        "winrate": round(winrate, 1),
                        "kind": kind,
                    },
                ))
    return chunks


# ---------------------------------------------------------------------------
# Raw benchmark persistence (for pro_patterns overlay)
# ---------------------------------------------------------------------------

def _write_benchmarks_raw(raw: dict[str, dict], patch: str) -> None:
    """Write machine-readable percentiles to LEARNED_DIR/_benchmarks_raw.json."""
    path = LEARNED_DIR / BENCHMARKS_RAW_FILE
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "patch": patch,
            "hero_count": len(raw),
        },
        "heroes": raw,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    logger.info("Raw benchmarks written: %s (%d heroes)", path.name, len(raw))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def learn_stats(
    langs: tuple[str, ...] = ("zh", "en"),
    heroes_limit: int | None = None,
) -> dict:
    """Learn benchmark / item timing / matchup knowledge from OpenDota.

    Returns ``{"chunks_written": int, "heroes": int, "api_calls": int}``.
    Degrades gracefully: on missing gamedata or network failure it logs a
    warning and returns a 0-chunk summary without touching previous output.
    """
    summary = {"chunks_written": 0, "heroes": 0, "api_calls": 0}
    try:
        catalog = _hero_catalog()
        if not catalog:
            logger.warning("learn_stats: no hero catalog available, skipping run")
            return summary
        id_map = {hid: (name, slug) for hid, name, slug in catalog}
        if heroes_limit is not None:
            catalog = catalog[:heroes_limit]

        item_names = _item_display_map()
        patch = current_patch_version()
        chunks: list[LearnedChunk] = []
        raw_benchmarks: dict[str, dict] = {}
        consecutive_errors = 0

        for hero_id, hero_name, slug in catalog:
            if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                logger.warning(
                    "learn_stats: %d consecutive API errors, aborting early",
                    consecutive_errors,
                )
                break
            hero_ok = False

            # a) percentile benchmarks
            try:
                summary["api_calls"] += 1
                data = opendota_get("benchmarks", {"hero_id": hero_id})
                bench_chunks, metrics = _benchmark_chunks(
                    hero_name, slug, hero_id, data.get("result") or {}, patch, langs
                )
                chunks.extend(bench_chunks)
                if metrics:
                    raw_benchmarks[slug] = {
                        "hero": hero_name,
                        "hero_id": hero_id,
                        **metrics,
                    }
                consecutive_errors = 0
                hero_ok = True
            except Exception as exc:
                consecutive_errors += 1
                logger.warning("learn_stats: benchmarks failed for %s: %s", hero_name, exc)

            # b) item timing scenarios
            try:
                summary["api_calls"] += 1
                rows = opendota_get("scenarios/itemTimings", {"hero_id": hero_id})
                chunks.extend(_item_timing_chunks(
                    hero_name, slug, hero_id, rows, item_names, patch, langs
                ))
                consecutive_errors = 0
                hero_ok = True
            except Exception as exc:
                consecutive_errors += 1
                logger.warning("learn_stats: itemTimings failed for %s: %s", hero_name, exc)

            # c) matchups
            try:
                summary["api_calls"] += 1
                rows = opendota_get(f"heroes/{hero_id}/matchups")
                chunks.extend(_matchup_chunks(
                    hero_name, slug, hero_id, rows, id_map, patch, langs
                ))
                consecutive_errors = 0
                hero_ok = True
            except Exception as exc:
                consecutive_errors += 1
                logger.warning("learn_stats: matchups failed for %s: %s", hero_name, exc)

            if hero_ok:
                summary["heroes"] += 1

        if not chunks:
            logger.warning(
                "learn_stats: produced no chunks, keeping previous learned files"
            )
            return summary

        save_learned_file("stats_opendota", chunks, meta={
            "patch": patch,
            "heroes": summary["heroes"],
            "api_calls": summary["api_calls"],
            "langs": list(langs),
        })
        if raw_benchmarks:
            _write_benchmarks_raw(raw_benchmarks, patch)
        summary["chunks_written"] = len(chunks)
        return summary
    except Exception as exc:
        logger.warning("learn_stats failed: %s", exc)
        return summary


if __name__ == "__main__":  # pragma: no cover — manual CLI, run from backend/
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(learn_stats(heroes_limit=limit))
