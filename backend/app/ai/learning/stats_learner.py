"""Level 1 learner: statistical self-learning from OpenDota aggregate endpoints.

Zero LLM cost. For every hero it pulls three aggregate endpoints:

- ``/benchmarks?hero_id=N``        -> per-metric percentiles (GPM, XPM, ...)
- ``/scenarios/itemTimings?hero_id=N`` -> win rate when an item is completed
  before a given time bucket
- ``/heroes/{id}/matchups``        -> per-opponent win rates

and renders them into bilingual chunks (one LearnedChunk per language) that
state *statistics* only — they are labelled as such when injected and are
merged into the editorial matchup tip for the same pair instead of competing
with it. Raw percentile values are additionally persisted to
``<output_dir> / "_benchmarks_raw.json"`` in machine-readable form so that
``pro_patterns`` can overlay live values onto its hardcoded benchmark tables.

The hero list comes from ``data_manager.get_hero_catalog()``, which works
without fetched game data (the frontend catalog is committed), so a
concurrent / failed game-data update can no longer leave this learner
without heroes.

Budget: ~3 API calls per hero; a full run over ~125 heroes is ~375 calls
(~7 minutes with the 1.1s rate floor in ``opendota_get``). Use
``heroes_limit`` for tests.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .. import data_manager
from .base import (
    LEARNED_DIR,
    LearnedChunk,
    current_patch_version,
    load_learned_file,
    opendota_get,
    save_learned_file,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

__all__ = ["learn_stats", "BENCHMARKS_RAW_FILE", "LEARNED_FILE_NAME"]

# Machine-readable percentile values consumed by pro_patterns (not chunks)
BENCHMARKS_RAW_FILE = "_benchmarks_raw.json"
LEARNED_FILE_NAME = "stats_opendota"

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

def _tag_slug(name: str) -> str:
    """Canonical hero tag slug (shared with the knowledge store)."""
    return data_manager.hero_slug(name)


def _bench_key(name: str) -> str:
    """Key of a hero in _benchmarks_raw.json; must match pro_patterns' lookup."""
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
    """Return ``[(hero_id, display_name, tag_slug)]`` sorted by hero id."""
    try:
        return [(h["id"], h["name"], h["slug"]) for h in data_manager.get_hero_catalog()]
    except Exception as exc:
        logger.warning("Hero catalog unavailable: %s", exc)
        return []


def _item_display_name(key: str) -> str:
    """internal item key (e.g. 'bfury') -> display name (e.g. 'Battle Fury')."""
    try:
        name = data_manager.resolve_item_name(key)
    except Exception:
        name = None
    return name or key.replace("_", " ").title()


def _chunk_hero(chunk_id: str) -> str:
    """Hero slug encoded in a stats chunk id ("learned:<kind>:<hero>:...")."""
    parts = chunk_id.split(":")
    return parts[2] if len(parts) > 2 else ""


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
                parts.append(f"median {en_label} {_fmt(m['p50'], nd)}, top 10% {_fmt(m['p90'], nd)}")
            else:
                parts.append(f"{zh_label}中位数 {_fmt(m['p50'], nd)}，前10%达到 {_fmt(m['p90'], nd)}")
        if lang == "en":
            content = (
                f"{hero_name} performance benchmarks (all ranks): " + "; ".join(parts)
                + ". A metric clearly below the median is the first thing to improve."
            )
        else:
            content = (
                f"{hero_name} 数据基准（全分段）：" + "；".join(parts)
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
    patch: str,
    langs: tuple[str, ...],
) -> list[LearnedChunk]:
    """Render /scenarios/itemTimings rows into top-N timing insight chunks."""
    # Best (win rate, sample) time bucket per item, min sample size applied
    best_by_item: dict[str, tuple[float, int, int]] = {}  # item -> (wr, games, time)
    for row in rows or []:
        if not isinstance(row, dict):
            continue
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
        best_by_item.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[0]), reverse=True
    )[:_TOP_ITEM_INSIGHTS]

    chunks: list[LearnedChunk] = []
    for item, (winrate, games, bucket) in top:
        item_slug = data_manager.item_key(item)
        display = _item_display_name(item_slug)
        minutes = _fmt_minutes(bucket)
        for lang in langs:
            if lang == "en":
                content = (
                    f"{hero_name}: games where {display} was completed before {minutes} min "
                    f"had a {winrate:.1f}% win rate ({games} games)."
                )
            else:
                content = (
                    f"{hero_name}：{minutes} 分钟前做出 {display} 的对局胜率 "
                    f"{winrate:.1f}%（样本 {games} 场）。"
                )
            chunks.append(LearnedChunk(
                chunk_id=f"learned:item_timing:{slug}:{item_slug}",
                content=content,
                lang=lang,
                category="item_coaching",
                tags=[f"hero:{slug}", f"item:{item_slug}"],
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
    """Render /heroes/{id}/matchups into worst-3 / best-3 matchup statistics."""
    ranked: list[tuple[float, int, int]] = []  # (winrate, games, opponent_id)
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        try:
            opponent_id = int(row.get("hero_id") or 0)
            games = int(row.get("games_played") or 0)
            wins = int(row.get("wins") or 0)
        except (TypeError, ValueError):
            continue
        if games < _MIN_MATCHUP_GAMES or opponent_id not in id_map:
            continue
        ranked.append((wins / games * 100, games, opponent_id))
    ranked.sort(key=lambda t: (t[0], t[1], t[2]))

    worst = ranked[:_TOP_MATCHUPS]
    remaining = ranked[_TOP_MATCHUPS:]
    best = list(reversed(remaining[-_TOP_MATCHUPS:]))

    chunks: list[LearnedChunk] = []
    for kind, entries in (("worst", worst), ("best", best)):
        for winrate, games, opponent_id in entries:
            opp_name, opp_slug = id_map[opponent_id]
            for lang in langs:
                if lang == "en":
                    verdict = "unfavorable" if kind == "worst" else "favorable"
                    content = (
                        f"{hero_name} vs {opp_name}: {winrate:.1f}% win rate over {games} games "
                        f"({verdict} matchup)."
                    )
                else:
                    verdict = "劣势" if kind == "worst" else "优势"
                    content = (
                        f"{hero_name} 对阵 {opp_name}：胜率 {winrate:.1f}%（样本 {games} 场，"
                        f"统计上属于{verdict}对局）。"
                    )
                chunks.append(LearnedChunk(
                    chunk_id=f"learned:matchup:{slug}:{opp_slug}",
                    content=content,
                    lang=lang,
                    category="hero_matchup",
                    tags=[f"matchup:{slug}:{opp_slug}"],
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

def _write_benchmarks_raw(raw: dict[str, dict], patch: str, output_dir: Path) -> None:
    """Merge machine-readable percentiles into <output_dir>/_benchmarks_raw.json."""
    path = output_dir / BENCHMARKS_RAW_FILE
    merged: dict[str, dict] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            previous = json.load(f).get("heroes") or {}
        if isinstance(previous, dict):
            merged.update(previous)
    except (OSError, ValueError, AttributeError):
        pass
    merged.update(raw)
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "patch": patch,
            "hero_count": len(merged),
        },
        "heroes": merged,
    }
    write_json_atomic(path, payload)
    logger.info("Raw benchmarks written: %s (%d heroes)", path.name, len(merged))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def learn_stats(
    langs: tuple[str, ...] = ("zh", "en"),
    heroes_limit: int | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Learn benchmark / item timing / matchup statistics from OpenDota.

    Returns ``{"chunks_written": int, "heroes": int, "api_calls": int}`` plus
    ``"error"`` when nothing could be learned. Heroes that were not
    refreshed this run keep their previous chunks (a partially failed run
    never shrinks the stored knowledge). Never raises.
    """
    output_dir = output_dir or LEARNED_DIR
    summary: dict = {"chunks_written": 0, "heroes": 0, "api_calls": 0}
    try:
        full_catalog = _hero_catalog()
        if not full_catalog:
            logger.warning("learn_stats: no hero catalog available, skipping run")
            summary["error"] = "no_hero_catalog"
            return summary
        id_map = {hid: (name, slug) for hid, name, slug in full_catalog}
        catalog = full_catalog[:heroes_limit] if heroes_limit is not None else full_catalog

        patch = current_patch_version()
        chunks: list[LearnedChunk] = []
        raw_benchmarks: dict[str, dict] = {}
        refreshed: set[str] = set()
        consecutive_errors = 0

        for hero_id, hero_name, slug in catalog:
            if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                logger.warning(
                    "learn_stats: %d consecutive API errors, aborting early",
                    consecutive_errors,
                )
                break
            hero_chunks: list[LearnedChunk] = []
            hero_ok = False

            # a) percentile benchmarks
            try:
                summary["api_calls"] += 1
                data = opendota_get("benchmarks", {"hero_id": hero_id})
                result = data.get("result") if isinstance(data, dict) else None
                bench_chunks, metrics = _benchmark_chunks(
                    hero_name, slug, hero_id, result or {}, patch, langs
                )
                hero_chunks.extend(bench_chunks)
                if metrics:
                    raw_benchmarks[_bench_key(hero_name)] = {
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
                hero_chunks.extend(_item_timing_chunks(
                    hero_name, slug, hero_id, rows if isinstance(rows, list) else [], patch, langs
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
                hero_chunks.extend(_matchup_chunks(
                    hero_name, slug, hero_id, rows if isinstance(rows, list) else [],
                    id_map, patch, langs,
                ))
                consecutive_errors = 0
                hero_ok = True
            except Exception as exc:
                consecutive_errors += 1
                logger.warning("learn_stats: matchups failed for %s: %s", hero_name, exc)

            if hero_ok:
                summary["heroes"] += 1
                if hero_chunks:
                    refreshed.add(slug)
                    chunks.extend(hero_chunks)

        if not chunks:
            logger.warning("learn_stats: produced no chunks, keeping previous learned files")
            summary["error"] = "no_data"
            return summary

        # Keep previous chunks of heroes that were not refreshed this run.
        kept = [
            c for c in load_learned_file(LEARNED_FILE_NAME, output_dir)
            if _chunk_hero(c.chunk_id) not in refreshed
        ]
        save_learned_file(LEARNED_FILE_NAME, kept + chunks, meta={
            "patch": patch,
            "heroes_refreshed": len(refreshed),
            "api_calls": summary["api_calls"],
            "langs": list(langs),
        }, output_dir=output_dir)
        if raw_benchmarks:
            _write_benchmarks_raw(raw_benchmarks, patch, output_dir)
        summary["chunks_written"] = len(chunks)
        return summary
    except Exception as exc:
        logger.warning("learn_stats failed: %s", exc)
        summary["error"] = "unexpected_error"
        return summary


if __name__ == "__main__":  # pragma: no cover — manual CLI, run from the project root
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    from .base import SMOKE_DIR
    print(learn_stats(heroes_limit=limit, output_dir=SMOKE_DIR))
