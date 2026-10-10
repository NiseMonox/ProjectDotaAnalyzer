"""Level 1 learner: statistical self-learning from OpenDota aggregate endpoints.

Zero LLM cost. For every hero it pulls three aggregate endpoints:

- ``/benchmarks?hero_id=N``        -> per-metric percentiles (GPM, XPM, ...),
  all ranks and all positions of the hero together; asked again with
  ``role=core`` and ``role=support`` for the raw file below
- ``/scenarios/itemTimings?hero_id=N`` -> games and wins per item and time
  bucket (the bucket a purchase fell into, up to 30 min)
- ``/heroes/{id}/matchups``        -> per-opponent games and wins in the
  professional matches of the last twelve months

and renders them into bilingual chunks (one LearnedChunk per language) that
state *statistics* only — they are labelled as such when injected and are
merged into the editorial matchup tip for the same pair instead of competing
with it. Raw percentile values are additionally persisted to
``<output_dir> / "_benchmarks_raw.json"`` in machine-readable form so that
``pro_patterns`` can overlay live values onto its hardcoded benchmark tables.
It only takes the core / support halves: on the first full run the mixed
percentiles rated a position 5 Techies (382 GPM) "below average" against a
median of 418 that mid-lane Techies games had pulled up.

The hero list comes from ``data_manager.get_hero_catalog()``, which works
without fetched game data (the frontend catalog is committed), so a
concurrent / failed game-data update can no longer leave this learner
without heroes.

Budget: 5 API calls per hero, 635 for 127 heroes. With 3 per hero (381
calls, before the two role benchmarks) a full run took 12 minutes on
2026-10-11: the matchup query answers in about 3 s, and one or two closed
rate-limit windows were sat out (see ``base.opendota_get``). Use
``heroes_limit`` for tests.

A run that takes this long gets interrupted (a server restart, a DEV
reload): every :data:`_CHECKPOINT_EVERY` heroes it saves what it has and
notes the finished heroes in ``<output_dir> / PROGRESS_FILE``; the next run
of the same patch within :data:`_RESUME_MAX_AGE_S` continues after them
instead of starting at hero 1 again.
"""
from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from .. import data_manager
from .base import (
    KEY_ITEM_MIN_COST,
    LEARNED_DIR,
    STATS_MIN_SAMPLES,
    LearnedChunk,
    current_patch_version,
    is_key_item,
    load_item_meta,
    load_learned_file,
    opendota_get,
    save_learned_file,
    stop_requested,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

__all__ = ["learn_stats", "BENCHMARKS_RAW_FILE", "LEARNED_FILE_NAME", "PROGRESS_FILE"]

# Machine-readable percentile values consumed by pro_patterns (not chunks)
BENCHMARKS_RAW_FILE = "_benchmarks_raw.json"
LEARNED_FILE_NAME = "stats_opendota"
# Heroes an unfinished run has completed ("_" prefix: never loaded as chunks).
PROGRESS_FILE = "_stats_progress.json"
# A run saves its chunks and its progress after this many heroes (about 90 s
# of API calls): what a killed process loses.
_CHECKPOINT_EVERY = 10
# An unfinished run is continued for this long; after that it starts over
# (the statistics of its first heroes would be a day older than the rest).
_RESUME_MAX_AGE_S = 24 * 3600

# Rows below these sample sizes are noise; the chunk loader applies the same
# limits to what is on disk (base.chunk_rejection).
_MIN_ITEM_GAMES = STATS_MIN_SAMPLES["item_timing"]
_MIN_MATCHUP_GAMES = STATS_MIN_SAMPLES["matchup"]
_TOP_ITEM_INSIGHTS = 4     # item timing insights kept per hero
_TOP_MATCHUPS = 3          # worst 3 + best 3 matchups per hero
# A matchup is only called (un)favorable when the win rate says so.
_UNFAVORABLE_BELOW = 48.0
_FAVORABLE_ABOVE = 52.0
_MAX_CONSECUTIVE_ERRORS = 5  # abort early when the API looks unreachable
# Share of failed endpoint calls above which a run counts as partial.
_MAX_FAILED_CALL_SHARE = 0.10

# Chunk kinds (second chunk_id segment), one per endpoint.
_KIND_BENCHMARKS = "benchmarks"
_KIND_ITEM_TIMING = "item_timing"
_KIND_MATCHUP = "matchup"
# The two halves of a hero's games that OpenDota benchmarks on request
# (``role``: "cores or supports, estimated from farm within each team").
# They yield no chunks; their percentiles are what pro_patterns rates a
# player against, by the side of the team he played on.
_BENCH_ROLES = ("core", "support")
_CALLS_PER_HERO = 3 + len(_BENCH_ROLES)

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


# Standard deviations a matchup's wins must lie from half its games. Three,
# not the usual two: every hero is compared with every opponent, several
# thousand pairs per run, and at two an even matchup is labelled once in
# twenty tries (first full run: 222 of the 409 pairs it kept were between 2
# and 2.5, the pile-up noise leaves just above a threshold).
_VERDICT_SIGMAS = 3.0


def _beyond_chance(wins: int, games: int) -> bool:
    """Whether a win count differs from an even matchup by more than chance.

    Normal approximation of the binomial at 50%: the distance from half the
    games has to exceed :data:`_VERDICT_SIGMAS` standard deviations
    (1.5 * sqrt(games) wins; 21 points at 50 games, 7.5 at 400).
    """
    return games > 0 and abs(wins - games / 2) > _VERDICT_SIGMAS * 0.5 * math.sqrt(games)


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


def _first_boots(key: str, item_meta: dict[str, dict]) -> bool:
    """Power Treads, Phase, Arcane or Tranquil Boots: every game has a pair.

    Counted by games they would take a slot of nearly every hero, and when
    they are finished says next to nothing (Anti-Mage, live: 51.9% before
    10 min against 51.0% overall). The boots a hero saves up for (Boots of
    Travel, Guardian Greaves, Boots of Bearing) stay.
    """
    info = item_meta.get(key, {}) if item_meta else {}
    cost = info.get("cost") or 0
    return info.get("category") == "boots" and 0 < cost < KEY_ITEM_MIN_COST


def _chunk_hero(chunk_id: str) -> str:
    """Hero slug encoded in a stats chunk id ("learned:<kind>:<hero>:...")."""
    parts = chunk_id.split(":")
    return parts[2] if len(parts) > 2 else ""


def _chunk_kind(chunk_id: str) -> str:
    """Kind encoded in a stats chunk id ("learned:<kind>:<hero>:...")."""
    parts = chunk_id.split(":")
    return parts[1] if len(parts) > 1 else ""


# ---------------------------------------------------------------------------
# Chunk builders (one LearnedChunk per language per insight)
# ---------------------------------------------------------------------------

def _benchmark_metrics(result: dict) -> dict[str, dict[str, float]]:
    """``{metric: {"p10", "p50", "p90"}}`` of one /benchmarks answer (p10 optional)."""
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
    return metrics


def _benchmark_chunks(
    hero_name: str,
    slug: str,
    hero_id: int,
    result: dict,
    patch: str,
    langs: tuple[str, ...],
) -> tuple[list[LearnedChunk], dict]:
    """Render /benchmarks percentiles into chunks + raw metric dict.

    Asked without ``role``, the endpoint mixes the hero's core and support
    games (Mirana, live: median 2.5 last hits a minute), so the text says so
    and gives no advice of its own: "below the median, improve it first"
    would tell a position 5 to farm more.
    """
    metrics = _benchmark_metrics(result)
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
                f"{hero_name} performance benchmarks (all ranks, every position the hero is "
                f"played in): " + "; ".join(parts)
                + ". Read them against your own position before calling a number low."
            )
        else:
            content = (
                f"{hero_name} 数据基准（全分段、该英雄的所有位置合计）：" + "；".join(parts)
                + "。判断某项数据偏低之前，请先结合自己打的位置。"
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
    item_meta: dict[str, dict] | None = None,
) -> list[LearnedChunk]:
    """Render /scenarios/itemTimings rows into top-N timing insight chunks.

    A row counts the games in which the item was completed inside one time
    bucket: after the previous bound and up to its own (OpenDota files a
    purchase under the first of 7.5 / 10 / 12 / 15 / 20 / 25 / 30 min that
    is not earlier than it and records nothing later). "Completed before N
    min" is therefore the sum of the buckets up to N. Per item the bound
    with the best such win rate is stated, followed by the win rate over
    everything recorded for the item, so the early timing is read against
    what it is compared with.

    Only key items are ranked (:func:`base.is_key_item`): components are
    bought before the item they build into and would otherwise take the
    slots of the finished items. The items the hero completes most often
    come first; ranked by win rate, the luxury items of games that were
    already won would push out the items every game of the hero is about.
    """
    if item_meta is None:
        item_meta = load_item_meta()
    buckets_by_item: dict[str, dict[int, list[int]]] = {}  # item key -> {bound: [games, wins]}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        try:
            item = data_manager.item_key(str(row.get("item") or ""))
            games = int(row.get("games") or 0)
            wins = int(row.get("wins") or 0)
            bucket = int(row.get("time") or 0)
        except (TypeError, ValueError):
            continue
        if (not item or games <= 0 or not 0 <= wins <= games or bucket <= 0
                or not is_key_item(item, item_meta) or _first_boots(item, item_meta)):
            continue
        counts = buckets_by_item.setdefault(item, {}).setdefault(bucket, [0, 0])
        counts[0] += games
        counts[1] += wins

    # item key -> (best (win rate, games, bound), all (win rate, games, bound));
    # the minimum sample applies to the games completed before a bound.
    insights: dict[str, tuple[tuple[float, int, int], tuple[float, int, int]]] = {}
    for item, buckets in buckets_by_item.items():
        best: tuple[float, int, int] | None = None
        games = wins = 0
        for bound in sorted(buckets):
            games += buckets[bound][0]
            wins += buckets[bound][1]
            if games >= _MIN_ITEM_GAMES and (best is None or wins / games * 100 > best[0]):
                best = (wins / games * 100, games, bound)
        if best is not None:
            insights[item] = (best, (wins / games * 100, games, max(buckets)))

    top = sorted(insights.items(), key=lambda kv: (-kv[1][1][1], kv[0]))[:_TOP_ITEM_INSIGHTS]

    chunks: list[LearnedChunk] = []
    for item_slug, ((winrate, games, bucket), (all_winrate, all_games, last_bucket)) in top:
        display = _item_display_name(item_slug)
        minutes = _fmt_minutes(bucket)
        all_minutes = _fmt_minutes(last_bucket)
        for lang in langs:
            if lang == "en":
                content = (
                    f"{hero_name}: games where {display} was completed before {minutes} min "
                    f"had a {winrate:.1f}% win rate ({games} games)"
                )
                if last_bucket > bucket:
                    content += (f"; over all {all_games} games where it was completed before "
                                f"{all_minutes} min the win rate was {all_winrate:.1f}%")
                content += "."
            else:
                content = (
                    f"{hero_name}：{minutes} 分钟前做出 {display} 的对局胜率 "
                    f"{winrate:.1f}%（样本 {games} 场）"
                )
                if last_bucket > bucket:
                    content += (f"；{all_minutes} 分钟前做出的全部 {all_games} 场对局胜率 "
                                f"{all_winrate:.1f}%")
                content += "。"
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
                    "sample": games,
                    "sample_unit": "games",
                    "winrate": round(winrate, 1),
                    "before_seconds": bucket,
                    "all_games": all_games,
                    "all_winrate": round(all_winrate, 1),
                    "all_before_seconds": last_bucket,
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
    """Render /heroes/{id}/matchups into worst-3 / best-3 matchup statistics.

    The rows are OpenDota's professional matches of the last twelve months
    (its ``matches`` table, ``start_time`` within one year), which the text
    says: the numbers span more than the current patch.

    The verdict follows the win rate, not the rank: only opponents below
    :data:`_UNFAVORABLE_BELOW` are "unfavorable" and only those above
    :data:`_FAVORABLE_ABOVE` "favorable"; a hero whose few qualifying
    matchups are all near 50% (or all good) gets no misleading label. And
    it needs the sample to carry it (:func:`_beyond_chance`): 44% over 60
    games is what an even matchup looks like often enough, so such a pair
    gets no chunk at all.
    """
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
        if (games < _MIN_MATCHUP_GAMES or not 0 <= wins <= games or opponent_id not in id_map
                or not _beyond_chance(wins, games)):
            continue
        ranked.append((wins / games * 100, games, opponent_id))
    ranked.sort(key=lambda t: (t[0], t[1], t[2]))

    worst = [row for row in ranked if row[0] < _UNFAVORABLE_BELOW][:_TOP_MATCHUPS]
    best = [row for row in reversed(ranked) if row[0] > _FAVORABLE_ABOVE][:_TOP_MATCHUPS]

    chunks: list[LearnedChunk] = []
    for kind, entries in (("worst", worst), ("best", best)):
        for winrate, games, opponent_id in entries:
            opp_name, opp_slug = id_map[opponent_id]
            for lang in langs:
                if lang == "en":
                    verdict = "unfavorable" if kind == "worst" else "favorable"
                    content = (
                        f"{hero_name} vs {opp_name}: {winrate:.1f}% win rate over {games} "
                        f"professional matches of the last 12 months ({verdict} matchup)."
                    )
                else:
                    verdict = "劣势" if kind == "worst" else "优势"
                    content = (
                        f"{hero_name} 对阵 {opp_name}：近12个月职业比赛胜率 {winrate:.1f}%"
                        f"（样本 {games} 场，统计上属于{verdict}对局）。"
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
                        "sample": games,
                        "sample_unit": "games",
                        "winrate": round(winrate, 1),
                        "kind": kind,
                    },
                ))
    return chunks


# ---------------------------------------------------------------------------
# Raw benchmark persistence (for pro_patterns overlay)
# ---------------------------------------------------------------------------

def _write_benchmarks_raw(raw: dict[str, dict], patch: str, output_dir: Path) -> None:
    """Merge machine-readable percentiles into <output_dir>/_benchmarks_raw.json.

    Per hero ``{"hero", "hero_id", <metric>: {p10, p50, p90} (all games),
    "roles": {"core" | "support": {<metric>: {...}}}}``. What a run could
    not fetch keeps its previous value, per hero and per role.
    """
    path = output_dir / BENCHMARKS_RAW_FILE
    merged: dict[str, dict] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            previous = json.load(f).get("heroes") or {}
        if isinstance(previous, dict):
            merged.update(previous)
    except (OSError, ValueError, AttributeError):
        pass
    for key, entry in raw.items():
        before = merged.get(key) if isinstance(merged.get(key), dict) else {}
        roles = {**(before.get("roles") if isinstance(before.get("roles"), dict) else {}),
                 **entry.get("roles", {})}
        merged[key] = {**before, **entry, **({"roles": roles} if roles else {})}
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
# Progress of an unfinished run
# ---------------------------------------------------------------------------

def _load_progress(output_dir: Path, patch: str, langs: tuple[str, ...]) -> set[str]:
    """Hero slugs an interrupted run of this patch and these languages finished.

    Empty when there is no such run, when it is older than
    :data:`_RESUME_MAX_AGE_S`, or when the file is unreadable.
    """
    try:
        with open(output_dir / PROGRESS_FILE, "r", encoding="utf-8") as f:
            progress = json.load(f)
        if progress.get("patch") != patch or progress.get("langs") != list(langs):
            return set()
        age = time.time() - float(progress.get("updated_at") or 0)
        if not 0 <= age <= _RESUME_MAX_AGE_S:
            return set()
        return {str(slug) for slug in progress.get("heroes") or []}
    except (OSError, ValueError, TypeError, AttributeError):
        return set()


def _save_progress(output_dir: Path, patch: str, langs: tuple[str, ...], heroes: set[str]) -> None:
    write_json_atomic(output_dir / PROGRESS_FILE, {
        "patch": patch, "langs": list(langs), "updated_at": time.time(), "heroes": sorted(heroes),
    })


def _clear_progress(output_dir: Path) -> None:
    try:
        (output_dir / PROGRESS_FILE).unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def learn_stats(
    langs: tuple[str, ...] = ("zh", "en"),
    heroes_limit: int | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Learn benchmark / item timing / matchup statistics from OpenDota.

    Returns ``{"chunks_written": int, "heroes": int, "api_calls": int,
    "failed_calls": int}`` plus ``"error"``: ``"no_data"`` when nothing
    could be learned, ``"partial"`` when the run was cut short or too many
    calls failed (what was learned is saved, the level stays due),
    ``"cancelled"`` on a stop request. ``"resumed"`` counts the heroes an
    interrupted earlier run had already finished (not asked again).

    Stored chunks are replaced per (hero, kind): an endpoint that answered
    replaces that hero's chunks of that kind (also with nothing), one that
    failed keeps the previous ones — a partially failed run never shrinks
    the stored knowledge. Chunks of heroes that are no longer in the catalog
    (including ids written with an old slug format) are dropped. Never raises.
    """
    output_dir = output_dir or LEARNED_DIR
    summary: dict = {"chunks_written": 0, "heroes": 0, "api_calls": 0, "failed_calls": 0}
    try:
        full_catalog = _hero_catalog()
        if not full_catalog:
            logger.warning("learn_stats: no hero catalog available, skipping run")
            summary["error"] = "no_hero_catalog"
            return summary
        id_map = {hid: (name, slug) for hid, name, slug in full_catalog}
        catalog = full_catalog[:heroes_limit] if heroes_limit is not None else full_catalog

        patch = current_patch_version()
        item_meta = load_item_meta()
        chunks: list[LearnedChunk] = []
        raw_benchmarks: dict[str, dict] = {}
        refreshed: set[tuple[str, str]] = set()  # (hero slug, kind) answered this run
        consecutive_errors = 0
        aborted = False
        cancelled = False
        known_slugs = {slug for _hid, _name, slug in full_catalog}

        # Heroes an interrupted run already finished are not asked again.
        finished = _load_progress(output_dir, patch, langs) & {slug for _hid, _name, slug in catalog}
        if len(finished) >= len(catalog):
            finished = set()    # nothing left of that run: this is a new one
        if finished:
            summary["resumed"] = len(finished)
            logger.info("learn_stats: continuing an interrupted run, %d of %d heroes are done",
                        len(finished), len(catalog))

        def save() -> None:
            """Write what this run has so far next to the untouched previous chunks.

            Kept: the previous chunks of every (hero, kind) that was not
            refreshed this run; dropped: those of heroes the catalog no
            longer has. Old files name heroes by display slug
            ("anti_mage"): the id segment is mapped through the canonical
            slug so they are replaced too.
            """
            kept = []
            for chunk in load_learned_file(LEARNED_FILE_NAME, output_dir):
                hero = data_manager.hero_slug(_chunk_hero(chunk.chunk_id))
                if hero in known_slugs and (hero, _chunk_kind(chunk.chunk_id)) not in refreshed:
                    kept.append(chunk)
            save_learned_file(LEARNED_FILE_NAME, kept + chunks, meta={
                "patch": patch,
                "heroes_refreshed": len({hero for hero, _kind in refreshed}),
                "api_calls": summary["api_calls"],
                "failed_calls": summary["failed_calls"],
                "langs": list(langs),
            }, output_dir=output_dir)
            if raw_benchmarks:
                _write_benchmarks_raw(raw_benchmarks, patch, output_dir)

        since_checkpoint = 0
        for hero_id, hero_name, slug in catalog:
            if slug in finished:
                continue
            if stop_requested():
                cancelled = True
                break

            def benchmarks() -> list[LearnedChunk]:
                data = opendota_get("benchmarks", {"hero_id": hero_id})
                result = data.get("result") if isinstance(data, dict) else None
                bench_chunks, metrics = _benchmark_chunks(
                    hero_name, slug, hero_id, result or {}, patch, langs
                )
                if metrics:
                    raw_entry().update(metrics)
                return bench_chunks

            def raw_entry() -> dict:
                return raw_benchmarks.setdefault(
                    _bench_key(hero_name), {"hero": hero_name, "hero_id": hero_id})

            def role_benchmarks(role: str):
                def fetch() -> list[LearnedChunk]:
                    data = opendota_get("benchmarks", {"hero_id": hero_id, "role": role})
                    metrics = _benchmark_metrics((data.get("result") if isinstance(data, dict) else None) or {})
                    if metrics:
                        raw_entry().setdefault("roles", {})[role] = metrics
                    return []
                return fetch

            def item_timings() -> list[LearnedChunk]:
                rows = opendota_get("scenarios/itemTimings", {"hero_id": hero_id})
                return _item_timing_chunks(
                    hero_name, slug, hero_id, rows if isinstance(rows, list) else [],
                    patch, langs, item_meta,
                )

            def matchups() -> list[LearnedChunk]:
                rows = opendota_get(f"heroes/{hero_id}/matchups")
                return _matchup_chunks(
                    hero_name, slug, hero_id, rows if isinstance(rows, list) else [],
                    id_map, patch, langs,
                )

            hero_ok = False
            failed_before = summary["failed_calls"]
            for kind, fetch in ((_KIND_BENCHMARKS, benchmarks),
                                *((f"{_KIND_BENCHMARKS}_{role}", role_benchmarks(role)) for role in _BENCH_ROLES),
                                (_KIND_ITEM_TIMING, item_timings),
                                (_KIND_MATCHUP, matchups)):
                summary["api_calls"] += 1
                try:
                    chunks.extend(fetch())
                except Exception as exc:
                    summary["failed_calls"] += 1
                    consecutive_errors += 1
                    logger.warning("learn_stats: %s failed for %s: %s", kind, hero_name, exc)
                    if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                        aborted = True
                        break
                    continue
                consecutive_errors = 0
                hero_ok = True
                refreshed.add((slug, kind))
            if hero_ok:
                summary["heroes"] += 1
            if aborted:
                logger.warning(
                    "learn_stats: %d consecutive API errors, aborting early",
                    consecutive_errors,
                )
                break
            if summary["failed_calls"] == failed_before:
                finished.add(slug)      # a hero with a failed call is asked again by a resumed run
            since_checkpoint += 1
            if since_checkpoint >= _CHECKPOINT_EVERY and chunks:
                since_checkpoint = 0
                save()
                _save_progress(output_dir, patch, langs, finished)

        if not chunks:
            logger.warning("learn_stats: produced no chunks, keeping previous learned files")
            summary["error"] = "cancelled" if cancelled else "no_data"
            return summary

        save()
        summary["chunks_written"] = len(chunks)

        # A run that stopped early or lost many calls is not a success: the
        # data above is kept, but the level must stay due (not "ok" for a week).
        expected_calls = _CALLS_PER_HERO * (len(catalog) - summary.get("resumed", 0))
        if cancelled:
            summary["error"] = "cancelled"
        elif aborted or summary["failed_calls"] > _MAX_FAILED_CALL_SHARE * expected_calls:
            logger.warning(
                "learn_stats: partial run (%d of %d calls failed, aborted=%s)",
                summary["failed_calls"], expected_calls, aborted,
            )
            summary["error"] = "partial"
        if summary.get("error"):
            _save_progress(output_dir, patch, langs, finished)   # the next run continues here
        else:
            _clear_progress(output_dir)
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
