"""Level 2 learner — mine pro / high-MMR matches through the project's own pipeline.

Pulls recent professional (or high-MMR public) matches from OpenDota, runs each
one through this project's :class:`~app.parser.OpenDotaParser` and position
derivation (:mod:`app.ai.positions`), and aggregates per-position performance
patterns (CS@10, GPM, deaths, kill participation, observer wards) plus
per-hero core-item first timings. The aggregates are rendered into bilingual
coaching chunks (category ``pro_pattern``) and the raw numbers are persisted
to ``LEARNED_DIR / MINING_AGGREGATES`` for the Level-3 LLM distiller.

Zero LLM cost — everything here is deterministic template rendering.
"""
from __future__ import annotations

import asyncio
import json
import logging
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .base import (
    LEARNED_DIR,
    MINING_AGGREGATES,
    LearnedChunk,
    current_patch_version,
    opendota_get,
    save_learned_file,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Position -> role tag, matching knowledge_chunks/_role_tag conventions.
_POS_ROLE_TAG = {
    1: "role:carry",
    2: "role:mid",
    3: "role:offlane",
    4: "role:support_4",
    5: "role:support_5",
}
_POS_LABEL_ZH = {1: "1号位", 2: "2号位", 3: "3号位", 4: "4号位", 5: "5号位"}
_POS_LABEL_EN = {
    1: "Position 1 (Carry)",
    2: "Position 2 (Mid)",
    3: "Position 3 (Offlaner)",
    4: "Position 4 (Support)",
    5: "Position 5 (Hard Support)",
}

# An item counts as "core" when it costs at least this much gold.
_CORE_ITEM_MIN_COST = 2000

# Fallback core-item keys used when frontend items.json is unavailable.
_FALLBACK_CORE_ITEMS = frozenset({
    "bfury", "maelstrom", "mjollnir", "radiance", "manta", "black_king_bar",
    "blink", "ultimate_scepter", "desolator", "greater_crit", "butterfly",
    "satanic", "skadi", "assault", "heart", "sheepstick", "orchid",
    "bloodthorn", "shivas_guard", "pipe", "crimson_guard", "vladmir",
    "spirit_vessel", "lotus_orb", "aeon_disk", "octarine_core", "refresher",
    "cyclone", "force_staff", "aether_lens", "glimmer_cape", "guardian_greaves",
    "hand_of_midas", "armlet", "echo_sabre", "sange_and_yasha", "kaya_and_sange",
    "yasha_and_kaya", "silver_edge", "abyssal_blade", "monkey_king_bar",
    "hurricane_pike", "nullifier", "dagon", "ethereal_blade", "sphere",
    "solar_crest", "mekansm", "gungir", "diffusal_blade", "mage_slayer",
    "witch_blade", "eternal_shroud", "bloodstone", "travel_boots",
})

# Between full-match fetches through OpenDotaParser (which bypasses the
# base.opendota_get throttle) keep the same polite rate-limit floor.
_MATCH_FETCH_INTERVAL = 1.1


# ---------------------------------------------------------------------------
# Item metadata
# ---------------------------------------------------------------------------

def _load_item_meta() -> dict[str, dict]:
    """Load item key -> {name, cost} from the frontend static data.

    Returns {} when the file is missing — callers fall back to the builtin
    core-item key set.
    """
    items_path = (
        Path(__file__).resolve().parents[4]
        / "frontend" / "static" / "data" / "items.json"
    )
    try:
        with open(items_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            key: {"name": info.get("name", key), "cost": info.get("cost", 0) or 0}
            for key, info in data.get("by_key", {}).items()
        }
    except Exception as exc:
        logger.warning("items.json unavailable (%s); using fallback core item set", exc)
        return {}


def _is_core_item(key: str, item_meta: dict[str, dict]) -> bool:
    if not key or key.startswith("recipe"):
        return False
    if item_meta:
        return item_meta.get(key, {}).get("cost", 0) >= _CORE_ITEM_MIN_COST
    return key in _FALLBACK_CORE_ITEMS


def _item_display_name(key: str, item_meta: dict[str, dict]) -> str:
    if key in item_meta and item_meta[key].get("name"):
        return item_meta[key]["name"]
    return key.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_match_ids(source: str, limit: int) -> list[int]:
    """Fetch candidate match ids from OpenDota. Raises on network failure."""
    if source == "high_mmr":
        rows = opendota_get("publicMatches", {"min_rank": 80})
        # duration == 0 rows are unparsed placeholders — filter them out.
        rows = [r for r in rows if r.get("duration")]
    else:  # "pro" (default)
        rows = opendota_get("proMatches")
    ids: list[int] = []
    for row in rows:
        mid = row.get("match_id")
        if isinstance(mid, int) and mid not in ids:
            ids.append(mid)
        if len(ids) >= limit:
            break
    return ids


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _new_position_bucket() -> dict[str, list]:
    return {"cs_at_10": [], "gpm": [], "deaths": [], "kill_participation": [], "obs_placed": []}


def _accumulate_match(
    match,
    pos_stats: dict[int, dict[str, list]],
    hero_stats: dict[str, dict],
    item_meta: dict[str, dict],
) -> None:
    """Fold one parsed MatchData into the running aggregates."""
    from ..positions import derive_match_positions

    positions = derive_match_positions(match.players)
    team_kills = {"radiant": 0, "dire": 0}
    for p in match.players:
        team_kills[p.team] = team_kills.get(p.team, 0) + p.kills

    for p in match.players:
        pos = positions.get(p.player_slot)
        if pos is None or pos.number not in _POS_ROLE_TAG:
            continue
        bucket = pos_stats[pos.number]

        if len(p.lh_t) > 10:
            bucket["cs_at_10"].append(p.lh_t[10])
        bucket["gpm"].append(p.gold_per_min)
        bucket["deaths"].append(p.deaths)
        tk = team_kills.get(p.team, 0)
        if tk > 0:
            bucket["kill_participation"].append(min(1.0, (p.kills + p.assists) / tk))
        bucket["obs_placed"].append(p.obs_placed)

        # Per-hero first-purchase timing of core items.
        slug = p.hero_slug or p.hero_name.lower().replace(" ", "_").replace("'", "").replace("-", "_")
        hero = hero_stats.setdefault(
            slug, {"hero_name": p.hero_name, "games": 0, "timings": {}}
        )
        hero["games"] += 1
        first_seen: dict[str, int] = {}
        for entry in p.purchase_log:
            key = entry.get("key", "")
            t = entry.get("time")
            if t is None or not _is_core_item(key, item_meta):
                continue
            if key not in first_seen or t < first_seen[key]:
                first_seen[key] = t
        for key, t in first_seen.items():
            hero["timings"].setdefault(key, []).append(t)


def _mean(values: list) -> float | None:
    return statistics.fmean(values) if values else None


def _p75(values: list) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    return statistics.quantiles(values, n=4)[2]


def _finalize_aggregates(
    pos_stats: dict[int, dict[str, list]],
    hero_stats: dict[str, dict],
    match_ids: list[int],
    source: str,
    patch: str,
) -> dict:
    """Reduce raw sample lists into the aggregates dict persisted for Level 3."""
    positions_out: dict[str, dict] = {}
    for pos, bucket in pos_stats.items():
        sample = len(bucket["gpm"])
        if sample == 0:
            continue
        kp_mean = _mean(bucket["kill_participation"])
        positions_out[str(pos)] = {
            "sample": sample,
            "cs_at_10_sample": len(bucket["cs_at_10"]),
            "cs_at_10_mean": _mean(bucket["cs_at_10"]),
            "cs_at_10_p75": _p75(bucket["cs_at_10"]),
            "gpm_mean": _mean(bucket["gpm"]),
            "deaths_mean": _mean(bucket["deaths"]),
            "kill_participation_mean": kp_mean,
            "obs_placed_mean": _mean(bucket["obs_placed"]),
        }

    heroes_out: dict[str, dict] = {}
    for slug, hero in hero_stats.items():
        items_out = {
            key: {"mean_first_time_s": _mean(times), "sample": len(times)}
            for key, times in sorted(hero["timings"].items())
            if times
        }
        heroes_out[slug] = {
            "hero_name": hero["hero_name"],
            "games": hero["games"],
            "items": items_out,
        }

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
        "patch": patch,
        "match_ids": match_ids,
        "matches_mined": len(match_ids),
        "positions": positions_out,
        "hero_item_timings": heroes_out,
    }


# ---------------------------------------------------------------------------
# Chunk rendering
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _core_position_content(pos: int, stats: dict, lang: str) -> str | None:
    cs_mean = stats.get("cs_at_10_mean")
    cs_p75 = stats.get("cs_at_10_p75")
    gpm = stats.get("gpm_mean")
    deaths = stats.get("deaths_mean")
    kp = stats.get("kill_participation_mean")
    n = stats["sample"]
    if cs_mean is None or gpm is None or deaths is None:
        return None
    cs_low = int(cs_mean * 0.8)
    deaths_hi = int(round(deaths)) + 2
    kp_pct = int(round(kp * 100)) if kp is not None else None
    if lang == "zh":
        parts = [
            f"职业对局中{_POS_LABEL_ZH[pos]}10分钟平均正补{cs_mean:.0f}"
            f"（前25%达到{cs_p75:.0f}），全场平均GPM {gpm:.0f}，"
            f"场均死亡{deaths:.1f}次（样本{n}人次）。"
        ]
        if kp_pct is not None:
            parts.append(f"击杀参与率平均{kp_pct}%。")
        parts.append(
            f"若你10分钟正补低于{cs_low}，通常意味着对线期出了问题（被压制、游走过多或漏刀）；"
            f"死亡超过{deaths_hi}次时，先复盘死因（视野、站位、贪线）再谈操作。"
        )
        return "".join(parts)
    parts = [
        f"In pro matches, {_POS_LABEL_EN[pos]} averages {cs_mean:.0f} last hits at 10 minutes "
        f"(top 25% reach {cs_p75:.0f}), {gpm:.0f} GPM overall, "
        f"and {deaths:.1f} deaths per game (sample: {n} player-games)."
    ]
    if kp_pct is not None:
        parts.append(f" Average kill participation is {kp_pct}%.")
    parts.append(
        f" If your CS at 10 is below {cs_low}, something usually went wrong in the laning stage "
        f"(pressured out, over-roaming, or missed last hits); with more than {deaths_hi} deaths, "
        f"review why you died (vision, positioning, greedy lane pushes) before blaming mechanics."
    )
    return "".join(parts)


def _support_position_content(pos: int, stats: dict, lang: str) -> str | None:
    obs = stats.get("obs_placed_mean")
    kp = stats.get("kill_participation_mean")
    deaths = stats.get("deaths_mean")
    n = stats["sample"]
    if obs is None or deaths is None:
        return None
    obs_low = max(1, int(obs * 0.6))
    kp_pct = int(round(kp * 100)) if kp is not None else None
    deaths_hi = int(round(deaths)) + 2
    if lang == "zh":
        parts = [
            f"职业对局中{_POS_LABEL_ZH[pos]}场均插{obs:.1f}个观察者眼，"
            f"场均死亡{deaths:.1f}次（样本{n}人次）。"
        ]
        if kp_pct is not None:
            parts.append(f"击杀参与率平均{kp_pct}%。")
        parts.append(
            f"若你的观察者眼少于{obs_low}个，说明视野投入明显不足；"
        )
        if kp_pct is not None:
            parts.append(f"参与率低于{max(0, kp_pct - 15)}%说明游走和跟团不够主动；")
        parts.append(
            f"死亡超过{deaths_hi}次时，多半是排眼/插眼路线太深或没有跟队友同步行动。"
        )
        return "".join(parts)
    parts = [
        f"In pro matches, {_POS_LABEL_EN[pos]} places {obs:.1f} observer wards "
        f"and dies {deaths:.1f} times per game (sample: {n} player-games)."
    ]
    if kp_pct is not None:
        parts.append(f" Average kill participation is {kp_pct}%.")
    parts.append(
        f" Fewer than {obs_low} observer wards means you are under-investing in vision;"
    )
    if kp_pct is not None:
        parts.append(
            f" kill participation below {max(0, kp_pct - 15)}% means you are not rotating "
            f"or joining fights enough;"
        )
    parts.append(
        f" more than {deaths_hi} deaths usually comes from warding/dewarding too deep "
        f"without backup."
    )
    return "".join(parts)


def _hero_timing_content(
    hero_name: str, games: int, timing_lines: list[tuple[str, float, int]], lang: str
) -> str:
    if lang == "zh":
        listed = "、".join(
            f"{name} 平均{_fmt_time(t)}（{n}场）" for name, t, n in timing_lines
        )
        return (
            f"职业对局中{hero_name}（样本{games}场）的关键装成型时间：{listed}。"
            f"如果你的同款关键装比这个时间晚3分钟以上，优先检查补刀效率、"
            f"打钱路线和死亡次数，而不是急着改出装。"
        )
    listed = "; ".join(
        f"{name} avg {_fmt_time(t)} ({n} games)" for name, t, n in timing_lines
    )
    return (
        f"In pro matches, {hero_name} (sample: {games} games) reaches key items at: {listed}. "
        f"If your timing on the same item is 3+ minutes later, review your last-hitting, "
        f"farming routes and death count before changing the build."
    )


def _build_chunks(aggregates: dict, langs: tuple[str, ...], patch: str) -> list[LearnedChunk]:
    chunks: list[LearnedChunk] = []
    source = aggregates.get("source", "pro")
    match_ids = aggregates.get("match_ids", [])

    # Per-position pattern chunks.
    for pos_str, stats in sorted(aggregates.get("positions", {}).items()):
        pos = int(pos_str)
        for lang in langs:
            if pos in (4, 5):
                content = _support_position_content(pos, stats, lang)
                tags = [_POS_ROLE_TAG[pos], "phase:laning", "mechanic:vision", "hero:all"]
            else:
                content = _core_position_content(pos, stats, lang)
                tags = [_POS_ROLE_TAG[pos], "phase:laning", "mechanic:cs", "hero:all"]
            if not content:
                continue
            chunks.append(LearnedChunk(
                chunk_id=f"learned:pro_pattern:position_{pos}:{lang}",
                content=content,
                lang=lang,
                category="pro_pattern",
                tags=tags,
                priority=2,
                source="match_mining",
                patch_version=patch,
                provenance={
                    "source": source,
                    "match_ids": match_ids,
                    "sample": stats["sample"],
                },
            ))

    # Per-hero core item timing chunks (heroes seen >= 2 times).
    item_meta = _load_item_meta()
    for slug, hero in sorted(aggregates.get("hero_item_timings", {}).items()):
        if hero.get("games", 0) < 2:
            continue
        timing_lines: list[tuple[str, float, int]] = []
        for key, info in hero.get("items", {}).items():
            # Only report timings observed in >= 2 of this hero's games.
            if info.get("sample", 0) >= 2 and info.get("mean_first_time_s") is not None:
                timing_lines.append(
                    (_item_display_name(key, item_meta), info["mean_first_time_s"], info["sample"])
                )
        if not timing_lines:
            continue
        timing_lines.sort(key=lambda x: x[1])
        timing_lines = timing_lines[:5]  # keep chunks small
        item_tags = [
            f"item:{key}" for key, info in hero.get("items", {}).items()
            if info.get("sample", 0) >= 2
        ][:5]
        for lang in langs:
            chunks.append(LearnedChunk(
                chunk_id=f"learned:pro_pattern:item_timing:{slug}:{lang}",
                content=_hero_timing_content(hero["hero_name"], hero["games"], timing_lines, lang),
                lang=lang,
                category="pro_pattern",
                tags=[f"hero:{slug}", "phase:mid_game", *item_tags],
                priority=2,
                source="match_mining",
                patch_version=patch,
                provenance={
                    "source": source,
                    "match_ids": match_ids,
                    "games": hero["games"],
                },
            ))

    return chunks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def learn_from_matches(
    source: str = "pro",
    limit: int = 10,
    langs: tuple[str, ...] = ("zh", "en"),
) -> dict:
    """Mine recent pro / high-MMR matches into per-position pattern chunks.

    Returns ``{"chunks_written": int, "matches_mined": int, "match_ids": [...]}``.
    Degrades gracefully (warning + zero-chunk summary) on any external failure.
    """
    empty = {"chunks_written": 0, "matches_mined": 0, "match_ids": []}

    try:
        from ...parser import OpenDotaParser
    except Exception as exc:
        logger.warning("OpenDotaParser unavailable, skipping match mining: %s", exc)
        return empty

    try:
        match_ids = _discover_match_ids(source, limit)
    except Exception as exc:
        logger.warning("Match discovery failed (source=%s): %s", source, exc)
        return empty
    if not match_ids:
        logger.warning("No candidate matches found (source=%s)", source)
        return empty
    logger.info("Mining %d matches (source=%s)", len(match_ids), source)

    parser = OpenDotaParser()
    item_meta = _load_item_meta()
    pos_stats: dict[int, dict[str, list]] = {p: _new_position_bucket() for p in range(1, 6)}
    hero_stats: dict[str, dict] = {}
    mined: list[int] = []

    for i, mid in enumerate(match_ids):
        if i > 0:
            time.sleep(_MATCH_FETCH_INTERVAL)
        try:
            match = asyncio.run(parser.parse(str(mid)))
        except Exception as exc:
            logger.warning("Failed to fetch/parse match %s: %s", mid, exc)
            continue
        if not match.gold_xp_timeline:
            logger.info("Match %s has no parsed timeline data, skipping", mid)
            continue
        try:
            _accumulate_match(match, pos_stats, hero_stats, item_meta)
        except Exception as exc:
            logger.warning("Failed to aggregate match %s: %s", mid, exc)
            continue
        mined.append(mid)

    if not mined:
        logger.warning("No usable parsed matches out of %d candidates", len(match_ids))
        return empty

    patch = current_patch_version()
    aggregates = _finalize_aggregates(pos_stats, hero_stats, mined, source, patch)

    # Persist raw aggregates for the Level-3 distiller.
    try:
        agg_path = LEARNED_DIR / MINING_AGGREGATES
        with open(agg_path, "w", encoding="utf-8") as f:
            json.dump(aggregates, f, ensure_ascii=False, indent=1)
        logger.info("Mining aggregates written: %s", agg_path.name)
    except Exception as exc:
        logger.warning("Failed to write mining aggregates: %s", exc)

    chunks = _build_chunks(aggregates, langs, patch)
    try:
        save_learned_file(
            "mining_patterns",
            chunks,
            meta={"source": source, "matches": mined, "patch": patch},
        )
    except Exception as exc:
        logger.warning("Failed to save mining_patterns chunks: %s", exc)
        return {"chunks_written": 0, "matches_mined": len(mined), "match_ids": mined}

    return {
        "chunks_written": len(chunks),
        "matches_mined": len(mined),
        "match_ids": mined,
    }
