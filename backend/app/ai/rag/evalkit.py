"""Retrieval benchmark: replay labelled decisions through the production ranker.

The benchmark (``backend/tests/data/rag_benchmark/``) holds 78 real decision
points with complete relevance labels (grades 0-3, see its README and
RUBRIC). Replaying one means: rebuild the :class:`DecisionContext` from the
stored record, run the same code a report runs (``build_query`` ->
``KnowledgeStore.rank`` -> ``select`` under the stored token budget) and
score what comes back.

Modes:

``baseline``  the query tags as they were before the situation-tag rules,
              lexical formula — the ranker the labels were collected against
``lexical``   today's ``build_query``, lexical formula (no vectors)
``hybrid``    today's ``build_query`` + the dense term from the vector files

Metrics per query, averaged: nDCG (gain ``2^grade - 1``), ``p2`` = share of
returned chunks graded >= 2, ``irr`` = share graded 0, ``rec2`` = recall of
the chunks graded >= 2; ``@5`` / ``@10`` on the ranking, ``@B`` on what fits
the token budget. Chunks the labels do not know (added to the corpus after
labelling) are left out of the scored lists and counted as ``unjudged``.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

MODES = ("baseline", "lexical", "hybrid")
SUMMARY_KEYS = ("ndcg@5", "ndcg@10", "p2@10", "irr@10", "rec2@10",
                "ndcg@B", "p2@B", "irr@B", "rec2@B", "n@B", "tokens@B")


def default_directory() -> Path:
    return Path(__file__).resolve().parents[3] / "tests" / "data" / "rag_benchmark"


@dataclass
class Benchmark:
    meta: dict
    queries: list[dict]
    labels: dict[str, dict[str, int]]           # qid -> {chunk_id: grade >= 1}
    corpus_ids: dict[str, frozenset]            # lang -> ids of the labelled corpus
    learned: dict[str, list[dict]] = field(default_factory=dict)   # lang -> learned chunk dicts


def load_benchmark(directory: Path | None = None) -> Benchmark:
    directory = Path(directory) if directory else default_directory()

    def read(name: str):
        with open(directory / name, "r", encoding="utf-8") as f:
            return json.load(f)

    queries = read("queries.json")
    corpus = read("corpus.json")
    return Benchmark(
        meta=queries["meta"],
        queries=queries["queries"],
        labels=read("labels.json"),
        corpus_ids={lang: frozenset(ids) for lang, ids in corpus["ids"].items()},
        learned=corpus["learned"],
    )


def context_from_record(record: dict):
    """The :class:`DecisionContext` of a benchmark record (what retrieval reads of it)."""
    from ..models import DeathEvent, DecisionContext, DecisionPoint, HeroState

    death = None
    if record["point_type"] == "death":
        death = DeathEvent(
            time=record["time"], player_slot=0, hero_name=record["hero"],
            killer_hero=record.get("killer") or "",
            is_teamfight=bool(record.get("is_teamfight_death")),
            overextended=bool(record.get("overextended")))
    point = DecisionPoint(
        time=record["time"], point_type=record["point_type"], player_slot=0,
        hero_name=record["hero"], severity=record.get("severity") or "major",
        brief=record.get("brief") or "", related_death=death,
        time_known=bool(record.get("time_known", True)), subtype=record.get("subtype") or "")
    ctx = DecisionContext(
        decision_point=point,
        player_state=HeroState(hero_name=record["hero"], player_slot=0, team="radiant",
                               items=list(record.get("items") or [])),
        enemy_states=[HeroState(hero_name=name, player_slot=128 + i, team="dire")
                      for i, name in enumerate(record.get("near_enemies") or [])],
        game_phase=record.get("phase") or "",
    )
    ctx.match_context.update({
        "player_position": record.get("position"),
        "vision_danger": record.get("vision_danger"),
        "gold_adv": record.get("gold_adv", 0),
    })
    return ctx


def build_stores(bench: Benchmark) -> dict:
    """Stores to replay against: today's static corpus + the benchmark's frozen learned layer."""
    from .. import knowledge_base

    return knowledge_base.build_stores(
        learned=bench.learned, operator_files=False, current_patch=bench.meta.get("patch") or None)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _dcg(grades) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def list_metrics(ids: list[str], relevant: dict[str, int], k: int | None = None) -> dict:
    """Metrics of a returned id list (cut at *k*) against complete labels."""
    ids = ids[:k] if k else ids
    grades = [relevant.get(chunk_id, 0) for chunk_id in ids]
    n = len(grades)
    ideal = sorted(relevant.values(), reverse=True)[: (k or n)]
    total = sum(1 for g in relevant.values() if g >= 2)
    return {
        "n": n,
        "ndcg": _dcg(grades) / _dcg(ideal) if ideal and _dcg(ideal) > 0 else 0.0,
        "p2": sum(g >= 2 for g in grades) / n if n else 0.0,
        "irr": sum(g == 0 for g in grades) / n if n else 0.0,
        "rec2": sum(g >= 2 for g in grades) / total if total else 0.0,
    }


def _mean(rows: list[dict], key: str) -> float:
    values = [row[key] for row in rows if key in row]
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict]) -> dict:
    return {key: round(_mean(rows, key), 4) for key in SUMMARY_KEYS}


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def evaluate(bench: Benchmark, stores: dict, mode: str = "hybrid", *, handles: bool = True,
             top_k: int = 0) -> dict:
    """Replay every query in *mode*; returns ``{"mode", "rows", "summary", ...}``.

    *handles* charges each selected chunk its ``[K#]`` line like the real
    assembly does (``False`` reproduces the budget of the labelling run);
    *top_k* > 0 caps the selection. ``hybrid_queries`` counts the queries
    that really were ranked with vectors.
    """
    from .. import knowledge_base
    from ..knowledge_store import build_query

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    rows: list[dict] = []
    unjudged = 0
    hybrid_queries = 0
    seconds = 0.0
    for record in bench.queries:
        lang = record["lang"]
        store = stores[lang]
        ctx = context_from_record(record)
        started = time.perf_counter()
        if mode == "baseline":
            ranked = store.rank(record["baseline_tags"], build_query(ctx)[1])
        else:
            ranked, info = knowledge_base._rank_for_context(ctx, store, dense=mode == "hybrid")
            hybrid_queries += info["mode"] == "hybrid"
        seconds += time.perf_counter() - started
        printed = set(record.get("printed_item_chunks") or [])
        ranked = [(chunk, score) for chunk, score in ranked if chunk.chunk_id not in printed]
        overhead = (lambda chunk: knowledge_base._entry_overhead(chunk, lang)) if handles else None
        selected = store.select(ranked, record["remaining_budget"], overhead=overhead, max_chunks=top_k)

        known = bench.corpus_ids[lang]
        ids = [chunk.chunk_id for chunk, _ in ranked if chunk.chunk_id in known]
        unjudged += sum(1 for chunk, _ in ranked[:10] if chunk.chunk_id not in known)
        relevant = bench.labels[record["qid"]]
        row = {"qid": record["qid"], "lang": lang, "stratum": record["stratum"],
               "point_key": record.get("point_key", ""), "top10": ids[:10]}
        for k in (5, 10):
            for name, value in list_metrics(ids, relevant, k).items():
                row[f"{name}@{k}"] = value
        judged = [chunk for chunk in selected if chunk.chunk_id in known]
        for name, value in list_metrics([chunk.chunk_id for chunk in judged], relevant).items():
            row[f"{name}@B"] = value
        row["tokens@B"] = sum(chunk.token_estimate for chunk in selected)
        row["selected"] = [chunk.chunk_id for chunk in selected]
        rows.append(row)
    return {
        "mode": mode,
        "rows": rows,
        "summary": summarize(rows),
        "queries": len(rows),
        "hybrid_queries": hybrid_queries,
        "unjudged@10": unjudged,
        "rank_ms": round(1000 * seconds / max(1, len(rows)), 3),
    }


def language_overlap(result: dict) -> float:
    """Mean Jaccard overlap of the top 10 of the same decision asked in zh and in en."""
    by_point: dict[str, dict[str, set]] = {}
    for row in result["rows"]:
        if row["point_key"]:
            by_point.setdefault(row["point_key"], {})[row["lang"]] = {
                _language_free(chunk_id) for chunk_id in row["top10"]}
    overlaps = []
    for pair in by_point.values():
        if len(pair) == 2:
            zh, en = pair["zh"], pair["en"]
            overlaps.append(len(zh & en) / len(zh | en) if zh | en else 1.0)
    return round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0


def _language_free(chunk_id: str) -> str:
    return chunk_id[:-3] if chunk_id.startswith("learned:") and chunk_id[-3:] in (":zh", ":en") else chunk_id


def format_table(results: list[dict], split: bool = False) -> str:
    """Markdown table of the summaries (one row per result; with *split* also per language / stratum)."""
    lines = ["| mode | " + " | ".join(SUMMARY_KEYS) + " |", "|---|" + "---|" * len(SUMMARY_KEYS)]

    def line(name: str, summary: dict) -> str:
        cells = [f"{summary[key]:.1f}" if key.startswith(("n@", "tokens@")) else f"{summary[key]:.3f}"
                 for key in SUMMARY_KEYS]
        return f"| {name} | " + " | ".join(cells) + " |"

    for result in results:
        lines.append(line(result["mode"], result["summary"]))
        if split:
            for field_ in ("lang", "stratum"):
                for value in sorted({row[field_] for row in result["rows"]}):
                    rows = [row for row in result["rows"] if row[field_] == value]
                    lines.append(line(f"{result['mode']} / {value} (n={len(rows)})", summarize(rows)))
    return "\n".join(lines)


__all__ = ["Benchmark", "MODES", "SUMMARY_KEYS", "build_stores", "context_from_record", "evaluate",
           "format_table", "language_overlap", "list_metrics", "load_benchmark", "summarize"]
