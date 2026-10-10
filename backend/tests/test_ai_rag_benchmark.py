"""Retrieval regression benchmark and the committed vector file.

78 labelled real decision points (backend/tests/data/rag_benchmark) are
replayed through the production ranker in the state a fresh deployment is
in: committed game data, committed vector file, numpy only, no embedding
model. Network-free and model-free.
"""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import pytest

from backend.app.ai import data_manager, knowledge_base
from backend.app.ai.knowledge_store import is_entity_scoped, is_learned_source
from backend.app.ai.rag import embedders, evalkit, runtime, situations, vectors

# Measured on 2026-10-11 with the lexical formula and the situation-tag rules
# (python -m backend.scripts.rag eval): nDCG@10 0.857, nDCG at the budget 0.852.
# With the query tags of the labelling run the same queries score 0.801 / 0.802.
LEXICAL_NDCG_10 = 0.857
LEXICAL_NDCG_BUDGET = 0.852
TOLERANCE = 0.01


@pytest.fixture
def fresh_deployment(tmp_path, monkeypatch):
    """Committed data and vectors only: no fetched game data, no learned files, no model."""
    from backend.app.ai.learning import base

    gamedata = tmp_path / "gamedata"
    gamedata.mkdir()
    learned = tmp_path / "learned"
    learned.mkdir()
    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", gamedata)
    monkeypatch.setattr(base, "LEARNED_DIR", learned)
    monkeypatch.setenv("RAG_EMBEDDINGS", "auto")
    monkeypatch.setenv("RAG_MODEL", embedders.DEFAULT_MODEL)
    monkeypatch.setenv("RAG_MODEL_DIR", str(tmp_path / "models"))     # no model here
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "rag-cache"))  # empty runtime cache
    for name in ("RAG_EMBED_API_BASE_URL", "RAG_EMBED_API_KEY", "RAG_EMBED_API_MODEL", "RAG_TOP_K"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runtime, "STATIC_DIR", None)
    data_manager.invalidate_all_caches()
    runtime.reset()
    yield
    data_manager.invalidate_all_caches()
    runtime.reset()


@pytest.fixture
def bench():
    return evalkit.load_benchmark()


def test_benchmark_is_complete_and_anonymous(bench):
    assert len(bench.queries) == 78 == len(bench.labels)
    assert {q["lang"] for q in bench.queries} == {"zh", "en"}
    strata = {q["stratum"] for q in bench.queries}
    assert strata == {"death_pick", "death_tf", "bad_trade", "good_play", "missed_cs", "overextend", "fight_engage"}
    for query in bench.queries:
        relevant = bench.labels[query["qid"]]
        assert relevant and set(relevant.values()) <= {1, 2, 3}
        assert set(relevant) <= bench.corpus_ids[query["lang"]]
        # A record carries the situation, never who played it
        assert not {"player_name", "persona", "account_id", "steam_id"} & set(query)
    size = sum(path.stat().st_size for path in evalkit.default_directory().iterdir() if path.is_file())
    assert size <= 1_500_000


def test_replayed_context_reproduces_the_labelled_query(bench, fresh_deployment):
    """The tag rules only ever add to / reweigh the query the labels were made for."""
    from backend.app.ai.knowledge_store import build_query

    for record in bench.queries:
        tags, _text = build_query(evalkit.context_from_record(record))
        baseline = record["baseline_tags"]
        dropped = set(baseline) - set(tags)
        teamfight_death = record["point_type"] == "death" and record["is_teamfight_death"]
        assert dropped == ({"mechanic:vision"} if teamfight_death else set()), record["qid"]
        assert all(tags[tag] >= weight for tag, weight in baseline.items() if tag in tags)


def test_lexical_ranking_keeps_the_measured_quality(bench, fresh_deployment):
    stores = evalkit.build_stores(bench)
    baseline = evalkit.evaluate(bench, stores, "baseline")["summary"]
    lexical = evalkit.evaluate(bench, stores, "lexical")
    assert lexical["hybrid_queries"] == 0 and lexical["unjudged@10"] == 0
    summary = lexical["summary"]
    assert summary["ndcg@10"] >= LEXICAL_NDCG_10 - TOLERANCE
    assert summary["ndcg@B"] >= LEXICAL_NDCG_BUDGET - TOLERANCE
    # ... and the situation-tag rules are what lifted it
    assert summary["ndcg@10"] >= baseline["ndcg@10"] + 0.03


def test_hybrid_with_the_committed_vectors_is_not_worse_than_lexical(bench, fresh_deployment):
    stores = evalkit.build_stores(bench)
    lexical = evalkit.evaluate(bench, stores, "lexical")
    hybrid = evalkit.evaluate(bench, stores, "hybrid")
    assert hybrid["hybrid_queries"] == len(bench.queries)   # numpy + the committed file are enough
    assert hybrid["summary"]["ndcg@10"] >= lexical["summary"]["ndcg@10"]
    assert hybrid["summary"]["ndcg@B"] >= lexical["summary"]["ndcg@B"]
    # Fewer irrelevant chunks reach the prompt, and both languages retrieve the same knowledge
    assert hybrid["summary"]["irr@B"] <= lexical["summary"]["irr@B"]
    assert evalkit.language_overlap(hybrid) >= evalkit.language_overlap(lexical)
    assert evalkit.language_overlap(hybrid) >= 0.9


def test_hybrid_puts_the_fitting_principle_on_top(bench, fresh_deployment):
    """The misses that motivated the change: principles for teamfight deaths and pick-offs."""
    stores = evalkit.build_stores(bench)
    after_laning = [q for q in bench.queries if q["phase"] in ("mid_game", "late_game")]
    teamfight = [q["qid"] for q in after_laning if q["stratum"] == "death_tf"]
    ganks = [q["qid"] for q in after_laning if q["stratum"] == "death_pick"]
    assert len(teamfight) >= 10 and len(ganks) >= 10

    def hits(mode, qids, chunk_id):
        rows = {row["qid"]: row for row in evalkit.evaluate(bench, stores, mode)["rows"]}
        return sum(chunk_id in rows[qid]["top10"] for qid in qids)

    assert hits("baseline", teamfight, "coaching:teamfight_positioning") == 0
    assert hits("baseline", ganks, "vision:safety_rules") == 0
    assert hits("hybrid", teamfight, "coaching:teamfight_positioning") == len(teamfight)
    assert hits("hybrid", ganks, "vision:safety_rules") == len(ganks)


def test_learned_principles_without_vectors_are_not_pushed_down(bench, fresh_deployment):
    """The default deployment has no embedding model: learned principle chunks never get a vector.

    With no dense credit at all they fell behind every covered principle
    (missed-CS nDCG@10 0.92 lexical -> 0.87, the player's own pro-pattern
    chunk from rank 2 to 13); with the neutral credit 0.89 and rank 5.
    """
    stores = evalkit.build_stores(bench)
    index = runtime.vector_index()
    uncovered = {c.chunk_id for c in stores["en"].chunks()
                 if c.index_text and not is_entity_scoped(c.tags)
                 and vectors.vector_key(index.model_id, vectors.DOC, c.index_text) not in index}
    assert len(uncovered) >= 10 and all(is_learned_source(stores["en"].get(c).source) for c in uncovered)
    lexical = {row["qid"]: row for row in evalkit.evaluate(bench, stores, "lexical")["rows"]}
    result = evalkit.evaluate(bench, stores, "hybrid")
    assert result["hybrid_queries"] == len(bench.queries)        # still hybrid: they do not count as missing
    missed_cs = [row for row in result["rows"] if row["stratum"] == "missed_cs"]
    assert len(missed_cs) == 9
    summary = evalkit.summarize(missed_cs)
    reference = evalkit.summarize([lexical[row["qid"]] for row in missed_cs])
    assert summary["ndcg@10"] >= 0.88 and summary["ndcg@10"] >= reference["ndcg@10"] - 0.03
    assert summary["ndcg@B"] >= reference["ndcg@B"]
    for row in missed_cs:                                        # the benchmark of the player's own position
        own = [chunk_id for chunk_id, grade in bench.labels[row["qid"]].items()
               if grade == 3 and chunk_id.startswith("learned:pro_pattern:")]
        assert len(own) == 1 and own[0][:-3] + ":en" in uncovered
        assert own[0] in row["top10"][:6], (row["qid"], row["top10"])


def test_pregame_decisions_get_the_rune_fight_principles(bench, fresh_deployment):
    """Every decision before the horn is about the rune fight (all five come from one real rune fight)."""
    stores = evalkit.build_stores(bench)
    pregame = [q["qid"] for q in bench.queries if q["phase"] == "pregame"]
    assert len(pregame) == 5
    results = {mode: {row["qid"]: row for row in evalkit.evaluate(bench, stores, mode)["rows"]}
               for mode in ("lexical", "hybrid")}
    for qid in pregame:
        row = results["hybrid"][qid]
        assert row["top10"][0] == "early_game:rune_contest_framework", row["top10"]
        assert {"early_game:rune_contest_framework", "early_game:level1_fight_assessment",
                "early_game:pregame_principles"} <= set(row["selected"]), row["selected"]
        assert "early_game:rune_contest_framework" in results["lexical"][qid]["selected"]
    for metric in ("ndcg@10", "ndcg@B"):                         # the slice where hybrid used to lose
        hybrid = evalkit.summarize([results["hybrid"][qid] for qid in pregame])[metric]
        lexical = evalkit.summarize([results["lexical"][qid] for qid in pregame])[metric]
        assert hybrid >= lexical >= 0.65, (metric, hybrid, lexical)


def test_teamfight_deaths_before_the_mid_game_keep_the_positioning_principle(bench, fresh_deployment):
    stores = evalkit.build_stores(bench)
    early = [q["qid"] for q in bench.queries if q["stratum"] == "death_tf" and q["phase"] in ("pregame", "laning")]
    assert len(early) == 7 and all(bench.labels[qid]["coaching:teamfight_positioning"] == 3 for qid in early)
    rows = {row["qid"]: row for row in evalkit.evaluate(bench, stores, "hybrid")["rows"]}
    in_block = sum("coaching:teamfight_positioning" in rows[qid]["selected"] for qid in early)
    assert in_block >= 6                                         # it was in none of the seven


@pytest.mark.parametrize("covered", [0.5, 0.1])
def test_sparse_vector_file_is_never_worse_than_lexical(bench, fresh_deployment, tmp_path, monkeypatch, covered):
    """A stale / half-built file: below the coverage floor the dense term is not used at all.

    Used anyway, it scored 0.840 (50 % of the principles covered) and 0.795
    (10 %) against the lexical 0.840 when this was measured.
    """
    np = pytest.importorskip("numpy")
    stores = evalkit.build_stores(bench)
    lexical = evalkit.evaluate(bench, stores, "lexical")["summary"]
    model = embedders.DEFAULT_MODEL
    spec = embedders.MODEL_REGISTRY[model]
    committed = runtime.static_path(runtime.settings(), model)
    keys, matrix, _meta = vectors.read_vector_file(committed, model, spec.dim, spec.params)
    principles = sorted({vectors.vector_key(model, vectors.DOC, c.index_text) for c in stores["en"].chunks()
                         if c.index_text and not is_entity_scoped(c.tags)} & set(keys))
    assert len(principles) >= 50
    dropped = set(principles[int(len(principles) * covered):])
    rows = [i for i, key in enumerate(keys) if key not in dropped]
    vectors.write_vector_file(tmp_path / committed.name, model, [keys[i] for i in rows], np.asarray(matrix)[rows],
                              params=spec.params)
    monkeypatch.setattr(runtime, "STATIC_DIR", tmp_path)
    runtime.reset()
    sparse = evalkit.evaluate(bench, stores, "hybrid")
    assert sparse["hybrid_queries"] == 0
    assert sparse["summary"] == lexical


def test_top_k_caps_the_selection(bench, fresh_deployment):
    stores = evalkit.build_stores(bench)
    capped = evalkit.evaluate(bench, stores, "hybrid", top_k=6)
    assert max(len(row["selected"]) for row in capped["rows"]) == 6


# ---------------------------------------------------------------------------
# The committed vector file
# ---------------------------------------------------------------------------

REBUILD_HINT = "run `python -m backend.scripts.rag build --static` and commit backend/data/rag/"


def test_committed_vector_file_covers_the_static_corpus(fresh_deployment):
    cfg = runtime.settings()
    path = runtime.static_path(cfg, embedders.DEFAULT_MODEL)
    assert path.is_file(), f"{path} is missing: {REBUILD_HINT}"
    assert path.stat().st_size <= 1_500_000
    spec = embedders.MODEL_REGISTRY[embedders.DEFAULT_MODEL]
    try:   # format version, model, embedder parameters, dimension, finite unit vectors
        keys, matrix, meta = vectors.read_vector_file(path, spec.model_id, spec.dim, spec.params)
    except vectors.VectorFileError as error:
        pytest.fail(f"{error}: {REBUILD_HINT}")
    assert matrix.dtype.name == "float16" and meta["model"] == embedders.DEFAULT_MODEL
    assert meta["params"] == spec.params and meta["arch"]
    norms = (matrix.astype("float32") ** 2).sum(axis=1) ** 0.5
    assert abs(norms - 1.0).max() < 5e-3                    # L2-normalised
    index = vectors.VectorIndex(embedders.DEFAULT_MODEL, path, None)

    missing_queries = [text for text in situations.situation_queries()
                       if vectors.vector_key(index.model_id, vectors.QUERY, text) not in index]
    assert not missing_queries, f"{len(missing_queries)} situation queries have no vector: {REBUILD_HINT}"
    assert len(situations.situation_queries()) == 216
    assert len(keys) == len(set(keys))

    texts = knowledge_base.static_index_texts()
    share = vectors.coverage(index, index.model_id, texts.values())
    assert share >= 0.95, f"only {share:.1%} of the static corpus has a vector: {REBUILD_HINT}"
    # Decision ranking only reads the generic chunks: none of those may be missing.
    store = knowledge_base._get_knowledge_store("en")
    generic = [c for c in store.chunks() if not is_entity_scoped(c.tags)]
    uncovered = [c.chunk_id for c in generic
                 if vectors.vector_key(index.model_id, vectors.DOC, c.index_text) not in index]
    assert len(generic) >= 50 and not uncovered, f"{uncovered}: {REBUILD_HINT}"


def test_zh_and_en_stores_share_one_index_text(fresh_deployment):
    stores = knowledge_base.warm_stores()
    english = {c.chunk_id: c for c in stores["en"].chunks()}
    for chunk in stores["zh"].chunks():
        assert chunk.index_text == english[chunk.chunk_id].content == english[chunk.chunk_id].index_text
    # The Chinese text itself is never what gets embedded
    principle = stores["zh"].get("coaching:teamfight_positioning")
    assert "团战" in principle.content and "Teamfight Positioning" in principle.index_text
