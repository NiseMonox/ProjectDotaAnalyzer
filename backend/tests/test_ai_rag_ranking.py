"""Pooled hybrid ranking, the fallback matrix, and the runtime (status / warm_up / refresh).

Vectors come from the deterministic hash embedder; no model, no network.
"""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from backend.app import osutil
from backend.app.ai import data_manager, knowledge_base, rag
from backend.app.ai.knowledge_store import KnowledgeChunk, KnowledgeStore, build_query
from backend.app.ai.models import DeathEvent, DecisionContext, DecisionPoint, HeroState
from backend.app.ai.rag import embedders, runtime, situations, vectors

STATUS_KEYS = {"mode", "embedder", "vectors", "chunks", "static_coverage", "building", "last_build", "error",
               "embedder_available", "embedder_reason"}


@pytest.fixture
def rag_env(tmp_path, monkeypatch):
    """Isolated data + vector directories; ``RAG_EMBEDDINGS=hash`` unless a test changes it."""
    from backend.app.ai.learning import base

    gamedata = tmp_path / "gamedata"
    gamedata.mkdir()
    learned = tmp_path / "learned"
    learned.mkdir()
    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", gamedata)
    monkeypatch.setattr(base, "LEARNED_DIR", learned)
    monkeypatch.setattr(runtime, "STATIC_DIR", tmp_path / "static")
    monkeypatch.setattr(vectors, "CHECK_INTERVAL_S", 0.0)
    monkeypatch.setenv("RAG_EMBEDDINGS", "hash")
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("RAG_MODEL_DIR", str(tmp_path / "models"))
    for name in ("RAG_MODEL", "RAG_EMBED_API_BASE_URL", "RAG_EMBED_API_KEY", "RAG_EMBED_API_MODEL", "RAG_TOP_K"):
        monkeypatch.delenv(name, raising=False)
    data_manager.invalidate_all_caches()
    runtime.reset()
    yield SimpleNamespace(root=tmp_path, learned=learned, cache=tmp_path / "cache" / "hash-256.npz",
                          static=tmp_path / "static" / "hash-256.npz")
    runtime.shutdown()
    assert runtime.wait_idle(30)
    runtime.reset()
    data_manager.invalidate_all_caches()


def _ctx(hero="Axe", point_type="death", killer="Viper", enemies=("Viper",), phase="mid_game",
         position=3, teamfight=True, lang_brief="Killed by Viper in a teamfight"):
    death = DeathEvent(time=1500, player_slot=0, hero_name=hero, killer_hero=killer,
                       is_teamfight=teamfight) if point_type == "death" else None
    ctx = DecisionContext(
        decision_point=DecisionPoint(time=1500, point_type=point_type, player_slot=0, hero_name=hero,
                                     severity="major", brief=lang_brief, related_death=death),
        player_state=HeroState(hero_name=hero, player_slot=0, team="radiant"),
        enemy_states=[HeroState(hero_name=e, player_slot=128 + i, team="dire") for i, e in enumerate(enemies)],
        game_phase=phase,
    )
    ctx.match_context["player_position"] = position
    return ctx


def _entry_ids(text):
    return [ref.chunk_id for ref in text.refs]


# ---------------------------------------------------------------------------
# The formula (hand-made dense scores)
# ---------------------------------------------------------------------------

def _formula_store():
    store = KnowledgeStore()
    for chunk in (
        KnowledgeChunk("hero:axe", "Axe initiates with Blink and Call.", category="a", tags={"hero:axe"}),
        KnowledgeChunk("g:best", "Stand behind your front line in fights.", category="b", tags={"mechanic:teamfight"}),
        KnowledgeChunk("g:mid", "Buy back only for decisive fights.", category="c", tags={"mechanic:teamfight"}),
        KnowledgeChunk("g:worst", "Stack camps at 53 seconds.", category="d", tags={"mechanic:teamfight"}),
        KnowledgeChunk("g:novector", "Count enemy heroes before fighting.", category="e", tags={"mechanic:teamfight"}),
        KnowledgeChunk("hero:invoker", "Invoker combos.", category="f", tags={"hero:invoker"}),
    ):
        store.add_chunk(chunk)
    return store


def test_pooled_formula():
    store = _formula_store()
    tags = {"hero:axe": 1.0, "mechanic:teamfight": 3.0}
    dense = {"g:best": 0.80, "g:mid": 0.70, "g:worst": 0.60,
             "hero:axe": 0.99, "hero:invoker": 0.99, "not:in:store": 0.9}
    scores = {chunk.chunk_id: score for chunk, score in store.rank(tags, dense=dense)}
    lexical = {chunk.chunk_id: score for chunk, score in store.rank(tags)}
    tag_part = {chunk_id: score / 0.7 for chunk_id, score in lexical.items()}   # sqrt(tag / best tag)
    assert tag_part["g:best"] == pytest.approx(1.0) and 0.5 < tag_part["hero:axe"] < 1.0

    # Entity chunk: the constant 0.5 (its own cosine is not read)
    assert scores["hero:axe"] == pytest.approx(0.6 * tag_part["hero:axe"] + 0.3 * 0.5)
    # Generic pool: cosine min-max normalised INSIDE the pool
    assert scores["g:best"] == pytest.approx(0.6 + 0.3 * 1.0)
    assert scores["g:mid"] == pytest.approx(0.6 + 0.3 * 0.5)
    assert scores["g:worst"] == pytest.approx(0.6 + 0.3 * 0.0)
    # A generic chunk without a vector is not ranked by cosine either: the same neutral 0.5
    # (0 put the learned / operator principles of a model-less server behind every covered one)
    assert scores["g:novector"] == pytest.approx(0.6 + 0.3 * 0.5) == scores["g:mid"]
    assert "hero:invoker" not in scores                           # the entity gate is untouched
    # Without dense scores every generic chunk scores the same: only the vectors separate them
    assert lexical["g:best"] == lexical["g:mid"] == lexical["g:worst"] == lexical["g:novector"]


def test_dense_scores_for_entity_chunks_only_mean_lexical():
    store = _formula_store()
    tags = {"hero:axe": 3.0, "mechanic:teamfight": 1.0}
    assert store.rank(tags, dense={"hero:axe": 0.9}) == store.rank(tags) == store.rank(tags, dense={})
    assert store.rank(tags)[0][1] == pytest.approx(0.7)           # 0.7 * tag, no dense share


def test_hybrid_uses_the_tighter_relative_cutoff():
    store = KnowledgeStore()
    store.add_chunk(KnowledgeChunk("hero:axe", "Axe", category="a", tags={"hero:axe"}))
    store.add_chunk(KnowledgeChunk("g:weak", "generic one", category="b", tags={"phase:laning"}))
    store.add_chunk(KnowledgeChunk("g:other", "generic two", category="c", tags={"phase:laning"}))
    tags = {"hero:axe": 3.0, "phase:laning": 0.5}
    lexical = dict((c.chunk_id, s) for c, s in store.rank(tags))
    assert 0.3 * lexical["hero:axe"] <= lexical["g:weak"] < 0.5 * lexical["hero:axe"]   # kept at 0.3
    hybrid = dict((c.chunk_id, s) for c, s in store.rank(tags, dense={"g:weak": 0.1, "g:other": 0.9}))
    assert "g:weak" not in hybrid and "g:other" in hybrid         # below 0.5 * best: cut


# ---------------------------------------------------------------------------
# End to end with vectors
# ---------------------------------------------------------------------------

def test_hybrid_block_after_a_build(rag_env):
    lexical = knowledge_base.get_knowledge_for_context(_ctx(), lang="en")
    assert lexical.retrieval["mode"] == "lexical" and rag.status()["mode"] == "lexical"

    stats = runtime.sync("test")
    assert stats["embedded"] > 700 and rag_env.cache.is_file()
    text = knowledge_base.get_knowledge_for_context(_ctx(), lang="en")
    assert text.retrieval["mode"] == "hybrid"
    assert text.retrieval["situation"].startswith("position 3 offlane, mid game: teamfight positioning")
    status = rag.status()
    assert set(status) == STATUS_KEYS
    assert status["mode"] == "hybrid" and status["embedder"] == "hash-256" and status["error"] is None
    assert status["chunks"] == len(knowledge_base._get_knowledge_store("en")) > 700
    assert status["vectors"] >= status["chunks"] and status["static_coverage"] == 1.0
    assert status["building"] is False and status["last_build"].endswith("Z")
    assert status["embedder_available"] is True and status["embedder_reason"] == ""
    json.dumps(status)
    # A store the caller holds is described as it is, whatever the cached ones are by now
    held = knowledge_base._get_knowledge_store("en")
    knowledge_base.reset_knowledge_stores()
    assert rag.status()["chunks"] == 0 and rag.status()["mode"] == "lexical"
    assert rag.status(held) == status

    # A second sync embeds nothing and leaves the file alone
    before = rag_env.cache.stat().st_mtime_ns
    assert runtime.sync("again")["embedded"] == 0 and rag_env.cache.stat().st_mtime_ns == before


def test_both_languages_retrieve_the_same_knowledge(rag_env):
    runtime.sync("test")
    english = knowledge_base.get_knowledge_for_context(_ctx(lang_brief="Killed by Viper in a teamfight"), lang="en")
    chinese = knowledge_base.get_knowledge_for_context(_ctx(lang_brief="团战中被Viper击杀"), lang="zh")
    assert english.retrieval["mode"] == chinese.retrieval["mode"] == "hybrid"
    assert english.retrieval["situation"] == chinese.retrieval["situation"]
    stores = knowledge_base.warm_stores()
    query, scores_en = rag.situation_scores(stores["en"], _ctx())
    _, scores_zh = rag.situation_scores(stores["zh"], _ctx(lang_brief="团战中被Viper击杀"))
    assert scores_en == scores_zh and len(scores_en) >= 50         # one vector per chunk id
    assert "团战站位" in chinese and "Teamfight Positioning" in english


# ---------------------------------------------------------------------------
# Fallback matrix: none of these fails a report
# ---------------------------------------------------------------------------

def _lexical_reference(ctx, lang="en"):
    store = knowledge_base._get_knowledge_store(lang)
    return [chunk.chunk_id for chunk, _ in store.rank(*build_query(ctx))]


def _ranked_ids(ctx, lang="en"):
    store = knowledge_base._get_knowledge_store(lang)
    ranked, info = knowledge_base._rank_for_context(ctx, store)
    return [chunk.chunk_id for chunk, _ in ranked], info["mode"]


def test_fallback_embeddings_off(rag_env, monkeypatch):
    runtime.sync("test")
    assert _ranked_ids(_ctx())[1] == "hybrid"
    monkeypatch.setenv("RAG_EMBEDDINGS", "off")
    assert _ranked_ids(_ctx()) == (_lexical_reference(_ctx()), "lexical")
    status = rag.status()
    assert (status["mode"], status["embedder"], status["vectors"], status["error"]) == ("lexical", None, 0, None)
    assert runtime.sync("off") == {"not_run": "off"}
    text = knowledge_base.get_knowledge_for_context(_ctx(), lang="en")
    assert text.refs and "[K1] " in text                         # the numbered block does not need vectors


@pytest.mark.parametrize("packages, reason", [(True, "model_missing"), (False, "deps_missing")])
def test_fallback_no_vector_file_and_no_model(rag_env, monkeypatch, packages, reason):
    # With and without the optional packages (onnxruntime / tokenizers), whatever this venv holds
    monkeypatch.setattr(embedders, "_importable", lambda *modules: packages)
    monkeypatch.setenv("RAG_EMBEDDINGS", "auto")                 # default model, nothing downloaded, no file
    assert _ranked_ids(_ctx()) == (_lexical_reference(_ctx()), "lexical")
    status = rag.status()
    assert status["mode"] == "lexical" and status["embedder"] is None and status["error"] is None
    assert runtime.sync("no model") == {"not_run": reason}
    monkeypatch.setenv("RAG_EMBEDDINGS", "onnx")                 # explicitly requested: reported
    assert rag.status()["error"] == reason


def test_fallback_numpy_missing(rag_env, monkeypatch):
    runtime.sync("test")
    hybrid_ids, mode = _ranked_ids(_ctx())
    assert mode == "hybrid"
    monkeypatch.setitem(sys.modules, "numpy", None)              # "import numpy" now raises ImportError
    assert vectors.numpy_or_none() is None
    assert _ranked_ids(_ctx()) == (_lexical_reference(_ctx()), "lexical")
    assert rag.situation_scores(knowledge_base._get_knowledge_store("en"), _ctx())[1] is None
    status = rag.status()
    assert status["mode"] == "lexical" and status["vectors"] == 0 and status["error"] == "numpy_missing"
    assert runtime.sync("no numpy") == {"not_run": "deps_missing"}
    assert knowledge_base.get_knowledge_for_context(_ctx(), lang="zh").refs


@pytest.mark.parametrize("damage", ["corrupt", "wrong_dimension", "other_model"])
def test_fallback_unusable_vector_file(rag_env, monkeypatch, damage, caplog):
    np = pytest.importorskip("numpy")
    monkeypatch.setenv("RAG_EMBEDDINGS", "auto")                 # the default model's files, numpy only
    path = rag_env.root / "static" / f"{embedders.DEFAULT_MODEL}.npz"
    path.parent.mkdir(parents=True)
    if damage == "corrupt":
        path.write_bytes(b"\x00\x01 not a vector file")
    else:
        model = embedders.DEFAULT_MODEL if damage == "wrong_dimension" else "another-model"
        key = vectors.vector_key(model, vectors.QUERY, situations.situation_queries()[0])
        vectors.write_vector_file(path, model, [key], np.ones((1, 16 if damage == "wrong_dimension" else 384)))
    with caplog.at_level("WARNING"):
        assert _ranked_ids(_ctx()) == (_lexical_reference(_ctx()), "lexical")
        assert _ranked_ids(_ctx(teamfight=False))[1] == "lexical"
    assert sum("Ignoring vector file" in record.message for record in caplog.records) == 1
    status = rag.status()
    assert status["mode"] == "lexical" and status["vectors"] == 0
    assert status["error"].startswith(f"committed file {path.name}: ")


# ---------------------------------------------------------------------------
# Vectors that would mis-rank: lexical, and the status says why
# ---------------------------------------------------------------------------

def _generic_keys(store, model="hash-256"):
    """``{chunk id: vector key}`` of the store's generic (principle) chunks."""
    from backend.app.ai.knowledge_store import is_entity_scoped
    return {c.chunk_id: vectors.vector_key(model, vectors.DOC, c.index_text)
            for c in store.chunks() if c.index_text and not is_entity_scoped(c.tags)}


def _rewrite_cache(rag_env, keep, matrix=None):
    """Rewrite the runtime cache with the rows ``keep(key)`` accepts (optionally another matrix)."""
    np = pytest.importorskip("numpy")
    keys, stored, _ = vectors.read_vector_file(rag_env.cache, "hash-256")
    stored = np.asarray(stored if matrix is None else matrix)
    rows = [i for i, key in enumerate(keys) if keep(key)]
    vectors.write_vector_file(rag_env.cache, "hash-256", [keys[i] for i in rows], stored[rows])


@pytest.mark.parametrize("covered, mode", [(1.0, "hybrid"), (0.9, "hybrid"), (0.5, "lexical"), (0.1, "lexical")])
def test_sparse_principle_coverage_falls_back_to_lexical(rag_env, covered, mode):
    """A stale or half-built file: with too few principles covered the dense term mis-ranks the rest."""
    runtime.sync("test")
    store = knowledge_base._get_knowledge_store("en")
    generic = sorted(_generic_keys(store).values())
    dropped = set(generic[int(len(generic) * covered):])
    _rewrite_cache(rag_env, lambda key: key not in dropped)
    assert _ranked_ids(_ctx())[1] == mode and _ranked_ids(_ctx(teamfight=False))[1] == mode
    status = rag.status()
    assert status["mode"] == mode
    if mode == "lexical":
        assert _ranked_ids(_ctx())[0] == _lexical_reference(_ctx())
        kept = len(generic) - len(dropped)
        assert status["error"] == f"low_coverage: {kept} of {len(generic)} principle chunks have a vector"
        assert knowledge_base.get_knowledge_for_context(_ctx(), lang="en").retrieval["mode"] == "lexical"
    else:
        assert status["error"] is None
    assert runtime.MIN_POOL_COVERAGE == 0.8


def test_one_covered_principle_is_not_a_ranking(rag_env):
    runtime.sync("test")
    store = knowledge_base._get_knowledge_store("en")
    generic = _generic_keys(store)
    only = generic["coaching:teamfight_positioning"]
    _rewrite_cache(rag_env, lambda key: key == only or key not in set(generic.values()))
    assert _ranked_ids(_ctx()) == (_lexical_reference(_ctx()), "lexical")
    assert rag.status()["error"].startswith("low_coverage: 1 of ")


def test_vectors_that_cannot_tell_chunks_apart_are_not_used(rag_env):
    np = pytest.importorskip("numpy")
    runtime.sync("test")
    keys, matrix, _ = vectors.read_vector_file(rag_env.cache, "hash-256")
    same = np.tile(np.asarray(matrix)[:1], (len(keys), 1))       # finite, unit length, all identical
    _rewrite_cache(rag_env, lambda key: True, matrix=same)
    assert _ranked_ids(_ctx()) == (_lexical_reference(_ctx()), "lexical")
    status = rag.status()
    assert status["mode"] == "lexical" and status["error"].startswith("degenerate_vectors")
    # Not-a-number vectors never get that far: the file is refused
    broken = np.asarray(matrix, dtype=np.float32).copy()
    broken[:] = np.nan
    _rewrite_cache(rag_env, lambda key: True, matrix=broken)
    assert _ranked_ids(_ctx()) == (_lexical_reference(_ctx()), "lexical")
    status = rag.status()
    assert status["vectors"] == 0 and "not finite" in status["error"] and rag_env.cache.name in status["error"]


def test_status_mode_is_what_a_decision_would_be_ranked_with(rag_env):
    """Mode comes from the built stores' usable principle vectors, not from the mere presence of a file."""
    runtime.sync("test")
    assert rag.status()["mode"] == "hybrid"
    store = knowledge_base._get_knowledge_store("en")
    documents = {vectors.vector_key("hash-256", vectors.DOC, c.index_text) for c in store.chunks()}
    _rewrite_cache(rag_env, lambda key: key not in documents)    # the situation queries only
    assert vectors.read_vector_file(rag_env.cache, "hash-256")[0]
    assert _ranked_ids(_ctx())[1] == "lexical"
    status = rag.status()
    assert status["mode"] == "lexical" and status["static_coverage"] == 0.0
    assert status["error"] == "low_coverage: no principle chunk has a vector"
    # ... and the other way round: chunk vectors without a single situation query
    runtime.sync("rebuild")
    queries = {vectors.vector_key("hash-256", vectors.QUERY, text) for text in situations.situation_queries()}
    _rewrite_cache(rag_env, lambda key: key not in queries)
    status = rag.status()
    assert status["mode"] == "lexical" and status["error"].startswith("no_query_vectors")
    # Before any store is built nothing can be said: lexical, and no fault to report
    runtime.sync("rebuild")
    assert rag.status()["mode"] == "hybrid"
    knowledge_base.reset_knowledge_stores()
    status = rag.status()
    assert (status["mode"], status["chunks"], status["error"]) == ("lexical", 0, None)
    knowledge_base._get_knowledge_store("zh")                    # one built store is enough
    assert rag.status()["mode"] == "hybrid"


def test_learned_principle_without_a_vector_keeps_its_standing(rag_env):
    """A server without a model: learned / operator principles have no vector. Neutral credit, still hybrid."""
    from backend.app.ai.learning.base import write_json_atomic

    runtime.sync("test")
    write_json_atomic(rag_env.learned / "mining_patterns.json", {"meta": {}, "chunks": [
        {"chunk_id": f"learned:pro_pattern:position_3:{lang}", "content": content, "lang": lang,
         "category": "pro_pattern", "tags": ["role:offlane", "phase:mid_game", "mechanic:teamfight"],
         "priority": 2, "source": "match_mining", "patch_version": ""}
        for lang, content in (("en", "Pro offlaners join 71% of their team's fights by minute 20."),
                              ("zh", "职业三号位在20分钟前参与了71%的团战。"))]})
    knowledge_base.reset_knowledge_stores()
    store = knowledge_base._get_knowledge_store("en")            # the server has no embedder: no sync
    learned = store.get("learned:pro_pattern:position_3:en")
    assert learned.index_text and learned.source_type == "match_mining"
    query, scores = rag.situation_scores(store, _ctx())
    assert scores and learned.chunk_id not in scores             # no vector for it, the others have theirs
    assert rag.status()["mode"] == "hybrid" and rag.status()["error"] is None   # not counted against coverage
    tags, text = build_query(_ctx())

    def score(dense):
        return {c.chunk_id: s for c, s in store.rank(tags, text, dense=dense)}[learned.chunk_id]

    low, high = min(scores.values()), max(scores.values())
    without_vector = score(scores)
    # ... exactly what a chunk in the middle of the pool gets, and 0.3 * 0.5 above the worst-ranked one
    assert without_vector == pytest.approx(score({**scores, learned.chunk_id: (low + high) / 2}))
    assert without_vector == pytest.approx(score({**scores, learned.chunk_id: low}) + 0.3 * 0.5)


def test_fallback_chunk_without_a_vector_is_ranked_by_tags(rag_env):
    runtime.sync("test")
    store = knowledge_base._get_knowledge_store("en")
    store.add_chunk(KnowledgeChunk(
        "coaching:brand_new", "## Fresh Principle\n- Regroup before the next fight.", category="coaching_new",
        tags={"mechanic:teamfight", "mechanic:trade", "phase:mid_game"}))
    store.get("coaching:brand_new").index_text = "## Fresh Principle\n- Regroup before the next fight."
    query, scores = rag.situation_scores(store, _ctx())
    assert scores and "coaching:brand_new" not in scores          # no vector yet
    ranked = {chunk.chunk_id: score for chunk, score in store.rank(*build_query(_ctx()), dense=scores)}
    assert ranked.get("coaching:brand_new", 0) > 0                # still a candidate, on its tags
    # A chunk without English text never gets a vector, and is still served
    store.add_chunk(KnowledgeChunk("coaching:zh_only", "只有中文的团战原则。", category="coaching_zh",
                                   tags={"mechanic:teamfight", "mechanic:trade", "phase:mid_game"}))
    assert store.get("coaching:zh_only").index_text == ""
    runtime.sync("after edit")
    _, scores = rag.situation_scores(store, _ctx())
    assert "coaching:zh_only" not in scores
    assert "coaching:zh_only" in {c.chunk_id for c, _ in store.rank(*build_query(_ctx()), dense=scores)}


def test_fallback_situation_without_a_query_vector(rag_env):
    runtime.sync("test")
    assert _ranked_ids(_ctx())[1] == "hybrid"
    unknown = _ctx(point_type="ward_placed")                      # no topic for this point type
    assert rag.situation_scores(knowledge_base._get_knowledge_store("en"), unknown) == ("", None)
    assert _ranked_ids(unknown) == (_lexical_reference(unknown), "lexical")
    # A file that holds chunk vectors but not this query: lexical for the query
    np = pytest.importorskip("numpy")
    keys, matrix, _ = vectors.read_vector_file(rag_env.cache, "hash-256")
    wanted = vectors.vector_key("hash-256", vectors.QUERY, situations.describe(_ctx()).text)
    keep = [i for i, key in enumerate(keys) if key != wanted]
    vectors.write_vector_file(rag_env.cache, "hash-256", [keys[i] for i in keep], np.asarray(matrix)[keep])
    assert _ranked_ids(_ctx())[1] == "lexical"
    assert _ranked_ids(_ctx(teamfight=False))[1] == "hybrid"      # other situations are unaffected


def test_vector_file_written_by_another_process_is_picked_up(rag_env, monkeypatch):
    assert _ranked_ids(_ctx())[1] == "lexical"
    # "Another process" (the CLI) builds the committed file while this one keeps serving
    texts = knowledge_base.static_index_texts()
    wanted = [(vectors.DOC, text) for text in texts.values()]
    wanted += [(vectors.QUERY, text) for text in situations.situation_queries()]
    vectors.sync_vector_file(rag_env.static, embedders.HashEmbedder(), wanted)
    assert _ranked_ids(_ctx())[1] == "hybrid"
    assert rag.status()["static_coverage"] == 1.0
    rag_env.static.unlink()                                       # ... and removing it is survivable too
    assert _ranked_ids(_ctx())[1] == "lexical"


def test_nothing_in_the_request_path_raises(rag_env, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("index exploded")

    monkeypatch.setattr(runtime, "situation_scores", boom)
    monkeypatch.setattr(runtime, "details", boom)
    monkeypatch.setattr(runtime, "warm_up", boom)
    monkeypatch.setattr(runtime, "refresh", boom)
    assert rag.situation_scores(object(), _ctx()) == ("", None)
    status = rag.status()
    assert set(status) == STATUS_KEYS and status["mode"] == "lexical" and "index exploded" in status["error"]
    rag.warm_up()
    rag.refresh("x")
    text = knowledge_base.get_knowledge_for_context(_ctx(), lang="en")
    assert text.refs and text.retrieval["mode"] == "lexical"


# ---------------------------------------------------------------------------
# warm_up / refresh
# ---------------------------------------------------------------------------

def test_warm_up_builds_in_the_calling_thread_and_refresh_in_the_background(rag_env):
    from backend.app.ai.learning.base import write_json_atomic

    # Before a warm-up, refresh builds nothing (tests, one-shot scripts)
    knowledge_base._get_knowledge_store("en")
    rag.refresh("early")
    assert runtime.wait_idle(5) and not rag_env.cache.exists()

    rag.warm_up()
    assert rag_env.cache.is_file() and rag.status()["mode"] == "hybrid"
    assert set(knowledge_base.built_stores()) == {"en", "zh"}
    vectors_before = rag.status()["vectors"]

    # New learned chunks: the store rebuild itself asks for a background sync
    write_json_atomic(rag_env.learned / "llm_distilled.json", {"meta": {}, "chunks": [
        {"chunk_id": "learned:distilled:99:blink_timing", "content": content, "lang": lang,
         "category": "pro_pattern", "tags": ["role:offlane", "phase:mid_game"], "priority": 2,
         "source": "llm_distilled", "patch_version": ""}
        for lang, content in (("en", "Position 3 should buy Blink before minute 14 to start fights."),
                              ("zh", "三号位应在14分钟前买出跳刀来开团。"))]})
    knowledge_base.reset_knowledge_stores()
    store = knowledge_base._get_knowledge_store("zh")             # rebuild -> rag.refresh("stores_rebuilt")
    assert store.get("learned:distilled:99:blink_timing").index_text.startswith("[AI summary")
    assert runtime.wait_idle(30)
    assert rag.status()["vectors"] == vectors_before + 1          # one new text, embedded once for both languages
    _, scores = rag.situation_scores(store, _ctx())
    assert "learned:distilled:99:blink_timing" in scores


def test_warm_up_after_a_shutdown_builds_completely_again(rag_env):
    """shutdown() leaves a stop request behind; without clearing it every later sync ended after one batch."""
    from backend.app.ai.learning.base import write_json_atomic

    rag.warm_up()
    complete = rag.status()["vectors"]
    assert complete > 100
    rag.shutdown()
    assert runtime.wait_idle(30)
    rag_env.cache.unlink()                                        # e.g. the cache directory was cleared
    assert runtime.sync("after shutdown")["embedded"] <= 16       # still stopped: one batch at most
    rag.refresh("ignored")                                        # ... and no background build either
    assert runtime.wait_idle(5)
    assert rag.status()["vectors"] < complete

    rag.warm_up()                                                 # the next lifespan of the same process
    assert rag.status()["vectors"] == complete and rag.status()["mode"] == "hybrid"
    write_json_atomic(rag_env.learned / "llm_distilled.json", {"meta": {}, "chunks": [
        {"chunk_id": "learned:distilled:98:ward_timing", "content": "Position 5 should place the first ward early.",
         "lang": "en", "category": "pro_pattern", "tags": ["role:support_5", "phase:laning"], "priority": 2,
         "source": "llm_distilled", "patch_version": ""}]})
    knowledge_base.reset_knowledge_stores()
    knowledge_base._get_knowledge_store("en")                     # rebuild -> background sync, armed again
    assert runtime.wait_idle(30)
    assert rag.status()["vectors"] == complete + 1


def test_sync_waits_for_another_process_building_the_same_cache(rag_env, monkeypatch):
    """`rag build` or a learning run next to the server's warm-up: one of them embeds, the other waits."""
    monkeypatch.setattr(vectors, "LOCK_POLL_S", 0.01)
    embedded: list[str] = []

    class Spy(embedders.HashEmbedder):
        def embed_documents(self, texts):
            embedded.extend(texts)
            return super().embed_documents(texts)

        embed_queries = embed_documents

    spy = Spy()
    monkeypatch.setattr(runtime, "get_embedder", lambda cfg=None: (spy, ""))
    stores = knowledge_base.warm_stores()
    wanted = sorted({text for store in stores.values() for text in runtime._index_texts(store).values()})
    wanted = [(vectors.DOC, text) for text in wanted] + runtime._query_wanted()

    # The "other process": holds the lock of the cache file (the lock belongs to the open
    # file, so a second lock object in this process is refused exactly like another process's).
    rag_env.cache.parent.mkdir(parents=True)
    other = osutil.FileLock(rag_env.cache.with_name(rag_env.cache.name + ".lock"))
    assert other.acquire(blocking=False)
    results: list[dict] = []
    thread = threading.Thread(target=lambda: results.append(runtime.sync("warm_up")))
    try:
        thread.start()
        thread.join(0.3)
        assert thread.is_alive() and embedded == []               # waiting, nothing embedded, no model work
        vectors.sync_vector_file(rag_env.cache, embedders.HashEmbedder(), wanted)   # the other one finishes
    finally:
        other.release()                                           # ... and releases the lock
    thread.join(30)
    assert not thread.is_alive()
    stats = results[0]
    assert stats["embedded"] == 0 and stats["kept"] == len(wanted) and embedded == []
    assert rag.status()["mode"] == "hybrid"

    # shutdown() while waiting: the sync gives up instead of holding the exit back
    other = osutil.FileLock(rag_env.cache.with_name(rag_env.cache.name + ".lock"))
    assert other.acquire(blocking=False)
    try:
        thread = threading.Thread(target=lambda: results.append(runtime.sync("refresh")))
        thread.start()
        thread.join(0.2)
        assert thread.is_alive()
        runtime.shutdown()
        thread.join(10)
        assert not thread.is_alive() and results[-1] == {"not_run": "stopped"}
    finally:
        other.release()


def test_a_change_of_the_ranking_mode_is_logged_once(rag_env, caplog):
    """A server without the start-time warm-up, or one whose files broke later, must say how it ranks."""
    def lines():
        return [(r.levelname, r.getMessage()) for r in caplog.records
                if r.getMessage().startswith("Knowledge retrieval: decisions")]

    with caplog.at_level("INFO", logger="backend.app.ai.rag.runtime"):
        store = knowledge_base._get_knowledge_store("en")
        for _ in range(3):                                        # no vectors yet
            assert rag.situation_scores(store, _ctx())[1] is None
        assert lines() == [("INFO", "Knowledge retrieval: decisions are now ranked lexically "
                                    "(no vectors for this knowledge base)")]
        runtime.sync("test")
        for _ in range(3):
            assert rag.situation_scores(store, _ctx())[1]
        assert len(lines()) == 2 and lines()[1][0] == "INFO"
        assert lines()[1][1].startswith("Knowledge retrieval: decisions are now ranked with vectors (hybrid) "
                                        "(vector space hash-256, ")
        # The cache breaks while the server runs: a warning, with the reason, once
        keys, matrix, _meta = vectors.read_vector_file(rag_env.cache, "hash-256")
        docs = {vectors.vector_key("hash-256", vectors.DOC, text) for text in runtime._index_texts(store).values()}
        keep = [i for i, key in enumerate(keys) if key not in docs or i % 5 == 0]
        vectors.write_vector_file(rag_env.cache, "hash-256", [keys[i] for i in keep], matrix[keep])
        for _ in range(3):
            assert rag.situation_scores(store, _ctx())[1] is None
        assert len(lines()) == 3 and lines()[2][0] == "WARNING"
        assert "ranked lexically (low_coverage: " in lines()[2][1]
        # One situation without a query vector is not a change of mode
        runtime.sync("repair")
        assert rag.situation_scores(store, _ctx())[1] and len(lines()) == 4
        assert runtime.query_scores(store, "a situation nobody embedded") is None
        assert len(lines()) == 4


def test_only_one_sync_runs_at_a_time(rag_env, monkeypatch):
    started, release = threading.Event(), threading.Event()
    real = runtime._sync_once
    calls = []

    def slow(reason):
        calls.append(reason)
        started.set()
        release.wait(10)
        return real(reason)

    monkeypatch.setattr(runtime, "_sync_once", slow)
    worker = threading.Thread(target=runtime.sync, args=("first",))
    worker.start()
    assert started.wait(10)
    assert rag.status()["building"] is True
    assert runtime.sync("second") == {"not_run": "busy"}          # does not wait, does not run twice at once
    release.set()
    worker.join(30)
    assert calls == ["first", "first"]                            # ... but the running thread makes one more pass
    assert rag.status()["building"] is False


# ---------------------------------------------------------------------------
# Settings that cannot work are named in the status
# ---------------------------------------------------------------------------

def test_status_names_a_configuration_that_cannot_work(rag_env, monkeypatch):
    monkeypatch.setattr(embedders, "_importable", lambda *modules: True)   # as if the optional packages were there
    monkeypatch.setenv("RAG_EMBEDDINGS", "auto")
    assert rag.status()["error"] is None                         # no model, no file: the plain lexical default
    # An embeddings API with URL and key but no model name
    monkeypatch.setenv("RAG_EMBED_API_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("RAG_EMBED_API_KEY", "sk-embedding-key-0123456789")
    status = rag.status()
    assert (status["mode"], status["embedder"], status["error"]) == ("lexical", None, "api_not_configured")
    assert _ranked_ids(_ctx()) == (_lexical_reference(_ctx()), "lexical")
    monkeypatch.setenv("RAG_EMBED_API_MODEL", "embed-x")         # complete: nothing to complain about
    status = rag.status()
    assert (status["embedder"], status["error"]) == ("openai:embed-x", None)
    for name in ("RAG_EMBED_API_BASE_URL", "RAG_EMBED_API_KEY", "RAG_EMBED_API_MODEL"):
        monkeypatch.delenv(name)
    # A model name the registry does not know
    monkeypatch.setenv("RAG_MODEL", "bge-typo-v9")
    assert rag.status()["error"] == "unknown_model"
    monkeypatch.delenv("RAG_MODEL")
    # Switched off: silent, whatever else is set
    monkeypatch.setenv("RAG_EMBEDDINGS", "off")
    monkeypatch.setenv("RAG_MODEL", "bge-typo-v9")
    assert rag.status()["error"] is None


def test_invalid_mode_is_logged_once(rag_env, monkeypatch, caplog):
    monkeypatch.setenv("RAG_EMBEDDINGS", "bananas")
    with caplog.at_level("WARNING"):
        for _ in range(5):                                       # settings are read for every ranking
            assert runtime.settings().embeddings == "auto"
        _ranked_ids(_ctx())
        _ranked_ids(_ctx(teamfight=False))
    assert sum("Invalid RAG_EMBEDDINGS='bananas'" in record.message for record in caplog.records) == 1


def test_embeddings_api_key_never_reaches_the_status_or_the_log(rag_env, monkeypatch, caplog):
    key = "sk-SECRET-EMBED-KEY-123456"
    monkeypatch.setenv("RAG_EMBEDDINGS", "openai")
    monkeypatch.setenv("RAG_EMBED_API_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("RAG_EMBED_API_KEY", key)
    monkeypatch.setenv("RAG_EMBED_API_MODEL", "embed-x")

    class Unauthorized(Exception):
        status_code = 401

    class EchoingProvider:
        def create(self, model, input):
            raise Unauthorized(f"Incorrect API key provided: {key}")

    echoing = embedders.OpenAIEmbedder("http://127.0.0.1:9/v1", key, "embed-x",
                                       client=SimpleNamespace(embeddings=EchoingProvider()))
    monkeypatch.setattr(runtime, "get_embedder", lambda cfg=None: (echoing, ""))
    with caplog.at_level("DEBUG"):
        stats = runtime.sync("api")
        status = rag.status()
    assert stats["error"] == status["error"] == "api_unreachable: Unauthorized (HTTP 401)"
    assert key not in json.dumps(status) and key not in caplog.text

    # Whatever else raises around the API (a library that puts the key into its message): masked
    class Leaky(embedders.HashEmbedder):
        id = "openai:embed-x"

        def embed_documents(self, texts):
            raise RuntimeError(f"connect to http://user:{key}@127.0.0.1:9 failed")

    monkeypatch.setattr(runtime, "get_embedder", lambda cfg=None: (Leaky(), ""))
    caplog.clear()
    with caplog.at_level("DEBUG"):
        stats = runtime.sync("leaky")
        status = rag.status()
    assert status["error"] == "RuntimeError: connect to http://user:***@127.0.0.1:9 failed" == stats["error"]
    assert key not in caplog.text and "***" in caplog.text
    assert runtime.redact(f"x {key} y") == "x *** y"


# ---------------------------------------------------------------------------
# The runtime cache heals itself; another vector space falls back to the committed file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("embedder", ["hash", "none"])
def test_unusable_cache_file_is_removed_by_the_next_sync(rag_env, monkeypatch, embedder, caplog):
    """Also when every text is in the committed file, and also on a server without a model."""
    runtime.build_static(embedders.HashEmbedder())               # the committed file holds everything
    rag_env.cache.parent.mkdir(parents=True, exist_ok=True)
    rag_env.cache.write_bytes(b"garbage")
    knowledge_base._get_knowledge_store("en")
    status = rag.status()
    assert status["mode"] == "hybrid" and rag_env.cache.name in status["error"]   # reported while it is there
    if embedder == "none":
        monkeypatch.setattr(runtime, "get_embedder", lambda cfg=None: (None, "model_missing"))
    with caplog.at_level("WARNING"):
        stats = runtime.sync("heal")
    assert stats == {"not_run": "model_missing"} if embedder == "none" else stats["embedded"] == 0
    assert not rag_env.cache.exists()
    status = rag.status()
    assert status["mode"] == "hybrid" and status["error"] is None
    assert sum("Removing unusable vector cache file" in record.message for record in caplog.records) == 1
    # The committed file is never touched, however broken
    rag_env.static.write_bytes(b"garbage")
    runtime.sync("again")
    assert rag_env.static.read_bytes() == b"garbage" and rag_env.static.name in rag.status()["error"]


def test_embedder_that_does_not_reproduce_the_committed_vectors_adds_nothing(rag_env, monkeypatch):
    """Other pooling / prefix / CPU kernels than the committed file's: its vectors are not mixed in."""
    from backend.app.ai.learning.base import write_json_atomic

    runtime.build_static(embedders.HashEmbedder())               # the committed file
    write_json_atomic(rag_env.learned / "mining_patterns.json", {"meta": {}, "chunks": [
        {"chunk_id": "learned:pro_pattern:position_3:en", "content": "Pro offlaners join most early fights.",
         "lang": "en", "category": "pro_pattern", "tags": ["role:offlane", "phase:mid_game"], "priority": 2,
         "source": "match_mining", "patch_version": ""}]})
    knowledge_base.reset_knowledge_stores()

    class Shifted(embedders.HashEmbedder):                       # same id, another vector space
        embedded = 0

        def _embed(self, text):
            type(self).embedded += 1
            return super()._embed("shifted " + text)

    shifted = Shifted()
    monkeypatch.setattr(runtime, "get_embedder", lambda cfg=None: (shifted, ""))
    stats = runtime.sync("mismatch")
    assert stats["error"].startswith("embedder_mismatch: cosine ") and "rag doctor" in stats["error"]
    assert Shifted.embedded == 25 and not rag_env.cache.exists()  # the comparison sample, nothing else
    status = rag.status()
    assert status["mode"] == "hybrid" and status["error"] == stats["error"]   # reports keep the committed vectors
    # The embedder the file was built with: the learned text is embedded, once the check has passed
    good = embedders.HashEmbedder()
    monkeypatch.setattr(runtime, "get_embedder", lambda cfg=None: (good, ""))
    stats = runtime.sync("match")
    assert stats["embedded"] == 1 and rag.status()["error"] is None and good._reproduces_committed is True


def _default_model_file(rag_env, store):
    """A committed file of the DEFAULT model (hash vectors of its dimension stand in for the real ones)."""
    np = pytest.importorskip("numpy")
    spec = embedders.MODEL_REGISTRY[embedders.DEFAULT_MODEL]
    fake = embedders.HashEmbedder(spec.dim)
    wanted = [(vectors.DOC, c.index_text) for c in store.chunks() if c.index_text]
    wanted += [(vectors.QUERY, text) for text in situations.situation_queries()]
    keys = [vectors.vector_key(spec.model_id, kind, text) for kind, text in wanted]
    path = rag_env.root / "static" / f"{spec.model_id}.npz"
    vectors.write_vector_file(path, spec.model_id, keys, np.asarray(fake.embed_documents([t for _, t in wanted])),
                              params=spec.params)
    return path


def test_decision_ranking_falls_back_to_the_committed_default_model(rag_env, monkeypatch):
    """An embeddings API that has embedded nothing (yet): reports keep the dense term of the committed file."""
    store = knowledge_base._get_knowledge_store("en")
    path = _default_model_file(rag_env, store)
    monkeypatch.setenv("RAG_EMBEDDINGS", "auto")
    monkeypatch.setenv("RAG_EMBED_API_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("RAG_EMBED_API_KEY", "sk-embedding-key-0123456789")
    monkeypatch.setenv("RAG_EMBED_API_MODEL", "embed-x")
    assert embedders.active_model_id(runtime.settings()) == "openai:embed-x"
    assert [index.model_id for index in runtime.decision_indexes()] == ["openai:embed-x", embedders.DEFAULT_MODEL]
    assert len(runtime.vector_index()) == 0                      # nothing in the API's space
    assert _ranked_ids(_ctx())[1] == "hybrid"
    status = rag.status()
    assert (status["mode"], status["embedder"], status["error"]) == ("hybrid", embedders.DEFAULT_MODEL, None)
    assert status["vectors"] == len(vectors.read_vector_file(path, embedders.DEFAULT_MODEL)[0])
    # A typo in RAG_MODEL: reported, and reports still use the committed vectors
    for name in ("RAG_EMBED_API_BASE_URL", "RAG_EMBED_API_KEY", "RAG_EMBED_API_MODEL"):
        monkeypatch.delenv(name)
    monkeypatch.setenv("RAG_MODEL", "bge-typo-v9")
    status = rag.status()
    assert (status["mode"], status["embedder"], status["error"]) == ("hybrid", embedders.DEFAULT_MODEL, "unknown_model")
    # Once the configured space has usable vectors of its own, it is the one that is read
    monkeypatch.delenv("RAG_MODEL")
    monkeypatch.setenv("RAG_EMBEDDINGS", "hash")
    assert rag.status()["embedder"] == embedders.DEFAULT_MODEL   # hash space still empty
    runtime.sync("hash")
    assert rag.status()["embedder"] == "hash-256" and _ranked_ids(_ctx())[1] == "hybrid"
    # Off means off
    monkeypatch.setenv("RAG_EMBEDDINGS", "off")
    assert runtime.decision_indexes() == [] and _ranked_ids(_ctx())[1] == "lexical"


def test_sync_failure_is_reported_not_raised(rag_env, monkeypatch):
    class Broken(embedders.HashEmbedder):
        def embed_documents(self, texts):
            raise embedders.EmbedderError("api_unreachable", "connection refused")

    monkeypatch.setattr(runtime, "get_embedder", lambda cfg=None: (Broken(), ""))
    stats = runtime.sync("broken")
    assert "api_unreachable" in stats["error"]
    status = rag.status()
    assert status["mode"] == "lexical" and "api_unreachable" in status["error"] and status["building"] is False
    assert knowledge_base.get_knowledge_for_context(_ctx(), lang="en").refs


# ---------------------------------------------------------------------------
# Package contract
# ---------------------------------------------------------------------------

def test_package_import_needs_only_the_standard_library():
    code = (
        "import sys\n"
        "for name in ('numpy', 'onnxruntime', 'tokenizers', 'openai'):\n"
        "    sys.modules[name] = None\n"            # importing any of them would raise
        "import backend.app.ai.rag as rag\n"
        "from backend.app.ai.rag import embedders, evalkit, freetext, runtime, situations, types, vectors\n"
        "status = rag.status()\n"
        "assert status['mode'] == 'lexical', status\n"
        "rag.refresh('x'); rag.shutdown()\n"
        "print(rag.top_k(), rag.agent_search_limit())\n"
    )
    env = {**os.environ, "RAG_EMBEDDINGS": "auto", "DEEPSEEK_API_KEY": "", "AI_COACH_AUTO_LEARN": "0",
           "GAMEDATA_AUTO_UPDATE": "0"}
    env.pop("RAG_TOP_K", None)
    env.pop("RAG_AGENT_SEARCH_LIMIT", None)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env, cwd=str(runtime._project_dirs()[0]), timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["0", "3"]


def test_settings_are_read_from_the_environment(rag_env, monkeypatch):
    monkeypatch.setenv("RAG_TOP_K", "9")
    monkeypatch.setenv("RAG_AGENT_SEARCH_LIMIT", "5")
    assert (rag.top_k(), rag.agent_search_limit()) == (9, 5)
    monkeypatch.setenv("RAG_TOP_K", "many")
    assert rag.top_k() == 0
    monkeypatch.setenv("RAG_INDEX_DIR", "relative/dir")
    cfg = runtime.settings()
    assert cfg.index_dir == runtime._project_dirs()[0] / "relative" / "dir"
    assert cfg.static_dir == rag_env.root / "static"


def test_top_k_setting_caps_the_block(rag_env, monkeypatch):
    full = knowledge_base.get_knowledge_for_context(_ctx(), lang="en")
    monkeypatch.setenv("RAG_TOP_K", "4")
    capped = knowledge_base.get_knowledge_for_context(_ctx(), lang="en")
    assert len(full.refs) > 4 and len(capped.refs) == 4
    assert _entry_ids(capped) == _entry_ids(full)[:4]


def test_concurrent_retrieval_while_vectors_and_stores_change(rag_env):
    np = pytest.importorskip("numpy")
    runtime.sync("test")
    keys, matrix, _ = vectors.read_vector_file(rag_env.cache, "hash-256")
    matrix = np.asarray(matrix)
    errors: list[BaseException] = []
    modes: set[str] = set()
    stop = threading.Event()

    def reader(lang):
        try:
            while not stop.is_set():
                text = knowledge_base.get_knowledge_for_context(_ctx(), lang=lang)
                assert text.refs and all(f"[{ref.ref}] " in text for ref in text.refs)
                modes.add(text.retrieval["mode"])
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=reader, args=(lang,)) for lang in ("zh", "en", "zh", "en")]
    for thread in threads:
        thread.start()
    for round_ in range(12):
        if round_ % 3 == 0:
            knowledge_base.reset_knowledge_stores()
        elif round_ % 3 == 1:                                    # "another process" rewrites the cache
            half = slice(0, len(keys) // 2) if round_ % 2 else slice(None)
            vectors.write_vector_file(rag_env.cache, "hash-256", keys[half], matrix[half])
        else:
            vectors.remove_file(rag_env.cache)       # (waits for a reader on Windows, like any process there)
        time.sleep(0.01)
    vectors.write_vector_file(rag_env.cache, "hash-256", keys, matrix)
    stop.set()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors
    assert modes <= {"hybrid", "lexical"} and modes
    assert knowledge_base.get_knowledge_for_context(_ctx(), lang="en").retrieval["mode"] == "hybrid"
