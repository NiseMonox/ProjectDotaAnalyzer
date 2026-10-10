"""Free-text knowledge search (the agent's own questions): scoping, fallbacks, block format."""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import re

import pytest

from backend.app.ai import data_manager, knowledge_base, rag
from backend.app.ai.knowledge_store import estimate_tokens
from backend.app.ai.models import DeathEvent, DecisionContext, DecisionPoint, HeroState
from backend.app.ai.rag import embedders, freetext, runtime, vectors
from backend.app.ai.rag.types import SearchResult


@pytest.fixture
def rag_env(tmp_path, monkeypatch):
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
    yield tmp_path
    assert runtime.wait_idle(30)
    runtime.reset()
    data_manager.invalidate_all_caches()


def _ctx(hero="Axe", killer="Lion", enemies=("Lion", "Viper"), allies=("Drow Ranger",), phase="mid_game", position=3):
    ctx = DecisionContext(
        decision_point=DecisionPoint(
            time=1500, point_type="death", player_slot=0, hero_name=hero, severity="major", brief="",
            related_death=DeathEvent(time=1500, player_slot=0, hero_name=hero, killer_hero=killer)),
        player_state=HeroState(hero_name=hero, player_slot=0, team="radiant", items=["Blink Dagger"]),
        enemy_states=[HeroState(hero_name=e, player_slot=128 + i, team="dire") for i, e in enumerate(enemies)],
        ally_states=[HeroState(hero_name=a, player_slot=1 + i, team="radiant") for i, a in enumerate(allies)],
        game_phase=phase,
    )
    ctx.match_context["player_position"] = position
    return ctx


def _store():
    return knowledge_base._get_knowledge_store("en")


# ---------------------------------------------------------------------------
# Entity recognition
# ---------------------------------------------------------------------------

def test_heroes_and_items_named_in_a_question(rag_env):
    heroes, items = freetext.find_entities(
        "How does Axe initiate on Queen of Pain once she has a Black King Bar? Does blink help vs Nature's Prophet")
    assert heroes == ["queenofpain", "furion", "axe"]            # longest names first, then single words
    assert items == ["black_king_bar", "blink"]
    # Short aliases are English words: "am" is not Anti-Mage, "tp" not a hero, lowercase "io" not Io
    assert freetext.find_entities("am i too far away to tp when io is pinged")[0] == []
    assert freetext.find_entities("Io tethers") == (["wisp"], [])
    assert freetext.find_entities("") == ([], []) and freetext.find_entities("团战怎么站位") == ([], [])


# ---------------------------------------------------------------------------
# Scoping: the hero-mention gate
# ---------------------------------------------------------------------------

def test_search_is_scoped_to_named_heroes_of_this_match(rag_env):
    result = rag.search(_store(), "how do I fight Lion and Invoker", player_hero="Axe",
                        match_heroes=["Axe", "Lion", "Viper"], top_k=6)
    assert isinstance(result, SearchResult) and result.heroes == ["hero:lion"]
    assert result.ignored_heroes == ["invoker"]                  # named, but not in this match
    ids = [chunk.chunk_id for chunk, _ in result.hits]
    assert ids and len(ids) <= 6
    for chunk, _score in result.hits:
        entity = [t for t in chunk.tags if t.startswith(("hero:", "matchup:")) and t != "hero:all"]
        assert all("lion" in t or "axe" in t for t in entity), chunk.chunk_id   # never Invoker, never Viper
    assert any("lion" in chunk_id for chunk_id in ids)
    # Without a restriction the named hero is searched
    free = rag.search(_store(), "how do I fight Invoker", player_hero="Axe", top_k=6)
    assert free.ignored_heroes == [] and any("invoker" in c.chunk_id for c, _ in free.hits)


def test_question_about_heroes_outside_the_match_returns_only_the_note(rag_env):
    """Nothing is left to look for: whatever ranked first would have nothing to do with the question."""
    question = "How do I play against Anti-Mage and Phantom Assassin?"
    match = ["Axe", "Lion", "Viper", "Drow Ranger"]
    result = rag.search(_store(), question, player_hero="Axe", match_heroes=match, position=3, phase="late_game")
    assert result.hits == [] and result.ignored_heroes == ["phantom_assassin", "antimage"]
    assert (result.mode, result.reason) == ("lexical", "heroes_not_in_match")
    block = knowledge_base.search_knowledge(_ctx(), question, lang="en")
    assert block.refs == [] and "<reference>" not in block
    assert block == ("The knowledge base has no entry on this question.\n"
                     "Not in this match, ignored: Phantom Assassin, Anti-Mage")
    assert block.retrieval["ignored_heroes"] == ["phantom_assassin", "antimage"]
    # Something else to look for (an item, a topic, a hero who is in the match): the search runs
    with_item = rag.search(_store(), "Do I need a Black King Bar against Anti-Mage?", player_hero="Axe",
                           match_heroes=match)
    assert with_item.ignored_heroes == ["antimage"] and "item:black_king_bar" in [c.chunk_id for c, _ in with_item.hits]
    assert rag.search(_store(), question, player_hero="Axe", match_heroes=match, topic="teamfight").hits
    assert rag.search(_store(), "Anti-Mage or Lion, who is the bigger threat?", player_hero="Axe",
                      match_heroes=match).hits
    # Without a match restriction nobody is ignored
    assert rag.search(_store(), question, player_hero="Axe").hits


def test_shown_entries_do_not_use_up_the_result(rag_env):
    first = rag.search(_store(), "how do I fight Lion", player_hero="Axe", heroes=["Lion"], top_k=3)
    ids = [chunk.chunk_id for chunk, _ in first.hits]
    assert len(ids) == 3 and first.already_shown == []
    again = rag.search(_store(), "how do I fight Lion", player_hero="Axe", heroes=["Lion"], top_k=3, shown=ids[:2])
    new_ids = [chunk.chunk_id for chunk, _ in again.hits]
    assert len(new_ids) == 3 and not set(ids[:2]) & set(new_ids)       # three NEW entries, not one
    assert again.already_shown == ids[:2] and new_ids[0] == ids[2]
    # The block names the repeated ones by their handle and numbers the new ones
    block = knowledge_base.search_knowledge(_ctx(), "how do I fight Lion", lang="en", heroes=["Lion"], top_k=3,
                                            first_ref=8, shown={ids[0]: "K2", ids[1]: "K5"})
    assert [ref.ref for ref in block.refs] == ["K8", "K9", "K10"]
    assert not {ref.chunk_id for ref in block.refs} & set(ids[:2])
    assert sorted(block.retrieval["already_shown"]) == ["K2", "K5"]           # in the order they ranked
    assert block.splitlines()[-1] in ("Already shown above: [K2] [K5]", "Already shown above: [K5] [K2]")


def test_explicit_arguments_scope_and_cap(rag_env):
    result = rag.search(_store(), "when to use it", player_hero="Axe", heroes=["Lion", "No Such Hero"],
                        items=["Blink Dagger", "nothing"], top_k=50)
    assert result.heroes == ["hero:lion"] and result.items == ["item:blink"]
    assert 0 < len(result.hits) <= freetext.MAX_TOP_K == 6
    assert "item:blink" in [chunk.chunk_id for chunk, _ in result.hits]
    exclude = {chunk.chunk_id for chunk, _ in result.hits[:2]}
    again = rag.search(_store(), "when to use it", player_hero="Axe", heroes=["Lion"], items=["Blink Dagger"],
                       top_k=6, exclude=exclude)
    assert not exclude & {chunk.chunk_id for chunk, _ in again.hits}
    one = rag.search(_store(), "x" * 5000, player_hero="Axe", top_k=0)
    assert len(one.hits) <= 4                                    # a bad top_k falls back to the default


def test_topic_selects_principles(rag_env):
    result = rag.search(_store(), "where should supports ward before a fight", player_hero="Lion",
                        topic="vision", position=5, phase="mid_game", top_k=4)
    ids = [chunk.chunk_id for chunk, _ in result.hits]
    assert any(i.startswith("vision:") for i in ids)
    assert rag.search(_store(), "anything", topic="no-such-topic").reason != "error"


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def test_dense_search_with_an_embedder_and_lexical_without(rag_env, monkeypatch):
    lexical = rag.search(_store(), "how to trade hits in lane against Viper", player_hero="Axe")
    assert lexical.mode == "lexical" and lexical.reason == "no_vectors"   # embedder there, nothing embedded yet
    runtime.sync("test")
    hybrid = rag.search(_store(), "how to trade hits in lane against Viper", player_hero="Axe")
    assert hybrid.mode == "hybrid" and hybrid.reason == "" and hybrid.hits
    assert any("viper" in chunk.chunk_id for chunk, _ in hybrid.hits)

    # No model in this process: lexical, with the reason
    monkeypatch.setattr(runtime, "get_embedder", lambda cfg=None: (None, "model_missing"))
    fallback = rag.search(_store(), "how to trade hits in lane against Viper", player_hero="Axe")
    assert (fallback.mode, fallback.reason) == ("lexical", "model_missing") and fallback.hits
    # ... except that a topic with a situation query can still use the prebuilt vectors
    topical = rag.search(_store(), "positioning in fights", player_hero="Axe", topic="teamfight",
                         position=3, phase="mid_game")
    assert topical.mode == "hybrid" and topical.hits


# Questions with an obvious answer entry, asked for one decision (Wraith King,
# position 1, mid game). With the store's own formula (0.7 tags, 0.3 BM25,
# the tags being the coached hero, role and phase) the entry was in the top 4
# for none of the first seven, and six of eight questions of a Chinese report
# returned the same four Wraith King entries.
_MATCH = ["Wraith King", "Lion", "Zeus", "Tidehunter", "Invoker", "Viper", "Axe", "Drow Ranger", "Hoodwink", "Rubick"]
_QUESTIONS = [
    ("how to avoid smoke ganks as a carry", "coaching:smoke_gank"),
    ("when should I buy Black King Bar against Zeus and Lion", "item:black_king_bar"),
    ("where to place observer wards in the mid game when behind", "vision:midlate_wards"),
    ("teamfight positioning against Tidehunter Ravage", "coaching:teamfight_positioning"),
    ("when to use buyback in the late game", "coaching:buyback_strategy"),
    ("how to trade hits in lane", "coaching:lane_trading"),
    ("how to deal with split push", "coaching:split_push"),
    ("should I fight at the rune before the horn", "early_game:rune_contest_framework"),
    ("Roshan timing and Aegis usage", "coaching:roshan_timing"),
    ("what does Reincarnation cost and how to play around its cooldown", "hero_abilities:skeleton_king"),
    ("how to defend high ground when behind", "coaching:highground_defense"),
    ("how does Lion kill me", "hero_abilities:lion"),
]


def _learned_statistics(directory):
    """The learned entries that used to come first whatever was asked."""
    import json

    patch = data_manager.get_current_patch()
    chunks = []
    for lang in ("en", "zh"):
        for hero, slug in (("Wraith King", "skeleton_king"), ("Lion", "lion"), ("Zeus", "zuus"),
                           ("Tidehunter", "tidehunter")):
            chunks.append({"chunk_id": f"learned:benchmarks:{slug}", "lang": lang, "category": "economy_patterns",
                           "content": f"{hero} performance benchmarks (all ranks): median GPM 520, top 10% 700.",
                           "tags": [f"hero:{slug}"], "priority": 2, "source": "opendota_stats",
                           "patch_version": patch})
            chunks.append({"chunk_id": f"learned:item_timing:{slug}:blink", "lang": lang, "category": "item_coaching",
                           "content": f"{hero}: games where Blink Dagger was completed before 15 min had a "
                                      "55.0% win rate (300 games).",
                           "tags": [f"hero:{slug}", "item:blink"], "priority": 2, "source": "opendota_stats",
                           "patch_version": patch, "provenance": {"games": 300}})
        chunks.append({"chunk_id": "learned:distilled:07:kill_participation_by_position", "lang": lang,
                       "category": "role_principles", "priority": 2, "source": "llm_distilled",
                       "content": "Position 1 averages 58% kill participation in pro matches, position 2 70%.",
                       "tags": ["role:carry", "role:mid", "phase:mid_game"], "patch_version": patch,
                       "provenance": {"sufficient": True}})
    (directory / "stats_opendota.json").write_text(json.dumps({"meta": {"quick": False}, "chunks": chunks}),
                                                   encoding="utf-8")
    knowledge_base.reset_knowledge_stores()


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("learned", [False, True])
def test_keyword_search_answers_the_question_that_was_asked(rag_env, monkeypatch, lang, learned):
    """Without an embedding model (the default deployment) the question's own words decide."""
    monkeypatch.setenv("RAG_EMBEDDINGS", "off")
    if learned:
        _learned_statistics(rag_env / "learned")
    store = knowledge_base._get_knowledge_store(lang)      # a Chinese report is searched in English too
    assert bool(store.get("learned:benchmarks:skeleton_king")) == learned
    results = {}
    for question, wanted in _QUESTIONS:
        result = rag.search(store, question, player_hero="Wraith King", position=1, phase="mid_game",
                            match_heroes=_MATCH)
        ids = [chunk.chunk_id for chunk, _ in result.hits]
        assert (result.mode, result.reason) == ("lexical", "off")
        assert wanted in ids[:4], (question, ids)
        results[question] = tuple(ids)
    assert len(set(results.values())) == len(_QUESTIONS)       # no two questions get the same entries
    # Farm statistics are no answer to any of them, least of all those of an enemy
    assert not [ids for ids in results.values() if any(i.startswith("learned:benchmarks:") for i in ids)]
    assert not [ids for ids in results.values()
                if {"learned:item_timing:lion:blink", "learned:item_timing:zuus:blink"} & set(ids)]


def test_keyword_search_falls_back_to_the_decision_when_the_question_names_nothing(rag_env, monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDINGS", "off")
    result = rag.search(_store(), "what should he have done differently", player_hero="Wraith King",
                        position=1, phase="mid_game", match_heroes=_MATCH)
    ids = [chunk.chunk_id for chunk, _ in result.hits]
    assert {"hero_abilities:skeleton_king", "hero_overview:skeleton_king"} <= set(ids[:4])
    # A question about farming or item builds does get the statistics, also of a hero it names
    _learned_statistics(rag_env / "learned")
    farm = rag.search(_store(), "how fast should Wraith King farm compared to other carries",
                      player_hero="Wraith King", position=1, phase="mid_game", match_heroes=_MATCH, top_k=6)
    assert "learned:benchmarks:skeleton_king" in [chunk.chunk_id for chunk, _ in farm.hits]
    builds = rag.search(_store(), "what does Lion build", player_hero="Wraith King", topic="items",
                        match_heroes=_MATCH, top_k=6)
    assert "learned:item_timing:lion:blink" in [chunk.chunk_id for chunk, _ in builds.hits]


def test_question_topic_words_map_onto_tags_the_store_has(rag_env):
    from backend.app.ai.knowledge_chunks import get_all_builtin_chunks
    from backend.app.ai.knowledge_store import normalize_tags
    from backend.app.ai.rag import situations

    chunk_tags = set()
    for chunk in get_all_builtin_chunks("en"):
        chunk_tags |= normalize_tags(chunk.tags)
    keywords = [word for words, _tags in situations.QUESTION_TOPICS for word in words]
    assert len(keywords) == len(set(keywords)) and all(word == word.lower() for word in keywords)
    for words, tags in situations.QUESTION_TOPICS:
        assert set(tags) <= chunk_tags, (words, sorted(set(tags) - chunk_tags))
    assert situations.question_topics("How do I avoid SMOKE ganks?") == [
        "mechanic:smoke", "mechanic:gank", "situation:gank", "mechanic:map_awareness"]
    assert situations.question_topics("split pushing and high ground") == ["mechanic:split_push", "mechanic:highground"]
    assert situations.question_topics("position 1 timing") == [] == situations.question_topics("团战怎么站位")


def test_text_rankings_leave_out_chunks_written_for_another_role(rag_env, monkeypatch):
    """Words and embeddings match topics, not who is asking: "Phase objectives (Late game, Position 5)" for a mid."""
    runtime.sync("test")
    store = _store()
    question = "phase objectives late game position hard support buyback gold ready vision last line of defense"
    monkeypatch.setattr(type(store), "rank", lambda self, tags, text="", dense=None: [])   # text rankings only

    def scoped(chunk, namespace):
        return {tag for tag in chunk.tags if tag.startswith(namespace)}

    unscoped = rag.search(store, question, player_hero="Axe", top_k=6)
    assert unscoped.mode == "hybrid"
    assert "phase_objectives:late_game:support_5" in [c.chunk_id for c, _ in unscoped.hits]
    for mode in ("hash", "off"):                    # with the dense ranking, and keywords alone
        monkeypatch.setenv("RAG_EMBEDDINGS", mode)
        mid = rag.search(store, question, player_hero="Axe", position=2, phase="mid_game", top_k=6)
        assert mid.mode == ("hybrid" if mode == "hash" else "lexical") and len(mid.hits) == 6
        for chunk, _score in mid.hits:
            assert not scoped(chunk, "role:") or "role:mid" in chunk.tags, chunk.chunk_id
        # The decision's phase does not filter: the question asks about the late game
        assert "phase_objectives:late_game:mid" in [c.chunk_id for c, _ in mid.hits]


def test_english_only_model_is_not_used_for_a_chinese_question(rag_env, monkeypatch):
    runtime.sync("test")

    class EnglishOnly(embedders.HashEmbedder):
        languages = frozenset({"en"})
        calls = 0

        def embed_queries(self, texts):
            type(self).calls += 1
            return super().embed_queries(texts)

    monkeypatch.setattr(runtime, "get_embedder", lambda cfg=None: (EnglishOnly(), ""))
    chinese = rag.search(_store(), "团战应该怎么站位", player_hero="Axe")
    assert (chinese.mode, chinese.reason) == ("lexical", "non_english_query") and EnglishOnly.calls == 0
    english = rag.search(_store(), "where to stand in a teamfight", player_hero="Axe")
    assert english.mode == "hybrid" and EnglishOnly.calls == 1


def test_failing_embedder_falls_back_to_lexical(rag_env, monkeypatch, caplog):
    runtime.sync("test")

    class Broken(embedders.HashEmbedder):
        def embed_queries(self, texts):
            raise embedders.EmbedderError("api_unreachable", "timeout")

    monkeypatch.setattr(runtime, "get_embedder", lambda cfg=None: (Broken(), ""))
    with caplog.at_level("WARNING"):
        result = rag.search(_store(), "how do I fight Lion", player_hero="Axe")
    assert (result.mode, result.reason) == ("lexical", "api_unreachable") and result.hits
    # ... and a crash anywhere inside never reaches the caller
    monkeypatch.setattr(freetext, "search", lambda *a, **k: 1 / 0)
    assert rag.search(_store(), "x") == SearchResult(reason="error")


# ---------------------------------------------------------------------------
# The block the agent receives
# ---------------------------------------------------------------------------

def test_search_block_numbering_and_repeats(rag_env):
    ctx = _ctx()
    block = knowledge_base.get_knowledge_for_context(ctx, lang="en")
    shown = {ref.chunk_id: ref.ref for ref in block.refs}
    first = len(block.refs) + 1

    result = knowledge_base.search_knowledge(ctx, "How do I play against Lion and Invoker?", lang="en",
                                             top_k=6, first_ref=first, shown=shown)
    assert isinstance(result, str) and isinstance(result, knowledge_base.KnowledgeText)
    assert result.startswith("[Knowledge search result]")
    assert result.count("<reference>") == result.count("</reference>") == 1
    assert len(result) <= knowledge_base.SEARCH_MAX_CHARS
    handles = re.findall(r"(?m)^\[(K\d+)\] ", result)
    assert handles == [ref.ref for ref in result.refs] == [f"K{first + i}" for i in range(len(handles))]
    assert not {ref.chunk_id for ref in result.refs} & set(shown)      # nothing is sent twice
    assert result.retrieval["already_shown"] and "Already shown above: [K" in result
    assert result.retrieval["ignored_heroes"] == ["invoker"] and "Not in this match, ignored: Invoker" in result
    assert all(ref.source_type in ("editorial", "builtin") for ref in result.refs)

    chinese = knowledge_base.search_knowledge(ctx, "How do I play against Lion?", lang="zh", top_k=2)
    assert chinese.startswith("【知识库检索结果】") and [ref.ref for ref in chinese.refs] == ["K1", "K2"][:len(chinese.refs)]
    assert estimate_tokens(chinese) < 1500


def test_search_block_without_hits_or_store(rag_env, monkeypatch):
    ctx = _ctx()
    everything = {chunk.chunk_id: "K1" for chunk in _store().chunks()}
    empty = knowledge_base.search_knowledge(ctx, "How do I play against Lion?", lang="en", shown=everything)
    assert empty.refs == [] and empty.startswith("The knowledge base has no entry on this question.")
    assert "<reference>" not in empty
    monkeypatch.setattr(knowledge_base, "_get_knowledge_store", lambda lang="zh": None)
    assert knowledge_base.search_knowledge(ctx, "anything", lang="zh") == "知识库中没有找到与该问题相关的条目。"
    monkeypatch.setattr(knowledge_base, "_get_knowledge_store", lambda lang="zh": 1 / 0)
    assert knowledge_base.search_knowledge(ctx, "anything", lang="en").retrieval["reason"] == "error"


def test_search_block_is_cut_at_an_entry_boundary_within_2400_characters(rag_env):
    """A tool message of the agent: capped on its own, never in the middle of an entry."""
    from backend.app.ai.knowledge_store import KnowledgeChunk

    store = _store()
    for i in range(5):
        body = " ".join(f"Zebra formation rule {i}.{n}: hold the high ground before the fight starts."
                        for n in range(12))                                   # about 880 characters each
        store.add_chunk(KnowledgeChunk(f"team:zebra_{i}", f"## Zebra formation {i}\n{body}", category=f"zebra_{i}",
                                       tags={"hero:axe"}, source="json"))
    store.finalize()
    hits = rag.search(store, "Axe zebra formation rule", player_hero="Axe", top_k=6).hits
    zebra = [chunk for chunk, _ in hits if chunk.chunk_id.startswith("team:zebra_")]
    assert len(zebra) == 5 and sum(len(chunk.content) for chunk in zebra) > 4000
    block = knowledge_base.search_knowledge(_ctx(), "Axe zebra formation rule", lang="en", top_k=6)
    assert knowledge_base.SEARCH_MAX_CHARS == 2400 and 1500 < len(block) <= 2400
    assert 1 <= len(block.refs) < len(hits)                                   # the tail did not fit
    assert block.count("<reference>") == block.count("</reference>") == 1
    entries = block.split("<reference>\n", 1)[1].split("\n</reference>", 1)[0].split("\n\n")
    assert len(entries) == len(block.refs)
    for entry, ref in zip(entries, block.refs):                               # every entry is whole
        handle, content = entry.split("\n", 1)
        assert handle.startswith(f"[{ref.ref}] ") and content == store.get(ref.chunk_id).content

