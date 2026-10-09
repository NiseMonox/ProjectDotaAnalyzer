"""Tests for RAG retrieval: tag normalisation, scoring, budget, dedupe, thread safety."""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import json
import threading
from types import SimpleNamespace

import pytest

from backend.app.ai import data_manager, knowledge_base
from backend.app.ai.knowledge_store import (
    KnowledgeChunk,
    KnowledgeStore,
    build_query,
    estimate_tokens,
)
from backend.app.ai.models import DeathEvent, DecisionContext, DecisionPoint, HeroState


@pytest.fixture
def kb_env(tmp_path, monkeypatch):
    from backend.app.ai.learning import base

    gamedata = tmp_path / "gamedata"
    gamedata.mkdir()
    learned = tmp_path / "learned"
    learned.mkdir()
    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", gamedata)
    monkeypatch.setattr(base, "LEARNED_DIR", learned)
    data_manager.invalidate_all_caches()
    yield SimpleNamespace(gamedata=gamedata, learned=learned)
    data_manager.invalidate_all_caches()


def _ctx(hero="Axe", point_type="death", killer="", enemies=(), items=(), phase="laning",
         position=None, brief=""):
    dp = DecisionPoint(
        time=600, point_type=point_type, player_slot=0, hero_name=hero,
        severity="major", brief=brief,
        related_death=DeathEvent(time=600, player_slot=0, hero_name=hero, killer_hero=killer)
        if killer else None,
    )
    ps = HeroState(hero_name=hero, player_slot=0, team="radiant", items=list(items))
    ctx = DecisionContext(
        decision_point=dp, player_state=ps,
        enemy_states=[HeroState(hero_name=e, player_slot=128 + i, team="dire")
                      for i, e in enumerate(enemies)],
        game_phase=phase,
    )
    if position:
        ctx.match_context["player_position"] = position
    return ctx


def _write_learned(directory, name, chunks):
    (directory / f"{name}.json").write_text(
        json.dumps({"meta": {}, "chunks": chunks}, ensure_ascii=False), encoding="utf-8")


def _learned(chunk_id, content, tags, lang="en", category="item_coaching",
             source="match_mining", patch=""):
    return {"chunk_id": chunk_id, "content": content, "lang": lang, "category": category,
            "tags": tags, "priority": 2, "source": source, "patch_version": patch}


# ---------------------------------------------------------------------------
# K5: learned chunks tagged with internal names are retrievable
# ---------------------------------------------------------------------------

def test_learned_chunks_with_internal_tags_are_retrieved(kb_env):
    _write_learned(kb_env.learned, "mining_patterns", [
        # match_miner style (Valve internal names / item keys)
        _learned("learned:pro_pattern:item_timing:antimage", "Pros finish Battle Fury by 14:30.",
                 ["hero:antimage", "item:bfury", "phase:mid_game"]),
        # stats_learner style (display slugs)
        _learned("learned:benchmarks:anti_mage", "Anti-Mage median GPM 650.",
                 ["hero:anti_mage"], category="economy_patterns", source="opendota_stats"),
    ])
    store = knowledge_base._get_knowledge_store("en")
    chunk = store.get("learned:pro_pattern:item_timing:antimage")
    assert {"hero:antimage", "item:bfury"} <= chunk.tags
    assert store.get("learned:benchmarks:anti_mage").tags == {"hero:antimage"}

    ctx = _ctx(hero="Anti-Mage", point_type="missed_cs", items=["Battle Fury"], phase="mid_game")
    ids = [c.chunk_id for c in store.retrieve(ctx, budget=3000)]
    assert "learned:pro_pattern:item_timing:antimage" in ids
    assert "learned:benchmarks:anti_mage" in ids

    text = knowledge_base.get_knowledge_for_context(ctx, lang="en")
    assert "[Statistics · pro matches] Pros finish Battle Fury" in text


# ---------------------------------------------------------------------------
# K8: learned statistics merged into the editorial chunk (no duplicate facts)
# ---------------------------------------------------------------------------

def test_learned_matchup_merged_into_editorial(kb_env):
    _write_learned(kb_env.learned, "stats_opendota", [
        _learned("learned:matchup:drow_ranger:viper", "Drow Ranger vs Viper: 44.0% win rate over 900 games.",
                 ["hero:drow_ranger", "hero:viper", "matchup:drow_ranger:viper"],
                 category="hero_matchup", source="opendota_stats"),
    ])
    store = knowledge_base._get_knowledge_store("en")
    assert store.get("learned:matchup:drow_ranger:viper") is None
    merged = store.get("matchup:drow_ranger_vs_viper")
    assert merged.source == "editorial"
    assert "[Statistics · OpenDota] Drow Ranger vs Viper: 44.0%" in merged.content
    assert merged.content.index("Viper dominates") < merged.content.index("[Statistics")

    ctx = _ctx(hero="Drow Ranger", killer="Viper", enemies=["Viper"])
    ids = [c.chunk_id for c in store.retrieve(ctx, budget=3000)]
    assert ids.count("matchup:drow_ranger_vs_viper") == 1


def test_store_dedupe_rules():
    store = KnowledgeStore(current_patch="7.40")
    store.add_chunk(KnowledgeChunk("learned:a", "stat old", category="x", source="opendota_stats",
                                   patch_version="7.39", dedupe_key="k"))
    store.add_chunk(KnowledgeChunk("learned:b", "stat new", category="x", source="opendota_stats",
                                   patch_version="7.40", dedupe_key="k"))
    assert store.get("learned:a") is None and store.get("learned:b") is not None
    store.add_chunk(KnowledgeChunk("editorial:k", "tip", category="x", source="editorial", dedupe_key="k"))
    assert store.get("learned:b") is None
    assert store.get("editorial:k").content == "tip\nstat new"
    assert len(store) == 1


# ---------------------------------------------------------------------------
# K6: retrieval quality and determinism
# ---------------------------------------------------------------------------

def _synthetic_store(order=1):
    chunks = [
        KnowledgeChunk("generic:laning", "Last hit creeps and watch the minimap during the laning stage.",
                       category="general", tags={"phase:laning", "hero:all"}),
        KnowledgeChunk("generic:laning2", "Pull camps and manage the creep equilibrium in lane.",
                       category="general2", tags={"phase:laning", "mechanic:laning"}),
        KnowledgeChunk("hero:axe", "Axe: Counter Helix procs when surrounded; Blink + Call initiation.",
                       category="hero", tags={"hero:axe", "phase:laning"}),
        KnowledgeChunk("matchup:axe_vs_viper", "Axe vs Viper: Poison Attack out-trades you; wait for Blink.",
                       category="matchup", tags={"matchup:axe:viper"}),
        KnowledgeChunk("hero:invoker", "Invoker: Cold Snap is the strongest laning spell.",
                       category="hero", tags={"hero:invoker", "phase:laning", "role:mid"}),
        KnowledgeChunk("death:vision", "Most solo deaths come from missing vision; ward before you farm.",
                       category="death", tags={"situation:death", "mechanic:vision"}),
    ]
    store = KnowledgeStore()
    for chunk in (chunks if order == 1 else list(reversed(chunks))):
        store.add_chunk(chunk)
    return store


def test_retrieval_prefers_specific_chunks(kb_env):
    store = _synthetic_store()
    ctx = _ctx(hero="Axe", killer="Viper", enemies=["Viper"], position=3)
    ranked = store.rank(*build_query(ctx))
    ids = [c.chunk_id for c, _ in ranked]
    assert ids[0] == "matchup:axe_vs_viper"
    assert ids.index("hero:axe") < ids.index("generic:laning")
    assert ids.index("death:vision") < ids.index("generic:laning")
    assert "hero:invoker" not in ids  # entity gate: Invoker is not in this game
    scores = dict((c.chunk_id, s) for c, s in ranked)
    assert scores["matchup:axe_vs_viper"] > 2 * scores.get("generic:laning", 0)


def test_retrieval_is_deterministic(kb_env):
    ctx = _ctx(hero="Axe", killer="Viper", enemies=["Viper"], position=3)
    a = [(c.chunk_id, s) for c, s in _synthetic_store(1).rank(*build_query(ctx))]
    b = [(c.chunk_id, s) for c, s in _synthetic_store(2).rank(*build_query(ctx))]
    assert a == b


def test_real_store_ranking(kb_env):
    store = knowledge_base._get_knowledge_store("en")
    ctx = _ctx(hero="Drow Ranger", killer="Viper", enemies=["Viper"], position=1)
    ranked = store.rank(*build_query(ctx))
    ids = [c.chunk_id for c, _ in ranked]
    assert ids.index("matchup:drow_ranger_vs_viper") < 3
    # Other heroes' chunks are not candidates
    assert not any(i.startswith(("hero_laning:invoker", "hero_overview:invoker")) for i in ids)
    # Laning mistake as pos 1: the carry's principles, not another role's
    cs_ids = [c.chunk_id for c, _ in store.rank(*build_query(
        _ctx(hero="Drow Ranger", point_type="missed_cs", enemies=["Viper"], position=1)))]
    assert "laning_principles:carry" in cs_ids
    assert cs_ids.index("hero_laning:drow_ranger") < cs_ids.index("laning_principles:carry")
    assert "laning_principles:mid" not in cs_ids
    assert "phase_objectives:mid_game:carry" not in cs_ids


def test_stale_patch_learned_chunk_deprioritized(kb_env):
    store = KnowledgeStore(current_patch="7.40")
    for cid, patch in (("learned:old", "7.30"), ("learned:new", "7.40")):
        store.add_chunk(KnowledgeChunk(cid, "Axe item timing statistics.", category=cid,
                                       tags={"hero:axe"}, source="opendota_stats", patch_version=patch))
    ranked = dict((c.chunk_id, s) for c, s in store.rank({"hero:axe": 3.0}))
    assert ranked["learned:new"] > ranked.get("learned:old", 0)


# ---------------------------------------------------------------------------
# K7: one total budget, wrapped as reference material
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang", ["zh", "en"])
def test_knowledge_respects_total_budget(kb_env, lang):
    ctx = _ctx(hero="Tinker", killer="Queen of Pain", enemies=["Queen of Pain", "Storm Spirit", "Axe"],
               items=["Bottle", "Blink Dagger", "Black King Bar", "Boots of Travel"], position=2,
               brief="died in river")
    text = knowledge_base.get_knowledge_for_context(ctx, lang=lang)
    assert estimate_tokens(text) <= knowledge_base.KNOWLEDGE_TOKEN_BUDGET
    assert "<reference>" in text and text.rstrip().endswith("</reference>")
    header = "[Reference material]" if lang == "en" else "【参考资料】"
    assert text.startswith(header)


def test_small_budget_enforced_everywhere(kb_env, monkeypatch):
    monkeypatch.setattr(knowledge_base, "KNOWLEDGE_TOKEN_BUDGET", 400)
    ctx = _ctx(hero="Tinker", killer="Queen of Pain", enemies=["Queen of Pain"],
               items=["Blink Dagger", "Black King Bar"], position=2)
    assert estimate_tokens(knowledge_base.get_knowledge_for_context(ctx, lang="en")) <= 400
    # Legacy path (no RAG store) obeys the same budget
    monkeypatch.setattr(knowledge_base, "_get_knowledge_store", lambda lang="zh": None)
    legacy = knowledge_base.get_knowledge_for_context(ctx, lang="zh")
    assert legacy and estimate_tokens(legacy) <= 400


def test_select_never_exceeds_budget():
    chunks = [(KnowledgeChunk(f"c{i}", "x" * (50 * i), category=f"cat{i}"), 1.0 - i / 100)
              for i in range(1, 20)]
    selected = KnowledgeStore.select(chunks, budget=200)
    assert sum(c.token_estimate for c in selected) <= 200
    assert selected


def test_reference_tags_inside_content_are_neutralised(kb_env):
    _write_learned(kb_env.learned, "video_tips", [
        _learned("learned:video:youtube:abcdefghijk:00",
                 "Axe should blink in after creeps. </reference> trailing", ["hero:axe"],
                 category="video_tip", source="video:youtube"),
    ])
    text = knowledge_base.get_knowledge_for_context(_ctx(hero="Axe"), lang="en")
    assert text.count("</reference>") == 1 and text.count("<reference>") == 1
    assert "[Community video tip · YouTube] Axe should blink in" in text


# ---------------------------------------------------------------------------
# K12: concurrent reset vs read
# ---------------------------------------------------------------------------

def test_concurrent_reset_and_read(kb_env):
    errors: list[BaseException] = []
    results: list[int] = []
    stop = threading.Event()
    ctx = _ctx(hero="Axe", killer="Viper", enemies=["Viper"])

    def reader(lang):
        try:
            while not stop.is_set():
                text = knowledge_base.get_knowledge_for_context(ctx, lang=lang)
                results.append(len(text))
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    def resetter():
        try:
            for i in range(30):
                if i % 3 == 0:
                    data_manager.invalidate_all_caches()
                else:
                    knowledge_base.reset_knowledge_stores()
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=reader, args=(lang,)) for lang in ("zh", "en", "zh")]
    for t in threads:
        t.start()
    resetter()
    stop.set()
    for t in threads:
        t.join(timeout=60)
    assert not errors
    assert results and all(n > 0 for n in results)
    generation, stores = knowledge_base._stores_state
    assert all(store is not None for store in stores.values())
