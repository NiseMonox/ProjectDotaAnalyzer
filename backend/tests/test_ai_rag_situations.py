"""Situation queries: the closed set of English query strings built from structured fields."""
import pytest

from backend.app.ai import data_manager
from backend.app.ai.models import DeathEvent, DecisionContext, DecisionPoint, HeroState
from backend.app.ai.rag import situations
from backend.app.ai.rag.situations import describe, situation_queries


def _ctx(point_type="death", phase="late_game", position=1, teamfight=False, subtype="",
         hero="Viper", killer="Axe", brief="Killed by Axe in a teamfight"):
    death = DeathEvent(time=1700, player_slot=0, hero_name=hero, killer_hero=killer,
                       is_teamfight=teamfight) if point_type == "death" else None
    ctx = DecisionContext(
        decision_point=DecisionPoint(time=1700, point_type=point_type, player_slot=0, hero_name=hero,
                                     severity="major", brief=brief, related_death=death, subtype=subtype),
        player_state=HeroState(hero_name=hero, player_slot=0, team="radiant", items=["Black King Bar"]),
        enemy_states=[HeroState(hero_name="Axe", player_slot=128, team="dire")],
        game_phase=phase,
    )
    if position:
        ctx.match_context["player_position"] = position
    return ctx


def test_the_set_is_closed_and_has_216_strings():
    queries = situation_queries()
    assert len(queries) == 216 == len(set(queries)) and queries == sorted(queries)
    assert len({*situations.ROLES.values(), situations.DEFAULT_ROLE}) == 6
    assert len({*situations.PHASES.values(), situations.DEFAULT_PHASE}) == 5
    assert len(set(situations.TOPICS.values())) == 7                  # 6 x 5 x 7 = 210 ...
    rune_fight = [q for q in queries if q.endswith(situations.RUNE_FIGHT_TOPIC)]
    assert len(rune_fight) == 6 and all(", pregame: " in q for q in rune_fight)   # ... + 6 for the rune fight
    # Every context maps into the set
    for point_type in ("death", "overextend", "bad_trade", "missed_cs", "fight_engage", "good_play"):
        for phase in ("pregame", "laning", "mid_game", "late_game", ""):
            for position in (1, 2, 3, 4, 5, None):
                for teamfight in (False, True):
                    for subtype in ("", "smoke", "multi_kill"):
                        query = describe(_ctx(point_type, phase, position, teamfight, subtype))
                        assert query is not None and query.text in queries


@pytest.mark.parametrize("kwargs, kind, text", [
    (dict(teamfight=True), "death_tf",
     "position 1 carry, late game: teamfight positioning, target priority, when to join or disengage, "
     "and whether the death was a good trade"),
    (dict(teamfight=False, position=5, phase="mid_game"), "death_pick",
     "position 5 hard support, mid game: how to avoid getting picked off: vision, map awareness, "
     "missing enemies, positioning, escape abilities and defensive items"),
    (dict(point_type="bad_trade", position=3, phase="laning"), "bad_trade",
     "position 3 offlane, laning phase: lane trading, harassing and HP / regeneration management in lane"),
    (dict(point_type="missed_cs", position=2, phase="laning"), "missed_cs",
     "position 2 mid, laning phase: last hitting, creep equilibrium, farming efficiency and CS benchmarks"),
    (dict(point_type="overextend", position=4, phase="laning"), "overextend",
     "position 4 support, laning phase: staying in lane on low HP: when to retreat, use regeneration "
     "and avoid a dive"),
    (dict(point_type="fight_engage", position=3, phase="mid_game"), "fight_engage",
     "position 3 offlane, mid game: teamfight execution: initiation, positioning, ability and item usage"),
    (dict(point_type="good_play", subtype="multi_kill", phase="laning"), "good_play",
     "position 1 carry, laning phase: teamfight execution: initiation, positioning, ability and item usage"),
    (dict(point_type="good_play", subtype="smoke", position=None, phase=""), "good_play_smoke",
     "core, game: smoke ganks, coordinated pick-offs and converting them into objectives"),
])
def test_describe(kwargs, kind, text):
    query = describe(_ctx(**kwargs))
    assert (query.kind, query.text) == (kind, text)


RUNE_FIGHT_TEXT = ("the bounty rune contest and level 1 fights: counting enemy heroes before entering the river, "
                   "level 1 fight strength and when to give up the rune")


@pytest.mark.parametrize("kwargs", [
    dict(teamfight=True), dict(teamfight=False), dict(point_type="good_play", subtype="multi_kill"),
    dict(point_type="good_play", subtype="smoke"), dict(point_type="overextend"), dict(point_type="bad_trade"),
])
def test_before_the_horn_every_decision_asks_about_the_rune_fight(kwargs):
    query = describe(_ctx(phase="pregame", position=5, **kwargs))
    assert (query.kind, query.text) == ("rune_fight", "position 5 hard support, pregame: " + RUNE_FIGHT_TEXT)
    # The topic belongs to its phase: asked for any other one it still names the pregame
    assert situations.situation_text(None, "late_game", "rune_fight") == "core, pregame: " + RUNE_FIGHT_TEXT
    assert describe(_ctx(phase="laning", position=5, **kwargs)).kind != "rune_fight"
    assert describe(_ctx(point_type="ward_placed", phase="pregame")) is None


def test_query_is_entity_free_and_language_independent():
    english = describe(_ctx(teamfight=True, brief="Killed by Axe in a teamfight"))
    chinese = describe(_ctx(teamfight=True, brief="团战中被Axe击杀"))
    assert english == chinese                                 # the brief is never read
    other_heroes = describe(_ctx(teamfight=True, hero="Lion", killer="Zeus"))
    assert other_heroes == english                            # nor are the heroes or the items
    for text in situation_queries():
        assert text.isascii()
        for hero in ("Axe", "Viper", "Lion", "Zeus", "Black King Bar"):
            assert hero.lower() not in text.lower().split()


def test_unknown_point_type_and_odd_fields_have_no_query():
    assert describe(_ctx(point_type="ward_placed")) is None
    assert situations.situation_text(1, "laning", "no_such_topic") == ""
    odd = _ctx()
    odd.match_context["player_position"] = "carry"            # not a number
    assert describe(odd).text.startswith("core, late game: ")
    odd.game_phase = "overtime"
    assert describe(odd).text.startswith("core, game: ")


def test_search_topics_use_tags_the_corpus_has(tmp_path, monkeypatch):
    from backend.app.ai.knowledge_chunks import get_all_builtin_chunks
    from backend.app.ai.knowledge_store import normalize_tags

    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", tmp_path)
    data_manager.invalidate_all_caches()
    chunk_tags = set()
    for chunk in get_all_builtin_chunks("en"):
        chunk_tags |= normalize_tags(chunk.tags)
    chunk_tags.add("mechanic:item_usage")                     # carried by the item chunks
    for topic, (tags, kind) in situations.SEARCH_TOPICS.items():
        assert set(tags) <= chunk_tags, (topic, sorted(set(tags) - chunk_tags))
        assert kind == "" or kind in situations.TOPICS
    data_manager.invalidate_all_caches()
