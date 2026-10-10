"""The game's rule values have one home: backend/app/game_constants.py.

The parser, the analyzer, the mock, the vision model, the data layer's
offline values and the principle texts all read it, and its patch stamp is
tied to the version the rest of the backend is built for: supporting a new
major patch fails here until the rules were reviewed.
"""
from __future__ import annotations

import re
from pathlib import Path

from backend.app import analyzer, game_constants as rules, mock_data, parser
from backend.app.ai import context_builder, data_manager, knowledge_base, phases, vision
from backend.app.ai.knowledge_chunks import get_all_builtin_chunks
from test_ai_pipeline_helpers import make_match

APP = Path(__file__).resolve().parents[1] / "app"


def test_rules_patch_is_the_version_the_backend_is_built_for():
    """Bumping DEFAULT_PATCH_LABEL to a new major version means: review game_constants.py, then bump RULES_PATCH."""
    assert re.fullmatch(r"\d+\.\d+[a-z]?", rules.RULES_PATCH)
    assert data_manager.patch_major(rules.RULES_PATCH) == data_manager.patch_major(data_manager.DEFAULT_PATCH_LABEL)
    assert data_manager.patch_major(rules.RULES_PATCH) == data_manager.DEFAULT_PATCH


def test_tables_are_well_formed():
    xp = rules.XP_FOR_LEVEL
    assert len(xp) == rules.MAX_HERO_LEVEL == 30 and xp[0] == 0
    assert all(a < b for a, b in zip(xp, xp[1:]))
    respawn = rules.RESPAWN_SECONDS
    assert len(respawn) == 25 and respawn[0] == 12 and all(a <= b for a, b in zip(respawn, respawn[1:]))
    assert list(rules.TALENT_LEVELS) == sorted(set(rules.TALENT_LEVELS)) and len(rules.TALENT_LEVELS) == 8
    old = rules.SKILL_POINT_LEVELS_BEFORE_TALENT_POINTS
    assert list(old) == sorted(set(old)) and not {17, 19, 21, 22, 23, 24} & set(old) and old[-1] == rules.MAX_HERO_LEVEL
    assert rules.HERO_ABILITY_LEVELS == 15 and rules.LAST_ULTIMATE_LEVEL_AT <= rules.LAST_SKILL_POINT_LEVEL
    assert rules.ROSHAN_RESPAWN_MIN_S < rules.ROSHAN_RESPAWN_MAX_S
    assert rules.TOWER_T1_NIGHT_VISION < rules.TOWER_NIGHT_VISION < rules.TOWER_DAY_VISION
    assert all(isinstance(v, int) and v > 0 for v in rules.ITEM_RULE_DEFAULTS.values())
    for since, name, before in rules.ITEM_RULE_HISTORY:
        assert since <= tuple(int(n) for n in data_manager.patch_major(rules.RULES_PATCH).split("."))
        assert before != rules.ITEM_RULE_DEFAULTS[name]


def test_every_consumer_reads_the_shared_values():
    # XP: the parser's skill-time estimate, the mock and the AI layer's hero level
    assert parser._XP_FOR_LEVEL is rules.XP_FOR_LEVEL
    for level, need in enumerate(rules.XP_FOR_LEVEL[1:], start=2):
        assert phases.level_from_xp(need) == level and phases.level_from_xp(need - 1) == level - 1
    assert phases.level_from_xp(10 ** 9) == rules.MAX_HERO_LEVEL
    # skill points
    assert parser._TALENT_LEVELS is rules.TALENT_LEVELS
    assert parser._POINT_LEVELS is rules.SKILL_POINT_LEVELS_BEFORE_TALENT_POINTS
    assert parser.talents_have_own_points("", rules.TALENT_POINTS_RELEASED)
    assert not parser.talents_have_own_points("", rules.TALENT_POINTS_RELEASED - 1)
    assert not parser.talents_have_own_points("7.39e") and parser.talents_have_own_points("7.40")
    # respawn, day and night
    assert [vision.respawn_seconds(level) for level in range(1, 26)] == list(rules.RESPAWN_SECONDS)
    assert vision.respawn_seconds(30) == rules.RESPAWN_SECONDS[-1]
    assert not vision._is_night(rules.DAY_NIGHT_PERIOD_S - 1) and vision._is_night(rules.DAY_NIGHT_PERIOD_S)
    # ward lifetimes: parser heuristic, mock, analyzer window count
    assert parser._WARD_LIFETIME_S is rules.WARD_LIFETIME_S is mock_data._WARD_LIFETIME_S
    assert (analyzer.OBSERVER_LIFETIME_S, analyzer.SENTRY_LIFETIME_S) == (
        rules.OBSERVER_WARD_DURATION_S, rules.SENTRY_WARD_DURATION_S)
    assert parser._dewarded_by_lifetime("sentry", 0, rules.SENTRY_WARD_DURATION_S - 6)
    assert not parser._dewarded_by_lifetime("sentry", 0, rules.SENTRY_WARD_DURATION_S - 5)
    # vision model and data layer
    assert vision.DEFAULT_RULES == vision.VisionRules(
        rules.OBSERVER_WARD_VISION, rules.OBSERVER_WARD_DURATION_S, rules.SENTRY_WARD_TRUE_SIGHT,
        rules.SENTRY_WARD_DURATION_S, rules.SMOKE_DURATION_S, rules.SMOKE_BREAK_RADIUS)
    assert data_manager.DEFAULT_HERO_VISION == (rules.HERO_DAY_VISION, rules.HERO_NIGHT_VISION) == (
        vision.HERO_DAY_VISION, vision.HERO_NIGHT_VISION)
    assert vision.tower_vision("mid_t1", night=True) == rules.TOWER_T1_NIGHT_VISION
    assert vision.tower_vision("mid_t3", night=True) == rules.TOWER_NIGHT_VISION
    assert vision.tower_vision("mid_t3", night=False) == rules.TOWER_DAY_VISION
    assert (vision.ANCIENT_VISION, vision.FOUNTAIN_VISION) == (rules.ANCIENT_VISION, rules.FOUNTAIN_VISION)
    assert {name: built_in for name, _item, _attr, built_in in data_manager._ITEM_RULES} == rules.ITEM_RULE_DEFAULTS


def test_offline_item_rules_are_the_built_in_values(monkeypatch, tmp_path):
    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", tmp_path)
    data_manager.invalidate_all_caches()
    try:
        assert data_manager.game_rules() == rules.ITEM_RULE_DEFAULTS
    finally:
        monkeypatch.undo()
        data_manager.invalidate_all_caches()


def test_an_old_match_gets_the_rule_of_its_time_from_the_history():
    assert rules.ITEM_RULE_HISTORY == (((7, 38), "sentry_ward_true_sight", 1000),)
    assert vision._ERA_RULES == (((7, 38), "sentry_true_sight", 1000),)


def test_roshan_counts_as_dead_for_his_shortest_respawn_time():
    match = make_match()
    match.objectives = [{"time": 1000, "type": "CHAT_MESSAGE_ROSHAN_KILL", "team": 2}]
    alive_again = 1000 + rules.ROSHAN_RESPAWN_MIN_S
    assert context_builder._roshan_killed_recently(match, alive_again)["seconds_ago"] == rules.ROSHAN_RESPAWN_MIN_S
    assert context_builder._roshan_killed_recently(match, alive_again + 1) is None


_ONE_HOME = {
    "XP table": r"\b63900\b",
    "respawn table": r"\b12,\s*15,\s*18,\s*21,\s*24\b",
    "talent levels": r"\b10,\s*15,\s*20,\s*25,\s*27\b",
    "release of 7.40": r"\b1765785600\b",
}


def test_a_rule_table_is_typed_in_one_file_only():
    """A second hand-typed copy is how the XP table came to live in two modules."""
    sources = {path: path.read_text(encoding="utf-8") for path in APP.rglob("*.py")}
    for what, pattern in _ONE_HOME.items():
        homes = sorted(path.relative_to(APP).as_posix() for path, text in sources.items() if re.search(pattern, text))
        assert homes == ["game_constants.py"], (what, homes)


def test_principle_texts_take_vision_and_roshan_numbers_from_the_constants(monkeypatch):
    def texts(lang):
        chunks = {chunk.chunk_id: chunk.content for chunk in get_all_builtin_chunks(lang)}
        return "\n".join(chunks[cid] for cid in ("vision:basic_mechanics", "vision:key_numbers", "coaching:roshan_timing"))

    assert "Hero vision: 1800 range during day, 800 range at night" in texts("en")
    assert "at night tier-1 towers 800, other towers 1100" in texts("en")
    assert "Alternates every 5 minutes" in texts("en") and "Roshan respawns randomly 8-11 minutes" in texts("en")
    assert "白天提供1900距离视野；夜间一塔800距离，其余防御塔1100距离" in texts("zh")
    assert "每5分钟交替一次" in texts("zh") and "Roshan被击杀后8-11分钟内随机重生" in texts("zh")

    changed = dict(knowledge_base._RULE_CONSTANTS, hero_day_vision=1750, hero_night_vision=850, tower_day_vision=1950,
                   tower_t1_night_vision=700, tower_night_vision=1200, day_night_minutes=4,
                   roshan_respawn_min_minutes=7.5, roshan_respawn_max_minutes=12)
    monkeypatch.setattr(knowledge_base, "_RULE_CONSTANTS", changed)
    en, zh = texts("en"), texts("zh")
    for needle in ("Hero vision: 1750 range during day, 850 range at night", "(850/1750)",
                   "Towers: 1950 range vision during day; at night tier-1 towers 700, other towers 1200",
                   "Hero daytime vision: 1750 range", "Hero nighttime vision: 850 range", "Tower daytime vision: 1950 range",
                   "Tower nighttime vision: tier-1 towers 700 range, other towers 1200",
                   "Alternates every 4 minutes", "Day/Night cycle period: every 4 minutes",
                   "Roshan respawns randomly 7.5-12 minutes"):
        assert needle in en, needle
    for needle in ("英雄视野：白天1750距离，夜间850距离", "（850/1750）",
                   "防御塔：白天提供1950距离视野；夜间一塔700距离，其余防御塔1200距离",
                   "英雄白天视野：1750距离", "英雄夜间视野：850距离", "防御塔白天视野：1950距离",
                   "防御塔夜间视野：一塔700距离，其余防御塔1200距离", "每4分钟交替一次", "昼夜交替周期：每4分钟",
                   "Roshan被击杀后7.5-12分钟内随机重生"):
        assert needle in zh, needle
    # the constants the texts use are the ones the code computes with
    monkeypatch.undo()
    assert knowledge_base._RULE_CONSTANTS == {
        "hero_day_vision": rules.HERO_DAY_VISION, "hero_night_vision": rules.HERO_NIGHT_VISION,
        "tower_day_vision": rules.TOWER_DAY_VISION, "tower_t1_night_vision": rules.TOWER_T1_NIGHT_VISION,
        "tower_night_vision": rules.TOWER_NIGHT_VISION, "day_night_minutes": rules.DAY_NIGHT_PERIOD_S / 60,
        "roshan_respawn_min_minutes": rules.ROSHAN_RESPAWN_MIN_S / 60,
        "roshan_respawn_max_minutes": rules.ROSHAN_RESPAWN_MAX_S / 60,
    }
