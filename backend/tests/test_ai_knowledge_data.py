"""Tests for the knowledge data layer: data_manager, game_data, knowledge_base accessors.

No network: game data is written to a temp dir or OpenDota calls are mocked.
"""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import json
from types import SimpleNamespace

import pytest

from backend.app.ai import data_manager, game_data, knowledge_base
from backend.app.ai.knowledge_store import normalize_tag
from backend.app.ai.models import DeathEvent, DecisionContext, DecisionPoint, HeroState
from backend.app.config import EDITORIAL_DIR


@pytest.fixture
def kb_env(tmp_path, monkeypatch):
    """Empty gamedata + learned dirs; caches invalidated before and after."""
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
         position=None, levels=None, brief=""):
    dp = DecisionPoint(
        time=600, point_type=point_type, player_slot=0, hero_name=hero,
        severity="major", brief=brief,
        related_death=DeathEvent(time=600, player_slot=0, hero_name=hero, killer_hero=killer)
        if killer else None,
    )
    ps = HeroState(hero_name=hero, player_slot=0, team="radiant", items=list(items),
                   ability_levels=dict(levels or {}))
    enemy_states = [HeroState(hero_name=e, player_slot=128 + i, team="dire")
                    for i, e in enumerate(enemies)]
    ctx = DecisionContext(decision_point=dp, player_state=ps, enemy_states=enemy_states,
                          game_phase=phase)
    if position:
        ctx.match_context["player_position"] = position
    return ctx


def _write_gamedata(directory, patch="7.99", cull_cd=(42, 41, 40)):
    heroes = {
        "Axe": {"id": 2, "internal_name": "axe", "localized_name": "Axe",
                "primary_attr": "str", "attack_type": "melee", "roles": ["Initiator"]},
    }
    abilities = {
        "Axe": {
            "Berserker's Call": {"key": "axe_berserkers_call", "cooldown": [17, 15, 13, 11],
                                 "mana_cost": [80, 90, 100, 110]},
            "Culling Blade": {"key": "axe_culling_blade", "cooldown": list(cull_cd),
                              "mana_cost": [60, 120, 180], "damage": [250, 350, 450],
                              "damage_type": "pure"},
        }
    }
    items = {"Black King Bar": {"key": "black_king_bar", "cost": 4050, "cooldown": 95,
                                "is_active": True}}
    for name, payload in (("heroes.json", heroes), ("abilities.json", abilities),
                          ("items.json", items),
                          ("meta.json", {"patch_version": patch, "last_fetched": "2026-01-01T00:00:00Z"})):
        (directory / name).write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# K1: editorial merged without fetched game data
# ---------------------------------------------------------------------------

def test_editorial_merged_without_gamedata(kb_env):
    heroes = knowledge_base.get_hero_data()
    editorial = json.loads((EDITORIAL_DIR / "hero_knowledge.json").read_text(encoding="utf-8"))
    assert len(heroes) >= len(editorial) >= 120
    assert heroes["Tinker"]["laning_tips"] == editorial["Tinker"]["laning_tips"]
    qop = knowledge_base.get_hero_info("queenofpain", lang="en")
    assert qop["power_spikes"] == editorial["Queen of Pain"]["power_spikes"]
    assert knowledge_base.get_hero_info("npc_dota_hero_queenofpain")["role"] == qop["role"]

    abilities = game_data.get_ability_table()
    assert len(abilities) >= 120
    assert any(a.get("coaching_tip") for a in abilities["Queen of Pain"].values())
    assert game_data.get_ability_data("Tinker", "Laser")["coaching_tip"]

    items = game_data.get_item_table()
    assert items["Daedalus"]["key"] == "greater_crit"
    assert items["Daedalus"]["coaching_tip"]
    assert game_data.get_item_data("item_black_king_bar")["category"] == "defense"
    assert game_data.get_item_data("bfury")["name"] == "Battle Fury"


def test_get_matchup_info_en_uses_editorial(kb_env):
    tips = json.loads((EDITORIAL_DIR / "matchup_tips.json").read_text(encoding="utf-8"))
    tip = tips["Drow Ranger:Viper"]["tip"]
    assert knowledge_base.get_matchup_info("Drow Ranger", "Viper", lang="en") == tip
    assert knowledge_base.get_matchup_info("drow_ranger", "npc_dota_hero_viper", lang="zh") == tip
    reverse = knowledge_base.get_matchup_info("Viper", "Drow Ranger", lang="en")
    assert reverse.startswith("(From Drow Ranger's perspective)") and tip in reverse
    reverse_zh = knowledge_base.get_matchup_info("Viper", "Drow Ranger", lang="zh")
    assert reverse_zh.startswith("（从Drow Ranger视角") and tip in reverse_zh
    # Renamed hero ("Outworld Destroyer" in editorial, "Outworld Devourer" in the catalog)
    od = tips["Outworld Destroyer:Silencer"]["tip"]
    assert knowledge_base.get_matchup_info("Outworld Devourer", "Silencer", lang="en") == od
    # No editorial tip: localized generic text, never a crash
    generic = knowledge_base.get_matchup_info("Totally Unknown", "Another", lang="en")
    assert "Totally Unknown" in generic


def test_unknown_hero_info_is_localized(kb_env):
    assert "暂无" in knowledge_base.get_hero_info("Nobody", lang="zh")["laning_tips"]
    assert "No detailed" in knowledge_base.get_hero_info("Nobody", lang="en")["laning_tips"]


def test_hero_info_returns_copies(kb_env):
    info = knowledge_base.get_hero_info("Axe")
    info["power_spikes"].append("MUTATED")
    assert "MUTATED" not in knowledge_base.get_hero_info("Axe")["power_spikes"]


# ---------------------------------------------------------------------------
# K2 / K11: accessors return reloaded data after invalidation
# ---------------------------------------------------------------------------

def test_accessors_return_reloaded_data(kb_env):
    old_heroes = knowledge_base.get_hero_data()
    old_store = knowledge_base._get_knowledge_store("en")
    assert data_manager.get_current_patch() == data_manager.DEFAULT_PATCH

    _write_gamedata(kb_env.gamedata, patch="7.99", cull_cd=(42, 41, 40))
    data_manager.invalidate_all_caches()

    assert data_manager.get_current_patch() == "7.99"
    culling = game_data.get_ability_data("Axe", "Culling Blade")
    assert culling["cooldown"] == [42, 41, 40]
    assert culling["coaching_tip"]  # editorial still merged on top
    assert culling.get("is_ultimate") is True
    assert game_data.ABILITY_DATA is game_data.get_ability_table()  # legacy name stays current
    assert knowledge_base.get_hero_data() is not old_heroes
    assert knowledge_base.HERO_DATA is knowledge_base.get_hero_data()
    new_store = knowledge_base._get_knowledge_store("en")
    assert new_store is not old_store
    assert new_store.current_patch == "7.99"
    # The fallback numeric data still fills fields the fetched data lacks
    assert game_data.get_item_data("Black King Bar")["category"] == "defense"
    assert game_data.get_item_data("Black King Bar")["cooldown"] == 95


def test_fetch_and_save_gamedata_invalidates_caches(kb_env, monkeypatch):
    payloads = {
        "patch": [{"name": "7.98"}, {"name": "7.99"}],
        "heroes": {"2": {"id": 2, "name": "npc_dota_hero_axe", "localized_name": "Axe",
                         "primary_attr": "str", "attack_type": "Melee", "roles": ["Initiator"]}},
        "hero_abilities": {"npc_dota_hero_axe": {"abilities": [
            "axe_berserkers_call", "axe_battle_hunger", "axe_counter_helix", "axe_culling_blade"]}},
        "abilities": {
            "axe_berserkers_call": {"dname": "Berserker's Call", "cd": "17 15 13 11", "behavior": "No Target"},
            "axe_battle_hunger": {"dname": "Battle Hunger", "cd": "20 15 10 5"},
            "axe_counter_helix": {"dname": "Counter Helix", "behavior": "Passive"},
            "axe_culling_blade": {"dname": "Culling Blade", "cd": "77 66 55", "dmg_type": "Pure"},
        },
        "items": {"black_king_bar": {"dname": "Black King Bar", "cost": 4050, "cd": 95,
                                     "abilities": [{"type": "active"}]},
                  "recipe_black_king_bar": {"dname": "Recipe: Black King Bar", "cost": 1450}},
    }
    calls = []

    def fake_api_get(endpoint, timeout=30):
        calls.append(endpoint)
        return payloads[endpoint]

    monkeypatch.setattr(data_manager, "_api_get", fake_api_get)
    knowledge_base._get_knowledge_store("zh")
    generation = data_manager.data_generation()

    result = data_manager.fetch_and_save_gamedata(force=False)
    assert result["status"] == "updated" and result["patch_version"] == "7.99"
    assert (kb_env.gamedata / "meta.json").exists()
    assert data_manager.data_generation() > generation
    assert knowledge_base._stores_state[1] == {}  # RAG stores dropped
    assert data_manager.get_current_patch() == "7.99"
    culling = game_data.get_ability_data("Axe", "Culling Blade")
    assert culling["cooldown"] == [77, 66, 55] and culling["is_ultimate"] is True
    assert game_data.get_ability_data("Axe", "Counter Helix")["type"] == "passive"
    assert game_data.get_item_data("Black King Bar")["is_active"] is True
    assert "Recipe: Black King Bar" not in game_data.get_item_table()

    # Same patch again, not forced: nothing fetched beyond the patch check
    calls.clear()
    assert data_manager.fetch_and_save_gamedata(force=False)["status"] == "up_to_date"
    assert calls == ["patch"]


def test_failed_fetch_writes_nothing(kb_env, monkeypatch):
    def fake_api_get(endpoint, timeout=30):
        if endpoint == "items":
            raise OSError("network down")
        return {"patch": [{"name": "7.99"}]}.get(endpoint, {"x": {}})

    monkeypatch.setattr(data_manager, "_api_get", fake_api_get)
    with pytest.raises(OSError):
        data_manager.fetch_and_save_gamedata(force=True)
    assert list(kb_env.gamedata.iterdir()) == []


# ---------------------------------------------------------------------------
# K10: patch handling
# ---------------------------------------------------------------------------

def test_get_current_patch(kb_env):
    assert data_manager.get_current_patch() == data_manager.DEFAULT_PATCH
    assert data_manager.get_fetched_patch() == ""
    assert data_manager.get_patch_version() == data_manager.DEFAULT_PATCH
    assert data_manager.get_meta()["status"] == "no_data"

    (kb_env.gamedata / "meta.json").write_text(json.dumps({"patch_version": "unknown"}))
    data_manager.invalidate_all_caches()
    assert data_manager.get_current_patch() == data_manager.DEFAULT_PATCH

    (kb_env.gamedata / "meta.json").write_text(
        json.dumps({"patch_version": "7.40", "last_fetched": "2026-01-01T00:00:00Z"}))
    data_manager.invalidate_all_caches()
    assert data_manager.get_current_patch() == "7.40"
    assert game_data.PATCH_VERSION == "7.40"
    meta = data_manager.get_meta()
    assert meta["patch_version"] == "7.40" and meta["age_days"] > 0


def test_default_patch_defined_once():
    """No other knowledge-layer module hardcodes a patch string."""
    import pathlib
    import re

    root = pathlib.Path(data_manager.__file__).parent
    pattern = re.compile(r"[\"']7\.\d\d[a-z]?[\"']")
    offenders = []
    for path in [root / "knowledge_base.py", root / "knowledge_store.py", root / "knowledge_chunks.py",
                 root / "game_data.py", *sorted((root / "learning").glob("*.py"))]:
        if pattern.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    assert offenders == []
    assert len(pattern.findall((root / "data_manager.py").read_text(encoding="utf-8"))) == 1


# ---------------------------------------------------------------------------
# K5: one canonical name normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["Anti-Mage", "antimage", "anti_mage", "npc_dota_hero_antimage", "ANTI MAGE"])
def test_hero_slug_canonical(kb_env, name):
    assert data_manager.hero_slug(name) == "antimage"
    assert normalize_tag(f"hero:{name}") == "hero:antimage"


def test_name_normalisation_variants(kb_env):
    assert data_manager.hero_slug("Nature's Prophet") == "furion"
    assert data_manager.resolve_hero_name("furion") == "Nature's Prophet"
    assert data_manager.hero_slug("Outworld Destroyer") == data_manager.hero_slug("Outworld Devourer")
    assert data_manager.hero_slug("all") == "all"
    for item in ("Battle Fury", "bfury", "item_bfury", "battle_fury"):
        assert data_manager.item_key(item) == "bfury"
        assert normalize_tag(f"item:{item}") == "item:bfury"
    assert data_manager.item_key("Daedalus") == "greater_crit"
    assert normalize_tag("matchup:Anti-Mage:Queen of Pain") == "matchup:antimage:queenofpain"
    assert normalize_tag("role:pos5") == "role:support_5"
    assert normalize_tag("role:hard_support") == "role:support_5"
    assert normalize_tag("phase:midgame") == "phase:mid_game"
    assert normalize_tag("Mechanic:Vision") == "mechanic:vision"
    assert normalize_tag("reference") == "reference"
    assert normalize_tag("hero:") is None and normalize_tag(None) is None


def test_hero_catalog_without_gamedata(kb_env):
    catalog = data_manager.get_hero_catalog()
    assert len(catalog) >= 120
    assert catalog[0] == {"id": 1, "name": "Anti-Mage", "slug": "antimage"}


# ---------------------------------------------------------------------------
# K3: ability / item knowledge actually renders
# ---------------------------------------------------------------------------

def test_format_ability_at_level_without_level(kb_env):
    text = game_data.format_ability_at_level("Axe", "Culling Blade", None, lang="en")
    assert text.startswith("Culling Blade:") and "/" in text
    assert game_data.format_ability_at_level("Axe", "Culling Blade", 2, lang="en").startswith(
        "Culling Blade (Lv2)")
    # Level beyond the table clamps instead of raising
    assert "Lv9" in game_data.format_ability_at_level("Axe", "Culling Blade", 9, lang="zh")
    assert game_data.format_ability_at_level("Axe", "Nope", None) == "Nope"


def test_format_hero_abilities_accepts_keys(kb_env):
    text = game_data.format_hero_abilities("Axe", {"axe_counter_helix": 2, "Berserker's Call": 1}, lang="en")
    assert "Counter Helix (Lv2)" in text and "Berserker's Call (Lv1)" in text
    assert "Not learned" in text


def test_killer_abilities_and_active_items_render(kb_env):
    ctx = _ctx(hero="Drow Ranger", killer="Lion", items=["Black King Bar", "Black King Bar", "Tango"])
    text = knowledge_base.get_ability_item_knowledge(ctx, lang="en")
    assert "Killer Key Abilities (Lion)" in text
    assert "Ultimate: Finger of Death" in text
    assert "Key ability: Earth Spike" in text  # disable beats plain damage
    assert text.count("Black King Bar (Active") == 1

    fight = _ctx(hero="Axe", point_type="fight_engage", items=["Blink Dagger", "Tango"])
    fight_text = knowledge_base.get_ability_item_knowledge(fight, lang="zh")
    assert "可用主动道具" in fight_text and "Blink Dagger（CD 15秒）" in fight_text


# ---------------------------------------------------------------------------
# K8 / K9: one source of truth for chunks; items are chunked
# ---------------------------------------------------------------------------

def test_gamedata_chunks_cover_items_and_all_heroes(kb_env):
    from backend.app.ai.knowledge_chunks import generate_gamedata_chunks, get_all_builtin_chunks

    chunks = generate_gamedata_chunks(
        data_manager.load_abilities(), data_manager.load_items(), data_manager.load_heroes(),
        lang="en", matchup_tips=data_manager.load_matchup_tips(),
    )
    by_id = {c.chunk_id: c for c in chunks}
    assert "item:greater_crit" in by_id and "Daedalus" in by_id["item:greater_crit"].content
    assert "item:greater_crit" in by_id["item:greater_crit"].tags
    counter = by_id["item_counter:orchid:storm_spirit"]
    assert counter.tags == {"item:orchid", "hero:storm_spirit"}
    assert "hero_laning:tinker" in by_id and "hero_abilities:queenofpain" in by_id
    assert by_id["matchup:drow_ranger_vs_viper"].tags == {"matchup:drow_ranger:viper"}
    assert len([c for c in chunks if c.category == "hero_laning"]) >= 120

    builtin = get_all_builtin_chunks("zh")
    assert not any(c.category in ("hero_overview", "hero_laning", "hero_matchup") for c in builtin)
    # Vision / death chunks are split from the canonical principle text
    vision_text = knowledge_base.get_vision_principles("zh")
    death_text = knowledge_base.get_death_analysis_rules("en")
    for chunk in builtin:
        if chunk.chunk_id.startswith("vision:"):
            assert chunk.content in vision_text
    for chunk in get_all_builtin_chunks("en"):
        if chunk.chunk_id.startswith("death_analysis:"):
            assert chunk.content in death_text
