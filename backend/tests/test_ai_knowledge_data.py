"""Tests for the knowledge data layer: data_manager, game_data, knowledge_base accessors.

No network: game data is written to a temp dir or OpenDota calls are mocked.
"""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import json
import re
import stat
from types import SimpleNamespace

import pytest

from backend.app.ai import data_manager, game_data, knowledge_base
from backend.app.ai.knowledge_store import normalize_tag
from backend.app.ai.models import DeathEvent, DecisionContext, DecisionPoint, HeroState
from backend.app.config import EDITORIAL_DIR


def _offline(url, timeout=30):
    raise OSError(f"offline: {url}")


@pytest.fixture
def kb_env(tmp_path, monkeypatch):
    """Empty gamedata + learned dirs; caches invalidated before and after.

    The network is down unless a test serves something: the dotaconstants
    mirrors, Valve's patch feed and the revision check all fail, so a test
    that only fakes ``_api_get`` exercises the last source (the OpenDota API).
    """
    from backend.app.ai.learning import base

    gamedata = tmp_path / "gamedata"
    gamedata.mkdir()
    learned = tmp_path / "learned"
    learned.mkdir()
    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", gamedata)
    monkeypatch.setattr(data_manager, "_http_get_json", _offline)
    monkeypatch.setattr(data_manager, "MIN_TABLE_SIZES", {})     # the fetch tests serve a hero or two
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


def _stale_layers(tmp_path, monkeypatch):
    """Copies of the committed editorial / fallback data with entries of an
    older patch put back, for tests of "fetched data wins".

    The committed files are regenerated for the current patch
    (backend/scripts/gen_static_catalogs.py), so they no longer disagree with
    a current fetch by themselves. Returns the editorial directory in use.
    """
    import shutil

    editorial, fallback = tmp_path / "editorial", tmp_path / "fallback"
    shutil.copytree(data_manager.EDITORIAL_DIR, editorial)
    shutil.copytree(data_manager.FALLBACK_DIR, fallback)

    def rewrite(path, change):
        data = json.loads(path.read_text(encoding="utf-8"))
        change(data)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def old_heroes(heroes):     # before Troll Warlord became ranged and Bane universal
        heroes["Troll Warlord"]["attack_type"] = "melee"
        heroes["Bane"]["attribute"] = "int"

    def old_item_tips(tips):    # removed from the game, still listed by OpenDota
        tips["eternal_shroud"] = {"heroes": ["Huskar"], "item_name": "Eternal Shroud", "is_active": False,
                                  "description": "+20% magic resistance. No active.",
                                  "coaching_tip": "Buy it against heavy magic damage."}

    def old_items(stats):
        stats["items"]["black_king_bar"].update(cooldown=75, duration="10/9/8/7/6/6")
        stats["items"]["eternal_shroud"] = {"item_name": "Eternal Shroud", "cost": 3700,
                                            "is_active": False, "category": "defense"}

    def old_abilities(stats):
        stats["abilities"]["dragon_knight_dragon_tail"].update(
            cooldown=[12, 11, 10, 9], mana_cost=[100, 100, 100, 100], damage=[25, 75, 125, 175],
            stun_duration=[2.5, 2.75, 3.0, 3.25])

    rewrite(editorial / "hero_knowledge.json", old_heroes)
    rewrite(editorial / "coaching_tips_items.json", old_item_tips)
    rewrite(fallback / "item_stats.json", old_items)
    rewrite(fallback / "ability_stats.json", old_abilities)
    monkeypatch.setattr(data_manager, "EDITORIAL_DIR", editorial)
    monkeypatch.setattr(data_manager, "FALLBACK_DIR", fallback)
    data_manager.invalidate_all_caches()
    return editorial


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
    assert reverse_zh.startswith("（从卓尔游侠视角") and tip in reverse_zh     # the hero in the report language
    # A hero with two spellings ("Outworld Devourer" is what dotaconstants still calls him)
    od = tips["Outworld Destroyer:Silencer"]["tip"]
    assert knowledge_base.get_matchup_info("Outworld Devourer", "Silencer", lang="en") == od
    assert knowledge_base.get_matchup_info("obsidian_destroyer", "Silencer", lang="en") == od
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
    # The fallback still supplies what fetched data cannot say (category) ...
    assert game_data.get_item_data("Black King Bar")["category"] == "defense"
    assert game_data.get_item_data("Black King Bar")["cooldown"] == 95
    # ... but none of its (patch-old) numbers reach an ability that was fetched
    call = game_data.get_ability_data("Axe", "Berserker's Call")
    assert call["cooldown"] == [17, 15, 13, 11] and "duration" not in call and "damage" not in call


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
    assert "可用主动道具" in fight_text and "闪烁匕首（CD 15秒）" in fight_text and "Blink Dagger（" not in fight_text


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


# ---------------------------------------------------------------------------
# Real OpenDota shape: a slice cut from the /constants payloads of patch 7.41
# (only the fields the normaliser reads; attrib lists trimmed)
# ---------------------------------------------------------------------------

_CONSTANTS_SLICE = json.loads(r'''
{
 "heroes": {
  "6": {"id": 6, "name": "npc_dota_hero_drow_ranger", "localized_name": "Drow Ranger", "primary_attr": "agi", "attack_type": "Ranged", "roles": ["Carry", "Disabler"]},
  "47": {"id": 47, "name": "npc_dota_hero_viper", "localized_name": "Viper", "primary_attr": "agi", "attack_type": "Ranged", "roles": ["Carry", "Durable"]},
  "49": {"id": 49, "name": "npc_dota_hero_dragon_knight", "localized_name": "Dragon Knight", "primary_attr": "str", "attack_type": "Melee", "roles": ["Carry", "Pusher"]},
  "82": {"id": 82, "name": "npc_dota_hero_meepo", "localized_name": "Meepo", "primary_attr": "agi", "attack_type": "Melee", "roles": ["Carry", "Escape"]},
  "84": {"id": 84, "name": "npc_dota_hero_ogre_magi", "localized_name": "Ogre Magi", "primary_attr": "str", "attack_type": "Melee", "roles": ["Support", "Nuker"]},
  "106": {"id": 106, "name": "npc_dota_hero_ember_spirit", "localized_name": "Ember Spirit", "primary_attr": "agi", "attack_type": "Melee", "roles": ["Carry", "Escape"]},
  "114": {"id": 114, "name": "npc_dota_hero_monkey_king", "localized_name": "Monkey King", "primary_attr": "agi", "attack_type": "Melee", "roles": ["Carry", "Escape"]},
  "126": {"id": 126, "name": "npc_dota_hero_void_spirit", "localized_name": "Void Spirit", "primary_attr": "all", "attack_type": "Melee", "roles": ["Carry", "Escape"]}
 },
 "hero_abilities": {
  "npc_dota_hero_ogre_magi": {"abilities": ["ogre_magi_fireblast", "ogre_magi_ignite", "ogre_magi_bloodlust", "ogre_magi_unrefined_fireblast", "ogre_magi_smash", "ogre_magi_dumb_luck", "ogre_magi_multicast"]},
  "npc_dota_hero_void_spirit": {"abilities": ["void_spirit_aether_remnant", "void_spirit_dissimilate", "void_spirit_resonant_pulse", "void_spirit_intrinsic_edge", "generic_hidden", "void_spirit_astral_step"]},
  "npc_dota_hero_ember_spirit": {"abilities": ["ember_spirit_searing_chains", "ember_spirit_sleight_of_fist", "ember_spirit_flame_guard", "ember_spirit_activate_fire_remnant", "ember_spirit_immolation", "ember_spirit_fire_remnant"]},
  "npc_dota_hero_meepo": {"abilities": ["meepo_earthbind", "meepo_poof", "meepo_ransack", "meepo_petrify", "meepo_megameepo", "meepo_divided_we_stand", "meepo_geomancy", "meepo_megameepo_fling"]},
  "npc_dota_hero_dragon_knight": {"abilities": ["dragon_knight_breathe_fire", "dragon_knight_dragon_tail", "dragon_knight_wyrms_wrath", "dragon_knight_fireball", "dragon_knight_dragon_blood", "dragon_knight_elder_dragon_form"]},
  "npc_dota_hero_drow_ranger": {"abilities": ["drow_ranger_frost_arrows", "drow_ranger_wave_of_silence", "drow_ranger_multishot", "drow_ranger_glacier", "drow_ranger_trueshot", "drow_ranger_marksmanship"]},
  "npc_dota_hero_viper": {"abilities": ["viper_poison_attack", "viper_nethertoxin", "viper_corrosive_skin", "viper_nose_dive", "viper_predator", "viper_viper_strike"]},
  "npc_dota_hero_monkey_king": {"abilities": ["monkey_king_boundless_strike", "monkey_king_tree_dance", "monkey_king_primal_spring", "monkey_king_jingu_mastery", "monkey_king_mischief", "monkey_king_wukongs_command", "monkey_king_primal_spring_early", ["monkey_king_untransform", "monkey_king_transfiguration"]]}
 },
 "abilities": {
  "ogre_magi_fireblast": {"dname": "Fireblast", "behavior": "Unit Target", "dmg_type": "Magical", "mc": ["70", "85", "100", "115"], "cd": ["11", "10", "9", "8"], "attrib": [{"key": "stun_duration", "header": "STUN DURATION:", "value": "1.2"}]},
  "ogre_magi_ignite": {"dname": "Ignite", "behavior": ["Unit Target", "AOE"], "dmg_type": "Magical", "mc": ["80", "90", "100", "110"], "cd": "17", "attrib": [{"key": "duration", "header": "DURATION:", "value": ["5", "6", "7", "8"]}]},
  "ogre_magi_bloodlust": {"dname": "Bloodlust", "behavior": ["Unit Target", "Autocast"], "mc": ["40", "50", "60", "70"], "cd": ["20", "18", "16", "14"], "attrib": [{"key": "duration", "header": "DURATION:", "value": "30"}]},
  "ogre_magi_unrefined_fireblast": {"dname": "Unrefined Fireblast", "behavior": ["Unit Target", "Hidden"], "dmg_type": "Magical", "mc": "400", "cd": "7", "attrib": [{"key": "stun_duration", "header": "STUN DURATION:", "value": "1.2"}]},
  "ogre_magi_smash": {"dname": "Fire Shield", "behavior": ["Unit Target", "Hidden"], "dmg_type": "Magical", "mc": "50", "cd": "15", "attrib": [{"key": "damage", "header": "FIREBALL DAMAGE:", "value": "160"}, {"key": "duration", "header": "SHIELD DURATION:", "value": "25"}]},
  "ogre_magi_dumb_luck": {"dname": "Dumb Luck", "is_innate": true, "behavior": ["Passive", "Hidden"]},
  "ogre_magi_multicast": {"dname": "Multicast", "behavior": "Passive"},
  "void_spirit_aether_remnant": {"dname": "Aether Remnant", "behavior": "Point Target", "dmg_type": "Magical", "mc": ["75", "80", "85", "90"], "cd": ["17", "15", "13", "11"], "attrib": [{"key": "duration", "header": "REMNANT LIFETIME:", "value": "17"}]},
  "void_spirit_dissimilate": {"dname": "Dissimilate", "behavior": "No Target", "dmg_type": "Magical", "dmg": ["120", "200", "280", "360"], "mc": "120", "cd": ["20", "17", "14", "11"]},
  "void_spirit_resonant_pulse": {"dname": "Resonant Pulse", "behavior": "No Target", "dmg_type": "Magical", "mc": ["115", "120", "125", "130"], "cd": "18", "attrib": [{"key": "damage", "header": "DAMAGE:", "value": ["60", "110", "160", "210"]}]},
  "void_spirit_intrinsic_edge": {"dname": "Intrinsic Edge", "is_innate": true, "behavior": "Passive"},
  "generic_hidden": {"behavior": "Hidden", "attrib": []},
  "void_spirit_astral_step": {"dname": "Astral Step", "behavior": "Point Target", "dmg_type": "Magical", "mc": "90", "cd": "0", "attrib": [{"key": "abilitychargerestoretime", "header": "CHARGE RESTORE TIME:", "value": ["25", "20", "15"]}]},
  "ember_spirit_searing_chains": {"dname": "Searing Chains", "behavior": "No Target", "dmg_type": "Magical", "mc": ["80", "90", "100", "110"], "cd": ["13", "12", "11", "10"], "attrib": [{"key": "duration", "header": "DURATION:", "value": ["1.25", "1.75", "2.25", "2.75"]}]},
  "ember_spirit_sleight_of_fist": {"dname": "Sleight of Fist", "behavior": ["Point Target", "AOE"], "dmg_type": "Physical", "mc": ["60", "65", "70", "75"], "cd": ["13", "11", "9", "7"], "attrib": [{"key": "abilitychargerestoretime", "header": "CHARGE RESTORE TIME:", "value": ["13", "11", "9", "7"]}]},
  "ember_spirit_flame_guard": {"dname": "Flame Guard", "behavior": "No Target", "dmg_type": "Magical", "mc": ["65", "80", "95", "110"], "cd": "32", "attrib": [{"key": "duration", "header": "DURATION:", "value": ["12", "14", "16", "18"]}]},
  "ember_spirit_activate_fire_remnant": {"dname": "Activate Fire Remnant", "behavior": "Point Target", "dmg_type": "Magical", "mc": ["100", "125", "150"], "cd": "0", "attrib": [{"key": "damage", "header": "DAMAGE:", "value": ["100", "200", "300"]}]},
  "ember_spirit_immolation": {"dname": "Immolation", "is_innate": true, "behavior": "Passive", "dmg_type": "Magical", "attrib": [{"key": "damage", "header": "DAMAGE PER SECOND:", "value": "10"}]},
  "ember_spirit_fire_remnant": {"dname": "Fire Remnant", "behavior": "Point Target", "dmg_type": "Magical", "mc": "0", "cd": "0.5", "attrib": [{"key": "abilitychargerestoretime", "header": "CHARGE RESTORE TIME:", "value": "35"}, {"key": "damage", "header": "DAMAGE:", "value": ["100", "200", "300"]}, {"key": "duration", "header": "DURATION:", "value": "45"}]},
  "meepo_earthbind": {"dname": "Earthbind", "behavior": ["AOE", "Point Target"], "mc": ["70", "80", "90", "100"], "cd": ["16", "14", "12", "10"], "attrib": [{"key": "duration", "header": "DURATION:", "value": "2"}]},
  "meepo_poof": {"dname": "Poof", "behavior": "Point Target", "dmg_type": "Magical", "mc": "80", "cd": ["12", "10", "8", "6"]},
  "meepo_ransack": {"dname": "Ransack", "behavior": "Passive", "dmg_type": "Pure"},
  "meepo_petrify": {"dname": "Dig", "behavior": ["No Target", "Hidden"], "mc": "125", "cd": "40", "attrib": [{"key": "duration", "header": "DURATION:", "value": "3"}]},
  "meepo_megameepo": {"dname": "MegaMeepo", "behavior": ["Instant Cast", "No Target"], "mc": "0", "cd": "60", "attrib": [{"key": "duration", "header": "DURATION:", "value": "25"}]},
  "meepo_divided_we_stand": {"dname": "Divided We Stand", "behavior": "Passive", "dmg": "0"},
  "meepo_geomancy": {"dname": "Geomancy", "is_innate": true, "behavior": ["Passive", "Hidden"]},
  "meepo_megameepo_fling": {"dname": "MegaMeepo Fling", "behavior": ["Unit Target", "Hidden"], "dmg_type": "Magical", "mc": "0", "cd": "0.5"},
  "dragon_knight_breathe_fire": {"dname": "Breathe Fire", "behavior": ["Point Target", "Unit Target"], "dmg_type": "Magical", "mc": ["90", "95", "100", "105"], "cd": ["14", "13", "12", "11"], "attrib": [{"key": "damage", "header": "DAMAGE:", "value": ["80", "160", "240", "320"]}, {"key": "duration", "header": "REDUCTION DURATION:", "value": "11"}]},
  "dragon_knight_dragon_tail": {"dname": "Dragon Tail", "behavior": "Unit Target", "dmg_type": "Magical", "mc": ["70", "80", "90", "100"], "cd": ["16", "14", "12", "10"], "attrib": [{"key": "stun_duration", "header": "STUN DURATION:", "value": ["1.8", "2", "2.2", "2.4"]}, {"key": "damage", "header": "DAMAGE:", "value": ["60", "90", "120", "150"]}]},
  "dragon_knight_wyrms_wrath": {"dname": "Wyrm's Wrath", "behavior": "Passive", "dmg_type": "Magical"},
  "dragon_knight_fireball": {"dname": "Fireball", "behavior": ["Point Target", "AOE", "Hidden"], "dmg_type": "Magical", "mc": "80", "cd": "20", "attrib": [{"key": "damage", "header": "DAMAGE PER SECOND:", "value": "85"}, {"key": "duration", "header": "DURATION:", "value": "6"}]},
  "dragon_knight_dragon_blood": {"dname": "Dragon Blood", "is_innate": true, "behavior": "Passive"},
  "dragon_knight_elder_dragon_form": {"dname": "Elder Dragon Form", "behavior": "No Target", "dmg_type": "Magical", "mc": "50", "cd": "100", "attrib": [{"key": "duration", "header": "DURATION:", "value": "60"}]},
  "drow_ranger_frost_arrows": {"dname": "Frost Arrows", "behavior": ["Unit Target", "Autocast", "Attack Modifier"], "dmg_type": "Physical", "dmg": "0", "mc": ["9", "10", "11", "12"], "cd": "0", "attrib": [{"key": "damage", "header": "BONUS DAMAGE:", "value": ["10", "15", "20", "25"]}]},
  "drow_ranger_wave_of_silence": {"dname": "Gust", "behavior": "Point Target", "mc": "70", "cd": ["19", "17", "15", "13"]},
  "drow_ranger_multishot": {"dname": "Multishot", "behavior": ["Point Target", "Channeled"], "dmg_type": "Physical", "mc": ["50", "70", "90", "110"], "cd": ["24", "21", "18", "15"]},
  "drow_ranger_glacier": {"dname": "Glacier", "behavior": "No Target", "dmg_type": "Magical", "mc": "50", "cd": "25"},
  "drow_ranger_trueshot": {"dname": "Precision Aura", "is_innate": true, "behavior": ["Passive", "Hidden"]},
  "drow_ranger_marksmanship": {"dname": "Marksmanship", "behavior": "Passive", "dmg_type": "Physical", "attrib": [{"key": "chance", "header": "CHANCE:", "value": ["30%", "35%", "40%"]}]},
  "viper_poison_attack": {"dname": "Poison Attack", "behavior": ["Unit Target", "Autocast", "Attack Modifier"], "dmg_type": "Magical", "dmg": "0", "mc": "20", "cd": "0", "attrib": [{"key": "duration", "header": "DURATION:", "value": "4"}, {"key": "damage", "header": "DAMAGE PER SECOND:", "value": ["4", "8", "12", "16"]}]},
  "viper_nethertoxin": {"dname": "Nethertoxin", "behavior": ["Point Target", "AOE"], "dmg_type": "Magical", "mc": "70", "cd": "14", "attrib": [{"key": "duration", "header": "DURATION:", "value": ["6.5", "7", "7.5", "8"]}]},
  "viper_corrosive_skin": {"dname": "Corrosive Skin", "behavior": "Passive", "dmg_type": "Magical", "attrib": [{"key": "duration", "header": "DURATION:", "value": "4"}, {"key": "damage", "header": "DAMAGE PER SECOND:", "value": ["10", "15", "20", "25"]}]},
  "viper_nose_dive": {"dname": "Nosedive", "behavior": ["Point Target", "AOE"], "dmg_type": "Magical", "mc": "75", "cd": "20", "attrib": [{"key": "duration", "header": "DURATION:", "value": "0.1"}]},
  "viper_predator": {"dname": "Predator", "is_innate": true, "behavior": "Passive", "dmg_type": "Physical", "attrib": [{"key": "damage", "header": "DAMAGE:", "value": "0.25"}]},
  "viper_viper_strike": {"dname": "Viper Strike", "behavior": "Unit Target", "dmg_type": "Magical", "mc": ["100", "150", "200"], "cd": ["50", "40", "30"], "attrib": [{"key": "duration", "header": "DURATION:", "value": "6"}, {"key": "damage", "header": "DAMAGE PER SECOND:", "value": ["70", "110", "150"]}]},
  "monkey_king_boundless_strike": {"dname": "Boundless Strike", "behavior": "Point Target", "dmg_type": "Physical", "mc": ["85", "90", "95", "100"], "cd": ["24", "21", "18", "15"], "attrib": [{"key": "stun_duration", "header": "STUN DURATION:", "value": ["0.7", "0.9", "1.1", "1.3"]}, {"key": "duration", "header": "DURATION:", "value": "0.2"}]},
  "monkey_king_tree_dance": {"dname": "Tree Dance", "behavior": "Unit Target", "mc": "0", "cd": ["0.9", "0.6", "0.3", "0"]},
  "monkey_king_primal_spring": {"dname": "Primal Spring", "behavior": ["Point Target", "AOE", "Channeled"], "dmg_type": "Magical", "mc": "100", "cd": ["18", "16", "14", "12"]},
  "monkey_king_jingu_mastery": {"dname": "Jingu Mastery", "behavior": "Passive", "dmg_type": "Physical", "cd": "0", "attrib": [{"key": "counter_duration", "header": "COUNTER DURATION:", "value": ["5.5", "7", "8.5", "10"]}]},
  "monkey_king_mischief": {"dname": "Mischief", "is_innate": true, "behavior": "No Target", "mc": "0", "cd": "24.5"},
  "monkey_king_wukongs_command": {"dname": "Wukong's Command", "behavior": ["Point Target", "AOE"], "dmg_type": "Physical", "mc": "100", "cd": ["100", "90", "80"], "attrib": [{"key": "duration", "header": "DURATION:", "value": "14"}]},
  "monkey_king_primal_spring_early": {"dname": "Spring Early", "behavior": ["Hidden", "No Target", "Instant Cast"]},
  "monkey_king_untransform": {"dname": "Revert Form", "behavior": ["No Target", "Hidden"], "mc": "0", "cd": "1"},
  "monkey_king_transfiguration": {"dname": "Changing of the Guard", "behavior": ["Hidden", "Point Target"], "mc": "0", "cd": "3"},
  "enigma_black_hole": {"dname": "Black Hole", "behavior": ["AOE", "Point Target", "Channeled"], "dmg_type": "Pure", "mc": ["300", "400", "500"], "cd": ["180", "170", "160"], "attrib": [{"key": "damage", "header": "DAMAGE PER SECOND:", "value": ["100", "150", "200"]}, {"key": "duration", "header": "DURATION:", "value": "4"}]},
  "batrider_sticky_napalm": {"dname": "Sticky Napalm", "behavior": ["AOE", "Point Target"], "dmg_type": "Magical", "mc": "22", "cd": "3", "attrib": [{"key": "damage", "header": "DAMAGE PER STACK:", "value": ["2.5", "5", "7.5", "10"]}, {"key": "duration", "header": "DURATION:", "value": "6"}]},
  "lion_impale": {"dname": "Earth Spike", "behavior": ["Point Target", "Unit Target"], "dmg_type": "Magical", "mc": ["90", "110", "130", "150"], "cd": ["14", "13", "12", "11"], "attrib": [{"key": "damage", "header": "DAMAGE:", "value": ["105", "170", "235", "300"]}, {"key": "duration", "header": "STUN DURATION:", "value": ["1.3", "1.6", "1.9", "2.2"]}]},
  "axe_culling_blade": {"dname": "Culling Blade", "behavior": "Unit Target", "dmg_type": "Pure", "mc": ["100", "125", "150"], "cd": ["80", "75", "70"], "attrib": [{"key": "damage", "header": "DAMAGE:", "value": ["275", "375", "475"]}]}
 },
 "items": {
  "dagon": {"dname": "Dagon", "cost": 3050, "cd": 27, "mc": 120, "qual": "rare", "behavior": "Unit Target", "components": ["point_booster", "crown", "wizard_hat"], "abilities": [{"type": "active", "title": "Energy Burst", "description": "Emits a powerful burst of magical damage upon a targeted enemy unit. Upgradable.\n\nDamage: 400\n Mana Cost: 120"}]},
  "dagon_2": {"dname": "Dagon", "cost": 4200, "cd": 27, "mc": 120, "qual": "rare", "behavior": "Unit Target", "abilities": [{"type": "active", "title": "Energy Burst"}]},
  "dagon_5": {"dname": "Dagon", "cost": 7650, "cd": 27, "mc": 120, "qual": "rare", "behavior": "Unit Target", "abilities": [{"type": "active", "title": "Energy Burst"}]},
  "diffusal_blade": {"dname": "Diffusal Blade", "cost": 2500, "cd": 15, "mc": 25, "qual": "artifact", "behavior": "Unit Target", "components": ["blade_of_alacrity", "robe"], "abilities": [{"type": "active", "title": "Inhibit", "description": "Targets an enemy, slowing it for 4 seconds."}, {"type": "passive", "title": "Manabreak", "description": "Each attack burns 40 mana from the target, and deals 1 physical damage per burned mana. \n\nIllusions do not burn mana."}]},
  "diffusal_blade_2": {"dname": "Diffusal Blade", "cost": 3850, "cd": 4, "mc": false, "qual": "artifact", "components": ["diffusal_blade", "recipe_diffusal_blade"]},
  "butterfly": {"dname": "Butterfly", "cost": 5450, "cd": false, "mc": false, "qual": "epic", "components": ["eagle", "claymore", "talisman_of_evasion"]},
  "eternal_shroud": {"dname": "Eternal Shroud", "cost": 3900, "cd": false, "mc": false, "qual": "epic", "components": ["cloak", "ogre_axe", "vitality_booster"], "abilities": [{"type": "passive", "title": "Shroud", "description": "Restores mana equal to 25% of incoming enemy spell damage before reductions."}, {"type": "passive", "title": "Eternal Endurance", "description": "For every 300 spell damage that equipped hero receives, it gains a stack of Endurance for 5s up to a maximum of 6 charges. Each charge increases Eternal Shroud's magic resistance by 3.5% ."}]},
  "travel_boots": {"dname": "Boots of Travel", "cost": 2500, "cd": false, "mc": false, "qual": "common", "components": ["boots"], "abilities": [{"type": "upgrade", "title": "Town Portal Scroll"}]},
  "black_king_bar": {"dname": "Black King Bar", "cost": 4050, "cd": 95, "mc": 50, "qual": "epic", "behavior": ["Instant Cast", "No Target"], "components": ["mithril_hammer", "ogre_axe"], "abilities": [{"type": "active", "title": "Avatar", "description": "Applies a basic dispel. Grants 60% magic resistance and immunity to reflected and pure damage. For the duration of the effect, any negative effect from enemy spells has no effect. \n\nDuration: 9s \nDispel Type: Basic Dispel"}]},
  "hand_of_midas": {"dname": "Hand of Midas", "cost": 2200, "cd": false, "mc": false, "qual": "common", "behavior": "Unit Target", "components": ["gloves"], "abilities": [{"type": "active", "title": "Transmute", "description": "Kills a non-hero target for 160 gold. Killing a neutral creep additionally grants a madstone bundle. \n\n Cannot be used on Ancient creeps."}]},
  "recipe_dagon": {"dname": "Dagon Recipe", "cost": 1150, "cd": false, "mc": false, "behavior": false, "components": null}
 }
}
''')


def _fetch_slice(monkeypatch, patch="7.41"):
    """Run the real fetcher on the slice (no network)."""
    payloads = {"patch": [{"name": patch}], **_CONSTANTS_SLICE}
    monkeypatch.setattr(data_manager, "_api_get", lambda endpoint, timeout=30: payloads[endpoint])
    return data_manager.fetch_and_save_gamedata(force=True)


def test_fetch_flags_ultimates_and_innates_on_real_shape(kb_env, monkeypatch):
    _fetch_slice(monkeypatch)
    # No "isult" in the payload: the hero_abilities order decides.
    expected = {"Ogre Magi": "Multicast",          # slot 6 is the hidden innate Dumb Luck
                "Void Spirit": "Astral Step",      # charge based, cooldown 0
                "Ember Spirit": "Fire Remnant",    # a 3-level sub-ability sits before it
                "Meepo": "Divided We Stand",       # passive
                "Dragon Knight": "Elder Dragon Form",
                "Drow Ranger": "Marksmanship",
                "Viper": "Viper Strike",
                "Monkey King": "Wukong's Command"}
    for hero, ultimate in expected.items():
        assert game_data.get_hero_ultimate(hero)[0] == ultimate
        flagged = [n for n, a in game_data.get_hero_abilities(hero).items() if a.get("is_ultimate")]
        assert flagged == [ultimate]
    table = game_data.get_ability_table()
    assert "generic_hidden" not in table["Void Spirit"]
    assert table["Void Spirit"]["Intrinsic Edge"]["is_innate"] is True
    assert "is_innate" not in table["Void Spirit"]["Astral Step"]
    # Key ability: a learnable one, ranked by the attrib-derived stun
    assert game_data.get_signature_ability("Ogre Magi")[0] == "Fireblast"
    assert game_data.get_signature_ability("Dragon Knight")[0] == "Dragon Tail"
    assert game_data.get_signature_ability("Void Spirit")[0] != "Intrinsic Edge"


def _learnable(hero):
    return [name for name, data in game_data.get_hero_abilities(hero).items() if game_data.is_learnable(data)]


def test_only_abilities_that_take_skill_points_are_learnable(kb_env, monkeypatch):
    _fetch_slice(monkeypatch)
    fetched = [hero["localized_name"] for hero in _CONSTANTS_SLICE["heroes"].values()]
    assert all(isinstance(a.get("skillable"), bool)
               for hero in fetched for a in game_data.get_hero_abilities(hero).values())
    # Three basic abilities and the ultimate. Shard / scepter abilities (Glacier,
    # Nosedive, MegaMeepo) and a sub-ability that levels with the ultimate
    # (Activate Fire Remnant) sit in visible slots but take no points.
    assert _learnable("Drow Ranger") == ["Frost Arrows", "Gust", "Multishot", "Marksmanship"]
    assert _learnable("Viper") == ["Poison Attack", "Nethertoxin", "Corrosive Skin", "Viper Strike"]
    assert _learnable("Meepo") == ["Earthbind", "Poof", "Ransack", "Divided We Stand"]
    assert _learnable("Ember Spirit") == ["Searing Chains", "Sleight of Fist", "Flame Guard", "Fire Remnant"]
    assert _learnable("Ogre Magi") == ["Fireblast", "Ignite", "Bloodlust", "Multicast"]
    # A basic ability behind a sub-ability still counts: it has four levels
    assert _learnable("Monkey King") == ["Boundless Strike", "Tree Dance", "Primal Spring", "Jingu Mastery",
                                         "Wukong's Command"]

    text = game_data.format_hero_abilities("Drow Ranger", {"drow_ranger_frost_arrows": 1}, lang="en")
    assert "- Not learned: Gust, Multishot, Marksmanship" in text.splitlines()
    # ... while a shard ability the hero does have is shown like any other
    with_shard = game_data.format_hero_abilities(
        "Drow Ranger", {"drow_ranger_frost_arrows": 1, "drow_ranger_glacier": 1}, lang="en")
    assert "- Glacier (Lv1): CD 25s, Mana 50" in with_shard.splitlines()
    assert "Nosedive" not in game_data.format_hero_abilities("Viper", {}, lang="zh")
    assert "Nosedive" not in game_data.format_hero_abilities("Viper", {"viper_nethertoxin": 1}, lang="zh")
    # The killer's "key ability" is one it skills (was Activate Fire Remnant: it lists damage)
    assert game_data.get_signature_ability("Ember Spirit")[0] == "Searing Chains"

    # Files written before the flag existed are completed on load
    path = kb_env.gamedata / "abilities.json"
    abilities = json.loads(path.read_text(encoding="utf-8"))
    assert abilities["Drow Ranger"]["Glacier"]["skillable"] is False
    for hero_abilities in abilities.values():
        for data in hero_abilities.values():
            del data["skillable"]
    path.write_text(json.dumps(abilities), encoding="utf-8")
    data_manager.invalidate_all_caches()
    assert _learnable("Drow Ranger") == ["Frost Arrows", "Gust", "Multishot", "Marksmanship"]
    assert _learnable("Monkey King")[3] == "Jingu Mastery"


def test_fetched_numbers_win_over_fallback(kb_env, monkeypatch, tmp_path):
    _stale_layers(tmp_path, monkeypatch)
    # Without game data the fallback (here: of an older patch) stands in ...
    assert game_data.get_ability_data("Dragon Knight", "Dragon Tail")["damage"] == [25, 75, 125, 175]
    _fetch_slice(monkeypatch)
    # ... with fetched data the current numbers come from OpenDota's attrib list
    tail = game_data.get_ability_data("Dragon Knight", "Dragon Tail")
    assert tail["damage"] == [60, 90, 120, 150]
    assert tail["stun_duration"] == [1.8, 2, 2.2, 2.4]
    text = game_data.format_ability_at_level("Dragon Knight", "Dragon Tail", 4, lang="en")
    assert text == "Dragon Tail (Lv4): CD 10s, Mana 100, Magical 150 damage, 2.4s stun"
    # A fetched ability without the number gets nothing from the fallback either
    abilities = json.loads((kb_env.gamedata / "abilities.json").read_text(encoding="utf-8"))
    for field in ("damage", "stun_duration", "attrib"):
        abilities["Dragon Knight"]["Dragon Tail"].pop(field, None)
    (kb_env.gamedata / "abilities.json").write_text(json.dumps(abilities), encoding="utf-8")
    data_manager.invalidate_all_caches()
    tail = game_data.get_ability_data("Dragon Knight", "Dragon Tail")
    assert "damage" not in tail and "stun_duration" not in tail
    assert tail["type"] == "active" and tail["coaching_tip"]


def test_keys_sharing_a_display_name_stay_resolvable(kb_env, monkeypatch):
    payloads = json.loads(json.dumps(_CONSTANTS_SLICE))
    payloads["patch"] = [{"name": "7.41"}]
    payloads["heroes"]["11"] = {"id": 11, "name": "npc_dota_hero_nevermore", "localized_name": "Shadow Fiend",
                                "primary_attr": "agi", "attack_type": "Ranged", "roles": ["Carry"]}
    payloads["hero_abilities"]["npc_dota_hero_nevermore"] = {"abilities": [
        "nevermore_shadowraze1", "nevermore_shadowraze2", "nevermore_shadowraze3", "nevermore_frenzy",
        "nevermore_dark_lord", "nevermore_requiem", "nevermore_necromastery"]}
    for i, cd in ((1, "10"), (2, "10"), (3, "10")):
        payloads["abilities"][f"nevermore_shadowraze{i}"] = {"dname": "Shadowraze", "behavior": "No Target",
                                                             "cd": cd, "mc": "75 80 85 90"}
    payloads["abilities"].update({
        "nevermore_frenzy": {"dname": "Feast of Souls", "behavior": ["No Target", "Instant Cast"],
                             "cd": "30 25 20 15", "mc": "50"},
        "nevermore_dark_lord": {"dname": "Presence of the Dark Lord", "behavior": "Passive"},
        "nevermore_requiem": {"dname": "Requiem of Souls", "behavior": "No Target",
                              "cd": "100 95 90", "mc": "150 175 200"},
        "nevermore_necromastery": {"dname": "Necromastery", "is_innate": True, "behavior": ["Passive", "Hidden"]},
    })
    monkeypatch.setattr(data_manager, "_api_get", lambda endpoint, timeout=30: payloads[endpoint])
    data_manager.fetch_and_save_gamedata(force=True)
    raze = game_data.get_ability_data("Shadow Fiend", "Shadowraze")
    assert data_manager.ability_keys(raze) == [f"nevermore_shadowraze{i}" for i in (1, 2, 3)]
    for i in (1, 2, 3):
        assert game_data.get_ability_data("Shadow Fiend", f"nevermore_shadowraze{i}") is raze
        assert game_data.get_ability_by_key(f"nevermore_shadowraze{i}")[1] == "Shadowraze"
        assert data_manager.resolve_ability_name(f"nevermore_shadowraze{i}") == "Shadowraze"
    # Skill points arrive under the first key only: the ability must count as learned
    text = game_data.format_hero_abilities("Shadow Fiend", {"nevermore_shadowraze1": 3}, lang="en")
    assert "- Shadowraze (Lv3)" in text
    assert "Not learned: Feast of Souls, Presence of the Dark Lord, Requiem of Souls" in text
    assert raze["coaching_tip"]   # the editorial tip (keyed by one of the three) is merged
    # The three keys are one basic ability: the next two slots are basic abilities too
    assert _learnable("Shadow Fiend") == ["Shadowraze", "Feast of Souls", "Presence of the Dark Lord",
                                          "Requiem of Souls"]


def test_attrib_derived_stats_and_zero_placeholders():
    raw = _CONSTANTS_SLICE["abilities"]
    frost = data_manager._normalize_ability(raw["drow_ranger_frost_arrows"], "drow_ranger_frost_arrows")
    # "dmg": "0" is a placeholder, and "BONUS DAMAGE:" (per attack) is not what the ability deals
    assert "damage" not in frost and "damage_per_second" not in frost
    assert frost["type"] == "autocast"
    spike = data_manager._normalize_ability(raw["lion_impale"], "lion_impale")
    assert spike["stun_duration"] == [1.3, 1.6, 1.9, 2.2]  # key "duration", header "STUN DURATION:"
    assert "duration" not in spike and spike["damage"] == [105, 170, 235, 300]
    passive = data_manager._normalize_ability(raw["meepo_divided_we_stand"], "meepo_divided_we_stand")
    assert "damage" not in passive
    # Only a plain total is "damage": per-second numbers get their own field, per-stack ones none
    hole = data_manager._normalize_ability(raw["enigma_black_hole"], "enigma_black_hole")
    assert "damage" not in hole and hole["damage_per_second"] == [100, 150, 200] and hole["duration"] == [4]
    strike = data_manager._normalize_ability(raw["viper_viper_strike"], "viper_viper_strike")
    assert "damage" not in strike and strike["damage_per_second"] == [70, 110, 150]
    napalm = data_manager._normalize_ability(raw["batrider_sticky_napalm"], "batrider_sticky_napalm")
    assert "damage" not in napalm and "damage_per_second" not in napalm and napalm["duration"] == [6]
    for header, field in (("DAMAGE:", "damage"), ("BASE DAMAGE:", "damage"), ("IMPACT DAMAGE:", "damage"),
                          ("DAMAGE (EXORT):\\N", "damage"), ("ROOT DAMAGE PER SECOND:", "damage_per_second"),
                          ("DAMAGE/HEAL PER SECOND:", "damage_per_second"), ("DAMAGE PER INSTANCE:", None),
                          ("ATTACK DAMAGE:", None), ("HEAL/DAMAGE:", None), ("DAMAGE/SHIELD:", None),
                          ("MOVEMENT SPEED AS DAMAGE:", None), ("DAMAGE EACH BOUNCE:", None)):
        stats = data_manager._attrib_stats([{"key": "damage", "header": header, "value": ["50", "100"]}])
        assert stats == ({field: [50, 100]} if field else {}), header


def test_damage_over_time_is_not_printed_as_plain_damage(kb_env, monkeypatch):
    _fetch_slice(monkeypatch)
    assert (game_data.format_ability_at_level("Viper", "Viper Strike", 2, lang="en")
            == "Viper Strike (Lv2): CD 40s, Mana 150, Magical 110 damage/s, Duration 6s")
    assert (game_data.format_ability_at_level("Viper", "Poison Attack", None, lang="zh")
            == "毒性攻击: 魔耗20, 每秒魔法伤害4/8/12/16, 持续4秒")
    assert "damage" not in game_data.format_ability_at_level("Drow Ranger", "Frost Arrows", 1, lang="en")

    # A file written before the headers were read stores the per-second list as "damage"
    path = kb_env.gamedata / "abilities.json"
    abilities = json.loads(path.read_text(encoding="utf-8"))
    strike = abilities["Viper"]["Viper Strike"]
    strike["damage"] = strike.pop("damage_per_second")
    path.write_text(json.dumps(abilities), encoding="utf-8")
    data_manager.invalidate_all_caches()
    strike = game_data.get_ability_data("Viper", "Viper Strike")
    assert "damage" not in strike and strike["damage_per_second"] == [70, 110, 150]

    # For the killer's "key ability", damage over time ranks like damage (above a plain active)
    viper = game_data.get_ability_table()["Viper"]
    assert game_data.get_signature_ability("Viper")[0] == "Poison Attack"
    monkeypatch.delitem(viper["Poison Attack"], "damage_per_second")
    monkeypatch.setitem(viper["Nethertoxin"], "damage_per_second", [30, 40, 50, 60])
    assert game_data.get_signature_ability("Viper")[0] == "Nethertoxin"


def test_format_omits_zero_damage_and_unknown_cooldown(kb_env, monkeypatch):
    _fetch_slice(monkeypatch)
    for lang in ("en", "zh"):
        text = game_data.format_ability_at_level("Meepo", "Divided We Stand", 1, lang=lang)
        assert "0" not in text.replace("Lv1", "")
        items = game_data.format_player_items(
            ["Hand of Midas", "Black King Bar", "Boots of Travel", "Butterfly"], lang=lang)
        assert "Boots of Travel (" not in items and "Butterfly (" not in items  # listed, not "active"
        assert "?" not in items
        assert ("Hand of Midas (Active):" if lang == "en" else "迈达斯之手（主动）:") in items
        assert ("Black King Bar (Active, CD 95s)" if lang == "en" else "黑皇杖（主动, CD 95秒）") in items
        assert ("Other items: Boots of Travel, Butterfly" if lang == "en" else "其他装备: 远行鞋、蝴蝶") in items
    # A stored all-zero list never becomes a "0 damage" fact, a cooldown of 0 never "CD 0s"
    monkeypatch.setitem(game_data.get_ability_table()["Meepo"]["Poof"], "damage", [0, 0, 0, 0])
    assert "damage" not in game_data.format_ability_at_level("Meepo", "Poof", 2, lang="en")
    monkeypatch.setitem(game_data.get_item_table()["Black King Bar"], "cooldown", 0)
    assert "- Black King Bar (Active):" in game_data.format_player_items(["Black King Bar"], lang="en")
    assert "- 黑皇杖（主动）:" in game_data.format_player_items(["Black King Bar"], lang="zh")


def test_innate_and_hidden_abilities_are_not_listed_as_unlearned(kb_env, monkeypatch):
    _fetch_slice(monkeypatch)
    text = game_data.format_hero_abilities("Dragon Knight", {"dragon_knight_dragon_tail": 4}, lang="en")
    line = next(l for l in text.splitlines() if l.startswith("- Not learned"))
    assert line == "- Not learned: Breathe Fire, Wyrm's Wrath, Elder Dragon Form"
    # An innate that does carry a level is still shown as learned
    levelled = game_data.format_hero_abilities("Dragon Knight", {"Dragon Blood": 2}, lang="zh")
    assert "- 龙族血统 (Lv2)" in levelled and "Fireball" not in levelled and "火球" not in levelled
    assert levelled.splitlines()[0] == "### 龙骑士 技能详情"
    assert levelled.splitlines()[-1] == "- 未学习: 火焰气息、神龙摆尾、飞龙之怒、古龙形态"
    unskilled = game_data.format_hero_abilities("Void Spirit", {}, lang="en")
    assert "Intrinsic Edge" not in unskilled and "Astral Step: (requires level 6)" in unskilled
    # Invoker's placeholder slots are neither Hidden nor innate
    assert not game_data.is_learnable({"key": "invoker_empty1", "behavior": "Passive"})
    assert game_data.is_learnable({"key": "invoker_quas", "behavior": "No Target"})


def test_item_name_collisions_keep_the_base_item(kb_env, monkeypatch):
    _fetch_slice(monkeypatch)
    dagon = game_data.get_item_data("Dagon")
    assert (dagon["key"], dagon["cost"]) == ("dagon", 3050)
    assert dagon["upgrade_keys"] == ["dagon_2", "dagon_5"]
    assert dagon["components"] == ["point_booster", "crown", "wizard_hat"]
    diffusal = game_data.get_item_data("Diffusal Blade")
    assert (diffusal["key"], diffusal["cost"], diffusal["cooldown"]) == ("diffusal_blade", 2500, 15)
    # Upgrade keys lead to the same entry / tag as the base item
    assert game_data.get_item_data("item_dagon_5") is dagon
    assert data_manager.item_key("dagon_5") == "dagon" and normalize_tag("item:Dagon") == "item:dagon"
    assert "Dagon Recipe" not in game_data.get_item_table()

    from backend.app.ai.knowledge_chunks import generate_gamedata_chunks
    chunks = {c.chunk_id: c for c in generate_gamedata_chunks(
        {}, data_manager.load_items(), {}, lang="en")}
    assert "item:dagon" in chunks["item:dagon"].tags and "3050 gold" in chunks["item:dagon"].content
    assert "item:dagon_5" not in chunks


def test_fetched_facts_win_over_editorial(kb_env, monkeypatch, tmp_path):
    EDITORIAL_DIR = _stale_layers(tmp_path, monkeypatch)
    editorial = json.loads((EDITORIAL_DIR / "hero_knowledge.json").read_text(encoding="utf-8"))
    # Editorial hero facts go stale (here: those of an older patch); the catalog decides.
    assert editorial["Troll Warlord"]["attack_type"] == "melee"
    assert knowledge_base.get_hero_info("Troll Warlord")["attack_type"] == "ranged"
    assert knowledge_base.get_hero_info("Bane")["attribute"] == "uni"
    assert knowledge_base.get_hero_info("Bane")["laning_tips"] == editorial["Bane"]["laning_tips"]
    # Items without an active are not "active" (also without fetched data)
    for name in ("Eternal Shroud", "Butterfly", "Boots of Travel"):
        assert game_data.get_item_data(name)["is_active"] is False
    assert game_data.get_item_data("Hand of Midas")["is_active"] is True  # editorial, nothing fetched
    bkb = game_data.get_item_data("Black King Bar")
    assert bkb["cooldown"] == 75 and "duration" in bkb                    # fallback, nothing fetched
    editorial_items = json.loads((EDITORIAL_DIR / "coaching_tips_items.json").read_text(encoding="utf-8"))
    fallback_items = json.loads((data_manager.FALLBACK_DIR / "item_stats.json").read_text(encoding="utf-8"))["items"]
    # What an item does: the description generated into the fallback; a hand-written
    # one only where there is none (the stale Eternal Shroud entry)
    assert bkb["description"] == fallback_items["black_king_bar"]["description"]
    assert "description" not in editorial_items["black_king_bar"]
    assert game_data.get_item_data("Eternal Shroud")["description"] == editorial_items["eternal_shroud"]["description"]

    # OpenDota: Boots of Travel only has an "upgrade" ability, Eternal Shroud two passives.
    # Pretend a patch also made Hand of Midas passive: the fetched fact must win.
    payloads = json.loads(json.dumps(_CONSTANTS_SLICE))
    payloads["items"]["hand_of_midas"]["abilities"] = [{"type": "passive", "title": "Transmute"}]
    payloads["patch"] = [{"name": "7.41"}]
    monkeypatch.setattr(data_manager, "_api_get", lambda endpoint, timeout=30: payloads[endpoint])
    data_manager.fetch_and_save_gamedata(force=True)
    for name in ("Boots of Travel", "Eternal Shroud", "Hand of Midas"):
        assert game_data.get_item_data(name)["is_active"] is False
    midas = game_data.get_item_data("Hand of Midas")
    assert midas["coaching_tip"] and midas["category"] == "utility"   # editorial text / category still merged
    # A fetched item takes no number from the (patch-old) fallback
    bkb = game_data.get_item_data("Black King Bar")
    assert bkb["is_active"] is True and bkb["cooldown"] == 95 and "duration" not in bkb
    # What an item does is a per-patch fact too: the fetched ability text (cut at a
    # sentence end) replaces the stand-ins, whose numbers go stale
    assert bkb["description"] == ("Active: Applies a basic dispel. Grants 60% magic resistance and immunity "
                                  "to reflected and pure damage.")
    assert game_data.get_item_data("Eternal Shroud")["description"].startswith("Passive: Restores mana equal to 25%")
    assert game_data.get_item_data("Diffusal Blade")["description"] == (
        "Active: Targets an enemy, slowing it for 4 seconds. Passive: Each attack burns 40 mana from the "
        "target, and deals 1 physical damage per burned mana.")
    # ... which only fill a gap (this slice of the constants says nothing about Butterfly)
    assert game_data.get_item_data("Butterfly")["description"] == fallback_items["butterfly"]["description"]
    assert midas["coaching_tip"] == editorial_items["hand_of_midas"]["coaching_tip"]
    line = "- 黑皇杖（主动, CD 95秒）: Active: Applies a basic dispel. Grants 60% magic resistance"
    assert line in game_data.format_player_items(["Black King Bar"], lang="zh")
    # The description goes with a coaching tip: an active without one is listed by name and cooldown
    monkeypatch.delitem(bkb, "coaching_tip")
    assert game_data.format_player_items(["Black King Bar"], lang="en").splitlines()[1:] == [
        "- Black King Bar (Active, CD 95s)"]

    describe = data_manager._item_description
    long_text = describe({"abilities": [{"type": "active", "description": "Does one thing. " * 30},
                                        {"type": "passive", "description": "x"}]})
    assert len(long_text) <= 160 and long_text.endswith("Does one thing.") and long_text.startswith("Active: Does")
    uncut = describe({"abilities": [{"type": "toggle", "description": "word " * 60}]})
    assert len(uncut) == 160 and uncut.endswith("…")
    assert describe({"abilities": [{"type": "active", "title": "No text"}]}) == ""
    # An item without ability text is described by the stats it lists, as many as fit
    stats = [{"key": "bonus_agility", "display": "+ {value} Agility", "value": "30"},
             {"key": "bonus_evasion", "display": "+ {value}% Evasion", "value": "35"},
             {"key": "hidden_number", "value": "7"},
             {"key": "bonus_all", "display": "+ {value} All Attributes", "value": ["6", "7"]}]
    assert describe({"attrib": stats}) == "+30 Agility, +35% Evasion, +6 / 7 All Attributes."
    assert describe({"attrib": stats, "abilities": [{"type": "passive", "description": "Does a thing."}]}) == (
        "Passive: Does a thing.")
    many = describe({"attrib": [{"key": f"k{n}", "display": "+ {value} Some Long Attribute Name", "value": n}
                                for n in range(20)]})
    assert len(many) <= 160 and many.endswith("Some Long Attribute Name…") and many.startswith("+0 Some")
    # An unresolved template value is not a fact; a percentage is
    assert describe({"abilities": [
        {"type": "active", "description": "Consume %abilityhealthcost% health to gain 170 mana."},
        {"type": "passive", "description": "Grants a 30% chance to deal 225% damage."}]}) == (
        "Passive: Grants a 30% chance to deal 225% damage.")


def test_legacy_gamedata_is_sanitised_and_refetched(kb_env, monkeypatch):
    """Files written before meta.json carried schema_version."""
    legacy_abilities = {"Void Spirit": {
        "Resonant Pulse": {"key": "void_spirit_resonant_pulse", "cooldown": [18], "dname": "Resonant Pulse",
                           "behavior": "No Target", "damage": [0],
                           "attrib": [{"key": "damage", "header": "DAMAGE:", "value": ["60", "110", "160", "210"]}]},
        "generic_hidden": {"key": "generic_hidden", "behavior": "Hidden", "attrib": []},
        "Astral Step": {"key": "void_spirit_astral_step", "cooldown": [0], "mana_cost": [90],
                        "dname": "Astral Step", "behavior": "Point Target", "attrib": []},
    }}
    legacy_items = {
        "Dagon": {"key": "dagon_5", "cost": 7650, "cooldown": 27, "mana_cost": 120, "name": "Dagon",
                  "is_active": False, "description": "Active: Energy Burst. Damage: 800"},
        "Hand of Midas": {"key": "hand_of_midas", "cost": 2200, "cooldown": False, "mana_cost": False,
                          "name": "Hand of Midas", "is_active": False},
        "Black King Bar": {"key": "black_king_bar", "cost": 4050, "cooldown": 95, "name": "Black King Bar",
                           "is_active": False},
        "Black King Bar Recipe": {"key": "recipe_black_king_bar", "cost": 1450, "cooldown": False,
                                  "name": "Black King Bar Recipe"},
    }
    for name, payload in (("abilities.json", legacy_abilities), ("items.json", legacy_items),
                          ("meta.json", {"patch_version": "7.41", "last_fetched": "2026-07-13T06:36:02Z"})):
        (kb_env.gamedata / name).write_text(json.dumps(payload), encoding="utf-8")
    data_manager.invalidate_all_caches()

    assert data_manager.get_schema_version() == 0 and data_manager.get_meta()["outdated_format"] is True
    void = game_data.get_hero_abilities("Void Spirit")
    assert "generic_hidden" not in void
    assert void["Resonant Pulse"]["damage"] == [60, 110, 160, 210] and void["Resonant Pulse"]["type"] == "active"
    assert game_data.get_hero_ultimate("Void Spirit")[0] == "Astral Step"  # editorial order as fallback
    items = game_data.get_item_table()
    assert "Black King Bar Recipe" not in items
    assert "cooldown" not in items["Hand of Midas"]              # false means "none", not a value
    assert items["Black King Bar"]["is_active"] is True          # the old flag was always false
    assert items["Dagon"]["key"] == "dagon" and "cost" not in items["Dagon"]  # was the dagon_5 entry
    assert "Energy Burst" not in items["Dagon"]["description"]   # nor the text of that upgrade level
    text = game_data.format_player_items(["Hand of Midas", "Black King Bar"], lang="en")
    assert "?" not in text and "Hand of Midas (Active):" in text

    # Same patch, but the format is outdated: the updater must refetch ...
    monkeypatch.setattr(data_manager, "_api_get", lambda endpoint, timeout=30: [{"name": "7.41"}])
    assert data_manager.check_patch_version() == ("7.41", True)
    _fetch_slice(monkeypatch)
    # ... and the refreshed files carry the marker, so it happens once
    meta = json.loads((kb_env.gamedata / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == data_manager.GAMEDATA_SCHEMA_VERSION
    assert data_manager.check_patch_version() == ("7.41", False)
    assert data_manager.get_meta()["outdated_format"] is False


def test_get_meta_staleness_follows_config(kb_env, monkeypatch):
    import time as _time
    from backend.app import config

    ten_days_ago = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(_time.time() - 10 * 86400))
    meta_path = kb_env.gamedata / "meta.json"
    meta_path.write_text(json.dumps({"patch_version": "7.41", "last_fetched": ten_days_ago}))
    monkeypatch.setattr(config, "GAMEDATA_STALE_DAYS", 7)
    assert data_manager.get_meta()["is_stale"] is True     # the updater refreshes after 7 days
    monkeypatch.setattr(config, "GAMEDATA_STALE_DAYS", 30)
    assert data_manager.get_meta()["is_stale"] is False
    for bad in ("not a date", 1234567890):
        meta_path.write_text(json.dumps({"patch_version": "7.41", "last_fetched": bad}))
        assert data_manager.get_meta()["is_stale"] is True


def test_game_data_written_by_another_process_is_noticed(kb_env, monkeypatch):
    _write_gamedata(kb_env.gamedata, patch="7.98", cull_cd=(42, 41, 40))
    data_manager.invalidate_all_caches()
    assert data_manager.get_current_patch() == "7.98"
    store = knowledge_base._get_knowledge_store("en")
    generation = data_manager.data_generation()

    # Another process (the update_gamedata CLI) replaces the files: no invalidate call here.
    _write_gamedata(kb_env.gamedata, patch="7.100", cull_cd=(70, 60, 50))
    assert data_manager.get_current_patch() == "7.98"       # not checked yet (throttled)
    monkeypatch.setattr(data_manager, "DISK_CHECK_INTERVAL_S", 0.0)
    monkeypatch.setattr(data_manager, "_next_disk_check", 0.0)
    assert data_manager.get_current_patch() == "7.100"
    assert data_manager.data_generation() > generation
    assert game_data.get_ability_data("Axe", "Culling Blade")["cooldown"] == [70, 60, 50]
    new_store = knowledge_base._get_knowledge_store("en")
    assert new_store is not store and new_store.current_patch == "7.100"


# ---------------------------------------------------------------------------
# Editorial data consistency
# ---------------------------------------------------------------------------

def test_editorial_ability_entries_are_keyed_to_their_ability(kb_env):
    editorial = json.loads((EDITORIAL_DIR / "coaching_tips_abilities.json").read_text(encoding="utf-8"))
    assert editorial["dazzle_nothl_projection"]["ability_name"] == "Nothl Projection"
    assert "Bad Juju" not in json.dumps(editorial["dazzle_nothl_projection"])
    assert editorial["naga_siren_rip_tide"]["ability_name"] == "Rip Tide" and "naga_siren_reel_in" not in editorial
    assert editorial["dragon_knight_dragon_blood"]["ability_name"] == "Dragon Blood"
    assert editorial["beastmaster_summon_razorback"]["ability_name"] == "Summon Razorback"
    assert "tinker_defense_matrix" not in editorial and "beastmaster_call_of_the_wild_boar" not in editorial
    # Without game data: one ultimate per hero, and never an innate / hidden ability
    table = game_data.get_ability_table()
    assert len(table) >= 120
    for hero, abilities in table.items():
        ultimates = [n for n, a in abilities.items() if a.get("is_ultimate")]
        assert len(ultimates) == 1, hero
        assert game_data.is_learnable(abilities[ultimates[0]]) or hero == "Invoker", hero
    assert game_data.get_hero_ultimate("Keeper of the Light")[0] == "Spirit Form"
    # Editorial innate flags keep "Dragon Blood" out of the not-learned list
    text = game_data.format_hero_abilities("Dragon Knight", {"Dragon Tail": 1}, lang="en")
    assert "Dragon Blood" not in text and "Elder Dragon Form" in text


def test_item_descriptions_are_generated_not_hand_written():
    """What an item does, with its numbers, is a per-patch fact: the catalog
    generator writes it into the fallback file from the same constants the
    fetch reads. The editorial file only holds the advice."""
    editorial = json.loads((EDITORIAL_DIR / "coaching_tips_items.json").read_text(encoding="utf-8"))
    fallback = json.loads((data_manager.FALLBACK_DIR / "item_stats.json").read_text(encoding="utf-8"))["items"]
    assert [key for key, info in editorial.items() if "description" in info] == []
    assert [key for key in editorial if not fallback[key].get("description")] == []
    assert fallback["butterfly"]["description"].startswith("+30 Agility, +35% Evasion")


def test_saved_game_data_is_readable_for_other_users(tmp_path):
    path = tmp_path / "gamedata" / "heroes.json"
    data_manager._save_json(path, {"x": 1})
    assert stat.S_IMODE(path.stat().st_mode) == 0o644     # mkstemp alone would leave 0600
    assert json.loads(path.read_text(encoding="utf-8")) == {"x": 1}


def test_skipped_editorial_abilities_are_logged_once(kb_env, monkeypatch, caplog):
    _fetch_slice(monkeypatch)
    editorial = dict(data_manager._load_dict(data_manager.EDITORIAL_DIR / "coaching_tips_abilities.json"))
    editorial["meepo_no_such_ability"] = {"hero": "Meepo", "ability_name": "No Such Ability",
                                          "coaching_tip": "x"}
    real_load = data_manager._load_dict
    monkeypatch.setattr(data_manager, "_load_dict", lambda path, key=None: (
        editorial if path.name == "coaching_tips_abilities.json" else real_load(path, key)))
    data_manager.invalidate_all_caches()
    with caplog.at_level("WARNING", logger=data_manager.logger.name):
        assert "No Such Ability" not in game_data.get_hero_abilities("Meepo")
    warnings = [r.getMessage() for r in caplog.records if "editorial ability tip" in r.getMessage()]
    assert len(warnings) == 1 and "meepo_no_such_ability" in warnings[0]


# ---------------------------------------------------------------------------
# Alias index + display-name helpers (every identifier form the parsers emit)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("identifier, key, name, name_zh", [
    ("Boots of Travel", "travel_boots", "Boots of Travel", "远行鞋"),
    ("travel_boots", "travel_boots", "Boots of Travel", "远行鞋"),
    ("item_travel_boots", "travel_boots", "Boots of Travel", "远行鞋"),
    ("boots_of_travel", "travel_boots", "Boots of Travel", "远行鞋"),       # entity class names
    ("dustof_appearance", "dust", "Dust of Appearance", "显影之尘"),
    ("invisibility_edge", "invis_sword", "Shadow Blade", "影刃"),
    ("empty_bottle", "bottle", "Bottle", "魔瓶"),
    ("bkb", "black_king_bar", "Black King Bar", "黑皇杖"),                  # abbreviations
    ("Midas", "hand_of_midas", "Hand of Midas", "迈达斯之手"),
    ("aghs", "ultimate_scepter", "Aghanim's Scepter", "阿哈利姆神杖"),
    ("dagon_3", "dagon", "Dagon", "达贡之神力"),                            # upgrade levels
    ("黑皇杖", "black_king_bar", "Black King Bar", "黑皇杖"),               # the game's Chinese name
    ("黑皇杖 (Black King Bar)", "black_king_bar", "Black King Bar", "黑皇杖"),
])
def test_item_alias_index(kb_env, identifier, key, name, name_zh):
    assert data_manager.resolve_item_key(identifier) == key
    assert data_manager.item_key(identifier) == key
    assert data_manager.resolve_item_name(identifier) == name       # lookups stay English
    assert data_manager.item_display_name(identifier) == data_manager.item_display_name(identifier, "en") == name
    assert data_manager.item_display_name(identifier, "zh") == name_zh
    assert data_manager.is_known_item(identifier)


@pytest.mark.parametrize("identifier, name, name_zh", [
    ("Wraith King", "Wraith King", "冥魂大帝"), ("skeleton_king", "Wraith King", "冥魂大帝"),
    ("npc_dota_hero_skeleton_king", "Wraith King", "冥魂大帝"), ("wk", "Wraith King", "冥魂大帝"),
    ("zuus", "Zeus", "宙斯"), ("SF", "Shadow Fiend", "影魔"), ("queenofpain", "Queen of Pain", "痛苦女王"),
    # the game's names, also for the spellings of dotaconstants
    ("Outworld Destroyer", "Outworld Destroyer", "殁境神蚀者"), ("Outworld Devourer", "Outworld Destroyer", "殁境神蚀者"),
    ("OD", "Outworld Destroyer", "殁境神蚀者"), ("Ring Master", "Ringmaster", "百戏大王"),
    ("npc_dota_hero_ringmaster", "Ringmaster", "百戏大王"),
    # the game's Chinese name and the bilingual form of a zh prompt find the hero as well
    ("冥魂大帝", "Wraith King", "冥魂大帝"), ("冥魂大帝（Wraith King）", "Wraith King", "冥魂大帝"),
])
def test_hero_alias_index(kb_env, identifier, name, name_zh):
    assert data_manager.resolve_hero_name(identifier) == name
    assert data_manager.hero_display_name(identifier) == data_manager.hero_display_name(identifier, "en") == name
    assert data_manager.hero_display_name(identifier, "zh") == name_zh
    assert data_manager.hero_slug(identifier) == data_manager.hero_slug(name)   # tags never get a Chinese name
    assert data_manager.is_known_hero(identifier)


def test_display_name_helpers_never_raise(kb_env):
    dm = data_manager
    # Unknown identifiers are kept and prettified, never dropped
    assert dm.hero_display_name("npc_dota_hero_brand_new") == "Brand New"
    assert dm.item_display_name("item_shiny_new_thing") == "Shiny New Thing"
    assert dm.item_display_name("Some Display Name") == "Some Display Name"
    assert not dm.is_known_hero("brand_new") and not dm.is_known_item("shiny_new_thing")
    # A recipe is named after its item; other units get a localized generic label
    assert dm.item_display_name("recipe_black_king_bar", "en") == "Black King Bar Recipe"
    assert dm.item_display_name("item_recipe_black_king_bar", "zh") == "图纸（黑皇杖）"
    assert dm.item_display_name("recipe_shiny_new_thing", "zh") == "图纸（Shiny New Thing）"
    assert dm.unit_display_name("npc_dota_hero_skeleton_king", "zh") == "冥魂大帝"
    assert dm.unit_display_name("npc_dota_hero_skeleton_king", "en") == "Wraith King"
    assert dm.unit_display_name("npc_dota_creep_badguys_melee", "zh") == "小兵"
    assert dm.unit_display_name("npc_dota_goodguys_tower1_mid", "en") == "Tower"
    assert dm.unit_display_name("npc_dota_roshan", "zh") == "肉山"
    assert dm.unit_display_name("npc_dota_lone_druid_bear1", "en") == "Lone Druid Bear1"
    assert dm.display_name("npc_dota_neutral_kobold", "en") == "Neutral creep"
    for helper in (dm.hero_display_name, dm.item_display_name, dm.ability_display_name,
                   dm.unit_display_name, dm.display_name):
        for junk in (None, "", 0, 3.5, [], {"a": 1}, object()):
            assert isinstance(helper(junk, "zh"), str)
        assert helper("", "en") == "" and helper(None) == ""
    # Names follow the language; without one they are English
    assert dm.ability_display_name("attack", "zh") == "普通攻击"
    assert dm.ability_display_name("attack", "en") == dm.display_name("attack", "en") == "Attack"
    assert (dm.ability_display_name("lion_impale", "en"), dm.ability_display_name("lion_impale", "zh")) == (
        "Earth Spike", "裂地尖刺")
    assert (dm.ability_display_name("item_dagon"), dm.ability_display_name("item_dagon", "zh")) == ("Dagon", "达贡之神力")
    assert (dm.display_name("npc_dota_hero_zuus"), dm.display_name("npc_dota_hero_zuus", "zh")) == ("Zeus", "宙斯")
    assert (dm.display_name("travel_boots"), dm.display_name("travel_boots", "zh")) == ("Boots of Travel", "远行鞋")
    assert dm.display_name("lion_impale", "zh") == "裂地尖刺"
    assert dm.display_name("some_new_thing") == dm.display_name("some_new_thing", "zh") == "Some New Thing"


def test_killer_section_uses_display_name(kb_env):
    ctx = _ctx(hero="Drow Ranger", killer="skeleton_king")
    text = knowledge_base.get_ability_item_knowledge(ctx, lang="zh")
    assert "击杀者关键技能（冥魂大帝）" in text and "skeleton_king" not in text and "Wraith King" not in text
    en = knowledge_base.get_ability_item_knowledge(ctx, lang="en")
    assert "Killer Key Abilities (Wraith King)" in en and "skeleton_king" not in en


# ---------------------------------------------------------------------------
# Talent names (template tokens) and the patch release table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, clean", [
    # the number opens the name: it is dropped, the sign stays
    ("+{s:bonus_damage_per_second} Ice Wall DPS", "+Ice Wall DPS"),
    ("-{s:bonus_AbilityCooldown}s Cold Snap Cooldown", "-Cold Snap Cooldown"),
    ("+{s:bonus_movement_speed}% Gush Slow", "+Gush Slow"),
    ("{s:value}% Larger Dissimilate Portals", "Larger Dissimilate Portals"),
    ("{s:bonus_treant_health}x Treant HP/Damage", "Treant HP/Damage"),
    ("+{s:bonus_a}/{s:bonus_b}s Drunken Brawler Brew Up / Extend Duration",
     "+Drunken Brawler Brew Up / Extend Duration"),
    ("+{s:bonus_ally_ms}% X Marks the Spot Move Speed", "+X Marks the Spot Move Speed"),
    # a number inside the sentence cannot be dropped without changing what it says
    ("Tidebringer Ignores {s:bonus_pierces_armor}% Armor", "Tidebringer Ignores X% Armor"),
    ("Dissimilate Roots for {s:bonus_root_duration}s", "Dissimilate Roots for Xs"),
    ("Add {s:bonus_attack_factor}% of Attack Damage to Talon Toss", "Add X% of Attack Damage to Talon Toss"),
    ("{s:bonus_mana_pct}% of Damage Dealt by Attacks Restored As Mana",
     "X% of Damage Dealt by Attacks Restored As Mana"),
    ("+{s:bonus_health} Health / +{s:bonus_damage} Damage  for Enchantress and her units",
     "+X Health / +X Damage for Enchantress and her units"),
    # nothing to clean / nothing left
    ("+200 Health", "+200 Health"), ("Outer Dissimilate Ring", "Outer Dissimilate Ring"),
    ("{s:value}", ""), ("+{s:value}%", ""), ("", ""), (None, ""),
])
def test_clean_ability_name_removes_template_tokens(raw, clean):
    assert data_manager.clean_ability_name(raw) == clean
    assert "{" not in data_manager.clean_ability_name(raw)


def test_clean_ability_name_fills_known_values():
    clean = data_manager.clean_ability_name
    assert clean("+{s:value} Damage", {"value": "25"}) == "+25 Damage"
    assert clean("-{s:value}s Spear Cooldown", {"value": "-3"}) == "-3s Spear Cooldown"   # one sign only
    assert clean("{s:value} AoE Shadow Strike", {"value": "400"}) == "400 AoE Shadow Strike"
    # a value for only some of the tokens fills nothing; neither does junk from a stored file
    assert clean("+{s:a}/{s:b}s Brew Up Duration", {"a": "1"}) == "+Brew Up Duration"
    assert clean("+{s:value} Damage", ["value"]) == clean("+{s:value} Damage", {}) == "+Damage"
    # only plain numbers of an attrib list are values (no per-level lists, no "-20%")
    assert data_manager._template_values([
        {"key": "value", "value": "25"}, {"key": "per_level", "value": ["1", "2"]},
        {"key": "slow", "value": "-20%"}, {"key": "ratio", "value": 1.5}, {"key": "flag", "value": True},
        {"value": "3"}, "junk"]) == {"value": "25", "ratio": "1.5"}
    assert data_manager._template_values(None) == {}


_TALENT_SLICE = {
    "special_bonus_unique_lion_5": {"dname": "-{s:bonus_AbilityCooldown}s Hex Cooldown"},
    "special_bonus_unique_wraith_king_facet_9": {"dname": "+{s:bonus_blast_dot_duration}s Wraithfire Blast Slow Duration"},
    "special_bonus_unique_tidehunter_9": {"dname": "Tidebringer Ignores {s:bonus_pierces_armor}% Armor"},
    "special_bonus_unique_test_value": {"dname": "+{s:value} Test Damage",
                                        "attrib": [{"key": "value", "header": "", "value": "25"}]},
    "special_bonus_hp_200": {"dname": "+200 Health"},
    "special_bonus_attributes": {},
}


def test_fetch_stores_talent_names_and_they_resolve_clean(kb_env, monkeypatch):
    # Without fetched data: the committed catalog (generated without template tokens; the
    # source resolves this one by now); unknown talents are named after their key
    name = data_manager.ability_display_name
    assert name("special_bonus_unique_lion_5") == "-2.0s Hex Cooldown"
    assert name("special_bonus_hp_200") == "+200 Health"
    assert name("special_bonus_attributes") == "Attribute Bonus"         # in no catalog at all
    assert name("special_bonus_unique_wraith_king_facet_9") == "Wraith King Facet 9"
    assert name("special_bonus_unique_test_value") == "Test Value"
    assert data_manager.resolve_ability_name("special_bonus_unique_wraith_king_facet_9") is None
    assert data_manager.get_talent_patch() == ""                         # no talent table yet
    index = data_manager._cached("ability_index", data_manager._build_ability_index)
    assert len(index) > 2000 and not [n for n in index.values() if "{" in n or not n.strip()]

    payloads = {"patch": [{"name": "7.41"}], **_CONSTANTS_SLICE}
    payloads["abilities"] = {**_CONSTANTS_SLICE["abilities"], **_TALENT_SLICE}
    monkeypatch.setattr(data_manager, "_api_get", lambda endpoint, timeout=30: payloads[endpoint])
    assert data_manager.fetch_and_save_gamedata(force=True)["talents"] == 5
    assert data_manager.get_meta()["talent_count"] == 5
    assert data_manager.get_talent_patch() == "7.41"

    stored = json.loads((kb_env.gamedata / data_manager.TALENTS_FILE).read_text(encoding="utf-8"))
    assert set(stored) == set(_TALENT_SLICE) - {"special_bonus_attributes"}
    assert stored["special_bonus_unique_lion_5"] == {"name": "-{s:bonus_AbilityCooldown}s Hex Cooldown"}
    assert stored["special_bonus_unique_test_value"]["values"] == {"value": "25"}
    # The fetched talents name what the catalog does not know (facet_9 is made up: the
    # catalog cannot have it), tokens resolved or removed
    assert name("special_bonus_unique_wraith_king_facet_9") == "+Wraithfire Blast Slow Duration"
    assert name("special_bonus_unique_tidehunter_9") == "Tidebringer Ignores X% Armor"
    assert name("special_bonus_unique_test_value") == "+25 Test Damage"
    assert name("special_bonus_unique_lion_5", lang="zh") == "-Hex Cooldown"
    # Talents are not hero abilities: the per-hero table is unchanged
    assert all(not a["key"].startswith("special_bonus_")
               for abilities in game_data.get_ability_table().values() for a in abilities.values())


def test_gamedata_of_the_previous_format_is_refetched_but_not_sanitised(kb_env, monkeypatch):
    """Schema 2 has no talents / patch table (and, like schema 3, no source and
    no hero vision); its hero, ability and item files are current."""
    _write_gamedata(kb_env.gamedata, patch="7.41")
    (kb_env.gamedata / "items.json").write_text(json.dumps({
        "Black King Bar": {"key": "black_king_bar", "cost": 4050, "name": "Black King Bar", "is_active": False},
        "Dagon": {"key": "dagon_5", "cost": 7650, "name": "Dagon"}}))
    (kb_env.gamedata / "meta.json").write_text(json.dumps(
        {"schema_version": 2, "patch_version": "7.41", "last_fetched": "2026-10-01T00:00:00Z"}))
    data_manager.invalidate_all_caches()
    assert data_manager.GAMEDATA_SCHEMA_VERSION == 4
    assert data_manager.get_meta()["outdated_format"] is True
    assert (data_manager.get_fetched_patch(), data_manager.get_talent_patch()) == ("7.41", "")   # no talent table
    monkeypatch.setattr(data_manager, "_api_get", lambda endpoint, timeout=30: [{"name": "7.41"}])
    assert data_manager.check_patch_version() == ("7.41", True)
    # ... while what a schema-2 file says is taken as written: only files older
    # than 2 had always-false flags and upgrade entries under the base name
    # (see test_legacy_gamedata_is_sanitised_and_refetched)
    items = game_data.get_item_table()
    assert items["Black King Bar"]["is_active"] is False
    assert (items["Dagon"]["key"], items["Dagon"]["cost"]) == ("dagon_5", 7650)


_PATCH_LIST = [
    {"name": "7.39", "date": "2025-05-22T23:36:01.602Z", "id": 58},
    {"name": "7.40", "date": "2025-12-16T00:50:40.281Z", "id": 59},
    {"name": "7.41", "date": "2026-03-24T00:50:59.580Z", "id": 60},
]


def _utc(text):
    import calendar
    import time as _time
    return calendar.timegm(_time.strptime(text, "%Y-%m-%d"))


def _write_patch_table(directory, fetched_at, patches=_PATCH_LIST):
    (directory / data_manager.PATCHES_FILE).write_text(json.dumps({"fetched_at": fetched_at, "patches": patches}))


def test_patch_for_time_reads_the_stored_table(kb_env, monkeypatch):
    from backend.app import config

    patch_for_time = data_manager.patch_for_time
    assert patch_for_time(1775643322) == ""                    # nothing fetched yet: no label
    _write_patch_table(kb_env.gamedata, "2026-04-10T00:00:00Z")
    assert patch_for_time(1775643322) == "7.41"                # 2026-04-08, real match 8762775364
    assert patch_for_time(1772470634) == "7.40"                # 2026-03-02, real match 8713267082
    assert patch_for_time(_utc("2025-12-15")) == "7.39"
    assert patch_for_time(_utc("2025-01-01")) == ""            # before the first entry
    assert patch_for_time(float(_utc("2026-03-25"))) == "7.41"
    for junk in (0, -5, None, "", "soon", float("nan"), float("inf") * 0, True, [], {}):
        assert patch_for_time(junk) == ""

    # After the newest known patch the table only vouches while it is fresh enough
    monkeypatch.setattr(config, "GAMEDATA_STALE_DAYS", 7)
    assert patch_for_time(_utc("2026-04-16")) == "7.41"        # 6 days after the download
    assert patch_for_time(_utc("2026-04-18")) == ""            # 8 days: a newer patch may exist
    assert patch_for_time(_utc("2026-03-02")) == "7.40"        # between two known patches: certain
    monkeypatch.setattr(config, "GAMEDATA_STALE_DAYS", 30)
    assert patch_for_time(_utc("2026-04-18")) == "7.41"

    # A table refreshed by another process (cron) counts at once, without a cache drop
    import os
    path = kb_env.gamedata / data_manager.PATCHES_FILE
    stamp = path.stat().st_mtime_ns
    _write_patch_table(kb_env.gamedata, "2026-06-01T00:00:00Z",
                       _PATCH_LIST + [{"name": "7.42", "date": "2026-05-20T00:00:00Z"}])
    os.utime(path, ns=(stamp + 10**9, stamp + 10**9))
    assert patch_for_time(_utc("2026-05-25")) == "7.42" and patch_for_time(_utc("2026-04-18")) == "7.41"

    # Unusable files give no label instead of an error
    for content in ("{broken", "[]", json.dumps({"patches": [{"name": "7.41", "date": "yesterday"}, 7]})):
        path.write_text(content)
        os.utime(path, ns=(stamp + 2 * 10**9 + len(content), stamp + 2 * 10**9 + len(content)))
        assert patch_for_time(1775643322) == ""


def test_patch_table_is_stored_by_the_fetch_and_the_patch_check(kb_env, monkeypatch):
    path = kb_env.gamedata / data_manager.PATCHES_FILE
    payloads = {"patch": _PATCH_LIST, **_CONSTANTS_SLICE}
    calls = []

    def api(endpoint, timeout=30):
        calls.append(endpoint)
        return payloads[endpoint]

    monkeypatch.setattr(data_manager, "_api_get", api)
    assert data_manager.fetch_and_save_gamedata(force=True)["patch_version"] == "7.41"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["patches"] == [{"name": p["name"], "date": p["date"]} for p in _PATCH_LIST]
    assert data_manager._parse_utc(stored["fetched_at"]) is not None
    assert calls.count("patch") == 1                            # the list of the patch check is reused
    import time as _time
    assert data_manager.patch_for_time(_time.time()) == "7.41"

    # An up-to-date check and the standalone check both refresh the table (and nothing else)
    for check in (lambda: data_manager.fetch_and_save_gamedata(force=False), data_manager.check_patch_version):
        path.unlink()
        calls.clear()
        check()
        assert calls == ["patch"] and path.exists()

    # A list without release dates (or a failed check) leaves the stored table alone
    before = path.read_text(encoding="utf-8")
    payloads["patch"] = [{"name": "7.41"}]
    assert data_manager.check_patch_version() == ("7.41", False)
    payloads["patch"] = []
    data_manager.fetch_and_save_gamedata(force=True)            # patch check fails, the fetch goes on
    assert path.read_text(encoding="utf-8") == before
