"""ClarityParser._transform_data on analyzer output: schema 3, schema 2 and legacy.

Synthetic fixtures plus ``fixtures/clarity_8762775364_schema2.json``, the real
(cut down, anonymised) schema-2 output of a current-patch replay.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import pytest  # noqa: E402

from backend.app import geo, parser  # noqa: E402
from backend.app.parser import ClarityParser  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module", autouse=True)
def committed_catalogs_only(tmp_path_factory):
    """Item and ability names come from the data layer: pin it to the committed
    catalogs, whatever game data this machine has fetched."""
    from backend.app.ai import data_manager

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(data_manager, "GAMEDATA_DIR", tmp_path_factory.mktemp("gamedata"))
        data_manager.invalidate_all_caches()
        yield
    data_manager.invalidate_all_caches()


# Lane equilibrium points (world units) used to build laning snapshots
TOP, MID, BOT, JUNGLE = (-5700.0, 4900.0), (-500.0, -400.0), (5400.0, -5600.0), (-1000.0, -4000.0)


def _snapshots(points: list[tuple[float, float]]) -> list[dict]:
    """One snapshot every 5s from 60s to 595s cycling through ``points``."""
    snaps = []
    for i, t in enumerate(range(0, 600, 5)):
        x, y = points[i % len(points)]
        snaps.append({"time": t, "x": x, "y": y, "hp": 600, "max_hp": 700, "mana": 300, "max_mana": 400,
                      "level": 3, "last_hits": i // 4, "denies": 0, "net_worth": 600 + i * 10})
    return snaps


def _player(idx: int, hero_name: str, lane_points, **extra) -> dict:
    slot = idx if idx < 5 else 128 + idx - 5
    base = {
        "player_slot": slot, "hero_id": 0, "hero_name": hero_name, "hero_npc_name": "",
        "player_name": f"name{idx}", "steam_id": 0,
        "kills": 1, "deaths": 1, "assists": 1, "level": 12, "last_hits": 80, "denies": 3,
        "gold_per_min": 400, "xp_per_min": 450, "hero_damage": 5000, "tower_damage": 100,
        "hero_healing": 0, "net_worth": 9000, "net_worth_source": "entity", "total_gold": 12000,
        "purchase_log": [], "ability_casts": [], "wards_placed": [], "starting_items": [],
        "final_items": [], "backpack": [], "neutral_item": "", "ability_upgrades": [],
        "gold_t": [0, 500, 1100], "xp_t": [0, 400, 900], "lh_t": [0, 4, 9],
        "killed": {}, "killed_by": {}, "damage": {}, "damage_taken": {}, "damage_inflictor": {},
        "gold_reasons": {"1": 100}, "xp_reasons": {}, "buyback_count": 0, "roshans_killed": 0,
        "courier_kills": 0, "observer_kills": 1, "sentry_kills": 0, "tower_kills": 0, "lane_kills": 50,
        "neutral_kills": 10, "ancient_kills": 2, "runes_log": [{"time": 120, "key": "5"}], "runes": {"5": 1},
        "lane": 0, "lane_role": 0,
        "laning_snapshots": _snapshots(lane_points),
        "laning_damage_events": [{"time": 70, "target_idx": 5, "damage": 40, "inflictor": "attack"}],
        "laning_regen_events": [{"time": 90, "item": "tango"}],
        "laning_creep_kills": [{"time": 65, "creep_type": "melee", "gold": 40, "extra": "ignored"}],
        "laning_denies": [{"time": 66, "creep_type": "ranged", "gold": 0}],
    }
    base.update(extra)
    return base


def _fixture() -> dict:
    players = [
        _player(0, "anti_mage", [BOT], hero_npc_name="npc_dota_hero_antimage",
                purchase_log=[{"time": -85, "key": "tango"}, {"time": -80, "key": "ward_observer"},
                              {"time": 400, "key": "power_treads"}],
                starting_items=["tango", "branches", "branches"],
                final_items=["power_treads", "bfury"], backpack=["magic_stick"], neutral_item="",
                ability_upgrades=[
                    {"name": "antimage_mana_break", "level": 1, "time": -60},
                    {"name": "special_bonus_unique_antimage", "level": 1, "time": 600},
                    {"name": "ability_lamp_use", "level": 1, "time": 10},
                    {"name": "hoodwink_sharpshooter_release", "level": 1, "time": 20},
                    {"name": "", "level": 2, "time": 30},
                ],
                ability_casts=[{"name": "antimage_blink", "time": 300}]),
        _player(1, "queen_of_pain", [MID]),                       # class-name conversion
        _player(2, "unknown_2", [TOP], hero_id=5),                # resolved via hero_id
        _player(3, "nevermore", [TOP, JUNGLE, MID, BOT]),         # roaming
        _player(4, "furion", [JUNGLE]),
        _player(5, "zuus", [TOP]),                                 # dire safe lane
        _player(6, "drow_ranger", [BOT]),                          # dire off lane
        _player(7, "lina", [MID]),
        _player(8, "lion", [(-7100.0, -6600.0)]),                  # only in a base -> unknown
        _player(9, "axe", [BOT]),
    ]
    return {
        "schema_version": 2,
        "coord_space": "world",
        "partial": True,
        "match_id": 777,
        "duration": 1800,
        "radiant_win": False,
        "game_mode": 23,
        "start_time": 1700000000,
        "radiant_score": 5,
        "dire_score": 9,
        "players": players,
        "wards": [
            {"type": "observer", "x": -1000.5, "y": 2000.25, "time": -40, "player_slot": 0, "team": "radiant",
             "destroyed_time": 320, "ehandle": 1},
            {"type": "observer", "x": -1000.5, "y": 2000.25, "time": 400, "player_slot": 0, "team": "radiant",
             "destroyed_time": None, "ehandle": 2},
            {"type": "sentry", "x": 3000.0, "y": 3000.0, "time": 500, "player_slot": -1, "team": "dire",
             "destroyed_time": 600, "ehandle": 3},
        ],
        "smokes": [{"time": 900, "end_time": 930, "team": "dire", "x": 4000.0, "y": 4000.0,
                    "path": [[4000.0, 4000.0], [2000.0, 1000.0]], "result": "caught",
                    "kills_gained": 0, "deaths_suffered": 2, "objective_taken": None,
                    "player_slot": 128, "player_slots": [128, 129]}],
        "teamfights": [{"start": 1000, "end": 1020, "radiant_deaths": 2, "dire_deaths": 1, "gold_delta": 0,
                        "players": [{"player_slot": 0, "hero_name": "anti_mage", "x": 10.0, "y": 20.0,
                                     "kills": 1, "deaths": 0, "damage_dealt": 900, "damage_taken": 300,
                                     "healing": 0, "gold_delta": 0, "xp_delta": 0,
                                     "positions": [{"x": 10.0, "y": 20.0, "time_offset": -2.0}],
                                     "ability_casts": [{"name": "antimage_blink", "key": "antimage_blink",
                                                        "time_offset": 1.5, "x": 11.0, "y": 21.0}],
                                     "death_time_offset": None, "abilities_used": ["antimage_blink"]}]}],
        "objectives": [
            {"time": 650, "type": "tower_kill", "key": "goodguys_tower1_mid", "team": "dire",
             "building_team": "radiant", "denied": False, "player_slot": 130},
            {"time": 1200, "type": "barracks_kill", "key": "badguys_range_rax_bot", "team": "dire",  # wrong team
             "player_slot": 3},
            {"time": 1300, "type": "roshan_kill", "team": "dire", "player_slot": 131},
            {"time": 128, "type": "CHAT_MESSAGE_FIRSTBLOOD", "slot": 5, "player_slot": 128, "key": "anti_mage"},
        ],
        "buildings": [
            {"name": "npc_dota_goodguys_tower1_mid", "team": "radiant", "type": "tower",
             "x": -1544.0, "y": -1408.0, "destroyed_time": 650},
            {"name": "", "team": "radiant", "type": "tower", "x": -6230.0, "y": 1810.0, "destroyed_time": 900},
            {"name": "dota_goodguys_tower4", "team": "radiant", "type": "tower",
             "x": -5420.0, "y": -5180.0, "destroyed_time": 1750},
            {"name": "npc_dota_badguys_range_rax_bot", "team": "dire", "type": "barracks",
             "x": 6100.0, "y": 3500.0, "destroyed_time": 1200},
            {"name": "npc_dota_badguys_fort", "team": "dire", "type": "fort", "x": 5500.0, "y": 5000.0,
             "destroyed_time": None},
        ],
        "radiant_gold_adv": [0, 100, -50],
        "radiant_xp_adv": [0, 20, 40],
        "chat": [{"time": 10, "type": "chat", "player_slot": 0, "key": "glhf"}],
        "picks_bans": [{"is_pick": False, "hero_id": 14, "team": 1, "order": 0}],
        "tower_status_radiant": 0, "tower_status_dire": 0,
        "barracks_status_radiant": 0, "barracks_status_dire": 0,
        "first_blood_time": 128, "region": 0, "patch": "",
        "parse_errors": {},
    }


@pytest.fixture(scope="module")
def match():
    return ClarityParser()._transform_data(_fixture())


def test_hero_lookup_normalizes_slugs(match):
    by_slot = {p.player_slot: p for p in match.players}
    assert (by_slot[0].hero_id, by_slot[0].hero_slug, by_slot[0].hero_name) == (1, "antimage", "Anti-Mage")
    assert (by_slot[1].hero_id, by_slot[1].hero_slug) == (39, "queenofpain")
    assert (by_slot[2].hero_id, by_slot[2].hero_slug) == (5, "crystal_maiden")
    assert by_slot[3].hero_slug == "nevermore"
    assert by_slot[4].hero_name == "Nature's Prophet"
    assert by_slot[128].hero_slug == "zuus"


def test_normalize_hero_key_variants():
    for name in ("Queen of Pain", "queenofpain", "npc_dota_hero_queenofpain", "queen_of_pain", "QueenOfPain"):
        assert parser.normalize_hero_key(name) == "queenofpain"


def test_lane_inference_from_laning_snapshots(match):
    lanes = {p.player_slot: (p.lane, p.lane_role, p.is_roaming) for p in match.players}
    assert lanes[0] == (1, 1, False)      # radiant bottom = safe
    assert lanes[1] == (2, 2, False)
    assert lanes[2] == (3, 3, False)      # radiant top = off
    assert lanes[3][2] is True and lanes[3][1] == 4   # roaming
    assert lanes[4] == (4, 4, False)      # jungle
    assert lanes[128] == (3, 1, False)    # dire top = safe
    assert lanes[129] == (1, 3, False)    # dire bottom = off
    assert lanes[131] == (0, 0, False)    # only base positions -> unknown


def test_buildings_are_canonical_and_bitmasks_computed(match):
    names = {b["name"] for b in match.buildings}
    assert "npc_dota_goodguys_tower1_mid" in names
    assert "npc_dota_goodguys_tower1_top" in names           # inferred from position
    assert "npc_dota_goodguys_tower4_bot" in names           # side from nearest default
    assert "npc_dota_badguys_range_rax_bot" in names
    for b in match.buildings:
        assert set(b) == {"name", "team", "type", "x", "y", "destroyed_time"}
        assert b["type"] in ("tower", "barracks", "fort", "other")
        assert b["team"] in ("radiant", "dire")
    bit = {k: 1 << i for i, k in enumerate(parser.TOWER_BIT_ORDER)}
    assert match.tower_status_radiant == 2047 - bit["mid_t1"] - bit["top_t1"] - bit["ancient_bot"]
    assert match.tower_status_dire == 2047
    assert match.barracks_status_dire == 63 - (1 << parser.BARRACKS_BIT_ORDER.index("bot_range"))
    assert match.barracks_status_radiant == 63


def test_objectives_normalized_with_team_from_building(match):
    kinds = [(o["type"], o.get("key"), o.get("team")) for o in match.objectives]
    assert ("building_kill", "npc_dota_goodguys_tower1_mid", "dire") in kinds
    # J8: a Dire building killed by a Radiant hero is a Radiant objective
    assert ("building_kill", "npc_dota_badguys_range_rax_bot", "radiant") in kinds
    rosh = next(o for o in match.objectives if o["type"] == "CHAT_MESSAGE_ROSHAN_KILL")
    assert rosh["team"] == "dire" and rosh["player_slot"] == 131
    assert [o["time"] for o in match.objectives] == sorted(o["time"] for o in match.objectives)
    assert "unit" not in match.objectives[0]
    # First blood: player_slot = killer, key = the victim's player_slot as a string
    # (schema 2 names the victim by hero: "anti_mage" is the hero in slot 0).
    first_blood = next(o for o in match.objectives if o["type"] == "CHAT_MESSAGE_FIRSTBLOOD")
    assert first_blood["player_slot"] == 128 and first_blood["key"] == "0"


def test_wards_attached_to_owners_with_world_coords(match):
    p0 = next(p for p in match.players if p.player_slot == 0)
    assert [w.time for w in p0.wards] == [-40, 400]          # re-ward of the same spot kept
    assert p0.wards[0].destroyed_time == 320 and p0.wards[1].destroyed_time is None
    assert p0.wards[0].x == -1000.5 and p0.wards[0].y == 2000.25
    assert p0.obs_placed == 2 and p0.sen_placed == 0
    unknown = next(w for w in match.wards if w.player_slot == -1)
    assert unknown.team == "dire" and unknown.ward_type == "sentry"


def test_items_skills_and_inventory(match):
    p0 = next(p for p in match.players if p.player_slot == 0)
    starting = sorted(it.item_key for it in p0.items if it.time < 0)
    # purchase log (tango, observer) + snapshot items missing from it (2x branches)
    assert starting == ["branches", "branches", "tango", "ward_observer"]
    assert p0.final_item_keys == ["power_treads", "bfury"]
    assert p0.final_items[0] == "Power Treads"
    assert p0.backpack == ["magic_stick"]
    assert [s.ability_key for s in p0.skills] == ["antimage_mana_break", "special_bonus_unique_antimage"]
    assert p0.skills[0].ability_name == "Mana Break"
    assert p0.ability_events[0].ability_key == "antimage_blink"
    assert p0.laning_creep_kills[0].creep_type == "melee"
    assert p0.runes == {"5": 1} and p0.observer_kills == 1 and p0.ancient_kills == 2


def test_match_level_fields(match):
    assert match.partial is True
    assert match.winner_known is False   # schema 2 cannot know the winner of a truncated replay
    assert match.game_mode == "turbo"
    assert match.smokes[0].result == "caught" and match.smokes[0].path[1] == [2000.0, 1000.0]
    assert match.teamfights[0].players[0].hero_slug == "antimage"
    assert match.teamfights[0].players[0].ability_casts[0].ability_name == "Blink"
    assert len(match.gold_xp_timeline) == 3
    assert match.picks_bans[0]["team"] == 1


def test_final_items_fall_back_to_purchases():
    raw = _fixture()
    raw["players"][0]["final_items"] = []
    raw["players"][0]["purchase_log"] = [{"time": 100, "key": k} for k in
                                         ("tango", "boots", "recipe_x", "blink", "ward_sentry", "bfury")]
    p0 = ClarityParser()._transform_data(raw).players[0]
    assert p0.final_item_keys == ["boots", "blink", "bfury"]
    # the consumables of the current game are no final items either (lotus fruits, madstone, ...)
    raw["players"][0]["purchase_log"] += [{"time": 200, "key": k} for k in (
        "famango", "great_famango", "greater_famango", "madstone_bundle", "royal_jelly",
        "foragers_health", "foragers_mana", "foragers_stats")]
    assert ClarityParser()._transform_data(raw).players[0].final_item_keys == ["boots", "blink", "bfury"]


def test_game_mode_codes_cover_the_current_enum():
    # DOTA_GameMode (dota_shared_enums.proto): 0-25 and, new, 26 = DOTA_GAMEMODE_BOT_CHALLENGE
    assert sorted(parser.GAME_MODES) == list(range(27))
    assert parser.game_mode_code(26) == "bot_challenge" and parser.game_mode_code("26") == "bot_challenge"
    assert parser.game_mode_code(23) == "turbo" and parser.game_mode_code(27) == "mode_27"
    raw = _fixture()
    raw["game_mode"] = 26
    assert ClarityParser()._transform_data(raw).game_mode == "bot_challenge"


def test_fetched_game_data_names_what_the_bundled_catalog_lacks(tmp_path, monkeypatch):
    """After a game update the fetched data knows items and abilities the committed
    catalog does not have yet (and current names); the catalog fills the rest."""
    from backend.app.ai import data_manager

    assert parser.item_display_name("brand_new_blade") == "Brand New Blade"      # title-cased key
    assert parser.ability_display_name("lion_brand_new_roar") == "lion_brand_new_roar"
    (tmp_path / "items.json").write_text(json.dumps({
        "Blade of Tomorrow": {"key": "brand_new_blade", "cost": 4200},
        "Flicker Dagger": {"key": "blink", "cost": 2250},
    }), encoding="utf-8")
    (tmp_path / "abilities.json").write_text(json.dumps({
        "lion": {"Roar of Tomorrow": {"key": "lion_brand_new_roar"}},
    }), encoding="utf-8")
    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", tmp_path)
    data_manager.invalidate_all_caches()
    try:
        assert parser.item_display_name("brand_new_blade") == "Blade of Tomorrow"
        assert parser.item_display_name("blink") == "Flicker Dagger"             # the fetched name wins
        assert parser.item_display_name("travel_boots") == "Boots of Travel"     # catalog only
        assert parser.ability_display_name("lion_brand_new_roar") == "Roar of Tomorrow"
        assert parser.ability_display_name("lion_impale") == "Earth Spike"
        raw = _fixture()
        raw["players"][0]["purchase_log"] = [{"time": 100, "key": "brand_new_blade"}]
        raw["players"][0]["ability_upgrades"] = [{"name": "lion_brand_new_roar", "level": 1, "time": 5}]
        p0 = ClarityParser()._transform_data(raw).players[0]
        assert {it.item_key: it.item_name for it in p0.items}["brand_new_blade"] == "Blade of Tomorrow"
        assert (p0.skills[0].ability_key, p0.skills[0].ability_name) == ("lion_brand_new_roar", "Roar of Tomorrow")
    finally:
        monkeypatch.undo()
        data_manager.invalidate_all_caches()
    # without the data layer the bundled catalog still names everything it has
    monkeypatch.setattr(parser, "_data_layer", lambda function: None)
    assert parser.item_display_name("travel_boots") == "Boots of Travel"
    assert parser.item_display_name("brand_new_blade") == "Brand New Blade"
    assert parser.ability_display_name("lion_impale") == "Earth Spike"


def test_world_interaction_casts_stay_in_the_match():
    """They are no hero abilities (the analyzer leaves them out of its ability
    statistics), but a cast is a known time and position of the hero."""
    raw = _fixture()
    raw["players"][0]["ability_casts"] = [
        {"name": "ability_lamp_use", "time": 250, "x": 100.0, "y": 200.0},
        {"name": "antimage_blink", "time": 300},
    ]
    p0 = ClarityParser()._transform_data(raw).players[0]
    assert [(e.ability_key, e.time, e.x, e.y) for e in p0.ability_events] == [
        ("ability_lamp_use", 250, 100.0, 200.0), ("antimage_blink", 300, None, None)]
    assert parser.is_utility_ability("ability_lamp_use") and parser.is_utility_ability("twin_gate_portal_warp")
    assert not parser.is_utility_ability("antimage_blink") and not parser.is_utility_ability("")


def test_legacy_output_coordinates_are_converted():
    raw = copy.deepcopy(_fixture())
    del raw["coord_space"]
    # legacy pseudo space: cell * 62.5 - 8000 (cell 128 -> 0, cell 64 -> -4000)
    raw["wards"] = [{"type": "observer", "x": 0.0, "y": -4000.0, "time": 5, "player_slot": 0}]
    raw["buildings"] = []
    match = ClarityParser()._transform_data(raw)
    ward = match.wards[0]
    assert ward.x == pytest.approx(geo.cell_to_world(128))
    assert ward.y == pytest.approx(geo.cell_to_world(64))
    assert ward.destroyed_time is None and ward.team == "radiant"


def test_minimal_output_does_not_crash():
    match = ClarityParser()._transform_data({"players": [{"player_slot": 0, "hero_name": None}], "wards": None})
    # Codes instead of the English placeholders "Unknown" / "Player 1"
    assert match.players[0].hero_name == "unknown_0" and match.players[0].player_name == ""
    assert match.tower_status_radiant == 2047  # no destruction evidence -> standing
    assert match.partial is False and match.buildings == []
    assert match.kills_log == [] and match.winner_known is False and match.players[0].position_log == []


# ---------------------------------------------------------------------------
# Names and keys (contract W3)
# ---------------------------------------------------------------------------

def test_unit_keys_use_display_hero_names():
    raw = _fixture()
    raw["players"][0].update({
        "killed": {"zuus": 2, "npc_dota_hero_lina": 1, "creep_badguys_melee": 40},
        "killed_by": {"zuus": 1, "badguys_tower1_mid": 1},
        "damage": {"zuus": 300, "lina": 20, "creep_badguys_melee": 50},
        "damage_taken": {"Zeus": 120, "npc_dota_neutral_kobold": 9},
        "damage_inflictor": {"dota_unknown": 10, "attack": 5, "": 1, "antimage_mana_break": 7},
        "laning_damage_events": [
            {"time": 70, "target_idx": 5, "damage": 40, "inflictor": "dota_unknown"},
            {"time": 71, "target_idx": 5, "damage": 30},
            {"time": 72, "target_idx": 5, "damage": 20, "inflictor": "antimage_mana_break"},
        ],
    })
    p0 = ClarityParser()._transform_data(raw).players[0]
    assert p0.killed == {"Zeus": 2, "Lina": 1}                   # heroes of the match only
    assert p0.killed_by == {"Zeus": 1}
    assert p0.damage == {"Zeus": 300, "Lina": 20, "npc_dota_creep_badguys_melee": 50}
    assert p0.damage_taken == {"Zeus": 120, "npc_dota_neutral_kobold": 9}
    assert p0.damage_inflictor == {"attack": 16, "antimage_mana_break": 7}
    assert [e.inflictor for e in p0.laning_damage_events] == ["attack", "attack", "antimage_mana_break"]


def test_item_keys_are_canonical_and_never_dropped():
    raw = _fixture()
    raw["players"][0].update({
        "final_items": ["boots_of_travel", "dustof_appearance", "invisibility_edge", "brand_new_item"],
        "backpack": ["empty_bottle", "item_tpscroll"],
        "neutral_item": "planeswalkers_cloak",
        "starting_items": ["gloves_of_haste"],
        "purchase_log": [{"time": 30, "key": "recipe_manta_style"}],
    })
    p0 = ClarityParser()._transform_data(raw).players[0]
    assert p0.final_item_keys == ["travel_boots", "dust", "invis_sword", "brand_new_item"]
    assert p0.final_items == ["Boots of Travel", "Dust of Appearance", "Shadow Blade", "Brand New Item"]
    assert p0.backpack == ["bottle", "tpscroll"]
    assert p0.neutral_item == "cloak"
    assert [(it.time, it.item_key) for it in p0.items] == [(-1, "gloves"), (30, "recipe_manta")]
    # every class-name leftover of seven 7.41f replays (271 item classes): the nine whose
    # snake case is not the key, and their recipes
    leftovers = {"cranium_basher": "basher", "gem_of_true_sight": "gem", "ghost_scepter": "ghost",
                 "greater_critical": "greater_crit", "gris_gris": "grisgris", "lesser_critical": "lesser_crit",
                 "orchid_malevolence": "orchid", "refresher_orb_shard": "refresher_shard",
                 "sheep_stick": "sheepstick", "recipe_sheep_stick": "recipe_sheepstick",
                 "recipe_greater_critical": "recipe_greater_crit"}
    assert {name: parser.canonical_item_key(name) for name in leftovers} == leftovers
    assert all(key in parser._items_by_key() for key in leftovers.values())
    # the items 7.41 added need no alias
    for key in ("crellas_crozier", "essence_distiller", "consecrated_wraps", "hydras_breath", "shawl", "wizard_hat"):
        assert parser.canonical_item_key(key) == key and key in parser._items_by_key()


def test_purchase_log_matches_items_without_ward_stack_merges():
    raw = _fixture()
    raw["players"][0].update({
        "purchase_log": [{"time": -89, "key": "ward_dispenser"}, {"time": -80, "key": "ward_sentry"},
                         {"time": 200, "key": "ward_dispenser"}, {"time": 400, "key": "boots"}],
        # inventory at the horn; a merged stack holds both ward types
        "starting_items": ["tango", "ward_dispenser", "magic_stick"],
    })
    p0 = ClarityParser()._transform_data(raw).players[0]
    expected = [(-80, "ward_sentry"), (-1, "tango"), (-1, "ward_observer"), (-1, "magic_stick"), (400, "boots")]
    assert [(it.time, it.item_key) for it in p0.items] == expected
    assert [(e["time"], e["key"]) for e in p0.purchase_log] == expected     # both views agree


def test_empty_real_inventory_is_not_replaced_by_purchases():
    raw = _fixture()
    raw["players"][0].update({
        "final_items": [], "final_items_source": "inventory",
        "purchase_log": [{"time": 100, "key": "blink"}],
    })
    assert ClarityParser()._transform_data(raw).players[0].final_item_keys == []
    raw["players"][0]["final_items_source"] = "purchases"
    assert ClarityParser()._transform_data(raw).players[0].final_item_keys == ["blink"]


def test_placeholder_names_become_codes():
    raw = _fixture()
    raw["players"][2].update({"player_name": "Player 3"})                      # analyzer placeholder
    raw["players"][3].update({"hero_name": "largo", "hero_npc_name": "npc_dota_hero_largo", "hero_id": 0})
    raw["players"][4].update({"hero_name": "unknown_4", "hero_npc_name": "", "hero_id": 999})
    players = ClarityParser()._transform_data(raw).players
    assert players[2].player_name == "" and players[0].player_name == "name0"
    # not in the static catalog: the internal name is all there is
    assert (players[3].hero_name, players[3].hero_slug) == ("Largo", "largo")
    assert (players[4].hero_name, players[4].hero_slug, players[4].hero_id) == ("hero_999", "", 999)


# ---------------------------------------------------------------------------
# Series, lane efficiency, draft, buildings
# ---------------------------------------------------------------------------

def test_minute_series_stop_at_game_end():
    raw = _fixture()
    raw["duration"] = 130                                   # minutes 0, 1, 2
    for p in raw["players"]:
        p["gold_t"] = [0, 500, 1100, 1150, 1150, 1150]      # three post-game samples (schema 2)
        p["xp_t"] = [0, 400, 900, 950, 950, 950]
        p["lh_t"] = [0, 4, 9, 9, 9, 9]
    raw["radiant_gold_adv"] = [0, 100, -50, -60, -60, -60]
    raw["radiant_xp_adv"] = [0, 20, 40, 41, 41, 41]
    match = ClarityParser()._transform_data(raw)
    assert match.players[0].gold_t == [0, 500, 1100]
    assert match.players[0].xp_t == [0, 400, 900] and match.players[0].lh_t == [0, 4, 9]
    assert match.radiant_gold_adv == [0, 100, -50] and match.radiant_xp_adv == [0, 20, 40]
    assert [s.time for s in match.gold_xp_timeline] == [0, 60, 120]


def test_lane_efficiency_and_denies_per_minute():
    raw = _fixture()
    raw["players"][0]["gold_t"] = [0] + [400 * m for m in range(1, 13)]      # 4000 gold at 10:00
    for snap in raw["players"][0]["laning_snapshots"]:
        snap["denies"] = snap["time"] // 120
        snap["time"] += 1                                   # real output is not on exact 5 s marks
    p0, p1 = ClarityParser()._transform_data(raw).players[:2]
    assert p0.lane_efficiency == pytest.approx(4000 / 4948) and p0.lane_efficiency_pct == 80
    assert p0.dn_t == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4]     # minutes 0..10 (last snapshot is 5 s early)
    assert (p1.lane_efficiency, p1.lane_efficiency_pct) == (0.0, 0.0)        # gold_t too short


def test_draft_fallback_gets_hero_ids_and_no_invented_ban_teams():
    raw = _fixture()
    raw["picks_bans"] = [
        {"is_pick": False, "hero_id": 14, "team": 0, "order": 0},
        {"is_pick": False, "hero_id": 23, "team": 1, "order": 1},
        {"is_pick": True, "hero_id": 0, "hero_name": "zuus", "team": 1, "order": 2},
    ]
    draft = ClarityParser()._transform_data(raw).picks_bans
    assert [d["team"] for d in draft] == [None, None, 1]    # schema 2 alternated the ban teams
    assert draft[2]["hero_id"] == 22 and draft[2]["is_pick"] is True
    # a draft from the replay's file info (no hero_name on picks) keeps its teams
    raw["picks_bans"] = [{"is_pick": False, "hero_id": 14, "team": 1, "order": 0},
                         {"is_pick": True, "hero_id": 22, "team": 0, "order": 1}]
    assert [d["team"] for d in ClarityParser()._transform_data(raw).picks_bans] == [1, 0]


def test_barracks_entity_names_and_unnamed_barracks_are_canonical():
    raw = _fixture()
    raw["objectives"] = []
    raw["buildings"] = [
        {"name": "good_rax_melee_top", "team": "radiant", "type": "barracks",
         "x": -6336.0, "y": -3758.0, "destroyed_time": 1500},
        {"name": "", "team": "radiant", "type": "barracks", "x": -6844.0, "y": -3759.0, "destroyed_time": None},
        {"name": "", "team": "dire", "type": "barracks", "x": 4336.0, "y": 4183.0, "destroyed_time": 1600},
        {"name": "", "team": "dire", "type": "barracks", "x": 0.0, "y": 0.0, "destroyed_time": None},
        {"name": "bad_rax_range_bot", "team": "dire", "type": "barracks", "x": 0.0, "y": 0.0, "destroyed_time": 1700},
    ]
    match = ClarityParser()._transform_data(raw)
    assert [b["name"] for b in match.buildings] == [
        "npc_dota_goodguys_melee_rax_top", "npc_dota_goodguys_range_rax_top",
        "npc_dota_badguys_range_rax_mid", "",              # nowhere near a barracks: left unnamed
        "npc_dota_badguys_range_rax_bot",                  # the entity name alone is enough
    ]
    order = parser.BARRACKS_BIT_ORDER
    assert match.barracks_status_radiant == 63 - (1 << order.index("top_melee"))
    assert match.barracks_status_dire == 63 - (1 << order.index("mid_range")) - (1 << order.index("bot_range"))
    assert parser.parse_building_name("good_rax_melee_top") == {
        "team": "radiant", "type": "barracks", "key": "top_melee"}
    assert parser.parse_building_name("bad_rax_range_bot") == {"team": "dire", "type": "barracks", "key": "bot_range"}
    assert parser.parse_building_name("npc_dota_badguys_range_rax_bot") == parser.parse_building_name(
        "bad_rax_range_bot")


def test_sided_tier4_objective_clears_the_right_bit():
    bit = {k: 1 << i for i, k in enumerate(parser.TOWER_BIT_ORDER)}
    sided = [{"type": "building_kill", "key": "npc_dota_goodguys_tower4_bot"}]
    assert parser.building_status_bitmasks([], sided)["tower_status_radiant"] == 2047 - bit["ancient_bot"]
    # a combat-log name without the side can only be reconciled in order
    sideless = [{"type": "building_kill", "key": "npc_dota_goodguys_tower4"}]
    assert parser.building_status_bitmasks([], sideless)["tower_status_radiant"] == 2047 - bit["ancient_top"]
    both = sided + [{"type": "building_kill", "key": "npc_dota_goodguys_tower4_top"}]
    status = parser.building_status_bitmasks([], both)["tower_status_radiant"]
    assert status == 2047 - bit["ancient_top"] - bit["ancient_bot"]


# ---------------------------------------------------------------------------
# Analyzer schema 3
# ---------------------------------------------------------------------------

def _v3_fixture() -> dict:
    """The synthetic fixture plus the fields analyzer schema 3 adds or changes."""
    raw = _fixture()
    raw.update({
        "schema_version": 3, "partial": False, "match_id": 8762775364, "match_id_source": "file_info",
        "radiant_win": None, "game_started": True, "cluster": 151, "region": 5, "draft_source": "game_rules",
        "kills_log": [
            {"time": 1004.5, "victim_slot": 128, "killer_slot": 0, "killer_unit": "npc_dota_hero_antimage",
             "killer_team": "radiant", "x": 12.0, "y": 22.0, "assist_slots": [1, 2, 77], "deny": False},
            {"time": 1010.04, "victim_slot": 1, "killer_slot": None, "killer_unit": "npc_dota_badguys_tower1_mid",
             "killer_team": "dire", "x": None, "y": None, "assist_slots": [], "deny": False},
            # written by an analyzer before 1.4.0: no killer_team
            {"time": 500.0, "victim_slot": 2, "killer_slot": 129, "killer_unit": "npc_dota_lycan_wolf1",
             "x": -5.5, "y": 6.5, "assist_slots": [128], "deny": False},
            {"time": 1300, "victim_slot": 3, "killer_slot": 4, "killer_unit": "npc_dota_hero_furion",
             "killer_team": "radiant", "x": 1.0, "y": 2.0, "assist_slots": [], "deny": True},
            {"time": 5.0, "victim_slot": 77, "killer_slot": 0, "killer_unit": "", "assist_slots": []},
        ],
        "picks_bans": [{"is_pick": False, "hero_id": 14, "team": None, "order": 10},
                       {"is_pick": True, "hero_id": 1, "hero_name": "antimage", "team": 0, "order": 0,
                        "player_slot": 0}],
    })
    # The flag wins over the lifetime: an observer killed in its last second
    # (lifetime 360 s), and a sentry that vanished early without an enemy.
    raw["wards"][0]["dewarded"] = True
    raw["wards"][2]["dewarded"] = False
    raw["players"][0]["position_log"] = [[-90, -7100, -6600, 100], [-85, -7000.4, -6500.6, 97],
                                         [0, 5400, -5600, 0], "bad", [5, 1, 2]]
    fight = raw["teamfights"][0]
    fight.update({"start": 989, "end": 1035, "first_death": 1004, "last_death": 1010, "gold_delta": -350})
    fight["players"][0].update({"kill_targets": [{"target_slot": 128, "time_offset": 15.5}],
                                "item_uses": {"item_black_king_bar": 1}, "buybacks": 1, "gold_delta": 420})
    fight["players"].append({"player_slot": 128, "hero_name": "zuus", "x": 12.0, "y": 22.0, "kills": 0,
                             "deaths": 1, "damage_dealt": 0, "damage_taken": 900, "healing": 0,
                             "death_time_offset": 15.5, "kill_targets": [], "abilities_used": []})
    raw["objectives"] += [
        {"time": 1750, "type": "tower_kill", "key": "goodguys_tower4", "building": "npc_dota_goodguys_tower4_bot",
         "team": "dire", "building_team": "radiant"},
        {"time": 140, "type": "CHAT_MESSAGE_FIRSTBLOOD", "slot": 0, "player_slot": 0, "key": "zuus",
         "victim_slot": 128},
    ]
    return raw


@pytest.fixture(scope="module")
def match_v3():
    return ClarityParser()._transform_data(_v3_fixture())


def test_v3_match_fields(match_v3):
    assert match_v3.match_id == 8762775364              # 64-bit id survives
    assert match_v3.winner_known is False and match_v3.radiant_win is False
    assert (match_v3.cluster, match_v3.region) == (151, 5)
    # a known winner on a schema-3 partial replay is trusted
    raw = _v3_fixture()
    raw.update({"radiant_win": True, "partial": True})
    known = ClarityParser()._transform_data(raw)
    assert known.winner_known is True and known.radiant_win is True


def test_v3_kills_log(match_v3):
    kills = match_v3.kills_log
    assert [k.time for k in kills] == [500.0, 1004.5, 1010.0, 1300.0]      # time order, unknown victim dropped
    summon, hero, tower, deny = kills
    assert (summon.victim_slot, summon.killer_slot, summon.killer_unit) == (2, 129, "npc_dota_lycan_wolf1")
    assert (summon.x, summon.y, summon.assist_slots) == (-5.5, 6.5, [128])
    assert (hero.killer_slot, hero.killer_unit, hero.assist_slots) == (0, "", [1, 2])
    assert (tower.killer_slot, tower.killer_unit, tower.x, tower.y) == (None, "npc_dota_badguys_tower1_mid", None, None)
    assert deny.deny is True and not hero.deny
    assert all(k.confirmed for k in kills)
    assert [k.killer_team for k in kills] == [None, "radiant", "dire", "radiant"]


def test_v3_kills_log_names_the_side_of_a_unit_whose_name_does_not():
    """Match 9038370833 at 23:53: a dire hero dies to the radiant fountain and Valve
    counts the kill for Radiant (4-29). "dota_fountain" names no side; killer_team does."""
    raw = _v3_fixture()
    raw["kills_log"] = [
        {"time": 1433.8, "victim_slot": 129, "killer_slot": None, "killer_unit": "dota_fountain",
         "killer_team": "radiant", "x": -7051.0312, "y": -6588.875, "assist_slots": [], "deny": False},
        {"time": 240.2, "victim_slot": 128, "killer_slot": None, "killer_unit": "npc_dota_neutral_dark_troll",
         "killer_team": "neutral", "x": -5047.9375, "y": 4367.0312, "assist_slots": [], "deny": False},
        {"time": 300.0, "victim_slot": 0, "killer_slot": None, "killer_unit": "", "killer_team": 3,
         "x": 1.0, "y": 2.0, "assist_slots": [], "deny": False},                    # not a code
    ]
    neutral, unknown, fountain = ClarityParser()._transform_data(raw).kills_log
    assert (fountain.killer_slot, fountain.killer_unit, fountain.killer_team) == (None, "dota_fountain", "radiant")
    assert neutral.killer_team == "neutral" and unknown.killer_team is None


def test_v3_position_log_and_wards(match_v3):
    p0 = match_v3.players[0]
    assert p0.position_log == [[-90, -7100, -6600, 100], [-85, -7000, -6500, 97], [0, 5400, -5600, 0], [5, 1, 2, -1]]
    assert match_v3.players[1].position_log == []
    by_time = {w.time: w for w in match_v3.wards}
    assert by_time[-40].dewarded is True                 # lifetime alone would say "expired"
    assert by_time[500].dewarded is False                # lifetime alone would say "dewarded"
    assert by_time[400].dewarded is False and by_time[400].destroyed_time is None


def test_v3_teamfight_kill_targets_and_objectives(match_v3):
    fight = match_v3.teamfights[0]
    killer, victim = fight.players
    assert killer.kill_targets == [{"target": "Zeus", "target_slot": 128, "time_offset": 15.5}]
    assert killer.item_uses == {"black_king_bar": 1} and killer.buybacks == 1 and killer.gold_delta == 420
    assert victim.death_time_offset == 15.5 and victim.hero_name == "Zeus"
    assert (fight.start_time, fight.end_time, fight.gold_delta) == (989, 1035, -350)
    keys = [o.get("key") for o in match_v3.objectives if o["type"] == "building_kill"]
    assert "npc_dota_goodguys_tower4_bot" in keys        # the entity name, with the side
    bit = {k: 1 << i for i, k in enumerate(parser.TOWER_BIT_ORDER)}
    assert not match_v3.tower_status_radiant & bit["ancient_bot"] and match_v3.tower_status_radiant & bit["ancient_top"]
    first_blood = [o for o in match_v3.objectives if o["type"] == "CHAT_MESSAGE_FIRSTBLOOD" and o["time"] == 140][0]
    assert (first_blood["player_slot"], first_blood["key"]) == (0, "128")
    assert [(d["is_pick"], d["hero_id"], d["team"]) for d in match_v3.picks_bans] == [(False, 14, None), (True, 1, 0)]


def test_kill_targets_come_from_the_kill_log_when_the_analyzer_has_none():
    raw = _v3_fixture()
    for tp in raw["teamfights"][0]["players"]:
        tp.pop("kill_targets", None)
        tp["death_time_offset"] = None
    fight = ClarityParser()._transform_data(raw).teamfights[0]
    killer, victim = fight.players
    assert killer.kill_targets == [{"target": "Zeus", "target_slot": 128, "time_offset": 15.5}]
    assert victim.death_time_offset == 15.5


def test_decode_logs_unreliable_match_id_and_combat_log_scoreboard(caplog):
    raw = _v3_fixture()
    # Truncated replay (no file info) parsed without the 64-bit decoder: the
    # value is the real id modulo 2^32, i.e. not the match id.
    raw.update({"match_id": 172840772, "match_id_source": "game_rules_32bit"})
    raw["players"][0]["kda_source"] = "combat_log"
    with caplog.at_level("INFO", logger="backend.app.parser"):
        match = ClarityParser()._decode_and_transform(json.dumps(raw).encode("utf-8"), "", "cut.dem")
    assert match.match_id == 0               # unknown (it used to be passed through as the id)
    assert "32 bits" in caplog.text and "kda_source" in caplog.text
    # the analyzer normally reads the full id from the game rules as well
    raw.update({"match_id": 8762775364, "match_id_source": "game_rules"})
    assert ClarityParser()._transform_data(raw).match_id == 8762775364
    with pytest.raises(parser.ReplayParseError):
        ClarityParser()._decode_and_transform(b"[1, 2]", "", "cut.dem")


def test_decode_logs_diagnostics_apart_from_handler_errors(caplog):
    """``parse_errors`` also carries counts that describe the match (match 9038374031 has
    ``score_credits: 1``): they are not reported as failed handlers."""
    def logged(parse_errors) -> list[str]:
        raw = _v3_fixture()
        raw["parse_errors"] = parse_errors
        caplog.clear()
        with caplog.at_level("INFO", logger="backend.app.parser"):
            ClarityParser()._decode_and_transform(json.dumps(raw).encode("utf-8"), "", "9038374031.dem")
        return [r.getMessage() for r in caplog.records if "9038374031.dem" in r.getMessage()]

    assert logged({"score_credits": 1}) == ["Clarity diagnostics for 9038374031.dem: {'score_credits': 1}"]
    assert logged({"abilities.poll": 2, "skill_points": 1, "entity_names": 40}) == [
        "Clarity handler errors for 9038374031.dem: {'abilities.poll': 2, 'entity_names': 40}",
        "Clarity diagnostics for 9038374031.dem: {'skill_points': 1}",
    ]
    assert logged({}) == [] and logged(None) == []


def test_v3_scoreboard_extras_smoke_user_ward_killer_and_deltas():
    raw = _v3_fixture()
    raw.update({"game_build": 6762, "radiant_score": 5})
    raw["players"][0].update({
        "stuns": 12.5, "teamfight_participation": 0.75, "actions_per_min": 210,
        "multi_kills": {"2": 1}, "kill_streaks": {"3": 1, "4": 1},
        "buyback_log": [{"time": 1500}, {"time": 900}], "buyback_count": 2,
    })
    raw["wards"][0]["killer_slot"] = 128           # dewarded by a hero
    raw["wards"][2]["killer_slot"] = 0             # not dewarded: no killer
    raw["teamfights"][0]["xp_delta"] = 640
    match = ClarityParser()._transform_data(raw)
    p0, p1 = match.players[0], match.players[1]
    assert (p0.stuns, p0.teamfight_participation, p0.actions_per_min) == (12.5, 0.75, 210)
    assert (p0.multi_kills, p0.kill_streaks) == ({"2": 1}, {"3": 1, "4": 1})
    assert p0.buyback_log == [{"time": 900}, {"time": 1500}] and p0.buyback_count == 2
    # without the game's own counter: (kills + assists) / team kills
    assert p1.teamfight_participation == pytest.approx((1 + 1) / 5) and p1.buyback_log == []
    by_time = {w.time: w for w in match.wards}
    assert by_time[-40].killer_slot == 128 and by_time[500].killer_slot is None
    assert match.smokes[0].player_slot == 128      # the hero that used the smoke
    assert match.teamfights[0].xp_delta == 640 and match.game_build == 6762
    # schema 2 named the first hero the smoke reached there: not the user
    assert ClarityParser()._transform_data(_fixture()).smokes[0].player_slot is None


def test_kill_announcements_come_from_the_kill_log_when_the_analyzer_has_none():
    raw = _v3_fixture()
    # Slot 0: kills at 100, 110 (double), 127 (triple), 400 (new chain, 4 in a
    # row), dies at 500, kills again at 900.
    raw["kills_log"] = [
        {"time": t, "victim_slot": 128 + i % 5, "killer_slot": 0, "killer_unit": "", "assist_slots": [], "deny": False}
        for i, t in enumerate((100.0, 110.0, 127.9, 400.0))
    ] + [
        {"time": 500.0, "victim_slot": 0, "killer_slot": 128, "killer_unit": "", "assist_slots": [], "deny": False},
        {"time": 900.0, "victim_slot": 129, "killer_slot": 0, "killer_unit": "", "assist_slots": [], "deny": False},
        {"time": 905.0, "victim_slot": 1, "killer_slot": 0, "killer_unit": "", "assist_slots": [], "deny": True},
    ]
    match = ClarityParser()._transform_data(raw)
    assert match.players[0].multi_kills == {"2": 1, "3": 1}
    assert match.players[0].kill_streaks == {"3": 1, "4": 1}
    assert match.players[5].multi_kills == {} and match.players[5].kill_streaks == {}


def test_schema3_purchase_log_is_complete_without_the_horn_snapshot():
    # Schema 3 lists the strategy-time purchases itself. The horn snapshot can
    # hold an item that was assembled before the horn (not a purchase): it
    # must not come back as a starting item.
    raw = _v3_fixture()
    raw["players"][0].update({
        "purchase_log": [{"time": -90, "key": "magic_stick"}, {"time": -90, "key": "branches"},
                         {"time": -90, "key": "branches"}, {"time": -60, "key": "recipe_magic_wand"}],
        "starting_items": ["magic_wand"],
    })
    items = ClarityParser()._transform_data(raw).players[0].items
    assert [(it.time, it.item_key) for it in items] == [
        (-90, "magic_stick"), (-90, "branches"), (-90, "branches"), (-60, "recipe_magic_wand")]
    raw["schema_version"] = 2                       # older output: the snapshot fills the gap
    older = ClarityParser()._transform_data(raw).players[0].items
    assert (-1, "magic_wand") in [(it.time, it.item_key) for it in older]


def test_courier_lost_objective_keeps_team_killer_and_bounty():
    raw = _v3_fixture()
    raw["objectives"].append({"time": 647, "type": "CHAT_MESSAGE_COURIER_LOST", "slot": 4, "player_slot": 4,
                              "team": "dire", "value": 45})
    lost = next(o for o in ClarityParser()._transform_data(raw).objectives if o["type"] == "CHAT_MESSAGE_COURIER_LOST")
    assert lost == {"time": 647, "player_slot": 4, "type": "CHAT_MESSAGE_COURIER_LOST", "team": "dire",
                    "slot": 4, "value": 45}


# ---------------------------------------------------------------------------
# Real replay output (match 8762775364, analyzer schema 2 of the pre-fix jar)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_raw():
    return json.loads((FIXTURES / "clarity_8762775364_schema2.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real(real_raw):
    return ClarityParser()._transform_data(copy.deepcopy(real_raw))


def test_real_inventory_keys_match_the_catalog(real):
    by_hero = {p.hero_name: p for p in real.players}
    # class-name leftovers of this build: boots_of_travel, dustof_appearance,
    # invisibility_edge, empty_bottle
    assert "travel_boots" in by_hero["Invoker"].final_item_keys
    assert {"invis_sword", "dust"} <= set(by_hero["Lion"].final_item_keys)
    assert "bottle" in by_hero["Void Spirit"].backpack
    known = parser._items_by_key()
    for p in real.players:
        for key in p.final_item_keys + p.backpack:
            # every key resolves, except items newer than the static catalog (kept as they are)
            assert key in known or key in ("consecrated_wraps", "splintmail", "shawl", "wizard_hat"), key
    assert by_hero["Invoker"].neutral_item == "spellslinger"


def test_real_buildings_named_by_position_and_bitmasks(real, real_raw):
    assert all(b["name"] == "" for b in real_raw["buildings"])     # this build exposes no entity names
    names = [b["name"] for b in real.buildings]
    assert len(set(names)) == 36 and all(parser.parse_building_name(n) for n in names)
    assert sum(1 for n in names if "_rax_" in n) == 12
    # identical to Valve's status for this match
    assert (real.tower_status_radiant, real.tower_status_dire) == (256, 1972)
    assert (real.barracks_status_radiant, real.barracks_status_dire) == (48, 63)


def test_real_series_and_scoreboard_fields(real):
    minutes = real.duration // 60 + 1                     # 34:51 -> 35 samples (the raw output has 38)
    assert all(len(p.gold_t) == len(p.xp_t) == len(p.lh_t) == minutes for p in real.players)
    assert len(real.radiant_gold_adv) == len(real.radiant_xp_adv) == len(real.gold_xp_timeline) == minutes
    invoker = real.players[0]
    assert invoker.lane_efficiency == pytest.approx(0.7332, abs=1e-3) and invoker.lane_efficiency_pct == 73
    assert invoker.dn_t[10] == 12                         # OpenDota's dn_t[10] for the same player
    assert invoker.killed == {"Wraith King": 2, "Zeus": 2, "Void Spirit": 2, "Lion": 1}
    assert set(invoker.killed_by) == {"Void Spirit", "Wraith King", "Zeus"}
    assert "attack" in invoker.damage_inflictor and "dota_unknown" not in invoker.damage_inflictor
    assert real.winner_known is True and real.radiant_win is False


def test_real_purchases_draft_and_first_blood(real):
    for p in real.players:
        assert all(e["key"] != "ward_dispenser" for e in p.purchase_log)
        assert [(e["time"], e["key"]) for e in p.purchase_log] == [(it.time, it.item_key) for it in p.items]
    legion = next(p for p in real.players if p.hero_name == "Legion Commander")
    # bought in strategy time: no combat-log row, only the inventory at the horn
    assert sorted(e["key"] for e in legion.purchase_log if e["time"] <= 0) == [
        "faerie_fire", "gauntlets", "gauntlets", "quelling_blade", "tango"]
    picks = [d for d in real.picks_bans if d["is_pick"]]
    bans = [d for d in real.picks_bans if not d["is_pick"]]
    assert [d["hero_id"] for d in picks] == [p.hero_id for p in real.players]     # the raw picks carry 0
    assert bans and all(d["team"] is None for d in bans)
    first_blood = next(o for o in real.objectives if o["type"] == "CHAT_MESSAGE_FIRSTBLOOD")
    assert (first_blood["player_slot"], first_blood["key"]) == (4, "128")     # Hoodwink killed Lion
    assert {m["key"] for m in real.chat if m["type"] == "chatwheel"} == {"chatwheel_71"}


def test_real_wards_fall_back_to_the_lifetime_rule(real):
    from collections import Counter
    counts = Counter((w.ward_type, w.dewarded) for w in real.wards)
    # the same split OpenDota's attacker names give for this match
    assert counts == {("observer", True): 8, ("observer", False): 22, ("sentry", True): 23, ("sentry", False): 23}
