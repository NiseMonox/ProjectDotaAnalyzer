"""ClarityParser._transform_data on synthetic analyzer output (schema v2 and legacy)."""
from __future__ import annotations

import copy
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import pytest  # noqa: E402

from backend.app import geo, parser  # noqa: E402
from backend.app.parser import ClarityParser  # noqa: E402

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
    assert match.players[0].hero_name == "Unknown"
    assert match.tower_status_radiant == 2047  # no destruction evidence -> standing
    assert match.partial is False and match.buildings == []
