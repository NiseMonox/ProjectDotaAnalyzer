"""OpenDotaParser: transform of a synthetic match payload and API error mapping.

Also ``fixtures/opendota_8762775364.json``, the real (cut down, anonymised)
API response of a match with Wraith King reincarnations, and
``fixtures/opendota_constants.json`` with the constants it needs.

No network: urlopen, _api_get and the constants download are monkeypatched.
"""
from __future__ import annotations

import asyncio
import copy
import io
import json
import os
import threading
import urllib.error
from collections import Counter
from pathlib import Path

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import pytest  # noqa: E402

from backend.app import geo, parser  # noqa: E402
from backend.app.parser import OpenDotaError, OpenDotaParser  # noqa: E402


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module", autouse=True)
def committed_catalogs_only(tmp_path_factory):
    """Names and the patch-by-time fallback come from the data layer: pin it to
    the committed catalogs, whatever game data this machine has fetched."""
    from backend.app.ai import data_manager

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(data_manager, "GAMEDATA_DIR", tmp_path_factory.mktemp("gamedata"))
        data_manager.invalidate_all_caches()
        yield
    data_manager.invalidate_all_caches()


def _player(slot: int, hero_id: int, **extra) -> dict:
    base = {
        "player_slot": slot,
        "hero_id": hero_id,
        "personaname": f"p{slot}",
        "kills": 3, "deaths": 2, "assists": 5, "level": 18,
        "last_hits": 150, "denies": 8, "gold_per_min": 500, "xp_per_min": 600,
        "hero_damage": 12000, "tower_damage": 900, "hero_healing": 0,
        "net_worth": 15000,
        "gold_t": [0, 600, 1400], "xp_t": [0, 300, 700], "lh_t": [0, 5, 12],
    }
    base.update(extra)
    return base


def _fixture() -> dict:
    radiant = [_player(i, 1 + i) for i in range(5)]
    dire = [_player(128 + i, 6 + i) for i in range(5)]
    p0 = radiant[0]
    p0.update({
        "obs_log": [
            {"time": 100, "x": 128, "y": 128, "ehandle": 11},
            {"time": 200, "x": 100.5, "y": 150.25},           # no ehandle -> positional match
            {"time": 900, "x": 100.5, "y": 150.25},           # re-ward of the same spot, alive
        ],
        "obs_left_log": [
            {"time": 460, "x": 128, "y": 128, "ehandle": 11},
            {"time": 260, "x": 100.5, "y": 150.25},
        ],
        "sen_log": [{"time": -30, "x": 64, "y": 191, "ehandle": 22}],
        "sen_left_log": [],
        "purchase_log": [
            {"time": -80, "key": "tango"},
            {"time": 20, "key": "branches"},
            {"time": 25, "key": "branches"},
            {"time": 40, "key": "quelling_blade"},
            {"time": 300, "key": "boots"},
        ],
        "item_0": 1, "item_1": 0, "item_2": 999999,
        "backpack_0": 1, "item_neutral": 1,
        "lane": 1, "lane_role": 1,
        "roshan_kills": 2,
        "buyback_log": [{"time": 1500}],
        "ability_upgrades_arr": [5003, 5004, 5003],
        "xp_t": [0, 300, 700, 1300, 1900],
    })
    radiant[1].update({"lane": 5, "lane_role": 0, "is_roaming": True})   # dire-jungle code
    dire[0].update({"lane": 3})                                           # dire top = safe
    dire[1].update({"lane_role": 3})                                      # dire off = bottom
    return {
        "match_id": 123,
        "duration": 2400,
        "radiant_win": True,
        "game_mode": 22,
        "start_time": 1700000000,
        "radiant_score": 30,
        "dire_score": 20,
        "players": radiant + dire,
        "teamfights": [{
            "start": 600, "end": 640,
            "players": [
                {"deaths": 1 if i == 5 else 0,
                 "deaths_pos": {"130": {"126": 1}} if i == 5 else {},
                 "killed": {"npc_dota_hero_x": 1} if i == 0 else {},
                 "damage": 100 * i, "healing": 0, "gold_delta": 50,
                 "ability_uses": {"abc": 1}}
                for i in range(10)
            ],
        }],
        "objectives": [{"time": 700, "type": "building_kill", "key": "npc_dota_badguys_tower1_mid"}],
        "picks_bans": [{"is_pick": True, "hero_id": 1, "team": 0, "order": 0}],
        "tower_status_radiant": 2047,
        "tower_status_dire": 2039,
        "patch": 56,
        "region": 3,
    }


def _transform(raw: dict | None = None, ability_ids: dict | None = None, **constants):
    item_map = {1: ("Blink Dagger", "blink")}
    hero_map = {i: {"name": f"Hero{i}", "slug": f"hero{i}"} for i in range(1, 12)}
    return OpenDotaParser()._transform(raw or _fixture(), hero_map, item_map, ability_ids or {}, **constants)


def test_ward_coordinates_are_world_units():
    match = _transform()
    p0 = match.players[0]
    first = p0.wards[1]  # sorted by time: sentry (-30) first
    assert first.ward_type == "observer"
    assert first.x == pytest.approx(geo.opendota_cell_to_world(128)) == 0
    sentry = p0.wards[0]
    assert sentry.ward_type == "sentry"
    assert (sentry.x, sentry.y) == (geo.opendota_cell_to_world(64), geo.opendota_cell_to_world(191))
    for w in match.wards:
        assert 0 <= geo.world_to_pct(w.x) <= 100 and 0 <= geo.world_to_pct(w.y) <= 100
        assert w.team == "radiant"


def test_ward_destroyed_time_by_ehandle_and_position():
    by_time = {w.time: w for w in _transform().players[0].wards}
    assert by_time[100].destroyed_time == 460      # ehandle match
    assert by_time[200].destroyed_time == 260      # positional fallback
    assert by_time[900].destroyed_time is None     # same spot re-warded later: still alive
    assert by_time[-30].destroyed_time is None


def test_purchases_keep_the_time_opendota_logged():
    # This used to assert that early purchases are moved to time -1 as
    # "starting items". On real matches every such move was wrong: the items
    # were bought after the horn, the strategy-time purchases are simply not in
    # OpenDota's log. A purchase keeps its real time.
    items = _transform().players[0].items
    assert [(it.time, it.item_key) for it in items] == [
        (-80, "tango"), (20, "branches"), (25, "branches"), (40, "quelling_blade"), (300, "boots")]
    assert all(it.item_name for it in items)


def test_lane_semantics_follow_contract():
    players = {p.player_slot: p for p in _transform().players}
    assert (players[0].lane, players[0].lane_role) == (1, 1)
    # OpenDota "5" (dire jungle) becomes 4; lane_role derived from the lane
    assert (players[1].lane, players[1].lane_role, players[1].is_roaming) == (4, 4, True)
    # dire: absolute top is their safe lane
    assert (players[128].lane, players[128].lane_role) == (3, 1)
    # dire: off lane is the bottom lane
    assert (players[129].lane, players[129].lane_role) == (1, 3)
    assert (players[130].lane, players[130].lane_role) == (0, 0)


def test_items_inventory_backpack_neutral_and_counters():
    p0 = _transform().players[0]
    # An id that is not in the catalog keeps its slot as a placeholder code
    # (it used to be dropped silently); empty slots are skipped.
    assert p0.final_item_keys == ["blink", "item_999999"]
    assert p0.final_items == ["Blink Dagger", "Item 999999"]
    assert p0.backpack == ["blink"]
    assert p0.neutral_item == "blink"
    assert p0.roshans_killed == 2
    assert p0.buyback_count == 1
    assert p0.net_worth == 15000


def test_unknown_item_ids_resolve_through_the_item_constants():
    raw = _fixture()
    raw["players"][0].update({"backpack_1": 1854, "item_neutral": 1859})
    p0 = _transform(raw, item_ids={"999999": "new_shiny", "1854": "consecrated_wraps", "1859": "spellslinger"}).players[0]
    assert p0.final_item_keys == ["blink", "new_shiny"] and p0.final_items == ["Blink Dagger", "New Shiny"]
    assert p0.backpack == ["blink", "consecrated_wraps"] and p0.neutral_item == "spellslinger"


def test_skill_build_names_and_estimated_times():
    ids = {"5003": "antimage_mana_break", "5004": "antimage_blink"}
    skills = _transform(ability_ids=ids).players[0].skills
    assert [s.ability_key for s in skills] == ["antimage_mana_break", "antimage_blink", "antimage_mana_break"]
    assert [s.level for s in skills] == [1, 1, 2]
    times = [s.time for s in skills]
    assert times == sorted(times) and times[0] == 0
    # level 2 needs 240 XP; xp_t crosses it between minute 0 and 1 (0 -> 300)
    assert times[1] == int(240 / 300 * 60)


def test_skill_build_without_constants_keeps_ids():
    skills = _transform().players[0].skills
    assert skills[0].ability_key == "5003"
    assert skills[0].ability_name == "ability_5003"


def _level_times(xp_t: list[int]) -> dict[int, int]:
    """Second at which a hero with the per-minute XP curve *xp_t* reached each level."""
    times = {1: 0}
    for level, need in enumerate(parser._XP_FOR_LEVEL[1:], start=2):
        minute = next((m for m, xp in enumerate(xp_t) if xp >= need), None)
        if minute is None:
            break
        prev = xp_t[minute - 1]
        times[level] = int((minute - 1 + (need - prev) / (xp_t[minute] - prev)) * 60)
    return times


def test_skill_points_follow_the_rules_of_the_match_patch():
    # 1000 XP a minute: level 10 (5900 XP) at 5:54, level 11 at 7:00, level 12 at 8:12
    xp_t = [1000 * minute for minute in range(40)]
    reached = _level_times(xp_t)
    talent = "special_bonus_unique_lion_5"
    keys = ["lion_impale"] * 10 + [talent, "lion_voodoo", "special_bonus_attributes"]

    # Since 7.40 a talent has its own point: the first one at level 10, and the
    # skill points go on one per level (the 11th and 12th at levels 11 and 12).
    new = parser._estimate_upgrade_times(keys, xp_t, separate_talents=True)
    assert new[9] == new[10] == reached[10]
    assert new[11:] == [reached[11], reached[12]]
    # Before, the talent took the point of level 11 and pushed the rest back a level.
    old = parser._estimate_upgrade_times(keys, xp_t, separate_talents=False)
    assert old[9] == reached[10] and old[10] == reached[11]
    assert old[11:] == [reached[12], reached[13]]
    # the attribute bonus is no talent; later talents come at 15, 20, 25, 27, ...
    assert parser.is_talent(talent) and not parser.is_talent("special_bonus_attributes")
    assert parser._upgrade_levels([talent] * 9, True) == [10, 15, 20, 25, 27, 28, 29, 30, 30]
    # every level up to 22 grants a skill point, none after it (Invoker has 24 orb levels)
    orbs = ["invoker_quas", "invoker_wex", "invoker_exort"] * 8
    assert parser._upgrade_levels(orbs, True) == list(range(1, 23)) + [22, 22]
    assert parser._upgrade_levels(orbs[:19], False) == list(range(1, 17)) + [18, 20, 25]


# Skill builds of match 9037980041 (7.41f) as Valve lists them (OpenDota ability_upgrades_arr).
_DRAGON_KNIGHT = [
    "dragon_knight_breathe_fire", "dragon_knight_breathe_fire", "dragon_knight_dragon_tail",
    "dragon_knight_wyrms_wrath", "dragon_knight_breathe_fire", "dragon_knight_elder_dragon_form",
    "dragon_knight_breathe_fire", "dragon_knight_wyrms_wrath", "dragon_knight_wyrms_wrath",
    "dragon_knight_wyrms_wrath", "special_bonus_unique_dragon_knight_3", "dragon_knight_dragon_tail",
    "dragon_knight_elder_dragon_form", "dragon_knight_dragon_tail", "dragon_knight_dragon_tail",
    "special_bonus_unique_dragon_knight_2", "dragon_knight_elder_dragon_form",
]
_TECHIES = [
    "techies_suicide", "techies_reactive_tazer", "techies_reactive_tazer", "techies_sticky_bomb",
    "techies_reactive_tazer", "techies_land_mines", "techies_reactive_tazer", "techies_suicide",
    "techies_suicide", "special_bonus_magic_resistance_15", "techies_suicide", "special_bonus_attributes",
    "techies_land_mines", "special_bonus_attributes", "special_bonus_attributes",
    "special_bonus_unique_techies_tazer_duration", "special_bonus_attributes", "special_bonus_attributes",
    "special_bonus_attributes", "techies_land_mines",
]
# Winter Wyvern of match 9037747302: the fourth Arctic Burn entry is the level of Eldwurm's Edda.
_WINTER_WYVERN = [
    "winter_wyvern_arctic_burn", "winter_wyvern_cold_embrace", "winter_wyvern_arctic_burn",
    "winter_wyvern_splinter_blast", "winter_wyvern_arctic_burn", "winter_wyvern_arctic_burn",
    "winter_wyvern_winters_curse", "winter_wyvern_splinter_blast", "winter_wyvern_splinter_blast",
    "winter_wyvern_arctic_burn", "special_bonus_attack_damage_30", "winter_wyvern_splinter_blast",
    "winter_wyvern_cold_embrace", "winter_wyvern_winters_curse", "winter_wyvern_cold_embrace",
    "winter_wyvern_cold_embrace", "special_bonus_unique_winter_wyvern_6",
]


def test_third_ultimate_level_is_placed_at_hero_level_18():
    # Dragon Knight had his fourteen ability levels at level 14. Levels 15-17 have nothing to
    # learn: their points go into the attribute bonus by themselves and Valve lists no entry.
    # The fifteenth ability entry is Elder Dragon Form 3 at level 18 (31:26 in the replay;
    # as the fifteenth point it was placed at level 15, seven minutes early).
    assert parser._upgrade_levels(_DRAGON_KNIGHT, True) == (
        list(range(1, 11)) + [10] + [11, 12, 13, 14] + [15] + [18])
    xp_t = [1000 * minute for minute in range(40)]
    reached = _level_times(xp_t)
    times = parser._estimate_upgrade_times(_DRAGON_KNIGHT, xp_t, separate_talents=True)
    assert times[14] == reached[14] and times[16] == reached[18]

    # Techies took six attribute bonuses himself: every point has its entry, and the third
    # level of his ultimate is the 18th of them.
    assert parser._upgrade_levels(_TECHIES, True) == (
        list(range(1, 10)) + [10] + [10, 11, 12, 13, 14] + [15] + [15, 16, 17, 18])

    # A fifth level is not one of the fifteen: with the Edda level her fifteen ability entries
    # are two levels of the ultimate and thirteen others, all learned by level 15 ...
    assert parser._upgrade_levels(_WINTER_WYVERN, True) == list(range(1, 11)) + [10] + [11, 12, 13, 14, 15] + [15]
    # ... and the third level of Winter's Curse still waits for level 18.
    assert parser._upgrade_levels(_WINTER_WYVERN + ["winter_wyvern_winters_curse"], True)[-1] == 18
    # Invoker's fifteenth orb level is no ultimate (level 13 in match 9038370833)
    orbs = ["invoker_exort", "invoker_quas", "invoker_wex"] * 5
    assert parser._upgrade_levels(orbs, True) == list(range(1, 16))
    # without the id table talents are unknown keys like the abilities (six keys, not four):
    # nothing says which entry completes the fifteen, the entries keep their places
    unnamed = [f"ability_{5000 + n % 6}" for n in range(17)]
    assert parser._upgrade_levels(unnamed, True) == list(range(1, 18))
    # before 7.40 a talent took the point of its level: the 17th entry came with level 18 anyway
    assert parser._upgrade_levels(_DRAGON_KNIGHT, False)[-1] == 18


def test_the_skill_rules_are_chosen_by_patch_then_by_start_time():
    assert parser.talents_have_own_points("7.40") and parser.talents_have_own_points("7.41f")
    assert not parser.talents_have_own_points("7.39e") and not parser.talents_have_own_points("7.37")
    assert parser.talents_have_own_points("", 1765785600)           # 7.40 was released then
    assert not parser.talents_have_own_points("", 1765785599)
    assert not parser.talents_have_own_points("7.39", 1791622274)   # a known patch decides
    assert parser.talents_have_own_points()                         # nothing known: today's rules

    ids = {"5003": "antimage_mana_break", "5004": "special_bonus_unique_antimage"}
    raw = _fixture()                                                # started 2023-11-14: 7.34
    raw["players"][0]["ability_upgrades_arr"] = [5003] * 10 + [5004, 5003]
    raw["players"][0]["xp_t"] = [1000 * minute for minute in range(40)]
    raw["duration"] = 39 * 60
    reached = _level_times(raw["players"][0]["xp_t"])
    old = [s.time for s in _transform(copy.deepcopy(raw), ability_ids=ids).players[0].skills]
    assert old[10:] == [reached[11], reached[12]]
    raw["start_time"] = 1791622274                                  # 2026-10-10: 7.41f
    new = [s.time for s in _transform(copy.deepcopy(raw), ability_ids=ids).players[0].skills]
    assert new[10:] == [reached[10], reached[11]]
    # the patch table names the version of the match: it wins over the start time
    table = [{"id": 56, "name": "7.37"}]
    assert [s.time for s in _transform(copy.deepcopy(raw), ability_ids=ids, patches=table).players[0].skills] == old


def test_teamfight_kills_positions_and_teams():
    tf = _transform().teamfights[0]
    assert tf.dire_deaths == 1 and tf.radiant_deaths == 0
    assert tf.players[0].kills == 1                       # from the "killed" dict
    victim = tf.players[5]
    assert victim.player_slot == 128
    assert (victim.x, victim.y) == (geo.opendota_cell_to_world(130), geo.opendota_cell_to_world(126))
    # others are placed at the fight centre, not (0, 0)
    assert (tf.players[1].x, tf.players[1].y) == (victim.x, victim.y)
    assert tf.gold_delta == 0  # 5 * 50 - 5 * 50


def test_building_objectives_get_teams_from_the_building():
    obj = _transform().objectives[0]
    assert obj["building_team"] == "dire" and obj["team"] == "radiant"


def test_match_level_fields():
    match = _transform()
    assert match.game_mode == "ranked_all_pick"
    # OpenDota's numeric patch id is not a version: resolved through the
    # patch constants, or left empty (it used to be passed through as "56").
    assert match.patch == "" and match.region == 3
    named = _transform(patches=[{"id": 55, "name": "7.36"}, {"id": 56, "name": "7.37"}])
    assert named.patch == "7.37"
    assert match.winner_known is True and match.radiant_win is True
    assert match.tower_status_dire == 2039
    assert match.buildings == [] and match.partial is False
    assert len(match.gold_xp_timeline) == 3
    assert match.gold_xp_timeline[1].radiant_gold == 5 * 600


def test_transform_tolerates_missing_and_malformed_data():
    raw = {"match_id": 1, "players": [{"player_slot": 0, "hero_id": None, "kills": None, "purchase_log": None,
                                       "obs_log": [None, {"time": "5", "x": "128", "y": 128}]}]}
    match = _transform(raw)
    assert match.players[0].kills == 0
    assert match.players[0].wards[0].time == 5
    assert match.gold_xp_timeline == []
    empty = _transform({"players": []})
    assert empty.players == [] and empty.game_mode == "unknown"


# ---------------------------------------------------------------------------
# API errors
# ---------------------------------------------------------------------------

def _raise(exc):
    def _urlopen(*_args, **_kwargs):
        raise exc
    return _urlopen


@pytest.mark.parametrize("status,expected", [(404, 404), (429, 429), (503, 503), (400, 400)])
def test_http_errors_map_to_opendota_error(monkeypatch, status, expected):
    err = urllib.error.HTTPError("https://api.opendota.com", status, "x", {}, io.BytesIO(b""))
    monkeypatch.setattr(parser.urllib.request, "urlopen", _raise(err))
    with pytest.raises(OpenDotaError) as info:
        OpenDotaParser()._api_get("/matches/1")
    assert info.value.status == expected
    assert "api.opendota.com" not in str(info.value)


def test_network_errors_have_no_status(monkeypatch):
    monkeypatch.setattr(parser.urllib.request, "urlopen", _raise(urllib.error.URLError("dns failure")))
    with pytest.raises(OpenDotaError) as info:
        OpenDotaParser()._api_get("/matches/1")
    assert info.value.status is None
    assert "dns" not in str(info.value)


def test_parse_rejects_invalid_ids_and_error_payloads(monkeypatch):
    with pytest.raises(OpenDotaError) as info:
        asyncio.run(OpenDotaParser().parse("abc"))
    assert info.value.status == 400

    monkeypatch.setattr(OpenDotaParser, "_api_get", lambda self, path: {"error": "Not Found"})
    with pytest.raises(OpenDotaError) as info:
        asyncio.run(OpenDotaParser().parse("42"))
    assert info.value.status == 404


@pytest.fixture
def constants_cache():
    parser.reset_opendota_constants_cache()
    yield
    parser.reset_opendota_constants_cache()


def _stub_match(monkeypatch, raw: dict, constants) -> list[str]:
    """Serve *raw* as the match and the constants through *constants*(name); returns the call log."""
    calls: list[str] = []

    def fake_get(self, path):
        calls.append(path)
        return json.loads(json.dumps(raw))

    def fake_download(self, name):
        calls.append(f"constants:{name}")
        return constants(name)

    monkeypatch.setattr(OpenDotaParser, "_api_get", fake_get)
    monkeypatch.setattr(OpenDotaParser, "_download_constants", fake_download)
    return calls


def test_parse_downloads_only_the_ids_the_catalog_lacks(monkeypatch, constants_cache):
    raw = _fixture()
    tables = {"ability_ids": {"99990001": "brand_new_ability"}, "item_ids": {"999999": "new_shiny"},
              "patch": [{"id": 56, "name": "7.37"}]}
    calls = _stub_match(monkeypatch, raw, tables.__getitem__)
    match = asyncio.run(OpenDotaParser().parse("123"))
    # ability ids 5003 / 5004 are in the bundled catalog: no download for them. The
    # item table only because the fixture has an inventory id the catalog lacks.
    assert calls[0] == "/matches/123" and sorted(calls[1:]) == ["constants:item_ids", "constants:patch"]
    assert [s.ability_key for s in match.players[0].skills] == [
        "antimage_mana_break", "antimage_blink", "antimage_mana_break"]
    assert match.players[0].final_item_keys == ["blink", "new_shiny"] and match.patch == "7.37"

    # an ability the game lists under two ids (the catalog entry's `ids`): both resolve, no download
    assert parser._ability_id_map()["3060"] == parser._ability_id_map()["1617"] == "templar_assassin_hidden_gates"
    raw["players"][0]["ability_upgrades_arr"] = [5003, 1617]
    match = asyncio.run(OpenDotaParser().parse("123"))
    assert [s.ability_key for s in match.players[0].skills] == ["antimage_mana_break", "templar_assassin_hidden_gates"]
    assert "constants:ability_ids" not in calls

    # an ability the catalog does not know yet: the id table is downloaded, once per process
    raw["players"][0]["ability_upgrades_arr"] = [5003, 99990001]
    for _ in range(2):
        match = asyncio.run(OpenDotaParser().parse("123"))
    assert [s.ability_key for s in match.players[0].skills] == ["antimage_mana_break", "brand_new_ability"]
    assert Counter(calls) == {"/matches/123": 4, "constants:ability_ids": 1, "constants:item_ids": 1,
                              "constants:patch": 1}


def test_parse_survives_missing_constants(monkeypatch, constants_cache):
    raw = _fixture()
    raw["players"][0]["ability_upgrades_arr"] = [5003, 99990001]

    def unavailable(name):
        raise OSError("no route to host")

    calls = _stub_match(monkeypatch, raw, unavailable)
    match = asyncio.run(OpenDotaParser().parse("123"))
    # what the bundled catalog knows is still named; the rest keeps its id
    assert [s.ability_key for s in match.players[0].skills] == ["antimage_mana_break", "99990001"]
    assert match.patch == "" and match.players[0].final_item_keys == ["blink", "item_999999"]
    # a failed download is not retried on the next parse
    asyncio.run(OpenDotaParser().parse("123"))
    assert Counter(calls)["constants:patch"] == 1 and Counter(calls)["constants:ability_ids"] == 1


def test_parse_downloads_its_tables_side_by_side(monkeypatch, constants_cache):
    """Three tables one after another meant three waits for a source that times out
    (10 s each, per source): they are fetched at the same time."""
    raw = _fixture()
    raw["players"][0]["ability_upgrades_arr"] = [5003, 99990001]
    tables = {"ability_ids": {"99990001": "brand_new_ability"}, "item_ids": {"999999": "new_shiny"},
              "patch": [{"id": 56, "name": "7.37"}]}
    together = threading.Barrier(3, timeout=5)

    def constants(name):
        together.wait()  # raises unless all three downloads are under way at once
        return tables[name]

    _stub_match(monkeypatch, raw, constants)
    match = asyncio.run(OpenDotaParser().parse("123"))
    assert match.patch == "7.37" and match.players[0].final_item_keys == ["blink", "new_shiny"]
    assert match.players[0].skills[1].ability_key == "brand_new_ability"


def test_parse_asks_the_stored_patch_table_before_the_network(monkeypatch, constants_cache):
    """The game-data store usually names the version of the match's start time
    (OpenDota derives its patch id from that time as well): no download then."""
    from backend.app.ai import data_manager

    raw = _fixture()
    started = raw["start_time"]
    monkeypatch.setattr(data_manager, "patch_for_time", lambda when: "7.34" if when == started else "")
    calls = _stub_match(monkeypatch, raw, {"item_ids": {"999999": "new_shiny"}}.__getitem__)
    assert asyncio.run(OpenDotaParser().parse("123")).patch == "7.34"
    assert calls == ["/matches/123", "constants:item_ids"]

    # a start time the store cannot name (no table yet, or a table too old for it): the download
    raw["start_time"] += 86400
    calls = _stub_match(monkeypatch, raw, {"item_ids": {}, "patch": [{"id": 56, "name": "7.37"}]}.__getitem__)
    assert asyncio.run(OpenDotaParser().parse("123")).patch == "7.37"
    assert "constants:patch" in calls


def test_constants_come_from_the_data_layers_sources(monkeypatch, constants_cache):
    from backend.app.ai import data_manager

    asked = []

    def fake_download(name, timeout=30):
        asked.append((name, timeout))
        return {"1": "blink"}

    monkeypatch.setattr(data_manager, "download_constants", fake_download)
    monkeypatch.setattr(OpenDotaParser, "_api_get", lambda self, path: pytest.fail("the API is the data layer's last source"))
    assert OpenDotaParser()._load_item_ids() == {"1": "blink"}
    assert asked == [("item_ids", OpenDotaParser.CONSTANTS_TIMEOUT_S)]
    # ... in its order: the current master branch before OpenDota's months-old release
    assert [source for source, _url in data_manager.CONSTANTS_SOURCES][-1] == "opendota_api"

    # without the data layer the parser still has OpenDota's own copy
    parser.reset_opendota_constants_cache()
    monkeypatch.setattr(parser, "_data_layer", lambda function: None)
    monkeypatch.setattr(OpenDotaParser, "_api_get", lambda self, path: {"2": "blades_of_attack", "path": path})
    assert OpenDotaParser()._load_item_ids() == {"2": "blades_of_attack", "path": "/constants/item_ids"}


def test_patch_falls_back_to_the_stored_table_by_start_time(monkeypatch):
    from backend.app.ai import data_manager

    monkeypatch.setattr(data_manager, "patch_for_time", lambda when: "7.34" if when == 1700000000 else "")
    assert _transform().patch == "7.34"                                    # no patch table: by time
    assert _transform(patches=[{"id": 56, "name": "7.37"}]).patch == "7.37"  # the table decides
    raw = _fixture()
    raw["patch"] = None
    assert _transform(raw).patch == ""                                     # OpenDota named no patch


def test_api_key_is_appended(monkeypatch):
    monkeypatch.setenv("OPENDOTA_API_KEY", "k&y")
    assert OpenDotaParser()._url("/matches/1").endswith("/matches/1?api_key=k%26y")
    monkeypatch.delenv("OPENDOTA_API_KEY")
    assert OpenDotaParser()._url("/matches/1").endswith("/matches/1")


# ---------------------------------------------------------------------------
# Contract shapes shared with the replay parser
# ---------------------------------------------------------------------------

def test_wards_without_xy_use_the_key_or_are_skipped():
    raw = _fixture()
    raw["players"][0].update({
        "obs_log": [
            {"time": 100, "key": "[120, 130]", "ehandle": 1},           # older logs: position only in key
            {"time": 200, "key": [100.5, 150.25]},
            {"time": 300},                                              # no position at all
            {"time": 400, "x": None, "y": 140, "key": "not json"},
        ],
        "obs_left_log": [{"time": 460, "key": "[120, 130]", "ehandle": 1}],
        "sen_log": [], "sen_left_log": [],
    })
    wards = _transform(raw).players[0].wards
    assert [(w.time, w.x, w.y) for w in wards] == [
        (100, geo.opendota_cell_to_world(120), geo.opendota_cell_to_world(130)),
        (200, geo.opendota_cell_to_world(100.5), geo.opendota_cell_to_world(150.25)),
    ]
    assert wards[0].destroyed_time == 460
    for w in wards:
        assert 0 <= geo.world_to_pct(w.x) <= 100 and 0 <= geo.world_to_pct(w.y) <= 100


def test_dewarded_comes_from_the_attacker_of_the_left_log():
    raw = _fixture()
    raw["duration"] = 2400
    raw["players"][0].update({
        "obs_log": [{"time": t, "x": 100 + i, "y": 100, "ehandle": 10 + i} for i, t in enumerate((100, 200, 300, 400, 2300))],
        "obs_left_log": [
            {"time": 460, "ehandle": 10, "attackername": "npc_dota_hero_hero1"},        # own hero: expired
            {"time": 559, "ehandle": 11, "attackername": "npc_dota_hero_hero7"},        # enemy, last second
            {"time": 350, "ehandle": 12, "attackername": "npc_dota_creep_badguys_melee"},
            {"time": 500, "ehandle": 13},                                               # no attacker: lifetime
            {"time": 2660, "ehandle": 14, "attackername": "npc_dota_hero_hero1"},       # after the game ended
        ],
        "sen_log": [], "sen_left_log": [],
    })
    wards = {w.time: w for w in _transform(raw).players[0].wards}
    assert [(t, wards[t].dewarded) for t in (100, 200, 300, 400)] == [(100, False), (200, True), (300, True), (400, True)]
    assert wards[2300].destroyed_time is None and wards[2300].dewarded is False


def test_kills_log_teamfight_offsets_and_kill_targets():
    raw = _fixture()
    players = raw["players"]
    players[0].update({"kills": 2, "kills_log": [{"time": 618, "key": "npc_dota_hero_hero6"},
                                                 {"time": 900, "key": "npc_dota_hero_hero7"},
                                                 {"time": 950, "key": "npc_dota_hero_hero2"},     # an ally: deny
                                                 {"time": 960, "key": "npc_dota_hero_other"}]})   # not in the match
    players[5].update({"deaths": 1, "kills": 1, "kills_log": [{"time": 625, "key": "npc_dota_hero_hero1"}]})
    players[6].update({"deaths": 1})
    players[1].update({"deaths": 1})
    match = _transform(raw)
    assert [(k.time, k.killer_slot, k.victim_slot, k.deny, k.confirmed) for k in match.kills_log] == [
        (618.0, 0, 128, False, True), (625.0, 128, 0, False, True),
        (900.0, 0, 129, False, True), (950.0, 0, 1, True, True),
    ]
    # the only death of the fight with a deaths_pos cell gets that position
    assert (match.kills_log[0].x, match.kills_log[0].y) == (geo.opendota_cell_to_world(130), geo.opendota_cell_to_world(126))
    assert match.kills_log[2].x is None
    fight = match.teamfights[0]                           # 600-640, deaths between 615 and 625
    killer, victim = fight.players[0], fight.players[5]
    assert killer.kill_targets == [{"target": "Hero6", "target_slot": 128, "time_offset": 18.0}]
    assert victim.death_time_offset == 18.0 and victim.kill_targets == [
        {"target": "Hero1", "target_slot": 0, "time_offset": 25.0}]
    # a death the per-player teamfight count misses is taken from the log
    assert killer.death_time_offset == 25.0 and killer.deaths == 1
    assert (fight.radiant_deaths, fight.dire_deaths) == (1, 1)
    assert all(tp.death_time_offset is None for tp in fight.players[1:5])


def test_unit_dicts_chat_and_first_blood_follow_the_contract():
    raw = _fixture()
    raw["players"][0].update({
        "killed": {"npc_dota_hero_hero6": 2, "npc_dota_creep_badguys_melee": 80, "illusion_npc_dota_hero_hero7": 1},
        "killed_by": {"npc_dota_hero_hero7": 1, "npc_dota_badguys_tower1_mid": 1},
        "damage": {"npc_dota_hero_hero6": 4000, "npc_dota_creep_badguys_melee": 900,
                   "illusion_npc_dota_hero_hero7": 250},
        "damage_taken": {"npc_dota_hero_hero7": 3000, "dota_fountain": 600},
        "damage_inflictor": {"null": 1500, "hero1_spell": 2500},
        "dn_t": [0, 1, 3], "lane_efficiency": 0.61, "lane_efficiency_pct": 61, "personaname": None,
    })
    raw["chat"] = [{"time": 5, "type": "chatwheel", "key": "71", "slot": 0, "player_slot": 0},
                   {"time": 6, "type": "chatwheel", "key": "chatwheel_9", "slot": 0, "player_slot": 0},
                   {"time": 7, "type": "chat", "key": "71", "slot": 0, "player_slot": 0}]
    raw["objectives"] = [
        {"time": 138, "type": "CHAT_MESSAGE_FIRSTBLOOD", "key": "5", "slot": 4, "player_slot": 4,
         "victim_player_slot": 128},
        {"time": 139, "type": "CHAT_MESSAGE_FIRSTBLOOD", "key": 6, "slot": 4, "player_slot": 4},   # older form
    ]
    match = _transform(raw)
    p0 = match.players[0]
    assert p0.killed == {"Hero6": 2} and p0.killed_by == {"Hero7": 1}          # heroes of the match only
    # other units keep OpenDota's name: an illusion is not its hero, the fountain is "dota_fountain"
    assert p0.damage == {"Hero6": 4000, "npc_dota_creep_badguys_melee": 900, "illusion_npc_dota_hero_hero7": 250}
    assert p0.damage_taken == {"Hero7": 3000, "dota_fountain": 600}
    assert p0.damage_inflictor == {"attack": 1500, "hero1_spell": 2500}
    assert p0.dn_t == [0, 1, 3] and (p0.lane_efficiency, p0.lane_efficiency_pct) == (0.61, 61)
    assert p0.player_name == "" and match.players[1].player_name == "p1"       # no English placeholder
    assert [m["key"] for m in match.chat] == ["chatwheel_71", "chatwheel_9", "71"]
    assert [(o["player_slot"], o["key"]) for o in match.objectives] == [(4, "128"), (4, "129")]


def test_unknown_winner_and_hero_codes():
    raw = _fixture()
    raw["radiant_win"] = None
    raw["players"][0]["hero_id"] = 999
    raw["players"][1]["hero_id"] = None
    match = _transform(raw)
    assert match.winner_known is False and match.radiant_win is False
    assert (match.players[0].hero_name, match.players[1].hero_name) == ("hero_999", "unknown_1")
    # ... and the same code wherever the hero is named
    fighters = match.teamfights[0].players
    assert [(tp.hero_name, tp.hero_slug) for tp in fighters] == [(p.hero_name, p.hero_slug) for p in match.players]
    assert [tp.hero_name for tp in fighters[:3]] == ["hero_999", "unknown_1", "Hero3"]


def test_purchase_log_and_items_are_the_same_list_in_time_order():
    raw = _fixture()
    raw["players"][0]["purchase_log"] = [
        {"time": 30, "key": "tango", "charges": 3}, {"time": -80, "key": "ring_of_regen"},
        {"time": -79, "key": "sobi_mask"}, {"time": -78, "key": "wind_lace"},
    ]
    p0 = _transform(raw).players[0]
    assert [(it.time, it.item_key) for it in p0.items] == [
        (-80, "ring_of_regen"), (-79, "sobi_mask"), (-78, "wind_lace"), (30, "tango")]
    assert p0.purchase_log == [{"time": -80, "key": "ring_of_regen"}, {"time": -79, "key": "sobi_mask"},
                               {"time": -78, "key": "wind_lace"}, {"time": 30, "key": "tango", "charges": 3}]


def test_objective_teams_are_codes_and_the_courier_killer_is_a_slot():
    raw = _fixture()
    raw["objectives"] = [
        {"time": 648, "type": "CHAT_MESSAGE_COURIER_LOST", "team": 3, "value": 45, "killer": 6},
        {"time": 700, "type": "CHAT_MESSAGE_COURIER_LOST", "team": 2, "value": 50, "killer": -1},
        {"time": 1878, "type": "CHAT_MESSAGE_ROSHAN_KILL", "team": 3},
        {"time": 1900, "type": "CHAT_MESSAGE_ROSHAN_KILL", "team": 9},
    ]
    assert _transform(raw).objectives == [
        # the Dire courier, killed by the player with index 6 (slot 129)
        {"time": 648, "type": "CHAT_MESSAGE_COURIER_LOST", "team": "dire", "value": 45, "player_slot": 129},
        {"time": 700, "type": "CHAT_MESSAGE_COURIER_LOST", "team": "radiant", "value": 50},
        {"time": 1878, "type": "CHAT_MESSAGE_ROSHAN_KILL", "team": "dire"},
        {"time": 1900, "type": "CHAT_MESSAGE_ROSHAN_KILL"},
    ]


def test_counters_follow_the_replay_parser_definitions():
    raw = _fixture()
    raw["radiant_score"] = 12
    raw["players"][0].update({
        # enemy lane creeps incl. siege; OpenDota's own number adds denies and skips siege
        "lane_kills": 61,
        "killed": {"npc_dota_creep_badguys_melee": 40, "npc_dota_creep_badguys_ranged_upgraded": 9,
                   "npc_dota_badguys_siege": 3, "npc_dota_creep_goodguys_melee": 12,
                   "npc_dota_neutral_kobold": 7, "npc_dota_hero_hero6": 2},
        "obs_placed": 9, "sen_placed": 0,                 # the logs say 3 and 1
        "damage": {"npc_dota_hero_hero1": 1970, "npc_dota_hero_hero6": 4000},
        "damage_taken": {"npc_dota_hero_hero1": 1970, "npc_dota_hero_hero7": 3000},
        "teamfight_participation": 0.25,
        "buyback_log": [{"time": 1500, "slot": 0, "type": "buyback_log", "player_slot": 0}],
    })
    raw["players"][1].update({"lane_kills": 33, "obs_placed": 4, "sen_placed": 2, "kills": 4, "assists": 2})
    raw["players"][1].update({"observer_kills": 2, "sentry_kills": 5})
    p0, p1 = _transform(raw).players[:2]
    assert (p1.observer_kills, p1.sentry_kills) == (2, 5)  # no attackers in the left logs: the counters
    assert p0.lane_kills == 52 and p1.lane_kills == 33     # no breakdown: OpenDota's number
    assert (p0.obs_placed, p0.sen_placed) == (3, 1) == (
        sum(w.ward_type == "observer" for w in p0.wards), sum(w.ward_type == "sentry" for w in p0.wards))
    assert (p1.obs_placed, p1.sen_placed) == (4, 2)        # no logs: the summary counters
    assert p0.damage == {"Hero6": 4000} and p0.damage_taken == {"Hero7": 3000}   # nothing done to itself
    assert p0.teamfight_participation == 0.25
    assert p1.teamfight_participation == pytest.approx((4 + 2) / 12)            # not parsed: from the score
    assert p0.buyback_log == [{"time": 1500}] and p0.buyback_count == 1
    fight = _transform(raw).teamfights[0]
    assert fight.gold_delta == 0 and fight.xp_delta == 0


def test_dewarding_hero_and_teamfight_xp_delta():
    raw = _fixture()
    raw["players"][0].update({
        "obs_log": [{"time": 100, "x": 100, "y": 100, "ehandle": 10}, {"time": 200, "x": 101, "y": 100, "ehandle": 11},
                    {"time": 300, "x": 102, "y": 100, "ehandle": 12}],
        "obs_left_log": [
            {"time": 150, "ehandle": 10, "attackername": "npc_dota_hero_hero7"},            # enemy hero
            {"time": 250, "ehandle": 11, "attackername": "npc_dota_creep_badguys_melee"},   # enemy creep
            {"time": 660, "ehandle": 12, "attackername": "npc_dota_hero_hero1"},            # expired
        ],
        "sen_log": [], "sen_left_log": [],
    })
    raw["players"][6].update({"observer_kills": 3, "sentry_kills": 2})
    for i, tp in enumerate(raw["teamfights"][0]["players"]):
        tp["xp_delta"] = 100 if i < 5 else 30
    match = _transform(raw)
    wards = {w.time: w for w in match.players[0].wards}
    assert [(wards[t].dewarded, wards[t].killer_slot) for t in (100, 200, 300)] == [
        (True, 129), (True, None), (False, None)]
    assert match.teamfights[0].xp_delta == 5 * 100 - 5 * 30
    # ward kills are the enemy wards a hero removed: OpenDota's own counter
    # (here 3 for hero 7) also has a deny of an own ward
    assert (match.players[6].observer_kills, match.players[6].sentry_kills) == (1, 0)
    assert (match.players[0].observer_kills, match.players[0].sentry_kills) == (0, 0)


# ---------------------------------------------------------------------------
# Real API response (match 8762775364)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_raw():
    return json.loads((FIXTURES / "opendota_8762775364.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def constants():
    return json.loads((FIXTURES / "opendota_constants.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def real(real_raw, constants):
    return OpenDotaParser()._transform(
        copy.deepcopy(real_raw), parser.load_hero_map(), parser._item_id_map(),
        constants["ability_ids"], constants["item_ids"], constants["patch"])


def test_real_kills_log_reconciles_reincarnations(real, real_raw):
    by_hero = {p.hero_name: p for p in real.players}
    slot = {p.hero_name: p.player_slot for p in real.players}
    assert sum(len(p["kills_log"]) for p in real_raw["players"]) == 75
    # Wraith King: 8 logged "deaths", 3 on the scoreboard. Hoodwink (7 entries,
    # 5 kills) and Ogre Magi (1 entry, 0 kills) can only have hit reincarnations.
    assert by_hero["Wraith King"].deaths == 3
    assert len(real.kills_log) == 72
    on_wk = [k for k in real.kills_log if k.victim_slot == slot["Wraith King"]]
    assert {k.killer_slot for k in on_wk} == {slot["Invoker"], slot["Legion Commander"]}
    # Legion Commander's kill is certain; two of Invoker's four are real, which two is unknown.
    assert [(k.time, k.confirmed) for k in on_wk] == [
        (290.0, False), (385.0, True), (820.0, False), (1241.0, False), (1711.0, False)]
    assert all(k.confirmed for k in real.kills_log if k.victim_slot != slot["Wraith King"])
    for p in real.players:
        if p.hero_name in ("Wraith King", "Invoker"):
            continue
        assert sum(1 for k in real.kills_log if k.victim_slot == p.player_slot) == p.deaths, p.hero_name
        assert sum(1 for k in real.kills_log if k.killer_slot == p.player_slot and not k.deny) == p.kills, p.hero_name
    assert [k.time for k in real.kills_log] == sorted(k.time for k in real.kills_log)
    # the proven reincarnations also leave the kill counts
    assert "Wraith King" not in by_hero["Hoodwink"].killed and by_hero["Ogre Magi"].killed == {}
    # Invoker has two entries more than kills and they can only be on Wraith
    # King: the counts are exact although the two entries are not identified.
    # (This used to expect 4, i.e. two reincarnations counted as kills.)
    assert by_hero["Wraith King"].killed_by == {"Legion Commander": 1, "Invoker": 2}
    assert by_hero["Invoker"].killed == {"Void Spirit": 2, "Zeus": 2, "Lion": 1, "Wraith King": 2}
    for p in real.players:
        assert sum(p.killed.values()) == p.kills, p.hero_name
        assert sum(p.killed_by.values()) <= p.deaths, p.hero_name


def test_real_teamfights_have_death_offsets_and_kill_targets(real):
    for fight in real.teamfights:
        for tp in fight.players:
            assert (tp.deaths > 0) == (tp.death_time_offset is not None), (fight.start_time, tp.hero_name)
            assert tp.kills == len(tp.kill_targets), (fight.start_time, tp.hero_name)
            for target in tp.kill_targets:
                assert set(target) == {"target", "target_slot", "time_offset"}
                assert 0 <= target["time_offset"] <= fight.end_time - fight.start_time
        assert fight.radiant_deaths + fight.dire_deaths == sum(tp.deaths for tp in fight.players)
    first = real.teamfights[0]                             # 123-166: first blood on Lion at 138
    hoodwink = next(tp for tp in first.players if tp.hero_name == "Hoodwink")
    lion = next(tp for tp in first.players if tp.hero_name == "Lion")
    assert hoodwink.kill_targets == [{"target": "Lion", "target_slot": 128, "time_offset": 15.0}]
    assert lion.death_time_offset == 15.0
    # the fight at 699-746 contained a Wraith King reincarnation (by Hoodwink)
    reincarnation_fight = next(tf for tf in real.teamfights if tf.start_time == 699)
    wraith_king = next(tp for tp in reincarnation_fight.players if tp.hero_name == "Wraith King")
    assert wraith_king.deaths == 0 and wraith_king.death_time_offset is None
    assert sum(1 for k in real.kills_log if k.x is not None) >= 40      # positions from deaths_pos


def test_real_items_wards_and_match_fields(real, real_raw):
    by_hero = {p.hero_name: p for p in real.players}
    # ids 1854 / 1859 / 1873 are 7.41 items (missing from catalogs generated before it)
    assert len(by_hero["Tidehunter"].final_item_keys) == 6 and "consecrated_wraps" in by_hero["Tidehunter"].final_item_keys
    assert (by_hero["Invoker"].neutral_item, by_hero["Zeus"].neutral_item) == ("spellslinger", "partisans_brand")
    assert by_hero["Invoker"].dn_t[10] == 12 and len(by_hero["Invoker"].dn_t) == real.duration // 60 + 1
    assert "attack" in by_hero["Invoker"].damage_inflictor and "null" not in by_hero["Invoker"].damage_inflictor
    assert (real.patch, real.cluster, real.region, real.match_id) == ("7.41", 151, 5, 8762775364)
    assert {m["key"] for m in real.chat if m["type"] == "chatwheel"} == {"chatwheel_71"}
    first_blood = next(o for o in real.objectives if o["type"] == "CHAT_MESSAGE_FIRSTBLOOD")
    assert (first_blood["player_slot"], first_blood["key"]) == (4, "128")
    assert any(w.dewarded for w in real.wards) and any(w.destroyed_time and not w.dewarded for w in real.wards)
    for w in real.wards:
        assert not w.dewarded or w.destroyed_time is not None
    # with the bundled catalog alone the 7.41 items are named, the patch is not
    bare = OpenDotaParser()._transform(copy.deepcopy(real_raw), parser.load_hero_map(), parser._item_id_map())
    tide = next(p for p in bare.players if p.hero_name == "Tidehunter")
    assert "consecrated_wraps" in tide.final_item_keys and bare.patch == ""
    # an id no table has stays in its slot as a placeholder code
    older = {item_id: entry for item_id, entry in parser._item_id_map().items() if item_id != 1854}
    bare = OpenDotaParser()._transform(copy.deepcopy(real_raw), parser.load_hero_map(), older)
    tide = next(p for p in bare.players if p.hero_name == "Tidehunter")
    assert "item_1854" in tide.final_item_keys and len(tide.final_item_keys) == 6
