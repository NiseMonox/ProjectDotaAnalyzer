"""OpenDotaParser: transform of a synthetic match payload and API error mapping.

No network: urlopen / _api_get are monkeypatched.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import urllib.error

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import pytest  # noqa: E402

from backend.app import geo, parser  # noqa: E402
from backend.app.parser import OpenDotaError, OpenDotaParser  # noqa: E402


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


def _transform(raw: dict | None = None, ability_ids: dict | None = None):
    item_map = {1: ("Blink Dagger", "blink")}
    hero_map = {i: {"name": f"Hero{i}", "slug": f"hero{i}"} for i in range(1, 12)}
    return OpenDotaParser()._transform(raw or _fixture(), hero_map, item_map, ability_ids or {})


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


def test_starting_items_are_inferred_from_early_purchases():
    items = _transform().players[0].items
    starting = sorted(it.item_key for it in items if it.time < 0)
    assert starting == ["branches", "branches", "quelling_blade", "tango"]
    assert [it.item_key for it in items if it.time > 0] == ["boots"]
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
    assert p0.final_item_keys == ["blink"]          # empty / unknown ids skipped
    assert p0.final_items == ["Blink Dagger"]
    assert p0.backpack == ["blink"]
    assert p0.neutral_item == "blink"
    assert p0.roshans_killed == 2
    assert p0.buyback_count == 1
    assert p0.net_worth == 15000


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
    assert match.patch == "56" and match.region == 3
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


def test_parse_uses_ability_constants(monkeypatch):
    raw = _fixture()
    calls = []

    def fake_get(self, path):
        calls.append(path)
        if path == "/constants/ability_ids":
            return {"5003": "antimage_mana_break", "5004": "antimage_blink"}
        return json.loads(json.dumps(raw))

    monkeypatch.setattr(OpenDotaParser, "_api_get", fake_get)
    monkeypatch.setattr(parser, "_ability_ids", None)
    monkeypatch.setattr(parser, "_ability_ids_failed_at", 0.0)
    match = asyncio.run(OpenDotaParser().parse("123"))
    assert calls == ["/matches/123", "/constants/ability_ids"]
    assert match.players[0].skills[0].ability_key == "antimage_mana_break"
    # cached for the process
    asyncio.run(OpenDotaParser().parse("123"))
    assert calls.count("/constants/ability_ids") == 1


def test_api_key_is_appended(monkeypatch):
    monkeypatch.setenv("OPENDOTA_API_KEY", "k&y")
    assert OpenDotaParser()._url("/matches/1").endswith("/matches/1?api_key=k%26y")
    monkeypatch.delenv("OPENDOTA_API_KEY")
    assert OpenDotaParser()._url("/matches/1").endswith("/matches/1")
