"""Tests for backend.app.analyzer: edge cases, team attribution and C7 codes."""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import json
import re

import pytest

from backend.app import analyzer
from backend.app.analyzer import full_analysis
from backend.app.models import (
    LaningCreepKill,
    LaningSnapshot,
    MatchData,
    PlayerData,
    SmokeEvent,
    TeamfightEvent,
    TeamfightPlayer,
    WardEvent,
)

CJK = re.compile(r"[一-鿿]")


def make_player(slot: int, **kw) -> PlayerData:
    team = "radiant" if slot < 128 else "dire"
    base = dict(
        player_slot=slot, hero_id=slot + 1, hero_name=f"Hero{slot}", hero_slug=f"hero{slot}",
        player_name=f"p{slot}", team=team, kills=0, deaths=0, assists=0, level=1,
        last_hits=0, denies=0, gold_per_min=0, xp_per_min=0, hero_damage=0,
        tower_damage=0, hero_healing=0, net_worth=0, items=[], backpack=[],
        final_items=[], skills=[], wards=[], ability_events=[],
    )
    base.update(kw)
    return PlayerData(**base)


def make_match(players=None, **kw) -> MatchData:
    base = dict(
        match_id=42, duration=1800, radiant_win=True, game_mode="All Pick",
        start_time=0, radiant_score=0, dire_score=0,
        players=players if players is not None else [], wards=[], smokes=[],
        teamfights=[], ability_events=[],
    )
    base.update(kw)
    return MatchData(**base)


def test_empty_match_does_not_crash():
    result = full_analysis(make_match(duration=0))
    assert result["match_overview"]["players"] == []
    assert result["combat"]["kill_matrix"]["matrix"] == []
    assert result["wards"]["radiant"]["avg_obs_lifetime"] == 0
    assert result["smokes"]["radiant"]["success_rate"] == 0
    assert result["laning"] == {"players": []}
    json.dumps(result)  # JSON-serializable


def test_format_time_handles_floats_negatives_and_none():
    assert analyzer.format_time(65.7) == "1:05"
    assert analyzer.format_time(-90) == "-1:30"
    assert analyzer.format_time(None) == "0:00"


def test_mock_payload_has_no_localized_labels():
    from backend.app.mock_data import generate_mock_match

    result = full_analysis(generate_mock_match())
    for section in ("laning", "farm", "objectives_analysis", "wards", "smokes", "teamfights"):
        assert not CJK.search(json.dumps(result[section], ensure_ascii=False)), section


@pytest.mark.parametrize(
    "team,lane,lane_role,roaming,expected_name,expected_role,expected_side",
    [
        ("radiant", 1, 1, False, "safe", 1, "bottom"),
        ("dire", 1, 3, False, "off", 3, "bottom"),
        ("dire", 3, 0, False, "safe", 1, "top"),      # derived from absolute lane
        ("radiant", 3, 0, False, "off", 3, "top"),    # derived from absolute lane
        ("radiant", 2, 0, False, "mid", 2, "mid"),
        ("dire", 4, 4, False, "roaming", 4, "jungle"),
        ("radiant", 1, 1, True, "roaming", 1, "bottom"),
        ("radiant", 0, 0, False, "unknown", 0, "unknown"),
    ],
)
def test_lane_codes(team, lane, lane_role, roaming, expected_name, expected_role, expected_side):
    slot = 0 if team == "radiant" else 128
    p = make_player(slot, lane=lane, lane_role=lane_role, is_roaming=roaming)
    row = analyzer.analyze_laning(make_match([p]))["players"][0]
    assert row["lane_name"] == expected_name
    assert row["lane_role"] == expected_role
    assert row["lane_side"] == expected_side
    assert row["lane"] == lane


def test_laning_ten_minute_values_fall_back_to_snapshots():
    p = make_player(
        0,
        laning_snapshots=[
            LaningSnapshot(time=300, last_hits=20, denies=2, net_worth=1500),
            LaningSnapshot(time=600, last_hits=45, denies=6, net_worth=3500),
            LaningSnapshot(time=605, last_hits=46, denies=6, net_worth=3550),
        ],
        laning_denies=[LaningCreepKill(time=100)],
    )
    row = analyzer.analyze_laning(make_match([p]))["players"][0]
    assert row["lh_ten"] == 45
    assert row["dn_ten"] == 6
    assert row["gold_ten"] == 3500

    q = make_player(0, lh_t=list(range(0, 120, 10)), gold_t=[i * 100 for i in range(12)],
                    laning_denies=[LaningCreepKill(time=100), LaningCreepKill(time=700)])
    row = analyzer.analyze_laning(make_match([q]))["players"][0]
    assert row["lh_ten"] == 100
    assert row["gold_ten"] == 1000
    assert row["dn_ten"] == 1


def test_farm_reason_codes_follow_game_enums():
    p = make_player(
        0,
        gold_reasons={"13": 100, "14": 50, "15": 30, "16": 5, "17": 7, "12": 200, "99": 1},
        xp_reasons={"1": 10, "2": 20, "4": 30, "5": 40},
    )
    farm = analyzer.analyze_farm(make_match([p]))
    assert farm["gold_reasons"][0]["reasons"] == {
        "creep": 100, "neutral": 50, "roshan": 30, "courier": 5,
        "bounty_rune": 7, "hero": 200, "unknown_99": 1,
    }
    assert farm["xp_reasons"][0]["reasons"] == {"hero": 10, "creep": 20, "tome": 30, "outpost": 40}


def test_rune_codes_follow_dota_enum():
    p = make_player(0, runes={"5": 3, "6": 1, "0": 2, "42": 1})
    runes = analyzer.analyze_objectives(make_match([p]))["runes"][0]
    assert runes["runes"] == {"bounty": 3, "arcane": 1, "double_damage": 2, "unknown_42": 1}
    assert runes["total"] == 7


def test_objective_and_chat_slot_is_index_not_player_slot():
    players = [make_player(s) for s in (0, 1, 2, 3, 4, 128, 129, 130, 131, 132)]
    match = make_match(
        players,
        objectives=[
            {"time": 100, "type": "CHAT_MESSAGE_FIRSTBLOOD", "slot": 7},
            {"time": 50, "type": "CHAT_MESSAGE_ROSHAN_KILL", "team": 3},
            {"time": 900, "type": "building_kill", "unit": "npc_dota_creep_goodguys_melee",
             "key": "npc_dota_badguys_tower1_mid"},
            {"time": 950, "type": "building_kill", "key": "npc_dota_goodguys_tower1_top"},
            {"time": 960, "type": "building_kill", "unit": "npc_dota_hero_hero131",
             "key": "npc_dota_goodguys_tower2_top"},
        ],
        chat=[{"time": 10, "type": "chat", "key": "gl hf", "slot": 6}],
    )
    objs = analyzer.analyze_objectives(match)["objectives"]
    assert [o["time"] for o in objs] == [50, 100, 900, 950, 960]
    by_time = {o["time"]: o for o in objs}
    assert by_time[50]["team"] == "dire"
    assert by_time[100]["team"] == "dire"            # index 7 → slot 130
    assert by_time[100]["hero_name"] == "Hero130"
    assert by_time[900]["team"] == "radiant"         # killer unit is radiant
    assert by_time[950]["team"] == "dire"            # radiant tower lost → dire credited
    assert by_time[960]["hero_name"] == "Hero131"
    assert by_time[960]["team"] == "dire"

    msg = analyzer.analyze_chat(match)["messages"][0]
    assert msg["player_slot"] == 129
    assert msg["team"] == "dire"


def test_wards_team_attribution_and_dewards():
    players = [make_player(0), make_player(128)]
    match = make_match(
        players,
        duration=1000,
        wards=[
            WardEvent(ward_type="observer", x=0, y=0, time=100, player_slot=0, destroyed_time=200),   # dewarded
            WardEvent(ward_type="observer", x=0, y=0, time=100, player_slot=0, destroyed_time=460),   # expired
            WardEvent(ward_type="observer", x=0, y=0, time=900, player_slot=128, destroyed_time=1000),  # game end
            WardEvent(ward_type="sentry", x=0, y=0, time=100, player_slot=128, destroyed_time=520),   # expired (420 s)
            WardEvent(ward_type="sentry", x=0, y=0, time=0, player_slot=-1),                         # owner unknown
        ],
    )
    wards = analyzer.analyze_wards(match)
    assert wards["radiant"]["total_obs"] == 2
    assert wards["radiant"]["dewarded_obs"] == 1
    assert wards["dire"]["total_obs"] == 1
    assert wards["dire"]["dewarded_obs"] == 0
    assert wards["dire"]["dewarded_sent"] == 0
    assert wards["radiant"]["total_sent"] == 0  # unknown owner is not radiant
    assert len(wards["all_wards"]) == 5
    assert wards["all_wards"][-1]["team"] is None
    assert wards["all_wards"][-1]["hero_name"] == ""


def test_smoke_unknown_team_kept_in_timeline_only():
    match = make_match(
        [make_player(0)],
        smokes=[
            SmokeEvent(time=300, team="radiant", player_slots=[0], end_time=330, result="kill"),
            SmokeEvent(time=200, team="unknown", player_slots=[]),
        ],
    )
    smokes = analyzer.analyze_smokes(match)
    assert smokes["radiant"]["total"] == 1
    assert smokes["dire"]["total"] == 0
    assert [s["time"] for s in smokes["timeline"]] == [200, 300]


def test_teamfight_killer_lookup():
    tf = TeamfightEvent(
        start_time=600, end_time=630, radiant_deaths=1, dire_deaths=0,
        players=[
            TeamfightPlayer(player_slot=0, hero_name="A", hero_slug="a", x=0, y=0, damage_dealt=0,
                            damage_taken=0, kills=0, deaths=1, abilities_used=[], death_time_offset=5.0),
            TeamfightPlayer(player_slot=128, hero_name="B", hero_slug="b", x=0, y=0, damage_dealt=0,
                            damage_taken=0, kills=1, deaths=0, abilities_used=[],
                            kill_targets=[{"target": "A", "time_offset": 5.2}]),
        ],
    )
    result = analyzer.analyze_teamfights(make_match([make_player(0), make_player(128)], teamfights=[tf]))
    assert result[0]["winner"] == "dire"
    assert result[0]["kill_timeline"] == [
        {"victim": "A", "victim_slug": "a", "victim_team": "radiant", "killer": "B", "time_offset": 5.0},
    ]
    assert result[0]["players"][1]["team"] == "dire"


def test_purchase_log_tolerates_missing_times():
    p = make_player(0, purchase_log=[{"time": None, "key": "tango"}, {"key": "branches"}, {"time": 400, "key": "boots"}])
    result = full_analysis(make_match([p], duration=600))
    log = result["match_overview"]["players"][0]["purchase_log"]
    assert [e["time"] for e in log] == [0, 0, 400]
    phases = result["purchases"]["players"][0]["phases"]
    assert [i["key"] for i in phases["starting"]] == ["tango", "branches"]
    assert [i["key"] for i in phases["5-10"]] == ["boots"]


def test_ward_team_field_is_preferred_when_valid():
    players = [make_player(0), make_player(128)]
    wards = [
        WardEvent(ward_type="observer", x=0, y=0, time=10, player_slot=-1),
        WardEvent(ward_type="observer", x=0, y=0, time=10, player_slot=0),
    ]
    if "team" in WardEvent.model_fields:
        wards[0] = WardEvent(ward_type="observer", x=0, y=0, time=10, player_slot=-1, team="dire")
        wards.append(WardEvent(ward_type="observer", x=0, y=0, time=10, player_slot=0, team="bogus"))
    result = analyzer.analyze_wards(make_match(players, wards=wards))
    teams = [w["team"] for w in result["all_wards"]]
    if "team" in WardEvent.model_fields:
        assert teams == ["dire", "radiant", "radiant"]
    else:
        assert teams == [None, "radiant"]


def test_smoke_caught_counted_and_overview_extras():
    match = make_match(
        [make_player(0)],
        smokes=[SmokeEvent(time=300, team="radiant", player_slots=[0], result="caught", deaths_suffered=2)],
    )
    result = full_analysis(match)
    stats = result["smokes"]["radiant"]
    assert stats["caught"] == 1 and stats["successful"] == 0 and stats["total"] == 1
    overview = result["match_overview"]
    assert overview["buildings"] == [] and overview["partial"] is False
    assert overview["players"][0]["neutral_item"] == ""
