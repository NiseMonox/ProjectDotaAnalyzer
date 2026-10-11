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
    AbilityCast,
    AbilityEvent,
    KillEvent,
    LaningCreepKill,
    LaningSnapshot,
    MatchData,
    PlayerData,
    SkillEvent,
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
    assert result["laning"] == {"players": [], "lane_efficiency_applicable": True}
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
        xp_reasons={"1": 10, "2": 20, "4": 30, "5": 40, "6": 50, "7": 60},
    )
    farm = analyzer.analyze_farm(make_match([p]))
    assert farm["gold_reasons"][0]["reasons"] == {
        "creep": 100, "neutral": 50, "roshan": 30, "courier": 5,
        "bounty_rune": 7, "hero": 200, "unknown_99": 1,
    }
    # Reason 4 is wisdom-rune / Shrine of Wisdom XP in current patches (the tome no
    # longer exists); 6 and 7 are DOTA_ModifyXP_CatchUp and DOTA_ModifyXP_HeroAbility.
    assert farm["xp_reasons"][0]["reasons"] == {
        "hero": 10, "creep": 20, "wisdom_rune": 30, "outpost": 40, "catch_up": 50, "ability": 60,
    }


def test_illusion_bounty_has_a_code_and_empty_reasons_are_left_out():
    """Values of a 7.41f replay: gold reason 22 (illusion kills) and an XP reason with 0 XP."""
    p = make_player(
        0,
        gold_reasons={"12": 2998, "22": 159, "16": 60, "21": 30, "19": 0},
        xp_reasons={"1": 4717, "4": 600, "8": 0},
    )
    farm = analyzer.analyze_farm(make_match([p]))
    assert farm["gold_reasons"][0]["reasons"] == {"hero": 2998, "illusion": 159, "courier": 90}
    assert farm["xp_reasons"][0]["reasons"] == {"hero": 4717, "wisdom_rune": 600}


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
    """Dewards are counted from WardEvent.dewarded, never from the lifetime."""
    players = [make_player(0), make_player(128)]
    match = make_match(
        players,
        duration=1000,
        wards=[
            WardEvent(ward_type="observer", x=0, y=0, time=100, player_slot=0, destroyed_time=200, dewarded=True),
            WardEvent(ward_type="observer", x=0, y=0, time=100, player_slot=0, destroyed_time=460),   # expired
            # Short-lived but not flagged (e.g. removed with its owner's team): no deward.
            WardEvent(ward_type="observer", x=0, y=0, time=900, player_slot=128, destroyed_time=950),
            # A full natural lifetime and still dewarded in its last second.
            WardEvent(ward_type="sentry", x=0, y=0, time=100, player_slot=128, destroyed_time=519, dewarded=True),
            WardEvent(ward_type="sentry", x=0, y=0, time=0, player_slot=-1),                         # owner unknown
        ],
    )
    wards = analyzer.analyze_wards(match)
    assert wards["radiant"]["total_obs"] == 2
    assert wards["radiant"]["dewarded_obs"] == 1
    assert wards["dire"]["total_obs"] == 1
    assert wards["dire"]["dewarded_obs"] == 0
    assert wards["dire"]["dewarded_sent"] == 1
    assert wards["radiant"]["total_sent"] == 0  # unknown owner is not radiant
    assert [w["dewarded"] for w in wards["radiant"]["observers"]] == [True, False]
    assert len(wards["all_wards"]) == 5
    assert [w["dewarded"] for w in wards["all_wards"]] == [True, False, False, True, False]
    assert [w["lifetime"] for w in wards["all_wards"]] == [100, 360, 50, 419, None]
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


def _fight(start=600, end=630, **players) -> TeamfightEvent:
    """Teamfight between slot 0 ("A") and slot 128 ("B"); kwargs override a player's fields."""
    base = {
        0: dict(player_slot=0, hero_name="A", hero_slug="a"),
        128: dict(player_slot=128, hero_name="B", hero_slug="b"),
    }
    built = []
    for slot, fields in base.items():
        fields.update(x=0, y=0, damage_dealt=0, damage_taken=0, kills=0, deaths=0, abilities_used=[])
        fields.update(players.get(f"p{slot}", {}))
        built.append(TeamfightPlayer(**fields))
    return TeamfightEvent(
        start_time=start, end_time=end, players=built,
        radiant_deaths=built[0].deaths, dire_deaths=built[1].deaths,
    )


def test_teamfight_killer_lookup_without_kill_log():
    """Sources without a kill log: one death per player, killer from kill_targets."""
    tf = _fight(
        p0=dict(deaths=1, death_time_offset=5.0),
        p128=dict(kills=1, kill_targets=[{"target": "A", "time_offset": 5.2}]),
    )
    result = analyzer.analyze_teamfights(make_match([make_player(0), make_player(128)], teamfights=[tf]))
    assert result[0]["winner"] == "dire"
    assert result[0]["kill_timeline"] == [{
        "time": 605.0, "time_str": "10:05", "time_offset": 5.0,
        "victim": "A", "victim_slug": "a", "victim_slot": 0, "victim_team": "radiant",
        "killer": "B", "killer_slug": "b", "killer_slot": 128, "killer_team": "dire", "killer_unit": "",
        "deny": False, "confirmed": True, "counted": True,
    }]
    assert result[0]["players"][1]["team"] == "dire"

    # Contract W3 form: target_slot, and no time when the source has none.
    tf = _fight(
        p0=dict(deaths=1, death_time_offset=5.0),
        p128=dict(kills=1, kill_targets=[{"target": "Display Name", "target_slot": 0, "time_offset": None}]),
    )
    result = analyzer.analyze_teamfights(make_match([make_player(0), make_player(128)], teamfights=[tf]))
    assert result[0]["kill_timeline"][0]["killer_slot"] == 128


def test_teamfight_kill_timeline_comes_from_kill_log():
    players = [make_player(0), make_player(128)]
    # The per-player teamfight data is deliberately wrong / incomplete: one
    # death_time_offset per player cannot describe two deaths (buyback).
    fights = [
        _fight(600, 630, p0=dict(deaths=2, death_time_offset=1.0),
               p128=dict(kills=2, kill_targets=[{"target": "A", "time_offset": 1.0}])),
        _fight(900, 920, p128=dict(deaths=1)),
    ]
    kills_log = [
        KillEvent(time=300.0, victim_slot=128, killer_slot=0),                         # outside any fight
        KillEvent(time=599.0, victim_slot=128, killer_slot=0),                         # a whole second early
        KillEvent(time=599.4, victim_slot=128, killer_slot=0),                         # start - slack
        KillEvent(time=604.4, victim_slot=0, killer_slot=128, x=10.0, y=20.0),
        KillEvent(time=611.0, victim_slot=128, killer_slot=None,
                  killer_unit="npc_dota_goodguys_tower1_mid"),
        KillEvent(time=630.6, victim_slot=0, killer_slot=128, confirmed=False),        # end + slack
        KillEvent(time=631.0, victim_slot=128, killer_slot=0),                         # a whole second late
        KillEvent(time=905.0, victim_slot=128, killer_slot=128, deny=True),
    ]
    result = analyzer.analyze_teamfights(make_match(players, teamfights=fights, kills_log=kills_log))

    # The slack is for fractional kill times only: a kill one whole second
    # outside the fight (OpenDota times are whole seconds) is not part of it.
    first = result[0]["kill_timeline"]
    assert [(k["victim"], k["killer"], k["time_offset"]) for k in first] == [
        ("Hero128", "Hero0", -0.6), ("Hero0", "Hero128", 4.4), ("Hero128", None, 11.0), ("Hero0", "Hero128", 30.6),
    ]
    first = first[1:]
    assert first[0]["victim_team"] == "radiant" and first[0]["killer_team"] == "dire"
    assert first[0]["killer_slug"] == "hero128" and first[0]["time_str"] == "10:04"
    assert first[1]["killer_unit"] == "npc_dota_goodguys_tower1_mid"
    assert first[1]["killer_slot"] is None and first[1]["killer_team"] == "radiant"
    # Unconfirmed and not backed by the scoreboard (Hero0 has no deaths there).
    assert first[2]["confirmed"] is False and first[2]["counted"] is False
    assert [k["counted"] for k in first[:2]] == [True, True]

    second = result[1]["kill_timeline"]
    assert [(k["victim_slot"], k["deny"], k["time_offset"]) for k in second] == [(128, True, 5.0)]
    assert second[0]["counted"] is True                                               # a deny is a real death


def test_teamfight_abilities_used_are_keys_with_display_names():
    cast = AbilityCast(ability_name="Custom Spell", ability_key="custom_spell", time_offset=1.0, x=0, y=0)
    tf = _fight(
        p0=dict(abilities_used=["custom_spell", "no_such_ability_key"]),
        p128=dict(abilities_used=["Custom Spell"], ability_casts=[cast]),   # display name (old mock data)
    )
    players = analyzer.analyze_teamfights(make_match([make_player(0), make_player(128)], teamfights=[tf]))[0]["players"]
    assert players[0]["abilities_used"] == ["custom_spell", "no_such_ability_key"]
    assert players[0]["abilities_used_names"] == ["custom_spell", "no_such_ability_key"]  # unknown: key itself
    assert players[1]["abilities_used"] == ["custom_spell"]
    assert len(players[1]["abilities_used_names"]) == 1

    from backend.app.parser import ability_display_name

    known = "antimage_mana_break"
    if ability_display_name(known) != known:  # static catalog present
        tf = _fight(p0=dict(abilities_used=[known]))
        match = make_match([make_player(0), make_player(128)], teamfights=[tf])
        row = analyzer.analyze_teamfights(match)[0]["players"][0]
        assert row["abilities_used"] == [known]
        assert row["abilities_used_names"] == [ability_display_name(known)]


def test_world_interactions_are_not_listed_as_abilities():
    """Watchers, outposts, Twin Gates and High Five have no display name (the
    raw key was shown) and are not hero abilities."""
    def event(key: str, time: int) -> AbilityEvent:
        return AbilityEvent(player_slot=0, hero_name="Hero0", ability_name=key, ability_key=key, time=time)

    def cast(key: str, offset: float) -> AbilityCast:
        return AbilityCast(ability_name=key, ability_key=key, time_offset=offset, x=0, y=0)

    utility = ["ability_lamp_use", "ability_capture", "twin_gate_portal_warp", "plus_high_five"]
    tf = _fight(p0=dict(abilities_used=["custom_spell", *utility],
                        ability_casts=[cast("ability_lamp_use", 1.0), cast("custom_spell", 2.0)]))
    match = make_match(
        [make_player(0), make_player(128)], teamfights=[tf],
        ability_events=[event("custom_spell", 10), *(event(key, 20 + i) for i, key in enumerate(utility)),
                        event("custom_spell", 30)],
    )
    result = full_analysis(match)

    abilities = result["abilities"]
    row = abilities["by_player"][0]
    assert [e["ability_key"] for e in row["events"]] == ["custom_spell", "custom_spell"]
    assert (row["total_casts"], row["ability_counts"]) == (2, {"custom_spell": 2})
    assert abilities["most_used"] == [("custom_spell", 2)]

    fighter = result["teamfights"][0]["players"][0]
    assert fighter["abilities_used"] == fighter["abilities_used_names"] == ["custom_spell"]
    assert [c["ability_key"] for c in fighter["ability_casts"]] == ["custom_spell"]
    # the casts themselves stay in the match: the AI layer reads their time and position
    assert len(match.ability_events) == 6 and len(tf.players[0].ability_casts) == 2


def test_combat_matrices_and_kill_feed_use_kill_log():
    players = [
        make_player(0, hero_name="Wraith King", hero_slug="skeleton_king",
                    killed={"Zeus": 9}, damage={"Zeus": 1500, "npc_dota_creep_badguys_melee": 99}),
        make_player(128, hero_name="Zeus", hero_slug="zuus", damage={"npc_dota_hero_skeleton_king": 700}),
    ]
    kills_log = [
        KillEvent(time=100.0, victim_slot=128, killer_slot=0, assist_slots=[1], x=5.0, y=6.0),
        KillEvent(time=200.5, victim_slot=0, killer_slot=128, confirmed=False),     # reincarnation?
        KillEvent(time=300.0, victim_slot=0, killer_slot=None, killer_unit="npc_dota_badguys_tower1_mid"),
        KillEvent(time=400.0, victim_slot=0, killer_slot=None, killer_unit="npc_dota_neutral_kobold"),
        KillEvent(time=500.0, victim_slot=128, killer_slot=128, deny=True),
        KillEvent(time=50.0, victim_slot=128, killer_slot=0),
    ]
    combat = analyzer.analyze_combat(make_match(players, kills_log=kills_log))
    assert combat["kill_matrix"]["heroes"] == ["Wraith King", "Zeus"]
    # Row = killer. The "killed" counter (9) is ignored when a kill log exists;
    # unconfirmed entries, denies and non-hero kills are not hero kills.
    assert combat["kill_matrix"]["matrix"] == [[0, 2], [0, 0]]
    # Damage keys: display name (contract) and legacy npc name both resolve.
    assert combat["damage_matrix"]["matrix"] == [[0, 1500], [700, 0]]

    feed = combat["kills"]
    assert [k["time"] for k in feed] == [50.0, 100.0, 200.5, 300.0, 400.0, 500.0]
    assert [(k["radiant_score"], k["dire_score"]) for k in feed] == [
        (1, 0), (2, 0), (2, 0), (2, 1), (2, 1), (2, 1),
    ]
    assert feed[1] == {
        "time": 100.0, "time_str": "1:40", "victim": "Zeus", "victim_slug": "zuus", "victim_slot": 128,
        "victim_team": "dire", "killer": "Wraith King", "killer_slug": "skeleton_king", "killer_slot": 0,
        "killer_team": "radiant", "killer_unit": "", "deny": False, "confirmed": True, "counted": True,
        "assist_slots": [1], "x": 5.0, "y": 6.0, "radiant_score": 2, "dire_score": 0,
    }
    assert [k["counted"] for k in feed] == [True, True, False, True, True, True]
    assert feed[3]["killer"] is None and feed[3]["killer_team"] == "dire"
    assert feed[4]["killer_team"] is None


def test_kill_feed_scores_a_unit_kill_for_the_side_the_source_reports():
    """Replay 9038370833: a Dire hero dies to the Radiant fountain at 23:53.

    The unit name ("dota_fountain") carries no side, so the feed ended 3-29
    where the scoreboard says 4-29. ``KillEvent.killer_team`` names the side.
    """
    players = [make_player(0, hero_name="Axe", hero_slug="axe"),
               make_player(129, hero_name="Zeus", hero_slug="zuus")]

    def last_score(*kills):
        feed = analyzer.analyze_combat(make_match(players, kills_log=list(kills)))["kills"]
        return feed[-1]["killer_team"], (feed[-1]["radiant_score"], feed[-1]["dire_score"])

    fountain = dict(time=1433.8, victim_slot=129, killer_slot=None, killer_unit="dota_fountain")
    assert last_score(KillEvent(**fountain, killer_team="radiant")) == ("radiant", (1, 0))
    # a match stored before the field existed: the name is all there is
    assert last_score(KillEvent(**fountain)) == (None, (0, 0))
    assert last_score(KillEvent(time=5.0, victim_slot=0, killer_slot=None, killer_team=None,
                                killer_unit="npc_dota_badguys_tower1_mid")) == ("dire", (0, 1))
    # the reported side beats what a name suggests; a neutral kill scores for nobody
    assert last_score(KillEvent(time=5.0, victim_slot=129, killer_slot=None, killer_team="neutral",
                                killer_unit="npc_dota_neutral_kobold")) == ("neutral", (0, 0))
    assert last_score(KillEvent(time=5.0, victim_slot=129, killer_slot=None, killer_team="neutral",
                                killer_unit="npc_dota_roshan")) == ("neutral", (0, 0))
    # a credited hero decides the side even when a neutral landed the blow
    assert last_score(KillEvent(time=5.0, victim_slot=129, killer_slot=0, killer_team="neutral",
                                killer_unit="npc_dota_neutral_kobold")) == ("radiant", (1, 0))
    # a deny carries the victim's own side and never scores, with or without a hero
    assert last_score(KillEvent(time=5.0, victim_slot=129, killer_slot=None, killer_team="dire",
                                killer_unit="npc_dota_badguys_tower1_mid", deny=True)) == ("dire", (0, 0))
    # a unit of the victim's own side that is not flagged as a deny does not score either
    assert last_score(KillEvent(time=5.0, victim_slot=129, killer_slot=None, killer_team="dire",
                                killer_unit="dota_fountain")) == ("dire", (0, 0))


def test_unconfirmed_kills_are_reconciled_with_the_scoreboard():
    """OpenDota logs a reincarnation like a kill; the scoreboard says how many were real."""
    players = [
        make_player(0, hero_name="Zeus", hero_slug="zuus", kills=2),
        make_player(1, hero_name="Lion", hero_slug="lion", kills=0),
        make_player(128, hero_name="Wraith King", hero_slug="skeleton_king", deaths=2),
    ]
    kills_log = [
        KillEvent(time=100.0, victim_slot=128, killer_slot=0, confirmed=False),   # reincarnation
        KillEvent(time=108.0, victim_slot=128, killer_slot=0, confirmed=False),   # the real death
        KillEvent(time=500.0, victim_slot=128, killer_slot=1, confirmed=False),   # Lion has no kill
        KillEvent(time=503.0, victim_slot=128, killer_slot=0, confirmed=False),
    ]
    combat = analyzer.analyze_combat(make_match(players, kills_log=kills_log))
    # Zeus: 2 scoreboard kills, WK: 2 scoreboard deaths. Lion never gets one.
    assert combat["kill_matrix"]["matrix"] == [[0, 0, 2], [0, 0, 0], [0, 0, 0]]
    feed = combat["kills"]
    assert [k["confirmed"] for k in feed] == [False] * 4
    # He died again within seconds: the first entry of each pair was the
    # reincarnation, the second the death that counted.
    assert [k["counted"] for k in feed] == [False, True, False, True]
    assert [(k["time"], k["radiant_score"]) for k in feed] == [(100.0, 0), (108.0, 1), (500.0, 1), (503.0, 2)]

    # Every death already explained by a confirmed entry: nothing else counts.
    kills_log.append(KillEvent(time=900.0, victim_slot=128, killer_slot=0))
    kills_log.append(KillEvent(time=950.0, victim_slot=128, killer_slot=None, killer_unit="npc_dota_neutral_kobold"))
    combat = analyzer.analyze_combat(make_match(players, kills_log=kills_log))
    assert combat["kill_matrix"]["matrix"] == [[0, 0, 1], [0, 0, 0], [0, 0, 0]]
    assert [k["counted"] for k in combat["kills"]] == [False] * 4 + [True, True]


def test_unconfirmed_kills_without_a_follow_up_death_count_earliest_first():
    """Match 8762775364 (OpenDota): Wraith King, 3 deaths, 5 kill-log entries.

    The replay shows he died at 4:50, 6:25 and 13:40; at 20:41 and 28:31 he
    reincarnated and survived. Nothing follows those entries, so the only
    evidence is that reincarnation does not exist in the first minutes.
    """
    players = [
        make_player(0, hero_name="Invoker", hero_slug="invoker", kills=3),        # Zeus once, Wraith King twice
        make_player(1, hero_name="Legion Commander", hero_slug="legion_commander", kills=1),
        make_player(128, hero_name="Wraith King", hero_slug="skeleton_king", deaths=3),
        make_player(129, hero_name="Zeus", hero_slug="zuus", deaths=1),
    ]
    kills_log = [
        KillEvent(time=290.0, victim_slot=128, killer_slot=0, confirmed=False),
        KillEvent(time=385.0, victim_slot=128, killer_slot=1),
        KillEvent(time=600.0, victim_slot=129, killer_slot=0),
        KillEvent(time=820.0, victim_slot=128, killer_slot=0, confirmed=False),
        KillEvent(time=1241.0, victim_slot=128, killer_slot=0, confirmed=False),
        KillEvent(time=1711.0, victim_slot=128, killer_slot=0, confirmed=False),
    ]
    match = make_match(players, kills_log=kills_log)
    combat = analyzer.analyze_combat(match)
    assert combat["kill_matrix"]["matrix"] == [[0, 0, 2, 1], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert [(k["time"], k["counted"], k["radiant_score"]) for k in combat["kills"]] == [
        (290.0, True, 1), (385.0, True, 2), (600.0, True, 3), (820.0, True, 4),
        (1241.0, False, 4), (1711.0, False, 4),
    ]

    # The teamfight timeline carries the same flag.
    fights = [_fight(810, 830), _fight(1230, 1250)]
    timelines = [tf["kill_timeline"] for tf in analyzer.analyze_teamfights(make_match(
        players, kills_log=kills_log, teamfights=fights))]
    assert [[(k["time"], k["counted"]) for k in tl] for tl in timelines] == [[(820.0, True)], [(1241.0, False)]]


def test_combat_kill_matrix_falls_back_to_killed_counters():
    players = [
        make_player(0, hero_name="Wraith King", hero_slug="skeleton_king",
                    killed={"Zeus": 3, "npc_dota_creep_badguys_melee": 40}),
        make_player(128, hero_name="Zeus", hero_slug="zuus",
                    killed={"npc_dota_hero_skeleton_king": 2}),
        make_player(129, hero_name="Lion", hero_slug="lion", killed={"skeleton_king": 1}),
    ]
    combat = analyzer.analyze_combat(make_match(players))
    assert combat["kill_matrix"]["matrix"] == [[0, 3, 0], [2, 0, 0], [1, 0, 0]]
    assert combat["kills"] == []


def test_talent_flag_follows_ability_keys():
    def skill(name: str, key: str, level: int) -> SkillEvent:
        return SkillEvent(ability_name=name, ability_key=key, level=level, time=level * 60)

    skills = [
        skill("Mana Break", "antimage_mana_break", 1),
        skill("+20 Movement Speed", "special_bonus_movement_speed_20", 10),
        skill("special_bonus_unique_lion_3", "special_bonus_unique_lion_3", 15),   # name not resolved
        skill("special_bonus_attributes", "special_bonus_attributes", 17),         # attribute level-up
        skill("talent_left_lvl20", "talent_left_lvl20", 20),                       # old mock scheme
        skill("talent_right_lvl25", "", 25),                                       # no key at all
    ]
    order = analyzer.analyze_skill_builds(make_match([make_player(0, skills=skills)]))[0]["skill_order"]
    assert [s["is_talent"] for s in order] == [False, True, True, False, True, True]


@pytest.fixture
def gamedata(tmp_path, monkeypatch):
    """An empty game-data directory for the data layer (caches dropped before and after)."""
    from backend.app.ai import data_manager

    directory = tmp_path / "gamedata"
    directory.mkdir()
    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", directory)
    data_manager.invalidate_all_caches()
    yield directory
    data_manager.invalidate_all_caches()


def test_talents_are_named_from_the_game_data_at_analysis_time(gamedata, tmp_path):
    """A match parsed and stored before the game data was fetched must not keep
    the stand-in talent names for good."""
    from backend.app import storage
    from backend.app.ai import data_manager

    def skill(name: str, key: str, level: int) -> SkillEvent:
        return SkillEvent(ability_name=name, ability_key=key, level=level, time=level * 60)

    facet = "special_bonus_unique_probe_hero_facet_1"
    skills = [
        skill("Probe Strike", "probe_hero_strike", 1),                         # not a talent
        skill("Probe Hero Facet 1", facet, 10),                                # stand-in made from the key
        skill("+Probe Old Name", "special_bonus_unique_probe_hero_2", 15),     # named by older data
        skill("+Probe Removed", "special_bonus_unique_probe_hero_9", 20),      # a key the new data dropped
        skill("", "special_bonus_unique_probe_hero_7", 25),                    # nothing stored, nothing known
        skill("talent_left_lvl20", "talent_left_lvl20", 26),                   # old mock scheme: no catalog key
    ]
    store = storage.MatchStore(tmp_path / "store", lambda: 10)
    store.save("replay_x", make_match([make_player(0, skills=skills)]), {"parse_source": "clarity"})

    def names() -> list[str]:
        match, _meta = store.load("replay_x")
        return [s["ability"] for s in full_analysis(match)["skill_builds"][0]["skill_order"]]

    assert names() == ["Probe Strike", "Probe Hero Facet 1", "+Probe Old Name", "+Probe Removed",
                       "Probe Hero 7", "talent_left_lvl20"]

    # the fetch finishes after the match was parsed and stored
    (gamedata / data_manager.TALENTS_FILE).write_text(json.dumps({
        facet: {"name": "+{s:bonus_blast_dot_duration}s Probe Blast Slow Duration"},
        "special_bonus_unique_probe_hero_2": {"name": "+{s:value} Probe New Name", "values": {"value": "25"}},
    }))
    data_manager.invalidate_all_caches()
    assert names() == ["Probe Strike", "+Probe Blast Slow Duration", "+25 Probe New Name", "+Probe Removed",
                       "Probe Hero 7", "talent_left_lvl20"]


def test_talent_names_of_the_match_patch_survive_newer_game_data(gamedata):
    """Talent keys are reused from patch to patch: names given by game data of
    the match's own patch must not be replaced by those of a later one."""
    from backend.app.ai import data_manager

    key = "special_bonus_unique_probe_hero_3"
    skills = [SkillEvent(ability_name="+Probe Firefly Duration", ability_key=key, level=1, time=600)]

    def fetch(patch: str, name: str) -> None:
        (gamedata / data_manager.TALENTS_FILE).write_text(json.dumps({key: {"name": name}}))
        (gamedata / "meta.json").write_text(json.dumps({"schema_version": 3, "patch_version": patch}))
        data_manager.invalidate_all_caches()

    def shown(**match_fields) -> str:
        match = make_match([make_player(0, skills=skills)], **match_fields)
        return analyzer.analyze_skill_builds(match)[0]["skill_order"][0]["ability"]

    fetch("7.42", "+Probe Lasso Duration")                 # the key is another talent in the new patch
    assert shown(patch="7.41", talent_names_patch="7.41") == "+Probe Firefly Duration"
    # named by data of another patch than the match's, or by none: the current data is the best guess
    assert shown(patch="7.41", talent_names_patch="7.40") == "+Probe Lasso Duration"
    assert shown(patch="7.41", talent_names_patch="") == "+Probe Lasso Duration"
    assert shown(patch="", talent_names_patch="7.41") == "+Probe Lasso Duration"          # patch of the match unknown
    assert shown(patch="7.42", talent_names_patch="7.42") == "+Probe Lasso Duration"      # the data in use now
    fetch("7.41", "+Probe Firefly Duration II")            # refetched on the same patch: current names
    assert shown(patch="7.41", talent_names_patch="7.41") == "+Probe Firefly Duration II"


def test_a_broken_data_layer_leaves_the_stored_talent_names(monkeypatch):
    skills = [SkillEvent(ability_name="+20 Movement Speed", ability_key="special_bonus_movement_speed_20",
                         level=1, time=600)]
    monkeypatch.setitem(__import__("sys").modules, "backend.app.ai.data_manager", None)
    order = analyzer.analyze_skill_builds(make_match([make_player(0, skills=skills)]))[0]["skill_order"]
    assert [s["ability"] for s in order] == ["+20 Movement Speed"]


def test_item_builds_carry_neutral_item_and_backpack():
    p = make_player(0, neutral_item="mysterious_hat", backpack=["tpscroll", "smoke_of_deceit"])
    row = analyzer.analyze_item_builds(make_match([p, make_player(128)]))
    assert row[0]["neutral_item"] == "mysterious_hat"
    assert row[0]["backpack"] == ["tpscroll", "smoke_of_deceit"]
    assert row[1]["neutral_item"] == "" and row[1]["backpack"] == []


def test_overview_winner():
    assert full_analysis(make_match(radiant_win=True))["match_overview"]["winner"] == "radiant"
    overview = full_analysis(make_match(radiant_win=False))["match_overview"]
    assert overview["winner"] == "dire" and overview["radiant_win"] is False
    # Truncated replay: radiant_win is only a placeholder.
    overview = full_analysis(make_match(radiant_win=False, winner_known=False, partial=True))["match_overview"]
    assert overview["winner"] is None and overview["radiant_win"] is None and overview["partial"] is True


def test_overview_passes_cluster_and_region_through():
    overview = full_analysis(make_match(cluster=151, region=5))["match_overview"]
    assert (overview["cluster"], overview["region"]) == (151, 5)
    overview = full_analysis(make_match())["match_overview"]
    assert (overview["cluster"], overview["region"]) == (0, 0)       # unknown


def test_draft_team_is_null_for_a_ban_no_team_owns():
    """Contract W3: picks_bans[].team is 0/1, or null (every ban of an uploaded replay)."""
    players = [make_player(0, hero_id=74, hero_name="Invoker", hero_slug="invoker"),
               make_player(128, hero_id=42, hero_name="Wraith King", hero_slug="skeleton_king")]
    match = make_match(players, picks_bans=[
        {"is_pick": True, "hero_id": 74, "team": 0, "order": 0},
        {"is_pick": True, "hero_id": 42, "team": 1, "order": 1},
        {"is_pick": False, "hero_id": 38, "team": None, "order": 2},
        {"is_pick": False, "hero_id": 23, "order": 3},                    # no team at all
        {"is_pick": False, "hero_id": 68, "team": 0, "order": 4},
        {"is_pick": False, "hero_id": 57, "team": 1, "order": 5},
        {"is_pick": False, "hero_id": 95, "team": 7, "order": 6},         # not a team
        {"is_pick": False, "hero_id": 88, "team": [], "order": 7},        # malformed
    ])
    draft = analyzer.analyze_draft(match)
    assert draft["has_draft"] is True
    assert [e["team"] for e in draft["picks_bans"]] == ["radiant", "dire", None, None, "radiant", "dire", None, None]
    assert [e["hero_name"] for e in draft["picks_bans"][:2]] == ["Invoker", "Wraith King"]
    bans = [e for e in draft["picks_bans"] if not e["is_pick"]]
    assert sum(e["team"] == "dire" for e in bans) == 1 and sum(e["team"] is None for e in bans) == 4


def test_lane_efficiency_is_computed_when_the_source_has_none():
    gold_t = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3628, 4000]
    computed = make_player(0, gold_t=gold_t)
    from_source = make_player(1, gold_t=gold_t, lane_efficiency=0.5, lane_efficiency_pct=50)
    short_game = make_player(2, gold_t=gold_t[:8])
    rows = analyzer.analyze_laning(make_match([computed, from_source, short_game]))["players"]
    # Same value OpenDota reports for 3628 gold at 10:00 (match 8762775364, slot 0).
    assert rows[0]["lane_efficiency"] == pytest.approx(0.7332255456750202)
    assert rows[0]["lane_efficiency_pct"] == 73
    assert (rows[1]["lane_efficiency"], rows[1]["lane_efficiency_pct"]) == (0.5, 50)
    assert (rows[2]["lane_efficiency"], rows[2]["lane_efficiency_pct"]) == (0.0, 0.0)


def test_lane_efficiency_is_not_given_for_turbo():
    """Turbo replay 9038388762: 189 %, 170 %, 165 % ... under a column defined as the share of the
    ideal lane income at 10:00. The ideal is the standard game's; Turbo doubles the gold."""
    gold_t = [0, 900, 1800, 2700, 3600, 4500, 5400, 6300, 7200, 8100, 9352, 10000]
    players = [make_player(0, gold_t=gold_t),                                             # 189 % by the formula
               make_player(1, gold_t=gold_t, lane_efficiency=1.7, lane_efficiency_pct=170)]   # the source's own
    turbo = analyzer.analyze_laning(make_match(players, game_mode="turbo"))
    assert turbo["lane_efficiency_applicable"] is False
    assert [(r["lane_efficiency"], r["lane_efficiency_pct"]) for r in turbo["players"]] == [(None, None)] * 2
    assert [r["gold_ten"] for r in turbo["players"]] == [9352, 9352]        # the gold itself is still shown
    for mode in ("all_pick", "ranked_all_pick", "captains_mode", "ability_draft", "unknown", "All Pick", ""):
        standard = analyzer.analyze_laning(make_match(players, game_mode=mode))
        assert standard["lane_efficiency_applicable"] is True, mode
        assert [r["lane_efficiency_pct"] for r in standard["players"]] == [189, 170], mode
    # the match page says which mode it is, in the same payload
    assert full_analysis(make_match(players, game_mode="turbo"))["match_overview"]["game_mode"] == "turbo"


def test_dn_ten_comes_from_the_deny_timeline():
    dn_t = [0, 1, 1, 2, 4, 5, 6, 8, 9, 11, 12, 14]
    opendota_like = make_player(0, denies=14, dn_t=dn_t)            # per-minute timeline, no snapshots
    timeline_first = make_player(1, denies=14, dn_t=dn_t, laning_snapshots=[LaningSnapshot(time=600, denies=3)])
    short_game = make_player(2, denies=5, dn_t=dn_t[:6])
    rows = analyzer.analyze_laning(make_match([opendota_like, timeline_first, short_game]))["players"]
    assert [r["dn_ten"] for r in rows] == [12, 12, 5]


def test_ten_minute_values_are_null_without_a_source():
    """No timeline, snapshot or log: the value is unknown, not 0."""
    unparsed = make_player(0, last_hits=180, denies=14, net_worth=15000)   # OpenDota scoreboard only
    nothing_happened = make_player(1, last_hits=0, denies=0)
    with_logs = make_player(
        2, last_hits=90, denies=3,
        laning_creep_kills=[LaningCreepKill(time=30), LaningCreepKill(time=590), LaningCreepKill(time=601)],
        laning_denies=[LaningCreepKill(time=50), LaningCreepKill(time=650)],
    )
    with_snapshot = make_player(3, last_hits=90, denies=3, laning_snapshots=[
        LaningSnapshot(time=600, last_hits=44, denies=2, net_worth=3100)])
    rows = analyzer.analyze_laning(make_match([unparsed, nothing_happened, with_logs, with_snapshot]))["players"]
    assert [r["lh_ten"] for r in rows] == [None, 0, 2, 44]
    assert [r["dn_ten"] for r in rows] == [None, 0, 1, 2]
    assert [r["gold_ten"] for r in rows] == [None, None, None, 3100]
    assert [r["xp_ten"] for r in rows] == [None, None, None, None]
    json.dumps(rows)


def test_lane_pos_histogram_in_world_units():
    # OpenDota: raw cell histogram {x: {y: samples}}.
    od = make_player(0, lane_pos={"76": {"78": 36, "79": 2}, "128": {"128": 5}, "bad": {"1": 1}, "90": "x"})
    # Replay sources: 5-second snapshots (two in the same 128-unit cell, one after 10:00).
    replay = make_player(1, laning_snapshots=[
        LaningSnapshot(time=0, x=-6600.0, y=-6400.0),
        LaningSnapshot(time=5, x=-6590.0, y=-6390.0),
        LaningSnapshot(time=10, x=100.0, y=200.0),
        LaningSnapshot(time=605, x=5000.0, y=5000.0),
    ])
    nothing = make_player(2)
    rows = analyzer.analyze_laning(make_match([od, replay, nothing]))["players"]
    assert rows[0]["lane_pos"] == [[-6592.0, -6336.0, 36], [-6592.0, -6208.0, 2], [64.0, 64.0, 5]]
    assert rows[1]["lane_pos"] == [[-6592.0, -6336.0, 2], [64.0, 192.0, 1]]
    assert rows[2]["lane_pos"] == []
    for x, y, _ in rows[0]["lane_pos"] + rows[1]["lane_pos"]:
        assert 0 <= analyzer.geo.world_to_pct(x) <= 100 and 0 <= analyzer.geo.world_to_pct(y) <= 100


def test_first_blood_objective_names_the_victim():
    players = [make_player(0), make_player(128)]
    match = make_match(players, objectives=[
        {"time": 90, "type": "CHAT_MESSAGE_FIRSTBLOOD", "player_slot": 128, "key": "0"},
        {"time": 95, "type": "CHAT_MESSAGE_FIRSTBLOOD", "player_slot": 128, "key": "hoodwink"},   # pre-contract data
        {"time": 99, "type": "building_kill", "key": "npc_dota_goodguys_tower1_top"},
    ])
    objs = analyzer.analyze_objectives(match)["objectives"]
    assert objs[0]["hero_name"] == "Hero128" and objs[0]["team"] == "dire"
    assert (objs[0]["victim_slot"], objs[0]["victim_hero_name"], objs[0]["victim_hero_slug"]) == (0, "Hero0", "hero0")
    assert (objs[1]["victim_slot"], objs[1]["victim_hero_name"]) == (None, "")
    assert "victim_slot" not in objs[2]


def test_position_log_is_not_part_of_the_payload():
    p = make_player(0, position_log=[[0, 987654, 876543, 100], [5, 987655, 876544, 0]])
    text = json.dumps(full_analysis(make_match([p])))
    assert "987654" not in text and "position_log" not in text


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


# ---------------------------------------------------------------------------
# Courier kills, starting items, pre-game-only replays, patch label, extra data
# ---------------------------------------------------------------------------

def test_courier_kill_is_credited_to_the_killers():
    """The source's team is the side that LOST the courier; player_slot is the enemy killer."""
    players = [make_player(0), make_player(128)]
    match = make_match(players, objectives=[
        {"time": 647, "type": "CHAT_MESSAGE_COURIER_LOST", "team": "dire", "player_slot": 0, "value": 45},
        {"time": 700, "type": "CHAT_MESSAGE_COURIER_LOST", "team": "radiant"},              # creeps / tower
        {"time": 800, "type": "CHAT_MESSAGE_COURIER_LOST", "player_slot": 128, "value": 30},  # no team given
        {"time": 900, "type": "CHAT_MESSAGE_COURIER_LOST", "team": 3, "value": True},        # Valve team id
        {"time": 950, "type": "CHAT_MESSAGE_COURIER_LOST"},
        {"time": 990, "type": "CHAT_MESSAGE_ROSHAN_KILL", "team": "dire", "value": 5},
    ])
    objs = analyzer.analyze_objectives(match)["objectives"]
    assert [(o["team"], o["courier_team"], o["hero_name"], o["value"]) for o in objs[:5]] == [
        ("radiant", "dire", "Hero0", 45),
        ("dire", "radiant", "", None),
        ("dire", "radiant", "Hero128", 30),
        ("radiant", "dire", "", None),
        (None, None, "", None),
    ]
    # other objectives keep "team" = the side that took it and get no courier fields
    assert objs[5]["team"] == "dire" and "courier_team" not in objs[5] and "value" not in objs[5]


def test_starting_items_are_everything_bought_before_the_horn():
    """Strategy-time purchases carry the pre-game start, which is not always -90."""
    p = make_player(0, purchase_log=[
        {"time": -95, "key": "tango"}, {"time": -90, "key": "branches"}, {"time": -1, "key": "ward_observer"},
        {"time": 0, "key": "clarity"}, {"time": 1, "key": "flask"}, {"time": 300, "key": "boots"}])
    phases = analyzer.analyze_purchases(make_match([p], duration=600))["players"][0]["phases"]
    assert [i["key"] for i in phases["starting"]] == ["tango", "branches", "ward_observer", "clarity"]
    assert [i["key"] for i in phases["0-5"]] == ["flask", "boots"]
    assert sum(len(items) for items in phases.values()) == 6          # every purchase in exactly one phase


def test_replay_cut_before_the_horn_is_reported_as_pre_game_only():
    """Truncated before 0:00: duration 0, no winner, every time <= 0, empty minute series."""
    p0 = make_player(0, purchase_log=[{"time": -90, "key": "tango"}],
                     wards=[WardEvent(ward_type="observer", x=0, y=0, time=-56, player_slot=0, team="radiant")])
    match = make_match([p0, make_player(128)], duration=0, partial=True, winner_known=False, radiant_win=False,
                       wards=p0.wards, match_id=8762775364, game_build=6762)
    result = full_analysis(match)
    overview = result["match_overview"]
    assert overview["pre_game_only"] is True and overview["partial"] is True
    assert (overview["winner"], overview["radiant_win"], overview["duration"]) == (None, None, 0)
    assert (overview["match_id"], overview["game_build"], overview["patch"]) == (8762775364, 6762, "")
    # what did happen is still there; nothing pretends the game reached 0:00
    assert len(result["wards"]["all_wards"]) == 1 and result["wards"]["ward_density"] == []
    assert list(result["purchases"]["players"][0]["phases"]) == ["starting"]
    laning = result["laning"]["players"][0]
    assert laning["gold_ten"] is None and laning["xp_ten"] is None and laning["lane_efficiency_pct"] == 0
    assert result["gold_xp_timeline"] == [] and result["graphs"]["radiant_gold_adv"] == []
    json.dumps(result, allow_nan=False)

    # A truncated replay that did start, and a complete match of unknown length, are not "pre-game only"
    assert full_analysis(make_match(duration=126, partial=True, winner_known=False))["match_overview"][
        "pre_game_only"] is False
    assert full_analysis(make_match(duration=0))["match_overview"]["pre_game_only"] is False
    # ... but the coach has nothing to review in a recording of the first minutes either
    assert overview["coach_available"] is False
    for duration, partial, available in ((4, True, False), (133, True, False), (299, True, False), (300, True, True),
                                         (659, True, True), (0, False, True), (126, False, True)):
        made = make_match(duration=duration, partial=partial, winner_known=not partial)
        assert full_analysis(made)["match_overview"]["coach_available"] is available, (duration, partial)
    from backend.app import analyzer
    from backend.app.ai import coach
    assert analyzer.COACH_MIN_PARTIAL_SECONDS == coach.MIN_PARTIAL_GAME_SECONDS == 300
    for duration, partial in ((0, True), (133, True), (300, True), (133, False)):
        made = make_match(duration=duration, partial=partial)
        assert coach.has_no_game(made) is (not analyzer._coach_available(made))


def test_a_replay_cut_within_the_first_minute_has_no_per_minute_rates():
    """cut_2mb.dem ends 4 s after the horn: the scoreboard read "621/0" under GPM/XPM for all ten
    players, the gold of four seconds extrapolated to a minute."""
    players = [make_player(0, gold_per_min=621, xp_per_min=0, net_worth=648), make_player(128, gold_per_min=621)]
    for duration, partial, rates in ((0, True, False), (4, True, False), (59, True, False), (60, True, True),
                                     (133, True, True), (1800, False, True), (0, False, True)):
        match = make_match(players, duration=duration, partial=partial, winner_known=not partial)
        rows = full_analysis(match)["match_overview"]["players"]
        assert [(r["gpm"], r["xpm"]) for r in rows] == ([(621, 0), (621, 0)] if rates else [(None, None)] * 2), (
            duration, partial)
        assert rows[0]["net_worth"] == 648                         # what the players own is still shown


def test_patch_label_comes_from_the_match_or_its_start_time(tmp_path, monkeypatch):
    from backend.app.ai import data_manager

    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", tmp_path)
    replay = dict(start_time=1775643322)                               # 2026-04-08, no patch in a replay
    assert full_analysis(make_match(**replay))["match_overview"]["patch"] == ""      # no patch table yet
    (tmp_path / data_manager.PATCHES_FILE).write_text(json.dumps({
        "fetched_at": "2026-04-10T00:00:00Z",
        "patches": [{"name": "7.40", "date": "2025-12-16T00:50:40.281Z"},
                    {"name": "7.41", "date": "2026-03-24T00:50:59.580Z"}]}))
    assert full_analysis(make_match(**replay))["match_overview"]["patch"] == "7.41"
    assert full_analysis(make_match(start_time=1772470634))["match_overview"]["patch"] == "7.40"
    # the source's own label wins; an unknown start time gives none
    assert full_analysis(make_match(patch="7.39", **replay))["match_overview"]["patch"] == "7.39"
    assert full_analysis(make_match(start_time=0))["match_overview"]["patch"] == ""
    # a broken data layer never breaks the analysis
    monkeypatch.setitem(__import__("sys").modules, "backend.app.ai.data_manager", None)
    assert full_analysis(make_match(**replay))["match_overview"]["patch"] == ""


def test_patch_label_carries_the_letter_patch_when_valves_release_list_is_known(tmp_path, monkeypatch):
    """7.41b, not 7.41, for a replay of 2026-04-08; the major version stays what the talents are compared by."""
    from backend.app.ai import data_manager

    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", tmp_path)
    majors = {"fetched_at": "2026-04-10T00:00:00Z",
              "patches": [{"name": "7.40", "date": "2025-12-16T00:50:40Z"},
                          {"name": "7.41", "date": "2026-03-24T00:50:59Z"}]}
    releases = {"releases_fetched_at": "2026-04-10T00:00:00Z",
                "releases": [{"name": "7.40c", "date": "2026-01-21T08:00:00Z"},
                             {"name": "7.41", "date": "2026-03-24T08:00:00Z"},
                             {"name": "7.41a", "date": "2026-03-27T08:00:00Z"},
                             {"name": "7.41b", "date": "2026-04-07T08:00:00Z"}]}

    def patch_of(**match) -> str:
        return full_analysis(make_match(**match))["match_overview"]["patch"]

    replay = dict(start_time=1775643322)                               # 2026-04-08
    (tmp_path / data_manager.PATCHES_FILE).write_text(json.dumps({**majors, **releases}))
    assert patch_of(**replay) == "7.41b"
    assert patch_of(start_time=1772470634) == "7.40c"                  # 2026-03-02
    assert patch_of(patch="7.41", **replay) == "7.41b"                 # OpenDota names the major version only
    assert patch_of(patch="7.41f", **replay) == "7.41f"                # a source that names the letter itself
    # The two lists disagree in the hours between their release times: the major version wins.
    assert patch_of(start_time=1774314000) == "7.41"                   # 2026-03-24 01:00 UTC
    assert patch_of(patch="7.39", **replay) == "7.39"
    assert analyzer._patch_versions(make_match(**replay)) == ("7.41", "7.41b")
    # Only Valve's list on disk: it names the version on its own.
    (tmp_path / data_manager.PATCHES_FILE).write_text(json.dumps(releases))
    assert analyzer._patch_versions(make_match(**replay)) == ("7.41", "7.41b")
    # Only the major versions (game data fetched before the feed was read): as before.
    (tmp_path / data_manager.PATCHES_FILE).write_text(json.dumps(majors))
    assert patch_of(**replay) == "7.41"


def test_stored_talent_names_are_compared_by_major_version(tmp_path, monkeypatch):
    """A letter patch in the label must not make the stored names look foreign."""
    from backend.app import parser
    from backend.app.ai import data_manager

    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", tmp_path)
    (tmp_path / data_manager.PATCHES_FILE).write_text(json.dumps({
        "fetched_at": "2026-04-10T00:00:00Z",
        "patches": [{"name": "7.41", "date": "2026-03-24T00:50:59Z"}],
        "releases_fetched_at": "2026-04-10T00:00:00Z",
        "releases": [{"name": "7.41", "date": "2026-03-24T08:00:00Z"},
                     {"name": "7.41b", "date": "2026-04-07T08:00:00Z"}]}))
    monkeypatch.setattr(parser, "talent_names_patch", lambda: "7.42")  # the game data has moved on
    match = make_match(start_time=1775643322, talent_names_patch="7.41")
    assert full_analysis(match)["match_overview"]["patch"] == "7.41b"
    assert analyzer.keeps_stored_talent_names(match) is True
    assert analyzer.keeps_stored_talent_names(match.model_copy(update={"talent_names_patch": "7.40"})) is False


def test_payload_exposes_buybacks_streaks_dewarders_and_smoke_users():
    carry = make_player(0, buyback_count=2, buyback_log=[{"time": 1831}, {"time": 2000}],
                        multi_kills={"2": 3, "3": 1}, kill_streaks={"3": 1}, stuns=12.5,
                        actions_per_min=310, teamfight_participation=0.75)
    support = make_player(128)
    match = make_match(
        [carry, support], game_build=6762,
        wards=[
            WardEvent(ward_type="observer", x=0, y=0, time=10, player_slot=128, destroyed_time=60,
                      dewarded=True, killer_slot=0),
            WardEvent(ward_type="sentry", x=0, y=0, time=10, player_slot=128, destroyed_time=70, dewarded=True),
            WardEvent(ward_type="observer", x=0, y=0, time=20, player_slot=0, destroyed_time=380),
            # a killer on a ward that was not dewarded is not shown (inconsistent source data)
            WardEvent(ward_type="observer", x=0, y=0, time=30, player_slot=0, destroyed_time=390, killer_slot=128),
        ],
        smokes=[SmokeEvent(time=300, team="radiant", player_slots=[0], player_slot=0),
                SmokeEvent(time=400, team="dire", player_slots=[128])],
        teamfights=[TeamfightEvent(start_time=600, end_time=640, radiant_deaths=0, dire_deaths=0,
                                   gold_delta=-599, xp_delta=404, players=[])],
    )
    result = full_analysis(match)
    row = result["match_overview"]["players"][0]
    assert row["buyback_count"] == 2
    assert row["buyback_log"] == [{"time": 1831, "time_str": "30:31"}, {"time": 2000, "time_str": "33:20"}]
    assert (row["multi_kills"], row["kill_streaks"]) == ({"2": 3, "3": 1}, {"3": 1})
    assert (row["stuns"], row["actions_per_min"], row["teamfight_participation"]) == (12.5, 310, 0.75)
    other = result["match_overview"]["players"][1]
    assert (other["buyback_count"], other["buyback_log"], other["multi_kills"]) == (0, [], {})
    assert result["match_overview"]["game_build"] == 6762

    wards = result["wards"]
    assert [(w["killer_slot"], w["killer_hero_name"], w["killer_hero_slug"]) for w in wards["all_wards"]] == [
        (0, "Hero0", "hero0"), (None, "", ""), (None, "", ""), (None, "", "")]
    assert wards["dire"]["observers"][0]["killer_hero_name"] == "Hero0"
    assert wards["dire"]["sentries"][0]["killer_slot"] is None

    timeline = result["smokes"]["timeline"]
    assert [(s["player_slot"], s["hero_name"], s["hero_slug"]) for s in timeline] == [
        (0, "Hero0", "hero0"), (None, "", "")]
    assert (result["teamfights"][0]["gold_delta"], result["teamfights"][0]["xp_delta"]) == (-599, 404)
    json.dumps(result, allow_nan=False)


# ---------------------------------------------------------------------------
# Frontend fix pass: what the tabs showed wrongly because of the payload
# ---------------------------------------------------------------------------

def test_a_ward_that_fell_early_without_an_enemy_did_not_expire():
    """Demo 16:42: a sentry gone after 18 s, ``dewarded`` false, was labelled "Expired"."""
    wards = [
        WardEvent(ward_type="sentry", x=0, y=0, time=1002, player_slot=0, destroyed_time=1020),              # denied
        WardEvent(ward_type="sentry", x=0, y=0, time=100, player_slot=0, destroyed_time=520),               # 7:00
        WardEvent(ward_type="observer", x=0, y=0, time=100, player_slot=0, destroyed_time=457),             # ~6:00
        WardEvent(ward_type="observer", x=0, y=0, time=100, player_slot=0, destroyed_time=454),             # 6 s short
        WardEvent(ward_type="observer", x=0, y=0, time=100, player_slot=128, destroyed_time=150, dewarded=True),
        WardEvent(ward_type="sentry", x=0, y=0, time=1700, player_slot=128),                                # still up
    ]
    result = analyzer.analyze_wards(make_match([make_player(0), make_player(128)], wards=wards))
    assert [w["state"] for w in result["all_wards"]] == [
        "destroyed", "expired", "expired", "destroyed", "dewarded", "active"]
    assert [w["state"] for w in result["radiant"]["sentries"]] == ["destroyed", "expired"]
    # "dewarded" stays the parser's word (an enemy did it): the counters do not guess
    assert result["radiant"]["dewarded_sent"] == 0 and result["dire"]["dewarded_obs"] == 1


def test_unit_kills_a_scoreboard_only_source_cannot_break_down_are_unknown():
    """An OpenDota match that was never parsed: 193 last hits and "0 creeps, 0 neutrals" in the farm tab."""
    basic = make_match([make_player(0, kills=6, last_hits=193), make_player(128, kills=2, last_hits=40)])
    assert analyzer.analyze_farm(basic)["unit_kills"][0] == {
        "player_slot": 0, "hero_name": "Hero0", "hero_slug": "hero0", "team": "radiant", "hero_kills": 6,
        "lane_kills": None, "neutral_kills": None, "ancient_kills": None, "tower_kills": None,
        "courier_kills": None, "roshans_killed": None}
    # one count anywhere means the source has the breakdown: a zero is a zero
    parsed = make_match([make_player(0, kills=6, last_hits=193, lane_kills=150),
                         make_player(128, kills=2, last_hits=40)])
    rows = analyzer.analyze_farm(parsed)["unit_kills"]
    assert (rows[0]["lane_kills"], rows[0]["neutral_kills"], rows[1]["lane_kills"]) == (150, 0, 0)
    # nobody has a last hit (a replay cut before the horn): real zeros as well
    assert analyzer.analyze_farm(make_match([make_player(0)]))["unit_kills"][0]["lane_kills"] == 0
    json.dumps(analyzer.analyze_farm(basic), allow_nan=False)


def test_skill_build_says_which_ability_is_the_ultimate(monkeypatch):
    """Turbo Skywrath Mage: Mystic Flare was skilled at levels the frontend's guess did not take
    for an ultimate's, and its row was drawn as a basic ability."""
    from backend.app.ai import game_data

    known = {"skywrath_mage_mystic_flare": {"is_ultimate": True}, "skywrath_mage_arcane_bolt": {}}
    monkeypatch.setattr(game_data, "get_ability_by_key",
                        lambda key: ("Skywrath Mage", key, known[key]) if key in known else None)
    skills = [SkillEvent(ability_name=name, ability_key=key, level=1, time=60 * n)
              for n, (name, key) in enumerate([
                  ("Arcane Bolt", "skywrath_mage_arcane_bolt"), ("Mystic Flare", "skywrath_mage_mystic_flare"),
                  ("+8 Intelligence", "special_bonus_intelligence_8"), ("Attribute Bonus", "special_bonus_attributes"),
                  ("Something New", "brand_new_ability")])]
    order = analyzer.analyze_skill_builds(make_match([make_player(0, skills=skills)]))[0]["skill_order"]
    # None: a talent, the attribute bonus, and an ability the game data does not know
    assert [s["is_ultimate"] for s in order] == [False, True, None, None, None]

    def broken(key):
        raise RuntimeError("data layer down")
    monkeypatch.setattr(game_data, "get_ability_by_key", broken)
    order = analyzer.analyze_skill_builds(make_match([make_player(0, skills=skills)]))[0]["skill_order"]
    assert [s["is_ultimate"] for s in order] == [None] * 5


# ---------------------------------------------------------------------------
# Polish pass: positions, ward denies, respawns in fights, chat-wheel phrases
# ---------------------------------------------------------------------------

def test_a_denied_ward_names_the_hero_of_its_own_side():
    """A ward its own team attacked was "destroyed", like one nothing is known about."""
    wards = [
        WardEvent(ward_type="sentry", x=0, y=0, time=1002, player_slot=0, destroyed_time=1020,
                  denied=True, killer_slot=1),                                        # an ally attacked it
        WardEvent(ward_type="observer", x=0, y=0, time=100, player_slot=0, destroyed_time=150, denied=True),
        WardEvent(ward_type="observer", x=0, y=0, time=100, player_slot=128, destroyed_time=150,
                  dewarded=True, killer_slot=0),
        WardEvent(ward_type="sentry", x=0, y=0, time=10, player_slot=128, destroyed_time=30),    # the source cannot tell
        WardEvent(ward_type="sentry", x=0, y=0, time=10, player_slot=128, destroyed_time=430, killer_slot=129),
    ]
    result = analyzer.analyze_wards(make_match([make_player(0), make_player(1), make_player(128), make_player(129)],
                                               wards=wards))
    listed = result["all_wards"]
    assert [w["state"] for w in listed] == ["denied", "denied", "dewarded", "destroyed", "expired"]
    assert [(w["denied"], w["dewarded"]) for w in listed] == [
        (True, False), (True, False), (False, True), (False, False), (False, False)]
    # the hero that ended it: an ally for a deny, an enemy for a deward, nobody without either flag
    assert [(w["killer_slot"], w["killer_hero_name"], w["killer_hero_slug"]) for w in listed] == [
        (1, "Hero1", "hero1"), (None, "", ""), (0, "Hero0", "hero0"), (None, "", ""), (None, "", "")]
    mine = result["radiant"]["sentries"][0]
    assert (mine["state"], mine["denied"], mine["killer_hero_name"]) == ("denied", True, "Hero1")
    # a deny is no deward: the counters stay the enemy's work
    assert (result["radiant"]["dewarded_sent"], result["radiant"]["dewarded_obs"], result["dire"]["dewarded_obs"]) == (0, 0, 1)


def _log(alive, start=560, end=700):
    """A 5 s position log: health on the samples inside one of the ``alive`` spans."""
    return [[t, 0, 0, 80 if any(a <= t < b for a, b in alive) else 0] for t in range(start, end + 1, 5)]


def test_fight_players_carry_the_times_they_were_alive():
    """Playback knew deaths and buybacks only: a hero that respawned inside a long fight
    window stayed faded to its end, and one still dead from before it looked alive."""
    players = [
        # dies in the fight and is back before its end (analyzer 1.5.0 names the respawn)
        make_player(0, deaths=1, position_log=_log([(560, 610), (640, 701)])),
        # died before the fight, walks in late
        make_player(1, deaths=1, position_log=_log([(560, 590), (615, 701)])),
        # dies, buys back at once, dies again and stays down
        make_player(2, deaths=2, buyback_log=[{"time": 625}], position_log=_log([(560, 620), (625, 650)])),
        # no respawn time on the death (older analyzer): the first sample that shows health again
        make_player(3, deaths=1, position_log=_log([(560, 612), (633, 701)])),
        # never dies / dead the whole window / a reincarnation is no death
        make_player(4, position_log=_log([(560, 701)])),
        make_player(128, deaths=1, position_log=_log([(560, 595)])),
        make_player(129, position_log=_log([(560, 630), (635, 701)])),
        # the source cannot tell: no positions (OpenDota), no health in them, a death the log misses
        make_player(130, deaths=1),
        make_player(131, deaths=1, position_log=[[t, 0, 0, -1] for t in range(560, 701, 5)]),
        make_player(132, deaths=2, position_log=_log([(560, 610), (640, 701)])),
    ]
    kills = [
        KillEvent(time=590.0, victim_slot=1, killer_slot=128, respawn_time=615.0),
        KillEvent(time=595.0, victim_slot=128, killer_slot=0),
        KillEvent(time=610.4, victim_slot=0, killer_slot=128, respawn_time=640.0),
        KillEvent(time=610.4, victim_slot=132, killer_slot=0, respawn_time=640.0),
        KillEvent(time=612.3, victim_slot=3, killer_slot=128),
        KillEvent(time=612.3, victim_slot=130, killer_slot=0, respawn_time=640.0),
        KillEvent(time=612.3, victim_slot=131, killer_slot=0),
        KillEvent(time=620.0, victim_slot=2, killer_slot=128, respawn_time=625.9),
        KillEvent(time=650.0, victim_slot=2, killer_slot=128),
    ]
    fight = TeamfightEvent(start_time=600, end_time=660, radiant_deaths=4, dire_deaths=4, players=[
        TeamfightPlayer(player_slot=p.player_slot, hero_name=p.hero_name, x=0, y=0, damage_dealt=0, damage_taken=0,
                        kills=0, deaths=0, abilities_used=[]) for p in players])
    result = analyzer.analyze_teamfights(make_match(players, kills_log=kills, teamfights=[fight]))
    alive = {p["player_slot"]: p["alive_intervals"] for p in result[0]["players"]}
    assert alive == {
        0: [[0.0, 10.4], [40.0, 60.0]],
        1: [[15.0, 60.0]],
        2: [[0.0, 20.0], [25.0, 50.0]],         # the buyback (exact), not the sample a second later
        3: [[0.0, 12.3], [35.0, 60.0]],         # 635: the 5 s log, up to that much late
        4: [[0.0, 60.0]],
        128: [],
        129: [[0.0, 60.0]],
        130: None, 131: None, 132: None,
    }
    json.dumps(result, allow_nan=False)
    # the mock has respawns too; OpenDota-like data (no position log) says null for everyone
    from backend.app.mock_data import generate_mock_match
    mock = generate_mock_match()
    fights = analyzer.analyze_teamfights(mock)
    assert all(p["alive_intervals"] is not None for tf in fights for p in tf["players"])
    assert any(len(p["alive_intervals"]) == 2 for tf in fights for p in tf["players"])
    for tf in fights:
        for p in tf["players"]:
            flat = [edge for span in p["alive_intervals"] for edge in span]
            assert flat == sorted(flat) and all(0 <= edge <= tf["duration"] for edge in flat), (tf["index"], p)
            if p["death_time_offset"] is not None:      # a death in the fight ends a span
                assert any(abs(span[1] - p["death_time_offset"]) < 0.11 for span in p["alive_intervals"]), p
    for player in mock.players:
        player.position_log = []
    assert all(p["alive_intervals"] is None for tf in analyzer.analyze_teamfights(mock) for p in tf["players"])


def test_overview_players_carry_the_position_the_coach_derives(monkeypatch):
    from backend.app.ai import positions
    from backend.app.mock_data import generate_mock_match

    match = generate_mock_match()
    rows = full_analysis(match)["match_overview"]["players"]
    derived = positions.derive_match_positions(match.players)       # what a coach report is built on
    assert len(rows) == 10
    for row in rows:
        mine = derived[row["player_slot"]]
        assert row["position"] == {"number": mine.number, "confidence": mine.confidence, "method": mine.method}
        assert row["position"]["confidence"] in ("high", "medium", "low")
        assert row["position"]["method"] in ("lane_nw", "archetype_nw", "nw_only")
    for team in ("radiant", "dire"):
        assert sorted(row["position"]["number"] for row in rows if row["team"] == team) == [1, 2, 3, 4, 5]

    # a replay that ends before the horn or in the first minutes has no lanes and no farm to
    # derive a role from (7.41b replay cut 4 s after the horn: three of five dire roles were
    # guessed wrong); the coach does not review it either
    for duration, known in ((0, False), (4, False), (299, False), (300, True)):
        cut = generate_mock_match()
        cut.partial, cut.duration, cut.winner_known = True, duration, False
        overview = full_analysis(cut)["match_overview"]
        assert overview["coach_available"] is known
        assert all((row["position"] is not None) is known for row in overview["players"]), duration

    # the AI layer never breaks the analysis: missing, failing or answering nonsense
    def broken(_match):
        raise RuntimeError("knowledge base unavailable")

    monkeypatch.setattr(positions, "match_positions_payload", broken)
    assert [row["position"] for row in full_analysis(match)["match_overview"]["players"]] == [None] * 10
    monkeypatch.setattr(positions, "match_positions_payload", lambda _match: {0: {"number": 7}, 1: "carry", 2: {
        "number": 3, "confidence": "low", "method": "nw_only", "label_zh": "3号位 (offlaner)"}})
    answered = {row["player_slot"]: row["position"] for row in full_analysis(match)["match_overview"]["players"]}
    assert answered[0] is None and answered[1] is None and answered[128] is None
    assert answered[2] == {"number": 3, "confidence": "low", "method": "nw_only"}
    monkeypatch.setitem(__import__("sys").modules, "backend.app.ai.positions", None)   # the import itself fails
    assert [row["position"] for row in full_analysis(match)["match_overview"]["players"]] == [None] * 10


def test_chat_wheel_messages_carry_their_phrase_in_both_languages():
    """A chat-wheel line was {type: "chatwheel", key: "chatwheel_<id>"} and the page showed "#71"."""
    players = [make_player(0), make_player(128, hero_id=22, hero_name="Zeus", hero_slug="zuus")]
    match = make_match(players, chat=[
        {"time": 10, "type": "chat", "key": "gl hf", "player_slot": 0},
        {"time": 20, "type": "chatwheel", "key": "chatwheel_2", "player_slot": 0},
        # "<hero> is missing!": the replay parser says which hero; OpenDota does not
        {"time": 30, "type": "chatwheel", "key": "chatwheel_71", "player_slot": 0, "hero_id": 22},
        {"time": 40, "type": "chatwheel", "key": "chatwheel_71", "player_slot": 0},
        {"time": 45, "type": "chatwheel", "key": "chatwheel_71", "player_slot": 0, "hero_id": 99999},
        {"time": 50, "type": "chatwheel", "key": "chatwheel_12001", "player_slot": 128},        # a hero voice line
        {"time": 60, "type": "chatwheel", "key": "chatwheel_805", "player_slot": 128},          # a spray
        {"time": 70, "type": "chatwheel", "key": "chatwheel_99999999", "player_slot": 128},     # not in the catalog
        {"time": 80, "type": "chatwheel", "key": "", "player_slot": 128},
    ])
    chat, okay, named, unnamed, stranger, voice, spray, unknown, empty = analyzer.analyze_chat(match)["messages"]
    assert "phrase" not in chat and chat["key"] == "gl hf"
    assert (okay["key"], okay["phrase"], okay["sound"], okay["spray"]) == (
        "chatwheel_2", {"en": "Get Back!", "zh": "撤退！"}, False, False)
    assert named["phrase"] == {"en": "Zeus is missing!", "zh": "宙斯不见了！"}
    assert unnamed["phrase"] == stranger["phrase"] == {"en": "… is missing!", "zh": "…不见了！"}
    assert voice["sound"] is True and voice["spray"] is False
    assert voice["phrase"]["en"].startswith("Ha ha") and voice["phrase"]["zh"].startswith("哈哈")
    assert (spray["phrase"], spray["sound"], spray["spray"]) == (None, False, True)
    for message in (unknown, empty):
        assert (message["phrase"], message["sound"], message["spray"]) == (None, False, False)
    json.dumps(analyzer.analyze_chat(match), allow_nan=False)
