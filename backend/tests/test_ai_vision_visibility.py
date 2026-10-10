"""Enemy visibility states, danger and the vision timeline (vision.py, contract A4).

An enemy is "visible", in the "fog" (position known, outside vision),
"unknown" (no position data: never missing) or "dead".
"""
from __future__ import annotations

from backend.app import geo
from backend.app.ai import vision
from backend.app.models import KillEvent, TeamfightEvent, TeamfightPlayer, WardEvent
from test_ai_pipeline_helpers import CJK_RE, default_players, make_match, observer

# Far from every tower: only heroes and wards give vision here.
SPOT = (1500.0, -3500.0)
RADIANT_BASE = (-6500.0, -6000.0)
DIRE_BASE = (6500.0, 6000.0)


def stand(pos: tuple[float, float], start: int, end: int, hp: int = 100) -> list[list[int]]:
    return [[t, int(pos[0]), int(pos[1]), hp] for t in range(start, end + 1, 5)]


def offset(dx: float, dy: float = 0.0) -> tuple[float, float]:
    return (SPOT[0] + dx, SPOT[1] + dy)


def logged_match(logs: dict[int, list[list[int]]], start: int = 500, end: int = 900, **kw):
    """Juggernaut (slot 0) stands at SPOT, everybody else in their base, unless ``logs`` says otherwise.

    A slot mapped to ``[]`` has no position data at all.
    """
    players = default_players()
    for p in players:
        p.laning_snapshots = []
        home = RADIANT_BASE if p.team == "radiant" else DIRE_BASE
        p.position_log = stand(SPOT if p.player_slot == 0 else home, start, end)
        if p.player_slot in logs:
            p.position_log = logs[p.player_slot]
    return make_match(players, **kw)


def states_at(match, t: int, team: str = "radiant") -> dict:
    slot = 0 if team == "radiant" else 128
    return {e.hero_name: e for e in vision._track_enemy_visibility(match, team, slot, t)}


# ---------------------------------------------------------------------------
# The four states
# ---------------------------------------------------------------------------

def _four_state_match():
    return logged_match({
        128: stand(offset(1000), 500, 900),             # inside Juggernaut's vision
        129: stand(offset(3000), 500, 900),             # known, outside vision, close
        130: [],                                        # no position data
        131: stand(offset(500), 500, 900, hp=0),        # dead
    })                                                  # 132: in the Dire base, far away


def test_enemy_states_visible_fog_unknown_dead():
    vis = states_at(_four_state_match(), 700)
    assert {name: e.state for name, e in vis.items()} == {
        "Phantom Assassin": "visible", "Lina": "fog", "Axe": "unknown",
        "Lion": "dead", "Witch Doctor": "fog",
    }
    assert vis["Phantom Assassin"].is_visible and vis["Phantom Assassin"].last_seen_source == "hero"
    assert not any(vis[name].is_visible for name in ("Lina", "Axe", "Lion", "Witch Doctor"))
    assert vis["Axe"].seconds_missing == 0 and vis["Lion"].seconds_missing == 0


def test_missing_counts_cover_fog_only_and_unknown_is_listed_apart():
    va = vision.analyze_vision(_four_state_match(), 0, 700, SPOT, lang="en")
    state = va.state_at_decision
    assert state.missing_enemy_count == 2
    assert sorted(state.missing_enemy_names) == ["Lina", "Witch Doctor"]
    assert state.unknown_enemy_count == 1 and state.unknown_enemy_names == ["Axe"]
    text = va.formatted_text
    assert "- Axe: no position data (not counted as missing)" in text
    assert "- Lion: dead" in text
    # In the fog for the whole lookback: "over", and where it really was.
    assert "- Lina: in the fog for over 120s (not visible to the player's team); actual position:" in text
    assert "about 3000 units from the player" in text
    assert "no position for 1 enemy hero(es)" in text
    assert not CJK_RE.search(text)
    zh = vision.analyze_vision(_four_state_match(), 0, 700, SPOT, lang="zh").formatted_text
    assert "- Axe: 无位置数据（不计入消失）" in zh and "- Lion: 阵亡" in zh


def test_only_fog_enemies_within_reach_raise_the_danger():
    match = _four_state_match()
    vis = vision._track_enemy_visibility(match, "radiant", 0, 700)
    threats = vision.fog_threats(match, vis, SPOT, 700)
    assert [e.hero_name for e in threats] == ["Lina"]       # Witch Doctor is 10,000 units away
    state = vision.analyze_vision(match, 0, 700, SPOT, lang="en").state_at_decision
    reasons = " | ".join(state.danger_reasons)
    assert "Lina hidden in the fog" in reasons
    assert "Witch Doctor" not in reasons and "Axe" not in reasons
    # Without a player position every fog enemy counts, an unknown one never.
    assert len(vision.fog_threats(match, vis, (0.0, 0.0), 700)) == 2
    # The state carries the same subset for text about the moment (AP56: the rule text named
    # every fog enemy on the map as "out of sight" at a death).
    assert (state.threat_enemy_count, state.threat_enemy_names) == (1, ["Lina"])
    assert state.threat_longest_seconds == vis[1].seconds_missing > 0
    assert sorted(state.missing_enemy_names) == ["Lina", "Witch Doctor"]


def test_unknown_enemies_produce_no_danger_events_or_critical_moment():
    # OpenDota-like: a kill log, no positions for anybody.
    players = default_players()
    for p in players:
        p.laning_snapshots = []
    match = make_match(players, kills_log=[KillEvent(time=100.0, victim_slot=3, killer_slot=128)])
    va = vision.analyze_vision(match, 0, 1000, (3000.0, 3000.0), lang="en")
    state = va.state_at_decision
    assert state.missing_enemy_count == 0 and state.longest_missing_seconds == 0
    assert state.unknown_enemy_count == 5
    assert all("fog" not in r and "missing" not in r for r in state.danger_reasons)
    assert not [e for e in va.timeline.events if e.event_type in ("enemy_missing", "enemy_seen")]
    assert va.timeline.critical_description == ""
    # Said once for a source without positions: five "no position data" lines read like five
    # enemies missing (live verdicts: "four enemies unaccounted for on the minimap").
    assert "no position data (not counted as missing)" not in va.formatted_text
    assert "- the other 5 enemy hero(es): this match's data has no positions" in va.formatted_text
    assert "cannot be told (this match's data has no positions)" in va.formatted_text
    assert "did not retreat" not in va.formatted_text
    # A replay in which one enemy has no data keeps naming that enemy.
    partly = vision.analyze_vision(logged_match({130: []}), 0, 700, SPOT, lang="en").formatted_text
    assert partly.count("no position data (not counted as missing)") == 1


# ---------------------------------------------------------------------------
# Fog duration, creep vision, fights
# ---------------------------------------------------------------------------

def test_fog_duration_is_measured_from_the_last_sighting():
    match = logged_match({129: stand(offset(1000), 500, 700) + stand(offset(3000), 705, 900)})
    lina = states_at(match, 730)["Lina"]
    assert lina.state == "fog" and lina.seconds_missing == 30
    assert lina.last_seen_time == 700 and lina.last_seen_source == "hero"
    assert (lina.last_seen_x, lina.last_seen_y) == offset(1000)
    text = vision.analyze_vision(match, 0, 730, SPOT, lang="en").formatted_text
    assert "- Lina: in the fog for 30s (not visible to the player's team); last seen at" in text


def test_fog_duration_is_never_made_up_when_the_data_starts_late():
    match = logged_match({129: stand(offset(3000), 720, 900)})
    lina = states_at(match, 730)["Lina"]
    assert lina.state == "fog"
    assert lina.seconds_missing == 10                 # known for 10 s, not "120 s"
    assert lina.last_seen_source == "unknown" and lina.last_seen_time == 0
    assert states_at(match, 715)["Lina"].state == "unknown"


def test_lane_opponent_stays_visible_at_night_while_laning():
    match = logged_match({128: stand(offset(1500), 300, 1200)}, start=300, end=1200)
    assert vision._is_night(400) and vision._is_night(1000)
    assert states_at(match, 400)["Phantom Assassin"].state == "visible"   # allied creep wave
    assert states_at(match, 1000)["Phantom Assassin"].state == "fog"      # hero night vision is 800


def _fight(*players: TeamfightPlayer, start: int = 690, end: int = 720) -> TeamfightEvent:
    return TeamfightEvent(start_time=start, end_time=end, radiant_deaths=0, dire_deaths=0,
                          players=list(players))


def _fighter(slot: int, name: str, **kw) -> TeamfightPlayer:
    data = dict(player_slot=slot, hero_name=name, x=0.0, y=0.0, damage_dealt=300, damage_taken=0,
                kills=0, deaths=0, abilities_used=[])
    data.update(kw)
    return TeamfightPlayer(**data)


def test_fighting_reveals_a_hero_next_to_an_ally_but_not_across_the_map():
    tf = _fight(_fighter(0, "Juggernaut"), _fighter(128, "Phantom Assassin"), _fighter(129, "Lina"))
    match = logged_match({
        128: stand(offset(2200), 500, 900),      # outside hero vision, fighting Juggernaut
        129: stand(offset(0, 9000), 500, 900),   # "in the fight" on the other side of the map
    }, teamfights=[tf])
    vis = states_at(match, 700)
    assert vis["Phantom Assassin"].state == "visible"
    assert vis["Phantom Assassin"].last_seen_source == "teamfight"
    assert vis["Lina"].state == "fog"


def test_fighter_without_position_data_is_visible_without_a_made_up_place():
    # OpenDota: the record's x/y is the centre of the fight, not the hero.
    players = default_players()
    for p in players:
        p.laning_snapshots = []
    tf = _fight(_fighter(1, "Puck", x=2000.0, y=2000.0), _fighter(129, "Lina", x=2000.0, y=2000.0),
                _fighter(130, "Axe", x=2000.0, y=2000.0, damage_dealt=0), start=1500, end=1545)
    match = make_match(players, teamfights=[tf],
                       kills_log=[KillEvent(time=100.0, victim_slot=3, killer_slot=128)])
    vis = states_at(match, 1510)
    assert vis["Lina"].state == "visible" and vis["Lina"].last_seen_source == "teamfight"
    assert (vis["Lina"].last_seen_x, vis["Lina"].last_seen_y) == (0.0, 0.0)
    assert vis["Axe"].state == "unknown"             # in the record, took no part
    text = vision.analyze_vision(match, 1, 1510, lang="en").formatted_text
    assert "- Lina: visible (fighting the player's side; the data has no exact position)" in text


def test_heroes_about_to_kill_an_ally_are_visible():
    """8762775364 LC 17:25 (night): Lion, 830 units away and credited with the assist, was
    "out of sight"; so was every killer outside a recorded teamfight on OpenDota data."""
    kill = KillEvent(time=1003.4, victim_slot=0, killer_slot=128, assist_slots=[129, 130],
                     x=SPOT[0], y=SPOT[1])
    match = logged_match({
        128: stand(offset(600), 900, 1100),        # inside night vision anyway
        129: stand(offset(1200), 900, 1100),       # outside night vision, part of the kill
        130: stand(offset(0, 9000), 900, 1100),    # assist with a spell from across the map
        131: stand(offset(1200, 300), 900, 1100),  # as close as 129, not part of the kill
    }, start=900, end=1100, kills_log=[kill])
    assert vision._is_night(1000)
    vis = states_at(match, 1000)
    assert vis["Lina"].state == "visible" and vis["Lina"].last_seen_source == "teamfight"
    assert vis["Axe"].state == "fog" and vis["Lion"].state == "fog"
    assert states_at(match, 990)["Lina"].state == "fog"         # not yet fighting
    assert states_at(match, 1010)["Lina"].state == "fog"        # the kill is over
    # The victim's own kills do not reveal anybody on his side.
    assert all(e.state != "visible" or e.last_seen_source != "teamfight"
               for e in states_at(match, 1000, team="dire").values())

    # Without positions (OpenDota) the killer is fighting the victim, the others stay unknown.
    players = default_players()
    for p in players:
        p.laning_snapshots = []
    blind = make_match(players, kills_log=[KillEvent(time=1003.0, victim_slot=0, killer_slot=128),
                                           KillEvent(time=1003.0, victim_slot=128, killer_slot=1, deny=True),
                                           KillEvent(time=1004.0, victim_slot=1, killer_slot=129, confirmed=False)])
    vis = states_at(blind, 1000)
    assert vis["Phantom Assassin"].state == "visible"
    assert (vis["Phantom Assassin"].last_seen_x, vis["Phantom Assassin"].last_seen_y) == (0.0, 0.0)
    assert vis["Lina"].state == "unknown"            # an unconfirmed entry may be a reincarnation
    assert states_at(blind, 1000, team="dire")["Puck"].state == "unknown"   # a deny is not a kill


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def test_standing_outside_coverage_is_not_an_event():
    va = vision.analyze_vision(logged_match({}), 0, 700, SPOT, lang="en")
    assert not [e for e in va.timeline.events if e.event_type == "entered_fog"]
    assert va.timeline.critical_description == ""


def _walk_out_match(hidden_enemies: int, to: tuple[float, float] = SPOT):
    tower = geo.TOWER_POSITIONS["radiant"]["mid_t1"]
    logs = {0: stand((tower[0] + 300, tower[1]), 500, 650) + stand(to, 655, 900)}
    for slot, pos in list({129: offset(3000), 128: offset(2000, 2500)}.items())[:hidden_enemies]:
        logs[slot] = stand(pos, 500, 900)
    return logged_match(logs)


def test_walking_away_from_safety_is_one_event_at_the_moment_it_happens():
    va = vision.analyze_vision(_walk_out_match(0), 0, 700, SPOT, lang="en")
    left = [e for e in va.timeline.events if e.event_type == "entered_fog"]
    assert [e.time for e in left] == [655]
    assert left[0].description.startswith("Player moved away from wards and towers into")
    # Nobody is hidden nearby: leaving the towers is not "the" mistake.
    assert va.timeline.critical_description == ""


def test_stepping_out_of_tower_range_in_lane_is_not_an_event():
    tower = geo.TOWER_POSITIONS["radiant"]["mid_t1"]
    in_lane = (tower[0] + 1500, tower[1] + 1300)          # ~2000 units from the tower
    va = vision.analyze_vision(_walk_out_match(2, to=in_lane), 0, 700, in_lane, lang="en")
    assert not [e for e in va.timeline.events if e.event_type == "entered_fog"]


def test_walking_away_is_critical_only_with_several_enemies_hidden_nearby():
    one = vision.analyze_vision(_walk_out_match(1), 0, 700, SPOT, lang="en")
    assert one.timeline.critical_description == ""
    two = vision.analyze_vision(_walk_out_match(2), 0, 700, SPOT, lang="en")
    assert two.timeline.critical_moment == 655
    assert "moved away from wards and towers while several enemy heroes were hidden" in (
        two.timeline.critical_description)


def _walk_out_together(allies: int):
    """Like ``_walk_out_match(2)``, with ``allies`` teammates walking out next to the player."""
    tower = geo.TOWER_POSITIONS["radiant"]["mid_t1"]
    at_tower = stand((tower[0] + 300, tower[1]), 500, 650)
    logs = {0: at_tower + stand(SPOT, 655, 900),
            129: stand(offset(3000), 500, 900), 128: stand(offset(2000, 2500), 500, 900)}
    for slot in range(1, allies + 1):
        logs[slot] = at_tower + stand(offset(-400 * slot), 655, 900)
    return logged_match(logs)


def test_leaving_cover_with_the_team_is_the_teams_move():
    """Live verdicts turned a four-hero push into "a support farming deep in unwarded enemy
    territory": every "Player moved away from wards and towers" was read as the player's choice."""
    grouped = vision.analyze_vision(_walk_out_together(2), 0, 700, SPOT, lang="en")
    moves = [e for e in grouped.timeline.events if e.event_type in ("entered_fog", "group_move")]
    assert [(e.time, e.event_type) for e in moves] == [(655, "group_move")]
    assert moves[0].description.startswith("Player and 2 allies moved away from wards and towers together, into")
    # Not the player's key mistake (two enemies are hidden nearby), and not flagged in the text.
    assert grouped.timeline.critical_description == ""
    line = next(text for text in grouped.formatted_text.splitlines() if "moved away" in text)
    assert not line.endswith("!!!")
    zh = vision.analyze_vision(_walk_out_together(2), 0, 700, SPOT, lang="zh").formatted_text
    assert "玩家与2名队友一起离开守卫和防御塔的保护范围" in zh

    paired = vision.analyze_vision(_walk_out_together(1), 0, 700, SPOT, lang="en")   # company, not a group
    assert [e.event_type for e in paired.timeline.events if "moved away" in e.description] == ["entered_fog"]
    assert paired.timeline.critical_moment == 655


def test_failed_to_retreat_is_only_said_about_a_death():
    match = _walk_out_match(2)
    died = vision.analyze_vision(match, 0, 700, SPOT, lang="en", died=True).formatted_text
    other = vision.analyze_vision(match, 0, 700, SPOT, lang="en").formatted_text
    # Only the enemies hidden within reach are named, not the three in their base.
    assert ("After this, Phantom Assassin, Lina stayed in the fog and the player did not retreat "
            "to safety in time.") in died
    assert "did not retreat" not in other
    assert "Still in the fog at this moment: Phantom Assassin, Lina." in other
    zh = vision.analyze_vision(match, 0, 700, SPOT, lang="zh").formatted_text
    assert "未及时撤退" not in zh and "此刻仍在迷雾中的敌方英雄：Phantom Assassin、Lina" in zh


def test_enemy_leaving_vision_is_an_event_but_dying_is_not():
    match = logged_match({
        128: stand(offset(1000), 500, 670) + stand(offset(3000), 675, 900),          # walks into the fog
        129: stand(offset(1000), 500, 670) + stand(offset(1000), 675, 900, hp=0),    # dies in vision
    }, kills_log=[KillEvent(time=672.0, victim_slot=129, killer_slot=0, x=offset(1000)[0], y=SPOT[1])])
    events = vision.analyze_vision(match, 0, 700, SPOT, lang="en").timeline.events
    missing = [e for e in events if e.event_type == "enemy_missing"]
    assert len(missing) == 1 and missing[0].time == 675
    assert missing[0].description.startswith("Phantom Assassin left allied vision")


def test_window_before_the_horn_is_not_clamped_at_zero():
    players = default_players()
    for p in players:
        p.laning_snapshots = []
    wards = [observer(0.0, 0.0, slot=4, time=-40), observer(500.0, 500.0, slot=4, time=-5)]
    match = make_match(players, wards=wards)
    va = vision.analyze_vision(match, 0, -10, (0.0, 0.0), lang="en")
    assert [(e.event_type, e.time) for e in va.timeline.events] == [("ward_placed", -40)]
    assert va.timeline.critical_description == ""
    assert "Current time: -0:10" in va.formatted_text


# ---------------------------------------------------------------------------
# Wards and towers
# ---------------------------------------------------------------------------

def test_observer_never_outlives_its_lifetime():
    late = observer(0.0, 0.0, slot=4, time=0, destroyed=368)      # entity removed late
    match = make_match(wards=[late])
    assert vision._get_active_wards(match, 359) == [late]
    assert vision._get_active_wards(match, 361) == []
    assert [w for _i, w in vision._active_observers(match, "radiant", 361)] == []
    assert vision._ward_expiry_time(late) == 360


def test_dewarded_flag_decides_between_destroyed_and_expired():
    def ward(x: float, dewarded: bool) -> WardEvent:
        return WardEvent(ward_type="observer", x=x, y=0.0, time=100, player_slot=4,
                         destroyed_time=300, dewarded=dewarded)

    match = make_match(wards=[ward(0.0, True), ward(3000.0, False)])
    events = vision.analyze_vision(match, 0, 330, (0.0, 0.0), lang="en").timeline.events
    kinds = sorted(e.event_type for e in events if e.event_type.startswith("ward_"))
    assert kinds == ["ward_destroyed", "ward_expired"]


def test_lost_ward_far_from_the_player_is_listed_but_not_the_critical_moment():
    def lost(x: float, y: float):
        ward = WardEvent(ward_type="observer", x=x, y=y, time=400, player_slot=4, destroyed_time=700)
        return vision.analyze_vision(make_match(wards=[ward]), 0, 730, (-6000.0, -6000.0), lang="en")

    va = lost(6000.0, 6000.0)
    assert [e.event_type for e in va.timeline.events] == ["ward_expired"]
    assert va.timeline.critical_description == ""
    assert "!!!" not in va.formatted_text          # nor flagged: it says nothing about this decision
    near = lost(-5000.0, -5000.0)
    assert "expired and was not replaced" in near.timeline.critical_description
    assert "vision lost !!!" in near.formatted_text


def test_replaced_ward_is_not_reported_as_not_replaced():
    old = WardEvent(ward_type="observer", x=-5000.0, y=-5000.0, time=400, player_slot=4, destroyed_time=700)
    new = WardEvent(ward_type="observer", x=-4800.0, y=-5100.0, time=705, player_slot=4)
    va = vision.analyze_vision(make_match(wards=[old, new]), 0, 730, (-6000.0, -6000.0), lang="en")
    assert sorted(e.event_type for e in va.timeline.events) == ["ward_expired", "ward_placed"]
    assert va.timeline.critical_description == ""


def _tier4(top: int | None, bot: int | None) -> list[dict]:
    pos = geo.TOWER_POSITIONS["radiant"]
    return [
        {"name": "npc_dota_goodguys_tower4_top", "team": "radiant", "type": "tower",
         "x": pos["ancient_top"][0], "y": pos["ancient_top"][1], "destroyed_time": top},
        {"name": "npc_dota_goodguys_tower4_bot", "team": "radiant", "type": "tower",
         "x": pos["ancient_bot"][0], "y": pos["ancient_bot"][1], "destroyed_time": bot},
    ]


def _unsided(*times: int) -> list[dict]:
    return [{"time": t, "type": "building_kill", "key": "npc_dota_goodguys_tower4", "team": "dire"}
            for t in times]


def _standing(match, t: int) -> set[str]:
    return {name for name, _x, _y in vision._get_alive_towers(match, "radiant", t)}


def test_unsided_tier4_kill_does_not_destroy_the_tower_that_still_stands():
    match = make_match(buildings=_tier4(None, 2036), objectives=_unsided(2036))
    assert _standing(match, 2100) == {"ancient_top"}
    match = make_match(buildings=_tier4(2054, 2036), objectives=_unsided(2036, 2054))
    assert _standing(match, 2040) == {"ancient_top"}          # the bottom one fell first
    assert _standing(match, 2060) == set()


def test_unsided_tier4_kills_date_towers_without_a_time_of_their_own():
    # The building records have no times: the objectives are all there is.
    match = make_match(buildings=_tier4(None, None), objectives=_unsided(2036))
    assert _standing(match, 2040) == {"ancient_bot"}
    # No buildings at all (OpenDota): kills are taken in order, top first.
    match = make_match(objectives=_unsided(2036, 2054))
    assert "ancient_top" in _standing(match, 2000) and "ancient_top" not in _standing(match, 2040)
    assert "ancient_bot" in _standing(match, 2040) and "ancient_bot" not in _standing(match, 2060)


def test_side_of_the_map_ends_at_the_river():
    """The side used to be "nearer to which ancient": the spot where the Dire safe-lane waves
    meet, 1200 units in front of the Dire tier-1 tower, was "Radiant top lane", and the live
    verdict read "killed near the enemy top tower ... pushed too far"."""
    assert vision.area_code(-5800.0, 4990.0) == ("lane", "dire", "top")      # 8762775364, death at 4:50
    assert vision.describe_area(-5800.0, 4990.0, "zh") == "夜魇上路"
    assert vision.area_code(5476.0, -5488.0) == ("lane", "radiant", "bot")   # Radiant safe lane, same match
    assert vision.area_code(-5739.0, 2725.0) == ("lane", "radiant", "top")   # pushed past the river
    for team in ("radiant", "dire"):
        for name, (x, y) in geo.TOWER_POSITIONS[team].items():
            assert vision.area_code(x, y)[1] == team, name
    # Between two lanes the river itself has no side; it bends at the mid lane.
    (tx, ty), (mx, my) = vision._RIVER[0], vision._RIVER[1]
    assert vision.area_code((tx + mx) / 2, (ty + my) / 2) == ("river", "", "top")
    assert all(abs(vision.river_offset(x, y)) < 1 for x, y in vision._RIVER)
    assert vision.river_offset(*geo.ANCIENT_POSITIONS["radiant"]) < 0 < vision.river_offset(
        *geo.ANCIENT_POSITIONS["dire"])


def test_danger_is_not_called_safe_about_enemies_without_position_data():
    blind = logged_match({slot: [] for slot in (128, 129, 130, 131, 132)})
    text = vision.analyze_vision(blind, 0, 700, SPOT, lang="en").formatted_text
    assert "Danger from hidden enemies (estimate): cannot be told (5 enemy hero(es) have no position data)" in text
    assert "estimate): Safe" not in text
    zh = vision.analyze_vision(blind, 0, 700, SPOT, lang="zh").formatted_text
    assert "无法判断（5名敌方英雄没有位置数据）" in zh and "推算）: 安全" not in zh
    assert "estimate): Safe" in vision.analyze_vision(logged_match({}), 0, 700, SPOT, lang="en").formatted_text


def test_fountain_label_does_not_swallow_the_ancient():
    for team in ("radiant", "dire"):
        assert vision.area_code(*geo.ANCIENT_POSITIONS[team])[:2] == ("base", team)
        assert vision.area_code(*geo.FOUNTAIN_POSITIONS[team])[:2] == ("fountain", team)


def test_no_ward_is_no_danger_factor_next_to_an_own_tower():
    tower = geo.TOWER_POSITIONS["radiant"]["mid_t1"]
    match = logged_match({0: stand((tower[0] + 300, tower[1]), 500, 900)})
    safe = vision.analyze_vision(match, 0, 700, (tower[0] + 300, tower[1]), lang="en").state_at_decision
    assert safe.player_in_safe_zone and not safe.nearby_ward_coverage
    assert not any("no observer ward" in r for r in safe.danger_reasons)
    exposed = vision.analyze_vision(logged_match({}), 0, 700, SPOT, lang="en").state_at_decision
    assert any("no observer ward" in r for r in exposed.danger_reasons)
