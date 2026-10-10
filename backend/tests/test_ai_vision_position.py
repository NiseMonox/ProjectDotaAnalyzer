"""Where a hero is and whether it is alive at time T (vision.py, contract A2).

Synthetic matches only: a position log (replay / mock), a kill log without
positions (OpenDota), or the older sources (laning snapshots, teamfights).
"""
from __future__ import annotations

import gc
import threading

import pytest

from backend.app.ai import vision
from backend.app.models import KillEvent, PositionSnapshot, TeamfightEvent, TeamfightPlayer
from test_ai_pipeline_helpers import default_players, make_match


def stand(x: float, y: float, start: int, end: int, hp: int = 100) -> list[list[int]]:
    """Position-log rows every 5 s standing at (x, y)."""
    return [[t, int(x), int(y), hp] for t in range(start, end + 1, 5)]


def bare_players() -> list:
    """Ten players without any position source."""
    players = default_players()
    for p in players:
        p.laning_snapshots = []
    return players


def kill(time: float, victim: int, killer: int | None = None, **kw) -> KillEvent:
    return KillEvent(time=time, victim_slot=victim, killer_slot=killer, **kw)


def fight_player(slot: int, name: str, x: float = 0.0, y: float = 0.0, **kw) -> TeamfightPlayer:
    data = dict(player_slot=slot, hero_name=name, x=x, y=y, damage_dealt=0, damage_taken=0,
                kills=0, deaths=0, abilities_used=[])
    data.update(kw)
    return TeamfightPlayer(**data)


# ---------------------------------------------------------------------------
# Position log
# ---------------------------------------------------------------------------

def _log_match():
    players = bare_players()
    # Puck walks east, dies at 7.4 s, lies dead and respawns in the fountain.
    players[1].position_log = [
        [0, 0, 1000, 100], [5, 500, 1000, 80], [10, 700, 1000, 0], [15, 700, 1000, 0],
        [20, -7000, -6500, 100], [25, -6800, -6300, 100],
    ]
    return make_match(players, kills_log=[kill(7.4, 1, 129, x=700.0, y=1000.0)])


def test_position_log_is_interpolated_between_samples():
    match = _log_match()
    assert vision.position_at(match, 1, 2) == pytest.approx((200.0, 1000.0))
    assert vision.position_at(match, 1, 5) == (500.0, 1000.0)
    # After the last sample before a death the hero moves towards the death point.
    x, y = vision.position_at(match, 1, 6)
    assert 500 < x < 700 and y == pytest.approx(1000.0)


def test_dead_hero_has_no_position_and_respawns_without_interpolation():
    match = _log_match()
    assert vision.hero_alive_at(match, 1, 7) is True          # dies during second 7
    assert vision.position_at(match, 1, 7) is not None
    for t in (8, 10, 15, 19):
        assert vision.hero_alive_at(match, 1, t) is False, t
        assert vision.position_at(match, 1, t) is None, t
        assert vision.estimate_position(match, 1, t) == (0.0, 0.0)
    assert vision.hero_alive_at(match, 1, 20) is True
    assert vision.position_at(match, 1, 20) == (-7000.0, -6500.0)
    assert vision.position_at(match, 1, -30) is None          # before the first spawn
    assert vision.position_at(match, 1, 600) is None          # after the log ended


def test_teleport_between_two_samples_is_not_interpolated():
    players = bare_players()
    players[0].position_log = [[100, -7000, -6500, 100], [105, 3000, 3000, 100]]
    match = make_match(players)
    assert vision.position_at(match, 0, 101) == (-7000.0, -6500.0)
    assert vision.position_at(match, 0, 104) == (3000.0, 3000.0)


def test_log_death_without_kill_entry_is_a_death_too():
    # Aegis / reincarnation: hp 0 in the log, nothing in the kill log.
    players = bare_players()
    players[0].position_log = stand(100, 100, 0, 10) + stand(100, 100, 15, 15, hp=0) + stand(100, 100, 20, 30)
    match = make_match(players)
    assert vision.hero_alive_at(match, 0, 16) is False
    assert vision.hero_alive_at(match, 0, 21) is True


def test_buyback_revives_before_the_next_sample():
    players = bare_players()
    players[0].position_log = stand(100, 100, 0, 5) + stand(100, 100, 10, 10, hp=0) + stand(-7000, -6500, 15, 30)
    players[0].buyback_log = [{"time": 12}]
    match = make_match(players, kills_log=[kill(8.0, 0, 128, x=100.0, y=100.0)])
    assert vision.hero_alive_at(match, 0, 11) is False
    assert vision.hero_alive_at(match, 0, 13) is True
    assert vision.position_at(match, 0, 13) == (-7000.0, -6500.0)   # not the corpse


# ---------------------------------------------------------------------------
# No position log (OpenDota, older data)
# ---------------------------------------------------------------------------

def test_kill_log_with_respawn_timer_decides_alive_without_a_position_log():
    players = bare_players()
    players[0].xp_t = [0, 300, 700, 1200, 1800, 2500]      # level 6 at minute 5 -> 26 s
    match = make_match(players, kills_log=[
        kill(300.0, 0, 128, x=1000.0, y=1000.0),
        kill(900.0, 1, 129, confirmed=False),              # may be a reincarnation
    ])
    assert vision.respawn_seconds(6) == 26
    assert vision.hero_alive_at(match, 0, 299) is True
    assert vision.hero_alive_at(match, 0, 300) is True      # dies during this second
    assert vision.hero_alive_at(match, 0, 301) is False
    assert vision.hero_alive_at(match, 0, 325) is False
    assert vision.hero_alive_at(match, 0, 327) is True
    assert vision.hero_alive_at(match, 2, 301) is True      # no entry for this hero
    assert vision.hero_alive_at(match, 1, 905) is None      # unconfirmed: unknown
    # The death point places the victim for the few seconds before the death only.
    assert vision.position_at(match, 0, 297) == (1000.0, 1000.0)
    assert vision.position_at(match, 0, 290) is None
    assert vision.position_at(match, 0, 305) is None


def test_no_death_data_at_all_means_unknown():
    match = make_match(bare_players())
    assert vision.hero_alive_at(match, 0, 900) is None
    assert vision.position_at(match, 0, 900) is None


def test_fight_centre_is_not_a_position_for_a_bystander():
    # OpenDota: no samples; TeamfightPlayer.x/y is the centre of the fight
    # for everybody who did not die there.
    tf = TeamfightEvent(start_time=1500, end_time=1545, radiant_deaths=1, dire_deaths=0, players=[
        fight_player(1, "Puck", 2000.0, 2000.0, deaths=1, death_time_offset=20.0),
        fight_player(0, "Juggernaut", 2000.0, 2000.0, damage_dealt=500),
        fight_player(129, "Lina", 2000.0, 2000.0, kills=1, damage_dealt=900),
        fight_player(130, "Axe", 2000.0, 2000.0),
    ])
    match = make_match(bare_players(), teamfights=[tf],
                       kills_log=[kill(1520.0, 1, 129, x=1900.0, y=2100.0)])
    for slot in (0, 129, 130):
        assert vision.position_at(match, slot, 1520) is None, slot
    assert vision.position_at(match, 1, 1518) == (1900.0, 2100.0)
    assert vision.count_nearby_heroes(match, "dire", 1900.0, 2100.0, 1518) == 0


def test_real_teamfight_samples_and_snapshots_are_still_used():
    players = default_players()                      # laning snapshots, no position log
    for p in players[1:]:
        p.laning_snapshots = []
    tf = TeamfightEvent(start_time=1500, end_time=1520, radiant_deaths=0, dire_deaths=0, players=[
        fight_player(129, "Lina", 0.0, 0.0, damage_dealt=100,
                     positions=[PositionSnapshot(x=800.0, y=900.0, time_offset=4.0)]),
    ])
    match = make_match(players, teamfights=[tf])
    assert vision.position_at(match, 129, 1505) == (800.0, 900.0)
    assert vision.position_at(match, 0, 302) == (4500.0, -6000.0)
    assert vision.position_at(match, 0, 900) is None


def test_snapshot_with_zero_hp_is_a_dead_hero():
    players = default_players()
    for s in players[0].laning_snapshots:
        if 300 <= s.time <= 320:
            s.hp = 0
    match = make_match(players)
    assert vision.hero_alive_at(match, 0, 310) is False
    assert vision.position_at(match, 0, 310) is None
    assert vision.hero_alive_at(match, 0, 330) is True
    assert vision.last_alive_time(match, 0, 300) == 295
    assert vision.hp_pct_at(match, 0, 295) == 100


# ---------------------------------------------------------------------------
# Helpers built on it
# ---------------------------------------------------------------------------

def test_state_before_a_death_is_the_last_sample_alive():
    players = bare_players()
    players[1].position_log = (stand(0, 0, 0, 340) + [[345, 0, 0, 41]]
                               + stand(0, 0, 350, 370, hp=0))
    match = make_match(players, kills_log=[kill(349.6, 1, 129, x=0.0, y=10.0)])
    assert vision.last_alive_time(match, 1, 349) == 345
    assert vision.hp_pct_at(match, 1, 345) == 41
    assert vision.hp_pct_at(match, 1, 352) == 0
    # A death right on a sample second: at least two seconds earlier.
    assert vision.last_alive_time(match, 1, 346) == 340
    # No samples: the second before the death.
    assert vision.last_alive_time(match, 0, 900) == 899
    assert vision.hp_pct_at(match, 0, 899) is None


def test_fight_participants_need_more_than_a_cast():
    tf = TeamfightEvent(start_time=0, end_time=30, radiant_deaths=0, dire_deaths=0, players=[
        fight_player(0, "Juggernaut", damage_dealt=10),
        fight_player(1, "Puck", abilities_used=["puck_illusory_orb"]),     # farming elsewhere
        fight_player(2, "Tinker", healing=50),
        fight_player(128, "Phantom Assassin", damage_taken=10),
        fight_player(129, "Lina", deaths=1),
        fight_player(130, "Axe"),
    ])
    assert vision.fight_participants(tf) == ([0, 2], [128, 129])


def test_score_before_a_moment_follows_the_kill_log():
    match = make_match(kills_log=[
        kill(100.0, 128, 0),                                   # radiant kill
        kill(200.0, 1, 129),                                   # dire kill
        kill(250.0, 2, 3, deny=True),                          # denied: nobody scores
        kill(300.0, 4, None, killer_unit="npc_dota_neutral_kobold"),
        kill(400.0, 129, None, killer_unit="npc_dota_goodguys_tower1_mid"),
    ])
    assert vision.kills_before(match, 100) == (0, 0)           # strictly before
    assert vision.kills_before(match, 101) == (1, 0)
    assert vision.kills_before(match, 350) == (1, 1)
    assert vision.kills_before(match, 500) == (2, 1)           # a tower kill scores
    assert vision.kills_before(make_match(), 500) is None


def test_index_is_cached_per_match_and_dropped_with_it():
    match = _log_match()
    idx = vision._index(match)
    assert vision._index(match) is idx
    copy = match.model_copy(deep=True)
    assert vision._index(copy) is not idx
    key = id(copy)
    del copy
    gc.collect()
    assert key not in vision._INDEXES


def test_changed_position_log_rebuilds_the_index():
    match = _log_match()
    assert vision.position_at(match, 0, 3) is None
    match.players[0].position_log = stand(50, 60, 0, 10)
    assert vision.position_at(match, 0, 3) == (50.0, 60.0)


def test_concurrent_lookups_agree_with_the_single_thread_result():
    match = _log_match()
    expected = [(vision.position_at(match, 1, t), vision.hero_alive_at(match, 1, t)) for t in range(0, 26)]
    shared = match.model_copy(deep=True)      # cold index, built under contention
    results: list = []

    def work() -> None:
        results.append([(vision.position_at(shared, 1, t), vision.hero_alive_at(shared, 1, t))
                        for t in range(0, 26)])

    threads = [threading.Thread(target=work) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(results) == 8 and all(r == expected for r in results)


def test_score_does_not_depend_on_the_analyzer_feed_being_there(monkeypatch):
    from backend.app import analyzer

    def gone(*_a, **_k):
        raise RuntimeError("kill feed changed")

    monkeypatch.setattr(analyzer, "_kill_feed", gone)
    match = make_match(kills_log=[kill(100.0, 128, 0), kill(200.0, 1, 129), kill(250.0, 2, 3, deny=True)])
    assert vision.kills_before(match, 300) == (1, 1)
