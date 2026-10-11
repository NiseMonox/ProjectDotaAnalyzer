"""Mock match data: determinism, world coordinates, buildings, codes and the
internal consistency of the kill log, positions and economy series."""
from __future__ import annotations

import asyncio
import os
import random
from collections import Counter

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import pytest  # noqa: E402

from backend.app import geo, mock_data, parser  # noqa: E402
from backend.app.mock_data import _respawn_time, _smoke_result, generate_mock_match  # noqa: E402


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


@pytest.fixture(scope="module")
def match():
    return generate_mock_match()


def _all_points(m) -> list[tuple[float, float]]:
    pts = [(w.x, w.y) for w in m.wards]
    for s in m.smokes:
        pts.append((s.x, s.y))
        pts.extend((x, y) for x, y in s.path)
    for tf in m.teamfights:
        for tp in tf.players:
            pts.append((tp.x, tp.y))
            pts.extend((ps.x, ps.y) for ps in tp.positions)
            pts.extend((ac.x, ac.y) for ac in tp.ability_casts)
    pts.extend((e.x, e.y) for e in m.ability_events)
    pts.extend((k.x, k.y) for k in m.kills_log)
    for p in m.players:
        pts.extend((s.x, s.y) for s in p.laning_snapshots)
        pts.extend((x, y) for _t, x, y, _hp in p.position_log)
    pts.extend((b["x"], b["y"]) for b in m.buildings)
    return pts


def test_mock_is_deterministic_and_leaves_global_random_alone():
    random.seed(1234)
    expected_next = random.random()
    random.seed(1234)
    first = generate_mock_match()
    assert random.random() == expected_next
    assert generate_mock_match().model_dump() == first.model_dump()


def test_all_coordinates_fall_inside_the_map(match):
    points = _all_points(match)
    assert len(points) > 1000
    for x, y in points:
        assert 0 <= geo.world_to_pct(x) <= 100, (x, y)
        assert 0 <= geo.world_to_pct(y) <= 100, (x, y)
    # world units, not the old -4000..3875 pseudo space
    assert max(abs(c) for pt in points for c in pt) > 4500


def test_real_wards_use_opendota_cell_conversion(match):
    hoodwink_first_obs = next(w for w in match.wards if w.player_slot == 0 and w.time == 960)
    assert hoodwink_first_obs.x == pytest.approx(geo.opendota_cell_to_world(114.7), abs=0.1)
    assert hoodwink_first_obs.y == pytest.approx(geo.opendota_cell_to_world(160.4), abs=0.1)


def test_buildings_and_status_are_consistent(match):
    assert len(match.buildings) == 2 * (11 + 6 + 1)
    for b in match.buildings:
        assert set(b) == {"name", "team", "type", "x", "y", "destroyed_time"}
        assert parser.parse_building_name(b["name"])["team"] == b["team"]
    status = parser.building_status_bitmasks(match.buildings)
    assert match.tower_status_radiant == status["tower_status_radiant"] == 4 + 256  # top/bot t3 left
    assert match.tower_status_dire == status["tower_status_dire"]
    assert match.barracks_status_radiant == 63 - 4 - 8
    radiant_fort = next(b for b in match.buildings if b["name"] == "npc_dota_goodguys_fort")
    assert radiant_fort["destroyed_time"] == match.duration  # Dire won
    kills = [o for o in match.objectives if o["type"] == "building_kill"]
    destroyed = [b for b in match.buildings if b["destroyed_time"] is not None]
    assert len(kills) == len(destroyed)


def test_codes_not_labels(match):
    assert match.game_mode == "ranked_all_pick"
    assert {s.result for s in match.smokes} <= {"kill", "traded", "objective", "no_contact", "caught"}
    for s in match.smokes:  # the result follows from the counts, like the replay analyzer's
        expected = ("traded" if s.kills_gained and s.deaths_suffered else "kill" if s.kills_gained
                    else "objective" if s.objective_taken else "caught" if s.deaths_suffered else "no_contact")
        assert s.result == expected, s
    # ... whatever the seed rolled: traded > kill > objective > caught > no_contact
    assert [_smoke_result(*counts) for counts in ((2, 1, None), (1, 1, "tower"), (2, 0, "roshan"), (0, 1, "tower"),
                                                  (0, 1, None), (0, 0, None))] == [
        "traded", "traded", "kill", "objective", "caught", "no_contact"]
    for p in match.players:
        assert p.lane in (1, 2, 3) and p.lane_role in (1, 2, 3)
        inferred = parser.infer_lane([(s.x, s.y) for s in p.laning_snapshots if 60 <= s.time <= 600], p.team)
        assert inferred[:2] == (p.lane, p.lane_role), p.hero_name


def test_mock_parser_returns_mock_match():
    m = asyncio.run(parser.MockParser().parse(""))
    # modelled on match 8713267082, but not that match: no id, no server region to show
    assert (m.match_id, m.cluster, m.region) == (0, 0, 0) and len(m.players) == 10


# ---------------------------------------------------------------------------
# One kill log drives scoreboard, teamfights and positions
# ---------------------------------------------------------------------------

def _dead_intervals(match) -> dict[int, list[tuple[float, float]]]:
    dead: dict[int, list[tuple[float, float]]] = {p.player_slot: [] for p in match.players}
    for k in match.kills_log:
        dead[k.victim_slot].append((k.time, k.time + _respawn_time(k.time)))
    return dead


def test_kills_log_matches_the_scoreboard(match):
    kills = match.kills_log
    assert [k.time for k in kills] == sorted(k.time for k in kills)
    assert len(kills) == match.radiant_score + match.dire_score == 70
    names = {p.player_slot: p.hero_name for p in match.players}
    for p in match.players:
        assert sum(1 for k in kills if k.victim_slot == p.player_slot) == p.deaths, p.hero_name
        assert sum(1 for k in kills if k.killer_slot == p.player_slot) == p.kills, p.hero_name
        assert sum(1 for k in kills if p.player_slot in k.assist_slots) == p.assists, p.hero_name
        assert p.killed == dict(Counter(names[k.victim_slot] for k in kills if k.killer_slot == p.player_slot))
        assert p.killed_by == dict(Counter(names[k.killer_slot] for k in kills if k.victim_slot == p.player_slot))
    for k in kills:
        assert (k.victim_slot < 128) != (k.killer_slot < 128) and not k.deny and k.confirmed
        assert k.killer_slot not in k.assist_slots and len(set(k.assist_slots)) == len(k.assist_slots)
        assert all((a < 128) == (k.killer_slot < 128) for a in k.assist_slots)
    # nobody dies or scores a kill while dead
    dead = _dead_intervals(match)
    for k in kills:
        assert not any(start < k.time < end for start, end in dead[k.victim_slot]), k
        assert not any(start <= k.time < end for start, end in dead[k.killer_slot]), k
    first = kills[0]
    assert (first.time, first.killer_slot, first.victim_slot) == (match.first_blood_time, 131, 2)
    first_blood = next(o for o in match.objectives if o["type"] == "CHAT_MESSAGE_FIRSTBLOOD")
    assert (first_blood["player_slot"], first_blood["key"]) == (131, "2")


def test_teamfights_follow_the_kill_log(match):
    assert len(match.teamfights) >= 5
    in_fights = 0
    for tf in match.teamfights:
        kills = [k for k in match.kills_log if tf.start_time <= k.time <= tf.end_time]
        in_fights += len(kills)
        assert len(kills) >= 3 and tf.radiant_deaths + tf.dire_deaths == len(kills)
        # padded like OpenDota / the analyzer: 15 s around the first and last death
        assert tf.start_time == int(kills[0].time) - 15 and 15 <= tf.end_time - kills[-1].time < 16
        assert tf.radiant_deaths == sum(1 for k in kills if k.victim_slot < 128)
        for tp in tf.players:
            deaths = [k for k in kills if k.victim_slot == tp.player_slot]
            own_kills = [k for k in kills if k.killer_slot == tp.player_slot]
            assert tp.deaths == len(deaths) <= 1 and tp.kills == len(own_kills)
            assert tp.kill_targets == [
                {"target": next(p.hero_name for p in match.players if p.player_slot == k.victim_slot),
                 "target_slot": k.victim_slot, "time_offset": round(k.time - tf.start_time, 1)}
                for k in own_kills
            ]
            if deaths:
                assert tp.death_time_offset == round(deaths[0].time - tf.start_time, 1)
                last = tp.positions[-1]      # the track ends where the hero died
                assert (last.x, last.y, last.time_offset) == (deaths[0].x, deaths[0].y, tp.death_time_offset)
            else:
                assert tp.death_time_offset is None
            # ability keys, not display names
            assert all(key == key.lower() and " " not in key for key in tp.abilities_used)
            assert set(tp.abilities_used) == {c.ability_key for c in tp.ability_casts}
    assert 0 < in_fights < len(match.kills_log)      # pick-offs exist outside the fights


def test_position_log_agrees_with_deaths_and_laning_snapshots(match):
    dead = _dead_intervals(match)
    for p in match.players:
        log = p.position_log
        assert [e[0] for e in log] == list(range(-90, match.duration + 1, 5))
        by_time = {e[0]: e for e in log}
        for t, _x, _y, hp in log:
            is_dead = any(start <= t < end for start, end in dead[p.player_slot])
            assert (hp == 0) == is_dead and 0 <= hp <= 100, (p.hero_name, t, hp)
        for snap in p.laning_snapshots:          # same place, same life state
            _t, x, y, hp = by_time[snap.time]
            assert abs(x - snap.x) <= 1 and abs(y - snap.y) <= 1, (p.hero_name, snap.time)
            assert (snap.hp == 0) == (hp == 0)
        for k in match.kills_log:                # a dead hero stays where it died
            if k.victim_slot != p.player_slot:
                continue
            sample = -(-k.time // 5) * 5
            if sample <= match.duration and sample < k.time + _respawn_time(k.time):
                _t, x, y, hp = by_time[sample]
                assert hp == 0 and abs(x - k.x) <= 1 and abs(y - k.y) <= 1, k
        for e in p.ability_events:               # casts happen where the hero is, never while dead
            assert not any(start <= e.time < end for start, end in dead[p.player_slot])
    assert [s.time for s in match.players[0].laning_snapshots] == list(range(0, 601, 5))


def test_economy_series_are_consistent(match):
    minutes = match.duration // 60 + 1
    for p in match.players:
        assert len(p.gold_t) == len(p.xp_t) == len(p.lh_t) == len(p.dn_t) == minutes
        for series in (p.gold_t, p.xp_t, p.lh_t, p.dn_t):
            assert series[0] == 0 and series == sorted(series)
        total_gold = p.gold_per_min * match.duration / 60
        assert 0.9 * total_gold < p.gold_t[-1] <= total_gold
        assert p.lh_t[-1] <= p.last_hits and p.dn_t[-1] <= p.denies
        at_ten = p.laning_snapshots[-1]
        assert (p.lh_t[10], p.dn_t[10]) == (at_ten.last_hits, at_ten.denies)
        assert at_ten.last_hits == len(p.laning_creep_kills) and at_ten.denies == len(p.laning_denies)
        assert p.lane_efficiency == pytest.approx(p.gold_t[10] / 4948) and 0 < p.lane_efficiency < 1
        assert p.laning_snapshots[-1].net_worth == 600 + p.gold_t[10]
    for minute, snap in enumerate(match.gold_xp_timeline):
        assert snap.time == minute * 60
        assert snap.radiant_gold == sum(p.gold_t[minute] for p in match.players if p.team == "radiant")
        assert snap.dire_xp == sum(p.xp_t[minute] for p in match.players if p.team == "dire")
        assert match.radiant_gold_adv[minute] == snap.radiant_gold - snap.dire_gold
        assert match.radiant_xp_adv[minute] == snap.radiant_xp - snap.dire_xp
    assert len(match.gold_xp_timeline) == minutes


def test_identifiers_match_the_real_parsers(match):
    items = parser._items_by_key()
    abilities = parser._load_static_json(parser.ABILITIES_JSON_PATH)["by_key"]
    for p in match.players:
        for key in p.final_item_keys + [it.item_key for it in p.items]:
            assert key in items, key                 # e.g. "boots", not "boots_of_speed"
        assert [(e["time"], e["key"]) for e in p.purchase_log] == [(it.time, it.item_key) for it in p.items]
        talents = [s for s in p.skills if s.ability_key.startswith("special_bonus_")]
        assert len(talents) == sum(1 for lvl in (10, 15, 20, 25) if p.level >= lvl)
        assert all(s.ability_key in abilities and s.level == 1 for s in talents)
        # named like the talents of a parsed match (one function names them all)
        assert all(s.ability_name == parser.ability_display_name(s.ability_key) for s in talents)
        assert all(s.ability_name.startswith("+") and "{" not in s.ability_name for s in talents)
        assert not any(s.ability_key.startswith("talent_") for s in p.skills)
        assert [s.time for s in p.skills] == sorted(s.time for s in p.skills)
        # damage dicts: enemy heroes by display name, "attack" for plain attacks
        enemies = {q.hero_name for q in match.players if q.team != p.team}
        assert set(p.damage) == enemies and sum(p.damage.values()) == p.hero_damage
        assert sum(p.damage_inflictor.values()) == p.hero_damage and "attack" in p.damage_inflictor
        assert p.damage_taken == {q.hero_name: q.damage[p.hero_name] for q in match.players if q.team != p.team}
        assert {e.inflictor for e in p.laning_damage_events} <= {"attack"} | {key for key in abilities}
    assert (match.cluster, match.region, match.patch, match.winner_known) == (0, 0, "7.40", True)   # no server: generated


def test_mock_is_a_consistent_match_of_its_patch(match):
    """The mock stands in for a 7.40c game of 2026-03-02: its patch label, its start
    time and its skill rules belong together, and everything it uses still exists."""
    assert (match.patch, match.start_time) == ("7.40", 1772470634)
    assert parser.talents_have_own_points(match.patch) and parser.talents_have_own_points("", match.start_time)
    items = parser._items_by_key()
    used = {key for p in match.players
            for key in [e.item_key for e in p.items] + p.final_item_keys + p.backpack + [p.neutral_item] if key}
    assert len(used) > 30
    assert [key for key in sorted(used) if key not in items or items[key].get("obsolete")] == []
    abilities = parser._load_static_json(parser.ABILITIES_JSON_PATH)["by_key"]
    skilled = {s.ability_key for p in match.players for s in p.skills}
    assert len(skilled) > 40 and [key for key in sorted(skilled) if key not in abilities] == []


def test_skill_builds_follow_the_rules_of_the_mock_patch(match):
    """7.40: a talent has a point of its own, so the hero also learns an ability at
    levels 10 and 15 (before 7.40 the talent took that level's skill point)."""
    assert parser.talents_have_own_points(match.patch)
    for p in match.players:
        learned = [s for s in p.skills if not parser.is_talent(s.ability_key)]
        talents = [s for s in p.skills if parser.is_talent(s.ability_key)]
        # one ability level per hero level up to 14; nothing can take the points of levels
        # 15-17 (the game spends them on the attribute bonus); the third ultimate level at 18
        assert len(learned) == min(p.level, 14) + (p.level >= 18), p.hero_name
        if talents:
            assert sum(1 for s in learned if s.time <= talents[0].time) == 10, p.hero_name
        ultimate = learned[5].ability_key                       # learned at level 6
        levels = Counter()
        for point, s in enumerate(learned, start=1):            # the n-th point is spent at level n
            levels[s.ability_key] += 1
            assert s.level == levels[s.ability_key]
            hero_level = 18 if point == 15 else point
            if s.ability_key == ultimate:
                assert hero_level == 6 * s.level, p.hero_name
            else:
                assert s.level <= 4 and hero_level >= 2 * s.level - 1, (p.hero_name, s.ability_key, s.level)
        assert len(levels) == 4
        # the estimate the OpenDota parser makes for such a list puts every entry at the same level
        keys = [s.ability_key for s in p.skills]
        expected, point, talent_levels = [], 0, iter((10, 15, 20, 25))
        for key in keys:
            if parser.is_talent(key):
                expected.append(next(talent_levels))
            else:
                point += 1
                expected.append(18 if point == 15 else point)
        assert parser._upgrade_levels(keys, True) == expected, p.hero_name


def test_player_names_are_invented(match):
    """The mock is modelled on a real match but must not show its players' persona names."""
    names = [p.player_name for p in match.players]
    assert names == mock_data.PLAYER_NAMES and len(set(names)) == 10
    assert all(name.startswith("Mock ") for name in names)


def test_wards_carry_team_and_deward_flag(match):
    for w in match.wards:
        assert w.team == ("radiant" if w.player_slot < 128 else "dire")
        assert not w.dewarded or w.destroyed_time is not None
        assert w.destroyed_time is None or w.destroyed_time <= match.duration
    counts = Counter((w.ward_type, w.dewarded) for w in match.wards)
    # as in the real match the mock is modelled on: 19 dewards, and the sentry Warlock
    # attacked himself 18 s after placing it (the lifetime alone would call it a deward)
    assert counts[("observer", True)] == 6 and counts[("sentry", True)] == 13
    denied = [w for w in match.wards if w.denied]
    assert [(w.ward_type, w.time, w.destroyed_time, w.player_slot, w.killer_slot, w.dewarded) for w in denied] == [
        ("sentry", 1003, 1021, 2, 2, False)]


def test_every_death_has_the_respawn_the_position_log_shows(match):
    by_slot = {p.player_slot: p for p in match.players}
    for kill in match.kills_log:
        log = by_slot[kill.victim_slot].position_log
        if kill.respawn_time is None:       # the game ended first
            assert all(hp == 0 for t, _x, _y, hp in log if t > kill.time), kill
            continue
        assert kill.time + 10 <= kill.respawn_time <= match.duration, kill
        # dead on every sample up to the respawn, alive on those after it until the next death
        again = min((k.time for k in match.kills_log if k.victim_slot == kill.victim_slot and k.time > kill.time),
                    default=match.duration + 1)
        assert kill.respawn_time <= again, kill
        assert all(hp == 0 for t, _x, _y, hp in log if kill.time < t < kill.respawn_time), kill
        assert all(hp > 0 for t, _x, _y, hp in log if kill.respawn_time <= t < again), kill
    assert sum(k.respawn_time is None for k in match.kills_log) == 1
