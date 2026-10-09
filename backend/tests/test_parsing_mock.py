"""Mock match data: determinism, world coordinates, buildings and codes."""
from __future__ import annotations

import asyncio
import os
import random

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import pytest  # noqa: E402

from backend.app import geo, parser  # noqa: E402
from backend.app.mock_data import generate_mock_match  # noqa: E402


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
    for p in m.players:
        pts.extend((s.x, s.y) for s in p.laning_snapshots)
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
    for p in match.players:
        assert p.lane in (1, 2, 3) and p.lane_role in (1, 2, 3)
        inferred = parser.infer_lane([(s.x, s.y) for s in p.laning_snapshots if 60 <= s.time <= 600], p.team)
        assert inferred[:2] == (p.lane, p.lane_role), p.hero_name


def test_mock_parser_returns_mock_match():
    m = asyncio.run(parser.MockParser().parse(""))
    assert m.match_id == 8713267082 and len(m.players) == 10
