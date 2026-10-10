"""Tests for the shared map-geometry helpers."""
import json
from pathlib import Path

import pytest

from backend.app import geo, parser

FIXTURES = Path(__file__).parent / "fixtures"


def test_cell_to_world_uses_offset_and_vec():
    assert geo.cell_to_world(128, 0) == 0
    assert geo.cell_to_world(64, 0) == -8192
    assert geo.cell_to_world(128) == 64  # cell centre when vec is unknown


def test_opendota_cells_map_to_world():
    assert geo.opendota_cell_to_world(128) == 0
    assert geo.opendota_cell_to_world(64) == -8192
    assert geo.opendota_cell_to_world(191) == 8064


@pytest.mark.parametrize("cell", [64, 100, 128, 150.5, 191])
def test_pct_matches_legacy_frontend_mapping(cell):
    """New world→pct mapping must place things exactly where the old
    ``(cell*62.5 - 8000 + 4000) / 7937.5 * 100`` formula did."""
    legacy = (cell * 62.5 - 8000 + 4000) / 7937.5 * 100
    assert geo.world_to_pct(geo.opendota_cell_to_world(cell)) == pytest.approx(legacy)


def test_pct_roundtrip():
    for value in (-8192.0, -1234.5, 0.0, 8064.0):
        assert geo.pct_to_world(geo.world_to_pct(value)) == pytest.approx(value)


def test_tower_positions_inside_map():
    for team in geo.TOWER_POSITIONS.values():
        for x, y in team.values():
            assert 0 <= geo.world_to_pct(x) <= 100
            assert 0 <= geo.world_to_pct(y) <= 100


def test_pct_is_the_frontend_formula_and_can_be_clamped():
    for value in (-8192.0, -1234.5, 0.0, 8064.0):
        assert geo.world_to_pct(value) == pytest.approx((value + 8192) / 16256 * 100)
    # real hero positions slightly exceed the minimap area (measured: x 8508, y -8405)
    assert geo.world_to_pct(8508.0) > 100 and geo.world_to_pct(-8405.0) < 0
    assert geo.world_to_pct(8508.0, clamp=True) == 100.0
    assert geo.world_to_pct(-8405.0, clamp=True) == 0.0
    assert geo.world_to_pct(0.0, clamp=True) == geo.world_to_pct(0.0)


def test_building_defaults_match_the_replay_positions():
    """Entity positions of replay 8762775364; the old defaults were up to 540 units off."""
    real = {
        ("dire", "bot_t1"): (6269.3, -2240.0),
        ("radiant", "mid_t2"): (-3190.3, -2926.3),
        ("radiant", "top_t2"): (-6501.0, -872.0),
        ("radiant", "bot_t2"): (-360.0, -6256.0),
        ("radiant", "bot_t1"): (4859.9, -6379.3),
        ("radiant", "ancient_bot"): (-5392.0, -5192.0),
        ("dire", "ancient_top"): (4944.0, 4776.0),
    }
    for (team, key), (x, y) in real.items():
        tx, ty = geo.TOWER_POSITIONS[team][key]
        assert geo.distance(tx, ty, x, y) < 5, (team, key)
    assert geo.distance(*geo.BARRACKS_POSITIONS["radiant"]["mid_melee"], -4672.0, -4552.0) < 5
    assert geo.distance(*geo.BARRACKS_POSITIONS["dire"]["top_range"], 3894.0, 6025.0) < 5
    assert geo.distance(*geo.ANCIENT_POSITIONS["radiant"], -5920.0, -5352.0) < 5


def test_building_defaults_are_those_of_the_current_game_build():
    """All 36 building entities of a 7.41f replay (match 9037980041, game build 6956).

    A patch that moves a tower shows up here once this fixture is replaced by the buildings
    of a replay of that patch (``buildings`` of the analyzer output)."""
    data = json.loads((FIXTURES / "buildings_7_41f.json").read_text(encoding="utf-8"))
    assert data["game_build"] == 6956 and len(data["buildings"]) == 36
    seen = set()
    for building in data["buildings"]:
        info = parser.parse_building_name(building["name"])
        table = {"tower": geo.TOWER_POSITIONS, "barracks": geo.BARRACKS_POSITIONS}.get(info["type"])
        expected = table[info["team"]][info["key"]] if table else geo.ANCIENT_POSITIONS[info["team"]]
        assert geo.distance(building["x"], building["y"], *expected) < 1.0, building["name"]
        seen.add((info["team"], info["type"], info["key"]))
    assert len(seen) == 36


def test_default_buildings_are_complete_and_distinct():
    for team in ("radiant", "dire"):
        assert len(geo.TOWER_POSITIONS[team]) == 11 and len(geo.BARRACKS_POSITIONS[team]) == 6
        points = list(geo.TOWER_POSITIONS[team].values()) + list(geo.BARRACKS_POSITIONS[team].values())
        for x, y in points:
            assert 0 <= geo.world_to_pct(x) <= 100 and 0 <= geo.world_to_pct(y) <= 100
            assert (x + y < 0) == (team == "radiant")       # each base on its own side of the river
        # far enough apart for the nearest-position naming in the parser
        assert min(geo.distance(*a, *b) for i, a in enumerate(points) for b in points[i + 1:]) > 400
