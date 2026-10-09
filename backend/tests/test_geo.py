"""Tests for the shared map-geometry helpers."""
import pytest

from backend.app import geo


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
