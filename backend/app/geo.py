"""Map geometry shared by the parsers, the analyzer and the AI coach.

Every x/y position stored in ``MatchData`` (wards, smokes, teamfight
positions, ability casts, laning snapshots, buildings) is in Dota 2 *world
units*. Source 2 encodes an entity origin as a 128-unit cell plus an offset
inside that cell, so::

    world = cell * 128 + vec - 16384

OpenDota's ``obs_log`` / ``sen_log`` x/y values are already in cell units
(roughly 64..192, possibly fractional), so for those ``world = (x - 128) * 128``.

The playable area shown on ``frontend/static/assets/dota2_map.jpg`` spans
cells 64..191, i.e. world -8192..8064. ``world_to_pct`` maps that range to
0..100 for ECharts; the frontend uses the same formula.

Exception: ``PlayerData.lane_pos`` keeps OpenDota's raw cell heatmap
(``{x: {y: count}}``) because it is a histogram, not a position list.
"""
from __future__ import annotations

import math

CELL_SIZE = 128
WORLD_ORIGIN_OFFSET = 16384

# Playable area covered by the minimap image (cells 64..191).
MAP_WORLD_MIN = -8192.0
MAP_WORLD_MAX = 8064.0
MAP_WORLD_SPAN = MAP_WORLD_MAX - MAP_WORLD_MIN  # 16256


def cell_to_world(cell: float, vec: float | None = None) -> float:
    """Convert a Source 2 cell (+ optional in-cell offset) to world units.

    When the in-cell offset is unknown the cell centre is used.
    """
    if vec is None:
        vec = CELL_SIZE / 2
    return cell * CELL_SIZE + vec - WORLD_ORIGIN_OFFSET


def opendota_cell_to_world(cell: float) -> float:
    """Convert an OpenDota ward-log / lane_pos cell coordinate to world units."""
    return (cell - 128) * CELL_SIZE


def world_to_pct(value: float) -> float:
    """Map a world coordinate to 0..100 (minimap / ECharts space)."""
    return (value - MAP_WORLD_MIN) / MAP_WORLD_SPAN * 100.0


def pct_to_world(pct: float) -> float:
    """Inverse of :func:`world_to_pct`."""
    return pct / 100.0 * MAP_WORLD_SPAN + MAP_WORLD_MIN


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Euclidean distance between two world-space points."""
    return math.hypot(ax - bx, ay - by)


# Approximate building positions in world units (accurate to a few hundred
# units). Prefer positions extracted from the replay (``MatchData.buildings``)
# when available. Keys follow the OpenDota tower bit layout:
# t1/t2/t3 per lane plus the two tier-4 towers ("ancient_top"/"ancient_bot").
TOWER_POSITIONS: dict[str, dict[str, tuple[float, float]]] = {
    "radiant": {
        "top_t1": (-6250.0, 1830.0),
        "top_t2": (-6120.0, -870.0),
        "top_t3": (-6600.0, -3400.0),
        "mid_t1": (-1550.0, -1400.0),
        "mid_t2": (-3500.0, -2700.0),
        "mid_t3": (-4600.0, -4100.0),
        "bot_t1": (4900.0, -6100.0),
        "bot_t2": (-550.0, -6100.0),
        "bot_t3": (-3950.0, -6100.0),
        "ancient_top": (-5700.0, -4850.0),
        "ancient_bot": (-5400.0, -5200.0),
    },
    "dire": {
        "top_t1": (-5300.0, 5900.0),
        "top_t2": (-100.0, 6000.0),
        "top_t3": (3550.0, 5800.0),
        "mid_t1": (520.0, 650.0),
        "mid_t2": (2500.0, 2100.0),
        "mid_t3": (4300.0, 3750.0),
        "bot_t1": (6250.0, -1700.0),
        "bot_t2": (6300.0, 400.0),
        "bot_t3": (6300.0, 3050.0),
        "ancient_top": (4950.0, 4800.0),
        "ancient_bot": (5250.0, 4450.0),
    },
}

ANCIENT_POSITIONS: dict[str, tuple[float, float]] = {
    "radiant": (-5900.0, -5350.0),
    "dire": (5500.0, 5000.0),
}

FOUNTAIN_POSITIONS: dict[str, tuple[float, float]] = {
    "radiant": (-7100.0, -6600.0),
    "dire": (7000.0, 6450.0),
}
