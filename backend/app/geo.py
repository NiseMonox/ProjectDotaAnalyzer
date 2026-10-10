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
0..100 for ECharts; the frontend uses the same formula
(``(v + 8192) / 16256 * 100``).

Real positions can lie slightly outside that range: heroes walking along the
map edge were measured at x = 8508 and y = -8405 (102.7 % / -1.3 %). The
mapping is deliberately not widened (it has to match the minimap image), so a
renderer that must stay inside the image clamps the percentage
(``world_to_pct(v, clamp=True)``).

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


def world_to_pct(value: float, clamp: bool = False) -> float:
    """Map a world coordinate to 0..100 (minimap / ECharts space).

    Positions on the very edge of the map give values a little below 0 or
    above 100; ``clamp=True`` pins them to the image.
    """
    pct = (value - MAP_WORLD_MIN) / MAP_WORLD_SPAN * 100.0
    return max(0.0, min(100.0, pct)) if clamp else pct


def pct_to_world(pct: float) -> float:
    """Inverse of :func:`world_to_pct`."""
    return pct / 100.0 * MAP_WORLD_SPAN + MAP_WORLD_MIN


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Euclidean distance between two world-space points."""
    return math.hypot(ax - bx, ay - by)


# Default building positions in world units, measured from the building
# entities of two replays (game builds 6717 and 6762, patches 7.40 and 7.41).
# Only the two outer tier-1 towers moved between those builds (by 110-190
# units); the newer position is used. Seven replays of 7.41f (build 6956)
# have all 36 buildings within half a unit of these values (tests/fixtures/
# buildings_7_41f.json). Prefer the positions extracted from the replay
# (``MatchData.buildings``) when available. Keys follow the OpenDota tower bit
# layout: t1/t2/t3 per lane plus the two tier-4 towers
# ("ancient_top"/"ancient_bot").
TOWER_POSITIONS: dict[str, dict[str, tuple[float, float]]] = {
    "radiant": {
        "top_t1": (-6336.0, 1856.0),
        "top_t2": (-6501.0, -872.0),
        "top_t3": (-6592.0, -3408.0),
        "mid_t1": (-1544.0, -1408.0),
        "mid_t2": (-3190.0, -2926.0),
        "mid_t3": (-4640.0, -4144.0),
        "bot_t1": (4860.0, -6379.0),
        "bot_t2": (-360.0, -6256.0),
        "bot_t3": (-3952.0, -6112.0),
        "ancient_top": (-5712.0, -4864.0),
        "ancient_bot": (-5392.0, -5192.0),
    },
    "dire": {
        "top_t1": (-5275.0, 6036.0),
        "top_t2": (-128.0, 6016.0),
        "top_t3": (3552.0, 5776.0),
        "mid_t1": (524.0, 652.0),
        "mid_t2": (2496.0, 2112.0),
        "mid_t3": (4272.0, 3759.0),
        "bot_t1": (6269.0, -2240.0),
        "bot_t2": (6400.0, 384.0),
        "bot_t3": (6336.0, 3032.0),
        "ancient_top": (4944.0, 4776.0),
        "ancient_bot": (5280.0, 4432.0),
    },
}

# Barracks, keyed like the OpenDota barracks bit layout ("<lane>_melee" /
# "<lane>_range"). Same source as TOWER_POSITIONS.
BARRACKS_POSITIONS: dict[str, dict[str, tuple[float, float]]] = {
    "radiant": {
        "top_melee": (-6336.0, -3758.0), "top_range": (-6844.0, -3759.0),
        "mid_melee": (-4672.0, -4552.0), "mid_range": (-5060.0, -4199.0),
        "bot_melee": (-4280.0, -6360.0), "bot_range": (-4279.0, -5853.0),
    },
    "dire": {
        "top_melee": (3898.0, 5496.0), "top_range": (3894.0, 6025.0),
        "mid_melee": (4702.0, 3824.0), "mid_range": (4336.0, 4183.0),
        "bot_melee": (6592.0, 3392.0), "bot_range": (6064.0, 3376.0),
    },
}

ANCIENT_POSITIONS: dict[str, tuple[float, float]] = {
    "radiant": (-5920.0, -5352.0),
    "dire": (5528.0, 5000.0),
}

FOUNTAIN_POSITIONS: dict[str, tuple[float, float]] = {
    "radiant": (-7100.0, -6600.0),
    "dire": (7000.0, 6450.0),
}
