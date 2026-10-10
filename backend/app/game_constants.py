"""Game rules the backend computes with: one place, one patch stamp.

Two kinds of values live here:

* rules no fetched file states (XP per level, respawn times, day / night,
  building vision, Roshan's respawn window, the skill-point schedule);
* the built-in values of the rules that *are* fetched (ward, smoke and
  true-sight numbers of ``data_manager.game_rules()``, the default hero
  vision of ``data_manager.hero_vision()``): what the code uses while no
  game data is on disk.

``RULES_PATCH`` names the game version every value was checked against
(patch audit of 2026-10-10: ``items.txt`` / ``npc_units.txt`` of build 6957,
dotaconstants, the 7.36-7.41f patch notes and real replays). A test ties its
major version to ``data_manager.DEFAULT_PATCH_LABEL``: supporting a new major
patch turns the suite red until this file was reviewed.

Not here on purpose: heuristics (phase bounds, "nearby" radii, teamfight
gaps), benchmark tables (``ai/pro_patterns.py``), map geometry (``geo.py``)
and facts only the knowledge texts state (rune timings, Aegis duration,
buyback cost), which have no second copy in code.

Stdlib only, so the parser, the analyzer, the mock and the AI layer can all
import it. World units and seconds unless a name says otherwise.
"""
from __future__ import annotations

RULES_PATCH = "7.41f"

# ---------------------------------------------------------------------------
# Experience and skill points
# ---------------------------------------------------------------------------

# Cumulative XP needed to reach each hero level (index i = level i + 1):
# dotaconstants ``xp_level``, all 30 values.
XP_FOR_LEVEL: tuple[int, ...] = (
    0, 240, 640, 1160, 1760, 2440, 3200, 4000, 4900, 5900,          # 1-10
    7000, 8200, 9500, 10900, 12400, 14000, 15700, 17500, 19400, 21400,  # 11-20
    23600, 26000, 28600, 31400, 34400, 38400, 43400, 49400, 56400, 63900,  # 21-30
)
MAX_HERO_LEVEL = len(XP_FOR_LEVEL)

# Since 7.40 (2025-12-15) talents have points of their own, one at each of
# these levels, and every hero level up to 22 grants a skill point: the k-th
# upgrade that is not a talent is made at level k at the earliest.
TALENT_LEVELS: tuple[int, ...] = (10, 15, 20, 25, 27, 28, 29, 30)
LAST_SKILL_POINT_LEVEL = 22
TALENT_POINTS_SINCE = (7, 40)
TALENT_POINTS_RELEASED = 1765785600  # release of 7.40 (Valve's patch feed), unix time
# Before 7.40 a talent cost a skill point: the n-th upgrade of a build
# (abilities and talents alike) was made at the n-th of these hero levels
# (17, 19 and 21-24 granted no point).
SKILL_POINT_LEVELS_BEFORE_TALENT_POINTS: tuple[int, ...] = (
    *range(1, 17), 18, 20, *range(25, 31),
)
# A hero has fifteen ability levels, four of each basic ability and three of
# the ultimate, and the third level of the ultimate needs hero level 18.
BASIC_ABILITY_LEVELS = 4
ULTIMATE_LEVELS = 3
HERO_ABILITY_LEVELS = 3 * BASIC_ABILITY_LEVELS + ULTIMATE_LEVELS
LAST_ULTIMATE_LEVEL_AT = 18

# ---------------------------------------------------------------------------
# Respawn, day and night, Roshan
# ---------------------------------------------------------------------------

# Base respawn time by hero level (index i = level i + 1; buybacks and
# talents aside). Levels 1-19 were confirmed on 180 deaths of three replays;
# the audit found no sample for level 20 and above.
RESPAWN_SECONDS: tuple[int, ...] = (
    12, 15, 18, 21, 24, 26, 28, 30, 32, 34, 36, 44, 46, 48, 50, 52, 54,
    65, 70, 75, 80, 85, 90, 95, 100,
)

DAY_NIGHT_PERIOD_S = 300        # 5 minutes per half-cycle, the game starts at day

# Roshan respawns at a random time in this window after his death (7.38 notes).
ROSHAN_RESPAWN_MIN_S = 480
ROSHAN_RESPAWN_MAX_S = 660

# ---------------------------------------------------------------------------
# Vision of heroes and buildings
# ---------------------------------------------------------------------------

# Default hero vision (day, night). The few heroes that differ (Night Stalker
# 800 / 1800, Slark 1800 / 1800, Sniper 1800 / 1400 ...) are in the fetched
# hero data and the committed hero catalog.
HERO_DAY_VISION = 1800
HERO_NIGHT_VISION = 800
# Buildings (npc_units.txt; not in the fetched game data).
TOWER_DAY_VISION = 1900
TOWER_T1_NIGHT_VISION = 800     # tier 1
TOWER_NIGHT_VISION = 1100       # tier 2, 3 and 4
ANCIENT_VISION = 2600           # day and night
FOUNTAIN_VISION = 1800          # day and night

# ---------------------------------------------------------------------------
# Wards, smoke, true sight: built-in values of the fetched item attributes
# ---------------------------------------------------------------------------

OBSERVER_WARD_DURATION_S = 360
OBSERVER_WARD_VISION = 1600
SENTRY_WARD_DURATION_S = 420
SENTRY_WARD_TRUE_SIGHT = 1050   # true sight only, no ground vision
SMOKE_DURATION_S = 45
SMOKE_APPLICATION_RADIUS = 1200
SMOKE_BREAK_RADIUS = 1025       # an enemy hero or tower this close ends the smoke
GEM_TRUE_SIGHT = 900
DUST_RADIUS = 1050
DUST_DURATION_S = 12

# The same values under the names ``data_manager.game_rules()`` uses.
ITEM_RULE_DEFAULTS: dict[str, int] = {
    "observer_ward_duration": OBSERVER_WARD_DURATION_S,
    "observer_ward_vision": OBSERVER_WARD_VISION,
    "sentry_ward_duration": SENTRY_WARD_DURATION_S,
    "sentry_ward_true_sight": SENTRY_WARD_TRUE_SIGHT,
    "smoke_duration": SMOKE_DURATION_S,
    "smoke_application_radius": SMOKE_APPLICATION_RADIUS,
    "smoke_break_radius": SMOKE_BREAK_RADIUS,
    "gem_true_sight": GEM_TRUE_SIGHT,
    "dust_radius": DUST_RADIUS,
    "dust_duration": DUST_DURATION_S,
}

# Natural ward lifetimes by ``WardEvent.ward_type``.
WARD_LIFETIME_S: dict[str, int] = {
    "observer": OBSERVER_WARD_DURATION_S,
    "sentry": SENTRY_WARD_DURATION_S,
}

# Item rules that changed with a patch and whose old value is known:
# (first version with today's value, name in ``ITEM_RULE_DEFAULTS``, value
# before it). A match of a known older version is judged by the rule of its
# time; every other rule is today's (sentry true sight: 7.38 notes).
ITEM_RULE_HISTORY: tuple[tuple[tuple[int, int], str, int], ...] = (
    ((7, 38), "sentry_ward_true_sight", 1000),
)
