"""Mock data generator for development and testing.

Produces a deterministic MatchData modelled on real match 8713267082. All
positions are Dota 2 world units (see ``geo.py``); randomness comes from a
private seeded RNG so the global ``random`` state is never touched.

The match is internally consistent, with the same identifiers a parsed match
has (see the ``models`` module docstring): one kill log drives the scoreboard
kills/deaths/assists, ``killed`` / ``killed_by``, the teamfights and the hero
positions (``position_log``, laning snapshots, casts); the team gold/XP
timeline is the sum of the per-player series.
"""
from __future__ import annotations

import math
import random
import threading
from collections import Counter
from dataclasses import dataclass, field

from . import geo
from .game_constants import WARD_LIFETIME_S, XP_FOR_LEVEL
from .models import (
    AbilityCast,
    AbilityEvent,
    ItemEvent,
    KillEvent,
    LaningCreepKill,
    LaningDamageEvent,
    LaningRegenEvent,
    LaningSnapshot,
    MatchData,
    PlayerData,
    PositionSnapshot,
    SkillEvent,
    SmokeEvent,
    TeamfightEvent,
    TeamfightPlayer,
    WardEvent,
)

# (hero_id, localized_name, slug, [(display_name, ability_key), ...])
HEROES = [
    (1, "Anti-Mage", "antimage", [
        ("Mana Break", "antimage_mana_break"), ("Blink", "antimage_blink"),
        ("Counterspell", "antimage_counterspell"), ("Mana Void", "antimage_mana_void"),
    ]),
    (2, "Axe", "axe", [
        ("Berserker's Call", "axe_berserkers_call"), ("Battle Hunger", "axe_battle_hunger"),
        ("Counter Helix", "axe_counter_helix"), ("Culling Blade", "axe_culling_blade"),
    ]),
    (5, "Crystal Maiden", "crystal_maiden", [
        ("Crystal Nova", "crystal_maiden_crystal_nova"), ("Frostbite", "crystal_maiden_frostbite"),
        ("Arcane Aura", "crystal_maiden_brilliance_aura"), ("Freezing Field", "crystal_maiden_freezing_field"),
    ]),
    (6, "Drow Ranger", "drow_ranger", [
        ("Frost Arrows", "drow_ranger_frost_arrows"), ("Gust", "drow_ranger_wave_of_silence"),
        ("Multishot", "drow_ranger_multishot"), ("Marksmanship", "drow_ranger_marksmanship"),
    ]),
    (7, "Earthshaker", "earthshaker", [
        ("Fissure", "earthshaker_fissure"), ("Enchant Totem", "earthshaker_enchant_totem"),
        ("Aftershock", "earthshaker_aftershock"), ("Echo Slam", "earthshaker_echo_slam"),
    ]),
    (8, "Juggernaut", "juggernaut", [
        ("Blade Fury", "juggernaut_blade_fury"), ("Healing Ward", "juggernaut_healing_ward"),
        ("Blade Dance", "juggernaut_blade_dance"), ("Omnislash", "juggernaut_omni_slash"),
    ]),
    (9, "Mirana", "mirana", [
        ("Sacred Arrow", "mirana_arrow"), ("Leap", "mirana_leap"),
        ("Starstorm", "mirana_starfall"), ("Moonlight Shadow", "mirana_invis"),
    ]),
    (11, "Shadow Fiend", "nevermore", [
        ("Shadowraze", "nevermore_shadowraze1"), ("Necromastery", "nevermore_necromastery"),
        ("Presence of the Dark Lord", "nevermore_dark_lord"), ("Requiem of Souls", "nevermore_requiem"),
    ]),
    (14, "Pudge", "pudge", [
        ("Meat Hook", "pudge_meat_hook"), ("Rot", "pudge_rot"),
        ("Flesh Heap", "pudge_flesh_heap"), ("Dismember", "pudge_dismember"),
    ]),
    (25, "Lina", "lina", [
        ("Dragon Slave", "lina_dragon_slave"), ("Light Strike Array", "lina_light_strike_array"),
        ("Fiery Soul", "lina_fiery_soul"), ("Laguna Blade", "lina_laguna_blade"),
    ]),
    (26, "Lion", "lion", [
        ("Earth Spike", "lion_impale"), ("Hex", "lion_voodoo"),
        ("Mana Drain", "lion_mana_drain"), ("Finger of Death", "lion_finger_of_death"),
    ]),
    (30, "Witch Doctor", "witch_doctor", [
        ("Paralyzing Cask", "witch_doctor_paralyzing_cask"), ("Voodoo Restoration", "witch_doctor_voodoo_restoration"),
        ("Maledict", "witch_doctor_maledict"), ("Death Ward", "witch_doctor_death_ward"),
    ]),
    (31, "Lich", "lich", [
        ("Frost Blast", "lich_frost_nova"), ("Frost Shield", "lich_frost_shield"),
        ("Sinister Gaze", "lich_sinister_gaze"), ("Chain Frost", "lich_chain_frost"),
    ]),
    (35, "Sniper", "sniper", [
        ("Shrapnel", "sniper_shrapnel"), ("Headshot", "sniper_headshot"),
        ("Take Aim", "sniper_take_aim"), ("Assassinate", "sniper_assassinate"),
    ]),
    (36, "Templar Assassin", "templar_assassin", [
        ("Refraction", "templar_assassin_refraction"), ("Meld", "templar_assassin_meld"),
        ("Psi Blades", "templar_assassin_psi_blades"), ("Psionic Trap", "templar_assassin_psionic_trap"),
    ]),
    (41, "Faceless Void", "faceless_void", [
        ("Time Walk", "faceless_void_time_walk"), ("Time Dilation", "faceless_void_time_dilation"),
        ("Time Lock", "faceless_void_time_lock"), ("Chronosphere", "faceless_void_chronosphere"),
    ]),
    (47, "Viper", "viper", [
        ("Poison Attack", "viper_poison_attack"), ("Nethertoxin", "viper_nethertoxin"),
        ("Corrosive Skin", "viper_corrosive_skin"), ("Viper Strike", "viper_viper_strike"),
    ]),
    (53, "Nature's Prophet", "furion", [
        ("Sprout", "furion_sprout"), ("Teleportation", "furion_teleportation"),
        ("Nature's Call", "furion_force_of_nature"), ("Wrath of Nature", "furion_wrath_of_nature"),
    ]),
    (74, "Invoker", "invoker", [
        ("Quas", "invoker_quas"), ("Wex", "invoker_wex"),
        ("Exort", "invoker_exort"), ("Invoke", "invoker_invoke"),
    ]),
    (86, "Rubick", "rubick", [
        ("Telekinesis", "rubick_telekinesis"), ("Fade Bolt", "rubick_fade_bolt"),
        ("Arcane Supremacy", "rubick_null_field"), ("Spell Steal", "rubick_spell_steal"),
    ]),
]

# Items: (display_name, item_key for CDN)
EARLY_ITEMS = [
    ("Tango", "tango"),
    ("Healing Salve", "flask"),
    ("Iron Branch", "branches"),
    ("Faerie Fire", "faerie_fire"),
    ("Clarity", "clarity"),
    ("Enchanted Mango", "enchanted_mango"),
    ("Quelling Blade", "quelling_blade"),
    ("Gauntlets of Strength", "gauntlets"),
    ("Slippers of Agility", "slippers"),
    ("Mantle of Intelligence", "mantle"),
    ("Circlet", "circlet"),
    ("Magic Stick", "magic_stick"),
]

MID_ITEMS = [
    ("Boots of Speed", "boots"),
    ("Magic Wand", "magic_wand"),
    ("Bracer", "bracer"),
    ("Wraith Band", "wraith_band"),
    ("Null Talisman", "null_talisman"),
    ("Power Treads", "power_treads"),
    ("Phase Boots", "phase_boots"),
    ("Arcane Boots", "arcane_boots"),
    ("Ring of Health", "ring_of_health"),
    ("Void Stone", "void_stone"),
    ("Mekansm", "mekansm"),
    ("Blink Dagger", "blink"),
    ("Force Staff", "force_staff"),
    ("Ghost Scepter", "ghost"),
    ("Drum of Endurance", "ancient_janggo"),
]

LATE_ITEMS = [
    ("Black King Bar", "black_king_bar"),
    ("Butterfly", "butterfly"),
    ("Daedalus", "greater_crit"),
    ("Desolator", "desolator"),
    ("Eye of Skadi", "skadi"),
    ("Heart of Tarrasque", "heart"),
    ("Manta Style", "manta"),
    ("Monkey King Bar", "monkey_king_bar"),
    ("Satanic", "satanic"),
    ("Aghanim's Scepter", "ultimate_scepter"),
    ("Refresher Orb", "refresher"),
    ("Scythe of Vyse", "sheepstick"),
    ("Assault Cuirass", "assault"),
    ("Shiva's Guard", "shivas_guard"),
    ("Linken's Sphere", "sphere"),
]

SUPPORT_ITEMS = [
    ("Observer Ward", "ward_observer"),
    ("Sentry Ward", "ward_sentry"),
    ("Smoke of Deceit", "smoke_of_deceit"),
    ("Dust of Appearance", "dust"),
    ("Town Portal Scroll", "tpscroll"),
    ("Glimmer Cape", "glimmer_cape"),
    ("Aether Lens", "aether_lens"),
    ("Lotus Orb", "lotus_orb"),
]

# Talent choices per hero level (ability keys). Real keys, so the mock takes
# the same "special_bonus_*" code path as parsed matches; the display names
# come from parser.ability_display_name, like theirs.
_TALENTS: dict[int, list[str]] = {
    10: ["special_bonus_hp_200", "special_bonus_attack_speed_20"],
    15: ["special_bonus_attack_damage_20", "special_bonus_magic_resistance_10"],
    20: ["special_bonus_all_stats_10", "special_bonus_lifesteal_15"],
    25: ["special_bonus_hp_300", "special_bonus_spell_amplify_10"],
}

# Private RNG (seeded per generate_mock_match call) and a lock so concurrent
# callers cannot interleave draws and break determinism.
_rng = random.Random()
_MOCK_LOCK = threading.Lock()
_MOCK_SEED = 42

# Invented player names, in the order of the player table below (the mock
# must not show the persona names of the real match it is modelled on).
PLAYER_NAMES = [
    "Mock Squirrel", "Mock Archer", "Mock Warlock", "Mock Stormcaller", "Mock Axeman",
    "Mock Shaman", "Mock Viper", "Mock Lion", "Mock Dragon", "Mock Bristle",
]

# Playable area bounds in world units (the minimap spans -8192..8064).
MAP_MIN = -7400.0
MAP_MAX = 7400.0

POSITION_LOG_START = -90   # first spawn (strategy time is over 90 s before the horn)
POSITION_LOG_STEP = 5      # seconds between position_log samples
LANING_END = 600           # last laning snapshot


def _clamp(value: float) -> float:
    return max(MAP_MIN, min(MAP_MAX, value))


def _rand_map_pos() -> tuple[float, float]:
    return (
        _rng.uniform(MAP_MIN, MAP_MAX),
        _rng.uniform(MAP_MIN, MAP_MAX),
    )


def _team_of(idx: int) -> str:
    """Team of a player by index into the 10-player table (0-4 Radiant)."""
    return "radiant" if idx < 5 else "dire"


def _slot_of(idx: int) -> int:
    return idx if idx < 5 else 128 + idx - 5


def _generate_item_build(
    role: str, duration: int,
) -> tuple[list[ItemEvent], list[str], list[str]]:
    """Returns (item_events, final_display_names, final_item_keys)."""
    items: list[ItemEvent] = []
    item_keys: list[str] = []
    # Starting items purchased during strategy phase (-90s to 0s)
    t = -_rng.randint(85, 89)

    for _ in range(_rng.randint(3, 5)):
        name, key = _rng.choice(EARLY_ITEMS)
        items.append(ItemEvent(item_name=name, time=t, item_key=key))
        item_keys.append(key)
        t += _rng.randint(0, 3)

    # Early-mid game items (after horn at 0:00)
    t = _rng.randint(60, 180)
    num_mid = _rng.randint(3, 6)
    for _ in range(num_mid):
        t += _rng.randint(60, 300)
        if t > duration:
            break
        pool = MID_ITEMS if role != "support" else MID_ITEMS + SUPPORT_ITEMS
        name, key = _rng.choice(pool)
        items.append(ItemEvent(item_name=name, time=t, item_key=key))
        item_keys.append(key)

    # Late game items
    num_late = _rng.randint(2, 4) if role != "support" else _rng.randint(0, 2)
    for _ in range(num_late):
        t += _rng.randint(180, 480)
        if t > duration:
            break
        name, key = _rng.choice(LATE_ITEMS)
        items.append(ItemEvent(item_name=name, time=t, item_key=key))
        item_keys.append(key)

    n = min(6, len(items))
    final_names = [items[-i].item_name for i in range(1, n + 1)]
    final_keys = [items[-i].item_key for i in range(1, n + 1)]
    return items, final_names, final_keys


# ---------------------------------------------------------------------------
# Economy and experience curves
# ---------------------------------------------------------------------------

def _per_minute_series(total: float, duration: int) -> list[int]:
    """Cumulative per-minute series (minute 0 .. floor(duration / 60)) that
    reaches ``total`` at ``duration``. Income is lower during the laning phase."""
    minutes = duration // 60
    rates = [(0.3 + 0.01 * m if m < 10 else 1.0) * _rng.uniform(0.85, 1.15) for m in range(minutes + 1)]
    rates[minutes] *= (duration - minutes * 60) / 60  # the last, partial minute
    scale = total / sum(rates)
    series = [0]
    earned = 0.0
    for m in range(minutes):
        earned += rates[m] * scale
        series.append(int(earned))
    return series


def _series_at(series: list[int], total: float, t: float, duration: int) -> float:
    """Value of a per-minute series at game time ``t`` (linear between samples)."""
    if t <= 0:
        return 0.0
    minute = int(t // 60)
    if minute >= len(series) - 1:
        last_t = (len(series) - 1) * 60
        frac = (t - last_t) / max(1, duration - last_t)
        return series[-1] + (total - series[-1]) * max(0.0, min(1.0, frac))
    frac = (t - minute * 60) / 60
    return series[minute] + (series[minute + 1] - series[minute]) * frac


def _level_at(xp: float, xp_table: tuple[int, ...]) -> int:
    level = 1
    for lvl, need in enumerate(xp_table, start=1):
        if xp >= need:
            level = lvl
    return level


def _level_times(
    xp_t: list[int], total_xp: float, max_level: int, duration: int, xp_table: tuple[int, ...],
) -> dict[int, int]:
    """Game time at which each hero level up to ``max_level`` is reached."""
    times = {1: 0}
    level, t = 1, 0
    while level < max_level and t <= duration:
        t += 5
        reached = min(max_level, _level_at(_series_at(xp_t, total_xp, t, duration), xp_table))
        for lvl in range(level + 1, reached + 1):
            times[lvl] = t
        level = max(level, reached)
    for lvl in range(2, max_level + 1):  # levels the curve never reaches: end of game
        times.setdefault(lvl, duration)
    return times


def _counted_series(event_times: list[int], final: int, duration: int) -> list[int]:
    """Per-minute counter: the laning events themselves up to 10:00, then a
    steady climb to ``final`` at the end of the game."""
    minutes = duration // 60
    at_ten = sum(1 for t in event_times if t <= LANING_END)
    series = []
    for m in range(minutes + 1):
        if m <= 10:
            series.append(sum(1 for t in event_times if t <= m * 60))
        else:
            series.append(at_ten + int((final - at_ten) * (m * 60 - LANING_END) / (duration - LANING_END)))
    return series


def _generate_skill_build(abilities: list[tuple[str, str]], level_times: dict[int, int]) -> list[SkillEvent]:
    """Skill build under the rules of the mock's patch (7.40).

    abilities: list of (display_name, ability_key) tuples, the ultimate last.
    Every hero level brings a skill point, spent when the level is reached;
    a talent has a point of its own (it took the level's skill point before
    7.40). The ultimate is learned at levels 6, 12 and 18, the n-th level of
    another ability at hero level 2n - 1 at the earliest.
    """
    from .parser import ability_display_name  # lazy: parser imports this module lazily too

    skills: list[SkillEvent] = []
    skill_levels = {name: 0 for name, _ in abilities}
    ult_name, ult_key = abilities[3]

    for lvl in sorted(level_times):
        t = level_times[lvl]

        if lvl in _TALENTS:
            key = _rng.choice(_TALENTS[lvl])
            skills.append(SkillEvent(ability_name=ability_display_name(key), ability_key=key, level=1, time=t))

        if lvl in (6, 12, 18) and skill_levels[ult_name] < 3:
            skill_levels[ult_name] += 1
            skills.append(SkillEvent(
                ability_name=ult_name, ability_key=ult_key,
                level=skill_levels[ult_name], time=t,
            ))
            continue

        regular = [(n, k) for n, k in abilities[:3] if skill_levels[n] < min(4, (lvl + 1) // 2)]
        if not regular:
            # Nothing can take the point (levels 15-17): the game puts it into
            # the attribute bonus by itself, and a skill build has no entry for that.
            continue

        chosen_name, chosen_key = _rng.choice(regular)
        skill_levels[chosen_name] += 1
        skills.append(SkillEvent(
            ability_name=chosen_name, ability_key=chosen_key,
            level=skill_levels[chosen_name], time=t,
        ))

    return skills


def _cell_to_game(cx: float, cy: float) -> tuple[float, float]:
    """Convert OpenDota ward-log cell coordinates to world units."""
    return round(geo.opendota_cell_to_world(cx), 1), round(geo.opendota_cell_to_world(cy), 1)


# Real ward data from match 8713267082 (OpenDota API).
# Format per slot: list of (ward_type, time, cell_x, cell_y, destroyed_time|None)
_REAL_WARD_RAW: dict[int, list[tuple[str, int, float, float, int | None]]] = {
    0: [  # Hoodwink (support)
        ("observer", 960, 114.7, 160.4, 1320),
        ("observer", 1628, 99.8, 80.7, None),
        ("sentry", 166, 165.7, 91.8, 374),
        ("sentry", 509, 177.5, 83.8, 929),
        ("sentry", 939, 96, 129.4, 1359),
        ("sentry", 956, 108.5, 162.8, 1376),
        ("sentry", 1366, 85.3, 100.1, 1740),
        ("sentry", 1371, 95.6, 89.4, 1774),
        ("sentry", 1626, 109.3, 79.1, 2046),
    ],
    1: [  # Drow Ranger (carry)
        ("sentry", 1530, 66.9, 121.8, 1950),
    ],
    2: [  # Warlock (support)
        ("observer", -47, 139.4, 133.8, 313),
        ("observer", 223, 78.8, 173.6, 583),
        ("observer", 410, 86.5, 159.8, 566),
        ("observer", 509, 100.1, 177.8, 869),
        ("observer", 947, 118.3, 94, 1307),
        ("observer", 1129, 94.2, 119.6, 1192),
        ("observer", 1152, 127.5, 110.4, 1512),
        ("observer", 1382, 70, 118.4, 1742),
        ("observer", 1842, 80.2, 81.4, None),
        ("observer", 1844, 78.1, 86.1, None),
        ("observer", 1845, 79.6, 83.9, None),
        ("observer", 1847, 81.4, 82.3, None),
        ("sentry", -36, 126.1, 125.2, 384),
        ("sentry", 59, 95.3, 164.2, 117),
        ("sentry", 407, 84, 154.2, 690),
        ("sentry", 728, 81.1, 152.8, 1148),
        ("sentry", 1003, 116, 98.3, 1021),
        ("sentry", 1127, 94.1, 119.9, 1198),
        ("sentry", 1141, 120.4, 113.2, 1561),
        ("sentry", 1246, 98.8, 96.1, 1666),
        ("sentry", 1347, 102.2, 102.2, 1744),
        ("sentry", 1530, 69.3, 119.9, 1950),
        ("sentry", 1556, 81.5, 113.4, 1596),
        ("sentry", 1628, 93.6, 119.7, 2048),
        ("sentry", 1808, 76, 83.5, None),
        ("sentry", 1837, 77.7, 81.9, None),
        ("sentry", 1838, 78.7, 80.4, None),
        ("sentry", 1839, 76.9, 84.8, None),
        ("sentry", 1841, 78.7, 82.9, None),
        ("sentry", 1858, 82.3, 81, None),
        ("sentry", 1860, 76.9, 87.8, None),
        ("sentry", 1868, 80.8, 80.2, None),
    ],
    3: [],  # Razor (offlane)
    4: [  # Axe (mid)
        ("observer", -45, 133.5, 125.4, 315),
        ("observer", 352, 113.6, 129.3, 712),
    ],
    128: [  # Shadow Shaman (support)
        ("observer", 200, 172, 86.7, 211),
        ("observer", 372, 162.3, 92.5, 732),
        ("observer", 706, 152.6, 81, 1066),
        ("observer", 859, 158, 94.3, 1219),
        ("observer", 983, 118.3, 94, 1006),
        ("observer", 1143, 133.2, 79.4, 1503),
        ("observer", 1358, 94.3, 119.9, 1631),
        ("observer", 1364, 83.5, 114.4, 1560),
        ("sentry", 369, 168.1, 94.2, 789),
        ("sentry", 825, 178, 104, 1246),
        ("sentry", 983, 122.3, 97.4, 1010),
        ("sentry", 1047, 127.6, 115.3, 1155),
        ("sentry", 1165, 108.1, 77.1, 1585),
        ("sentry", 1172, 118.2, 93.9, 1592),
        ("sentry", 1268, 95.4, 105.2, 1351),
        ("sentry", 1361, 86, 116.7, 1781),
        ("sentry", 1406, 67.6, 107.4, 1826),
        ("sentry", 1737, 86.1, 101.8, None),
        ("sentry", 1741, 97.7, 102, None),
        ("sentry", 1761, 90.5, 94.9, None),
    ],
    129: [],  # Viper (carry)
    130: [  # Lion (support)
        ("observer", 373, 70.3, 154.3, 733),
        ("observer", 521, 102, 161.9, 881),
        ("observer", 536, 70.7, 167.4, 896),
        ("observer", 746, 70.6, 154, 1106),
        ("observer", 1606, 110.3, 130.3, None),
        ("observer", 1608, 103.8, 132, None),
        ("observer", 1912, 80.1, 82, None),
        ("sentry", 113, 98.8, 170.3, 533),
        ("sentry", 270, 87.8, 163.8, 690),
        ("sentry", 687, 81.9, 154.2, 732),
        ("sentry", 741, 70.2, 164, 1161),
        ("sentry", 775, 181.2, 91.9, 1195),
        ("sentry", 860, 171.9, 102.9, 1280),
        ("sentry", 923, 137.9, 76.9, 1343),
        ("sentry", 1155, 77.8, 137.8, 1575),
        ("sentry", 1189, 93.9, 120.3, 1609),
        ("sentry", 1265, 107.4, 99.2, 1354),
        ("sentry", 1310, 117.5, 94.1, 1730),
        ("sentry", 1603, 119.8, 128.5, 2024),
        ("sentry", 1610, 107.2, 140.1, 2030),
        ("sentry", 1912, 79.5, 81.6, None),
    ],
    131: [],  # Dragon Knight (mid)
    132: [],  # Bristleback (offlane)
}

# Ward lifetimes in seconds. The table above has no "who removed it" column,
# so a ward that died clearly before its expiry counts as dewarded.
_WARD_LIFETIME_S = WARD_LIFETIME_S


def _real_wards_for_player(player_slot: int, duration: int) -> list[WardEvent]:
    """Return real ward data for a player in match 8713267082."""
    raw = _REAL_WARD_RAW.get(player_slot, [])
    wards = []
    for ward_type, t, cx, cy, dt in raw:
        gx, gy = _cell_to_game(cx, cy)
        if dt is not None and dt > duration:
            dt = None  # expired after the game ended: alive at game end
        wards.append(WardEvent(
            ward_type=ward_type, x=gx, y=gy,
            time=t, player_slot=player_slot,
            destroyed_time=dt,
            team="radiant" if player_slot < 128 else "dire",
            dewarded=dt is not None and dt - t < _WARD_LIFETIME_S[ward_type] - 5,
        ))
    return sorted(wards, key=lambda w: w.time)


# Smoke gank targets in world units (enemy jungle / river / lanes)
SMOKE_TARGETS_RADIANT = [
    (4100, 4100), (0, 0), (-400, 3000), (7000, 5100), (5100, 1600),
]
SMOKE_TARGETS_DIRE = [
    (-4100, -4100), (0, 0), (-400, 3000), (-7000, -5100), (-5100, -1600),
]


def _generate_smoke_path(
    team: str, num_waypoints: int,
) -> tuple[float, float, list[list[float]]]:
    """Generate smoke start + movement path toward a target."""
    if team == "radiant":
        sx = _rng.uniform(-6000, -3000)
        sy = _rng.uniform(-6000, -3000)
        target = _rng.choice(SMOKE_TARGETS_RADIANT)
    else:
        sx = _rng.uniform(3000, 6000)
        sy = _rng.uniform(3000, 6000)
        target = _rng.choice(SMOKE_TARGETS_DIRE)

    tx, ty = target
    path: list[list[float]] = []
    for i in range(num_waypoints):
        t_ratio = (i + 1) / (num_waypoints + 1)
        wx = sx + (tx - sx) * t_ratio + _rng.uniform(-800, 800)
        wy = sy + (ty - sy) * t_ratio + _rng.uniform(-800, 800)
        path.append([round(_clamp(wx), 1), round(_clamp(wy), 1)])

    return round(sx, 1), round(sy, 1), path


def _smoke_result(kills_gained: int, deaths_suffered: int, objective_taken: str | None) -> str:
    """Result code of a smoke, with the replay analyzer's precedence:
    traded > kill > objective > caught > no_contact."""
    if kills_gained > 0 and deaths_suffered > 0:
        return "traded"
    if kills_gained > 0:
        return "kill"
    if objective_taken:
        return "objective"
    if deaths_suffered > 0:
        return "caught"
    return "no_contact"


def _generate_smokes(team: str, duration: int) -> list[SmokeEvent]:
    smokes: list[SmokeEvent] = []
    num_smokes = _rng.randint(4, 10)
    slots = list(range(0, 5)) if team == "radiant" else list(range(128, 133))

    for _ in range(num_smokes):
        t = _rng.randint(300, duration - 60)
        participants = _rng.sample(slots, _rng.randint(2, 5))
        num_waypoints = _rng.randint(3, 8)
        sx, sy, path = _generate_smoke_path(team, num_waypoints)
        smoke_duration = _rng.randint(10, 35)

        # Roll what happened; the result code follows from the counts.
        outcome_roll = _rng.random()
        objective = None
        if outcome_roll < 0.35:
            kills_gained, deaths_suffered = _rng.randint(1, 3), 0
        elif outcome_roll < 0.55:
            kills_gained, deaths_suffered = _rng.randint(1, 2), _rng.randint(1, 2)
        elif outcome_roll < 0.75:
            kills_gained, deaths_suffered = 0, 0
            objective = _rng.choice(["tower", "roshan"])
        else:
            kills_gained, deaths_suffered = 0, _rng.randint(0, 1)

        smokes.append(SmokeEvent(
            time=t, team=team, player_slots=participants, player_slot=participants[0],
            x=sx, y=sy, end_time=t + smoke_duration, path=path,
            result=_smoke_result(kills_gained, deaths_suffered, objective),
            kills_gained=kills_gained,
            deaths_suffered=deaths_suffered, objective_taken=objective,
        ))

    return sorted(smokes, key=lambda s: s.time)


# ---------------------------------------------------------------------------
# Kill log: every hero death of the match, consistent with the scoreboard
# ---------------------------------------------------------------------------

@dataclass
class _Kill:
    time: float
    fight: int | None           # index of the teamfight the death belongs to
    victim: int = -1            # indices into the 10-player table
    killer: int = -1
    assists: list[int] = field(default_factory=list)
    x: float = 0.0              # victim position
    y: float = 0.0


def _respawn_time(t: float) -> int:
    """Seconds a hero that dies at game time ``t`` stays dead."""
    return int(min(60, 10 + max(0.0, t) / 40))


FIRST_BLOOD_TIME = 128


def _plan_kill_times(total: int, duration: int) -> tuple[list[_Kill], int]:
    """Death times of the match: teamfights (deaths less than 15 s apart) after
    the laning phase plus isolated pick-offs. Returns (kills in time order,
    number of fights)."""
    num_fights = _rng.randint(10, 11)
    laning_solo = _rng.randint(5, 8)          # besides the first blood
    late_solo = _rng.randint(14, 18)
    in_fights = total - 1 - laning_solo - late_solo
    kills: list[_Kill] = []
    blocked: list[tuple[float, float]] = []   # windows no pick-off may fall into
    segment = (duration - 80 - (LANING_END + 20)) / num_fights
    for fight in range(num_fights):
        deaths = in_fights // num_fights + (1 if fight < in_fights % num_fights else 0)
        t = LANING_END + 20 + fight * segment + _rng.uniform(10, segment - 75)
        first = last = t
        for _ in range(deaths):
            kills.append(_Kill(time=round(t, 1), fight=fight))
            last = t
            t += _rng.uniform(2, 10)
        blocked.append((first - 20, last + 20))

    # Pick-offs keep 20 s from every other death, so they never chain into a fight.
    solo = [float(FIRST_BLOOD_TIME)]
    while len(solo) < 1 + laning_solo + late_solo:
        if len(solo) <= laning_solo:
            t = round(_rng.uniform(150, LANING_END - 10), 1)
        else:
            t = round(_rng.uniform(LANING_END + 10, duration - 20), 1)
        if any(lo <= t <= hi for lo, hi in blocked) or any(abs(t - other) < 20 for other in solo):
            continue
        solo.append(t)
    kills.extend(_Kill(time=t, fight=None) for t in solo)
    kills.sort(key=lambda k: k.time)
    return kills, num_fights


def _assign_kills(kills: list[_Kill], table_kills: list[int], table_deaths: list[int],
                  fight_starts: dict[int, float]) -> dict[int, list[tuple[float, float]]]:
    """Give every planned death a victim and a killer so that the totals equal
    the scoreboard. Returns the death intervals (death, respawn) per player.

    A victim is alive (also when its fight began) and dies once per fight; the
    hero with the largest share of its deaths still to place goes first, so
    every hero's deaths spread over the game and none is left with deaths
    that fit nowhere. The killer is an enemy with kills left, preferably alive
    and not yet fallen in the fight. Raises RuntimeError when a planned death
    fits nobody; the tests cover the plan of the fixed seed.
    """
    deaths_left = list(table_deaths)
    kills_left = list(table_kills)
    dead: dict[int, list[tuple[float, float]]] = {i: [] for i in range(10)}
    died_in_fight: dict[int, set[int]] = {}

    def _alive(idx: int, t: float) -> bool:
        return all(not (start <= t < end) for start, end in dead[idx])

    for kill in kills:
        fallen = died_in_fight.setdefault(kill.fight, set()) if kill.fight is not None else set()
        start = fight_starts[kill.fight] if kill.fight is not None else kill.time

        def _present(idx: int) -> bool:
            return idx not in fallen and _alive(idx, kill.time) and _alive(idx, start)

        if kill.time == FIRST_BLOOD_TIME:
            victim, killer = 2, 8  # Dragon Knight draws first blood on Warlock
        else:
            order = list(range(10))
            _rng.shuffle(order)  # random tie-break between equally loaded heroes
            victims = sorted((v for v in order if deaths_left[v] and _present(v)),
                             key=lambda v: -deaths_left[v] / table_deaths[v])
            victim = killer = -1
            for strict in (True, False):
                for v in victims:
                    enemies = [k for k in order if _team_of(k) != _team_of(v) and kills_left[k]
                               and (_present(k) or not strict)]
                    if enemies:
                        victim, killer = v, max(enemies, key=lambda k: kills_left[k] / table_kills[k])
                        break
                if victim >= 0:
                    break
            if victim < 0:
                raise RuntimeError(f"mock kill plan: no hero can die at {kill.time}")
        kill.victim, kill.killer = victim, killer
        deaths_left[victim] -= 1
        kills_left[killer] -= 1
        dead[victim].append((kill.time, kill.time + _respawn_time(kill.time)))
        fallen.add(victim)
    return dead


def _assign_assists(kills: list[_Kill], table_assists: list[int], dead: dict[int, list[tuple[float, float]]],
                    fight_starts: dict[int, float]) -> None:
    """Spread each player's scoreboard assists over the kills of the own team.

    An assister took part: alive at a pick-off, or alive when the fight began
    (a hero that fell earlier in the fight still gets the assist).
    """
    def _alive(idx: int, t: float) -> bool:
        return all(not (start <= t < end) for start, end in dead[idx])

    for idx, wanted in enumerate(table_assists):
        eligible = [
            k for k in kills
            if _team_of(k.killer) == _team_of(idx) and k.killer != idx and len(k.assists) < 4
            and _alive(idx, k.time if k.fight is None else fight_starts[k.fight])
        ]
        for kill in _rng.sample(eligible, min(wanted, len(eligible))):
            kill.assists.append(idx)


# ---------------------------------------------------------------------------
# Teamfights
# ---------------------------------------------------------------------------

def _interpolate_position(
    positions: list[PositionSnapshot], time_offset: float,
) -> tuple[float, float]:
    """Linear interpolation of position at a given time_offset."""
    if not positions:
        return (0.0, 0.0)
    if time_offset <= positions[0].time_offset:
        return (positions[0].x, positions[0].y)
    if time_offset >= positions[-1].time_offset:
        return (positions[-1].x, positions[-1].y)
    for i in range(len(positions) - 1):
        p0, p1 = positions[i], positions[i + 1]
        if p0.time_offset <= time_offset <= p1.time_offset:
            dt = p1.time_offset - p0.time_offset
            if dt == 0:
                return (p0.x, p0.y)
            ratio = (time_offset - p0.time_offset) / dt
            return (p0.x + (p1.x - p0.x) * ratio, p0.y + (p1.y - p0.y) * ratio)
    return (positions[-1].x, positions[-1].y)


def _generate_teamfights(
    heroes: list[tuple[int, str, str, list[tuple[str, str]]]],
    kills: list[_Kill],
    num_fights: int,
    dead: dict[int, list[tuple[float, float]]],
) -> list[TeamfightEvent]:
    """One TeamfightEvent per planned fight. The window is padded by 15 s
    around the first and last death (like OpenDota and the replay analyzer);
    deaths, kills, kill targets and death positions come from the kill log."""
    teamfights: list[TeamfightEvent] = []

    for fight in range(num_fights):
        fight_kills = [k for k in kills if k.fight == fight]
        start = int(fight_kills[0].time) - 15
        end = int(math.ceil(fight_kills[-1].time)) + 15
        fight_duration = end - start
        cx, cy = _rand_map_pos()

        players = []
        for i, (hero_id, hero_name, slug, abilities) in enumerate(heroes):
            px = _clamp(cx + _rng.uniform(-1200, 1200))
            py = _clamp(cy + _rng.uniform(-1200, 1200))
            own_death = next((k for k in fight_kills if k.victim == i), None)
            # A hero that is still dead when the fight starts takes no part in it.
            absent = any(s <= start < e for s, e in dead[i])

            # Position snapshots across the window (6-10 per player)
            num_snapshots = _rng.randint(6, 10)
            positions: list[PositionSnapshot] = []
            cur_x, cur_y = px, py
            for s_idx in range(num_snapshots):
                t_off = round(
                    (s_idx / max(1, num_snapshots - 1)) * fight_duration, 1
                )
                positions.append(PositionSnapshot(
                    x=round(cur_x, 1), y=round(cur_y, 1),
                    time_offset=t_off,
                ))
                cur_x += _rng.uniform(-400, 400)
                cur_y += _rng.uniform(-400, 400)
                cur_x = _clamp(max(cx - 2000, min(cx + 2000, cur_x)))
                cur_y = _clamp(max(cy - 2000, min(cy + 2000, cur_y)))

            # Dead heroes: the track ends where and when they died
            death_off = None
            if own_death is not None:
                death_off = round(own_death.time - start, 1)
                dx, dy = _interpolate_position(positions, death_off)
                own_death.x, own_death.y = round(dx, 1), round(dy, 1)
                positions = [p for p in positions if p.time_offset < death_off]
                positions.append(PositionSnapshot(x=own_death.x, y=own_death.y, time_offset=death_off))
            if absent:
                positions = []

            # Ability casts (1-4 per player) while alive
            num_casts = _rng.randint(1, 4)
            ability_casts: list[AbilityCast] = []
            for _ in range(num_casts):
                ab_name, ab_key = _rng.choice(abilities)
                latest = death_off if death_off is not None else fight_duration * 0.9
                cast_off = round(_rng.uniform(0, latest), 1)
                hx, hy = _interpolate_position(positions, cast_off)
                cast = AbilityCast(
                    ability_name=ab_name, ability_key=ab_key,
                    time_offset=cast_off,
                    x=round(_clamp(hx + _rng.uniform(-300, 300)), 1),
                    y=round(_clamp(hy + _rng.uniform(-300, 300)), 1),
                )
                if not absent:
                    ability_casts.append(cast)
            ability_casts.sort(key=lambda a: a.time_offset)

            own_kills = [k for k in fight_kills if k.killer == i]
            own_assists = sum(1 for k in fight_kills if i in k.assists)
            damage_dealt, damage_taken = _rng.randint(200, 3000), _rng.randint(100, 2500)
            healing = _rng.randint(0, 500)
            gold_delta = 220 * len(own_kills) + 70 * own_assists + _rng.randint(20, 160)
            xp_delta = 300 * len(own_kills) + 180 * own_assists + _rng.randint(40, 260)
            players.append(TeamfightPlayer(
                player_slot=_slot_of(i), hero_name=hero_name, hero_slug=slug,
                x=round(px, 1), y=round(py, 1),
                damage_dealt=0 if absent else damage_dealt,
                damage_taken=0 if absent else damage_taken,
                healing=0 if absent else healing,
                kills=len(own_kills), deaths=1 if own_death is not None else 0,
                abilities_used=list(dict.fromkeys(a.ability_key for a in ability_casts)),
                positions=positions,
                ability_casts=ability_casts,
                death_time_offset=death_off,
                kill_targets=[
                    {"target": heroes[k.victim][1], "target_slot": _slot_of(k.victim),
                     "time_offset": round(k.time - start, 1)}
                    for k in own_kills
                ],
                gold_delta=0 if absent else gold_delta,
                xp_delta=0 if absent else xp_delta,
            ))

        teamfights.append(TeamfightEvent(
            start_time=start, end_time=end,
            radiant_deaths=sum(1 for k in fight_kills if k.victim < 5),
            dire_deaths=sum(1 for k in fight_kills if k.victim >= 5),
            gold_delta=sum(p.gold_delta if p.player_slot < 128 else -p.gold_delta for p in players),
            xp_delta=sum(p.xp_delta if p.player_slot < 128 else -p.xp_delta for p in players),
            players=players,
        ))

    return teamfights


# ---------------------------------------------------------------------------
# Laning phase
# ---------------------------------------------------------------------------

# Where each lane's creep equilibrium sits during the laning phase (world units)
_LANE_AREAS: dict[int, tuple[float, float]] = {
    1: (5400.0, -5600.0),   # bottom
    2: (-500.0, -400.0),    # mid
    3: (-5700.0, 4900.0),   # top
}


def _generate_laning_snapshots(
    lane: int,
    creep_kills: list[LaningCreepKill],
    denies: list[LaningCreepKill],
    net_worth_at,
    level_at,
) -> list[LaningSnapshot]:
    """121 snapshots (every 5 seconds, 0-600) for the laning phase. Last hits
    and denies count the laning events; net worth and level follow the
    player's gold and XP curves."""
    snapshots = []
    base_hp = 550 + _rng.randint(0, 100)
    base_mana = 250 + _rng.randint(0, 100)

    for t in range(0, LANING_END + 1, 5):
        level = level_at(t)
        max_hp = base_hp + level * 40
        max_mana = base_mana + level * 20
        hp = int(max_hp * _rng.uniform(0.4, 1.0))
        mana = int(max_mana * _rng.uniform(0.2, 1.0))

        # Position: around the assigned lane's creep equilibrium
        lx, ly = _LANE_AREAS.get(lane, _LANE_AREAS[2])
        x = lx + _rng.uniform(-900, 900)
        y = ly + _rng.uniform(-900, 900)

        snapshots.append(LaningSnapshot(
            time=t, x=round(x, 1), y=round(y, 1),
            hp=hp, max_hp=max_hp,
            mana=mana, max_mana=max_mana,
            level=level,
            last_hits=sum(1 for c in creep_kills if c.time <= t),
            denies=sum(1 for d in denies if d.time <= t),
            net_worth=net_worth_at(t),
        ))
    return snapshots


def _generate_laning_creep_kills(is_core: bool, limit: int) -> list[LaningCreepKill]:
    """Generate realistic last-hit events during laning phase (at most ``limit``)."""
    kills = []
    num = _rng.randint(30, 55) if is_core else _rng.randint(5, 15)
    for _ in range(num):
        t = _rng.randint(30, 599)
        creep_type = _rng.choice(["melee", "melee", "melee", "ranged", "siege"])
        if creep_type == "melee":
            gold = _rng.randint(36, 44)
        elif creep_type == "ranged":
            gold = _rng.randint(44, 52)
        else:
            gold = _rng.randint(60, 80)
        kills.append(LaningCreepKill(time=t, creep_type=creep_type, gold=gold))
    return sorted(kills[:limit], key=lambda k: k.time)


def _generate_laning_denies(limit: int) -> list[LaningCreepKill]:
    """Generate a few deny events during laning phase (at most ``limit``)."""
    denies = []
    num = _rng.randint(1, 8)
    for _ in range(num):
        t = _rng.randint(60, 599)
        creep_type = _rng.choice(["melee", "melee", "ranged"])
        denies.append(LaningCreepKill(time=t, creep_type=creep_type, gold=0))
    return sorted(denies[:limit], key=lambda d: d.time)


def _generate_laning_damage_events(
    hero_idx: int, num_players: int, abilities: list[tuple[str, str]],
) -> list[LaningDamageEvent]:
    """Generate hero-vs-hero damage trades during laning phase.

    target_idx is the index of the target in match.players (0-9), matching
    the clarity Java analyzer output — NOT player_slot (dire slots are 128+).
    The inflictor is "attack" for a plain attack, else an ability key.
    """
    events = []
    num = _rng.randint(10, 30)
    # Target indices (into match.players) from opposing team
    if hero_idx < 5:
        possible_targets = list(range(5, num_players))
    else:
        possible_targets = list(range(0, 5))
    if not possible_targets:
        return events
    for _ in range(num):
        t = _rng.randint(30, 599)
        target_idx = _rng.choice(possible_targets)
        damage = _rng.randint(20, 150)
        inflictor = _rng.choice(["attack", "attack", "attack", _rng.choice(abilities[:3])[1]])
        events.append(LaningDamageEvent(
            time=t, target_idx=target_idx,
            damage=damage, inflictor=inflictor,
        ))
    return sorted(events, key=lambda e: e.time)


def _generate_laning_regen_events() -> list[LaningRegenEvent]:
    """Generate tango/salve/clarity usage during laning phase."""
    events = []
    regen_items = [
        ("tango", 3, 6),
        ("flask", 0, 2),
        ("clarity", 0, 2),
        ("enchanted_mango", 0, 2),
        ("faerie_fire", 0, 1),
    ]
    for item, min_uses, max_uses in regen_items:
        num = _rng.randint(min_uses, max_uses)
        for _ in range(num):
            t = _rng.randint(30, 599)
            events.append(LaningRegenEvent(time=t, item=item))
    return sorted(events, key=lambda e: e.time)


# ---------------------------------------------------------------------------
# Hero movement: one track per player that every position in the match shares
# ---------------------------------------------------------------------------

class _Track:
    """Where a hero is over the game: linear movement between keyframes,
    standing still (dead) from each death until the respawn."""

    def __init__(self, deaths: list[tuple[float, float]]) -> None:
        self.deaths = deaths                      # (death time, respawn time)
        self.death_pos: dict[float, tuple[float, float]] = {}
        self._frames: list[tuple[float, float, float]] = []

    def add(self, t: float, x: float, y: float) -> None:
        self._frames.append((t, x, y))

    def finish(self, fountain: tuple[float, float]) -> None:
        """Drop keyframes that fall into a death and respawn at the fountain."""
        frames = [f for f in self._frames
                  if not any(start < f[0] < end for start, end in self.deaths)]
        for start, end in self.deaths:
            x, y = self.death_pos[start]
            frames.append((start, x, y))
            frames.append((end, fountain[0], fountain[1]))
        self._frames = sorted(frames, key=lambda f: f[0])

    def dead_at(self, t: float) -> tuple[float, float] | None:
        """The death position while the hero is dead at ``t``, else None."""
        for start, end in self.deaths:
            if start <= t < end:
                return self.death_pos[start]
        return None

    def at(self, t: float) -> tuple[float, float]:
        dead = self.dead_at(t)
        if dead is not None:
            return dead
        frames = self._frames
        if t <= frames[0][0]:
            return frames[0][1], frames[0][2]
        for (t0, x0, y0), (t1, x1, y1) in zip(frames, frames[1:]):
            if t0 <= t <= t1:
                ratio = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio
        return frames[-1][1], frames[-1][2]


def _generate_ability_events(
    player_slot: int, hero_name: str,
    abilities: list[tuple[str, str]], duration: int, track: _Track,
) -> list[AbilityEvent]:
    """Casts over the game, located where the hero is at that time."""
    events: list[AbilityEvent] = []
    for _ in range(_rng.randint(30, 80)):
        t = _rng.randint(60, duration)
        name, key = _rng.choice(abilities)
        dx, dy = _rng.uniform(-300, 300), _rng.uniform(-300, 300)
        if track.dead_at(t) is not None:
            continue  # dead heroes cast nothing
        x, y = track.at(t)
        events.append(AbilityEvent(
            player_slot=player_slot, hero_name=hero_name,
            ability_name=name, ability_key=key, time=t,
            x=round(_clamp(x + dx), 1), y=round(_clamp(y + dy), 1),
        ))
    return sorted(events, key=lambda e: e.time)


def _position_log(track: _Track, snapshots: list[LaningSnapshot], duration: int) -> list[list[int]]:
    """``[time, x, y, hp_pct]`` every 5 s from the first spawn to game end."""
    laning_hp = {s.time: int(100 * s.hp / s.max_hp) for s in snapshots if s.max_hp}
    log: list[list[int]] = []
    for t in range(POSITION_LOG_START, duration + 1, POSITION_LOG_STEP):
        x, y = track.at(t)
        if track.dead_at(t) is not None:
            hp = 0
        elif t in laning_hp:
            hp = max(1, laning_hp[t])
        elif track.dead_at(t + POSITION_LOG_STEP) is not None:
            hp = _rng.randint(5, 30)   # about to die
        else:
            hp = _rng.randint(45, 100)
        log.append([t, int(round(x)), int(round(y)), hp])
    return log


def _split(total: int, parts: int) -> list[int]:
    """Split ``total`` into ``parts`` random non-negative integers."""
    weights = [_rng.uniform(0.3, 1.0) for _ in range(parts)]
    shares = [int(total * w / sum(weights)) for w in weights]
    shares[0] += total - sum(shares)
    return shares


# Buildings destroyed in the mock game (Dire wins at 31:53):
# (team, key, destroyed_time, killer player_slot)
_MOCK_BUILDING_KILLS: list[tuple[str, str, int, int]] = [
    ("radiant", "mid_t1", 655, 131),
    ("dire", "bot_t1", 712, 1),
    ("radiant", "top_t1", 798, 132),
    ("radiant", "bot_t1", 905, 129),
    ("radiant", "mid_t2", 1210, 131),
    ("radiant", "top_t2", 1388, 129),
    ("dire", "top_t1", 1402, 3),
    ("radiant", "bot_t2", 1525, 131),
    ("radiant", "mid_t3", 1702, 131),
    ("radiant", "mid_melee", 1731, 129),
    ("radiant", "mid_range", 1736, 130),
    ("radiant", "ancient_top", 1858, 131),
    ("radiant", "ancient_bot", 1879, 129),
    ("radiant", "fort", 1913, 131),
]


def _building_name(team: str, key: str) -> str:
    side = "goodguys" if team == "radiant" else "badguys"
    if key == "fort":
        return f"npc_dota_{side}_fort"
    lane, rest = key.split("_", 1)
    if rest in ("melee", "range"):
        return f"npc_dota_{side}_{rest}_rax_{lane}"
    if lane == "ancient":
        return f"npc_dota_{side}_tower4_{rest}"
    return f"npc_dota_{side}_tower{rest[1:]}_{lane}"


def _generate_buildings() -> tuple[list[dict], list[dict]]:
    """Return (buildings, building_kill objectives) for the mock match."""
    destroyed = {(team, key): (t, slot) for team, key, t, slot in _MOCK_BUILDING_KILLS}
    buildings: list[dict] = []
    for team in ("radiant", "dire"):
        entries = (
            [("tower", key, pos) for key, pos in geo.TOWER_POSITIONS[team].items()]
            + [("barracks", key, pos) for key, pos in geo.BARRACKS_POSITIONS[team].items()]
            + [("fort", "fort", geo.ANCIENT_POSITIONS[team])]
        )
        for btype, key, (x, y) in entries:
            kill = destroyed.get((team, key))
            buildings.append({
                "name": _building_name(team, key),
                "team": team,
                "type": btype,
                "x": x,
                "y": y,
                "destroyed_time": kill[0] if kill else None,
            })
    objectives = []
    for team, key, t, slot in _MOCK_BUILDING_KILLS:
        objectives.append({
            "time": t,
            "type": "building_kill",
            "key": _building_name(team, key),
            "player_slot": slot,
            "team": "dire" if team == "radiant" else "radiant",
            "building_team": team,
        })
    return buildings, objectives


# (lane, lane_role) per role and team; lane is absolute (1=bottom, 2=mid, 3=top)
_MOCK_LANES: dict[str, dict[str, tuple[int, int]]] = {
    "radiant": {"carry": (1, 1), "mid": (2, 2), "offlane": (3, 3)},
    "dire": {"carry": (3, 1), "mid": (2, 2), "offlane": (1, 3)},
}
# Supports by slot: which lane they laned in (soft support with the offlaner,
# hard support with the carry).
_MOCK_SUPPORT_LANES: dict[int, tuple[int, int]] = {
    0: (3, 3), 2: (1, 1), 128: (3, 1), 130: (1, 3),
}


def generate_mock_match() -> MatchData:
    """Generate a deterministic mock match based on real match 8713267082."""
    with _MOCK_LOCK:
        _rng.seed(_MOCK_SEED)
        return _build_mock_match()


def _build_mock_match() -> MatchData:
    # lazy: parser imports this module lazily too
    from .parser import (
        _gold_xp_timeline, building_status_bitmasks, kill_announcements, kill_participation,
        lane_efficiency, talent_names_patch,
    )

    match_id = 8713267082
    duration = 1913  # 31:53
    radiant_win = False

    # Scoreboard of real match 8713267082; the player names are invented.
    # (slot, hero_id, hero_name, slug, player_name, role, K, D, A, lvl, lh, dn, gpm, xpm, hdmg, tdmg, heal, nw, final_items, final_keys, abilities)
    real_players = [
        # Radiant
        (0, 123, "Hoodwink", "hoodwink", PLAYER_NAMES[0], "support",
         7, 5, 7, 15, 49, 4, 294, 433, 13784, 0, 0, 8357,
         ["Power Treads", "Arcane Boots", "Rod of Atos", "Perseverance"],
         ["power_treads", "arcane_boots", "rod_of_atos", "pers"],
         [("Acorn Shot", "hoodwink_acorn_shot"), ("Bushwhack", "hoodwink_bushwhack"),
          ("Scurry", "hoodwink_scurry"), ("Sharpshooter", "hoodwink_sharpshooter")]),
        (1, 6, "Drow Ranger", "drow_ranger", PLAYER_NAMES[1], "carry",
         4, 9, 9, 15, 190, 7, 422, 426, 18527, 1055, 0, 11058,
         ["Power Treads", "Quelling Blade", "Magic Wand", "Falcon Blade", "Aghanim's Scepter", "Dragon Lance"],
         ["power_treads", "quelling_blade", "magic_wand", "falcon_blade", "ultimate_scepter", "dragon_lance"],
         [("Frost Arrows", "drow_ranger_frost_arrows"), ("Gust", "drow_ranger_wave_of_silence"),
          ("Multishot", "drow_ranger_multishot"), ("Marksmanship", "drow_ranger_marksmanship")]),
        (2, 37, "Warlock", "warlock", PLAYER_NAMES[2], "support",
         2, 12, 11, 13, 58, 2, 269, 329, 19765, 119, 529, 6151,
         ["Arcane Boots", "Magic Wand", "Force Staff"],
         ["arcane_boots", "magic_wand", "force_staff"],
         [("Fatal Bonds", "warlock_fatal_bonds"), ("Shadow Word", "warlock_shadow_word"),
          ("Upheaval", "warlock_upheaval"), ("Chaotic Offering", "warlock_rain_of_chaos")]),
        (3, 15, "Razor", "razor", PLAYER_NAMES[3], "offlane",
         2, 12, 12, 16, 116, 9, 350, 464, 16909, 0, 0, 10364,
         ["Aghanim's Scepter", "Power Treads", "Infused Raindrops", "Pipe of Insight", "Wraith Band"],
         ["ultimate_scepter", "power_treads", "infused_raindrop", "pipe", "wraith_band"],
         [("Plasma Field", "razor_plasma_field"), ("Static Link", "razor_static_link"),
          ("Storm Surge", "razor_storm_surge"), ("Eye of the Storm", "razor_eye_of_the_storm")]),
        (4, 2, "Axe", "axe", PLAYER_NAMES[4], "mid",
         4, 13, 5, 16, 116, 9, 312, 447, 15307, 513, 0, 7654,
         ["Phase Boots", "Blade Mail", "Blink Dagger", "Urn of Shadows", "Quelling Blade"],
         ["phase_boots", "blade_mail", "blink", "urn_of_shadows", "quelling_blade"],
         [("Berserker's Call", "axe_berserkers_call"), ("Battle Hunger", "axe_battle_hunger"),
          ("Counter Helix", "axe_counter_helix"), ("Culling Blade", "axe_culling_blade")]),
        # Dire
        (128, 27, "Shadow Shaman", "shadow_shaman", PLAYER_NAMES[5], "support",
         3, 7, 20, 17, 44, 0, 329, 494, 12242, 771, 0, 8122,
         ["Blink Dagger", "Aether Lens", "Observer and Sentry Wards", "Null Talisman", "Tranquil Boots"],
         ["blink", "aether_lens", "ward_dispenser", "null_talisman", "tranquil_boots"],
         [("Ether Shock", "shadow_shaman_ether_shock"), ("Hex", "shadow_shaman_voodoo"),
          ("Shackles", "shadow_shaman_shackles"), ("Mass Serpent Ward", "shadow_shaman_mass_serpent_ward")]),
        (129, 47, "Viper", "viper", PLAYER_NAMES[6], "carry",
         18, 3, 21, 19, 137, 11, 608, 635, 24619, 9117, 0, 18090,
         ["Hurricane Pike", "Manta Style", "Butterfly", "Power Treads", "Wraith Band", "Boots of Speed"],
         ["hurricane_pike", "manta", "butterfly", "power_treads", "wraith_band", "boots"],
         [("Poison Attack", "viper_poison_attack"), ("Nethertoxin", "viper_nethertoxin"),
          ("Corrosive Skin", "viper_corrosive_skin"), ("Viper Strike", "viper_viper_strike")]),
        (130, 26, "Lion", "lion", PLAYER_NAMES[7], "support",
         10, 2, 23, 21, 22, 4, 380, 776, 12195, 1077, 525, 11278,
         ["Blink Dagger", "Glimmer Cape", "Tranquil Boots", "Aether Lens", "Blood Grenade"],
         ["blink", "glimmer_cape", "tranquil_boots", "aether_lens", "blood_grenade"],
         [("Earth Spike", "lion_impale"), ("Hex", "lion_voodoo"),
          ("Mana Drain", "lion_mana_drain"), ("Finger of Death", "lion_finger_of_death")]),
        (131, 49, "Dragon Knight", "dragon_knight", PLAYER_NAMES[8], "mid",
         20, 1, 9, 24, 320, 6, 871, 1066, 36926, 15592, 0, 26059,
         ["Black King Bar", "Mjollnir", "Power Treads", "Aghanim's Scepter", "Satanic", "Blink Dagger"],
         ["black_king_bar", "mjollnir", "power_treads", "ultimate_scepter", "satanic", "blink"],
         [("Breathe Fire", "dragon_knight_breathe_fire"), ("Dragon Tail", "dragon_knight_dragon_tail"),
          ("Dragon Blood", "dragon_knight_dragon_blood"), ("Elder Dragon Form", "dragon_knight_elder_dragon_form")]),
        (132, 99, "Bristleback", "bristleback", PLAYER_NAMES[9], "offlane",
         0, 6, 17, 17, 154, 6, 426, 528, 8532, 5088, 0, 13390,
         ["Black King Bar", "Power Treads", "Lotus Orb", "Magic Wand", "Bracer", "Bracer"],
         ["black_king_bar", "power_treads", "lotus_orb", "magic_wand", "bracer", "bracer"],
         [("Viscous Nasal Goo", "bristleback_viscous_nasal_goo"), ("Quill Spray", "bristleback_quill_spray"),
          ("Bristleback", "bristleback_bristleback"), ("Warpath", "bristleback_warpath")]),
    ]

    num_players = len(real_players)
    hero_names = [p[2] for p in real_players]
    # Format: (hero_id, hero_name, slug, abilities) - abilities is last element
    selected_heroes = [(p[1], p[2], p[3], p[-1]) for p in real_players]

    # Every hero death first: scoreboard, teamfights, positions and the
    # laning snapshots all follow from this one kill log.
    kills, num_fights = _plan_kill_times(sum(p[7] for p in real_players), duration)
    fight_starts = {f: next(k.time for k in kills if k.fight == f) - 15 for f in range(num_fights)}
    dead = _assign_kills(kills, [p[6] for p in real_players], [p[7] for p in real_players], fight_starts)
    _assign_assists(kills, [p[8] for p in real_players], dead, fight_starts)
    teamfights = _generate_teamfights(selected_heroes, kills, num_fights, dead)
    team_kills = {"radiant": sum(p[6] for p in real_players[:5]), "dire": sum(p[6] for p in real_players[5:])}

    lanes: list[tuple[int, int]] = []
    for slot, _id, _name, _slug, _pname, role, *_rest in real_players:
        team = "radiant" if slot < 128 else "dire"
        lanes.append(_MOCK_SUPPORT_LANES.get(slot, (0, 4)) if role == "support" else _MOCK_LANES[team][role])

    # Per-player curves and laning data (needed before the pick-off positions)
    curves = []
    laning = []
    for hero_idx, (slot, _id, _name, _slug, _pname, role,
                   _k, _d, _a, level, last_hits, denies_, gpm, xpm, *_rest) in enumerate(real_players):
        is_core = role in ("carry", "mid", "offlane")
        total_gold = gpm * duration / 60
        total_xp = xpm * duration / 60
        gold_t = _per_minute_series(total_gold, duration)
        xp_t = _per_minute_series(total_xp, duration)
        level_times = _level_times(xp_t, total_xp, level, duration, XP_FOR_LEVEL)
        creep_kills = _generate_laning_creep_kills(is_core, limit=last_hits)
        laning_denies = _generate_laning_denies(limit=denies_)
        snapshots = _generate_laning_snapshots(
            lanes[hero_idx][0], creep_kills, laning_denies,
            net_worth_at=lambda t, g=gold_t, total=total_gold: 600 + int(_series_at(g, total, t, duration)),
            level_at=lambda t, times=level_times: max(lvl for lvl, at in times.items() if at <= t),
        )
        curves.append((gold_t, xp_t, level_times))
        laning.append((snapshots, creep_kills, laning_denies))

    # Positions: pick-offs happen where the victim is; killer and assisters come to it.
    tracks = [_Track(dead[i]) for i in range(num_players)]
    for kill in kills:
        if kill.fight is None:
            if kill.time <= LANING_END:
                snap = laning[kill.victim][0][int(kill.time // 5)]
                kill.x, kill.y = snap.x, snap.y
            else:
                x, y = _rand_map_pos()
                kill.x, kill.y = round(x, 1), round(y, 1)
        tracks[kill.victim].death_pos[kill.time] = (kill.x, kill.y)

    for hero_idx, (slot, *_rest) in enumerate(real_players):
        team = "radiant" if slot < 128 else "dire"
        track = tracks[hero_idx]
        snapshots = laning[hero_idx][0]
        fountain = geo.FOUNTAIN_POSITIONS[team]

        # Laning: gank rotations and deaths show up in the 5 s snapshots.
        for kill in kills:
            if kill.fight is None and kill.time <= LANING_END and hero_idx in [kill.killer, *kill.assists]:
                for snap in snapshots:
                    if kill.time - 10 <= snap.time <= kill.time + 5:
                        snap.x = round(_clamp(kill.x + _rng.uniform(-250, 250)), 1)
                        snap.y = round(_clamp(kill.y + _rng.uniform(-250, 250)), 1)
        for snap in snapshots:
            death = track.dead_at(snap.time)
            if death is not None:
                snap.x, snap.y, snap.hp = death[0], death[1], 0

        track.add(POSITION_LOG_START, *fountain)
        track.add(-30, snapshots[0].x, snapshots[0].y)
        for snap in snapshots:
            track.add(snap.time, snap.x, snap.y)
        # After laning: roam around the own lane, join every fight and pick-off.
        lane_x, lane_y = _LANE_AREAS.get(lanes[hero_idx][0], _LANE_AREAS[2])
        busy = [(tf.start_time - 25, tf.end_time + 10) for tf in teamfights]
        busy += [(k.time - 20, k.time + 20) for k in kills
                 if k.fight is None and hero_idx in [k.victim, k.killer, *k.assists]]
        for t in range(LANING_END + 40, duration, 45):
            wx = _clamp(lane_x * 0.5 + _rng.uniform(-3000, 3000))
            wy = _clamp(lane_y * 0.5 + _rng.uniform(-3000, 3000))
            if not any(lo <= t <= hi for lo, hi in busy):
                track.add(t, wx, wy)
        for tf in teamfights:
            for pos in tf.players[hero_idx].positions:
                track.add(tf.start_time + pos.time_offset, pos.x, pos.y)
        for kill in kills:
            if kill.fight is None and kill.time > LANING_END and hero_idx in [kill.killer, *kill.assists]:
                track.add(kill.time, _clamp(kill.x + _rng.uniform(-300, 300)), _clamp(kill.y + _rng.uniform(-300, 300)))
        track.finish(fountain)

    kills_log = [
        KillEvent(
            time=k.time,
            victim_slot=_slot_of(k.victim),
            killer_slot=_slot_of(k.killer),
            killer_team="radiant" if _slot_of(k.killer) < 128 else "dire",
            x=k.x, y=k.y,
            assist_slots=sorted(_slot_of(a) for a in k.assists),
        )
        for k in kills
    ]

    all_wards: list[WardEvent] = []
    all_ability_events: list[AbilityEvent] = []
    players: list[PlayerData] = []

    for hero_idx, (slot, hero_id, hero_name, slug, player_name, role,
         kills_, deaths, assists, level, last_hits, denies_,
         gpm, xpm, hero_damage, tower_damage, hero_healing, net_worth,
         final_items, final_keys, abilities) in enumerate(real_players):

        team = "radiant" if slot < 128 else "dire"
        lane, lane_role = lanes[hero_idx]
        gold_t, xp_t, level_times = curves[hero_idx]
        laning_snapshots, laning_creep_kills, laning_denies = laning[hero_idx]
        items, _, _ = _generate_item_build(role, duration)
        skills = _generate_skill_build(abilities, level_times)
        wards = _real_wards_for_player(slot, duration)
        ability_events = _generate_ability_events(slot, hero_name, abilities, duration, tracks[hero_idx])
        laning_damage_events = _generate_laning_damage_events(hero_idx, num_players, abilities)
        laning_regen_events = _generate_laning_regen_events()

        all_wards.extend(wards)
        all_ability_events.extend(ability_events)

        # Damage dealt to each enemy hero and by each source, adding up to hero_damage
        enemies = [i for i in range(num_players) if _team_of(i) != team]
        damage = dict(zip((hero_names[i] for i in enemies), _split(hero_damage, len(enemies))))
        inflictor_keys = ["attack"] + [key for _name, key in abilities]
        damage_inflictor = dict(zip(inflictor_keys, _split(hero_damage, len(inflictor_keys))))
        lane_eff, lane_eff_pct = lane_efficiency(gold_t)
        multi_kills, kill_streaks = kill_announcements(kills_log, slot)

        players.append(PlayerData(
            player_slot=slot,
            hero_id=hero_id,
            hero_name=hero_name,
            hero_slug=slug,
            player_name=player_name,
            team=team,
            kills=kills_, deaths=deaths, assists=assists,
            level=level,
            last_hits=last_hits,
            denies=denies_,
            gold_per_min=gpm,
            xp_per_min=xpm,
            hero_damage=hero_damage,
            tower_damage=tower_damage,
            hero_healing=hero_healing,
            net_worth=net_worth,
            items=items,
            backpack=[],
            final_items=final_items,
            final_item_keys=final_keys,
            skills=skills,
            wards=wards,
            ability_events=ability_events,
            gold_t=gold_t,
            xp_t=xp_t,
            lh_t=_counted_series([c.time for c in laning_creep_kills], last_hits, duration),
            dn_t=_counted_series([d.time for d in laning_denies], denies_, duration),
            lane=lane,
            lane_role=lane_role,
            is_roaming=False,
            lane_efficiency=lane_eff,
            lane_efficiency_pct=lane_eff_pct,
            killed=dict(Counter(hero_names[k.victim] for k in kills if k.killer == hero_idx)),
            killed_by=dict(Counter(hero_names[k.killer] for k in kills if k.victim == hero_idx)),
            damage=damage,
            damage_inflictor=damage_inflictor,
            teamfight_participation=kill_participation(kills_, assists, team_kills[team]),
            obs_placed=sum(1 for w in wards if w.ward_type == "observer"),
            sen_placed=sum(1 for w in wards if w.ward_type == "sentry"),
            multi_kills=multi_kills,
            kill_streaks=kill_streaks,
            purchase_log=[{"time": it.time, "key": it.item_key} for it in items],
            position_log=_position_log(tracks[hero_idx], laning_snapshots, duration),
            laning_snapshots=laning_snapshots,
            laning_creep_kills=laning_creep_kills,
            laning_denies=laning_denies,
            laning_damage_events=laning_damage_events,
            laning_regen_events=laning_regen_events,
        ))
    for player in players:
        player.damage_taken = {other.hero_name: other.damage[player.hero_name]
                               for other in players if player.hero_name in other.damage}

    all_wards.sort(key=lambda w: w.time)
    all_ability_events.sort(key=lambda e: e.time)
    # The ward table does not say who dewarded: the enemy supports take turns.
    for ward_team, enemy_supports in (("radiant", (128, 130)), ("dire", (0, 2))):
        for n, ward in enumerate(w for w in all_wards if w.team == ward_team and w.dewarded):
            ward.killer_slot = enemy_supports[n % len(enemy_supports)]

    radiant_smokes = _generate_smokes("radiant", duration)
    dire_smokes = _generate_smokes("dire", duration)
    smokes = sorted(radiant_smokes + dire_smokes, key=lambda s: s.time)

    gold_xp_timeline = _gold_xp_timeline(players)

    buildings, building_objectives = _generate_buildings()
    first_blood = kills_log[0]
    objectives = sorted(
        building_objectives + [
            {"time": FIRST_BLOOD_TIME, "type": "CHAT_MESSAGE_FIRSTBLOOD",
             "player_slot": first_blood.killer_slot, "key": str(first_blood.victim_slot)},
            {"time": 1440, "type": "CHAT_MESSAGE_ROSHAN_KILL", "team": "dire"},
            {"time": 1442, "type": "CHAT_MESSAGE_AEGIS", "player_slot": 131},
        ],
        key=lambda o: o["time"],
    )

    return MatchData(
        match_id=match_id,
        duration=duration,
        radiant_win=radiant_win,
        game_mode="ranked_all_pick",
        start_time=1772470634,  # 2026-03-02 16:57 UTC
        radiant_score=19,
        dire_score=51,
        players=players,
        wards=all_wards,
        smokes=smokes,
        teamfights=teamfights,
        ability_events=all_ability_events,
        gold_xp_timeline=gold_xp_timeline,
        kills_log=kills_log,
        radiant_gold_adv=[s.radiant_gold - s.dire_gold for s in gold_xp_timeline],
        radiant_xp_adv=[s.radiant_xp - s.dire_xp for s in gold_xp_timeline],
        # The draft order is not modelled: picks in slot order, no bans.
        picks_bans=[
            {"is_pick": True, "hero_id": p.hero_id, "team": 0 if p.team == "radiant" else 1,
             "order": order, "player_slot": p.player_slot}
            for order, p in enumerate(players)
        ],
        objectives=objectives,
        buildings=buildings,
        first_blood_time=FIRST_BLOOD_TIME,
        cluster=152,
        region=5,
        patch="7.40",
        talent_names_patch=talent_names_patch(),
        **building_status_bitmasks(buildings, objectives),
    )
