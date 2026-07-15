"""Mock data generator for development and testing."""
import random
import time

from .models import (
    AbilityCast,
    AbilityEvent,
    GoldXpSnapshot,
    ItemEvent,
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

PLAYER_NAMES = [
    "AllMight", "ShadowBlade", "IceStorm", "DragonFly", "NightStalker",
    "SunRay", "ThunderBolt", "StarLight", "MoonWalker", "FireStorm",
]

# Playable area bounds (matching OpenDota cell range 64-190)
# cell * 62.5 - 8000: cell 64 → -4000, cell 190 → 3875
MAP_MIN = -4000
MAP_MAX = 3875


def _rand_map_pos() -> tuple[float, float]:
    return (
        random.uniform(MAP_MIN, MAP_MAX),
        random.uniform(MAP_MIN, MAP_MAX),
    )


def _generate_item_build(
    role: str, duration: int,
) -> tuple[list[ItemEvent], list[str], list[str]]:
    """Returns (item_events, final_display_names, final_item_keys)."""
    items: list[ItemEvent] = []
    item_keys: list[str] = []
    # Starting items purchased during strategy phase (-90s to 0s)
    t = -random.randint(85, 89)

    for _ in range(random.randint(3, 5)):
        name, key = random.choice(EARLY_ITEMS)
        items.append(ItemEvent(item_name=name, time=t, item_key=key))
        item_keys.append(key)
        t += random.randint(0, 3)

    # Early-mid game items (after horn at 0:00)
    t = random.randint(60, 180)
    num_mid = random.randint(3, 6)
    for _ in range(num_mid):
        t += random.randint(60, 300)
        if t > duration:
            break
        pool = MID_ITEMS if role != "support" else MID_ITEMS + SUPPORT_ITEMS
        name, key = random.choice(pool)
        items.append(ItemEvent(item_name=name, time=t, item_key=key))
        item_keys.append(key)

    # Late game items
    num_late = random.randint(2, 4) if role != "support" else random.randint(0, 2)
    for _ in range(num_late):
        t += random.randint(180, 480)
        if t > duration:
            break
        name, key = random.choice(LATE_ITEMS)
        items.append(ItemEvent(item_name=name, time=t, item_key=key))
        item_keys.append(key)

    n = min(6, len(items))
    final_names = [items[-i].item_name for i in range(1, n + 1)]
    final_keys = [items[-i].item_key for i in range(1, n + 1)]
    return items, final_names, final_keys


def _generate_skill_build(abilities: list[tuple[str, str]], duration: int) -> list[SkillEvent]:
    """abilities: list of (display_name, ability_key) tuples."""
    skills: list[SkillEvent] = []
    max_level = min(30, max(1, duration // 80 + 1))
    # Track levels by display name
    skill_levels = {name: 0 for name, _ in abilities}
    ult_name, ult_key = abilities[3]

    for lvl in range(1, max_level + 1):
        t = lvl * random.randint(60, 120)
        if t > duration:
            break

        if lvl in (10, 15, 20, 25):
            side = random.choice(["left", "right"])
            skills.append(SkillEvent(
                ability_name=f"talent_{side}_lvl{lvl}",
                ability_key=f"talent_{side}_lvl{lvl}",
                level=lvl, time=t,
            ))
            continue

        if lvl in (6, 12, 18) and skill_levels[ult_name] < 3:
            skill_levels[ult_name] += 1
            skills.append(SkillEvent(
                ability_name=ult_name, ability_key=ult_key,
                level=skill_levels[ult_name], time=t,
            ))
            continue

        regular = [(n, k) for n, k in abilities[:3] if skill_levels[n] < 4]
        if not regular:
            regular = [(n, k) for n, k in abilities if skill_levels[n] < 4]
        if not regular:
            continue

        chosen_name, chosen_key = random.choice(regular)
        skill_levels[chosen_name] += 1
        skills.append(SkillEvent(
            ability_name=chosen_name, ability_key=chosen_key,
            level=skill_levels[chosen_name], time=t,
        ))

    return skills


def _generate_wards(player_slot: int, team: str, role: str, duration: int) -> list[WardEvent]:
    wards: list[WardEvent] = []
    num_wards = random.randint(0, 3) if role not in ("support", "roamer") else random.randint(8, 20)

    for i in range(num_wards):
        # First 1-2 wards for supports are placed during pre-game (-50s to -10s)
        if i < 2 and role in ("support", "roamer"):
            t = -random.randint(10, 50)
        else:
            t = random.randint(60, duration - 30)
        x, y = _rand_map_pos()
        # Bias ward placement toward team's side of the map
        if team == "radiant":
            y = max(MAP_MIN, min(MAP_MAX, y * 0.6 + random.uniform(-1500, 2500)))
        else:
            y = max(MAP_MIN, min(MAP_MAX, y * 0.6 + random.uniform(-2500, 1500)))
        x = max(MAP_MIN, min(MAP_MAX, x))

        ward_type = random.choice(["observer", "observer", "sentry"])
        destroyed = None
        if random.random() < 0.3:
            destroyed = t + random.randint(30, 360)
            if destroyed > duration:
                destroyed = None

        wards.append(WardEvent(
            ward_type=ward_type,
            x=round(x, 1), y=round(y, 1),
            time=t, player_slot=player_slot,
            destroyed_time=destroyed,
        ))

    return sorted(wards, key=lambda w: w.time)


def _cell_to_game(cx: float, cy: float) -> tuple[float, float]:
    """Convert OpenDota cell coordinates (0-255) to game coordinates (-8000..8000).

    OpenDota cell axes are swapped relative to game axes:
    - cell x (row) corresponds to game Y
    - cell y (column) corresponds to game X
    Both cell axes increase towards negative game coords (Radiant corner).
    """
    return round(cx * 62.5 - 8000, 1), round(cy * 62.5 - 8000, 1)


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


def _real_wards_for_player(player_slot: int) -> list[WardEvent]:
    """Return real ward data for a player in match 8713267082."""
    raw = _REAL_WARD_RAW.get(player_slot, [])
    wards = []
    for ward_type, t, cx, cy, dt in raw:
        gx, gy = _cell_to_game(cx, cy)
        wards.append(WardEvent(
            ward_type=ward_type, x=gx, y=gy,
            time=t, player_slot=player_slot,
            destroyed_time=dt,
        ))
    return sorted(wards, key=lambda w: w.time)


SMOKE_TARGETS_RADIANT = [
    (2000, 2000), (0, 0), (-200, 1500), (3500, 2500), (2500, 800),
]
SMOKE_TARGETS_DIRE = [
    (-2000, -2000), (0, 0), (-200, 1500), (-3500, -2500), (-2500, -800),
]


def _generate_smoke_path(
    team: str, num_waypoints: int,
) -> tuple[float, float, list[list[float]]]:
    """Generate smoke start + movement path toward a target."""
    if team == "radiant":
        sx = random.uniform(-3800, -1500)
        sy = random.uniform(-3800, -1500)
        target = random.choice(SMOKE_TARGETS_RADIANT)
    else:
        sx = random.uniform(1500, 3800)
        sy = random.uniform(1500, 3800)
        target = random.choice(SMOKE_TARGETS_DIRE)

    tx, ty = target
    path: list[list[float]] = []
    for i in range(num_waypoints):
        t_ratio = (i + 1) / (num_waypoints + 1)
        wx = sx + (tx - sx) * t_ratio + random.uniform(-500, 500)
        wy = sy + (ty - sy) * t_ratio + random.uniform(-500, 500)
        wx = max(MAP_MIN, min(MAP_MAX, wx))
        wy = max(MAP_MIN, min(MAP_MAX, wy))
        path.append([round(wx, 1), round(wy, 1)])

    return round(sx, 1), round(sy, 1), path


def _generate_smokes(team: str, duration: int) -> list[SmokeEvent]:
    smokes: list[SmokeEvent] = []
    num_smokes = random.randint(4, 10)
    slots = list(range(0, 5)) if team == "radiant" else list(range(128, 133))

    for _ in range(num_smokes):
        t = random.randint(300, duration - 60)
        participants = random.sample(slots, random.randint(2, 5))
        num_waypoints = random.randint(3, 8)
        sx, sy, path = _generate_smoke_path(team, num_waypoints)
        smoke_duration = random.randint(10, 35)

        # Random result for the smoke gank
        result_roll = random.random()
        if result_roll < 0.35:
            result = "kill"
            kills_gained = random.randint(1, 3)
            deaths_suffered = random.randint(0, 1)
        elif result_roll < 0.55:
            result = "traded"
            kills_gained = random.randint(1, 2)
            deaths_suffered = random.randint(1, 2)
        elif result_roll < 0.75:
            result = "objective"
            kills_gained = random.randint(0, 1)
            deaths_suffered = 0
        else:
            result = "no_contact"
            kills_gained = 0
            deaths_suffered = random.randint(0, 1)

        objective = None
        if result == "objective":
            objective = random.choice(["tower", "roshan"])

        smokes.append(SmokeEvent(
            time=t, team=team, player_slots=participants,
            x=sx, y=sy, end_time=t + smoke_duration, path=path,
            result=result, kills_gained=kills_gained,
            deaths_suffered=deaths_suffered, objective_taken=objective,
        ))

    return sorted(smokes, key=lambda s: s.time)


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
    duration: int,
) -> list[TeamfightEvent]:
    teamfights: list[TeamfightEvent] = []

    for _ in range(random.randint(5, 12)):
        start = random.randint(600, duration - 120)
        fight_duration = random.randint(10, 45)
        end = start + fight_duration
        cx, cy = _rand_map_pos()

        players = []
        r_deaths = d_deaths = 0

        for i, (hero_id, hero_name, slug, abilities) in enumerate(heroes):
            slot = i if i < 5 else 128 + (i - 5)
            px = max(MAP_MIN, min(MAP_MAX, cx + random.uniform(-1500, 1500)))
            py = max(MAP_MIN, min(MAP_MAX, cy + random.uniform(-1500, 1500)))
            died = random.random() < 0.3
            kills = random.randint(0, 2) if not died else random.randint(0, 1)

            if died:
                if i < 5:
                    r_deaths += 1
                else:
                    d_deaths += 1

            # Generate position snapshots (4-8 per player)
            num_snapshots = random.randint(4, 8)
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
                cur_x += random.uniform(-300, 300)
                cur_y += random.uniform(-300, 300)
                cur_x = max(MAP_MIN, min(MAP_MAX, max(cx - 2000, min(cx + 2000, cur_x))))
                cur_y = max(MAP_MIN, min(MAP_MAX, max(cy - 2000, min(cy + 2000, cur_y))))

            # Dead heroes: trim positions to before death
            if died:
                death_off = random.uniform(fight_duration * 0.3, fight_duration * 0.9)
                positions = [p for p in positions if p.time_offset <= death_off]
                if not positions:
                    positions = [PositionSnapshot(x=round(px, 1), y=round(py, 1), time_offset=0)]

            # Generate ability casts (1-4 per player)
            num_casts = random.randint(1, 4)
            ability_casts: list[AbilityCast] = []
            for _ in range(num_casts):
                ab_name, ab_key = random.choice(abilities)
                cast_off = round(random.uniform(0, fight_duration * 0.9), 1)
                hx, hy = _interpolate_position(positions, cast_off)
                ability_casts.append(AbilityCast(
                    ability_name=ab_name, ability_key=ab_key,
                    time_offset=cast_off,
                    x=round(hx + random.uniform(-200, 200), 1),
                    y=round(hy + random.uniform(-200, 200), 1),
                ))
            ability_casts.sort(key=lambda a: a.time_offset)

            used_names = list({a.ability_name for a in ability_casts})

            # Record death time (death_off was set earlier when trimming positions)
            death_time_off = round(death_off, 1) if died else None

            players.append(TeamfightPlayer(
                player_slot=slot, hero_name=hero_name, hero_slug=slug,
                x=round(px, 1), y=round(py, 1),
                damage_dealt=random.randint(200, 3000),
                damage_taken=random.randint(100, 2500),
                healing=random.randint(0, 500),
                kills=kills, deaths=1 if died else 0,
                abilities_used=used_names,
                positions=positions,
                ability_casts=ability_casts,
                death_time_offset=death_time_off,
                kill_targets=[],  # filled below
            ))

        # Assign kill_targets: for each dead hero, pick a random killer from the other team
        dead_heroes = [(p, p.death_time_offset) for p in players if p.deaths > 0 and p.death_time_offset is not None]
        for dead_p, dt in dead_heroes:
            dead_team = "radiant" if dead_p.player_slot < 128 else "dire"
            killers = [p for p in players if p.kills > 0 and
                       ("radiant" if p.player_slot < 128 else "dire") != dead_team]
            if killers:
                killer = random.choice(killers)
                killer.kill_targets.append({
                    "target": dead_p.hero_name,
                    "target_slug": dead_p.hero_slug,
                    "time_offset": dt,
                })

        teamfights.append(TeamfightEvent(
            start_time=start, end_time=end,
            radiant_deaths=r_deaths, dire_deaths=d_deaths,
            players=players,
        ))

    return sorted(teamfights, key=lambda tf: tf.start_time)


def _generate_ability_events(
    player_slot: int, hero_name: str,
    abilities: list[tuple[str, str]], duration: int,
) -> list[AbilityEvent]:
    events: list[AbilityEvent] = []
    for _ in range(random.randint(30, 80)):
        t = random.randint(60, duration)
        name, key = random.choice(abilities)
        x, y = _rand_map_pos()
        events.append(AbilityEvent(
            player_slot=player_slot, hero_name=hero_name,
            ability_name=name, ability_key=key, time=t,
            x=round(x, 1), y=round(y, 1),
        ))
    return sorted(events, key=lambda e: e.time)


def _generate_gold_xp_timeline(duration: int, radiant_win: bool) -> list[GoldXpSnapshot]:
    """Generate per-minute gold/xp timeline for both teams."""
    snapshots = []
    r_gold, d_gold = 0, 0
    r_xp, d_xp = 0, 0
    # Base gold/xp growth rates (per minute)
    r_gold_rate = random.randint(1800, 2400)
    d_gold_rate = random.randint(1800, 2400)
    r_xp_rate = random.randint(2200, 2800)
    d_xp_rate = random.randint(2200, 2800)
    # Winner gradually gains advantage
    advantage_factor = 1.0
    for minute in range(0, duration // 60 + 1):
        t = minute * 60
        # Laning phase (0-10 min): fairly even
        if minute <= 10:
            jitter_gold = random.randint(-300, 300)
            jitter_xp = random.randint(-200, 200)
        # Mid game (10-25 min): leads develop
        elif minute <= 25:
            advantage_factor += 0.02 if radiant_win else -0.02
            jitter_gold = random.randint(-500, 500)
            jitter_xp = random.randint(-400, 400)
        # Late game (25+ min): swings possible
        else:
            advantage_factor += 0.015 if radiant_win else -0.015
            jitter_gold = random.randint(-800, 800)
            jitter_xp = random.randint(-500, 500)

        r_gold += int(r_gold_rate * max(0.7, min(1.3, advantage_factor)) + jitter_gold)
        d_gold += int(d_gold_rate * max(0.7, min(1.3, 2 - advantage_factor)) + jitter_gold * -0.5)
        r_xp += int(r_xp_rate * max(0.7, min(1.3, advantage_factor)) + jitter_xp)
        d_xp += int(d_xp_rate * max(0.7, min(1.3, 2 - advantage_factor)) + jitter_xp * -0.5)

        snapshots.append(GoldXpSnapshot(
            time=t,
            radiant_gold=max(0, r_gold),
            dire_gold=max(0, d_gold),
            radiant_xp=max(0, r_xp),
            dire_xp=max(0, d_xp),
        ))
    return snapshots


def _generate_laning_snapshots(hero_idx: int, is_core: bool = True) -> list[LaningSnapshot]:
    """Generate 120 snapshots (every 5 seconds, 0-595) for laning phase."""
    snapshots = []
    base_hp = 550 + random.randint(0, 100)
    base_mana = 250 + random.randint(0, 100)
    cs = 0
    denies = 0
    gold = 625
    level = 1

    for t in range(0, 600, 5):
        # Simulate CS growth (cores get more)
        if t > 30 and random.random() < (0.12 if is_core else 0.03):
            cs += 1
            gold += random.randint(35, 55)
        if t > 30 and random.random() < 0.02:
            denies += 1

        # Level up roughly
        if t > 120 and level < 2:
            level = 2
        if t > 200 and level < 3:
            level = 3
        if t > 280 and level < 4:
            level = 4
        if t > 360 and level < 5:
            level = 5
        if t > 480 and level < 6:
            level = 6

        max_hp = base_hp + level * 40
        max_mana = base_mana + level * 20
        hp = int(max_hp * random.uniform(0.4, 1.0))
        mana = int(max_mana * random.uniform(0.2, 1.0))

        # Position (lane area roughly)
        x = random.uniform(-1500, 1500)
        y = random.uniform(-1500, 1500)

        snapshots.append(LaningSnapshot(
            time=t, x=round(x, 1), y=round(y, 1),
            hp=hp, max_hp=max_hp,
            mana=mana, max_mana=max_mana,
            level=level, last_hits=cs,
            denies=denies, net_worth=gold,
        ))
    return snapshots


def _generate_laning_creep_kills(is_core: bool = True) -> list[LaningCreepKill]:
    """Generate realistic last-hit events during laning phase."""
    kills = []
    num = random.randint(30, 55) if is_core else random.randint(5, 15)
    for _ in range(num):
        t = random.randint(30, 599)
        creep_type = random.choice(["melee", "melee", "melee", "ranged", "siege"])
        if creep_type == "melee":
            gold = random.randint(36, 44)
        elif creep_type == "ranged":
            gold = random.randint(44, 52)
        else:
            gold = random.randint(60, 80)
        kills.append(LaningCreepKill(time=t, creep_type=creep_type, gold=gold))
    return sorted(kills, key=lambda k: k.time)


def _generate_laning_denies() -> list[LaningCreepKill]:
    """Generate a few deny events during laning phase."""
    denies = []
    num = random.randint(1, 8)
    for _ in range(num):
        t = random.randint(60, 599)
        creep_type = random.choice(["melee", "melee", "ranged"])
        denies.append(LaningCreepKill(time=t, creep_type=creep_type, gold=0))
    return sorted(denies, key=lambda d: d.time)


def _generate_laning_damage_events(
    hero_idx: int, num_players: int,
) -> list[LaningDamageEvent]:
    """Generate hero-vs-hero damage trades during laning phase.

    target_idx is the index of the target in match.players (0-9), matching
    the clarity Java analyzer output — NOT player_slot (dire slots are 128+).
    """
    events = []
    num = random.randint(10, 30)
    # Target indices (into match.players) from opposing team
    if hero_idx < 5:
        possible_targets = list(range(5, num_players))
    else:
        possible_targets = list(range(0, 5))
    if not possible_targets:
        return events
    for _ in range(num):
        t = random.randint(30, 599)
        target_idx = random.choice(possible_targets)
        damage = random.randint(20, 150)
        inflictor = random.choice(["attack", "attack", "attack", "ability"])
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
        num = random.randint(min_uses, max_uses)
        for _ in range(num):
            t = random.randint(30, 599)
            events.append(LaningRegenEvent(time=t, item=item))
    return sorted(events, key=lambda e: e.time)


def generate_mock_match() -> MatchData:
    """Generate mock match based on real match 8713267082."""
    random.seed(42)

    match_id = 8713267082
    duration = 1913  # 31:53
    radiant_win = False

    # Scoreboard of real match 8713267082; the player names are invented.
    # (slot, hero_id, hero_name, slug, player_name, role, K, D, A, lvl, lh, dn, gpm, xpm, hdmg, tdmg, heal, nw, final_items, final_keys, abilities)
    real_players = [
        # Radiant
        (0, 123, "Hoodwink", "hoodwink", "Mock Squirrel", "support",
         7, 5, 7, 15, 49, 4, 294, 433, 13784, 0, 0, 8357,
         ["Power Treads", "Arcane Boots", "Rod of Atos", "Perseverance"],
         ["power_treads", "arcane_boots", "rod_of_atos", "pers"],
         [("Acorn Shot", "hoodwink_acorn_shot"), ("Bushwhack", "hoodwink_bushwhack"),
          ("Scurry", "hoodwink_scurry"), ("Sharpshooter", "hoodwink_sharpshooter")]),
        (1, 6, "Drow Ranger", "drow_ranger", "Mock Archer", "carry",
         4, 9, 9, 15, 190, 7, 422, 426, 18527, 1055, 0, 11058,
         ["Power Treads", "Quelling Blade", "Magic Wand", "Falcon Blade", "Aghanim's Scepter", "Dragon Lance"],
         ["power_treads", "quelling_blade", "magic_wand", "falcon_blade", "ultimate_scepter", "dragon_lance"],
         [("Frost Arrows", "drow_ranger_frost_arrows"), ("Gust", "drow_ranger_wave_of_silence"),
          ("Multishot", "drow_ranger_multishot"), ("Marksmanship", "drow_ranger_marksmanship")]),
        (2, 37, "Warlock", "warlock", "Mock Warlock", "support",
         2, 12, 11, 13, 58, 2, 269, 329, 19765, 119, 529, 6151,
         ["Arcane Boots", "Magic Wand", "Force Staff"],
         ["arcane_boots", "magic_wand", "force_staff"],
         [("Fatal Bonds", "warlock_fatal_bonds"), ("Shadow Word", "warlock_shadow_word"),
          ("Upheaval", "warlock_upheaval"), ("Chaotic Offering", "warlock_rain_of_chaos")]),
        (3, 15, "Razor", "razor", "Mock Stormcaller", "offlane",
         2, 12, 12, 16, 116, 9, 350, 464, 16909, 0, 0, 10364,
         ["Aghanim's Scepter", "Power Treads", "Infused Raindrops", "Pipe of Insight", "Wraith Band"],
         ["ultimate_scepter", "power_treads", "infused_raindrop", "pipe", "wraith_band"],
         [("Plasma Field", "razor_plasma_field"), ("Static Link", "razor_static_link"),
          ("Storm Surge", "razor_unstable_current"), ("Eye of the Storm", "razor_eye_of_the_storm")]),
        (4, 2, "Axe", "axe", "Mock Axeman", "mid",
         4, 13, 5, 16, 116, 9, 312, 447, 15307, 513, 0, 7654,
         ["Phase Boots", "Blade Mail", "Blink Dagger", "Urn of Shadows", "Quelling Blade"],
         ["phase_boots", "blade_mail", "blink", "urn_of_shadows", "quelling_blade"],
         [("Berserker's Call", "axe_berserkers_call"), ("Battle Hunger", "axe_battle_hunger"),
          ("Counter Helix", "axe_counter_helix"), ("Culling Blade", "axe_culling_blade")]),
        # Dire
        (128, 27, "Shadow Shaman", "shadow_shaman", "Mock Shaman", "support",
         3, 7, 20, 17, 44, 0, 329, 494, 12242, 771, 0, 8122,
         ["Blink Dagger", "Aether Lens", "Observer and Sentry Wards", "Null Talisman", "Tranquil Boots"],
         ["blink", "aether_lens", "ward_dispenser", "null_talisman", "tranquil_boots"],
         [("Ether Shock", "shadow_shaman_ether_shock"), ("Hex", "shadow_shaman_voodoo"),
          ("Shackles", "shadow_shaman_shackles"), ("Mass Serpent Ward", "shadow_shaman_mass_serpent_ward")]),
        (129, 47, "Viper", "viper", "Mock Viper", "carry",
         18, 3, 21, 19, 137, 11, 608, 635, 24619, 9117, 0, 18090,
         ["Hurricane Pike", "Manta Style", "Butterfly", "Power Treads", "Wraith Band", "Boots of Speed"],
         ["hurricane_pike", "manta", "butterfly", "power_treads", "wraith_band", "boots_of_speed"],
         [("Poison Attack", "viper_poison_attack"), ("Nethertoxin", "viper_nethertoxin"),
          ("Corrosive Skin", "viper_corrosive_skin"), ("Viper Strike", "viper_viper_strike")]),
        (130, 26, "Lion", "lion", "Mock Lion", "support",
         10, 2, 23, 21, 22, 4, 380, 776, 12195, 1077, 525, 11278,
         ["Blink Dagger", "Glimmer Cape", "Tranquil Boots", "Aether Lens", "Blood Grenade"],
         ["blink", "glimmer_cape", "tranquil_boots", "aether_lens", "blood_grenade"],
         [("Earth Spike", "lion_impale"), ("Hex", "lion_voodoo"),
          ("Mana Drain", "lion_mana_drain"), ("Finger of Death", "lion_finger_of_death")]),
        (131, 49, "Dragon Knight", "dragon_knight", "Mock Dragon", "mid",
         20, 1, 9, 24, 320, 6, 871, 1066, 36926, 15592, 0, 26059,
         ["Black King Bar", "Mjollnir", "Power Treads", "Aghanim's Scepter", "Satanic", "Blink Dagger"],
         ["black_king_bar", "mjollnir", "power_treads", "ultimate_scepter", "satanic", "blink"],
         [("Breathe Fire", "dragon_knight_breathe_fire"), ("Dragon Tail", "dragon_knight_dragon_tail"),
          ("Dragon Blood", "dragon_knight_dragon_blood"), ("Elder Dragon Form", "dragon_knight_elder_dragon_form")]),
        (132, 99, "Bristleback", "bristleback", "Mock Bristle", "offlane",
         0, 6, 17, 17, 154, 6, 426, 528, 8532, 5088, 0, 13390,
         ["Black King Bar", "Power Treads", "Lotus Orb", "Magic Wand", "Bracer", "Bracer"],
         ["black_king_bar", "power_treads", "lotus_orb", "magic_wand", "bracer", "bracer"],
         [("Viscous Nasal Goo", "bristleback_viscous_nasal_goo"), ("Quill Spray", "bristleback_quill_spray"),
          ("Bristleback", "bristleback_bristleback"), ("Warpath", "bristleback_warpath")]),
    ]

    all_wards: list[WardEvent] = []
    all_ability_events: list[AbilityEvent] = []
    players: list[PlayerData] = []

    # Build hero list for teamfight generation
    # Format: (hero_id, hero_name, slug, abilities) - abilities is last element
    selected_heroes = [(p[1], p[2], p[3], p[-1]) for p in real_players]

    num_players = len(real_players)

    for hero_idx, (slot, hero_id, hero_name, slug, player_name, role,
         kills, deaths, assists, level, last_hits, denies_,
         gpm, xpm, hero_damage, tower_damage, hero_healing, net_worth,
         final_items, final_keys, abilities) in enumerate(real_players):

        team = "radiant" if slot < 128 else "dire"
        is_core = role in ("carry", "mid", "offlane")
        items, _, _ = _generate_item_build(role, duration)
        skills = _generate_skill_build(abilities, duration)
        wards = _real_wards_for_player(slot)
        ability_events = _generate_ability_events(slot, hero_name, abilities, duration)

        # Generate laning phase data
        laning_snapshots = _generate_laning_snapshots(hero_idx, is_core=is_core)
        laning_creep_kills = _generate_laning_creep_kills(is_core=is_core)
        laning_denies = _generate_laning_denies()
        laning_damage_events = _generate_laning_damage_events(hero_idx, num_players)
        laning_regen_events = _generate_laning_regen_events()

        all_wards.extend(wards)
        all_ability_events.extend(ability_events)

        players.append(PlayerData(
            player_slot=slot,
            hero_id=hero_id,
            hero_name=hero_name,
            hero_slug=slug,
            player_name=player_name,
            team=team,
            kills=kills, deaths=deaths, assists=assists,
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
            laning_snapshots=laning_snapshots,
            laning_creep_kills=laning_creep_kills,
            laning_denies=laning_denies,
            laning_damage_events=laning_damage_events,
            laning_regen_events=laning_regen_events,
        ))

    all_wards.sort(key=lambda w: w.time)
    all_ability_events.sort(key=lambda e: e.time)

    radiant_smokes = _generate_smokes("radiant", duration)
    dire_smokes = _generate_smokes("dire", duration)
    smokes = sorted(radiant_smokes + dire_smokes, key=lambda s: s.time)
    teamfights = _generate_teamfights(selected_heroes, duration)

    gold_xp_timeline = _generate_gold_xp_timeline(duration, radiant_win)

    return MatchData(
        match_id=match_id,
        duration=duration,
        radiant_win=radiant_win,
        game_mode="All Pick",
        start_time=1741024620,  # 2025-03-03 17:57 UTC (2025-03-04 01:57 UTC+8)
        radiant_score=19,
        dire_score=51,
        players=players,
        wards=all_wards,
        smokes=smokes,
        teamfights=teamfights,
        ability_events=all_ability_events,
        gold_xp_timeline=gold_xp_timeline,
    )
