"""Data models for Dota 2 match analysis.

Coordinates: every x/y below (wards, smokes incl. path, teamfight positions and
ability casts, ability events, laning snapshots, buildings) is in Dota 2 world
units (see ``geo.py``). The only exception is ``PlayerData.lane_pos``, which is
OpenDota's raw cell heatmap.

Enumerations are stable lowercase codes (no localized labels); the frontend
translates them.
"""
from __future__ import annotations

from pydantic import BaseModel


class ItemEvent(BaseModel):
    item_name: str
    time: int  # game time in seconds
    item_key: str = ""  # internal key for CDN icon lookup
    charges: int | None = None


class SkillEvent(BaseModel):
    ability_name: str
    ability_key: str = ""  # internal key for CDN icon lookup
    level: int
    time: int


class WardEvent(BaseModel):
    ward_type: str  # "observer" or "sentry"
    x: float        # world units
    y: float
    time: int       # game time of placement (negative = before the horn)
    player_slot: int  # owner; -1 when the owner is unknown (see ``team``)
    destroyed_time: int | None = None  # dewarded/expired; None = alive at game end
    team: str = ""  # "radiant"/"dire"; empty = derive from player_slot


class SmokeEvent(BaseModel):
    time: int
    team: str  # "radiant" or "dire"
    player_slots: list[int]
    x: float = 0.0              # smoke activation position (world units)
    y: float = 0.0
    end_time: int = 0           # when smoke breaks
    path: list[list[float]] = []  # [[x1,y1], ...] waypoints (world units)
    # Result code: "kill", "traded", "objective", "caught" (only own deaths),
    # "no_contact" or "unknown"
    result: str = "unknown"
    kills_gained: int = 0       # enemy deaths near the group, -5..+30s around the break
    deaths_suffered: int = 0    # own deaths near the group in the same window
    objective_taken: str | None = None  # "tower", "barracks", "ancient", "roshan" or None


class PositionSnapshot(BaseModel):
    """Hero position at a point during a teamfight."""
    x: float
    y: float
    time_offset: float  # seconds from teamfight start


class AbilityCast(BaseModel):
    """A single ability cast during a teamfight."""
    ability_name: str
    ability_key: str = ""
    time_offset: float  # seconds from teamfight start
    x: float
    y: float


class TeamfightPlayer(BaseModel):
    player_slot: int
    hero_name: str
    hero_slug: str = ""
    x: float
    y: float
    damage_dealt: int
    damage_taken: int
    healing: int = 0
    kills: int
    deaths: int
    abilities_used: list[str]
    positions: list[PositionSnapshot] = []
    ability_casts: list[AbilityCast] = []
    death_time_offset: float | None = None  # seconds from teamfight start to death
    kill_targets: list[dict] = []  # [{"target": "hero_name", "time_offset": float}]
    gold_delta: int = 0
    xp_delta: int = 0
    item_uses: dict[str, int] = {}  # items used during fight
    buybacks: int = 0


class TeamfightEvent(BaseModel):
    start_time: int
    end_time: int
    radiant_deaths: int
    dire_deaths: int
    gold_delta: int = 0  # positive = radiant advantage
    players: list[TeamfightPlayer]


class AbilityEvent(BaseModel):
    player_slot: int
    hero_name: str
    ability_name: str
    ability_key: str = ""  # internal key for CDN icon lookup
    time: int
    target: str | None = None
    x: float | None = None
    y: float | None = None


class LaningSnapshot(BaseModel):
    """Hero state snapshot during laning phase (every 5 seconds, 0-600s)."""
    time: int = 0
    x: float = 0.0
    y: float = 0.0
    hp: int = 0
    max_hp: int = 0
    mana: int = 0
    max_mana: int = 0
    level: int = 0
    last_hits: int = 0
    denies: int = 0
    net_worth: int = 0


class LaningCreepKill(BaseModel):
    """A single last-hit or deny event during laning."""
    time: int = 0
    creep_type: str = ""  # "melee", "ranged", "siege", "neutral" or "unknown"
    gold: int = 0


class LaningDamageEvent(BaseModel):
    """Hero-vs-hero damage event during laning."""
    time: int = 0
    target_idx: int = 0  # index of the target in match.players (0-9), NOT player_slot
    damage: int = 0
    inflictor: str = ""


class LaningRegenEvent(BaseModel):
    """Regen item usage during laning."""
    time: int = 0
    item: str = ""


class PlayerData(BaseModel):
    player_slot: int
    hero_id: int
    hero_name: str
    hero_slug: str = ""  # internal name for CDN icon lookup
    player_name: str
    team: str  # "radiant" or "dire"
    kills: int
    deaths: int
    assists: int
    level: int
    last_hits: int
    denies: int
    gold_per_min: int
    xp_per_min: int
    hero_damage: int
    tower_damage: int
    hero_healing: int
    net_worth: int              # real net worth when the source provides it
    items: list[ItemEvent]
    backpack: list[str]         # item keys in the backpack at game end
    final_items: list[str]      # display names of the end-of-game inventory
    final_item_keys: list[str] = []  # internal keys for CDN icons
    neutral_item: str = ""      # item key of the neutral item slot, "" if none/unknown
    skills: list[SkillEvent]
    wards: list[WardEvent]
    ability_events: list[AbilityEvent]

    # Per-minute timelines (for individual player graphs)
    gold_t: list[int] = []      # gold at each minute
    xp_t: list[int] = []        # xp at each minute
    lh_t: list[int] = []        # last hits at each minute

    # Lane data
    # lane is ABSOLUTE (OpenDota convention): 1=bottom, 2=mid, 3=top, 4=jungle, 0=unknown
    lane: int = 0
    # lane_role is RELATIVE to the team: 1=safe, 2=mid, 3=off, 4=jungle/roaming, 0=unknown
    lane_role: int = 0
    is_roaming: bool = False
    lane_efficiency: float = 0.0
    lane_efficiency_pct: float = 0.0

    # Damage breakdowns
    killed: dict[str, int] = {}         # {hero_name: count} - heroes this player killed
    killed_by: dict[str, int] = {}      # {hero_name: count} - killed by
    damage: dict[str, int] = {}         # damage dealt to each unit
    damage_taken: dict[str, int] = {}   # damage received from each unit
    damage_inflictor: dict[str, int] = {} # damage by ability/item name

    # Economy reasons
    gold_reasons: dict[str, int] = {}   # {reason: amount}
    xp_reasons: dict[str, int] = {}     # {reason: amount}

    # Benchmarks
    benchmarks: dict[str, dict] = {}    # {stat: {raw, pct}}

    # Runes (rune type ids as string keys, OpenDota numbering)
    runes_log: list[dict] = []          # [{time, key}]
    runes: dict[str, int] = {}          # {rune_type: count}

    # Additional stats
    stuns: float = 0.0
    actions_per_min: int = 0
    teamfight_participation: float = 0.0
    buyback_count: int = 0
    obs_placed: int = 0
    sen_placed: int = 0
    observer_kills: int = 0
    sentry_kills: int = 0
    lane_kills: int = 0
    neutral_kills: int = 0
    ancient_kills: int = 0
    tower_kills: int = 0
    roshans_killed: int = 0
    courier_kills: int = 0
    multi_kills: dict[str, int] = {}    # {count: times}
    kill_streaks: dict[str, int] = {}   # {streak: times}
    purchase_log: list[dict] = []       # [{time, key}] raw purchase timeline

    # Lane position heatmap
    lane_pos: dict = {}                 # {x: {y: count}} - raw cell data for heatmap

    # Enhanced laning phase data (from Java parser)
    laning_snapshots: list[LaningSnapshot] = []
    laning_creep_kills: list[LaningCreepKill] = []
    laning_denies: list[LaningCreepKill] = []
    laning_damage_events: list[LaningDamageEvent] = []
    laning_regen_events: list[LaningRegenEvent] = []


class GoldXpSnapshot(BaseModel):
    """Team gold/xp at a point in time (for economy graph)."""
    time: int  # game time in seconds
    radiant_gold: int
    dire_gold: int
    radiant_xp: int
    dire_xp: int


class MatchData(BaseModel):
    match_id: int
    duration: int  # seconds
    radiant_win: bool
    game_mode: str  # code, e.g. "ranked_all_pick", "turbo", "captains_mode", "mode_<id>"
    start_time: int
    radiant_score: int
    dire_score: int
    players: list[PlayerData]
    wards: list[WardEvent]
    smokes: list[SmokeEvent]
    teamfights: list[TeamfightEvent]
    ability_events: list[AbilityEvent]
    gold_xp_timeline: list[GoldXpSnapshot] = []  # per-minute snapshots

    # Advantage graphs
    radiant_gold_adv: list[int] = []    # per-minute radiant gold advantage
    radiant_xp_adv: list[int] = []     # per-minute radiant xp advantage

    # Draft
    picks_bans: list[dict] = []        # [{is_pick, hero_id, team, order}]

    # Objectives (OpenDota-style): building kills use type "building_kill" with the
    # destroyed building's unit name in "key" (e.g. "npc_dota_goodguys_tower1_top")
    # and the team that took it in "team"; Roshan kills use "CHAT_MESSAGE_ROSHAN_KILL".
    objectives: list[dict] = []        # [{time, type, key, player_slot, team, ...}]

    # Chat
    chat: list[dict] = []              # [{time, type, key, slot, player_slot}]

    # Building status at game end (OpenDota/Valve bit layout, set bit = alive).
    # Towers: bit0-2 top t1-t3, bit3-5 mid t1-t3, bit6-8 bot t1-t3,
    # bit9 ancient top (t4), bit10 ancient bottom (t4).
    # Barracks: bit0 top melee, bit1 top ranged, bit2 mid melee, bit3 mid ranged,
    # bit4 bot melee, bit5 bot ranged.
    tower_status_radiant: int = 0
    tower_status_dire: int = 0
    barracks_status_radiant: int = 0
    barracks_status_dire: int = 0

    # Buildings with real positions (world units) when the source provides them:
    # [{"name": "npc_dota_goodguys_tower1_top", "team": "radiant"|"dire",
    #   "type": "tower"|"barracks"|"fort"|"other", "x": float, "y": float,
    #   "destroyed_time": int | None}]
    # Empty when unknown; consumers then fall back to geo.TOWER_POSITIONS.
    buildings: list[dict] = []

    # True when the replay was truncated/corrupted and only part of it was parsed.
    partial: bool = False

    # First blood
    first_blood_time: int = 0

    # Region/patch
    region: int = 0
    patch: str = ""
