"""Data models for Dota 2 match analysis."""
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
    x: float
    y: float
    time: int
    player_slot: int
    destroyed_time: int | None = None


class SmokeEvent(BaseModel):
    time: int
    team: str  # "radiant" or "dire"
    player_slots: list[int]
    x: float = 0.0              # smoke activation position
    y: float = 0.0
    end_time: int = 0           # when smoke breaks
    path: list[list[float]] = []  # [[x1,y1], ...] waypoints
    result: str = "unknown"     # "kill", "no_contact", "traded", "objective"
    kills_gained: int = 0       # kills within 30s after smoke breaks
    deaths_suffered: int = 0    # deaths within 30s after smoke breaks
    objective_taken: str | None = None  # "tower", "roshan", None


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


class TeamfightEvent(BaseModel):
    start_time: int
    end_time: int
    radiant_deaths: int
    dire_deaths: int
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
    net_worth: int
    items: list[ItemEvent]
    backpack: list[str]
    final_items: list[str]
    final_item_keys: list[str] = []  # internal keys for CDN icons
    skills: list[SkillEvent]
    wards: list[WardEvent]
    ability_events: list[AbilityEvent]


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
    game_mode: str
    start_time: int
    radiant_score: int
    dire_score: int
    players: list[PlayerData]
    wards: list[WardEvent]
    smokes: list[SmokeEvent]
    teamfights: list[TeamfightEvent]
    ability_events: list[AbilityEvent]
    gold_xp_timeline: list[GoldXpSnapshot] = []  # per-minute snapshots


class AnalysisResult(BaseModel):
    match: MatchData
    parse_source: str  # "clarity", "mock", etc.
