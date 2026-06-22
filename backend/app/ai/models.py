"""Data models for AI coaching system."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DeathEvent:
    """A player death occurrence extracted from match data."""
    time: int               # game second when death occurred
    player_slot: int
    hero_name: str
    killer_hero: str = ""   # hero that got the killing blow
    x: float = 0.0          # position where death occurred
    y: float = 0.0
    is_teamfight: bool = False
    teamfight_idx: int = -1 # index into match teamfights list


@dataclass
class DecisionPoint:
    """A moment where player made a significant decision."""
    time: int               # game second
    point_type: str         # "death", "bad_trade", "missed_cs", "no_retreat",
                            # "fight_engage", "overextend", "good_play"
    player_slot: int
    hero_name: str
    severity: str           # "critical", "major", "minor", "positive"
    brief: str              # one-line summary (Chinese)
    related_death: DeathEvent | None = None


@dataclass
class HeroState:
    """Snapshot of a hero's state at a point in time."""
    hero_name: str
    player_slot: int
    team: str
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
    items: list[str] = field(default_factory=list)
    abilities_available: list[str] = field(default_factory=list)
    ability_levels: dict[str, int] = field(default_factory=dict)  # {ability_name: current_level}
    is_alive: bool = True


@dataclass
class EventEntry:
    """A single event in a timeline."""
    time: int
    event_type: str         # "damage_dealt", "damage_taken", "ability_cast",
                            # "item_use", "movement", "kill", "death"
    description: str        # human-readable description (Chinese)
    details: dict = field(default_factory=dict)


@dataclass
class EnemyVisibility:
    """Tracking info for an enemy hero's visibility."""
    hero_name: str
    player_slot: int
    is_visible: bool        # whether visible right now
    last_seen_time: int     # game second when last visible
    last_seen_x: float = 0.0
    last_seen_y: float = 0.0
    seconds_missing: int = 0  # how long since last seen (0 if visible)
    last_seen_source: str = ""  # "ward", "tower", "hero", "ability", "unknown"


@dataclass
class VisionEvent:
    """A vision-related event in a timeline."""
    time: int
    event_type: str         # "ward_placed", "ward_expired", "ward_destroyed",
                            # "enemy_seen", "enemy_missing", "entered_fog",
                            # "tower_lost", "night_start", "day_start"
    description: str        # human-readable (Chinese)


@dataclass
class VisionState:
    """Complete vision picture at a specific moment."""
    time: int
    is_night: bool
    allied_observer_count: int          # active observer wards on our team
    enemy_visibility: list[EnemyVisibility] = field(default_factory=list)
    missing_enemy_count: int = 0        # enemies in fog
    missing_enemy_names: list[str] = field(default_factory=list)
    longest_missing_seconds: int = 0    # longest time any enemy has been missing
    danger_level: str = "safe"          # "safe", "caution", "danger", "critical"
    danger_reasons: list[str] = field(default_factory=list)  # Chinese reasons
    player_in_safe_zone: bool = True    # is the player near tower/base
    nearby_ward_coverage: bool = False  # is there a ward covering player's area


@dataclass
class VisionTimeline:
    """Vision analysis for a time window (e.g., 60s before death)."""
    events: list[VisionEvent] = field(default_factory=list)
    critical_moment: int = 0            # when the key vision mistake happened
    critical_description: str = ""      # what happened (Chinese)
    summary: str = ""                   # overall assessment (Chinese)


@dataclass
class VisionAnalysis:
    """Complete vision analysis for a decision point."""
    state_at_decision: VisionState      # vision picture at the exact moment
    timeline: VisionTimeline            # vision changes leading up to it
    formatted_text: str = ""            # pre-formatted text for LLM prompt (Chinese)


@dataclass
class CausalStep:
    """A single step in the causal chain leading to a death."""
    time: int
    step_type: str              # "vision_loss", "enemy_missing", "ignore_warning",
                                # "position_forward", "position_unsafe", "no_retreat",
                                # "ally_absent", "enemy_approach", "cc_chain",
                                # "damage_burst", "death"
    description: str            # Chinese description
    significance: str           # "root_cause", "contributing", "consequence", "final"
    position: tuple[float, float] = (0.0, 0.0)
    danger_level: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class CausalChain:
    """Complete causal chain for a death event."""
    death_event: DeathEvent
    steps: list[CausalStep] = field(default_factory=list)
    root_cause_index: int = 0
    root_cause_type: str = ""       # "vision_loss", "overextend", "ignore_missing",
                                    # "bad_positioning", "greedy_farm", "isolated", "teamfight_timing"
    root_cause_summary: str = ""    # one-line Chinese summary
    window_start: int = 0
    window_end: int = 0
    formatted_text: str = ""        # pre-formatted Chinese text for LLM


@dataclass
class DecisionContext:
    """Full context for evaluating a decision point."""
    decision_point: DecisionPoint
    player_state: HeroState
    enemy_states: list[HeroState] = field(default_factory=list)
    ally_states: list[HeroState] = field(default_factory=list)
    event_timeline: list[EventEntry] = field(default_factory=list)
    map_state: dict = field(default_factory=dict)   # vision, towers
    game_phase: str = ""    # "laning", "mid_game", "late_game"
    match_context: dict = field(default_factory=dict)  # gold/xp adv, score
    vision_analysis: VisionAnalysis | None = None   # detailed vision/fog analysis
    causal_chain: CausalChain | None = None  # death causal chain analysis


@dataclass
class Evaluation:
    """AI evaluation of a single decision point."""
    decision_point: DecisionPoint
    score: str              # "S", "A", "B", "C", "D"
    analysis: str           # detailed analysis text (Chinese)
    mistake: str = ""       # what went wrong
    better_play: str = ""   # suggested alternative
    lesson: str = ""        # general takeaway
    _parsed_ok: bool = True # whether LLM response was parsed successfully
    causal_chain_data: dict = field(default_factory=dict)  # serialized causal chain for frontend
    agent_trace: list = field(default_factory=list)  # tool-calling investigation steps (agent mode)


@dataclass
class PhaseReview:
    """Per-phase review of a player's performance.

    Phases are 0-10 min (laning), 10-20 (early_mid), 20-30 (mid_late),
    30-40 (late), 40+ (ultra_late). Generated by evaluator.evaluate_phase().
    """

    phase: str                  # "laning" / "early_mid" / "mid_late" / "late" / "ultra_late"
    time_range: str             # human label, e.g. "0-10 min"
    score: str = "C"            # "S" / "A" / "B" / "C" / "D"
    summary: str = ""           # one-line analysis
    highlights: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)   # serialized PhaseStats
    used_llm: bool = False


@dataclass
class CoachingReport:
    """Complete coaching report for a player in a match."""
    match_id: int
    player_slot: int
    hero_name: str
    overall_grade: str      # "S", "A", "B", "C", "D"
    summary: str            # overall summary (Chinese)
    death_analyses: list[Evaluation] = field(default_factory=list)
    key_moments: list[Evaluation] = field(default_factory=list)
    laning_review: dict = field(default_factory=dict)
    improvement_areas: list[dict] = field(default_factory=list)
    strengths: list[dict] = field(default_factory=list)
    pro_comparison: dict = field(default_factory=dict)  # professional pattern comparison data
    phase_reviews: list[PhaseReview] = field(default_factory=list)
    position: dict = field(default_factory=dict)  # {number, label_zh, label_en, confidence, method}

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "match_id": self.match_id,
            "player_slot": self.player_slot,
            "hero_name": self.hero_name,
            "overall_grade": self.overall_grade,
            "summary": self.summary,
            "death_analyses": [
                {
                    "time": e.decision_point.time,
                    "point_type": e.decision_point.point_type,
                    "hero_name": e.decision_point.hero_name,
                    "severity": e.decision_point.severity,
                    "brief": e.decision_point.brief,
                    "score": e.score,
                    "analysis": e.analysis,
                    "mistake": e.mistake,
                    "better_play": e.better_play,
                    "lesson": e.lesson,
                    "causal_chain": e.causal_chain_data if e.causal_chain_data else None,
                    "agent_trace": e.agent_trace if e.agent_trace else None,
                }
                for e in self.death_analyses
            ],
            "key_moments": [
                {
                    "time": e.decision_point.time,
                    "point_type": e.decision_point.point_type,
                    "hero_name": e.decision_point.hero_name,
                    "severity": e.decision_point.severity,
                    "brief": e.decision_point.brief,
                    "score": e.score,
                    "analysis": e.analysis,
                    "mistake": e.mistake,
                    "better_play": e.better_play,
                    "lesson": e.lesson,
                    "agent_trace": e.agent_trace if e.agent_trace else None,
                }
                for e in self.key_moments
            ],
            "laning_review": self.laning_review,
            "improvement_areas": self.improvement_areas,
            "strengths": self.strengths,
            "pro_comparison": self.pro_comparison,
            "position": self.position,
            "phase_reviews": [
                {
                    "phase": p.phase,
                    "time_range": p.time_range,
                    "score": p.score,
                    "summary": p.summary,
                    "highlights": p.highlights,
                    "issues": p.issues,
                    "suggestions": p.suggestions,
                    "stats": p.stats,
                    "used_llm": p.used_llm,
                }
                for p in self.phase_reviews
            ],
        }
