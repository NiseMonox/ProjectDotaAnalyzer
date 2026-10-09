"""Data models for the AI coaching system.

Coordinates: every x/y stored here (death positions, hero states, causal-step
positions, last-seen positions) is in Dota 2 *world units* like the rest of
``MatchData`` (see ``backend/app/geo.py``). ``(0.0, 0.0)`` means "unknown".
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Version of the CoachingReport payload shape (bumped whenever a field's
# meaning changes). Re-exported by ``coach.REPORT_SCHEMA_VERSION``; the server
# puts it into the disk-cache file name so stale reports are never served.
REPORT_SCHEMA_VERSION: int = 2

VALID_GRADES: tuple[str, ...] = ("S", "A", "B", "C", "D")


def normalize_grade(value: object, default: str = "C") -> str:
    """Return a valid letter grade ("S".."D") for any input.

    Accepts things like ``"b"``, ``"B+"``, ``" A "``; anything unusable maps
    to ``default`` so reports never carry empty/None grades.
    """
    if isinstance(value, str):
        text = value.strip().upper()
        if text and text[0] in VALID_GRADES:
            return text[0]
    return default if default in VALID_GRADES else "C"


@dataclass
class DeathEvent:
    """A player death occurrence extracted from match data."""
    time: int               # game second when death occurred
    player_slot: int
    hero_name: str
    killer_hero: str = ""   # hero that got the killing blow
    x: float = 0.0          # death position (world units), 0/0 = unknown
    y: float = 0.0
    is_teamfight: bool = False
    teamfight_idx: int = -1  # index into match teamfights list
    overextended: bool = False  # died far from allied towers on the enemy side
    source: str = ""        # detection strategy: "teamfight", "laning", "abilities", "gold"


@dataclass
class DecisionPoint:
    """A moment where player made a significant decision."""
    time: int               # game second
    point_type: str         # "death", "bad_trade", "missed_cs",
                            # "fight_engage", "overextend", "good_play"
    player_slot: int
    hero_name: str
    severity: str           # "critical", "major", "minor", "positive"
    brief: str              # one-line summary (in the report language)
    related_death: DeathEvent | None = None


def is_time_unknown(dp: DecisionPoint) -> bool:
    """Whether this decision point's timestamp is unknown.

    good_play points built from multi-kill / kill-streak counters carry no
    timestamp in the source data (time=0); treat their time as unknown so
    downstream context/prompt builders do not misclassify them as pregame.
    """
    return dp.point_type == "good_play" and dp.time <= 0


@dataclass
class HeroState:
    """Snapshot of a hero's state at a point in time."""
    hero_name: str
    player_slot: int
    team: str
    x: float = 0.0          # world units; 0/0 = position unknown
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
    distance_to_player: float | None = None  # world units; None = unknown/self

    @property
    def has_position(self) -> bool:
        return not (self.x == 0.0 and self.y == 0.0)


@dataclass
class EventEntry:
    """A single event in a timeline."""
    time: int
    event_type: str         # "damage_dealt", "damage_taken", "ability_cast",
                            # "ward_placed", "smoke_used", "kill", "death", vision events
    description: str        # human-readable description (report language)
    details: dict = field(default_factory=dict)


@dataclass
class EnemyVisibility:
    """Tracking info for an enemy hero's visibility."""
    hero_name: str
    player_slot: int
    is_visible: bool        # whether visible right now
    last_seen_time: int     # game second when last visible (0 = never/unknown)
    last_seen_x: float = 0.0
    last_seen_y: float = 0.0
    seconds_missing: int = 0  # how long since last seen (0 if visible)
    last_seen_source: str = ""  # "ward", "tower", "hero", "ability", "laning", "teamfight", "unknown"


@dataclass
class VisionEvent:
    """A vision-related event in a timeline."""
    time: int
    event_type: str         # "ward_placed", "ward_expired", "ward_destroyed",
                            # "enemy_seen", "enemy_missing", "entered_fog",
                            # "tower_lost", "night_start", "day_start"
    description: str        # human-readable (report language)


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
    danger_reasons: list[str] = field(default_factory=list)  # report language
    player_in_safe_zone: bool = True    # is the player near tower/base
    nearby_ward_coverage: bool = False  # is there a ward covering player's area


@dataclass
class VisionTimeline:
    """Vision analysis for a time window (e.g., 60s before death)."""
    events: list[VisionEvent] = field(default_factory=list)
    critical_moment: int = 0            # when the key vision mistake happened
    critical_description: str = ""      # what happened (report language)
    summary: str = ""                   # overall assessment (report language)


@dataclass
class VisionAnalysis:
    """Complete vision analysis for a decision point."""
    state_at_decision: VisionState      # vision picture at the exact moment
    timeline: VisionTimeline            # vision changes leading up to it
    formatted_text: str = ""            # pre-formatted text for the LLM prompt


@dataclass
class CausalStep:
    """A single step in the causal chain leading to a death."""
    time: int
    step_type: str              # "vision_loss", "enemy_missing", "ignore_warning",
                                # "position_forward", "position_unsafe", "no_retreat",
                                # "ally_absent", "farm_under_threat", "enemy_approach",
                                # "cc_chain", "damage_burst", "rune_contest", "death"
    description: str            # report language
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
                                    # "bad_positioning", "greedy_farm", "isolated",
                                    # "teamfight_outnumbered", "teamfight_positioning",
                                    # "rune_contest_unfavorable"
    root_cause_summary: str = ""    # one-line summary (report language)
    window_start: int = 0
    window_end: int = 0
    formatted_text: str = ""        # pre-formatted text for the LLM prompt


@dataclass
class DecisionContext:
    """Full context for evaluating a decision point."""
    decision_point: DecisionPoint
    player_state: HeroState
    enemy_states: list[HeroState] = field(default_factory=list)  # enemies NEAR the player
    ally_states: list[HeroState] = field(default_factory=list)   # allies NEAR the player
    event_timeline: list[EventEntry] = field(default_factory=list)
    map_state: dict = field(default_factory=dict)   # vision, towers, distant heroes
    game_phase: str = ""    # "pregame", "laning", "mid_game", "late_game"
    match_context: dict = field(default_factory=dict)  # gold/xp adv (team view), score
    vision_analysis: VisionAnalysis | None = None   # detailed vision/fog analysis
    causal_chain: CausalChain | None = None  # death causal chain analysis


@dataclass
class Evaluation:
    """AI evaluation of a single decision point."""
    decision_point: DecisionPoint
    score: str              # "S", "A", "B", "C", "D"
    analysis: str           # detailed analysis text (report language)
    mistake: str = ""       # what went wrong
    better_play: str = ""   # suggested alternative
    lesson: str = ""        # general takeaway
    _parsed_ok: bool = True  # whether the LLM response was parsed successfully
    causal_chain_data: dict = field(default_factory=dict)  # serialized causal chain for frontend
    agent_trace: list = field(default_factory=list)  # tool-calling investigation steps (agent mode)
    used_llm: bool = False   # the verdict came from an LLM
    degraded: bool = False   # an LLM was supposed to answer but failed; rules were used


@dataclass
class PhaseReview:
    """Per-phase review of a player's performance.

    Phases are 0-10 min (laning), 10-20 (early_mid), 20-30 (mid_late),
    30-40 (late), 40+ (ultra_late). Generated by evaluator.evaluate_phase().
    """

    phase: str                  # "laning" / "early_mid" / "mid_late" / "late" / "ultra_late"
    time_range: str             # e.g. "0-10 min"
    score: str = "C"            # "S" / "A" / "B" / "C" / "D"
    summary: str = ""           # one-line analysis
    highlights: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)   # serialized PhaseStats
    used_llm: bool = False
    degraded: bool = False      # LLM configured but failed; rule-based review used


def _evaluation_dict(e: Evaluation, *, with_chain: bool) -> dict:
    item = {
        "time": e.decision_point.time,
        "point_type": e.decision_point.point_type,
        "hero_name": e.decision_point.hero_name,
        "severity": e.decision_point.severity,
        "brief": e.decision_point.brief,
        "score": normalize_grade(e.score),
        "analysis": e.analysis,
        "mistake": e.mistake,
        "better_play": e.better_play,
        "lesson": e.lesson,
    }
    if with_chain:
        item["causal_chain"] = e.causal_chain_data if e.causal_chain_data else None
    item["agent_trace"] = e.agent_trace if e.agent_trace else None
    item["llm_used"] = bool(e.used_llm)
    return item


@dataclass
class CoachingReport:
    """Complete coaching report for a player in a match."""
    match_id: int
    player_slot: int
    hero_name: str
    overall_grade: str      # "S", "A", "B", "C", "D"
    summary: str            # overall summary (report language)
    death_analyses: list[Evaluation] = field(default_factory=list)
    key_moments: list[Evaluation] = field(default_factory=list)
    laning_review: dict = field(default_factory=dict)
    improvement_areas: list[dict] = field(default_factory=list)
    strengths: list[dict] = field(default_factory=list)
    pro_comparison: dict = field(default_factory=dict)  # professional pattern comparison data
    phase_reviews: list[PhaseReview] = field(default_factory=list)
    position: dict = field(default_factory=dict)  # {number, label_zh, label_en, confidence, method}
    llm_used: bool = False   # some LLM output made it into the report
    degraded: bool = False   # an LLM was configured but at least one call failed
    report_version: int = REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "match_id": self.match_id,
            "player_slot": self.player_slot,
            "hero_name": self.hero_name,
            "overall_grade": normalize_grade(self.overall_grade, default="B"),
            "summary": self.summary,
            "death_analyses": [
                _evaluation_dict(e, with_chain=True) for e in self.death_analyses
            ],
            "key_moments": [
                _evaluation_dict(e, with_chain=False) for e in self.key_moments
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
                    "score": normalize_grade(p.score),
                    "summary": p.summary,
                    "highlights": p.highlights,
                    "issues": p.issues,
                    "suggestions": p.suggestions,
                    "stats": p.stats,
                    "used_llm": p.used_llm,
                    "degraded": p.degraded,
                }
                for p in self.phase_reviews
            ],
            "llm_used": bool(self.llm_used),
            "degraded": bool(self.degraded),
            "report_version": int(self.report_version),
        }
