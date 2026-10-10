"""Data models for the AI coaching system.

Coordinates: every x/y stored here (death positions, hero states, causal-step
positions, last-seen positions) is in Dota 2 *world units* like the rest of
``MatchData`` (see ``backend/app/geo.py``). ``(0.0, 0.0)`` means "unknown".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Version of the CoachingReport payload shape (bumped whenever a field's
# meaning changes). Re-exported by ``coach.REPORT_SCHEMA_VERSION``; the server
# puts it into the disk-cache file name so stale reports are never served.
# 4: evaluations carry ``sources`` (knowledge entries behind a verdict) and the
# report a ``knowledge`` object (how those entries were retrieved).
REPORT_SCHEMA_VERSION: int = 4

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
    killer_hero: str = ""   # display name of the hero that got the killing blow ("" = none/unknown)
    killer_slot: int | None = None  # player_slot of that hero; None = no hero killer or unknown
    killer_unit: str = ""   # npc name of the killing unit when it is not a hero (tower, creep, ...)
    assist_slots: list[int] = field(default_factory=list)  # player_slots credited with an assist
    denied: bool = False    # an ally got the killing blow (killer_slot / killer_hero are that ally)
    confirmed: bool = True  # False = the source cannot tell this death from a reincarnation
    x: float = 0.0          # death position (world units), 0/0 = unknown
    y: float = 0.0
    # Part of a teamfight: inside a fight window and, when positions are
    # known, with another death nearby (a pick-off elsewhere is not).
    is_teamfight: bool = False
    teamfight_idx: int = -1  # index into match teamfights list; -1 = none
    # Died far from allied towers on the enemy side; only judged outside
    # teamfights and when the death position is known.
    overextended: bool = False
    source: str = ""        # "kills_log" (exact), or a heuristic: "teamfight", "laning", "abilities", "gold"


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
    time_known: bool = True  # False = the source has no timestamp for this point (time is 0)
    # Finer kind of a good_play: "multi_kill", "kill_streak" or "smoke"; "" otherwise.
    subtype: str = ""


def is_time_unknown(dp: DecisionPoint) -> bool:
    """Whether this decision point's timestamp is unknown.

    A point with ``time_known=False`` carries no timestamp in the source data
    (a multi-kill or streak the kill log could not place); downstream
    context/prompt builders must not treat it as a pregame moment and the
    payload reports ``time: null``. The flag alone decides: a point dated at
    or before 0:00 is a real moment (kills happen before the horn).
    """
    return not dp.time_known


# English unit labels of data_manager.unit_display_name that read as common
# nouns and take an article inside a sentence ("killed by a tower").
_GENERIC_UNITS_EN = ("Lane creep", "Siege creep", "Neutral creep", "Tower", "Ancient", "Barracks",
                     "Fountain", "Courier")


# The fountain's unit name carries no "npc_dota_" prefix and no side
# ("dota_fountain"), so the generic unit labels do not reach it.
_FOUNTAIN_TEXT = {
    "zh": {"radiant": "天辉泉水", "dire": "夜魇泉水", "": "敌方泉水"},
    "en": {"radiant": "Radiant fountain", "dire": "Dire fountain", "": "enemy fountain"},
}


def _is_fountain(unit: str) -> bool:
    return "fountain" in str(unit or "").lower()


def unit_text(unit: str, lang: str, *, article: bool = True, team: str = "") -> str:
    """Display name of a non-hero unit as it stands inside a sentence.

    An English generic unit gets its article and lower case ("a tower", "a
    neutral creep"); ``article=False`` keeps the bare label ("Tower") for a
    field such as "Killer: ...". A fountain is named with its side when
    ``team`` ("radiant" / "dire") tells it ("the Dire fountain" / "夜魇泉水"),
    else as the victim's enemy fountain: a fountain only attacks enemy
    heroes. "" for an empty key.
    """
    if not unit:
        return ""
    from .data_manager import unit_display_name

    if _is_fountain(unit):
        labels = _FOUNTAIN_TEXT["en" if lang == "en" else "zh"]
        name = labels.get(team if team in ("radiant", "dire") else "", labels[""])
        if lang != "en":
            return name
        return f"the {name}" if article else name[:1].upper() + name[1:]
    name = unit_display_name(unit, lang)
    if article and lang == "en" and name in _GENERIC_UNITS_EN:
        return f"{'an' if name[0] in 'AEIOU' else 'a'} {name.lower()}"
    return name


def killed_by_fountain(death: DeathEvent | None) -> bool:
    """Whether the enemy fountain got the kill.

    Such a death has one cause whatever else went on: the hero walked into
    the fountain's attack range inside the enemy base.
    """
    return (death is not None and not death.killer_hero and not death.denied
            and _is_fountain(death.killer_unit))


def killed_by_tower(death: DeathEvent | None) -> bool:
    """Whether an enemy tower got the kill (the hero stood inside its attack range)."""
    return (death is not None and not death.killer_hero and not death.denied
            and "tower" in str(death.killer_unit or "").lower())


def killer_text(death: DeathEvent | None, lang: str, *, article: bool = True) -> str:
    """Display name of the hero or unit that got the kill ("" = unknown)."""
    if death is None:
        return ""
    if death.killer_hero:
        from .data_manager import hero_display_name

        return hero_display_name(death.killer_hero, lang)
    # The side of a unit that kills a hero outright is the other team's.
    side = "" if death.denied else ("dire" if death.player_slot < 128 else "radiant")
    return unit_text(death.killer_unit, lang, article=article, team=side)


@dataclass
class HeroState:
    """Snapshot of a hero's state at a point in time."""
    hero_name: str
    player_slot: int
    team: str
    x: float = 0.0          # world units; 0/0 = position unknown
    y: float = 0.0
    hp: int = 0             # absolute HP / mana: laning snapshots only (0 = not in the data)
    max_hp: int = 0
    mana: int = 0
    max_mana: int = 0
    hp_pct: int | None = None  # HP in percent (0-100) at state_time; None = unknown, 0 = dead
    level: int = 0
    last_hits: int = 0
    denies: int | None = 0  # None = the source has no deny count for this moment
    net_worth: int = 0      # laning snapshots: net worth; later: gold earned so far (gold_t)
    items: list[str] = field(default_factory=list)
    abilities_available: list[str] = field(default_factory=list)
    # {display name: skill points}; the name is the current one for the
    # ability key (data_manager.ability_display_name), not the parsed name.
    ability_levels: dict[str, int] = field(default_factory=dict)
    # False = the purchase log has nothing for this moment (it starts later,
    # or is empty): an empty ``items`` then means "not in the data", not "none".
    items_known: bool = True
    is_alive: bool = True
    distance_to_player: float | None = None  # world units; None = unknown/self
    # Game second this state describes. For a death it is a few seconds BEFORE
    # the death (the last moment the hero was alive), never the death tick.
    state_time: int | None = None

    @property
    def has_position(self) -> bool:
        return not (self.x == 0.0 and self.y == 0.0)


@dataclass
class EventEntry:
    """A single event in a timeline."""
    time: int
    event_type: str         # "damage_dealt", "damage_taken", "ability_cast",
                            # "ward_placed", "smoke_used", "kill", "death",
                            # "buyback", vision events
    description: str        # human-readable description (report language)
    details: dict = field(default_factory=dict)


@dataclass
class EnemyVisibility:
    """Tracking info for an enemy hero's visibility."""
    hero_name: str
    player_slot: int
    is_visible: bool        # whether visible right now
    # Game second of the last sighting; 0 with last_seen_source "unknown"
    # when there was none within the lookback.
    last_seen_time: int
    last_seen_x: float = 0.0
    last_seen_y: float = 0.0
    # State "fog": measured length of the unbroken fog run (never a guess);
    # 0 for every other state.
    seconds_missing: int = 0
    # "ward", "tower" (a tower, the ancient or the fountain), "hero",
    # "teamfight" (fighting an allied hero), "unknown"
    last_seen_source: str = ""
    # "visible": inside allied vision; "fog": position known and outside vision
    # (also a hero under Smoke of Deceit walking through a ward's range);
    # "unknown": the match data has no position for this hero at this time
    # (never reported as missing); "dead": waiting to respawn.
    state: str = "visible"


@dataclass
class VisionEvent:
    """A vision-related event in a timeline."""
    time: int
    event_type: str         # "ward_placed", "ward_expired", "ward_destroyed",
                            # "enemy_seen", "enemy_missing", "entered_fog" (the
                            # player walked away from wards and towers),
                            # "group_move" (the same, with two or more allies),
                            # "tower_lost", "night_start", "day_start"
    description: str        # human-readable (report language)
    # Changed the vision around the player (a ward or tower lost near him,
    # his own walk out of cover); flagged in the prompt text.
    important: bool = False


@dataclass
class VisionState:
    """Complete vision picture at a specific moment."""
    time: int
    is_night: bool
    allied_observer_count: int          # active observer wards on our team
    enemy_visibility: list[EnemyVisibility] = field(default_factory=list)
    missing_enemy_count: int = 0        # enemies in fog (state "fog" only)
    missing_enemy_names: list[str] = field(default_factory=list)
    unknown_enemy_count: int = 0        # enemies with no position data (state "unknown")
    unknown_enemy_names: list[str] = field(default_factory=list)
    longest_missing_seconds: int = 0    # longest time any enemy has been missing
    # The enemies in fog that are close enough to the player to matter
    # (vision.fog_threats: within vision.THREAT_RADIUS). Text about "enemies
    # out of sight" at a decision is written from these, not from every
    # hero hidden somewhere on the map.
    threat_enemy_count: int = 0
    threat_enemy_names: list[str] = field(default_factory=list)
    threat_longest_seconds: int = 0
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
    step_type: str              # "vision_loss", "enemy_missing" (left allied vision),
                                # "hidden_enemy_near" (already hidden, now within reach),
                                # "ignore_warning", "position_forward", "position_unsafe",
                                # "no_retreat", "ally_absent", "allies_fallen" (allies died
                                # first in a fight that did not start outnumbered),
                                # "farm_under_threat", "enemy_approach", "cc_chain",
                                # "damage_burst", "rune_contest", "death"
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
    # Tool-calling investigation steps (agent mode). Kept for logs and tests;
    # not part of the report payload (the frontend does not show it).
    agent_trace: list = field(default_factory=list)
    used_llm: bool = False   # the verdict came from an LLM
    degraded: bool = False   # an LLM was supposed to answer but failed; rules were used
    # Knowledge entries behind the evaluation, as the report lists them: see
    # :func:`sources_payload`.
    sources: list[dict] = field(default_factory=list)
    # ``K#`` handles an LLM named as used, in citation order. The LLM paths
    # reduce it to the handles the model was really shown.
    cited_refs: list[str] = field(default_factory=list)


# Most entries an evaluation lists as sources; a rule-based evaluation (which
# never read the knowledge) lists fewer.
MAX_SOURCES = 4
RULE_SOURCES = 3


def sources_payload(refs, cited=(), limit: int = MAX_SOURCES) -> list[dict]:
    """The ``sources`` list of an evaluation from knowledge refs.

    ``refs`` are :class:`rag.types.KnowledgeRef` in rank order, ``cited`` the
    handles ("K2") a verdict relied on. Cited entries come first, in
    citation order and marked ``cited: true``; the best-ranked others fill
    the list up to ``limit``. One entry per knowledge chunk; a handle that is
    not among ``refs`` is ignored. Items are ``{id, title, source_type,
    patch, cited}``.
    """
    by_handle = {ref.ref: ref for ref in refs}
    ordered = [(by_handle[h], True) for h in dict.fromkeys(cited) if h in by_handle]
    ordered += [(ref, False) for ref in refs]
    out: list[dict] = []
    seen: set[str] = set()
    for ref, is_cited in ordered:
        if ref.chunk_id in seen:
            continue
        seen.add(ref.chunk_id)
        out.append(ref.to_payload(cited=is_cited))
        if len(out) >= limit:
            break
    return out


@dataclass
class PhaseReview:
    """Per-phase review of a player's performance.

    Phases are 0-10 min (laning), 10-20 (early_mid), 20-30 (mid_late),
    30-40 (late), 40+ (ultra_late). Generated by evaluator.evaluate_phase().
    """

    phase: str                  # "laning" / "early_mid" / "mid_late" / "late" / "ultra_late"
    time_range: str             # localized, e.g. "0-10 min" / "0-10 分钟"
    start_min: int = 0          # phase start in minutes
    end_min: int | None = None  # phase end in minutes; None = open-ended (until game end)
    score: str = "C"            # "S" / "A" / "B" / "C" / "D"
    summary: str = ""           # one-line analysis
    highlights: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)   # serialized PhaseStats
    used_llm: bool = False
    degraded: bool = False      # LLM configured but failed; rule-based review used


def _json_safe(value):
    """Copy of a payload fragment that ``json.dumps(allow_nan=False)`` accepts.

    Non-finite floats become None; tuples and sets become lists.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _evaluation_dict(e: Evaluation, *, with_chain: bool) -> dict:
    item = {
        "time": None if is_time_unknown(e.decision_point) else e.decision_point.time,
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
        item["causal_chain"] = _json_safe(e.causal_chain_data) if e.causal_chain_data else None
    item["llm_used"] = bool(e.used_llm)
    item["degraded"] = bool(e.degraded)
    item["sources"] = _json_safe(list(e.sources or []))
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
    # How the knowledge entries of this report were retrieved:
    # {mode: "hybrid" | "lexical", embedder: str | None, chunks: int, vectors: int}.
    knowledge: dict = field(default_factory=dict)
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
            "laning_review": _json_safe(self.laning_review),
            "improvement_areas": _json_safe(self.improvement_areas),
            "strengths": _json_safe(self.strengths),
            "pro_comparison": _json_safe(self.pro_comparison),
            "position": _json_safe(self.position),
            "phase_reviews": [
                {
                    "phase": p.phase,
                    "time_range": p.time_range,
                    "start_min": p.start_min,
                    "end_min": p.end_min,
                    "score": normalize_grade(p.score),
                    "summary": p.summary,
                    "highlights": p.highlights,
                    "issues": p.issues,
                    "suggestions": p.suggestions,
                    "stats": _json_safe(p.stats),
                    "llm_used": bool(p.used_llm),
                    "degraded": bool(p.degraded),
                }
                for p in self.phase_reviews
            ],
            "llm_used": bool(self.llm_used),
            "degraded": bool(self.degraded),
            "knowledge": _json_safe(self.knowledge),
            "report_version": int(self.report_version),
        }
