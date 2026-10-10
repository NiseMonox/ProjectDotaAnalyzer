"""Derive Dota 2 positions (1-5) from match data.

Single source of truth for "which role did this player play". Dota positions
are role assignments, not lanes:

  Position 1 (carry)         — safe-lane core, takes most farm
  Position 2 (mid)           — solo mid lane
  Position 3 (offlaner)      — off-lane core
  Position 4 (support)       — soft support, often roams
  Position 5 (hard support)  — hard support, usually with the carry

Each player's position is derived from three signals, in order of
trustworthiness:

1. **Lane role** — ``PlayerData.lane_role`` is relative to the team
   (1=safe, 2=mid, 3=off, 4=jungle/roaming, 0=unknown). When it is missing it
   is derived from the absolute ``lane`` (1=bottom, 2=mid, 3=top) and the
   team: Radiant's safe lane is the bottom, Dire's is the top.
2. **Farm ranking within a lane** — the safe lane holds pos 1 + pos 5, the
   off lane pos 3 + pos 4; the higher net-worth player is the core.
3. **Hero archetype** from the knowledge base — for players that could not
   be placed by lane (roamers / unknown lane, or a match without lane data):
   the open core positions go to the heroes that are usually cores, richest
   first, and the rest are supports.

Every position 1-5 is assigned at most once per team.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models import MatchData, PlayerData

# Position labels — single source of truth.
_POSITION_LABELS_ZH = {
    1: "1号位 (carry)",
    2: "2号位 (mid)",
    3: "3号位 (offlaner)",
    4: "4号位 (support)",
    5: "5号位 (hard support)",
}
_POSITION_LABELS_EN = {
    1: "Position 1 (Carry)",
    2: "Position 2 (Mid)",
    3: "Position 3 (Offlaner)",
    4: "Position 4 (Support)",
    5: "Position 5 (Hard Support)",
}

# Role tags used by the knowledge base / pro benchmarks <-> position number.
POSITION_TO_ROLE: dict[int, str] = {
    1: "carry",
    2: "mid",
    3: "offlane",
    4: "support_4",
    5: "support_5",
}
ROLE_TO_POSITION: dict[str, int] = {role: pos for pos, role in POSITION_TO_ROLE.items()}
# Loose aliases some data sources use.
ROLE_TO_POSITION.update({"support": 5, "hard_support": 5, "soft_support": 4, "roamer": 4})

# Relative lane roles (PlayerData.lane_role).
LANE_ROLE_SAFE = 1
LANE_ROLE_MID = 2
LANE_ROLE_OFF = 3
LANE_ROLE_ROAM = 4


@dataclass
class Position:
    """A derived Dota position for one player."""

    number: int           # 1..5; 0 if undecidable
    label_zh: str
    label_en: str
    confidence: str       # "high", "medium", "low"
    method: str           # "lane_nw", "archetype_nw", "nw_only", "unknown"

    def label(self, lang: str = "zh") -> str:
        return self.label_en if lang == "en" else self.label_zh

    @property
    def role(self) -> str:
        return POSITION_TO_ROLE.get(self.number, "")


def role_for_position(number: int | None) -> str:
    """Map a position number (1-5) to a role tag, "" when unknown."""
    try:
        return POSITION_TO_ROLE.get(int(number), "") if number is not None else ""
    except (TypeError, ValueError):
        return ""


def position_for_role(role: str | None) -> int:
    """Map a role tag to a position number (1-5), 0 when unknown."""
    if not role:
        return 0
    return ROLE_TO_POSITION.get(str(role).lower(), 0)


def _make(number: int, confidence: str, method: str) -> Position:
    return Position(
        number=number,
        label_zh=_POSITION_LABELS_ZH[number],
        label_en=_POSITION_LABELS_EN[number],
        confidence=confidence,
        method=method,
    )


def hero_typical_position(hero_name: str) -> int:
    """The hero's typical position from the knowledge base (0 if unknown).

    Imported lazily to avoid circular imports; never raises.
    """
    try:
        from .knowledge_base import get_hero_info

        info = get_hero_info(hero_name, lang="en") or {}
        return position_for_role(info.get("role", ""))
    except Exception:
        return 0


def lane_role_of(player: PlayerData) -> int:
    """Relative lane role (1=safe, 2=mid, 3=off, 4=roaming, 0=unknown).

    Uses ``lane_role`` when the source provided it, otherwise derives it from
    the absolute ``lane`` and the team.
    """
    if player.lane_role in (1, 2, 3, 4):
        return player.lane_role
    lane = player.lane
    if lane == 2:
        return LANE_ROLE_MID
    if lane == 4:
        return LANE_ROLE_ROAM
    if lane in (1, 3):
        safe_lane = 1 if player.team == "radiant" else 3
        return LANE_ROLE_SAFE if lane == safe_lane else LANE_ROLE_OFF
    return 0


def _farm_key(p: PlayerData) -> tuple[int, int, int]:
    """Farm ranking key: net worth, then GPM, then last hits."""
    return (p.net_worth or 0, p.gold_per_min or 0, p.last_hits or 0)


def derive_team_positions(team_players: list[PlayerData]) -> dict[int, Position]:
    """Compute positions 1-5 for one team (up to 5 players).

    Returns {player_slot: Position}. Each position number is used at most
    once; falls back gracefully when fewer than 5 players are present or when
    lane data is missing entirely.
    """
    if not team_players:
        return {}

    positions: dict[int, Position] = {}
    available = [1, 2, 3, 4, 5]

    def _take(p: PlayerData, number: int, confidence: str, method: str) -> None:
        positions[p.player_slot] = _make(number, confidence, method)
        available.remove(number)

    roles = {p.player_slot: lane_role_of(p) for p in team_players}

    # --- Step 1: lock in the mid player.
    mid_candidates = [p for p in team_players if roles[p.player_slot] == LANE_ROLE_MID]
    if mid_candidates:
        # Several players flagged mid (rare): prefer a mid archetype, then farm.
        mid_candidates.sort(
            key=lambda p: (hero_typical_position(p.hero_name) == 2, _farm_key(p)),
            reverse=True,
        )
        _take(mid_candidates[0], 2, "high", "lane_nw")

    # --- Step 2: safe lane -> core 1 + support 5, off lane -> core 3 + support 4.
    def _assign_lane(lane_role: int, core_pos: int, support_pos: int) -> None:
        group = [
            p for p in team_players
            if p.player_slot not in positions and roles[p.player_slot] == lane_role
        ]
        if not group:
            return
        group.sort(key=_farm_key, reverse=True)
        if len(group) == 1:
            # Lone player on a lane: most likely the core.
            _take(group[0], core_pos, "medium", "lane_nw")
            return
        _take(group[0], core_pos, "high", "lane_nw")
        # Poorest player of the lane is the hard-laning support; anyone in
        # between (3 players in one lane) is decided in step 3.
        _take(group[-1], support_pos, "high" if len(group) == 2 else "medium", "lane_nw")

    _assign_lane(LANE_ROLE_SAFE, core_pos=1, support_pos=5)
    _assign_lane(LANE_ROLE_OFF, core_pos=3, support_pos=4)

    # --- Step 3: leftovers (roamers, unknown lane, 3rd player in a lane; the
    # whole team when the match has no lane data). Split them into cores and
    # supports first: the open core positions go to the players ranked by
    # "usually played as a core", then farm. A plain farm ranking would turn
    # a support that out-farmed a losing core into the core, and assigning
    # the poorest first would push a rich core into position 5.
    rest = [p for p in team_players if p.player_slot not in positions]
    archetypes = {p.player_slot: hero_typical_position(p.hero_name) for p in rest}
    open_cores = [n for n in available if n <= 3]
    open_supports = [n for n in available if n >= 4]
    ranked = sorted(rest, key=lambda p: (archetypes[p.player_slot] in (1, 2, 3), _farm_key(p)), reverse=True)
    core_archetypes = sum(1 for p in rest if archetypes[p.player_slot] in (1, 2, 3))
    n_cores = min(len(open_cores), max(len(rest) - len(open_supports), core_archetypes))
    cores, supports = ranked[:n_cores], ranked[n_cores:]

    def _place(group: list[PlayerData], order: tuple[int, ...]) -> None:
        for p in sorted(group, key=_farm_key, reverse=True):
            archetype = archetypes[p.player_slot]
            if archetype in order and archetype in available:
                _take(p, archetype, "medium", "archetype_nw")
                continue
            for candidate in order:
                if candidate in available:
                    _take(p, candidate, "low", "nw_only")
                    break

    _place(cores, (1, 2, 3))
    _place(supports, (4, 5))
    # More supports than open support positions (incomplete teams only).
    _place([p for p in supports if p.player_slot not in positions], (3, 1, 2))

    return positions


def derive_match_positions(players: list[PlayerData]) -> dict[int, Position]:
    """Compute positions for all players in a match (both teams)."""
    radiant = [p for p in players if p.team == "radiant"]
    dire = [p for p in players if p.team == "dire"]
    out: dict[int, Position] = {}
    out.update(derive_team_positions(radiant))
    out.update(derive_team_positions(dire))
    return out


def player_position(match: MatchData, player_slot: int) -> Position | None:
    """Derived position of one player of ``match`` (None when not found)."""
    return derive_match_positions(match.players).get(player_slot)


def position_number_label(number: object, lang: str = "zh") -> str:
    """Localized label of a position number (1-5); "" for anything else."""
    labels = _POSITION_LABELS_EN if lang == "en" else _POSITION_LABELS_ZH
    return labels.get(number, "") if isinstance(number, int) else ""


def position_label(position: Position | None, lang: str = "zh", fallback: str = "") -> str:
    """Render a Position to a localized label, with safe fallback."""
    if position is None or position.number == 0:
        return fallback
    return position.label(lang)
