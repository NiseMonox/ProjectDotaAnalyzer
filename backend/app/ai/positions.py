"""Derive Dota 2 positions (1-5) from match data.

Dota positions are role assignments, not lanes:
  Position 1 (carry)         — safe-lane core, takes most farm
  Position 2 (mid)           — solo mid lane
  Position 3 (offlaner)      — off-lane core
  Position 4 (support)       — soft support, often roams
  Position 5 (hard support)  — hard support, usually with the carry

We derive each player's position by combining three signals, in order of
trustworthiness:

1. **Lane** — OpenDota's parsed lane (1=safe-side bot, 2=mid, 3=off-side top
   *from Radiant's perspective*; flipped for Dire). lane=4/0 means the parser
   couldn't fit them into a single lane (almost always a roaming support in
   modern Dota; "Jungle" as a real position is essentially extinct post-7.20).
2. **Net worth ranking within lane pair** — the safe lane has pos 1 + pos 5;
   the off lane has pos 3 + pos 4. The higher-net-worth player is the core.
3. **Hero archetype** from `knowledge_base.HERO_DATA["role"]` — used as a
   tiebreaker for unclassified players (lane=4 / lane=0).

The resulting Position tuple is what UI labels and LLM prompts should use,
not raw lane numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..models import PlayerData

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

# Map knowledge_base role tags -> typical position number.
_ROLE_TO_POS = {
    "carry": 1,
    "mid": 2,
    "offlane": 3,
    "support_4": 4,
    "support_5": 5,
}


@dataclass
class Position:
    """A derived Dota position for one player."""

    number: int           # 1..5; 0 if undecidable
    label_zh: str
    label_en: str
    confidence: str       # "high", "medium", "low"
    method: str           # "lane+nw", "archetype+nw", "nw_only", "unknown"

    def label(self, lang: str = "zh") -> str:
        return self.label_en if lang == "en" else self.label_zh


def _hero_typical_position(hero_name: str) -> int:
    """Look up the hero's typical position from the knowledge base.

    Returns 0 if unknown. Imported lazily to avoid circular imports.
    """
    try:
        from .knowledge_base import get_hero_info

        info = get_hero_info(hero_name, lang="en")
        return _ROLE_TO_POS.get(info.get("role", "unknown"), 0)
    except Exception:
        return 0


def _is_safe_lane(player: PlayerData) -> bool:
    """Map (lane, team) -> is-safe-lane.

    Radiant safe = bottom (lane 1); Dire safe = top (lane 3).
    """
    if player.team == "radiant":
        return player.lane == 1
    return player.lane == 3


def _is_off_lane(player: PlayerData) -> bool:
    if player.team == "radiant":
        return player.lane == 3
    return player.lane == 1


def _is_mid(player: PlayerData) -> bool:
    return player.lane == 2 or player.lane_role == 2


def derive_team_positions(team_players: list[PlayerData]) -> dict[int, Position]:
    """Compute positions 1-5 for one team (5 players).

    Returns {player_slot: Position}. Falls back gracefully when fewer than 5
    players are present or when lane data is missing entirely.
    """
    if not team_players:
        return {}

    positions: dict[int, Position] = {}
    nw_sorted = sorted(team_players, key=lambda p: p.net_worth, reverse=True)

    # --- Step 1: lock in the mid player.
    mid_candidates = [p for p in team_players if _is_mid(p)]
    mid_player: PlayerData | None = None
    if mid_candidates:
        # If multiple players are flagged as mid (rare), pick the one whose
        # archetype is "mid", else the one with higher net worth.
        mid_candidates.sort(
            key=lambda p: (_hero_typical_position(p.hero_name) == 2, p.net_worth),
            reverse=True,
        )
        mid_player = mid_candidates[0]
        positions[mid_player.player_slot] = Position(
            number=2,
            label_zh=_POSITION_LABELS_ZH[2],
            label_en=_POSITION_LABELS_EN[2],
            confidence="high",
            method="lane+nw",
        )

    remaining = [p for p in team_players if p.player_slot not in positions]

    # --- Step 2: split remaining 4 into safe-lane vs off-lane vs unclassified.
    safe = [p for p in remaining if _is_safe_lane(p)]
    off = [p for p in remaining if _is_off_lane(p)]
    rest = [p for p in remaining if p.player_slot not in {q.player_slot for q in safe + off}]

    def _assign_pair(pair: list[PlayerData], core_pos: int, support_pos: int):
        if not pair:
            return
        if len(pair) == 1:
            # Lone player on a lane — likely the core; complement goes to rest.
            p = pair[0]
            positions[p.player_slot] = Position(
                number=core_pos,
                label_zh=_POSITION_LABELS_ZH[core_pos],
                label_en=_POSITION_LABELS_EN[core_pos],
                confidence="medium",
                method="lane+nw",
            )
            return
        pair.sort(key=lambda p: p.net_worth, reverse=True)
        positions[pair[0].player_slot] = Position(
            number=core_pos,
            label_zh=_POSITION_LABELS_ZH[core_pos],
            label_en=_POSITION_LABELS_EN[core_pos],
            confidence="high",
            method="lane+nw",
        )
        for support_player in pair[1:]:
            positions[support_player.player_slot] = Position(
                number=support_pos,
                label_zh=_POSITION_LABELS_ZH[support_pos],
                label_en=_POSITION_LABELS_EN[support_pos],
                confidence="high",
                method="lane+nw",
            )

    _assign_pair(safe, core_pos=1, support_pos=5)
    _assign_pair(off, core_pos=3, support_pos=4)

    # --- Step 3: the leftovers (lane=0 or lane=4 — almost always supports
    # in modern Dota). Use hero archetype + remaining net-worth gaps to
    # decide pos 4 vs pos 5.
    used_positions = {pos.number for pos in positions.values()}
    available_positions = [p for p in (1, 2, 3, 4, 5) if p not in used_positions]

    for p in sorted(rest, key=lambda x: x.net_worth):
        archetype = _hero_typical_position(p.hero_name)
        chosen: int = 0
        method = "archetype+nw"

        # First try: archetype's typical position is still available.
        if archetype in available_positions:
            chosen = archetype
        # Else: pick whichever support slot is open, preferring 5 if the player
        # has the lowest net worth on the team (classic hard support pattern).
        else:
            for candidate in (5, 4, 3, 2, 1):
                if candidate in available_positions:
                    chosen = candidate
                    method = "nw_only"
                    break

        if chosen == 0:
            continue
        available_positions.remove(chosen)
        positions[p.player_slot] = Position(
            number=chosen,
            label_zh=_POSITION_LABELS_ZH[chosen],
            label_en=_POSITION_LABELS_EN[chosen],
            confidence="medium" if archetype == chosen else "low",
            method=method,
        )

    # --- Step 4: any player still unclassified gets pure net-worth ranking
    # against the still-open slots. This handles lane=0/4 with no archetype.
    for p in nw_sorted:
        if p.player_slot in positions:
            continue
        if not available_positions:
            break
        chosen = available_positions[0]
        available_positions.pop(0)
        positions[p.player_slot] = Position(
            number=chosen,
            label_zh=_POSITION_LABELS_ZH[chosen],
            label_en=_POSITION_LABELS_EN[chosen],
            confidence="low",
            method="nw_only",
        )

    return positions


def derive_match_positions(players: list[PlayerData]) -> dict[int, Position]:
    """Compute positions for all players in a match (both teams)."""
    radiant = [p for p in players if p.team == "radiant"]
    dire = [p for p in players if p.team == "dire"]
    out = {}
    out.update(derive_team_positions(radiant))
    out.update(derive_team_positions(dire))
    return out


def position_label(position: Position | None, lang: str = "zh", fallback: str = "") -> str:
    """Render a Position to a localized label, with safe fallback."""
    if position is None or position.number == 0:
        return fallback
    return position.label(lang)
