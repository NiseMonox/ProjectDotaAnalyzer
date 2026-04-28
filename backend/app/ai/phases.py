"""Slice a Dota match into game-time phases and aggregate per-phase stats.

Phases:
  laning      0–10 min     pre-fight, lane setup, last-hits
  early_mid   10–20 min    rotations, first towers, key item spikes
  mid_late    20–30 min    teamfights, Roshan, map control
  late        30–40 min    high-ground attempts, big buybacks
  ultra_late  40+ min      throne plays, end-game decisions

For each phase that the match actually reached, we extract the player's
KDA in the window, net-worth/level delta, gold/xp per minute in that
window, deaths in window, key teamfights, and a brief synopsis of the
team-level state (gold lead, towers).

This module is pure data extraction — no LLM calls, no rendering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..models import MatchData, PlayerData

# Phase definitions: (key, start_seconds, end_seconds_or_None_for_open_ended)
PHASE_BOUNDS: list[tuple[str, int, int | None]] = [
    ("laning",     0,    10 * 60),
    ("early_mid",  10*60, 20*60),
    ("mid_late",   20*60, 30*60),
    ("late",       30*60, 40*60),
    ("ultra_late", 40*60, None),  # open-ended
]


@dataclass
class PhaseStats:
    """Aggregated stats for one player in one phase window."""

    phase: str
    time_range: str           # human-friendly, e.g. "0-10 min" or "40+ min"
    start_sec: int
    end_sec: int              # inclusive upper bound (clamped to match duration)
    duration_sec: int

    # Per-player numbers in this window.
    kills: int = 0
    deaths: int = 0
    assists: int = 0          # rough — counts assists where ally killed during window
    last_hits_delta: int = 0
    gold_delta: int = 0
    xp_delta: int = 0
    gpm_in_phase: int = 0
    xpm_in_phase: int = 0
    level_at_end: int = 0
    net_worth_at_end: int = 0

    # Key events in this window.
    death_times: list[int] = field(default_factory=list)
    teamfight_count: int = 0
    teamfight_wins: int = 0    # this team's gold delta > 0 during the fight
    towers_taken_radiant: int = 0
    towers_taken_dire: int = 0
    smokes_used_self_team: int = 0

    # Match-level state at end of window.
    radiant_gold_lead: int = 0      # +ve = radiant ahead
    radiant_xp_lead: int = 0

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "time_range": self.time_range,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "duration_sec": self.duration_sec,
            "kills": self.kills,
            "deaths": self.deaths,
            "assists": self.assists,
            "last_hits_delta": self.last_hits_delta,
            "gold_delta": self.gold_delta,
            "xp_delta": self.xp_delta,
            "gpm_in_phase": self.gpm_in_phase,
            "xpm_in_phase": self.xpm_in_phase,
            "level_at_end": self.level_at_end,
            "net_worth_at_end": self.net_worth_at_end,
            "death_times": self.death_times,
            "teamfight_count": self.teamfight_count,
            "teamfight_wins": self.teamfight_wins,
            "towers_taken_radiant": self.towers_taken_radiant,
            "towers_taken_dire": self.towers_taken_dire,
            "smokes_used_self_team": self.smokes_used_self_team,
            "radiant_gold_lead": self.radiant_gold_lead,
            "radiant_xp_lead": self.radiant_xp_lead,
        }


def _clamp_minute(m: int, series: list[int] | None) -> int:
    """Read series[m] safely, returning 0 if out of range."""
    if not series or m < 0 or m >= len(series):
        return 0
    return series[m]


def _format_range(start_sec: int, end_sec: int, open_ended: bool) -> str:
    start_min = start_sec // 60
    end_min = end_sec // 60
    return f"{start_min}+ min" if open_ended else f"{start_min}-{end_min} min"


def _team_kills_at(match: MatchData, slot: int, t: int) -> bool:
    """Approximate: did this player's team get a kill near time t?"""
    return False  # unused; handled via teamfight + objectives below


def applicable_phases(match_duration_sec: int) -> list[tuple[str, int, int | None]]:
    """Return only the phases this match actually reached."""
    out: list[tuple[str, int, int | None]] = []
    for key, start, end in PHASE_BOUNDS:
        if match_duration_sec <= start:
            break
        out.append((key, start, end))
    return out


def slice_player_phases(match: MatchData, player: PlayerData) -> list[PhaseStats]:
    """Build a PhaseStats per applicable phase for one player.

    Uses per-minute timeline series (gold_t / xp_t / lh_t) for delta math, and
    raw event lists (objectives, teamfights, smokes) for situational counts.
    """
    duration = match.duration or 0
    out: list[PhaseStats] = []

    # Pre-index helpful lookups.
    obj_events = match.objectives or []
    teamfights = match.teamfights or []
    smokes = match.smokes or []

    for key, start, end_or_none in applicable_phases(duration):
        open_ended = end_or_none is None
        end = duration if open_ended else min(end_or_none, duration)
        if end <= start:
            continue

        start_min = start // 60
        end_min = end // 60  # int division; level/net-worth read at end_min

        gold_start = _clamp_minute(start_min, player.gold_t)
        gold_end = _clamp_minute(end_min, player.gold_t)
        xp_start = _clamp_minute(start_min, player.xp_t)
        xp_end = _clamp_minute(end_min, player.xp_t)
        lh_start = _clamp_minute(start_min, player.lh_t)
        lh_end = _clamp_minute(end_min, player.lh_t)

        gold_delta = gold_end - gold_start
        xp_delta = xp_end - xp_start
        lh_delta = lh_end - lh_start
        minutes = max(1, end_min - start_min)

        # Approximate level at end-of-phase: use xp_t curve to back-compute
        # is overkill; fall back to player.level when phase covers end of game.
        level_at_end = player.level if open_ended or end >= duration else 0

        # Deaths in window — derive from team kills counter? we don't have a
        # per-second death log on PlayerData, so use teamfights as proxy plus
        # per-minute stuns/buyback markers if needed. Simpler: scan
        # `match.objectives` for CHAT_MESSAGE_HERO_KILL with this slot as
        # target (OpenDota emits these).
        deaths_here: list[int] = []
        for obj in obj_events:
            t = obj.get("time", 0)
            if not (start <= t < end):
                continue
            if obj.get("type") in ("CHAT_MESSAGE_HERO_KILL", "hero_kill"):
                # This event format varies; some OpenDota dumps put the
                # killed hero in "key", and the killer slot in "slot".
                target_slot = obj.get("target_slot")
                if target_slot == player.player_slot:
                    deaths_here.append(t)

        # Kills in window — use teamfights (most reliable structured data).
        kills_here = 0
        assists_here = 0
        teamfight_count = 0
        teamfight_wins = 0
        for tf in teamfights:
            tf_start = tf.start_time
            tf_end = tf.end_time
            if tf_end <= start or tf_start >= end:
                continue
            teamfight_count += 1
            tf_player = next(
                (tp for tp in tf.players if tp.player_slot == player.player_slot),
                None,
            )
            if tf_player:
                kills_here += tf_player.kills
            # team-level result: positive gold_delta means radiant won the fight
            radiant_won = tf.gold_delta > 0
            if (radiant_won and player.team == "radiant") or (
                not radiant_won and player.team == "dire"
            ):
                teamfight_wins += 1
            # Crude assist proxy: any teammate had kills in this fight while
            # the player was present and not dead -> +1 assist
            if tf_player and tf_player.kills == 0 and tf_player.deaths == 0:
                ally_kills = sum(
                    tp.kills for tp in tf.players
                    if tp.player_slot != player.player_slot
                    and ((player.team == "radiant" and tp.player_slot < 128) or
                         (player.team == "dire" and tp.player_slot >= 128))
                )
                assists_here += min(ally_kills, 2)  # cap to avoid runaway

        # Towers taken in window (per side).
        towers_radiant = 0
        towers_dire = 0
        for obj in obj_events:
            t = obj.get("time", 0)
            if not (start <= t < end):
                continue
            otype = obj.get("type", "")
            if otype not in ("CHAT_MESSAGE_TOWER_KILL", "building_kill", "tower_kill"):
                continue
            # Tower belongs to the team whose unit is being killed; the
            # killer side is the *other* one.
            unit = (obj.get("key") or obj.get("unit") or "").lower()
            if "goodguys" in unit or "radiant" in unit:
                towers_dire += 1  # dire killed a radiant tower
            elif "badguys" in unit or "dire" in unit:
                towers_radiant += 1

        # Smokes used by player's team in window.
        smoke_count_self = sum(
            1 for s in smokes
            if start <= s.time < end and s.team == player.team
        )

        # Gold/XP lead at end of window.
        radiant_gold_lead = _clamp_minute(end_min, match.radiant_gold_adv)
        radiant_xp_lead = _clamp_minute(end_min, match.radiant_xp_adv)

        out.append(PhaseStats(
            phase=key,
            time_range=_format_range(start, end, open_ended),
            start_sec=start,
            end_sec=end,
            duration_sec=end - start,
            kills=kills_here,
            deaths=len(deaths_here),
            assists=assists_here,
            last_hits_delta=lh_delta,
            gold_delta=gold_delta,
            xp_delta=xp_delta,
            gpm_in_phase=int(gold_delta / minutes),
            xpm_in_phase=int(xp_delta / minutes),
            level_at_end=level_at_end,
            net_worth_at_end=player.net_worth if open_ended or end >= duration else 0,
            death_times=deaths_here,
            teamfight_count=teamfight_count,
            teamfight_wins=teamfight_wins,
            towers_taken_radiant=towers_radiant,
            towers_taken_dire=towers_dire,
            smokes_used_self_team=smoke_count_self,
            radiant_gold_lead=radiant_gold_lead,
            radiant_xp_lead=radiant_xp_lead,
        ))

    return out
