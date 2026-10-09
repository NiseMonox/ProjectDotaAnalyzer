"""Game-phase helpers and per-phase stat aggregation.

Single source of truth for phase boundaries used across the AI pipeline:

* :func:`game_phase` classifies one moment (decision points, contexts,
  knowledge retrieval): ``pregame`` (< 0:30, rune contest), ``laning``
  (< 10:00), ``mid_game`` (< 25:00), ``late_game``. All upper bounds are
  exclusive.
* :data:`PHASE_BOUNDS` are the 10-minute review windows used for the
  per-phase reviews (same 10:00 laning boundary):

  laning      0–10 min     lane setup, last-hits
  early_mid   10–20 min    rotations, first towers, key item spikes
  mid_late    20–30 min    teamfights, Roshan, map control
  late        30–40 min    high-ground attempts, big buybacks
  ultra_late  40+ min      throne plays, end-game decisions

For each phase that the match actually reached, :func:`slice_player_phases`
extracts the player's KDA in the window, gold/xp per minute in that window,
deaths, teamfights the player took part in, and the team-level state at the
end of the window (gold/xp lead, towers).

This module is pure data extraction — no LLM calls, no rendering.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Iterable

from ..models import MatchData, PlayerData

# ---------------------------------------------------------------------------
# Moment classification (shared by detector, context builder, prompts)
# ---------------------------------------------------------------------------

PREGAME_END = 30          # rune contest / first creep wave
LANING_END = 10 * 60
MID_GAME_END = 25 * 60

GAME_PHASES: tuple[str, ...] = ("pregame", "laning", "mid_game", "late_game")


def game_phase(time_s: int) -> str:
    """Classify a game second into pregame / laning / mid_game / late_game."""
    if time_s < PREGAME_END:
        return "pregame"
    if time_s < LANING_END:
        return "laning"
    if time_s < MID_GAME_END:
        return "mid_game"
    return "late_game"


def is_laning(time_s: int) -> bool:
    """Whether a game second falls in the laning stage (incl. pregame)."""
    return time_s < LANING_END


# Cumulative XP needed per level (Dota 2, 7.3x). Index i = XP for level i+1.
_XP_TABLE: tuple[int, ...] = (
    0, 240, 640, 1160, 1760, 2440, 3200, 4000, 4900, 5900,          # 1-10
    7000, 8200, 9500, 10900, 12400, 14000, 15700, 17500, 19400,     # 11-19
    21400, 23600, 26000, 28600, 31400, 34400, 38400, 43400,         # 20-27
    49400, 56400, 63900,                                            # 28-30
)


def level_from_xp(xp: int) -> int:
    """Estimate the hero level from total XP (1..30)."""
    if xp <= 0:
        return 1
    return max(1, min(30, bisect.bisect_right(_XP_TABLE, xp)))


# ---------------------------------------------------------------------------
# Review windows
# ---------------------------------------------------------------------------

# (key, start_seconds, end_seconds_or_None_for_open_ended)
PHASE_BOUNDS: list[tuple[str, int, int | None]] = [
    ("laning",     0,            LANING_END),
    ("early_mid",  LANING_END,   20 * 60),
    ("mid_late",   20 * 60,      30 * 60),
    ("late",       30 * 60,      40 * 60),
    ("ultra_late", 40 * 60,      None),  # open-ended
]

# A trailing phase shorter than this is merged into the previous one: a
# 30-second "early_mid" window is noise and would waste an LLM call.
_MIN_PHASE_SECONDS = 180


@dataclass
class PhaseStats:
    """Aggregated stats for one player in one phase window."""

    phase: str
    time_range: str           # e.g. "0-10 min" or "40+ min"
    start_sec: int
    end_sec: int              # exclusive upper bound (clamped to match duration)
    duration_sec: int

    # Per-player numbers in this window.
    kills: int = 0
    deaths: int = 0
    assists: int = 0          # rough proxy from teamfights the player took part in
    last_hits_delta: int = 0
    gold_delta: int = 0
    xp_delta: int = 0
    gpm_in_phase: int = 0
    xpm_in_phase: int = 0
    level_at_end: int = 0
    net_worth_at_end: int = 0

    # Key events in this window.
    death_times: list[int] = field(default_factory=list)
    teamfight_count: int = 0   # teamfights the player took part in
    teamfight_wins: int = 0    # ...of which the player's team won
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


def series_at(series: list[int] | None, minute: int) -> int:
    """Read a per-minute series at ``minute``, clamped to the series range.

    Past the end the last value is returned (series usually stop at the
    final minute); empty series read as 0.
    """
    if not series:
        return 0
    if minute <= 0:
        return series[0]
    return series[min(minute, len(series) - 1)]


def _format_range(start_sec: int, end_sec: int, open_ended: bool) -> str:
    start_min = start_sec // 60
    end_min = end_sec // 60
    return f"{start_min}+ min" if open_ended else f"{start_min}-{end_min} min"


def applicable_phases(match_duration_sec: int) -> list[tuple[str, int, int | None]]:
    """Return the review windows this match reached.

    A trailing window shorter than ``_MIN_PHASE_SECONDS`` is folded into the
    previous one (whose end then becomes open-ended, i.e. the match end).
    """
    out: list[tuple[str, int, int | None]] = []
    for key, start, end in PHASE_BOUNDS:
        if match_duration_sec <= start:
            break
        out.append((key, start, end))
    if len(out) >= 2:
        key, start, _end = out[-1]
        if match_duration_sec - start < _MIN_PHASE_SECONDS:
            out.pop()
            prev_key, prev_start, _ = out[-1]
            out[-1] = (prev_key, prev_start, None)
    return out


def _tower_owner(unit: str) -> str:
    """Team owning a destroyed tower unit name, "" when not a tower."""
    unit = unit.lower()
    if "tower" not in unit:
        return ""
    if "goodguys" in unit or "radiant" in unit:
        return "radiant"
    if "badguys" in unit or "dire" in unit:
        return "dire"
    return ""


def _participated(tp) -> bool:
    return bool(tp.damage_dealt or tp.damage_taken or tp.kills or tp.deaths or tp.healing)


def _radiant_won_fight(tf) -> bool | None:
    """True/False when the fight had a clear winner, None for an even trade."""
    if tf.gold_delta:
        return tf.gold_delta > 0
    if tf.radiant_deaths != tf.dire_deaths:
        return tf.radiant_deaths < tf.dire_deaths
    return None


def slice_player_phases(
    match: MatchData,
    player: PlayerData,
    death_times: Iterable[int] | None = None,
) -> list[PhaseStats]:
    """Build a PhaseStats per applicable phase for one player.

    Uses per-minute timeline series (gold_t / xp_t / lh_t) for delta math, and
    raw event lists (objectives, teamfights, smokes) for situational counts.

    Parameters
    ----------
    death_times : Iterable[int] | None
        Optional precise death timestamps (seconds) for this player, e.g. from
        the decision detector's DeathEvents. When provided they are used for
        per-phase death counting; otherwise falls back to scanning
        ``match.objectives`` (whose format varies by dump and often lacks
        ``target_slot``).
    """
    duration = match.duration or 0
    out: list[PhaseStats] = []

    # Normalize once so a generator input is not exhausted mid-loop.
    known_death_times = sorted(death_times) if death_times is not None else None

    obj_events = match.objectives or []
    teamfights = match.teamfights or []
    smokes = match.smokes or []
    is_radiant = player.team == "radiant"

    for key, start, end_or_none in applicable_phases(duration):
        open_ended = end_or_none is None
        end = duration if open_ended else min(end_or_none, duration)
        if end <= start:
            continue
        last_window = open_ended or end >= duration

        start_min = start // 60
        end_min = end // 60

        gold_delta = series_at(player.gold_t, end_min) - series_at(player.gold_t, start_min)
        xp_delta = series_at(player.xp_t, end_min) - series_at(player.xp_t, start_min)
        lh_delta = series_at(player.lh_t, end_min) - series_at(player.lh_t, start_min)
        minutes = max(1, end_min - start_min)

        if last_window and player.level:
            level_at_end = player.level
        elif player.xp_t:
            level_at_end = level_from_xp(series_at(player.xp_t, end_min))
        else:
            level_at_end = 0

        def _in_window(t: int) -> bool:
            return start <= t < end or (last_window and t == end)

        deaths_here: list[int] = []
        if known_death_times is not None:
            deaths_here = [dt for dt in known_death_times if _in_window(dt)]
        else:
            for obj in obj_events:
                t = obj.get("time", 0)
                if not _in_window(t):
                    continue
                if obj.get("type") in ("CHAT_MESSAGE_HERO_KILL", "hero_kill"):
                    if obj.get("target_slot") == player.player_slot:
                        deaths_here.append(t)

        # Kills / teamfights the player took part in (most reliable structured data).
        kills_here = 0
        assists_here = 0
        teamfight_count = 0
        teamfight_wins = 0
        for tf in teamfights:
            if tf.end_time <= start or tf.start_time >= end:
                continue
            tf_player = next(
                (tp for tp in tf.players if tp.player_slot == player.player_slot),
                None,
            )
            if tf_player is None or not _participated(tf_player):
                continue
            teamfight_count += 1
            kills_here += tf_player.kills
            radiant_won = _radiant_won_fight(tf)
            if radiant_won is not None and radiant_won == is_radiant:
                teamfight_wins += 1
            # Crude assist proxy: teammates got kills while the player took
            # part without a kill or death of their own.
            if tf_player.kills == 0 and tf_player.deaths == 0:
                ally_kills = sum(
                    tp.kills for tp in tf.players
                    if tp.player_slot != player.player_slot
                    and (tp.player_slot < 128) == is_radiant
                )
                assists_here += min(ally_kills, 2)  # cap to avoid runaway

        # Towers taken in window (per side). "building_kill" also covers
        # barracks/fort, so only count units that are towers.
        towers_radiant = 0
        towers_dire = 0
        for obj in obj_events:
            t = obj.get("time", 0)
            if not (start <= t < end):
                continue
            if obj.get("type", "") not in ("CHAT_MESSAGE_TOWER_KILL", "building_kill", "tower_kill"):
                continue
            owner = _tower_owner(str(obj.get("key") or obj.get("unit") or ""))
            if owner == "radiant":
                towers_dire += 1      # dire destroyed a radiant tower
            elif owner == "dire":
                towers_radiant += 1

        smoke_count_self = sum(
            1 for s in smokes
            if start <= s.time < end and s.team == player.team
        )

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
            net_worth_at_end=player.net_worth if last_window else 0,
            death_times=deaths_here,
            teamfight_count=teamfight_count,
            teamfight_wins=teamfight_wins,
            towers_taken_radiant=towers_radiant,
            towers_taken_dire=towers_dire,
            smokes_used_self_team=smoke_count_self,
            radiant_gold_lead=series_at(match.radiant_gold_adv, end_min),
            radiant_xp_lead=series_at(match.radiant_xp_adv, end_min),
        ))

    return out
