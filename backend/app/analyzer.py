"""Analysis modules for Dota 2 match data.

Provides higher-level analysis on top of parsed match data:
- Item build analysis with timing benchmarks
- Skill build analysis
- Ward coverage analysis
- Smoke effectiveness analysis
- Teamfight analysis
- Ability usage patterns

Enumerations in the payload are stable lowercase codes (lane names, gold /
XP reasons, rune types, ...); the frontend translates them. Positions are
passed through unchanged in world units (see ``geo.py``). Deaths, killers
and the running score come from ``MatchData.kills_log`` when the source has
one. ``PlayerData.position_log`` is deliberately not part of the payload.
"""
from __future__ import annotations

import bisect
import logging
import math

from . import geo
from .game_constants import OBSERVER_WARD_DURATION_S, SENTRY_WARD_DURATION_S
from .models import KillEvent, MatchData, PlayerData, SkillEvent, TeamfightPlayer

logger = logging.getLogger(__name__)

# Natural ward lifetimes (seconds), used for wards without a recorded end
# when counting the wards active in a time window.
OBSERVER_LIFETIME_S = OBSERVER_WARD_DURATION_S
SENTRY_LIFETIME_S = SENTRY_WARD_DURATION_S

# OpenDota's lane efficiency: net worth at 10:00 relative to the gold one
# lane yields by then (OpenDota's constant).
LANE_GOLD_AT_TEN = 4948
# The constant describes the standard game. Turbo doubles the gold: every
# core of a Turbo replay read 130-190 % "of the ideal lane income", so the
# figure is not given for that mode (whoever computed it, OpenDota included).
LANE_EFFICIENCY_NA_MODES = frozenset({"turbo"})

# Shorter than any respawn time (12 s at level 1): a hero that "dies" again
# this soon was alive in between, so the first entry was a reincarnation
# (Wraith King, Aegis). Only an instant buyback into another death beats it.
REINCARNATION_WINDOW_S = 10

# PlayerData.lane_role (relative, OpenDota convention) → lane code.
# lane_role=4 is OpenDota's "jungle" bucket; in modern Dota that is mostly
# roaming supports / pushed-out offlaners, so it maps to "roaming".
LANE_ROLE_CODES = {1: "safe", 2: "mid", 3: "off", 4: "roaming"}
# PlayerData.lane (absolute map lane, OpenDota convention) → side code.
LANE_SIDE_CODES = {1: "bottom", 2: "mid", 3: "top", 4: "jungle", 5: "jungle"}

# Source 2 EDOTA_ModifyGold_Reason (same numbering in OpenDota and Clarity).
# The published enum ends at 21. Reason 22 is named from replays of patch
# 7.41f: every such gain follows the receiving hero's killing blow on an
# illusion (63 of 63 events in one match), so it is the illusion bounty.
GOLD_REASON_CODES = {
    "0": "other", "1": "death", "2": "buyback", "3": "purchase", "4": "purchase",
    "5": "abandon", "6": "sell", "7": "ability_cost", "8": "cheat",
    "9": "selection_penalty", "10": "passive", "11": "building", "12": "hero",
    "13": "creep", "14": "neutral", "15": "roshan", "16": "courier",
    "17": "bounty_rune", "18": "shared", "19": "ability", "20": "ward",
    "21": "courier", "22": "illusion",
}
# Source 2 EDOTA_ModifyXP_Reason. The game never renamed reason 4 (it is
# still "TomeOfKnowledge" in the enum) but reuses it: wisdom-rune XP from
# 7.33, Shrine of Wisdom XP since 7.38. The code stays ``wisdom_rune`` for
# both; the label is the frontend's (by match patch).
XP_REASON_CODES = {
    "0": "other", "1": "hero", "2": "creep", "3": "roshan",
    "4": "wisdom_rune", "5": "outpost", "6": "catch_up", "7": "ability",
}
# DOTA_RUNES enum (OpenDota runes / runes_log keys). 8 is DOTA_RUNE_XP: a
# wisdom rune pickup until 7.37, a Shrine of Wisdom capture since 7.38.
RUNE_CODES = {
    "0": "double_damage", "1": "haste", "2": "illusion", "3": "invisibility",
    "4": "regeneration", "5": "bounty", "6": "arcane", "7": "water",
    "8": "wisdom", "9": "shield",
}


def format_time(seconds: int | float | None) -> str:
    """Convert seconds to MM:SS format (negative times get a leading '-')."""
    seconds = int(seconds or 0)
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    return f"{sign}{seconds // 60}:{seconds % 60:02d}"


def _code_for(key, table: dict[str, str]) -> str:
    """Map a numeric enum key to its code; unknown keys become unknown_<key>."""
    key = str(key)
    return table.get(key) or f"unknown_{key}"


def _amounts_by_code(amounts: dict, table: dict[str, str]) -> dict[str, int]:
    """Gold / XP per reason code (several enum values can share one code).

    Reasons that add up to nothing are left out: replays also list reasons
    with an amount of 0, which would only show as empty chart series.
    """
    totals: dict[str, int] = {}
    for key, amount in amounts.items():
        code = _code_for(key, table)
        totals[code] = totals.get(code, 0) + amount
    return {code: total for code, total in totals.items() if total}


def _slot_map(match: MatchData) -> dict[int, PlayerData]:
    return {p.player_slot: p for p in match.players}


def _team_for_slot(slot: int | None, slot_map: dict[int, PlayerData]) -> str | None:
    """Team of a player slot ("radiant"/"dire"), or None when unknown."""
    if slot is None:
        return None
    player = slot_map.get(slot)
    if player is not None and player.team in ("radiant", "dire"):
        return player.team
    if 0 <= slot < 128:
        return "radiant"
    if 128 <= slot < 256:
        return "dire"
    return None


def _resolve_event_slot(entry: dict, match: MatchData, slot_map: dict[int, PlayerData]) -> int | None:
    """Player slot of an OpenDota objective/chat entry.

    ``player_slot`` is the real slot (0-4 / 128-132); ``slot`` is the
    player *index* 0-9, so it must not be compared against 128.
    """
    slot = entry.get("player_slot")
    if isinstance(slot, int) and not isinstance(slot, bool):
        return slot
    index = entry.get("slot")
    if not isinstance(index, int) or isinstance(index, bool):
        return None
    ordered = sorted(match.players, key=lambda p: p.player_slot)
    if 0 <= index < len(ordered):
        return ordered[index].player_slot
    return None


def _hero_fields(player: PlayerData | None, prefix: str) -> dict:
    """``<prefix>hero_name`` / ``<prefix>hero_slug`` of a player ("" when unknown)."""
    return {
        f"{prefix}hero_name": player.hero_name if player else "",
        f"{prefix}hero_slug": player.hero_slug if player else "",
    }


def _pre_game_only(match: MatchData) -> bool:
    """Whether a truncated replay ends before the horn.

    Such a match has no game time at all: duration 0, every event time
    <= 0, no winner. Sections then only hold what happened in the pre-game.
    """
    return match.partial and match.duration <= 0


# A truncated replay that ends this early in the game gets no coaching report
# (the same bound as ai.coach.MIN_PARTIAL_GAME_SECONDS; a test ties the two).
COACH_MIN_PARTIAL_SECONDS = 300


def _coach_available(match: MatchData) -> bool:
    """Whether the match holds enough of a game for a coaching report.

    False for a truncated replay that ends before the horn or within the
    first minutes after it: nothing in it can be reviewed or compared.
    """
    return not (match.partial and match.duration < COACH_MIN_PARTIAL_SECONDS)


def _player_positions(match: MatchData) -> dict[int, dict]:
    """Position (1-5) of every player: ``{player_slot: {number, confidence,
    method}}`` with ``confidence`` high | medium | low and ``method``
    lane_nw | archetype_nw | nw_only.

    This is the AI layer's own derivation (``ai.positions``), so the
    scoreboard names the position a coach report names. A player it could
    not place is missing. Empty for a replay the coach does not review
    (``_coach_available``: it ends before the horn or in the first minutes,
    when lanes and farm say nothing yet and the roles could only be guessed
    from the heroes) and when the AI layer is unavailable; it never fails
    the analysis.
    """
    if not _coach_available(match):
        return {}
    try:
        from .ai.positions import match_positions_payload

        derived = match_positions_payload(match)
    except Exception as exc:
        logger.warning("Player positions are left out of the analysis: %s", exc)
        return {}
    positions: dict[int, dict] = {}
    for slot, position in derived.items() if isinstance(derived, dict) else ():
        number = position.get("number") if isinstance(position, dict) else None
        if isinstance(number, int) and not isinstance(number, bool) and 1 <= number <= 5:
            positions[slot] = {
                "number": number,
                "confidence": str(position.get("confidence") or ""),
                "method": str(position.get("method") or ""),
            }
    return positions


def _patch_versions(match: MatchData) -> tuple[str, str]:
    """Game version of the match as ``(major, label)``: ("7.41", "7.41b").

    ``major`` is the source's own label when it has one (OpenDota, mock); a
    replay only knows when it was played, which the data layer's patch
    table maps to the version. ``label`` is the full version with its
    letter patch, from Valve's release list by the same start time. That
    list is only trusted where it agrees with the major version (on a
    release day the two sources can be hours apart); otherwise, and without
    the list, ``label`` is the major version. Both are "" when unknown.
    """
    try:
        from .ai.data_manager import patch_for_time, patch_label_for_time, patch_major
    except ImportError:  # data layer unavailable
        return match.patch, match.patch
    major = patch_major(match.patch) or patch_for_time(match.start_time)
    if match.patch and match.patch != major:  # the source names the letter patch itself
        return major, match.patch
    label = patch_label_for_time(match.start_time)
    if label and (not major or patch_major(label) == major):
        return patch_major(label), label
    return major, major


def _is_talent(ability_key: str) -> bool:
    """Whether a skill-build entry is a talent pick.

    Talents are ``special_bonus_*`` abilities; ``special_bonus_attributes``
    is the plain attribute level-up, not a talent. ``talent_*`` is the key
    scheme of older mock data.
    """
    if ability_key.startswith("special_bonus_"):
        return ability_key != "special_bonus_attributes"
    return ability_key.startswith("talent_")


def keeps_stored_talent_names(match: MatchData) -> bool:
    """Whether the talent names stored with the match beat the current ones.

    The game data knows the talents of one patch, and talent keys are reused
    from patch to patch (the same key can be another talent in the next
    one). Names given at parse time by data of the match's own patch are
    the right ones, so they are kept once the game data has moved on.
    """
    from .parser import talent_names_patch

    named_with = match.talent_names_patch
    return bool(named_with) and named_with != talent_names_patch() and named_with == _patch_versions(match)[0]


def _skill_name(skill: SkillEvent, keep_stored_talent: bool) -> str:
    """Display name of a skill-build entry.

    A talent is named from its key at analysis time, not with the name
    stored when the match was parsed: talent names come from the fetched
    game data, and a match parsed before that data arrived (first start,
    OpenDota unreachable, a new game patch) would keep its stand-in names
    ("Wraith King Facet 1") for good. The stored name remains when it is
    the better one (``keeps_stored_talent_names``) and for a key no
    catalog knows. Other abilities keep the parser's name.
    """
    key = skill.ability_key
    if not key.startswith("special_bonus_") or (keep_stored_talent and skill.ability_name):
        return skill.ability_name
    try:
        from .ai.data_manager import ability_display_name, resolve_ability_name
    except ImportError:  # data layer unavailable
        return skill.ability_name
    return resolve_ability_name(key) or skill.ability_name or ability_display_name(key)


def _hero_index(match: MatchData) -> dict[str, int]:
    """Normalized hero identifier → index in ``match.players``.

    Matches display names (the contract for ``killed`` / ``damage`` keys),
    slugs and ``npc_dota_hero_*`` names alike.
    """
    from .parser import normalize_hero_key

    index: dict[str, int] = {}
    for i, p in enumerate(match.players):
        for name in (p.hero_name, p.hero_slug):
            key = normalize_hero_key(name)
            if key:
                index.setdefault(key, i)
    return index


def _kill_entry(kill: KillEvent, slot_map: dict[int, PlayerData], counted: bool) -> dict:
    """Payload form of one kill-log entry (names resolved, teams as codes).

    ``killer`` is None when no hero made the kill; ``killer_unit`` then
    names the unit (tower, creep, Roshan, ...) when the source knows it.
    ``killer_team`` is the side the kill belongs to: the credited hero's,
    else the side of the unit that landed it as the source reports it
    (``KillEvent.killer_team``: "radiant", "dire", or "neutral" for camp
    creeps, Roshan and the Tormentor), else the side its name carries; None
    when nothing tells. ``counted`` says whether the entry is taken for a
    real death: always for a confirmed one, for an unconfirmed one only when
    ``_hero_kills`` matched it with the scoreboard (the others are probable
    reincarnations).
    """
    victim = slot_map.get(kill.victim_slot)
    killer = slot_map.get(kill.killer_slot) if kill.killer_slot is not None else None
    killer_team = _team_for_slot(kill.killer_slot, slot_map)
    if killer_team is None:
        # The source's own answer first: a unit name does not always carry
        # the side ("dota_fountain"). Matches stored before the field
        # existed only have the name.
        killer_team = kill.killer_team or _team_from_unit_name(kill.killer_unit)
    return {
        "time": kill.time,
        "time_str": format_time(kill.time),
        "victim": victim.hero_name if victim else "",
        "victim_slug": victim.hero_slug if victim else "",
        "victim_slot": kill.victim_slot,
        "victim_team": _team_for_slot(kill.victim_slot, slot_map),
        "killer": killer.hero_name if killer else None,
        "killer_slug": killer.hero_slug if killer else "",
        "killer_slot": kill.killer_slot,
        "killer_team": killer_team,
        "killer_unit": kill.killer_unit,
        "deny": kill.deny,
        "confirmed": kill.confirmed,
        "counted": counted,
    }


def analyze_item_builds(match: MatchData) -> list[dict]:
    """Analyze item builds for all players."""
    results = []
    for player in match.players:
        item_phases = {"starting": [], "early": [], "mid": [], "late": []}
        for item in player.items:
            if item.time <= 0:
                phase = "starting"
            elif item.time <= 600:  # 10 min
                phase = "early"
            elif item.time <= 1500:  # 25 min
                phase = "mid"
            else:
                phase = "late"

            item_phases[phase].append({
                "item": item.item_name,
                "item_key": item.item_key,
                "time": item.time,
                "time_str": format_time(item.time),
            })

        results.append({
            "player_slot": player.player_slot,
            "hero_name": player.hero_name,
            "hero_slug": player.hero_slug,
            "player_name": player.player_name,
            "team": player.team,
            "final_items": player.final_items,
            "final_item_keys": player.final_item_keys,
            "neutral_item": player.neutral_item,
            "backpack": player.backpack,
            "item_phases": item_phases,
            "total_items": len(player.items),
            "net_worth": player.net_worth,
        })
    return results


def _is_ultimate(ability_key: str) -> bool | None:
    """Whether the game data calls the ability its hero's ultimate; None when
    it does not know the ability (the reader may then go by the build)."""
    try:
        from .ai import game_data
        hit = game_data.get_ability_by_key(ability_key) if ability_key else None
    except Exception:  # data layer unavailable
        return None
    return bool(hit[2].get("is_ultimate")) if hit else None


def analyze_skill_builds(match: MatchData) -> list[dict]:
    """Analyze skill progression for all players.

    ``is_ultimate`` of an entry is the game data's word (True / False), or
    None for a talent, the attribute bonus and an ability it does not know:
    the order of the skill points alone cannot tell an ultimate that was
    skilled late or early (Turbo) from a basic ability.
    """
    keep_stored_talents = keeps_stored_talent_names(match)
    results = []
    for player in match.players:
        skill_order = []
        for skill in player.skills:
            is_talent = _is_talent(skill.ability_key or skill.ability_name)
            plain = is_talent or skill.ability_key == "special_bonus_attributes"
            skill_order.append({
                "ability": _skill_name(skill, keep_stored_talents),
                "ability_key": skill.ability_key,
                "level": skill.level,
                "time": skill.time,
                "time_str": format_time(skill.time),
                "is_talent": is_talent,
                "is_ultimate": None if plain else _is_ultimate(skill.ability_key),
            })

        results.append({
            "player_slot": player.player_slot,
            "hero_name": player.hero_name,
            "hero_slug": player.hero_slug,
            "player_name": player.player_name,
            "team": player.team,
            "max_level": player.level,
            "skill_order": skill_order,
        })
    return results


# A ward that fell this many seconds before its full lifetime did not expire.
_WARD_EARLY_S = 5


def _ward_state(ward) -> str:
    """How a ward ended: ``active`` (still standing when the data ends),
    ``dewarded`` (an enemy destroyed it), ``denied`` (its own side did: an
    ally's or the owner's attack) — both are the parser's flags —,
    ``expired`` (it ran its full time) or ``destroyed``: it fell clearly
    before its time and the data does not name who ended it (a source that
    cannot tell a deny, or an attacker of neither side).
    """
    if ward.destroyed_time is None:
        return "active"
    if ward.dewarded:
        return "dewarded"
    if ward.denied:
        return "denied"
    full = OBSERVER_LIFETIME_S if ward.ward_type == "observer" else SENTRY_LIFETIME_S
    return "destroyed" if ward.destroyed_time - ward.time < full - _WARD_EARLY_S else "expired"


def analyze_wards(match: MatchData) -> dict:
    """Analyze ward placements for both teams.

    Every ward carries ``dewarded`` (an enemy destroyed it) and ``denied``
    (its own side did) — the parser's flags, never a lifetime guess —,
    ``state`` (see :func:`_ward_state`) and ``lifetime`` (seconds until it
    was destroyed or expired; None while alive at game end).
    ``killer_slot`` / ``killer_hero_name`` / ``killer_hero_slug`` name the
    hero that dewarded or denied it (None / "" when neither happened, or
    not by a hero, or unknown): an enemy for ``dewarded``, a hero of the
    ward's own team for ``denied``.
    """
    slot_map = _slot_map(match)
    radiant_obs = []
    radiant_sent = []
    dire_obs = []
    dire_sent = []
    all_wards = []

    for ward in match.wards:
        player = slot_map.get(ward.player_slot)
        hero_name = player.hero_name if player else ""
        team = getattr(ward, "team", None)
        if team not in ("radiant", "dire"):
            team = _team_for_slot(ward.player_slot, slot_map)
        lifetime = (
            ward.destroyed_time - ward.time if ward.destroyed_time is not None else None
        )
        ended_by_hero = (ward.dewarded or ward.denied) and ward.killer_slot is not None
        killer = slot_map.get(ward.killer_slot) if ended_by_hero else None
        deward = {"killer_slot": killer.player_slot if killer else None, **_hero_fields(killer, "killer_")}

        ward_info = {
            "x": ward.x,
            "y": ward.y,
            "time": ward.time,
            "time_str": format_time(ward.time),
            "player_slot": ward.player_slot,
            "hero_name": hero_name,
            "survived": ward.destroyed_time is None,
            "lifetime": lifetime,
            "destroyed_time": ward.destroyed_time,
            "dewarded": ward.dewarded,
            "denied": ward.denied,
            "state": _ward_state(ward),
            **deward,
        }

        # Wards whose owner/team is unknown are listed in all_wards only.
        if team == "radiant":
            (radiant_obs if ward.ward_type == "observer" else radiant_sent).append(ward_info)
        elif team == "dire":
            (dire_obs if ward.ward_type == "observer" else dire_sent).append(ward_info)

        all_wards.append({
            "ward_type": ward.ward_type,
            "x": ward.x,
            "y": ward.y,
            "time": ward.time,
            "time_str": format_time(ward.time),
            "player_slot": ward.player_slot,
            "team": team,
            "hero_name": hero_name,
            "destroyed_time": ward.destroyed_time,
            "lifetime": lifetime,
            "dewarded": ward.dewarded,
            "denied": ward.denied,
            "state": _ward_state(ward),
            **deward,
        })

    # Advanced ward statistics
    def _avg_lifetime(wards_list):
        lifetimes = [w["lifetime"] for w in wards_list if w["lifetime"] is not None]
        return round(sum(lifetimes) / len(lifetimes), 1) if lifetimes else 0

    def _deward_count(wards_list):
        return sum(1 for w in wards_list if w["dewarded"])

    def _active(wards_list, max_life, t0, t1):
        return sum(
            1 for w in wards_list
            if w["time"] <= t1 and w["time"] + (w["lifetime"] if w["lifetime"] is not None else max_life) >= t0
        )

    # Ward density per 5-minute phase (no phases for a game that never started)
    ward_density = []
    phase_end = match.duration // 60 + 1 if match.duration > 0 else 0
    for m_start in range(0, phase_end, 5):
        t0, t1 = m_start * 60, (m_start + 5) * 60
        ward_density.append({
            "phase": f"{m_start}-{m_start + 5}",
            "radiant": _active(radiant_obs, OBSERVER_LIFETIME_S, t0, t1)
            + _active(radiant_sent, SENTRY_LIFETIME_S, t0, t1),
            "dire": _active(dire_obs, OBSERVER_LIFETIME_S, t0, t1)
            + _active(dire_sent, SENTRY_LIFETIME_S, t0, t1),
        })

    def _team_stats(obs, sent):
        return {
            "observers": obs,
            "sentries": sent,
            "total_obs": len(obs),
            "total_sent": len(sent),
            "avg_obs_lifetime": _avg_lifetime(obs),
            "avg_sent_lifetime": _avg_lifetime(sent),
            "dewarded_obs": _deward_count(obs),
            "dewarded_sent": _deward_count(sent),
        }

    return {
        "radiant": _team_stats(radiant_obs, radiant_sent),
        "dire": _team_stats(dire_obs, dire_sent),
        "ward_density": ward_density,
        "duration": match.duration,
        "all_wards": all_wards,
    }


def analyze_smokes(match: MatchData) -> dict:
    """Analyze smoke usage for both teams.

    ``player_slot`` / ``hero_name`` / ``hero_slug`` name the hero that used
    the smoke (None / "" when the source does not say); ``participants``
    are the heroes it covered.
    """
    slot_map = _slot_map(match)
    radiant_smokes = []
    dire_smokes = []
    other_smokes = []  # team unknown: timeline only

    for smoke in match.smokes:
        participants = []
        for slot in smoke.player_slots:
            player = slot_map.get(slot)
            if player:
                participants.append({
                    "player_slot": slot,
                    "hero_name": player.hero_name,
                })

        user = slot_map.get(smoke.player_slot) if smoke.player_slot is not None else None
        smoke_info = {
            "time": smoke.time,
            "time_str": format_time(smoke.time),
            "team": smoke.team,
            "player_slot": user.player_slot if user else None,
            **_hero_fields(user, ""),
            "participants": participants,
            "num_players": len(participants),
            "x": smoke.x,
            "y": smoke.y,
            "end_time": smoke.end_time,
            "end_time_str": format_time(smoke.end_time) if smoke.end_time else None,
            "duration": smoke.end_time - smoke.time if smoke.end_time else 0,
            "path": smoke.path,
            "result": smoke.result,
            "kills_gained": smoke.kills_gained,
            "deaths_suffered": smoke.deaths_suffered,
            "objective_taken": smoke.objective_taken,
        }

        if smoke.team == "radiant":
            radiant_smokes.append(smoke_info)
        elif smoke.team == "dire":
            dire_smokes.append(smoke_info)
        else:
            other_smokes.append(smoke_info)

    def _smoke_stats(smoke_list):
        total = len(smoke_list)
        kills = sum(1 for s in smoke_list if s["result"] == "kill")
        objectives = sum(1 for s in smoke_list if s["result"] == "objective")
        traded = sum(1 for s in smoke_list if s["result"] == "traded")
        no_contact = sum(1 for s in smoke_list if s["result"] == "no_contact")
        caught = sum(1 for s in smoke_list if s["result"] == "caught")
        # Profitable trades (more kills than deaths) count as successful
        traded_won = sum(
            1 for s in smoke_list
            if s["result"] == "traded" and s["kills_gained"] > s["deaths_suffered"]
        )
        successful = kills + objectives + traded_won
        return {
            "smokes": smoke_list,
            "total": total,
            "successful": successful,
            "success_rate": round(successful / max(1, total) * 100, 1),
            "kills": kills,
            "objectives": objectives,
            "traded": traded,
            "traded_won": traded_won,
            "no_contact": no_contact,
            "caught": caught,
        }

    return {
        "radiant": _smoke_stats(radiant_smokes),
        "dire": _smoke_stats(dire_smokes),
        "timeline": sorted(
            radiant_smokes + dire_smokes + other_smokes, key=lambda s: s["time"]
        ),
    }


def _abilities_used(tp: TeamfightPlayer) -> tuple[list[str], list[str]]:
    """Abilities a player used in a fight: (keys, display names), parallel.

    Sources list ability keys; a display name (older mock data) is mapped
    back to its key through the player's own casts where possible. World
    interactions and cosmetic actions (``parser.is_utility_ability``) are
    not hero abilities and are left out.
    """
    from .parser import ability_display_name, is_utility_ability

    key_by_name = {c.ability_name: c.ability_key for c in tp.ability_casts if c.ability_key}
    keys = [key_by_name.get(entry, entry) for entry in tp.abilities_used]
    keys = [key for key in keys if not is_utility_ability(key)]
    return keys, [ability_display_name(key) for key in keys]


def _guessed_kill_timeline(tf_start: int, players_detail: list[dict]) -> list[dict]:
    """Kill timeline of a fight for sources without a kill log.

    One entry per player that died (``death_time_offset``); the killer is
    the player whose ``kill_targets`` names the victim at about that time.
    """
    timeline = []
    for p in players_detail:
        if p["deaths"] <= 0 or p["death_time_offset"] is None:
            continue
        victim_names = {p["hero_name"], p["hero_slug"]} - {""}
        killer = next(
            (
                k
                for k in players_detail
                for kt in k["kill_targets"]
                if (kt.get("target_slot") == p["player_slot"] or kt.get("target") in victim_names)
                and (kt.get("time_offset") is None or abs(kt["time_offset"] - p["death_time_offset"]) < 0.5)
            ),
            None,
        )
        death_time = tf_start + p["death_time_offset"]
        timeline.append({
            "time": death_time,
            "time_str": format_time(death_time),
            "victim": p["hero_name"],
            "victim_slug": p["hero_slug"],
            "victim_slot": p["player_slot"],
            "victim_team": p["team"],
            "killer": killer["hero_name"] if killer else None,
            "killer_slug": killer["hero_slug"] if killer else "",
            "killer_slot": killer["player_slot"] if killer else None,
            "killer_team": killer["team"] if killer else None,
            "killer_unit": "",
            "deny": False,
            "confirmed": True,
            "counted": True,
            "time_offset": p["death_time_offset"],
        })
    return timeline


# A natural respawn takes at least 12 s and a buyback has a log of its own:
# a position-log sample this close after a death can still show the hero's
# last health and is no sign of a respawn.
_RESPAWN_MIN_S = 1.0


def _dead_spans(player: PlayerData, deaths: list[KillEvent]) -> list[tuple[float, float | None]] | None:
    """When a hero was dead: ``(death, respawn)`` spans in time order.

    ``deaths`` are the hero's kill-log entries in time order. The respawn
    is the earliest of: the hero's buyback after the death (exact), the
    entry's own ``respawn_time`` (the replay parser sees it up to a second
    late) and, for an entry without one, the first position-log sample
    that shows health again (one sample every 5 s, so up to that much
    late). A respawn of None means the hero was still dead when the data
    ends. Reincarnations (Aegis, Wraith King) are no kill-log deaths and
    leave no span. None when the source cannot tell: no position log
    (OpenDota), one without health values, or a kill log that misses deaths
    of the hero.
    """
    samples = [s for s in player.position_log if len(s) > 3]
    if not any(s[3] >= 0 for s in samples) or len(deaths) < player.deaths:
        return None
    alive_at = [s[0] for s in samples if s[3] > 0]
    buybacks = sorted(float(entry.get("time") or 0) for entry in player.buyback_log)
    spans: list[tuple[float, float | None]] = []
    for n, kill in enumerate(deaths):
        death = kill.time
        next_death = deaths[n + 1].time if n + 1 < len(deaths) else math.inf
        back = [t for t in buybacks if death <= t < next_death][:1]
        if kill.respawn_time is not None:
            back.append(kill.respawn_time)
        else:
            seen = bisect.bisect_left(alive_at, death + _RESPAWN_MIN_S)
            back += alive_at[seen:seen + 1]
        respawn = min(back) if back else None
        if next_death != math.inf and (respawn is None or respawn > next_death):
            respawn = next_death    # alive again between two samples, then dead once more
        spans.append((death, respawn))
    return spans


def _alive_intervals(spans: list[tuple[float, float | None]] | None, start: float, end: float) -> list[list[float]] | None:
    """``[[from, to], ...]`` in seconds from ``start``: when a hero was alive
    between ``start`` and ``end`` (see ``_dead_spans``). ``[]`` = dead all
    the time, None = unknown."""
    if spans is None:
        return None
    intervals: list[list[float]] = []
    cursor = start
    for death, respawn in spans:
        if death >= end:
            break
        if respawn is not None and respawn <= cursor:
            continue
        if death > cursor:
            intervals.append([cursor, death])
        cursor = end if respawn is None else respawn
    if cursor < end:
        intervals.append([cursor, end])
    return [[round(float(a - start), 1), round(float(b - start), 1)] for a, b in intervals]


def analyze_teamfights(match: MatchData) -> list[dict]:
    """Analyze teamfight data.

    ``kill_timeline`` lists the hero deaths of a fight in time order. It
    comes from ``MatchData.kills_log`` (every death, real killer); only
    sources without a kill log fall back to the per-player teamfight data.
    Entries with ``counted`` false are probable reincarnations (see
    ``_kill_entry``). ``abilities_used`` holds ability keys,
    ``abilities_used_names`` the matching display names; like
    ``ability_casts`` they list hero abilities only (see
    ``_abilities_used``). ``gold_delta`` / ``xp_delta`` of a fight are
    Radiant minus Dire (positive = Radiant came out ahead).

    ``alive_intervals`` of a player are the parts of the fight window in
    which the hero was alive, as ``[[from, to], ...]`` in seconds from the
    fight start: a death ends one, a buyback or a respawn starts the next,
    and a hero that was still dead from before the fight starts late. ``[]``
    means dead for the whole window; None means the source cannot tell
    (OpenDota). Deaths and buybacks are exact, a respawn is up to a second
    late (see ``_dead_spans``).
    """
    from .parser import is_utility_ability

    slot_map = _slot_map(match)
    kills = sorted(match.kills_log, key=lambda k: k.time)
    deaths_by_slot: dict[int, list[KillEvent]] = {}
    for kill in kills:
        deaths_by_slot.setdefault(kill.victim_slot, []).append(kill)
    dead_spans = {p.player_slot: _dead_spans(p, deaths_by_slot.get(p.player_slot, [])) for p in match.players}
    hero_kills = {id(k) for k in _hero_kills(match, slot_map)}
    assigned: set[int] = set()  # kills already attributed to an earlier fight
    results = []
    for i, tf in enumerate(match.teamfights, 1):
        duration = tf.end_time - tf.start_time
        winner = "radiant" if tf.dire_deaths > tf.radiant_deaths else "dire"
        if tf.dire_deaths == tf.radiant_deaths:
            winner = "draw"

        players_detail = []
        for tp in tf.players:
            ability_keys, ability_names = _abilities_used(tp)
            players_detail.append({
                "player_slot": tp.player_slot,
                "hero_name": tp.hero_name,
                "hero_slug": tp.hero_slug,
                "team": _team_for_slot(tp.player_slot, slot_map),
                "x": tp.x,
                "y": tp.y,
                "damage_dealt": tp.damage_dealt,
                "damage_taken": tp.damage_taken,
                "healing": tp.healing,
                "kills": tp.kills,
                "deaths": tp.deaths,
                "gold_delta": tp.gold_delta,
                "xp_delta": tp.xp_delta,
                "abilities_used": ability_keys,
                "abilities_used_names": ability_names,
                "item_uses": tp.item_uses,
                "buybacks": tp.buybacks,
                "death_time_offset": tp.death_time_offset,
                "alive_intervals": _alive_intervals(dead_spans.get(tp.player_slot), tf.start_time, tf.end_time),
                "kill_targets": tp.kill_targets,
                "positions": [
                    {"x": pos.x, "y": pos.y, "time_offset": pos.time_offset}
                    for pos in tp.positions
                ],
                "ability_casts": [
                    {
                        "ability_name": ac.ability_name,
                        "ability_key": ac.ability_key,
                        "time_offset": ac.time_offset,
                        "x": ac.x, "y": ac.y,
                    }
                    for ac in tp.ability_casts
                    if not is_utility_ability(ac.ability_key)
                ],
            })

        if kills:
            kill_timeline = []
            for idx, kill in enumerate(kills):
                # Fight bounds are whole seconds, kill times need not be: allow
                # less than 1 s of slack (whole-second times must be in bounds).
                if idx in assigned or not tf.start_time - 1 < kill.time < tf.end_time + 1:
                    continue
                assigned.add(idx)
                kill_timeline.append({
                    **_kill_entry(kill, slot_map, kill.confirmed or id(kill) in hero_kills),
                    "time_offset": round(kill.time - tf.start_time, 1),
                })
        else:
            kill_timeline = _guessed_kill_timeline(tf.start_time, players_detail)
            kill_timeline.sort(key=lambda k: k["time_offset"])

        results.append({
            "index": i,
            "start_time": tf.start_time,
            "end_time": tf.end_time,
            "start_time_str": format_time(tf.start_time),
            "duration": duration,
            "radiant_deaths": tf.radiant_deaths,
            "dire_deaths": tf.dire_deaths,
            "winner": winner,
            "gold_delta": tf.gold_delta,
            "xp_delta": tf.xp_delta,
            "players": players_detail,
            "kill_timeline": kill_timeline,
        })
    return results


def analyze_abilities(match: MatchData) -> dict:
    """Analyze ability usage patterns.

    Hero abilities only: world interactions and cosmetic actions
    (``parser.is_utility_ability``: watchers, outposts, Twin Gates, High
    Five) are not counted or listed.
    """
    from .parser import is_utility_ability

    by_player: dict[int, list[dict]] = {}
    by_ability: dict[str, int] = {}

    for event in match.ability_events:
        if is_utility_ability(event.ability_key):
            continue
        entry = {
            "ability_name": event.ability_name,
            "ability_key": event.ability_key,
            "time": event.time,
            "time_str": format_time(event.time),
            "hero_name": event.hero_name,
            "target": event.target,
        }

        if event.player_slot not in by_player:
            by_player[event.player_slot] = []
        by_player[event.player_slot].append(entry)

        by_ability[event.ability_name] = by_ability.get(event.ability_name, 0) + 1

    player_summaries = []
    for player in match.players:
        events = by_player.get(player.player_slot, [])
        ability_counts: dict[str, int] = {}
        ability_key_map: dict[str, str] = {}
        for e in events:
            name = e["ability_name"]
            ability_counts[name] = ability_counts.get(name, 0) + 1
            if e["ability_key"]:
                ability_key_map[name] = e["ability_key"]

        player_summaries.append({
            "player_slot": player.player_slot,
            "hero_name": player.hero_name,
            "hero_slug": player.hero_slug,
            "player_name": player.player_name,
            "team": player.team,
            "total_casts": len(events),
            "ability_counts": ability_counts,
            "ability_key_map": ability_key_map,
            "events": events,
        })

    return {
        "by_player": player_summaries,
        "most_used": sorted(
            by_ability.items(), key=lambda x: x[1], reverse=True
        )[:20],
    }


def _effective_lane_role(p: PlayerData) -> int:
    """Relative lane role (1 safe, 2 mid, 3 off, 4 roaming/jungle, 0 unknown).

    Uses ``lane_role`` when set, otherwise derives it from the absolute
    ``lane`` (1 bottom, 2 mid, 3 top, 4/5 jungle) and the player's team:
    bottom is Radiant's safe lane and Dire's off lane, top the reverse.
    """
    if p.lane_role in LANE_ROLE_CODES:
        return p.lane_role
    if p.lane == 2:
        return 2
    if p.lane in (4, 5):
        return 4
    if p.lane in (1, 3) and p.team in ("radiant", "dire"):
        safe_lane = 1 if p.team == "radiant" else 3
        return 1 if p.lane == safe_lane else 3
    return 0


def _at_ten_minutes(timeline: list[int]) -> int | None:
    """Value at minute 10 of a per-minute timeline (last value if shorter)."""
    if not timeline:
        return None
    return timeline[10] if len(timeline) > 10 else timeline[-1]


def _snapshot_at_ten(p: PlayerData):
    """Latest laning snapshot at or before 10:00, or None."""
    best = None
    for snap in p.laning_snapshots:
        if snap.time <= 600 and (best is None or snap.time >= best.time):
            best = snap
    return best


def _lane_positions(p: PlayerData) -> list[list[float]]:
    """Laning-phase position histogram: ``[[x, y, samples], ...]``.

    x/y are the centres of 128-unit cells in world units. OpenDota provides
    the histogram itself (``lane_pos``, one sample per second); for replay
    sources it is built from the 5-second laning snapshots. Empty when the
    source has neither.
    """
    cells: dict[tuple[int, int], int] = {}
    if p.lane_pos:
        for cell_x, column in p.lane_pos.items():
            if not isinstance(column, dict):
                continue
            for cell_y, samples in column.items():
                try:
                    key = (int(cell_x), int(cell_y))
                    cells[key] = cells.get(key, 0) + int(samples)
                except (TypeError, ValueError):
                    continue
    else:
        for snap in p.laning_snapshots:
            if 0 <= snap.time <= 600:
                key = (
                    int((snap.x + geo.WORLD_ORIGIN_OFFSET) // geo.CELL_SIZE),
                    int((snap.y + geo.WORLD_ORIGIN_OFFSET) // geo.CELL_SIZE),
                )
                cells[key] = cells.get(key, 0) + 1
    return [
        [geo.cell_to_world(cell_x), geo.cell_to_world(cell_y), samples]
        for (cell_x, cell_y), samples in sorted(cells.items())
    ]


def analyze_laning(match: MatchData) -> dict:
    """Analyze laning phase data for all players.

    ``lane_name`` is a code: "safe" | "mid" | "off" | "roaming" | "unknown"
    (from the relative lane role); ``lane_side`` is the absolute map lane:
    "bottom" | "mid" | "top" | "jungle" | "unknown". The 10:00 values
    (``lh_ten``, ``dn_ten``, ``gold_ten``, ``xp_ten``) come from the
    per-minute timelines, else from the laning snapshots / creep logs; each
    is None when the source has nothing to derive it from (an OpenDota match
    without parsed data), never a made-up 0. ``lane_pos`` is the position
    histogram of the laning phase (see ``_lane_positions``).

    ``lane_efficiency_applicable`` is False for a game mode the lane
    efficiency does not describe (``LANE_EFFICIENCY_NA_MODES``: Turbo);
    ``lane_efficiency`` and ``lane_efficiency_pct`` of every player are
    None then.
    """
    efficiency_applies = str(match.game_mode or "").strip().lower() not in LANE_EFFICIENCY_NA_MODES
    # LaningDamageEvent.target_idx is the index of the target in
    # match.players (0-9, as emitted by the clarity Java analyzer and mock
    # data), NOT player_slot (dire slots are 128+).
    damage_taken_by_idx: dict[int, int] = {}
    for other_p in match.players:
        for e in other_p.laning_damage_events:
            damage_taken_by_idx[e.target_idx] = damage_taken_by_idx.get(e.target_idx, 0) + e.damage

    players = []
    for idx, p in enumerate(match.players):
        lane_role = _effective_lane_role(p)
        lane_name = "roaming" if p.is_roaming else LANE_ROLE_CODES.get(lane_role, "unknown")
        snap = _snapshot_at_ten(p)

        lh_ten = _at_ten_minutes(p.lh_t)
        if lh_ten is None:
            if snap is not None:
                lh_ten = snap.last_hits
            elif p.laning_creep_kills or not p.last_hits:
                lh_ten = sum(1 for c in p.laning_creep_kills if c.time <= 600)
            # else: last hits, but the source has no timeline of them
        dn_ten = _at_ten_minutes(p.dn_t)
        if dn_ten is None:
            if snap is not None:
                dn_ten = snap.denies
            elif p.laning_denies or not p.denies:
                dn_ten = sum(1 for d in p.laning_denies if d.time <= 600)
        gold_ten = _at_ten_minutes(p.gold_t)
        if gold_ten is None and snap is not None:
            gold_ten = snap.net_worth
        xp_ten = _at_ten_minutes(p.xp_t)

        lane_efficiency = p.lane_efficiency
        lane_efficiency_pct = p.lane_efficiency_pct
        if not efficiency_applies:
            lane_efficiency = lane_efficiency_pct = None
        elif not lane_efficiency and len(p.gold_t) > 10:
            # Only OpenDota ships this value; same formula for the others.
            lane_efficiency = p.gold_t[10] / LANE_GOLD_AT_TEN
            lane_efficiency_pct = int(lane_efficiency * 100)

        players.append({
            "player_slot": p.player_slot,
            "hero_name": p.hero_name,
            "hero_slug": p.hero_slug,
            "team": p.team,
            "lane": p.lane or 0,
            "lane_role": lane_role,
            "lane_name": lane_name,
            "lane_side": LANE_SIDE_CODES.get(p.lane or 0, "unknown"),
            "is_roaming": p.is_roaming,
            "lane_efficiency": lane_efficiency,
            "lane_efficiency_pct": lane_efficiency_pct,
            "lane_pos": _lane_positions(p),
            "lh_ten": lh_ten,
            "dn_ten": dn_ten,
            "gold_ten": gold_ten,
            "xp_ten": xp_ten,
            # Enhanced laning data
            "laning_snapshots": [s.model_dump() for s in p.laning_snapshots],
            "creep_kills": [c.model_dump() for c in p.laning_creep_kills],
            "denies_log": [d.model_dump() for d in p.laning_denies],
            "damage_events": [e.model_dump() for e in p.laning_damage_events],
            "regen_events": [r.model_dump() for r in p.laning_regen_events],
            # Computed laning stats
            "total_lh_laning": len(p.laning_creep_kills),
            "total_denies_laning": len(p.laning_denies),
            "total_damage_dealt_laning": sum(e.damage for e in p.laning_damage_events),
            "total_damage_taken_laning": damage_taken_by_idx.get(idx, 0),
            "regen_items_used": len(p.laning_regen_events),
        })

    return {"players": players, "lane_efficiency_applicable": efficiency_applies}


def _hero_kills(match: MatchData, slot_map: dict[int, PlayerData]) -> list[KillEvent]:
    """The kill-log entries that count as one hero killing another.

    Confirmed entries always count. An unconfirmed one (the source cannot
    tell a death from a reincarnation) only counts while the scoreboard
    still shows an unexplained kill for its killer and an unexplained death
    for its victim. Among them, an entry followed by another death of the
    same hero within REINCARNATION_WINDOW_S is the reincarnation before
    that death and is considered last; the rest is taken earliest first,
    because reincarnation and the Aegis only come into play later in a game.
    """
    candidates = [
        k for k in match.kills_log
        if not k.deny and k.killer_slot in slot_map and k.victim_slot in slot_map
    ]
    counted = [k for k in candidates if k.confirmed]
    kills_left = {slot: p.kills for slot, p in slot_map.items()}
    deaths_left = {slot: p.deaths for slot, p in slot_map.items()}
    for k in counted:
        kills_left[k.killer_slot] -= 1
    death_times: dict[int, list[float]] = {}
    for k in match.kills_log:  # every death, whoever the killer was
        death_times.setdefault(k.victim_slot, []).append(k.time)
        if k.confirmed and k.victim_slot in deaths_left:
            deaths_left[k.victim_slot] -= 1

    def died_again(k: KillEvent) -> bool:
        return any(0 < t - k.time <= REINCARNATION_WINDOW_S for t in death_times[k.victim_slot])

    for k in sorted((k for k in candidates if not k.confirmed), key=lambda k: (died_again(k), k.time)):
        if kills_left[k.killer_slot] > 0 and deaths_left[k.victim_slot] > 0:
            kills_left[k.killer_slot] -= 1
            deaths_left[k.victim_slot] -= 1
            counted.append(k)
    return counted


def _kill_feed(match: MatchData, slot_map: dict[int, PlayerData]) -> list[dict]:
    """Every hero death in time order with the running score after it.

    The score follows the kill log by the rule of the game's own team
    counters (``MatchData.radiant_score`` / ``dire_score``): a death counts
    for the other team when an enemy hero (see ``_hero_kills``) or an enemy
    unit (tower, lane creep, fountain) made the kill; denies, deaths to
    neutrals and entries that are not ``counted`` leave it unchanged. A
    team's score can therefore be lower than the other team's deaths. It
    can differ from the scoreboard by the few kills the game credits to a
    hero that no logged death explains.
    """
    feed = []
    score = {"radiant": 0, "dire": 0}
    hero_kills = {id(k) for k in _hero_kills(match, slot_map)}
    for kill in sorted(match.kills_log, key=lambda k: k.time):
        entry = _kill_entry(kill, slot_map, kill.confirmed or id(kill) in hero_kills)
        scorer = entry["killer_team"]
        by_unit = kill.killer_slot is None and kill.confirmed and not kill.deny
        if (id(kill) in hero_kills or by_unit) and scorer in score and scorer != entry["victim_team"]:
            score[scorer] += 1
        feed.append({
            **entry,
            "assist_slots": kill.assist_slots,
            "x": kill.x,
            "y": kill.y,
            "radiant_score": score["radiant"],
            "dire_score": score["dire"],
        })
    return feed


def analyze_combat(match: MatchData) -> dict:
    """Kill and damage matrices between players, plus the kill log.

    ``matrix[i][j]`` is what player i did to player j (indices follow
    ``heroes``). The kill matrix counts the hero kills of
    ``MatchData.kills_log`` (see ``_hero_kills``); the per-player ``killed``
    counters are only used when the source has no kill log. ``kills`` is
    the kill feed (see ``_kill_feed``; entries as in ``_kill_entry``).
    """
    from .parser import normalize_hero_key

    heroes = [p.hero_name for p in match.players]
    hero_idx = _hero_index(match)
    slot_map = _slot_map(match)
    slot_idx = {p.player_slot: i for i, p in enumerate(match.players)}
    num = len(match.players)

    kill_matrix = [[0] * num for _ in range(num)]
    if match.kills_log:
        for kill in _hero_kills(match, slot_map):
            kill_matrix[slot_idx[kill.killer_slot]][slot_idx[kill.victim_slot]] += 1
    else:
        for i, p in enumerate(match.players):
            for unit_name, count in p.killed.items():
                j = hero_idx.get(normalize_hero_key(unit_name))
                if j is not None:
                    kill_matrix[i][j] = count

    damage_matrix = [[0] * num for _ in range(num)]
    for i, p in enumerate(match.players):
        for unit_name, amount in p.damage.items():
            j = hero_idx.get(normalize_hero_key(unit_name))
            if j is not None:
                damage_matrix[i][j] = amount

    return {
        "kill_matrix": {
            "heroes": heroes,
            "matrix": kill_matrix,
        },
        "damage_matrix": {
            "heroes": heroes,
            "matrix": damage_matrix,
        },
        "kills": _kill_feed(match, slot_map),
    }


def analyze_farm(match: MatchData) -> dict:
    """Analyze farming patterns, gold/xp sources, and last hit timelines.

    ``reasons`` dicts are keyed by codes from GOLD_REASON_CODES /
    XP_REASON_CODES (unknown numeric reasons become ``unknown_<n>``;
    reasons with a total of 0 are left out). The counts of ``unit_kills``
    other than ``hero_kills`` are None when the source has no breakdown.
    """

    unit_kills = []
    gold_reasons_list = []
    xp_reasons_list = []
    lh_timeline = []

    # A scoreboard-only source (an OpenDota match whose replay was never
    # parsed) has last hits but no word on what was killed: every count
    # below is 0 for every player. That is "unknown" (None), not a zero.
    breakdown = ("lane_kills", "neutral_kills", "ancient_kills", "tower_kills", "courier_kills", "roshans_killed")
    known = (any(getattr(p, field) for p in match.players for field in breakdown)
             or not any(p.last_hits for p in match.players))

    for p in match.players:
        base = {
            "player_slot": p.player_slot,
            "hero_name": p.hero_name,
            "hero_slug": p.hero_slug,
            "team": p.team,
        }

        unit_kills.append({
            **base,
            "hero_kills": p.kills,
            **{field: getattr(p, field) if known else None for field in breakdown},
        })

        gold_reasons_list.append({**base, "reasons": _amounts_by_code(p.gold_reasons, GOLD_REASON_CODES)})
        xp_reasons_list.append({**base, "reasons": _amounts_by_code(p.xp_reasons, XP_REASON_CODES)})

        lh_timeline.append({**base, "lh_t": p.lh_t})

    return {
        "unit_kills": unit_kills,
        "gold_reasons": gold_reasons_list,
        "xp_reasons": xp_reasons_list,
        "lh_timeline": lh_timeline,
    }


def analyze_purchases(match: MatchData) -> dict:
    """Analyze item purchase timelines grouped into 5-minute phases."""
    players = []
    for p in match.players:
        base = {
            "player_slot": p.player_slot,
            "hero_name": p.hero_name,
            "hero_slug": p.hero_slug,
            "team": p.team,
        }

        # Build enriched purchase log
        purchase_log = []
        for entry in p.purchase_log:
            time_val = int(entry.get("time") or 0)
            purchase_log.append({
                "time": time_val,
                "key": entry.get("key", ""),
                "time_str": format_time(time_val),
            })

        # Group into phases: starting (before the horn) + 5-minute phases
        max_time = match.duration
        phases: dict[str, list] = {}

        # Starting items: everything bought up to the horn. Strategy-time
        # purchases carry the pre-game start, which need not be exactly -90.
        starting_items = [
            item for item in purchase_log if item["time"] <= 0
        ]
        if starting_items:
            phases["starting"] = starting_items

        for m_start in range(0, max(0, max_time) // 60 + 5, 5):
            phase_key = f"{m_start}-{m_start + 5}"
            t0, t1 = m_start * 60, (m_start + 5) * 60
            phase_items = [
                item for item in purchase_log if t0 < item["time"] <= t1
            ]
            if phase_items or t0 < max_time:
                phases[phase_key] = phase_items
            if t0 >= max_time:
                break

        players.append({
            **base,
            "purchase_log": purchase_log,
            "phases": phases,
        })

    return {"players": players}


def analyze_graphs(match: MatchData) -> dict:
    """Provide graph data: team advantages and per-player timelines."""
    player_gold = []
    player_xp = []
    player_lh = []

    for p in match.players:
        base = {
            "player_slot": p.player_slot,
            "hero_name": p.hero_name,
            "hero_slug": p.hero_slug,
            "team": p.team,
        }
        player_gold.append({**base, "gold_t": p.gold_t})
        player_xp.append({**base, "xp_t": p.xp_t})
        player_lh.append({**base, "lh_t": p.lh_t})

    return {
        "radiant_gold_adv": match.radiant_gold_adv,
        "radiant_xp_adv": match.radiant_xp_adv,
        "player_gold": player_gold,
        "player_xp": player_xp,
        "player_lh": player_lh,
    }


_OTHER_TEAM = {"radiant": "dire", "dire": "radiant"}


def _team_from_unit_name(name) -> str | None:
    """Team encoded in a unit name ("..._goodguys_..." / "..._badguys_...")."""
    if not isinstance(name, str):
        return None
    if "goodguys" in name:
        return "radiant"
    if "badguys" in name:
        return "dire"
    return None


def analyze_objectives(match: MatchData) -> dict:
    """Analyze objectives (towers, rax, roshan) and rune pickups.

    ``team`` is the team credited with the objective (None when unknown).
    First-blood entries also name the victim (``victim_*``). A courier
    kill (``CHAT_MESSAGE_COURIER_LOST``) is credited to the killers:
    ``hero_name`` is the killing hero, ``courier_team`` the side that lost
    the courier and ``value`` the gold bounty (None when unknown). Rune
    counts are keyed by RUNE_CODES codes.
    """
    slot_map = _slot_map(match)
    slug_to_player = {p.hero_slug: p for p in match.players if p.hero_slug}

    objectives = []
    for obj in match.objectives:
        time_val = int(obj.get("time") or 0)
        unit = obj.get("unit") or ""
        key = obj.get("key", "")
        slot = _resolve_event_slot(obj, match, slot_map)
        player = slot_map.get(slot) if slot is not None else None
        if player is None and isinstance(unit, str) and unit.startswith("npc_dota_hero_"):
            player = slug_to_player.get(unit[len("npc_dota_hero_"):])

        team = obj.get("team")
        if isinstance(team, int) and not isinstance(team, bool):
            team = "radiant" if team == 2 else "dire" if team == 3 else None
        elif team not in ("radiant", "dire"):
            team = None
        courier_team = None
        if obj.get("type") == "CHAT_MESSAGE_COURIER_LOST":
            # The source's team is the side that LOST the courier and the
            # player the enemy hero that killed it: credit the other side.
            killer_team = player.team if player is not None else _team_for_slot(slot, slot_map)
            courier_team = team or _OTHER_TEAM.get(killer_team)
            team = _OTHER_TEAM.get(courier_team)
        if team is None and player is not None:
            team = player.team
        if team is None and slot is not None:
            team = _team_for_slot(slot, slot_map)
        if team is None:
            team = _team_from_unit_name(unit)
        if team is None:
            # A destroyed building is credited to the other team.
            owner = _team_from_unit_name(key)
            if owner is not None:
                team = "dire" if owner == "radiant" else "radiant"

        entry = {
            "time": time_val,
            "time_str": format_time(time_val),
            "type": obj.get("type", ""),
            "team": team,
            "key": key or unit,
            "hero_name": player.hero_name if player else "",
            "hero_slug": player.hero_slug if player else "",
        }
        if entry["type"] == "CHAT_MESSAGE_COURIER_LOST":
            value = obj.get("value")
            entry["courier_team"] = courier_team
            entry["value"] = value if isinstance(value, int) and not isinstance(value, bool) else None
        if entry["type"] == "CHAT_MESSAGE_FIRSTBLOOD":
            # key = the victim's player_slot as a string (the killer is the
            # entry's own player).
            victim = slot_map.get(int(key)) if str(key).isascii() and str(key).isdigit() else None
            entry["victim_slot"] = victim.player_slot if victim else None
            entry["victim_hero_name"] = victim.hero_name if victim else ""
            entry["victim_hero_slug"] = victim.hero_slug if victim else ""
        objectives.append(entry)

    objectives.sort(key=lambda o: o["time"])

    # Rune analysis per player
    runes = []
    for p in match.players:
        base = {
            "player_slot": p.player_slot,
            "hero_name": p.hero_name,
            "hero_slug": p.hero_slug,
            "team": p.team,
        }
        rune_counts: dict[str, int] = {}
        for key, count in p.runes.items():
            code = _code_for(key, RUNE_CODES)
            rune_counts[code] = rune_counts.get(code, 0) + (count or 0)
        total = sum(rune_counts.values())
        if total > 0:
            runes.append({**base, "runes": rune_counts, "total": total})

    return {
        "objectives": objectives,
        "runes": runes,
    }


def analyze_benchmarks(match: MatchData) -> dict:
    """Return benchmark data for all players."""
    players = []
    for p in match.players:
        players.append({
            "player_slot": p.player_slot,
            "hero_name": p.hero_name,
            "hero_slug": p.hero_slug,
            "team": p.team,
            "benchmarks": p.benchmarks,
        })
    return {"players": players}


def _load_hero_id_map() -> dict[int, dict]:
    """Load hero_id -> {hero_name, hero_slug} from static heroes.json.

    Thin wrapper over the shared parser helper (single heroes.json loader);
    only remaps the key names used by the draft analysis below.
    """
    from .parser import load_hero_map
    return {
        hid: {"hero_name": info.get("name", ""), "hero_slug": info.get("slug", "")}
        for hid, info in load_hero_map().items()
    }


def _draft_team(value) -> str | None:
    """Team of a draft entry: 0 → "radiant", 1 → "dire", anything else None."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return {0: "radiant", 1: "dire"}.get(value)


def analyze_draft(match: MatchData) -> dict:
    """Analyze draft picks and bans.

    ``team`` is "radiant" | "dire" | None; None is a ban no team owns (the
    all-pick ban phase of a replay: the nominations are not attributed).
    """
    # Build hero_id -> hero info from static data (for bans - heroes not in match)
    hero_id_map = _load_hero_id_map()

    # Also build from match players for better slug resolution
    hero_slug_map = {}  # slug/internal_name -> player info (for Clarity picks)
    for p in match.players:
        info = {"hero_name": p.hero_name, "hero_slug": p.hero_slug}
        hero_id_map[p.hero_id] = info
        if p.hero_slug:
            hero_slug_map[p.hero_slug] = info
            hero_slug_map[p.hero_slug.replace("_", "")] = info

    picks_bans = []
    for entry in match.picks_bans:
        hero_id = entry.get("hero_id", 0)
        team = _draft_team(entry.get("team"))

        hero_info = hero_id_map.get(hero_id, {})

        # Fallback: resolve from hero_name field (Clarity outputs slug as hero_name)
        if not hero_info and entry.get("hero_name"):
            slug = entry["hero_name"]
            hero_info = hero_slug_map.get(slug, hero_slug_map.get(slug.replace("_", ""), {}))

        picks_bans.append({
            "is_pick": entry.get("is_pick", False),
            "hero_id": hero_id,
            "hero_name": hero_info.get("hero_name", entry.get("hero_name", "")),
            "hero_slug": hero_info.get("hero_slug", entry.get("hero_slug", "")),
            "team": team,
            "order": entry.get("order", 0),
        })

    return {
        "picks_bans": picks_bans,
        "has_draft": len(picks_bans) > 0,
    }


# Stands for the hero in a chat-wheel line about one ("… is missing!") when
# the source does not say which hero it was.
_UNNAMED_HERO = "…"


def _chat_wheel_fields(msg: dict, match: MatchData) -> dict:
    """What a chat-wheel message said, from the chat-wheel catalog.

    ``phrase`` is the line in both languages, ``{"en", "zh"}``; None for a
    spray and for a message id the catalog does not have. ``sound`` says
    the line was also heard (a voice line), ``spray`` that the message was a
    spray on the ground. The two lines that name a hero ("Zeus is
    missing!") get its name from the message's ``hero_id`` (replay parser);
    a source without it (OpenDota) leaves "…" in its place.
    """
    from .parser import chat_wheel_entry, hero_names

    entry = chat_wheel_entry(msg.get("key")) or {}
    phrase = None
    if entry.get("en"):
        phrase = {"en": str(entry["en"]), "zh": str(entry.get("zh") or entry["en"])}
        if any("{hero}" in text for text in phrase.values()):
            hero_id = msg.get("hero_id")
            en, zh = hero_names(hero_id) if isinstance(hero_id, int) and not isinstance(hero_id, bool) else ("", "")
            # the name the rest of the payload uses for a hero of this match
            en = next((p.hero_name for p in match.players if hero_id and p.hero_id == hero_id), en)
            names = {"en": en or _UNNAMED_HERO, "zh": zh or en or _UNNAMED_HERO}
            phrase = {lang: text.replace("{hero}", names[lang]) for lang, text in phrase.items()}
    return {"phrase": phrase, "sound": bool(entry.get("sound")), "spray": bool(entry.get("spray"))}


def analyze_chat(match: MatchData) -> dict:
    """Analyze chat messages.

    ``type`` is "chat" (``key`` is the text) or "chatwheel" (``key`` is
    "chatwheel_<id>"); a chat-wheel message also carries ``phrase``,
    ``sound`` and ``spray`` (see ``_chat_wheel_fields``).
    """
    slot_map = _slot_map(match)

    messages = []
    for msg in match.chat:
        time_val = int(msg.get("time") or 0)
        slot = _resolve_event_slot(msg, match, slot_map)
        msg_type = msg.get("type", "chat")
        key = msg.get("key", "")

        hero_name = ""
        hero_slug = ""
        player = slot_map.get(slot) if slot is not None else None
        if player:
            hero_name = player.hero_name
            hero_slug = player.hero_slug
        team = _team_for_slot(slot, slot_map)

        messages.append({
            "time": time_val,
            "time_str": format_time(time_val),
            "type": msg_type,
            "key": key,
            "player_slot": slot,
            "hero_name": hero_name,
            "hero_slug": hero_slug,
            "team": team,
            **(_chat_wheel_fields(msg, match) if msg_type == "chatwheel" else {}),
        })

    messages.sort(key=lambda m: m["time"])
    return {"messages": messages}


def full_analysis(match: MatchData) -> dict:
    """Run all analysis modules and return combined results.

    ``match_overview.pre_game_only`` is true for a truncated replay that
    ends before the horn (see ``_pre_game_only``): the frontend shows it as
    an incomplete replay instead of a 0:00 match. ``coach_available`` is
    false when the coach refuses the match (409 ``coach_no_game``): such a
    replay, or one cut within the first minutes of the game. ``position``
    of a player is ``{number, confidence, method}`` or None (see
    ``_player_positions``). ``gpm`` / ``xpm`` of a player are None for a
    truncated replay that ends within the first minute of the game.
    """
    # A truncated replay has no result: never report a guessed winner.
    radiant_win = match.radiant_win if match.winner_known else None
    positions = _player_positions(match)
    # A per-minute rate needs a minute of game: a replay cut four seconds
    # after the horn showed "GPM 621" for all ten players (the gold of those
    # seconds, extrapolated to a minute). A complete match keeps the rates
    # of its source, also when its length is unknown.
    has_rates = not (match.partial and match.duration < 60)
    return {
        "match_overview": {
            "match_id": match.match_id,
            "duration": match.duration,
            "duration_str": format_time(match.duration),
            "radiant_win": radiant_win,
            "winner": None if radiant_win is None else "radiant" if radiant_win else "dire",
            "game_mode": match.game_mode,
            "radiant_score": match.radiant_score,
            "dire_score": match.dire_score,
            "tower_status_radiant": match.tower_status_radiant,
            "tower_status_dire": match.tower_status_dire,
            "barracks_status_radiant": match.barracks_status_radiant,
            "barracks_status_dire": match.barracks_status_dire,
            "first_blood_time": match.first_blood_time,
            "start_time": match.start_time,
            "cluster": match.cluster,
            "region": match.region,
            # Full game version when it is known ("7.41b"), else the major
            # version ("7.41"), else "".
            "patch": _patch_versions(match)[1],
            "game_build": match.game_build,
            # Building positions in world units ([] when unknown) and whether
            # the replay was only partially parsed.
            "buildings": getattr(match, "buildings", []),
            "partial": getattr(match, "partial", False),
            "pre_game_only": _pre_game_only(match),
            "coach_available": _coach_available(match),
            "players": [
                {
                    "player_slot": p.player_slot,
                    "hero_id": p.hero_id,
                    "hero_name": p.hero_name,
                    "hero_slug": p.hero_slug,
                    "player_name": p.player_name,
                    "team": p.team,
                    "position": positions.get(p.player_slot),
                    "kills": p.kills,
                    "deaths": p.deaths,
                    "assists": p.assists,
                    "level": p.level,
                    "last_hits": p.last_hits,
                    "denies": p.denies,
                    "gpm": p.gold_per_min if has_rates else None,
                    "xpm": p.xp_per_min if has_rates else None,
                    "hero_damage": p.hero_damage,
                    "tower_damage": p.tower_damage,
                    "hero_healing": p.hero_healing,
                    "net_worth": p.net_worth,
                    "final_items": p.final_items,
                    "final_item_keys": p.final_item_keys,
                    "neutral_item": getattr(p, "neutral_item", ""),
                    "backpack": p.backpack,
                    "stuns": p.stuns,
                    "actions_per_min": p.actions_per_min,
                    "teamfight_participation": p.teamfight_participation,
                    "buyback_count": p.buyback_count,
                    "buyback_log": [
                        {"time": int(entry.get("time") or 0), "time_str": format_time(entry.get("time"))}
                        for entry in p.buyback_log
                    ],
                    "multi_kills": p.multi_kills,
                    "kill_streaks": p.kill_streaks,
                    "purchase_log": [
                        {
                            "time": int(entry.get("time") or 0),
                            "key": entry.get("key", ""),
                            "time_str": format_time(entry.get("time")),
                        }
                        for entry in p.purchase_log
                    ] if p.purchase_log else [
                        {"time": e.time, "key": e.item_key or e.item_name, "time_str": format_time(e.time)}
                        for e in p.items
                    ],
                }
                for p in match.players
            ],
        },
        "item_builds": analyze_item_builds(match),
        "skill_builds": analyze_skill_builds(match),
        "wards": analyze_wards(match),
        "smokes": analyze_smokes(match),
        "teamfights": analyze_teamfights(match),
        "abilities": analyze_abilities(match),
        "gold_xp_timeline": [
            {
                "time": s.time,
                "time_str": format_time(s.time),
                "radiant_gold": s.radiant_gold,
                "dire_gold": s.dire_gold,
                "radiant_xp": s.radiant_xp,
                "dire_xp": s.dire_xp,
                "gold_diff": s.radiant_gold - s.dire_gold,
                "xp_diff": s.radiant_xp - s.dire_xp,
            }
            for s in match.gold_xp_timeline
        ],
        "laning": analyze_laning(match),
        "combat": analyze_combat(match),
        "farm": analyze_farm(match),
        "purchases": analyze_purchases(match),
        "graphs": analyze_graphs(match),
        "objectives_analysis": analyze_objectives(match),
        "benchmarks_analysis": analyze_benchmarks(match),
        "draft": analyze_draft(match),
        "chat_analysis": analyze_chat(match),
    }
