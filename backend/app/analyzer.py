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
passed through unchanged in world units (see ``geo.py``).
"""
from __future__ import annotations

from .models import MatchData, PlayerData

# Natural ward lifetimes (seconds). Sentries last 7 min in current patches;
# using the shorter value never mistakes a natural expiry for a deward.
OBSERVER_LIFETIME_S = 360
SENTRY_LIFETIME_S = 420

# PlayerData.lane_role (relative, OpenDota convention) → lane code.
# lane_role=4 is OpenDota's "jungle" bucket; in modern Dota that is mostly
# roaming supports / pushed-out offlaners, so it maps to "roaming".
LANE_ROLE_CODES = {1: "safe", 2: "mid", 3: "off", 4: "roaming"}
# PlayerData.lane (absolute map lane, OpenDota convention) → side code.
LANE_SIDE_CODES = {1: "bottom", 2: "mid", 3: "top", 4: "jungle", 5: "jungle"}

# Source 2 EDOTA_GoldModifyReason (same numbering in OpenDota and Clarity).
GOLD_REASON_CODES = {
    "0": "other", "1": "death", "2": "buyback", "3": "purchase", "4": "purchase",
    "5": "abandon", "6": "sell", "7": "ability_cost", "8": "cheat",
    "9": "selection_penalty", "10": "passive", "11": "building", "12": "hero",
    "13": "creep", "14": "neutral", "15": "roshan", "16": "courier",
    "17": "bounty_rune", "18": "shared", "19": "ability", "20": "ward",
    "21": "courier",
}
# Source 2 EDOTA_ModifyXP_Reason.
XP_REASON_CODES = {
    "0": "other", "1": "hero", "2": "creep", "3": "roshan",
    "4": "tome", "5": "outpost", "6": "shared",
}
# DOTA_RUNES enum (OpenDota runes / runes_log keys).
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
            "item_phases": item_phases,
            "total_items": len(player.items),
            "net_worth": player.net_worth,
        })
    return results


def analyze_skill_builds(match: MatchData) -> list[dict]:
    """Analyze skill progression for all players."""
    results = []
    for player in match.players:
        skill_order = []
        for skill in player.skills:
            skill_order.append({
                "ability": skill.ability_name,
                "ability_key": skill.ability_key,
                "level": skill.level,
                "time": skill.time,
                "time_str": format_time(skill.time),
                "is_talent": skill.ability_name.startswith("talent_"),
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


def analyze_wards(match: MatchData) -> dict:
    """Analyze ward placements for both teams."""
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
        })

    # Advanced ward statistics
    def _avg_lifetime(wards_list):
        lifetimes = [w["lifetime"] for w in wards_list if w["lifetime"] is not None]
        return round(sum(lifetimes) / len(lifetimes), 1) if lifetimes else 0

    def _deward_count(wards_list, max_life):
        # Destroyed before natural expiry (5 s tolerance) = dewarded. Wards
        # "destroyed" by the game ending are not dewards.
        game_end = match.duration - 1
        return sum(
            1 for w in wards_list
            if w["lifetime"] is not None
            and w["lifetime"] < max_life - 5
            and (match.duration <= 0 or w["destroyed_time"] < game_end)
        )

    def _active(wards_list, max_life, t0, t1):
        return sum(
            1 for w in wards_list
            if w["time"] <= t1 and w["time"] + (w["lifetime"] if w["lifetime"] is not None else max_life) >= t0
        )

    # Ward density per 5-minute phase
    ward_density = []
    for m_start in range(0, max(0, match.duration) // 60 + 1, 5):
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
            "dewarded_obs": _deward_count(obs, OBSERVER_LIFETIME_S),
            "dewarded_sent": _deward_count(sent, SENTRY_LIFETIME_S),
        }

    return {
        "radiant": _team_stats(radiant_obs, radiant_sent),
        "dire": _team_stats(dire_obs, dire_sent),
        "ward_density": ward_density,
        "duration": match.duration,
        "all_wards": all_wards,
    }


def analyze_smokes(match: MatchData) -> dict:
    """Analyze smoke usage for both teams."""
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

        smoke_info = {
            "time": smoke.time,
            "time_str": format_time(smoke.time),
            "team": smoke.team,
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


def analyze_teamfights(match: MatchData) -> list[dict]:
    """Analyze teamfight data."""
    slot_map = _slot_map(match)
    results = []
    for i, tf in enumerate(match.teamfights, 1):
        duration = tf.end_time - tf.start_time
        winner = "radiant" if tf.dire_deaths > tf.radiant_deaths else "dire"
        if tf.dire_deaths == tf.radiant_deaths:
            winner = "draw"

        players_detail = []
        for tp in tf.players:
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
                "abilities_used": tp.abilities_used,
                "item_uses": tp.item_uses,
                "buybacks": tp.buybacks,
                "death_time_offset": tp.death_time_offset,
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
                ],
            })

        # Build kill timeline sorted by time
        kill_timeline = []
        for p in players_detail:
            if p["deaths"] > 0 and p["death_time_offset"] is not None:
                victim_names = {p["hero_name"], p["hero_slug"]} - {""}
                killer_name = next(
                    (
                        k["hero_name"]
                        for k in players_detail
                        for kt in k["kill_targets"]
                        if kt.get("target") in victim_names
                        and kt.get("time_offset") is not None
                        and abs(kt["time_offset"] - p["death_time_offset"]) < 0.5
                    ),
                    None,
                )
                kill_timeline.append({
                    "victim": p["hero_name"],
                    "victim_slug": p["hero_slug"],
                    "victim_team": p["team"],
                    "killer": killer_name,
                    "time_offset": p["death_time_offset"],
                })
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
            "players": players_detail,
            "kill_timeline": kill_timeline,
        })
    return results


def analyze_abilities(match: MatchData) -> dict:
    """Analyze ability usage patterns."""
    by_player: dict[int, list[dict]] = {}
    by_ability: dict[str, int] = {}

    for event in match.ability_events:
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


def analyze_laning(match: MatchData) -> dict:
    """Analyze laning phase data for all players.

    ``lane_name`` is a code: "safe" | "mid" | "off" | "roaming" | "unknown"
    (from the relative lane role); ``lane_side`` is the absolute map lane:
    "bottom" | "mid" | "top" | "jungle" | "unknown".
    """
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
            lh_ten = snap.last_hits if snap else sum(1 for c in p.laning_creep_kills if c.time <= 600)
        if snap is not None:
            dn_ten = snap.denies
        else:
            dn_ten = sum(1 for d in p.laning_denies if d.time <= 600)
        gold_ten = _at_ten_minutes(p.gold_t)
        if gold_ten is None:
            gold_ten = snap.net_worth if snap else 0
        xp_ten = _at_ten_minutes(p.xp_t) or 0

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
            "lane_efficiency": p.lane_efficiency,
            "lane_efficiency_pct": p.lane_efficiency_pct,
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

    return {"players": players}


def analyze_combat(match: MatchData) -> dict:
    """Analyze kill and damage matrices between players."""
    heroes = [p.hero_name for p in match.players]
    slug_to_idx = {}
    for i, p in enumerate(match.players):
        # Fall back to hero_name when hero_slug is empty (possible on
        # ClarityParser fallback, where hero_name holds the internal name)
        # so matrix rows are never silently dropped.
        key = p.hero_slug or p.hero_name
        if key:
            slug_to_idx.setdefault(key, i)

    num = len(match.players)

    # Kill matrix
    kill_matrix = [[0] * num for _ in range(num)]
    for i, p in enumerate(match.players):
        for unit_name, count in p.killed.items():
            # unit_name is like "npc_dota_hero_antimage"
            slug = unit_name.replace("npc_dota_hero_", "")
            j = slug_to_idx.get(slug)
            if j is not None:
                kill_matrix[i][j] = count

    # Damage matrix
    damage_matrix = [[0] * num for _ in range(num)]
    for i, p in enumerate(match.players):
        for unit_name, amount in p.damage.items():
            slug = unit_name.replace("npc_dota_hero_", "")
            j = slug_to_idx.get(slug)
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
    }


def analyze_farm(match: MatchData) -> dict:
    """Analyze farming patterns, gold/xp sources, and last hit timelines.

    ``reasons`` dicts are keyed by codes from GOLD_REASON_CODES /
    XP_REASON_CODES (unknown numeric reasons become ``unknown_<n>``).
    """

    unit_kills = []
    gold_reasons_list = []
    xp_reasons_list = []
    lh_timeline = []

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
            "lane_kills": p.lane_kills,
            "neutral_kills": p.neutral_kills,
            "ancient_kills": p.ancient_kills,
            "tower_kills": p.tower_kills,
            "courier_kills": p.courier_kills,
            "roshans_killed": p.roshans_killed,
        })

        reasons_mapped: dict[str, int] = {}
        for key, amount in p.gold_reasons.items():
            code = _code_for(key, GOLD_REASON_CODES)
            reasons_mapped[code] = reasons_mapped.get(code, 0) + amount
        gold_reasons_list.append({**base, "reasons": reasons_mapped})

        xp_mapped: dict[str, int] = {}
        for key, amount in p.xp_reasons.items():
            code = _code_for(key, XP_REASON_CODES)
            xp_mapped[code] = xp_mapped.get(code, 0) + amount
        xp_reasons_list.append({**base, "reasons": xp_mapped})

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

        # Group into phases: starting (-90s to 0s) + 5-minute phases
        max_time = match.duration
        phases: dict[str, list] = {}

        # Starting items: purchased during strategy phase (-90s to 0s)
        starting_items = [
            item for item in purchase_log if -90 <= item["time"] <= 0
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
    Rune counts are keyed by RUNE_CODES codes.
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

        objectives.append({
            "time": time_val,
            "time_str": format_time(time_val),
            "type": obj.get("type", ""),
            "team": team,
            "key": key or unit,
            "hero_name": player.hero_name if player else "",
            "hero_slug": player.hero_slug if player else "",
        })

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


def analyze_draft(match: MatchData) -> dict:
    """Analyze draft picks and bans."""
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
        team_val = entry.get("team", 0)
        team = "radiant" if team_val == 0 else "dire"

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


def analyze_chat(match: MatchData) -> dict:
    """Analyze chat messages."""
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
        })

    messages.sort(key=lambda m: m["time"])
    return {"messages": messages}


def full_analysis(match: MatchData) -> dict:
    """Run all analysis modules and return combined results."""
    return {
        "match_overview": {
            "match_id": match.match_id,
            "duration": match.duration,
            "duration_str": format_time(match.duration),
            "radiant_win": match.radiant_win,
            "game_mode": match.game_mode,
            "radiant_score": match.radiant_score,
            "dire_score": match.dire_score,
            "tower_status_radiant": match.tower_status_radiant,
            "tower_status_dire": match.tower_status_dire,
            "barracks_status_radiant": match.barracks_status_radiant,
            "barracks_status_dire": match.barracks_status_dire,
            "first_blood_time": match.first_blood_time,
            "start_time": match.start_time,
            "region": match.region,
            "patch": match.patch,
            # Building positions in world units ([] when unknown) and whether
            # the replay was only partially parsed.
            "buildings": getattr(match, "buildings", []),
            "partial": getattr(match, "partial", False),
            "players": [
                {
                    "player_slot": p.player_slot,
                    "hero_id": p.hero_id,
                    "hero_name": p.hero_name,
                    "hero_slug": p.hero_slug,
                    "player_name": p.player_name,
                    "team": p.team,
                    "kills": p.kills,
                    "deaths": p.deaths,
                    "assists": p.assists,
                    "level": p.level,
                    "last_hits": p.last_hits,
                    "denies": p.denies,
                    "gpm": p.gold_per_min,
                    "xpm": p.xp_per_min,
                    "hero_damage": p.hero_damage,
                    "tower_damage": p.tower_damage,
                    "hero_healing": p.hero_healing,
                    "net_worth": p.net_worth,
                    "final_items": p.final_items,
                    "final_item_keys": p.final_item_keys,
                    "neutral_item": getattr(p, "neutral_item", ""),
                    "backpack": p.backpack,
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
