"""Analysis modules for Dota 2 match data.

Provides higher-level analysis on top of parsed match data:
- Item build analysis with timing benchmarks
- Skill build analysis
- Ward coverage analysis
- Smoke effectiveness analysis
- Teamfight analysis
- Ability usage patterns
"""
from __future__ import annotations

from .models import MatchData


def format_time(seconds: int) -> str:
    """Convert seconds to MM:SS format."""
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    return f"{sign}{seconds // 60}:{seconds % 60:02d}"


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
    radiant_obs = []
    radiant_sent = []
    dire_obs = []
    dire_sent = []

    for ward in match.wards:
        player = next(
            (p for p in match.players if p.player_slot == ward.player_slot), None
        )
        hero_name = player.hero_name if player else "Unknown"

        ward_info = {
            "x": ward.x,
            "y": ward.y,
            "time": ward.time,
            "time_str": format_time(ward.time),
            "player_slot": ward.player_slot,
            "hero_name": hero_name,
            "survived": ward.destroyed_time is None,
            "lifetime": (ward.destroyed_time - ward.time) if ward.destroyed_time else None,
        }

        team = "radiant" if ward.player_slot < 128 else "dire"
        if ward.ward_type == "observer":
            (radiant_obs if team == "radiant" else dire_obs).append(ward_info)
        else:
            (radiant_sent if team == "radiant" else dire_sent).append(ward_info)

    # Advanced ward statistics
    def _avg_lifetime(wards_list):
        lifetimes = [w["lifetime"] for w in wards_list if w["lifetime"] is not None]
        return round(sum(lifetimes) / max(1, len(lifetimes)), 1) if lifetimes else 0

    def _deward_count(wards_list, ward_type="observer"):
        # Observer wards last 360s max, sentries 480s.
        # If destroyed before natural expiry, it was dewarded.
        max_life = 360 if ward_type == "observer" else 480
        return sum(
            1 for w in wards_list
            if w["lifetime"] is not None and w["lifetime"] < max_life - 5  # 5s tolerance
        )

    # Ward density per 5-minute phase
    all_wards_flat = radiant_obs + radiant_sent + dire_obs + dire_sent
    ward_density = []
    for m_start in range(0, match.duration // 60 + 1, 5):
        t0, t1 = m_start * 60, (m_start + 5) * 60
        r_active = sum(1 for w in radiant_obs + radiant_sent
                       if w["time"] <= t1 and (w["time"] + (w["lifetime"] or 360)) >= t0)
        d_active = sum(1 for w in dire_obs + dire_sent
                       if w["time"] <= t1 and (w["time"] + (w["lifetime"] or 360)) >= t0)
        ward_density.append({
            "phase": f"{m_start}-{m_start + 5}",
            "radiant": r_active,
            "dire": d_active,
        })

    return {
        "radiant": {
            "observers": radiant_obs,
            "sentries": radiant_sent,
            "total_obs": len(radiant_obs),
            "total_sent": len(radiant_sent),
            "avg_obs_lifetime": _avg_lifetime(radiant_obs),
            "avg_sent_lifetime": _avg_lifetime(radiant_sent),
            "dewarded_obs": _deward_count(radiant_obs, "observer"),
            "dewarded_sent": _deward_count(radiant_sent, "sentry"),
        },
        "dire": {
            "observers": dire_obs,
            "sentries": dire_sent,
            "total_obs": len(dire_obs),
            "total_sent": len(dire_sent),
            "avg_obs_lifetime": _avg_lifetime(dire_obs),
            "avg_sent_lifetime": _avg_lifetime(dire_sent),
            "dewarded_obs": _deward_count(dire_obs, "observer"),
            "dewarded_sent": _deward_count(dire_sent, "sentry"),
        },
        "ward_density": ward_density,
        "duration": match.duration,
        "all_wards": [
            {
                "ward_type": w.ward_type,
                "x": w.x,
                "y": w.y,
                "time": w.time,
                "time_str": format_time(w.time),
                "player_slot": w.player_slot,
                "team": "radiant" if w.player_slot < 128 else "dire",
                "hero_name": next(
                    (p.hero_name for p in match.players if p.player_slot == w.player_slot),
                    "Unknown",
                ),
                "destroyed_time": w.destroyed_time,
            }
            for w in match.wards
        ],
    }


def analyze_smokes(match: MatchData) -> dict:
    """Analyze smoke usage for both teams."""
    radiant_smokes = []
    dire_smokes = []

    for smoke in match.smokes:
        participants = []
        for slot in smoke.player_slots:
            player = next((p for p in match.players if p.player_slot == slot), None)
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
        else:
            dire_smokes.append(smoke_info)

    def _smoke_stats(smoke_list):
        total = len(smoke_list)
        kills = sum(1 for s in smoke_list if s["result"] == "kill")
        objectives = sum(1 for s in smoke_list if s["result"] == "objective")
        traded = sum(1 for s in smoke_list if s["result"] == "traded")
        no_contact = sum(1 for s in smoke_list if s["result"] == "no_contact")
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
        }

    return {
        "radiant": _smoke_stats(radiant_smokes),
        "dire": _smoke_stats(dire_smokes),
        "timeline": sorted(
            radiant_smokes + dire_smokes, key=lambda s: s["time"]
        ),
    }


def analyze_teamfights(match: MatchData) -> list[dict]:
    """Analyze teamfight data."""
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
                "team": "radiant" if tp.player_slot < 128 else "dire",
                "x": tp.x,
                "y": tp.y,
                "damage_dealt": tp.damage_dealt,
                "damage_taken": tp.damage_taken,
                "kills": tp.kills,
                "deaths": tp.deaths,
                "abilities_used": tp.abilities_used,
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
                killer_name = None
                for k in players_detail:
                    for kt in k["kill_targets"]:
                        if kt["target"] == p["hero_name"] and abs(kt["time_offset"] - p["death_time_offset"]) < 0.5:
                            killer_name = k["hero_name"]
                            break
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
    }
