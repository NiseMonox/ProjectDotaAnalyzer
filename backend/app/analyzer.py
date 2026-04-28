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

from pathlib import Path

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


def analyze_laning(match: MatchData) -> dict:
    """Analyze laning phase data for all players."""
    # OpenDota's lane_role=4 is officially "Jungle", but in modern Dota
    # (7.20+) dedicated jungling is dead — that bucket is mostly roaming
    # supports / pushed-out offlaners. is_roaming below still overrides.
    LANE_NAMES = {1: "安全路", 2: "中路", 3: "劣势路", 4: "游走"}

    players = []
    for p in match.players:
        lane = p.lane or 0
        lane_role = p.lane_role or 0
        lane_name = LANE_NAMES.get(lane, "未知")
        if p.is_roaming:
            lane_name = "游走"

        lh_ten = p.lh_t[10] if len(p.lh_t) > 10 else p.lh_t[-1] if p.lh_t else 0
        dn_ten = 0  # denies timeline not available; use 0
        gold_ten = p.gold_t[10] if len(p.gold_t) > 10 else p.gold_t[-1] if p.gold_t else 0
        xp_ten = p.xp_t[10] if len(p.xp_t) > 10 else p.xp_t[-1] if p.xp_t else 0

        # Computed laning stats
        total_damage_taken_laning = sum(
            e.damage for other_p in match.players
            for e in other_p.laning_damage_events
            if e.target_idx == p.player_slot
        )

        players.append({
            "player_slot": p.player_slot,
            "hero_name": p.hero_name,
            "hero_slug": p.hero_slug,
            "team": p.team,
            "lane": lane,
            "lane_role": lane_role,
            "lane_name": lane_name,
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
            "total_damage_taken_laning": total_damage_taken_laning,
            "regen_items_used": len(p.laning_regen_events),
        })

    return {"players": players}


def analyze_combat(match: MatchData) -> dict:
    """Analyze kill and damage matrices between players."""
    heroes = [p.hero_name for p in match.players]
    slug_to_idx = {}
    for i, p in enumerate(match.players):
        slug_to_idx[p.hero_slug] = i

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
    """Analyze farming patterns, gold/xp sources, and last hit timelines."""
    GOLD_REASON_NAMES = {
        "0": "其他", "1": "死亡", "2": "买活", "6": "出售",
        "11": "建筑", "12": "英雄", "13": "小兵", "14": "肉山", "15": "信使",
    }
    XP_REASON_NAMES = {
        "1": "英雄", "2": "小兵", "3": "肉山", "5": "建筑", "4": "其他",
    }

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

        # Map gold reason keys to Chinese names
        reasons_mapped = {}
        for key, amount in p.gold_reasons.items():
            name = GOLD_REASON_NAMES.get(str(key), f"未知({key})")
            reasons_mapped[name] = reasons_mapped.get(name, 0) + amount
        gold_reasons_list.append({**base, "reasons": reasons_mapped})

        # Map xp reason keys to Chinese names
        xp_mapped = {}
        for key, amount in p.xp_reasons.items():
            name = XP_REASON_NAMES.get(str(key), f"未知({key})")
            xp_mapped[name] = xp_mapped.get(name, 0) + amount
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
            time_val = entry.get("time", 0)
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

        for m_start in range(0, max_time // 60 + 5, 5):
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


def analyze_objectives(match: MatchData) -> dict:
    """Analyze objectives (towers, rax, roshan) and rune pickups."""
    RUNE_NAMES = {
        "0": "双倍", "1": "急速", "2": "幻象", "3": "隐身",
        "4": "恢复", "5": "奥术", "6": "赏金", "7": "圣水",
    }

    # Build slug-to-player lookup
    slot_to_player = {p.player_slot: p for p in match.players}

    objectives = []
    for obj in match.objectives:
        time_val = obj.get("time", 0)
        obj_type = obj.get("type", "")
        slot = obj.get("player_slot", obj.get("slot"))
        unit = obj.get("unit", "")
        key = obj.get("key", "")

        # Determine team
        team = obj.get("team")
        if team is None and slot is not None:
            team = "radiant" if slot < 128 else "dire"
        elif team is None and unit:
            if "goodguys" in unit or "radiant" in unit.lower():
                team = "radiant"
            elif "badguys" in unit or "dire" in unit.lower():
                team = "dire"

        # Map team int to string if needed
        if isinstance(team, int):
            team = "radiant" if team == 2 else "dire" if team == 3 else None

        hero_name = ""
        hero_slug = ""
        if slot is not None:
            player = slot_to_player.get(slot)
            if player:
                hero_name = player.hero_name
                hero_slug = player.hero_slug

        objectives.append({
            "time": time_val,
            "time_str": format_time(time_val),
            "type": obj_type,
            "team": team,
            "key": key or unit,
            "hero_name": hero_name,
            "hero_slug": hero_slug,
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
        rune_counts = {}
        for key, count in p.runes.items():
            rune_name = RUNE_NAMES.get(str(key), f"未知({key})")
            rune_counts[rune_name] = count
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
    """Load hero_id -> {name, slug} from static heroes.json."""
    import json
    heroes_path = Path(__file__).parent.parent.parent / "frontend" / "static" / "data" / "heroes.json"
    if not heroes_path.exists():
        return {}
    with open(heroes_path) as f:
        data = json.load(f)
    result = {}
    for hid, info in data.get("by_id", {}).items():
        result[int(hid)] = {"hero_name": info.get("name", ""), "hero_slug": info.get("slug", "")}
    return result


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
    slot_to_player = {p.player_slot: p for p in match.players}

    messages = []
    for msg in match.chat:
        time_val = msg.get("time", 0)
        slot = msg.get("player_slot", msg.get("slot"))
        msg_type = msg.get("type", "chat")
        key = msg.get("key", "")

        hero_name = ""
        hero_slug = ""
        team = None
        if slot is not None:
            player = slot_to_player.get(slot)
            if player:
                hero_name = player.hero_name
                hero_slug = player.hero_slug
                team = player.team
            else:
                team = "radiant" if slot < 128 else "dire"

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
                    "backpack": p.backpack,
                    "purchase_log": [
                        {"time": entry.get("time", 0), "key": entry.get("key", ""), "time_str": format_time(entry.get("time", 0))}
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
