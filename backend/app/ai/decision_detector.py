"""Decision detection module for AI coaching system.

Analyzes a MatchData object and a target player_slot to identify all critical
decision points in the match. The primary focus is death analysis (inferring
death timestamps from multiple data sources), but also detects bad trades,
overextensions, missed CS, good plays, and teamfight engagement quality.
"""
from __future__ import annotations

import math
from collections import defaultdict

from ..models import (
    AbilityEvent,
    LaningDamageEvent,
    LaningSnapshot,
    MatchData,
    PlayerData,
    TeamfightEvent,
    TeamfightPlayer,
)
from .models import DeathEvent, DecisionPoint


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANING_PHASE_END = 600  # 10 minutes in seconds
DEATH_DEDUP_WINDOW = 5  # seconds: two inferred deaths within this = same death
RESPAWN_HP_RATIO = 0.85  # snapshot HP jumps above this after 0 => probable respawn
LOW_HP_RATIO = 0.08  # HP ratio threshold to suspect death
OVEREXTEND_TOWER_DIST = 35.0  # map-percentage units from nearest allied tower
BAD_TRADE_HP_RATIO = 2.0  # enemy lost < 1/ratio of what player lost => bad trade
BAD_TRADE_MIN_DAMAGE = 80  # minimum damage received to consider a trade
LANING_CS_CHECK_INTERVAL = 60  # seconds between CS benchmark checks
EXPECTED_CS_PER_MIN = {1: 5.0, 2: 7.0, 3: 4.0, 4: 1.0, 0: 2.0}  # by lane_role, 0=unknown

# Approximate tower positions in map-percentage coords (0-100 range).
# These are rough positions; good enough for overextension detection.
RADIANT_TOWERS = [
    # Safe lane (bot)
    (72, 20), (78, 12), (83, 8),
    # Mid
    (48, 42), (40, 50),
    # Off lane (top)
    (18, 72), (12, 78), (8, 83),
    # Base
    (22, 22), (15, 15),
]
DIRE_TOWERS = [
    # Safe lane (top)
    (28, 80), (22, 88), (17, 92),
    # Mid
    (52, 58), (60, 50),
    # Off lane (bot)
    (82, 28), (88, 22), (92, 17),
    # Base
    (78, 78), (85, 85),
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _find_player(match: MatchData, player_slot: int) -> PlayerData | None:
    """Return the PlayerData for the given slot, or None."""
    for p in match.players:
        if p.player_slot == player_slot:
            return p
    return None


def _player_team(player_slot: int) -> str:
    """Determine team from slot number (0-4 = radiant, 128-132 = dire)."""
    return "radiant" if player_slot < 128 else "dire"


def map_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance in map-percentage space."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _nearest_allied_tower_dist(x: float, y: float, team: str) -> float:
    """Distance from (x,y) to the nearest allied tower."""
    towers = RADIANT_TOWERS if team == "radiant" else DIRE_TOWERS
    if not towers:
        return 999.0
    return min(map_dist(x, y, tx, ty) for tx, ty in towers)


def game_to_map(game_x: float, game_y: float) -> tuple[float, float]:
    """Convert game coordinates (-4000..3875) to map percentage (0..100).

    Uses the OpenDota formula: (coord + 4000) / 7937.5 * 100.
    If the input already appears to be in 0-100 range, pass through.
    """
    if 0 <= game_x <= 100 and 0 <= game_y <= 100:
        return game_x, game_y
    mx = (game_x + 4000) / 7937.5 * 100
    my = (game_y + 4000) / 7937.5 * 100
    return mx, my


# Backward-compat aliases (these helpers were previously module-private).
_dist = map_dist
_game_to_map = game_to_map


def _game_phase(time: int) -> str:
    """Return game phase string for a given time."""
    if time < 30:          # before 0:30 — rune contest / first creep wave
        return "pregame"
    elif time <= LANING_PHASE_END:
        return "laning"
    elif time <= 1500:  # 25 minutes
        return "mid_game"
    return "late_game"


def _dedup_deaths(deaths: list[DeathEvent]) -> list[DeathEvent]:
    """Remove duplicate deaths that are within DEATH_DEDUP_WINDOW of each other.

    Keeps the death with the most information (e.g. has position or killer).
    """
    if not deaths:
        return []
    deaths_sorted = sorted(deaths, key=lambda d: d.time)
    result: list[DeathEvent] = []
    for death in deaths_sorted:
        if result and abs(death.time - result[-1].time) <= DEATH_DEDUP_WINDOW:
            # Merge: keep the one with more info
            existing = result[-1]
            # Prefer the one with a known killer
            if not existing.killer_hero and death.killer_hero:
                result[-1] = death
            # Prefer the one with a position
            elif (existing.x == 0 and existing.y == 0
                  and (death.x != 0 or death.y != 0)):
                result[-1] = death
            # Prefer teamfight-flagged
            elif not existing.is_teamfight and death.is_teamfight:
                # Merge teamfight info into existing
                result[-1] = DeathEvent(
                    time=existing.time,
                    player_slot=existing.player_slot,
                    hero_name=existing.hero_name,
                    killer_hero=existing.killer_hero or death.killer_hero,
                    x=existing.x if existing.x != 0 else death.x,
                    y=existing.y if existing.y != 0 else death.y,
                    is_teamfight=True,
                    teamfight_idx=death.teamfight_idx,
                )
            # Otherwise keep existing (earlier detection is usually more accurate)
        else:
            result.append(death)
    return result


# ---------------------------------------------------------------------------
# Death Detection: multiple strategies combined
# ---------------------------------------------------------------------------

def _detect_teamfight_deaths(
    match: MatchData, player: PlayerData
) -> list[DeathEvent]:
    """Strategy 1: Extract deaths from teamfight data (most reliable)."""
    deaths: list[DeathEvent] = []
    for tf_idx, tf in enumerate(match.teamfights):
        for tfp in tf.players:
            if tfp.player_slot != player.player_slot:
                continue
            if tfp.deaths > 0 and tfp.death_time_offset is not None:
                death_time = tf.start_time + int(tfp.death_time_offset)
                # Try to find killer from kill_targets of other players
                killer = ""
                for other_tfp in tf.players:
                    if other_tfp.player_slot == player.player_slot:
                        continue
                    for kt in other_tfp.kill_targets:
                        if kt.get("target") == player.hero_name:
                            killer = other_tfp.hero_name
                            break
                    if killer:
                        break

                mx, my = game_to_map(tfp.x, tfp.y)
                deaths.append(DeathEvent(
                    time=death_time,
                    player_slot=player.player_slot,
                    hero_name=player.hero_name,
                    killer_hero=killer,
                    x=mx,
                    y=my,
                    is_teamfight=True,
                    teamfight_idx=tf_idx,
                ))
            break  # found our player in this teamfight
    return deaths


def _detect_laning_deaths(
    match: MatchData, player: PlayerData
) -> list[DeathEvent]:
    """Strategy 2: Infer deaths from laning snapshots.

    If a snapshot shows very low HP followed by a snapshot with HP jumping
    back to a high ratio (respawn), or if HP drops to 0, that indicates death.
    """
    deaths: list[DeathEvent] = []
    snapshots = player.laning_snapshots
    if len(snapshots) < 2:
        return deaths

    for i in range(len(snapshots) - 1):
        curr = snapshots[i]
        nxt = snapshots[i + 1]

        if curr.max_hp <= 0 or nxt.max_hp <= 0:
            continue

        curr_hp_ratio = curr.hp / curr.max_hp
        nxt_hp_ratio = nxt.hp / nxt.max_hp

        # Pattern: HP drops to very low, then jumps back high (died and respawned)
        is_death = False

        if curr.hp == 0:
            is_death = True
        elif (curr_hp_ratio <= LOW_HP_RATIO
              and nxt_hp_ratio >= RESPAWN_HP_RATIO
              and nxt.hp > curr.hp + curr.max_hp * 0.5):
            is_death = True

        if is_death:
            # Death probably happened just before or at this snapshot
            mx, my = game_to_map(curr.x, curr.y)
            deaths.append(DeathEvent(
                time=curr.time,
                player_slot=player.player_slot,
                hero_name=player.hero_name,
                x=mx,
                y=my,
            ))

    return deaths


def _detect_deaths_from_gold_timeline(
    match: MatchData, player: PlayerData
) -> list[DeathEvent]:
    """Strategy 3: Detect deaths from significant gold drops in gold_t.

    When a hero dies, they lose gold. A sudden drop in net worth between
    consecutive minutes is a strong signal.
    """
    deaths: list[DeathEvent] = []
    gold_t = player.gold_t
    if len(gold_t) < 2:
        return deaths

    for i in range(2, len(gold_t)):  # Start at minute 2 to avoid early noise
        gold_drop = gold_t[i - 1] - gold_t[i]
        # Minimum gold loss to suspect a death (scales with level/time)
        # At minute i, expected death cost is roughly 30 * level
        min_drop = max(50, 20 * (i + 1))
        if gold_drop >= min_drop:
            # Death happened somewhere in this minute
            death_time = i * 60  # approximate to minute boundary
            deaths.append(DeathEvent(
                time=death_time,
                player_slot=player.player_slot,
                hero_name=player.hero_name,
            ))

    return deaths


def _detect_deaths_from_ability_events(
    match: MatchData, player: PlayerData
) -> list[DeathEvent]:
    """Strategy 4: Cross-reference killed_by with ability events.

    We know from killed_by WHICH heroes killed this player and how many times.
    We look for bursts of enemy abilities targeting our hero to estimate timing.
    """
    deaths: list[DeathEvent] = []
    if not player.killed_by:
        return deaths

    # Collect all ability events where enemies target this player
    # (target field matches our hero_name)
    target_events: list[AbilityEvent] = []
    for ae in match.ability_events:
        if ae.target == player.hero_name and ae.player_slot != player.player_slot:
            target_events.append(ae)

    if not target_events:
        return deaths

    # Group by killer heroes we know about from killed_by
    killer_heroes = set(player.killed_by.keys())
    events_by_killer: dict[str, list[AbilityEvent]] = defaultdict(list)
    for ae in target_events:
        if ae.hero_name in killer_heroes:
            events_by_killer[ae.hero_name].append(ae)

    # For each killer, find clusters of events (potential kill moments)
    for killer_hero, events in events_by_killer.items():
        kill_count = player.killed_by[killer_hero]
        events_sorted = sorted(events, key=lambda e: e.time)

        # Find clusters: group events within 10 seconds
        clusters: list[list[AbilityEvent]] = []
        current_cluster: list[AbilityEvent] = []
        for evt in events_sorted:
            if current_cluster and evt.time - current_cluster[-1].time > 10:
                clusters.append(current_cluster)
                current_cluster = [evt]
            else:
                current_cluster.append(evt)
        if current_cluster:
            clusters.append(current_cluster)

        # Take the top N clusters by event density as potential kills
        clusters.sort(key=lambda c: len(c), reverse=True)
        for cluster in clusters[:kill_count]:
            # Death time ~ end of the cluster (last ability hit)
            last_event = cluster[-1]
            x, y = 0.0, 0.0
            if last_event.x is not None and last_event.y is not None:
                x, y = game_to_map(last_event.x, last_event.y)
            deaths.append(DeathEvent(
                time=last_event.time,
                player_slot=player.player_slot,
                hero_name=player.hero_name,
                killer_hero=killer_hero,
                x=x,
                y=y,
            ))

    return deaths


def _enrich_deaths_with_killed_by(
    deaths: list[DeathEvent], player: PlayerData
) -> None:
    """Fill in killer_hero on deaths that don't have it, using killed_by dict.

    We match unattributed deaths to remaining kill counts from killed_by.
    """
    if not player.killed_by:
        return

    # Count how many deaths already have each killer attributed
    attributed: dict[str, int] = defaultdict(int)
    for d in deaths:
        if d.killer_hero:
            attributed[d.killer_hero] += 1

    # Remaining kills per killer
    remaining: dict[str, int] = {}
    for killer, count in player.killed_by.items():
        left = count - attributed.get(killer, 0)
        if left > 0:
            remaining[killer] = left

    if not remaining:
        return

    # Assign remaining killers to unattributed deaths in chronological order
    unattributed = [d for d in deaths if not d.killer_hero]
    killer_list: list[str] = []
    for killer, count in remaining.items():
        killer_list.extend([killer] * count)
    # Sort killers alphabetically for determinism, then assign
    killer_list.sort()

    for i, death in enumerate(unattributed):
        if i < len(killer_list):
            death.killer_hero = killer_list[i]


def _flag_teamfight_deaths(
    deaths: list[DeathEvent], match: MatchData
) -> None:
    """Mark deaths that occur during a teamfight if not already flagged."""
    for death in deaths:
        if death.is_teamfight:
            continue
        for tf_idx, tf in enumerate(match.teamfights):
            # Consider a generous window around the teamfight
            if tf.start_time - 5 <= death.time <= tf.end_time + 5:
                death.is_teamfight = True
                death.teamfight_idx = tf_idx
                break


def _validate_death_count(
    deaths: list[DeathEvent], player: PlayerData
) -> list[DeathEvent]:
    """Ensure we don't report more deaths than the player actually had.

    If we have too many, keep the most confident ones (teamfight > laning > inferred).
    If we have too few, that's acceptable -- we report what we can find.
    """
    expected = player.deaths
    if expected <= 0 or len(deaths) <= expected:
        return deaths

    # Prioritize: teamfight deaths first, then those with killers, then others
    def _confidence(d: DeathEvent) -> int:
        score = 0
        if d.is_teamfight:
            score += 3
        if d.killer_hero:
            score += 2
        if d.x != 0 or d.y != 0:
            score += 1
        return score

    deaths.sort(key=lambda d: (-_confidence(d), d.time))
    return deaths[:expected]


# ---------------------------------------------------------------------------
# Decision Point Detection
# ---------------------------------------------------------------------------

def _detect_death_decisions(
    deaths: list[DeathEvent], player: PlayerData, lang: str = "zh"
) -> list[DecisionPoint]:
    """Create a DecisionPoint for every death."""
    decisions: list[DecisionPoint] = []
    team = _player_team(player.player_slot)

    for death in deaths:
        phase = _game_phase(death.time)
        if lang == "en":
            killer_info = f"killed by {death.killer_hero}" if death.killer_hero else "died"
        else:
            killer_info = f"被{death.killer_hero}击杀" if death.killer_hero else "阵亡"

        if death.is_teamfight:
            if lang == "en":
                brief = f"Killed by {death.killer_hero} in teamfight" if death.killer_hero else "Died in teamfight"
            else:
                brief = f"团战中{killer_info}"
        elif phase == "pregame":
            if lang == "en":
                brief = f"Killed during bounty rune contest by {death.killer_hero}" if death.killer_hero else "Died during bounty rune contest"
            else:
                killer_text = f"被{death.killer_hero}击杀" if death.killer_hero else "阵亡"
                brief = f"赏金符争夺中{killer_text}"
        elif phase == "laning":
            if lang == "en":
                brief = f"Laning phase: {killer_info}"
            else:
                brief = f"对线期{killer_info}"
        else:
            brief = f"{killer_info}"

        # Check if overextension contributed
        if death.x != 0 or death.y != 0:
            tower_dist = _nearest_allied_tower_dist(death.x, death.y, team)
            if tower_dist > OVEREXTEND_TOWER_DIST:
                if lang == "en":
                    brief += " (overextended)"
                else:
                    brief += "（位置过深）"

        decisions.append(DecisionPoint(
            time=death.time,
            point_type="death",
            player_slot=player.player_slot,
            hero_name=player.hero_name,
            severity="critical",
            brief=brief,
            related_death=death,
        ))

    return decisions


def _detect_overextend_decisions(
    deaths: list[DeathEvent], player: PlayerData, lang: str = "zh"
) -> list[DecisionPoint]:
    """Detect overextension: player died far from allied towers."""
    decisions: list[DecisionPoint] = []
    team = _player_team(player.player_slot)

    for death in deaths:
        if death.x == 0 and death.y == 0:
            continue
        tower_dist = _nearest_allied_tower_dist(death.x, death.y, team)
        if tower_dist > OVEREXTEND_TOWER_DIST:
            if lang == "en":
                brief = f"Overextended into enemy territory, ~{tower_dist:.0f} units from nearest allied tower"
            else:
                brief = f"过于深入敌方区域，距离最近友方塔约{tower_dist:.0f}单位"
            decisions.append(DecisionPoint(
                time=death.time,
                point_type="overextend",
                player_slot=player.player_slot,
                hero_name=player.hero_name,
                severity="major",
                brief=brief,
                related_death=death,
            ))

    return decisions


def _detect_bad_trades(
    match: MatchData, player: PlayerData, lang: str = "zh"
) -> list[DecisionPoint]:
    """Detect bad trades during laning: player takes significantly more damage
    than they deal in a short window."""
    decisions: list[DecisionPoint] = []
    damage_events = player.laning_damage_events
    if not damage_events:
        return decisions

    # Collect damage dealt BY player (from other players' damage events
    # where the source is our player). We'll approximate by checking
    # match-level ability events targeting the opponent during laning.
    # But laning_damage_events are damage this player DEALT, with target_idx
    # being the target. We need to figure out damage RECEIVED.
    #
    # Actually, laning_damage_events on a PlayerData is damage dealt BY that player.
    # We need to check OTHER players' laning_damage_events targeting our player.

    # Build a map of damage received by our player in time windows
    player_idx = None
    for idx, p in enumerate(match.players):
        if p.player_slot == player.player_slot:
            player_idx = idx
            break

    if player_idx is None:
        return decisions

    # Collect damage taken from enemies in 15-second windows
    damage_received_by_window: dict[int, list[LaningDamageEvent]] = defaultdict(list)
    damage_dealt_by_window: dict[int, list[LaningDamageEvent]] = defaultdict(list)

    for p in match.players:
        if p.team == player.team:
            continue
        for de in p.laning_damage_events:
            if de.target_idx == player_idx:
                window = de.time // 15
                damage_received_by_window[window].append(de)

    for de in damage_events:
        window = de.time // 15
        damage_dealt_by_window[window].append(de)

    # Analyze each window
    for window_key in sorted(damage_received_by_window.keys()):
        received = damage_received_by_window[window_key]
        dealt = damage_dealt_by_window.get(window_key, [])

        total_received = sum(de.damage for de in received)
        total_dealt = sum(de.damage for de in dealt)

        if total_received < BAD_TRADE_MIN_DAMAGE:
            continue

        # Bad trade: received much more than dealt
        if total_dealt == 0 or total_received / max(total_dealt, 1) >= BAD_TRADE_HP_RATIO:
            time_approx = received[0].time
            attacker_names = set()
            for p in match.players:
                if p.team != player.team:
                    for de in p.laning_damage_events:
                        if de.target_idx == player_idx and de.time // 15 == window_key:
                            attacker_names.add(p.hero_name)

            if lang == "en":
                attackers = ", ".join(attacker_names) if attacker_names else "enemy heroes"
                if total_dealt == 0:
                    brief = f"Took {total_received} damage from {attackers} with no effective retaliation"
                else:
                    brief = (
                        f"Unfavorable trade with {attackers}"
                        f" (took {total_received}, dealt only {total_dealt})"
                    )
            else:
                attackers = "、".join(attacker_names) if attacker_names else "敌方英雄"
                if total_dealt == 0:
                    brief = f"被{attackers}消耗{total_received}血量，未造成有效反击"
                else:
                    brief = (
                        f"与{attackers}交换血量不利"
                        f"（受到{total_received}伤害，仅造成{total_dealt}伤害）"
                    )

            decisions.append(DecisionPoint(
                time=time_approx,
                point_type="bad_trade",
                player_slot=player.player_slot,
                hero_name=player.hero_name,
                severity="major" if total_received >= 200 else "minor",
                brief=brief,
            ))

    return decisions


def _detect_missed_cs(
    match: MatchData, player: PlayerData, lang: str = "zh"
) -> list[DecisionPoint]:
    """Detect CS gaps during laning compared to expected rates."""
    decisions: list[DecisionPoint] = []
    snapshots = player.laning_snapshots
    if len(snapshots) < 2:
        return decisions

    role = player.lane_role if player.lane_role else player.lane
    expected_per_min = EXPECTED_CS_PER_MIN.get(role, 4.0)

    # Check CS at each minute mark during laning
    checked_minutes: set[int] = set()
    for snap in snapshots:
        if snap.time <= 0 or snap.time > LANING_PHASE_END:
            continue
        minute = snap.time // 60
        if minute in checked_minutes or minute < 1:
            continue
        checked_minutes.add(minute)

        expected_cs = expected_per_min * minute
        actual_cs = snap.last_hits

        # Significant gap: less than 60% of expected
        if expected_cs > 0 and actual_cs < expected_cs * 0.6:
            deficit = expected_cs - actual_cs
            decisions.append(DecisionPoint(
                time=snap.time,
                point_type="missed_cs",
                player_slot=player.player_slot,
                hero_name=player.hero_name,
                severity="minor",
                brief=(
                    f"Minute {minute}: {actual_cs} last hits, below expected {expected_cs:.0f} (deficit: {deficit:.0f})"
                    if lang == "en" else
                    f"第{minute}分钟补刀{actual_cs}个，"
                    f"低于预期{expected_cs:.0f}个（差距{deficit:.0f}个）"
                ),
            ))

    # Deduplicate: only keep the worst CS gap per 2-minute window
    if len(decisions) > 1:
        best_per_window: dict[int, DecisionPoint] = {}
        for dp in decisions:
            window = dp.time // 120
            if window not in best_per_window:
                best_per_window[window] = dp
            # keep the one with the larger deficit (longer brief text as proxy)
        decisions = list(best_per_window.values())

    return decisions


def _detect_good_plays(
    match: MatchData, player: PlayerData, lang: str = "zh"
) -> list[DecisionPoint]:
    """Detect positive plays: multi-kills, kill streaks, successful smoke ganks."""
    decisions: list[DecisionPoint] = []

    # Multi-kills (double kill, triple kill, etc.)
    if lang == "en":
        multi_kill_names = {
            "2": "Double Kill",
            "3": "Triple Kill",
            "4": "Ultra Kill",
            "5": "Rampage",
        }
    else:
        multi_kill_names = {
            "2": "双杀",
            "3": "三杀",
            "4": "四杀",
            "5": "五杀（Rampage）",
        }
    for count_str, times in player.multi_kills.items():
        if int(count_str) >= 2:
            if lang == "en":
                name = multi_kill_names.get(count_str, f"{count_str}-kill combo")
            else:
                name = multi_kill_names.get(count_str, f"{count_str}连杀")
            for _ in range(times):
                # We don't have exact timestamps for multi-kills,
                # so we won't add a time-specific decision.
                # Instead, we note it as a general positive moment.
                decisions.append(DecisionPoint(
                    time=0,  # unknown exact time
                    point_type="good_play",
                    player_slot=player.player_slot,
                    hero_name=player.hero_name,
                    severity="positive",
                    brief=f"Achieved {name}" if lang == "en" else f"完成{name}",
                ))

    # Kill streaks
    if lang == "en":
        streak_names = {
            "3": "Killing Spree",
            "4": "Dominating",
            "5": "Mega Kill",
            "6": "Unstoppable",
            "7": "Wicked Sick",
            "8": "Monster Kill",
            "9": "Godlike",
            "10": "Beyond Godlike",
        }
    else:
        streak_names = {
            "3": "大杀特杀",
            "4": "主宰比赛",
            "5": "杀人如麻",
            "6": "妖怪般的杀戮",
            "7": "超越神的杀戮",
            "8": "接近传说的杀戮",
            "9": "传说般的杀戮",
            "10": "万夫莫敌",
        }
    for streak_str, times in player.kill_streaks.items():
        streak_num = int(streak_str)
        if streak_num >= 3:
            if lang == "en":
                name = streak_names.get(streak_str, f"{streak_str} kill streak")
            else:
                name = streak_names.get(streak_str, f"{streak_str}连杀")
            for _ in range(times):
                decisions.append(DecisionPoint(
                    time=0,
                    point_type="good_play",
                    player_slot=player.player_slot,
                    hero_name=player.hero_name,
                    severity="positive",
                    brief=f"Achieved {name} streak" if lang == "en" else f"达成{name}连杀",
                ))

    # Successful smoke ganks
    team = _player_team(player.player_slot)
    for smoke in match.smokes:
        if smoke.team != team:
            continue
        if player.player_slot not in smoke.player_slots:
            continue
        if smoke.result == "kill" and smoke.kills_gained > 0:
            if lang == "en":
                brief = f"Successful smoke gank, {smoke.kills_gained} kill(s)"
            else:
                brief = f"烟雾突袭成功，击杀{smoke.kills_gained}人"
            decisions.append(DecisionPoint(
                time=smoke.time,
                point_type="good_play",
                player_slot=player.player_slot,
                hero_name=player.hero_name,
                severity="positive",
                brief=brief,
            ))
        elif smoke.result == "objective" and smoke.objective_taken:
            if lang == "en":
                obj_names = {
                    "tower": "tower",
                    "roshan": "Roshan",
                }
            else:
                obj_names = {
                    "tower": "防御塔",
                    "roshan": "肉山",
                }
            obj_name = obj_names.get(smoke.objective_taken, smoke.objective_taken)
            if lang == "en":
                brief = f"Smoke play secured {obj_name}"
            else:
                brief = f"烟雾行动成功推进{obj_name}"
            decisions.append(DecisionPoint(
                time=smoke.time,
                point_type="good_play",
                player_slot=player.player_slot,
                hero_name=player.hero_name,
                severity="positive",
                brief=brief,
            ))

    return decisions


def _detect_fight_engagement(
    match: MatchData, player: PlayerData, lang: str = "zh"
) -> list[DecisionPoint]:
    """Analyze teamfight participation quality."""
    decisions: list[DecisionPoint] = []

    for tf_idx, tf in enumerate(match.teamfights):
        our_player: TeamfightPlayer | None = None
        for tfp in tf.players:
            if tfp.player_slot == player.player_slot:
                our_player = tfp
                break

        if our_player is None:
            continue

        # Calculate participation metrics
        team = _player_team(player.player_slot)
        team_kills = 0
        team_deaths = 0
        for tfp in tf.players:
            tfp_team = _player_team(tfp.player_slot)
            if tfp_team == team:
                team_kills += tfp.kills
                team_deaths += tfp.deaths

        # Notably bad teamfight: died without contributing
        if (our_player.deaths > 0
                and our_player.kills == 0
                and our_player.damage_dealt < 200):
            dmg = our_player.damage_dealt
            if lang == "en":
                brief = f"Died in teamfight with minimal contribution ({dmg} damage, 0 kills)"
            else:
                brief = (
                    f"团战中阵亡但贡献极低"
                    f"（造成{dmg}伤害，0击杀）"
                )
            decisions.append(DecisionPoint(
                time=tf.start_time,
                point_type="fight_engage",
                player_slot=player.player_slot,
                hero_name=player.hero_name,
                severity="major",
                brief=brief,
            ))
        # Notably good teamfight: high kills without dying
        elif (our_player.kills >= 2 and our_player.deaths == 0):
            kills = our_player.kills
            if lang == "en":
                brief = f"Excellent teamfight ({kills} kills, 0 deaths)"
            else:
                brief = (
                    f"团战表现出色"
                    f"（{kills}击杀，0阵亡）"
                )
            decisions.append(DecisionPoint(
                time=tf.start_time,
                point_type="fight_engage",
                player_slot=player.player_slot,
                hero_name=player.hero_name,
                severity="positive",
                brief=brief,
            ))
        # Lost teamfight where player was key death
        elif (our_player.deaths > 0
              and team_deaths > team_kills
              and our_player.death_time_offset is not None
              and our_player.death_time_offset < 3.0):
            if lang == "en":
                brief = "Instantly killed at teamfight start — poor engagement positioning"
            else:
                brief = "团战初期即被秒杀，可能开团位置不佳"
            decisions.append(DecisionPoint(
                time=tf.start_time,
                point_type="fight_engage",
                player_slot=player.player_slot,
                hero_name=player.hero_name,
                severity="major",
                brief=brief,
            ))

    return decisions


def _detect_laning_overextend(
    match: MatchData, player: PlayerData, deaths: list[DeathEvent], lang: str = "zh"
) -> list[DecisionPoint]:
    """Detect overextension during laning from snapshots (even without a death).

    Look for moments where the player is far from their tower and at low HP.
    """
    decisions: list[DecisionPoint] = []
    snapshots = player.laning_snapshots
    if not snapshots:
        return decisions

    team = _player_team(player.player_slot)
    death_times = {d.time for d in deaths}

    for snap in snapshots:
        if snap.time > LANING_PHASE_END or snap.max_hp <= 0:
            continue

        hp_ratio = snap.hp / snap.max_hp
        mx, my = game_to_map(snap.x, snap.y)
        tower_dist = _nearest_allied_tower_dist(mx, my, team)

        # Far from tower AND low HP (but not already a death)
        if (tower_dist > OVEREXTEND_TOWER_DIST
                and hp_ratio < 0.4
                and not any(abs(snap.time - dt) <= 10 for dt in death_times)):
            if lang == "en":
                brief = f"Overextended during laning with low HP ({hp_ratio:.0%}), at risk of being killed"
            else:
                brief = f"对线期位置过深且血量偏低（{hp_ratio:.0%}HP），有被击杀风险"
            decisions.append(DecisionPoint(
                time=snap.time,
                point_type="overextend",
                player_slot=player.player_slot,
                hero_name=player.hero_name,
                severity="minor",
                brief=brief,
            ))

    # Deduplicate: at most one overextend per 30 seconds
    if len(decisions) > 1:
        deduped: list[DecisionPoint] = [decisions[0]]
        for dp in decisions[1:]:
            if dp.time - deduped[-1].time > 30:
                deduped.append(dp)
        decisions = deduped

    return decisions


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_decisions(
    match: MatchData, player_slot: int, lang: str = "zh"
) -> tuple[list[DeathEvent], list[DecisionPoint]]:
    """Analyze a match for a specific player and return all death events
    and significant decision points.

    Args:
        match: Full parsed match data.
        player_slot: The player slot to analyze (0-4 radiant, 128-132 dire).
        lang: Language for user-facing text ("zh" or "en").

    Returns:
        A tuple of (death_events, decision_points), each sorted by time.
        death_events: All inferred death events for the player.
        decision_points: All significant decision moments (includes deaths).
    """
    player = _find_player(match, player_slot)
    if player is None:
        return [], []

    # --- Phase 1: Death Detection (multiple strategies) ---
    all_deaths: list[DeathEvent] = []

    # Strategy 1: Teamfight deaths (highest confidence)
    all_deaths.extend(_detect_teamfight_deaths(match, player))

    # Strategy 2: Laning snapshot deaths
    all_deaths.extend(_detect_laning_deaths(match, player))

    # Strategy 3: Gold timeline drops
    all_deaths.extend(_detect_deaths_from_gold_timeline(match, player))

    # Strategy 4: Ability event cross-reference
    all_deaths.extend(_detect_deaths_from_ability_events(match, player))

    # Deduplicate deaths within DEATH_DEDUP_WINDOW seconds
    all_deaths = _dedup_deaths(all_deaths)

    # Enrich: fill in killer_hero where possible
    _enrich_deaths_with_killed_by(all_deaths, player)

    # Flag deaths that overlap with teamfights
    _flag_teamfight_deaths(all_deaths, match)

    # Filter out deaths before game start (artifacts)
    all_deaths = [d for d in all_deaths if d.time >= 0]

    # Validate: don't exceed known death count
    all_deaths = _validate_death_count(all_deaths, player)

    # Final sort by time
    all_deaths.sort(key=lambda d: d.time)

    # --- Phase 2: Decision Point Detection ---
    all_decisions: list[DecisionPoint] = []

    # Death decisions (one per death, severity=critical)
    all_decisions.extend(_detect_death_decisions(all_deaths, player, lang))

    # Overextension (deaths far from towers)
    all_decisions.extend(_detect_overextend_decisions(all_deaths, player, lang))

    # Bad trades during laning
    all_decisions.extend(_detect_bad_trades(match, player, lang))

    # Missed CS during laning
    all_decisions.extend(_detect_missed_cs(match, player, lang))

    # Good plays (multi-kills, streaks, successful smokes)
    all_decisions.extend(_detect_good_plays(match, player, lang))

    # Teamfight engagement quality
    all_decisions.extend(_detect_fight_engagement(match, player, lang))

    # Laning overextension (non-fatal)
    all_decisions.extend(_detect_laning_overextend(match, player, all_deaths, lang))

    # Sort all decisions by time
    all_decisions.sort(key=lambda d: d.time)

    return all_deaths, all_decisions
