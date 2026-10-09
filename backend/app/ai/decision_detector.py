"""Decision detection module for AI coaching system.

Analyzes a MatchData object and a target player_slot to identify all critical
decision points in the match. The primary focus is death analysis (inferring
death timestamps from multiple data sources), but also detects bad trades,
overextensions, missed CS, good plays, and teamfight engagement quality.

Positions are world units (``backend/app/geo.py``); tower positions come
from the replay's buildings when present (see :mod:`vision`).
"""
from __future__ import annotations

import bisect
import logging
from collections import defaultdict

from .. import geo
from ..models import (
    AbilityEvent,
    LaningDamageEvent,
    MatchData,
    PlayerData,
    TeamfightPlayer,
)
from .models import DeathEvent, DecisionPoint
from .phases import LANING_END, game_phase
from .positions import lane_role_of, player_position, role_for_position
from .vision import (
    _distance,
    estimate_position,
    has_position,
    in_own_base,
    nearest_tower_distance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANING_PHASE_END = LANING_END  # kept for backward compatibility
DEATH_DEDUP_WINDOW = 5    # seconds: two inferred deaths within this = same death
RESPAWN_HP_RATIO = 0.85   # snapshot HP jumps above this after 0 => probable respawn
LOW_HP_RATIO = 0.08       # HP ratio threshold to suspect death

# Overextension (world units): far from every standing allied tower and
# closer to the enemy's structures than to ours, or right under an enemy tower.
OVEREXTEND_TOWER_DIST = 3000.0
TOWER_DIVE_RADIUS = 900.0

BAD_TRADE_HP_RATIO = 2.0   # received >= ratio x dealt => bad trade
BAD_TRADE_MIN_DAMAGE = 80  # minimum damage received to consider a trade
BAD_TRADE_WINDOW = 15      # seconds per trade window
MAX_BAD_TRADES = 4         # keep only the worst trade episodes

# Missed CS: compared against the pro CS curve of the player's derived role
# (supports are never judged on last hits). Minutes 1-2 are too noisy.
MISSED_CS_FIRST_MINUTE = 3
MISSED_CS_RATIO = 0.6


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _find_player(match: MatchData, player_slot: int) -> PlayerData | None:
    for p in match.players:
        if p.player_slot == player_slot:
            return p
    return None


def _player_team(player_slot: int) -> str:
    """Team from slot number (0-4 = radiant, 128-132 = dire)."""
    return "radiant" if player_slot < 128 else "dire"


def _enemy_team(team: str) -> str:
    return "dire" if team == "radiant" else "radiant"


def map_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two world-space points."""
    return _distance(x1, y1, x2, y2)


def _game_phase(time: int) -> str:
    """Game phase for a time (shared definition in :mod:`phases`)."""
    return game_phase(time)


def is_overextended(match: MatchData, team: str, x: float, y: float, time_s: int) -> bool:
    """Whether a position at ``time_s`` is an overextension for ``team``.

    True when the hero stands right under an enemy tower (tower dive), or is
    more than ``OVEREXTEND_TOWER_DIST`` from every standing allied tower and
    closer to enemy structures than to allied ones. The own base never counts.
    """
    if not has_position((x, y)) or in_own_base(team, x, y):
        return False
    enemy = _enemy_team(team)
    d_ally = nearest_tower_distance(match, team, x, y, time_s)
    d_enemy = nearest_tower_distance(match, enemy, x, y, time_s)
    ex, ey = geo.ANCIENT_POSITIONS[enemy]
    d_enemy = min(d_enemy, _distance(x, y, ex, ey))
    if d_enemy <= TOWER_DIVE_RADIUS:
        return True
    return d_ally > OVEREXTEND_TOWER_DIST and d_enemy < d_ally


def _dedup_deaths(deaths: list[DeathEvent]) -> list[DeathEvent]:
    """Merge inferred deaths within DEATH_DEDUP_WINDOW of each other.

    Keeps the most informative record (killer, position, teamfight flag).
    """
    if not deaths:
        return []
    result: list[DeathEvent] = []
    for death in sorted(deaths, key=lambda d: d.time):
        if result and abs(death.time - result[-1].time) <= DEATH_DEDUP_WINDOW:
            existing = result[-1]
            merged = DeathEvent(
                time=existing.time,
                player_slot=existing.player_slot,
                hero_name=existing.hero_name,
                killer_hero=existing.killer_hero or death.killer_hero,
                x=existing.x,
                y=existing.y,
                is_teamfight=existing.is_teamfight or death.is_teamfight,
                teamfight_idx=existing.teamfight_idx if existing.is_teamfight else death.teamfight_idx,
                source=existing.source if existing.source != "gold" else death.source,
            )
            if not has_position((merged.x, merged.y)) and has_position((death.x, death.y)):
                merged.x, merged.y = death.x, death.y
            # Teamfight records carry the most precise time.
            if death.is_teamfight and not existing.is_teamfight:
                merged.time = death.time
            result[-1] = merged
        else:
            result.append(death)
    return result


# ---------------------------------------------------------------------------
# Death Detection: multiple strategies combined
# ---------------------------------------------------------------------------

def _tf_position_at(tfp: TeamfightPlayer, offset: float | None) -> tuple[float, float]:
    """Position of a teamfight participant closest to ``offset`` seconds."""
    if offset is not None and tfp.positions:
        best = min(tfp.positions, key=lambda p: abs(p.time_offset - offset))
        if has_position((best.x, best.y)):
            return (best.x, best.y)
    return (tfp.x, tfp.y)


def _detect_teamfight_deaths(match: MatchData, player: PlayerData) -> list[DeathEvent]:
    """Strategy 1: deaths recorded in teamfight data (most reliable)."""
    deaths: list[DeathEvent] = []
    for tf_idx, tf in enumerate(match.teamfights):
        for tfp in tf.players:
            if tfp.player_slot != player.player_slot:
                continue
            if tfp.deaths > 0 and tfp.death_time_offset is not None:
                death_time = tf.start_time + int(tfp.death_time_offset)
                killer = ""
                for other in tf.players:
                    if other.player_slot == player.player_slot or _player_team(other.player_slot) == player.team:
                        continue
                    if any(kt.get("target") == player.hero_name for kt in other.kill_targets):
                        killer = other.hero_name
                        break
                x, y = _tf_position_at(tfp, tfp.death_time_offset)
                deaths.append(DeathEvent(
                    time=death_time,
                    player_slot=player.player_slot,
                    hero_name=player.hero_name,
                    killer_hero=killer,
                    x=x,
                    y=y,
                    is_teamfight=True,
                    teamfight_idx=tf_idx,
                    source="teamfight",
                ))
            break  # found our player in this teamfight
    return deaths


def _detect_laning_deaths(match: MatchData, player: PlayerData) -> list[DeathEvent]:
    """Strategy 2: infer deaths from laning snapshots.

    HP at 0, or very low HP followed by a jump back to (almost) full HP
    (respawn) between consecutive snapshots.
    """
    deaths: list[DeathEvent] = []
    snapshots = player.laning_snapshots
    if len(snapshots) < 2:
        return deaths

    for curr, nxt in zip(snapshots, snapshots[1:]):
        if curr.max_hp <= 0 or nxt.max_hp <= 0:
            continue
        curr_ratio = curr.hp / curr.max_hp
        nxt_ratio = nxt.hp / nxt.max_hp
        is_death = curr.hp == 0 or (
            curr_ratio <= LOW_HP_RATIO
            and nxt_ratio >= RESPAWN_HP_RATIO
            and nxt.hp > curr.hp + curr.max_hp * 0.5
        )
        if is_death:
            deaths.append(DeathEvent(
                time=curr.time,
                player_slot=player.player_slot,
                hero_name=player.hero_name,
                x=curr.x,
                y=curr.y,
                source="laning",
            ))
    return deaths


def _detect_deaths_from_gold_timeline(match: MatchData, player: PlayerData) -> list[DeathEvent]:
    """Strategy 3: significant drops in the per-minute gold series."""
    deaths: list[DeathEvent] = []
    gold_t = player.gold_t
    if len(gold_t) < 3:
        return deaths
    for i in range(2, len(gold_t)):  # minute 2+ to avoid early noise
        gold_drop = gold_t[i - 1] - gold_t[i]
        min_drop = max(50, 20 * (i + 1))
        if gold_drop >= min_drop:
            deaths.append(DeathEvent(
                time=i * 60,  # approximated to the minute boundary
                player_slot=player.player_slot,
                hero_name=player.hero_name,
                source="gold",
            ))
    return deaths


def _detect_deaths_from_ability_events(match: MatchData, player: PlayerData) -> list[DeathEvent]:
    """Strategy 4: clusters of enemy abilities on the player, matched with
    the killers known from ``killed_by``."""
    deaths: list[DeathEvent] = []
    if not player.killed_by:
        return deaths

    killer_heroes = set(player.killed_by.keys())
    events_by_killer: dict[str, list[AbilityEvent]] = defaultdict(list)
    for ae in match.ability_events:
        if ae.target == player.hero_name and ae.player_slot != player.player_slot:
            if ae.hero_name in killer_heroes:
                events_by_killer[ae.hero_name].append(ae)

    for killer_hero, events in events_by_killer.items():
        kill_count = player.killed_by[killer_hero]
        clusters: list[list[AbilityEvent]] = []
        current: list[AbilityEvent] = []
        for evt in sorted(events, key=lambda e: e.time):
            if current and evt.time - current[-1].time > 10:
                clusters.append(current)
                current = [evt]
            else:
                current.append(evt)
        if current:
            clusters.append(current)

        clusters.sort(key=len, reverse=True)
        for cluster in clusters[:kill_count]:
            last_event = cluster[-1]
            # Prefer the victim's own position; the caster stood nearby.
            x, y = estimate_position(match, player.player_slot, last_event.time)
            if not has_position((x, y)) and last_event.x is not None and last_event.y is not None:
                x, y = float(last_event.x), float(last_event.y)
            deaths.append(DeathEvent(
                time=last_event.time,
                player_slot=player.player_slot,
                hero_name=player.hero_name,
                killer_hero=killer_hero,
                x=x,
                y=y,
                source="abilities",
            ))
    return deaths


def _enrich_deaths_with_killed_by(deaths: list[DeathEvent], player: PlayerData) -> None:
    """Fill in killer_hero on unattributed deaths using the killed_by counts."""
    if not player.killed_by:
        return
    attributed: dict[str, int] = defaultdict(int)
    for d in deaths:
        if d.killer_hero:
            attributed[d.killer_hero] += 1
    killer_list: list[str] = []
    for killer, count in player.killed_by.items():
        left = count - attributed.get(killer, 0)
        if left > 0:
            killer_list.extend([killer] * left)
    killer_list.sort()  # deterministic assignment
    unattributed = [d for d in deaths if not d.killer_hero]
    for death, killer in zip(unattributed, killer_list):
        death.killer_hero = killer


def _flag_teamfight_deaths(deaths: list[DeathEvent], match: MatchData) -> None:
    """Mark deaths that occur during a teamfight if not already flagged."""
    for death in deaths:
        if death.is_teamfight:
            continue
        for tf_idx, tf in enumerate(match.teamfights):
            if tf.start_time - 5 <= death.time <= tf.end_time + 5:
                death.is_teamfight = True
                death.teamfight_idx = tf_idx
                break


def _fill_death_positions(deaths: list[DeathEvent], match: MatchData) -> None:
    """Estimate a position for deaths that do not carry one.

    Deaths inferred from the per-minute gold series only know the minute,
    so a position guessed at the minute mark would be misleading.
    """
    for death in deaths:
        if death.source != "gold" and not has_position((death.x, death.y)):
            death.x, death.y = estimate_position(match, death.player_slot, death.time)


def _validate_death_count(deaths: list[DeathEvent], player: PlayerData) -> list[DeathEvent]:
    """Never report more deaths than the player actually had.

    Keeps the most confident ones (teamfight > known killer > known position).
    """
    expected = player.deaths
    if expected <= 0 or len(deaths) <= expected:
        return deaths

    def _confidence(d: DeathEvent) -> int:
        score = 0
        if d.is_teamfight:
            score += 3
        if d.killer_hero:
            score += 2
        if has_position((d.x, d.y)):
            score += 1
        return score

    ranked = sorted(deaths, key=lambda d: (-_confidence(d), d.time))
    return ranked[:expected]


# ---------------------------------------------------------------------------
# Decision Point Detection
# ---------------------------------------------------------------------------

def _detect_death_decisions(
    deaths: list[DeathEvent], player: PlayerData, lang: str = "zh",
) -> list[DecisionPoint]:
    """One DecisionPoint per death (overextension is noted in the brief)."""
    decisions: list[DecisionPoint] = []
    en = lang == "en"
    for death in deaths:
        phase = game_phase(death.time)
        killer = death.killer_hero
        if death.is_teamfight:
            if en:
                brief = f"Killed by {killer} in a teamfight" if killer else "Died in a teamfight"
            else:
                brief = f"团战中被{killer}击杀" if killer else "团战中阵亡"
        elif phase == "pregame":
            if en:
                brief = f"Killed by {killer} during the bounty rune contest" if killer else "Died during the bounty rune contest"
            else:
                brief = f"赏金符争夺中被{killer}击杀" if killer else "赏金符争夺中阵亡"
        elif phase == "laning":
            if en:
                brief = f"Laning phase: killed by {killer}" if killer else "Laning phase: died"
            else:
                brief = f"对线期被{killer}击杀" if killer else "对线期阵亡"
        else:
            if en:
                brief = f"Killed by {killer}" if killer else "Died"
            else:
                brief = f"被{killer}击杀" if killer else "阵亡"

        if death.overextended:
            brief += " (overextended)" if en else "（位置过深）"

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


def _detect_bad_trades(match: MatchData, player: PlayerData, lang: str = "zh") -> list[DecisionPoint]:
    """Laning trades where the player took far more hero damage than dealt.

    ``laning_damage_events`` on a PlayerData are damage dealt BY that player;
    damage received is read from the enemies' events targeting this player.
    Adjacent windows are merged into one trade episode and only the worst
    ``MAX_BAD_TRADES`` episodes are kept.
    """
    player_idx = next(
        (i for i, p in enumerate(match.players) if p.player_slot == player.player_slot),
        None,
    )
    if player_idx is None:
        return []

    received: dict[int, list[tuple[LaningDamageEvent, str]]] = defaultdict(list)
    dealt: dict[int, int] = defaultdict(int)
    for p in match.players:
        if p.team == player.team:
            continue
        for de in p.laning_damage_events:
            if de.target_idx == player_idx and de.time < LANING_END:
                received[de.time // BAD_TRADE_WINDOW].append((de, p.hero_name))
    for de in player.laning_damage_events:
        if de.time < LANING_END:
            dealt[de.time // BAD_TRADE_WINDOW] += de.damage

    # Bad windows, then merged into consecutive episodes.
    bad_windows: list[int] = []
    for window in sorted(received):
        total_received = sum(de.damage for de, _ in received[window])
        total_dealt = dealt.get(window, 0)
        if total_received < BAD_TRADE_MIN_DAMAGE:
            continue
        if total_dealt == 0 or total_received / max(total_dealt, 1) >= BAD_TRADE_HP_RATIO:
            bad_windows.append(window)

    episodes: list[list[int]] = []
    for window in bad_windows:
        if episodes and window - episodes[-1][-1] <= 1:
            episodes[-1].append(window)
        else:
            episodes.append([window])

    scored: list[tuple[int, int, int, list[str]]] = []
    for ep in episodes:
        events = [item for w in ep for item in received[w]]
        total_received = sum(de.damage for de, _ in events)
        total_dealt = sum(dealt.get(w, 0) for w in ep)
        attackers = sorted({name for _, name in events})
        start_time = min(de.time for de, _ in events)
        scored.append((total_received, total_dealt, start_time, attackers))

    scored.sort(key=lambda s: s[0], reverse=True)
    decisions: list[DecisionPoint] = []
    for total_received, total_dealt, start_time, attackers in scored[:MAX_BAD_TRADES]:
        if lang == "en":
            names = ", ".join(attackers) if attackers else "enemy heroes"
            if total_dealt == 0:
                brief = f"Took {total_received} damage from {names} with no effective retaliation"
            else:
                brief = f"Unfavorable trade with {names} (took {total_received}, dealt only {total_dealt})"
        else:
            names = "、".join(attackers) if attackers else "敌方英雄"
            if total_dealt == 0:
                brief = f"被{names}消耗{total_received}血量，未造成有效反击"
            else:
                brief = f"与{names}交换血量不利（受到{total_received}伤害，仅造成{total_dealt}伤害）"
        decisions.append(DecisionPoint(
            time=start_time,
            point_type="bad_trade",
            player_slot=player.player_slot,
            hero_name=player.hero_name,
            severity="major" if total_received >= 200 else "minor",
            brief=brief,
        ))
    decisions.sort(key=lambda d: d.time)
    return decisions


def _role_for_cs(player: PlayerData, position: int | None) -> str:
    """Role tag used for the CS benchmark ("" = do not judge CS)."""
    role = role_for_position(position)
    if role:
        return role
    return {1: "carry", 2: "mid", 3: "offlane"}.get(lane_role_of(player), "")


def _cs_at_minute(player: PlayerData, minute: int) -> int | None:
    """Last hits at an exact minute mark (snapshots, else lh_t)."""
    target = minute * 60
    snaps = player.laning_snapshots
    if snaps:
        times = [s.time for s in snaps]
        i = bisect.bisect_left(times, target)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(snaps) and (best is None or abs(snaps[j].time - target) < abs(best.time - target)):
                best = snaps[j]
        if best is not None and abs(best.time - target) <= 5:
            return best.last_hits
    if len(player.lh_t) > minute:
        return player.lh_t[minute]
    return None


def _detect_missed_cs(
    match: MatchData, player: PlayerData, lang: str = "zh", position: int | None = None,
) -> list[DecisionPoint]:
    """CS gaps during laning against the role's CS curve.

    Supports (pos 4/5) are not judged on last hits. At most one point is
    kept for the early (minutes 3-6) and one for the late (7-10) laning
    stage: the worst ratio of actual to expected CS.
    """
    role = _role_for_cs(player, position)
    if role not in ("carry", "mid", "offlane"):
        return []
    try:
        from .pro_patterns import get_cs_benchmark
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("CS benchmarks unavailable: %s", exc)
        return []

    worst: dict[str, tuple[float, DecisionPoint]] = {}
    last_minute = min(LANING_END // 60, max(0, (match.duration or LANING_END) // 60))
    for minute in range(MISSED_CS_FIRST_MINUTE, last_minute + 1):
        actual = _cs_at_minute(player, minute)
        if actual is None:
            continue
        expected = get_cs_benchmark(player.hero_name, role, minute)
        if expected <= 0 or actual >= expected * MISSED_CS_RATIO:
            continue
        ratio = actual / expected
        deficit = expected - actual
        dp = DecisionPoint(
            time=minute * 60,
            point_type="missed_cs",
            player_slot=player.player_slot,
            hero_name=player.hero_name,
            severity="minor",
            brief=(
                f"Minute {minute}: {actual} last hits, below the expected {expected:.0f} (deficit {deficit:.0f})"
                if lang == "en" else
                f"第{minute}分钟补刀{actual}个，低于预期{expected:.0f}个（差距{deficit:.0f}个）"
            ),
        )
        bucket = "early" if minute <= 6 else "late"
        if bucket not in worst or ratio < worst[bucket][0]:
            worst[bucket] = (ratio, dp)
    return sorted((dp for _ratio, dp in worst.values()), key=lambda d: d.time)


def _detect_good_plays(match: MatchData, player: PlayerData, lang: str = "zh") -> list[DecisionPoint]:
    """Positive plays: multi-kills, kill streaks, successful smoke ganks."""
    decisions: list[DecisionPoint] = []
    en = lang == "en"

    multi_kill_names = (
        {"2": "Double Kill", "3": "Triple Kill", "4": "Ultra Kill", "5": "Rampage"}
        if en else
        {"2": "双杀", "3": "三杀", "4": "四杀", "5": "五杀（Rampage）"}
    )
    for count_str, times in player.multi_kills.items():
        try:
            count = int(count_str)
        except (TypeError, ValueError):
            continue
        if count < 2:
            continue
        name = multi_kill_names.get(str(count), f"{count}-kill combo" if en else f"{count}连杀")
        for _ in range(int(times or 0)):
            # Multi-kill counters carry no timestamp (time=0 = unknown).
            decisions.append(DecisionPoint(
                time=0,
                point_type="good_play",
                player_slot=player.player_slot,
                hero_name=player.hero_name,
                severity="positive",
                brief=f"Achieved a {name}" if en else f"完成{name}",
            ))

    streak_names = (
        {"3": "Killing Spree", "4": "Dominating", "5": "Mega Kill", "6": "Unstoppable",
         "7": "Wicked Sick", "8": "Monster Kill", "9": "Godlike", "10": "Beyond Godlike"}
        if en else
        {"3": "大杀特杀", "4": "主宰比赛", "5": "杀人如麻", "6": "无人能挡",
         "7": "变态杀戮", "8": "妖怪般的杀戮", "9": "如同神一般", "10": "超神"}
    )
    for streak_str, times in player.kill_streaks.items():
        try:
            streak = int(streak_str)
        except (TypeError, ValueError):
            continue
        if streak < 3:
            continue
        name = streak_names.get(str(streak), f"{streak} kill streak" if en else f"{streak}连杀")
        for _ in range(int(times or 0)):
            decisions.append(DecisionPoint(
                time=0,
                point_type="good_play",
                player_slot=player.player_slot,
                hero_name=player.hero_name,
                severity="positive",
                brief=f"Achieved a {name} streak" if en else f"达成{name}",
            ))

    team = _player_team(player.player_slot)
    obj_names = (
        {"tower": "a tower", "barracks": "barracks", "ancient": "the Ancient", "roshan": "Roshan"}
        if en else
        {"tower": "防御塔", "barracks": "兵营", "ancient": "遗迹", "roshan": "肉山"}
    )
    for smoke in match.smokes:
        if smoke.team != team or player.player_slot not in smoke.player_slots:
            continue
        if smoke.result == "kill" and smoke.kills_gained > 0:
            brief = (f"Successful smoke gank: {smoke.kills_gained} kill(s)" if en
                     else f"烟雾突袭成功，击杀{smoke.kills_gained}人")
        elif smoke.result == "objective" and smoke.objective_taken:
            obj = obj_names.get(smoke.objective_taken, smoke.objective_taken)
            brief = f"Smoke play secured {obj}" if en else f"烟雾行动成功拿下{obj}"
        else:
            continue
        decisions.append(DecisionPoint(
            time=smoke.time,
            point_type="good_play",
            player_slot=player.player_slot,
            hero_name=player.hero_name,
            severity="positive",
            brief=brief,
        ))
    return decisions


def _detect_fight_engagement(match: MatchData, player: PlayerData, lang: str = "zh") -> list[DecisionPoint]:
    """Teamfight participation quality."""
    decisions: list[DecisionPoint] = []
    en = lang == "en"
    team = _player_team(player.player_slot)

    for tf in match.teamfights:
        ours = next((tfp for tfp in tf.players if tfp.player_slot == player.player_slot), None)
        if ours is None:
            continue
        team_kills = sum(t.kills for t in tf.players if _player_team(t.player_slot) == team)
        team_deaths = sum(t.deaths for t in tf.players if _player_team(t.player_slot) == team)

        if ours.deaths > 0 and ours.kills == 0 and ours.damage_dealt < 200:
            dmg = ours.damage_dealt
            brief = (f"Died in a teamfight with minimal contribution ({dmg} damage, 0 kills)" if en
                     else f"团战中阵亡但贡献极低（造成{dmg}伤害，0击杀）")
            severity = "major"
        elif ours.kills >= 2 and ours.deaths == 0:
            brief = (f"Excellent teamfight ({ours.kills} kills, 0 deaths)" if en
                     else f"团战表现出色（{ours.kills}击杀，0阵亡）")
            severity = "positive"
        elif (ours.deaths > 0 and team_deaths > team_kills
              and ours.death_time_offset is not None and ours.death_time_offset < 3.0):
            brief = ("Killed instantly at the start of a lost teamfight — poor engagement positioning" if en
                     else "团战初期即被秒杀，开团站位可能不佳")
            severity = "major"
        else:
            continue
        decisions.append(DecisionPoint(
            time=tf.start_time,
            point_type="fight_engage",
            player_slot=player.player_slot,
            hero_name=player.hero_name,
            severity=severity,
            brief=brief,
        ))
    return decisions


def _detect_laning_overextend(
    match: MatchData, player: PlayerData, deaths: list[DeathEvent], lang: str = "zh",
) -> list[DecisionPoint]:
    """Non-fatal laning overextensions: low HP in an overextended spot.

    Moments within 10 s of a death are skipped (the death covers them);
    at most one point per 30 seconds.
    """
    decisions: list[DecisionPoint] = []
    team = _player_team(player.player_slot)
    death_times = [d.time for d in deaths]

    for snap in player.laning_snapshots:
        if snap.time >= LANING_END or snap.max_hp <= 0 or snap.hp <= 0:
            continue
        hp_ratio = snap.hp / snap.max_hp
        if hp_ratio >= 0.4 or any(abs(snap.time - dt) <= 10 for dt in death_times):
            continue
        if not is_overextended(match, team, snap.x, snap.y, snap.time):
            continue
        if decisions and snap.time - decisions[-1].time <= 30:
            continue
        brief = (
            f"Overextended during laning with low HP ({hp_ratio:.0%}) — at risk of being killed"
            if lang == "en" else
            f"对线期位置过深且血量偏低（{hp_ratio:.0%}），有被击杀风险"
        )
        decisions.append(DecisionPoint(
            time=snap.time,
            point_type="overextend",
            player_slot=player.player_slot,
            hero_name=player.hero_name,
            severity="minor",
            brief=brief,
        ))
    return decisions


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_deaths(match: MatchData, player: PlayerData) -> list[DeathEvent]:
    """All inferred deaths of ``player``, deduplicated and sorted by time."""
    all_deaths: list[DeathEvent] = []
    all_deaths.extend(_detect_teamfight_deaths(match, player))
    all_deaths.extend(_detect_laning_deaths(match, player))
    all_deaths.extend(_detect_deaths_from_gold_timeline(match, player))
    all_deaths.extend(_detect_deaths_from_ability_events(match, player))

    all_deaths = _dedup_deaths(all_deaths)
    _enrich_deaths_with_killed_by(all_deaths, player)
    _flag_teamfight_deaths(all_deaths, match)
    all_deaths = [d for d in all_deaths if d.time >= 0]
    all_deaths = _validate_death_count(all_deaths, player)
    _fill_death_positions(all_deaths, match)
    for death in all_deaths:
        death.overextended = is_overextended(match, player.team, death.x, death.y, death.time)
    all_deaths.sort(key=lambda d: d.time)
    return all_deaths


def detect_decisions(
    match: MatchData, player_slot: int, lang: str = "zh", position: int | None = None,
) -> tuple[list[DeathEvent], list[DecisionPoint]]:
    """Analyze a match for one player.

    Args:
        match: Full parsed match data.
        player_slot: The player slot to analyze (0-4 radiant, 128-132 dire).
        lang: Language for user-facing text ("zh" or "en").
        position: Derived position 1-5 (see :mod:`positions`); derived here
            when omitted.

    Returns:
        (death_events, decision_points), each sorted by time. Every death
        yields exactly one "death" decision point.
    """
    player = _find_player(match, player_slot)
    if player is None:
        return [], []
    if position is None:
        pos = player_position(match, player_slot)
        position = pos.number if pos else None

    all_deaths = detect_deaths(match, player)

    all_decisions: list[DecisionPoint] = []
    all_decisions.extend(_detect_death_decisions(all_deaths, player, lang))
    all_decisions.extend(_detect_bad_trades(match, player, lang))
    all_decisions.extend(_detect_missed_cs(match, player, lang, position=position))
    all_decisions.extend(_detect_good_plays(match, player, lang))
    all_decisions.extend(_detect_fight_engagement(match, player, lang))
    all_decisions.extend(_detect_laning_overextend(match, player, all_deaths, lang))

    all_decisions.sort(key=lambda d: d.time)
    return all_deaths, all_decisions
