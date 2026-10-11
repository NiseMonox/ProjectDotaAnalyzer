"""Decision detection module for AI coaching system.

Analyzes a MatchData object and a target player_slot to identify all critical
decision points in the match. The primary focus is death analysis, but it
also detects bad trades, overextensions, missed CS, good plays, and teamfight
engagement quality.

Deaths come from ``MatchData.kills_log``: one death per real kill-log entry
of the player, with the real killer, position, assists and time. Only a
source without a kill log (legacy data, an unparsed OpenDota match) falls
back to inferring deaths from teamfight records, laning snapshots and
ability events; a death found that way keeps an empty killer unless the data
proves who it was.

Positions are world units (``backend/app/geo.py``); tower positions come
from the replay's buildings when present (see :mod:`vision`).
"""
from __future__ import annotations

import bisect
import logging
from collections import Counter, defaultdict

from .. import geo
from ..models import (
    AbilityEvent,
    KillEvent,
    LaningDamageEvent,
    MatchData,
    PlayerData,
    TeamfightEvent,
    TeamfightPlayer,
)
from . import data_manager
from .models import DeathEvent, DecisionPoint, killed_by_fountain, killer_text
from .phases import LANING_END, counted_kills, game_phase
from .positions import lane_role_of, player_position, role_for_position
from .vision import (
    _distance,
    area_code,
    estimate_position,
    has_position,
    in_own_base,
    nearest_tower_distance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEATH_DEDUP_WINDOW = 5    # seconds: two inferred deaths within this = same death
RESPAWN_HP_RATIO = 0.85   # snapshot HP jumps above this after 0 => probable respawn
LOW_HP_RATIO = 0.08       # HP ratio threshold to suspect death

# Overextension (world units): far from every standing allied tower and
# closer to the enemy's structures than to ours, or right under an enemy tower.
OVEREXTEND_TOWER_DIST = 3000.0
# "Under an enemy tower": a tower attacks within 700 units; the margin covers
# a death point a step outside that range (a modelling choice, not a game value).
TOWER_DIVE_RADIUS = 900.0

# A kill-log death is part of a teamfight when it falls into the fight's time
# window and another hero died within this distance during that window.
# Fight windows are clusters of deaths in time only: a pick-off in another
# lane at the same moment is not a teamfight death.
TEAMFIGHT_DEATH_RADIUS = 3000.0

# "First to die, burst down": the last position_log sample 3-8 s before the
# death still showed at least this HP percentage.
BURST_HP_PCT = 70
LOW_FIGHT_DAMAGE = 200    # hero damage in a whole fight below this is "almost none"

BAD_TRADE_HP_RATIO = 2.0   # received >= ratio x dealt => bad trade
BAD_TRADE_MIN_DAMAGE = 80  # minimum damage received per window to consider a trade
BAD_TRADE_WINDOW = 15      # seconds per trade window
BAD_TRADE_HP_FRACTION = 0.30  # an episode must cost at least this share of max HP
BAD_TRADE_DEATH_GRACE = 15    # a death this soon after an episode still belongs to it
MAX_BAD_TRADES = 4         # keep only the worst trade episodes

# Missed CS: compared against the pro CS curve of the player's derived role
# (supports are never judged on last hits). Minutes 1-2 are too noisy.
MISSED_CS_FIRST_MINUTE = 3
MISSED_CS_RATIO = 0.6

_MULTI_KILL_WINDOW_S = 18  # the game's rule: a kill this soon after the previous one extends a multi-kill


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


def _hero_text(name: str, lang: str) -> str:
    """A hero name as written in text (never a raw code such as "hero_150")."""
    return data_manager.hero_display_name(name, lang) or name


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


# ---------------------------------------------------------------------------
# Deaths from the kill log (primary source)
# ---------------------------------------------------------------------------

def _fight_of_kill(match: MatchData, kill: KillEvent, kill_log: list[KillEvent]) -> int:
    """Index of the teamfight a kill-log death belongs to, -1 for none.

    The death must fall into the fight window (padded windows of two fights
    can overlap: the one the death is deepest inside wins) and, when the
    positions are known, another hero must have died within
    ``TEAMFIGHT_DEATH_RADIUS`` during that window.
    """
    best_idx, best_depth = -1, -1.0
    for idx, tf in enumerate(match.teamfights):
        if not (tf.start_time - 1 < kill.time < tf.end_time + 1):
            continue
        depth = min(kill.time - tf.start_time, tf.end_time - kill.time)
        if depth > best_depth:
            best_idx, best_depth = idx, depth
    if best_idx < 0 or kill.x is None or kill.y is None:
        return best_idx
    tf = match.teamfights[best_idx]
    others = [
        k for k in kill_log
        if k is not kill and k.x is not None and k.y is not None
        and tf.start_time - 1 < k.time < tf.end_time + 1
    ]
    if not others:
        return best_idx  # nothing to compare with: trust the time window
    if any(_distance(kill.x, kill.y, k.x, k.y) <= TEAMFIGHT_DEATH_RADIUS for k in others):
        return best_idx
    return -1


def _deaths_from_kills_log(match: MatchData, player: PlayerData) -> list[DeathEvent]:
    """One DeathEvent per real kill-log entry of the player (exact data)."""
    kill_log = counted_kills(match)
    names = {p.player_slot: p.hero_name for p in match.players}
    deaths: list[DeathEvent] = []
    for kill in kill_log:
        if kill.victim_slot != player.player_slot:
            continue
        killer_slot = kill.killer_slot if kill.killer_slot in names else None
        suicide = killer_slot == player.player_slot
        tf_idx = _fight_of_kill(match, kill, kill_log)
        deaths.append(DeathEvent(
            time=int(kill.time),
            player_slot=player.player_slot,
            hero_name=player.hero_name,
            killer_hero=names[killer_slot] if killer_slot is not None and not suicide else "",
            killer_slot=killer_slot,
            killer_unit=kill.killer_unit or "",
            assist_slots=list(kill.assist_slots),
            denied=bool(kill.deny) or suicide,
            confirmed=bool(kill.confirmed),
            x=float(kill.x) if kill.x is not None and kill.y is not None else 0.0,
            y=float(kill.y) if kill.x is not None and kill.y is not None else 0.0,
            is_teamfight=tf_idx >= 0,
            teamfight_idx=tf_idx,
            source="kills_log",
        ))
    return deaths


# ---------------------------------------------------------------------------
# Death heuristics (only for matches without a kill log)
# ---------------------------------------------------------------------------

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
                killer_slot=existing.killer_slot if existing.killer_hero else death.killer_slot,
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
                killer, killer_slot = "", None
                for other in tf.players:
                    if other.player_slot == player.player_slot or _player_team(other.player_slot) == player.team:
                        continue
                    if any(kt.get("target") == player.hero_name for kt in other.kill_targets):
                        killer, killer_slot = other.hero_name, other.player_slot
                        break
                x, y = _tf_position_at(tfp, tfp.death_time_offset)
                deaths.append(DeathEvent(
                    time=death_time,
                    player_slot=player.player_slot,
                    hero_name=player.hero_name,
                    killer_hero=killer,
                    killer_slot=killer_slot,
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

    A death is the snapshot where HP turns 0 (a dead hero stays at 0 HP
    until the respawn, so later 0-HP snapshots are the same death), or a
    very low HP followed by a jump back to (almost) full HP when the death
    itself fell between two snapshots.
    """
    deaths: list[DeathEvent] = []
    snapshots = player.laning_snapshots
    if len(snapshots) < 2:
        return deaths

    prev = None
    for curr, nxt in zip(snapshots, snapshots[1:]):
        was_dead = prev is not None and prev.max_hp > 0 and prev.hp <= 0
        prev = curr
        if curr.max_hp <= 0 or nxt.max_hp <= 0:
            continue
        if curr.hp <= 0:
            is_death = not was_dead
        else:
            is_death = (
                curr.hp / curr.max_hp <= LOW_HP_RATIO
                and nxt.hp / nxt.max_hp >= RESPAWN_HP_RATIO
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


def _killers_by_hero(match: MatchData, player: PlayerData) -> dict[str, tuple[int, int | None]]:
    """``killed_by`` as {display name: (count, player_slot)} for heroes of the match."""
    slots = {p.hero_name: p.player_slot for p in match.players if p.team != player.team}
    out: dict[str, tuple[int, int | None]] = {}
    for key, count in (player.killed_by or {}).items():
        name = data_manager.hero_display_name(key, "en")
        if not name:
            continue
        prev = out.get(name, (0, None))[0]
        out[name] = (prev + int(count or 0), slots.get(name))
    return out


def _detect_deaths_from_ability_events(match: MatchData, player: PlayerData) -> list[DeathEvent]:
    """Strategy 4: clusters of enemy abilities on the player, matched with
    the killers known from ``killed_by``."""
    deaths: list[DeathEvent] = []
    killers = _killers_by_hero(match, player)
    if not killers:
        return deaths

    events_by_killer: dict[str, list[AbilityEvent]] = defaultdict(list)
    for ae in match.ability_events:
        if ae.target == player.hero_name and ae.player_slot != player.player_slot:
            if ae.hero_name in killers:
                events_by_killer[ae.hero_name].append(ae)

    for killer_hero, events in events_by_killer.items():
        kill_count, killer_slot = killers[killer_hero]
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
                killer_slot=killer_slot,
                x=x,
                y=y,
                source="abilities",
            ))
    return deaths


def _enrich_deaths_with_killed_by(deaths: list[DeathEvent], match: MatchData, player: PlayerData) -> None:
    """Name the killer of unattributed deaths when ``killed_by`` proves it.

    That is only the case when a single hero is left over after the deaths
    that already have a killer, every death of the player had a hero killer,
    and that hero has at least as many kills left as there are unattributed
    deaths. Anything else stays empty: pairing deaths with killers by order
    would be a guess.
    """
    killers = _killers_by_hero(match, player)
    if not killers or sum(count for count, _slot in killers.values()) < player.deaths:
        return
    attributed: Counter[str] = Counter(d.killer_hero for d in deaths if d.killer_hero)
    left = {name: count - attributed.get(name, 0) for name, (count, _slot) in killers.items()}
    left = {name: count for name, count in left.items() if count > 0}
    unattributed = [d for d in deaths if not d.killer_hero]
    if len(left) != 1 or not unattributed:
        return
    (name, count), = left.items()
    if len(unattributed) > count:
        return
    for death in unattributed:
        death.killer_hero = name
        death.killer_slot = killers[name][1]


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


def _deaths_from_heuristics(match: MatchData, player: PlayerData) -> list[DeathEvent]:
    """Inferred deaths for a match without a kill log."""
    all_deaths: list[DeathEvent] = []
    all_deaths.extend(_detect_teamfight_deaths(match, player))
    all_deaths.extend(_detect_laning_deaths(match, player))
    all_deaths.extend(_detect_deaths_from_gold_timeline(match, player))
    all_deaths.extend(_detect_deaths_from_ability_events(match, player))

    all_deaths = _dedup_deaths(all_deaths)
    _enrich_deaths_with_killed_by(all_deaths, match, player)
    _flag_teamfight_deaths(all_deaths, match)
    all_deaths = [d for d in all_deaths if d.time >= 0]
    all_deaths = _validate_death_count(all_deaths, player)
    _fill_death_positions(all_deaths, match)
    return all_deaths


# ---------------------------------------------------------------------------
# Decision Point Detection
# ---------------------------------------------------------------------------

def _fight_record(tf: TeamfightEvent, player_slot: int) -> TeamfightPlayer | None:
    return next((tfp for tfp in tf.players if tfp.player_slot == player_slot), None)


def _fight_lost(tf: TeamfightEvent, team: str) -> bool:
    ours = tf.radiant_deaths if team == "radiant" else tf.dire_deaths
    theirs = tf.dire_deaths if team == "radiant" else tf.radiant_deaths
    return ours > theirs


def _hp_pct_before(player: PlayerData, time_s: int) -> int | None:
    """HP percentage 3-8 s before ``time_s`` from the position log (None = unknown)."""
    log = player.position_log
    if not log:
        return None
    i = bisect.bisect_right(log, [time_s - 3, float("inf")]) - 1
    if i < 0:
        return None
    sample = log[i]
    if len(sample) < 4 or sample[0] < time_s - 8 or sample[3] <= 0:
        return None
    return int(sample[3])


def _teamfight_death_note(match: MatchData, player: PlayerData, death: DeathEvent, lang: str) -> str:
    """Facts about the player's part in the fight he died in ("" = nothing notable).

    These used to be separate "fight_engage" points for the same moment as
    the death; they are a remark on the death instead, and only when the
    data supports them.
    """
    if not (0 <= death.teamfight_idx < len(match.teamfights)):
        return ""
    tf = match.teamfights[death.teamfight_idx]
    en = lang == "en"

    # First to die and burst down, in a fight the team lost. Needs the kill
    # log (who died first) and the position log (HP shortly before).
    if death.source == "kills_log" and _fight_lost(tf, player.team):
        first = min(
            (k.time for k in counted_kills(match) if tf.start_time - 1 < k.time < tf.end_time + 1),
            default=None,
        )
        hp = _hp_pct_before(player, death.time)
        if first is not None and int(first) >= death.time and hp is not None and hp >= BURST_HP_PCT:
            return (f"first to die in a lost fight, from {hp}% HP within seconds" if en
                    else f"在这场输掉的团战中最先阵亡，数秒内从{hp}%血量被击杀")

    record = _fight_record(tf, player.player_slot)
    if (record is not None and record.kills == 0 and record.healing < LOW_FIGHT_DAMAGE
            and record.damage_dealt < LOW_FIGHT_DAMAGE):
        dmg = record.damage_dealt
        if dmg <= 0:
            return "dealt no hero damage in the fight" if en else "本次团战未造成英雄伤害"
        return (f"dealt only {dmg} hero damage in the fight" if en
                else f"本次团战仅造成{dmg}点英雄伤害")
    return ""


def _death_brief(match: MatchData, player: PlayerData, death: DeathEvent, lang: str) -> str:
    en = lang == "en"
    phase = game_phase(death.time)
    killer = _hero_text(death.killer_hero, lang)
    # A unit as it is written inside a sentence ("a tower"), like the analysis.
    unit = "" if killer else killer_text(death, lang)

    if killed_by_fountain(death):
        # The fountain says all there is to say: no fight or lane framing.
        return f"Killed by {unit}" if en else f"被{unit}击杀"

    if death.denied:
        if death.killer_slot == death.player_slot:
            brief = "Died to their own ability or item" if en else "死于自己的技能或物品"
        elif en:
            brief = f"Denied by ally {killer}" if killer else "Denied by an ally"
        else:
            brief = f"被队友{killer}反补" if killer else "被队友反补"
    elif death.is_teamfight:
        if en:
            brief = (f"Killed by {killer} in a teamfight" if killer
                     else f"Killed by {unit} during a teamfight" if unit else "Died in a teamfight")
        else:
            brief = (f"团战中被{killer}击杀" if killer
                     else f"团战中被{unit}击杀" if unit else "团战中阵亡")
    elif phase == "pregame":
        # What a level-1 fight was about (a rune, a lane) is not in the data.
        if en:
            brief = (f"Killed by {killer} before the creeps met" if killer
                     else f"Killed by {unit} before the creeps met" if unit else "Died before the creeps met")
        else:
            brief = (f"兵线接触前被{killer}击杀" if killer
                     else f"兵线接触前被{unit}击杀" if unit else "兵线接触前阵亡")
    elif phase == "laning":
        if en:
            brief = (f"Laning phase: killed by {killer}" if killer
                     else f"Laning phase: killed by {unit}" if unit else "Laning phase: died")
        else:
            brief = (f"对线期被{killer}击杀" if killer
                     else f"对线期被{unit}击杀" if unit else "对线期阵亡")
    else:
        if en:
            brief = f"Killed by {killer}" if killer else f"Killed by {unit}" if unit else "Died"
        else:
            brief = f"被{killer}击杀" if killer else f"被{unit}击杀" if unit else "阵亡"

    notes: list[str] = []
    if death.is_teamfight and not death.denied:
        note = _teamfight_death_note(match, player, death, lang)
        if note:
            notes.append(note)
    if death.overextended:
        notes.append("overextended" if en else "位置过深")
    if notes:
        brief += f" ({'; '.join(notes)})" if en else f"（{'；'.join(notes)}）"
    return brief


def _detect_death_decisions(
    match: MatchData, deaths: list[DeathEvent], player: PlayerData, lang: str = "zh",
) -> list[DecisionPoint]:
    """One DecisionPoint per death (fight and overextension notes go into the brief)."""
    return [
        DecisionPoint(
            time=death.time,
            point_type="death",
            player_slot=player.player_slot,
            hero_name=player.hero_name,
            severity="critical",
            brief=_death_brief(match, player, death, lang),
            related_death=death,
        )
        for death in deaths
    ]


def _max_hp_at(player: PlayerData, time_s: int) -> int:
    """Max HP from the laning snapshot nearest to ``time_s`` (0 = unknown)."""
    snaps = [s for s in player.laning_snapshots if s.max_hp > 0]
    if not snaps:
        return 0
    return min(snaps, key=lambda s: abs(s.time - time_s)).max_hp


def _detect_bad_trades(
    match: MatchData, player: PlayerData, deaths: list[DeathEvent], lang: str = "zh",
) -> list[DecisionPoint]:
    """Laning trades the player clearly lost.

    ``laning_damage_events`` on a PlayerData are damage dealt BY that player;
    damage received is read from the enemies' events targeting this player.
    A 15 s window is bad when the player took far more hero damage than he
    dealt; adjacent bad windows form one episode. An episode only counts when
    it cost at least ``BAD_TRADE_HP_FRACTION`` of the player's max HP (small
    exchanges happen in every lane), did not end in the player's death (the
    death is analysed on its own) and did not end in the death of one of the
    attackers (that exchange was not lost). Only the worst
    ``MAX_BAD_TRADES`` are kept.
    """
    player_idx = next(
        (i for i, p in enumerate(match.players) if p.player_slot == player.player_slot),
        None,
    )
    if player_idx is None:
        return []

    received: dict[int, list[tuple[LaningDamageEvent, PlayerData]]] = defaultdict(list)
    dealt: dict[int, int] = defaultdict(int)
    for p in match.players:
        if p.team == player.team:
            continue
        for de in p.laning_damage_events:
            if de.target_idx == player_idx and de.time < LANING_END:
                received[de.time // BAD_TRADE_WINDOW].append((de, p))
    for de in player.laning_damage_events:
        if de.time < LANING_END and de.target_idx != player_idx:
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

    death_times = [d.time for d in deaths]
    kill_log = counted_kills(match)
    scored: list[tuple[int, int, int, list[str]]] = []
    for ep in episodes:
        events = [item for w in ep for item in received[w]]
        total_received = sum(de.damage for de, _ in events)
        total_dealt = sum(dealt.get(w, 0) for w in ep)
        start_time = min(de.time for de, _ in events)
        end_time = (ep[-1] + 1) * BAD_TRADE_WINDOW + BAD_TRADE_DEATH_GRACE
        if any(start_time <= dt <= end_time for dt in death_times):
            continue
        attacker_slots = {p.player_slot for _, p in events}
        if any(k.victim_slot in attacker_slots and start_time <= k.time <= end_time for k in kill_log):
            continue
        max_hp = _max_hp_at(player, start_time)
        if max_hp > 0 and total_received < max_hp * BAD_TRADE_HP_FRACTION:
            continue
        attackers = sorted({_hero_text(p.hero_name, lang) for _, p in events})
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
    """Last hits at an exact minute mark (lh_t, else the nearest snapshot).

    Same definition as the benchmark comparison and the laning review.
    """
    if len(player.lh_t) > minute:
        return player.lh_t[minute]
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
    return None


def _detect_missed_cs(
    match: MatchData, player: PlayerData, lang: str = "zh", position: int | None = None,
) -> list[DecisionPoint]:
    """CS gaps during laning against the role's CS curve.

    Supports (pos 4/5) are not judged on last hits, and game modes the
    benchmarks were not made for (Turbo, ...) are skipped. At most one point
    is kept for the early (minutes 3-6) and one for the late (7-10) laning
    stage: the worst ratio of actual to expected CS. A half the player
    finished at or above the curve gives no point (a slow first wave that
    was made up for is not a CS problem).
    """
    role = _role_for_cs(player, position)
    if role not in ("carry", "mid", "offlane"):
        return []
    try:
        from .pro_patterns import benchmarks_apply, get_cs_benchmark
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("CS benchmarks unavailable: %s", exc)
        return []
    if not benchmarks_apply(match.game_mode):
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
    for bucket, end_minute in (("early", min(6, last_minute)), ("late", last_minute)):
        at_end = _cs_at_minute(player, end_minute)
        if bucket in worst and at_end is not None and at_end >= get_cs_benchmark(player.hero_name, role, end_minute):
            del worst[bucket]
    return sorted((dp for _ratio, dp in worst.values()), key=lambda d: d.time)


# ---------------------------------------------------------------------------
# Good plays
# ---------------------------------------------------------------------------

def _kill_sequences(
    match: MatchData, player_slot: int, death_first: bool | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Finished multi-kills and kill streaks of a player from the kill log.

    Returns ``(multi_kills, streaks)`` as lists of (length, time of the last
    kill). Same rules as ``parser.kill_announcements``: a kill within 18 s of
    the previous one extends a multi-kill (from 2), a kill without dying in
    between extends a streak (from 3). Each sequence is listed once, at its
    final length.

    ``death_first`` settles the order of the player's own death and a kill
    of his with the same timestamp (OpenDota rounds to seconds): None keeps
    the order of the log.
    """
    multi: list[tuple[int, int]] = []
    streaks: list[tuple[int, int]] = []
    streak = chain = 0
    last_kill: float | None = None
    kill_log = counted_kills(match)
    if death_first is not None:
        kill_log = sorted(kill_log, key=lambda k: (k.time, (k.victim_slot == player_slot) != death_first))

    def close_chain() -> None:
        if chain >= 2 and last_kill is not None:
            multi.append((chain, int(last_kill)))

    def close_streak() -> None:
        if streak >= 3 and last_kill is not None:
            streaks.append((streak, int(last_kill)))

    for kill in kill_log:
        if kill.victim_slot == player_slot:
            close_streak()
            streak = 0
        if kill.killer_slot != player_slot or kill.deny:
            continue
        if last_kill is not None and kill.time - last_kill <= _MULTI_KILL_WINDOW_S:
            chain += 1
        else:
            close_chain()
            chain = 1
        streak += 1
        last_kill = kill.time
    close_chain()
    close_streak()
    return multi, streaks


def _finished_counts(counter: dict, minimum: int) -> Counter:
    """Sequences per final length from the game's announcement counters.

    The counters count every level a sequence reached (a triple kill is also
    counted as a double kill), so the number that ended at length n is
    count[n] - count[n + 1].
    """
    levels: dict[int, int] = {}
    for key, times in (counter or {}).items():
        try:
            levels[int(key)] = int(times or 0)
        except (TypeError, ValueError):
            continue
    finished: Counter = Counter()
    for level, times in levels.items():
        ended = times - levels.get(level + 1, 0)
        if level >= minimum and ended > 0:
            finished[level] = ended
    return finished


def _sequence_points(
    derived: list[tuple[int, int]], counter: dict, minimum: int, has_counters: bool,
) -> list[tuple[int, int | None]]:
    """(length, time or None) for every finished sequence.

    The game's own counters decide how many sequences of each length there
    were; the kill log supplies the time when it shows exactly that many.
    Without counters in the match the kill log alone is used.
    """
    by_length: dict[int, list[int]] = defaultdict(list)
    for length, time_s in derived:
        by_length[length].append(time_s)
    if not has_counters:
        return [(length, t) for length, times in sorted(by_length.items()) for t in times]
    points: list[tuple[int, int | None]] = []
    for length, count in sorted(_finished_counts(counter, minimum).items()):
        times = by_length.get(length, [])
        if len(times) == count:
            points.extend((length, t) for t in times)
        else:
            points.extend((length, None) for _ in range(count))
    return points


def _detect_good_plays(match: MatchData, player: PlayerData, lang: str = "zh") -> list[DecisionPoint]:
    """Positive plays: multi-kills, kill streaks, successful smoke ganks.

    A multi-kill or streak is one point at its final length (a triple kill is
    not also a double kill), timed from the kill log; when the log cannot
    place it, the point carries ``time_known=False``.
    """
    decisions: list[DecisionPoint] = []
    en = lang == "en"

    def add(brief: str, time_s: int | None, subtype: str) -> None:
        decisions.append(DecisionPoint(
            time=time_s if time_s is not None else 0,
            point_type="good_play",
            player_slot=player.player_slot,
            hero_name=player.hero_name,
            severity="positive",
            brief=brief,
            time_known=time_s is not None,
            subtype=subtype,
        ))

    has_counters = any(p.multi_kills or p.kill_streaks for p in match.players)
    multi: list[tuple[int, int]] = []
    streaks: list[tuple[int, int]] = []
    if match.kills_log:
        # A kill and the player's own death in the same second can be read
        # both ways: take the reading the game's counters confirm.
        wanted = (_finished_counts(player.multi_kills, 2), _finished_counts(player.kill_streaks, 3))
        for death_first in (None, False, True):
            multi, streaks = _kill_sequences(match, player.player_slot, death_first)
            found = (Counter(n for n, _t in multi), Counter(n for n, _t in streaks))
            if not has_counters or found == wanted:
                break
        else:
            multi, streaks = _kill_sequences(match, player.player_slot)

    multi_kill_names = (
        {2: "Double Kill", 3: "Triple Kill", 4: "Ultra Kill", 5: "Rampage"}
        if en else
        {2: "双杀", 3: "三杀", 4: "疯狂杀戮", 5: "暴走"}     # the game's own announcements
    )
    def a(name: str) -> str:
        return f"an {name}" if name[:1] in "AEIOU" else f"a {name}"

    for count, time_s in _sequence_points(multi, player.multi_kills, 2, has_counters):
        name = multi_kill_names.get(count, f"{count}-kill combo" if en else f"{count}连杀")
        add(f"Achieved {a(name)}" if en else f"完成{name}", time_s, "multi_kill")

    streak_names = (
        {3: "Killing Spree", 4: "Dominating", 5: "Mega Kill", 6: "Unstoppable",
         7: "Wicked Sick", 8: "Monster Kill", 9: "Godlike", 10: "Beyond Godlike"}
        if en else
        {3: "大杀特杀", 4: "主宰比赛", 5: "杀人如麻", 6: "无人能挡",
         7: "变态杀戮", 8: "妖怪般的杀戮", 9: "如同神一般", 10: "超越神的杀戮"}
    )
    for streak, time_s in _sequence_points(streaks, player.kill_streaks, 3, has_counters):
        name = streak_names[min(streak, 10)]
        add(f"Achieved {a(name)} streak ({streak} kills without dying)" if en
            else f"达成{name}（连续{streak}次击杀未阵亡）", time_s, "kill_streak")

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
            n = smoke.kills_gained
            brief = (f"Successful smoke gank: {n} {'kill' if n == 1 else 'kills'}" if en
                     else f"开雾抓人成功，击杀{n}人")
        elif smoke.result == "objective" and smoke.objective_taken:
            obj = obj_names.get(smoke.objective_taken, smoke.objective_taken)
            brief = f"Smoke play secured {obj}" if en else f"开雾后拿下{obj}"
        else:
            continue
        add(brief, smoke.time, "smoke")
    return decisions


def _detect_fight_engagement(
    match: MatchData, player: PlayerData, deaths: list[DeathEvent], lang: str = "zh",
    multi_kill_times: tuple[int, ...] = (),
) -> list[DecisionPoint]:
    """Teamfight participation quality.

    One event is one decision point. A fight in which the player has a death
    decision gets no second point for the same moment: what there is to say
    about his part in it is a note on that death (see
    ``_teamfight_death_note``). A good fight that already is a multi-kill
    point (``multi_kill_times``) is not listed again either.
    """
    decisions: list[DecisionPoint] = []
    en = lang == "en"
    died_in = {d.teamfight_idx for d in deaths if d.teamfight_idx >= 0}
    kill_log = counted_kills(match)

    for tf_idx, tf in enumerate(match.teamfights):
        ours = _fight_record(tf, player.player_slot)
        if ours is None:
            continue
        has_death_point = tf_idx in died_in or any(
            tf.start_time - 5 <= d.time <= tf.end_time + 5 for d in deaths
        )
        # The window is padded around the deaths: date the point at the
        # player's first kill in it when the kill log has it.
        time_s = tf.start_time
        if ours.kills >= 2 and ours.deaths == 0:
            if any(tf.start_time <= t <= tf.end_time for t in multi_kill_times):
                continue
            own_kills = [
                k.time for k in kill_log
                if k.killer_slot == player.player_slot and not k.deny
                and tf.start_time - 1 < k.time < tf.end_time + 1
            ]
            if own_kills:
                time_s = int(min(own_kills))
            brief = (f"Excellent teamfight ({ours.kills} kills, 0 deaths)" if en
                     else f"团战表现出色（{ours.kills}击杀，0阵亡）")
            severity = "positive"
        elif (ours.deaths > 0 and not has_death_point and ours.kills == 0
              and ours.damage_dealt < LOW_FIGHT_DAMAGE):
            # A death no other source located: this is its only record.
            dmg = ours.damage_dealt
            brief = (f"Died in a teamfight with minimal contribution ({dmg} damage, 0 kills)" if en
                     else f"团战中阵亡但贡献极低（造成{dmg}伤害，0击杀）")
            severity = "major"
        else:
            continue
        decisions.append(DecisionPoint(
            time=time_s,
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
    at most one point per 30 seconds. A hero standing in his own lane is not
    flagged unless he is right under an enemy tower: creep waves are not in
    the data, and where they meet (for an offlaner: close to the enemy
    tower) is where a laner belongs, low on HP or not.
    """
    decisions: list[DecisionPoint] = []
    team = _player_team(player.player_slot)
    death_times = [d.time for d in deaths]
    own_lane = {1: "bot", 2: "mid", 3: "top"}.get(player.lane)

    for snap in player.laning_snapshots:
        if snap.time >= LANING_END or snap.max_hp <= 0 or snap.hp <= 0:
            continue
        hp_ratio = snap.hp / snap.max_hp
        if hp_ratio >= 0.4 or any(abs(snap.time - dt) <= 10 for dt in death_times):
            continue
        if not is_overextended(match, team, snap.x, snap.y, snap.time):
            continue
        kind, _side, lane = area_code(snap.x, snap.y)
        if kind == "lane" and lane == own_lane and nearest_tower_distance(
                match, _enemy_team(team), snap.x, snap.y, snap.time) > TOWER_DIVE_RADIUS:
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
    """All deaths of ``player``, sorted by time.

    From the kill log when the match has one (``source="kills_log"``: exact
    time, killer, position, assists; an OpenDota entry that may be a
    reincarnation only counts while the scoreboard has a death left for it).
    Otherwise inferred by the heuristics, at most ``player.deaths`` of them.

    ``overextended`` is only set for a death outside a teamfight whose
    position is known.
    """
    if match.kills_log:
        all_deaths = _deaths_from_kills_log(match, player)
    else:
        all_deaths = _deaths_from_heuristics(match, player)
    for death in all_deaths:
        death.overextended = (not death.is_teamfight and not death.denied and is_overextended(
            match, player.team, death.x, death.y, death.time))
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
        yields exactly one "death" decision point and no other point for the
        same moment.
    """
    player = _find_player(match, player_slot)
    if player is None:
        return [], []
    if position is None:
        pos = player_position(match, player_slot)
        position = pos.number if pos else None

    all_deaths = detect_deaths(match, player)

    all_decisions: list[DecisionPoint] = []
    all_decisions.extend(_detect_death_decisions(match, all_deaths, player, lang))
    all_decisions.extend(_detect_bad_trades(match, player, all_deaths, lang))
    all_decisions.extend(_detect_missed_cs(match, player, lang, position=position))
    good_plays = _detect_good_plays(match, player, lang)
    all_decisions.extend(good_plays)
    multi_kill_times = tuple(dp.time for dp in good_plays if dp.time_known and dp.subtype == "multi_kill")
    all_decisions.extend(_detect_fight_engagement(match, player, all_deaths, lang, multi_kill_times))
    all_decisions.extend(_detect_laning_overextend(match, player, all_deaths, lang))

    all_decisions.sort(key=lambda d: d.time)
    return all_deaths, all_decisions
