"""
Laning Phase Comparison Engine.

Compares a player's laning phase (0-10 min) against a reference player
(pro/high-rank) to detect divergence points and generate coaching insights.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import MatchData, PlayerData, LaningSnapshot


@dataclass
class Divergence:
    """A detected divergence point between player and reference."""
    time: int                    # game second
    div_type: str                # "cs", "positioning", "trade", "regen", "death", "item_timing"
    severity: float              # 0.0 - 1.0
    player_state: dict           # player's state at this moment
    ref_state: dict              # reference player's state
    description_zh: str          # Chinese description of the divergence


@dataclass
class ComparisonResult:
    """Full comparison result between two laning phases."""
    player_hero: str
    ref_hero: str
    player_lane: str
    ref_lane: str
    divergences: list[Divergence] = field(default_factory=list)
    per_minute_comparison: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


def find_player_by_slot(match: MatchData, slot: int) -> PlayerData | None:
    """Find a player in match data by player_slot."""
    for p in match.players:
        if p.player_slot == slot:
            return p
    return None


def get_snapshot_at_time(snapshots: list[LaningSnapshot], time: int) -> LaningSnapshot | None:
    """Get the closest snapshot to a given time."""
    if not snapshots:
        return None
    closest = min(snapshots, key=lambda s: abs(s.time - time))
    if abs(closest.time - time) <= 5:
        return closest
    return None


def build_per_minute_comparison(player: PlayerData, ref: PlayerData) -> dict:
    """Build per-minute comparison data for charts."""
    result = {
        "cs": {"player": [], "ref": []},
        "denies": {"player": [], "ref": []},
        "gold": {"player": [], "ref": []},
        "xp": {"player": [], "ref": []},
        "hp_pct": {"player": [], "ref": []},
    }

    for minute in range(11):  # 0-10
        time_sec = minute * 60

        # CS from timeline
        p_cs = player.lh_t[minute] if minute < len(player.lh_t) else 0
        r_cs = ref.lh_t[minute] if minute < len(ref.lh_t) else 0
        result["cs"]["player"].append(p_cs)
        result["cs"]["ref"].append(r_cs)

        # Gold from timeline
        p_gold = player.gold_t[minute] if minute < len(player.gold_t) else 0
        r_gold = ref.gold_t[minute] if minute < len(ref.gold_t) else 0
        result["gold"]["player"].append(p_gold)
        result["gold"]["ref"].append(r_gold)

        # XP from timeline
        p_xp = player.xp_t[minute] if minute < len(player.xp_t) else 0
        r_xp = ref.xp_t[minute] if minute < len(ref.xp_t) else 0
        result["xp"]["player"].append(p_xp)
        result["xp"]["ref"].append(r_xp)

        # HP% from snapshots
        p_snap = get_snapshot_at_time(player.laning_snapshots, time_sec)
        r_snap = get_snapshot_at_time(ref.laning_snapshots, time_sec)
        p_hp_pct = (p_snap.hp / max(p_snap.max_hp, 1) * 100) if p_snap and p_snap.max_hp > 0 else 100
        r_hp_pct = (r_snap.hp / max(r_snap.max_hp, 1) * 100) if r_snap and r_snap.max_hp > 0 else 100
        result["hp_pct"]["player"].append(round(p_hp_pct, 1))
        result["hp_pct"]["ref"].append(round(r_hp_pct, 1))

        # Denies from laning_denies events
        p_denies = sum(1 for d in player.laning_denies if d.time <= time_sec)
        r_denies = sum(1 for d in ref.laning_denies if d.time <= time_sec)
        result["denies"]["player"].append(p_denies)
        result["denies"]["ref"].append(r_denies)

    return result


def detect_cs_divergences(player: PlayerData, ref: PlayerData) -> list[Divergence]:
    """Detect CS (last-hit) divergence points."""
    divergences = []

    for minute in range(1, 11):
        p_cs = player.lh_t[minute] if minute < len(player.lh_t) else 0
        r_cs = ref.lh_t[minute] if minute < len(ref.lh_t) else 0

        if r_cs <= 0:
            continue

        cs_gap = r_cs - p_cs
        gap_ratio = cs_gap / max(r_cs, 1)

        # Flag if player is >20% behind reference
        if gap_ratio > 0.2 and cs_gap >= 3:
            severity = min(gap_ratio, 1.0)
            p_snap = get_snapshot_at_time(player.laning_snapshots, minute * 60)
            r_snap = get_snapshot_at_time(ref.laning_snapshots, minute * 60)

            divergences.append(Divergence(
                time=minute * 60,
                div_type="cs",
                severity=severity,
                player_state={
                    "cs": p_cs,
                    "gold": player.gold_t[minute] if minute < len(player.gold_t) else 0,
                    "position": {"x": p_snap.x, "y": p_snap.y} if p_snap else None,
                },
                ref_state={
                    "cs": r_cs,
                    "gold": ref.gold_t[minute] if minute < len(ref.gold_t) else 0,
                    "position": {"x": r_snap.x, "y": r_snap.y} if r_snap else None,
                },
                description_zh=f"第{minute}分钟补刀差距: 你{p_cs}个，参考{r_cs}个，落后{cs_gap}个补刀（约{cs_gap * 45}金币差距）",
            ))

    return divergences


def detect_trade_divergences(player: PlayerData, ref: PlayerData) -> list[Divergence]:
    """Detect differences in trading patterns (damage dealt/taken)."""
    divergences = []

    # Aggregate damage per 60-second window
    for minute in range(10):
        start = minute * 60
        end = (minute + 1) * 60

        p_dmg_dealt = sum(e.damage for e in player.laning_damage_events if start <= e.time < end)
        p_dmg_taken = sum(
            e.damage for other_p_events in [ref.laning_damage_events]  # simplified
            for e in other_p_events if start <= e.time < end
        )
        r_dmg_dealt = sum(e.damage for e in ref.laning_damage_events if start <= e.time < end)

        # Significant difference in aggression
        if r_dmg_dealt > 0 and p_dmg_dealt > 0:
            aggression_ratio = p_dmg_dealt / max(r_dmg_dealt, 1)

            if aggression_ratio < 0.3 and r_dmg_dealt > 50:
                p_snap = get_snapshot_at_time(player.laning_snapshots, start + 30)
                r_snap = get_snapshot_at_time(ref.laning_snapshots, start + 30)

                divergences.append(Divergence(
                    time=start + 30,
                    div_type="trade",
                    severity=min(1.0 - aggression_ratio, 1.0),
                    player_state={
                        "damage_dealt": p_dmg_dealt,
                        "hp_pct": round(p_snap.hp / max(p_snap.max_hp, 1) * 100) if p_snap and p_snap.max_hp > 0 else None,
                    },
                    ref_state={
                        "damage_dealt": r_dmg_dealt,
                        "hp_pct": round(r_snap.hp / max(r_snap.max_hp, 1) * 100) if r_snap and r_snap.max_hp > 0 else None,
                    },
                    description_zh=f"第{minute+1}分钟换血: 你造成{p_dmg_dealt}伤害，参考造成{r_dmg_dealt}伤害。你的对线压制力不足",
                ))

    return divergences


def detect_regen_divergences(player: PlayerData, ref: PlayerData) -> list[Divergence]:
    """Detect differences in regen item usage patterns."""
    divergences = []

    p_regen_count = len(player.laning_regen_events)
    r_regen_count = len(ref.laning_regen_events)

    if p_regen_count > r_regen_count + 3:
        divergences.append(Divergence(
            time=300,
            div_type="regen",
            severity=min((p_regen_count - r_regen_count) / 10.0, 1.0),
            player_state={"regen_used": p_regen_count},
            ref_state={"regen_used": r_regen_count},
            description_zh=f"你使用了{p_regen_count}次恢复品，参考只用了{r_regen_count}次。你承受了过多不必要的伤害",
        ))

    return divergences


def detect_positioning_divergences(player: PlayerData, ref: PlayerData) -> list[Divergence]:
    """Detect positioning differences using snapshots."""
    divergences = []

    for time_sec in range(30, 601, 30):  # check every 30 seconds
        p_snap = get_snapshot_at_time(player.laning_snapshots, time_sec)
        r_snap = get_snapshot_at_time(ref.laning_snapshots, time_sec)

        if not p_snap or not r_snap:
            continue

        # Calculate distance between positions
        dx = p_snap.x - r_snap.x
        dy = p_snap.y - r_snap.y
        dist = (dx * dx + dy * dy) ** 0.5

        # If player is much further from creep wave (toward base = too safe)
        # This is a simplified heuristic - real implementation would need creep wave position
        if dist > 2000:  # significant position difference
            # Check if player has lower CS at this point - suggests too passive
            if p_snap.last_hits < r_snap.last_hits - 2:
                divergences.append(Divergence(
                    time=time_sec,
                    div_type="positioning",
                    severity=min(dist / 4000.0, 1.0),
                    player_state={
                        "x": p_snap.x, "y": p_snap.y,
                        "hp_pct": round(p_snap.hp / max(p_snap.max_hp, 1) * 100),
                        "cs": p_snap.last_hits,
                    },
                    ref_state={
                        "x": r_snap.x, "y": r_snap.y,
                        "hp_pct": round(r_snap.hp / max(r_snap.max_hp, 1) * 100),
                        "cs": r_snap.last_hits,
                    },
                    description_zh=f"第{time_sec//60}分{time_sec%60}秒走位差异: 你的位置偏离参考约{int(dist)}距离，且补刀落后{r_snap.last_hits - p_snap.last_hits}个",
                ))

    return divergences


def generate_summary(player: PlayerData, ref: PlayerData, divergences: list[Divergence]) -> dict:
    """Generate a summary of the comparison."""
    # CS comparison
    p_cs_10 = player.lh_t[10] if len(player.lh_t) > 10 else 0
    r_cs_10 = ref.lh_t[10] if len(ref.lh_t) > 10 else 0

    # Gold comparison
    p_gold_10 = player.gold_t[10] if len(player.gold_t) > 10 else 0
    r_gold_10 = ref.gold_t[10] if len(ref.gold_t) > 10 else 0

    # Damage comparison
    p_dmg = sum(e.damage for e in player.laning_damage_events)
    r_dmg = sum(e.damage for e in ref.laning_damage_events)

    # Grade calculation (simplified)
    cs_ratio = p_cs_10 / max(r_cs_10, 1)
    grade = "S" if cs_ratio >= 1.0 else "A" if cs_ratio >= 0.85 else "B" if cs_ratio >= 0.7 else "C" if cs_ratio >= 0.55 else "D"

    # Top differences
    top_diffs = []
    cs_gap = r_cs_10 - p_cs_10
    if cs_gap > 0:
        top_diffs.append({"area": "补刀", "gap": f"-{cs_gap}", "impact": f"约损失{cs_gap * 45}金币"})
    gold_gap = r_gold_10 - p_gold_10
    if gold_gap > 0:
        top_diffs.append({"area": "经济", "gap": f"-{gold_gap}", "impact": f"前10分钟经济落后{gold_gap}金"})
    if p_dmg < r_dmg * 0.5 and r_dmg > 200:
        top_diffs.append({"area": "换血", "gap": f"{p_dmg} vs {r_dmg}", "impact": "对线压制力不足"})

    # Count divergence types
    div_counts = {}
    for d in divergences:
        div_counts[d.div_type] = div_counts.get(d.div_type, 0) + 1

    return {
        "grade": grade,
        "cs_10min": {"player": p_cs_10, "ref": r_cs_10},
        "gold_10min": {"player": p_gold_10, "ref": r_gold_10},
        "damage_laning": {"player": p_dmg, "ref": r_dmg},
        "regen_used": {"player": len(player.laning_regen_events), "ref": len(ref.laning_regen_events)},
        "top_differences": top_diffs,
        "divergence_counts": div_counts,
        "total_divergences": len(divergences),
    }


def compare_laning(player_match: MatchData, player_slot: int,
                   ref_match: MatchData, ref_slot: int) -> ComparisonResult:
    """
    Main comparison function.

    Compares a player's laning phase against a reference player's laning phase.
    Both should be playing the same hero in a similar matchup.
    """
    player = find_player_by_slot(player_match, player_slot)
    ref = find_player_by_slot(ref_match, ref_slot)

    if not player or not ref:
        return ComparisonResult(
            player_hero=player.hero_name if player else "unknown",
            ref_hero=ref.hero_name if ref else "unknown",
            player_lane="unknown",
            ref_lane="unknown",
            summary={"error": "Player not found"},
        )

    LANE_NAMES = {1: "安全路", 2: "中路", 3: "劣势路", 4: "野区"}

    # Detect divergences
    divergences = []
    divergences.extend(detect_cs_divergences(player, ref))
    divergences.extend(detect_trade_divergences(player, ref))
    divergences.extend(detect_regen_divergences(player, ref))
    divergences.extend(detect_positioning_divergences(player, ref))

    # Sort by time
    divergences.sort(key=lambda d: d.time)

    # Build comparison data
    per_minute = build_per_minute_comparison(player, ref)
    summary = generate_summary(player, ref, divergences)

    return ComparisonResult(
        player_hero=player.hero_name,
        ref_hero=ref.hero_name,
        player_lane=LANE_NAMES.get(player.lane or 0, "未知"),
        ref_lane=LANE_NAMES.get(ref.lane or 0, "未知"),
        divergences=divergences,
        per_minute_comparison=per_minute,
        summary=summary,
    )
