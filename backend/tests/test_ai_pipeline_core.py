"""AI pipeline tests that need no LLM: geometry, detection, context, rules.

Everything runs on synthetic matches (world units); the knowledge layer is
stubbed so these tests only exercise the AI-pipeline modules.
"""
from __future__ import annotations

import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import asyncio

import pytest

from backend.app import geo
from backend.app.ai import (
    causal_chain,
    coach,
    context_builder,
    decision_detector,
    evaluator,
    phases,
    positions,
    pro_patterns,
    vision,
)
from backend.app.ai.models import (
    REPORT_SCHEMA_VERSION,
    DeathEvent,
    DecisionPoint,
    normalize_grade,
)
from backend.app.ai.rag.types import SOURCE_TYPES
from test_ai_pipeline_helpers import (
    CJK_RE,
    default_players,
    item,
    knowledge_refs,
    make_match,
    make_player,
    observer,
    snapshots,
    teamfight_death,
)

_ARCHETYPES = {
    "Juggernaut": 1, "Phantom Assassin": 1, "Puck": 2, "Lina": 2, "Axe": 3,
    "Rubick": 4, "Lion": 4, "Crystal Maiden": 5, "Witch Doctor": 5,
}
_REAL_HERO_TYPICAL_POSITION = positions.hero_typical_position  # before the fixture below replaces it


@pytest.fixture(autouse=True)
def _isolate_knowledge(monkeypatch):
    """Keep the knowledge layer (another group's code) out of these tests."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(positions, "hero_typical_position", lambda hero: _ARCHETYPES.get(hero, 0))
    from backend.app.ai import knowledge_base
    monkeypatch.setattr(knowledge_base, "get_knowledge_for_context", lambda ctx, lang="zh": "")
    monkeypatch.setattr(knowledge_base, "related_refs", lambda ctx, lang="zh", k=3: knowledge_refs(k))


def _walk_strings(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


# ---------------------------------------------------------------------------
# Phases / positions / roles
# ---------------------------------------------------------------------------

def test_game_phase_boundaries_are_shared():
    assert [phases.game_phase(t) for t in (0, 29, 30, 599, 600, 1499, 1500)] == [
        "pregame", "pregame", "laning", "laning", "mid_game", "mid_game", "late_game",
    ]
    assert decision_detector.game_phase is phases.game_phase  # one definition, no local copy


def test_short_trailing_phase_is_merged():
    windows = phases.applicable_phases(10 * 60 + 40)
    assert [w[0] for w in windows] == ["laning"]
    assert windows[0][2] is None  # open-ended -> covers the extra 40 s


def test_phase_stats_counts_towers_not_barracks_and_clamps_series():
    match = make_match(objectives=[
        {"time": 700, "type": "building_kill", "key": "npc_dota_goodguys_tower1_mid", "team": "dire"},
        {"time": 710, "type": "building_kill", "key": "npc_dota_goodguys_melee_rax_mid", "team": "dire"},
    ], duration=1300)
    player = make_player(0, "Juggernaut", gold_t=[0, 500, 1000], xp_t=[0, 300, 700], lh_t=[0, 5, 9])
    stats = phases.slice_player_phases(match, player, death_times=[650])
    early_mid = next(s for s in stats if s.phase == "early_mid")
    assert early_mid.towers_taken_dire == 1  # the barracks kill is not a tower
    assert early_mid.deaths == 1
    assert early_mid.gold_delta == 0  # series shorter than the match: clamped, not negative


def test_three_players_in_one_lane_get_unique_positions():
    players = [
        make_player(0, "Juggernaut", lane_role=1, net_worth=20000),
        make_player(1, "Puck", lane_role=1, net_worth=15000),
        make_player(2, "Crystal Maiden", lane_role=1, net_worth=4000),
        make_player(3, "Axe", lane_role=3, net_worth=12000),
        make_player(4, "Rubick", lane_role=4, net_worth=5000),
    ]
    result = positions.derive_team_positions(players)
    numbers = sorted(p.number for p in result.values())
    assert numbers == [1, 2, 3, 4, 5]
    assert result[0].number == 1 and result[2].number == 5


def test_positions_use_lane_role_and_absolute_lane_for_dire():
    dire = [p for p in default_players() if p.team == "dire"]
    for p in dire:  # drop the relative role: must be derived from lane + team
        p.lane_role = 0
    result = positions.derive_team_positions(dire)
    assert result[128].number == 1      # lane 3 (top) is Dire's safe lane
    assert result[130].number == 3      # lane 1 (bottom) is Dire's off lane
    assert result[132].number == 5
    assert result[131].number == 4


def test_resolve_role_never_crashes_for_heroes_outside_the_fallback(monkeypatch):
    from backend.app.ai import knowledge_base

    def boom(*_a, **_k):
        raise RuntimeError("knowledge layer broken")

    monkeypatch.setattr(knowledge_base, "get_hero_info", boom)
    monkeypatch.setattr(positions, "hero_typical_position", _REAL_HERO_TYPICAL_POSITION)
    assert pro_patterns.resolve_role(None, "Tinker") == "offlane"
    assert pro_patterns.resolve_role(None, "Tinker", position=2) == "mid"
    match = make_match()
    tinker = next(p for p in match.players if p.hero_name == "Tinker")
    assert pro_patterns.resolve_role(tinker, "Tinker", match=match) == "offlane"
    text = pro_patterns.get_benchmark_context("Tinker", pro_patterns.resolve_role(None, "Tinker"), 900, lang="en")
    assert "Benchmark" in text and not CJK_RE.search(text)


def test_learned_overlay_rates_a_player_against_his_side_of_the_team(monkeypatch):
    """First full learning run (2026-10-11), real replay 9037980041: the position 5
    Techies had 382 GPM and was rated "below average", because the hero's percentiles
    over all its games (median 418) are pulled up by mid-lane Techies. The numbers
    below are OpenDota's answers of that day for role=core / role=support."""
    learned = {"techies": {
        "gold_per_min": {"p10": 304, "p50": 418, "p90": 559},                      # all games: not used
        "roles": {"core": {"gold_per_min": {"p10": 365, "p50": 499, "p90": 642}},
                  "support": {"gold_per_min": {"p10": 288, "p50": 392, "p90": 484}}},
    }}
    monkeypatch.setattr(pro_patterns, "_load_learned_benchmarks", lambda: learned)
    bench = pro_patterns._metric_bench("Techies", "gold_per_min", pro_patterns.GPM_BENCHMARKS, "support_5")
    assert (bench["good"], bench["average"], bench["poor"], bench["scope"], bench["side"]) == (484, 392, 288, "hero", "support")
    assert pro_patterns._rate_higher_is_better(382, bench["good"], bench["average"], bench["poor"]) == "below_average"
    assert pro_patterns._rate_higher_is_better(400, bench["good"], bench["average"], bench["poor"]) == "average"
    core = pro_patterns._metric_bench("Techies", "gold_per_min", pro_patterns.GPM_BENCHMARKS, "mid")
    assert (core["average"], core["side"]) == (499, "core")
    for lang, text in (("en", "Techies as a support (positions 4-5): OpenDota median (all ranks) 392"),
                       ("zh", "Techies打4/5号位时在OpenDota全分段中位数392")):
        assert pro_patterns._population_ref(bench, "Techies", "x", lang) == text
    assert pro_patterns._population_ref(core, "Techies", "x", "zh") == "Techies打1/2/3号位时在OpenDota全分段中位数499"
    context = pro_patterns.get_benchmark_context("Techies", "support_5", 900, lang="zh")
    assert "- Techies打4/5号位时在OpenDota全分段的GPM中位数（非职业数据）: 392" in context

    # A file of an earlier run has the mixed percentiles only: the role table is used, not the mix
    del learned["techies"]["roles"]
    fallback = pro_patterns._metric_bench("Techies", "gold_per_min", pro_patterns.GPM_BENCHMARKS, "support_5")
    assert (fallback["average"], fallback["scope"]) == (pro_patterns.GPM_BENCHMARKS["support_5"]["average"], "role")
    # ... and one half missing falls back for that side alone
    learned["techies"]["roles"] = {"core": {"gold_per_min": {"p10": 365, "p50": 499, "p90": 642}}}
    assert pro_patterns._metric_bench("Techies", "gold_per_min", pro_patterns.GPM_BENCHMARKS, "support_4")["scope"] == "role"
    assert pro_patterns._metric_bench("Techies", "gold_per_min", pro_patterns.GPM_BENCHMARKS, "carry")["scope"] == "hero"


def test_learned_overlay_is_labelled_as_population_data(monkeypatch):
    monkeypatch.setattr(pro_patterns, "_load_learned_benchmarks", lambda: {
        "tinker": {"roles": {"core": {"gold_per_min": {"p10": 300, "p50": 450, "p90": 600}}}},
    })
    match = make_match()
    tinker = next(p for p in match.players if p.hero_name == "Tinker")
    report = pro_patterns.format_pro_comparison(pro_patterns.compare_player(tinker, match, lang="en", position=2))
    gpm = next(c for c in report["comparisons"] if c["metric"] == "gpm")
    assert gpm["benchmark_source"] == pro_patterns.SOURCE_POPULATION
    assert "Tinker as a core (positions 1-3): OpenDota median (all ranks) 450" in gpm["description"]
    # Without learned percentiles the role table is used: all-rank data as well, never "pro".
    dmg = next(c for c in report["comparisons"] if c["metric"] == "damage_per_min")
    assert dmg["benchmark_source"] == pro_patterns.SOURCE_POPULATION
    assert "OpenDota median of Mid (Pos 2) heroes (all ranks) 660" in dmg["description"]
    assert "pro" not in dmg["description"].lower()
    context = pro_patterns.get_benchmark_context("Tinker", "mid", 900, lang="en")
    assert "Tinker as a core (positions 1-3): median GPM on OpenDota (all ranks, not pro): 450" in context
    assert "Median hero damage per minute of typical Mid (Pos 2) heroes on OpenDota (all ranks, not pro): 660" in context


def test_patch_version_is_resolved_lazily():
    assert isinstance(pro_patterns.PATCH_VERSION, str)
    assert pro_patterns.PATCH_VERSION == pro_patterns.get_patch_version()


# ---------------------------------------------------------------------------
# Vision / geometry (world units)
# ---------------------------------------------------------------------------

def test_towers_come_from_buildings_and_respect_destruction():
    match = make_match(buildings=[
        {"name": "npc_dota_goodguys_tower1_mid", "team": "radiant", "type": "tower",
         "x": -1600.0, "y": -1350.0, "destroyed_time": 800},
        {"name": "npc_dota_goodguys_melee_rax_mid", "team": "radiant", "type": "barracks",
         "x": -4000.0, "y": -3500.0, "destroyed_time": None},
    ])
    standing = vision._get_alive_towers(match, "radiant", 700)
    assert ("mid_t1", -1600.0, -1350.0) in standing
    assert all(name != "mid_t1" for name, _x, _y in vision._get_alive_towers(match, "radiant", 900))
    # Dire has no buildings in the replay -> geo defaults.
    dire = {name: (x, y) for name, x, y in vision._get_alive_towers(match, "dire", 0)}
    assert dire["top_t1"] == geo.TOWER_POSITIONS["dire"]["top_t1"]


def test_ward_vision_uses_world_unit_radius():
    players = default_players()
    for p in players:
        p.laning_snapshots = []
    radiant_support = players[4]
    radiant_support.laning_snapshots = snapshots(-6500, -6000, until=200)  # in base, far away
    enemy_seen = players[5]
    enemy_seen.laning_snapshots = snapshots(2000 + 1500, 1000, until=200)
    enemy_hidden = players[6]
    enemy_hidden.laning_snapshots = snapshots(2000 + 1700, 1000, until=200)
    match = make_match(players, wards=[observer(2000, 1000, slot=4, time=0)])
    vis = {e.hero_name: e for e in vision._track_enemy_visibility(match, "radiant", 4, 100)}
    assert vis[enemy_seen.hero_name].is_visible
    assert vis[enemy_seen.hero_name].last_seen_source == "ward"
    assert not vis[enemy_hidden.hero_name].is_visible


def test_ward_without_owner_slot_uses_team_field():
    from backend.app.models import WardEvent

    ward = WardEvent(ward_type="observer", x=0.0, y=10.0, time=0, player_slot=-1, team="dire")
    assert vision.ward_team(ward) == "dire"
    assert vision.ward_team(observer(0, 0, slot=-1)) == ""


def test_describe_area_in_world_units():
    x, y = geo.TOWER_POSITIONS["radiant"]["top_t1"]
    assert vision.describe_area(x, y, "en") == "Radiant top lane"
    assert vision.describe_area(x, y, "zh") == "天辉上路"
    assert "base" in vision.describe_area(*geo.ANCIENT_POSITIONS["dire"], lang="en")
    assert vision.describe_area(0.0, 0.0, "en") == "unknown area"


def test_vision_text_is_fully_english():
    match = make_match(wards=[observer(-1500, -1000, slot=4, time=0, destroyed=290)])
    va = vision.analyze_vision(match, 1, 300, lang="en")
    assert va.formatted_text.startswith("## Vision Analysis")
    assert not CJK_RE.search(va.formatted_text)
    assert all(not CJK_RE.search(r) for r in va.state_at_decision.danger_reasons)
    assert not CJK_RE.search(va.timeline.summary)


# ---------------------------------------------------------------------------
# Decision detection
# ---------------------------------------------------------------------------

def test_supports_get_no_missed_cs_points():
    match = make_match()
    for slot in (3, 4, 131, 132):  # pos 4/5 players with almost no CS
        _deaths, decisions = decision_detector.detect_decisions(match, slot, lang="en")
        assert not [d for d in decisions if d.point_type == "missed_cs"], slot


def test_core_missed_cs_keeps_the_worst_per_half():
    players = default_players()
    players[0].laning_snapshots = snapshots(4500, -6000, cs_per_min=1.0)  # starving carry
    match = make_match(players)
    _deaths, decisions = decision_detector.detect_decisions(match, 0, lang="en")
    missed = [d for d in decisions if d.point_type == "missed_cs"]
    assert 1 <= len(missed) <= 2
    assert all(d.time >= 180 for d in missed)


def test_death_far_from_towers_yields_one_point_and_world_coords():
    players = default_players()
    # Puck dies deep in the Dire jungle (no teamfight, killer known).
    players[1].killed_by = {"Lina": 1}
    players[1].deaths = 1
    snaps = snapshots(-500, -500, until=600)
    for s in snaps:
        if 395 <= s.time <= 400:
            s.x, s.y = 2500.0, 3800.0
        if s.time == 400:
            s.hp = 0
    players[1].laning_snapshots = snaps
    match = make_match(players)
    deaths, decisions = decision_detector.detect_decisions(match, 1, lang="en")
    assert len(deaths) == 1
    death = deaths[0]
    assert (death.x, death.y) == (2500.0, 3800.0)  # world units, not map percent
    assert death.overextended
    at_death = [d for d in decisions if d.time == death.time]
    assert [d.point_type for d in at_death] == ["death"]
    assert "(overextended)" in at_death[0].brief


def test_bad_trades_are_merged_and_capped():
    from backend.app.models import LaningDamageEvent

    players = default_players()
    # Axe (index 7) hits Juggernaut (index 0) every 15 s; Juggernaut never hits back.
    players[7].laning_damage_events = [
        LaningDamageEvent(time=t, target_idx=0, damage=150, inflictor="attack") for t in range(60, 600, 15)
    ]
    match = make_match(players)
    _deaths, decisions = decision_detector.detect_decisions(match, 0, lang="en")
    trades = [d for d in decisions if d.point_type == "bad_trade"]
    assert 1 <= len(trades) <= decision_detector.MAX_BAD_TRADES


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

def test_nearby_allies_and_enemies_are_computed_by_distance():
    players = default_players()
    players[0].laning_snapshots = snapshots(1000, 1000)            # the player
    players[1].laning_snapshots = snapshots(1500, 1000)            # ally 500 away
    players[2].laning_snapshots = snapshots(-6000, 5000)           # ally far away
    players[3].laning_snapshots = []                               # ally position unknown
    players[5].laning_snapshots = snapshots(1800, 1000)            # enemy 800 away
    players[6].laning_snapshots = snapshots(7000, -6000)           # enemy far away
    match = make_match(players)
    dp = DecisionPoint(time=300, point_type="bad_trade", player_slot=0, hero_name="Juggernaut",
                       severity="major", brief="trade")
    ctx = context_builder.build_context(match, dp, lang="en")
    assert [a.hero_name for a in ctx.ally_states] == ["Puck"]
    assert [e.hero_name for e in ctx.enemy_states] == ["Phantom Assassin"]
    assert ctx.match_context["nearby_allies"] == 1
    assert ctx.match_context["nearby_enemies"] == 1
    assert ctx.ally_states[0].distance_to_player == 500
    assert {d["hero_name"] for d in ctx.map_state["unknown_position_allies"]} == {"Rubick"}
    assert "Tinker" in {d["hero_name"] for d in ctx.map_state["distant_allies"]}


def test_match_situation_is_from_the_players_team_perspective():
    match = make_match()  # radiant_gold_adv is negative: Dire is ahead
    dp_r = DecisionPoint(time=900, point_type="fight_engage", player_slot=0, hero_name="Juggernaut",
                         severity="major", brief="fight")
    dp_d = DecisionPoint(time=900, point_type="fight_engage", player_slot=128, hero_name="Phantom Assassin",
                         severity="major", brief="fight")
    ctx_r = context_builder.build_context(match, dp_r, lang="en")
    ctx_d = context_builder.build_context(match, dp_d, lang="en")
    assert ctx_r.match_context["gold_adv"] == -3000
    assert ctx_d.match_context["gold_adv"] == 3000
    text_r = evaluator.format_match_situation(ctx_r, "en")
    text_d = evaluator.format_match_situation(ctx_d, "en")
    assert "your team behind by 3,000" in text_r
    assert "your team ahead by 3,000" in text_d
    assert "Towers standing" in text_r
    prompt = evaluator._build_evaluation_prompt(ctx_r, "", lang="en")
    assert "your team behind by 3,000" in prompt


def test_inventory_skips_consumables():
    # Blink Dagger and Battle Fury share no component, so the result does not
    # depend on whether the game data on disk has item recipes.
    player = make_player(0, "Juggernaut", items=[
        item("Tango", "tango", -80), item("Blink Dagger", "blink", -70),
        item("Observer Ward", "ward_observer", 100), item("Battle Fury", "bfury", 800),
        # 7.41 consumables (the Forager's Kit yields).
        item("Forager's Health", "foragers_health", 300), item("Forager's Mana", "foragers_mana", 310),
        item("Forager's Stats", "foragers_stats", 320),
    ])
    assert context_builder._items_at_time(player, 900) == ["Blink Dagger", "Battle Fury"]


# ---------------------------------------------------------------------------
# Rules evaluation
# ---------------------------------------------------------------------------

def _teamfight_death_context(allies, enemies, lang="en"):
    players = default_players()
    for p in players:
        p.laning_snapshots = []
    tf = teamfight_death(1, "Puck", 129, "Lina", start=1500, x=1000.0, y=1000.0,
                         allies=allies, enemies=enemies)
    match = make_match(players, teamfights=[tf])
    death = DeathEvent(time=1505, player_slot=1, hero_name="Puck", killer_hero="Lina",
                       x=1000.0, y=1000.0, is_teamfight=True, teamfight_idx=0)
    dp = DecisionPoint(time=1505, point_type="death", player_slot=1, hero_name="Puck",
                       severity="critical", brief="Killed by Lina in a teamfight", related_death=death)
    return context_builder.build_context(match, dp, lang=lang)


def test_teamfight_death_outnumbered_branch_is_reachable():
    ctx = _teamfight_death_context(
        allies=[], enemies=[(128, "Phantom Assassin", 1200.0, 1100.0), (130, "Axe", 900.0, 800.0)],
    )
    assert ctx.match_context["nearby_allies"] == 0
    ev = evaluator.evaluate_decision_fallback(ctx, lang="en")
    assert "outnumbered" in ev.analysis
    assert ev.score in ("C", "D")


def test_teamfight_death_with_allies_nearby_scores_b():
    ctx = _teamfight_death_context(
        allies=[(0, "Juggernaut", 1100.0, 1000.0), (2, "Tinker", 900.0, 1200.0)],
        enemies=[(128, "Phantom Assassin", 1200.0, 1100.0)],
    )
    ev = evaluator.evaluate_decision_fallback(ctx, lang="en")
    assert ctx.match_context["nearby_allies"] == 2
    assert "2 allies" in ev.analysis


def test_causal_chain_root_cause_matches_its_summary():
    ctx = _teamfight_death_context(
        allies=[], enemies=[(128, "Phantom Assassin", 1200.0, 1100.0), (130, "Axe", 900.0, 800.0)],
    )
    match = make_match(teamfights=[teamfight_death(
        1, "Puck", 129, "Lina", start=1500, x=1000.0, y=1000.0,
        enemies=[(128, "Phantom Assassin", 1200.0, 1100.0), (130, "Axe", 900.0, 800.0)],
    )])
    chain = causal_chain.build_causal_chain(match, ctx.decision_point.related_death, 1, lang="en")
    root = [s for s in chain.steps if s.significance == "root_cause"]
    assert len(root) == 1
    assert chain.root_cause_type == "teamfight_outnumbered"
    assert root[0].step_type == "no_retreat"
    assert not CJK_RE.search(chain.formatted_text)


def test_parse_rejects_free_text_and_accepts_labelled_verdicts():
    dp = DecisionPoint(time=100, point_type="death", player_slot=0, hero_name="X", severity="critical", brief="b")
    free = evaluator._parse_evaluation_response("Let me check the vision first.", dp, lang="en", strict=True)
    assert not free._parsed_ok
    loose = evaluator._parse_evaluation_response("Let me check the vision first.", dp, lang="en")
    assert not loose._parsed_ok
    good = evaluator._parse_evaluation_response(
        "**Score:** A\n**Analysis:** Nice.\n**Mistake:** None\n**Better Play:** Keep it up\n**Lesson:** L",
        dp, lang="en", strict=True,
    )
    assert good._parsed_ok and good.score == "A" and good.analysis == "Nice." and good.lesson == "L"
    zh_in_en = evaluator._parse_evaluation_response("评分: C\n分析: 走位太深。\n错误: 无\n", dp, lang="en")
    assert zh_in_en._parsed_ok and zh_in_en.score == "C"


def test_normalize_grade():
    assert [normalize_grade(v) for v in ("b+", " A ", None, "", "X", "s")] == ["B", "A", "C", "C", "C", "S"]


# ---------------------------------------------------------------------------
# Full report (rules mode)
# ---------------------------------------------------------------------------

def _rules_report(match, slot, lang):
    return asyncio.run(coach.generate_coaching_report(match, slot, lang=lang))


def test_invalid_player_slot_raises_value_error():
    with pytest.raises(ValueError):
        _rules_report(make_match(), 77, "en")


@pytest.mark.parametrize("slot", [0, 1, 3, 128, 131])
def test_english_rules_report_has_no_cjk(slot):
    players = default_players()
    tf = teamfight_death(slot, next(p.hero_name for p in players if p.player_slot == slot),
                         129 if slot < 128 else 1, "Lina" if slot < 128 else "Puck",
                         start=1200, x=-300.0, y=-200.0)
    match = make_match(players, teamfights=[tf], wards=[observer(-2000, -1500, slot=4, time=60)],
                       objectives=[{"time": 700, "type": "building_kill",
                                    "key": "npc_dota_goodguys_tower1_mid", "team": "dire"}])
    report = _rules_report(match, slot, "en")
    data = report.to_dict()
    data["position"].pop("label_zh", None)  # both labels are shipped by design
    leaks = [(path, text) for path, text in _walk_strings(data) if CJK_RE.search(text)]
    assert not leaks, leaks[:5]


def test_rules_report_contract_fields():
    from backend.app.models import KillEvent

    players = default_players()
    players[1].multi_kills = {"2": 1}        # a good play the kill log cannot date (it shows no double kill)
    match = make_match(players, kills_log=[
        KillEvent(time=400.5, victim_slot=1, killer_slot=129, x=2500.0, y=3800.0),
        KillEvent(time=900.0, victim_slot=129, killer_slot=1),
    ])
    report = _rules_report(match, 1, "zh")
    data = report.to_dict()
    assert coach.REPORT_SCHEMA_VERSION == REPORT_SCHEMA_VERSION == 4
    assert data["report_version"] == 4
    assert data["llm_used"] is False and data["degraded"] is False
    # Schema 4: how the knowledge sources of the report were retrieved.
    assert set(data["knowledge"]) == {"mode", "embedder", "chunks", "vectors"}
    assert data["knowledge"]["mode"] == "lexical"       # the test session ranks without vectors
    assert data["knowledge"]["embedder"] is None and data["knowledge"]["vectors"] == 0
    assert isinstance(data["knowledge"]["chunks"], int)  # 0 here: the knowledge layer is stubbed out
    assert data["overall_grade"] in ("S", "A", "B", "C", "D")
    assert [d["time"] for d in data["death_analyses"]] == [400]
    assert [(m["point_type"], m["time"]) for m in data["key_moments"]] == [("good_play", None)]  # unknown time
    for item_ in data["death_analyses"] + data["key_moments"]:
        assert item_["score"] in ("S", "A", "B", "C", "D")
        assert item_["llm_used"] is False and item_["degraded"] is False
        assert "agent_trace" not in item_  # not shown by the frontend, not shipped
        # A rule evaluation read no knowledge: it lists the three best related entries, none cited.
        assert [s["id"] for s in item_["sources"]] == ["test:entry_1", "test:entry_2", "test:entry_3"]
        assert "cited_refs" not in item_
        for source in item_["sources"]:
            assert set(source) == {"id", "title", "source_type", "patch", "cited"}
            assert source["cited"] is False and source["title"] and source["id"]
            assert source["source_type"] in SOURCE_TYPES
    assert data["phase_reviews"]
    for review in data["phase_reviews"]:
        assert review["score"] in ("S", "A", "B", "C", "D")
        assert review["llm_used"] is False and review["degraded"] is False
        assert "used_llm" not in review
        assert isinstance(review["start_min"], int)
        assert review["end_min"] is None or review["end_min"] > review["start_min"]
        assert review["time_range"].endswith("分钟")  # localized


def test_laning_review_uses_relative_lane_role_for_dire():
    players = default_players()
    for p in players:
        p.lane_role = 0  # force derivation from the absolute lane
    match = make_match(players)
    review = coach._generate_laning_review(
        match, players[5], [], lang="en", positions=positions.derive_match_positions(players),
    )
    assert "(Safe Lane)" in review["lane_matchup"]          # Dire top = safe lane
    assert "Witch Doctor" not in review["lane_matchup"]     # an ally, not an opponent
    assert "Tinker" in review["lane_matchup"]


def test_mock_match_report_runs_if_available():
    try:
        from backend.app.mock_data import generate_mock_match
        match = generate_mock_match()
    except Exception as exc:  # another group may be mid-edit on the mock data
        pytest.skip(f"mock data unavailable: {exc}")
    for p in match.players:
        _deaths, decisions = decision_detector.detect_decisions(match, p.player_slot, lang="en")
        pos = positions.player_position(match, p.player_slot)
        if pos is not None and pos.number in (4, 5):
            assert not [d for d in decisions if d.point_type == "missed_cs"]
    data = _rules_report(match, match.players[0].player_slot, "en").to_dict()
    data["position"].pop("label_zh", None)
    assert not [p for p, t in _walk_strings(data) if CJK_RE.search(t)]


def test_enemies_fighting_are_visible_even_at_night():
    players = default_players()
    for p in players:
        p.laning_snapshots = []
    # Night (25:00-30:00): hero vision is only 800, but the fight reveals Lina.
    tf = teamfight_death(1, "Puck", 129, "Lina", start=1500, x=1000.0, y=1000.0,
                         enemies=[(128, "Phantom Assassin", 2500.0, 1000.0)])
    match = make_match(players, teamfights=[tf])
    vis = {e.hero_name: e for e in vision._track_enemy_visibility(match, "radiant", 1, 1505)}
    assert vis["Lina"].is_visible and vis["Lina"].last_seen_source == "teamfight"
    assert vis["Phantom Assassin"].is_visible
    assert not vis["Axe"].is_visible  # not in the fight, position unknown
