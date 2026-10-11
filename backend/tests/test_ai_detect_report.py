"""Report building: phases, positions, benchmarks, laning review, summary, payload.

Synthetic matches plus the real-match fixtures of the parsing tests and the
mock match; rules mode only (no LLM, no network).
"""
from __future__ import annotations

import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import asyncio
import json
import re

import pytest

from backend.app.ai import coach, decision_detector, phases, positions, pro_patterns
from backend.app.ai.models import DeathEvent, DecisionPoint, Evaluation
from backend.app.models import LaningCreepKill, SmokeEvent
from test_ai_detect_deaths import _clarity_fixture, _mock, _opendota_fixture, kill
from test_ai_pipeline_helpers import CJK_RE, default_players, item, make_match, make_player

_ARCHETYPES = {
    "Juggernaut": 1, "Phantom Assassin": 1, "Puck": 2, "Lina": 2, "Axe": 3, "Tinker": 3,
    "Rubick": 4, "Lion": 4, "Crystal Maiden": 5, "Witch Doctor": 5,
    "Wraith King": 1, "Void Spirit": 2, "Zeus": 2, "Tidehunter": 3,
}


@pytest.fixture(autouse=True)
def _isolate_knowledge(monkeypatch):
    """Keep the knowledge layer (another group's code) out of these tests."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(positions, "hero_typical_position", lambda hero: _ARCHETYPES.get(hero, 0))
    from backend.app.ai import knowledge_base
    monkeypatch.setattr(knowledge_base, "get_knowledge_for_context", lambda ctx, lang="zh": "")
    monkeypatch.setattr(knowledge_base, "related_refs", lambda ctx, lang="zh", k=3: [])


def _report(match, slot, lang="en"):
    return asyncio.run(coach.generate_coaching_report(match, slot, lang=lang)).to_dict()


def _strings(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def _evaluation(point_type, score, time=100, brief="b", mistake="", **dp_kw):
    dp = DecisionPoint(time=time, point_type=point_type, player_slot=0, hero_name="Juggernaut",
                       severity="critical", brief=brief, **dp_kw)
    return Evaluation(decision_point=dp, score=score, analysis="a", mistake=mistake)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def test_phase_kda_comes_from_the_kill_log_and_adds_up():
    players = default_players()
    jugg = players[0]
    jugg.kills, jugg.deaths, jugg.assists = 3, 2, 2
    match = make_match(players, duration=1500, kills_log=[
        kill(-12.0, 0, 129),                              # before the horn: laning
        kill(300.0, 128, 0, assist_slots=[1]),
        kill(700.0, 129, 1, assist_slots=[0]),
        kill(1250.0, 130, 0), kill(1260.0, 131, 0),
        kill(1300.0, 132, 1, assist_slots=[0, 2]),
        kill(1500.0, 0, 130),                             # on the final second: last window
        kill(1400.0, 0, 4, deny=True),                    # a deny is a death, not a kill for anyone
    ])
    jugg.deaths = 3
    stats = phases.slice_player_phases(match, jugg, death_times=[9999], lang="en")
    assert [(s.phase, s.kills, s.deaths, s.assists) for s in stats] == [
        ("laning", 1, 1, 0), ("early_mid", 0, 0, 1), ("mid_late", 2, 2, 1),
    ]
    assert (sum(s.kills for s in stats), sum(s.deaths for s in stats)) == (jugg.kills, jugg.deaths)
    assert not any(s.kda_estimated for s in stats)
    assert stats[0].death_times == [-12]
    crystal = players[4]
    assert sum(s.kills for s in phases.slice_player_phases(match, crystal)) == 0   # the deny is no kill


def test_phase_assists_are_unknown_not_zero_when_the_log_has_none():
    players = default_players()
    match = make_match(players, kills_log=[kill(300.0, 128, 0), kill(900.0, 129, 0)])   # OpenDota: no assists
    stats = phases.slice_player_phases(match, players[1], lang="en")
    assert [s.assists for s in stats] == [None] * len(stats)
    assert [s.to_dict()["assists"] for s in stats] == [None] * len(stats)
    with_assist = make_match(players, kills_log=[kill(300.0, 128, 0, assist_slots=[1])])
    assert [s.assists for s in phases.slice_player_phases(with_assist, players[1])][0] == 1


def test_phase_kda_without_kill_log_is_flagged_as_estimate():
    from test_ai_pipeline_helpers import teamfight_death

    players = default_players()
    tf = teamfight_death(128, "Phantom Assassin", 1, "Puck", start=700, x=0.0, y=0.0)
    stats = phases.slice_player_phases(make_match(players, teamfights=[tf]), players[1], death_times=[650])
    early = next(s for s in stats if s.phase == "early_mid")
    assert early.kda_estimated and (early.kills, early.deaths, early.assists) == (1, 1, None)


def test_phase_time_range_is_localized_with_numeric_bounds():
    match = make_match(duration=34 * 60 + 51)
    player = match.players[0]
    en = phases.slice_player_phases(match, player, lang="en")
    zh = phases.slice_player_phases(match, player, lang="zh")
    assert [s.time_range for s in en] == ["0-10 min", "10-20 min", "20-30 min", "30-34 min"]
    assert [s.time_range for s in zh] == ["0-10 分钟", "10-20 分钟", "20-30 分钟", "30-34 分钟"]
    assert [(s.start_min, s.end_min) for s in en] == [(0, 10), (10, 20), (20, 30), (30, 34)]
    # A short trailing window is merged: the last one is open-ended.
    merged = phases.slice_player_phases(make_match(duration=20 * 60 + 40), player, lang="zh")
    assert [(s.time_range, s.start_min, s.end_min) for s in merged] == [("0-10 分钟", 0, 10), ("10+ 分钟", 10, None)]
    assert phases.format_time_range(40, None, "en") == "40+ min"


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def test_without_lane_data_a_rich_core_is_not_position_5():
    """Unparsed OpenDota match: no lane for anyone (real Dire team of 8762775364)."""
    team = [
        make_player(128, "Lion", net_worth=9368),
        make_player(129, "Wraith King", net_worth=21902),
        make_player(130, "Zeus", net_worth=12307),          # played position 4 here, usually a mid
        make_player(131, "Void Spirit", net_worth=20481),
        make_player(132, "Tidehunter", net_worth=17447),
    ]
    result = positions.derive_team_positions(team)
    assert {slot: pos.number for slot, pos in result.items()} == {129: 1, 131: 2, 132: 3, 130: 4, 128: 5}
    # A support that out-farmed a losing core stays a support.
    team = [
        make_player(0, "Rubick", net_worth=8400), make_player(1, "Juggernaut", net_worth=11000),
        make_player(2, "Crystal Maiden", net_worth=6100), make_player(3, "Tinker", net_worth=10300),
        make_player(4, "Puck", net_worth=7700),
    ]
    numbers = {slot: pos.number for slot, pos in positions.derive_team_positions(team).items()}
    assert numbers == {1: 1, 4: 2, 3: 3, 0: 4, 2: 5}


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

_MINED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "mining_aggregates_positions.json")
_POSITION_ROLES = {"1": "carry", "2": "mid", "3": "offlane", "4": "support_4", "5": "support_5"}


@pytest.fixture
def mined_dir(tmp_path, monkeypatch):
    """A learned directory that holds the miner's aggregates (the fixture is its run of 2026-10-10)."""
    from backend.app.ai.learning import base

    learned = tmp_path / "learned"
    learned.mkdir()
    monkeypatch.setattr(base, "LEARNED_DIR", learned)
    return learned


def _mined(directory, **changes):
    with open(_MINED, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload.update(changes)
    (directory / "_mining_aggregates.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_pro_reference_tables_are_the_mined_pro_matches():
    """The hand-written rows contradicted the project's own pro data: position 3 had 35 last hits
    at 10:00 (mined: 55), position 1 160 at 20:00 (203), kill participation was 8-15 points low."""
    with open(_MINED, "r", encoding="utf-8") as f:
        mined = json.load(f)
    assert mined["source"] == "pro" and mined["sufficient"] is True and mined["matches_mined"] == 100
    for position, role in _POSITION_ROLES.items():
        stats = mined["positions"][position]
        assert stats["sample"] == 200
        for minute, row in stats["cs_curve"].items():
            assert row["sample"] >= 150
            assert pro_patterns.CS_BENCHMARKS[role][int(minute)] == pytest.approx(row["mean"], abs=0.51), (role, minute)
            if int(minute) in pro_patterns.CS_TOP_QUARTER.get(role, {}):
                assert pro_patterns.CS_TOP_QUARTER[role][int(minute)] == pytest.approx(row["p75"], abs=0.76)
        table = pro_patterns.KILL_PARTICIPATION_PCT[role]
        assert table["average"] == round(stats["kill_participation_mean"] * 100)
        assert table["good"] == round(stats["kill_participation_p75"] * 100)
        assert table["poor"] == table["average"] - 15
    assert set(pro_patterns.CS_TOP_QUARTER) == {role for role, minutes in pro_patterns._CS_COMPARE_MINUTES.items()
                                                if minutes}
    # The carry's 10-minute number is the one the laning principles name
    from backend.app.ai import knowledge_base
    assert "65+ CS by 10 minutes" in knowledge_base._LANING_PRINCIPLES_EN["carry"]
    assert pro_patterns.CS_BENCHMARKS["carry"][10] == 65


def test_no_hero_has_a_last_hit_curve_of_its_own():
    """Ten hand-written curves were adjustments of the old role rows ("Axe 30 at 10:00")."""
    for hero in ("Zeus", "Axe", "Anti-Mage", "Juggernaut"):
        for role, row in pro_patterns.CS_BENCHMARKS.items():
            assert pro_patterns.get_cs_benchmark(hero, role, 10) == row[10]
            assert pro_patterns.get_cs_benchmark(hero, role, 20) == row[20]
    assert pro_patterns.get_cs_benchmark("Axe", "no_such_role", 10) == pro_patterns.CS_BENCHMARKS["offlane"][10]
    assert "Zeus" not in pro_patterns.get_benchmark_context("Zeus", "mid", 600, lang="en").split("OpenDota")[0]


def test_offlaner_is_rated_against_what_pro_offlaners_farm():
    """With 35 as the 10-minute benchmark, 10 of 16 real offlaners were "excellent" on last hits."""
    players = default_players()
    offlaner = players[2]
    offlaner.lh_t = [round(4.5 * m) for m in range(0, 40)]          # 45 at 10:00, 90 at 20:00
    match = make_match(players)
    rows = {c.metric: c for c in pro_patterns.compare_player(offlaner, match, lang="en", position=3).comparisons}
    ten = rows["cs_10min"]
    assert (ten.actual_value, ten.benchmark_avg, ten.benchmark_good) == (45, 55, 64)
    assert ten.rating == "below_average" and ten.benchmark_source == pro_patterns.SOURCE_PRO
    assert ten.description == "45 CS at 10 min, 10 below benchmark (55)"
    assert (rows["cs_20min"].benchmark_avg, rows["cs_20min"].benchmark_good) == (150, 174)
    # Kill participation: the mean of the position in pro matches, its top quarter as "good"
    kp = rows["kill_participation"]
    assert (kp.benchmark_avg, kp.benchmark_good, kp.benchmark_poor) == (63, 72, 48)
    assert "Offlane (Pos 3) benchmark 63%" in kp.description and kp.benchmark_source == pro_patterns.SOURCE_PRO


def test_mined_pro_matches_of_this_machine_are_the_benchmark(mined_dir):
    payload = _mined(mined_dir)
    three = payload["positions"]["3"]
    three["cs_curve"]["10"].update(mean=58.4, p75=66.0)
    three["cs_curve"]["30"]["sample"] = 12                          # too few games reached minute 30
    three.update(kill_participation_mean=0.661, kill_participation_p75=0.74)
    (mined_dir / "_mining_aggregates.json").write_text(json.dumps(payload), encoding="utf-8")
    assert pro_patterns.get_cs_benchmark("Tidehunter", "offlane", 10) == 58.4
    assert pro_patterns.get_cs_benchmark("Tidehunter", "offlane", 30) == pro_patterns.CS_BENCHMARKS["offlane"][30]
    assert pro_patterns.get_cs_benchmark("Tidehunter", "offlane", 8) == pytest.approx(20 + (58.4 - 20) * 3 / 5, abs=0.1)
    players = default_players()
    offlaner = players[2]
    offlaner.lh_t = [6 * m for m in range(0, 40)]
    rows = {c.metric: c for c in pro_patterns.compare_player(offlaner, make_match(players), lang="en",
                                                             position=3).comparisons}
    assert (rows["cs_10min"].benchmark_avg, rows["cs_10min"].benchmark_good) == (58.4, 66.0)
    assert (rows["kill_participation"].benchmark_avg, rows["kill_participation"].benchmark_good,
            rows["kill_participation"].benchmark_poor) == (66, 74, 51)
    assert "~58" in pro_patterns.get_benchmark_context("Tidehunter", "offlane", 600, lang="en")
    # Aggregates of an older miner hold minute 10 only: that point is laid over the table
    del three["cs_curve"]
    three.update(cs_at_10_mean=60.2, cs_at_10_p75=67.0, cs_at_10_sample=198)
    del three["kill_participation_p75"]
    (mined_dir / "_mining_aggregates.json").write_text(json.dumps(payload), encoding="utf-8")
    os.utime(mined_dir / "_mining_aggregates.json", (1, 1))        # another mtime: the file is read again
    assert pro_patterns.get_cs_benchmark("Tidehunter", "offlane", 10) == 60.2
    assert pro_patterns.get_cs_benchmark("Tidehunter", "offlane", 20) == pro_patterns.CS_BENCHMARKS["offlane"][20]
    assert pro_patterns._kill_participation_bench("offlane") == {"good": 75, "average": 66, "poor": 51}


@pytest.mark.parametrize("changes", [
    {"sufficient": False},                      # the miner itself found them too thin
    {"source": "high_mmr"},                     # public matches are not the pro reference
    {"positions": {"3": {"sample": 12, "cs_at_10_mean": 90.0, "cs_at_10_sample": 12,
                         "kill_participation_mean": 0.9}}},
    {"positions": "broken"},
])
def test_thin_or_foreign_aggregates_are_not_a_benchmark(mined_dir, changes):
    _mined(mined_dir, **changes)
    assert pro_patterns.get_cs_benchmark("Tidehunter", "offlane", 10) == pro_patterns.CS_BENCHMARKS["offlane"][10]
    assert pro_patterns._kill_participation_bench("offlane") == pro_patterns.KILL_PARTICIPATION_PCT["offlane"]
    (mined_dir / "_mining_aggregates.json").write_text("{not json", encoding="utf-8")
    os.utime(mined_dir / "_mining_aggregates.json", (2, 2))
    assert pro_patterns.get_cs_benchmark("Tidehunter", "offlane", 10) == pro_patterns.CS_BENCHMARKS["offlane"][10]


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_positions_4_and_5_get_no_last_hit_line_in_the_benchmark_text(lang):
    """A live report added a 4's denies to his last hits to "reach the position-4 benchmark (about 10)"."""
    cs = "CS at" if lang == "en" else "补刀"
    for role in ("carry", "mid", "offlane"):
        text = pro_patterns.get_benchmark_context("Zeus", role, 600, lang=lang)
        assert sum(cs in line for line in text.splitlines()) == 1
        assert f"~{pro_patterns.CS_BENCHMARKS[role][10]}" in text
    for role in ("support_4", "support_5"):
        text = pro_patterns.get_benchmark_context("Zeus", role, 600, lang=lang)
        assert cs not in text and ("GPM" in text) and ("deaths" in text or "死亡" in text)


def test_supports_get_no_cs_comparison():
    players = default_players()
    for p in players:
        p.lh_t = list(range(0, 40))
    match = make_match(players)
    for slot, position in ((3, 4), (4, 5)):
        report = pro_patterns.compare_player(next(p for p in players if p.player_slot == slot),
                                             match, lang="en", position=position)
        assert not [c for c in report.comparisons if c.metric.startswith("cs_")]
    core = pro_patterns.compare_player(players[0], match, lang="en", position=1)
    assert [c.metric for c in core.comparisons if c.metric.startswith("cs_")] == ["cs_10min", "cs_20min", "cs_30min"]


_ITEMS = {
    "blink": {"key": "blink", "cost": 2250},
    "hand_of_midas": {"key": "hand_of_midas", "cost": 2200},
    "maelstrom": {"key": "maelstrom", "cost": 2950, "components": ["mithril_hammer", "javelin"]},
    "mithril_hammer": {"key": "mithril_hammer", "cost": 1600},
    "javelin": {"key": "javelin", "cost": 900},
    "black_king_bar": {"key": "black_king_bar", "cost": 4050, "components": ["ogre_axe", "mithril_hammer"]},
    "ogre_axe": {"key": "ogre_axe", "cost": 1000},
    "radiance": {"key": "radiance", "cost": 4700, "components": ["relic", "talisman_of_evasion"]},
    "relic": {"key": "relic", "cost": 3400},
    "talisman_of_evasion": {"key": "talisman_of_evasion", "cost": 1300},
    "phase_boots": {"key": "phase_boots", "cost": 1450, "components": ["boots"]},
    "boots": {"key": "boots", "cost": 500},
}


def _stub_items(monkeypatch, with_components=True):
    table = {k: ({**v} if with_components else {x: y for x, y in v.items() if x != "components"})
             for k, v in _ITEMS.items()}
    monkeypatch.setattr(pro_patterns, "_item_table", lambda: table)
    monkeypatch.setattr(pro_patterns, "_item_info", lambda identifier: table.get(identifier))


def _timings(player, match, position):
    report = pro_patterns.compare_player(player, match, lang="en", position=position)
    return {c.metric: c for c in report.comparisons if c.metric.startswith("item_")}


def test_item_timing_is_only_rated_in_the_purchase_order_it_assumes(monkeypatch):
    _stub_items(monkeypatch)
    match = make_match()
    # Real Dragon Knight build: Midas and Maelstrom first, Blink third, BKB fourth.
    mid = make_player(1, "Puck", lane=2, lane_role=2, items=[
        item("Hand of Midas", "hand_of_midas", 579), item("Mithril Hammer", "mithril_hammer", 827),
        item("Maelstrom", "maelstrom", 857), item("Blink Dagger", "blink", 981),
        item("Black King Bar", "black_king_bar", 1488),
    ])
    assert _timings(mid, match, 2) == {}       # neither is bought when its timing assumes
    # Blink first, BKB second: both timings apply.
    mid.items = [
        item("Boots of Speed", "boots", 120), item("Phase Boots", "phase_boots", 400),
        item("Blink Dagger", "blink", 640), item("Ogre Axe", "ogre_axe", 800),
        item("Mithril Hammer", "mithril_hammer", 900), item("Black King Bar", "black_king_bar", 1000),
    ]
    rated = _timings(mid, match, 2)
    assert set(rated) == {"item_blink_dagger", "item_black_king_bar"}
    assert rated["item_blink_dagger"].value_text == "10:40"
    assert rated["item_blink_dagger"].benchmark_text == "13:00"
    # A component is not an item of its own: Radiance after its Sacred Relic is still the first item.
    carry = make_player(0, "Juggernaut", lane=1, lane_role=1, items=[
        item("Phase Boots", "phase_boots", 510), item("Sacred Relic", "relic", 1059),
        item("Talisman of Evasion", "talisman_of_evasion", 1157), item("Radiance", "radiance", 1176),
    ])
    assert set(_timings(carry, match, 1)) == {"item_radiance"}
    # BKB as the very first item is not compared with the second-item timing either.
    carry.items = [item("Black King Bar", "black_king_bar", 900)]
    assert _timings(carry, match, 1) == {}


def test_every_decision_context_carries_the_positions_of_the_match():
    """The prompt builders name each hero's role from this, not from the hero's usual role."""
    match = make_match()
    derived = positions.derive_match_positions(match.players)
    decisions = [DecisionPoint(time=905, point_type="good_play", player_slot=1, hero_name="Puck",
                               severity="positive", brief="Achieved a Double Kill")]
    ctx = coach._context_step(match, decisions, {}, 2, "en", None, derived)[0][0]
    by_team = ctx.match_context["positions"]
    assert by_team == coach._positions_by_team(match, derived)
    assert [n for _hero, n in by_team["radiant"]] == [1, 2, 3, 4, 5] == [n for _hero, n in by_team["dire"]]
    assert ["Puck", 2] in by_team["radiant"] and ["Lina", 2] in by_team["dire"]
    # Heroes without a derived position are left out; without any, the context has no such key.
    del derived[1]
    assert "Puck" not in [hero for hero, _n in coach._positions_by_team(match, derived)["radiant"]]
    assert "positions" not in coach._context_step(match, decisions, {}, 2, "en")[0][0].match_context
    data = asyncio.run(coach.generate_coaching_report(match, 1, lang="en")).to_dict()
    assert "positions" not in data                # prompt material, not part of the report


def test_metric_and_position_codes_are_plain_lowercase_codes(monkeypatch):
    """Codes must work as keys: no apostrophe from an item name, no "+" in the position method."""
    _stub_items(monkeypatch)
    monkeypatch.setitem(pro_patterns._item_table(), "ultimate_scepter", {"key": "ultimate_scepter", "cost": 4200})
    match = make_match()
    mid = make_player(1, "Puck", lane=2, lane_role=2, items=[item("Aghanim's Scepter", "ultimate_scepter", 880)])
    assert set(_timings(mid, match, 2)) == {"item_aghanims_scepter"}
    report = pro_patterns.compare_player(match.players[0], match, lang="zh", position=1)
    assert all(re.fullmatch(r"[a-z0-9_]+", c.metric) for c in report.comparisons)
    methods = {p.method for p in positions.derive_match_positions(match.players).values()}
    assert methods and methods <= {"lane_nw", "archetype_nw", "nw_only", "unknown"}


def test_item_timing_without_component_data_only_rates_a_provable_first_item(monkeypatch):
    _stub_items(monkeypatch, with_components=False)
    match = make_match()
    mid = make_player(1, "Puck", lane=2, lane_role=2, items=[
        item("Phase Boots", "phase_boots", 400), item("Blink Dagger", "blink", 640),
        item("Black King Bar", "black_king_bar", 1000),
    ])
    assert set(_timings(mid, match, 2)) == {"item_blink_dagger"}
    carry = make_player(0, "Juggernaut", lane=1, lane_role=1, items=[
        item("Sacred Relic", "relic", 1059), item("Radiance", "radiance", 1176),
    ])
    assert _timings(carry, match, 1) == {}     # the relic may be a separate item for all we know
    monkeypatch.setattr(pro_patterns, "_item_info", lambda identifier: None)      # no item data at all
    assert _timings(mid, match, 2) == {}


def test_turbo_match_gets_no_benchmark_judgement():
    assert pro_patterns.benchmarks_apply("ranked_all_pick") and pro_patterns.benchmarks_apply("")
    assert not pro_patterns.benchmarks_apply("turbo") and not pro_patterns.benchmarks_apply("ability_draft")
    match = make_match(game_mode="turbo")
    carry = match.players[0]
    carry.gold_per_min = 844
    report = pro_patterns.format_pro_comparison(pro_patterns.compare_player(carry, match, lang="en", position=1))
    assert report["applicable"] is False and report["comparisons"] == [] and report["partial"] is False
    assert report["summary"] == ("This match was played in Turbo mode: the benchmarks describe standard matches, "
                                 "so no comparison is made.")
    # No label gets an article in front of it ("This is a Ability Draft match")
    for code, label in pro_patterns._MODE_LABELS["en"].items():
        note = pro_patterns._mode_note(code, "en")
        assert note.startswith(f"This match was played in {label} mode: ") and " a " not in note.split(":")[0]
    assert set(pro_patterns._MODE_LABELS["en"]) == set(pro_patterns._MODE_LABELS["zh"]) == pro_patterns._NON_STANDARD_MODES
    zh = pro_patterns.compare_player(carry, match, lang="zh", position=1).summary
    assert "加速模式" in zh and not any("a" <= ch <= "z" for ch in zh.lower())
    assert "Turbo" in pro_patterns.get_benchmark_context("Juggernaut", "carry", 600, lang="en", game_mode="turbo")
    data = _report(match, 0, "en")
    assert data["pro_comparison"]["applicable"] is False
    assert "Turbo" in data["summary"]          # the comparison card is empty, so the summary says why
    assert _report(make_match(), 0, "en")["pro_comparison"]["applicable"] is True


def test_turbo_laning_review_has_no_lane_creep_efficiency():
    """"41 of about 82 lane creeps (50%)" is a standard-game yardstick."""
    def review(mode):
        match = _lane_match()
        match.game_mode = mode
        core = match.players[0]
        core.laning_creep_kills = [LaningCreepKill(time=10 * i, creep_type="melee", gold=40) for i in range(41)]
        pos = positions.derive_match_positions(match.players)
        return coach._generate_laning_review(match, core, [], lang="en", positions=pos)

    normal, turbo = review("ranked_all_pick"), review("turbo")
    assert normal["cs_efficiency"] == "Average (50%)" and "lane creeps (50%)" in normal["summary"]
    assert turbo["cs_efficiency"] == "" and "lane creeps" not in turbo["summary"] and "%" not in turbo["summary"]
    assert turbo["cs_10min"] == normal["cs_10min"] == 62 and turbo["lane_cs_10min"] == 41


def test_comparison_rows_carry_ready_to_show_texts_and_rounded_numbers():
    players = default_players()
    support = players[3]
    support.obs_placed, support.sen_placed, support.hero_damage = 7, 9, 12345
    match = make_match(players, duration=2091)
    for lang in ("en", "zh"):
        rows = pro_patterns.format_pro_comparison(
            pro_patterns.compare_player(support, match, lang=lang, position=4))["comparisons"]
        assert rows
        for row in rows:
            assert row["value_text"] and row["benchmark_text"], row["metric"]
            for key in ("actual_value", "benchmark_good", "benchmark_avg"):
                assert round(row[key], 1) == row[key], (row["metric"], key, row[key])
            if lang == "en":
                assert not CJK_RE.search(row["value_text"] + row["benchmark_text"])
        by_metric = {row["metric"]: row for row in rows}
        assert by_metric["kill_participation"]["value_text"].endswith("%")
        assert by_metric["deaths"]["value_text"] == ("0.9 per 30 min" if lang == "en" else "每30分钟0.9次")
        assert by_metric["obs_wards"]["benchmark_text"] == ("4.0 per 30 min" if lang == "en" else "每30分钟4.0个")


def test_summary_only_says_pro_for_pro_reference_rows(monkeypatch):
    monkeypatch.setattr(pro_patterns, "_load_learned_benchmarks", lambda: {
        # the player's 450 is below the median of Tinker's core games
        "tinker": {"roles": {"core": {"gold_per_min": {"p10": 500, "p50": 600, "p90": 700}}}},
    })
    match = make_match()
    tinker = next(p for p in match.players if p.hero_name == "Tinker")
    tinker.hero_damage, tinker.kills, tinker.assists = 40000, 15, 10
    en = pro_patterns.compare_player(tinker, match, lang="en", position=2).summary
    assert "Gold Per Minute" in en and "below the benchmark" in en and "pro" not in en.split("Needs improvement")[1]
    zh = pro_patterns.compare_player(tinker, match, lang="zh", position=2).summary
    assert "低于基准水平" in zh and "职业" not in zh.split("需要改进")[1]
    # The role tables for GPM and damage are all-rank data too: still no "pro".
    monkeypatch.setattr(pro_patterns, "_load_learned_benchmarks", lambda: {})
    fallback = pro_patterns.compare_player(tinker, match, lang="en", position=2)
    assert "Gold Per Minute (GPM) below the benchmark" in fallback.summary and "pro reference" not in fallback.summary
    # Only a row of a pro-reference table is called a pro reference.
    tinker.gold_per_min, tinker.deaths = 700, 14
    assert "Death Control below the pro reference" in pro_patterns.compare_player(
        tinker, match, lang="en", position=2).summary


def test_item_timings_are_guidelines_not_pro_numbers(monkeypatch):
    """Nothing in this project measures them: the row and its text must not say "pro"."""
    _stub_items(monkeypatch)
    players = default_players()
    axe = players[2]
    axe.items = [item("Blink Dagger", "blink", 1500)]               # later than the 18:00 "late" line
    match = make_match(players)
    for lang, word, never in (("en", "guideline 11:00 (Excellent)/14:00 (Average)", "pro"),
                              ("zh", "参考时间11:00（优秀）/14:00（一般）", "职业")):
        report = pro_patterns.compare_player(axe, match, lang=lang, position=3)
        row = next(c for c in report.comparisons if c.metric == "item_blink_dagger")
        assert row.benchmark_source == pro_patterns.SOURCE_GUIDELINE == "guideline" and row.rating == "poor"
        assert word in row.description and never not in row.description.lower()
        # ... and a summary whose weak row is a guideline does not call it a pro level
        only = pro_patterns.ProPatternReport(hero_name="Axe", role="offlane", role_label="3", comparisons=[row])
        pro_patterns._compute_summary(only, lang=lang)
        assert ("Timing below the benchmark" in only.summary) if lang == "en" else ("低于基准水平" in only.summary)
        assert "pro reference" not in only.summary and "职业" not in only.summary
    sources = {c.benchmark_source for c in pro_patterns.compare_player(axe, match, lang="en", position=3).comparisons}
    assert sources == {"pro_reference", "guideline", "opendota_population"}


def test_gpm_and_damage_tables_are_current_all_rank_percentiles():
    """Audit of 2026-10-10 (7.41f): the old tables rated a median position 5 above 'good'."""
    assert pro_patterns.GPM_BENCHMARKS == {
        "carry": {"good": 750, "average": 620, "poor": 470},
        "mid": {"good": 650, "average": 520, "poor": 400},
        "offlane": {"good": 570, "average": 480, "poor": 380},
        "support_4": {"good": 450, "average": 370, "poor": 295},
        "support_5": {"good": 410, "average": 350, "poor": 280},
    }
    assert pro_patterns.DAMAGE_PER_MIN == {
        "carry": {"good": 980, "average": 660, "poor": 410},
        "mid": {"good": 930, "average": 660, "poor": 440},
        "offlane": {"good": 750, "average": 560, "poor": 400},
        "support_4": {"good": 675, "average": 490, "poor": 350},
        "support_5": {"good": 550, "average": 405, "poor": 295},
    }
    # A median position 5 of today is average, not excellent.
    players = default_players()
    support = players[4]
    support.gold_per_min, support.hero_damage = 350, 405 * 35
    match = make_match(players)
    rows = {c.metric: c for c in pro_patterns.compare_player(support, match, lang="en", position=5).comparisons}
    assert rows["gpm"].rating == rows["damage_per_min"].rating == "average"


@pytest.mark.parametrize("lang, never", [("en", ("pro ", "Pro ", "pro-", "Pro-")), ("zh", ("职业",))])
def test_population_tables_are_never_called_pro(monkeypatch, lang, never):
    monkeypatch.setattr(pro_patterns, "_load_learned_benchmarks", lambda: {})
    for position, role in enumerate(("carry", "mid", "offlane", "support_4", "support_5"), start=1):
        players = default_players()
        player = players[0]
        report = pro_patterns.compare_player(player, make_match(players), lang=lang, position=position)
        rows = {c.metric: c for c in report.comparisons}
        for metric in ("gpm", "damage_per_min"):
            row = rows[metric]
            assert row.benchmark_source == pro_patterns.SOURCE_POPULATION
            assert "OpenDota" in row.description and not any(word in row.description for word in never)
            assert ("all ranks" if lang == "en" else "全分段") in row.description
        lines = pro_patterns.get_benchmark_context(player.hero_name, role, 1200, lang=lang).splitlines()
        metric_lines = [line for line in lines if "GPM" in line or "damage" in line or "伤害" in line]
        assert len(metric_lines) == 2
        for line in metric_lines:
            assert ("all ranks, not pro" if lang == "en" else "全分段") in line
            assert ("非职业数据" in line) if lang == "zh" else ("Pro reference" not in line)
            assert str(pro_patterns.GPM_BENCHMARKS[role]["average"]) in line or str(
                pro_patterns.DAMAGE_PER_MIN[role]["average"]) in line


def test_ward_benchmarks_fit_the_ward_supply():
    """A team gets 2 observers at the start and one more every 135 s: 15 in 30 minutes."""
    supply = 2 + 30 * 60 // 135
    wards = pro_patterns.WARDS_PER_30MIN
    assert supply == 15 and set(wards) == pro_patterns._WARD_ROLES == {"support_4", "support_5"}
    assert sum(row["obs_good"] for row in wards.values()) <= supply
    assert sum(row["sen_good"] for row in wards.values()) <= 3 + 30 * 60 // 70
    assert all(row["obs_avg"] < row["obs_good"] and row["sen_avg"] < row["sen_good"] for row in wards.values())
    # A core is not rated on wards its supports placed: no ward row, no "needs improvement".
    players = default_players()
    offlaner = players[2]
    offlaner.obs_placed = offlaner.sen_placed = 0
    report = pro_patterns.compare_player(offlaner, make_match(players), lang="en", position=3)
    assert not [c for c in report.comparisons if c.metric in ("obs_wards", "sen_wards")]
    assert "Ward" not in report.summary


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_replay_that_ends_early_is_not_rated_on_whole_game_figures(lang):
    """The first 11 minutes of a mid player were rated "GPM 377 vs median 520: Poor ... overall Below Average"."""
    players = default_players()
    mid = players[1]
    mid.lh_t = [6 * m for m in range(0, 11)]                        # 60 last hits at 10:00
    mid.gold_per_min, mid.hero_damage, mid.deaths = 377, 2000, 1
    match = make_match(players, duration=659)
    match.partial, match.winner_known = True, False
    report = pro_patterns.format_pro_comparison(pro_patterns.compare_player(mid, match, lang=lang, position=2))
    assert report["partial"] is True and report["applicable"] is True and report["overall_rating"] == ""
    assert [row["metric"] for row in report["comparisons"]] == ["cs_10min"]         # what the recording covers
    assert report["comparisons"][0]["actual_value"] == 60
    if lang == "en":
        assert report["summary"] == (
            "The replay ends at 10:59, before the game did: only what the recording covers is compared "
            "(10min CS). Whole-game figures such as GPM, damage, deaths and kill participation are not rated.")
    else:
        assert report["summary"] == ("录像在10:59处中断，比赛尚未结束：只对比录像覆盖到的数据（10分钟补刀），"
                                     "GPM、伤害、死亡次数、击杀参与率等全场数据不做评价。")
    # A support, or a core whose recording ends before minute 10: nothing is compared
    support = pro_patterns.compare_player(players[3], match, lang=lang, position=4)
    assert support.partial and not support.applicable and support.comparisons == [] and support.overall_rating == ""
    match.duration = 133
    mid.lh_t = [0, 5, 11]
    early = pro_patterns.compare_player(mid, match, lang=lang, position=2)
    assert not early.applicable and early.comparisons == []
    assert ("2:13" in early.summary) and ("nothing in it can be compared" in early.summary
                                           or "因此不做对比" in early.summary)
    # The report says so next to the grade; the complete match is compared as before
    match.duration = 659
    mid.lh_t = [6 * m for m in range(0, 11)]
    data = _report(match, 1, lang)
    assert data["pro_comparison"]["partial"] is True and report["summary"] in data["summary"]
    whole = make_match(default_players())
    full = pro_patterns.format_pro_comparison(pro_patterns.compare_player(whole.players[1], whole, lang=lang, position=2))
    assert full["partial"] is False and full["overall_rating"] and {"gpm", "deaths"} <= {
        row["metric"] for row in full["comparisons"]}


@pytest.mark.parametrize("lang, label", [("en", "Bot Challenge"), ("zh", "机器人挑战")])
def test_bot_challenge_gets_no_benchmark_judgement(lang, label):
    """Game mode 26 of the current client: its matches are not standard 5v5 games."""
    assert not pro_patterns.benchmarks_apply("bot_challenge") and pro_patterns.benchmarks_apply("ranked_all_pick")
    players = default_players()
    match = make_match(players, game_mode="bot_challenge")
    report = pro_patterns.compare_player(players[0], match, lang=lang, position=1)
    assert report.applicable is False and report.comparisons == [] and label in report.summary
    text = pro_patterns.get_benchmark_context("Juggernaut", "carry", 900, lang=lang, game_mode="bot_challenge")
    assert label in text and "GPM" not in text
    data = _report(match, 0, lang)
    assert data["pro_comparison"]["applicable"] is False and label in data["summary"]


# ---------------------------------------------------------------------------
# Laning review
# ---------------------------------------------------------------------------

def _series(value_at_10):
    return [round(value_at_10 * m / 10) for m in range(0, 36)]


def _lane_match(strip_snapshots=True):
    """Radiant safe lane (Juggernaut + Crystal Maiden) against Axe + Lion."""
    players = default_players()
    numbers = {0: (62, 9, 4200, 4100), 4: (4, 2, 1900, 2600),          # lh, dn, gold, xp at 10:00
               130: (38, 3, 3000, 3300), 131: (6, 1, 1500, 2200)}
    for p in players:
        lh, dn, gold, xp = numbers.get(p.player_slot, (30, 2, 2500, 3000))
        p.lh_t, p.dn_t, p.gold_t, p.xp_t = _series(lh), _series(dn), _series(gold), _series(xp)
        p.denies = 25                       # whole-match total: must not be shown as the 10-minute value
        if strip_snapshots:
            p.laning_snapshots = []
    return make_match(players)


def test_laning_review_reads_ten_minute_values_from_the_series():
    match = _lane_match()
    pos = positions.derive_match_positions(match.players)
    review = coach._generate_laning_review(match, match.players[0], [], lang="en", positions=pos)
    assert (review["cs_10min"], review["denies_10min"]) == (62, 9)
    assert review["lane_result"] == "won"                      # 6100 gold / 6700 XP against 4500 / 5500
    assert (review["lane_gold_diff"], review["lane_xp_diff"]) == (1600, 1200)
    assert review["lane_cs_10min"] is None and review["cs_efficiency"] == ""   # no creep log: no efficiency claim
    assert "62 last hits and 9 denies at 10:00 (Axe: 38)" in review["summary"]
    assert "+1,600 gold" in review["summary"] and "efficiency" not in review["summary"]
    # The lane partner gets the same result; the enemy lane the opposite.
    partner = coach._generate_laning_review(match, match.players[4], [], lang="en", positions=pos)
    enemy = coach._generate_laning_review(match, match.players[7], [], lang="zh", positions=pos)
    assert partner["lane_result"] == "won" and enemy["lane_result"] == "lost"
    assert "落后" in enemy["summary"] and not any("a" <= c <= "z" for c in enemy["summary"].replace("Axe", "")
                                                 .replace("Juggernaut", "").replace("Crystal Maiden", "").lower())
    # cs_10min is the number the benchmark comparison shows.
    row = next(c for c in pro_patterns.compare_player(match.players[0], match, lang="en", position=1).comparisons
               if c.metric == "cs_10min")
    assert row.actual_value == review["cs_10min"]


def test_laning_review_does_not_judge_a_support_on_last_hits():
    match = _lane_match()
    cm = match.players[4]
    cm.laning_creep_kills = [LaningCreepKill(time=100 + i, creep_type="melee", gold=40) for i in range(3)]
    pos = positions.derive_match_positions(match.players)
    for lang, phrase in (("en", "A position 5 is not measured by last hits."), ("zh", "5号位不以补刀数衡量。")):
        review = coach._generate_laning_review(match, cm, [], lang=lang, positions=pos)
        assert review["cs_efficiency"] == "" and phrase in review["summary"]
        assert "%" not in review["summary"]
    core = match.players[0]
    core.laning_creep_kills = (
        [LaningCreepKill(time=10 * i, creep_type="melee", gold=40) for i in range(40)]
        + [LaningCreepKill(time=300, creep_type="unknown", gold=40)]          # an untyped lane creep counts
        + [LaningCreepKill(time=310, creep_type="neutral", gold=60)]
        + [LaningCreepKill(time=650, creep_type="melee", gold=40)]            # after 10:00
    )
    review = coach._generate_laning_review(match, core, [], lang="en", positions=pos)
    assert review["lane_cs_10min"] == 41 and review["cs_efficiency"] == "Average (50%)"
    assert review["cs_10min"] == 62                 # still all last hits at 10:00, not the lane-creep count


def test_laning_review_states_nothing_it_does_not_know():
    # Unparsed OpenDota match: no series, no snapshots.
    players = default_players()
    for p in players:
        p.laning_snapshots, p.lh_t, p.gold_t, p.xp_t = [], [], [], []
    match = make_match(players)
    assert coach._generate_laning_review(match, players[0], [], lang="en") == {}
    assert _report(match, 0, "en")["laning_review"] == {}
    # A game that ended before 10:00 has no 10-minute numbers either.
    assert coach._generate_laning_review(make_match(duration=480), default_players()[0], [], lang="en") == {}
    # Last hits known, denies and lane economy not: no zero, no "even".
    players[0].lh_t = _series(55)
    review = coach._generate_laning_review(match, players[0], [], lang="en")
    assert review["cs_10min"] == 55 and review["denies_10min"] is None
    assert review["lane_result"] is None and review["lane_gold_diff"] is None
    assert "denies" not in review["summary"] and "about even" not in review["summary"]
    # A roamer has no lane to win or lose, and the lane he left is 1 against 2:
    # the totals of unequal groups say nothing.
    lane_match = _lane_match()
    roamer = lane_match.players[3]
    roamer.lane, roamer.lane_role = 4, 4
    assert coach._generate_laning_review(lane_match, roamer, [], lang="en")["lane_result"] is None
    solo = coach._generate_laning_review(lane_match, lane_match.players[2], [], lang="en")
    assert solo["lane_result"] is None and "cannot be judged" in solo["summary"]


# ---------------------------------------------------------------------------
# Improvement areas, strengths, grade, summary
# ---------------------------------------------------------------------------

def test_improvement_areas_are_sentences_built_from_briefs():
    paragraph = "1) Stayed in lane at 40% HP. 2) Ignored three missing heroes; 3) no TP. " * 6   # an LLM "mistake"
    evals = [
        _evaluation("death", "D", time=350, brief="Laning phase: killed by Lion", mistake=paragraph),
        _evaluation("death", "C", time=905, brief="Killed by Wraith King", mistake=paragraph),
        _evaluation("death", "C", time=1300, brief="Killed by Zeus", mistake=paragraph),
        _evaluation("death", "C", time=1500, brief="Killed by Zeus", mistake=paragraph),
        _evaluation("bad_trade", "C", time=200, brief="Unfavorable trade with Lion (took 300, dealt only 40)"),
    ]
    areas = coach._detect_improvement_areas(evals, lang="en")
    assert [a["category_code"] for a in areas] == ["map_awareness"]      # one lost trade is not a pattern
    area = areas[0]
    assert area["occurrences"] == 4 and len(area["examples"]) == 3
    assert area["examples"][0] == "[5:50] Laning phase: killed by Lion"
    assert area["description"].startswith("4 decisions in this area, average grade C. Examples: [5:50] ")
    assert "Stayed in lane" not in area["description"] and len(area["description"]) < 400
    zh = coach._detect_improvement_areas(evals, lang="zh")[0]["description"]
    assert zh.startswith("该方面共有4个决策点，平均评分C。例如：[5:50] ")


def test_laning_is_not_an_improvement_area_for_a_lane_winner_and_advice_matches_the_points():
    trades = [_evaluation("bad_trade", "C", time=t, brief=f"Unfavorable trade {t}", mistake="Poor trade timing")
              for t in (100, 300)]
    assert [a["category_code"] for a in coach._detect_improvement_areas(trades, lang="en")] == ["laning"]
    assert coach._detect_improvement_areas(trades, lang="en", lane_result="won") == []
    advice = coach._detect_improvement_areas(trades, lang="en", lane_result="lost")[0]["description"]
    assert "trade timing" in advice and "last-hitting" not in advice       # no CS advice without a CS point
    with_cs = trades + [_evaluation("missed_cs", "C", time=420, brief="Minute 7: 20 last hits",
                                    mistake="Missed a significant number of last hits")]
    assert "last-hitting" in coach._detect_improvement_areas(with_cs, lang="en")[0]["description"]


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_deaths_without_a_named_mistake_make_no_improvement_area(lang):
    """8762775364 Ogre Magi from OpenDota: two solo deaths nothing is known about became
    "Map Awareness ... check the minimap more frequently"."""
    blank = [_evaluation("death", "C", time=t, brief="Killed by Zeus") for t in (646, 1003, 1500)]
    assert coach._detect_improvement_areas(blank, lang=lang) == []
    stock = [_evaluation("death", "C", time=t, brief="b", mistake=m) for t, m in ((646, "None"), (1003, "无"))]
    assert coach._detect_improvement_areas(stock, lang=lang) == []
    named = blank + [_evaluation("death", "D", time=t, brief="Killed by Lion", mistake="Was caught alone")
                     for t in (700, 1700)]
    area = coach._detect_improvement_areas(named, lang=lang)[0]
    assert area["category_code"] == "map_awareness" and area["occurrences"] == 5
    assert len(area["examples"]) == 2 and all("Killed by Lion" in e for e in area["examples"])


def test_teamfight_death_counts_as_teamfight_and_strengths_are_sentences():
    death = DeathEvent(time=1200, player_slot=0, hero_name="Juggernaut", killer_hero="Lina", is_teamfight=True)
    evals = [
        _evaluation("death", "C", time=1200, brief="Killed by Lina in a teamfight", related_death=death),
        _evaluation("death", "C", time=1500, brief="Killed by Lina in a teamfight", related_death=death),
        _evaluation("good_play", "S", time=0, brief="Achieved a Double Kill", time_known=False),
        _evaluation("good_play", "A", time=900, brief="Successful smoke gank: 2 kills"),
    ]
    assert coach._detect_improvement_areas(evals, lang="en") == []       # teamfight average is B
    assert coach._detect_strengths(evals, lang="en") == []
    strengths = coach._detect_strengths(evals[2:], lang="en")
    assert strengths[0]["description"] == (
        "2 decisions in this area, average grade S. "
        "Examples: [15:00] Successful smoke gank: 2 kills; Achieved a Double Kill.")


def test_overall_grade_does_not_ignore_deaths_missing_from_the_data():
    good = [_evaluation("good_play", "S"), _evaluation("good_play", "A")]
    assert coach._calculate_overall_grade(good) == "S"
    assert coach._calculate_overall_grade(good, unlocated_deaths=9) == "C"     # 0/9/9 is not an S game
    assert coach._calculate_overall_grade([]) == "B"
    assert coach._calculate_overall_grade([], unlocated_deaths=3) == "C"


def test_summary_does_not_state_a_result_when_the_winner_is_unknown():
    match = make_match(winner_known=False)
    for lang, banned in (("en", ("won", "lost")), ("zh", ("获胜", "失败"))):
        summary = _report(match, 1, lang)["summary"]
        assert not any(word in summary for word in banned), summary
    assert "lost" in _report(make_match(), 1, "en")["summary"]         # radiant_win False, Puck is Radiant
    assert "获胜" in _report(make_match(), 129, "zh")["summary"]


def test_report_without_located_deaths_says_so_instead_of_praising():
    players = default_players()
    for p in players:
        p.laning_snapshots, p.lh_t, p.gold_t, p.xp_t = [], [], [], []
    players[1].deaths = 6
    data = _report(make_match(players), 1, "en")
    assert data["death_analyses"] == [] and data["overall_grade"] == "C"
    assert "6 more deaths are on the scoreboard but not in the match data, so they are not analyzed." in data["summary"]
    assert "No individual decision points were found" in data["summary"]
    assert "Solid performance" not in data["summary"]
    zh = _report(make_match(players), 1, "zh")["summary"]
    assert "另有6次死亡" in zh and "评分仅为默认值" in zh


def test_english_counts_are_written_out():
    """146 strings in 100 reports read "N death(s) analyzed", "1 kill(s)", "N more death(s) are"."""
    players = default_players()
    players[1].deaths = 2
    match = make_match(players, kills_log=[kill(900.0, 1, 129, x=100.0, y=100.0)])
    summary = _report(match, 1, "en")["summary"]
    assert "1 death analyzed — review" in summary
    assert "1 more death is on the scoreboard but not in the match data, so it is not analyzed." in summary
    players[1].deaths = 1
    match = make_match(players, kills_log=[kill(900.0, 1, 129, x=100.0, y=100.0)])
    match.smokes = [SmokeEvent(time=700, team="radiant", player_slots=[1, 2], result="kill", kills_gained=1)]
    data = _report(match, 1, "en")
    assert [m["brief"] for m in data["key_moments"] if m["point_type"] == "good_play"] == [
        "Successful smoke gank: 1 kill"]
    assert not [text for _path, text in _strings(data) if "(s)" in text or "(es)" in text]


def test_good_play_before_the_horn_keeps_its_time_in_the_payload():
    """A double kill finished at -3 s (8713267082 has four kills between -10.3 s and 5.4 s) was
    "time unknown": payload time null, sorted after every dated moment."""
    players = default_players()
    players[1].multi_kills = {"2": 1}
    match = make_match(players, kills_log=[kill(-12.4, 128, 1, x=100.0, y=100.0), kill(-3.2, 129, 1, x=100.0, y=100.0)])
    _deaths, decisions = decision_detector.detect_decisions(match, 1, lang="en")
    double, = [d for d in decisions if d.subtype == "multi_kill"]
    assert (double.time, double.time_known) == (-3, True)
    data = _report(match, 1, "en")
    moment = next(m for m in data["key_moments"] if m["brief"] == "Achieved a Double Kill")
    assert moment["time"] == -3
    assert data["key_moments"][0] is moment                       # first in time, not listed as undated
    undated = DecisionPoint(time=0, point_type="good_play", player_slot=1, hero_name="Puck",
                            severity="positive", brief="b", time_known=False)
    assert coach.is_time_unknown(undated) and not coach.is_time_unknown(double)


def test_llm_slots_cover_the_whole_game_when_there_are_more_deaths_than_slots():
    def point(kind, t):
        return DecisionPoint(time=t, point_type=kind, player_slot=0, hero_name="Juggernaut",
                             severity="critical" if kind == "death" else "positive", brief="b")

    deaths = [point("death", 120 * i) for i in range(1, 15)]            # 14 deaths, 2:00 .. 28:00
    points = [point("good_play", 60)] + deaths
    picked = [points[i] for i in coach._llm_slots(points)]
    assert len(picked) == coach.MAX_LLM_EVALS and all(p.point_type == "death" for p in picked)
    assert picked[0].time == 120 and picked[-1].time == 120 * 14        # first and last death included
    assert len({p.time for p in picked}) == coach.MAX_LLM_EVALS
    few = [point("good_play", 60)] + deaths[:3]
    assert [few[i].point_type for i in coach._llm_slots(few)] == ["death", "death", "death", "good_play"]


def test_llm_verdict_is_not_patched_with_rule_text():
    ev = _evaluation("death", "b+")
    ev.better_play = ""
    out = coach._llm_verdict(ev)
    assert (out.score, out.used_llm, out.degraded) == ("B", True, False)
    assert out.mistake == "" and out.better_play == "" and out.lesson == ""
    assert not hasattr(coach, "_fill_missing_fields")
    assert coach._error_text(TimeoutError()) == "TimeoutError" and coach._error_text(ValueError("x")) == "x"


# ---------------------------------------------------------------------------
# Whole reports on every source
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", params=["clarity", "opendota", "mock"])
def source_match(request):
    loader = {"clarity": _clarity_fixture, "opendota": _opendota_fixture, "mock": _mock}[request.param]
    try:
        return request.param, loader()
    except Exception as exc:  # another group owns the parsers and the fixtures
        pytest.skip(f"{request.param} match unavailable: {exc}")


def test_reports_are_consistent_with_the_scoreboard_on_every_source(source_match):
    source, match = source_match
    for p in match.players:
        data = _report(match, p.player_slot, "en")
        json.dumps(data, allow_nan=False)                           # finite numbers only
        located = sum(1 for k in phases.counted_kills(match) if k.victim_slot == p.player_slot)
        assert len(data["death_analyses"]) == located <= p.deaths, p.hero_name
        stats = [r["stats"] for r in data["phase_reviews"]]
        assert sum(s["deaths"] for s in stats) == located, p.hero_name
        assert not any(s["kda_estimated"] for s in stats)
        if source == "opendota":
            assert all(s["assists"] is None for s in stats)         # no assists in OpenDota's log
        else:
            assert (sum(s["kills"] for s in stats), sum(s["assists"] for s in stats)) == (p.kills, p.assists)
            assert sum(s["deaths"] for s in stats) == p.deaths
        for moment in data["key_moments"]:
            assert moment["time"] is None or isinstance(moment["time"], int)
        data["position"].pop("label_zh", None)
        assert not [(path, text) for path, text in _strings(data) if CJK_RE.search(text)]
        assert not [text for _path, text in _strings(data) if "npc_dota" in text]


def test_lane_result_is_the_same_from_replay_and_opendota():
    try:
        clarity, opendota = _clarity_fixture(), _opendota_fixture()
    except Exception as exc:
        pytest.skip(f"fixtures unavailable: {exc}")
    results = []
    for match in (clarity, opendota):
        pos = positions.derive_match_positions(match.players)
        results.append({
            p.hero_name: coach._generate_laning_review(match, p, [], lang="en", positions=pos)["lane_result"]
            for p in match.players
        })
    assert results[0] == results[1]
    assert set(results[0].values()) >= {"won", "lost"}              # not "even" for everybody
    assert results[0]["Legion Commander"] == results[0]["Hoodwink"] == "won"
    assert results[0]["Wraith King"] == results[0]["Lion"] == "lost"
    assert [p.number for p in positions.derive_match_positions(clarity.players).values()] == [
        p.number for p in positions.derive_match_positions(opendota.players).values()]


def test_decision_points_of_a_report_do_not_double_count(source_match):
    _source, match = source_match
    for p in match.players:
        _deaths, decisions = decision_detector.detect_decisions(match, p.player_slot, lang="zh")
        multi = [d for d in decisions if d.subtype == "multi_kill" and d.time_known]
        for fight_point in (d for d in decisions if d.point_type == "fight_engage"):
            tf = next(t for t in match.teamfights if t.start_time - 1 < fight_point.time < t.end_time + 1)
            assert not [m for m in multi if tf.start_time <= m.time <= tf.end_time], p.hero_name


# ---------------------------------------------------------------------------
# Frontend fix pass: report texts that contradicted the page around them
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang", ["en", "zh"])
def test_cs_sentence_quotes_the_benchmark_the_table_shows(lang, mined_dir):
    """Benchmark column "203", sentence under it "117 below benchmark (202)": the column was
    rounded and the sentence truncated (113 of 188 rows of real reports)."""
    payload = _mined(mined_dir)
    payload["positions"]["3"]["cs_curve"]["10"].update(mean=58.6, p75=66.0)
    payload["positions"]["3"]["cs_curve"]["20"].update(mean=150.5, p75=174.0)
    (mined_dir / "_mining_aggregates.json").write_text(json.dumps(payload), encoding="utf-8")
    players = default_players()
    offlaner = players[2]
    offlaner.lh_t = [round(4.5 * m) for m in range(0, 40)]          # 45 at 10:00, 90 at 20:00
    rows = {c.metric: c for c in pro_patterns.compare_player(offlaner, make_match(players), lang=lang,
                                                             position=3).comparisons}
    ten, twenty = rows["cs_10min"], rows["cs_20min"]
    assert (ten.benchmark_avg, ten.benchmark_text, twenty.benchmark_text) == (58.6, "59", "150")
    if lang == "en":
        assert ten.description == "45 CS at 10 min, 14 below benchmark (59)"
        assert twenty.description == "90 CS at 20 min, 60 below benchmark (150)"
    else:
        assert ten.description == "10分钟补刀45，低于基准59约14刀"
    for row in (ten, twenty):
        quoted = [int(n) for n in re.findall(r"\d+", row.description)]
        assert int(row.benchmark_text) in quoted and abs(row.actual_value - int(row.benchmark_text)) in quoted


def test_a_scoreboard_only_match_has_no_phase_reviews():
    """OpenDota match 271145478 (never parsed): five phase cards, each "Kills / deaths (est.) 0/0",
    for a hero that finished 17/2/6."""
    players = default_players()
    for p in players:
        p.gold_t, p.xp_t, p.lh_t = [], [], []
    players[1].kills, players[1].deaths, players[1].assists = 17, 2, 6
    bare = make_match(players, duration=2625)
    assert not phases.has_timed_data(bare, players[1]) and not phases.has_timed_data(bare, players[1], [])
    assert coach._phase_stats_step(bare, players[1], [], "en") == []
    report = _report(bare, players[1].player_slot)
    assert report["phase_reviews"] == [] and report["overall_grade"]
    # anything with a time brings the phases back: a series, a located death, a kill log
    assert phases.has_timed_data(bare, players[1], [700])
    players[1].gold_t = [400 * m for m in range(45)]
    assert phases.has_timed_data(bare, players[1])
    assert [s.phase for s in coach._phase_stats_step(bare, players[1], [], "en")] == [
        "laning", "early_mid", "mid_late", "late", "ultra_late"]
    assert len(_report(bare, players[1].player_slot)["phase_reviews"]) == 5


@pytest.mark.parametrize("lang,note", [("en", "Note: this match was played on patch 6.78"), ("zh", "这场比赛的版本是6.78")])
def test_a_match_of_another_era_is_rated_with_a_caveat(lang, note):
    """Match 271145478 (2013, patch 6.78): "GPM 535 ... below the OpenDota median 644" with no word
    that the median is the one of today's game."""
    from backend.app.ai import data_manager

    current = data_manager.get_current_patch()
    major, minor = (int(part) for part in current.split("."))
    players = default_players()
    old = make_match(players, patch="6.78")
    summary = pro_patterns.compare_player(players[1], old, lang=lang, position=2).summary
    assert note in summary and current in summary
    for near in (current, f"{major}.{minor - 2}", ""):          # the same game, or a version nobody knows
        match = make_match(players, patch=near)
        assert pro_patterns._era_note(match, lang) == ""
        assert "6.78" not in pro_patterns.compare_player(players[1], match, lang=lang, position=2).summary
    assert pro_patterns._era_note(make_match(players, patch=f"{major}.{minor - 3}"), lang)
    assert pro_patterns._era_note(make_match(players, patch="event build"), lang) == ""
