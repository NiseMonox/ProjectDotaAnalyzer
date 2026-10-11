"""Decision detection: deaths from the kill log, fight notes, good plays, trades.

Synthetic matches (world units) plus the real-match fixtures of the parsing
tests and the mock match; no LLM, no network, no JVM.
"""
from __future__ import annotations

import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import copy
import json
from pathlib import Path

import pytest

from backend.app.ai import decision_detector, phases, positions
from backend.app.ai.models import DecisionPoint
from backend.app.models import KillEvent, LaningDamageEvent, TeamfightEvent, TeamfightPlayer
from test_ai_pipeline_helpers import CJK_RE, default_players, make_match, snapshots

FIXTURES = Path(__file__).parent / "fixtures"
MATCH_ID = 8762775364

_ARCHETYPES = {
    "Juggernaut": 1, "Phantom Assassin": 1, "Puck": 2, "Lina": 2, "Axe": 3,
    "Rubick": 4, "Lion": 4, "Crystal Maiden": 5, "Witch Doctor": 5,
}


@pytest.fixture(autouse=True)
def _isolate_knowledge(monkeypatch):
    """Keep the knowledge layer (another group's code) out of these tests."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(positions, "hero_typical_position", lambda hero: _ARCHETYPES.get(hero, 0))


def kill(time, victim, killer=None, x=None, y=None, **kw) -> KillEvent:
    return KillEvent(time=time, victim_slot=victim, killer_slot=killer, x=x, y=y, **kw)


def fight(start, end, records=(), radiant_deaths=0, dire_deaths=0) -> TeamfightEvent:
    """A teamfight window; ``records`` are (slot, hero, extra TeamfightPlayer fields)."""
    players = []
    for slot, hero, extra in records:
        data = dict(player_slot=slot, hero_name=hero, x=0.0, y=0.0, damage_dealt=600,
                    damage_taken=600, kills=0, deaths=0, abilities_used=[])
        data.update(extra)
        players.append(TeamfightPlayer(**data))
    return TeamfightEvent(start_time=start, end_time=end, radiant_deaths=radiant_deaths,
                          dire_deaths=dire_deaths, players=players)


def _player(match, slot):
    return next(p for p in match.players if p.player_slot == slot)


def _points(match, slot, lang="en", kind=None):
    _deaths, decisions = decision_detector.detect_decisions(match, slot, lang=lang)
    return [d for d in decisions if kind is None or d.point_type == kind]


# ---------------------------------------------------------------------------
# Deaths from the kill log
# ---------------------------------------------------------------------------

def test_deaths_come_from_the_kill_log_with_real_killer_position_and_assists():
    players = default_players()
    puck = players[1]
    puck.deaths = 2
    puck.killed_by = {"Axe": 2}              # stale / misleading counters must not matter
    snaps = snapshots(-500, -500, until=600)
    for s in snaps:                          # dead for 25 s: used to become three deaths
        if 400 <= s.time <= 425:
            s.hp = 0
    puck.laning_snapshots = snaps
    match = make_match(players, kills_log=[
        kill(400.6, 1, 129, x=2500.0, y=3800.0, assist_slots=[128]),
        kill(900.2, 1, None, x=-1500.0, y=-1300.0, killer_unit="npc_dota_badguys_tower1_mid"),
    ])
    deaths, decisions = decision_detector.detect_decisions(match, 1, lang="en")
    assert [(d.time, d.source) for d in deaths] == [(400, "kills_log"), (900, "kills_log")]
    first, second = deaths
    assert (first.killer_hero, first.killer_slot, first.assist_slots) == ("Lina", 129, [128])
    assert (first.x, first.y) == (2500.0, 3800.0) and first.overextended and not first.is_teamfight
    assert (second.killer_hero, second.killer_slot) == ("", None)
    assert second.killer_unit == "npc_dota_badguys_tower1_mid"
    briefs = [d.brief for d in decisions if d.point_type == "death"]
    # A unit is written as in the analysis ("was killed by a tower"), not as a label.
    assert briefs == ["Laning phase: killed by Lina (overextended)", "Killed by a tower"]
    zh = [d.brief for d in _points(match, 1, "zh", "death")]
    assert zh == ["对线期被莉娜击杀（位置过深）", "被防御塔击杀"]   # hero names follow the report language


@pytest.mark.parametrize("unit, en, zh", [
    ("npc_dota_neutral_kobold", "Killed by a neutral creep before the creeps met", "兵线接触前被野怪击杀"),
    ("", "Killed by Lina before the creeps met", "兵线接触前被莉娜击杀"),
])
def test_death_before_the_creeps_meet_is_not_called_a_rune_contest(unit, en, zh):
    """The data has no rune information: "Killed by X during the bounty rune contest"."""
    players = default_players()
    players[1].deaths = 1
    entry = kill(-10.3, 1, None if unit else 129, x=-5000.0, y=-4000.0, killer_unit=unit)
    match = make_match(players, kills_log=[entry])
    assert [d.brief for d in _points(match, 1, "en", "death")] == [en]
    assert [d.brief for d in _points(match, 1, "zh", "death")] == [zh]
    unknown = make_match(players, kills_log=[kill(-10.3, 1, None, x=-5000.0, y=-4000.0)])
    assert [d.brief for d in _points(unknown, 1, "en", "death")] == ["Died before the creeps met"]


@pytest.mark.parametrize("victim, side, en, zh", [
    (129, "radiant", "Killed by the Radiant fountain", "被天辉泉水击杀"),      # a Dire hero dives the Radiant base
    (1, "dire", "Killed by the Dire fountain", "被夜魇泉水击杀"),
    (129, None, "Killed by the Radiant fountain", "被天辉泉水击杀"),           # older data: no killer_team
])
def test_death_to_the_fountain_names_the_fountain(victim, side, en, zh):
    """'dota_fountain' used to be printed as 'Dota Fountain' in both languages."""
    from backend.app.ai import causal_chain, context_builder, evaluator
    from backend.app.ai.models import killer_text, unit_text

    players = default_players()
    hero = next(p for p in players if p.player_slot == victim)
    hero.deaths = 1
    home = (-7000.0, -6500.0) if victim >= 128 else (6900.0, 6400.0)       # inside the enemy fountain
    entry = kill(1433.8, victim, None, x=home[0], y=home[1], killer_unit="dota_fountain", killer_team=side)
    match = make_match(players, kills_log=[entry])
    for lang, expected in (("en", en), ("zh", zh)):
        deaths, decisions = decision_detector.detect_decisions(match, victim, lang=lang)
        point = next(d for d in decisions if d.point_type == "death")
        assert point.brief.startswith(expected), point.brief
        assert "Dota Fountain" not in point.brief
        chain = causal_chain.build_causal_chain(match, deaths[0], victim, lang=lang)
        ctx = context_builder.build_context(match, point, causal_chain=chain, lang=lang)
        ev = evaluator.evaluate_decision_fallback(ctx, lang=lang)
        texts = [chain.formatted_text, ev.analysis, *(e.description for e in ctx.event_timeline)]
        assert not [t for t in texts if "Dota Fountain" in t or "dota_fountain" in t]
        label = expected.split("Killed by the ")[-1] if lang == "en" else expected[1:-2]
        assert label in chain.formatted_text and label in ev.analysis
        assert any(label in e.description for e in ctx.event_timeline if e.event_type == "death")
        if lang == "en":
            assert not CJK_RE.search(point.brief + ev.analysis + chain.formatted_text)
            assert f"Killer: {label}" in chain.formatted_text            # the bare label, no article
    # Without a side the fountain is the victim's enemy fountain; other units keep their labels.
    assert (unit_text("dota_fountain", "en"), unit_text("dota_fountain", "zh")) == ("the enemy fountain", "敌方泉水")
    assert unit_text("dota_fountain", "en", article=False) == "Enemy fountain"
    assert unit_text("npc_dota_goodguys_tower1_mid", "en") == "a tower" and unit_text("", "en") == ""
    assert killer_text(None, "en") == ""


def _fountain_dive(death_time: float, *, duration: int, dire_wins: bool, winner_known: bool = True):
    """Lina (Dire, slot 129) walks into the Radiant fountain during a teamfight and dies to it."""
    from backend.app.ai import causal_chain, context_builder, evaluator

    players = default_players()
    lina = next(p for p in players if p.player_slot == 129)
    lina.deaths = 1
    t = int(death_time)
    outside, inside = (-5200.0, -4800.0), (-7000.0, -6500.0)
    lina.position_log = ([[s, int(outside[0]), int(outside[1]), 100] for s in range(t - 60, t - 10, 5)]
                         + [[s, int(inside[0]), int(inside[1]), 100 if s < t else 0] for s in range(t - 10, t + 5, 5)])
    entry = kill(death_time, 129, None, x=inside[0], y=inside[1], killer_unit="dota_fountain", killer_team="radiant")
    # The fight record says "lost" (one Dire death, no Radiant one) and Lina took the most damage.
    tf = fight(t - 30, t + 15, [(129, "Lina", {"deaths": 1, "damage_taken": 2400}),
                                (128, "Phantom Assassin", {}), (0, "Juggernaut", {})], dire_deaths=1)
    match = make_match(players, kills_log=[entry], teamfights=[tf], duration=duration,
                       radiant_win=not dire_wins, winner_known=winner_known)
    out = {}
    for lang in ("en", "zh"):
        deaths, decisions = decision_detector.detect_decisions(match, 129, lang=lang)
        point = next(d for d in decisions if d.point_type == "death")
        chain = causal_chain.build_causal_chain(match, deaths[0], 129, lang=lang)
        ctx = context_builder.build_context(match, point, causal_chain=chain, lang=lang)
        out[lang] = (point, chain, ctx, evaluator.evaluate_decision_fallback(ctx, lang=lang))
    return out


def test_fountain_death_is_a_walk_into_the_fountain_not_a_teamfight_mistake():
    """9038370888, Zeus 26:37: the report blamed "the enemy's first target in a lost fight ... keep distance
    from the enemy initiators" for a hero that walked into the enemy fountain 14 s before his team won."""
    from backend.app.ai import evaluator

    for lang, (point, chain, ctx, ev) in _fountain_dive(1597.8, duration=1611, dire_wins=True).items():
        assert point.brief == ("Killed by the Radiant fountain" if lang == "en" else "被天辉泉水击杀")
        assert [(s.step_type, s.significance) for s in chain.steps] == [
            ("position_forward", "root_cause"), ("death", "final")]
        # outside the fountain area at the 26:22 sample, inside at 26:27: entered on the way between them
        assert 1582 < chain.steps[0].time <= 1587 and chain.steps[1].time == 1597
        assert chain.root_cause_type == "overextend"
        assert ctx.match_context["match_end"] == {"seconds": 14, "won": True}
        assert (ev.score, ev.mistake, ev.better_play) == ("B", "", "")
        text = " ".join((point.brief, ev.analysis, ev.lesson, chain.formatted_text))
        for wrong in ("first target", "initiator", "lost fight", "首要集火", "先手", "输掉的团战", "teamfight", "团战"):
            assert wrong not in text, wrong
        situation = evaluator.format_match_situation(ctx, lang)
        if lang == "en":
            assert "went into the fountain's attack range" in ev.analysis and "won 14s later" in ev.analysis
            assert chain.steps[0].description == "Went into the attack range of the Radiant fountain"
            assert chain.root_cause_summary == "Went into the attack range of the enemy fountain"
            assert "The match ended 14s after this moment: the player's team won" in situation
            assert not CJK_RE.search(text + situation)
        else:
            assert "走进了敌方基地内泉水的攻击范围" in ev.analysis and "14秒后以胜利结束" in ev.analysis
            assert chain.steps[0].description == "走进了天辉泉水的攻击范围"
            assert "这一刻之后14秒比赛结束：玩家所在队伍获胜" in situation


@pytest.mark.parametrize("duration, dire_wins, winner_known, end", [
    (2100, True, True, None),                                  # mid-game dive: nothing ends soon
    (1611, False, True, {"seconds": 14, "won": False}),        # the game ended, but it was lost
    (1611, True, False, {"seconds": 14, "won": None}),         # truncated replay: no result
])
def test_fountain_death_that_was_not_free_names_the_walk_as_the_mistake(duration, dire_wins, winner_known, end):
    from backend.app.ai import evaluator

    dive = _fountain_dive(1597.8, duration=duration, dire_wins=dire_wins, winner_known=winner_known)
    for lang, (_point, _chain, ctx, ev) in dive.items():
        assert ctx.match_context.get("match_end") == end
        assert ev.score == "C" and ev.better_play
        assert ev.mistake == ("Went into the attack range of the enemy fountain" if lang == "en"
                              else "走进了敌方泉水的攻击范围")
        assert "cost nothing" not in ev.analysis and "没有造成损失" not in ev.analysis
        situation = evaluator.format_match_situation(ctx, lang)
        if end is None:
            assert "match ended" not in situation and "比赛结束" not in situation
        elif end["won"] is False:
            assert ("the player's team lost" if lang == "en" else "玩家所在队伍失败") in situation
        else:
            assert ("the result is not in the data" if lang == "en" else "数据中没有胜负结果") in situation


def test_a_moment_long_before_the_end_gets_no_match_end_note():
    from backend.app.ai import context_builder

    match = make_match(duration=2100)
    for t, expected in ((2039, None), (2040, {"seconds": 60, "won": True}), (2100, {"seconds": 0, "won": True})):
        dp = DecisionPoint(time=t, point_type="good_play", player_slot=129, hero_name="Lina",
                           severity="positive", brief="Achieved a Double Kill")
        assert context_builder.build_context(match, dp, lang="en").match_context.get("match_end") == expected


def test_last_hit_of_a_tower_is_said_and_gets_tower_advice():
    """9038370888, Spirit Breaker 8:02: killed by the enemy tier-1 tower, advised to "avoid being focused"."""
    from backend.app.ai import causal_chain, context_builder, evaluator

    players = default_players()
    players[1].deaths = 1
    tower = kill(482.4, 1, None, x=-4700.0, y=5900.0, killer_unit="npc_dota_badguys_tower1_top", killer_team="dire")
    hero = kill(900.0, 1, 129, x=0.0, y=0.0)
    match = make_match(players, kills_log=[tower, hero])
    players[1].deaths = 2
    for lang, said, advice in (
        ("en", "The killing blow came from the tower: the hero was inside its attack range.",
         "Fight outside the enemy tower's range, or step out of it as soon as the tower hits you"),
        ("zh", "最后一击来自防御塔：英雄当时处在敌方防御塔的攻击范围内。", "尽量在敌方防御塔的射程外作战，被塔攻击时立即退出射程"),
    ):
        deaths, decisions = decision_detector.detect_decisions(match, 1, lang=lang)
        by_tower, by_hero = [d for d in decisions if d.point_type == "death"]
        evs = []
        for point, death in ((by_tower, deaths[0]), (by_hero, deaths[1])):
            chain = causal_chain.build_causal_chain(match, death, 1, lang=lang)
            evs.append(evaluator.evaluate_decision_fallback(
                context_builder.build_context(match, point, causal_chain=chain, lang=lang), lang=lang))
        assert said in evs[0].analysis and said not in evs[1].analysis
        if not evs[0].mistake:
            assert evs[0].better_play == advice
        assert evs[1].better_play != advice


def test_death_without_position_is_not_overextended_and_keeps_zero_coordinates():
    players = default_players()
    players[1].deaths = 1
    match = make_match(players, kills_log=[kill(1300.0, 1, 129)])     # OpenDota: no position
    death, = decision_detector.detect_deaths(match, players[1])
    assert (death.x, death.y) == (0.0, 0.0) and not death.overextended
    assert death.killer_hero == "Lina"


def test_deny_and_suicide_do_not_read_as_enemy_kills():
    players = default_players()
    players[1].deaths = 2
    match = make_match(players, kills_log=[
        kill(700.0, 1, 0, x=100.0, y=100.0, deny=True),       # denied by Juggernaut (ally)
        kill(800.0, 1, 1, x=100.0, y=100.0, deny=True),       # own ability
    ])
    deaths, decisions = decision_detector.detect_decisions(match, 1, lang="en")
    assert [d.denied for d in deaths] == [True, True]
    assert deaths[0].killer_hero == "Juggernaut" and deaths[1].killer_hero == ""
    briefs = [d.brief for d in decisions if d.point_type == "death"]
    assert briefs == ["Denied by ally Juggernaut", "Died to their own ability or item"]
    assert not any(d.overextended for d in deaths)


def test_unconfirmed_entries_never_exceed_the_scoreboard():
    """OpenDota logs a reincarnation like a death; the scoreboard decides how many count."""
    players = default_players()
    king, invoker = players[6], players[0]        # Lina (129) plays the reincarnating victim
    king.deaths = 3
    invoker.kills = 2                             # scoreboard: two real kills on the victim
    players[1].kills = 1
    log = [
        kill(290.0, 129, 0, confirmed=False),
        kill(385.0, 129, 1),
        kill(820.0, 129, 0, confirmed=False),
        kill(1241.0, 129, 0, confirmed=False),    # reincarnation
        kill(1711.0, 129, 0, confirmed=False),    # reincarnation
    ]
    match = make_match(players, kills_log=log)
    deaths = decision_detector.detect_deaths(match, king)
    assert [d.time for d in deaths] == [290, 385, 820]
    assert [d.confirmed for d in deaths] == [False, True, False]
    assert len(phases.counted_kills(match)) == 3

    # A "death" followed by another one within seconds is the reincarnation;
    # the victim's scoreboard deaths limit the count even when the killer has kills to spare.
    king.deaths, invoker.kills = 1, 5
    match = make_match(players, kills_log=[
        kill(1500.0, 129, 0, confirmed=False), kill(1505.0, 129, 0, confirmed=False),
    ])
    assert [d.time for d in decision_detector.detect_deaths(match, king)] == [1505]
    spread = make_match(players, kills_log=[
        kill(600.0, 129, 0, confirmed=False), kill(1500.0, 129, 0, confirmed=False),
    ])
    assert [d.time for d in decision_detector.detect_deaths(spread, king)] == [600]


def test_teamfight_flag_needs_another_death_nearby():
    """A fight window is a cluster of deaths in time; a pick-off elsewhere is not part of it."""
    players = default_players()
    for p in players:
        p.deaths = 1
    tf = fight(120, 170, radiant_deaths=1, dire_deaths=3)
    match = make_match(players, teamfights=[tf], kills_log=[
        kill(137.0, 131, 2, x=-2950.0, y=5390.0),      # Lion alone in the top lane
        kill(140.0, 0, 130, x=6500.0, y=-3300.0),
        kill(146.0, 130, 0, x=6760.0, y=-4300.0),
        kill(150.0, 132, 4),                           # no position: the window decides
    ])
    by_slot = {slot: decision_detector.detect_deaths(match, _player(match, slot))[0]
               for slot in (131, 0, 130, 132)}
    assert (by_slot[131].is_teamfight, by_slot[131].teamfight_idx) == (False, -1)
    assert (by_slot[0].is_teamfight, by_slot[0].teamfight_idx) == (True, 0)
    assert by_slot[130].is_teamfight and by_slot[132].is_teamfight
    assert _points(match, 131, "en", "death")[0].brief == "Laning phase: killed by Tinker"
    assert _points(match, 0, "en", "death")[0].brief == "Killed by Axe in a teamfight"


def test_overlapping_fight_windows_pick_the_fight_the_death_is_inside():
    players = default_players()
    players[1].deaths = 1
    # Padded windows of two fights overlap by 15 s; the death is in the second one.
    match = make_match(players, teamfights=[fight(1000, 1045), fight(1030, 1080)],
                       kills_log=[kill(1046.0, 1, 129)])
    death, = decision_detector.detect_deaths(match, players[1])
    assert death.teamfight_idx == 1


# ---------------------------------------------------------------------------
# One event, one decision point
# ---------------------------------------------------------------------------

def _lost_fight_match(position_log, death_time=1215.0, first_death=1215.0, damage=120):
    players = default_players()
    puck = players[1]
    puck.deaths = 1
    players[0].deaths = 1
    puck.position_log = position_log
    tf = fight(1200, 1250, radiant_deaths=2, dire_deaths=0, records=[
        (1, "Puck", dict(deaths=1, damage_dealt=damage, death_time_offset=death_time - 1200)),
        (0, "Juggernaut", dict(deaths=1)),
        (129, "Lina", dict(kills=2)),
    ])
    return make_match(players, teamfights=[tf], kills_log=sorted([
        kill(death_time, 1, 129, x=500.0, y=500.0),
        kill(first_death if first_death != death_time else death_time + 6, 0, 129, x=700.0, y=600.0),
    ], key=lambda k: k.time))


def test_teamfight_death_is_one_decision_point_with_a_factual_note():
    match = _lost_fight_match(position_log=[])
    decisions = _points(match, 1, "en")
    in_fight = [d for d in decisions if 1195 <= d.time <= 1255]
    assert [d.point_type for d in in_fight] == ["death"]         # no fight_engage for the same moment
    assert in_fight[0].brief == "Killed by Lina in a teamfight (dealt only 120 hero damage in the fight)"
    zh = _points(match, 1, "zh", "death")[0].brief
    assert zh == "团战中被莉娜击杀（本次团战仅造成120点英雄伤害）"
    # A real contribution: nothing to remark.
    plain = _points(_lost_fight_match(position_log=[], damage=900), 1, "en", "death")[0].brief
    assert plain == "Killed by Lina in a teamfight"


def test_killed_at_the_start_of_a_lost_fight_only_when_the_data_shows_it():
    healthy = [[1205, 500, 500, 95], [1210, 500, 500, 88], [1215, 500, 500, 0]]
    brief = _points(_lost_fight_match(position_log=healthy), 1, "en", "death")[0].brief
    assert brief == "Killed by Lina in a teamfight (first to die in a lost fight, from 88% HP within seconds)"
    zh = _points(_lost_fight_match(position_log=healthy), 1, "zh", "death")[0].brief
    assert zh == "团战中被莉娜击杀（在这场输掉的团战中最先阵亡，数秒内从88%血量被击杀）"

    worn_down = [[1205, 500, 500, 60], [1210, 500, 500, 35], [1215, 500, 500, 0]]
    assert "first to die" not in _points(_lost_fight_match(position_log=worn_down), 1, "en", "death")[0].brief
    # Second to die, even only two seconds after the first: not "first".
    for first_death in (1215.0, 1233.0):
        late = _lost_fight_match(position_log=[[1225, 500, 500, 90], [1230, 500, 500, 85]],
                                 death_time=1235.0, first_death=first_death)
        assert "first to die" not in _points(late, 1, "en", "death")[0].brief
    # No position log (OpenDota): the HP before the death is unknown, so nothing is claimed.
    assert "first to die" not in _points(_lost_fight_match(position_log=[]), 1, "en", "death")[0].brief


def test_good_fight_with_a_multi_kill_is_listed_once():
    players = default_players()
    jugg = players[0]
    jugg.kills, jugg.deaths = 3, 0
    jugg.multi_kills = {"2": 1, "3": 1}
    jugg.kill_streaks = {"3": 1}
    tf = fight(985, 1040, dire_deaths=3, records=[(0, "Juggernaut", dict(kills=3))])
    match = make_match(players, teamfights=[tf], kills_log=[
        kill(1000.0, 128, 0, x=0.0, y=100.0), kill(1010.0, 129, 0, x=0.0, y=150.0),
        kill(1025.0, 130, 0, x=50.0, y=100.0),
    ])
    positive = [d for d in _points(match, 0, "en") if d.severity == "positive"]
    assert [(d.point_type, d.subtype, d.time, d.brief) for d in positive] == [
        ("good_play", "multi_kill", 1025, "Achieved a Triple Kill"),
        ("good_play", "kill_streak", 1025, "Achieved a Killing Spree streak (3 kills without dying)"),
    ]
    # Without a multi-kill the fight itself is the point, dated at the first kill.
    jugg.multi_kills, jugg.kill_streaks = {}, {}
    spread = make_match(players, teamfights=[tf], kills_log=[
        kill(1000.0, 128, 0, x=0.0, y=100.0), kill(1030.0, 129, 0, x=0.0, y=150.0),
    ])
    tf.players[0].kills = 2
    point, = [d for d in _points(spread, 0, "en") if d.severity == "positive"]
    assert (point.point_type, point.time) == ("fight_engage", 1000)


# ---------------------------------------------------------------------------
# Multi-kills and streaks
# ---------------------------------------------------------------------------

def test_announcement_counters_alone_give_finished_sequences_without_a_time():
    """OpenDota counts every level a streak reached: one 11-streak is one point, not nine."""
    players = default_players()
    jugg = players[0]
    jugg.kill_streaks = {str(n): 2 for n in range(3, 10)} | {"10": 1, "11": 1}
    jugg.multi_kills = {"2": 4, "3": 1, "4": 1, "5": 1}
    points = _points(make_match(players), 0, "en", "good_play")
    assert sorted(d.brief for d in points) == sorted([
        "Achieved a Double Kill"] * 3 + ["Achieved a Rampage",
        "Achieved a Godlike streak (9 kills without dying)",
        "Achieved a Beyond Godlike streak (11 kills without dying)",
    ])
    assert all(not d.time_known and d.time == 0 for d in points)
    # "a Unstoppable streak" / "a Ultra Kill"
    jugg.kill_streaks = {str(n): 1 for n in range(3, 7)}
    jugg.multi_kills = {"2": 1, "3": 1, "4": 1}
    assert sorted(d.brief for d in _points(make_match(players), 0, "en", "good_play")) == [
        "Achieved an Ultra Kill", "Achieved an Unstoppable streak (6 kills without dying)"]


def test_chinese_briefs_use_the_announcements_of_the_game():
    """The coach wrote 四杀 / 五杀（Rampage） and 超神 where the game announces (and the Combat tab of
    the same match shows) 疯狂杀戮 / 暴走 and 超越神的杀戮."""
    players = default_players()
    jugg = players[0]
    for multi, streaks, briefs in (
            ({"2": 1}, {"3": 1}, ["完成双杀", "达成大杀特杀（连续3次击杀未阵亡）"]),
            ({"2": 1, "3": 1}, {}, ["完成三杀"]),
            ({"2": 1, "3": 1, "4": 1}, {str(n): 1 for n in range(3, 10)}, ["完成疯狂杀戮", "达成如同神一般（连续9次击杀未阵亡）"]),
            ({"2": 1, "3": 1, "4": 1, "5": 1}, {str(n): 1 for n in range(3, 12)},
             ["完成暴走", "达成超越神的杀戮（连续11次击杀未阵亡）"])):
        jugg.multi_kills, jugg.kill_streaks = multi, streaks
        points = _points(make_match(players), 0, "zh", "good_play")
        assert sorted(d.brief for d in points) == sorted(briefs)
        assert {d.subtype for d in points} <= {"multi_kill", "kill_streak"}


def test_kill_log_dates_the_sequences_when_it_agrees_with_the_counters():
    players = default_players()
    jugg = players[0]
    jugg.multi_kills = {"2": 2}
    jugg.kill_streaks = {"3": 1, "4": 1}
    log = [kill(500.0, 128, 0), kill(510.0, 129, 0), kill(900.0, 130, 0), kill(915.0, 131, 0)]
    match = make_match(players, kills_log=log)
    points = _points(match, 0, "en", "good_play")
    assert [(d.brief, d.time, d.time_known) for d in points] == [
        ("Achieved a Double Kill", 510, True),
        ("Achieved a Double Kill", 915, True),
        ("Achieved a Dominating streak (4 kills without dying)", 915, True),
    ]
    # The counters disagree with the log (a logged kill was a reincarnation):
    # their count is reported, without inventing a time.
    jugg.multi_kills = {"2": 1}
    points = _points(match, 0, "en", "good_play")
    doubles = [d for d in points if d.subtype == "multi_kill"]
    assert [(d.time, d.time_known) for d in doubles] == [(0, False)]


def test_kill_and_own_death_in_the_same_second_follow_the_counters():
    players = default_players()
    jugg = players[0]
    jugg.deaths = 1
    jugg.kill_streaks = {"3": 1}                 # the game counted the third kill before the death
    log = [kill(100.0, 128, 0), kill(200.0, 129, 0), kill(300.0, 0, 130), kill(300.0, 131, 0)]
    points = _points(make_match(players, kills_log=log), 0, "en", "good_play")
    assert [(d.subtype, d.time, d.time_known) for d in points] == [("kill_streak", 300, True)]


# ---------------------------------------------------------------------------
# Laning trades and CS
# ---------------------------------------------------------------------------

def _trade_match(damage_events, own_events=(), kills_log=(), deaths=0):
    players = default_players()                       # max HP 600 in the snapshots
    players[0].deaths = deaths
    players[7].laning_damage_events = [
        LaningDamageEvent(time=t, target_idx=0, damage=d, inflictor="attack") for t, d in damage_events
    ]
    players[0].laning_damage_events = [
        LaningDamageEvent(time=t, target_idx=7, damage=d, inflictor="attack") for t, d in own_events
    ]
    return make_match(players, kills_log=list(kills_log))


def test_small_exchanges_are_not_bad_trades():
    # 100 damage (17% of max HP) every other minute, never answered.
    match = _trade_match([(t, 100) for t in range(60, 600, 120)])
    assert _points(match, 0, "en", "bad_trade") == []
    # 250 damage in one window (42% of max HP), 60 dealt back: a lost trade.
    lost, = _points(_trade_match([(300, 250)], own_events=[(302, 60)]), 0, "en", "bad_trade")
    assert lost.brief == "Unfavorable trade with Axe (took 250, dealt only 60)"


def test_trade_that_ends_in_a_death_or_in_a_kill_is_not_a_bad_trade():
    died = _trade_match([(300, 250), (310, 200)], kills_log=[kill(318.0, 0, 130, x=4500.0, y=-6000.0)], deaths=1)
    assert _points(died, 0, "en", "bad_trade") == []          # the death point covers it
    assert len(_points(died, 0, "en", "death")) == 1
    won = _trade_match([(300, 250)], kills_log=[kill(312.0, 130, 0, x=4500.0, y=-6000.0)])
    assert _points(won, 0, "en", "bad_trade") == []           # the attacker died: not a lost exchange


def test_low_hp_in_the_own_lane_is_not_an_overextension():
    """8713267082 Bristleback 1:25: a laner at 29% HP where his creeps meet the enemy wave, 3620
    units from his tower, was "pushed too far into unsafe territory"."""
    def points(lane, spot, hp=150):
        players = default_players()
        puck = players[1]
        puck.lane = lane
        puck.laning_snapshots = snapshots(spot[0], spot[1], hp=hp)
        return _points(make_match(players), 1, "en", "overextend")

    creep_line = (-6000.0, 5200.0)                # top lane, 1100 units short of the Dire tier-1 tower
    assert decision_detector.is_overextended(make_match(), "radiant", *creep_line, 100)
    assert points(3, creep_line) == []            # the Radiant offlaner belongs there
    roamer = points(2, creep_line)                # a mid hero up there at low HP is far from home
    assert roamer and roamer[0].brief.startswith("Overextended during laning with low HP (25%)")
    under_tower = (-5600.0, 5900.0)               # 350 units from that tower: a dive
    assert decision_detector.nearest_tower_distance(make_match(), "dire", *under_tower, 100) < 900
    assert points(3, under_tower) != []
    assert points(3, (1500.0, 3500.0)) != []      # low on HP in the enemy jungle
    assert points(3, (1500.0, 3500.0), hp=400) == []


def test_missed_cs_needs_a_half_that_ends_behind_the_curve():
    players = default_players()
    carry = players[0]
    carry.laning_snapshots = []
    # A slow start (2 last hits at minute 3) made up for by minute 6; then on the curve.
    carry.lh_t = [0, 0, 1, 2, 12, 26, 40, 46, 52, 58, 66]
    match = make_match(players)
    assert _points(match, 0, "en", "missed_cs") == []
    carry.lh_t = [0, 0, 1, 2, 5, 8, 12, 16, 20, 24, 28]         # starving the whole lane
    missed = _points(match, 0, "en", "missed_cs")
    assert [d.time for d in missed] == [180, 420]
    # Turbo has different gold and creep rates: the curve does not apply.
    assert _points(make_match(players, game_mode="turbo"), 0, "en", "missed_cs") == []


# ---------------------------------------------------------------------------
# Heuristics (matches without a kill log)
# ---------------------------------------------------------------------------

def test_dead_hero_is_one_death_however_long_the_snapshots_show_zero_hp():
    players = default_players()
    puck = players[1]
    puck.deaths = 4
    snaps = snapshots(-500, -500, until=600)
    for s in snaps:
        if 345 <= s.time <= 370 or 540 <= s.time <= 565:      # two deaths, 25 s dead each
            s.hp = 0
    puck.laning_snapshots = snaps
    deaths = decision_detector.detect_deaths(make_match(players), puck)
    assert [d.time for d in deaths] == [345, 540]
    assert all(d.source == "laning" for d in deaths)


def test_killer_is_only_named_when_the_data_proves_it():
    def detect(killed_by, deaths):
        players = default_players()
        puck = players[1]
        puck.killed_by, puck.deaths = killed_by, deaths
        snaps = snapshots(-500, -500, until=600)
        for s in snaps:
            if s.time in (200, 400):
                s.hp = 0
        puck.laning_snapshots = snaps
        return decision_detector.detect_deaths(make_match(players), puck)

    assert [d.killer_hero for d in detect({"Lina": 1, "Axe": 1}, 2)] == ["", ""]   # which death was whose?
    both = detect({"Lina": 2}, 2)
    assert [(d.killer_hero, d.killer_slot) for d in both] == [("Lina", 129), ("Lina", 129)]
    assert [d.killer_hero for d in detect({"Lina": 2}, 3)] == ["", ""]             # one death had no hero killer
    assert [d.killer_hero for d in detect({}, 2)] == ["", ""]


def test_unlocated_teamfight_death_keeps_its_only_record():
    """Without a kill log a death may only show up in the fight record."""
    players = default_players()
    players[1].deaths = 1
    tf = fight(1200, 1250, radiant_deaths=1, records=[
        (1, "Puck", dict(deaths=1, damage_dealt=50, death_time_offset=None)),
    ])
    points = _points(make_match(players, teamfights=[tf]), 1, "en")
    assert [(d.point_type, d.severity) for d in points if d.time >= 1200] == [("fight_engage", "major")]


# ---------------------------------------------------------------------------
# Real match (parsing fixtures) and the mock: every source
# ---------------------------------------------------------------------------

def _clarity_fixture():
    from backend.app.parser import ClarityParser

    raw = json.loads((FIXTURES / f"clarity_v3_{MATCH_ID}_trimmed.json").read_text(encoding="utf-8"))
    return ClarityParser()._transform_data(copy.deepcopy(raw))


def _opendota_fixture():
    from backend.app import parser
    from backend.app.parser import OpenDotaParser

    raw = json.loads((FIXTURES / f"opendota_{MATCH_ID}.json").read_text(encoding="utf-8"))
    constants = json.loads((FIXTURES / "opendota_constants.json").read_text(encoding="utf-8"))
    return OpenDotaParser()._transform(copy.deepcopy(raw), parser.load_hero_map(), parser._item_id_map(),
                                       constants["ability_ids"], constants["item_ids"], constants["patch"])


def _mock():
    from backend.app.mock_data import generate_mock_match

    return generate_mock_match()


@pytest.fixture(scope="module", params=["clarity", "opendota", "mock"])
def source_match(request):
    loader = {"clarity": _clarity_fixture, "opendota": _opendota_fixture, "mock": _mock}[request.param]
    try:
        return request.param, loader()
    except Exception as exc:  # another group owns the parsers and the fixtures
        pytest.skip(f"{request.param} match unavailable: {exc}")


def test_every_source_gets_its_deaths_with_the_logged_killer(source_match):
    source, match = source_match
    names = {p.player_slot: p.hero_name for p in match.players}
    log = phases.counted_kills(match)
    for p in match.players:
        deaths, decisions = decision_detector.detect_decisions(match, p.player_slot, lang="en")
        mine = [k for k in log if k.victim_slot == p.player_slot]
        assert len(deaths) == len(mine) <= p.deaths, p.hero_name
        if source != "opendota":                    # OpenDota logs no death without a hero killer
            assert len(deaths) == p.deaths, p.hero_name
        for death, entry in zip(deaths, mine):
            assert death.source == "kills_log" and death.time == int(entry.time)
            assert death.killer_slot == entry.killer_slot
            assert death.killer_hero == (names[entry.killer_slot] if entry.killer_slot is not None else "")
        # one decision point per death, and no second point for the same fight
        death_points = [d for d in decisions if d.point_type == "death"]
        assert [d.related_death for d in death_points] == deaths
        assert not [d for d in decisions if d.point_type == "fight_engage" and d.severity != "positive"]
        for d in decisions:
            assert not CJK_RE.search(d.brief) and "npc_dota" not in d.brief, d.brief


def test_real_match_from_opendota_is_no_longer_without_deaths():
    """Ogre Magi 0/9/9: analysed by match id it had no death at all (and grade B)."""
    try:
        match = _opendota_fixture()
    except Exception as exc:
        pytest.skip(f"OpenDota fixture unavailable: {exc}")
    ogre = next(p for p in match.players if p.hero_name == "Ogre Magi")
    assert (ogre.kills, ogre.deaths, ogre.assists) == (0, 9, 9)
    deaths = decision_detector.detect_deaths(match, ogre)
    assert len(deaths) == 9 and all(d.killer_hero for d in deaths)
    # Wraith King: five logged "deaths", three real ones on the scoreboard.
    king = next(p for p in match.players if p.hero_name == "Wraith King")
    assert len([k for k in match.kills_log if k.victim_slot == king.player_slot]) > king.deaths
    assert len(decision_detector.detect_deaths(match, king)) == king.deaths == 3
