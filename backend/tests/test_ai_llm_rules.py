"""Rule-based evaluator, prompt text and verdict parser.

Contexts are built by hand so every fact is under control: the rule text
must state exactly what the context supports (state before a death, one head
count, fog vs. no position data, display names) and nothing else.
"""
from __future__ import annotations

import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import re

import pytest

from backend.app.ai import evaluator
from backend.app.ai.models import (
    CausalChain,
    DeathEvent,
    DecisionContext,
    DecisionPoint,
    EventEntry,
    HeroState,
    PhaseReview,
    VisionAnalysis,
    VisionState,
    VisionTimeline,
)
from test_ai_pipeline_helpers import CJK_RE

# Three or more lower-case English words in a row: an English sentence fragment.
ENGLISH_RUN_RE = re.compile(r"\b[a-z]{2,}(?: [a-z]{2,}){2,}\b")
HEAD_COUNT_RE = re.compile(r"\dv\d")


def hero(name: str, slot: int, **kw) -> HeroState:
    data = dict(hero_name=name, player_slot=slot, team="radiant" if slot < 128 else "dire",
                x=1000.0, y=1000.0, level=5)
    data.update(kw)
    return HeroState(**data)


def vision(*, missing=(), unknown=(), danger="safe", longest=0, reasons=(), critical=None,
           safe_zone=True, wards=True, time=350, far=(), far_longest=0) -> VisionAnalysis:
    """``missing`` are hidden within reach of the player, ``far`` hidden elsewhere on the map."""
    state = VisionState(
        time=time, is_night=False, allied_observer_count=1,
        missing_enemy_count=len(missing) + len(far), missing_enemy_names=list(missing) + list(far),
        unknown_enemy_count=len(unknown), unknown_enemy_names=list(unknown),
        longest_missing_seconds=max(longest, far_longest),
        threat_enemy_count=len(missing), threat_enemy_names=list(missing), threat_longest_seconds=longest,
        danger_level=danger, danger_reasons=list(reasons),
        player_in_safe_zone=safe_zone, nearby_ward_coverage=wards,
    )
    timeline = VisionTimeline()
    if critical:
        timeline.critical_moment, timeline.critical_description = critical
    return VisionAnalysis(state_at_decision=state, timeline=timeline, formatted_text="## Vision Analysis\nbody")


def death_ctx(*, time=350, phase="laning", killer="Lion", killer_unit="", is_tf=False, player=None,
              allies=(), enemies=(), mc=None, map_state=None, va=None, chain=None, lang="en",
              overextended=False) -> DecisionContext:
    death = DeathEvent(time=time, player_slot=1, hero_name="Legion Commander", killer_hero=killer,
                       killer_unit=killer_unit, x=1000.0, y=1000.0, is_teamfight=is_tf,
                       teamfight_idx=0 if is_tf else -1, overextended=overextended)
    brief = "Killed in lane" if lang == "en" else "对线期阵亡"
    dp = DecisionPoint(time=time, point_type="death", player_slot=1, hero_name="Legion Commander",
                       severity="critical", brief=brief, related_death=death)
    ctx = DecisionContext(
        decision_point=dp, player_state=player or hero("Legion Commander", 1),
        ally_states=list(allies), enemy_states=list(enemies), game_phase=phase,
        match_context={"team": "radiant", **(mc or {})}, map_state=map_state or {}, vision_analysis=va,
    )
    if chain is not None:
        ctx.causal_chain = CausalChain(death_event=death, root_cause_type=chain[0], root_cause_summary=chain[1],
                                       formatted_text="## Death Causal Chain\nsteps")
    return ctx


def rules(ctx, lang="en"):
    return evaluator.evaluate_decision_fallback(ctx, lang=lang)


def all_text(ev) -> str:
    return " ".join([ev.analysis, ev.mistake, ev.better_play, ev.lesson])


# ---------------------------------------------------------------------------
# HP before a death (A3)
# ---------------------------------------------------------------------------

def test_death_hp_is_the_state_before_the_death():
    """8762775364, Legion Commander 5:50: 406/978 HP five seconds earlier, not "0%"."""
    player = hero("Legion Commander", 1, hp=406, max_hp=978, state_time=345)
    en = rules(death_ctx(player=player, killer="lion"))
    assert "42%" in en.analysis and "5s before the death" in en.analysis
    assert "0%)" not in en.analysis and "killed by Lion" in en.analysis and "lion" not in en.analysis
    zh = rules(death_ctx(player=player, lang="zh"), "zh")
    assert "阵亡前5秒" in zh.analysis and "42%" in zh.analysis and "0%）" not in zh.analysis


def test_low_hp_before_a_laning_death_is_named():
    player = hero("Legion Commander", 1, hp=180, max_hp=978, state_time=347)
    ev = rules(death_ctx(player=player))
    assert "18%" in ev.analysis and "3s before the death" in ev.analysis
    assert "low HP" in ev.mistake
    # Killed from a healthy HP bar: how is not in the data, so no mistake is guessed.
    healthy = rules(death_ctx(player=hero("Legion Commander", 1, hp=406, max_hp=978, state_time=345)))
    assert "42%" in healthy.analysis and healthy.mistake == "" and healthy.better_play


def test_hp_percentage_from_the_position_log_is_used():
    """After the laning stage only the percentage is known (HeroState.hp_pct, pre_death_hp_pct)."""
    player = hero("Legion Commander", 1, state_time=345)
    player.hp_pct = 64
    ev = rules(death_ctx(player=player, mc={"pre_death_hp_pct": 64, "seconds_before_death": 5}))
    assert "64%" in ev.analysis and "5s before" in ev.analysis
    line = evaluator.format_player_state(death_ctx(player=player, mc={"seconds_before_death": 5}), "en")
    assert "HP 64%" in line and "HP 0" not in line


def test_position_log_percentage_wins_over_a_snapshot_of_another_moment():
    """8713267082 Hoodwink, death at -0:10: the first laning snapshot (0:00, already dead) said 0/566."""
    player = hero("Hoodwink", 1, hp=0, max_hp=566, state_time=-15)
    player.hp_pct = 83
    ctx = death_ctx(time=-10, phase="pregame", player=player, mc={"seconds_before_death": 5})
    line = evaluator.format_player_state(ctx, "en")
    assert "HP 83%" in line and "0/566" not in line and "no data" not in line
    stale = hero("Razor", 2, hp=31, max_hp=692, mana=10, max_mana=300)
    stale.hp_pct = 36
    line = evaluator._format_hero_state(stale, "Ally", "en")
    assert "HP 36%" in line and "31/692" not in line and "Mana" not in line and "Status: Alive" in line


@pytest.mark.parametrize("player", [
    hero("Legion Commander", 1, hp=0, max_hp=978, state_time=350),      # sampled at the death tick
    hero("Legion Commander", 1, hp=406, max_hp=978),                    # no state_time: moment unknown
    hero("Legion Commander", 1, hp=406, max_hp=978, state_time=290),    # a minute old
    hero("Legion Commander", 1),                                        # no HP data at all (OpenDota)
])
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_unknown_hp_before_a_death_is_not_mentioned(player, lang):
    ev = rules(death_ctx(player=player, lang=lang), lang)
    assert "%" not in all_text(ev) and "HP" not in ev.analysis and "血量" not in ev.analysis


def test_death_prompt_shows_the_state_before_the_kill():
    player = hero("Legion Commander", 1, hp=406, max_hp=978, mana=100, max_mana=300, state_time=345)
    ctx = death_ctx(player=player, allies=[hero("Lion", 2, hp=0, max_hp=604, distance_to_player=300.0)])
    line = evaluator.format_player_state(ctx, "en")
    assert "state 5s before the death, at 05:45" in line and "HP 406/978 (42%)" in line
    assert "Status" not in line  # neither "Alive" nor "Dead" for the hero whose death is evaluated
    prompt = evaluator._build_evaluation_prompt(ctx, "", lang="en")
    assert "HP 0/" not in prompt  # a hero at 0 HP is dead, not "HP 0/604 ... Alive"
    assert "[Ally] Lion" in prompt and "Status: Dead" in prompt
    assert "Hero states below are from 05:45 (5s before the player's death)" in prompt
    assert "Item lists are reconstructed from the purchase log" in prompt  # an estimate, and said so

    dead_tick = death_ctx(player=hero("Legion Commander", 1, hp=0, max_hp=978, state_time=350))
    line = evaluator.format_player_state(dead_tick, "en")
    assert "HP" not in line.replace("HP/mana before the death: no data", "") and "Alive" not in line
    zh = evaluator.format_player_state(death_ctx(player=player, lang="zh"), "zh")
    assert "阵亡前5秒（05:45）的状态" in zh and "状态:" not in zh


# ---------------------------------------------------------------------------
# Killer names (A5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("killer, unit, en_text, zh_text", [
    ("skeleton_king", "", "was killed by Wraith King", "被Wraith King击杀"),
    ("npc_dota_hero_zuus", "", "was killed by Zeus", "被Zeus击杀"),
    ("Lina", "", "was killed by Lina", "被Lina击杀"),
    ("", "npc_dota_badguys_tower1_mid", "was killed by a tower", "被防御塔击杀"),
    ("", "npc_dota_creep_badguys_melee", "was killed by a lane creep", "被小兵击杀"),
    ("", "", "Legion Commander died", "Legion Commander阵亡"),
])
def test_killer_is_a_display_name_or_left_out(killer, unit, en_text, zh_text):
    en = rules(death_ctx(killer=killer, killer_unit=unit, phase="mid_game", time=900))
    zh = rules(death_ctx(killer=killer, killer_unit=unit, phase="mid_game", time=900, lang="zh"), "zh")
    assert en_text in en.analysis and zh_text in zh.analysis
    for raw in ("skeleton_king", "zuus", "npc_dota", "an enemy", "敌方击杀"):
        assert raw not in en.analysis and raw not in zh.analysis
    if not killer and not unit:
        assert "killed by" not in en.analysis and "击杀" not in zh.analysis.split("。")[0]


def test_deny_and_own_spell_are_not_reported_as_enemy_kills():
    def denied(killer, killer_slot, lang):
        ctx = death_ctx(killer=killer, phase="mid_game", time=900, lang=lang)
        ctx.decision_point.related_death.denied = True
        ctx.decision_point.related_death.killer_slot = killer_slot
        return rules(ctx, lang).analysis

    assert denied("Lion", 2, "en").startswith("Legion Commander was denied by ally Lion.")
    assert denied("Lion", 2, "zh").startswith("Legion Commander被队友Lion反补。")
    assert denied("Legion Commander", 1, "en").startswith("Legion Commander died to their own ability or item.")
    assert denied("Legion Commander", 1, "zh").startswith("Legion Commander死于自己的技能或物品。")


def test_overextended_death_does_not_say_far_from_the_towers_twice():
    """The analysis read: '... deep in enemy territory, far from allied towers. The player was
    away from allied towers with no ward coverage nearby.'"""
    va = vision(danger="safe", safe_zone=False, wards=False, time=900)
    deep = rules(death_ctx(phase="mid_game", time=900, va=va, overextended=True))
    assert deep.analysis.endswith("far from allied towers. No ward covered the area.")
    assert "away from allied towers" not in deep.analysis
    plain = rules(death_ctx(phase="mid_game", time=900, va=va))
    assert plain.analysis.endswith("The player was away from allied towers with no ward coverage nearby.")
    zh = rules(death_ctx(phase="mid_game", time=900, va=va, overextended=True, lang="zh"), "zh")
    assert zh.analysis.endswith("远离友方防御塔。周围也没有眼位覆盖。") and "当时不在友方防御塔附近" not in zh.analysis


def test_death_text_does_not_repeat_the_brief():
    """The card shows the brief on its own: "X was killed by Y. Killed by Y." said it twice."""
    ctx = death_ctx(phase="mid_game", time=900)
    ctx.decision_point.brief = "Killed by Lion"
    assert rules(ctx).analysis == "Legion Commander was killed by Lion."
    laning = death_ctx()  # no HP sample: the laning stage is all that is known
    assert rules(laning).analysis == "Legion Commander was killed by Lion during laning."
    assert rules(death_ctx(lang="zh"), "zh").analysis == "Legion Commander在对线期被Lion击杀。"


# ---------------------------------------------------------------------------
# One head count (A2 / AP14 / AP47)
# ---------------------------------------------------------------------------

def test_teamfight_numbers_come_from_the_context_head_count():
    near = {"headcount_basis": "nearby", "headcount_allies": 2, "headcount_enemies": 2,
            "nearby_allies": 2, "nearby_enemies": 2}
    ev = rules(death_ctx(is_tf=True, phase="mid_game", time=1156, mc=near))
    assert "2 allies were fighting nearby" in ev.analysis and ev.score == "B"

    outnumbered = {"headcount_basis": "nearby", "headcount_allies": 0, "headcount_enemies": 3}
    ev = rules(death_ctx(is_tf=True, phase="mid_game", time=1156, mc=outnumbered))
    assert "outnumbered (1v3)" in ev.analysis and ev.score == "C"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_fight_without_positions_counts_participants(lang):
    """OpenDota fights: nobody has a position, the fight record says who took part."""
    mc = {"headcount_basis": "participation", "headcount_allies": 2, "headcount_enemies": 5,
          "nearby_allies": 0, "nearby_enemies": 1, "fight_allies": 2, "fight_enemies": 5}
    ctx = death_ctx(is_tf=True, phase="mid_game", time=1156, mc=mc, lang=lang,
                    player=hero("Legion Commander", 1, x=0.0, y=0.0))
    ev = rules(ctx, lang)
    if lang == "en":
        assert "outnumbered (3v5 by the heroes involved in the fight)" in ev.analysis
        assert "nearby" not in ev.analysis and "1v1" not in ev.analysis
    else:
        assert "按参战英雄计3v5" in ev.analysis and "附近" not in ev.analysis
    situation = evaluator.format_match_situation(ctx, lang)
    assert ("your team 3 (player included), enemy team 5" in situation) if lang == "en" else ("己方3名（含玩家），敌方5名" in situation)
    assert ("not a head count" in situation) if lang == "en" else ("这不是双方人数" in situation)

    even = dict(mc, headcount_allies=3, headcount_enemies=3)
    ev = rules(death_ctx(is_tf=True, phase="mid_game", time=1156, mc=even, lang=lang), lang)
    assert ("3 allies and 3 enemy heroes took part in the fight" in ev.analysis) if lang == "en" else (
        "有3名队友和3名敌方英雄参与了这场团战" in ev.analysis)
    assert ev.score == "B" and ev.mistake == ""    # a fight with the team, even by participants


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_one_participant_more_is_not_outnumbered(lang):
    """8762775364 LC 13:40 via OpenDota: "outnumbered (4v5 by the heroes involved)" while the
    replay shows two allies fighting next to her. A fight record covers the whole map."""
    mc = {"headcount_basis": "participation", "headcount_allies": 3, "headcount_enemies": 5,
          "fight_allies": 3, "fight_enemies": 5}
    ev = rules(death_ctx(is_tf=True, phase="mid_game", time=820, mc=mc, lang=lang), lang)
    assert "outnumbered" not in all_text(ev) and "人数劣势" not in all_text(ev)
    assert ("3 allies and 5 enemy heroes took part" in ev.analysis) if lang == "en" else (
        "有3名队友和5名敌方英雄参与了这场团战" in ev.analysis)
    assert ev.mistake == "" and ev.better_play == "" and ev.score == "C"
    # The chain's "outnumbered" root cause needs the same margin of two.
    chain = ("teamfight_outnumbered", "More enemy than allied heroes took part in the fight")
    ev = rules(death_ctx(is_tf=True, phase="mid_game", time=820, mc=mc, lang=lang, chain=chain), lang)
    assert "Root cause" not in ev.analysis and "根因" not in ev.analysis


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("point_type", ["death", "fight_engage"])
def test_unknown_head_count_makes_no_numbers_claim(lang, point_type):
    """The old text said "(1v0)" or "5v5 nearby" for fights nobody had a position in."""
    ctx = death_ctx(is_tf=True, phase="mid_game", time=1156, lang=lang,
                    mc={"headcount_basis": "unknown", "headcount_allies": 0, "headcount_enemies": 0,
                        "nearby_allies": 0, "nearby_enemies": 0})
    if point_type == "fight_engage":
        ctx.decision_point = DecisionPoint(time=1156, point_type="fight_engage", player_slot=1,
                                           hero_name="Legion Commander", severity="major",
                                           brief="Died early in the fight" if lang == "en" else "团战中过早阵亡")
    ev = rules(ctx, lang)
    text = all_text(ev)
    assert not HEAD_COUNT_RE.search(text)
    for claim in ("outnumbered", "even or better", "few allies", "人数劣势", "人数不吃亏", "队友较少"):
        assert claim not in ev.analysis and claim not in ev.mistake


def test_context_without_a_head_count_uses_the_nearby_lists():
    ctx = death_ctx(is_tf=True, phase="mid_game", time=1156,
                    allies=[hero("Lion", 2), hero("Zeus", 3)], enemies=[hero("Lina", 129)])
    assert "2 allies were fighting nearby" in rules(ctx).analysis
    blind = death_ctx(is_tf=True, phase="mid_game", time=1156, player=hero("Legion Commander", 1, x=0.0, y=0.0))
    assert not HEAD_COUNT_RE.search(all_text(rules(blind)))


def test_heroes_without_a_position_block_claims_that_need_them():
    """A solo death "1v3" is only "outnumbered" when no living ally is unaccounted for."""
    mc = {"headcount_basis": "nearby", "headcount_allies": 0, "headcount_enemies": 3}
    alone = rules(death_ctx(phase="mid_game", time=900, mc=mc))
    assert "outnumbered (1v3)" in alone.analysis and alone.mistake.endswith("with no ally nearby")
    with_ally = rules(death_ctx(phase="mid_game", time=900, mc=dict(mc, headcount_allies=1)))
    assert "outnumbered (2v3)" in with_ally.analysis and "no ally" not in with_ally.mistake
    unsure = rules(death_ctx(phase="mid_game", time=900, mc=dict(mc, unknown_position_allies=2)))
    assert "outnumbered" not in all_text(unsure) and not HEAD_COUNT_RE.search(all_text(unsure))
    # A fight elsewhere says nothing about a death that was not part of it.
    elsewhere = {"headcount_basis": "participation", "headcount_allies": 0, "headcount_enemies": 4}
    assert "outnumbered" not in all_text(rules(death_ctx(phase="mid_game", time=900, mc=elsewhere)))


def test_root_cause_must_agree_with_the_head_count():
    """8762775364 LC 19:16: "2 allies were fighting nearby" + mistake "Fought while outnumbered"."""
    chain = ("teamfight_outnumbered", "Took a teamfight while outnumbered; wait for teammates or avoid the fight")
    not_outnumbered = {"headcount_basis": "nearby", "headcount_allies": 2, "headcount_enemies": 2}
    ev = rules(death_ctx(is_tf=True, phase="mid_game", time=1156, mc=not_outnumbered, chain=chain))
    assert "2 allies were fighting nearby" in ev.analysis
    assert "outnumbered" not in all_text(ev) and "Root cause" not in ev.analysis

    outnumbered = {"headcount_basis": "nearby", "headcount_allies": 0, "headcount_enemies": 3}
    ev = rules(death_ctx(is_tf=True, phase="mid_game", time=1156, mc=outnumbered, chain=chain))
    assert ev.mistake.startswith("Fought while outnumbered: Took a teamfight") and "Root cause" in ev.analysis
    assert "fight.." not in ev.analysis
    # A summary that already opens with the label is not prefixed with it again.
    same = ("teamfight_outnumbered", "Fought while outnumbered; wait for teammates or avoid the fight")
    ev = rules(death_ctx(is_tf=True, phase="mid_game", time=1156, mc=outnumbered, chain=same))
    assert ev.mistake == same[1]

    unknown = {"headcount_basis": "unknown"}  # nothing to contradict: the chain is the only source
    ev = rules(death_ctx(is_tf=True, phase="mid_game", time=1156, mc=unknown, chain=chain))
    assert ev.mistake.startswith("Fought while outnumbered")

    other = rules(death_ctx(phase="mid_game", time=900, mc=not_outnumbered,
                            chain=("overextend", "Ignored rising danger and kept pushing")))
    assert other.mistake.startswith("Overextended")


# 8762775364 Legion Commander 22:20: the enemy opens with Ravage, three allies die at 22:07-22:13,
# she kills Zeus and dies last. Counted at her last second alive that is "1v4".
_LAST_ONE_STANDING = {"headcount_basis": "nearby", "headcount_allies": 0, "headcount_enemies": 4,
                      "fallen_allies": 3, "fallen_enemies": 1, "opening_time": 1327,
                      "opening_allies": 3, "opening_enemies": 4, "opening_outnumbered": False}


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("is_tf, phase, time", [(True, "mid_game", 1340), (False, "mid_game", 1340),
                                                 (False, "laning", 349), (True, "pregame", 5)])
def test_last_survivor_of_an_even_fight_is_not_outnumbered(lang, is_tf, phase, time):
    """17 of 47 "outnumbered (XvY)" analyses and 10 of 15 "Fought while outnumbered; wait for
    teammates" root causes on the two replays described fights that were even before the first
    allied death."""
    chain = ("teamfight_outnumbered", "Fought while outnumbered; wait for teammates or avoid the fight")
    ev = rules(death_ctx(is_tf=is_tf, phase=phase, time=time, mc=_LAST_ONE_STANDING, lang=lang, chain=chain), lang)
    text = all_text(ev)
    assert ev.mistake == "" and ev.better_play == ""
    for claim in ("outnumbered", "wait for teammates", "larger group", "no ally", "Stayed", "人数劣势",
                  "没有队友", "抓到", "Root cause", "根因"):
        assert claim not in text, text
    if lang == "en":
        assert "after 3 allies had already died in" in ev.analysis and "(1v4 by then)" in ev.analysis
    else:
        assert "此前已有3名队友在" in ev.analysis and "（此时为1v4）" in ev.analysis
    assert ev.score == ("C" if phase == "pregame" else "B")      # she fought with the team around her


def test_survivor_text_states_each_fact_once():
    en = rules(death_ctx(is_tf=True, phase="mid_game", time=1340, mc=_LAST_ONE_STANDING))
    assert en.analysis == ("Legion Commander was killed by Lion in a teamfight after 3 allies had already "
                           "died in it (1v4 by then).")
    one = dict(_LAST_ONE_STANDING, fallen_allies=1, headcount_enemies=2, opening_allies=1, opening_enemies=2,
               pre_death_hp_pct=53, seconds_before_death=4)
    laning = rules(death_ctx(phase="laning", time=349, mc=one, player=hero("Warlock", 2, state_time=345)))
    assert laning.analysis == ("Legion Commander was killed by Lion during laning after 1 ally had already "
                               "died in the same fight (1v2 by then). HP was 53% 4s before the death.")
    assert laning.score == "C"                                   # a 2v2 lane, not a fight with the team
    zh = rules(death_ctx(is_tf=True, phase="mid_game", time=1340, mc=_LAST_ONE_STANDING, lang="zh"), "zh")
    assert zh.analysis == "Legion Commander在团战中被Lion击杀，此前已有3名队友在这场团战中阵亡（此时为1v4）。"
    # Still with the team and not outnumbered: the ordinary sentence is right as it is.
    grouped = dict(_LAST_ONE_STANDING, headcount_allies=3, headcount_enemies=3, fallen_allies=1)
    assert "3 allies were fighting nearby" in rules(
        death_ctx(is_tf=True, phase="mid_game", time=1340, mc=grouped)).analysis


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_fight_that_started_outnumbered_says_so_with_the_opening_numbers(lang):
    """8762775364 LC 20:52: 4v5 when the first hero died, one ally fell, she died at 1v3."""
    mc = dict(_LAST_ONE_STANDING, headcount_enemies=3, fallen_allies=1, opening_enemies=5,
              opening_outnumbered=True)
    chain = ("teamfight_outnumbered", "Fought while outnumbered; wait for teammates or avoid the fight"
             if lang == "en" else "在人数劣势下作战，应等待队友集合或避免交战")
    ev = rules(death_ctx(is_tf=True, phase="mid_game", time=1252, mc=mc, lang=lang, chain=chain), lang)
    if lang == "en":
        assert ("in a teamfight while outnumbered (4v5 when the first hero died, 1 ally died before the player)"
                in ev.analysis)
        assert "1v3" not in ev.analysis and ev.mistake.startswith("Fought while outnumbered")
    else:
        assert "人数劣势（首个英雄阵亡时4v5，此后1名队友先于玩家阵亡）的团战中" in ev.analysis
        assert "1v3" not in ev.analysis and ev.mistake == "人数劣势下接团：在人数劣势下作战，应等待队友集合或避免交战"
    solo = rules(death_ctx(phase="mid_game", time=1045, mc=mc, lang=lang), lang)
    assert "no ally nearby" not in solo.mistake and "附近没有队友" not in solo.mistake   # her ally had just died there
    # The numbers had turned by the time she died: no "outnumbered", whatever the opening was.
    turned = dict(mc, headcount_allies=3, headcount_enemies=3)
    ev = rules(death_ctx(is_tf=True, phase="mid_game", time=1156, mc=turned, lang=lang, chain=chain), lang)
    assert "outnumbered" not in all_text(ev) and "人数劣势" not in all_text(ev)


def test_pregame_death_states_numbers_only_when_known():
    mc = {"headcount_basis": "nearby", "headcount_allies": 0, "headcount_enemies": 3}
    known = rules(death_ctx(time=-10, phase="pregame", mc=mc))
    assert "outnumbered (1v3 nearby)" in known.analysis and known.score == "C"
    # Even numbers: stated, but why the fight was lost is not in the data (8713267082 Warlock
    # 0:05 got a kill in that fight six seconds earlier and was told he "died carelessly").
    even = rules(death_ctx(time=-10, phase="pregame", mc=dict(mc, headcount_allies=2)))
    assert "not unfavorable (3v3 nearby)" in even.analysis and even.score == "C"
    assert even.mistake == "" and "careless" not in all_text(even) and "free death" not in all_text(even).lower()
    # Who took part in a fight record says too little for "died despite even numbers".
    record = {"headcount_basis": "participation", "headcount_allies": 2, "headcount_enemies": 3}
    by_record = rules(death_ctx(time=-10, phase="pregame", mc=record))
    assert by_record.score == "C" and "not unfavorable" not in by_record.analysis
    for lang in ("en", "zh"):
        blind = rules(death_ctx(time=-10, phase="pregame", lang=lang, mc={"headcount_basis": "unknown"}), lang)
        assert not HEAD_COUNT_RE.search(all_text(blind)) and blind.score == "C"
        assert "outnumbered" not in blind.analysis and "人数" not in blind.analysis


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_pregame_text_asserts_no_rune_contest(lang):
    """The data has no rune information: "while contesting the bounty rune outnumbered" and the
    mistake "Contested the bounty rune despite being outnumbered" were made up."""
    near = {"headcount_basis": "nearby", "headcount_allies": 0, "headcount_enemies": 3}
    for mc in (near, dict(near, headcount_allies=2), {"headcount_basis": "unknown"},
               dict(_LAST_ONE_STANDING, headcount_enemies=5, fallen_allies=1)):
        ev = rules(death_ctx(time=-10, phase="pregame", mc=mc, lang=lang), lang)
        assert "rune" not in all_text(ev).lower() and "符" not in all_text(ev), all_text(ev)
    out = rules(death_ctx(time=-10, phase="pregame", mc=near, lang=lang), lang)
    assert ("in a level-1 fight while outnumbered (1v3 nearby)" in out.analysis) if lang == "en" else (
        "在人数劣势（附近1v3）的1级战斗中" in out.analysis)
    assert out.mistake == ("Fought at level 1 while outnumbered" if lang == "en" else "1级时在人数劣势下接战")
    # Razor 0:00 in 8713267082: "3v5" only after Hoodwink had died in the same fight.
    razor = rules(death_ctx(time=0, phase="pregame", lang=lang,
                            mc=dict(_LAST_ONE_STANDING, headcount_allies=2, headcount_enemies=5, fallen_allies=1)), lang)
    assert razor.mistake == "" and "outnumbered" not in razor.analysis and "人数劣势" not in razor.analysis
    assert ("after 1 ally had already died in it (3v5 by then)" in razor.analysis) if lang == "en" else (
        "此前已有1名队友在这场战斗中阵亡（此时为3v5）" in razor.analysis)


# ---------------------------------------------------------------------------
# Only what the data shows is called a mistake
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang", ["en", "zh"])
def test_death_without_evidence_names_no_mistake(lang):
    """A death nothing is known about got the "mistake" "Dying costs gold and time ..."."""
    for phase, time in (("mid_game", 900), ("laning", 350)):
        ev = rules(death_ctx(phase=phase, time=time, lang=lang, player=hero("Legion Commander", 1),
                             mc={"headcount_basis": "unknown"}), lang)
        assert ev.mistake == "" and ev.better_play == "" and ev.lesson and ev.score == "C"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_teamfight_death_with_few_allies_states_the_count_only(lang):
    """8762775364 Hoodwink 12:04 was the first death of the fight and was told she "arrived too late"."""
    mc = {"headcount_basis": "nearby", "headcount_allies": 1, "headcount_enemies": 2}
    ev = rules(death_ctx(is_tf=True, phase="mid_game", time=724, mc=mc, lang=lang), lang)
    text = all_text(ev)
    assert "2v2" in ev.analysis and ev.mistake == ""
    for guess in ("too late", "cut off", "被分割", "到晚了", "may have"):
        assert guess not in text
    grouped = dict(mc, headcount_allies=3, headcount_enemies=4)
    assert rules(death_ctx(is_tf=True, phase="mid_game", time=724, mc=grouped, lang=lang), lang).mistake == ""


def test_laning_death_against_the_lane_opponents_is_not_a_gank():
    """8713267082 Warlock 5:49: alone against Viper and Lion in lane, told to "check the minimap"."""
    mc = {"headcount_basis": "nearby", "headcount_allies": 0, "headcount_enemies": 2,
          "pre_death_hp_pct": 53, "seconds_before_death": 4}
    player = hero("Warlock", 2, state_time=345)
    en = rules(death_ctx(phase="laning", time=349, mc=mc, player=player))
    assert "during laning, 1v2 nearby" in en.analysis and "HP was 53% 4s before the death" in en.analysis
    assert en.mistake == "Stayed within reach of 2 enemy heroes with no ally nearby" and en.score == "C"
    assert "minimap" not in all_text(en) and "larger group" not in all_text(en)
    zh = rules(death_ctx(phase="laning", time=349, mc=mc, player=player, lang="zh"), "zh")
    assert "当时附近是1v2" in zh.analysis and "阵亡前4秒血量为53%" in zh.analysis and "小地图" not in all_text(zh)
    # After the laning stage a lone hero against three was caught.
    later = rules(death_ctx(phase="mid_game", time=900, mc=dict(mc, headcount_enemies=3)))
    assert later.mistake.endswith("with no ally nearby") and "minimap" in later.better_play


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("is_tf", [True, False])
def test_dying_in_the_own_base_is_a_defence(lang, is_tf):
    """8762775364 LC 32:53, 4v5 on her own high ground: "wait for teammates or avoid the fight"."""
    from backend.app import geo
    x, y = geo.ANCIENT_POSITIONS["radiant"]
    mc = {"headcount_basis": "nearby", "headcount_allies": 3, "headcount_enemies": 5}
    ev = rules(death_ctx(is_tf=is_tf, phase="late_game", time=1973, mc=mc, lang=lang,
                         player=hero("Legion Commander", 1, x=x + 300, y=y)), lang)
    assert ("defending the base outnumbered (4v5)" in ev.analysis) if lang == "en" else (
        "人数劣势（4v5）的基地防守" in ev.analysis)
    assert ev.mistake == "" and ev.better_play == ""
    # The same numbers anywhere else on the map are a fight that could have been declined.
    away = rules(death_ctx(is_tf=is_tf, phase="late_game", time=1973, mc=mc, lang=lang), lang)
    assert away.mistake and "base" not in away.analysis and "基地" not in away.analysis


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_missed_cs_at_ten_minutes_is_a_laning_point(lang):
    """The point at 10:00 sums up the laning stage; it read "during the mid game phase"."""
    dp = DecisionPoint(time=600, point_type="missed_cs", player_slot=1, hero_name="Drow Ranger",
                       severity="minor", brief="Minute 10: 32 last hits" if lang == "en" else "第10分钟补刀32个")
    ctx = DecisionContext(decision_point=dp, player_state=hero("Drow Ranger", 1), game_phase="mid_game")
    ev = rules(ctx, lang)
    assert ("during the laning phase" in ev.analysis) if lang == "en" else ("在对线期" in ev.analysis)


def test_player_state_says_when_components_are_still_listed():
    ctx = death_ctx(mc={"seconds_before_death": 3, "inventory_exact": False})
    assert "components of items built later may still be listed" in evaluator.format_player_state(ctx, "en")
    assert "散件可能仍被列出" in evaluator.format_player_state(ctx, "zh")
    exact = death_ctx(mc={"seconds_before_death": 3, "inventory_exact": True})
    assert "components" not in evaluator.format_player_state(exact, "en")


# ---------------------------------------------------------------------------
# Vision: one number per moment, fog vs. no data (A4 / AP47)
# ---------------------------------------------------------------------------

def test_missing_counts_are_dated_and_not_contradictory():
    """Old text: "2 enemy heroes were missing. 5 enemy heroes were missing for over 10s at the same time."."""
    va = vision(missing=("Lion", "Void Spirit"), danger="danger", time=1725,
                critical=(1685, "5 enemy heroes were missing for over 10s at the same time"))
    ev = rules(death_ctx(phase="late_game", time=1725, va=va))
    assert ("At that moment 2 enemy heroes were out of sight within 5000 units of the player "
            "(Lion, Void Spirit).") in ev.analysis
    assert "Earlier, at 28:05: 5 enemy heroes were missing for over 10s at the same time." in ev.analysis
    zh = rules(death_ctx(phase="late_game", time=1725, lang="zh",
                         va=vision(missing=("Lion", "Void Spirit"), danger="danger", time=1725,
                                   critical=(1685, "5名敌方英雄同时消失超过10秒，形势危险"))), "zh")
    assert "当时有2名敌方英雄藏在玩家5000码范围内的迷雾中（Lion、Void Spirit）。" in zh.analysis
    assert "此前在28:05：" in zh.analysis


def test_single_missing_enemy_is_singular():
    for danger in ("caution", "danger", "critical"):
        ev = rules(death_ctx(phase="mid_game", time=900, va=vision(missing=("Lion",), danger=danger, longest=45)))
        assert "1 enemy hero was out of sight within 5000 units of the player (Lion), for 45s." in ev.analysis
        assert "1 enemy heroes" not in ev.analysis and "the longest" not in ev.analysis


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_enemies_hidden_across_the_map_are_not_part_of_a_death(lang):
    """8713267082 Hoodwink 16:25 named Dragon Knight, 7,759 units away, as "out of sight" next to
    the root cause "ignored missing-enemy signals"; 13 of 21 such remarks counted far enemies."""
    va = vision(missing=("Lion",), far=("Dragon Knight", "Viper"), danger="caution", longest=12,
                far_longest=90, time=985)
    ctx = death_ctx(phase="mid_game", time=985, va=va, lang=lang)
    ev = rules(ctx, lang)
    assert "Dragon Knight" not in ev.analysis and "Viper" not in ev.analysis and "90" not in ev.analysis
    assert ("1 enemy hero was out of sight within 5000 units of the player (Lion)." in ev.analysis
            if lang == "en" else "当时有1名敌方英雄藏在玩家5000码范围内的迷雾中（Lion）。" in ev.analysis)
    situation = evaluator.format_match_situation(ctx, lang)
    if lang == "en":
        assert "Enemy heroes hidden in the fog within 5000 units of the player: 1 (Lion)" in situation
        assert "Enemy heroes out of sight elsewhere on the map (more than 5000 units from the player): 2" in situation
    else:
        assert "藏在玩家5000码范围内迷雾中的敌方英雄: 1名（Lion）" in situation
        assert "地图其他位置（距玩家超过5000码）不在视野内的敌方英雄: 2名" in situation
    # Only far-away enemies in the fog: nothing is said about the death, whatever the danger level.
    only_far = vision(far=("Dragon Knight",), danger="caution", far_longest=90, time=985)
    quiet = rules(death_ctx(phase="mid_game", time=985, va=only_far, lang=lang), lang)
    assert "out of sight" not in quiet.analysis and "迷雾" not in quiet.analysis


def test_critical_danger_dates_the_moment_to_back_off():
    va = vision(missing=("Lion", "Zeus", "Viper"), danger="critical", safe_zone=False,
                reasons=["3 enemy heroes are missing in the fog"], time=900,
                critical=(860, "Observer ward at Dire jungle expired; vision lost"))
    ev = rules(death_ctx(phase="mid_game", time=900, va=va))
    assert ev.score == "D" and "Should have backed off at 14:20" in ev.better_play


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_enemies_without_position_data_are_unknown_not_missing(lang):
    """OpenDota matches: no positions outside fights. Nothing may be called missing."""
    va = vision(unknown=("Lion", "Zeus", "Viper", "Axe", "Lina"), danger="safe", time=1500)
    ctx = death_ctx(phase="late_game", time=1500, va=va, lang=lang,
                    mc={"enemies_missing": 0, "enemies_unknown": 5, "nearby_allies": 0, "nearby_enemies": 0,
                        "headcount_basis": "unknown"})
    ev = rules(ctx, lang)
    if lang == "en":
        assert "no position for 5 enemy heroes" in ev.analysis
        assert "missing" not in ev.analysis and "out of sight" not in ev.analysis
    else:
        assert "5名敌方英雄当时没有位置数据" in ev.analysis and "消失" not in ev.analysis
    situation = evaluator.format_match_situation(ctx, lang)
    expected = ("Enemy heroes with no position data (unknown, not counted as missing): 5 (Lion, Zeus, Viper and 2 more)"
                if lang == "en" else "没有位置数据的敌方英雄（未知，不算消失）: 5名（Lion、Zeus、Viper等5名）")
    assert expected in situation
    # "0 hidden within 5000 units" would be a claim about heroes nothing is known about.
    assert "hidden in the fog within 5000 units" not in situation and "迷雾中的敌方英雄" not in situation
    known = death_ctx(phase="late_game", time=1500, va=vision(danger="safe", time=1500), lang=lang)
    assert ("hidden in the fog within 5000 units of the player: 0" if lang == "en"
            else "藏在玩家5000码范围内迷雾中的敌方英雄: 0名") in evaluator.format_match_situation(known, lang)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_overextension_text_claims_only_what_the_context_shows(lang):
    """The rule always said "advanced too deep without vision or team support ... place wards before
    pushing"; in 6 of 6 real points an ally stood next to the player."""
    brief = "Overextended during laning with low HP (32%)" if lang == "en" else "对线期位置过深且血量偏低（32%）"
    dp = DecisionPoint(time=360, point_type="overextend", player_slot=1, hero_name="Weaver",
                       severity="minor", brief=brief)

    def verdict(allies, **va_kw):
        ctx = DecisionContext(decision_point=dp, player_state=hero("Weaver", 1), game_phase="laning",
                              match_context={"team": "radiant", "nearby_allies": allies},
                              vision_analysis=vision(time=360, **va_kw))
        return rules(ctx, lang)

    with_ally = verdict(1, wards=False, safe_zone=False)            # the six real cases
    assert with_ally.mistake == "" and with_ally.score == "B"
    for claim in ("vision", "wards", "ward cover", "support", "pushed too far", "视野", "队友", "眼", "深入"):
        assert claim not in with_ally.analysis and claim not in with_ally.mistake, with_ally.analysis
    alone = verdict(0, wards=False, safe_zone=False)
    assert alone.score == "C" and alone.mistake == (
        "Stayed forward at low HP with no ally nearby, no ward covering the area" if lang == "en"
        else "残血时仍停留在靠前的位置，而附近没有队友、所在区域没有眼位")
    warded = verdict(0)
    assert warded.mistake == ("Stayed forward at low HP with no ally nearby" if lang == "en"
                              else "残血时仍停留在靠前的位置，而附近没有队友")
    hidden = verdict(1, wards=False, safe_zone=False, missing=("Lion", "Zeus", "Viper"), far=("Axe",))
    assert hidden.score == "D" and ("3 enemy heroes out of sight within reach" in hidden.mistake
                                    if lang == "en" else "有3名敌方英雄藏在附近的迷雾中" in hidden.mistake)
    far_only = verdict(1, wards=False, safe_zone=False, far=("Axe", "Lion", "Zeus"))
    assert far_only.mistake == "" and far_only.score == "B"       # hidden on the other side of the map


def test_overextension_is_not_a_calculated_risk_when_enemies_are_unknown():
    dp = DecisionPoint(time=900, point_type="overextend", player_slot=1, hero_name="Legion Commander",
                       severity="minor", brief="Deep in the Dire jungle at low HP")
    seen = DecisionContext(decision_point=dp, player_state=hero("Legion Commander", 1), game_phase="mid_game",
                           vision_analysis=vision(time=900))
    blind = DecisionContext(decision_point=dp, player_state=hero("Legion Commander", 1), game_phase="mid_game",
                            vision_analysis=vision(unknown=("Lion", "Zeus"), time=900))
    assert rules(seen).score == "B" and rules(blind).score == "C"


def test_good_play_grade_follows_the_subtype_in_both_languages():
    def grade(brief, subtype, lang):
        dp = DecisionPoint(time=900, point_type="good_play", player_slot=1, hero_name="Legion Commander",
                           severity="positive", brief=brief, subtype=subtype)
        ctx = DecisionContext(decision_point=dp, player_state=hero("Legion Commander", 1), game_phase="mid_game")
        return rules(ctx, lang).score

    assert grade("Achieved a Double Kill", "multi_kill", "en") == grade("完成双杀", "multi_kill", "zh") == "S"
    assert grade("Achieved a Killing Spree streak", "kill_streak", "en") == "A"
    assert grade("达成连杀", "kill_streak", "zh") == "A"  # was S in zh only: "连杀" matched a word list
    assert grade("Achieved a Double Kill", "", "en") == "S"  # points without a subtype: by the brief
    # A run of kills is praised as such, not as "an excellent decision".
    for lang, never in (("en", "decision"), ("zh", "决策")):
        dp = DecisionPoint(time=900, point_type="good_play", player_slot=1, hero_name="Legion Commander",
                           severity="positive", brief="x", subtype="kill_streak")
        ctx = DecisionContext(decision_point=dp, player_state=hero("Legion Commander", 1), game_phase="mid_game")
        assert never not in all_text(rules(ctx, lang)).lower()


# ---------------------------------------------------------------------------
# Language
# ---------------------------------------------------------------------------

def _rule_contexts(lang):
    chain = ("ignore_missing", "Multiple enemies disappeared" if lang == "en" else "多名敌人消失后没有后撤")
    low = hero("Legion Commander", 1, hp=180, max_hp=978, state_time=347)
    fine = hero("Legion Commander", 1, hp=700, max_hp=978, state_time=345)
    near = {"headcount_basis": "nearby", "headcount_allies": 0, "headcount_enemies": 3}
    yield death_ctx(player=low, lang=lang)
    yield death_ctx(player=fine, lang=lang, chain=chain, overextended=True)
    yield death_ctx(killer="", lang=lang, phase="mid_game", time=900)
    yield death_ctx(lang=lang, phase="mid_game", time=900, mc=near,
                    enemies=[hero("Lion", 129, hp=600, max_hp=604), hero("Zeus", 130, hp=600, max_hp=604)])
    for mc in (near, dict(near, headcount_allies=3), dict(near, headcount_allies=1, headcount_enemies=2),
               {"headcount_basis": "participation", "headcount_allies": 1, "headcount_enemies": 4},
               {"headcount_basis": "participation", "headcount_allies": 1, "headcount_enemies": 2},
               {"headcount_basis": "unknown"}):
        yield death_ctx(is_tf=True, lang=lang, phase="mid_game", time=1156, mc=mc)
        yield death_ctx(lang=lang, phase="pregame", time=5, mc=mc)
    reason = "3 enemy heroes are missing in the fog" if lang == "en" else "3名敌方英雄消失在迷雾中"
    crit = "Observer ward at Dire jungle expired; vision lost" if lang == "en" else "夜魇野区的观察守卫到期，视野丢失"
    for danger in ("safe", "caution", "danger", "critical"):
        yield death_ctx(lang=lang, phase="mid_game", time=900,
                        va=vision(missing=("Lion", "Zeus"), unknown=("Viper",), danger=danger, longest=40,
                                  reasons=[reason], critical=(860, crit), safe_zone=False, wards=False, time=900))


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_rule_text_follows_the_report_language(lang):
    seen = 0
    for ctx in _rule_contexts(lang):
        points = [ctx.decision_point]
        for point_type, severity in (("fight_engage", "major"), ("fight_engage", "positive"),
                                     ("bad_trade", "major"), ("missed_cs", "minor"), ("good_play", "positive")):
            points.append(DecisionPoint(time=ctx.decision_point.time, point_type=point_type, player_slot=1,
                                        hero_name="Legion Commander", severity=severity,
                                        brief=ctx.decision_point.brief))
        for dp in points:
            ctx.decision_point = dp
            ev = rules(ctx, lang)
            seen += 1
            assert ev.score in "SABCD" and ev.analysis.strip()
            for text in (ev.analysis, ev.mistake, ev.better_play, ev.lesson):
                assert "  " not in text and ".." not in text and "。。" not in text, text
                assert text == text.strip()
                if lang == "en":
                    assert not CJK_RE.search(text), text
                else:
                    assert not ENGLISH_RUN_RE.search(text), text
    assert seen > 100


# ---------------------------------------------------------------------------
# Prompts built from the corrected context
# ---------------------------------------------------------------------------

def test_prompt_translates_codes_and_marks_estimates():
    ctx = death_ctx(va=vision(), chain=("overextend", "Ignored rising danger"))
    for lang, type_line, severity_line in (("en", "- Type: Death", "- Severity: critical"),
                                           ("zh", "- 类型: 死亡", "- 严重程度: 严重")):
        prompt = evaluator._build_evaluation_prompt(ctx, "", lang=lang)
        assert type_line in prompt and severity_line in prompt
        note = evaluator.pipeline_note(lang)
        assert prompt.count(note) == 2  # above the vision block and above the causal chain
        assert "## Vision Analysis" not in prompt and "## Death Causal Chain\n## Death" not in prompt
        assert evaluator._t(lang)["writing_rules"] in prompt
    en = evaluator._t("en")["writing_rules"]
    assert "never received them as warnings" in en and "NOT 'missing'" in en
    assert "1号位…5号位" in evaluator._t("zh")["writing_rules"]


def test_prompt_says_what_the_data_cannot_show():
    """Live verdicts stated as facts what the prompt only seemed to show: "8,189 gold unspent"
    (the figure is gold earned), "no regeneration items" (consumables are not listed),
    "Reincarnation was ready" (there are no cooldowns) and "2v2" for two allies plus the player."""
    ctx = death_ctx(player=hero("Legion Commander", 1, hp=406, max_hp=978, state_time=345, net_worth=3349),
                    mc={"nearby_allies": 2, "nearby_enemies": 2})
    en = evaluator.format_player_state(ctx, "en")
    assert "Net worth 3349" in en and "Gold 3349" not in en
    assert "Net worth is the total gold earned so far, not unspent gold" in en
    assert "Consumables (tangoes, salves, clarities, town portal scrolls, ...) are left out" in en
    assert "cooldowns are not in the data" in en and "mana is unknown unless it is listed" in en
    zh = evaluator.format_player_state(death_ctx(player=ctx.player_state, lang="zh"), "zh")
    assert "总经济 3349" in zh and "不是身上没花的现金" in zh and "不含消耗品" in zh and "是否在冷却也不在数据中" in zh
    assert not CJK_RE.search(en)
    assert "Alive nearby: 2 allied (not counting the player) and 2 enemy hero(es)" in (
        evaluator.format_match_situation(ctx, "en"))
    assert "附近存活英雄: 队友2名（不含玩家本人），敌人2名" in evaluator.format_match_situation(ctx, "zh")
    for lang, cooldowns, names in (("en", "no ability or item cooldowns", "exactly as"),
                                   ("zh", "没有技能和物品的冷却状态", "一律照抄")):
        rules_text = evaluator._t(lang)["writing_rules"]
        assert cooldowns in rules_text and names in rules_text
    assert "不要自行翻译" in evaluator._t("zh")["writing_rules"]   # "风行者" for Hoodwink in a live verdict


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_match_situation_states_the_aftermath_of_a_death(lang):
    after = {"seconds": 30, "enemy_deaths": ["Lion", "Zeus"], "allied_deaths": ["Weaver"],
             "towers_taken": [], "towers_lost": ["top_t1"]}
    text = evaluator.format_match_situation(death_ctx(mc={"aftermath": after}, lang=lang), lang)
    nothing = evaluator.format_match_situation(
        death_ctx(mc={"aftermath": dict(after, enemy_deaths=[], allied_deaths=[], towers_lost=[])}, lang=lang), lang)
    if lang == "en":
        assert ("In the 30s after this death: your team killed Lion, Zeus; your team also lost Weaver; "
                "your team lost the top T1 tower") in text
        assert "In the 30s after this death: no other hero kill and no tower loss is recorded" in nothing
        assert not CJK_RE.search(text + nothing)
    else:
        assert "阵亡后30秒内: 己方击杀了Lion、Zeus；己方又有Weaver阵亡；己方失去了上路一塔" in text
        assert "阵亡后30秒内: 没有其他击杀记录，也没有防御塔被摧毁" in nothing
        assert not ENGLISH_RUN_RE.search(text + nothing)
    assert "30" not in evaluator.format_match_situation(death_ctx(lang=lang), lang)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_states_the_role_and_what_a_positionless_source_cannot_show(lang):
    """Live report from OpenDota data: a mid Axe was coached as "the position 3 offlaner" (the
    knowledge block only knows the hero's usual role), "with no ally anywhere near" (the allies
    had no position data) who "blinked in" (no cast was recorded)."""
    unknown_ally = {"hero_name": "Lion", "level": 5, "is_alive": True}
    ctx = death_ctx(lang=lang, map_state={"unknown_position_allies": [unknown_ally]},
                    mc={"player_position": 2, "lane_role": 2, "has_positions": False, "has_cast_log": False,
                        "nearby_allies": 0, "nearby_enemies": 1, "headcount_basis": "unknown"})
    ctx.vision_analysis = vision(unknown=("Lion", "Zeus", "Viper", "Axe"), time=350)
    prompt = evaluator._build_evaluation_prompt(ctx, "", lang=lang)
    situation = evaluator.format_match_situation(ctx, lang)
    # Nothing is itemized that would read like a head count or like enemies missing.
    assert "Lion" not in situation and "0" not in situation
    partial = evaluator.format_match_situation(
        death_ctx(lang=lang, mc={"has_positions": True, "nearby_allies": 0, "nearby_enemies": 1,
                                 "headcount_basis": "unknown"}), lang)
    if lang == "en":
        assert "- Role in this match: Position 2 (Mid), laning in the mid lane" in prompt
        assert "- Data coverage: the source of this match recorded no hero positions or HP" in prompt
        assert "and no ability cast times" in prompt and "do not write that the player was alone" in prompt
        assert "which allies were nearby cannot be told" in prompt and "No allies nearby" not in prompt
        assert "No ally with a known position nearby" in evaluator.allies_missing_note(
            death_ctx(map_state={"unknown_position_allies": [unknown_ally]}), "en")   # a replay with one gap
        assert "## Other Heroes (this match's data has no positions: where they were is unknown)" in prompt
        assert "- Lion (Lv5, Alive)\n" in prompt and "position data unavailable\n" not in prompt
        assert "does not show whether the player was alone" in partial
        assert not CJK_RE.search(evaluator.decision_extras(ctx, lang) + situation + partial)
    else:
        assert "- 本场位置: 2号位 (mid)，对线期在中路" in prompt
        assert "- 数据范围: 这场比赛的数据来源没有记录英雄的位置和血量" in prompt
        assert "也没有技能释放的时间" in prompt and "不要说玩家“孤身一人”" in prompt
        assert "无法判断哪些队友在附近" in prompt and "附近没有队友\n" not in prompt
        assert "## 其他英雄（本场数据没有位置记录：他们当时在哪里未知）" in prompt
        assert "- Lion (Lv5, 存活)\n" in prompt and "位置数据缺失\n" not in prompt
        assert "也不说明玩家是否孤身一人" in partial
    # A replay (positions known, all allies located) gets neither the note nor the hedge.
    replay = death_ctx(lang=lang, phase="mid_game", mc={"player_position": 4, "lane_role": 3, "has_positions": True})
    text = evaluator._build_evaluation_prompt(replay, "", lang=lang)
    assert evaluator._t(lang)["label_coverage"] not in text
    assert evaluator._t(lang)["no_allies"] in text and evaluator._t(lang)["no_allies_known"] not in text
    assert ("- Role in this match: Position 4 (Support)\n" if lang == "en" else "- 本场位置: 4号位 (support)\n") in text
    assert evaluator._t(lang)["label_role"] not in evaluator._build_evaluation_prompt(death_ctx(lang=lang), "", lang=lang)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_lists_the_position_every_hero_played(lang):
    """Live report: an enemy Hoodwink was "the ranged support" and a mid Axe's lane opponent got the role
    of his hero note. The positions derived for this match are one line of the match state."""
    positions = {"radiant": [["Drow Ranger", 1], ["Axe", 2], ["Hoodwink", 5]], "dire": [["Viper", 1], ["Lion", 5]]}
    radiant = death_ctx(lang=lang, mc={"player_position": 2, "positions": positions})
    dire = death_ctx(lang=lang, mc={"team": "dire", "player_position": 1, "positions": positions})
    ours, theirs = evaluator.positions_text(radiant, lang), evaluator.positions_text(dire, lang)
    prompt = evaluator._build_evaluation_prompt(radiant, "", lang=lang)
    rule = evaluator._t(lang)["position_rule"]
    if lang == "en":
        assert ours == ("Positions in this match (derived from lanes and net worth): your team: Drow Ranger (Position 1), "
                        "Axe (Position 2), Hoodwink (Position 5); enemy team: Viper (Position 1), Lion (Position 5)")
        assert theirs.endswith("your team: Viper (Position 1), Lion (Position 5); enemy team: "
                               "Drow Ranger (Position 1), Axe (Position 2), Hoodwink (Position 5)")
        assert "do not call a Position 4 or 5 just 'a support'" in rule and not CJK_RE.search(ours + rule)
    else:
        assert ours == ("各英雄本场位置（按分路和经济推算）: 己方 Drow Ranger 1号位、Axe 2号位、Hoodwink 5号位；"
                        "敌方 Viper 1号位、Lion 5号位")
        assert theirs.endswith("己方 Viper 1号位、Lion 5号位；敌方 Drow Ranger 1号位、Axe 2号位、Hoodwink 5号位")
        assert "不要把4号位、5号位只称为“辅助”" in rule
    assert ours in evaluator.format_match_situation(radiant, lang) and ours in prompt and rule in prompt
    # One side unknown is said, nothing known is left out together with its rule.
    half = evaluator.positions_text(death_ctx(lang=lang, mc={"positions": {"radiant": [["Axe", 2]], "dire": []}}), lang)
    assert half.endswith("enemy team: unknown" if lang == "en" else "敌方 未知")
    for mc in ({}, {"positions": {"radiant": [], "dire": []}}, {"positions": None}):
        bare = death_ctx(lang=lang, mc=mc)
        assert evaluator.positions_text(bare, lang) == ""
        assert rule not in evaluator._build_evaluation_prompt(bare, "", lang=lang)


def test_aftermath_line_reports_a_buyback():
    after = {"seconds": 30, "enemy_deaths": [], "allied_deaths": [], "towers_taken": [], "towers_lost": [],
             "buyback_after": 2}
    assert ("In the 30s after this death: the player bought back 2s after dying; no other hero kill"
            in evaluator.format_match_situation(death_ctx(mc={"aftermath": after}), "en"))
    assert "阵亡后30秒内: 玩家在阵亡2秒后买活；没有其他击杀记录" in evaluator.format_match_situation(
        death_ctx(mc={"aftermath": after}, lang="zh"), "zh")
    # A kill of his own after the buyback is his, not "a teammate's" (live verdict).
    mine = dict(after, enemy_deaths=["Viper", "Lion"], own_kills=["Viper"], died_again_after=17)
    assert ("the player bought back 2s after dying; the player died again 17s after this death; "
            "your team killed Viper (the player's own kill), Lion") in evaluator.format_match_situation(
        death_ctx(mc={"aftermath": mine}), "en")
    assert "玩家在阵亡2秒后买活；玩家在这次阵亡17秒后再次阵亡；己方击杀了" in evaluator.format_match_situation(
        death_ctx(mc={"aftermath": mine}, lang="zh"), "zh")
    assert "己方击杀了Viper（玩家本人击杀）、Lion" in evaluator.format_match_situation(
        death_ctx(mc={"aftermath": mine}, lang="zh"), "zh")
    twice = dict(after, allied_deaths=["Drow Ranger", "Warlock", "Warlock"])     # died, bought back, died again
    assert "your team also lost Drow Ranger, Warlock×2" in evaluator.format_match_situation(
        death_ctx(mc={"aftermath": twice}), "en")


def test_match_situation_gives_the_players_own_score_and_the_limits_of_a_kill_log():
    """A live verdict called a player's sixth death "your 9th death of the game" (9 was his
    deny count) and a kill listed without assists a "solo kill" (OpenDota records no assists)."""
    ctx = death_ctx(mc={"player_kills": 3, "player_deaths": 5, "has_positions": False, "has_assists": False})
    assert "The player before this moment: 3 kill(s), 5 death(s)" in evaluator.format_match_situation(ctx, "en")
    assert "玩家本人在此之前: 击杀3次，阵亡5次" in evaluator.format_match_situation(ctx, "zh")
    assert "before this moment" not in evaluator.format_match_situation(death_ctx(), "en")
    note = evaluator.coverage_note(ctx, "en")
    assert "a kill listed without assists is not necessarily a solo kill" in note
    assert "otherwise do not name a place" in note
    zh = evaluator.coverage_note(ctx, "zh")
    assert "没有列出助攻的击杀不一定是单杀" in zh and "否则不要说它发生在哪里" in zh
    with_assists = death_ctx(mc={"has_positions": False, "has_assists": True})
    assert "solo kill" not in evaluator.coverage_note(with_assists, "en")


def test_prompt_lists_dead_and_unlocated_heroes_as_such():
    map_state = {
        "distant_allies": [{"hero_name": "Zeus", "level": 9, "is_alive": True, "distance": 6100.0, "area": "Dire jungle"}],
        "unknown_position_enemies": [
            {"hero_name": "Lion", "level": 8, "is_alive": True},
            {"hero_name": "Viper", "level": 10, "is_alive": False},
        ],
    }
    text = evaluator._format_other_heroes(map_state, "en")
    assert "Zeus (Lv9, Alive) — about 6100 units away, at Dire jungle" in text
    assert "Lion (Lv8, Alive) — position data unavailable" in text
    assert "Viper (Lv10) — dead, waiting to respawn" in text
    zh = evaluator._format_other_heroes(map_state, "zh")
    assert "位置数据缺失" in zh and "已阵亡，等待复活" in zh and "位置未知" not in zh


def test_match_situation_states_the_score_and_the_dead():
    ctx = death_ctx(mc={"team_kills": 8, "enemy_kills": 13, "kills_estimated": False,
                        "nearby_allies": 1, "nearby_enemies": 2, "dead_allies": 1, "dead_enemies": 0})
    text = evaluator.format_match_situation(ctx, "en")
    assert "Score at this time (kills): your team 8 - 13 enemy team" in text
    assert "Dead at this time: 1 allied and 0 enemy hero(es)" in text
    ctx.match_context["kills_estimated"] = True
    assert "recorded teamfights only, may undercount" in evaluator.format_match_situation(ctx, "en")
    assert "此刻比分（击杀数）: 己方 8 - 13 敌方" in evaluator.format_match_situation(
        death_ctx(mc={"team_kills": 8, "enemy_kills": 13}), "zh")


def test_decision_without_a_timestamp_gets_no_zero_minute_state():
    """OpenDota "Achieved a Double Kill": the prompt showed "Lv1 ... starting items"."""
    dp = DecisionPoint(time=0, point_type="good_play", player_slot=1, hero_name="Wraith King",
                       severity="positive", brief="Achieved a Double Kill", time_known=False)
    early = hero("Wraith King", 1, level=1, items=["Quelling Blade"])
    ctx = DecisionContext(decision_point=dp, player_state=early, match_context={"team": "radiant"})
    line = evaluator.format_player_state(ctx, "en")
    assert "Lv1" not in line and "Quelling Blade" not in line and "no timestamp" in line
    final = hero("Wraith King", 1, level=22, items=["Radiance"])
    ctx = DecisionContext(decision_point=dp, player_state=final,
                          match_context={"team": "radiant", "state_is_final": True})
    line = evaluator.format_player_state(ctx, "en")
    assert "Lv22" in line and "Radiance" in line and "end-of-match state" in line and "Status" not in line
    prompt = evaluator._build_evaluation_prompt(ctx, "", lang="en")
    assert "Unknown (the data source recorded no exact timestamp" in prompt


def test_hero_line_leaves_unknown_values_out():
    line = evaluator._format_hero_state(hero("Zeus", 3, level=0), "Ally", "en")
    assert "Lv?" in line and "HP" not in line and "Mana" not in line and "Skills: unknown" in line
    level_one = evaluator._format_hero_state(hero("Zeus", 3, level=1), "Ally", "en")
    assert "Not yet skilled" in level_one
    no_mana = evaluator._format_hero_state(hero("Zeus", 3, hp=300, max_hp=600), "Ally", "en")
    assert "HP 300/600 (50%)" in no_mana and "Mana" not in no_mana
    # After the laning stage "CS 88/0" claimed zero denies for a hero with nine.
    known = evaluator._format_hero_state(hero("Zeus", 3, last_hits=88, denies=9, net_worth=6000), "Ally", "en")
    # "Gold 6000" was read as 6000 unspent gold, "CS 103/9" as a score ("your 9th death").
    assert "Net worth 6000, Last hits 88, denies 9," in known
    unknown = evaluator._format_hero_state(hero("Zeus", 3, last_hits=88, denies=None, net_worth=6000), "Ally", "en")
    assert "Last hits 88, Status" in unknown and "denies" not in unknown
    zh = evaluator._format_hero_state(hero("Zeus", 3, last_hits=88, denies=9, net_worth=6000), "队友", "zh")
    assert "总经济 6000, 正补 88、反补 9," in zh


def test_unknown_inventory_is_not_shown_as_no_items():
    """OpenDota's purchase log can start after the horn: "Items: None" for a hero at 2:26."""
    ogre = hero("Ogre Magi", 3, level=2)
    ogre.items_known = False
    en = evaluator._format_hero_state(ogre, "Ally", "en")
    assert "Items: not in the data for this moment (the purchase log starts later)" in en and "None" not in en
    zh = evaluator._format_hero_state(ogre, "队友", "zh")
    assert "装备: 数据中没有这一时刻的装备（购买记录从之后才开始）" in zh and "装备: 无" not in zh
    assert "Items: None" in evaluator._format_hero_state(hero("Ogre Magi", 3, level=2), "Ally", "en")


def test_times_before_the_horn_are_shown_as_negative():
    dp = DecisionPoint(time=-10, point_type="fight_engage", player_slot=1, hero_name="Hoodwink",
                       severity="major", brief="First blood fight")
    assert evaluator._time_display(dp, "en") == "-00:10 (-10s) (before the horn)"
    events = [EventEntry(time=-12, event_type="ability_cast", description="Hoodwink cast Bushwhack"),
              EventEntry(time=65, event_type="kill", description="kill")]
    assert evaluator._format_event_timeline(events, "en") == "  [-00:12] Hoodwink cast Bushwhack\n  [01:05] kill"


# ---------------------------------------------------------------------------
# Verdict parser (AP44)
# ---------------------------------------------------------------------------

DP = DecisionPoint(time=100, point_type="death", player_slot=0, hero_name="X", severity="critical", brief="b")


def parse(text, lang="en", strict=True):
    return evaluator._parse_evaluation_response(text, DP, lang=lang, strict=strict)


def test_bold_emphasis_keeps_its_words():
    ev = parse("Score: C\nAnalysis: Legion Commander **stayed in lane at 40% HP** while **three enemies** "
               "were missing.\nMistake: m\nBetter Play: b\nLesson: l")
    assert ev.analysis == "Legion Commander stayed in lane at 40% HP while three enemies were missing."
    zh = parse("评分: C\n分析: 军团在**血量只有40%**时仍留在线上。\n错误: 无\n更好的做法: 继续保持\n教训: 注意小地图", "zh")
    assert zh.analysis == "军团在血量只有40%时仍留在线上。" and zh.lesson == "注意小地图"


@pytest.mark.parametrize("text", [
    "**Score:** A\n**Analysis:** Nice **one**.\n**Mistake:** None\n**Better Play:** Keep it up\n**Lesson:** L",
    "**Score**: A\n**Analysis**: Nice one.\n**Mistake**: None\n**Better Play**: Keep it up\n**Lesson**: L",
    "## Score\nA\n## Analysis\nNice one.\n## Mistake\nNone\n## Better Play\nKeep it up\n## Lesson\nL",
    "score: A\nanalysis: Nice one.\nmistake: None\nBetter play: Keep it up\nLESSON: L",
])
def test_label_layouts_are_all_understood(text):
    ev = parse(text)
    assert ev._parsed_ok and ev.score == "A"
    assert (ev.analysis, ev.mistake, ev.better_play, ev.lesson) == ("Nice one.", "None", "Keep it up", "L")


def test_a_word_that_starts_a_sentence_is_not_a_label():
    ev = parse("Score: C\nAnalysis: He walked up alone.\nMistakes were made early: he overextended.\n"
               "Mistake: m\nBetter Play: b\nLesson: l")
    assert ev.analysis == "He walked up alone.\nMistakes were made early: he overextended." and ev.mistake == "m"
    zh = parse("评分: C\n分析: 走位靠前。\n错误地估计了对方伤害，被秒。\n错误: 走位\n更好的做法: 后撤\n教训: 小心", "zh")
    assert zh.analysis == "走位靠前。\n错误地估计了对方伤害，被秒。" and zh.mistake == "走位"
    full_width = parse("评分：B\n分析：还行。\n错误：无\n更好的做法：继续保持\n教训：稳", "zh")
    assert (full_width.score, full_width.analysis, full_width.lesson) == ("B", "还行。", "稳")


def test_strict_grade_and_stock_answers():
    assert [evaluator.strict_grade(v) for v in ("b", " A+ ", "S", "Average", "S|A|B|C|D", "", None, 3)] == [
        "B", "A", "S", None, None, None, None, None]
    assert evaluator.localize_stock_answer("None", "zh") == "无"
    assert evaluator.localize_stock_answer("Keep it up.", "zh") == "继续保持"
    assert evaluator.localize_stock_answer("无", "en") == "None"
    assert evaluator.localize_stock_answer("Stay with the team", "zh") == "Stay with the team"


# ---------------------------------------------------------------------------
# Phase review rules (AP49, A6)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stats, score", [
    # the three real examples that were graded C
    ({"kills": 9, "deaths": 1, "assists": 2, "gpm_in_phase": 1303, "teamfight_count": 5, "teamfight_wins": 5}, "A"),
    ({"kills": 5, "deaths": 1, "assists": 0, "teamfight_count": 4, "teamfight_wins": 4}, "A"),
    ({"kills": 0, "deaths": 0, "assists": 2, "teamfight_count": 1, "teamfight_wins": 1}, "B"),
    ({"kills": 0, "deaths": 0, "assists": 0}, "B"),                 # a quiet phase is neutral
    # ...whatever the GPM: nothing says what strong farm is within one phase, position and mode
    ({"kills": 0, "deaths": 0, "assists": 0, "gpm_in_phase": 650}, "B"),
    ({"kills": 1, "deaths": 1, "assists": 0, "gpm_in_phase": 792, "teamfight_count": 1}, "C"),   # a mid after 30:00
    ({"kills": 1, "deaths": 2, "assists": 4, "gpm_in_phase": 663, "teamfight_count": 5, "teamfight_wins": 4}, "C"),  # Turbo
    ({"kills": 0, "deaths": 0, "assists": 12}, "A"),
    ({"kills": 0, "deaths": 1, "assists": 12}, "A"),
    ({"kills": 1, "deaths": 1, "assists": 1}, "B"),
    ({"kills": 0, "deaths": 1, "assists": 0}, "C"),
    ({"kills": 15, "deaths": 2, "assists": 10}, "B"),
    ({"kills": 1, "deaths": 2, "assists": 2}, "C"),
    ({"kills": 2, "deaths": 3, "assists": 5}, "C"),
    ({"kills": 0, "deaths": 3, "assists": 2}, "D"),
    ({"kills": 0, "deaths": 4, "assists": 0, "gpm_in_phase": 900}, "D"),   # farm does not excuse feeding
    # no assist data (OpenDota): fights won with the player in them count instead
    ({"kills": 0, "deaths": 1, "assists": None, "teamfight_count": 3, "teamfight_wins": 2}, "B"),
    ({"kills": 0, "deaths": 1, "assists": None}, "C"),
])
def test_phase_rule_grade_weighs_deaths_against_impact(stats, score):
    review = evaluator._phase_review_fallback(PhaseReview(phase="mid_late", time_range="20-30 min"), stats, lang="en")
    assert review.score == score and not review.used_llm


def test_phase_rule_summary_says_what_the_numbers_are():
    def summary(stats, lang="en"):
        return evaluator._phase_review_fallback(PhaseReview(phase="laning", time_range=""), stats, lang=lang).summary

    full = {"kills": 9, "deaths": 1, "assists": 2, "last_hits_delta": 127, "gpm_in_phase": 1303,
            "teamfight_count": 5, "teamfight_wins": 5}
    assert summary(full) == "KDA 9/1/2, +127 last hits, 1303 GPM, teamfights won 5/5."
    assert summary(full, "zh") == "KDA 9/1/2，补刀+127，GPM 1303，团战胜 5/5。"
    assert summary({"kills": 1, "deaths": 2, "assists": None}) == "Kills/deaths 1/2."
    assert "KDA" not in summary({"kills": 1, "deaths": 2, "assists": None}, "zh")
    estimated = summary({"kills": 1, "deaths": 0, "assists": 3, "kda_estimated": True})
    assert "recorded teamfights" in estimated and "KDA" not in estimated and "3" not in estimated
    assert estimated == "1 kill in recorded teamfights, 0 located deaths."
    assert summary({"kills": 2, "deaths": 1, "kda_estimated": True}) == (
        "2 kills in recorded teamfights, 1 located death.")
    review = evaluator._phase_review_fallback(
        PhaseReview(phase="late", time_range="30+ min"), {"start_min": 30, "end_min": None, "kills": 0}, lang="en")
    assert (review.start_min, review.end_min) == (30, None)


def test_phase_prompt_labels_the_kda_honestly():
    def prompt(stats, lang="en"):
        return evaluator._build_phase_prompt(hero_name="Lion", position_label="Position 5", team="dire",
                                             phase_key="early_mid", phase_stats=stats, lang=lang)

    exact = prompt({"kills": 1, "deaths": 2, "assists": 7, "death_times": [724, 1175], "time_range": "10-20 min"})
    assert "- Kills / deaths / assists in this phase: 1/2/7" in exact
    assert "- Deaths at (game clock): 12:04, 19:35" in exact and "KDA:" not in exact
    no_assists = prompt({"kills": 1, "deaths": 2, "assists": None})
    assert "- Kills / deaths in this phase: 1/2 (the data source records no assists" in no_assists
    estimated = prompt({"kills": 1, "deaths": 0, "assists": 2, "kda_estimated": True})
    assert "Kills in recorded teamfights: 1 (an undercount" in estimated and "NOT the player's full KDA" in estimated
    zh = prompt({"kills": 1, "deaths": 2, "assists": None, "death_times": []}, "zh")
    assert "本阶段击杀/死亡：1/2（数据源没有助攻记录" in zh and "死亡时间点（游戏时间）：无" in zh
    assert "farm" not in zh and "gank" not in zh
    # A live review of a position 4: "23 last hits suggests farming instead of supporting".
    assert "A Position 4 or 5 is not measured by last hits" in exact and "do not guess" in exact
    assert "4号位和5号位不以补刀数衡量" in zh and "不要根据某个数字去猜测" in zh
