"""Decision context and death causal chain (context_builder.py, causal_chain.py; contract A3).

Synthetic matches; item recipes and ability names are stubbed so the tests
do not depend on the game data on disk.
"""
from __future__ import annotations

import pytest

from backend.app import geo
from backend.app.ai import causal_chain, context_builder, data_manager, evaluator, game_data, vision
from backend.app.ai.models import CausalStep, DeathEvent, DecisionPoint
from backend.app.models import (
    AbilityEvent,
    KillEvent,
    LaningDamageEvent,
    SkillEvent,
    SmokeEvent,
    TeamfightEvent,
    TeamfightPlayer,
    WardEvent,
)
from test_ai_pipeline_helpers import CJK_RE, default_players, item, make_match, make_player

# Far from every tower: only heroes and wards give vision here.
SPOT = (1500.0, -3500.0)
RADIANT_BASE = (-6500.0, -6000.0)
DIRE_BASE = (6500.0, 6000.0)
NAMES = {0: "Juggernaut", 1: "Puck", 2: "Tinker", 3: "Rubick", 4: "Crystal Maiden",
         128: "Phantom Assassin", 129: "Lina", 130: "Axe", 131: "Lion", 132: "Witch Doctor"}


def stand(pos: tuple[float, float], start: int, end: int, hp: int = 100) -> list[list[int]]:
    return [[t, int(pos[0]), int(pos[1]), hp] for t in range(start, end + 1, 5)]


def near(dx: float, dy: float = 0.0) -> tuple[float, float]:
    return (SPOT[0] + dx, SPOT[1] + dy)


def logged_match(logs: dict[int, list[list[int]]], start: int = 0, end: int = 1200, **kw):
    """Puck (slot 1) stands at SPOT, everybody else in their base, unless ``logs`` says otherwise.

    A slot mapped to ``[]`` has no position data at all.
    """
    players = default_players()
    for p in players:
        p.laning_snapshots = []
        home = RADIANT_BASE if p.team == "radiant" else DIRE_BASE
        p.position_log = stand(SPOT if p.player_slot == 1 else home, start, end)
        if p.player_slot in logs:
            p.position_log = logs[p.player_slot]
    return make_match(players, **kw)


def positionless_match(**kw):
    """OpenDota-like: a kill log, no positions."""
    players = default_players()
    for p in players:
        p.laning_snapshots = []
    return make_match(players, **kw)


def kill(time: float, victim: int, killer: int | None = None, **kw) -> KillEvent:
    return KillEvent(time=time, victim_slot=victim, killer_slot=killer, **kw)


def death_of(slot: int, time: int, killer: int | None = 129, *, x: float = SPOT[0], y: float = SPOT[1],
             tf_idx: int = -1) -> DeathEvent:
    return DeathEvent(time=time, player_slot=slot, hero_name=NAMES[slot],
                      killer_hero=NAMES[killer] if killer is not None else "", killer_slot=killer,
                      x=x, y=y, is_teamfight=tf_idx >= 0, teamfight_idx=tf_idx, source="kills_log")


def death_point(death: DeathEvent) -> DecisionPoint:
    return DecisionPoint(time=death.time, point_type="death", player_slot=death.player_slot,
                         hero_name=death.hero_name, severity="critical", brief="Killed",
                         related_death=death)


def fighter(slot: int, **kw) -> TeamfightPlayer:
    data = dict(player_slot=slot, hero_name=NAMES[slot], x=0.0, y=0.0, damage_dealt=300,
                damage_taken=100, kills=0, deaths=0, abilities_used=[])
    data.update(kw)
    return TeamfightPlayer(**data)


def fight(start: int, end: int, *players: TeamfightPlayer) -> TeamfightEvent:
    return TeamfightEvent(start_time=start, end_time=end, radiant_deaths=1, dire_deaths=0,
                          players=list(players))


# ---------------------------------------------------------------------------
# State before a death, dead heroes, score
# ---------------------------------------------------------------------------

def _death_match():
    logs = {
        1: stand(SPOT, 0, 340) + [[345, int(SPOT[0]), int(SPOT[1]), 41]] + stand(SPOT, 350, 400, hp=0),
        0: stand(near(300), 0, 335) + stand(near(300), 340, 400, hp=0),     # ally, died just before
        129: stand(near(600), 0, 1200),                                     # the killer
        128: stand(near(0, 9000), 0, 1200),
        130: [],                                                            # no data
    }
    return logged_match(logs, kills_log=[
        kill(100.0, 128, 0),
        kill(338.0, 0, 129, x=near(300)[0], y=SPOT[1]),
        kill(349.6, 1, 129, x=SPOT[0], y=SPOT[1] + 10, assist_slots=[128]),
    ])


def test_death_state_is_the_last_moment_alive_not_the_death_tick():
    match = _death_match()
    ctx = context_builder.build_context(match, death_point(death_of(1, 349)), lang="en")
    ps = ctx.player_state
    assert ps.state_time == 345 and ps.hp_pct == 41 and ps.is_alive
    assert (ps.x, ps.y) == SPOT
    mc = ctx.match_context
    assert mc["state_time"] == 345 and mc["seconds_before_death"] == 4 and mc["pre_death_hp_pct"] == 41
    assert all(s.state_time == 345 for s in ctx.ally_states + ctx.enemy_states)


def test_dead_heroes_are_not_nearby_and_are_listed_as_dead():
    ctx = context_builder.build_context(_death_match(), death_point(death_of(1, 349)), lang="en")
    assert [s.hero_name for s in ctx.ally_states] == []
    assert [s.hero_name for s in ctx.enemy_states] == ["Lina"]
    assert ctx.enemy_states[0].distance_to_player == 600
    mc, ms = ctx.match_context, ctx.map_state
    assert (mc["nearby_allies"], mc["nearby_enemies"]) == (0, 1)
    assert mc["dead_allies"] == 1 and ms["dead_allies"] == ["Juggernaut"]
    dead = [d for d in ms["unknown_position_allies"] if d["hero_name"] == "Juggernaut"]
    assert dead and dead[0]["is_alive"] is False and dead[0]["hp_pct"] == 0
    # A living hero without data is unknown, not dead and not nearby.
    assert mc["unknown_position_enemies"] == 1
    axe = [d for d in ms["unknown_position_enemies"] if d["hero_name"] == "Axe"]
    assert axe and axe[0]["is_alive"] is True
    assert mc["enemies_unknown"] == 1
    assert "no position data for 1 (Axe)" in ms["vision_summary"]


def test_score_is_exact_and_from_the_players_side():
    match = _death_match()
    radiant = context_builder.build_context(match, death_point(death_of(1, 349)), lang="en").match_context
    assert (radiant["team_kills"], radiant["enemy_kills"]) == (1, 1)     # before this death
    assert radiant["kills_estimated"] is False
    dp = DecisionPoint(time=400, point_type="fight_engage", player_slot=129, hero_name="Lina",
                       severity="major", brief="fight")
    dire = context_builder.build_context(match, dp, lang="en").match_context
    assert (dire["team_kills"], dire["enemy_kills"]) == (2, 1)
    assert (dire["radiant_kills"], dire["dire_kills"]) == (1, 2)


def test_without_a_kill_log_the_score_stays_an_estimate():
    dp = DecisionPoint(time=400, point_type="bad_trade", player_slot=1, hero_name="Puck",
                       severity="major", brief="trade")
    assert context_builder.build_context(make_match(), dp, lang="en").match_context["kills_estimated"] is True


def test_snapshot_only_death_takes_the_vitals_before_the_kill():
    players = default_players()
    for s in players[1].laning_snapshots:
        if s.time == 345:
            s.hp = 246
        if s.time >= 350:
            s.hp = 0
    ctx = context_builder.build_context(make_match(players), death_point(death_of(1, 350, x=-500.0, y=-500.0)),
                                        lang="en")
    ps = ctx.player_state
    assert ps.state_time == 345 and (ps.hp, ps.max_hp) == (246, 600) and ps.hp_pct == 41


def test_state_before_the_first_snapshot_does_not_borrow_it():
    """8713267082 Hoodwink, killed at -0:10: the state at -0:15 carried "0/566 HP" from the
    0:00 snapshot, taken when she was already dead."""
    players = default_players()
    for s in players[1].laning_snapshots:
        if s.time <= 20:
            s.hp = 0                                   # dead when the snapshots start
    players[1].position_log = stand(SPOT, -85, -15, hp=83) + stand(SPOT, -10, 20, hp=0)
    match = make_match(players, kills_log=[kill(-10.3, 1, 129, x=SPOT[0], y=SPOT[1])])
    ps = context_builder.build_context(match, death_point(death_of(1, -10)), lang="en").player_state
    assert ps.state_time == -15 and ps.hp_pct == 83
    assert (ps.hp, ps.max_hp) == (0, 0) and ps.level == 1        # no absolute numbers for that moment
    assert context_builder._snapshot_at(players[1].laning_snapshots, -15) is None
    assert context_builder._snapshot_at(players[1].laning_snapshots, 7).time == 5


def test_death_before_the_horn_is_not_in_its_own_score_and_shows_no_later_gold():
    """8713267082 Hoodwink -0:10 (first blood): the prompt said "score 0 - 1" and gave the
    killer the 297 gold he only had after the kill."""
    players = default_players()
    players[1].position_log = stand(SPOT, -85, -15, hp=83) + stand(SPOT, -10, 20, hp=0)
    players[6].position_log = stand(near(300), -85, 600)
    players[6].gold_t = [297, 600, 900]                # value at 0:00, after the first-blood bounty
    match = make_match(players, kills_log=[kill(-10.3, 1, 129, x=SPOT[0], y=SPOT[1])])
    ctx = context_builder.build_context(match, death_point(death_of(1, -10)), lang="en")
    assert (ctx.match_context["team_kills"], ctx.match_context["enemy_kills"]) == (0, 0)
    lina = next(e for e in ctx.enemy_states if e.hero_name == "Lina")
    assert lina.net_worth == 0 and lina.level == 1
    # After the horn the series is the value at the start of the minute, as before.
    players[6].laning_snapshots = []
    assert context_builder._build_hero_state(players[6], 70, match).net_worth == 600


def test_denies_after_laning_come_from_the_series_or_are_unknown():
    players = default_players()
    players[1].lh_t = [0] * 14 + [88, 95, 99]
    players[1].dn_t = [0] * 14 + [9, 9, 9]
    match = make_match(players)
    dp = DecisionPoint(time=900, point_type="fight_engage", player_slot=1, hero_name="Puck",
                       severity="major", brief="fight")
    ps = context_builder.build_context(match, dp, lang="en").player_state
    assert (ps.last_hits, ps.denies) == (95, 9)
    players[1].dn_t = []                               # a source without a deny series
    assert context_builder.build_context(make_match(players), dp, lang="en").player_state.denies is None


def test_match_context_carries_the_game_mode():
    dp = DecisionPoint(time=400, point_type="bad_trade", player_slot=1, hero_name="Puck",
                       severity="major", brief="trade")
    match = make_match(game_mode="turbo")
    assert context_builder.build_context(match, dp, lang="en").match_context["game_mode"] == "turbo"


def test_killer_without_position_is_placed_at_the_death_by_slot():
    match = positionless_match(kills_log=[kill(900.0, 1, 129, x=SPOT[0], y=SPOT[1])])
    ctx = context_builder.build_context(match, death_point(death_of(1, 900)), lang="en")
    assert [(s.hero_name, s.distance_to_player) for s in ctx.enemy_states] == [("Lina", None)]
    assert ctx.match_context["nearby_enemies"] == 1
    assert "Lina" not in [d["hero_name"] for d in ctx.map_state["unknown_position_enemies"]]
    assert ctx.player_state.has_position and getattr(ctx.player_state, "hp_pct", None) is None


def test_point_without_timestamp_gets_the_end_of_match_state():
    player = make_player(
        1, "Puck", level=22, last_hits=310, net_worth=18000,
        final_items=["Blink Dagger"], final_item_keys=["blink"], neutral_item="",
        items=[item("Iron Branch", "branches", -80)],
        skills=[SkillEvent(ability_name="Illusory Orb", ability_key="puck_illusory_orb", level=1, time=0),
                SkillEvent(ability_name="Illusory Orb", ability_key="puck_illusory_orb", level=2, time=900)],
    )
    players = default_players()
    players[1] = player
    dp = DecisionPoint(time=0, point_type="good_play", player_slot=1, hero_name="Puck",
                       severity="positive", brief="Double kill", time_known=False)
    ctx = context_builder.build_context(make_match(players), dp, lang="en")
    ps = ctx.player_state
    assert ps.level == 22 and ps.net_worth == 18000 and ps.state_time is None
    assert ps.items == [data_manager.item_display_name("blink")]
    assert list(ps.ability_levels.values()) == [2]
    assert ctx.match_context.get("state_is_final") is True
    assert ctx.vision_analysis is None and ctx.event_timeline == []
    assert ctx.ally_states == [] and ctx.enemy_states == []


def test_good_play_before_the_horn_is_a_real_moment():
    """8713267082: four kills between -10.3 s and 5.4 s. A double kill finished at -3 s was
    "time unknown" and the prompt described the level-1 fight with the end-of-match state."""
    player = make_player(
        1, "Puck", level=24, last_hits=310, net_worth=18000, final_item_keys=["black_king_bar"],
        items=[item("Iron Branch", "branches", -80), item("Black King Bar", "black_king_bar", 1500)],
        skills=[SkillEvent(ability_name="Illusory Orb", ability_key="puck_illusory_orb", level=1, time=-40),
                SkillEvent(ability_name="Illusory Orb", ability_key="puck_illusory_orb", level=2, time=900)],
        xp_t=[420, 900, 1500], gold_t=[700, 1200, 1800],
        position_log=stand(SPOT, -85, 600),
    )
    players = default_players()
    players[1] = player
    dp = DecisionPoint(time=-3, point_type="good_play", player_slot=1, hero_name="Puck",
                       severity="positive", brief="Achieved a Double Kill", subtype="multi_kill")
    ctx = context_builder.build_context(make_match(players), dp, lang="en")
    ps = ctx.player_state
    assert ps.state_time == -3 and ps.level == 1 and ps.net_worth == 0
    assert ps.items == [data_manager.item_display_name("branches")] and list(ps.ability_levels.values()) == [1]
    assert ctx.game_phase == "pregame" and "state_is_final" not in ctx.match_context
    assert ctx.vision_analysis is not None
    line = evaluator.format_player_state(ctx, "en")
    assert "Lv1" in line and "no timestamp" not in line and "Black King Bar" not in line
    assert evaluator._time_display(dp, "en").startswith("-00:03")


def test_level_between_two_minute_marks_is_not_a_minute_behind():
    """OpenDota prompts: "Ogre Magi (Lv1) ... Skills: Ignite(Lv1), Fireblast(Lv1)" at 2:25. The
    series was read at the start of the minute; OpenDota skill times come from the same series,
    interpolated. On the replays the interpolated level equals the exact one in 93% of the
    laning samples (start of the minute: 78%)."""
    player = make_player(3, "Rubick", xp_t=[0, 230, 650, 1200], items=[])
    match = make_match([player] + default_players()[1:])
    at = lambda t: context_builder._build_hero_state(player, t, match)  # noqa: E731
    assert [at(t).level for t in (60, 65, 110, 120, 150, 180)] == [1, 2, 2, 3, 3, 4]    # 240 / 640 / 1160 XP
    assert at(-10).level == 1 and at(5000).level == 4            # before the horn / past the series
    assert context_builder._xp_at([0, 230, 650], 90) == 440


def test_empty_inventory_is_unknown_when_the_purchase_log_starts_later():
    """OpenDota's purchase log of 8762775364 starts at 1:20 (tango) for Ogre Magi: three early
    deaths were described with "Items: None"."""
    late = make_player(3, "Rubick", items=[item("Tango", "tango", 80), item("Boots of Speed", "boots", 330)])
    match = make_match([late] + default_players()[1:])
    early = context_builder._build_hero_state(late, 146, match)
    assert early.items == [] and early.items_known is False
    assert "not in the data for this moment" in evaluator._format_hero_state(early, "Ally", "en")
    later = context_builder._build_hero_state(late, 400, match)
    assert later.items == [data_manager.item_display_name("boots")] and later.items_known
    # A log that has the starting purchases: nothing but consumables really is "none".
    full = make_player(3, "Rubick", items=[item("Tango", "tango", -80)])
    state = context_builder._build_hero_state(full, 146, make_match([full] + default_players()[1:]))
    assert state.items == [] and state.items_known
    assert "Items: None" in evaluator._format_hero_state(state, "Ally", "en")


# ---------------------------------------------------------------------------
# Head counts
# ---------------------------------------------------------------------------

def test_head_count_uses_positions_when_they_are_known():
    tf = fight(880, 920, fighter(1, deaths=1, death_time_offset=20.0), fighter(0), fighter(2),
               fighter(128), fighter(129), fighter(130))
    match = logged_match({
        129: stand(near(500), 0, 1200), 130: stand(near(-400), 0, 1200),     # two enemies on Puck
        0: stand(near(0, 8000), 0, 1200), 2: stand(near(0, 8200), 0, 1200),  # allies fight elsewhere
        128: stand(near(300, 8000), 0, 1200),
    }, teamfights=[tf], kills_log=[kill(900.0, 1, 129, x=SPOT[0], y=SPOT[1])])
    mc = context_builder.build_context(match, death_point(death_of(1, 900, tf_idx=0)), lang="en").match_context
    assert mc["headcount_basis"] == "nearby"
    assert (mc["headcount_allies"], mc["headcount_enemies"]) == (0, 2)
    assert (mc["fight_allies"], mc["fight_enemies"]) == (2, 3)      # the record covers the whole map


def test_head_count_falls_back_to_fight_participation_without_positions():
    tf = fight(880, 920, fighter(1, deaths=1, death_time_offset=20.0), fighter(0),
               fighter(128), fighter(129), fighter(130), fighter(131, damage_dealt=0, damage_taken=0))
    match = positionless_match(teamfights=[tf], kills_log=[kill(900.0, 1, 129, x=SPOT[0], y=SPOT[1])])
    mc = context_builder.build_context(match, death_point(death_of(1, 900, tf_idx=0)), lang="en").match_context
    assert mc["headcount_basis"] == "participation"
    assert (mc["headcount_allies"], mc["headcount_enemies"]) == (1, 3)
    # Outside a fight nothing is known about the numbers.
    mc = context_builder.build_context(match, death_point(death_of(1, 1500)), lang="en").match_context
    assert mc["headcount_basis"] == "unknown"


# ---------------------------------------------------------------------------
# The fight as it started, not as it ended (AP24)
# ---------------------------------------------------------------------------

def _fight_match(allies: dict[int, float], enemies: tuple[int, ...] = (128, 129, 130, 131), *,
                 puck: list[list[int]] | None = None, extra_kills: tuple = (), buyback: float | None = None):
    """Puck fights at SPOT with ``allies`` ({slot: time of death}) against ``enemies`` and dies at 900.4."""
    tf = fight(870, 930, fighter(1, deaths=1, death_time_offset=30.0), *[fighter(s) for s in allies],
               *[fighter(s) for s in enemies])
    logs = {1: puck or stand(SPOT, 0, 895) + stand(SPOT, 900, 960, hp=0)}
    kills = list(extra_kills)
    for i, (slot, died) in enumerate(allies.items()):
        spot = near(200 + 100 * i, 300)
        last = int(died) - int(died) % 5
        logs[slot] = stand(spot, 0, last) + stand(spot, last + 5, 1200, hp=0)
        if died < 1200:
            kills.append(kill(died, slot, 129, x=spot[0], y=spot[1]))
    for i, slot in enumerate(enemies):
        logs[slot] = stand(near(-300 - 150 * i, -400), 0, 1200)
    kills.append(kill(900.4, 1, 129, x=SPOT[0], y=SPOT[1]))
    match = logged_match(logs, teamfights=[tf], kills_log=sorted(kills, key=lambda k: k.time))
    if buyback is not None:
        match.players[1].buyback_log = [{"time": buyback}]
    return match, tf


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_last_survivor_of_a_4v4_is_not_told_he_fought_outnumbered(lang):
    """8762775364 Legion Commander 22:20: Weaver, Ogre Magi and Hoodwink die at 22:07-22:13, she
    dies last and was told "killed in a teamfight while outnumbered (1v4) ... Root cause: Fought
    while outnumbered; wait for teammates", with the step dated when it was still 3v4."""
    match, tf = _fight_match({0: 885.5, 2: 888.2, 3: 892.7})
    death = death_of(1, 900, tf_idx=0)
    eng = vision.engagement(match, 1, 900, tf)
    assert eng == vision.Engagement(fallen_allies=3, fallen_enemies=0, open_time=885,
                                    open_allies=3, open_enemies=4, opened_outnumbered=False)
    chain = _chain(match, death, lang)
    types = [s.step_type for s in chain.steps]
    assert "no_retreat" not in types and types.count("allies_fallen") == 1
    assert chain.root_cause_type == "" and chain.root_cause_summary == "" and _roots(chain) == []
    step = next(s for s in chain.steps if s.step_type == "allies_fallen")
    assert step.time == 885 and step.details["root_eligible"] is False
    assert step.description == (
        "From here on 3 allies died near the player before the player did; "
        "1v4 when the player died (heroes within 2500 units of the player)" if lang == "en"
        else "此后有3名队友先于玩家在其附近阵亡；玩家阵亡时为1v4（按玩家2500码范围内的英雄计）")
    ctx = context_builder.build_context(match, death_point(death), causal_chain=chain, lang=lang)
    mc = ctx.match_context
    assert (mc["headcount_allies"], mc["headcount_enemies"]) == (0, 4)       # who was left
    assert (mc["fallen_allies"], mc["fallen_enemies"], mc["opening_time"]) == (3, 0, 885)
    assert (mc["opening_allies"], mc["opening_enemies"], mc["opening_outnumbered"]) == (3, 4, False)
    ev = evaluator.evaluate_decision_fallback(ctx, lang=lang)
    text = " ".join([ev.analysis, ev.mistake, ev.better_play])
    assert ev.mistake == "" and "outnumbered" not in text and "人数劣势" not in text
    assert ("after 3 allies had already died in it (1v4 by then)" in ev.analysis) if lang == "en" else (
        "此前已有3名队友在这场团战中阵亡（此时为1v4）" in ev.analysis)
    situation = evaluator.format_match_situation(ctx, lang)
    assert ("3 allied and 0 enemy hero(es) died near the player (the first at 14:45)" in situation
            if lang == "en" else "此前已有3名队友和0名敌方英雄在玩家附近阵亡（最早在14:45）" in situation)


def test_fight_that_started_outnumbered_is_dated_and_counted_at_its_first_death():
    match, tf = _fight_match({0: 885.5, 2: 5000.0}, enemies=(128, 129, 130, 131, 132))
    death = death_of(1, 900, tf_idx=0)
    eng = vision.engagement(match, 1, 900, tf)
    assert (eng.fallen_allies, eng.open_allies, eng.open_enemies, eng.opened_outnumbered) == (1, 2, 5, True)
    chain = _chain(match, death)
    assert chain.root_cause_type == "teamfight_outnumbered"
    root, = _roots(chain)
    assert root.step_type == "no_retreat" and root.time == 885          # 3v5 was true then; 2v5 at the end
    assert root.description == ("Fought outnumbered 3v5 (heroes within 2500 units of the player) "
                                "when the first hero died; 1 ally died before the player")
    assert "allies_fallen" not in [s.step_type for s in chain.steps]
    ctx = context_builder.build_context(match, death_point(death), causal_chain=chain, lang="en")
    ev = evaluator.evaluate_decision_fallback(ctx, lang="en")
    assert "while outnumbered (3v5 when the first hero died, 1 ally died before the player)" in ev.analysis
    assert ev.mistake.startswith("Fought while outnumbered")


def test_allies_who_died_before_the_player_arrived_are_not_his_fight():
    """8713267082 Axe 24:15: two allies died while he was at the fountain; he teleported into a
    2v5 and "Fought while outnumbered" is exactly what happened."""
    arrives = stand(RADIANT_BASE, 0, 890) + [[895, int(SPOT[0]), int(SPOT[1]), 100]] + stand(SPOT, 900, 960, hp=0)
    match, tf = _fight_match({0: 885.5, 2: 888.2}, enemies=(128, 129, 130), puck=arrives)
    assert vision.engagement(match, 1, 900, tf) is None
    assert vision.deaths_near(match, 1, 870.0, 900.4) == []
    death = death_of(1, 900, tf_idx=0)
    chain = _chain(match, death)
    assert chain.root_cause_type == "teamfight_outnumbered" and "1v3" in _roots(chain)[0].description
    mc = context_builder.build_context(match, death_point(death), lang="en").match_context
    assert "fallen_allies" not in mc and (mc["headcount_allies"], mc["headcount_enemies"]) == (0, 3)


def test_fight_before_the_players_own_previous_death_is_another_fight():
    """8713267082 Warlock 28:51: died, bought back and came back alone 13 s later. The allies who
    fell before his first death do not make the second one a "last survivor"."""
    puck = (stand(SPOT, 0, 875) + [[880, int(SPOT[0]), int(SPOT[1]), 0], [885, int(SPOT[0]), int(SPOT[1]), 100]]
            + stand(SPOT, 890, 895) + stand(SPOT, 900, 960, hp=0))
    match, tf = _fight_match({0: 872.5, 2: 876.1}, enemies=(128, 129, 130), puck=puck,
                             extra_kills=(kill(879.6, 1, 129, x=SPOT[0], y=SPOT[1]),), buyback=882.0)
    assert vision.engagement(match, 1, 900, tf) is None
    first = vision.engagement(match, 1, 879, tf)
    assert (first.fallen_allies, first.open_allies, first.open_enemies) == (2, 2, 3)
    assert not first.opened_outnumbered


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_ally_dying_first_outside_a_recorded_fight_is_the_same_story(lang):
    """8713267082 Warlock 5:49: two deaths are no teamfight record, and "1v2 ... stayed within
    reach of 2 enemy heroes with no ally nearby" was said of a 2v2 his lane partner died in first."""
    match, _tf = _fight_match({0: 890.5}, enemies=(128, 129))
    match.teamfights = []
    death = death_of(1, 900)
    chain = _chain(match, death, lang)
    step, = [s for s in chain.steps if s.step_type == "allies_fallen"]
    assert step.time == 890 and ("1v2 when the player died" in step.description if lang == "en"
                                 else "玩家阵亡时为1v2" in step.description)
    assert _roots(chain) == []
    ctx = context_builder.build_context(match, death_point(death), causal_chain=chain, lang=lang)
    ev = evaluator.evaluate_decision_fallback(ctx, lang=lang)
    assert ev.mistake == "" and "no ally" not in ev.analysis and "没有队友" not in ev.analysis
    assert ("after 1 ally had already died in the same fight (1v2 by then)" in ev.analysis) if lang == "en" else (
        "此前已有1名队友在同一场战斗中阵亡（此时为1v2）" in ev.analysis)


def test_numbers_that_tip_for_one_sample_do_not_make_a_fight_outnumbered():
    """8713267082 Razor 0:00: 4v5 in the second the first hero died, 5v5 a few seconds before
    (an ally 2250 units away at the edge of the radius)."""
    match, tf = _fight_match({0: 885.5, 2: 5000.0, 3: 5000.0})
    edge = near(2400, 300)
    match.players[3].position_log = (stand(edge, 0, 880) + stand(near(2600, 300), 885, 1200))   # steps out at 14:45
    eng = vision.engagement(match, 1, 900, tf)
    assert (eng.open_allies, eng.open_enemies, eng.opened_outnumbered) == (2, 4, False)
    assert "no_retreat" not in [s.step_type for s in _chain(match, death_of(1, 900, tf_idx=0)).steps]


def test_death_before_the_horn_finds_its_own_fight():
    """Times are truncated towards zero: a death at -10.3 is dated -10."""
    logs = {1: stand(SPOT, -85, -15) + stand(SPOT, -10, 60, hp=0),
            0: stand(near(300), -85, -20) + stand(near(300), -15, 60, hp=0),
            129: stand(near(-300), -85, 600)}
    match = logged_match(logs, start=-85, kills_log=[kill(-17.2, 0, 129, x=near(300)[0], y=SPOT[1]),
                                                     kill(-10.3, 1, 129, x=SPOT[0], y=SPOT[1])])
    eng = vision.engagement(match, 1, -10)
    assert (eng.fallen_allies, eng.open_time, eng.open_allies, eng.open_enemies) == (1, -18, 1, 1)


# ---------------------------------------------------------------------------
# Inventory and abilities
# ---------------------------------------------------------------------------

_ITEMS = {
    "boots": 500, "blades_of_attack": 450, "chainmail": 550, "magic_stick": 200, "branches": 50,
    "ogre_axe": 1000, "belt_of_strength": 450, "blade_of_alacrity": 1000, "boots_of_elves": 450,
    "mithril_hammer": 1600, "blink": 2250, "aghanims_shard": 1400, "bracer": 505, "wraith_band": 505,
    "null_talisman": 505, "gloves": 450, "circlet": 155,
}
_RECIPES = {
    "phase_boots": (1500, ["boots", "chainmail", "blades_of_attack"]),
    "magic_wand": (450, ["magic_stick", "branches", "branches"]),
    "sange": (2100, ["ogre_axe", "belt_of_strength"]),
    "yasha": (2100, ["blade_of_alacrity", "boots_of_elves"]),
    "sange_and_yasha": (4200, ["sange", "yasha"]),
    "black_king_bar": (4050, ["mithril_hammer", "ogre_axe"]),
    "dagon": (3050, ["blades_of_attack"]),
}


def _stub_items(monkeypatch, with_components: bool = True) -> None:
    def fake(name):
        key = str(name).lower()
        if key.startswith("dagon_"):
            key = "dagon"                                   # upgrade levels resolve to the base item
        if key in _RECIPES:
            entry = {"key": key, "cost": _RECIPES[key][0]}
            if with_components:
                entry["components"] = list(_RECIPES[key][1])
            return entry
        return {"key": key, "cost": _ITEMS[key]} if key in _ITEMS else None

    monkeypatch.setattr(game_data, "get_item_data", fake)
    monkeypatch.setattr(context_builder, "_item_cache", {"generation": None, "facts": {}})


def _keys(names: list[str]) -> list[str]:
    return [data_manager.resolve_item_key(n) for n in names]


def _buyer(purchases: list[tuple[str, int]], **kw):
    return make_player(0, "Juggernaut", items=[item(key, key, t) for key, t in purchases], **kw)


def test_components_leave_the_inventory_when_the_item_is_built(monkeypatch):
    _stub_items(monkeypatch)
    player = _buyer([
        ("boots", -80), ("tango", -80), ("branches", -80), ("branches", -80),
        ("magic_stick", 100), ("recipe_magic_wand", 200), ("magic_wand", 200),
        ("blades_of_attack", 300), ("chainmail", 400), ("phase_boots", 400),
        ("ogre_axe", 900), ("belt_of_strength", 950), ("sange", 950),
        ("blade_of_alacrity", 1200), ("boots_of_elves", 1250), ("yasha", 1250), ("sange_and_yasha", 1250),
    ])
    at = lambda t: _keys(context_builder._items_at_time(player, t))  # noqa: E731
    assert at(150) == ["boots", "branches", "magic_stick"]
    assert at(350) == ["boots", "magic_wand", "blades_of_attack"]
    assert at(1000) == ["magic_wand", "phase_boots", "sange"]
    assert at(1300) == ["magic_wand", "phase_boots", "sange_and_yasha"]


def test_item_built_in_the_same_second_as_its_last_component(monkeypatch):
    _stub_items(monkeypatch)
    player = _buyer([("ogre_axe", 400), ("black_king_bar", 500), ("mithril_hammer", 500),
                     ("recipe_black_king_bar", 500)])
    assert _keys(context_builder._items_at_time(player, 500)) == ["black_king_bar"]


def test_upgrade_level_and_permanent_buffs(monkeypatch):
    _stub_items(monkeypatch)
    player = _buyer([("blades_of_attack", 100), ("dagon", 600), ("dagon_2", 900), ("aghanims_shard", 1000),
                     ("smoke_of_deceit", 1100), ("tpscroll", 1100)])
    assert _keys(context_builder._items_at_time(player, 1200)) == ["dagon", "aghanims_shard"]


def test_without_recipe_data_final_items_survive_the_slot_cap(monkeypatch):
    _stub_items(monkeypatch, with_components=False)
    cheap = ["circlet", "branches", "gloves", "boots", "magic_stick", "bracer", "wraith_band",
             "null_talisman", "blades_of_attack", "chainmail"]
    purchases = [(key, 10 * i) for i, key in enumerate(cheap)] + [("blink", 900), ("black_king_bar", 1500)]
    player = _buyer(purchases, final_item_keys=["blink", "black_king_bar", "circlet"], backpack=["branches"])
    held = _keys(context_builder._items_at_time(player, 2000))
    assert len(held) == 9
    assert {"blink", "black_king_bar", "circlet", "branches"} <= set(held)      # still owned at the end
    assert "magic_stick" not in held and "gloves" not in held                   # cheapest of the rest
    assert held.index("circlet") < held.index("blink") < held.index("black_king_bar")  # purchase order


def test_context_says_whether_the_inventory_could_be_collapsed(monkeypatch):
    dp = DecisionPoint(time=400, point_type="bad_trade", player_slot=1, hero_name="Puck",
                       severity="major", brief="trade")
    for table, expected in (({"Phase Boots": {"key": "phase_boots", "components": ["boots"]}}, True),
                            ({"Phase Boots": {"key": "phase_boots"}}, False)):
        monkeypatch.setattr(game_data, "get_item_table", lambda table=table: table)
        monkeypatch.setattr(context_builder, "_item_cache", {"generation": None, "facts": {}})
        assert context_builder.recipes_known() is expected
        ctx = context_builder.build_context(make_match(), dp, lang="en")
        assert ctx.match_context["inventory_exact"] is expected


def test_ability_levels_use_the_current_name_of_the_key(monkeypatch):
    names = {"dragon_knight_dragon_blood": "Dragon Blood", "dragon_knight_breathe_fire": "Breathe Fire",
             "special_bonus_hp_200": "+200 Health", "special_bonus_unique_x": "+{s:value} Damage"}
    monkeypatch.setattr(data_manager, "ability_display_name",
                        lambda key, lang="en": names.get(key, key.replace("_", " ").title()))
    monkeypatch.setattr(data_manager, "resolve_ability_name", lambda key: names.get(key))
    monkeypatch.setattr(context_builder, "_ability_cache", {"generation": None, "names": {}})

    def skill(key: str, stale_name: str, t: int) -> SkillEvent:
        return SkillEvent(ability_name=stale_name, ability_key=key, level=1, time=t)

    player = make_player(131, "Dragon Knight", skills=[
        skill("dragon_knight_dragon_blood", "Wyrm's Wrath", 0),          # stale catalog name
        skill("dragon_knight_breathe_fire", "Breathe Fire", 100),
        skill("dragon_knight_dragon_blood", "Wyrm's Wrath", 200),
        skill("special_bonus_attributes", "Attribute Bonus", 300),
        skill("special_bonus_hp_200", "+200 Health", 400),
        skill("special_bonus_unique_x", "+{s:value} Damage", 500),      # unresolved template
        # A talent no catalog knows would be named after its key ("Wraith King Facet 1").
        skill("special_bonus_unique_wraith_king_facet_1", "Wraith King Facet 1", 550),
        skill("dragon_knight_breathe_fire", "Breathe Fire", 900),
    ])
    assert context_builder._ability_levels_at_time(player, 600) == {
        "Dragon Blood": 2, "Breathe Fire": 1, "+200 Health": 1}
    assert context_builder._abilities_at_time(player, 150) == ["Dragon Blood", "Breathe Fire"]


# ---------------------------------------------------------------------------
# Event timeline
# ---------------------------------------------------------------------------

def _cast(slot: int, key: str, t: int, name: str = "") -> AbilityEvent:
    return AbilityEvent(player_slot=slot, hero_name=NAMES[slot], ability_name=name or key,
                        ability_key=key, time=t)


def test_timeline_cap_keeps_what_matters_and_drops_orb_switches(monkeypatch):
    monkeypatch.setattr(data_manager, "ability_display_name", lambda key, lang="en": key)
    monkeypatch.setattr(context_builder, "_ability_cache", {"generation": None, "names": {}})
    players = default_players()
    for p in players:
        p.laning_snapshots = []
    me = players[1]
    me.ability_events = (
        [_cast(1, "invoker_wex", t) for t in range(1440, 1500)]              # 60 no-op casts
        + [_cast(1, "puck_illusory_orb", 1441)]
        + [_cast(1, "puck_waning_rift", t) for t in (1480, 1482, 1484)]      # one entry, "×3"
    )
    others = [_cast(slot, f"spell_{slot}", t) for slot in NAMES if slot != 1 for t in range(1441, 1500, 6)]
    match = make_match(
        players, ability_events=me.ability_events + others,
        smokes=[SmokeEvent(time=1442, team="dire", player_slots=[128, 129])],
        kills_log=[kill(1445.3, 0, 129, assist_slots=[130]), kill(1499.7, 1, 129)],
    )
    events = context_builder._build_event_timeline(
        match, me, 1499, None, lang="en", nearby_slots=set(NAMES) - {1})
    assert len(events) == context_builder._MAX_TIMELINE_EVENTS
    assert [e.time for e in events] == sorted(e.time for e in events)
    text = [e.description for e in events]
    assert not any("invoker_wex" in d for d in text)
    assert "Puck cast puck_illusory_orb" in text                              # 58 s before, still there
    assert "Puck cast puck_waning_rift ×3" in text
    assert "Enemy team used Smoke of Deceit" in text
    assert "Lina killed Juggernaut (assists: Axe)" in text
    assert events[-1].event_type == "death" and events[-1].description == "Lina killed Puck"
    assert sum(1 for e in events if e.event_type == "ability_cast") < 40


def test_world_interactions_are_not_listed_as_casts(monkeypatch):
    """Twin Gates, Watchers, outposts and cosmetic actions say nothing about a fight."""
    monkeypatch.setattr(data_manager, "ability_display_name", lambda key, lang="en": key)
    monkeypatch.setattr(context_builder, "_ability_cache", {"generation": None, "names": {}})
    players = default_players()
    noise = ["twin_gate_portal_warp", "ability_lamp_use", "ability_capture", "plus_guild_banner",
             "ability_pluck_famango", "abyssal_underlord_portal_warp", "plus_high_five", "invoker_invoke"]
    players[1].ability_events = [_cast(1, key, 880 + i) for i, key in enumerate(noise)] + [
        _cast(1, "puck_phase_shift", 895)]
    match = make_match(players)
    events = context_builder._build_event_timeline(match, players[1], 900, None, lang="en")
    assert [e.description for e in events] == ["Puck cast puck_phase_shift"]


def test_unconfirmed_kill_entries_are_not_reported_as_deaths():
    match = positionless_match(kills_log=[kill(890.0, 0, 129, confirmed=False), kill(895.0, 2, 129)])
    events = context_builder._build_event_timeline(match, match.players[1], 900, None, lang="en")
    assert [e.description for e in events] == ["Lina killed Tinker"]


def test_laning_damage_is_summed_named_and_never_against_oneself():
    players = default_players()
    players[0].laning_damage_events = [
        LaningDamageEvent(time=100, target_idx=5, damage=50, inflictor="attack"),
        LaningDamageEvent(time=101, target_idx=5, damage=60, inflictor="dota_unknown"),   # older parser
        LaningDamageEvent(time=102, target_idx=0, damage=170, inflictor="soul_ring"),     # self damage
    ]
    players[5].laning_damage_events = [LaningDamageEvent(time=103, target_idx=0, damage=80, inflictor="attack")]
    match = make_match(players)
    en = [e.description for e in context_builder._build_event_timeline(match, players[0], 120, None, lang="en")]
    assert en == ["Juggernaut dealt 110 damage to Phantom Assassin (Attack)",
                  "Juggernaut took 80 damage from Phantom Assassin (Attack)"]
    zh = [e.description for e in context_builder._build_event_timeline(match, players[0], 120, None, lang="zh")]
    assert zh == ["主宰 对 幻影刺客 造成了 110 点伤害（普通攻击）",      # hero names in the report language
                  "主宰 受到了来自 幻影刺客 的 80 点伤害（普通攻击）"]
    assert all("dota_unknown" not in d and "soul_ring" not in d for d in en + zh)


def test_event_window_reaches_back_before_the_horn(monkeypatch):
    monkeypatch.setattr(data_manager, "ability_display_name", lambda key, lang="en": key)
    monkeypatch.setattr(context_builder, "_ability_cache", {"generation": None, "names": {}})
    players = default_players()
    players[1].ability_events = [_cast(1, "puck_phase_shift", -30)]
    match = make_match(players, kills_log=[kill(-10.3, 1, 129)])
    events = context_builder._build_event_timeline(match, players[1], -10, None, lang="en")
    assert [(e.time, e.event_type) for e in events] == [(-30, "ability_cast"), (-10, "death")]


def test_last_casts_of_a_far_away_killer_are_part_of_the_death(monkeypatch):
    """8762775364, 14:15: Zeus killed a 9% HP Hoodwink with Thundergod's Wrath from his fountain.
    The cast was not in the timeline (Zeus was not nearby), so the death had no visible cause."""
    monkeypatch.setattr(data_manager, "ability_display_name", lambda key, lang="en": key)
    monkeypatch.setattr(context_builder, "_ability_cache", {"generation": None, "names": {}})
    casts = [_cast(129, "zuus_thundergods_wrath", 899), _cast(129, "zuus_arc_lightning", 880),
             _cast(130, "axe_berserkers_call", 899)]
    match = logged_match({}, ability_events=casts, kills_log=[kill(900.2, 1, 129, x=SPOT[0], y=SPOT[1])])
    ctx = context_builder.build_context(match, death_point(death_of(1, 900, killer=129)), lang="en")
    cast_lines = [e.description for e in ctx.event_timeline if e.event_type == "ability_cast"]
    assert cast_lines == ["Lina cast zuus_thundergods_wrath"]   # not her older cast, not another far hero's


def test_casts_are_placed_by_where_the_player_stood_when_they_were_cast(monkeypatch):
    """8762775364, missed-CS point at 5:00 with the player dead since 4:50: every enemy cast on
    the map was listed, and the verdict counted an Ignite cast 14,000 units away among the spells
    "that landed on you"."""
    monkeypatch.setattr(data_manager, "ability_display_name", lambda key, lang="en": key)
    monkeypatch.setattr(context_builder, "_ability_cache", {"generation": None, "names": {}})
    logs = {1: stand(SPOT, 0, 885) + stand(SPOT, 890, 960, hp=0),      # dead from 890 on
            129: stand(near(500), 0, 1200)}                             # Lina next to him; Axe in his base
    casts = [_cast(129, "lina_dragon_slave", 880), _cast(130, "axe_berserkers_call", 880),
             _cast(129, "lina_light_strike_array", 895)]                # after his death: he has no place
    match = logged_match(logs, ability_events=casts, kills_log=[kill(890.0, 1, 129, x=SPOT[0], y=SPOT[1])])
    point = DecisionPoint(time=900, point_type="missed_cs", player_slot=1, hero_name="Puck",
                          severity="minor", brief="Low CS")
    ctx = context_builder.build_context(match, point, lang="en")
    assert not ctx.player_state.has_position
    assert [e.description for e in ctx.event_timeline if e.event_type == "ability_cast"] == [
        "Lina cast lina_dragon_slave"]


def test_aftermath_of_a_death_lists_what_followed_it():
    tower = {"type": "building_kill", "time": 915, "key": "npc_dota_badguys_tower1_mid"}
    match = positionless_match(objectives=[tower], kills_log=[
        kill(880.0, 2, 129),                        # before the death: not aftermath
        kill(900.4, 1, 129),                        # the death itself
        kill(900.9, 0, 129),                        # same second: already in the timeline
        kill(905.0, 129, 0), kill(912.0, 3, 130),
        kill(920.0, 131, 0, confirmed=False),       # may be a reincarnation
        kill(931.0, 130, 0),                        # 31 s later
    ])
    ctx = context_builder.build_context(match, death_point(death_of(1, 900, killer=129)), lang="en")
    assert ctx.match_context["aftermath"] == {
        "seconds": 30, "enemy_deaths": ["Lina"], "allied_deaths": ["Rubick"], "own_kills": [],
        "towers_taken": ["mid_t1"], "towers_lost": [], "buyback_after": None, "died_again_after": None}
    # The player's own score before this second: he killed nobody, and this death is his first.
    assert (ctx.match_context["player_kills"], ctx.match_context["player_deaths"]) == (0, 0)
    assert ctx.match_context["has_assists"] is False        # an OpenDota-like kill log
    # Only a death has an aftermath; without a kill log nothing is known.
    other = DecisionPoint(time=900, point_type="fight_engage", player_slot=1, hero_name="Puck",
                          severity="major", brief="Fight")
    assert "aftermath" not in context_builder.build_context(match, other, lang="en").match_context
    assert context_builder._aftermath(positionless_match(), match.players[1], 900) is None


def test_buyback_is_in_the_timeline_and_in_the_aftermath():
    """8713267082: Axe died at 28:08, bought back at 28:10 and died again at 28:25. Neither verdict
    mentioned the buyback: nothing in the prompt showed it."""
    players = default_players()
    for p in players:
        p.laning_snapshots = []
    players[1].buyback_log = [{"time": 1690}]
    match = make_match(players, kills_log=[kill(1688.2, 1, 131), kill(1705.6, 1, 129)])
    match.kills_log.insert(1, kill(1703.0, 129, 1, assist_slots=[0]))     # his own kill after buying back
    first = context_builder.build_context(match, death_point(death_of(1, 1688, killer=131)), lang="en")
    assert first.match_context["aftermath"]["buyback_after"] == 2
    assert first.match_context["aftermath"]["own_kills"] == ["Lina"]
    # His second death is his own fate, not one more "allied death".
    assert first.match_context["aftermath"]["died_again_after"] == 17
    assert first.match_context["aftermath"]["allied_deaths"] == []
    second = context_builder.build_context(match, death_point(death_of(1, 1705, killer=129)), lang="zh")
    assert second.match_context["aftermath"]["buyback_after"] is None
    assert (second.match_context["player_kills"], second.match_context["player_deaths"]) == (1, 1)
    assert second.match_context["has_assists"] is True
    assert [(e.time, e.event_type, e.description) for e in second.event_timeline] == [
        (1688, "death", "莱恩 击杀了 帕克"), (1690, "buyback", "帕克 买活"),
        (1703, "kill", "帕克 击杀了 莉娜（助攻：主宰）"), (1705, "death", "莉娜 击杀了 帕克")]
    # the data behind the text keeps the English names every lookup uses
    assert second.event_timeline[0].details["killer"] == "Lion" and second.event_timeline[0].details["victim"] == "Puck"


def test_match_without_position_logs_says_so_and_names_no_place_for_the_player():
    """8713267082 from OpenDota, 4:34: the mid laner's death was placed in "the top lane", where
    his team had warded a minute before; the wards were listed among "his" events."""
    ward = WardEvent(ward_type="observer", x=-5300.0, y=5500.0, time=860, player_slot=4)
    match = positionless_match(wards=[ward], kills_log=[kill(900.0, 1, 129)])
    death = death_of(1, 900, killer=129, x=0.0, y=0.0)          # OpenDota: no place for a solo kill
    ctx = context_builder.build_context(match, death_point(death), lang="en")
    assert ctx.match_context["has_positions"] is False and ctx.match_context["has_cast_log"] is False
    assert ctx.match_context["lane_role"] == 2
    assert not [e for e in ctx.event_timeline if "ward" in e.event_type]     # none of unknown relevance
    vision_text = ctx.vision_analysis.formatted_text
    assert "Player position: unknown (the data has no position for this moment" in vision_text
    assert "Observer ward placed at" in vision_text                          # still listed as a map event
    located = logged_match({}, wards=[WardEvent(ward_type="observer", x=SPOT[0] + 500, y=SPOT[1], time=860,
                                                 player_slot=4)], kills_log=[kill(900.0, 1, 129)])
    near = context_builder.build_context(located, death_point(death_of(1, 900, killer=129)), lang="en")
    assert near.match_context["has_positions"] is True
    assert "Allied team placed an Observer Ward" in [e.description for e in near.event_timeline]
    assert "Player position: unknown" not in near.vision_analysis.formatted_text


# ---------------------------------------------------------------------------
# Causal chain
# ---------------------------------------------------------------------------

def _chain(match, death: DeathEvent, lang: str = "en"):
    return causal_chain.build_causal_chain(match, death, death.player_slot, lang=lang)


def _roots(chain) -> list:
    return [s for s in chain.steps if s.significance == "root_cause"]


def _outnumbered_match(enemies_near: bool):
    tf = fight(885, 930, fighter(1, deaths=1, death_time_offset=15.0), fighter(0), fighter(2),
               fighter(128), fighter(129), fighter(130))
    spot = near if enemies_near else (lambda dx, dy=0.0: near(dx, 9000 + dy))
    logs = {
        1: stand(SPOT, 0, 895) + stand(SPOT, 900, 960, hp=0),
        0: stand(near(0, 8000), 0, 1200), 2: stand(near(200, 8000), 0, 1200),
        128: stand(spot(500), 0, 1200), 129: stand(spot(-500), 0, 1200), 130: stand(spot(0, 600), 0, 1200),
    }
    return logged_match(logs, teamfights=[tf], kills_log=[kill(900.0, 1, 129, x=SPOT[0], y=SPOT[1])])


def test_outnumbered_counts_the_heroes_near_the_player():
    match = _outnumbered_match(enemies_near=True)
    death = death_of(1, 900, tf_idx=0)
    chain = _chain(match, death)
    assert chain.root_cause_type == "teamfight_outnumbered"
    root = _roots(chain)
    assert len(root) == 1 and root[0].step_type == "no_retreat"
    assert "1v3" in root[0].description and "within 2500 units" in root[0].description
    # The context reports the same numbers for the same moment.
    mc = context_builder.build_context(match, death_point(death), causal_chain=chain, lang="en").match_context
    assert (mc["headcount_allies"] + 1, mc["headcount_enemies"]) == (1, 3)
    zh = _chain(match, death, "zh")
    assert "1v3" in _roots(zh)[0].description and "人数劣势" in zh.root_cause_summary


def test_fight_record_alone_does_not_make_a_death_outnumbered():
    # Three enemies "took part" in the fight, on the other side of the map.
    chain = _chain(_outnumbered_match(enemies_near=False), death_of(1, 900, tf_idx=0))
    assert not [s for s in chain.steps if s.step_type == "no_retreat"]
    assert chain.root_cause_type == "" and chain.root_cause_summary == "" and _roots(chain) == []


def test_steps_are_dated_before_the_death_and_the_death_comes_last():
    # The fight window starts at the first death: its steps used to be dated after it.
    tf = fight(900, 930, fighter(1, deaths=1, death_time_offset=0.0, damage_taken=900), fighter(0),
               fighter(128), fighter(129), fighter(130))
    logs = {1: stand(SPOT, 0, 895) + stand(SPOT, 900, 960, hp=0),
            128: stand(near(500), 0, 1200), 129: stand(near(-500), 0, 1200), 130: stand(near(0, 600), 0, 1200)}
    match = logged_match(logs, teamfights=[tf], kills_log=[kill(900.4, 1, 129, x=SPOT[0], y=SPOT[1])])
    chain = _chain(match, death_of(1, 900, tf_idx=0))
    assert len(chain.steps) >= 3
    assert all(s.time <= 900 for s in chain.steps)
    assert [s.step_type for s in chain.steps].count("death") == 1 and chain.steps[-1].step_type == "death"
    assert [s.time for s in chain.steps] == sorted(s.time for s in chain.steps)
    assert all(s.time < 900 for s in _roots(chain))
    focus = [s for s in chain.steps if s.step_type == "position_forward"]
    assert focus and "First on the team to die" in focus[0].description


def test_participation_is_worded_as_such_and_needs_a_clear_margin():
    def chain_for(*enemies: int):
        tf = fight(885, 930, fighter(1, deaths=1, death_time_offset=15.0), fighter(0),
                   *[fighter(s) for s in enemies])
        match = positionless_match(teamfights=[tf], kills_log=[kill(900.0, 1, 129, x=SPOT[0], y=SPOT[1])])
        return _chain(match, death_of(1, 900, tf_idx=0))

    clear = chain_for(128, 129, 130, 131)                       # 2v4
    step = next(s for s in clear.steps if s.step_type == "no_retreat")
    assert "4 enemy heroes took part in the fight against 2 allied (2v4)" in step.description
    assert "nearby" not in step.description and "within" not in step.description
    assert clear.root_cause_type == "teamfight_outnumbered"
    assert "took part" in clear.root_cause_summary
    close = chain_for(128, 129, 130)                            # 2v3: one more somewhere on the map
    assert [s.step_type for s in close.steps] == ["no_retreat", "death"]
    assert close.root_cause_type == "" and _roots(close) == []


def test_enemies_without_position_data_never_explain_a_death():
    match = positionless_match(kills_log=[kill(900.0, 1, 129, x=SPOT[0], y=SPOT[1])])
    for lang in ("en", "zh"):
        chain = _chain(match, death_of(1, 900), lang)
        assert [s.step_type for s in chain.steps] == ["death"]
        assert chain.root_cause_type == "" and chain.root_cause_summary == ""
    assert not CJK_RE.search(_chain(match, death_of(1, 900), "en").formatted_text)


def test_enemies_slipping_into_the_fog_nearby_are_a_step():
    logs = {
        128: stand(near(1000), 0, 860) + stand(near(3000), 865, 1200),
        129: stand(near(-1000), 0, 860) + stand(near(2000, 2500), 865, 1200),
        1: stand(SPOT, 0, 895) + stand(SPOT, 900, 960, hp=0),
    }
    match = logged_match(logs, kills_log=[kill(900.0, 1, 129, x=SPOT[0], y=SPOT[1])])
    chain = _chain(match, death_of(1, 900))
    missing = [s for s in chain.steps if s.step_type == "enemy_missing"]
    assert len(missing) == 1 and missing[0].time == 865
    assert "Phantom Assassin, Lina moved out of allied vision close to the player" == missing[0].description
    assert chain.root_cause_type in ("ignore_missing", "bad_positioning", "overextend")
    assert len(_roots(chain)) == 1 and _roots(chain)[0].time < 900


def test_hidden_enemy_coming_within_reach_did_not_leave_vision_just_now():
    """8762775364 Legion Commander 31:04: "Wraith King, Void Spirit moved out of allied vision close
    to the player" while both had been unseen for up to 34 s; 30 of 63 names in such steps had
    been in the fog all along (the player walked towards them)."""
    a, b = (-1500.0, -3000.0), (2500.0, -3000.0)
    hide = ((6800.0, -2500.0), (6600.0, -3800.0))                    # 8300 from a, 4300 from b
    logs = {1: stand(a, 0, 850) + stand(b, 855, 895) + stand(b, 900, 960, hp=0)}
    for slot, spot in zip((128, 130), hide):                         # seen next to Puck until 13:30
        logs[slot] = stand((a[0] + 600, a[1] + 100 * (slot - 128)), 0, 810) + stand(spot, 815, 1200)
    match = logged_match(logs, kills_log=[kill(900.0, 1, 128, x=b[0], y=b[1])])
    for lang in ("en", "zh"):
        chain = _chain(match, death_of(1, 900, killer=128, x=b[0], y=b[1]), lang)
        assert not [s for s in chain.steps if s.step_type == "enemy_missing"]
        step, = [s for s in chain.steps if s.step_type == "hidden_enemy_near"]
        assert step.time == 855
        assert step.description == (
            "Phantom Assassin, Axe, already out of sight (for up to 45s), were now within 5000 units of the player"
            if lang == "en" else "幻影刺客、斧王此前就不在己方视野内（最长已45秒），此时距玩家已不足5000码")
        assert "moved out of" not in chain.formatted_text and "脱离己方视野" not in chain.formatted_text
    assert causal_chain._classify_root_type(step) == "ignore_missing"
    assert "slipped out of vision" not in causal_chain._root_summary(step, [step], death_of(1, 900), "ignore_missing", "en")


def test_enemy_that_respawns_nearby_neither_left_vision_nor_was_hidden():
    """4 of 63 names in such steps had just respawned (the player was at the enemy base)."""
    siege = (DIRE_BASE[0] - 3000.0, DIRE_BASE[1] - 1500.0)
    logs = {1: stand(siege, 0, 895) + stand(siege, 900, 960, hp=0)}
    for slot in (128, 130):                              # dead until 14:25, then back at their fountain
        logs[slot] = stand(DIRE_BASE, 0, 860, hp=0) + stand(DIRE_BASE, 865, 1200)
    for slot in (129, 131, 132):
        logs[slot] = stand(RADIANT_BASE, 0, 1200)        # the rest of the Dire is far away
    match = logged_match(logs, kills_log=[kill(900.0, 1, 128, x=siege[0], y=siege[1])])
    chain = _chain(match, death_of(1, 900, killer=128, x=siege[0], y=siege[1]))
    assert chain.window_start <= 860                     # the respawn at 14:25 is inside the window
    assert not [s for s in chain.steps if s.step_type in ("enemy_missing", "hidden_enemy_near")]
    assert "moved out of" not in chain.formatted_text and "out of sight" not in chain.formatted_text


def test_ally_who_died_next_to_the_player_did_not_leave_him():
    def absent_step(ally_dies: bool):
        logs = {1: stand(SPOT, 0, 895) + stand(SPOT, 900, 960, hp=0),
                128: stand(near(3000), 0, 1200), 129: stand(near(2000, 2500), 0, 1200)}
        kills = [kill(900.0, 1, 129, x=SPOT[0], y=SPOT[1])]
        if ally_dies:
            logs[0] = stand(near(300), 0, 865) + stand(near(300), 870, 1200, hp=0)
            kills.insert(0, kill(867.3, 0, 129, x=near(300)[0], y=SPOT[1]))
        else:
            logs[0] = stand(near(300), 0, 865) + stand(RADIANT_BASE, 870, 1200)
        chain = _chain(logged_match(logs, kills_log=kills), death_of(1, 900))
        step, = [s for s in chain.steps if s.step_type == "ally_absent"]
        return chain, step

    chain, died = absent_step(True)
    assert died.description.startswith("The ally near the player died") and died.details["root_eligible"] is False
    assert chain.root_cause_type != "isolated"
    _chain2, left = absent_step(False)
    assert left.description.startswith("No allied hero near the player any more") and left.details["root_eligible"]


def test_no_step_is_dated_after_the_death_whatever_built_it(monkeypatch):
    """Guard of build_causal_chain: a late step is pulled back to the death second and the death
    still closes the chain (AP25: the root cause used to be dated after the death)."""
    late = [CausalStep(time=905, step_type="vision_loss", description="late", significance="contributing"),
            CausalStep(time=900, step_type="enemy_missing", description="same second", significance="contributing")]
    monkeypatch.setattr(causal_chain, "_detect_transitions", lambda *a, **kw: list(late))
    match = logged_match({1: stand(SPOT, 0, 895) + stand(SPOT, 900, 960, hp=0)},
                         kills_log=[kill(900.0, 1, 129, x=SPOT[0], y=SPOT[1])])
    chain = _chain(match, death_of(1, 900))
    assert [(s.step_type, s.time) for s in chain.steps] == [
        ("vision_loss", 900), ("enemy_missing", 900), ("death", 900)]
    assert _roots(chain) == []                         # too late to react to: no root cause
    # Fight steps are added after the death step: one in the death second must still precede it.
    fight_step = CausalStep(time=903, step_type="position_forward", description="fight", significance="contributing")
    monkeypatch.setattr(causal_chain, "_enrich_teamfight", lambda steps, *a, **kw: (steps + [fight_step], False))
    match.teamfights = [fight(885, 930, fighter(1, deaths=1, death_time_offset=15.0), fighter(129))]
    chain = _chain(match, death_of(1, 900, tf_idx=0))
    assert [(s.step_type, s.time) for s in chain.steps][-2:] == [("position_forward", 900), ("death", 900)]


def _warded_match(ward: WardEvent, puck: list[list[int]]):
    # Two enemies hidden nearby keep the analysis window open for the whole two minutes.
    logs = {1: puck, 128: stand(near(3000), 0, 1200), 129: stand(near(2000, 2500), 0, 1200)}
    return logged_match(logs, wards=[ward], kills_log=[kill(900.0, 1, 129, x=SPOT[0], y=SPOT[1])])


def test_lost_ward_is_a_step_only_when_the_ward_is_really_gone():
    dead = stand(SPOT, 900, 960, hp=0)
    killed = WardEvent(ward_type="observer", x=SPOT[0], y=SPOT[1], time=600, player_slot=4,
                       destroyed_time=880, dewarded=True)
    chain = _chain(_warded_match(killed, stand(SPOT, 0, 895) + dead), death_of(1, 900))
    lost = [s for s in chain.steps if s.step_type == "vision_loss"]
    assert len(lost) == 1 and lost[0].time == 880
    assert lost[0].description.startswith("The observer ward at ") and "was destroyed" in lost[0].description
    # The player walking out of the range of a ward that still stands lost no ward.
    standing = WardEvent(ward_type="observer", x=SPOT[0], y=SPOT[1], time=600, player_slot=4)
    walked = stand(SPOT, 0, 850) + stand(near(2500), 855, 895) + dead
    chain = _chain(_warded_match(standing, walked), death_of(1, 900, x=near(2500)[0]))
    assert not [s for s in chain.steps if s.step_type == "vision_loss"]
    assert "lack of vision" not in chain.root_cause_summary


def test_farming_after_laning_is_stated_per_minute_not_per_second():
    def farm_step(last_hits_in_minute: int):
        logs = {1: stand(SPOT, 0, 895) + stand(SPOT, 900, 960, hp=0), 128: stand(near(3000), 0, 1200),
                129: stand(near(2000, 2500), 0, 1200), 130: stand(near(3500, 1000), 0, 1200)}
        match = logged_match(logs, kills_log=[kill(900.0, 1, 129, x=SPOT[0], y=SPOT[1])])
        match.players[1].lh_t = [100] * 15 + [100 + last_hits_in_minute, 200]     # minute 14 -> 15
        return [s for s in _chain(match, death_of(1, 900)).steps if s.step_type == "farm_under_threat"]

    steps = farm_step(6)
    assert len(steps) == 1 and steps[0].time == 840
    assert "(6 last hits in that minute) while 3 enemies were hidden in the fog nearby" in steps[0].description
    assert farm_step(1) == []


def test_last_hits_of_a_group_are_not_the_player_farming():
    """8762775364, 24:18: four Radiant heroes at the Dire tier-1 tower; the chain said the player
    "kept farming at Dire jungle (6 last hits in that minute)"."""
    def farm_steps(allies: int):
        logs = {1: stand(SPOT, 0, 895) + stand(SPOT, 900, 960, hp=0), 128: stand(near(3000), 0, 1200),
                129: stand(near(2000, 2500), 0, 1200), 130: stand(near(3500, 1000), 0, 1200)}
        for slot in (0, 2, 3)[:allies]:
            logs[slot] = stand(near(-300 * (slot + 1)), 0, 1200)
        match = logged_match(logs, kills_log=[kill(900.0, 1, 129, x=SPOT[0], y=SPOT[1])])
        match.players[1].lh_t = [100] * 15 + [106, 200]
        return [s for s in _chain(match, death_of(1, 900)).steps if s.step_type == "farm_under_threat"]

    assert farm_steps(2) == [] and len(farm_steps(1)) == 1


def test_window_does_not_reach_back_past_the_previous_death():
    logs = {1: (stand(SPOT, 0, 995) + stand(SPOT, 1000, 1035, hp=0)          # first death
                + stand(near(100), 1040, 1065) + stand(near(100), 1070, 1200, hp=0))}
    match = logged_match(logs, kills_log=[kill(1000.0, 1, 129, x=SPOT[0], y=SPOT[1]),
                                          kill(1069.5, 1, 129, x=near(100)[0], y=SPOT[1])])
    chain = _chain(match, death_of(1, 1069, x=near(100)[0]))
    assert chain.window_start >= 1040
    assert not [s for s in chain.steps if s.step_type == "damage_burst"]


def test_buyback_death_is_not_explained_by_the_death_before_it():
    """8713267082 Warlock 28:51, 13 s after his previous death: "HP dropped from 53% to 0%"."""
    tf = fight(985, 1040, fighter(1, deaths=2, death_time_offset=15.0), fighter(0), fighter(129))
    logs = {1: (stand(SPOT, 0, 990) + [[995, 1500, -3500, 60], [1000, 1500, -3500, 0],
                                       [1005, 1500, -3500, 100], [1010, 1500, -3500, 80]]
                + stand(SPOT, 1015, 1200, hp=0)),
            129: stand(near(400), 0, 1200)}
    match = logged_match(logs, teamfights=[tf], kills_log=[kill(999.6, 1, 129, x=SPOT[0], y=SPOT[1]),
                                                           kill(1013.5, 1, 129, x=SPOT[0], y=SPOT[1])])
    match.players[1].buyback_log = [{"time": 1003}]
    chain = _chain(match, death_of(1, 1013, tf_idx=0))
    assert chain.window_start > 999
    assert not [s for s in chain.steps if "0%" in s.description]
    assert all(s.time > 999 for s in chain.steps)


def test_pick_off_during_a_fight_window_is_not_part_of_that_fight():
    """8762775364 Lion 2:17: killed elsewhere while a fight was recorded on the other lane."""
    tf = fight(885, 930, fighter(1, damage_taken=900), fighter(0), fighter(128), fighter(129),
               fighter(130), fighter(131))
    match = positionless_match(teamfights=[tf], kills_log=[kill(900.0, 1, 129, x=SPOT[0], y=SPOT[1])])
    alone = death_of(1, 900)                                  # teamfight_idx -1, from the kill log
    assert causal_chain._find_fight(match, alone) is None
    assert [s.step_type for s in _chain(match, alone).steps] == ["death"]
    mc = context_builder.build_context(match, death_point(alone), lang="en").match_context
    assert "fight_allies" not in mc and mc["headcount_basis"] == "unknown"
    # A death located by the old heuristics has no fight index of its own: the window decides.
    guessed = death_of(1, 900)
    guessed.source = "abilities"
    assert causal_chain._find_fight(match, guessed) is tf
    assert _chain(match, death_of(1, 900, tf_idx=0)).steps[0].step_type == "no_retreat"


ENEMY_HALF = ((1500.0, 1200.0), (2500.0, 2000.0))     # Dire mid lane, 4000 -> 5300 from Radiant towers
OWN_HALF = ((-300.0, -3900.0), (900.0, -3300.0))      # Radiant jungle, 2350 -> 3100 from them


def _walk_match(path, allies_along: bool = False, death_at: float = 900.0):
    """Puck walks from path[0] to path[1] at 855 with two enemies hidden close by, and dies there."""
    a, b = path
    logs = {1: stand(a, 0, 850) + stand(b, 855, 895) + stand(b, 900, 960, hp=0),
            128: stand((b[0] + 2500, b[1] + 300), 0, 1200), 129: stand((b[0] + 300, b[1] + 2600), 0, 1200)}
    if allies_along:
        for slot, dx in ((0, 150), (2, -150)):
            logs[slot] = stand((a[0] + dx, a[1]), 0, 850) + stand((b[0] + dx, b[1]), 855, 1200)
    return logged_match(logs, kills_log=[kill(death_at, 1, 129, x=b[0], y=b[1])])


def test_pushing_forward_is_the_players_move_only_alone_and_on_the_enemy_half():
    """8713267082 Dragon Knight 24:16 died sieging the enemy base with four allies and was
    "over-extended, 10436 units from the nearest allied tower"; 8762775364 LC 19:16 had
    "moved toward Radiant bottom lane" (her own side) a minute before a 4-hero fight."""
    def steps(path, **kw):
        chain = _chain(_walk_match(path, **kw), death_of(1, 900, x=path[1][0], y=path[1][1]))
        return chain, [s.step_type for s in chain.steps]

    chain, types = steps(ENEMY_HALF)
    assert "position_forward" in types and "position_unsafe" in types
    forward = next(s for s in chain.steps if s.step_type == "position_forward")
    assert forward.time == 855 and "Dire mid lane" in forward.description
    assert chain.root_cause_type in ("overextend", "bad_positioning", "ignore_missing")

    chain, types = steps(OWN_HALF)
    assert "position_forward" not in types and chain.root_cause_type != "overextend"
    assert "position_unsafe" in types                  # alone in an unwarded part of the own jungle

    chain, types = steps(ENEMY_HALF, allies_along=True)
    assert "position_forward" not in types and "position_unsafe" not in types
    assert chain.root_cause_type != "overextend"


def test_standing_in_lane_while_laning_is_not_poor_positioning():
    """8762775364 Tidehunter 2:21 died in his own lane: "was in an unwarded area ... only go
    deep with vision coverage" was the root cause."""
    lane = (2500.0, 2000.0)                            # Dire mid lane, far from Radiant towers
    def chain_at(death_time: int):
        logs = {1: stand(lane, 0, death_time - 5) + stand(lane, death_time, death_time + 60, hp=0),
                128: stand((lane[0] + 2500, lane[1] + 300), 0, 1200),
                129: stand((lane[0] + 300, lane[1] + 2600), 0, 1200)}
        match = logged_match(logs, kills_log=[kill(float(death_time), 1, 129, x=lane[0], y=lane[1])])
        return _chain(match, death_of(1, death_time, x=lane[0], y=lane[1]))

    laning = chain_at(300)
    assert "position_unsafe" not in [s.step_type for s in laning.steps]
    assert laning.root_cause_type != "bad_positioning"
    assert "position_unsafe" in [s.step_type for s in chain_at(900).steps]      # after the laning stage


def test_staying_in_danger_is_ignoring_missing_enemies_not_overextending():
    step = causal_chain.CausalStep(time=800, step_type="ignore_warning", description="", significance="")
    assert causal_chain._classify_root_type(step) == "ignore_missing"


def test_step_in_the_last_seconds_is_no_root_cause():
    """8713267082 Warlock 19:57: a ward died 4 s before him and "he neither checked the minimap
    nor retreated" was the root cause."""
    def chain_for(gone: int):
        logs = {1: stand(SPOT, 0, 895) + stand(SPOT, 900, 960, hp=0),
                128: stand(near(1000), 0, gone - 5) + stand(near(3000), gone, 1200),
                129: stand(near(-1000), 0, gone - 5) + stand(near(2000, 2500), gone, 1200)}
        match = logged_match(logs, kills_log=[kill(897.5, 1, 129, x=SPOT[0], y=SPOT[1])])
        return _chain(match, death_of(1, 897))

    late = chain_for(895)                              # both enemies leave vision after 14:52
    missing = [s for s in late.steps if s.step_type == "enemy_missing"]
    assert len(missing) == 1 and 897 - missing[0].time < 5
    assert late.root_cause_type == "" and late.root_cause_summary == "" and _roots(late) == []
    in_time = chain_for(880)                           # a quarter of a minute to react
    assert len(_roots(in_time)) == 1 and _roots(in_time)[0].time <= 880


def test_fight_with_the_team_is_not_explained_by_the_walk_before_it():
    a, b = ENEMY_HALF
    tf = fight(885, 930, fighter(1, deaths=1, death_time_offset=15.0), fighter(0), fighter(2),
               fighter(128), fighter(129))
    logs = {1: stand(a, 0, 820) + stand(b, 825, 895) + stand(b, 900, 960, hp=0),
            0: stand(RADIANT_BASE, 0, 875) + stand((b[0] + 200, b[1]), 880, 1200),
            2: stand(RADIANT_BASE, 0, 875) + stand((b[0] - 200, b[1]), 880, 1200),
            128: stand((b[0] + 2500, b[1] + 300), 0, 880) + stand((b[0] + 600, b[1]), 885, 1200),
            129: stand((b[0] + 300, b[1] + 2600), 0, 880) + stand((b[0], b[1] + 600), 885, 1200)}
    match = logged_match(logs, teamfights=[tf], kills_log=[kill(900.0, 1, 129, x=b[0], y=b[1])])
    chain = _chain(match, death_of(1, 900, x=b[0], y=b[1], tf_idx=0))
    assert "position_forward" in [s.step_type for s in chain.steps]      # still part of the story
    assert chain.root_cause_type == "" and _roots(chain) == []            # 3v2 with the team around


def test_fight_with_the_team_stays_one_after_the_team_has_died():
    """The walk into a 3v2 does not become the root cause because both allies fell before the
    player and he was alone at the end."""
    a, b = ENEMY_HALF
    tf = fight(885, 930, fighter(1, deaths=1, death_time_offset=15.0), fighter(0), fighter(2),
               fighter(128), fighter(129))
    logs = {1: stand(a, 0, 820) + stand(b, 825, 895) + stand(b, 900, 960, hp=0),
            128: stand((b[0] + 2500, b[1] + 300), 0, 880) + stand((b[0] + 600, b[1]), 885, 1200),
            129: stand((b[0] + 300, b[1] + 2600), 0, 880) + stand((b[0], b[1] + 600), 885, 1200)}
    kills = []
    for slot, dx, died in ((0, 200, 891.5), (2, -200, 893.2)):
        logs[slot] = (stand(RADIANT_BASE, 0, 875) + stand((b[0] + dx, b[1]), 880, 890)
                      + stand((b[0] + dx, b[1]), 895, 1200, hp=0))
        kills.append(kill(died, slot, 129, x=b[0] + dx, y=b[1]))
    kills.append(kill(900.0, 1, 129, x=b[0], y=b[1]))
    match = logged_match(logs, teamfights=[tf], kills_log=kills)
    chain = _chain(match, death_of(1, 900, x=b[0], y=b[1], tf_idx=0))
    types = [s.step_type for s in chain.steps]
    assert "position_forward" in types and "allies_fallen" in types and "no_retreat" not in types
    assert chain.root_cause_type == "" and _roots(chain) == []


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_outnumbered_in_the_own_base_is_a_defence_not_a_root_cause(lang):
    """8762775364 LC 32:53: 4v5 on the Radiant high ground, "wait for teammates or avoid the fight"."""
    ax, ay = geo.ANCIENT_POSITIONS["radiant"]
    home = (ax + 400.0, ay + 300.0)
    tf = fight(885, 930, fighter(1, deaths=1, death_time_offset=15.0), fighter(128), fighter(129), fighter(130))
    logs = {1: stand(home, 0, 895) + stand(home, 900, 960, hp=0)}
    for slot in (0, 2, 3, 4):                          # the rest of the team is not at home
        logs[slot] = stand(SPOT, 0, 1200)
    for i, slot in enumerate((128, 129, 130)):
        logs[slot] = stand((home[0] + 500 + 100 * i, home[1] + 400), 0, 1200)
    match = logged_match(logs, teamfights=[tf], kills_log=[kill(900.0, 1, 129, x=home[0], y=home[1])])
    chain = _chain(match, death_of(1, 900, x=home[0], y=home[1], tf_idx=0), lang)
    step = next(s for s in chain.steps if s.step_type == "no_retreat")
    assert ("Defended the base outnumbered 1v3" in step.description) if lang == "en" else (
        "以1v3的人数劣势防守基地" in step.description)
    assert chain.root_cause_type == "" and _roots(chain) == []


def test_repeated_danger_is_one_step_per_half_minute(monkeypatch):
    """8762775364 Hoodwink 19:35: "became dangerous" at 19:19 and "very dangerous" at 19:24."""
    b = ENEMY_HALF[1]
    logs = {1: stand(b, 0, 895) + stand(b, 900, 960, hp=0)}
    for i, slot in enumerate((128, 129, 130, 131)):       # they leave Puck's vision one by one
        gone = 860 + 5 * i
        logs[slot] = stand((b[0] + 900, b[1] + 100 * i), 0, gone - 5) + stand(
            (b[0] + 2600, b[1] + 100 * i), gone, 1200)
    match = logged_match(logs, kills_log=[kill(900.0, 1, 129, x=b[0], y=b[1])])

    def warnings():
        return [s for s in _chain(match, death_of(1, 900, x=b[0], y=b[1])).steps
                if s.step_type == "ignore_warning"]

    assert len(warnings()) == 1
    assert "but the player did not move back towards safety" in warnings()[0].description
    # With the team around, staying is the team's decision and is worded that way.
    for slot in (0, 2):
        match.players[slot].position_log = stand((b[0] - 300, b[1] - 200 * slot), 0, 1200)
    grouped = _chain(make_match(match.players, kills_log=match.kills_log), death_of(1, 900, x=b[0], y=b[1]))
    stayed = [s.description for s in grouped.steps if s.step_type == "ignore_warning"]
    assert stayed and all("but the player and the 2 allies nearby did not move back" in d for d in stayed)
    zh = _chain(make_match(match.players, kills_log=match.kills_log), death_of(1, 900, x=b[0], y=b[1]), "zh")
    assert any("但玩家和身边的2名队友没有向安全位置后撤" in s.description for s in zh.steps)
    for slot in (0, 2):
        match.players[slot].position_log = stand(RADIANT_BASE, 0, 1200)
    match = make_match(match.players, kills_log=match.kills_log)
    monkeypatch.setattr(causal_chain, "_WARNING_GAP", 0)
    assert len(warnings()) >= 2                        # the scenario does escalate twice


def test_danger_step_needs_enemies_that_are_really_hidden(monkeypatch):
    """A single enemy that left vision seconds ago is not listed among the danger reasons, and
    the step then read "became dangerous (far from towers; no ward) ... move back as soon as
    nearby enemies drop out of vision"."""
    b = ENEMY_HALF[1]

    def warnings(hidden_slots):               # at night, so one hidden enemy already means "danger"
        logs = {1: stand(b, 0, 1095) + stand(b, 1100, 1160, hp=0)}
        for i, slot in enumerate(hidden_slots):
            logs[slot] = (stand((b[0] + 600, b[1] + 100 * i), 0, 1075)       # gone for 20 s at most
                          + stand((b[0] + 2600, b[1] + 100 * i), 1080, 1200))
        match = logged_match(logs, kills_log=[kill(1100.0, 1, 129, x=b[0], y=b[1])])
        chain = _chain(match, death_of(1, 1100, x=b[0], y=b[1]))
        return [s for s in chain.steps if s.step_type == "ignore_warning"]

    assert warnings((128,)) == []
    two = warnings((128, 130))
    assert len(two) == 1 and "2 enemy heroes hidden in the fog" in two[0].description
    monkeypatch.setattr(causal_chain, "_LONE_ENEMY_HIDDEN_S", -1)
    assert warnings((128,)) != []             # the scenario is one the old rule fired on


def test_death_before_the_lanes_start_is_dated_before_the_death():
    match = logged_match({}, start=-85, kills_log=[kill(-10.3, 1, 129, x=SPOT[0], y=SPOT[1])])
    chain = _chain(match, death_of(1, -10))
    assert [s.time for s in chain.steps] == [-15, -10]
    assert chain.root_cause_type == "rune_contest_unfavorable"
    assert chain.steps[-1].description == "Puck was killed by Lina in that fight."
    # The summary states the fact; whether the fight was a bad one is the evaluator's call
    # (it said "in an unfavorable situation" next to "the numbers were not unfavorable").
    assert "unfavorable" not in chain.root_cause_summary and "level 1" in chain.root_cause_summary
    assert "不利" not in _chain(match, death_of(1, -10), "zh").root_cause_summary


def test_unit_killer_is_named_in_the_report_language():
    death = death_of(1, 900, killer=None)
    death.killer_unit = "npc_dota_goodguys_tower1_mid"
    match = positionless_match(kills_log=[kill(900.0, 1, None, killer_unit=death.killer_unit)])
    # Written like the analysis ("was killed by a tower"), not as a label.
    assert _chain(match, death, "en").steps[-1].description.startswith("Killed by a tower")
    assert "Killer: Tower" in _chain(match, death, "en").formatted_text
    ctx = context_builder.build_context(match, death_point(death), lang="en")
    assert [e.description for e in ctx.event_timeline if e.event_type == "death"] == ["Puck died to a tower"]
    assert _chain(match, death, "zh").steps[-1].description.startswith("被防御塔击杀")


def test_denied_death_is_not_reported_as_an_enemy_kill():
    death = death_of(1, 900, killer=0)                  # Juggernaut is an ally
    death.denied = True
    match = positionless_match(kills_log=[kill(900.0, 1, 0, deny=True)])
    chain = _chain(match, death)
    assert chain.steps[-1].description == "Denied by ally Juggernaut at Radiant jungle (bottom side)"
    assert "Killer:" not in chain.formatted_text
    assert _chain(match, death, "zh").steps[-1].description.startswith("被队友主宰反补")
    ctx = context_builder.build_context(match, death_point(death), lang="en")
    assert ctx.enemy_states == []                       # nobody is moved next to the victim
    kills = [e.description for e in ctx.event_timeline if e.event_type in ("kill", "death")]
    assert kills == ["Puck was denied by Juggernaut"]
