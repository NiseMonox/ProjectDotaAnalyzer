"""Game values of the vision model (vision.py): where they come from and what they do.

Ward ranges / lifetimes and the Smoke of Deceit values are read from the game
data (``data_manager.game_rules``), hero vision per hero
(``data_manager.hero_vision``); the built-in 7.41f values only stand in when
there is no data. Towers see 800 at night on tier 1 and 1100 on tiers 2-4,
the ancient and the fountain see their base, and a hero under Smoke of Deceit
is hidden from wards. A value that changed with a patch follows the match's
own patch when the old value is known (sentry true sight: 1000 before 7.38).
"""
from __future__ import annotations

import pytest

from backend.app import geo
from backend.app.ai import context_builder, data_manager, vision
from backend.app.models import SmokeEvent, WardEvent
from test_ai_pipeline_helpers import CJK_RE, default_players, make_match, observer

# Far from every tower: only heroes and wards give vision here.
SPOT = (1500.0, -3500.0)
RADIANT_BASE = (-6500.0, -6000.0)
DIRE_BASE = (6500.0, 6000.0)
NIGHT, DAY = 1000, 1300         # 15:00-20:00 is night, 20:00-25:00 is day

RULES_7_41F = {
    "observer_ward_duration": 360, "observer_ward_vision": 1600, "sentry_ward_duration": 420,
    "sentry_ward_true_sight": 1050, "smoke_duration": 45, "smoke_application_radius": 1200,
    "smoke_break_radius": 1025, "gem_true_sight": 900, "dust_radius": 1050, "dust_duration": 12,
}


def stand(pos, start: int, end: int, hp: int = 100) -> list[list[int]]:
    return [[t, int(pos[0]), int(pos[1]), hp] for t in range(start, end + 1, 5)]


def logged_match(logs: dict[int, list[list[int]]], start: int = 500, end: int = 1500, **kw):
    """Juggernaut (slot 0) stands at SPOT, everybody else at home, unless ``logs`` says otherwise."""
    players = default_players()
    for p in players:
        p.laning_snapshots = []
        home = RADIANT_BASE if p.team == "radiant" else DIRE_BASE
        p.position_log = stand(SPOT if p.player_slot == 0 else home, start, end)
        if p.player_slot in logs:
            p.position_log = logs[p.player_slot]
    return make_match(players, **kw)


def state_of(match, hero: str, t: int, team: str = "radiant"):
    slot = 0 if team == "radiant" else 128
    return next(e for e in vision._track_enemy_visibility(match, team, slot, t) if e.hero_name == hero)


@pytest.fixture
def rules(monkeypatch):
    """Pin the game data: 7.41f values, default hero vision; returns the rules dict to edit."""
    values = dict(RULES_7_41F)
    monkeypatch.setattr(data_manager, "game_rules", lambda: values)
    monkeypatch.setattr(data_manager, "hero_vision", lambda hero: data_manager.DEFAULT_HERO_VISION)
    monkeypatch.setattr(data_manager, "patch_label_for_time", lambda t: "")
    monkeypatch.setattr(data_manager, "patch_for_time", lambda t: "")
    return values


# ---------------------------------------------------------------------------
# Where the values come from
# ---------------------------------------------------------------------------

def test_built_in_values_are_the_current_game():
    """The offline fallback is 7.41f: sentry true sight 1050 (it was 1000), smoke 45 s."""
    assert vision.DEFAULT_RULES == vision.VisionRules(
        observer_vision=1600, observer_duration=360, sentry_true_sight=1050, sentry_duration=420,
        smoke_duration=45, smoke_break_radius=1025)
    assert (vision.HERO_DAY_VISION, vision.HERO_NIGHT_VISION) == data_manager.DEFAULT_HERO_VISION == (1800, 800)
    assert [vision.tower_vision(key, night=True) for key in ("mid_t1", "mid_t2", "bot_t3", "ancient_top")] == [
        800, 1100, 1100, 1100]
    assert {vision.tower_vision(key, night=False) for key in ("mid_t1", "mid_t2", "ancient_bot")} == {1900}
    assert (vision.ANCIENT_VISION, vision.FOUNTAIN_VISION) == (2600, 1800)


def test_rules_are_read_from_the_game_data(rules):
    assert vision.vision_rules() == vision.DEFAULT_RULES
    rules.update(observer_ward_vision=1400, observer_ward_duration=300, sentry_ward_true_sight=1100,
                 sentry_ward_duration=480, smoke_duration=40, smoke_break_radius=900)
    assert vision.vision_rules() == vision.VisionRules(1400, 300, 1100, 480, 40, 900)


@pytest.mark.parametrize("broken", [{}, {"observer_ward_vision": None, "smoke_duration": "45", "sentry_ward_duration": 0,
                                         "sentry_ward_true_sight": True}])
def test_missing_or_unusable_data_falls_back_per_value(monkeypatch, rules, broken):
    monkeypatch.setattr(data_manager, "game_rules", lambda: broken)
    assert vision.vision_rules() == vision.DEFAULT_RULES

    def boom():
        raise RuntimeError("no game data")

    monkeypatch.setattr(data_manager, "game_rules", boom)
    assert vision.vision_rules() == vision.DEFAULT_RULES


def test_offline_values_of_both_layers_agree(monkeypatch, tmp_path):
    """Without fetched game data the data layer and the vision model must not disagree."""
    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", tmp_path)
    data_manager.invalidate_all_caches()
    try:
        offline = data_manager.game_rules()
        assert vision.vision_rules() == vision.DEFAULT_RULES
        for field, source in vision._RULE_NAMES.items():
            assert getattr(vision.DEFAULT_RULES, field) == offline[source], field
        # ...and the committed hero catalog knows the heroes with their own vision.
        assert data_manager.hero_vision("Sniper") == (1800, 1400)
    finally:
        monkeypatch.undo()
        data_manager.invalidate_all_caches()


@pytest.mark.parametrize("patch, start_label, expected", [
    ("7.37e", "", 1000),      # before 7.38 a sentry revealed 1000 units
    ("7.33", "", 1000),
    ("7.38", "", 1050),
    ("7.41b", "", 1050),
    ("", "7.37d", 1000),      # a replay has no patch field: the stored patch table dates it
    ("", "7.40c", 1050),
    ("", "", 1050),           # unknown: today's rule, no claim about the past
    ("garbage", "", 1050),
])
def test_sentry_range_follows_the_patch_of_the_match(monkeypatch, rules, patch, start_label, expected):
    monkeypatch.setattr(data_manager, "patch_label_for_time", lambda t: start_label)
    match = make_match(patch=patch, start_time=1_700_000_000)
    assert vision.vision_rules(match).sentry_true_sight == expected
    # Nothing else is back-dated: only the sentry range has a known old value.
    assert vision.vision_rules(match)._replace(sentry_true_sight=1050) == vision.DEFAULT_RULES
    wards = [WardEvent(ward_type="sentry", x=100.0, y=200.0, time=600, player_slot=4),
             observer(300.0, 400.0, slot=4, time=600)]
    match = make_match(patch=patch, start_time=1_700_000_000, wards=wards)
    points = context_builder._build_map_state(match, 700)["vision_points"]
    assert [(p["radius"], p["team"]) for p in points] == [(expected, "radiant"), (1600, "radiant")]


def test_ward_lifetimes_come_from_the_rules(monkeypatch, rules):
    ward = observer(SPOT[0], SPOT[1], slot=4, time=600)
    sentry = WardEvent(ward_type="sentry", x=0.0, y=0.0, time=600, player_slot=4)
    match = logged_match({}, wards=[ward, sentry])
    assert vision._ward_expiry_time(ward) == 960 and vision._ward_expiry_time(sentry) == 1020
    assert vision._get_active_wards(match, 959) == [ward, sentry] and vision._get_active_wards(match, 1019) == [sentry]
    # A game-data update replaces the values: the per-match index must not keep the old ones.
    rules.update(observer_ward_duration=300, sentry_ward_duration=360)
    assert vision._get_active_wards(match, 905) == [ward, sentry]          # same data generation: still cached
    generation = vision._data_generation()
    monkeypatch.setattr(vision, "_data_generation", lambda: generation + 1)
    assert vision._get_active_wards(match, 905) == [sentry]
    assert [w for _i, w in vision._active_observers(match, "radiant", 899)] == [ward]
    assert vision._active_observers(match, "radiant", 900) == []


# ---------------------------------------------------------------------------
# What the values do
# ---------------------------------------------------------------------------

def test_observer_range_decides_what_a_ward_sees(rules):
    ward = observer(SPOT[0] + 4000, SPOT[1], slot=4, time=600)
    enemy_at = (SPOT[0] + 4000 + 1500, SPOT[1])                  # 1500 from the ward, 5500 from Juggernaut
    match = logged_match({128: stand(enemy_at, 500, 1500)}, wards=[ward])
    assert state_of(match, "Phantom Assassin", 800).state == "visible"
    assert state_of(match, "Phantom Assassin", 800).last_seen_source == "ward"
    rules["observer_ward_vision"] = 1400
    match = logged_match({128: stand(enemy_at, 500, 1500)}, wards=[ward])
    assert state_of(match, "Phantom Assassin", 800).state == "fog"


def test_tower_night_vision_depends_on_the_tier(rules):
    t1, t2 = geo.TOWER_POSITIONS["radiant"]["mid_t1"], geo.TOWER_POSITIONS["radiant"]["mid_t2"]
    # 1000 units from the tower, away from every hero.
    match = logged_match({0: stand(RADIANT_BASE, 500, 1500),
                          128: stand((t1[0] + 1000, t1[1]), 500, 1500),
                          129: stand((t2[0] - 1000, t2[1]), 500, 1500)})
    assert state_of(match, "Phantom Assassin", NIGHT).state == "fog"          # tier 1: 800 at night
    assert state_of(match, "Lina", NIGHT).state == "visible"                  # tier 2: 1100 at night
    assert state_of(match, "Lina", NIGHT).last_seen_source == "tower"
    assert state_of(match, "Phantom Assassin", DAY).state == "visible"        # 1900 by day for every tower


def test_hero_vision_is_per_hero(monkeypatch, rules):
    """Night Stalker sees 1800 at night and 800 by day; Sniper 1400 at night."""
    table = {"Juggernaut": (800, 1800)}
    monkeypatch.setattr(data_manager, "hero_vision", lambda hero: table.get(hero, (1800, 800)))
    match = logged_match({128: stand((SPOT[0] + 1500, SPOT[1]), 500, 1500)})
    assert state_of(match, "Phantom Assassin", NIGHT).state == "visible"   # 1500 < 1800 at night
    assert state_of(match, "Phantom Assassin", DAY).state == "fog"               # 1500 > 800 by day
    table["Juggernaut"] = (1800, 1400)
    match = logged_match({128: stand((SPOT[0] + 1500, SPOT[1]), 500, 1500)})
    assert state_of(match, "Phantom Assassin", NIGHT).state == "fog"       # 1500 > 1400
    assert state_of(match, "Phantom Assassin", DAY).state == "visible"
    # Data without a usable value: the default.
    monkeypatch.setattr(data_manager, "hero_vision", lambda hero: (0, 0))
    match = logged_match({128: stand((SPOT[0] + 1500, SPOT[1]), 500, 1500)})
    assert state_of(match, "Phantom Assassin", DAY).state == "visible"
    assert state_of(match, "Phantom Assassin", NIGHT).state == "fog"


def test_real_game_data_knows_the_night_stalker():
    assert data_manager.hero_vision("Night Stalker") == (800, 1800)
    assert data_manager.hero_vision("Axe") == (1800, 800)


def test_ancient_and_fountain_see_their_base(rules):
    """Enemies on the high ground used to be 'hidden in the fog' once the tier-3 towers were gone."""
    ancient = geo.ANCIENT_POSITIONS["radiant"]
    towers = [{"time": 100, "type": "building_kill", "key": f"npc_dota_goodguys_tower{n}_{lane}", "team": "dire"}
              for lane in ("top", "mid", "bot") for n in (1, 2, 3)]
    towers += [{"time": 100, "type": "building_kill", "key": "npc_dota_goodguys_tower4", "team": "dire"}] * 2
    inside, outside = (ancient[0] + 2500, ancient[1]), (ancient[0] + 2700, ancient[1] + 2700)
    match = logged_match({0: stand(SPOT, 500, 1500), 1: stand(SPOT, 500, 1500), 2: stand(SPOT, 500, 1500),
                          3: stand(SPOT, 500, 1500), 4: stand(SPOT, 500, 1500),
                          128: stand(inside, 500, 1500), 129: stand(outside, 500, 1500)}, objectives=towers)
    assert vision._get_alive_towers(match, "radiant", NIGHT) == []
    for t in (NIGHT, DAY):
        assert state_of(match, "Phantom Assassin", t).state == "visible"
        assert state_of(match, "Lina", t).state == "fog"


# ---------------------------------------------------------------------------
# Smoke of Deceit
# ---------------------------------------------------------------------------

def _smoke_match(enemy_pos, smoke_end: int, **kw):
    ward = observer(enemy_pos[0] + 300, enemy_pos[1], slot=4, time=1000)
    smoke = SmokeEvent(time=1200, team="dire", player_slots=[128, 129], player_slot=128,
                       x=enemy_pos[0], y=enemy_pos[1], end_time=smoke_end)
    return logged_match({128: stand(enemy_pos, 500, 1500), 129: stand(enemy_pos, 500, 1500)},
                        wards=[ward], smokes=[smoke], **kw)


def test_smoked_heroes_are_hidden_from_wards(rules):
    """A smoke gank walked through a ward and was reported as 'visible' all the way."""
    far = (SPOT[0] + 5000, SPOT[1])
    match = _smoke_match(far, smoke_end=1230)
    assert state_of(match, "Phantom Assassin", 1195).state == "visible"        # on the ward before the smoke
    assert state_of(match, "Phantom Assassin", 1195).last_seen_source == "ward"
    for t in (1200, 1215, 1229):
        assert state_of(match, "Phantom Assassin", t).state == "fog", t
        assert state_of(match, "Lina", t).state == "fog", t
        assert vision.is_smoked(match, 128, t)
    assert state_of(match, "Phantom Assassin", 1230).state == "visible"        # the smoke broke
    assert not vision.is_smoked(match, 128, 1230) and not vision.is_smoked(match, 130, 1215)
    # The fog run is measured from the moment the ward lost them.
    hidden = state_of(match, "Phantom Assassin", 1225)
    assert hidden.seconds_missing == 30 and hidden.last_seen_source == "ward"
    # The prompt text says why the ward does not help, in both languages.
    en = vision.analyze_vision(match, 0, 1225, SPOT, lang="en").formatted_text
    assert "- Phantom Assassin: in the fog for 30s" in en and "under Smoke of Deceit at this moment" in en
    zh = vision.analyze_vision(match, 0, 1225, SPOT, lang="zh").formatted_text
    assert "处于诡计之雾中（眼位看不到" in zh and not CJK_RE.search(en)
    after = vision.analyze_vision(match, 0, 1235, SPOT, lang="en").formatted_text
    assert "Smoke of Deceit" not in after


def test_a_smoke_lasts_at_most_its_duration_and_needs_an_end(rules):
    far = (SPOT[0] + 5000, SPOT[1])
    endless = _smoke_match(far, smoke_end=1400)            # an end later than the smoke can last
    assert state_of(endless, "Phantom Assassin", 1244).state == "fog"
    assert state_of(endless, "Phantom Assassin", 1245).state == "visible"      # 45 s after the activation
    rules["smoke_duration"] = 20
    shorter = _smoke_match(far, smoke_end=1400)
    assert state_of(shorter, "Phantom Assassin", 1220).state == "visible"
    # Older data has no end time (it equals the start): nobody is hidden on a guess.
    unknown = _smoke_match(far, smoke_end=1200)
    assert state_of(unknown, "Phantom Assassin", 1210).state == "visible"


def test_smoke_breaks_next_to_a_hero_or_tower(rules):
    near = (SPOT[0] + 1000, SPOT[1])          # within 1025 of Juggernaut: the smoke could not hold
    assert state_of(_smoke_match(near, smoke_end=1230), "Phantom Assassin", 1215).state == "visible"
    assert state_of(_smoke_match(near, smoke_end=1230), "Phantom Assassin", 1215).last_seen_source == "hero"
    edge = (SPOT[0] + 1500, SPOT[1])          # inside day vision (1800), outside the break range
    assert state_of(_smoke_match(edge, smoke_end=1230), "Phantom Assassin", 1215).state == "fog"
    rules["smoke_break_radius"] = 1600
    assert state_of(_smoke_match(edge, smoke_end=1230), "Phantom Assassin", 1215).state == "visible"
    # Allied smoke does not blind the own team.
    match = _smoke_match((SPOT[0] + 5000, SPOT[1]), smoke_end=1230)
    assert state_of(match, "Juggernaut", 1215, team="dire").state in ("fog", "visible")
    assert not vision.is_smoked(match, 0, 1215)
