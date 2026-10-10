"""The data contract every parser backend fills (see the ``models`` module docstring).

``check_contract`` holds the invariants: kill log vs scoreboard, position log
sampling and life state, identical key formats, wards, per-minute series. It
runs on

- ``fixtures/clarity_v3_8762775364_trimmed.json``: the real analyzer output
  (schema 3) of match 8762775364, cut down and anonymised. Two heroes keep
  their whole position log, the others the samples around their deaths.
- ``fixtures/opendota_8762775364.json``: OpenDota's response for the same match.
- the mock match.

The Clarity fixture is also compared with the OpenDota one field by field:
both describe the same game, so a regression in either transform (or a jar
that stops matching Valve's numbers) shows up here.
"""
from __future__ import annotations

import copy
import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import pytest  # noqa: E402

from backend.app import geo, parser  # noqa: E402
from backend.app.mock_data import generate_mock_match  # noqa: E402
from backend.app.models import MatchData  # noqa: E402
from backend.app.parser import ClarityParser, OpenDotaParser  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
TOOLS = Path(__file__).resolve().parents[2] / "tools"
MATCH_ID = 8762775364

# Heroes walk a little outside the playable square geo.py maps to 0-100 %.
_WORLD_MARGIN = 600


def _team(slot: int) -> str:
    return "radiant" if slot < 128 else "dire"


def _in_world(x: float, y: float) -> bool:
    return all(geo.MAP_WORLD_MIN - _WORLD_MARGIN <= v <= geo.MAP_WORLD_MAX + _WORLD_MARGIN for v in (x, y))


def _is_item_key(key: str) -> bool:
    # canonical: the OpenDota key; "item_<id>" only for an id nothing resolves
    return bool(key) and (parser.canonical_item_key(key) == key or re.fullmatch(r"item_\d+", key) is not None)


def _is_unit_name(unit: str, source: str) -> bool:
    """A non-hero key of ``damage`` / ``damage_taken``: the unit's own name."""
    if unit.startswith("npc_dota_"):
        # the prefix once, and never in front of a name that is complete without it
        return "npc_dota_" not in unit[len("npc_dota_"):] and not unit.startswith("npc_dota_dota_")
    # OpenDota also has the units the game names outside npc_dota_* ("dota_fountain")
    # and marks illusions ("illusion_npc_dota_hero_viper"); the analyzer and the mock do not
    return source == "opendota" and re.fullmatch(r"[a-z0-9_]+", unit) is not None


def check_contract(match: MatchData, source: str) -> None:
    """Assert the cross-source contract on ``match`` (source: clarity / opendota / mock)."""
    exact = source in ("clarity", "mock")      # the kill log is the scoreboard; positions are known
    names = {p.player_slot: p.hero_name for p in match.players}
    slots = set(names)
    minutes = match.duration // 60 + 1
    assert len(match.players) == 10 and len(set(names.values())) == 10
    assert isinstance(match.winner_known, bool) and isinstance(match.match_id, int)

    # ---- kill log
    kills = match.kills_log
    assert kills and [k.time for k in kills] == sorted(k.time for k in kills)
    for k in kills:
        assert k.victim_slot in slots and (k.killer_slot is None or k.killer_slot in slots), k
        assert set(k.assist_slots) <= slots and k.killer_slot not in k.assist_slots, k
        if k.killer_slot is not None and k.killer_slot != k.victim_slot:
            assert (_team(k.killer_slot) == _team(k.victim_slot)) == k.deny, k
        assert not k.killer_unit.startswith("npc_dota_hero_") or k.killer_slot is None, k
        # the side of the unit that landed the kill: the victim's own for a deny, else the
        # credited hero's (or a neutral's, when the scoreboard credits a kill a neutral landed)
        if k.deny:
            assert k.killer_team in (None, _team(k.victim_slot)), k
        elif k.killer_slot is not None:
            assert k.killer_team in (_team(k.killer_slot), "neutral") or (source == "clarity" and k.killer_team is None), k
        else:
            assert k.killer_team in (None, "radiant", "dire", "neutral"), k
        assert (k.x is None) == (k.y is None) and (k.x is None or _in_world(k.x, k.y)), k
        if exact:
            assert k.confirmed and k.x is not None, k
    if exact:
        assert match.radiant_score == sum(1 for k in kills if not k.deny and k.victim_slot >= 128)
        assert match.dire_score == sum(1 for k in kills if not k.deny and k.victim_slot < 128)
    for p in match.players:
        slot = p.player_slot
        deaths = [k for k in kills if k.victim_slot == slot]
        own = [k for k in kills if k.killer_slot == slot and not k.deny]
        assert p.team == _team(slot) and set(p.killed) | set(p.killed_by) <= set(names.values()), p.hero_name
        assert p.hero_name not in p.damage and p.hero_name not in p.damage_taken, p.hero_name
        for unit in list(p.damage) + list(p.damage_taken):
            assert unit in names.values() or _is_unit_name(unit, source), (p.hero_name, unit)
        assert not {"", "null", "dota_unknown"} & set(p.damage_inflictor), p.hero_name
        assert 0 <= p.teamfight_participation <= 1 and p.buyback_count == len(p.buyback_log), p.hero_name
        if exact:
            assert (len(own), len(deaths)) == (p.kills, p.deaths), p.hero_name
            assert sum(1 for k in kills if slot in k.assist_slots and not k.deny) == p.assists, p.hero_name
            assert p.killed == dict(Counter(names[k.victim_slot] for k in own)), p.hero_name
            assert p.killed_by == dict(Counter(
                names[k.killer_slot] for k in deaths if k.killer_slot is not None and not k.deny)), p.hero_name
            # the game's multi-kill / streak announcements follow from the kill log
            assert (p.multi_kills, p.kill_streaks) == parser.kill_announcements(kills, slot), p.hero_name
        else:
            assert sum(1 for k in deaths if k.confirmed) <= p.deaths, p.hero_name
            assert sum(p.killed.values()) <= p.kills and sum(p.killed_by.values()) <= p.deaths, p.hero_name

    # ---- position log: [time_s, x, y, hp_pct] every 5 s; hp 0 while dead
    for p in match.players:
        log = p.position_log
        if source == "opendota":
            assert log == []
            continue
        assert log and all(len(e) == 4 and all(type(v) is int for v in e) for e in log), p.hero_name
        times = [e[0] for e in log]
        assert times == sorted(set(times)) and all(t % 5 == 0 for t in times), p.hero_name
        assert times[0] <= 0 and times[-1] == match.duration // 5 * 5, (p.hero_name, times[0], times[-1])
        assert all(-1 <= hp <= 100 and _in_world(x, y) for _t, x, y, hp in log), p.hero_name
        by_time = {e[0]: e for e in log}
        for k in kills:
            if k.victim_slot != p.player_slot:
                continue
            after = int(k.time // 5) * 5 + 5       # first sample after the death
            bought_back = any(k.time <= b["time"] <= after for b in p.buyback_log)
            if after in by_time and not bought_back:
                assert by_time[after][3] == 0, (p.hero_name, k.time, by_time[after])

    # ---- keys and names
    for p in match.players:
        keys = [it.item_key for it in p.items] + p.final_item_keys + p.backpack
        assert all(_is_item_key(key) for key in keys + ([p.neutral_item] if p.neutral_item else [])), p.hero_name
        assert "ward_dispenser" not in [it.item_key for it in p.items], p.hero_name
        assert [(e["time"], e["key"]) for e in p.purchase_log] == [(it.time, it.item_key) for it in p.items]
        assert [it.time for it in p.items] == sorted(it.time for it in p.items), p.hero_name
        for skill in p.skills:
            assert skill.ability_key and skill.ability_key == skill.ability_key.lower() and skill.level >= 1
            assert not skill.ability_key.startswith("talent_"), skill
            # a display name: never a template ("+{s:bonus_damage} ...") or the raw talent key
            assert skill.ability_name and "{" not in skill.ability_name, skill
            assert not skill.ability_name.startswith("special_bonus_"), skill
            if skill.ability_key.startswith("special_bonus_"):
                assert skill.ability_name == parser.ability_display_name(skill.ability_key), skill
        assert len(p.gold_t) == len(p.xp_t) == len(p.lh_t) == len(p.dn_t) == minutes, p.hero_name
        assert all(e.inflictor for e in p.laning_damage_events), p.hero_name
    assert len(match.radiant_gold_adv) == len(match.radiant_xp_adv) == len(match.gold_xp_timeline) == minutes

    # ---- teamfights
    for tf in match.teamfights:
        assert tf.start_time < tf.end_time and len(tf.players) == 10
        assert tf.radiant_deaths == sum(tp.deaths for tp in tf.players if tp.player_slot < 128)
        assert tf.dire_deaths == sum(tp.deaths for tp in tf.players if tp.player_slot >= 128)
        for attr in ("gold_delta", "xp_delta"):
            signed = sum(getattr(tp, attr) * (1 if tp.player_slot < 128 else -1) for tp in tf.players)
            assert getattr(tf, attr) == signed, (tf.start_time, attr)
        for tp in tf.players:
            where = (tf.start_time, tp.hero_name)
            assert tp.hero_name == names[tp.player_slot], where
            assert tp.kills == len(tp.kill_targets) and (tp.deaths > 0) == (tp.death_time_offset is not None), where
            assert tp.death_time_offset is None or 0 <= tp.death_time_offset <= tf.end_time - tf.start_time, where
            for target in tp.kill_targets:
                assert set(target) == {"target", "target_slot", "time_offset"}, where
                assert target["target"] == names[target["target_slot"]], where
                assert _team(target["target_slot"]) != _team(tp.player_slot), where
                assert 0 <= target["time_offset"] <= tf.end_time - tf.start_time, where
            assert all(_is_item_key(key) for key in tp.item_uses), where

    # ---- wards
    assert [w.time for w in match.wards] == sorted(w.time for w in match.wards)
    for w in match.wards:
        assert w.ward_type in ("observer", "sentry") and w.team in ("radiant", "dire") and _in_world(w.x, w.y), w
        assert w.player_slot == -1 or (w.player_slot in slots and _team(w.player_slot) == w.team), w
        assert w.destroyed_time is None or w.time <= w.destroyed_time <= match.duration, w
        assert not w.dewarded or w.destroyed_time is not None, w
        if w.killer_slot is not None:
            assert w.dewarded and w.killer_slot in slots and _team(w.killer_slot) != w.team, w
    for p in match.players:
        own = [w for w in match.wards if w.player_slot == p.player_slot]
        assert p.wards == own, p.hero_name
        assert (p.obs_placed, p.sen_placed) == (
            sum(w.ward_type == "observer" for w in own), sum(w.ward_type == "sentry" for w in own)), p.hero_name

    # ---- smokes, objectives, chat, draft
    for s in match.smokes:
        assert s.team in ("radiant", "dire") and set(s.player_slots) <= slots, s
        assert s.player_slot is None or (s.player_slot in slots and _team(s.player_slot) == s.team), s
    assert [o["time"] for o in match.objectives] == sorted(o["time"] for o in match.objectives)
    for o in match.objectives:
        assert o.get("team") in (None, "radiant", "dire"), o
        assert o.get("player_slot") is None or o["player_slot"] in slots, o
        if o["type"] == "building_kill":
            assert o["key"].startswith("npc_dota_") and {o["team"], o["building_team"]} == {"radiant", "dire"}, o
        elif o["type"] == "CHAT_MESSAGE_FIRSTBLOOD":
            assert int(o["key"]) in slots and _team(int(o["key"])) != _team(o["player_slot"]), o
    for message in match.chat:
        assert message["type"] != "chatwheel" or message["key"].startswith("chatwheel_"), message
    picks = [d for d in match.picks_bans if d["is_pick"]]
    assert sorted(d["hero_id"] for d in picks) == sorted(p.hero_id for p in match.players)
    for d in match.picks_bans:
        assert d["hero_id"] > 0 and (d["team"] in (0, 1) or (d["team"] is None and not d["is_pick"])), d
    by_hero_id = {p.hero_id: p for p in match.players}
    assert all(d["team"] == (0 if by_hero_id[d["hero_id"]].team == "radiant" else 1) for d in picks)


# ---------------------------------------------------------------------------
# The three sources
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def committed_catalogs_only(tmp_path_factory):
    """Talent names come from the data layer: pin it to the committed catalogs,
    whatever game data this machine has fetched."""
    from backend.app.ai import data_manager

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(data_manager, "GAMEDATA_DIR", tmp_path_factory.mktemp("gamedata"))
        data_manager.invalidate_all_caches()
        yield
    data_manager.invalidate_all_caches()


@pytest.fixture(scope="module")
def clarity_raw():
    return json.loads((FIXTURES / f"clarity_v3_{MATCH_ID}_trimmed.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def clarity(clarity_raw):
    return ClarityParser()._transform_data(copy.deepcopy(clarity_raw))


@pytest.fixture(scope="module")
def opendota_raw():
    return json.loads((FIXTURES / f"opendota_{MATCH_ID}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def opendota(opendota_raw):
    constants = json.loads((FIXTURES / "opendota_constants.json").read_text(encoding="utf-8"))
    return OpenDotaParser()._transform(copy.deepcopy(opendota_raw), parser.load_hero_map(), parser._item_id_map(),
                                       constants["ability_ids"], constants["item_ids"], constants["patch"])


def test_clarity_fixture_is_anonymised_schema3(clarity_raw):
    assert clarity_raw["schema_version"] == 3 and clarity_raw["match_id_source"] == "file_info"
    # written by the clarity 4 analyzer: the replay's post-game record and the team counters were read
    assert clarity_raw["match_record"] is True and clarity_raw["score_source"] == "scoreboard"
    assert {p["healing_source"] for p in clarity_raw["players"]} == {"match_record"}
    assert clarity_raw["parse_errors"] == {}
    assert [p["player_name"] for p in clarity_raw["players"]] == [f"player{i}" for i in range(10)]
    assert all(p["steam_id"] == 0 for p in clarity_raw["players"])
    assert {c["key"] for c in clarity_raw["chat"] if c["type"] == "chat"} == {"gg"}


def test_fixture_jar_and_sources_are_one_analyzer_version(clarity_raw):
    """``analyzer_version`` tells a stale JAR or a stale output from a current one: the
    sources (build.gradle), the committed JAR (its manifest) and the fixture it wrote agree."""
    gradle = (TOOLS / "clarity-analyzer" / "build.gradle").read_text(encoding="utf-8")
    version = re.search(r"^version = '(\d+\.\d+\.\d+)'$", gradle, re.MULTILINE).group(1)
    assert clarity_raw["analyzer_version"] == version
    with zipfile.ZipFile(TOOLS / "clarity-analyzer.jar") as jar:
        manifest = jar.read("META-INF/MANIFEST.MF").decode("utf-8").splitlines()
    assert f"Implementation-Version: {version}" in manifest
    # written by 1.4.0 or later: every death names the side of the unit that landed the kill
    assert {k["killer_team"] for k in clarity_raw["kills_log"]} == {"radiant", "dire"}


def test_clarity_satisfies_the_contract(clarity):
    check_contract(clarity, "clarity")


def test_opendota_satisfies_the_contract(opendota):
    check_contract(opendota, "opendota")


def test_mock_satisfies_the_contract():
    mock = generate_mock_match()
    check_contract(mock, "mock")
    # the mock fills what real data has, and every position log is complete
    assert all(len(p.position_log) == (mock.duration // 5 * 5 + 90) // 5 + 1 for p in mock.players)
    assert all(s.player_slot in s.player_slots for s in mock.smokes)
    assert any(w.killer_slot is not None for w in mock.wards)
    assert all((w.killer_slot is not None) == w.dewarded for w in mock.wards)
    assert any(p.multi_kills for p in mock.players) and any(p.kill_streaks for p in mock.players)
    assert any(tf.xp_delta for tf in mock.teamfights)


def test_talent_names_are_clean_and_the_same_from_every_source(clarity, opendota):
    """OpenDota's constants name most talents with a template ("+{s:bonus_damage} Shukuchi
    Damage"); the committed catalog (gen_static_catalogs) has the values filled in."""
    names: dict[str, str] = {}
    for source, match in (("clarity", clarity), ("opendota", opendota), ("mock", generate_mock_match())):
        talents = [(s.ability_key, s.ability_name) for p in match.players for s in p.skills
                   if s.ability_key.startswith("special_bonus_")]
        assert len(talents) > 15, source
        for key, name in talents:
            assert names.setdefault(key, name) == name, (source, key)     # one name per key
        casts = [c.ability_name for tf in match.teamfights for tp in tf.players for c in tp.ability_casts]
        assert not [n for n in casts + [e.ability_name for e in match.ability_events] if "{" in n], source
    by_key = {s.ability_key: s.ability_name for p in clarity.players for s in p.skills}
    assert by_key["special_bonus_unique_weaver_1"] == "+90 Shukuchi Damage"
    assert by_key["special_bonus_unique_lion_5"] == "-2.0s Hex Cooldown"
    assert by_key["special_bonus_unique_tidehunter_5"] == "+20% Gush Slow"
    assert by_key["special_bonus_hp_300"] == "+300 Health"                       # had no template
    assert by_key["special_bonus_unique_wraith_king_facet_1"] == "+2s Wraithfire Blast Slow Duration"
    # a talent no catalog knows is named after its key until game data is fetched
    assert parser.ability_display_name("special_bonus_unique_lion_99") == "Lion 99"
    shared = {s.ability_key for p in opendota.players for s in p.skills} & set(by_key)
    assert len([k for k in shared if k.startswith("special_bonus_")]) >= 20
    assert parser.ability_display_name("special_bonus_attributes") == "Attribute Bonus"
    assert parser.ability_display_name("antimage_mana_break") == "Mana Break"
    assert parser.ability_display_name("not_in_any_catalog") == "not_in_any_catalog"
    # 7.41 additions and the map actions the game gives no name
    assert parser.ability_display_name("dragon_knight_wyrms_wrath") == "Wyrm's Wrath"
    assert parser.ability_display_name("ability_capture") == "Capture Outpost"
    assert parser.item_display_name("crellas_crozier") == "Crella's Crozier"
    assert parser.item_display_name("recipe_abyssal_blade") == "Abyssal Blade Recipe"
    assert parser.item_display_name("not_in_any_catalog") == "Not In Any Catalog"


def test_every_source_records_the_game_data_patch_behind_its_talent_names(clarity, opendota, clarity_raw,
                                                                         opendota_raw, monkeypatch):
    # this module runs on the committed catalogs only: no fetched talent table, no patch to record
    assert [m.talent_names_patch for m in (clarity, opendota, generate_mock_match())] == ["", "", ""]
    monkeypatch.setattr(parser, "talent_names_patch", lambda: "7.41")
    constants = json.loads((FIXTURES / "opendota_constants.json").read_text(encoding="utf-8"))
    matches = [
        ClarityParser()._transform_data(copy.deepcopy(clarity_raw)),
        OpenDotaParser()._transform(copy.deepcopy(opendota_raw), parser.load_hero_map(), parser._item_id_map(),
                                    constants["ability_ids"], constants["item_ids"], constants["patch"]),
        generate_mock_match(),
    ]
    assert [m.talent_names_patch for m in matches] == ["7.41", "7.41", "7.41"]


# ---------------------------------------------------------------------------
# Exact values of the real match
# ---------------------------------------------------------------------------

def test_clarity_match_fields(clarity):
    assert clarity.match_id == MATCH_ID > 2 ** 32                # the full 64-bit id
    assert (clarity.duration, clarity.radiant_score, clarity.dire_score) == (2091, 26, 44)
    assert (clarity.winner_known, clarity.radiant_win, clarity.partial) == (True, False, False)
    assert (clarity.start_time, clarity.cluster, clarity.region, clarity.game_build) == (1775643322, 151, 5, 6762)
    assert (clarity.game_mode, clarity.first_blood_time, clarity.patch) == ("ranked_all_pick", 137, "")
    assert (clarity.tower_status_radiant, clarity.tower_status_dire) == (256, 1972)
    assert (clarity.barracks_status_radiant, clarity.barracks_status_dire) == (48, 63)
    assert len(clarity.buildings) == 36 and all(parser.parse_building_name(b["name"]) for b in clarity.buildings)
    assert len(clarity.kills_log) == 70 and len(clarity.wards) == 76 and len(clarity.teamfights) == 10


def test_clarity_position_logs(clarity):
    by_hero = {p.hero_name: p for p in clarity.players}
    for hero in ("Legion Commander", "Wraith King"):          # kept whole in the fixture
        times = [e[0] for e in by_hero[hero].position_log]
        assert times == list(range(-85, 2091, 5)), hero
    # Wraith King "dies" without a kill-log entry: two reincarnations and an Aegis death
    wraith_king = by_hero["Wraith King"]
    real_deaths = [k.time for k in clarity.kills_log if k.victim_slot == wraith_king.player_slot]
    assert real_deaths == [290.3, 385.3, 820.1]
    dead_samples = [t for t, _x, _y, hp in wraith_king.position_log if hp == 0]
    assert {715, 1470, 1980} <= set(dead_samples)
    assert all(any(0 <= t - death < 120 for death in real_deaths) for t in dead_samples if t not in (715, 1470, 1980))


def test_clarity_players_have_real_keys(clarity):
    by_hero = {p.hero_name: p for p in clarity.players}
    invoker, lion, legion = by_hero["Invoker"], by_hero["Lion"], by_hero["Legion Commander"]
    assert "travel_boots" in invoker.final_item_keys and {"invis_sword", "dust"} <= set(lion.final_item_keys)
    assert "bottle" in by_hero["Void Spirit"].backpack and invoker.neutral_item == "spellslinger"
    talents = [(s.ability_key, s.level, s.time) for s in legion.skills if s.ability_key.startswith("special_bonus_")]
    assert talents == [("special_bonus_unique_legion_commander_outfight_them_armor", 1, 1094),
                       ("special_bonus_unique_legion_commander_4", 1, 1740)]
    assert "special_bonus_attributes" not in [s.ability_key for p in clarity.players for s in p.skills]
    # bought during strategy time: no combat-log row, read from the inventory at spawn
    assert sorted(e["key"] for e in legion.purchase_log if e["time"] <= 0) == [
        "faerie_fire", "gauntlets", "gauntlets", "quelling_blade", "tango"]
    assert not any(it.time == -1 for p in clarity.players for it in p.items)
    assert set(invoker.killed) == {"Void Spirit", "Zeus", "Lion", "Wraith King"} and invoker.killed["Wraith King"] == 2
    assert invoker.damage_inflictor["attack"] > 0 and set(invoker.damage) <= {p.hero_name for p in clarity.players}
    void_spirit = by_hero["Void Spirit"]
    assert void_spirit.kill_streaks == {"3": 3, "4": 2, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1}
    assert void_spirit.multi_kills == {"2": 1} and by_hero["Ogre Magi"].kill_streaks == {}
    assert (invoker.actions_per_min, by_hero["Ogre Magi"].teamfight_participation) == (325, pytest.approx(8 / 26))


def test_clarity_objectives_and_draft(clarity):
    kinds = Counter(o["type"] for o in clarity.objectives)
    assert kinds == {"building_kill": 19, "CHAT_MESSAGE_FIRSTBLOOD": 1, "CHAT_MESSAGE_COURIER_LOST": 1,
                     "CHAT_MESSAGE_ROSHAN_KILL": 1, "CHAT_MESSAGE_AEGIS": 1}
    first_blood = next(o for o in clarity.objectives if o["type"] == "CHAT_MESSAGE_FIRSTBLOOD")
    assert (first_blood["time"], first_blood["player_slot"], first_blood["key"]) == (137, 4, "128")
    courier = next(o for o in clarity.objectives if o["type"] == "CHAT_MESSAGE_COURIER_LOST")
    assert (courier["team"], courier["player_slot"], courier["value"]) == ("dire", 4, 45)
    tier4 = [(o["key"], o["time"]) for o in clarity.objectives if "tower4" in o.get("key", "")]
    assert tier4 == [("npc_dota_goodguys_tower4_top", 2036), ("npc_dota_goodguys_tower4_bot", 2054)]
    picks = [d for d in clarity.picks_bans if d["is_pick"]]
    assert [(d["hero_id"], d["team"]) for d in picks] == [
        (74, 0), (104, 0), (26, 1), (22, 1), (84, 0), (123, 0), (29, 1), (126, 1), (63, 0), (42, 1)]
    bans = [d for d in clarity.picks_bans if not d["is_pick"]]
    assert len(bans) == 16 and all(d["team"] is None for d in bans)      # the replay does not say who banned


def test_opendota_keeps_the_names_of_other_units(opendota, opendota_raw):
    heroes = {p.hero_name for p in opendota.players}
    for raw_player, player in zip(opendota_raw["players"], opendota.players):
        for field in ("damage", "damage_taken"):
            # only the heroes of the match are re-keyed (and the hero itself is left out)
            others = {unit: n for unit, n in getattr(player, field).items() if unit not in heroes}
            assert others == {unit: n for unit, n in raw_player[field].items()
                              if not unit.startswith("npc_dota_hero_")}, (player.hero_name, field)
    by_hero = {p.hero_name: p for p in opendota.players}
    assert by_hero["Wraith King"].damage_taken["dota_fountain"] == 1846
    assert by_hero["Void Spirit"].damage["illusion_npc_dota_hero_void_spirit"] == 4


# ---------------------------------------------------------------------------
# Clarity and OpenDota describe the same match
# ---------------------------------------------------------------------------

SCOREBOARD = (
    "player_slot", "hero_id", "hero_name", "hero_slug", "team", "kills", "deaths", "assists", "level",
    "last_hits", "denies", "gold_per_min", "xp_per_min", "hero_damage", "tower_damage", "hero_healing",
    "net_worth", "neutral_item", "lane", "lane_role", "is_roaming", "dn_t", "killed", "killed_by",
    "obs_placed", "sen_placed", "observer_kills", "sentry_kills", "lane_kills", "neutral_kills",
    "ancient_kills", "tower_kills", "roshans_killed", "courier_kills", "buyback_count", "runes",
    "multi_kills", "kill_streaks", "actions_per_min",
)


def test_scoreboard_is_the_same_from_both_sources(clarity, opendota):
    for field in ("match_id", "duration", "radiant_win", "winner_known", "game_mode", "start_time",
                  "radiant_score", "dire_score", "first_blood_time", "cluster", "region",
                  "tower_status_radiant", "tower_status_dire", "barracks_status_radiant", "barracks_status_dire"):
        assert getattr(clarity, field) == getattr(opendota, field), field
    for ours, theirs in zip(clarity.players, opendota.players):
        for field in SCOREBOARD:
            assert getattr(ours, field) == getattr(theirs, field), (ours.hero_name, field)
        assert sorted(ours.final_item_keys) == sorted(theirs.final_item_keys), ours.hero_name
        assert sorted(ours.backpack) == sorted(theirs.backpack), ours.hero_name
        assert ours.teamfight_participation == pytest.approx(theirs.teamfight_participation, abs=1e-4)
        # OpenDota's last sample is up to a second older than the end of the game
        assert ours.stuns == pytest.approx(theirs.stuns, abs=0.5), ours.hero_name
        assert ours.lane_efficiency == pytest.approx(theirs.lane_efficiency, abs=0.02), ours.hero_name
        # the minute series are sampled at the tick OpenDota samples them: every value is equal
        # (before the analyzer followed OpenDota's frame clock, 19 of these 1400 values differed)
        for series in ("gold_t", "xp_t", "lh_t", "dn_t"):
            assert getattr(ours, series) == getattr(theirs, series), (ours.hero_name, series)
        # Valve caps its upgrade list at 25 entries (Invoker has 27)
        mine = [(s.ability_key, s.level) for s in ours.skills]
        assert mine[:len(theirs.skills)] == [(s.ability_key, s.level) for s in theirs.skills], ours.hero_name
        assert len(mine) - len(theirs.skills) == (2 if ours.hero_name == "Invoker" else 0)


def test_advantage_series_are_the_same_from_both_sources(clarity, opendota):
    assert clarity.radiant_gold_adv == opendota.radiant_gold_adv
    assert clarity.radiant_xp_adv == opendota.radiant_xp_adv


def test_wards_are_the_same_from_both_sources(clarity, opendota):
    theirs = list(opendota.wards)
    for ward in clarity.wards:
        twin = next(w for w in theirs if w.ward_type == ward.ward_type and w.player_slot == ward.player_slot
                    and abs(w.time - ward.time) <= 1 and geo.distance(w.x, w.y, ward.x, ward.y) < 100)
        theirs.remove(twin)
        assert (ward.team, ward.dewarded, ward.killer_slot) == (twin.team, twin.dewarded, twin.killer_slot), ward
        assert (ward.destroyed_time is None) == (twin.destroyed_time is None), ward
        assert ward.destroyed_time is None or abs(ward.destroyed_time - twin.destroyed_time) <= 1, ward
    assert not theirs
    assert Counter((w.ward_type, w.dewarded) for w in clarity.wards) == {
        ("observer", True): 8, ("observer", False): 22, ("sentry", True): 23, ("sentry", False): 23}


def test_kill_logs_agree(clarity, opendota):
    ours = [(k.time, k.victim_slot, k.killer_slot) for k in clarity.kills_log]

    def _in_ours(kill) -> bool:
        return any(abs(t - kill.time) <= 2 and (victim, killer) == (kill.victim_slot, kill.killer_slot)
                   for t, victim, killer in ours)

    confirmed = [k for k in opendota.kills_log if k.confirmed]
    assert all(_in_ours(k) for k in confirmed)
    # OpenDota cannot tell which two of Invoker's four "kills" of Wraith King were reincarnations
    unconfirmed = [k for k in opendota.kills_log if not k.confirmed]
    assert sorted(k.time for k in unconfirmed if not _in_ours(k)) == [1241.0, 1711.0]
    assert len(confirmed) + len(unconfirmed) - 2 == len(ours) == 70
    # OpenDota only knows the place of a death inside a teamfight; the replay knows all of them
    for kill in opendota.kills_log:
        twin = next((k for k in clarity.kills_log if abs(k.time - kill.time) <= 2
                     and k.victim_slot == kill.victim_slot), None)
        if kill.x is not None and twin is not None:
            assert geo.distance(kill.x, kill.y, twin.x, twin.y) < 400, kill


def test_teamfights_and_objectives_agree(clarity, opendota):
    assert len(clarity.teamfights) == len(opendota.teamfights) == 10
    for ours, theirs in zip(clarity.teamfights, opendota.teamfights):
        # the analyzer truncates the clock, OpenDota rounds it
        assert abs(ours.start_time - theirs.start_time) <= 1 and abs(ours.end_time - theirs.end_time) <= 1
        for a, b in zip(ours.players, theirs.players):
            assert (a.player_slot, a.hero_name, a.buybacks) == (b.player_slot, b.hero_name, b.buybacks)
            if a.hero_name not in ("Invoker", "Wraith King"):      # see test_kill_logs_agree
                assert (a.kills, a.deaths) == (b.kills, b.deaths), (ours.start_time, a.hero_name)
                assert [t["target_slot"] for t in a.kill_targets] == [t["target_slot"] for t in b.kill_targets]

    def _events(match, kind):
        return [(o.get("team"), o.get("key") if kind == "CHAT_MESSAGE_FIRSTBLOOD" else None)
                for o in match.objectives if o["type"] == kind]

    for kind in ("CHAT_MESSAGE_FIRSTBLOOD", "CHAT_MESSAGE_ROSHAN_KILL", "CHAT_MESSAGE_COURIER_LOST",
                 "CHAT_MESSAGE_AEGIS"):
        assert _events(clarity, kind) == _events(opendota, kind), kind
    courier = [next(o for o in m.objectives if o["type"] == "CHAT_MESSAGE_COURIER_LOST") for m in (clarity, opendota)]
    assert [(o["team"], o["player_slot"], o["value"]) for o in courier] == [("dire", 4, 45)] * 2

    def _buildings(match):
        # OpenDota's log has no side for a tier-4 tower
        return [(re.sub(r"tower4_(top|bot)$", "tower4", o["key"]), o["team"], o.get("player_slot"))
                for o in match.objectives if o["type"] == "building_kill"]

    assert _buildings(clarity) == _buildings(opendota)
    picks = [[(d["hero_id"], d["team"], d["order"]) for d in m.picks_bans if d["is_pick"]] for m in (clarity, opendota)]
    assert picks[0] == picks[1]
    ours = [d["hero_id"] for d in clarity.picks_bans if not d["is_pick"]]
    theirs = [d["hero_id"] for d in opendota.picks_bans if not d["is_pick"]]
    assert ours[:len(theirs)] == theirs       # the player-nominated bans come first, in Valve's order
