"""The committed stand-ins for fetched game data must not rot.

``frontend/static/data/*.json`` (catalogs), ``backend/data/fallback/*.json``
(numbers) and ``backend/data/editorial/*.json`` (coaching text) are committed
and describe one game version. These tests fail when one of them lags:

- the catalogs must be exactly what ``backend/scripts/gen_static_catalogs``
  builds from the committed data set (``fixtures/gamedata/constants.json.gz``,
  the dotaconstants files reduced to what the generator reads) — whoever
  updates the data set, the generator or a catalog has to carry the change
  through (one command, see the script);
- every editorial / fallback key must be an item or ability of that version,
  and every number an editorial text quotes must be one that item or ability
  states in the data set;
- on a machine that has fetched game data (``backend/data/gamedata``), the
  catalogs must cover it — which is how a new patch shows up without a
  network in the test run.
"""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import gzip
import json
import re
from pathlib import Path

import pytest

from backend.app import config, parser
from backend.app.ai import data_manager
from backend.scripts import gen_static_catalogs as generator

REGENERATE = "run `python -m backend.scripts.gen_static_catalogs` and commit the result"
CATALOG_DIR = generator.CATALOG_DIR


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def catalogs() -> dict[str, dict]:
    return {name: _load(CATALOG_DIR / f"{name}.json") for name, _table in generator.CATALOGS}


@pytest.fixture(scope="module")
def data_set() -> dict:
    return generator.read_fixture()


# ---------------------------------------------------------------------------
# Catalogs == generator(data set)
# ---------------------------------------------------------------------------

def test_committed_catalogs_are_what_the_generator_builds_from_the_data_set(catalogs, data_set):
    built = generator.build(data_set)
    for name, table in generator.CATALOGS:
        committed = generator.current_part(catalogs[name], table)
        assert committed.keys() == built[name].keys()
        missing = sorted(set(built[name][table]) - set(committed[table]))
        assert not missing, f"{name}.json lacks {len(missing)} keys of the data set, e.g. {missing[:5]}: {REGENERATE}"
        differing = sorted(key for key, entry in built[name][table].items() if committed[table][key] != entry)
        assert not differing, f"{name}.json differs for {differing[:5]}: {REGENERATE}"
        assert committed == built[name], f"{name}.json lags the data set: {REGENERATE}"
        assert list(committed[table]) == list(built[name][table]), f"{name}.json: entry order changed: {REGENERATE}"


def test_a_lagging_catalog_is_detected(catalogs, data_set):
    """The comparison above really fails for the kinds of rot found in the
    7.40-era catalogs: a missing item, a wrong cost, a template token."""
    built = generator.build(data_set)["items"]
    committed = generator.current_part(catalogs["items"], "by_key")
    assert committed == built
    for damage in (lambda c: c["by_key"].pop("consecrated_wraps"),
                   lambda c: c["by_key"]["heart"].update(cost=5100),
                   lambda c: c["name_to_key"].update({"Dagon": "dagon_5"})):
        stale = json.loads(json.dumps(committed))
        damage(stale)
        assert stale != built


def test_catalog_stamp_names_the_version_the_defaults_are_for(catalogs):
    """DEFAULT_PATCH_LABEL (what the app assumes without fetched data) and the
    committed catalogs are for the same game version: bump them together."""
    for name, catalog in catalogs.items():
        meta = catalog["meta"]
        assert meta["patch"] == data_manager.DEFAULT_PATCH_LABEL, (
            f"{name}.json is stamped {meta['patch']}, DEFAULT_PATCH_LABEL is {data_manager.DEFAULT_PATCH_LABEL}: "
            f"regenerate the catalogs, or bump DEFAULT_PATCH_LABEL / DEFAULT_PATCH_RELEASED in data_manager.py")
        assert data_manager._parse_utc(meta["generated_at"]) is not None and meta["source"]
        # generated on or after the release it is stamped with
        assert meta["generated_at"] >= data_manager.DEFAULT_PATCH_RELEASED


# ---------------------------------------------------------------------------
# Content (PA04)
# ---------------------------------------------------------------------------

def test_items_cover_the_current_game_with_recipes_and_costs(catalogs):
    items = catalogs["items"]["by_key"]
    # 7.41 items, components and neutral artifacts that the 7.40 catalog lacked
    for key in ("consecrated_wraps", "crellas_crozier", "essence_distiller", "specialists_array",
                "hydras_breath", "shawl", "splintmail", "wizard_hat", "chasm_stone",
                "partisans_brand", "spellslinger", "foragers_kit"):
        assert items[key]["name"] and isinstance(items[key]["id"], int), key
    # costs of 7.41c-f (Valve's item feed)
    assert {key: items[key]["cost"] for key in (
        "heart", "greater_crit", "dagon", "octarine_core", "heavens_halberd", "dragon_lance", "hurricane_pike")} == {
        "heart": 5300, "greater_crit": 5200, "dagon": 3000, "octarine_core": 5100,
        "heavens_halberd": 3300, "dragon_lance": 2000, "hurricane_pike": 4550}
    recipes = {key: entry for key, entry in items.items() if key.startswith("recipe_")}
    assert len(recipes) > 70 and all(entry["name"].endswith(" Recipe") for entry in recipes.values())
    assert recipes["recipe_abyssal_blade"]["cost"] > 0
    assert recipes["recipe_abyssal_blade"]["img"].endswith("/items/recipe.png")
    ids = [entry["id"] for entry in items.values() if entry.get("id") is not None]
    assert len(ids) == len(set(ids))
    assert all(entry["key"] == key for key, entry in items.items())
    # a name is one line: Valve's "(Warlock)" second line is not part of it
    assert items["black_grimoire"]["name"] == "Black Grimoire"
    assert catalogs["items"]["name_to_key"]["Black Grimoire"] == "black_grimoire"
    assert not [key for key, entry in items.items() if "\\n" in entry["name"] or "\n" in entry["name"]]


def test_a_shared_display_name_belongs_to_the_base_item(catalogs):
    items = catalogs["items"]
    for name, base in (("Dagon", "dagon"), ("Diffusal Blade", "diffusal_blade"), ("Necronomicon", "necronomicon")):
        assert items["name_to_key"][name] == base
        same_name = [key for key, entry in items["by_key"].items() if entry["name"] == name]
        assert same_name[0] == base and len(same_name) > 1        # first in table order, too
    # every mapped name points at an entry that carries it
    for name, key in items["name_to_key"].items():
        assert items["by_key"][key]["name"] == name
    abilities = catalogs["abilities"]
    assert abilities["name_to_key"]["Mana Break"] == "antimage_mana_break"   # the hero's, not a creep's
    for name, key in abilities["name_to_key"].items():
        assert abilities["by_key"][key]["name"] == name


def test_removed_items_are_flagged_and_never_shadow_a_live_one(catalogs):
    items = catalogs["items"]["by_key"]
    obsolete = {key for key, entry in items.items() if entry.get("obsolete")}
    assert {"eternal_shroud", "wraith_pact", "cornucopia", "hood_of_defiance"} <= obsolete
    assert not obsolete & {"heart", "black_king_bar", "consecrated_wraps", "dagon", "gungir"}
    legacy = {key for key, entry in items.items() if entry.get("legacy")}
    assert legacy and not legacy & obsolete                      # legacy: not even listed upstream
    live_names = {entry["name"] for key, entry in items.items() if key not in obsolete | legacy}
    for name, key in catalogs["items"]["name_to_key"].items():
        if name in live_names:
            assert key not in obsolete | legacy, name


def test_no_ability_name_keeps_a_template_token(catalogs):
    abilities = catalogs["abilities"]["by_key"]
    assert len(abilities) > 2800
    bad = [key for key, entry in abilities.items()
           if not str(entry["name"]).strip() or "{" in entry["name"] or "}" in entry["name"]]
    assert bad == []
    assert not [name for name in catalogs["abilities"]["name_to_key"] if "{" in name]
    assert not [key for key, entry in catalogs["items"]["by_key"].items() if "{" in str(entry["name"])]
    # abilities of current heroes that the 7.40 catalog lacked
    for key in ("mirana_celestial_quiver", "dragon_knight_wyrms_wrath", "tidehunter_leviathans_catch",
                "techies_mutually_assured_destruction", "rubick_curiosity"):
        assert abilities[key]["name"], key
    # no source names these: built-in labels
    assert {key: abilities[key]["name"] for key in data_manager.BUILTIN_ABILITY_LABELS} == {
        "special_bonus_attributes": "Attribute Bonus", "ability_capture": "Capture Outpost",
        "ability_lamp_use": "Activate Watcher", "twin_gate_portal_warp": "Twin Gate"}
    ids = [entry["id"] for entry in abilities.values() if "id" in entry]
    assert len(ids) > 2700 and len(ids) == len(set(ids))
    # an ability the source lists under two ids, and one it only knows by id any more
    assert abilities["templar_assassin_hidden_gates"]["ids"] == [3060, 1617]
    assert abilities["templar_assassin_hidden_gates"]["id"] == 3060
    assert abilities["axe_coat_of_blood"]["legacy"] is True and abilities["axe_coat_of_blood"]["id"] == 1201


def test_heroes_are_complete_and_carry_their_vision(catalogs):
    heroes = catalogs["heroes"]
    assert len(heroes["by_id"]) >= 127
    for name in ("Largo", "Kez", "Ringmaster", "Outworld Destroyer"):
        assert name in heroes["name_to_id"]
    # the game's names where dotaconstants has another one; the old spelling still finds the hero
    assert {old: heroes["by_id"][str(heroes["name_to_id"][old])]["name"]
            for old in ("Ring Master", "Outworld Devourer")} == {
        "Ring Master": "Ringmaster", "Outworld Devourer": "Outworld Destroyer"}
    assert len(heroes["name_to_id"]) == len(heroes["by_id"]) + len(generator.HERO_NAMES)
    vision = {entry["name"]: (entry["day_vision"], entry["night_vision"]) for entry in heroes["by_id"].values()}
    exceptions = {name: v for name, v in vision.items() if v != data_manager.DEFAULT_HERO_VISION}
    assert exceptions == {"Night Stalker": (800, 1800), "Slark": (1800, 1800), "Sniper": (1800, 1400),
                          "Bane": (1800, 1200), "Winter Wyvern": (1800, 1200),
                          "Bounty Hunter": (1800, 1000), "Gyrocopter": (1800, 1000)}
    for entry in heroes["by_id"].values():                      # what frontend/static/js/app.js reads
        assert heroes["name_to_id"][entry["name"]] == entry["id"] and entry["slug"]


def test_the_readers_in_the_parser_get_what_they_need():
    """backend/app/parser.py reads the catalogs for ids, names and canonical keys."""
    ids = parser._item_id_map()
    assert ids[1854] == ("Consecrated Wraps", "consecrated_wraps")
    assert ids[104] == ("Dagon", "dagon") and ids[204] == ("Dagon", "dagon_5")
    assert parser._item_name_index()["dagon"] == "dagon"
    assert parser.canonical_item_key("item_recipe_abyssal_blade") == "recipe_abyssal_blade"
    assert parser.canonical_item_key("boots_of_travel") == "travel_boots"
    assert parser.item_display_name("crellas_crozier") == "Crella's Crozier"
    assert parser.item_display_name("recipe_dagon") == "Dagon Recipe"
    assert parser.ability_display_name("dragon_knight_wyrms_wrath") == "Wyrm's Wrath"
    assert parser.ability_display_name("ability_capture") == "Capture Outpost"
    assert parser.load_hero_map()[155]["name"] == "Largo"


# ---------------------------------------------------------------------------
# Editorial and fallback data (PA12)
# ---------------------------------------------------------------------------

def _usable(catalog: dict) -> set[str]:
    """Keys of the current game: neither carried over from an old catalog nor removed."""
    return {key for key, entry in catalog["by_key"].items() if not entry.get("legacy") and not entry.get("obsolete")}


def test_every_editorial_and_fallback_item_is_in_the_current_game(catalogs):
    current = _usable(catalogs["items"])
    names = {key: entry["name"] for key, entry in catalogs["items"]["by_key"].items()}
    tips = _load(config.EDITORIAL_DIR / "coaching_tips_items.json")
    fallback = _load(data_manager.FALLBACK_DIR / "item_stats.json")["items"]
    for label, table in (("coaching_tips_items.json", tips), ("item_stats.json", fallback)):
        gone = sorted(set(table) - current)
        assert not gone, f"{label} has entries for items that are not in the game any more: {gone}"
        renamed = {key: (info["item_name"], names[key]) for key, info in table.items()
                   if info["item_name"] != names[key]}
        assert not renamed, f"{label}: display names changed: {renamed}"
    assert set(tips) == set(fallback)                            # one category / cost / description per tip
    for key in ("consecrated_wraps", "crellas_crozier", "essence_distiller", "specialists_array", "hydras_breath"):
        assert tips[key]["coaching_tip"] and fallback[key]["description"] and fallback[key]["category"]
    for key in ("eternal_shroud", "wraith_pact"):                # removed in 7.41 / earlier
        assert key not in tips and key not in fallback
    for info in tips.values():                                   # typical buyers are heroes of the game
        for hero in info["heroes"]:
            assert hero == "all" or data_manager.is_known_hero(hero), hero


def test_fallback_numbers_equal_the_current_data(catalogs):
    """The fallback is generated; a catalog regenerated without it shows here."""
    items = catalogs["items"]["by_key"]
    fallback = _load(data_manager.FALLBACK_DIR / "item_stats.json")
    wrong = {key: (info["cost"], items[key]["cost"]) for key, info in fallback["items"].items()
             if info["cost"] != items[key]["cost"]}
    assert not wrong, f"item_stats.json costs lag the catalog: {wrong}: {REGENERATE}"
    for name in ("item_stats.json", "ability_stats.json"):
        assert data_manager.DEFAULT_PATCH_LABEL in _load(data_manager.FALLBACK_DIR / name)["_comment"]
    # values the 7.37-era file had wrong (audit: BKB 75, Call 16/14/12/10, Finger 160/100/40)
    abilities = _load(data_manager.FALLBACK_DIR / "ability_stats.json")["abilities"]
    assert fallback["items"]["black_king_bar"]["cooldown"] == 95
    assert abilities["axe_berserkers_call"]["cooldown"] == [18, 16, 14, 12]
    assert abilities["lion_finger_of_death"]["cooldown"] == [110, 70, 30]
    assert abilities["pudge_flesh_heap"]["ability_name"] == "Meat Shield"   # renamed, same key


def test_every_editorial_and_fallback_ability_is_in_the_current_game(catalogs, data_set):
    current = _usable(catalogs["abilities"])
    by_key = catalogs["abilities"]["by_key"]
    of_hero: dict[str, set[str]] = {}
    for npc_name, info in data_set["hero_abilities"].items():
        keys = of_hero.setdefault(npc_name.replace("npc_dota_hero_", ""), set())
        for ability in info["abilities"]:
            keys.update(ability if isinstance(ability, list) else [ability])
    tips = _load(config.EDITORIAL_DIR / "coaching_tips_abilities.json")
    fallback = _load(data_manager.FALLBACK_DIR / "ability_stats.json")["abilities"]
    for label, table in (("coaching_tips_abilities.json", tips), ("ability_stats.json", fallback)):
        gone = sorted(set(table) - current)
        assert not gone, f"{label} has entries for abilities that are not in the game any more: {gone}"
        for key, info in table.items():
            assert key in of_hero[data_manager.hero_slug(info["hero"])], (label, key, info["hero"])
    for label, table in (("coaching_tips_abilities.json", tips), ("ability_stats.json", fallback)):
        renamed = {key: (info["ability_name"], by_key[key]["name"]) for key, info in table.items()
                   if info["ability_name"] != by_key[key]["name"]}
        # (a renamed ability was usually reworked: read its tip again before taking the new name)
        assert not renamed, f"{label}: display names changed: {renamed}"


def test_editorial_hero_facts_equal_the_catalog(catalogs):
    heroes = {entry["name"]: entry for entry in catalogs["heroes"]["by_id"].values()}
    editorial = _load(config.EDITORIAL_DIR / "hero_knowledge.json")
    attr = {"all": "uni", "str": "str", "agi": "agi", "int": "int"}
    unknown = sorted(set(editorial) - set(heroes))
    assert not unknown, f"hero_knowledge.json names heroes the catalog does not have: {unknown}"
    wrong = {name: (info["attribute"], info["attack_type"]) for name, info in editorial.items()
             if (info["attribute"], info["attack_type"]) != (attr[heroes[name]["attr"]],
                                                           heroes[name]["attack_type"].lower())}
    assert not wrong, f"hero_knowledge.json: outdated attribute / attack type: {wrong}"
    for pair in _load(config.EDITORIAL_DIR / "matchup_tips.json"):
        for hero in pair.split(":"):
            assert data_manager.is_known_hero(hero), pair


# Numbers an editorial text may quote although the item's / ability's own
# constants do not state them: (file, key) -> the numbers, and where they are from.
_QUOTED_FROM_ELSEWHERE = {
    # on sale from minute 15: items.txt ItemInitialStockTime 990 s, 90 s of them before the horn
    ("coaching_tips_items.json", "aghanims_shard"): {"15"},
    # the game's own text spells the number out ("five seconds earlier")
    ("coaching_tips_abilities.json", "weaver_time_lapse"): {"5"},
}
_PATCH_NUMBER = re.compile(r"\b7\.\d\d[a-z]?\b")       # "since 7.38" is a version, not a stat


def _texts(value) -> str:
    if isinstance(value, dict):
        return " ".join(_texts(v) for v in value.values())
    return " ".join(_texts(v) for v in value) if isinstance(value, list) else str(value or "")


def test_numbers_in_editorial_texts_are_the_games(data_set):
    """A tip that says "40% amplified magic damage" when the game says 30% is
    worse than a tip without a number. Every number in an editorial item /
    ability text has to be one the same item / ability states in the data set
    (its costs, cooldowns, attribute values and ability texts), so the tips
    that quote a changed value fail here after a regenerated data set."""
    numbers = data_set["numbers"]
    assert len(numbers["items"]) > 380 and len(numbers["abilities"]) > 700
    wrong = {}
    for name, table, fields in (
            ("coaching_tips_items.json", "items", ("description", "coaching_tip", "counter_tips")),
            ("coaching_tips_abilities.json", "abilities", ("description", "coaching_tip"))):
        for key, info in _load(config.EDITORIAL_DIR / name).items():
            text = _PATCH_NUMBER.sub("", _texts([info.get(field) for field in fields]))
            unknown = (generator.text_numbers(text) - set(numbers[table][key])
                       - _QUOTED_FROM_ELSEWHERE.get((name, key), set()))
            if unknown:
                wrong[key] = sorted(unknown, key=float)
    assert not wrong, (f"editorial texts quote numbers the game data does not state for that item / ability: "
                       f"{wrong}. Correct the text from the current values.")
    # the generated descriptions are held to the same rule (nobody edits them by hand)
    fallback = _load(data_manager.FALLBACK_DIR / "item_stats.json")["items"]
    edited = {key: sorted(generator.text_numbers(info["description"]) - set(numbers["items"][key]))
              for key, info in fallback.items()
              if generator.text_numbers(info["description"]) - set(numbers["items"][key])}
    assert not edited, f"item_stats.json descriptions differ from the data set: {edited}: {REGENERATE}"


def test_the_number_check_catches_the_stale_tips_of_7_37(data_set):
    """The texts this check was written for (all corrected since)."""
    numbers = data_set["numbers"]
    for table, key, stale, quoted in (
            ("items", "ghost", "the 40% amplified magic damage can get you killed", "40"),
            ("items", "holy_locket", "The 35% heal amp affects all heals", "35"),
            ("items", "hurricane_pike", "granting 4 unlimited-range attacks", "4"),
            ("items", "guardian_greaves", "when allies are below 20% HP", "20"),
            ("items", "heart", "+250 HP, +45 Str. Regenerate 1.6% of max HP", "1.6"),
            ("items", "sheepstick", "Hex target enemy for 3.5s", "3.5"),
            ("abilities", "axe_culling_blade", "kill thresholds (250/350/450 HP)", "250"),
            ("abilities", "viper_viper_strike", "10s max-level CD", "10")):
        assert quoted in generator.text_numbers(stale) - set(numbers[table][key]), key
    # ... and what it reads as a number
    assert generator.text_numbers("2.50 dmg, .5s, -30%, 9 / 8 / 7, 1200") == {"2.5", "0.5", "30", "9", "8", "7", "1200"}
    assert generator.text_numbers("1v1 with dagon_5 at level25") == set()
    assert generator.text_numbers([4.0, "5", None, True, ["6"]]) == {"4", "5", "6"}
    assert generator.entry_numbers({"cost": 5300, "cd": False, "id": 114, "img": "/heart.png?t=1593393829403",
                                    "attrib": [{"key": "bonus_strength", "value": "40"}],
                                    "abilities": [{"description": "Regenerates 1.5% of missing health."}]}) == [
        "1.5", "40", "5300"]


def test_no_editorial_text_describes_a_removed_mechanic():
    """Mechanics the tips of the 7.37 era still described (7.38 - 7.41 removed them)."""
    tips = {key: info["coaching_tip"] for key, info in
            _load(config.EDITORIAL_DIR / "coaching_tips_items.json").items()}
    for key, gone in (("gungir", "Maelstrom"), ("maelstrom", "Gleipnir"),          # 7.38: no Maelstrom in Gleipnir
                      ("bloodthorn", "crit"), ("angels_demise", "crit"),             # critical strikes removed
                      ("hand_of_midas", "maximum XP"),                              # 7.40: no experience multiplier
                      ("diffusal_blade", "illusions also burn"),                    # 7.40
                      ("guardian_greaves", "armor aura"), ("guardian_greaves", "20%"),   # 7.40
                      ("kaya_and_sange", "status resistance"),                      # 7.36: slow resistance
                      ("arcane_blink", "cast point"), ("solar_crest", "debuff the enemy"),
                      ("heart", "out-of-combat"), ("aether_lens", "Octarine")):
        assert gone.lower() not in tips[key].lower(), (key, gone)
    assert "60%" in tips["bloodstone"] and "3.5 seconds" in tips["heavens_halberd"]
    assert "status resistance" not in tips["heavens_halberd"]
    abilities = _load(config.EDITORIAL_DIR / "coaching_tips_abilities.json")
    for key, gone in (("naga_siren_ensnare", "goes through BKB"),                    # 7.36: only with Aghanim's Scepter
                      ("riki_backstab", "from behind deal bonus damage"),           # 7.36: Backstab is the innate
                      ("omniknight_martyr", "strong dispel"), ("tinker_rearm", "items"),
                      ("vengefulspirit_command_aura", "illusion"), ("bounty_hunter_track", "speed"),
                      ("life_stealer_ghoul_frenzy", "slow"), ("axe_battle_hunger", "move speed"),
                      # heroes whose first entries were written before their abilities were known
                      ("kez_echo_slash", "dash"), ("largo_frogstomp", "leaps"),
                      ("ringmaster_the_box", "trap enemies"), ("ringmaster_wheel", "random")):
        text = (abilities[key]["description"] + " " + abilities[key]["coaching_tip"]).lower()
        assert gone.lower() not in text, (key, gone)
    for pattern in (r"Eternal Shroud", r"Wraith Pact", r"Cornucopia", r"\bfacets?\b", r"Tome of Knowledge",
                    r"Wisdom Rune", r"side shop",
                    # abilities the game no longer has (renamed or replaced since 7.37)
                    r"Bad Juju", r"Heat-Seeking Missile", r"Heavenly Grace", r"Sidekick", r"Silken Bola",
                    r"Spell Shield"):
        for name in ("coaching_tips_items.json", "coaching_tips_abilities.json", "hero_knowledge.json",
                     "matchup_tips.json"):
            found = re.findall(pattern, (config.EDITORIAL_DIR / name).read_text(encoding="utf-8"), re.IGNORECASE)
            assert not found, (name, pattern)


def test_editorial_texts_do_not_contradict_the_abilities_own_data():
    """Statements a pass over all 973 editorial entries found contradicted by the ability data of
    dotaconstants 51138e3d950f (7.41f): its description, attributes, damage type, behaviour and
    spell-immunity flag, and the hero's Aghanim's upgrades. (Text, phrase that must stay gone.)"""
    abilities = _load(config.EDITORIAL_DIR / "coaching_tips_abilities.json")
    for key, gone in (
            ("abaddon_frostmourne", "silence"),                  # slow, damage over time, attack speed for attackers
            ("arc_warden_flux", "dispelled"), ("bane_enfeeble", "status resistance"),
            ("bloodseeker_thirst", "attack damage"), ("bounty_hunter_shuriken_toss", "ministun"),
            ("brewmaster_drunken_brawler", "passively grants"),  # stances since the rework
            ("chaos_knight_phantasm", "stuns"), ("dawnbreaker_celestial_hammer", "teleport"),
            ("drow_ranger_frost_arrows", "magic damage"), ("drow_ranger_marksmanship", "agility"),
            ("enigma_midnight_pulse", "pure"), ("hoodwink_sharpshooter", "reducing healing"),
            ("wisp_tether", "stuns"), ("keeper_of_the_light_will_o_wisp", "stun"),
            ("kunkka_tidebringer", "passively"), ("legion_commander_press_the_attack", "attack speed"),
            ("leshrac_diabolic_edict", "physical"), ("life_stealer_rage", "attack speed"),
            ("lina_laguna_blade", "pierces bkb"), ("lycan_summon_wolves", "critical strike"),
            ("marci_grapple", "stun"), ("furion_teleportation", "channel"), ("pangolier_lucky_shot", "disarm"),
            ("phantom_assassin_blur", "evasion"),                # Immaterial has the evasion now
            ("phoenix_sun_ray", "channels"), ("pudge_flesh_heap", "passive damage reduction"),
            ("pudge_rot", "flesh heap stacks"), ("sandking_sand_storm", "channels"),
            ("sandking_epicenter", "channel"), ("silencer_glaives_of_wisdom", "pure"),
            ("skeleton_king_mortal_strike", "chance"), ("skeleton_king_vampiric_spirit", "allied heroes"),
            ("techies_suicide", "silenc"), ("treant_natures_grasp", "root"),
            ("troll_warlord_berserkers_rage", "bash"), ("viper_nethertoxin", "passives"),
            ("witch_doctor_death_ward", "physical")):
        text = (abilities[key]["description"] + " " + abilities[key]["coaching_tip"]).lower()
        assert gone not in text, (key, gone)
    heroes = _load(config.EDITORIAL_DIR / "hero_knowledge.json")
    for hero, gone in (
            ("Phantom Assassin", "Blur evasion"), ("Legion Commander", "attack speed + HP regen"),
            ("Sand King", "channel"), ("Pangolier", "disarm"), ("Brewmaster", "passive bonuses"),
            ("Phoenix", "channels"), ("Marci", "stun —"), ("Spirit Breaker", "magic immunity"),
            ("Witch Doctor", "physical damage"), ("Treant Protector", "limited charges"),
            ("Treant Protector", "vines that root"), ("Keeper of the Light", "true sight"),
            ("Oracle", "block the heal"), ("Silencer", "disarms"), ("Silencer", "pure damage"),
            ("Leshrac", "physical damage"), ("Viper", "breaks passives"), ("Techies", "silences"),
            ("Techies", "50%"), ("Enigma", "Demonic Conversion"), ("Doom", "and items"),
            ("Beastmaster", "Call of the Wild")):
        assert gone not in _texts(heroes[hero]), (hero, gone)
    matchups = _load(config.EDITORIAL_DIR / "matchup_tips.json")
    for pair, gone in (
            ("Templar Assassin:Viper", "Nethertoxin's break"), ("Oracle:Huskar", "physical damage immunity"),
            ("Beastmaster:Anti-Mage", "prevents Blink"), ("Juggernaut:Legion Commander", "attack speed"),
            ("Luna:Clockwerk", "channeling"), ("Anti-Mage:Sand King", "channel"),
            ("Faceless Void:Sand King", "channel"), ("Phantom Assassin:Slardar", "Blur evasion"),
            ("Axe:Phantom Assassin", "evasion from Blur")):
        assert gone not in matchups[pair]["tip"], (pair, gone)


# ---------------------------------------------------------------------------
# Against fetched game data, where a machine has it (PA15)
# ---------------------------------------------------------------------------

def test_catalogs_cover_the_fetched_game_data(catalogs):
    """Fetched data newer than the catalogs is how a patch reaches this suite
    without a network: items or hero abilities the catalogs lack turn it red."""
    gamedata = config.GAMEDATA_DIR
    if not (gamedata / "items.json").is_file() or not (gamedata / "abilities.json").is_file():
        pytest.skip("no fetched game data on this machine (backend/data/gamedata)")
    fetched_items = _load(gamedata / "items.json")
    items = catalogs["items"]["by_key"]
    missing = sorted({str(info["key"]) for info in fetched_items.values() if info.get("key")} - set(items))
    assert not missing, f"items.json lacks fetched items {missing[:10]}: {REGENERATE}"
    abilities = catalogs["abilities"]["by_key"]
    # (files older than schema 2 list placeholder slots under their key: not abilities)
    fetched_abilities = {key for hero in _load(gamedata / "abilities.json").values()
                         for name, data in hero.items() if name != data.get("key")
                         for key in data_manager.ability_keys(data)}
    missing = sorted(fetched_abilities - set(abilities))
    assert not missing, f"abilities.json lacks fetched hero abilities {missing[:10]}: {REGENERATE}"

    # Costs only say "the catalog lags" when the fetch is the newer of the two.
    meta = _load(gamedata / "meta.json") if (gamedata / "meta.json").is_file() else {}
    fetched_at = str(meta.get("source_date") or "")
    if not fetched_at or fetched_at <= str(catalogs["items"]["meta"].get("source_date") or ""):
        return
    wrong = {info["key"]: (items[info["key"]]["cost"], info.get("cost")) for info in fetched_items.values()
             if info.get("key") in items and info.get("cost") != items[info["key"]]["cost"]}
    assert not wrong, f"items.json costs lag the fetched data {dict(list(wrong.items())[:10])}: {REGENERATE}"


def test_built_in_rule_values_equal_the_fetched_item_attributes():
    """data_manager.game_rules() falls back to built-in ward / smoke values.
    When fetched data states another value, a patch changed a rule: update
    the built-in value and review whatever still hard-codes it."""
    path = config.GAMEDATA_DIR / "items.json"
    if not path.is_file():
        pytest.skip("no fetched game data on this machine (backend/data/gamedata)")
    by_key = {info.get("key"): info for info in _load(path).values() if isinstance(info, dict)}
    changed = {}
    for name, item, attribute, built_in in data_manager._ITEM_RULES:
        for entry in by_key.get(item, {}).get("attributes") or []:
            if isinstance(entry, dict) and entry.get("key") == attribute:
                values = data_manager._to_num_list(entry.get("value"))
                if len(values) == 1 and values[0] != built_in:
                    changed[name] = (built_in, values[0])
    assert not changed, f"rule values changed in the fetched game data (built-in, fetched): {changed}"


# ---------------------------------------------------------------------------
# The generator itself
# ---------------------------------------------------------------------------

_ITEMS_TXT = '''
"DOTAAbilities"
{
	"Version"		"1"
	// "item_commented_out" { "IsObsolete" "1" }
	"item_blink"
	{
		"ID"	"1"
		"AbilityValues" { "blink_range" "1200"  "IsObsolete" "1" }
		"IsObsolete"		"0"
	}
	"item_eternal_shroud"
	{
		"ItemCost"	"3900"	// a comment with "quotes" and { braces }
		"IsObsolete"	"1"
	}
	"item_recipe_wraith_pact" { "IsObsolete" "1" }
	"item_tome \\"of\\" knowledge" { "IsObsolete" "1" }
}
'''


def test_obsolete_items_are_read_from_valves_item_script():
    assert generator.obsolete_items(_ITEMS_TXT) == ["eternal_shroud", "recipe_wraith_pact", 'tome \\"of\\" knowledge']
    assert generator.obsolete_items("") == [] and generator.obsolete_items('"a" { "b" "c" }') == []


def _source_dir(tmp_path: Path, data_set: dict, items_txt: str | None = None) -> Path:
    """dotaconstants-style files for a slice of the data set (Axe, a few items)."""
    source = tmp_path / "source"
    source.mkdir()
    abilities = {key: data_set["abilities"][key]
                 for key in ("axe_berserkers_call", "axe_culling_blade", "special_bonus_hp_250")}
    abilities["axe_berserkers_call"] = {**abilities["axe_berserkers_call"], "cd": ["18", "16", "14", "12"]}
    items = {key: data_set["items"][key] for key in (
        "dagon_5", "dagon", "recipe_dagon", "black_king_bar", "eternal_shroud", "heart")}
    items["black_king_bar"] = {**items["black_king_bar"], "attrib": [      # what it says about itself
        {"key": "bonus_strength", "display": "+ {value} Strength", "value": "10"},
        {"key": "duration", "value": "9 / 8 / 7"}]}
    files = {
        "heroes": {"2": data_set["heroes"]["2"]},
        "hero_abilities": {"npc_dota_hero_axe": {"abilities": ["axe_berserkers_call", "axe_culling_blade"]}},
        "abilities": abilities,
        "items": items,
        "ability_ids": {aid: key for aid, key in data_set["ability_ids"].items() if key in abilities},
    }
    for name, payload in files.items():
        (source / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    if items_txt is not None:
        (source / "items.txt").write_text(items_txt, encoding="utf-8")
    return source


def test_generator_writes_checks_and_keeps_removed_keys(tmp_path, data_set, capsys):
    source = _source_dir(tmp_path, data_set, _ITEMS_TXT)
    out, fallback, fixture = tmp_path / "catalogs", tmp_path / "fallback", tmp_path / "fixture.json.gz"
    out.mkdir()
    fallback.mkdir()
    # a previous catalog with an item the source no longer lists, and a fallback to refresh
    (out / "items.json").write_text(json.dumps({"by_key": {
        "old_neutral": {"id": 9999, "key": "old_neutral", "name": "Old Neutral", "img": None, "cost": 0}},
        "name_to_key": {"Old Neutral": "old_neutral"}, "cdn_base": generator.CDN}), encoding="utf-8")
    (fallback / "item_stats.json").write_text(json.dumps({"_comment": "old", "items": {
        "black_king_bar": {"item_name": "Black King Bar", "cost": 1, "cooldown": 75, "duration": 9, "category": "defense"},
        "eternal_shroud": {"item_name": "Eternal Shroud", "cost": 3700, "category": "defense"}}}), encoding="utf-8")
    (fallback / "ability_stats.json").write_text(json.dumps({"_comment": "old", "abilities": {
        "axe_berserkers_call": {"hero": "Axe", "ability_name": "Berserker's Call", "cooldown": [16, 14, 12, 10],
                                "cast_range": [0]},
        "axe_removed_ability": {"hero": "Axe", "ability_name": "Removed"}}}), encoding="utf-8")
    args = ["--source", str(source), "--out", str(out), "--fallback", str(fallback),
            "--fixture", str(fixture), "--patch", "7.41f"]

    assert generator.main(args + ["--check"]) == 1                # lagging, and nothing written
    assert not (out / "heroes.json").exists() and not fixture.exists()
    assert generator.main(args) == 0
    assert generator.main(args + ["--check"]) == 0                # now current
    capsys.readouterr()

    items = _load(out / "items.json")
    assert items["meta"]["patch"] == "7.41f" and items["meta"]["source"] == "files"
    assert items["name_to_key"]["Dagon"] == "dagon"
    assert [key for key in items["by_key"] if key.startswith("dagon")] == ["dagon", "dagon_5"]   # base first
    assert items["by_key"]["old_neutral"]["legacy"] is True and items["name_to_key"]["Old Neutral"] == "old_neutral"
    assert items["by_key"]["eternal_shroud"]["obsolete"] is True and "obsolete" not in items["by_key"]["heart"]
    assert (out / "items.json").read_text(encoding="utf-8").count("\n") > len(items["by_key"])   # a line per entry
    abilities = _load(out / "abilities.json")
    assert abilities["by_key"]["axe_culling_blade"]["name"] == "Culling Blade"
    assert abilities["by_key"]["ability_capture"]["name"] == "Capture Outpost"
    assert _load(out / "heroes.json")["by_id"]["2"]["slug"] == "axe"

    # the fallback keeps its entries and their category, takes every number and the description
    # from the source (hand-typed extras such as "duration" / "cast_range" go) and loses what
    # the game removed
    stats = _load(fallback / "item_stats.json")
    assert stats["items"] == {"black_king_bar": {
        "item_name": "Black King Bar", "cost": data_set["items"]["black_king_bar"]["cost"],
        "is_active": False, "description": "+10 Strength.",   # (this slice of the source lists no abilities)
        "category": "defense"}}
    # the numbers an editorial text about the item may quote go into the fixture
    assert generator.read_fixture(fixture)["numbers"]["items"]["black_king_bar"] == [
        "7", "8", "9", "10", str(data_set["items"]["black_king_bar"]["cost"])]
    assert "7.41f" in stats["_comment"]
    assert _load(fallback / "ability_stats.json")["abilities"] == {"axe_berserkers_call": {
        "hero": "Axe", "ability_name": "Berserker's Call", "cooldown": [18, 16, 14, 12]}}

    # the fixture rebuilds the same catalogs, and a source change is noticed
    assert generator.current_part(items, "by_key") == generator.build(generator.read_fixture(fixture))["items"]
    raw_items = _load(source / "items.json")
    raw_items["heart"]["cost"] += 100
    (source / "items.json").write_text(json.dumps(raw_items), encoding="utf-8")
    assert generator.main(args + ["--check"]) == 1
    assert "lag the source" in capsys.readouterr().out


def test_obsolete_flags_survive_a_run_without_valves_item_script(tmp_path, data_set, capsys):
    source = _source_dir(tmp_path, data_set, _ITEMS_TXT)
    args = ["--source", str(source), "--out", str(tmp_path / "catalogs"), "--fallback", str(tmp_path / "fallback"),
            "--fixture", str(tmp_path / "fixture.json.gz")]
    assert generator.main(args) == 0
    (source / "items.txt").unlink()                               # the mirror is unreachable
    assert generator.main(args + ["--check"]) == 0
    assert "keeping the obsolete flags" in capsys.readouterr().out
    assert _load(tmp_path / "catalogs" / "items.json")["by_key"]["eternal_shroud"]["obsolete"] is True


def _stamps(out: Path, fallback: Path) -> set[str]:
    """The game versions the five stamped files name."""
    stamps = {_load(out / f"{name}.json")["meta"]["patch"] for name, _table in generator.CATALOGS}
    for name in ("item_stats.json", "ability_stats.json"):
        stamps.update(re.findall(r"\(game version ([^)]*)\)", _load(fallback / name)["_comment"]) or [""])
    return stamps


def test_a_letter_patch_restamps_every_file(tmp_path, data_set, capsys):
    """The usual letter patch moves an item cost and nothing else. All five
    stamped files must then name the new version — test_catalog_stamp_names_...
    and test_fallback_numbers_... compare every one of them with DEFAULT_PATCH_LABEL."""
    source = _source_dir(tmp_path, data_set, _ITEMS_TXT)
    out, fallback, fixture = tmp_path / "catalogs", tmp_path / "fallback", tmp_path / "fixture.json.gz"
    args = ["--source", str(source), "--out", str(out), "--fallback", str(fallback), "--fixture", str(fixture)]
    assert generator.main(args + ["--patch", "7.41f"]) == 0
    assert _stamps(out, fallback) == {"7.41f"}

    raw_items = _load(source / "items.json")
    raw_items["heart"]["cost"] += 100
    (source / "items.json").write_text(json.dumps(raw_items), encoding="utf-8")
    heroes_before = _load(out / "heroes.json")
    assert generator.main(args + ["--patch", "7.41g", "--check"]) == 1
    assert _stamps(out, fallback) == {"7.41f"}                    # --check writes nothing
    assert generator.main(args + ["--patch", "7.41g"]) == 0
    assert _stamps(out, fallback) == {"7.41g"}
    heroes = _load(out / "heroes.json")                           # same heroes, new stamp
    assert generator.current_part(heroes, "by_id") == generator.current_part(heroes_before, "by_id")
    assert _load(out / "items.json")["by_key"]["heart"]["cost"] == raw_items["heart"]["cost"]
    assert generator.main(args + ["--patch", "7.41g", "--check"]) == 0
    capsys.readouterr()

    # a patch that changes nothing in the data is another version all the same
    assert generator.main(args + ["--patch", "7.41h", "--check"]) == 1
    assert "heroes.json: 1 entries (0 legacy), stamp is not 7.41h" in capsys.readouterr().out
    assert generator.main(args + ["--patch", "7.41h"]) == 0
    assert _stamps(out, fallback) == {"7.41h"}

    # no version to stamp (files on disk, no --patch): equal content is left alone
    files = [out / f"{name}.json" for name, _table in generator.CATALOGS] + sorted(fallback.iterdir())
    before = {path: path.read_bytes() for path in files}
    assert generator.main(args + ["--check"]) == 0 and generator.main(args) == 0
    assert {path: path.read_bytes() for path in files} == before


def test_check_reads_the_fixture_not_its_compressed_bytes(tmp_path, data_set, capsys):
    """gzip output differs between zlib builds and Python versions: a fixture
    written on one machine must not count as stale on another."""
    source = _source_dir(tmp_path, data_set, _ITEMS_TXT)
    fixture = tmp_path / "fixture.json.gz"
    args = ["--source", str(source), "--out", str(tmp_path / "catalogs"), "--fallback", str(tmp_path / "fallback"),
            "--fixture", str(fixture)]
    assert generator.main(args) == 0
    text = gzip.decompress(fixture.read_bytes())
    other = gzip.compress(text, compresslevel=1, mtime=1_700_000_000)   # the same content from "another zlib"
    assert other != fixture.read_bytes()
    fixture.write_bytes(other)
    capsys.readouterr()
    assert generator.main(args + ["--check"]) == 0
    assert "fixture.json.gz: " in capsys.readouterr().out
    assert generator.main(args) == 0 and fixture.read_bytes() == other    # ... and is not rewritten either
    # content that differs, or a file that is not gzip at all, does count
    changed = json.loads(text)
    changed["items"]["heart"]["cost"] += 1
    fixture.write_bytes(gzip.compress(json.dumps(changed, separators=(",", ":")).encode("utf-8")))
    assert generator.main(args + ["--check"]) == 1
    fixture.write_bytes(b"not a gzip file")
    assert generator.main(args + ["--check"]) == 1
    assert generator.main(args) == 0 and gzip.decompress(fixture.read_bytes()) == text
    # the order of the entries is content (which key gets a shared name)
    reordered = json.loads(text)
    reordered["items"] = dict(reversed(list(reordered["items"].items())))
    fixture.write_bytes(gzip.compress(json.dumps(reordered, ensure_ascii=False, separators=(",", ":")).encode("utf-8")))
    assert generator.main(args + ["--check"]) == 1


def test_a_removed_item_never_takes_the_name_of_a_live_one():
    """Two keys, one display name, the cheaper one removed from the game: the
    name (and the first slot in the table) belongs to the live item."""
    raw = {"items": {
        "old_orb": {"id": 1, "dname": "Orb of Something", "cost": 100},
        "new_orb": {"id": 2, "dname": "Orb of Something", "cost": 900},
        "recipe_new_orb": {"id": 3, "cost": 300},
    }}
    live = generator._items({**raw, "obsolete_items": ["old_orb"]})
    assert live["name_to_key"]["Orb of Something"] == "new_orb"
    assert list(live["by_key"]) == ["new_orb", "old_orb", "recipe_new_orb"]
    assert live["by_key"]["old_orb"]["obsolete"] is True and "obsolete" not in live["by_key"]["new_orb"]
    assert live["by_key"]["recipe_new_orb"]["name"] == "Orb of Something Recipe"
    # without the flag the usual rule decides: the cheapest of the keys that do not spell the name
    plain = generator._items({**raw, "obsolete_items": []})
    assert plain["name_to_key"]["Orb of Something"] == "old_orb" and list(plain["by_key"])[0] == "old_orb"


def test_an_item_script_that_names_no_obsolete_item_keeps_the_flags(tmp_path, data_set, capsys):
    """Valve's items.txt always flags dozens of items: one that flags none was
    read wrongly (a format change), which must not clear every flag."""
    source = _source_dir(tmp_path, data_set, _ITEMS_TXT)
    args = ["--source", str(source), "--out", str(tmp_path / "catalogs"), "--fallback", str(tmp_path / "fallback"),
            "--fixture", str(tmp_path / "fixture.json.gz")]
    assert generator.main(args) == 0
    (source / "items.txt").write_text('"DOTAAbilities" { "item_blink" { "ID" "1" "Obsolete" "yes" } }', encoding="utf-8")
    capsys.readouterr()
    assert generator.main(args + ["--check"]) == 0
    assert "items.txt names no obsolete item; keeping the obsolete flags" in capsys.readouterr().out
    assert generator.main(args) == 0
    assert _load(tmp_path / "catalogs" / "items.json")["by_key"]["eternal_shroud"]["obsolete"] is True


def test_ability_ids_reach_carried_over_keys_and_joined_ids(tmp_path, data_set):
    source = _source_dir(tmp_path, data_set, _ITEMS_TXT)
    ids = _load(source / "ability_ids.json")
    cull = next(aid for aid, key in ids.items() if key == "axe_culling_blade")
    del ids[cull]
    ids[f"{cull},9002"] = "axe_culling_blade"                     # dotaconstants: one ability, two ids
    ids["9001"] = "axe_coat_of_blood"                             # an id is all the source still has of it
    (source / "ability_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    out = tmp_path / "catalogs"
    out.mkdir()
    (out / "abilities.json").write_text(json.dumps({"by_key": {
        "axe_coat_of_blood": {"key": "axe_coat_of_blood", "name": "Coat of Blood", "img": None},
        "axe_gone_for_good": {"key": "axe_gone_for_good", "name": "Gone", "img": None}},
        "name_to_key": {"Coat of Blood": "axe_coat_of_blood", "Gone": "axe_gone_for_good"}}), encoding="utf-8")
    args = ["--source", str(source), "--out", str(out), "--fallback", str(tmp_path / "fallback"),
            "--fixture", str(tmp_path / "fixture.json.gz")]
    assert generator.main(args) == 0 and generator.main(args + ["--check"]) == 0
    abilities = _load(out / "abilities.json")["by_key"]
    assert abilities["axe_coat_of_blood"] == {"key": "axe_coat_of_blood", "name": "Coat of Blood", "img": None,
                                              "legacy": True, "id": 9001}
    assert abilities["axe_gone_for_good"] == {"key": "axe_gone_for_good", "name": "Gone", "img": None, "legacy": True}
    assert (abilities["axe_culling_blade"]["id"], abilities["axe_culling_blade"]["ids"]) == (int(cull), [int(cull), 9002])
    assert "ids" not in abilities["axe_berserkers_call"]
    assert parser_ids(out)[str(cull)] == "axe_culling_blade" and parser_ids(out)["9001"] == "axe_coat_of_blood"


def parser_ids(catalog_dir: Path) -> dict[str, str]:
    """What backend/app/parser.py reads from an ability catalog: ``id`` -> key."""
    return {str(entry["id"]): key for key, entry in _load(catalog_dir / "abilities.json")["by_key"].items()
            if isinstance(entry.get("id"), int)}


# ---------------------------------------------------------------------------
# Chinese names and missing pictures
# ---------------------------------------------------------------------------

def test_catalogs_carry_the_games_chinese_names(catalogs):
    """The Chinese UI showed "Hoodwink", "Sentry Ward" and "Scurry": the
    catalogs only had the English names."""
    heroes = catalogs["heroes"]["by_id"]
    assert all(hero.get("name_zh") for hero in heroes.values())
    assert (heroes["123"]["name"], heroes["123"]["name_zh"]) == ("Hoodwink", "森海飞霞")
    items = catalogs["items"]["by_key"]
    assert items["ward_sentry"]["name_zh"] == "岗哨守卫" and items["recipe_dagon"]["name_zh"] == "图纸（达贡之神力）"
    current = [key for key, item in items.items() if item["name"] and not item.get("legacy") and not item.get("obsolete")]
    assert sum(1 for key in current if items[key].get("name_zh")) >= 0.95 * len(current)
    abilities = catalogs["abilities"]["by_key"]
    assert abilities["hoodwink_scurry"]["name_zh"] == "密林奔走"
    assert abilities["ability_capture"]["name_zh"] == "占领前哨" and "name_zh" not in abilities["special_bonus_attributes"]
    # a talent is a template in Valve's list: the number is the one of the English name
    assert abilities["special_bonus_unique_axe_4"]["name"] == "+40 Counter Helix Damage"
    assert abilities["special_bonus_unique_axe_4"]["name_zh"] == "+40 反击螺旋伤害"
    for name in ("heroes", "items", "abilities"):
        table = catalogs[name]["by_id" if name == "heroes" else "by_key"]
        named = [entry["name_zh"] for entry in table.values() if "name_zh" in entry]
        assert named and all(zh and "{" not in zh and "\n" not in zh for zh in named)
    # match payloads carry English names: the name tables stay English
    assert "森海飞霞" not in catalogs["heroes"]["name_to_id"] and "岗哨守卫" not in catalogs["items"]["name_to_key"]


def test_abilities_without_a_picture_have_no_image_url(catalogs):
    """Viper's Predator (an innate ability) had an image URL the CDN answers
    404 for: a failed request on the laning tab of every Viper game."""
    abilities = catalogs["abilities"]["by_key"]
    assert abilities["viper_predator"]["img"] is None
    assert abilities["axe_coat_of_blood"]["legacy"] and abilities["axe_coat_of_blood"]["img"] is None   # carried-over key
    assert abilities["viper_poison_attack"]["img"].endswith("/abilities/viper_poison_attack.png")


def test_chinese_names_are_read_from_valves_lists():
    feeds = {
        "herolist": {"result": {"data": {"heroes": [
            {"id": 2, "name": "npc_dota_hero_axe", "name_loc": "斧王", "name_english_loc": "Axe"},
            {"id": 3, "name": "npc_dota_hero_bane", "name_loc": "", "name_english_loc": "Bane"}]}}},
        "itemlist": {"result": {"data": {"itemabilities": [
            {"id": 1, "name": "item_blink", "name_loc": "闪烁匕首", "name_english_loc": "Blink Dagger"},
            {"id": 9, "name": "item_black_grimoire", "name_loc": "Black Grimoire", "name_english_loc": "Black Grimoire"},
            {"id": 7, "name": "item_recipe_dagon", "name_loc": "图纸（达贡之神力）\\n（说明）", "name_english_loc": "Dagon Recipe"}]}}},
        "abilitylist": {"result": {"data": {"itemabilities": [
            {"id": 5, "name": "axe_culling_blade", "name_loc": "淘汰之刃", "name_english_loc": "Culling Blade"},
            {"id": 6, "name": "special_bonus_unique_axe_4", "name_loc": "+{s:bonus_damage}  反击螺旋伤害",
             "name_english_loc": "+{s:bonus_damage} Counter Helix Damage"},
            "not a row"]}}},
    }
    assert generator.names_zh(feeds) == {
        "heroes": {"axe": "斧王"},
        "items": {"blink": "闪烁匕首", "recipe_dagon": "图纸（达贡之神力）"},     # untranslated left out, one line
        "abilities": {"axe_culling_blade": "淘汰之刃", "special_bonus_unique_axe_4": "+{s:bonus_damage} 反击螺旋伤害"},
        "ability_templates": {"special_bonus_unique_axe_4": "+{s:bonus_damage} Counter Helix Damage"},
    }
    assert generator.names_zh({}) == {"heroes": {}, "items": {}, "abilities": {}, "ability_templates": {}}


def test_a_chinese_talent_name_takes_the_numbers_of_the_english_one():
    values = data_manager.template_values_from_name
    assert values("+{s:bonus_damage} Counter Helix Damage", "+40 Counter Helix Damage") == {"bonus_damage": "40"}
    assert values("{s:a}% slow for {s:b}s", "60% slow for 1.5s") == {"a": "60", "b": "1.5"}
    assert values("+{s:value} Health", "+Health") == {}                      # the source has no number
    assert values("+{s:value} Health", "+{s:value} Health") == {}            # still a template
    assert values("{s:a} / {s:a} Damage", "10 / 20 Damage") == {}            # one token, two numbers
    assert values("Culling Blade", "Culling Blade") == {}
    fill = data_manager.filled_ability_name
    assert fill("+{s:bonus_damage} 反击螺旋伤害", None, {"bonus_damage": "40"}) == "+40 反击螺旋伤害"
    assert fill("+{s:value} 生命", [{"key": "value", "value": "250"}]) == "+250 生命"
    assert fill("+{s:value} 生命", []) == "" and fill("", None) == ""        # unknown number: no name, not "+X"
    assert fill("淘汰之刃", None) == "淘汰之刃"

    raw = {"abilities": {"special_bonus_unique_axe_4": {"dname": "+40 Counter Helix Damage"},
                         "special_bonus_unique_axe_5": {"dname": "+Berserker's Call Armor"}},
           "hero_abilities": {}, "ability_ids": {},
           "names_zh": {"abilities": {"special_bonus_unique_axe_4": "+{s:bonus_damage} 反击螺旋伤害",
                                      "special_bonus_unique_axe_5": "+{s:bonus_armor} 狂战士之吼护甲"},
                        "ability_templates": {"special_bonus_unique_axe_4": "+{s:bonus_damage} Counter Helix Damage",
                                              "special_bonus_unique_axe_5": "+{s:bonus_armor} Berserker's Call Armor"}}}
    built = generator._abilities(raw)["by_key"]
    assert built["special_bonus_unique_axe_4"]["name_zh"] == "+40 反击螺旋伤害"
    assert "name_zh" not in built["special_bonus_unique_axe_5"]              # readers fall back to the English name


def test_missing_ability_pictures_are_the_ones_the_cdn_answers_404_for():
    abilities = {
        "viper_predator": {"dname": "Predator", "img": "/apps/dota2/images/dota_react/abilities/viper_predator.png"},
        "viper_poison_attack": {"dname": "Poison Attack", "img": "/a/viper_poison_attack.png?t=1"},
        "axe_coat_of_blood": {"dname": "Coat of Blood", "img": "/a/axe_coat_of_blood.png"},
        "axe_culling_blade": {"dname": "Culling Blade", "img": "/a/axe_culling_blade.png"},
        "special_bonus_hp_250": {"dname": "+250 Health"},                   # no path: never asked
        "nameless": {"img": "/a/nameless.png"},
    }
    asked: list[str] = []

    def status(url: str) -> int:
        asked.append(url)
        return {"viper_predator": 404, "viper_poison_attack": 200}.get(url.rsplit("/", 1)[-1][:-4], 0)

    previous = {"by_key": {"axe_coat_of_blood": {"img": None}, "axe_culling_blade": {"img": "https://x/y.png"},
                           # keys the source no longer has are carried over: their pictures are asked for too
                           "viper_predator_old": {"img": "https://x/viper_predator.png", "legacy": True},
                           "old_talent": {"img": None, "legacy": True}, "older_talent": {"img": "", "legacy": True}}}
    # no clear answer (a timeout, a 503): what the previous catalog says stays
    assert generator.missing_ability_images(abilities, previous, status=status) == [
        "axe_coat_of_blood", "viper_predator", "viper_predator_old"]
    assert sorted(asked) == sorted([generator.CDN + path for path in (
        "/apps/dota2/images/dota_react/abilities/viper_predator.png", "/a/viper_poison_attack.png",
        "/a/axe_coat_of_blood.png", "/a/axe_culling_blade.png")] + ["https://x/viper_predator.png"])


def test_generator_carries_chinese_names_and_missing_pictures(tmp_path, data_set, capsys):
    source = _source_dir(tmp_path, data_set, _ITEMS_TXT)
    (source / "names_zh.json").write_text(json.dumps({
        "heroes": {"axe": "斧王", "not_in_the_source": "无"},
        "items": {"dagon": "达贡之神力", "recipe_dagon": "图纸（达贡之神力）", "heart": "恐鳌之心"},
        "abilities": {"axe_culling_blade": "淘汰之刃", "special_bonus_hp_250": "+{s:value} 生命"},
        "ability_templates": {"special_bonus_hp_250": "+{s:value} Health"}}), encoding="utf-8")
    (source / "missing_images.json").write_text(json.dumps(["axe_culling_blade", "axe_old", "not_anywhere"]), encoding="utf-8")
    out, fixture = tmp_path / "catalogs", tmp_path / "fixture.json.gz"
    out.mkdir()
    (out / "abilities.json").write_text(json.dumps({"by_key": {      # a key only the previous catalog has
        "axe_old": {"key": "axe_old", "name": "Old", "img": "https://x/axe_old.png"}}, "name_to_key": {"Old": "axe_old"}}),
        encoding="utf-8")
    args = ["--source", str(source), "--out", str(out), "--fallback", str(tmp_path / "fallback"), "--fixture", str(fixture)]
    assert generator.main(args) == 0 and generator.main(args + ["--check"]) == 0

    hero = _load(out / "heroes.json")["by_id"]["2"]
    assert hero["name_zh"] == "斧王" and list(hero)[:3] == ["id", "name", "name_zh"]
    items = _load(out / "items.json")
    assert items["by_key"]["dagon"]["name_zh"] == "达贡之神力" and "name_zh" not in items["by_key"]["black_king_bar"]
    assert items["by_key"]["recipe_dagon"]["name_zh"] == "图纸（达贡之神力）" and "达贡之神力" not in items["name_to_key"]
    abilities = _load(out / "abilities.json")["by_key"]
    assert abilities["axe_culling_blade"]["name_zh"] == "淘汰之刃" and abilities["axe_culling_blade"]["img"] is None
    assert abilities["axe_berserkers_call"]["img"] and "name_zh" not in abilities["axe_berserkers_call"]
    assert (abilities["special_bonus_hp_250"]["name"], abilities["special_bonus_hp_250"]["name_zh"]) == ("+250 Health", "+250 生命")
    assert abilities["axe_old"] == {"key": "axe_old", "name": "Old", "img": None, "legacy": True}
    # the fixture holds what the catalogs were built from, and nothing the source does not have
    stored = generator.read_fixture(fixture)
    assert stored["names_zh"]["heroes"] == {"axe": "斧王"} and stored["missing_images"] == ["axe_culling_blade"]
    assert generator.current_part(_load(out / "abilities.json"), "by_key") == generator.build(stored)["abilities"]

    # Valve's lists are unreachable on the next run: names and missing pictures stay
    (source / "names_zh.json").unlink()
    (source / "missing_images.json").unlink()
    capsys.readouterr()
    assert generator.main(args + ["--check"]) == 0
    assert "keeping the ones of the previous catalogs" in capsys.readouterr().out
    assert generator.main(args) == 0
    assert _load(out / "abilities.json")["by_key"]["axe_culling_blade"] == abilities["axe_culling_blade"]
    assert _load(out / "abilities.json")["by_key"]["special_bonus_hp_250"]["name_zh"] == "+250 生命"
    assert _load(out / "heroes.json")["by_id"]["2"]["name_zh"] == "斧王"


# ---------------------------------------------------------------------------
# Chat wheel (Valve's scripts and localisation files)
# ---------------------------------------------------------------------------

# The message ids of the nine replays the parser was checked on (2 of 7.40 / 7.41b, 7 of 7.41f,
# one of them Turbo): general phrases, a spray, hero voice lines, voice lines of casters.
CHAT_WHEEL_IDS_SEEN = (
    0, 2, 3, 7, 9, 12, 49, 54, 61, 62, 63, 65, 68, 69, 71, 76, 801, 802, 805, 818, 819, 822, 823,
    12001, 12002, 12004, 15001, 20004, 22001, 37001, 37002, 37003, 37006, 47001, 49002, 49004, 62001, 62002,
    79001, 83001, 86001, 88001, 90001, 91001, 91002, 105001, 107001, 107002, 112001, 112002, 112003, 112004,
    401007, 401016, 401829, 401914, 401985, 402090)

_CHAT_WHEEL_SCRIPT = '''
"chat_wheel"
{
    "messages"
    {
        // "Retired" { "message_id" "9" "label" "gone" }
        "Ok"
        {
            "message_id"    "0"
            "label"         "#dota_chatwheel_label_Ok"
            "message"       "#dota_chatwheel_message_Ok"
            "always_unlocked"   "1"
        }
        "MissingHero"
        {
            "message_id"    "71"
            "label"         "#dota_chatwheel_label_MissingHero"
            "message"       "#dota_chatwheel_message_MissingHero"
        }
        "Caster_Chinese"
        {
            "message_id"    "802"
            "label"         "喂喂喂？"
            "message"       "喂喂喂？ Hello？队友呢？"
            "translation"   "#Dota_Chatwheel_hello_team"
            "sound"         "soundboard.hello_team"
            "all_chat"      "1"
        }
        "Caster_English"
        {
            "message_id"    "801"
            "label"         "Its Too Good"
            "message"       "Omg its \\"too good\\"..."
            "sound"         "soundboard.too_good"
        }
        "Spray"
        {
            "message_id"    "805"
            "image"         "sprays/dark_carnival/dc_beastmaster_icon.png"
        }
        "Old_Fan_Line"
        {
            "message_id"    "140116"
            "label"         "#dota_chatwheel_teamfandom_2_36_voiceline_140116_label"
            "message"       "#dota_chatwheel_teamfandom_2_36_voiceline_140116_message"
            "sound"         "teamfandom.2.36.140116"
        }
        "Caption_Only"
        {
            "message_id"    "111"
            "label"         "#dota_chatwheel_label_Day_Night"
            "message"       "#dota_chatwheel_message_Day_Night"
        }
        "No_Id" { "label" "nothing" }
    }
}
'''
_CHAT_WHEEL_TOKENS = {
    "english": '''"lang" { "Language" "English" "Tokens" {
        "dota_chatwheel_label_Ok"       "Okay"
        "dota_chatwheel_message_Ok"     "Okay."
        "dota_chatwheel_message_MissingHero:f"  "{hero_name} {hero_name, gender, other{is missing!}}"
        "Dota_Chatwheel_hello_team"     "Hihihi? Hello? Team?"
        "dota_chatwheel_label_Day_Night"    "Day/Night Cycle"
        "dota_chatwheel_teamfandom_13_2163_voiceline_143035_label"      "Referee!"
        "dota_chatwheel_teamfandom_13_2163_voiceline_143035_message"    "Referee! Referee!"
        "unrelated_token"               "never read"
    } }''',
    "schinese": '''"lang" { "Language" "Schinese" "Tokens" {
        "dota_chatwheel_label_Ok"       "好的"
        "dota_chatwheel_message_Ok"     "好的。"
        "dota_chatwheel_message_MissingHero:f"  "{hero_name}{hero_name, gender, other{不见了！}}"
        "Dota_Chatwheel_hello_team"     " "
        "dota_chatwheel_label_Day_Night"    "白昼/夜晚循环"
        "dota_chatwheel_teamfandom_13_2163_voiceline_143035_message"    "裁判！裁判！"
    } }''',
}


def _chat_wheel_input() -> dict:
    tokens = {language: generator.localisation_tokens(text) for language, text in _CHAT_WHEEL_TOKENS.items()}
    return generator.chat_wheel_input([_CHAT_WHEEL_SCRIPT], tokens)


def test_committed_chat_wheel_is_what_the_generator_builds_from_the_data_set(catalogs, data_set):
    committed = _load(CATALOG_DIR / "chat_wheel.json")
    built = generator.build_chat_wheel(data_set["chat_wheel"])
    assert {key: value for key, value in committed.items() if key != "meta"} == built, (
        f"chat_wheel.json lags the data set: {REGENERATE}")
    assert list(committed["by_id"]) == list(built["by_id"]) == sorted(built["by_id"], key=int)
    # one stamp with the other catalogs, and its own source: Valve's files at one revision of the mirror
    meta = committed["meta"]
    assert meta["patch"] == catalogs["heroes"]["meta"]["patch"] and data_manager._parse_utc(meta["generated_at"])
    assert meta["source"] == generator.CHAT_WHEEL_REPO == data_set["chat_wheel"]["source"]["source"]
    assert re.fullmatch(r"[0-9a-f]{12}", meta["source_revision"]) and data_manager._parse_utc(meta["source_date"])
    # a phrase that changed upstream is noticed
    stale = json.loads(json.dumps(committed["by_id"]))
    stale["2"]["zh"] = "回来！"
    assert stale != built["by_id"]


def test_chat_wheel_has_every_line_in_both_languages(data_set):
    by_id = _load(CATALOG_DIR / "chat_wheel.json")["by_id"]
    assert len(by_id) > 3000
    missing = [mid for mid in CHAT_WHEEL_IDS_SEEN if str(mid) not in by_id]
    assert not missing, f"message ids seen in real replays are not in the catalog: {missing}"
    lines = {mid: entry for mid, entry in by_id.items() if not entry.get("spray")}
    assert all(set(entry) <= {"en", "zh", "sound"} and entry["en"] and entry["zh"] for entry in lines.values())
    assert all(entry == {"spray": True} for mid, entry in by_id.items() if mid not in lines) and len(lines) > 2900
    # display text: no token, no template, on one line
    for mid, entry in lines.items():
        for text in (entry["en"], entry["zh"]):
            assert not text.startswith("#") and "\n" not in text and text == text.strip(), (mid, text)
            assert "{" not in text.replace("{hero}", ""), (mid, text)
    # Valve translates most of them (voice lines of casters often are not, for want of a translation)
    assert sum(1 for entry in lines.values() if entry["zh"] != entry["en"]) > 2000
    assert sum(1 for entry in lines.values() if CJK_TEXT.search(entry["zh"])) > 2000
    assert by_id["0"] == {"en": "Okay.", "zh": "好的。"} and by_id["805"] == {"spray": True}
    assert by_id["12001"]["sound"] is True and "sound" not in by_id["2"]
    # the two lines that name a hero keep it as {hero}
    assert {mid for mid, entry in lines.items() if "{hero}" in entry["en"]} == {"71", "72"}
    assert by_id["71"] == {"en": "{hero} is missing!", "zh": "{hero}不见了！"}
    # a line spoken in Chinese shows as spoken there, with its translation in English
    assert by_id["402090"]["zh"] == "每一把都当最后一把，然后就也不要怕输" and by_id["402090"]["en"].startswith("Play every game")
    # the fixture holds no more than the catalog was built from
    wheel = data_set["chat_wheel"]
    named = {str(entry[field])[1:].lower() for entry in wheel["messages"].values() for field in generator.CHAT_WHEEL_TEXT_FIELDS
             if str(entry.get(field) or "").startswith("#")}
    assert set(wheel["tokens"]) == {"english", "schinese"}
    assert all(set(table) <= named for table in wheel["tokens"].values())


CJK_TEXT = re.compile(r"[一-鿿]")


def test_chat_wheel_scripts_and_tokens_are_read_like_valve_writes_them():
    messages = generator.chat_wheel_messages(_CHAT_WHEEL_SCRIPT)
    assert list(messages) == ["0", "71", "802", "801", "805", "140116", "111"]      # a comment and a block without id are none
    assert messages["0"] == {"label": "#dota_chatwheel_label_Ok", "message": "#dota_chatwheel_message_Ok"}
    assert messages["802"] == {"label": "喂喂喂？", "message": "喂喂喂？ Hello？队友呢？",
                               "translation": "#Dota_Chatwheel_hello_team", "sound": 1}
    assert messages["805"] == {"image": 1}
    tokens = generator.localisation_tokens(_CHAT_WHEEL_TOKENS["english"])
    # lower case, and without the suffix that says how the game formats the text
    assert tokens["dota_chatwheel_message_missinghero"].startswith("{hero_name}")
    assert tokens["dota_chatwheel_hello_team"] == "Hihihi? Hello? Team?" and tokens["language"] == "English"
    pairs = list(generator.keyvalues('"a" { "b" "1" // "c" "2"\n "d" { "e" "x \\"y\\"" } } "f" "3"'))
    assert pairs == [(("a",), "b", "1"), (("a", "d"), "e", 'x \\"y\\"'), ((), "f", "3")]

    assert generator.chat_wheel_phrase("{hero_name} {hero_name, gender, other{is missing!}}") == "{hero} is missing!"
    assert generator.chat_wheel_phrase("{hero_name}{hero_name, gender, male{他} female{她} other{不见了！}}") == "{hero}不见了！"
    assert generator.chat_wheel_phrase('  Omg its \\"too good\\"...\\nreally ') == 'Omg its "too good"... really'
    for no_phrase in ("", " ", None, "#dota_chatwheel_message_Ok", "{s:value} wards", "{count, plural, one{a} other{b}}"):
        assert generator.chat_wheel_phrase(no_phrase) == ""


def test_chat_wheel_catalog_takes_the_line_a_message_shows_in_chat():
    source = _chat_wheel_input()
    assert generator.build_chat_wheel(source) == {"by_id": {
        "0": {"en": "Okay.", "zh": "好的。"},                                           # the message, not the caption
        "71": {"en": "{hero} is missing!", "zh": "{hero}不见了！"},
        # no text for the message: the caption on the wheel
        "111": {"en": "Day/Night Cycle", "zh": "白昼/夜晚循环"},
        # a literal line without a translation is the same in both languages
        "801": {"en": 'Omg its "too good"...', "zh": 'Omg its "too good"...', "sound": True},
        # a line spoken in Chinese: the English translation, and the line itself where Valve left
        # the translation blank
        "802": {"en": "Hihihi? Hello? Team?", "zh": "喂喂喂？ Hello？队友呢？", "sound": True},
        "805": {"spray": True},
        # fan content of the teams: from the localisation file, the id is in the token
        "143035": {"en": "Referee! Referee!", "zh": "裁判！裁判！", "sound": True},
    }}                                                                               # 140116: no text left in the game
    # reduced to what is read: fields up to the first with a text in both languages, tokens that are named
    assert source["messages"]["0"] == {"message": "#dota_chatwheel_message_Ok"}
    assert source["messages"]["802"] == {"translation": "#Dota_Chatwheel_hello_team",
                                         "message": "喂喂喂？ Hello？队友呢？", "sound": 1}
    assert source["messages"]["805"] == {"image": 1} and list(source["messages"]) == sorted(source["messages"], key=int)
    assert "unrelated_token" not in source["tokens"]["english"] and "dota_chatwheel_label_ok" not in source["tokens"]["english"]
    assert "dota_chatwheel_label_day_night" in source["tokens"]["schinese"]
    # an earlier script wins for an id; without Chinese text the English one is shown
    second = _CHAT_WHEEL_SCRIPT.replace('"#dota_chatwheel_message_Ok"', '"Sure."')
    tokens = {"english": generator.localisation_tokens(_CHAT_WHEEL_TOKENS["english"]), "schinese": {}}
    english_only = generator.build_chat_wheel(generator.chat_wheel_input([_CHAT_WHEEL_SCRIPT, second], tokens))["by_id"]
    assert english_only["0"] == {"en": "Okay.", "zh": "Okay."} and english_only["143035"]["zh"] == "Referee! Referee!"
    assert english_only["802"] == {"en": "Hihihi? Hello? Team?", "zh": "喂喂喂？ Hello？队友呢？", "sound": True}


def test_generator_writes_and_checks_the_chat_wheel(tmp_path, data_set, capsys):
    source = _source_dir(tmp_path, data_set, _ITEMS_TXT)
    out, fixture = tmp_path / "catalogs", tmp_path / "fixture.json.gz"
    args = ["--source", str(source), "--out", str(out), "--fallback", str(tmp_path / "fallback"),
            "--fixture", str(fixture), "--patch", "7.41f"]
    # no chat wheel files and none from an earlier run: the catalog is simply not written
    assert generator.main(args) == 0 and generator.main(args + ["--check"]) == 0
    assert not (out / "chat_wheel.json").exists() and "chat_wheel" not in generator.read_fixture(fixture)

    wheel = {"source": {"source": "dotabuff/d2vpkr", "source_revision": "0123456789ab",
                        "source_date": "2026-10-10T06:00:18Z"}, **_chat_wheel_input()}
    (source / "chat_wheel.json").write_text(json.dumps(wheel, ensure_ascii=False), encoding="utf-8")
    assert generator.main(args + ["--check"]) == 1 and not (out / "chat_wheel.json").exists()
    assert generator.main(args) == 0 and generator.main(args + ["--check"]) == 0
    written = _load(out / "chat_wheel.json")
    assert written["by_id"] == generator.build_chat_wheel(wheel)["by_id"]
    assert written["meta"]["patch"] == "7.41f" == _load(out / "items.json")["meta"]["patch"]
    assert (written["meta"]["source"], written["meta"]["source_revision"]) == ("dotabuff/d2vpkr", "0123456789ab")
    text = (out / "chat_wheel.json").read_text(encoding="utf-8")
    assert '"802": {"en": "Hihihi? Hello? Team?", "zh": "喂喂喂？ Hello？队友呢？", "sound": true},\n' in text    # a line per entry
    # the fixture rebuilds it
    assert generator.build_chat_wheel(generator.read_fixture(fixture)["chat_wheel"])["by_id"] == written["by_id"]

    # a phrase Valve changed is noticed and carried through
    wheel["tokens"]["schinese"]["dota_chatwheel_message_ok"] = "好。"
    (source / "chat_wheel.json").write_text(json.dumps(wheel, ensure_ascii=False), encoding="utf-8")
    capsys.readouterr()
    assert generator.main(args + ["--check"]) == 1
    assert "chat_wheel.json: 7 entries, CHANGED" in capsys.readouterr().out
    assert generator.main(args) == 0 and _load(out / "chat_wheel.json")["by_id"]["0"]["zh"] == "好。"

    # the mirror is unreachable on the next run: the phrases of the previous one stay
    (source / "chat_wheel.json").unlink()
    capsys.readouterr()
    assert generator.main(args + ["--check"]) == 0
    assert "keeping the phrases of the previous run" in capsys.readouterr().out
    assert generator.main(args) == 0 and _load(out / "chat_wheel.json")["by_id"]["0"]["zh"] == "好。"


def test_the_chat_wheel_download_keeps_the_stored_revision_while_the_phrases_are_the_same(monkeypatch):
    """The mirror gets a commit with every client build; the chat wheel changes a few times a
    year. The stamp must not move, and the fixture must not go stale, without a new phrase."""
    commit = "a" * 40
    files = {
        generator.CHAT_WHEEL_COMMIT_URL: json.dumps({"sha": commit, "commit": {"committer": {"date": "2026-10-12T06:00:00Z"}}}),
        generator.CHAT_WHEEL_LISTING_URL.format(path=generator.CHAT_WHEEL_DIR, commit=commit): json.dumps([
            {"name": "stickers_chat_wheel_13.txt", "path": "dota/scripts/chat_wheels/stickers_chat_wheel_13.txt"},
            {"name": "teamfandom_chat_wheel_13_2163.txt", "path": "dota/scripts/chat_wheels/teamfandom_chat_wheel_13_2163.txt"}]),
    }
    raw = generator.CHAT_WHEEL_FILE_URL
    files[raw.format(commit=commit, path="dota/scripts/chat_wheel.txt")] = _CHAT_WHEEL_SCRIPT
    files[raw.format(commit=commit, path="dota/scripts/chat_wheel_heroes.txt")] = (
        '"chat_wheel" { "hero_messages" { "12" { "pl_laugh" { "message_id" "12001" "message" "Ha!" "sound" "x" } } } }')
    files[raw.format(commit=commit, path="dota/scripts/chat_wheels/stickers_chat_wheel_13.txt")] = (
        '"chat_wheel" { "messages" { "talent" { "message_id" "402090" "message" "每一把" "sound" "x" } } }')
    for language, text in _CHAT_WHEEL_TOKENS.items():
        for name in generator.CHAT_WHEEL_TOKEN_FILES:
            path = generator.CHAT_WHEEL_TOKENS.format(name=name, language=language)
            files[raw.format(commit=commit, path=path)] = text if name == "dota" else '"x" { }'
    asked: list[str] = []

    def download(url, timeout=60):
        asked.append(url)
        return files[url]                       # a KeyError is a file the generator must not ask for

    monkeypatch.setattr(generator, "_download_text", download)
    first = generator._download_chat_wheel(None)
    assert first["source"] == {"source": "dotabuff/d2vpkr", "source_revision": "a" * 12, "source_date": "2026-10-12T06:00:00Z"}
    by_id = generator.build_chat_wheel(first)["by_id"]
    assert by_id["12001"] == {"en": "Ha!", "zh": "Ha!", "sound": True} and by_id["402090"]["zh"] == "每一把"
    assert by_id["0"] == {"en": "Okay.", "zh": "好的。"} and len(asked) == len(files)
    # the per-team fan content is not downloaded script by script
    assert not any("teamfandom_chat_wheel_" in url for url in asked)

    stored = {**first, "source": {**first["source"], "source_revision": "0123456789ab"}}
    assert generator._download_chat_wheel(stored) is stored             # same phrases: the stored revision stays
    files[raw.format(commit=commit, path="dota/scripts/chat_wheel_heroes.txt")] = (
        '"chat_wheel" { "hero_messages" { "12" { "pl_laugh" { "message_id" "12001" "message" "Ha ha!" "sound" "x" } } } }')
    assert generator._download_chat_wheel(stored)["source"]["source_revision"] == "a" * 12     # a new phrase: the new one
    # an unreachable mirror, or an answer that is not what was asked for, is no data
    del files[generator.CHAT_WHEEL_COMMIT_URL]
    assert generator._download_chat_wheel(stored) is None
    files[generator.CHAT_WHEEL_COMMIT_URL] = json.dumps({"sha": "not a commit"})
    assert generator._download_chat_wheel(stored) is None


def test_the_parser_reads_the_chat_wheel_catalog():
    assert parser.chat_wheel_entry("chatwheel_2") == {"en": "Get Back!", "zh": "撤退！"}
    assert parser.chat_wheel_entry(2) == parser.chat_wheel_entry("2") == parser.chat_wheel_entry("chatwheel_002")
    assert parser.chat_wheel_entry("chatwheel_805") == {"spray": True}
    for unknown in ("chatwheel_99999999", "chatwheel_", "chatwheel_-1", "chatwheel_x", "", None, "２"):
        assert parser.chat_wheel_entry(unknown) is None
    assert parser.hero_names(22) == ("Zeus", "宙斯") and parser.hero_names("22") == ("Zeus", "宙斯")
    assert parser.hero_names(0) == parser.hero_names(None) == parser.hero_names("x") == ("", "")
