"""Regenerate the committed stand-ins for fetched game data.

The server fetches hero / item / ability data at runtime (``data_manager``),
but three copies of it are committed because they must work without a
network. This script rebuilds all of them from the same dotaconstants files,
taken from the same sources in the same order
(``data_manager.CONSTANTS_SOURCES``: master on GitHub, its jsDelivr mirror,
then the OpenDota API):

1. ``frontend/static/data/{heroes,items,abilities}.json`` — the catalogs the
   frontend, ``backend/app/parser.py`` and the data layer read: ids, keys,
   display names, icons, item costs.
2. ``backend/data/fallback/{ability,item}_stats.json`` — the numbers, and
   for items the description, the AI coach uses before the first successful
   fetch. Only the listed entries are refreshed; adding one (with its
   ``category``) is an editorial decision.
3. ``backend/tests/fixtures/gamedata/constants.json.gz`` — the input reduced
   to the fields this script reads, plus the numbers every item and hero
   ability states. ``backend/tests/test_gamedata_catalogs.py`` rebuilds the
   catalogs from it, so a change of the generator, of the fixture or of a
   catalog that is not carried through fails the suite, and it checks the
   numbers quoted in ``backend/data/editorial`` against it.

A fourth catalog comes from another source, Valve's own game files:
``frontend/static/data/chat_wheel.json``, the phrases of the chat wheel in
English and Simplified Chinese by message id (see "Chat wheel" below). Its
input is part of the same fixture and it follows the same rules.

Usage (from the repository root)::

    python -m backend.scripts.gen_static_catalogs           # download, rewrite all three
    python -m backend.scripts.gen_static_catalogs --check   # write nothing; exit 1 when they lag
    python -m backend.scripts.gen_static_catalogs --source DIR   # dotaconstants build files in DIR

Run it after every game patch, major or letter (item costs change in letter
patches), bump ``DEFAULT_PATCH_LABEL`` / ``DEFAULT_PATCH_RELEASED`` in
``backend/app/ai/data_manager.py`` to the stamped version (for a new major
version also review the rule values in ``backend/app/game_constants.py`` and
bump its ``RULES_PATCH``: a test fails until then), run the suite
(editorial texts that quote a changed number fail there), rebuild the
committed retrieval vectors (``python -m backend.scripts.rag build --static``:
item texts are part of that corpus) and commit the result. The catalogs and
the fallback files carry one version stamp: when one of them changes, or the
version does, all are written again. ``--check`` needs the network and is
meant for a scheduled job; the test suite itself is network-free.

Catalog shape (unchanged; ``meta`` and the optional fields are additions):

- heroes.json: ``{"by_id": {"<id>": {id, name[, name_zh], slug, img, icon,
  attr, roles, attack_type, day_vision, night_vision}}, "name_to_id",
  "cdn_base", "meta"}`` — two heroes carry the game's name instead of the
  source's (``HERO_NAMES``); ``name_to_id`` knows both spellings.
- items.json: ``{"by_key": {"<key>": {id, key, name[, name_zh], img, cost[,
  tier][, qual][, obsolete][, legacy]}}, "name_to_key", "cdn_base", "meta"}``
  — recipes included ("Abyssal Blade Recipe"); a display name shared by
  several keys maps to the base item (Dagon -> dagon), which also comes first
  among them in ``by_key``.
- abilities.json: ``{"by_key": {"<key>": {key, name[, name_zh], img[, id][,
  ids][, legacy]}}, "name_to_key", "cdn_base", "meta"}`` — every named ability
  and talent; no name keeps a ``{s:...}`` template token; a name shared by
  several keys maps to the hero ability; ``ids`` only where the source lists
  several; ``img`` is null for talents and for abilities the image CDN has no
  picture of (most innate abilities).

``name_zh`` is the name in the game's Simplified Chinese localisation, read
from Valve's own lists (``NAMES_ZH_FEEDS``; dotaconstants is English only).
It is left out where Valve has none, or where it is a template whose number
the data does not have: readers fall back to ``name``. The ``name_to_*``
tables stay English, like every name in a match payload. When Valve's lists
cannot be reached the names of the previous run are kept, and likewise the
pictures found missing (``missing_ability_images``).

- chat_wheel.json: ``{"by_id": {"<message id>": {en, zh[, sound]} |
  {"spray": true}}, "meta"}`` — the line a chat-wheel message shows in chat,
  in both languages (``zh`` is the English text where Valve has no Chinese
  one); ``sound`` marks a line that is also heard; a spray has a picture and
  no text. The two lines that name a hero keep it as ``{hero}``
  ("{hero} is missing!"). The reader is ``backend/app/parser.py``
  (``chat_wheel_entry``); the analysis puts the phrase into the payload.

Keys of the previous catalog that the source no longer has are kept with
``"legacy": true``: old replays still carry removed items and abilities.
dotaconstants also keeps items the game has removed (Eternal Shroud, Wraith
Pact); those are told from Valve's ``items.txt`` (``IsObsolete``, read from
the SteamDatabase GameTracking mirror) and flagged ``"obsolete": true``. They
never take a display name from a live item, and entries of the fallback
files for them are dropped. When that mirror cannot be reached the flags of
the previous catalog are kept.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CDN = "https://cdn.akamai.steamstatic.com"
_IMAGES = "/apps/dota2/images/dota_react"

CATALOG_DIR = PROJECT_ROOT / "frontend" / "static" / "data"
FALLBACK_DIR = PROJECT_ROOT / "backend" / "data" / "fallback"
FIXTURE_PATH = PROJECT_ROOT / "backend" / "tests" / "fixtures" / "gamedata" / "constants.json.gz"

# The dotaconstants files the three outputs are built from.
SOURCE_FILES = ("heroes", "items", "abilities", "hero_abilities", "ability_ids")
# Valve's item script (KeyValues text), for the IsObsolete flags.
ITEMS_TXT = "game/dota/pak01_dir/scripts/npc/items.txt"
ITEMS_TXT_URLS = (
    "https://raw.githubusercontent.com/SteamDatabase/GameTracking-Dota2/master/" + ITEMS_TXT,
    "https://cdn.jsdelivr.net/gh/SteamDatabase/GameTracking-Dota2@master/" + ITEMS_TXT,
)
# Valve's lists of heroes, items and abilities in Simplified Chinese:
# (catalog, feed, key of the list in the answer, prefix the feed puts in
# front of our keys).
NAMES_ZH_URL = "https://www.dota2.com/datafeed/{feed}?language=schinese"
NAMES_ZH_FEEDS = (
    ("heroes", "herolist", "heroes", "npc_dota_hero_"),
    ("items", "itemlist", "itemabilities", "item_"),
    ("abilities", "abilitylist", "itemabilities", ""),
)
# (catalog file, its table)
CATALOGS = (("heroes", "by_id"), ("items", "by_key"), ("abilities", "by_key"))

# The chat wheel: Valve's scripts name the messages by id, its localisation
# files have their text. Both are read from the mirror of the game's files
# that Dotabuff keeps on GitHub (one commit per client build), at one commit.
CHAT_WHEEL_REPO = "dotabuff/d2vpkr"
CHAT_WHEEL_COMMIT_URL = f"https://api.github.com/repos/{CHAT_WHEEL_REPO}/commits/master"
CHAT_WHEEL_LISTING_URL = f"https://api.github.com/repos/{CHAT_WHEEL_REPO}/contents/{{path}}?ref={{commit}}"
CHAT_WHEEL_FILE_URL = f"https://raw.githubusercontent.com/{CHAT_WHEEL_REPO}/{{commit}}/{{path}}"
# The general wheel and the hero voice lines ...
CHAT_WHEEL_SCRIPTS = ("dota/scripts/chat_wheel.txt", "dota/scripts/chat_wheel_heroes.txt")
# ... and the voice lines of casters and players: the files of this directory
# whose names start like this, one per season. (The fan content of the teams
# is hundreds of small files there; its lines are taken from the localisation
# file instead, which names each by its message id.)
CHAT_WHEEL_DIR = "dota/scripts/chat_wheels"
CHAT_WHEEL_DIR_FILES = ("stickers_chat_wheel_", "ti2021_casters_chat_wheel")
CHAT_WHEEL_TOKENS = "dota/resource/localization/{name}_{language}.txt"
CHAT_WHEEL_TOKEN_FILES = ("dota", "hero_chat_wheel", "teamfandom")
# catalog language -> Valve's name of it
CHAT_WHEEL_LANGUAGES = {"en": "english", "zh": "schinese"}
# What a message says, in the order the catalog takes it: the translation of
# a line spoken in another language, the line itself, the caption on the wheel.
CHAT_WHEEL_TEXT_FIELDS = ("translation", "message", "label")

_NON_ALNUM = re.compile(r"[^a-z0-9]")


def _squash(text: Any) -> str:
    return _NON_ALNUM.sub("", str(text or "").lower())


def _image(path: Any) -> str | None:
    """CDN URL of an upstream image path ("/apps/...png?t=123"); None without one."""
    return CDN + str(path).split("?")[0] if path else None


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

_KV_TOKEN = re.compile(r'//[^\n]*|"((?:[^"\\]|\\.)*)"|([{}])')


def obsolete_items(items_txt: str) -> list[str]:
    """Item keys (without ``item_``) that Valve's items.txt flags ``IsObsolete``.

    A scan of the KeyValues text: the blocks one level below the root are
    the items, ``"IsObsolete" "1"`` directly inside one marks it.
    """
    found: list[str] = []
    depth = 0
    pending: str | None = None  # a key waiting for its value or block
    block: str | None = None    # the item block being read
    for match in _KV_TOKEN.finditer(items_txt):
        text, brace = match.group(1), match.group(2)
        if brace == "{":
            depth += 1
            if depth == 2:
                block = pending
            pending = None
        elif brace == "}":
            depth -= 1
            pending = None
        elif text is not None:
            if pending is None:
                pending = text
                continue
            if depth == 2 and pending == "IsObsolete" and text == "1" and block:
                found.append(block[len("item_"):] if block.startswith("item_") else block)
            pending = None
    return sorted(set(found))


def _download_items_txt() -> str | None:
    for url in ITEMS_TXT_URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "DotaAnalyzer/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"warning: {url} is unreachable ({exc})")
    return None


def names_zh(feeds: dict[str, Any]) -> dict[str, dict[str, str]]:
    """``{"heroes": {slug: name}, "items": {key: name}, "abilities": {key:
    name}, "ability_templates": {key: English name}}`` from the answers of
    Valve's lists (*feeds*: feed name -> JSON).

    Talent names are still templates here ("+{s:bonus_damage} 反击螺旋伤害");
    ``ability_templates`` has the English template of each of them
    ("+{s:bonus_damage} Counter Helix Damage"), which tells where the number
    stands in the name dotaconstants resolved ("+40 Counter Helix Damage").
    An entry without a Chinese name, or whose Chinese name is the English
    one, is left out.
    """
    from backend.app.ai.data_manager import clean_item_name

    names: dict[str, dict[str, str]] = {"ability_templates": {}}
    for catalog, feed, table, prefix in NAMES_ZH_FEEDS:
        answer = feeds.get(feed)
        rows = (((answer or {}).get("result") or {}).get("data") or {}).get(table) if isinstance(answer, dict) else None
        found: dict[str, str] = {}
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            key, name = str(row.get("name") or ""), str(row.get("name_loc") or "")
            english = " ".join(str(row.get("name_english_loc") or "").split())
            name = clean_item_name(name) if catalog == "items" else " ".join(name.split())
            if not (key.startswith(prefix) and len(key) > len(prefix)) or not name or name == english:
                continue
            found[key[len(prefix):]] = name
            if catalog == "abilities" and "{s:" in name and english:
                names["ability_templates"][key] = english
        names[catalog] = found
    return names


def _download_names_zh() -> dict[str, dict[str, str]] | None:
    """Valve's Chinese names; None when a list is unreachable or empty."""
    feeds: dict[str, Any] = {}
    for _catalog, feed, _table, _prefix in NAMES_ZH_FEEDS:
        url = NAMES_ZH_URL.format(feed=feed)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "DotaAnalyzer/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                feeds[feed] = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            print(f"warning: {url} is unreachable ({exc})")
            return None
    names = names_zh(feeds)
    return names if all(names.get(catalog) for catalog, *_rest in NAMES_ZH_FEEDS) else None


def _previous_names_zh(previous: dict[str, dict]) -> dict[str, dict[str, str]]:
    """The Chinese names the previous catalogs carry, in the shape of :func:`names_zh`."""
    names: dict[str, dict[str, str]] = {}
    for catalog, table in CATALOGS:
        entries = ((previous.get(catalog) or {}).get(table) or {}).values()
        names[catalog] = {str(entry["slug" if catalog == "heroes" else "key"]): entry["name_zh"]
                          for entry in entries if isinstance(entry, dict) and entry.get("name_zh")}
    return names


def _image_status(url: str) -> int:
    """HTTP status of a HEAD request for *url* (0: no answer after a second try)."""
    for _attempt in range(2):
        try:
            request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "DotaAnalyzer/1.0"})
            with urllib.request.urlopen(request, timeout=20) as response:
                return int(response.status)
        except urllib.error.HTTPError as exc:
            return int(exc.code)
        except Exception:
            continue
    return 0


def missing_ability_images(abilities: dict[str, Any], previous: dict | None = None, status=_image_status) -> list[str]:
    """Keys of the named abilities whose picture the image CDN does not have.

    dotaconstants gives every ability an image path, but the CDN has no
    picture for most innate abilities (Viper's Predator): the page would
    request a file that is not there. Each path is asked for once; only a
    404 counts as missing, and a path that got no clear answer keeps what
    the *previous* ability catalog says about it. The keys only that
    catalog still has (they are carried over, see :func:`merge_legacy`) are
    asked for as well.
    """
    from concurrent.futures import ThreadPoolExecutor

    before = {key: entry for key, entry in ((previous or {}).get("by_key") or {}).items() if isinstance(entry, dict)}
    urls = {key: _image(ability.get("img")) for key, ability in abilities.items()
            if isinstance(ability, dict) and ability.get("dname") and ability.get("img")}
    urls.update({key: entry["img"] for key, entry in before.items()
                 if key not in urls and isinstance(entry.get("img"), str) and entry["img"]
                 and not (isinstance(abilities.get(key), dict) and abilities[key].get("dname"))})
    with ThreadPoolExecutor(max_workers=16) as pool:
        statuses = dict(zip(urls, pool.map(status, urls.values())))
    unclear = [key for key, code in statuses.items() if code not in (200, 404)]
    if unclear:
        print(f"warning: no clear answer for {len(unclear)} ability pictures; keeping the previous catalog's word on them")
    return sorted(key for key, code in statuses.items()
                  if code == 404 or (code != 200 and key in before and before[key].get("img") is None))


def load_source(directory: Path | None, previous_items: dict | None = None,
                previous: dict[str, dict] | None = None,
                stored: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    """The raw input and what it is: ``source`` (an id of
    ``data_manager.CONSTANTS_SOURCES``, or ``files`` for *directory*) and,
    for a download, the revision, its date and the game version it can
    reflect (``data_manager.download_constants_snapshot``).

    The constants files by name plus what no dotaconstants file says:

    - ``obsolete_items``: the keys Valve's items.txt flags
      (``<directory>/items.txt`` when reading files). Without that file — or
      when it names no obsolete item at all, which means its format changed
      — the flags of *previous_items* (the committed item catalog) are
      carried over.
    - ``names_zh``: the Chinese names (:func:`names_zh`;
      ``<directory>/names_zh.json`` in that shape when reading files).
    - ``missing_images``: abilities without a picture
      (:func:`missing_ability_images`; ``<directory>/missing_images.json``,
      a list of keys, when reading files).
    - ``chat_wheel``: the chat-wheel input (:func:`_download_chat_wheel`;
      ``<directory>/chat_wheel.json`` in that shape when reading files).
      Missing from the result when there is none at all.

    When Valve cannot be asked, the last three are the ones of the *stored*
    data set (the test fixture of the previous run), else — the first two —
    what the *previous* catalogs (by catalog name) carry.
    """
    previous, stored = previous or {}, stored or {}
    if directory is not None:
        raw = {name: json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
               for name in SOURCE_FILES}
        identity = {"source": "files"}
        script = directory / "items.txt"
        items_txt = script.read_text(encoding="utf-8", errors="replace") if script.exists() else None
        chinese = _read_json(directory / "names_zh.json") or None
        try:
            missing = json.loads((directory / "missing_images.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            missing = None
        chat_wheel = _read_json(directory / "chat_wheel.json") or None
    else:
        from backend.app.ai import data_manager
        raw, identity = data_manager.download_constants_snapshot(SOURCE_FILES)
        items_txt = _download_items_txt()
        chinese = _download_names_zh()
        missing = missing_ability_images(raw["abilities"], previous.get("abilities"))
        chat_wheel = _download_chat_wheel(stored.get("chat_wheel"))
    flagged = obsolete_items(items_txt) if items_txt is not None else []
    if not flagged:
        print(f"warning: {'items.txt names no obsolete item' if items_txt is not None else 'no items.txt'}; "
              "keeping the obsolete flags of the previous catalog")
        flagged = sorted(
            key for key, entry in ((previous_items or {}).get("by_key") or {}).items()
            if isinstance(entry, dict) and entry.get("obsolete"))
    raw["obsolete_items"] = flagged
    if chinese is None:
        print("warning: no Chinese names; keeping the ones of the previous catalogs")
        chinese = stored.get("names_zh") if isinstance(stored.get("names_zh"), dict) else _previous_names_zh(previous)
    raw["names_zh"] = {name: dict(chinese.get(name) or {})
                       for name in [catalog for catalog, _table in CATALOGS] + ["ability_templates"]}
    if not isinstance(missing, list):
        missing = stored.get("missing_images")
    if not isinstance(missing, list):
        missing = [key for key, entry in ((previous.get("abilities") or {}).get("by_key") or {}).items()
                   if isinstance(entry, dict) and entry.get("img") is None
                   and isinstance(raw["abilities"].get(key), dict) and raw["abilities"][key].get("img")]
    raw["missing_images"] = sorted(str(key) for key in missing)
    if chat_wheel is None and isinstance(stored.get("chat_wheel"), dict):
        print("warning: no chat wheel files; keeping the phrases of the previous run")
        chat_wheel = stored["chat_wheel"]
    if isinstance(chat_wheel, dict) and isinstance(chat_wheel.get("messages"), dict):
        raw["chat_wheel"] = chat_wheel
    return raw, identity


# A number in running text: "400", "2.5", ".5" — not the digits of "dagon_5",
# "1v1" or a longer number.
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])(?:\d+(?:\.\d+)?|\.\d+)(?![0-9]|v[0-9])")
# Where a dotaconstants item / ability entry states numbers (besides the
# values of its ``attrib`` rows and the texts of an item's ``abilities``).
_NUMBER_FIELDS = ("cost", "cd", "mc", "hc", "dmg", "charges", "desc", "notes", "hint")


def text_numbers(value: Any) -> set[str]:
    """The numbers *value* states (a text, a number or a list of them), each
    in one spelling: "2.50", "2.5" and 2.5 are all "2.5"; signs are dropped."""
    if isinstance(value, bool) or value is None:
        return set()
    if isinstance(value, (list, tuple)):
        return set().union(*(text_numbers(v) for v in value)) if value else set()
    return {f"{float(match):f}".rstrip("0").rstrip(".") for match in _NUMBER.findall(str(value))}


def entry_numbers(entry: Any) -> list[str]:
    """Every number a raw item / ability entry states about itself, sorted:
    what a hand-written text about it may quote (see ``trim``)."""
    if not isinstance(entry, dict):
        return []
    found = text_numbers([entry.get(field) for field in _NUMBER_FIELDS])
    found |= text_numbers([row.get("value") for row in entry.get("attrib") or [] if isinstance(row, dict)])
    found |= text_numbers([ability.get("description") for ability in entry.get("abilities") or []
                           if isinstance(ability, dict)])
    return sorted(found, key=float)


def trim(raw: dict[str, Any]) -> dict[str, Any]:
    """*raw* reduced to the test fixture: what :func:`build` reads, plus the
    numbers every item and hero ability states (``numbers``), which the
    hand-written editorial texts are checked against.

    ``build(trim(raw)) == build(raw)`` — :func:`main` checks it before it
    writes anything.
    """
    from backend.app.ai import data_manager

    def pick(entry: Any, fields: tuple[str, ...]) -> dict:
        return {f: entry[f] for f in fields if isinstance(entry, dict) and entry.get(f) not in (None, "")}

    abilities: dict[str, dict] = {}
    for key, ability in raw["abilities"].items():
        if not isinstance(ability, dict) or not ability.get("dname"):
            continue
        entry = pick(ability, ("dname", "img"))
        if "{s:" in str(ability["dname"]):  # the values that may fill the template
            entry["attrib"] = [pick(row, ("key", "value")) for row in ability.get("attrib") or []
                               if isinstance(row, dict)]
        abilities[key] = entry
    chinese = raw.get("names_zh") or {}
    kept = {"heroes": {str(hero["name"]).replace("npc_dota_hero_", "") for hero in raw["heroes"].values()},
            "items": set(raw["items"]), "abilities": set(abilities), "ability_templates": set(abilities)}
    for key in set(abilities) & set(chinese.get("abilities") or {}):
        if "{s:" in chinese["abilities"][key] and "attrib" not in abilities[key]:
            # the values that may fill the Chinese template
            abilities[key]["attrib"] = [pick(row, ("key", "value")) for row in raw["abilities"][key].get("attrib") or []
                                        if isinstance(row, dict)]
    hero_keys = _hero_ability_keys(raw)
    numbers = {
        "items": {key: entry_numbers(item) for key, item in raw["items"].items()
                  if isinstance(item, dict) and item.get("dname") and not key.startswith("recipe_")},
        "abilities": {key: entry_numbers(raw["abilities"][key]) for key in abilities if key in hero_keys},
    }
    return {
        "heroes": {hid: pick(hero, ("id", "name", "localized_name", "primary_attr", "attack_type",
                                    "roles", "day_vision", "night_vision"))
                   for hid, hero in raw["heroes"].items()},
        "items": {key: pick(item, ("id", "dname", "cost", "img", "tier", "qual"))
                  for key, item in raw["items"].items() if isinstance(item, dict)},
        "abilities": abilities,
        "hero_abilities": {hero: {"abilities": info.get("abilities", [])}
                           for hero, info in raw["hero_abilities"].items() if isinstance(info, dict)},
        "ability_ids": {aid: key for aid, key in raw["ability_ids"].items()
                        if key in abilities or key in data_manager.BUILTIN_ABILITY_LABELS},
        "obsolete_items": [key for key in raw.get("obsolete_items") or [] if key in raw["items"]],
        "names_zh": {catalog: {key: name for key, name in (chinese.get(catalog) or {}).items() if key in keys}
                     for catalog, keys in kept.items()},
        "missing_images": [key for key in raw.get("missing_images") or [] if key in abilities],
        "numbers": numbers,
        # (already reduced to what build_chat_wheel reads, see chat_wheel_input)
        **({"chat_wheel": raw["chat_wheel"]} if raw.get("chat_wheel") else {}),
    }


# ---------------------------------------------------------------------------
# Catalogs
# ---------------------------------------------------------------------------

# Heroes that dotaconstants names differently from the game: slug -> the name
# in Valve's hero list (https://www.dota2.com/datafeed/herolist?language=english,
# read 2026-10-11; the other 125 names agree). The source's spelling stays in
# ``name_to_id``, so matches stored under it still find their hero.
HERO_NAMES = {"obsidian_destroyer": "Outworld Destroyer", "ringmaster": "Ringmaster"}


def _zh(raw: dict[str, Any], catalog: str, key: str) -> dict[str, str]:
    """``{"name_zh": ...}`` for an entry Valve names in Chinese, else ``{}``."""
    name = ((raw.get("names_zh") or {}).get(catalog) or {}).get(key)
    return {"name_zh": name} if name else {}


def _heroes(raw: dict[str, Any]) -> dict:
    by_id: dict[str, dict] = {}
    name_to_id: dict[str, int] = {}
    for _hid, hero in sorted(raw["heroes"].items(), key=lambda kv: int(kv[0])):
        slug = str(hero["name"]).replace("npc_dota_hero_", "")
        name = HERO_NAMES.get(slug, hero["localized_name"])
        entry = {
            "id": hero["id"], "name": name, **_zh(raw, "heroes", slug), "slug": slug,
            "img": f"{CDN}{_IMAGES}/heroes/{slug}.png",
            "icon": f"{CDN}{_IMAGES}/heroes/icons/{slug}.png",
            "attr": hero.get("primary_attr"), "roles": hero.get("roles", []),
            "attack_type": hero.get("attack_type"),
        }
        for field in ("day_vision", "night_vision"):
            if isinstance(hero.get(field), (int, float)):
                entry[field] = hero[field]
        by_id[str(hero["id"])] = entry
        name_to_id[name] = hero["id"]
        name_to_id.setdefault(hero["localized_name"], hero["id"])
    return {"by_id": by_id, "name_to_id": name_to_id, "cdn_base": CDN}


def _items(raw: dict[str, Any]) -> dict:
    from backend.app.ai.data_manager import clean_item_name

    items = raw["items"]
    obsolete = set(raw.get("obsolete_items") or [])
    entries: dict[str, dict] = {}
    for key, item in items.items():
        recipe = key.startswith("recipe_")
        name = clean_item_name(item.get("dname"))
        if not name and recipe:
            base = clean_item_name((items.get(key[len("recipe_"):]) or {}).get("dname"))
            name = f"{base} Recipe" if base else ""
        entry = {"id": item.get("id"), "key": key, "name": name, **(_zh(raw, "items", key) if name else {}),
                 "img": f"{CDN}{_IMAGES}/items/recipe.png" if recipe else _image(item.get("img")),
                 "cost": item.get("cost")}
        for field in ("tier", "qual"):  # neutral tier, shop quality
            if item.get(field):
                entry[field] = item[field]
        if key in obsolete:
            entry["obsolete"] = True
        entries[key] = entry

    # A display name shared by several keys (Dagon and its upgrades) belongs
    # to the base item: the key that spells the name, else the cheapest —
    # the rule of data_manager.normalise_constants — and to a live item
    # before a removed one.
    groups: dict[str, list[str]] = {}
    for key, entry in entries.items():
        if entry["name"] and not key.startswith("recipe_"):
            groups.setdefault(entry["name"], []).append(key)

    def rank(key: str, name: str) -> tuple:
        cost = entries[key]["cost"]
        cost = cost if isinstance(cost, (int, float)) and not isinstance(cost, bool) else float("inf")
        return (key in obsolete, _squash(key) != _squash(name), cost, key)

    name_to_key = {name: min(keys, key=lambda k: rank(k, name)) for name, keys in groups.items()}
    # Readers that index names in table order (parser._item_name_index) must
    # meet the base item first: it takes the first slot of its group.
    order = list(entries)
    for name, keys in groups.items():
        base = name_to_key[name]
        if len(keys) > 1 and keys[0] != base:
            slots = sorted(order.index(k) for k in keys)
            for slot, key in zip(slots, [base] + [k for k in keys if k != base]):
                order[slot] = key
    return {"by_key": {key: entries[key] for key in order}, "name_to_key": name_to_key, "cdn_base": CDN}


def _hero_ability_keys(raw: dict[str, Any]) -> set[str]:
    """Every ability key a hero lists (sub-abilities are nested lists)."""
    keys: set[str] = set()
    for info in raw["hero_abilities"].values():
        for ability in info.get("abilities", []) if isinstance(info, dict) else []:
            keys.update(ability if isinstance(ability, list) else [ability])
    return keys


def number_abilities(by_key: dict[str, dict], raw: dict[str, Any]) -> None:
    """Set ``id`` on the entries the source's ``ability_ids`` knows, in place.

    An ability with several ids (dotaconstants joins them with a comma:
    "3060,1617") gets the first as ``id`` and all of them as ``ids``.
    """
    ids: dict[str, list[int]] = {}
    for joined, key in raw["ability_ids"].items():
        ids.setdefault(key, []).extend(int(aid) for aid in str(joined).split(",") if aid.strip().isdigit())
    for key, entry in by_key.items():
        if ids.get(key):
            entry["id"] = ids[key][0]
            if len(ids[key]) > 1:
                entry["ids"] = ids[key]


def _abilities(raw: dict[str, Any]) -> dict:
    from backend.app.ai import data_manager

    hero_keys = _hero_ability_keys(raw)
    chinese = (raw.get("names_zh") or {}).get("abilities") or {}
    templates = (raw.get("names_zh") or {}).get("ability_templates") or {}
    no_picture = set(raw.get("missing_images") or [])
    by_key: dict[str, dict] = {}
    for key, ability in raw["abilities"].items():
        name = data_manager.raw_ability_name(ability)
        if name:
            # A Chinese template is only used with every number filled in:
            # the ones the English name states, else the entry's own values.
            stated = data_manager.template_values_from_name(templates.get(key), ability.get("dname"))
            zh = data_manager.filled_ability_name(chinese.get(key), ability.get("attrib"), stated)
            by_key[key] = {"key": key, "name": name, **({"name_zh": zh} if zh else {}),
                           "img": None if key in no_picture else _image(ability.get("img"))}
    # Abilities no source names (capturing an outpost, the attribute bonus).
    for key, (zh, en) in data_manager.BUILTIN_ABILITY_LABELS.items():
        by_key[key] = {"key": key, "name": en, **({"name_zh": zh} if zh != en else {}),
                       "img": by_key.get(key, {}).get("img")}
    number_abilities(by_key, raw)

    name_to_key: dict[str, str] = {}
    for key, entry in by_key.items():  # a hero's ability wins a shared name
        current = name_to_key.get(entry["name"])
        if current is None or (key in hero_keys and current not in hero_keys):
            name_to_key[entry["name"]] = key
    return {"by_key": by_key, "name_to_key": name_to_key, "cdn_base": CDN}


def build(raw: dict[str, Any]) -> dict[str, dict]:
    """The three catalogs for the raw (or trimmed) constants, without
    ``meta`` and without the legacy keys of a previous catalog."""
    return {"heroes": _heroes(raw), "items": _items(raw), "abilities": _abilities(raw)}


def merge_legacy(catalog: dict, previous: dict, table: str) -> int:
    """Carry over the keys of the *previous* catalog that the source no
    longer has, flagged ``legacy``. Returns how many were kept."""
    from backend.app.ai import data_manager

    names = "name_to_id" if table == "by_id" else "name_to_key"
    kept = 0
    for key, entry in (previous.get(table) or {}).items():
        if key in catalog[table] or not isinstance(entry, dict):
            continue
        entry = {**entry, "legacy": True}
        if table == "by_key" and "{s:" in str(entry.get("name")):
            entry["name"] = data_manager.clean_ability_name(entry["name"])
        catalog[table][key] = entry
        if entry.get("name"):
            catalog[names].setdefault(entry["name"], entry.get("id") if table == "by_id" else key)
        kept += 1
    return kept


def current_part(catalog: dict, table: str) -> dict:
    """*catalog* without ``meta`` and legacy keys: what :func:`build` returns."""
    names = "name_to_id" if table == "by_id" else "name_to_key"
    entries = {key: entry for key, entry in catalog[table].items() if not entry.get("legacy")}
    legacy = {key for key, entry in catalog[table].items() if entry.get("legacy")}
    return {
        table: entries,
        names: {name: key for name, key in catalog[names].items() if str(key) not in legacy},
        "cdn_base": catalog.get("cdn_base"),
    }


def dump_catalog(catalog: dict) -> str:
    """JSON text with one table entry per line, so a patch shows up in a
    diff as the lines of the items that changed."""
    def compact(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))

    parts = []
    for section, value in catalog.items():
        if isinstance(value, dict) and section != "meta":
            rows = ",\n".join(f"{compact(key)}: {compact(entry)}" for key, entry in value.items())
            parts.append(f"{compact(section)}: {{\n{rows}\n}}")
        else:
            parts.append(f"{compact(section)}: {compact(value)}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


# ---------------------------------------------------------------------------
# Chat wheel
# ---------------------------------------------------------------------------

def keyvalues(text: str):
    """Every ``"key" "value"`` pair of a KeyValues text, as ``(path, key,
    value)``: ``path`` is the tuple of the names of the blocks around it."""
    path: list[str | None] = []
    pending: str | None = None  # a key waiting for its value or block
    for match in _KV_TOKEN.finditer(text):
        string, brace = match.group(1), match.group(2)
        if brace == "{":
            path.append(pending)
            pending = None
        elif brace == "}":
            if path:
                path.pop()
            pending = None
        elif string is not None:
            if pending is None:
                pending = string
            else:
                yield tuple(path), pending, string
                pending = None


def chat_wheel_messages(script: str) -> dict[str, dict]:
    """The messages of one of Valve's chat-wheel scripts by id: ``{"<id>":
    {label, message, translation, sound, image}}`` with the fields the script
    gives. The texts are as written there, a literal or a ``#token`` of the
    localisation files; ``sound`` / ``image`` are 1 when the message has one.
    """
    blocks: dict[tuple, dict[str, str]] = {}
    for path, key, value in keyvalues(script):
        if key == "message_id" or key in CHAT_WHEEL_TEXT_FIELDS or key in ("sound", "image"):
            blocks.setdefault(path, {})[key] = value
    found: dict[str, dict] = {}
    for block in blocks.values():
        message_id = block.get("message_id", "")
        if not (message_id.isascii() and message_id.isdigit()):
            continue
        entry: dict[str, Any] = {field: block[field] for field in CHAT_WHEEL_TEXT_FIELDS if block.get(field)}
        entry.update({flag: 1 for flag in ("sound", "image") if block.get(flag)})
        found.setdefault(str(int(message_id)), entry)
    return found


# "dota_chatwheel_message_MissingHero:f": the suffix says how the game formats the text.
_TOKEN_SUFFIX = re.compile(r":[a-z]+$")
# A voice line of a team's fan content: the message id is part of the token.
_FANDOM_LINE = re.compile(r"dota_chatwheel_teamfandom_\d+_\d+_voiceline_(\d+)_message")


def localisation_tokens(text: str) -> dict[str, str]:
    """``{token: text}`` of one of Valve's localisation files, tokens in
    lower case and without their format suffix."""
    tokens: dict[str, str] = {}
    for _path, key, value in keyvalues(text):
        tokens.setdefault(_TOKEN_SUFFIX.sub("", key.lower()), value)
    return tokens


def _token(value: Any) -> str | None:
    """The localisation token a script value refers to; None for a literal."""
    value = str(value or "")
    return value[1:].lower() if value.startswith("#") else None


def chat_wheel_input(scripts: list[str], tokens: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Valve's files reduced to what :func:`build_chat_wheel` reads.

    *scripts* are chat-wheel scripts (an earlier one wins for an id),
    *tokens* the localisation tokens by Valve's language name. The result is
    ``{"messages": {"<id>": {...}}, "tokens": {language: {token: text}}}``:
    of a message only the fields up to the first one that has a text in
    every language, of the tokens only those the messages name. The voice
    lines of the teams' fan content are added from the tokens themselves.
    """
    messages: dict[str, dict] = {}
    for script in scripts:
        for message_id, entry in chat_wheel_messages(script).items():
            messages.setdefault(message_id, entry)
    for token in tokens.get(CHAT_WHEEL_LANGUAGES["en"]) or {}:
        line = _FANDOM_LINE.fullmatch(token)
        if line:
            messages.setdefault(str(int(line.group(1))), {"message": f"#{token}", "sound": 1})

    tables = [tokens.get(language) or {} for language in CHAT_WHEEL_LANGUAGES.values()]
    reduced: dict[str, dict] = {}
    for message_id, entry in sorted(messages.items(), key=lambda item: int(item[0])):
        if entry.get("image"):
            reduced[message_id] = {"image": 1}    # a spray: a picture, no line
            continue
        kept: dict[str, Any] = {}
        for field in CHAT_WHEEL_TEXT_FIELDS:
            if not entry.get(field):
                continue
            kept[field] = entry[field]
            token = _token(entry[field])
            # (Valve leaves the translation of a line blank in the language it is spoken in)
            if token is None or all(chat_wheel_phrase(table.get(token)) for table in tables):
                break
        if entry.get("sound"):
            kept["sound"] = 1
        reduced[message_id] = kept
    named = {token for entry in reduced.values() for field in CHAT_WHEEL_TEXT_FIELDS
             if (token := _token(entry.get(field))) is not None}
    return {
        "messages": reduced,
        "tokens": {language: {token: table[token] for token in sorted(named) if token in table}
                   for language, table in ((language, tokens.get(language) or {})
                                           for language in CHAT_WHEEL_LANGUAGES.values())},
    }


# "{hero_name, gender, other{is missing!}}": the wording by the hero's gender;
# English and Chinese only have the one for "other".
_GENDER_SELECT = re.compile(r"\{\s*hero_name\s*,\s*gender\s*,(?:\s*\w+\s*\{[^{}]*\})*?\s*other\s*\{([^{}]*)\}\s*\}")


def chat_wheel_phrase(text: Any) -> str:
    """The line a localised chat-wheel text shows; "" when it is none.

    Two of Valve's texts are templates that name a hero
    ("{hero_name} {hero_name, gender, other{is missing!}}"): the hero stays
    in the phrase as ``{hero}``. A text with any other template, or a token
    that was not resolved, is no phrase.
    """
    text = _GENDER_SELECT.sub(lambda match: match.group(1), str(text or ""))
    text = text.replace("{hero_name}", "{hero}").replace("\\n", " ").replace('\\"', '"')
    text = " ".join(text.split())
    return "" if text.startswith("#") or "{" in text.replace("{hero}", "") else text


def build_chat_wheel(source: dict[str, Any]) -> dict:
    """The chat-wheel catalog (without ``meta``) for the input of
    :func:`chat_wheel_input`: by message id, a spray as ``{"spray": true}``,
    a line as ``{"en", "zh"[, "sound"]}``. A message without an English text
    in the game's files (retired fan content) is left out; one without a
    Chinese text shows the English one.
    """
    tokens = source.get("tokens") or {}
    by_id: dict[str, dict] = {}
    for message_id, message in (source.get("messages") or {}).items():
        if not isinstance(message, dict):
            continue
        if message.get("image"):
            by_id[message_id] = {"spray": True}
            continue
        texts: dict[str, str] = {}
        for lang, language in CHAT_WHEEL_LANGUAGES.items():
            table = tokens.get(language) or {}
            for field in CHAT_WHEEL_TEXT_FIELDS:
                token = _token(message.get(field))
                phrase = chat_wheel_phrase(message.get(field) if token is None else table.get(token))
                if phrase:
                    texts[lang] = phrase
                    break
        if not texts.get("en"):
            continue
        entry: dict[str, Any] = {"en": texts["en"], "zh": texts.get("zh") or texts["en"]}
        if message.get("sound"):
            entry["sound"] = True
        by_id[message_id] = entry
    return {"by_id": by_id}


def _download_text(url: str, timeout: int = 60) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "DotaAnalyzer/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8-sig", errors="replace")


def _download_chat_wheel(stored: Any = None) -> dict[str, Any] | None:
    """The chat-wheel input from Valve's files (:func:`chat_wheel_input`
    plus ``source``: where it was read), or None when they cannot be read.

    The mirror gets a commit with every client build while the chat wheel
    changes a few times a year: when the phrases are those of *stored* (the
    input of the previous run), *stored* is returned, so that ``source``
    keeps naming the revision the content was first read from.
    """
    try:
        head = json.loads(_download_text(CHAT_WHEEL_COMMIT_URL, timeout=30))
        commit = str(head["sha"])
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError("unexpected answer of the revision check")
        date = str(((head.get("commit") or {}).get("committer") or {}).get("date") or "")
        listing = json.loads(_download_text(CHAT_WHEEL_LISTING_URL.format(path=CHAT_WHEEL_DIR, commit=commit), timeout=30))
        seasons = sorted(str(entry.get("path")) for entry in listing if isinstance(entry, dict)
                         and str(entry.get("name") or "").startswith(CHAT_WHEEL_DIR_FILES))
        if not seasons:
            raise ValueError(f"no voice-line scripts in {CHAT_WHEEL_DIR}")
        scripts = [_download_text(CHAT_WHEEL_FILE_URL.format(commit=commit, path=path))
                   for path in (*CHAT_WHEEL_SCRIPTS, *seasons)]
        tokens: dict[str, dict[str, str]] = {}
        for language in CHAT_WHEEL_LANGUAGES.values():
            table: dict[str, str] = {}
            for name in CHAT_WHEEL_TOKEN_FILES:
                path = CHAT_WHEEL_TOKENS.format(name=name, language=language)
                for token, text in localisation_tokens(_download_text(CHAT_WHEEL_FILE_URL.format(commit=commit, path=path))).items():
                    table.setdefault(token, text)
            tokens[language] = table
    except Exception as exc:
        print(f"warning: the chat wheel files of {CHAT_WHEEL_REPO} are unreachable ({exc})")
        return None
    current = chat_wheel_input(scripts, tokens)
    if isinstance(stored, dict) and all(stored.get(part) == current[part] for part in ("messages", "tokens")):
        return stored
    return {"source": {"source": CHAT_WHEEL_REPO, "source_revision": commit[:12], "source_date": date}, **current}


# ---------------------------------------------------------------------------
# Numeric fallback of the AI coach
# ---------------------------------------------------------------------------

# What the data layer derives for an ability / an item (normalise_constants).
# Anything else in an entry was typed by hand and cannot be kept current, so
# a refreshed entry holds exactly these, plus who it belongs to.
_ABILITY_FIELDS = ("type", "damage_type", "pierces_bkb", "cooldown", "mana_cost", "damage",
                   "damage_per_second", "stun_duration", "duration", "is_ultimate")
_ITEM_FIELDS = ("cost", "is_active", "cooldown", "description")
_ITEM_KEPT_FIELDS = ("category",)  # editorial: nothing upstream says it


def build_fallback(raw: dict[str, Any], abilities: dict, items: dict) -> tuple[dict, dict, list[str]]:
    """Refreshed ``abilities`` / ``items`` tables of the fallback files for
    the raw constants, and the keys that were dropped because the game no
    longer has them."""
    from backend.app.ai import data_manager

    tables = data_manager.normalise_constants(raw)
    obsolete = set(raw.get("obsolete_items") or [])
    ability_by_key: dict[str, tuple[str, str, dict]] = {}
    for hero, hero_abilities in tables["abilities"].items():
        hero = HERO_NAMES.get((tables["heroes"].get(hero) or {}).get("internal_name"), hero)   # as the catalog names it
        for name, data in hero_abilities.items():
            for key in data_manager.ability_keys(data):
                ability_by_key.setdefault(key, (hero, name, data))
    item_by_key = {str(data["key"]): (name, data) for name, data in tables["items"].items()}

    dropped: list[str] = []
    new_abilities: dict[str, dict] = {}
    for key in abilities:
        if key not in ability_by_key:
            dropped.append(key)
            continue
        hero, name, data = ability_by_key[key]
        entry: dict[str, Any] = {"hero": hero, "ability_name": name}
        entry.update({field: data[field] for field in _ABILITY_FIELDS if field in data})
        new_abilities[key] = entry
    new_items: dict[str, dict] = {}
    for key, old in items.items():
        if key not in item_by_key or key in obsolete:
            dropped.append(key)
            continue
        name, data = item_by_key[key]
        entry = {"item_name": name}
        entry.update({field: data[field] for field in _ITEM_FIELDS if field in data})
        entry.setdefault("is_active", False)
        entry.update({field: old[field] for field in _ITEM_KEPT_FIELDS if field in old})
        new_items[key] = entry
    return new_abilities, new_items, dropped


def dump_fallback(payload: dict) -> str:
    """Indented JSON with the level lists on one line (the files' style)."""
    def render(value: Any, depth: int) -> str:
        if isinstance(value, dict) and value:
            pad = "  " * (depth + 1)
            rows = ",\n".join(f"{pad}{json.dumps(k, ensure_ascii=False)}: {render(v, depth + 1)}"
                              for k, v in value.items())
            return "{\n" + rows + "\n" + "  " * depth + "}"
        return json.dumps(value, ensure_ascii=False)
    return render(payload, 0) + "\n"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_text(path: Path, text: str | bytes) -> None:
    """Replace *path* atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8") if isinstance(text, str) else text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    """The trimmed constants of the test fixture."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _fixture_text(trimmed: dict[str, Any]) -> str:
    # Key order is the source's order and matters (which key gets a shared name).
    return json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))


def _stored_fixture_text(path: Path) -> str | None:
    """The JSON text inside the fixture (None: missing or unreadable).

    The comparison is made on this, not on the file: the same text
    compresses to different bytes under another zlib build or Python version.
    """
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read()
    except (OSError, EOFError, ValueError, zlib.error):
        return None


def _stamp(identity: dict[str, str], patch: str) -> dict[str, str]:
    """``meta`` of a catalog: when and from what it was generated.

    ``patch`` is ``--patch`` when given, else the newest game version the
    downloaded data can reflect ("" when that is unknown, as for files on
    disk), see ``data_manager.download_constants_snapshot``.
    """
    stamp = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "patch": patch or identity.get("data_label", ""), "source": identity["source"]}
    if identity["source"] != "files":
        stamp.update(source_revision=identity["source_revision"], source_date=identity["source_date"])
    return stamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=None,
                        help="directory with dotaconstants build files (offline); default: download")
    parser.add_argument("--out", type=Path, default=CATALOG_DIR, help="catalog directory")
    parser.add_argument("--fallback", type=Path, default=FALLBACK_DIR, help="fallback directory")
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH, help="test fixture (.json.gz)")
    parser.add_argument("--patch", default="",
                        help="version label to stamp, e.g. 7.41f (default: derived from the source revision)")
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 when a committed file differs from the source")
    args = parser.parse_args(argv)

    committed = {name: _read_json(args.out / f"{name}.json") for name, _table in CATALOGS}
    try:
        stored = json.loads(_stored_fixture_text(args.fixture) or "{}")
    except ValueError:
        stored = {}
    raw, identity = load_source(args.source, committed["items"], committed, stored)
    trimmed = trim(raw)
    catalogs = build(raw)
    if build(trimmed) != catalogs:
        print("error: the trimmed fixture does not rebuild the catalogs; fix trim()")
        return 2
    stamp = _stamp(identity, args.patch)
    patch = stamp["patch"]

    # The three catalogs and the two fallback files carry one stamp. When the
    # content of one of them changes, or the version to stamp is another one,
    # all five are written again: a letter patch that only moves an item cost
    # must not leave the hero catalog stamped with the version before it.
    outputs: list[tuple[Path, str, bool, bool, str]] = []   # path, new text, changed, other stamp, summary
    for name, table in CATALOGS:
        path = args.out / f"{name}.json"
        previous = _read_json(path)
        kept = merge_legacy(catalogs[name], previous, table)
        if name == "abilities":
            number_abilities(catalogs[name][table], raw)        # the carried-over keys, too
            for key in raw["missing_images"]:
                if catalogs[name][table].get(key, {}).get("legacy"):
                    catalogs[name][table][key]["img"] = None
        # (compared as text: the order of the entries is part of the content)
        changed = dump_catalog({k: v for k, v in previous.items() if k != "meta"}) != dump_catalog(catalogs[name])
        restamp = bool(patch) and (previous.get("meta") or {}).get("patch") != patch
        outputs.append((path, dump_catalog({**catalogs[name], "meta": stamp}), changed, restamp,
                        f"{len(catalogs[name][table])} entries ({kept} legacy)"))

    comment = (f"Fallback generated from dotaconstants by backend/scripts/gen_static_catalogs.py"
               f"{f' (game version {patch})' if patch else ''}; used only for what "
               "backend/data/gamedata/ has not fetched yet. {extra}Coaching text lives in "
               "backend/data/editorial/.")
    ability_file, item_file = args.fallback / "ability_stats.json", args.fallback / "item_stats.json"
    old_abilities, old_items = _read_json(ability_file), _read_json(item_file)
    abilities, items, dropped = build_fallback(
        raw, old_abilities.get("abilities") or {}, old_items.get("items") or {})
    for path, old, table, entries, extra in (
            (ability_file, old_abilities, "abilities", abilities, ""),
            (item_file, old_items, "items", items,
             "'category' is editorial and also merged into fetched data. ")):
        note = comment.format(extra=extra)
        outputs.append((path, dump_fallback({"_comment": note, table: entries}), old.get(table) != entries,
                        bool(patch) and old.get("_comment") != note, f"{len(entries)} entries"))

    # The chat wheel: its own source, the same stamp and the same rules.
    if raw.get("chat_wheel"):
        path = args.out / "chat_wheel.json"
        previous = _read_json(path)
        wheel = build_chat_wheel(raw["chat_wheel"])
        origin = raw["chat_wheel"].get("source")
        meta = {"generated_at": stamp["generated_at"], "patch": patch,
                **(origin if isinstance(origin, dict) else {"source": "files"})}
        outputs.append((path, dump_catalog({**wheel, "meta": meta}),
                        dump_catalog({k: v for k, v in previous.items() if k != "meta"}) != dump_catalog(wheel),
                        bool(patch) and (previous.get("meta") or {}).get("patch") != patch,
                        f"{len(wheel['by_id'])} entries"))

    stale = any(changed or restamp for _path, _text, changed, restamp, _summary in outputs)
    for path, text, changed, restamp, summary in outputs:
        state = "CHANGED" if changed else f"stamp is not {patch}" if restamp else "unchanged"
        print(f"{path.name}: {summary}, {state}")
        if stale and not args.check:
            _write_text(path, text)
    if dropped:
        print("no longer in the game, dropped from the fallback: " + ", ".join(sorted(dropped)))

    text = _fixture_text(trimmed)
    changed = _stored_fixture_text(args.fixture) != text
    stale |= changed
    # mtime 0: no timestamp in the file. The bytes still depend on the zlib build.
    packed = gzip.compress(text.encode("utf-8"), compresslevel=9, mtime=0)
    print(f"{args.fixture.name}: {len(packed) // 1024} KiB, {'CHANGED' if changed else 'unchanged'}")
    if changed and not args.check:
        _write_text(args.fixture, packed)

    if args.check and stale:
        print("The committed files lag the source: run python -m backend.scripts.gen_static_catalogs")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
