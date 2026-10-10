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

- heroes.json: ``{"by_id": {"<id>": {id, name, slug, img, icon, attr, roles,
  attack_type, day_vision, night_vision}}, "name_to_id", "cdn_base", "meta"}``
  — two heroes carry the game's name instead of the source's (``HERO_NAMES``);
  ``name_to_id`` knows both spellings.
- items.json: ``{"by_key": {"<key>": {id, key, name, img, cost[, tier][,
  qual][, obsolete][, legacy]}}, "name_to_key", "cdn_base", "meta"}`` — recipes included
  ("Abyssal Blade Recipe"); a display name shared by several keys maps to the
  base item (Dagon -> dagon), which also comes first among them in ``by_key``.
- abilities.json: ``{"by_key": {"<key>": {key, name, img[, id][, ids][,
  legacy]}}, "name_to_key", "cdn_base", "meta"}`` — every named ability and
  talent; no name keeps a ``{s:...}`` template token; a name shared by several
  keys maps to the hero ability; ``ids`` only where the source lists several.

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
# (catalog file, its table)
CATALOGS = (("heroes", "by_id"), ("items", "by_key"), ("abilities", "by_key"))

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


def load_source(directory: Path | None, previous_items: dict | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    """The raw input and what it is: ``source`` (an id of
    ``data_manager.CONSTANTS_SOURCES``, or ``files`` for *directory*) and,
    for a download, the revision, its date and the game version it can
    reflect (``data_manager.download_constants_snapshot``).

    The constants files by name plus ``obsolete_items``: the keys Valve's
    items.txt flags (``<directory>/items.txt`` when reading files). Without
    that file — or when it names no obsolete item at all, which means its
    format changed — the flags of *previous_items* (the committed item
    catalog) are carried over.
    """
    if directory is not None:
        raw = {name: json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
               for name in SOURCE_FILES}
        identity = {"source": "files"}
        script = directory / "items.txt"
        items_txt = script.read_text(encoding="utf-8", errors="replace") if script.exists() else None
    else:
        from backend.app.ai import data_manager
        raw, identity = data_manager.download_constants_snapshot(SOURCE_FILES)
        items_txt = _download_items_txt()
    flagged = obsolete_items(items_txt) if items_txt is not None else []
    if not flagged:
        print(f"warning: {'items.txt names no obsolete item' if items_txt is not None else 'no items.txt'}; "
              "keeping the obsolete flags of the previous catalog")
        flagged = sorted(
            key for key, entry in ((previous_items or {}).get("by_key") or {}).items()
            if isinstance(entry, dict) and entry.get("obsolete"))
    raw["obsolete_items"] = flagged
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
        "numbers": numbers,
    }


# ---------------------------------------------------------------------------
# Catalogs
# ---------------------------------------------------------------------------

# Heroes that dotaconstants names differently from the game: slug -> the name
# in Valve's hero list (https://www.dota2.com/datafeed/herolist?language=english,
# read 2026-10-11; the other 125 names agree). The source's spelling stays in
# ``name_to_id``, so matches stored under it still find their hero.
HERO_NAMES = {"obsidian_destroyer": "Outworld Destroyer", "ringmaster": "Ringmaster"}


def _heroes(raw: dict[str, Any]) -> dict:
    by_id: dict[str, dict] = {}
    name_to_id: dict[str, int] = {}
    for _hid, hero in sorted(raw["heroes"].items(), key=lambda kv: int(kv[0])):
        slug = str(hero["name"]).replace("npc_dota_hero_", "")
        name = HERO_NAMES.get(slug, hero["localized_name"])
        entry = {
            "id": hero["id"], "name": name, "slug": slug,
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
        entry = {"id": item.get("id"), "key": key, "name": name,
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
    by_key: dict[str, dict] = {}
    for key, ability in raw["abilities"].items():
        name = data_manager.raw_ability_name(ability)
        if name:
            by_key[key] = {"key": key, "name": name, "img": _image(ability.get("img"))}
    # Abilities no source names (capturing an outpost, the attribute bonus).
    for key, (_zh, en) in data_manager.BUILTIN_ABILITY_LABELS.items():
        by_key[key] = {"key": key, "name": en, "img": by_key.get(key, {}).get("img")}
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

    raw, identity = load_source(args.source, _read_json(args.out / "items.json"))
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
