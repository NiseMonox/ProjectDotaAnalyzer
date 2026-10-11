"""Game data manager: layered hero / ability / item data with editorial knowledge.

Layers (merged in this order, later layers win for the fields they provide):

- Layer A (raw stats): auto-fetched dotaconstants files, written to
  ``backend/data/gamedata/*.json`` (gitignored, refreshed at runtime). They
  are taken from the first of :data:`CONSTANTS_SOURCES` that delivers a
  complete, valid set: the dotaconstants master branch on GitHub, its
  jsDelivr mirror, then OpenDota's ``/api/constants`` (the npm release, which
  lags the game by months). ``meta.json`` records which source and revision
  the files came from. A fetch never replaces the stored set by a worse
  one (a truncated build, or the API's older release over data from the
  master branch). Until a fetch succeeds, a generated fallback
  (``backend/data/fallback/*.json``: ability / item numbers and item
  descriptions) and the committed frontend catalogs
  (``frontend/static/data/*.json``, see ``backend/scripts/gen_static_catalogs``)
  stand in for it. Both go stale: once a hero / item has fetched stats the
  fallback contributes no numbers, only what the fetched files lack (an
  item's ``category``), and fetched names / costs win over the catalog. The
  same fetch stores the talent names (``talents.json``) and the patch tables
  (``patches.json``: OpenDota's major versions for :func:`patch_for_time`,
  Valve's releases with letter patches for :func:`get_patch_info`).
- Layer B (editorial): hand-written coaching knowledge in
  ``backend/data/editorial/*.json`` — the canonical source of hero, matchup,
  ability and item tips. Always merged, with or without Layer A, and never
  overwritten by auto-updates. Per-patch *facts* it also carries (attribute,
  attack type, active / innate flags) only fill gaps: Layer A wins. What an
  item does is not written by hand at all (the fetch and the fallback
  generate it from the same constants).
- Layer C (structural principles): laning / vision / death-analysis rules in
  ``knowledge_base.py``.

Hero and item identifiers are normalised here once (display names such as
"Queen of Pain", heroes.json slugs such as "queenofpain", npc names such as
"npc_dota_hero_queenofpain", item keys such as "bfury" or "item_bfury",
replay class-name leftovers such as "boots_of_travel", common abbreviations
such as "bkb"): see :func:`resolve_hero_name`, :func:`hero_slug`,
:func:`resolve_item_name` and :func:`item_key`. Code that only needs text
for a prompt or a report uses the never-raising :func:`hero_display_name`,
:func:`item_display_name`, :func:`ability_display_name`,
:func:`unit_display_name` and :func:`display_name`. They follow the report
language: ``lang="zh"`` gives the game's Simplified Chinese name from the
committed catalogs (``name_zh``, what the UI shows) and the English name
where the catalog has none. Everything else stays keyed by the English
names and slugs; the lookups also resolve a Chinese name back to them.
Text that is not written through the helpers (the English reference
knowledge of a zh prompt, an LLM answer) is handled by
:func:`name_glossary`, :func:`localize_names` and :func:`strip_names`.

Every loader returns a shared, cached object that callers must treat as
read-only. :func:`invalidate_all_caches` atomically drops all caches (this
module, ``game_data`` and the RAG stores in ``knowledge_base``); it runs
automatically after :func:`fetch_and_save_gamedata` writes new data. Data
written by another process (the ``update_gamedata`` CLI next to a running
server) is picked up too: ``meta.json`` is checked at most every
:data:`DISK_CHECK_INTERVAL_S` seconds.
"""
from __future__ import annotations

import calendar
import json
import logging
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

from .. import config, game_constants, osutil

logger = logging.getLogger(__name__)

OPENDOTA_BASE = "https://api.opendota.com/api/constants"

# Where the constants (the dotaconstants "build" files) are downloaded from:
# (id recorded in meta.json, URL template), in the order they are tried. The
# master branch follows the game; the OpenDota API serves the npm release,
# which is only cut on major patches (in October 2026 it still was the March
# release), so it is the last resort. One fetch takes every file from the
# same source — files of two revisions are never mixed.
CONSTANTS_SOURCES: tuple[tuple[str, str], ...] = (
    ("dotaconstants_github",
     "https://raw.githubusercontent.com/odota/dotaconstants/master/build/{name}.json"),
    ("dotaconstants_jsdelivr",
     "https://cdn.jsdelivr.net/gh/odota/dotaconstants@master/build/{name}.json"),
    ("opendota_api", OPENDOTA_BASE + "/{name}"),
)
OPENDOTA_SOURCE = "opendota_api"
JSDELIVR_SOURCE = "dotaconstants_jsdelivr"
# Last commit that changed the build files of the master branch: the revision
# (and date) of what the first two sources serve. jsDelivr caches a branch
# for up to twelve hours, so a fetch that knows the revision asks the mirror
# for exactly that commit (see _sources).
SOURCE_REVISION_URL = (
    "https://api.github.com/repos/odota/dotaconstants/commits?path=build&sha=master&per_page=1")
_COMMIT_RE = re.compile(r"[0-9a-f]{12,40}")
# Valve's own list of game versions with letter patches and release times.
# OpenDota's patch table only has the major versions.
VALVE_PATCH_FEED = "https://www.dota2.com/datafeed/patchnoteslist?language=english"

# Game version the builtin / editorial knowledge and the committed catalogs
# were checked against, and its release time (Valve's patch feed). Used
# whenever the live version is unknown (no successful fetch yet). This is the
# ONLY place a patch string may be hardcoded — everything else must call
# get_current_patch() / get_patch_info(). The one exception is the stamp of
# the hand-kept rule values, game_constants.RULES_PATCH, which a test ties to
# this label (same major version).
DEFAULT_PATCH_LABEL = "7.41f"
DEFAULT_PATCH_RELEASED = "2026-09-15T07:00:00Z"
_PATCH_LABEL_RE = re.compile(r"(\d+\.\d+)([a-z]*)")


def patch_major(label: Any) -> str:
    """Major version of a patch label: 7.41f -> 7.41 (other text is returned stripped)."""
    text = str(label or "").strip()
    match = _PATCH_LABEL_RE.fullmatch(text)
    return match.group(1) if match else text


# The major version: what patch-specific knowledge is compared by.
DEFAULT_PATCH = patch_major(DEFAULT_PATCH_LABEL)

# Format of the files written by fetch_and_save_gamedata, recorded in
# meta.json as "schema_version". Bump it whenever the normalisation changes:
# check_patch_version() then reports needs_update, so installs refetch
# instead of consuming files in an older shape (files without the marker
# predate it and are sanitised on load until they are refreshed).
#   2: ultimates / innates / skillable abilities flagged, attrib-derived
#      damage / damage per second / stun / duration, item name collisions
#      resolved, item components and ability texts (description) kept.
#   3: talent names (talents.json) and the patch release table
#      (patches.json) added; heroes / abilities / items are as in 2.
#   4: heroes carry day_vision / night_vision; meta.json names the source,
#      its revision and the full patch label. Files of schema 3 were fetched
#      from the OpenDota API only, i.e. from the frozen npm release.
GAMEDATA_SCHEMA_VERSION = 4
# Hero / ability / item files older than this lack the fetch-time flags and
# are sanitised on load (_build_abilities, _build_items) until refetched.
_SANITISE_BELOW_SCHEMA = 2

TALENTS_FILE = "talents.json"
PATCHES_FILE = "patches.json"

# Talents are abilities whose key starts with this; the one exception is the
# attribute bonus, which no catalog names.
TALENT_PREFIX = "special_bonus_"
# Abilities without a display name in every source: key -> (zh, en). The
# attribute bonus is shown untranslated like every ability name; the others
# are map actions every hero has (terms of Valve's Chinese 7.33 notes).
BUILTIN_ABILITY_LABELS: dict[str, tuple[str, str]] = {
    "special_bonus_attributes": ("Attribute Bonus", "Attribute Bonus"),
    "ability_capture": ("占领前哨", "Capture Outpost"),
    "ability_lamp_use": ("启动观察者", "Activate Watcher"),
    "twin_gate_portal_warp": ("双生之门", "Twin Gate"),
}
_BUILTIN_ABILITY_NAMES = {key: en for key, (_zh, en) in BUILTIN_ABILITY_LABELS.items()}

# How often (seconds) meta.json is checked for a change made by another
# process. Read at call time, so tests may monkeypatch it.
DISK_CHECK_INTERVAL_S = 30.0

# Data directories (module attributes so tests can monkeypatch them).
GAMEDATA_DIR: Path = config.GAMEDATA_DIR
EDITORIAL_DIR: Path = config.EDITORIAL_DIR
FALLBACK_DIR: Path = config.DATA_DIR / "fallback"
STATIC_DATA_DIR: Path = config.STATIC_DIR / "data"

# Squashed alias -> internal hero name, for names the squashed-name lookup
# cannot infer by itself: display-name renames and the abbreviations players
# (and LLM-written learned tips) use. Only unambiguous ones are listed.
_HERO_ALIASES: dict[str, str] = {
    "outworlddestroyer": "obsidian_destroyer",
    "outworlddevourer": "obsidian_destroyer",
    "od": "obsidian_destroyer",
    "am": "antimage",
    "sf": "nevermore",
    "wk": "skeleton_king",
    "qop": "queenofpain",
    "np": "furion",
    "ck": "chaos_knight",
    "dk": "dragon_knight",
    "pa": "phantom_assassin",
    "pl": "phantom_lancer",
    "ta": "templar_assassin",
    "cm": "crystal_maiden",
    "wr": "windrunner",
    "bb": "bristleback",
    "lc": "legion_commander",
    "ns": "night_stalker",
    "bh": "bounty_hunter",
    "tb": "terrorblade",
    "kotl": "keeper_of_the_light",
    "aa": "ancient_apparition",
    "mk": "monkey_king",
    "dp": "death_prophet",
    "ld": "lone_druid",
    "wd": "witch_doctor",
    "naix": "life_stealer",
    "clock": "rattletrap",
    "timber": "shredder",
}

# Squashed alias -> canonical item key (the OpenDota items.json key). Covers
# the class-name leftovers some replays carry instead of the item key, and
# common abbreviations. Real keys and display names always win over these.
_ITEM_ALIASES: dict[str, str] = {
    # entity class names
    "bootsoftravel": "travel_boots",
    "dustofappearance": "dust",
    "invisibilityedge": "invis_sword",
    "emptybottle": "bottle",
    # abbreviations
    "midas": "hand_of_midas",
    "bkb": "black_king_bar",
    "aghs": "ultimate_scepter",
    "aghanims": "ultimate_scepter",
    "scepter": "ultimate_scepter",
    "shard": "aghanims_shard",
    "aghsshard": "aghanims_shard",
    "mkb": "monkey_king_bar",
    "sny": "sange_and_yasha",
    "bots": "travel_boots",
    "tp": "tpscroll",
    "euls": "cyclone",
    "eul": "cyclone",
    "force": "force_staff",
    "glimmer": "glimmer_cape",
    "linkens": "sphere",
    "linken": "sphere",
    "lotus": "lotus_orb",
    "hex": "sheepstick",
    "scythe": "sheepstick",
    "shivas": "shivas_guard",
    "deso": "desolator",
    "halberd": "heavens_halberd",
    "pike": "hurricane_pike",
    "atos": "rod_of_atos",
    "wand": "magic_wand",
    "stick": "magic_stick",
    "treads": "power_treads",
    "phase": "phase_boots",
    "arcanes": "arcane_boots",
    "greaves": "guardian_greaves",
    "mek": "mekansm",
    "vlads": "vladmir",
    "vessel": "spirit_vessel",
    "urn": "urn_of_shadows",
    "diffusal": "diffusal_blade",
    "octarine": "octarine_core",
    "aeon": "aeon_disk",
    "eblade": "ethereal_blade",
    "abyssal": "abyssal_blade",
    "obs": "ward_observer",
    "observer": "ward_observer",
    "sentry": "ward_sentry",
    "smoke": "smoke_of_deceit",
    "salve": "flask",
    "mango": "enchanted_mango",
    "raindrop": "infused_raindrop",
    "raindrops": "infused_raindrop",
}

# Damage inflictor of a plain attack (parsers map their own spellings to it).
ATTACK_INFLICTOR = "attack"
_ATTACK_LABELS = {"zh": "普通攻击", "en": "Attack"}
_RECIPE_LABELS = {"zh": "图纸（{}）", "en": "{} Recipe"}

# Non-hero units by a substring of their npc name -> (zh, en); first match wins.
_UNIT_LABELS: tuple[tuple[str, tuple[str, str]], ...] = (
    ("roshan", ("肉山", "Roshan")),
    ("tower", ("防御塔", "Tower")),
    ("fort", ("遗迹", "Ancient")),
    ("rax", ("兵营", "Barracks")),
    ("fountain", ("泉水", "Fountain")),
    ("courier", ("信使", "Courier")),
    ("siege", ("攻城车", "Siege creep")),
    ("neutral", ("野怪", "Neutral creep")),
    ("creep", ("小兵", "Lane creep")),
)

# OpenDota primary_attr -> our attribute code
_ATTR_MAP = {"agi": "agi", "str": "str", "int": "int", "all": "uni", "uni": "uni"}


# ---------------------------------------------------------------------------
# Cache (atomic snapshot + generation counter)
# ---------------------------------------------------------------------------

_cache_lock = threading.RLock()
_cache: dict[str, Any] = {}
_generation = 0
_fetch_lock = threading.Lock()

# Signature of meta.json when the caches were last dropped, and the
# monotonic time of the next check (see _check_disk_change).
_disk_signature: tuple | None = None
_next_disk_check = 0.0
_build_depth = 0  # >0 while a builder runs (only touched under _cache_lock)


def _meta_signature() -> tuple:
    """Identity of the game data on disk: meta.json is written last by a fetch."""
    path = GAMEDATA_DIR / "meta.json"
    try:
        stat = path.stat()
        return (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return (str(path), None, None)


def _check_disk_change() -> None:
    """Drop the caches when another process replaced the game data.

    Throttled to one ``stat`` per :data:`DISK_CHECK_INTERVAL_S`. Only swaps
    the snapshot and bumps the generation; ``game_data`` and the RAG stores
    notice through :func:`data_generation`. It must not call into them: a
    reader may already hold one of their locks while it loads data here.
    """
    global _cache, _generation, _disk_signature, _next_disk_check
    now = time.monotonic()
    if now < _next_disk_check:
        return
    with _cache_lock:
        if now < _next_disk_check or _build_depth:
            return  # mid-build: swapping now would cache a mix of old and new data
        _next_disk_check = now + max(0.0, float(DISK_CHECK_INTERVAL_S))
        signature = _meta_signature()
        if _disk_signature is None:
            _disk_signature = signature
        elif signature != _disk_signature:
            _disk_signature = signature
            _cache = {}
            _generation += 1
            logger.info("Game data on disk changed; caches dropped (generation %d)", _generation)


def data_generation() -> int:
    """Monotonic counter bumped whenever the game-data caches are dropped
    (:func:`invalidate_all_caches`, or a change on disk made by another
    process).

    Downstream caches (game_data lookups, RAG stores) compare it to know
    when to rebuild.
    """
    _check_disk_change()
    return _generation


def _cached(name: str, builder: Callable[[], Any]) -> Any:
    """Return the cached value for *name*, building it once under the lock.

    Builders run while holding the (re-entrant) lock, so concurrent first
    calls never build twice and readers never observe half-built data.
    """
    global _build_depth
    _check_disk_change()
    value = _cache.get(name)
    if value is not None:
        return value
    with _cache_lock:
        value = _cache.get(name)
        if value is None:
            _build_depth += 1
            try:
                value = builder()
            finally:
                _build_depth -= 1
            _cache[name] = value
        return value


def invalidate_all_caches() -> None:
    """Drop every game-data cache so the next access reloads from disk.

    Thread-safe: the cache snapshot is swapped atomically. Also resets the
    dependent caches in ``game_data`` and the per-language RAG stores in
    ``knowledge_base`` (both additionally track :func:`data_generation`).
    """
    global _cache, _generation, _disk_signature, _next_disk_check
    with _cache_lock:
        _cache = {}
        _generation += 1
        _disk_signature = _meta_signature()
        _next_disk_check = time.monotonic() + max(0.0, float(DISK_CHECK_INTERVAL_S))
    try:
        from .game_data import invalidate_cache as _invalidate_game_data
        _invalidate_game_data()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to invalidate game_data cache: %s", exc)
    try:
        from .knowledge_base import reset_knowledge_stores
        reset_knowledge_stores()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to reset knowledge stores: %s", exc)
    logger.info("Game data caches invalidated (generation %d)", _generation)


def clear_caches() -> None:
    """Backward-compatible alias of :func:`invalidate_all_caches`."""
    invalidate_all_caches()


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _http_get_json(url: str, timeout: int = 30) -> Any:
    """GET *url* and return the parsed JSON body. The module's only network call."""
    logger.info("Fetching %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "DotaAnalyzer/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _api_get(endpoint: str, timeout: int = 30) -> Any:
    """GET from the OpenDota constants API (the last of :data:`CONSTANTS_SOURCES`)."""
    return _http_get_json(f"{OPENDOTA_BASE}/{endpoint}", timeout)


def _source_get(source: str, template: str, name: str, timeout: int = 30) -> Any:
    """One constants file from one source."""
    if source == OPENDOTA_SOURCE:
        return _api_get(name, timeout)
    return _http_get_json(template.format(name=name), timeout)


def _sources(revision: Any = None) -> tuple[tuple[str, str], ...]:
    """:data:`CONSTANTS_SOURCES` as one fetch uses them.

    With the *revision* of the master branch known (:func:`_source_revision`)
    the jsDelivr mirror is asked for that commit instead of the branch: the
    answer is then exactly the revision that gets recorded, never a cached
    older copy of the branch.
    """
    commit = str(revision.get("commit") or "") if isinstance(revision, dict) else ""
    if not _COMMIT_RE.fullmatch(commit):
        return CONSTANTS_SOURCES
    return tuple((source, template.replace("@master/", f"@{commit}/") if source == JSDELIVR_SOURCE else template)
                 for source, template in CONSTANTS_SOURCES)


# How long (seconds) a source that could not be reached is tried last by
# download_constants_set. Read at call time, so tests may monkeypatch it.
SOURCE_RETRY_S = 1800.0
# source id -> monotonic time until which it counts as unreachable (this
# process only; guarded by _source_lock: downloads run in several threads).
_source_down_until: dict[str, float] = {}
_source_lock = threading.Lock()


def _is_unreachable(exc: BaseException) -> bool:
    """Whether a failed request says the source cannot be reached right now
    (no connection, a timeout, a server error) rather than that it has no
    such file or sent something unusable."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500 or exc.code == 429
    return isinstance(exc, (urllib.error.URLError, TimeoutError, OSError))


def _reachable_first(sources: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    """*sources* with the ones recently found unreachable moved to the end
    (their order kept; they remain the last resort)."""
    now = time.monotonic()
    with _source_lock:
        down = {source for source, until in _source_down_until.items() if until > now}
    return tuple(s for s in sources if s[0] not in down) + tuple(s for s in sources if s[0] in down)


def _download_set(names: tuple[str, ...], validate: Callable[[dict], Any] | None = None,
                  timeout: int = 30,
                  sources: tuple[tuple[str, str], ...] | None = None) -> tuple[dict[str, Any], str, Any]:
    """Every file of *names* from the first source that delivers all of them.

    Returns ``(payloads, source id, validate(payloads))``. A source that
    fails to answer for one file, or whose payloads *validate* rejects (it
    raises), is given up as a whole and the next one is tried, so the files
    of one fetch always belong together. Raises the last source's error when
    none delivers. *sources* defaults to :data:`CONSTANTS_SOURCES`. A source
    that could not be reached is remembered for :data:`SOURCE_RETRY_S` (see
    :func:`download_constants_set`); one that delivers is cleared again.
    """
    last_error: Exception | None = None
    for source, template in CONSTANTS_SOURCES if sources is None else sources:
        try:
            payloads = {name: _source_get(source, template, name, timeout) for name in names}
        except Exception as exc:
            if _is_unreachable(exc):
                with _source_lock:
                    _source_down_until[source] = time.monotonic() + max(0.0, float(SOURCE_RETRY_S))
            logger.warning("Game-data source %s unusable for %s: %s", source, "/".join(names), exc)
            last_error = exc
            continue
        with _source_lock:
            _source_down_until.pop(source, None)
        try:
            checked = validate(payloads) if validate is not None else None
        except Exception as exc:
            logger.warning("Game-data source %s unusable for %s: %s", source, "/".join(names), exc)
            last_error = exc
            continue
        return payloads, source, checked
    raise last_error if last_error is not None else RuntimeError("No game-data source configured")


def download_constants_set(names: tuple[str, ...], timeout: int = 30) -> tuple[dict[str, Any], str]:
    """Several dotaconstants files (``items``, ``item_ids``, ``ability_ids``,
    ``patch`` ...) from ONE source: the first of :data:`CONSTANTS_SOURCES`
    that answers for all of them.

    For callers that need files the game-data fetch does not store (the
    catalog generator, the OpenDota match parser). Returns ``({name: parsed
    JSON}, source id)``; raises when no source delivers the whole set.

    A source this process could not reach within the last
    :data:`SOURCE_RETRY_S` seconds is tried after the others: on a host
    that cannot reach GitHub or jsDelivr only the first download waits for
    their timeouts, the following ones go straight to the source that
    answers. (The game-data fetch itself always walks the sources in
    order: which source may replace stored data is its decision.)
    """
    payloads, source, _checked = _download_set(
        tuple(names), timeout=timeout, sources=_reachable_first(CONSTANTS_SOURCES))
    return payloads, source


def download_constants(name: str, timeout: int = 30) -> Any:
    """One dotaconstants file by name, see :func:`download_constants_set`."""
    return download_constants_set((name,), timeout)[0][name]


def _source_revision() -> dict[str, str]:
    """``{"sha", "commit", "date"}`` of the last commit that changed the
    build files of dotaconstants master (what the first two sources serve):
    the short id that is recorded, the full one, its date. Raises on errors.
    """
    commits = _http_get_json(SOURCE_REVISION_URL, timeout=15)
    commit = commits[0] if isinstance(commits, list) and commits else None
    sha = str(commit.get("sha") or "") if isinstance(commit, dict) else ""
    if not _COMMIT_RE.fullmatch(sha):
        raise RuntimeError("Unexpected answer of the dotaconstants revision check")
    stamp = ((commit.get("commit") or {}).get("committer") or {}).get("date")
    return {"sha": sha[:12], "commit": sha, "date": str(stamp) if _parse_utc(stamp) is not None else ""}


def _data_identity(source: str, releases: Any, revision: Any) -> dict[str, str]:
    """What constants downloaded from *source* just now are (*revision* being
    the one the download was made with, see :func:`_sources`).

    ``source_revision`` / ``source_date``: the revision of the master branch
    for its two mirrors, "" for the OpenDota API (an npm release of unknown
    age) or when the revision check failed. ``data_label``: the newest game
    version released by ``source_date`` according to Valve's *releases* —
    the newest version the data can reflect; "" when either is unknown.
    """
    from_master = source != OPENDOTA_SOURCE and isinstance(revision, dict)
    source_date = str(revision.get("date") or "") if from_master else ""
    released_by = _parse_utc(source_date)
    return {
        "source": source,
        "source_url": dict(_sources(revision)).get(source, "").rsplit("/", 1)[0],
        "source_revision": str(revision.get("sha") or "") if from_master else "",
        "source_date": source_date,
        # (the list was downloaded just now: it is not too old to vouch for that date)
        "data_label": _name_at(releases, float("inf"), released_by) if releases and released_by else "",
    }


def download_constants_snapshot(names: tuple[str, ...], timeout: int = 30) -> tuple[dict[str, Any], dict[str, str]]:
    """Several dotaconstants files from one source, with what they are.

    For tools that keep a copy of the constants (the catalog generator).
    Like :func:`download_constants_set`, but the revision of the master
    branch is read first and the mirror is asked for exactly that commit, so
    the returned identity (``source``, ``source_url``, ``source_revision``,
    ``source_date``, ``data_label``, see :func:`_data_identity`) describes
    the files that were downloaded. A part the network does not tell is "".
    Raises when no source delivers the whole set.
    """
    releases = revision = None
    try:
        revision = _source_revision()
    except Exception as exc:
        logger.warning("The source revision is unavailable: %s", exc)
    payloads, source, _checked = _download_set(tuple(names), timeout=timeout, sources=_sources(revision))
    try:
        releases = _valve_releases()
    except Exception as exc:
        logger.warning("Valve's patch feed is unavailable: %s", exc)
    return payloads, _data_identity(source, releases, revision)


def _save_json(path: Path, data: Any) -> None:
    """Atomically write JSON (temp file + rename) so readers never see a partial file.

    UTF-8 with ``\\n`` line ends on every system (Windows would write
    ``\\r\\n`` in text mode: the same data, other bytes and another size).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # mkstemp creates 0600 files; game data fetched by another user
        # (a root cron job) must stay readable for the service.
        os.chmod(tmp, 0o644)
        osutil.atomic_replace(tmp, path)  # Windows: waits while a reader has the old file open
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    logger.info("Saved %s", path)


def _load_json(path: Path) -> Any | None:
    """Load JSON from *path*; None when missing or unreadable (logged)."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("Cannot read %s: %s", path, exc)
        return None


def _load_dict(path: Path, key: str | None = None) -> dict:
    """Load a JSON object (optionally a nested object under *key*); {} otherwise."""
    data = _load_json(path)
    if key is not None and isinstance(data, dict):
        data = data.get(key)
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# Name normalisation (heroes + items)
# ---------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _squash_hero(name: Any) -> str:
    """'Queen of Pain' / 'queenofpain' / 'npc_dota_hero_queenofpain' -> 'queenofpain'."""
    text = str(name or "").strip().lower()
    if text.startswith("npc_dota_hero_"):
        text = text[len("npc_dota_hero_"):]
    return _NON_ALNUM.sub("", text)


def _squash_item(name: Any) -> str:
    """'Battle Fury' / 'item_bfury' / 'battle_fury' -> 'battlefury' / 'bfury'."""
    text = str(name or "").strip().lower()
    if text.startswith("item_"):
        text = text[len("item_"):]
    return _NON_ALNUM.sub("", text)


def _zh_text(name: Any) -> str:
    """A Chinese catalog name as a lookup key: one line, no outer blanks."""
    return " ".join(str(name or "").split())


_NAME_LINE_BREAK = re.compile(r"\\n|[\r\n]")


def clean_item_name(name: Any) -> str:
    """Item display name on one line ("" when there is none).

    Valve writes a few names over two lines, the second one a qualifier
    ("Black Grimoire" / "(Warlock)"); dotaconstants passes the break on as a
    literal backslash-n. Only the first line is the name.
    """
    lines = (" ".join(line.split()) for line in _NAME_LINE_BREAK.split(str(name or "")))
    return next((line for line in lines if line), "")


def display_slug(name: Any) -> str:
    """Generic slug of a display name: lowercase, non-alphanumerics -> "_".

    'Anti-Mage' -> 'anti_mage', "Nature's Prophet" -> 'nature_s_prophet'.
    Used for identifiers that are not hero/item tags (e.g. topic slugs).
    """
    return _NON_ALNUM.sub("_", str(name or "").lower()).strip("_")


def _build_hero_index() -> dict:
    """Merge every hero catalog we have into one alias index.

    Returns ``{"by_internal": {internal: entry}, "alias": {squashed: internal},
    "alias_zh": {Chinese name: internal}}`` where entry is ``{"id", "name",
    "internal", "roles", "attack_type", "attr"}`` plus ``day_vision`` /
    ``night_vision`` when a source states them and ``name_zh`` (the game's
    Simplified Chinese name) when the committed catalog has it.
    Display names come from the committed frontend catalog first (what the
    parsers emit), then fetched game data, then editorial data. The facts of
    a hero (attribute, attack type, roles, vision) are per-patch: there the
    fetched value replaces the catalog's.
    """
    by_internal: dict[str, dict] = {}
    alias: dict[str, str] = {}
    alias_zh: dict[str, str] = {}

    def add(internal: str, name: str, replace: bool = False, **fields: Any) -> None:
        internal = str(internal or "").strip().lower()
        if internal.startswith("npc_dota_hero_"):
            internal = internal[len("npc_dota_hero_"):]
        if not internal:
            return
        entry = by_internal.get(internal)
        if entry is None:
            entry = {"id": None, "name": name or internal, "internal": internal,
                     "roles": [], "attack_type": "", "attr": ""}
            by_internal[internal] = entry
        for key, value in fields.items():
            if value and (replace or not entry.get(key)):
                entry[key] = value
        for key in (internal, name):
            squashed = _squash_hero(key)
            if squashed:
                alias.setdefault(squashed, internal)

    static = _load_dict(STATIC_DATA_DIR / "heroes.json", "by_id")
    for hid, info in static.items():
        if not isinstance(info, dict):
            continue
        try:
            hero_id = int(info.get("id") or hid)
        except (TypeError, ValueError):
            hero_id = None
        add(info.get("slug", ""), info.get("name", ""), id=hero_id,
            roles=list(info.get("roles") or []),
            attack_type=str(info.get("attack_type") or "").lower(),
            attr=_ATTR_MAP.get(str(info.get("attr") or ""), str(info.get("attr") or "")),
            day_vision=_vision_range(info.get("day_vision")),
            night_vision=_vision_range(info.get("night_vision")),
            name_zh=_zh_text(info.get("name_zh")))

    gamedata = _load_dict(GAMEDATA_DIR / "heroes.json")
    for localized, info in gamedata.items():
        if not isinstance(info, dict):
            continue
        internal = info.get("internal_name") or _squash_hero(localized)
        add(internal, localized, replace=True, id=info.get("id"),
            roles=list(info.get("roles") or []),
            attack_type=str(info.get("attack_type") or "").lower(),
            attr=_ATTR_MAP.get(str(info.get("primary_attr") or ""), ""),
            day_vision=_vision_range(info.get("day_vision")),
            night_vision=_vision_range(info.get("night_vision")))

    for squashed, internal in _HERO_ALIASES.items():
        if internal in by_internal:
            alias.setdefault(squashed, internal)

    # Editorial-only heroes (not in any catalog) still get an entry.
    editorial = _load_dict(EDITORIAL_DIR / "hero_knowledge.json")
    for name in editorial:
        squashed = _squash_hero(name)
        if squashed and squashed not in alias:
            add(squashed, name)

    for internal, entry in by_internal.items():
        if entry.get("name_zh"):
            alias_zh.setdefault(entry["name_zh"], internal)
    return {"by_internal": by_internal, "alias": alias, "alias_zh": alias_zh}


def _hero_index() -> dict:
    return _cached("hero_index", _build_hero_index)


def _hero_entry(name: Any) -> dict | None:
    squashed = _squash_hero(name)
    index = _hero_index()
    internal = index["alias"].get(squashed) if squashed else None
    if internal is None:  # the game's Chinese name ("斧王"), as zh reports and prompts write it
        internal = index.get("alias_zh", {}).get(_zh_text(name))
    return index["by_internal"].get(internal) if internal else None


def resolve_hero_name(name: Any) -> str | None:
    """Canonical (English) display name for any hero identifier (None if unknown).

    Also resolves the game's Chinese name ("斧王" -> "Axe") and the bilingual
    form prompts use ("斧王 (Axe)"), so a name copied from a zh report or
    sent back by the LLM still finds its hero.
    """
    entry = _hero_entry(name)
    return entry["name"] if entry else None


def hero_slug(name: Any) -> str:
    """Canonical hero tag slug (internal name, e.g. "queenofpain", "furion").

    Unknown names fall back to their squashed form so producers and
    consumers still agree. ``"all"`` is preserved for universal chunks.
    """
    squashed = _squash_hero(name)
    if squashed == "all":
        return "all"
    entry = _hero_entry(name)
    return entry["internal"] if entry else squashed


def is_known_hero(name: Any) -> bool:
    """Whether *name* resolves to a hero of the catalog (any identifier form)."""
    return _hero_entry(name) is not None


def get_hero_catalog() -> list[dict]:
    """Every known hero with an id: ``[{"id", "name", "slug"}]`` sorted by id.

    Works without fetched game data (the frontend catalog is committed), so
    learners never lose their hero list to a concurrent / failed fetch.
    """
    def build() -> list[dict]:
        out = [
            {"id": int(e["id"]), "name": e["name"], "slug": e["internal"]}
            for e in _hero_index()["by_internal"].values()
            if isinstance(e.get("id"), int) and e["id"] > 0
        ]
        out.sort(key=lambda e: e["id"])
        return out
    return _cached("hero_catalog", build)


def hero_vision(identifier: Any) -> tuple[int, int]:
    """``(day, night)`` vision range of a hero in world units. Never raises.

    From the fetched game data, else the committed catalog; a hero neither
    of them knows (or a value they do not state) gets
    :data:`DEFAULT_HERO_VISION`, which is right for all but a few heroes
    (Night Stalker 800 / 1800, Slark 1800 / 1800, Sniper 1800 / 1400 ...).
    These are base values: items, abilities and talents that change vision
    are not applied.
    """
    try:
        entry = _hero_entry(identifier) or {}
        return (int(entry.get("day_vision") or DEFAULT_HERO_VISION[0]),
                int(entry.get("night_vision") or DEFAULT_HERO_VISION[1]))
    except Exception:  # pragma: no cover - defensive
        return DEFAULT_HERO_VISION


def _build_item_index() -> dict:
    """Alias index for items: ``{"by_key": {key: name}, "alias": {squashed: key},
    "zh": {key: Chinese name}, "alias_zh": {Chinese name: key}}``.

    Fetched game data comes first, so its names win; the committed catalog
    then adds what the fetch does not store (recipes, items of older patches)
    and the game's Simplified Chinese names.
    """
    by_key: dict[str, str] = {}
    alias: dict[str, str] = {}
    zh: dict[str, str] = {}
    alias_zh: dict[str, str] = {}

    def add(key: str, name: str) -> None:
        key = str(key or "").strip().lower()
        if key.startswith("item_"):
            key = key[len("item_"):]
        if not key:
            return
        if key not in by_key or not by_key[key]:
            by_key[key] = name or by_key.get(key, "")
        alias.setdefault(_squash_item(key), key)

    def add_name(name: str, key: str) -> None:
        squashed = _squash_item(name)
        if squashed and not str(name).lower().startswith("recipe"):
            alias.setdefault(squashed, key)

    # (cleaned again: files fetched before names were cleaned keep the raw ones)
    fetched = [(clean_item_name(name), info) for name, info in _load_dict(GAMEDATA_DIR / "items.json").items()
               if isinstance(info, dict) and info.get("key")]
    static = _load_json(STATIC_DATA_DIR / "items.json")
    static = static if isinstance(static, dict) else {}
    by_key_static = static.get("by_key") if isinstance(static.get("by_key"), dict) else {}
    # Keys first (exact) — a display name must never shadow another item's key.
    for name, info in fetched:
        add(info["key"], name)
        for upgrade in info.get("upgrade_keys") or []:
            add(upgrade, name)
    for key, info in by_key_static.items():
        if isinstance(info, dict):
            add(key, str(info.get("name") or ""))
            name_zh = _zh_text(info.get("name_zh"))
            if name_zh:
                zh[str(key).lower()] = name_zh
    # A Chinese name shared by several keys ("远行鞋": both levels) is the first one's.
    for key, name_zh in zh.items():
        alias_zh.setdefault(name_zh, key)
    # Then names: fetched, the catalog's preferred mapping, its other names.
    for name, info in fetched:
        add_name(name, str(info["key"]).lower())
    for name, key in (static.get("name_to_key") or {}).items():
        if isinstance(key, str):
            add_name(name, key)
    for key, info in by_key_static.items():
        if isinstance(info, dict) and info.get("name"):
            add_name(info["name"], key)
    # Upgrade levels are the same item for coaching: "dagon_5" -> "dagon"
    # (only when both keys carry the same display name).
    for key, name in list(by_key.items()):
        base, _, level = key.rpartition("_")
        if base and level.isdigit() and name and by_key.get(base) == name:
            alias[_squash_item(key)] = base
    for data in (_load_dict(FALLBACK_DIR / "item_stats.json", "items"),
                 _load_dict(EDITORIAL_DIR / "coaching_tips_items.json")):
        for key, info in data.items():
            if isinstance(info, dict):
                add(key, str(info.get("item_name") or ""))
                add_name(str(info.get("item_name") or ""), key)
    # Class-name leftovers and abbreviations, after every real key / name.
    for squashed, key in _ITEM_ALIASES.items():
        if key in by_key:
            alias.setdefault(squashed, key)
    return {"by_key": by_key, "alias": alias, "zh": zh, "alias_zh": alias_zh}


def _item_index() -> dict:
    return _cached("item_index", _build_item_index)


def resolve_item_key(name: Any) -> str | None:
    """Internal item key for a display name / key / ``item_`` name (None if unknown).

    The game's Chinese name ("黑皇杖") and the bilingual form of the prompts
    ("黑皇杖 (Black King Bar)") resolve as well.
    """
    squashed = _squash_item(name)
    index = _item_index()
    key = index["alias"].get(squashed) if squashed else None
    if key is None:
        key = index.get("alias_zh", {}).get(_zh_text(name))
    return key


def item_key(name: Any) -> str:
    """Canonical item tag slug (internal key such as "bfury"; squashed fallback)."""
    return resolve_item_key(name) or _squash_item(name)


def is_known_item(name: Any) -> bool:
    """Whether *name* resolves to an item of the catalogs (any identifier form)."""
    return resolve_item_key(name) is not None


def resolve_item_name(name: Any) -> str | None:
    """Canonical display name for an item identifier (None if unknown)."""
    key = resolve_item_key(name)
    if not key:
        return None
    return _item_index()["by_key"].get(key) or None


# ---------------------------------------------------------------------------
# Display names for text (never raise, never return "")
# ---------------------------------------------------------------------------

_PREFIX_RE = re.compile(r"^(npc_dota_hero_|npc_dota_|item_)")


def _prettify(identifier: Any) -> str:
    """'npc_dota_hero_new_guy' / 'item_some_thing' -> 'New Guy' / 'Some Thing'."""
    text = str(identifier or "").strip()
    if not text:
        return ""
    if " " in text or text != text.lower():
        return text  # already a display name
    words = _PREFIX_RE.sub("", text).replace("_", " ").split()
    return " ".join(w.capitalize() for w in words) or text


def _is_zh(lang: Any) -> bool:
    """Whether display names are wanted in Chinese (every language but "en",
    like the rest of the coach: an unknown language gives the zh report)."""
    return lang != "en"


def hero_display_name(identifier: Any, lang: str = "en") -> str:
    """Display name of a hero for any identifier form. Never raises.

    Accepts display names, heroes.json slugs, internal / npc names and the
    aliases in ``_HERO_ALIASES`` ("skeleton_king", "npc_dota_hero_zuus",
    "wk" ...). ``lang="zh"`` gives the game's Simplified Chinese name from
    the committed catalog ("冥魂大帝") and the English name when the catalog
    has none. Unknown identifiers come back prettified ("new_hero" -> "New
    Hero"), an empty one as "". The result is for text only: lookups keep
    using the English name or the slug (:func:`resolve_hero_name`,
    :func:`hero_slug`).
    """
    try:
        entry = _hero_entry(identifier)
        if entry is None:
            return _prettify(identifier)
        return (_is_zh(lang) and entry.get("name_zh")) or entry["name"]
    except Exception:  # pragma: no cover - defensive
        return _prettify(identifier)


def item_display_name(identifier: Any, lang: str = "en") -> str:
    """Display name of an item for any identifier form. Never raises.

    Accepts display names, item keys with or without ``item_``, the replay
    class-name leftovers ("boots_of_travel", "dustof_appearance",
    "invisibility_edge", "empty_bottle") and common abbreviations ("bkb").
    ``lang="zh"`` gives the catalog's Chinese name ("黑皇杖", "图纸（达贡之神力）")
    and the English one when there is none. A key missing from every catalog
    is kept and prettified ("new_item" -> "New Item"); a recipe the catalog
    does not name is named after its item ("recipe_new_item" -> "New Item
    Recipe" / "图纸（New Item）", the wording of the game's own recipe names).
    """
    try:
        key = resolve_item_key(identifier) or str(identifier or "").strip().lower()
        if key.startswith("item_"):
            key = key[len("item_"):]
        if _is_zh(lang):
            name_zh = _item_index().get("zh", {}).get(key)
            if name_zh:
                return name_zh
        # Before the catalog name: the catalog lists recipes in English only
        # where the game has no Chinese name for them.
        if key.startswith("recipe_") and len(key) > len("recipe_"):
            label = _RECIPE_LABELS["zh" if _is_zh(lang) else "en"]
            return label.format(item_display_name(key[len("recipe_"):], lang))
        return resolve_item_name(identifier) or _prettify(identifier)
    except Exception:  # pragma: no cover - defensive
        return _prettify(identifier)


def unit_display_name(identifier: Any, lang: str = "en") -> str:
    """Display name of any unit that can kill or damage a hero. Never raises.

    Heroes resolve like :func:`hero_display_name`. Other npc names get a
    localized generic label ("npc_dota_creep_badguys_melee" -> "Lane creep" /
    "小兵", towers, Roshan, neutrals ...); anything else is prettified.
    """
    try:
        if resolve_hero_name(identifier):
            return hero_display_name(identifier, lang)
        text = str(identifier or "").strip().lower()
        if text.startswith("npc_dota_"):
            for needle, (zh, en) in _UNIT_LABELS:
                if needle in text:
                    return en if lang == "en" else zh
        return _prettify(identifier)
    except Exception:  # pragma: no cover - defensive
        return _prettify(identifier)


# Valve writes talent names as templates: "+{s:bonus_damage} Shukuchi Damage",
# "-{s:bonus_AbilityCooldown}s Hex Cooldown". The number is a value of the
# ability the talent modifies; OpenDota's constants carry neither that value
# nor a resolved name, so a token can only be filled when the talent's own
# entry has the value (``attrib``, as on every other ability entry).
_TEMPLATE_TOKEN = r"\{s:([^{}]*)\}"
_TEMPLATE_RE = re.compile(_TEMPLATE_TOKEN)
_TEMPLATE_UNIT = r"(?:%|[sx]\b)?"  # "{..}%", "{..}s", "{..}x"
# A name that opens with its number(s): "+{a} Name", "{a}% Name", "+{a}/{b}s Name"
_TEMPLATE_LEAD_RE = re.compile(
    r"^([+-]?)" + _TEMPLATE_TOKEN + _TEMPLATE_UNIT
    + r"(?:\s*/\s*[+-]?" + _TEMPLATE_TOKEN + _TEMPLATE_UNIT + r")*\s+(?=\S)")
_PLAIN_NUMBER_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")


def _template_values(attrib: Any) -> dict[str, str]:
    """``{key: number}`` for the plain numeric values of an ``attrib`` list."""
    values: dict[str, str] = {}
    for entry in attrib if isinstance(attrib, list) else []:
        if not isinstance(entry, dict) or not entry.get("key"):
            continue
        value = entry.get("value")
        text = str(value).strip() if isinstance(value, (str, int, float)) and not isinstance(value, bool) else ""
        if _PLAIN_NUMBER_RE.fullmatch(text):
            values[str(entry["key"])] = text
    return values


def clean_ability_name(name: Any, values: dict | None = None) -> str:
    """Ability / talent display name without ``{s:...}`` template tokens.

    When *values* has a number for every token they are filled in
    ("+{s:value} Damage" -> "+25 Damage"). Otherwise the number is unknown:
    a name that opens with it simply loses it and keeps its sign
    ("+{s:bonus_damage} Shukuchi Damage" -> "+Shukuchi Damage",
    "-{s:bonus_AbilityCooldown}s Hex Cooldown" -> "-Hex Cooldown"); a number
    inside the sentence becomes "X", because dropping it would change what
    the name says ("Tidebringer Ignores {s:..}% Armor" -> "Tidebringer
    Ignores X% Armor"). A name without tokens is returned as it is.
    """
    text = " ".join(str(name or "").split())
    if "{s:" not in text:
        return text
    keys = _TEMPLATE_RE.findall(text)
    if isinstance(values, dict) and keys and all(key in values for key in keys):
        def fill(match: re.Match) -> str:
            value = str(values[match.group(1)])
            signed = text[:match.start()].endswith(("+", "-"))  # the template brings its own sign
            return value.lstrip("+-") if signed else value
        return _TEMPLATE_RE.sub(fill, text)
    lead = _TEMPLATE_LEAD_RE.match(text)
    if lead:
        rest = text[lead.end():]
        if "{s:" not in rest and not rest.lower().startswith("of "):
            return lead.group(1) + rest
    text = _TEMPLATE_RE.sub("X", text)
    return text if re.search(r"[A-Za-z]{2}", text) else ""  # only the number: no name


def raw_ability_name(raw: Any) -> str:
    """Display name of a raw dotaconstants ability entry, without template
    tokens ("" when it has no name). Tokens are filled from the entry's own
    ``attrib`` values where it has them, see :func:`clean_ability_name`."""
    if not isinstance(raw, dict):
        return ""
    return clean_ability_name(raw.get("dname"), _template_values(raw.get("attrib")))


_NAME_NUMBER = r"([0-9]+(?:[.,][0-9]+)*)"


def template_values_from_name(template: Any, resolved: Any) -> dict[str, str]:
    """The numbers a resolved name states for the tokens of its template:
    ("+{s:bonus_damage} Counter Helix Damage", "+40 Counter Helix Damage")
    -> {"bonus_damage": "40"}. {} when the two do not line up (another
    wording, or a token that stands for no plain number)."""
    template = " ".join(str(template or "").split())
    resolved = " ".join(str(resolved or "").split())
    keys = _TEMPLATE_RE.findall(template)
    if not keys or "{s:" in resolved:
        return {}
    pattern = _NAME_NUMBER.join(re.escape(part) for part in _TEMPLATE_RE.split(template)[::2])
    match = re.fullmatch(pattern, resolved, flags=re.IGNORECASE)
    if match is None:
        return {}
    values: dict[str, str] = {}
    for key, value in zip(keys, match.groups()):
        if values.setdefault(key, value) != value:
            return {}
    return values


def filled_ability_name(name: Any, attrib: Any, values: dict[str, str] | None = None) -> str:
    """*name* (in any language) with every template token filled from
    *values* and the ``attrib`` values of its ability; "" when it has no
    name or a number is unknown. For localised names, where
    :func:`clean_ability_name`'s way of shortening an unfilled English name
    does not apply."""
    text = " ".join(str(name or "").split())
    known = {**_template_values(attrib), **(values or {})}
    if any(key not in known for key in _TEMPLATE_RE.findall(text)):
        return ""
    return clean_ability_name(text, known)


def _talent_fallback_name(key: str) -> str:
    """Readable name for a talent no catalog knows: the key without its prefix."""
    text = key.strip().lower()
    for prefix in (TALENT_PREFIX + "unique_", TALENT_PREFIX):
        if text.startswith(prefix) and len(text) > len(prefix):
            return _prettify(text[len(prefix):])
    return _prettify(key)


def ability_keys(data: dict) -> list[str]:
    """Every ability key an ability entry answers to (its own key first).

    A hero can have several keys with one display name (nevermore_shadowraze1
    / 2 / 3); the table keeps one entry for them (``same_name_keys``).
    """
    keys = [str(data["key"])] if data.get("key") else []
    return keys + [str(k) for k in data.get("same_name_keys") or [] if k]


def _build_ability_index() -> dict[str, str]:
    """Ability key -> display name (fetched data first, then the frontend catalog).

    Talents are included; no name keeps a template token (see
    :func:`clean_ability_name`).
    """
    names: dict[str, str] = dict(_BUILTIN_ABILITY_NAMES)
    for abilities in _load_dict(GAMEDATA_DIR / "abilities.json").values():
        if not isinstance(abilities, dict):
            continue
        for name, data in abilities.items():
            if isinstance(data, dict) and data.get("key") and name != data["key"]:
                for key in ability_keys(data):
                    names.setdefault(key, str(name))
    for key, info in _load_dict(GAMEDATA_DIR / TALENTS_FILE).items():
        if isinstance(info, dict) and info.get("name"):
            name = clean_ability_name(info["name"], info.get("values"))
            if name:
                names.setdefault(str(key), name)
    static = _load_dict(STATIC_DATA_DIR / "abilities.json", "by_key")
    for key, info in static.items():
        if isinstance(info, dict) and info.get("name"):
            name = clean_ability_name(info["name"])
            if name:
                names.setdefault(str(key), name)
    for data in (_load_dict(FALLBACK_DIR / "ability_stats.json", "abilities"),
                 _load_dict(EDITORIAL_DIR / "coaching_tips_abilities.json")):
        for key, info in data.items():
            if isinstance(info, dict) and info.get("ability_name"):
                names.setdefault(str(key), str(info["ability_name"]))
    return names


def resolve_ability_name(key: Any) -> str | None:
    """Display name of an ability or talent key such as "lion_impale" (None if unknown)."""
    text = str(key or "").strip().lower()
    if not text:
        return None
    return _cached("ability_index", _build_ability_index).get(text)


def _build_ability_zh_index() -> dict:
    """Chinese ability / talent names of the committed catalog:
    ``{"by_key": {key: (English name, Chinese name)}, "by_name": {english
    name, lowercase: Chinese name}}``.

    A display name several abilities share takes the Chinese name most of
    them have, the first one on a tie (:func:`ability_display_name` asks
    the hero's own abilities first when it is told the hero). A talent is
    found under its full name, numbers included.
    """
    by_key: dict[str, tuple[str, str]] = {}
    votes: dict[str, dict[str, int]] = {}
    static = _load_dict(STATIC_DATA_DIR / "abilities.json", "by_key")
    for key, info in static.items():
        if not isinstance(info, dict):
            continue
        name, name_zh = clean_ability_name(info.get("name")), _zh_text(info.get("name_zh"))
        if not name or not name_zh or "{s:" in name_zh:
            continue
        key = str(key).lower()
        by_key[key] = (name, name_zh)
        counts = votes.setdefault(name.lower(), {})
        counts[name_zh] = counts.get(name_zh, 0) + 1
    by_name = {name: max(counts, key=counts.get) for name, counts in votes.items()}
    return {"by_key": by_key, "by_name": by_name}


_NUMBER_IN_NAME = re.compile(r"\d+(?:\.\d+)?")


def _ability_name_zh(key: str, name: str) -> str:
    """Chinese name of ability *key* whose English name is *name* ("" = none).

    Talent keys are reused between patches with other numbers, and the
    fetched game data can be newer than the committed catalog: when the
    English names differ only in their numbers, the Chinese name takes the
    numbers of *name* ("+30 Counter Helix Damage" with the catalog's "+40
    ..." / "+40 反击螺旋伤害" gives "+30 反击螺旋伤害"); when they differ in
    wording it is another talent by now and has no Chinese name here.
    """
    entry = _cached("ability_zh_index", _build_ability_zh_index)["by_key"].get(key)
    if not entry:
        return ""
    catalog_name, name_zh = entry
    if not name or name == catalog_name:
        return name_zh
    if _NUMBER_IN_NAME.sub("#", name) != _NUMBER_IN_NAME.sub("#", catalog_name):
        return ""
    # The Chinese wording can state the numbers in another order: match them by value.
    pairs = list(zip(_NUMBER_IN_NAME.findall(catalog_name), _NUMBER_IN_NAME.findall(name)))
    complete = True

    def swap(match: re.Match) -> str:
        nonlocal complete
        for index, (ours, theirs) in enumerate(pairs):
            if ours == match.group(0):
                del pairs[index]
                return theirs
        complete = False
        return match.group(0)

    text = _NUMBER_IN_NAME.sub(swap, name_zh)
    return text if complete and not pairs else ""


def _hero_ability_key(hero: Any, name: str) -> str:
    """Key of the ability *hero* has under the English display name *name* ("" = none)."""
    abilities = load_abilities().get(resolve_hero_name(hero) or "") or {}
    data = abilities.get(name)
    return str(data.get("key") or "").lower() if isinstance(data, dict) else ""


def ability_display_name(identifier: Any, lang: str = "en", hero: Any = None) -> str:
    """Display name of an ability key or damage inflictor. Never raises.

    ``"attack"`` (the plain-attack inflictor) is localized ("Attack" /
    "普通攻击"); item inflictors ("item_dagon", "dagon") resolve to the item
    name; unknown keys are prettified. Talent keys ("special_bonus_*")
    resolve to the talent's name without template tokens (see
    :func:`clean_ability_name`); a talent no catalog knows is named after
    its key ("special_bonus_unique_lion_11" -> "Lion 11"). The map actions
    in :data:`BUILTIN_ABILITY_LABELS` ("ability_capture" -> "Capture
    Outpost" / "占领前哨") are localized as well. ``lang="zh"`` gives the
    catalog's Chinese name of the ability ("lion_impale" / "Earth Spike" ->
    "裂地尖刺"), the English one when the catalog has none; an English
    display name is accepted in place of the key. With *hero* (any
    identifier form) such a display name is looked up among that hero's
    abilities first, then among all abilities and talents: "Blink" is
    Anti-Mage's "闪烁" there, not the item an inflictor of that name
    would be.
    """
    try:
        text = str(identifier or "").strip()
        low = text.lower()
        zh = _is_zh(lang)
        if low == ATTACK_INFLICTOR:
            return _ATTACK_LABELS["zh" if zh else "en"]
        if zh and hero:
            key = _hero_ability_key(hero, text)
            if key:
                return _ability_name_zh(key, text) or text
            # A display name the hero's table does not have (a talent, a name of
            # another patch): an ability of that name before an item of that name.
            name_zh = _cached("ability_zh_index", _build_ability_zh_index)["by_name"].get(" ".join(low.split()))
            if name_zh and resolve_ability_name(text) is None:
                return name_zh
        if zh and low in BUILTIN_ABILITY_LABELS:
            return BUILTIN_ABILITY_LABELS[low][0]
        name = resolve_ability_name(text)
        if name:
            return (zh and _ability_name_zh(low, name)) or name
        item_name = resolve_item_name(text)
        if item_name:
            return item_display_name(text, lang) if zh else item_name
        if zh:  # an English display name instead of a key
            name_zh = _cached("ability_zh_index", _build_ability_zh_index)["by_name"].get(" ".join(low.split()))
            if name_zh:
                return name_zh
        if low.startswith(TALENT_PREFIX):
            return _talent_fallback_name(text)
        return _prettify(text)
    except Exception:  # pragma: no cover - defensive
        return _prettify(identifier)


def display_name(identifier: Any, lang: str = "en") -> str:
    """Display name for an identifier of unknown kind. Never raises.

    Tries hero, then ability / attack, then item, then other units; falls
    back to a prettified form. Use the specific helpers when the kind is
    known (a few item keys are also hero abbreviations). Like them it
    follows *lang*.
    """
    try:
        text = str(identifier or "").strip()
        if text.lower() == ATTACK_INFLICTOR:
            return _ATTACK_LABELS["zh" if _is_zh(lang) else "en"]
        if resolve_hero_name(text):
            return hero_display_name(text, lang)
        if resolve_ability_name(text):
            return ability_display_name(text, lang)
        if not resolve_item_name(text) and text.lower().startswith("npc_dota_"):
            return unit_display_name(text, lang)
        return item_display_name(text, lang)  # also names recipes; prettifies the rest
    except Exception:  # pragma: no cover - defensive
        return _prettify(identifier)


# ---------------------------------------------------------------------------
# Names in running text (English reference material, LLM answers)
# ---------------------------------------------------------------------------
#
# Text this code writes takes its names from the helpers above. Two kinds of
# text are not written here: the English reference knowledge shown to the
# model, and what the model writes back. For those the names are found in
# the text itself.

def _build_text_names() -> dict:
    """English display names as they stand in text, with their Chinese names.

    ``{"heroes": {name: zh}, "items": {name: zh}, "abilities": {hero name:
    {ability name: zh}}, "all": frozenset of every English name}``. Only
    names with a Chinese catalog entry are in the three tables. Items are
    the ones a shop sells plus every name of two or more words: the
    one-word names of enchantments and other drops ("Wise", "Vast",
    "Tough") are English words first.
    """
    heroes: dict[str, str] = {}
    everything: set[str] = set()
    index = _hero_index()
    for entry in index["by_internal"].values():
        everything.add(entry["name"])
        if entry.get("name_zh"):
            heroes[entry["name"]] = entry["name_zh"]
    # Earlier spellings of renamed heroes ("Outworld Devourer") are still written.
    static_heroes = _load_json(STATIC_DATA_DIR / "heroes.json")
    for name in (static_heroes.get("name_to_id") if isinstance(static_heroes, dict) else None) or {}:
        entry = _hero_entry(name)
        if entry is not None:
            everything.add(str(name))
            if entry.get("name_zh"):
                heroes.setdefault(str(name), entry["name_zh"])

    items: dict[str, str] = {}
    item_index = _item_index()
    costs = {str(key).lower(): info.get("cost") for key, info in
             _load_dict(STATIC_DATA_DIR / "items.json", "by_key").items() if isinstance(info, dict)}
    for key, name in item_index["by_key"].items():
        if not name:
            continue
        everything.add(name)
        name_zh = item_index["zh"].get(key)
        sold = isinstance(costs.get(key), (int, float)) and costs[key] > 0
        if name_zh and name_zh != name and (sold or " " in name):
            items.setdefault(name, name_zh)

    abilities: dict[str, dict[str, str]] = {}
    for hero, table in load_abilities().items():
        named: dict[str, str] = {}
        for name, data in table.items():
            everything.add(str(name))
            key = str(data.get("key") or "").lower() if isinstance(data, dict) else ""
            name_zh = _ability_name_zh(key, str(name)) if key else ""
            if name_zh and name_zh != name:
                named[str(name)] = name_zh
        if named:
            abilities[hero] = named
    return {"heroes": heroes, "items": items, "abilities": abilities,
            "all": frozenset(name for name in everything if len(name) >= 3)}


_CJK_CHARS = "\u3400-\u4dbf\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"


def _names_regex(names: Iterable[str]) -> re.Pattern | None:
    """Matches any of *names* as a whole name (the longest one first), with
    an English possessive when it has one ("Lina's"): group 1 is the name,
    group 2 the possessive."""
    ordered = sorted({name for name in names if name}, key=lambda name: (-len(name), name))
    if not ordered:
        return None
    return re.compile(r"(?<![A-Za-z])(" + "|".join(re.escape(name) for name in ordered) + r")('s)?(?![A-Za-z])")


# A replaced name is marked while the blanks around it are tidied up: a blank
# sets a Latin name apart from Chinese text ("被 Lina 击杀") and has no place
# next to the Chinese name; two names with only a blank between them are a
# list. Private-use characters: no text contains them.
_NAME_START, _NAME_END = "\ue000", "\ue001"
# "冥魂大帝（Wraith King）" / "Wraith King（冥魂大帝）": both names of one thing, once the
# English one is replaced. One is enough.
_NAME_TWICE_RE = re.compile(
    rf"([^\s（）(){_NAME_START}{_NAME_END}]{{2,}}?)\s*[（(]\s*{_NAME_START}\1{_NAME_END}\s*[)）]"
    rf"|{_NAME_START}([^{_NAME_END}]+){_NAME_END}\s*[（(]\s*\2\s*[)）]")
_BLANK_BETWEEN_NAMES_RE = re.compile(rf"(?<={_NAME_END}) +(?={_NAME_START})")
_BLANK_BEFORE_NAME_RE = re.compile(rf"(?<=[{_CJK_CHARS}]) +(?={_NAME_START})")
_BLANK_AFTER_NAME_RE = re.compile(rf"(?<={_NAME_END}) +(?=[{_CJK_CHARS}])")


# Patterns kept per set of heroes (one per match being coached; compiling one takes ~30 ms).
_TEXT_NAME_PATTERNS_MAX = 64
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2}")


def _text_names(heroes: Iterable[Any] | None) -> tuple[re.Pattern | None, dict[str, str]]:
    """``(pattern, {English name: Chinese name})`` for text about *heroes*.

    Every hero and item name is looked for; ability names only of *heroes*
    (any identifier form): among all 780 of them are too many English words
    ("Rage", "Return", "Fire Remnant" is fine, "Leap" is not) to replace
    them in a text that is not about their hero.
    """
    table = _cached("text_names", _build_text_names)
    scope = tuple(sorted({name for name in (resolve_hero_name(h) for h in heroes or ()) if name}))
    patterns = _cached("text_name_patterns", dict)
    hit = patterns.get(scope)
    if hit is None:
        names = dict(table["items"])
        for hero in scope:
            names.update(table["abilities"].get(hero) or {})
        names.update(table["heroes"])     # a hero's name wins over an ability of the same name
        hit = (_names_regex(names), names)
        with _cache_lock:
            while len(patterns) >= _TEXT_NAME_PATTERNS_MAX:   # the oldest one goes
                patterns.pop(next(iter(patterns)), None)
            patterns[scope] = hit
    return hit


_HAN_NAME_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{2}")


def find_zh_names(text: Any) -> tuple[list[str], list[str]]:
    """``(hero slugs, item keys)`` whose Chinese catalog name is written in *text*.

    For a question or a tool argument that names heroes and items the way a
    zh report does ("冥界亚龙的毒怎么解", "黑皇杖"). The longest name wins
    where two overlap ("图纸（远行鞋）" is the recipe). Never raises.
    """
    try:
        text = str(text or "")
        if not _HAN_NAME_RE.search(text):
            return [], []
        candidates = [(name, "hero", internal) for name, internal in _hero_index().get("alias_zh", {}).items()]
        candidates += [(name, "item", key) for name, key in _item_index().get("alias_zh", {}).items()]
        found: list[tuple[int, str, str]] = []
        for name, kind, value in sorted(candidates, key=lambda c: (-len(c[0]), c[0])):
            at = text.find(name)
            if at < 0 or not _HAN_NAME_RE.search(name):
                continue
            found.append((at, kind, value))
            text = text.replace(name, " " * len(name))
        found.sort()
        return ([v for _at, kind, v in found if kind == "hero"], [v for _at, kind, v in found if kind == "item"])
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Chinese name lookup failed: %s", exc)
        return [], []


def name_glossary(text: Any, lang: str = "zh", heroes: Iterable[Any] | None = None) -> list[tuple[str, str]]:
    """``[(English name, Chinese name)]`` of the hero, item and ability names
    written in English in *text*, in order of appearance. Never raises.

    For a zh prompt: the reference knowledge is partly English, and the
    model is told to write the Chinese names. Ability names are only looked
    for among the abilities of *heroes* (see :func:`_text_names`). [] for
    ``lang="en"``.
    """
    try:
        text = str(text or "")
        if not _is_zh(lang) or not _LATIN_WORD_RE.search(text):
            return []
        pattern, names = _text_names(heroes)
        if pattern is None:
            return []
        found = dict.fromkeys(match.group(1) for match in pattern.finditer(text))
        return [(name, names[name]) for name in found]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Name glossary failed: %s", exc)
        return []


def localize_names(text: Any, lang: str = "zh", heroes: Iterable[Any] | None = None) -> str:
    """*text* with the English hero, item and ability names it contains
    replaced by their Chinese names (``lang="zh"``; unchanged for "en").
    Never raises.

    For text that was not written through the display-name helpers: an LLM
    answer of a zh report, a title taken from a data file. A name without a
    Chinese catalog entry stays as it is, and so does every other English
    word. "冥魂大帝（Wraith King）" becomes "冥魂大帝", and the blanks that
    set an English name apart from Chinese text go with it ("被 Lina 击杀"
    -> "被莉娜击杀"). Ability names are only replaced for *heroes* (see
    :func:`_text_names`).
    """
    try:
        if not isinstance(text, str) or not _is_zh(lang) or not _LATIN_WORD_RE.search(text):
            return text if isinstance(text, str) else str(text or "")
        pattern, names = _text_names(heroes)
        if pattern is None or _NAME_START in text or _NAME_END in text:
            return text
        marked = pattern.sub(
            lambda match: _NAME_START + names[match.group(1)] + _NAME_END + ("的" if match.group(2) else ""), text)
        if marked == text:
            return text
        marked = _NAME_TWICE_RE.sub(lambda m: _NAME_START + (m.group(1) or m.group(2)) + _NAME_END, marked)
        marked = _BLANK_BETWEEN_NAMES_RE.sub("、", marked)
        marked = _BLANK_AFTER_NAME_RE.sub("", _BLANK_BEFORE_NAME_RE.sub("", marked))
        return marked.replace(_NAME_START, "").replace(_NAME_END, "")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Name localisation failed: %s", exc)
        return text if isinstance(text, str) else str(text or "")


def strip_names(text: Any) -> str:
    """*text* without the English hero, item and ability names of the catalogs.

    What is left tells the language a text is written in: a Chinese
    sentence about "Aghanim's Scepter" is not half English. Never raises.
    """
    try:
        text = str(text or "")
        if not _LATIN_WORD_RE.search(text):
            return text
        pattern = _cached("all_names_pattern", lambda: _names_regex(_cached("text_names", _build_text_names)["all"]))
        return pattern.sub(" ", text) if pattern is not None else text
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Name stripping failed: %s", exc)
        return str(text or "")


# ---------------------------------------------------------------------------
# Patch version
# ---------------------------------------------------------------------------

def _valid_patch(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in ("", "unknown", "none", "null") else text


def _meta() -> dict:
    """meta.json of the game data on disk ({} when missing or unreadable). Cached, read-only."""
    return _cached("meta", lambda: _load_dict(GAMEDATA_DIR / "meta.json"))


def get_fetched_patch() -> str:
    """Major version recorded by the last successful game-data fetch ("" if unknown)."""
    # Cache "" too (store as a tuple so the falsy value is not rebuilt).
    return _cached("fetched_patch",
                   lambda: (patch_major(_valid_patch(_meta().get("patch_version"))),))[0]


def get_talent_patch() -> str:
    """Patch whose talents the fetched game data names.

    "" when there is no talent table: nothing was fetched yet, or the files
    were written before it existed. Talent names then come from the bundled
    catalog or from the key (see :func:`ability_display_name`).
    """
    def build() -> str:
        return get_fetched_patch() if (GAMEDATA_DIR / TALENTS_FILE).is_file() else ""
    return _cached("talent_patch", lambda: (build(),))[0]


def get_current_patch() -> str:
    """Current major version: the fetched meta patch, else :data:`DEFAULT_PATCH`.

    Always without a letter (7.41, never 7.41f): patch-specific knowledge is
    stamped and compared with it. The full version is :func:`get_patch_label`.
    """
    return get_fetched_patch() or DEFAULT_PATCH


def get_patch_version() -> str:
    """Backward-compatible alias of :func:`get_current_patch`."""
    return get_current_patch()


def get_schema_version() -> int:
    """``schema_version`` of the game data on disk (0: none, or written
    before the marker existed)."""
    def build() -> int:
        value = _meta().get("schema_version")
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0
    # Cache 0 too (store as a tuple so the falsy value is not rebuilt).
    return _cached("schema_version", lambda: (build(),))[0]


def _check_patch_list(patches: Any) -> list:
    """*patches* when it is a usable OpenDota patch list, else RuntimeError."""
    if not isinstance(patches, list) or not patches:
        raise RuntimeError("Empty patch data")
    newest = patches[-1]
    if not _valid_patch(newest.get("name") if isinstance(newest, dict) else None):
        raise RuntimeError("Malformed patch data")
    return patches


def _remote_patches(revision: Any = None) -> list:
    """The patch list of the constants source, oldest first: ``[{"name",
    "date", "id"}]`` (major versions only). Raises on errors."""
    return _download_set(("patch",), lambda payloads: _check_patch_list(payloads["patch"]),
                         timeout=15, sources=_sources(revision))[2]


def _valve_releases() -> list[tuple[float, str]]:
    """Valve's patch feed as ``[(release time, label)]``, oldest first, with
    letter patches (7.41, 7.41a ... 7.41f). Raises on errors."""
    data = _http_get_json(VALVE_PATCH_FEED, timeout=15)
    rows = data.get("patches") if isinstance(data, dict) else None
    releases: list[tuple[float, str]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        label = _valid_patch(row.get("patch_number") or row.get("patch_name"))
        stamp = row.get("patch_timestamp")
        if (_PATCH_LABEL_RE.fullmatch(label) and isinstance(stamp, (int, float))
                and not isinstance(stamp, bool) and stamp > 0):
            releases.append((float(stamp), label))
    if not releases:
        raise RuntimeError("Unexpected answer of Valve's patch feed")
    return sorted(releases)


def _remote_state() -> dict[str, Any]:
    """What the network says about the game version and the data source.

    ``{"releases": Valve's list, "revision": {"sha", "commit", "date"} of
    the source, "patches": the source's major-version list}``. A part that
    could not be read is None (logged): each one has a fallback. Never raises.
    """
    state: dict[str, Any] = {"releases": None, "revision": None, "patches": None}
    # (the revision first: the mirror is then asked for that commit)
    for part, read in (("releases", _valve_releases), ("revision", _source_revision),
                       ("patches", lambda: _remote_patches(state["revision"]))):
        try:
            state[part] = read()
        except Exception as exc:
            logger.warning("Patch check: %s unavailable: %s", part, exc)
    return state


def _newest_major(patches: Any) -> str:
    newest = patches[-1] if isinstance(patches, list) and patches else None
    return _valid_patch(newest.get("name")) if isinstance(newest, dict) else ""


def _patch_status(state: dict[str, Any]) -> tuple[str, str]:
    """``(remote version, reason)`` for what :func:`_remote_state` read.

    *remote version* is the newest game version the network reports: Valve's
    full label (7.41f), else the newest major version of the source's patch
    list. *reason* says why the local game data should be fetched again and
    is "" when it is up to date:

    - ``no_data`` / ``outdated_format``: nothing usable on disk.
    - ``fallback_source``: the last fetch had to use the OpenDota API (the
      npm release) while the current source is reachable now.
    - ``new_source_revision``: the source published a new build since.
    - ``new_patch`` / ``new_release``: only without a readable source
      revision — the source's patch list names another major version, or
      Valve released a (letter) patch after the last fetch.

    Raises when the network told nothing about the version.
    """
    releases, revision = state.get("releases"), state.get("revision")
    table_major = _newest_major(state.get("patches"))
    remote = releases[-1][1] if releases else table_major
    if not remote:
        raise RuntimeError("Neither Valve's patch feed nor a constants source answered")
    local, meta = get_fetched_patch(), _meta()
    if not local:
        return remote, "no_data"
    if get_schema_version() < GAMEDATA_SCHEMA_VERSION:
        return remote, "outdated_format"
    if revision:
        # The source's build files are the data: nothing new without a new revision.
        if str(meta.get("source") or OPENDOTA_SOURCE) == OPENDOTA_SOURCE:
            return remote, "fallback_source"
        return remote, "" if revision["sha"] == meta.get("source_revision") else "new_source_revision"
    if table_major and table_major != local:
        return remote, "new_patch"
    if releases and releases[-1][0] > (_parse_utc(meta.get("last_fetched")) or 0.0):
        return remote, "new_release"
    return remote, ""


def check_patch_version() -> tuple[str, bool]:
    """Check the remote game version and data source against the local data.

    Returns ``(remote_version, needs_update)``: the newest version the
    network reports (Valve's full label such as 7.41f; the major version
    when only the source's patch list answered) and whether the game data
    should be fetched again (see :func:`_patch_status`; also True for files
    written in an older format, :data:`GAMEDATA_SCHEMA_VERSION`). The
    downloaded lists also refresh the stored patch tables (see
    :func:`patch_for_time`, :func:`get_patch_info`). Raises on network errors.
    """
    state = _remote_state()
    remote, reason = _patch_status(state)
    _save_patch_table(state["patches"], state["releases"])
    return remote, bool(reason)


# ---------------------------------------------------------------------------
# Patch tables (which version is current, which one a match was played on)
# ---------------------------------------------------------------------------

_UTC_FMT = "%Y-%m-%dT%H:%M:%S"

# (file signature, tables) of patches.json, see _patch_tables
_patch_table_cache: tuple[tuple, dict[str, Any]] | None = None
_NO_TABLES: dict[str, Any] = {"fetched_at": 0.0, "patches": [], "releases_fetched_at": 0.0, "releases": []}


def _parse_utc(value: Any) -> float | None:
    """Unix time of an ISO-8601 UTC stamp ("2026-03-24T00:50:59.580Z"); None if malformed."""
    try:
        return float(calendar.timegm(time.strptime(str(value or "").strip()[:19], _UTC_FMT)))
    except (ValueError, OverflowError):
        return None


def _utc_stamp(unix_time: float | None = None) -> str:
    return time.strftime(_UTC_FMT + "Z", time.gmtime(unix_time))


def _patch_table_payload(patches: Any) -> dict | None:
    """Content of patches.json for an OpenDota patch list (None: nothing usable)."""
    rows = [
        {"name": _valid_patch(patch.get("name")), "date": str(patch.get("date"))}
        for patch in (patches if isinstance(patches, list) else [])
        if isinstance(patch, dict) and _valid_patch(patch.get("name"))
        and _parse_utc(patch.get("date")) is not None
    ]
    if not rows:
        return None
    return {"fetched_at": _utc_stamp(), "patches": rows}


def _save_patch_table(patches: Any, releases: Any = None) -> None:
    """Store downloaded patch lists as patches.json: the source's major
    versions (*patches*) and Valve's releases (*releases*). A list that is
    missing or unusable keeps the stored one. Never raises: the tables only
    label versions, a failed write must not fail the caller."""
    try:
        update = _patch_table_payload(patches) or {}
        if isinstance(releases, list) and releases:
            update["releases_fetched_at"] = _utc_stamp()
            update["releases"] = [{"name": label, "date": _utc_stamp(released)}
                                  for released, label in releases]
        if update:
            stored = _load_dict(GAMEDATA_DIR / PATCHES_FILE)
            kept = {key: stored[key] for key in ("fetched_at", "patches", "releases_fetched_at", "releases")
                    if key in stored}
            _save_json(GAMEDATA_DIR / PATCHES_FILE, {**kept, **update})
    except Exception as exc:
        logger.warning("Could not store the patch table: %s", exc)


def _patch_rows(rows: Any) -> list[tuple[float, str]]:
    out = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        released, name = _parse_utc(row.get("date")), _valid_patch(row.get("name"))
        if released is not None and name:
            out.append((released, name))
    return sorted(out)


def _patch_tables() -> dict[str, Any]:
    """patches.json as ``{"fetched_at", "patches", "releases_fetched_at",
    "releases"}``: download times and ``[(release time, name)]`` lists,
    oldest first (major versions / Valve's releases with letter patches).

    Re-read whenever the file changes (one ``stat`` per call), so a table
    refreshed by another process counts at once. Empty lists without a file.
    """
    global _patch_table_cache
    path = GAMEDATA_DIR / PATCHES_FILE
    try:
        stat = path.stat()
    except OSError:
        return _NO_TABLES
    signature = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _patch_table_cache
    if cached is not None and cached[0] == signature:
        return cached[1]
    data = _load_json(path)
    data = data if isinstance(data, dict) else {}
    tables = {
        "fetched_at": _parse_utc(data.get("fetched_at")) or 0.0,
        "patches": _patch_rows(data.get("patches")),
        "releases_fetched_at": _parse_utc(data.get("releases_fetched_at")) or 0.0,
        "releases": _patch_rows(data.get("releases")),
    }
    _patch_table_cache = (signature, tables)
    return tables


def _name_at(rows: list[tuple[float, str]], fetched_at: float, unix_time: Any) -> str:
    """Name of the last row released at or before *unix_time* ("" when it
    cannot be told, see :func:`patch_for_time`)."""
    if isinstance(unix_time, bool):
        return ""
    when = float(unix_time)
    if not when > 0:  # also NaN
        return ""
    name = ""
    newest = False
    for index, (released, patch) in enumerate(rows):
        if released > when:
            break
        name, newest = patch, index == len(rows) - 1
    if newest and when > fetched_at + float(config.GAMEDATA_STALE_DAYS) * 86400:
        return ""
    return name


def patch_for_time(unix_time: Any) -> str:
    """Game version a match that started at *unix_time* was played on.

    The name of the last patch released at or before that time, from the
    stored OpenDota patch table (major versions only, no letter patches;
    :func:`patch_label_for_time` has those). "" when it cannot be told: no
    table yet (nothing was ever fetched), not a valid time, a time before
    the first entry, or a time after the newest known patch while the table
    is too old to rule out a newer one (it was downloaded more than
    GAMEDATA_STALE_DAYS before that time). Never raises and never touches
    the network.
    """
    try:
        tables = _patch_tables()
        return _name_at(tables["patches"], tables["fetched_at"], unix_time)
    except Exception:  # pragma: no cover - defensive
        return ""


def patch_label_for_time(unix_time: Any) -> str:
    """Full game version, with its letter, at *unix_time* (7.41b for a match
    of 2026-04-08), from Valve's stored release list.

    "" under the conditions of :func:`patch_for_time`, and whenever Valve's
    feed was never downloaded. Valve's release times are a few hours later
    than OpenDota's, so on a release day the two may name different
    versions. Never raises and never touches the network.
    """
    try:
        tables = _patch_tables()
        return _name_at(tables["releases"], tables["releases_fetched_at"], unix_time)
    except Exception:  # pragma: no cover - defensive
        return ""


def get_patch_info() -> dict:
    """The game version as far as it is known locally. Never raises, no network.

    - ``patch``: major version the local game data and the patch-specific
      knowledge are for (:func:`get_current_patch`).
    - ``label``: the game's current full version (7.41f): the newest release
      of Valve's patch feed as last downloaded; before the first download the
      label recorded with the game data, else the built-in default.
    - ``released``: UTC release time of ``label`` (ISO 8601, "" if unknown).
    - ``checked_at``: when Valve's feed was last downloaded ("" never).
    - ``data_label``: game version that was current when the data source
      last published, i.e. the newest version the local game data can
      reflect ("" when unknown: no game data, or a source without a
      revision date such as the OpenDota API).
    - ``data_date``: date of that source revision ("" if unknown).
    - ``data_current``: True when ``data_label`` is ``label``, False when the
      game data is known to be older than the current version, None when
      that cannot be told.
    """
    info: dict[str, Any] = {
        "patch": DEFAULT_PATCH, "label": DEFAULT_PATCH_LABEL, "released": DEFAULT_PATCH_RELEASED,
        "checked_at": "", "data_label": "", "data_date": "", "data_current": None,
    }
    try:
        patch = get_current_patch()
        meta = _meta()
        tables = _patch_tables()
        data_label = _valid_patch(meta.get("data_label"))
        info.update(patch=patch, data_label=data_label, data_date=str(meta.get("source_date") or ""))
        if tables["releases"]:
            released, label = tables["releases"][-1]
            info.update(label=label, released=_utc_stamp(released),
                        checked_at=_utc_stamp(tables["releases_fetched_at"]))
        elif data_label:
            info.update(label=data_label, released="")
        elif patch != DEFAULT_PATCH:
            info.update(label=patch, released="")
        if data_label:
            info["data_current"] = data_label == info["label"]
        elif patch_major(info["label"]) != patch:
            info["data_current"] = False
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Patch info unavailable: %s", exc)
    return info


def get_patch_label() -> str:
    """The game's current full version with its letter (7.41f). Never raises.

    For display. Compare knowledge by :func:`get_current_patch` (the major
    version) instead: a letter patch does not invalidate it.
    """
    return str(get_patch_info()["label"])


# ---------------------------------------------------------------------------
# Fetch and save game data (Layer A)
# ---------------------------------------------------------------------------

def _to_num_list(val: Any) -> list:
    """Convert API value to a list of numbers. Handles str, list[str], list[int], etc."""
    if val is None or isinstance(val, bool):
        return []
    if isinstance(val, (int, float)):
        return [val]
    if isinstance(val, str):
        # Could be "50" or "50 / 60 / 70"
        parts = [p.strip() for p in val.replace("/", " ").split()]
        nums = []
        for p in parts:
            try:
                nums.append(float(p) if "." in p else int(p))
            except ValueError:
                pass
        return nums
    if isinstance(val, list):
        result = []
        for item in val:
            if isinstance(item, bool):
                continue
            if isinstance(item, (int, float)):
                result.append(item)
            elif isinstance(item, str):
                result.extend(_to_num_list(item))
        return result
    return []


def _behavior_text(raw: Any) -> str:
    if isinstance(raw, list):
        return " ".join(str(x) for x in raw)
    return str(raw or "")


def _ability_type(behavior: str) -> str:
    if "Passive" in behavior:
        return "passive"
    if "Autocast" in behavior:
        return "autocast"
    return "active"


_STUN_HEADER = "STUN DURATION:"
# Headers under which a ``damage`` attrib is what one cast deals as a whole.
_TOTAL_DAMAGE_LABELS = frozenset({
    "DAMAGE", "BASE DAMAGE", "IMPACT DAMAGE", "TOTAL DAMAGE", "FULL DAMAGE", "MAX DAMAGE",
})
_HEADER_NOTE_RE = re.compile(r"\([^)]*\)")  # "DAMAGE (EXORT):\N" -> "DAMAGE"


def _damage_field(header: str) -> str | None:
    """Field a ``damage`` attrib fills, judged by its (upper-case) header.

    ``damage`` for a plain total, ``damage_per_second`` for a "... PER
    SECOND" header, None for every other qualified number (per stack / per
    instance / bonus / attack damage, "HEAL/DAMAGE" ...): printed as "N
    damage" those would state something the ability does not do.
    """
    label = _HEADER_NOTE_RE.sub("", header.split(":", 1)[0]).strip()
    if "PER SECOND" in label:
        return "damage_per_second"
    return "damage" if not label or label in _TOTAL_DAMAGE_LABELS else None


def _attrib_stats(attrib: Any) -> dict[str, list]:
    """Damage / stun / duration numbers from an OpenDota ``attrib`` list.

    OpenDota keeps most ability numbers there, not in top-level fields.
    Only unambiguous entries are used: key ``damage`` (see
    :func:`_damage_field`); key ``stun_duration`` or a "STUN DURATION:"
    header; key ``duration`` (unless its header says it is the stun).
    All-zero values are placeholders and ignored.
    """
    stats: dict[str, list] = {}
    for entry in attrib if isinstance(attrib, list) else []:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key") or "")
        header = str(entry.get("header") or "").strip().upper()
        if key == "damage":
            field = _damage_field(header)
        elif key == "stun_duration" or header == _STUN_HEADER:
            field = "stun_duration"
        elif key == "duration":
            field = "duration"
        else:
            field = None
        if field is None:
            continue
        values = _to_num_list(entry.get("value"))
        if values and any(values) and (field not in stats or key == field):
            stats[field] = values
    return stats


def _repeats_qualified_damage(damage: list, attrib: Any) -> bool:
    """*damage* is the value of a ``damage`` attrib that is not a total."""
    return any(
        isinstance(entry, dict) and entry.get("key") == "damage"
        and _damage_field(str(entry.get("header") or "").strip().upper()) != "damage"
        and _to_num_list(entry.get("value")) == damage
        for entry in (attrib if isinstance(attrib, list) else [])
    )


def _complete_ability(entry: dict) -> dict:
    """Fill the derived fields of a normalised ability, in place.

    Idempotent. Runs at fetch time and again on load, which also upgrades
    files written before these fields existed (:data:`GAMEDATA_SCHEMA_VERSION`).
    """
    damage = entry.get("damage")
    # OpenDota "dmg": "0" means "see attrib", not zero damage. A stored list
    # that repeats a per-second / per-stack attrib was derived before the
    # headers were read: it is not the ability's damage either.
    if isinstance(damage, list) and (
            not any(damage) or _repeats_qualified_damage(damage, entry.get("attrib"))):
        del entry["damage"]
    for field, values in _attrib_stats(entry.get("attrib")).items():
        entry.setdefault(field, values)
    if "type" not in entry:
        behavior = _behavior_text(entry.get("behavior"))
        if behavior:
            entry["type"] = _ability_type(behavior)
    return entry


def _is_hidden(ability: dict) -> bool:
    """Hidden-behaviour ability (placeholder slot, sub-ability, shard / scepter grant)."""
    return "Hidden" in _behavior_text(ability.get("behavior"))


def _normalize_ability(raw: dict, ability_key: str) -> dict:
    """Normalize OpenDota ability fields to our schema."""
    result: dict[str, Any] = {"key": ability_key}

    if "cd" in raw:
        result["cooldown"] = _to_num_list(raw["cd"])
    if "mc" in raw:
        result["mana_cost"] = _to_num_list(raw["mc"])
    if "dmg" in raw:
        result["damage"] = _to_num_list(raw["dmg"])

    if "bkbpierce" in raw:
        val = raw["bkbpierce"]
        if isinstance(val, str):
            result["pierces_bkb"] = val == "Yes"
        elif isinstance(val, list):
            result["pierces_bkb"] = "Yes" in val
    if raw.get("dmg_type"):
        dtype_map = {"Magical": "magical", "Physical": "physical", "Pure": "pure"}
        dmg_type_val = raw["dmg_type"]
        if isinstance(dmg_type_val, list):
            dmg_type_val = dmg_type_val[0] if dmg_type_val else ""
        if isinstance(dmg_type_val, str):
            result["damage_type"] = dtype_map.get(dmg_type_val, dmg_type_val.lower())

    for field in ("dname", "desc", "behavior", "attrib"):
        if field in raw:
            result[field] = raw[field]
    if raw.get("is_innate"):
        result["is_innate"] = True
    if "isult" in raw:
        result["is_ultimate"] = bool(raw["isult"])
    return _complete_ability(result)


def _ultimate_key(ability_keys: Any, abilities_raw: dict) -> str | None:
    """Key of a hero's ultimate, from the OpenDota ``hero_abilities`` order.

    The sixth slot holds the ultimate. When that slot is a placeholder (no
    display name, or Hidden) it is the next later named ability that is
    neither Hidden nor innate (Ogre Magi: Multicast after the hidden innate
    Dumb Luck). Heroes with fewer than six slots are left to the load-time
    heuristic (:func:`_mark_ultimates`).
    """
    slots = [(k[0] if isinstance(k, list) and k else k)
             for k in (ability_keys if isinstance(ability_keys, list) else [])]
    if len(slots) < 6:
        return None

    def usable(key: Any, allow_innate: bool) -> bool:
        raw = abilities_raw.get(key) if isinstance(key, str) else None
        if not isinstance(raw, dict) or not raw.get("dname") or _is_hidden(raw):
            return False
        return allow_innate or not raw.get("is_innate")

    if usable(slots[5], True):
        return slots[5]
    return next((key for key in slots[6:] if usable(key, False)), None)


# A basic ability sits in one of the first three slots and has four levels.
_BASIC_ABILITY_SLOTS = 3
_BASIC_ABILITY_LEVELS = 4


def _level_count(ability: dict) -> int:
    """How many levels a normalised ability's numbers come in (0: none listed per level)."""
    counts = [len(ability[field]) for field in ("cooldown", "mana_cost", "damage")
              if isinstance(ability.get(field), list)]
    attrib = ability.get("attrib")
    for entry in attrib if isinstance(attrib, list) else []:
        value = entry.get("value") if isinstance(entry, dict) else None
        if isinstance(value, list):
            counts.append(len(value))
    return max(counts, default=0)


def _flag_skillable(abilities: dict[str, dict]) -> None:
    """Set ``skillable`` on every ability of one hero (given in slot order).

    Skill points go into the ultimate and the basic abilities: the first
    three abilities that are neither innate nor Hidden (the Q / W / E
    slots) and any later one that has four levels like them (Monkey King's
    Jingu Mastery sits behind the Primal Spring sub-ability, Troll Warlord's
    Berserker's Rage behind both Whirling Axes). Everything else takes no
    points: shard / scepter abilities (one level), sub-abilities that level
    with the ultimate (three levels), placeholder slots.

    Idempotent; runs at fetch time and again on load (files written before
    the flag existed). A hero without a flagged ultimate is left alone: the
    slots cannot be told apart, and ``game_data.is_learnable`` then judges
    each ability by itself.
    """
    entries = list(abilities.values())
    if any("skillable" in a for a in entries) or not any(a.get("is_ultimate") for a in entries):
        return
    basics = 0
    for ability in entries:
        if ability.get("is_innate") or _is_hidden(ability):
            ability["skillable"] = False
        elif ability.get("is_ultimate"):
            ability["skillable"] = True
        else:
            basics += 1
            ability["skillable"] = (basics <= _BASIC_ABILITY_SLOTS
                                    or _level_count(ability) >= _BASIC_ABILITY_LEVELS)


# Item descriptions are quoted in prompts and chunks: keep them short (the
# limit is the one game_data.format_player_items shortens to).
_ITEM_DESCRIPTION_CHARS = 160
_UNRESOLVED_RE = re.compile(r"%[A-Za-z_]+%")  # "Consume %abilityhealthcost% health"


def _item_stat_line(raw: dict) -> str:
    """The stats an item lists ("+30 Agility, +35% Evasion."; "" for none):
    the attributes OpenDota gives a display text, with their value filled
    in — as many as fit into :data:`_ITEM_DESCRIPTION_CHARS`."""
    line = ""
    attrib = raw.get("attrib")
    for entry in attrib if isinstance(attrib, list) else []:
        display = entry.get("display") if isinstance(entry, dict) else None
        if not isinstance(display, str) or "{value}" not in display:
            continue
        value = entry.get("value")
        if isinstance(value, list):
            value = " / ".join(str(v) for v in value)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)) or not str(value).strip():
            continue
        stat = " ".join(display.replace("{value}", str(value).strip()).split())
        stat = re.sub(r"^([+-]) (?=\d)", r"\1", stat)  # "+ 30 Agility" -> "+30 Agility"
        if len(line) + len(stat) + 3 > _ITEM_DESCRIPTION_CHARS:
            return line + "…" if line else ""
        line = f"{line}, {stat}" if line else stat
    return line + "." if line else ""


def _item_description(raw: dict) -> str:
    """What an item does, in OpenDota's words ("" when it says nothing).

    ``desc`` when present, else the texts of the item's abilities, each
    prefixed with its kind ("Active: ... Passive: ..."); a text with an
    unresolved template value is left out. An item that has no such text
    (Butterfly, the components) is described by the stats it lists. Cut at
    a sentence end once it exceeds :data:`_ITEM_DESCRIPTION_CHARS`.
    """
    desc = raw.get("desc")
    parts = [desc] if isinstance(desc, str) and desc.strip() else []
    abilities = raw.get("abilities")
    for ability in abilities if isinstance(abilities, list) and not parts else []:
        if not isinstance(ability, dict) or not isinstance(ability.get("description"), str):
            continue
        text = " ".join(ability["description"].split())
        kind = str(ability.get("type") or "").strip().capitalize()
        if text and not _UNRESOLVED_RE.search(text):
            parts.append(f"{kind}: {text}" if kind else text)
    if not parts:
        return _item_stat_line(raw)
    text = " ".join(" ".join(parts).split())
    if len(text) <= _ITEM_DESCRIPTION_CHARS:
        return text
    head = text[:_ITEM_DESCRIPTION_CHARS]
    cut = head.rfind(". ")
    if cut >= _ITEM_DESCRIPTION_CHARS // 3:
        return head[:cut + 1]
    return head[:-1].rstrip() + "…"


def _normalize_item(raw: dict, item_key_: str) -> dict:
    """Normalize OpenDota item fields to our schema."""
    result: dict[str, Any] = {"key": item_key_}
    if "cost" in raw:
        result["cost"] = raw["cost"]
    # OpenDota writes ``false`` (and 0 == False) for "no cooldown / mana cost":
    # both are left out, so a missing field always means "none known".
    if "cd" in raw and raw["cd"] not in (False, None):
        result["cooldown"] = raw["cd"]
    if "mc" in raw and raw["mc"] not in (False, None):
        result["mana_cost"] = raw["mc"]
    components = raw.get("components")
    if isinstance(components, list) and components:
        result["components"] = [str(c) for c in components if c]
    if clean_item_name(raw.get("dname")):
        result["name"] = clean_item_name(raw["dname"])
    if "qual" in raw:
        result["quality"] = raw["qual"]
    description = _item_description(raw)
    if description:
        result["description"] = description
    if "attrib" in raw:
        result["attributes"] = raw["attrib"]
    item_abilities = raw.get("abilities")
    if isinstance(item_abilities, list) and item_abilities:
        result["is_active"] = any(
            isinstance(a, dict) and a.get("type") in ("active", "toggle") for a in item_abilities
        )
    elif "behavior" in raw:
        behavior = _behavior_text(raw.get("behavior"))
        result["is_active"] = bool(behavior) and "Passive" not in behavior
    elif "abilities" in raw:
        result["is_active"] = False  # explicitly no abilities (plain stat item)
    return result


# Hero vision ranges (world units) when no source states them: the values of
# npc_dota_hero_base. A few heroes differ (Night Stalker 800 / 1800).
DEFAULT_HERO_VISION = (game_constants.HERO_DAY_VISION, game_constants.HERO_NIGHT_VISION)

# The constants files one fetch needs; all taken from the same source.
_DATA_FILES = ("heroes", "hero_abilities", "abilities", "items")


def _vision_range(value: Any) -> int | None:
    """A plausible vision range from a hero entry (None when it states none)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value < 10000:
        return int(value)
    return None


def normalise_constants(raw: dict[str, Any]) -> dict[str, dict]:
    """The stored tables (``heroes`` / ``abilities`` / ``items`` / ``talents``)
    from the raw constants files of one source.

    Raises RuntimeError when a payload is not usable, which makes
    :func:`_download_set` give the source up.
    """
    for label in _DATA_FILES:
        if not isinstance(raw.get(label), dict) or not raw[label]:
            raise RuntimeError(f"Unexpected '{label}' payload")
    heroes_raw, hero_abilities_raw = raw["heroes"], raw["hero_abilities"]
    abilities_raw, items_raw = raw["abilities"], raw["items"]

    # 1. Heroes
    name_map: dict[str, str] = {}  # internal_name -> localized_name
    heroes_normalized: dict[str, dict] = {}
    for hero in heroes_raw.values():
        if not isinstance(hero, dict):
            continue
        internal = str(hero.get("name", "")).replace("npc_dota_hero_", "")
        localized = hero.get("localized_name") or internal
        if not internal:
            continue
        name_map[internal] = localized
        heroes_normalized[localized] = {
            "id": hero.get("id"),
            "internal_name": internal,
            "localized_name": localized,
            "primary_attr": hero.get("primary_attr", ""),
            "attack_type": str(hero.get("attack_type", "")).lower(),
            "roles": hero.get("roles", []),
            "legs": hero.get("legs", 2),
        }
        for field in ("day_vision", "night_vision"):
            vision = _vision_range(hero.get(field))
            if vision is not None:
                heroes_normalized[localized][field] = vision

    # 2. Abilities grouped per hero, in the hero's ability order
    abilities_by_hero: dict[str, dict[str, dict]] = {}
    for hero_internal, ability_info in hero_abilities_raw.items():
        if not isinstance(ability_info, dict):
            continue
        hero_internal_clean = str(hero_internal).replace("npc_dota_hero_", "")
        localized = name_map.get(hero_internal_clean, hero_internal_clean)
        hero_abilities_dict: dict[str, dict] = {}
        keys: list[str] = []
        for ability_key in ability_info.get("abilities", []):
            # OpenDota may nest sub-abilities as lists (e.g. Monkey King)
            if isinstance(ability_key, list):
                keys.extend(k for k in ability_key if isinstance(k, str))
            elif isinstance(ability_key, str):
                keys.append(ability_key)
        ultimate = _ultimate_key(ability_info.get("abilities"), abilities_raw)
        for ability_key in keys:
            raw_ability = abilities_raw.get(ability_key)
            if not isinstance(raw_ability, dict):
                continue
            display_name = raw_ability.get("dname")
            if not display_name:
                continue
            shown = hero_abilities_dict.get(display_name)
            if shown is not None:
                # Several keys share one display name (the three
                # Shadowrazes): the first slot stands for all of them
                # and the other keys stay resolvable.
                shown.setdefault("same_name_keys", []).append(ability_key)
                if ability_key == ultimate:
                    shown["is_ultimate"] = True
                continue
            normalized = _normalize_ability(raw_ability, ability_key)
            if ability_key == ultimate:
                normalized["is_ultimate"] = True
            hero_abilities_dict[display_name] = normalized
        _flag_skillable(hero_abilities_dict)
        if hero_abilities_dict:
            abilities_by_hero[localized] = hero_abilities_dict

    # 3. Items. Several keys can share one display name (Dagon and its
    # four upgrades): keep the base item — the key that spells the name,
    # else the cheapest — and record the others as upgrade_keys.
    same_name: dict[str, list[tuple[str, dict]]] = {}
    for key, item_data in items_raw.items():
        if not isinstance(item_data, dict):
            continue
        display_name = clean_item_name(item_data.get("dname"))
        if not display_name or str(key).startswith("recipe_"):
            continue
        same_name.setdefault(display_name, []).append((str(key), item_data))

    def base_rank(entry: tuple[str, dict], display_name: str) -> tuple:
        cost = entry[1].get("cost")
        cost = cost if isinstance(cost, (int, float)) and not isinstance(cost, bool) else float("inf")
        return (_squash_item(entry[0]) != _squash_item(display_name), cost, entry[0])

    items_normalized: dict[str, dict] = {}
    for display_name, entries in same_name.items():
        base_key, base_data = min(entries, key=lambda e: base_rank(e, display_name))
        normalized_item = _normalize_item(base_data, base_key)
        upgrades = sorted(key for key, _ in entries if key != base_key)
        if upgrades:
            normalized_item["upgrade_keys"] = upgrades
        items_normalized[display_name] = normalized_item

    # 4. Talent names (templates; see clean_ability_name)
    talents: dict[str, dict] = {}
    for key, raw_ability in abilities_raw.items():
        if (not str(key).startswith(TALENT_PREFIX) or not isinstance(raw_ability, dict)
                or not raw_ability.get("dname")):
            continue
        talent: dict[str, Any] = {"name": str(raw_ability["dname"])}
        values = _template_values(raw_ability.get("attrib"))
        if values:
            talent["values"] = values
        talents[str(key)] = talent

    if not heroes_normalized or not abilities_by_hero or not items_normalized:
        raise RuntimeError("Constants payload was empty after normalisation")
    return {"heroes": heroes_normalized, "abilities": abilities_by_hero,
            "items": items_normalized, "talents": talents}


# The master branch is a moving target: a half-built or refactored revision
# must not replace a complete data set. A table smaller than this is not one
# (7.41f: 127 heroes, 127 with abilities, 402 items). Module attributes, so
# tests with a three-hero data set can lower them.
MIN_TABLE_SIZES: dict[str, int] = {"heroes": 100, "abilities": 100, "items": 300}
# Nor does a table lose more than this share of what is stored (the counts in
# meta.json) from one fetch to the next: fetch_and_save_gamedata refuses
# that unless it is told to (``allow_shrink``).
MAX_TABLE_SHRINK = 0.10
_META_COUNTS = (("heroes", "hero_count"), ("abilities", "ability_hero_count"),
                ("items", "item_count"), ("talents", "talent_count"))


def _plausible_tables(raw: dict[str, Any], allow_shrink: bool = False) -> dict[str, dict]:
    """:func:`normalise_constants` of *raw*, refused (RuntimeError, so the
    next source is tried) when a table is implausibly small: below
    :data:`MIN_TABLE_SIZES`, or more than :data:`MAX_TABLE_SHRINK` below the
    stored data of the current format."""
    tables = normalise_constants(raw)
    stored = _meta() if get_schema_version() == GAMEDATA_SCHEMA_VERSION else {}
    for table, count_key in _META_COUNTS:
        size, least = len(tables[table]), int(MIN_TABLE_SIZES.get(table, 0))
        if size < least:
            raise RuntimeError(f"Implausible constants: {size} {table}, a complete set has at least {least}")
        before = stored.get(count_key)
        if (not allow_shrink and isinstance(before, int) and not isinstance(before, bool)
                and size < before * (1 - MAX_TABLE_SHRINK)):
            raise RuntimeError(f"Implausible constants: {size} {table}, the stored data has {before}")
    return tables


def _major_key(label: Any) -> tuple[int, ...]:
    """Sortable form of a major version (7.41 -> (7, 41)); () when it is not one."""
    parts = patch_major(label).split(".")
    return tuple(int(part) for part in parts) if all(part.isdigit() for part in parts) else ()


def _api_release_is_newer() -> bool:
    """Whether the OpenDota API (the npm release) is for a newer major
    version than the stored data — the one case in which it may replace data
    that came from the master branch."""
    try:
        api_major = _newest_major(_check_patch_list(_api_get("patch", timeout=15)))
    except Exception as exc:
        logger.warning("The OpenDota API did not name its version: %s", exc)
        return False
    return bool(_major_key(api_major)) and _major_key(api_major) > _major_key(get_fetched_patch())


def _download_tables(revision: Any, allow_shrink: bool) -> tuple[str, dict[str, dict]]:
    """``(source id, normalised tables)`` of one fetch.

    The sources are tried in order (:func:`_download_set`,
    :func:`_plausible_tables`), with one restriction: data that came from
    the master branch is not replaced by the OpenDota API's. That release
    is months older (Heart of Tarrasque 5100 instead of 5300 in October
    2026), so while master is unreachable the stored data stays — unless
    the API has moved on to a newer major version. Raises when nothing
    usable was delivered.
    """
    def validate(payloads: dict) -> dict[str, dict]:
        return _plausible_tables(payloads, allow_shrink)

    sources = _sources(revision)
    meta = _meta()
    stored_source = str(meta.get("source") or OPENDOTA_SOURCE)
    from_master = (bool(get_fetched_patch()) and stored_source != OPENDOTA_SOURCE
                   and get_schema_version() == GAMEDATA_SCHEMA_VERSION)
    if not from_master:
        _raw, source, tables = _download_set(_DATA_FILES, validate, sources=sources)
        return source, tables
    try:
        _raw, source, tables = _download_set(
            _DATA_FILES, validate, sources=tuple(s for s in sources if s[0] != OPENDOTA_SOURCE))
    except Exception as exc:
        if not _api_release_is_newer():
            # (what was decided first: callers show the start of the text)
            raise RuntimeError(
                f"Stored game data kept ({stored_source} {meta.get('source_revision') or ''}".rstrip()
                + "): the OpenDota API has no newer major version, and master failed with "
                  f"{type(exc).__name__}: {exc}") from exc
        _raw, source, tables = _download_set(
            _DATA_FILES, validate, sources=tuple(s for s in sources if s[0] == OPENDOTA_SOURCE))
    return source, tables


def _same_on_disk(path: Path, data: Any) -> bool:
    """Whether *path* already holds exactly *data* (compared as JSON values)."""
    if not path.exists():
        return False
    return _load_json(path) == json.loads(json.dumps(data, ensure_ascii=False))


def _note_meta_rewritten() -> None:
    """meta.json was replaced while the data behind it stayed the same:
    forget what is cached from it and keep every other cache (and the
    generation, so no dependent cache is rebuilt)."""
    global _disk_signature
    with _cache_lock:
        for name in ("meta", "fetched_patch", "schema_version", "talent_patch"):
            _cache.pop(name, None)
        _disk_signature = _meta_signature()


def fetch_and_save_gamedata(force: bool = False, allow_shrink: bool = False) -> dict:
    """Fetch all game data and save it to disk.

    Heroes, abilities, items and talent names from the first of
    :data:`CONSTANTS_SOURCES` that delivers a complete, valid set, plus the
    patch tables from the patch check (also refreshed when the data itself
    is up to date). Without *force* nothing is downloaded when the check
    finds the local data current (see :func:`_patch_status`).

    Everything is downloaded and validated before any file is written, and
    each file is replaced atomically, so a failed fetch never leaves a mix
    of old and new data behind. Stored data is not replaced by worse data
    either: a set that lost a large part of a table is refused (a broken
    upstream build; *allow_shrink* accepts it after a look at the cause),
    and data from the master branch is not replaced by the older release of
    the OpenDota API (see :func:`_download_tables`). ``meta.json`` records
    the source, its revision and date, the major version (``patch_version``)
    and the full version the data can reflect (``data_label``). All caches
    are invalidated afterwards — unless the fetched data equals what was
    stored, which only renews ``meta.json``. Returns a summary dict
    (``status``: ``updated`` or ``up_to_date``); raises on network / payload
    errors and when only worse data was offered.
    """
    with _fetch_lock:
        state = _remote_state()
        releases, revision = state["releases"], state["revision"]
        try:
            remote_ver, reason = _patch_status(state)
        except Exception as exc:
            logger.warning("Patch check failed, proceeding with a full update: %s", exc)
            reason = "patch_check_failed"
        else:
            if not force and not reason:
                _save_patch_table(state["patches"], releases)
                # the local major version; the label is the game's (it may be ahead of the data)
                return {"status": "up_to_date", "patch_version": get_current_patch(),
                        "patch_label": remote_ver}

        source, data = _download_tables(revision, allow_shrink)

        # The major version is the source's own (its patch list), else that
        # of the newest version the data can reflect, else Valve's current
        # one; the previous label is kept when the network named none.
        identity = _data_identity(source, releases, revision)
        data_label = identity.pop("data_label")
        patch = (_newest_major(state["patches"]) or patch_major(data_label)
                 or patch_major(releases[-1][1] if releases else "") or get_fetched_patch())
        if patch_major(data_label) != patch:
            data_label = ""
        meta = {
            "schema_version": GAMEDATA_SCHEMA_VERSION,
            "patch_version": patch,
            "data_label": data_label,
            "last_fetched": _utc_stamp(),
            **identity,
            "hero_count": len(data["heroes"]),
            "ability_hero_count": len(data["abilities"]),
            "item_count": len(data["items"]),
            "talent_count": len(data["talents"]),
        }
        files = (("heroes.json", data["heroes"]), ("abilities.json", data["abilities"]),
                 ("items.json", data["items"]), (TALENTS_FILE, data["talents"]))
        changed = [name for name, payload in files if not _same_on_disk(GAMEDATA_DIR / name, payload)]
        same_data = (not changed and patch == get_fetched_patch()
                     and get_schema_version() == GAMEDATA_SCHEMA_VERSION)
        for name, payload in files:
            if name in changed:
                _save_json(GAMEDATA_DIR / name, payload)
        _save_patch_table(state["patches"], releases)  # keeps the previous tables when the check failed
        _save_json(GAMEDATA_DIR / "meta.json", meta)  # last: marks the set complete

    if same_data:
        _note_meta_rewritten()
        logger.info("Fetched game data equals the stored data (%s); caches kept", source)
    else:
        # Data on disk changed: every derived cache (incl. RAG stores) is stale.
        invalidate_all_caches()
    return {
        "status": "updated",
        "changed": not same_data,
        "reason": reason or "forced",
        "patch_version": patch or DEFAULT_PATCH,
        "patch_label": get_patch_label(),
        "data_label": data_label,
        "source": source,
        "source_revision": meta["source_revision"],
        "heroes": len(data["heroes"]),
        "abilities": len(data["abilities"]),
        "items": len(data["items"]),
        "talents": len(data["talents"]),
    }


def refresh_gamedata_if_outdated() -> dict:
    """Bring the game data up to date when it is outdated. Never raises.

    Meant for a scheduler (the server runs it at start and once a day).
    Three small requests — Valve's patch feed, the source's patch list and
    its revision — tell whether anything changed; the constants themselves
    (about 3 MB) are only downloaded when they did (see
    :func:`_patch_status`), when there is no local data, or when it is older
    than ``GAMEDATA_STALE_DAYS``. Returns the summary of
    :func:`fetch_and_save_gamedata`, or ``{"status": "failed", "error":
    <text>, "patch_version": <local major version>}`` when the network or the
    payload failed, or only worse data than the stored set was offered (the
    local data is untouched then).
    """
    try:
        meta = get_meta()
        stale = meta.get("status") == "no_data" or bool(meta.get("is_stale"))
        return fetch_and_save_gamedata(force=stale)
    except Exception as exc:
        logger.warning("Game data refresh failed: %s", exc)
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:200],
                "patch_version": get_current_patch()}


# ---------------------------------------------------------------------------
# Load and merge (Layer A + fallback + Layer B -> memory)
# ---------------------------------------------------------------------------

def _infer_role(roles: list[str]) -> str:
    """Map OpenDota role labels to our role system (rough, editorial wins)."""
    roles_lower = [str(r).lower() for r in roles or []]
    if not roles_lower:
        return "unknown"
    if "carry" in roles_lower:
        return "carry"
    if "support" in roles_lower:
        return "support_5"
    if "initiator" in roles_lower or "durable" in roles_lower:
        return "offlane"
    if "nuker" in roles_lower:
        return "mid"
    return "unknown"


def _build_heroes() -> dict[str, dict]:
    editorial = _load_dict(EDITORIAL_DIR / "hero_knowledge.json")
    by_display: dict[str, dict] = {}
    for entry in _hero_index()["by_internal"].values():
        by_display[entry["name"]] = {
            "role": _infer_role(entry.get("roles", [])),
            "attribute": entry.get("attr") or "unknown",
            "attack_type": entry.get("attack_type") or "unknown",
            "power_spikes": [],
            "weaknesses": [],
            "laning_tips": "",
            "internal_name": entry["internal"],
            "id": entry.get("id"),
            "roles": list(entry.get("roles", [])),
        }
    for name, info in editorial.items():
        if not isinstance(info, dict):
            continue
        display = resolve_hero_name(name) or name
        hero = by_display.setdefault(display, {
            "role": "unknown", "attribute": "unknown", "attack_type": "unknown",
            "power_spikes": [], "weaknesses": [], "laning_tips": "",
            "internal_name": hero_slug(name), "id": None, "roles": [],
        })
        for field in ("role", "laning_tips", "level1_strength", "level1_note"):
            if info.get(field):
                hero[field] = info[field]
        # Per-patch facts: the catalog / fetched value wins, editorial only
        # fills a gap (it is written once and goes stale).
        for field in ("attribute", "attack_type"):
            if info.get(field) and hero.get(field) in (None, "", "unknown"):
                hero[field] = info[field]
        for field in ("power_spikes", "weaknesses"):
            if isinstance(info.get(field), list):
                hero[field] = [str(x) for x in info[field] if x]
        hero["has_editorial"] = True
    logger.info("Loaded %d heroes (%d with editorial knowledge)",
                len(by_display), sum(1 for h in by_display.values() if h.get("has_editorial")))
    return by_display


def load_heroes() -> dict[str, dict]:
    """Hero data keyed by display name (catalog + fetched data + editorial).

    Fields: role, attribute, attack_type, power_spikes, weaknesses,
    laning_tips, level1_strength, level1_note, internal_name, id, roles.
    Always contains editorial knowledge for every editorial hero, whether or
    not game data has been fetched.
    """
    return _cached("heroes", _build_heroes)


def _mark_ultimates(abilities: dict[str, dict], editorial_order: bool = False,
                    editorial_last: str | None = None) -> None:
    """Flag the hero's ultimate when no source flagged it explicitly.

    Used for data without fetch-time flags (files written before schema 2,
    short ability lists, the no-gamedata fallback). In order: the last
    learnable ability with 3 levels of cooldown / mana cost; else
    *editorial_last*, the last ability the editorial data lists for the hero
    (covers passive / charge-based ultimates); else, for editorial-only
    heroes (no numbers at all), the last listed ability.
    """
    if not abilities or any(a.get("is_ultimate") for a in abilities.values()):
        return
    candidates = [
        name for name, a in abilities.items()
        if not a.get("is_innate") and not _is_hidden(a)
        and (len(a.get("cooldown") or []) == 3 or len(a.get("mana_cost") or []) == 3)
    ]
    if candidates:
        abilities[candidates[-1]]["is_ultimate"] = True
    elif editorial_last in abilities:
        abilities[editorial_last]["is_ultimate"] = True
    elif editorial_order:
        abilities[list(abilities)[-1]]["is_ultimate"] = True


# Fallback fields that describe what an ability is rather than its numbers:
# the only ones merged onto an ability that fetched data already provides.
_FALLBACK_STRUCTURAL_FIELDS = ("type",)


def _build_abilities() -> dict[str, dict[str, dict]]:
    gamedata = _load_dict(GAMEDATA_DIR / "abilities.json")
    fallback = _load_dict(FALLBACK_DIR / "ability_stats.json", "abilities")
    editorial = _load_dict(EDITORIAL_DIR / "coaching_tips_abilities.json")
    legacy = get_schema_version() < _SANITISE_BELOW_SCHEMA

    merged: dict[str, dict[str, dict]] = {}
    heroes_with_stats: set[str] = set()
    for hero_name, abilities in gamedata.items():
        if not isinstance(abilities, dict):
            continue
        display = resolve_hero_name(hero_name) or hero_name
        merged[display] = {
            # Files written before schema 2 store placeholder slots under
            # their key ("generic_hidden") and lack the derived fields.
            name: _complete_ability(dict(data)) for name, data in abilities.items()
            if isinstance(data, dict) and name != data.get("key")
        }
        if not legacy:
            _flag_skillable(merged[display])
        heroes_with_stats.add(display)

    def find(display: str, key: str, ability_name: str) -> tuple[str, dict] | None:
        abilities = merged.get(display) or {}
        for name, data in abilities.items():
            if key in ability_keys(data):
                return name, data
        squashed = _NON_ALNUM.sub("", str(ability_name).lower())
        for name, data in abilities.items():
            if _NON_ALNUM.sub("", name.lower()) == squashed:
                return name, data
        return None

    # Numeric fallback (hand-written, goes stale): the full entry only stands
    # in for heroes without fetched stats. An ability that fetched data
    # provides takes nothing numeric from it — fetched numbers always win.
    fallback_heroes: set[str] = set()
    fallback_ultimates: dict[str, str] = {}
    for key, info in fallback.items():
        if not isinstance(info, dict) or not info.get("ability_name"):
            continue
        display = resolve_hero_name(info.get("hero")) or str(info.get("hero"))
        found = find(display, key, info["ability_name"])
        if found is not None:
            name, existing = found
            for field in _FALLBACK_STRUCTURAL_FIELDS:
                if field in info:
                    existing.setdefault(field, info[field])
            if info.get("is_ultimate"):
                fallback_ultimates[display] = name
        elif display not in heroes_with_stats:
            stats = {k: v for k, v in info.items() if k not in ("hero", "ability_name")}
            merged.setdefault(display, {})[info["ability_name"]] = {"key": key, **stats}
            fallback_heroes.add(display)
    for display, name in fallback_ultimates.items():
        abilities = merged[display]
        if not any(a.get("is_ultimate") for a in abilities.values()):
            abilities[name]["is_ultimate"] = True

    # Editorial coaching text (canonical) — always merged.
    editorial_only: set[str] = set()
    editorial_last: dict[str, str] = {}
    skipped: list[str] = []
    for key, info in editorial.items():
        if not isinstance(info, dict):
            continue
        ability_name = info.get("ability_name") or key
        display = resolve_hero_name(info.get("hero")) or str(info.get("hero") or "")
        if not display:
            continue
        found = find(display, key, ability_name)
        if found is None:
            if display in heroes_with_stats:
                skipped.append(key)
                continue
            name = ability_name
            existing = merged.setdefault(display, {}).setdefault(ability_name, {"key": key})
            if display not in fallback_heroes:
                editorial_only.add(display)
        else:
            name, existing = found
            if legacy and key not in ability_keys(existing):
                # Old files kept the last of several same-named keys
                # (Shadowraze): the editorial key is one of the others.
                existing.setdefault("same_name_keys", []).append(key)
        editorial_last[display] = name
        if info.get("coaching_tip"):
            existing["coaching_tip"] = info["coaching_tip"]
        if info.get("description"):
            existing["description"] = info["description"]
        if info.get("is_ultimate") is True and not any(
                a.get("is_ultimate") for a in merged[display].values()):
            existing["is_ultimate"] = True
        # What an ability is (innate / Hidden) is a per-patch fact: the
        # editorial flags only stand in while fetched data cannot say.
        if display not in heroes_with_stats or legacy:
            for field in ("is_innate", "behavior"):
                if info.get(field):
                    existing.setdefault(field, info[field])
    if skipped:
        logger.warning(
            "%d editorial ability tip(s) match no ability in the fetched data and were skipped: %s",
            len(skipped), ", ".join(sorted(skipped)[:10]) + (" ..." if len(skipped) > 10 else ""))

    for display, abilities in merged.items():
        _mark_ultimates(abilities, editorial_order=display in editorial_only,
                        editorial_last=editorial_last.get(display))
    logger.info("Loaded abilities for %d heroes (%d with fetched stats)",
                len(merged), len(heroes_with_stats))
    return merged


def load_abilities() -> dict[str, dict[str, dict]]:
    """Abilities keyed by hero display name -> ability display name -> data.

    Data merges fetched stats with the editorial ``description`` /
    ``coaching_tip``; editorial text is merged even when no game data has
    been fetched. The numeric fallback only stands in for
    heroes without fetched stats — fetched numbers are never mixed with it.
    Exactly one ability per hero carries ``is_ultimate``; ``is_innate`` and
    a ``behavior`` containing "Hidden" mark abilities nobody skills, and
    fetched heroes also carry ``skillable`` on every ability (False for
    shard / scepter abilities and sub-abilities, see :func:`_flag_skillable`).
    ``damage`` is what one cast deals; damage over time is
    ``damage_per_second``. One entry per display name: :func:`ability_keys`
    lists every key it covers.
    """
    return _cached("abilities", _build_abilities)


def _sanitize_fetched_item(name: str, data: dict, legacy: bool) -> dict | None:
    """Copy of a fetched item entry in the current shape (None: not an item).

    ``false`` cooldown / mana cost mean "none" and are dropped. *legacy*
    files (no schema_version) additionally contain recipes, an ``is_active``
    flag that was computed wrongly (always false) and, for display names
    shared by several keys (Dagon), the entry of the last upgrade: that one
    is re-keyed to the base item and loses its upgrade-level numbers and text.
    """
    entry = dict(data)
    entry["name"] = clean_item_name(entry.get("name")) or name
    key = str(entry.get("key") or "")
    for field in ("cooldown", "mana_cost"):
        if entry.get(field) in (False, None) and field in entry:
            del entry[field]
    if not legacy:
        return entry
    if key.startswith("recipe_"):
        return None
    if entry.get("is_active") is False:
        del entry["is_active"]
    base_key = resolve_item_key(name)
    if key and base_key and base_key != key and _squash_item(base_key) == _squash_item(name):
        for field in ("cost", "cooldown", "mana_cost", "attributes", "description"):
            entry.pop(field, None)
        entry["key"] = base_key
    return entry


# Fallback item fields merged onto an item that fetched data already provides,
# where it lacks them: the category (nothing upstream states one) and the
# description (files of an older format have none for some items).
_FALLBACK_STRUCTURAL_ITEM_FIELDS = ("category", "description")


def _build_items() -> dict[str, dict]:
    gamedata = _load_dict(GAMEDATA_DIR / "items.json")
    fallback = _load_dict(FALLBACK_DIR / "item_stats.json", "items")
    editorial = _load_dict(EDITORIAL_DIR / "coaching_tips_items.json")
    legacy = get_schema_version() < _SANITISE_BELOW_SCHEMA

    merged: dict[str, dict] = {}
    by_key: dict[str, dict] = {}
    fetched_keys: set[str] = set()
    for name, data in gamedata.items():
        if not isinstance(data, dict):
            continue
        name = clean_item_name(name) or name
        entry = _sanitize_fetched_item(name, data, legacy)
        if entry is None:
            continue
        merged[name] = entry
        if entry.get("key"):
            by_key[str(entry["key"])] = entry
            fetched_keys.add(str(entry["key"]))

    def ensure(key: str, fallback_name: str) -> dict:
        entry = by_key.get(key)
        if entry is None:
            name = resolve_item_name(key) or fallback_name or key
            entry = merged.get(name)
            if entry is None:
                entry = {"key": key, "name": name}
                merged[name] = entry
            by_key[key] = entry
        return entry

    # Numeric fallback (hand-written, goes stale): complete for items without
    # fetched stats; a fetched item only takes what fetched data cannot say.
    fallback_flags: dict[str, bool] = {}
    for key, info in fallback.items():
        if not isinstance(info, dict):
            continue
        entry = ensure(key, str(info.get("item_name") or ""))
        fetched = str(entry.get("key")) in fetched_keys
        for field, value in info.items():
            if field == "item_name":
                continue
            if field == "is_active":
                fallback_flags[key] = bool(value)
            elif not fetched or field in _FALLBACK_STRUCTURAL_ITEM_FIELDS:
                entry.setdefault(field, value)

    for key, info in editorial.items():
        if not isinstance(info, dict):
            continue
        entry = ensure(key, str(info.get("item_name") or ""))
        for field in ("coaching_tip", "heroes", "counter_tips"):
            if info.get(field):
                entry[field] = info[field]
        # What an item does (with its numbers) and whether it has an active
        # are per-patch facts: the fetched text / flag wins, the editorial
        # one only fills a gap.
        if info.get("description"):
            entry.setdefault("description", info["description"])
        if isinstance(info.get("is_active"), bool):
            entry.setdefault("is_active", info["is_active"])
    for key, flag in fallback_flags.items():
        by_key[key].setdefault("is_active", flag)
    logger.info("Loaded %d items (%d with fetched stats)", len(merged), len(fetched_keys))
    return merged


def load_items() -> dict[str, dict]:
    """Items keyed by display name: fetched stats (or fallback) + editorial tips.

    One entry per display name, keyed by the base item (``upgrade_keys``
    lists e.g. dagon_2..5). A missing ``cooldown`` means none is known.
    ``is_active`` comes from fetched data when it says so, else from the
    editorial / fallback flag; ``description`` likewise is the fetched
    ability text when there is one, else the editorial description.
    """
    return _cached("items", _build_items)


# Rule values the game states on items, read from the fetched item
# attributes: (rule name, item key, attribute key, built-in value). The last
# column stands in until game data is fetched; it is the value of
# ``game_constants.RULES_PATCH``. Durations are seconds, ranges world units.
_ITEM_RULE_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("observer_ward_duration", "ward_observer", "lifetime"),
    ("observer_ward_vision", "ward_observer", "vision_range_tooltip"),
    ("sentry_ward_duration", "ward_sentry", "lifetime"),
    ("sentry_ward_true_sight", "ward_sentry", "true_sight_range"),
    ("smoke_duration", "smoke_of_deceit", "duration"),
    ("smoke_application_radius", "smoke_of_deceit", "application_radius"),
    ("smoke_break_radius", "smoke_of_deceit", "visibility_radius"),
    ("gem_true_sight", "gem", "radius"),
    ("dust_radius", "dust", "radius"),
    ("dust_duration", "dust", "duration"),
)
_ITEM_RULES: tuple[tuple[str, str, str, int], ...] = tuple(
    (name, item, attribute, game_constants.ITEM_RULE_DEFAULTS[name])
    for name, item, attribute in _ITEM_RULE_SOURCES
)


def _build_game_rules() -> dict[str, int | float]:
    rules: dict[str, int | float] = {name: default for name, _item, _attr, default in _ITEM_RULES}
    try:
        by_key = {str(data.get("key")): data for data in load_items().values()}
        for name, item, attribute, _default in _ITEM_RULES:
            for entry in by_key.get(item, {}).get("attributes") or []:
                if not isinstance(entry, dict) or entry.get("key") != attribute:
                    continue
                values = _to_num_list(entry.get("value"))
                if len(values) == 1 and values[0] > 0:
                    rules[name] = int(values[0]) if float(values[0]).is_integer() else values[0]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Game rules fall back to the built-in values: %s", exc)
    return rules


def game_rules() -> dict[str, int | float]:
    """Ward, smoke and true-sight values of the current game data. Never raises.

    Always has every name of :data:`_ITEM_RULES` (``observer_ward_duration``,
    ``observer_ward_vision``, ``sentry_ward_duration``,
    ``sentry_ward_true_sight``, ``smoke_duration``,
    ``smoke_application_radius``, ``smoke_break_radius``, ``gem_true_sight``,
    ``dust_radius``, ``dust_duration``): the fetched item attribute, else
    the built-in value. Seconds and world units. Shared and read-only.
    """
    return _cached("game_rules", _build_game_rules)


def _build_matchups() -> dict[str, dict[str, str]]:
    editorial = _load_dict(EDITORIAL_DIR / "matchup_tips.json")
    result: dict[str, dict[str, str]] = {}
    for key, value in editorial.items():
        if ":" not in key:
            continue
        hero_a, hero_b = key.split(":", 1)
        tip = value.get("tip", "") if isinstance(value, dict) else value
        if not isinstance(tip, str) or not tip.strip():
            continue
        a = resolve_hero_name(hero_a) or hero_a.strip()
        b = resolve_hero_name(hero_b) or hero_b.strip()
        result.setdefault(a, {})[b] = tip.strip()
    return result


def load_matchup_tips() -> dict[str, dict[str, str]]:
    """Editorial matchup tips: ``{hero_display: {opponent_display: tip}}``."""
    return _cached("matchups", _build_matchups)


def get_meta() -> dict:
    """Return metadata about the current game data state (a fresh dict).

    ``patch_version`` is the major version of the data, ``patch_label`` /
    ``patch_released`` the game's current full version and its release time,
    ``source`` where the files came from (an id of
    :data:`CONSTANTS_SOURCES`; ``fallback`` without fetched data) with
    ``source_revision`` / ``source_date`` when the source has them, and
    ``data_label`` / ``data_current`` say how the data relates to the
    current version; ``checked_at`` is when the game version was last
    checked against Valve's feed, "" when it never was (see
    :func:`get_patch_info`).
    """
    info = get_patch_info()
    labels = {"patch_label": info["label"], "patch_released": info["released"],
              "data_label": info["data_label"], "data_current": info["data_current"],
              "checked_at": info["checked_at"]}
    meta = _load_json(GAMEDATA_DIR / "meta.json")
    if not isinstance(meta, dict):
        return {
            "status": "no_data",
            "patch_version": DEFAULT_PATCH,
            "source": "fallback",
            **labels,
        }
    meta = dict(meta)
    meta["patch_version"] = patch_major(_valid_patch(meta.get("patch_version"))) or DEFAULT_PATCH
    # Files written before the source was recorded all came from the OpenDota API.
    meta["source"] = str(meta.get("source") or OPENDOTA_SOURCE)
    meta.update(labels)
    meta["outdated_format"] = get_schema_version() < GAMEDATA_SCHEMA_VERSION
    last_fetched = meta.get("last_fetched", "")
    if last_fetched:
        try:
            fetched_ts = calendar.timegm(time.strptime(last_fetched, "%Y-%m-%dT%H:%M:%SZ"))
            age_days = (time.time() - fetched_ts) / 86400
            meta["age_days"] = round(max(0.0, age_days), 1)
            # Same threshold as the startup updater (main._check_and_update_gamedata).
            meta["is_stale"] = age_days > config.GAMEDATA_STALE_DAYS
        except (TypeError, ValueError, OverflowError):
            meta["is_stale"] = True
    return meta


__all__ = [
    "DEFAULT_PATCH",
    "DEFAULT_PATCH_LABEL",
    "DEFAULT_PATCH_RELEASED",
    "CONSTANTS_SOURCES",
    "patch_major",
    "get_current_patch",
    "get_fetched_patch",
    "get_talent_patch",
    "get_patch_version",
    "get_patch_info",
    "get_patch_label",
    "check_patch_version",
    "patch_for_time",
    "patch_label_for_time",
    "download_constants",
    "download_constants_set",
    "download_constants_snapshot",
    "normalise_constants",
    "clean_item_name",
    "raw_ability_name",
    "fetch_and_save_gamedata",
    "refresh_gamedata_if_outdated",
    "hero_vision",
    "DEFAULT_HERO_VISION",
    "game_rules",
    "BUILTIN_ABILITY_LABELS",
    "load_heroes",
    "load_abilities",
    "load_items",
    "load_matchup_tips",
    "get_meta",
    "get_schema_version",
    "GAMEDATA_SCHEMA_VERSION",
    "get_hero_catalog",
    "resolve_hero_name",
    "hero_slug",
    "is_known_hero",
    "is_known_item",
    "resolve_item_key",
    "resolve_item_name",
    "item_key",
    "resolve_ability_name",
    "clean_ability_name",
    "ability_keys",
    "hero_display_name",
    "item_display_name",
    "ability_display_name",
    "unit_display_name",
    "display_name",
    "ATTACK_INFLICTOR",
    "display_slug",
    "data_generation",
    "invalidate_all_caches",
    "clear_caches",
]
