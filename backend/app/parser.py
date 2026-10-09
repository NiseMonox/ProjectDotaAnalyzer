"""Replay parser abstraction layer.

Parser backends:
- ClarityParser: runs the clarity Java analyzer (tools/clarity-analyzer.jar) on a .dem file
- OpenDotaParser: fetches a match from the OpenDota API by match ID
- MockParser: returns deterministic mock data for development/testing

Every position in the returned :class:`MatchData` is in Dota 2 world units
(see ``geo.py``). Enumerations are stable lowercase codes, never localized labels.

Errors (all ``RuntimeError`` subclasses; ``str(exc)`` is short and user-safe):
- :class:`ParserUnavailableError` - Java runtime or analyzer JAR missing
- :class:`ReplayParseError` - the replay could not be parsed (details are logged)
- :class:`OpenDotaError` - an OpenDota request failed (``.status`` = HTTP status or None)

Environment:
- ``CLARITY_MAX_CONCURRENCY`` (default 1): parallel Java parses per process
- ``CLARITY_JVM_XMX`` (default 1536m): JVM heap limit
- ``CLARITY_TIMEOUT_S`` (default 300): parse timeout; the JVM is killed when exceeded
- ``OPENDOTA_API_KEY`` (optional): appended to OpenDota API requests
"""
from __future__ import annotations

import asyncio
import http.client
import json
import logging
import math
import os
import re
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any

from . import geo
from .config import CLARITY_JAR_PATH, JAVA_PATH
from .models import (
    AbilityCast,
    AbilityEvent,
    GoldXpSnapshot,
    ItemEvent,
    LaningCreepKill,
    LaningDamageEvent,
    LaningRegenEvent,
    LaningSnapshot,
    MatchData,
    PlayerData,
    PositionSnapshot,
    SkillEvent,
    SmokeEvent,
    TeamfightEvent,
    TeamfightPlayer,
    WardEvent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ParserUnavailableError(RuntimeError):
    """The requested parser cannot run here (Java runtime or analyzer JAR missing)."""


class ReplayParseError(RuntimeError):
    """A replay could not be parsed. ``str(exc)`` is short and safe to show to users."""


class OpenDotaError(RuntimeError):
    """An OpenDota API request failed.

    ``status`` is the HTTP status code (e.g. 404 for an unknown match), or None
    for network errors / invalid responses.
    """

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


# ---------------------------------------------------------------------------
# Static game data (frontend/static/data/*.json), cached by mtime
# ---------------------------------------------------------------------------

_STATIC_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "static" / "data"
HEROES_JSON_PATH = _STATIC_DATA_DIR / "heroes.json"
ITEMS_JSON_PATH = _STATIC_DATA_DIR / "items.json"
ABILITIES_JSON_PATH = _STATIC_DATA_DIR / "abilities.json"

# path -> (mtime, data, last mtime check); the mtime is re-checked at most every few seconds
_json_cache: dict[Path, tuple[float, dict, float]] = {}
_json_cache_lock = threading.Lock()
_JSON_RECHECK_S = 5.0


def _load_static_json(path: Path) -> dict:
    """Load a static JSON file (re-read only when its mtime changes)."""
    now = time.monotonic()
    with _json_cache_lock:
        cached = _json_cache.get(path)
        if cached is not None and now - cached[2] < _JSON_RECHECK_S:
            return cached[1]
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if cached is not None and cached[0] == mtime:
        with _json_cache_lock:
            _json_cache[path] = (mtime, cached[1], now)
        return cached[1]
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("Failed to load static data %s: %s", path.name, exc)
        return {}
    if not isinstance(data, dict):
        data = {}
    with _json_cache_lock:
        _json_cache[path] = (mtime, data, now)
    return data


def load_hero_map() -> dict[int, dict]:
    """Load hero id -> {name, slug} map from static data."""
    result: dict[int, dict] = {}
    for hid, info in (_load_static_json(HEROES_JSON_PATH).get("by_id") or {}).items():
        if not isinstance(info, dict):
            continue
        try:
            result[int(hid)] = {"name": info.get("name", ""), "slug": info.get("slug", "")}
        except (TypeError, ValueError):
            continue
    return result


def normalize_hero_key(name: str | None) -> str:
    """Normalize a hero identifier for matching.

    Accepts display names ("Queen of Pain"), heroes.json slugs ("queenofpain"),
    npc names ("npc_dota_hero_queenofpain") and entity-class conversions
    ("queen_of_pain"); all map to "queenofpain".
    """
    n = (name or "").strip().lower()
    if n.startswith("npc_dota_hero_"):
        n = n[len("npc_dota_hero_"):]
    return re.sub(r"[^a-z0-9]", "", n)


def _hero_lookup() -> dict[str, dict]:
    """normalized slug / display name -> {id, name, slug}."""
    lookup: dict[str, dict] = {}
    for hid, info in load_hero_map().items():
        entry = {"id": hid, "name": info.get("name") or info.get("slug") or "", "slug": info.get("slug") or ""}
        for key in (info.get("slug"), info.get("name")):
            norm = normalize_hero_key(key)
            if norm:
                lookup.setdefault(norm, entry)
    return lookup


def _items_by_key() -> dict[str, dict]:
    data = _load_static_json(ITEMS_JSON_PATH).get("by_key") or {}
    return data if isinstance(data, dict) else {}


def _item_id_map() -> dict[int, tuple[str, str]]:
    """item id -> (display name, key)."""
    result: dict[int, tuple[str, str]] = {}
    for key, info in _items_by_key().items():
        if isinstance(info, dict) and isinstance(info.get("id"), int):
            result[info["id"]] = (info.get("name") or key, info.get("key") or key)
    return result


def item_display_name(key: str) -> str:
    """Display name for an item key (falls back to a title-cased key)."""
    info = _items_by_key().get(key)
    if isinstance(info, dict) and info.get("name"):
        return info["name"]
    return key.replace("_", " ").title()


def ability_display_name(key: str) -> str:
    """Display name for an ability key (falls back to the key itself)."""
    abilities = _load_static_json(ABILITIES_JSON_PATH).get("by_key") or {}
    info = abilities.get(key) if isinstance(abilities, dict) else None
    if isinstance(info, dict) and info.get("name"):
        return info["name"]
    return key


# ---------------------------------------------------------------------------
# Small value helpers
# ---------------------------------------------------------------------------

def _as_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return int(value) if isinstance(value, bool) else default
    if isinstance(value, int):
        return value
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return int(f)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _dicts(value: Any) -> list[dict]:
    """Only the dict entries of a JSON list (tolerates None / malformed data)."""
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, dict)]


def _int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {str(k): _as_int(v) for k, v in value.items()}


def _team_for_slot(slot: int) -> str:
    return "radiant" if slot < 128 else "dire"


# ---------------------------------------------------------------------------
# Game modes (codes, translated by the frontend)
# ---------------------------------------------------------------------------

# Valve game_mode id -> stable code
GAME_MODES: dict[int, str] = {
    0: "unknown", 1: "all_pick", 2: "captains_mode", 3: "random_draft",
    4: "single_draft", 5: "all_random", 6: "intro", 7: "diretide",
    8: "reverse_captains_mode", 9: "greeviling", 10: "tutorial", 11: "mid_only",
    12: "least_played", 13: "limited_heroes", 14: "compendium_matchmaking",
    15: "custom", 16: "captains_draft", 17: "balanced_draft", 18: "ability_draft",
    19: "event", 20: "all_random_death_match", 21: "one_v_one_mid",
    22: "ranked_all_pick", 23: "turbo", 24: "mutation", 25: "coaches_challenge",
}


def game_mode_code(mode: Any) -> str:
    """Map a game_mode id (or an existing code string) to its code."""
    if isinstance(mode, str) and not mode.strip().lstrip("-").isdigit():
        return mode.strip() or "unknown"
    mode_id = _as_int(mode)
    return GAME_MODES.get(mode_id, f"mode_{mode_id}")


# ---------------------------------------------------------------------------
# Buildings: canonical names and OpenDota-layout status bitmasks
# ---------------------------------------------------------------------------

_BUILDING_NAME_RE = re.compile(
    r"(goodguys|badguys)_(?:tower([1-4])(?:_(top|mid|bot))?|(melee|range)_rax_(top|mid|bot)|(fort))"
)

TOWER_BIT_ORDER = [
    "top_t1", "top_t2", "top_t3",
    "mid_t1", "mid_t2", "mid_t3",
    "bot_t1", "bot_t2", "bot_t3",
    "ancient_top", "ancient_bot",
]
BARRACKS_BIT_ORDER = ["top_melee", "top_range", "mid_melee", "mid_range", "bot_melee", "bot_range"]
_ALL_TOWERS_ALIVE = (1 << len(TOWER_BIT_ORDER)) - 1
_ALL_BARRACKS_ALIVE = (1 << len(BARRACKS_BIT_ORDER)) - 1


def parse_building_name(name: str | None) -> dict | None:
    """Parse a building unit/entity name.

    Returns {"team", "type", "key"} where key is a TOWER_BIT_ORDER /
    BARRACKS_BIT_ORDER entry ("top_t1", "mid_melee"), "fort", or
    "ancient" for a tier-4 tower whose side (top/bot) is not in the name.
    """
    m = _BUILDING_NAME_RE.search((name or "").lower())
    if not m:
        return None
    team = "radiant" if m.group(1) == "goodguys" else "dire"
    if m.group(2):
        tier = int(m.group(2))
        lane = m.group(3)
        if tier == 4:
            key = f"ancient_{lane}" if lane in ("top", "bot") else "ancient"
        elif lane:
            key = f"{lane}_t{tier}"
        else:
            return None
        return {"team": team, "type": "tower", "key": key}
    if m.group(4):
        return {"team": team, "type": "barracks", "key": f"{m.group(5)}_{m.group(4)}"}
    return {"team": team, "type": "fort", "key": "fort"}


def _canonical_building_name(team: str, building_type: str, key: str) -> str:
    side = "goodguys" if team == "radiant" else "badguys"
    if building_type == "fort":
        return f"npc_dota_{side}_fort"
    if building_type == "barracks":
        lane, kind = key.split("_", 1)
        return f"npc_dota_{side}_{kind}_rax_{lane}"
    if key.startswith("ancient"):
        lane = key.split("_", 1)[1] if "_" in key else ""
        return f"npc_dota_{side}_tower4" + (f"_{lane}" if lane else "")
    lane, tier = key.split("_", 1)
    return f"npc_dota_{side}_tower{tier[1:]}_{lane}"


def _nearest_tower_key(team: str, x: float, y: float, max_dist: float = 1500.0) -> str | None:
    best_key, best_d = None, max_dist
    for key, (tx, ty) in geo.TOWER_POSITIONS.get(team, {}).items():
        d = geo.distance(x, y, tx, ty)
        if d <= best_d:
            best_key, best_d = key, d
    return best_key


def building_status_bitmasks(buildings: list[dict], objectives: list[dict] | None = None) -> dict[str, int]:
    """Compute end-of-game tower/barracks status (set bit = alive).

    Uses ``buildings`` (destroyed_time) and building-kill ``objectives``; a
    building without any destruction evidence counts as alive.
    """
    destroyed: dict[str, set[str]] = {"radiant": set(), "dire": set()}
    # Tier-4 kills whose side (top/bot) is unknown, counted per team.
    sideless_t4: Counter = Counter()

    def _mark(parsed: dict | None) -> None:
        if not parsed or parsed["type"] not in ("tower", "barracks"):
            return
        if parsed["key"] == "ancient":
            sideless_t4[parsed["team"]] += 1
        else:
            destroyed[parsed["team"]].add(parsed["key"])

    for b in buildings or []:
        if b.get("destroyed_time") is None:
            continue
        _mark(parse_building_name(b.get("name")))
    objective_t4: Counter = Counter()
    for obj in objectives or []:
        if str(obj.get("type", "")) not in ("building_kill", "tower_kill", "barracks_kill",
                                              "CHAT_MESSAGE_TOWER_KILL"):
            continue
        parsed = parse_building_name(str(obj.get("key") or obj.get("unit") or ""))
        if parsed and parsed["type"] == "tower" and parsed["key"].startswith("ancient"):
            objective_t4[parsed["team"]] += 1  # reconciled with the buildings below
            continue
        _mark(parsed)
    for team in ("radiant", "dire"):
        known = sum(1 for k in ("ancient_top", "ancient_bot") if k in destroyed[team])
        missing = max(sideless_t4[team], objective_t4[team]) - known
        for key in ("ancient_top", "ancient_bot"):
            if missing <= 0:
                break
            if key not in destroyed[team]:
                destroyed[team].add(key)
                missing -= 1

    result: dict[str, int] = {}
    for team in ("radiant", "dire"):
        towers = _ALL_TOWERS_ALIVE
        for i, key in enumerate(TOWER_BIT_ORDER):
            if key in destroyed[team]:
                towers &= ~(1 << i)
        barracks = _ALL_BARRACKS_ALIVE
        for i, key in enumerate(BARRACKS_BIT_ORDER):
            if key in destroyed[team]:
                barracks &= ~(1 << i)
        result[f"tower_status_{team}"] = towers
        result[f"barracks_status_{team}"] = barracks
    return result


# ---------------------------------------------------------------------------
# Lane inference from world positions (Clarity laning snapshots)
# ---------------------------------------------------------------------------

_LANE_CODES = {"bot": 1, "mid": 2, "top": 3, "jungle": 4}
# Relative lane_role per team: 1=safe, 2=mid, 3=off, 4=jungle/roaming
_LANE_ROLE = {
    "radiant": {1: 1, 2: 2, 3: 3, 4: 4},
    "dire": {1: 3, 2: 2, 3: 1, 4: 4},
}


def classify_lane_area(x: float, y: float) -> str | None:
    """Classify a world position as "top", "mid", "bot", "jungle" or None (base)."""
    for team in ("radiant", "dire"):
        ax, ay = geo.ANCIENT_POSITIONS[team]
        fx, fy = geo.FOUNTAIN_POSITIONS[team]
        if geo.distance(x, y, ax, ay) < 2400 or geo.distance(x, y, fx, fy) < 1800:
            return None
    if (x < -4800 and y > -3500) or (y > 4600 and x < 3500):
        return "top"
    if (y < -4600 and x > -3500) or (x > 4800 and y < 3500):
        return "bot"
    if abs(x - y) < 1500:
        return "mid"
    return "jungle"


def infer_lane(positions: list[tuple[float, float]], team: str) -> tuple[int, int, bool]:
    """Infer (lane, lane_role, is_roaming) from early-game world positions.

    ``lane`` is absolute (1=bot, 2=mid, 3=top, 4=jungle, 0=unknown); ``lane_role``
    is relative to ``team`` (1=safe, 2=mid, 3=off, 4=jungle/roaming, 0=unknown).
    """
    areas = [a for a in (classify_lane_area(x, y) for x, y in positions) if a]
    if not areas:
        return 0, 0, False
    counts = Counter(areas)
    area, n = counts.most_common(1)[0]
    lane = _LANE_CODES[area]
    is_roaming = n / len(areas) < 0.5
    lane_role = 4 if is_roaming else _LANE_ROLE.get(team, _LANE_ROLE["radiant"])[lane]
    return lane, lane_role, is_roaming


# ---------------------------------------------------------------------------
# Parser base classes
# ---------------------------------------------------------------------------

class ReplayParser(ABC):
    """Abstract base class for replay parsers."""

    @abstractmethod
    async def parse(self, file_path: str) -> MatchData:
        """Parse a .dem replay file and return structured match data."""
        ...

    @abstractmethod
    def name(self) -> str:
        ...


class MockParser(ReplayParser):
    """Returns mock data for development and testing."""

    async def parse(self, file_path: str) -> MatchData:
        from .mock_data import generate_mock_match
        return generate_mock_match()

    def name(self) -> str:
        return "mock"


# ---------------------------------------------------------------------------
# Clarity (Java) parser
# ---------------------------------------------------------------------------

_DEFAULT_JVM_XMX = "1536m"
_DEFAULT_TIMEOUT_S = 300
_XMX_RE = re.compile(r"^\d+[kKmMgG]?$")
_SLOT_POLL_INTERVAL_S = 0.2
_STDERR_LOG_LIMIT = 2000
_MIN_JAVA_MAJOR = 17

# Analyzer exit codes (see ReplayAnalyzer.java) -> user-safe message
_EXIT_MESSAGES = {
    3: "Replay parser ran out of memory",
    64: "Replay parser was invoked incorrectly",
    65: "The file is not a valid Dota 2 replay or is corrupted",
    66: "Replay file could not be read",
    70: "Replay parsing failed",
}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(float(raw))
    except ValueError:
        logger.warning("Ignoring invalid %s=%r (using %s)", name, raw, default)
        return default
    return max(minimum, value)


def _jvm_xmx() -> str:
    raw = os.environ.get("CLARITY_JVM_XMX", "").strip() or _DEFAULT_JVM_XMX
    if not _XMX_RE.match(raw):
        logger.warning("Ignoring invalid CLARITY_JVM_XMX=%r (using %s)", raw, _DEFAULT_JVM_XMX)
        return _DEFAULT_JVM_XMX
    return raw


def _clarity_timeout_s() -> int:
    return _env_int("CLARITY_TIMEOUT_S", _DEFAULT_TIMEOUT_S, minimum=5)


# Process-wide concurrency limit for JVM parses. A threading semaphore (polled
# without blocking) works across event loops and threads, and an abandoned wait
# (task cancellation) never leaks a permit.
_clarity_slots: threading.BoundedSemaphore | None = None
_clarity_slots_lock = threading.Lock()


def _get_clarity_slots() -> threading.BoundedSemaphore:
    global _clarity_slots
    with _clarity_slots_lock:
        if _clarity_slots is None:
            _clarity_slots = threading.BoundedSemaphore(_env_int("CLARITY_MAX_CONCURRENCY", 1))
        return _clarity_slots


class _ClaritySlot:
    """Async context manager holding one Clarity concurrency permit."""

    def __init__(self) -> None:
        self._sem: threading.BoundedSemaphore | None = None

    async def __aenter__(self) -> "_ClaritySlot":
        sem = _get_clarity_slots()
        while not sem.acquire(blocking=False):
            await asyncio.sleep(_SLOT_POLL_INTERVAL_S)
        self._sem = sem
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._sem is not None:
            self._sem.release()
            self._sem = None


# java -version probe results, keyed by java executable (probed once per process)
_java_probe: dict[str, bool] = {}
_java_probe_lock = threading.Lock()


def _run_java_version(java_path: str) -> bool:
    try:
        result = subprocess.run(
            [java_path, "-version"], capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Java runtime not usable at %s: %s", java_path, exc)
        return False
    if result.returncode != 0:
        logger.warning("'%s -version' exited with %s", java_path, result.returncode)
        return False
    match = re.search(r'version "(\d+)(?:\.(\d+))?', (result.stderr or "") + (result.stdout or ""))
    if match:
        major = int(match.group(1))
        if major == 1 and match.group(2):  # "1.8.0" style
            major = int(match.group(2))
        if major < _MIN_JAVA_MAJOR:
            logger.warning("Java %s found at %s; clarity needs Java %s+", major, java_path, _MIN_JAVA_MAJOR)
            return False
    return True


def _java_ok(java_path: str) -> bool:
    """Whether ``java_path`` is a usable Java 17+ runtime (cached per process)."""
    with _java_probe_lock:
        cached = _java_probe.get(java_path)
        if cached is not None:
            return cached
        ok = _run_java_version(java_path)
        _java_probe[java_path] = ok
        if ok:
            logger.info("Java runtime available for the clarity parser: %s", java_path)
        return ok


def reset_clarity_probe_cache() -> None:
    """Forget cached Java probe results (tests / after installing Java)."""
    with _java_probe_lock:
        _java_probe.clear()


async def check_clarity_available(jar_path: Path | None = None, java_path: str | None = None) -> bool:
    """Whether the clarity parser can run (analyzer JAR present and Java 17+ usable).

    ``java -version`` runs at most once per process (in a worker thread); the JAR
    check is a cheap stat on every call.
    """
    jar = Path(jar_path or CLARITY_JAR_PATH)
    java = java_path or JAVA_PATH
    if not jar.is_file():
        return False
    with _java_probe_lock:
        cached = _java_probe.get(java)
    if cached is not None:
        return cached
    return await asyncio.to_thread(_java_ok, java)


async def _kill_process(proc: asyncio.subprocess.Process) -> None:
    """Kill a child process (and its process group on POSIX) and reap it."""
    if proc.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass
    except OSError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=10)
    except asyncio.TimeoutError:
        logger.warning("Clarity process %s did not exit after SIGKILL", proc.pid)


async def _run_subprocess(cmd: list[str], timeout: float) -> tuple[bytes, bytes, int]:
    """Run ``cmd`` capturing output; kill it on timeout or task cancellation."""
    kwargs: dict[str, Any] = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True  # own process group -> killpg on timeout
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs,
        )
    except FileNotFoundError as exc:
        raise ParserUnavailableError("Java runtime not found; the replay parser is unavailable") from exc
    except OSError as exc:
        logger.error("Failed to start the clarity parser: %s", exc)
        raise ParserUnavailableError("The replay parser could not be started") from exc

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        await _kill_process(proc)
        raise ReplayParseError("Replay parsing timed out") from None
    except asyncio.CancelledError:
        await _kill_process(proc)
        raise
    return stdout, stderr, proc.returncode if proc.returncode is not None else -1


def _tail(text: str, limit: int = _STDERR_LOG_LIMIT) -> str:
    lines = [ln for ln in text.splitlines() if not ln.startswith("Picked up JAVA_TOOL_OPTIONS")]
    text = "\n".join(lines).strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def _legacy_to_world(value: Any) -> float:
    """Convert the pre-v2 analyzer pseudo-space (cell*62.5-8000) to world units."""
    cell = (_as_float(value) + 8000.0) / 62.5
    return geo.cell_to_world(cell)


# Abilities that never consume a skill point (filtered from skill builds)
_SKIP_ABILITIES = {
    "ability_lamp_use", "ability_capture", "plus_high_five", "ability_pluck_famango",
    "twin_gate_portal_warp", "plus_guild_banner", "abyssal_underlord_portal_warp",
}

# Consumable items excluded when approximating final items from purchases
_CONSUMABLES = {
    "tango", "clarity", "enchanted_mango", "flask", "faerie_fire",
    "ward_observer", "ward_sentry", "dust", "smoke_of_deceit",
    "tome_of_knowledge", "tp_scroll", "tpscroll", "town_portal_scroll",
    "ward_dispenser", "infused_raindrop", "blood_grenade",
}


class ClarityParser(ReplayParser):
    """Parse replays with the clarity Java analyzer.

    Requires a Java 17+ runtime and tools/clarity-analyzer.jar
    (``bash tools/setup_parser.sh``).
    """

    def __init__(self, jar_path: Path | None = None, java_path: str | None = None):
        self.jar_path = Path(jar_path or CLARITY_JAR_PATH)
        self.java_path = java_path or JAVA_PATH

    def name(self) -> str:
        return "clarity"

    def unavailable_reason(self) -> str | None:
        """Why this parser cannot run (None when it can). Probes Java at most once."""
        if not self.jar_path.is_file():
            return "Replay parser is not installed (clarity-analyzer.jar missing)"
        if not _java_ok(self.java_path):
            return "Java 17+ runtime not found; the replay parser is unavailable"
        return None

    def _check_requirements(self) -> None:
        """Deprecated sync check kept for older callers; raises ParserUnavailableError."""
        reason = self.unavailable_reason()
        if reason:
            raise ParserUnavailableError(reason)

    def _command(self, file_path: str) -> list[str]:
        return [
            self.java_path,
            f"-Xmx{_jvm_xmx()}",
            "-XX:+ExitOnOutOfMemoryError",
            "-Dfile.encoding=UTF-8",
            "-jar",
            str(self.jar_path),
            str(file_path),
        ]

    async def parse(self, file_path: str) -> MatchData:
        if not self.jar_path.is_file():
            raise ParserUnavailableError("Replay parser is not installed (clarity-analyzer.jar missing)")
        replay_name = Path(file_path).name
        timeout = _clarity_timeout_s()
        started = time.monotonic()

        async with _ClaritySlot():
            stdout, stderr, returncode = await _run_subprocess(self._command(file_path), timeout)

        stderr_text = stderr.decode("utf-8", errors="replace")
        elapsed = time.monotonic() - started
        if returncode != 0:
            logger.warning(
                "Clarity parser failed for %s (exit %s, %.1fs): %s",
                replay_name, returncode, elapsed, _tail(stderr_text),
            )
            if returncode == 3:
                logger.warning("Raise CLARITY_JVM_XMX (currently %s) if replays run out of memory", _jvm_xmx())
            raise ReplayParseError(_EXIT_MESSAGES.get(returncode, "Replay parsing failed"))

        try:
            raw = json.loads(stdout.decode("utf-8", errors="replace"))
        except ValueError:
            logger.warning(
                "Clarity parser produced invalid JSON for %s (%d bytes): %s",
                replay_name, len(stdout), _tail(stderr_text),
            )
            raise ReplayParseError("Replay parser produced unreadable output") from None
        if not isinstance(raw, dict):
            raise ReplayParseError("Replay parser produced unreadable output")

        if raw.get("partial"):
            logger.warning("Replay %s is incomplete; returning partial match data", replay_name)
        if raw.get("parse_errors"):
            logger.info("Clarity handler errors for %s: %s", replay_name, raw.get("parse_errors"))
        logger.info("Parsed %s with clarity in %.1fs", replay_name, elapsed)
        logger.debug("Clarity stderr for %s: %s", replay_name, _tail(stderr_text))

        try:
            return self._transform_data(raw)
        except Exception as exc:
            logger.exception("Failed to transform clarity output for %s", replay_name)
            raise ReplayParseError("Replay data could not be processed") from exc

    # -- transform helpers --------------------------------------------------

    @staticmethod
    def _resolve_hero(p: dict, lookup: dict[str, dict], hero_map: dict[int, dict]) -> tuple[int, str, str]:
        """(hero_id, display name, slug) from npc/internal name, falling back to hero_id."""
        internal = str(p.get("hero_name") or "")
        for candidate in (p.get("hero_npc_name"), internal):
            info = lookup.get(normalize_hero_key(candidate))
            if info:
                return info["id"], info["name"], info["slug"]
        hero_id = _as_int(p.get("hero_id"))
        info = hero_map.get(hero_id) if hero_id else None
        if info:
            return hero_id, info.get("name") or info.get("slug") or internal, info.get("slug", "")
        return hero_id, internal or "Unknown", ""

    def _transform_data(self, raw: dict) -> MatchData:
        """Transform clarity JSON output into our data models."""
        world = raw.get("coord_space") == "world"
        conv = _as_float if world else _legacy_to_world
        if not world:
            logger.info("Clarity output uses legacy coordinates; converting to world units")

        hero_lookup = _hero_lookup()
        hero_map = load_hero_map()

        # Wards are emitted at match level; attach them to their owners below.
        all_wards: list[WardEvent] = []
        for w in _dicts(raw.get("wards")):
            slot = _as_int(w.get("player_slot"), -1)
            team = w.get("team") if w.get("team") in ("radiant", "dire") else ""
            if not team and slot >= 0:
                team = _team_for_slot(slot)
            destroyed = w.get("destroyed_time")
            all_wards.append(WardEvent(
                ward_type="sentry" if w.get("type") == "sentry" else "observer",
                x=conv(w.get("x")),
                y=conv(w.get("y")),
                time=_as_int(w.get("time")),
                player_slot=slot,
                destroyed_time=_as_int(destroyed) if destroyed is not None else None,
                team=team,
            ))
        wards_by_slot: dict[int, list[WardEvent]] = {}
        for w in all_wards:
            wards_by_slot.setdefault(w.player_slot, []).append(w)

        players: list[PlayerData] = []
        all_ability_events: list[AbilityEvent] = []

        for idx, p in enumerate(_dicts(raw.get("players"))):
            slot = _as_int(p.get("player_slot"), idx if idx < 5 else 128 + idx - 5)
            team = _team_for_slot(slot)
            hero_id, hero_name, hero_slug = self._resolve_hero(p, hero_lookup, hero_map)

            purchase_log = [
                {"time": _as_int(e.get("time")), "key": str(e.get("key") or "")}
                for e in _dicts(p.get("purchase_log") or p.get("item_purchases"))
                if e.get("key")
            ]
            items = [
                ItemEvent(item_name=item_display_name(e["key"]), time=e["time"], item_key=e["key"])
                for e in purchase_log
            ]
            # Starting inventory snapshot (at the horn): add what the pre-game
            # purchase log is missing (multiset difference keeps real purchases).
            snapshot = [s for s in (p.get("starting_items") or []) if isinstance(s, str) and s]
            if snapshot:
                missing = Counter(snapshot) - Counter(it.item_key for it in items if it.time <= 0)
                for key in snapshot:
                    if missing[key] > 0:
                        items.append(ItemEvent(item_name=item_display_name(key), time=-1, item_key=key))
                        missing[key] -= 1
                items.sort(key=lambda it: it.time)

            skills: list[SkillEvent] = []
            for s in _dicts(p.get("ability_upgrades")):
                key = str(s.get("name") or "")
                if not key or key in _SKIP_ABILITIES or "_release" in key:
                    continue
                skills.append(SkillEvent(
                    ability_name=ability_display_name(key),
                    ability_key=str(s.get("key") or key),
                    level=max(1, _as_int(s.get("level"), 1)),
                    time=_as_int(s.get("time")),
                ))

            ability_events = []
            for a in _dicts(p.get("ability_casts")):
                key = str(a.get("name") or "")
                ability_events.append(AbilityEvent(
                    player_slot=slot,
                    hero_name=hero_name,
                    ability_name=ability_display_name(key) if key else "",
                    ability_key=key,
                    time=_as_int(a.get("time")),
                    target=a.get("target"),
                    x=conv(a["x"]) if a.get("x") is not None else None,
                    y=conv(a["y"]) if a.get("y") is not None else None,
                ))
            all_ability_events.extend(ability_events)

            # End-of-game inventory (keys) from the analyzer; purchase heuristic otherwise
            final_keys = [k for k in (p.get("final_items") or []) if isinstance(k, str) and k]
            if not final_keys and purchase_log:
                non_consumable = [e["key"] for e in purchase_log
                                  if e["key"] not in _CONSUMABLES and not e["key"].startswith("recipe_")]
                final_keys = non_consumable[-6:]
            final_items = [item_display_name(k) for k in final_keys]
            backpack = [k for k in (p.get("backpack") or []) if isinstance(k, str) and k]

            snapshots = [
                LaningSnapshot(
                    time=_as_int(s.get("time")),
                    x=conv(s.get("x")),
                    y=conv(s.get("y")),
                    hp=_as_int(s.get("hp")),
                    max_hp=_as_int(s.get("max_hp")),
                    mana=_as_int(s.get("mana")),
                    max_mana=_as_int(s.get("max_mana")),
                    level=_as_int(s.get("level")),
                    last_hits=_as_int(s.get("last_hits")),
                    denies=_as_int(s.get("denies")),
                    net_worth=_as_int(s.get("net_worth")),
                )
                for s in _dicts(p.get("laning_snapshots"))
            ]
            lane = _as_int(p.get("lane"))
            lane_role = _as_int(p.get("lane_role"))
            is_roaming = bool(p.get("is_roaming", False))
            if not lane:
                lane, lane_role, is_roaming = infer_lane(
                    [(s.x, s.y) for s in snapshots if 60 <= s.time <= 600], team,
                )

            player_wards = sorted(wards_by_slot.get(slot, []), key=lambda w: w.time)
            players.append(PlayerData(
                player_slot=slot,
                hero_id=hero_id,
                hero_name=hero_name,
                hero_slug=hero_slug,
                player_name=str(p.get("player_name") or f"Player {idx + 1}"),
                team=team,
                kills=_as_int(p.get("kills")),
                deaths=_as_int(p.get("deaths")),
                assists=_as_int(p.get("assists")),
                level=max(1, _as_int(p.get("level"), 1)),
                last_hits=_as_int(p.get("last_hits")),
                denies=_as_int(p.get("denies")),
                gold_per_min=_as_int(p.get("gold_per_min")),
                xp_per_min=_as_int(p.get("xp_per_min")),
                hero_damage=_as_int(p.get("hero_damage")),
                tower_damage=_as_int(p.get("tower_damage")),
                hero_healing=_as_int(p.get("hero_healing")),
                net_worth=_as_int(p.get("net_worth")),
                items=items,
                backpack=backpack,
                final_items=final_items,
                final_item_keys=final_keys,
                neutral_item=str(p.get("neutral_item") or ""),
                skills=skills,
                wards=player_wards,
                ability_events=ability_events,
                gold_t=[_as_int(v) for v in (p.get("gold_t") or [])],
                xp_t=[_as_int(v) for v in (p.get("xp_t") or [])],
                lh_t=[_as_int(v) for v in (p.get("lh_t") or [])],
                lane=lane,
                lane_role=lane_role,
                is_roaming=is_roaming,
                killed=_int_dict(p.get("killed")),
                killed_by=_int_dict(p.get("killed_by")),
                damage=_int_dict(p.get("damage")),
                damage_taken=_int_dict(p.get("damage_taken")),
                damage_inflictor=_int_dict(p.get("damage_inflictor")),
                gold_reasons=_int_dict(p.get("gold_reasons")),
                xp_reasons=_int_dict(p.get("xp_reasons")),
                runes_log=_dicts(p.get("runes_log")),
                runes=_int_dict(p.get("runes")),
                buyback_count=_as_int(p.get("buyback_count")),
                obs_placed=sum(1 for w in player_wards if w.ward_type == "observer"),
                sen_placed=sum(1 for w in player_wards if w.ward_type == "sentry"),
                observer_kills=_as_int(p.get("observer_kills")),
                sentry_kills=_as_int(p.get("sentry_kills")),
                lane_kills=_as_int(p.get("lane_kills")),
                neutral_kills=_as_int(p.get("neutral_kills")),
                ancient_kills=_as_int(p.get("ancient_kills")),
                tower_kills=_as_int(p.get("tower_kills")),
                roshans_killed=_as_int(p.get("roshans_killed")),
                courier_kills=_as_int(p.get("courier_kills")),
                purchase_log=purchase_log,
                laning_snapshots=snapshots,
                laning_creep_kills=[LaningCreepKill(**_pick(c, LaningCreepKill)) for c in _dicts(p.get("laning_creep_kills"))],
                laning_denies=[LaningCreepKill(**_pick(d, LaningCreepKill)) for d in _dicts(p.get("laning_denies"))],
                laning_damage_events=[LaningDamageEvent(**_pick(e, LaningDamageEvent)) for e in _dicts(p.get("laning_damage_events"))],
                laning_regen_events=[LaningRegenEvent(**_pick(r, LaningRegenEvent)) for r in _dicts(p.get("laning_regen_events"))],
            ))

        slot_to_player = {p.player_slot: p for p in players}

        smokes = []
        for s in _dicts(raw.get("smokes")):
            start = _as_int(s.get("time"))
            slots = s.get("player_slots", s.get("players", []))
            smokes.append(SmokeEvent(
                time=start,
                team=s.get("team") if s.get("team") in ("radiant", "dire") else "radiant",
                player_slots=[_as_int(v) for v in slots] if isinstance(slots, list) else [],
                x=conv(s.get("x")),
                y=conv(s.get("y")),
                end_time=_as_int(s.get("end_time"), start),
                path=[[conv(pt[0]), conv(pt[1])] for pt in (s.get("path") or [])
                      if isinstance(pt, (list, tuple)) and len(pt) >= 2],
                result=str(s.get("result") or "unknown"),
                kills_gained=_as_int(s.get("kills_gained")),
                deaths_suffered=_as_int(s.get("deaths_suffered")),
                objective_taken=s.get("objective_taken") or None,
            ))

        teamfights = []
        for tf in _dicts(raw.get("teamfights")):
            tf_players = []
            for tp in _dicts(tf.get("players")):
                tp_slot = _as_int(tp.get("player_slot"))
                owner = slot_to_player.get(tp_slot)
                if owner is not None:
                    tf_hero_name, tf_hero_slug = owner.hero_name, owner.hero_slug
                else:
                    info = hero_lookup.get(normalize_hero_key(tp.get("hero_name")), {})
                    tf_hero_name = info.get("name", tp.get("hero_name", ""))
                    tf_hero_slug = info.get("slug", "")
                death_off = tp.get("death_time_offset")
                tf_players.append(TeamfightPlayer(
                    player_slot=tp_slot,
                    hero_name=tf_hero_name,
                    hero_slug=tf_hero_slug,
                    x=conv(tp.get("x")),
                    y=conv(tp.get("y")),
                    damage_dealt=_as_int(tp.get("damage_dealt")),
                    damage_taken=_as_int(tp.get("damage_taken")),
                    healing=_as_int(tp.get("healing")),
                    kills=_as_int(tp.get("kills")),
                    deaths=_as_int(tp.get("deaths")),
                    abilities_used=[str(a) for a in (tp.get("abilities_used") or [])],
                    gold_delta=_as_int(tp.get("gold_delta")),
                    xp_delta=_as_int(tp.get("xp_delta")),
                    item_uses=_int_dict(tp.get("item_uses")),
                    buybacks=_as_int(tp.get("buybacks")),
                    death_time_offset=_as_float(death_off) if death_off is not None else None,
                    positions=[
                        PositionSnapshot(x=conv(ps.get("x")), y=conv(ps.get("y")),
                                         time_offset=_as_float(ps.get("time_offset")))
                        for ps in _dicts(tp.get("positions"))
                    ],
                    ability_casts=[
                        AbilityCast(
                            ability_name=ability_display_name(str(a.get("name") or "")) if a.get("name") else "",
                            ability_key=str(a.get("key") or a.get("name") or ""),
                            time_offset=_as_float(a.get("time_offset")),
                            x=conv(a.get("x")),
                            y=conv(a.get("y")),
                        )
                        for a in _dicts(tp.get("ability_casts"))
                    ],
                ))
            teamfights.append(TeamfightEvent(
                start_time=_as_int(tf.get("start", tf.get("start_time"))),
                end_time=_as_int(tf.get("end", tf.get("end_time"))),
                radiant_deaths=_as_int(tf.get("radiant_deaths")),
                dire_deaths=_as_int(tf.get("dire_deaths")),
                gold_delta=_as_int(tf.get("gold_delta")),
                players=tf_players,
            ))

        objectives = [_normalize_clarity_objective(o) for o in _dicts(raw.get("objectives"))]
        objectives.sort(key=lambda o: _as_int(o.get("time")))
        buildings = _normalize_buildings(_dicts(raw.get("buildings")), conv)
        status = building_status_bitmasks(buildings, objectives)

        duration = _as_int(raw.get("duration"))
        return MatchData(
            match_id=_as_int(raw.get("match_id")),
            duration=duration,
            radiant_win=bool(raw.get("radiant_win", False)),
            game_mode=game_mode_code(raw.get("game_mode", 0)),
            start_time=_as_int(raw.get("start_time")),
            radiant_score=_as_int(raw.get("radiant_score")),
            dire_score=_as_int(raw.get("dire_score")),
            players=players,
            wards=sorted(all_wards, key=lambda w: w.time),
            smokes=sorted(smokes, key=lambda s: s.time),
            teamfights=teamfights,
            ability_events=sorted(all_ability_events, key=lambda e: e.time),
            gold_xp_timeline=_gold_xp_timeline(players),
            radiant_gold_adv=[_as_int(v) for v in (raw.get("radiant_gold_adv") or [])],
            radiant_xp_adv=[_as_int(v) for v in (raw.get("radiant_xp_adv") or [])],
            picks_bans=_dicts(raw.get("picks_bans")),
            objectives=objectives,
            chat=_dicts(raw.get("chat")),
            first_blood_time=_as_int(raw.get("first_blood_time")),
            region=_as_int(raw.get("region")),
            patch=str(raw.get("patch") or ""),
            buildings=buildings,
            partial=bool(raw.get("partial", False)),
            **status,
        )


def _pick(data: dict, model: type) -> dict:
    """Keep only the fields a pydantic model declares (ignores extra/unknown keys)."""
    fields = model.model_fields
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key not in fields or value is None:
            continue
        annotation = fields[key].annotation
        if annotation is int:
            value = _as_int(value)
        elif annotation is float:
            value = _as_float(value)
        elif annotation is str:
            value = str(value)
        out[key] = value
    return out


def _normalize_clarity_objective(obj: dict) -> dict:
    """Map analyzer objective events onto the OpenDota-style shape consumers expect."""
    otype = str(obj.get("type") or "")
    out: dict[str, Any] = {"time": _as_int(obj.get("time"))}
    if obj.get("player_slot") is not None:
        out["player_slot"] = _as_int(obj.get("player_slot"))
    if otype in ("tower_kill", "barracks_kill", "ancient_kill", "building_kill"):
        key = str(obj.get("key") or "")
        if key and not key.startswith("npc_dota_"):
            key = "npc_dota_" + key
        parsed = parse_building_name(key)
        out["type"] = "building_kill"
        out["key"] = key
        if parsed:
            # J8: the destroyed building's own name decides the teams.
            out["building_team"] = parsed["team"]
            out["team"] = "dire" if parsed["team"] == "radiant" else "radiant"
        elif obj.get("team") in ("radiant", "dire"):
            out["team"] = obj["team"]
        if obj.get("denied"):
            out["denied"] = True
        return out
    if otype == "roshan_kill":
        out["type"] = "CHAT_MESSAGE_ROSHAN_KILL"
    else:
        out["type"] = otype
    for key in ("team", "key", "slot"):
        if obj.get(key) is not None:
            out[key] = obj[key]
    return out


def _annotate_building_objective(obj: dict) -> dict:
    """Add ``building_team`` / ``team`` (the side that took it) to a building kill.

    OpenDota's building_kill entries carry the killer in ``unit`` and the
    building in ``key`` but no team; derive both from the building's name.
    """
    if obj.get("type") != "building_kill":
        return obj
    parsed = parse_building_name(str(obj.get("key") or ""))
    if not parsed:
        return obj
    out = dict(obj)
    out.setdefault("building_team", parsed["team"])
    if out.get("team") not in ("radiant", "dire"):
        out["team"] = "dire" if parsed["team"] == "radiant" else "radiant"
    return out


def _normalize_buildings(raw_buildings: list[dict], conv) -> list[dict]:
    """Canonical building list (contract shape), names inferred from position if needed."""
    out: list[dict] = []
    ancient_seen: dict[str, int] = {"radiant": 0, "dire": 0}
    for b in raw_buildings:
        x, y = conv(b.get("x")), conv(b.get("y"))
        btype = b.get("type") if b.get("type") in ("tower", "barracks", "fort") else "other"
        parsed = parse_building_name(b.get("name"))
        team = b.get("team") if b.get("team") in ("radiant", "dire") else (parsed or {}).get("team")
        if team not in ("radiant", "dire"):
            team = "radiant" if x + y < 0 else "dire"
        name = str(b.get("name") or "")
        if parsed:
            key = parsed["key"]
            btype = parsed["type"]
            if key == "ancient":
                # Tier-4 tower without side: decide by the nearest default position.
                near = _nearest_tower_key(team, x, y, max_dist=900.0)
                if near in ("ancient_top", "ancient_bot"):
                    key = near
                else:
                    key = ("ancient_top", "ancient_bot")[min(ancient_seen[team], 1)]
                    ancient_seen[team] += 1
            name = _canonical_building_name(parsed["team"], btype, key)
        elif btype == "tower":
            near = _nearest_tower_key(team, x, y)
            if near:
                name = _canonical_building_name(team, "tower", near)
        elif btype == "fort":
            name = _canonical_building_name(team, "fort", "fort")
        destroyed = b.get("destroyed_time")
        out.append({
            "name": name,
            "team": team,
            "type": btype,
            "x": x,
            "y": y,
            "destroyed_time": _as_int(destroyed) if destroyed is not None else None,
        })
    return out


def _gold_xp_timeline(players: list[PlayerData]) -> list[GoldXpSnapshot]:
    """Team gold/xp per minute from the per-player gold_t / xp_t timelines."""
    radiant = [p for p in players if p.team == "radiant"]
    dire = [p for p in players if p.team == "dire"]
    if not radiant or not dire:
        return []
    min_len = min(len(p.gold_t) for p in radiant + dire)
    timeline: list[GoldXpSnapshot] = []
    for minute in range(min_len):
        timeline.append(GoldXpSnapshot(
            time=minute * 60,
            radiant_gold=sum(p.gold_t[minute] for p in radiant),
            dire_gold=sum(p.gold_t[minute] for p in dire),
            radiant_xp=sum(p.xp_t[minute] for p in radiant if len(p.xp_t) > minute),
            dire_xp=sum(p.xp_t[minute] for p in dire if len(p.xp_t) > minute),
        ))
    return timeline


# ---------------------------------------------------------------------------
# OpenDota parser
# ---------------------------------------------------------------------------

# Cumulative XP needed to reach each hero level (patch 7.3x; approximate).
_XP_FOR_LEVEL = [
    0, 240, 640, 1160, 1760, 2440, 3200, 4000, 4900, 5900,
    7000, 8200, 9500, 10900, 12400, 14000, 15700, 17500, 19400, 21400,
    23600, 26000, 28600, 31400, 34400, 38400, 43400, 49400, 56400, 63900,
]
# Hero levels that grant an ability point (17, 19 and 21-24 do not).
_POINT_LEVELS = list(range(1, 17)) + [18, 20] + list(range(25, 31))

_ability_ids: dict[str, str] | None = None
_ability_ids_failed_at = 0.0
_ability_ids_lock = threading.Lock()
_ABILITY_IDS_RETRY_S = 600


def _estimate_upgrade_times(count: int, xp_t: list[int]) -> list[int]:
    """Approximate when each ability point was spent, from the per-minute XP curve."""
    if not xp_t:
        return [(i + 1) * 60 for i in range(count)]
    level_time: dict[int, float] = {1: 0.0}
    for level in range(2, len(_XP_FOR_LEVEL) + 1):
        need = _XP_FOR_LEVEL[level - 1]
        for minute, xp in enumerate(xp_t):
            if xp >= need:
                if minute == 0:
                    level_time[level] = 0.0
                else:
                    prev = xp_t[minute - 1]
                    frac = (need - prev) / (xp - prev) if xp > prev else 1.0
                    level_time[level] = (minute - 1 + max(0.0, min(1.0, frac))) * 60
                break
    times: list[int] = []
    prev_t = 0.0
    for i in range(count):
        level = _POINT_LEVELS[i] if i < len(_POINT_LEVELS) else _POINT_LEVELS[-1]
        t = level_time.get(level)
        if t is None:
            t = prev_t + 60
        t = max(t, prev_t)
        times.append(int(t))
        prev_t = t
    return times


class OpenDotaParser(ReplayParser):
    """Fetch and parse match data from the OpenDota API."""

    API_BASE = "https://api.opendota.com/api"
    HEADERS = {"User-Agent": "DotaAnalyzer/1.0"}
    TIMEOUT_S = 30

    def name(self) -> str:
        return "opendota"

    def _url(self, path: str) -> str:
        url = f"{self.API_BASE}{path}"
        api_key = os.environ.get("OPENDOTA_API_KEY", "").strip()
        if api_key:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode({"api_key": api_key})
        return url

    def _request(self, path: str, method: str = "GET") -> Any:
        req = urllib.request.Request(self._url(path), method=method, headers=self.HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=self.TIMEOUT_S) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            logger.warning("OpenDota %s %s failed: HTTP %s", method, path, status)
            if status == 404:
                raise OpenDotaError("Match not found on OpenDota", status=404) from None
            if status == 429:
                raise OpenDotaError("OpenDota rate limit reached, please retry later", status=429) from None
            if status >= 500:
                raise OpenDotaError("OpenDota is temporarily unavailable", status=status) from None
            raise OpenDotaError(f"OpenDota request failed (HTTP {status})", status=status) from None
        except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
            logger.warning("OpenDota %s %s failed: %s", method, path, exc)
            raise OpenDotaError("Could not reach OpenDota", status=None) from None
        try:
            return json.loads(body)
        except ValueError:
            logger.warning("OpenDota %s %s returned invalid JSON (%d bytes)", method, path, len(body))
            raise OpenDotaError("Invalid response from OpenDota", status=None) from None

    def _api_get(self, path: str) -> Any:
        return self._request(path, "GET")

    def _api_post(self, path: str) -> Any:
        return self._request(path, "POST")

    def request_parse(self, match_id: int) -> dict:
        """Request OpenDota to parse a match. Returns job info."""
        result = self._api_post(f"/request/{int(match_id)}")
        return result if isinstance(result, dict) else {}

    def _load_item_map(self) -> dict[int, tuple[str, str]]:
        """Load item id -> (name, key) map from static data."""
        return _item_id_map()

    def _load_ability_ids(self) -> dict[str, str]:
        """OpenDota ability id -> ability key (cached; retried after failures)."""
        global _ability_ids, _ability_ids_failed_at
        with _ability_ids_lock:
            if _ability_ids is not None:
                return _ability_ids
            if time.monotonic() - _ability_ids_failed_at < _ABILITY_IDS_RETRY_S and _ability_ids_failed_at:
                return {}
        try:
            data = self._api_get("/constants/ability_ids")
        except OpenDotaError as exc:
            logger.warning("OpenDota ability id constants unavailable: %s", exc)
            data = None
        with _ability_ids_lock:
            if isinstance(data, dict) and data:
                _ability_ids = {str(k): str(v) for k, v in data.items()}
                return _ability_ids
            _ability_ids_failed_at = time.monotonic()
            return {}

    async def parse(self, match_id_str: str) -> MatchData:
        try:
            match_id = int(str(match_id_str).strip())
        except ValueError:
            raise OpenDotaError("Invalid match ID", status=400) from None
        if match_id <= 0:
            raise OpenDotaError("Invalid match ID", status=400)
        # Blocking urllib calls run in worker threads (keeps the event loop free).
        raw = await asyncio.to_thread(self._api_get, f"/matches/{match_id}")
        if not isinstance(raw, dict) or ("error" in raw and not raw.get("players")):
            raise OpenDotaError("Match not found on OpenDota", status=404)
        ability_ids: dict[str, str] = {}
        if any(p.get("ability_upgrades_arr") for p in _dicts(raw.get("players"))):
            ability_ids = await asyncio.to_thread(self._load_ability_ids)
        return self._transform(raw, load_hero_map(), self._load_item_map(), ability_ids)

    # Typical starting items (fallback costs when items.json has no cost)
    _STARTING_ITEM_COSTS: dict[str, int] = {
        "tango": 90, "flask": 110, "faerie_fire": 70, "enchanted_mango": 65,
        "clarity": 50, "branches": 50, "circlet": 150, "slippers": 135,
        "gauntlets": 135, "mantle": 135, "quelling_blade": 100, "magic_stick": 200,
        "ring_of_protection": 175, "orb_of_venom": 275, "blight_stone": 300,
        "wind_lace": 250, "blood_grenade": 65,
        "ward_observer": 0, "ward_sentry": 50, "smoke_of_deceit": 50,
    }
    _STARTING_GOLD = 600
    _STARTING_INFER_WINDOW = 90  # seconds after game start

    def _starting_item_cost(self, key: str) -> int | None:
        if key not in self._STARTING_ITEM_COSTS:
            return None
        info = _items_by_key().get(key)
        cost = info.get("cost") if isinstance(info, dict) else None
        return cost if isinstance(cost, int) and cost >= 0 else self._STARTING_ITEM_COSTS[key]

    def _infer_starting_items(self, items: list[ItemEvent]) -> list[ItemEvent]:
        """Reclassify early purchases as starting items when purchase_log is incomplete.

        OpenDota's purchase_log often drops pre-game purchases. If the recorded
        starting purchases cost clearly less than the starting gold, typical
        starting items bought in the first 90 seconds are moved to time -1.
        """
        starting_cost = sum(self._starting_item_cost(it.item_key) or 0 for it in items if it.time <= 0)
        if starting_cost >= self._STARTING_GOLD - 50:
            return items

        budget = self._STARTING_GOLD - starting_cost
        result = list(items)
        for i, it in enumerate(result):
            if it.time <= 0:
                continue
            if it.time > self._STARTING_INFER_WINDOW:
                break
            cost = self._starting_item_cost(it.item_key)
            if cost is None:
                continue  # not a typical starting item
            if cost <= budget:
                result[i] = ItemEvent(item_name=it.item_name, time=-1, item_key=it.item_key, charges=it.charges)
                budget -= cost
                if budget < 50:
                    break
        result.sort(key=lambda x: x.time)
        return result

    @staticmethod
    def _lane_fields(p: dict, team: str) -> tuple[int, int, bool]:
        """(lane, lane_role, is_roaming) per the shared contract.

        OpenDota lane: 1=bot, 2=mid, 3=top, 4=radiant jungle, 5=dire jungle.
        Both jungles become 4; lane_role is derived from lane when missing.
        """
        lane = _as_int(p.get("lane"))
        if lane == 5:
            lane = 4
        if lane not in (0, 1, 2, 3, 4):
            lane = 0
        lane_role = _as_int(p.get("lane_role"))
        if lane_role not in (0, 1, 2, 3, 4):
            lane_role = 0
        if lane and not lane_role:
            lane_role = _LANE_ROLE[team][lane]
        if lane_role and not lane:
            inverse = {v: k for k, v in _LANE_ROLE[team].items()}
            lane = inverse.get(lane_role, 0)
        return lane, lane_role, bool(p.get("is_roaming", False))

    @staticmethod
    def _ward_events(p: dict, slot: int, team: str) -> list[WardEvent]:
        """Ward placements with world coordinates and destruction times."""
        wards: list[WardEvent] = []
        for log_key, left_key, ward_type in (("obs_log", "obs_left_log", "observer"),
                                             ("sen_log", "sen_left_log", "sentry")):
            left_entries = _dicts(p.get(left_key))
            by_handle = {w["ehandle"]: _as_int(w.get("time")) for w in left_entries if w.get("ehandle")}
            by_pos: dict[tuple, list[int]] = {}
            for w in left_entries:
                if not w.get("ehandle"):
                    by_pos.setdefault((round(_as_float(w.get("x")), 1), round(_as_float(w.get("y")), 1)), []).append(
                        _as_int(w.get("time")))
            for w in _dicts(p.get(log_key)):
                placed = _as_int(w.get("time"))
                destroyed = None
                ehandle = w.get("ehandle")
                if ehandle and ehandle in by_handle:
                    destroyed = by_handle[ehandle]
                else:
                    pos_key = (round(_as_float(w.get("x")), 1), round(_as_float(w.get("y")), 1))
                    candidates = sorted(t for t in by_pos.get(pos_key, []) if t >= placed)
                    if candidates:
                        destroyed = candidates[0]
                        by_pos[pos_key].remove(destroyed)
                wards.append(WardEvent(
                    ward_type=ward_type,
                    x=geo.opendota_cell_to_world(_as_float(w.get("x"))),
                    y=geo.opendota_cell_to_world(_as_float(w.get("y"))),
                    time=placed,
                    player_slot=slot,
                    destroyed_time=destroyed,
                    team=team,
                ))
        return sorted(wards, key=lambda w: w.time)

    def _transform(self, raw: dict, hero_map: dict, item_map: dict,
                   ability_ids: dict[str, str] | None = None) -> MatchData:
        ability_ids = ability_ids or {}
        raw_players = _dicts(raw.get("players"))
        players: list[PlayerData] = []
        all_wards: list[WardEvent] = []

        for idx, p in enumerate(raw_players):
            slot = _as_int(p.get("player_slot"), idx if idx < 5 else 128 + idx - 5)
            team = _team_for_slot(slot)
            hero_id = _as_int(p.get("hero_id"))
            hero_info = hero_map.get(hero_id, {})
            hero_name = hero_info.get("name") or f"Hero {hero_id}"
            hero_slug = hero_info.get("slug", "")

            purchase_log = _dicts(p.get("purchase_log"))
            items = [
                ItemEvent(
                    item_name=item_display_name(str(e.get("key") or "")),
                    time=_as_int(e.get("time")),
                    item_key=str(e.get("key") or ""),
                    charges=_as_int(e["charges"]) if e.get("charges") is not None else None,
                )
                for e in purchase_log if e.get("key")
            ]
            items = self._infer_starting_items(items)

            final_items: list[str] = []
            final_keys: list[str] = []
            for i in range(6):
                iid = _as_int(p.get(f"item_{i}"))
                if iid and iid in item_map:
                    name, key = item_map[iid]
                    final_items.append(name)
                    final_keys.append(key)
            backpack = [item_map[iid][1] for iid in (_as_int(p.get(f"backpack_{i}")) for i in range(3))
                        if iid and iid in item_map]
            neutral_id = _as_int(p.get("item_neutral"))
            neutral_item = item_map[neutral_id][1] if neutral_id in item_map else ""

            # Skill build: ability ids -> keys via OpenDota constants, times from the XP curve
            ability_arr = [a for a in (p.get("ability_upgrades_arr") or []) if a is not None]
            upgrade_times = _estimate_upgrade_times(len(ability_arr), [_as_int(v) for v in (p.get("xp_t") or [])])
            skills: list[SkillEvent] = []
            levels: Counter = Counter()
            for i, ab_id in enumerate(ability_arr):
                key = ability_ids.get(str(ab_id)) or f"ability_{ab_id}"
                levels[key] += 1
                skills.append(SkillEvent(
                    ability_name=ability_display_name(key),
                    ability_key=key if str(ab_id) in ability_ids else str(ab_id),
                    level=levels[key],
                    time=upgrade_times[i],
                ))

            wards = self._ward_events(p, slot, team)
            all_wards.extend(wards)
            lane, lane_role, is_roaming = self._lane_fields(p, team)

            players.append(PlayerData(
                player_slot=slot,
                hero_id=hero_id,
                hero_name=hero_name,
                hero_slug=hero_slug,
                player_name=p.get("personaname") or p.get("name") or f"Player {idx + 1}",
                team=team,
                kills=_as_int(p.get("kills")),
                deaths=_as_int(p.get("deaths")),
                assists=_as_int(p.get("assists")),
                level=max(1, _as_int(p.get("level"), 1)),
                last_hits=_as_int(p.get("last_hits")),
                denies=_as_int(p.get("denies")),
                gold_per_min=_as_int(p.get("gold_per_min")),
                xp_per_min=_as_int(p.get("xp_per_min")),
                hero_damage=_as_int(p.get("hero_damage")),
                tower_damage=_as_int(p.get("tower_damage")),
                hero_healing=_as_int(p.get("hero_healing")),
                net_worth=_as_int(p.get("net_worth")) or _as_int(p.get("total_gold")),
                items=items,
                backpack=backpack,
                final_items=final_items,
                final_item_keys=final_keys,
                neutral_item=neutral_item,
                skills=skills,
                wards=wards,
                ability_events=[],
                gold_t=[_as_int(v) for v in (p.get("gold_t") or [])],
                xp_t=[_as_int(v) for v in (p.get("xp_t") or [])],
                lh_t=[_as_int(v) for v in (p.get("lh_t") or [])],
                lane=lane,
                lane_role=lane_role,
                is_roaming=is_roaming,
                lane_efficiency=_as_float(p.get("lane_efficiency")),
                lane_efficiency_pct=_as_float(p.get("lane_efficiency_pct")),
                killed=_int_dict(p.get("killed")),
                killed_by=_int_dict(p.get("killed_by")),
                damage=_int_dict(p.get("damage")),
                damage_taken=_int_dict(p.get("damage_taken")),
                damage_inflictor=_int_dict(p.get("damage_inflictor")),
                gold_reasons=_int_dict(p.get("gold_reasons")),
                xp_reasons=_int_dict(p.get("xp_reasons")),
                benchmarks=p.get("benchmarks") if isinstance(p.get("benchmarks"), dict) else {},
                runes_log=_dicts(p.get("runes_log")),
                runes=_int_dict(p.get("runes")),
                stuns=_as_float(p.get("stuns")),
                actions_per_min=_as_int(p.get("actions_per_min")),
                teamfight_participation=_as_float(p.get("teamfight_participation")),
                buyback_count=len(_dicts(p.get("buyback_log"))) or _as_int(p.get("buyback_count")),
                obs_placed=_as_int(p.get("obs_placed")),
                sen_placed=_as_int(p.get("sen_placed")),
                observer_kills=_as_int(p.get("observer_kills")),
                sentry_kills=_as_int(p.get("sentry_kills")),
                lane_kills=_as_int(p.get("lane_kills")),
                neutral_kills=_as_int(p.get("neutral_kills")),
                ancient_kills=_as_int(p.get("ancient_kills")),
                tower_kills=_as_int(p.get("tower_kills")),
                roshans_killed=_as_int(p.get("roshan_kills", p.get("roshans_killed"))),
                courier_kills=_as_int(p.get("courier_kills")),
                multi_kills=_int_dict(p.get("multi_kills")),
                kill_streaks=_int_dict(p.get("kill_streaks")),
                purchase_log=purchase_log,
                lane_pos=p.get("lane_pos") if isinstance(p.get("lane_pos"), dict) else {},
            ))

        teamfights = [self._teamfight(tf, raw_players, hero_map) for tf in _dicts(raw.get("teamfights"))]

        mode = raw.get("game_mode", 0)
        return MatchData(
            match_id=_as_int(raw.get("match_id")),
            duration=_as_int(raw.get("duration")),
            radiant_win=bool(raw.get("radiant_win", False)),
            game_mode=game_mode_code(mode),
            start_time=_as_int(raw.get("start_time")),
            radiant_score=_as_int(raw.get("radiant_score")),
            dire_score=_as_int(raw.get("dire_score")),
            players=players,
            wards=sorted(all_wards, key=lambda w: w.time),
            smokes=[],
            teamfights=teamfights,
            ability_events=[],
            gold_xp_timeline=_gold_xp_timeline(players),
            radiant_gold_adv=[_as_int(v) for v in (raw.get("radiant_gold_adv") or [])],
            radiant_xp_adv=[_as_int(v) for v in (raw.get("radiant_xp_adv") or [])],
            picks_bans=_dicts(raw.get("picks_bans")),
            objectives=[_annotate_building_objective(o) for o in _dicts(raw.get("objectives"))],
            chat=_dicts(raw.get("chat")),
            tower_status_radiant=_as_int(raw.get("tower_status_radiant")),
            tower_status_dire=_as_int(raw.get("tower_status_dire")),
            barracks_status_radiant=_as_int(raw.get("barracks_status_radiant")),
            barracks_status_dire=_as_int(raw.get("barracks_status_dire")),
            first_blood_time=_as_int(raw.get("first_blood_time")),
            region=_as_int(raw.get("region")),
            patch=str(raw.get("patch") or ""),
        )

    @staticmethod
    def _teamfight(tf: dict, raw_players: list[dict], hero_map: dict) -> TeamfightEvent:
        """OpenDota teamfight -> TeamfightEvent (positions from deaths_pos cells)."""
        tf_entries = _dicts(tf.get("players"))

        def _death_points(entry: dict) -> list[tuple[float, float, int]]:
            points = []
            pos = entry.get("deaths_pos")
            if isinstance(pos, dict):
                for cx, ys in pos.items():
                    if not isinstance(ys, dict):
                        continue
                    for cy, count in ys.items():
                        points.append((geo.opendota_cell_to_world(_as_float(cx)),
                                       geo.opendota_cell_to_world(_as_float(cy)), max(1, _as_int(count, 1))))
            return points

        all_points = [pt for e in tf_entries for pt in _death_points(e)]
        if all_points:
            total = sum(c for _, _, c in all_points)
            center = (sum(x * c for x, _, c in all_points) / total, sum(y * c for _, y, c in all_points) / total)
        else:
            center = (0.0, 0.0)

        players: list[TeamfightPlayer] = []
        r_gold_delta = 0
        r_deaths = d_deaths = 0
        for i, tp in enumerate(tf_entries):
            source = raw_players[i] if i < len(raw_players) else {}
            slot = _as_int(source.get("player_slot"), i if i < 5 else 128 + i - 5)
            team = _team_for_slot(slot)
            hero_info = hero_map.get(_as_int(source.get("hero_id")), {})
            gold_delta = _as_int(tp.get("gold_delta"))
            r_gold_delta += gold_delta if team == "radiant" else -gold_delta
            deaths = _as_int(tp.get("deaths"))
            if team == "radiant":
                r_deaths += deaths
            else:
                d_deaths += deaths
            killed = tp.get("killed") if isinstance(tp.get("killed"), dict) else {}
            own_points = _death_points(tp)
            if own_points:
                c = sum(n for _, _, n in own_points)
                px = sum(x * n for x, _, n in own_points) / c
                py = sum(y * n for _, y, n in own_points) / c
            else:
                px, py = center
            players.append(TeamfightPlayer(
                player_slot=slot,
                hero_name=hero_info.get("name", ""),
                hero_slug=hero_info.get("slug", ""),
                x=px,
                y=py,
                damage_dealt=_as_int(tp.get("damage")),
                damage_taken=0,
                healing=_as_int(tp.get("healing")),
                kills=sum(_as_int(v) for v in killed.values()) if killed else _as_int(tp.get("kills")),
                deaths=deaths,
                abilities_used=list((tp.get("ability_uses") or {}).keys()) if isinstance(tp.get("ability_uses"), dict) else [],
                gold_delta=gold_delta,
                xp_delta=_as_int(tp.get("xp_delta")),
                item_uses=_int_dict(tp.get("item_uses")),
                buybacks=_as_int(tp.get("buybacks")),
            ))
        return TeamfightEvent(
            start_time=_as_int(tf.get("start")),
            end_time=_as_int(tf.get("end")),
            radiant_deaths=r_deaths,
            dire_deaths=d_deaths,
            gold_delta=r_gold_delta,
            players=players,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_parser(parser_type: str = "auto") -> ReplayParser:
    """Return a parser instance.

    Args:
        parser_type: "clarity", "opendota", "mock", or "auto". "auto" returns the
            clarity parser and raises :class:`ParserUnavailableError` when Java or
            the analyzer JAR is unavailable (it never falls back to mock data).
            The Java probe runs at most once per process; call
            ``await check_clarity_available()`` first to keep it off the event loop.
    """
    if parser_type == "mock":
        return MockParser()
    if parser_type == "clarity":
        return ClarityParser()
    if parser_type == "opendota":
        return OpenDotaParser()
    if parser_type == "auto":
        clarity = ClarityParser()
        reason = clarity.unavailable_reason()
        if reason:
            logger.warning("Clarity parser unavailable: %s", reason)
            raise ParserUnavailableError(reason)
        return clarity
    raise ValueError(f"Unknown parser type: {parser_type}")
