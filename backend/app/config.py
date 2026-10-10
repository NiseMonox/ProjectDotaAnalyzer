"""Application configuration: project paths plus environment-driven settings.

Every tunable here can be overridden with an environment variable (see
``.env.example`` at the project root for the full list). ``<project>/.env``
is loaded here, so importing this module first is enough. Values are read
once at import time; security-sensitive values that operators may rotate
(``ADMIN_TOKEN``) are read per request in :mod:`backend.app.security`.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def env_str(name: str, default: str = "") -> str:
    """Return the stripped value of ``name`` or ``default`` when unset/blank."""
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def env_int(name: str, default: int, minimum: int | None = None) -> int:
    """Parse an integer env var, falling back to ``default`` on bad input."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        logger.warning("Invalid integer for %s=%r, using default %d", name, raw, default)
        return default
    if minimum is not None and value < minimum:
        logger.warning("%s=%d is below minimum %d, using %d", name, value, minimum, minimum)
        return minimum
    return value


def env_float(name: str, default: float, minimum: float | None = None) -> float:
    """Parse a float env var, falling back to ``default`` on bad input."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError:
        logger.warning("Invalid number for %s=%r, using default %s", name, raw, default)
        return default
    if minimum is not None and value < minimum:
        return minimum
    return value


def env_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var ("1/true/yes/on" vs "0/false/no/off")."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    logger.warning("Invalid boolean for %s=%r, using default %s", name, raw, default)
    return default


def env_list(name: str) -> list[str]:
    """Parse a comma-separated env var into a list of non-empty items."""
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load <project>/.env before any setting below is read. Real environment
# variables (systemd EnvironmentFile, shell exports) take precedence.
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv is a hard requirement
    load_dotenv = None
if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env", override=False)

REPLAY_DIR = BASE_DIR / "replays"
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
CACHE_DIR = BASE_DIR / "cache"

# Default test replay (used by /api/demo when present)
DEFAULT_REPLAY = REPLAY_DIR / "8713267082.dem"


# ---------------------------------------------------------------------------
# Parser settings
# ---------------------------------------------------------------------------

def project_path(value: str) -> Path:
    """Resolve a configured path; relative paths are relative to BASE_DIR."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else BASE_DIR / path


CLARITY_JAR_PATH = project_path(env_str("CLARITY_JAR_PATH", "tools/clarity-analyzer.jar"))

# Java 17+ is required by clarity 4.x (5.x would need Java 21). Resolution order:
#   1. $JAVA_PATH (explicit override)
#   2. project-local JDK in tools/jdk17 (macOS bundle or Linux layout)
#   3. $JAVA_HOME/bin/java
#   4. `java` on PATH
_LOCAL_JDK_CANDIDATES = (
    BASE_DIR / "tools" / "jdk17" / "Contents" / "Home" / "bin" / "java",  # macOS .jdk bundle
    BASE_DIR / "tools" / "jdk17" / "bin" / "java",                        # Linux tarball
)


def _resolve_java_path() -> str:
    explicit = env_str("JAVA_PATH")
    if explicit:
        # A bare command name ("java") is looked up on PATH by the OS.
        return str(project_path(explicit)) if ("/" in explicit or "\\" in explicit) else explicit
    for candidate in _LOCAL_JDK_CANDIDATES:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    java_home = env_str("JAVA_HOME")
    if java_home:
        candidate = Path(java_home) / "bin" / "java"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which("java") or "java"


JAVA_PATH = _resolve_java_path()


# ---------------------------------------------------------------------------
# Server settings
# ---------------------------------------------------------------------------

# Used by `python -m backend.app.main` (run.sh / systemd pass them to uvicorn).
HOST = env_str("HOST", "0.0.0.0")
PORT = env_int("PORT", 8000, minimum=1)
# DEV=1 enables uvicorn --reload when started via `python -m backend.app.main`.
DEV = env_bool("DEV", False)

LOG_LEVEL = env_str("LOG_LEVEL", "INFO").upper()

# CORS: comma-separated origin allowlist. Empty → no CORS middleware at all
# (the bundled frontend is served from the same origin).
ALLOWED_ORIGINS = env_list("ALLOWED_ORIGINS")

# Honour X-Real-IP / X-Forwarded-For for client IPs (only behind a proxy you
# control, otherwise clients can spoof their rate-limit identity).
TRUST_PROXY_HEADERS = env_bool("TRUST_PROXY_HEADERS", False)

# Startup maintenance (game data refresh first, then self-learning). The
# on/off switches GAMEDATA_AUTO_UPDATE / AI_COACH_AUTO_LEARN are read by
# main.run_startup_maintenance() at startup.
GAMEDATA_STALE_DAYS = env_int("GAMEDATA_STALE_DAYS", 7, minimum=1)
# The running server repeats the game-version check this often, so a patch
# released after the start is picked up without a restart (0 = only at
# startup). Skipped with GAMEDATA_AUTO_UPDATE=0. Every round is three
# requests, one of them to GitHub's API (60 per hour without a token): a
# typo like 0.001 must not turn into a request loop, so anything between 0
# and GAMEDATA_RECHECK_MIN_HOURS counts as that minimum.
GAMEDATA_RECHECK_MIN_HOURS = 0.25


def _recheck_hours() -> float:
    hours = env_float("GAMEDATA_RECHECK_HOURS", 24.0, minimum=0.0)
    if 0.0 < hours < GAMEDATA_RECHECK_MIN_HOURS:
        logger.warning(
            "GAMEDATA_RECHECK_HOURS=%s is below the minimum of %s (0 switches the check off), using %s",
            hours, GAMEDATA_RECHECK_MIN_HOURS, GAMEDATA_RECHECK_MIN_HOURS,
        )
        return GAMEDATA_RECHECK_MIN_HOURS
    return hours


GAMEDATA_RECHECK_HOURS = _recheck_hours()

# Worker threads for blocking request work (analysis, disk I/O, OpenDota
# calls, AI coach steps). asyncio's default of min(32, cpu + 4) is only 6 on
# a 2-vCPU VM, where a few slow OpenDota fetches would stall everything else.
THREADPOOL_SIZE = env_int("THREADPOOL_SIZE", 32, minimum=4)


# ---------------------------------------------------------------------------
# Upload / storage limits
# ---------------------------------------------------------------------------

MAX_UPLOAD_MB = env_int("MAX_UPLOAD_MB", 300, minimum=1)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
# Concurrent uploads being received/stored at once (each may spool up to
# MAX_UPLOAD_MB to the temp dir).
UPLOAD_MAX_CONCURRENCY = env_int("UPLOAD_MAX_CONCURRENCY", 3, minimum=1)
# ... and how many of those one client IP may hold, so a single client
# cannot occupy every slot with slow uploads.
UPLOAD_MAX_PER_CLIENT = env_int("UPLOAD_MAX_PER_CLIENT", 1, minimum=1)
# Seconds a client has to send the whole upload body before it is dropped.
UPLOAD_RECEIVE_TIMEOUT_S = env_int("UPLOAD_RECEIVE_TIMEOUT_S", 900, minimum=10)
# ... and how long the body may stall in between: a connection that died
# without closing must not hold its client's upload slot until that deadline.
UPLOAD_IDLE_TIMEOUT_S = env_int("UPLOAD_IDLE_TIMEOUT_S", 60, minimum=5)
# Uploaded replays kept on disk (content-hash named; oldest pruned first).
REPLAY_MAX_FILES = env_int("REPLAY_MAX_FILES", 50, minimum=1)

# Parsed matches persisted to cache/matches/ (survive restarts / LRU eviction).
MATCH_CACHE_DIR = CACHE_DIR / "matches"
MATCH_CACHE_MAX_FILES = env_int("MATCH_CACHE_MAX_FILES", 200, minimum=1)
# Parsed matches kept in memory (LRU).
MATCH_MEMORY_CACHE_SIZE = env_int("MATCH_MEMORY_CACHE_SIZE", 12, minimum=1)
# /api/fetch downloads an OpenDota match without parsed replay data again
# once our copy is older than this (the copy itself keeps being served).
OPENDOTA_UNPARSED_TTL_S = env_int("OPENDOTA_UNPARSED_TTL_S", 120, minimum=0)


# ---------------------------------------------------------------------------
# AI coach
# ---------------------------------------------------------------------------

# AI coaching result cache
COACH_CACHE_DIR = CACHE_DIR / "coaching"
COACH_MAX_FILES = env_int("COACH_CACHE_MAX_FILES", 2000, minimum=1)
# Global cap on cold LLM-backed coach reports per UTC day (0 = unlimited).
AI_COACH_DAILY_LIMIT = env_int("AI_COACH_DAILY_LIMIT", 200, minimum=0)
AI_COACH_USAGE_FILE = CACHE_DIR / "ai_coach_usage.json"
# Overall pipeline deadline (also read by the AI pipeline itself); the server
# adds a grace period on top before abandoning a generation.
AI_COACH_TIMEOUT_S = env_float("AI_COACH_TIMEOUT_S", 180.0, minimum=1.0)
# Degraded reports (LLM failed, rules used) are only memory-cached this long.
AI_COACH_DEGRADED_TTL_S = env_int("AI_COACH_DEGRADED_TTL_S", 600, minimum=0)


# ---------------------------------------------------------------------------
# Game data directories (auto-updated + editorial)
# ---------------------------------------------------------------------------

DATA_DIR = BASE_DIR / "backend" / "data"
GAMEDATA_DIR = DATA_DIR / "gamedata"
EDITORIAL_DIR = DATA_DIR / "editorial"

# Self-learned knowledge chunks (output of the ai.learning pipeline).
# Canonical path — ai/learning/base.py re-exports it as LEARNED_DIR.
LEARNED_DATA_DIR = DATA_DIR / "learned"

# Ensure directories exist
for _dir in (REPLAY_DIR, COACH_CACHE_DIR, MATCH_CACHE_DIR, GAMEDATA_DIR, EDITORIAL_DIR, LEARNED_DATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
