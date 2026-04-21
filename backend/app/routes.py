"""FastAPI routes for the Dota 2 match analysis API."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from .analyzer import full_analysis
from .config import COACH_CACHE_DIR, DEFAULT_REPLAY, REPLAY_DIR
from .models import AnalysisResult, MatchData
from .parser import ClarityParser, OpenDotaParser, get_parser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# In-memory cache of parsed matches (analysis results)
_match_cache: dict[str, dict] = {}

# In-memory storage of parsed MatchData objects
parsed_matches: dict[str, MatchData] = {}


# ---------------------------------------------------------------------------
# Persistent coaching result cache
# ---------------------------------------------------------------------------

def _coach_cache_path(match_id: int, player_slot: int, lang: str = "zh") -> Path:
    """Return the JSON cache file path for a coaching result."""
    return COACH_CACHE_DIR / f"{match_id}_{player_slot}_{lang}.json"


def _load_coach_cache(match_id: int, player_slot: int, lang: str = "zh") -> dict | None:
    """Load a cached coaching result from disk, or None if not found."""
    path = _coach_cache_path(match_id, player_slot, lang)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Failed to load coaching cache %s: %s", path, exc)
        return None


def _save_coach_cache(match_id: int, player_slot: int, result: dict, lang: str = "zh") -> None:
    """Save a coaching result to disk as JSON."""
    path = _coach_cache_path(match_id, player_slot, lang)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to save coaching cache %s: %s", path, exc)


@router.post("/upload")
async def upload_replay(file: UploadFile = File(...)):
    """Upload a .dem replay file for analysis."""
    if not file.filename or not file.filename.endswith(".dem"):
        raise HTTPException(400, "Only .dem replay files are supported")

    save_path = REPLAY_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Parse the replay
    parser = get_parser("auto")
    try:
        match_data = await parser.parse(str(save_path))
    except Exception as e:
        raise HTTPException(500, f"Failed to parse replay: {e}")

    # Run analysis
    analysis = full_analysis(match_data)
    file_key = file.filename
    _match_cache[file_key] = {
        "analysis": analysis,
        "parse_source": parser.name(),
        "_match_key": file_key,
    }
    parsed_matches[file_key] = match_data

    return {
        "status": "ok",
        "file_key": file_key,
        "match_id": analysis["match_overview"]["match_id"],
        "parse_source": parser.name(),
    }


@router.get("/demo")
async def get_demo_analysis():
    """Get analysis using default test replay.

    Priority: ClarityParser (real .dem) → OpenDota API → MockParser
    """
    if "demo" not in _match_cache:
        parse_source = "mock"
        match_data = None

        # Try real replay with ClarityParser
        if DEFAULT_REPLAY.exists():
            try:
                clarity = ClarityParser()
                clarity._check_requirements()
                match_data = await clarity.parse(str(DEFAULT_REPLAY))
                parse_source = "clarity"
            except Exception:
                pass

        # Fallback: fetch from OpenDota using match ID from filename
        if match_data is None and DEFAULT_REPLAY.exists():
            try:
                match_id = DEFAULT_REPLAY.stem  # e.g. "8713267082"
                parser = OpenDotaParser()
                match_data = await parser.parse(match_id)
                parse_source = "opendota"
            except Exception:
                pass

        # Final fallback: mock data
        if match_data is None:
            parser = get_parser("mock")
            match_data = await parser.parse("")
            parse_source = "mock"

        analysis = full_analysis(match_data)
        _match_cache["demo"] = {
            "analysis": analysis,
            "parse_source": parse_source,
            "_match_key": "demo",
        }
        parsed_matches["demo"] = match_data

    return _match_cache["demo"]


@router.get("/fetch/{match_id}")
async def fetch_match_by_id(match_id: int):
    """Fetch match data from OpenDota API by match ID."""
    cache_key = f"opendota_{match_id}"
    cached = _match_cache.get(cache_key)
    if cached and cached.get("has_parsed_data"):
        return cached

    parser = OpenDotaParser()
    try:
        match_data = await parser.parse(str(match_id))
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch from OpenDota: {e}")

    analysis = full_analysis(match_data)
    has_parsed = bool(match_data.gold_xp_timeline)

    _match_cache[cache_key] = {
        "analysis": analysis,
        "parse_source": "opendota",
        "has_parsed_data": has_parsed,
        "_match_key": cache_key,
    }
    parsed_matches[cache_key] = match_data

    return _match_cache[cache_key]


@router.post("/request-parse/{match_id}")
async def request_parse(match_id: int):
    """Request OpenDota to parse a match replay."""
    parser = OpenDotaParser()
    try:
        result = parser.request_parse(match_id)
        # Clear cache so next fetch gets fresh data
        cache_key = f"opendota_{match_id}"
        _match_cache.pop(cache_key, None)
        return {"status": "ok", "job": result.get("job", {})}
    except Exception as e:
        raise HTTPException(502, f"Failed to request parse: {e}")


@router.get("/parse-local/{filename}")
async def parse_local_replay(filename: str):
    """Parse a .dem file already present in the replays/ directory.

    This avoids the need to upload large replay files via HTTP.
    """
    if not filename.endswith(".dem"):
        raise HTTPException(400, "Only .dem replay files are supported")

    # Sanitize: prevent path traversal
    safe_name = Path(filename).name
    replay_path = REPLAY_DIR / safe_name

    if not replay_path.exists():
        raise HTTPException(404, f"Replay file not found: {safe_name} (looked in {REPLAY_DIR})")

    cache_key = f"local_{safe_name}"
    if cache_key in _match_cache:
        return _match_cache[cache_key]

    # Try ClarityParser first, fall back to OpenDota by match ID
    match_data = None
    parse_source = "unknown"

    try:
        clarity = ClarityParser()
        clarity._check_requirements()
        match_data = await clarity.parse(str(replay_path))
        parse_source = "clarity"
    except Exception as e:
        logger.warning(f"Clarity parse failed for {safe_name}: {e}")

    if match_data is None:
        # Try OpenDota using filename stem as match ID
        try:
            match_id = replay_path.stem
            int(match_id)  # Validate it's numeric
            parser = OpenDotaParser()
            match_data = await parser.parse(match_id)
            parse_source = "opendota"
        except (ValueError, Exception) as e:
            raise HTTPException(500, f"Failed to parse replay {safe_name}: {e}")

    analysis = full_analysis(match_data)
    _match_cache[cache_key] = {
        "analysis": analysis,
        "parse_source": parse_source,
        "filename": safe_name,
        "_match_key": cache_key,
    }
    parsed_matches[cache_key] = match_data

    return _match_cache[cache_key]


@router.get("/match/{file_key}")
async def get_match_analysis(file_key: str):
    """Get analysis results for a previously uploaded match."""
    if file_key not in _match_cache:
        raise HTTPException(404, "Match not found. Please upload the replay first.")

    return _match_cache[file_key]


@router.get("/match/{file_key}/ai-coach/{player_slot}")
async def get_ai_coaching(file_key: str, player_slot: int, lang: str = Query("zh")):
    """Generate AI coaching report for a specific player."""
    # Validate lang
    if lang not in ("zh", "en"):
        lang = "zh"

    # Check in-memory cache first
    cache_key = f"{file_key}_coach_{player_slot}_{lang}"
    if cache_key in _match_cache:
        return _match_cache[cache_key]

    if file_key not in parsed_matches:
        raise HTTPException(404, "Match not found. Please upload the replay first.")

    match_data = parsed_matches[file_key]
    match_id = match_data.match_id

    # Check persistent disk cache (survives server restarts)
    cached = _load_coach_cache(match_id, player_slot, lang)
    if cached is not None:
        _match_cache[cache_key] = cached
        return cached

    try:
        from .ai.coach import generate_coaching_report

        report = await generate_coaching_report(match_data, player_slot, lang=lang)
        result = report.to_dict()
        _match_cache[cache_key] = result
        _save_coach_cache(match_id, player_slot, result, lang)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("AI coaching failed for %s slot %d", file_key, player_slot)
        raise HTTPException(500, f"AI coaching analysis failed: {e}")


@router.get("/match/{file_key}/{section}")
async def get_match_section(file_key: str, section: str):
    """Get a specific analysis section for a match."""
    if file_key not in _match_cache:
        raise HTTPException(404, "Match not found")

    analysis = _match_cache[file_key]["analysis"]
    if section not in analysis:
        valid = list(analysis.keys())
        raise HTTPException(400, f"Invalid section. Valid: {valid}")

    return {section: analysis[section]}


# ---------------------------------------------------------------------------
# Admin endpoints: game data management
# ---------------------------------------------------------------------------

@router.post("/admin/update-gamedata")
async def update_gamedata(force: bool = False):
    """Fetch latest game data from OpenDota API.

    This updates hero, ability, and item stats (Layer A).
    Editorial coaching tips (Layer B) are never overwritten.
    """
    import asyncio

    try:
        from .ai.data_manager import fetch_and_save_gamedata
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: fetch_and_save_gamedata(force=force)
        )
        return result
    except Exception as e:
        logger.exception("Game data update failed")
        raise HTTPException(500, f"Update failed: {e}")


@router.get("/admin/gamedata-status")
async def gamedata_status():
    """Return current game data status (patch version, freshness, coverage)."""
    try:
        from .ai.data_manager import get_meta
        return get_meta()
    except ImportError:
        return {"status": "module_unavailable", "patch_version": "7.37"}
