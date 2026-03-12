"""FastAPI routes for the Dota 2 match analysis API."""
from __future__ import annotations

import logging
import shutil
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from .analyzer import full_analysis
from .comparison import compare_laning, ComparisonResult
from .config import DEFAULT_REPLAY, REPLAY_DIR
from .models import AnalysisResult, MatchData
from .parser import ClarityParser, OpenDotaParser, get_parser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# In-memory cache of parsed matches (analysis results)
_match_cache: dict[str, dict] = {}

# In-memory storage of parsed MatchData objects (for comparison features)
parsed_matches: dict[str, MatchData] = {}

# In-memory storage of reference replay MatchData (for comparison)
ref_matches: dict[str, MatchData] = {}


class CompareLaningRequest(BaseModel):
    player_match_key: str
    player_slot: int
    ref_match_key: str
    ref_slot: int


class SelfCompareRequest(BaseModel):
    match_key: str
    player_slot: int
    ref_slot: int


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
    if cache_key in _match_cache:
        return _match_cache[cache_key]

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


# ── Laning Comparison Endpoints ────────────────────────────────────────


@router.post("/compare/upload-ref")
async def upload_reference_replay(file: UploadFile = File(...)):
    """Upload a .dem replay file as a reference for laning comparison."""
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
        raise HTTPException(500, f"Failed to parse reference replay: {e}")

    ref_key = f"ref_{file.filename}"
    ref_matches[ref_key] = match_data

    # Build player list with lane info
    players_info = [
        {
            "hero_name": p.hero_name,
            "player_slot": p.player_slot,
            "lane": p.lane,
            "team": p.team,
            "player_name": p.player_name,
        }
        for p in match_data.players
    ]

    return {
        "status": "ok",
        "ref_key": ref_key,
        "match_id": match_data.match_id,
        "parse_source": parser.name(),
        "players": players_info,
    }


@router.post("/compare/laning")
async def compare_laning_endpoint(req: CompareLaningRequest):
    """Compare laning phase between a player and a reference player."""
    # Look up player match data
    if req.player_match_key in parsed_matches:
        player_match = parsed_matches[req.player_match_key]
    else:
        raise HTTPException(404, f"Player match not found: {req.player_match_key}")

    # Look up reference match data (check ref_matches first, then parsed_matches)
    if req.ref_match_key in ref_matches:
        ref_match = ref_matches[req.ref_match_key]
    elif req.ref_match_key in parsed_matches:
        ref_match = parsed_matches[req.ref_match_key]
    else:
        raise HTTPException(404, f"Reference match not found: {req.ref_match_key}")

    try:
        result = compare_laning(player_match, req.player_slot, ref_match, req.ref_slot)
    except Exception as e:
        raise HTTPException(500, f"Comparison failed: {e}")

    return asdict(result)


@router.post("/compare/self")
async def self_compare_endpoint(req: SelfCompareRequest):
    """Compare two players within the same match (e.g., your mid vs enemy mid)."""
    # Look up match data in both stores
    if req.match_key in parsed_matches:
        match_data = parsed_matches[req.match_key]
    elif req.match_key in ref_matches:
        match_data = ref_matches[req.match_key]
    else:
        raise HTTPException(404, f"Match not found: {req.match_key}")

    try:
        result = compare_laning(match_data, req.player_slot, match_data, req.ref_slot)
    except Exception as e:
        raise HTTPException(500, f"Comparison failed: {e}")

    return asdict(result)


# ── AI Coaching Endpoints ──────────────────────────────────────────────


class CoachingRequest(BaseModel):
    player_match_key: str
    player_slot: int
    ref_match_key: str
    ref_slot: int


@router.post("/compare/coaching")
async def get_coaching_report(req: SelfCompareRequest):
    """Generate AI coaching report for a laning comparison."""
    from .ai_coach import is_available, generate_coaching_report, generate_fallback_report
    from .comparison import compare_laning

    # Find match data
    if req.match_key in parsed_matches:
        match_data = parsed_matches[req.match_key]
    elif req.match_key in ref_matches:
        match_data = ref_matches[req.match_key]
    else:
        raise HTTPException(404, f"Match not found: {req.match_key}")

    # Run comparison
    result = compare_laning(match_data, req.player_slot, match_data, req.ref_slot)

    # Generate report
    if is_available():
        report = await generate_coaching_report(result)
    else:
        report = generate_fallback_report(result)

    return report


@router.post("/compare/coaching-cross")
async def get_coaching_report_cross(req: CoachingRequest):
    """Generate AI coaching report comparing across matches."""
    from .ai_coach import is_available, generate_coaching_report, generate_fallback_report
    from .comparison import compare_laning

    # Find player match
    player_match = parsed_matches.get(req.player_match_key)
    if not player_match:
        raise HTTPException(404, f"Player match not found: {req.player_match_key}")

    # Find ref match
    ref_match = ref_matches.get(req.ref_match_key) or parsed_matches.get(req.ref_match_key)
    if not ref_match:
        raise HTTPException(404, f"Reference match not found: {req.ref_match_key}")

    result = compare_laning(player_match, req.player_slot, ref_match, req.ref_slot)

    if is_available():
        report = await generate_coaching_report(result)
    else:
        report = generate_fallback_report(result)

    return report


@router.get("/ai-status")
async def ai_status():
    """Check if AI coaching is available."""
    from .ai_coach import is_available
    return {"available": is_available()}
