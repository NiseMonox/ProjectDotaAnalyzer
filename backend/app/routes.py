"""FastAPI routes for the Dota 2 match analysis API."""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from .analyzer import full_analysis
from .config import DEFAULT_REPLAY, REPLAY_DIR
from .models import AnalysisResult
from .parser import ClarityParser, OpenDotaParser, get_parser

router = APIRouter(prefix="/api")

# In-memory cache of parsed matches
_match_cache: dict[str, dict] = {}


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
    }

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
        }

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
    }

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
