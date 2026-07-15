"""FastAPI routes for the Dota 2 match analysis API."""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections import OrderedDict
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from .analyzer import full_analysis
from .config import COACH_CACHE_DIR, DEFAULT_REPLAY, REPLAY_DIR
from .models import MatchData
from .parser import ClarityParser, OpenDotaParser, get_parser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Max entries kept in each in-memory cache (LRU eviction beyond this)
_CACHE_MAX_ENTRIES = 12

# In-memory cache of parsed matches (analysis results), LRU-evicted
_match_cache: OrderedDict[str, dict] = OrderedDict()

# In-memory storage of parsed MatchData objects, LRU-evicted
parsed_matches: OrderedDict[str, MatchData] = OrderedDict()

# In-memory AI coach reports, LRU-evicted. Kept separate from _match_cache so
# per-(player, lang) coach fetches cannot evict a match's analysis entry.
_coach_report_cache: OrderedDict[str, dict] = OrderedDict()


def _cache_get(cache: OrderedDict, key: str):
    """Return the cached value (marking it most-recently-used), or None."""
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_put(cache: OrderedDict, key: str, value) -> None:
    """Insert a value, evicting the least-recently-used entries when full."""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_MAX_ENTRIES:
        cache.popitem(last=False)


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
    # Sanitize: strip client-supplied directory components (path traversal)
    safe_name = Path(file.filename).name if file.filename else ""
    if not safe_name or not safe_name.endswith(".dem"):
        raise HTTPException(400, "Only .dem replay files are supported")

    save_path = REPLAY_DIR / safe_name
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
    file_key = safe_name
    _cache_put(_match_cache, file_key, {
        "analysis": analysis,
        "parse_source": parser.name(),
        "_match_key": file_key,
    })
    _cache_put(parsed_matches, file_key, match_data)

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
    cached = _cache_get(_match_cache, "demo")
    if cached is None:
        parse_source = "mock"
        match_data = None

        # Try real replay with ClarityParser
        if DEFAULT_REPLAY.exists():
            try:
                clarity = ClarityParser()
                await asyncio.to_thread(clarity._check_requirements)
                match_data = await clarity.parse(str(DEFAULT_REPLAY))
                parse_source = "clarity"
            except Exception as exc:
                logger.warning("Clarity parse failed for default replay %s: %s", DEFAULT_REPLAY, exc)

        # Fallback: fetch from OpenDota using match ID from filename
        if match_data is None and DEFAULT_REPLAY.exists():
            try:
                match_id = DEFAULT_REPLAY.stem  # e.g. "8713267082"
                parser = OpenDotaParser()
                match_data = await parser.parse(match_id)
                parse_source = "opendota"
            except Exception as exc:
                logger.warning("OpenDota fallback failed for match %s: %s", DEFAULT_REPLAY.stem, exc)

        # Final fallback: mock data
        if match_data is None:
            logger.warning("Falling back to mock data for /api/demo")
            parser = get_parser("mock")
            match_data = await parser.parse("")
            parse_source = "mock"

        analysis = full_analysis(match_data)
        cached = {
            "analysis": analysis,
            "parse_source": parse_source,
            "_match_key": "demo",
        }
        _cache_put(_match_cache, "demo", cached)
        _cache_put(parsed_matches, "demo", match_data)

    return cached


@router.get("/fetch/{match_id}")
async def fetch_match_by_id(match_id: int):
    """Fetch match data from OpenDota API by match ID."""
    cache_key = f"opendota_{match_id}"
    cached = _cache_get(_match_cache, cache_key)
    if cached and cached.get("has_parsed_data"):
        return cached

    parser = OpenDotaParser()
    try:
        match_data = await parser.parse(str(match_id))
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch from OpenDota: {e}")

    analysis = full_analysis(match_data)
    has_parsed = bool(match_data.gold_xp_timeline)

    result = {
        "analysis": analysis,
        "parse_source": "opendota",
        "has_parsed_data": has_parsed,
        "_match_key": cache_key,
    }
    _cache_put(_match_cache, cache_key, result)
    _cache_put(parsed_matches, cache_key, match_data)

    return result


@router.post("/request-parse/{match_id}")
async def request_parse(match_id: int):
    """Request OpenDota to parse a match replay."""
    parser = OpenDotaParser()
    try:
        # Synchronous urllib call runs in a worker thread (keeps loop free)
        result = await asyncio.to_thread(parser.request_parse, match_id)
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
    cached = _cache_get(_match_cache, cache_key)
    if cached is not None:
        return cached

    # Try ClarityParser first, fall back to OpenDota by match ID
    match_data = None
    parse_source = "unknown"

    try:
        clarity = ClarityParser()
        await asyncio.to_thread(clarity._check_requirements)
        match_data = await clarity.parse(str(replay_path))
        parse_source = "clarity"
    except Exception as e:
        logger.warning(f"Clarity parse failed for {safe_name}: {e}")

    if match_data is None:
        # Try OpenDota using filename stem as match ID
        match_id = replay_path.stem
        try:
            int(match_id)  # Validate it's numeric
        except ValueError:
            raise HTTPException(
                400,
                f"Cannot fall back to OpenDota: filename {safe_name} is not a numeric match ID",
            )
        try:
            parser = OpenDotaParser()
            match_data = await parser.parse(match_id)
            parse_source = "opendota"
        except Exception as e:
            raise HTTPException(500, f"Failed to parse replay {safe_name}: {e}")

    analysis = full_analysis(match_data)
    result = {
        "analysis": analysis,
        "parse_source": parse_source,
        "filename": safe_name,
        "_match_key": cache_key,
    }
    _cache_put(_match_cache, cache_key, result)
    _cache_put(parsed_matches, cache_key, match_data)

    return result


@router.get("/match/{file_key}")
async def get_match_analysis(file_key: str):
    """Get analysis results for a previously uploaded match."""
    cached = _cache_get(_match_cache, file_key)
    if cached is None:
        raise HTTPException(404, "Match not found. Please upload the replay first.")

    return cached


@router.get("/match/{file_key}/ai-coach/{player_slot}")
async def get_ai_coaching(file_key: str, player_slot: int, lang: str = Query("zh")):
    """Generate AI coaching report for a specific player."""
    # Validate lang
    if lang not in ("zh", "en"):
        lang = "zh"

    # Check in-memory cache first
    cache_key = f"{file_key}_coach_{player_slot}_{lang}"
    coach_cached = _cache_get(_coach_report_cache, cache_key)
    if coach_cached is not None:
        return coach_cached

    match_data = _cache_get(parsed_matches, file_key)
    if match_data is None:
        raise HTTPException(404, "Match not found. Please upload the replay first.")

    match_id = match_data.match_id

    # Check persistent disk cache (survives server restarts)
    cached = _load_coach_cache(match_id, player_slot, lang)
    if cached is not None:
        _cache_put(_coach_report_cache, cache_key, cached)
        return cached

    try:
        from .ai.coach import generate_coaching_report

        report = await generate_coaching_report(match_data, player_slot, lang=lang)
        result = report.to_dict()
        _cache_put(_coach_report_cache, cache_key, result)
        _save_coach_cache(match_id, player_slot, result, lang)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("AI coaching failed for %s slot %d", file_key, player_slot)
        raise HTTPException(500, f"AI coaching analysis failed: {e}")


@router.get("/match/{file_key}/ai-coach-stream/{player_slot}")
async def stream_ai_coaching(file_key: str, player_slot: int, lang: str = Query("zh")):
    """Stream AI coaching report progress via Server-Sent Events.

    Each event is `data: <json>\\n\\n` with one of these shapes:
    - {"type": "progress", "stage": "evaluate", "current": 3, "total": 8}
    - {"type": "done", "data": <full report dict>}
    - {"type": "error", "message": "..."}
    """
    if lang not in ("zh", "en"):
        lang = "zh"

    cache_key = f"{file_key}_coach_{player_slot}_{lang}"

    async def event_generator():
        def _pack(payload: dict) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # In-memory cache hit → emit one done event and exit.
        coach_cached = _cache_get(_coach_report_cache, cache_key)
        if coach_cached is not None:
            yield _pack({"type": "done", "data": coach_cached})
            return

        match_data = _cache_get(parsed_matches, file_key)
        if match_data is None:
            yield _pack({"type": "error", "message": "Match not found. Please upload the replay first."})
            return

        # Disk cache hit → also emit single done event.
        cached = _load_coach_cache(match_data.match_id, player_slot, lang)
        if cached is not None:
            _cache_put(_coach_report_cache, cache_key, cached)
            yield _pack({"type": "done", "data": cached})
            return

        # Cold path: run generate_coaching_report with a progress callback that
        # pushes events into a queue, drained by this generator.
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        async def progress_cb(payload: dict) -> None:
            await queue.put({"type": "progress", **payload})

        async def run_report() -> None:
            try:
                from .ai.coach import generate_coaching_report

                report = await generate_coaching_report(
                    match_data, player_slot, lang=lang,
                    progress_callback=progress_cb,
                )
                result = report.to_dict()
                _cache_put(_coach_report_cache, cache_key, result)
                _save_coach_cache(match_data.match_id, player_slot, result, lang)
                await queue.put({"type": "done", "data": result})
            except Exception as exc:
                logger.exception("AI coach stream failed for %s slot %d", file_key, player_slot)
                await queue.put({"type": "error", "message": f"AI coaching analysis failed: {exc}"})
            finally:
                await queue.put(SENTINEL)

        worker = asyncio.create_task(run_report())

        # Optimistic kick-off event so the client UI moves immediately.
        yield _pack({"type": "progress", "stage": "starting", "current": 0, "total": 1})

        try:
            while True:
                item = await queue.get()
                if item is SENTINEL:
                    break
                yield _pack(item)
        finally:
            if not worker.done():
                worker.cancel()
            try:
                await worker
            except (asyncio.CancelledError, Exception):
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disables nginx buffering when fronted later
        },
    )


@router.get("/match/{file_key}/{section}")
async def get_match_section(file_key: str, section: str):
    """Get a specific analysis section for a match."""
    cached = _cache_get(_match_cache, file_key)
    if cached is None:
        raise HTTPException(404, "Match not found")

    analysis = cached["analysis"]
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
