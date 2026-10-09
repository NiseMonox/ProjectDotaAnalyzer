"""FastAPI routes for the Dota 2 match analysis API.

Match results live in three tiers: an in-memory LRU (fast path), a bounded
disk store (``cache/matches``, survives eviction and restarts) and the
original source (replay file / OpenDota). Expensive work — replay parsing,
OpenDota fetches and AI-coach generation — is de-duplicated per key so
concurrent requests share one computation.

Errors returned to clients are short, localized (``?lang=`` or
Accept-Language) and carry a stable ``code``; details are only logged.
"""
from __future__ import annotations

import asyncio
import functools
import gzip
import json
import logging
import math
import os
import time
import urllib.error
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response, StreamingResponse
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import config, storage
from . import parser as parser_mod
from .analyzer import full_analysis
from .models import MatchData
from .security import (
    ApiError,
    ai_coach_daily_quota,
    enforce_rate_limit,
    rate_limited_error,
    request_lang,
    require_admin,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# In-memory match cache (LRU) + persistent disk store
# ---------------------------------------------------------------------------

@dataclass
class _MatchRecord:
    entry: dict          # API payload: {"analysis", "parse_source", "_match_key", ...}
    match: MatchData
    expires_at: float | None = None  # monotonic deadline for short-lived entries
    body: bytes = b""       # ``entry`` as JSON, serialized once off the event loop
    body_gzip: bytes = b""  # gzip of ``body`` (empty when too small to bother)


_matches: OrderedDict[str, _MatchRecord] = OrderedDict()

match_store = storage.MatchStore(config.MATCH_CACHE_DIR, lambda: config.MATCH_CACHE_MAX_FILES)


def _memory_get(key: str) -> _MatchRecord | None:
    record = _matches.get(key)
    if record is None:
        return None
    if record.expires_at is not None and time.monotonic() > record.expires_at:
        _matches.pop(key, None)
        return None
    _matches.move_to_end(key)
    return record


def _memory_put(key: str, record: _MatchRecord) -> None:
    _matches[key] = record
    _matches.move_to_end(key)
    while len(_matches) > config.MATCH_MEMORY_CACHE_SIZE:
        _matches.popitem(last=False)


def _forget_match(key: str) -> None:
    _matches.pop(key, None)


def _finite(value: Any) -> Any:
    """Replace NaN/Infinity (invalid JSON for browsers) with None."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):  # pydantic models
        return value.model_dump(mode="json")
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    return str(value)


def _json_bytes(payload: Any) -> bytes:
    """Compact UTF-8 JSON like FastAPI's, but tolerant of NaN/odd types."""
    options = {"ensure_ascii": False, "allow_nan": False, "separators": (",", ":"), "default": _json_default}
    try:
        text = json.dumps(payload, **options)
    except ValueError:
        text = json.dumps(_finite(payload), **options)
    return text.encode("utf-8")


def _build_record(key: str, match: MatchData, parse_source: str, extra: dict | None) -> _MatchRecord:
    """Analyze and pre-serialize a match (CPU-bound: run in a worker thread)."""
    entry = {
        "analysis": full_analysis(match),
        "parse_source": parse_source,
        **(extra or {}),
        "_match_key": key,
    }
    body = _json_bytes(entry)
    body_gzip = gzip.compress(body, compresslevel=6) if len(body) >= 1024 else b""
    return _MatchRecord(entry, match, body=body, body_gzip=body_gzip)


def _match_response(request: Request, record: _MatchRecord) -> Response:
    """Serve a match payload from its cached JSON (gzip when accepted)."""
    headers = {"Vary": "Accept-Encoding"}
    if record.body_gzip and "gzip" in request.headers.get("accept-encoding", ""):
        headers["Content-Encoding"] = "gzip"
        return Response(record.body_gzip, media_type="application/json", headers=headers)
    return Response(record.body, media_type="application/json", headers=headers)


async def _store_match(
    key: str,
    match: MatchData,
    parse_source: str,
    extra: dict | None = None,
    *,
    persist: bool = True,
    ttl_s: float | None = None,
) -> _MatchRecord:
    """Analyze ``match`` off the event loop, cache it and optionally persist it."""
    record = await asyncio.to_thread(_build_record, key, match, parse_source, extra)
    record.expires_at = time.monotonic() + ttl_s if ttl_s is not None else None
    _memory_put(key, record)
    if persist:
        try:
            await asyncio.to_thread(
                match_store.save, key, match, {"parse_source": parse_source, "extra": extra or {}},
            )
        except Exception as exc:  # cache only; the response is still valid
            logger.warning("Failed to persist match %s: %s", key, exc)
    return record


async def _load_stored_match(key: str) -> _MatchRecord | None:
    loaded = await asyncio.to_thread(match_store.load, key)
    if loaded is None:
        return None
    match, meta = loaded
    record = await asyncio.to_thread(
        _build_record, key, match, str(meta.get("parse_source") or "unknown"), meta.get("extra"),
    )
    _memory_put(key, record)
    logger.info("Reloaded match %s from disk store", key)
    return record


async def _get_record(key: str) -> _MatchRecord | None:
    """Look a match up in memory, then in the disk store."""
    record = _memory_get(key)
    if record is not None:
        return record
    if not match_store.path_for(key).is_file():  # cheap miss, no thread hop
        return None
    return await _singleflight(f"load:{key}", lambda: _load_stored_match(key))


# ---------------------------------------------------------------------------
# In-flight de-duplication
# ---------------------------------------------------------------------------

_inflight: dict[str, asyncio.Task] = {}


def _inflight_done(key: str, task: asyncio.Task) -> None:
    if _inflight.get(key) is task:
        del _inflight[key]
    if not task.cancelled():
        task.exception()  # mark retrieved; awaiters handle/log it


async def _singleflight(key: str, factory: Callable[[], Awaitable[Any]]) -> Any:
    """Run ``factory()`` once per key; concurrent callers await the same task.

    The task is shielded: a caller that goes away (client disconnect) does
    not cancel work other callers are waiting for.
    """
    loop = asyncio.get_running_loop()
    task = _inflight.get(key)
    if task is None or task.done() or task.get_loop() is not loop:
        task = loop.create_task(factory())
        _inflight[key] = task
        task.add_done_callback(functools.partial(_inflight_done, key))
    return await asyncio.shield(task)


# ---------------------------------------------------------------------------
# Parser access and error mapping (parser.py contract C2)
# ---------------------------------------------------------------------------

class _ParserUnavailable(RuntimeError):
    """Local stand-in when parser.ParserUnavailableError is not defined."""


def _parser_exc(name: str, *fallbacks: type) -> tuple[type, ...]:
    """Exception classes exported by parser.py (looked up lazily)."""
    cls = getattr(parser_mod, name, None)
    found = (cls,) if isinstance(cls, type) and issubclass(cls, BaseException) else ()
    return found + tuple(fallbacks)


async def _get_replay_parser() -> parser_mod.ReplayParser:
    """Return a parser for .dem files or raise ParserUnavailableError.

    Never silently falls back to mock data for user uploads.
    """
    check = getattr(parser_mod, "check_clarity_available", None)
    if check is not None:
        if not await check():
            unavailable = getattr(parser_mod, "ParserUnavailableError", _ParserUnavailable)
            raise unavailable("Clarity parser (Java/jar) is not available")
        return parser_mod.ClarityParser()
    parser = await asyncio.to_thread(parser_mod.get_parser, "auto")
    if parser.name() == "mock":
        raise _ParserUnavailable("Clarity parser (Java/jar) is not available")
    return parser


def _map_error(exc: BaseException, lang: str, context: str) -> ApiError:
    """Translate parser / OpenDota failures into short user-safe API errors.

    ``context`` is "replay" or "opendota" and decides how unknown errors
    are reported. Details are logged, never returned.
    """
    if isinstance(exc, ApiError):
        return exc
    if isinstance(exc, _parser_exc("ParserUnavailableError", _ParserUnavailable)):
        logger.warning("Replay parser unavailable: %s", exc)
        return ApiError(503, "parser_unavailable", lang)
    if isinstance(exc, _parser_exc("ReplayParseError")):
        logger.warning("Replay parse failed: %s", exc)
        return ApiError(422, "replay_parse_failed", lang)
    if isinstance(exc, _parser_exc("OpenDotaError")):
        status = getattr(exc, "status", None)
        logger.warning("OpenDota request failed (status=%s): %s", status, exc)
        if status == 404:
            return ApiError(404, "opendota_not_found", lang)
        return ApiError(502, "opendota_unavailable", lang)
    if isinstance(exc, urllib.error.HTTPError):
        logger.warning("OpenDota HTTP error %s: %s", exc.code, exc.reason)
        if exc.code == 404:
            return ApiError(404, "opendota_not_found", lang)
        return ApiError(502, "opendota_unavailable", lang)
    if context == "opendota":
        logger.warning("OpenDota request failed: %r", exc)
        return ApiError(502, "opendota_unavailable", lang)
    logger.error("Unexpected %s failure", context, exc_info=exc)
    return ApiError(500, "internal_error", lang)


def _parse_match_id(raw: str, lang: str) -> int:
    raw = raw.strip()
    if not raw.isdigit() or len(raw) > 19 or int(raw) <= 0:
        raise ApiError(400, "invalid_match_id", lang)
    return int(raw)


# ---------------------------------------------------------------------------
# Health / capabilities
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok"}


@router.get("/capabilities")
async def capabilities():
    """What the server can do right now, so the UI can adapt before uploading.

    ``ai_coach_daily_remaining`` is None when there is no daily cap.
    """
    try:
        await _get_replay_parser()
        replay_parser = True
    except Exception:
        replay_parser = False
    limit = config.AI_COACH_DAILY_LIMIT
    return {
        "replay_parser": replay_parser,
        "max_upload_mb": config.MAX_UPLOAD_MB,
        "ai_coach_llm": _llm_configured(),
        "ai_coach_daily_remaining": max(0, limit - ai_coach_daily_quota.used()) if limit > 0 else None,
    }


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

_uploads_in_progress = 0

_UPLOAD_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {"file": {"type": "string", "format": "binary"}},
                },
            },
        },
    },
}


async def _receive_upload(request: Request, lang: str) -> storage.StoredReplay:
    """Parse the multipart body and store the replay under its content hash."""
    try:
        form = await request.form(max_files=1, max_fields=8)
    except ApiError:
        raise
    except StarletteHTTPException as exc:
        logger.info("Rejected malformed upload: %s", exc.detail)
        raise ApiError(400, "bad_request", lang) from None
    try:
        upload = form.get("file")
        if not isinstance(upload, StarletteUploadFile):
            raise ApiError(400, "upload_missing_file", lang)
        if not PurePath(upload.filename or "").name.lower().endswith(".dem"):
            raise ApiError(400, "upload_bad_extension", lang)
        try:
            return await asyncio.to_thread(
                storage.store_upload, upload.file, config.REPLAY_DIR, config.MAX_UPLOAD_BYTES,
            )
        except storage.UploadRejected as exc:
            raise ApiError(exc.status, exc.code, lang, limit_mb=config.MAX_UPLOAD_MB) from None
        except OSError:
            logger.exception("Failed to store uploaded replay")
            raise ApiError(500, "upload_failed", lang) from None
    finally:
        await form.close()


async def _parse_uploaded(key: str, path: Path) -> _MatchRecord:
    try:
        parser = await _get_replay_parser()
        match_data = await parser.parse(str(path))
    except Exception:
        # Unparseable (or unparseable here) — don't keep the file around.
        await asyncio.to_thread(storage.delete_file, path)
        raise
    return await _store_match(key, match_data, parser.name())


@router.post("/upload", openapi_extra=_UPLOAD_OPENAPI)
async def upload_replay(request: Request):
    """Upload a .dem replay file (multipart field ``file``) for analysis.

    The replay is stored under its content hash, so re-uploading the same
    file is answered from cache without parsing again.
    """
    global _uploads_in_progress
    lang = request_lang(request)
    enforce_rate_limit(request, "upload")
    try:
        await _get_replay_parser()  # fail fast before receiving a large body
    except Exception as exc:
        raise _map_error(exc, lang, "replay") from None
    if _uploads_in_progress >= config.UPLOAD_MAX_CONCURRENCY:
        raise ApiError(503, "server_busy", lang, headers={"Retry-After": "30"}, extra={"retry_after": 30})

    _uploads_in_progress += 1
    try:
        stored = await _receive_upload(request, lang)
    finally:
        _uploads_in_progress -= 1

    file_key = f"replay_{stored.digest}"
    record = await _get_record(file_key)
    cached = record is not None
    if record is None:
        try:
            record = await _singleflight(f"parse:{file_key}", lambda: _parse_uploaded(file_key, stored.path))
        except Exception as exc:
            raise _map_error(exc, lang, "replay") from None

    try:
        await asyncio.to_thread(
            storage.prune_replays, config.REPLAY_DIR, config.REPLAY_MAX_FILES, {stored.path},
        )
    except Exception as exc:
        logger.warning("Replay pruning failed: %s", exc)

    return {
        "status": "ok",
        "file_key": file_key,
        "match_id": record.match.match_id,
        "parse_source": record.entry.get("parse_source"),
        "cached": cached,
        "_match_key": file_key,
    }


# ---------------------------------------------------------------------------
# Demo / OpenDota / local replays
# ---------------------------------------------------------------------------

async def _build_demo() -> _MatchRecord:
    """Default replay via Clarity → OpenDota by match ID → mock data."""
    match_data: MatchData | None = None
    parse_source = "mock"
    demo = config.DEFAULT_REPLAY

    if demo.exists():
        try:
            parser = await _get_replay_parser()
            match_data = await parser.parse(str(demo))
            parse_source = parser.name()
        except Exception as exc:
            logger.warning("Clarity parse failed for default replay %s: %s", demo.name, exc)

    if match_data is None and demo.exists() and demo.stem.isdigit():
        try:
            match_data = await parser_mod.OpenDotaParser().parse(demo.stem)
            parse_source = "opendota"
        except Exception as exc:
            logger.warning("OpenDota fallback failed for demo match %s: %s", demo.stem, exc)

    if match_data is None:
        logger.warning("Falling back to mock data for /api/demo")
        match_data = await parser_mod.get_parser("mock").parse("")
        parse_source = "mock"

    # Mock data is regenerated on demand; only persist real results.
    return await _store_match("demo", match_data, parse_source, persist=parse_source != "mock")


@router.get("/demo")
async def get_demo_analysis(request: Request):
    """Get analysis for the default demo match (real replay if available)."""
    record = await _get_record("demo")
    if record is None:
        record = await _singleflight("parse:demo", _build_demo)
    return _match_response(request, record)


async def _fetch_opendota(key: str, match_id: int) -> _MatchRecord:
    match_data = await parser_mod.OpenDotaParser().parse(str(match_id))
    has_parsed = bool(match_data.gold_xp_timeline)
    # Unparsed (basic) results are short-lived so a later fetch picks up
    # OpenDota's parsed data; parsed results are persisted.
    return await _store_match(
        key, match_data, "opendota", {"has_parsed_data": has_parsed},
        persist=has_parsed,
        ttl_s=None if has_parsed else float(config.OPENDOTA_UNPARSED_TTL_S),
    )


@router.get("/fetch/{match_id}")
async def fetch_match_by_id(request: Request, match_id: str):
    """Fetch match data from the OpenDota API by match ID."""
    lang = request_lang(request)
    mid = _parse_match_id(match_id, lang)
    key = f"opendota_{mid}"
    record = await _get_record(key)
    if record is None:
        enforce_rate_limit(request, "fetch")
        try:
            record = await _singleflight(f"parse:{key}", lambda: _fetch_opendota(key, mid))
        except Exception as exc:
            raise _map_error(exc, lang, "opendota") from None
    return _match_response(request, record)


@router.post("/request-parse/{match_id}")
async def request_parse(request: Request, match_id: str):
    """Ask OpenDota to parse a match replay; drops our cached copy of it."""
    lang = request_lang(request)
    mid = _parse_match_id(match_id, lang)
    enforce_rate_limit(request, "request_parse")
    try:
        result = await asyncio.to_thread(parser_mod.OpenDotaParser().request_parse, mid)
    except Exception as exc:
        raise _map_error(exc, lang, "opendota") from None
    key = f"opendota_{mid}"
    _forget_match(key)
    await asyncio.to_thread(match_store.delete, key)
    job = result.get("job", {}) if isinstance(result, dict) else {}
    return {"status": "ok", "job": job}


class _OpenDotaFallbackFailed(Exception):
    """Wraps the OpenDota error after a local replay could not be parsed."""

    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.cause = cause


async def _parse_local(key: str, replay_path: Path, safe_name: str) -> _MatchRecord:
    try:
        parser = await _get_replay_parser()
        match_data = await parser.parse(str(replay_path))
        return await _store_match(key, match_data, parser.name(), {"filename": safe_name})
    except Exception as exc:
        if not replay_path.stem.isdigit():
            raise
        logger.warning("Clarity parse failed for %s (%s); trying OpenDota", safe_name, exc)
    try:
        match_data = await parser_mod.OpenDotaParser().parse(replay_path.stem)
    except Exception as exc:
        raise _OpenDotaFallbackFailed(exc) from exc
    return await _store_match(key, match_data, "opendota", {"filename": safe_name})


@router.get("/parse-local/{filename}", dependencies=[Depends(require_admin)])
async def parse_local_replay(request: Request, filename: str):
    """Parse a .dem file already present in replays/ (admin only).

    Avoids uploading large replay files via HTTP.
    """
    lang = request_lang(request)
    safe_name = PurePath(filename).name
    if safe_name != filename or safe_name.startswith(".") or not safe_name.lower().endswith(".dem"):
        raise ApiError(400, "invalid_filename", lang)
    replay_path = config.REPLAY_DIR / safe_name
    if not replay_path.is_file():
        raise ApiError(404, "replay_not_found", lang)

    key = f"local_{safe_name}"
    record = await _get_record(key)
    if record is None:
        try:
            record = await _singleflight(f"parse:{key}", lambda: _parse_local(key, replay_path, safe_name))
        except _OpenDotaFallbackFailed as exc:
            raise _map_error(exc.cause, lang, "opendota") from None
        except Exception as exc:
            raise _map_error(exc, lang, "replay") from None
    return _match_response(request, record)


@router.get("/match/{file_key}")
async def get_match_analysis(request: Request, file_key: str):
    """Get analysis results for a previously uploaded / fetched match."""
    record = await _get_record(file_key)
    if record is None:
        raise ApiError(404, "match_not_found", request_lang(request))
    return _match_response(request, record)


# ---------------------------------------------------------------------------
# AI coach
# ---------------------------------------------------------------------------

# Seconds between SSE keep-alive comments while a report is generating.
SSE_HEARTBEAT_S = 15.0
# Extra time on top of AI_COACH_TIMEOUT_S before the server gives up.
_COACH_TIMEOUT_GRACE_S = 60.0
_COACH_MEMORY_SIZE = 32

_coach_report_cache: OrderedDict[str, tuple[dict, float | None]] = OrderedDict()


class _CoachJob:
    """One in-flight report generation shared by all waiting requests."""

    def __init__(self, coach_id: str) -> None:
        self.coach_id = coach_id
        self.listeners: set[asyncio.Queue] = set()
        self.last_progress: dict | None = None
        self.task: asyncio.Task | None = None

    async def on_progress(self, payload: dict) -> None:
        event = {"type": "progress", **payload}
        self.last_progress = event
        for queue in tuple(self.listeners):
            queue.put_nowait(event)

    def finish(self) -> None:
        for queue in tuple(self.listeners):
            queue.put_nowait(_JOB_FINISHED)


_JOB_FINISHED = object()
_coach_jobs: dict[str, _CoachJob] = {}


def _coach_lang(lang: str | None) -> str:
    return lang if lang in ("zh", "en") else "zh"


def _coach_schema_version() -> int:
    try:
        from .ai import coach
    except Exception:  # pragma: no cover - AI package broken
        return 1
    try:
        return int(getattr(coach, "REPORT_SCHEMA_VERSION", 1))
    except (TypeError, ValueError):
        return 1


def _coach_id(record: _MatchRecord, player_slot: int, lang: str) -> str:
    """Identity of a report: same match data + player + language."""
    source = str(record.entry.get("parse_source") or "unknown")
    if source == "opendota" and not record.entry.get("has_parsed_data", True):
        source = "opendota-basic"
    return f"{record.match.match_id}_{source}_{player_slot}_{lang}"


def _coach_disk_path(coach_id: str) -> Path:
    return storage.coach_cache_path(config.COACH_CACHE_DIR, coach_id, _coach_schema_version())


def _coach_mem_get(coach_id: str) -> dict | None:
    item = _coach_report_cache.get(coach_id)
    if item is None:
        return None
    result, expires_at = item
    if expires_at is not None and time.monotonic() > expires_at:
        _coach_report_cache.pop(coach_id, None)
        return None
    _coach_report_cache.move_to_end(coach_id)
    return result


def _coach_mem_put(coach_id: str, result: dict, degraded: bool) -> None:
    if degraded:
        if config.AI_COACH_DEGRADED_TTL_S <= 0:
            return
        expires_at: float | None = time.monotonic() + config.AI_COACH_DEGRADED_TTL_S
    else:
        expires_at = None
    _coach_report_cache[coach_id] = (result, expires_at)
    _coach_report_cache.move_to_end(coach_id)
    while len(_coach_report_cache) > _COACH_MEMORY_SIZE:
        _coach_report_cache.popitem(last=False)


def _llm_configured() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())


def _active_coach_job(coach_id: str) -> _CoachJob | None:
    job = _coach_jobs.get(coach_id)
    if job is None or job.task is None or job.task.done():
        return None
    if job.task.get_loop() is not asyncio.get_running_loop():
        return None
    return job


async def _run_coach_job(job: _CoachJob, match: MatchData, player_slot: int, lang: str, disk_path: Path) -> dict:
    try:
        from .ai import coach

        report = await asyncio.wait_for(
            coach.generate_coaching_report(match, player_slot, lang=lang, progress_callback=job.on_progress),
            timeout=config.AI_COACH_TIMEOUT_S + _COACH_TIMEOUT_GRACE_S,
        )
        result = report.to_dict()
        degraded = bool(result.get("degraded"))
        _coach_mem_put(job.coach_id, result, degraded)
        if degraded:
            logger.info("AI coach report %s is degraded; not persisting to disk", job.coach_id)
        else:
            try:
                await asyncio.to_thread(storage.save_json, disk_path, result, config.COACH_MAX_FILES)
            except Exception as exc:
                logger.warning("Failed to save coaching cache %s: %s", disk_path.name, exc)
        return result
    finally:
        job.finish()


def _coach_job_done(job: _CoachJob, task: asyncio.Task) -> None:
    if _coach_jobs.get(job.coach_id) is job:
        del _coach_jobs[job.coach_id]
    if task.cancelled():
        logger.warning("AI coach generation %s was cancelled", job.coach_id)
        return
    exc = task.exception()
    if exc is not None:
        logger.error("AI coach generation %s failed", job.coach_id, exc_info=exc)


def _start_coach_job(coach_id: str, match: MatchData, player_slot: int, lang: str, disk_path: Path) -> _CoachJob:
    job = _CoachJob(coach_id)
    job.task = asyncio.get_running_loop().create_task(
        _run_coach_job(job, match, player_slot, lang, disk_path),
    )
    _coach_jobs[coach_id] = job
    job.task.add_done_callback(functools.partial(_coach_job_done, job))
    logger.info("AI coach generation started: %s", coach_id)
    return job


async def _resolve_coach(
    request: Request, file_key: str, player_slot: int, lang: str,
) -> tuple[dict | None, _CoachJob | None]:
    """Return (cached report, None) or (None, in-flight job).

    Only a cold start (no cached report, no in-flight job) is subject to
    the per-IP rate limit and the global daily quota.
    """
    record = await _get_record(file_key)
    if record is None:
        raise ApiError(404, "match_not_found", lang)
    if not any(p.player_slot == player_slot for p in record.match.players):
        raise ApiError(400, "invalid_player_slot", lang)

    coach_id = _coach_id(record, player_slot, lang)
    cached = _coach_mem_get(coach_id)
    if cached is not None:
        return cached, None
    job = _active_coach_job(coach_id)
    if job is not None:
        return None, job

    # First call imports the AI pipeline (slow): keep it off the event loop.
    disk_path = await asyncio.to_thread(_coach_disk_path, coach_id)
    cached = await asyncio.to_thread(storage.load_json, disk_path)
    if cached is not None:
        _coach_mem_put(coach_id, cached, degraded=False)
        return cached, None

    # Re-check: another request may have started the job while we read disk.
    job = _active_coach_job(coach_id)
    if job is not None:
        return None, job

    enforce_rate_limit(request, "ai_coach")
    if _llm_configured():
        wait = ai_coach_daily_quota.try_acquire()
        if wait:
            logger.warning("AI coach daily quota exhausted (%d used)", ai_coach_daily_quota.used())
            raise rate_limited_error(wait, lang, code="daily_limit")
    return None, _start_coach_job(coach_id, record.match, player_slot, lang, disk_path)


def _map_coach_error(exc: BaseException, lang: str) -> ApiError:
    if isinstance(exc, ApiError):
        return exc
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return ApiError(504, "coach_timeout", lang)
    # The player slot was validated up front, so any ValueError here is a
    # pipeline failure rather than a bad request.
    return ApiError(500, "coach_failed", lang)


@router.get("/match/{file_key}/ai-coach/{player_slot}")
async def get_ai_coaching(request: Request, file_key: str, player_slot: int, lang: str = Query("zh")):
    """Generate (or return the cached) AI coaching report for one player."""
    lang = _coach_lang(lang)
    cached, job = await _resolve_coach(request, file_key, player_slot, lang)
    if cached is None:
        try:
            cached = await asyncio.shield(job.task)
        except Exception as exc:
            raise _map_coach_error(exc, lang) from None
    return Response(_json_bytes(cached), media_type="application/json")


def _sse_pack(payload: dict) -> str:
    return f"data: {_json_bytes(payload).decode('utf-8')}\n\n"


def _sse_error(exc: ApiError) -> dict:
    return {"type": "error", "message": exc.detail, "code": exc.code, "status": exc.status_code, **exc.extra}


async def _sse_once(payload: dict) -> AsyncIterator[str]:
    yield _sse_pack(payload)


async def _sse_follow(job: _CoachJob, lang: str) -> AsyncIterator[str]:
    """Relay a job's progress, heartbeat while idle, then the final event.

    A client disconnect only detaches this listener; the generation keeps
    running and fills the cache for everyone else (and for reconnects).
    """
    queue: asyncio.Queue = asyncio.Queue()
    job.listeners.add(queue)
    try:
        yield _sse_pack(job.last_progress or {"type": "progress", "stage": "starting", "current": 0, "total": 1})
        while not job.task.done():
            try:
                item = await asyncio.wait_for(queue.get(), timeout=SSE_HEARTBEAT_S)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if item is _JOB_FINISHED:
                break
            yield _sse_pack(item)
        try:
            result = await asyncio.shield(job.task)
        except Exception as exc:
            yield _sse_pack(_sse_error(_map_coach_error(exc, lang)))
        else:
            yield _sse_pack({"type": "done", "data": result})
    finally:
        job.listeners.discard(queue)


def _sse_response(body: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )


@router.get("/match/{file_key}/ai-coach-stream/{player_slot}")
async def stream_ai_coaching(request: Request, file_key: str, player_slot: int, lang: str = Query("zh")):
    """Stream AI coaching report progress via Server-Sent Events.

    Each event is ``data: <json>\\n\\n`` with one of these shapes:

    - ``{"type": "progress", "stage": str, "current": int, "total": int}``
    - ``{"type": "done", "data": <full report dict>}``
    - ``{"type": "error", "message": str, "code": str, "status": int[, "retry_after": int]}``

    ``: ping`` comments are sent every ~15 s while waiting. Errors are sent
    as events (HTTP 200) so EventSource clients can show the message. A
    reconnecting client attaches to the in-flight generation.
    """
    lang = _coach_lang(lang)
    try:
        cached, job = await _resolve_coach(request, file_key, player_slot, lang)
    except ApiError as exc:
        return _sse_response(_sse_once(_sse_error(exc)))
    if cached is not None:
        return _sse_response(_sse_once({"type": "done", "data": cached}))
    return _sse_response(_sse_follow(job, lang))


@router.get("/match/{file_key}/{section}")
async def get_match_section(request: Request, file_key: str, section: str):
    """Get a specific analysis section for a match."""
    lang = request_lang(request)
    record = await _get_record(file_key)
    if record is None:
        raise ApiError(404, "match_not_found", lang)
    analysis = record.entry["analysis"]
    if section not in analysis:
        raise ApiError(400, "invalid_section", lang, extra={"valid": sorted(analysis.keys())})
    return Response(_json_bytes({section: analysis[section]}), media_type="application/json")


# ---------------------------------------------------------------------------
# Admin endpoints (X-Admin-Token)
# ---------------------------------------------------------------------------

def _current_patch() -> str | None:
    try:
        from .ai.data_manager import get_current_patch
    except (ImportError, AttributeError):
        return None
    try:
        return get_current_patch()
    except Exception as exc:
        logger.warning("get_current_patch failed: %s", exc)
        return None


@router.post("/admin/update-gamedata", dependencies=[Depends(require_admin)])
async def update_gamedata(request: Request, force: bool = False):
    """Fetch latest game data from the OpenDota API.

    Updates hero, ability and item stats (Layer A). Editorial coaching
    tips (Layer B) are never overwritten.
    """
    try:
        from .ai.data_manager import fetch_and_save_gamedata

        return await asyncio.to_thread(fetch_and_save_gamedata, force=force)
    except Exception:
        logger.exception("Game data update failed")
        raise ApiError(502, "gamedata_update_failed", request_lang(request)) from None


@router.get("/admin/gamedata-status", dependencies=[Depends(require_admin)])
async def gamedata_status():
    """Return current game data status (patch version, freshness, coverage)."""
    try:
        from .ai.data_manager import get_meta
    except (ImportError, AttributeError):
        return {"status": "module_unavailable", "patch_version": _current_patch()}
    meta = dict(await asyncio.to_thread(get_meta))
    current = await asyncio.to_thread(_current_patch)
    if current:
        meta["current_patch"] = current
    return meta


@router.get("/admin/status", dependencies=[Depends(require_admin)])
async def server_status():
    """Operational counters: caches, in-flight work and AI coach quota."""
    return {
        "matches_in_memory": len(_matches),
        "coach_reports_in_memory": len(_coach_report_cache),
        "inflight_tasks": sorted(_inflight),
        "coach_jobs": sorted(_coach_jobs),
        "uploads_in_progress": _uploads_in_progress,
        "ai_coach_daily": {
            "used": ai_coach_daily_quota.used(),
            "limit": config.AI_COACH_DAILY_LIMIT,
            "llm_configured": _llm_configured(),
        },
    }
