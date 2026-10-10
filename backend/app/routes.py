"""FastAPI routes for the Dota 2 match analysis API.

Match results live in three tiers: an in-memory LRU (fast path), a bounded
disk store (``cache/matches``, survives eviction and restarts) and the
original source (replay file / OpenDota). Expensive work — replay parsing,
OpenDota fetches and AI-coach generation — is de-duplicated per key so
concurrent requests share one computation. The analysis of a cached match
is computed again when the game data it reads (talent names, patch label)
has changed since.

A match that was served once stays available to ``/api/match``, the section
endpoints and the AI coach: OpenDota results without parsed data are kept
like any other match and only ``/api/fetch`` decides when to download them
again; the demo and uploaded replays are rebuilt from their source when no
cached copy is left.

Errors returned to clients are short, localized (``?lang=`` or
Accept-Language) and carry a stable ``code``; details are only logged.
"""
from __future__ import annotations

import asyncio
import functools
import gzip
import http.client
import importlib.util
import json
import logging
import math
import os
import re
import ssl
import time
import urllib.error
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response, StreamingResponse
from starlette.datastructures import FormData
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect

from . import config, storage
from . import parser as parser_mod
from . import analyzer
from .analyzer import full_analysis
from .models import MatchData
from .security import (
    ApiError,
    ai_coach_daily_quota,
    client_ip,
    enforce_rate_limit,
    rate_limit_key,
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
    # API payload: {"analysis", "parse_source", "has_parsed_data", "_match_key", ...}
    entry: dict
    match: MatchData
    body: bytes = b""       # ``entry`` as JSON, serialized once off the event loop
    body_gzip: bytes = b""  # gzip of ``body`` (empty when too small to bother)
    # OpenDota results only: when the match was downloaded (wall clock),
    # whether the next /api/fetch must download it again (also stored on
    # disk) and, after a failed download, when to try again at the earliest.
    # None of this stops the record from being served to /api/match, the
    # sections or the AI coach.
    fetched_at: float = 0.0
    refetch: bool = False
    retry_at: float = 0.0
    # data_manager.data_generation() the analysis was computed under. The
    # game data names the talents and labels the patch, so a payload from
    # before it changed (the first fetch after a start, a new game patch)
    # is computed again when the match is next asked for.
    data_generation: int = 0


_matches: OrderedDict[str, _MatchRecord] = OrderedDict()

match_store = storage.MatchStore(config.MATCH_CACHE_DIR, lambda: config.MATCH_CACHE_MAX_FILES)


def _memory_get(key: str) -> _MatchRecord | None:
    record = _matches.get(key)
    if record is None:
        return None
    _matches.move_to_end(key)
    return record


def _memory_put(key: str, record: _MatchRecord) -> None:
    _matches[key] = record
    _matches.move_to_end(key)
    while len(_matches) > config.MATCH_MEMORY_CACHE_SIZE:
        _matches.popitem(last=False)


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


def _data_generation() -> int:
    """Counter of the game data the analysis reads (see ``_MatchRecord``)."""
    try:
        from .ai.data_manager import data_generation
    except ImportError:  # no data layer: the analysis does not depend on it either
        return 0
    return data_generation()


def _serialized(entry: dict) -> tuple[bytes, bytes]:
    """``entry`` as JSON and the gzip of it (empty when too small to bother)."""
    body = _json_bytes(entry)
    return body, gzip.compress(body, compresslevel=6) if len(body) >= 1024 else b""


def _build_record(key: str, match: MatchData, parse_source: str, extra: dict | None) -> _MatchRecord:
    """Analyze and pre-serialize a match (CPU-bound: run in a worker thread)."""
    generation = _data_generation()  # read first: data that changes meanwhile leaves the record stale
    entry = {
        "analysis": full_analysis(match),
        "parse_source": parse_source,
        **(extra or {}),
        # False only for an OpenDota match whose replay OpenDota has not
        # parsed (yet): scoreboard data only, no timelines / wards / fights.
        "has_parsed_data": parse_source != "opendota" or bool(match.gold_xp_timeline),
        "_match_key": key,
    }
    body, body_gzip = _serialized(entry)
    return _MatchRecord(entry, match, body=body, body_gzip=body_gzip, data_generation=generation)


def _reanalyzed(record: _MatchRecord) -> tuple[dict, bytes, bytes, int]:
    """Payload of a cached match analyzed again (CPU-bound: run in a worker thread)."""
    generation = _data_generation()
    entry = {**record.entry, "analysis": full_analysis(record.match)}
    return (entry, *_serialized(entry), generation)


async def _refresh_record(key: str, record: _MatchRecord) -> _MatchRecord:
    """Bring a cached match up to date after the game data changed."""
    entry, body, body_gzip, generation = await asyncio.to_thread(_reanalyzed, record)
    record.entry, record.body, record.body_gzip, record.data_generation = entry, body, body_gzip, generation
    logger.info("Analyzed match %s again: the game data changed", key)
    return record


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
    fetched_at: float = 0.0,
) -> _MatchRecord:
    """Analyze ``match`` off the event loop, cache it and optionally persist it."""
    record = await asyncio.to_thread(_build_record, key, match, parse_source, extra)
    record.fetched_at = fetched_at
    _memory_put(key, record)
    if persist:
        meta = {"parse_source": parse_source, "extra": extra or {}}
        if fetched_at:
            meta["fetched_at"] = fetched_at
        try:
            await asyncio.to_thread(match_store.save, key, match, meta)
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
    try:
        record.fetched_at = float(meta.get("fetched_at") or 0.0)
    except (TypeError, ValueError):
        record.fetched_at = 0.0
    record.refetch = bool(meta.get("refetch"))
    _memory_put(key, record)
    logger.info("Reloaded match %s from disk store", key)
    return record


async def _get_record(key: str) -> _MatchRecord | None:
    """Look a match up in memory, then in the disk store."""
    record = _memory_get(key)
    if record is not None:
        if record.data_generation != _data_generation():
            await _singleflight(f"refresh:{key}", lambda: _refresh_record(key, record))
        return record
    if not match_store.path_for(key).is_file():  # cheap miss, no thread hop
        return None
    return await _singleflight(f"load:{key}", lambda: _load_stored_match(key))


_UPLOAD_KEY_RE = re.compile(r"replay_([0-9a-f]{%d})" % storage.UPLOAD_HASH_LEN)


async def _rebuild_record(key: str) -> _MatchRecord | None:
    """Recreate a match that is in neither cache from its source, if we have it.

    Covers the demo (always succeeds: it falls back to mock data) and
    uploaded replays whose file is still in ``replays/`` — e.g. after the
    stored copy was dropped because an older parser version wrote it.
    Other keys (OpenDota, parse-local) are re-created by their endpoints.
    """
    if key == "demo":
        return await _singleflight("parse:demo", _build_demo)
    found = _UPLOAD_KEY_RE.fullmatch(key)
    if found is None:
        return None
    path = config.REPLAY_DIR / f"{found.group(1)}.dem"
    if not path.is_file():
        return None
    try:
        record = await _singleflight(f"parse:{key}", lambda: _parse_replay(key, path))
    except Exception as exc:
        logger.warning("Could not rebuild %s from its stored replay: %s", key, exc)
        if isinstance(exc, _parser_exc("ReplayParseError")):
            await asyncio.to_thread(storage.delete_file, path)
        return None
    logger.info("Rebuilt match %s from its stored replay", key)
    return record


async def _require_record(key: str, lang: str) -> _MatchRecord:
    """The match for ``key`` (cached, stored or rebuilt) or 404 match_not_found."""
    record = await _get_record(key) or await _rebuild_record(key)
    if record is None:
        raise ApiError(404, "match_not_found", lang)
    return record


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


# "Could not talk to OpenDota", as opposed to a bug in our own transform or
# analysis code running in the same request.
_NETWORK_ERRORS = (
    urllib.error.URLError, http.client.HTTPException, ConnectionError, ssl.SSLError,
    TimeoutError, asyncio.TimeoutError,
)


def _map_error(exc: BaseException, lang: str, context: str) -> ApiError:
    """Translate parser / OpenDota failures into short user-safe API errors.

    ``context`` is "replay" or "opendota"; in the latter, network-type
    failures are reported as OpenDota being unavailable. Anything else is
    our own fault: logged with its traceback and answered with a 500.
    Details are logged, never returned.
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
    if context == "opendota" and isinstance(exc, _NETWORK_ERRORS):
        logger.warning("OpenDota request failed: %r", exc)
        return ApiError(502, "opendota_unavailable", lang)
    logger.error("Unexpected %s failure", context, exc_info=exc)
    return ApiError(500, "internal_error", lang)


def _parse_match_id(raw: str, lang: str) -> int:
    """Match ID from a path segment: ASCII digits only, a positive int64."""
    raw = raw.strip()
    if not re.fullmatch(r"[0-9]{1,19}", raw) or not 0 < int(raw) < 2 ** 63:
        raise ApiError(400, "invalid_match_id", lang)
    return int(raw)


# ---------------------------------------------------------------------------
# Game data / knowledge retrieval state (optional AI data layer)
# ---------------------------------------------------------------------------

def rag_package():
    """The knowledge-retrieval package ``ai.rag``, or None when it cannot be imported.

    Its ``status()`` / ``warm_up()`` / ``refresh(reason)`` never raise;
    without the package the coach's knowledge is ranked lexically.
    """
    try:
        from .ai import rag
    except Exception as exc:
        logger.debug("Knowledge retrieval package unavailable: %s", exc)
        return None
    return rag


def _rag_status() -> dict | None:
    """``ai.rag.status()``: ``{mode, embedder, vectors, chunks, static_coverage,
    building, last_build, error}``; None without the package. Blocking (the
    first call reads the vector files)."""
    rag = rag_package()
    if rag is None:
        return None
    try:
        return dict(rag.status())
    except Exception as exc:
        logger.warning("Knowledge retrieval status failed: %s", exc)
        return None


def _knowledge_mode() -> str:
    """How the coach's reference knowledge is retrieved right now: "hybrid"
    (tags + BM25 + vectors) or "lexical" (tags + BM25 only). Blocking."""
    status = _rag_status() or {}
    return "hybrid" if status.get("mode") == "hybrid" else "lexical"


def _game_patch() -> str | None:
    """The game's current full version ("7.41f"), or None when the server
    does not know it.

    It only counts as known while Valve's release list was downloaded
    within the last GAMEDATA_STALE_DAYS (the rule a replay's patch label
    follows, applied to this moment): never the version compiled into the
    data layer, and not a list too old to rule out a newer patch. So a
    server that is offline or runs with GAMEDATA_AUTO_UPDATE=0 stops
    naming a version instead of naming an outdated one. Blocking, never
    raises, no network (the list is refreshed by the maintenance tasks).
    """
    try:
        from .ai.data_manager import patch_label_for_time
    except ImportError:
        return None
    try:
        return patch_label_for_time(time.time()) or None
    except Exception as exc:
        logger.warning("patch_label_for_time failed: %s", exc)
        return None


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
    ``upload_extensions`` are the accepted file-name endings (compressed
    replays are unpacked by the server). ``game_patch`` is the game's
    current version as last looked up (None while the server has not
    checked it recently, see ``_game_patch``) and ``knowledge_mode`` how
    the AI coach retrieves its reference knowledge: ``hybrid`` (with
    vectors) or ``lexical``.
    """
    try:
        await _get_replay_parser()
        replay_parser = True
    except Exception:
        replay_parser = False
    limit = config.AI_COACH_DAILY_LIMIT
    game_patch, knowledge_mode = await asyncio.to_thread(lambda: (_game_patch(), _knowledge_mode()))
    return {
        "replay_parser": replay_parser,
        "max_upload_mb": config.MAX_UPLOAD_MB,
        "upload_extensions": list(storage.UPLOAD_SUFFIXES),
        "ai_coach_llm": _llm_configured(),
        "ai_coach_daily_remaining": max(0, limit - ai_coach_daily_quota.used()) if limit > 0 else None,
        "game_patch": game_patch,
        "knowledge_mode": knowledge_mode,
    }


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

_uploads_in_progress = 0
_uploads_by_client: dict[str, int] = {}  # rate-limit identity -> uploads being received
# Only the file part is used; other multipart fields may be at most this big.
_UPLOAD_MAX_FIELD_BYTES = 16 * 1024

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


def _is_multipart(request: Request) -> bool:
    content_type = request.headers.get("content-type", "")
    return content_type.split(";", 1)[0].strip().lower() == "multipart/form-data"


class _UploadStalled(Exception):
    """No body data arrived for UPLOAD_IDLE_TIMEOUT_S."""


async def _read_upload_form(request: Request, lang: str) -> FormData:
    """Receive and parse the multipart body within the receive deadlines.

    Two limits apply: the whole body must arrive within
    UPLOAD_RECEIVE_TIMEOUT_S, and no single read may wait longer than
    UPLOAD_IDLE_TIMEOUT_S (a connection that died without closing would
    otherwise keep its client's upload slot until the overall deadline).

    When this fails part-way (deadline, disconnect, malformed body) nothing
    is left behind: the spooled part is an anonymous temp file that is
    released together with the abandoned parser.
    """
    receive = request.receive

    async def receive_or_stall() -> dict:
        try:
            return await asyncio.wait_for(receive(), timeout=config.UPLOAD_IDLE_TIMEOUT_S)
        except (asyncio.TimeoutError, TimeoutError):
            raise _UploadStalled() from None

    body = Request(request.scope, receive_or_stall)
    try:
        return await asyncio.wait_for(
            body.form(max_files=1, max_fields=8, max_part_size=_UPLOAD_MAX_FIELD_BYTES),
            timeout=config.UPLOAD_RECEIVE_TIMEOUT_S,
        )
    except ApiError:
        raise
    except (_UploadStalled, asyncio.TimeoutError, TimeoutError) as exc:
        if isinstance(exc, _UploadStalled):
            logger.info(
                "Upload from %s dropped: no data received for %d s",
                client_ip(request), config.UPLOAD_IDLE_TIMEOUT_S,
            )
        else:
            logger.info(
                "Upload from %s dropped: body not received within %d s",
                client_ip(request), config.UPLOAD_RECEIVE_TIMEOUT_S,
            )
        # The rest of the body may still be on its way: do not wait for it.
        raise ApiError(400, "upload_timeout", lang, headers={"Connection": "close"}) from None
    except ClientDisconnect:
        # Cancelled in the browser or connection lost; nobody reads the answer.
        logger.info("Upload aborted by client %s", client_ip(request))
        raise ApiError(400, "bad_request", lang) from None
    except StarletteHTTPException as exc:
        logger.info("Rejected malformed upload: %s", exc.detail)
        raise ApiError(400, "bad_request", lang) from None


async def _receive_upload(request: Request, lang: str) -> storage.StoredReplay:
    """Parse the multipart body and store the replay under its content hash.

    The file name only has to end like a replay; whether the content is a
    demo file, bzip2 or Zstandard is told from its first bytes.
    """
    form = await _read_upload_form(request, lang)
    try:
        upload = form.get("file")
        if not isinstance(upload, StarletteUploadFile):
            raise ApiError(400, "upload_missing_file", lang)
        if not PurePath(upload.filename or "").name.lower().endswith(storage.UPLOAD_SUFFIXES):
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


async def _parse_replay(key: str, path: Path) -> _MatchRecord:
    parser = await _get_replay_parser()
    match_data = await parser.parse(str(path))
    return await _store_match(key, match_data, parser.name())


async def _parse_uploaded(key: str, path: Path) -> _MatchRecord:
    try:
        return await _parse_replay(key, path)
    except Exception:
        # Unparseable (or unparseable here) — don't keep the file around.
        await asyncio.to_thread(storage.delete_file, path)
        raise


@router.post("/upload", openapi_extra=_UPLOAD_OPENAPI)
async def upload_replay(request: Request):
    """Upload a .dem replay file (multipart field ``file``) for analysis.

    A bzip2 or Zstandard compressed replay (``.dem.bz2`` / ``.dem.zst``, as
    downloaded from Valve's replay servers) is unpacked by the server. The
    replay is stored under the content hash of the demo file itself, so
    re-uploading it, compressed or not, is answered from cache without
    parsing again.

    Everything that can be refused is refused before the body is read; the
    connection stays open so the client still gets the JSON error after it
    finished sending (see ``BodySizeLimitMiddleware``).
    """
    global _uploads_in_progress
    lang = request_lang(request)
    # Starlette only caps multipart bodies; never hand it anything else.
    if not _is_multipart(request):
        raise ApiError(400, "bad_request", lang)
    enforce_rate_limit(request, "upload")
    try:
        await _get_replay_parser()  # fail fast before receiving a large body
    except Exception as exc:
        raise _map_error(exc, lang, "replay") from None
    client = rate_limit_key(client_ip(request))
    if _uploads_by_client.get(client, 0) >= config.UPLOAD_MAX_PER_CLIENT:
        raise ApiError(429, "upload_in_progress", lang, headers={"Retry-After": "30"}, extra={"retry_after": 30})
    if _uploads_in_progress >= config.UPLOAD_MAX_CONCURRENCY:
        raise ApiError(503, "server_busy", lang, headers={"Retry-After": "30"}, extra={"retry_after": 30})

    _uploads_in_progress += 1
    _uploads_by_client[client] = _uploads_by_client.get(client, 0) + 1
    try:
        stored = await _receive_upload(request, lang)
    finally:
        _uploads_in_progress -= 1
        remaining = _uploads_by_client.get(client, 1) - 1
        if remaining > 0:
            _uploads_by_client[client] = remaining
        else:
            _uploads_by_client.pop(client, None)

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

    # Mock data is regenerated on demand (see _rebuild_record); only persist
    # real results.
    return await _store_match("demo", match_data, parse_source, persist=parse_source != "mock")


@router.get("/demo")
async def get_demo_analysis(request: Request):
    """Get analysis for the default demo match (real replay if available)."""
    record = await _require_record("demo", request_lang(request))
    return _match_response(request, record)


async def _fetch_opendota(key: str, match_id: int) -> _MatchRecord:
    match_data = await parser_mod.OpenDotaParser().parse(str(match_id))
    # Replaces whatever copy we had (basic data is overwritten once OpenDota
    # has parsed the replay).
    return await _store_match(key, match_data, "opendota", fetched_at=time.time())


def _needs_refetch(record: _MatchRecord) -> bool:
    """Whether /api/fetch should download this OpenDota match again.

    True after a request-parse, and for a match without parsed data once
    our copy is older than OPENDOTA_UNPARSED_TTL_S (so the parsed data is
    picked up when OpenDota has it); not while backing off after a failed
    download. Only /api/fetch looks at this.
    """
    if time.time() < record.retry_at:
        return False
    if record.refetch:
        return True
    if record.entry.get("has_parsed_data"):
        return False
    return time.time() - record.fetched_at >= config.OPENDOTA_UNPARSED_TTL_S


@router.get("/fetch/{match_id}")
async def fetch_match_by_id(request: Request, match_id: str):
    """Fetch match data from the OpenDota API by match ID.

    ``has_parsed_data`` is false while OpenDota only has the scoreboard of
    the match; calling this again later returns the parsed data once it
    exists (see ``/api/request-parse``).
    """
    lang = request_lang(request)
    mid = _parse_match_id(match_id, lang)
    key = f"opendota_{mid}"
    record = await _get_record(key)
    if record is None or _needs_refetch(record):
        try:
            enforce_rate_limit(request, "fetch")
            record = await _singleflight(f"parse:{key}", lambda: _fetch_opendota(key, mid))
        except Exception as exc:
            error = _map_error(exc, lang, "opendota")
            if record is None:
                raise error from None
            # The copy we have is still valid; try again after the TTL. A
            # request-parse mark stays until a download succeeds.
            logger.info("Serving the cached copy of %s (refresh failed: %s)", key, error.code)
            record.retry_at = time.time() + config.OPENDOTA_UNPARSED_TTL_S
    return _match_response(request, record)


@router.post("/request-parse/{match_id}")
async def request_parse(request: Request, match_id: str):
    """Ask OpenDota to parse a match replay.

    Our copy of the match keeps being served (match view, sections, AI
    coach); it is only marked so the next ``/api/fetch`` downloads it again.
    The mark is stored with the match, so it survives eviction and restarts.
    """
    lang = request_lang(request)
    mid = _parse_match_id(match_id, lang)
    enforce_rate_limit(request, "request_parse")
    try:
        result = await asyncio.to_thread(parser_mod.OpenDotaParser().request_parse, mid)
    except Exception as exc:
        raise _map_error(exc, lang, "opendota") from None
    key = f"opendota_{mid}"
    record = await _get_record(key)
    if record is not None:
        record.retry_at = 0.0  # OpenDota answered: stop backing off
        if not record.refetch:
            record.refetch = True
            try:
                await asyncio.to_thread(match_store.update_meta, key, {"refetch": True})
            except Exception as exc:  # the in-memory mark still works
                logger.warning("Failed to persist the refetch mark of %s: %s", key, exc)
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
    record = await _require_record(file_key, request_lang(request))
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


def _parse_player_slot(raw: str, lang: str) -> int:
    """Player slot from a path segment (0-4 Radiant, 128-132 Dire)."""
    raw = raw.strip()
    if not re.fullmatch(r"[0-9]{1,3}", raw):
        raise ApiError(400, "invalid_player_slot", lang)
    return int(raw)


def _coach_id(record: _MatchRecord, player_slot: int, lang: str) -> str:
    """Identity of a report: same match data + player + language + LLM mode.

    Built on the match key (the content hash for uploads), not on the match
    ID: replays without an ID, or a truncated and a complete replay of one
    match, must never share a report. The demo key is the one key that
    names a place instead of a match (whatever ``DEFAULT_REPLAY`` holds), so
    there the match ID is part of the identity: a report written for the
    previous demo replay is not served for the next one. Reports written
    without an LLM have their own identity, so they stop being served once
    a key is configured.
    """
    source = str(record.entry.get("parse_source") or "unknown")
    if source == "opendota" and not record.entry.get("has_parsed_data", True):
        source = "opendota-basic"
    mode = "llm" if _llm_configured() else "rules"
    key = record.entry["_match_key"]
    if key == "demo":
        key = f"demo-{record.match.match_id}"
    return f"{key}_{source}_{player_slot}_{lang}_{mode}"


def _coach_disk_path(coach_id: str) -> Path:
    # A report is only as current as the match data it was written from.
    data_id = f"{coach_id}_d{storage.MatchStore.FORMAT}"
    return storage.coach_cache_path(config.COACH_CACHE_DIR, data_id, _coach_schema_version())


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


@functools.lru_cache(maxsize=1)
def _openai_installed() -> bool:
    return importlib.util.find_spec("openai") is not None


def _llm_configured() -> bool:
    """Whether new reports are LLM-backed: API key set and SDK installed
    (the AI pipeline's own condition; otherwise it writes rule-based reports)."""
    return bool(os.environ.get("DEEPSEEK_API_KEY", "").strip()) and _openai_installed()


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
    the per-IP rate limit (``ai_coach`` with an LLM configured, the much
    higher ``ai_coach_rules`` for rule-based reports) and, with an LLM, to
    the global daily quota. A replay that ends
    before the game starts, or within its first minutes, is refused (409
    coach_no_game).
    """
    record = await _require_record(file_key, lang)
    if not any(p.player_slot == player_slot for p in record.match.players):
        raise ApiError(400, "invalid_player_slot", lang)
    if not analyzer._coach_available(record.match):
        # A replay cut before or just after the horn has no game to review:
        # a report on it could only make things up.
        raise ApiError(409, "coach_no_game", lang)

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

    llm = _llm_configured()
    enforce_rate_limit(request, "ai_coach" if llm else "ai_coach_rules")
    if llm:
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
    if type(exc).__name__ == "NoGameError":     # ai.coach's refusal; the check above normally comes first
        return ApiError(409, "coach_no_game", lang)
    # The player slot was validated up front, so any ValueError here is a
    # pipeline failure rather than a bad request.
    return ApiError(500, "coach_failed", lang)


@router.get("/match/{file_key}/ai-coach/{player_slot}")
async def get_ai_coaching(request: Request, file_key: str, player_slot: str, lang: str = Query("zh")):
    """Generate (or return the cached) AI coaching report for one player.

    ``player_slot`` is the player's slot number (0-4 Radiant, 128-132 Dire).
    """
    lang = _coach_lang(lang)
    cached, job = await _resolve_coach(request, file_key, _parse_player_slot(player_slot, lang), lang)
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
async def stream_ai_coaching(request: Request, file_key: str, player_slot: str, lang: str = Query("zh")):
    """Stream AI coaching report progress via Server-Sent Events.

    Each event is ``data: <json>\\n\\n`` with one of these shapes:

    - ``{"type": "progress", "stage": str, "current": int, "total": int}``
    - ``{"type": "done", "data": <full report dict>}``
    - ``{"type": "error", "message": str, "code": str, "status": int[, "retry_after": int]}``

    ``: ping`` comments are sent every ~15 s while waiting. Errors (a bad
    ``player_slot`` included) are sent as events (HTTP 200) so EventSource
    clients can show the message. A reconnecting client attaches to the
    in-flight generation.
    """
    lang = _coach_lang(lang)
    try:
        cached, job = await _resolve_coach(request, file_key, _parse_player_slot(player_slot, lang), lang)
    except ApiError as exc:
        return _sse_response(_sse_once(_sse_error(exc)))
    if cached is not None:
        return _sse_response(_sse_once({"type": "done", "data": cached}))
    return _sse_response(_sse_follow(job, lang))


@router.get("/match/{file_key}/{section}")
async def get_match_section(request: Request, file_key: str, section: str):
    """Get a specific analysis section for a match."""
    lang = request_lang(request)
    record = await _require_record(file_key, lang)
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


def _update_gamedata(force: bool, allow_shrink: bool = False) -> dict:
    """Fetch the game data; new data is announced to the retrieval index. Blocking."""
    from .ai.data_manager import fetch_and_save_gamedata

    # (allow_shrink is only passed on when asked for: the default call stays
    # the one every data layer understands)
    result = fetch_and_save_gamedata(force=force, **({"allow_shrink": True} if allow_shrink else {}))
    if result.get("status") == "updated" and result.get("changed", True):
        rag = rag_package()
        if rag is not None:
            rag.refresh("gamedata")
    return result


@router.post("/admin/update-gamedata", dependencies=[Depends(require_admin)])
async def update_gamedata(request: Request, force: bool = False, allow_shrink: bool = False):
    """Fetch the latest game data (dotaconstants; see ``ai.data_manager``).

    Without ``force`` nothing is downloaded while the local data is current.
    Updates hero, ability and item stats (Layer A). Editorial coaching
    tips (Layer B) are never overwritten. The data layer refuses a download
    that lost more than a tenth of the heroes, items or talents (a broken
    upstream build) and keeps the stored data; ``allow_shrink`` accepts
    such a loss after a look at the cause (the server log names it).
    """
    try:
        return await asyncio.to_thread(_update_gamedata, force, allow_shrink)
    except Exception:
        logger.exception("Game data update failed")
        raise ApiError(502, "gamedata_update_failed", request_lang(request)) from None


@router.get("/admin/gamedata-status", dependencies=[Depends(require_admin)])
async def gamedata_status():
    """Return the game data status as the data layer reports it: major
    version (``patch_version``), the game's full version (``patch_label``),
    source and revision, age, counts."""
    try:
        from .ai.data_manager import get_meta
    except (ImportError, AttributeError):
        return {"status": "module_unavailable", "patch_version": _current_patch()}
    meta = dict(await asyncio.to_thread(get_meta))
    current = await asyncio.to_thread(_current_patch)
    if current:
        meta["current_patch"] = current
    return meta


def _learning_status() -> dict | None:
    """Self-learning state: levels that are due and the last recorded run
    (``{due_levels, levels, last_run, patch}``, see ``pipeline.learning_status``).

    None when the learning package is unavailable. Blocking (reads the
    manifest and the current patch).
    """
    try:
        from .ai.learning import pipeline
    except Exception:
        return None
    return pipeline.learning_status()


# What /api/admin/status shows of data_manager.get_meta().
_GAMEDATA_STATUS_KEYS = (
    "patch_version", "patch_label", "data_label", "data_current",
    "source", "source_date", "last_fetched", "age_days", "is_stale",
)


def _gamedata_status() -> dict | None:
    """Game data in short: which version it is for, where it came from, how old it is.

    ``patch_label`` is the game's current full version ("7.41f"),
    ``patch_version`` the major version of the local data, ``data_label``
    the version that data reflects and ``data_current`` whether the two
    agree (None = cannot be told); ``source`` names the download source
    ("fallback" before the first fetch) and ``age_days`` the time since
    ``last_fetched``. ``checked_at`` is when the game version was last
    looked up (the periodic check is alive while it keeps moving). A value
    the data layer does not report is None. The whole object is None when
    the data layer is unavailable. Blocking.
    """
    try:
        from .ai.data_manager import get_meta
    except ImportError:
        return None
    try:
        meta = get_meta()
    except Exception as exc:
        logger.warning("get_meta failed: %s", exc)
        return None
    status = {key: meta.get(key) for key in _GAMEDATA_STATUS_KEYS}
    try:
        from .ai.data_manager import get_patch_info

        status["checked_at"] = get_patch_info().get("checked_at") or None
    except Exception:  # a data layer without it: not reported
        status["checked_at"] = None
    return status


def _maintenance_state(request: Request) -> str:
    """State of the startup maintenance task (game data, then learning)."""
    task = getattr(request.app.state, "startup_task", None)
    if task is None:
        return "not_started"
    if not task.done():
        return "running"
    if task.cancelled():
        return "cancelled"
    return "failed" if task.exception() is not None else "finished"


@router.get("/admin/status", dependencies=[Depends(require_admin)])
async def server_status(request: Request):
    """Operational state: caches, in-flight work, AI coach quota, learning,
    game data (``gamedata``) and the knowledge-retrieval index (``rag``)."""
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
        "startup_maintenance": _maintenance_state(request),
        "learning": await asyncio.to_thread(_learning_status),
        "gamedata": await asyncio.to_thread(_gamedata_status),
        "rag": await asyncio.to_thread(_rag_status),
    }
