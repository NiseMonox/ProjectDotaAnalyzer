"""Request-level protections for the public server.

- :class:`ApiError`: short, user-safe, localized (zh/en) error responses
  carrying a stable ``code`` (rendered as ``{"detail", "code", ...}``). The
  same shape is used for every other error under ``/api/`` (unknown route,
  wrong method, request validation, unhandled exception).
- Admin auth (``X-Admin-Token`` vs env ``ADMIN_TOKEN``; unset → 403).
- Per-client-IP sliding-window rate limits and a global daily quota for
  cold LLM-backed AI-coach reports.
- ASGI middlewares: upload body size limit (413) and security headers.

All state is in-memory and per-process: run a single uvicorn worker.
"""
from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from fastapi import HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import config

logger = logging.getLogger(__name__)

SUPPORTED_LANGS = ("zh", "en")
DEFAULT_LANG = "zh"


# ---------------------------------------------------------------------------
# Localized error messages
# ---------------------------------------------------------------------------

MESSAGES: dict[str, dict[str, str]] = {
    "rate_limited": {
        "en": "Too many requests. Please retry in {retry_after} s.",
        "zh": "请求过于频繁，请在 {retry_after} 秒后重试。",
    },
    "daily_limit": {
        "en": "Today's AI coach quota has been used up. Please try again tomorrow.",
        "zh": "今日 AI 教练分析额度已用完，请明天再试。",
    },
    "admin_disabled": {
        "en": "Admin endpoints are disabled on this server.",
        "zh": "此服务器未启用管理接口。",
    },
    "admin_forbidden": {
        "en": "Invalid admin token.",
        "zh": "管理令牌无效。",
    },
    "upload_too_large": {
        "en": "Replay file is too large (limit {limit_mb} MB).",
        "zh": "录像文件过大（上限 {limit_mb} MB）。",
    },
    "server_busy": {
        "en": "The server is busy with other uploads. Please retry shortly.",
        "zh": "服务器正忙于处理其他上传，请稍后重试。",
    },
    "upload_in_progress": {
        "en": "Another upload from your network is still in progress. Please wait for it to finish.",
        "zh": "您的网络已有录像正在上传，请等待其完成后再试。",
    },
    "upload_timeout": {
        "en": "The upload took too long and was cancelled. Please check your connection and try again.",
        "zh": "上传耗时过长，已被取消。请检查网络后重试。",
    },
    "upload_missing_file": {
        "en": "No replay file was uploaded.",
        "zh": "未收到录像文件。",
    },
    "bad_request": {
        "en": "Malformed request.",
        "zh": "请求格式错误。",
    },
    "not_found": {
        "en": "This API endpoint does not exist.",
        "zh": "该接口不存在。",
    },
    "method_not_allowed": {
        "en": "This request method is not allowed here.",
        "zh": "该接口不支持此请求方法。",
    },
    "invalid_match_id": {
        "en": "Invalid match ID.",
        "zh": "无效的比赛 ID。",
    },
    "upload_bad_extension": {
        "en": "Only .dem replay files are supported (also compressed: .dem.bz2, .dem.zst).",
        "zh": "仅支持 .dem 录像文件（也可上传压缩文件 .dem.bz2、.dem.zst）。",
    },
    # Only for a format the server cannot unpack itself: Zstandard without
    # the optional zstandard package (bzip2 uploads are always unpacked).
    "replay_compressed": {
        "en": "This replay is Zstandard-compressed (Valve's replay servers use it, even for files named "
              ".dem.bz2) and this server cannot unpack it. Please decompress it to a .dem file first, "
              "for example with the zstd tool (zstd -d).",
        "zh": "该录像为 Zstandard 压缩文件（Valve 录像服务器使用此格式，文件名可能仍是 .dem.bz2），"
              "本服务器无法解压。请先解压为 .dem 文件（例如使用 zstd 工具：zstd -d）后再上传。",
    },
    # bzip2 on a Python built without the bz2 module.
    "replay_compressed_bzip2": {
        "en": "This replay is bzip2-compressed and this server cannot unpack it. Please decompress it "
              "to a .dem file first, for example with bunzip2.",
        "zh": "该录像为 bzip2 压缩文件，本服务器无法解压。请先解压为 .dem 文件（例如使用 bunzip2）后再上传。",
    },
    "replay_decompress_failed": {
        "en": "This compressed replay is damaged or incomplete. Please download it again, "
              "or decompress it yourself and upload the .dem file.",
        "zh": "该压缩录像已损坏或不完整，请重新下载，或自行解压后上传 .dem 文件。",
    },
    # An intact Zstandard file whose window is above storage.ZSTD_MAX_WINDOW.
    "replay_window_too_large": {
        "en": "This replay was compressed with a Zstandard window larger than this server unpacks "
              "(zstd --ultra or --long). Please decompress it yourself (zstd -d) and upload the .dem file.",
        "zh": "该录像压缩时使用的 Zstandard 窗口超出了本服务器的解压上限（zstd --ultra 或 --long）。"
              "请自行解压（zstd -d）后上传 .dem 文件。",
    },
    "replay_invalid": {
        "en": "This file is not a valid Dota 2 replay.",
        "zh": "该文件不是有效的 Dota 2 录像。",
    },
    "upload_failed": {
        "en": "Failed to store the uploaded file. Please try again.",
        "zh": "上传文件保存失败，请重试。",
    },
    "parser_unavailable": {
        "en": "Replay parsing is not available on this server right now. "
              "Please use \"fetch by match ID\" instead.",
        "zh": "服务器暂时无法解析录像文件，请改用比赛 ID 获取。",
    },
    "replay_parse_failed": {
        "en": "Failed to parse the replay. The file may be corrupted or from an unsupported game version.",
        "zh": "录像解析失败，文件可能已损坏或游戏版本不受支持。",
    },
    "opendota_not_found": {
        "en": "Match not found on OpenDota.",
        "zh": "OpenDota 上未找到该比赛。",
    },
    "opendota_unavailable": {
        "en": "OpenDota is unavailable right now. Please try again later.",
        "zh": "OpenDota 暂时不可用，请稍后重试。",
    },
    "match_not_found": {
        "en": "Match not found. Please upload the replay or fetch the match again.",
        "zh": "未找到比赛数据，请重新上传录像或重新获取比赛。",
    },
    "invalid_section": {
        "en": "Invalid analysis section.",
        "zh": "无效的分析模块。",
    },
    "invalid_player_slot": {
        "en": "This player is not part of the match.",
        "zh": "该玩家不在本场比赛中。",
    },
    "invalid_filename": {
        "en": "Invalid replay file name.",
        "zh": "录像文件名无效。",
    },
    "replay_not_found": {
        "en": "Replay file not found.",
        "zh": "未找到录像文件。",
    },
    "coach_no_game": {
        "en": "This replay ends before the game starts or within its first minutes, so there is nothing "
              "to analyze. Please upload the complete replay.",
        "zh": "该录像在比赛开始前或刚开始几分钟就已结束，没有可分析的对局内容，请上传完整的录像。",
    },
    "coach_failed": {
        "en": "AI coaching analysis failed. Please try again later.",
        "zh": "AI 教练分析失败，请稍后重试。",
    },
    "coach_timeout": {
        "en": "AI coaching analysis timed out. Please try again later.",
        "zh": "AI 教练分析超时，请稍后重试。",
    },
    "gamedata_update_failed": {
        "en": "Game data update failed.",
        "zh": "游戏数据更新失败。",
    },
    "internal_error": {
        "en": "Internal server error.",
        "zh": "服务器内部错误。",
    },
}


def normalize_lang(value: str | None) -> str | None:
    """Return "zh"/"en" for a lang code or Accept-Language value, else None."""
    if not value:
        return None
    primary = value.split(",")[0].split(";")[0].strip().lower()
    if primary.startswith("zh"):
        return "zh"
    if primary.startswith("en"):
        return "en"
    return None


def lang_from_scope(scope: Scope) -> str:
    """Pick the response language: ``?lang=`` first, then Accept-Language."""
    from urllib.parse import parse_qs

    query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
    lang = normalize_lang((query.get("lang") or [""])[0])
    if lang:
        return lang
    headers = Headers(scope=scope)
    return normalize_lang(headers.get("accept-language")) or DEFAULT_LANG


def request_lang(request: Request) -> str:
    """Language for user-facing messages of this request ("zh" or "en")."""
    return lang_from_scope(request.scope)


def message(code: str, lang: str = DEFAULT_LANG, **params) -> str:
    """Return the localized message for ``code`` (falls back to internal_error)."""
    entry = MESSAGES.get(code) or MESSAGES["internal_error"]
    template = entry.get(lang) or entry["en"]
    try:
        return template.format(**params)
    except (KeyError, IndexError, ValueError):
        return template


class ApiError(HTTPException):
    """HTTP error with a stable machine-readable ``code`` and a localized detail.

    Rendered by :func:`api_error_handler` as ``{"detail": str, "code": str,
    **extra}``. Without that handler FastAPI still renders ``{"detail": str}``.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        lang: str = DEFAULT_LANG,
        *,
        headers: dict[str, str] | None = None,
        extra: dict | None = None,
        **params,
    ) -> None:
        super().__init__(status_code=status_code, detail=message(code, lang, **params), headers=headers)
        self.code = code
        self.extra = dict(extra or {})


def rate_limited_error(retry_after: int, lang: str, code: str = "rate_limited") -> ApiError:
    """429 with ``retry_after`` in the body and a Retry-After header."""
    retry_after = max(1, int(retry_after))
    return ApiError(
        429, code, lang,
        headers={"Retry-After": str(retry_after)},
        extra={"retry_after": retry_after},
        retry_after=retry_after,
    )


def error_payload(exc: ApiError) -> dict:
    """JSON body for an :class:`ApiError` (also used for SSE error events)."""
    return {"detail": exc.detail, "code": exc.code, **exc.extra}


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(error_payload(exc), status_code=exc.status_code, headers=exc.headers)


def _is_api_path(request: Request) -> bool:
    path = request.scope.get("path", "")
    return path == "/api" or path.startswith("/api/")


# Framework-raised HTTP errors (no ApiError involved) → error code.
_HTTP_STATUS_CODES = {404: "not_found", 405: "method_not_allowed"}


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> Response:
    """Render framework HTTP errors under ``/api/`` like an :class:`ApiError`.

    Covers unknown routes (404) and wrong methods (405); other paths (static
    files) keep FastAPI's default body.
    """
    if not _is_api_path(request):
        return await http_exception_handler(request, exc)
    code = _HTTP_STATUS_CODES.get(exc.status_code)
    if code is None:
        code = "bad_request" if exc.status_code < 500 else "internal_error"
    error = ApiError(exc.status_code, code, request_lang(request), headers=getattr(exc, "headers", None))
    return await api_error_handler(request, error)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Path/query validation failures: 400 ``bad_request`` instead of FastAPI's
    422 with a list of English error objects."""
    logger.debug("Request validation failed for %s: %s", request.scope.get("path"), exc.errors())
    return await api_error_handler(request, ApiError(400, "bad_request", request_lang(request)))


async def unhandled_error_handler(request: Request, exc: Exception) -> Response:
    """Body of the 500 answer for an unhandled exception (Starlette logs the
    traceback and calls this)."""
    if not _is_api_path(request):
        return PlainTextResponse("Internal Server Error", status_code=500)
    return await api_error_handler(request, ApiError(500, "internal_error", request_lang(request)))


def install_exception_handlers(app) -> None:
    """Register the error renderers on a FastAPI app.

    After this every error response under ``/api/`` is
    ``{"detail": <localized str>, "code": <code>, ...}``.
    """
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)


# ---------------------------------------------------------------------------
# Client identity
# ---------------------------------------------------------------------------

def client_ip(request: Request) -> str:
    """Best-effort client IP used as the rate-limit identity.

    Proxy headers are honoured only when TRUST_PROXY_HEADERS=1: X-Real-IP
    first, then the right-most X-Forwarded-For entry (the one appended by
    the trusted proxy; left-most entries are client-controlled).
    """
    if config.TRUST_PROXY_HEADERS:
        real_ip = request.headers.get("x-real-ip", "").strip()
        if real_ip:
            return real_ip
        forwarded = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if forwarded:
            return forwarded[-1]
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def rate_limit_key(ip: str) -> str:
    """Rate-limit identity for an IP: IPv6 clients are grouped by /64
    (one host can trivially rotate addresses inside its prefix)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            return str(addr.ipv4_mapped)
        return str(ipaddress.IPv6Network((addr, 64), strict=False))
    return str(addr)


# ---------------------------------------------------------------------------
# Admin auth
# ---------------------------------------------------------------------------

def require_admin(request: Request) -> None:
    """FastAPI dependency guarding admin-only endpoints.

    ADMIN_TOKEN is read per request so it can be rotated without code
    changes; when it is unset every admin request is refused (403).
    """
    lang = request_lang(request)
    expected = os.environ.get("ADMIN_TOKEN", "").strip()
    if not expected:
        raise ApiError(403, "admin_disabled", lang)
    supplied = request.headers.get("x-admin-token", "")
    if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        logger.warning("Rejected admin request to %s from %s", request.url.path, client_ip(request))
        raise ApiError(403, "admin_forbidden", lang)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class SlidingWindowRateLimiter:
    """Allow at most ``limit`` hits per ``window_s`` seconds for each key."""

    def __init__(
        self,
        limit: int,
        window_s: float,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = 10000,
    ) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> int:
        """Record a hit for ``key``.

        Returns 0 when allowed, otherwise the number of seconds (>= 1)
        until the next hit would be allowed (the hit is not recorded).
        """
        now = self._clock()
        cutoff = now - self.window_s
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                if len(self._hits) >= self._max_keys:
                    self._sweep(cutoff)
                hits = self._hits[key] = deque()
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(1, math.ceil(hits[0] + self.window_s - now))
            hits.append(now)
            return 0

    def _sweep(self, cutoff: float) -> None:
        stale = [k for k, h in self._hits.items() if not h or h[-1] <= cutoff]
        for k in stale:
            del self._hits[k]
        # Still full (many active clients): drop the oldest-inserted keys.
        overflow = len(self._hits) - self._max_keys + 1
        for k in list(self._hits)[:max(0, overflow)]:
            del self._hits[k]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


# name → default "count/seconds"; override with env RATE_LIMIT_<NAME>
# ("0" or "off" disables that limit, a blank value means the default).
# "ai_coach" guards the LLM budget: cold reports while an API key is set.
# Without a key a report is rule-based and costs 1-400 ms of CPU, and one
# match has ten players in two languages: those get the far higher
# "ai_coach_rules" (under the LLM limit the 11th player of an hour was
# answered 429 with a retry time of an hour). Its window is ten minutes:
# looking through six matches in both languages used up 120 an hour, and
# the page then asked the visitor to come back in 54 minutes for a report
# that takes 0.1 s to write.
RATE_LIMIT_DEFAULTS: dict[str, str] = {
    "upload": "10/3600",
    "fetch": "30/600",
    "request_parse": "10/600",
    "ai_coach": "10/3600",
    "ai_coach_rules": "200/600",
}

_limiters: dict[str, SlidingWindowRateLimiter | None] = {}
_limiters_lock = threading.Lock()


def parse_rate_spec(spec: str) -> tuple[int, float] | None:
    """Parse "count/seconds" (e.g. "10/3600"); None means disabled."""
    spec = (spec or "").strip().lower()
    if spec in ("", "0", "off", "none", "disabled"):
        return None
    count_s, _, window_s = spec.partition("/")
    count = int(count_s)
    window = float(window_s) if window_s else 60.0
    if count <= 0 or window <= 0:
        return None
    return count, window


def get_rate_limiter(name: str) -> SlidingWindowRateLimiter | None:
    """Limiter for ``name`` built lazily from env (None when disabled)."""
    with _limiters_lock:
        if name in _limiters:
            return _limiters[name]
        env_name = f"RATE_LIMIT_{name.upper()}"
        default = RATE_LIMIT_DEFAULTS.get(name, "0")
        spec = os.environ.get(env_name, "").strip() or default
        try:
            parsed = parse_rate_spec(spec)
        except ValueError:
            logger.warning("Invalid %s=%r (expected count/seconds), using %s", env_name, spec, default)
            parsed = parse_rate_spec(default)
        limiter = SlidingWindowRateLimiter(*parsed) if parsed else None
        _limiters[name] = limiter
        return limiter


def reset_rate_limits() -> None:
    """Forget all limiter state and re-read env on next use (tests/admin)."""
    with _limiters_lock:
        _limiters.clear()


def enforce_rate_limit(request: Request, name: str) -> None:
    """Raise a 429 :class:`ApiError` when the client exceeded limit ``name``."""
    limiter = get_rate_limiter(name)
    if limiter is None:
        return
    ip = client_ip(request)
    retry_after = limiter.hit(rate_limit_key(ip))
    if retry_after:
        logger.info("Rate limit %s hit by %s (retry in %ds)", name, ip, retry_after)
        raise rate_limited_error(retry_after, request_lang(request))


# ---------------------------------------------------------------------------
# Daily quota for cold LLM-backed AI coach reports
# ---------------------------------------------------------------------------

class DailyQuota:
    """Global per-UTC-day counter persisted to a small JSON file."""

    def __init__(
        self,
        path: Path,
        limit_getter: Callable[[], int],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = path
        self._limit_getter = limit_getter
        self._clock = clock
        self._lock = threading.Lock()
        self._day: str | None = None
        self._count = 0
        self._loaded = False

    def _today(self) -> str:
        return datetime.fromtimestamp(self._clock(), tz=timezone.utc).strftime("%Y-%m-%d")

    def _seconds_until_reset(self) -> int:
        now = datetime.fromtimestamp(self._clock(), tz=timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return max(1, math.ceil((tomorrow - now).total_seconds()))

    def _load(self) -> None:
        self._loaded = True
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._day = str(data.get("date") or "")
            self._count = int(data.get("count") or 0)
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Ignoring unreadable AI coach usage file %s: %s", self._path, exc)

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"date": self._day, "count": self._count}), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError as exc:
            logger.warning("Failed to persist AI coach usage to %s: %s", self._path, exc)

    def _roll(self) -> None:
        if not self._loaded:
            self._load()
        today = self._today()
        if self._day != today:
            self._day = today
            self._count = 0

    def used(self) -> int:
        with self._lock:
            self._roll()
            return self._count

    def try_acquire(self) -> int:
        """Consume one unit. Returns 0 on success, else seconds until reset."""
        limit = self._limit_getter()
        with self._lock:
            self._roll()
            if limit > 0 and self._count >= limit:
                return self._seconds_until_reset()
            self._count += 1
            self._save()
            return 0

    def reset(self) -> None:
        with self._lock:
            self._loaded = True
            self._day = None
            self._count = 0


ai_coach_daily_quota = DailyQuota(
    config.AI_COACH_USAGE_FILE,
    limit_getter=lambda: config.AI_COACH_DAILY_LIMIT,
)


# ---------------------------------------------------------------------------
# ASGI middlewares
# ---------------------------------------------------------------------------

class RequestTooLarge(ApiError):
    """Raised from the receive channel once a request body exceeds its limit.

    The rest of such a body has no known end, so the connection is closed.
    """

    def __init__(self, lang: str, limit_bytes: int) -> None:
        super().__init__(
            413, "upload_too_large", lang,
            headers={"Connection": "close"},
            limit_mb=max(1, limit_bytes // (1024 * 1024)),
        )


async def _send_json(send: Send, status: int, payload: dict, close: bool = False) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    raw_headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
    ]
    if close:
        raw_headers.append((b"connection", b"close"))
    await send({"type": "http.response.start", "status": status, "headers": raw_headers})
    await send({"type": "http.response.body", "body": body})


class BodySizeLimitMiddleware:
    """Reject oversized request bodies on selected paths with 413.

    Checks Content-Length up front (before any byte is read) and also
    counts streamed bytes, so chunked uploads cannot bypass the limit.
    ``max_bytes`` is a callable so the limit can follow config changes.

    Browsers only read the response after they sent the whole body, and a
    server that closes the socket with unread data resets the connection,
    which destroys the answer (measured against uvicorn: the client gets
    "connection reset" instead of the JSON). The early 413 therefore keeps
    the connection open and lets the HTTP server discard the rest of the
    body; only a body declared larger than ``DISCARD_FACTOR`` times the
    limit is cut off with ``Connection: close``.
    """

    # Allowance for multipart boundaries / part headers around the file.
    MULTIPART_OVERHEAD = 64 * 1024
    # An over-limit body up to this multiple of the limit is still received
    # (and thrown away) so the client can read the 413.
    DISCARD_FACTOR = 2

    def __init__(self, app: ASGIApp, paths: tuple[str, ...], max_bytes: Callable[[], int]) -> None:
        self.app = app
        self.paths = paths
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") not in self.paths:
            await self.app(scope, receive, send)
            return

        file_limit = self.max_bytes()
        limit = file_limit + self.MULTIPART_OVERHEAD
        lang = lang_from_scope(scope)
        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = -1
            if declared > limit:
                exc = RequestTooLarge(lang, file_limit)
                await _send_json(send, 413, error_payload(exc), close=declared > limit * self.DISCARD_FACTOR)
                return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            msg = await receive()
            if msg["type"] == "http.request":
                received += len(msg.get("body", b""))
                if received > limit:
                    raise RequestTooLarge(lang, file_limit)
            return msg

        async def tracking_send(msg: Message) -> None:
            nonlocal response_started
            if msg["type"] == "http.response.start":
                response_started = True
            await send(msg)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except RequestTooLarge as exc:
            # Normally rendered by the ApiError handler; this is the fallback
            # when the exception escapes the app before a response started.
            if response_started:
                raise
            await _send_json(send, 413, error_payload(exc), close=True)


class SecurityHeadersMiddleware:
    """Add baseline security headers (and static cache hints) to responses."""

    HEADERS = (
        (b"x-content-type-options", b"nosniff"),
        (b"referrer-policy", b"strict-origin-when-cross-origin"),
        (b"x-frame-options", b"SAMEORIGIN"),
    )

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        cache_control: bytes | None = None
        if path.startswith("/static/assets/"):
            cache_control = b"public, max-age=86400"
        elif not path.startswith("/api/"):
            # HTML/JS/CSS/JSON data: always revalidate (ETag/Last-Modified)
            # so deploys are picked up immediately.
            cache_control = b"no-cache"

        async def send_with_headers(msg: Message) -> None:
            if msg["type"] == "http.response.start":
                headers = list(msg.get("headers", []))
                present = {k.lower() for k, _ in headers}
                for key, value in self.HEADERS:
                    if key not in present:
                        headers.append((key, value))
                if cache_control is not None and b"cache-control" not in present:
                    headers.append((b"cache-control", cache_control))
                msg = {**msg, "headers": headers}
            await send(msg)

        await self.app(scope, receive, send_with_headers)
