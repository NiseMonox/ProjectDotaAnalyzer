"""Unit tests for backend.app.security (rate limiting, quota, IPs, messages)."""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import asyncio
import json
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.requests import Request as StarletteRequest

from backend.app import config, security
from backend.app.security import (
    ApiError,
    BodySizeLimitMiddleware,
    DailyQuota,
    SlidingWindowRateLimiter,
    message,
    normalize_lang,
    parse_rate_spec,
)


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def make_request(headers=None, client=("10.0.0.1", 1234), query=b"") -> StarletteRequest:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/x",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "query_string": query,
        "client": client,
    }
    return StarletteRequest(scope)


def test_sliding_window_limits_and_recovers():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(2, 60, clock=clock)
    assert limiter.hit("a") == 0
    assert limiter.hit("a") == 0
    retry = limiter.hit("a")
    assert retry == 60
    assert limiter.hit("b") == 0  # other keys unaffected
    clock.now += 30
    assert limiter.hit("a") == 30
    clock.now += 30.5
    assert limiter.hit("a") == 0


def test_sliding_window_sweeps_stale_keys():
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(1, 10, clock=clock, max_keys=3)
    for key in "abc":
        limiter.hit(key)
    clock.now += 11
    limiter.hit("d")
    assert set(limiter._hits) == {"d"}


@pytest.mark.parametrize("spec,expected", [
    ("10/3600", (10, 3600.0)),
    ("5", (5, 60.0)),
    ("0", None),
    ("off", None),
    ("", None),
])
def test_parse_rate_spec(spec, expected):
    assert parse_rate_spec(spec) == expected


def test_rate_limiter_env_override(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_UPLOAD", "off")
    monkeypatch.setenv("RATE_LIMIT_FETCH", "garbage")
    security.reset_rate_limits()
    try:
        assert security.get_rate_limiter("upload") is None
        fetch = security.get_rate_limiter("fetch")  # invalid → default
        assert fetch is not None and fetch.limit == 30
    finally:
        security.reset_rate_limits()


def test_blank_rate_limit_means_default_not_disabled(monkeypatch):
    """`RATE_LIMIT_UPLOAD=` in .env must not switch the limit off."""
    monkeypatch.setenv("RATE_LIMIT_UPLOAD", "")
    monkeypatch.setenv("RATE_LIMIT_FETCH", "   ")
    monkeypatch.setenv("RATE_LIMIT_AI_COACH", "0")
    monkeypatch.delenv("RATE_LIMIT_REQUEST_PARSE", raising=False)
    security.reset_rate_limits()
    try:
        upload = security.get_rate_limiter("upload")
        assert upload is not None and (upload.limit, upload.window_s) == (10, 3600.0)
        assert security.get_rate_limiter("fetch").limit == 30
        assert security.get_rate_limiter("request_parse").limit == 10
        assert security.get_rate_limiter("ai_coach") is None      # 0 / off still disables
    finally:
        security.reset_rate_limits()


def test_enforce_rate_limit_raises_429(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_FETCH", "1/100")
    security.reset_rate_limits()
    try:
        req = make_request({"accept-language": "en-US,en;q=0.9"})
        security.enforce_rate_limit(req, "fetch")
        with pytest.raises(ApiError) as info:
            security.enforce_rate_limit(req, "fetch")
        exc = info.value
        assert exc.status_code == 429
        assert exc.code == "rate_limited"
        assert exc.extra["retry_after"] >= 1
        assert exc.headers["Retry-After"] == str(exc.extra["retry_after"])
        assert "retry" in exc.detail.lower()
    finally:
        security.reset_rate_limits()


def test_daily_quota_counts_persists_and_rolls_over(tmp_path):
    clock = FakeClock(datetime(2026, 1, 1, 23, 0, tzinfo=timezone.utc).timestamp())
    path = tmp_path / "usage.json"
    quota = DailyQuota(path, limit_getter=lambda: 2, clock=clock)
    assert quota.try_acquire() == 0
    assert quota.try_acquire() == 0
    wait = quota.try_acquire()
    assert wait == 3600  # until UTC midnight
    assert json.loads(path.read_text(encoding="utf-8")) == {"date": "2026-01-01", "count": 2}

    # A fresh instance (server restart) keeps today's count.
    again = DailyQuota(path, limit_getter=lambda: 2, clock=clock)
    assert again.try_acquire() > 0

    clock.now += 3600
    assert again.try_acquire() == 0
    assert again.used() == 1


def test_daily_quota_is_persisted_while_the_usage_file_is_in_use(tmp_path, monkeypatch):
    """Windows refuses to replace a file that is open (a virus scanner looking at it): the write
    waits a moment instead of losing the count."""
    from backend.app import osutil

    real_replace, refusals = os.replace, [2]

    def replace(src, dst):
        if refusals[0] > 0:
            refusals[0] -= 1
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        real_replace(src, dst)

    monkeypatch.setattr(osutil, "IS_WINDOWS", True)
    monkeypatch.setattr(osutil.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(os, "replace", replace)
    path = tmp_path / "usage.json"
    clock = FakeClock(datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp())
    assert DailyQuota(path, limit_getter=lambda: 5, clock=clock).try_acquire() == 0
    assert refusals == [0]
    assert json.loads(path.read_text(encoding="utf-8")) == {"date": "2026-01-01", "count": 1}


def test_daily_quota_zero_means_unlimited(tmp_path):
    quota = DailyQuota(tmp_path / "u.json", limit_getter=lambda: 0)
    assert all(quota.try_acquire() == 0 for _ in range(50))


def test_client_ip_ignores_proxy_headers_unless_trusted(monkeypatch):
    headers = {"x-forwarded-for": "1.1.1.1, 2.2.2.2", "x-real-ip": "3.3.3.3"}
    monkeypatch.setattr(config, "TRUST_PROXY_HEADERS", False)
    assert security.client_ip(make_request(headers)) == "10.0.0.1"
    monkeypatch.setattr(config, "TRUST_PROXY_HEADERS", True)
    assert security.client_ip(make_request(headers)) == "3.3.3.3"
    assert security.client_ip(make_request({"x-forwarded-for": "1.1.1.1, 2.2.2.2"})) == "2.2.2.2"
    assert security.client_ip(make_request({}, client=None)) == "unknown"


def test_language_selection():
    assert normalize_lang("zh-CN,zh;q=0.9") == "zh"
    assert normalize_lang("en-GB") == "en"
    assert normalize_lang("fr") is None
    assert security.request_lang(make_request({"accept-language": "en"}, query=b"lang=zh")) == "zh"
    assert security.request_lang(make_request({"accept-language": "en-US"})) == "en"
    assert security.request_lang(make_request({})) == "zh"


def test_every_message_has_both_languages():
    for code, entry in security.MESSAGES.items():
        assert set(entry) == {"zh", "en"}, code
    assert message("upload_too_large", "en", limit_mb=300) == "Replay file is too large (limit 300 MB)."
    assert message("no_such_code", "en") == "Internal server error."


def test_require_admin(monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    with pytest.raises(ApiError) as info:
        security.require_admin(make_request({"x-admin-token": "anything"}))
    assert info.value.status_code == 403 and info.value.code == "admin_disabled"

    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    with pytest.raises(ApiError) as info:
        security.require_admin(make_request({"x-admin-token": "wrong"}))
    assert info.value.code == "admin_forbidden"
    with pytest.raises(ApiError):
        security.require_admin(make_request({}))
    security.require_admin(make_request({"x-admin-token": "s3cret"}))


def _body_echo_app(install_handler: bool) -> FastAPI:
    app = FastAPI()
    if install_handler:
        security.install_exception_handlers(app)

    @app.post("/api/upload")
    async def upload(request: Request):
        body = await request.body()
        return {"size": len(body)}

    app.add_middleware(BodySizeLimitMiddleware, paths=("/api/upload",), max_bytes=lambda: 10)
    return app


@pytest.mark.parametrize("install_handler", [True, False])
def test_body_limit_rejects_declared_content_length(install_handler):
    client = TestClient(_body_echo_app(install_handler))
    big = b"x" * (BodySizeLimitMiddleware.MULTIPART_OVERHEAD + 11)
    resp = client.post("/api/upload", content=big)
    assert resp.status_code == 413
    assert resp.json()["code"] == "upload_too_large"
    ok = client.post("/api/upload", content=b"small")
    assert ok.status_code == 200 and ok.json() == {"size": 5}


def _declared_upload(app, declared: int) -> dict:
    """Send only the headers of an upload declaring ``declared`` bytes; return the response start."""
    sent: list[dict] = []

    async def run():
        async def receive():
            raise AssertionError("an over-limit body must be refused before it is read")

        async def send(msg):
            sent.append(msg)

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "path": "/api/upload", "raw_path": b"/api/upload",
            "root_path": "", "scheme": "http", "query_string": b"lang=en",
            "headers": [(b"host", b"test"), (b"content-length", str(declared).encode())],
            "client": ("1.2.3.4", 1), "server": ("test", 80),
        }
        await app(scope, receive, send)

    asyncio.run(run())
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = json.loads(b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))
    return {"status": start["status"], "headers": dict(start["headers"]), "body": body}


def test_early_413_keeps_the_connection_so_browsers_can_read_it():
    """Closing with the body unread resets the connection and the client never sees the JSON.

    (Measured against uvicorn: curl reports "connection reset by peer".) The
    connection is only closed for bodies far beyond the limit.
    """
    app = _body_echo_app(install_handler=True)
    limit = 10 + BodySizeLimitMiddleware.MULTIPART_OVERHEAD
    slightly_over = _declared_upload(app, limit + 1)
    assert slightly_over["status"] == 413
    assert slightly_over["body"]["code"] == "upload_too_large"
    assert b"connection" not in slightly_over["headers"]
    at_bound = _declared_upload(app, limit * BodySizeLimitMiddleware.DISCARD_FACTOR)
    assert b"connection" not in at_bound["headers"]

    huge = _declared_upload(app, limit * BodySizeLimitMiddleware.DISCARD_FACTOR + 1)
    assert huge["status"] == 413 and huge["body"]["code"] == "upload_too_large"
    assert huge["headers"][b"connection"] == b"close"


@pytest.mark.parametrize("install_handler", [True, False])
def test_body_limit_counts_streamed_chunks(install_handler):
    """Chunked bodies without Content-Length are cut off once over the limit."""
    app = _body_echo_app(install_handler)
    limit = BodySizeLimitMiddleware.MULTIPART_OVERHEAD + 10
    chunks = [b"y" * 4096 for _ in range(limit // 4096 + 2)]
    sent: list[dict] = []

    async def run():
        queue = list(chunks)

        async def receive():
            if not queue:
                return {"type": "http.disconnect"}
            body = queue.pop(0)
            return {"type": "http.request", "body": body, "more_body": bool(queue)}

        async def send(msg):
            sent.append(msg)

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "path": "/api/upload", "raw_path": b"/api/upload",
            "root_path": "", "scheme": "http", "query_string": b"",
            "headers": [(b"host", b"test")], "client": ("1.2.3.4", 1), "server": ("test", 80),
        }
        await app(scope, receive, send)

    asyncio.run(run())
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 413
    body = json.loads(b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))
    assert "too large" in body["detail"] or "过大" in body["detail"]
    if install_handler:
        assert body["code"] == "upload_too_large"
    # A body without a declared end is cut off, not drained.
    assert (b"connection", b"close") in [(k.lower(), v.lower()) for k, v in start["headers"]]


@pytest.mark.parametrize("ip,key", [
    ("203.0.113.7", "203.0.113.7"),
    ("2001:db8:1:2:aaaa::1", "2001:db8:1:2::/64"),
    ("2001:db8:1:2:bbbb::9", "2001:db8:1:2::/64"),
    ("::ffff:198.51.100.4", "198.51.100.4"),
    ("testclient", "testclient"),
])
def test_rate_limit_key_groups_ipv6_prefixes(ip, key):
    assert security.rate_limit_key(ip) == key
