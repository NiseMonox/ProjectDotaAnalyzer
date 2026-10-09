"""API tests for routes.py: demo/match/fetch/upload/admin endpoints.

Network, Java and the LLM are never touched: OpenDota and the replay parser
are replaced with in-process fakes, and every disk path points at tmp_path.
"""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import hashlib
import sys
import types
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.app import config, routes, security, storage
from backend.app import parser as parser_mod
from backend.app.main import app
from backend.app.models import GoldXpSnapshot, MatchData, PlayerData

DEM = b"PBDEMS2\x00"
REAL_GET_REPLAY_PARSER = routes._get_replay_parser


class FakeParserUnavailable(RuntimeError):
    pass


class FakeReplayParseError(RuntimeError):
    pass


class FakeOpenDotaError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status


def make_player(slot: int) -> PlayerData:
    return PlayerData(
        player_slot=slot, hero_id=slot + 1, hero_name=f"Hero{slot}", hero_slug=f"hero{slot}",
        player_name=f"p{slot}", team="radiant" if slot < 128 else "dire", kills=1, deaths=1,
        assists=1, level=10, last_hits=10, denies=1, gold_per_min=400, xp_per_min=500,
        hero_damage=1000, tower_damage=100, hero_healing=0, net_worth=5000, items=[],
        backpack=[], final_items=[], skills=[], wards=[], ability_events=[],
        gold_reasons={"13": 100},
    )


def make_match(match_id: int = 7, parsed: bool = True) -> MatchData:
    return MatchData(
        match_id=match_id, duration=1200, radiant_win=True, game_mode="All Pick",
        start_time=0, radiant_score=10, dire_score=5,
        players=[make_player(0), make_player(128)], wards=[], smokes=[], teamfights=[],
        ability_events=[],
        gold_xp_timeline=[GoldXpSnapshot(time=60, radiant_gold=1, dire_gold=1, radiant_xp=1, dire_xp=1)]
        if parsed else [],
    )


class FakeReplayParser:
    def __init__(self, state):
        self.state = state

    def name(self) -> str:
        return "clarity"

    async def parse(self, path: str) -> MatchData:
        self.state.parse_calls.append(path)
        if self.state.parse_error is not None:
            raise self.state.parse_error
        return make_match(self.state.next_match_id)


@pytest.fixture
def server(tmp_path, monkeypatch):
    state = SimpleNamespace(
        parse_calls=[], parse_error=None, next_match_id=1001,
        od_calls=[], od_errors={}, od_unparsed=set(), od_request_calls=[],
    )
    replays = tmp_path / "replays"
    for name, value in {
        "REPLAY_DIR": replays,
        "COACH_CACHE_DIR": tmp_path / "coaching",
        "MATCH_CACHE_DIR": tmp_path / "matches",
        "DEFAULT_REPLAY": replays / "no_such_demo.dem",
    }.items():
        value.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(config, name, value)
    for d in (replays, tmp_path / "coaching", tmp_path / "matches"):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(routes, "match_store",
                        storage.MatchStore(tmp_path / "matches", lambda: config.MATCH_CACHE_MAX_FILES))
    monkeypatch.setattr(routes, "ai_coach_daily_quota",
                        security.DailyQuota(tmp_path / "usage.json", lambda: config.AI_COACH_DAILY_LIMIT))
    for name in security.RATE_LIMIT_DEFAULTS:
        monkeypatch.setenv(f"RATE_LIMIT_{name.upper()}", "0")
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    security.reset_rate_limits()
    for cache in (routes._matches, routes._coach_report_cache, routes._inflight, routes._coach_jobs):
        cache.clear()

    monkeypatch.setattr(parser_mod, "ParserUnavailableError", FakeParserUnavailable, raising=False)
    monkeypatch.setattr(parser_mod, "ReplayParseError", FakeReplayParseError, raising=False)
    monkeypatch.setattr(parser_mod, "OpenDotaError", FakeOpenDotaError, raising=False)

    async def fake_get_replay_parser():
        return FakeReplayParser(state)

    monkeypatch.setattr(routes, "_get_replay_parser", fake_get_replay_parser)

    class FakeOpenDota:
        def name(self):
            return "opendota"

        async def parse(self, mid_str):
            mid = int(mid_str)
            state.od_calls.append(mid)
            if mid in state.od_errors:
                raise state.od_errors[mid]
            return make_match(mid, parsed=mid not in state.od_unparsed)

        def request_parse(self, mid):
            state.od_request_calls.append(mid)
            return {"job": {"jobId": 99}}

    monkeypatch.setattr(parser_mod, "OpenDotaParser", FakeOpenDota)

    with TestClient(app) as client:
        state.client = client
        state.tmp = tmp_path
        state.replays = replays
        yield state

    security.reset_rate_limits()
    for cache in (routes._matches, routes._coach_report_cache, routes._inflight, routes._coach_jobs):
        cache.clear()


def upload(client, data: bytes, filename: str = "match.dem", **kw):
    return client.post("/api/upload", files={"file": (filename, data, "application/octet-stream")}, **kw)


def replay_bytes(tag: str = "a", size: int = 4096) -> bytes:
    return DEM + (tag.encode() * size)[:size]


# ---------------------------------------------------------------------------
# Demo / match / sections
# ---------------------------------------------------------------------------

def test_demo_match_and_sections(server):
    c = server.client
    demo = c.get("/api/demo", headers={"Accept-Encoding": "gzip"})
    assert demo.status_code == 200
    assert demo.headers.get("content-encoding") == "gzip"
    body = demo.json()
    assert body["parse_source"] == "mock"
    assert body["_match_key"] == "demo"
    assert "match_overview" in body["analysis"]

    assert c.get("/api/match/demo").json()["analysis"]["match_overview"] == body["analysis"]["match_overview"]
    laning = c.get("/api/match/demo/laning").json()
    assert set(laning) == {"laning"}

    bad = c.get("/api/match/demo/nope", headers={"Accept-Language": "en"})
    assert bad.status_code == 400
    assert bad.json()["code"] == "invalid_section"
    assert "laning" in bad.json()["valid"]

    missing = c.get("/api/match/unknown_key")
    assert missing.status_code == 404
    assert missing.json()["code"] == "match_not_found"

    # Mock demo data is never persisted.
    assert not list((server.tmp / "matches").iterdir())


def test_unknown_match_keys_are_safe(server):
    c = server.client
    for key in ("..", "...", "a" * 300, "x;rm -rf", "%2e%2e%2fetc"):
        assert c.get(f"/api/match/{key}").status_code == 404
    assert not list((server.tmp / "matches").iterdir())


def test_security_headers_present(server):
    resp = server.client.get("/api/match/missing")
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["referrer-policy"] == "strict-origin-when-cross-origin"


# ---------------------------------------------------------------------------
# OpenDota fetch / request-parse / disk store
# ---------------------------------------------------------------------------

def test_fetch_parsed_match_is_persisted_and_reloaded_after_eviction(server, monkeypatch):
    c = server.client
    monkeypatch.setattr(config, "MATCH_MEMORY_CACHE_SIZE", 1)
    first = c.get("/api/fetch/111")
    assert first.status_code == 200
    assert first.json()["has_parsed_data"] is True
    assert first.json()["_match_key"] == "opendota_111"
    assert routes.match_store.path_for("opendota_111").exists()

    assert c.get("/api/fetch/222").status_code == 200   # evicts 111 from memory
    assert "opendota_111" not in routes._matches

    again = c.get("/api/match/opendota_111")
    assert again.status_code == 200
    assert again.json()["analysis"]["match_overview"]["match_id"] == 111
    assert again.json()["has_parsed_data"] is True
    assert server.od_calls == [111, 222]  # reloaded from disk, not refetched

    assert c.get("/api/fetch/111").status_code == 200
    assert server.od_calls == [111, 222]


def test_fetch_unparsed_match_is_short_lived(server, monkeypatch):
    c = server.client
    server.od_unparsed.add(333)
    r = c.get("/api/fetch/333")
    assert r.status_code == 200 and r.json()["has_parsed_data"] is False
    c.get("/api/fetch/333")
    assert server.od_calls == [333]              # within TTL → cached
    assert not routes.match_store.path_for("opendota_333").exists()

    monkeypatch.setattr(config, "OPENDOTA_UNPARSED_TTL_S", 0)
    routes._matches.clear()
    c.get("/api/fetch/333")
    c.get("/api/fetch/333")
    assert server.od_calls == [333, 333, 333]    # no TTL → refetch every time


def test_request_parse_clears_memory_and_disk(server):
    c = server.client
    assert c.get("/api/fetch/444").status_code == 200
    assert routes.match_store.path_for("opendota_444").exists()
    r = c.post("/api/request-parse/444")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "job": {"jobId": 99}}
    assert server.od_request_calls == [444]
    assert "opendota_444" not in routes._matches
    assert not routes.match_store.path_for("opendota_444").exists()
    c.get("/api/fetch/444")
    assert server.od_calls == [444, 444]


@pytest.mark.parametrize("error,status,code", [
    (FakeOpenDotaError("Not Found", status=404), 404, "opendota_not_found"),
    (FakeOpenDotaError("upstream 503", status=503), 502, "opendota_unavailable"),
    (FakeOpenDotaError("timed out"), 502, "opendota_unavailable"),
    (RuntimeError("secret internals /etc/passwd"), 502, "opendota_unavailable"),
])
def test_fetch_errors_are_mapped_and_user_safe(server, error, status, code):
    server.od_errors[555] = error
    r = server.client.get("/api/fetch/555", headers={"Accept-Language": "en"})
    assert r.status_code == status
    assert r.json()["code"] == code
    assert str(error) not in r.json()["detail"]
    assert "opendota_555" not in routes._matches


@pytest.mark.parametrize("raw", ["abc", "0", "-5", "1" * 25])
def test_fetch_rejects_invalid_match_ids(server, raw):
    r = server.client.get(f"/api/fetch/{raw}")
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_match_id"
    assert server.od_calls == []


def test_fetch_and_request_parse_rate_limits(server, monkeypatch):
    c = server.client
    monkeypatch.setenv("RATE_LIMIT_FETCH", "1/600")
    monkeypatch.setenv("RATE_LIMIT_REQUEST_PARSE", "1/600")
    security.reset_rate_limits()
    assert c.get("/api/fetch/1").status_code == 200
    assert c.get("/api/fetch/1").status_code == 200   # cache hit is free
    limited = c.get("/api/fetch/2")
    assert limited.status_code == 429
    assert limited.json()["code"] == "rate_limited"
    assert limited.json()["retry_after"] >= 1
    assert limited.headers["Retry-After"] == str(limited.json()["retry_after"])
    assert server.od_calls == [1]

    assert c.post("/api/request-parse/1").status_code == 200
    assert c.post("/api/request-parse/1").status_code == 429


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def test_upload_stores_by_content_hash_and_reuses_cache(server):
    c = server.client
    data = replay_bytes("x")
    r = upload(c, data, "8713267082.dem")
    assert r.status_code == 200, r.text
    body = r.json()
    digest = hashlib.sha256(data).hexdigest()[:32]
    assert body["file_key"] == f"replay_{digest}"
    assert body["match_id"] == 1001
    assert body["parse_source"] == "clarity"
    assert body["cached"] is False
    assert (server.replays / f"{digest}.dem").read_bytes() == data
    assert not (server.replays / "8713267082.dem").exists()

    analysis = c.get(f"/api/match/{body['file_key']}")
    assert analysis.status_code == 200
    assert analysis.json()["analysis"]["match_overview"]["match_id"] == 1001

    again = upload(c, data, "renamed.dem")
    assert again.status_code == 200
    assert again.json()["file_key"] == body["file_key"]
    assert again.json()["cached"] is True
    assert len(server.parse_calls) == 1

    # Different content with the same client filename does not collide.
    other = upload(c, replay_bytes("y"), "8713267082.dem")
    assert other.json()["file_key"] != body["file_key"]
    assert len(server.parse_calls) == 2


def test_upload_ignores_client_path_components(server):
    r = upload(server.client, replay_bytes(), "../../../../tmp/evil.dem")
    assert r.status_code == 200
    names = sorted(p.name for p in server.replays.iterdir() if p.is_file())
    assert len(names) == 1 and names[0].endswith(".dem") and "evil" not in names[0]
    assert not (server.tmp / "evil.dem").exists()


@pytest.mark.parametrize("data,filename,code", [
    (b"HL2DEMO\x00" + b"x" * 100, "old.dem", "replay_invalid"),
    (b"BZh91AY&SY" + b"x" * 100, "match.dem", "replay_compressed"),
    (b"", "empty.dem", "replay_invalid"),
    (DEM + b"x" * 100, "match.txt", "upload_bad_extension"),
])
def test_upload_rejects_invalid_files(server, data, filename, code):
    r = upload(server.client, data, filename)
    assert r.status_code == 400
    assert r.json()["code"] == code
    assert server.parse_calls == []
    assert not [p for p in server.replays.rglob("*") if p.is_file()]


def test_upload_requires_file_field(server):
    r = server.client.post("/api/upload", files={"other": ("a.dem", DEM)})
    assert r.status_code == 400
    assert r.json()["code"] == "upload_missing_file"
    r = server.client.post("/api/upload", data={"x": "1"})
    assert r.status_code == 400


def test_upload_size_limit(server, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1024)
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    c = server.client
    # Over the file limit but within multipart slack: rejected while storing.
    r = upload(c, replay_bytes(size=2048))
    assert r.status_code == 413
    assert r.json()["code"] == "upload_too_large"
    # Far over the limit: rejected by the middleware from Content-Length.
    r = upload(c, replay_bytes(size=200 * 1024), headers={"Accept-Language": "en"})
    assert r.status_code == 413
    assert "1 MB" in r.json()["detail"]
    assert server.parse_calls == []
    assert not [p for p in server.replays.rglob("*") if p.is_file()]
    assert upload(c, replay_bytes(size=512)).status_code == 200


@pytest.mark.parametrize("error,status,code", [
    (FakeReplayParseError("Corrupted demo"), 422, "replay_parse_failed"),
    (FakeParserUnavailable("no java"), 503, "parser_unavailable"),
    (RuntimeError("Exception in thread main java.lang.OutOfMemoryError"), 500, "internal_error"),
])
def test_upload_parse_failures_delete_file_and_hide_details(server, error, status, code):
    server.parse_error = error
    r = upload(server.client, replay_bytes())
    assert r.status_code == status
    assert r.json()["code"] == code
    assert str(error) not in r.text
    assert not [p for p in server.replays.rglob("*.dem")]
    assert routes._matches == {}


def test_upload_without_parser_returns_503(server, monkeypatch):
    """The real parser lookup never falls back to mock data for uploads."""
    monkeypatch.setattr(routes, "_get_replay_parser", REAL_GET_REPLAY_PARSER)

    async def unavailable():
        return False

    monkeypatch.setattr(parser_mod, "check_clarity_available", unavailable, raising=False)
    r = upload(server.client, replay_bytes())
    assert r.status_code == 503
    assert r.json()["code"] == "parser_unavailable"
    assert not list(server.replays.glob("*.dem"))

    # Legacy parser API (no check function, get_parser may return mock).
    monkeypatch.delattr(parser_mod, "check_clarity_available", raising=False)
    monkeypatch.setattr(parser_mod, "get_parser", lambda kind="auto": parser_mod.MockParser())
    r = upload(server.client, replay_bytes())
    assert r.status_code == 503


def test_upload_retention_keeps_newest_uploads_only(server, monkeypatch):
    monkeypatch.setattr(config, "REPLAY_MAX_FILES", 2)
    manual = server.replays / "manual_12345.dem"
    manual.write_bytes(DEM)
    keys = []
    for tag in "abc":
        r = upload(server.client, replay_bytes(tag))
        assert r.status_code == 200
        keys.append(r.json()["file_key"])
        os.utime(server.replays / f"{keys[-1][len('replay_'):]}.dem", (len(keys), len(keys)))
    remaining = sorted(p.name for p in server.replays.glob("*.dem"))
    assert manual.name in remaining
    assert len(remaining) == 3  # 2 uploads + the manual file
    assert f"{keys[0][len('replay_'):]}.dem" not in remaining


def test_upload_rate_limit_and_busy(server, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_UPLOAD", "1/3600")
    security.reset_rate_limits()
    assert upload(server.client, replay_bytes("a")).status_code == 200
    limited = upload(server.client, replay_bytes("b"))
    assert limited.status_code == 429
    assert "Retry-After" in limited.headers

    monkeypatch.setenv("RATE_LIMIT_UPLOAD", "0")
    security.reset_rate_limits()
    monkeypatch.setattr(routes, "_uploads_in_progress", config.UPLOAD_MAX_CONCURRENCY)
    busy = upload(server.client, replay_bytes("c"))
    assert busy.status_code == 503
    assert busy.json()["code"] == "server_busy"


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

ADMIN_ENDPOINTS = [
    ("get", "/api/admin/gamedata-status"),
    ("post", "/api/admin/update-gamedata"),
    ("get", "/api/admin/status"),
    ("get", "/api/parse-local/8713267082.dem"),
]


@pytest.mark.parametrize("method,url", ADMIN_ENDPOINTS)
def test_admin_endpoints_disabled_without_token(server, method, url):
    r = getattr(server.client, method)(url, headers={"X-Admin-Token": ""})
    assert r.status_code == 403
    assert r.json()["code"] == "admin_disabled"


@pytest.mark.parametrize("method,url", ADMIN_ENDPOINTS)
def test_admin_endpoints_reject_wrong_token(server, monkeypatch, method, url):
    monkeypatch.setenv("ADMIN_TOKEN", "correct-horse")
    r = getattr(server.client, method)(url, headers={"X-Admin-Token": "nope"})
    assert r.status_code == 403
    assert r.json()["code"] == "admin_forbidden"


@pytest.fixture
def fake_data_manager(monkeypatch):
    mod = types.ModuleType("backend.app.ai.data_manager")
    mod.get_meta = lambda: {"patch_version": "7.40", "hero_count": 126}
    mod.get_current_patch = lambda: "7.40b"

    def fetch_and_save_gamedata(force=False):
        raise RuntimeError("HTTP 500 from https://api.opendota.com secret")

    mod.fetch_and_save_gamedata = fetch_and_save_gamedata
    monkeypatch.setitem(sys.modules, "backend.app.ai.data_manager", mod)
    return mod


def test_admin_gamedata_endpoints(server, monkeypatch, fake_data_manager):
    monkeypatch.setenv("ADMIN_TOKEN", "tok")
    hdr = {"X-Admin-Token": "tok"}
    status = server.client.get("/api/admin/gamedata-status", headers=hdr)
    assert status.status_code == 200
    assert status.json() == {"patch_version": "7.40", "hero_count": 126, "current_patch": "7.40b"}

    failed = server.client.post("/api/admin/update-gamedata", headers=hdr)
    assert failed.status_code == 502
    assert failed.json()["code"] == "gamedata_update_failed"
    assert "secret" not in failed.text

    del fake_data_manager.get_meta
    fallback = server.client.get("/api/admin/gamedata-status", headers=hdr)
    assert fallback.json() == {"status": "module_unavailable", "patch_version": "7.40b"}

    ops = server.client.get("/api/admin/status", headers=hdr).json()
    assert ops["ai_coach_daily"]["limit"] == config.AI_COACH_DAILY_LIMIT


def test_parse_local(server, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "tok")
    hdr = {"X-Admin-Token": "tok"}
    c = server.client
    assert c.get("/api/parse-local/missing.dem", headers=hdr).json()["code"] == "replay_not_found"
    assert c.get("/api/parse-local/.hidden.dem", headers=hdr).json()["code"] == "invalid_filename"
    assert c.get("/api/parse-local/notes.txt", headers=hdr).status_code == 400
    assert c.get("/api/parse-local/..%2F..%2Fetc%2Fpasswd.dem", headers=hdr).status_code in (400, 404)

    (server.replays / "local_game.dem").write_bytes(DEM)
    r = c.get("/api/parse-local/local_game.dem", headers=hdr)
    assert r.status_code == 200
    assert r.json()["filename"] == "local_game.dem"
    assert r.json()["_match_key"] == "local_local_game.dem"
    c.get("/api/parse-local/local_game.dem", headers=hdr)
    assert len(server.parse_calls) == 1

    # Non-numeric name: a parse failure is reported, not an OpenDota lookup.
    (server.replays / "other.dem").write_bytes(DEM)
    server.parse_error = FakeReplayParseError("bad")
    r = c.get("/api/parse-local/other.dem", headers=hdr)
    assert r.status_code == 422

    # Numeric name: falls back to OpenDota by match ID.
    (server.replays / "98765.dem").write_bytes(DEM)
    r = c.get("/api/parse-local/98765.dem", headers=hdr)
    assert r.status_code == 200 and r.json()["parse_source"] == "opendota"
    assert server.od_calls == [98765]


# ---------------------------------------------------------------------------
# Health / capabilities / serialization
# ---------------------------------------------------------------------------

def test_health_and_capabilities(server, monkeypatch):
    c = server.client
    assert c.get("/api/health").json() == {"status": "ok"}
    caps = c.get("/api/capabilities").json()
    assert caps["replay_parser"] is True  # fake parser in this fixture
    assert caps["max_upload_mb"] == config.MAX_UPLOAD_MB
    assert caps["ai_coach_llm"] is False
    monkeypatch.setattr(config, "AI_COACH_DAILY_LIMIT", 0)
    assert c.get("/api/capabilities").json()["ai_coach_daily_remaining"] is None

    async def unavailable():
        raise FakeParserUnavailable("no java")

    monkeypatch.setattr(routes, "_get_replay_parser", unavailable)
    assert c.get("/api/capabilities").json()["replay_parser"] is False


def test_match_payload_served_with_and_without_gzip(server):
    c = server.client
    c.get("/api/fetch/1234")
    plain = c.get("/api/match/opendota_1234", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers
    zipped = c.get("/api/match/opendota_1234", headers={"Accept-Encoding": "gzip"})
    assert zipped.headers["content-encoding"] == "gzip"
    assert plain.json() == zipped.json()
    assert plain.headers["vary"] == "Accept-Encoding"


def test_json_bytes_replaces_non_finite_numbers():
    import json as _json

    payload = {"a": float("nan"), "b": [1.5, float("inf")], "c": ("x", -float("inf")), "d": "中文"}
    decoded = _json.loads(routes._json_bytes(payload))
    assert decoded == {"a": None, "b": [1.5, None], "c": ["x", None], "d": "中文"}
