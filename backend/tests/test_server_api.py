"""API tests for routes.py: demo/match/fetch/upload/admin endpoints.

Network, Java and the LLM are never touched: OpenDota and the replay parser
are replaced with in-process fakes, and every disk path points at tmp_path.
"""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import asyncio
import bz2
import gzip
import hashlib
import json
import sys
import time
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
    for cache in (routes._matches, routes._coach_report_cache, routes._inflight, routes._coach_jobs,
                  routes._uploads_by_client):
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
    for cache in (routes._matches, routes._coach_report_cache, routes._inflight, routes._coach_jobs,
                  routes._uploads_by_client):
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
    assert body["has_parsed_data"] is True
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


def test_mock_demo_is_rebuilt_after_eviction(server, monkeypatch):
    """The demo never answers match_not_found, whoever asks first."""
    c = server.client
    monkeypatch.setattr(config, "MATCH_MEMORY_CACHE_SIZE", 1)
    # Nobody called /api/demo yet: the match and section endpoints build it.
    assert c.get("/api/match/demo/draft").status_code == 200
    assert c.get("/api/fetch/111").status_code == 200       # evicts the demo
    assert "demo" not in routes._matches
    again = c.get("/api/match/demo")
    assert again.status_code == 200
    assert again.json()["parse_source"] == "mock"


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


def test_unparsed_match_is_kept_and_only_fetch_refreshes_it(server, monkeypatch):
    """"/api/fetch should download again" is separate from "the match exists"."""
    c = server.client
    server.od_unparsed.add(333)
    r = c.get("/api/fetch/333")
    assert r.status_code == 200 and r.json()["has_parsed_data"] is False
    c.get("/api/fetch/333")
    assert server.od_calls == [333]              # within the TTL → no second download
    assert routes.match_store.path_for("opendota_333").exists()   # survives eviction / restarts

    # TTL over: only /api/fetch downloads again; everything else keeps serving.
    monkeypatch.setattr(config, "OPENDOTA_UNPARSED_TTL_S", 0)
    assert c.get("/api/match/opendota_333").json()["has_parsed_data"] is False
    assert c.get("/api/match/opendota_333/draft").status_code == 200
    assert server.od_calls == [333]
    c.get("/api/fetch/333")
    c.get("/api/fetch/333")
    assert server.od_calls == [333, 333, 333]    # TTL 0 → every fetch downloads

    # After a restart (memory empty) the stored copy is served and is stale for /fetch.
    routes._matches.clear()
    assert c.get("/api/match/opendota_333").status_code == 200
    assert server.od_calls == [333, 333, 333]

    # OpenDota has parsed the replay meanwhile: the basic copy is replaced, everywhere.
    server.od_unparsed.discard(333)
    assert c.get("/api/fetch/333").json()["has_parsed_data"] is True
    assert c.get("/api/match/opendota_333").json()["has_parsed_data"] is True
    routes._matches.clear()
    assert c.get("/api/match/opendota_333").json()["has_parsed_data"] is True
    c.get("/api/fetch/333")
    assert server.od_calls == [333, 333, 333, 333]   # parsed → never refetched on its own


def test_failed_refresh_keeps_serving_the_cached_copy(server, monkeypatch):
    c = server.client
    server.od_unparsed.add(334)
    assert c.get("/api/fetch/334").status_code == 200
    monkeypatch.setattr(config, "OPENDOTA_UNPARSED_TTL_S", 0)

    server.od_errors[334] = FakeOpenDotaError("upstream 503", status=503)
    stale = c.get("/api/fetch/334")
    assert stale.status_code == 200 and stale.json()["has_parsed_data"] is False
    assert server.od_calls == [334, 334]

    # ... also when the refresh is refused by our own rate limit.
    del server.od_errors[334]
    monkeypatch.setenv("RATE_LIMIT_FETCH", "1/600")
    security.reset_rate_limits()
    assert c.get("/api/fetch/334").status_code == 200    # uses the one allowed fetch
    assert c.get("/api/fetch/334").status_code == 200    # limited → cached copy, no 429
    assert server.od_calls == [334, 334, 334]
    assert c.get("/api/fetch/999").status_code == 429    # no copy to fall back on


def test_request_parse_keeps_the_match_and_marks_it_for_refetch(server):
    c = server.client
    server.od_unparsed.add(444)
    assert c.get("/api/fetch/444").json()["has_parsed_data"] is False
    r = c.post("/api/request-parse/444")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "job": {"jobId": 99}}
    assert server.od_request_calls == [444]

    # The frontend fires request-parse right after an unparsed fetch: the
    # match view, the sections and the AI coach must keep working.
    assert c.get("/api/match/opendota_444").status_code == 200
    assert c.get("/api/match/opendota_444/laning").status_code == 200
    assert routes.match_store.path_for("opendota_444").exists()
    assert server.od_calls == [444]

    # The next /api/fetch downloads again although the TTL (120 s) is not over.
    server.od_unparsed.discard(444)
    assert c.get("/api/fetch/444").json()["has_parsed_data"] is True
    assert server.od_calls == [444, 444]
    c.get("/api/fetch/444")
    assert server.od_calls == [444, 444]

    # Also for a match that was already parsed, and for one we never fetched.
    assert c.post("/api/request-parse/444").status_code == 200
    c.get("/api/fetch/444")
    assert server.od_calls == [444, 444, 444]
    assert c.post("/api/request-parse/445").status_code == 200
    assert "opendota_445" not in routes._matches


def test_refetch_mark_survives_eviction_and_a_failed_refresh(server, monkeypatch):
    """A request-parse is honoured by the next successful download, whenever that is."""
    c = server.client
    assert c.get("/api/fetch/446").json()["has_parsed_data"] is True
    assert c.post("/api/request-parse/446").status_code == 200

    # Server restart / LRU eviction: the mark comes back with the stored match.
    routes._matches.clear()
    assert c.get("/api/match/opendota_446").status_code == 200
    assert server.od_calls == [446]                      # only /api/fetch downloads
    assert routes.match_store.load("opendota_446")[1]["refetch"] is True

    # OpenDota is down: the copy is served, and not asked for again right away ...
    server.od_errors[446] = FakeOpenDotaError("upstream 503", status=503)
    assert c.get("/api/fetch/446").status_code == 200
    assert c.get("/api/fetch/446").status_code == 200
    assert server.od_calls == [446, 446]
    # ... but once the back-off is over (or OpenDota answered a new
    # request-parse) the match is downloaded although it has parsed data.
    del server.od_errors[446]
    assert c.post("/api/request-parse/446").status_code == 200
    assert c.get("/api/fetch/446").status_code == 200
    assert server.od_calls == [446, 446, 446]
    assert "refetch" not in routes.match_store.load("opendota_446")[1]
    c.get("/api/fetch/446")
    assert server.od_calls == [446, 446, 446]            # the mark was used up

    # The same without a second request-parse: the back-off (the TTL) ends.
    assert c.post("/api/request-parse/446").status_code == 200
    monkeypatch.setattr(config, "OPENDOTA_UNPARSED_TTL_S", 0)
    server.od_errors[446] = FakeOpenDotaError("upstream 503", status=503)
    assert c.get("/api/fetch/446").status_code == 200
    del server.od_errors[446]
    assert c.get("/api/fetch/446").status_code == 200
    assert server.od_calls == [446, 446, 446, 446, 446]


@pytest.mark.parametrize("error,status,code", [
    (FakeOpenDotaError("Not Found", status=404), 404, "opendota_not_found"),
    (FakeOpenDotaError("upstream 503", status=503), 502, "opendota_unavailable"),
    (FakeOpenDotaError("timed out"), 502, "opendota_unavailable"),
    (ConnectionResetError("peer reset secret-host"), 502, "opendota_unavailable"),
    (TimeoutError("read timed out secret-host"), 502, "opendota_unavailable"),
    # Not a network failure: a bug in our own transform / analysis code.
    (RuntimeError("secret internals /etc/passwd"), 500, "internal_error"),
    (KeyError("secret_field"), 500, "internal_error"),
])
def test_fetch_errors_are_mapped_and_user_safe(server, error, status, code):
    server.od_errors[555] = error
    r = server.client.get("/api/fetch/555", headers={"Accept-Language": "en"})
    assert r.status_code == status
    assert r.json()["code"] == code
    assert str(error) not in r.json()["detail"]
    assert "opendota_555" not in routes._matches


@pytest.mark.parametrize("raw", ["abc", "0", "-5", "1" * 25, "9" * 19, "²", "١٢٣", "12３", "1e3", "+5"])
def test_fetch_rejects_invalid_match_ids(server, raw):
    for response in (server.client.get(f"/api/fetch/{raw}"), server.client.post(f"/api/request-parse/{raw}")):
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_match_id"
    assert server.od_calls == [] and server.od_request_calls == []


def test_analysis_bug_during_fetch_is_not_blamed_on_opendota(server, monkeypatch, caplog):
    def broken_analysis(match):
        raise TypeError("unsupported operand in analyzer")

    monkeypatch.setattr(routes, "full_analysis", broken_analysis)
    with caplog.at_level("ERROR", logger="backend.app.routes"):
        r = server.client.get("/api/fetch/556", headers={"Accept-Language": "en"})
    assert r.status_code == 500 and r.json()["code"] == "internal_error"
    assert "analyzer" not in r.text
    assert any(rec.exc_info for rec in caplog.records)   # the operator gets the traceback


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
    (b"BZh91AY&SY" + b"x" * 100, "match.dem", "replay_decompress_failed"),   # bzip2 header + junk
    (bz2.compress(DEM + b"x" * 5000)[:-20], "match.dem.bz2", "replay_decompress_failed"),  # cut off
    (bz2.compress(b"some text file"), "notes.dem.bz2", "replay_invalid"),
    (b"", "empty.dem", "replay_invalid"),
    (DEM + b"x" * 100, "match.txt", "upload_bad_extension"),
    (bz2.compress(DEM + b"x" * 100), "match.bz2", "upload_bad_extension"),   # not named like a replay
    (DEM + b"x" * 100, "match.dem.gz", "upload_bad_extension"),
])
def test_upload_rejects_invalid_files(server, data, filename, code):
    r = upload(server.client, data, filename, headers={"Accept-Language": "en"})
    assert r.status_code == 400
    assert r.json()["code"] == code
    assert r.json()["detail"] == security.MESSAGES[code]["en"]
    assert server.parse_calls == []
    assert not [p for p in server.replays.rglob("*") if p.is_file()]


def test_compressed_upload_is_unpacked_and_shares_the_cache_with_the_plain_file(server):
    """Most users have the .dem.bz2 they downloaded: it is the same match as its .dem."""
    c = server.client
    data = replay_bytes("z", size=200_000)
    digest = hashlib.sha256(data).hexdigest()[:32]

    packed = upload(c, bz2.compress(data), "8713267082_1234567.DEM.BZ2")
    assert packed.status_code == 200, packed.text
    assert packed.json()["file_key"] == f"replay_{digest}" and packed.json()["cached"] is False
    assert (server.replays / f"{digest}.dem").read_bytes() == data      # stored unpacked
    assert server.parse_calls == [str(server.replays / f"{digest}.dem")]

    plain = upload(c, data, "8713267082.dem")
    assert plain.json() == {**packed.json(), "cached": True}
    assert len(server.parse_calls) == 1


def test_zstd_upload_under_a_bz2_name_is_unpacked(server):
    """Valve's replay servers send Zstandard data in files named .dem.bz2."""
    zstandard = pytest.importorskip("zstandard")
    data = replay_bytes("v", size=200_000)
    for filename in ("9037980041_1122334455.dem.bz2", "match.dem.zst", "match.dem"):
        r = upload(server.client, zstandard.ZstdCompressor().compress(data), filename)
        assert r.status_code == 200, r.text
        assert r.json()["file_key"] == "replay_" + hashlib.sha256(data).hexdigest()[:32]
    assert len(server.parse_calls) == 1


def test_zstd_upload_without_a_decoder_tells_the_user_to_unpack_it(server, monkeypatch):
    monkeypatch.setattr(storage, "_zstd_module", lambda: None)
    for lang, word in (("en", "Zstandard"), ("zh", "解压")):
        r = upload(server.client, storage.ZSTD_MAGIC + b"\x00" * 64, "match.dem.bz2",
                   headers={"Accept-Language": lang})
        assert r.status_code == 400 and r.json()["code"] == "replay_compressed"
        assert word in r.json()["detail"]
    assert server.parse_calls == []


def test_archives_the_server_does_not_unpack_are_not_called_damaged(server, monkeypatch):
    """An intact file gets the reason and what to do, in the user's language."""
    pytest.importorskip("zstandard")
    # A Zstandard frame that asks for a 64 MiB window (zstd --ultra / --long), 17 bytes.
    wide = storage.ZSTD_MAGIC + b"\x00" + bytes([(26 - 10) << 3]) + b"\x41\x00\x00" + DEM
    for lang in ("en", "zh"):
        r = upload(server.client, wide, "match.dem.zst", headers={"Accept-Language": lang})
        assert r.status_code == 400 and r.json()["code"] == "replay_window_too_large"
        assert r.json()["detail"] == security.MESSAGES["replay_window_too_large"][lang] and "--ultra" in r.json()["detail"]

    monkeypatch.setattr(storage, "_bz2_module", lambda: None)      # a Python built without libbz2
    for lang in ("en", "zh"):
        r = upload(server.client, bz2.compress(replay_bytes()), "match.dem.bz2", headers={"Accept-Language": lang})
        assert r.status_code == 400 and r.json()["code"] == "replay_compressed_bzip2"
        assert r.json()["detail"] == security.MESSAGES["replay_compressed_bzip2"][lang] and "bunzip2" in r.json()["detail"]
    assert server.parse_calls == []
    assert not [p for p in server.replays.rglob("*") if p.is_file()]


def test_upload_limit_counts_the_unpacked_replay(server, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1024 * 1024)
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    bomb = bz2.compress(DEM + b"\0" * (4 * 1024 * 1024))
    assert len(bomb) < 1024
    r = upload(server.client, bomb, "bomb.dem.bz2", headers={"Accept-Language": "en"})
    assert r.status_code == 413 and r.json()["code"] == "upload_too_large"
    assert "1 MB" in r.json()["detail"]
    assert not [p for p in server.replays.rglob("*") if p.is_file()]


def test_upload_requires_file_field(server):
    r = server.client.post("/api/upload", files={"other": ("a.dem", DEM)})
    assert r.status_code == 400
    assert r.json()["code"] == "upload_missing_file"
    r = server.client.post("/api/upload", data={"x": "1"})
    assert r.status_code == 400


@pytest.mark.parametrize("content_type", [
    "application/x-www-form-urlencoded", "application/json", "text/plain", "multipart/mixed; boundary=x", None,
])
def test_upload_refuses_non_multipart_bodies_without_parsing_them(server, monkeypatch, content_type):
    """Starlette's urlencoded parser has no size cap: it must never see an upload body."""
    async def must_not_run(request, lang):
        raise AssertionError("form parser reached")

    monkeypatch.setattr(routes, "_read_upload_form", must_not_run)
    headers = {"Content-Type": content_type} if content_type else {}
    r = server.client.post("/api/upload", content=b"a=" + b"x" * 200_000, headers=headers)
    assert r.status_code == 400
    assert r.json()["code"] == "bad_request"
    assert routes._uploads_in_progress == 0 and routes._uploads_by_client == {}


def test_upload_caps_non_file_multipart_fields(server):
    """CVE-2024-47874: a part without a filename must not be buffered without limit."""
    big_field = "x" * (routes._UPLOAD_MAX_FIELD_BYTES + 1)
    r = server.client.post(
        "/api/upload", data={"note": big_field},
        files={"file": ("match.dem", replay_bytes(), "application/octet-stream")},
    )
    assert r.status_code == 400
    assert r.json()["code"] == "bad_request"
    assert server.parse_calls == []
    assert not [p for p in server.replays.rglob("*") if p.is_file()]
    small = server.client.post(
        "/api/upload", data={"note": "ok"},
        files={"file": ("match.dem", replay_bytes(), "application/octet-stream")},
    )
    assert small.status_code == 200


def _raw_upload(receive_script, timeout: float = 5.0) -> list[dict]:
    """POST /api/upload straight through ASGI with a scripted receive channel.

    ``receive_script`` items are ASGI messages, "hang" to never deliver
    anything more, or a number: seconds to wait before the next message.
    Returns the messages the app sent.
    """
    boundary = "testboundary"
    sent: list[dict] = []
    script = list(receive_script)

    async def run():
        async def receive():
            if not script:
                return {"type": "http.disconnect"}
            item = script.pop(0)
            if item == "hang":
                await asyncio.sleep(3600)
            if isinstance(item, (int, float)):
                await asyncio.sleep(item)
                item = script.pop(0)
            return item

        async def send(msg):
            sent.append(msg)

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
            "path": "/api/upload", "raw_path": b"/api/upload", "root_path": "", "scheme": "http",
            "query_string": b"lang=en", "server": ("test", 80), "client": ("7.7.7.7", 1),
            "headers": [(b"host", b"test"),
                        (b"content-type", f"multipart/form-data; boundary={boundary}".encode())],
        }
        await asyncio.wait_for(app(scope, receive, send), timeout)

    asyncio.run(run())
    return sent


def _first_chunk() -> dict:
    head = (b"--testboundary\r\nContent-Disposition: form-data; name=\"file\"; filename=\"m.dem\"\r\n"
            b"Content-Type: application/octet-stream\r\n\r\n")
    return {"type": "http.request", "body": head + replay_bytes(), "more_body": True}


def test_cancelled_upload_is_not_a_server_error(server, caplog):
    """xhr.abort() / a dropped connection: no 500, no traceback, slots released."""
    with caplog.at_level("INFO"):
        sent = _raw_upload([_first_chunk(), {"type": "http.disconnect"}])
    statuses = [m["status"] for m in sent if m["type"] == "http.response.start"]
    assert statuses == [400]
    assert any("Upload aborted by client" in rec.getMessage() for rec in caplog.records)
    assert not [rec for rec in caplog.records if rec.levelname in ("ERROR", "CRITICAL")]
    assert routes._uploads_in_progress == 0 and routes._uploads_by_client == {}
    assert server.parse_calls == []
    assert not [p for p in server.replays.rglob("*") if p.is_file()]


def test_upload_receive_deadline_frees_the_slot(server, monkeypatch):
    """A client that trickles (or stops sending) its body cannot hold a slot forever."""
    monkeypatch.setattr(config, "UPLOAD_RECEIVE_TIMEOUT_S", 0.1)
    sent = _raw_upload([_first_chunk(), "hang"])
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 400
    assert (b"connection", b"close") in [(k.lower(), v.lower()) for k, v in start["headers"]]
    body = json.loads(b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))
    assert body["code"] == "upload_timeout"
    assert body["detail"] == security.message("upload_timeout", "en")
    assert routes._uploads_in_progress == 0 and routes._uploads_by_client == {}
    assert not [p for p in server.replays.rglob("*") if p.is_file()]


def test_stalled_upload_frees_the_slot_long_before_the_deadline(server, monkeypatch, caplog):
    """A connection that died without closing: the idle timeout, not the 15-minute deadline."""
    monkeypatch.setattr(config, "UPLOAD_IDLE_TIMEOUT_S", 0.1)
    assert config.UPLOAD_RECEIVE_TIMEOUT_S >= 10
    with caplog.at_level("INFO", logger="backend.app.routes"):
        sent = _raw_upload([_first_chunk(), "hang"])
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 400
    assert (b"connection", b"close") in [(k.lower(), v.lower()) for k, v in start["headers"]]
    body = json.loads(b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body"))
    assert body == {"detail": security.message("upload_timeout", "en"), "code": "upload_timeout"}
    assert any("no data received" in rec.getMessage() for rec in caplog.records)
    assert routes._uploads_in_progress == 0 and routes._uploads_by_client == {}
    assert server.parse_calls == []
    assert not [p for p in server.replays.rglob("*") if p.is_file()]


def test_slow_but_steady_upload_is_not_a_stall(server, monkeypatch):
    """The idle timeout restarts with every read: only silence counts."""
    monkeypatch.setattr(config, "UPLOAD_IDLE_TIMEOUT_S", 1.0)
    more = {"type": "http.request", "body": b"x" * 1000, "more_body": True}
    last = {"type": "http.request", "body": b"\r\n--testboundary--\r\n", "more_body": False}
    # 1.2 s in total, never more than 0.4 s without data.
    sent = _raw_upload([_first_chunk(), 0.4, more, 0.4, more, 0.4, last])
    assert [m["status"] for m in sent if m["type"] == "http.response.start"] == [200]
    assert len(server.parse_calls) == 1
    stored = [p for p in server.replays.glob("*.dem")]
    assert len(stored) == 1 and stored[0].stat().st_size == len(replay_bytes()) + 2000


def test_one_client_cannot_hold_every_upload_slot(server, monkeypatch):
    c = server.client
    # Another upload from this client ("testclient") is still being received.
    monkeypatch.setitem(routes._uploads_by_client, "testclient", 1)
    monkeypatch.setattr(routes, "_uploads_in_progress", 1)
    r = upload(c, replay_bytes("a"), headers={"Accept-Language": "en"})
    assert r.status_code == 429
    assert r.json()["code"] == "upload_in_progress"
    assert r.json()["retry_after"] == 30 and r.headers["Retry-After"] == "30"
    assert routes._uploads_by_client == {"testclient": 1}
    assert server.parse_calls == []

    # Other clients are not affected by it.
    monkeypatch.setattr(config, "TRUST_PROXY_HEADERS", True)
    assert upload(c, replay_bytes("a"), headers={"X-Real-IP": "8.8.8.8"}).status_code == 200
    assert routes._uploads_by_client == {"testclient": 1} and routes._uploads_in_progress == 1

    monkeypatch.setattr(config, "UPLOAD_MAX_PER_CLIENT", 2)
    monkeypatch.setattr(config, "TRUST_PROXY_HEADERS", False)
    assert upload(c, replay_bytes("b")).status_code == 200
    assert routes._uploads_by_client == {"testclient": 1}


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


def test_stored_match_of_another_format_is_rebuilt_from_the_replay(server):
    """Results written by an older parser are never served; the upload is parsed again."""
    c = server.client
    key = upload(c, replay_bytes("z")).json()["file_key"]
    assert len(server.parse_calls) == 1
    path = routes.match_store.path_for(key)
    stored = json.loads(gzip.decompress(path.read_bytes()))
    assert stored["format"] == storage.MatchStore.FORMAT

    stored["format"] = storage.MatchStore.FORMAT - 1
    stored["match"]["match_id"] = 31337          # what the "old parser" had produced
    path.write_bytes(gzip.compress(json.dumps(stored).encode()))
    routes._matches.clear()                      # server restart
    server.next_match_id = 2002

    again = c.get(f"/api/match/{key}")
    assert again.status_code == 200
    assert again.json()["analysis"]["match_overview"]["match_id"] == 2002
    assert len(server.parse_calls) == 2
    assert json.loads(gzip.decompress(path.read_bytes()))["format"] == storage.MatchStore.FORMAT

    # Without the replay file there is nothing to rebuild from.
    routes._matches.clear()
    path.unlink()
    (server.replays / f"{key[len('replay_'):]}.dem").unlink()
    assert c.get(f"/api/match/{key}").json()["code"] == "match_not_found"
    assert len(server.parse_calls) == 2


@pytest.mark.parametrize("error,file_kept", [
    (FakeReplayParseError("Corrupted demo"), False),
    (FakeParserUnavailable("no java right now"), True),
])
def test_failed_rebuild_is_a_plain_miss(server, error, file_kept):
    c = server.client
    key = upload(c, replay_bytes("q")).json()["file_key"]
    replay = server.replays / f"{key[len('replay_'):]}.dem"
    routes._matches.clear()
    routes.match_store.delete(key)
    server.parse_error = error
    r = c.get(f"/api/match/{key}/laning", headers={"Accept-Language": "en"})
    assert r.status_code == 404 and r.json()["code"] == "match_not_found"
    assert replay.exists() is file_kept


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


@pytest.fixture
def fake_rag(monkeypatch):
    """Stub of the knowledge-retrieval package (its frozen interface: status / warm_up / refresh)."""
    import backend.app.ai as ai_pkg

    mod = types.ModuleType("backend.app.ai.rag")
    mod.state = {"mode": "hybrid", "embedder": "bge-small-en-v1.5-int8", "vectors": 1042, "chunks": 832,
                 "static_coverage": 1.0, "building": False, "last_build": "2026-10-11T00:00:00Z", "error": None}
    mod.refreshed = []
    mod.status = lambda: dict(mod.state)
    mod.warm_up = lambda: None
    mod.refresh = mod.refreshed.append
    monkeypatch.setitem(sys.modules, "backend.app.ai.rag", mod)
    monkeypatch.setattr(ai_pkg, "rag", mod, raising=False)
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
    assert ops["gamedata"] is None            # no get_meta: the data layer cannot say


def test_admin_status_reports_game_version_source_and_age(server, monkeypatch, fake_data_manager):
    """Is the server on the current patch, where did its data come from, how old is it?"""
    monkeypatch.setenv("ADMIN_TOKEN", "tok")
    hdr = {"X-Admin-Token": "tok"}
    fake_data_manager.get_meta = lambda: {
        "patch_version": "7.41", "patch_label": "7.41f", "patch_released": "2026-09-15T07:00:00Z",
        "data_label": "7.41f", "data_current": True, "source": "dotaconstants_github",
        "source_revision": "0123abc", "source_date": "2026-10-04T12:00:00Z", "schema_version": 4,
        "last_fetched": "2026-10-10T03:00:00Z", "age_days": 0.9, "is_stale": False, "hero_count": 127,
    }
    fake_data_manager.get_patch_info = lambda: {"label": "7.41f", "checked_at": "2026-10-11T03:00:05Z"}
    assert server.client.get("/api/admin/status", headers=hdr).json()["gamedata"] == {
        "patch_version": "7.41", "patch_label": "7.41f", "data_label": "7.41f", "data_current": True,
        "source": "dotaconstants_github", "source_date": "2026-10-04T12:00:00Z",
        "last_fetched": "2026-10-10T03:00:00Z", "age_days": 0.9, "is_stale": False,
        "checked_at": "2026-10-11T03:00:05Z",     # the periodic version check is alive
    }
    # The detailed endpoint passes everything on.
    detail = server.client.get("/api/admin/gamedata-status", headers=hdr).json()
    assert detail["source_revision"] == "0123abc" and detail["patch_label"] == "7.41f"

    # Before the first fetch the data layer has little to say: same keys, None for the rest.
    fake_data_manager.get_meta = lambda: {"status": "no_data", "patch_version": "7.41", "source": "fallback",
                                          "patch_label": "7.41f", "data_label": "", "data_current": None}
    fake_data_manager.get_patch_info = lambda: {"label": "7.41f", "checked_at": ""}   # never checked
    gamedata = server.client.get("/api/admin/status", headers=hdr).json()["gamedata"]
    assert gamedata["source"] == "fallback" and gamedata["patch_label"] == "7.41f"
    assert gamedata["age_days"] is None and gamedata["last_fetched"] is None
    assert gamedata["checked_at"] is None
    del fake_data_manager.get_patch_info                                             # an older data layer
    assert server.client.get("/api/admin/status", headers=hdr).json()["gamedata"]["checked_at"] is None

    def broken():
        raise RuntimeError("meta.json unreadable")

    fake_data_manager.get_meta = broken
    assert server.client.get("/api/admin/status", headers=hdr).json()["gamedata"] is None


def test_admin_status_reports_the_retrieval_index(server, monkeypatch, fake_rag):
    import backend.app.ai as ai_pkg

    monkeypatch.setenv("ADMIN_TOKEN", "tok")
    hdr = {"X-Admin-Token": "tok"}
    assert server.client.get("/api/admin/status", headers=hdr).json()["rag"] == fake_rag.state

    fake_rag.state = {**fake_rag.state, "mode": "lexical", "vectors": 0, "error": "deps_missing: numpy"}
    assert server.client.get("/api/admin/status", headers=hdr).json()["rag"]["error"] == "deps_missing: numpy"

    def broken():
        raise RuntimeError("index is corrupt")

    fake_rag.status = broken                  # "never raises" is not taken on trust
    assert server.client.get("/api/admin/status", headers=hdr).json()["rag"] is None

    monkeypatch.delattr(ai_pkg, "rag")        # no such package at all
    monkeypatch.setitem(sys.modules, "backend.app.ai.rag", None)
    ops = server.client.get("/api/admin/status", headers=hdr)
    assert ops.status_code == 200 and ops.json()["rag"] is None


def test_admin_gamedata_update_tells_the_retrieval_index_about_new_data(server, monkeypatch,
                                                                        fake_data_manager, fake_rag):
    monkeypatch.setenv("ADMIN_TOKEN", "tok")
    hdr = {"X-Admin-Token": "tok"}
    results = [
        {"status": "up_to_date", "patch_version": "7.41", "patch_label": "7.41f"},
        {"status": "updated", "changed": False, "reason": "forced", "patch_version": "7.41"},
        {"status": "updated", "changed": True, "reason": "new_source_revision", "patch_version": "7.41"},
    ]
    forced = []

    def fetch_and_save_gamedata(force=False):
        forced.append(force)
        return results[len(forced) - 1]

    fake_data_manager.fetch_and_save_gamedata = fetch_and_save_gamedata
    for expected in results:
        r = server.client.post("/api/admin/update-gamedata?force=true", headers=hdr)
        assert r.status_code == 200 and r.json() == expected
    assert forced == [True, True, True]
    assert fake_rag.refreshed == ["gamedata"]     # only once: when the data on disk changed


def test_admin_gamedata_update_can_accept_a_shrunken_data_set(server, monkeypatch, fake_data_manager):
    """The data layer refuses a download that lost a tenth of a table; the operator may accept it."""
    monkeypatch.setenv("ADMIN_TOKEN", "tok")
    hdr = {"X-Admin-Token": "tok"}
    calls = []

    def fetch_and_save_gamedata(force=False, allow_shrink=False):
        calls.append((force, allow_shrink))
        if not allow_shrink:
            raise RuntimeError("Implausible constants: 96 heroes, the stored data has 127")
        return {"status": "updated", "changed": False, "reason": "forced", "patch_version": "7.41"}

    fake_data_manager.fetch_and_save_gamedata = fetch_and_save_gamedata
    refused = server.client.post("/api/admin/update-gamedata?force=true&lang=en", headers=hdr)
    assert refused.status_code == 502 and refused.json()["code"] == "gamedata_update_failed"
    accepted = server.client.post("/api/admin/update-gamedata?force=true&allow_shrink=true", headers=hdr)
    assert accepted.status_code == 200 and accepted.json()["status"] == "updated"
    assert calls == [(True, False), (True, True)]
    bad = server.client.post("/api/admin/update-gamedata?allow_shrink=perhaps&lang=en", headers=hdr)
    assert bad.status_code == 400 and bad.json()["code"] == "bad_request" and len(calls) == 2


def test_admin_status_reports_learning_and_maintenance(server, monkeypatch, tmp_path):
    from backend.app.ai.learning import pipeline

    # The route reports pipeline.learning_status(): the manifest is the pipeline's business
    manifest = tmp_path / "_last_learn.json"
    monkeypatch.setattr(pipeline, "LAST_LEARN_FILE", manifest)
    monkeypatch.setattr(pipeline, "due_levels", lambda: (2, 4))
    monkeypatch.setenv("ADMIN_TOKEN", "tok")
    hdr = {"X-Admin-Token": "tok"}

    ops = server.client.get("/api/admin/status", headers=hdr).json()
    assert ops["learning"] == {"due_levels": [2, 4], "next_due": {"1": None, "2": None, "3": None, "4": None},
                               "levels": {}, "last_run": None, "patch": None}
    assert ops["startup_maintenance"] in ("running", "finished")

    manifest.write_text(json.dumps({
        "timestamp": "2026-10-01T00:00:00Z", "patch": "7.40",
        "levels": {"1": {"status": "ok", "chunks_written": 12}, "2": {"status": "failed", "error": "offline"}},
        "summary": {"big": "x" * 1000},
    }))
    learning = server.client.get("/api/admin/status", headers=hdr).json()["learning"]
    assert learning["last_run"] == "2026-10-01T00:00:00Z" and learning["patch"] == "7.40"
    assert learning["levels"]["2"] == {"status": "failed", "error": "offline"}
    assert "summary" not in learning

    manifest.write_text("{broken")
    assert server.client.get("/api/admin/status", headers=hdr).json()["learning"]["due_levels"] == [2, 4]


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
    assert caps["upload_extensions"] == [".dem", ".dem.bz2", ".dem.zst"]
    assert caps["ai_coach_llm"] is False
    # From the real data layer and retrieval package, whatever state they are in.
    assert caps["knowledge_mode"] in ("hybrid", "lexical")
    assert caps["game_patch"] is None or caps["game_patch"][:2] == "7."
    monkeypatch.setattr(config, "AI_COACH_DAILY_LIMIT", 0)
    assert c.get("/api/capabilities").json()["ai_coach_daily_remaining"] is None

    async def unavailable():
        raise FakeParserUnavailable("no java")

    monkeypatch.setattr(routes, "_get_replay_parser", unavailable)
    assert c.get("/api/capabilities").json()["replay_parser"] is False


def test_capabilities_name_the_game_version_and_the_knowledge_mode(server, monkeypatch,
                                                                    fake_data_manager, fake_rag):
    """Public: the game's current version as last looked up, and whether retrieval has vectors."""
    import backend.app.ai as ai_pkg

    c = server.client
    asked: list[float] = []
    fake_data_manager.patch_label_for_time = lambda when: asked.append(when) or "7.41f"
    caps = c.get("/api/capabilities").json()
    assert (caps["game_patch"], caps["knowledge_mode"]) == ("7.41f", "hybrid")
    assert len(asked) == 1 and abs(asked[0] - time.time()) < 60  # the version of this moment
    assert "embedder" not in caps and "rag" not in caps          # details stay behind the admin token
    fake_data_manager.patch_label_for_time = lambda when: ""     # the data layer cannot tell
    assert c.get("/api/capabilities").json()["game_patch"] is None
    fake_data_manager.patch_label_for_time = lambda when: "7.41f"

    fake_rag.state = {**fake_rag.state, "mode": "lexical"}
    assert c.get("/api/capabilities").json()["knowledge_mode"] == "lexical"
    fake_rag.state = {"mode": "something new"}
    assert c.get("/api/capabilities").json()["knowledge_mode"] == "lexical"

    # Neither the data layer nor the retrieval package: still a full answer.
    del fake_data_manager.patch_label_for_time
    monkeypatch.delattr(ai_pkg, "rag")
    monkeypatch.setitem(sys.modules, "backend.app.ai.rag", None)
    r = c.get("/api/capabilities")
    assert r.status_code == 200
    assert (r.json()["game_patch"], r.json()["knowledge_mode"]) == (None, "lexical")


def test_game_patch_is_only_named_while_the_version_was_checked_recently(server, monkeypatch, tmp_path, fake_rag):
    """With the real data layer: never its built-in version, nor a release list too old to trust."""
    from backend.app.ai import data_manager

    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", tmp_path)

    def game_patch():
        return server.client.get("/api/capabilities").json()["game_patch"]

    def release_list_downloaded(days_ago: float, releases: dict[str, str]) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - days_ago * 86400))
        (tmp_path / data_manager.PATCHES_FILE).write_text(json.dumps({
            "releases_fetched_at": stamp,
            "releases": [{"name": name, "date": date} for name, date in releases.items()],
        }), encoding="utf-8")

    # Valve's feed was never read: the data layer still names a version (compiled in) ...
    assert data_manager.get_patch_info()["checked_at"] == ""
    assert data_manager.get_patch_label() == data_manager.DEFAULT_PATCH_LABEL
    assert game_patch() is None                 # ... which the server does not pass off as known

    release_list_downloaded(1, {"7.41e": "2026-08-20T07:00:00Z", "7.41f": "2026-09-15T07:00:00Z"})
    assert game_patch() == "7.41f"
    # Offline (or GAMEDATA_AUTO_UPDATE=0) since then: a newer patch may be out by now.
    release_list_downloaded(config.GAMEDATA_STALE_DAYS + 1, {"7.41f": "2026-09-15T07:00:00Z"})
    assert data_manager.get_patch_label() == "7.41f"
    assert game_patch() is None


# ---------------------------------------------------------------------------
# Error contract W9: every /api/ error is {"detail": <localized str>, "code"}
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method,url,status,code", [
    ("get", "/api/nope", 404, "not_found"),
    ("get", "/api/fetch/https://www.opendota.com/matches/123", 404, "not_found"),
    ("post", "/api/demo", 405, "method_not_allowed"),
    ("put", "/api/upload", 405, "method_not_allowed"),
    ("delete", "/api/match/demo", 405, "method_not_allowed"),
    # GET on a POST-only path falls through to the static mount: a plain miss.
    ("get", "/api/upload", 404, "not_found"),
])
def test_framework_errors_follow_the_error_contract(server, method, url, status, code):
    for lang in ("zh", "en"):
        r = getattr(server.client, method)(url, params={"lang": lang})
        assert r.status_code == status
        assert r.json() == {"detail": security.message(code, lang), "code": code}


def test_validation_errors_follow_the_error_contract(server, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "tok")
    r = server.client.post("/api/admin/update-gamedata?force=maybe&lang=en", headers={"X-Admin-Token": "tok"})
    assert r.status_code == 400
    assert r.json() == {"detail": "Malformed request.", "code": "bad_request"}


def test_unhandled_exception_follows_the_error_contract(server, monkeypatch):
    def boom():
        raise RuntimeError("secret detail")

    monkeypatch.setattr(routes, "_llm_configured", boom)
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/api/capabilities", headers={"Accept-Language": "en"})
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error.", "code": "internal_error"}


def test_non_api_paths_keep_the_default_not_found(server):
    r = server.client.get("/no-such-page.html")
    assert r.status_code == 404
    assert "code" not in r.json()


def test_match_payload_served_with_and_without_gzip(server):
    c = server.client
    c.get("/api/fetch/1234")
    plain = c.get("/api/match/opendota_1234", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers
    zipped = c.get("/api/match/opendota_1234", headers={"Accept-Encoding": "gzip"})
    assert zipped.headers["content-encoding"] == "gzip"
    assert plain.json() == zipped.json()
    assert plain.headers["vary"] == "Accept-Encoding"


def test_cached_match_is_analyzed_again_when_the_game_data_changes(server, monkeypatch, tmp_path):
    """A match served before the game data was fetched (first start) must not keep
    the names of that moment for as long as it stays in memory."""
    from backend.app.ai import data_manager

    def talents(payload: dict) -> dict[str, str]:
        return {s["ability_key"]: s["ability"] for build in payload["analysis"]["skill_builds"]
                for s in build["skill_order"] if s["is_talent"]}

    gamedata = tmp_path / "gamedata"
    gamedata.mkdir()
    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", gamedata)
    data_manager.invalidate_all_caches()
    try:
        c = server.client
        before = talents(c.get("/api/demo").json())
        record = routes._matches["demo"]
        key = sorted(before)[0]
        assert before[key] != "+Probe Talent"

        # the background fetch finishes: new files, caches dropped
        (gamedata / data_manager.TALENTS_FILE).write_text(json.dumps({key: {"name": "+Probe Talent"}}))
        data_manager.invalidate_all_caches()

        for url, headers in (("/api/match/demo", {"Accept-Encoding": "identity"}),
                             ("/api/match/demo", {"Accept-Encoding": "gzip"}),
                             ("/api/match/demo/skill_builds", {})):
            body = c.get(url, headers=headers).json()
            after = talents(body if "analysis" in body else {"analysis": body})   # a section: {name: data}
            assert after[key] == "+Probe Talent"
            assert {k: v for k, v in after.items() if k != key} == {k: v for k, v in before.items() if k != key}
        assert routes._matches["demo"] is record          # the same match, not generated or parsed again

        # nothing changed since: the payload is served as cached
        body = record.body
        c.get("/api/match/demo")
        assert record.body is body
    finally:
        data_manager.invalidate_all_caches()


def test_json_bytes_replaces_non_finite_numbers():
    import json as _json

    payload = {"a": float("nan"), "b": [1.5, float("inf")], "c": ("x", -float("inf")), "d": "中文"}
    decoded = _json.loads(routes._json_bytes(payload))
    assert decoded == {"a": None, "b": [1.5, None], "c": ["x", None], "d": "中文"}
