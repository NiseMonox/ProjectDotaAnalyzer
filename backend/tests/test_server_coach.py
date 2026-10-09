"""Tests for the AI coach routes (JSON + SSE).

``backend.app.ai.coach`` is replaced by a fake module, so neither the real
pipeline nor any LLM is involved.
"""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import asyncio
import json
import sys
import threading
import types
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import backend.app.ai as ai_pkg
from backend.app import config, routes, security, storage
from backend.app import parser as parser_mod
from backend.app.main import app
from backend.app.models import GoldXpSnapshot, MatchData, PlayerData

JSON_URL = "/api/match/opendota_777/ai-coach/{slot}?lang={lang}"
SSE_URL = "/api/match/opendota_777/ai-coach-stream/{slot}?lang={lang}"


def make_player(slot: int) -> PlayerData:
    return PlayerData(
        player_slot=slot, hero_id=slot + 1, hero_name=f"Hero{slot}", hero_slug=f"hero{slot}",
        player_name=f"p{slot}", team="radiant" if slot < 128 else "dire", kills=0, deaths=0,
        assists=0, level=1, last_hits=0, denies=0, gold_per_min=0, xp_per_min=0, hero_damage=0,
        tower_damage=0, hero_healing=0, net_worth=0, items=[], backpack=[], final_items=[],
        skills=[], wards=[], ability_events=[],
    )


def make_match(match_id: int) -> MatchData:
    return MatchData(
        match_id=match_id, duration=1200, radiant_win=True, game_mode="All Pick", start_time=0,
        radiant_score=0, dire_score=0, players=[make_player(0), make_player(128)], wards=[],
        smokes=[], teamfights=[], ability_events=[],
        gold_xp_timeline=[GoldXpSnapshot(time=60, radiant_gold=1, dire_gold=1, radiant_xp=1, dire_xp=1)],
    )


class FakeReport:
    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return dict(self._data)


class FakeCoach:
    def __init__(self):
        self.calls = []
        self.degraded = False
        self.delay = 0.0
        self.error: Exception | None = None
        self.gate: threading.Event | None = None

    async def generate_coaching_report(self, match, player_slot, lang="zh", progress_callback=None):
        self.calls.append((match.match_id, player_slot, lang))
        if progress_callback:
            await progress_callback({"stage": "detect", "current": 0, "total": 2})
        if self.gate is not None:
            while not self.gate.is_set():
                await asyncio.sleep(0.01)
        if self.delay:
            await asyncio.sleep(self.delay)
        if progress_callback:
            await progress_callback({"stage": "evaluate", "current": 2, "total": 2})
        if self.error is not None:
            raise self.error
        return FakeReport({
            "match_id": match.match_id, "player_slot": player_slot, "lang": lang,
            "overall_grade": "B", "summary": "ok", "death_analyses": [],
            "llm_used": not self.degraded, "degraded": self.degraded, "report_version": 2,
            "n": len(self.calls),
        })


def make_request(headers=None) -> Request:
    return Request({
        "type": "http", "method": "GET", "path": "/api/x", "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": ("9.9.9.9", 1),
    })


@pytest.fixture
def coach_env(tmp_path, monkeypatch):
    for name, value in {
        "REPLAY_DIR": tmp_path / "replays",
        "COACH_CACHE_DIR": tmp_path / "coaching",
        "MATCH_CACHE_DIR": tmp_path / "matches",
        "DEFAULT_REPLAY": tmp_path / "replays" / "none.dem",
    }.items():
        monkeypatch.setattr(config, name, value)
    for d in ("replays", "coaching", "matches"):
        (tmp_path / d).mkdir()
    monkeypatch.setattr(routes, "match_store",
                        storage.MatchStore(tmp_path / "matches", lambda: config.MATCH_CACHE_MAX_FILES))
    monkeypatch.setattr(routes, "ai_coach_daily_quota",
                        security.DailyQuota(tmp_path / "usage.json", lambda: config.AI_COACH_DAILY_LIMIT))
    for name in security.RATE_LIMIT_DEFAULTS:
        monkeypatch.setenv(f"RATE_LIMIT_{name.upper()}", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    security.reset_rate_limits()
    for cache in (routes._matches, routes._coach_report_cache, routes._inflight, routes._coach_jobs):
        cache.clear()

    class FakeOpenDota:
        def name(self):
            return "opendota"

        async def parse(self, mid_str):
            return make_match(int(mid_str))

    monkeypatch.setattr(parser_mod, "OpenDotaParser", FakeOpenDota)

    fake = FakeCoach()
    module = types.ModuleType("backend.app.ai.coach")
    module.REPORT_SCHEMA_VERSION = 2
    module.generate_coaching_report = fake.generate_coaching_report
    monkeypatch.setitem(sys.modules, "backend.app.ai.coach", module)
    monkeypatch.setattr(ai_pkg, "coach", module, raising=False)

    with TestClient(app) as client:
        assert client.get("/api/fetch/777").status_code == 200
        yield SimpleNamespace(client=client, fake=fake, module=module, tmp=tmp_path)

    security.reset_rate_limits()
    for cache in (routes._matches, routes._coach_report_cache, routes._inflight, routes._coach_jobs):
        cache.clear()


def sse_events(text: str) -> list[dict]:
    return [json.loads(line[len("data: "):]) for line in text.splitlines() if line.startswith("data: ")]


def coach_files(env) -> list[str]:
    return sorted(p.name for p in (env.tmp / "coaching").iterdir())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_slot_and_unknown_match(coach_env):
    c = coach_env.client
    r = c.get(JSON_URL.format(slot=5, lang="en"))
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_player_slot"
    assert r.json()["detail"] == "This player is not part of the match."

    r = c.get("/api/match/nope/ai-coach/0")
    assert r.status_code == 404 and r.json()["code"] == "match_not_found"

    events = sse_events(c.get(SSE_URL.format(slot=5, lang="zh")).text)
    assert events == [{
        "type": "error", "message": security.message("invalid_player_slot", "zh"),
        "code": "invalid_player_slot", "status": 400,
    }]
    events = sse_events(c.get("/api/match/nope/ai-coach-stream/0").text)
    assert events[0]["code"] == "match_not_found"
    assert coach_env.fake.calls == []


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def test_report_cached_in_memory_and_versioned_on_disk(coach_env):
    c = coach_env.client
    first = c.get(JSON_URL.format(slot=0, lang="zh"))
    assert first.status_code == 200
    assert first.json()["overall_grade"] == "B"
    assert coach_files(coach_env) == ["777_opendota_0_zh_v2.json"]

    assert c.get(JSON_URL.format(slot=0, lang="zh")).json() == first.json()
    routes._coach_report_cache.clear()  # e.g. after a restart: served from disk
    assert c.get(JSON_URL.format(slot=0, lang="zh")).json() == first.json()
    assert len(coach_env.fake.calls) == 1

    # Separate cache entry per language.
    assert c.get(JSON_URL.format(slot=0, lang="en")).json()["lang"] == "en"
    assert len(coach_env.fake.calls) == 2

    # A new report schema invalidates old disk entries.
    coach_env.module.REPORT_SCHEMA_VERSION = 3
    routes._coach_report_cache.clear()
    c.get(JSON_URL.format(slot=0, lang="zh"))
    assert len(coach_env.fake.calls) == 3
    assert "777_opendota_0_zh_v3.json" in coach_files(coach_env)

    # SSE on a cached report: one done event.
    events = sse_events(c.get(SSE_URL.format(slot=0, lang="zh")).text)
    assert [e["type"] for e in events] == ["done"]
    assert len(coach_env.fake.calls) == 3


def test_degraded_reports_are_not_persisted(coach_env, monkeypatch):
    c = coach_env.client
    coach_env.fake.degraded = True
    r = c.get(JSON_URL.format(slot=0, lang="zh"))
    assert r.status_code == 200 and r.json()["degraded"] is True
    assert coach_files(coach_env) == []
    c.get(JSON_URL.format(slot=0, lang="zh"))
    assert len(coach_env.fake.calls) == 1  # short-lived memory cache

    monkeypatch.setattr(config, "AI_COACH_DEGRADED_TTL_S", 0)
    routes._coach_report_cache.clear()
    c.get(JSON_URL.format(slot=0, lang="zh"))
    c.get(JSON_URL.format(slot=0, lang="zh"))
    assert len(coach_env.fake.calls) == 3

    coach_env.fake.degraded = False
    c.get(JSON_URL.format(slot=0, lang="zh"))
    assert coach_files(coach_env) == ["777_opendota_0_zh_v2.json"]


def test_coach_works_after_match_evicted_from_memory(coach_env, monkeypatch):
    c = coach_env.client
    monkeypatch.setattr(config, "MATCH_MEMORY_CACHE_SIZE", 1)
    assert c.get("/api/fetch/888").status_code == 200
    assert "opendota_777" not in routes._matches
    r = c.get(JSON_URL.format(slot=128, lang="zh"))
    assert r.status_code == 200
    assert r.json()["match_id"] == 777


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------

def test_generation_failure_is_user_safe_and_not_cached(coach_env):
    c = coach_env.client
    coach_env.fake.error = RuntimeError("Traceback: secret key sk-123")
    r = c.get(JSON_URL.format(slot=0, lang="en"))
    assert r.status_code == 500
    assert r.json()["code"] == "coach_failed"
    assert "secret" not in r.text

    events = sse_events(c.get(SSE_URL.format(slot=0, lang="en")).text)
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "coach_failed"
    assert "secret" not in json.dumps(events)

    coach_env.fake.error = None
    assert c.get(JSON_URL.format(slot=0, lang="en")).status_code == 200
    assert len(coach_env.fake.calls) == 3


def test_generation_timeout(coach_env, monkeypatch):
    monkeypatch.setattr(config, "AI_COACH_TIMEOUT_S", 0.05)
    monkeypatch.setattr(routes, "_COACH_TIMEOUT_GRACE_S", 0.0)
    coach_env.fake.delay = 2.0
    r = coach_env.client.get(JSON_URL.format(slot=0, lang="zh"))
    assert r.status_code == 504
    assert r.json()["code"] == "coach_timeout"


# ---------------------------------------------------------------------------
# Limits: per-IP rate limit and daily quota (cold generations only)
# ---------------------------------------------------------------------------

def test_rate_limit_applies_to_cold_generations_only(coach_env, monkeypatch):
    c = coach_env.client
    monkeypatch.setenv("RATE_LIMIT_AI_COACH", "1/3600")
    security.reset_rate_limits()
    assert c.get(JSON_URL.format(slot=0, lang="zh")).status_code == 200
    assert c.get(JSON_URL.format(slot=0, lang="zh")).status_code == 200  # cached
    r = c.get(JSON_URL.format(slot=128, lang="zh"))
    assert r.status_code == 429
    assert r.json()["code"] == "rate_limited"
    assert int(r.headers["Retry-After"]) == r.json()["retry_after"] > 0

    events = sse_events(c.get(SSE_URL.format(slot=128, lang="zh")).text)
    assert events[0]["type"] == "error" and events[0]["code"] == "rate_limited"
    assert events[0]["status"] == 429 and events[0]["retry_after"] > 0
    assert len(coach_env.fake.calls) == 1


def test_daily_quota_only_counts_llm_backed_cold_reports(coach_env, monkeypatch):
    c = coach_env.client
    monkeypatch.setattr(config, "AI_COACH_DAILY_LIMIT", 1)
    # No LLM configured: the quota is not consumed.
    assert c.get(JSON_URL.format(slot=0, lang="en")).status_code == 200
    assert routes.ai_coach_daily_quota.used() == 0

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    assert c.get(JSON_URL.format(slot=0, lang="zh")).status_code == 200
    assert routes.ai_coach_daily_quota.used() == 1
    r = c.get(JSON_URL.format(slot=128, lang="zh"))
    assert r.status_code == 429
    assert r.json()["code"] == "daily_limit"
    assert r.json()["retry_after"] > 0
    assert c.get(JSON_URL.format(slot=0, lang="zh")).status_code == 200  # cached is free

    monkeypatch.setattr(config, "AI_COACH_DAILY_LIMIT", 0)  # unlimited
    assert c.get(JSON_URL.format(slot=128, lang="zh")).status_code == 200


# ---------------------------------------------------------------------------
# In-flight de-duplication and SSE behaviour
# ---------------------------------------------------------------------------

def test_concurrent_json_and_sse_requests_share_one_generation(coach_env):
    fake = coach_env.fake
    fake.gate = threading.Event()

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            tasks = [asyncio.create_task(client.get(JSON_URL.format(slot=0, lang="zh"))) for _ in range(3)]
            tasks.append(asyncio.create_task(client.get(SSE_URL.format(slot=0, lang="zh"))))
            for _ in range(300):
                await asyncio.sleep(0.01)
                job = routes._coach_jobs.get("777_opendota_0_zh")
                if job is not None and job.listeners:
                    break
            await asyncio.sleep(0.05)
            fake.gate.set()
            return await asyncio.gather(*tasks)

    responses = asyncio.run(scenario())
    assert len(fake.calls) == 1
    bodies = [r.json() for r in responses[:3]]
    assert all(b == bodies[0] for b in bodies)
    events = sse_events(responses[3].text)
    assert events[-1] == {"type": "done", "data": bodies[0]}
    assert {"type": "progress", "stage": "evaluate", "current": 2, "total": 2} in events


def test_sse_progress_heartbeat_and_no_gzip(coach_env, monkeypatch):
    monkeypatch.setattr(routes, "SSE_HEARTBEAT_S", 0.02)
    coach_env.fake.delay = 0.25
    r = coach_env.client.get(
        SSE_URL.format(slot=0, lang="zh"),
        headers={"Accept-Encoding": "gzip", "Accept": "*/*"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "content-encoding" not in r.headers
    assert r.headers["cache-control"] == "no-cache"
    assert ": ping\n\n" in r.text
    events = sse_events(r.text)
    assert events[0]["type"] == "progress"
    assert events[-1]["type"] == "done"
    assert events[-1]["data"]["player_slot"] == 0


def test_sse_disconnect_does_not_cancel_and_reconnect_attaches(coach_env, monkeypatch):
    fake = coach_env.fake
    fake.gate = threading.Event()
    monkeypatch.setenv("RATE_LIMIT_AI_COACH", "1/3600")
    security.reset_rate_limits()
    coach_id = "777_opendota_0_zh"

    async def scenario():
        request = make_request()
        cached, job = await routes._resolve_coach(request, "opendota_777", 0, "zh")
        assert cached is None
        stream = routes._sse_follow(job, "zh")
        first = await stream.__anext__()
        assert json.loads(first[len("data: "):])["type"] == "progress"
        await stream.aclose()  # client went away
        await asyncio.sleep(0.05)
        assert not job.task.done()
        assert not job.listeners

        # EventSource reconnect: attaches to the same job; no new generation
        # and no extra rate-limit hit (the limit of 1 is already used).
        cached2, job2 = await routes._resolve_coach(request, "opendota_777", 0, "zh")
        assert cached2 is None and job2 is job
        stream2 = routes._sse_follow(job2, "zh")
        replay = json.loads((await stream2.__anext__())[len("data: "):])
        assert replay == {"type": "progress", "stage": "detect", "current": 0, "total": 2}
        fake.gate.set()
        rest = [chunk async for chunk in stream2]
        return await job.task, rest

    result, rest = asyncio.run(scenario())
    assert len(fake.calls) == 1
    assert json.loads(rest[-1][len("data: "):]) == {"type": "done", "data": result}
    assert routes._coach_mem_get(coach_id) == result
    assert coach_files(coach_env) == ["777_opendota_0_zh_v2.json"]
    assert coach_id not in routes._coach_jobs
