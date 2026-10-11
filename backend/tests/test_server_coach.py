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
import time
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
# Report identity: match key + parse source + slot + language + LLM mode.
COACH_ID = "opendota_777_opendota_0_zh_rules"
DEM = b"PBDEMS2\x00"
# What every served report says about forcing a new run: a final report cannot be refreshed.
FINAL = {"available": False, "retry_after": None}


def report_file(coach_id: str = COACH_ID, version: int = 2) -> str:
    """Cache file name: identity + match-store format + report schema version."""
    return f"{coach_id}_d{storage.MatchStore.FORMAT}_v{version}.json"


def make_player(slot: int) -> PlayerData:
    return PlayerData(
        player_slot=slot, hero_id=slot + 1, hero_name=f"Hero{slot}", hero_slug=f"hero{slot}",
        player_name=f"p{slot}", team="radiant" if slot < 128 else "dire", kills=0, deaths=0,
        assists=0, level=1, last_hits=0, denies=0, gold_per_min=0, xp_per_min=0, hero_damage=0,
        tower_damage=0, hero_healing=0, net_worth=0, items=[], backpack=[], final_items=[],
        skills=[], wards=[], ability_events=[],
    )


def make_match(match_id: int, parsed: bool = True) -> MatchData:
    return MatchData(
        match_id=match_id, duration=1200, radiant_win=True, game_mode="All Pick", start_time=0,
        radiant_score=0, dire_score=0, players=[make_player(0), make_player(128)], wards=[],
        smokes=[], teamfights=[], ability_events=[],
        gold_xp_timeline=[GoldXpSnapshot(time=60, radiant_gold=1, dire_gold=1, radiant_xp=1, dire_xp=1)]
        if parsed else [],
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
    for cache in (routes._matches, routes._coach_report_cache, routes._inflight, routes._coach_jobs,
                  routes._uploads_by_client):
        cache.clear()

    unparsed: set[int] = set()

    class FakeOpenDota:
        def name(self):
            return "opendota"

        async def parse(self, mid_str):
            return make_match(int(mid_str), parsed=int(mid_str) not in unparsed)

        def request_parse(self, mid):
            return {"job": {"jobId": 1}}

    monkeypatch.setattr(parser_mod, "OpenDotaParser", FakeOpenDota)

    class FakeReplayParser:
        """Every uploaded replay parses to a match without a match ID."""

        def name(self):
            return "clarity"

        async def parse(self, path):
            return make_match(0)

    async def fake_get_replay_parser():
        return FakeReplayParser()

    monkeypatch.setattr(routes, "_get_replay_parser", fake_get_replay_parser)

    fake = FakeCoach()
    module = types.ModuleType("backend.app.ai.coach")
    module.REPORT_SCHEMA_VERSION = 2
    module.generate_coaching_report = fake.generate_coaching_report
    monkeypatch.setitem(sys.modules, "backend.app.ai.coach", module)
    monkeypatch.setattr(ai_pkg, "coach", module, raising=False)

    with TestClient(app) as client:
        assert client.get("/api/fetch/777").status_code == 200
        yield SimpleNamespace(client=client, fake=fake, module=module, tmp=tmp_path, unparsed=unparsed)

    security.reset_rate_limits()
    for cache in (routes._matches, routes._coach_report_cache, routes._inflight, routes._coach_jobs,
                  routes._uploads_by_client):
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


@pytest.mark.parametrize("duration", [0, 4, 133])
def test_replay_cut_before_the_horn_gets_no_report(coach_env, monkeypatch, duration):
    """Nothing happened in such a replay: a report could only invent a game. The same for one cut
    seconds after the horn ("KDA 0/0/0 after 0 minutes ... grade B", a default)."""
    class PreGameParser:
        def name(self):
            return "clarity"

        async def parse(self, path):
            return make_match(8762775364).model_copy(update={
                "duration": duration, "partial": True, "winner_known": False, "gold_xp_timeline": []})

    async def get_parser():
        return PreGameParser()

    monkeypatch.setattr(routes, "_get_replay_parser", get_parser)
    monkeypatch.setenv("RATE_LIMIT_AI_COACH", "1/3600")
    c = coach_env.client
    key = c.post("/api/upload", files={"file": ("m.dem", DEM + b"p" * 64, "application/octet-stream")}).json()["file_key"]
    overview = c.get(f"/api/match/{key}").json()["analysis"]["match_overview"]
    assert overview["pre_game_only"] is (duration == 0) and overview["coach_available"] is False

    for _ in range(2):                                        # refused before any rate limit or quota
        r = c.get(f"/api/match/{key}/ai-coach/0?lang=en")
        assert r.status_code == 409 and r.json()["code"] == "coach_no_game"
        assert "before the game starts" in r.json()["detail"]
    events = sse_events(c.get(f"/api/match/{key}/ai-coach-stream/0?lang=zh").text)
    assert events == [{"type": "error", "message": security.message("coach_no_game", "zh"),
                       "code": "coach_no_game", "status": 409}]
    assert coach_env.fake.calls == [] and coach_files(coach_env) == []
    # an unknown player is still the first thing reported
    assert c.get(f"/api/match/{key}/ai-coach/5?lang=en").json()["code"] == "invalid_player_slot"


@pytest.mark.parametrize("slot", ["abc", "-1", "1.5", "１", "1234"])
def test_malformed_slot_follows_the_error_contract(coach_env, slot):
    """No FastAPI 422 with a list in "detail"; the SSE URL still answers with an event."""
    c = coach_env.client
    r = c.get(JSON_URL.format(slot=slot, lang="en"))
    assert r.status_code == 400
    assert r.json() == {"detail": "This player is not part of the match.", "code": "invalid_player_slot"}

    stream = c.get(SSE_URL.format(slot=slot, lang="zh"))
    assert stream.status_code == 200
    assert stream.headers["content-type"].startswith("text/event-stream")
    assert sse_events(stream.text) == [{
        "type": "error", "message": security.message("invalid_player_slot", "zh"),
        "code": "invalid_player_slot", "status": 400,
    }]
    assert coach_env.fake.calls == []


# ---------------------------------------------------------------------------
# Knowledge entries behind a report's sources
# ---------------------------------------------------------------------------

def test_a_knowledge_entry_can_be_read_by_the_id_a_report_lists(coach_env):
    """A report listed ``sources: [{id, title, ...}]`` and no route returned the text behind an id:
    the page showed titles under "so the basis of a verdict can be checked" with nothing to read."""
    c = coach_env.client
    zh = c.get("/api/knowledge/coaching:smoke_gank?lang=zh")
    en = c.get("/api/knowledge/coaching%3Asmoke_gank?lang=en")          # as encodeURIComponent sends it
    assert zh.status_code == en.status_code == 200
    assert zh.headers["content-type"] == "application/json"
    assert list(zh.json()) == ["id", "title", "source_type", "source_label", "patch", "text", "lang"]
    assert zh.json() | {"text": ""} == {
        "id": "coaching:smoke_gank", "title": "开雾Gank原则", "source_type": "builtin",
        "source_label": "教练原则", "patch": "", "text": "", "lang": "zh"}
    assert zh.json()["text"].startswith("## 开雾Gank原则\n- 诡计之雾使友方单位隐身")
    assert en.json() | {"text": ""} == {
        "id": "coaching:smoke_gank", "title": "Smoke Gank Principles", "source_type": "builtin",
        "source_label": "Coaching principle", "patch": "", "text": "", "lang": "en"}
    assert en.json()["text"].startswith("## Smoke Gank Principles\n- Smoke makes allied units invisible")
    # the title is the one a report shows: names in the report language, the text as the model gets it
    laning = c.get("/api/knowledge/hero_laning:skeleton_king?lang=zh").json()
    assert laning["title"] == "冥魂大帝 — 对线要点" and laning["source_label"] == "编辑整理"
    assert laning["text"].startswith("Wraith King — 对线要点：")
    # any other language value is the zh report, as on the coach routes
    assert c.get("/api/knowledge/coaching:smoke_gank?lang=ja").json()["lang"] == "zh"
    assert c.get("/api/knowledge/coaching:smoke_gank").json() == zh.json()
    assert coach_env.fake.calls == []


@pytest.mark.parametrize("entry_id", ["nope", "coaching:", "learned:matchup:axe:nobody", "a/b", "x" * 201,
                                      "..%2F..%2Fetc%2Fpasswd", "coaching:smoke_gank%00"])
def test_an_unknown_knowledge_entry_follows_the_error_contract(coach_env, entry_id):
    """Also the id of a learned entry that a later learning run dropped."""
    c = coach_env.client
    r = c.get(f"/api/knowledge/{entry_id}?lang=en")
    assert r.status_code == 404
    assert r.json() == {"detail": "This entry is no longer in the knowledge base (a later learning run may have "
                                  "replaced it).", "code": "knowledge_not_found"}
    assert c.get(f"/api/knowledge/{entry_id}?lang=zh").json()["detail"] == security.message("knowledge_not_found", "zh")


def test_knowledge_entry_route_survives_a_missing_knowledge_base(coach_env, monkeypatch):
    def broken(lang="zh"):
        raise RuntimeError("store build failed")

    monkeypatch.setattr(routes, "_knowledge_entry", lambda entry_id, lang: None)
    assert coach_env.client.get("/api/knowledge/coaching:smoke_gank").json()["code"] == "knowledge_not_found"
    # the lookup itself never raises: a failing store is "no entry", not a 500
    from backend.app.ai import knowledge_base
    monkeypatch.setattr(knowledge_base, "_get_knowledge_store", broken)
    assert knowledge_base.knowledge_entry("coaching:smoke_gank", "zh") is None


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def test_report_cached_in_memory_and_versioned_on_disk(coach_env):
    c = coach_env.client
    first = c.get(JSON_URL.format(slot=0, lang="zh"))
    assert first.status_code == 200
    assert first.json()["overall_grade"] == "B"
    assert coach_files(coach_env) == [report_file()]

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
    assert report_file(version=3) in coach_files(coach_env)

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
    assert coach_files(coach_env) == [report_file()]


def end_cooldown(coach_id: str = COACH_ID) -> None:
    """Let the refresh cooldown of the cached degraded report run out."""
    routes._coach_report_cache[coach_id].refresh_at = time.monotonic() - 1


def test_a_degraded_report_can_be_refreshed_and_only_a_degraded_one(coach_env):
    """The UI's "generate again" got the same degraded report from memory for ten minutes."""
    c, fake = coach_env.client, coach_env.fake
    fake.degraded = True
    first = c.get(JSON_URL.format(slot=0, lang="zh")).json()
    assert first["degraded"] is True and first["n"] == 1
    # the envelope says a new run can be forced, and from when
    assert first["refresh"]["available"] is True
    assert 0 < first["refresh"]["retry_after"] <= config.AI_COACH_REFRESH_COOLDOWN_S == 60

    # within the cooldown: the cached report, no run, and the time left
    early = c.get(JSON_URL.format(slot=0, lang="zh") + "&refresh=1")
    assert early.status_code == 200 and early.json()["n"] == 1 and early.json()["refresh"]["retry_after"] > 0
    assert len(fake.calls) == 1

    end_cooldown()
    assert c.get(JSON_URL.format(slot=0, lang="zh")).json()["refresh"] == {"available": True, "retry_after": 0}
    assert len(fake.calls) == 1                         # asking without refresh never starts a run
    fake.degraded = False                               # the LLM answers again
    fresh = c.get(JSON_URL.format(slot=0, lang="zh") + "&refresh=1").json()
    assert fresh["n"] == 2 and fresh["degraded"] is False and fresh["refresh"] == FINAL
    assert coach_files(coach_env) == [report_file()]    # a full report is kept like any other

    # a full report is never run again, whatever the query says
    for value in ("1", "true"):
        again = c.get(JSON_URL.format(slot=0, lang="zh") + f"&refresh={value}").json()
        assert again == fresh
    events = sse_events(c.get(SSE_URL.format(slot=0, lang="zh") + "&refresh=1").text)
    assert events == [{"type": "done", "data": fresh}]
    routes._coach_report_cache.clear()                  # ... also when it comes from disk
    assert c.get(JSON_URL.format(slot=0, lang="zh") + "&refresh=1").json() == fresh
    assert len(fake.calls) == 2


def test_refresh_of_a_degraded_report_over_sse_and_still_degraded(coach_env):
    c, fake = coach_env.client, coach_env.fake
    fake.degraded = True
    assert c.get(JSON_URL.format(slot=0, lang="zh")).json()["n"] == 1
    end_cooldown()
    events = sse_events(c.get(SSE_URL.format(slot=0, lang="zh") + "&refresh=1").text)
    assert [e["type"] for e in events][0] == "progress" and events[-1]["type"] == "done"
    again = events[-1]["data"]
    # a new run that is degraded again replaces the old report and starts a new cooldown
    assert again["n"] == 2 and again["degraded"] is True
    assert again["refresh"]["available"] is True and again["refresh"]["retry_after"] > 0
    # an EventSource that reconnects with the same URL gets that report, not another run
    assert sse_events(c.get(SSE_URL.format(slot=0, lang="zh") + "&refresh=1").text)[-1]["data"]["n"] == 2
    assert c.get(JSON_URL.format(slot=0, lang="zh")).json()["n"] == 2
    assert len(fake.calls) == 2 and coach_files(coach_env) == []
    # anything but a clear yes is no request for a new run
    end_cooldown()
    for value in ("", "0", "false", "no", "2"):
        assert c.get(JSON_URL.format(slot=0, lang="zh") + f"&refresh={value}").json()["n"] == 2
    assert len(fake.calls) == 2


def test_a_forced_run_is_a_cold_report_for_the_limits(coach_env, monkeypatch):
    c, fake = coach_env.client, coach_env.fake
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("RATE_LIMIT_AI_COACH", "2/3600")
    security.reset_rate_limits()
    llm_id = COACH_ID.replace("_rules", "_llm")
    fake.degraded = True
    assert c.get(JSON_URL.format(slot=0, lang="zh")).json()["degraded"] is True
    assert routes.ai_coach_daily_quota.used() == 1
    end_cooldown(llm_id)
    assert c.get(JSON_URL.format(slot=0, lang="zh") + "&refresh=1").json()["n"] == 2
    assert routes.ai_coach_daily_quota.used() == 2      # it spends the LLM budget like a first report

    # the per-IP limit is used up: the next forced run is refused on both routes ...
    end_cooldown(llm_id)
    refused = c.get(JSON_URL.format(slot=0, lang="zh") + "&refresh=1")
    assert refused.status_code == 429 and refused.json()["code"] == "rate_limited" and refused.json()["retry_after"] > 0
    event = sse_events(c.get(SSE_URL.format(slot=0, lang="zh") + "&refresh=1").text)[0]
    assert (event["type"], event["code"], event["status"]) == ("error", "rate_limited", 429)
    # ... while the report itself keeps being served, and still offers the refresh
    kept = c.get(JSON_URL.format(slot=0, lang="zh")).json()
    assert kept["n"] == 2 and kept["refresh"] == {"available": True, "retry_after": 0}

    # and so is the daily quota
    monkeypatch.setenv("RATE_LIMIT_AI_COACH", "0")
    security.reset_rate_limits()
    monkeypatch.setattr(config, "AI_COACH_DAILY_LIMIT", 2)
    refused = c.get(JSON_URL.format(slot=0, lang="zh") + "&refresh=1")
    assert refused.status_code == 429 and refused.json()["code"] == "daily_limit"
    assert len(fake.calls) == 2


def test_concurrent_refreshes_share_one_run(coach_env):
    fake = coach_env.fake
    fake.degraded = True
    assert coach_env.client.get(JSON_URL.format(slot=0, lang="zh")).json()["n"] == 1
    end_cooldown()
    fake.gate = threading.Event()

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            forced = [asyncio.create_task(client.get(JSON_URL.format(slot=0, lang="zh") + "&refresh=1"))
                      for _ in range(3)]
            forced.append(asyncio.create_task(client.get(SSE_URL.format(slot=0, lang="zh") + "&refresh=1")))
            for _ in range(300):
                await asyncio.sleep(0.01)
                job = routes._coach_jobs.get(COACH_ID)
                if job is not None and job.listeners:
                    break
            await asyncio.sleep(0.05)
            # meanwhile the degraded report is still there for a reader who does not ask for more
            waiting = await client.get(JSON_URL.format(slot=0, lang="zh"))
            fake.degraded = False
            fake.gate.set()
            return waiting, await asyncio.gather(*forced)

    waiting, responses = asyncio.run(scenario())
    assert waiting.json()["n"] == 1 and waiting.json()["degraded"] is True
    assert len(fake.calls) == 2                                     # one run for the four of them
    bodies = [r.json() for r in responses[:3]]
    assert all(body == bodies[0] for body in bodies) and bodies[0]["n"] == 2 and bodies[0]["refresh"] == FINAL
    assert sse_events(responses[3].text)[-1] == {"type": "done", "data": bodies[0]}
    assert COACH_ID not in routes._coach_jobs


def test_a_failed_forced_run_keeps_the_degraded_report_and_restarts_the_cooldown(coach_env):
    c, fake = coach_env.client, coach_env.fake
    fake.degraded = True
    assert c.get(JSON_URL.format(slot=0, lang="zh")).json()["n"] == 1
    end_cooldown()
    fake.error = RuntimeError("provider down")
    failed = c.get(JSON_URL.format(slot=0, lang="zh") + "&refresh=1")
    assert failed.status_code == 500 and failed.json()["code"] == "coach_failed"
    fake.error = None
    kept = c.get(JSON_URL.format(slot=0, lang="zh") + "&refresh=1").json()     # at once: not another run
    assert kept["n"] == 1 and kept["degraded"] is True and kept["refresh"]["retry_after"] > 0
    assert len(fake.calls) == 2


def test_refresh_without_a_cached_degraded_report_is_an_ordinary_request(coach_env, monkeypatch):
    c, fake = coach_env.client, coach_env.fake
    # nothing cached yet: the first report, written once
    assert c.get(JSON_URL.format(slot=0, lang="zh") + "&refresh=1").json()["n"] == 1
    assert c.get(JSON_URL.format(slot=0, lang="zh") + "&refresh=1").json()["n"] == 1
    # with the degraded cache switched off every request is a run of its own, as before
    monkeypatch.setattr(config, "AI_COACH_DEGRADED_TTL_S", 0)
    fake.degraded = True
    report = c.get(JSON_URL.format(slot=128, lang="zh") + "&refresh=1").json()
    assert report["degraded"] is True and report["refresh"] == {"available": True, "retry_after": 0}
    assert c.get(JSON_URL.format(slot=128, lang="zh")).json()["n"] == report["n"] + 1


def test_coach_works_after_match_evicted_from_memory(coach_env, monkeypatch):
    c = coach_env.client
    monkeypatch.setattr(config, "MATCH_MEMORY_CACHE_SIZE", 1)
    assert c.get("/api/fetch/888").status_code == 200
    assert "opendota_777" not in routes._matches
    r = c.get(JSON_URL.format(slot=128, lang="zh"))
    assert r.status_code == 200
    assert r.json()["match_id"] == 777


def test_coach_works_for_unparsed_match_after_request_parse(coach_env, monkeypatch):
    """The normal UI flow: fetch (basic data) → request-parse → AI coach."""
    c = coach_env.client
    coach_env.unparsed.add(555)
    assert c.get("/api/fetch/555").json()["has_parsed_data"] is False
    assert c.post("/api/request-parse/555").status_code == 200

    r = c.get("/api/match/opendota_555/ai-coach/0?lang=zh")
    assert r.status_code == 200 and r.json()["match_id"] == 555
    events = sse_events(c.get("/api/match/opendota_555/ai-coach-stream/128?lang=zh").text)
    assert events[-1]["type"] == "done"

    # ... and long after the fetch (TTL over, evicted from memory).
    monkeypatch.setattr(config, "OPENDOTA_UNPARSED_TTL_S", 0)
    routes._matches.clear()
    routes._coach_report_cache.clear()
    assert c.get("/api/match/opendota_555/ai-coach/0?lang=en").status_code == 200

    # Basic and parsed data of one match never share a report.
    basic_id = "opendota_555_opendota-basic_0_zh_rules"
    assert report_file(basic_id) in coach_files(coach_env)
    coach_env.unparsed.discard(555)
    assert c.get("/api/fetch/555").json()["has_parsed_data"] is True
    calls = len(coach_env.fake.calls)
    assert c.get("/api/match/opendota_555/ai-coach/0?lang=zh").status_code == 200
    assert len(coach_env.fake.calls) == calls + 1
    assert report_file("opendota_555_opendota_0_zh_rules") in coach_files(coach_env)


def test_coach_works_for_mock_demo_after_eviction(coach_env, monkeypatch):
    c = coach_env.client
    monkeypatch.setattr(config, "MATCH_MEMORY_CACHE_SIZE", 1)
    demo = c.get("/api/demo").json()
    assert demo["parse_source"] == "mock"
    slot = demo["analysis"]["match_overview"]["players"][0]["player_slot"]
    assert c.get("/api/fetch/888").status_code == 200       # pushes the demo out of memory
    assert "demo" not in routes._matches

    r = c.get(f"/api/match/demo/ai-coach/{slot}?lang=zh")
    assert r.status_code == 200
    match_id = demo["analysis"]["match_overview"]["match_id"]
    assert report_file(f"demo-{match_id}_mock_{slot}_zh_rules") in coach_files(coach_env)
    events = sse_events(c.get(f"/api/match/demo/ai-coach-stream/{slot}?lang=zh").text)
    assert [e["type"] for e in events] == ["done"]


def test_demo_report_is_not_served_for_another_demo_match(coach_env):
    """The key "demo" names whatever DEFAULT_REPLAY holds: a report belongs to one match."""
    c = coach_env.client
    first = make_match(1001)
    asyncio.run(routes._store_match("demo", first, "clarity"))
    assert c.get("/api/match/demo/ai-coach/0?lang=zh").json()["match_id"] == 1001
    assert report_file("demo-1001_clarity_0_zh_rules") in coach_files(coach_env)
    calls = len(coach_env.fake.calls)

    # The demo replay was replaced: same key, another match, a server restart in between
    routes._matches.clear()
    routes._coach_report_cache.clear()
    asyncio.run(routes._store_match("demo", make_match(2002), "clarity"))
    report = c.get("/api/match/demo/ai-coach/0?lang=zh").json()
    assert report["match_id"] == 2002 and len(coach_env.fake.calls) == calls + 1
    assert report_file("demo-2002_clarity_0_zh_rules") in coach_files(coach_env)
    # ... while the same match is still served from the cache
    routes._coach_report_cache.clear()
    assert c.get("/api/match/demo/ai-coach/0?lang=zh").json()["match_id"] == 2002
    assert len(coach_env.fake.calls) == calls + 1


def test_reports_cached_for_older_match_data_are_not_served(coach_env):
    """A report is written from the match data of one store format (and by the coach of that round)."""
    c = coach_env.client
    assert c.get("/api/fetch/777").status_code == 200
    stale = {"match_id": 777, "player_slot": 0, "lang": "zh", "overall_grade": "S", "summary": "stale",
             "death_analyses": [], "llm_used": False, "degraded": False, "report_version": 2}
    for fmt in range(1, storage.MatchStore.FORMAT):
        old_name = f"{COACH_ID}_d{fmt}_v2.json"
        (config.COACH_CACHE_DIR / old_name).write_text(json.dumps(stale), encoding="utf-8")
    report = c.get(JSON_URL.format(slot=0, lang="zh")).json()
    assert report["summary"] == "ok" and len(coach_env.fake.calls) == 1
    assert report_file() in coach_files(coach_env)


def test_uploads_never_share_reports_through_the_match_id(coach_env):
    """Two different replays without a match ID (match_id 0) are two different games."""
    c = coach_env.client
    keys = []
    for tag in (b"a", b"b"):
        r = c.post("/api/upload", files={"file": ("m.dem", DEM + tag * 64, "application/octet-stream")})
        assert r.status_code == 200 and r.json()["match_id"] == 0
        keys.append(r.json()["file_key"])
    assert keys[0] != keys[1]

    first = c.get(f"/api/match/{keys[0]}/ai-coach/0?lang=zh").json()
    second = c.get(f"/api/match/{keys[1]}/ai-coach/0?lang=zh").json()
    assert len(coach_env.fake.calls) == 2
    assert first["n"] == 1 and second["n"] == 2          # not the first user's report
    assert coach_files(coach_env) == sorted(report_file(f"{key}_clarity_0_zh_rules") for key in keys)

    # Each is still cached on its own.
    assert c.get(f"/api/match/{keys[0]}/ai-coach/0?lang=zh").json() == first
    assert len(coach_env.fake.calls) == 2


def test_rules_only_reports_are_not_served_once_an_llm_is_configured(coach_env, monkeypatch):
    c = coach_env.client
    coach_env.fake.degraded = False
    rules = c.get(JSON_URL.format(slot=0, lang="zh")).json()      # no key: rule-based report
    assert coach_files(coach_env) == [report_file()]
    assert len(coach_env.fake.calls) == 1

    # The operator fixes .env and restarts: the cached rule-based report must
    # not shadow the LLM report forever.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    routes._coach_report_cache.clear()
    llm = c.get(JSON_URL.format(slot=0, lang="zh")).json()
    assert len(coach_env.fake.calls) == 2 and llm["n"] == 2
    assert report_file("opendota_777_opendota_0_zh_llm") in coach_files(coach_env)
    routes._coach_report_cache.clear()
    assert c.get(JSON_URL.format(slot=0, lang="zh")).json() == llm    # cached like any other
    assert len(coach_env.fake.calls) == 2

    # Without the key the rule-based report is served again, not regenerated.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    routes._coach_report_cache.clear()
    assert c.get(JSON_URL.format(slot=0, lang="zh")).json() == rules
    assert len(coach_env.fake.calls) == 2

    # A key without the openai SDK is not an LLM either (the pipeline then
    # writes rule-based reports): same identity as "no key", no quota used.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(routes, "_openai_installed", lambda: False)
    routes._coach_report_cache.clear()
    assert c.get("/api/capabilities").json()["ai_coach_llm"] is False
    used = routes.ai_coach_daily_quota.used()
    assert c.get(JSON_URL.format(slot=0, lang="zh")).json() == rules
    assert routes.ai_coach_daily_quota.used() == used


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
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")     # LLM-backed reports: the limit that guards the budget
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


def test_rule_based_reports_are_not_held_to_the_llm_limit(coach_env, monkeypatch):
    """Without a key a report costs milliseconds, and one match has ten players in two languages:
    under the LLM limit (10 an hour) the 11th cold report was answered 429, retry in an hour."""
    c = coach_env.client
    assert security.RATE_LIMIT_DEFAULTS["ai_coach"] == "10/3600"
    # 120 an hour was used up by looking through six matches in both languages, and the next
    # report (0.1 s of work) was refused for up to an hour: now a short window with more room.
    count, window = security.parse_rate_spec(security.RATE_LIMIT_DEFAULTS["ai_coach_rules"])
    assert security.RATE_LIMIT_DEFAULTS["ai_coach_rules"] == "200/600" and count >= 200 and window <= 600
    monkeypatch.setenv("RATE_LIMIT_AI_COACH", "1/3600")
    monkeypatch.delenv("RATE_LIMIT_AI_COACH_RULES")                 # the default: 200 in ten minutes
    security.reset_rate_limits()
    cold = [(slot, lang) for lang in ("zh", "en") for slot in (0, 128)]     # every player of the test match
    for slot, lang in cold:
        assert c.get(JSON_URL.format(slot=slot, lang=lang)).status_code == 200, (slot, lang)
    assert len(coach_env.fake.calls) == 4
    # They have a limit of their own (RATE_LIMIT_AI_COACH_RULES), spent by cold reports only
    for path in config.COACH_CACHE_DIR.iterdir():
        path.unlink()
    routes._coach_report_cache.clear()
    monkeypatch.setenv("RATE_LIMIT_AI_COACH_RULES", "2/3600")
    security.reset_rate_limits()
    assert [c.get(JSON_URL.format(slot=slot, lang=lang)).status_code for slot, lang in cold[:3]] == [200, 200, 429]
    assert c.get(JSON_URL.format(slot=0, lang="zh")).status_code == 200          # cached: free
    events = sse_events(c.get(SSE_URL.format(slot=128, lang="en")).text)
    assert events[0]["code"] == "rate_limited" and events[0]["status"] == 429 and events[0]["retry_after"] > 0


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
                job = routes._coach_jobs.get(COACH_ID)
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
    coach_id = COACH_ID

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
    # the pipeline's report, served with the envelope's word on refreshing it
    assert "refresh" not in result
    assert json.loads(rest[-1][len("data: "):]) == {"type": "done", "data": {**result, "refresh": FINAL}}
    assert routes._coach_mem_get(coach_id) == result
    assert coach_files(coach_env) == [report_file()]
    assert coach_id not in routes._coach_jobs
