"""Shared builders for the AI-pipeline tests (no tests in this module).

* :func:`make_match` builds a small, fully synthetic MatchData in world
  units, so the AI tests do not depend on the mock-data generator.
* :class:`FakeLLM` is an AsyncOpenAI-like client: ``client.chat.completions
  .create(**kwargs)`` records every call and delegates to a handler.
"""
from __future__ import annotations

import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import json
import re
from types import SimpleNamespace
from typing import Any, Awaitable, Callable

from backend.app.models import (
    ItemEvent,
    LaningSnapshot,
    MatchData,
    PlayerData,
    PositionSnapshot,
    TeamfightEvent,
    TeamfightPlayer,
    WardEvent,
)

CJK_RE = re.compile(r"[　-〿一-鿿＀-￯]")


def snapshots(x: float, y: float, *, cs_per_min: float = 6.0, hp: int = 600, until: int = 600,
              start: int = 0) -> list[LaningSnapshot]:
    """Laning snapshots every 5 s standing at (x, y)."""
    out = []
    for t in range(start, until, 5):
        out.append(LaningSnapshot(
            time=t, x=x, y=y, hp=hp, max_hp=600, mana=300, max_mana=300,
            level=1 + t // 120, last_hits=int(cs_per_min * t / 60), denies=0,
            net_worth=600 + 5 * t,
        ))
    return out


def make_player(slot: int, hero: str, **kw: Any) -> PlayerData:
    team = "radiant" if slot < 128 else "dire"
    data: dict[str, Any] = dict(
        player_slot=slot, hero_id=slot + 1, hero_name=hero, player_name=f"p{slot}", team=team,
        kills=2, deaths=1, assists=3, level=18, last_hits=150, denies=5,
        gold_per_min=450, xp_per_min=550, hero_damage=15000, tower_damage=1000,
        hero_healing=0, net_worth=12000, items=[], backpack=[], final_items=[],
        skills=[], wards=[], ability_events=[],
    )
    data.update(kw)
    return PlayerData(**data)


def default_players() -> list[PlayerData]:
    """Five per team with lane data; positions spread over the map."""
    return [
        make_player(0, "Juggernaut", lane=1, lane_role=1, net_worth=20000,
                    laning_snapshots=snapshots(4500, -6000)),
        make_player(1, "Puck", lane=2, lane_role=2, net_worth=16000,
                    laning_snapshots=snapshots(-500, -500)),
        make_player(2, "Tinker", lane=3, lane_role=3, net_worth=14000,
                    laning_snapshots=snapshots(-6200, 2500)),
        make_player(3, "Rubick", lane=3, lane_role=3, net_worth=7000,
                    laning_snapshots=snapshots(-6100, 2600, cs_per_min=0.5)),
        make_player(4, "Crystal Maiden", lane=1, lane_role=1, net_worth=5000,
                    laning_snapshots=snapshots(4600, -5900, cs_per_min=0.3)),
        make_player(128, "Phantom Assassin", lane=3, lane_role=1, net_worth=21000,
                    laning_snapshots=snapshots(-5500, 6000)),
        make_player(129, "Lina", lane=2, lane_role=2, net_worth=15000,
                    laning_snapshots=snapshots(500, 500)),
        make_player(130, "Axe", lane=1, lane_role=3, net_worth=13000,
                    laning_snapshots=snapshots(6000, -2500)),
        make_player(131, "Lion", lane=1, lane_role=3, net_worth=6000,
                    laning_snapshots=snapshots(6100, -2600, cs_per_min=0.4)),
        make_player(132, "Witch Doctor", lane=3, lane_role=1, net_worth=5500,
                    laning_snapshots=snapshots(-5400, 6100, cs_per_min=0.2)),
    ]


def make_match(players: list[PlayerData] | None = None, **kw: Any) -> MatchData:
    data: dict[str, Any] = dict(
        match_id=4242, duration=2100, radiant_win=False, game_mode="ranked_all_pick",
        start_time=0, radiant_score=20, dire_score=30,
        players=players if players is not None else default_players(),
        wards=[], smokes=[], teamfights=[], ability_events=[],
        radiant_gold_adv=[0] + [-200 * m for m in range(1, 36)],
        radiant_xp_adv=[0] + [-150 * m for m in range(1, 36)],
    )
    data.update(kw)
    return MatchData(**data)


def teamfight_death(slot: int, hero: str, killer_slot: int, killer: str, *,
                    start: int, x: float, y: float, death_offset: float = 5.0,
                    allies: list[tuple[int, str, float, float]] | None = None,
                    enemies: list[tuple[int, str, float, float]] | None = None) -> TeamfightEvent:
    """A teamfight in which ``slot`` dies to ``killer_slot``."""
    def tp(s: int, name: str, px: float, py: float, **extra: Any) -> TeamfightPlayer:
        base = dict(player_slot=s, hero_name=name, x=px, y=py, damage_dealt=300,
                    damage_taken=300, kills=0, deaths=0, abilities_used=[],
                    positions=[PositionSnapshot(x=px, y=py, time_offset=0.0)])
        base.update(extra)
        return TeamfightPlayer(**base)

    players = [tp(slot, hero, x, y, deaths=1, death_time_offset=death_offset)]
    players.append(tp(killer_slot, killer, x + 300, y, kills=1,
                      kill_targets=[{"target": hero, "time_offset": death_offset}]))
    for s, name, px, py in allies or []:
        players.append(tp(s, name, px, py))
    for s, name, px, py in enemies or []:
        players.append(tp(s, name, px, py))
    radiant_deaths = 1 if slot < 128 else 0
    return TeamfightEvent(start_time=start, end_time=start + 20, radiant_deaths=radiant_deaths,
                          dire_deaths=1 - radiant_deaths, gold_delta=-500 if slot < 128 else 500,
                          players=players)


def observer(x: float, y: float, slot: int, time: int = 0, destroyed: int | None = None) -> WardEvent:
    return WardEvent(ward_type="observer", x=x, y=y, time=time, player_slot=slot, destroyed_time=destroyed)


def item(name: str, key: str, time: int) -> ItemEvent:
    return ItemEvent(item_name=name, item_key=key, time=time)


# ---------------------------------------------------------------------------
# Fake LLM client
# ---------------------------------------------------------------------------

def tool_call(name: str, args: dict | None = None, call_id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args or {})))


def response(content: str | None = None, tool_calls: list | None = None) -> SimpleNamespace:
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def submit(score: str = "B", analysis: str = "Solid analysis.", **extra: str) -> SimpleNamespace:
    args = {"score": score, "analysis": analysis, **extra}
    return response(None, [tool_call("submit_evaluation", args, call_id="call_submit")])


TEXT_VERDICT_EN = (
    "Score: B\nAnalysis: Reasonable play.\nMistake: Minor positioning issue.\n"
    "Better Play: Stay closer to the team.\nLesson: Respect the map."
)
PHASE_JSON = '{"score":"A","summary":"Good phase.","highlights":["h"],"issues":["i"],"suggestions":["s"]}'


def is_phase_prompt(kwargs: dict) -> bool:
    msgs = kwargs.get("messages") or []
    return any('"score":"S|A|B|C|D"' in str(m.get("content", "")) for m in msgs)


class FakeCompletions:
    def __init__(self, handler: Callable[[dict], Awaitable[Any]]) -> None:
        self.handler = handler
        self.calls: list[dict] = []
        self.inflight = 0
        self.max_inflight = 0

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            return await self.handler(kwargs)
        finally:
            self.inflight -= 1


class FakeLLM:
    """AsyncOpenAI-like object exposing ``chat.completions.create``."""

    def __init__(self, handler: Callable[[dict], Awaitable[Any]]) -> None:
        self.completions = FakeCompletions(handler)
        self.chat = SimpleNamespace(completions=self.completions)

    @property
    def calls(self) -> list[dict]:
        return self.completions.calls


class FakeBadRequest(Exception):
    """Mimics openai.BadRequestError (HTTP 400)."""

    status_code = 400
