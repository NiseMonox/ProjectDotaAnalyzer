"""LLM path against a strict fake endpoint and a fake reasoning model.

Covers what the live DeepSeek run showed: reasoning tokens eat ``max_tokens``
(``finish_reason`` "length", empty or cut answers), a forced ``tool_choice``
is refused while the model reasons, placeholder verdicts, tool results cut
before the model saw them, and a truncated knowledge block. The fake rejects
what the real API rejects, so a broken conversation fails the test.
"""
from __future__ import annotations

import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import asyncio
import json
import logging

import pytest

from backend.app.ai import agent, coach, context_builder, evaluator
from backend.app.ai.models import DeathEvent, DecisionPoint
from test_ai_pipeline_helpers import (
    ANALYSIS_EN,
    ANALYSIS_ZH,
    PHASE_JSON,
    TEXT_VERDICT_EN,
    FakeBadRequest,
    default_players,
    install_fake,
    llm_env,  # noqa: F401  (fixture)
    make_match,
    protocol_violations,
    response,
    submit,
    teamfight_death,
    tool_call,
)

pytestmark = pytest.mark.usefixtures("llm_env")

THINKING_OFF = {"thinking": {"type": "disabled"}}
FORCED = {"type": "function", "function": {"name": "submit_evaluation"}}
SUBMIT_TOOL = {"type": "function", "function": {"name": "submit_evaluation", "parameters": {"type": "object"}}}


def death_context(lang: str = "en"):
    players = default_players()
    tf = teamfight_death(1, "Puck", 129, "Lina", start=1500, x=1000.0, y=1000.0,
                         enemies=[(128, "Phantom Assassin", 1200.0, 1100.0)])
    match = make_match(players, teamfights=[tf])
    death = DeathEvent(time=1505, player_slot=1, hero_name="Puck", killer_hero="Lina",
                       x=1000.0, y=1000.0, is_teamfight=True, teamfight_idx=0)
    brief = "Killed by Lina in a teamfight" if lang == "en" else "团战中被Lina击杀"
    dp = DecisionPoint(time=1505, point_type="death", player_slot=1, hero_name="Puck",
                       severity="critical", brief=brief, related_death=death)
    return match, context_builder.build_context(match, dp, lang=lang)


def _tool_messages(call: dict) -> list[dict]:
    return [m for m in call["messages"] if m.get("role") == "tool"]


def _run_agent(match, ctx, knowledge="", lang="en"):
    return asyncio.run(agent.run_decision_agent(match, ctx, knowledge, lang=lang))


# ---------------------------------------------------------------------------
# The fake is as strict as a real endpoint
# ---------------------------------------------------------------------------

def _request(messages, **extra):
    return {"model": "m", "messages": messages, **extra}


def _assistant(*ids, args="{}"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": i, "type": "function", "function": {"name": "get_recent_events", "arguments": args}} for i in ids]}


def _tool(call_id, content="result"):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


USER = {"role": "user", "content": "brief"}


@pytest.mark.parametrize("request_kwargs, expected", [
    (_request([USER, _assistant("a", "b"), _tool("a"), USER]), "were not answered"),
    (_request([USER, _assistant("a"), _tool("a"), _tool("a")]), "unknown or already answered"),
    (_request([USER, _tool("ghost")]), "unknown or already answered"),
    (_request([USER, _assistant("a")]), "were not answered"),
    (_request([USER, _assistant("a", args='{"x": ')]), "not a JSON object"),
    (_request([USER, _assistant("a", "a"), _tool("a")]), "duplicate tool_call id"),
    (_request([USER, {"role": "assistant", "content": ""}, USER]), "without content"),
    (_request([USER, {"role": "assistant", "content": "done"}]), "last message is an assistant"),
    (_request([USER, _assistant("a"), _tool("a", content=None)]), "tool content must be a string"),
    (_request([USER], tools=[SUBMIT_TOOL], tool_choice={"type": "function", "function": {"name": "other"}}),
     "not in tools"),
    (_request([USER], tool_choice=FORCED), "only allowed when tools"),
    (_request([USER], tools=[SUBMIT_TOOL], tool_choice={"name": "submit_evaluation"}), "invalid tool_choice"),
    (_request([USER], tools=[]), "non-empty list"),
    (_request([USER], purpose="agent"), "unknown request arguments"),
    (_request([USER], max_tokens=0), "invalid max_tokens"),
    (_request([]), "non-empty list"),
])
def test_fake_rejects_what_a_real_endpoint_rejects(request_kwargs, expected):
    errs = protocol_violations(request_kwargs)
    assert any(expected in e for e in errs), errs


def test_fake_accepts_a_valid_tool_conversation():
    ok = _request([{"role": "system", "content": "s"}, USER, _assistant("a", "b"), _tool("b"), _tool("a"), USER],
                  tools=[SUBMIT_TOOL], tool_choice=FORCED, max_tokens=100, temperature=0.3,
                  extra_body=THINKING_OFF)
    assert protocol_violations(ok) == []


def test_fake_answers_a_broken_request_with_http_400(monkeypatch):
    async def handler(kwargs):
        return response("unreachable")

    fake = install_fake(monkeypatch, handler, allow_violations=True)

    async def send():
        return await evaluator.llm_chat(max_tokens=50, messages=[USER, _tool("ghost")])

    with pytest.raises(FakeBadRequest):
        asyncio.run(send())
    assert fake.violations


def test_thinking_fake_spends_reasoning_from_max_tokens(monkeypatch):
    async def handler(kwargs):
        return response(TEXT_VERDICT_EN, reasoning_tokens=3000)

    fake = install_fake(monkeypatch, handler, thinking=True)

    async def ask(max_tokens, **extra):
        return await evaluator.llm_chat(max_tokens=max_tokens, messages=[USER], **extra)

    starved = asyncio.run(ask(3000))       # the live failure: 3000 reasoning tokens, no answer
    assert evaluator.finish_reason(starved) == "length" and starved.choices[0].message.content == ""
    assert starved.usage.completion_tokens_details.reasoning_tokens == 3000
    cut = asyncio.run(ask(3010))
    assert evaluator.is_truncated(cut) and 0 < len(cut.choices[0].message.content) < len(TEXT_VERDICT_EN)
    whole = asyncio.run(ask(8192))
    assert not evaluator.is_truncated(whole) and whole.choices[0].message.content == TEXT_VERDICT_EN
    direct = asyncio.run(ask(3000, thinking=False))
    assert not evaluator.is_truncated(direct) and fake.calls[-1]["extra_body"] == THINKING_OFF
    assert fake.usages[-1]["reasoning_tokens"] == 0


# ---------------------------------------------------------------------------
# Token budgets and cut-off answers
# ---------------------------------------------------------------------------

def test_default_budgets_leave_room_for_reasoning(monkeypatch):
    """The live single-prompt run failed on every call with max_tokens=3000."""
    monkeypatch.setenv("AI_COACH_AGENT", "0")

    async def handler(kwargs):
        return response(TEXT_VERDICT_EN, reasoning_tokens=3000)

    fake = install_fake(monkeypatch, handler, thinking=True)
    match, ctx = death_context()
    ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="en", match=match))
    assert ev.used_llm and not ev.degraded and ev.lesson == "Respect the map."
    assert len(fake.calls) == 1 and fake.usages[0]["finish_reason"] == "stop"


def test_cut_off_single_prompt_answer_is_retried_with_more_room(monkeypatch):
    monkeypatch.setenv("AI_COACH_AGENT", "0")
    monkeypatch.setenv("AI_COACH_MAX_TOKENS", "3000")

    async def handler(kwargs):
        return response(TEXT_VERDICT_EN, reasoning_tokens=3000)

    fake = install_fake(monkeypatch, handler, thinking=True)
    match, ctx = death_context()
    ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="en", match=match))
    assert ev.used_llm and not ev.degraded and ev.analysis == ANALYSIS_EN
    assert [u["finish_reason"] for u in fake.usages] == ["length", "stop"]
    first, retry = fake.calls
    assert first["max_tokens"] == 3000 and "extra_body" not in first
    assert retry["max_tokens"] == 6000 and retry["extra_body"] == THINKING_OFF
    assert retry["messages"] == first["messages"]  # nothing of the cut answer is fed back


def test_cut_off_answer_is_never_accepted_as_a_verdict(monkeypatch):
    """Live r2: 'Score: D / Analysis: ...the player entered the' was stored as an LLM verdict."""
    monkeypatch.setenv("AI_COACH_AGENT", "0")

    async def handler(kwargs):
        return response("Score: D\nAnalysis: This was a critical mid-game death caused by a bad fight "
                        "selection. From 18:35 onwards the player entered the", finish_reason="length")

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="en", match=match))
    assert ev.degraded and not ev.used_llm
    assert "player entered the" not in ev.analysis
    assert len(fake.calls) == 2  # one bounded retry, then rules


def test_answer_without_finish_reason_must_be_complete(monkeypatch):
    """An endpoint that reports no finish_reason: a missing last field means cut off."""
    monkeypatch.setenv("AI_COACH_AGENT", "0")

    async def handler(kwargs):
        reply = response(f"Score: C\nAnalysis: {ANALYSIS_EN}\nMistake: Walked in al")
        reply.choices[0].finish_reason = None
        return reply

    install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="en", match=match))
    assert ev.degraded and not ev.used_llm


@pytest.mark.parametrize("reason, cut", [
    ("stop", False), ("STOP", False), ("tool_calls", False), ("eos", False), ("end_turn", False), ("", False),
    ("length", True), ("LENGTH", True), ("max_tokens", True), ("content_filter", True),
    ("insufficient_system_resource", True),
])
def test_only_known_cut_off_reasons_count_as_truncation(reason, cut):
    reply = response(TEXT_VERDICT_EN, finish_reason="stop")
    reply.choices[0].finish_reason = reason or None
    assert evaluator.is_truncated(reply) is cut
    assert evaluator.ended_normally(reply) is (reason.lower() in ("stop", "tool_calls"))


@pytest.mark.parametrize("agent_on", ["1", "0"])
@pytest.mark.parametrize("reason", ["STOP", "eos", "end_turn"])
def test_endpoint_with_its_own_finish_reason_is_judged_by_the_answer(monkeypatch, agent_on, reason):
    """Any finish_reason other than stop / tool_calls counted as a cut-off answer: behind another
    OpenAI-compatible server every answer was retried, rejected, and every report rules-only."""
    monkeypatch.setenv("AI_COACH_AGENT", agent_on)

    async def handler(kwargs):
        reply = submit("C", ANALYSIS_EN) if kwargs.get("tools") else response(TEXT_VERDICT_EN)
        reply.choices[0].finish_reason = reason
        return reply

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="en", match=match))
    assert ev.used_llm and not ev.degraded and ev.analysis == ANALYSIS_EN
    assert len(fake.calls) == 1                       # accepted as it is: no retry


def test_own_finish_reason_does_not_hide_a_cut_off_answer(monkeypatch):
    monkeypatch.setenv("AI_COACH_AGENT", "0")

    async def handler(kwargs):
        reply = response(f"Score: C\nAnalysis: {ANALYSIS_EN}\nMistake: Walked in al")
        reply.choices[0].finish_reason = "eos"
        return reply

    install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="en", match=match))
    assert ev.degraded and not ev.used_llm


def test_agent_round_cut_off_is_retried_once(monkeypatch):
    """Live r1: submit_evaluation arguments cut mid-JSON were answered as 'invalid arguments'."""
    monkeypatch.setenv("AI_COACH_MAX_TOKENS", "2600")

    async def handler(kwargs):
        if len(fake.calls) == 1:
            return response(None, [tool_call("get_recent_events", call_id="e1")], reasoning_tokens=100)
        return submit("C", ANALYSIS_EN * 4, reasoning_tokens=2500)

    fake = install_fake(monkeypatch, handler, thinking=True)
    match, ctx = death_context()
    ev = _run_agent(match, ctx)
    assert ev.used_llm and ev.score == "C" and ev.analysis == ANALYSIS_EN * 4
    assert [u["finish_reason"] for u in fake.usages] == ["tool_calls", "length", "tool_calls"]
    cut, retry = fake.calls[1], fake.calls[2]
    assert retry["max_tokens"] == evaluator.retry_max_tokens(2600) and retry["extra_body"] == THINKING_OFF
    # The cut turn (invalid JSON arguments) never enters the history.
    assert retry["messages"] == cut["messages"]


def test_agent_keeps_the_larger_budget_after_a_cut_off(monkeypatch):
    """A model that out-reasons the budget once would do it again in the verdict round."""
    monkeypatch.setenv("AI_COACH_MAX_TOKENS", "2000")

    async def handler(kwargs):
        if not any(m["role"] == "tool" for m in kwargs["messages"]):
            return response(None, [tool_call("get_recent_events", call_id="e1")])
        return submit("C", ANALYSIS_EN)

    fake = install_fake(monkeypatch, handler, thinking=True, reasoning_tokens=2600)
    match, ctx = death_context()
    ev = _run_agent(match, ctx)
    assert ev.used_llm and ev.analysis == ANALYSIS_EN
    assert [u["finish_reason"] for u in fake.usages] == ["length", "tool_calls", "tool_calls"]
    assert [c["max_tokens"] for c in fake.calls] == [2000, 4048, 4048]
    assert "extra_body" not in fake.calls[0] and fake.calls[1]["extra_body"] == fake.calls[2]["extra_body"] == THINKING_OFF


def test_direct_answer_when_the_deadline_is_close(monkeypatch):
    async def handler(kwargs):
        return response("ok")

    fake = install_fake(monkeypatch, handler, thinking=True)

    async def ask(seconds):
        with evaluator.llm_deadline(seconds):
            await evaluator.llm_chat(max_tokens=4096, messages=[USER], direct_below=evaluator.DIRECT_ANSWER_BELOW_S)

    asyncio.run(ask(300))
    asyncio.run(ask(40))
    assert "extra_body" not in fake.calls[0] and fake.calls[1]["extra_body"] == THINKING_OFF


def test_agent_gives_up_after_a_second_cut_off(monkeypatch):
    async def handler(kwargs):
        return response("Score: C\nAnalysis: The player", finish_reason="length")

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    with pytest.raises(evaluator.LLMTruncatedError):
        _run_agent(match, ctx)
    assert len(fake.calls) == 2
    ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="en", match=match))
    assert ev.degraded and not ev.used_llm


def test_cut_off_phase_review_is_retried_then_degraded(monkeypatch):
    replies = iter([response('{"score":"A","summary":"Strong ph', finish_reason="length"), response(PHASE_JSON)])

    async def handler(kwargs):
        return next(replies)

    fake = install_fake(monkeypatch, handler)

    def review():
        return asyncio.run(evaluator.evaluate_phase(
            hero_name="Puck", position_label="Position 2 (Mid)", team="radiant", phase_key="laning",
            phase_time_range="0-10 min", phase_stats_dict={"kills": 1, "deaths": 0, "start_min": 0, "end_min": 10},
            lang="en"))

    good = review()
    assert good.used_llm and not good.degraded and good.score == "A" and (good.start_min, good.end_min) == (0, 10)
    assert fake.calls[0]["max_tokens"] == evaluator.DEFAULT_PHASE_MAX_TOKENS
    assert fake.calls[1]["max_tokens"] == 8192 and fake.calls[1]["extra_body"] == THINKING_OFF

    async def always_cut(kwargs):
        return response('{"score":"A","summary":"Strong ph', finish_reason="length")

    install_fake(monkeypatch, always_cut)
    bad = review()
    assert bad.degraded and not bad.used_llm


@pytest.mark.parametrize("payload", [
    '{"score":"Average","summary":"A decent phase with steady farm and one avoidable death."}',
    '{"score":"B","summary":"<1-2 sentences>"}',
    '{"score":"B","summary":"placeholder"}',
    '{"score":"B","summary":"这个阶段打得不错，补刀稳定而且没有死亡。"}',
    'not json at all',
])
def test_unusable_phase_review_is_degraded(monkeypatch, payload):
    async def handler(kwargs):
        return response(payload)

    fake = install_fake(monkeypatch, handler)
    review = asyncio.run(evaluator.evaluate_phase(
        hero_name="Puck", position_label="Position 2 (Mid)", team="radiant", phase_key="laning",
        phase_time_range="0-10 min", phase_stats_dict={"kills": 1, "deaths": 0}, lang="en"))
    assert review.degraded and not review.used_llm and payload not in review.summary
    assert len(fake.calls) == 2


def test_short_but_real_phase_summary_is_accepted(monkeypatch):
    async def handler(kwargs):
        return response('{"score":"b","summary":"Quiet but solid laning.","highlights":["x", 3, ""],"issues":"none"}')

    install_fake(monkeypatch, handler)
    review = asyncio.run(evaluator.evaluate_phase(
        hero_name="Puck", position_label="Position 2 (Mid)", team="radiant", phase_key="laning",
        phase_time_range="0-10 min", phase_stats_dict={"kills": 1, "deaths": 0}, lang="en"))
    assert review.used_llm and review.score == "B" and review.summary == "Quiet but solid laning."
    assert review.highlights == ["x"] and review.issues == []  # only real strings survive


def test_max_tokens_above_the_endpoint_limit_is_learnt(monkeypatch):
    monkeypatch.setenv("AI_COACH_AGENT", "0")

    async def handler(kwargs):
        return response(TEXT_VERDICT_EN)

    fake = install_fake(monkeypatch, handler, max_tokens_limit=4096)
    match, ctx = death_context()
    ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="en", match=match))
    assert ev.used_llm and not ev.degraded
    assert [c["max_tokens"] for c in fake.calls] == [8192, 4096]
    asyncio.run(coach._safe_evaluate(ctx, "", lang="en", match=match))
    assert fake.calls[-1]["max_tokens"] == 4096 and len(fake.calls) == 3  # remembered for this model


# ---------------------------------------------------------------------------
# The final round
# ---------------------------------------------------------------------------

def test_final_round_is_forced_on_a_reasoning_model(monkeypatch):
    """DeepSeek answers 400 to a forced tool_choice unless thinking is switched off."""
    monkeypatch.setenv("AI_COACH_AGENT_MAX_STEPS", "2")

    async def handler(kwargs):
        if kwargs.get("tool_choice"):
            return submit("C", ANALYSIS_EN)
        return response(None, [tool_call("analyze_vision_and_fog", call_id="v1")])

    fake = install_fake(monkeypatch, handler, thinking=True)
    match, ctx = death_context()
    ev = _run_agent(match, ctx)
    assert ev.used_llm and ev.analysis == ANALYSIS_EN
    assert len(fake.calls) == 2  # no rejected request, no unforced retry
    assert fake.calls[1]["tool_choice"] == FORCED and fake.calls[1]["extra_body"] == THINKING_OFF
    assert not agent._forced_choice_unsupported_models and not evaluator._no_thinking_switch


def test_final_round_on_an_endpoint_without_the_thinking_switch(monkeypatch):
    monkeypatch.setenv("AI_COACH_AGENT_MAX_STEPS", "2")

    async def handler(kwargs):
        if len(kwargs["tools"]) == 1:
            return submit("C", ANALYSIS_EN)
        return response(None, [tool_call("analyze_vision_and_fog", call_id="v1")])

    fake = install_fake(monkeypatch, handler, thinking=True, thinking_switch=False, reasoning_tokens=500)
    match, ctx = death_context()
    ev = _run_agent(match, ctx)
    assert ev.used_llm
    # tools round, switch rejected, forced choice rejected while reasoning, unforced verdict
    assert len(fake.calls) == 4
    assert "extra_body" not in fake.calls[2] and "tool_choice" not in fake.calls[3]
    model = evaluator.get_model()
    assert model in evaluator._no_thinking_switch and model in agent._forced_choice_unsupported_models
    # Remembered: the next decision does not repeat the two rejected requests.
    _run_agent(match, ctx)
    assert len(fake.calls) == 6


def test_final_round_ignores_tools_that_were_not_offered(monkeypatch):
    """Live r3: with only submit_evaluation offered the model called other tools, and they ran."""
    monkeypatch.setenv("AI_COACH_AGENT_MAX_STEPS", "2")
    agent._forced_choice_unsupported_models.add(evaluator.get_model())  # the round cannot be forced

    async def handler(kwargs):
        n = len(fake.calls)
        if n == 1:
            return response(None, [tool_call("get_recent_events", call_id="e1")])
        if n == 2:
            return response(None, [tool_call("analyze_vision_and_fog", call_id="v1"),
                                   tool_call("get_nearby_heroes", call_id="n1")])
        return submit("C", ANALYSIS_EN)

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = _run_agent(match, ctx)
    assert ev.used_llm and len(fake.calls) == 3
    assert [s["tool"] for s in ev.agent_trace] == ["get_recent_events", "submit_evaluation"]
    answers = {m["tool_call_id"]: m["content"] for m in _tool_messages(fake.calls[2])}
    assert "not available" in answers["v1"] and "not available" in answers["n1"]
    assert "Vision" not in answers["v1"]


def test_final_round_without_a_verdict_fails_after_the_reask(monkeypatch):
    monkeypatch.setenv("AI_COACH_AGENT_MAX_STEPS", "2")
    agent._forced_choice_unsupported_models.add(evaluator.get_model())

    async def handler(kwargs):
        return response(None, [tool_call("analyze_vision_and_fog", call_id=f"v{len(fake.calls)}")])

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    with pytest.raises(agent.AgentNoVerdictError):
        _run_agent(match, ctx)
    assert len(fake.calls) == 3  # round, final round, one re-ask


def test_agent_does_not_start_without_time_to_investigate(monkeypatch):
    """Under 30 s the agent used to send one forced request graded from the brief alone."""
    async def handler(kwargs):
        return response(TEXT_VERDICT_EN)

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()

    async def run():
        with evaluator.llm_deadline(20):
            with pytest.raises(agent.AgentSkippedError):
                await agent.run_decision_agent(match, ctx, "", lang="en")
            return await coach._safe_evaluate(ctx, "KNOWLEDGE BLOCK", lang="en", match=match)

    ev = asyncio.run(run())
    assert ev.used_llm and not ev.degraded
    assert len(fake.calls) == 1 and "tools" not in fake.calls[0]
    prompt = fake.calls[0]["messages"][0]["content"]
    assert "KNOWLEDGE BLOCK" in prompt and "## Nearby Enemies" in prompt  # the full context, not the brief
    assert fake.calls[0]["extra_body"] == THINKING_OFF  # too little time left for hidden reasoning
    assert agent.is_agent_enabled()  # nothing is wrong with the model


# ---------------------------------------------------------------------------
# Verdict validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("args, problem", [
    ({"score": "B", "analysis": "Placeholder"}, "placeholder"),
    ({"score": "B", "analysis": "placeholder."}, "placeholder"),
    ({"score": "B", "analysis": "TODO"}, "placeholder"),
    ({"score": "B", "analysis": "..."}, "placeholder"),
    ({"score": "B", "analysis": ""}, "empty"),
    ({"score": "B", "analysis": "Bad fight."}, "short"),
    ({"score": "B", "analysis": agent._TOOL_DESCRIPTIONS["en"]["p_analysis"]}, "placeholder"),
    ({"score": "S|A|B|C|D", "analysis": ANALYSIS_EN}, "score"),
    ({"score": "Average", "analysis": ANALYSIS_EN}, "score"),
    ({"score": "", "analysis": ANALYSIS_EN}, "score"),
    ({"score": "B", "analysis": {"text": ANALYSIS_EN}}, "type"),
    ({"score": "B", "analysis": ANALYSIS_EN, "lesson": ["a"]}, "type"),
    ({"score": "B", "analysis": ANALYSIS_ZH}, "language"),
    ({"score": "B", "analysis": ANALYSIS_EN, "mistake": "站位太靠前"}, "language"),
    ({"score": "B", "analysis": ANALYSIS_EN + " analyze_vision_and_fog showed every enemy."}, "internal"),
    ({"score": "B", "analysis": ANALYSIS_EN, "lesson": "Call trace_death_cause earlier."}, "internal"),
    ({}, "format"),
])
def test_unusable_submit_arguments_are_named(args, problem):
    assert agent._submit_problem(args, "en") == problem


def test_usable_submit_arguments_pass():
    assert agent._submit_problem({"score": "b+", "analysis": ANALYSIS_EN}, "en") == ""
    assert agent._submit_problem({"score": "B", "analysis": ANALYSIS_EN, "mistake": "None"}, "en") == ""
    assert agent._submit_problem({"score": "B", "analysis": ANALYSIS_EN, "mistake": "无"}, "en") == ""  # localized later
    zh = {"score": "C", "analysis": "Puck 在 25:05 被 Lina 和 Phantom Assassin 包夹击杀，附近没有队友。",
          "mistake": "无", "better_play": "继续保持", "lesson": "GPM 520"}
    assert agent._submit_problem(zh, "zh") == ""
    assert agent._submit_problem({"score": "C", "analysis": ANALYSIS_EN}, "zh") == "language"


def test_placeholder_verdict_is_reasked_once_then_rules(monkeypatch):
    """Live r3: {"score": "B", "analysis": "Placeholder"} was shown to the user as the LLM analysis."""
    async def handler(kwargs):
        return submit("B", "Placeholder", mistake=None, better_play=None, lesson=None,
                      call_id=f"s{len(fake.calls)}")

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    with pytest.raises(agent.AgentNoVerdictError):
        _run_agent(match, ctx)
    assert len(fake.calls) == 2
    note = _tool_messages(fake.calls[1])[-1]["content"]
    assert "placeholder" in note and "submit_evaluation" in note

    ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="en", match=match))
    assert ev.degraded and not ev.used_llm and "laceholder" not in ev.analysis


def test_rejected_verdict_followed_by_a_real_one_is_used(monkeypatch):
    async def handler(kwargs):
        if len(fake.calls) == 1:
            return submit("B", "Placeholder", call_id="s1")
        return submit("C", ANALYSIS_EN, call_id="s2")

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = _run_agent(match, ctx)
    assert ev.used_llm and ev.score == "C" and ev.analysis == ANALYSIS_EN and len(fake.calls) == 2


def test_verdict_in_the_wrong_language_is_reasked(monkeypatch):
    async def handler(kwargs):
        if len(fake.calls) == 1:
            return submit("C", ANALYSIS_EN, call_id="s1")
        return submit("C", ANALYSIS_ZH, mistake="None", better_play="Keep it up", lesson="先看小地图再进场。",
                      call_id="s2")

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context("zh")
    ev = _run_agent(match, ctx, lang="zh")
    assert ev.analysis == ANALYSIS_ZH and len(fake.calls) == 2
    assert "中文" in _tool_messages(fake.calls[1])[-1]["content"]
    # Stock answers of the other language are localized, not shown as English words.
    assert (ev.mistake, ev.better_play) == ("无", "继续保持")


def test_text_verdict_is_validated_like_a_submit(monkeypatch):
    async def handler(kwargs):
        if len(fake.calls) == 1:
            return response("Score: B\nAnalysis: placeholder\nMistake: None\nBetter Play: Keep it up\nLesson: tbd")
        return response(TEXT_VERDICT_EN)

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = _run_agent(match, ctx)
    assert ev.analysis == ANALYSIS_EN and len(fake.calls) == 2
    assert fake.calls[1]["messages"][-1]["role"] == "user"
    assert "not accepted" in fake.calls[1]["messages"][-1]["content"]


def test_fields_the_model_left_out_stay_empty(monkeypatch):
    """The agent itself never fills a missing field with text from somewhere else."""
    async def handler(kwargs):
        return submit("A", "Unavoidable: a smoked three-man gank, and Puck reacted correctly.",
                      mistake=None, better_play=None, lesson=None)

    install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = _run_agent(match, ctx)
    assert ev.score == "A" and (ev.mistake, ev.better_play, ev.lesson) == ("", "", "")
    # ...and neither does the coach: rule text next to an LLM analysis can contradict it.
    ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="en", match=match))
    assert ev.used_llm and (ev.mistake, ev.better_play, ev.lesson) == ("", "", "")
    required = agent._tools_schema("en", True)[-1]["function"]["parameters"]["required"]
    assert required == ["score", "analysis", "mistake", "better_play", "lesson"]


# ---------------------------------------------------------------------------
# Conversation hygiene
# ---------------------------------------------------------------------------

def test_knowledge_block_reaches_the_model_untruncated(monkeypatch):
    """The 6000-char tool cap cut the ability/item tail off every English knowledge block."""
    knowledge = "<reference>\n" + "generic coaching text. " * 380 + "\n## Ability & Item Info\nTAIL-MARKER\n</reference>"
    assert 8000 < len(knowledge) < evaluator.MAX_KNOWLEDGE_CHARS

    async def handler(kwargs):
        if len(fake.calls) == 1:
            return response(None, [tool_call("retrieve_coaching_knowledge", call_id="k1"),
                                   tool_call("analyze_vision_and_fog", call_id="v1")])
        return submit()

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    _run_agent(match, ctx, knowledge)
    answers = {m["tool_call_id"]: m["content"] for m in _tool_messages(fake.calls[1])}
    assert answers["k1"] == knowledge and "TAIL-MARKER" in answers["k1"]
    assert agent._TOOL_HISTORY_BUDGET >= 2 * evaluator.MAX_KNOWLEDGE_CHARS


def test_results_the_model_has_not_seen_are_never_trimmed(monkeypatch):
    monkeypatch.setattr(agent, "_TOOL_HISTORY_BUDGET", 500)
    first_turn = ["get_player_state", "get_nearby_heroes", "get_recent_events", "analyze_vision_and_fog",
                  "trace_death_cause", "get_match_situation"]

    async def handler(kwargs):
        n = len(fake.calls)
        if n == 1:
            return response(None, [tool_call(name, call_id=f"a{i}") for i, name in enumerate(first_turn)])
        if n == 2:
            return response(None, [tool_call("get_pro_benchmarks", call_id="b1")])
        if n == 3:
            return response(None, [tool_call("get_nearby_heroes", call_id="again")])
        return submit()

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    _run_agent(match, ctx)
    second = _tool_messages(fake.calls[1])
    assert len(second) == 6 and not any(m["content"].endswith("reviewed]") for m in second)
    assert sum(len(m["content"]) for m in second) > 500  # over the budget, still complete
    full_nearby = second[1]["content"]
    third = {m["tool_call_id"]: m["content"] for m in _tool_messages(fake.calls[2])}
    assert third["a3"].endswith("[trimmed; already reviewed]")  # an older round may be shortened
    # The model asks for a shortened result again and gets it back in full.
    fourth = {m["tool_call_id"]: m["content"] for m in _tool_messages(fake.calls[3])}
    assert fourth["again"] == full_nearby


def test_benchmark_tool_knows_the_game_mode():
    """A Turbo match got the normal-mode CS / GPM numbers from get_pro_benchmarks."""
    match, ctx = death_context()
    normal = agent._t_benchmark(agent._ToolCtx(match=match, context=ctx, knowledge="", lang="en"), {})
    assert "CS at 25 min" in normal
    match.game_mode = "turbo"
    turbo = agent._t_benchmark(agent._ToolCtx(match=match, context=ctx, knowledge="", lang="en"), {})
    assert "CS at" not in turbo and "GPM" not in turbo and "Turbo" in turbo


@pytest.mark.parametrize("lang, line, rule", [
    ("en", "Positions in this match (derived from lanes and net worth): your team: Puck (Position 2); "
           "enemy team: Lina (Position 2), Lion (Position 5)", "do not call a Position 4 or 5 just 'a support'"),
    ("zh", "各英雄本场位置（按分路和经济推算）: 己方 Puck 2号位；敌方 Lina 2号位、Lion 5号位", "不要把4号位、5号位只称为“辅助”"),
])
def test_agent_brief_and_situation_tool_list_every_heros_position(lang, line, rule):
    """The agent names roles from the positions of this match, like the single prompt."""
    match, ctx = death_context(lang)
    assert "Position 2" not in agent._user_brief(ctx, lang) and "2号位" not in agent._user_brief(ctx, lang)
    ctx.match_context["positions"] = {"radiant": [["Puck", 2]], "dire": [["Lina", 2], ["Lion", 5]]}
    assert line in agent._user_brief(ctx, lang)
    assert line in agent._t_match_situation(agent._ToolCtx(match=match, context=ctx, knowledge="", lang=lang), {})
    assert rule in agent._system_prompt(lang, True) and rule in agent._system_prompt(lang, False)


def test_vision_tool_fallback_judges_a_death_where_the_player_died():
    match, ctx = death_context()
    ctx.vision_analysis = None                      # context built without it: the tool computes it
    ctx.player_state.x, ctx.player_state.y = 0.0, 0.0
    text = agent._t_vision(agent._ToolCtx(match=match, context=ctx, knowledge="", lang="en"), {})
    assert "Player position:" in text and "unknown area" not in text


def test_trimmed_reference_block_stays_closed():
    block = "<reference>\n" + "x" * 2000 + "\n</reference>"
    messages = [USER, _assistant("k"), _tool("k", block), _assistant("b"), _tool("b", "y" * 30000), USER]
    out = agent._trimmed(messages)
    assert out[2]["content"].endswith("[trimmed; already reviewed]")
    assert out[2]["content"].count("<reference>") == out[2]["content"].count("</reference>") == 1


def test_garbled_tool_arguments_do_not_break_the_conversation(monkeypatch):
    async def handler(kwargs):
        if len(fake.calls) == 1:
            call = tool_call("get_hero_matchup", call_id="m1")
            call.function.arguments = '{"opponent_hero": "Li'
            return response(None, [call, tool_call("no_such_tool", call_id="x1")])
        return submit()

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = _run_agent(match, ctx)
    assert ev.used_llm and len(fake.calls) == 2  # the strict fake accepted request 2
    echoed = fake.calls[1]["messages"][2]["tool_calls"]
    assert [json.loads(c["function"]["arguments"]) for c in echoed] == [{}, {}]
    answers = {m["tool_call_id"]: m["content"] for m in _tool_messages(fake.calls[1])}
    assert "not available" in answers["x1"]


def test_empty_reply_does_not_put_an_empty_assistant_turn_in_history(monkeypatch):
    async def handler(kwargs):
        if len(fake.calls) == 1:
            return response("")
        return submit()

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = _run_agent(match, ctx)
    assert ev.used_llm
    roles = [m["role"] for m in fake.calls[1]["messages"]]
    assert roles == ["system", "user"]  # the nudge is merged into the brief, no empty assistant turn
    assert "submit_evaluation" in fake.calls[1]["messages"][1]["content"]


@pytest.mark.parametrize("lang, never, estimate", [
    ("en", "never mention a tool", "estimates computed from the replay"),
    ("zh", "不要提到任何工具", "赛后根据录像数据推算"),
])
def test_prompts_keep_pipeline_vocabulary_out_of_the_verdict(monkeypatch, lang, never, estimate):
    """Live r1 verdicts quoted tool names and '系统已明确告警' as if the player had seen an alert."""
    async def handler(kwargs):
        if len(fake.calls) == 1:
            return response(None, [tool_call("analyze_vision_and_fog", call_id="v1"),
                                   tool_call("trace_death_cause", call_id="c1")])
        if lang == "zh":
            return submit("C", ANALYSIS_ZH, mistake="无", better_play="继续保持", lesson="先看小地图。")
        return submit()

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context(lang)
    _run_agent(match, ctx, lang=lang)
    system = fake.calls[0]["messages"][0]["content"]
    assert never in system
    for message in _tool_messages(fake.calls[1]):
        assert estimate in message["content"].splitlines()[0]
    brief = fake.calls[0]["messages"][1]["content"]
    # Type and severity are words of the report language, not internal codes.
    assert ("- Type: Death" in brief) if lang == "en" else ("类型：死亡" in brief and "严重程度：严重" in brief)
    assert "player_slot" not in brief


@pytest.mark.parametrize("lang, leak, clean", [
    ("zh", "analyze_vision_and_fog 显示所有敌人都可见，" + ANALYSIS_ZH, ANALYSIS_ZH),
    ("en", ANALYSIS_EN + " The get_match_situation tool shows a 3k gold deficit.", ANALYSIS_EN),
])
def test_verdict_naming_a_tool_is_not_shown_to_the_player(monkeypatch, lang, leak, clean):
    """Live r1: 'analyze_vision_and_fog 显示所有敌人都可见' was the analysis the player read. The
    prompt forbids it; a verdict that does it anyway is asked for once more, then rules take over."""
    other = dict(mistake="无", better_play="继续保持", lesson="先看小地图。") if lang == "zh" else {}
    replies = [submit("C", leak, call_id="s1", **other), submit("C", clean, call_id="s2", **other)]

    async def handler(kwargs):
        return replies[min(len(fake.calls), len(replies)) - 1]

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context(lang)
    ev = _run_agent(match, ctx, lang=lang)
    assert ev.analysis == clean and len(fake.calls) == 2
    note = _tool_messages(fake.calls[1])[-1]["content"]
    assert ("names a tool" in note) if lang == "en" else ("工具名称" in note)

    replies[1] = replies[0]                                  # the model does it again
    fake = install_fake(monkeypatch, handler)
    ev = asyncio.run(coach._safe_evaluate(ctx, "", lang=lang, match=match))
    assert ev.degraded and not ev.used_llm
    assert "analyze_vision_and_fog" not in ev.analysis and "get_match_situation" not in ev.analysis


@pytest.mark.parametrize("agent_on", [True, False])
def test_verdict_quoting_an_analysis_label_is_asked_for_once_more(monkeypatch, agent_on):
    """Live: "从死亡因果链看，唯一明确记录的步骤就是…" - a heading of the prompt, read by the player.
    Unlike a tool name this is a blemish: one re-ask, and the second answer stands either way."""
    monkeypatch.setenv("AI_COACH_AGENT", "1" if agent_on else "0")
    leak = "The death causal chain shows only the kill itself. " + ANALYSIS_EN

    def reply(analysis):
        if agent_on:
            return submit("C", analysis, call_id=f"s{len(fake.calls)}")
        return response(f"Score: C\nAnalysis: {analysis}\nMistake: None\nBetter Play: Keep it up\nLesson: Look first.")

    answers = [leak, ANALYSIS_EN]

    async def handler(kwargs):
        return reply(answers[min(len(fake.calls), len(answers)) - 1])

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="en", match=match))
    assert ev.used_llm and not ev.degraded and ev.analysis == ANALYSIS_EN and len(fake.calls) == 2
    asked = fake.calls[1]["messages"][-1]["content"]
    assert "quotes a label of the analysis data" in asked

    answers[1] = leak                                        # the model does it again: kept, not degraded
    fake = install_fake(monkeypatch, handler)
    ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="en", match=match))
    assert ev.used_llm and not ev.degraded and len(fake.calls) == 2
    assert evaluator.quotes_labels("局势的威胁等级是危险") and not evaluator.quotes_labels("同一个根因：站位过前", None, 3)

    # A re-ask that brings nothing usable (cut off, garbage, an API error) does not cost the verdict.
    async def cut_off(kwargs):
        return reply(leak) if len(fake.calls) == 1 else response("Score: C\nAnalysis: The", finish_reason="length")

    async def garbage(kwargs):
        return reply(leak) if len(fake.calls) == 1 else response("I would rather look at the vision first.")

    async def broken(kwargs):
        if len(fake.calls) == 1:
            return reply(leak)
        raise RuntimeError("connection reset")

    for second in (cut_off, garbage, broken):
        fake = install_fake(monkeypatch, second)
        ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="en", match=match))
        assert ev.used_llm and not ev.degraded and ev.analysis == leak, second.__name__
        if agent_on:
            assert ev.agent_trace[-1]["tool"] == "submit_evaluation"


# ---------------------------------------------------------------------------
# Logging for the live validation
# ---------------------------------------------------------------------------

def test_debug_log_has_one_line_per_call_and_the_fallback_reason(monkeypatch, caplog):
    monkeypatch.setenv("AI_COACH_MAX_TOKENS", "3000")

    async def handler(kwargs):
        return response(TEXT_VERDICT_EN, reasoning_tokens=20000)  # never fits: both attempts are cut off

    install_fake(monkeypatch, handler, thinking=True, thinking_switch=False)
    match, ctx = death_context()
    monkeypatch.setenv("AI_COACH_AGENT", "0")
    with caplog.at_level(logging.DEBUG, logger="backend.app.ai"):
        ev = asyncio.run(coach._safe_evaluate(ctx, "SECRET-KNOWLEDGE", lang="en", match=match))
    assert ev.degraded
    calls = [r.getMessage() for r in caplog.records if r.getMessage().startswith("LLM call")]
    assert len(calls) == 3  # first attempt, retry with the switch (rejected), retry without it
    assert "purpose=evaluation " in calls[0] and "max_tokens=3000" in calls[0] and "finish=length" in calls[0]
    assert "reasoning_tokens=3000" in calls[0] and "prompt_tokens=" in calls[0] and "completion_tokens=3000" in calls[0]
    assert "purpose=evaluation-retry" in calls[1] and "thinking=off" in calls[1] and "error=" in calls[1]
    assert "max_tokens=6000" in calls[2] and "thinking=default" in calls[2]
    assert f"model={evaluator.DEFAULT_MODEL}" in calls[0]
    reasons = [r.getMessage() for r in caplog.records if "cut off" in r.getMessage() or "truncated" in r.getMessage()]
    assert reasons, "the reason of the fallback is logged"
    everything = "\n".join(r.getMessage() for r in caplog.records)
    assert "SECRET-KNOWLEDGE" not in everything and "test-key" not in everything
    assert "Decision Point" not in everything  # no prompt text at any level


def test_agent_logs_why_it_gave_up(monkeypatch, caplog):
    async def handler(kwargs):
        return submit("B", "Placeholder", call_id=f"s{len(fake.calls)}")

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    with caplog.at_level(logging.DEBUG, logger="backend.app.ai"):
        ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="en", match=match))
    assert ev.degraded
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "unusable verdict (placeholder)" in text and "purpose=agent " in text
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("placeholder" in w for w in warnings)  # the coach's warning carries the reason
