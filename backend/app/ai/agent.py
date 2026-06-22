"""Tool-calling agent for AI coaching.

This turns the coaching *evaluation* step from a fixed, single-prompt
"workflow" into a genuine **agent**: instead of the developer pre-stuffing all
context into one prompt, the LLM is given a small brief plus a toolbox and it
autonomously decides *which* tools to call (vision/fog analysis, death causal
chain, RAG knowledge retrieval, pro benchmarks, matchup tips, ...), in what
order, and *when* it has gathered enough evidence to judge. It ends by calling
the terminal ``submit_evaluation`` tool.

Every tool is a thin wrapper over an existing analysis module, so the agent
reuses the full data pipeline rather than duplicating it. The output is the
same :class:`Evaluation` shape the rest of the system already consumes, so no
downstream/frontend changes are required.

Robustness: ``run_decision_agent`` raises on hard failure (e.g. the endpoint
does not support tool calling); the caller (:mod:`coach`) then falls back to the
single-prompt :func:`evaluator.evaluate_decision` and finally to the rule-based
evaluator. The whole agent path can be disabled with ``AI_COACH_AGENT=0``.

Environment variables
---------------------
- ``AI_COACH_AGENT``        : "0"/"false"/"off" disables the agent (default on).
- ``AI_COACH_AGENT_MODEL``  : model override for the agent loop (default: the
  same model the single-prompt evaluator uses). Set to a tool-calling capable
  model such as ``deepseek-chat`` if the default model rejects ``tools``.
- ``AI_COACH_AGENT_MAX_STEPS`` : max tool-calling rounds before forcing a
  verdict (default 6, minimum 2).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from ..models import MatchData
from .models import DecisionContext, Evaluation
from . import evaluator as _ev

logger = logging.getLogger(__name__)

_VALID_SCORES = ("S", "A", "B", "C", "D")
_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def is_agent_enabled() -> bool:
    """Whether the tool-calling agent path is enabled (default: on)."""
    return os.environ.get("AI_COACH_AGENT", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _agent_model() -> str:
    """Model id for the agent loop (falls back to the evaluator's model)."""
    return os.environ.get("AI_COACH_AGENT_MODEL", "").strip() or _ev.MODEL


def _max_steps() -> int:
    """Max tool-calling rounds before the agent is forced to conclude."""
    try:
        return max(2, int(os.environ.get("AI_COACH_AGENT_MAX_STEPS", "6")))
    except ValueError:
        return 6


# ---------------------------------------------------------------------------
# Tool execution context
# ---------------------------------------------------------------------------

@dataclass
class _ToolCtx:
    """State shared by all tool handlers for a single decision point."""
    match: MatchData
    context: DecisionContext
    knowledge: str
    lang: str
    _player: Any = None              # cached PlayerData
    _benchmark: Optional[str] = None  # cached pro-benchmark text


def _player_of(tc: _ToolCtx):
    """Locate the PlayerData for the decision's player_slot (cached)."""
    if tc._player is None:
        slot = tc.context.decision_point.player_slot
        for p in tc.match.players:
            if p.player_slot == slot:
                tc._player = p
                break
    return tc._player


def _no_data(lang: str) -> str:
    return "No data available." if lang == "en" else "暂无相关数据。"


# ---------------------------------------------------------------------------
# Tool handlers — thin wrappers over existing analysis modules
# ---------------------------------------------------------------------------

def _t_player_state(tc: _ToolCtx, args: dict) -> str:
    t = _ev._PROMPT_TEMPLATES.get(tc.lang, _ev._PROMPT_TEMPLATES["zh"])
    return _ev._format_hero_state(tc.context.player_state, t["label_player"], lang=tc.lang)


def _t_nearby_heroes(tc: _ToolCtx, args: dict) -> str:
    t = _ev._PROMPT_TEMPLATES.get(tc.lang, _ev._PROMPT_TEMPLATES["zh"])
    lines: list[str] = []

    if tc.context.ally_states:
        lines.append(f"## {t['section_allies']}")
        for a in tc.context.ally_states:
            lines.append("  " + _ev._format_hero_state(a, t["label_ally"], lang=tc.lang))
    else:
        lines.append(t["no_allies"])

    if tc.context.enemy_states:
        lines.append(f"## {t['section_enemies']}")
        for e in tc.context.enemy_states:
            lines.append("  " + _ev._format_hero_state(e, t["label_enemy"], lang=tc.lang))
    else:
        lines.append(t["no_enemies"])

    distant = (tc.context.map_state or {}).get("distant_enemies", [])
    if distant:
        lines.append(f"## {t['section_distant_enemies']}")
        for de in distant:
            alive = t["label_alive"] if de.get("is_alive", True) else t["label_dead"]
            lines.append(f"  - {de.get('hero_name', '?')} (Lv{de.get('level', '?')}, {alive})")

    return "\n".join(lines)


def _t_recent_events(tc: _ToolCtx, args: dict) -> str:
    return _ev._format_event_timeline(tc.context.event_timeline, lang=tc.lang)


def _t_match_situation(tc: _ToolCtx, args: dict) -> str:
    t = _ev._PROMPT_TEMPLATES.get(tc.lang, _ev._PROMPT_TEMPLATES["zh"])
    mc = tc.context.match_context or {}
    ms = tc.context.map_state or {}
    parts: list[str] = []

    if mc.get("gold_adv") is not None:
        side = t["gold_adv_radiant"] if mc["gold_adv"] > 0 else t["gold_adv_dire"]
        parts.append(f"{t['label_gold_diff']}: {side} {abs(mc['gold_adv'])} {t['label_gold_unit']}")
    if mc.get("xp_adv") is not None:
        side = t["gold_adv_radiant"] if mc["xp_adv"] > 0 else t["gold_adv_dire"]
        parts.append(f"{t['label_xp_diff']}: {side} {abs(mc['xp_adv'])}")
    if mc.get("radiant_score") is not None:
        parts.append(
            f"{t['label_score']}: {t['label_radiant']} {mc.get('radiant_score', 0)} - "
            f"{mc.get('dire_score', 0)} {t['label_dire']}"
        )
    if ms.get("radiant_towers"):
        parts.append(f"{t['radiant_towers']}: {ms['radiant_towers']}")
    if ms.get("dire_towers"):
        parts.append(f"{t['dire_towers']}: {ms['dire_towers']}")

    return "\n".join(parts) if parts else t["no_match_info"]


def _t_vision(tc: _ToolCtx, args: dict) -> str:
    va = tc.context.vision_analysis
    if va and va.formatted_text:
        return va.formatted_text
    # Compute on demand if the context did not already carry it.
    try:
        from .vision import analyze_vision
        ps = tc.context.player_state
        dp = tc.context.decision_point
        va2 = analyze_vision(tc.match, dp.player_slot, dp.time, (ps.x, ps.y), lang=tc.lang)
        return va2.formatted_text or _no_data(tc.lang)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("vision tool failed: %s", exc)
        return _no_data(tc.lang)


def _t_causal(tc: _ToolCtx, args: dict) -> str:
    cc = tc.context.causal_chain
    if cc and cc.formatted_text:
        return cc.formatted_text
    dp = tc.context.decision_point
    if dp.point_type != "death" or not dp.related_death:
        return (
            "This decision point is not a death, so there is no causal chain."
            if tc.lang == "en"
            else "该决策点不是死亡事件，没有死亡因果链。"
        )
    try:
        from .causal_chain import build_causal_chain
        chain = build_causal_chain(tc.match, dp.related_death, dp.player_slot, lang=tc.lang)
        return chain.formatted_text or _no_data(tc.lang)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("causal tool failed: %s", exc)
        return _no_data(tc.lang)


def _t_knowledge(tc: _ToolCtx, args: dict) -> str:
    if tc.knowledge and tc.knowledge.strip():
        return tc.knowledge.strip()
    try:
        from .knowledge_base import get_knowledge_for_context
        return get_knowledge_for_context(tc.context, lang=tc.lang) or _no_data(tc.lang)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("knowledge tool failed: %s", exc)
        return _no_data(tc.lang)


def _t_matchup(tc: _ToolCtx, args: dict) -> str:
    opponent = str((args or {}).get("opponent_hero", "")).strip()
    if not opponent:
        return (
            "Please provide the opponent_hero argument."
            if tc.lang == "en"
            else "请提供 opponent_hero 参数（敌方英雄名）。"
        )
    try:
        from .knowledge_base import get_matchup_info
        hero = tc.context.player_state.hero_name
        return get_matchup_info(hero, opponent, lang=tc.lang) or _no_data(tc.lang)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("matchup tool failed: %s", exc)
        return _no_data(tc.lang)


def _t_benchmark(tc: _ToolCtx, args: dict) -> str:
    if tc._benchmark is not None:
        return tc._benchmark
    try:
        from .pro_patterns import get_benchmark_context, resolve_role
        hero = tc.context.player_state.hero_name
        role = resolve_role(_player_of(tc), hero)
        txt = get_benchmark_context(hero, role, tc.context.decision_point.time, lang=tc.lang)
        tc._benchmark = txt or _no_data(tc.lang)
        return tc._benchmark
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("benchmark tool failed: %s", exc)
        return _no_data(tc.lang)


_HANDLERS = {
    "get_player_state": _t_player_state,
    "get_nearby_heroes": _t_nearby_heroes,
    "get_recent_events": _t_recent_events,
    "get_match_situation": _t_match_situation,
    "analyze_vision_and_fog": _t_vision,
    "trace_death_cause": _t_causal,
    "retrieve_coaching_knowledge": _t_knowledge,
    "get_hero_matchup": _t_matchup,
    "get_pro_benchmarks": _t_benchmark,
}


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

def _no_arg_params() -> dict:
    return {"type": "object", "properties": {}, "additionalProperties": False}


_TOOL_DESCRIPTIONS = {
    "zh": {
        "get_player_state": "获取该玩家在决策时刻的状态：血量/魔法/等级/装备/技能/补刀/经济/位置。",
        "get_nearby_heroes": "获取决策时刻附近的队友和敌人状态，以及在地图其他位置的敌方英雄。",
        "get_recent_events": "获取决策点前约60秒内的事件时间线（伤害/技能/击杀等）。",
        "get_match_situation": "获取比赛整体局势：经济差、经验差、比分、双方防御塔状态。",
        "analyze_vision_and_fog": "分析该时刻的视野与迷雾：消失的敌方英雄、最长消失时间、眼位覆盖、危险等级。死亡分析的关键工具。",
        "trace_death_cause": "回溯这次死亡的因果链，定位最早可改变的根本原因（仅对死亡决策点有效）。",
        "retrieve_coaching_knowledge": "从教练知识库检索与当前英雄/对局/局面相关的知识（RAG 检索）。",
        "get_hero_matchup": "获取本方英雄针对指定敌方英雄的对线/对位建议。",
        "get_pro_benchmarks": "获取该英雄/位置在当前时间点的职业基准（补刀/GPM/伤害/死亡数等）。",
        "submit_evaluation": "提交对该决策点的最终评价。完成调查后必须调用本工具来结束。",
        "p_opponent": "敌方英雄名称（与上下文中显示的英雄名一致）。",
        "p_score": "评分：S=完美, A=很好, B=合理但有改进空间, C=较差有明显错误, D=严重失误。",
        "p_analysis": "详细分析：这个时间点发生了什么，玩家为什么这么做。",
        "p_mistake": "如果有错误，具体指出错在哪；操作正确则填\"无\"。",
        "p_better": "如果有错误，建议更好的做法；操作正确则填\"继续保持\"。",
        "p_lesson": "可以从该决策中提炼的一般性教训。",
    },
    "en": {
        "get_player_state": "Get the player's state at the decision moment: HP/mana/level/items/skills/CS/net worth/position.",
        "get_nearby_heroes": "Get nearby allied and enemy hero states at the decision moment, plus enemies elsewhere on the map.",
        "get_recent_events": "Get the event timeline (~60s before the decision): damage, ability casts, kills, etc.",
        "get_match_situation": "Get the overall match state: gold/xp advantage, score, and both teams' tower status.",
        "analyze_vision_and_fog": "Analyze vision/fog at this moment: missing enemy heroes, longest missing time, ward coverage, danger level. Key tool for deaths.",
        "trace_death_cause": "Trace the causal chain of this death to the earliest changeable root cause (death decision points only).",
        "retrieve_coaching_knowledge": "Retrieve coaching knowledge relevant to the current hero/matchup/situation (RAG retrieval).",
        "get_hero_matchup": "Get laning/matchup advice for your hero against a specific enemy hero.",
        "get_pro_benchmarks": "Get pro-level benchmarks for this hero/role at the current game time (CS/GPM/damage/deaths).",
        "submit_evaluation": "Submit the final evaluation for this decision point. You MUST call this to finish once your investigation is done.",
        "p_opponent": "Enemy hero name (matching the name shown in the context).",
        "p_score": "Grade: S=perfect, A=great, B=reasonable but improvable, C=poor with a clear mistake, D=serious mistake.",
        "p_analysis": "Detailed analysis: what happened at this moment and why the player made this decision.",
        "p_mistake": "If there was a mistake, state exactly what went wrong; write \"None\" if the play was correct.",
        "p_better": "If there was a mistake, suggest a better play; write \"Keep it up\" if the play was correct.",
        "p_lesson": "A general lesson to take away from this decision.",
    },
}


def _tools_schema(lang: str, is_death: bool) -> list[dict]:
    d = _TOOL_DESCRIPTIONS.get(lang, _TOOL_DESCRIPTIONS["zh"])

    def fn(name: str, params: dict) -> dict:
        return {
            "type": "function",
            "function": {"name": name, "description": d[name], "parameters": params},
        }

    tools = [
        fn("get_player_state", _no_arg_params()),
        fn("get_nearby_heroes", _no_arg_params()),
        fn("get_recent_events", _no_arg_params()),
        fn("get_match_situation", _no_arg_params()),
        fn("analyze_vision_and_fog", _no_arg_params()),
    ]
    if is_death:
        tools.append(fn("trace_death_cause", _no_arg_params()))
    tools.extend([
        fn("retrieve_coaching_knowledge", _no_arg_params()),
        fn("get_hero_matchup", {
            "type": "object",
            "properties": {"opponent_hero": {"type": "string", "description": d["p_opponent"]}},
            "required": ["opponent_hero"],
            "additionalProperties": False,
        }),
        fn("get_pro_benchmarks", _no_arg_params()),
        fn("submit_evaluation", {
            "type": "object",
            "properties": {
                "score": {"type": "string", "enum": list(_VALID_SCORES), "description": d["p_score"]},
                "analysis": {"type": "string", "description": d["p_analysis"]},
                "mistake": {"type": "string", "description": d["p_mistake"]},
                "better_play": {"type": "string", "description": d["p_better"]},
                "lesson": {"type": "string", "description": d["p_lesson"]},
            },
            "required": ["score", "analysis"],
            "additionalProperties": False,
        }),
    ])
    return tools


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _system_prompt(lang: str, is_death: bool) -> str:
    if lang == "en":
        base = (
            "You are a high-level Dota 2 coach (~7000 MMR) acting as an investigation "
            "AGENT. You are reviewing ONE decision point for one player.\n\n"
            "You have tools to fetch information on demand: player state, nearby heroes, "
            "recent events, match situation, vision/fog analysis, death causal chain, "
            "coaching knowledge (RAG), hero matchup tips, and pro benchmarks.\n\n"
            "How to work:\n"
            "1. First decide what actually matters for THIS decision point.\n"
            "2. Call only the tools you genuinely need, gathering evidence step by step. "
            "Do not blindly call every tool.\n"
            "3. Pay special attention to vision — many deaths stem from ignoring map vision.\n"
            "4. When you have enough evidence, call submit_evaluation with your verdict. "
            "You MUST finish by calling submit_evaluation, and respond entirely in English.\n\n"
            "Grading: S=perfect, A=great, B=reasonable but improvable, C=poor with a clear "
            "mistake, D=serious mistake causing major loss."
        )
        if is_death:
            base += (
                "\n\nThis decision point is a DEATH. Before concluding, you should normally "
                "inspect analyze_vision_and_fog and trace_death_cause, and target the "
                "earliest changeable root cause rather than the death itself."
            )
        return base

    base = (
        "你是一名 Dota 2 高分教练（约 7000 MMR），现在以\"调查代理（agent）\"的方式复盘"
        "一名玩家在某一个决策点的表现。\n\n"
        "你可以按需调用一组工具获取信息：玩家状态、附近英雄、最近事件、比赛局势、"
        "视野/迷雾分析、死亡因果链、教练知识库（RAG）、对位建议、职业基准。\n\n"
        "工作方式：\n"
        "1. 先判断这个决策点真正的关键点是什么。\n"
        "2. 只调用你确实需要的工具，逐步收集证据，不要无脑把所有工具都调一遍。\n"
        "3. 特别关注视野信息——很多死亡都源于忽视地图视野。\n"
        "4. 证据足够后，调用 submit_evaluation 给出最终评价。必须以调用该工具结束。\n\n"
        "评分标准：S=完美，A=很好，B=合理但有改进空间，C=较差且有明显错误，D=严重失误造成重大损失。"
    )
    if is_death:
        base += (
            "\n\n这是一次死亡决策点。下结论前通常应先查看 analyze_vision_and_fog 和 "
            "trace_death_cause，并针对\"最早可以改变的根本原因\"给建议，而不是只分析死亡本身。"
        )
    return base


def _user_brief(context: DecisionContext, lang: str) -> str:
    t = _ev._PROMPT_TEMPLATES.get(lang, _ev._PROMPT_TEMPLATES["zh"])
    dp = context.decision_point
    mm, ss = dp.time // 60, dp.time % 60
    phase_map = {
        "pregame": t["phase_pregame"],
        "laning": t["phase_laning"],
        "mid_game": t["phase_mid"],
        "late_game": t["phase_late"],
    }
    phase_str = phase_map.get(context.game_phase, context.game_phase or t["phase_unknown"])

    if lang == "en":
        return (
            f"Decision point to investigate:\n"
            f"- Time: {mm:02d}:{ss:02d} ({dp.time}s)\n"
            f"- Type: {dp.point_type}\n"
            f"- Severity: {dp.severity}\n"
            f"- Phase: {phase_str}\n"
            f"- Hero: {dp.hero_name} (player_slot {dp.player_slot})\n"
            f"- Summary: {dp.brief}\n\n"
            f"Investigate with the tools, then submit your evaluation."
        )
    return (
        f"待调查的决策点：\n"
        f"- 时间：{mm:02d}:{ss:02d}（{dp.time}s）\n"
        f"- 类型：{dp.point_type}\n"
        f"- 严重程度：{dp.severity}\n"
        f"- 阶段：{phase_str}\n"
        f"- 英雄：{dp.hero_name}（player_slot {dp.player_slot}）\n"
        f"- 简述：{dp.brief}\n\n"
        f"请用工具调查后，调用 submit_evaluation 给出评价。"
    )


def _nudge(lang: str) -> str:
    return (
        "Based on what you've gathered, call submit_evaluation now with your final verdict."
        if lang == "en"
        else "请基于已收集到的信息，现在调用 submit_evaluation 给出最终评价。"
    )


# ---------------------------------------------------------------------------
# Loop helpers
# ---------------------------------------------------------------------------

def _parse_args(raw: Optional[str]) -> dict:
    if not raw:
        return {}
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _assistant_dict(msg: Any, tool_calls: Any) -> dict:
    """Serialize an assistant turn for the messages history."""
    if tool_calls:
        return {
            "role": "assistant",
            "content": msg.content or None,
            "tool_calls": [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments or "{}",
                    },
                }
                for c in tool_calls
            ],
        }
    return {"role": "assistant", "content": msg.content or ""}


def _preview(value: Any, limit: int = 600) -> str:
    s = value if isinstance(value, str) else str(value)
    return s if len(s) <= limit else s[:limit] + "…"


def _execute(name: str, tc: _ToolCtx, args: dict) -> str:
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool: {name}"
    try:
        out = handler(tc, args)
        return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("agent tool %s failed: %s", name, exc)
        return f"(tool {name} error: {exc})"


def _evaluation_from_submit(args: dict, context: DecisionContext, lang: str) -> Evaluation:
    score = str(args.get("score", "C")).strip()[:1].upper()
    if score not in _VALID_SCORES:
        score = "C"
    analysis = str(args.get("analysis", "")).strip()
    if not analysis:
        analysis = context.decision_point.brief or ("Evaluation complete" if lang == "en" else "评估完成")
    return Evaluation(
        decision_point=context.decision_point,
        score=score,
        analysis=analysis,
        mistake=str(args.get("mistake", "")).strip(),
        better_play=str(args.get("better_play", "")).strip(),
        lesson=str(args.get("lesson", "")).strip(),
        _parsed_ok=True,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_decision_agent(
    match: MatchData,
    context: DecisionContext,
    knowledge: str,
    lang: str = "zh",
) -> Evaluation:
    """Evaluate a decision point via an autonomous tool-calling loop.

    Raises on hard failure (no API, endpoint rejects tools, no verdict reached)
    so the caller can fall back to the single-prompt / rule-based evaluators.
    """
    if not _ev.is_api_available():
        raise RuntimeError("DeepSeek API not available")

    from openai import AsyncOpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=_DEEPSEEK_BASE_URL,
        timeout=120.0,
        max_retries=1,
    )

    tc = _ToolCtx(match=match, context=context, knowledge=knowledge or "", lang=lang)
    dp = context.decision_point
    is_death = dp.point_type == "death"
    tools = _tools_schema(lang, is_death)

    messages: list[dict] = [
        {"role": "system", "content": _system_prompt(lang, is_death)},
        {"role": "user", "content": _user_brief(context, lang)},
    ]

    model = _agent_model()
    trace: list[dict] = []
    last_text = ""

    for step in range(_max_steps()):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=0.3,
            max_tokens=3000,
        )
        msg = response.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if tool_calls:
            messages.append(_assistant_dict(msg, tool_calls))
            for call in tool_calls:
                name = call.function.name
                args = _parse_args(call.function.arguments)

                if name == "submit_evaluation":
                    ev = _evaluation_from_submit(args, context, lang)
                    trace.append({
                        "step": step + 1,
                        "tool": "submit_evaluation",
                        "args": {"score": ev.score},
                        "result": "",
                    })
                    ev.agent_trace = trace
                    logger.info(
                        "agent: decision t=%ss resolved in %d step(s), %d tool call(s)",
                        dp.time, step + 1, len(trace),
                    )
                    return ev

                result = _execute(name, tc, args)
                trace.append({
                    "step": step + 1,
                    "tool": name,
                    "args": args,
                    "result": _preview(result),
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })
            continue

        # No tool calls: the model answered free-form. Try to parse it as a
        # final evaluation; otherwise nudge it to use submit_evaluation.
        last_text = msg.content or ""
        if last_text:
            ev = _ev._parse_evaluation_response(last_text, dp, lang=lang)
            if getattr(ev, "_parsed_ok", False):
                ev.agent_trace = trace
                return ev
        messages.append(_assistant_dict(msg, None))
        messages.append({"role": "user", "content": _nudge(lang)})

    # Ran out of steps without an explicit submit — salvage any free-form text.
    if last_text:
        ev = _ev._parse_evaluation_response(last_text, dp, lang=lang)
        ev.agent_trace = trace
        return ev
    raise RuntimeError("agent did not reach a verdict within the step budget")


__all__ = ["run_decision_agent", "is_agent_enabled"]
