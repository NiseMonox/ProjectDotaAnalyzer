"""LLM-based decision evaluation for AI coaching system.

Takes a DecisionContext (with knowledge injected) and produces an Evaluation
using the DeepSeek API (OpenAI-compatible), with a rule-based fallback when
the API is unavailable.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading

from .context_builder import count_nearby_allies
from .models import (
    DecisionContext,
    DecisionPoint,
    Evaluation,
    EventEntry,
    HeroState,
    PhaseReview,
    VisionAnalysis,
    is_time_unknown,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DeepSeek API configuration (OpenAI-compatible)
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# DeepSeek v4 is reasoning-by-default; v4-flash is the chat-equivalent (cheaper
# than v4-pro but still reasons before answering). max_tokens must cover both
# the hidden reasoning trace AND the final structured answer.
MODEL = "deepseek-v4-flash"

try:
    from openai import AsyncOpenAI

    _HAS_OPENAI = True
except ImportError:  # pragma: no cover
    _HAS_OPENAI = False

# Shared AsyncOpenAI client, lazily created and reused by evaluate_decision,
# evaluate_phase, and the tool-calling agent (avoids per-call construction).
_shared_client = None
_shared_client_key: str | None = None
_shared_client_lock = threading.Lock()


def get_shared_client():
    """Return the lazily-created shared AsyncOpenAI client.

    The client is rebuilt if DEEPSEEK_API_KEY changes (e.g. set after startup).
    """
    global _shared_client, _shared_client_key
    if not _HAS_OPENAI:
        raise RuntimeError("openai SDK is not installed")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if _shared_client is None or _shared_client_key != api_key:
        with _shared_client_lock:
            if _shared_client is None or _shared_client_key != api_key:
                _shared_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=DEEPSEEK_BASE_URL,
                    timeout=120.0,
                    max_retries=1,
                )
                _shared_client_key = api_key
    return _shared_client

try:
    from .game_data import get_item_data
    _HAS_GAME_DATA = True
except ImportError:
    _HAS_GAME_DATA = False


def is_api_available() -> bool:
    """Check if DeepSeek API key is configured and SDK is installed."""
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    return _HAS_OPENAI and bool(key)


# ---------------------------------------------------------------------------
# Internationalization: prompt templates
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "zh": {
        "system_role": "你是一个Dota 2高水平教练（约7000 MMR），正在复盘一位玩家的操作。",
        "section_decision": "决策点信息",
        "label_time": "时间",
        "label_type": "类型",
        "label_severity": "严重程度",
        "label_brief": "简述",
        "label_phase": "游戏阶段",
        "section_player": "玩家状态",
        "section_allies": "附近队友",
        "section_enemies": "附近敌人",
        "section_timeline": "事件时间线（最近60秒内）",
        "section_vision": "视野与地图状态",
        "section_causal": "死亡因果链",
        "section_match": "比赛整体情况",
        "section_knowledge": "英雄知识/对线知识",
        "section_task": "任务",
        "task_intro": "请评估该玩家在这个时间点的决策。特别注意视野信息——很多死亡和错误都源于对地图视野的忽视。",
        "eval_points": (
            "评估要点:\n"
            "1. 玩家是否在视野缺失时冒险推进？\n"
            "2. 消失的敌方英雄是否构成了威胁？\n"
            "3. 玩家是否利用了可用的视野信息？\n"
            "4. 在当时的视野条件下，玩家的位置是否安全？\n"
            "5. 玩家是否有可用的技能或物品可以改变局面？（如BKB、Blink、关键控制技能）\n"
            "6. 敌方关键技能的冷却时间是否被考虑在内？\n"
            "7. 如果提供了死亡因果链，请重点分析根本原因（标记为[根因]的步骤），而非仅仅分析死亡本身。建议应该针对最早可以改变的决策点。\n"
            "8. 如果是开局阶段（0:00附近）：\n"
            "   a. 该英雄在1级有什么技能可选？（所有英雄1级都可以选择一个基础技能，不要说英雄没有技能）\n"
            "   b. 双方队伍的1级团战能力对比如何？（控制/AOE/肉度）\n"
            "   c. 争夺赏金神符的风险回报比？（~40金 vs 死亡代价100+金）\n"
            "   d. 玩家是否处于人数劣势？"
        ),
        "response_format": (
            "用以下格式回答：\n\n"
            "评分: [S/A/B/C/D]\n"
            "分析: [详细分析这个时间点发生了什么，以及玩家为什么做出了这个决策]\n"
            "错误: [如果有错误的话，指出具体错了什么。如果操作正确，写\"无\"]\n"
            "更好的做法: [如果有错误，建议玩家应该怎么做。如果操作正确，写\"继续保持\"]\n"
            "教训: [从这个决策中可以得到的一般性教训]"
        ),
        "grading_scale": (
            "评分标准:\n"
            "- S: 完美操作，教科书般的决策\n"
            "- A: 很好的操作，只有微小的改进空间\n"
            "- B: 合理的操作，但有明显的改进空间\n"
            "- C: 较差的操作，犯了明显的错误\n"
            "- D: 很差的操作，犯了严重的错误导致极大损失"
        ),
        "phase_pregame": "开局阶段 (0:00附近，赏金神符争夺)",
        "phase_laning": "对线期 (0-10分钟)",
        "phase_mid": "中期 (10-25分钟)",
        "phase_late": "后期 (25分钟+)",
        "phase_unknown": "未知",
        "time_unknown": "未知（数据源未记录该事件的具体时间戳）",
        "label_player": "玩家",
        "label_ally": "队友",
        "label_enemy": "敌人",
        "no_allies": "无附近队友",
        "no_enemies": "无附近敌人",
        "section_distant_enemies": "其他敌方英雄（不在附近，在其他位置）",
        "no_distant_enemies": "",
        "no_events": "无近期事件记录",
        "no_map_state": "无地图状态信息",
        "no_match_info": "无额外比赛信息",
        "no_knowledge": "无额外知识",
        "label_gold": "经济",
        "label_cs": "补刀",
        "label_alive": "存活",
        "label_dead": "死亡",
        "label_status": "状态",
        "label_items": "装备",
        "label_skills": "技能",
        "label_none": "无",
        "label_active": "主动",
        "gold_adv_radiant": "天辉领先",
        "gold_adv_dire": "夜魇领先",
        "label_gold_diff": "经济差",
        "label_gold_unit": "金币",
        "label_xp_diff": "经验差",
        "label_score": "比分",
        "label_radiant": "天辉",
        "label_dire": "夜魇",
        "radiant_towers": "天辉防御塔状态",
        "dire_towers": "夜魇防御塔状态",
        "nearby_wards": "附近眼位",
        "vision_info": "视野信息",
        "enemies_missing": "名敌人消失",
        "longest_missing": "最长消失",
        "longest_missing_unit": "秒",
        "no_ward_coverage": "无视野覆盖",
        "danger_level": "危险等级",
        "vision_prefix": "视野",
    },
    "en": {
        "system_role": "You are a high-level Dota 2 coach (approximately 7000 MMR), reviewing a player's gameplay.",
        "section_decision": "Decision Point",
        "label_time": "Time",
        "label_type": "Type",
        "label_severity": "Severity",
        "label_brief": "Summary",
        "label_phase": "Game Phase",
        "section_player": "Player State",
        "section_allies": "Nearby Allies",
        "section_enemies": "Nearby Enemies",
        "section_timeline": "Event Timeline (last 60 seconds)",
        "section_vision": "Vision & Map State",
        "section_causal": "Death Causal Chain",
        "section_match": "Overall Match State",
        "section_knowledge": "Hero Knowledge / Laning Knowledge",
        "section_task": "Task",
        "task_intro": "Evaluate the player's decision at this point in the game. Pay special attention to vision information — many deaths and mistakes stem from ignoring map vision.",
        "eval_points": (
            "Evaluation criteria:\n"
            "1. Did the player take risks while vision was missing?\n"
            "2. Were missing enemy heroes a threat?\n"
            "3. Did the player make use of available vision information?\n"
            "4. Was the player's position safe given the vision state?\n"
            "5. Did the player have usable abilities or items that could have changed the outcome? (e.g. BKB, Blink, key disables)\n"
            "6. Were enemy key ability cooldowns taken into account?\n"
            "7. If a death causal chain is provided, focus on the root cause (steps marked [Root Cause]) rather than the death itself. Suggestions should target the earliest decision point that could have been changed.\n"
            "8. If this is the pregame phase (around 0:00):\n"
            "   a. What skills can this hero choose at level 1? (All heroes can skill one basic ability at level 1 — never say the hero has no skills)\n"
            "   b. How do the two teams compare in level 1 fight capability? (CC/AoE/tankiness)\n"
            "   c. What is the risk/reward of contesting the bounty rune? (~40 gold vs death cost 100+ gold)\n"
            "   d. Was the player outnumbered?"
        ),
        "response_format": (
            "IMPORTANT: You MUST respond entirely in English, even if some of the context data above is in Chinese.\n\n"
            "Respond in the following format:\n\n"
            "Score: [S/A/B/C/D]\n"
            "Analysis: [Detailed analysis of what happened at this point and why the player made this decision]\n"
            "Mistake: [If there was a mistake, specify what went wrong. If the play was correct, write \"None\"]\n"
            "Better Play: [If there was a mistake, suggest what the player should have done. If the play was correct, write \"Keep it up\"]\n"
            "Lesson: [A general lesson to take away from this decision]"
        ),
        "grading_scale": (
            "Grading scale:\n"
            "- S: Perfect play, textbook decision\n"
            "- A: Great play with only minor room for improvement\n"
            "- B: Reasonable play but with clear room for improvement\n"
            "- C: Poor play with an obvious mistake\n"
            "- D: Very poor play with a serious mistake leading to major loss"
        ),
        "phase_pregame": "Pregame (around 0:00, bounty rune contest)",
        "phase_laning": "Laning Phase (0-10 min)",
        "phase_mid": "Mid Game (10-25 min)",
        "phase_late": "Late Game (25+ min)",
        "phase_unknown": "Unknown",
        "time_unknown": "Unknown (the data source recorded no exact timestamp for this event)",
        "label_player": "Player",
        "label_ally": "Ally",
        "label_enemy": "Enemy",
        "no_allies": "No nearby allies",
        "no_enemies": "No nearby enemies",
        "section_distant_enemies": "Other Enemy Heroes (NOT nearby — at other locations on the map)",
        "no_distant_enemies": "",
        "no_events": "No recent events",
        "no_map_state": "No map state info",
        "no_match_info": "No additional match info",
        "no_knowledge": "No additional knowledge",
        "label_gold": "Net Worth",
        "label_cs": "CS",
        "label_alive": "Alive",
        "label_dead": "Dead",
        "label_status": "Status",
        "label_items": "Items",
        "label_skills": "Skills",
        "label_none": "None",
        "label_active": "Active",
        "gold_adv_radiant": "Radiant leads",
        "gold_adv_dire": "Dire leads",
        "label_gold_diff": "Gold advantage",
        "label_gold_unit": "gold",
        "label_xp_diff": "XP advantage",
        "label_score": "Score",
        "label_radiant": "Radiant",
        "label_dire": "Dire",
        "radiant_towers": "Radiant tower status",
        "dire_towers": "Dire tower status",
        "nearby_wards": "Nearby wards",
        "vision_info": "Vision info",
        "enemies_missing": "enemies missing",
        "longest_missing": "longest missing for",
        "longest_missing_unit": "s",
        "no_ward_coverage": "No ward coverage",
        "danger_level": "Danger level",
        "vision_prefix": "Vision",
    },
}

# Response field labels for parsing, keyed by language
_RESPONSE_LABELS: dict[str, dict[str, str]] = {
    "zh": {
        "score": "评分",
        "analysis": "分析",
        "mistake": "错误",
        "better_play": "更好的做法",
        "lesson": "教训",
        "grading_scale": "评分标准",
    },
    "en": {
        "score": "Score",
        "analysis": "Analysis",
        "mistake": "Mistake",
        "better_play": "Better Play",
        "lesson": "Lesson",
        "grading_scale": "Grading scale",
    },
}


# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------

def _format_hero_state(state: HeroState, label: str = "", lang: str = "zh") -> str:
    """Format a HeroState into a readable string."""
    t = _PROMPT_TEMPLATES.get(lang, _PROMPT_TEMPLATES["zh"])
    hp_pct = (state.hp / state.max_hp * 100) if state.max_hp > 0 else 0
    mana_pct = (state.mana / state.max_mana * 100) if state.max_mana > 0 else 0
    alive_str = t["label_alive"] if state.is_alive else t["label_dead"]
    prefix = f"[{label}] " if label else ""

    # Abilities with levels
    if state.ability_levels:
        ability_parts = []
        for name, lvl in state.ability_levels.items():
            ability_parts.append(f"{name}(Lv{lvl})")
        sep = ", " if lang == "en" else "、"
        abilities_str = sep.join(ability_parts)
    elif state.abilities_available:
        sep = ", " if lang == "en" else "、"
        abilities_str = sep.join(state.abilities_available)
    else:
        if lang == "en":
            abilities_str = "Not yet skilled (can choose any basic skill at level 1)"
        else:
            abilities_str = "尚未选择加点（1级可选任意基础技能）"

    # Items with active indicator
    sep = ", " if lang == "en" else "、"
    active_label = t["label_active"]
    if state.items and _HAS_GAME_DATA:
        item_parts = []
        for item_name in state.items:
            item_info = get_item_data(item_name)
            if item_info and item_info.get("is_active"):
                item_parts.append(f"{item_name}({active_label})")
            else:
                item_parts.append(item_name)
        items_str = sep.join(item_parts) if item_parts else t["label_none"]
    else:
        items_str = sep.join(state.items) if state.items else t["label_none"]

    return (
        f"{prefix}{state.hero_name} (Lv{state.level}, {state.team}) - "
        f"HP {state.hp}/{state.max_hp} ({hp_pct:.0f}%), "
        f"Mana {state.mana}/{state.max_mana} ({mana_pct:.0f}%), "
        f"{t['label_gold']} {state.net_worth}, {t['label_cs']} {state.last_hits}/{state.denies}, "
        f"{t['label_status']}: {alive_str}, {t['label_items']}: {items_str}, {t['label_skills']}: {abilities_str}"
    )


def _format_event_timeline(events: list[EventEntry], lang: str = "zh") -> str:
    """Format event timeline into readable text."""
    t = _PROMPT_TEMPLATES.get(lang, _PROMPT_TEMPLATES["zh"])
    if not events:
        return t["no_events"]
    lines = []
    for evt in events:
        minutes = evt.time // 60
        seconds = evt.time % 60
        lines.append(f"  [{minutes:02d}:{seconds:02d}] {evt.description}")
    return "\n".join(lines)


def _format_map_state(map_state: dict, lang: str = "zh") -> str:
    """Format map state info (vision, towers) into readable text."""
    t = _PROMPT_TEMPLATES.get(lang, _PROMPT_TEMPLATES["zh"])
    if not map_state:
        return t["no_map_state"]
    parts = []
    if "radiant_towers" in map_state:
        parts.append(f"{t['radiant_towers']}: {map_state['radiant_towers']}")
    if "dire_towers" in map_state:
        parts.append(f"{t['dire_towers']}: {map_state['dire_towers']}")
    if "nearby_wards" in map_state:
        parts.append(f"{t['nearby_wards']}: {map_state['nearby_wards']}")
    if "vision_info" in map_state:
        parts.append(f"{t['vision_info']}: {map_state['vision_info']}")
    return "\n".join(parts) if parts else t["no_map_state"]


def _format_vision_summary(ctx: DecisionContext, lang: str = "zh") -> str:
    """Create a one-line vision summary for rule-based evaluations."""
    t = _PROMPT_TEMPLATES.get(lang, _PROMPT_TEMPLATES["zh"])
    va = ctx.vision_analysis
    if not va or not va.state_at_decision:
        return ""
    vs = va.state_at_decision
    parts: list[str] = []
    if vs.missing_enemy_count > 0:
        parts.append(f"{vs.missing_enemy_count}{t['enemies_missing']}")
    if vs.longest_missing_seconds > 15:
        parts.append(f"{t['longest_missing']}{vs.longest_missing_seconds}{t['longest_missing_unit']}")
    if not vs.nearby_ward_coverage:
        parts.append(t["no_ward_coverage"])
    parts.append(f"{t['danger_level']}: {vs.danger_level}")
    sep = ", " if lang == "en" else "，"
    return f"{t['vision_prefix']}: " + sep.join(parts) if parts else ""


def _build_evaluation_prompt(context: DecisionContext, knowledge: str, lang: str = "zh") -> str:
    """Build the full prompt for LLM evaluation."""
    t = _PROMPT_TEMPLATES.get(lang, _PROMPT_TEMPLATES["zh"])
    dp = context.decision_point
    minutes = dp.time // 60
    seconds = dp.time % 60
    # Some good_play points carry no timestamp (time=0); tell the LLM the
    # time is unknown instead of presenting a misleading 00:00.
    if is_time_unknown(dp):
        time_str = t["time_unknown"]
    else:
        time_str = f"{minutes:02d}:{seconds:02d} ({dp.time}s)"

    # Game phase
    phase_map = {
        "pregame": t["phase_pregame"],
        "laning": t["phase_laning"],
        "mid_game": t["phase_mid"],
        "late_game": t["phase_late"],
    }
    phase_str = phase_map.get(context.game_phase, context.game_phase or t["phase_unknown"])

    # Match context
    match_ctx_parts = []
    mc = context.match_context
    if mc.get("gold_adv") is not None:
        side = t["gold_adv_radiant"] if mc["gold_adv"] > 0 else t["gold_adv_dire"]
        match_ctx_parts.append(f"{t['label_gold_diff']}: {side} {abs(mc['gold_adv'])} {t['label_gold_unit']}")
    if mc.get("xp_adv") is not None:
        side = t["gold_adv_radiant"] if mc["xp_adv"] > 0 else t["gold_adv_dire"]
        match_ctx_parts.append(f"{t['label_xp_diff']}: {side} {abs(mc['xp_adv'])}")
    if mc.get("radiant_score") is not None:
        match_ctx_parts.append(
            f"{t['label_score']}: {t['label_radiant']} {mc.get('radiant_score', 0)} - "
            f"{mc.get('dire_score', 0)} {t['label_dire']}"
        )
    match_ctx_str = "\n".join(match_ctx_parts) if match_ctx_parts else t["no_match_info"]

    # Nearby allies
    ally_lines = []
    for ally in context.ally_states:
        ally_lines.append("  " + _format_hero_state(ally, t["label_ally"], lang=lang))
    allies_str = "\n".join(ally_lines) if ally_lines else "  " + t["no_allies"]

    # Nearby enemies
    enemy_lines = []
    for enemy in context.enemy_states:
        enemy_lines.append("  " + _format_hero_state(enemy, t["label_enemy"], lang=lang))
    enemies_str = "\n".join(enemy_lines) if enemy_lines else "  " + t["no_enemies"]

    # Distant enemies (from map_state, populated by context_builder)
    distant_enemies_section = ""
    distant_list = context.map_state.get("distant_enemies", []) if context.map_state else []
    if distant_list:
        de_lines = []
        for de in distant_list:
            alive_str = t["label_alive"] if de.get("is_alive", True) else t["label_dead"]
            de_lines.append(f"  - {de['hero_name']} (Lv{de.get('level', '?')}, {alive_str})")
        distant_enemies_section = (
            f"\n## {t['section_distant_enemies']}\n" + "\n".join(de_lines)
        )

    # Knowledge section
    knowledge_section = knowledge.strip() if knowledge else t["no_knowledge"]

    # Vision analysis section (most important for deaths/overextensions)
    if context.vision_analysis and context.vision_analysis.formatted_text:
        vision_section = context.vision_analysis.formatted_text
    else:
        vision_section = _format_map_state(context.map_state, lang=lang)

    # Causal chain section (for death analysis)
    causal_chain_section = ""
    if context.causal_chain and context.causal_chain.formatted_text:
        causal_chain_section = f"## {t['section_causal']}\n{context.causal_chain.formatted_text}"

    prompt = f"""{t['system_role']}

## {t['section_decision']}
- {t['label_time']}: {time_str}
- {t['label_type']}: {dp.point_type}
- {t['label_severity']}: {dp.severity}
- {t['label_brief']}: {dp.brief}
- {t['label_phase']}: {phase_str}

## {t['section_player']}
{_format_hero_state(context.player_state, t['label_player'], lang=lang)}

## {t['section_allies']}
{allies_str}

## {t['section_enemies']}
{enemies_str}
{distant_enemies_section}

## {t['section_timeline']}
{_format_event_timeline(context.event_timeline, lang=lang)}

## {t['section_vision']}
{vision_section}

{causal_chain_section}

## {t['section_match']}
{match_ctx_str}

## {t['section_knowledge']}
{knowledge_section}

## {t['section_task']}
{t['task_intro']}

{t['eval_points']}

{t['response_format']}

{t['grading_scale']}"""

    return prompt


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_evaluation_response(text: str, decision_point: DecisionPoint, lang: str = "zh") -> Evaluation:
    """Parse the structured response from the LLM into an Evaluation.

    Handles multiple format variations:
    - Plain:    Label: Value
    - Markdown: **Label:** Value  or  **Label**: Value
    - Header:   ## Label\\nValue
    """
    labels = _RESPONSE_LABELS.get(lang, _RESPONSE_LABELS["zh"])
    score_label = labels["score"]
    analysis_label = labels["analysis"]
    mistake_label = labels["mistake"]
    better_label = labels["better_play"]
    lesson_label = labels["lesson"]
    grading_label = labels["grading_scale"]

    score = "C"
    analysis = ""
    mistake = ""
    better_play = ""
    lesson = ""

    # Normalize markdown bold markers for easier parsing:
    # "**Label:** S" -> "Label: S"   and   "**Label**: S" -> "Label: S"
    # The colon may be inside or outside the bold markers
    cleaned = re.sub(r"\*\*([^*:：]+)[:\uff1a]?\*\*[:\uff1a]?\s*", r"\1: ", text)
    # Also handle "## Label" headers -> "Label:"
    cleaned = re.sub(r"^##\s*", "", cleaned, flags=re.MULTILINE)

    # Parse score — works for both Chinese "评分" and English "Score"
    score_match = re.search(rf"{re.escape(score_label)}[:\uff1a\s]*([SABCD])", cleaned)
    if not score_match:
        # Fallback: try both languages
        score_match = re.search(r"(?:评分|Score)[:\uff1a\s]*([SABCD])", cleaned)
    if score_match:
        score = score_match.group(1)

    # For multi-line fields, lookahead for next section label or end.
    _SEP = r"[:\uff1a]?\s*"  # optional colon + whitespace
    _NEXT_MISTAKE = rf"\n\s*{re.escape(mistake_label)}{_SEP}"
    _NEXT_BETTER = rf"\n\s*{re.escape(better_label)}{_SEP}"
    _NEXT_LESSON = rf"\n\s*{re.escape(lesson_label)}{_SEP}"

    analysis_match = re.search(
        rf"{re.escape(analysis_label)}{_SEP}(.+?)(?={_NEXT_MISTAKE}|\Z)", cleaned, re.DOTALL
    )
    if analysis_match:
        analysis = analysis_match.group(1).strip()

    mistake_match = re.search(
        rf"{re.escape(mistake_label)}{_SEP}(.+?)(?={_NEXT_BETTER}|\Z)", cleaned, re.DOTALL
    )
    if mistake_match:
        mistake = mistake_match.group(1).strip()

    better_match = re.search(
        rf"{re.escape(better_label)}{_SEP}(.+?)(?={_NEXT_LESSON}|\Z)", cleaned, re.DOTALL
    )
    if better_match:
        better_play = better_match.group(1).strip()

    lesson_match = re.search(
        rf"{re.escape(lesson_label)}{_SEP}(.+?)(?=\n\s*{re.escape(grading_label)}|\n##|\Z)",
        cleaned,
        re.DOTALL,
    )
    if lesson_match:
        lesson = lesson_match.group(1).strip()

    # If analysis is still empty, use the raw API response as the analysis
    # (better than a generic placeholder)
    if not analysis:
        # Strip the score line from the raw text to avoid repetition
        fallback_text = re.sub(
            rf"{re.escape(score_label)}[:\uff1a]\s*[SABCD]\s*\n?", "", text
        ).strip()
        analysis = fallback_text[:500] if fallback_text else ""
        if analysis:
            logger.debug("API response parsed with fallback (raw text used for analysis)")
        else:
            logger.warning("API response could not be parsed at all: %.200s", text)

    parsed_any = bool(analysis or mistake or better_play or lesson)

    default_analysis = (
        decision_point.brief or
        ("Evaluation complete" if lang == "en" else "评估完成")
    )

    return Evaluation(
        decision_point=decision_point,
        score=score,
        analysis=analysis or default_analysis,
        mistake=mistake,
        better_play=better_play,
        lesson=lesson,
        _parsed_ok=parsed_any,
    )


# ---------------------------------------------------------------------------
# Main evaluation functions
# ---------------------------------------------------------------------------

async def evaluate_decision(
    context: DecisionContext, knowledge: str, lang: str = "zh",
) -> Evaluation:
    """Evaluate a single decision point using DeepSeek API.

    Parameters
    ----------
    context : DecisionContext
        Full context for the decision being evaluated.
    knowledge : str
        Injected knowledge text (matchup tips, hero-specific advice, etc.).
    lang : str
        Language code ("zh" or "en"). Defaults to "zh".

    Returns
    -------
    Evaluation
        The structured evaluation result.
    """
    if not is_api_available():
        logger.warning("DeepSeek API 不可用，使用规则回退")
        return evaluate_decision_fallback(context, lang=lang)

    try:
        client = get_shared_client()
        prompt = _build_evaluation_prompt(context, knowledge, lang=lang)

        response = await client.chat.completions.create(
            model=MODEL,
            max_tokens=3000,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.choices[0].message.content or ""

        return _parse_evaluation_response(response_text, context.decision_point, lang=lang)

    except Exception as exc:
        exc_str = str(exc)
        if "rate_limit" in exc_str or "429" in exc_str:
            logger.info("DeepSeek API 速率限制")
        elif "overloaded" in exc_str or "529" in exc_str:
            logger.info("DeepSeek API 过载")
        else:
            logger.warning("DeepSeek API 调用失败: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

_SCORE_VALUES = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}


def evaluate_decision_fallback(context: DecisionContext, lang: str = "zh") -> Evaluation:
    """Rule-based fallback when DeepSeek API is unavailable.

    Uses simple heuristics based on the decision point type and surrounding
    context to produce a rough evaluation.
    """
    dp = context.decision_point
    ps = context.player_state
    phase = context.game_phase

    score = "C"
    analysis = ""
    mistake = ""
    better_play = ""
    lesson = ""

    if dp.point_type == "death":
        if context.game_phase == "pregame":
            score, analysis, mistake, better_play, lesson = _evaluate_pregame_death(
                dp, ps, context, lang=lang
            )
        else:
            score, analysis, mistake, better_play, lesson = _evaluate_death(
                dp, ps, context, lang=lang
            )
    elif dp.point_type == "bad_trade":
        score, analysis, mistake, better_play, lesson = _evaluate_bad_trade(
            dp, ps, context, lang=lang
        )
    elif dp.point_type == "overextend":
        score, analysis, mistake, better_play, lesson = _evaluate_overextend(
            dp, ps, context, lang=lang
        )
    elif dp.point_type == "good_play":
        score, analysis, mistake, better_play, lesson = _evaluate_good_play(
            dp, ps, context, lang=lang
        )
    elif dp.point_type == "fight_engage":
        score, analysis, mistake, better_play, lesson = _evaluate_fight_engage(
            dp, ps, context, lang=lang
        )
    elif dp.point_type == "missed_cs":
        if lang == "en":
            score = "C"
            analysis = f"{dp.hero_name} had low CS efficiency during {phase or 'unknown'} phase."
            mistake = "Missed a significant number of last hits"
            better_play = "Watch creep HP closely and start attack animations earlier"
            lesson = "Last-hitting is the foundation of farming — every missed creep is lost gold"
        else:
            score = "C"
            analysis = f"{dp.hero_name}在{phase or '未知'}阶段补刀效率偏低。"
            mistake = "漏补了较多小兵"
            better_play = "注意小兵血量，提前准备攻击动画"
            lesson = "补刀是经济的基础，每个漏掉的兵都是金币损失"
    else:
        # Generic fallback for unknown types
        if lang == "en":
            score = "B"
            analysis = dp.brief or f"{dp.hero_name} made a decision at {dp.time // 60} minutes."
            lesson = "Keep reviewing replays to improve game sense"
        else:
            score = "B"
            analysis = dp.brief or f"{dp.hero_name}在{dp.time // 60}分钟做出了一个决策。"
            lesson = "持续复盘提高比赛理解"

    return Evaluation(
        decision_point=dp,
        score=score,
        analysis=analysis,
        mistake=mistake,
        better_play=better_play,
        lesson=lesson,
    )


# ---------------------------------------------------------------------------
# Rule-based sub-evaluators
# ---------------------------------------------------------------------------

def _evaluate_pregame_death(
    dp: DecisionPoint,
    ps: HeroState,
    ctx: DecisionContext,
    lang: str = "zh",
) -> tuple[str, str, str, str, str]:
    """Evaluate a death during the pregame phase (bounty rune contest)."""
    death_event = dp.related_death
    killer = death_event.killer_hero if death_event else ("Unknown" if lang == "en" else "未知")

    # enemy_states is already proximity-filtered to nearby enemies, but
    # ally_states holds ALL 4 teammates regardless of position — compare
    # against allies actually near the player, or the "outnumbered" branch
    # can never trigger and the NvM headcount in the text is wrong.
    num_enemies = len(ctx.enemy_states)
    num_allies = count_nearby_allies(ps, ctx.ally_states, ctx.game_phase or "pregame")
    outnumbered = num_enemies > num_allies + 1

    if outnumbered:
        # Dying in a contested rune fight while outnumbered: the real mistake
        # was contesting at all, but the death itself is understandable — C.
        score = "C"
        if lang == "en":
            analysis = (
                f"{dp.hero_name} was killed by {killer} while contesting the bounty rune "
                f"outnumbered ({num_allies + 1}v{num_enemies}). "
                f"A bounty rune is worth ~40 gold, but dying costs 100+ gold plus lane time."
            )
            mistake = "Contested the bounty rune despite being outnumbered"
            better_play = "When the enemy shows more numbers, abandon the bounty rune and go straight to lane"
            lesson = "A bounty rune is not worth dying for — staying alive and getting to lane is more valuable"
        else:
            analysis = (
                f"{dp.hero_name}在人数劣势（{num_allies + 1}v{num_enemies}）的赏金符争夺中被{killer}击杀。"
                f"赏金符价值约40金，死亡代价远超此数。"
            )
            mistake = "在人数劣势下仍然争夺赏金符"
            better_play = "看到敌方人多时应放弃赏金符，直接去线上"
            lesson = "赏金符不值得用命换，保命上线更重要"
    else:
        # Numbers were even or favorable and the player still died before the
        # game even started — a careless death, graded more harshly — D.
        score = "D"
        if lang == "en":
            analysis = (
                f"{dp.hero_name} was killed by {killer} in the pregame phase even though "
                f"the numbers were not unfavorable ({num_allies + 1}v{num_enemies}). "
                f"A free death before creeps spawn costs 100+ gold plus lane time."
            )
            mistake = "Died carelessly in a pregame skirmish despite even or favorable numbers"
            better_play = "Respect enemy level 1 damage and disables — keep full HP and stay with allies instead of face-checking"
            lesson = "Even a winnable pregame fight is lost by careless positioning — never give away a free first blood"
        else:
            analysis = (
                f"{dp.hero_name}在人数并不吃亏（{num_allies + 1}v{num_enemies}）的开局阶段"
                f"被{killer}击杀。开局白送一次死亡损失100+金并耽误上线。"
            )
            mistake = "开局人数不亏却大意送出死亡"
            better_play = "尊重敌方1级的伤害和控制，保持满血并与队友抱团，不要贸然探脸"
            lesson = "开局团人数占优也可能因站位大意而翻车，绝不能白送一血"

    return score, analysis, mistake, better_play, lesson


def _evaluate_death(
    dp: DecisionPoint,
    ps: HeroState,
    ctx: DecisionContext,
    lang: str = "zh",
) -> tuple[str, str, str, str, str]:
    """Evaluate a death event."""
    death_event = dp.related_death
    killer = death_event.killer_hero if death_event else ("Unknown" if lang == "en" else "未知")
    is_tf = death_event.is_teamfight if death_event else False

    # Count nearby enemies and allies at the time
    num_enemies = len(ctx.enemy_states)
    num_allies = len(ctx.ally_states)

    # Check if enemies were near-full HP (suggesting the player was caught out)
    enemies_healthy = sum(
        1
        for e in ctx.enemy_states
        if e.max_hp > 0 and e.hp / e.max_hp > 0.7
    )

    if lang == "en":
        score, analysis, mistake, better_play, lesson = _evaluate_death_en(
            dp, ps, ctx, killer, is_tf, num_enemies, num_allies, enemies_healthy
        )
    else:
        score, analysis, mistake, better_play, lesson = _evaluate_death_zh(
            dp, ps, ctx, killer, is_tf, num_enemies, num_allies, enemies_healthy
        )

    # Check if player had unused defensive items
    if _HAS_GAME_DATA:
        defensive_items = []
        for item in ps.items:
            item_info = get_item_data(item)
            if item_info and item_info.get("is_active") and item_info.get("category") == "defense":
                defensive_items.append(item)
        if defensive_items:
            sep = ", " if lang == "en" else "、"
            items_text = sep.join(defensive_items)
            if not is_tf:  # Solo death with defensive items = worse
                if lang == "en":
                    analysis += f" Player had defensive active items ({items_text}) but failed to use them effectively."
                else:
                    analysis += f" 玩家拥有{items_text}等防御性主动装备，但未能有效利用。"
                if _SCORE_VALUES.get(score, 0) > _SCORE_VALUES.get("C", 0):
                    score = "C"

    # Check vision analysis for more specific feedback
    va = ctx.vision_analysis
    if va and va.state_at_decision:
        vs = va.state_at_decision
        if vs.danger_level == "critical":
            score = "D"
            if lang == "en":
                analysis += f" Vision analysis shows {vs.missing_enemy_count} enemy heroes were missing,"
                if vs.longest_missing_seconds > 20:
                    analysis += f" with the longest missing for {vs.longest_missing_seconds}s."
                if not vs.player_in_safe_zone:
                    reasons = "; ".join(vs.danger_reasons[:2])
                    mistake = f"In a dangerous position with severely lacking vision, {reasons}"
                if va.timeline.critical_description:
                    better_play = f"Should have retreated at {va.timeline.critical_description}"
            else:
                analysis += f" 视野分析显示当时{vs.missing_enemy_count}名敌方英雄消失，"
                if vs.longest_missing_seconds > 20:
                    analysis += f"其中最长消失{vs.longest_missing_seconds}秒。"
                if not vs.player_in_safe_zone:
                    mistake = "在视野严重不足的情况下处于危险位置，" + "；".join(vs.danger_reasons[:2])
                if va.timeline.critical_description:
                    better_play = f"在{va.timeline.critical_description}时就应该撤退"
        elif vs.danger_level == "danger":
            if _SCORE_VALUES.get(score, 0) > _SCORE_VALUES.get("C", 0):
                score = "C"
            if lang == "en":
                analysis += f" {vs.missing_enemy_count} enemy heroes were missing."
                if va.timeline.critical_description:
                    analysis += f" {va.timeline.critical_description}."
            else:
                analysis += f" 有{vs.missing_enemy_count}名敌方英雄消失。"
                if va.timeline.critical_description:
                    analysis += f" {va.timeline.critical_description}。"
        elif vs.danger_level == "caution":
            if vs.missing_enemy_count > 0:
                if lang == "en":
                    analysis += f" Note: {vs.missing_enemy_count} enemy heroes were missing."
                else:
                    analysis += f" 注意：有{vs.missing_enemy_count}名敌方英雄消失。"
        else:
            if not vs.player_in_safe_zone and not vs.nearby_ward_coverage:
                if lang == "en":
                    analysis += " Was far from tower with no ward coverage nearby."
                else:
                    analysis += " 当时远离防御塔且附近无眼位覆盖。"

    # Causal chain root cause enhancement
    cc = ctx.causal_chain
    if cc and cc.root_cause_summary:
        if cc.root_cause_type in ("ignore_missing", "overextend", "bad_positioning", "greedy_farm"):
            if lang == "en":
                analysis += f" Root cause: {cc.root_cause_summary}."
                if cc.root_cause_type == "ignore_missing":
                    mistake = f"Ignored missing enemy signals: {cc.root_cause_summary}"
                    better_play = "Retreat to safety immediately when enemy heroes go missing"
                elif cc.root_cause_type == "overextend":
                    mistake = f"Overextended: {cc.root_cause_summary}"
                    better_play = "Stay within vision and tower protection range"
            else:
                analysis += f" 根因分析：{cc.root_cause_summary}。"
                if cc.root_cause_type == "ignore_missing":
                    mistake = f"忽视敌方消失信号：{cc.root_cause_summary}"
                    better_play = "看到敌方英雄消失时立即撤退到安全位置"
                elif cc.root_cause_type == "overextend":
                    mistake = f"走位过深：{cc.root_cause_summary}"
                    better_play = "保持在视野和塔保护范围内"

    return score, analysis, mistake, better_play, lesson


def _evaluate_death_zh(
    dp: DecisionPoint,
    ps: HeroState,
    ctx: DecisionContext,
    killer: str,
    is_tf: bool,
    num_enemies: int,
    num_allies: int,
    enemies_healthy: int,
) -> tuple[str, str, str, str, str]:
    """Chinese death evaluation text."""
    if is_tf:
        if num_allies >= 2:
            score = "B"
            analysis = (
                f"{dp.hero_name}在团战中被{killer}击杀。"
                f"当时有{num_allies}名队友在附近参与团战。"
            )
            mistake = "团战中站位可能有问题，或技能释放时机不佳"
            better_play = "注意团战中保持安全距离，优先输出同时避免被集火"
            lesson = "团战中活下来和打出输出同样重要"
        else:
            score = "C"
            analysis = (
                f"{dp.hero_name}在团战中被{killer}击杀，"
                f"但附近队友较少（{num_allies}人），可能是被分割或支援到晚了。"
            )
            mistake = "加入团战的时机不对，或者团战时走位不当"
            better_play = "等待队友集合再参与团战，或选择更好的切入时机"
            lesson = "团战要确保人齐再打，单独行动很容易被击杀"
    elif enemies_healthy and num_enemies > num_allies:
        score = "D"
        analysis = (
            f"{dp.hero_name}在人数劣势（{num_allies + 1}v{num_enemies}）的情况下"
            f"被{killer}击杀。敌方多名英雄状态良好。"
        )
        mistake = "在没有视野保护的情况下处于危险区域，被敌方包夹"
        better_play = "注意看小地图，当敌方消失时不要单独推进到危险区域"
        lesson = "地图意识是生存的关键，缺少视野时不要冒险"
    elif ctx.game_phase == "laning":
        hp_pct = (ps.hp / ps.max_hp * 100) if ps.max_hp > 0 else 0
        if hp_pct < 30:
            score = "C"
            analysis = (
                f"{dp.hero_name}在对线期被{killer}击杀。"
                f"被击杀前血量较低（{hp_pct:.0f}%），可能是换血不利后没有回家。"
            )
            mistake = "血量过低还留在线上，应该提前回城或使用恢复品"
            better_play = "血量低于40%时考虑回城补给，不要贪补刀"
            lesson = "生存优先于补刀，死亡的损失远大于漏几个兵"
        else:
            score = "D"
            analysis = (
                f"{dp.hero_name}在对线期被{killer}击杀。"
                f"被击杀时血量仍有{hp_pct:.0f}%，说明被对方成功找到了机会。"
            )
            mistake = "可能走位过于靠前，或没有预判到对方的击杀意图"
            better_play = "注意对手的技能冷却和攻击范围，保持安全距离"
            lesson = "对线要尊重对手的击杀能力，了解双方英雄的强弱时段"
    else:
        score = "C"
        analysis = f"{dp.hero_name}被{killer}击杀。{dp.brief}"
        mistake = "被击杀意味着丢失金钱和时间，给对手提供了优势"
        better_play = "更加注意走位和地图信息，减少不必要的风险"
        lesson = "每次死亡都是一次学习机会，分析自己可以在哪个时间点做出不同选择"

    return score, analysis, mistake, better_play, lesson


def _evaluate_death_en(
    dp: DecisionPoint,
    ps: HeroState,
    ctx: DecisionContext,
    killer: str,
    is_tf: bool,
    num_enemies: int,
    num_allies: int,
    enemies_healthy: int,
) -> tuple[str, str, str, str, str]:
    """English death evaluation text."""
    if is_tf:
        if num_allies >= 2:
            score = "B"
            analysis = (
                f"{dp.hero_name} was killed by {killer} during a teamfight. "
                f"{num_allies} allies were nearby."
            )
            mistake = "Positioning in the teamfight may have been off, or ability timing was poor"
            better_play = "Maintain safe distance in teamfights — deal damage while avoiding being focused"
            lesson = "Surviving a teamfight is just as important as dealing damage"
        else:
            score = "C"
            analysis = (
                f"{dp.hero_name} was killed by {killer} in a teamfight, "
                f"but few allies were nearby ({num_allies}). May have been isolated or arrived too late."
            )
            mistake = "Poor timing joining the fight, or bad positioning during the teamfight"
            better_play = "Wait for teammates to gather before engaging, or find a better entry point"
            lesson = "Teamfights require full commitment from the team — avoid fighting alone"
    elif enemies_healthy and num_enemies > num_allies:
        score = "D"
        analysis = (
            f"{dp.hero_name} was killed by {killer} while outnumbered "
            f"({num_allies + 1}v{num_enemies}). Multiple enemies were near full HP."
        )
        mistake = "Was in a dangerous area without vision, and got caught by the enemy"
        better_play = "Check the minimap — do not push into dangerous areas alone when enemies are missing"
        lesson = "Map awareness is key to survival — do not take risks without vision"
    elif ctx.game_phase == "laning":
        hp_pct = (ps.hp / ps.max_hp * 100) if ps.max_hp > 0 else 0
        if hp_pct < 30:
            score = "C"
            analysis = (
                f"{dp.hero_name} was killed by {killer} during laning. "
                f"HP was low ({hp_pct:.0f}%) before the kill — likely lost a trade and didn't ferry regen or TP out."
            )
            mistake = "Stayed in lane with dangerously low HP instead of using regen or TPing to base"
            better_play = "Consider TPing home when below 40% HP — don't be greedy for last hits"
            lesson = "Survival takes priority over CS — the cost of dying far exceeds a few missed creeps"
        else:
            score = "D"
            analysis = (
                f"{dp.hero_name} was killed by {killer} during laning. "
                f"HP was still at {hp_pct:.0f}%, suggesting the opponent found a kill opportunity."
            )
            mistake = "May have been positioned too aggressively or failed to anticipate the enemy's kill threat"
            better_play = "Respect the enemy's skill cooldowns and attack range — maintain safe distance"
            lesson = "Respect your opponent's kill potential and understand power spikes"
    else:
        score = "C"
        analysis = f"{dp.hero_name} was killed by {killer}. {dp.brief}"
        mistake = "Being killed means losing gold, time, and giving the enemy an advantage"
        better_play = "Pay more attention to positioning and map info — reduce unnecessary risks"
        lesson = "Every death is a learning opportunity — analyze what decision could have been different"

    return score, analysis, mistake, better_play, lesson


def _evaluate_bad_trade(
    dp: DecisionPoint,
    ps: HeroState,
    ctx: DecisionContext,
    lang: str = "zh",
) -> tuple[str, str, str, str, str]:
    """Evaluate a bad trade (took more damage than dealt)."""
    if lang == "en":
        score = "C"
        analysis = f"{dp.hero_name} came out behind in a trade. {dp.brief}"
        mistake = "Poor trade timing or execution — took unnecessary damage"
        better_play = (
            "Choose better trade windows, such as when the opponent is last-hitting, "
            "or exploit cooldown advantages"
        )
        lesson = "Good trades should be taken when you have an advantage — avoid getting counter-traded"
    else:
        score = "C"
        analysis = (
            f"{dp.hero_name}在换血中吃了亏。{dp.brief}"
        )
        mistake = "换血的时机或方式不佳，承受了不必要的伤害"
        better_play = (
            "选择更好的换血时机，比如对手正在补刀时进行消耗，"
            "或利用技能CD差进行有利交换"
        )
        lesson = "好的换血应该是在自己有优势的条件下进行，避免被反打"
    return score, analysis, mistake, better_play, lesson


def _evaluate_overextend(
    dp: DecisionPoint,
    ps: HeroState,
    ctx: DecisionContext,
    lang: str = "zh",
) -> tuple[str, str, str, str, str]:
    """Evaluate an overextension."""
    num_enemies_missing = ctx.match_context.get("enemies_missing", 0)
    score = "C" if num_enemies_missing <= 1 else "D"

    if lang == "en":
        analysis = f"{dp.hero_name} pushed too far into unsafe territory. {dp.brief}"
        mistake = "Advanced too deep without vision or team support"
        better_play = "Stay in areas with ward coverage; place wards before pushing"
        lesson = "Conservative positioning beats risky pushes, especially when enemies are missing"
    else:
        analysis = (
            f"{dp.hero_name}过度深入了不安全的区域。{dp.brief}"
        )
        mistake = "在缺少视野或队友支援的情况下推进太深"
        better_play = "保持在有视野覆盖的区域内活动，推进前先做好视野"
        lesson = "保守的走位比冒险推进更稳定，特别是敌方消失时"

    # Enhance with vision analysis if available
    va = ctx.vision_analysis
    if va and va.state_at_decision:
        vs = va.state_at_decision
        if vs.missing_enemy_count >= 3 and not vs.nearby_ward_coverage:
            score = "D"
            if lang == "en":
                mistake = f"Pushed deep with {vs.missing_enemy_count} enemies missing and no ward coverage"
                if vs.danger_reasons:
                    analysis += f" {'; '.join(vs.danger_reasons[:2])}."
            else:
                mistake = f"在{vs.missing_enemy_count}名敌方消失且无视野覆盖的情况下过深推进"
                if vs.danger_reasons:
                    analysis += f" {'; '.join(vs.danger_reasons[:2])}。"
        elif vs.missing_enemy_count >= 2 and not vs.player_in_safe_zone:
            if _SCORE_VALUES.get(score, 0) > _SCORE_VALUES.get("C", 0):
                score = "C"
            if lang == "en":
                analysis += f" {vs.missing_enemy_count} enemy heroes were missing at the time."
            else:
                analysis += f" 当时有{vs.missing_enemy_count}名敌方英雄消失。"

    return score, analysis, mistake, better_play, lesson


def _evaluate_good_play(
    dp: DecisionPoint,
    ps: HeroState,
    ctx: DecisionContext,
    lang: str = "zh",
) -> tuple[str, str, str, str, str]:
    """Evaluate a positive play (multi-kill, clutch save, etc.)."""
    has_multi_kill = any(
        "多杀" in evt.description or "multi" in evt.event_type.lower()
        for evt in ctx.event_timeline
    )

    if lang == "en":
        no_mistake = "None"
        keep_up = "Keep it up"
        if has_multi_kill:
            score = "S"
            analysis = f"{dp.hero_name} pulled off an outstanding multi-kill! {dp.brief}"
            lesson = "Capitalizing on enemy mistakes for maximum impact is the mark of a skilled player"
        else:
            score = "A"
            analysis = f"{dp.hero_name} made an excellent decision. {dp.brief}"
            lesson = "Good decisions come from accurate reads on the game state"
        mistake = no_mistake
        better_play = keep_up
    else:
        no_mistake = "无"
        keep_up = "继续保持"
        if has_multi_kill:
            score = "S"
            analysis = f"{dp.hero_name}完成了一次精彩的多杀操作！{dp.brief}"
            lesson = "抓住敌方失误的机会进行最大化收益是高手的标志"
        else:
            score = "A"
            analysis = f"{dp.hero_name}做出了一个很好的决策。{dp.brief}"
            lesson = "好的决策来自于对局势的准确判断"
        mistake = no_mistake
        better_play = keep_up

    return score, analysis, mistake, better_play, lesson


def _evaluate_fight_engage(
    dp: DecisionPoint,
    ps: HeroState,
    ctx: DecisionContext,
    lang: str = "zh",
) -> tuple[str, str, str, str, str]:
    """Evaluate a fight engagement decision."""
    num_allies = len(ctx.ally_states)
    num_enemies = len(ctx.enemy_states)

    if lang == "en":
        if num_allies >= num_enemies:
            score = "B"
            analysis = (
                f"{dp.hero_name} initiated a teamfight ({num_allies + 1}v{num_enemies}). "
                f"{dp.brief}"
            )
            mistake = "Engagement timing could be better"
            better_play = "Make sure key abilities are off cooldown before initiating"
            lesson = "Initiate when you have advantages in numbers, health, and cooldowns"
        else:
            score = "C"
            analysis = (
                f"{dp.hero_name} engaged while outnumbered "
                f"({num_allies + 1}v{num_enemies}). {dp.brief}"
            )
            mistake = "Should not have initiated while outnumbered"
            better_play = "Wait for teammates to regroup, or retreat to defend objectives"
            lesson = "Avoid fighting at a numbers disadvantage unless you have a massive lead in farm or levels"
    else:
        if num_allies >= num_enemies:
            score = "B"
            analysis = (
                f"{dp.hero_name}发起了一次团战（{num_allies + 1}v{num_enemies}）。"
                f"{dp.brief}"
            )
            mistake = "开团时机可以更好"
            better_play = "确保关键技能都准备好再开团"
            lesson = "开团要选择人数和状态都有优势的时机"
        else:
            score = "C"
            analysis = (
                f"{dp.hero_name}在人数劣势下发起了战斗"
                f"（{num_allies + 1}v{num_enemies}）。{dp.brief}"
            )
            mistake = "不应该在人数劣势时开团"
            better_play = "等待队友汇合，或选择撤退保塔"
            lesson = "除非有极大优势（装备/等级/位置），否则不要打人数劣势的团"

    return score, analysis, mistake, better_play, lesson


# ---------------------------------------------------------------------------
# Phase review (one LLM call per game-time phase)
# ---------------------------------------------------------------------------

_PHASE_LABELS_ZH = {
    "laning":     "对线期",
    "early_mid":  "前中期",
    "mid_late":   "中后期",
    "late":       "后期",
    "ultra_late": "大后期",
}
_PHASE_LABELS_EN = {
    "laning":     "Laning",
    "early_mid":  "Early-Mid Game",
    "mid_late":   "Mid-Late Game",
    "late":       "Late Game",
    "ultra_late": "Ultra-Late Game",
}


def _format_lead(value: int, lang: str) -> str:
    if value == 0:
        return "持平" if lang == "zh" else "even"
    side = ("天辉" if value > 0 else "夜魇") if lang == "zh" else ("Radiant" if value > 0 else "Dire")
    return f"{side} +{abs(value)}"


def _build_phase_prompt(
    *, hero_name: str, position_label: str, team: str,
    phase_key: str, phase_stats: dict, lang: str,
) -> str:
    """Build a position-aware prompt for one phase review."""
    if lang == "en":
        phase_label = _PHASE_LABELS_EN.get(phase_key, phase_key)
        team_label = "Radiant" if team == "radiant" else "Dire"
        gold_lead = _format_lead(phase_stats.get("radiant_gold_lead", 0), "en")
        xp_lead = _format_lead(phase_stats.get("radiant_xp_lead", 0), "en")
        return (
            f"You are a Dota 2 high-MMR coach. Evaluate one player in a single "
            f"phase of one match.\n\n"
            f"Player: hero={hero_name}, role={position_label}, team={team_label}\n"
            f"Phase: {phase_label} ({phase_stats.get('time_range','')})\n\n"
            f"Stats in this phase window:\n"
            f"- KDA: {phase_stats.get('kills',0)}/"
            f"{phase_stats.get('deaths',0)}/{phase_stats.get('assists',0)}\n"
            f"- Last hits gained: {phase_stats.get('last_hits_delta',0)}\n"
            f"- Gold gained: {phase_stats.get('gold_delta',0)} "
            f"(GPM={phase_stats.get('gpm_in_phase',0)})\n"
            f"- XP gained: {phase_stats.get('xp_delta',0)} "
            f"(XPM={phase_stats.get('xpm_in_phase',0)})\n"
            f"- Death timestamps (sec): {phase_stats.get('death_times',[])}\n"
            f"- Teamfights involved: {phase_stats.get('teamfight_count',0)} "
            f"(team won {phase_stats.get('teamfight_wins',0)})\n"
            f"- Smokes by your team: {phase_stats.get('smokes_used_self_team',0)}\n"
            f"- Towers taken (Radiant/Dire): "
            f"{phase_stats.get('towers_taken_radiant',0)}/"
            f"{phase_stats.get('towers_taken_dire',0)}\n"
            f"- End-of-phase gold lead: {gold_lead}\n"
            f"- End-of-phase XP lead: {xp_lead}\n\n"
            f"Judge through the lens of role={position_label}. Different positions "
            f"have different responsibilities each phase (Position 5 cares about "
            f"wards/saves/pulls more than last hits; Position 1 cares about farm "
            f"efficiency more than early kills).\n\n"
            f"Output STRICT JSON only (no markdown, no prose), exactly this shape:\n"
            f'{{"score":"S|A|B|C|D","summary":"<1-2 sentences>","highlights":["..."],"issues":["..."],"suggestions":["..."]}}'
            f"\n\nKeep each list item under 30 words. Return [] for empty lists."
        )

    phase_label = _PHASE_LABELS_ZH.get(phase_key, phase_key)
    team_label = "天辉" if team == "radiant" else "夜魇"
    gold_lead = _format_lead(phase_stats.get("radiant_gold_lead", 0), "zh")
    xp_lead = _format_lead(phase_stats.get("radiant_xp_lead", 0), "zh")
    return (
        f"你是一名 Dota 2 高分教练。请评价一名玩家在一场比赛的某个阶段的表现。\n\n"
        f"玩家信息：英雄={hero_name}，位置={position_label}，队伍={team_label}\n"
        f"阶段：{phase_label}（{phase_stats.get('time_range','')}）\n\n"
        f"该阶段统计：\n"
        f"- KDA：{phase_stats.get('kills',0)}/"
        f"{phase_stats.get('deaths',0)}/{phase_stats.get('assists',0)}\n"
        f"- 补刀增量：{phase_stats.get('last_hits_delta',0)}\n"
        f"- 经济增量：{phase_stats.get('gold_delta',0)} "
        f"（GPM={phase_stats.get('gpm_in_phase',0)}）\n"
        f"- 经验增量：{phase_stats.get('xp_delta',0)} "
        f"（XPM={phase_stats.get('xpm_in_phase',0)}）\n"
        f"- 死亡时间点（秒）：{phase_stats.get('death_times',[])}\n"
        f"- 参与团战：{phase_stats.get('teamfight_count',0)} 次"
        f"（队伍赢 {phase_stats.get('teamfight_wins',0)}）\n"
        f"- 队伍用雾次数：{phase_stats.get('smokes_used_self_team',0)}\n"
        f"- 推塔（双方）：天辉 {phase_stats.get('towers_taken_radiant',0)} / "
        f"夜魇 {phase_stats.get('towers_taken_dire',0)}\n"
        f"- 阶段末经济差：{gold_lead}\n"
        f"- 阶段末经验差：{xp_lead}\n\n"
        f"请以该玩家的位置（{position_label}）的视角评价这段表现。"
        f"不同位置在该阶段的关注点不同：5 号位重保人/插眼/拉野，1 号位重 farm 效率，"
        f"3 号位重承伤换人头，2 号位重节奏，4 号位重游走 gank。\n\n"
        f"请严格只输出下列 JSON 格式（不要 markdown 代码块、不要解释）：\n"
        f'{{"score":"S|A|B|C|D","summary":"1-2 句中文","highlights":["..."],"issues":["..."],"suggestions":["..."]}}'
        f"\n\n每条列表项不超过 30 字。空列表用 []。"
    )


def _parse_phase_response(text: str) -> dict:
    """Parse the LLM's JSON output for a phase review."""
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


async def evaluate_phase(
    *, hero_name: str, position_label: str, team: str,
    phase_key: str, phase_time_range: str, phase_stats_dict: dict,
    lang: str = "zh",
) -> PhaseReview:
    """Generate a per-phase review for one player; never raises."""
    base = PhaseReview(
        phase=phase_key,
        time_range=phase_time_range,
        stats=phase_stats_dict,
    )

    if not is_api_available():
        return _phase_review_fallback(base, phase_stats_dict, lang=lang)

    try:
        client = get_shared_client()
        prompt = _build_phase_prompt(
            hero_name=hero_name,
            position_label=position_label,
            team=team,
            phase_key=phase_key,
            phase_stats=phase_stats_dict,
            lang=lang,
        )
        response = await client.chat.completions.create(
            model=MODEL,
            max_tokens=2200,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = _parse_phase_response(content)
        if not parsed or "score" not in parsed:
            logger.info("phase review: LLM output unparseable, falling back (phase=%s)", phase_key)
            return _phase_review_fallback(base, phase_stats_dict, lang=lang)

        score = str(parsed.get("score", "C"))[:1].upper() or "C"
        base.score = score if score in ("S", "A", "B", "C", "D") else "C"
        base.summary = parsed.get("summary", "") or ""
        base.highlights = [str(x) for x in (parsed.get("highlights") or [])][:5]
        base.issues = [str(x) for x in (parsed.get("issues") or [])][:5]
        base.suggestions = [str(x) for x in (parsed.get("suggestions") or [])][:5]
        base.used_llm = True
        return base

    except Exception as exc:
        logger.warning("evaluate_phase 异常 (phase=%s): %s", phase_key, exc)
        return _phase_review_fallback(base, phase_stats_dict, lang=lang)


def _phase_review_fallback(base: PhaseReview, stats: dict, *, lang: str) -> PhaseReview:
    """Rule-based phase review used when LLM is disabled or errors out."""
    deaths = int(stats.get("deaths", 0))
    kills = int(stats.get("kills", 0))
    teamfight_count = int(stats.get("teamfight_count", 0))
    teamfight_wins = int(stats.get("teamfight_wins", 0))

    if deaths == 0 and kills >= 2:
        score = "A"
    elif deaths >= 3:
        score = "D"
    elif deaths >= 2:
        score = "C"
    elif kills > 0 and deaths == 0:
        score = "B"
    else:
        score = "C"

    if lang == "en":
        base.summary = (
            f"KDA {kills}/{deaths}/{stats.get('assists',0)}, "
            f"teamfights {teamfight_wins}/{teamfight_count}."
        )
    else:
        base.summary = (
            f"KDA {kills}/{deaths}/{stats.get('assists',0)}，"
            f"团战 {teamfight_wins}/{teamfight_count}。"
        )
    base.score = score
    return base
