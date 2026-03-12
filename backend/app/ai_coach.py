"""
AI Coaching Module using Claude API.

Generates detailed coaching insights by analyzing laning phase comparison data
between a player and a reference (pro/high-rank) player.
"""
from __future__ import annotations

import os
import json
from dataclasses import asdict
from typing import Optional

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from .comparison import ComparisonResult, Divergence


# Default to environment variable, can be overridden
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = "claude-sonnet-4-6-20250514"


def is_available() -> bool:
    """Check if AI coaching is available (anthropic SDK installed + API key set)."""
    return HAS_ANTHROPIC and bool(ANTHROPIC_API_KEY)


def _build_coaching_prompt(result: ComparisonResult) -> str:
    """Build the prompt for Claude to analyze the laning comparison."""
    summary = result.summary
    divergences_data = [asdict(d) for d in result.divergences[:15]]  # Limit to top 15
    per_minute = result.per_minute_comparison

    prompt = f"""你是一位专业的Dota 2对线教练。请根据以下对线数据对比分析，为玩家提供详细的改进建议。

## 对局信息
- 玩家英雄: {result.player_hero}
- 参考英雄: {result.ref_hero}
- 玩家分路: {result.player_lane}
- 参考分路: {result.ref_lane}

## 总体评分: {summary.get('grade', 'N/A')}

## 10分钟关键数据对比
- 补刀: 玩家 {summary.get('cs_10min', {}).get('player', 0)} vs 参考 {summary.get('cs_10min', {}).get('ref', 0)}
- 经济: 玩家 {summary.get('gold_10min', {}).get('player', 0)} vs 参考 {summary.get('gold_10min', {}).get('ref', 0)}
- 对线伤害: 玩家 {summary.get('damage_laning', {}).get('player', 0)} vs 参考 {summary.get('damage_laning', {}).get('ref', 0)}
- 恢复品使用: 玩家 {summary.get('regen_used', {}).get('player', 0)} vs 参考 {summary.get('regen_used', {}).get('ref', 0)}

## 每分钟补刀对比
玩家: {json.dumps(per_minute.get('cs', {}).get('player', []))}
参考: {json.dumps(per_minute.get('cs', {}).get('ref', []))}

## 每分钟经济对比
玩家: {json.dumps(per_minute.get('gold', {}).get('player', []))}
参考: {json.dumps(per_minute.get('gold', {}).get('ref', []))}

## 检测到的差异点 ({len(result.divergences)}个)
{json.dumps(divergences_data, ensure_ascii=False, indent=2)}

## 请提供以下分析:

1. **总体评价** (2-3句话概括对线表现)
2. **关键差距分析** (逐项分析补刀、换血、走位、续航等方面的差距及原因)
3. **具体改进建议** (针对每个差异点，给出可操作的具体建议)
4. **练习重点** (列出3-5个优先练习的方向)
5. **进阶技巧** (如果基础没问题，给出更高阶的建议)

注意：
- 请用中文回答
- Dota 2是复杂游戏，不是每个兵都能补，要考虑补刀和维持血量的平衡
- 考虑不同英雄的特性（近战/远程、攻击前摇、弹道速度等）
- 对线不只是补刀，还包括压制对手、控制兵线平衡、合理使用消耗品
"""
    return prompt


async def generate_coaching_report(result: ComparisonResult) -> dict:
    """
    Generate an AI coaching report based on comparison results.

    Returns a dict with:
    - available: bool
    - report: str (markdown formatted coaching report)
    - error: str (if any)
    """
    if not is_available():
        return {
            "available": False,
            "report": "",
            "error": "AI coaching not available. Install anthropic SDK and set ANTHROPIC_API_KEY.",
        }

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = _build_coaching_prompt(result)

        message = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        report_text = ""
        for block in message.content:
            if block.type == "text":
                report_text += block.text

        return {
            "available": True,
            "report": report_text,
            "error": "",
        }
    except Exception as e:
        return {
            "available": False,
            "report": "",
            "error": str(e),
        }


def generate_coaching_report_sync(result: ComparisonResult) -> dict:
    """Synchronous version for non-async contexts."""
    if not is_available():
        return {
            "available": False,
            "report": "",
            "error": "AI coaching not available.",
        }

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = _build_coaching_prompt(result)

        message = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        report_text = ""
        for block in message.content:
            if block.type == "text":
                report_text += block.text

        return {
            "available": True,
            "report": report_text,
            "error": "",
        }
    except Exception as e:
        return {
            "available": False,
            "report": "",
            "error": str(e),
        }


def generate_fallback_report(result: ComparisonResult) -> dict:
    """
    Generate a rule-based coaching report when AI is not available.
    Uses heuristics based on the comparison data.
    """
    summary = result.summary
    divergences = result.divergences

    lines = []
    lines.append(f"## 对线分析报告\n")
    lines.append(f"**评分: {summary.get('grade', 'N/A')}**\n")

    # CS analysis
    cs_data = summary.get('cs_10min', {})
    p_cs = cs_data.get('player', 0)
    r_cs = cs_data.get('ref', 0)
    if r_cs > 0:
        ratio = p_cs / r_cs
        if ratio >= 0.9:
            lines.append(f"### 补刀: 优秀")
            lines.append(f"你的补刀({p_cs})接近参考水平({r_cs})，保持得很好。\n")
        elif ratio >= 0.7:
            lines.append(f"### 补刀: 一般")
            lines.append(f"你的补刀({p_cs})低于参考({r_cs})，差距{r_cs - p_cs}个。建议练习对线补刀节奏。\n")
        else:
            lines.append(f"### 补刀: 需要改进")
            lines.append(f"你的补刀({p_cs})远低于参考({r_cs})，差距{r_cs - p_cs}个，约损失{(r_cs - p_cs) * 45}金币。这是最需要改进的方面。\n")

    # Damage analysis
    dmg_data = summary.get('damage_laning', {})
    p_dmg = dmg_data.get('player', 0)
    r_dmg = dmg_data.get('ref', 0)
    if r_dmg > 0 and p_dmg < r_dmg * 0.5:
        lines.append("### 换血: 过于保守")
        lines.append(f"你的对线伤害({p_dmg})仅为参考({r_dmg})的{int(p_dmg/r_dmg*100)}%。考虑在补刀间隙进行更多骚扰。\n")
    elif r_dmg > 0 and p_dmg > r_dmg * 1.5:
        lines.append("### 换血: 过于激进")
        lines.append(f"你的对线伤害({p_dmg})高于参考({r_dmg})，但注意不要因换血而漏掉补刀。\n")

    # Regen analysis
    regen_data = summary.get('regen_used', {})
    p_regen = regen_data.get('player', 0)
    r_regen = regen_data.get('ref', 0)
    if p_regen > r_regen + 3:
        lines.append("### 续航: 消耗品使用过多")
        lines.append(f"你使用了{p_regen}次恢复品，参考只用了{r_regen}次。减少不必要的吃伤害。\n")

    # Divergence details
    if divergences:
        lines.append("### 关键差异时刻\n")
        for d in divergences[:8]:
            minutes = d.time // 60
            seconds = d.time % 60
            lines.append(f"- **{minutes}:{seconds:02d}** [{d.div_type}] {d.description_zh}")

    # Practice suggestions
    lines.append("\n### 练习建议\n")
    div_types = set(d.div_type for d in divergences)
    if 'cs' in div_types:
        lines.append("1. 在Demo模式下练习补刀，目标10分钟60+补刀")
    if 'trade' in div_types:
        lines.append("2. 练习在补刀间隙进行骚扰，特别是对手准备补刀时")
    if 'positioning' in div_types:
        lines.append("3. 注意站位，不要过于靠后失去对兵线的控制")
    if 'regen' in div_types:
        lines.append("4. 减少不必要的吃伤害，学会走位躲避技能")
    if not div_types:
        lines.append("1. 继续保持当前水平，尝试在更高难度对手面前维持表现")

    return {
        "available": True,
        "report": "\n".join(lines),
        "error": "",
        "is_fallback": True,
    }
