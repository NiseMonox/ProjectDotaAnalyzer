"""Level 3 learner: LLM distillation of match-mining aggregates.

Reads the raw statistical aggregates persisted by Level 2 (match_miner) from
``LEARNED_DIR / MINING_AGGREGATES`` and asks the shared DeepSeek client (from
``ai.evaluator``) to distill them into a small set of natural-language
coaching insights: cross-position comparisons, outliers, and actionable
advice a human coach would give — not stat recitation.

Entry point::

    distill(langs=("zh", "en"), max_chunks=20) -> {"chunks_written": int}

Degrades gracefully: missing aggregates file, missing API key, or a failed /
unparseable LLM call all log a warning and return a 0-chunk summary.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

from .base import (
    LEARNED_DIR,
    MINING_AGGREGATES,
    LearnedChunk,
    current_patch_version,
    save_learned_file,
)

logger = logging.getLogger(__name__)

# Keep the aggregates payload sent to the LLM bounded (chars, not tokens —
# rough but sufficient to stay well inside the context window).
_MAX_AGGREGATES_CHARS = 24000

# Categories already used by knowledge_store / knowledge_chunks; anything the
# LLM invents outside this set is coerced to a sane default.
_ALLOWED_CATEGORIES = {
    "hero_overview",
    "hero_laning",
    "hero_matchup",
    "role_principles",
    "phase_objectives",
    "death_analysis",
    "vision_principles",
    "combat_mechanics",
    "economy_patterns",
    "item_coaching",
    "ability_coaching",
}
_DEFAULT_CATEGORY = "role_principles"

_TAG_RE = re.compile(r"^[a-z0-9_]+:[a-z0-9_.'\-]+(:[a-z0-9_.'\-]+)*$")

_SYSTEM_PROMPT = (
    "You are a 7000-MMR Dota 2 coach. You are given statistical aggregates "
    "mined from real pro / high-MMR matches (per-position and per-phase "
    "patterns: farm rates, death timings, item timings, ward habits, etc.). "
    "Distill them into concrete coaching insights an amateur player can act "
    "on. Compare positions against each other, spot outliers and surprising "
    "numbers, and translate them into advice — do NOT just recite the stats "
    "back. Every insight must be bilingual (Chinese and English) and "
    "specific enough that a coach could say it during a replay review."
)


def _build_user_prompt(aggregates_json: str, max_chunks: int) -> str:
    return (
        f"Here are the mined aggregates as JSON:\n\n{aggregates_json}\n\n"
        f"Produce AT MOST {max_chunks} coaching insights. Respond with STRICT "
        "JSON only, an object of the form:\n"
        "{\n"
        '  "insights": [\n'
        "    {\n"
        '      "topic": "short_snake_case_slug",\n'
        '      "zh": "中文教练建议（1-3句，具体、可执行）",\n'
        '      "en": "English coaching advice (1-3 sentences, concrete and actionable)",\n'
        '      "tags": ["role:support_4", "phase:laning"],\n'
        '      "category": "role_principles",\n'
        '      "priority": 2\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- tags use the conventions hero:<slug>, role:<carry|mid|offlane|support_4|support_5>, "
        "phase:<laning|mid_game|late_game>, mechanic:<name>, item:<slug>. Include 2-4 tags per insight.\n"
        f"- category must be one of: {', '.join(sorted(_ALLOWED_CATEGORIES))}.\n"
        "- priority: 1 = critical, 2 = important, 3 = supplementary.\n"
        "- zh and en must express the SAME insight, natively phrased in each language.\n"
        "- Skip anything generic that could be said without looking at the data.\n"
        "- Output nothing but the JSON object."
    )


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return slug[:48] or "insight"


def _normalize_tags(raw) -> list[str]:
    tags: list[str] = []
    if not isinstance(raw, list):
        return tags
    for t in raw:
        if not isinstance(t, str):
            continue
        t = t.strip().lower().replace(" ", "_")
        if _TAG_RE.match(t) and t not in tags:
            tags.append(t)
    return tags


def _validate_insights(items, max_chunks: int) -> list[dict]:
    """Filter LLM output down to well-formed insight dicts."""
    valid: list[dict] = []
    if not isinstance(items, list):
        return valid
    for item in items:
        if not isinstance(item, dict):
            continue
        zh = str(item.get("zh") or "").strip()
        en = str(item.get("en") or "").strip()
        if not zh or not en:
            continue
        category = str(item.get("category") or "").strip()
        if category not in _ALLOWED_CATEGORIES:
            category = _DEFAULT_CATEGORY
        try:
            priority = int(item.get("priority", 2))
        except (TypeError, ValueError):
            priority = 2
        if priority not in (1, 2, 3):
            priority = 2
        valid.append(
            {
                "topic": _slugify(item.get("topic") or zh[:20]),
                "zh": zh,
                "en": en,
                "tags": _normalize_tags(item.get("tags")),
                "category": category,
                "priority": priority,
            }
        )
        if len(valid) >= max_chunks:
            break
    return valid


async def _call_llm(aggregates_json: str, max_chunks: int) -> list[dict]:
    from ..evaluator import MODEL, get_shared_client

    client = get_shared_client()
    response = await client.chat.completions.create(
        model=MODEL,
        max_tokens=8000,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(aggregates_json, max_chunks)},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    # Defensive: strip code fences even though json_object should prevent them
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        return parsed.get("insights") or parsed.get("chunks") or []
    return parsed if isinstance(parsed, list) else []


def distill(langs: tuple[str, ...] = ("zh", "en"), max_chunks: int = 20) -> dict:
    """Distill Level-2 mining aggregates into bilingual coaching chunks.

    Returns ``{"chunks_written": int}``. Never raises.
    """
    aggregates_path = LEARNED_DIR / MINING_AGGREGATES
    if not aggregates_path.exists():
        logger.warning(
            "LLM distiller: aggregates file missing (%s); run match_miner first",
            aggregates_path,
        )
        return {"chunks_written": 0}

    try:
        with open(aggregates_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning("LLM distiller: cannot read aggregates: %s", exc)
        return {"chunks_written": 0}

    try:
        from ..evaluator import is_api_available
        if not is_api_available():
            logger.warning("LLM distiller: DEEPSEEK_API_KEY not set, skipping")
            return {"chunks_written": 0}
    except Exception as exc:  # openai SDK missing etc.
        logger.warning("LLM distiller: LLM client unavailable: %s", exc)
        return {"chunks_written": 0}

    aggregates_meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    aggregates_json = json.dumps(payload, ensure_ascii=False)
    if len(aggregates_json) > _MAX_AGGREGATES_CHARS:
        aggregates_json = aggregates_json[:_MAX_AGGREGATES_CHARS] + "…(truncated)"

    try:
        raw_items = asyncio.run(_call_llm(aggregates_json, max_chunks))
    except Exception as exc:
        logger.warning("LLM distiller: LLM call failed: %s", exc)
        return {"chunks_written": 0}

    insights = _validate_insights(raw_items, max_chunks)
    if not insights:
        logger.warning("LLM distiller: no valid insights in LLM response")
        return {"chunks_written": 0}

    patch = current_patch_version()
    provenance = {"aggregates_meta": aggregates_meta, "source_file": MINING_AGGREGATES}
    chunks: list[LearnedChunk] = []
    for idx, ins in enumerate(insights):
        for lang in langs:
            content = ins.get(lang)
            if not content:
                continue
            chunks.append(
                LearnedChunk(
                    chunk_id=f"learned:distilled:{idx:02d}:{ins['topic']}",
                    content=content,
                    lang=lang,
                    category=ins["category"],
                    tags=list(ins["tags"]),
                    priority=ins["priority"],
                    source="llm_distilled",
                    patch_version=patch,
                    provenance=provenance,
                )
            )

    save_learned_file(
        "llm_distilled",
        chunks,
        meta={"insight_count": len(insights), "aggregates_meta": aggregates_meta},
    )
    logger.info("LLM distiller: wrote %d chunks (%d insights)", len(chunks), len(insights))
    return {"chunks_written": len(chunks)}
