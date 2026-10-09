"""Shared contract for the self-learning pipeline.

The learning package teaches the AI coach automatically instead of relying on
hand-written tips. Four learners write KnowledgeChunk-compatible JSON into
``backend/data/learned/``, which knowledge_base loads into the RAG store
alongside the builtin / editorial layers:

- Level 1  stats_learner.learn_stats(langs, heroes_limit=None, output_dir=None)
    OpenDota aggregate stats (benchmark percentiles, item timings, matchup
    win rates) rendered into bilingual, clearly labelled statistics chunks.
    Zero LLM cost.
- Level 2  match_miner.learn_from_matches(source, limit, langs, output_dir=None)
    Pro / high-MMR matches pulled through OpenDotaParser and this project's
    position derivation, aggregated into per-position / per-hero pattern
    chunks (minimum sample sizes enforced). Also persists raw aggregates to
    ``<output_dir> / MINING_AGGREGATES`` for Level 3. Zero LLM cost.
- Level 3  llm_distiller.distill(langs, max_chunks, output_dir=None)
    Feeds Level 2 aggregates to the LLM and distills coaching tips.
- Level 4  video_learner.learn_from_videos(langs, max_videos, output_dir=None)
    Subtitle transcripts of coaching videos (YouTube / Bilibili via yt-dlp),
    distilled by the LLM into sourced, sanitised tips.

Every learner returns ``{"chunks_written": int, ...}`` and never raises. A
summary carries ``"error": <code>`` when the level failed (network, missing
prerequisite — the pipeline retries it next time) or ``"skipped": <code>``
when it was skipped by configuration (no API key, no yt-dlp).

Quick / smoke runs write into :data:`SMOKE_DIR` (a sub-directory the chunk
loader ignores), so they never clobber full outputs or the run manifest.

All LLM calls go through :func:`llm_json_completion`, which creates and
closes a dedicated client inside the running event loop (learners run in a
worker thread under ``asyncio.run``; a client must never be shared across
loops). Third-party text (video transcripts, LLM output) is sanitised with
:func:`sanitize_learned_text` before it is stored, and again when loaded.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# backend/data/learned — sibling of gamedata/ and editorial/.
# config.py owns the path (single source of truth); it also mkdirs it.
from ...config import DATA_DIR, LEARNED_DATA_DIR as LEARNED_DIR  # noqa: E402

# Quick / smoke-run outputs (ignored by load_learned_chunks: not top-level).
SMOKE_DIR = LEARNED_DIR / "_smoke"

# Raw Level-2 aggregates consumed by the Level-3 distiller (not a chunk file)
MINING_AGGREGATES = "_mining_aggregates.json"

# User-editable video source list (created with defaults by video_learner)
VIDEO_SOURCES_FILE = DATA_DIR / "learning_sources.json"

OPENDOTA_BASE = "https://api.opendota.com/api"
# Free tier: 60 calls/min, 2000/day. Keep a polite floor between calls.
_MIN_CALL_INTERVAL = 1.1
_last_call_at = 0.0
_call_lock = threading.Lock()

# Hard caps for stored learned text (characters).
MAX_TIP_CHARS = 500           # one LLM-distilled / video tip in one language
MAX_LEARNED_CHUNK_CHARS = 900  # any learned chunk content when loaded

# LLM settings (same env knobs as the coaching pipeline).
_DEFAULT_BASE_URL = "https://api.deepseek.com"
_DEFAULT_MODEL = "deepseek-v4-flash"
_LLM_TIMEOUT_S = 120.0


def output_dir_for(quick: bool) -> Path:
    """Directory a learner writes to: :data:`SMOKE_DIR` for quick runs."""
    path = SMOKE_DIR if quick else LEARNED_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class LearnedChunk:
    """One learned knowledge chunk, serialized 1:1 into learned/*.json.

    Field names match knowledge_store.KnowledgeChunk so the loader can pass
    the dict straight through; ``lang``/``provenance``/``learned_at`` are
    learning-layer extras the loader strips.
    """

    chunk_id: str            # e.g. "learned:item_timing:pudge:blink"
    content: str             # coaching text in ONE language
    lang: str                # "zh" | "en"
    category: str            # reuse existing categories where possible
    tags: list[str] = field(default_factory=list)
    priority: int = 2        # 1 critical / 2 important / 3 supplementary
    source: str = "learned"  # "opendota_stats" | "match_mining" | "llm_distilled" | "video:youtube" | "video:bilibili"
    patch_version: str = ""
    provenance: dict = field(default_factory=dict)  # urls / match counts / sample sizes
    learned_at: str = ""

    def __post_init__(self) -> None:
        if not self.learned_at:
            self.learned_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# OpenDota
# ---------------------------------------------------------------------------

def opendota_get(path: str, params: dict | None = None, timeout: int = 30):
    """Rate-limited GET against the OpenDota API. Returns parsed JSON.

    Uses ``OPENDOTA_API_KEY`` when set. Raises URLError/HTTPError on
    failure — callers decide how to degrade.
    """
    global _last_call_at
    query = dict(params or {})
    api_key = os.environ.get("OPENDOTA_API_KEY", "").strip()
    if api_key:
        query["api_key"] = api_key
    url = f"{OPENDOTA_BASE}/{path.lstrip('/')}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    with _call_lock:
        wait = _MIN_CALL_INTERVAL - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()
    req = urllib.request.Request(url, headers={"User-Agent": "DotaAnalyzer-Learning/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Learned files
# ---------------------------------------------------------------------------

def write_json_atomic(path: Path, payload: Any, indent: int | None = 1) -> None:
    """Write JSON via a temp file + rename so readers never see partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=indent)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_learned_file(
    name: str,
    chunks: list[LearnedChunk],
    meta: dict | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Write chunks (all languages together) to ``<output_dir>/<name>.json``.

    Replaces the previous file of the same learner atomically. Learners that
    refresh only part of their data merge the untouched previous chunks in
    themselves (see :func:`load_learned_file`).
    """
    path = (output_dir or LEARNED_DIR) / f"{name}.json"
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "chunk_count": len(chunks),
            **(meta or {}),
        },
        "chunks": [asdict(c) for c in chunks],
    }
    write_json_atomic(path, payload)
    logger.info("Learned file written: %s (%d chunks)", path, len(chunks))
    return path


def load_learned_file(name: str, output_dir: Path | None = None) -> list[LearnedChunk]:
    """Read back a learner's previous chunks ([] when missing / malformed)."""
    path = (output_dir or LEARNED_DIR) / f"{name}.json"
    if not path.exists():
        return []
    names = {f.name for f in dataclass_fields(LearnedChunk)}
    prior: list[LearnedChunk] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for c in payload.get("chunks", []):
            if isinstance(c, dict) and c.get("chunk_id") and c.get("content") and c.get("lang"):
                prior.append(LearnedChunk(**{k: v for k, v in c.items() if k in names}))
    except Exception as exc:
        logger.warning("Cannot read previous learned file %s: %s", path.name, exc)
        return []
    return prior


def load_learned_chunks(lang: str) -> list[dict]:
    """Load every learned chunk for one language across all learner files.

    Only top-level ``*.json`` files are read; names starting with "_" (raw
    aggregates, registries, the manifest) and the smoke directory are
    skipped. Each chunk is sanitised again (third-party text must never
    reach a prompt unchecked); suspicious or empty chunks are dropped.
    Malformed files are logged and skipped — a bad learn run must never
    break coaching.
    """
    chunks: list[dict] = []
    dropped = 0
    for path in sorted(LEARNED_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for c in payload.get("chunks", []) if isinstance(payload, dict) else []:
                if not isinstance(c, dict) or c.get("lang") != lang or not c.get("chunk_id"):
                    continue
                content = sanitize_learned_text(c.get("content"), MAX_LEARNED_CHUNK_CHARS)
                if not content:
                    dropped += 1
                    continue
                item = dict(c)
                item["content"] = content
                tags = item.get("tags")
                item["tags"] = [t for t in tags if isinstance(t, str)] if isinstance(tags, list) else []
                chunks.append(item)
        except Exception as exc:
            logger.warning("Skipping malformed learned file %s: %s", path.name, exc)
    if dropped:
        logger.warning("Dropped %d learned chunk(s) that failed sanitisation (%s)", dropped, lang)
    return chunks


# ---------------------------------------------------------------------------
# Sanitisation of third-party / LLM text (prompt-injection defence)
# ---------------------------------------------------------------------------

_CONTROL_RE = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f"
    "\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060-\u2064\ufeff]"
)
_URL_RE = re.compile(r"(?i)\b(?:https?://|ftp://|www\.)\S+")
_MD_LINK_RE = re.compile(r"\[([^\]]{0,200})\]\([^)]*\)")
_HTML_RE = re.compile(r"<[^<>]{0,300}>")
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]{0,40}\|>|\[/?(?:INST|SYS)\]", re.IGNORECASE)
_FENCE_RE = re.compile(r"`{3,}[a-zA-Z]*")
_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_SPACES_RE = re.compile(r"[ \t]{2,}")

_INSTRUCTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(instruction|instructions|prompt|prompts|rules?|above|previous|prior|context|everything)\b",
        r"\b(system|developer)\s+(prompt|message|instruction)s?\b",
        r"\byou\s+are\s+(now\s+)?(an?\s+|the\s+)?(ai|assistant|language model|chatbot|gpt|llm)\b",
        r"\b(pretend|role-?play)\s+(to\s+be|as)\b",
        r"\b(respond|reply|answer|output)\s+only\b",
        r"\bnew\s+instructions?\b",
        r"\b(jailbreak|prompt\s+injection)\b",
        r"(?m)^\s*(assistant|system|user)\s*:",
        r"(忽略|无视|忘记|忘掉|不要理会).{0,15}(指令|提示|规则|要求|以上|之前|前面|上述)",
        r"(系统|开发者)(提示|指令|消息)",
        r"你(现在)?(是|扮演|作为).{0,10}(助手|模型|AI|人工智能|机器人|GPT)",
        r"(只|仅)(输出|回复|回答)",
        r"新的?指令",
        r"<\|[^|]{0,40}\|>",
        r"\[/?(?:INST|SYS)\]",
    )
]


def looks_like_instruction(text: str) -> bool:
    """Heuristic: does *text* address the model instead of the player?"""
    return any(p.search(text or "") for p in _INSTRUCTION_PATTERNS)


def clean_text(text: Any, max_chars: int) -> str:
    """Strip control chars, markup, URLs and headings; collapse whitespace; cap length."""
    text = str(text or "")
    text = _CONTROL_RE.sub(" ", text)
    text = _SPECIAL_TOKEN_RE.sub(" ", text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _FENCE_RE.sub(" ", text)
    text = _HEADING_RE.sub("", text)
    text = text.replace("\r", "\n")
    lines = [_SPACES_RE.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 1)].rstrip() + "…"
    return text


def sanitize_learned_text(text: Any, max_chars: int = MAX_TIP_CHARS) -> str | None:
    """Cleaned text, or None when empty or when it looks like a prompt injection."""
    cleaned = clean_text(text, max_chars)
    if not cleaned:
        return None
    if looks_like_instruction(cleaned) or looks_like_instruction(str(text or "")):
        logger.warning("Dropping learned text that looks like instructions: %r", cleaned[:80])
        return None
    return cleaned


# ---------------------------------------------------------------------------
# Patch + tags
# ---------------------------------------------------------------------------

def current_patch_version() -> str:
    """Patch of the last successful game-data fetch ("" when unknown).

    Learned chunks are stamped with it; an unknown patch is recorded as ""
    rather than guessed, so it never wrongly marks chunks stale.
    """
    try:
        from ..data_manager import get_fetched_patch
        return get_fetched_patch() or ""
    except Exception:
        return ""


# Tag namespaces a learner (or the LLM) may emit.
_TAG_NAMESPACES = ("hero", "role", "phase", "mechanic", "item", "matchup", "situation")


def canonical_tags(tags: Any) -> list[str]:
    """Normalise tags with the knowledge layer's canonical rules (sorted, deduped).

    Only namespaced tags of the known kinds survive (``hero:``, ``role:``,
    ``phase:``, ``mechanic:``, ``item:``, ``matchup:``, ``situation:``).
    """
    from ..knowledge_store import normalize_tag
    out: set[str] = set()
    for tag in tags if isinstance(tags, (list, tuple, set)) else []:
        norm = normalize_tag(tag)
        if norm and norm.split(":", 1)[0] in _TAG_NAMESPACES and ":" in norm:
            out.add(norm)
    return sorted(out)


# ---------------------------------------------------------------------------
# LLM (dedicated client per event loop)
# ---------------------------------------------------------------------------

def llm_available() -> bool:
    """An API key is configured and the openai SDK is importable."""
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        return False
    return importlib.util.find_spec("openai") is not None


def llm_model() -> str:
    return os.environ.get("AI_COACH_MODEL", "").strip() or _DEFAULT_MODEL


_FENCE_STRIP_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


async def llm_json_completion(system: str, user: str, max_tokens: int,
                              temperature: float = 0.3) -> Any:
    """One JSON-mode chat completion with a client bound to the running loop.

    The client is created and closed inside this coroutine, so it is safe
    under ``asyncio.run`` in any thread. Raises on API / JSON errors.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY", "").strip(),
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "").strip() or _DEFAULT_BASE_URL,
        timeout=_LLM_TIMEOUT_S,
        max_retries=1,
    )
    try:
        response = await client.chat.completions.create(
            model=llm_model(),
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
    finally:
        await client.close()
    text = (response.choices[0].message.content or "").strip()
    return json.loads(_FENCE_STRIP_RE.sub("", text))


__all__ = [
    "LEARNED_DIR",
    "SMOKE_DIR",
    "MINING_AGGREGATES",
    "VIDEO_SOURCES_FILE",
    "LearnedChunk",
    "opendota_get",
    "output_dir_for",
    "save_learned_file",
    "load_learned_file",
    "load_learned_chunks",
    "write_json_atomic",
    "clean_text",
    "sanitize_learned_text",
    "looks_like_instruction",
    "current_patch_version",
    "canonical_tags",
    "llm_available",
    "llm_model",
    "llm_json_completion",
]
