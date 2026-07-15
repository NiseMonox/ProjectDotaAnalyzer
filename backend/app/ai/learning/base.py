"""Shared contract for the self-learning pipeline.

The learning package teaches the AI coach automatically instead of relying on
hand-written tips. Four learners write KnowledgeChunk-compatible JSON into
``backend/data/learned/``, which knowledge_base loads into the RAG store
alongside the builtin / gamedata / editorial layers:

- Level 1  stats_learner.learn_stats(langs, heroes_limit=None) -> summary dict
    OpenDota aggregate stats (benchmarks percentiles, item timings, matchups)
    rendered into bilingual template chunks. Zero LLM cost.
- Level 2  match_miner.learn_from_matches(source, limit, langs) -> summary dict
    Pro / high-MMR matches pulled through OpenDotaParser and this project's
    own analyzers (positions, phases, decision detection), aggregated into
    per-position / per-phase pattern chunks. Also persists raw aggregates to
    ``LEARNED_DIR / MINING_AGGREGATES`` for Level 3. Zero LLM cost.
- Level 3  llm_distiller.distill(langs, max_chunks) -> summary dict
    Feeds Level 2 aggregates to the LLM (shared DeepSeek client from
    ai.evaluator) and distills natural-language coaching tips.
- Level 4  video_learner.learn_from_videos(langs, max_videos) -> summary dict
    Subtitle transcripts of coaching / replay-analysis videos (YouTube and
    Bilibili via the yt-dlp CLI, Bilibili discovery via its search API),
    distilled by the LLM into sourced tips.

Every learner returns ``{"chunks_written": int, ...}`` and must degrade
gracefully (log a warning and return 0-chunk summary) when its external
dependency is unavailable. All chunk text is bilingual: one chunk object per
language, tagged with the conventions used by knowledge_store (e.g.
``hero:pudge``, ``role:support_4``, ``phase:laning``).
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# backend/data/learned — sibling of gamedata/ and editorial/.
# config.py owns the path (single source of truth); it also mkdirs it.
from ...config import LEARNED_DATA_DIR as LEARNED_DIR  # noqa: E402

# Raw Level-2 aggregates consumed by the Level-3 distiller (not a chunk file)
MINING_AGGREGATES = "_mining_aggregates.json"

# User-editable video source list (created with defaults by video_learner)
VIDEO_SOURCES_FILE = Path(__file__).resolve().parents[3] / "data" / "learning_sources.json"

OPENDOTA_BASE = "https://api.opendota.com/api"
# Free tier: 60 calls/min, 2000/day. Keep a polite floor between calls.
_MIN_CALL_INTERVAL = 1.1
_last_call_at = 0.0


@dataclass
class LearnedChunk:
    """One learned knowledge chunk, serialized 1:1 into learned/*.json.

    Field names match knowledge_store.KnowledgeChunk so the loader can pass
    the dict straight through; ``lang``/``provenance``/``learned_at`` are
    learning-layer extras the loader strips.
    """

    chunk_id: str            # e.g. "learned:item_timing:pudge:aether_lens"
    content: str             # coaching text in ONE language
    lang: str                # "zh" | "en"
    category: str            # reuse existing categories where possible
    tags: list[str] = field(default_factory=list)
    priority: int = 2        # 1 critical / 2 important / 3 supplementary
    source: str = "learned"  # "opendota_stats" | "match_mining" | "llm_distilled" | "video:youtube" | "video:bilibili"
    patch_version: str = ""
    provenance: dict = field(default_factory=dict)  # urls / match_ids / sample sizes
    learned_at: str = ""

    def __post_init__(self) -> None:
        if not self.learned_at:
            self.learned_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def opendota_get(path: str, params: dict | None = None, timeout: int = 30):
    """Rate-limited GET against the OpenDota API. Returns parsed JSON.

    Raises URLError/HTTPError on failure — callers decide how to degrade.
    """
    global _last_call_at
    wait = _MIN_CALL_INTERVAL - (time.monotonic() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    url = f"{OPENDOTA_BASE}/{path.lstrip('/')}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "DotaAnalyzer-Learning/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    _last_call_at = time.monotonic()
    return data


def save_learned_file(name: str, chunks: list[LearnedChunk], meta: dict | None = None) -> Path:
    """Write chunks (all languages together) to LEARNED_DIR/<name>.json.

    Overwrites the previous run of the same learner — learned files are
    regenerated wholesale, never appended, so stale patch data can't linger.
    """
    path = LEARNED_DIR / f"{name}.json"
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "chunk_count": len(chunks),
            **(meta or {}),
        },
        "chunks": [asdict(c) for c in chunks],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    logger.info("Learned file written: %s (%d chunks)", path.name, len(chunks))
    return path


def load_learned_chunks(lang: str) -> list[dict]:
    """Load every learned chunk for one language across all learner files.

    Files whose name starts with "_" (raw aggregates) are skipped. Malformed
    files are logged and skipped — a bad learn run must never break coaching.
    """
    chunks: list[dict] = []
    for path in sorted(LEARNED_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for c in payload.get("chunks", []):
                if c.get("lang") == lang and c.get("content"):
                    chunks.append(c)
        except Exception as exc:
            logger.warning("Skipping malformed learned file %s: %s", path.name, exc)
    return chunks


def current_patch_version() -> str:
    """Best-effort current patch string from data_manager (may be "")."""
    try:
        from ..data_manager import get_patch_version
        return get_patch_version() or ""
    except Exception:
        return ""
