"""RAG knowledge retrieval engine with tag-based retrieval and BM25 fallback.

Provides in-memory knowledge storage and retrieval for the AI coaching system.
No external dependencies required - uses a simple character-level Chinese tokenizer
and a lightweight BM25 implementation for text search fallback.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: CJK chars individually, English words by whitespace.

    Handles Chinese text by splitting each character as a separate token,
    while grouping Latin words and numbers as whole tokens.
    """
    return [
        m.group().lower()
        for m in re.finditer(r'[\u4e00-\u9fff]|[a-zA-Z_]+|[0-9]+', text)
    ]


# ---------------------------------------------------------------------------
# SimpleBM25
# ---------------------------------------------------------------------------

class SimpleBM25:
    """Lightweight BM25 implementation with no external dependencies.

    Uses character-level tokenization for Chinese text and standard word
    tokenization for English. Suitable for small-to-medium document collections
    (hundreds to low thousands of documents).
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.tf: dict[str, dict[str, int]] = {}   # doc_id -> {term: count}
        self.df: dict[str, int] = {}               # term -> doc frequency
        self.doc_len: dict[str, int] = {}           # doc_id -> token count
        self.avg_dl: float = 0
        self.N: int = 0

    def index(self, doc_id: str, text: str) -> None:
        """Index a document for later retrieval."""
        tokens = _tokenize(text)
        self.doc_ids.append(doc_id)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        self.tf[doc_id] = tf
        self.doc_len[doc_id] = len(tokens)
        for term in set(tokens):
            self.df[term] = self.df.get(term, 0) + 1

    def finalize(self) -> None:
        """Compute corpus-level statistics. Must be called after all indexing."""
        self.N = len(self.doc_ids)
        total = sum(self.doc_len.values())
        self.avg_dl = total / self.N if self.N > 0 else 1

    def query(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return up to *top_k* ``(doc_id, score)`` pairs ranked by BM25."""
        tokens = _tokenize(text)
        query_terms = set(tokens)
        scores: dict[str, float] = {}

        for doc_id in self.doc_ids:
            score = 0.0
            dl = self.doc_len.get(doc_id, 1)
            doc_tf = self.tf.get(doc_id, {})
            for term in query_terms:
                if term not in doc_tf:
                    continue
                tf_val = doc_tf[term]
                df_val = self.df.get(term, 0)
                idf = math.log((self.N - df_val + 0.5) / (df_val + 0.5) + 1)
                tf_norm = (tf_val * (self.k1 + 1)) / (
                    tf_val + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                )
                score += idf * tf_norm
            if score > 0:
                scores[doc_id] = score

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]


# ---------------------------------------------------------------------------
# KnowledgeChunk
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeChunk:
    """A single piece of coaching knowledge that can be retrieved by the RAG system.

    Attributes:
        chunk_id: Unique identifier, e.g. ``"hero:anti_mage:laning"``.
        content: Chinese coaching text.
        category: One of ``hero_overview``, ``hero_laning``, ``hero_matchup``,
            ``role_principles``, ``phase_objectives``, ``death_analysis``,
            ``vision_principles``, ``combat_mechanics``, ``economy_patterns``,
            ``item_coaching``, ``ability_coaching``.
        tags: Set of tags for retrieval, e.g. ``{"hero:lion", "role:support_4"}``.
        priority: 1 = critical, 2 = important, 3 = supplementary.
        token_estimate: Approximate token count (auto-estimated from content).
        source: ``"builtin"``, ``"user"``, or ``"json"``.
        patch_version: Dota 2 patch version this knowledge applies to.
    """

    chunk_id: str
    content: str
    category: str = ""
    tags: set[str] = field(default_factory=set)
    priority: int = 2
    token_estimate: int = 0
    source: str = "builtin"
    patch_version: str = "7.37"

    def __post_init__(self) -> None:
        # Auto-derive category from chunk_id prefix if not provided
        if not self.category and ":" in self.chunk_id:
            self.category = self.chunk_id.split(":")[0]
        # Convert list tags to set
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
        # Auto-estimate tokens from content length (rough: 1 CJK char ≈ 0.5 token)
        if self.token_estimate <= 0 and self.content:
            self.token_estimate = max(1, len(self.content) // 2)


# ---------------------------------------------------------------------------
# KnowledgeStore
# ---------------------------------------------------------------------------

class KnowledgeStore:
    """In-memory knowledge store with tag-based retrieval and BM25 fallback.

    The primary retrieval path scores chunks by tag overlap with a query
    derived from a :class:`DecisionContext`.  When fewer than 3 results are
    found via tags, a lightweight BM25 text-search supplements the results.
    A token-budget aware selector ensures the final result set stays within
    LLM context limits and maintains category diversity.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, KnowledgeChunk] = {}   # chunk_id -> chunk
        self._tag_index: dict[str, set[str]] = {}       # tag -> set of chunk_ids
        self._bm25: SimpleBM25 | None = None
        self._bm25_dirty: bool = True

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_chunk(self, chunk: KnowledgeChunk) -> None:
        """Add a chunk to the store and update the tag index.

        If ``token_estimate`` is not set (or <= 0), it is auto-estimated from
        the content length using a rough heuristic of ~1.5 chars per token for
        Chinese text (i.e. ``len(content) // 2``).
        """
        if chunk.token_estimate <= 0:
            chunk.token_estimate = max(1, len(chunk.content) // 2)
        self._chunks[chunk.chunk_id] = chunk
        for tag in chunk.tags:
            self._tag_index.setdefault(tag, set()).add(chunk.chunk_id)
        self._bm25_dirty = True

    # ------------------------------------------------------------------
    # Retrieval (public)
    # ------------------------------------------------------------------

    def retrieve(self, ctx, budget: int = 3000) -> list[KnowledgeChunk]:
        """Retrieve relevant chunks for a :class:`DecisionContext` within token budget.

        Steps:
        1. Build query tags from the context.
        2. Score all chunks by tag overlap.
        3. If fewer than 3 tag-matched results, supplement with BM25 (discounted).
        4. Select top results within *budget* tokens, with category diversity.
        """
        query_tags = self._build_query_tags(ctx)
        scored = self._tag_retrieve(query_tags)

        # If too few results from tags, supplement with BM25
        if len(scored) < 3:
            bm25_query = self._build_bm25_query(ctx)
            bm25_results = self._bm25_retrieve(bm25_query, top_k=5)
            existing_ids = {c.chunk_id for c, _ in scored}
            for chunk, score in bm25_results:
                if chunk.chunk_id not in existing_ids:
                    scored.append((chunk, score * 0.5))  # discount BM25 results

        return self._select_within_budget(scored, budget)

    # ------------------------------------------------------------------
    # Query construction (private)
    # ------------------------------------------------------------------

    def _build_query_tags(self, ctx) -> set[str]:
        """Generate query tags from a :class:`DecisionContext`."""
        from .models import DecisionContext  # noqa: F401 — type check only

        tags: set[str] = set()

        # Hero
        hero_slug = (
            ctx.player_state.hero_name.lower()
            .replace(" ", "_")
            .replace("'", "")
        )
        tags.add(f"hero:{hero_slug}")

        # Phase
        if ctx.game_phase:
            tags.add(f"phase:{ctx.game_phase}")

        # Situation
        tags.add(f"situation:{ctx.decision_point.point_type}")

        # Enemy heroes
        for enemy in ctx.enemy_states:
            e_slug = (
                enemy.hero_name.lower()
                .replace(" ", "_")
                .replace("'", "")
            )
            tags.add(f"hero:{e_slug}")
            tags.add(f"matchup:{hero_slug}:{e_slug}")

        # Items
        for item in ctx.player_state.items:
            item_slug = (
                item.lower()
                .replace(" ", "_")
                .replace("'", "")
            )
            tags.add(f"item:{item_slug}")

        # Mechanic tags based on situation type
        if ctx.decision_point.point_type in ("death", "overextend"):
            tags.add("mechanic:vision")
            tags.add("mechanic:positioning")
        if ctx.decision_point.point_type in ("bad_trade", "missed_cs"):
            tags.add("mechanic:cs")
            tags.add("mechanic:trading")
        if ctx.decision_point.point_type in ("fight_engage", "good_play"):
            tags.add("mechanic:teamfight")

        # Always include hero:all so universal tips are retrieved
        tags.add("hero:all")

        return tags

    def _build_bm25_query(self, ctx) -> str:
        """Build a text query for BM25 fallback from a :class:`DecisionContext`."""
        parts = [ctx.player_state.hero_name, ctx.decision_point.point_type]
        if ctx.decision_point.brief:
            parts.append(ctx.decision_point.brief)
        if ctx.game_phase:
            parts.append(ctx.game_phase)
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Tag retrieval (private)
    # ------------------------------------------------------------------

    def _tag_retrieve(
        self, query_tags: set[str]
    ) -> list[tuple[KnowledgeChunk, float]]:
        """Score all chunks by tag overlap with *query_tags*.

        Scoring formula:
        - Raw score = number of matching tags.
        - Normalized by ``sqrt(chunk_tag_count)`` to prefer focused chunks.
        - Multiplied by a priority weight (critical=1.5, important=1.0,
          supplementary=0.7).
        """
        chunk_scores: dict[str, float] = {}
        for tag in query_tags:
            for chunk_id in self._tag_index.get(tag, set()):
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + 1

        results: list[tuple[KnowledgeChunk, float]] = []
        for chunk_id, raw_score in chunk_scores.items():
            chunk = self._chunks[chunk_id]
            # Normalize by chunk tag count to prefer focused chunks
            tag_count = len(chunk.tags) or 1
            score = raw_score / (tag_count ** 0.5)
            # Chunks tagged hero:all matched via the query's hero:all tag
            # get a slight discount so hero-specific chunks rank higher
            if "hero:all" in chunk.tags and raw_score == 1:
                score *= 0.6
            # Priority boost
            priority_weight = {1: 1.5, 2: 1.0, 3: 0.7}.get(chunk.priority, 1.0)
            score *= priority_weight
            results.append((chunk, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ------------------------------------------------------------------
    # BM25 retrieval (private)
    # ------------------------------------------------------------------

    def _bm25_retrieve(
        self, query: str, top_k: int = 5
    ) -> list[tuple[KnowledgeChunk, float]]:
        """BM25 text search fallback.

        Lazily rebuilds the BM25 index when the store has been mutated since
        the last build (tracked via ``_bm25_dirty``).
        """
        if self._bm25_dirty or self._bm25 is None:
            self._bm25 = SimpleBM25()
            for chunk_id, chunk in self._chunks.items():
                self._bm25.index(chunk_id, chunk.content)
            self._bm25.finalize()
            self._bm25_dirty = False

        results = self._bm25.query(query, top_k=top_k)
        return [
            (self._chunks[cid], score)
            for cid, score in results
            if cid in self._chunks
        ]

    # ------------------------------------------------------------------
    # Budget selection (private)
    # ------------------------------------------------------------------

    def _select_within_budget(
        self,
        scored: list[tuple[KnowledgeChunk, float]],
        budget: int,
    ) -> list[KnowledgeChunk]:
        """Select top chunks within token *budget* with category diversity.

        Enforces a maximum of 3 chunks per category to ensure the LLM receives
        a well-rounded set of coaching knowledge.
        """
        scored.sort(key=lambda x: x[1], reverse=True)
        selected: list[KnowledgeChunk] = []
        used_tokens = 0
        category_counts: dict[str, int] = {}

        for chunk, score in scored:
            if score <= 0:
                break
            # Category diversity: max 3 chunks per category
            cat = chunk.category
            if category_counts.get(cat, 0) >= 3:
                continue
            if used_tokens + chunk.token_estimate > budget:
                continue
            selected.append(chunk)
            used_tokens += chunk.token_estimate
            category_counts[cat] = category_counts.get(cat, 0) + 1

        return selected

    # ------------------------------------------------------------------
    # Bulk loading
    # ------------------------------------------------------------------

    def load_builtin(self, lang: str = "zh") -> None:
        """Load built-in knowledge chunks from the ``knowledge_chunks`` module.

        Silently skips if the module is not yet available.
        """
        try:
            from .knowledge_chunks import get_all_builtin_chunks
            for chunk in get_all_builtin_chunks(lang=lang):
                self.add_chunk(chunk)
        except ImportError:
            pass  # knowledge_chunks not yet available

    def load_json(self, path: str) -> None:
        """Load chunks from a JSON file.

        Expected format is a JSON array of objects, each with at least
        ``chunk_id``, ``content``, and optionally ``category``, ``tags``,
        ``priority``, ``token_estimate``, ``source``, ``patch_version``.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            chunk = KnowledgeChunk(
                chunk_id=item["chunk_id"],
                content=item["content"],
                category=item.get("category", "general"),
                tags=set(item.get("tags", [])),
                priority=item.get("priority", 2),
                token_estimate=item.get("token_estimate", 0),
                source=item.get("source", "json"),
                patch_version=item.get("patch_version", "7.37"),
            )
            self.add_chunk(chunk)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return store statistics (total chunks and per-category counts)."""
        cats: dict[str, int] = {}
        for chunk in self._chunks.values():
            cats[chunk.category] = cats.get(chunk.category, 0) + 1
        return {
            "total_chunks": len(self._chunks),
            "categories": cats,
        }


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "KnowledgeChunk",
    "KnowledgeStore",
    "SimpleBM25",
]
