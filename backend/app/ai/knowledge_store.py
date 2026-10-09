"""RAG knowledge retrieval engine: weighted tag matching + BM25 lexical relevance.

In-memory knowledge storage and retrieval for the AI coaching system, with no
external dependencies.

Ranking (deterministic, see :meth:`KnowledgeStore.rank`):

1. The decision context is turned into *weighted* query tags: specific tags
   (the exact matchup, the player's hero, the killer, the situation) weigh far
   more than generic ones (game phase, ``hero:all``). Each matched tag is
   further weighted by its inverse document frequency, so tags carried by
   hundreds of chunks contribute little.
2. A BM25 lexical score over the chunk text (CJK bigrams + Latin words)
   rewards chunks that actually talk about the heroes / situation.
3. ``0.7 * sqrt(tag score) + 0.3 * lexical score`` (both normalised to the
   best candidate), times priority, times a penalty for learned chunks that
   were produced on a different patch. Ties break on ``chunk_id``.
4. Chunks are picked best-first within ONE token budget (estimated with
   :func:`estimate_tokens`), at most 3 per category and one per dedupe key.

Tags use one canonical normalisation (:func:`normalize_tag`) for both chunk
producers and query building: hero tags are internal hero names
(``hero:queenofpain``), item tags internal item keys (``item:bfury``) — any
display name / slug / npc name is accepted on input.
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"[　-〿㐀-䶿一-鿿＀-￯]")


def estimate_tokens(text: str) -> int:
    """Rough LLM token count for mixed Chinese / English text.

    Uses DeepSeek's published ratios (~0.6 token per CJK character, ~0.3 per
    other character), rounded up. Deliberately a slight over-estimate.
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return max(1, math.ceil(cjk * 0.6 + other * 0.3))


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[一-鿿]+|[a-z][a-z0-9']*")
_STOPWORDS = frozenset(
    "a an and are as at be but by can do for from has have he her his if in into is it its "
    "of on or so than that the their them then there these they this to was were when which "
    "while who will with you your not no more most very also just only".split()
)


def _tokenize(text: str) -> list[str]:
    """Tokenize mixed text: Latin words (stopwords dropped) + CJK bigrams.

    A run of CJK characters yields its overlapping bigrams (a single
    character yields itself), which works far better for Chinese BM25 than
    single characters.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer((text or "").lower()):
        tok = match.group()
        if "一" <= tok[0] <= "鿿":
            if len(tok) == 1:
                tokens.append(tok)
            else:
                tokens.extend(tok[i:i + 2] for i in range(len(tok) - 1))
        else:
            tok = tok.strip("'")
            if len(tok) >= 2 and tok not in _STOPWORDS:
                tokens.append(tok)
    return tokens


# ---------------------------------------------------------------------------
# SimpleBM25
# ---------------------------------------------------------------------------

class SimpleBM25:
    """Lightweight BM25 with an inverted index (no external dependencies)."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.postings: dict[str, dict[str, int]] = {}  # term -> {doc_id: tf}
        self.doc_len: dict[str, int] = {}
        self.avg_dl: float = 1.0
        self.N: int = 0

    def index(self, doc_id: str, text: str) -> None:
        """Index a document for later retrieval."""
        tokens = _tokenize(text)
        self.doc_ids.append(doc_id)
        self.doc_len[doc_id] = len(tokens)
        for term in tokens:
            bucket = self.postings.setdefault(term, {})
            bucket[doc_id] = bucket.get(doc_id, 0) + 1

    def finalize(self) -> None:
        """Compute corpus-level statistics. Must be called after all indexing."""
        self.N = len(self.doc_ids)
        total = sum(self.doc_len.values())
        self.avg_dl = (total / self.N) if self.N else 1.0
        if self.avg_dl <= 0:
            self.avg_dl = 1.0

    def scores(self, text: str) -> dict[str, float]:
        """BM25 score for every document sharing at least one query term."""
        out: dict[str, float] = {}
        for term in sorted(set(_tokenize(text))):
            bucket = self.postings.get(term)
            if not bucket:
                continue
            df = len(bucket)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            for doc_id, tf in bucket.items():
                dl = self.doc_len.get(doc_id, 0)
                norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
                out[doc_id] = out.get(doc_id, 0.0) + idf * norm
        return out

    def query(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Return up to *top_k* ``(doc_id, score)`` pairs ranked by BM25."""
        ranked = sorted(self.scores(text).items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]


# ---------------------------------------------------------------------------
# Tag normalisation (single source of truth for producers AND queries)
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")

_ROLE_ALIASES = {
    "carry": "carry", "pos1": "carry", "position_1": "carry", "position1": "carry",
    "safelane": "carry", "safe_lane": "carry", "hard_carry": "carry",
    "mid": "mid", "pos2": "mid", "position_2": "mid", "position2": "mid",
    "midlane": "mid", "mid_lane": "mid", "midlaner": "mid",
    "offlane": "offlane", "pos3": "offlane", "position_3": "offlane", "position3": "offlane",
    "offlaner": "offlane", "off_lane": "offlane",
    "support_4": "support_4", "pos4": "support_4", "position_4": "support_4",
    "position4": "support_4", "soft_support": "support_4", "roamer": "support_4",
    "roaming_support": "support_4", "support4": "support_4",
    "support_5": "support_5", "pos5": "support_5", "position_5": "support_5",
    "position5": "support_5", "hard_support": "support_5", "support": "support_5",
    "support5": "support_5",
}
_PHASE_ALIASES = {
    "pregame": "pregame", "pre_game": "pregame", "draft": "pregame",
    "laning": "laning", "lane": "laning", "early": "laning", "early_game": "laning",
    "earlygame": "laning", "laning_phase": "laning",
    "mid_game": "mid_game", "midgame": "mid_game", "mid": "mid_game", "middle_game": "mid_game",
    "late_game": "late_game", "lategame": "late_game", "late": "late_game",
}


def _slug(text: Any) -> str:
    return _SLUG_RE.sub("_", str(text or "").strip().lower()).strip("_")


def _hero_slug(name: Any) -> str:
    try:
        from .data_manager import hero_slug
        return hero_slug(name)
    except Exception:  # pragma: no cover - data layer unavailable
        return _SLUG_RE.sub("", str(name or "").lower())


def _item_slug(name: Any) -> str:
    try:
        from .data_manager import item_key
        return item_key(name)
    except Exception:  # pragma: no cover - data layer unavailable
        return _SLUG_RE.sub("", str(name or "").lower())


def hero_tag(name: Any) -> str:
    """``hero:<internal name>`` for any hero identifier (``hero:all`` kept)."""
    return f"hero:{_hero_slug(name)}"


def item_tag(name: Any) -> str:
    """``item:<internal key>`` for any item identifier."""
    return f"item:{_item_slug(name)}"


def matchup_tag(hero: Any, opponent: Any) -> str:
    """``matchup:<hero>:<opponent>`` (directional: tips are from *hero*'s side)."""
    return f"matchup:{_hero_slug(hero)}:{_hero_slug(opponent)}"


def normalize_tag(tag: Any) -> str | None:
    """Canonical form of a tag, or None when it is unusable.

    ``hero:Anti-Mage`` / ``hero:anti_mage`` -> ``hero:antimage``;
    ``item:Battle Fury`` -> ``item:bfury``; ``role:pos5`` -> ``role:support_5``;
    ``phase:midgame`` -> ``phase:mid_game``; other namespaces are slugged.
    Bare tags without a namespace (e.g. ``reference``) are slugged as-is.
    """
    if not isinstance(tag, str):
        return None
    text = tag.strip()
    if not text:
        return None
    namespace, sep, rest = text.partition(":")
    namespace = _slug(namespace)
    if not namespace:
        return None
    if not sep:
        return namespace
    rest = rest.strip()
    if not rest:
        return None
    if namespace == "hero":
        value = _hero_slug(rest)
        return f"hero:{value}" if value else None
    if namespace == "matchup":
        a, _, b = rest.partition(":")
        a_slug, b_slug = _hero_slug(a), _hero_slug(b)
        return f"matchup:{a_slug}:{b_slug}" if a_slug and b_slug else None
    if namespace == "item":
        value = _item_slug(rest)
        return f"item:{value}" if value else None
    value = _slug(rest.replace(":", "_"))
    if not value:
        return None
    if namespace == "role":
        value = _ROLE_ALIASES.get(value, value)
    elif namespace == "phase":
        value = _PHASE_ALIASES.get(value, value)
    return f"{namespace}:{value}"


def normalize_tags(tags: Iterable[Any]) -> set[str]:
    """Normalise an iterable of tags, dropping unusable ones."""
    out: set[str] = set()
    for tag in tags or ():
        norm = normalize_tag(tag)
        if norm:
            out.add(norm)
    return out


def _query_slug(name: str) -> str:
    """Backward-compatible helper: canonical hero slug of *name*."""
    return _hero_slug(name)


# Position number (1-5) -> role tag used by the chunk producers
_POSITION_ROLE_TAGS = {
    1: "role:carry",
    2: "role:mid",
    3: "role:offlane",
    4: "role:support_4",
    5: "role:support_5",
}

# Sources produced by the self-learning pipeline (vs builtin / editorial).
LEARNED_SOURCES = ("opendota_stats", "match_mining", "llm_distilled", "learned")


def is_learned_source(source: str) -> bool:
    """Whether a chunk source comes from the self-learning pipeline."""
    source = source or ""
    return source in LEARNED_SOURCES or source.startswith("video")


# ---------------------------------------------------------------------------
# KnowledgeChunk
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeChunk:
    """A single piece of coaching knowledge that can be retrieved by the RAG system.

    Attributes:
        chunk_id: Unique identifier, e.g. ``"hero_laning:antimage"``.
        content: Coaching text (one language per store).
        category: e.g. ``hero_overview``, ``hero_laning``, ``hero_matchup``,
            ``role_principles``, ``phase_objectives``, ``death_analysis``,
            ``vision_principles``, ``combat_mechanics``, ``economy_patterns``,
            ``item_coaching``, ``ability_coaching`` (defaults to the chunk_id prefix).
        tags: Retrieval tags (normalised by the store on insert).
        priority: 1 = critical, 2 = important, 3 = supplementary.
        token_estimate: Approximate token count (auto-estimated from content).
        source: ``"builtin"``, ``"editorial"``, ``"gamedata"``, ``"json"`` or a
            learned source (``"opendota_stats"``, ``"match_mining"``,
            ``"llm_distilled"``, ``"video:<platform>"``).
        patch_version: Patch the knowledge was produced for ("" = not patch-specific).
        dedupe_key: Chunks sharing a key are the same fact; the store keeps one
            (learned statistics are merged into the editorial chunk).
    """

    chunk_id: str
    content: str
    category: str = ""
    tags: set[str] = field(default_factory=set)
    priority: int = 2
    token_estimate: int = 0
    source: str = "builtin"
    patch_version: str = ""
    dedupe_key: str = ""

    def __post_init__(self) -> None:
        if not self.category and ":" in self.chunk_id:
            self.category = self.chunk_id.split(":")[0]
        if not isinstance(self.tags, set):
            self.tags = set(self.tags or ())
        if self.token_estimate <= 0 and self.content:
            self.token_estimate = estimate_tokens(self.content)


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

# Base weight per query-tag kind (before IDF): specific beats generic.
_W_MATCHUP = 4.0
_W_PLAYER_HERO = 3.0
_W_KILLER = 2.0
_W_SITUATION = 2.0
_W_ENEMY = 1.5
_W_ITEM = 1.0
_W_ROLE = 1.2
_W_MECHANIC = 1.0
_W_PHASE = 0.5
_W_UNIVERSAL = 0.25

_POINT_TYPE_KEYWORDS = {
    "death": "death died killed gank 死亡 阵亡 被杀 被抓",
    "overextend": "overextend position push vision 站位 深入 推进 视野",
    "bad_trade": "trade harass lane 换血 消耗 对拼 对线",
    "missed_cs": "last hit cs farm 补刀 正补 发育 刷钱",
    "fight_engage": "teamfight engage initiation 团战 先手 开团",
    "good_play": "teamfight kill 团战 击杀",
}
_PHASE_KEYWORDS = {
    "pregame": "rune bounty 开局 神符",
    "laning": "laning lane 对线",
    "mid_game": "mid game roshan 中期 肉山",
    "late_game": "late game high ground buyback 后期 高地 买活",
}


def build_query(ctx) -> tuple[dict[str, float], str]:
    """Weighted query tags + BM25 query text for a :class:`DecisionContext`."""
    weights: dict[str, float] = {}

    def add(tag: str | None, weight: float) -> None:
        tag = normalize_tag(tag) if tag else None
        if tag and weight > weights.get(tag, 0.0):
            weights[tag] = weight

    dp = ctx.decision_point
    player_hero = ctx.player_state.hero_name
    add(hero_tag(player_hero), _W_PLAYER_HERO)
    add("hero:all", _W_UNIVERSAL)
    if ctx.game_phase:
        add(f"phase:{ctx.game_phase}", _W_PHASE)
    add(f"situation:{dp.point_type}", _W_SITUATION)

    match_context = ctx.match_context or {}
    add(_POSITION_ROLE_TAGS.get(match_context.get("player_position")), _W_ROLE)
    if match_context.get("vision_danger") in ("danger", "critical"):
        add("situation:danger", _W_SITUATION * 0.75)
    try:
        if float(match_context.get("gold_adv", 0) or 0) <= -5000:
            add("situation:behind", _W_SITUATION * 0.75)
    except (TypeError, ValueError):
        pass

    text_parts = [player_hero, dp.point_type.replace("_", " "),
                  _POINT_TYPE_KEYWORDS.get(dp.point_type, "")]
    killer = getattr(dp.related_death, "killer_hero", "") if dp.related_death else ""
    if killer:
        add(hero_tag(killer), _W_KILLER)
        add(matchup_tag(player_hero, killer), _W_MATCHUP)
        add(matchup_tag(killer, player_hero), _W_MATCHUP * 0.75)
        text_parts.append(killer)
    for enemy in list(ctx.enemy_states)[:5]:
        add(hero_tag(enemy.hero_name), _W_ENEMY)
        add(matchup_tag(player_hero, enemy.hero_name), _W_MATCHUP)
        add(matchup_tag(enemy.hero_name, player_hero), _W_MATCHUP * 0.75)
        text_parts.append(enemy.hero_name)
    for item in list(dict.fromkeys(ctx.player_state.items or []))[:12]:
        add(item_tag(item), _W_ITEM)

    if dp.point_type in ("death", "overextend"):
        add("mechanic:vision", _W_MECHANIC)
        add("mechanic:positioning", _W_MECHANIC)
    if dp.point_type in ("bad_trade", "missed_cs"):
        add("mechanic:cs", _W_MECHANIC)
        add("mechanic:trading", _W_MECHANIC)
        add("mechanic:laning", _W_MECHANIC * 0.8)
    if dp.point_type in ("fight_engage", "good_play"):
        add("mechanic:teamfight", _W_MECHANIC)

    if dp.brief:
        text_parts.append(dp.brief)
    if ctx.game_phase:
        text_parts.append(_PHASE_KEYWORDS.get(ctx.game_phase, ctx.game_phase))
    return weights, " ".join(p for p in text_parts if p)


# ---------------------------------------------------------------------------
# KnowledgeStore
# ---------------------------------------------------------------------------

_PRIORITY_WEIGHT = {1: 1.2, 2: 1.0, 3: 0.85}
_STALE_PATCH_FACTOR = 0.5
_SCOPE_MISMATCH_FACTOR = 0.5
_ITEM_NOT_OWNED_FACTOR = 0.6
_SCOPED_NAMESPACES = ("phase:", "role:")
_RELATIVE_CUTOFF = 0.3
_TAG_SHARE = 0.7
_LEXICAL_SHARE = 0.3
_MIN_RELEVANCE = 0.12
_MAX_PER_CATEGORY = 3


class KnowledgeStore:
    """In-memory knowledge store with weighted-tag + BM25 retrieval.

    Build it single-threaded (``add_chunk`` ...), then share it read-only:
    indexes are (re)built lazily under a lock, retrieval never mutates state.
    """

    def __init__(self, current_patch: str = "") -> None:
        self._chunks: dict[str, KnowledgeChunk] = {}   # chunk_id -> chunk
        self._tag_index: dict[str, set[str]] = {}       # tag -> chunk_ids
        self._dedupe: dict[str, str] = {}               # dedupe_key -> chunk_id
        self._bm25: SimpleBM25 | None = None
        self._bm25_dirty: bool = True
        self._index_lock = threading.Lock()
        self.current_patch = current_patch or ""

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _unindex(self, chunk: KnowledgeChunk) -> None:
        for tag in chunk.tags:
            ids = self._tag_index.get(tag)
            if ids is not None:
                ids.discard(chunk.chunk_id)
                if not ids:
                    del self._tag_index[tag]
        if chunk.dedupe_key and self._dedupe.get(chunk.dedupe_key) == chunk.chunk_id:
            del self._dedupe[chunk.dedupe_key]

    def _insert(self, chunk: KnowledgeChunk) -> None:
        old = self._chunks.get(chunk.chunk_id)
        if old is not None:
            self._unindex(old)
        self._chunks[chunk.chunk_id] = chunk
        for tag in chunk.tags:
            self._tag_index.setdefault(tag, set()).add(chunk.chunk_id)
        if chunk.dedupe_key:
            self._dedupe[chunk.dedupe_key] = chunk.chunk_id
        self._bm25_dirty = True

    def remove_chunk(self, chunk_id: str) -> None:
        """Remove a chunk (no-op when absent)."""
        old = self._chunks.pop(chunk_id, None)
        if old is not None:
            self._unindex(old)
            self._bm25_dirty = True

    def add_chunk(self, chunk: KnowledgeChunk) -> None:
        """Add (or replace by ``chunk_id``) a chunk; tags are normalised.

        When another chunk already holds the same ``dedupe_key``: a learned
        chunk is merged into an editorial/builtin one (its text appended as
        labelled statistics), an editorial chunk absorbs a previously added
        learned one, and between two chunks of the same kind the one for the
        current patch (else the first one) wins.
        """
        chunk.tags = normalize_tags(chunk.tags)
        chunk.token_estimate = estimate_tokens(chunk.content)
        key = chunk.dedupe_key
        existing_id = self._dedupe.get(key) if key else None
        if existing_id and existing_id != chunk.chunk_id and existing_id in self._chunks:
            existing = self._chunks[existing_id]
            new_learned = is_learned_source(chunk.source)
            old_learned = is_learned_source(existing.source)
            if new_learned and not old_learned:
                self._insert(_merge_chunks(existing, chunk))
                return
            if old_learned and not new_learned:
                self.remove_chunk(existing_id)
                self._insert(_merge_chunks(chunk, existing))
                return
            if not (self._is_current(chunk) and not self._is_current(existing)):
                return  # keep the first one
            self.remove_chunk(existing_id)
        self._insert(chunk)

    def _is_current(self, chunk: KnowledgeChunk) -> bool:
        return not chunk.patch_version or not self.current_patch or (
            chunk.patch_version == self.current_patch
        )

    def finalize(self) -> None:
        """Build the lexical index now (otherwise built on first retrieval)."""
        self._ensure_index()

    def _ensure_index(self) -> SimpleBM25:
        bm25 = self._bm25
        if bm25 is not None and not self._bm25_dirty:
            return bm25
        with self._index_lock:
            if self._bm25 is None or self._bm25_dirty:
                fresh = SimpleBM25()
                for chunk_id in sorted(self._chunks):
                    fresh.index(chunk_id, self._chunks[chunk_id].content)
                fresh.finalize()
                self._bm25 = fresh
                self._bm25_dirty = False
            return self._bm25

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, chunk_id: str) -> KnowledgeChunk | None:
        return self._chunks.get(chunk_id)

    def __len__(self) -> int:
        return len(self._chunks)

    def rank(self, query_tags: dict[str, float], text: str = "") -> list[tuple[KnowledgeChunk, float]]:
        """Score candidates for weighted *query_tags* + lexical *text*.

        Returns ``[(chunk, score)]`` sorted best-first (ties by chunk_id),
        filtered to chunks with meaningful relevance.
        """
        n_chunks = len(self._chunks)
        if not n_chunks:
            return []
        tag_scores: dict[str, float] = {}
        for tag in sorted(query_tags):
            ids = self._tag_index.get(tag)
            if not ids:
                continue
            weight = query_tags[tag] * math.log(1.0 + n_chunks / len(ids))
            for chunk_id in ids:
                tag_scores[chunk_id] = tag_scores.get(chunk_id, 0.0) + weight
        for chunk_id in list(tag_scores):
            breadth = len(self._chunks[chunk_id].tags)
            tag_scores[chunk_id] /= 1.0 + 0.1 * max(0, breadth - 3)

        lexical = self._ensure_index().scores(text) if text else {}
        max_tag = max(tag_scores.values(), default=0.0)
        max_lex = max(lexical.values(), default=0.0)

        scoped = [
            (ns, {t for t in query_tags if t.startswith(ns)}) for ns in _SCOPED_NAMESPACES
        ]
        scoped = [(ns, wanted) for ns, wanted in scoped if wanted]
        query_items = {t for t in query_tags if t.startswith("item:")}
        results: list[tuple[KnowledgeChunk, float]] = []
        for chunk_id in set(tag_scores) | set(lexical):
            chunk = self._chunks.get(chunk_id)
            if chunk is None or not _entities_match(chunk.tags, query_tags):
                continue
            t = math.sqrt(tag_scores.get(chunk_id, 0.0) / max_tag) if max_tag > 0 else 0.0
            lx = lexical.get(chunk_id, 0.0) / max_lex if max_lex > 0 else 0.0
            score = _TAG_SHARE * t + _LEXICAL_SHARE * lx
            score *= _PRIORITY_WEIGHT.get(chunk.priority, 1.0)
            if is_learned_source(chunk.source) and not self._is_current(chunk):
                score *= _STALE_PATCH_FACTOR
            for namespace, wanted in scoped:
                if _scope_mismatch(chunk.tags, namespace, wanted):
                    score *= _SCOPE_MISMATCH_FACTOR
            if _scope_mismatch(chunk.tags, "item:", query_items):
                # About an item the player does not own (e.g. build advice).
                score *= _ITEM_NOT_OWNED_FACTOR
            if score >= _MIN_RELEVANCE:
                results.append((chunk, round(score, 9)))
        results.sort(key=lambda cs: (-cs[1], cs[0].chunk_id))
        if results:
            floor = results[0][1] * _RELATIVE_CUTOFF
            results = [cs for cs in results if cs[1] >= floor]
        return results

    def retrieve(self, ctx, budget: int = 3000) -> list[KnowledgeChunk]:
        """Retrieve relevant chunks for a :class:`DecisionContext` within *budget* tokens."""
        query_tags, text = build_query(ctx)
        return self.select(self.rank(query_tags, text), budget)

    # Backward-compatible internal names
    def _build_query_tags(self, ctx) -> set[str]:
        return set(build_query(ctx)[0])

    def _tag_retrieve(self, query_tags: set[str]) -> list[tuple[KnowledgeChunk, float]]:
        return self.rank({t: 1.0 for t in normalize_tags(query_tags)})

    @staticmethod
    def select(
        scored: list[tuple[KnowledgeChunk, float]],
        budget: int,
        per_category: int = _MAX_PER_CATEGORY,
    ) -> list[KnowledgeChunk]:
        """Pick best-first within *budget* tokens with category / dedupe diversity."""
        selected: list[KnowledgeChunk] = []
        used = 0
        per_cat: dict[str, int] = {}
        keys: set[str] = set()
        for chunk, score in sorted(scored, key=lambda cs: (-cs[1], cs[0].chunk_id)):
            if score <= 0:
                break
            if per_cat.get(chunk.category, 0) >= per_category:
                continue
            if chunk.dedupe_key and chunk.dedupe_key in keys:
                continue
            cost = chunk.token_estimate or estimate_tokens(chunk.content)
            if used + cost > budget:
                continue
            selected.append(chunk)
            used += cost
            per_cat[chunk.category] = per_cat.get(chunk.category, 0) + 1
            if chunk.dedupe_key:
                keys.add(chunk.dedupe_key)
        return selected

    _select_within_budget = select

    # ------------------------------------------------------------------
    # Bulk loading
    # ------------------------------------------------------------------

    def load_builtin(self, lang: str = "zh") -> None:
        """Load the structural built-in chunks (principles) for *lang*."""
        try:
            from .knowledge_chunks import get_all_builtin_chunks
        except ImportError:  # pragma: no cover
            logger.warning("knowledge_chunks unavailable; no builtin chunks loaded")
            return
        for chunk in get_all_builtin_chunks(lang=lang):
            self.add_chunk(chunk)

    def load_json(self, path: str) -> int:
        """Load chunks from a JSON array file; returns the number loaded.

        Each object needs ``chunk_id`` and ``content``; ``category``,
        ``tags``, ``priority``, ``token_estimate``, ``source``,
        ``patch_version`` and ``dedupe_key`` are optional.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = 0
        for item in data if isinstance(data, list) else []:
            if not isinstance(item, dict) or not item.get("chunk_id") or not item.get("content"):
                continue
            try:
                priority = int(item.get("priority", 2))
            except (TypeError, ValueError):
                priority = 2
            self.add_chunk(KnowledgeChunk(
                chunk_id=str(item["chunk_id"]),
                content=str(item["content"]),
                category=str(item.get("category") or "general"),
                tags=set(item.get("tags") or []),
                priority=priority if priority in (1, 2, 3) else 2,
                source=str(item.get("source") or "json"),
                patch_version=str(item.get("patch_version") or ""),
                dedupe_key=str(item.get("dedupe_key") or ""),
            ))
            count += 1
        return count

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
            "categories": dict(sorted(cats.items())),
        }


_ENTITY_PREFIXES = ("hero:", "item:", "matchup:")


def _entities_match(tags: set[str], query_tags: dict[str, float]) -> bool:
    """Entity gate: a chunk about specific heroes / items / matchups is only
    eligible when at least one of those entities is part of the query (a tip
    about Invoker is noise in a game without Invoker, however well its
    generic phase/role tags match). Chunks without entity tags always pass.
    """
    entities = [t for t in tags if t.startswith(_ENTITY_PREFIXES) and t != "hero:all"]
    return not entities or any(t in query_tags for t in entities)


def _scope_mismatch(tags: set[str], namespace: str, wanted: set[str]) -> bool:
    """True when the chunk is scoped to phases/roles and none is the query's."""
    scoped = [t for t in tags if t.startswith(namespace)]
    return bool(scoped) and not any(t in wanted for t in scoped)


def _merge_chunks(primary: KnowledgeChunk, extra: KnowledgeChunk) -> KnowledgeChunk:
    """Editorial/builtin *primary* chunk with a learned *extra* appended."""
    content = primary.content.rstrip()
    addition = extra.content.strip()
    if addition and addition not in content:
        content = f"{content}\n{addition}"
    return KnowledgeChunk(
        chunk_id=primary.chunk_id,
        content=content,
        category=primary.category,
        tags=set(primary.tags) | set(extra.tags),
        priority=min(primary.priority, extra.priority),
        source=primary.source,
        patch_version=primary.patch_version,
        dedupe_key=primary.dedupe_key,
    )


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "KnowledgeChunk",
    "KnowledgeStore",
    "SimpleBM25",
    "estimate_tokens",
    "normalize_tag",
    "normalize_tags",
    "hero_tag",
    "item_tag",
    "matchup_tag",
    "build_query",
    "is_learned_source",
    "LEARNED_SOURCES",
]
