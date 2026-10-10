"""RAG knowledge retrieval engine: weighted tags + BM25, plus a dense term for principles.

In-memory knowledge storage and retrieval for the AI coaching system, with no
external dependencies (the dense scores are computed by :mod:`rag` and handed
to :meth:`KnowledgeStore.rank`).

Ranking (deterministic, see :meth:`KnowledgeStore.rank`):

1. The decision context is turned into *weighted* query tags: specific tags
   (the exact matchup, the player's hero, the killer, the situation) weigh far
   more than generic ones (game phase, ``hero:all``). Each matched tag is
   further weighted by its inverse document frequency, so tags carried by
   hundreds of chunks contribute little.
2. A BM25 lexical score over the chunk text (CJK bigrams + Latin words)
   rewards chunks that actually talk about the heroes / situation.
3. Lexical formula: ``0.7 * sqrt(tag score) + 0.3 * lexical score`` (both
   normalised to the best *eligible* candidate), times priority, times a
   penalty for learned chunks that were produced on a different patch.
   Ties break on ``chunk_id``.
   Pooled hybrid formula, used when dense scores are given: ``0.6 * sqrt(tag
   score) + 0.1 * lexical score + 0.3 * dense``, same multipliers. The dense
   term ranks the *generic* chunks only (no hero / matchup / item tag): their
   cosine to the situation query, min-max normalised inside that pool. A
   chunk it does not rank gets the neutral constant 0.5: an entity chunk
   (tags decide its relevance; cosine similarity mostly matches hero names)
   and a generic chunk that has no vector (a learned or operator text on a
   server without an embedding model).
4. Chunks are picked best-first within ONE token budget (estimated with
   :func:`estimate_tokens`), at most 3 per category, one per dedupe key and
   at most 2 written by an LLM or a third party.

Eligibility (the entity gate, :func:`_entities_match`): a chunk about a
matchup needs that matchup in the query; otherwise a chunk about specific
heroes needs one of those heroes in the game — the player's own hero for a
farm / item-timing statistic (:data:`ECONOMY_TAG`), which describes players
of that hero; otherwise a chunk about items needs the player to own one (or
to be a listed ``buyer:`` of it). Such a statistic is also halved for every
decision that is not about farming (``topic:`` is a scoped namespace like
``phase:`` and ``role:``).

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
from typing import Any, Callable, Iterable

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

# Namespaces that name a concrete hero / item / matchup.
_ENTITY_PREFIXES = ("hero:", "item:", "matchup:")

_ROLE_ALIASES = {
    "carry": "carry", "pos1": "carry", "pos_1": "carry", "position_1": "carry",
    "position1": "carry", "safelane": "carry", "safe_lane": "carry", "hard_carry": "carry",
    "mid": "mid", "pos2": "mid", "pos_2": "mid", "position_2": "mid", "position2": "mid",
    "midlane": "mid", "mid_lane": "mid", "midlaner": "mid",
    "offlane": "offlane", "pos3": "offlane", "pos_3": "offlane", "position_3": "offlane",
    "position3": "offlane", "offlaner": "offlane", "off_lane": "offlane",
    "support_4": "support_4", "pos4": "support_4", "pos_4": "support_4", "position_4": "support_4",
    "position4": "support_4", "soft_support": "support_4", "roamer": "support_4",
    "roaming_support": "support_4", "support4": "support_4",
    "support_5": "support_5", "pos5": "support_5", "pos_5": "support_5", "position_5": "support_5",
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


def buyer_tag(name: Any) -> str:
    """``buyer:<internal name>``: the hero is a typical buyer of the chunk's item.

    Unlike ``hero:`` it does not make the chunk "about" that hero: only the
    player's own hero is ever queried with it (see :func:`build_query`).
    """
    return f"buyer:{_hero_slug(name)}"


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
    if namespace in ("hero", "buyer"):
        value = _hero_slug(rest)
        return f"{namespace}:{value}" if value else None
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


def _known_entity(tag: str) -> bool:
    """Whether a normalised hero / matchup / item tag names something we know."""
    try:
        from .data_manager import is_known_hero, is_known_item
    except Exception:  # pragma: no cover - data layer unavailable
        return True
    namespace, _, rest = tag.partition(":")
    if namespace == "hero":
        return rest == "all" or is_known_hero(rest)
    if namespace == "matchup":
        a, _, b = rest.partition(":")
        return is_known_hero(a) and is_known_hero(b)
    if namespace == "item":
        return is_known_item(rest)
    return True


def scoped_tags(tags: Iterable[Any]) -> set[str] | None:
    """Normalised tags of a third-party (learned) chunk, or None to drop it.

    :func:`normalize_tag` keeps a name it cannot resolve as a plain slug
    (``item:midas`` before the alias existed, ``hero:xyz``). No query ever
    emits such a tag, so it would silently change who sees the chunk:

    - an unresolved hero or matchup tag: the chunk is dropped (None) —
      without the tag it would be shown for every hero;
    - an unresolved item tag is removed when a hero or matchup tag still
      scopes the chunk, otherwise the chunk is dropped as well.
    """
    normalised = normalize_tags(tags)
    unknown = {t for t in normalised if t.startswith(_ENTITY_PREFIXES) and not _known_entity(t)}
    if not unknown:
        return normalised
    if any(not t.startswith("item:") for t in unknown):
        return None
    kept = normalised - unknown
    scoped = any(t.startswith(("hero:", "matchup:")) and t != "hero:all" for t in kept)
    return kept if scoped else None


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

# Topic of a chunk or a query (a scoped namespace: a chunk that states a
# topic is halved for a query about another one). Farm and item-timing
# statistics carry ECONOMY_TAG (the store builder tags them): they say how
# fast a hero's players usually earn and buy, which explains a missed-CS
# decision and next to nothing about a death, a fight or a kill (first run
# with real learned data: 9.5 % of the sources listed under deaths and 32 %
# of those under good plays were such statistics).
ECONOMY_TAG = "topic:economy"
FIGHT_TAG = "topic:fight"
_ECONOMY_CATEGORIES = frozenset({"economy_patterns", "item_coaching"})


def is_economy_statistic(category: str, source: str, tags: Iterable[str]) -> bool:
    """Whether a learned chunk is a farm / item-timing statistic.

    Hero benchmarks and item timings (OpenDota), item orders of a hero in
    pro matches, and what the distiller files under the same categories.
    The per-position pattern lines of the match miner (CS, deaths, kill
    participation, wards in one chunk) are not: they are role background.
    """
    if not is_learned_source(source) or (source or "").startswith("video"):
        return False
    return category in _ECONOMY_CATEGORIES or (category == "pro_pattern" and is_entity_scoped(tags))


def is_learned_source(source: str) -> bool:
    """Whether a chunk source comes from the self-learning pipeline."""
    source = source or ""
    return source in LEARNED_SOURCES or source.startswith("video")


# Source types shown next to a verdict (rag.types.SOURCE_TYPES); prose of
# the untrusted ones was written by an LLM or a third party.
_UNTRUSTED_SOURCE_TYPES = frozenset({"llm_distilled", "video"})


def source_type_of(source: str) -> str:
    """Source-type code of a chunk ``source``.

    ``editorial`` | ``builtin`` | ``operator`` | ``opendota_stats`` |
    ``match_mining`` | ``llm_distilled`` | ``video``. ``learned`` — what the
    store builder gives a learned chunk whose own source it does not know —
    counts as ``llm_distilled`` (untrusted); any other string is the label
    of an operator file (``backend/data/knowledge``) and counts as
    ``operator``.
    """
    source = source or ""
    if source.startswith("video"):
        return "video"
    if source in ("opendota_stats", "match_mining", "llm_distilled"):
        return source
    if source == "learned":
        return "llm_distilled"
    if source == "builtin":
        return "builtin"
    if source in ("editorial", "gamedata"):
        return "editorial"
    return "operator"


def is_untrusted_source(source: str) -> bool:
    """Whether a chunk ``source`` is prose written by an LLM or a third party."""
    return source_type_of(source) in _UNTRUSTED_SOURCE_TYPES


def is_entity_scoped(tags: Iterable[str]) -> bool:
    """Whether a chunk is about specific heroes / matchups / items.

    Chunks that are not ("generic": principles, role and phase notes) are
    the pool the dense term of the ranker orders.
    """
    return any(t.startswith(_ENTITY_PREFIXES) and t != "hero:all" for t in tags)


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
        title: Short human title for citations, in the store's language
            ("" = derived from the first line, see :attr:`display_title`).
        index_text: Canonical ENGLISH text of the chunk, identical in the zh
            and en stores: what gets embedded. "" = the chunk has no English
            text and is ranked lexically. Set by ``knowledge_base`` when it
            builds the stores.
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
    title: str = ""
    index_text: str = ""

    def __post_init__(self) -> None:
        if not self.category and ":" in self.chunk_id:
            self.category = self.chunk_id.split(":")[0]
        if not isinstance(self.tags, set):
            self.tags = set(self.tags or ())
        if self.token_estimate <= 0 and self.content:
            self.token_estimate = estimate_tokens(self.content)

    @property
    def source_type(self) -> str:
        """Source-type code, see :func:`source_type_of`."""
        return source_type_of(self.source)

    @property
    def trusted(self) -> bool:
        """False for prose written by an LLM or a third party (video tips)."""
        return not is_untrusted_source(self.source)

    @property
    def display_title(self) -> str:
        """:attr:`title`, or the chunk's first line without markup (at most 80 characters)."""
        if self.title:
            return self.title
        for line in self.content.splitlines():
            line = _TITLE_MARKUP_RE.sub("", line).strip(" :：")
            if line:
                return line if len(line) <= 80 else line[:79].rstrip() + "…"
        return self.chunk_id


# Leading markdown / learned-label markup of a content line ("## ", "- ", "[Statistics · OpenDota] ").
_TITLE_MARKUP_RE = re.compile(r"^\s*(?:#{1,6}\s*|[-*]\s+|\*\*|\[[^\]]{0,60}\]\s*|【[^】]{0,60}】\s*)+")


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

# Base weight per query-tag kind (before IDF): specific beats generic.
_W_MATCHUP = 4.0
# The opponent's side of a matchup (a tip written for the enemy player) is
# useful context but must rank below the player's own hero chunks.
_REVERSE_MATCHUP_SHARE = 0.5
_W_PLAYER_HERO = 3.0
_W_KILLER = 2.0
_W_SITUATION = 2.0
_W_ENEMY = 1.5
_W_ITEM = 1.0
_W_BUYER = 0.5   # item tips for a typical buyer who does not own the item yet
_W_ROLE = 1.2
_W_MECHANIC = 1.0
_W_PHASE = 0.5
_W_UNIVERSAL = 0.25
# A tag that scopes without ranking (the query's topic): far below the
# relevance floor on its own.
_W_SCOPE_ONLY = 0.01

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


def _hero_display(name: Any) -> str:
    """Display name of a hero for the lexical query (chunk text uses display names)."""
    try:
        from .data_manager import resolve_hero_name
        return resolve_hero_name(name) or str(name or "")
    except Exception:  # pragma: no cover - data layer unavailable
        return str(name or "")


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
    add(buyer_tag(player_hero), _W_BUYER)  # the player only, never an enemy
    add("hero:all", _W_UNIVERSAL)
    if ctx.game_phase:
        add(f"phase:{ctx.game_phase}", _W_PHASE)
    add(f"situation:{dp.point_type}", _W_SITUATION)
    # What the decision is about, for chunks that state a topic of their own.
    add(ECONOMY_TAG if dp.point_type == "missed_cs" else FIGHT_TAG, _W_SCOPE_ONLY)

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
        add(matchup_tag(killer, player_hero), _W_MATCHUP * _REVERSE_MATCHUP_SHARE)
        text_parts.append(_hero_display(killer))  # killer_hero may be an npc name
    for enemy in list(ctx.enemy_states)[:5]:
        add(hero_tag(enemy.hero_name), _W_ENEMY)
        add(matchup_tag(player_hero, enemy.hero_name), _W_MATCHUP)
        add(matchup_tag(enemy.hero_name, player_hero), _W_MATCHUP * _REVERSE_MATCHUP_SHARE)
        text_parts.append(enemy.hero_name)
    for item in list(dict.fromkeys(ctx.player_state.items or []))[:12]:
        add(item_tag(item), _W_ITEM)

    # What kind of death decides which principles apply (measured on the
    # retrieval benchmark: without these the teamfight-positioning chunk never
    # reached the top 10 of a teamfight death, nor the vision-safety chunk
    # that of a pick-off).
    teamfight_death = dp.point_type == "death" and bool(getattr(dp.related_death, "is_teamfight", False))
    if dp.point_type in ("death", "overextend"):
        add("mechanic:positioning", _W_MECHANIC)
        if not teamfight_death:  # a teamfight death is rarely a vision problem
            add("mechanic:vision", _W_MECHANIC)
    if teamfight_death:
        add("mechanic:teamfight", _W_MECHANIC * 1.5)
        add("mechanic:trade", _W_MECHANIC)
    elif dp.point_type == "death":  # pick-off / gank
        add("situation:gank", _W_SITUATION * 0.75)
        add("situation:danger", _W_SITUATION * 0.75)
        add("mechanic:map_awareness", _W_MECHANIC)
    if dp.point_type == "overextend":  # staying in lane on low HP is a lost trade
        add("situation:bad_trade", _W_SITUATION)
        add("mechanic:trading", _W_MECHANIC)
        add("mechanic:laning", _W_MECHANIC * 0.8)
    if ctx.game_phase == "pregame":
        # Before the horn the rune fight IS the situation, whatever the point
        # type: the phase and rune tags weigh like a situation tag (with the
        # 0.5 of a phase the rune-contest and level-1 chunks ended up behind
        # every death-analysis section, or outside the block).
        add("phase:pregame", _W_SITUATION)
        add("mechanic:rune", _W_SITUATION)
        add("mechanic:teamfight", _W_MECHANIC)
    if dp.point_type in ("bad_trade", "missed_cs"):
        add("mechanic:cs", _W_MECHANIC)
        add("mechanic:laning", _W_MECHANIC * 0.8)
    if dp.point_type == "bad_trade":
        add("mechanic:trading", _W_MECHANIC)
    if dp.point_type in ("fight_engage", "good_play"):
        add("mechanic:teamfight", _W_MECHANIC)
    if dp.point_type == "good_play" and getattr(dp, "subtype", "") == "smoke":
        # A kill out of a smoke is judged by the smoke-gank principles first
        # (without these tags that chunk sat at rank 5 behind teamfight
        # positioning in all five real smoke plays).
        add("mechanic:smoke", _W_MECHANIC)
        add("mechanic:gank", _W_MECHANIC)

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
_SCOPED_NAMESPACES = ("phase:", "role:", "topic:")
_RELATIVE_CUTOFF = 0.3
_TAG_SHARE = 0.7
_LEXICAL_SHARE = 0.3
_MIN_RELEVANCE = 0.12
_MAX_PER_CATEGORY = 3
# Entries of one block whose prose was written by an LLM or a third party
# (source types llm_distilled / video): background, never the bulk of a block.
_MAX_UNTRUSTED = 2
# Pooled hybrid formula (used when dense scores are available). Constants
# measured on the labelled benchmark (backend/tests/data/rag_benchmark).
_HYBRID_TAG_SHARE = 0.6
_HYBRID_LEXICAL_SHARE = 0.1
_HYBRID_DENSE_SHARE = 0.3
# Dense credit of a chunk the dense term does not rank: a hero / matchup /
# item chunk, and a generic chunk without a vector. Neutral, not 0: on a
# server without an embedding model the learned and operator principles have
# no vector, and 0 pushed them behind every principle that has one (measured:
# missed-CS nDCG@10 0.92 lexical, 0.87 with 0, 0.89 with 0.5; overall equal).
_NEUTRAL_DENSE = 0.5
# The dense term lifts more generic chunks over the relative cut-off; 0.5
# keeps the share of irrelevant chunks down. On the lexical formula the same
# value costs recall, so that one stays at 0.3.
_HYBRID_RELATIVE_CUTOFF = 0.5


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
        self.revision = 0   # bumped by every mutation (caches derived from the chunks key on it)

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
        self.revision += 1

    def remove_chunk(self, chunk_id: str) -> None:
        """Remove a chunk (no-op when absent)."""
        old = self._chunks.pop(chunk_id, None)
        if old is not None:
            self._unindex(old)
            self._bm25_dirty = True
            self.revision += 1

    def add_chunk(self, chunk: KnowledgeChunk) -> None:
        """Add (or replace by ``chunk_id``) a chunk; tags are normalised.

        When another chunk already holds the same ``dedupe_key``: a learned
        chunk is merged into an editorial/builtin one (its text appended as
        labelled statistics), an editorial chunk absorbs a previously added
        learned one, and between two chunks of the same kind the one for the
        current patch (else the first one) wins. A learned chunk from
        another patch is never merged: inside the editorial chunk it would
        lose both its patch and the stale-patch penalty, so the editorial
        chunk is kept alone.

        A chunk written by an LLM or a third party is never "critical":
        priority 1 (x1.2 in the ranking) is lowered to 2, whoever produced
        the chunk.
        """
        chunk.tags = normalize_tags(chunk.tags)
        chunk.token_estimate = estimate_tokens(chunk.content)
        if not chunk.trusted and chunk.priority < 2:
            chunk.priority = 2
        key = chunk.dedupe_key
        existing_id = self._dedupe.get(key) if key else None
        if existing_id and existing_id != chunk.chunk_id and existing_id in self._chunks:
            existing = self._chunks[existing_id]
            new_learned = is_learned_source(chunk.source)
            old_learned = is_learned_source(existing.source)
            if new_learned and not old_learned:
                if self._is_current(chunk):
                    self._insert(_merge_chunks(existing, chunk))
                return
            if old_learned and not new_learned:
                self.remove_chunk(existing_id)
                self._insert(_merge_chunks(chunk, existing) if self._is_current(existing) else chunk)
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

    def chunks(self) -> list[KnowledgeChunk]:
        """Every chunk, sorted by ``chunk_id`` (treat them as read-only)."""
        return [self._chunks[chunk_id] for chunk_id in sorted(self._chunks)]

    def rank(self, query_tags: dict[str, float], text: str = "",
             dense: dict[str, float] | None = None) -> list[tuple[KnowledgeChunk, float]]:
        """Score candidates for weighted *query_tags* + lexical *text*.

        *dense* maps chunk ids to the cosine similarity between the chunk
        and the situation query. Only generic chunks are read from it (see
        the module docstring); when it holds none, ranking is lexical. The
        caller decides whether the scores are worth using (enough of the
        principles covered, see :func:`rag.runtime.query_scores`).

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
        # The dense term orders the generic pool only (generic chunks always
        # pass the entity gate, so every chunk of the pool is a candidate).
        pool: dict[str, float] = {}
        for chunk_id, cosine in (dense or {}).items():
            chunk = self._chunks.get(chunk_id)
            if chunk is not None and not is_entity_scoped(chunk.tags):
                pool[chunk_id] = float(cosine)
        hybrid = bool(pool)
        low, high = (min(pool.values()), max(pool.values())) if hybrid else (0.0, 0.0)
        if hybrid:
            tag_share, lexical_share, dense_share = (
                _HYBRID_TAG_SHARE, _HYBRID_LEXICAL_SHARE, _HYBRID_DENSE_SHARE)
        else:
            tag_share, lexical_share, dense_share = _TAG_SHARE, _LEXICAL_SHARE, 0.0
        # Entity gate first: both parts are normalised to the best ELIGIBLE
        # candidate, not to a chunk about a hero that is not in the game.
        eligible = [
            chunk_id for chunk_id in set(tag_scores) | set(lexical) | set(pool)
            if chunk_id in self._chunks
            and _entities_match(self._chunks[chunk_id].tags, query_tags)
        ]
        max_tag = max((tag_scores.get(c, 0.0) for c in eligible), default=0.0)
        max_lex = max((lexical.get(c, 0.0) for c in eligible), default=0.0)

        scoped = [
            (ns, {t for t in query_tags if t.startswith(ns)}) for ns in _SCOPED_NAMESPACES
        ]
        scoped = [(ns, wanted) for ns, wanted in scoped if wanted]
        query_items = {t for t in query_tags if t.startswith("item:")}
        results: list[tuple[KnowledgeChunk, float]] = []
        for chunk_id in eligible:
            chunk = self._chunks[chunk_id]
            t = math.sqrt(tag_scores.get(chunk_id, 0.0) / max_tag) if max_tag > 0 else 0.0
            lx = lexical.get(chunk_id, 0.0) / max_lex if max_lex > 0 else 0.0
            score = tag_share * t + lexical_share * lx
            if hybrid:
                if chunk_id in pool:
                    dn = (pool[chunk_id] - low) / (high - low) if high > low else _NEUTRAL_DENSE
                else:  # not ranked by cosine: an entity chunk, or a generic chunk without a vector
                    dn = _NEUTRAL_DENSE
                score += dense_share * dn
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
            floor = results[0][1] * (_HYBRID_RELATIVE_CUTOFF if hybrid else _RELATIVE_CUTOFF)
            results = [cs for cs in results if cs[1] >= floor]
        return results

    def retrieve(self, ctx, budget: int = 3000) -> list[KnowledgeChunk]:
        """Retrieve relevant chunks for a :class:`DecisionContext` within *budget* tokens.

        Lexical ranking; :func:`knowledge_base.get_knowledge_for_context`
        adds the dense term.
        """
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
        overhead: Callable[[KnowledgeChunk], int] | None = None,
        max_chunks: int = 0,
        max_untrusted: int = _MAX_UNTRUSTED,
    ) -> list[KnowledgeChunk]:
        """Pick best-first within *budget* tokens with category / dedupe diversity.

        *overhead* returns the tokens a chunk costs on top of its content
        (its numbered handle line); *max_chunks* > 0 caps the selection;
        at most *max_untrusted* chunks written by an LLM or a third party
        are picked (see :attr:`KnowledgeChunk.trusted`; negative = no cap).
        """
        selected: list[KnowledgeChunk] = []
        used = 0
        untrusted = 0
        per_cat: dict[str, int] = {}
        keys: set[str] = set()
        for chunk, score in sorted(scored, key=lambda cs: (-cs[1], cs[0].chunk_id)):
            if score <= 0 or (max_chunks > 0 and len(selected) >= max_chunks):
                break
            if per_cat.get(chunk.category, 0) >= per_category:
                continue
            if chunk.dedupe_key and chunk.dedupe_key in keys:
                continue
            if not chunk.trusted and 0 <= max_untrusted <= untrusted:
                continue
            cost = chunk.token_estimate or estimate_tokens(chunk.content)
            if overhead is not None:
                cost += overhead(chunk)
            if used + cost > budget:
                continue
            selected.append(chunk)
            used += cost
            untrusted += not chunk.trusted
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
        ``patch_version``, ``dedupe_key``, ``title`` and ``index_text`` (the
        English text of a chunk written in another language) are optional.
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
                title=str(item.get("title") or ""),
                index_text=str(item.get("index_text") or ""),
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


def _entities_match(tags: set[str], query_tags: dict[str, float]) -> bool:
    """Entity gate: is a chunk about specific heroes / items / matchups relevant?

    Decided by the most specific entity namespace the chunk carries (a tip
    about Invoker is noise in a game without Invoker, however well its
    generic phase / role tags or its item tag match):

    - ``matchup:`` tags: one of those matchups must be in the query;
    - else ``hero:`` tags (other than ``hero:all``): one of those heroes
      must be in the game — owning the item a "[hero, item]" statistic
      talks about is not enough. A farm / item-timing statistic
      (:data:`ECONOMY_TAG`) needs the hero to be the player's own
      (``buyer:<hero>`` is only ever queried for him): the median GPM of the
      hero who killed the player says nothing about the player's decision;
    - else ``item:`` tags: the player owns one of the items, or is a listed
      ``buyer:`` of it (an enemy on the buyer list does not count: only the
      player's own hero is queried as buyer).

    Chunks without entity tags always pass.
    """
    matchups = [t for t in tags if t.startswith("matchup:")]
    if matchups:
        return any(t in query_tags for t in matchups)
    heroes = [t for t in tags if t.startswith("hero:") and t != "hero:all"]
    if heroes:
        if ECONOMY_TAG in tags:
            return any("buyer:" + t[len("hero:"):] in query_tags for t in heroes)
        return any(t in query_tags for t in heroes)
    if any(t.startswith("item:") for t in tags):
        return any(t in query_tags for t in tags if t.startswith(("item:", "buyer:")))
    return True


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
        priority=primary.priority,  # a statistic does not make the tip more important
        source=primary.source,
        patch_version=primary.patch_version,
        dedupe_key=primary.dedupe_key,
        title=primary.title,
        # index_text is assigned from the finished English store (knowledge_base)
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
    "scoped_tags",
    "hero_tag",
    "item_tag",
    "buyer_tag",
    "matchup_tag",
    "build_query",
    "is_learned_source",
    "is_untrusted_source",
    "is_entity_scoped",
    "is_economy_statistic",
    "source_type_of",
    "LEARNED_SOURCES",
    "ECONOMY_TAG",
    "FIGHT_TAG",
]
