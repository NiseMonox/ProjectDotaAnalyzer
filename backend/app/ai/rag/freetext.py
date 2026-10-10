"""Free-text knowledge search (the agent's own questions).

Unlike a decision, a free-text question has no calibrated situation tags, so
the ranking differs from :meth:`KnowledgeStore.rank`:

- *Hero-mention gate.* The entity gate is built from the heroes and items
  the question names (plus the coached hero): a chunk about any other hero
  is out, however similar its text. Named heroes that are not in this match
  are ignored and reported; a question that is left with nothing to look
  for (every hero it names is ignored, no item, no topic) returns no hits.
- *Keyword rankings*, always:

  a. the tags of what the question names — its heroes and items, the
     ``topic`` argument, and the topic words it uses ("smoke", "buyback",
     "wards": :data:`situations.QUESTION_TOPICS`). The coached hero, role
     and phase are NOT ranked here: every question of a run shares them,
     and with them the same hero entries came first whatever was asked.
     They only rank when the question names nothing at all. A ``topic``
     that maps onto a situation query adds the prebuilt vectors of that
     query to this ranking;
  b. BM25 of the question over the English index text of the chunks that
     pass the gate (a second index per store, so a Chinese report is
     searched with the same English question). A question that is not in
     English is matched against the store's own text instead.

- *Dense ranking*, when an embedding model is loaded in this process
  (``RAG_EMBEDDINGS=onnx`` / ``openai`` with the model available): the
  cosine of the embedded question to every chunk that passes the gate. An
  English-only model is only used for a question written in English.

The rankings are fused by reciprocal rank (none of them has calibrated
weights here). Without a model — the default deployment — the search is
keyword-only. Chunks written for another role are left out of (b) and of
the dense ranking; the decision's phase does not filter (a question may be
about a later one). Measured on 15 questions with an obvious answer entry
(replay 8762775364, Wraith King, real learned data on disk): that entry is
in the top 4 for 15 (keyword-only) and 14 (with the model) in both report
languages. Before, the keyword side was the store's own formula over the
context tags (coached hero, role, phase): 0 of the 7 questions measured
then, and with the model 10 (en) / 9 (zh) of the 15.
"""
from __future__ import annotations

import logging
import re
import threading
import weakref
from typing import Iterable

from . import situations, vectors
from .types import SearchResult

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 300
MAX_TOP_K = 6
RRF_K = 60

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SQUASH_RE = re.compile(r"[^a-z0-9]+")
# Weights of the query tags (same scale as knowledge_store.build_query).
_W_NAMED_HERO = 3.0
_W_PLAYER_HERO = 1.0
_W_MATCHUP = 4.0
_W_REVERSE_MATCHUP = 2.0
_W_ITEM = 1.0
_W_TOPIC = 1.5
_W_ROLE = 0.6
_W_PHASE = 0.5
_W_UNIVERSAL = 0.25
# Role, phase and topic of the decision next to tags the question itself
# named: too small to rank anything, enough for the store to halve a chunk
# written for another role, phase or topic.
_W_SCOPE_ONLY = 0.01
_ROLE_TAGS = {1: "role:carry", 2: "role:mid", 3: "role:offlane", 4: "role:support_4", 5: "role:support_5"}
# BM25 hits below this share of the best one are word noise, not answers.
_KEYWORD_CUTOFF = 0.25
# Words of a question that say nothing about its subject (the store's
# tokenizer keeps them: in a chunk "how" and "when" are rare enough to match).
_QUESTION_WORDS = frozenset(
    "how what when where why which who whom should would could shall may might must does did do done "
    "am i me my mine we us our he him she they them it's about against versus vs "
    "use used using play played playing deal get got need needs want best good bad better worth".split()
)
# Topic tags that make a question one about farming or item builds.
_ECONOMY_QUESTION = frozenset({"mechanic:farming", "mechanic:cs", "mechanic:item_usage"})

# store -> (store revision, BM25 over the English index text)
_english_indexes: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_english_lock = threading.Lock()


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def find_entities(query: str) -> tuple[list[str], list[str]]:
    """``(hero slugs, item keys)`` named in *query*.

    Looks at runs of one to three words. A run shorter than four letters
    only counts when it is a capitalised hero name ("Axe", "Io"): the short
    aliases ("am", "tp", "pa") are English words.
    """
    from .. import data_manager

    words = _WORD_RE.findall(query or "")
    used = [False] * len(words)
    heroes: list[str] = []
    items: list[str] = []
    for size in (3, 2, 1):
        for start in range(len(words) - size + 1):
            if any(used[start:start + size]):
                continue
            phrase = " ".join(words[start:start + size])
            letters = len(_SQUASH_RE.sub("", phrase.lower()))
            name = data_manager.resolve_hero_name(phrase)
            if name and letters < 4:
                exact = size == 1 and phrase[:1].isupper() and name.lower() == phrase.lower()
                name = name if exact else None
            if name:
                heroes.append(data_manager.hero_slug(name))
                used[start:start + size] = [True] * size
                continue
            key = data_manager.resolve_item_key(phrase) if letters >= 4 else None
            if key:
                items.append(key)
                used[start:start + size] = [True] * size
    return _dedupe(heroes), _dedupe(items)


def _is_english(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and sum(ch.isascii() for ch in letters) / len(letters) >= 0.7


def _fuse(*rankings: list) -> list[tuple[object, float]]:
    """Reciprocal-rank fusion of chunk rankings: ``[(chunk, score)]``, best first."""
    scores: dict[str, float] = {}
    chunks: dict[str, object] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            chunks[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(((chunks[cid], round(score, 9)) for cid, score in scores.items()),
                  key=lambda cs: (-cs[1], cs[0].chunk_id))


def _english_index(store):
    """BM25 over the English index text of *store* (built once per store revision).

    A chunk without English text (an operator file written in Chinese) is
    indexed with its own text, so it stays findable by the names it mentions.
    """
    from .. import knowledge_store as ks

    revision = getattr(store, "revision", None)
    with _english_lock:
        cached = _english_indexes.get(store)
        if cached is not None and cached[0] == revision:
            return cached[1]
        index = ks.SimpleBM25()
        for chunk in store.chunks():
            index.index(chunk.chunk_id, getattr(chunk, "index_text", "") or chunk.content)
        index.finalize()
        _english_indexes[store] = (revision, index)
        return index


def _keyword_ranking(store, query: str, gate: dict[str, float]) -> list:
    """Chunks that pass the entity gate of *gate*, ordered by BM25 of *query*.

    An English question is matched against the English index text, any other
    against the store's own text. A chunk written for another role is left
    out (the phase is not checked: a question may ask about a later one),
    and so are hits far below the best one.
    """
    from .. import knowledge_store as ks

    english = _is_english(query)
    if english:
        query = " ".join(word for word in query.split() if word.strip("?!.,;:'\"()").lower() not in _QUESTION_WORDS)
    if not query:
        return []
    index = _english_index(store) if english else store._ensure_index()
    roles = {tag for tag in gate if tag.startswith("role:")}
    hits = [(store.get(chunk_id), score) for chunk_id, score in index.scores(query).items()]
    hits = [(chunk, score) for chunk, score in hits
            if chunk is not None and score > 0 and ks._entities_match(chunk.tags, gate)
            and not (roles and ks._scope_mismatch(chunk.tags, "role:", roles))]
    if not hits:
        return []
    floor = max(score for _chunk, score in hits) * _KEYWORD_CUTOFF
    hits = [cs for cs in hits if cs[1] >= floor]
    hits.sort(key=lambda cs: (-cs[1], cs[0].chunk_id))
    return [chunk for chunk, _score in hits]


def search(
    store,
    query: str,
    *,
    player_hero: str = "",
    position=None,
    phase: str = "",
    match_heroes: Iterable[str] = (),
    heroes: Iterable[str] = (),
    items: Iterable[str] = (),
    topic: str = "",
    top_k: int = 4,
    exclude: Iterable[str] = (),
    shown: Iterable[str] = (),
) -> SearchResult:
    """Search *store* with a free-text question; see the module docstring.

    *match_heroes* (any identifier form) restricts hero scoping to this
    match; empty = no restriction. *exclude* holds chunk ids that must not
    be returned. *shown* holds chunk ids the caller has already shown: they
    are not returned either and do not use up *top_k*, but the ones the
    search found again are listed in ``already_shown``. At most
    ``min(top_k, 6)`` hits, one per dedupe key, at most two from untrusted
    sources.
    """
    from .. import data_manager, knowledge_store as ks
    from . import runtime

    query = " ".join(str(query or "").split())[:MAX_QUERY_CHARS]
    top_k = max(1, min(int(top_k or 4), MAX_TOP_K))
    excluded = set(exclude or ())
    result = SearchResult()

    named_heroes, named_items = find_entities(query)
    for hero in heroes or ():
        if data_manager.is_known_hero(hero):
            named_heroes.append(data_manager.hero_slug(hero))
    for item in items or ():
        key = data_manager.resolve_item_key(item)
        if key:
            named_items.append(key)
    named_heroes, named_items = _dedupe(named_heroes), _dedupe(named_items)
    allowed = {data_manager.hero_slug(h) for h in match_heroes or () if h}
    player = data_manager.hero_slug(player_hero) if player_hero else ""
    if allowed:
        result.ignored_heroes = [h for h in named_heroes if h not in allowed and h != player]
        named_heroes = [h for h in named_heroes if h in allowed or h == player]

    # ``tags`` decides what passes the entity gate and which role and topic
    # a chunk is checked against; ``named`` holds only what the question
    # itself asked about (the tag ranking).
    tags: dict[str, float] = {"hero:all": _W_UNIVERSAL}
    named: dict[str, float] = {}

    def add(tag: str, weight: float, asked: bool = False) -> None:
        tag = ks.normalize_tag(tag)
        if not tag:
            return
        for target in (tags, named) if asked else (tags,):
            if weight > target.get(tag, 0.0):
                target[tag] = weight

    if player:
        add(f"hero:{player}", _W_PLAYER_HERO)
    for hero in named_heroes:
        add(f"hero:{hero}", _W_NAMED_HERO, asked=True)
        if player and hero != player:
            add(f"matchup:{player}:{hero}", _W_MATCHUP, asked=True)
            add(f"matchup:{hero}:{player}", _W_REVERSE_MATCHUP, asked=True)
    for key in named_items:
        add(f"item:{key}", _W_ITEM, asked=True)
    topic_tags, topic_kind = situations.SEARCH_TOPICS.get(str(topic or "").strip().lower(), ((), ""))
    result.heroes = [f"hero:{h}" for h in named_heroes]
    result.items = [f"item:{k}" for k in named_items]
    if result.ignored_heroes and not (named_heroes or named_items or topic_tags):
        # The question was about heroes that are not in this match and names
        # nothing else: whatever ranked first would be unrelated to it.
        result.reason = "heroes_not_in_match"
        return result
    for tag in topic_tags:
        add(tag, _W_TOPIC, asked=True)
    asked_topics = situations.question_topics(query)
    # Farm and item-timing statistics answer questions about farming and
    # item builds only, there also for the heroes the question names; for
    # anything else they are halved, and those of other heroes are out.
    economy = bool(_ECONOMY_QUESTION.intersection([*topic_tags, *asked_topics]))
    scope: list[str] = [ks.ECONOMY_TAG if economy else ks.FIGHT_TAG]
    add(scope[-1], _W_SCOPE_ONLY)
    for hero in ([player] if player else []) + (named_heroes if economy else []):
        scope.append(f"buyer:{hero}")
        add(scope[-1], _W_SCOPE_ONLY)
    try:
        scope.append(_ROLE_TAGS.get(int(position), ""))
        add(scope[-1], _W_ROLE)
    except (TypeError, ValueError):
        pass
    if phase:
        scope.append(f"phase:{phase}")
        add(scope[-1], _W_PHASE)

    # Keyword rankings (see the module docstring).
    for tag in asked_topics:
        add(tag, _W_TOPIC, asked=True)
    if not named:   # the question names nothing: rank by what the decision is about
        named = dict(tags)
    asked = set(named)
    for tag in scope:
        add(tag, _W_SCOPE_ONLY, asked=True)
    dense = None
    if topic_kind:
        dense = runtime.query_scores(store, situations.situation_text(position, phase, topic_kind))
    # A chunk ranks here for a tag the question asked for (or, with the
    # situation vectors of a topic, for its closeness to that): when no
    # chunk carries an asked tag, the scoping tags alone must not rank.
    rankings = [[chunk for chunk, _score in store.rank(named, "", dense=dense) if dense or asked & chunk.tags],
                _keyword_ranking(store, query, tags)]
    if dense:
        result.mode = "hybrid"

    # Dense ranking: embed the question with the model loaded in this process.
    embedder, reason = runtime.get_embedder()
    index = runtime.vector_index()
    np = vectors.numpy_or_none()
    if embedder is None or index is None or np is None:
        result.reason = reason or "deps_missing"
    elif "*" not in embedder.languages and not _is_english(query):
        result.reason = "non_english_query"
    else:
        try:
            # Like the keyword ranking: the decision's role and the topic
            # scope the candidates, its phase does not.
            scopes = [(ns, wanted) for ns, wanted in
                      ((ns, {t for t in tags if t.startswith(ns)}) for ns in ("role:", "topic:")) if wanted]
            candidates = [c for c in store.chunks()
                          if getattr(c, "index_text", "") and ks._entities_match(c.tags, tags)
                          and not any(ks._scope_mismatch(c.tags, ns, wanted) for ns, wanted in scopes)]
            keys = [vectors.vector_key(index.model_id, vectors.DOC, c.index_text) for c in candidates]
            positions, matrix = index.matrix(keys)
            if matrix is None:
                result.reason = "no_vectors"
            else:
                vector = np.asarray(embedder.embed_queries([query])[0], dtype=np.float32)
                cosines = (matrix @ vector).tolist()
                dense_order = sorted(zip(positions, cosines),
                                     key=lambda pc: (-pc[1], candidates[pc[0]].chunk_id))
                rankings.append([candidates[position_] for position_, _cos in dense_order])
                result.mode, result.reason = "hybrid", ""
        except Exception as exc:
            logger.warning("Query embedding failed, searching lexically: %s", runtime.redact(exc))
            result.reason = getattr(exc, "reason", "") or "embed_failed"
    if result.mode == "hybrid":
        result.reason = ""
    fused = _fuse(*rankings)

    shown_ids = set(shown or ())
    seen_keys: set[str] = set()
    untrusted = 0
    for chunk, score in fused:
        if chunk.chunk_id in excluded or (chunk.dedupe_key and chunk.dedupe_key in seen_keys):
            continue
        if chunk.dedupe_key:
            seen_keys.add(chunk.dedupe_key)
        if chunk.chunk_id in shown_ids:
            if len(result.already_shown) < MAX_TOP_K:
                result.already_shown.append(chunk.chunk_id)
            continue
        if not chunk.trusted:
            if untrusted >= ks._MAX_UNTRUSTED:
                continue
            untrusted += 1
        result.hits.append((chunk, score))
        if len(result.hits) >= top_k:
            break
    return result


__all__ = ["MAX_QUERY_CHARS", "MAX_TOP_K", "find_entities", "search"]
