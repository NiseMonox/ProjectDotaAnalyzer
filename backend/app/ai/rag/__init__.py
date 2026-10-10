"""Retrieval stack of the AI coach's knowledge base (the "RAG" package).

What it adds to the tag + BM25 ranker of :mod:`knowledge_store`:

- a dense term for the *generic* chunks (principles, role / phase notes,
  untagged tips), read from prebuilt vector files — numpy only, no model in
  the server (:mod:`.vectors`, :mod:`.runtime`, :mod:`.situations`);
- embedding backends for building those files and for the agent's free-text
  search (:mod:`.embedders`, :mod:`.freetext`);
- a labelled retrieval benchmark (:mod:`.evalkit`).

Importing this package needs only the standard library; numpy, onnxruntime,
tokenizers and openai are imported lazily and are optional. The functions
below are the interface other modules use; they never raise — when anything
is missing, retrieval is lexical.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_LEXICAL_STATUS = {
    "mode": "lexical", "embedder": None, "vectors": 0, "chunks": 0,
    "static_coverage": 0.0, "building": False, "last_build": None, "error": None,
    "embedder_available": False, "embedder_reason": "",
}


def status(store=None) -> dict:
    """State of the retrieval stack for ``/api/admin/status``.

    ``{"mode": "hybrid"|"lexical", "embedder": str|None, "vectors": int,
    "chunks": int, "static_coverage": float, "building": bool,
    "last_build": iso|None, "error": str|None, "embedder_available": bool,
    "embedder_reason": str}``. ``mode`` is what a decision would be ranked
    with right now: ``hybrid`` when the knowledge stores are built and their
    principle chunks have usable vectors, ``lexical`` otherwise (also before
    the warm-up or the first report built the stores). With *store* (a
    knowledge store the caller holds) the answer describes ranking over
    that store, whatever happened to the cached ones in the meantime.
    ``embedder`` names the vector space (model id) in use; ``chunks`` counts
    the built store; ``static_coverage`` is the share of the editorial +
    built-in chunks that have a vector. ``embedder_available`` says whether
    this process can embed new texts (learned chunks, the agent's own
    questions): False on an install without the optional packages or the
    model, where ``embedder_reason`` is the code why (``deps_missing``,
    ``model_missing``, ``off``, ``api_not_configured`` ...) — the committed
    vectors still give hybrid ranking then. ``error`` is a short
    code, sometimes followed by ``: detail`` — ``numpy_missing``,
    ``unknown_model``, ``api_not_configured``, ``model_missing``,
    ``deps_missing``, ``api_unreachable``, ``embedder_mismatch``,
    ``low_coverage``, ``no_query_vectors``, ``degenerate_vectors`` — or the
    name of an unusable vector file with the reason. The embeddings API key never
    appears in it.
    """
    try:
        from . import runtime
        full = runtime.details(store)
        return {key: full[key] for key in runtime.STATUS_KEYS}
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        try:
            from . import runtime
            text = runtime.redact(text)
        except Exception:  # pragma: no cover - the runtime itself is what failed
            pass
        logger.warning("RAG status unavailable: %s", text)
        return {**_LEXICAL_STATUS, "error": text}


def warm_up() -> None:
    """Build the stores and any missing vectors in the calling thread.

    Call it from a daemon thread at server start: without an embedding
    model it returns within a fraction of a second, with one it embeds the
    chunks the vector files do not cover yet.
    """
    try:
        from . import runtime
        runtime.warm_up()
    except Exception as exc:
        logger.warning("RAG warm-up failed: %s", exc)


def refresh(reason: str) -> None:
    """Tell the stack that stores, learned data or game data changed.

    Returns at once. In a warmed-up process a daemon thread embeds the new
    texts; elsewhere only cached lookups are dropped.
    """
    try:
        from . import runtime
        runtime.refresh(str(reason or ""))
    except Exception as exc:
        logger.warning("RAG refresh failed: %s", exc)


def shutdown() -> None:
    """Stop a running vector build and release the embedding model."""
    try:
        from . import runtime
        runtime.shutdown()
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("RAG shutdown failed: %s", exc)


def situation_scores(store, ctx) -> tuple[str, dict | None]:
    """``(situation query, {chunk_id: cosine} or None)`` for a decision context.

    The scores cover the store's generic chunks that have a vector; None
    means "rank lexically".
    """
    try:
        from . import runtime
        return runtime.situation_scores(store, ctx)
    except Exception as exc:
        logger.warning("Dense scoring failed, ranking lexically: %s", exc)
        return "", None


def search(store, query: str, **kwargs):
    """Free-text knowledge search (see :func:`rag.freetext.search`); never raises."""
    from .types import SearchResult
    try:
        from . import freetext
        return freetext.search(store, query, **kwargs)
    except Exception as exc:
        logger.warning("Knowledge search failed: %s", exc)
        return SearchResult(reason="error")


def top_k() -> int:
    """``RAG_TOP_K``: cap on retrieved entries per decision (0 = no cap)."""
    try:
        from . import runtime
        return runtime.settings().top_k
    except Exception:
        return 0


def agent_search_limit() -> int:
    """``RAG_AGENT_SEARCH_LIMIT``: free-text searches allowed per agent run."""
    try:
        from . import runtime
        return runtime.settings().agent_search_limit
    except Exception:
        return 3


__all__ = ["agent_search_limit", "refresh", "search", "shutdown", "situation_scores", "status",
           "top_k", "warm_up"]
