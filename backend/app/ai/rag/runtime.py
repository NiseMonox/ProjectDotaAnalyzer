"""Process-wide state of the retrieval stack: settings, vector index, embedder, builds.

How a report is served: :func:`situation_scores` looks the decision's
situation query and the store's generic chunks up in the vector files and
returns their cosine similarities — a file read at first use and a small
matrix product afterwards. No model is loaded for that. When numpy, the
vector files or the query vector are missing, when too few of the principle
chunks have a vector or the vectors cannot tell them apart, it returns None
and the caller ranks lexically (:func:`details` says why).

Decision ranking needs no embedder, only vectors of one model for the chunks
and the query. It reads the configured vector space and, when that one
cannot be used (an embeddings API that has not embedded the corpus yet or is
unreachable, an unknown ``RAG_MODEL``), the committed file of the default
model.

How vectors get built: :func:`sync` embeds what the files do not hold yet
(chunks changed by a game-data update, learned chunks) into the runtime
cache. It runs from the CLI, from :func:`warm_up` (a daemon thread at server
start) and from :func:`refresh` (a daemon thread of its own), only when an
embedder is available — never inside a request.

Settings are read from the environment on every call (tests change them):

``RAG_EMBEDDINGS``  auto | off | onnx | openai | hash (default auto)
``RAG_MODEL``       registry key of the local model
``RAG_MODEL_DIR``   directory holding ``<model>/`` (default backend/data/models)
``RAG_EMBED_API_BASE_URL`` / ``RAG_EMBED_API_KEY`` / ``RAG_EMBED_API_MODEL``
``RAG_INDEX_DIR``   runtime vector cache (default cache/rag)
``RAG_TOP_K``       cap on retrieved entries per decision (0 = no cap)
``RAG_AGENT_SEARCH_LIMIT``  free-text searches per agent run (default 3, at most 6; 0 = off)
``RAG_EMBED_THREADS`` / ``RAG_MODEL_IDLE_UNLOAD_S``  local model tuning
"""
from __future__ import annotations

import logging
import os
import threading
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import embedders, situations, vectors

logger = logging.getLogger(__name__)

VALID_MODES = ("auto", "off", "onnx", "openai", "hash")
# Upper bound of RAG_AGENT_SEARCH_LIMIT: every search is a tool message of up
# to 2,400 characters in one agent run (99 was accepted and ran 99 searches).
MAX_AGENT_SEARCH_LIMIT = 6
# Directory of the committed vector files; None = <backend/data>/rag.
STATIC_DIR: Path | None = None
# Dense ranking is used only when at least this share of the store's built-in
# and editorial principle chunks has a vector. Below it the dense term
# mis-ranks more than it helps (measured on the benchmark by removing vectors
# at random: nDCG@10 0.87 at full coverage, 0.86 at 80 %, below the lexical
# 0.84 at 10 %); the usual cause is a stale or half-built vector file.
MIN_POOL_COVERAGE = 0.8
# Cosine of a re-embedded text to its stored vector (:func:`embedder_consistency`):
# on the machine type that built the file anything lower means other
# parameters or another model; elsewhere (other int8 kernels) small
# differences are expected and tolerated down to the second value.
SAME_ARCH_MIN_COSINE = 0.999
OTHER_ARCH_MIN_COSINE = 0.98
_STATIC_SOURCE_TYPES = ("builtin", "editorial")
_warned_modes: set[str] = set()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    embeddings: str
    model: str
    model_dir: Path
    index_dir: Path
    static_dir: Path
    api_base_url: str
    api_key: str
    api_model: str
    top_k: int
    agent_search_limit: int
    embed_threads: int
    idle_unload_s: float

    @property
    def api_configured(self) -> bool:
        return bool(self.api_base_url and self.api_key)


def _project_dirs() -> tuple[Path, Path, Path]:
    """``(project root, cache dir, data dir)`` from the app configuration."""
    try:
        from ... import config
        return config.BASE_DIR, config.CACHE_DIR, config.DATA_DIR
    except Exception:  # pragma: no cover - configuration unavailable
        base = Path(__file__).resolve().parents[4]
        return base, base / "cache", base / "backend" / "data"


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or "").strip() or default


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(_env(name, str(default))))
    except ValueError:
        return default


def _env_path(name: str, default: Path, base: Path) -> Path:
    raw = _env(name)
    if not raw:
        return default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else base / path


def settings() -> Settings:
    base, cache_dir, data_dir = _project_dirs()
    mode = _env("RAG_EMBEDDINGS", "auto").lower()
    if mode not in VALID_MODES:
        if mode not in _warned_modes:   # settings() runs for every ranking
            _warned_modes.add(mode)
            logger.warning("Invalid RAG_EMBEDDINGS=%r, using auto", mode)
        mode = "auto"
    try:
        idle = max(0.0, float(_env("RAG_MODEL_IDLE_UNLOAD_S", "900")))
    except ValueError:
        idle = 900.0
    return Settings(
        embeddings=mode,
        model=_env("RAG_MODEL", embedders.DEFAULT_MODEL),
        model_dir=_env_path("RAG_MODEL_DIR", data_dir / "models", base),
        index_dir=_env_path("RAG_INDEX_DIR", cache_dir / "rag", base),
        static_dir=STATIC_DIR if STATIC_DIR is not None else data_dir / "rag",
        api_base_url=_env("RAG_EMBED_API_BASE_URL"),
        api_key=_env("RAG_EMBED_API_KEY"),
        api_model=_env("RAG_EMBED_API_MODEL"),
        top_k=_env_int("RAG_TOP_K", 0),
        agent_search_limit=min(MAX_AGENT_SEARCH_LIMIT, _env_int("RAG_AGENT_SEARCH_LIMIT", 3)),
        embed_threads=_env_int("RAG_EMBED_THREADS", 1, minimum=1),
        idle_unload_s=idle,
    )


def static_path(cfg: Settings, model_id: str) -> Path:
    return cfg.static_dir / f"{vectors.file_slug(model_id)}.npz"


def cache_path(cfg: Settings, model_id: str) -> Path:
    return cfg.index_dir / f"{vectors.file_slug(model_id)}.npz"


def redact(text: object, cfg: Settings | None = None) -> str:
    """*text* without the embeddings API key.

    Error texts are logged and served by ``/api/admin/status``; an exception
    raised around an API call can carry the key (a provider echoing the
    request, a URL with credentials).
    """
    text = str(text)
    try:
        key = (cfg or settings()).api_key
    except Exception:  # pragma: no cover - settings never raise
        key = ""
    return text.replace(key, "***") if len(key) >= 8 else text


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

_lock = threading.RLock()
# What the last decision was ranked with ("hybrid" | "lexical"); a change is
# logged (see _note_ranking_mode).
_ranking_mode: str | None = None
_indexes: dict[tuple, vectors.VectorIndex] = {}     # (model id, committed path, cache path) -> index
_embedder_entry: tuple[tuple, object, str] | None = None
# store -> {model id: ((store revision, index identity, index version), pool)}
_pools: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_coverage_entry: tuple[tuple, float] | None = None


class _BuildState:
    def __init__(self) -> None:
        self.lock = threading.Lock()        # held by the one thread that is syncing
        self.armed = False                  # warm_up() ran: refresh() may start builds
        self.building = False
        self.dirty = False                  # a refresh arrived while a sync was running
        self.last_build: str | None = None  # ISO time of the last finished sync
        self.error: str | None = None
        self.last_stats: dict = {}
        self.stop = threading.Event()
        self.threads: list[threading.Thread] = []   # background syncs started by refresh()


_build = _BuildState()


def _file_expectations(model_id: str) -> tuple[int, str]:
    """``(dimension, embedder parameter fingerprint)`` a vector file of *model_id* must have."""
    spec = embedders.MODEL_REGISTRY.get(model_id)
    return (spec.dim, spec.params) if spec else (0, "")


def _index(cfg: Settings, model_id: str) -> vectors.VectorIndex | None:
    """The merged vector index (committed file + runtime cache) of one model."""
    if not model_id or cfg.embeddings == "off" or vectors.numpy_or_none() is None:
        return None
    signature = (model_id, static_path(cfg, model_id), cache_path(cfg, model_id))
    with _lock:
        index = _indexes.get(signature)
        if index is None:
            dim, params = _file_expectations(model_id)
            index = vectors.VectorIndex(model_id, signature[1], signature[2],
                                        expected_dim=dim, expected_params=params)
            _indexes[signature] = index
        return index


def vector_index(cfg: Settings | None = None) -> vectors.VectorIndex | None:
    """The vector index of the configured model, or None (off / no numpy)."""
    cfg = cfg or settings()
    return _index(cfg, embedders.active_model_id(cfg))


def decision_indexes(cfg: Settings | None = None) -> list[vectors.VectorIndex]:
    """Vector spaces decision ranking may read, best first.

    The configured one, then the default model's (its committed file serves
    reports while the configured space has no usable vectors).
    """
    cfg = cfg or settings()
    found = (_index(cfg, model_id)
             for model_id in dict.fromkeys((embedders.active_model_id(cfg), embedders.DEFAULT_MODEL)))
    return [index for index in found if index is not None]


def get_embedder(cfg: Settings | None = None) -> tuple[object | None, str]:
    """``(embedder, "")`` or ``(None, reason)``; one instance per configuration."""
    global _embedder_entry
    cfg = cfg or settings()
    signature = (cfg.embeddings, cfg.model, str(cfg.model_dir), cfg.api_base_url, cfg.api_key,
                 cfg.api_model, cfg.embed_threads, cfg.idle_unload_s)
    with _lock:
        entry = _embedder_entry
        if entry is not None and entry[0] == signature and entry[1] is not None:
            return entry[1], ""
        embedder, reason = embedders.resolve_embedder(cfg)
        if entry is not None and entry[1] is not None and entry[1] is not embedder:
            try:
                entry[1].close()
            except Exception:  # pragma: no cover - best effort
                pass
        _embedder_entry = (signature, embedder, reason)
        return embedder, reason


def reset() -> None:
    """Forget every cached object (tests, and after a configuration change)."""
    global _embedder_entry, _coverage_entry, _ranking_mode
    with _lock:
        entry, _embedder_entry = _embedder_entry, None
        _indexes.clear()
        _coverage_entry = None
        _ranking_mode = None
        _pools.clear()
    _warned_modes.clear()
    if entry is not None and entry[1] is not None:
        try:
            entry[1].close()
        except Exception:  # pragma: no cover - best effort
            pass
    _build.armed = False
    _build.dirty = False
    _build.error = None
    _build.last_build = None
    _build.last_stats = {}
    _build.stop.clear()


# ---------------------------------------------------------------------------
# Dense scores for a decision (the pooled term of the ranker)
# ---------------------------------------------------------------------------

def _index_texts(store) -> dict[str, str]:
    """``{chunk_id: canonical English text}`` of the chunks that have one."""
    return {c.chunk_id: c.index_text for c in store.chunks() if getattr(c, "index_text", "")}


@dataclass(frozen=True)
class _Pool:
    """The store's generic chunks that have a vector in one index."""

    ids: tuple            # chunk ids, row by row
    matrix: object        # float32 [len(ids), dim], or None when no chunk has a vector
    covered: int          # of the reference chunks, how many have a vector ...
    total: int            # ... and how many there are

    @property
    def coverage(self) -> float:
        return self.covered / self.total if self.total else 0.0


def _generic_pool(store, index: vectors.VectorIndex) -> _Pool:
    """Vectors of the store's generic chunks, and how much of the principles they cover.

    Coverage is counted over the built-in and editorial generic chunks (all
    generic chunks when the store has none of those): the committed file is
    meant to hold every one of them. Learned and operator chunks without a
    vector are the normal state of a server without an embedding model; the
    ranker gives those a neutral dense credit.
    """
    from .. import knowledge_store

    key = (getattr(store, "revision", None), id(index), index.version)
    with _lock:
        cached = (_pools.get(store) or {}).get(index.model_id)
    if cached is not None and cached[0] == key:
        return cached[1]
    generic = [c for c in store.chunks()
               if getattr(c, "index_text", "") and not knowledge_store.is_entity_scoped(c.tags)]
    keys = [vectors.vector_key(index.model_id, vectors.DOC, c.index_text) for c in generic]
    positions, matrix = index.matrix(keys)
    reference = {i for i, c in enumerate(generic) if c.source_type in _STATIC_SOURCE_TYPES}
    reference = reference or set(range(len(generic)))
    pool = _Pool(ids=tuple(generic[p].chunk_id for p in positions), matrix=matrix,
                 covered=len(reference.intersection(positions)), total=len(reference))
    with _lock:
        _pools.setdefault(store, {})[index.model_id] = (key, pool)
    return pool


def _pool_scores(store, index: vectors.VectorIndex, situation_text: str) -> tuple[dict[str, float] | None, str]:
    """``(scores, "")`` of one vector space for a situation query, or ``(None, reason)``.

    Reasons: ``no_query_vector`` and ``no_vectors`` (nothing built for this
    space: the normal state before a build), ``low_coverage``,
    ``dimension_mismatch`` and ``degenerate_vectors`` (a problem with the
    files; :func:`details` reports it).
    """
    query = index.get(vectors.vector_key(index.model_id, vectors.QUERY, situation_text))
    if query is None:
        return None, "no_query_vector"
    pool = _generic_pool(store, index)
    if pool.matrix is None or not pool.ids:
        return None, "no_vectors"
    if pool.matrix.shape[1] != query.shape[0]:
        return None, "dimension_mismatch"
    if pool.coverage < MIN_POOL_COVERAGE:
        return None, f"low_coverage: {pool.covered} of {pool.total} principle chunks have a vector"
    np = vectors.numpy_or_none()
    scores = pool.matrix @ query
    if not bool(np.isfinite(scores).all()) or float(scores.max() - scores.min()) < 1e-6:
        return None, "degenerate_vectors: the vectors do not tell the principle chunks apart"
    return dict(zip(pool.ids, scores.tolist())), ""


def _note_ranking_mode(mode: str, detail: str) -> None:
    """Log the mode decisions are ranked with, once per change.

    Nothing else tells an operator that reports quietly went lexical (a
    vector file that became unreadable, a game-data update that left too
    few principle chunks covered) on a server that runs without the
    start-time warm-up, or long after it.
    """
    global _ranking_mode
    if mode == _ranking_mode:
        return
    with _lock:
        previous, _ranking_mode = _ranking_mode, mode
    if previous == mode:
        return
    logger.log(logging.WARNING if (previous, mode) == ("hybrid", "lexical") else logging.INFO,
               "Knowledge retrieval: decisions are now ranked %s (%s)",
               "with vectors (hybrid)" if mode == "hybrid" else "lexically", detail)


def query_scores(store, situation_text: str) -> dict[str, float] | None:
    """Cosine of every generic chunk (that has a vector) to a situation query.

    None when dense ranking is not possible or not worth it: embeddings off,
    numpy or the vector files missing, no vector for this query, fewer than
    :data:`MIN_POOL_COVERAGE` of the principle chunks covered, or vectors
    that give every chunk the same score.
    """
    if not situation_text:
        return None
    cfg = settings()
    problem = ""
    this_query_only = False
    for index in decision_indexes(cfg):
        scores, reason = _pool_scores(store, index, situation_text)
        if scores is not None:
            _note_ranking_mode("hybrid", f"vector space {index.model_id}, {len(index)} vectors")
            return scores
        if reason == "no_query_vector" and len(index):
            this_query_only = True      # says nothing about the next decision
        elif reason not in ("no_vectors", "no_query_vector"):
            problem = problem or reason
    if not this_query_only:
        _note_ranking_mode("lexical", problem or (
            "RAG_EMBEDDINGS=off" if cfg.embeddings == "off"
            else "numpy is not installed" if vectors.numpy_or_none() is None else "no vectors for this knowledge base"))
    return None


def situation_scores(store, ctx) -> tuple[str, dict[str, float] | None]:
    """``(situation query text, dense scores or None)`` for a decision context."""
    query = situations.describe(ctx)
    if query is None:
        return "", None
    return query.text, query_scores(store, query.text)


# ---------------------------------------------------------------------------
# Building vectors
# ---------------------------------------------------------------------------

def _query_wanted() -> list[tuple[str, str]]:
    return [(vectors.QUERY, text) for text in situations.situation_queries()]


def build_static(embedder=None, path: Path | None = None) -> dict:
    """(Re)build the committed vector file: static corpus + situation queries.

    The static corpus is what :func:`knowledge_base.static_index_texts`
    returns for the data that is loaded right now; the CLI points the game
    data at the committed files first. Raises when no embedder is available.
    """
    from .. import knowledge_base

    cfg = settings()
    if embedder is None:
        embedder, reason = get_embedder(cfg)
        if embedder is None:
            raise embedders.EmbedderError(reason, f"no embedder available ({reason})")
    wanted = [(vectors.DOC, text) for text in knowledge_base.static_index_texts().values()]
    wanted += _query_wanted()
    target = path or static_path(cfg, embedder.id)
    stats = vectors.sync_vector_file(target, embedder, wanted)
    stats["path"] = str(target)
    index = vector_index(cfg)
    if index is not None:
        index.reload()
    return stats


def _discard_unusable_cache(cfg: Settings) -> None:
    """Remove the configured model's runtime cache file when it cannot be read."""
    model_id = embedders.active_model_id(cfg)
    if not model_id or vectors.numpy_or_none() is None:
        return
    if vectors.discard_unusable(cache_path(cfg, model_id), model_id, *_file_expectations(model_id)):
        index = vector_index(cfg)
        if index is not None:
            index.reload()


def _require_consistent(embedder) -> None:
    """Raise unless *embedder* reproduces the committed vectors of its model.

    New vectors are only worth storing next to the committed ones when both
    come from the same space (see :func:`embedder_consistency`). Checked
    once per embedder instance; nothing to check when the model has no
    committed file. :func:`build_static` is not bound by it: that is how the
    committed file is rebuilt on this machine.
    """
    if getattr(embedder, "_reproduces_committed", False):
        return
    check = embedder_consistency(embedder)
    if check["checked"] and not check["ok"]:
        raise embedders.EmbedderError(
            "embedder_mismatch",
            f"cosine {check['min_cosine']} to the committed vectors (built on {check['file_arch'] or '?'}, "
            f"this machine is {check['arch']}; required {check['threshold']}); "
            "see `python -m backend.scripts.rag doctor`")
    try:
        embedder._reproduces_committed = True
    except AttributeError:  # pragma: no cover - an embedder that takes no attributes is checked every time
        pass


def _sync_once(reason: str) -> dict:
    """One pass: build the stores, embed what is missing into the runtime cache."""
    from .. import knowledge_base

    cfg = settings()
    stores = knowledge_base.warm_stores()   # also without embeddings: the first request finds them built
    if cfg.embeddings == "off":
        return {"not_run": "off"}
    _discard_unusable_cache(cfg)            # also without an embedder: a broken cache must not stay reported
    embedder, why = get_embedder(cfg)
    if embedder is None:
        return {"not_run": why}
    if vectors.numpy_or_none() is None:
        return {"not_run": "deps_missing"}
    texts: set[str] = set()
    for store in stores.values():
        if store is not None:
            texts.update(_index_texts(store).values())
    wanted = [(vectors.DOC, text) for text in sorted(texts)] + _query_wanted()
    committed = vectors.VectorIndex(embedder.id, static_path(cfg, embedder.id), None,
                                    expected_dim=int(getattr(embedder, "dim", 0) or 0),
                                    expected_params=str(getattr(embedder, "params", "") or ""))
    target = cache_path(cfg, embedder.id)
    # One builder per cache file across processes (this server's warm-up, a
    # `rag build`, a learning run): whoever waits finds the other's vectors
    # in the file afterwards and embeds only what is still missing.
    with vectors.build_lock(target, should_stop=_build.stop.is_set) as held:
        if not held:
            return {"not_run": "stopped"}
        _require_consistent(embedder)
        stats = vectors.sync_vector_file(
            target, embedder, wanted,
            known=committed.__contains__, should_stop=_build.stop.is_set)
    index = vector_index(cfg)
    if index is not None:
        index.reload()
    if stats["embedded"] or stats["dropped"]:
        logger.info("RAG vectors synced (%s): %d embedded, %d dropped, %d in the committed file, %.1f s",
                    reason, stats["embedded"], stats["dropped"], stats["skipped"], stats["seconds"])
    return stats


def sync(reason: str = "manual") -> dict:
    """Bring the runtime vector cache up to date (blocking; one sync at a time).

    Returns the statistics of the last pass, ``{"not_run": reason}`` when
    nothing could be built (``busy``: another thread is syncing and will run
    one more pass; ``stopped``: :func:`shutdown` came while waiting for
    another process's build) or ``{"error": text}``. Never raises. While
    another process builds the same cache file this waits for it (see
    :func:`vectors.build_lock`) and then embeds only what that left out.
    """
    if not _build.lock.acquire(blocking=False):
        _build.dirty = True
        return {"not_run": "busy"}
    stats: dict = {}
    try:
        _build.building = True
        while True:
            _build.dirty = False
            try:
                stats = _sync_once(reason)
                _build.error = None
            except Exception as exc:
                reason_code = getattr(exc, "reason", "")
                _build.error = redact(f"{reason_code}: {exc}" if reason_code else f"{type(exc).__name__}: {exc}")
                logger.warning("RAG vector sync failed (%s): %s", reason, _build.error)
                stats = {"error": _build.error}
            _build.last_stats = stats
            _build.last_build = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if not _build.dirty or _build.stop.is_set():
                break
    finally:
        _build.building = False
        _build.lock.release()
    return stats


def warm_up() -> None:
    """Build the knowledge stores and whatever vectors are missing, in this thread.

    Meant for a daemon thread at server start (and for the CLI). Without an
    embedder it only builds the stores and loads the committed vectors,
    which takes a fraction of a second.
    """
    _build.stop.clear()
    _build.armed = True
    sync("warm_up")
    try:
        for index in decision_indexes():
            len(index)   # read the files now instead of in the first request
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Vector preload failed: %s", exc)


def refresh(reason: str) -> None:
    """Note that stores / learned data / game data changed; returns at once.

    Cached pools are dropped. When this process was warmed up (a server),
    a daemon thread embeds the new texts; other processes (tests, one-shot
    scripts) build nothing here — a CLI calls :func:`sync` instead.
    """
    global _coverage_entry
    with _lock:
        _pools.clear()
        _coverage_entry = None
    if not _build.armed or settings().embeddings == "off":
        return
    thread = threading.Thread(target=sync, args=(reason,), name="rag-sync", daemon=True)
    with _lock:
        _build.threads = [t for t in _build.threads if t.is_alive()] + [thread]
    thread.start()


def wait_idle(timeout: float = 30.0) -> bool:
    """Wait until no sync is running or queued; False when *timeout* seconds were not enough."""
    with _lock:
        threads = list(_build.threads)
    for thread in threads:
        thread.join(timeout)
    if not _build.lock.acquire(timeout=timeout):
        return False
    _build.lock.release()
    return not any(thread.is_alive() for thread in threads)


def shutdown() -> None:
    """Ask a running sync to stop after the current batch and release the model.

    Background syncs are off until the next :func:`warm_up`.
    """
    _build.armed = False
    _build.stop.set()
    with _lock:
        entry = _embedder_entry
    if entry is not None and entry[1] is not None:
        try:
            entry[1].close()
        except Exception:  # pragma: no cover - best effort
            pass


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def _static_coverage(index: vectors.VectorIndex | None) -> float:
    """Share of the static corpus (as loaded now) that has a vector; cached."""
    global _coverage_entry
    from .. import knowledge_base

    if index is None:
        return 0.0
    key = (knowledge_base.data_version(), id(index), index.version)
    with _lock:
        entry = _coverage_entry
    if entry is not None and entry[0] == key:
        return entry[1]
    value = round(vectors.coverage(index, index.model_id, knowledge_base.static_index_texts().values()), 4)
    with _lock:
        _coverage_entry = (key, value)
    return value


def _queries_ready(index: vectors.VectorIndex | None) -> int:
    if index is None:
        return 0
    return sum(1 for text in situations.situation_queries()
               if vectors.vector_key(index.model_id, vectors.QUERY, text) in index)


def _decision_state(store, cfg: Settings) -> tuple[vectors.VectorIndex | None, str]:
    """``(vector space decision ranking uses for *store*, "")`` or ``(None, why it is lexical)``.

    Judged with one situation query per space, through the same checks a
    ranking goes through. The reason is "" when nothing was built for any
    space (not a fault); otherwise it names what is wrong with the files.
    """
    problem = ""
    for index in decision_indexes(cfg):
        probe = next((text for text in situations.situation_queries()
                      if vectors.vector_key(index.model_id, vectors.QUERY, text) in index), None)
        if probe is None:
            if len(index) and not problem:
                problem = "no_query_vectors: the vector files hold no situation query"
            continue
        scores, reason = _pool_scores(store, index, probe)
        if scores is not None:
            return index, ""
        if not problem and reason not in ("no_vectors", "no_query_vector"):
            problem = reason
        elif not problem and reason == "no_vectors":
            problem = "low_coverage: no principle chunk has a vector"
    return None, problem


def _configuration_error(cfg: Settings, embedder, reason: str) -> str | None:
    """What makes the configured vector space unusable, as a status error (None = nothing)."""
    if cfg.embeddings == "off":
        return None
    if vectors.numpy_or_none() is None:
        return "numpy_missing"
    if embedder is None and reason in ("api_not_configured", "unknown_model"):
        return reason           # a setting is wrong, whatever the mode
    if embedder is None and cfg.embeddings in ("onnx", "openai"):
        return reason           # an explicitly requested backend is unavailable
    return None


def details(reference=None) -> dict:
    """Everything :func:`status` reports plus what the CLI shows. May raise.

    ``mode`` is what a decision would be ranked with right now: ``hybrid``
    only when the knowledge stores are built and their principle chunks have
    usable vectors, ``lexical`` otherwise — also before the stores exist
    (they are built by the warm-up or the first report). *reference* is the
    store to describe (default: the English one that is built right now).
    """
    from .. import knowledge_base

    cfg = settings()
    model_id = embedders.active_model_id(cfg)
    index = vector_index(cfg)
    if index is not None:
        len(index)              # read the files: index.error is only known after that
    embedder, reason = get_embedder(cfg)
    if reference is None:
        stores = knowledge_base.built_stores()
        reference = stores.get("en") or next((s for s in stores.values() if s is not None), None)
    used, problem = (None, "") if reference is None else _decision_state(reference, cfg)
    hybrid = used is not None
    shown = used if hybrid else index           # the space the numbers below describe
    error = (_configuration_error(cfg, embedder, reason) or _build.error
             or (index.error if index is not None else None)
             or (used.error if hybrid and used is not index else None) or problem or None)
    return {
        "mode": "hybrid" if hybrid else "lexical",
        "embedder": used.model_id if hybrid else (model_id if embedder is not None and model_id else None),
        "vectors": len(shown) if shown is not None else 0,
        "chunks": len(reference) if reference is not None else 0,
        "static_coverage": _static_coverage(used) if hybrid else 0.0,
        "building": bool(_build.building),
        "last_build": _build.last_build,
        "error": redact(error, cfg) if error else None,
        # CLI extras
        "setting": cfg.embeddings,
        "model_id": model_id,
        "embedder_available": embedder is not None,
        "embedder_reason": reason,
        "situation_queries": _queries_ready(shown),
        "numpy": vectors.numpy_or_none() is not None,
        "files": shown.files() if shown is not None else [],
        "static_path": str(static_path(cfg, model_id)) if model_id else "",
        "cache_path": str(cache_path(cfg, model_id)) if model_id else "",
        "model_dir": str(cfg.model_dir),
        "top_k": cfg.top_k,
        "agent_search_limit": cfg.agent_search_limit,
        "last_stats": dict(_build.last_stats),
    }


STATUS_KEYS = ("mode", "embedder", "vectors", "chunks", "static_coverage", "building", "last_build", "error",
               "embedder_available", "embedder_reason")


# ---------------------------------------------------------------------------
# Does this machine's embedder reproduce the committed vectors?
# ---------------------------------------------------------------------------

def embedder_consistency(embedder=None, documents: int = 20, queries: int = 5) -> dict:
    """Re-embed a sample of the committed file's texts and compare with the stored vectors.

    The committed file is built on one machine and served everywhere; new
    texts (learned chunks, the agent's questions) are embedded locally. A
    different pooling, prefix or model file — or, for an int8 model, other
    CPU kernels — would put the two into slightly different spaces.

    Returns ``{"checked", "min_cosine", "mean_cosine", "file_arch", "arch",
    "threshold", "ok"}``; ``checked`` is 0 (with ``reason``) when there is
    nothing to compare. The threshold is :data:`SAME_ARCH_MIN_COSINE` on the
    machine type that built the file, else :data:`OTHER_ARCH_MIN_COSINE`.
    Raises what the embedder raises.
    """
    import platform

    from .. import knowledge_base

    cfg = settings()
    if embedder is None:
        embedder, reason = get_embedder(cfg)
        if embedder is None:
            return {"checked": 0, "reason": reason}
    np = vectors.numpy_or_none()
    if np is None:
        return {"checked": 0, "reason": "numpy_missing"}
    committed = vectors.VectorIndex(embedder.id, static_path(cfg, embedder.id), None,
                                    expected_dim=int(getattr(embedder, "dim", 0) or 0),
                                    expected_params=str(getattr(embedder, "params", "") or ""))
    if not len(committed):
        return {"checked": 0, "reason": committed.error or "no_committed_file"}

    def sample(kind: str, texts: list[str], count: int) -> list[tuple[str, object]]:
        stored = [(text, committed.get(vectors.vector_key(embedder.id, kind, text))) for text in texts]
        stored = [(text, vector) for text, vector in stored if vector is not None]
        step = max(1, len(stored) // max(1, count))
        return stored[::step][:count]

    static_texts = knowledge_base.static_index_texts()
    docs = sample(vectors.DOC, [static_texts[chunk_id] for chunk_id in sorted(static_texts)], documents)
    asked = sample(vectors.QUERY, situations.situation_queries(), queries)
    fresh = (embedder.embed_documents([text for text, _ in docs]) if docs else []) + \
            (embedder.embed_queries([text for text, _ in asked]) if asked else [])
    cosines = [float(np.asarray(new, dtype=np.float32) @ old)
               for new, (_text, old) in zip(fresh, docs + asked)]
    if not cosines:
        return {"checked": 0, "reason": "no_committed_text"}
    files = committed.files()
    file_arch = files[0].get("arch", "") if files else ""
    arch = platform.machine()
    threshold = SAME_ARCH_MIN_COSINE if file_arch == arch else OTHER_ARCH_MIN_COSINE
    return {
        "checked": len(cosines),
        "min_cosine": round(min(cosines), 5),
        "mean_cosine": round(sum(cosines) / len(cosines), 5),
        "file_arch": file_arch,
        "arch": arch,
        "threshold": threshold,
        "ok": min(cosines) >= threshold,
    }


__all__ = [
    "MAX_AGENT_SEARCH_LIMIT", "MIN_POOL_COVERAGE", "OTHER_ARCH_MIN_COSINE", "SAME_ARCH_MIN_COSINE", "STATUS_KEYS",
    "Settings",
    "build_static", "cache_path", "decision_indexes", "details", "embedder_consistency", "get_embedder",
    "query_scores", "redact", "refresh", "reset", "settings", "shutdown", "situation_scores", "static_path",
    "sync", "vector_index", "wait_idle", "warm_up",
]
