"""Vector files: content-addressed embeddings in one ``.npz`` per model.

File ``<model slug>.npz`` (numpy, no pickle):

- ``keys``    — ``sha256(model id + "\\n" + kind + "\\n" + text)`` as hex, one
  per vector; ``kind`` is ``doc`` or ``query``. A text that changes gets a
  new key, so a stale vector can never be served for it.
- ``vectors`` — float16 ``[n, dim]``, L2-normalised.
- ``model``, ``dim``, ``format``, ``built_at`` — metadata.
- ``params`` — fingerprint of how the embedder produced the vectors (pooling,
  prefixes, model file; "" when the backend has none). The keys only name
  the model, so a file built with other parameters is refused as a whole.
- ``arch`` — machine the file was built on (``rag doctor`` compares the
  local model against it; an int8 model runs on other kernels elsewhere).

Two files exist per model: the committed one for the static corpus and the
situation queries (``backend/data/rag/``) and a runtime cache for everything
else (``cache/rag/``). :class:`VectorIndex` reads both lazily, merges them
and reloads when a file's mtime, size or identity changes (another process —
the CLI or a learning run — rebuilt it). Files are written through a temporary file
and an atomic replace; :func:`build_lock` keeps two processes from embedding
the same missing texts into one file at the same time. A file is read into
memory as a whole and closed again (no memory map): on Windows an open file
can be neither replaced nor removed.

numpy is imported inside the functions; without it :func:`numpy_or_none`
returns None and the callers stay lexical.
"""
from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import platform
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

from ... import osutil

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2
# A stored row must have this L2 norm within the tolerance (float16 rounding
# of a unit vector is about 1e-3).
NORM_TOLERANCE = 0.02
DOC = "doc"
QUERY = "query"
# How often (seconds) the files are checked for a change made by another process.
CHECK_INTERVAL_S = 2.0

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def numpy_or_none():
    """The numpy module, or None when it is not installed."""
    try:
        import numpy
        return numpy
    except ImportError:
        return None


def vector_key(model_id: str, kind: str, text: str) -> str:
    return hashlib.sha256(f"{model_id}\n{kind}\n{text}".encode("utf-8")).hexdigest()


def file_slug(model_id: str) -> str:
    """File name stem of a model id (``openai:text-embedding-3-small`` -> ``openai-text-...``)."""
    return _SLUG_RE.sub("-", model_id or "").strip("-") or "model"


class VectorFileError(ValueError):
    """A vector file exists but cannot be used."""


def _usable_rows(np, matrix):
    """Per row: finite and of unit length (what a cosine by dot product needs)."""
    norms = np.sqrt((np.asarray(matrix, dtype=np.float32) ** 2).sum(axis=1))
    return np.isfinite(norms) & (np.abs(norms - 1.0) <= NORM_TOLERANCE)


def read_vector_file(path: Path, model_id: str, expected_dim: int = 0, expected_params: str | None = None):
    """``(keys, matrix, meta)`` of a vector file.

    Raises :class:`FileNotFoundError` when it does not exist and
    :class:`VectorFileError` when it is unreadable, has another format
    version, was built with another model or (with *expected_params*) other
    embedder parameters, has an unexpected shape / dimension, or holds
    vectors that are not finite and L2-normalised.
    """
    np = numpy_or_none()
    if np is None:
        raise VectorFileError("numpy is not installed")
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = [k.decode("ascii") if isinstance(k, bytes) else str(k) for k in data["keys"].tolist()]
            matrix = data["vectors"]
            model = str(data["model"]) if "model" in data.files else ""
            version = int(data["format"]) if "format" in data.files else 0
            meta = {
                "model": model,
                "built_at": str(data["built_at"]) if "built_at" in data.files else "",
                "arch": str(data["arch"]) if "arch" in data.files else "",
                "params": str(data["params"]) if "params" in data.files else "",
            }
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise VectorFileError(f"{path.name}: unreadable ({type(exc).__name__}: {exc})") from exc
    if version != FORMAT_VERSION:
        raise VectorFileError(f"{path.name}: format {version}, this version reads format {FORMAT_VERSION}")
    if model != model_id:
        raise VectorFileError(f"{path.name}: built with model {model!r}, not {model_id!r}")
    if expected_params is not None and meta["params"] != expected_params:
        raise VectorFileError(f"{path.name}: built with other embedder parameters "
                              f"({meta['params'] or '-'}, expected {expected_params or '-'})")
    if matrix.ndim != 2 or matrix.shape[0] != len(keys) or matrix.dtype.kind != "f":
        raise VectorFileError(f"{path.name}: {len(keys)} keys for a {matrix.shape} {matrix.dtype} matrix")
    if expected_dim and matrix.shape[1] != expected_dim:
        raise VectorFileError(f"{path.name}: dimension {matrix.shape[1]}, expected {expected_dim}")
    bad = int((~_usable_rows(np, matrix)).sum()) if len(keys) else 0
    if bad:
        raise VectorFileError(f"{path.name}: {bad} of {len(keys)} vectors are not finite and L2-normalised")
    return keys, matrix, meta


def write_vector_file(path: Path, model_id: str, keys: Sequence[str], matrix, params: str = "") -> None:
    """Write a vector file atomically (float16; *matrix* is ``[len(keys), dim]``).

    *params* is the embedder's parameter fingerprint (see the module docstring).
    """
    np = numpy_or_none()
    if np is None:
        raise VectorFileError("numpy is not installed")
    matrix = np.asarray(matrix, dtype=np.float16)
    if matrix.ndim != 2 or matrix.shape[0] != len(keys):
        raise VectorFileError(f"{len(keys)} keys for a {matrix.shape} matrix")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(
                f,
                keys=np.asarray(list(keys), dtype="S64"),
                vectors=matrix,
                model=np.asarray(model_id),
                dim=np.asarray(matrix.shape[1]),
                format=np.asarray(FORMAT_VERSION),
                built_at=np.asarray(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
                arch=np.asarray(platform.machine()),
                params=np.asarray(str(params or "")),
            )
            f.flush()
            os.fsync(f.fileno())   # the rename below must not outlive the data after a power loss
        os.chmod(tmp, 0o644)   # mkstemp creates 0600; a CLI run under another user must stay readable
        osutil.atomic_replace(tmp, path)   # Windows: waits while a reader has the old file open
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# How often (seconds) a process waiting for build_lock() tries again.
LOCK_POLL_S = 0.2


@contextlib.contextmanager
def build_lock(path: Path, should_stop: Callable[[], bool] | None = None) -> Iterator[bool]:
    """Hold the build lock of the vector file at *path* across processes.

    A server's warm-up and a ``rag build`` / learning run started next to it
    would otherwise both embed every text the file lacks (two model loads,
    the work done twice, the slower one overwriting the other's file). The
    lock is :class:`osutil.FileLock` on ``<file name>.lock`` beside the file
    (``flock`` on POSIX, a byte-range lock on Windows); the OS releases it
    when the holder dies, so it cannot go stale. A second process waits —
    when it gets the lock the first one's vectors are in the file and
    :func:`sync_vector_file` keeps them.

    Yields True with the lock held, False when *should_stop* became true
    while waiting (nothing may be built then). Where no lock is possible
    (a directory that cannot be written, a file system without locks) it
    yields True unguarded: the build itself is still atomic per file.
    """
    path = Path(path)
    lock_path = path.with_name(path.name + ".lock")
    lock: osutil.FileLock | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = osutil.FileLock(lock_path)
    except OSError as exc:
        logger.warning("Cannot take the vector build lock %s: %s", lock_path, exc)
    try:
        if lock is not None:
            waiting = False
            while True:
                try:
                    if lock.acquire(blocking=False):
                        break
                except OSError as exc:  # no lock file, or a file system without locks: unguarded
                    logger.warning("Vector build lock unavailable for %s: %s", path.name, exc)
                    break
                if should_stop is not None and should_stop():
                    yield False
                    return
                if not waiting:
                    waiting = True
                    logger.info("Another process is building %s; waiting for it", path.name)
                time.sleep(LOCK_POLL_S)
        yield True
    finally:
        if lock is not None:
            lock.release()


# How long (seconds) remove_file() keeps trying on Windows, and how often.
_REMOVE_WAIT_S = 1.0
_REMOVE_POLL_S = 0.05


def remove_file(path: Path) -> None:
    """Remove *path* (no error when it is not there).

    On Windows a file cannot be removed while it is open, and another
    thread or process may be reading the vector file at this moment (a
    read takes milliseconds): a ``PermissionError`` is retried for up to
    :data:`_REMOVE_WAIT_S` seconds before it is raised. On POSIX this is
    ``Path.unlink(missing_ok=True)``.
    """
    path = Path(path)
    deadline = time.monotonic() + _REMOVE_WAIT_S
    while True:
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if not osutil.IS_WINDOWS or time.monotonic() >= deadline:
                raise
            time.sleep(_REMOVE_POLL_S)


def _signature(path: Path | None) -> tuple:
    """What tells one state of a vector file from the next.

    The file identity (inode / Windows file index) is part of it: a rebuild
    that replaces one text keeps the number of vectors and with it the size,
    and the modification time alone does not tell two writes within one
    clock tick apart (about 16 ms on NTFS, 2 s on FAT). Every write goes
    through a new temporary file, so the identity always changes.
    """
    if path is None:
        return ()
    try:
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size, stat.st_ino)
    except OSError:
        return ()


class VectorIndex:
    """Read-only merged view of the committed and the runtime vector file of one model.

    Thread-safe. Nothing is read before the first lookup; a missing or
    unusable file contributes no vectors and is reported in :attr:`error`
    (logged once per file state). :attr:`version` changes whenever the
    content was reloaded, so callers can cache what they derived from it.
    """

    def __init__(self, model_id: str, static_path: Path | None, cache_path: Path | None,
                 expected_dim: int = 0, expected_params: str | None = None) -> None:
        self.model_id = model_id
        self.static_path = static_path
        self.cache_path = cache_path
        self.expected_dim = int(expected_dim or 0)
        self.expected_params = expected_params
        self._lock = threading.Lock()
        self._signature: tuple | None = None      # None = never loaded
        self._checked_at = 0.0
        # One immutable snapshot, swapped as a whole: ({key: (file number, row)}, [matrix], [file info])
        self._data: tuple[dict, list, list] = ({}, [], [])
        self._version = 0
        self.error: str | None = None

    # -- loading ---------------------------------------------------------

    def _load(self, signature: tuple) -> None:
        rows: dict[str, tuple[int, int]] = {}
        matrices: list = []
        metas: list[dict] = []
        errors: list[str] = []
        dim = self.expected_dim
        # Both files carry the same name: say which one a problem is about.
        for label, path in (("committed file", self.static_path), ("runtime cache", self.cache_path)):
            if path is None:
                continue
            try:
                keys, matrix, meta = read_vector_file(path, self.model_id, dim, self.expected_params)
            except FileNotFoundError:
                continue
            except VectorFileError as exc:
                errors.append(f"{label} {exc}")
                logger.warning("Ignoring vector file (%s): %s", label, exc)
                continue
            dim = dim or int(matrix.shape[1])
            number = len(matrices)
            matrices.append(matrix)
            metas.append({**meta, "path": str(path), "vectors": len(keys)})
            for row, key in enumerate(keys):
                rows.setdefault(key, (number, row))
        self._data = (rows, matrices, metas)
        self._signature = signature
        self._version += 1
        self.error = "; ".join(errors) or None

    def _ensure(self) -> None:
        now = time.monotonic()
        if self._signature is not None and now - self._checked_at < CHECK_INTERVAL_S:
            return
        with self._lock:
            signature = (_signature(self.static_path), _signature(self.cache_path))
            self._checked_at = time.monotonic()
            if signature != self._signature:
                self._load(signature)

    def reload(self) -> None:
        """Check the files now (after this process wrote one)."""
        with self._lock:
            self._checked_at = 0.0
        self._ensure()

    # -- lookups ---------------------------------------------------------

    @property
    def version(self) -> int:
        self._ensure()
        return self._version

    @property
    def dim(self) -> int:
        self._ensure()
        matrices = self._data[1]
        return int(matrices[0].shape[1]) if matrices else 0

    def files(self) -> list[dict]:
        self._ensure()
        return [dict(meta) for meta in self._data[2]]

    def __len__(self) -> int:
        self._ensure()
        return len(self._data[0])

    def __contains__(self, key: object) -> bool:
        self._ensure()
        return key in self._data[0]

    def get(self, key: str):
        """The float32 vector stored under *key*, or None."""
        self._ensure()
        rows, matrices, _ = self._data
        hit = rows.get(key)
        if hit is None:
            return None
        np = numpy_or_none()
        return None if np is None else matrices[hit[0]][hit[1]].astype(np.float32)

    def matrix(self, keys: Sequence[str]):
        """``(positions, float32 matrix)`` for the *keys* that have a vector.

        ``positions[i]`` is the index into *keys* of row ``i``.
        """
        self._ensure()
        np = numpy_or_none()
        rows, matrices, _ = self._data
        if np is None or not matrices:
            return [], None
        positions: list[int] = []
        picked = []
        for position, key in enumerate(keys):
            hit = rows.get(key)
            if hit is not None:
                positions.append(position)
                picked.append(matrices[hit[0]][hit[1]])
        if not picked:
            return [], None
        return positions, np.stack(picked).astype(np.float32)


def sync_vector_file(
    path: Path,
    embedder,
    wanted: Iterable[tuple[str, str]],
    *,
    known: Callable[[str], bool] | None = None,
    should_stop: Callable[[], bool] | None = None,
    save_every: int = 200,
) -> dict:
    """Bring the vector file at *path* in line with *wanted* ``[(kind, text)]``.

    Incremental: a key already in the file is kept, a key for which
    ``known(key)`` is true (it lives in the committed file) is not stored
    here, every other key is embedded; keys that are no longer wanted are
    dropped. The file is rewritten only when something changed (and every
    *save_every* new vectors, so an interrupted build is not lost).

    Self-healing: an existing file that cannot be used (damaged, another
    format or other embedder parameters) is replaced, or removed when
    nothing is left to store in it. When the embedder starts returning
    vectors of another length (an API model was changed on the provider's
    side) the stored vectors are discarded and embedded again. A vector that
    is not finite or has no length (a text without any token) is not stored.

    Returns ``{"wanted", "kept", "embedded", "dropped", "skipped", "invalid",
    "vectors", "seconds", "written"}``. Raises what the embedder raises.
    """
    np = numpy_or_none()
    if np is None:
        raise VectorFileError("numpy is not installed")
    started = time.perf_counter()
    path = Path(path)
    model_id = embedder.id
    params = str(getattr(embedder, "params", "") or "")
    texts: dict[str, tuple[str, str]] = {}
    for kind, text in wanted:
        if text:
            texts.setdefault(vector_key(model_id, kind, text), (kind, text))

    have: dict[str, object] = {}
    unusable = False
    try:
        keys, matrix, _meta = read_vector_file(path, model_id, int(getattr(embedder, "dim", 0) or 0), params)
        have = {key: matrix[row] for row, key in enumerate(keys)}
    except FileNotFoundError:
        pass
    except VectorFileError as exc:
        unusable = True
        logger.warning("Rebuilding vector file: %s", exc)
    before = len(have)

    stats = {"wanted": len(texts), "kept": 0, "embedded": 0, "dropped": 0, "skipped": 0, "invalid": 0,
             "vectors": 0, "seconds": 0.0, "written": False}
    todo: list[str] = []
    keep: dict[str, object] = {}
    for key in texts:
        if key in have:
            keep[key] = have[key]
        elif known is not None and known(key):
            stats["skipped"] += 1
        else:
            todo.append(key)
    stats["kept"] = len(keep)
    stats["dropped"] = before - len(keep)

    def save() -> None:
        if not keep:   # no file is better than an empty one (its dimension would be unknown)
            remove_file(path)
            return
        ordered = sorted(keep)
        write_vector_file(path, model_id, ordered,
                          np.stack([np.asarray(keep[k], dtype=np.float16) for k in ordered]), params)
        stats["written"] = True

    stored_dim = len(next(iter(keep.values()))) if keep else 0
    pending = 0
    start = 0
    while start < len(todo):
        if should_stop is not None and should_stop():
            break
        batch = todo[start:start + 16]
        start += len(batch)
        queries = [k for k in batch if texts[k][0] == QUERY]
        docs = [k for k in batch if texts[k][0] != QUERY]
        for group, embed in ((docs, embedder.embed_documents), (queries, embedder.embed_queries)):
            if not group:
                continue
            vectors = embed([texts[k][1] for k in group])
            for key, vector in zip(group, vectors):
                vector = np.asarray(vector, dtype=np.float32)
                if vector.ndim != 1 or not vector.shape[0] or not bool(_usable_rows(np, vector[None, :])[0]):
                    stats["invalid"] += 1
                    continue
                if stored_dim and vector.shape[0] != stored_dim:
                    if stats["embedded"]:
                        raise VectorFileError(f"{model_id} returned vectors of {stored_dim} and "
                                              f"{vector.shape[0]} dimensions in one build")
                    # The stored vectors come from another revision of the model: embed them again.
                    logger.warning("Vector dimension of %s changed from %d to %d; rebuilding %s",
                                   model_id, stored_dim, vector.shape[0], path.name)
                    todo.extend(keep)
                    stats["dropped"] += len(keep)
                    stats["kept"] = 0
                    keep.clear()
                stored_dim = vector.shape[0]
                keep[key] = vector.astype(np.float16)
                stats["embedded"] += 1
                pending += 1
        if pending >= save_every:
            save()
            pending = 0
    if pending or stats["dropped"] or unusable:
        save()
    stats["vectors"] = len(keep)
    stats["seconds"] = round(time.perf_counter() - started, 3)
    return stats


def discard_unusable(path: Path, model_id: str, expected_dim: int = 0,
                     expected_params: str | None = None) -> bool:
    """Remove a runtime cache file that exists but cannot be used; True when it was removed.

    For the cache only (it is rebuilt from the texts); never call it on the
    committed file, whose damage the operator has to see.
    """
    try:
        read_vector_file(Path(path), model_id, expected_dim, expected_params)
        return False
    except FileNotFoundError:
        return False
    except VectorFileError as exc:
        if numpy_or_none() is None:
            return False   # unreadable only because numpy is missing
        logger.warning("Removing unusable vector cache file: %s", exc)
        try:
            remove_file(Path(path))
        except OSError as unlink_error:
            logger.warning("Could not remove %s: %s", path, unlink_error)
            return False
        return True


def coverage(index: VectorIndex | None, model_id: str, texts: Iterable[str], kind: str = DOC) -> float:
    """Share of *texts* that have a vector in *index* (1.0 for no texts)."""
    texts = [t for t in texts if t]
    if not texts:
        return 1.0
    if index is None:
        return 0.0
    return sum(1 for t in texts if vector_key(model_id, kind, t) in index) / len(texts)


__all__ = [
    "CHECK_INTERVAL_S", "DOC", "FORMAT_VERSION", "LOCK_POLL_S", "QUERY", "VectorFileError", "VectorIndex",
    "build_lock", "coverage", "discard_unusable", "file_slug", "numpy_or_none", "read_vector_file",
    "remove_file", "sync_vector_file", "vector_key", "write_vector_file",
]
