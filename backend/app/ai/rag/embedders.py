"""Embedding backends for the knowledge retrieval.

Three backends behind one small interface (``id``, ``dim``, ``params``,
``languages``, ``embed_documents``, ``embed_queries``, ``close``):

- :class:`OnnxEmbedder` — a local model through ``onnxruntime`` +
  ``tokenizers`` (optional packages, ``backend/requirements-rag.txt``). The
  model files are fetched once with ``python -m backend.scripts.rag
  download-model``; nothing here ever downloads.
- :class:`OpenAIEmbedder` — any OpenAI-compatible ``/embeddings`` endpoint
  with its own base URL / key / model (never the coach's LLM key).
- :class:`HashEmbedder` — deterministic feature hashing, for tests.

Serving a report needs none of them: decision ranking reads prebuilt vectors
(:mod:`.vectors`). An embedder is used by the vector builder and by the
agent's free-text search.

Only the standard library is imported at module import; numpy, onnxruntime,
tokenizers and openai are imported inside the functions that need them.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import math
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "bge-small-en-v1.5-int8"
DEFAULT_HF_ENDPOINT = "https://huggingface.co"
HASH_DIM = 256


class EmbedderError(RuntimeError):
    """An embedder cannot produce vectors; ``reason`` is a short code."""

    def __init__(self, reason: str, message: str = "") -> None:
        super().__init__(message or reason)
        self.reason = reason


class Embedder(Protocol):
    id: str                       # identity of the vector space (part of every vector key)
    dim: int                      # 0 = not known before the first call
    params: str                   # fingerprint of how vectors are produced ("" = none), see ModelSpec.params
    languages: frozenset          # {"en"} or {"*"}

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Model registry (local ONNX models)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """A local ONNX embedding model: where it comes from and how to run it."""

    model_id: str
    repo: str                                   # Hugging Face repository
    files: dict = field(default_factory=dict)   # relative path -> (bytes, sha256)
    onnx_file: str = "onnx/model_quantized.onnx"
    dim: int = 384
    pooling: str = "cls"                        # "cls" | "mean"
    query_prefix: str = ""
    document_prefix: str = ""
    max_tokens: int = 512
    languages: frozenset = frozenset({"en"})
    revision: str = "main"

    @property
    def params(self) -> str:
        """Fingerprint of everything besides the text that decides a vector.

        Stored in the vector files: a vector key only names the model, so
        after a change of the pooling, a prefix, the token limit or the model
        file the old vectors must not be served (the file is refused and has
        to be rebuilt).
        """
        parts = [self.pooling, self.query_prefix, self.document_prefix, str(self.max_tokens), self.onnx_file]
        parts += [f"{name}:{sha256}" for name, (_size, sha256) in sorted(self.files.items())]
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


MODEL_REGISTRY: dict[str, ModelSpec] = {
    # BAAI/bge-small-en-v1.5, int8-quantised ONNX export (34 MB + tokenizer).
    DEFAULT_MODEL: ModelSpec(
        model_id=DEFAULT_MODEL,
        repo="Xenova/bge-small-en-v1.5",
        files={
            "onnx/model_quantized.onnx": (
                34014426, "6c9c6101a956d62dfb5e7190c538226c0c5bb9cb27b651234b6df063ee7dbfe4"),
            "tokenizer.json": (
                711396, "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"),
        },
        onnx_file="onnx/model_quantized.onnx",
        dim=384,
        pooling="cls",
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_files_state(spec: ModelSpec, directory: Path, *, checksum: bool = False) -> str:
    """"" when the model files are usable, else ``model_missing`` / ``model_checksum``.

    Sizes are always compared; the sha256 only with ``checksum=True`` (it
    reads the whole model file).
    """
    for relative, (size, sha256) in spec.files.items():
        path = directory / relative
        try:
            actual = path.stat().st_size
        except OSError:
            return "model_missing"
        if actual != size or (checksum and _sha256(path) != sha256):
            return "model_checksum"
    return ""


def download_model(spec: ModelSpec, directory: Path, endpoint: str = "",
                   progress: Callable[[str], None] | None = None, timeout: float = 60.0) -> list[Path]:
    """Fetch the model files into *directory* (CLI only; the server never calls this).

    *endpoint* defaults to ``$HF_ENDPOINT`` or huggingface.co (a mirror such
    as ``https://hf-mirror.com`` serves the same files). Every file is
    written to a temporary name, checked against the pinned size and sha256
    and only then moved into place; a file that already matches is kept.
    Raises :class:`EmbedderError` (``model_checksum`` / ``download_failed``).
    """
    endpoint = (endpoint or os.environ.get("HF_ENDPOINT", "").strip() or DEFAULT_HF_ENDPOINT).rstrip("/")
    say = progress or (lambda message: None)
    written: list[Path] = []
    for relative, (size, sha256) in spec.files.items():
        target = directory / relative
        if target.is_file() and target.stat().st_size == size and _sha256(target) == sha256:
            say(f"{relative}: already present, checksum ok")
            written.append(target)
            continue
        url = f"{endpoint}/{spec.repo}/resolve/{spec.revision}/{relative}"
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        say(f"{relative}: downloading {url}")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "DotaAnalyzer-RAG/1.0"})
            digest = hashlib.sha256()
            received = 0
            with urllib.request.urlopen(request, timeout=timeout) as response, open(partial, "wb") as out:
                for block in iter(lambda: response.read(1 << 20), b""):
                    out.write(block)
                    digest.update(block)
                    received += len(block)
        except Exception as exc:
            partial.unlink(missing_ok=True)
            raise EmbedderError("download_failed", f"{url}: {exc}") from exc
        if received != size or digest.hexdigest() != sha256:
            partial.unlink(missing_ok=True)
            raise EmbedderError(
                "model_checksum",
                f"{relative}: got {received} bytes, sha256 {digest.hexdigest()}; "
                f"expected {size} bytes, sha256 {sha256}")
        os.replace(partial, target)
        say(f"{relative}: {received} bytes, checksum ok")
        written.append(target)
    return written


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

def _normalised(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in vector))
    return [float(v) / norm for v in vector] if norm > 0 else [0.0 for _ in vector]


_HASH_TOKEN_RE = re.compile(r"[a-z0-9']+|[一-鿿]")


class HashEmbedder:
    """Deterministic bag-of-words hashing into ``dim`` signed buckets (tests only).

    Texts sharing words get a positive cosine; nothing else is implied.
    """

    languages = frozenset({"*"})
    params = ""

    def __init__(self, dim: int = HASH_DIM) -> None:
        self.dim = int(dim)
        self.id = f"hash-{self.dim}"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _HASH_TOKEN_RE.findall((text or "").lower()):
            value = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")
            vector[(value >> 1) % self.dim] += 1.0 if value & 1 else -1.0
        return _normalised(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    embed_queries = embed_documents

    def close(self) -> None:
        return None


class OnnxEmbedder:
    """A registry model run with onnxruntime; loaded on first use.

    One text per inference run: an int8 model's vector depends slightly on
    what a text is padded with, so batching would make a chunk's vector
    depend on its neighbours in the batch. After ``idle_unload_s`` seconds
    without use the session is released (``0`` keeps it).
    """

    def __init__(self, spec: ModelSpec, directory: Path, threads: int = 1,
                 idle_unload_s: float = 900.0) -> None:
        self.spec = spec
        self.directory = Path(directory)
        self.id = spec.model_id
        self.dim = spec.dim
        self.params = spec.params
        self.languages = spec.languages
        self.threads = max(1, int(threads))
        self.idle_unload_s = max(0.0, float(idle_unload_s))
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(2)   # bounds CPU use of concurrent searches
        self._session = None
        self._tokenizer = None
        self._inputs: set[str] = set()
        self._last_used = 0.0
        self._timer: threading.Timer | None = None

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def _load(self):
        with self._lock:
            if self._session is not None:
                return self._session, self._tokenizer
            state = model_files_state(self.spec, self.directory, checksum=True)
            if state:
                raise EmbedderError(state, f"model files in {self.directory} are unusable ({state})")
            try:
                import onnxruntime
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise EmbedderError("deps_missing", str(exc)) from exc
            started = time.perf_counter()
            tokenizer = Tokenizer.from_file(str(self.directory / "tokenizer.json"))
            tokenizer.enable_truncation(max_length=self.spec.max_tokens)
            tokenizer.no_padding()
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            session = onnxruntime.InferenceSession(
                str(self.directory / self.spec.onnx_file), sess_options=options,
                providers=["CPUExecutionProvider"])
            self._inputs = {i.name for i in session.get_inputs()}
            self._session, self._tokenizer = session, tokenizer
            logger.info("Embedding model %s loaded in %.2f s (%d thread(s))",
                        self.id, time.perf_counter() - started, self.threads)
            return session, tokenizer

    def _schedule_unload(self) -> None:
        if self.idle_unload_s <= 0:
            return
        with self._lock:
            if self._timer is not None or self._session is None:
                return
            timer = threading.Timer(self.idle_unload_s, self._unload_if_idle)
            timer.daemon = True
            self._timer = timer
        timer.start()

    def _unload_if_idle(self) -> None:
        with self._lock:
            self._timer = None
            idle = time.monotonic() - self._last_used
            if self._session is None:
                return
            if idle >= self.idle_unload_s:
                self._session = self._tokenizer = None
                logger.info("Embedding model %s released after %.0f s idle", self.id, idle)
                return
        self._schedule_unload()

    def _embed(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        import numpy as np

        session, tokenizer = self._load()
        out: list[list[float]] = []
        for text in texts:
            with self._slots:
                encoding = tokenizer.encode(prefix + (text or ""))
                ids = np.asarray([encoding.ids], dtype=np.int64)
                mask = np.asarray([encoding.attention_mask], dtype=np.int64)
                feed = {"input_ids": ids, "attention_mask": mask}
                if "token_type_ids" in self._inputs:
                    feed["token_type_ids"] = np.zeros_like(ids)
                hidden = session.run(None, feed)[0]
            if hidden.ndim == 3:
                if self.spec.pooling == "cls":
                    pooled = hidden[0, 0]
                else:
                    weights = mask[0][:, None].astype(hidden.dtype)
                    pooled = (hidden[0] * weights).sum(0) / max(float(weights.sum()), 1e-9)
            else:
                pooled = hidden[0]
            pooled = pooled.astype(np.float32)
            norm = float(np.linalg.norm(pooled))
            out.append((pooled / norm if norm > 0 else pooled).tolist())
        self._last_used = time.monotonic()
        self._schedule_unload()
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, self.spec.document_prefix)

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, self.spec.query_prefix)

    def close(self) -> None:
        with self._lock:
            timer, self._timer = self._timer, None
            self._session = self._tokenizer = None
        if timer is not None:
            timer.cancel()


class OpenAIEmbedder:
    """An OpenAI-compatible ``/embeddings`` endpoint (vectors are L2-normalised here).

    The provider receives chunk texts and query strings (hero names and
    situation descriptions; never player names).
    """

    languages = frozenset({"*"})
    params = ""

    def __init__(self, base_url: str, api_key: str, model: str, *, timeout: float = 10.0,
                 batch: int = 32, client=None) -> None:
        self.base_url = base_url
        self.model = model
        self.id = f"openai:{model}"
        self.dim = 0
        self.timeout = float(timeout)
        self.batch = max(1, int(batch))
        self._api_key = api_key
        self._client = client
        self._lock = threading.Lock()

    def _get_client(self):
        with self._lock:
            if self._client is None:
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise EmbedderError("deps_missing", str(exc)) from exc
                self._client = OpenAI(base_url=self.base_url, api_key=self._api_key,
                                      timeout=self.timeout, max_retries=1)
            return self._client

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        client = self._get_client()
        out: list[list[float]] = []
        texts = list(texts)
        for start in range(0, len(texts), self.batch):
            batch = texts[start:start + self.batch]
            try:
                response = client.embeddings.create(model=self.model, input=batch)
                rows = sorted(response.data, key=lambda row: getattr(row, "index", 0))
                vectors = [_normalised(row.embedding) for row in rows]
            except EmbedderError:
                raise
            except Exception as exc:
                # Only the kind of failure. A provider's error body can echo
                # the request, the API key included, and this text ends up in
                # the log and in /api/admin/status.
                status = getattr(exc, "status_code", None)
                detail = type(exc).__name__ + (f" (HTTP {status})" if isinstance(status, int) else "")
                raise EmbedderError("api_unreachable", detail) from None
            if len(vectors) != len(batch) or any(len(v) != len(vectors[0]) for v in vectors):
                raise EmbedderError("api_unreachable", "embeddings response does not match the request")
            self.dim = len(vectors[0])
            out.extend(vectors)
        return out

    embed_queries = embed_documents

    def close(self) -> None:
        with self._lock:
            client, self._client = self._client, None
        try:
            if client is not None and hasattr(client, "close"):
                client.close()
        except Exception:  # pragma: no cover - best effort
            pass


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _importable(*modules: str) -> bool:
    for name in modules:
        try:
            if importlib.util.find_spec(name) is None:
                return False
        except (ImportError, ValueError):
            return False
    return True


def active_model_id(settings) -> str:
    """Identity of the vector space the configuration selects (also without an embedder)."""
    if settings.embeddings == "hash":
        return f"hash-{HASH_DIM}"
    if settings.embeddings == "openai" or (settings.embeddings == "auto" and settings.api_configured):
        return f"openai:{settings.api_model}" if settings.api_model else ""
    return settings.model


def resolve_embedder(settings) -> tuple[Embedder | None, str]:
    """``(embedder, "")`` or ``(None, reason)`` for the configuration.

    Never raises, never touches the network and never loads a model (that
    happens on the first ``embed_*`` call). Reasons: ``off``,
    ``api_not_configured``, ``deps_missing``, ``unknown_model``,
    ``model_missing``, ``model_checksum``.
    """
    try:
        mode = settings.embeddings
        if mode == "off":
            return None, "off"
        if mode == "hash":
            return HashEmbedder(), ""
        if mode == "openai" or (mode == "auto" and settings.api_configured):
            if not (settings.api_configured and settings.api_model):
                return None, "api_not_configured"
            if not _importable("openai"):
                return None, "deps_missing"
            return OpenAIEmbedder(settings.api_base_url, settings.api_key, settings.api_model), ""
        spec = MODEL_REGISTRY.get(settings.model)
        if spec is None:
            return None, "unknown_model"
        if not _importable("numpy", "onnxruntime", "tokenizers"):
            return None, "deps_missing"
        directory = settings.model_dir / spec.model_id
        state = model_files_state(spec, directory)
        if state:
            return None, state
        return OnnxEmbedder(spec, directory, threads=settings.embed_threads,
                            idle_unload_s=settings.idle_unload_s), ""
    except Exception as exc:  # pragma: no cover - defensive: resolution must not raise
        logger.warning("Embedder resolution failed: %s", exc)
        return None, "error"


__all__ = [
    "DEFAULT_MODEL", "Embedder", "EmbedderError", "HASH_DIM", "HashEmbedder", "MODEL_REGISTRY",
    "ModelSpec", "OnnxEmbedder", "OpenAIEmbedder", "active_model_id", "download_model",
    "model_files_state", "resolve_embedder",
]
