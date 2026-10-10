"""Embedding backends: hash / OpenAI-compatible / ONNX, resolution, model download."""
import hashlib
import http.server
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.ai.rag import embedders, runtime
from backend.app.ai.rag.embedders import (
    EmbedderError,
    HashEmbedder,
    ModelSpec,
    OnnxEmbedder,
    OpenAIEmbedder,
    resolve_embedder,
)


@pytest.fixture
def rag_env(tmp_path, monkeypatch):
    for name in ("RAG_EMBEDDINGS", "RAG_MODEL", "RAG_EMBED_API_BASE_URL", "RAG_EMBED_API_KEY",
                 "RAG_EMBED_API_MODEL", "RAG_TOP_K", "RAG_AGENT_SEARCH_LIMIT", "HF_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAG_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(runtime, "STATIC_DIR", tmp_path / "static")
    runtime.reset()
    yield tmp_path
    runtime.reset()


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Hash embedder (tests)
# ---------------------------------------------------------------------------

def test_hash_embedder_is_deterministic_and_normalised():
    embedder = HashEmbedder()
    a, b, c = embedder.embed_documents([
        "teamfight positioning and target priority",
        "positioning in a teamfight: who to target",
        "stack the ancient camp at 53 seconds",
    ])
    assert embedder.id == "hash-256" and len(a) == embedder.dim == 256
    assert _dot(a, a) == pytest.approx(1.0)
    assert a == HashEmbedder().embed_queries(["teamfight positioning and target priority"])[0]
    assert _dot(a, b) > _dot(a, c)          # shared words -> closer
    assert embedder.embed_documents([""])[0] == [0.0] * 256
    assert "*" in embedder.languages


# ---------------------------------------------------------------------------
# Resolution: every row that leads to "no embedder" has a reason, none raises
# ---------------------------------------------------------------------------

@pytest.fixture
def optional_packages(monkeypatch):
    """Behave as if onnxruntime / tokenizers were installed, whatever this venv holds.

    The rows below are about model files and settings; without the optional
    packages every one of them would read ``deps_missing`` (that row is
    :func:`test_optional_packages_missing`).
    """
    monkeypatch.setattr(embedders, "_importable", lambda *modules: True)


def test_resolution_matrix(rag_env, optional_packages, monkeypatch):
    def resolve(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return resolve_embedder(runtime.settings())

    assert resolve(RAG_EMBEDDINGS="off") == (None, "off")
    embedder, reason = resolve(RAG_EMBEDDINGS="hash")
    assert isinstance(embedder, HashEmbedder) and reason == ""
    # The default: a local model that is not downloaded
    assert resolve(RAG_EMBEDDINGS="auto") == (None, "model_missing")
    assert resolve(RAG_EMBEDDINGS="onnx") == (None, "model_missing")
    assert resolve(RAG_EMBEDDINGS="onnx", RAG_MODEL="no-such-model") == (None, "unknown_model")
    monkeypatch.delenv("RAG_MODEL")
    # An invalid mode falls back to auto
    assert resolve(RAG_EMBEDDINGS="banana") == (None, "model_missing")
    # API backend: needs its own base URL, key and model
    assert resolve(RAG_EMBEDDINGS="openai") == (None, "api_not_configured")
    assert resolve(RAG_EMBED_API_BASE_URL="http://127.0.0.1:9/v1", RAG_EMBED_API_KEY="k") == (
        None, "api_not_configured")          # no model name
    embedder, reason = resolve(RAG_EMBED_API_MODEL="text-embed-x")
    assert isinstance(embedder, OpenAIEmbedder) and embedder.id == "openai:text-embed-x" and reason == ""
    # ... and with API settings present, auto prefers it over the local model
    embedder, _ = resolve(RAG_EMBEDDINGS="auto")
    assert isinstance(embedder, OpenAIEmbedder)


def test_optional_packages_missing(rag_env, monkeypatch):
    spec = embedders.MODEL_REGISTRY[embedders.DEFAULT_MODEL]
    directory = rag_env / "models" / spec.model_id
    for relative, (size, _sha) in spec.files.items():
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        with open(directory / relative, "wb") as f:
            f.truncate(size)                         # right size, wrong content
    for hidden in ("onnxruntime", "tokenizers", "numpy"):
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, hidden, None)   # "import x" raises ImportError
            assert resolve_embedder(runtime.settings()) == (None, "deps_missing")
            # ... also when the model files are not there: the packages are asked for first
            patch.setenv("RAG_MODEL_DIR", str(rag_env / "no-models"))
            assert resolve_embedder(runtime.settings()) == (None, "deps_missing")
    if embedders._importable("numpy", "onnxruntime", "tokenizers"):
        embedder, reason = resolve_embedder(runtime.settings())
        assert isinstance(embedder, OnnxEmbedder) and reason == ""
        # The full checksum is verified when the model is loaded, not per request
        with pytest.raises(EmbedderError) as error:
            embedder.embed_queries(["x"])
        assert error.value.reason == "model_checksum" and not embedder.loaded


def test_the_coach_llm_key_is_never_used_for_embeddings(rag_env, optional_packages, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-coach-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.example/v1")
    cfg = runtime.settings()
    assert not cfg.api_configured and cfg.api_key == ""
    assert resolve_embedder(cfg) == (None, "model_missing")
    monkeypatch.setenv("RAG_EMBEDDINGS", "openai")
    assert resolve_embedder(runtime.settings()) == (None, "api_not_configured")


# ---------------------------------------------------------------------------
# OpenAI-compatible backend (injected client: no network)
# ---------------------------------------------------------------------------

class _FakeEmbeddings:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def create(self, model, input):
        self.calls.append((model, list(input)))
        if self.fail:
            raise TimeoutError("timed out")
        # Rows deliberately out of order: the client must sort by index
        rows = [SimpleNamespace(index=i, embedding=[float(len(text)), 0.0, 3.0]) for i, text in enumerate(input)]
        return SimpleNamespace(data=list(reversed(rows)))


def test_openai_embedder_batches_orders_and_normalises():
    api = _FakeEmbeddings()
    embedder = OpenAIEmbedder("http://127.0.0.1:9/v1", "key", "m", batch=2,
                              client=SimpleNamespace(embeddings=api))
    vectors = embedder.embed_documents(["a", "bbbb", "cc"])
    assert [call[1] for call in api.calls] == [["a", "bbbb"], ["cc"]] and api.calls[0][0] == "m"
    assert embedder.dim == 3 and embedder.id == "openai:m"
    assert vectors[1] == pytest.approx([0.8, 0.0, 0.6])        # (4, 0, 3) / 5: "bbbb" stayed second
    assert all(_dot(v, v) == pytest.approx(1.0) for v in vectors)
    assert embedder.embed_queries(["a"]) == [vectors[0]]


def test_openai_embedder_failure_is_an_embedder_error():
    embedder = OpenAIEmbedder("http://127.0.0.1:9/v1", "key", "m",
                              client=SimpleNamespace(embeddings=_FakeEmbeddings(fail=True)))
    with pytest.raises(EmbedderError) as error:
        embedder.embed_documents(["a"])
    assert error.value.reason == "api_unreachable" and str(error.value) == "TimeoutError"


def test_openai_embedder_error_never_carries_the_provider_text():
    """A provider can echo the request in its error body, the API key included."""
    key = "sk-SECRET-EMBED-KEY-123456"

    class Unauthorized(Exception):
        status_code = 401

    class Echoing:
        def create(self, model, input):
            raise Unauthorized(f"Incorrect API key provided: {key}. Request was: {input}")

    embedder = OpenAIEmbedder("http://127.0.0.1:9/v1", key, "m", client=SimpleNamespace(embeddings=Echoing()))
    with pytest.raises(EmbedderError) as error:
        embedder.embed_queries(["position 1 carry, laning phase: last hitting"])
    assert str(error.value) == "Unauthorized (HTTP 401)" and error.value.reason == "api_unreachable"
    assert error.value.__cause__ is None                      # nothing to print with a traceback either
    assert key not in repr(error.value) and "last hitting" not in str(error.value)


# ---------------------------------------------------------------------------
# Model files: state, download with checksum, idle unload
# ---------------------------------------------------------------------------

def _tiny_spec(payloads: dict[str, bytes]) -> ModelSpec:
    return ModelSpec(
        model_id="tiny-test-model", repo="acme/tiny",
        files={name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in payloads.items()},
        onnx_file="onnx/model.onnx", dim=4)


def test_model_files_state(tmp_path):
    payloads = {"onnx/model.onnx": b"model-bytes", "tokenizer.json": b"{}"}
    spec = _tiny_spec(payloads)
    assert embedders.model_files_state(spec, tmp_path) == "model_missing"
    for name, data in payloads.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(data)
    assert embedders.model_files_state(spec, tmp_path, checksum=True) == ""
    (tmp_path / "tokenizer.json").write_bytes(b"[]")           # same size, other content
    assert embedders.model_files_state(spec, tmp_path) == ""
    assert embedders.model_files_state(spec, tmp_path, checksum=True) == "model_checksum"
    (tmp_path / "tokenizer.json").write_bytes(b"{} ")
    assert embedders.model_files_state(spec, tmp_path) == "model_checksum"
    with pytest.raises(EmbedderError) as error:                 # loading never trusts a bad file
        OnnxEmbedder(spec, tmp_path).embed_documents(["x"])
    assert error.value.reason == "model_checksum"


@pytest.fixture
def model_mirror(tmp_path):
    """A loopback HTTP server laid out like a Hugging Face endpoint."""
    root = tmp_path / "mirror"
    payloads = {"onnx/model.onnx": b"\x00onnx" * 500, "tokenizer.json": b'{"version": "1.0"}'}
    for name, data in payloads.items():
        target = root / "acme" / "tiny" / "resolve" / "main" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_address[1]}", payloads=payloads, root=root)
    server.shutdown()
    server.server_close()


def test_download_model_verifies_checksums(model_mirror, tmp_path, monkeypatch):
    spec = _tiny_spec(model_mirror.payloads)
    target = tmp_path / "models" / spec.model_id
    messages = []
    monkeypatch.setenv("HF_ENDPOINT", model_mirror.url + "/")       # the mirror variable is honoured
    written = embedders.download_model(spec, target, progress=messages.append)
    assert sorted(p.relative_to(target).as_posix() for p in written) == sorted(model_mirror.payloads)
    assert embedders.model_files_state(spec, target, checksum=True) == ""
    assert not list(target.rglob("*.part"))
    # A second run downloads nothing
    messages.clear()
    embedders.download_model(spec, target, progress=messages.append)
    assert all("already present" in m for m in messages)

    # The mirror serves something else: nothing is kept
    served = model_mirror.root / "acme" / "tiny" / "resolve" / "main" / "tokenizer.json"
    served.write_bytes(b'{"version": "evil"}')
    (target / "tokenizer.json").unlink()
    with pytest.raises(EmbedderError) as error:
        embedders.download_model(spec, target, endpoint=model_mirror.url)
    assert error.value.reason == "model_checksum"
    assert not (target / "tokenizer.json").exists() and not list(target.rglob("*.part"))
    # Unreachable file
    with pytest.raises(EmbedderError) as error:
        embedders.download_model(_tiny_spec({"missing.bin": b"x"}), tmp_path / "other", endpoint=model_mirror.url)
    assert error.value.reason == "download_failed"


def test_model_parameters_have_a_fingerprint():
    """What decides a vector besides the text is part of the vector files (see test_ai_rag_vectors)."""
    import dataclasses

    spec = embedders.MODEL_REGISTRY[embedders.DEFAULT_MODEL]
    assert len(spec.params) == 16 and spec.params == spec.params
    assert OnnxEmbedder(spec, Path("nowhere")).params == spec.params
    assert HashEmbedder().params == "" == OpenAIEmbedder("http://127.0.0.1:9/v1", "k", "m").params
    onnx_file = spec.onnx_file
    for change in (dict(pooling="mean"), dict(query_prefix=""), dict(document_prefix="passage: "),
                   dict(max_tokens=256), dict(files={**spec.files, onnx_file: (spec.files[onnx_file][0], "0" * 64)})):
        assert dataclasses.replace(spec, **change).params != spec.params, change
    # Not part of it: where the files are fetched from
    assert dataclasses.replace(spec, repo="mirror/bge", revision="abc").params == spec.params


def test_default_model_is_run_the_way_its_vectors_were_built():
    """BAAI/bge-small-en-v1.5: CLS pooling, the retrieval instruction in front of queries only."""
    spec = embedders.MODEL_REGISTRY[embedders.DEFAULT_MODEL]
    assert (spec.pooling, spec.dim, spec.document_prefix) == ("cls", 384, "")
    assert spec.query_prefix == "Represent this sentence for searching relevant passages: "


class _FakeTokenizer:
    def __init__(self):
        self.texts = []

    def encode(self, text):
        self.texts.append(text)
        length = 2 + len(text.split())
        return SimpleNamespace(ids=list(range(1, length + 1)), attention_mask=[1] * length)


class _FakeSession:
    """Hidden states in which the first ([CLS]) row and the mean of the rows point in different directions."""

    def run(self, _outputs, feed):
        np = pytest.importorskip("numpy")
        length = feed["input_ids"].shape[1]
        assert set(feed) == {"input_ids", "attention_mask", "token_type_ids"}
        hidden = np.zeros((1, length, 4), dtype=np.float32)
        hidden[0, 0] = [3.0, 4.0, 0.0, 0.0]                    # CLS
        hidden[0, 1:, 2] = 5.0                                 # every other token
        return [hidden]


def _loaded_embedder(tmp_path, **spec_changes):
    import dataclasses

    spec = dataclasses.replace(_tiny_spec({"onnx/model.onnx": b"x"}), **spec_changes)
    embedder = OnnxEmbedder(spec, tmp_path, idle_unload_s=0)
    embedder._session, embedder._tokenizer = _FakeSession(), _FakeTokenizer()   # as if loaded
    embedder._inputs = {"input_ids", "attention_mask", "token_type_ids"}
    return embedder


def test_onnx_embedder_prefixes_queries_and_pools_as_the_spec_says(tmp_path):
    pytest.importorskip("numpy")
    embedder = _loaded_embedder(tmp_path, pooling="cls", query_prefix="Query: ", document_prefix="")
    document = embedder.embed_documents(["stand behind the front line"])[0]
    query = embedder.embed_queries(["where to stand"])[0]
    assert embedder._tokenizer.texts == ["stand behind the front line", "Query: where to stand"]
    assert document == pytest.approx([0.6, 0.8, 0.0, 0.0]) == query          # the [CLS] row, L2-normalised
    # One text per run: nothing is padded, so a vector cannot depend on its neighbours
    embedder._tokenizer.texts.clear()
    assert embedder.embed_documents(["a", "b c d"]) == [pytest.approx([0.6, 0.8, 0.0, 0.0])] * 2
    assert embedder._tokenizer.texts == ["a", "b c d"]

    mean = _loaded_embedder(tmp_path, pooling="mean", document_prefix="passage: ")
    pooled = mean.embed_documents(["one two"])[0]              # 5 tokens: (3, 4, 0, 0) and four times (0, 0, 5, 0)
    assert mean._tokenizer.texts == ["passage: one two"]
    norm = 17 ** 0.5                                           # of the mean (0.6, 0.8, 4, 0)
    assert pooled == pytest.approx([0.6 / norm, 0.8 / norm, 4.0 / norm, 0.0], abs=1e-6)
    assert _dot(pooled, pooled) == pytest.approx(1.0)


def test_idle_model_is_released(tmp_path):
    embedder = OnnxEmbedder(_tiny_spec({"onnx/model.onnx": b"x"}), tmp_path, idle_unload_s=0.05)
    embedder._session, embedder._tokenizer = object(), object()     # as if loaded
    embedder._last_used = time.monotonic()
    embedder._unload_if_idle()                                      # used just now: kept, checked again later
    assert embedder.loaded and embedder._timer is not None
    embedder._last_used = time.monotonic() - 10
    embedder._timer.cancel()
    embedder._unload_if_idle()
    assert not embedder.loaded
    embedder.close()
    keep = OnnxEmbedder(_tiny_spec({"onnx/model.onnx": b"x"}), tmp_path, idle_unload_s=0)
    keep._session = object()
    keep._schedule_unload()
    assert keep._timer is None                                      # 0 = never unload


@pytest.mark.skipif(not os.environ.get("RAG_TEST_MODEL_DIR"),
                    reason="set RAG_TEST_MODEL_DIR to a directory holding bge-small-en-v1.5-int8/ to run the real model")
def test_real_onnx_model():
    spec = embedders.MODEL_REGISTRY[embedders.DEFAULT_MODEL]
    embedder = OnnxEmbedder(spec, Path(os.environ["RAG_TEST_MODEL_DIR"]) / spec.model_id, idle_unload_s=0)
    try:
        query = embedder.embed_queries(["position 1 carry, late game: teamfight positioning"])[0]
        docs = embedder.embed_documents([
            "## Teamfight Positioning Principles\n- Core damage dealers stay in the back line",
            "## Support Stacking and Pulling\n- Pull at x:15 or x:45",
        ])
        assert len(query) == spec.dim == 384 and _dot(query, query) == pytest.approx(1.0, abs=1e-4)
        assert _dot(query, docs[0]) > _dot(query, docs[1])
        # One text per run: a vector does not depend on what else is embedded with it
        assert embedder.embed_documents(["## Support Stacking and Pulling\n- Pull at x:15 or x:45"])[0] == docs[1]
    finally:
        embedder.close()


@pytest.mark.skipif(not os.environ.get("RAG_TEST_MODEL_DIR"),
                    reason="set RAG_TEST_MODEL_DIR to a directory holding bge-small-en-v1.5-int8/ to run the real model")
def test_real_model_reproduces_the_committed_vectors(monkeypatch):
    """The committed file was built with exactly this model, pooling and prefix (and, here, these kernels)."""
    import platform

    for name in ("RAG_EMBED_API_BASE_URL", "RAG_EMBED_API_KEY", "RAG_EMBED_API_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RAG_EMBEDDINGS", "onnx")
    monkeypatch.setenv("RAG_MODEL", embedders.DEFAULT_MODEL)
    monkeypatch.setenv("RAG_MODEL_DIR", os.environ["RAG_TEST_MODEL_DIR"])
    monkeypatch.setattr(runtime, "STATIC_DIR", None)            # the committed file
    runtime.reset()
    try:
        check = runtime.embedder_consistency()
        assert check["checked"] == 25, check
        if check["file_arch"] == platform.machine():
            assert check["threshold"] == runtime.SAME_ARCH_MIN_COSINE == 0.999
        else:
            assert check["threshold"] == runtime.OTHER_ARCH_MIN_COSINE == 0.98
        assert check["ok"] and check["min_cosine"] >= check["threshold"], check
    finally:
        runtime.reset()
