"""Vector files: content keys, atomic float16 files, merged lazy index, incremental builds."""
import os

import pytest

np = pytest.importorskip("numpy")

from backend.app.ai.rag import vectors
from backend.app.ai.rag.embedders import HashEmbedder
from backend.app.ai.rag.vectors import DOC, QUERY, VectorFileError, VectorIndex, vector_key

MODEL = "hash-256"


class SpyEmbedder(HashEmbedder):
    """Hash embedder that records what it was asked to embed."""

    def __init__(self):
        super().__init__()
        self.documents: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts):
        self.documents.extend(texts)
        return super().embed_documents(texts)

    def embed_queries(self, texts):
        self.queries.extend(texts)
        return super().embed_documents(texts)


def _file(path, texts, kind=DOC, model=MODEL):
    embedder = HashEmbedder()
    keys = [vector_key(model, kind, text) for text in texts]
    vectors.write_vector_file(path, model, keys, np.asarray(embedder.embed_documents(texts)))
    return keys


def test_keys_depend_on_model_kind_and_text():
    key = vector_key("m", DOC, "text")
    assert len(key) == 64 and key == vector_key("m", DOC, "text")
    assert len({key, vector_key("m2", DOC, "text"), vector_key("m", QUERY, "text"), vector_key("m", DOC, "text ")}) == 4
    assert vectors.file_slug("openai:text-embedding/3 small") == "openai-text-embedding-3-small"


def test_file_round_trip_is_float16_and_atomic(tmp_path):
    path = tmp_path / "rag" / "hash-256.npz"
    keys = _file(path, ["alpha beta", "gamma"])
    read_keys, matrix, meta = vectors.read_vector_file(path, MODEL, 256)
    assert read_keys == keys and matrix.dtype == np.float16 and matrix.shape == (2, 256)
    assert meta["model"] == MODEL and meta["built_at"].endswith("Z") and meta["arch"]
    assert [p.name for p in path.parent.iterdir()] == ["hash-256.npz"]      # no temp file left
    with np.load(path, allow_pickle=False) as data:                         # plain arrays, no pickle
        assert set(data.files) >= {"keys", "vectors", "model", "dim", "format"}
    with pytest.raises(FileNotFoundError):
        vectors.read_vector_file(tmp_path / "nope.npz", MODEL)
    with pytest.raises(VectorFileError, match="built with model"):
        vectors.read_vector_file(path, "another-model")
    with pytest.raises(VectorFileError, match="dimension 256, expected 384"):
        vectors.read_vector_file(path, MODEL, 384)
    path.write_bytes(b"not an npz file")
    with pytest.raises(VectorFileError, match="unreadable"):
        vectors.read_vector_file(path, MODEL)
    with pytest.raises(VectorFileError):
        vectors.write_vector_file(path, MODEL, ["one key"], np.zeros((2, 4)))


def test_file_of_another_format_version_is_refused(tmp_path):
    path = tmp_path / "old.npz"
    vector = np.asarray(HashEmbedder().embed_documents(["alpha beta"]), dtype=np.float16)
    key = np.asarray([vector_key(MODEL, DOC, "alpha beta")], dtype="S64")
    for stored in (1, 99, None):                             # an older writer, a newer one, no version at all
        fields = dict(keys=key, vectors=vector, model=np.asarray(MODEL), dim=np.asarray(256))
        if stored is not None:
            fields["format"] = np.asarray(stored)
        np.savez(path, **fields)
        with pytest.raises(VectorFileError, match=f"format {stored or 0}, this version reads format 2"):
            vectors.read_vector_file(path, MODEL)
    np.savez(path, keys=key, vectors=vector, model=np.asarray(MODEL), format=np.asarray(vectors.FORMAT_VERSION))
    assert vectors.read_vector_file(path, MODEL)[0] == [key[0].decode()]


def test_embedder_parameters_are_part_of_the_file(tmp_path, monkeypatch):
    """Keys only name the model: vectors made with another pooling / prefix must not be served for it."""
    path = tmp_path / "m.npz"
    texts = ["alpha beta", "gamma"]
    keys = [vector_key(MODEL, DOC, text) for text in texts]
    vectors.write_vector_file(path, MODEL, keys, np.asarray(HashEmbedder().embed_documents(texts)), params="cls-v1")
    assert vectors.read_vector_file(path, MODEL)[2]["params"] == "cls-v1"         # not asked: not checked
    assert vectors.read_vector_file(path, MODEL, 256, "cls-v1")[0] == keys
    with pytest.raises(VectorFileError, match="other embedder parameters"):
        vectors.read_vector_file(path, MODEL, 256, "mean-v1")
    with pytest.raises(VectorFileError, match="other embedder parameters"):
        vectors.read_vector_file(path, MODEL, 256, "")                            # a backend without parameters

    monkeypatch.setattr(vectors, "CHECK_INTERVAL_S", 0.0)
    assert len(VectorIndex(MODEL, path, None, expected_params="cls-v1")) == 2
    stale = VectorIndex(MODEL, path, None, expected_params="mean-v1")
    assert len(stale) == 0 and "other embedder parameters" in stale.error

    # The builder writes its embedder's fingerprint and rebuilds a file that carries another one
    class Tuned(SpyEmbedder):
        params = "tuned-v2"

    embedder = Tuned()
    stats = vectors.sync_vector_file(path, embedder, [(DOC, text) for text in texts])
    assert stats["embedded"] == 2 and stats["kept"] == 0 and sorted(embedder.documents) == sorted(texts)
    assert vectors.read_vector_file(path, MODEL, 256, "tuned-v2")[2]["params"] == "tuned-v2"
    assert vectors.sync_vector_file(path, Tuned(), [(DOC, text) for text in texts])["embedded"] == 0


@pytest.mark.parametrize("damage", ["nan", "inf", "not_normalised", "zero_row"])
def test_file_with_unusable_vectors_is_refused(tmp_path, damage):
    """A cosine by dot product needs finite unit vectors; such a file would silently mis-rank."""
    path = tmp_path / "m.npz"
    texts = ["alpha beta", "gamma delta", "epsilon"]
    matrix = np.asarray(HashEmbedder().embed_documents(texts), dtype=np.float32)
    if damage == "nan":
        matrix[1, 3] = np.nan
    elif damage == "inf":
        matrix[1, 3] = np.inf
    elif damage == "not_normalised":
        matrix[1] *= 3.0
    else:
        matrix[1] = 0.0
    vectors.write_vector_file(path, MODEL, [vector_key(MODEL, DOC, text) for text in texts], matrix)
    with pytest.raises(VectorFileError, match="1 of 3 vectors are not finite and L2-normalised"):
        vectors.read_vector_file(path, MODEL)
    index = VectorIndex(MODEL, path, None)
    assert len(index) == 0 and "not finite" in index.error


def test_write_reaches_the_disk_before_the_rename(tmp_path, monkeypatch):
    order = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (order.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (order.append("replace"), real_replace(a, b))[1])
    _file(tmp_path / "m.npz", ["alpha"])
    assert order == ["fsync", "replace"]


def test_index_merges_the_committed_file_and_the_cache(tmp_path, monkeypatch):
    static, cache = tmp_path / "static.npz", tmp_path / "cache.npz"
    index = VectorIndex(MODEL, static, cache)
    assert len(index) == 0 and index.error is None and index.get("x") is None      # no files yet: fine
    assert index.matrix(["x"]) == ([], None)

    monkeypatch.setattr(vectors, "CHECK_INTERVAL_S", 0.0)
    static_keys = _file(static, ["committed chunk", "shared chunk"])
    cache_keys = _file(cache, ["shared chunk", "learned chunk"])
    assert len(index) == 3 and index.dim == 256
    assert static_keys[0] in index and cache_keys[1] in index
    vector = index.get(cache_keys[1])
    assert vector.dtype == np.float32 and float(vector @ vector) == pytest.approx(1.0, abs=1e-3)
    positions, matrix = index.matrix([static_keys[0], "missing", cache_keys[1]])
    assert positions == [0, 2] and matrix.shape == (2, 256) and matrix.dtype == np.float32
    assert [entry["vectors"] for entry in index.files()] == [2, 2]


def test_index_reloads_when_another_process_rewrites_a_file(tmp_path, monkeypatch):
    path = tmp_path / "cache.npz"
    index = VectorIndex(MODEL, None, path)
    _file(path, ["first"])
    assert len(index) == 1
    version = index.version
    new_key = _file(path, ["first", "second"])[1]
    os.utime(path, ns=(1, 1))                              # whatever the clock resolution
    assert new_key not in index                            # not checked again yet (throttled)
    monkeypatch.setattr(vectors, "CHECK_INTERVAL_S", 0.0)
    assert new_key in index and index.version > version
    assert index.version == index.version                  # unchanged file: no reload


@pytest.mark.parametrize("damage", ["corrupt", "other_model", "other_dimension"])
def test_unusable_file_is_ignored_not_fatal(tmp_path, damage, caplog):
    static, cache = tmp_path / "static.npz", tmp_path / "cache.npz"
    good = _file(static, ["committed chunk"])
    if damage == "corrupt":
        cache.write_bytes(b"\x00" * 100)
    elif damage == "other_model":
        _file(cache, ["x"], model="some-other-model")
    else:
        vectors.write_vector_file(cache, MODEL, [vector_key(MODEL, DOC, "x")], np.ones((1, 8)))
    index = VectorIndex(MODEL, static, cache)
    with caplog.at_level("WARNING"):
        assert len(index) == 1 and good[0] in index        # the good file still serves
    assert index.error.startswith("runtime cache cache.npz: ")       # which of the two files, and why
    assert sum("Ignoring vector file" in r.message for r in caplog.records) == 1
    assert len(index) == 1
    assert sum("Ignoring vector file" in r.message for r in caplog.records) == 1   # logged once per file state


def test_sync_embeds_only_what_is_missing(tmp_path):
    path = tmp_path / "cache" / "hash-256.npz"
    embedder = SpyEmbedder()
    wanted = [(DOC, "chunk one"), (DOC, "chunk two"), (QUERY, "a situation"), (DOC, ""), (DOC, "chunk one")]
    stats = vectors.sync_vector_file(path, embedder, wanted)
    assert stats["wanted"] == 3 and stats["embedded"] == 3 and stats["written"] is True
    assert sorted(embedder.documents) == ["chunk one", "chunk two"] and embedder.queries == ["a situation"]

    # Nothing changed: nothing embedded, file untouched
    before = path.stat().st_mtime_ns
    embedder = SpyEmbedder()
    stats = vectors.sync_vector_file(path, embedder, wanted)
    assert (stats["embedded"], stats["kept"], stats["dropped"], stats["written"]) == (0, 3, 0, False)
    assert embedder.documents == [] and path.stat().st_mtime_ns == before

    # One chunk edited, one deleted: one embedding, the stale vectors are dropped
    embedder = SpyEmbedder()
    stats = vectors.sync_vector_file(path, embedder, [(DOC, "chunk one, edited"), (QUERY, "a situation")])
    assert embedder.documents == ["chunk one, edited"] and embedder.queries == []
    assert (stats["embedded"], stats["kept"], stats["dropped"], stats["vectors"]) == (1, 1, 2, 2)
    keys, matrix, _ = vectors.read_vector_file(path, MODEL)
    assert set(keys) == {vector_key(MODEL, DOC, "chunk one, edited"), vector_key(MODEL, QUERY, "a situation")}
    assert matrix.dtype == np.float16

    # Keys the committed file already holds are not stored a second time
    embedder = SpyEmbedder()
    committed = {vector_key(MODEL, DOC, "in the committed file")}
    stats = vectors.sync_vector_file(
        path, embedder, [(DOC, "in the committed file"), (DOC, "new"), (QUERY, "a situation")],
        known=committed.__contains__)
    assert embedder.documents == ["new"] and stats["skipped"] == 1 and stats["vectors"] == 2


def test_sync_rebuilds_an_unusable_file_and_can_be_stopped(tmp_path):
    path = tmp_path / "hash-256.npz"
    path.write_bytes(b"garbage")
    stats = vectors.sync_vector_file(path, SpyEmbedder(), [(DOC, "a"), (DOC, "b")])
    assert stats["embedded"] == 2 and len(vectors.read_vector_file(path, MODEL)[0]) == 2

    # A stop request ends the build between batches; what was embedded is saved
    texts = [(DOC, f"chunk number {i}") for i in range(40)]
    calls = {"n": 0}

    def stop_after_first_batch():
        calls["n"] += 1
        return calls["n"] > 1

    stats = vectors.sync_vector_file(tmp_path / "partial.npz", SpyEmbedder(), texts,
                                     should_stop=stop_after_first_batch)
    assert 0 < stats["embedded"] < 40
    assert len(vectors.read_vector_file(tmp_path / "partial.npz", MODEL)[0]) == stats["embedded"]
    # ... and the next run finishes the job
    stats = vectors.sync_vector_file(tmp_path / "partial.npz", SpyEmbedder(), texts)
    assert stats["kept"] + stats["embedded"] == 40 == stats["vectors"]


def test_coverage(tmp_path):
    path = tmp_path / "x.npz"
    _file(path, ["a", "b", "c"])
    index = VectorIndex(MODEL, path, None)
    assert vectors.coverage(index, MODEL, ["a", "b", "c", "d"]) == 0.75
    assert vectors.coverage(index, MODEL, []) == 1.0 and vectors.coverage(None, MODEL, ["a"]) == 0.0
    assert vectors.coverage(index, MODEL, ["a"], kind=QUERY) == 0.0


def test_sync_removes_an_unusable_file_it_has_nothing_to_put_into(tmp_path, caplog):
    """A damaged cache whose every text lives in the committed file: nothing to embed, but the file must go."""
    path = tmp_path / "hash-256.npz"
    path.write_bytes(b"garbage")
    everything_known = lambda key: True                    # noqa: E731
    stats = vectors.sync_vector_file(path, SpyEmbedder(), [(DOC, "a"), (DOC, "b")], known=everything_known)
    assert (stats["embedded"], stats["skipped"], stats["vectors"], stats["written"]) == (0, 2, 0, False)
    assert not path.exists() and list(tmp_path.iterdir()) == []
    # The same for a file whose last wanted text went away
    _file(path, ["a"])
    assert vectors.sync_vector_file(path, SpyEmbedder(), [])["dropped"] == 1 and not path.exists()


def test_discard_unusable_only_removes_what_cannot_be_read(tmp_path, caplog):
    good, bad, missing = tmp_path / "good.npz", tmp_path / "bad.npz", tmp_path / "missing.npz"
    _file(good, ["a"])
    bad.write_bytes(b"\x00" * 64)
    with caplog.at_level("WARNING"):
        assert vectors.discard_unusable(good, MODEL, 256, "") is False and good.is_file()
        assert vectors.discard_unusable(missing, MODEL) is False
        assert vectors.discard_unusable(bad, MODEL) is True and not bad.exists()
        assert vectors.discard_unusable(good, MODEL, 256, "other-params") is True and not good.exists()
    assert sum("Removing unusable vector cache file" in r.message for r in caplog.records) == 2


def test_sync_does_not_store_vectors_it_could_not_serve(tmp_path):
    class Flaky(SpyEmbedder):
        def embed_documents(self, texts):
            out = super().embed_documents(texts)
            for text, vector in zip(texts, out):
                if text == "broken":
                    vector[0] = float("nan")
            return out

    path = tmp_path / "hash-256.npz"
    stats = vectors.sync_vector_file(path, Flaky(), [(DOC, "fine text"), (DOC, "broken"), (DOC, "!!! ???")])
    assert (stats["embedded"], stats["invalid"], stats["vectors"]) == (1, 2, 1)   # NaN, and a text without a token
    assert vectors.read_vector_file(path, MODEL)[0] == [vector_key(MODEL, DOC, "fine text")]


def test_sync_rebuilds_when_the_embedder_changes_its_dimension(tmp_path, caplog):
    """An API model swapped on the provider's side: unknown dimension up front, other vectors afterwards."""
    class Api:
        id = "openai:drift"
        dim = 0                                             # not known before the first call
        params = ""
        languages = frozenset({"*"})

        def __init__(self, dim):
            self.inner = HashEmbedder(dim)
            self.calls = []

        def embed_documents(self, texts):
            self.calls.extend(texts)
            return self.inner.embed_documents(texts)

        embed_queries = embed_documents

    path = tmp_path / "openai-drift.npz"
    old = [(DOC, "alpha beta"), (DOC, "gamma delta")]
    assert vectors.sync_vector_file(path, Api(64), old)["vectors"] == 2
    api = Api(128)
    with caplog.at_level("WARNING"):
        stats = vectors.sync_vector_file(path, api, old + [(DOC, "new text here")])
    assert sorted(api.calls) == ["alpha beta", "gamma delta", "new text here"]    # the stored ones again
    assert (stats["embedded"], stats["kept"], stats["dropped"], stats["vectors"]) == (3, 0, 2, 3)
    assert vectors.read_vector_file(path, "openai:drift")[1].shape == (3, 128)
    assert any("dimension of openai:drift changed from 64 to 128" in r.message for r in caplog.records)
    assert vectors.sync_vector_file(path, Api(128), old + [(DOC, "new text here")])["embedded"] == 0



# ---------------------------------------------------------------------------
# The build lock (one builder per vector file across processes)
# ---------------------------------------------------------------------------

_HOLD_LOCK = (
    "import sys\n"
    "from pathlib import Path\n"
    "from backend.app.ai.rag import vectors\n"
    "with vectors.build_lock(Path(sys.argv[1])) as held:\n"
    "    print('held' if held else 'not held', flush=True)\n"
    "    sys.stdin.readline()\n"            # until the test says so
)


def test_build_lock_makes_a_second_process_wait_for_the_first(tmp_path, monkeypatch):
    """A server's warm-up next to a `rag build`: the texts must not be embedded twice."""
    import subprocess
    import sys
    import threading

    pytest.importorskip("fcntl")
    monkeypatch.setattr(vectors, "LOCK_POLL_S", 0.01)
    path = tmp_path / "cache" / "hash-256.npz"
    texts = [(DOC, f"chunk number {i}") for i in range(20)]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    other = subprocess.Popen([sys.executable, "-c", _HOLD_LOCK, str(path)], cwd=root, text=True,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        assert other.stdout.readline().strip() == "held"
        assert (tmp_path / "cache" / "hash-256.npz.lock").is_file()
        # The other process "builds" the file while it holds the lock ...
        vectors.sync_vector_file(path, HashEmbedder(), texts)
        ours = SpyEmbedder()
        result: dict = {}

        def build():
            with vectors.build_lock(path) as held:
                result["held"] = held
                result["stats"] = vectors.sync_vector_file(path, ours, texts + [(DOC, "one more")])

        thread = threading.Thread(target=build)
        thread.start()
        thread.join(0.3)
        assert thread.is_alive() and not result            # ... and we wait for it
        other.stdin.write("\n")
        other.stdin.flush()
        thread.join(10)
        assert not thread.is_alive() and result["held"] is True
    finally:
        other.kill()
        other.wait(10)
        other.stdin.close()
        other.stdout.close()
    # What the first process built is kept; only the text it did not have is embedded here
    assert ours.documents == ["one more"]
    assert (result["stats"]["kept"], result["stats"]["embedded"]) == (20, 1)


def test_build_lock_gives_up_on_a_stop_request_and_is_released_afterwards(tmp_path, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    monkeypatch.setattr(vectors, "LOCK_POLL_S", 0.01)
    path = tmp_path / "hash-256.npz"
    polls = {"n": 0}

    def stop_on_the_third_poll():
        polls["n"] += 1
        return polls["n"] >= 3

    with vectors.build_lock(path) as first:
        assert first is True
        with vectors.build_lock(path, should_stop=stop_on_the_third_poll) as second:
            assert second is False and polls["n"] == 3
    with vectors.build_lock(path, should_stop=lambda: True) as again:    # released: no waiting, no stop check
        assert again is True
    # A lock file of another user (root's cron run): read access is enough to lock it
    os.chmod(tmp_path / "hash-256.npz.lock", 0o444)
    with vectors.build_lock(path) as readonly:
        assert readonly is True
    # No lock possible (the directory is a file): the build goes on unguarded
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("x")
    with vectors.build_lock(blocked / "hash-256.npz") as unguarded:
        assert unguarded is True
    assert fcntl is vectors.fcntl
