"""``python -m backend.scripts.rag``: status, doctor, build, query, eval (hash embedder, no network)."""
import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import json
import sys
import tempfile

import pytest

from backend.app.ai import data_manager, knowledge_base
from backend.app.ai.rag import embedders, runtime, situations, vectors
from backend.scripts import rag as cli

QUERIES = 216   # situation queries: 6 roles x 5 phases x 7 topics + the rune fight of the 6 roles


@pytest.fixture
def rag_env(tmp_path, monkeypatch):
    from backend.app.ai.learning import base

    gamedata = tmp_path / "gamedata"
    gamedata.mkdir()
    learned = tmp_path / "learned"
    learned.mkdir()
    monkeypatch.setattr(data_manager, "GAMEDATA_DIR", gamedata)
    monkeypatch.setattr(base, "LEARNED_DIR", learned)
    monkeypatch.setattr(runtime, "STATIC_DIR", tmp_path / "static")
    monkeypatch.setattr(vectors, "CHECK_INTERVAL_S", 0.0)
    monkeypatch.setenv("RAG_EMBEDDINGS", "hash")
    monkeypatch.setenv("RAG_INDEX_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("RAG_MODEL_DIR", str(tmp_path / "models"))
    for name in ("RAG_MODEL", "RAG_EMBED_API_BASE_URL", "RAG_EMBED_API_KEY", "RAG_EMBED_API_MODEL", "RAG_TOP_K"):
        monkeypatch.delenv(name, raising=False)
    data_manager.invalidate_all_caches()
    runtime.reset()
    yield tmp_path
    assert runtime.wait_idle(30)
    runtime.reset()
    data_manager.invalidate_all_caches()


def _run(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["rag", *argv])
    code = cli.main()
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_status_and_build(rag_env, monkeypatch, capsys):
    code, out, _ = _run(monkeypatch, capsys, "status")
    assert code == 0 and "mode:              lexical" in out and "no vector file" in out

    code, out, _ = _run(monkeypatch, capsys, "build")
    stats = json.loads(out)
    assert code == 0 and stats["embedded"] > 900 and stats["written"] is True and stats["peak_rss_mb"] > 0
    assert (rag_env / "cache" / "hash-256.npz").is_file()
    code, out, _ = _run(monkeypatch, capsys, "build")
    assert code == 0 and json.loads(out)["embedded"] == 0

    code, out, _ = _run(monkeypatch, capsys, "status", "--json")
    info = json.loads(out)
    assert code == 0 and info["mode"] == "hybrid" and info["situation_queries"] == QUERIES
    assert info["static_coverage"] == 1.0 and info["embedder_available"] is True
    # A one-shot process has no stores yet: the command builds them, or it could only ever say "lexical"
    knowledge_base.reset_knowledge_stores()
    code, out, _ = _run(monkeypatch, capsys, "status")
    assert code == 0 and "mode:              hybrid" in out and "error" not in out


def test_static_build_uses_the_committed_game_data(rag_env, monkeypatch, capsys):
    # Fetched game data on this machine must not leak into the committed file
    fetched = rag_env / "gamedata"
    (fetched / "items.json").write_text(json.dumps({"Blink Dagger": {
        "key": "blink", "cost": 99999, "description": "FETCHED-ONLY TEXT", "is_active": True}}), encoding="utf-8")
    (fetched / "meta.json").write_text(json.dumps({"patch_version": "9.99"}), encoding="utf-8")
    data_manager.invalidate_all_caches()
    fetched_texts = knowledge_base.static_index_texts()
    assert "99999 gold" in fetched_texts["item:blink"]                # what this machine would embed
    scratch = rag_env / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    seen = []
    real_build = runtime.build_static
    monkeypatch.setattr(runtime, "build_static", lambda embedder=None, path=None: (
        seen.append((data_manager.GAMEDATA_DIR, knowledge_base.static_index_texts())), real_build(embedder, path))[1])
    code, out, _ = _run(monkeypatch, capsys, "build", "--static")
    stats = json.loads(out)
    target = rag_env / "static" / "hash-256.npz"
    assert code == 0 and stats["path"] == str(target) and stats["embedded"] == stats["wanted"]
    (looked_at, texts), = seen
    assert looked_at != fetched and looked_at.parent == scratch       # the build looked elsewhere ...
    assert "99999 gold" not in texts["item:blink"] and "FETCHED-ONLY TEXT" not in texts["item:blink"]
    index = vectors.VectorIndex("hash-256", target, None)
    assert len(index) == stats["wanted"] == len(set(texts.values())) + QUERIES
    assert vectors.coverage(index, "hash-256", texts.values()) == 1.0
    assert vectors.coverage(index, "hash-256", situations.situation_queries(), kind=vectors.QUERY) == 1.0
    assert not (rag_env / "cache").exists()                           # the runtime cache is not involved
    # ... and left nothing behind: the empty directory is gone, this process sees its own data again
    assert list(scratch.iterdir()) == [] and data_manager.GAMEDATA_DIR == fetched
    assert knowledge_base.static_index_texts() == fetched_texts


@pytest.mark.parametrize("packages, reason", [(True, "model_missing"), (False, "deps_missing")])
def test_build_without_an_embedder_explains_itself(rag_env, monkeypatch, capsys, packages, reason):
    # With and without the optional packages (onnxruntime / tokenizers), whatever this venv holds
    monkeypatch.setattr(embedders, "_importable", lambda *modules: packages)
    monkeypatch.setenv("RAG_EMBEDDINGS", "auto")
    code, out, err = _run(monkeypatch, capsys, "build")
    assert code == 1 and reason in err and "requirements-rag.txt" in err and "download-model" in err
    monkeypatch.setenv("RAG_EMBEDDINGS", "off")
    assert _run(monkeypatch, capsys, "build", "--static")[0] == 1


def test_doctor_reports_without_failing(rag_env, monkeypatch, capsys):
    code, out, _ = _run(monkeypatch, capsys, "doctor")
    assert code == 0
    assert "RAG_EMBEDDINGS=hash" in out and "model_missing" in out and "embedder: hash-256, dim 256" in out
    assert "no vector file for this model" in out and "mode:     lexical" in out
    assert "consistency: not checked (no_committed_file)" in out
    monkeypatch.setenv("RAG_MODEL", "no-such-model")
    code, out, _ = _run(monkeypatch, capsys, "doctor")
    assert code == 1 and "is not in the registry" in out


def test_doctor_checks_that_this_embedder_reproduces_the_committed_vectors(rag_env, monkeypatch, capsys):
    """The committed file is built on one machine and served everywhere; new texts are embedded locally."""
    assert _run(monkeypatch, capsys, "build", "--static")[0] == 0
    code, out, _ = _run(monkeypatch, capsys, "doctor")
    assert code == 0 and "consistency: 25 committed texts re-embedded" in out
    assert "cosine to the stored vectors min " in out and "required 0.999" in out and "mode:     hybrid" in out
    check = runtime.embedder_consistency()
    assert check["checked"] == 25 and check["ok"] and check["file_arch"] == check["arch"]
    assert check["min_cosine"] >= 0.9995                              # float16 rounding, nothing else

    # Same model name, other vectors (another pooling, a dropped prefix, other kernels): doctor fails
    class Shifted(embedders.HashEmbedder):
        def _embed(self, text):
            return super()._embed("shifted " + text)

    monkeypatch.setattr(runtime, "get_embedder", lambda cfg=None: (Shifted(), ""))
    check = runtime.embedder_consistency()
    assert check["checked"] == 25 and not check["ok"] and check["min_cosine"] < 0.999
    code, out, _ = _run(monkeypatch, capsys, "doctor")
    assert code == 1 and "does not reproduce the committed vectors" in out
    # A file built on another machine type is compared with the wider tolerance
    monkeypatch.setattr("platform.machine", lambda: "riscv128")
    check = runtime.embedder_consistency(embedders.HashEmbedder())
    assert (check["threshold"], check["ok"], check["arch"]) == (0.98, True, "riscv128")
    assert check["file_arch"] != "riscv128"


def test_query_and_eval(rag_env, monkeypatch, capsys):
    _run(monkeypatch, capsys, "build")
    code, out, _ = _run(monkeypatch, capsys, "query", "--kind", "death_tf", "--position", "1",
                        "--phase", "late_game", "--top-k", "3")
    assert code == 0 and out.startswith("situation query: position 1 carry, late game: teamfight positioning")
    assert len(out.strip().splitlines()) == 4
    assert _run(monkeypatch, capsys, "query", "--kind", "nonsense")[0] == 2
    assert _run(monkeypatch, capsys, "query")[0] == 2
    code, out, _ = _run(monkeypatch, capsys, "query", "how do I fight Lion", "--hero", "Axe",
                        "--match-heroes", "Axe,Lion", "--top-k", "2")
    assert code == 0 and "[K1] " in out and "-- mode=hybrid" in out

    report = rag_env / "eval.json"
    code, out, _ = _run(monkeypatch, capsys, "eval", "--modes", "baseline,lexical", "--json", str(report))
    assert code == 0 and "78 queries; vector space(s): hash-256 (" in out and "| lexical | " in out
    assert "ndcg@10" in out
    assert [r["mode"] for r in json.loads(report.read_text(encoding="utf-8"))] == ["baseline", "lexical"]
    code, _, err = _run(monkeypatch, capsys, "eval", "--modes", "magic")
    assert code == 2 and "Unknown mode" in err


def test_bad_arguments(rag_env, monkeypatch, capsys):
    assert _run(monkeypatch, capsys, "download-model", "--model", "no-such-model")[0] == 2
    monkeypatch.setattr(sys, "argv", ["rag", "frobnicate"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
