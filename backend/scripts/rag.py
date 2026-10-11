"""CLI for the knowledge retrieval stack (vectors, embedding model, benchmark).

Usage:
    python -m backend.scripts.rag status            # mode, vectors, coverage, files
    python -m backend.scripts.rag doctor            # packages, model checksum, a test embedding, and
                                                    # whether this machine reproduces the committed vectors
    python -m backend.scripts.rag download-model    # fetch the embedding model (HF_ENDPOINT honoured)
    python -m backend.scripts.rag build             # embed what the vector files lack (runtime cache)
    python -m backend.scripts.rag build --static    # rebuild the committed file backend/data/rag/<model>.npz
    python -m backend.scripts.rag query "how do I play against Lion" --hero Axe --lang en
    python -m backend.scripts.rag query --kind death_tf --position 1 --phase late_game
    python -m backend.scripts.rag eval              # benchmark table: baseline / lexical / hybrid

The server needs none of this to run: without vectors it ranks lexically.
The optional packages for ``doctor`` / ``download-model`` / ``build`` are in
backend/requirements-rag.txt. A running server picks rebuilt vector files up
by itself within a few seconds.

Exit codes: 0 = ok, 1 = the command failed or found a problem, 2 = bad arguments.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import osutil  # noqa: E402  (imports nothing from the project)

# One-shot commands never start background fetches or a learning run.
os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")


def _peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (0.0 where the platform does not report it)."""
    if sys.platform == "win32":
        return _peak_working_set_mb()
    try:
        import resource   # Unix only
    except ImportError:
        return 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def _peak_working_set_mb() -> float:
    """Windows: the peak working set of this process in MB (0.0 when the system call fails)."""
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):   # PROCESS_MEMORY_COUNTERS
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        # (kernel32 exports GetProcessMemoryInfo under this name since Windows 7)
        get_info = kernel32.K32GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if not get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return 0.0
        return round(counters.PeakWorkingSetSize / (1024 * 1024), 1)
    except Exception:   # a number for the build summary: never worth failing a finished build
        return 0.0


def _cmd_status(args) -> int:
    from backend.app.ai import knowledge_base
    from backend.app.ai.rag import runtime

    knowledge_base.warm_stores()   # the mode is judged against the built stores
    info = runtime.details()
    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0
    print(f"mode:              {info['mode']}")
    print(f"RAG_EMBEDDINGS:    {info['setting']}")
    print(f"vector space:      {info['model_id'] or '-'}")
    print(f"embedder:          {'available' if info['embedder_available'] else 'unavailable (' + info['embedder_reason'] + ')'}")
    print(f"numpy:             {'yes' if info['numpy'] else 'no (lexical only)'}")
    print(f"vectors:           {info['vectors']}  (situation queries: {info['situation_queries']})")
    print(f"static coverage:   {info['static_coverage']:.1%}")
    for entry in info["files"]:
        print(f"  {entry['path']}: {entry['vectors']} vectors, built {entry['built_at'] or '?'} on {entry['arch'] or '?'}")
    if not info["files"]:
        print(f"  no vector file (committed: {info['static_path']}, cache: {info['cache_path']})")
    if info["error"]:
        print(f"error:             {info['error']}")
    return 0


def _cmd_doctor(args) -> int:
    from backend.app.ai import knowledge_base
    from backend.app.ai.rag import embedders, runtime, situations, vectors

    ok = True
    cfg = runtime.settings()
    print(f"RAG_EMBEDDINGS={cfg.embeddings}  RAG_MODEL={cfg.model}")
    for name in ("numpy", "onnxruntime", "tokenizers", "openai"):
        try:
            module = __import__(name)
            print(f"  {name:12s} {getattr(module, '__version__', 'installed')}")
        except ImportError:
            needed = name == "numpy" or (name in ("onnxruntime", "tokenizers") and cfg.embeddings in ("auto", "onnx"))
            print(f"  {name:12s} not installed" + ("  (pip install -r backend/requirements-rag.txt)"
                                                    if name != "openai" and needed else ""))
    spec = embedders.MODEL_REGISTRY.get(cfg.model)
    if spec is None:
        print(f"model:  {cfg.model!r} is not in the registry ({', '.join(embedders.MODEL_REGISTRY)})")
        ok = False
    else:
        directory = cfg.model_dir / spec.model_id
        state = embedders.model_files_state(spec, directory, checksum=True)
        print(f"model:  {directory}: {state or 'present, checksum ok'}")
        if state == "model_missing":
            print("        run: python -m backend.scripts.rag download-model")
    embedder, reason = runtime.get_embedder(cfg)
    if embedder is None:
        print(f"embedder: unavailable ({reason}) - vectors cannot be built here; serving uses the files below")
    else:
        try:
            started = time.perf_counter()
            vector = embedder.embed_queries(["position 1 carry, laning phase: last hitting"])[0]
            norm = sum(v * v for v in vector) ** 0.5
            print(f"embedder: {embedder.id}, dim {len(vector)}, norm {norm:.3f}, "
                  f"first embedding {1000 * (time.perf_counter() - started):.0f} ms")
        except Exception as exc:
            print(f"embedder: {embedder.id} FAILED: {runtime.redact(exc)}")
            ok = False
        else:
            ok = _report_consistency(runtime, embedder) and ok
    knowledge_base.warm_stores()   # the mode is judged against the built stores
    info = runtime.details()
    for entry in info["files"]:
        print(f"vectors:  {entry['path']}: {entry['vectors']} vectors, built {entry['built_at'] or '?'} on {entry['arch'] or '?'}")
    if not info["files"]:
        print("vectors:  no vector file for this model")
    if info["error"]:
        print(f"          {info['error']}")
        ok = False
    print(f"mode:     {info['mode']} ({info['vectors']} vectors, {info['situation_queries']} of "
          f"{len(situations.situation_queries())} situation queries, "
          f"static coverage {info['static_coverage']:.1%})")
    if vectors.numpy_or_none() is None:
        ok = False
    return 0 if ok else 1


def _report_consistency(runtime, embedder) -> bool:
    """Print whether *embedder* reproduces the committed vectors; False when it does not."""
    try:
        check = runtime.embedder_consistency(embedder)
    except Exception as exc:
        print(f"consistency: FAILED: {runtime.redact(exc)}")
        return False
    if not check["checked"]:
        print(f"consistency: not checked ({check['reason']}): no committed vectors of this model to compare with")
        return True
    where = (f"file built on {check['file_arch'] or '?'}, this machine is {check['arch']}")
    print(f"consistency: {check['checked']} committed texts re-embedded, cosine to the stored vectors "
          f"min {check['min_cosine']:.4f} / mean {check['mean_cosine']:.4f} ({where}; required {check['threshold']})")
    if not check["ok"]:
        print("          this embedder does not reproduce the committed vectors: new texts and search\n"
              "          queries would land in a slightly different space. Keep RAG_EMBEDDINGS=auto without\n"
              "          the local model (the committed vectors alone still rank reports), or rebuild the\n"
              "          file on this machine: python -m backend.scripts.rag build --static")
    return bool(check["ok"])


def _cmd_download(args) -> int:
    from backend.app.ai.rag import embedders, runtime

    cfg = runtime.settings()
    spec = embedders.MODEL_REGISTRY.get(args.model or cfg.model)
    if spec is None:
        print(f"Unknown model {args.model or cfg.model!r}; known: {', '.join(embedders.MODEL_REGISTRY)}", file=sys.stderr)
        return 2
    directory = cfg.model_dir / spec.model_id
    try:
        embedders.download_model(spec, directory, endpoint=args.endpoint or "", progress=print)
    except embedders.EmbedderError as exc:
        print(f"Download failed ({exc.reason}): {exc}", file=sys.stderr)
        print(f"If this host cannot reach Hugging Face, set HF_ENDPOINT to a mirror or copy "
              f"{', '.join(spec.files)} into {directory}", file=sys.stderr)
        return 1
    print(f"Model ready in {directory}")
    return 0


@contextlib.contextmanager
def _committed_game_data():
    """Inside the block this process sees only the committed game data (no fetched files).

    The committed vector file must match the corpus a fresh checkout builds,
    not the one of this machine's last game-data fetch: the fetched-data
    directory is pointed at an empty temporary one, which is removed again.
    """
    from backend.app.ai import data_manager

    previous = data_manager.GAMEDATA_DIR
    with tempfile.TemporaryDirectory(prefix="rag-static-gamedata-") as empty:
        data_manager.GAMEDATA_DIR = Path(empty)
        data_manager.invalidate_all_caches()
        try:
            yield
        finally:
            data_manager.GAMEDATA_DIR = previous
            data_manager.invalidate_all_caches()


def _cmd_build(args) -> int:
    from backend.app.ai.rag import embedders, runtime

    started = time.perf_counter()
    embedder, reason = runtime.get_embedder()
    if embedder is None:
        print(f"No embedder available ({reason}). Install backend/requirements-rag.txt and run "
              f"`python -m backend.scripts.rag download-model`, or configure RAG_EMBED_API_*.", file=sys.stderr)
        return 1
    try:
        if args.static:
            with _committed_game_data():
                stats = runtime.build_static(embedder)
        else:
            stats = runtime.sync("cli")
    except embedders.EmbedderError as exc:
        print(f"Build failed ({exc.reason}): {exc}", file=sys.stderr)
        return 1
    if "error" in stats or "not_run" in stats:
        print(f"Build did not run: {stats.get('error') or stats.get('not_run')}", file=sys.stderr)
        return 1
    stats["total_seconds"] = round(time.perf_counter() - started, 2)
    stats["peak_rss_mb"] = _peak_rss_mb()
    print(json.dumps(stats, ensure_ascii=False))
    return 0


def _cmd_query(args) -> int:
    from backend.app.ai import knowledge_base
    from backend.app.ai.models import DecisionContext, DecisionPoint, HeroState
    from backend.app.ai.rag import runtime, situations

    store = knowledge_base._get_knowledge_store(args.lang)
    if store is None:
        print("The knowledge store could not be built.", file=sys.stderr)
        return 1
    if args.kind:
        text = situations.situation_text(args.position, args.phase, args.kind)
        if not text:
            print(f"Unknown --kind {args.kind!r}; known: {', '.join(situations.TOPICS)}", file=sys.stderr)
            return 2
        scores = runtime.query_scores(store, text)
        print(f"situation query: {text}")
        if not scores:
            print("no vectors for this query (lexical mode)")
            return 1
        for chunk_id, score in sorted(scores.items(), key=lambda kv: -kv[1])[: args.top_k]:
            print(f"  {score:.4f}  {chunk_id}")
        return 0
    if not args.text:
        print("Give a question, or --kind for a situation query.", file=sys.stderr)
        return 2
    hero = args.hero or "Axe"
    ctx = DecisionContext(
        decision_point=DecisionPoint(time=600, point_type="death", player_slot=0, hero_name=hero,
                                     severity="major", brief=""),
        player_state=HeroState(hero_name=hero, player_slot=0, team="radiant"),
        game_phase=args.phase or "",
    )
    if args.position:
        ctx.match_context["player_position"] = args.position
    result = knowledge_base.search_knowledge(
        ctx, args.text, lang=args.lang, topic=args.topic or "", top_k=args.top_k,
        match_heroes=[] if not args.match_heroes else args.match_heroes.split(","))
    print(result)
    print(f"-- mode={result.retrieval.get('mode')} reason={result.retrieval.get('reason') or '-'}")
    for ref in result.refs:
        print(f"   {ref.ref} {ref.score:.4f} {ref.chunk_id}")
    return 0


def _cmd_eval(args) -> int:
    from backend.app.ai.rag import evalkit, runtime

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in evalkit.MODES]
    if unknown:
        print(f"Unknown mode(s) {unknown}; known: {', '.join(evalkit.MODES)}", file=sys.stderr)
        return 2
    try:
        bench = evalkit.load_benchmark(Path(args.benchmark) if args.benchmark else None)
    except (OSError, ValueError, KeyError) as exc:
        print(f"Benchmark not readable ({exc}); it lives in backend/tests/data/rag_benchmark.", file=sys.stderr)
        return 1
    stores = evalkit.build_stores(bench)
    results = [evalkit.evaluate(bench, stores, mode, handles=not args.no_handles, top_k=args.top_k)
               for mode in modes]
    spaces = runtime.decision_indexes()
    print(f"{len(bench.queries)} queries; vector space(s): "
          + (", ".join(f"{index.model_id} ({len(index)} vectors)" for index in spaces) or "none (lexical only)"))
    print(evalkit.format_table(results, split=args.split))
    for result in results:
        extra = f", ranked with vectors: {result['hybrid_queries']}/{result['queries']}" if result["mode"] == "hybrid" else ""
        print(f"{result['mode']}: {result['rank_ms']} ms per ranking, zh/en top-10 overlap "
              f"{evalkit.language_overlap(result):.2f}, unjudged chunks in a top 10: {result['unjudged@10']}{extra}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            [{k: v for k, v in r.items()} for r in results], ensure_ascii=False, indent=1), encoding="utf-8")
    hybrid = next((r for r in results if r["mode"] == "hybrid"), None)
    if hybrid is not None and hybrid["hybrid_queries"] == 0:
        print("note: no vectors were available, the hybrid row is lexical")
    return 0


def main() -> int:
    # Output redirected to a file or pipe on Windows is encoded with the system code page:
    # a character it lacks is escaped instead of ending the command.
    osutil.harden_stdio()
    parser = argparse.ArgumentParser(description="Knowledge retrieval (RAG) tools")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Mode, vectors and coverage")
    status.add_argument("--json", action="store_true", help="Print everything as JSON")
    sub.add_parser("doctor", help="Check packages, model files and vector files")
    download = sub.add_parser("download-model", help="Fetch the embedding model files")
    download.add_argument("--model", help="Registry key (default: RAG_MODEL)")
    download.add_argument("--endpoint", help="Hugging Face endpoint (default: $HF_ENDPOINT or huggingface.co)")
    build = sub.add_parser("build", help="Embed missing chunks / queries")
    build.add_argument("--static", action="store_true",
                       help="Rebuild the committed file for the static corpus and the situation queries")
    query = sub.add_parser("query", help="Debug a free-text search or a situation query")
    query.add_argument("text", nargs="?", help="Free-text question")
    query.add_argument("--lang", choices=("zh", "en"), default="en")
    query.add_argument("--hero", help="Coached hero (free-text search)")
    query.add_argument("--match-heroes", help="Comma-separated heroes of the match (default: no restriction)")
    query.add_argument("--topic", help="matchup|laning|teamfight|vision|items|farming|objectives|general")
    query.add_argument("--kind", help="Situation topic code (death_tf, death_pick, bad_trade ...)")
    query.add_argument("--position", type=int, choices=(1, 2, 3, 4, 5))
    query.add_argument("--phase", choices=("pregame", "laning", "mid_game", "late_game"))
    query.add_argument("--top-k", type=int, default=6)
    evaluate = sub.add_parser("eval", help="Run the retrieval benchmark")
    evaluate.add_argument("--modes", default="baseline,lexical,hybrid")
    evaluate.add_argument("--benchmark", help="Benchmark directory (default: backend/tests/data/rag_benchmark)")
    evaluate.add_argument("--no-handles", action="store_true",
                          help="Do not charge the [K#] lines against the budget (as when the labels were made)")
    evaluate.add_argument("--top-k", type=int, default=0, help="Cap on selected chunks (0 = none)")
    evaluate.add_argument("--split", action="store_true", help="Also print rows per language and stratum")
    evaluate.add_argument("--json", help="Write the per-query results to this file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    handler = {
        "status": _cmd_status, "doctor": _cmd_doctor, "download-model": _cmd_download,
        "build": _cmd_build, "query": _cmd_query, "eval": _cmd_eval,
    }[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
