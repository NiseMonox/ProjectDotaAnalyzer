# Deployment Run-Book

Example target: an Ubuntu 22.04 server (login user `ubuntu`, Python 3.10), written `<server-ip>` below.
Install path: `/opt/dota-analyzer`.
URL without a reverse proxy: `http://<server-ip>:8001`.

## First-time install

```bash
# System deps (Java 17 runs the replay parser)
sudo apt-get update
sudo apt-get install -y openjdk-17-jdk-headless python3.10-venv python3-pip git

# App
sudo mkdir -p /opt/dota-analyzer && sudo chown ubuntu:ubuntu /opt/dota-analyzer
git clone https://github.com/NiseMonox/ProjectDotaAnalyzer.git /opt/dota-analyzer
cd /opt/dota-analyzer
python3 -m venv backend/venv
backend/venv/bin/pip install -U pip
backend/venv/bin/pip install -r backend/requirements.txt   # everything the server needs (wheels only, no compiler)

# Replay parser jar (tools/clarity-analyzer.jar): committed, nothing to build.
# This only checks that Java 17 is there and the jar starts:
bash tools/setup_parser.sh
# Do NOT rebuild the jar on the server. A build with any JDK other than the
# one it was committed with (Temurin 17.0.13+11) is not byte-identical, git
# then sees the tracked jar as modified and a later `git pull` that brings a
# new jar is refused. Rebuilding is for developers who changed
# tools/clarity-analyzer/:  REBUILD=1 bash tools/setup_parser.sh
#   (MAVEN_MIRROR=https://maven.aliyun.com/repository/public for a host that
#   cannot reach Maven Central; GRADLE_CMD=<path> for another Gradle, also
#   where services.gradle.org is unreachable. Read from the environment only.)

# Configuration + secrets (see .env.example for every option)
umask 077
cp .env.example .env
# edit .env: set at least DEEPSEEK_API_KEY and ADMIN_TOKEN, e.g.
#   DEEPSEEK_API_KEY=sk-...
#   ADMIN_TOKEN=$(openssl rand -hex 32)
chmod 600 .env

# Systemd
sudo cp deploy/dota-analyzer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dota-analyzer
systemctl is-active dota-analyzer
```

Open TCP 8001 in the cloud provider's firewall (security group) before expecting the app to answer from the public internet.

Run exactly **one** uvicorn worker (the unit does): rate limits, the AI-coach
daily quota, caches and in-flight de-duplication are kept per process.

## Configuration that matters on a public server

All settings are environment variables, read from `/opt/dota-analyzer/.env`
(documented in `.env.example`). The important ones:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | empty | LLM for the AI coach; empty = rule-based reports only |
| `ADMIN_TOKEN` | empty | Required (`X-Admin-Token` header) for `/api/admin/*` and `/api/parse-local/*`; empty = those endpoints return 403 |
| `AI_COACH_MODEL` | `deepseek-flash` | Model of the coach and of the learning job (the name the endpoint's `/models` lists). One LLM-backed report measured 14-31 requests, 53-137k prompt and 38-75k completion tokens, 60-140 s (six runs on 2026-10-11, table in the top-level `README.md`) |
| `AI_COACH_DAILY_LIMIT` | `200` | Max cold LLM-backed coach reports per UTC day (0 = unlimited); cached reports are free |
| `AI_COACH_DEGRADED_TTL_S` / `AI_COACH_REFRESH_COOLDOWN_S` | `600` / `60` | Seconds a degraded report (the LLM failed, rules were used) is served from memory, and seconds before a reader may force a new run of it ("generate again", `?refresh=1`); a forced run counts as a cold report for the two limits around this row |
| `AI_COACH_MAX_TOKENS` / `AI_COACH_PHASE_MAX_TOKENS` | `8192` / `4096` | `max_tokens` of one decision evaluation / one phase review (512-16384); the model's hidden reasoning is billed inside it |
| `RATE_LIMIT_UPLOAD` | `10/3600` | Per-IP uploads per window (`count/seconds`, `0` disables) |
| `RATE_LIMIT_FETCH` | `30/600` | Per-IP OpenDota fetches (cache hits are free) |
| `RATE_LIMIT_REQUEST_PARSE` | `10/600` | Per-IP OpenDota parse requests |
| `RATE_LIMIT_AI_COACH` | `10/3600` | Per-IP cold AI-coach generations while an LLM is configured (cached reports are free) |
| `RATE_LIMIT_AI_COACH_RULES` | `200/600` | The same for rule-based reports (no `DEEPSEEK_API_KEY`): they cost milliseconds, and one match has ten players in two languages; the short window keeps the wait after a refusal to minutes |
| `MAX_UPLOAD_MB` | `300` | Replay size limit (HTTP 413 beyond it), for the upload as sent and for what a compressed upload unpacks to |
| `UPLOAD_MAX_CONCURRENCY` / `UPLOAD_MAX_PER_CLIENT` | `3` / `1` | Uploads received at once in total (503 beyond) and per client IP (429 beyond) |
| `UPLOAD_RECEIVE_TIMEOUT_S` / `UPLOAD_IDLE_TIMEOUT_S` | `900` / `60` | Seconds a client has to send the whole upload, and seconds it may send nothing in between, before it is dropped (HTTP 400 `upload_timeout`; its upload slot is free again) |
| `REPLAY_MAX_FILES` | `50` | Uploaded replays kept on disk (oldest deleted) |
| `MATCH_CACHE_MAX_FILES` | `200` | Parsed matches kept in `cache/matches/` |
| `TRUST_PROXY_HEADERS` | `0` | Use `X-Real-IP` / `X-Forwarded-For` as the client IP (only behind nginx) |
| `ALLOWED_ORIGINS` | empty | CORS allowlist; empty = same-origin only (the bundled frontend) |
| `CLARITY_MAX_CONCURRENCY` / `CLARITY_JVM_XMX` | `1` / `1536m` | Parallel replay parses and JVM heap per parse — size to the server's RAM |
| `OPENDOTA_API_KEY` | empty | Optional OpenDota key: higher rate limits for match fetches, parse requests and the learning job |
| `THREADPOOL_SIZE` | `32` | Worker threads for blocking request work (analysis, disk, OpenDota calls) |
| `GAMEDATA_AUTO_UPDATE` / `GAMEDATA_RECHECK_HOURS` | `1` / `24` | Refresh the game data at startup and then every so many hours (`0` hours = only at startup, otherwise at least `0.25`) |
| `RAG_EMBEDDINGS` | `auto` | Knowledge retrieval of the AI coach: `auto` uses the prebuilt vectors plus an embedder if one is installed, `off` = lexical ranking only (see "Knowledge retrieval") |

Rate-limited requests get HTTP 429 with `{"detail", "code", "retry_after"}`
and a `Retry-After` header. A blank `RATE_LIMIT_*=` means the default; only
`0` / `off` disables a limit. Every error under `/api/` has the same
`{"detail": <zh/en text>, "code"}` body.

The interactive API documentation (`/docs`, `/redoc`, `/openapi.json`) is
only served with `DEV=1`: it lists every route, the admin ones included.

A replay that was cut off is still analysed (`match_overview.partial`), but
the AI coach refuses one that ends before the game starts or within its
first five minutes (HTTP 409 `coach_no_game`; `match_overview.coach_available`
is false), and compares a later cut only on what the recording covers.

Uploads that are refused (too large, rate-limited, parser missing, busy) are
answered before the body is read, and the connection is kept open so the
browser, which only reads the answer after it sent the file, still receives
the JSON. Only a body declared larger than twice `MAX_UPLOAD_MB` is cut off.

Replays may be uploaded as they come from Valve's replay servers: a file
named `.dem`, `.dem.bz2` or `.dem.zst` whose content is a demo file, bzip2 or
Zstandard (those servers now send Zstandard under the `.dem.bz2` name; the
content decides, not the name). The server unpacks it in a worker thread and
stores the plain `.dem` under its content hash, so the same match uploaded
both ways is parsed once. Measured on real replays: bzip2 unpacks at about
32 MB/s (2.1 s for a 68 MB replay), Zstandard at about 400 MB/s, with less
than 20 MB of extra memory either way. An archive is unpacked in pieces of
at most 4 MiB and counted against `MAX_UPLOAD_MB` as it grows, so a
decompression bomb never holds more than about 8 MiB of unpacked data in
memory, or more than `MAX_UPLOAD_MB` on disk, before it is refused
(measured with archives of the highest ratio the format allows: +32 MB of
process memory for one upload, +83 MB for three at once). bzip2 needs
nothing extra; Zstandard needs the `zstandard` package from
`requirements.txt` — without it such an upload gets HTTP 400
`replay_compressed` asking the user to unpack it first (as a bzip2 upload
gets `replay_compressed_bzip2` on a Python built without the `bz2` module).
A damaged or cut-off archive gets `replay_decompress_failed`; an intact
Zstandard file written with a window above 32 MiB (`zstd --ultra` /
`--long`) gets `replay_window_too_large`, because the window is memory the
server would have to allocate.

Admin calls:

```bash
TOKEN=$(grep ^ADMIN_TOKEN= /opt/dota-analyzer/.env | cut -d= -f2-)
curl -sS -H "X-Admin-Token: $TOKEN" http://127.0.0.1:8001/api/admin/status   # caches, quota, learning levels, game data, retrieval index
curl -sS -H "X-Admin-Token: $TOKEN" http://127.0.0.1:8001/api/admin/gamedata-status
curl -sS -X POST -H "X-Admin-Token: $TOKEN" "http://127.0.0.1:8001/api/admin/update-gamedata?force=true"
```

## Game data and AI-coach knowledge from the command line

The server refreshes the game data by itself at startup and then once a day
(`GAMEDATA_RECHECK_HOURS`), and runs the learning job at startup and after a
check that brought new data (see Diagnostics). To do it by hand, as the
service user or as root, from `/opt/dota-analyzer`:

```bash
backend/venv/bin/python -m backend.scripts.update_gamedata   # heroes, abilities, items, talents, patch tables
backend/venv/bin/python -m backend.scripts.update_gamedata --check-only   # version, source, whether an update is due
backend/venv/bin/python -m backend.scripts.learn             # the learning levels that are due
backend/venv/bin/python -m backend.scripts.learn --status    # what was learned, what is due
```

- The game data comes from the dotaconstants master branch on GitHub
  (`raw.githubusercontent.com`), then its jsDelivr mirror, then the OpenDota
  API as a last resort (that one serves the npm release, which lags the game
  by months). The game's full version with its letter patch (7.41f) comes
  from Valve's patch feed (`www.dota2.com/datafeed/patchnoteslist`).
  Outbound HTTPS hosts of this step, in the order they are asked:
  `raw.githubusercontent.com`, `cdn.jsdelivr.net`, `api.opendota.com` (the
  data, first one that answers), `www.dota2.com` (the game version) and
  `api.github.com` (the revision of the data, optional). Every one has a
  fallback and none is needed to serve: the host needs at least one of the
  three data sources to get new data at all. `/api/admin/status` shows
  under `gamedata` which version (`patch_label`), source and age the server
  is on, and whether the data matches the current version (`data_current`).
- Stored data is never replaced by worse data. While the master branch is
  unreachable, data that came from it is kept instead of the OpenDota API's
  older release, and a download that lost more than a tenth of the heroes,
  items or talents is refused as a broken upstream build (the update then
  fails with `Stored game data kept (...)` or `Implausible constants: ...`
  in the log). When such a loss is real, accept it with
  `update_gamedata --allow-shrink` or `POST /api/admin/update-gamedata?force=true&allow_shrink=true`.
- The public `/api/capabilities` names the game version (`game_patch`) only
  while Valve's feed was read within the last `GAMEDATA_STALE_DAYS`
  (`gamedata.checked_at` in the admin status). On a server that never
  reached it, or has been offline or on `GAMEDATA_AUTO_UPDATE=0` for longer,
  it is `null` rather than a version that may be outdated.
- A running server picks the new files up by itself within about 30 seconds;
  no restart is needed after either command.
- Only one learning run can run at a time (a lock file under
  `backend/data/learned/`, also across users). A second one — the CLI while
  the server's startup run is still going, or the other way round — ends at
  once with `already_running` (CLI exit code 1) and learns nothing; run it
  again later.
- Learning levels 1 and 2 only call OpenDota (no key needed, no cost): 635
  calls for level 1 (five per hero) and about 105 for level 2, against
  OpenDota's limits of 60 calls a minute and 3000 a day per address without
  `OPENDOTA_API_KEY`. Measured on 2026-10-11: level 1 16 minutes for its 635
  calls (none failed; six 30 s pauses for the minute limit), level 2 3.3
  minutes for 100 professional matches. A call refused for the minute limit
  is repeated after a 30 s pause (the server's own match fetches count
  against the same limit), so a run gets slower instead of failing; only a
  spent daily limit ends level 1 as `partial`, and it is due again then.
- Level 1 saves what it has every ten heroes. A run that was stopped (a
  restart, a failed network) is continued by the next one within 24 hours
  after the heroes it had finished, instead of asking for all of them again.
- `learn --status` (and `learning.next_due` in the admin status) say when
  each level is due next: seven days after its last run, earlier when the
  game patch changes.
- Level 2 is also where the coach's "pro reference" for last hits and kill
  participation comes from (`backend/data/learned/_mining_aggregates.json`:
  the mean per position, its top quarter as "excellent"). Without that file
  the tables in `backend/app/ai/pro_patterns.py` are used, which hold the
  values of the run of 2026-10-10.
- Level 3 writes notes that state the mined numbers and what to compare with
  them. A note that judges the reader by a professional average ("below 65
  means slow farming") is dropped when it is written and when stored files
  are loaded, so `llm_distilled.json` of an earlier run needs no manual
  clean-up; it is replaced when level 3 runs again.
- Learning level 3 is one LLM request (about 30 s); level 4 one per video.
- Learning level 4 (video transcripts) needs `yt-dlp` on the `PATH`. It is
  optional and not in `requirements.txt`: without it (or without
  `DEEPSEEK_API_KEY`) the level is skipped and the other levels still run.
  Bilibili subtitles additionally need a cookies file (`BILIBILI_COOKIES_FILE`).
  Search results are only used when their title names a Dota lesson (guide,
  coaching, replay analysis, 教学, 复盘, 上分 ...); videos pinned under
  `videos` in `backend/data/learning_sources.json` are taken as they are.
  YouTube answers anonymous subtitle requests from some addresses with HTTP
  429 (on 2026-10-11: two of three videos); those videos are not marked as
  done and are tried again on the next run.
- The game data files carry a format version. After a deploy that changes it,
  the first start (or `update_gamedata`) downloads the data once more even on
  the same patch; until then the older files keep being used.
- The patch shown for an uploaded replay comes from the patch tables in the
  game data (a replay only knows when it was played): the full version with
  its letter (7.41b) when Valve's release list is there, else the major
  version. The label is left empty for a match played more than
  `GAMEDATA_STALE_DAYS` after the tables were last downloaded, because a
  newer patch could have been released since; the daily check keeps them
  current.

## Knowledge retrieval (RAG)

The AI coach puts reference knowledge next to every decision it reviews. The
knowledge is ranked by tags and BM25, and for the general (not hero- or
item-specific) part additionally by embedding similarity when vectors exist.
`/api/capabilities` reports `knowledge_mode` (`hybrid` with vectors, else
`lexical`); `/api/admin/status` has the details under `rag` (`mode`,
`embedder`, `vectors`, `chunks`, `static_coverage`, `building`, `last_build`,
`error`, `embedder_available`, `embedder_reason`).

`embedder` names the vector space the ranking uses (the model the committed
file was built with), not something that is installed: it reads the same on
a fresh install without the optional packages. Whether this server can embed
new texts itself is `embedder_available`; when it is false,
`embedder_reason` says why (`deps_missing`: the packages of
`requirements-rag.txt` are not installed, `model_missing`: no downloaded
model, `off`, `api_not_configured`, ...).

Nothing has to be installed for this: the repository contains a prebuilt
vector file (`backend/data/rag/<model>.npz`, about 1 MB) for the built-in and
editorial knowledge and the situation queries, and the server only does a
numpy dot product with it. No model is loaded for serving reports.

An embedder is only needed for texts that are not in that file — knowledge
written by the self-learning job, texts changed by a game-data update — and
for the tool-calling agent's free-text search. Without one those texts stay
retrievable through their tags and keywords and get a neutral dense credit
(0.5, the same as hero and item texts); the optional model, or a
`cache/rag/<model>.npz` copied from a machine that built it, gives them
vectors.

Without an embedder the agent's free-text search is **keyword-only**: the
question's own words decide (the heroes and items it names, topic words such
as "smoke" or "buyback", and BM25 over the English text of the entries), and
no meaning beyond the words is matched. Measured on 15 questions with an
obvious answer entry, that entry was among the first four for all 15
keyword-only and for 14 with the local model, in both report languages.

To add an embedder, either

```bash
# (a) a local model: about 35 MB on disk; a vector build peaked at 170-240 MB of memory when measured
backend/venv/bin/pip install -r backend/requirements-rag.txt     # onnxruntime + tokenizers (wheels, no compiler)
backend/venv/bin/python -m backend.scripts.rag download-model    # into backend/data/models/ (checksums pinned)
#     behind a firewall: HF_ENDPOINT=https://hf-mirror.com ... download-model,
#     or copy backend/data/models/<model>/ from another machine (scp -r)
backend/venv/bin/python -m backend.scripts.rag doctor            # packages, model checksum, a test embedding, and
                                                                 # whether this machine reproduces the committed vectors
backend/venv/bin/python -m backend.scripts.rag build             # embed what is missing into cache/rag/
backend/venv/bin/python -m backend.scripts.rag status            # mode, coverage, files
```

or (b) set `RAG_EMBED_API_BASE_URL`, `RAG_EMBED_API_KEY` and
`RAG_EMBED_API_MODEL` to an OpenAI-compatible embeddings endpoint (its own
key; the coach's `DEEPSEEK_API_KEY` is never used for embeddings). Vectors of
different models are never mixed: the whole knowledge base, about a thousand
short texts, is embedded through the API once and cached in `cache/rag/`.
Until the API's vectors cover the knowledge base, and whenever the API cannot
be reached, decisions keep being ranked with the prebuilt file (the status
then shows `embedder` = the local model's name and the API problem under
`error`).

Run `rag doctor` once on the server before relying on the local model: it
re-embeds 25 texts of the committed file and compares them with the stored
vectors (the file was built on arm64; an int8 model can differ slightly on
other CPUs, and behaviour on x86 has not been verified). On a mismatch it
exits 1, and the server keeps ranking with the committed vectors and does
not embed new texts (`error`: `embedder_mismatch`).

- The server never downloads a model. With an embedder available it embeds
  missing texts in a background thread at startup and after a learning run
  or game-data update, never inside a request. With `RAG_WARM_UP=0` the
  server embeds nothing at all (not at startup and not after those
  updates): run `rag build` from the command line or cron then. Only
  missing texts are embedded, files are replaced atomically and a running
  server notices new files by itself. After the warm-up the server logs one
  line with the mode it ended up in (`Knowledge retrieval: hybrid ranking,
  993 vectors (model ...), static coverage ..., error: none`; a warning
  when the index reports an error). Whenever the mode decisions are really
  ranked with changes later — also on a server without the warm-up — one
  more line says so (`Knowledge retrieval: decisions are now ranked
  lexically (low_coverage: ...)`, a warning when it fell back from hybrid).
- A command-line build next to a running server is safe: a lock file
  (`cache/rag/<model>.npz.lock`) lets only one process embed at a time, the
  other waits and then finds the vectors already there. `learn` embeds the
  chunks it wrote at the end of a full run when the model is installed.
- `static_coverage` below 1.0 is expected on a server without an embedder
  once a game-data update changed hero and item texts (0.86 after the
  first download on a fresh install, when measured): the prebuilt file was
  made from the texts in the repository. Decision ranking is not affected,
  because only the general knowledge is ranked by vectors and those texts do
  not change with the game data; an embedder (or `rag build`) brings the
  number back to 1.0.
- The local model is loaded only while it is needed and unloaded after
  `RAG_MODEL_IDLE_UNLOAD_S` (15 minutes). On a host with little memory keep
  `CLARITY_JVM_XMX` in mind: a replay parse and a vector build can coincide.
- `cache/rag/` can be deleted at any time; it is rebuilt when an embedder is
  available. The committed file is rebuilt by developers with
  `python -m backend.scripts.rag build --static` after the built-in or
  editorial knowledge changed (a test fails when it no longer covers it).
- `RAG_EMBEDDINGS=off` switches all of it off (lexical ranking only).
- `error` in the status is a short code, sometimes followed by `: detail`:
  `numpy_missing`, `unknown_model`, `api_not_configured`, `model_missing`,
  `deps_missing`, `api_unreachable`, `embedder_mismatch`, `low_coverage`,
  `no_query_vectors`, `degenerate_vectors`, or `committed file <name>: ...` /
  `runtime cache <name>: ...` for a vector file that cannot be read. It can
  be set while `mode` is still `hybrid` (the committed vectors serve).

## Upgrading to the 7.41f / retrieval release

Nothing has to be migrated by hand, with one check before the pull: this
release brings a new parser jar, and a jar that was rebuilt on the server
(the install steps used to do that) differs from the committed one and makes
`git pull` stop.

```bash
cd /opt/dota-analyzer
git status --short tools/clarity-analyzer.jar              # prints " M tools/clarity-analyzer.jar" when it was rebuilt here
git checkout -- tools/clarity-analyzer.jar                 # only then: back to the committed jar
git pull
backend/venv/bin/pip install -r backend/requirements.txt   # new: numpy, zstandard (wheels for Python 3.10-3.13)
sudo systemctl restart dota-analyzer
```

- **Replay parser**: `tools/clarity-analyzer.jar` is committed and was
  rebuilt against clarity 4.0.3 (still Java 17; clarity 5 would need Java
  21). The previous jar stops early on some replays of current game builds.
  Nothing to build: `bash tools/setup_parser.sh` now only checks Java and
  the jar (`REBUILD=1` builds, for developers).
- **Game data**: the file format version changed and the data now comes from
  dotaconstants master instead of the OpenDota API, so the first start
  downloads it once more (about 3 MB). Until that succeeds the old files keep
  working. Check `gamedata.source` and `gamedata.patch_label` in
  `/api/admin/status` afterwards.
- **Caches**: parsed matches (`cache/matches/`) and AI-coach reports
  (`cache/coaching/`) written by the previous version are ignored and rebuilt
  on demand (the match store format was bumped, and reports are cached per
  format). Old files go away as they are met or by the normal pruning.
  Vector files of an older format in `cache/rag/` are removed and rebuilt by
  the next build.
- **AI-coach reports** are payload version 5. Since version 4 every
  evaluation lists the knowledge entries behind it (`sources`, with `cited`
  for the ones the model named) and the report says how they were retrieved
  (`knowledge`). Version 5 adds the report's own `lang`, writes hero, item
  and ability names in that language (Chinese names in a zh report) and
  leaves `analysis` / `mistake` / `better_play` empty where there is nothing
  to say (a positive moment); a served report also carries
  `refresh: {available, retry_after}`, which says whether a degraded report
  can be generated again (`?refresh=1`, see `AI_COACH_REFRESH_COOLDOWN_S`).
  The frontend of this release reads all of it; reports cached as version 4
  or older are not served. The text of a listed entry is read with
  `GET /api/knowledge/{id}?lang=zh|en` (the `id` of a `sources` item; 404
  `knowledge_not_found` once a later learning run has dropped the entry).
  With a tool-calling model the agent may additionally search the knowledge
  base `RAG_AGENT_SEARCH_LIMIT` times per evaluation (default 3, at most 6,
  0 = off).
- **Learned knowledge**: output of a quick or under-sampled learning run is no
  longer loaded (a hero's item timings need 10 pro games and 8 purchases per
  item now), and neither are LLM notes that judge the reader by a pro
  average. Run `backend/venv/bin/python -m backend.scripts.learn --levels 1,2 --force`
  once (free OpenDota calls, about 20 minutes) to learn per-hero benchmarks for
  the current patch; the startup job does the same when it is due. The
  coach rates GPM and damage against a hero's own percentiles only when
  `backend/data/learned/_benchmarks_raw.json` has them split into the hero's
  core and support games (`roles`, written by level 1 of this release); a
  file of an earlier run leaves the role tables in charge until level 1 has
  run again.
- **New settings** (all optional, see `.env.example`): `GAMEDATA_RECHECK_HOURS`,
  `RATE_LIMIT_AI_COACH_RULES`, `AI_COACH_MAX_TOKENS`, `AI_COACH_PHASE_MAX_TOKENS`,
  `AI_COACH_REFRESH_COOLDOWN_S`,
  `RAG_EMBEDDINGS`, `RAG_MODEL`, `RAG_MODEL_DIR`, `RAG_EMBED_THREADS`,
  `RAG_MODEL_IDLE_UNLOAD_S`, `RAG_EMBED_API_BASE_URL` / `_KEY` / `_MODEL`,
  `RAG_INDEX_DIR`, `RAG_TOP_K`, `RAG_AGENT_SEARCH_LIMIT`, `RAG_WARM_UP`.
- **Optional**: `backend/requirements-rag.txt` and a downloaded model for
  embedding learned knowledge (see "Knowledge retrieval"). Not needed for the
  server to run or for hybrid ranking of the built-in knowledge.
- **`AI_COACH_MODEL`**: the default is `deepseek-flash` (the name the
  endpoint lists; `deepseek-v4-flash` in an existing `.env` keeps working as
  an alias). The learning job now uses the same default.
- **After the next game patch** (developers): run
  `python -m backend.scripts.gen_static_catalogs`, bump `DEFAULT_PATCH_LABEL`
  / `DEFAULT_PATCH_RELEASED` in `backend/app/ai/data_manager.py` when the
  label changed; for a new major version also review the rule values in
  `backend/app/game_constants.py` (XP and respawn tables, building vision,
  ward and smoke fallbacks, skill-point schedule) and bump its `RULES_PATCH`,
  which a test ties to `DEFAULT_PATCH_LABEL`. Then run the test suite (it
  names editorial texts that quote changed numbers), run
  `python -m backend.scripts.rag build --static` and commit the catalogs, the
  fallback files and `backend/data/rag/`. The same run refreshes the chat-wheel
  phrases (`frontend/static/data/chat_wheel.json`, English and Chinese, from
  Valve's game files as mirrored in `dotabuff/d2vpkr` on GitHub, about 10 MB
  of downloads); when that mirror cannot be reached the phrases of the
  previous run are kept.
- **nginx**: no change. Uploads may now be named `.dem.bz2` / `.dem.zst`.

## Redeploy after a `git push`

A clone made before the history rewrite of 2026-10 cannot `git pull` once:
follow section 9 of [`docs/WINDOWS.md`](../docs/WINDOWS.md) (fetch, then
`git reset --hard origin/master`) the first time.

```bash
cd /opt/dota-analyzer
git pull
backend/venv/bin/pip install -r backend/requirements.txt   # no-op when unchanged
sudo systemctl restart dota-analyzer
journalctl -u dota-analyzer -n 50 --no-pager
```

Check `.env.example` for new settings after pulling. The parser jar comes
with the pull (it is committed together with `tools/clarity-analyzer/`):
never rebuild it on the server. If `git pull` stops at
`tools/clarity-analyzer.jar`, the jar was rebuilt here once: run
`git checkout -- tools/clarity-analyzer.jar` and pull again.

Cached results never outlive the code that produced them: parsed matches in
`cache/matches/` carry a store format version (`MatchStore.FORMAT` in
`backend/app/storage.py`) and AI-coach reports are cached per format and report
schema. After an update that changes parser output, old entries are ignored
and rebuilt when they are needed again (uploads are parsed again from
`replays/`, the demo is rebuilt, an OpenDota match is downloaded again by the
next fetch); nothing has to be deleted by hand.

## Diagnostics

```bash
systemctl status dota-analyzer
journalctl -u dota-analyzer -f             # follow logs (LOG_LEVEL=DEBUG for more)
journalctl -u dota-analyzer --since "10 min ago"
ss -tlnp | grep 8001                       # is it listening?
curl -sS http://127.0.0.1:8001/api/demo | head -c 200
```

On startup the app refreshes game data in the background first and only then
runs the AI-coach self-learning job (set `GAMEDATA_AUTO_UPDATE=0` /
`AI_COACH_AUTO_LEARN=0` to skip either; with `DEV=1` the learning job is
never started by the server, because every reload would restart it). The
game-data check is repeated every `GAMEDATA_RECHECK_HOURS` (three small
requests; the data itself is only downloaded when the game version or the
data source changed), and the learning job runs again when such a check
brought new data. A start within 15 minutes of the last check does not check
again (restarts in a row would otherwise use up GitHub's 60 anonymous
requests an hour). None of this
blocks serving requests or a restart: a job still running at shutdown is
told to stop and is not waited for (its files are written atomically); what
it did not finish is due again on the next start.
`/api/admin/status` shows `startup_maintenance`, the learning levels, the
game data (`gamedata`) and the retrieval index (`rag`).

## Rollback

```bash
sudo systemctl stop dota-analyzer
sudo systemctl disable dota-analyzer
# Optional: sudo rm -rf /opt/dota-analyzer
```

Nothing else on this host is affected — the app is self-contained.

## Later: add a domain + TLS (nginx)

When a domain is ready (e.g. `dota.example.com` → `<server-ip>`):

1. In the service unit change `--host 0.0.0.0` to `--host 127.0.0.1`, then
   `sudo systemctl daemon-reload && sudo systemctl restart dota-analyzer`.
2. `sudo apt-get install -y nginx` and install `deploy/nginx-dota-analyzer.conf`
   (instructions inside; set `server_name`, and let `sudo nginx -t` pass before
   the reload — the file is an example that has not run on this host yet). It
   already has the important bits:
   `client_max_body_size` at twice `MAX_UPLOAD_MB` (so the app's own localized
   "too large" answer is what users get), `proxy_buffering off` and a
   long `proxy_read_timeout` for the AI-coach SSE stream, long timeouts for
   uploads, and `X-Real-IP` / `X-Forwarded-For` headers.
3. Set `TRUST_PROXY_HEADERS=1` in `.env` so per-IP rate limits see real client
   IPs. Only do this when uvicorn listens on 127.0.0.1 — otherwise clients
   could reach port 8001 directly and spoof those headers.
4. `sudo certbot --nginx -d dota.example.com -m you@example.com --agree-tos --redirect`.
5. Close TCP 8001 in the security group; only 80/443 should stay open publicly.
6. CORS is off by default (same-origin). Only if another origin must call the
   API, set `ALLOWED_ORIGINS=https://other.example.com` in `.env`.
