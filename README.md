# Dota 2 Match Analyzer

A web application that turns one Dota 2 match into a full post-game review. It reads a replay
file (`.dem`) with its own replay parser, or fetches a public match from OpenDota by its ID, and
shows the match on 15 tabs: scoreboard and draft, laning, combat, farm, items, economy, skill
builds, objectives, vision, smokes, teamfights with a positional playback, chat, and an **AI
coach** tab. The coach reviews one player's deaths and key decisions. With an LLM key it is a
tool-calling agent that looks up evidence in the match data and in a retrieved knowledge base
and names the entries it used; without a key it is a rule engine that says so on screen. The
interface is in Chinese and English.

A short Chinese summary is at [the end](#中文简介).

## What is in it

| Layer | What it does | Where |
|---|---|---|
| Replay analyzer | Java 17 program on the [Clarity](https://github.com/skadistats/clarity) library. One pass over a `.dem` file gives one JSON document: scoreboard, kill log, wards, smokes, teamfights, per-minute series, positions every 5 s. | `tools/clarity-analyzer/` (source), `tools/clarity-analyzer.jar` (committed build), `tools/clarity-analyzer/OUTPUT_SCHEMA.md` (every field) |
| Parsers | One interface, three sources: the analyzer's JSON, the OpenDota API, generated mock data. All of them produce the same `MatchData`. | `backend/app/parser.py`, `backend/app/models.py`, `backend/app/mock_data.py` |
| Analyzer | Turns `MatchData` into the payload the page shows (one section per tab). Recomputed when a stored match is loaded, so cached matches follow the current code. | `backend/app/analyzer.py` |
| Server | FastAPI. Upload (also `.dem.bz2` / `.dem.zst`), fetch by match ID, per-IP rate limits, a daily quota for LLM reports, versioned disk caches, localized errors with stable codes, an SSE stream for report progress. | `backend/app/routes.py`, `security.py`, `storage.py`, `main.py` |
| AI coach | Finds decision points (deaths, bad trades, missed farm, overextensions, fight entries, good plays), builds the context of each (state of the ten heroes, a 60 s event timeline, a vision / fog reconstruction, a causal chain for deaths), evaluates it and writes the report. | `backend/app/ai/` (`coach.py`, `decision_detector.py`, `context_builder.py`, `vision.py`, `causal_chain.py`, `evaluator.py`) |
| Agent | The evaluation of a decision is an LLM tool-calling loop: nine evidence tools (player state, nearby heroes, recent events, match situation, vision and fog, death cause, coaching knowledge, hero matchup, benchmarks) and a terminal `submit_evaluation`. A verdict is checked before it is accepted; a failed run falls back to rules and the report is marked as degraded. | `backend/app/ai/agent.py` |
| Knowledge retrieval (RAG) | Reference entries are ranked for each decision by tags, BM25 and, for the general entries, embedding similarity. The vectors of the built-in corpus are committed, so serving needs numpy only and no model. Every evaluation lists its entries (`sources`, with `cited` for the ones the model named), and the page can open each entry's text. | `backend/app/ai/knowledge_base.py`, `knowledge_store.py`, `knowledge_chunks.py`, `rag/`, `backend/data/rag/` |
| Self-learning | Four levels that add knowledge without hand-writing it: (1) OpenDota aggregate statistics, (2) statistics mined from professional matches, (3) an LLM summary of level 2, (4) tips distilled from coaching-video transcripts, off unless `AI_COACH_VIDEO_LEARN=1` (it searches Bilibili and YouTube and downloads subtitles). Levels 1 and 2 cost nothing; every learned entry is labelled with its source. | `backend/app/ai/learning/`, `backend/scripts/learn.py` |
| Frontend | A single page without a build step: Vue 3, ECharts and Tailwind as pinned copies served by the app, seven plain script files. | `frontend/index.html`, `frontend/static/js/`, `frontend/static/vendor/` |

Data flow:

```
.dem file ──> clarity-analyzer.jar ──> raw JSON ─┐
match ID  ──> OpenDota API ──────────────────────┼─> parser ──> MatchData ──> analyzer ──> page (15 tabs)
(no input) ─> mock data ─────────────────────────┘                  │
                                                                    v
   decision detection ──> context / vision / causal chain ──> knowledge retrieval ──> agent with tools ──> report with sources
                                                                    ^
                                             built-in + curated entries, learned statistics (self-learning job)
```

The built-in knowledge base is 777 entries, the corpus the committed vector file covers: 55
coaching principles written for this project and 722 curated notes on heroes, matchups,
abilities and items. The self-learning job adds statistics on top; after a full run on the
development machine (2026-10-11) the store held 1,593 entries.

## Run it

Requirements: Python 3.10 to 3.13. Java 17 or newer for replay uploads (without it the page
still fetches matches by ID). An LLM key is optional.

```bash
./run.sh                       # creates backend/venv, installs backend/requirements.txt, serves http://localhost:8000
DEV=1 ./run.sh                 # the same with auto-reload
bash tools/setup_parser.sh     # checks Java and that the committed parser jar starts
```

- **Windows.** `.\run.ps1` in PowerShell does what `run.sh` does. The guide for developing on
  Windows is [`docs/WINDOWS.md`](docs/WINDOWS.md) (in Chinese): what to install, every command in
  PowerShell form, the local data a clone does not contain, and what differs from macOS / Linux.
- **Demo.** "Open the demo match" on the start page parses `replays/8713267082.dem` when that
  file exists. The `replays/` directory is not part of the repository (replays are 40 to 70 MB),
  so in a fresh clone the demo is generated mock data: an invented match, without chat and
  without benchmarks. To see a real match, enter a match ID (for example `8713267082`) or upload
  a replay.
- **AI coach without a key** writes rule-based reports (templated sentences, no citations) and
  labels them as such. Set `DEEPSEEK_API_KEY` in `.env` for LLM-written reports. The client
  speaks the OpenAI protocol (`DEEPSEEK_BASE_URL`, `AI_COACH_MODEL`); only DeepSeek has been
  run live.
- **Configuration.** Every setting is an environment variable with a default, documented in
  [`.env.example`](.env.example) (copy it to `.env`). Running it on a server (systemd unit,
  nginx, limits, game-data updates, the learning job, upgrading) is covered in
  [`deploy/README.md`](deploy/README.md).

## Tests

```bash
backend/venv/bin/pip install -r backend/requirements-dev.txt     # pytest, httpx
DEEPSEEK_API_KEY="" backend/venv/bin/python -m pytest -q
```

The suite needs no network and no key: `backend/tests/conftest.py` blocks every connection that
is not to the loopback interface, so a test that forgets to stub OpenDota or the LLM fails
instead of calling out, and it keeps `.env` from being read, so the key and the settings in it
stay out of the tests. Result on 2026-10-11 (macOS): **1731 passed, 6 skipped**. Three of the
skipped tests need a downloaded embedding model or a directory of real replays, named by
`RAG_TEST_MODEL_DIR` and `CLARITY_TEST_REPLAYS`; the other three test the Windows scripts and
need PowerShell. A fresh clone skips two more until the game data has been fetched once.
The Java analyzer has its own JUnit tests under `tools/clarity-analyzer/src/test/`.

On Windows the interpreter of the virtual environment is `backend\venv\Scripts\python.exe`; the
commands in PowerShell form are in [`docs/WINDOWS.md`](docs/WINDOWS.md). Every push to GitHub runs
the suite on Windows, Linux and macOS, a smoke test of the real server and the command-line
tools, and the analyzer's JUnit tests (`.github/workflows/ci.yml`).

Related commands:

```bash
backend/venv/bin/python -m backend.scripts.gen_static_catalogs --check   # the committed hero / item / ability catalogs are current
backend/venv/bin/python -m backend.scripts.rag eval                      # the retrieval benchmark below
backend/venv/bin/python -m backend.scripts.rag status                    # retrieval mode, vectors, coverage
python3 tools/compare_with_opendota.py <raw.json> <opendota.json>        # analyzer output against OpenDota's
```

## Measured results

All numbers were measured on 2026-10-11.

**Analyzer against OpenDota.** `tools/compare_with_opendota.py` compares the analyzer's output
for a replay with OpenDota's JSON of the same match, value by value. Analyzer 1.5.0, five replays:

| Replay | Game version | Differences | Equal checks | Explained notes |
|---|---|---|---|---|
| 8762775364 | 7.41b | 0 | 778 | 31 |
| 8713267082 | 7.40c | 0 | 644 | 16 |
| 9038370833 | 7.41f | 0 | 577 | 18 |
| 9038374031 | 7.41f | 0 | 694 | 20 |
| 9038388762 (Turbo) | 7.41f | 0 | 772 | 20 |

A note is a difference with a known, legitimate reason; `OUTPUT_SCHEMA.md` lists them (for
example: OpenDota counts five Wraith King reincarnations as kills, and has no strategy-time
purchases for 8762775364). The same document records the comparison on four more 7.41f replays.

**Retrieval benchmark.** 78 labelled queries: 66 decision points, nearly all from the two
reference replays, 12 of them in both report languages. For each query every knowledge entry
was graded 0 to 3 against a written rubric (`backend/tests/data/rag_benchmark/`). The benchmark
replays the queries through the production ranker, and the test suite fails when lexical
ranking falls below its measured quality or hybrid ranking is worse than lexical.

| Ranking | nDCG@10 | Irrelevant entries in the knowledge block |
|---|---|---|
| Baseline: the ranker the labels were collected against (tags + BM25) | 0.801 | 9.8 % |
| Lexical: plus situation-tag rules, no vectors | 0.856 | 9.9 % |
| Hybrid: plus embedding similarity for the general entries (committed vector file) | 0.891 | 1.7 % |

Ranking one decision takes about 1.2 ms in hybrid mode. The embedding model is
`bge-small-en-v1.5` (int8); it is needed only to embed new texts, not to serve reports.

**Live LLM reports.** Six reports were generated against the DeepSeek API with the model
`deepseek-flash`. The first four ran on an earlier report format of the same day, the last two
on the current one (version 5).

| Mode | Match, hero, language | Requests | Prompt tokens (cached) | Completion tokens | Time | Outcome |
|---|---|---|---|---|---|---|
| Agent | 8762775364 replay, Wraith King, zh | 27 | 104,775 (55 %) | 38,202 | 60.5 s | complete |
| Agent | 8713267082 via OpenDota, Axe, zh | 30 | 125,435 (58 %) | 42,549 | 85.3 s | complete |
| Agent | 8713267082 via OpenDota, Axe, zh | 31 | 129,988 (66 %) | 44,132 | 76.2 s | complete |
| Single prompt | 8762775364 replay, Hoodwink, en | 14 | 52,688 (0 %) | 60,536 | 81.7 s | complete |
| Agent, v5 | 8762775364 replay, Wraith King, zh | 28 | 136,904 (59 %) | 41,176 | 65.5 s | complete; 8 of 9 evaluations and all 4 phase reviews written by the model, 23 cited entries |
| Single prompt, v5 | 9038370833 replay (7.41f), Disruptor, zh | 17 | 88,364 (7 %) | 75,135 | 139.7 s | degraded: 6 answers were cut off at the token limit, one evaluation fell back to rules and is marked |

No request failed in any of the six runs. At most eight decisions of a report go to the model
(`MAX_LLM_EVALS`); the others are rule-based and marked. Most completion tokens are the model's
hidden reasoning (30,600 to 69,342 per report).

**Rule-based reports and the page.** 100 rule-based reports (the two reference replays, the
same two matches from OpenDota's data, and the mock; ten players each; both languages) hold
1,068 evaluations: none broke a report invariant, and each of the 172 knowledge entries they
list (counted per language) could be read back by its id. A browser pass with headless Chrome
over 480 screen states (the demo replay, an uploaded replay, a Turbo replay and a replay cut
four seconds after the horn; every tab and sub-view; both languages; 1280 px and 390 px wide)
showed no console error or warning, no failed request and no layout overflow. Uploading the
68 MB reference replay to the local server took 3.0 s, parsing included.

## Limits

- An install without an LLM key shows the rule engine, not the agent: the sentences are
  templated and the listed knowledge entries were retrieved, not cited.
- The curated hero and item notes are written in English and are shown in English also in the
  Chinese interface; names and titles follow the language.
- Hero and item icons come from Steam's CDN and an optional web font from Google Fonts, so the
  page is not fully usable offline. Vue, ECharts and Tailwind are served by the app itself.
- The Benchmarks tab has data only for matches fetched from OpenDota. Smokes, laning details and
  the teamfight playback need a replay.
- Turbo and other special modes get no benchmark comparison, and lane efficiency is not computed
  for Turbo.
- A teamfight follows OpenDota's definition (three or more deaths in a chain), so an early lane
  skirmish can be called a teamfight.
- One server process: rate limits, the daily quota and the caches are kept per process.

## Repository map

```
backend/app/            FastAPI app: routes, parser, analyzer, storage, security
backend/app/ai/         AI coach: detection, context, vision, causal chain, evaluator, agent, knowledge base
backend/app/ai/rag/     vectors, embedders, free-text search, the benchmark kit
backend/app/ai/learning/  the four learning levels
backend/data/           curated notes (editorial/), committed vectors (rag/), fallback tables
backend/scripts/        command-line tools: learn, rag, update_gamedata, gen_static_catalogs
backend/tests/          the test suite, fixtures, the retrieval benchmark
frontend/               index.html and static/ (js, css, catalogs of heroes / items / abilities)
tools/                  the Java replay analyzer, its jar, the OpenDota comparison tool
deploy/                 run-book, systemd unit, nginx example
```

## Credits

- [Clarity](https://github.com/skadistats/clarity) by Martin Schrodt: the replay parsing library
  the analyzer is built on.
- [OpenDota](https://www.opendota.com) and [dotaconstants](https://github.com/odota/dotaconstants):
  the match API, the reference the analyzer is compared with, the game data and the map image.
- [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5): the embedding model
  behind the committed vectors.
- Dota 2 and its hero, item and ability names and images belong to Valve Corporation. This
  project is not affiliated with Valve.

No licence has been chosen for this repository yet.

## 中文简介

这是一个 Dota 2 单场比赛复盘工具：上传录像文件（`.dem`，也支持 `.dem.bz2` / `.dem.zst`），或输入比赛
ID 从 OpenDota 获取公开比赛，即可在 15 个分析页中查看整场比赛，包括总览与阵容、对线、战斗、经济、出装、
加点、目标、视野、开雾、团战回放、聊天，以及 **AI 教练**。

- **录像解析**：基于 Clarity 自行编写的 Java 解析器。在五场真实录像（7.40c、7.41b、7.41f，含一场加速模式）
  上与 OpenDota 的数据逐项对比，差异为 0。
- **AI 教练**：先找出死亡和关键决策点，再为每个决策点重建当时的局面（十名英雄的状态、此前 60 秒的事件、
  视野与迷雾、死亡的因果链）。配置大模型密钥后，评价由一个会调用工具的智能体完成（九个取证工具加一个
  提交结论的工具）；没有密钥时由规则引擎生成，并在页面上明确标注。
- **知识检索（RAG）**：按标签、BM25 和向量相似度为每个决策点检索参考知识。在 78 条按书面评分细则标注的查询上，
  nDCG@10 从 0.801（原有排序）提高到 0.891（混合检索）。每条评价都列出所依据的知识条目，点开即可阅读原文。
- **自学习**：四个层级自动补充知识，分别是 OpenDota 统计、职业比赛数据挖掘、大模型归纳和教学视频字幕提炼，
  每条学到的内容都标明来源。第 4 层会搜索 B 站和 YouTube 并下载字幕，默认关闭，
  设 `AI_COACH_VIDEO_LEARN=1` 才运行。
- **运行**：`./run.sh`（Windows 上是 `.\run.ps1`，详见 [`docs/WINDOWS.md`](docs/WINDOWS.md)），需要
  Python 3.10 至 3.13；上传录像还需要 Java 17。配置项见 `.env.example`，服务器部署见
  `deploy/README.md`。仓库不包含录像文件，因此刚克隆下来时“演示比赛”是生成的模拟数据。
- **测试**：`backend/venv/bin/python -m pytest -q`，无需联网和密钥；2026-10-11 在 macOS 上的结果为
  1731 项通过、6 项跳过。GitHub Actions 在 Windows、Linux 和 macOS 上运行同一套测试。
