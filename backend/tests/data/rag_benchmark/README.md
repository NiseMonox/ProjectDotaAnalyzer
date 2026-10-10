# Coach knowledge retrieval benchmark

78 labelled queries = 66 real decision points (replays 8762775364 and 8713267082, all ten players, plus two
mock fight-engage points) x report language (38 zh, 40 en; 12 points appear in both languages). Labelled
2026-10-10 against the 832-chunk knowledge corpus of that day. No player names or account ids are stored: a
record is the situation (hero, position, phase, point type, killer, nearby heroes, items).

| file | content |
|---|---|
| `queries.json` | `meta` + one record per query: the situation fields retrieval reads, the token budget that was left for retrieved chunks (`remaining_budget`), item tips already printed elsewhere in the block (`printed_item_chunks`), the query tags of the ranker at labelling time (`baseline_tags`) and the slug lists the rubric reads |
| `labels.json` | per query, every chunk graded >= 1 (`{chunk_id: grade}`); a chunk that is not listed is grade 0. Complete over the labelled corpus, not pooled |
| `corpus.json` | `ids`: the chunk ids of the labelled corpus per language; `learned`: the learned layer of that day (58 chunks per language), replayed instead of whatever `backend/data/learned` holds |
| `RUBRIC.md` | the grading rubric (0-3) |
| `judge.py` | the rubric as code; `python backend/tests/data/rag_benchmark/judge.py` checks `labels.json` against it, `--write` regrades after the corpus changed |

How it is replayed (`backend/app/ai/rag/evalkit.py`): the decision context is rebuilt from the record and goes
through the production path (`build_query` -> `KnowledgeStore.rank` -> `select` under `remaining_budget`).
The static chunks come from the repository as it is today, so editing a principle or a tag moves the numbers;
chunks added after labelling are left out of the scored lists (`unjudged`).

    python -m backend.scripts.rag eval                 # baseline / lexical / hybrid table
    python -m backend.scripts.rag eval --split         # also per language and stratum

Metrics: nDCG with gain `2^grade - 1`; `p2` = share of returned chunks graded >= 2; `irr` = share graded 0;
`rec2` = recall of the chunks graded >= 2; `@5` / `@10` on the ranking, `@B` on what fits the token budget.

Measured 2026-10-11, third pass (bge-small-en-v1.5 int8 vectors, `[K#]` lines charged against the budget,
committed game data, committed vector file, no embedding model: the state of a fresh deployment):

| mode | nDCG@10 | nDCG@B | irr@B | rec2@B | chunks@B |
|---|---|---|---|---|---|
| baseline (query tags of the labelling run, lexical formula) | 0.801 | 0.802 | 0.098 | 0.759 | 14.3 |
| lexical (situation-tag rules) | 0.857 | 0.852 | 0.099 | 0.774 | 14.4 |
| hybrid (+ pooled dense term, relative cut-off 0.5) | 0.891 | 0.903 | 0.017 | 0.780 | 12.7 |
| hybrid, learned chunks embedded too (a server with the optional model) | 0.891 | 0.905 | 0.021 | 0.777 | 12.6 |

The third pass scoped the learned farm and item-timing statistics (`topic:economy`): they are ranked for a hero's
own player only and halved for every decision that is not about farming. The rows before it were 0.798 / 0.855 /
0.889 / 0.890 at 10 and 0.109 / 0.107 / 0.026 / 0.030 irrelevant chunks in the block; in 240 rule-based reports over nine replays
such statistics were 9.5 % of the sources listed under deaths and 32 % of those under good plays, and are none
now (22 % of the sources of missed-CS points, where they belong). One chunk added after labelling
(`item:consecrated_wraps`) now reaches one hybrid top 10 and is left out of the scored list.

Before the second pass the rows read 0.795 / 0.840 / 0.873 (nDCG@10). What moved them: the rune-fight topic and
tag weights for the pregame phase (that slice: hybrid 0.54 -> 0.72 at 10, 0.62 -> 0.77 at the budget, lexical
0.53 -> 0.69), the teamfight-positioning principle no longer tied to mid / late game (teamfight deaths 0.75 ->
0.81), a neutral dense credit for principle chunks that have no vector (missed CS 0.87 -> 0.89; 0.93 once they
are embedded), untrusted text capped at priority 2 and at two entries per block.

`backend/tests/test_ai_rag_benchmark.py` keeps the lexical row within 0.01 of these values, requires the
hybrid row (committed vector file, numpy only) to be at least as good, and replays a vector file that covers
only 50 % / 10 % of the principles (the result must equal the lexical row: below 80 % coverage the dense term
is not used).

Limits: one judge, rubric written as code; the situation-tag rules were written after looking at this set's
misses, so their gain is optimistic; two matches, only 4 fight-engage and 5 pregame queries. The five pregame
queries are five views of ONE rune fight (the pregame changes of the second pass were tuned on them: treat
that slice as fitted, not measured); no query is a smoke gank (that rule was checked on the five real smoke
plays of the two replays, outside this set).
