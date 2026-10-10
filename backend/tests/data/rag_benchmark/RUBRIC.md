# Relevance rubric (grades 0-3)

Judge: one reader (Claude), 2026-10-10/11. Unit judged: (decision situation, knowledge chunk).
The situation is everything stored in a query record of `queries.json` (hero, position, phase, point type,
teamfight flag, killer, nearby enemies, all heroes of the game, owned items, causal root cause, gold lead).

| grade | meaning |
|---|---|
| 3 | directly about this exact hero / matchup AND this situation, or the principle chunk written for exactly this situation type |
| 2 | directly about a hero involved in the situation (player, killer, lane opponent) or directly about the situation type, not both |
| 1 | generic but applicable: right role + phase background, an enemy who is in the game but not in this situation, a marginal topical link |
| 0 | not applicable: hero not involved, wrong role or phase, unrelated topic, statistics about another hero |

## How the grade is produced (reproducible: `judge.py`)

1. **Hero / matchup / item chunks** (editorial + learned statistics, about 720 of 832): the grade depends on
   what the chunk is and on the role its hero plays in the situation.
   - roles: `self`, `killer`, `near` (nearby enemy; only when at most two enemies are nearby, i.e. lane opponents),
     `near_bg` (nearby enemy in a 3+ enemy fight), `far` (enemy elsewhere on the map), `ally`, `absent`.
   - hero overview: self / killer / near = 2, near_bg / far = 1, ally / absent = 0.
   - hero laning tips: self = 3 during laning (2 for a good play), 1 in mid game, 0 later; killer / near = 2 during
     laning, 1 in mid game; near_bg = 1 during laning; otherwise 0.
   - hero abilities: self = 2 (1 for missed CS); killer = 3 for a death, else 2; near = 2 (1 for missed CS / good play);
     near_bg = 1; far = 1 from mid game on; otherwise 0.
   - matchup tip "A vs B" (written for A): A = self and B = killer -> 3; B = lane opponent during laning -> 3;
     other nearby B -> 2. Written for the opponent (B = self): killer or lane opponent -> 2, other nearby -> 1.
     self vs an enemy elsewhere -> 1. Anything not involving the player's hero -> 0. Win-rate statistics are
     capped at 2 (own side) / 1 (opponent side).
   - item tip: only for owned items (or 1 for a listed typical buyer when the item class fits the situation).
     Defensive / escape actives = 2 for deaths; fight actives = 2 for teamfight deaths, fight engages and good plays;
     farming items = 2 for missed CS (Battle Fury 3); other owned items 0-1.
   - item counter tip "item vs hero": hero is killer / lane opponent -> 3 if owned else 2; other enemy -> 2 / 1.
   - learned item-timing statistic: own hero -> 2 for missed CS, else 1; other heroes 0.
   - learned hero benchmark: own hero -> 3 for missed CS, else 1; other heroes 0.
2. **Generic chunks** (55 built-in principles + 58 learned): each chunk was read once and given an applicability
   function over the situation (table `GENERIC` in `judge.py`). Examples: `coaching:teamfight_positioning` = 3 for a
   teamfight death or fight engage, 2 for a multi-kill, 1 for a pick-off from mid game on, else 0;
   `vision:safety_rules` = 3 for a gank (pick-off from mid game on, or flagged by the causal chain), 2 for a lane
   death; `phase_objectives:<phase>:<role>` = 2 only when both phase and role match (1 for a good play), else 0.
   The chunk's retrieval tags are not read by these functions.
3. **Manual review**: pooled results of seven queries (one per stratum) were read chunk by chunk; the review led
   to rule changes (lane death vs gank, background enemies in 3+ fights) rather than per-pair exceptions.
   `overrides.json` ({qid: {chunk_id: grade}}) is applied on top and is currently empty.

## Known limits

- One judge, no second annotator; the rubric is strict about entities and lenient about hero background
  (grade 2 for the overview / abilities of every hero in focus), so a query has on average 14 chunks graded >= 2.
- The rubric shares the idea "a chunk about an uninvolved hero is noise" with the ranker's entity gate. That is
  the rule the task prescribes; it means dense retrieval can only win on ordering and on generic chunks.
- Grades are per situation type, not per individual death: two pick-offs of the same hero by the same killer in
  the same phase get the same labels.
