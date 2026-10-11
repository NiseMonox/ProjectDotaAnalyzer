# Analyzer output (raw JSON), schema_version 3

`java -jar tools/clarity-analyzer.jar <replay.dem>` prints one JSON document to stdout
(diagnostics go to stderr; exit code 0, also for a truncated replay with `"partial": true`, or
65 when nothing could be read). `ClarityParser._transform_data` in `backend/app/parser.py`
turns it into `MatchData`; the last section says which fields it uses.

This file describes schema 3 relative to schema 2, which the Python transform still accepts
(cached output, test fixtures): new fields are additive, and the SEMANTIC changes of existing
fields are marked [changed]. Schema 2 never had a document of its own; the fields it already
had and schema 3 left as they were are listed under "Fields unchanged from schema 2" near the
end, so that every field of the output is described in this file.
`"schema_version": 3` tells the versions apart. Positions are Dota 2 world units, times are
seconds on the in-game clock (negative before the horn).

Every example is copied from the output for the two reference replays, matches 8762775364
(7.41b) and 8713267082 (7.40c), except player names: those are made up.

The analyzer is built against clarity 4.0.3 (Java 17). clarity 3.1.3 gives up on some replays
of the current game build ("Entity not found for update", `partial: true` after a few minutes):
2 of 7 replays of 7.41f (build 6956, 2026-10-10) stopped at 309 s of 2036 s and 347 s of 1337 s;
4.0.3 reads all 7 to the end, and its output for the two reference replays was byte-identical
to 3.1.3's before the changes of the next section.

Verification: `python3 tools/compare_with_opendota.py <raw.json> <opendota.json> --ability-ids <ids> --item-ids <ids>`
compares an output with OpenDota's JSON of the same match. It reports 0 DIFF on the two
reference replays (778 ok / 31 notes and 644 ok / 16 notes) and on seven replays of 7.41f: two
league games (9037980041, 9037747302), four ranked games (9038370833, 9038370888, 9038374031,
9038377636) and one Turbo game (9038388762). The notes are listed under "Remaining differences
from OpenDota". `backend/tests/fixtures/clarity_v3_8762775364_trimmed.json`
is the first output cut down and anonymised; `backend/tests/test_parsing_contract.py` checks
the contract on it and compares it with the OpenDota fixture of the same match.

## Changes for 7.41f (analyzer 1.3.0, still schema 3)

Found by comparing the seven current replays with OpenDota; all additive or value fixes.

| field | change |
|---|---|
| `objectives[]` `roshan_kill`, `players[].roshans_killed` | Only `npc_dota_roshan` is Roshan. The expiry of Roshan's Banner (`npc_dota_unit_roshans_banner`, dropped by every second and later Roshan since 7.37) was a third "Roshan kill" in 9037980041. |
| `players[].roshans_killed`, `tower_kills` | Valve's counters (`m_vecDataTeam.N.m_iRoshanKills` / `m_iTowerKills`) at game end; the combat-log count is the fallback. A tower a hero denies counts for him, as in the game (and OpenDota). |
| `radiant_score`, `dire_score` | Valve's team counters (CDOTATeam `m_iHeroKills`) at game end. Fallback: the deaths of `kills_log` that score - not a deny, and a player is credited or a unit of the other side (lane creep, tower) made the kill. A death to neutral creeps counted before (13 vs 12 in 9037747302). |
| `score_source` (new) | `"scoreboard"` or `"kills_log"`. |
| `kills_log[].assist_slots` | The combat-log assist list is ignored for a death that does not score for the other side (neutral creeps made the kill and nobody is credited): the log still lists who had damaged the victim, the game gives no assist. |
| `players[].hero_healing` | From Valve's post-game match record when the replay has it (see `match_record`), else the in-game counter `m_fHealing`, else the combat log. The record is what the post-game screen and OpenDota show; the in-game counter differs for 12 of the 70 players of the current replays (it leaves out the regeneration Urn of Shadows / Spirit Vessel / Essence Distiller give an ally; Terrorblade had 656 there and 0 in the record). |
| `players[].healing_source` (new) | `"match_record"`, `"scoreboard"` or `"combat_log"`. |
| `players[].gold_per_min`, `xp_per_min` | From the match record when present (Valve's own division; XPM of a level-30 hero in Turbo was 3168 from the counter, 3195 in the record: the earned-XP counter stops at the level cap). Fallback as before. |
| `match_record` (new, top level) | bool: the replay carries Valve's post-game record (CMsgDOTAMatch in the match metadata at the very end of a complete replay). false for a truncated replay. |
| `players[].gold_t`, `xp_t`, `lh_t`, `dn_t`, `radiant_gold_adv`, `radiant_xp_adv` | Sampled exactly where OpenDota samples: at the start of the first tick whose frame clock - `round((server tick of the previous frame - paused ticks) / 30)` - has reached `round(horn) + 60 * minute`. Minute 0 is therefore taken up to a second before the horn. All 9960 values of the nine replays equal OpenDota's (289 differed before, up to 931 XP when a kill fell on the sampling frame). |
| `players[].ability_upgrades` | When several abilities rise for one skill point, the one the player ordered to train (`DOTA_UNIT_ORDER_TRAIN_ABILITY`) is the pick. Before, slot order chose `ember_spirit_activate_fire_remnant`, `keeper_of_the_light_radiant_bind` and `techies_minefield_sign` over the ultimates they level with. |
| `teamfights[]` | The 15 s chain rule is measured on OpenDota's rounded clock (`round(t) - round(horn)`): deaths at 347.5 s and 362.3 s chain (9038377636). Window bounds and offsets are unchanged. |
| `parse_errors` | Two diagnostic keys that are no handler failures: `score_credits` = scoreboard kill / assist increments that belong to no death of `kills_log` (1 assist in 9038374031), `skill_points` = skill points seen spent that no ability level-up explained (never seen). |

## Changes of analyzer 1.4.0 (still schema 3)

| field | change |
|---|---|
| `analyzer_version` (new, top level) | Version of the analyzer that wrote the output (`version` of build.gradle, read from the JAR manifest's `Implementation-Version`); `""` when the classes do not run from the JAR. Absent in the output of older JARs, so a stale JAR or a cached output can be told from a current one. `java -jar clarity-analyzer.jar` without arguments prints it too. |
| `kills_log[].killer_team` (new) | Side of the unit that landed the kill, see the table below. Needed for a unit whose name names no side: a dire hero that dies to the radiant fountain (`killer_unit: "dota_fountain"`, 9038370833 at 23:53 and 9038370888) scores for Radiant. With it the kill log gives Valve's score on all nine replays: per death, the side of the credited hero, else `killer_team` (the compare tool checks this against OpenDota's `radiant_score` / `dire_score`). |
| `players[].ability_upgrades` | A level the player ordered to train while no skill point was spent is an entry: Winter Wyvern's Eldwurm's Edda (7.41 innate: an item she can consume after 10 minutes to raise the level and the maximum level of a basic ability by one). 9037747302: train order for Arctic Burn at 11:34 (694 s), level 3 -> 4, no decrease of `m_iAbilityPoints`; 9038377636: level 2 -> 3 at 10:16. The entry was missing and the next level of that ability was numbered one too high (Arctic Burn 1, 2, 3, 5). Only when the replay has `m_iAbilityPoints`, for one level per order. |
| stderr summary | `diagnostics: {score_credits=1}` is printed apart from `handler errors: {...}`. |

## Changes of analyzer 1.5.0 (still schema 3)

All additive; the output of 1.4.0 for the reference replays is otherwise unchanged.

| field | change |
|---|---|
| `wards[].denied`, `wards[].killer_team` (new), `wards[].killer_slot` | Who ended a ward, from the attacker of its combat-log death, see `wards[]`. Before, the output only told an enemy (`dewarded: true`) from everything else: a ward its own side attacked looked like one that ran out early. In the log a ward that runs out is its own attacker (`npc_dota_sentry_wards` kills `npc_dota_sentry_wards`, damage source = the owner), a deny has a hero of the ward's side as attacker. `killer_slot` is now also set for a deny. Six replays: 5 denied wards (8713267082 at 16:42: Warlock attacks his own sentry 18 s after placing it; 9038374031 at 7:25: Terrorblade denies Jakiro's sentry), 183 expired, 115 dewarded (9 of them by lane creeps: no `killer_slot`); every one agrees with the attacker OpenDota's left log names (the compare tool checks it). |
| `kills_log[].respawn_time` (new) | Game time the victim was alive again, see `kills_log[]`. Deaths and buybacks had times, a natural respawn had none: a consumer could only read it off `position_log`, 5 s apart. 381 deaths of six replays: 360 with a time (12 of them buybacks), 21 null (dead when the game ended). Of the 278 natural respawns of the five standard games 269 took the base respawn time of the hero's level (`backend/app/game_constants.py`, within the second of the sampling); the other 9 were longer or shorter by an effect, and the Turbo game has times of its own. |
| `chat[].hero_id` (new) | For a chat-wheel message that names a hero (`chatwheel_71` "<hero> is missing!", `chatwheel_72`): the hero's id, `param_hero_id` of the game's own message. Absent otherwise. All 29 `chatwheel_71` lines of the six replays carry one. |

## Removed since schema 2

- Top level `tower_status_radiant` / `tower_status_dire` / `barracks_status_radiant` /
  `barracks_status_dire`: were always 0. Python derives the bitmasks from `buildings` +
  `objectives`.
- `players[].wards_placed`: was always `[]`. The wards are in `wards[]`.

## Top level

| field | type | meaning | real example |
|---|---|---|---|
| `schema_version` | int | 3 | `3` |
| `analyzer_version` (new in 1.4.0) | str | Version of the analyzer JAR that wrote the output; `""` outside the JAR. Absent before 1.4.0. | `"1.5.0"` |
| `match_id` [changed] | int (64-bit) | The full id. From CDemoFileInfo whenever the replay has it, else from the game-rules entity (see `match_id_source`). Schema 2 only had the low 32 bits. | `8762775364` (was 172840772) |
| `match_id_source` (new) | str | `"file_info"`; `"game_rules"` = no file info (truncated replay), the full id from the game-rules entity; `"game_rules_32bit"` = the same but decoded as 32 bits (Clarity has no decoder for the field type `MatchID_t`; the analyzer registers a 64-bit one before the run, and this value means that registration failed): the REAL ID MODULO 2^32, which Python turns into match_id 0; `"none"` = match_id 0. | `"file_info"`; truncated: `8762775364` / `"game_rules"` |
| `radiant_win` [changed] | bool or null | `null` when the winner is unknown (truncated replay). Was `false`. | `false`; truncated: `null` |
| `game_started` (new) | bool | `false` when the replay ends before the horn. Then `duration` is 0 and every event time is <= 0 (relative to the scheduled horn = pre-game start + 90 s, else to the end of the data). | `true` |
| `duration` [changed] | int | 0 when `game_started` is false (was the raw clock). Otherwise unchanged. | `2091` |
| `start_time` [changed] | int (unix s) | `end_time - (playback_ticks + server_start_tick) / tick_rate` = when the server started. Equals OpenDota's start_time exactly on both replays (was 409-429 s late). 0 without file info. | `1775643322` |
| `cluster` (new) | int | From the demo header server name `(srcds1006-sgp2.151.19)`; 0 when unknown. | `151` |
| `region` [changed] | int | OpenDota region id mapped from `cluster` (table from OpenDota constants/cluster embedded in the jar); 0 when unknown. Was always 0. | `5` |
| `game_build` (new) | int | Dota build from the header game directory (`dota_v6762`); 0 when unknown. | `6762` |
| `patch` | str | Always `""`: the replay does not name its game version. | `""` |
| `first_blood_time` [changed] | int | Time of the game's own CHAT_MESSAGE_FIRSTBLOOD event, clamped to >= 0; falls back to the first death with an enemy killer. | `137`; `0` on 8713267082 (first blood at -10 s) |
| `kills_log` (new) | list | See below. | 70 entries on both replays |
| `picks_bans` [changed] | list | See "Draft". | |
| `draft_source` (new) | str | `"file_info"` (CDemoFileInfo picks_bans: teams and order are Valve's) or `"game_rules"` (every all-pick replay: picks from the player resource, bans from the game rules). | `"game_rules"` |
| `radiant_gold_adv`, `radiant_xp_adv` [changed] | list[int] | Stop at game end: `duration // 60 + 1` entries. | 35 entries for 2091 s (was 38) |
| `parse_errors` | map | Unchanged shape (handler -> count). New key `entity_names` = number of failed lookups when the EntityNames table never resolved a single name (the name-index property was renamed again; also a `[warn]` line on stderr). | `{}` |

### `kills_log[]` - every REAL hero death, time-ordered
No illusions, no reincarnation / aegis deaths. Denies are included with `deny: true`.

| field | type | meaning |
|---|---|---|
| `time` | float (1 decimal) | game time of the death (negative before the horn) |
| `victim_slot` | int | player_slot (0-4 / 128-132) |
| `killer_slot` | int or null | player_slot credited with the kill: the scoreboard's kill counter decides (so a kill credited without a hero last hit is attributed), else the hero that landed it, or the owner of the summon that did. null = no hero (creep, tower, neutral, Roshan). For a deny: the denying hero. |
| `killer_unit` | str | full npc name of the unit that landed the kill; `""` when unknown |
| `killer_team` (new in 1.4.0) | str or null | side of that unit (the attacker team of the combat-log row): `"radiant"`, `"dire"`, `"neutral"` (camp creeps, Roshan, Tormentor); null when the row names none. For a deny it is the victim's own side. It can be `"neutral"` while `killer_slot` is a hero: the scoreboard credits the hero, a neutral creep landed the kill. |
| `x`, `y` | float or null | victim position, world units |
| `assist_slots` | list[int] | sorted player_slots: union of the combat-log assist list (only for a death that scores for the other side) and the players whose scoreboard assist counter went up in the tick of this death; never contains the killer |
| `deny` | bool | victim was killed by its own team |
| `respawn_time` (new in 1.5.0) | float (1 decimal) or null | game time the victim was alive again: the time of its buyback when it bought back (exact), else the first of the hero's one-per-second samples that shows health after one that showed it dead, so a respawn is up to a second late. null = not seen alive again (the game or the recording ended first). Around a pause the sample clock can run up to 3 s ahead of the combat log's; a sample that still carries the last health after the death is therefore no respawn (8762775364 at 28:25). |

Examples:
`{"time": 141.6, "victim_slot": 132, "killer_slot": 3, "killer_unit": "npc_dota_hero_weaver", "killer_team": "radiant", "x": 6496.3438, "y": -3305.1562, "assist_slots": [2], "deny": false, "respawn_time": 157.0}`
`{"time": 805.9, "victim_slot": 129, "killer_slot": 2, "killer_unit": "npc_dota_warlock_golem", "killer_team": "radiant", "x": 5478.375, "y": -4190.8438, "assist_slots": [0, 3], "deny": false, "respawn_time": 835.0}`
`{"time": 1433.8, "victim_slot": 129, "killer_slot": null, "killer_unit": "dota_fountain", "killer_team": "radiant", "x": -7051.0312, "y": -6588.875, "assist_slots": [], "deny": false, "respawn_time": null}` (9038370833; the game ends at 24:12)
`{"time": 1687.2, "victim_slot": 4, ..., "deny": false, "respawn_time": 1690.1}` (8713267082: a buyback 2.9 s after the death)

Invariants, all EXACT for the 20 players of the two reference replays:
entries with `victim_slot == slot` == `deaths`; entries with `killer_slot == slot and not deny` ==
`kills`; entries with `slot in assist_slots and not deny` == `assists`. Every scoreboard kill /
assist increment was found within one tick of its death. On the seven 7.41f replays they hold
for 69 of 70 players: one assist of the scoreboard (Pudge, 9038374031) came in a tick in which
only two of his own team died; it belongs to no death and is reported as
`parse_errors.score_credits`.

## players[]

| field | type | meaning | real example |
|---|---|---|---|
| `hero_id` [changed] | int | Real hero id from CDOTA_PlayerResource `m_nSelectedHeroID` (was always 0). | `74` |
| `player_name` [changed] | str | `""` when unknown (was the English placeholder `"Player N"`); only happens without file info. | `"SomePlayer"` (made up); truncated: `""` |
| `hero_name` | str | Unchanged, incl. the `"unknown_<i>"` code for a hero that never resolved. | `"invoker"` |
| `kills`, `deaths`, `assists` [changed] | int | Valve scoreboard (CDOTA_PlayerResource) at game end; combat-log count only as fallback. | `7 / 10 / 23` (Zeus, was 7/10/18) |
| `kda_source` (new) | str | `"scoreboard"` or `"combat_log"`. | `"scoreboard"` |
| `hero_damage`, `tower_damage`, `hero_healing` [changed] | int | Valve scoreboard (CDOTA_DataRadiant/Dire `m_flHeroDamage`, `m_flTowerDamage`, `m_fHealing`) at game end. The combat-log fallback now uses the same rules (summons count for their hero, no self damage, no self heals) and reproduces the scoreboard for 59 of 60 values. `hero_healing` prefers the post-game match record, see "Changes for 7.41f". | `24895 / 12216 / 0` (Wraith King, was 19134 / 6184 / 13199) |
| `damage_source` (new) | str | `"scoreboard"` or `"combat_log"` (hero and tower damage; `healing_source` for the healing). | `"scoreboard"` |
| `net_worth`, `total_gold`, `last_hits`, `denies`, `level` [changed] | int | Captured at GAME END (state 6), not at the end of the recording. | Lion `net_worth 9368` (was 9610) |
| `gold_per_min`, `xp_per_min` [changed] | int | The post-game match record when the replay has it, else `floor(total / unrounded game minutes)` (equal to OpenDota for 20/20 on the reference replays either way). | `514`, `680` |
| `total_xp` (new) | int | Total earned XP at game end. | `23738` |
| `stuns` (new) | float or null | Seconds of stun dealt, Valve scoreboard (`m_vecDataTeam.N.m_fStuns`) at game end; null when the property is missing. | `68.03287` |
| `teamfight_participation` (new) | float or null | Valve scoreboard (`m_vecPlayerTeamData.N.m_flTeamFightParticipation`), 0-1; null when missing (Python then uses (kills + assists) / team kills). | `0.5` |
| `actions_per_min` (new) | int | Unit orders of the whole recording (`CDOTAUserMsg_SpectatorPlayerUnitOrders`, counted per player controller) * 60 / duration: OpenDota's definition. 0 when the game never started. | `325` |
| `multi_kills` (new) | map[str,int] | The game's multi-kill announcements (`DOTA_COMBATLOG_MULTIKILL`): value -> times; a triple kill also counts under `"2"`. | `{"2": 1}` |
| `kill_streaks` (new) | map[str,int] | The game's kill-streak announcements (`DOTA_COMBATLOG_KILLSTREAK`): streak length (3+) -> times; a streak of 5 counts under `"3"`, `"4"`, `"5"`. | `{"3": 3, "4": 2, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1}` (Void Spirit) |
| `gold_t`, `xp_t`, `lh_t` [changed] | list[int] | `duration // 60 + 1` entries (minute 0 .. last full minute). Sampled at the tick OpenDota samples them (see "Changes for 7.41f"; up to 1 s ahead of the real minute): equal to OpenDota's series in every value. | 35 entries |
| `dn_t` (new) | list[int] | Denies per minute, same sampling and length as `lh_t`. Equal to OpenDota's `dn_t` in every value. | `[0, 0, 1, 5, 6, 9, ...]` (35 entries) |
| `position_log` (new) | list[[int,int,int,int]] | `[time_s, x, y, hp_pct]`, one sample every 5 s from the first spawn (negative times) to game end; times are multiples of 5. Each sample is the state at that game second (first tick of it). World-unit ints. `hp_pct` 0-100, 0 = dead, -1 = unknown (hero entity gone). A dead hero keeps the position where it died. Checked against kills_log: after 139 of 140 deaths the next sample has hp_pct 0 (the other one bought back within the second). | `[[-85, -6597, -5806, 100], [-80, -5261, -5292, 100], ...]` (436 entries) |
| `ability_upgrades` [changed] | list | Same shape `{name, level, time}`. Equal to OpenDota's `ability_upgrades_arr` for 20/20 players (Invoker has 2 more: Valve caps the list at 25) and for 69 of the 70 players of the 7.41f replays; for the other one (Vengeful Spirit, 9038388762) Valve's list has her level-10 talent twice. Talents are real `special_bonus_*` keys with level 1; `special_bonus_attributes` only when a point was spent on it; innates never. | `{"name": "special_bonus_unique_hoodwink_bushwhack_damage", "level": 1, "time": 1101}` |
| `purchase_log` [changed] | list | (a) no `ward_dispenser` rows (stack merges, not purchases); (b) starts with the strategy-time purchases, stamped with the pre-game start: one row per purchase (a ward stack gives one row per ward, 3 mangoes 3 rows, a 3-charge tango 1 row). | `{"key": "ward_observer", "time": -90}` |
| `starting_items` | list | UNCHANGED: inventory snapshot at the horn (may contain `ward_dispenser`). Python only uses it for schema-2 output: in schema 3 `purchase_log` is complete, and merging the snapshot would bring an item assembled before the horn back as a fake starting item. | |
| `buyback_log` (new) | list[{time:int}] | One entry per buyback; `buyback_count` == its length. | `[{"time": 1831}]` |
| `damage_inflictor` [changed] | map | Key of a plain attack is `"attack"` (was `"dota_unknown"`). | `{"attack": 4846, "invoker_cold_snap": 2044, ...}` |
| `laning_damage_events[].inflictor` [changed] | str | `"attack"` for a plain attack; always present now. | `{"time": 34, "target_idx": 8, "damage": 52, "inflictor": "attack"}` |
| `damage`, `damage_taken`, `laning_damage_events` [changed values] | | Same key format (hero slug). The dealer is the damage SOURCE: summon damage is credited to the hero; self damage and damage to allies are not counted. | |
| `killed`, `killed_by` [changed values] | map | UNCHANGED key format: hero slug for heroes (`"skeleton_king"`), npc name with `npc_dota_` stripped for other units (`"creep_badguys_melee"`). Python maps the hero slugs to display names (`MatchData` keeps heroes only there). Built from the credited kill log: sums equal kills / deaths. | `{"skeleton_king": 5, "zuus": 1, "void_spirit": 5, "lion": 2, "tidehunter": 1}` |
| `gold_reasons`, `xp_reasons` [changed values] | map | The rows of the first ticks are no longer lost: reason `"0"` (600 starting gold) is present for all 10 players. | |
| `final_items`, `backpack`, `neutral_item`, `starting_items` [changed values] | | Real item keys from the EntityNames table on every build (`travel_boots`, `dust`, `invis_sword`, `bottle`); the class-name fallback knows the keys that are not plain snake case (21 from the reference replays, 14 more from the 7.41f replays: it now gives the right key for all 283 item classes seen in the nine replays). | |
| `ancient_kills` [changed] | int | Counts the ancient frog camp: equal to OpenDota 20/20. | |
| `laning_creep_kills[].creep_type` | str | Flagbearer creeps are `"melee"` (was `"unknown"`). No new code. | |

## wards[]

| field | type | meaning |
|---|---|---|
| `player_slot` [changed] | int | Owner from the ward entity's own `m_nPlayerOwnerID` (161/161 equal to OpenDota); -1 only when nothing resolved it. |
| `owner_source` | str | `"entity"`, `"purchase"` (last-resort heuristic) or `"unknown"`. All 161 are `"entity"`. |
| `destroyed_time` [changed] | int or null | Game time the ward DIED (life-state change): now -1..0 s from OpenDota's left-log time (was +4..+9). null = alive at game end. |
| `dewarded` (new) | bool | true = destroyed by the enemy team; false = denied, expired, or still alive at game end. The key is ABSENT when the ward died but no combat-log death was matched (did not happen on the two replays: 137/137 matched) - apply the lifetime fallback in Python then. |
| `denied` (new in 1.5.0) | bool | true = destroyed by a unit of its own side that is not the ward itself (an ally's or the owner's attack). false for a ward that expired: it is its own attacker in the combat log. Present exactly when `dewarded` is, for a ward that died; never true together with it. |
| `killer_team` (new in 1.5.0) | str | Side of the unit that ended the ward (the attacker team of the combat-log row): `"radiant"`, `"dire"` or `"neutral"`; the ward's own side when it expired or was denied. Only for a ward that died with a matched combat-log death; absent when the row names no side. |
| `killer_slot` (new) | int | Only when `dewarded` or (since 1.5.0) `denied` is true and a hero (or its summon) did it: that hero's player_slot. An enemy for `dewarded`, a hero of the ward's side for `denied`. Never for an expiry. |

Examples:
`{"type": "sentry", "x": -4530.75, "y": 3675.0, "time": -39, "player_slot": 128, "team": "dire", "destroyed_time": 91, "ehandle": 2081703, "owner_source": "entity", "dewarded": true, "denied": false, "killer_team": "radiant", "killer_slot": 4}`
`{"type": "observer", "x": 541.78125, "y": -161.28125, "time": -56, "player_slot": 131, "team": "dire", "destroyed_time": 303, "ehandle": 6521761, "owner_source": "entity", "dewarded": false, "denied": false, "killer_team": "dire"}` (expired)
`{"type": "sentry", "x": -1532.9375, "y": -3796.4062, "time": 1002, "player_slot": 2, "team": "radiant", "destroyed_time": 1020, "ehandle": 1049567, "owner_source": "entity", "dewarded": false, "denied": true, "killer_team": "radiant", "killer_slot": 2}` (8713267082: denied by its owner)
`{"type": "sentry", ..., "time": 1346, "team": "radiant", "destroyed_time": 1743, ..., "dewarded": true, "denied": false, "killer_team": "dire"}` (8713267082: a lane creep, no hero)
`{"type": "sentry", ..., "time": 1686, "destroyed_time": null, ..., "dewarded": false}`

## teamfights[]

| field | type | meaning |
|---|---|---|
| `start`, `end` [changed] | int | PADDED window like OpenDota: first death of the chain - 15 s .. last death + 15 s (end capped at game end). Was first death .. last death. Every `time_offset` / `death_time_offset` is relative to the new `start`. The window covers the whole seconds start..end. |
| `first_death`, `last_death` (new) | int | The unpadded bounds: first / last REAL death (game time). |
| `gold_delta` [changed] | int | Sum of the Radiant players' `gold_delta` minus the Dire players' (positive = Radiant advantage). Was 0. |
| `xp_delta` (new) | int | Same for XP. |
| `players[].gold_delta`, `players[].xp_delta` [changed] | int | Sum of the player's combat-log gold / XP rows in the window (OpenDota's definition; gold lost on death counts negative; no passive income). Was 0. |
| `players[].buybacks` (new) | int | Buybacks inside the window. |
| `players[].item_uses` (new) | map[str,int] | Item key (no `item_` prefix) -> uses inside the window. |
| `players[].kill_targets` (new) | list | `[{"target_slot": <victim player_slot>, "time_offset": <float>}]`, one entry per kill credited to this player (denies excluded). Python adds `"target"` (display name). `players[].kills` == its length. |
| `players[].death_time_offset` [changed] | float | Offset of the player's FIRST death in the fight (was the last). |
| `players[].healing` | int | Healing done to real heroes in the window, self heals included (as before). |
| `players[].damage_dealt`, `damage_taken` [changed values] | int | Dealer = damage source; no self / ally damage; padded window. |
| `players[].positions`, `ability_casts` [changed values] | | Cover the padded window (about 17-25 positions per player). |

Fight detection: deaths chain while the gap is under 15 whole seconds of OpenDota's rounded
clock (was "at most 15.0 s", then whole seconds of the in-game clock); a
reincarnation death keeps the chain alive but is not counted as a death; a fight needs at least 3
real deaths. Result: the same fights as OpenDota on all nine replays (windows within 1 s) plus,
on five of them, one at the very end that OpenDota never closes.

Example: `{"start": 803, "end": 837, "first_death": 818, "last_death": 822, "radiant_deaths": 2, "dire_deaths": 2, "gold_delta": -599, "xp_delta": 404, "players": [{"player_slot": 0, "hero_name": "invoker", "kills": 1, "deaths": 0, "damage_dealt": 1219, "damage_taken": 260, "healing": 0, "gold_delta": 570, "xp_delta": 648, "buybacks": 0, "item_uses": {"hand_of_midas": 1}, "kill_targets": [{"target_slot": 129, "time_offset": 17.1}], ...}, ...]}`

## objectives[]

| field | type | meaning |
|---|---|---|
| building kills: `building` (new) | str | Canonical name of the building ENTITY that died, with the side for tier-4 towers. `key` is unchanged (combat-log unit name; `"goodguys_tower4"` has no side). Absent when no entity matched. |
| `CHAT_MESSAGE_FIRSTBLOOD`: `victim_slot` (new) | int | player_slot of the victim. `player_slot` (killer) and `key` (victim hero slug) are unchanged; in `MatchData` the key is `str(victim_slot)`, mapped in Python. |
| `CHAT_MESSAGE_COURIER_LOST` (new type) | | `team` = "radiant"/"dire", the side that LOST the courier; `slot` / `player_slot` = the killing hero (absent when no hero did it); `value` = the gold bounty. |

Examples:
`{"type": "tower_kill", "key": "goodguys_tower4", "team": "dire", "building_team": "radiant", "denied": false, "player_slot": 129, "building": "npc_dota_goodguys_tower4_top", "time": 2036}`
`{"time": 137, "type": "CHAT_MESSAGE_FIRSTBLOOD", "slot": 4, "player_slot": 4, "key": "lion", "victim_slot": 128}`
`{"time": 647, "type": "CHAT_MESSAGE_COURIER_LOST", "slot": 4, "player_slot": 4, "team": "dire", "value": 45}`

## buildings[]
- `name` [changed values]: all 36 are canonical on both builds. Barracks: `npc_dota_goodguys_melee_rax_mid` (the map entity is called `good_rax_melee_mid`). Tier 4: `npc_dota_goodguys_tower4_top` / `_bot`. Names were all `""` on the current build.
- `destroyed_time` [changed]: the building's own death (entity life state) decides which building died; the combat-log time is attached to it at most once, so one tier-4 death can never mark both tier-4 towers.
- Example: `{"name": "npc_dota_goodguys_melee_rax_mid", "team": "radiant", "type": "barracks", "x": -4672.0, "y": -4552.0, "destroyed_time": 1665}`

## Draft: picks_bans[]
- Picks: `{"is_pick": true, "hero_id": 26, "hero_name": "lion", "team": 1, "order": 2, "player_slot": 128}`; orders 0-9 in lock-in order (falls back to the game's pick order, then to slot order).
- Bans from the game rules (all pick): `{"is_pick": false, "hero_id": 38, "team": null, "order": 10}`, orders after the picks. `team` is null: the replay does not say who nominated a ban. The list also contains the bans the game added itself (16 in total; Valve's API / OpenDota list only the 9 and 6 player-nominated ones, which come first and in the same order).
- When CDemoFileInfo has the draft (captains mode), it is used unchanged (`team` 0/1, Valve's order) and `draft_source` is `"file_info"`.

## smokes[]
- `player_slot` [changed]: the hero that USED the smoke (was the first hero that received the modifier). `player_slots` unchanged.
- `objective_taken` can now be `"tower"` / `"barracks"` / `"ancient"` on current builds (building names resolve; 3 smokes on the two replays).

## Fields unchanged from schema 2

Fields that have no row in the sections above: they exist since schema 2 with the meaning given
here. Examples are from 8762775364 unless noted.

Top level:

| field | type | meaning |
|---|---|---|
| `coord_space` | str | Always `"world"`: every `x` / `y` in the output is in Dota 2 world units. The Python transform converts positions only for output without it (the pre-v2 grid). |
| `partial` | bool | The replay could not be read to its end (cut off or damaged); see "Truncated replays". `false` for a complete replay. |
| `game_mode` | int | Valve's game-mode id (`22` ranked all pick, `23` Turbo). Python maps it to a code (`parser.GAME_MODES`). |
| `players`, `wards`, `smokes`, `teamfights`, `objectives`, `buildings`, `chat` | list | The sections described in this file; `players` always has 10 entries, Radiant first. |

`players[]`:

| field | type | meaning |
|---|---|---|
| `player_slot` | int | 0-4 Radiant, 128-132 Dire. |
| `hero_npc_name` | str | Full unit name of the hero (`"npc_dota_hero_invoker"`); `hero_name` is the same without the prefix. `""` when the hero never resolved. |
| `steam_id` | int | 64-bit Steam id from the file info; 0 without it. Not mapped by Python. |
| `ability_casts` | list | `{name, time}`: every ability the hero used, from the combat log, in time order (1577 entries for Invoker). |
| `lane_kills` | int | Lane creeps (melee, ranged, flagbearer, siege) the hero last-hit; denies are not counted. |
| `neutral_kills` | int | Neutral creeps the hero last-hit, ancients included (`ancient_kills` is the part that were ancients). |
| `courier_kills` | int | Couriers the hero killed. |
| `observer_kills`, `sentry_kills` | int | Enemy wards of that kind the hero destroyed. |
| `runes_log` | list | `{time, key}` per rune the player picked up; `key` is Valve's rune id as a string: `"0"` double damage, `"1"` haste, `"2"` illusion, `"3"` invisibility, `"4"` regeneration, `"5"` bounty, `"6"` arcane, `"7"` water, `"8"` XP (a wisdom rune until 7.37, a Shrine of Wisdom capture since 7.38), `"9"` shield. |
| `runes` | map[str,int] | The same pickups counted per rune id: `{"3": 1, "0": 1, "5": 2, "8": 2}`. |
| `lane`, `lane_role` | int | Always `0`: the analyzer does not judge lanes. Python derives both from `laning_snapshots`. |
| `laning_snapshots` | list | One sample every 5 s from 0:00 to 10:00 (121 entries for a game that lasts that long): `{time, x, y, hp, max_hp, mana, max_mana, level, last_hits, denies, net_worth}`, the state of the hero at that game second. |
| `laning_creep_kills` | list | `{time, creep_type, gold}` per lane or neutral creep the hero last-hit in the first 10 minutes; `creep_type` is `"melee"`, `"ranged"`, `"siege"`, `"neutral"` or `"unknown"`, `gold` the bounty of the combat-log row. |
| `laning_denies` | list | The same shape for lane creeps of the hero's own side that it denied in the first 10 minutes. |
| `laning_damage_events` | list | `{time, target_idx, damage, inflictor}` per instance of damage to an enemy hero in the first 10 minutes; `target_idx` is the INDEX of the target in `players` (0-9), not its player_slot. See `players[]` above for the rules of who counts as dealer. |
| `laning_regen_events` | list | `{time, item}` per use of a regeneration item in the first 10 minutes (item keys seen: `tango`, `tango_single`, `flask`, `clarity`, `enchanted_mango`, `famango`, `faerie_fire`, `bottle`). |
| `buyback_count` | int | Number of buybacks (the length of `buyback_log`). |

`wards[]`: `type` (`"observer"` / `"sentry"`), `x`, `y`, `time` (when it was placed; negative
before the horn), `team` (`"radiant"` / `"dire"`, the owner's side) and `ehandle` (the entity
handle, a diagnostic) are unchanged.

`smokes[]` (one entry per Smoke of Deceit used):

| field | type | meaning |
|---|---|---|
| `time`, `end_time` | int | When the smoke was used, and when the first member of the group lost it (revealed, or it ran out); 45 s after `time`, capped at the end of the data, when no end was seen. |
| `team` | str | `"radiant"` / `"dire"`. |
| `x`, `y` | float | Where it was used (position of the user). |
| `path` | list[[x, y]] | The user's positions from `time` to `end_time` (one about every 2 s) and the position at `end_time` as the last point. |
| `kills_gained`, `deaths_suffered` | int | Hero deaths from 5 s before to 30 s after `end_time` near the path or a member of the group: enemies that died (denies excluded) and heroes of the smoking side that died. |
| `objective_taken` | str or null | `"tower"`, `"barracks"`, `"ancient"` or `"roshan"` when the smoking side took one near the path from 5 s before to 60 s after `end_time`; else null. |
| `result` | str | `"traded"` (kills on both sides), `"kill"`, `"objective"` (no kill, an objective), `"caught"` (only own deaths) or `"no_contact"`, decided in that order. |
| `player_slots` | list[int] | player_slots of every hero that was under this smoke. |

`teamfights[]`: `radiant_deaths` / `dire_deaths` (int, real deaths of each side in the fight);
per player `player_slot`, `hero_name`, `x` / `y` (float, the hero's first position in the
window), `kills` and `deaths` (int, inside the window), `positions` (`{x, y, time_offset}`),
`ability_casts` (`{name, key, time_offset, x, y}`: `key` repeats `name`, `x` / `y` are where the
caster stood) and `abilities_used` (list[str], the distinct `name`s of `ability_casts` in order
of first use).

`objectives[]`: every entry has `time` and `type`. The types are the game's own chat events
(`CHAT_MESSAGE_FIRSTBLOOD`, `CHAT_MESSAGE_AEGIS`, `CHAT_MESSAGE_COURIER_LOST`; `slot` is the
acting player's index in `players` (0-9), `player_slot` his player_slot) and the kills taken
from the combat log: `tower_kill`, `barracks_kill`, `ancient_kill` (`key`, `team` = the side that destroyed it,
`building_team`, `denied` and, when a hero did it, `player_slot`) and `roshan_kill` (`team`,
`player_slot`).

`buildings[]`: `team`, `type` (`"tower"`, `"barracks"`, `"fort"`), `x`, `y` are unchanged; all 36
buildings of the map are listed, `destroyed_time` null for one that stood at the end.

`chat[]`: `type` is `"chat"` (typed in all chat) or `"chatwheel"`; `key` is the typed text, or
`chatwheel_<id>` with the game's phrase id; `time`; `player_slot` (-1 for somebody who is not one
of the ten players).

## Truncated replays (exit code 0 with `"partial": true`, or 65)
`radiant_win: null`, `match_id_source: "game_rules"` (the full id; checked on cuts of both
reference replays at 2 / 5 / 30 / 40 MB), `start_time: 0`,
`player_name: ""`, `steam_id: 0`; hero ids, pick order, scoreboard values, kills_log and
position_log are still filled from the entities (state at the cut). A cut before the horn: `game_started: false`,
`duration: 0`, times <= 0 on the real pre-game clock (ward at -56 s, same as in the full replay).

## Remaining differences from OpenDota (all legitimate; `note` lines of the compare tool)
- Skill build: Invoker has 27 entries, OpenDota 25 (Valve's cap). One player of the 7.41f
  replays (Vengeful Spirit, 9038388762) has her level-10 talent twice in Valve's list, before
  and after the same Magic Missile level; a talent has one level, and the replay shows it
  learned once. The compare tool accepts only such a repeated talent as a note; any other
  entry the analyzer lacks is a DIFF.
- Draft: a pick that was taken back (repick) stays a pick in Valve's list although nobody played
  the hero (the replay has it among the unavailable heroes), and a hero assigned when the pick
  time ran out has no pick entry there. `picks_bans` picks are the ten heroes played.
- kills_log: a kill the scoreboard credits to a player although a tower or creep landed it is
  not in OpenDota's per-player log (2 in 9038377636).
- Teamfight gold / XP deltas: OpenDota takes differences of its 1 s samples, the analyzer sums
  the combat-log rows of the window; a hero kill within a second of a window edge moves a whole
  bounty (588 and 2844 XP in two fights of the current replays).
- purchase_log: a stack bought in strategy time (two tangos, two mangoes) is ONE OpenDota row
  with `charges`; the analyzer has one row per purchase.
- A ward dewarded in the last second of the game has no OpenDota left-log entry (9038374031).
- `observer_kills`: like `sentry_kills`, OpenDota also counts a deny of the team's own ward.
- Chat of somebody who is not one of the ten players (a league admin): `player_slot` -1 here,
  absent in OpenDota. `region`: 0 here, null there, for a cluster the table does not have.
- kills_log: OpenDota lists 5 Wraith King reincarnation "kills" (8762775364); they are not deaths.
- Teamfights: OpenDota counts those reincarnations as deaths; window bounds differ by up to 1 s
  (OpenDota's clock is `round(raw) - round(horn)`, ours `trunc(raw - horn)`, the in-game clock);
  one extra fight at game end; fight damage excludes self damage.
- Event times in general (wards, objectives, chat): -1..+1 s for the same reason.
- purchase_log: strategy-time ward stacks are listed (OpenDota has no row for a merged
  observer/sentry stack); OpenDota has NO strategy-time purchases at all for 8762775364 (its parser
  lost them on the current build), the analyzer has them.
- Draft: ban teams are null; 7 and 10 extra system bans.
- `lane_kills`: definition (OpenDota counts denied lane creeps and skips siege creeps).
- `sentry_kills`: OpenDota counts one kill of the team's own sentry (Warlock, 8713267082).
- `stuns`: OpenDota's last sample is up to a second before game end (1 of 20 players differs, by 0.4 s).
- hero_damage from the combat log alone would be 282 too high for Invoker; irrelevant because the
  scoreboard value is used.

## How the Python transform (ClarityParser._transform_data) uses the raw fields
Every field above is mapped into MatchData / PlayerData, with these deliberate exceptions:
- diagnostics, not mapped: `game_started` (a pre-game-only replay is `partial` with `duration` 0),
  `match_id_source`, `draft_source`, `parse_errors`, `kda_source`, `damage_source` (these four are
  logged; of `parse_errors` the keys `score_credits` and `skill_points` as "diagnostics", the rest
  as "handler errors"), `analyzer_version` (logged at debug level),
  `score_source`, `match_record`, `healing_source`, `net_worth_source`,
  `final_items_source` (decides the purchase fallback),
  `ability_upgrades_source`, `wards[].ehandle`, `wards[].owner_source`, `objectives[].building_team`
  (derived from the building name), `teamfights[].players[].hero_name` (the slot decides).
- not mapped: `players[].steam_id` (not needed, privacy), `total_gold` / `total_xp` (gold_t / xp_t and
  the per-minute rates are), `starting_items` for schema 3 (see players[]), `patch` (always "": the
  backend derives the version from `start_time`, see `data_manager.patch_for_time`).
- diagnostics, not mapped (1.5.0): `wards[].killer_team` (`dewarded` / `denied` say what it means for
  the ward; the compare tool reads it).
- model fields that exist for raw data of schema 3: `MatchData.game_build`, `TeamfightEvent.xp_delta`,
  `WardEvent.killer_slot`, `SmokeEvent.player_slot`, `PlayerData.buyback_log`, `KillEvent.killer_team`
  (None for output written before 1.4.0), `WardEvent.denied` (False before 1.5.0) and
  `KillEvent.respawn_time` (None before 1.5.0: the analysis then reads the respawn off
  `position_log`). `chat[]` entries are passed on as they are, `hero_id` included.

