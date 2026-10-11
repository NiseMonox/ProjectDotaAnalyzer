#!/usr/bin/env python3
"""Compare a raw clarity-analyzer JSON with the OpenDota JSON of the same match.

Dev tool for verifying tools/clarity-analyzer.jar against Valve's own numbers:

    java -jar tools/clarity-analyzer.jar replays/<id>.dem > raw.json
    curl https://api.opendota.com/api/matches/<id> > od.json
    python3 tools/compare_with_opendota.py raw.json od.json \
        [--ability-ids ability_ids.json] [--item-ids item_ids.json] [-v]

On Windows, make the two files in cmd.exe, not in Windows PowerShell 5.1: its ">" stores
a program's output re-encoded as UTF-16 (player names of the analyzer's UTF-8 JSON are
damaged on the way), and its "curl" is not curl:

    cmd /c "java -jar tools/clarity-analyzer.jar replays/<id>.dem > raw.json"
    curl.exe -o od.json https://api.opendota.com/api/matches/<id>
    python tools/compare_with_opendota.py raw.json od.json

It prints one line per mismatch ("DIFF"), per accepted source difference ("note") and,
with -v, per matching check ("ok"). Exit code 1 when there is at least one DIFF.
No network access: OpenDota stores items and abilities as numeric ids, so the id tables
are read from files. --item-ids / --ability-ids take dumps of OpenDota
/constants/item_ids and /constants/ability_ids; without --item-ids the ids found in
frontend/static/data/items.json are used; without --ability-ids the skill-build order
is not compared.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# OpenDota cell (64..192) -> world units, as backend/app/geo.py
CELL = 128.0
ORIGIN = 16384.0
# Full lifetime of a ward in seconds (backend/app/game_constants.py).
WARD_LIFETIME_S = {"observer": 360, "sentry": 420}

LANES = ("top", "mid", "bot")
STEAM_ID_BASE = 76561197960265728
TEAM_PREFIX = {"radiant": "goodguys", "dire": "badguys"}


class Report:
    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.diffs = 0
        self.notes = 0
        self.oks = 0

    def section(self, title: str) -> None:
        print(f"== {title}")

    def ok(self, msg: str) -> None:
        self.oks += 1
        if self.verbose:
            print(f"  ok    {msg}")

    def diff(self, msg: str) -> None:
        self.diffs += 1
        print(f"  DIFF  {msg}")

    def note(self, msg: str) -> None:
        self.notes += 1
        print(f"  note  {msg}")

    def check(self, label: str, ours, theirs, tol: float = 0, note_tol: float | None = None) -> None:
        """Equal within tol -> ok; within note_tol -> note (accepted difference); else DIFF."""
        if ours == theirs:
            self.ok(f"{label}: {ours}")
            return
        numeric = isinstance(ours, (int, float)) and isinstance(theirs, (int, float)) \
            and not isinstance(ours, bool) and not isinstance(theirs, bool)
        if numeric and abs(ours - theirs) <= tol:
            self.ok(f"{label}: {ours} ~ {theirs}")
        elif numeric and note_tol is not None and abs(ours - theirs) <= note_tol:
            self.note(f"{label}: analyzer {ours} vs OpenDota {theirs}")
        else:
            self.diff(f"{label}: analyzer {ours!r} vs OpenDota {theirs!r}")


def load_json(path: str | Path):
    """A JSON file in UTF-8, or in what its byte order mark says (Windows tools add one:
    "UTF-8 with BOM", and UTF-16 when PowerShell 5.1 wrote the file with ">")."""
    with open(path, "rb") as fh:
        head = fh.read(2)
    encoding = "utf-16" if head in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    with open(path, encoding=encoding) as fh:
        return json.load(fh)


def load_item_ids(path: str | None) -> dict[int, str]:
    if path:
        return {int(k): str(v) for k, v in load_json(path).items()}
    catalog = REPO / "frontend" / "static" / "data" / "items.json"
    if not catalog.is_file():
        return {}
    by_key = load_json(catalog).get("by_key", {})
    return {int(v["id"]): k for k, v in by_key.items() if isinstance(v, dict) and v.get("id") is not None}


def slot_label(slot: int, players: dict[int, dict]) -> str:
    p = players.get(slot) or {}
    return f"slot {slot} ({p.get('hero_name', '?')})"


def hero_key(npc: str) -> str:
    return npc[len("npc_dota_hero_"):] if npc.startswith("npc_dota_hero_") else npc


# ---------------------------------------------------------------- match level

def compare_match(rep: Report, raw: dict, od: dict) -> None:
    rep.section("match")
    rep.check("match_id", raw.get("match_id"), od.get("match_id"))
    rep.check("duration", raw.get("duration"), od.get("duration"), tol=1)
    rep.check("radiant_win", raw.get("radiant_win"), od.get("radiant_win"))
    rep.check("game_mode", raw.get("game_mode"), od.get("game_mode"))
    rep.check("radiant_score", raw.get("radiant_score"), od.get("radiant_score"))
    rep.check("dire_score", raw.get("dire_score"), od.get("dire_score"))
    rep.check("first_blood_time", raw.get("first_blood_time"), od.get("first_blood_time"), tol=1)
    rep.check("start_time", raw.get("start_time"), od.get("start_time"), tol=2, note_tol=60)
    if "cluster" in raw:
        rep.check("cluster", raw.get("cluster"), od.get("cluster"))
    # Unknown (a cluster OpenDota's table does not have): 0 here, null there.
    rep.check("region", raw.get("region") or 0, od.get("region") or 0)


# ---------------------------------------------------------------- players

SCOREBOARD_FIELDS = (
    # (raw field, OpenDota field, tolerance, note tolerance)
    ("hero_id", "hero_id", 0, None),
    ("kills", "kills", 0, None),
    ("deaths", "deaths", 0, None),
    ("assists", "assists", 0, None),
    ("last_hits", "last_hits", 0, None),
    ("denies", "denies", 0, None),
    ("level", "level", 0, None),
    ("net_worth", "net_worth", 0, None),
    # Valve divides by its own clock: one off is a rounding difference.
    ("gold_per_min", "gold_per_min", 0, 1),
    ("xp_per_min", "xp_per_min", 0, 1),
    ("hero_damage", "hero_damage", 0, None),
    ("tower_damage", "tower_damage", 0, None),
    ("hero_healing", "hero_healing", 0, None),
    ("buyback_count", "buyback_count", 0, None),
    ("roshans_killed", "roshan_kills", 0, None),
    ("tower_kills", "tower_kills", 0, None),
    ("courier_kills", "courier_kills", 0, None),
    ("neutral_kills", "neutral_kills", 0, None),
    ("ancient_kills", "ancient_kills", 0, None),
    # OpenDota also counts a kill (deny) of the team's own ward.
    ("observer_kills", "observer_kills", 0, 1),
    ("sentry_kills", "sentry_kills", 0, 1),
    # Different definition: OpenDota counts denied lane creeps and skips siege creeps.
    ("lane_kills", "lane_kills", 0, 30),
    ("actions_per_min", "actions_per_min", 0, None),
    ("teamfight_participation", "teamfight_participation", 0.0001, None),
    # OpenDota's last sample is up to a second before the game ended.
    ("stuns", "stuns", 0.01, 1.0),
)


def series_check(rep: Report, label: str, ours: list, theirs: list, tol: float) -> None:
    """A per-minute series. The analyzer samples where OpenDota samples, so the values are
    expected to be equal; a difference within tol is reported as a note, a larger one as DIFF."""
    differing = [abs(x - y) for x, y in zip(ours, theirs) if x != y]
    if not differing:
        rep.ok(f"{label} values")
    elif max(differing) > tol:
        rep.diff(f"{label}: max difference {max(differing)} (> {tol}), {len(differing)} values differ")
    else:
        rep.note(f"{label}: {len(differing)} of {min(len(ours), len(theirs))} values differ, at most by {max(differing)}")


def is_talent(key: str) -> bool:
    """A talent pick: any special_bonus_* key but the attribute bonus (as backend/app/parser.py)."""
    return key.startswith("special_bonus_") and key != "special_bonus_attributes"


def is_subsequence(short: list, long: list) -> bool:
    it = iter(long)
    return all(x in it for x in short)


def compare_players(rep: Report, raw: dict, od: dict, rp: dict[int, dict], op: dict[int, dict],
                    item_ids: dict[int, str], ability_ids: dict[str, str] | None) -> None:
    rep.section("players: scoreboard")
    rep.check("player slots", sorted(rp), sorted(op))
    for slot in sorted(rp):
        if slot not in op:
            continue
        a, b = rp[slot], op[slot]
        who = slot_label(slot, rp)
        # OpenDota account_id is the low part of the 64-bit Steam id; absent for hidden profiles.
        if a.get("steam_id") and b.get("account_id"):
            rep.check(f"{who} account id", a["steam_id"] - STEAM_ID_BASE, b["account_id"])
        for field, od_field, tol, note_tol in SCOREBOARD_FIELDS:
            if field not in a or b.get(od_field) is None:
                continue
            rep.check(f"{who} {field}", a.get(field), b.get(od_field), tol=tol, note_tol=note_tol)

    rep.section("players: minute series")
    for slot in sorted(rp):
        if slot not in op:
            continue
        a, b = rp[slot], op[slot]
        who = slot_label(slot, rp)
        for field, tol in (("gold_t", 150), ("xp_t", 150), ("lh_t", 3), ("dn_t", 2)):
            ours, theirs = a.get(field) or [], b.get(field) or []
            if not theirs or field not in a:
                continue
            rep.check(f"{who} {field} length", len(ours), len(theirs))
            series_check(rep, f"{who} {field}", ours, theirs, tol)
    for field in ("radiant_gold_adv", "radiant_xp_adv"):
        ours, theirs = raw.get(field) or [], od.get(field) or []
        if theirs:
            rep.check(f"{field} length", len(ours), len(theirs))
            series_check(rep, field, ours, theirs, 1000)

    rep.section("players: items")
    if not item_ids:
        rep.note("no item id table: items not compared")
    for slot in sorted(rp):
        if slot not in op or not item_ids:
            continue
        a, b = rp[slot], op[slot]
        who = slot_label(slot, rp)

        def od_items(prefix: str, count: int) -> tuple[list[str], list[int]]:
            keys, unknown = [], []
            for i in range(count):
                iid = b.get(f"{prefix}_{i}") or 0
                if not iid:
                    continue
                if iid in item_ids:
                    keys.append(item_ids[iid])
                else:
                    unknown.append(iid)
            return keys, unknown

        for field, prefix, count in (("final_items", "item", 6), ("backpack", "backpack", 3)):
            theirs, unknown = od_items(prefix, count)
            ours = [k for k in (a.get(field) or []) if k]
            if unknown:
                rep.note(f"{who} {field}: OpenDota item ids {unknown} are not in the id table")
                if len(ours) == len(theirs) + len(unknown) and not Counter(theirs) - Counter(ours):
                    continue
            rep.check(f"{who} {field}", sorted(ours), sorted(theirs))
        neutral_id = b.get("item_neutral") or 0
        if neutral_id and neutral_id not in item_ids:
            rep.note(f"{who} neutral_item: OpenDota item id {neutral_id} is not in the id table")
        else:
            rep.check(f"{who} neutral_item", a.get("neutral_item") or "",
                      item_ids.get(neutral_id, "") if neutral_id else "")
        if a.get("final_items_source") not in (None, "inventory"):
            rep.diff(f"{who} final_items_source is {a.get('final_items_source')!r}, not the inventory")

    rep.section("players: purchases")
    # Strategy-time rows carry the pre-game start time: the earliest time of any purchase.
    first = min((e.get("time", 0) for p in rp.values() for e in (p.get("purchase_log") or [])), default=0)
    strategy_end = min(first + 5, -30)
    for slot in sorted(rp):
        if slot not in op:
            continue
        a, b = rp[slot], op[slot]
        who = slot_label(slot, rp)
        theirs = Counter(e["key"] for e in (b.get("purchase_log") or []))
        ours = Counter(e["key"] for e in (a.get("purchase_log") or []) if not str(e.get("key", "")).startswith("recipe_"))
        if "ward_dispenser" in ours:
            rep.diff(f"{who} purchase_log has {ours['ward_dispenser']} ward_dispenser rows (stack merges)")
        extra = ours - theirs
        missing = theirs - ours
        # Strategy-time purchases have no combat-log row; both parsers take them from the
        # inventory at spawn (the analyzer stamps them with the pre-game start: about -90 s,
        # -60 s in Turbo).
        strategy = Counter(e["key"] for e in (a.get("purchase_log") or []) if e.get("time", 0) <= strategy_end)
        od_strategy = [e for e in (b.get("purchase_log") or []) if e["time"] <= strategy_end]
        od_has_strategy = bool(od_strategy)
        # Wards bought then arrive as one merged stack: OpenDota has no row for them, the
        # analyzer counts the stack's charges.
        ward_extra = extra & Counter({k: v for k, v in strategy.items() if k in ("ward_observer", "ward_sentry")})
        extra = extra - ward_extra
        if ward_extra:
            rep.note(f"{who} purchase_log: strategy-time wards OpenDota does not list: {dict(ward_extra)}")
        # Any other stack bought in strategy time is ONE OpenDota row with its charges (two
        # tangos: charges 6); the analyzer has one row per purchase.
        stacked = {e["key"] for e in od_strategy if (e.get("charges") or 0) > 1}
        stack_extra = extra & Counter({k: v for k, v in strategy.items() if k in stacked})
        extra = extra - stack_extra
        if stack_extra:
            rep.note(f"{who} purchase_log: strategy-time stacks that are one OpenDota row with charges: "
                     f"{dict(stack_extra)}")
        if extra and not od_has_strategy and not extra - strategy:
            rep.note(f"{who} purchase_log: OpenDota has no strategy-time purchases for this player; "
                     f"analyzer: {dict(extra)}")
            extra = Counter()
        if extra:
            rep.diff(f"{who} purchase_log extra keys: {dict(extra)}")
        if missing:
            rep.diff(f"{who} purchase_log missing keys (in OpenDota only): {dict(missing)}")
        if not extra and not missing:
            rep.ok(f"{who} purchase_log ({sum(ours.values())} rows)")

    rep.section("players: skill builds")
    if ability_ids is None:
        rep.note("no --ability-ids table: skill builds not compared")
    for slot in sorted(rp):
        if slot not in op or ability_ids is None:
            continue
        a, b = rp[slot], op[slot]
        who = slot_label(slot, rp)
        theirs = [ability_ids.get(str(x), f"ability_{x}") for x in (b.get("ability_upgrades_arr") or [])]
        ours = [str(u.get("name")) for u in (a.get("ability_upgrades") or [])]
        if a.get("ability_upgrades_source") not in (None, "entity"):
            rep.diff(f"{who} ability_upgrades_source is {a.get('ability_upgrades_source')!r}")
        if ours == theirs:
            rep.ok(f"{who} skill build ({len(ours)} upgrades)")
        elif len(theirs) == 25 and ours[:25] == theirs:
            # Valve's list is capped at 25 entries.
            rep.note(f"{who} skill build: equal for OpenDota's 25 entries, analyzer has {len(ours) - 25} more: "
                     f"{ours[25:]}")
        elif len(theirs) > len(ours) and is_subsequence(ours, theirs) \
                and all(is_talent(key) and key in ours for key in (Counter(theirs) - Counter(ours)).elements()):
            # A talent has one level: a second entry for it in Valve's list is no level-up
            # (Vengeful Spirit in 9038388762 has her level-10 talent twice). Any other entry
            # the analyzer lacks is a DIFF: an ability level missing from the build.
            rep.note(f"{who} skill build: equal except for {len(theirs) - len(ours)} repeated talent entries of "
                     f"OpenDota: {list((Counter(theirs) - Counter(ours)).elements())}")
        else:
            first = next((i for i, (x, y) in enumerate(zip(ours, theirs)) if x != y), min(len(ours), len(theirs)))
            rep.diff(f"{who} skill build differs at #{first + 1}: analyzer {ours[first:first + 3]} "
                     f"vs OpenDota {theirs[first:first + 3]} (lengths {len(ours)}/{len(theirs)})")

    rep.section("players: runes, buybacks, kill announcements")
    for slot in sorted(rp):
        if slot not in op:
            continue
        a, b = rp[slot], op[slot]
        who = slot_label(slot, rp)
        if b.get("runes") is not None:
            rep.check(f"{who} runes", {str(k): v for k, v in (a.get("runes") or {}).items()},
                      {str(k): v for k, v in (b.get("runes") or {}).items()})
        for field in ("multi_kills", "kill_streaks"):
            if field in a and b.get(field) is not None:
                rep.check(f"{who} {field}", {str(k): v for k, v in a[field].items()},
                          {str(k): v for k, v in b[field].items()})
        if "buyback_log" in a and b.get("buyback_log") is not None:
            ours = sorted(e["time"] for e in a["buyback_log"])
            theirs = sorted(e["time"] for e in b["buyback_log"])
            if len(ours) != len(theirs) or any(abs(x - y) > 2 for x, y in zip(ours, theirs)):
                rep.diff(f"{who} buyback_log: analyzer {ours} vs OpenDota {theirs}")
            else:
                rep.ok(f"{who} buyback_log {ours}")


# ---------------------------------------------------------------- wards

def compare_wards(rep: Report, raw: dict, od: dict, rp: dict[int, dict], op: dict[int, dict]) -> None:
    rep.section("wards")
    theirs: dict[int, dict] = {}
    left: dict[int, dict] = {}
    for slot, p in op.items():
        for kind, log, left_log in (("observer", "obs_log", "obs_left_log"), ("sentry", "sen_log", "sen_left_log")):
            for w in p.get(log) or []:
                theirs[w["ehandle"]] = dict(w, kind=kind, slot=slot)
            for w in p.get(left_log) or []:
                left[w["ehandle"]] = w
    ours = {w.get("ehandle"): w for w in raw.get("wards") or []}
    by_hero = {hero_key(p.get("hero_npc_name") or p.get("hero_name") or ""): slot for slot, p in rp.items()}
    rep.check("ward count", len(ours), len(theirs))
    for kind in ("observer", "sentry"):
        rep.check(f"{kind} count", sum(1 for w in ours.values() if w.get("type") == kind),
                  sum(1 for w in theirs.values() if w["kind"] == kind))
    duration = od.get("duration") or 0
    owner_bad = pos_bad = time_bad = left_bad = deward_bad = deny_bad = 0
    checked = deward_checked = deny_checked = denies = unknown_deward = 0
    for eh, w in ours.items():
        t = theirs.get(eh)
        if t is None:
            rep.diff(f"ward ehandle {eh} ({w.get('type')} at {w.get('time')}s) is not in OpenDota")
            continue
        checked += 1
        label = f"{w.get('type')} ward @{w.get('time')}s ehandle {eh}"
        if w.get("type") != t["kind"]:
            rep.diff(f"{label}: type {w.get('type')} vs OpenDota {t['kind']}")
        if w.get("player_slot") != t["slot"]:
            owner_bad += 1
            rep.diff(f"{label}: owner slot {w.get('player_slot')} ({w.get('owner_source')}) vs OpenDota {t['slot']}")
        wx, wy = t["x"] * CELL - ORIGIN, t["y"] * CELL - ORIGIN
        # OpenDota rounds to 0.1 cell (12.8 units).
        if abs(w.get("x", 0) - wx) > 13 or abs(w.get("y", 0) - wy) > 13:
            pos_bad += 1
            rep.diff(f"{label}: position ({w.get('x'):.0f},{w.get('y'):.0f}) vs OpenDota ({wx:.0f},{wy:.0f})")
        if abs(w.get("time", 0) - t["time"]) > 2:
            time_bad += 1
            rep.diff(f"{label}: placed {w.get('time')} vs OpenDota {t['time']}")
        lf = left.get(eh)
        destroyed = w.get("destroyed_time")
        if lf is None or lf["time"] > duration:
            # Still standing when the game ended (OpenDota also logs post-game expiries).
            if destroyed is not None and destroyed >= duration - 1 and w.get("dewarded"):
                # Dewarded in the very last second (the combat log has the kill): OpenDota's
                # left log is written a moment later, when the game is already over.
                rep.note(f"{label}: dewarded at {destroyed}, in the last second of the game; "
                         f"OpenDota has no left-log entry")
            elif destroyed is not None:
                left_bad += 1
                rep.diff(f"{label}: destroyed_time {destroyed} but OpenDota has it alive at game end")
        elif destroyed is None or abs(destroyed - lf["time"]) > 2:
            left_bad += 1
            rep.diff(f"{label}: destroyed_time {destroyed} vs OpenDota {lf['time']}")
        if lf is not None and lf["time"] <= duration:
            if "dewarded" not in w:
                unknown_deward += 1
            else:
                deward_checked += 1
                # OpenDota names the hero behind the death: the owner for a ward that expired,
                # an enemy hero for a dewarded one.
                attacker = by_hero.get(hero_key(str(lf.get("attackername") or "")))
                if attacker is None:
                    continue
                by_enemy = (attacker < 128) != (t["slot"] < 128)
                if bool(w["dewarded"]) != by_enemy:
                    deward_bad += 1
                    rep.diff(f"{label}: dewarded {w['dewarded']} but OpenDota attacker is slot {attacker} "
                             f"(owner slot {t['slot']})")
                elif by_enemy and w.get("killer_slot") not in (None, attacker):
                    deward_bad += 1
                    rep.diff(f"{label}: killer_slot {w.get('killer_slot')} vs OpenDota attacker slot {attacker}")
                elif not by_enemy and "denied" in w:
                    # Analyzer 1.5.0: an attack of the ward's own side. OpenDota names the
                    # owner for a ward that ran out and the attacking ally for a deny, so
                    # there only the lifetime tells the two apart.
                    deny_checked += 1
                    early = lf["time"] - t["time"] < WARD_LIFETIME_S[t["kind"]] - 5
                    denies += bool(w["denied"])
                    if bool(w["denied"]) != early:
                        deny_bad += 1
                        rep.diff(f"{label}: denied {w['denied']} but it stood {lf['time'] - t['time']} s and "
                                 f"OpenDota's attacker is slot {attacker} of its own side")
                    elif w["denied"] and w.get("killer_slot") != attacker:
                        deny_bad += 1
                        rep.diff(f"{label}: denied by slot {w.get('killer_slot')} vs OpenDota attacker slot {attacker}")
    for eh in theirs.keys() - ours.keys():
        rep.diff(f"OpenDota ward ehandle {eh} is missing from the analyzer output")
    if not (owner_bad or pos_bad or time_bad or left_bad):
        rep.ok(f"{checked} wards: owner, position, placement and end time match")
    if "dewarded" in next(iter(ours.values()), {}) or deward_checked:
        if not deward_bad:
            rep.ok(f"dewarded flag matches the OpenDota attacker for {deward_checked} destroyed wards")
        if unknown_deward:
            rep.note(f"{unknown_deward} destroyed wards have no dewarded flag (no combat-log death matched)")
    if deny_checked and not deny_bad:
        rep.ok(f"denied flag and denier match OpenDota for {deny_checked} wards ended by their own side "
               f"({denies} denied)")
    for slot in sorted(rp):
        if slot not in op:
            continue
        who = slot_label(slot, rp)
        b = op[slot]
        n_obs = sum(1 for w in ours.values() if w.get("player_slot") == slot and w.get("type") == "observer")
        n_sen = sum(1 for w in ours.values() if w.get("player_slot") == slot and w.get("type") == "sentry")
        rep.check(f"{who} observers placed", n_obs, len(b.get("obs_log") or []))
        rep.check(f"{who} sentries placed", n_sen, len(b.get("sen_log") or []))


# ---------------------------------------------------------------- objectives / buildings

def compare_objectives(rep: Report, raw: dict, od: dict) -> None:
    rep.section("objectives")
    ours = raw.get("objectives") or []
    theirs = od.get("objectives") or []

    def od_buildings():
        return sorted((o["key"], o["time"]) for o in theirs if o.get("type") == "building_kill")

    def our_buildings():
        return sorted(("npc_dota_" + str(o.get("key")), o.get("time")) for o in ours
                      if o.get("type") in ("tower_kill", "barracks_kill", "ancient_kill"))

    a, b = our_buildings(), od_buildings()
    rep.check("building kills", [k for k, _ in a], [k for k, _ in b])
    if len(a) == len(b):
        for (key, t1), (_, t2) in zip(a, b):
            rep.check(f"building kill {key} time", t1, t2, tol=1)
    for our_type, od_type in (("roshan_kill", "CHAT_MESSAGE_ROSHAN_KILL"), ("CHAT_MESSAGE_AEGIS", "CHAT_MESSAGE_AEGIS"),
                              ("CHAT_MESSAGE_FIRSTBLOOD", "CHAT_MESSAGE_FIRSTBLOOD"),
                              ("CHAT_MESSAGE_COURIER_LOST", "CHAT_MESSAGE_COURIER_LOST")):
        t1 = sorted(o.get("time") for o in ours if o.get("type") == our_type)
        t2 = sorted(o.get("time") for o in theirs if o.get("type") == od_type)
        if len(t1) != len(t2) or any(abs(x - y) > 1 for x, y in zip(t1, t2)):
            rep.diff(f"{od_type}: analyzer times {t1} vs OpenDota {t2}")
        else:
            rep.ok(f"{od_type}: {t1}")
    for o in theirs:
        if o.get("type") == "CHAT_MESSAGE_FIRSTBLOOD":
            mine = next((x for x in ours if x.get("type") == "CHAT_MESSAGE_FIRSTBLOOD"), None)
            if mine is not None:
                rep.check("first blood killer slot", mine.get("player_slot"), o.get("player_slot"))
                if "victim_slot" in mine and o.get("victim_player_slot") is not None:
                    rep.check("first blood victim slot", mine.get("victim_slot"), o.get("victim_player_slot"))
    # Courier kills: OpenDota has the losing team as a Valve team id and the killer as a
    # player_slot (0-4 / 128-132; absent when no hero made the kill).
    slots = {p.get("player_slot") for p in od.get("players") or []}
    mine = sorted((o for o in ours if o.get("type") == "CHAT_MESSAGE_COURIER_LOST"), key=lambda o: o.get("time"))
    lost = sorted((o for o in theirs if o.get("type") == "CHAT_MESSAGE_COURIER_LOST"), key=lambda o: o.get("time"))
    for a, b in zip(mine, lost):
        killer = b.get("killer")
        rep.check(f"courier lost @{b.get('time')}: team, killer slot, bounty",
                  [a.get("team"), a.get("player_slot"), a.get("value")],
                  [{2: "radiant", 3: "dire"}.get(b.get("team")),
                   killer if killer in slots else None, b.get("value")])


def building_bitmasks(buildings: list[dict]) -> dict[str, int]:
    """Valve tower / barracks status bits from the raw buildings list (1 = standing)."""
    out = {"tower_status_radiant": 0, "tower_status_dire": 0, "barracks_status_radiant": 0, "barracks_status_dire": 0}
    for b in buildings:
        name = str(b.get("name") or "")
        if b.get("destroyed_time") is not None:
            continue
        for team, prefix in TEAM_PREFIX.items():
            head = f"npc_dota_{prefix}_"
            if not name.startswith(head):
                continue
            rest = name[len(head):]
            if rest.startswith("tower4"):
                bit = 9 if rest.endswith("top") else 10
                out[f"tower_status_{team}"] |= 1 << bit
            elif rest.startswith("tower"):
                tier, lane = int(rest[5]), rest.split("_")[-1]
                if lane in LANES:
                    out[f"tower_status_{team}"] |= 1 << (LANES.index(lane) * 3 + tier - 1)
            elif "_rax_" in rest:
                kind, lane = rest.split("_rax_")
                if lane in LANES:
                    out[f"barracks_status_{team}"] |= 1 << (LANES.index(lane) * 2 + (1 if kind == "range" else 0))
    return out


def compare_buildings(rep: Report, raw: dict, od: dict) -> None:
    rep.section("buildings")
    buildings = raw.get("buildings") or []
    rep.check("building count", len(buildings), 36)
    unnamed = [b for b in buildings if not str(b.get("name") or "").startswith("npc_dota_")]
    if unnamed:
        rep.diff(f"{len(unnamed)} buildings without a canonical name: {sorted({b.get('name') for b in unnamed})[:6]}")
    names = Counter(b.get("name") for b in buildings)
    dup = {n: c for n, c in names.items() if c > 1}
    if dup:
        rep.diff(f"duplicate building names: {dup}")
    masks = building_bitmasks(buildings)
    for key, value in masks.items():
        rep.check(key, value, od.get(key))
    kills = {o["key"]: o["time"] for o in od.get("objectives") or [] if o.get("type") == "building_kill"}
    for b in buildings:
        name = str(b.get("name") or "")
        od_name = name.replace("_tower4_top", "_tower4").replace("_tower4_bot", "_tower4")
        if b.get("destroyed_time") is not None and od_name not in kills:
            rep.diff(f"{name} destroyed at {b['destroyed_time']} but OpenDota has no building_kill for it")
        elif b.get("destroyed_time") is None and od_name in kills and "tower4" not in name:
            rep.diff(f"{name} alive but OpenDota has a building_kill at {kills[od_name]}")


# ---------------------------------------------------------------- draft

def compare_draft(rep: Report, raw: dict, od: dict) -> None:
    rep.section("draft")
    ours = raw.get("picks_bans") or []
    theirs = od.get("picks_bans") or []
    if not theirs:
        rep.note("OpenDota has no picks_bans")
        return
    our_picks = sorted(ours, key=lambda e: e.get("order", 0))
    our_picks = [e for e in our_picks if e.get("is_pick")]
    od_picks = [e for e in sorted(theirs, key=lambda e: e.get("order", 0)) if e.get("is_pick")]
    our_bans = [e.get("hero_id") for e in sorted(ours, key=lambda e: e.get("order", 0)) if not e.get("is_pick")]
    od_bans = [e.get("hero_id") for e in sorted(theirs, key=lambda e: e.get("order", 0)) if not e.get("is_pick")]
    # The picks are the ten heroes that were played, on the side that played them.
    played = {p.get("hero_id"): 0 if p.get("player_slot", 0) < 128 else 1 for p in od.get("players") or []}
    rep.check("picks are the heroes played", {e.get("hero_id"): e.get("team") for e in our_picks}, played)
    # Valve's list keeps a pick that was taken back (repick) as a pick, although nobody played
    # the hero; the replay has it among the unavailable heroes (bans). It has no entry for a
    # hero that was assigned when the pick time ran out.
    repicked = [e.get("hero_id") for e in od_picks if e.get("hero_id") not in played]
    assigned = [e.get("hero_id") for e in our_picks if e.get("hero_id") not in {x.get("hero_id") for x in od_picks}]
    od_picks = [e for e in od_picks if e.get("hero_id") in played]
    if repicked:
        (rep.note if set(repicked) <= set(our_bans) else rep.diff)(
            f"OpenDota picks nobody played (repicked): {repicked}; the replay lists them as unavailable")
    if assigned:
        rep.note(f"picks without an OpenDota pick entry (assigned at the end of the draft): {assigned}")
    rep.check("pick teams", {e.get("hero_id"): e.get("team") for e in our_picks if e.get("hero_id") not in assigned},
              {e.get("hero_id"): e.get("team") for e in od_picks})
    a = [e.get("hero_id") for e in our_picks if e.get("hero_id") not in assigned]
    b = [e.get("hero_id") for e in od_picks]
    if a == b:
        rep.ok("pick order")
    elif repicked:
        # A repicking player's final hero has no order of its own in Valve's list.
        rep.note(f"pick order not comparable after {len(repicked)} repicks: analyzer {a} vs OpenDota {b}")
    else:
        moved = [h for i, h in enumerate(a) if i >= len(b) or b[i] != h]
        # All pick: both teams pick in the same phase, so neighbours can swap.
        swapped_only = all(abs(a.index(h) - b.index(h)) <= 1 for h in a if h in b)
        (rep.note if swapped_only else rep.diff)(f"pick order: analyzer {a} vs OpenDota {b} (moved: {moved})")
    missing = [h for h in od_bans if h not in our_bans]
    if missing:
        rep.diff(f"bans in OpenDota only: {missing}")
    else:
        rep.ok(f"all {len(od_bans)} OpenDota bans are present")
    if len(our_bans) > len(od_bans):
        rep.note(f"{len(our_bans) - len(od_bans)} bans in the replay that Valve's API does not list "
                 f"(not nominated by a player): {[h for h in our_bans if h not in od_bans]}")
    ban_teams = {e.get("team") for e in ours if not e.get("is_pick")}
    if ban_teams - {None} and raw.get("draft_source") != "file_info":
        rep.diff(f"bans carry invented teams: {ban_teams}")
    elif ban_teams == {None}:
        rep.note("ban teams are null (the replay does not say who nominated a ban)")


# ---------------------------------------------------------------- kills log

def compare_kills_log(rep: Report, raw: dict, od: dict, rp: dict[int, dict], op: dict[int, dict]) -> None:
    rep.section("kills_log (W1)")
    log = raw.get("kills_log")
    if log is None:
        rep.note("raw JSON has no kills_log (schema 2)")
        return
    times = [k.get("time") for k in log]
    if times != sorted(times):
        rep.diff("kills_log is not time-ordered")
    by_hero = {hero_key(p.get("hero_npc_name") or p.get("hero_name") or ""): slot for slot, p in rp.items()}
    # Scoreboard credits the analyzer could attach to no death (it reports how many): the
    # scoreboard total is then that much above the log.
    loose = (raw.get("parse_errors") or {}).get("score_credits") or None
    for slot in sorted(rp):
        p = rp[slot]
        who = slot_label(slot, rp)
        rep.check(f"{who} deaths == kills_log victims", sum(1 for k in log if k.get("victim_slot") == slot),
                  p.get("deaths"))
        rep.check(f"{who} kills == kills_log killers",
                  sum(1 for k in log if k.get("killer_slot") == slot and not k.get("deny")), p.get("kills"),
                  note_tol=loose)
        rep.check(f"{who} assists == kills_log assist_slots",
                  sum(1 for k in log if slot in (k.get("assist_slots") or []) and not k.get("deny")), p.get("assists"),
                  note_tol=loose)
    # Valve's score from the log alone: a death scores for the side of the credited hero, else
    # for the side of the unit that landed the kill (killer_team: a tower, a creep, the fountain).
    if all("killer_team" in k for k in log):
        def side(slot: int) -> str:
            return "radiant" if slot < 128 else "dire"
        scored = Counter((side(k["killer_slot"]) if k.get("killer_slot") is not None else k.get("killer_team"),
                          side(k["victim_slot"])) for k in log if not k.get("deny"))
        rep.check("radiant_score == kills_log deaths that score for radiant", scored["radiant", "dire"],
                  od.get("radiant_score"), note_tol=loose)
        rep.check("dire_score == kills_log deaths that score for dire", scored["dire", "radiant"],
                  od.get("dire_score"), note_tol=loose)
    else:
        rep.note("kills_log has no killer_team (analyzer before 1.4.0): score not derived from the log")
    for k in log:
        if k.get("killer_slot") is not None and k.get("killer_slot") in (k.get("assist_slots") or []):
            rep.diff(f"kills_log @{k.get('time')}: killer {k.get('killer_slot')} is also listed as assist")
        if not k.get("deny") and k.get("killer_slot") is not None \
                and (k["killer_slot"] < 128) == (k["victim_slot"] < 128):
            rep.diff(f"kills_log @{k.get('time')}: killer and victim on the same team without deny")
    # OpenDota: per killer {time, key = victim npc name}; includes reincarnation "kills".
    od_kills = []
    for slot, p in op.items():
        for e in p.get("kills_log") or []:
            od_kills.append((e["time"], slot, by_hero.get(hero_key(e["key"]))))
    unmatched_od = list(od_kills)
    missing = 0
    for k in log:
        if k.get("deny") or k.get("killer_slot") is None:
            continue
        hit = next((e for e in unmatched_od if e[1] == k["killer_slot"] and e[2] == k["victim_slot"]
                    and abs(e[0] - k["time"]) <= 2), None)
        if hit is None and k.get("killer_unit") and not str(k["killer_unit"]).startswith("npc_dota_hero_"):
            # The scoreboard credits the player; OpenDota's log only has kills a hero landed.
            rep.note(f"kills_log @{k['time']}: {k['killer_slot']} -> {k['victim_slot']} was landed by "
                     f"{k['killer_unit']} and credited by the scoreboard; not in OpenDota kills_log")
        elif hit is None:
            missing += 1
            rep.diff(f"kills_log @{k['time']}: {k['killer_slot']} -> {k['victim_slot']} is not in OpenDota kills_log")
        else:
            unmatched_od.remove(hit)
    if not missing:
        rep.ok(f"every hero kill of kills_log is in OpenDota's kills_log ({len(od_kills) - len(unmatched_od)})")
    if unmatched_od:
        deaths = Counter(e[2] for e in od_kills)
        surplus = {v: deaths[v] - (op[v].get("deaths") or 0) for v in deaths if v in op}
        explained = sum(max(0, n) for n in surplus.values())
        msg = (f"{len(unmatched_od)} OpenDota kills_log entries have no analyzer kill "
               f"(reincarnation deaths: OpenDota lists {explained} more deaths than the scoreboard): "
               f"{[(t, a, v) for t, a, v in sorted(unmatched_od)]}")
        (rep.note if len(unmatched_od) <= explained else rep.diff)(msg)


# ---------------------------------------------------------------- teamfights / chat

def compare_teamfights(rep: Report, raw: dict, od: dict) -> None:
    rep.section("teamfights")
    ours = raw.get("teamfights") or []
    theirs = od.get("teamfights") or []
    rep.check("teamfight count", len(ours), len(theirs), note_tol=3)
    kill_times = [k.get("time", 0) for k in raw.get("kills_log") or []]
    used = set()
    for tf in theirs:
        best = None
        for i, mine in enumerate(ours):
            if i in used:
                continue
            overlap = min(mine.get("end", 0), tf["end"]) - max(mine.get("start", 0), tf["start"])
            if overlap > 0 and (best is None or overlap > best[1]):
                best = (i, overlap)
        if best is None:
            rep.diff(f"OpenDota teamfight {tf['start']}-{tf['end']} ({tf.get('deaths')} deaths) has no analyzer fight")
            continue
        used.add(best[0])
        mine = ours[best[0]]
        label = f"teamfight {tf['start']}-{tf['end']}"
        rep.check(f"{label} start", mine.get("start"), tf["start"], tol=2, note_tol=20)
        rep.check(f"{label} end", mine.get("end"), tf["end"], tol=2, note_tol=20)
        n = mine.get("radiant_deaths", 0) + mine.get("dire_deaths", 0)
        # OpenDota counts reincarnation deaths as deaths.
        rep.check(f"{label} deaths", n, tf.get("deaths"), note_tol=2)
        mp = {p.get("player_slot"): p for p in mine.get("players") or []}
        slots = sorted(mp)
        for idx, tp in enumerate(tf.get("players") or []):
            if idx >= len(slots):
                break
            m = mp[slots[idx]]
            if "buybacks" in m:
                rep.check(f"{label} slot {slots[idx]} buybacks", m.get("buybacks"), tp.get("buybacks"))
            if mine.get("start") == tf["start"] and mine.get("end") == tf["end"]:
                # OpenDota's deltas are differences of its 1 s samples on its rounded clock, the
                # analyzer sums the combat-log rows of the window's whole seconds: a hero kill
                # within a second of a window edge puts its whole bounty on different sides.
                edge_kill = any(abs(t - tf["start"]) <= 1 or abs(t - tf["end"] - 0.5) <= 1 for t in kill_times)
                for field in ("gold_delta", "xp_delta"):
                    rep.check(f"{label} slot {slots[idx]} {field}", m.get(field), tp.get(field),
                              tol=60, note_tol=float("inf") if edge_kill else 400)
    for i, mine in enumerate(ours):
        if i not in used:
            rep.note(f"analyzer teamfight {mine.get('start')}-{mine.get('end')} "
                     f"({mine.get('radiant_deaths', 0) + mine.get('dire_deaths', 0)} deaths) is not in OpenDota")


def compare_chat(rep: Report, raw: dict, od: dict) -> None:
    rep.section("chat")
    ours = raw.get("chat") or []
    theirs = od.get("chat") or []
    rep.check("chat entries", len(ours), len(theirs))
    rep.check("chat types", dict(Counter(c.get("type") for c in ours)), dict(Counter(c.get("type") for c in theirs)))
    # A message from somebody who is not one of the ten players (broadcaster, admin): -1 here,
    # no player_slot there.
    rep.check("chat slots", [c.get("player_slot") if c.get("player_slot", -1) >= 0 else None for c in ours],
              [c.get("player_slot") for c in theirs])


def compare_position_log(rep: Report, raw: dict, od: dict, rp: dict[int, dict], op: dict[int, dict]) -> None:
    rep.section("position_log (W2)")
    duration = raw.get("duration") or 0
    for slot in sorted(rp):
        p = rp[slot]
        who = slot_label(slot, rp)
        log = p.get("position_log")
        if log is None:
            rep.note("raw JSON has no position_log (schema 2)")
            return
        if not log:
            rep.diff(f"{who} position_log is empty")
            continue
        times = [e[0] for e in log]
        steps = {b - a for a, b in zip(times, times[1:])}
        if steps - {5}:
            rep.diff(f"{who} position_log steps are {sorted(steps)}, expected 5 s")
        if times[0] > 0 or times[-1] < duration - 5 or times[-1] > duration:
            rep.diff(f"{who} position_log covers {times[0]}..{times[-1]} for a {duration}s game")
        if any(len(e) != 4 or not all(isinstance(v, int) for v in e) for e in log):
            rep.diff(f"{who} position_log entries are not [int, int, int, int]")
        if any(e[3] < -1 or e[3] > 100 for e in log):
            rep.diff(f"{who} position_log hp_pct out of range")
        # Dead time: OpenDota life_state_dead counts seconds (life_state 1 + 2).
        dead = sum(5 for e in log if e[0] >= 0 and e[3] == 0)
        od_dead = (op.get(slot) or {}).get("life_state_dead")
        if od_dead is not None:
            rep.check(f"{who} seconds dead (hp_pct 0)", dead, od_dead, tol=max(20, 0.15 * od_dead), note_tol=60)
        # Laning position: compare with OpenDota lane_pos cells (first 10 minutes).
        lane_pos = (op.get(slot) or {}).get("lane_pos") or {}
        cells = {(int(x), int(y)) for x, ys in lane_pos.items() for y in ys}
        if cells:
            lan = [e for e in log if 0 <= e[0] < 600]
            near = sum(1 for e in lan if any(abs((e[1] + ORIGIN) / CELL - cx) <= 2 and abs((e[2] + ORIGIN) / CELL - cy) <= 2
                                             for cx, cy in cells))
            if lan and near / len(lan) < 0.9:
                rep.diff(f"{who} position_log: only {near}/{len(lan)} laning samples are near an OpenDota lane_pos cell")
            else:
                rep.ok(f"{who} position_log: {near}/{len(lan)} laning samples near OpenDota lane_pos cells")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_json", help="stdout of the analyzer jar")
    ap.add_argument("opendota_json", help="OpenDota /api/matches/<id> response")
    ap.add_argument("--ability-ids", help="dump of OpenDota /constants/ability_ids")
    ap.add_argument("--item-ids", help="dump of OpenDota /constants/item_ids")
    ap.add_argument("-v", "--verbose", action="store_true", help="also print matching checks")
    args = ap.parse_args()
    # Player names and chat lines are printed as they are: output redirected to a file
    # on Windows is in the system code page, which must not make a name a crash.
    for stream in (sys.stdout, sys.stderr):
        if getattr(stream, "errors", None) == "strict" and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")

    raw = load_json(args.raw_json)
    od = load_json(args.opendota_json)
    item_ids = load_item_ids(args.item_ids)
    ability_ids = {str(k): str(v) for k, v in load_json(args.ability_ids).items()} if args.ability_ids else None
    rp = {p.get("player_slot"): p for p in raw.get("players") or []}
    op = {p.get("player_slot"): p for p in od.get("players") or []}

    rep = Report(args.verbose)
    print(f"analyzer schema_version {raw.get('schema_version')} match {raw.get('match_id')} "
          f"vs OpenDota match {od.get('match_id')}")
    if raw.get("partial"):
        rep.note("the analyzer output is partial (truncated replay)")
    compare_match(rep, raw, od)
    compare_players(rep, raw, od, rp, op, item_ids, ability_ids)
    compare_wards(rep, raw, od, rp, op)
    compare_objectives(rep, raw, od)
    compare_buildings(rep, raw, od)
    compare_draft(rep, raw, od)
    compare_kills_log(rep, raw, od, rp, op)
    compare_position_log(rep, raw, od, rp, op)
    compare_teamfights(rep, raw, od)
    compare_chat(rep, raw, od)
    errors = dict(raw.get("parse_errors") or {})
    credits = errors.pop("score_credits", 0)
    if credits:
        rep.note(f"{credits} scoreboard kill / assist credits belong to no death of kills_log")
    if errors:
        rep.diff(f"parse_errors: {errors}")
    print(f"== summary: {rep.diffs} DIFF, {rep.notes} notes, {rep.oks} ok")
    return 1 if rep.diffs else 0


if __name__ == "__main__":
    sys.exit(main())
