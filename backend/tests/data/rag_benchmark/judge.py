"""Relevance judge for the RAG benchmark (grades 0-3).

The judge is a written rubric applied to (situation, chunk) pairs:

  3 = directly about this exact hero / matchup AND the situation at hand, or the
      principle chunk that is exactly about this situation type
  2 = directly about a hero involved in the situation (player, killer, nearby
      enemy) or directly about the situation type, but not both
  1 = generic but applicable (right role+phase background, an enemy who is in the
      game but not in this situation, marginal topical link)
  0 = not applicable: a hero that is not involved, wrong role / phase, unrelated topic

How the grade is produced
- Hero / matchup / item chunks (editorial, ~720 of 832): the grade is a function of
  WHAT the chunk is (overview / laning / abilities / matchup / item tip) and the
  ROLE of its hero in the situation (self / killer / near enemy / far enemy /
  ally / absent), the phase and the point type.  See `_hero_chunk_grade`.
- Generic chunks (55 builtin + 58 learned): every chunk was read once and given an
  applicability function over the situation features that matter for it (point
  type, teamfight vs pick-off, phase, role, root cause, behind or not ...).  The
  chunk's own retrieval tags are NOT used.  See `GENERIC`.
- `overrides.json` (optional, next to this file) holds per-pair corrections:
  ``{qid: {chunk_id: grade}}``. None were needed so far.

Usage (from the project root; read-only unless ``--write`` is given):

    backend/venv/bin/python backend/tests/data/rag_benchmark/judge.py          # compare labels.json with the rubric
    backend/venv/bin/python backend/tests/data/rag_benchmark/judge.py --write  # regrade the current corpus, rewrite labels.json / corpus.json ids

``--write`` is how new or re-tagged chunks get labels; a chunk without a rule
(a new principle or learned chunk) must first be read and added to `GENERIC`.

The judge never looks at retrieval scores.  It does share one concept with the
current ranker's entity gate ("a chunk about a hero that is not in the situation
is noise"), because that is the rubric the task prescribes.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

CORE_ROLES = {1: "carry", 2: "mid", 3: "offlane"}
ROLES = {1: "carry", 2: "mid", 3: "offlane", 4: "support_4", 5: "support_5"}

# ---------------------------------------------------------------------------
# Item classes (assigned by reading each editorial item tip)
# ---------------------------------------------------------------------------
ITEM_DEF = {  # saves / escapes / survivability actives: relevant when the player died
    "aeon_disk", "black_king_bar", "blade_mail", "cyclone", "force_staff", "ghost", "glimmer_cape",
    "hurricane_pike", "invis_sword", "lotus_orb", "manta", "satanic", "sphere", "wind_waker",
    "guardian_greaves", "mekansm", "pipe", "crimson_guard", "heart", "eternal_shroud", "phase_boots",
    "wraith_pact", "disperser", "armlet", "ethereal_blade", "boots_of_bearing", "solar_crest",
    "holy_locket", "blink", "silver_edge", "shivas_guard", "tranquil_boots",
}
ITEM_OFF = {  # initiation / lockdown / fight actives: relevant in fights
    "abyssal_blade", "arcane_blink", "basher", "blink", "bloodthorn", "dagon", "gungir", "harpoon",
    "heavens_halberd", "meteor_hammer", "nullifier", "orchid", "overwhelming_blink", "rod_of_atos",
    "sheepstick", "silver_edge", "swift_blink", "veil_of_discord", "diffusal_blade", "ancient_janggo",
    "arcane_boots", "assault", "refresher", "mjollnir", "spirit_vessel", "aether_lens", "echo_sabre",
    "mage_slayer", "shivas_guard", "invis_sword", "black_king_bar", "satanic", "manta", "armlet",
    "blade_mail", "pipe", "crimson_guard", "guardian_greaves", "mekansm", "lotus_orb", "wraith_pact",
    "boots_of_bearing", "solar_crest", "ethereal_blade", "disperser", "vladmir", "desolator",
    "maelstrom", "radiance",
}
ITEM_FARM = {"bfury", "hand_of_midas", "maelstrom", "mjollnir", "radiance", "helm_of_the_overlord",
             "travel_boots", "power_treads"}


def _item_grade(item: str, owned: bool, buyer: bool, f: dict) -> int:
    t = f["type"]
    fight = t in ("fight_engage", "good_play") or (t == "death" and f["tf"])
    death = t in ("death", "overextend")
    g = 0
    if death and item in ITEM_DEF:
        g = 2
    if fight and item in ITEM_OFF:
        g = max(g, 2)
    if t == "missed_cs" and item in ITEM_FARM:
        g = max(g, 3 if item == "bfury" else 2)
    if t == "bad_trade" and item in ("tranquil_boots",):
        g = max(g, 2)
    if g == 0:
        # an owned item whose tip is not about this kind of situation
        if fight or (death and item in ITEM_FARM | ITEM_OFF) or (t == "bad_trade" and item in ("phase_boots", "power_treads")):
            g = 1
    if owned:
        return g
    if buyer:  # "what to build" advice for this hero, only when on-topic
        return 1 if g >= 2 else 0
    return 0


# ---------------------------------------------------------------------------
# Situation features
# ---------------------------------------------------------------------------

def features(q: dict) -> dict:
    mc = q.get("match_context") or q   # benchmark records store gold_adv / vision_danger flat
    role = ROLES.get(q.get("position"))
    t = q["point_type"]
    try:
        gold_adv = float(mc.get("gold_adv") or 0)
    except (TypeError, ValueError):
        gold_adv = 0.0
    return _finish({
        "type": t,
        "tf": bool(q.get("is_teamfight_death")),
        "pick": t == "death" and not q.get("is_teamfight_death"),
        "phase": q.get("phase") or "",
        "rc_raw": q.get("root_cause_type") or "",
        "role": role,
        "core": role in ("carry", "mid", "offlane"),
        "support": role in ("support_4", "support_5"),
        "rc": q.get("root_cause_type") or "",
        "behind": gold_adv <= -5000,
        "slightly_behind": gold_adv < -1500,
        "time": q.get("time") or 0,
        "time_known": bool(q.get("time_known", True)),
        "subtype": q.get("subtype") or "",
        "overext": bool(q.get("overextended")),
        "danger": mc.get("vision_danger") in ("danger", "critical"),
        "self": q["self_slug"],
        "killer": q.get("killer_slug") or "",
        "near": set(q.get("near_enemy_slugs") or []),
        "enemies": set(q.get("enemy_slugs") or []),
        "allies": set(q.get("ally_slugs") or []),
        "items": set(q.get("item_keys") or []),
    })


def _finish(f: dict) -> dict:
    """Derived features: a pick-off is a *gank* (vision / map problem) from the mid game on or
    when the causal chain says so; before that it is a *lane death* against the lane opponents.
    With more than two enemies around, only the killer is in focus; the others are background."""
    vision_rc = f["rc"] in ("vision_loss", "ignore_missing") or f["danger"]
    f["gank"] = f["pick"] and (f["phase"] in ("mid_game", "late_game") or vision_rc)
    f["lane_death"] = f["pick"] and not f["gank"]
    f["focus"] = set(f["near"]) if len(f["near"]) <= 2 else set()
    if f["killer"]:
        f["focus"].add(f["killer"])
    return f


def _hero_role(slug: str, f: dict) -> str:
    if slug == f["self"]:
        return "self"
    if slug and slug == f["killer"]:
        return "killer"
    if slug in f["focus"]:
        return "near"
    if slug in f["near"]:
        return "near_bg"
    if slug in f["enemies"]:
        return "far"
    if slug in f["allies"]:
        return "ally"
    return "absent"


# ---------------------------------------------------------------------------
# Generic chunks: one applicability function per chunk (written after reading it)
# ---------------------------------------------------------------------------

def _vision_rc(f):
    return f["rc"] in ("vision_loss", "ignore_missing") or f["danger"]


def _isolated_rc(f):
    return f["overext"] or f["rc"] in ("isolated", "overextend", "greedy_farm", "bad_positioning")


LANING = ("pregame", "laning")
MIDLATE = ("mid_game", "late_game")


def _laning_principles(role):
    def fn(f):
        if f["role"] != role:
            return 0
        if f["phase"] == "pregame":
            return 1
        if f["phase"] != "laning":
            return 0
        t = f["type"]
        if t == "missed_cs":
            return 3 if role in ("carry", "mid") else 2
        if t == "overextend":
            return 3 if role == "carry" else 2
        if t in ("bad_trade", "death"):
            return 2
        return 1
    return fn


def _phase_objectives(phase, role):
    def fn(f):
        if f["role"] != role or f["phase"] != phase:
            return 0
        return 1 if f["type"] == "good_play" else 2
    return fn


def _pro_pattern(role):
    def fn(f):
        if f["role"] != role:
            return 0
        t = f["type"]
        if role in ("carry", "mid", "offlane"):
            if t == "missed_cs":
                return 3
            return 1 if t in ("death", "bad_trade", "overextend") else 0
        if t == "death":
            return 2
        return 1 if t in ("overextend", "bad_trade") else 0
    return fn


def _self_only(slug, fn):
    return lambda f: fn(f) if f["self"] == slug else 0


GENERIC = {
    # --- builtin coaching principles -------------------------------------
    "coaching:buyback_strategy": lambda f: (2 if f["phase"] == "late_game" else 1 if f["phase"] == "mid_game" else 0)
        if f["type"] == "death" else (1 if f["type"] == "fight_engage" and f["phase"] == "late_game" else 0),
    "coaching:comeback_mechanics": lambda f: (2 if f["behind"] else 1 if f["slightly_behind"] else 0)
        if f["type"] in ("death", "fight_engage") and f["phase"] in MIDLATE else 0,
    "coaching:farming_efficiency": lambda f: (2 if f["core"] else 1) if f["type"] == "missed_cs"
        else (2 if f["type"] == "death" and f["rc"] == "greedy_farm" else 0),
    "coaching:highground_defense": lambda f: (2 if f["behind"] else 1)
        if f["phase"] == "late_game" and f["type"] in ("death", "fight_engage") else 0,
    "coaching:lane_trading": lambda f: 3 if f["type"] in ("bad_trade", "overextend") and f["phase"] in LANING + ("mid_game",)
        else (2 if f["type"] == "death" and f["phase"] == "laning"
              else 1 if f["type"] == "missed_cs" and f["phase"] == "laning" else 0),
    "coaching:map_awareness": lambda f: (3 if _vision_rc(f) else 2) if f["gank"] else 1 if f["lane_death"]
        else {"overextend": 2, "bad_trade": 1, "fight_engage": 1}.get(f["type"], 1 if f["type"] == "death" else 0),
    "coaching:roshan_timing": lambda f: 0,
    "coaching:rune_control": lambda f: 3 if f["rc"] == "rune_contest_unfavorable"
        else (2 if f["phase"] == "pregame" and f["type"] == "death"
              else 1 if f["role"] == "mid" and f["phase"] == "laning" and f["type"] in ("bad_trade", "death", "missed_cs") else 0),
    "coaching:smoke_gank": lambda f: 3 if f["subtype"] == "smoke"
        else (1 if f["pick"] and f["phase"] == "mid_game" else 0),
    "coaching:split_push": lambda f: 2 if f["pick"] and f["core"] and f["phase"] in MIDLATE and _isolated_rc(f) else 0,
    "coaching:stack_pull": lambda f: 1 if f["support"] and f["phase"] == "laning" and f["type"] in ("bad_trade", "death", "overextend") else 0,
    "coaching:teamfight_positioning": lambda f: 3 if (f["type"] == "death" and f["tf"]) or f["type"] == "fight_engage"
        else (2 if f["type"] == "good_play" and f["subtype"] != "smoke"
              else 1 if f["pick"] and f["phase"] in MIDLATE else 0),
    "coaching:tower_aggro": lambda f: 1 if f["phase"] == "laning" and f["type"] in ("death", "bad_trade", "missed_cs", "overextend") else 0,
    "coaching:tp_scroll": lambda f: 1 if f["type"] == "death" else 0,
    # --- builtin death analysis ------------------------------------------
    "death_analysis:positioning": lambda f: (2 if f["tf"] and f["rc"] != "teamfight_positioning" else 3) if f["type"] == "death"
        else (3 if f["type"] == "overextend" else 0),
    "death_analysis:scoring_rubric": lambda f: 2 if f["type"] == "death" else (1 if f["type"] == "overextend" else 0),
    "death_analysis:skill_item": lambda f: 2 if f["type"] == "death" else (1 if f["type"] == "overextend" else 0),
    "death_analysis:timing": lambda f: 2 if f["type"] == "death" else 0,
    "death_analysis:trade_eval": lambda f: (3 if f["tf"] else 2) if f["type"] == "death" else 0,
    "death_analysis:vision": lambda f: (3 if (f["gank"] or _vision_rc(f)) else 2) if f["type"] == "death"
        else (2 if f["type"] == "overextend" else 0),
    # --- builtin early game ----------------------------------------------
    "early_game:level1_fight_assessment": lambda f: 3 if f["phase"] == "pregame" and f["type"] in ("death", "fight_engage", "good_play", "bad_trade")
        else (2 if f["time_known"] and f["time"] < 120 and f["type"] in ("death", "bad_trade") else 0),
    "early_game:pregame_principles": lambda f: 3 if f["phase"] == "pregame"
        else (1 if f["time_known"] and f["time"] < 90 and f["phase"] == "laning" else 0),
    "early_game:rune_contest_framework": lambda f: 3 if (f["phase"] == "pregame" and f["type"] == "death") or f["rc"] == "rune_contest_unfavorable"
        else (1 if f["phase"] == "pregame" else 0),
    # --- builtin vision ---------------------------------------------------
    "vision:basic_mechanics": lambda f: (2 if _vision_rc(f) else 1) if f["gank"] else 0,
    "vision:death_checklist": lambda f: 3 if f["gank"] else 2 if f["lane_death"] else (1 if f["type"] == "death" else 2 if f["type"] == "overextend" else 0),
    "vision:key_numbers": lambda f: 1 if f["gank"] and _vision_rc(f) else 0,
    "vision:laning_wards": lambda f: (2 if (f["support"] or f["role"] == "mid") else 1)
        if f["pick"] and f["phase"] == "laning" else (1 if f["type"] == "overextend" and f["phase"] == "laning" else 0),
    "vision:midlate_wards": lambda f: (2 if f["support"] else 1) if f["pick"] and f["phase"] in MIDLATE
        else (1 if f["type"] == "death" and f["support"] and f["phase"] in MIDLATE else 0),
    "vision:role_responsibilities": lambda f: 2 if f["gank"] else (1 if f["type"] in ("death", "overextend") else 0),
    "vision:safety_rules": lambda f: 3 if f["gank"] else 2 if f["lane_death"] else {"overextend": 2, "fight_engage": 1}.get(f["type"], 1 if f["type"] == "death" else 0),
    # --- learned: LLM-distilled role statistics ---------------------------
    "learned:distilled:00:pos3_high_cs_aggressive_farming": lambda f: ({"missed_cs": 3, "bad_trade": 1, "death": 1, "overextend": 1}.get(f["type"], 0))
        if f["role"] == "offlane" and f["phase"] == "laning" else 0,
    "learned:distilled:01:pos4_pos5_survival_low_deaths": lambda f: ((3 if f["pick"] else 2) if f["type"] == "death"
        else 2 if f["type"] in ("bad_trade", "overextend") else 0) if f["support"] else 0,
    "learned:distilled:02:pos3_mid_game_kill_participation": lambda f: ({"fight_engage": 2, "good_play": 2, "death": 1, "missed_cs": 1}.get(f["type"], 0))
        if f["role"] in ("carry", "offlane") and f["phase"] == "mid_game" else 0,
    "learned:distilled:03:pos1_gpm_focus_farming_pattern": lambda f: ((3 if f["phase"] == "mid_game" else 2) if f["type"] == "death"
        else 1 if f["type"] in ("missed_cs", "overextend") else 0) if f["role"] == "carry" else 0,
    "learned:distilled:04:pos2_warding_mid": lambda f: ((3 if (_vision_rc(f) and f["gank"]) else 2) if f["pick"]
        else 1 if f["type"] in ("death", "bad_trade", "overextend") else 0) if f["role"] == "mid" and f["phase"] in ("laning", "mid_game") else 0,
    "learned:distilled:05:item_timings_beastmaster": _self_only("beastmaster", lambda f: 2 if f["type"] == "missed_cs" else 1),
    "learned:distilled:06:item_timings_gyrocopter": _self_only("gyrocopter", lambda f: 2 if f["type"] == "missed_cs" else 1),
    "learned:distilled:07:pos4_obs_ward_efficiency": lambda f: (2 if f["role"] == "support_4" else 1)
        if f["support"] and f["pick"] and f["phase"] in MIDLATE else 0,
    # --- learned: community video tips (read one by one) ------------------
    "learned:video:youtube:QanzOBMmzBo:00": _self_only("faceless_void", lambda f: 2 if f["phase"] == "laning" else 0),
    "learned:video:youtube:QanzOBMmzBo:01": lambda f: (2 if f["support"] else 1)
        if f["phase"] == "laning" and f["type"] in ("bad_trade", "missed_cs", "overextend") else 0,
    "learned:video:youtube:QanzOBMmzBo:02": lambda f: 0,
    "learned:video:youtube:QanzOBMmzBo:03": lambda f: (2 if f["type"] == "death" and f["rc"] in ("greedy_farm", "isolated") else 1)
        if f["core"] and f["phase"] == "mid_game" and f["type"] in ("death", "missed_cs") else 0,
    "learned:video:youtube:QanzOBMmzBo:04": _self_only("faceless_void", lambda f: 1),
    "learned:video:youtube:Z-y1Hb2d20Y:00": _self_only("invoker", lambda f: ({"bad_trade": 2, "missed_cs": 2, "death": 1, "overextend": 1}.get(f["type"], 0))
        if f["phase"] in LANING else 0),
    "learned:video:youtube:Z-y1Hb2d20Y:01": _self_only("invoker", lambda f: (2 if f["type"] == "bad_trade" else 1 if f["type"] in ("missed_cs", "death", "overextend") else 0)
        if f["phase"] in LANING else 0),
    "learned:video:youtube:Z-y1Hb2d20Y:02": _self_only("invoker", lambda f: (2 if f["type"] == "missed_cs" else 1)
        if f["phase"] in ("laning", "mid_game") else 0),
    "learned:video:youtube:Z-y1Hb2d20Y:03": _self_only("invoker", lambda f: 2 if f["type"] == "missed_cs" else (1 if f["phase"] == "mid_game" else 0)),
    "learned:video:youtube:Z-y1Hb2d20Y:04": _self_only("invoker", lambda f: (2 if f["type"] == "death" and f["phase"] == "mid_game" and "hand_of_midas" in f["items"] else 1)),
    "learned:video:youtube:Z-y1Hb2d20Y:05": _self_only("invoker", lambda f: 2 if f["pick"] and f["phase"] in MIDLATE else 1 if f["type"] == "death" else 0),
    "learned:video:youtube:Z-y1Hb2d20Y:06": _self_only("invoker", lambda f: 2 if f["phase"] in MIDLATE and (f["tf"] or f["type"] == "fight_engage") else (1 if f["phase"] in MIDLATE else 0)),
}
for _role in ("carry", "mid", "offlane", "support_4", "support_5"):
    GENERIC[f"laning_principles:{_role}"] = _laning_principles(_role)
    for _phase in ("pregame", "laning", "mid_game", "late_game"):
        GENERIC[f"phase_objectives:{_phase}:{_role}"] = _phase_objectives(_phase, _role)
for _n, _role in ROLES.items():
    GENERIC[f"learned:pro_pattern:position_{_n}"] = _pro_pattern(_role)


# ---------------------------------------------------------------------------
# Hero / matchup / item chunks
# ---------------------------------------------------------------------------

def _hero_chunk_grade(chunk: dict, f: dict) -> int | None:
    cid = chunk["chunk_id"]
    tags = chunk["tags"]
    t, phase = f["type"], f["phase"]
    laning = phase in LANING
    heroes = [x.split(":", 1)[1] for x in tags if x.startswith("hero:") and x != "hero:all"]
    matchups = [x.split(":")[1:] for x in tags if x.startswith("matchup:")]
    items = [x.split(":", 1)[1] for x in tags if x.startswith("item:")]
    buyers = {x.split(":", 1)[1] for x in tags if x.startswith("buyer:")}

    if cid.startswith("hero_overview:"):
        r = _hero_role(heroes[0], f)
        return {"self": 2, "killer": 2, "near": 2, "near_bg": 1, "far": 1}.get(r, 0)
    if cid.startswith("hero_laning:"):
        r = _hero_role(heroes[0], f)
        if r == "self":
            if laning:
                return 3 if t in ("bad_trade", "missed_cs", "overextend", "death") else 2
            return 1 if phase == "mid_game" else 0
        if r in ("killer", "near"):
            return 2 if laning else (1 if phase == "mid_game" else 0)
        if r == "near_bg":
            return 1 if laning else 0
        return 0
    if cid.startswith("hero_abilities:"):
        r = _hero_role(heroes[0], f)
        if r == "self":
            return 1 if t == "missed_cs" else 2
        if r == "killer":
            return 3 if t == "death" else 2
        if r == "near":
            return 1 if t in ("missed_cs", "good_play") else 2
        if r == "near_bg":
            return 1
        if r == "far":
            return 0 if laning or not phase else 1
        return 0
    if matchups:
        a, b = matchups[0][0], matchups[0][1]
        ra, rb = _hero_role(a, f), _hero_role(b, f)
        stat = cid.startswith("learned:")
        if ra == "self" and rb in ("killer", "near", "near_bg"):
            g = 3 if (rb == "killer" or (laning and rb == "near")) else 2
            return min(g, 2) if stat else g
        if rb == "self" and ra in ("killer", "near", "near_bg"):
            g = 2 if (ra == "killer" or (laning and ra == "near")) else 1
            return min(g, 1) if stat else g
        if ra == "self" and rb == "far":
            return 1
        return 0
    if cid.startswith("item_counter:"):
        r = _hero_role(heroes[0], f) if heroes else "absent"
        owned = bool(items) and items[0] in f["items"]
        if r in ("killer", "near"):
            return 3 if owned else 2
        if r in ("far", "near_bg"):
            return 2 if owned else 1
        return 1 if r == "self" else 0
    if cid.startswith("learned:item_timing:"):
        r = _hero_role(heroes[0], f) if heroes else "absent"
        return (2 if t == "missed_cs" else 1) if r == "self" else 0
    if cid.startswith("learned:benchmarks:"):
        r = _hero_role(heroes[0], f) if heroes else "absent"
        return (3 if t == "missed_cs" else 1) if r == "self" else 0
    if cid.startswith("item:"):
        item = cid.split(":", 1)[1]
        return _item_grade(item, item in f["items"], f["self"] in buyers, f)
    return None


def _norm_id(cid: str) -> str:
    for suffix in (":zh", ":en"):
        if cid.startswith("learned:pro_pattern:") and cid.endswith(suffix):
            return cid[: -len(suffix)]
    return cid


_OVERRIDES: dict | None = None


def _overrides() -> dict:
    global _OVERRIDES
    if _OVERRIDES is None:
        path = os.path.join(HERE, "overrides.json")
        _OVERRIDES = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    return _OVERRIDES


def grade(q: dict, chunk: dict, use_overrides: bool = True) -> int:
    """Relevance grade (0-3) of *chunk* for the situation *q* (a query record)."""
    cid = _norm_id(chunk["chunk_id"])
    if use_overrides:
        ov = _overrides().get(q["qid"], {})
        if cid in ov:
            return int(ov[cid])
    f = features(q)
    if cid in GENERIC:
        return int(GENERIC[cid](f))
    g = _hero_chunk_grade(chunk, f)
    if g is None:
        raise KeyError(f"no rubric rule for chunk {cid}")
    return int(g)


def _main() -> int:
    """Compare (or rewrite) labels.json against the rubric for the corpus as it is built today."""
    import sys

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
    sys.path.insert(0, root)
    os.environ.setdefault("DEEPSEEK_API_KEY", "")
    os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
    os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")
    from backend.app.ai.rag import evalkit

    write = "--write" in sys.argv[1:]
    bench = evalkit.load_benchmark(HERE)
    stores = evalkit.build_stores(bench)
    labels: dict[str, dict[str, int]] = {}
    without_rule: set[str] = set()
    differing = 0
    for q in bench.queries:
        graded: dict[str, int] = {}
        for chunk in stores[q["lang"]].chunks():
            try:
                g = grade(q, {"chunk_id": chunk.chunk_id, "tags": sorted(chunk.tags)})
            except KeyError:
                without_rule.add(chunk.chunk_id)
                continue
            if g >= 1:
                graded[chunk.chunk_id] = g
        labels[q["qid"]] = dict(sorted(graded.items()))
        known = bench.corpus_ids[q["lang"]]
        stored = bench.labels[q["qid"]]
        live = {c.chunk_id for c in stores[q["lang"]].chunks()}
        differing += sum(1 for cid in known & live if stored.get(cid, 0) != graded.get(cid, 0))
    new_ids = {lang: sorted(c.chunk_id for c in stores[lang].chunks()) for lang in ("zh", "en")}
    added = sum(len(set(new_ids[lang]) - bench.corpus_ids[lang]) for lang in new_ids)
    removed = sum(len(bench.corpus_ids[lang] - set(new_ids[lang])) for lang in new_ids)
    print(f"queries {len(bench.queries)}; labels differing from the rubric: {differing}; "
          f"chunks added since labelling: {added}, removed: {removed}; without a rule: {len(without_rule)}")
    if without_rule:
        print("  no rubric rule (add to GENERIC):", sorted(without_rule)[:10])
    if not write:
        return 1 if differing or without_rule else 0
    if without_rule:
        print("not written: every chunk needs a rule first")
        return 1
    with open(os.path.join(HERE, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    with open(os.path.join(HERE, "corpus.json"), "r", encoding="utf-8") as f:
        corpus = json.load(f)
    corpus["ids"] = new_ids
    with open(os.path.join(HERE, "corpus.json"), "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print("labels.json and corpus.json written")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
