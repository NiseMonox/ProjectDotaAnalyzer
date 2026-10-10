"""Situation queries: the closed set of English queries the dense ranker uses.

A decision is described by ONE entity-free English string built from
structured fields only (never from the brief, which is in the report
language and names heroes)::

    "<role>, <phase>: <topic>"

6 roles x 5 phases x 7 topic texts = 210 strings, plus the rune-fight topic
of the pregame phase for the 6 roles = 216. Because the set is closed, its
vectors are built offline together with the chunk vectors and the server
looks a query vector up by its text: no embedding model runs in a request.
The same string is used for a Chinese and an English report, so both retrieve
the same principles.

The texts are part of the vector keys: changing one means rebuilding the
vector files (``python -m backend.scripts.rag build --static``).
"""
from __future__ import annotations

import re

from .types import SituationQuery

# Position number -> role phrase; DEFAULT_ROLE when the position is unknown.
ROLES: dict[int, str] = {
    1: "position 1 carry",
    2: "position 2 mid",
    3: "position 3 offlane",
    4: "position 4 support",
    5: "position 5 hard support",
}
DEFAULT_ROLE = "core"

# Game phase code -> phrase; DEFAULT_PHASE when the phase is unknown.
PHASES: dict[str, str] = {
    "pregame": "pregame",
    "laning": "laning phase",
    "mid_game": "mid game",
    "late_game": "late game",
}
DEFAULT_PHASE = "game"

# Topic code -> what knowledge the situation asks for.
TOPICS: dict[str, str] = {
    "death_pick": ("how to avoid getting picked off: vision, map awareness, missing enemies, "
                   "positioning, escape abilities and defensive items"),
    "death_tf": ("teamfight positioning, target priority, when to join or disengage, "
                 "and whether the death was a good trade"),
    "bad_trade": "lane trading, harassing and HP / regeneration management in lane",
    "missed_cs": "last hitting, creep equilibrium, farming efficiency and CS benchmarks",
    "overextend": "staying in lane on low HP: when to retreat, use regeneration and avoid a dive",
    "fight_engage": "teamfight execution: initiation, positioning, ability and item usage",
    "good_play": "teamfight execution: initiation, positioning, ability and item usage",
    "good_play_smoke": "smoke ganks, coordinated pick-offs and converting them into objectives",
}

# Before the horn every decision is about the rune fight, whatever its point
# type: one topic, only ever combined with the pregame phase. (With the
# teamfight-death topic the dense term ranked the death-analysis sections
# above the rune-contest and level-1 principles of a pregame death.)
RUNE_FIGHT = "rune_fight"
RUNE_FIGHT_PHASE = "pregame"
RUNE_FIGHT_TOPIC = ("the bounty rune contest and level 1 fights: counting enemy heroes before entering "
                    "the river, level 1 fight strength and when to give up the rune")

# Topic argument of the agent's knowledge search -> (query tags, topic code or
# "" when the topic has no situation query). The tags are the ones the chunk
# producers use (see knowledge_store.normalize_tag).
SEARCH_TOPICS: dict[str, tuple[tuple[str, ...], str]] = {
    "matchup": ((), ""),
    "laning": (("mechanic:laning", "mechanic:trading", "mechanic:cs", "situation:bad_trade"), "bad_trade"),
    "teamfight": (("mechanic:teamfight", "mechanic:positioning", "situation:fight_engage"), "death_tf"),
    "vision": (("mechanic:vision", "mechanic:ward", "mechanic:map_awareness", "situation:gank",
                "situation:danger"), "death_pick"),
    "items": (("mechanic:item_usage",), ""),
    "farming": (("mechanic:farming", "mechanic:cs", "situation:missed_cs"), "missed_cs"),
    "objectives": (("mechanic:roshan", "mechanic:smoke", "mechanic:gank", "mechanic:split_push",
                    "mechanic:highground", "mechanic:buyback"), "good_play_smoke"),
    "general": ((), ""),
}

# Words of a free-text question -> tags of the principles that answer it. Used
# by the keyword search (no embedding model): there the question's own words
# are all that says what it is about. Keys are lowercase words or two-word
# phrases as a coach writes them; the tags are the ones the built-in chunk
# producers use (a test fails when one of them is no longer in the store).
QUESTION_TOPICS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("smoke", "smokes", "smoked"), ("mechanic:smoke", "mechanic:gank")),
    (("gank", "ganks", "ganked", "ganking", "pickoff", "pickoffs", "pick off", "picked off"),
     ("mechanic:gank", "situation:gank", "mechanic:map_awareness")),
    (("ward", "wards", "warding", "observer", "observers", "sentry", "sentries", "deward", "dewarding",
      "vision"), ("mechanic:ward", "mechanic:vision")),
    (("teamfight", "teamfights", "teamfighting", "team fight", "team fights"),
     ("mechanic:teamfight", "situation:fight_engage")),
    (("positioning",), ("mechanic:positioning",)),
    (("buyback", "buybacks", "buy back"), ("mechanic:buyback",)),
    (("roshan", "rosh", "aegis"), ("mechanic:roshan",)),
    (("splitpush", "split push", "split pushing"), ("mechanic:split_push",)),
    (("highground", "high ground"), ("mechanic:highground",)),
    (("rune", "runes", "bounty"), ("mechanic:rune",)),
    (("trade", "trades", "trading", "harass", "harassing"), ("mechanic:trading", "situation:bad_trade")),
    (("cs", "lasthit", "last hit", "last hits", "last hitting", "creep score"),
     ("mechanic:cs", "situation:missed_cs")),
    (("farm", "farms", "farming"), ("mechanic:farming", "mechanic:cs")),
    (("stack", "stacks", "stacking", "pull", "pulls", "pulling"), ("mechanic:stack_pull",)),
    (("dive", "dives", "diving", "tower aggro"), ("mechanic:tower_aggro",)),
    (("tp", "teleport", "town portal"), ("mechanic:tp_scroll",)),
    (("comeback", "come back", "behind"), ("mechanic:comeback", "situation:behind")),
    (("minimap", "map awareness"), ("mechanic:map_awareness",)),
    (("lane", "laning"), ("mechanic:laning",)),
)
_QUESTION_WORD_RE = re.compile(r"[a-z]+")


def question_topics(query: str) -> list[str]:
    """Tags of the topics a free-text question names (:data:`QUESTION_TOPICS`), in table order."""
    words = _QUESTION_WORD_RE.findall(str(query or "").lower())
    phrases = set(words) | {f"{a} {b}" for a, b in zip(words, words[1:])}
    tags: list[str] = []
    for keywords, topic_tags in QUESTION_TOPICS:
        if phrases.intersection(keywords):
            tags.extend(tag for tag in topic_tags if tag not in tags)
    return tags


def situation_text(position, phase, kind: str) -> str:
    """The query string for a role / phase / topic code ("" for an unknown topic).

    The rune-fight topic belongs to the pregame phase: *phase* is not read for it.
    """
    if kind == RUNE_FIGHT:
        topic, phase = RUNE_FIGHT_TOPIC, RUNE_FIGHT_PHASE
    else:
        topic = TOPICS.get(kind)
    if not topic:
        return ""
    try:
        role = ROLES.get(int(position), DEFAULT_ROLE) if position is not None else DEFAULT_ROLE
    except (TypeError, ValueError):
        role = DEFAULT_ROLE
    return f"{role}, {PHASES.get(str(phase or ''), DEFAULT_PHASE)}: {topic}"


def situation_queries() -> list[str]:
    """Every string :func:`situation_text` can return (sorted, without duplicates)."""
    roles = [*ROLES.values(), DEFAULT_ROLE]
    phases = [*PHASES.values(), DEFAULT_PHASE]
    queries = {f"{role}, {phase}: {topic}"
               for role in roles for phase in phases for topic in TOPICS.values()}
    queries.update(f"{role}, {PHASES[RUNE_FIGHT_PHASE]}: {RUNE_FIGHT_TOPIC}" for role in roles)
    return sorted(queries)


def topic_kind(ctx) -> str:
    """Topic code of a decision context ("" when the point type has none)."""
    dp = ctx.decision_point
    kind = str(getattr(dp, "point_type", "") or "")
    if kind == "death":
        death = getattr(dp, "related_death", None)
        kind = "death_tf" if getattr(death, "is_teamfight", False) else "death_pick"
    elif kind == "good_play" and getattr(dp, "subtype", "") == "smoke":
        kind = "good_play_smoke"
    if kind not in TOPICS:
        return ""
    return RUNE_FIGHT if getattr(ctx, "game_phase", "") == RUNE_FIGHT_PHASE else kind


def describe(ctx) -> SituationQuery | None:
    """Situation query of a :class:`DecisionContext`, or None for an unknown point type."""
    kind = topic_kind(ctx)
    if not kind:
        return None
    position = (getattr(ctx, "match_context", None) or {}).get("player_position")
    return SituationQuery(kind=kind, text=situation_text(position, getattr(ctx, "game_phase", ""), kind))


__all__ = ["DEFAULT_PHASE", "DEFAULT_ROLE", "PHASES", "QUESTION_TOPICS", "ROLES", "RUNE_FIGHT",
           "RUNE_FIGHT_TOPIC", "SEARCH_TOPICS", "TOPICS", "describe", "question_topics",
           "situation_queries", "situation_text", "topic_kind"]
