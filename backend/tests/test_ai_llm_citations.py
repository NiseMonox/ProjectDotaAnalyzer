"""Knowledge citations and the agent's knowledge search (report schema 4).

Retrieved knowledge reaches a prompt as numbered entries (``[K1]`` ...). Both
LLM paths may cite the entries they relied on; the report then lists them as
``sources`` of the evaluation. Covered here, against the strict fake endpoint
of ``test_ai_pipeline_helpers``: handles in, sources out; unknown or forged
handles; handles written into the text; the shared numbering and the limits
of the agent's free-text search; and that nothing of this ever costs a
verdict. Entries are built by ``knowledge_block`` so the tests do not depend
on what the knowledge base holds; two tests at the end use the real one.
"""
from __future__ import annotations

import os

os.environ.setdefault("AI_COACH_AUTO_LEARN", "0")
os.environ.setdefault("GAMEDATA_AUTO_UPDATE", "0")

import asyncio
import json
import re

import pytest

from backend.app.ai import agent, coach, context_builder, evaluator, knowledge_base
from backend.app.ai.models import MAX_SOURCES, DeathEvent, DecisionPoint, sources_payload
from backend.app.ai.rag.types import SOURCE_TYPES, KnowledgeRef
from test_ai_pipeline_helpers import (
    ANALYSIS_EN,
    ANALYSIS_ZH,
    PHASE_JSON,
    TEXT_VERDICT_EN,
    argument_violations,
    default_players,
    install_fake,
    is_phase_prompt,
    knowledge_block,
    knowledge_refs,
    llm_env,  # noqa: F401  (fixture)
    make_match,
    response,
    schema_violations,
    submit,
    teamfight_death,
    tool_call,
)

HANDLE_RE = re.compile(r"[\[［【(（]\s*[Kk]\d")
# A handle in any spelling, bracketed or not (the frontend is promised that no string holds one).
BARE_HANDLE_RE = re.compile(r"(?<![A-Za-z0-9])[KkＫКк]\s?-?\d{1,4}(?![A-Za-z0-9])")


def death_context(lang: str = "en"):
    players = default_players()
    tf = teamfight_death(1, "Puck", 129, "Lina", start=1500, x=1000.0, y=1000.0,
                         enemies=[(128, "Phantom Assassin", 1200.0, 1100.0)])
    match = make_match(players, teamfights=[tf])
    death = DeathEvent(time=1505, player_slot=1, hero_name="Puck", killer_hero="Lina",
                       x=1000.0, y=1000.0, is_teamfight=True, teamfight_idx=0)
    brief = "Killed by Lina in a teamfight" if lang == "en" else "团战中被Lina击杀"
    dp = DecisionPoint(time=1505, point_type="death", player_slot=1, hero_name="Puck",
                       severity="critical", brief=brief, related_death=death)
    return match, context_builder.build_context(match, dp, lang=lang)


def _single(ctx, knowledge, lang="en"):
    return asyncio.run(evaluator.evaluate_decision(ctx, knowledge, lang=lang))


def _agent(match, ctx, knowledge, lang="en"):
    return asyncio.run(agent.run_decision_agent(match, ctx, knowledge, lang=lang))


def _tool_results(call: dict) -> dict[str, str]:
    return {m["tool_call_id"]: m["content"] for m in call["messages"] if m.get("role") == "tool"}


def _ids(sources: list[dict], cited: bool | None = None) -> list[str]:
    return [s["id"] for s in sources if cited is None or s["cited"] is cited]


def _strings(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


# ---------------------------------------------------------------------------
# Handles in text and in a references value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text, cleaned, cited", [
    ("Lina had Laguna Blade ready [K2], so backing off was right [K1][K3].",
     "Lina had Laguna Blade ready, so backing off was right.", ["K2", "K1", "K3"]),
    ("根据参考资料[K2]，Lina 的大招是关键【K12、K3】。", "根据参考资料，Lina 的大招是关键。", ["K2", "K12", "K3"]),
    # ... with the words that only introduce them
    ("Entries [K1, K4] and [k2-K5] agree (see [K9]).", "agree.", ["K1", "K4", "K2", "K5", "K9"]),
    ("Vision was the issue [K1], as [K3] explains.", "Vision was the issue.", ["K1", "K3"]),
    # Every other way a model writes a handle (11 of 25 adversarial answers left one in a field)
    ("He walked forward while Tidehunter was missing. (see K1)", "He walked forward while Tidehunter was missing.", ["K1"]),
    ("As K2 says, he stepped past the river ward.", "he stepped past the river ward.", ["K2"]),
    ("Play one screen further back. K3", "Play one screen further back.", ["K3"]),
    ("Tidehunter 不在视野内（见K1）。", "Tidehunter 不在视野内。", ["K1"]),
    ("正如K2所说，越过了河道眼位。", "越过了河道眼位。", ["K2"]),
    ("Void Spirit jumped him. [K 1]", "Void Spirit jumped him.", ["K1"]),
    ("Play further back. [K-2] [#K3] [K1][K2]", "Play further back.", ["K2", "K3", "K1"]),
    ("Void Spirit jumped him. [Ｋ1]", "Void Spirit jumped him.", ["K1"]),                 # full-width K
    ("Void Spirit jumped him. [K1234]", "Void Spirit jumped him.", ["K1234"]),
    ("Play further back. [K1, 2]", "Play further back.", ["K1", "K2"]),
    ("Void Spirit jumped him.[^K1] Play further back. ^K2", "Void Spirit jumped him. Play further back.", ["K1", "K2"]),
    # A references line under a label the parser does not know, also on the line of the text itself
    ("Play further back.\nSources: K1 K2", "Play further back.", ["K1", "K2"]),
    ("Play further back.\nCited entries: K1, K2", "Play further back.", ["K1", "K2"]),
    ("Play further back. References used: K1", "Play further back.", ["K1"]),
    ("站位应当后撤一屏。\n引用的条目：K1 K2", "站位应当后撤一屏。", ["K1", "K2"]),
    ("站位应当后撤一屏。 参考: K1、K2", "站位应当后撤一屏。", ["K1", "K2"]),
    ("Play further back.\nSources: None", "Play further back.", []),
    ("站位应当后撤一屏。\n引用编号：无", "站位应当后撤一屏。", []),
    # "(K7)" is how the knowledge layer writes a look-alike found inside an entry: removed, never a citation.
    ("The tip (K7) says to wait.", "The tip says to wait.", []),
    ("K/D/A 3/2/1 with BKB [B] and a [KDA] note, 2K gold.", "K/D/A 3/2/1 with BKB [B] and a [KDA] note, 2K gold.", []),
    # Ordinary text with a K, a colon or the word "none" is left alone
    ("He had 20k gold and a 5K lead; Level 6: Tidehunter has Ravage.",
     "He had 20k gold and a 5K lead; Level 6: Tidehunter has Ravage.", []),
    ("Remember: None of this works without vision. Key takeaway: none.",
     "Remember: None of this works without vision. Key takeaway: none.", []),
    ("见好就收：不要贪。According to the minimap, three heroes were missing.",
     "见好就收：不要贪。According to the minimap, three heroes were missing.", []),
    ("", "", []),
])
def test_handles_are_removed_from_text_and_counted(text, cleaned, cited):
    assert evaluator.split_handles(text) == (cleaned, cited)
    assert evaluator.strip_handles(text) == cleaned and not HANDLE_RE.search(cleaned)
    assert not BARE_HANDLE_RE.search(cleaned)


@pytest.mark.parametrize("text", [
    "[" + "K1, " * 5000 + "x",                      # a list that never closes
    "【" + "K12、" * 20000,                          # "K12" must not also read as "K1" + "2": 2^20000 ways
    "(" + "K1 " * 20000,
    "[K1" + " and" * 20000 + "]",
    "K" + "1" * 100000,
    "[" * 50000 + "K1" + "]" * 50000,
    "x " * 30000 + "\nReferences: " + "K1 " * 5000,
    "see " * 20000 + "K1",
    "a: " * 20000 + ". " * 20000 + "Refs: K1",
])
def test_handle_sweep_stays_linear_on_hostile_text(text):
    """A model's answer is untrusted input to these patterns."""
    import time

    started = time.perf_counter()
    evaluator.split_handles(text)
    evaluator.cited_handles(text)
    assert time.perf_counter() - started < 2.0


@pytest.mark.parametrize("value, expected", [
    ("K2 K5", ["K2", "K5"]),
    ("[K2], [K5] and k9, K2 again", ["K2", "K5", "K9"]),
    (["K2", "k5", "[K7]"], ["K2", "K5", "K7"]),
    ([2, "5", True, None, {"ref": "K9"}], ["K2", "K5"]),      # numbers of a list are handles, junk is skipped
    ("None", []), ("无", []), ("see above", []), ("2, 5", []), (None, []), (7, []), ({"K1": True}, []),
    ("BKB, MKB and 2K gold", []),
])
def test_reference_values_models_write(value, expected):
    assert evaluator.cited_handles(value) == expected


def test_sources_list_cited_entries_first_and_at_most_four():
    refs = knowledge_refs(6)
    sources = sources_payload(refs, ["K6", "K2", "K6", "K99"])
    assert len(sources) == MAX_SOURCES == 4
    assert [(s["id"], s["cited"]) for s in sources] == [
        ("test:entry_6", True), ("test:entry_2", True), ("test:entry_1", False), ("test:entry_3", False)]
    assert all(set(s) == {"id", "title", "source_type", "patch", "cited"} for s in sources)
    assert sources[2]["patch"] == "7.41" and sources[0]["patch"] == ""
    assert all(s["source_type"] in SOURCE_TYPES for s in sources)
    # The same chunk under two handles (block and search) is one source.
    again = KnowledgeRef(ref="K9", chunk_id="test:entry_1", title="Entry 1", source_type="editorial",
                         source_label="Curated note")
    twice = sources_payload(refs[:2] + [again], ["K9"])
    assert [(s["id"], s["cited"]) for s in twice] == [("test:entry_1", True), ("test:entry_2", False)]
    assert sources_payload([], ["K1"]) == [] and sources_payload(refs, limit=3)[-1]["id"] == "test:entry_3"


def test_capped_block_is_cut_between_entries_and_cut_entries_cannot_be_cited():
    """A block over the cap used to be sliced mid-entry: half a tip under a citable handle."""
    block = knowledge_block(12, body="word " * 240)        # about 1.2 k characters per entry
    assert len(block) > evaluator.MAX_KNOWLEDGE_CHARS
    text = evaluator.format_knowledge(block, "en")
    assert len(text) <= evaluator.MAX_KNOWLEDGE_CHARS + 20
    assert text.endswith("…\n</reference>") and text.count("<reference>") == text.count("</reference>") == 1
    shown = re.findall(r"(?m)^\[(K\d+)\] ", text)
    assert shown == [f"K{n}" for n in range(1, len(shown) + 1)] and 4 <= len(shown) < 12
    for entry in text.split("\n\n")[:-1]:
        assert entry.rstrip().endswith("word"), entry[-40:]                   # every entry is whole
    assert text.split("\n\n")[-1].split("\n…")[0].rstrip().endswith("word")
    refs = evaluator.knowledge_refs(block, text)
    assert [r.ref for r in refs] == shown
    # A plain string (a test double, the legacy assembly) has no entries at all.
    assert evaluator.knowledge_refs("REFERENCE [K1] TIPS") == [] and evaluator.knowledge_refs("") == []
    # Text without entries is still cut where the cap is.
    assert len(evaluator.format_knowledge("x" * 30000, "en")) < evaluator.MAX_KNOWLEDGE_CHARS + 50


# ---------------------------------------------------------------------------
# Single prompt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang", ["en", "zh"])
def test_single_prompt_explains_handles_only_when_there_are_entries(lang):
    _match, ctx = death_context(lang)
    t = evaluator._t(lang)
    with_entries = evaluator._build_evaluation_prompt(ctx, knowledge_block(3), lang=lang)
    assert t["citation_rule"] in with_entries and t["response_references"] in with_entries
    assert "[K1] Curated note — Entry 1" in with_entries
    # The references line is the last line of the answer format, after the lesson.
    assert with_entries.index(t["response_references"]) > with_entries.index(t["response_format"].splitlines()[-1])
    # Says what the handles are, that citing is optional, and where they must not appear.
    rule = t["citation_rule"]
    for needle in (("handles", "optional", "really relied", "Do not write the handles inside")
                   if lang == "en" else ("代号", "可选", "确实用到", "正文里不要出现")):
        assert needle in rule
    for knowledge in ("REFERENCE TIPS", ""):
        plain = evaluator._build_evaluation_prompt(ctx, knowledge, lang=lang)
        assert t["citation_rule"] not in plain and t["response_references"] not in plain


def test_single_prompt_citations_become_sources(monkeypatch, llm_env):
    monkeypatch.setenv("AI_COACH_AGENT", "0")

    async def handler(kwargs):
        return response(
            "Score: C\n"
            "Analysis: Puck walked into Lina's Laguna Blade range without Phase Shift ready [K3], "
            "and the team was too far away to follow up.\n"
            "Mistake: Stood in front of the team (K1).\n"
            "Better Play: Wait behind the frontline until Lina has used her stun.\n"
            "Lesson: Respect burst damage.\n"
            "References: K2, K9")

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = asyncio.run(coach._safe_evaluate(ctx, knowledge_block(5), lang="en", match=match))
    assert ev.used_llm and not ev.degraded and ev.score == "C"
    assert len(fake.calls) == 1                                  # an unknown handle is no reason to ask again
    # K2 from the references line, K3 from the text; K9 was never shown; "(K1)" is not a citation.
    assert ev.cited_refs == ["K2", "K3"]
    assert [(s["id"], s["cited"]) for s in ev.sources] == [
        ("test:entry_2", True), ("test:entry_3", True), ("test:entry_1", False), ("test:entry_4", False)]
    for text in (ev.analysis, ev.mistake, ev.better_play, ev.lesson):
        assert not HANDLE_RE.search(text) and "References" not in text
    assert ev.analysis.startswith("Puck walked into Lina's Laguna Blade range without Phase Shift ready, and")
    assert ev.mistake == "Stood in front of the team." and ev.lesson == "Respect burst damage."


@pytest.mark.parametrize("line, cited", [
    ("引用: K1、K3", ["test:entry_1", "test:entry_3"]),
    ("引用：[K3]", ["test:entry_3"]),
    ("引用: 无", []),
    ("", []),                                  # the line is optional
    ("**引用:** K2", ["test:entry_2"]),
    # The label of the other language, or a near miss: still the references line, never part of the lesson.
    ("References: K2", ["test:entry_2"]),
    ("引用条目：K1", ["test:entry_1"]),
    ("Citations: none", []),
])
def test_chinese_references_line(monkeypatch, llm_env, line, cited):
    monkeypatch.setenv("AI_COACH_AGENT", "0")

    async def handler(kwargs):
        return response(f"评分: C\n分析: {ANALYSIS_ZH}\n错误: 独自走进迷雾。\n更好的做法: 等队友一起行动。\n"
                        f"教训: 敌人不在视野内时要保守。\n{line}")

    fake = install_fake(monkeypatch, handler)
    _match, ctx = death_context("zh")
    ev = _single(ctx, knowledge_block(3), lang="zh")
    assert len(fake.calls) == 1 and ev.lesson == "敌人不在视野内时要保守。"
    assert _ids(ev.sources, cited=True) == cited and len(ev.sources) == 3


_ADVERSARIAL = {
    "en": ("Score: C\nAnalysis: {a}\nMistake: {m}\nBetter Play: {b}\nLesson: {l}{tail}",
           "Puck walked forward on the bottom river while Lina was out of sight and Phantom Assassin jumped him.",
           "He stepped past the river ward without knowing where Lina was.",
           "Hold the wave under the tower until the missing hero shows on the minimap.",
           "A missing initiator is a reason to play one screen further back."),
    "zh": ("评分: C\n分析: {a}\n错误: {m}\n更好的做法: {b}\n教训: {l}{tail}",
           "Puck 在下路河道继续前压时 Lina 不在视野内，随后被 Phantom Assassin 先手击杀。",
           "不知道 Lina 位置的情况下越过了河道眼位。",
           "在敌方先手英雄露头之前，把兵线控在塔下。",
           "敌方先手英雄消失时，站位应当后撤一屏。"),
}


def _adversarial_answers(lang: str) -> dict[str, tuple[str, list[str]]]:
    """``{name: (answer text, entries it cites)}``: the handle forms that used to stay in a field."""
    template, a, m, b, l = _ADVERSARIAL[lang]
    en = lang == "en"

    def mk(a=a, m=m, b=b, l=l, tail=""):
        return template.format(a=a, m=m, b=b, l=l, tail=tail)

    one, two, three = "test:entry_1", "test:entry_2", "test:entry_3"
    return {
        "bare handles": (mk(a=a + (" (see K1)" if en else "（见K1）"),
                            m=("As K2 says, " + m) if en else ("正如K2所说，" + m), l=l + " K3"), [one, two, three]),
        "spaced and marked": (mk(a=a + " [K 1]", l=l + " [K-2] [#K3]"), [one, two, three]),
        "full-width K": (mk(a=a + " [Ｋ1]"), [one]),
        "four digits": (mk(a=a + " [K1234]"), []),
        "number in a list": (mk(l=l + " [K1, 2]"), [one, two]),
        "footnotes": (mk(a=a + "[^K1]", l=l + " ^K2"), [one, two]),
        "label Sources": (mk(tail="\nSources: K1 K2" if en else "\n引用资料: K1 K2"), [one, two]),
        "label Refs": (mk(tail="\nRefs: K1 K2" if en else "\n参考: K1、K2"), [one, two]),
        "label Cited entries": (mk(tail="\nCited entries: K1, K2" if en else "\n引用编号：K1、K2"), [one, two]),
        "label References used": (mk(tail="\nReferences used: K1, K2" if en else "\n引用的条目：K1 K2"), [one, two]),
        "on the lesson's line": (mk(l=l + (" References: K1" if en else " 引用: K1")), [one]),
    }


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("name", sorted(_adversarial_answers("en")))
def test_no_way_of_writing_a_handle_reaches_the_player(monkeypatch, llm_env, lang, name):
    """11 of 25 adversarial answers left handle text in a field, in both languages."""
    monkeypatch.setenv("AI_COACH_AGENT", "0")
    text, cited = _adversarial_answers(lang)[name]
    _template, a, m, b, l = _ADVERSARIAL[lang]

    async def handler(kwargs):
        return response(text)

    fake = install_fake(monkeypatch, handler)
    _match, ctx = death_context(lang)
    ev = _single(ctx, knowledge_block(3), lang=lang)
    assert len(fake.calls) == 1 and ev.used_llm and not ev.degraded
    # The verdict is the text the model wrote, minus the handles and the words that introduce them
    assert (ev.analysis, ev.mistake, ev.better_play, ev.lesson) == (a, m, b, l)
    assert sorted(_ids(ev.sources, cited=True)) == cited and len(ev.sources) == 3
    for value in (ev.analysis, ev.mistake, ev.better_play, ev.lesson, *(s["title"] for s in ev.sources)):
        assert not BARE_HANDLE_RE.search(value) and not HANDLE_RE.search(value), value


@pytest.mark.parametrize("references", [
    "References: all of the above were useful",
    "References: K999 K0",
    "References: [the handles of the reference entries you actually used]",
    "References:",
])
def test_a_useless_references_line_never_costs_the_verdict(monkeypatch, llm_env, references):
    monkeypatch.setenv("AI_COACH_AGENT", "0")

    async def handler(kwargs):
        return response(f"{TEXT_VERDICT_EN}\n{references}")

    fake = install_fake(monkeypatch, handler)
    _match, ctx = death_context()
    ev = _single(ctx, knowledge_block(3))
    assert ev.used_llm and ev.analysis == ANALYSIS_EN and ev.lesson == "Respect the map."
    assert len(fake.calls) == 1 and ev.cited_refs == []
    assert _ids(ev.sources) == ["test:entry_1", "test:entry_2", "test:entry_3"] and not _ids(ev.sources, cited=True)


@pytest.mark.parametrize("lang, cap", [("en", "at most the three entries"), ("zh", "最关键的三条")])
def test_both_llm_paths_ask_for_the_few_entries_the_advice_rests_on(lang, cap):
    """Live runs cited three to five of eleven entries in every verdict while nothing bounded the list."""
    assert cap in evaluator._t(lang)["citation_rule"]
    assert cap in agent._system_prompt(lang, True) and cap in agent._system_prompt(lang, False)


def test_citations_without_entries_are_ignored(monkeypatch, llm_env):
    """A plain knowledge text shows no entry, so nothing can be cited (and nothing is listed)."""
    monkeypatch.setenv("AI_COACH_AGENT", "0")

    async def handler(kwargs):
        return response(f"{TEXT_VERDICT_EN}\nReferences: K1 K2")

    install_fake(monkeypatch, handler)
    _match, ctx = death_context()
    ev = _single(ctx, "REFERENCE TIPS")
    assert ev.used_llm and ev.cited_refs == [] and ev.sources == []


# ---------------------------------------------------------------------------
# Agent: schema
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang", ["en", "zh"])
def test_knowledge_tool_takes_a_search_and_the_verdict_takes_references(lang):
    tools = {t["function"]["name"]: t["function"] for t in agent._tools_schema(lang, True)}
    knowledge = tools["retrieve_coaching_knowledge"]
    params = knowledge["parameters"]
    assert set(params["properties"]) == {"query", "heroes", "items", "topic", "top_k"}
    assert "required" not in params and params["additionalProperties"] is False     # every argument is optional
    assert params["properties"]["topic"]["enum"] == [
        "matchup", "laning", "teamfight", "vision", "items", "farming", "objectives", "general"]
    assert all(not schema_violations(t["parameters"]) for t in tools.values())
    # The description tells the model when a search is worth a call, and how many it has.
    text = knowledge["description"]
    for needle in (("Without arguments", "only when the selected entries do not cover", "At most 3 searches", "[K1]")
                   if lang == "en" else ("不带参数", "没有覆盖", "最多检索3次", "[K1]")):
        assert needle in text, needle
    # Arguments a model may send, checked against the published schema.
    for args in ({}, {"query": "How does Puck survive Lina's burst?"},
                 {"query": "Blink timing", "heroes": ["Lina"], "items": ["Blink Dagger"], "topic": "items", "top_k": 6}):
        assert argument_violations(args, params) == []
    assert argument_violations({"topic": "jungle"}, params) and argument_violations({"limit": 3}, params)

    submit_params = tools["submit_evaluation"]["parameters"]
    assert submit_params["properties"]["references"] == {
        "type": "array", "items": {"type": "string"},
        "description": agent._TOOL_DESCRIPTIONS[lang]["p_references"]}
    assert "references" not in submit_params["required"]                           # citing is optional
    assert argument_violations({"score": "B", "analysis": "a", "mistake": "b", "better_play": "c",
                                "lesson": "d", "references": ["K1"]}, submit_params) == []
    prompt = agent._system_prompt(lang, True)
    for needle in (("only handles", "optional", "Never write a [K#] handle inside")
                   if lang == "en" else ("编号只是代号", "可选", "正文里不要出现 [K#]")):
        assert needle in prompt, needle


def test_search_arguments_disappear_when_searching_is_switched_off(monkeypatch):
    monkeypatch.setenv("RAG_AGENT_SEARCH_LIMIT", "0")
    tools = {t["function"]["name"]: t["function"] for t in agent._tools_schema("en", True)}
    assert tools["retrieve_coaching_knowledge"]["parameters"] == {
        "type": "object", "properties": {}, "additionalProperties": False}
    assert "search" not in tools["retrieve_coaching_knowledge"]["description"].lower()
    # A model that searches anyway gets the block of the decision.
    match, ctx = death_context()
    tc = agent._ToolCtx(match=match, context=ctx, knowledge=knowledge_block(2), lang="en")
    assert "[K2] Curated note — Entry 2" in agent._t_knowledge(tc, {"query": "anything"}) and tc.searches == 0


# ---------------------------------------------------------------------------
# Agent: citations
# ---------------------------------------------------------------------------

def test_agent_citations_are_checked_against_what_the_tools_returned(monkeypatch, llm_env):
    async def handler(kwargs):
        if len(fake.calls) == 1:
            return response(None, [tool_call("retrieve_coaching_knowledge", call_id="k1")])
        return submit("C", "Puck stood in front of the team [K1] and was burst by Lina before Phase Shift came up.",
                      mistake="Positioned ahead of the frontline 【K3】.",
                      references=["K2", "K7", "k2", "[K1]"])

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    block = knowledge_block(3)
    ev = _agent(match, ctx, block)
    assert _tool_results(fake.calls[1])["k1"] == str(block)       # the block arrives whole, entries numbered
    # references first (in order, repeats merged), then the handles found in the text; K7 was never shown.
    assert ev.cited_refs == ["K2", "K1", "K3"]
    assert [(s["id"], s["cited"]) for s in ev.sources] == [
        ("test:entry_2", True), ("test:entry_1", True), ("test:entry_3", True)]
    assert ev.analysis == "Puck stood in front of the team and was burst by Lina before Phase Shift came up."
    assert ev.mistake == "Positioned ahead of the frontline."
    assert len(fake.calls) == 2 and ev.used_llm


def test_agent_cannot_cite_knowledge_it_never_asked_for(monkeypatch, llm_env):
    """The handles K1..K3 exist, but no tool result showed them to the model."""
    async def handler(kwargs):
        return submit("B", ANALYSIS_EN, references=["K1", "K3"])

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = _agent(match, ctx, knowledge_block(3))
    assert len(fake.calls) == 1 and ev.cited_refs == []
    # What the pipeline selected for the decision is still listed, as related material.
    assert [(s["id"], s["cited"]) for s in ev.sources] == [
        ("test:entry_1", False), ("test:entry_2", False), ("test:entry_3", False)]


@pytest.mark.parametrize("references", ["K2", "K2 and K3", ["K2", 3, None], [["K2"]], {"K2": True}, 2, "all of them"])
def test_malformed_references_never_reject_a_verdict(monkeypatch, llm_env, references):
    async def handler(kwargs):
        if len(fake.calls) == 1:
            return response(None, [tool_call("retrieve_coaching_knowledge", call_id="k1")])
        return submit("B", ANALYSIS_EN, references=references)

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = _agent(match, ctx, knowledge_block(3))
    assert len(fake.calls) == 2 and ev.analysis == ANALYSIS_EN          # accepted at once, never re-asked
    assert set(ev.cited_refs) <= {"K2", "K3"} and len(ev.sources) == 3
    assert agent._submit_problem({"score": "B", "analysis": ANALYSIS_EN, "references": references}, "en") == ""


def test_text_verdict_of_the_agent_can_cite_too(monkeypatch, llm_env):
    async def handler(kwargs):
        if len(fake.calls) == 1:
            return response(None, [tool_call("retrieve_coaching_knowledge", call_id="k1")])
        return response(f"{TEXT_VERDICT_EN}\nReferences: K3")

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = _agent(match, ctx, knowledge_block(3))
    assert ev.lesson == "Respect the map." and _ids(ev.sources, cited=True) == ["test:entry_3"]


# ---------------------------------------------------------------------------
# Agent: free-text search
# ---------------------------------------------------------------------------

def _fake_search(calls: list[dict], hits: int = 2):
    """Stand-in for knowledge_base.search_knowledge: ``hits`` new entries numbered from first_ref."""
    def search(ctx, query, lang="zh", *, heroes=(), items=(), topic="", top_k=4, match_heroes=None,
               first_ref=1, shown=None):
        calls.append({"query": query, "lang": lang, "heroes": list(heroes), "items": list(items), "topic": topic,
                      "top_k": top_k, "match_heroes": list(match_heroes or []), "first_ref": first_ref,
                      "shown": dict(shown or {})})
        return knowledge_block(min(hits, top_k), first=first_ref, prefix=f"test:search{len(calls)}")
    return search


def test_search_results_continue_the_numbering_of_the_run(monkeypatch, llm_env):
    searches: list[dict] = []
    monkeypatch.setattr(knowledge_base, "search_knowledge", _fake_search(searches))
    first = {"query": "How does Puck survive Lina's Laguna Blade?", "heroes": ["Lina"], "topic": "matchup", "top_k": 2}
    second = {"query": "When to buy Black King Bar", "items": ["Black King Bar"]}
    params = next(t["function"]["parameters"] for t in agent._tools_schema("en", True)
                  if t["function"]["name"] == "retrieve_coaching_knowledge")
    assert argument_violations(first, params) == argument_violations(second, params) == []

    async def handler(kwargs):
        n = len(fake.calls)
        if n == 1:
            return response(None, [tool_call("retrieve_coaching_knowledge", call_id="block"),
                                   tool_call("retrieve_coaching_knowledge", first, call_id="s1")])
        if n == 2:
            return response(None, [tool_call("retrieve_coaching_knowledge", second, call_id="s2")])
        return submit("B", ANALYSIS_EN, references=["K5", "K1", "K8"])

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = _agent(match, ctx, knowledge_block(3))

    assert [s["first_ref"] for s in searches] == [4, 6]            # K1-K3 block, K4-K5 first search
    assert searches[0]["query"] == first["query"] and searches[0]["heroes"] == ["Lina"]
    assert searches[0]["topic"] == "matchup" and searches[0]["top_k"] == 2 and searches[0]["lang"] == "en"
    assert searches[1]["items"] == ["Black King Bar"] and searches[1]["top_k"] == 4
    # Heroes are restricted to this match: all ten, not only the ones near the player.
    assert searches[0]["match_heroes"] == [p.hero_name for p in match.players]
    # Entries the model has seen are named so the search does not repeat them.
    assert searches[0]["shown"] == {"test:entry_1": "K1", "test:entry_2": "K2", "test:entry_3": "K3"}
    assert searches[1]["shown"] == {**searches[0]["shown"], "test:search1_4": "K4", "test:search1_5": "K5"}
    results = _tool_results(fake.calls[1])
    assert "[K4] Curated note — Entry 4" in results["s1"] and "[K1]" not in results["s1"]
    assert "[K6] Curated note — Entry 6" in _tool_results(fake.calls[2])["s2"]
    # A search hit and a block entry are cited alike; K8 was never handed out.
    assert ev.cited_refs == ["K5", "K1"]
    assert [(s["id"], s["cited"]) for s in ev.sources] == [
        ("test:search1_5", True), ("test:entry_1", True), ("test:entry_2", False), ("test:entry_3", False)]
    assert [(s["tool"], s["args"]) for s in ev.agent_trace[:3]] == [
        ("retrieve_coaching_knowledge", {}), ("retrieve_coaching_knowledge", first),
        ("retrieve_coaching_knowledge", second)]


def test_a_search_before_the_block_does_not_take_its_numbers(monkeypatch, llm_env):
    searches: list[dict] = []
    monkeypatch.setattr(knowledge_base, "search_knowledge", _fake_search(searches, hits=1))

    async def handler(kwargs):
        n = len(fake.calls)
        if n == 1:
            return response(None, [tool_call("retrieve_coaching_knowledge", {"query": "Lina burst"}, call_id="s1")])
        if n == 2:
            return response(None, [tool_call("retrieve_coaching_knowledge", call_id="block")])
        return submit("B", ANALYSIS_EN, references=["K1", "K4"])

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = _agent(match, ctx, knowledge_block(3))
    assert searches[0]["first_ref"] == 4 and searches[0]["shown"] == {}     # block not shown yet, numbers reserved
    assert "[K4]" in _tool_results(fake.calls[1])["s1"]
    assert "[K1] Curated note — Entry 1" in _tool_results(fake.calls[2])["block"]
    assert _ids(ev.sources, cited=True) == ["test:entry_1", "test:search1_4"]


def test_search_limit_per_run(monkeypatch, llm_env):
    searches: list[dict] = []
    monkeypatch.setattr(knowledge_base, "search_knowledge", _fake_search(searches, hits=1))
    queries = [f"question number {i}" for i in range(1, 5)]

    async def handler(kwargs):
        n = len(fake.calls)
        if n == 1:   # three searches and a repeat of the first one in a single turn
            return response(None, [tool_call("retrieve_coaching_knowledge", {"query": q}, call_id=f"s{i}")
                                   for i, q in enumerate(queries[:3] + queries[:1], start=1)])
        if n == 2:
            return response(None, [tool_call("retrieve_coaching_knowledge", {"query": queries[3]}, call_id="s5"),
                                   tool_call("retrieve_coaching_knowledge", call_id="block")])
        return submit("B", ANALYSIS_EN, references=["K4", "K6"])

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = _agent(match, ctx, knowledge_block(3))
    assert [s["query"] for s in searches] == queries[:3]              # the knowledge base is asked three times
    assert [s["first_ref"] for s in searches] == [4, 5, 6]
    first_round = _tool_results(fake.calls[1])
    assert "already returned" in first_round["s4"].lower()            # a repeat costs nothing and finds nothing new
    second_round = _tool_results(fake.calls[2])
    assert "limit of 3 knowledge searches" in second_round["s5"] and "[K" not in second_round["s5"]
    assert "[K1] Curated note" in second_round["block"]               # the block itself is not a search
    assert _ids(ev.sources, cited=True) == ["test:search1_4", "test:search3_6"]


def test_search_limit_note_is_localized(monkeypatch):
    monkeypatch.setattr(knowledge_base, "search_knowledge", _fake_search([], hits=1))
    monkeypatch.setenv("RAG_AGENT_SEARCH_LIMIT", "1")
    match, ctx = death_context("zh")
    tc = agent._ToolCtx(match=match, context=ctx, knowledge=knowledge_block(2), lang="zh")
    assert "[K3]" in agent._t_knowledge(tc, {"query": "first"})
    note = agent._t_knowledge(tc, {"query": "second"})
    assert "最多1次" in note and not re.search(r"[A-Za-z]{4,}", note)
    # Sloppy arguments are tolerated: one bare hero name, a top_k that is no number, an unknown topic.
    monkeypatch.setenv("RAG_AGENT_SEARCH_LIMIT", "3")
    calls: list[dict] = []
    monkeypatch.setattr(knowledge_base, "search_knowledge", _fake_search(calls, hits=1))
    agent._t_knowledge(tc, {"heroes": "Lina", "items": ["Blink Dagger", 7, ""], "topic": "Jungle", "top_k": "many"})
    assert calls[0]["heroes"] == ["Lina"] and calls[0]["items"] == ["Blink Dagger"]
    assert calls[0]["topic"] == "" and calls[0]["top_k"] == 4 and calls[0]["query"] == ""
    # ... and a top_k no int can hold (the model got "No data available." and lost a search)
    for huge in (float("inf"), 1e400, float("nan"), -float("inf")):
        fresh = agent._ToolCtx(match=match, context=ctx, knowledge=knowledge_block(2), lang="zh")
        del calls[:]
        assert "[K3]" in agent._execute("retrieve_coaching_knowledge", fresh, {"query": "q", "top_k": huge})
        assert calls[0]["top_k"] == 4


def test_search_limit_is_bounded_and_zero_switches_searching_off(monkeypatch, llm_env):
    from backend.app.ai import rag

    for value, limit in (("99", 6), ("6", 6), ("7", 6), ("2", 2), ("0", 0), ("-3", 0), ("abc", 3), ("", 3)):
        monkeypatch.setenv("RAG_AGENT_SEARCH_LIMIT", value)
        assert rag.agent_search_limit() == limit == agent._search_limit(), value
    # Switched off, every call of the tool is the same call, whatever arguments come with it:
    # the block is sent once (five different "queries" used to get the 8 KB block five times).
    monkeypatch.setenv("RAG_AGENT_SEARCH_LIMIT", "0")
    searches: list[dict] = []
    monkeypatch.setattr(knowledge_base, "search_knowledge", _fake_search(searches, hits=1))

    async def handler(kwargs):
        if len(fake.calls) == 1:
            return response(None, [tool_call("retrieve_coaching_knowledge", {"query": f"question {i}"},
                                             call_id=f"s{i}") for i in range(1, 6)])
        return submit("B", ANALYSIS_EN, references=["K1"])

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context()
    ev = _agent(match, ctx, knowledge_block(3))
    results = _tool_results(fake.calls[1])
    assert searches == [] and "[K1] Curated note" in results["s1"]
    assert all("already returned" in results[f"s{i}"].lower() for i in range(2, 6))
    assert _ids(ev.sources, cited=True) == ["test:entry_1"]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _report_match():
    players = default_players()
    fights = [
        teamfight_death(1, "Puck", 129, "Lina", start=900, x=500.0, y=300.0),
        teamfight_death(1, "Puck", 128, "Phantom Assassin", start=1500, x=1500.0, y=1200.0),
    ]
    players[1].deaths = 2
    players[1].multi_kills = {"2": 1}        # a good play: with MAX_LLM_EVALS=2 it stays rule-based
    return make_match(players, teamfights=fights)


@pytest.fixture
def knowledge_env(monkeypatch, llm_env):
    """Five numbered entries per LLM-bound decision, three related ones for the others."""
    monkeypatch.setattr(knowledge_base, "get_knowledge_for_context", lambda ctx, lang="zh": knowledge_block(5))
    monkeypatch.setattr(knowledge_base, "related_refs",
                        lambda ctx, lang="zh", k=3: knowledge_refs(k, prefix="test:related"))
    monkeypatch.setattr(coach, "MAX_LLM_EVALS", 2)


def test_report_lists_sources_for_every_evaluation(monkeypatch, knowledge_env):
    async def handler(kwargs):
        if is_phase_prompt(kwargs):   # a bracketed handle and a bare one (phase summaries kept "K3")
            return response(PHASE_JSON.replace("Strong phase", "Strong phase [K1]").replace("every teamfight won.",
                                                                                         "every teamfight won. K3"))
        if not any(m.get("role") == "tool" for m in kwargs["messages"]):
            return response(None, [tool_call("retrieve_coaching_knowledge", call_id="k")])
        return submit("C", "Puck joined the fight late [K4] and was focused first by the enemy cores.",
                      references=["K2"])

    install_fake(monkeypatch, handler)
    report = asyncio.run(coach.generate_coaching_report(_report_match(), 1, lang="en"))
    data = report.to_dict()
    json.dumps(data, allow_nan=False)
    assert data["report_version"] == 4 and set(data["knowledge"]) == {"mode", "embedder", "chunks", "vectors"}
    deaths, moments = data["death_analyses"], data["key_moments"]
    assert len(deaths) == 2 and all(d["llm_used"] for d in deaths) and moments
    for d in deaths:      # LLM verdicts: cited entries first, then the best of the rest, four at most
        assert [(s["id"], s["cited"]) for s in d["sources"]] == [
            ("test:entry_2", True), ("test:entry_4", True), ("test:entry_1", False), ("test:entry_3", False)]
    for m in moments:     # rule-based: three related entries, never cited
        assert not m["llm_used"]
        assert [(s["id"], s["cited"]) for s in m["sources"]] == [
            (f"test:related_{n}", False) for n in (1, 2, 3)]
    # No handle reaches the player: not in a verdict, not in a phase review, nowhere.
    leaks = [(path, text) for path, text in _strings(data) if HANDLE_RE.search(text) or BARE_HANDLE_RE.search(text)]
    assert not leaks, leaks
    assert any(review["llm_used"] and review["summary"].startswith("Strong phase: no deaths")
               and review["summary"].endswith("every teamfight won.") for review in data["phase_reviews"])
    assert all(p["llm_used"] and p["summary"].startswith("Strong phase:") for p in data["phase_reviews"])
    assert "cited_refs" not in deaths[0] and "agent_trace" not in deaths[0]


def test_failed_llm_evaluation_lists_the_block_uncited(monkeypatch, knowledge_env):
    async def handler(kwargs):
        raise RuntimeError("boom")

    install_fake(monkeypatch, handler)
    data = asyncio.run(coach.generate_coaching_report(_report_match(), 1, lang="en")).to_dict()
    assert data["degraded"] is True and data["llm_used"] is False
    for d in data["death_analyses"]:
        assert d["degraded"] and [(s["id"], s["cited"]) for s in d["sources"]] == [
            (f"test:entry_{n}", False) for n in (1, 2, 3)]


def test_rule_text_and_phase_fallback_lose_handles():
    """Defence in depth: rule text is built from match data, which could hold anything."""
    match, ctx = death_context()
    ctx.decision_point.brief = "Killed by Lina [K3] in a teamfight"
    ev = evaluator.evaluate_decision_fallback(ctx, lang="en")
    assert not any(HANDLE_RE.search(t) for t in (ev.analysis, ev.mistake, ev.better_play, ev.lesson))
    assert ev.sources == [] and ev.cited_refs == []      # the rules cite nothing; the coach adds related entries


def test_knowledge_step_degrades_without_the_knowledge_layer(monkeypatch):
    """A broken retrieval leaves the report without sources, not without evaluations."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")

    def boom(*_a, **_k):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(knowledge_base, "related_refs", boom)
    data = asyncio.run(coach.generate_coaching_report(_report_match(), 1, lang="en")).to_dict()
    assert data["death_analyses"] and all(d["sources"] == [] for d in data["death_analyses"])


@pytest.mark.parametrize("lang, benchmark, note", [
    ("en", "Median GPM of typical Mid (Pos 2) heroes", "This match was played in Turbo mode"),
    ("zh", "2号位常见英雄在OpenDota全分段的GPM中位数", "本场为加速模式比赛"),
])
def test_the_knowledge_block_of_a_turbo_match_has_no_normal_mode_benchmarks(lang, benchmark, note):
    """The game mode reaches the real knowledge layer with the decision context."""
    decisions = [DecisionPoint(time=905, point_type="death", player_slot=1, hero_name="Puck",
                               severity="critical", brief="Killed by Lina")]

    def block(mode: str) -> str:
        match = _report_match()
        match.game_mode = mode
        ctx, knowledge, _rules = coach._context_step(match, decisions, {}, 2, lang, {0})[0]
        assert ctx.match_context["game_mode"] == mode and knowledge.refs
        return str(knowledge)

    normal, turbo = block("ranked_all_pick"), block("turbo")
    assert benchmark in normal and note not in normal
    assert note in turbo and benchmark not in turbo


@pytest.mark.parametrize("status, blocks, expected", [
    ({"mode": "hybrid", "embedder": "bge-small-en-v1.5-int8", "chunks": 777, "vectors": 993}, [],
     {"mode": "hybrid", "embedder": "bge-small-en-v1.5-int8", "chunks": 777, "vectors": 993}),
    # Ranked without vectors: the vector space is not part of the answer.
    ({"mode": "lexical", "embedder": "bge-small-en-v1.5-int8", "chunks": 777, "vectors": 993}, [],
     {"mode": "lexical", "embedder": None, "chunks": 777, "vectors": 0}),
    # What the blocks of this report say wins over the momentary status.
    ({"mode": "hybrid", "embedder": "m", "chunks": 5, "vectors": 9}, ["lexical", "lexical"],
     {"mode": "lexical", "embedder": None, "chunks": 5, "vectors": 0}),
    ({"mode": "hybrid", "embedder": "m", "chunks": 5, "vectors": 9}, ["lexical", "hybrid"],
     {"mode": "hybrid", "embedder": "m", "chunks": 5, "vectors": 9}),
    ({}, [], {"mode": "lexical", "embedder": None, "chunks": 0, "vectors": 0}),
])
def test_report_level_knowledge_object(monkeypatch, status, blocks, expected):
    from backend.app.ai import rag

    monkeypatch.setattr(rag, "status", lambda store=None: dict(status))
    monkeypatch.setattr(knowledge_base, "warm_stores", lambda: {})
    texts = ["", "plain text"] + [
        knowledge_base.KnowledgeText("block", retrieval={"mode": mode}) for mode in blocks]
    assert coach._knowledge_info(texts) == expected
    # Rule-based evaluations have no block: their related entries carry the record
    related = [knowledge_base.RefList([], {"mode": mode}) for mode in blocks]
    assert coach._knowledge_info(related) == expected


def test_no_game_to_review_is_refused():
    """A replay cut before the horn used to get 'KDA 0/0/0 in 0 minutes, grade B'."""
    match = _report_match()
    match.partial, match.duration, match.winner_known = True, 0, False
    with pytest.raises(coach.NoGameError):
        asyncio.run(coach.generate_coaching_report(match, 1, lang="en"))
    assert issubclass(coach.NoGameError, ValueError)
    # ... and so did one cut 4 s or 133 s after it ("grade B: ... the grade is only a default")
    for duration in (4, 133, 299):
        match.duration = duration
        with pytest.raises(coach.NoGameError):
            asyncio.run(coach.generate_coaching_report(match, 1, lang="en"))
    match.duration = 600                       # truncated later: there is a game to review
    assert asyncio.run(coach.generate_coaching_report(match, 1, lang="en")).to_dict()["report_version"] == 4
    match.partial, match.duration = False, 133  # a complete match is never refused for its length
    assert asyncio.run(coach.generate_coaching_report(match, 1, lang="en")).to_dict()["report_version"] == 4


def test_knowledge_object_describes_the_retrieval_of_this_report(monkeypatch):
    """It was read from the process state at the END of the report: with the stores reset in between
    (a learning run, a game-data update) 60 of 60 reports said "lexical, 0 chunks" over their sources."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    real = coach._phase_stats_step

    def reset_then_slice(*args, **kwargs):      # runs after the context step, like another thread would
        knowledge_base.reset_knowledge_stores()
        return real(*args, **kwargs)

    monkeypatch.setattr(coach, "_phase_stats_step", reset_then_slice)
    data = asyncio.run(coach.generate_coaching_report(_report_match(), 1, lang="en")).to_dict()
    assert data["death_analyses"] and data["death_analyses"][0]["sources"]
    assert data["knowledge"]["mode"] == "lexical" and data["knowledge"]["chunks"] > 700
    assert knowledge_base.built_stores() == {}                     # the reset did happen after the retrieval
    # A report without a decision in a process that has built nothing yet: the stores are built for the answer
    quiet = _report_match()
    quiet.kills_log, quiet.teamfights = [], []
    for player in quiet.players:
        player.deaths = player.kills = player.assists = 0
    monkeypatch.setattr(coach, "_detect_step", lambda *a, **k: ([], []))
    knowledge_base.reset_knowledge_stores()
    data = asyncio.run(coach.generate_coaching_report(quiet, 1, lang="en")).to_dict()
    assert data["death_analyses"] == [] and data["knowledge"]["chunks"] > 700


# ---------------------------------------------------------------------------
# With the real knowledge base (static chunks of the repository, lexical ranking)
# ---------------------------------------------------------------------------

def test_real_block_and_search_share_one_numbering(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    match, ctx = death_context()
    ctx.match_context["player_position"] = 2
    block = knowledge_base.get_knowledge_for_context(ctx, lang="en")
    assert block.refs, "the knowledge base returned no entries for a teamfight death"
    tc = agent._ToolCtx(match=match, context=ctx, knowledge=block, lang="en")
    shown = agent._t_knowledge(tc, {})
    handles = re.findall(r"(?m)^\[(K\d+)\] ", shown)
    assert handles == [ref.ref for ref in block.refs] == [f"K{n}" for n in range(1, len(handles) + 1)]
    assert tc.shown == {ref.chunk_id: ref.ref for ref in block.refs}

    result = agent._t_knowledge(tc, {"query": "How should Puck play against Lina in lane?", "heroes": ["Lina", "Zeus"],
                                     "topic": "laning", "top_k": 6})
    new = re.findall(r"(?m)^\[(K\d+)\] ", result)
    assert new and new == [f"K{n}" for n in range(len(handles) + 1, len(handles) + len(new) + 1)]
    assert len(new) <= 6 and "Zeus" in result                       # not in this match: named as ignored
    # Capped between entries: the reference block is closed and the last entry is whole.
    assert result.count("<reference>") == result.count("</reference>") == 1
    body = result.split("<reference>\n", 1)[1].split("\n</reference>", 1)[0]
    assert len(body) <= knowledge_base.SEARCH_MAX_CHARS
    by_handle = {ref.ref: ref for ref in tc.refs.values()}
    for entry in body.split("\n\n"):
        head = re.match(r"\[(K\d+)\] ", entry)
        assert head and head.group(1) in by_handle and len(entry.split("\n", 1)) == 2
    # Nothing the block already showed comes back under a second number.
    assert not {tc.refs[h].chunk_id for h in new} & {ref.chunk_id for ref in block.refs}
    assert tc.searches == 1 and tc.next_ref == len(handles) + len(new) + 1


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_rules_report_with_the_real_knowledge_base(monkeypatch, lang):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    data = asyncio.run(coach.generate_coaching_report(_report_match(), 1, lang=lang)).to_dict()
    json.dumps(data, allow_nan=False)
    evaluations = data["death_analyses"] + data["key_moments"]
    assert evaluations and data["knowledge"]["mode"] == "lexical" and data["knowledge"]["chunks"] > 100
    for e in evaluations:
        assert 1 <= len(e["sources"]) <= 3
        for s in e["sources"]:
            assert s["cited"] is False and s["title"].strip() and s["source_type"] in SOURCE_TYPES
            if lang == "en":
                assert not re.search(r"[一-鿿]", s["title"]), s["title"]
    assert not [(p, t) for p, t in _strings(data) if HANDLE_RE.search(t)]
