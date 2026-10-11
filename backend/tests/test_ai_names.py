"""Hero, item and ability names follow the report language.

A zh report showed "本场比赛Templar Assassin失败" under a header saying 圣堂刺客:
the UI takes its names from the catalogs' ``name_zh``, the backend wrote the
English ones. Covered here, end to end:

- the display-name helpers of ``data_manager`` (Chinese catalog name for zh,
  English as the fallback and for en);
- names in text that is not written through them: the English reference
  knowledge of a zh prompt (a glossary), an LLM verdict (replaced), the title
  of a knowledge entry (replaced);
- lookups keyed by English names and slugs, which must keep resolving, also
  from a Chinese name a model sends back;
- whole rule-based reports in both languages;
- the positions the analyzer asks for (``positions.match_positions_payload``).
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from backend.app import config
from backend.app.ai import agent, coach, data_manager, evaluator, knowledge_base, positions, pro_patterns
from backend.app.ai.rag import freetext
from backend.app.mock_data import generate_mock_match
from backend.app.models import ItemEvent
from test_ai_llm_agent import death_context
from test_ai_pipeline_helpers import (
    ANALYSIS_ZH,
    CJK_RE,
    default_players,
    install_fake,
    knowledge_block,
    llm_env,  # noqa: F401  (fixture)
    make_match,
    response,
    submit,
)

STATIC = Path(config.STATIC_DIR) / "data"
# Report fields that hold codes and keys (the frontend maps them itself), not prose.
_CODE_KEYS = {"hero_name", "label_en", "label_zh", "id", "source_type", "point_type", "severity", "score",
              "overall_grade", "category_code", "metric", "role", "phase", "type", "significance",
              "root_cause_type", "method", "confidence", "lane_result", "rating", "benchmark_source",
              "overall_rating", "mode", "embedder", "patch", "lang"}


def _catalog_names() -> dict[str, str]:
    """English name -> Chinese name of every hero and item the committed catalogs name in Chinese."""
    names: dict[str, str] = {}
    for hero in json.loads((STATIC / "heroes.json").read_text(encoding="utf-8"))["by_id"].values():
        names[hero["name"]] = hero["name_zh"]
    for item in json.loads((STATIC / "items.json").read_text(encoding="utf-8"))["by_key"].values():
        if item.get("name") and item.get("name_zh") and (item.get("cost") or " " in item["name"]):
            names.setdefault(item["name"], item["name_zh"])
    return names


def _prose(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            if key not in _CODE_KEYS:
                yield from _prose(value, f"{path}.{key}")
    elif isinstance(node, list):
        for value in node:
            yield from _prose(value, f"{path}[]")
    elif isinstance(node, str):
        yield path, node


def _english_names_in(text: str, names: dict[str, str]) -> list[str]:
    pattern = re.compile(r"(?<![A-Za-z])(" + "|".join(
        re.escape(name) for name in sorted(names, key=len, reverse=True) if len(name) >= 3) + r")(?![A-Za-z])")
    return pattern.findall(text)


# ---------------------------------------------------------------------------
# data_manager: display names
# ---------------------------------------------------------------------------

def test_display_names_are_chinese_for_zh_and_english_otherwise():
    dm = data_manager
    for identifier in ("Templar Assassin", "templar_assassin", "npc_dota_hero_templar_assassin", "ta", "圣堂刺客"):
        assert dm.hero_display_name(identifier, "zh") == "圣堂刺客"
        assert dm.hero_display_name(identifier, "en") == dm.hero_display_name(identifier) == "Templar Assassin"
    for identifier in ("Black King Bar", "black_king_bar", "item_black_king_bar", "bkb", "黑皇杖"):
        assert dm.item_display_name(identifier, "zh") == "黑皇杖"
        assert dm.item_display_name(identifier, "en") == dm.item_display_name(identifier) == "Black King Bar"
    assert (dm.ability_display_name("skeleton_king_reincarnation", "zh"),
            dm.ability_display_name("skeleton_king_reincarnation", "en")) == ("绝冥再生", "Reincarnation")
    assert dm.unit_display_name("npc_dota_hero_skeleton_king", "zh") == "冥魂大帝"
    assert dm.display_name("skeleton_king", "zh") == "冥魂大帝" and dm.display_name("bkb", "zh") == "黑皇杖"
    # every hero of the catalog has a Chinese name; none comes back empty or as its English name
    catalog = json.loads((STATIC / "heroes.json").read_text(encoding="utf-8"))["by_id"]
    assert len(catalog) >= 127
    for hero in catalog.values():
        assert dm.hero_display_name(hero["slug"], "zh") == hero["name_zh"] != hero["name"]
        assert dm.hero_display_name(hero["name_zh"], "en") == dm.resolve_hero_name(hero["name"])
    # a name the catalog has no Chinese for stays English; an unknown identifier is prettified as before
    assert dm.ability_display_name("special_bonus_attributes", "zh") == "Attribute Bonus"
    assert dm.hero_display_name("npc_dota_hero_brand_new", "zh") == "Brand New"
    assert dm.item_display_name("item_shiny_new_thing", "zh") == "Shiny New Thing"
    # any language that is not "en" is the zh report, as everywhere in the coach
    assert dm.hero_display_name("Axe", "") == dm.hero_display_name("Axe", "ja") == "斧王"


def test_recipes_take_the_games_chinese_name():
    item = data_manager.item_display_name
    assert (item("recipe_black_king_bar", "en"), item("recipe_black_king_bar", "zh")) == (
        "Black King Bar Recipe", "图纸（黑皇杖）")
    assert item("Black King Bar Recipe", "zh") == "图纸（黑皇杖）"
    # a recipe no catalog names: after its item, in the wording of the game's own recipe names
    assert (item("recipe_shiny_thing", "en"), item("recipe_shiny_thing", "zh")) == (
        "Shiny Thing Recipe", "图纸（Shiny Thing）")


def test_an_ability_named_in_english_is_looked_up_among_its_heros_abilities():
    """The states hold abilities by their English display name ("Blink"); an inflictor of that
    name is the item (Blink Dagger). With the hero the name is unambiguous."""
    name = data_manager.ability_display_name
    assert name("Blink", "zh", hero="Anti-Mage") == name("Blink", "zh", hero="antimage") == "闪烁"
    assert name("Blink", "zh") == "闪烁匕首" and name("Blink", "en", hero="Anti-Mage") == "Blink Dagger"
    assert name("Reincarnation", "zh", hero="skeleton_king") == "绝冥再生"
    assert name("Reincarnation", "zh") == "绝冥再生"                       # by name alone when nothing else has it
    assert name("Not An Ability", "zh", hero="Axe") == "Not An Ability"
    # a talent is found under its full name, numbers included
    talent = data_manager.resolve_ability_name("special_bonus_unique_axe_4")
    assert name(talent, "zh", hero="Axe") == name("special_bonus_unique_axe_4", "zh")
    assert re.fullmatch(r"\+\d+ 反击螺旋伤害", name(talent, "zh", hero="Axe"))


def test_a_talent_takes_the_numbers_of_the_data_not_of_the_catalog(monkeypatch):
    """Talent keys are reused between patches with other numbers, and the fetched game data
    can be newer than the committed catalog."""
    index = data_manager._build_ability_zh_index()
    monkeypatch.setattr(data_manager, "_build_ability_zh_index", lambda: {
        "by_key": {**index["by_key"], "special_bonus_test": ("+40 Counter Helix Damage", "+40 反击螺旋伤害"),
                   "special_bonus_swap": ("+2s Duration / +10 Damage", "+10 伤害 / +2秒持续时间")},
        "by_name": index["by_name"]})
    data_manager.invalidate_all_caches()
    try:
        assert data_manager._ability_name_zh("special_bonus_test", "+40 Counter Helix Damage") == "+40 反击螺旋伤害"
        assert data_manager._ability_name_zh("special_bonus_test", "+30 Counter Helix Damage") == "+30 反击螺旋伤害"
        # the Chinese wording states the numbers in another order: matched by value
        assert data_manager._ability_name_zh("special_bonus_swap", "+3s Duration / +15 Damage") == "+15 伤害 / +3秒持续时间"
        # another wording is another talent by now: no Chinese name rather than a wrong one
        assert data_manager._ability_name_zh("special_bonus_test", "+40 Berserker's Call Armor") == ""
        assert data_manager._ability_name_zh("special_bonus_nobody_knows", "+1 Thing") == ""
    finally:
        monkeypatch.undo()
        data_manager.invalidate_all_caches()


# ---------------------------------------------------------------------------
# Lookups keep working
# ---------------------------------------------------------------------------

def test_lookups_resolve_chinese_and_bilingual_names_to_the_english_keys():
    """Tags, retrieval and every table are keyed by English names / slugs. A model that is shown
    "冥魂大帝（Wraith King）" may send either back as a tool argument."""
    dm = data_manager
    for written in ("冥魂大帝", "冥魂大帝（Wraith King）", "冥魂大帝 (Wraith King)", "Wraith King", " 冥魂大帝 "):
        assert dm.resolve_hero_name(written) == "Wraith King" and dm.hero_slug(written) == "skeleton_king"
        assert dm.is_known_hero(written)
    for written in ("闪烁匕首", "闪烁匕首（Blink Dagger）", "Blink Dagger"):
        assert dm.resolve_item_key(written) == dm.item_key(written) == "blink"
        assert dm.resolve_item_name(written) == "Blink Dagger"
    assert not dm.is_known_hero("辅助") and dm.resolve_item_key("辅助") is None and dm.resolve_hero_name("") is None
    # the knowledge layer reaches the same tables
    assert knowledge_base.get_hero_info("冥魂大帝", lang="en")["role"] == knowledge_base.get_hero_info(
        "Wraith King", lang="en")["role"] != "unknown"
    assert knowledge_base._find_matchup("卓尔游侠", "冥界亚龙") == knowledge_base._find_matchup("Drow Ranger", "Viper")
    assert knowledge_base._find_matchup("Drow Ranger", "Viper") is not None


def test_a_question_may_name_heroes_and_items_in_chinese():
    assert data_manager.find_zh_names("冥界亚龙的毒怎么解，要不要出黑皇杖？") == (["viper"], ["black_king_bar"])
    assert data_manager.find_zh_names("图纸（远行鞋）") == ([], ["recipe_travel_boots"])   # the longest name wins
    assert data_manager.find_zh_names("how to lane against Viper") == ([], [])
    assert data_manager.find_zh_names(None) == ([], [])
    assert freetext.find_entities("how to play against 冥界亚龙 with Wraith King, buy 闪烁匕首?") == (
        ["skeleton_king", "viper"], ["blink"])
    # an English question is read exactly as before
    assert freetext.find_entities("how do I lane against Viper as Wraith King") == (["skeleton_king", "viper"], [])


# ---------------------------------------------------------------------------
# Names in running text
# ---------------------------------------------------------------------------

_MATCH_HEROES = ["Wraith King", "Legion Commander", "Lion", "Anti-Mage"]


def test_glossary_lists_the_english_names_of_a_reference_text():
    text = ("Wraith King — 对线要点：Wraithfire Blast is a reliable stun. With Reincarnation you can play "
            "aggressively. Radiance timing; Blink Dagger timing. Duel at 6 forces a 1v1. Wise players respect BKB.")
    pairs = data_manager.name_glossary(text, "zh", _MATCH_HEROES)
    assert pairs == [("Wraith King", "冥魂大帝"), ("Wraithfire Blast", "冥火爆击"), ("Reincarnation", "绝冥再生"),
                     ("Radiance", "辉耀"), ("Blink Dagger", "闪烁匕首"), ("Duel", "决斗")]
    assert data_manager.name_glossary(text, "en", _MATCH_HEROES) == []
    assert data_manager.name_glossary("全是中文，没有英文名称。", "zh", _MATCH_HEROES) == []
    # abilities only of the heroes the text is about: "Duel" is a word unless Legion Commander plays
    assert ("Duel", "决斗") not in data_manager.name_glossary(text, "zh", ["Wraith King"])
    assert evaluator.names_glossary(text, "zh", _MATCH_HEROES) == (
        "名称对照（以上资料中的英文名称 = 回答里要写的中文名）：Wraith King = 冥魂大帝；Wraithfire Blast = 冥火爆击；"
        "Reincarnation = 绝冥再生；Radiance = 辉耀；Blink Dagger = 闪烁匕首；Duel = 决斗")
    assert evaluator.names_glossary(text, "en", _MATCH_HEROES) == ""


@pytest.mark.parametrize("written, expected", [
    ("Puck 在下路河道继续前压时 Lina 不在视野内，随后被 Phantom Assassin 先手击杀。",
     "帕克在下路河道继续前压时莉娜不在视野内，随后被幻影刺客先手击杀。"),
    # the bilingual form of the prompt, copied into the answer
    ("冥魂大帝（Wraith King）被 Legion Commander 的 Duel 击杀。", "冥魂大帝被军团指挥官的决斗击杀。"),
    ("冥魂大帝 (Wraith King) 没有 Black King Bar。", "冥魂大帝没有黑皇杖。"),
    ("Wraith King（冥魂大帝）买了 Radiance（辉耀）。", "冥魂大帝买了辉耀。"),
    ("Wraith King's Reincarnation 没有魔法。", "冥魂大帝的绝冥再生没有魔法。"),
    ("先出 Blink Dagger、Force Staff，再出 Aghanim's Scepter。", "先出闪烁匕首、原力法杖，再出阿哈利姆神杖。"),
    ("Lina Lion 一起来抓", "莉娜、莱恩一起来抓"),
    # everything that is not a catalog name with a Chinese entry stays as it is
    ("KDA 3/2/1，GPM 500，被 Gank 了两次；TP 在 CD。", "KDA 3/2/1，GPM 500，被 Gank 了两次；TP 在 CD。"),
    ("Wise 的选择是 BKB。", "Wise 的选择是 BKB。"),                       # "Wise" is an enchantment, and a word
    ("Viper Strike 很疼", "冥界亚龙 Strike 很疼"),                        # not an ability of these heroes: only the hero
    ("没有任何英文", "没有任何英文"),
])
def test_english_names_in_a_chinese_text_are_replaced_by_the_catalog_names(written, expected):
    assert data_manager.localize_names(written, "zh", _MATCH_HEROES) == expected
    assert data_manager.localize_names(written, "en", _MATCH_HEROES) == written      # an en report is never touched


def test_text_helpers_never_raise():
    for helper in (data_manager.localize_names, data_manager.strip_names, data_manager.find_zh_names):
        for junk in (None, "", 0, 3.5, [], {"a": 1}, object()):
            helper(junk)
    assert data_manager.localize_names(None, "zh") == "" and data_manager.name_glossary(None) == []
    assert data_manager.name_glossary(object(), "zh", [None, 3, "nobody"]) == []


def test_a_proper_name_is_not_the_language_of_a_verdict():
    """ "Black King Bar" as the whole "mistake" of a zh verdict was 14 Latin letters and no Chinese:
    rejected as written in the wrong language. A name is not a sentence."""
    ok = evaluator._language_ok
    assert ok("Black King Bar", "zh") and ok("Aghanim's Scepter 和 Outworld Destroyer", "zh")
    assert ok("没有及时使用 Black King Bar，被 Legion Commander 的 Duel 抓到。", "zh")
    assert not ok("He walked in alone and died before the team could help.", "zh")     # an English sentence still is
    assert not ok("Lina killed him because he walked forward without any vision at all.", "zh")
    assert ok("帕克独自前压。", "zh") and not ok("帕克独自前压。", "en")
    assert evaluator.verdict_problem("C", ANALYSIS_ZH, "Black King Bar", "尽早做出 Black King Bar。", lang="zh") == ""
    assert evaluator.verdict_problem("C", ANALYSIS_ZH, "He should have bought a defensive item first.",
                                     lang="zh") == "language"


# ---------------------------------------------------------------------------
# Whole reports (rules mode)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slot", [0, 2, 129])
def test_rules_report_names_everything_in_its_language(slot):
    """Every hero and item name of a zh report is the catalog's Chinese one; the en report of
    the same player has no Chinese in it. Codes and keys (``hero_name``, ids) stay English."""
    match = generate_mock_match()
    hero = next(p.hero_name for p in match.players if p.player_slot == slot)
    names = _catalog_names()
    zh = asyncio.run(coach.generate_coaching_report(match, slot, lang="zh")).to_dict()
    en = asyncio.run(coach.generate_coaching_report(match, slot, lang="en")).to_dict()
    assert (zh["lang"], en["lang"]) == ("zh", "en") and zh["report_version"] == en["report_version"] == 5
    assert zh["hero_name"] == en["hero_name"] == hero                    # the key the frontend looks the hero up by
    left = [(path, name) for path, text in _prose(zh) for name in _english_names_in(text, names)]
    assert not left, left[:8]
    assert names[hero] in zh["summary"] and hero not in zh["summary"] and hero in en["summary"]
    assert names[hero] in zh["laning_review"]["lane_matchup"]
    chinese = [(path, text) for path, text in _prose(en) if CJK_RE.search(text)]
    assert not chinese, chinese[:5]
    # the same decisions, graded the same, in both languages
    for group in ("death_analyses", "key_moments"):
        assert [(e["time"], e["point_type"], e["score"]) for e in zh[group]] == [
            (e["time"], e["point_type"], e["score"]) for e in en[group]]
        assert all(e["hero_name"] == hero for e in zh[group])
    # titles of knowledge entries follow the language too (the id of an entry is the same in both)
    zh_sources = {s["id"]: s["title"] for e in zh["death_analyses"] + zh["key_moments"] for s in e["sources"]}
    en_sources = {s["id"]: s["title"] for e in en["death_analyses"] + en["key_moments"] for s in e["sources"]}
    assert zh_sources and en_sources
    laning = f"hero_laning:{data_manager.hero_slug(hero)}"
    assert zh_sources.get(laning, f"{names[hero]} — 对线要点") == f"{names[hero]} — 对线要点"
    assert en_sources.get(laning, f"{hero} — laning tips") == f"{hero} — laning tips"
    assert laning in zh_sources or laning in en_sources


def test_report_fields_with_nothing_to_say_are_empty():
    """Positive moments carried "None" / "无" as their mistake, "Keep it up" as the better play
    and an analysis that repeated the brief behind a sentence of praise."""
    match = generate_mock_match()
    for lang in ("zh", "en"):
        seen = 0
        for slot in (0, 1, 129, 131):
            report = asyncio.run(coach.generate_coaching_report(match, slot, lang=lang)).to_dict()
            for item in report["death_analyses"] + report["key_moments"]:
                for field in ("analysis", "mistake", "better_play", "lesson"):
                    assert item[field].strip().rstrip(".。").lower() not in ("none", "n/a", "无", "keep it up", "继续保持")
                assert item["brief"]
                if item["point_type"] != "death":      # (a death's analysis is a sentence of its own about the kill)
                    assert item["brief"].rstrip(".。") not in item["analysis"]                  # said once, as the title
                if item["severity"] == "positive":
                    seen += 1
                    assert item["mistake"] == "" and item["better_play"] == "" and item["lesson"]
        assert seen >= 3


def test_item_timing_rows_name_the_item_in_the_report_language():
    players = default_players()
    players[0].items = [ItemEvent(item_name="Battle Fury", item_key="bfury", time=780)]
    match = make_match(players)
    rows = {}
    for lang in ("zh", "en"):
        report = pro_patterns.compare_player(players[0], match, lang=lang, position=1)
        row, = [c for c in report.comparisons if c.metric == "item_battle_fury"]      # the code is the same
        rows[lang] = (row.metric_label, row.description)
    assert rows["zh"][0] == "狂战斧出装时间" and rows["zh"][1].startswith("狂战斧在13:00完成，")
    assert rows["en"][0] == "Battle Fury Timing" and rows["en"][1].startswith("Battle Fury completed at 13:00, ")


# ---------------------------------------------------------------------------
# LLM paths: the prompt gives the names, the verdict is checked
# ---------------------------------------------------------------------------

_REFERENCE_BODY = "Lina — laning tips: Dragon Slave pushes the wave; a Black King Bar answers Laguna Blade."


def test_zh_prompt_writes_chinese_names_and_maps_the_english_reference(llm_env):
    match, ctx = death_context("zh")
    ctx.match_context["positions"] = {"radiant": [["Puck", 2]], "dire": [["Lina", 2], ["Lion", 5]]}
    ctx.player_state.items = ["Blink Dagger", "Power Treads"]
    ctx.player_state.ability_levels = {"Illusory Orb": 4, "Waning Rift": 2}
    prompt = evaluator._build_evaluation_prompt(ctx, knowledge_block(2, body=_REFERENCE_BODY), lang="zh")
    state = evaluator.format_player_state(ctx, "zh")
    assert state.startswith("[玩家] 帕克 (Lv")
    assert "装备: 闪烁匕首(主动)、动力鞋(主动)" in state and "技能: 幻象法球(Lv4)、新月之痕(Lv2)" in state
    assert "己方 帕克（Puck）2号位；敌方 莉娜（Lina）2号位、莱恩（Lion）5号位" in prompt
    # the reference text itself is not rewritten; what it names in English is mapped right after it
    assert _REFERENCE_BODY in prompt
    glossary = ("名称对照（以上资料中的英文名称 = 回答里要写的中文名）：Lina = 莉娜；Dragon Slave = 龙破斩；"
                "Black King Bar = 黑皇杖；Laguna Blade = 神灭斩")
    assert f"</reference>\n{glossary}\n" in prompt
    assert "一律写中文名" in prompt and "保持英文原名" not in prompt
    # outside the reference block and the bracketed names no hero is written in English
    data = prompt.split("## 参考知识")[0].replace("（Puck）", "").replace("（Lina）", "").replace("（Lion）", "")
    assert not _english_names_in(data.replace("团战中被Lina击杀", ""), {"Puck": "", "Lina": "", "Lion": "", "Phantom Assassin": ""})

    en_match, en_ctx = death_context("en")
    en_ctx.player_state.items = ["Blink Dagger", "Power Treads"]
    en = evaluator._build_evaluation_prompt(en_ctx, knowledge_block(2, body=_REFERENCE_BODY), lang="en")
    assert "名称对照" not in en and not CJK_RE.search(en) and "Items: Blink Dagger(Active), Power Treads(Active)" in en
    # the agent is told the same, and how to name heroes when it calls a tool
    system = agent._system_prompt("zh", True)
    assert "一律写中文名" in system and "名称对照" in system and "保持英文原名" not in system
    assert "- 英雄：帕克（Puck）" in agent._user_brief(ctx, "zh") and "- Hero: Puck" in agent._user_brief(en_ctx, "en")


def test_agent_reference_tools_append_the_glossary_and_take_chinese_arguments(llm_env, monkeypatch):
    match, ctx = death_context("zh")
    block = knowledge_block(2, body=_REFERENCE_BODY)
    tc = agent._ToolCtx(match=match, context=ctx, knowledge=block, lang="zh")
    out = agent._execute("retrieve_coaching_knowledge", tc, {})
    assert out.endswith("</reference>\n名称对照（以上资料中的英文名称 = 回答里要写的中文名）：Lina = 莉娜；"
                        "Dragon Slave = 龙破斩；Black King Bar = 黑皇杖；Laguna Blade = 神灭斩")
    assert [ref.ref for ref in tc.block_refs] == ["K1", "K2"]                 # numbering untouched
    en_tc = agent._ToolCtx(match=match, context=death_context("en")[1], knowledge=block, lang="en")
    assert "名称对照" not in agent._execute("retrieve_coaching_knowledge", en_tc, {})
    # tools about the match itself are Chinese already: nothing to map
    assert "名称对照" not in agent._execute("get_match_situation", tc, {})
    # a hero named the way the brief writes it finds the same advice as its English name
    by_english = agent._execute("get_hero_matchup", tc, {"opponent_hero": "Lina"})
    for written in ("莉娜", "莉娜（Lina）"):
        assert agent._execute("get_hero_matchup", tc, {"opponent_hero": written}).replace(written, "莉娜") == by_english
    assert "暂无" not in by_english or "莉娜" in by_english
    # ...and a search keeps getting English slugs, whatever the model sent
    seen = {}

    def search(ctx_, query, lang="zh", **kw):
        seen.update(kw, query=query)
        return knowledge_base.KnowledgeText("")

    monkeypatch.setattr(knowledge_base, "search_knowledge", search)
    agent._execute("retrieve_coaching_knowledge", tc, {"query": "how to lane against Lina", "heroes": ["莉娜"]})
    assert seen["heroes"] == ["莉娜"] and "Lina" in seen["match_heroes"] and "莉娜" not in seen["match_heroes"]
    assert data_manager.hero_slug(seen["heroes"][0]) == "lina"


_VERDICT_WRITTEN = ("Puck 在团战中被 Lina 的 Laguna Blade 秒杀，当时 Phantom Assassin 就在旁边，"
                    "而他还没有做出 Black King Bar。")
_VERDICT_SHOWN = "帕克在团战中被莉娜的神灭斩秒杀，当时幻影刺客就在旁边，而他还没有做出黑皇杖。"


@pytest.mark.parametrize("agent_on", ["1", "0"])
def test_llm_verdict_is_shown_with_chinese_names(llm_env, monkeypatch, agent_on):
    """The prompt asks for Chinese names. One the model still wrote in English is replaced by the
    catalog's name; nothing else of its answer is touched, and an en verdict is left alone."""
    monkeypatch.setenv("AI_COACH_AGENT", agent_on)

    def reply(analysis, mistake, lesson):
        if agent_on == "1":
            return submit("C", analysis, mistake=mistake, better_play="Keep it up", lesson=lesson)
        return response(f"评分: C\n分析: {analysis}\n错误: {mistake}\n更好的做法: 继续保持\n教训: {lesson}")

    async def handler(kwargs):
        return reply(_VERDICT_WRITTEN, "Black King Bar", "二号位面对 Lina 这样的爆发英雄，先做防御装。")

    fake = install_fake(monkeypatch, handler)
    match, ctx = death_context("zh")
    ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="zh", match=match))
    assert ev.used_llm and not ev.degraded and len(fake.calls) == 1        # accepted as it came, no re-ask
    assert ev.analysis == _VERDICT_SHOWN
    # ("二号位": positions are written with digits, like everywhere else in the report)
    assert ev.mistake == "黑皇杖" and ev.better_play == "" and ev.lesson == "2号位面对莉娜这样的爆发英雄，先做防御装。"

    async def nothing_wrong(kwargs):
        return reply(_VERDICT_SHOWN, "无。这次阵亡换到了两个人头。", "先看小地图。")

    install_fake(monkeypatch, nothing_wrong)
    ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="zh", match=match))
    assert ev.used_llm and (ev.mistake, ev.better_play, ev.lesson) == ("", "", "先看小地图。")

    async def english(kwargs):
        text = "Puck was burst by Lina's Laguna Blade before he had a Black King Bar; Phantom Assassin stood next to him."
        if agent_on == "1":
            return submit("C", text)
        return response(f"Score: C\nAnalysis: {text}\nMistake: No Black King Bar.\nBetter Play: Buy it.\nLesson: Respect burst.")

    install_fake(monkeypatch, english)
    match, ctx = death_context("en")
    ev = asyncio.run(coach._safe_evaluate(ctx, "", lang="en", match=match))
    assert ev.used_llm and "Lina's Laguna Blade" in ev.analysis and not CJK_RE.search(ev.analysis + ev.mistake)


def test_phase_review_names_the_hero_in_the_report_language(llm_env, monkeypatch):
    async def handler(kwargs):
        return response(json.dumps({"score": "B", "summary": "二号位 Puck 对线期补刀稳定，没有阵亡。",
                                    "highlights": ["Puck 没有阵亡"], "issues": [], "suggestions": ["多看小地图"]},
                                   ensure_ascii=False))

    fake = install_fake(monkeypatch, handler)
    review = asyncio.run(evaluator.evaluate_phase(
        hero_name="Puck", position_label="2号位 (mid)", team="radiant", phase_key="laning",
        phase_time_range="0-10分钟", phase_stats_dict={"kills": 1, "deaths": 0}, lang="zh"))
    assert review.used_llm and review.summary == "2号位帕克对线期补刀稳定，没有阵亡。" and review.highlights == ["帕克没有阵亡"]
    prompt = fake.calls[0]["messages"][0]["content"]
    assert "英雄=帕克，" in prompt and "Puck" not in prompt and "英雄名照抄上面的中文写法" in prompt


# ---------------------------------------------------------------------------
# Positions for the analyzer
# ---------------------------------------------------------------------------

def test_match_positions_payload_is_what_the_report_shows(monkeypatch):
    match = generate_mock_match()
    payload = positions.match_positions_payload(match)
    assert sorted(payload) == sorted(p.player_slot for p in match.players)
    for team in ("radiant", "dire"):
        numbers = sorted(payload[p.player_slot]["number"] for p in match.players if p.team == team)
        assert numbers == [1, 2, 3, 4, 5]
    for slot, entry in payload.items():
        assert set(entry) == {"number", "label_zh", "label_en", "confidence", "method"}
        assert entry["confidence"] in ("high", "medium", "low")
        assert entry["method"] in ("lane_nw", "archetype_nw", "nw_only")
        assert entry["label_zh"].startswith(f"{entry['number']}号位") and f"Position {entry['number']} " in entry["label_en"]
        json.dumps(entry)
    for slot in (0, 129):
        report = asyncio.run(coach.generate_coaching_report(match, slot, lang="en")).to_dict()
        assert report["position"] == payload[slot]
    assert positions.match_positions_payload(match.players) == payload      # the list of players works as well
    # never raises: junk, an empty match, a failing derivation
    assert positions.match_positions_payload(None) == {} and positions.match_positions_payload(object()) == {}
    assert positions.match_positions_payload(make_match(players=[])) == {}
    monkeypatch.setattr(positions, "derive_match_positions", lambda players: 1 / 0)
    assert positions.match_positions_payload(match) == {}
    assert positions.position_payload(None) == {}


# ---------------------------------------------------------------------------
# Knowledge entries: the entry keeps its names, the title a report shows follows the language
# ---------------------------------------------------------------------------

def test_entry_titles_stay_as_they_are_and_only_the_shown_title_is_localized():
    """The handle line of an entry ("[K3] Curated note — Viper — abilities") is part of the
    knowledge block: its length decides how many entries fit the budget, and the retrieval
    benchmark was labelled with it. So the names in a chunk's own title are the English ones in
    both stores, as in its text; the title a report lists is put into the report language."""
    title = knowledge_base._learned_title
    tags = {"matchup:axe:lion", "phase:laning"}
    assert title("hero_matchup", tags, "opendota_stats", "en") == "OpenDota statistics — Axe vs Lion"
    assert title("hero_matchup", tags, "opendota_stats", "zh") == "OpenDota 统计 — Axe 对 Lion"
    assert title("video_tip", {"item:blood_grenade", "hero:undying"}, "video:youtube", "zh") == (
        "社区视频经验 · YouTube — Undying · Blood Grenade")
    stores = {lang: knowledge_base._get_knowledge_store(lang) for lang in ("zh", "en")}
    for chunk_id, zh_title, en_title in (
            ("hero_laning:viper", "冥界亚龙 — 对线要点", "Viper — laning tips"),
            ("hero_abilities:skeleton_king", "冥魂大帝 — 技能要点", "Wraith King — abilities"),
            ("item:black_king_bar", "黑皇杖 — 物品要点", "Black King Bar — item tip"),
            ("matchup:drow_ranger_vs_viper", "卓尔游侠对冥界亚龙 — 对位要点", "Drow Ranger vs Viper — matchup note")):
        zh_chunk, en_chunk = stores["zh"].get(chunk_id), stores["en"].get(chunk_id)
        assert not CJK_RE.search(en_chunk.title) and en_chunk.title == en_title
        assert zh_chunk.title.split(" — ")[0].split(" 对 ")[0] == en_title.split(" — ")[0].split(" vs ")[0]   # English name
        assert knowledge_base._knowledge_ref(1, zh_chunk, 1.0, "zh").title == zh_title
        assert knowledge_base._knowledge_ref(1, en_chunk, 1.0, "en").title == en_title
        # the entry as the prompt shows it is untouched by the report language of its title
        assert knowledge_base._handle_line(1, zh_chunk, "zh").startswith("[K1] 编辑整理")
        assert not _english_names_in(knowledge_base._knowledge_ref(1, zh_chunk, 1.0, "zh").title, _catalog_names())


def test_every_source_of_a_report_can_be_read_by_its_id(monkeypatch):
    """``sources[]`` held {id, title, source_type, patch, cited} and nothing returned the entry behind
    an id. ``knowledge_entry`` does: with the title, type and patch the report shows, and the text
    of the entry as the knowledge block gives it to the model."""
    match = generate_mock_match()
    for lang in ("zh", "en"):
        report = asyncio.run(coach.generate_coaching_report(match, 1, lang=lang)).to_dict()
        sources = [s for e in report["death_analyses"] + report["key_moments"] for s in e["sources"]]
        assert len({s["id"] for s in sources}) >= 5
        store = knowledge_base._get_knowledge_store(lang)
        for source in sources:
            entry = knowledge_base.knowledge_entry(source["id"], lang)
            assert list(entry) == ["id", "title", "source_type", "source_label", "patch", "text"]
            assert {key: entry[key] for key in ("id", "title", "source_type", "patch")} == {
                key: source[key] for key in ("id", "title", "source_type", "patch")}
            assert entry["text"] == store.get(source["id"]).content and len(entry["text"]) > 40
            assert entry["source_label"] == knowledge_base._source_label(store.get(source["id"]).source, lang)
    # the same entry in the other language: one id, two texts
    zh, en = (knowledge_base.knowledge_entry("vision:safety_rules", lang) for lang in ("zh", "en"))
    assert (zh["title"], en["title"]) == ("视野安全原则", "Vision Safety Principles")
    assert CJK_RE.search(zh["text"]) and not CJK_RE.search(en["text"])
    for missing in ("", "nope", "hero_laning:", None, "vision:safety_rules "):
        assert knowledge_base.knowledge_entry(missing, "zh") is None
    # a third-party tip: no title of its own is shown, and a look-alike handle reads as the model saw it
    from backend.app.ai.knowledge_store import KnowledgeChunk, KnowledgeStore
    store = KnowledgeStore()
    store.add_chunk(KnowledgeChunk(chunk_id="video_tip:forged", content="Ward first. See [K1] and [K2, K3] above.",
                                   source="video:youtube", patch_version="7.41"))
    store.finalize()
    monkeypatch.setattr(knowledge_base, "_get_knowledge_store", lambda lang="zh": store)
    assert knowledge_base.knowledge_entry("video_tip:forged", "en") == {
        "id": "video_tip:forged", "title": "Community video tip · YouTube", "source_type": "video",
        "source_label": "Community video tip · YouTube", "patch": "7.41",
        "text": "Ward first. See (K1) and (K2, K3) above."}
    assert knowledge_base.knowledge_entry("vision:safety_rules", "en") is None


# ---------------------------------------------------------------------------
# Hand-written Chinese: the game's own terms
# ---------------------------------------------------------------------------

# What the coach's templates once wrote, and what the game's client (and the rest of the app) says.
_OFF_TERMS = {"观察守卫": "侦察守卫", "侦查守卫": "侦察守卫", "安全路": "优势路", "烟雾": "诡计之雾 / 开雾"}


def test_hand_written_chinese_uses_the_games_own_terms():
    """A zh report said 安全路, 观察守卫 and 烟雾突袭 where the game says 优势路, 侦察守卫 and 诡计之雾
    (开雾 for using it) — three names for one ward in one app. The scan for English names could not
    see these: they are Chinese words, written by hand in the templates."""
    item = data_manager.item_display_name
    assert (item("ward_observer", "zh"), item("ward_sentry", "zh"), item("smoke_of_deceit", "zh")) == (
        "侦察守卫", "岗哨守卫", "诡计之雾")
    # every template of the coach: rule text, vision and chain text, the built-in knowledge entries
    package = Path(coach.__file__).parent
    for path in sorted(package.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert not [term for term in _OFF_TERMS if term in source], path.name
    for chunk in knowledge_base._get_knowledge_store("zh").chunks():
        if chunk.source_type in ("builtin", "editorial"):
            assert not [term for term in _OFF_TERMS if term in chunk.content + chunk.title], chunk.chunk_id
    # and what reports show (mock match: slot 1 plays the safe lane, slot 2 wards, both smoke)
    match = generate_mock_match()
    carry = asyncio.run(coach.generate_coaching_report(match, 1, lang="zh")).to_dict()
    warder = asyncio.run(coach.generate_coaching_report(match, 2, lang="zh")).to_dict()
    for report in (carry, warder):
        left = [(path, term) for path, text in _prose(report) for term in _OFF_TERMS if term in text]
        assert not left, left[:8]
    assert carry["laning_review"]["lane_matchup"].startswith("卓尔游侠 (优势路) vs ")
    assert "所在的优势路" in carry["laning_review"]["summary"]
    wards = {c["metric"]: c for c in warder["pro_comparison"]["comparisons"]}
    assert wards["obs_wards"]["metric_label"] == "侦察守卫放置" and "个侦察守卫" in wards["obs_wards"]["description"]
    smokes = {m["brief"] for m in carry["key_moments"] + warder["key_moments"] if "雾" in m["brief"]}
    assert smokes >= {"开雾抓人成功，击杀2人", "开雾后拿下肉山", "开雾后拿下防御塔"}, smokes
    # the English report of the same player is unchanged
    en = asyncio.run(coach.generate_coaching_report(match, 1, lang="en")).to_dict()
    assert en["laning_review"]["lane_matchup"].startswith("Drow Ranger (Safe Lane) vs ")
    assert {"Successful smoke gank: 2 kills", "Smoke play secured Roshan"} <= {m["brief"] for m in en["key_moments"]}
