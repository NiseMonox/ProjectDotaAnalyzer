"""Knowledge corpus in KnowledgeChunk format.

Two families of chunks, both bilingual through the ``lang`` parameter:

- Structural chunks (:func:`get_all_builtin_chunks`): laning principles and
  phase objectives per role, vision and death-analysis sections, general
  coaching and early-game principles. The vision / death-analysis text is
  NOT duplicated here: it is split from the canonical sections defined in
  ``knowledge_base`` (``_VISION_SECTIONS`` / ``_DEATH_SECTIONS``).
- Data-driven chunks (:func:`generate_gamedata_chunks`): one overview and
  one laning chunk per hero, one ability chunk per hero, one chunk per
  matchup tip and per item tip — all rendered from the merged data of
  ``data_manager`` whose canonical text is the editorial JSON.

Editorial content (tips, power spikes ...) is written in English; ``lang``
controls the structural labels so the chunk reads naturally in a Chinese
prompt too.
"""
from __future__ import annotations

from .knowledge_store import KnowledgeChunk, buyer_tag, hero_tag, item_tag, matchup_tag
from .knowledge_base import (
    _DEATH_SECTIONS,
    _LANING_PRINCIPLES,
    _LANING_PRINCIPLES_EN,
    _PHASE_OBJECTIVES,
    _PHASE_OBJECTIVES_EN,
    _VISION_SECTIONS,
    render_rules,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _hero_slug(name: str) -> str:
    """Canonical hero tag slug (internal hero name); see knowledge_store.hero_tag."""
    return hero_tag(name).split(":", 1)[1]


_ROLE_LABELS = {
    "zh": {
        "carry": "1号位（大哥）", "mid": "2号位（中单）", "offlane": "3号位（劣势路）",
        "support_4": "4号位（游走辅助）", "support_5": "5号位（硬辅助）", "unknown": "未知",
    },
    "en": {
        "carry": "Position 1 (carry)", "mid": "Position 2 (mid)", "offlane": "Position 3 (offlane)",
        "support_4": "Position 4 (roaming support)", "support_5": "Position 5 (hard support)",
        "unknown": "unknown",
    },
}
_ATTR_LABELS = {
    "zh": {"str": "力量", "agi": "敏捷", "int": "智力", "uni": "全才"},
    "en": {"str": "Strength", "agi": "Agility", "int": "Intelligence", "uni": "Universal"},
}
_PHASE_LABELS = {
    "zh": {"pregame": "开局", "laning": "对线期", "mid_game": "中期", "late_game": "后期"},
    "en": {"pregame": "Pregame", "laning": "Laning", "mid_game": "Mid game", "late_game": "Late game"},
}
_LEVEL1_LABELS = {
    "zh": {"strong": "强", "average": "一般", "weak": "弱"},
    "en": {"strong": "strong", "average": "average", "weak": "weak"},
}


def _lbl(table: dict, lang: str) -> dict:
    return table["en"] if lang == "en" else table["zh"]


def usual_role_text(role, lang: str, lead: str = "") -> str:
    """"usually played as Position 3 (offlane)" for a hero's editorial role.

    The editorial ``role`` is the position a hero is most often picked for,
    not the one a player took in a given match (an Axe can be the mid), so a
    hero text never states it as a plain fact. "" for an unknown role;
    *lead* is put in front of a non-empty result.
    """
    label = _lbl(_ROLE_LABELS, lang).get(str(role or ""))
    if not label or role == "unknown":
        return ""
    return lead + (f"usually played as {label}" if lang == "en" else f"通常打{label}")


# -------------------------------------------------------------------
# 1. Role chunks (laning principles + phase objectives)
# -------------------------------------------------------------------

def _build_role_chunks(lang: str = "zh") -> list[KnowledgeChunk]:
    """One chunk per role for laning principles, one per role+phase for objectives."""
    chunks: list[KnowledgeChunk] = []
    laning_src = _LANING_PRINCIPLES_EN if lang == "en" else _LANING_PRINCIPLES
    for role, text in laning_src.items():
        chunks.append(KnowledgeChunk(
            chunk_id=f"laning_principles:{role}",
            content=text,
            category="role_principles",
            tags=[f"role:{role}", "phase:laning", "mechanic:laning"],
            priority=2,
        ))

    phase_src = _PHASE_OBJECTIVES_EN if lang == "en" else _PHASE_OBJECTIVES
    role_labels = _lbl(_ROLE_LABELS, lang)
    phase_labels = _lbl(_PHASE_LABELS, lang)
    for phase, roles in phase_src.items():
        for role, text in roles.items():
            if lang == "en":
                scope = f"{phase_labels.get(phase, phase)}, {role_labels.get(role, role)}"
                content = f"Phase objectives ({scope}): {text}"
                title = f"Phase objectives ({scope})"
            else:
                scope = f"{phase_labels.get(phase, phase)}，{role_labels.get(role, role)}"
                content = f"阶段目标（{scope}）：{text}"
                title = f"阶段目标（{scope}）"
            chunks.append(KnowledgeChunk(
                chunk_id=f"phase_objectives:{phase}:{role}",
                content=content,
                category="phase_objectives",
                tags=[f"role:{role}", f"phase:{phase}"],
                priority=2,
                title=title,
            ))
    return chunks


# -------------------------------------------------------------------
# 2./3. Vision + death analysis chunks (split from canonical sections)
# -------------------------------------------------------------------

def _section_chunks(sections: list[dict], prefix: str, category: str, lang: str) -> list[KnowledgeChunk]:
    key = "en" if lang == "en" else "zh"
    return [
        KnowledgeChunk(
            chunk_id=f"{prefix}:{section['id']}",
            content=render_rules(section[key]),
            category=category,
            tags=list(section["tags"]),
            priority=section.get("priority", 2),
        )
        for section in sections
    ]


def _build_vision_chunks(lang: str = "zh") -> list[KnowledgeChunk]:
    """Vision principles, one chunk per canonical section."""
    return _section_chunks(_VISION_SECTIONS, "vision", "vision_principles", lang)


def _build_death_analysis_chunks(lang: str = "zh") -> list[KnowledgeChunk]:
    """Death analysis rules, one chunk per canonical section."""
    return _section_chunks(_DEATH_SECTIONS, "death_analysis", "death_analysis", lang)


# -------------------------------------------------------------------
# 4. General coaching chunks (new knowledge not in hardcoded system)
# -------------------------------------------------------------------

# Tags of the teamfight-positioning principle. No phase tags: the principle
# holds for a rune brawl and a lane fight as much as for a late teamfight, and
# with mid/late-game tags the phase factor kept it out of every laning and
# pregame teamfight death (7 of 7 real cases).
_TEAMFIGHT_POSITIONING_TAGS = ["mechanic:teamfight", "mechanic:positioning", "situation:fight_engage",
                               "situation:good_play"]


def _build_general_coaching_chunks(lang: str = "zh") -> list[KnowledgeChunk]:
    """~10-15 general coaching knowledge chunks.

    Game facts in these texts follow the 7.36-7.41f patch notes; the smoke
    radius and Roshan's respawn window are ``{rule}`` placeholders filled
    from the game data and ``game_constants`` (see
    ``knowledge_base.render_rules``). Two facts could not be confirmed from
    the notes and are kept as they were:

    - Roshan's third-kill drop. 7.37: "Refresher Shard and Cheese during
      daytime or Aghanim's Blessing during night"; 7.38: "Roshan no longer
      drops Aghanim's Blessing", without saying what a night-time kill
      drops instead. The text states Cheese + Refresher Shard for every
      third kill.
    - The buyback cost formula (200 + net worth / 13) and its penalties: no
      patch note in that range touches the general rule (the 7.36 / 7.37c
      "buyback costs 15% / 20% more" lines are Doom's Devil's Bargain facet).
    """
    chunks: list[KnowledgeChunk] = []

    if lang == "en":
        # 1. Roshan timing
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:roshan_timing",
            content=render_rules(
                "## Roshan Timing Management\n"
                "- Roshan respawns randomly {roshan_respawn_min_minutes}-{roshan_respawn_max_minutes} minutes after being killed\n"
                "- Aegis of the Immortal lasts 5 minutes, then disappears and fully restores HP and mana\n"
                "- From the second kill Roshan also drops Roshan's Banner; from the third also Cheese and a Refresher Shard. Aghanim's Blessing no longer drops\n"
                "- Record Roshan kill time and set up vision control before the respawn window\n"
                "- After taking Aegis, play aggressively and push; don't waste the Aegis time window"
            ),
            tags=["mechanic:roshan", "phase:mid_game", "phase:late_game"],
            priority=2,
        ))
        # 2. Buyback strategy
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:buyback_strategy",
            content=(
                "## Buyback Strategy\n"
                "- Buyback cost = 200 + net worth / 13; late-game core buyback can cost thousands of gold\n"
                "- Buyback applies a temporary gold and XP penalty, and increases next death respawn time\n"
                "- Core heroes should always reserve buyback gold in late game (net worth - buyback cost = spendable gold)\n"
                "- Buyback timing: defending high ground, contesting Roshan, critical teamfights; don't buyback for meaningless skirmishes\n"
                "- Never die again after buyback; dying after buyback is one of the most severe mistakes"
            ),
            tags=["mechanic:buyback", "phase:mid_game", "phase:late_game", "situation:death"],
            priority=2,
        ))
        # 3. Tower aggro mechanics
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:tower_aggro",
            content=(
                "## Tower Aggro Mechanics\n"
                "- Towers prioritize: enemy units attacking allied heroes > nearest enemy unit\n"
                "- Right-clicking (A-attack) your own creep under tower shifts tower aggro to creeps\n"
                "- Tower aggro switch has a cooldown; cannot be done indefinitely\n"
                "- When tower diving, calculate tower damage (tower damage ramps up per hit); don't underestimate tower DPS\n"
                "- Use tower aggro mechanics to safely last hit and harass under tower"
            ),
            tags=["mechanic:tower_aggro", "mechanic:laning", "phase:laning"],
            priority=3,
        ))
        # 4. Smoke ganking principles
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:smoke_gank",
            content=render_rules(
                "## Smoke Gank Principles\n"
                "- Smoke makes allied units invisible and undetectable by wards; dispels within {smoke_break_radius} range of enemy heroes or buildings\n"
                "- Best smoke timing: after clearing a wave, after placing key wards, when enemy core position is spotted\n"
                "- Recommended smoke group: 3-4 heroes; too many leaves other lanes exposed and wastes farming time\n"
                "- Smoke route should avoid likely enemy Sentry Ward positions\n"
                "- Successful smoke gank should convert into pushing a tower or taking Roshan; don't just kill and go home"
            ),
            tags=["mechanic:smoke", "mechanic:gank", "phase:mid_game"],
            priority=2,
        ))
        # 5. Power rune control and timing
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:rune_control",
            content=(
                "## Rune Control and Timing\n"
                "- Bounty runes spawn every 4 minutes (0:00, 4:00, 8:00...) and give every teammate gold (40 at 0:00, +6 per 4 minutes); no XP\n"
                "- Water runes spawn in the river at 2:00 and 4:00; power runes every 2 minutes from 6:00 (Double Damage, Haste, Invisibility, Illusion, Arcane, Regeneration, Shield)\n"
                "- Shrines of Wisdom activate every 7 minutes: 200 / 500 / 800... XP to the capturing hero and the lowest-XP teammate\n"
                "- Mid should push the wave 10 seconds before even minutes then move to secure the rune; Bottle can be refilled\n"
                "- Haste and Invisibility runes are best for ganking; Double Damage is good for farming or fighting"
            ),
            tags=["mechanic:rune", "phase:laning", "phase:mid_game", "role:mid"],
            priority=3,
        ))
        # 6. Comeback mechanics
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:comeback_mechanics",
            content=(
                "## Comeback Mechanics\n"
                "- Killing high net-worth enemies yields much higher bounty than killing low net-worth enemies\n"
                "- Kill streak bonus: heroes on longer kill streaks provide higher bounty when killed\n"
                "- Team wipe bonus: pushing towers/Roshan after a team wipe can rapidly close the gold gap\n"
                "- The losing team should avoid scattered small skirmishes; concentrate forces for big teamfights for comeback potential\n"
                "- When behind, don't rush fights; use defensive formations and high ground advantage to wait for enemy mistakes"
            ),
            tags=["mechanic:comeback", "situation:behind", "phase:mid_game", "phase:late_game"],
            priority=3,
        ))
        # 7. High ground defense
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:highground_defense",
            content=(
                "## High Ground Defense Principles\n"
                "- High ground defenders have vision advantage and uphill miss chance\n"
                "- When defending high ground, don't chase downhill; this is the most common comeback throw\n"
                "- Use wave-clear abilities to slow enemy push tempo; stall for XP and gold\n"
                "- Don't engage until ultimates are off cooldown; wait for full abilities before seeking counter-initiation\n"
                "- If you have buyback gold, consider fighting one round then buying back for a second engagement"
            ),
            tags=["mechanic:highground", "situation:behind", "phase:late_game"],
            priority=2,
        ))
        # 8. Split push strategy
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:split_push",
            content=(
                "## Split Push Strategy\n"
                "- The core idea is using one hero to tie down multiple enemies, creating space for teammates\n"
                "- Heroes suited for split push: those with escape abilities (e.g. Anti-Mage Blink, Nature's Prophet Teleport) and fast wave clear\n"
                "- Always carry a TP scroll when split pushing; be ready to TP back to defend or join a teamfight\n"
                "- Teammates should apply pressure on other lanes or control vision during split push; don't idle\n"
                "- Split push risk: getting caught can lead to a 4v5 disadvantaged teamfight; requires strong map awareness"
            ),
            tags=["mechanic:split_push", "phase:mid_game", "phase:late_game"],
            priority=3,
        ))
        # 9. TP scroll usage
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:tp_scroll",
            content=(
                "## TP Scroll Usage Principles\n"
                "- All heroes at all stages must carry at least one TP scroll; this is the most basic game habit\n"
                "- TP support: TP-ing in when a teammate is ganked can result in a counter-kill; this is a game-changing play\n"
                "- TP defense: TP-ing back when enemies push towers is the fastest response\n"
                "- TP cooldown management: don't stay in dangerous positions when TP is on cooldown; you've lost your most important escape\n"
                "- Multiple TPs to the same building increase subsequent TP channel time; spread out TP targets"
            ),
            tags=["mechanic:tp_scroll", "phase:laning", "phase:mid_game", "phase:late_game"],
            priority=2,
        ))
        # 10. Stacking and pulling for supports
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:stack_pull",
            content=(
                "## Support Stacking and Pulling\n"
                "- Small pull (pull near camp): Pull at x:15 or x:45 to bring the creep wave back under your tower\n"
                "- Big pull (chain pull far camp): After small pull, immediately pull the far camp to fully deny one creep wave\n"
                "- Stacking: Attack neutral creeps at x:53-x:55 to draw them out of camp; a new wave spawns next minute\n"
                "- Multi-stacking provides massive gold for cores (ancient camp stacking is especially profitable)\n"
                "- Supports should develop a habit of timed pulling and stacking; this is a core laning skill for economy and wave control"
            ),
            tags=["mechanic:stack_pull", "mechanic:laning", "role:support_5", "role:support_4", "phase:laning"],
            priority=2,
        ))
        # 11. Teamfight positioning
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:teamfight_positioning",
            content=(
                "## Teamfight Positioning Principles\n"
                "- Core damage dealers stay in the back line; wait for the front line to absorb abilities before dealing damage\n"
                "- Initiators (e.g. Axe, Tidehunter) need good engagement angles; don't charge head-on\n"
                "- Supports' primary task in teamfights is to use CC abilities and protect cores, not to deal damage\n"
                "- Don't cluster together; easy to be wiped by AOE abilities (e.g. Black Hole, Ravage, Echo Slam)\n"
                "- Before teamfight begins, identify your priority attack target and which teammates to protect"
            ),
            tags=list(_TEAMFIGHT_POSITIONING_TAGS),
            priority=2,
        ))
        # 12. Farming efficiency
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:farming_efficiency",
            content=(
                "## Farming Efficiency Principles\n"
                "- Core heroes should establish a fixed farming route to maximize GPM\n"
                "- Don't wait at one camp for respawn; move to the next camp immediately after clearing\n"
                "- Lane creep gold takes priority over jungle gold (lane gold is higher and doesn't waste travel time)\n"
                "- Heroes with wave-clear abilities should push the lane quickly then move to jungle, not wait on lane"
            ),
            tags=["mechanic:farming", "mechanic:cs", "situation:missed_cs", "role:carry", "role:mid",
                  "phase:laning", "phase:mid_game"],
            priority=3,
        ))
        # 13. Map awareness basics
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:map_awareness",
            content=(
                "## Map Awareness Basics\n"
                "- Check the minimap every 15-20 seconds; develop a fixed map-checking habit\n"
                "- Ping teammates immediately when an enemy hero disappears (especially mid missing)\n"
                "- Observe creep wave push direction to predict where enemies are (wave being pushed = enemy hero on that lane)\n"
                "- From mid-game onward, track enemy economy and item progression to decide if strategy adjustments are needed\n"
                "- Watch enemy hero levels; a level advantage enemy usually wins 1v1"
            ),
            tags=["mechanic:map_awareness", "phase:laning", "phase:mid_game", "phase:late_game"],
            priority=2,
        ))
        # 14. Lane trading (HP / mana trades without a kill)
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:lane_trading",
            content=(
                "## Lane Trading Principles\n"
                "- Trade when the enemy goes for a last hit or has just used their key spell: they cannot hit back without giving something up\n"
                "- Attacking an enemy hero within about 500 range of enemy creeps pulls the creep wave onto you; hit from outside that range or when few enemy creeps are alive\n"
                "- Compare resources before committing: regeneration items, mana for one more spell, levels (an extra skill point at level 2 or 3 wins trades) and which side has the bigger wave\n"
                "- A trade is only good if you lose less than the enemy relative to the regeneration each side has; taking damage without hitting back, with no kill threat, is a lost trade\n"
                "- After a lost trade do not stay in harass range on low HP: use regeneration at once or drop back behind your creeps, low HP in lane invites a dive or a rotation"
            ),
            tags=["situation:bad_trade", "mechanic:trading", "mechanic:laning", "phase:laning"],
            priority=2,
        ))
    else:
        # 1. Roshan timing
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:roshan_timing",
            content=render_rules(
                "## Roshan时间管理\n"
                "- Roshan被击杀后{roshan_respawn_min_minutes}-{roshan_respawn_max_minutes}分钟内随机重生\n"
                "- 不朽盾（Aegis）持续5分钟，过期后自动消失并回满生命和魔法\n"
                "- 从第二次击杀起Roshan还会掉落 Roshan's Banner（战旗）；从第三次起还会掉落奶酪（Cheese）和刷新碎片。A杖祝福不再掉落\n"
                "- 记录Roshan击杀时间，提前在重生窗口前做好视野控制\n"
                "- 拿盾后应积极打架推进，不要浪费盾的时间窗口"
            ),
            tags=["mechanic:roshan", "phase:mid_game", "phase:late_game"],
            priority=2,
        ))
        # 2. Buyback strategy
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:buyback_strategy",
            content=(
                "## 买活策略\n"
                "- 买活金额 = 200 + 净资产/13，后期核心买活金可达数千金\n"
                "- 买活后有临时金钱和经验惩罚，且下次死亡复活时间更长\n"
                "- 核心英雄后期应始终预留买活金（净资产-买活金 = 可花费金额）\n"
                "- 买活时机：守高地、打Roshan、关键团战，不要在无意义的小规模战斗中买活\n"
                "- 买活后绝对不能再死，买活后二次死亡是最严重的失误之一"
            ),
            tags=["mechanic:buyback", "phase:mid_game", "phase:late_game", "situation:death"],
            priority=2,
        ))
        # 3. Tower aggro mechanics
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:tower_aggro",
            content=(
                "## 塔仇恨机制\n"
                "- 防御塔优先攻击：攻击己方英雄的敌方单位 > 最近的敌方单位\n"
                "- 在塔下右键点击（A键攻击）己方小兵可以切换塔仇恨到小兵身上\n"
                "- 塔仇恨切换有冷却时间，不能无限切换\n"
                "- 越塔击杀时注意计算塔的伤害（塔会逐次增加伤害），不要低估塔的输出\n"
                "- 利用塔仇恨机制可以安全地在塔下补刀和骚扰"
            ),
            tags=["mechanic:tower_aggro", "mechanic:laning", "phase:laning"],
            priority=3,
        ))
        # 4. Smoke ganking principles
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:smoke_gank",
            content=render_rules(
                "## 开雾Gank原则\n"
                "- 诡计之雾使友方单位隐身且不会被守卫发现，靠近敌方英雄或建筑{smoke_break_radius}距离时自动消散\n"
                "- 最佳开雾时机：打完一波兵线后、刚做好关键眼位后、对方核心位置被发现时\n"
                "- 开雾人数建议3-4人，太多会让其他路暴露且浪费刷钱时间\n"
                "- 开雾路线避开敌方可能的岗哨守卫位置\n"
                "- 成功的开雾gank可以转化为推塔或Roshan，不要只是杀人后回家"
            ),
            tags=["mechanic:smoke", "mechanic:gank", "phase:mid_game"],
            priority=2,
        ))
        # 5. Power rune control and timing
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:rune_control",
            content=(
                "## 神符控制与时机\n"
                "- 赏金神符每4分钟刷新一次（0:00, 4:00, 8:00...），给每名队友金钱（0:00为40金，每4分钟+6金），不提供经验\n"
                "- 圣水神符在2:00和4:00刷新于河道；强化神符从6:00起每2分钟刷新（双倍伤害、极速、隐身、幻象、秘法、再生、护盾）\n"
                "- 智慧圣坛每7分钟激活一次：占领的英雄和经验最低的队友获得200 / 500 / 800...经验\n"
                "- 中单应在偶数分钟前10秒推线后去抢符，瓶子可以装满\n"
                "- 极速和隐身神符最适合gank，双倍伤害适合刷钱或打架"
            ),
            tags=["mechanic:rune", "phase:laning", "phase:mid_game", "role:mid"],
            priority=3,
        ))
        # 6. Comeback mechanics
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:comeback_mechanics",
            content=(
                "## 翻盘机制\n"
                "- 击杀高净资产敌人获得的赏金远高于击杀低净资产敌人\n"
                "- 连杀奖励：连杀越多的英雄被击杀，提供的赏金越高\n"
                "- 团灭奖励：团灭后推塔/Roshan可以快速缩小经济差距\n"
                "- 劣势方应避免小规模分散战斗，集中力量打大团战才有翻盘机会\n"
                "- 劣势时不要急于打架，利用防守阵型和高地优势等待敌方失误"
            ),
            tags=["mechanic:comeback", "situation:behind", "phase:mid_game", "phase:late_game"],
            priority=3,
        ))
        # 7. High ground defense
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:highground_defense",
            content=(
                "## 守高地原则\n"
                "- 高地防守方拥有高低地视野优势和上坡miss几率\n"
                "- 守高地时不要轻易下高地追击，这是最常见的翻盘失误\n"
                "- 利用兵线清波能力消耗敌方推进节奏，拖延时间获取经验和金钱\n"
                "- 大招CD好之前不要开团，等关键技能齐全后再寻找反打机会\n"
                "- 如果有买活金，可以考虑打一波消耗后等买活再打第二波"
            ),
            tags=["mechanic:highground", "situation:behind", "phase:late_game"],
            priority=2,
        ))
        # 8. Split push strategy
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:split_push",
            content=(
                "## 分推策略\n"
                "- 分推的核心是用一个英雄牵制多个敌人，为队友创造空间\n"
                "- 适合分推的英雄：有逃生技能（如Anti-Mage闪烁、Nature's Prophet传送）、推线速度快\n"
                "- 分推时必须携带TP，随时准备TP回防或支援团战\n"
                "- 队友应在分推时在其他路线施压或控制视野，不要无所事事\n"
                "- 分推的风险：被抓后可能导致4v5劣势团战，需要良好的地图感知"
            ),
            tags=["mechanic:split_push", "phase:mid_game", "phase:late_game"],
            priority=3,
        ))
        # 9. TP scroll usage
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:tp_scroll",
            content=(
                "## TP卷轴使用原则\n"
                "- 所有英雄在所有阶段都必须携带至少一个TP卷轴，这是最基本的游戏习惯\n"
                "- TP支援：队友被gank时TP支援可以反杀，这是改变局势的关键操作\n"
                "- TP回防：敌方推塔时TP回防是最快的响应方式\n"
                "- TP冷却管理：TP刚用完时不要待在危险位置，因为你失去了最重要的逃生手段\n"
                "- 多人TP到同一建筑会增加后续TP的传送时间，注意分散传送目标"
            ),
            tags=["mechanic:tp_scroll", "phase:laning", "phase:mid_game", "phase:late_game"],
            priority=2,
        ))
        # 10. Stacking and pulling for supports
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:stack_pull",
            content=(
                "## 4号位、5号位的拉野与堆野\n"
                "- 小拉（拉近野）：x:15或x:45时拉野，将兵线拉回己方塔下\n"
                "- 大拉（连拉远野）：小拉后立即拉远野，可以完全清掉一波己方小兵\n"
                "- 堆野：x:53-x:55时攻击中立生物将其引出营地，下一分钟会刷新新的一波\n"
                "- 多层堆叠可以为核心提供大量经济（远古野堆叠尤其有利可图）\n"
                "- 4号位和5号位应养成定时拉野堆野的习惯，这是对线期经济和兵线控制的核心技能"
            ),
            tags=["mechanic:stack_pull", "mechanic:laning", "role:support_5", "role:support_4", "phase:laning"],
            priority=2,
        ))
        # 11. Teamfight positioning
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:teamfight_positioning",
            content=(
                "## 团战站位原则\n"
                "- 核心输出英雄站后排，等前排吸收技能后再输出\n"
                "- 先手英雄（如Axe、Tidehunter）需要找到好的切入角度，不要正面硬冲\n"
                "- 4号位和5号位在团战中的首要任务是释放控制技能和保护核心，不是输出\n"
                "- 不要所有人挤在一起，容易被AOE技能团灭（如黑洞、毁灭、回音击）\n"
                "- 团战开始前确定你的优先攻击目标和需要保护的队友"
            ),
            tags=list(_TEAMFIGHT_POSITIONING_TAGS),
            priority=2,
        ))
        # 12. Farming efficiency
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:farming_efficiency",
            content=(
                "## 刷钱效率原则\n"
                "- 核心英雄应建立固定的刷野路线，最大化每分钟金钱收入\n"
                "- 不要在一个营地等刷新，打完一个营地立刻移动到下一个\n"
                "- 兵线上的补刀金优先于野怪金（兵线金更高且不会浪费时间走路）\n"
                "- 带推线技能的英雄应快速清线后进野区，而不是在线上等兵"
            ),
            tags=["mechanic:farming", "mechanic:cs", "situation:missed_cs", "role:carry", "role:mid",
                  "phase:laning", "phase:mid_game"],
            priority=3,
        ))
        # 13. Map awareness basics
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:map_awareness",
            content=(
                "## 地图意识基础\n"
                "- 每15-20秒看一次小地图，养成固定的地图检查习惯\n"
                "- 发现敌方英雄消失立即用信号通知队友（特别是中单消失）\n"
                "- 观察兵线推进方向判断敌方可能在哪里（兵线被推 = 该路有敌方英雄）\n"
                "- 中期开始关注敌方经济和装备进度，判断是否需要调整策略\n"
                "- 注意敌方英雄等级，等级领先的敌人在1v1中通常有优势"
            ),
            tags=["mechanic:map_awareness", "phase:laning", "phase:mid_game", "phase:late_game"],
            priority=2,
        ))
        # 14. Lane trading (HP / mana trades without a kill)
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:lane_trading",
            content=(
                "## 对线换血原则\n"
                "- 在对手去补刀、或刚交出关键技能时换血：此时他还手就必须付出代价\n"
                "- 在敌方小兵约500范围内攻击敌方英雄会吸引兵线仇恨；应在范围外出手，或等敌方小兵数量少时再打\n"
                "- 出手前比较双方资源：回复品、是否还有一个技能的蓝量、等级（2级、3级多一点技能点往往决定换血胜负）以及哪一方兵线更多\n"
                "- 换血是否赚要看双方损失与各自回复能力的对比；只挨打不还手、又没有击杀威胁，就是亏的换血\n"
                "- 换血吃亏后不要残血停留在对方消耗范围内：立刻用回复品或退到己方小兵身后，线上残血会招来越塔或游走"
            ),
            tags=["situation:bad_trade", "mechanic:trading", "mechanic:laning", "phase:laning"],
            priority=2,
        ))

    return chunks


# -------------------------------------------------------------------
# 5. Early game chunks (pregame / rune contest / level 1)
# -------------------------------------------------------------------

def _build_early_game_chunks(lang: str = "zh") -> list[KnowledgeChunk]:
    """3 chunks covering pregame and early-game principles."""
    chunks: list[KnowledgeChunk] = []

    if lang == "en":
        rune_contest = (
            "## Bounty Rune Contest Decision Framework\n"
            "- Bounty rune = ~40 gold per person, not worth dying for (death = 100+ gold + XP + lane time lost)\n"
            "- Observe before entering the river: 3+ enemies at the bounty rune spot -> don't contest without a numbers advantage\n"
            "- Mid should usually go straight to mid lane (mid creep wave value is far higher than a 40 gold bounty rune)\n"
            "- Don't pre-skill abilities -- stay flexible, decide based on whether you need to fight or escape\n"
            "- Plan your retreat route before entering the river\n"
            "- Level 1 death penalty is small (fast respawn), but missing the first creep wave XP can cause a laning disadvantage"
        )
        level1_fight = (
            "## Level 1 Fight Assessment\n"
            "Key factors (by importance):\n"
            "1. Number of CC abilities (stun > slow > silence)\n"
            "2. Number of AOE abilities (multi-target abilities are extremely important in level 1 fights)\n"
            "3. Number of tanky heroes (Strength heroes are generally tougher at level 1)\n"
            "4. Base attack damage and attack speed (melee heroes typically have higher base damage)\n"
            "5. Numbers advantage (the most critical factor in level 1 fights)\n\n"
            "Level 1 strong heroes: Tusk, Clockwerk, Mars, Ogre Magi, Undying, Spirit Breaker, Witch Doctor, Jakiro, Treant Protector\n"
            "Level 1 weak heroes: Invoker, Spectre, Medusa, Anti-Mage (abilities have limited effect at level 1 or are level-dependent)\n\n"
            "Notes:\n"
            "- First Blood is worth ~150 gold, but trading 2 deaths for 1 First Blood is a net loss\n"
            "- Starting items affect level 1 combat: Orb of Venom/Boots = combat advantage; regen items = sustain advantage\n"
            "- All heroes can skill one basic ability at level 1; there is no such thing as 'having no skills'"
        )
        pregame_principles = (
            "## Pregame Principles\n"
            "- Count visible enemy heroes before entering fog/dark areas\n"
            "- Check enemy starting items -- Orb of Venom or Boots means they want to fight\n"
            "- Supports should arrive at bounty rune spot or pull camp before 0:00\n"
            "- Cores decide whether to help contest runes or go directly to lane based on matchup\n"
            "- If your team's level 1 fight ability is weak, don't contest the bounty rune\n"
            "- Prioritize early vision (first ward placement) over bounty runes"
        )
    else:
        rune_contest = (
            "## 赏金符争夺决策框架\n"
            "- 赏金符 = 每人约40金，不值得用命换（死亡 = 100+金 + 经验 + 线上时间损失）\n"
            "- 进入河道前先观察：3+敌人在赏金符点 → 没有人数优势不要争夺\n"
            "- 中单通常应直接去中路（中路兵线价值远高于40金赏金符）\n"
            "- 不要提前学技能——保持灵活性，根据是否需要打架/逃跑再决定\n"
            "- 进入河道前规划好撤退路线\n"
            "- 1级死亡惩罚虽然不大（复活快），但会错过首波兵线经验，可能导致对线劣势"
        )
        level1_fight = (
            "## 1级团战评估要点\n"
            "关键因素（按重要性排序）：\n"
            "1. 控制技能数量（眩晕 > 减速 > 沉默）\n"
            "2. AOE技能数量（影响多人的技能在1级团战中极其重要）\n"
            "3. 肉盾英雄数量（力量英雄通常1级更耐打）\n"
            "4. 基础攻击力和攻击速度（近战英雄通常基础攻击力更高）\n"
            "5. 人数优势（1级团战中人数是最关键的因素）\n\n"
            "1级强势英雄：Tusk, Clockwerk, Mars, Ogre Magi, Undying, Spirit Breaker, Witch Doctor, Jakiro, Treant Protector\n"
            "1级弱势英雄：Invoker, Spectre, Medusa, Anti-Mage（技能在1级效果有限或依赖等级）\n\n"
            "注意：\n"
            "- 先杀（First Blood）价值约150金，但用2条命换1个先杀是亏本的\n"
            "- 出门装备影响1级战斗力：Orb of Venom/鞋子 = 战斗优势；回复品 = 续航优势\n"
            "- 所有英雄1级都可以选择一个基础技能加点，不存在\"没有技能\"的情况"
        )
        pregame_principles = (
            "## 开局阶段原则\n"
            "- 进入暗区前先数清楚可见的敌方英雄人数\n"
            "- 检查敌方出门装备——出了Orb of Venom或鞋子意味着对方想打架\n"
            "- 4号位和5号位应在0:00前到达赏金符点或拉野点\n"
            "- 核心根据对位匹配决定是否帮忙抢符或直接去线上\n"
            "- 如果己方1级团战能力较弱，不要参与赏金符争夺\n"
            "- 优先考虑开局的视野（第一个眼位放置）而不是赏金符"
        )

    chunks.append(KnowledgeChunk(
        chunk_id="early_game:rune_contest_framework",
        content=rune_contest,
        # Not tagged situation:death: it is only relevant to pregame deaths,
        # which the phase tag already captures.
        tags=["phase:pregame", "mechanic:rune"],
        priority=2,
    ))

    chunks.append(KnowledgeChunk(
        chunk_id="early_game:level1_fight_assessment",
        content=level1_fight,
        tags=["phase:pregame", "phase:laning", "mechanic:teamfight"],
        priority=2,
    ))

    chunks.append(KnowledgeChunk(
        chunk_id="early_game:pregame_principles",
        content=pregame_principles,
        tags=["phase:pregame"],
        priority=2,
    ))

    return chunks


# -------------------------------------------------------------------
# Public API (structural chunks)
# -------------------------------------------------------------------

def get_all_builtin_chunks(lang: str = "zh") -> list[KnowledgeChunk]:
    """Return all structural built-in chunks (principles; no hero/item data).

    Hero, matchup, ability and item chunks come from
    :func:`generate_gamedata_chunks` so that the editorial JSON stays the
    single source of truth for them.
    """
    chunks: list[KnowledgeChunk] = []
    chunks.extend(_build_role_chunks(lang))
    chunks.extend(_build_vision_chunks(lang))
    chunks.extend(_build_death_analysis_chunks(lang))
    chunks.extend(_build_general_coaching_chunks(lang))
    chunks.extend(_build_early_game_chunks(lang))
    return chunks


# -------------------------------------------------------------------
# Data-driven chunks (editorial knowledge + game data, all heroes)
# -------------------------------------------------------------------

def _heroes_to_tags(heroes_list: list[str] | None) -> list[str]:
    """Editorial ``heroes`` list -> canonical hero tags.

    ["Axe"] -> ["hero:axe"]; ["all"] -> ["hero:all"]; None -> [].
    """
    if not heroes_list:
        return []
    tags: list[str] = []
    for h in heroes_list:
        tag = "hero:all" if str(h).strip().lower() == "all" else hero_tag(h)
        if tag not in tags:
            tags.append(tag)
    return tags


def _join(values: list[str], lang: str) -> str:
    return ("; " if lang == "en" else "；").join(v for v in values if v)


def _hero_chunks(hero_name: str, info: dict, lang: str) -> list[KnowledgeChunk]:
    """Overview + laning chunk for one hero (only when editorial text exists)."""
    power_spikes = [str(s) for s in info.get("power_spikes") or []]
    weaknesses = [str(w) for w in info.get("weaknesses") or []]
    laning_tips = str(info.get("laning_tips") or "").strip()
    level1 = str(info.get("level1_strength") or "")
    level1_note = str(info.get("level1_note") or "").strip()
    if not (power_spikes or weaknesses or laning_tips or level1_note):
        return []

    slug = _hero_slug(hero_name)
    attr_text = _lbl(_ATTR_LABELS, lang).get(info.get("attribute", ""), info.get("attribute", ""))
    level1_text = _lbl(_LEVEL1_LABELS, lang).get(level1, level1)
    ranged = info.get("attack_type") == "ranged"
    # Hero chunks carry no role tags: knowledge about a hero matters whatever
    # position it is played in (the hero tag decides relevance). Only the
    # laning chunk is phase-scoped.
    tag = hero_tag(hero_name)

    if lang == "en":
        lines = [f"Hero: {hero_name} — {attr_text}, {'ranged' if ranged else 'melee'}"
                 + usual_role_text(info.get("role"), lang, "; ")]
        if power_spikes:
            lines.append(f"Power spikes: {_join(power_spikes, lang)}")
        if weaknesses:
            lines.append(f"Weaknesses: {_join(weaknesses, lang)}")
        if level1 or level1_note:
            lines.append(f"Level-1 fights: {level1_text}" + (f" — {level1_note}" if level1_note else ""))
    else:
        lines = [f"英雄：{hero_name}（{attr_text}，{'远程' if ranged else '近战'}"
                 + usual_role_text(info.get("role"), lang, "；") + "）"]
        if power_spikes:
            lines.append(f"强势期：{_join(power_spikes, lang)}")
        if weaknesses:
            lines.append(f"弱点：{_join(weaknesses, lang)}")
        if level1 or level1_note:
            lines.append(f"1级团战能力：{level1_text}" + (f" — {level1_note}" if level1_note else ""))

    chunks = [KnowledgeChunk(
        chunk_id=f"hero_overview:{slug}",
        content="\n".join(lines),
        category="hero_overview",
        tags=[tag],
        priority=2,
        source="editorial",
        dedupe_key=f"hero_overview:{slug}",
        title=f"{hero_name} — hero overview" if lang == "en" else f"{hero_name} — 英雄概览",
    )]
    if laning_tips:
        label = "Laning tips" if lang == "en" else "对线要点"
        sep = ": " if lang == "en" else "："
        chunks.append(KnowledgeChunk(
            chunk_id=f"hero_laning:{slug}",
            content=f"{hero_name} — {label}{sep}{laning_tips}",
            category="hero_laning",
            tags=[tag, "phase:laning", "mechanic:laning"],
            priority=2,
            source="editorial",
            dedupe_key=f"hero_laning:{slug}",
            title=f"{hero_name} — {label.lower() if lang == 'en' else label}",
        ))
    return chunks


_DTYPE_LABELS = {
    "zh": {"magical": "魔法", "physical": "物理", "pure": "纯粹"},
    "en": {"magical": "Magical", "physical": "Physical", "pure": "Pure"},
}


def _ability_chunk(hero_name: str, hero_abilities: dict[str, dict], lang: str) -> KnowledgeChunk | None:
    """One chunk listing a hero's abilities with their coaching tips."""
    dtype_labels = _lbl(_DTYPE_LABELS, lang)
    lines: list[str] = []
    for ability_name, data in hero_abilities.items():
        text = str(data.get("coaching_tip") or data.get("description") or "").strip()
        if not text:
            continue
        flags: list[str] = []
        if data.get("is_ultimate"):
            flags.append("Ultimate" if lang == "en" else "大招")
        if dtype_labels.get(data.get("damage_type", "")):
            flags.append(dtype_labels[data["damage_type"]])
        if data.get("pierces_bkb"):
            flags.append("pierces BKB" if lang == "en" else "无视BKB")
        if data.get("type") == "passive":
            flags.append("Passive" if lang == "en" else "被动")
        flag_text = f" ({', '.join(flags)})" if lang == "en" and flags else (
            f"（{'，'.join(flags)}）" if flags else "")
        lines.append(f"- {ability_name}{flag_text}: {text}")
        if len(lines) >= 6:
            break
    if not lines:
        return None
    slug = _hero_slug(hero_name)
    header = f"{hero_name} abilities:" if lang == "en" else f"{hero_name} 技能要点："
    return KnowledgeChunk(
        chunk_id=f"hero_abilities:{slug}",
        content=header + "\n" + "\n".join(lines),
        category="ability_coaching",
        tags=[hero_tag(hero_name), "mechanic:skill_usage"],
        priority=2,
        source="editorial",
        dedupe_key=f"hero_abilities:{slug}",
        title=f"{hero_name} — abilities" if lang == "en" else f"{hero_name} — 技能要点",
    )


def _matchup_chunks(matchup_tips: dict[str, dict[str, str]], lang: str) -> list[KnowledgeChunk]:
    """One chunk per editorial matchup tip.

    A tip is written for the player of *hero_a* ("you"). Queries also fetch
    it for the *hero_b* player (the opponent's side is often the only tip
    for the pair), so the text itself says whom it addresses — otherwise a
    model coaching *hero_b* reads the opponent's advice as its own.
    """
    chunks: list[KnowledgeChunk] = []
    for hero_a in sorted(matchup_tips):
        for hero_b, tip in sorted(matchup_tips[hero_a].items()):
            if not tip:
                continue
            slug_a, slug_b = _hero_slug(hero_a), _hero_slug(hero_b)
            tag = matchup_tag(hero_a, hero_b)
            if lang == "en":
                content = (f"Matchup tip written for the {hero_a} player (\"you\" = {hero_a}) "
                           f"against {hero_b}: {tip}")
            else:
                content = (f"写给 {hero_a} 玩家的对位要点（文中的 you 指 {hero_a}），"
                           f"对手 {hero_b}：{tip}")
            chunks.append(KnowledgeChunk(
                chunk_id=f"matchup:{slug_a}_vs_{slug_b}",
                content=content,
                category="hero_matchup",
                # Only the pair tag: a matchup tip is irrelevant unless both
                # heroes are in the game (queries add both directions).
                tags=[tag],
                priority=2,
                source="editorial",
                dedupe_key=tag,
                title=(f"{hero_a} vs {hero_b} — matchup note" if lang == "en"
                       else f"{hero_a} 对 {hero_b} — 对位要点"),
            ))
    return chunks


def _item_chunks(items: dict[str, dict], lang: str) -> list[KnowledgeChunk]:
    """One chunk per item with a coaching tip, plus per-hero counter tips."""
    chunks: list[KnowledgeChunk] = []
    for item_name in sorted(items):
        data = items[item_name]
        tip = str(data.get("coaching_tip") or "").strip()
        key = str(data.get("key") or "")
        if not key:
            continue
        tag = item_tag(key)
        slug = tag.split(":", 1)[1]
        if tip:
            desc = str(data.get("description") or "").strip()
            cost = data.get("cost")
            active = bool(data.get("is_active"))
            if lang == "en":
                meta = ", ".join(x for x in (
                    f"{cost} gold" if isinstance(cost, (int, float)) and cost else "",
                    "active" if active else "passive",
                ) if x)
                content = f"Item: {item_name} ({meta})" + (f" — {desc}" if desc else "") + f"\nTip: {tip}"
            else:
                meta = "，".join(x for x in (
                    f"{cost}金" if isinstance(cost, (int, float)) and cost else "",
                    "主动" if active else "被动",
                ) if x)
                content = f"物品：{item_name}（{meta}）" + (f" — {desc}" if desc else "") + f"\n要点：{tip}"
            # Typical buyers go under "buyer:", not "hero:": the tip is about
            # the item, and an enemy on the list must not make it eligible.
            buyer_tags = [buyer_tag(t.split(":", 1)[1])
                          for t in _heroes_to_tags(data.get("heroes")) if t != "hero:all"]
            chunks.append(KnowledgeChunk(
                chunk_id=f"item:{slug}",
                content=content,
                category="item_coaching",
                tags=[tag, "mechanic:item_usage", *buyer_tags],
                priority=2,
                source="editorial",
                dedupe_key=f"item:{slug}",
                title=f"{item_name} — item tip" if lang == "en" else f"{item_name} — 物品要点",
            ))
        counters = data.get("counter_tips")
        if isinstance(counters, dict):
            for hero, counter_tip in sorted(counters.items()):
                if not counter_tip:
                    continue
                hslug = _hero_slug(hero)
                content = (f"{item_name} vs {hero}: {counter_tip}" if lang == "en"
                           else f"{item_name} 对抗 {hero}：{counter_tip}")
                chunks.append(KnowledgeChunk(
                    chunk_id=f"item_counter:{slug}:{hslug}",
                    content=content,
                    category="item_coaching",
                    tags=[tag, hero_tag(hero)],
                    priority=2,
                    source="editorial",
                    dedupe_key=f"item_counter:{slug}:{hslug}",
                    title=f"{item_name} vs {hero}" if lang == "en" else f"{item_name} 对抗 {hero}",
                ))
    return chunks


def generate_gamedata_chunks(
    abilities: dict[str, dict[str, dict]],
    items: dict[str, dict],
    heroes: dict[str, dict],
    lang: str = "zh",
    matchup_tips: dict[str, dict[str, str]] | None = None,
) -> list[KnowledgeChunk]:
    """Generate retrievable chunks from the merged game data.

    - per hero: overview + laning chunk (editorial), ability chunk
    - per matchup tip: one chunk (``matchup:<a>_vs_<b>``)
    - per item with a tip: one chunk (``item:<key>``) + counter-tip chunks

    Inputs are the ``data_manager.load_*`` results (or any dicts of the same
    shape); hero / item names may be display names, slugs or npc names.
    """
    chunks: list[KnowledgeChunk] = []
    for hero_name in sorted(heroes):
        chunks.extend(_hero_chunks(hero_name, heroes[hero_name], lang))
    for hero_name in sorted(abilities):
        chunk = _ability_chunk(hero_name, abilities[hero_name], lang)
        if chunk is not None:
            chunks.append(chunk)
    if matchup_tips:
        chunks.extend(_matchup_chunks(matchup_tips, lang))
    if items:
        chunks.extend(_item_chunks(items, lang))
    return chunks
