"""Built-in knowledge corpus migrated into KnowledgeChunk format.

Converts hardcoded hero data, matchup tips, laning principles, phase
objectives, vision principles, and death analysis rules from
knowledge_base.py into discrete, tagged KnowledgeChunks suitable for
retrieval-augmented coaching prompts.

Supports Chinese (zh-CN) and English (en) via ``lang`` parameter.
"""
from __future__ import annotations

from .knowledge_store import KnowledgeChunk
from .knowledge_base import (
    _FALLBACK_HERO_DATA as HERO_DATA,
    _FALLBACK_MATCHUP_TIPS as _MATCHUP_TIPS,
    _FALLBACK_MATCHUP_TIPS_EN as _MATCHUP_TIPS_EN,
    _LANING_PRINCIPLES,
    _LANING_PRINCIPLES_EN,
    _PHASE_OBJECTIVES,
    _PHASE_OBJECTIVES_EN,
    _DEATH_ANALYSIS_RULES,
    _DEATH_ANALYSIS_RULES_EN,
    _VISION_PRINCIPLES,
    _VISION_PRINCIPLES_EN,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _hero_slug(name: str) -> str:
    """Lowercase, spaces to underscores, strip apostrophes/hyphens."""
    return name.lower().replace(" ", "_").replace("'", "").replace("-", "_")


def _role_tag(role: str) -> str:
    """Normalise role string into a tag value."""
    return role.lower().replace(" ", "_")


# -------------------------------------------------------------------
# 1. Hero chunks
# -------------------------------------------------------------------

def _build_hero_chunks(lang: str = "zh") -> list[KnowledgeChunk]:
    """Two chunks per hero: overview + laning.

    Editorial content (power_spikes, weaknesses, laning_tips) is now always
    in English. The lang parameter only controls structural labels/headers.
    """
    chunks: list[KnowledgeChunk] = []

    for hero_name, data in HERO_DATA.items():
        slug = _hero_slug(hero_name)
        role = data.get("role", "unknown")
        attribute = data.get("attribute", "unknown")
        attack_type = data.get("attack_type", "unknown")
        # Editorial content is now English — no _en suffix needed
        power_spikes = data.get("power_spikes", [])
        weaknesses = data.get("weaknesses", [])
        laning_tips = data.get("laning_tips", "")

        if lang == "en":
            attr_map = {"str": "Strength", "agi": "Agility", "int": "Intelligence", "uni": "Universal"}
            attack_label = "Ranged" if attack_type == "ranged" else "Melee"
            spikes_text = ", ".join(power_spikes) if power_spikes else "N/A"
            weak_text = ", ".join(weaknesses) if weaknesses else "N/A"
            overview_content = (
                f"Hero: {hero_name}\n"
                f"Role: {role}\n"
                f"Attribute: {attr_map.get(attribute, attribute)} ({attack_label})\n"
                f"Power Spikes: {spikes_text}\n"
                f"Weaknesses: {weak_text}"
            )
        else:
            attr_map = {"str": "力量", "agi": "敏捷", "int": "智力", "uni": "全能"}
            attr_cn = attr_map.get(attribute, attribute)
            attack_cn = "远程" if attack_type == "ranged" else "近战"
            # Content is English even in zh mode (editorial is always English)
            spikes_text = ", ".join(power_spikes) if power_spikes else "N/A"
            weak_text = ", ".join(weaknesses) if weaknesses else "N/A"
            overview_content = (
                f"英雄：{hero_name}\n"
                f"定位：{role}\n"
                f"属性：{attr_cn}（{attack_cn}）\n"
                f"Power Spikes: {spikes_text}\n"
                f"Weaknesses: {weak_text}"
            )

        overview_tags = [f"hero:{slug}", f"role:{_role_tag(role)}"]
        # Add phase tags based on power spikes keywords (English keywords now)
        if any("Level 1" in s or "Level 2" in s or "Level 3" in s for s in power_spikes):
            overview_tags.append("phase:laning")
        if any("BKB" in s or "Blink" in s for s in power_spikes):
            overview_tags.append("phase:mid_game")
        if any("Refresher" in s or "Butterfly" in s or "Daedalus" in s for s in power_spikes):
            overview_tags.append("phase:late_game")

        chunks.append(KnowledgeChunk(
            chunk_id=f"hero_overview:{slug}",
            content=overview_content,
            tags=overview_tags,
            priority=2,
            token_estimate=len(overview_content) // 2,
        ))

        # -- Laning chunk --
        if laning_tips:
            if lang == "en":
                laning_content = f"Hero: {hero_name}\nLaning Tips: {laning_tips}"
            else:
                laning_content = f"英雄：{hero_name}\nLaning Tips: {laning_tips}"
            laning_tags = [
                f"hero:{slug}",
                f"role:{_role_tag(role)}",
                "phase:laning",
                "mechanic:laning",
            ]
            chunks.append(KnowledgeChunk(
                chunk_id=f"hero_laning:{slug}",
                content=laning_content,
                tags=laning_tags,
                priority=2,
                token_estimate=len(laning_content) // 2,
            ))

    return chunks


# -------------------------------------------------------------------
# 2. Matchup chunks
# -------------------------------------------------------------------

def _build_matchup_chunks(lang: str = "zh") -> list[KnowledgeChunk]:
    """One chunk per matchup pair.

    Matchup tips are now always English. The lang parameter only controls
    the structural label (vs / 对阵).
    """
    chunks: list[KnowledgeChunk] = []

    # Use English matchup tips (primary source now)
    matchup_src = _MATCHUP_TIPS_EN
    for hero_name, opponents in matchup_src.items():
        slug1 = _hero_slug(hero_name)
        for opp_name, tip in opponents.items():
            slug2 = _hero_slug(opp_name)
            if lang == "en":
                content = f"{hero_name} vs {opp_name}: {tip}"
            else:
                content = f"{hero_name} 对阵 {opp_name}: {tip}"
            chunks.append(KnowledgeChunk(
                chunk_id=f"matchup:{slug1}_vs_{slug2}",
                content=content,
                tags=[
                    f"matchup:{slug1}:{slug2}",
                    f"hero:{slug1}",
                    f"hero:{slug2}",
                ],
                priority=2,
                token_estimate=len(content) // 2,
            ))

    return chunks


# -------------------------------------------------------------------
# 3. Role chunks (laning principles + phase objectives)
# -------------------------------------------------------------------

def _build_role_chunks(lang: str = "zh") -> list[KnowledgeChunk]:
    """One chunk per role for laning principles, one per role+phase for objectives."""
    chunks: list[KnowledgeChunk] = []

    # Laning principles per role
    laning_src = _LANING_PRINCIPLES_EN if lang == "en" else _LANING_PRINCIPLES
    for role, text in laning_src.items():
        chunks.append(KnowledgeChunk(
            chunk_id=f"laning_principles:{role}",
            content=text,
            tags=[f"role:{role}", "phase:laning", "mechanic:laning"],
            priority=2,
            token_estimate=len(text) // 2,
        ))

    # Phase objectives per role + phase
    phase_src = _PHASE_OBJECTIVES_EN if lang == "en" else _PHASE_OBJECTIVES
    for phase, roles in phase_src.items():
        for role, text in roles.items():
            if lang == "en":
                content = f"Phase: {phase}, Role: {role}\nObjectives: {text}"
            else:
                content = f"阶段：{phase}，定位：{role}\n目标：{text}"
            chunks.append(KnowledgeChunk(
                chunk_id=f"phase_objectives:{phase}:{role}",
                content=content,
                tags=[f"role:{role}", f"phase:{phase}"],
                priority=2,
                token_estimate=len(content) // 2,
            ))

    return chunks


# -------------------------------------------------------------------
# 4. Vision chunks (split into 7 sub-chunks)
# -------------------------------------------------------------------

def _build_vision_chunks(lang: str = "zh") -> list[KnowledgeChunk]:
    """Split _VISION_PRINCIPLES into 7 topical sub-chunks."""
    chunks: list[KnowledgeChunk] = []

    if lang == "en":
        # Sub-chunk 1: Basic vision mechanics
        basic_vision = (
            "## Basic Vision Mechanics\n"
            "- Observer Ward: Provides 1600 range ground vision, lasts 360 seconds (6 minutes), no limit but shop restock has cooldown\n"
            "- Sentry Ward: Does not provide ground vision, only 850 range true sight (detection/dewarding), used with Observer Wards\n"
            "- Hero vision: 1800 range during day, 800 range at night. Some heroes are exceptions (e.g. Night Stalker has 1800 night vision)\n"
            "- Towers: 1900 range vision during day, only 800 at night. Destroyed towers permanently lose fixed vision in that area\n"
            "- Day/Night cycle: Alternates every 5 minutes (0:00-5:00 day, 5:00-10:00 night, and so on)\n"
            "- Night vision reduction: Night vision is about 44% of daytime (800/1800), effective control area is only about 19% of daytime\n"
            "- High/low ground vision: Low ground units cannot see units on high ground; high ground has natural vision advantage\n"
            "- Tree obstruction: Trees block vision propagation; dense forest areas are natural fog zones"
        )
        # Sub-chunk 2: Safety rules
        safety_rules = (
            "## Vision Safety Principles\n"
            "- When 3+ enemy heroes disappear from the map for over 15 seconds, immediately retreat to a safe area (under tower or warded zone)\n"
            "- Replace expired Observer Wards within 10 seconds, especially at river, Roshan pit, and jungle entrances\n"
            "- Moving through unwarded areas at night is extremely dangerous -- 800 vision means enemies can ambush from very close range\n"
            "- Before pushing, ensure flanks have ward coverage; place wards on both sides of the push route at least 30 seconds ahead\n"
            "- If you don't know enemy positions, assume they are in the most dangerous spot (on your gank route)\n"
            "- Support heroes should secure key area vision 60 seconds before a teamfight, not after it has already started\n"
            "- Before entering enemy jungle to farm, confirm the positions of at least 3 enemy heroes"
        )
        # Sub-chunk 3: Laning phase wards
        laning_wards = (
            "## Laning Phase Vision (0-10 minutes)\n"
            "- River ward: Controls runes and mid gank routes, most important ward during laning\n"
            "- Enemy pull camp ward: Prevents enemy support from pulling, keeps creep wave in favorable position\n"
            "- Rune ward: Place before even minutes to get early rune information\n"
            "- Priority: River > Pull camp > Aggressive vision\n"
            "- Laning phase wards should focus on defense and information; don't risk deep aggressive wards"
        )
        # Sub-chunk 4: Mid-game wards
        midgame_wards = (
            "## Mid-Game Vision (10-25 minutes)\n"
            "- Enemy jungle ward: Track enemy core farming routes, provide information for ganks\n"
            "- Roshan pit vision: Control Roshan contest information, mid-game Roshan is very valuable\n"
            "- Push route flank wards: Prevent flanking during pushes, place at least 30 seconds ahead\n"
            "- Priority: Roshan > Enemy jungle > Push routes\n\n"
            "## Late-Game Vision (25+ minutes)\n"
            "- High ground approach wards: Essential for attacking high ground, defenders need to deward\n"
            "- Roshan monitoring ward: Late-game Roshan is extremely valuable (Aegis + Cheese + Refresher Shard)\n"
            "- Enemy base entrance ward: Monitor enemy movement routes\n"
            "- Own jungle defense ward: Prevent enemy from harassing your farming\n"
            "- Priority: Roshan > High ground > Base entrance"
        )
        # Sub-chunk 5: Role responsibilities for vision
        role_vision = (
            "## Vision Responsibilities by Role\n"
            "- Position 5 (Hard Support): Primary ward buyer, maintain vision network in key areas. Goal: always have 3-4 Observer Wards active on the map\n"
            "- Position 4 (Roaming Support): Supplementary ward buyer, responsible for aggressive vision (deep enemy wards, dewarding). Prioritize deep wards after Smoke\n"
            "- Core positions (1/2/3): Don't need to buy wards, but must read vision information to make safe decisions. Must react when enemies disappear\n"
            "- All positions: Obligation to deward enemy vision when passing through Sentry Ward coverage areas; this is a team responsibility"
        )
        # Sub-chunk 6: Death-related vision checklist
        death_vision = (
            "## Vision and Death Correlation\n"
            "Most non-teamfight deaths (~70%) are directly related to lack of vision.\n\n"
            "### Pre-Death Vision Checklist\n"
            "1. Are there wards covering the surrounding area? Is the coverage radius sufficient?\n"
            "2. How many enemy hero positions are unknown? (3+ unknown = extreme danger)\n"
            "3. How long have missing enemies been gone? (>15s = possibly en route to gank)\n"
            "4. Is it day or night? (Night doubles the danger factor)\n"
            "5. Are there allied heroes nearby who can assist? (Within TP range? Walking range?)\n"
            "6. How far is the nearest safe area (tower/fountain)? (>10s travel = dangerous)\n"
            "7. Does the enemy have Smoke available? (Smoke bypasses ward vision)"
        )
        # Sub-chunk 7: Key numbers reference
        key_numbers = (
            "## Vision Key Numbers Reference\n"
            "- Observer Ward vision: 1600 range, lasts 360 seconds (6 minutes)\n"
            "- Sentry Ward true sight range: 850, lasts 420 seconds (7 minutes)\n"
            "- Hero daytime vision: 1800 range\n"
            "- Hero nighttime vision: 800 range\n"
            "- Tower daytime vision: 1900 range\n"
            "- Tower nighttime vision: 800 range\n"
            "- Day/Night cycle period: every 5 minutes\n"
            "- Smoke of Deceit duration: 35 seconds\n"
            "- Smoke detection distance: 1025 range (auto-dispels near enemy heroes/towers)"
        )
    else:
        # Sub-chunk 1: Basic vision mechanics
        basic_vision = (
            "## 视野基础机制\n"
            "- 观察守卫（Observer Ward）：提供1600距离的地面视野，持续360秒（6分钟），无数量限制但商店补货有冷却时间\n"
            "- 岗哨守卫（Sentry Ward）：不提供地面视野，仅提供850距离的真视（反隐/排眼），与观察守卫配合使用\n"
            "- 英雄视野：白天1800距离，夜间800距离。部分英雄例外（如夜魇猎人夜间视野1800）\n"
            "- 防御塔：白天提供1900距离视野，夜间仅800距离。塔被摧毁后该区域永久失去固定视野\n"
            "- 昼夜交替：每5分钟交替一次（0:00-5:00白天，5:00-10:00夜间，以此类推）\n"
            "- 夜间视野衰减：夜间视野约为白天的44%（800/1800），有效控制区域仅为白天的约19%\n"
            "- 高低地视野：低地单位无法看到高地上的单位，高地具有天然视野优势\n"
            "- 树木遮挡：树木会阻挡视野传播，密林区域是天然的雾区"
        )
        # Sub-chunk 2: Safety rules
        safety_rules = (
            "## 视野安全原则\n"
            "- 当3名以上敌方英雄从地图消失超过15秒时，应立即撤退到安全区域（塔下或有视野覆盖区）\n"
            "- 观察守卫到期后应在10秒内补放新眼，特别是河道、肉山坑、己方野区入口等关键位置\n"
            "- 夜间在没有视野的区域移动极其危险——视野仅800意味着敌方可以从极近距离发动突袭\n"
            "- 推进前必须确保侧翼有视野覆盖，至少提前30秒在推进路线两侧插眼\n"
            "- 如果不知道敌方位置，默认假设他们在最危险的位置（你当前位置的gank路线上）\n"
            "- 支援型英雄应在团战前60秒确保关键区域视野，而不是团战已经开始后再去插眼\n"
            "- 进入敌方野区刷野前，必须确认至少3名敌方英雄的位置"
        )
        # Sub-chunk 3: Laning phase wards
        laning_wards = (
            "## 对线期视野（0-10分钟）\n"
            "- 河道眼：控制神符和中路gank路线，对线期最重要的眼位\n"
            "- 对面拉野点眼：阻止敌方辅助拉野，维持兵线在有利位置\n"
            "- 符点眼：偶数分钟前放置，提前获取神符信息\n"
            "- 优先级排序：河道 > 拉野点 > 入侵性视野\n"
            "- 对线期眼位应以防守和信息为主，不要冒险深入插进攻眼"
        )
        # Sub-chunk 4: Mid-game wards
        midgame_wards = (
            "## 中期视野（10-25分钟）\n"
            "- 入侵敌方野区眼：掌握敌方核心刷野路线，为抓人提供信息\n"
            "- 肉山坑视野：控制Roshan争夺信息，中期Roshan价值很高\n"
            "- 推进路线侧翼眼：防止推进时被包夹，至少提前30秒放置\n"
            "- 优先级排序：肉山 > 入侵野区 > 推进路线\n\n"
            "## 后期视野（25分钟+）\n"
            "- 高地前方眼：攻方推高地必备，守方需要反掉\n"
            "- Roshan持续监控眼：后期Roshan价值极高（不朽盾+奶酪+刷新碎片）\n"
            "- 敌方基地入口眼：监控敌方出击路线\n"
            "- 己方野区防守眼：防止敌方深入骚扰刷野\n"
            "- 优先级排序：Roshan > 高地 > 基地入口"
        )
        # Sub-chunk 5: Role responsibilities for vision
        role_vision = (
            "## 不同位置的视野责任\n"
            "- 5号位（硬辅助）：主要买眼责任，维持关键区域视野网络。目标：地图上始终有3-4个观察守卫存活\n"
            "- 4号位（游走辅助）：补充买眼，负责侵略性视野（深入敌方区域插眼、反眼）。Smoke后优先放置深入眼\n"
            "- 核心位（1/2/3号位）：不需要主动买眼，但必须读取视野信息做出安全决策。看到敌方消失必须有反应\n"
            "- 所有位置：路过岗哨守卫覆盖区域时有义务排除敌方视野（反眼），这是团队责任"
        )
        # Sub-chunk 6: Death-related vision checklist
        death_vision = (
            "## 死亡与视野的关系\n"
            "大多数非团战死亡（约70%）都与视野缺失直接相关。\n\n"
            "### 死亡前视野检查清单\n"
            "1. 是否有守卫覆盖周围区域？覆盖半径是否足够？\n"
            "2. 有多少敌方英雄位置未知？（≥3人未知 = 极度危险）\n"
            "3. 未知敌人消失了多久？（>15秒 = 可能正在gank路上）\n"
            "4. 当前是白天还是夜间？（夜间危险系数翻倍）\n"
            "5. 附近是否有友方英雄可以支援？（TP距离内？走路距离内？）\n"
            "6. 距离最近的安全区域（塔/泉水）有多远？（>10秒路程 = 危险）\n"
            "7. 敌方是否有烟雾可用？（烟雾可以绕过守卫视野）"
        )
        # Sub-chunk 7: Key numbers reference
        key_numbers = (
            "## 视野关键数值速查\n"
            "- Observer Ward 视野：1600距离，持续360秒（6分钟）\n"
            "- Sentry Ward 真视范围：850距离，持续420秒（7分钟）\n"
            "- 英雄白天视野：1800距离\n"
            "- 英雄夜间视野：800距离\n"
            "- 防御塔白天视野：1900距离\n"
            "- 防御塔夜间视野：800距离\n"
            "- 昼夜交替周期：每5分钟\n"
            "- Smoke of Deceit 持续：35秒\n"
            "- Smoke 探测距离：1025距离（靠近敌方英雄/塔时自动消散）"
        )

    chunks.append(KnowledgeChunk(
        chunk_id="vision:basic_mechanics",
        content=basic_vision,
        tags=["mechanic:vision", "mechanic:ward"],
        priority=2,
        token_estimate=len(basic_vision) // 2,
    ))

    chunks.append(KnowledgeChunk(
        chunk_id="vision:safety_rules",
        content=safety_rules,
        tags=["mechanic:vision", "situation:danger", "situation:gank"],
        priority=1,
        token_estimate=len(safety_rules) // 2,
    ))

    chunks.append(KnowledgeChunk(
        chunk_id="vision:laning_wards",
        content=laning_wards,
        tags=["mechanic:vision", "mechanic:ward", "phase:laning"],
        priority=2,
        token_estimate=len(laning_wards) // 2,
    ))

    chunks.append(KnowledgeChunk(
        chunk_id="vision:midlate_wards",
        content=midgame_wards,
        tags=["mechanic:vision", "mechanic:ward", "phase:mid_game", "phase:late_game"],
        priority=2,
        token_estimate=len(midgame_wards) // 2,
    ))

    chunks.append(KnowledgeChunk(
        chunk_id="vision:role_responsibilities",
        content=role_vision,
        tags=[
            "mechanic:vision", "mechanic:ward",
            "role:support_5", "role:support_4",
            "role:carry", "role:mid", "role:offlane",
        ],
        priority=2,
        token_estimate=len(role_vision) // 2,
    ))

    chunks.append(KnowledgeChunk(
        chunk_id="vision:death_checklist",
        content=death_vision,
        tags=["mechanic:vision", "situation:death", "situation:danger"],
        priority=1,
        token_estimate=len(death_vision) // 2,
    ))

    chunks.append(KnowledgeChunk(
        chunk_id="vision:key_numbers",
        content=key_numbers,
        tags=["mechanic:vision", "reference"],
        priority=2,
        token_estimate=len(key_numbers) // 2,
    ))

    return chunks


# -------------------------------------------------------------------
# 5. Death analysis chunks (split into 6 sub-chunks)
# -------------------------------------------------------------------

def _build_death_analysis_chunks(lang: str = "zh") -> list[KnowledgeChunk]:
    """Split _DEATH_ANALYSIS_RULES into topical sub-chunks."""
    chunks: list[KnowledgeChunk] = []

    if lang == "en":
        vision_dim = (
            "## Death Analysis: Vision and Information Dimension\n"
            "- Were there allied wards covering the area before death? If not, this is a vision-related death\n"
            "- Could enemy hero positions be seen on the minimap? How many enemies' information was missing?\n"
            "- Did the enemy use Smoke? If so, this is a high-quality gank and reduces player responsibility\n\n"
            "### Vision Dimension Scoring Criteria\n"
            "- S: Operating under full vision coverage, wards active even at night, proactively avoiding potential danger using vision info\n"
            "- A: Basic vision coverage present, maintained some vigilance toward missing enemies, chose conservative routes when vision was lacking\n"
            "- B: Partial vision coverage but incomplete, did not fully utilize available vision information\n"
            "- C: Took risks with insufficient vision, ignored enemy disappearance signals, did not replace expired wards promptly\n"
            "- D: Pushed deep into completely unwarded areas, multiple enemies missing for extended periods without retreating, solo movement in unwarded areas at night"
        )
        positioning_dim = (
            "## Death Analysis: Positioning Dimension\n"
            "- Was the death location in a dangerous area (enemy high ground, unwarded jungle, river)?\n"
            "- Was the player pushing or farming alone without teammates in support range?\n"
            "- Was the creep wave pushed too far? Was the player caught being greedy for last hits?\n\n"
            "### Team Coordination Dimension\n"
            "- Were teammate positions reasonable? Were they within support range?\n"
            "- Was the death caused by teammate mistakes (e.g. support not using CC, core not dealing damage)?\n"
            "- Was it a reasonable sacrifice (buying time for cores or protecting buildings)?"
        )
        skill_item_dim = (
            "## Death Analysis: Skill and Item Usage Dimension\n"
            "- Were escape abilities used properly (Blink, Invisibility, Time Walk, etc.)?\n"
            "- Were defensive items used in time (BKB, Eul's, Force Staff, etc.)?\n"
            "- Were key abilities on cooldown? If so, the player should not have been in a dangerous position\n"
            "- Key consideration: If escape abilities/defensive items are on CD, your safe position should be more conservative\n"
            "- Identify cases where abilities/items were available but unused -- this is the most common correctable mistake"
        )
        timing_dim = (
            "## Death Analysis: Timing Judgment Dimension\n"
            "- Did the player enter a dangerous area when enemy key abilities were off cooldown?\n"
            "- Was teamfight engagement timing correct (too early or too late)?\n"
            "- Did the player die again after buying back (second death is far worse than the first)?\n"
            "- Key principle: Buyback death penalty is much greater than normal death; choose the safest action after buyback\n"
            "- The core of timing judgment is understanding enemy ability cooldown cycles and teamfight tempo windows"
        )
        trade_dim = (
            "## Death Analysis: Trade Evaluation Dimension\n"
            "- Did the death result in something valuable (kills, tower, Roshan, buyback forced)?\n"
            "- Was a 1-for-1 trade worthwhile (compare net worth and importance of both heroes)?\n"
            "- Deaths in teamfights that result in multiple kills or buildings can be considered positive trades\n"
            "- Supports trading their life for a core's survival is usually a positive trade\n"
            "- Core deaths even with a return kill may be negative trades (longer respawn, greater gold loss)"
        )
        scoring = (
            "## Death Analysis Scoring Rubric\n"
            "- S (Perfect): Unavoidable death, or a valuable intentional sacrifice. E.g. Smoke ganked with no escape, or support sacrifice that wins the teamfight\n"
            "- A (Good): Only minor mistakes, fundamentally correct play. E.g. slightly off positioning but overall sound decision-making\n"
            "- B (Average): Clear judgment error but not a habitual problem. E.g. missed enemy disappearance signal but used available escape tools\n"
            "- C (Poor): Obvious positioning or decision error. E.g. pushed deep into enemy territory with no vision\n"
            "- D (Very Poor): Severe fundamental mistakes. E.g. obvious feeding, greedy farming in known danger, completely ignoring map information\n\n"
            "Score by combining five dimensions: vision, positioning, skill usage, timing, and trade value. The lowest-scoring dimension drives the overall rating."
        )
    else:
        vision_dim = (
            "## 死亡分析：视野与信息维度\n"
            "- 死亡前周围是否有己方眼位？如果没有，属于视野缺失导致的死亡\n"
            "- 小地图上是否能看到敌方英雄位置？缺失几个敌人的信息？\n"
            "- 敌方是否使用了烟雾（Smoke）？如果是，属于高质量gank，减轻玩家责任\n\n"
            "### 视野维度评分标准\n"
            "- S：在完整视野覆盖下操作，即使夜间也有守卫覆盖，利用视野信息主动规避了潜在危险\n"
            "- A：有基本视野覆盖，对消失的敌人保持了一定警惕，在视野不足时选择了保守路线\n"
            "- B：有部分视野但不完整，没有充分利用已有的视野信息\n"
            "- C：在视野不足的情况下冒险，忽视了敌方英雄消失的信号，守卫到期后未及时补放\n"
            "- D：在完全没有视野的区域深入，多名敌方长时间消失却不撤退，夜间在无视野区域单独行动"
        )
        positioning_dim = (
            "## 死亡分析：位置与走位维度\n"
            "- 死亡位置是否在危险区域（敌方高地、缺乏视野的野区、河道）？\n"
            "- 是否在没有队友支援范围内独自推进或刷野？\n"
            "- 兵线位置是否过深？是否贪兵导致被抓？\n\n"
            "### 团队配合维度\n"
            "- 队友位置是否合理？是否在支援范围内？\n"
            "- 是否因为队友失误导致的死亡（如辅助未控、核心未输出）？\n"
            "- 是否是合理的牺牲（为核心争取时间或保护建筑）？"
        )
        skill_item_dim = (
            "## 死亡分析：技能与物品使用维度\n"
            "- 是否合理使用了逃生技能（闪烁、隐身、时间漫步等）？\n"
            "- 是否及时使用了防御物品（BKB、风杖、推推棍等）？\n"
            "- 技能CD是否在冷却中？如果关键技能在CD，不应该待在危险位置\n"
            "- 关键考量：如果逃生技能/防御物品在CD，你的安全位置应该更保守\n"
            "- 判断是否存在「有技能/物品可用但未使用」的情况，这是最常见的可纠正错误"
        )
        timing_dim = (
            "## 死亡分析：时机判断维度\n"
            "- 是否在敌方关键技能CD时进入危险区域（如对面大招刚好冷却好）？\n"
            "- 团战参与时机是否正确（过早或过晚加入）？\n"
            "- 是否在买活后又死了一次（二次死亡比第一次严重得多）？\n"
            "- 关键原则：买活后死亡的惩罚远大于普通死亡，买活后应选择最安全的行动\n"
            "- 时机判断的核心是理解敌方技能CD周期和团战节奏窗口"
        )
        trade_dim = (
            "## 死亡分析：交换评估维度\n"
            "- 死亡是否换到了有价值的东西（击杀、推塔、Roshan、买活）？\n"
            "- 1换1是否值得（看双方净资产和重要程度）？\n"
            "- 团战中的死亡如果换到了多个击杀或建筑，可以视为正面交换\n"
            "- 辅助用命换核心存活通常是正面交换\n"
            "- 核心死亡即使换到击杀也可能是负面交换（死亡时间长、经济损失大）"
        )
        scoring = (
            "## 死亡分析评分标准\n"
            "- S（完美）：不可避免的死亡，或有价值的主动牺牲。例如：被烟雾gank且无逃生手段，或辅助牺牲保核心后赢下团战\n"
            "- A（良好）：只有小失误，基本操作正确。例如：走位稍有偏差但整体决策合理\n"
            "- B（一般）：有明显的判断失误但非致命习惯。例如：没有注意到敌方消失信号但使用了可用的逃生手段\n"
            "- C（较差）：明显的位置或决策错误。例如：在缺乏视野的情况下独自深入敌方区域\n"
            "- D（很差）：严重的基础失误。例如：明显送头、在明知危险时贪兵、完全无视地图信息\n\n"
            "评分时综合视野、位置、技能使用、时机、交换价值五个维度，取最低分维度为主导评价。"
        )

    chunks.append(KnowledgeChunk(
        chunk_id="death_analysis:vision",
        content=vision_dim,
        tags=["situation:death", "mechanic:vision"],
        priority=2,
        token_estimate=len(vision_dim) // 2,
    ))

    chunks.append(KnowledgeChunk(
        chunk_id="death_analysis:positioning",
        content=positioning_dim,
        tags=["situation:death", "mechanic:positioning"],
        priority=2,
        token_estimate=len(positioning_dim) // 2,
    ))

    chunks.append(KnowledgeChunk(
        chunk_id="death_analysis:skill_item",
        content=skill_item_dim,
        tags=["situation:death", "mechanic:skill_usage", "mechanic:item_usage"],
        priority=2,
        token_estimate=len(skill_item_dim) // 2,
    ))

    chunks.append(KnowledgeChunk(
        chunk_id="death_analysis:timing",
        content=timing_dim,
        tags=["situation:death", "mechanic:timing"],
        priority=2,
        token_estimate=len(timing_dim) // 2,
    ))

    chunks.append(KnowledgeChunk(
        chunk_id="death_analysis:trade_eval",
        content=trade_dim,
        tags=["situation:death", "mechanic:trade"],
        priority=2,
        token_estimate=len(trade_dim) // 2,
    ))

    chunks.append(KnowledgeChunk(
        chunk_id="death_analysis:scoring_rubric",
        content=scoring,
        tags=["situation:death", "reference"],
        priority=1,
        token_estimate=len(scoring) // 2,
    ))

    return chunks


# -------------------------------------------------------------------
# 6. General coaching chunks (new knowledge not in hardcoded system)
# -------------------------------------------------------------------

def _build_general_coaching_chunks(lang: str = "zh") -> list[KnowledgeChunk]:
    """~10-15 general coaching knowledge chunks."""
    chunks: list[KnowledgeChunk] = []

    if lang == "en":
        # 1. Roshan timing
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:roshan_timing",
            content=(
                "## Roshan Timing Management\n"
                "- Roshan respawns randomly 7-11 minutes after being killed (average ~8 minutes)\n"
                "- Aegis of the Immortal lasts 5 minutes, then disappears and fully restores HP and mana\n"
                "- Second Roshan drops Cheese additionally, third drops Refresher Shard or Aghanim's Blessing\n"
                "- Record Roshan kill time and set up vision control before the respawn window\n"
                "- After taking Aegis, play aggressively and push; don't waste the Aegis time window"
            ),
            tags=["mechanic:roshan", "phase:mid_game", "phase:late_game"],
            priority=2, token_estimate=0,
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
            priority=2, token_estimate=0,
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
            priority=3, token_estimate=0,
        ))
        # 4. Smoke ganking principles
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:smoke_gank",
            content=(
                "## Smoke Gank Principles\n"
                "- Smoke makes allied units invisible and undetectable by wards; dispels within 1025 range of enemy heroes or buildings\n"
                "- Best smoke timing: after clearing a wave, after placing key wards, when enemy core position is spotted\n"
                "- Recommended smoke group: 3-4 heroes; too many leaves other lanes exposed and wastes farming time\n"
                "- Smoke route should avoid likely enemy Sentry Ward positions\n"
                "- Successful smoke gank should convert into pushing a tower or taking Roshan; don't just kill and go home"
            ),
            tags=["mechanic:smoke", "mechanic:gank", "phase:mid_game"],
            priority=2, token_estimate=0,
        ))
        # 5. Power rune control and timing
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:rune_control",
            content=(
                "## Rune Control and Timing\n"
                "- Bounty runes spawn every 3 minutes in four jungle locations (0:00, 3:00, 6:00...), providing gold and XP\n"
                "- River runes spawn every 2 minutes on both sides of the river (even minutes: 2:00, 4:00, 6:00...)\n"
                "- River rune types: Double Damage, Haste, Invisibility, Illusion, Arcane, Regeneration\n"
                "- Mid should push the wave 10 seconds before even minutes then move to secure the rune; Bottle can be refilled\n"
                "- Haste and Invisibility runes are best for ganking; Double Damage is good for farming or fighting"
            ),
            tags=["mechanic:rune", "phase:laning", "phase:mid_game", "role:mid"],
            priority=3, token_estimate=0,
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
            priority=3, token_estimate=0,
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
            priority=2, token_estimate=0,
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
            priority=3, token_estimate=0,
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
            priority=2, token_estimate=0,
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
            priority=2, token_estimate=0,
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
            tags=["mechanic:teamfight", "mechanic:positioning", "phase:mid_game", "phase:late_game"],
            priority=2, token_estimate=0,
        ))
        # 12. Farming efficiency
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:farming_efficiency",
            content=(
                "## Farming Efficiency Principles\n"
                "- Core heroes should establish a fixed farming route to maximize GPM\n"
                "- Don't wait at one camp for respawn; move to the next camp immediately after clearing\n"
                "- Lane creep gold takes priority over jungle gold (lane gold is higher and doesn't waste travel time)\n"
                "- Heroes with wave-clear abilities should push the lane quickly then move to jungle, not wait on lane\n"
                "- Mid-game targets: carry 600+ GPM, mid 500+ GPM, offlane 400+ GPM"
            ),
            tags=["mechanic:farming", "role:carry", "role:mid", "phase:laning", "phase:mid_game"],
            priority=3, token_estimate=0,
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
            priority=2, token_estimate=0,
        ))
    else:
        # 1. Roshan timing
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:roshan_timing",
            content=(
                "## Roshan时间管理\n"
                "- Roshan被击杀后7-11分钟内随机重生（平均约8分钟）\n"
                "- 不朽盾（Aegis）持续5分钟，过期后自动消失并回满生命和魔法\n"
                "- 第二次Roshan会额外掉落奶酪（Cheese），第三次掉落刷新碎片或A杖祝福\n"
                "- 记录Roshan击杀时间，提前在重生窗口前做好视野控制\n"
                "- 拿盾后应积极打架推进，不要浪费盾的时间窗口"
            ),
            tags=["mechanic:roshan", "phase:mid_game", "phase:late_game"],
            priority=2, token_estimate=0,
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
            priority=2, token_estimate=0,
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
            priority=3, token_estimate=0,
        ))
        # 4. Smoke ganking principles
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:smoke_gank",
            content=(
                "## 烟雾Gank原则\n"
                "- 烟雾使友方单位隐身且不会被守卫发现，靠近敌方英雄或建筑1025距离时自动消散\n"
                "- 最佳开雾时机：打完一波兵线后、刚做好关键眼位后、对方核心位置被发现时\n"
                "- 开雾人数建议3-4人，太多会让其他路暴露且浪费刷钱时间\n"
                "- 开雾路线避开敌方可能的岗哨守卫位置\n"
                "- 成功的烟雾gank可以转化为推塔或Roshan，不要只是杀人后回家"
            ),
            tags=["mechanic:smoke", "mechanic:gank", "phase:mid_game"],
            priority=2, token_estimate=0,
        ))
        # 5. Power rune control and timing
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:rune_control",
            content=(
                "## 神符控制与时机\n"
                "- 赏金神符每3分钟刷新在四个野区（0:00, 3:00, 6:00...），提供金钱和经验\n"
                "- 河道神符每2分钟刷新在河道两侧（偶数分钟：2:00, 4:00, 6:00...）\n"
                "- 河道神符种类：双倍伤害、极速、隐身、幻象、秘法、再生\n"
                "- 中单应在偶数分钟前10秒推线后去抢符，瓶子可以装满\n"
                "- 极速和隐身神符最适合gank，双倍伤害适合刷钱或打架"
            ),
            tags=["mechanic:rune", "phase:laning", "phase:mid_game", "role:mid"],
            priority=3, token_estimate=0,
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
            priority=3, token_estimate=0,
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
            priority=2, token_estimate=0,
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
            priority=3, token_estimate=0,
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
            priority=2, token_estimate=0,
        ))
        # 10. Stacking and pulling for supports
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:stack_pull",
            content=(
                "## 辅助拉野与堆野\n"
                "- 小拉（拉近野）：x:15或x:45时拉野，将兵线拉回己方塔下\n"
                "- 大拉（连拉远野）：小拉后立即拉远野，可以完全清掉一波己方小兵\n"
                "- 堆野：x:53-x:55时攻击中立生物将其引出营地，下一分钟会刷新新的一波\n"
                "- 多层堆叠可以为核心提供大量经济（远古野堆叠尤其有利可图）\n"
                "- 辅助应养成定时拉野堆野的习惯，这是对线期经济和兵线控制的核心技能"
            ),
            tags=["mechanic:stack_pull", "mechanic:laning", "role:support_5", "role:support_4", "phase:laning"],
            priority=2, token_estimate=0,
        ))
        # 11. Teamfight positioning
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:teamfight_positioning",
            content=(
                "## 团战站位原则\n"
                "- 核心输出英雄站后排，等前排吸收技能后再输出\n"
                "- 先手英雄（如Axe、Tidehunter）需要找到好的切入角度，不要正面硬冲\n"
                "- 辅助在团战中的首要任务是释放控制技能和保护核心，不是输出\n"
                "- 不要所有人挤在一起，容易被AOE技能团灭（如黑洞、毁灭、回音击）\n"
                "- 团战开始前确定你的优先攻击目标和需要保护的队友"
            ),
            tags=["mechanic:teamfight", "mechanic:positioning", "phase:mid_game", "phase:late_game"],
            priority=2, token_estimate=0,
        ))
        # 12. Farming efficiency
        chunks.append(KnowledgeChunk(
            chunk_id="coaching:farming_efficiency",
            content=(
                "## 刷钱效率原则\n"
                "- 核心英雄应建立固定的刷野路线，最大化每分钟金钱收入\n"
                "- 不要在一个营地等刷新，打完一个营地立刻移动到下一个\n"
                "- 兵线上的补刀金优先于野怪金（兵线金更高且不会浪费时间走路）\n"
                "- 带推线技能的英雄应快速清线后进野区，而不是在线上等兵\n"
                "- 中期目标：大哥600+GPM，中单500+GPM，三号位400+GPM"
            ),
            tags=["mechanic:farming", "role:carry", "role:mid", "phase:laning", "phase:mid_game"],
            priority=3, token_estimate=0,
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
            priority=2, token_estimate=0,
        ))

    return chunks


# -------------------------------------------------------------------
# 7. Early game chunks (pregame / rune contest / level 1)
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
            "- 辅助应在0:00前到达赏金符点或拉野点\n"
            "- 核心根据对位匹配决定是否帮忙抢符或直接去线上\n"
            "- 如果己方1级团战能力较弱，不要参与赏金符争夺\n"
            "- 优先考虑开局的视野（第一个眼位放置）而不是赏金符"
        )

    chunks.append(KnowledgeChunk(
        chunk_id="early_game:rune_contest_framework",
        content=rune_contest,
        tags=["phase:pregame", "mechanic:rune", "situation:death"],
        priority=2,
        token_estimate=len(rune_contest) // 2,
    ))

    chunks.append(KnowledgeChunk(
        chunk_id="early_game:level1_fight_assessment",
        content=level1_fight,
        tags=["phase:pregame", "phase:laning", "mechanic:teamfight"],
        priority=2,
        token_estimate=len(level1_fight) // 2,
    ))

    chunks.append(KnowledgeChunk(
        chunk_id="early_game:pregame_principles",
        content=pregame_principles,
        tags=["phase:pregame"],
        priority=2,
        token_estimate=len(pregame_principles) // 2,
    ))

    return chunks


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------

def get_all_builtin_chunks(lang: str = "zh") -> list[KnowledgeChunk]:
    """Return all built-in knowledge chunks."""
    chunks: list[KnowledgeChunk] = []
    chunks.extend(_build_hero_chunks(lang))
    chunks.extend(_build_matchup_chunks(lang))
    chunks.extend(_build_role_chunks(lang))
    chunks.extend(_build_vision_chunks(lang))
    chunks.extend(_build_death_analysis_chunks(lang))
    chunks.extend(_build_general_coaching_chunks(lang))
    chunks.extend(_build_early_game_chunks(lang))
    return chunks


# -------------------------------------------------------------------
# 7. Game data chunks (from auto-fetched data, covers 130+ heroes)
# -------------------------------------------------------------------

def _heroes_to_tags(heroes_list: list[str] | None) -> list[str]:
    """Convert a heroes list from editorial data to hero:xxx tags.

    ["Axe"] -> ["hero:axe"]
    ["all"] -> ["hero:all"]
    None    -> []
    """
    if not heroes_list:
        return []
    tags = []
    for h in heroes_list:
        if h.lower() == "all":
            tags.append("hero:all")
        else:
            tags.append(f"hero:{_hero_slug(h)}")
    return tags


def generate_gamedata_chunks(
    abilities: dict[str, dict[str, dict]],
    items: dict[str, dict],
    heroes: dict[str, dict],
    lang: str = "zh",
    matchup_tips: dict[str, dict[str, str]] | None = None,
) -> list[KnowledgeChunk]:
    """Generate retrievable chunks from loaded game data.

    One chunk per hero (abilities overview) tagged with hero:slug.
    One chunk per notable item tagged with item:slug.
    One chunk per matchup pair from editorial matchup tips.
    Token budget control: each chunk is small (~100-200 tokens).

    Only generates chunks for heroes/matchups NOT already covered by builtin
    chunks, to avoid duplication.

    Editorial content (coaching tips, descriptions) is always in English.
    The lang parameter controls structural labels/headers only.
    """
    chunks: list[KnowledgeChunk] = []
    builtin_hero_slugs = {_hero_slug(name) for name in HERO_DATA}

    # Hero ability overview chunks (for heroes not in builtin)
    for hero_name, hero_abilities in abilities.items():
        slug = _hero_slug(hero_name)
        if slug in builtin_hero_slugs:
            continue  # already covered by builtin chunks

        # Build a concise ability overview
        ability_lines = []
        for ability_name, ability_data in hero_abilities.items():
            parts = [ability_name]
            if ability_data.get("damage_type"):
                dtype_label = {"magical": "Magical", "physical": "Physical", "pure": "Pure"}.get(
                    ability_data["damage_type"], ""
                )
                if dtype_label:
                    parts.append(dtype_label)
            if ability_data.get("is_ultimate"):
                parts.append("Ultimate")
            if ability_data.get("pierces_bkb"):
                parts.append("Pierces BKB")
            if ability_data.get("is_passive"):
                parts.append("Passive")
            # coaching_tip is now always English
            if ability_data.get("coaching_tip"):
                parts.append(ability_data["coaching_tip"])
            ability_lines.append("- " + ", ".join(parts))

        if not ability_lines:
            continue

        hero_info = heroes.get(hero_name, {})
        role = hero_info.get("role", "unknown")

        if lang == "en":
            content = f"Hero: {hero_name}\nAbilities:\n" + "\n".join(ability_lines[:6])
        else:
            content = f"英雄：{hero_name}\n技能概览：\n" + "\n".join(ability_lines[:6])

        tags = [f"hero:{slug}", f"role:{_role_tag(role)}"]
        chunks.append(KnowledgeChunk(
            chunk_id=f"gamedata_hero:{slug}",
            content=content,
            category="hero_overview",
            tags=tags,
            priority=3,  # supplementary (auto-generated, not hand-curated)
            source="gamedata",
        ))

    # Hero overview chunks with editorial data (for heroes not in builtin)
    for hero_name, hero_info in heroes.items():
        slug = _hero_slug(hero_name)
        if slug in builtin_hero_slugs:
            continue

        # Editorial content is now English regardless of lang
        laning_tips = hero_info.get("laning_tips", "")
        weaknesses = hero_info.get("weaknesses", [])
        power_spikes = hero_info.get("power_spikes", [])

        # Only generate if we have editorial content
        if not (laning_tips or weaknesses or power_spikes):
            continue

        if lang == "en":
            parts = [f"Hero: {hero_name}"]
        else:
            parts = [f"英雄：{hero_name}"]
        if power_spikes:
            parts.append(f"Power Spikes: {', '.join(power_spikes)}")
        if weaknesses:
            parts.append(f"Weaknesses: {', '.join(weaknesses)}")
        if laning_tips:
            parts.append(f"Laning Tips: {laning_tips}")

        content = "\n".join(parts)
        role = hero_info.get("role", "unknown")

        # Use heroes tag from editorial if available
        hero_tags = _heroes_to_tags(hero_info.get("heroes"))
        if not hero_tags:
            hero_tags = [f"hero:{slug}"]

        chunks.append(KnowledgeChunk(
            chunk_id=f"gamedata_hero_editorial:{slug}",
            content=content,
            category="hero_laning",
            tags=hero_tags + [f"role:{_role_tag(role)}", "phase:laning"],
            priority=2,
            source="editorial",
        ))

    # Editorial matchup tip chunks (skip pairs already in builtin)
    if matchup_tips:
        builtin_matchup_ids = {
            f"matchup:{_hero_slug(h1)}_vs_{_hero_slug(h2)}"
            for h1, opps in _MATCHUP_TIPS_EN.items()
            for h2 in opps
        }
        for hero_a, opponents in matchup_tips.items():
            slug_a = _hero_slug(hero_a)
            for hero_b, tip in opponents.items():
                slug_b = _hero_slug(hero_b)
                chunk_id = f"matchup:{slug_a}_vs_{slug_b}"
                if chunk_id in builtin_matchup_ids:
                    continue
                if lang == "en":
                    content = f"{hero_a} vs {hero_b}: {tip}"
                else:
                    content = f"{hero_a} 对阵 {hero_b}: {tip}"
                chunks.append(KnowledgeChunk(
                    chunk_id=chunk_id,
                    content=content,
                    category="matchup",
                    tags=[
                        f"matchup:{slug_a}:{slug_b}",
                        f"hero:{slug_a}",
                        f"hero:{slug_b}",
                    ],
                    priority=2,
                    source="editorial",
                ))

    return chunks
