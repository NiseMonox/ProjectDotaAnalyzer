"""Dota 2 knowledge base for AI coaching context injection.

Provides structured hero information, matchup advice, laning principles,
phase objectives, and death analysis rules for direct injection into
LLM prompts.

This is NOT a database; it is a Python module with dicts and functions
that provide coaching knowledge text for LLM context.
"""
from __future__ import annotations

import logging

from .models import DecisionContext

logger = logging.getLogger(__name__)

try:
    from .game_data import (
        format_hero_abilities,
        format_player_items,
        format_ability_at_level,
        ABILITY_DATA,
        ITEM_DATA,
        _ensure_data_loaded,
    )
    _HAS_GAME_DATA = True
except ImportError:
    _HAS_GAME_DATA = False
    _ensure_data_loaded = lambda: None

# RAG knowledge store (lazy initialized)
_knowledge_store = None


def _get_knowledge_store(lang: str = "zh"):
    """Get or create a per-language knowledge store."""
    global _knowledge_store
    # Use a dict keyed by lang for per-language stores
    if _knowledge_store is None or not isinstance(_knowledge_store, dict):
        _knowledge_store = {}
    if lang in _knowledge_store:
        return _knowledge_store[lang]
    try:
        from .knowledge_store import KnowledgeStore
        store = KnowledgeStore()
        store.load_builtin(lang=lang)

        # Load game data chunks (covers all 130+ heroes via RAG)
        try:
            from .knowledge_chunks import generate_gamedata_chunks
            from .data_manager import load_abilities, load_items, load_heroes, load_matchup_tips
            gamedata_chunks = generate_gamedata_chunks(
                load_abilities(), load_items(), load_heroes(), lang=lang,
                matchup_tips=load_matchup_tips(),
            )
            for chunk in gamedata_chunks:
                store.add_chunk(chunk)
            logger.info("Game data RAG chunks loaded (%s): %d", lang, len(gamedata_chunks))
        except ImportError:
            logger.debug("data_manager or generate_gamedata_chunks not available")
        except Exception as e:
            logger.warning("Failed to load game data chunks: %s", e)

        # Try loading user knowledge from data directory
        import os
        knowledge_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data", "knowledge"
        )
        if os.path.isdir(knowledge_dir):
            for fname in os.listdir(knowledge_dir):
                if fname.endswith(".json"):
                    try:
                        store.load_json(os.path.join(knowledge_dir, fname))
                    except Exception:
                        pass
        logger.info("Knowledge store loaded (%s): %s", lang, store.stats())
        _knowledge_store[lang] = store
    except ImportError:
        logger.info("knowledge_store module not available, using legacy knowledge injection")
        _knowledge_store[lang] = None  # keep None to trigger fallback
    return _knowledge_store.get(lang)

# ===================================================================
# Hero knowledge database (fallback + file-backed)
# ===================================================================

# Hardcoded fallback hero data (used when JSON files unavailable)
# All content is now in English (no _en suffix fields needed)
_FALLBACK_HERO_DATA: dict[str, dict] = {
    # ----- Mock match heroes (Radiant) -----
    "Hoodwink": {
        "role": "support_4",
        "attribute": "agi",
        "attack_type": "ranged",
        "power_spikes": ["Level 3 (Acorn Shot + Bushwhack combo)", "Level 6 (Sharpshooter)", "Blink Dagger timing"],
        "weaknesses": ["Low HP, easily burst down", "Skills depend on tree terrain", "Lacks hard disable"],
        "laning_tips": "Use Acorn Shot to harass and whittle down opponents. Bushwhack combined with allies for kills. Utilize tree terrain to create favorable engagements, pull camps to maintain creep equilibrium.",
        "level1_strength": "average",
        "level1_note": "Acorn Shot provides decent level 1 damage with bounces, but lacks disable. Average level 1 fighting ability.",
    },
    "Drow Ranger": {
        "role": "carry",
        "attribute": "agi",
        "attack_type": "ranged",
        "power_spikes": ["Level 6 (Marksmanship aura)", "Hurricane Pike", "Butterfly"],
        "weaknesses": ["Marksmanship disabled when enemies are close", "No escape ability", "Low HP in early game"],
        "laning_tips": "Use ranged attack advantage for stable last-hitting. Frost Arrows can continuously harass opponents. Keep distance — extremely vulnerable when closed on. Key is surviving early and farming safely until level 6 Marksmanship.",
        "level1_strength": "average",
        "level1_note": "Frost Arrows provide slow and bonus damage but lack burst. Average level 1 trading ability.",
    },
    "Warlock": {
        "role": "support_5",
        "attribute": "int",
        "attack_type": "ranged",
        "power_spikes": ["Level 6 (Chaotic Offering)", "Level 12 (Chaotic Offering level 2)", "Refresher Orb"],
        "weaknesses": ["Slow movement speed", "Ultimate has very long cooldown", "Lacks reliable disable (except ultimate)"],
        "laning_tips": "Shadow Word is a powerful laning tool — heals allies or damages enemies. Fatal Bonds combined with allies makes life miserable for opponents. Protect your core's last-hitting and pull camps when appropriate.",
        "level1_strength": "average",
        "level1_note": "Shadow Word provides sustained heal or damage but lacks burst and disable. Average level 1 fight contribution.",
    },
    "Razor": {
        "role": "offlane",
        "attribute": "agi",
        "attack_type": "ranged",
        "power_spikes": ["Level 6 (Eye of the Storm)", "BKB timing", "Aghanim's Scepter"],
        "weaknesses": ["No hard disable", "Static Link requires maintaining range", "Falls off late game"],
        "laning_tips": "Static Link is a lane-winning skill — steal the enemy's attack damage. Use Plasma Field to harass safely. After Eye of the Storm, fight actively, especially in the center of teamfights for maximum damage.",
        "level1_strength": "average",
        "level1_note": "Plasma Field provides AoE + slow for level 1 fights; Static Link steals attack damage in 1v1. Decent level 1 fighting ability.",
    },
    "Axe": {
        "role": "offlane",
        "attribute": "str",
        "attack_type": "melee",
        "power_spikes": ["Level 2-3 (Counter Helix procs frequently)", "Blink Dagger timing", "Level 6 (Culling Blade)"],
        "weaknesses": ["Weak initiation before Blink Dagger", "Vulnerable to kiting", "Not tanky enough late game"],
        "laning_tips": "Stand among enemy creeps to trigger Counter Helix — the most efficient laning technique. Battle Hunger sustains harass and prevents Blink. After Blink Dagger, look for Berserker's Call initiations.",
        "level1_strength": "strong",
        "level1_note": "Counter Helix deals high AoE damage, procs frequently when surrounded — extremely strong in level 1 fights.",
    },

    # ----- Mock match heroes (Dire) -----
    "Shadow Shaman": {
        "role": "support_5",
        "attribute": "int",
        "attack_type": "ranged",
        "power_spikes": ["Level 6 (Mass Serpent Ward for pushing)", "Blink Dagger", "Aether Lens"],
        "weaknesses": ["Cannot move during Shackles channel", "Low HP", "No escape ability"],
        "laning_tips": "Hex + Shackles is a 5+ second disable chain — almost guaranteed kill with allies. Ether Shock for wave clear and harass. Mass Serpent Ward pushes towers extremely fast; coordinate pushes after level 6.",
        "level1_strength": "average",
        "level1_note": "Ether Shock provides decent AoE damage but lacks disable at level 1. Average level 1 fight contribution.",
    },
    "Viper": {
        "role": "carry",
        "attribute": "agi",
        "attack_type": "ranged",
        "power_spikes": ["Level 1-3 (Poison Attack dominance)", "Level 6 (Viper Strike)", "Hurricane Pike"],
        "weaknesses": ["No escape ability", "Weak late-game damage output", "Requires good positioning for sustained teamfight DPS"],
        "laning_tips": "Poison Attack is one of the strongest laning harass tools. Nethertoxin zones restrict enemy movement. Corrosive Skin deters melee heroes. You almost never lose lane — the key is converting your advantage into push tempo.",
        "level1_strength": "average",
        "level1_note": "Poison Attack provides sustained slow and damage, excellent for lane harass, but lacks burst for level 1 fights.",
    },
    "Lion": {
        "role": "support_4",
        "attribute": "int",
        "attack_type": "ranged",
        "power_spikes": ["Level 6 (Finger of Death)", "Blink Dagger", "Aether Lens"],
        "weaknesses": ["Extremely low HP", "No escape", "Long ability cooldowns"],
        "laning_tips": "Earth Spike + Hex provides extremely long disable chain. Mana Drain solves mana issues. After level 6, Finger of Death burst is massive — nearly guarantees a kill with disable setup. Watch positioning, very easy to get burst down.",
        "level1_strength": "average",
        "level1_note": "Earth Spike provides a line stun, offering decent disable and damage contribution at level 1.",
    },
    "Dragon Knight": {
        "role": "mid",
        "attribute": "str",
        "attack_type": "melee",
        "power_spikes": ["Level 6 (Dragon Form ranged attack)", "BKB timing", "Aghanim's Scepter (level 3 Dragon Form)"],
        "weaknesses": ["Weak laning before level 6", "No escape ability", "Slow farming speed"],
        "laning_tips": "Dragon Blood provides high regen for stable laning. Breathe Fire for last-hitting and harass. After level 6 Dragon Form, actively push towers and fight. Mid game is your strongest timing — lead your team's push.",
        "level1_strength": "average",
        "level1_note": "Breathe Fire deals AoE damage and reduces enemy attack damage, providing decent level 1 fight contribution.",
    },
    "Bristleback": {
        "role": "offlane",
        "attribute": "str",
        "attack_type": "melee",
        "power_spikes": ["Level 3-4 (Bristleback stacks quickly)", "Vanguard timing", "Level 6 (Warpath)"],
        "weaknesses": ["Fragile when attacked from the front", "Vulnerable to purge effects (dispels Nasal Goo)", "Lacks disable"],
        "laning_tips": "Keep your back to enemies to trigger Bristleback passive. Nasal Goo stacking armor reduction is very strong. Spam Quill Spray in lane to zone enemies. Be careful when facing enemies head-on — retreat cautiously.",
        "level1_strength": "average",
        "level1_note": "Quill Spray provides AoE damage with stacking, decent in level 1 skirmishes but needs time to ramp up.",
    },

    # ----- Common heroes -----
    "Anti-Mage": {
        "role": "carry",
        "attribute": "agi",
        "attack_type": "melee",
        "power_spikes": ["Battle Fury completion", "Level 6 (Mana Void)", "Manta Style completion"],
        "weaknesses": ["Extremely weak early game", "Vulnerable to chain stuns", "Requires extensive farming time"],
        "laning_tips": "Extremely weak early — survival and last-hitting are top priority. Blink to dodge ganks but watch mana. Counterspell can reflect key abilities. Rush Battle Fury to accelerate farm.",
        "level1_strength": "weak",
        "level1_note": "Mana Break deals negligible damage at level 1. Melee hero with no ability to contribute meaningfully in level 1 fights.",
    },
    "Invoker": {
        "role": "mid",
        "attribute": "uni",
        "attack_type": "ranged",
        "power_spikes": ["Level 4 (two-spell combos)", "Level 7 (three-spell combos)", "Aghanim's Scepter", "Refresher Orb"],
        "weaknesses": ["Very weak before level 3", "Spell switching has delay", "Level-dependent"],
        "laning_tips": "Early laning with Ice Wall and Cold Snap. Alacrity with Forge Spirit provides last-hit advantage. Watch rune timing, call missing if mid disappears. Cold Snap is the strongest solo laning spell.",
        "level1_strength": "weak",
        "level1_note": "Only 1 orb instance at level 1 — Quas or Exort provides minimal bonuses. Cannot effectively contribute to level 1 fights.",
    },
    "Phantom Assassin": {
        "role": "carry",
        "attribute": "agi",
        "attack_type": "melee",
        "power_spikes": ["Level 6 (Coup de Grace crit)", "Battle Fury", "BKB timing", "Basher"],
        "weaknesses": ["Countered by MKB (pierces evasion)", "Magic damage ignores evasion", "Early survival depends on Blur"],
        "laning_tips": "Stifling Dagger for ranged last-hits and harass. Phantom Strike in then immediately Dagger for slow. After level 6, high crit chance enables kill attempts — but watch for enemy disable reserves.",
        "level1_strength": "weak",
        "level1_note": "Stifling Dagger provides ranged harass but limited damage. Weak in melee level 1 fights otherwise.",
    },
    "Crystal Maiden": {
        "role": "support_5",
        "attribute": "int",
        "attack_type": "ranged",
        "power_spikes": ["Level 2 (Crystal Nova + Frostbite combo)", "Level 6 (Freezing Field)", "BKB"],
        "weaknesses": ["Slowest movement speed in the game", "Extremely low HP", "Ultimate requires channeling, easily interrupted"],
        "laning_tips": "Arcane Aura provides mana regen for the entire team — one of the strongest support passives. Crystal Nova + Frostbite combo deals high damage with strong slow. Manage your HP carefully, very easy to get burst down.",
        "level1_strength": "average",
        "level1_note": "Crystal Nova provides AoE slow and damage, offering decent contribution in level 1 fights.",
    },
    "Pudge": {
        "role": "support_4",
        "attribute": "str",
        "attack_type": "melee",
        "power_spikes": ["Level 2-3 (Hook + Rot combo)", "Level 6 (Dismember)", "Blink Dagger", "Aghanim's Scepter"],
        "weaknesses": ["Hook requires prediction, easy to miss", "Cannot deal damage when kited", "Low armor early game"],
        "laning_tips": "Hook is a game-changing skill — requires prediction of enemy movement. Rot follows up Hook for sustained damage. Level 6 Dismember + Rot combo deals massive damage. Hooks from fog are more effective.",
        "level1_strength": "strong",
        "level1_note": "Rot deals sustained AoE damage and slows, extremely high close-range output in level 1 fights.",
    },
    "Juggernaut": {
        "role": "carry",
        "attribute": "agi",
        "attack_type": "melee",
        "power_spikes": ["Level 1 (Blade Fury magic immunity)", "Level 6 (Omnislash)", "Battle Fury", "Butterfly"],
        "weaknesses": ["Omnislash easily split by creeps", "Cannot attack during Blade Fury", "Healing Ward easily destroyed"],
        "laning_tips": "Blade Fury provides magic immunity — combine with allies for kills. Healing Ward sustains HP. High base damage makes last-hitting easy. Watch for creeps splitting Omnislash bounces.",
        "level1_strength": "strong",
        "level1_note": "Blade Fury at level 1 grants magic immunity + sustained AoE damage — extremely powerful in level 1 fights.",
    },
    "Sniper": {
        "role": "mid",
        "attribute": "agi",
        "attack_type": "ranged",
        "power_spikes": ["Level 3-4 (Shrapnel controls rune spots)", "Level 6 (Assassinate)", "Hurricane Pike", "Daedalus"],
        "weaknesses": ["No escape ability", "Vulnerable to gap-closing heroes", "Low HP"],
        "laning_tips": "Exploit your extreme attack range to harass. Shrapnel controls river vision and slows. Positioning is critical — almost certain death if closed on. Headshot passive provides bonus damage and slow for strong lane harass.",
        "level1_strength": "average",
        "level1_note": "Shrapnel provides AoE slow and damage, decent level 1 fight contribution but very fragile.",
    },
    "Zeus": {
        "role": "mid",
        "attribute": "int",
        "attack_type": "ranged",
        "power_spikes": ["Level 6 (Thundergod's Wrath global damage)", "Aghanim's Scepter", "Refresher Orb (double ultimate)"],
        "weaknesses": ["Low HP with no escape", "Countered by BKB", "Mana-starved early game"],
        "laning_tips": "Arc Lightning for last-hitting and harass — manage mana carefully. Lightning Bolt deals high damage but is mana-intensive. After level 6, Thundergod's Wrath secures kills globally or provides vision. Heavenly Jump offers some escape.",
        "level1_strength": "average",
        "level1_note": "Arc Lightning provides low-cooldown AoE damage, can be cast frequently in level 1 fights.",
    },
    "Witch Doctor": {
        "role": "support_5",
        "attribute": "int",
        "attack_type": "ranged",
        "power_spikes": ["Level 1 (Paralyzing Cask bouncing stun)", "Level 6 (Death Ward)", "BKB", "Aghanim's Scepter"],
        "weaknesses": ["Ultimate requires channeling", "Low HP", "Death Ward easily interrupted"],
        "laning_tips": "Paralyzing Cask bounces more in dual lanes for excellent results. Maledict is an extremely powerful laning damage skill. Voodoo Restoration heals yourself and allies. Level 6 Death Ward deals explosive damage but needs protection.",
        "level1_strength": "strong",
        "level1_note": "Paralyzing Cask bouncing stun is extremely powerful in multi-hero fights — one of the best level 1 abilities.",
    },
    "Faceless Void": {
        "role": "carry",
        "attribute": "agi",
        "attack_type": "melee",
        "power_spikes": ["Level 6 (Chronosphere)", "Battle Fury", "BKB", "Daedalus"],
        "weaknesses": ["Missing Chronosphere loses momentum", "Weak early, needs farm", "Chronosphere has long cooldown"],
        "laning_tips": "Time Walk provides escape and damage backtrack. Focus on last-hitting in lane, don't risk kills. Level 6 Chronosphere with allies can team wipe. Note: Chronosphere also freezes allies — cast carefully.",
        "level1_strength": "weak",
        "level1_note": "Time Walk only provides escape, cannot effectively contribute to level 1 fights.",
    },
    "Earthshaker": {
        "role": "support_4",
        "attribute": "str",
        "attack_type": "melee",
        "power_spikes": ["Level 6 (Echo Slam)", "Blink Dagger", "Aghanim's Scepter"],
        "weaknesses": ["No initiation before Blink Dagger", "Mana-starved", "Weak laning"],
        "laning_tips": "Fissure is a powerful ranged stun but very mana-intensive. Focus on pulling camps and accumulating gold in lane. Enchant Totem bonus damage helps with last-hits. Blink Dagger + Echo Slam initiation changes teamfights.",
        "level1_strength": "strong",
        "level1_note": "Fissure provides long-range line stun + terrain block — extremely strong disable in level 1 fights.",
    },
    "Tidehunter": {
        "role": "offlane",
        "attribute": "str",
        "attack_type": "melee",
        "power_spikes": ["Level 6 (Ravage)", "Blink Dagger", "Refresher Orb (double Ravage)"],
        "weaknesses": ["Ultimate has extremely long cooldown", "Slow farming speed", "Lacks sustained damage output"],
        "laning_tips": "Kraken Shell provides extreme lane sustain; armor reduction retaliates against melee opponents. Anchor Smash for harass and last-hits. Almost never loses lane but hard to kill enemies either. Level 6 Ravage is a teamfight-changing ultimate.",
        "level1_strength": "strong",
        "level1_note": "Gush provides slow and armor reduction. Combined with high base armor, very strong in level 1 melee fights.",
    },
    "Enigma": {
        "role": "offlane",
        "attribute": "int",
        "attack_type": "ranged",
        "power_spikes": ["Level 6 (Black Hole)", "BKB + Black Hole combo", "Refresher Orb"],
        "weaknesses": ["Black Hole requires channeling, easily interrupted", "Eidolons easily cleared by AoE", "Missing Black Hole loses momentum"],
        "laning_tips": "Eidolons control the creep wave and jungle. Midnight Pulse is a strong harass tool. Demonic Conversion can dominate large neutrals for extra economy. Push lane with Eidolons then jungle — very efficient.",
        "level1_strength": "weak",
        "level1_note": "Eidolons provide damage but are very fragile — easily killed in level 1 fights.",
    },
    "Phoenix": {
        "role": "support_4",
        "attribute": "str",
        "attack_type": "ranged",
        "power_spikes": ["Level 6 (Supernova)", "Level 12", "Refresher Orb"],
        "weaknesses": ["Supernova easily focused down", "Abilities cost HP", "Difficult HP management early game"],
        "laning_tips": "Sun Ray is an extremely strong harass tool with sustained damage and slow. Fire Spirits can harass and assist last-hitting. Abilities cost HP so manage health carefully. Timing your Icarus Dive engagement is crucial.",
        "level1_strength": "average",
        "level1_note": "Fire Spirits provide sustained attack speed reduction, useful in level 1 fights but require landing hits.",
    },
    "Rubick": {
        "role": "support_4",
        "attribute": "int",
        "attack_type": "ranged",
        "power_spikes": ["Level 6 (Spell Steal)", "Blink Dagger", "Aghanim's Scepter (steal two spells)"],
        "weaknesses": ["Low damage if no good spell is stolen", "Low HP", "Requires extremely high skill level"],
        "laning_tips": "Telekinesis is a powerful disable. Fade Bolt reduces enemy attack damage. After level 6, Spell Steal is the most flexible ultimate — watch enemy spell cast order. Arcane Supremacy enhances stolen spells.",
        "level1_strength": "average",
        "level1_note": "Telekinesis provides single-target disable, decent level 1 fight contribution but limited follow-up damage.",
    },
}


# ===================================================================
# Laning principles (by role)
# ===================================================================

_LANING_PRINCIPLES: dict[str, str] = {
    "carry": (
        "## 大哥（一号位）对线原则\n"
        "1. **补刀为王**：稳定补刀是最重要的任务，目标10分钟60刀以上。不要为了骚扰牺牲补刀\n"
        "2. **兵线控制**：将兵线维持在靠近己方塔前的位置，远离敌方塔和河道。如果兵线推过去了，考虑拉一波野\n"
        "3. **注意敌方游走**：随时关注小地图，辅助消失可能来抓你。保持一个TP防止被越塔\n"
        "4. **血量管理**：保持血量在安全线以上（至少60%），及时使用回复消耗品。低血量时退到塔下\n"
        "5. **出装节奏**：先出对线用品（树枝、铁皮鞋），再出刷钱道具。不要贪大件导致对线期裸身"
    ),
    "mid": (
        "## 中单（二号位）对线原则\n"
        "1. **控制河道神符**：每两分钟（偶数分钟）关注河道神符，提前推线抢符\n"
        "2. **补刀+反补**：中路是1v1，补刀和反补都很重要。高地视野优势要利用好\n"
        "3. **注意对面失踪**：敌方中单消失立刻报信号。同时注意对面辅助是否在游走\n"
        "4. **制造击杀机会**：利用等级优势（先到6级）创造击杀或推塔机会\n"
        "5. **控制瓶子节奏**：瓶子的使用和灌满时机非常重要，神符和回泉都是补充手段"
    ),
    "offlane": (
        "## 三号位（副核心/劣势路）对线原则\n"
        "1. **存活优先**：劣势路的第一目标是不死，获取经验比金钱更重要\n"
        "2. **骚扰敌方核心**：利用技能消耗敌方大哥，降低其补刀效率\n"
        "3. **兵线拉扯**：如果兵线推到敌方塔下，可以去骚扰敌方拉野，或者自己去打野\n"
        "4. **创造空间**：如果能在劣势路吸引2-3个人来抓你，就是在帮队伍创造空间\n"
        "5. **出装节奏**：先出对线生存装（如先锋盾、魔棒），关键装备如跳刀要尽快出"
    ),
    "support_4": (
        "## 四号位（游走辅助）对线原则\n"
        "1. **游走节奏**：开局帮劣势路对线，根据情况决定是否游走中路或己方大哥路\n"
        "2. **做视野**：关键位置（河道、敌方野区入口）放置眼位，提供信息优势\n"
        "3. **拉野堆野**：帮核心拉野控制兵线，堆积中立野给核心刷钱\n"
        "4. **保护核心补刀**：在线上时，用技能骚扰对手，让核心安心补刀\n"
        "5. **经济利用**：辅助经济有限，优先出团队装（魔棒、韧鼓、秘法鞋）"
    ),
    "support_5": (
        "## 五号位（硬辅助）对线原则\n"
        "1. **购买消耗品**：开局准备足够的树枝、净化和眼位，确保己方核心有良好的对线环境\n"
        "2. **拉野控线**：定时拉野（x:15/x:45）维持兵线在有利位置。小拉和大拉都要熟练\n"
        "3. **做视野**：是眼位的主要提供者，确保关键位置有眼（河道、入口、Roshan坑）\n"
        "4. **保护核心**：时刻关注核心的安全，对面gank时要能及时支援\n"
        "5. **适时换线/TP支援**：对线不顺时考虑换线，其他路有击杀机会时TP支援"
    ),
}

_LANING_PRINCIPLES_EN: dict[str, str] = {
    "carry": (
        "## Carry (Position 1) Laning Principles\n"
        "1. **Last-Hitting is King**: Stable last-hitting is the most important task — aim for 60+ CS by 10 minutes. Don't sacrifice CS for harass\n"
        "2. **Creep Equilibrium**: Keep the creep wave near your tower, away from the enemy tower and river. If pushed out, consider pulling a camp\n"
        "3. **Watch for Rotations**: Always check the minimap — missing supports may be coming to gank. Keep a TP scroll to prevent tower dives\n"
        "4. **HP Management**: Stay above a safe HP threshold (at least 60%), use regeneration consumables promptly. Retreat to tower when low\n"
        "5. **Itemization Pacing**: Build laning items first (branches, boots), then farming items. Don't greed for big items and lane with no stats"
    ),
    "mid": (
        "## Mid (Position 2) Laning Principles\n"
        "1. **Control River Runes**: Every 2 minutes (even minutes) watch for river runes — push the wave early to secure them\n"
        "2. **Last-Hit + Deny**: Mid is 1v1 — both last-hitting and denying matter. Use high ground vision advantage\n"
        "3. **Call Missing**: If the enemy mid disappears, ping immediately. Also watch for enemy support rotations\n"
        "4. **Create Kill Opportunities**: Use level advantage (hitting 6 first) to create kills or push the tower\n"
        "5. **Bottle Management**: Bottle usage and refill timing are critical — runes and fountain trips are both refill options"
    ),
    "offlane": (
        "## Offlane (Position 3) Laning Principles\n"
        "1. **Survival First**: The primary goal of the offlane is not dying — experience matters more than gold\n"
        "2. **Harass Enemy Carry**: Use abilities to pressure the enemy carry, reducing their last-hit efficiency\n"
        "3. **Creep Manipulation**: If the wave pushes to the enemy tower, harass their pull camp or go jungle\n"
        "4. **Create Space**: If you draw 2-3 heroes to gank you, you're creating space for your team\n"
        "5. **Itemization Pacing**: Prioritize survival items (Vanguard, Magic Wand), rush key items like Blink Dagger"
    ),
    "support_4": (
        "## Position 4 (Roaming Support) Laning Principles\n"
        "1. **Rotation Timing**: Start by helping the offlane, then decide whether to rotate mid or safelane based on the situation\n"
        "2. **Ward Placement**: Place wards at key positions (river, enemy jungle entrances) for information advantage\n"
        "3. **Pull and Stack**: Pull camps to control creep equilibrium, stack camps for cores to farm\n"
        "4. **Protect Core Farming**: When in lane, use abilities to harass opponents so your core can farm safely\n"
        "5. **Economy Usage**: Support economy is limited — prioritize team items (Magic Wand, Drum, Arcane Boots)"
    ),
    "support_5": (
        "## Position 5 (Hard Support) Laning Principles\n"
        "1. **Buy Consumables**: Start with enough branches, salves, and wards to ensure your core has a good laning environment\n"
        "2. **Pull Creeps**: Pull camps at x:15/x:45 to keep the wave in a favorable position. Master both small and big pulls\n"
        "3. **Ward Placement**: Primary ward provider — ensure key spots are warded (river, entrances, Roshan pit)\n"
        "4. **Protect Your Core**: Always watch your core's safety — be ready to respond to enemy ganks\n"
        "5. **Lane Swap / TP Support**: Consider swapping lanes if losing, TP to other lanes when there are kill opportunities"
    ),
}


# ===================================================================
# Phase objectives (by phase and role)
# ===================================================================

_PHASE_OBJECTIVES: dict[str, dict[str, str]] = {
    "pregame": {
        "carry": "根据对线情况决定是否帮忙抢符。如果对方1级更强，直接去线上更有价值。",
        "mid": "直接去中路等兵线，不要参与赏金符争夺（中路兵线价值远高于40金）。",
        "offlane": "和队友在安全侧集合。对方人多不要硬抢。",
        "support_4": "帮助争夺赏金符但注意统计敌方人数，人多立即撤退。",
        "support_5": "放好第一个眼后去赏金符点。对方3+人就放弃，去准备拉野。",
    },
    "laning": {
        "carry": "稳定补刀积累经济，目标10分钟70+补刀。不要冒险击杀，保持生存优先。关注敌方游走信息，及时TP防止队友被杀。",
        "mid": "利用1v1优势建立等级差距。控制神符节奏，有机会就游走其他路。6级时间点如果有大招爆发，可以主动寻找击杀。",
        "offlane": "压制敌方大哥的发育节奏。获取经验为主，不要为了补刀送命。如果对线顺利，可以尝试推塔施压。",
        "support_4": "帮助劣势路建立优势，适时游走中路或大哥路。做好视野工作，关注敌方gank路线。拉野堆野帮核心建立经济优势。",
        "support_5": "保护大哥安全补刀。拉野控制兵线位置。放置关键位置眼位。确保足够的回复品供应。",
    },
    "mid_game": {
        "carry": "加速刷钱或参与关键团战。出核心装备后可以积极打架。推塔拿地图控制权。注意带TP随时准备支援。",
        "mid": "积极参与团战和推进。利用等级和装备优势带节奏。控制Roshan时间点。帮助队伍建立视野控制。",
        "offlane": "先手开团是主要职责。出跳刀等先手道具后积极寻找机会。保护后排的同时创造空间。",
        "support_4": "做好视野控制。关键团战提供控制和技能输出。帮助核心入侵敌方野区。Smoke配合队友找机会击杀。",
        "support_5": "维持视野网络。团战中保护核心存活。购买团队道具（秘法、梅肯等）。控制Roshan入口视野。",
    },
    "late_game": {
        "carry": "团战输出核心。注意买活时间窗口。不要独自在危险区域。保持刷钱节奏但别错过关键团战。",
        "mid": "提供关键技能输出。保护高地或配合推高地。买活时机非常关键。注意BKB剩余时间管理。",
        "offlane": "先手开团决定胜负。保护高地时寻找反打机会。光环装备为团队提供增益。注意敌方核心位置再开团。",
        "support_4": "关键控制决定团战。Smoke开雾找机会。保持视野压制。注意保护后排防止被跳脸。",
        "support_5": "维持视野是最后的防线。团战中控制和保护核心。买活金准备好。关键物品使用（风杖、推推、莲花宝珠）。",
    },
}

_PHASE_OBJECTIVES_EN: dict[str, dict[str, str]] = {
    "pregame": {
        "carry": "Decide whether to help contest the bounty rune based on matchup. If the enemy is stronger at level 1, going directly to lane is more valuable.",
        "mid": "Go directly to mid lane and wait for creeps — do not participate in bounty rune contests (mid lane creep wave is worth far more than 40 gold).",
        "offlane": "Group with teammates on the safe side. Don't force a contest if the enemy has more heroes.",
        "support_4": "Help contest the bounty rune but count enemy heroes — retreat immediately if outnumbered.",
        "support_5": "Place the first ward then go to the bounty rune spot. If the enemy has 3+ heroes, give up and prepare to pull.",
    },
    "laning": {
        "carry": "Farm steadily — aim for 70+ CS by 10 minutes. Don't risk kills, prioritize survival. Watch for enemy rotation info, TP to save allies when needed.",
        "mid": "Leverage 1v1 advantage to build a level gap. Control rune timing, rotate to other lanes when opportunities arise. At the level 6 timing, look for kills if your ultimate has burst.",
        "offlane": "Disrupt the enemy carry's farming rhythm. Focus on gaining experience — don't die for last-hits. If laning goes well, try pushing the tower to apply pressure.",
        "support_4": "Help the offlane establish an advantage, rotate mid or safelane when appropriate. Maintain vision and watch for enemy gank paths. Pull and stack camps to build core economy.",
        "support_5": "Protect the carry's safe farming. Pull camps to control creep wave position. Place wards at key locations. Ensure adequate regeneration supply.",
    },
    "mid_game": {
        "carry": "Accelerate farming or join key teamfights. After core items, fight actively. Push towers to claim map control. Keep TP ready to support.",
        "mid": "Actively participate in fights and pushes. Use level and item advantage to set the tempo. Control Roshan timing. Help the team establish vision control.",
        "offlane": "Initiating teamfights is your primary role. After Blink or initiation items, actively look for opportunities. Protect backline while creating space.",
        "support_4": "Maintain vision control. Provide disable and ability damage in key fights. Help cores invade enemy jungle. Smoke with teammates to find pickoffs.",
        "support_5": "Maintain the ward network. Protect cores in teamfights. Buy team items (Arcane Boots, Mekansm, etc.). Control Roshan pit vision.",
    },
    "late_game": {
        "carry": "Core teamfight damage dealer. Watch buyback timing windows. Don't be alone in dangerous areas. Maintain farm tempo but don't miss key fights.",
        "mid": "Provide key ability damage. Defend high ground or help push high ground. Buyback timing is critical. Manage remaining BKB duration.",
        "offlane": "Initiation decides the game. Look for counter-initiation when defending high ground. Aura items provide team buffs. Check enemy carry position before initiating.",
        "support_4": "Key disables decide teamfights. Smoke to find opportunities. Maintain vision pressure. Protect backline from enemy initiators.",
        "support_5": "Vision is the last line of defense. Protect and disable for cores in teamfights. Have buyback gold ready. Use key items (Eul's, Force Staff, Lotus Orb).",
    },
}


# ===================================================================
# Death analysis rules
# ===================================================================

_DEATH_ANALYSIS_RULES = """## 死亡分析评判规则

### 一、视野与信息
- 死亡前周围是否有己方眼位？如果没有，属于视野缺失导致的死亡
- 小地图上是否能看到敌方英雄位置？缺失几个敌人的信息？
- 敌方是否使用了烟雾（Smoke）？如果是，属于高质量gank，减轻玩家责任

### 二、位置与走位
- 死亡位置是否在危险区域（敌方高地、缺乏视野的野区、河道）？
- 是否在没有队友支援范围内独自推进或刷野？
- 兵线位置是否过深？是否贪兵导致被抓？

### 三、技能与物品使用
- 是否合理使用了逃生技能（闪烁、隐身、时间漫步等）？
- 是否及时使用了防御物品（BKB、风杖、推推棍等）？
- 技能CD是否在冷却中？如果关键技能在CD，不应该待在危险位置

### 四、时机判断
- 是否在敌方关键技能CD时进入危险区域（如对面大招刚好冷却好）？
- 团战参与时机是否正确（过早或过晚加入）？
- 是否在买活后又死了一次（二次死亡比第一次严重得多）？

### 五、交换评估
- 死亡是否换到了有价值的东西（击杀、推塔、Roshan、买活）？
- 1换1是否值得（看双方净资产和重要程度）？
- 团战中的死亡如果换到了多个击杀或建筑，可以视为正面交换

### 六、团队配合
- 队友位置是否合理？是否在支援范围内？
- 是否因为队友失误导致的死亡（如辅助未控、核心未输出）？
- 是否是合理的牺牲（为核心争取时间或保护建筑）？

### 评分标准
- **S（完美）**：不可避免的死亡，或有价值的主动牺牲
- **A（良好）**：只有小失误，基本操作正确
- **B（一般）**：有明显的判断失误但非致命习惯
- **C（较差）**：明显的位置或决策错误
- **D（很差）**：严重的基础失误（送头、贪兵、无视信息）

## 视野维度评分标准

评分 S:
- 在完整视野覆盖下操作
- 即使在夜间也有守卫覆盖
- 利用视野信息主动规避了潜在危险

评分 A:
- 有基本视野覆盖
- 对消失的敌人保持了一定警惕
- 在视野不足时选择了保守路线

评分 B:
- 有部分视野但不完整
- 没有充分利用已有的视野信息

评分 C:
- 在视野不足的情况下冒险
- 忽视了敌方英雄消失的信号
- 守卫到期后未及时补放

评分 D:
- 在完全没有视野的区域深入
- 多名敌方长时间消失却不撤退
- 夜间在无视野区域单独行动"""

_DEATH_ANALYSIS_RULES_EN = """## Death Analysis Evaluation Rules

### 1. Vision & Information
- Were there allied wards nearby before death? If not, it's a vision-related death
- Could enemy hero positions be seen on the minimap? How many enemies were unaccounted for?
- Did the enemy use Smoke of Deceit? If so, it's a high-quality gank — reduces player blame

### 2. Positioning & Movement
- Was the death location in a dangerous area (enemy high ground, unlit jungle, river)?
- Was the player pushing or farming alone without ally support range?
- Was the creep wave too far forward? Was greeding for creeps the cause?

### 3. Ability & Item Usage
- Were escape abilities used properly (Blink, Invisibility, Time Walk, etc.)?
- Were defensive items used in time (BKB, Eul's, Force Staff, etc.)?
- Were key abilities on cooldown? If so, the player should not have been in a dangerous position

### 4. Timing Judgment
- Did the player enter a dangerous area when enemy key abilities were off cooldown?
- Was teamfight participation timing correct (too early or too late)?
- Did the player die again after buyback (second death is much worse than the first)?

### 5. Trade Evaluation
- Did the death result in something valuable (kill, tower, Roshan, buyback forced)?
- Was a 1-for-1 trade worth it (compare net worth and importance)?
- Teamfight deaths that result in multiple kills or buildings can be considered positive trades

### 6. Team Coordination
- Were allies positioned properly? Were they within support range?
- Was the death caused by ally mistakes (support didn't disable, core didn't deal damage)?
- Was it a reasonable sacrifice (buying time for cores or protecting buildings)?

### Grading Scale
- **S (Perfect)**: Unavoidable death, or a valuable intentional sacrifice
- **A (Good)**: Only minor mistakes, fundamentally correct play
- **B (Average)**: Clear judgment errors but not habitual problems
- **C (Poor)**: Obvious positioning or decision-making errors
- **D (Very Poor)**: Serious fundamental mistakes (feeding, greeding, ignoring information)

## Vision Dimension Grading

Grade S:
- Operating under complete vision coverage
- Ward coverage even during nighttime
- Actively used vision information to avoid potential danger

Grade A:
- Basic vision coverage in place
- Maintained some awareness of missing enemies
- Chose conservative routes when vision was lacking

Grade B:
- Partial but incomplete vision
- Did not fully utilize available vision information

Grade C:
- Took risks in areas with insufficient vision
- Ignored signals of missing enemy heroes
- Failed to replace expired wards promptly

Grade D:
- Went deep into areas with zero vision
- Multiple enemies missing for extended periods without retreating
- Solo movement in unlit areas during nighttime"""


# ===================================================================
# Matchup knowledge (fallback)
# ===================================================================

# Matchup tips are now all in English (no separate _EN variant needed)
_FALLBACK_MATCHUP_TIPS: dict[str, dict[str, str]] = {
    "Drow Ranger": {
        "Viper": "Viper dominates lanes — Frost Arrows can't trade with Poison Attack. Farm safely from distance, don't try to fight head-on. After level 6 Marksmanship you can contribute in teamfights, but still avoid getting chased by Viper Strike.",
        "Bristleback": "Bristleback is a nightmare matchup. He'll get in your face with Quill Spray making it impossible to last-hit comfortably. Gust can push him back but has a long cooldown. Maintain distance and let allies handle melee heroes.",
        "Axe": "Axe blinking in with Berserker's Call is almost certain death. Keeping distance is key. Gust can interrupt his Blink but requires prediction. Stand in the backline during teamfights, let the frontline absorb Axe's abilities.",
        "Lion": "Lion's disable chain (Earth Spike + Hex + Finger) can one-shot you. Maintain vision and don't get caught. BKB or Linken's Sphere is a necessary defensive item.",
        "Dragon Knight": "DK in Dragon Form can push from range, but your DPS is higher. Key is keeping distance to avoid Dragon Tail stun. Marksmanship and Multishot deal heavy teamfight damage against DK.",
    },
    "Axe": {
        "Dragon Knight": "DK's Dragon Blood regen is high, making Battle Hunger less effective. But Berserker's Call + Counter Helix can win fights before level 6. After Blink, DK is one of the highest priority initiation targets.",
        "Viper": "Viper is nearly unwinnable in lane — Poison Attack sustain harass is too strong. Aim to not die, then flank with Blink Dagger later. Corrosive Skin slow is annoying when you try to turn.",
        "Bristleback": "Two tanky heroes laning against each other — a stalemate. Battle Hunger can harass BB, but Quill Spray trades back hard. Key is who hits level 6 first — Culling Blade can execute low-HP Bristleback.",
    },
    "Razor": {
        "Viper": "Two strong laners facing off. After Static Link steals attack damage, Razor can dominate Viper. But Poison Attack harass makes it hard to maintain Link range. Key is fighting while stolen damage is high after Link completes.",
        "Dragon Knight": "Static Link is very effective against DK since he doesn't want to leave the creep wave. After stealing damage, DK's last-hitting becomes painful. Eye of the Storm also deals high sustained damage to DK.",
    },
}

# Backward compatibility alias
_FALLBACK_MATCHUP_TIPS_EN = _FALLBACK_MATCHUP_TIPS


# ===================================================================
# Vision / fog-of-war principles
# ===================================================================

_VISION_PRINCIPLES = """## 视野系统基础知识

### 视野来源与参数
- **观察守卫（Observer Ward）**: 提供1600距离的地面视野，持续360秒（6分钟）。无数量限制但商店补货有冷却时间
- **岗哨守卫（Sentry Ward）**: 不提供地面视野，仅提供850距离的真视（反隐/排眼）。与观察守卫配合使用
- **防御塔**: 白天提供1900距离视野，夜间仅800距离。塔被摧毁后该区域永久失去固定视野
- **英雄视野**: 白天1800距离，夜间800距离。部分英雄例外（如夜魇猎人夜间视野1800）
- **昼夜交替**: 每5分钟交替一次（0:00-5:00白天，5:00-10:00夜间，以此类推）
- **夜间视野衰减**: 夜间视野约为白天的44%（800/1800），这意味着夜间的有效控制区域仅为白天的约19%（面积比）
- **高低地视野**: 低地单位无法看到高地上的单位，高地具有天然视野优势
- **树木遮挡**: 树木会阻挡视野传播，密林区域是天然的雾区

### 视野安全原则
- 当3名以上敌方英雄从地图消失超过15秒时，应立即撤退到安全区域（塔下或有视野覆盖区）
- 观察守卫到期后应在10秒内补放新眼，特别是河道、肉山坑、己方野区入口等关键位置
- 夜间在没有视野的区域移动极其危险——视野仅800意味着敌方可以从极近距离发动突袭
- 推进前必须确保侧翼有视野覆盖，至少提前30秒在推进路线两侧插眼
- 如果不知道敌方位置，默认假设他们在最危险的位置（你当前位置的gank路线上）
- 支援型英雄应在团战前60秒确保关键区域视野，而不是团战已经开始后再去插眼
- 进入敌方野区刷野前，必须确认至少3名敌方英雄的位置

### 死亡与视野的关系
- 大多数非团战死亡（约70%）都与视野缺失直接相关
- **死亡前视野检查清单**:
  1. 是否有守卫覆盖周围区域？覆盖半径是否足够？
  2. 有多少敌方英雄位置未知？（≥3人未知 = 极度危险）
  3. 未知敌人消失了多久？（>15秒 = 可能正在gank路上）
  4. 当前是白天还是夜间？（夜间危险系数翻倍）
  5. 附近是否有友方英雄可以支援？（TP距离内？走路距离内？）
  6. 距离最近的安全区域（塔/泉水）有多远？（>10秒路程 = 危险）
  7. 敌方是否有烟雾可用？（烟雾可以绕过守卫视野）

### 各阶段视野重点
- **对线期（0-10分钟）**:
  - 河道眼（控制神符和中路gank路线）
  - 对面拉野点眼（阻止敌方辅助拉野）
  - 符点眼（偶数分钟前放置）
  - 优先级：河道 > 拉野点 > 入侵性视野

- **中期（10-25分钟）**:
  - 入侵敌方野区眼（掌握敌方核心刷野路线）
  - 肉山坑视野（控制Roshan争夺信息）
  - 推进路线侧翼眼（防止推进时被包夹）
  - 优先级：肉山 > 入侵野区 > 推进路线

- **后期（25分钟+）**:
  - 高地前方眼（攻方推高地必备）
  - Roshan持续监控眼（后期Roshan价值极高）
  - 敌方基地入口眼（监控敌方出击路线）
  - 己方野区防守眼（防止敌方深入）
  - 优先级：Roshan > 高地 > 基地入口

### 不同位置的视野责任
- **5号位（硬辅助）**: 主要买眼责任，维持关键区域视野网络。目标：地图上始终有3-4个观察守卫存活
- **4号位（游走辅助）**: 补充买眼，负责侵略性视野（深入敌方区域插眼、反眼）。Smoke后优先放置深入眼
- **核心位（1/2/3号位）**: 不需要主动买眼，但必须读取视野信息做出安全决策。看到敌方消失必须有反应
- **所有位置**: 路过岗哨守卫覆盖区域时有义务排除敌方视野（反眼），这是团队责任

### 关键视野数值速查
| 项目 | 数值 |
|------|------|
| Observer Ward 视野 | 1600距离 |
| Observer Ward 持续时间 | 360秒（6分钟） |
| Sentry Ward 真视范围 | 850距离 |
| Sentry Ward 持续时间 | 420秒（7分钟） |
| 英雄白天视野 | 1800距离 |
| 英雄夜间视野 | 800距离 |
| 防御塔白天视野 | 1900距离 |
| 防御塔夜间视野 | 800距离 |
| 昼夜交替周期 | 每5分钟 |
| Smoke of Deceit 持续 | 35秒 |
| Smoke 探测距离 | 1025距离（靠近敌方英雄/塔时自动消散） |
"""

_VISION_PRINCIPLES_EN = """## Vision System Fundamentals

### Vision Sources & Parameters
- **Observer Ward**: Provides 1600 range ground vision, lasts 360 seconds (6 minutes). No quantity limit but shop restock has cooldown
- **Sentry Ward**: Does not provide ground vision, only 850 range true sight (anti-invis/deward). Used in combination with Observer Wards
- **Towers**: 1900 range vision during day, only 800 at night. Destroyed towers permanently lose that fixed vision
- **Hero Vision**: 1800 range during day, 800 at night. Some hero exceptions (e.g., Night Stalker has 1800 night vision)
- **Day/Night Cycle**: Alternates every 5 minutes (0:00-5:00 day, 5:00-10:00 night, etc.)
- **Night Vision Reduction**: Night vision is ~44% of daytime (800/1800), meaning effective control area is only ~19% (area ratio)
- **High/Low Ground Vision**: Low ground units cannot see high ground units — high ground has natural vision advantage
- **Tree Obstruction**: Trees block vision propagation — dense forests are natural fog areas

### Vision Safety Principles
- When 3+ enemy heroes are missing from the map for over 15 seconds, immediately retreat to safety (tower or warded area)
- Replace Observer Wards within 10 seconds of expiry, especially at key spots (river, Roshan pit, jungle entrances)
- Moving in unlit areas at night is extremely dangerous — 800 vision means enemies can ambush from very close range
- Before pushing, ensure flanks have ward coverage — place wards at least 30 seconds ahead on both sides of the push path
- If you don't know enemy positions, assume they are at the most dangerous location (on your current gank path)
- Support heroes should ensure key area vision 60 seconds before a teamfight, not after the fight has already started
- Before entering enemy jungle to farm, confirm the positions of at least 3 enemy heroes

### Death & Vision Relationship
- Most non-teamfight deaths (~70%) are directly related to vision gaps
- **Pre-Death Vision Checklist**:
  1. Were there wards covering the surrounding area? Was coverage radius sufficient?
  2. How many enemy heroes had unknown positions? (3+ unknown = extreme danger)
  3. How long have unknown enemies been missing? (>15 seconds = possibly en route to gank)
  4. Is it currently day or night? (Night doubles the danger factor)
  5. Are there allied heroes within support range? (TP range? Walking distance?)
  6. How far is the nearest safe zone (tower/fountain)? (>10 seconds travel = danger)
  7. Does the enemy have Smoke available? (Smoke bypasses ward vision)

### Vision Priorities by Phase
- **Laning Phase (0-10 minutes)**:
  - River wards (control runes and mid gank routes)
  - Anti-pull wards (prevent enemy support pulls)
  - Rune spot wards (place before even minutes)
  - Priority: River > Pull spots > Aggressive vision

- **Mid Game (10-25 minutes)**:
  - Aggressive enemy jungle wards (track enemy core farm patterns)
  - Roshan pit vision (control Roshan contest information)
  - Push path flank wards (prevent flanking during pushes)
  - Priority: Roshan > Enemy jungle > Push routes

- **Late Game (25+ minutes)**:
  - High ground approach wards (essential for attackers)
  - Roshan continuous monitoring (late Roshan is extremely valuable)
  - Enemy base entrance wards (monitor enemy movement)
  - Defensive jungle wards (prevent enemy deep incursions)
  - Priority: Roshan > High ground > Base entrances

### Vision Responsibilities by Position
- **Position 5 (Hard Support)**: Primary ward buyer, maintain key area ward network. Goal: Always have 3-4 Observer Wards alive on the map
- **Position 4 (Roaming Support)**: Supplementary ward buyer, responsible for aggressive vision (deep enemy wards, dewarding). Prioritize deep wards after Smoke
- **Core Positions (1/2/3)**: Don't need to buy wards, but must read vision information for safe decisions. Must react when enemies go missing
- **All Positions**: When passing through Sentry Ward coverage, everyone is responsible for dewarding — it's a team duty

### Key Vision Values Reference
| Item | Value |
|------|-------|
| Observer Ward Vision | 1600 range |
| Observer Ward Duration | 360 seconds (6 minutes) |
| Sentry Ward True Sight | 850 range |
| Sentry Ward Duration | 420 seconds (7 minutes) |
| Hero Daytime Vision | 1800 range |
| Hero Nighttime Vision | 800 range |
| Tower Daytime Vision | 1900 range |
| Tower Nighttime Vision | 800 range |
| Day/Night Cycle | Every 5 minutes |
| Smoke of Deceit Duration | 35 seconds |
| Smoke Detection Range | 1025 range (dissipates near enemy heroes/towers) |
"""


def get_vision_principles(lang: str = "zh") -> str:
    """Return Dota 2 vision analysis principles for LLM injection."""
    if lang == "en":
        return _VISION_PRINCIPLES_EN
    return _VISION_PRINCIPLES


# ===================================================================
# Lazy-loaded hero and matchup data (file-backed with fallback)
# ===================================================================

# Module-level references: start as fallback, upgraded on first access
HERO_DATA = _FALLBACK_HERO_DATA
_MATCHUP_TIPS = _FALLBACK_MATCHUP_TIPS
_hero_data_loaded = False


def _ensure_hero_data_loaded() -> None:
    """Load file-backed hero data if available."""
    global HERO_DATA, _MATCHUP_TIPS, _hero_data_loaded
    if _hero_data_loaded:
        return
    _hero_data_loaded = True
    try:
        from .data_manager import load_heroes, load_matchup_tips
        heroes = load_heroes()
        if heroes:
            # Merge fallback fields for heroes in both sources
            for hero_name, hero_dict in heroes.items():
                fallback = _FALLBACK_HERO_DATA.get(hero_name, {})
                for key in ("level1_strength", "level1_note"):
                    if key not in hero_dict and key in fallback:
                        hero_dict[key] = fallback[key]
            HERO_DATA = heroes
            logger.info("Hero data loaded from data_manager (%d heroes)", len(heroes))
        matchups = load_matchup_tips()
        if matchups:
            _MATCHUP_TIPS = matchups
    except ImportError:
        logger.debug("data_manager not available, using hardcoded hero data")
    except Exception as e:
        logger.warning("Failed to load hero data from data_manager: %s", e)


# ===================================================================
# Public API
# ===================================================================

def get_hero_info(hero_name: str, lang: str = "zh") -> dict:
    """Get hero role, attributes, power spikes, etc.

    Returns a copy of the hero data dict, or a fallback dict with
    basic defaults if the hero is not in the database.
    When lang="en", English-language fields replace their Chinese counterparts.
    """
    _ensure_hero_data_loaded()
    if hero_name in HERO_DATA:
        data = HERO_DATA[hero_name]
        # Content is now always English — no _en suffix needed
        return dict(data)
    if lang == "en":
        return {
            "role": "unknown",
            "attribute": "unknown",
            "attack_type": "unknown",
            "power_spikes": [],
            "weaknesses": [],
            "laning_tips": f"No detailed laning info available for {hero_name}.",
        }
    return {
        "role": "unknown",
        "attribute": "unknown",
        "attack_type": "unknown",
        "power_spikes": [],
        "weaknesses": [],
        "laning_tips": f"暂无{hero_name}的详细对线信息。",
    }


def get_matchup_info(hero_name: str, opponent_name: str, lang: str = "zh") -> str:
    """Get matchup-specific advice.

    Returns specific matchup advice if available, otherwise a generic
    analysis based on hero attributes.
    When lang="en", returns English matchup tips if available.
    """
    _ensure_hero_data_loaded()

    if lang == "en":
        tips = _FALLBACK_MATCHUP_TIPS_EN
        # Check direct matchup (English)
        if hero_name in tips and opponent_name in tips[hero_name]:
            return tips[hero_name][opponent_name]
        # Check reverse matchup and invert perspective (English)
        if opponent_name in tips and hero_name in tips[opponent_name]:
            reverse = tips[opponent_name][hero_name]
            return f"(From {opponent_name}'s perspective) {reverse}"
        # Generate generic advice based on hero data (English)
        hero_info = get_hero_info(hero_name, lang="en")
        opp_info = get_hero_info(opponent_name, lang="en")
        parts = []
        if hero_info["attack_type"] == "melee" and opp_info["attack_type"] == "ranged":
            parts.append(f"{hero_name} is melee, be careful of harass from {opponent_name} (ranged)")
        elif hero_info["attack_type"] == "ranged" and opp_info["attack_type"] == "melee":
            parts.append(f"{hero_name} is ranged, use range advantage to harass {opponent_name} (melee)")
        if opp_info.get("weaknesses"):
            parts.append(f"{opponent_name} weaknesses: {', '.join(opp_info['weaknesses'][:2])}")
        if not parts:
            return f"No detailed matchup info for {hero_name} vs {opponent_name}. Play based on hero characteristics."
        return ". ".join(parts) + "."

    # Chinese (default)
    # Check direct matchup
    if hero_name in _MATCHUP_TIPS and opponent_name in _MATCHUP_TIPS[hero_name]:
        return _MATCHUP_TIPS[hero_name][opponent_name]

    # Check reverse matchup and invert perspective
    if opponent_name in _MATCHUP_TIPS and hero_name in _MATCHUP_TIPS[opponent_name]:
        reverse = _MATCHUP_TIPS[opponent_name][hero_name]
        return f"（从{opponent_name}视角的对位信息）{reverse}"

    # Generate generic advice based on hero data
    hero_info = get_hero_info(hero_name)
    opp_info = get_hero_info(opponent_name)

    parts = []
    if hero_info["attack_type"] == "melee" and opp_info["attack_type"] == "ranged":
        parts.append(f"{hero_name}是近战英雄，对线{opponent_name}（远程）时注意被消耗")
    elif hero_info["attack_type"] == "ranged" and opp_info["attack_type"] == "melee":
        parts.append(f"{hero_name}是远程英雄，对线{opponent_name}（近战）时利用射程优势骚扰")

    if opp_info.get("weaknesses"):
        parts.append(f"{opponent_name}的弱点：{'、'.join(opp_info['weaknesses'][:2])}")

    if not parts:
        return f"暂无{hero_name}对阵{opponent_name}的详细对位信息。根据双方英雄特性合理对线。"

    return "。".join(parts) + "。"


def get_laning_principles(role: str, lang: str = "zh") -> str:
    """Get general laning principles for a role.

    Parameters
    ----------
    role : str
        One of: "carry", "mid", "offlane", "support_4", "support_5"
    lang : str
        Language: "zh" for Chinese (default), "en" for English.
    """
    # Normalize role names
    role_map = {
        "carry": "carry",
        "mid": "mid",
        "offlane": "offlane",
        "support_4": "support_4",
        "support_5": "support_5",
        "support": "support_5",
        "roamer": "support_4",
        "jungle": "offlane",
    }
    normalized = role_map.get(role, "carry")
    if lang == "en":
        return _LANING_PRINCIPLES_EN.get(normalized, _LANING_PRINCIPLES_EN["carry"])
    return _LANING_PRINCIPLES.get(normalized, _LANING_PRINCIPLES["carry"])


def get_phase_objectives(phase: str, role: str, lang: str = "zh") -> str:
    """Get what a player should be doing in each game phase.

    Parameters
    ----------
    phase : str
        One of: "pregame", "laning", "mid_game", "late_game"
    role : str
        One of: "carry", "mid", "offlane", "support_4", "support_5"
    lang : str
        Language: "zh" for Chinese (default), "en" for English.
    """
    role_map = {
        "carry": "carry",
        "mid": "mid",
        "offlane": "offlane",
        "support_4": "support_4",
        "support_5": "support_5",
        "support": "support_5",
        "roamer": "support_4",
    }
    normalized_role = role_map.get(role, "carry")

    if lang == "en":
        phase_data = _PHASE_OBJECTIVES_EN.get(phase, _PHASE_OBJECTIVES_EN.get("mid_game", {}))
        return phase_data.get(normalized_role, phase_data.get("carry", "Adjust strategy flexibly based on the current situation."))

    phase_data = _PHASE_OBJECTIVES.get(phase, _PHASE_OBJECTIVES.get("mid_game", {}))
    return phase_data.get(normalized_role, phase_data.get("carry", "根据当前局势灵活调整策略。"))


def get_death_analysis_rules(lang: str = "zh") -> str:
    """Get rules for analyzing deaths."""
    if lang == "en":
        return _DEATH_ANALYSIS_RULES_EN
    return _DEATH_ANALYSIS_RULES


def get_ability_item_knowledge(ctx: DecisionContext, lang: str = "zh") -> str:
    """Generate ability and item knowledge text for a decision context.

    Selectively injects data based on decision type:
    - Death/overextend: player abilities + killer abilities + defensive items
    - Bad trade: player + opponent abilities
    - Fight engage: player abilities + key active items
    - Others: brief item info only

    Returns formatted text, or empty string if game_data
    is unavailable or context lacks relevant data.
    When lang="en", returns English-language headers and labels.
    """
    if not _HAS_GAME_DATA:
        return ""

    _ensure_data_loaded()
    parts: list[str] = []
    hero_name = ctx.player_state.hero_name
    point_type = ctx.decision_point.point_type

    # --- Label definitions based on language ---
    if lang == "en":
        lbl_player_abilities = "### Player Abilities"
        lbl_player_items = "### Player Items"
        lbl_ultimate = "- Ultimate:"
        lbl_signature = "- Signature:"
        lbl_enemy_abilities = "### Nearby Enemy Abilities"
        lbl_active_items = "### Available Active Items"
        lbl_section_header = "## Ability & Item Info"
        cd_fmt = "(CD: {cd}s)"
        desc_sep = ": "
        def killer_header(kh): return f"### Killer Key Abilities ({kh})"
    else:
        lbl_player_abilities = "### 玩家技能状态"
        lbl_player_items = "### 玩家装备"
        lbl_ultimate = "- 大招："
        lbl_signature = "- 标志技能："
        lbl_enemy_abilities = "### 附近敌方技能"
        lbl_active_items = "### 可用主动道具"
        lbl_section_header = "## 技能与装备信息"
        cd_fmt = "（CD: {cd}秒）"
        desc_sep = "："
        def killer_header(kh): return f"### 击杀者关键技能（{kh}）"

    # --- Player abilities (always include if available) ---
    ability_levels = ctx.player_state.ability_levels
    if ability_levels:
        abilities_text = format_hero_abilities(hero_name, ability_levels, lang=lang)
        if abilities_text:
            parts.append(f"{lbl_player_abilities}\n{abilities_text}")

    # --- Player items (always include) ---
    player_items = ctx.player_state.items
    if player_items:
        items_text = format_player_items(player_items, lang=lang)
        if items_text:
            parts.append(f"{lbl_player_items}\n{items_text}")

    # --- Death / overextend: include killer's key abilities ---
    if point_type in ("death", "overextend") and ctx.decision_point.related_death:
        killer_hero = ctx.decision_point.related_death.killer_hero
        if killer_hero and killer_hero in ABILITY_DATA:
            killer_parts_list: list[str] = []
            killer_abilities = ABILITY_DATA[killer_hero]
            # Include ultimate
            if "ultimate" in killer_abilities:
                ult_text = format_ability_at_level(
                    killer_hero, killer_abilities["ultimate"], level=None, lang=lang
                )
                if ult_text:
                    killer_parts_list.append(f"{lbl_ultimate} {ult_text}")
            # Include signature ability (first non-ultimate)
            if "signature" in killer_abilities:
                sig_text = format_ability_at_level(
                    killer_hero, killer_abilities["signature"], level=None, lang=lang
                )
                if sig_text:
                    killer_parts_list.append(f"{lbl_signature} {sig_text}")
            if killer_parts_list:
                parts.append(
                    f"{killer_header(killer_hero)}\n"
                    + "\n".join(killer_parts_list)
                )

    # --- Bad trade: include nearby enemy abilities ---
    if point_type == "bad_trade" and ctx.enemy_states:
        enemy_ability_parts: list[str] = []
        for enemy in ctx.enemy_states[:2]:  # limit to 2 enemies to control token count
            if enemy.ability_levels:
                enemy_text = format_hero_abilities(enemy.hero_name, enemy.ability_levels, lang=lang)
                if enemy_text:
                    enemy_ability_parts.append(
                        f"**{enemy.hero_name}**:\n{enemy_text}"
                    )
        if enemy_ability_parts:
            parts.append(
                f"{lbl_enemy_abilities}\n" + "\n".join(enemy_ability_parts)
            )

    # --- Fight engage: highlight key active items ---
    if point_type == "fight_engage" and player_items:
        active_items = [
            item for item in player_items
            if item in ITEM_DATA and ITEM_DATA[item].get("has_active")
        ]
        if active_items:
            active_texts = []
            for item_name in active_items[:4]:  # limit to 4 active items
                item_info = ITEM_DATA[item_name]
                desc = item_info.get("active_desc", "")
                cd = item_info.get("cooldown", "")
                line = f"- **{item_name}**"
                if desc:
                    line += f"{desc_sep}{desc}"
                if cd:
                    line += cd_fmt.format(cd=cd)
                active_texts.append(line)
            if active_texts:
                parts.append(
                    f"{lbl_active_items}\n" + "\n".join(active_texts)
                )

    if not parts:
        return ""

    return f"{lbl_section_header}\n\n" + "\n\n".join(parts)


def analyze_level1_team_strength(match, player_slot: int, lang: str = "zh") -> str:
    """Compare both teams' level 1 fight capability."""
    _ensure_hero_data_loaded()

    player_team = "radiant" if player_slot < 128 else "dire"
    teams = {"radiant": [], "dire": []}

    for p in match.players:
        hero_info = get_hero_info(p.hero_name, lang=lang)
        strength = hero_info.get("level1_strength", "average")
        teams[p.team].append({
            "hero": p.hero_name,
            "strength": strength,
            "attack_type": hero_info.get("attack_type", "unknown"),
        })

    def team_score(heroes):
        score_map = {"strong": 3, "average": 2, "weak": 1}
        return sum(score_map.get(h["strength"], 2) for h in heroes)

    ally_score = team_score(teams[player_team])
    enemy_team = "dire" if player_team == "radiant" else "radiant"
    enemy_score = team_score(teams[enemy_team])

    ally_strong = [h["hero"] for h in teams[player_team] if h["strength"] == "strong"]
    enemy_strong = [h["hero"] for h in teams[enemy_team] if h["strength"] == "strong"]
    ally_melee = sum(1 for h in teams[player_team] if h["attack_type"] == "melee")
    enemy_melee = sum(1 for h in teams[enemy_team] if h["attack_type"] == "melee")

    if lang == "en":
        lines = ["## Level 1 Team Comparison"]
        if ally_strong:
            lines.append(f"- Allied strong level 1 heroes: {', '.join(ally_strong)}")
        if enemy_strong:
            lines.append(f"- Enemy strong level 1 heroes: {', '.join(enemy_strong)}")
        lines.append(f"- Allied team score: {ally_score}/15, Enemy team score: {enemy_score}/15")
        if ally_score > enemy_score + 2:
            lines.append("- Assessment: YOUR TEAM is stronger at level 1 — favorable rune contest")
        elif enemy_score > ally_score + 2:
            lines.append("- Assessment: ENEMY TEAM is stronger at level 1 — avoid rune contest, go to lane")
        else:
            lines.append("- Assessment: Teams are roughly even at level 1 — contest depends on numbers at the rune spot")
        lines.append(f"- Bounty rune value: ~40 gold. Death cost: 100+ gold + missed lane XP/gold")
    else:
        lines = ["## 1级团战能力对比"]
        if ally_strong:
            lines.append(f"- 己方1级强势英雄：{'、'.join(ally_strong)}")
        if enemy_strong:
            lines.append(f"- 敌方1级强势英雄：{'、'.join(enemy_strong)}")
        lines.append(f"- 己方团队评分：{ally_score}/15，敌方团队评分：{enemy_score}/15")
        if ally_score > enemy_score + 2:
            lines.append("- 评估：己方1级更强——适合争夺赏金符")
        elif enemy_score > ally_score + 2:
            lines.append("- 评估：敌方1级更强——应避免争夺，直接去线上")
        else:
            lines.append("- 评估：双方1级实力接近——取决于赏金符点的人数")
        lines.append("- 赏金符价值约40金。死亡代价：100+金 + 错过的线上经验和金钱")

    return "\n".join(lines)


def get_knowledge_for_context(ctx: DecisionContext, lang: str = "zh") -> str:
    """Assemble knowledge text for a decision context.

    Uses RAG retrieval when available, falls back to hardcoded knowledge.
    """
    store = _get_knowledge_store(lang=lang)
    if store is not None:
        return _rag_knowledge(ctx, store, lang=lang)
    return _legacy_knowledge(ctx, lang=lang)


def _rag_knowledge(ctx: DecisionContext, store, lang: str = "zh") -> str:
    """Retrieve knowledge using the RAG store."""
    chunks = store.retrieve(ctx, budget=3000)
    if not chunks:
        return _legacy_knowledge(ctx, lang=lang)  # fallback if no chunks found

    sections = []
    for chunk in chunks:
        sections.append(chunk.content)

    # Always append ability/item data if available (not in RAG store)
    if _HAS_GAME_DATA:
        ability_item_text = get_ability_item_knowledge(ctx, lang=lang)
        if ability_item_text:
            sections.append(ability_item_text)

    # Always append pro benchmark data if available
    try:
        from .pro_patterns import get_benchmark_context, resolve_role
        hero_name = ctx.player_state.hero_name
        role = resolve_role(None, hero_name)
        benchmark_text = get_benchmark_context(hero_name, role, ctx.decision_point.time)
        if benchmark_text:
            sections.append(benchmark_text)
    except (ImportError, Exception):
        pass

    return "\n\n".join(sections)


def _legacy_knowledge(context: DecisionContext, lang: str = "zh") -> str:
    """Build complete knowledge injection string for a decision context.

    Assembles all relevant knowledge based on the context:
    - Hero info for the player and nearby enemies
    - Matchup info for the lane
    - Phase objectives
    - Laning principles (if in laning phase)
    - Death analysis rules (if decision is a death)

    Parameters
    ----------
    context : DecisionContext
        The full context for a decision point.
    lang : str
        Language: "zh" for Chinese (default), "en" for English.

    Returns
    -------
    str
        A formatted knowledge string ready for LLM injection.
    """
    sections: list[str] = []

    # --- Player hero info ---
    hero_name = context.player_state.hero_name
    hero_info = get_hero_info(hero_name, lang=lang)

    if lang == "en":
        spike_sep = ", "
        weak_sep = ", "
        sections.append(
            f"## Player Hero: {hero_name}\n"
            f"- Role: {hero_info['role']}\n"
            f"- Attribute: {hero_info['attribute']} ({hero_info['attack_type']})\n"
            f"- Power Spikes: {spike_sep.join(hero_info.get('power_spikes', []))}\n"
            f"- Weaknesses: {weak_sep.join(hero_info.get('weaknesses', []))}\n"
            f"- Laning Tips: {hero_info.get('laning_tips', 'N/A')}"
        )
    else:
        sections.append(
            f"## 玩家英雄：{hero_name}\n"
            f"- 定位：{hero_info['role']}\n"
            f"- 属性：{hero_info['attribute']}（{hero_info['attack_type']}）\n"
            f"- 强势时间点：{'、'.join(hero_info.get('power_spikes', []))}\n"
            f"- 弱点：{'、'.join(hero_info.get('weaknesses', []))}\n"
            f"- 对线要点：{hero_info.get('laning_tips', '暂无')}"
        )

    # --- Enemy hero info and matchups ---
    if context.enemy_states:
        if lang == "en":
            enemy_section_parts = ["## Enemy Hero Info"]
            for enemy in context.enemy_states:
                e_info = get_hero_info(enemy.hero_name, lang=lang)
                enemy_section_parts.append(
                    f"### {enemy.hero_name}\n"
                    f"- Role: {e_info['role']}, {e_info['attack_type']}\n"
                    f"- Power Spikes: {', '.join(e_info.get('power_spikes', [])[:2])}\n"
                    f"- Weaknesses: {', '.join(e_info.get('weaknesses', [])[:2])}"
                )
                matchup = get_matchup_info(hero_name, enemy.hero_name, lang=lang)
                if "No detailed" not in matchup:
                    enemy_section_parts.append(f"- Matchup Tips: {matchup}")
        else:
            enemy_section_parts = ["## 敌方英雄信息"]
            for enemy in context.enemy_states:
                e_info = get_hero_info(enemy.hero_name, lang=lang)
                enemy_section_parts.append(
                    f"### {enemy.hero_name}\n"
                    f"- 定位：{e_info['role']}，{e_info['attack_type']}\n"
                    f"- 强势点：{'、'.join(e_info.get('power_spikes', [])[:2])}\n"
                    f"- 弱点：{'、'.join(e_info.get('weaknesses', [])[:2])}"
                )
                matchup = get_matchup_info(hero_name, enemy.hero_name, lang=lang)
                if "暂无" not in matchup:
                    enemy_section_parts.append(f"- 对位要点：{matchup}")

        sections.append("\n".join(enemy_section_parts))

    # --- Phase objectives ---
    player_role = hero_info.get("role", "carry")
    phase = context.game_phase or "mid_game"

    if lang == "en":
        phase_name_map = {"pregame": "Pregame", "laning": "Laning", "mid_game": "Mid Game", "late_game": "Late Game"}
        phase_display = phase_name_map.get(phase, "Mid Game")
        objectives = get_phase_objectives(phase, player_role, lang=lang)
        sections.append(
            f"## Current Phase Objectives ({phase_display} - {player_role})\n{objectives}"
        )
    else:
        phase_name_map = {"pregame": "赛前", "laning": "对线期", "mid_game": "中期", "late_game": "后期"}
        phase_cn = phase_name_map.get(phase, "中期")
        objectives = get_phase_objectives(phase, player_role, lang=lang)
        sections.append(
            f"## 当前阶段目标（{phase_cn} - {player_role}）\n{objectives}"
        )

    # --- Laning principles (only during laning phase) ---
    if phase == "laning":
        principles = get_laning_principles(player_role, lang=lang)
        sections.append(principles)

    # --- Death analysis rules (if this is a death event) ---
    dp = context.decision_point
    if dp.point_type in ("death", "bad_trade", "overextend"):
        sections.append(get_death_analysis_rules(lang=lang))

    # --- Vision principles for deaths, overextensions, and dangerous situations ---
    vision_injected = False
    if dp.point_type in ("death", "overextend", "no_retreat"):
        sections.append(get_vision_principles(lang=lang))
        vision_injected = True

    # Also add vision principles if vision analysis shows danger
    if not vision_injected and context.vision_analysis and context.vision_analysis.state_at_decision:
        if context.vision_analysis.state_at_decision.danger_level in ("danger", "critical"):
            sections.append(get_vision_principles(lang=lang))

    # Professional benchmark context
    try:
        from .pro_patterns import get_benchmark_context, resolve_role
        _pro_hero = context.player_state.hero_name
        _pro_role = resolve_role(None, _pro_hero)  # resolve from hero data only (no player obj here)
        benchmark_text = get_benchmark_context(_pro_hero, _pro_role, context.decision_point.time)
        if benchmark_text:
            sections.append(benchmark_text)
    except ImportError:
        pass
    except Exception:
        pass

    # --- Ability and item data injection ---
    if _HAS_GAME_DATA:
        ability_item_text = get_ability_item_knowledge(context, lang=lang)
        if ability_item_text:
            sections.append(ability_item_text)

    return "\n\n---\n\n".join(sections)
