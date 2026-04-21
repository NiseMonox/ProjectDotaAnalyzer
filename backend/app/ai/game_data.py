"""Dota 2 ability and item data for AI coaching system.

Contains structured data for hero abilities and items, used to inject
specific mechanical information into LLM evaluation prompts.

Data loading priority:
1. File-backed data from data_manager (gamedata JSON + editorial JSON)
2. Hardcoded fallback dicts (_FALLBACK_ABILITY_DATA, _FALLBACK_ITEM_DATA)
"""
from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

# Dynamic patch version: prefer data_manager, fallback to hardcoded
def _get_patch_version() -> str:
    try:
        from .data_manager import get_patch_version
        return get_patch_version()
    except ImportError:
        return "7.37"

PATCH_VERSION = "7.37"  # static default; use _get_patch_version() for dynamic

# Each ability: key, type, damage_type, pierces_bkb, cooldown[], mana_cost[],
# optional: damage[], stun_duration[], duration[], cast_range[], radius[],
# slow_pct[], proc_chance, is_ultimate, description (Chinese), coaching_tip (Chinese)

# These hardcoded dicts serve as fallback when JSON data files are unavailable.
# The public ABILITY_DATA / ITEM_DATA names are lazy-loaded from data_manager.
_FALLBACK_ABILITY_DATA: dict[str, dict[str, dict]] = {
    # =========================================================================
    # 1. Hoodwink
    # =========================================================================
    "Hoodwink": {
        "Acorn Shot": {
            "key": "hoodwink_acorn_shot",
            "type": "active",
            "damage_type": "physical",
            "pierces_bkb": False,
            "cooldown": [18, 14, 10, 6],
            "mana_cost": [60, 70, 80, 90],
            "damage": [70, 100, 130, 160],
            "cast_range": [575, 575, 575, 575],
            "slow_pct": [100, 100, 100, 100],
            "duration": [0.3, 0.3, 0.3, 0.3],
            "description": "向目标发射橡果，弹射对周围敌人造成物理伤害并短暂减速。可对树木施放，种下新树并弹射。",
            "coaching_tip": "对树施放可以种树配合Bushwhack束缚；线上优先用来补刀或骚扰。",
        },
        "Bushwhack": {
            "key": "hoodwink_bushwhack",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [14, 14, 14, 14],
            "mana_cost": [90, 100, 110, 120],
            "damage": [90, 160, 230, 300],
            "stun_duration": [1.4, 1.6, 1.8, 2.0],
            "cast_range": [1100, 1100, 1100, 1100],
            "radius": [265, 265, 265, 265],
            "description": "投掷陷阱，将范围内靠近树木的敌人束缚在树上，造成魔法伤害和眩晕。",
            "coaching_tip": "先用Acorn Shot种树再接Bushwhack是核心连招；注意敌人必须在树木附近才会被束缚。",
        },
        "Scurry": {
            "key": "hoodwink_scurry",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [18, 16, 14, 12],
            "mana_cost": [35, 35, 35, 35],
            "duration": [4, 4, 4, 4],
            "description": "被动在树林附近获得闪避。主动施放获得树林穿行和移速加成。",
            "coaching_tip": "逃跑时主动使用穿越树林甩开追击；被动闪避在线上可以规避部分右键伤害。",
        },
        "Sharpshooter": {
            "key": "hoodwink_sharpshooter",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "is_ultimate": True,
            "cooldown": [45, 35, 25],
            "mana_cost": [125, 150, 175],
            "damage": [400, 600, 800],
            "cast_range": [3000, 3000, 3000],
            "slow_pct": [40, 50, 60],
            "duration": [4, 4, 4],
            "description": "蓄力后发射远程穿刺箭矢，造成高额魔法伤害并降低移速和治疗效果。",
            "coaching_tip": "蓄力时间较长且会暴露位置，优先在视野外使用；配合队友控制命中率更高。",
        },
    },
    # =========================================================================
    # 2. Drow Ranger
    # =========================================================================
    "Drow Ranger": {
        "Frost Arrows": {
            "key": "drow_ranger_frost_arrows",
            "type": "autocast",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [0, 0, 0, 0],
            "mana_cost": [12, 12, 12, 12],
            "damage": [10, 20, 30, 40],
            "slow_pct": [16, 24, 32, 40],
            "duration": [1.5, 1.5, 1.5, 1.5],
            "description": "自动施法攻击附加冰霜效果，造成额外魔法伤害并减速敌人。",
            "coaching_tip": "线上手动施放可以避免吸引仇恨；追杀时保持自动施放持续减速。",
        },
        "Gust": {
            "key": "drow_ranger_wave_of_silence",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [18, 15, 12, 9],
            "mana_cost": [70, 80, 90, 100],
            "duration": [3.0, 4.0, 5.0, 6.0],
            "cast_range": [900, 900, 900, 900],
            "description": "释放一股强风，击退并沉默前方扇形区域内的敌人。",
            "coaching_tip": "关键的自保技能；面对突脸英雄时预判使用可以拉开距离，沉默持续时间很长。",
        },
        "Multishot": {
            "key": "drow_ranger_multishot",
            "type": "active",
            "damage_type": "physical",
            "pierces_bkb": True,
            "cooldown": [26, 22, 18, 14],
            "mana_cost": [75, 85, 95, 105],
            "damage": [75, 100, 125, 150],
            "duration": [1.75, 1.75, 1.75, 1.75],
            "slow_pct": [40, 40, 40, 40],
            "description": "向锥形区域发射多波箭矢，造成物理伤害和减速，含攻击特效。",
            "coaching_tip": "团战中选好角度打出最大范围伤害；清野时配合使用加速刷野。",
        },
        "Marksmanship": {
            "key": "drow_ranger_marksmanship",
            "type": "passive",
            "damage_type": "physical",
            "pierces_bkb": True,
            "is_ultimate": True,
            "cooldown": [],
            "mana_cost": [],
            "proc_chance": 40,
            "description": "被动获得额外敏捷。攻击有概率无视护甲造成额外物理伤害，对小兵直接秒杀。",
            "coaching_tip": "敌人靠近400码内会失效，保持站位距离是核心；搭配飓风长矛等远程装备效果更佳。",
        },
    },
    # =========================================================================
    # 3. Warlock
    # =========================================================================
    "Warlock": {
        "Fatal Bonds": {
            "key": "warlock_fatal_bonds",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [24, 20, 16, 12],
            "mana_cost": [120, 130, 140, 150],
            "cast_range": [1000, 1000, 1000, 1000],
            "duration": [18, 18, 18, 18],
            "description": "将多个敌方单位连结，其中一个受到伤害时会将伤害分担给所有连结目标。",
            "coaching_tip": "团战前先手使用可以大幅增加团队总伤害；搭配大招AoE伤害效果极佳。",
        },
        "Shadow Word": {
            "key": "warlock_shadow_word",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [14, 12, 10, 8],
            "mana_cost": [120, 130, 140, 150],
            "damage": [15, 25, 35, 45],
            "duration": [12, 12, 12, 12],
            "cast_range": [600, 600, 600, 600],
            "description": "对友方单位施放持续治疗，对敌方单位施放持续伤害。",
            "coaching_tip": "线上用于维持carry血量或持续消耗对手；注意每秒伤害/治疗，总量可观。",
        },
        "Upheaval": {
            "key": "warlock_upheaval",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [50, 42, 34, 26],
            "mana_cost": [100, 110, 120, 130],
            "radius": [650, 650, 650, 650],
            "cast_range": [1200, 1200, 1200, 1200],
            "slow_pct": [30, 42, 54, 66],
            "description": "持续引导，对范围内敌人施加逐渐增强的减速效果。",
            "coaching_tip": "配合大招地狱火使用效果最佳；引导期间不要取消，减速会逐渐叠加到很高。",
        },
        "Chaotic Offering": {
            "key": "warlock_rain_of_chaos",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": True,
            "is_ultimate": True,
            "cooldown": [170, 150, 130],
            "mana_cost": [250, 350, 450],
            "damage": [100, 200, 300],
            "stun_duration": [1.0, 1.25, 1.5],
            "radius": [600, 600, 600],
            "cast_range": [1200, 1200, 1200],
            "description": "召唤地狱火，落地时对范围内敌人造成魔法伤害和眩晕。地狱火持续存在并攻击敌人。",
            "coaching_tip": "先Fatal Bonds再放大招是经典连招；可穿BKB眩晕，是关键先手/反手团控。",
        },
    },
    # =========================================================================
    # 4. Razor
    # =========================================================================
    "Razor": {
        "Plasma Field": {
            "key": "razor_plasma_field",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [13, 12, 11, 10],
            "mana_cost": [125, 125, 125, 125],
            "damage": [80, 130, 180, 230],
            "radius": [700, 700, 700, 700],
            "slow_pct": [30, 35, 40, 45],
            "description": "释放扩散再收缩的等离子场，对经过的敌人造成魔法伤害和减速。距离越远伤害越高。",
            "coaching_tip": "尽量在最大范围命中敌人以获得最高伤害；清兵清野效率很高。",
        },
        "Static Link": {
            "key": "razor_static_link",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": True,
            "cooldown": [32, 28, 24, 20],
            "mana_cost": [65, 65, 65, 65],
            "damage": [10, 20, 30, 40],
            "duration": [8, 8, 8, 8],
            "cast_range": [550, 550, 550, 550],
            "description": "与目标建立连接，持续窃取其攻击力。连接持续期间攻击力转移不断增加。",
            "coaching_tip": "对线期针对敌方核心使用可以压制其补刀；保持连接完整是关键，不要被拉断。",
        },
        "Storm Surge": {
            "key": "razor_storm_surge",
            "type": "passive",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [],
            "mana_cost": [],
            "damage": [20, 40, 60, 80],
            "description": "被动获得移速加成。被攻击时有概率释放闪电打击攻击者并减速。",
            "coaching_tip": "被动移速使Razor很难被风筝；配合Static Link追着敌人跑效果极佳。",
        },
        "Eye of the Storm": {
            "key": "razor_eye_of_the_storm",
            "type": "active",
            "damage_type": "physical",
            "pierces_bkb": True,
            "is_ultimate": True,
            "cooldown": [70, 60, 50],
            "mana_cost": [200, 250, 300],
            "damage": [60, 75, 90],
            "duration": [25, 30, 35],
            "description": "召唤风暴，持续对附近生命值最低的敌方单位释放闪电，每次命中降低其护甲。",
            "coaching_tip": "推塔时开大可以攻击建筑降低护甲；团战冲入人群配合BKB持续输出。",
        },
    },
    # =========================================================================
    # 5. Axe
    # =========================================================================
    "Axe": {
        "Berserker's Call": {
            "key": "axe_berserkers_call",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": True,
            "cooldown": [16, 14, 12, 10],
            "mana_cost": [80, 90, 100, 110],
            "duration": [2.0, 2.4, 2.8, 3.2],
            "radius": [315, 315, 315, 315],
            "description": "嘲讽范围内所有敌方单位强制攻击自己，期间获得额外护甲。可穿BKB。",
            "coaching_tip": "跳刀接Call是经典先手，尽量多嘲讽人；穿BKB特性使其成为顶级先手控制。",
        },
        "Battle Hunger": {
            "key": "axe_battle_hunger",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [5, 5, 5, 5],
            "mana_cost": [50, 60, 70, 80],
            "damage": [16, 24, 32, 40],
            "duration": [12, 12, 12, 12],
            "slow_pct": [12, 14, 16, 18],
            "cast_range": [750, 750, 750, 750],
            "description": "对目标施加持续伤害和减速，目标击杀单位后效果解除。每有一个受影响的敌人，Axe获得移速加成。",
            "coaching_tip": "线上持续施压对手，对方不补刀就持续受伤；多层叠加移速加成很可观。",
        },
        "Counter Helix": {
            "key": "axe_counter_helix",
            "type": "passive",
            "damage_type": "physical",
            "pierces_bkb": True,
            "cooldown": [0.3, 0.3, 0.3, 0.3],
            "mana_cost": [],
            "damage": [60, 100, 140, 180],
            "proc_chance": 20,
            "radius": [275, 275, 275, 275],
            "description": "被攻击时有概率旋转反击，对周围敌人造成物理伤害。",
            "coaching_tip": "切线/拉野时利用小兵围殴触发反击斩清兵极快；被多单位攻击时触发率更高。",
        },
        "Culling Blade": {
            "key": "axe_culling_blade",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": True,
            "is_ultimate": True,
            "cooldown": [75, 65, 55],
            "mana_cost": [100, 150, 200],
            "damage": [150, 250, 350],
            "cast_range": [200, 200, 200],
            "description": "斩杀生命值低于阈值的敌人（阈值随等级增加），成功斩杀后范围内友军获得移速和攻速加成。",
            "coaching_tip": "注意斩杀线：250/350/450血量以下直接击杀；成功斩杀后的团队加速别浪费。",
        },
    },
    # =========================================================================
    # 6. Shadow Shaman
    # =========================================================================
    "Shadow Shaman": {
        "Ether Shock": {
            "key": "shadow_shaman_ether_shock",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [14, 12, 10, 8],
            "mana_cost": [95, 110, 125, 140],
            "damage": [140, 200, 260, 320],
            "cast_range": [600, 600, 600, 600],
            "description": "对目标及其周围多个敌人释放闪电链，造成魔法伤害。",
            "coaching_tip": "线上高效的清兵和骚扰技能；注意蓝量消耗较高，合理选择使用时机。",
        },
        "Hex": {
            "key": "shadow_shaman_voodoo",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [18, 14, 10, 6],
            "mana_cost": [110, 120, 130, 140],
            "duration": [1.25, 2.0, 2.75, 3.5],
            "cast_range": [500, 500, 500, 500],
            "description": "将目标变为小动物，期间无法攻击、施法或使用物品，移速降低。",
            "coaching_tip": "极长的点控技能，满级3.5秒变羊非常强力；优先对高威胁核心使用。",
        },
        "Shackles": {
            "key": "shadow_shaman_shackles",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [10, 10, 10, 10],
            "mana_cost": [140, 150, 160, 170],
            "damage": [60, 120, 180, 240],
            "duration": [2.75, 3.25, 3.75, 4.25],
            "cast_range": [400, 400, 400, 400],
            "description": "引导技能，束缚目标使其无法行动，持续造成魔法伤害。",
            "coaching_tip": "引导期间自己无法行动容易被打断，注意保持安全位置；配合Hex连控时间极长。",
        },
        "Mass Serpent Ward": {
            "key": "shadow_shaman_mass_serpent_ward",
            "type": "active",
            "damage_type": "physical",
            "pierces_bkb": True,
            "is_ultimate": True,
            "cooldown": [120, 100, 80],
            "mana_cost": [200, 350, 500],
            "damage": [55, 75, 95],
            "duration": [45, 45, 45],
            "cast_range": [550, 550, 550],
            "description": "召唤一群毒蛇守卫围成一圈，攻击范围内敌人和建筑。",
            "coaching_tip": "推塔时放在塔旁可以快速拆塔；也可以用来困住敌人或封锁通道。",
        },
    },
    # =========================================================================
    # 7. Viper
    # =========================================================================
    "Viper": {
        "Poison Attack": {
            "key": "viper_poison_attack",
            "type": "autocast",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [0, 0, 0, 0],
            "mana_cost": [20, 20, 20, 20],
            "damage": [6, 12, 18, 24],
            "slow_pct": [10, 16, 22, 28],
            "duration": [4, 4, 4, 4],
            "description": "自动施法攻击附加毒素，造成持续魔法伤害和减速，可叠加。",
            "coaching_tip": "线上手动施放避免吸引仇恨，持续骚扰对手；毒素可叠加，追杀时保持攻击。",
        },
        "Nethertoxin": {
            "key": "viper_nethertoxin",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [14, 13, 12, 11],
            "mana_cost": [70, 70, 70, 70],
            "damage": [30, 55, 80, 105],
            "duration": [4, 4, 4, 4],
            "cast_range": [900, 900, 900, 900],
            "radius": [400, 400, 400, 400],
            "description": "在地面创建毒液池，范围内敌人受到持续魔法伤害且被动技能失效。",
            "coaching_tip": "可以克制依赖被动的英雄如Bristleback、PA；配合控制铺在敌人脚下效果最佳。",
        },
        "Corrosive Skin": {
            "key": "viper_corrosive_skin",
            "type": "passive",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [],
            "mana_cost": [],
            "damage": [10, 20, 30, 40],
            "slow_pct": [10, 15, 20, 25],
            "duration": [4, 4, 4, 4],
            "description": "被动获得魔法抗性。受到攻击或技能时对攻击者施加持续伤害和减速。",
            "coaching_tip": "被动使Viper非常肉，对线极难被杀；面对物理爆发英雄需注意不够时补甲。",
        },
        "Viper Strike": {
            "key": "viper_viper_strike",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": True,
            "is_ultimate": True,
            "cooldown": [30, 20, 10],
            "mana_cost": [125, 175, 225],
            "damage": [100, 145, 190],
            "slow_pct": [40, 60, 80],
            "duration": [5.0, 5.0, 5.0],
            "cast_range": [500, 500, 500],
            "description": "对目标喷射剧毒，造成持续魔法伤害和大幅减速。可穿BKB。",
            "coaching_tip": "满级CD仅10秒可频繁使用；穿BKB的强力减速非常适合粘住敌方核心。",
        },
    },
    # =========================================================================
    # 8. Lion
    # =========================================================================
    "Lion": {
        "Earth Spike": {
            "key": "lion_impale",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [12, 12, 12, 12],
            "mana_cost": [70, 100, 130, 160],
            "damage": [80, 140, 200, 260],
            "stun_duration": [1.4, 1.6, 1.8, 2.0],
            "cast_range": [500, 500, 500, 500],
            "description": "沿直线释放地刺，击中敌人后将其击飞并眩晕。",
            "coaching_tip": "线性技能可以同时控住多人，走位寻找多人命中角度；是主要先手技能。",
        },
        "Hex": {
            "key": "lion_voodoo",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [24, 20, 16, 12],
            "mana_cost": [125, 150, 175, 200],
            "duration": [1.75, 2.5, 3.25, 4.0],
            "cast_range": [500, 500, 500, 500],
            "description": "将目标变为青蛙，无法攻击、施法或使用物品。",
            "coaching_tip": "配合Earth Spike连控时间极长，是击杀保证；优先变羊有跳刀/BKB的目标。",
        },
        "Mana Drain": {
            "key": "lion_mana_drain",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [16, 12, 8, 4],
            "mana_cost": [10, 10, 10, 10],
            "duration": [5, 5, 5, 5],
            "cast_range": [850, 850, 850, 850],
            "description": "引导技能，持续吸取目标魔法值，自身获得等量魔法。",
            "coaching_tip": "对线期吸蓝可以让对手无法施法；满级CD极短可以保持蓝量充沛。",
        },
        "Finger of Death": {
            "key": "lion_finger_of_death",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "is_ultimate": True,
            "cooldown": [160, 100, 40],
            "mana_cost": [200, 420, 650],
            "damage": [600, 725, 850],
            "cast_range": [900, 900, 900],
            "description": "对目标释放死亡之指，造成高额魔法伤害。每次击杀会永久增加伤害。",
            "coaching_tip": "尽量用大招收割获得永久伤害加成；满级CD仅40秒不要舍不得用。",
        },
    },
    # =========================================================================
    # 9. Dragon Knight
    # =========================================================================
    "Dragon Knight": {
        "Breathe Fire": {
            "key": "dragon_knight_breathe_fire",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [14, 13, 12, 11],
            "mana_cost": [100, 110, 120, 130],
            "damage": [75, 150, 225, 300],
            "cast_range": [700, 700, 700, 700],
            "description": "向前方喷出火焰，对锥形区域敌人造成魔法伤害并降低其攻击力（仅降低攻击力，不降低攻速）。",
            "coaching_tip": "线上清兵和消耗的主要技能；降攻击力效果可以压制对方核心的对拼伤害（注意：只降攻击力不降攻速）。",
        },
        "Dragon Tail": {
            "key": "dragon_knight_dragon_tail",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [12, 11, 10, 9],
            "mana_cost": [100, 100, 100, 100],
            "damage": [25, 75, 125, 175],
            "stun_duration": [2.5, 2.75, 3.0, 3.25],
            "cast_range": [150, 150, 150, 150],
            "description": "近战眩晕目标，变龙后施法距离大幅增加。",
            "coaching_tip": "变龙后施法距离增加到400，是强力远程眩晕；眩晕时间很长用于先手或保护队友。",
        },
        "Dragon Blood": {
            "key": "dragon_knight_dragon_blood",
            "type": "passive",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [],
            "mana_cost": [],
            "description": "被动增加生命恢复和护甲，使Dragon Knight极为耐打。",
            "coaching_tip": "让DK可以持续赖线不回家，配合魂戒可以频繁使用技能；线上优势来自于续航。",
        },
        "Elder Dragon Form": {
            "key": "dragon_knight_elder_dragon_form",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "is_ultimate": True,
            "cooldown": [115, 100, 85],
            "mana_cost": [50, 50, 50],
            "duration": [60, 60, 60],
            "description": "变身为巨龙形态，获得远程攻击和额外能力。1级绿龙毒素，2级红龙溅射，3级蓝龙冰冻减速。",
            "coaching_tip": "变龙后推塔极快，配合团队节奏推进；3级蓝龙团战价值极高，注意大招持续时间。",
        },
    },
    # =========================================================================
    # 10. Bristleback
    # =========================================================================
    "Bristleback": {
        "Viscous Nasal Goo": {
            "key": "bristleback_viscous_nasal_goo",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [1.75, 1.5, 1.25, 1.0],
            "mana_cost": [12, 16, 20, 24],
            "duration": [5, 5, 5, 5],
            "slow_pct": [15, 20, 25, 30],
            "cast_range": [600, 600, 600, 600],
            "description": "向目标喷射鼻涕，降低其护甲和移速，可多次叠加。",
            "coaching_tip": "多层叠加后减甲和减速极高，追杀时保持施放；搭配Quill Spray物理伤害爆表。",
        },
        "Quill Spray": {
            "key": "bristleback_quill_spray",
            "type": "active",
            "damage_type": "physical",
            "pierces_bkb": True,
            "cooldown": [3, 3, 3, 3],
            "mana_cost": [35, 35, 35, 35],
            "damage": [25, 50, 75, 100],
            "radius": [700, 700, 700, 700],
            "description": "向四周射出棘刺，造成物理伤害。对同一目标的后续命中会叠加额外伤害。",
            "coaching_tip": "持续战斗中叠加层数后伤害递增，长时间打架越打越疼；CD极短要不停释放。",
        },
        "Bristleback": {
            "key": "bristleback_bristleback",
            "type": "passive",
            "damage_type": "physical",
            "pierces_bkb": True,
            "cooldown": [],
            "mana_cost": [],
            "description": "背部和侧面受到的伤害降低，受到一定伤害后自动释放Quill Spray。",
            "coaching_tip": "团战中背对敌人承受伤害可以自动反击；注意被动有触发阈值，持续小伤害很有效。",
        },
        "Warpath": {
            "key": "bristleback_warpath",
            "type": "passive",
            "damage_type": "none",
            "pierces_bkb": False,
            "is_ultimate": True,
            "cooldown": [],
            "mana_cost": [],
            "damage": [18, 24, 30],
            "duration": [14, 14, 14],
            "description": "每次施放技能获得一层移速和攻击力加成，可多层叠加。",
            "coaching_tip": "持续释放Quill Spray和Nasal Goo快速叠满层数；满层后攻击力和移速非常可观。",
        },
    },
    # =========================================================================
    # 11. Anti-Mage
    # =========================================================================
    "Anti-Mage": {
        "Mana Break": {
            "key": "antimage_mana_break",
            "type": "passive",
            "damage_type": "physical",
            "pierces_bkb": True,
            "cooldown": [],
            "mana_cost": [],
            "damage": [28, 40, 52, 64],
            "description": "每次攻击燃烧目标魔法值，并将燃烧量的一定比例转化为物理伤害。",
            "coaching_tip": "对蓝量依赖的英雄特别有效；配合高攻速装备快速烧干对手蓝量。",
        },
        "Blink": {
            "key": "antimage_blink",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [15, 12, 9, 6],
            "mana_cost": [45, 45, 45, 45],
            "cast_range": [925, 1000, 1075, 1150],
            "description": "短距离瞬移到目标位置。",
            "coaching_tip": "满级6秒CD使AM极难被抓；刷野时用Blink在营地间跳转加速效率。",
        },
        "Counterspell": {
            "key": "antimage_counterspell",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [15, 11, 7, 3],
            "mana_cost": [45, 50, 55, 60],
            "duration": [1.2, 1.2, 1.2, 1.2],
            "description": "被动获得魔法抗性。主动施放创造一面魔法盾，持续1.2秒反弹指向性技能。",
            "coaching_tip": "预判敌方关键技能使用反弹可以扭转战局；被动魔抗让AM对魔法伤害很肉。",
        },
        "Mana Void": {
            "key": "antimage_mana_void",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": True,
            "is_ultimate": True,
            "cooldown": [70, 70, 70],
            "mana_cost": [100, 200, 300],
            "damage": [0.8, 0.95, 1.1],
            "stun_duration": [0.15, 0.15, 0.15],
            "radius": [500, 500, 500],
            "cast_range": [600, 600, 600],
            "description": "对目标造成基于其已损失魔法值的魔法伤害，对范围内敌人也造成相同伤害并短暂眩晕。",
            "coaching_tip": "对空蓝的智力英雄伤害极高，先用Mana Break烧蓝再大；注意范围伤害可能多杀。",
        },
    },
    # =========================================================================
    # 12. Phantom Assassin
    # =========================================================================
    "Phantom Assassin": {
        "Stifling Dagger": {
            "key": "phantom_assassin_stifling_dagger",
            "type": "active",
            "damage_type": "physical",
            "pierces_bkb": True,
            "cooldown": [6, 5, 4, 3],
            "mana_cost": [30, 25, 20, 15],
            "damage": [65, 75, 85, 95],
            "slow_pct": [50, 50, 50, 50],
            "duration": [1.75, 2.5, 3.25, 4.0],
            "cast_range": [525, 750, 975, 1200],
            "description": "投掷一把匕首造成物理伤害并减速目标，可以触发攻击特效和暴击。",
            "coaching_tip": "线上最优补刀工具，满级施法距离极远；后期飞镖触发暴击的伤害非常恐怖。",
        },
        "Phantom Strike": {
            "key": "phantom_assassin_phantom_strike",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [11, 9, 7, 5],
            "mana_cost": [35, 40, 45, 50],
            "duration": [2.5, 2.5, 2.5, 2.5],
            "cast_range": [1000, 1000, 1000, 1000],
            "description": "瞬间闪现到目标身边，对敌方单位获得攻速加成。也可对友方单位使用。",
            "coaching_tip": "追杀时先飞镖减速再跳过去；也可以跳向友方单位逃跑，灵活运用。",
        },
        "Blur": {
            "key": "phantom_assassin_blur",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [60, 45, 30, 15],
            "mana_cost": [50, 50, 50, 50],
            "duration": [25, 25, 25, 25],
            "description": "被动获得闪避。主动施放短暂消失在小地图上，用于避免被发现。",
            "coaching_tip": "主动开启后敌方看不到你在小地图上的位置，适合偷Roshan或切入；闪避克制物攻。",
        },
        "Coup de Grace": {
            "key": "phantom_assassin_coup_de_grace",
            "type": "passive",
            "damage_type": "physical",
            "pierces_bkb": True,
            "is_ultimate": True,
            "cooldown": [],
            "mana_cost": [],
            "proc_chance": 15,
            "description": "攻击有概率造成暴击，暴击倍率极高。",
            "coaching_tip": "后期配合战斧或碎骨锤暴击秒人；注意MKB可以对抗闪避但无法对抗暴击。",
        },
    },
    # =========================================================================
    # 13. Crystal Maiden
    # =========================================================================
    "Crystal Maiden": {
        "Crystal Nova": {
            "key": "crystal_maiden_crystal_nova",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [12, 11, 10, 9],
            "mana_cost": [130, 145, 160, 175],
            "damage": [130, 170, 210, 260],
            "radius": [425, 425, 425, 425],
            "slow_pct": [30, 40, 50, 60],
            "duration": [5, 5, 5, 5],
            "cast_range": [700, 700, 700, 700],
            "description": "在目标区域引爆冰晶，造成魔法伤害并减缓范围内敌人的攻击速度和移动速度。",
            "coaching_tip": "线上清兵和骚扰的主要技能；减速持续时间很长，配合队友追杀效果好。",
        },
        "Frostbite": {
            "key": "crystal_maiden_frostbite",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [9, 8, 7, 6],
            "mana_cost": [140, 145, 150, 155],
            "damage": [100, 150, 200, 250],
            "duration": [1.5, 2.0, 2.5, 3.0],
            "cast_range": [550, 550, 550, 550],
            "description": "冻结目标使其无法移动，持续造成魔法伤害。对小兵持续时间更长。",
            "coaching_tip": "可靠的单体控制；对野怪持续时间为10秒可以用来单独打野补充经济。",
        },
        "Arcane Aura": {
            "key": "crystal_maiden_brilliance_aura",
            "type": "passive",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [],
            "mana_cost": [],
            "description": "全图被动光环，为所有友方英雄提供额外的魔法恢复。CM自身获得更高的加成。",
            "coaching_tip": "全图回蓝光环是CM的核心价值，对所有队友都有帮助；一级就点可以辅助全队。",
        },
        "Freezing Field": {
            "key": "crystal_maiden_freezing_field",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "is_ultimate": True,
            "cooldown": [110, 100, 90],
            "mana_cost": [200, 400, 600],
            "damage": [105, 170, 250],
            "radius": [835, 835, 835],
            "duration": [10, 10, 10],
            "slow_pct": [30, 30, 30],
            "description": "引导技能，在自身周围持续制造冰爆，每次爆炸造成魔法伤害和减速。",
            "coaching_tip": "配合BKB使用避免被打断；站位非常重要，需要在人群中引导才能打出最大伤害。",
        },
    },
    # =========================================================================
    # 14. Pudge
    # =========================================================================
    "Pudge": {
        "Meat Hook": {
            "key": "pudge_meat_hook",
            "type": "active",
            "damage_type": "pure",
            "pierces_bkb": True,
            "cooldown": [27, 22, 17, 12],
            "mana_cost": [110, 120, 130, 140],
            "damage": [150, 220, 290, 360],
            "cast_range": [1000, 1100, 1200, 1300],
            "description": "投掷肉钩，钩住路径上第一个命中的单位拉到身边，造成纯粹伤害。可穿BKB。",
            "coaching_tip": "从雾区或高低地视野差位置出钩命中率最高；钩中后立即开Rot追杀。",
        },
        "Rot": {
            "key": "pudge_rot",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [0, 0, 0, 0],
            "mana_cost": [0, 0, 0, 0],
            "damage": [30, 60, 90, 120],
            "radius": [250, 250, 250, 250],
            "slow_pct": [14, 20, 26, 32],
            "description": "开关技能，对周围敌人和自身持续造成魔法伤害，并减速范围内敌人。",
            "coaching_tip": "钩中目标后立刻开启保持减速；注意自伤会叠加腐肉堆积层数。",
        },
        "Flesh Heap": {
            "key": "pudge_flesh_heap",
            "type": "passive",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [0, 0, 0, 0],
            "mana_cost": [0, 0, 0, 0],
            "magic_resistance": [6, 8, 10, 12],
            "str_per_kill": [1.5, 2.0, 2.5, 3.0],
            "description": "被动获得魔法抗性。周围敌方英雄死亡时永久获得额外力量。",
            "coaching_tip": "越多击杀/助攻Pudge越肉；后期叠加足够力量后非常难以击杀。",
        },
        "Dismember": {
            "key": "pudge_dismember",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": True,
            "is_ultimate": True,
            "cooldown": [30, 25, 20],
            "mana_cost": [100, 130, 170],
            "damage": [60, 90, 120],
            "duration": [3.0, 3.0, 3.0],
            "strength_damage_mult": [0.3, 0.5, 0.7],
            "cast_range": [160, 160, 160],
            "description": "引导技能，持续撕咬目标造成基于力量的魔法伤害并治疗自身。可穿BKB。",
            "coaching_tip": "Hook + Rot + Dismember是标准连招；穿BKB的引导控制，力量越高伤害越高。",
        },
    },
    # =========================================================================
    # 15. Juggernaut
    # =========================================================================
    "Juggernaut": {
        "Blade Fury": {
            "key": "juggernaut_blade_fury",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [42, 34, 26, 18],
            "mana_cost": [110, 110, 110, 110],
            "damage": [80, 105, 130, 155],
            "radius": [260, 260, 260, 260],
            "duration": [5, 5, 5, 5],
            "description": "旋转刀刃对周围敌人造成魔法伤害，期间自身获得魔法免疫。对建筑无效。",
            "coaching_tip": "提供魔法免疫可以配合辅助控制对线杀人；注意期间无法对英雄普攻。",
        },
        "Healing Ward": {
            "key": "juggernaut_healing_ward",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [60, 60, 60, 60],
            "mana_cost": [120, 125, 130, 135],
            "heal_pct": [2, 3, 4, 5],
            "duration": [25, 25, 25, 25],
            "radius": [500, 500, 500, 500],
            "description": "召唤治疗守卫，按友方最大生命值百分比持续治疗周围友方单位。",
            "coaching_tip": "推进和打架后恢复极有价值；守卫只有1点血，注意保护不要被随手打掉。",
        },
        "Blade Dance": {
            "key": "juggernaut_blade_dance",
            "type": "passive",
            "damage_type": "physical",
            "pierces_bkb": True,
            "cooldown": [0, 0, 0, 0],
            "mana_cost": [0, 0, 0, 0],
            "proc_chance": [20, 25, 30, 35],
            "crit_multiplier": [180, 180, 180, 180],
            "description": "攻击有概率造成暴击伤害。",
            "coaching_tip": "配合攻速装备触发频率更高；无敌斩期间也能触发暴击提升输出。",
        },
        "Omnislash": {
            "key": "juggernaut_omni_slash",
            "type": "active",
            "damage_type": "physical",
            "pierces_bkb": True,
            "is_ultimate": True,
            "cooldown": [140, 120, 100],
            "mana_cost": [200, 275, 350],
            "damage": [200, 225, 250],
            "cast_range": [350, 350, 350],
            "duration": [3.0, 3.5, 4.0],
            "description": "在敌方单位间快速跳跃斩击，期间自身无敌。每次跳跃造成物理伤害并可触发攻击特效。",
            "coaching_tip": "确保目标附近没有多余小兵分散斩击次数；单目标时伤害最高。",
        },
    },
    # =========================================================================
    # 16. Sniper
    # =========================================================================
    "Sniper": {
        "Shrapnel": {
            "key": "sniper_shrapnel",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [0, 0, 0, 0],
            "mana_cost": [75, 75, 75, 75],
            "damage": [15, 35, 55, 75],
            "radius": [450, 450, 450, 450],
            "duration": [10, 10, 10, 10],
            "slow_pct": [15, 20, 25, 30],
            "cast_range": [1800, 1800, 1800, 1800],
            "charges": [3, 3, 3, 3],
            "description": "向目标区域发射榴霰弹，持续造成伤害和减速并提供视野。充能制，最多3发。",
            "coaching_tip": "提供视野可以用于侦查高地或Roshan坑；不要一次全放完，留一发应急。",
        },
        "Headshot": {
            "key": "sniper_headshot",
            "type": "passive",
            "damage_type": "physical",
            "pierces_bkb": True,
            "cooldown": [0, 0, 0, 0],
            "mana_cost": [0, 0, 0, 0],
            "damage": [15, 40, 65, 90],
            "proc_chance": [40, 40, 40, 40],
            "slow_pct": [100, 100, 100, 100],
            "duration": [0.5, 0.5, 0.5, 0.5],
            "description": "攻击有概率造成额外物理伤害和短暂100%减速（类似小眩晕效果）。",
            "coaching_tip": "高触发率配合远程攻击让对手很难靠近；线上持续右键骚扰效果极佳。",
        },
        "Take Aim": {
            "key": "sniper_take_aim",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [20, 18, 16, 14],
            "mana_cost": [50, 50, 50, 50],
            "attack_range_bonus": [100, 200, 300, 400],
            "duration": [3, 3, 3, 3],
            "description": "被动增加攻击距离。主动使用临时获得更大射程加成并忽视障碍物。",
            "coaching_tip": "满级被动加400射程已非常远；主动使用可跨地形攻击，配合Headshot骚扰对手。",
        },
        "Assassinate": {
            "key": "sniper_assassinate",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "is_ultimate": True,
            "cooldown": [20, 15, 10],
            "mana_cost": [175, 275, 375],
            "damage": [320, 485, 650],
            "cast_range": [3000, 3000, 3000],
            "stun_duration": [0.5, 1.0, 1.5],
            "description": "瞄准后对远距离目标发射高伤害子弹，命中时造成眩晕。有弹道飞行时间。",
            "coaching_tip": "超远距离收割残血的利器；满级CD仅10秒不要舍不得用；注意弹道可被闪避或格挡。",
        },
    },
    # =========================================================================
    # 17. Zeus
    # =========================================================================
    "Zeus": {
        "Arc Lightning": {
            "key": "zuus_arc_lightning",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [1.6, 1.6, 1.6, 1.6],
            "mana_cost": [80, 85, 90, 95],
            "damage": [85, 100, 115, 145],
            "cast_range": [850, 850, 850, 850],
            "bounce_count": [5, 7, 9, 15],
            "description": "对目标释放连锁闪电，在多个敌人之间弹跳造成魔法伤害。",
            "coaching_tip": "CD极短是主要补刀和消耗工具；注意蓝量消耗，后期弹跳数可观。",
        },
        "Lightning Bolt": {
            "key": "zuus_lightning_bolt",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [6, 6, 6, 6],
            "mana_cost": [120, 130, 140, 150],
            "damage": [100, 175, 275, 350],
            "cast_range": [700, 700, 700, 700],
            "stun_duration": [0.25, 0.25, 0.25, 0.25],
            "description": "对目标释放闪电箭，造成高额魔法伤害和短暂小眩晕，并提供该位置的真实视野。",
            "coaching_tip": "高额单体伤害是Zeus的主要输出技能；可用来排眼和反隐，提供真视非常实用。",
        },
        "Static Field": {
            "key": "zuus_static_field",
            "type": "passive",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [0, 0, 0, 0],
            "mana_cost": [0, 0, 0, 0],
            "hp_damage_pct": [4, 6, 8, 10],
            "radius": [1200, 1200, 1200, 1200],
            "description": "每次释放技能时对周围敌人造成其当前生命值百分比的魔法伤害。",
            "coaching_tip": "对高血量目标效果越好；连续释放技能时被动伤害累积非常可观。",
        },
        "Thundergod's Wrath": {
            "key": "zuus_thundergods_wrath",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "is_ultimate": True,
            "cooldown": [120, 110, 100],
            "mana_cost": [250, 350, 450],
            "damage": [300, 400, 500],
            "description": "全图对所有敌方英雄造成魔法伤害并提供其位置的短暂视野。",
            "coaching_tip": "全图收割残血或开团前提供敌方位置信息；配合刷新球双大伤害极其恐怖。",
        },
    },
    # =========================================================================
    # 18. Witch Doctor
    # =========================================================================
    "Witch Doctor": {
        "Paralyzing Cask": {
            "key": "witch_doctor_paralyzing_cask",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [20, 18, 16, 14],
            "mana_cost": [80, 100, 120, 140],
            "damage": [75, 75, 75, 75],
            "cast_range": [700, 700, 700, 700],
            "stun_duration": [1.0, 1.0, 1.0, 1.0],
            "bounce_count": [2, 4, 6, 8],
            "description": "投掷麻痹药剂在敌方单位间弹跳，每次弹跳造成伤害和短暂眩晕。",
            "coaching_tip": "双人对线时弹跳次数多控制极强；单独面对一个目标效果差，需要有多个弹跳单位。",
        },
        "Voodoo Restoration": {
            "key": "witch_doctor_voodoo_restoration",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [0, 0, 0, 0],
            "mana_cost": [20, 30, 40, 50],
            "heal_per_sec": [16, 24, 32, 40],
            "radius": [500, 500, 500, 500],
            "description": "开关技能，持续消耗魔法治疗周围友方单位。",
            "coaching_tip": "线上维持己方核心血量很实用；注意持续消耗蓝量不要长时间无意义地开着。",
        },
        "Maledict": {
            "key": "witch_doctor_maledict",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [28, 24, 20, 16],
            "mana_cost": [105, 110, 115, 120],
            "damage": [5, 10, 15, 20],
            "duration": [12, 12, 12, 12],
            "cast_range": [575, 575, 575, 575],
            "radius": [180, 180, 180, 180],
            "hp_loss_damage_bonus": [16, 24, 32, 40],
            "description": "对范围内敌人施加诅咒，每4秒根据持续期间损失的生命值造成额外爆发伤害。",
            "coaching_tip": "先把目标打低血再施放诅咒效果最强，持续12秒内损失血量越多伤害越高。",
        },
        "Death Ward": {
            "key": "witch_doctor_death_ward",
            "type": "active",
            "damage_type": "physical",
            "pierces_bkb": True,
            "is_ultimate": True,
            "cooldown": [80, 80, 80],
            "mana_cost": [200, 200, 200],
            "damage": [90, 130, 170],
            "duration": [8, 8, 8],
            "cast_range": [700, 700, 700],
            "description": "引导技能，放置死亡守卫攻击附近敌人，伤害极高可弹射。",
            "coaching_tip": "需要保护引导不被打断；配合BKB或隐蔽位置施放；A杖后弹射伤害翻倍。",
        },
    },
    # =========================================================================
    # 19. Faceless Void
    # =========================================================================
    "Faceless Void": {
        "Time Walk": {
            "key": "faceless_void_time_walk",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [24, 18, 12, 6],
            "mana_cost": [40, 40, 40, 40],
            "cast_range": [650, 650, 650, 650],
            "backtrack_duration": [2.0, 2.0, 2.0, 2.0],
            "description": "向前方短距离冲刺，并回溯过去2秒内受到的所有伤害。",
            "coaching_tip": "核心生存技能，受到爆发伤害后立刻使用可回满血；满级CD仅6秒极难被杀。",
        },
        "Time Dilation": {
            "key": "faceless_void_time_dilation",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [36, 30, 24, 18],
            "mana_cost": [60, 70, 80, 90],
            "damage": [20, 30, 40, 50],
            "radius": [775, 775, 775, 775],
            "duration": [6, 7, 8, 9],
            "slow_pct": [10, 12, 14, 16],
            "description": "冻结范围内敌人所有冷却中的技能CD并造成减速，每个冷却中的技能额外增加减速。",
            "coaching_tip": "对手释放技能后使用效果最好，可以让敌人长时间没有技能可用。",
        },
        "Time Lock": {
            "key": "faceless_void_time_lock",
            "type": "passive",
            "damage_type": "magical",
            "pierces_bkb": True,
            "cooldown": [0, 0, 0, 0],
            "mana_cost": [0, 0, 0, 0],
            "damage": [25, 30, 35, 40],
            "proc_chance": [10, 15, 20, 25],
            "stun_duration": [0.5, 0.65, 0.8, 0.95],
            "description": "攻击有概率造成额外魔法伤害和短暂眩晕，可穿BKB。",
            "coaching_tip": "穿BKB的被动Bash，在Chronosphere内连续触发极其致命。",
        },
        "Chronosphere": {
            "key": "faceless_void_chronosphere",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": True,
            "is_ultimate": True,
            "cooldown": [160, 150, 140],
            "mana_cost": [150, 175, 200],
            "radius": [500, 500, 500],
            "duration": [4.0, 4.5, 5.0],
            "cast_range": [600, 600, 600],
            "description": "创造时间停止球体，范围内只有虚空假面可以移动和攻击，敌我友均被冻结。",
            "coaching_tip": "施放前确认不会困住队友，放空或困队友代价极高；配合远程队友输出最佳。",
        },
    },
    # =========================================================================
    # 20. Earthshaker
    # =========================================================================
    "Earthshaker": {
        "Fissure": {
            "key": "earthshaker_fissure",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [18, 17, 16, 15],
            "mana_cost": [125, 140, 155, 170],
            "damage": [110, 160, 210, 260],
            "stun_duration": [1.0, 1.25, 1.5, 1.75],
            "cast_range": [1400, 1400, 1400, 1400],
            "duration": [6, 6, 6, 6],
            "description": "在地面制造裂缝，眩晕路径上所有敌人并创造持续6秒的不可通行地形。",
            "coaching_tip": "远距离线性控制可以分割战场和阻断退路；注意地形也会阻挡友方移动。",
        },
        "Enchant Totem": {
            "key": "earthshaker_enchant_totem",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [5, 5, 5, 5],
            "mana_cost": [20, 30, 40, 50],
            "attack_damage_mult": [100, 200, 300, 400],
            "description": "增幅下一次攻击的伤害倍率。满级为基础攻击力的400%额外伤害。",
            "coaching_tip": "满级400%额外伤害配合跳刀回音击后补一拳伤害极高；也可用于补刀。",
        },
        "Aftershock": {
            "key": "earthshaker_aftershock",
            "type": "passive",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [0, 0, 0, 0],
            "mana_cost": [0, 0, 0, 0],
            "damage": [60, 100, 140, 180],
            "radius": [300, 300, 300, 300],
            "stun_duration": [0.6, 0.9, 1.2, 1.5],
            "description": "每次释放技能后对周围敌人造成魔法伤害和短暂眩晕。",
            "coaching_tip": "让ES的每个技能都附带范围控制，连续放技能可以打出很长的控制链。",
        },
        "Echo Slam": {
            "key": "earthshaker_echo_slam",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "is_ultimate": True,
            "cooldown": [150, 130, 110],
            "mana_cost": [145, 205, 265],
            "damage": [100, 150, 200],
            "echo_damage": [60, 85, 110],
            "radius": [600, 600, 600],
            "description": "发出冲击波造成初始伤害，每个受影响的单位产生回音波再次对周围造成伤害。",
            "coaching_tip": "敌人越密集（包括小兵）伤害越高；跳刀跳入人群中释放效果最佳。",
        },
    },
    # =========================================================================
    # 21. Tidehunter
    # =========================================================================
    "Tidehunter": {
        "Gush": {
            "key": "tidehunter_gush",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [12, 12, 12, 12],
            "mana_cost": [90, 100, 110, 120],
            "damage": [110, 160, 210, 260],
            "cast_range": [800, 800, 800, 800],
            "slow_pct": [40, 40, 40, 40],
            "armor_reduction": [3, 4, 5, 6],
            "duration": [4, 4, 4, 4],
            "description": "对目标喷射水流，造成魔法伤害并降低其护甲和移动速度。",
            "coaching_tip": "减甲配合物理输出队友效果很好；A杖升级为地面范围释放威力大增。",
        },
        "Kraken Shell": {
            "key": "tidehunter_kraken_shell",
            "type": "passive",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [0, 0, 0, 0],
            "mana_cost": [0, 0, 0, 0],
            "damage_block": [14, 24, 34, 44],
            "damage_threshold": [500, 475, 450, 425],
            "description": "被动格挡每次受到的伤害。在短时间内累计受到伤害达到阈值后自动驱散所有负面效果。",
            "coaching_tip": "让Tide极难被控死，到达伤害阈值后自动解控；线上非常难被杀。",
        },
        "Anchor Smash": {
            "key": "tidehunter_anchor_smash",
            "type": "active",
            "damage_type": "physical",
            "pierces_bkb": False,
            "cooldown": [7, 6, 5, 4],
            "mana_cost": [30, 40, 50, 60],
            "damage": [45, 90, 135, 180],
            "radius": [375, 375, 375, 375],
            "attack_damage_reduction": [30, 40, 50, 60],
            "duration": [6, 6, 6, 6],
            "description": "范围物理伤害并大幅降低命中敌人的攻击力。",
            "coaching_tip": "CD短伤害不错用于清兵；满级降60%攻击力让敌方核心几乎无法输出。",
        },
        "Ravage": {
            "key": "tidehunter_ravage",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "is_ultimate": True,
            "cooldown": [150, 150, 150],
            "mana_cost": [150, 225, 325],
            "damage": [200, 290, 380],
            "radius": [1250, 1250, 1250],
            "stun_duration": [2.0, 2.4, 2.8],
            "description": "释放巨型触手从脚下向外扩散，眩晕并伤害范围内所有敌人。",
            "coaching_tip": "Dota中范围最大的团控之一，跳刀先手效果最佳；CD极长(150秒)切勿浪费。",
        },
    },
    # =========================================================================
    # 22. Enigma
    # =========================================================================
    "Enigma": {
        "Malefice": {
            "key": "enigma_malefice",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [15, 13, 11, 9],
            "mana_cost": [100, 120, 140, 160],
            "damage": [40, 60, 80, 120],
            "duration": [6, 6, 6, 6],
            "stun_duration": [0.4, 0.6, 0.8, 1.0],
            "stun_interval": [2, 2, 2, 2],
            "cast_range": [600, 600, 600, 600],
            "description": "对目标施加诅咒，每2秒造成伤害和短暂眩晕，持续6秒共触发3次。",
            "coaching_tip": "反复眩晕可以打断TP和引导类技能；配合追杀时控制效果很好。",
        },
        "Demonic Summoning": {
            "key": "enigma_demonic_conversion",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [35, 35, 35, 35],
            "mana_cost": [140, 150, 160, 170],
            "eidolon_count": [3, 3, 3, 3],
            "eidolon_hp": [180, 200, 220, 240],
            "eidolon_damage": [21, 30, 39, 48],
            "duration": [35, 35, 35, 35],
            "cast_range": [700, 700, 700, 700],
            "description": "将一个小兵分裂为3个小精灵，精灵攻击一定次数后会再分裂。",
            "coaching_tip": "Enigma核心刷野和推线技能，用精灵高效打野是建立经济优势的关键。",
        },
        "Midnight Pulse": {
            "key": "enigma_midnight_pulse",
            "type": "active",
            "damage_type": "pure",
            "pierces_bkb": True,
            "cooldown": [35, 35, 35, 35],
            "mana_cost": [50, 80, 110, 140],
            "hp_damage_pct": [3.0, 3.75, 4.5, 5.25],
            "radius": [600, 600, 600, 600],
            "duration": [11, 11, 11, 11],
            "cast_range": [700, 700, 700, 700],
            "description": "在目标区域创造黑暗脉冲，对范围内敌人持续造成最大生命值百分比的纯粹伤害。",
            "coaching_tip": "先放Midnight Pulse再接Black Hole让伤害叠加；穿BKB的纯粹伤害对肉盾也很有效。",
        },
        "Black Hole": {
            "key": "enigma_black_hole",
            "type": "active",
            "damage_type": "pure",
            "pierces_bkb": True,
            "is_ultimate": True,
            "cooldown": [200, 180, 160],
            "mana_cost": [275, 350, 425],
            "damage": [50, 100, 150],
            "radius": [420, 420, 420],
            "duration": [4.0, 4.0, 4.0],
            "cast_range": [275, 275, 275],
            "description": "引导技能，将范围内所有敌人拉向中心并持续造成纯粹伤害。穿BKB。",
            "coaching_tip": "Dota最强团控之一，BKB保护引导是标准用法；先Midnight Pulse再Black Hole最大化伤害。",
        },
    },
    # =========================================================================
    # 23. Phoenix
    # =========================================================================
    "Phoenix": {
        "Icarus Dive": {
            "key": "phoenix_icarus_dive",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [36, 32, 28, 24],
            "mana_cost": [0, 0, 0, 0],
            "damage": [15, 35, 55, 75],
            "duration": [4, 4, 4, 4],
            "slow_pct": [19, 22, 25, 28],
            "hp_cost_pct": [6, 6, 6, 6],
            "description": "沿弧线飞行，经过的敌人受到持续魔法伤害和减速。消耗自身当前生命值。",
            "coaching_tip": "进攻和逃生兼备，飞行途中可以使用Fire Spirits和Sun Ray；注意血量消耗。",
        },
        "Fire Spirits": {
            "key": "phoenix_fire_spirits",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [45, 40, 35, 30],
            "mana_cost": [80, 90, 100, 110],
            "damage": [20, 40, 60, 80],
            "duration": [4, 4, 4, 4],
            "attack_speed_slow": [40, 60, 80, 100],
            "spirit_count": [4, 4, 4, 4],
            "hp_cost_pct": [3, 3, 3, 3],
            "cast_range": [1400, 1400, 1400, 1400],
            "description": "召唤4个火焰精灵环绕自身，可逐个投掷。命中敌人造成持续伤害和攻速降低。",
            "coaching_tip": "满级降100攻速让敌方核心几乎无法输出；配合Supernova降攻速让敌人难以打掉太阳。",
        },
        "Sun Ray": {
            "key": "phoenix_sun_ray",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [26, 22, 18, 14],
            "mana_cost": [100, 100, 100, 100],
            "damage": [14, 22, 30, 38],
            "hp_damage_pct": [1.25, 2.5, 3.75, 5.0],
            "heal_pct": [0.625, 1.25, 1.875, 2.5],
            "duration": [6, 6, 6, 6],
            "hp_cost_pct": [6, 6, 6, 6],
            "description": "释放光线，对直线上敌人造成固定伤害加最大生命值百分比伤害，同时治疗友方。",
            "coaching_tip": "对高血量肉盾效果极好；引导期间可缓慢调整方向，兼顾输出和治疗。",
        },
        "Supernova": {
            "key": "phoenix_supernova",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "is_ultimate": True,
            "cooldown": [110, 110, 110],
            "mana_cost": [200, 200, 200],
            "damage": [60, 90, 120],
            "radius": [1000, 1000, 1000],
            "duration": [6, 6, 6],
            "attack_count_to_destroy": [5, 8, 11],
            "stun_duration": [1.5, 2.0, 2.5],
            "description": "变为太阳，对范围内敌人持续造成伤害和减速。持续时间结束后重生并眩晕范围内敌人。被攻击一定次数后太阳被摧毁。",
            "coaching_tip": "需要选择敌方无法集火的时机释放；配合Fire Spirits降攻速让敌人更难打掉太阳。",
        },
    },
    # =========================================================================
    # 24. Rubick
    # =========================================================================
    "Rubick": {
        "Fade Bolt": {
            "key": "rubick_fade_bolt",
            "type": "active",
            "damage_type": "magical",
            "pierces_bkb": False,
            "cooldown": [16, 14, 12, 10],
            "mana_cost": [120, 130, 140, 150],
            "damage": [100, 175, 250, 325],
            "cast_range": [800, 800, 800, 800],
            "attack_damage_reduction": [10, 18, 26, 34],
            "duration": [10, 10, 10, 10],
            "description": "释放弹射闪电，在敌方单位间弹跳造成伤害并降低每个目标的攻击力。",
            "coaching_tip": "线上主要消耗和辅助补刀技能；降攻效果在对线和团战中让敌方核心输出下降。",
        },
        "Arcane Supremacy": {
            "key": "rubick_arcane_supremacy",
            "type": "passive",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [0, 0, 0, 0],
            "mana_cost": [0, 0, 0, 0],
            "cast_range_bonus": [60, 120, 180, 240],
            "spell_amp_pct": [14, 18, 22, 26],
            "description": "被动增加施法距离和技能伤害增强百分比。",
            "coaching_tip": "让偷来的技能效果更强更远；施法距离优势使得安全距离释放技能成为可能。",
        },
        "Null Field": {
            "key": "rubick_null_field",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [30, 26, 22, 18],
            "mana_cost": [50, 50, 50, 50],
            "radius": [1200, 1200, 1200, 1200],
            "magic_resistance": [10, 16, 22, 28],
            "duration": [6, 6, 6, 6],
            "description": "主动释放，为范围内友方英雄提供额外魔法抗性。",
            "coaching_tip": "对抗魔法爆发阵容时团战前开启保护队友，持续6秒效果可观。",
        },
        "Spell Steal": {
            "key": "rubick_spell_steal",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "is_ultimate": True,
            "cooldown": [20, 14, 8],
            "mana_cost": [25, 25, 25],
            "cast_range": [1000, 1000, 1000],
            "description": "窃取目标最近使用的技能为己用，偷到的技能受Arcane Supremacy加成。",
            "coaching_tip": "观察敌方技能释放顺序是核心；偷到关键大招如Ravage/Black Hole可以扭转战局。A杖可保留两个偷取的技能。",
        },
    },
    # =========================================================================
    # 25. Invoker
    # =========================================================================
    "Invoker": {
        "Quas": {
            "key": "invoker_quas",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [0, 0, 0, 0, 0, 0, 0],
            "mana_cost": [0, 0, 0, 0, 0, 0, 0],
            "hp_regen": [1, 3, 5, 7, 9, 11, 13],
            "description": "切换冰元素球，每个球提供生命恢复加成。最多同时存在3个元素球。",
            "coaching_tip": "对线期挂三冰球续航极强；高等级Quas使Cold Snap和Ice Wall等技能更强力。",
        },
        "Wex": {
            "key": "invoker_wex",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [0, 0, 0, 0, 0, 0, 0],
            "mana_cost": [0, 0, 0, 0, 0, 0, 0],
            "attack_speed": [6, 12, 18, 24, 30, 36, 42],
            "move_speed_pct": [1, 2, 3, 4, 5, 6, 7],
            "description": "切换雷元素球，每个球提供攻击速度和移动速度加成。",
            "coaching_tip": "雷球提供高机动性，QW流Invoker靠Tornado+EMP组合打消耗和控制。",
        },
        "Exort": {
            "key": "invoker_exort",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "cooldown": [0, 0, 0, 0, 0, 0, 0],
            "mana_cost": [0, 0, 0, 0, 0, 0, 0],
            "bonus_damage": [4, 8, 12, 16, 20, 24, 28],
            "description": "切换火元素球，每个球提供额外攻击力。",
            "coaching_tip": "火球补刀优势明显，QE流靠Sun Strike+Cold Snap+Forge Spirit打线上击杀。",
        },
        "Invoke": {
            "key": "invoker_invoke",
            "type": "active",
            "damage_type": "none",
            "pierces_bkb": False,
            "is_ultimate": True,
            "cooldown": [6, 6, 6],
            "mana_cost": [60, 60, 60],
            "description": "根据当前三个元素球的组合召唤对应技能，最多同时保留两个召唤的技能。",
            "coaching_tip": "战前预读好两个技能准备好连招；团战中快速切换技能是Invoker操作的核心。",
        },
    },
}

# ===================================================================
# Invoker derived spells (from Invoke combinations)
# ===================================================================

INVOKER_SPELLS: dict[str, dict] = {
    "Cold Snap": {"damage_type": "magical", "description": "急速冷却：连续小眩晕", "coaching_tip": "对线消耗和抓人首选"},
    "Ghost Walk": {"damage_type": "none", "description": "幽灵漫步：隐身并减速周围敌人", "coaching_tip": "逃生技能，但有显示范围"},
    "Ice Wall": {"damage_type": "magical", "description": "寒冰之墙：大范围减速", "coaching_tip": "团战分割战场利器"},
    "EMP": {"damage_type": "pure", "description": "电磁脉冲：燃烧敌方魔法", "coaching_tip": "先手Tornado再接EMP"},
    "Tornado": {"damage_type": "magical", "description": "龙卷风：吹起并伤害敌人", "coaching_tip": "先手控制，注意不要吹散队友技能"},
    "Alacrity": {"damage_type": "none", "description": "灵动迅捷：增加攻速和攻击力", "coaching_tip": "给核心加BUFF或自己右击"},
    "Sun Strike": {"damage_type": "pure", "description": "阳炎冲击：全图纯粹伤害", "coaching_tip": "配合队友控制使用，注意延迟"},
    "Forge Spirit": {"damage_type": "none", "description": "熔炉精灵：召唤战斗单位", "coaching_tip": "推线和打野辅助"},
    "Chaos Meteor": {"damage_type": "magical", "description": "混沌陨石：滚动火球", "coaching_tip": "配合控制使用，单独命中率低"},
    "Deafening Blast": {"damage_type": "magical", "description": "超震声波：击退并缴械", "coaching_tip": "后期团战核心技能，缴械效果强力"},
}

_FALLBACK_ITEM_DATA: dict[str, dict] = {
    # ===== Boots =====
    "Phase Boots": {
        "key": "phase_boots", "cost": 1500, "is_active": True, "cooldown": 8,
        "duration": 3, "category": "boots",
        "description": "相位鞋：+18攻击力，+4护甲，主动获得22%移速",
        "coaching_tip": "追杀逃跑均可用，CD短多用",
    },
    "Power Treads": {
        "key": "power_treads", "cost": 1400, "is_active": True, "cooldown": 0,
        "category": "boots",
        "description": "动力鞋：+25属性切换，+45攻速",
        "coaching_tip": "施法前切智力省魔，受伤切力量保命",
    },
    "Arcane Boots": {
        "key": "arcane_boots", "cost": 1300, "is_active": True, "cooldown": 55,
        "category": "boots",
        "description": "秘法鞋：主动回复周围友军175魔法",
        "coaching_tip": "团战前先开一次补魔",
    },
    "Tranquil Boots": {
        "key": "tranquil_boots", "cost": 925, "is_active": False, "category": "boots",
        "description": "静谧之鞋：+65移速，脱战后高额回血",
        "coaching_tip": "攻击会破坏被动回血13秒，游走辅助首选",
    },
    "Boots of Travel": {
        "key": "travel_boots", "cost": 2500, "is_active": True, "cooldown": 80,
        "category": "boots",
        "description": "飞鞋：传送到友方建筑/小兵",
        "coaching_tip": "全图支援和带线分推核心装备",
    },
    "Guardian Greaves": {
        "key": "guardian_greaves", "cost": 5375, "is_active": True, "cooldown": 40,
        "category": "boots",
        "description": "卫士胫甲：主动回复队友生命和魔法，并驱散自身负面效果",
        "coaching_tip": "团战核心道具，20%血以下有额外加成",
    },
    # ===== Defense =====
    "Black King Bar": {
        "key": "black_king_bar", "cost": 4050, "is_active": True, "cooldown": 75,
        "duration": "10/9/8/7/6/6", "category": "defense",
        "description": "BKB：魔法免疫，+10力量，+24攻击力",
        "coaching_tip": "持续时间逐次缩短，必须在最关键的团战时刻使用",
    },
    "Glimmer Cape": {
        "key": "glimmer_cape", "cost": 1950, "is_active": True, "cooldown": 14,
        "duration": 5, "category": "defense",
        "description": "微光披风：对目标施加隐身和+50%魔抗",
        "coaching_tip": "可以对队友使用，救人神器",
    },
    "Ghost Scepter": {
        "key": "ghost", "cost": 1500, "is_active": True, "cooldown": 22,
        "duration": 4, "category": "defense",
        "description": "幽魂权杖：虚无态免疫物理但受到40%额外魔法伤害",
        "coaching_tip": "对抗物理爆发，但小心魔法伤害会更疼",
    },
    "Force Staff": {
        "key": "force_staff", "cost": 2200, "is_active": True, "cooldown": 19,
        "category": "defense",
        "description": "原力法杖：推动目标600距离",
        "coaching_tip": "救人逃跑翻越地形，可以对敌人使用把人推到不利位置",
    },
    "Eul's Scepter": {
        "key": "cyclone", "cost": 2725, "is_active": True, "cooldown": 23,
        "duration": 2.5, "category": "defense",
        "description": "风杖：吹风自己或敌人2.5秒，驱散自身负面效果",
        "coaching_tip": "自吹可驱散沉默等负面效果，吹敌人可接后续技能",
    },
    "Linken's Sphere": {
        "key": "sphere", "cost": 4600, "is_active": True, "cooldown": 12,
        "category": "defense",
        "description": "林肯法球：每12秒被动格挡一个指向性技能",
        "coaching_tip": "对抗单一先手控制如兽王吼/军团决斗",
    },
    "Lotus Orb": {
        "key": "lotus_orb", "cost": 3850, "is_active": True, "cooldown": 15,
        "duration": 6, "category": "defense",
        "description": "清莲宝珠：驱散并反弹指向性技能",
        "coaching_tip": "团战前给核心上盾，驱散+反弹双重效果",
    },
    "Aeon Disk": {
        "key": "aeon_disk", "cost": 3000, "is_active": False, "cooldown": 105,
        "category": "defense",
        "description": "永恒之盘：血量低于70%时自动触发强驱散+2.5秒无敌",
        "coaching_tip": "被秒杀英雄的保命装，CD很长注意把握",
    },
    "Pipe of Insight": {
        "key": "pipe", "cost": 3475, "is_active": True, "cooldown": 60,
        "duration": 12, "category": "defense",
        "description": "洞察烟斗：给周围队友400点魔法护盾",
        "coaching_tip": "对抗大量魔法AOE时团战前开启",
    },
    "Mekansm": {
        "key": "mekansm", "cost": 1875, "is_active": True, "cooldown": 50,
        "category": "defense",
        "description": "梅肯：回复周围队友275生命值",
        "coaching_tip": "团战中及时使用，不要等血量太低",
    },
    "Crimson Guard": {
        "key": "crimson_guard", "cost": 3600, "is_active": True, "cooldown": 35,
        "duration": 12, "category": "defense",
        "description": "赤红甲：给周围队友格挡物理伤害",
        "coaching_tip": "对抗多物理输出阵容",
    },
    "Heaven's Halberd": {
        "key": "heavens_halberd", "cost": 3550, "is_active": True, "cooldown": 18,
        "duration": "3(近战)/5(远程)", "category": "defense",
        "description": "天堂之戟：缴械目标使其无法攻击",
        "coaching_tip": "对抗物理核心，无视BKB的近战英雄缴械3秒",
    },
    "Spirit Vessel": {
        "key": "spirit_vessel", "cost": 2980, "is_active": True, "cooldown": 7,
        "category": "utility",
        "description": "魂之灵瓮：对敌减少回复并造成持续伤害，对友持续治疗",
        "coaching_tip": "对抗高回复英雄如Alch/Huskar/WK必出",
    },
    # ===== Offense =====
    "Desolator": {
        "key": "desolator", "cost": 3500, "is_active": False, "category": "offense",
        "description": "黯灭：攻击降低目标6点护甲，+50攻击力",
        "coaching_tip": "物理输出核心装备，推塔也有加成",
    },
    "Daedalus": {
        "key": "daedalus", "cost": 5150, "is_active": False, "category": "offense",
        "description": "代达罗斯之殇：30%暴击率，+225%暴击伤害",
        "coaching_tip": "纯输出装，需要先有攻速",
    },
    "Monkey King Bar": {
        "key": "monkey_king_bar", "cost": 4975, "is_active": False, "category": "offense",
        "description": "金箍棒：75%概率造成额外70魔法伤害（克制闪避）",
        "coaching_tip": "对抗Butterfly/PA闪避必出",
    },
    "Butterfly": {
        "key": "butterfly", "cost": 4975, "is_active": True, "cooldown": 25,
        "category": "offense",
        "description": "蝴蝶：+35闪避，+35敏捷，+30攻速，主动获得移速但失去闪避",
        "coaching_tip": "后期核心装，但被MKB克制",
    },
    "Assault Cuirass": {
        "key": "assault", "cost": 5125, "is_active": False, "category": "offense",
        "description": "强袭胸甲：+30攻速+10护甲光环，降低敌方5护甲",
        "coaching_tip": "团队增益装备，物理阵容核心",
    },
    "Eye of Skadi": {
        "key": "skadi", "cost": 5300, "is_active": False, "category": "offense",
        "description": "冰眼：+22全属性，攻击减速减回复",
        "coaching_tip": "后期万金油装，减速减回复效果强",
    },
    "Heart of Tarrasque": {
        "key": "heart", "cost": 5200, "is_active": False, "category": "offense",
        "description": "龙心：+250生命，脱战后每秒回复1.6%最大生命",
        "coaching_tip": "坦克装，脱战回血可以反复参团",
    },
    "Satanic": {
        "key": "satanic", "cost": 5050, "is_active": True, "cooldown": 30,
        "duration": 6, "category": "offense",
        "description": "撒旦：主动6秒内200%吸血，+25力量",
        "coaching_tip": "残血时开启一波反杀，配合BKB使用最佳",
    },
    "Nullifier": {
        "key": "nullifier", "cost": 4725, "is_active": True, "cooldown": 11,
        "duration": 6, "category": "offense",
        "description": "否决：持续驱散目标增益效果并减速",
        "coaching_tip": "克制Ghost Scepter/Glimmer等保命道具",
    },
    "Bloodthorn": {
        "key": "bloodthorn", "cost": 6800, "is_active": True, "cooldown": 15,
        "duration": 5, "category": "offense",
        "description": "血棘：沉默目标5秒，期间所有攻击暴击",
        "coaching_tip": "对抗高机动英雄如Storm/AM/Puck",
    },
    "Divine Rapier": {
        "key": "rapier", "cost": 5950, "is_active": False, "category": "offense",
        "description": "圣剑：+350攻击力，死亡后掉落",
        "coaching_tip": "绝境翻盘装，死亡会掉落给敌人",
    },
    "Mjollnir": {
        "key": "mjollnir", "cost": 5600, "is_active": True, "cooldown": 35,
        "category": "offense",
        "description": "雷神之锤：攻击有概率闪电链，主动给目标闪电护盾",
        "coaching_tip": "对抗幻象和召唤物阵容，攻速高时收益大",
    },
    "Radiance": {
        "key": "radiance", "cost": 5150, "is_active": True, "cooldown": 45,
        "category": "offense",
        "description": "辉耀：+60攻击力，被动灼烧光环+17%致盲",
        "coaching_tip": "幻象英雄核心装，越早出收益越大",
    },
    "Diffusal Blade": {
        "key": "diffusal_blade", "cost": 2500, "is_active": True, "cooldown": 15,
        "duration": 4, "category": "offense",
        "description": "净魂之刃：攻击燃烧40魔法，主动减速100%",
        "coaching_tip": "对抗低魔英雄效果极佳，主动减速可追杀",
    },
    "Silver Edge": {
        "key": "silver_edge", "cost": 5450, "is_active": True, "cooldown": 22,
        "duration": 14, "category": "offense",
        "description": "白银之锋：隐身攻击破隐+致残（禁用被动技能4秒）",
        "coaching_tip": "克制Bristleback/PA/Spectre等被动流英雄",
    },
    # ===== Initiation =====
    "Blink Dagger": {
        "key": "blink", "cost": 2250, "is_active": True, "cooldown": 15,
        "disabled_on_damage": 3, "category": "utility",
        "description": "跳刀：瞬移1200距离，受伤后3秒不可用",
        "coaching_tip": "先手开团核心装备，受伤后不可用所以不能当逃生用",
    },
    "Shadow Blade": {
        "key": "shadow_blade", "cost": 3000, "is_active": True, "cooldown": 25,
        "duration": 14, "category": "utility",
        "description": "影刃：隐身14秒，破隐攻击+175伤害",
        "coaching_tip": "先手和逃生兼备，但真眼和撒粉可以反制",
    },
    # ===== Utility =====
    "Aghanim's Scepter": {
        "key": "ultimate_scepter", "cost": 4200, "is_active": False, "category": "utility",
        "description": "A杖：升级英雄技能（效果因英雄而异）",
        "coaching_tip": "部分英雄的核心升级装备",
    },
    "Aghanim's Shard": {
        "key": "aghanims_shard", "cost": 1400, "is_active": False, "category": "utility",
        "description": "A晶：解锁或升级一个技能（效果因英雄而异）",
        "coaching_tip": "性价比极高的中期升级，15分钟后可购买",
    },
    "Refresher Orb": {
        "key": "refresher", "cost": 5000, "is_active": True, "cooldown": 160,
        "category": "utility",
        "description": "刷新球：刷新所有技能和物品CD",
        "coaching_tip": "大招CD长的英雄后期核心（Enigma/Tide/Warlock）",
    },
    "Scythe of Vyse": {
        "key": "sheepstick", "cost": 5675, "is_active": True, "cooldown": 22,
        "duration": 3.5, "category": "utility",
        "description": "羊刀：将目标变羊3.5秒",
        "coaching_tip": "最强单体控制道具，对抗BKB前先手",
    },
    "Rod of Atos": {
        "key": "rod_of_atos", "cost": 2750, "is_active": True, "cooldown": 18,
        "duration": 2, "category": "utility",
        "description": "阿托斯之棍：禁锢目标2秒（不可移动但可攻击施法）",
        "coaching_tip": "远距离先手控制，配合技能击杀",
    },
    "Orchid Malevolence": {
        "key": "orchid", "cost": 3475, "is_active": True, "cooldown": 18,
        "duration": 5, "category": "utility",
        "description": "紫苑：沉默目标5秒，结束时造成30%伤害加深",
        "coaching_tip": "对抗依赖技能的英雄如Storm/Puck/AM",
    },
    "Shiva's Guard": {
        "key": "shivas_guard", "cost": 4850, "is_active": True, "cooldown": 27,
        "category": "defense",
        "description": "希瓦：主动释放冰霜波减速，被动降低周围敌人攻速",
        "coaching_tip": "对抗物理攻速流阵容，团战AOE减速",
    },
    "Manta Style": {
        "key": "manta", "cost": 4600, "is_active": True, "cooldown": 30,
        "category": "utility",
        "description": "幻影斧：主动分身+基础驱散",
        "coaching_tip": "可驱散沉默等效果，敏捷核心常出",
    },
    "Ethereal Blade": {
        "key": "ethereal_blade", "cost": 4650, "is_active": True, "cooldown": 20,
        "category": "utility",
        "description": "虚灵之刃：目标虚无态+魔法伤害加深40%",
        "coaching_tip": "魔法爆发组合利器，也可自保对抗物理",
    },
    "Aether Lens": {
        "key": "aether_lens", "cost": 2275, "is_active": False, "category": "utility",
        "description": "以太之镜：+225施法距离，+300魔法",
        "coaching_tip": "增加先手距离，辅助核心过渡装",
    },
    "Wind Waker": {
        "key": "wind_waker", "cost": 6825, "is_active": True, "cooldown": 23,
        "category": "utility",
        "description": "风行者：对自己或队友施加龙卷风效果",
        "coaching_tip": "升级版风杖，可以对队友使用救人",
    },
    "Gleipnir": {
        "key": "gleipnir", "cost": 6150, "is_active": True, "cooldown": 18,
        "duration": 2, "category": "utility",
        "description": "蛛网：禁锢区域内所有敌人2秒",
        "coaching_tip": "AOE禁锢+闪电链，团战先手利器",
    },
    "Drum of Endurance": {
        "key": "ancient_janggo", "cost": 1700, "is_active": True, "cooldown": 26,
        "category": "utility",
        "description": "战鼓：主动给周围队友攻速移速加成",
        "coaching_tip": "中期推进打团性价比高的团队装",
    },
    "Holy Locket": {
        "key": "holy_locket", "cost": 2400, "is_active": True, "cooldown": 13,
        "category": "support",
        "description": "圣洁吊坠：储存治疗能量释放回血，+35%治疗增强",
        "coaching_tip": "治疗型辅助核心装备",
    },
    "Veil of Discord": {
        "key": "veil_of_discord", "cost": 1525, "is_active": True, "cooldown": 22,
        "duration": 16, "category": "utility",
        "description": "纷争面纱：降低区域内敌人18%魔抗",
        "coaching_tip": "魔法阵容推进团战前先套一个",
    },
}

ABILITY_KEY_LOOKUP: dict[str, tuple[str, str]] = {}


# ---------------------------------------------------------------------------
# Lazy-loaded public accessors
# ---------------------------------------------------------------------------

def _get_ability_data_dict() -> dict[str, dict[str, dict]]:
    """Return ability data, preferring file-backed data over hardcoded fallback."""
    try:
        from .data_manager import load_abilities
        data = load_abilities()
        if data:
            return data
    except ImportError:
        pass
    return _FALLBACK_ABILITY_DATA


def _get_item_data_dict() -> dict[str, dict]:
    """Return item data, preferring file-backed data over hardcoded fallback."""
    try:
        from .data_manager import load_items
        data = load_items()
        if data:
            return data
    except ImportError:
        pass
    return _FALLBACK_ITEM_DATA


# Public names: lazy properties backed by data_manager or fallback dicts.
# For backward compatibility, ABILITY_DATA and ITEM_DATA are module-level names
# that evaluate lazily on first access via the functions above.
# However, since Python module-level dicts cannot be lazy, we use a simple
# approach: the dict references start as the fallback, and callers who need
# file-backed data should use the getter functions.

ABILITY_DATA = _FALLBACK_ABILITY_DATA
ITEM_DATA = _FALLBACK_ITEM_DATA

# Flag to track if we've attempted to load from data_manager
_data_loaded = False


def _ensure_data_loaded() -> None:
    """Load file-backed data into module-level dicts if available."""
    global ABILITY_DATA, ITEM_DATA, _data_loaded
    if _data_loaded:
        return
    _data_loaded = True
    try:
        from .data_manager import load_abilities, load_items
        abilities = load_abilities()
        if abilities:
            ABILITY_DATA = abilities
        items = load_items()
        if items:
            ITEM_DATA = items
        _logger.info("Game data loaded from data_manager (%d heroes, %d items)",
                      len(ABILITY_DATA), len(ITEM_DATA))
    except ImportError:
        _logger.debug("data_manager not available, using hardcoded fallback")
    except Exception as e:
        _logger.warning("Failed to load from data_manager, using fallback: %s", e)


def _build_key_lookup() -> dict[str, tuple[str, str]]:
    """Build reverse lookup from ability_key to (hero_name, ability_name)."""
    _ensure_data_loaded()
    lookup = {}
    for hero, abilities in ABILITY_DATA.items():
        for ability_name, data in abilities.items():
            if "key" in data:
                lookup[data["key"]] = (hero, ability_name)
    return lookup


def get_ability_data(hero_name: str, ability_name: str) -> dict | None:
    _ensure_data_loaded()
    hero_abilities = ABILITY_DATA.get(hero_name, {})
    return hero_abilities.get(ability_name)


def get_ability_by_key(ability_key: str) -> tuple[str, str, dict] | None:
    global ABILITY_KEY_LOOKUP
    _ensure_data_loaded()
    if not ABILITY_KEY_LOOKUP:
        ABILITY_KEY_LOOKUP = _build_key_lookup()
    result = ABILITY_KEY_LOOKUP.get(ability_key)
    if result:
        hero, name = result
        abilities = ABILITY_DATA.get(hero, {})
        if name in abilities:
            return hero, name, abilities[name]
    return None


def get_item_data(item_name: str) -> dict | None:
    _ensure_data_loaded()
    return ITEM_DATA.get(item_name)


_FORMAT_LABELS = {
    "zh": {"cd": "CD {}秒", "mana": "魔耗{}", "stun": "眩晕{}秒", "duration": "持续{}秒",
           "damage_types": {"magical": "魔法", "physical": "物理", "pure": "纯粹"},
           "damage_fmt": "{}伤害{}"},
    "en": {"cd": "CD {}s", "mana": "Mana {}", "stun": "{}s stun", "duration": "Duration {}s",
           "damage_types": {"magical": "Magical", "physical": "Physical", "pure": "Pure"},
           "damage_fmt": "{} {} damage"},
}


def format_ability_at_level(hero_name: str, ability_name: str, level: int, lang: str = "zh") -> str:
    """Format a single ability at a specific level as localized text."""
    data = get_ability_data(hero_name, ability_name)
    if not data or level < 1:
        return f"{ability_name}"
    lbl = _FORMAT_LABELS.get(lang, _FORMAT_LABELS["zh"])
    idx = min(level - 1, 3)
    parts = [f"{ability_name} (Lv{level})"]
    if "cooldown" in data and data["cooldown"]:
        cd_list = data["cooldown"]
        cd_idx = min(idx, len(cd_list) - 1)
        parts.append(lbl["cd"].format(cd_list[cd_idx]))
    if "mana_cost" in data and data["mana_cost"]:
        mc_list = data["mana_cost"]
        mc_idx = min(idx, len(mc_list) - 1)
        parts.append(lbl["mana"].format(mc_list[mc_idx]))
    if "damage" in data and data["damage"]:
        dtype = lbl["damage_types"].get(data.get("damage_type", ""), "")
        dmg_list = data["damage"]
        dmg_idx = min(idx, len(dmg_list) - 1)
        if lang == "en":
            parts.append(lbl["damage_fmt"].format(dtype, dmg_list[dmg_idx]))
        else:
            parts.append(lbl["damage_fmt"].format(dtype, dmg_list[dmg_idx]))
    if "stun_duration" in data and data["stun_duration"]:
        sd_list = data["stun_duration"]
        sd_idx = min(idx, len(sd_list) - 1)
        parts.append(lbl["stun"].format(sd_list[sd_idx]))
    if "duration" in data and isinstance(data["duration"], list) and data["duration"]:
        dur_list = data["duration"]
        dur_idx = min(idx, len(dur_list) - 1)
        parts.append(lbl["duration"].format(dur_list[dur_idx]))
    return ": ".join([parts[0], ", ".join(parts[1:])]) if len(parts) > 1 else parts[0]


def format_hero_abilities(
    hero_name: str, ability_levels: dict[str, int], lang: str = "zh"
) -> str:
    """Format all abilities for a hero at their current levels."""
    _ensure_data_loaded()
    if hero_name not in ABILITY_DATA:
        return ""

    # When no skills have been chosen yet, show available non-ultimate skills
    if not ability_levels and hero_name in ABILITY_DATA:
        if lang == "en":
            lines = [f"### {hero_name} Available Skills (not yet skilled)"]
            lines.append("Note: The hero can choose to skill any ability at any time. Below are level 1 effects for each skill.")
        else:
            lines = [f"### {hero_name} 可选技能（尚未加点）"]
            lines.append("注意：英雄在任何时刻可以选择技能加点，以下是1级各技能效果。")
        for ability_name, data in ABILITY_DATA[hero_name].items():
            is_ult = data.get("is_ultimate", False)
            if is_ult:
                tag = "(requires level 6)" if lang == "en" else "(需要6级)"
                lines.append(f"- {ability_name}: {tag}")
            else:
                tag = "(selectable)" if lang == "en" else "(可选)"
                formatted = format_ability_at_level(hero_name, ability_name, 1, lang=lang)
                lines.append(f"- {tag} {formatted}")
                if data.get("coaching_tip"):
                    lines.append(f"  💡 {data['coaching_tip']}")
        return "\n".join(lines)

    # Normal case: show abilities at their current levels
    if lang == "en":
        lines = [f"### {hero_name} Ability Details"]
        not_learned = "Not learned"
    else:
        lines = [f"### {hero_name} 技能详情"]
        not_learned = "未学习"
    for ability_name in ABILITY_DATA[hero_name]:
        level = ability_levels.get(ability_name, 0)
        if level > 0:
            lines.append(
                f"- {format_ability_at_level(hero_name, ability_name, level, lang=lang)}"
            )
            data = ABILITY_DATA[hero_name][ability_name]
            if data.get("coaching_tip"):
                lines.append(f"  💡 {data['coaching_tip']}")
        else:
            lines.append(f"- {ability_name}: {not_learned}")
    return "\n".join(lines)


def format_player_items(items: list[str], lang: str = "zh") -> str:
    """Format player items with coaching-relevant info."""
    _ensure_data_loaded()
    if not items:
        return ""
    if lang == "en":
        lines = ["### Item Analysis"]
    else:
        lines = ["### 装备分析"]
    active_items = []
    passive_items = []
    for item_name in items:
        data = ITEM_DATA.get(item_name)
        if data and data.get("is_active"):
            active_items.append((item_name, data))
        elif data:
            passive_items.append((item_name, data))
        else:
            passive_items.append((item_name, None))
    # Active items first (more coaching-relevant)
    for name, data in active_items[:4]:
        cd = data.get("cooldown", "?")
        if lang == "en":
            lines.append(f"- {name} (Active, CD {cd}s): {data.get('description', '')}")
        else:
            lines.append(f"- {name}（主动, CD {cd}秒）: {data.get('description', '')}")
        if data.get("coaching_tip"):
            lines.append(f"  💡 {data['coaching_tip']}")
    # Passive items briefly
    if passive_items:
        passive_names = [name for name, _ in passive_items]
        if lang == "en":
            lines.append(f"- Other items: {', '.join(passive_names)}")
        else:
            lines.append(f"- 其他装备: {'、'.join(passive_names)}")
    return "\n".join(lines)


__all__ = [
    "PATCH_VERSION",
    "ABILITY_DATA",
    "INVOKER_SPELLS",
    "ITEM_DATA",
    "_FALLBACK_ABILITY_DATA",
    "_FALLBACK_ITEM_DATA",
    "ABILITY_KEY_LOOKUP",
    "get_ability_data",
    "get_ability_by_key",
    "get_item_data",
    "format_ability_at_level",
    "format_hero_abilities",
    "format_player_items",
    "_ensure_data_loaded",
]
