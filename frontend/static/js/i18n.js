/**
 * Dota 2 Match Analyzer - Internationalization (i18n)
 * Supports: zh (Chinese), en (English)
 */
window.I18N_MESSAGES = {
    zh: {
        // App title
        app_title: 'Dota 2 对局分析器',

        // Header / data source badges
        source_clarity: '本地解析 (Clarity)',
        source_clarity_short: '本地解析',
        source_opendota: 'OpenDota API',
        source_mock: '演示数据',

        // Upload section
        upload_heading: '上传 Replay 文件进行分析',
        upload_drag_hint: '拖拽 .dem 文件到此处，或点击选择文件',
        upload_format_hint: '支持 Dota 2 录像文件 (.dem)',
        upload_match_id_placeholder: '输入比赛ID（如 8713267082）',
        upload_fetch_btn: '获取数据',
        upload_opendota_hint: '通过 OpenDota API 获取比赛数据，无需上传录像文件',
        upload_demo_btn: '加载演示数据',
        upload_parsing: '正在解析录像文件...',

        // Navigation
        back: '返回',

        // Match banner
        radiant_victory: '天辉胜利',
        dire_victory: '夜魇胜利',

        // Teams
        radiant: '天辉',
        dire: '夜魇',

        // Scoreboard headers
        hero: '英雄',
        level: '等级',
        items_col: '装备',
        total: '合计',

        // Draft
        draft_order: 'BP 顺序',

        // Ability builds
        ability_build: '技能加点',

        // Clarity parse note
        clarity_note: '本地解析模式：部分高级数据（技能加点、表现评估、对线热力图等）需要 OpenDota API 才能获取，相关标签页可能显示为空。',

        // Tab labels
        tab_overview: '总览',
        tab_benchmarks: '表现评估',
        tab_laning: '对线',
        tab_combat: '战斗',
        tab_farm: '经济来源',
        tab_items: '出装',
        tab_purchases: '物品时间线',
        tab_economy: '经济曲线',
        tab_skills: '加点',
        tab_objectives: '目标',
        tab_wards: '视野',
        tab_smokes: '开雾',
        tab_teamfights: '团战',
        tab_chat: '聊天',
        tab_ai_coach: 'AI教练',

        // Phase labels
        phase_starting: '出门装',
        phase_early: '前期 (0-10min)',
        phase_mid: '中期 (10-25min)',
        phase_late: '后期 (25min+)',

        // Economy tab
        economy_title: '经济/经验曲线',
        economy_gold_diff: '经济差',
        economy_gold_total: '经济总量',
        economy_xp_diff: '经验差',
        economy_xp_total: '经验总量',
        economy_player_gold: '个人经济',
        economy_player_xp: '个人经验',
        economy_player_lh: '个人补刀',
        economy_diff_hint: '正值 = 天辉领先，负值 = 夜魇领先',
        economy_empty_title: '经济/经验曲线数据暂不可用',
        economy_empty_desc: '本地解析暂不提供分钟级经济经验数据，请通过 OpenDota API 获取完整数据',
        economy_radiant_lead: '天辉领先',
        economy_dire_lead: '夜魇领先',

        // Item builds
        items_title: '出装分析',
        items_empty_title: '出装数据暂不可用',
        items_empty_desc: '本地解析暂不提供详细出装时间线数据',

        // Skill builds
        skills_title: '加点分析',
        skills_empty_title: '技能加点数据暂不可用',
        skills_empty_desc: '本地 Clarity 解析暂不提供技能加点数据，请通过 OpenDota API 获取',
        skills_unavailable: '技能加点数据不可用',

        // Ward section
        wards_title: '眼位分析',
        wards_empty_title: '眼位数据暂不可用',
        wards_empty_desc: '本地解析暂不提供眼位数据，请通过 OpenDota API 获取完整数据',
        wards_observer: '侦查守卫',
        wards_sentry: '岗哨守卫',
        wards_timeline: '时间轴',
        wards_vision_range: '视野范围',
        wards_radiant: '天辉眼位',
        wards_dire: '夜魇眼位',
        wards_obs_short: '侦查',
        wards_sen_short: '岗哨',
        wards_dewarded_obs: '被反(侦查)',
        wards_dewarded_sen: '被反(岗哨)',
        wards_avg_obs_life: '侦查均活',
        wards_avg_sen_life: '岗哨均活',
        wards_active: '活跃眼位',
        wards_timeline_title: '眼位时间线',
        wards_dewarded: '被反',
        wards_alive: '存活',
        wards_placed: '放置',
        wards_vision_series: '视野范围',
        wards_destroyed: '被反眼',

        // Smoke section
        smokes_title: '开雾分析',
        smokes_empty_title: '开雾数据暂不可用',
        smokes_empty_desc: '本地解析暂不提供开雾事件数据，请通过 OpenDota API 获取完整数据',
        smokes_radiant: '天辉用雾',
        smokes_dire: '夜魇用雾',
        smokes_success_rate: '成功率',
        smokes_kills: '击杀',
        smokes_objective: '目标',
        smokes_no_contact: '无效',
        smokes_timeline_mode: '时间轴模式',
        smokes_radiant_legend: '天辉开雾',
        smokes_dire_legend: '夜魇开雾',
        smokes_movement_path: '移动路径',
        smokes_break_point: '烟破位置',
        smokes_detail_records: '详细记录',
        smokes_click_to_select: '点击选择显示',
        smokes_selected: '已选',
        smokes_show_all: '显示全部',
        smokes_kills_suffix: '杀',
        smokes_roshan: '肉山',
        smokes_tower: '推塔',
        smokes_seconds_suffix: '秒',
        smokes_smoke_break: '烟破',
        smokes_persons: '人',
        smokes_duration_label: '持续',
        smokes_result_label: '结果',
        smokes_start_pos: '开雾位置',
        smokes_direction: '方向',

        // Smoke result labels
        smoke_result_kill: '击杀',
        smoke_result_no_contact: '无接触',
        smoke_result_traded: '交换',
        smoke_result_objective: '目标',

        // Teamfight section
        teamfights_title: '团战分析',
        teamfights_empty_title: '团战数据暂不可用',
        teamfights_empty_desc: '本地解析暂不提供团战详情数据，请通过 OpenDota API 获取完整数据',
        teamfights_pause: '暂停',
        teamfights_play: '播放',
        teamfights_speed: '倍速',
        teamfights_abilities: '技能',
        teamfights_time: '时间',
        teamfights_duration: '持续',
        teamfights_result: '结果',
        teamfights_radiant_win: '天辉胜',
        teamfights_dire_win: '夜魇胜',
        teamfights_draw: '平局',
        teamfights_gold_change: '经济变化',
        teamfights_kill_order: '击杀顺序',
        teamfights_hero: '英雄',
        teamfights_damage: '输出',
        teamfights_taken: '承受',
        teamfights_healing: '治疗',
        teamfights_gold: '经济',
        teamfights_died: '阵亡',
        teamfights_ability_cast: '技能释放',
        teamfights_hero_pos: '英雄位置',

        // Laning section
        laning_title: '对线分析',
        laning_empty_title: '对线数据暂不可用',
        laning_empty_desc: '本地解析暂不提供对线期详细数据，请通过 OpenDota API 获取完整数据',
        laning_lane: '路线',
        laning_efficiency: '效率',
        laning_gold_10: '10分经济',
        laning_xp_10: '10分经验',
        laning_dmg_dealt: '伤害输出',
        laning_dmg_taken: '伤害承受',
        laning_regen_items: '回复道具',
        laning_heatmap_hint: '点击上方英雄查看对线期位置热力图',
        laning_cs_title: '对线期补刀曲线',
        laning_hp_title: '对线期血量百分比',
        laning_no_cs_data: '暂无补刀曲线数据',
        laning_no_hp_data: '暂无血量曲线数据',
        laning_minutes: '分钟',
        laning_last_hits: '补刀数',
        laning_lh: '补刀',
        laning_hp_percent: '血量百分比',

        // Lane name labels (for translating backend lane_name strings)
        lane_safe: '安全路',
        lane_mid: '中路',
        lane_off: '劣势路',
        lane_roaming: '游走',
        lane_unknown: '未知',

        // Combat section
        combat_title: '战斗分析',
        combat_empty_title: '战斗数据暂不可用',
        combat_empty_desc: '本地解析暂不提供击杀/伤害矩阵数据，请通过 OpenDota API 获取完整数据',
        combat_kill_matrix: '击杀矩阵',
        combat_damage_matrix: '伤害矩阵',
        combat_kills_arrow: '击杀 →',
        combat_damage_arrow: '伤害 →',

        // Farm section
        farm_title: '经济来源',
        farm_empty_title: '经济来源数据暂不可用',
        farm_empty_desc: '本地解析暂不提供经济来源详情，请通过 OpenDota API 获取完整数据',
        farm_gold_sources: '经济来源',
        farm_xp_sources: '经验来源',
        farm_unit_kills: '击杀单位',
        farm_lh_timeline: '补刀曲线',
        farm_heroes: '英雄',
        farm_creeps: '小兵',
        farm_neutrals: '中立',
        farm_ancients: '远古',
        farm_towers: '塔',
        farm_roshan: '肉山',
        farm_courier: '信使',
        farm_no_data: '暂无数据',

        // Farm reason colors
        farm_reason_creeps: '小兵',
        farm_reason_heroes: '英雄',
        farm_reason_buildings: '建筑',
        farm_reason_roshan: '肉山',
        farm_reason_courier: '信使',
        farm_reason_other: '其他',
        farm_reason_sell: '出售',
        farm_reason_death: '死亡',
        farm_reason_buyback: '买活',
        farm_reason_unknown: '未知',

        // Purchase section
        purchases_title: '物品购买时间线',
        purchases_empty_title: '物品购买数据暂不可用',
        purchases_empty_desc: '本地解析暂不提供购买时间线数据，请通过 OpenDota API 获取完整数据',

        // Benchmark section
        benchmarks_title: '表现评估',
        benchmarks_empty_title: '表现评估数据暂不可用',
        benchmarks_empty_desc: '表现评估需要 OpenDota API 的百分位数据，本地解析无法提供',
        benchmarks_kills_per_min: '击杀/分',
        benchmarks_lh_per_min: '补刀/分',
        benchmarks_dmg_per_min: '伤害/分',
        benchmarks_healing_per_min: '治疗/分',
        benchmarks_tower_dmg: '塔伤',

        // Objectives section
        objectives_title: '目标与事件',
        objectives_empty_title: '目标事件数据暂不可用',
        objectives_empty_desc: '本地解析暂不提供目标事件数据，请通过 OpenDota API 获取完整数据',
        objectives_timeline: '事件时间线',
        objectives_rune_stats: '符文统计',
        rune_dd: '双倍',
        rune_haste: '急速',
        rune_illusion: '幻象',
        rune_invis: '隐身',
        rune_regen: '恢复',
        rune_arcane: '奥术',
        rune_bounty: '赏金',
        rune_water: '圣水',

        // Chat section
        chat_title: '聊天记录',
        chat_empty: '暂无聊天记录',
        chat_chatwheel: '[语音]',

        // AI Coach section
        ai_coach_select_player: '选择要分析的玩家',
        ai_coach_loading: 'AI教练正在分析中，请稍候...',
        ai_coach_loading_detail: '正在分析关键决策点和死亡事件',

        // Progress stage labels (used by SSE stream of /ai-coach-stream)
        coach_stage_starting: '准备分析...',
        coach_stage_detect: '识别关键决策点...',
        coach_stage_causal_chain: '追溯死亡因果链...',
        coach_stage_context: '构建场景上下文...',
        coach_stage_evaluate: 'AI正在评估每个决策...',
        coach_stage_comparison: '对比职业玩家数据...',
        coach_stage_phases: '分阶段复盘 (对线/前中期/中后期/后期)...',
        coach_stage_summary: '生成总结报告...',
        coach_stage_done: '完成',

        // Phase reviews
        ai_coach_phase_reviews: '分阶段复盘',
        ai_coach_phase_highlights: '亮点',
        ai_coach_phase_issues: '问题',
        ai_coach_phase_suggestions: '建议',
        ai_coach_phase_lh: '补刀',
        phase_laning: '对线期',
        phase_early_mid: '前中期',
        phase_mid_late: '中后期',
        phase_late: '后期',
        phase_ultra_late: '大后期',
        ai_coach_overall: '总体评价',
        ai_coach_death_analysis: '死亡复盘',
        ai_coach_deaths_suffix: '次',
        ai_coach_issue: '问题',
        ai_coach_suggestion: '建议',
        ai_coach_lesson: '教训',
        ai_coach_causal_chain: '死亡因果链',
        ai_coach_root_cause: '根本原因',
        ai_coach_root_tag: '[根因]',
        ai_coach_death_tag: '[死亡]',
        ai_coach_key_moments: '关键时刻',
        ai_coach_laning_summary: '对线阶段总结',
        ai_coach_cs_10min: '10分钟补刀',
        ai_coach_denies_10min: '10分钟反补',
        ai_coach_lane_won: '优势',
        ai_coach_lane_lost: '劣势',
        ai_coach_lane_even: '均势',
        ai_coach_lane_result: '对线结果',
        ai_coach_lane_matchup: '对线匹配',
        ai_coach_pro_comparison: '职业基准对比',
        ai_coach_improvement: '需要改进',
        ai_coach_strengths: '表现亮点',
        ai_coach_failed: 'AI教练分析失败',

        // Time-related strings
        time_seconds_ago: '秒前',
        time_minutes_ago: '分钟前',
        time_hours_ago: '小时前',
        time_days_ago: '天前',
        time_months_ago: '个月前',

        // Chart labels
        chart_game_time: '游戏时间 (分钟)',
        chart_gold_diff: '经济差',
        chart_xp_diff: '经验差',

        // Error messages
        error_invalid_format: '请上传 .dem 格式的录像文件',
        error_upload_failed: '上传失败',
        error_demo_failed: '加载演示数据失败',
        error_fetch_failed: '获取比赛数据失败',
        error_basic_data_fetched: '已获取基础数据。详细数据（眼位/团战/经济曲线）需要等待 OpenDota 解析完成后重新获取。已自动请求解析。',

        // Empty state generic
        empty_local_parse_hint: '本地解析暂不提供此数据，请通过 OpenDota API 获取完整数据',
    },

    en: {
        // App title
        app_title: 'Dota 2 Match Analyzer',

        // Header / data source badges
        source_clarity: 'Local Parse (Clarity)',
        source_clarity_short: 'Local Parse',
        source_opendota: 'OpenDota API',
        source_mock: 'Demo Data',

        // Upload section
        upload_heading: 'Upload Replay File for Analysis',
        upload_drag_hint: 'Drag and drop .dem file here, or click to select',
        upload_format_hint: 'Supports Dota 2 replay files (.dem)',
        upload_match_id_placeholder: 'Enter Match ID (e.g. 8713267082)',
        upload_fetch_btn: 'Fetch Data',
        upload_opendota_hint: 'Fetch match data via OpenDota API, no replay file needed',
        upload_demo_btn: 'Load Demo Data',
        upload_parsing: 'Parsing replay file...',

        // Navigation
        back: 'Back',

        // Match banner
        radiant_victory: 'Radiant Victory',
        dire_victory: 'Dire Victory',

        // Teams
        radiant: 'Radiant',
        dire: 'Dire',

        // Scoreboard headers
        hero: 'Hero',
        level: 'Level',
        items_col: 'Items',
        total: 'Total',

        // Draft
        draft_order: 'Draft Order',

        // Ability builds
        ability_build: 'Ability Build',

        // Clarity parse note
        clarity_note: 'Local parse mode: Some advanced data (skill builds, benchmarks, laning heatmaps, etc.) requires OpenDota API. Related tabs may appear empty.',

        // Tab labels
        tab_overview: 'Overview',
        tab_benchmarks: 'Benchmarks',
        tab_laning: 'Laning',
        tab_combat: 'Combat',
        tab_farm: 'Farm',
        tab_items: 'Items',
        tab_purchases: 'Purchases',
        tab_economy: 'Economy',
        tab_skills: 'Skills',
        tab_objectives: 'Objectives',
        tab_wards: 'Vision',
        tab_smokes: 'Smokes',
        tab_teamfights: 'Teamfights',
        tab_chat: 'Chat',
        tab_ai_coach: 'AI Coach',

        // Phase labels
        phase_starting: 'Starting',
        phase_early: 'Early (0-10min)',
        phase_mid: 'Mid (10-25min)',
        phase_late: 'Late (25min+)',

        // Economy tab
        economy_title: 'Gold/XP Graphs',
        economy_gold_diff: 'Gold Diff',
        economy_gold_total: 'Gold Total',
        economy_xp_diff: 'XP Diff',
        economy_xp_total: 'XP Total',
        economy_player_gold: 'Player Gold',
        economy_player_xp: 'Player XP',
        economy_player_lh: 'Player LH',
        economy_diff_hint: 'Positive = Radiant lead, Negative = Dire lead',
        economy_empty_title: 'Gold/XP graph data not available',
        economy_empty_desc: 'Local parsing does not provide this data. Use OpenDota API for full data.',
        economy_radiant_lead: 'Radiant leads',
        economy_dire_lead: 'Dire leads',

        // Item builds
        items_title: 'Item Builds',
        items_empty_title: 'Item build data not available',
        items_empty_desc: 'Local parsing does not provide detailed item timeline data.',

        // Skill builds
        skills_title: 'Skill Builds',
        skills_empty_title: 'Skill build data not available',
        skills_empty_desc: 'Local Clarity parsing does not provide skill build data. Use OpenDota API.',
        skills_unavailable: 'Skill build data not available',

        // Ward section
        wards_title: 'Ward Analysis',
        wards_empty_title: 'Ward data not available',
        wards_empty_desc: 'Local parsing does not provide ward data. Use OpenDota API for full data.',
        wards_observer: 'Observer Wards',
        wards_sentry: 'Sentry Wards',
        wards_timeline: 'Timeline',
        wards_vision_range: 'Vision Range',
        wards_radiant: 'Radiant Wards',
        wards_dire: 'Dire Wards',
        wards_obs_short: 'Observer',
        wards_sen_short: 'Sentry',
        wards_dewarded_obs: 'Dewarded (Obs)',
        wards_dewarded_sen: 'Dewarded (Sen)',
        wards_avg_obs_life: 'Avg Obs Life',
        wards_avg_sen_life: 'Avg Sen Life',
        wards_active: 'Active Wards',
        wards_timeline_title: 'Ward Timeline',
        wards_dewarded: 'Dewarded',
        wards_alive: 'alive',
        wards_placed: 'Placed',
        wards_vision_series: 'Vision Range',
        wards_destroyed: 'Dewarded',

        // Smoke section
        smokes_title: 'Smoke Analysis',
        smokes_empty_title: 'Smoke data not available',
        smokes_empty_desc: 'Local parsing does not provide smoke event data. Use OpenDota API for full data.',
        smokes_radiant: 'Radiant Smokes',
        smokes_dire: 'Dire Smokes',
        smokes_success_rate: 'Success Rate',
        smokes_kills: 'Kills',
        smokes_objective: 'Objective',
        smokes_no_contact: 'No Contact',
        smokes_timeline_mode: 'Timeline Mode',
        smokes_radiant_legend: 'Radiant Smoke',
        smokes_dire_legend: 'Dire Smoke',
        smokes_movement_path: 'Movement Path',
        smokes_break_point: 'Smoke Break',
        smokes_detail_records: 'Detailed Records',
        smokes_click_to_select: 'Click to select',
        smokes_selected: 'Selected',
        smokes_show_all: 'Show All',
        smokes_kills_suffix: 'kill(s)',
        smokes_roshan: 'Roshan',
        smokes_tower: 'Tower',
        smokes_seconds_suffix: 's',
        smokes_smoke_break: 'Break',
        smokes_persons: 'heroes',
        smokes_duration_label: 'Duration',
        smokes_result_label: 'Result',
        smokes_start_pos: 'Smoke Start',
        smokes_direction: 'Direction',

        // Smoke result labels
        smoke_result_kill: 'Kill',
        smoke_result_no_contact: 'No Contact',
        smoke_result_traded: 'Traded',
        smoke_result_objective: 'Objective',

        // Teamfight section
        teamfights_title: 'Teamfight Analysis',
        teamfights_empty_title: 'Teamfight data not available',
        teamfights_empty_desc: 'Local parsing does not provide teamfight detail data. Use OpenDota API for full data.',
        teamfights_pause: 'Pause',
        teamfights_play: 'Play',
        teamfights_speed: 'Speed',
        teamfights_abilities: 'Abilities',
        teamfights_time: 'Time',
        teamfights_duration: 'Duration',
        teamfights_result: 'Result',
        teamfights_radiant_win: 'Radiant Win',
        teamfights_dire_win: 'Dire Win',
        teamfights_draw: 'Draw',
        teamfights_gold_change: 'Gold Change',
        teamfights_kill_order: 'Kill Order',
        teamfights_hero: 'Hero',
        teamfights_damage: 'Damage',
        teamfights_taken: 'Taken',
        teamfights_healing: 'Healing',
        teamfights_gold: 'Gold',
        teamfights_died: 'Died',
        teamfights_ability_cast: 'Ability Cast',
        teamfights_hero_pos: 'Hero Position',

        // Laning section
        laning_title: 'Laning Analysis',
        laning_empty_title: 'Laning data not available',
        laning_empty_desc: 'Local parsing does not provide detailed laning data. Use OpenDota API for full data.',
        laning_lane: 'Lane',
        laning_efficiency: 'Efficiency',
        laning_gold_10: 'Gold@10',
        laning_xp_10: 'XP@10',
        laning_dmg_dealt: 'Dmg Dealt',
        laning_dmg_taken: 'Dmg Taken',
        laning_regen_items: 'Regen Items',
        laning_heatmap_hint: 'Click a hero above to view laning heatmap',
        laning_cs_title: 'Laning CS Timeline',
        laning_hp_title: 'Laning HP%',
        laning_no_cs_data: 'No CS timeline data',
        laning_no_hp_data: 'No HP timeline data',
        laning_minutes: 'Minutes',
        laning_last_hits: 'Last Hits',
        laning_lh: 'LH',
        laning_hp_percent: 'HP%',

        // Lane name labels (for translating backend lane_name strings)
        lane_safe: 'Safe Lane',
        lane_mid: 'Mid Lane',
        lane_off: 'Off Lane',
        lane_roaming: 'Roaming',
        lane_unknown: 'Unknown',

        // Combat section
        combat_title: 'Combat Analysis',
        combat_empty_title: 'Combat data not available',
        combat_empty_desc: 'Local parsing does not provide kill/damage matrix data. Use OpenDota API for full data.',
        combat_kill_matrix: 'Kill Matrix',
        combat_damage_matrix: 'Damage Matrix',
        combat_kills_arrow: 'Kills \u2192',
        combat_damage_arrow: 'Damage \u2192',

        // Farm section
        farm_title: 'Farm Sources',
        farm_empty_title: 'Farm data not available',
        farm_empty_desc: 'Local parsing does not provide farm source details. Use OpenDota API for full data.',
        farm_gold_sources: 'Gold Sources',
        farm_xp_sources: 'XP Sources',
        farm_unit_kills: 'Unit Kills',
        farm_lh_timeline: 'LH Timeline',
        farm_heroes: 'Heroes',
        farm_creeps: 'Creeps',
        farm_neutrals: 'Neutrals',
        farm_ancients: 'Ancients',
        farm_towers: 'Towers',
        farm_roshan: 'Roshan',
        farm_courier: 'Courier',
        farm_no_data: 'No data',

        // Farm reason colors
        farm_reason_creeps: 'Creeps',
        farm_reason_heroes: 'Heroes',
        farm_reason_buildings: 'Buildings',
        farm_reason_roshan: 'Roshan',
        farm_reason_courier: 'Courier',
        farm_reason_other: 'Other',
        farm_reason_sell: 'Sell',
        farm_reason_death: 'Death',
        farm_reason_buyback: 'Buyback',
        farm_reason_unknown: 'Unknown',

        // Purchase section
        purchases_title: 'Item Purchase Timeline',
        purchases_empty_title: 'Purchase data not available',
        purchases_empty_desc: 'Local parsing does not provide purchase timeline data. Use OpenDota API for full data.',

        // Benchmark section
        benchmarks_title: 'Performance Benchmarks',
        benchmarks_empty_title: 'Benchmark data not available',
        benchmarks_empty_desc: 'Benchmarks require OpenDota API percentile data. Local parsing cannot provide this.',
        benchmarks_kills_per_min: 'Kills/min',
        benchmarks_lh_per_min: 'LH/min',
        benchmarks_dmg_per_min: 'Damage/min',
        benchmarks_healing_per_min: 'Healing/min',
        benchmarks_tower_dmg: 'Tower Dmg',

        // Objectives section
        objectives_title: 'Objectives & Events',
        objectives_empty_title: 'Objectives data not available',
        objectives_empty_desc: 'Local parsing does not provide objective event data. Use OpenDota API for full data.',
        objectives_timeline: 'Event Timeline',
        objectives_rune_stats: 'Rune Stats',
        rune_dd: 'DD',
        rune_haste: 'Haste',
        rune_illusion: 'Illusion',
        rune_invis: 'Invis',
        rune_regen: 'Regen',
        rune_arcane: 'Arcane',
        rune_bounty: 'Bounty',
        rune_water: 'Water',

        // Chat section
        chat_title: 'Chat Log',
        chat_empty: 'No chat messages',
        chat_chatwheel: '[Chatwheel]',

        // AI Coach section
        ai_coach_select_player: 'Select Player to Analyze',
        ai_coach_loading: 'AI Coach is analyzing, please wait...',
        ai_coach_loading_detail: 'Analyzing key decision points and deaths',

        // Progress stage labels (used by SSE stream of /ai-coach-stream)
        coach_stage_starting: 'Starting...',
        coach_stage_detect: 'Detecting decision points...',
        coach_stage_causal_chain: 'Tracing death causal chains...',
        coach_stage_context: 'Building scene context...',
        coach_stage_evaluate: 'AI evaluating decisions...',
        coach_stage_comparison: 'Comparing with pro players...',
        coach_stage_phases: 'Phase-by-phase review (laning / early-mid / mid-late / late)...',
        coach_stage_summary: 'Compiling final report...',
        coach_stage_done: 'Done',

        // Phase reviews
        ai_coach_phase_reviews: 'Phase-by-Phase Review',
        ai_coach_phase_highlights: 'Highlights',
        ai_coach_phase_issues: 'Issues',
        ai_coach_phase_suggestions: 'Suggestions',
        ai_coach_phase_lh: 'LH',
        phase_laning: 'Laning',
        phase_early_mid: 'Early-Mid',
        phase_mid_late: 'Mid-Late',
        phase_late: 'Late',
        phase_ultra_late: 'Ultra-Late',
        ai_coach_overall: 'Overall Assessment',
        ai_coach_death_analysis: 'Death Analysis',
        ai_coach_deaths_suffix: 'deaths',
        ai_coach_issue: 'Issue',
        ai_coach_suggestion: 'Suggestion',
        ai_coach_lesson: 'Lesson',
        ai_coach_causal_chain: 'Death Causal Chain',
        ai_coach_root_cause: 'Root Cause',
        ai_coach_root_tag: '[Root]',
        ai_coach_death_tag: '[Death]',
        ai_coach_key_moments: 'Key Moments',
        ai_coach_laning_summary: 'Laning Phase Summary',
        ai_coach_cs_10min: 'CS@10min',
        ai_coach_denies_10min: 'Denies@10min',
        ai_coach_lane_won: 'Won',
        ai_coach_lane_lost: 'Lost',
        ai_coach_lane_even: 'Even',
        ai_coach_lane_result: 'Lane Result',
        ai_coach_lane_matchup: 'Lane Matchup',
        ai_coach_pro_comparison: 'Pro Benchmark Comparison',
        ai_coach_improvement: 'Needs Improvement',
        ai_coach_strengths: 'Strengths',
        ai_coach_failed: 'AI Coach analysis failed',

        // Time-related strings
        time_seconds_ago: 's ago',
        time_minutes_ago: 'min ago',
        time_hours_ago: 'h ago',
        time_days_ago: 'd ago',
        time_months_ago: 'mo ago',

        // Chart labels
        chart_game_time: 'Game Time (minutes)',
        chart_gold_diff: 'Gold Diff',
        chart_xp_diff: 'XP Diff',

        // Error messages
        error_invalid_format: 'Please upload a .dem replay file',
        error_upload_failed: 'Upload failed',
        error_demo_failed: 'Failed to load demo data',
        error_fetch_failed: 'Failed to fetch match data',
        error_basic_data_fetched: 'Basic data fetched. Detailed data (wards/teamfights/economy) requires OpenDota parsing. Parse requested automatically.',

        // Empty state generic
        empty_local_parse_hint: 'Local parsing does not provide this data. Use OpenDota API for full data.',
    },
};
