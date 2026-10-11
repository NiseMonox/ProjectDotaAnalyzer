/**
 * Dota 2 Match Analyzer - Internationalization (i18n)
 * Supports: zh (Chinese), en (English)
 *
 * Three tables, one lookup (app.js: t / tn / te / codeLabel):
 *   I18N_MESSAGES  legacy flat keys, one block per language (keep both in step)
 *   I18N_PAIRS     'namespace.key': [zh, en] - use this for every new string
 *   I18N_CODES     labels of backend codes: codeLabel('game_mode', 'turbo')
 * The runtime at the bottom merges them and exposes window.I18N.
 */
window.I18N_MESSAGES = {
    zh: {
        // App title
        app_title: 'Dota 2 对局分析器',

        // Match banner
        radiant_victory: '天辉胜利',
        dire_victory: '夜魇胜利',

        // Scoreboard headers
        hero: '英雄',
        level: '等级',
        items_col: '装备',
        total: '合计',

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
        tab_ai_coach: 'AI 教练',

        // Economy tab
        economy_title: '经济/经验曲线',
        economy_gold_diff: '经济差',
        economy_gold_total: '经济总量',
        economy_xp_diff: '经验差',
        economy_xp_total: '经验总量',
        economy_player_gold: '个人经济',
        economy_player_xp: '个人经验',
        economy_player_lh: '个人补刀',
        economy_diff_hint: '零线以上（绿色）天辉领先，零线以下（红色）夜魇领先',
        economy_radiant_lead: '天辉领先',
        economy_dire_lead: '夜魇领先',

        // Item builds
        items_title: '出装分析',

        // Skill builds
        skills_title: '加点分析',

        // Ward section
        wards_title: '眼位分析',
        wards_observer: '侦察守卫',
        wards_sentry: '岗哨守卫',
        wards_timeline: '时间轴',
        wards_vision_range: '视野范围',
        wards_obs_short: '侦察',
        wards_sen_short: '岗哨',
        wards_dewarded_obs: '被反(侦察)',
        wards_dewarded_sen: '被反(岗哨)',
        wards_avg_obs_life: '侦察平均存活',
        wards_avg_sen_life: '岗哨平均存活',
        wards_active: '活跃眼位',
        wards_timeline_title: '眼位时间线',
        wards_placed: '放置',

        // Smoke section
        smokes_title: '开雾分析',
        smokes_success_rate: '成功率',
        smokes_timeline_mode: '时间轴模式',
        smokes_radiant_legend: '天辉开雾',
        smokes_dire_legend: '夜魇开雾',
        smokes_movement_path: '移动路径',
        smokes_break_point: '破雾位置',
        smokes_detail_records: '详细记录',
        smokes_click_to_select: '点击选择显示',
        smokes_show_all: '显示全部',
        smokes_duration_label: '持续',
        smokes_result_label: '结果',
        smokes_start_pos: '开雾位置',
        smokes_direction: '方向',

        // Teamfight section
        teamfights_title: '团战分析',
        teamfights_pause: '暂停',
        teamfights_play: '播放',
        teamfights_speed: '倍速',
        teamfights_abilities: '技能',
        teamfights_time: '时间',
        teamfights_duration: '持续',
        teamfights_result: '结果',
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
        laning_lane: '路线',
        laning_efficiency: '效率',
        laning_gold_10: '10 分经济',
        laning_xp_10: '10 分经验',
        laning_dmg_dealt: '伤害输出',
        laning_dmg_taken: '伤害承受',
        laning_regen_items: '回复道具',
        laning_cs_title: '对线期补刀曲线',
        laning_no_cs_data: '暂无补刀曲线数据',
        laning_no_hp_data: '暂无血量曲线数据',
        laning_lh: '补刀',

        // Combat section
        combat_title: '战斗分析',
        combat_kill_matrix: '击杀矩阵',
        combat_damage_matrix: '伤害矩阵',

        // Farm section
        farm_title: '经济来源',
        farm_gold_sources: '经济来源',
        farm_xp_sources: '经验来源',
        farm_unit_kills: '击杀单位',
        farm_lh_timeline: '补刀曲线',
        farm_heroes: '击杀英雄',
        farm_creeps: '小兵',
        farm_neutrals: '中立',
        farm_ancients: '远古',
        farm_towers: '塔',
        farm_roshan: '肉山',
        farm_courier: '信使',
        farm_no_data: '暂无数据',

        // Purchase section
        purchases_title: '物品购买时间线',

        // Benchmark section
        benchmarks_title: '表现评估',

        // Objectives section
        objectives_title: '目标与事件',
        objectives_timeline: '事件时间线',
        objectives_rune_stats: '神符统计',

        // Chat section
        chat_title: '聊天记录',

        // Chart labels
        chart_game_time: '游戏时间 (分钟)',

    },

    en: {
        // App title
        app_title: 'Dota 2 Match Analyzer',

        // Match banner
        radiant_victory: 'Radiant victory',
        dire_victory: 'Dire victory',

        // Scoreboard headers
        hero: 'Hero',
        level: 'Level',
        items_col: 'Items',
        total: 'Total',

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
        tab_ai_coach: 'AI coach',

        // Economy tab
        economy_title: 'Gold / XP graphs',
        economy_gold_diff: 'Gold diff',
        economy_gold_total: 'Gold total',
        economy_xp_diff: 'XP diff',
        economy_xp_total: 'XP total',
        economy_player_gold: 'Player gold',
        economy_player_xp: 'Player XP',
        economy_player_lh: 'Player LH',
        economy_diff_hint: 'Above zero (green) Radiant leads, below zero (red) Dire leads',
        economy_radiant_lead: 'Radiant leads',
        economy_dire_lead: 'Dire leads',

        // Item builds
        items_title: 'Item builds',

        // Skill builds
        skills_title: 'Skill builds',

        // Ward section
        wards_title: 'Ward analysis',
        wards_observer: 'Observer wards',
        wards_sentry: 'Sentry wards',
        wards_timeline: 'Timeline',
        wards_vision_range: 'Vision range',
        wards_obs_short: 'Observer',
        wards_sen_short: 'Sentry',
        wards_dewarded_obs: 'Dewarded (obs)',
        wards_dewarded_sen: 'Dewarded (sen)',
        wards_avg_obs_life: 'Avg obs life',
        wards_avg_sen_life: 'Avg sen life',
        wards_active: 'Active wards',
        wards_timeline_title: 'Ward timeline',
        wards_placed: 'Placed',

        // Smoke section
        smokes_title: 'Smoke analysis',
        smokes_success_rate: 'success rate',
        smokes_timeline_mode: 'Timeline mode',
        smokes_radiant_legend: 'Radiant smoke',
        smokes_dire_legend: 'Dire smoke',
        smokes_movement_path: 'Movement path',
        smokes_break_point: 'Smoke break',
        smokes_detail_records: 'Detailed records',
        smokes_click_to_select: 'Click to select',
        smokes_show_all: 'Show all',
        smokes_duration_label: 'Duration',
        smokes_result_label: 'Result',
        smokes_start_pos: 'Smoke start',
        smokes_direction: 'Direction',

        // Teamfight section
        teamfights_title: 'Teamfight analysis',
        teamfights_pause: 'Pause',
        teamfights_play: 'Play',
        teamfights_speed: 'Speed',
        teamfights_abilities: 'Abilities',
        teamfights_time: 'Time',
        teamfights_duration: 'Duration',
        teamfights_result: 'Result',
        teamfights_kill_order: 'Kill order',
        teamfights_hero: 'Hero',
        teamfights_damage: 'Damage',
        teamfights_taken: 'Taken',
        teamfights_healing: 'Healing',
        teamfights_gold: 'Gold',
        teamfights_died: 'Died',
        teamfights_ability_cast: 'Ability cast',
        teamfights_hero_pos: 'Hero position',

        // Laning section
        laning_title: 'Laning analysis',
        laning_lane: 'Lane',
        laning_efficiency: 'Efficiency',
        laning_gold_10: 'Gold@10',
        laning_xp_10: 'XP@10',
        laning_dmg_dealt: 'Dmg dealt',
        laning_dmg_taken: 'Dmg taken',
        laning_regen_items: 'Regen items',
        laning_cs_title: 'Laning CS timeline',
        laning_no_cs_data: 'No CS timeline data',
        laning_no_hp_data: 'No HP timeline data',
        laning_lh: 'LH',

        // Combat section
        combat_title: 'Combat analysis',
        combat_kill_matrix: 'Kill matrix',
        combat_damage_matrix: 'Damage matrix',

        // Farm section
        farm_title: 'Farm sources',
        farm_gold_sources: 'Gold sources',
        farm_xp_sources: 'XP sources',
        farm_unit_kills: 'Unit kills',
        farm_lh_timeline: 'LH timeline',
        farm_heroes: 'Hero kills',
        farm_creeps: 'Creeps',
        farm_neutrals: 'Neutrals',
        farm_ancients: 'Ancients',
        farm_towers: 'Towers',
        farm_roshan: 'Roshan',
        farm_courier: 'Courier',
        farm_no_data: 'No data',

        // Purchase section
        purchases_title: 'Item purchase timeline',

        // Benchmark section
        benchmarks_title: 'Performance benchmarks',

        // Objectives section
        objectives_title: 'Objectives & events',
        objectives_timeline: 'Event timeline',
        objectives_rune_stats: 'Rune stats',

        // Chat section
        chat_title: 'Chat log',

        // Chart labels
        chart_game_time: 'Game time (minutes)',

    },
};

/* ====================================================================
 * Paired strings — the convention for everything added from now on.
 *
 *   'namespace.key': [zh, en]
 *
 * Both languages sit on one line, so a string cannot exist in only one of
 * them. Placeholders are {name}; plural-aware strings come as
 * '<key>.one' / '<key>.other' and are read with tn(key, n).
 * The loader at the bottom of this file copies every pair into
 * I18N_MESSAGES, so t('namespace.key') works like any legacy key.
 * ==================================================================== */
window.I18N_PAIRS = {
    // --- Language names are autonyms: each is shown in its own language ---
    'lang.label': ['界面语言', 'Interface language'],
    'lang.zh': ['中文', '中文'],
    'lang.en': ['English', 'English'],

    // --- Common ---
    'common.retry': ['重试', 'Retry'],
    'common.cancel': ['取消', 'Cancel'],
    'common.dismiss': ['关闭提示', 'Dismiss'],
    'common.loading': ['加载中…', 'Loading…'],
    'common.no_data': ['暂无数据', 'No data'],
    'common.unknown': ['未知', 'Unknown'],
    'common.back': ['返回', 'Back'],
    'common.notifications': ['通知', 'Notifications'],

    // --- Durations and relative time ---
    'time.sec': ['{s} 秒', '{s} s'],
    'time.min': ['{m} 分钟', '{m} min'],
    'time.min_sec': ['{m} 分 {s} 秒', '{m} min {s} s'],
    'time.hour': ['{h} 小时', '{h} h'],
    'time.hour_min': ['{h} 小时 {m} 分', '{h} h {m} min'],
    'time.ago.now': ['刚刚', 'just now'],
    'time.ago.minutes.one': ['{n} 分钟前', '{n} minute ago'],
    'time.ago.minutes.other': ['{n} 分钟前', '{n} minutes ago'],
    'time.ago.hours.one': ['{n} 小时前', '{n} hour ago'],
    'time.ago.hours.other': ['{n} 小时前', '{n} hours ago'],
    'time.ago.days.one': ['{n} 天前', '{n} day ago'],
    'time.ago.days.other': ['{n} 天前', '{n} days ago'],
    'time.ago.months.one': ['{n} 个月前', '{n} month ago'],
    'time.ago.months.other': ['{n} 个月前', '{n} months ago'],
    'time.ago.years.one': ['{n} 年前', '{n} year ago'],
    'time.ago.years.other': ['{n} 年前', '{n} years ago'],

    // --- Start page ---
    'home.heading': ['分析一场 Dota 2 比赛', 'Analyze a Dota 2 match'],
    'home.tagline': ['上传录像或输入比赛 ID，查看整场比赛的数据分析与 AI 教练复盘。', 'Upload a replay or enter a match ID for full match analytics and an AI coach review.'],
    'home.or': ['或', 'or'],
    'home.status_label': ['服务器状态', 'Server status'],
    'home.status.patch': ['游戏数据 {patch}', 'Game data {patch}'],
    'home.status.parser_on': ['录像解析可用', 'Replay parsing available'],
    'home.status.parser_off': ['录像解析不可用', 'Replay parsing unavailable'],
    'home.status.coach_llm': ['AI 教练：大模型分析', 'AI coach: LLM analysis'],
    'home.status.coach_rules': ['AI 教练：规则分析', 'AI coach: rule-based'],
    'home.status.coach_rules_title': ['服务器未配置大模型，AI 教练报告由规则引擎生成。', 'No LLM is configured on this server; coach reports come from the rule engine.'],
    'home.status.quota': ['今日剩余 {n} 份', '{n} reports left today'],
    'home.status.offline': ['无法连接服务器', 'Server unreachable'],

    // --- Upload / fetch / demo flows ---
    'upload.drag_hint': ['将录像文件拖到此处，或点击选择文件', 'Drop a replay file here, or click to choose one'],
    'upload.drop_now': ['松开即可上传', 'Release to upload'],
    'upload.format_hint': ['支持 {exts}，单个文件不超过 {limit} MB', 'Accepts {exts} · up to {limit} MB per file'],
    'upload.format_hint_nolimit': ['支持 {exts}', 'Accepts {exts}'],
    'upload.file_label': ['选择录像文件', 'Choose a replay file'],
    'upload.match_id_label': ['比赛 ID', 'Match ID'],
    'upload.match_id_placeholder': ['比赛 ID 或 OpenDota 链接', 'Match ID or OpenDota link'],
    'upload.fetch_btn': ['获取比赛', 'Fetch match'],
    'upload.opendota_hint': ['通过 OpenDota 获取公开比赛的数据，无需上传录像（例如 8713267082）', 'Loads a public match from OpenDota, no replay upload needed (e.g. 8713267082)'],
    'upload.demo_btn': ['查看演示比赛', 'Open the demo match'],
    'upload.demo_hint': ['手边没有录像？可以先用演示比赛浏览各个分析页和 AI 教练。', 'No replay at hand? The demo match lets you look through the analysis tabs and the AI coach.'],
    'upload.uploading': ['正在上传录像', 'Uploading the replay'],
    'upload.parsing': ['上传完成，正在解析录像…', 'Upload complete, parsing the replay…'],
    'upload.loading_analysis': ['正在加载分析结果…', 'Loading the analysis…'],
    'upload.cancel': ['取消上传', 'Cancel upload'],
    'upload.time_left': ['剩余约 {time}', '{time} left'],
    'upload.slow_hint': ['上传速度较慢。.dem 文件较大（通常 50–200 MB），跨境网络可能要等很久，可以改用比赛 ID 直接获取数据。', 'The upload is slow. Replay files are large (usually 50–200 MB); fetching the match by its ID is much faster.'],
    'upload.cached': ['这份录像之前已解析过，已直接载入结果。', 'This replay was analyzed before, loaded straight from the cache.'],
    'upload.parser_off_title': ['服务器暂时无法解析录像文件', 'Replay parsing is unavailable on this server right now'],
    'upload.parser_off_text': ['可以改用比赛 ID，从 OpenDota 获取这场比赛。', 'You can fetch the match from OpenDota by its match ID instead.'],
    'upload.use_match_id': ['改用比赛 ID', 'Use a match ID instead'],
    'upload.fetching': ['正在从 OpenDota 获取比赛 {id}…', 'Fetching match {id} from OpenDota…'],
    'upload.loading_demo': ['正在加载演示比赛…', 'Loading the demo match…'],
    'upload.loading_match': ['正在加载比赛…', 'Loading the match…'],

    // --- Errors. Server errors arrive localized in `detail` (requests carry
    // ?lang=); these texts cover errors raised in the browser and re-render
    // a server error after the language was switched. ---
    'error.generic': ['出错了，请稍后重试。', 'Something went wrong. Please try again.'],
    'error.boot_cdn': ['页面所需的脚本（Vue / ECharts）未能加载，请检查网络后刷新页面。', 'The scripts this page needs (Vue / ECharts) could not be loaded. Check your network and reload.'],
    'error.network': ['无法连接服务器，请检查网络后重试。', 'Could not reach the server. Check your connection and try again.'],
    'error.network_upload': ['上传中断：与服务器的连接断开（服务器可能正在重启，或文件远超大小上限）。请重试。', 'The upload was interrupted: the connection to the server was lost (it may be restarting, or the file is far above the size limit). Please try again.'],
    'error.timeout': ['请求超时，请稍后重试。', 'The request timed out. Please try again.'],
    'error.server_error': ['服务器暂时出错（可能正在重启），请稍后重试。', 'The server had a temporary problem (it may be restarting). Please try again shortly.'],
    'error.bad_response': ['服务器返回了无法识别的数据，请稍后重试。', 'The server sent an unexpected response. Please try again.'],
    'error.http_error': ['请求失败（HTTP {status}）。', 'The request failed (HTTP {status}).'],
    'error.retry_in': ['{time}后可重试。', 'You can try again in {time}.'],
    'error.retry_now': ['现在可以重试了。', 'You can try again now.'],
    'error.upload_empty': ['该文件是空的。', 'This file is empty.'],
    'error.upload_bad_extension': ['仅支持 {exts} 文件。', 'Only {exts} files are supported.'],
    'error.upload_too_large_local': ['录像文件过大（{size} MB，上限 {limit} MB）。', 'The replay file is too large ({size} MB, limit {limit} MB).'],
    'error.upload_too_large': ['录像文件过大。', 'The replay file is too large.'],
    'error.invalid_match_id': ['比赛 ID 应为一串数字，例如 8713267082。', 'A match ID is a number, for example 8713267082.'],
    'error.rate_limited': ['请求过于频繁。', 'Too many requests.'],
    'error.daily_limit': ['今日 AI 教练分析额度已用完，请明天再试。', "Today's AI coach quota has been used up. Please try again tomorrow."],
    'error.server_busy': ['服务器正忙于处理其他上传，请稍后重试。', 'The server is busy with other uploads. Please retry shortly.'],
    'error.upload_in_progress': ['您的网络已有录像正在上传，请等待其完成后再试。', 'Another upload from your network is still in progress. Please wait for it to finish.'],
    'error.upload_timeout': ['上传耗时过长，已被取消。请检查网络后重试。', 'The upload took too long and was cancelled. Please check your connection and try again.'],
    'error.upload_missing_file': ['未收到录像文件。', 'No replay file was uploaded.'],
    'error.upload_failed': ['上传文件保存失败，请重试。', 'Failed to store the uploaded file. Please try again.'],
    'error.replay_invalid': ['该文件不是有效的 Dota 2 录像。', 'This file is not a valid Dota 2 replay.'],
    'error.replay_compressed': ['该录像为 Zstandard 压缩文件，本服务器无法解压。请先解压为 .dem 文件（例如 zstd -d）后再上传。', 'This replay is Zstandard-compressed and this server cannot unpack it. Please decompress it to a .dem file first (for example zstd -d).'],
    'error.replay_compressed_bzip2': ['该录像为 bzip2 压缩文件，本服务器无法解压。请先解压为 .dem 文件（例如 bunzip2）后再上传。', 'This replay is bzip2-compressed and this server cannot unpack it. Please decompress it to a .dem file first (for example with bunzip2).'],
    'error.replay_decompress_failed': ['该压缩录像已损坏或不完整，请重新下载，或自行解压后上传 .dem 文件。', 'This compressed replay is damaged or incomplete. Please download it again, or decompress it yourself and upload the .dem file.'],
    'error.replay_window_too_large': ['该录像的 Zstandard 压缩窗口超出了本服务器的解压上限，请自行解压（zstd -d）后上传 .dem 文件。', 'This replay was compressed with a Zstandard window larger than this server unpacks. Please decompress it yourself (zstd -d) and upload the .dem file.'],
    'error.replay_parse_failed': ['录像解析失败，文件可能已损坏或游戏版本不受支持。', 'Failed to parse the replay. The file may be corrupted or from an unsupported game version.'],
    'error.parser_unavailable': ['服务器暂时无法解析录像文件，请改用比赛 ID 获取。', 'Replay parsing is not available on this server right now. Please fetch the match by its ID instead.'],
    'error.opendota_not_found': ['OpenDota 上未找到该比赛。', 'Match not found on OpenDota.'],
    'error.opendota_unavailable': ['OpenDota 暂时不可用，请稍后重试。', 'OpenDota is unavailable right now. Please try again later.'],
    'error.match_not_found': ['未找到比赛数据，请重新上传录像或重新获取比赛。', 'Match not found. Please upload the replay or fetch the match again.'],
    'error.coach_no_game': ['该录像在比赛开始前或刚开始几分钟就已结束，没有可分析的对局内容，请上传完整的录像。', 'This replay ends before the game starts or within its first minutes, so there is nothing to analyze. Please upload the complete replay.'],
    'error.coach_failed': ['AI 教练分析失败，请稍后重试。', 'AI coaching analysis failed. Please try again later.'],
    'error.coach_timeout': ['AI 教练分析超时，请稍后重试。', 'AI coaching analysis timed out. Please try again later.'],
    'error.knowledge_not_found': ['知识库中已没有这条资料（之后的学习更新可能替换了它）。', 'This entry is no longer in the knowledge base (a later learning run may have replaced it).'],
    'error.invalid_player_slot': ['该玩家不在本场比赛中。', 'This player is not part of the match.'],
    'error.internal_error': ['服务器内部错误。', 'Internal server error.'],
    // Optional second line under an error
    'error_hint.opendota_not_found': ['请检查比赛 ID。非公开比赛，或刚结束几分钟的比赛，OpenDota 上可能还没有数据。', 'Check the match ID. Private matches, and matches that ended only minutes ago, may not be on OpenDota yet.'],
    'error_hint.upload_too_large': ['如果是公开比赛，可以改用比赛 ID 获取数据。', 'If it was a public match, you can fetch it by its match ID instead.'],
    'error_hint.replay_parse_failed': ['如果是公开比赛，可以改用比赛 ID 获取数据。', 'If it was a public match, you can fetch it by its match ID instead.'],
    'error.bad_link': ['这个链接不是有效的比赛地址，已回到首页。', 'This link is not a valid match address, so the start page is shown instead.'],
    'error_hint.link_not_found': ['这个链接对应的比赛已不在服务器上。', 'The match behind this link is no longer on the server.'],

    // --- Match header ---
    'match.result_unknown': ['胜负未知', 'Result unknown'],
    'match.pregame_only': ['仅有赛前阶段', 'Pre-game only'],
    'match.winner_badge': ['胜方', 'Winner'],
    'match.id_label': ['比赛 ID', 'Match ID'],
    'match.open_opendota': ['在 OpenDota 查看这场比赛', 'View this match on OpenDota'],
    'match.patch': ['版本 {patch}', 'Patch {patch}'],
    'match.build_title': ['游戏构建号 {build}', 'Game build {build}'],
    'match.region_title': ['服务器地区', 'Server region'],
    'match.played_title': ['比赛开始时间：{date}', 'Match start: {date}'],
    'match.score_title': ['击杀比分：天辉 {r}，夜魇 {d}', 'Kill score: Radiant {r}, Dire {d}'],
    'match.duration_title': ['比赛时长', 'Match duration'],
    'match.tabs_label': ['分析模块', 'Analysis sections'],
    'match.player_n': ['玩家 {n}', 'Player {n}'],
    'match.anonymous_title': ['数据中没有该玩家的昵称', 'The data has no name for this player'],

    // --- Notices shown above the tabs' content ---
    'notice.partial.title': ['录像不完整', 'Incomplete replay'],
    'notice.partial.text': ['这份录像在 {time} 处中断，比赛尚未结束：胜负未知，所有数据只统计到录像结束为止。', 'This recording stops at {time}, before the game ended: the winner is unknown and every number covers only the recorded part.'],
    'notice.partial.no_coach': ['录像中的比赛不足 5 分钟，无法进行 AI 教练分析。', 'It holds less than 5 minutes of game time, too little for an AI coach review.'],
    'notice.pregame.title': ['录像不完整（仅有赛前阶段）', 'Incomplete replay (pre-game only)'],
    'notice.pregame.text': ['这份录像在比赛开始（号角响起）前就结束了，只包含阵容和出门装，没有可分析的对局内容。请上传完整的录像。', 'This recording ends before the horn: it only holds the draft and the starting items, so there is no game to analyze. Please upload the complete replay.'],
    'notice.unparsed.title': ['OpenDota 尚未解析这场比赛的录像', 'OpenDota has not parsed the replay of this match yet'],
    'notice.unparsed.text': ['目前只有基础计分板数据（KDA、经济、最终出装）。眼位、团战、对线、经济曲线、购买时间和击杀记录都需要解析后的录像。', 'Only the basic scoreboard is available (KDA, net worth, final items). Wards, teamfights, laning, gold and XP graphs, purchase times and the kill log all need the parsed replay.'],
    'notice.unparsed.request_btn': ['请求 OpenDota 解析', 'Ask OpenDota to parse it'],
    'notice.unparsed.requesting': ['正在提交请求…', 'Sending the request…'],
    'notice.unparsed.requested': ['已向 OpenDota 提交解析请求，通常需要几分钟。稍后点击“重新获取”即可载入完整数据。', 'OpenDota was asked to parse the replay; this usually takes a few minutes. Use “Check again” afterwards to load the full data.'],
    'notice.unparsed.refetch_btn': ['重新获取', 'Check again'],
    'notice.unparsed.refetching': ['正在重新获取…', 'Checking…'],
    'notice.unparsed.still': ['OpenDota 还没有解析完成，请稍后再试。', 'OpenDota has not finished parsing yet. Try again a little later.'],
    'notice.unparsed.ready': ['已载入解析后的完整数据。', 'The parsed replay data is loaded.'],
    'notice.unparsed.upload_hint': ['也可以直接上传这场比赛的录像文件，由本站解析。', 'You can also upload the replay file of this match and have it parsed here.'],
    'notice.mock.title': ['演示数据', 'Demo data'],
    'notice.mock.text': ['当前显示的是程序生成的示例比赛，并非真实对局。', 'This is a generated sample match, not a real game.'],

    // --- Scoreboard ---
    'col.level_full': ['英雄等级', 'Hero level'],
    'col.k': ['K', 'K'],
    'col.k_full': ['击杀', 'Kills'],
    'col.d': ['D', 'D'],
    'col.d_full': ['死亡', 'Deaths'],
    'col.a': ['A', 'A'],
    'col.a_full': ['助攻', 'Assists'],
    'col.lhdn': ['正补/反补', 'LH/DN'],
    'col.lhdn_full': ['正补 / 反补', 'Last hits / Denies'],
    'col.net': ['财产总和', 'NET'],
    'col.net_full': ['财产总和', 'Net worth'],
    'col.gxpm': ['GPM/XPM', 'GPM/XPM'],
    'col.gxpm_full': ['每分钟金钱 / 每分钟经验', 'Gold per minute / XP per minute'],
    'col.hd': ['英雄伤害', 'HD'],
    'col.hd_full': ['对英雄造成的伤害', 'Hero damage'],
    'col.td': ['建筑伤害', 'TD'],
    'col.td_full': ['对建筑造成的伤害', 'Tower damage'],
    'col.hh': ['治疗', 'HH'],
    'col.hh_full': ['对英雄的治疗量', 'Hero healing'],
    'items.neutral': ['中立物品', 'Neutral item'],
    'items.neutral_named': ['中立物品：{name}', 'Neutral item: {name}'],
    'items.backpack_named': ['背包：{name}', 'Backpack: {name}'],
    'items.empty_slot': ['空栏位', 'Empty slot'],
    'items.bought_at': ['{name}（{time} 购入）', '{name} (bought at {time})'],
    'items.recipe_of': ['{name}卷轴', '{name} Recipe'],

    // --- Draft ---
    'draft.title': ['阵容与禁用', 'Draft'],
    'draft.bans': ['禁用', 'Bans'],
    'draft.bans_shared_hint': ['这个模式的禁用不属于任何一方，也没有先后顺序', 'Bans in this game mode belong to no team and have no order'],
    'draft.pick_title': ['{hero}（第 {n} 手）', '{hero} (pick #{n})'],
    'draft.ban_title': ['{hero}（禁用）', '{hero} (banned)'],

    // ================= Analysis tabs =================
    'common.list_sep': ['，', ', '],

    // --- Why a tab has nothing to show: 'empty.<tab>.title', then the
    // reason ('empty.<tab>.<reason>' when the tab has its own wording,
    // else 'empty.reason.<reason>'). Reasons: none | pregame | unparsed |
    // opendota (the source never has it) | replay (only OpenDota has it). ---
    'empty.reason.none': ['这场比赛没有这类数据。', 'This match has no data of this kind.'],
    'empty.reason.pregame': ['录像在比赛开始前就结束了，还没有这类数据。', 'The recording ends before the game starts, so there is nothing of this kind yet.'],
    'empty.reason.unparsed': ['这部分数据需要解析后的录像，而 OpenDota 尚未解析这场比赛（见上方提示）。', 'This needs the parsed replay, and OpenDota has not parsed this match yet (see the notice above).'],
    'empty.reason.opendota': ['OpenDota 不提供这类数据；上传这场比赛的录像文件即可查看。', 'OpenDota does not provide this data; upload the replay file of this match to see it.'],
    'empty.reason.replay': ['只有通过比赛 ID 从 OpenDota 获取的比赛才有这类数据。', 'Only matches fetched from OpenDota by match ID have this data.'],
    'empty.economy.title': ['没有经济 / 经验曲线', 'No gold / XP graphs'],
    'empty.items.title': ['没有出装数据', 'No item data'],
    'empty.skills.title': ['没有加点数据', 'No skill builds'],
    'empty.wards.title': ['没有眼位数据', 'No ward data'],
    'empty.wards.none': ['这场比赛没有记录到守卫。', 'No ward was recorded in this match.'],
    'empty.smokes.title': ['没有开雾数据', 'No smoke data'],
    'empty.smokes.none': ['这场比赛没有人使用诡计之雾。', 'Nobody used a Smoke of Deceit in this match.'],
    'empty.teamfights.title': ['没有团战数据', 'No teamfight data'],
    'empty.teamfights.none': ['这场比赛没有记录到团战。', 'No teamfight was recorded in this match.'],
    'empty.laning.title': ['没有对线数据', 'No laning data'],
    'empty.combat.title': ['没有战斗数据', 'No combat data'],
    'empty.farm.title': ['没有经济来源数据', 'No farm data'],
    'empty.purchases.title': ['没有购买记录', 'No purchase data'],
    'empty.benchmarks.title': ['没有表现评估数据', 'No benchmarks'],
    'empty.benchmarks.replay': ['表现评估是 OpenDota 的百分位统计：只有通过比赛 ID 从 OpenDota 获取的比赛才有。', 'Benchmarks are OpenDota percentiles: they only exist for matches fetched from OpenDota by match ID.'],
    'empty.objectives.title': ['没有目标事件', 'No objectives'],
    'empty.objectives.none': ['还没有建筑被摧毁，也没有记录到其他目标事件。', 'No building has fallen and no other objective was recorded.'],
    'empty.chat.title': ['没有聊天记录', 'No chat'],
    'empty.chat.none': ['这场比赛没有人打字，也没有人使用聊天轮盘。', 'Nobody typed or used the chat wheel in this match.'],

    // --- Economy ---
    'economy.minute_hint': ['每分钟一个数据点，到比赛结束为止。', 'One value per minute, up to the end of the game.'],

    // --- Items / purchases ---
    'items.backpack': ['背包', 'Backpack'],
    'items.no_timeline': ['这里只有最终出装，没有购买顺序。', 'Only the final builds are known here, not the order of purchase.'],
    'purchases.bucket': ['{from}–{to} 分钟', '{from}–{to} min'],

    // --- Skills ---
    'skills.hint': ['每一列是一次加点，格内数字是该技能加点后的等级，★ 为天赋。', 'Each column is one skill point; the number is the level the ability reached, ★ marks a talent.'],
    'skills.estimated_hint': ['OpenDota 不记录加点时间，带 ≈ 的时间为估算值。', 'OpenDota has no skill timestamps: times marked ≈ are estimates.'],
    'skills.level': ['{n} 级', 'Level {n}'],
    'skills.ability': ['技能', 'Ability'],
    'skills.casts': ['施放', 'Casts'],
    'skills.casts_title': ['整场比赛的施放次数', 'Times cast over the whole game'],
    'skills.ultimate': ['大招', 'Ult'],
    'skills.talents': ['天赋', 'Talents'],
    'skills.none': ['没有该玩家的加点记录。', 'No skill build recorded for this player.'],
    'skills.step_title': ['第 {n} 步：{name} {level} 级（{time}）', 'Step {n}: {name} level {level} ({time})'],
    'skills.step_talent': ['第 {n} 步：天赋「{name}」（{time}）', 'Step {n}: talent “{name}” ({time})'],

    // --- Vision ---
    'wards.unknown_owner': ['未知玩家', 'Unknown player'],
    'wards.dewarded_by': ['{time} 被 {hero} 反掉', 'Dewarded by {hero} at {time}'],
    'wards.dewarded_at': ['{time} 被反掉', 'Dewarded at {time}'],
    'wards.denied_by': ['{time} 被己方 {hero} 反补', 'Denied by {hero} at {time}'],
    'wards.denied_at': ['{time} 被己方反补', 'Denied at {time}'],
    'wards.expired_at': ['{time} 到时消失', 'Expired at {time}'],
    'wards.destroyed_at': ['{time} 被摧毁', 'Destroyed at {time}'],
    'wards.outcomes': ['守卫结局', 'How the wards ended'],
    'wards.observer_one': ['侦察守卫', 'Observer ward'],
    'wards.sentry_one': ['岗哨守卫', 'Sentry ward'],
    'wards.none_match': ['没有符合当前筛选条件的守卫。', 'No ward matches the current filters.'],
    'wards.standing': ['数据结束时仍在', 'Still standing at the end'],
    'wards.col_type': ['类型', 'Type'],
    'wards.col_owner': ['放置者', 'Placed by'],
    'wards.col_state': ['结局', 'Outcome'],
    'wards.col_life': ['存活时间', 'Lifetime'],
    'wards.life_title': ['存活 {life}（完整时长 {max}）', 'Stood for {life} (full duration {max})'],
    'wards.vision_title': ['侦察守卫视野 {obs}，岗哨守卫真视范围 {sen}（游戏内距离单位）', 'Observer vision {obs}, sentry true sight {sen} (game units)'],
    'wards.map_label': ['眼位地图', 'Ward map'],

    // --- Smokes ---
    'smokes.team_total': ['{team}开雾', '{team} smokes'],
    'smokes.selected': ['已选 {n} / {total}', '{n} of {total} selected'],
    'smokes.used_by': ['使用者：{hero}', 'Used by {hero}'],
    'smokes.heroes.one': ['{n} 名英雄', '{n} hero'],
    'smokes.heroes.other': ['{n} 名英雄', '{n} heroes'],
    'smokes.kills.one': ['击杀 {n}', '{n} kill'],
    'smokes.kills.other': ['击杀 {n}', '{n} kills'],
    'smokes.deaths.one': ['阵亡 {n}', '{n} death'],
    'smokes.deaths.other': ['阵亡 {n}', '{n} deaths'],
    'smokes.map_label': ['开雾路线地图', 'Smoke route map'],
    'smokes.timeline_label': ['开雾时间分布', 'Smokes over time'],

    // --- Teamfights ---
    'teamfights.pick_label': ['选择团战', 'Choose a teamfight'],
    'teamfights.deaths_title': ['阵亡：天辉 {r}，夜魇 {d}', 'Deaths: Radiant {r}, Dire {d}'],
    'teamfights.map_label': ['团战位置回放', 'Teamfight position playback'],
    'teamfights.slider_label': ['回放时间', 'Playback time'],
    'teamfights.no_positions': ['此数据来源只记录了英雄的阵亡位置，无法回放走位；上传这场比赛的录像文件可查看完整回放。', 'This data source only records where heroes died, so there is no movement to play back; uploading the replay file of this match gives the full playback.'],
    'teamfights.death_marker': ['阵亡位置', 'Death'],
    'teamfights.gold_swing': ['经济变化', 'Gold swing'],
    'teamfights.xp_swing': ['经验变化', 'XP swing'],
    'teamfights.swing_title': ['这场团战中哪一方多赚了多少', 'Which side came out ahead in this fight, and by how much'],
    'teamfights.kd_title': ['团战中的击杀 / 死亡', 'Kills / deaths in this fight'],
    'teamfights.xp': ['经验', 'XP'],
    'teamfights.used': ['技能与物品', 'Abilities & items'],
    'teamfights.buyback_title': ['在这场团战中买活', 'Bought back during this fight'],
    'teamfights.buybacks.one': ['买活', 'buyback'],
    'teamfights.buybacks.other': ['买活 ×{n}', 'buyback ×{n}'],
    'teamfights.item_used': ['{name} ×{n}', '{name} ×{n}'],

    // --- Laning ---
    'laning.efficiency_title': ['10:00 时的经济占理想对线收入的比例', 'Gold at 10:00 as a share of an ideal lane income'],
    'laning.efficiency_na': ['{mode}的金钱收入加倍，对线效率（以常规比赛的理想对线收入为基准）不适用，因此不计算。', 'Lane efficiency is not computed for {mode}: it measures gold against the ideal lane income of a standard match, and this mode doubles the gold.'],
    'laning.lh10': ['10 分正补', 'LH@10'],
    'laning.dn10': ['10 分反补', 'DN@10'],
    'laning.lh_title': ['10:00 时的正补数', 'Last hits at 10:00'],
    'laning.dn_title': ['10:00 时的反补数', 'Denies at 10:00'],
    'laning.pick_title': ['查看 {hero} 的对线期站位', 'Show where {hero} stood during laning'],
    'laning.no_details': ['OpenDota 没有对线期的伤害、回复道具和血量数据；上传这场比赛的录像文件可获得这些数据。', 'OpenDota has no laning damage, regen usage or HP data; uploading the replay file of this match provides them.'],
    'laning.no_details_partial': ['录像在记录到对线期的伤害、回复道具和血量数据之前就中断了。', 'The recording stops before it holds any laning damage, regen usage or HP data.'],
    'laning.no_details_none': ['这场比赛的数据中没有对线期的伤害、回复道具和血量数据。', 'The data of this match has no laning damage, regen usage or HP data.'],
    'laning.focus_title': ['单个英雄：对线期站位与状态（前 10 分钟）', 'One hero: position and vitals during laning (first 10 minutes)'],
    'laning.heat_title': ['对线期站位热力图', 'Laning position heat map'],
    'laning.vitals_title': ['生命与魔法百分比', 'Health and mana, % of maximum'],
    'laning.hp': ['生命', 'Health'],
    'laning.attack': ['普通攻击', 'Attack'],
    'laning.sources_title': ['对线期对英雄伤害的来源', 'Laning damage to heroes, by source'],
    'laning.regen_title': ['对线期使用的回复道具', 'Regen items used in lane'],
    'laning.mana': ['魔法', 'Mana'],
    'laning.heat_less': ['停留少', 'Less time'],
    'laning.heat_more': ['停留多', 'More time'],
    'laning.no_positions': ['数据中没有该英雄对线期的位置。', 'The data has no laning positions for this hero.'],

    // --- Combat ---
    'combat.feed': ['击杀记录', 'Kill feed'],
    'combat.stats': ['玩家数据', 'Player stats'],
    'combat.kills_hint': ['行是击杀者，列是被击杀者。', 'Rows are the killers, columns the victims.'],
    'combat.damage_hint': ['行是伤害来源，列是承受者（对英雄造成的伤害）。', 'Rows deal the damage, columns take it (damage to heroes).'],
    'combat.killer_victim': ['击杀者 ↓　被击杀 →', 'Killer ↓  Victim →'],
    'combat.dealer_target': ['伤害来源 ↓　承受者 →', 'Dealer ↓  Target →'],
    'combat.deaths_row': ['被击杀合计', 'Deaths'],
    'combat.taken_row': ['承受合计', 'Taken'],
    'combat.no_matrix': ['此数据来源没有这项矩阵数据。', 'This data source has no such matrix.'],
    'combat.feed_hint': ['整场比赛的每一次击杀；比分为该次击杀之后的比分。', 'Every kill of the game; the score is the one after that kill.'],
    'combat.denied': ['反补', 'Denied'],
    'combat.uncounted': ['疑似重生', 'Likely reincarnation'],
    'combat.uncounted_title': ['数据无法区分这是一次阵亡还是重生（冥魂大帝大招 / 不朽之守护）：按重生处理，不计入比分。', 'The data cannot tell this death from a reincarnation (Wraith King ultimate / Aegis): it is treated as a reincarnation and not counted in the score.'],
    'combat.unconfirmed_title': ['这次阵亡是根据计分板数据推断出来的。', 'This death was inferred from the scoreboard.'],
    'combat.unknown_killer': ['未知', 'Unknown'],
    'combat.assists_title': ['助攻：{names}', 'Assists: {names}'],
    'combat.col_stuns': ['控制时长', 'Stuns'],
    'combat.col_stuns_title': ['对敌方英雄造成的控制总时长', 'Total stun time dealt to enemy heroes'],
    'combat.col_apm': ['APM', 'APM'],
    'combat.col_apm_title': ['每分钟操作数', 'Actions per minute'],
    'combat.col_participation': ['参团率', 'Fight part.'],
    'combat.col_participation_title': ['参与的团战占全部团战的比例', 'Share of the teamfights the hero took part in'],
    'combat.col_buybacks': ['买活', 'Buybacks'],
    'combat.col_multi': ['最高多杀', 'Best multi-kill'],
    'combat.col_streak': ['最长连杀', 'Longest streak'],
    'combat.streak_beyond': ['{label}（{n} 连杀）', '{label} ({n} kills)'],

    // --- Farm ---
    'farm.gold_hint': ['按来源统计的金钱；负值是死亡等造成的损失。', 'Gold by source; negative bars are losses such as deaths.'],
    'farm.xp_hint': ['按来源统计的经验。', 'Experience by source.'],
    'farm.no_reasons': ['此数据来源没有按来源拆分的收入数据。', 'This data source has no income broken down by source.'],
    'farm.no_lh': ['数据中没有补刀曲线。', 'The data has no last-hit timeline.'],
    'farm.creeps_title': ['敌方兵线小兵（含攻城车）', 'Enemy lane creeps (siege included)'],
    'farm.no_breakdown': ['这里只有英雄击杀数，没有补刀对象的明细（小兵、野怪、建筑等）。', 'Only hero kills are known here, not what the last hits were (creeps, neutrals, buildings).'],

    // --- Benchmarks ---
    'benchmarks.hint': ['上行是本场数值，下行是百分位：超过了多少同英雄的近期对局（OpenDota 统计）。', 'Top line: the value in this match. Bottom line: the percentile among recent games on the same hero (OpenDota).'],

    // --- Objectives ---
    'objectives.map': ['建筑地图', 'Building map'],
    'objectives.no_events': ['数据中没有目标事件。', 'The data has no objective events.'],
    'objectives.map_hint': ['比赛（或录像）结束时双方建筑的状态；悬停可查看被摧毁的时间。', 'Buildings at the end of the game (or of the recording); hover one for the time it fell.'],
    'objectives.towers_left': ['剩余防御塔', 'Towers standing'],
    'objectives.barracks_left': ['剩余兵营', 'Barracks standing'],
    'objectives.map_label': ['建筑状态地图', 'Building status map'],
    'objectives.destroyed': ['已被摧毁', 'Destroyed'],
    'objectives.standing': ['仍然存在', 'Standing'],
    'objectives.destroyed_at': ['{time} 被摧毁', 'Destroyed at {time}'],
    'objectives.no_buildings': ['此数据来源不含建筑位置，无法绘制建筑地图；被摧毁的建筑见事件时间线。', 'This data source has no building positions, so the map cannot be drawn; the event timeline lists the buildings that fell.'],
    'objectives.no_runes': ['数据中没有神符拾取记录。', 'The data has no rune pickups.'],
    // Predicates of the timeline: the subject (hero or team) is shown before them
    'obj.destroyed': ['摧毁了{target}', 'destroyed the {target}'],
    'obj.denied': ['反补了{target}', 'denied the {target}'],
    'obj.tower_destroyed': ['摧毁了一座防御塔', 'destroyed a tower'],
    'obj.tower_denied': ['反补了一座防御塔', 'denied a tower'],
    'obj.barracks_destroyed': ['摧毁了一座兵营', 'destroyed a barracks'],
    'obj.first_blood': ['拿下一血', 'drew first blood'],
    'obj.first_blood_on': ['拿下一血，击杀了', 'drew first blood on'],
    'obj.roshan': ['击杀了肉山', 'killed Roshan'],
    'obj.aegis': ['拾取了不朽之守护', 'picked up the Aegis'],
    'obj.aegis_stolen': ['抢走了不朽之守护', 'stole the Aegis'],
    'obj.aegis_denied': ['摧毁了不朽之守护', 'denied the Aegis'],
    'obj.courier': ['击杀了{team}的信使', 'killed the {team} courier'],
    'obj.courier_plain': ['击杀了一只信使', 'killed a courier'],
    'obj.bounty': ['+{gold} 金钱', '+{gold} gold'],
    'obj.tormentor': ['击杀了痛苦魔方', 'killed a Tormentor'],
    // Buildings and other non-hero units
    'building.tower': ['{team}{lane}{tier}塔', '{team} {lane} tier {tier} tower'],
    'building.tower4': ['{team}四塔', '{team} tier 4 tower'],
    'building.barracks_melee': ['{team}{lane}近战兵营', '{team} {lane} melee barracks'],
    'building.barracks_ranged': ['{team}{lane}远程兵营', '{team} {lane} ranged barracks'],
    'building.ancient': ['{team}遗迹', '{team} Ancient'],
    'building.other': ['{team}建筑', '{team} building'],
    'unit.lane_creep': ['兵线小兵', 'Lane creeps'],
    'unit.lane_creep_of': ['{team}小兵', '{team} lane creeps'],
    'unit.neutral': ['野怪', 'Neutral creeps'],
    'unit.roshan': ['肉山', 'Roshan'],
    'unit.tormentor': ['痛苦魔方', 'Tormentor'],
    'unit.fountain': ['泉水', 'Fountain'],
    'unit.fountain_of': ['{team}泉水', '{team} fountain'],
    'unit.courier': ['信使', 'Courier'],
    'unit.courier_of': ['{team}信使', '{team} courier'],

    // --- Chat ---
    'chat.show_wheel': ['显示聊天轮盘消息（{n} 条）', 'Show chat-wheel lines ({n})'],
    'chat.wheel': ['聊天轮盘', 'Chat wheel'],
    'chat.wheel_n': ['聊天轮盘 #{n}', 'Chat wheel #{n}'],
    'chat.tag.phrase': ['轮盘', 'Wheel'],
    'chat.tag.voice': ['语音', 'Voice line'],
    'chat.tag.spray': ['喷漆', 'Spray'],
    'chat.title.phrase': ['聊天轮盘的预设短语', 'A preset phrase of the chat wheel'],
    'chat.title.voice': ['聊天轮盘的语音：游戏里会播放出来，这里是它的文字', 'A voice line of the chat wheel: played as sound in the game; this is its text'],
    'chat.title.spray': ['聊天轮盘的喷漆：一个图案，没有文字', 'A spray of the chat wheel: a picture, no text'],
    'chat.title.unknown': ['聊天轮盘的预设短语或音效：数据中只有它的编号，没有对应的文字', 'A preset chat-wheel phrase or sound: the data has its number only, no text for it'],
    'chat.unknown_sender': ['未知发送者', 'Unknown sender'],
    'chat.only_wheel': ['这场比赛没有人打字，只有聊天轮盘消息（已隐藏）。', 'Nobody typed in this match: it only has chat-wheel lines (hidden).'],
    // --- AI coach tab ---
    'coach.title': ['AI 教练', 'AI coach'],
    'coach.unavailable.title': ['这份录像无法进行 AI 教练分析', 'No AI coach review for this replay'],
    'coach.picker.title': ['选择要复盘的玩家', 'Pick a player to review'],
    'coach.picker.hint': ['每位玩家一份报告：综合评分、分阶段复盘、每次阵亡的原因和关键时刻。', 'One report per player: overall grade, phase reviews, the cause of every death and the key moments.'],
    'coach.picker.pick_title': ['查看 {hero} 的教练报告', 'Open the coach report for {hero}'],
    'coach.picker.kda_title': ['击杀 / 死亡 / 助攻', 'Kills / deaths / assists'],
    'coach.picker.unparsed': ['OpenDota 尚未解析这场比赛的录像，报告只能依据计分板数据，内容会比较简略。', 'OpenDota has not parsed the replay of this match yet: a report can only use the scoreboard and will be brief.'],
    'coach.mode.rules': ['本服务器未配置大模型：报告由规则引擎结合知识库生成，几秒内完成。', 'No LLM is configured on this server: reports come from the rule engine and the knowledge base, within seconds.'],
    'coach.mode.llm': ['报告由大模型结合知识库撰写，通常需要 1–2 分钟。', 'Reports are written by an LLM using the knowledge base and usually take 1–2 minutes.'],
    'coach.mode.quota': ['今日剩余额度：{n} 份。', "Left in today's quota: {n}."],
    'coach.idle.title': ['还没有选择玩家', 'No player selected yet'],
    'coach.idle.text': ['在上方选择一位玩家，生成他这一局的教练报告。', 'Pick a player above to get the coach report for their game.'],
    'coach.cancelled_title': ['已停止等待', 'Stopped waiting'],
    'coach.cancelled': ['已停止等待这份报告。服务器会把已经开始的分析做完并保存，再次选择该玩家即可查看。', 'Stopped waiting for this report. The server finishes the analysis it has started and keeps it: pick the player again to see it.'],
    'coach.loading.title': ['正在生成 {hero} 的教练报告', 'Writing the coach report for {hero}'],
    'coach.loading.fallback': ['进度连接已中断，正在等待服务器返回报告…', 'The progress stream was interrupted; waiting for the server to return the report…'],
    'coach.loading.hint_llm': ['大模型会逐个评估决策点，通常需要 1–2 分钟。可以先看其他页面，完成后会有提示。', 'The model reviews the decisions one by one, which usually takes 1–2 minutes. You can look at other tabs meanwhile: a notice tells you when it is done.'],
    'coach.loading.hint_rules': ['规则分析通常几秒内完成。', 'A rule-based review usually takes a few seconds.'],
    'coach.toast.ready': ['{hero} 的教练报告已生成，可在「AI 教练」页查看。', 'The coach report for {hero} is ready on the AI coach tab.'],
    'coach.error.title': ['未能获取 {hero} 的教练报告', 'Could not get the coach report for {hero}'],
    'coach.error.kept': ['下方仍是之前的报告。', 'The report below is the previous one.'],
    'coach.error.kept_lang': ['下方仍是之前的报告（{lang}版）。', 'The report below is the previous one, written in {lang}.'],
    'coach.lang.mismatch': ['这份报告是用 {lang} 写的。', 'This report is written in {lang}.'],
    'coach.lang.reload': ['获取当前界面语言的版本', 'Load it in the page language'],
    'coach.grade.overall': ['综合评分', 'Overall'],
    'coach.grade.title': ['综合评分：S 最好，D 最差', 'Overall grade: S is best, D is worst'],
    'coach.grade.phase_title': ['这一阶段的评分（S 最好，D 最差）', 'Grade of this phase (S best, D worst)'],
    'coach.grade.eval_title': ['这次决策的评分（S 最好，D 最差）', 'Grade of this decision (S best, D worst)'],
    'coach.position.title': ['位置判定依据：{method}；置信度：{confidence}', 'Position derived from {method}; confidence: {confidence}'],
    'coach.badge.llm': ['AI 模型分析', 'AI model'],
    'coach.badge.llm_title': ['评价文字由大模型撰写，撰写时参考了从知识库检索到的条目。', 'The review text was written by an LLM that was given the entries retrieved from the knowledge base.'],
    'coach.badge.rules': ['规则分析', 'Rule-based'],
    'coach.badge.rules_title': ['这份报告由规则引擎生成，没有使用大模型；每条评价下列出的知识条目是按相关度检索到的。', 'This report comes from the rule engine, without an LLM; the knowledge entries under each evaluation were retrieved by relevance.'],
    'coach.badge.degraded': ['部分降级', 'Degraded'],
    'coach.badge.partial': ['录像不完整', 'Incomplete replay'],
    'coach.badge.partial_title': ['录像在比赛结束前中断，报告只覆盖已录制的部分。', 'The recording stops before the game ended: the report covers the recorded part only.'],
    'coach.badge.no_benchmarks': ['{mode}：不做基准对比', '{mode}: no benchmarks'],
    'coach.badge.version': ['报告格式 v{n}', 'Report format v{n}'],
    'coach.badge.version_title': ['教练报告数据格式的版本', 'Schema version of the coach report'],
    'coach.tools.json': ['原始 JSON', 'Raw JSON'],
    'coach.tools.json_title': ['在新标签页打开这份报告的 JSON 数据', 'Open the JSON data of this report in a new tab'],
    'coach.tools.print': ['打印 / 存为 PDF', 'Print / save as PDF'],
    'coach.tools.print_title': ['打印这份报告（会展开全部知识来源）', 'Print this report (with every source list expanded)'],
    'coach.degraded.title': ['部分 AI 评价未能生成', 'Some AI evaluations are missing'],
    'coach.degraded.text': ['大模型没有对部分决策点或阶段给出回答（超时或出错），这些内容改用了规则分析，并标有「降级」。', 'The LLM gave no answer for some decisions or phases (a timeout or an error). Rule-based text is shown for them, marked "Fallback".'],
    'coach.degraded.hint': ['重新生成会让大模型把整份报告再写一遍，需要重新等待，并计入使用额度。', 'Generating again has the LLM write the whole report once more: it takes as long as the first time and counts toward the usage limits.'],
    'coach.degraded.wait': ['服务器在 {time}后才接受重新生成。', 'The server accepts a new run in {time}.'],
    'coach.degraded.retry': ['重新生成', 'Generate again'],
    'coach.degraded.still': ['这份报告仍有降级内容，{time}后可以再次重新生成。', 'The report still has fallback parts. It can be generated again in {time}.'],
    'coach.degraded.still_now': ['这份报告仍有降级内容，可以再次重新生成。', 'The report still has fallback parts. It can be generated again.'],
    'coach.mark.llm': ['AI', 'AI'],
    'coach.mark.llm_title': ['这一条由大模型撰写', 'Written by the LLM'],
    'coach.mark.rules': ['规则', 'Rules'],
    'coach.mark.rules_title': ['这一条由规则引擎生成', 'Produced by the rule engine'],
    'coach.mark.degraded': ['降级', 'Fallback'],
    'coach.mark.degraded_title': ['大模型没有给出这一条的回答，这里显示的是规则分析', 'The LLM gave no answer for this one: the rule-based text is shown'],
    'coach.nav.label': ['报告章节', 'Report sections'],
    'coach.nav.areas': ['优缺点', 'Strengths and weaknesses'],
    'coach.nav.phases': ['分阶段复盘', 'Phases'],
    'coach.nav.laning': ['对线', 'Laning'],
    'coach.nav.deaths': ['阵亡复盘', 'Deaths'],
    'coach.nav.moments': ['关键时刻', 'Key moments'],
    'coach.nav.benchmarks': ['基准对比', 'Benchmarks'],
    'coach.areas.improve': ['需要改进', 'Needs improvement'],
    'coach.areas.strengths': ['表现亮点', 'Strengths'],
    'coach.areas.grade_title': ['这一方面各决策点的平均评分', 'Average grade of the decisions in this area'],
    'coach.areas.decisions.one': ['{n} 个决策点', '{n} decision'],
    'coach.areas.decisions.other': ['{n} 个决策点', '{n} decisions'],
    'coach.phases.title': ['分阶段复盘', 'Phase-by-phase review'],
    'coach.phase.range': ['{from}–{to} 分钟', '{from}–{to} min'],
    'coach.phase.from': ['{from} 分钟起', 'from {from} min'],
    'coach.phase.highlights': ['亮点', 'Highlights'],
    'coach.phase.issues': ['问题', 'Issues'],
    'coach.phase.suggestions': ['建议', 'Suggestions'],
    'coach.phase.stat.kda': ['KDA', 'KDA'],
    'coach.phase.stat.kd': ['击杀 / 死亡', 'Kills / deaths'],
    'coach.phase.stat.kd_estimated': ['击杀 / 死亡（估算）', 'Kills / deaths (est.)'],
    'coach.phase.kd_title': ['该数据来源没有按时间记录的助攻', 'This data source has no timed assists'],
    'coach.phase.kd_estimated_title': ['这场比赛没有击杀记录，数字由其他数据推算，可能偏少', 'The match has no kill log: the numbers are inferred and may be an undercount'],
    'coach.phase.stat.lh': ['补刀', 'Last hits'],
    'coach.phase.stat.gpm': ['GPM', 'GPM'],
    'coach.phase.stat.xpm': ['XPM', 'XPM'],
    'coach.phase.stat.level': ['阶段末等级', 'Level at end'],
    'coach.phase.stat.fights': ['团战胜 / 参与', 'Fights won'],
    'coach.phase.fights_title': ['该玩家在这一阶段参与的团战中，己方获胜的场数', 'Of the teamfights the player joined in this phase, how many the team won'],
    'coach.phase.stat.net_worth': ['终局财产', 'Net worth'],
    'coach.phase.stat.team_gold': ['团队经济差', 'Team gold lead'],
    'coach.phase.team_gold_title': ['阶段结束时，己方总经济相对对方的差值', "The team's gold lead over the enemy when the phase ended"],
    'coach.laning.title': ['对线复盘（10 分钟）', 'Laning review (first 10 minutes)'],
    'coach.laning.cs': ['10 分钟补刀', 'Last hits at 10:00'],
    'coach.laning.denies': ['10 分钟反补', 'Denies at 10:00'],
    'coach.laning.lane_cs': ['线上小兵补刀', 'Lane creeps'],
    'coach.laning.lane_cs_title': ['前 10 分钟补到的线上小兵（不含野怪）及其占全部线上小兵的比例', 'Lane creeps last-hit in the first 10 minutes (neutrals not counted), and their share of all lane creeps'],
    'coach.laning.result': ['对线结果', 'Lane result'],
    'coach.laning.gold': ['全路经济差', 'Lane gold difference'],
    'coach.laning.xp': ['全路经验差', 'Lane XP difference'],
    'coach.laning.diff_title': ['10 分钟时，本路己方英雄合计相对对手的差值', 'At 10:00: the heroes of this lane together, against their opponents'],
    'coach.laning.trades': ['对线期事件', 'Laning events'],
    'coach.laning.none': ['没有对线复盘：数据中没有 10 分钟时的对线数据（比赛或录像不足 10 分钟，或数据来源不含分时数据）。', 'No laning review: the data has no laning numbers at 10:00 (the game or the recording is shorter, or the source has no timelines).'],
    'coach.sources.title': ['评价所依据的知识', 'Knowledge behind the evaluations'],
    'coach.sources.explainer': ['教练在评价每个决策点之前，会先从知识库中检索与当时情境最相关的条目（英雄与对位要点、教练原则、统计数据等）。每条评价下方的「知识来源」按相关度列出了这些条目，点开任意一条即可阅读原文，方便核对结论的依据。', 'Before judging a decision, the coach retrieves the knowledge-base entries that fit the situation best (hero and matchup notes, coaching principles, statistics). "Knowledge sources" under each evaluation lists them by relevance; open any entry to read its text and check what a verdict rests on.'],
    'coach.sources.note_cited': ['标有「模型引用」的条目是大模型在结论中实际用到的；其余条目被检索出来并交给了模型，但没有被引用。', 'Entries marked "Cited" are the ones the model actually used for its verdict; the others were retrieved and given to the model, but not cited.'],
    'coach.sources.note_rules': ['规则分析不引用条目：这里列出的是按相关度检索到的条目，可作为延伸阅读。', 'A rule-based review cites nothing: the entries listed were retrieved by relevance and serve as background reading.'],
    'coach.sources.total.one': ['全报告共 {n} 条', '{n} entry in this report'],
    'coach.sources.total.other': ['全报告共 {n} 条', '{n} entries in this report'],
    'coach.sources.count.one': ['知识来源 · {n} 条', 'Knowledge sources · {n}'],
    'coach.sources.count.other': ['知识来源 · {n} 条', 'Knowledge sources · {n}'],
    'coach.sources.cited_count.one': ['模型引用 {n} 条', '{n} cited'],
    'coach.sources.cited_count.other': ['模型引用 {n} 条', '{n} cited'],
    'coach.sources.cited': ['模型引用', 'Cited'],
    'coach.sources.retrieved': ['检索到', 'Retrieved'],
    'coach.sources.cited_title': ['大模型在这条评价中引用了该条目', 'The model cited this entry in its verdict'],
    'coach.sources.retrieved_title': ['该条目被检索到，但没有被引用', 'This entry was retrieved, but not cited'],
    'coach.sources.patch_title': ['该条目对应的游戏版本', 'Game version this entry belongs to'],
    'coach.sources.expand_all': ['展开全部来源', 'Expand all sources'],
    'coach.sources.collapse_all': ['收起全部来源', 'Collapse all sources'],
    'coach.sources.entry_open': ['阅读这条资料的原文', 'Read this entry'],
    'coach.sources.entry_close': ['收起原文', 'Hide the text'],
    'coach.sources.entry_loading': ['正在读取…', 'Loading…'],
    'coach.sources.entry_note': ['知识库中的原文，也就是检索后交给教练的内容（编辑整理的英雄和物品条目为英文）。', 'The entry as the knowledge base holds it: this is the text retrieval hands to the coach.'],
    'coach.sources.entry_retry': ['重试', 'Retry'],
    'coach.knowledge.chunks.one': ['知识库 {n} 条', 'knowledge base of {n} entry'],
    'coach.knowledge.chunks.other': ['知识库 {n} 条', 'knowledge base of {n} entries'],
    'coach.knowledge.vectors.one': ['{n} 个向量', '{n} vector'],
    'coach.knowledge.vectors.other': ['{n} 个向量', '{n} vectors'],
    'coach.knowledge.embedder': ['向量模型 {name}', 'embedding model {name}'],
    'coach.deaths.title': ['阵亡复盘', 'Death analyses'],
    'coach.deaths.hint': ['按时间顺序列出每次阵亡。评分针对阵亡前的决策：S 最好，D 最差。', 'Every death in order of time. The grade rates the decisions before the death: S is best, D is worst.'],
    'coach.moments.title': ['关键时刻', 'Key moments'],
    'coach.moments.hint': ['对线换血、漏刀、位置过深以及打得好的时刻，按时间顺序排列。', 'Lane trades, missed last hits, overextensions and good plays, in order of time.'],
    'coach.time.title': ['游戏时间', 'Game time'],
    'coach.time.before_horn': ['游戏时间（比赛开始的号角响起之前）', 'Game time (before the horn)'],
    'coach.time.unknown': ['时间未知', 'Time unknown'],
    'coach.time.unknown_title': ['数据中没有这一时刻的时间', 'The data has no timestamp for this moment'],
    'coach.eval.killer': ['击杀者', 'Killed by'],
    'coach.eval.assists_title': ['助攻的英雄', 'Assisting heroes'],
    'coach.eval.mistake': ['问题', 'Mistake'],
    'coach.eval.better': ['更好的打法', 'Better play'],
    'coach.eval.keep': ['建议', 'Advice'],
    'coach.eval.lesson': ['要点', 'Lesson'],
    'coach.eval.severity_title': ['这一决策对局势的影响程度', 'How much this decision mattered'],
    'coach.chain.title': ['阵亡因果链', 'What led to the death'],
    'coach.chain.root': ['根本原因', 'Root cause'],
    'coach.chain.before': ['{time}前', '{time} earlier'],
    'coach.bench.title': ['基准对比', 'Benchmarks'],
    'coach.bench.hint': ['每项数据都与同位置的基准比较。基准的来源各不相同：职业比赛参考值、OpenDota 全分段玩家的中位数，或人工整理的参考时间，见最后一列。', 'Each number is compared with a benchmark for the same position. The benchmarks differ in origin: a pro-match reference, the OpenDota median of players of all ranks, or a hand-written guideline; see the last column.'],
    'coach.bench.metric': ['指标', 'Metric'],
    'coach.bench.value': ['本场', 'This game'],
    'coach.bench.reference': ['基准', 'Benchmark'],
    'coach.bench.rating': ['评价', 'Rating'],
    'coach.bench.source': ['基准来源', 'Benchmark source'],
    'coach.bench.overall': ['总体：{rating}', 'Overall: {rating}'],
    'coach.bench.partial_title': ['录像不完整，只对比了录像覆盖到的数据', 'Incomplete replay: only what the recording covers is compared'],
    'coach.bench.not_applicable': ['这场比赛的模式不适用基准对比。', 'Benchmarks do not apply to the game mode of this match.'],
};

/* ====================================================================
 * Labels for backend codes: I18N_CODES[domain][value] = [zh, en], read
 * with code(domain, value) (key 'code.<domain>.<value>').
 *
 * Fallbacks, so the UI never shows "undefined" or a raw snake_case code:
 *   - a value ending in digits falls back to the entry whose digits are
 *     replaced by '#', with {n} = the number ('mode_31' -> 'mode_#');
 *   - an empty value uses the entry '_none' when the domain has one;
 *   - anything else is prettified ('some_new_code' -> 'Some New Code') and,
 *     in development, reported once in the console.
 * ==================================================================== */
window.I18N_CODES = {
    team: {
        radiant: ['天辉', 'Radiant'],
        dire: ['夜魇', 'Dire'],
        neutral: ['中立生物', 'Neutrals'],
        _none: ['未知', 'Unknown'],
    },
    source: {
        clarity: ['录像解析', 'Replay parse'],
        opendota: ['OpenDota 数据', 'OpenDota data'],
        mock: ['演示数据', 'Demo data'],
        unknown: ['来源未知', 'Unknown source'],
        _none: ['来源未知', 'Unknown source'],
    },
    source_title: {
        clarity: ['数据由本站直接解析录像文件得到（Clarity 解析器）', 'Parsed from the replay file on this server (Clarity parser)'],
        opendota: ['数据来自 OpenDota API', 'Data from the OpenDota API'],
        mock: ['程序生成的示例数据', 'Generated sample data'],
        unknown: ['数据来源未知', 'The data source is unknown'],
        _none: ['数据来源未知', 'The data source is unknown'],
    },
    game_mode: {
        unknown: ['未知模式', 'Unknown mode'],
        all_pick: ['全英雄选择', 'All Pick'],
        captains_mode: ['队长模式', 'Captains Mode'],
        random_draft: ['随机征召', 'Random Draft'],
        single_draft: ['单一征召', 'Single Draft'],
        all_random: ['全英雄随机', 'All Random'],
        intro: ['新手引导', 'Intro'],
        diretide: ['夜魇暗潮', 'Diretide'],
        reverse_captains_mode: ['反队长模式', 'Reverse Captains Mode'],
        greeviling: ['贪魔节', 'The Greeviling'],
        tutorial: ['教学', 'Tutorial'],
        mid_only: ['单中模式', 'Mid Only'],
        least_played: ['生疏模式', 'Least Played'],
        limited_heroes: ['新手英雄', 'Limited Heroes'],
        compendium_matchmaking: ['互动指南匹配', 'Compendium Matchmaking'],
        custom: ['自定义游戏', 'Custom Game'],
        captains_draft: ['队长征召', 'Captains Draft'],
        balanced_draft: ['平衡征召', 'Balanced Draft'],
        ability_draft: ['技能征召', 'Ability Draft'],
        event: ['活动模式', 'Event'],
        all_random_death_match: ['全随机死亡竞赛', 'All Random Deathmatch'],
        one_v_one_mid: ['中路单挑', '1v1 Solo Mid'],
        ranked_all_pick: ['天梯全英雄选择', 'Ranked All Pick'],
        turbo: ['加速模式', 'Turbo'],
        mutation: ['突变模式', 'Mutation'],
        coaches_challenge: ['教练挑战', "Coach's Challenge"],
        bot_challenge: ['机器人挑战', 'Bot Challenge'],
        'mode_#': ['模式 {n}', 'Mode {n}'],
        _none: ['未知模式', 'Unknown mode'],
    },
    // Valve region ids (OpenDota constants/region.json)
    region: {
        1: ['美国西部', 'US West'],
        2: ['美国东部', 'US East'],
        3: ['欧洲西部', 'Europe West'],
        5: ['东南亚（新加坡）', 'SE Asia (Singapore)'],
        6: ['迪拜', 'Dubai'],
        7: ['澳大利亚', 'Australia'],
        8: ['俄罗斯（斯德哥尔摩）', 'Russia (Stockholm)'],
        9: ['欧洲东部（奥地利）', 'Europe East (Austria)'],
        10: ['巴西', 'Brazil'],
        11: ['南非', 'South Africa'],
        12: ['完美世界电信（上海）', 'Perfect World Telecom (Shanghai)'],
        13: ['完美世界联通', 'Perfect World Unicom'],
        14: ['智利', 'Chile'],
        15: ['秘鲁', 'Peru'],
        16: ['印度', 'India'],
        17: ['完美世界电信（广东）', 'Perfect World Telecom (Guangdong)'],
        18: ['完美世界电信（浙江）', 'Perfect World Telecom (Zhejiang)'],
        19: ['日本', 'Japan'],
        20: ['完美世界电信（武汉）', 'Perfect World Telecom (Wuhan)'],
        25: ['完美世界联通（天津）', 'Perfect World Unicom (Tianjin)'],
        37: ['台湾', 'Taiwan'],
        38: ['阿根廷', 'Argentina'],
        '#': ['地区 {n}', 'Region {n}'],
    },
    knowledge_mode: {
        hybrid: ['知识检索：关键词 + 向量', 'Knowledge retrieval: keywords + vectors'],
        lexical: ['知识检索：关键词', 'Knowledge retrieval: keywords'],
    },
    // hero_name / item key / ability key values that are codes, not names
    hero: {
        'hero_#': ['未知英雄 #{n}', 'Unknown hero #{n}'],
        'unknown_#': ['未知英雄', 'Unknown hero'],
        _none: ['未知英雄', 'Unknown hero'],
    },
    item: {
        'item_#': ['未知物品 #{n}', 'Unknown item #{n}'],
        _none: ['未知物品', 'Unknown item'],
    },
    ability: {
        special_bonus_attributes: ['属性加成', 'Attribute Bonus'],
        _none: ['未知技能', 'Unknown ability'],
    },
    // ================= Analysis tabs =================
    // laning.players[].lane_name (relative to the team) and lane_side (map side)
    lane: {
        safe: ['优势路', 'Safe lane'],
        mid: ['中路', 'Mid lane'],
        off: ['劣势路', 'Off lane'],
        roaming: ['游走', 'Roaming'],
        unknown: ['未知', 'Unknown'],
        _none: ['未知', 'Unknown'],
    },
    lane_side: {
        top: ['上路', 'top'],
        mid: ['中路', 'mid'],
        bottom: ['下路', 'bottom'],
    },
    // Lane part of a building name (npc_dota_*_tower1_bot)
    map_lane: {
        top: ['上路', 'top'],
        mid: ['中路', 'mid'],
        bot: ['下路', 'bottom'],
    },
    tower_tier: {
        1: ['一', '1'],
        2: ['二', '2'],
        3: ['三', '3'],
        4: ['四', '4'],
    },
    // item_builds[].item_phases keys
    item_phase: {
        starting: ['出门装', 'Starting items'],
        early: ['前期（0–10 分钟）', 'Early (0–10 min)'],
        mid: ['中期（10–25 分钟）', 'Mid (10–25 min)'],
        late: ['后期（25 分钟后）', 'Late (25 min+)'],
    },
    // DotaGame.wardState() and what each state means (tooltip)
    ward_state: {
        dewarded: ['被反', 'Dewarded'],
        denied: ['被反补', 'Denied'],
        destroyed: ['被摧毁', 'Destroyed'],
        expired: ['到时消失', 'Expired'],
        active: ['仍在', 'Standing'],
    },
    ward_state_title: {
        dewarded: ['被敌方反掉：敌方英雄或其单位摧毁了它。', 'Dewarded: an enemy hero or unit destroyed it.'],
        denied: ['被己方反补：己方英雄（可能就是放置者）自己摧毁了它，敌方因此拿不到反眼的奖励。', 'Denied: a hero of its own team (possibly the one who placed it) destroyed it, so the enemy gets no reward for it.'],
        destroyed: ['在到时之前就消失了，数据中没有记录是谁摧毁的。', 'Gone well before its time ran out; the data does not say who destroyed it.'],
        expired: ['没有被摧毁，存满持续时间后自行消失。', 'Never destroyed: it stood for its full duration and ran out.'],
        active: ['数据结束时仍然存在：比赛在它到时之前结束，或录像在此中断。', 'Still standing when the data ends: the game was over before it ran out, or the recording stops there.'],
    },
    // smokes.timeline[].result and its meaning (tooltip)
    smoke_result: {
        kill: ['击杀', 'Kill'],
        objective: ['拿下目标', 'Objective'],
        traded: ['互换', 'Traded'],
        caught: ['被抓', 'Caught'],
        no_contact: ['无接触', 'No contact'],
        unknown: ['未知', 'Unknown'],
        _none: ['未知', 'Unknown'],
    },
    smoke_result_title: {
        kill: ['开雾后取得击杀，己方无人阵亡', 'Got kills without losing anyone'],
        objective: ['没有击杀，但拿下了建筑或肉山', 'No kill, but a building or Roshan was taken'],
        traded: ['双方都有英雄阵亡', 'Heroes died on both sides'],
        caught: ['己方有人阵亡，没有取得击杀', 'Lost heroes without getting a kill'],
        no_contact: ['没有引发战斗', 'No fight came of it'],
        unknown: ['数据中没有结果', 'The data has no outcome'],
        _none: ['数据中没有结果', 'The data has no outcome'],
    },
    smoke_objective: {
        tower: ['防御塔', 'Tower'],
        barracks: ['兵营', 'Barracks'],
        ancient: ['遗迹', 'Ancient'],
        roshan: ['肉山', 'Roshan'],
    },
    // teamfights[].winner
    tf_winner: {
        radiant: ['天辉胜', 'Radiant won'],
        dire: ['夜魇胜', 'Dire won'],
        draw: ['平局', 'Draw'],
        _none: ['未知', 'Unknown'],
    },
    // match_overview.players[].multi_kills / kill_streaks keys
    multi_kill: {
        2: ['双杀', 'Double kill'],
        3: ['三杀', 'Triple kill'],
        4: ['疯狂杀戮', 'Ultra kill'],
        5: ['暴走', 'Rampage'],
        '#': ['{n} 连杀', '{n} kills in a row'],
    },
    kill_streak: {
        3: ['大杀特杀', 'Killing spree'],
        4: ['主宰比赛', 'Dominating'],
        5: ['杀人如麻', 'Mega kill'],
        6: ['无人能挡', 'Unstoppable'],
        7: ['变态杀戮', 'Wicked sick'],
        8: ['妖怪般的杀戮', 'Monster kill'],
        9: ['如同神一般', 'Godlike'],
        10: ['超越神的杀戮', 'Beyond godlike'],
        '#': ['{n} 连杀', '{n}-kill streak'],   // below 3 (never announced); above 10 see streakLabel in app.js
    },
    // farm.gold_reasons / xp_reasons keys
    gold_reason: {
        other: ['其他', 'Other'],
        death: ['死亡损失', 'Deaths'],
        buyback: ['买活', 'Buyback'],
        purchase: ['购买', 'Purchases'],
        abandon: ['放弃比赛', 'Abandon'],
        sell: ['出售物品', 'Selling items'],
        ability_cost: ['技能消耗', 'Ability cost'],
        cheat: ['作弊指令', 'Cheats'],
        selection_penalty: ['选人惩罚', 'Pick penalty'],
        passive: ['工资', 'Passive income'],
        building: ['建筑', 'Buildings'],
        hero: ['击杀英雄', 'Hero kills'],
        creep: ['兵线小兵', 'Lane creeps'],
        neutral: ['野怪', 'Neutral creeps'],
        roshan: ['肉山', 'Roshan'],
        courier: ['信使', 'Couriers'],
        bounty_rune: ['赏金神符', 'Bounty runes'],
        shared: ['共享', 'Shared'],
        ability: ['技能收入', 'Ability gold'],
        ward: ['反眼', 'Wards killed'],
        illusion: ['幻象', 'Illusions'],
        'unknown_#': ['未知来源（{n}）', 'Unknown ({n})'],
    },
    xp_reason: {
        other: ['其他', 'Other'],
        hero: ['击杀英雄', 'Hero kills'],
        creep: ['小兵与野怪', 'Creeps'],
        roshan: ['肉山', 'Roshan'],
        wisdom_rune: ['智慧圣坛', 'Shrine of Wisdom'],
        wisdom_rune_legacy: ['智慧神符', 'Wisdom runes'],   // matches before 7.38
        tome: ['知识之书', 'Tome of Knowledge'],
        outpost: ['前哨', 'Outposts'],
        shared: ['共享', 'Shared'],
        catch_up: ['追赶经验', 'Catch-up'],
        ability: ['技能', 'Abilities'],
        'unknown_#': ['未知来源（{n}）', 'Unknown ({n})'],
    },
    // objectives_analysis.runes[].runes keys
    rune: {
        bounty: ['赏金', 'Bounty'],
        water: ['圣水', 'Water'],
        wisdom: ['智慧圣坛', 'Wisdom shrine'],
        wisdom_legacy: ['智慧', 'Wisdom'],   // matches before 7.38
        double_damage: ['双倍', 'Double damage'],
        haste: ['急速', 'Haste'],
        illusion: ['幻象', 'Illusion'],
        invisibility: ['隐身', 'Invisibility'],
        regeneration: ['恢复', 'Regeneration'],
        arcane: ['奥术', 'Arcane'],
        shield: ['护盾', 'Shield'],
        'unknown_#': ['未知神符（{n}）', 'Unknown ({n})'],
    },
    // benchmarks_analysis.players[].benchmarks keys
    benchmark: {
        gold_per_min: ['GPM', 'GPM'],
        xp_per_min: ['XPM', 'XPM'],
        kills_per_min: ['击杀/分', 'Kills/min'],
        deaths_per_min: ['死亡/分', 'Deaths/min'],
        assists_per_min: ['助攻/分', 'Assists/min'],
        last_hits_per_min: ['正补/分', 'LH/min'],
        denies_per_min: ['反补/分', 'Denies/min'],
        hero_damage_per_min: ['英雄伤害/分', 'Hero dmg/min'],
        hero_healing_per_min: ['治疗/分', 'Healing/min'],
        tower_damage: ['建筑伤害', 'Tower dmg'],
    },
    benchmark_title: {
        gold_per_min: ['每分钟金钱', 'Gold per minute'],
        xp_per_min: ['每分钟经验', 'Experience per minute'],
        kills_per_min: ['每分钟击杀', 'Kills per minute'],
        deaths_per_min: ['每分钟死亡（百分位越高表示死得越多）', 'Deaths per minute (a higher percentile means more deaths)'],
        assists_per_min: ['每分钟助攻', 'Assists per minute'],
        last_hits_per_min: ['每分钟正补', 'Last hits per minute'],
        denies_per_min: ['每分钟反补', 'Denies per minute'],
        hero_damage_per_min: ['每分钟对英雄造成的伤害', 'Hero damage per minute'],
        hero_healing_per_min: ['每分钟治疗量', 'Hero healing per minute'],
        tower_damage: ['对建筑造成的总伤害', 'Total tower damage'],
    },
    // ================= AI coach report (schema version 5) =================
    // Progress stages of the event stream
    coach_stage: {
        starting: ['准备分析', 'Getting ready'],
        detect: ['识别关键决策点', 'Finding the decision points'],
        causal_chain: ['追溯每次阵亡的起因', 'Tracing what led to each death'],
        context: ['还原每个决策点的局势', 'Rebuilding the situation of each decision'],
        evaluate: ['评估每个决策点', 'Evaluating the decisions'],
        phases: ['分阶段复盘', 'Reviewing the phases'],
        summary: ['汇总报告', 'Writing the summary'],
        done: ['完成', 'Done'],
    },
    // phase_reviews[].phase
    coach_phase: {
        laning: ['对线期', 'Laning'],
        early_mid: ['前中期', 'Early-mid game'],
        mid_late: ['中后期', 'Mid-late game'],
        late: ['后期', 'Late game'],
        ultra_late: ['大后期', 'Ultra-late game'],
    },
    // death_analyses[] / key_moments[]: point_type and severity
    point_type: {
        death: ['阵亡', 'Death'],
        bad_trade: ['换血吃亏', 'Bad trade'],
        missed_cs: ['漏刀', 'Missed last hits'],
        fight_engage: ['团战', 'Teamfight'],
        overextend: ['位置过深', 'Overextended'],
        good_play: ['打得好', 'Good play'],
        _none: ['决策点', 'Decision'],
    },
    severity: {
        critical: ['严重', 'Critical'],
        major: ['重要', 'Major'],
        minor: ['轻微', 'Minor'],
        positive: ['正面', 'Positive'],
    },
    // causal_chain.steps[]: type and significance; causal_chain.root_cause_type
    chain_step: {
        vision_loss: ['失去视野', 'Vision lost'],
        enemy_missing: ['敌人消失', 'Enemy missing'],
        hidden_enemy_near: ['附近有藏着的敌人', 'Hidden enemy nearby'],
        ignore_warning: ['忽视危险信号', 'Warning ignored'],
        position_forward: ['位置靠前', 'Forward position'],
        position_unsafe: ['站位不安全', 'Unsafe position'],
        no_retreat: ['没有后撤', 'No retreat'],
        ally_absent: ['队友不在身边', 'No ally nearby'],
        allies_fallen: ['队友先阵亡', 'Allies fallen'],
        farm_under_threat: ['冒险打钱', 'Farming under threat'],
        enemy_approach: ['敌人逼近', 'Enemy approaching'],
        cc_chain: ['被连续控制', 'Chain-disabled'],
        damage_burst: ['承受爆发伤害', 'Burst damage taken'],
        rune_contest: ['争夺神符', 'Rune contest'],
        death: ['阵亡', 'Death'],
        _none: ['事件', 'Event'],
    },
    chain_significance: {
        root_cause: ['根本原因', 'Root cause'],
        contributing: ['促成因素', 'Contributing'],
        consequence: ['后续', 'Consequence'],
        final: ['阵亡', 'Death'],
    },
    chain_significance_title: {
        root_cause: ['这一步是这次阵亡的根本原因', 'The root cause of this death'],
        contributing: ['促成这次阵亡的因素', 'A factor that contributed to the death'],
        consequence: ['由前面的情况引出的后续发展', 'What followed from the steps before'],
        final: ['阵亡的时刻', 'The moment of the death'],
    },
    root_cause: {
        vision_loss: ['视野缺失', 'Lack of vision'],
        overextend: ['位置过深', 'Overextension'],
        ignore_missing: ['无视敌人消失', 'Missing enemies ignored'],
        bad_positioning: ['站位失误', 'Bad positioning'],
        greedy_farm: ['贪刷', 'Greedy farming'],
        isolated: ['孤立无援', 'Isolated'],
        teamfight_outnumbered: ['团战人数劣势', 'Outnumbered in the fight'],
        teamfight_positioning: ['团战站位', 'Teamfight positioning'],
        rune_contest_unfavorable: ['劣势争符', 'Unfavorable rune contest'],
    },
    // laning_review.lane_result (null = unknown, shown as a dash)
    lane_result: {
        won: ['优势', 'Won'],
        even: ['均势', 'Even'],
        lost: ['劣势', 'Lost'],
    },
    // improvement_areas[] / strengths[]: category_code
    coach_category: {
        map_awareness: ['地图意识', 'Map awareness'],
        laning: ['对线能力', 'Laning'],
        teamfight: ['团战参与', 'Teamfight participation'],
        resources: ['资源利用', 'Resource efficiency'],
        vision: ['视野控制', 'Vision control'],
        _none: ['其他', 'Other'],
    },
    // pro_comparison: rating codes and where a benchmark comes from.
    // Only pro_reference is professional data.
    rating: {
        excellent: ['优秀', 'Excellent'],
        good: ['良好', 'Good'],
        average: ['一般', 'Average'],
        below_average: ['偏低', 'Below average'],
        poor: ['较差', 'Poor'],
    },
    benchmark_source: {
        pro_reference: ['职业比赛参考', 'Pro reference'],
        opendota_population: ['OpenDota 全分段中位数', 'OpenDota all-rank median'],
        guideline: ['经验参考', 'Guideline'],
        _none: ['未注明', 'Not stated'],
    },
    benchmark_source_title: {
        pro_reference: ['近期职业比赛中同位置选手的平均数据', 'Averages of players in the same position in recent professional matches'],
        opendota_population: ['OpenDota 上所有分段玩家的中位数，不代表职业水准', 'The median of OpenDota players of all ranks: not a professional standard'],
        guideline: ['人工整理的出装时间参考，不是统计数据', 'A hand-written item-timing guideline, not a statistic'],
    },
    // position: number, confidence and method of the position detection
    position: {
        1: ['1号位 (carry)', 'Position 1 (Carry)'],
        2: ['2号位 (mid)', 'Position 2 (Mid)'],
        3: ['3号位 (offlaner)', 'Position 3 (Offlaner)'],
        4: ['4号位 (support)', 'Position 4 (Support)'],
        5: ['5号位 (hard support)', 'Position 5 (Hard Support)'],
        '#': ['{n}号位', 'Position {n}'],
    },
    // The same position as a short chip (picker, scoreboard)
    position_short: {
        '#': ['{n}号位', 'Pos {n}'],
    },
    confidence: {
        high: ['高', 'high'],
        medium: ['中', 'medium'],
        low: ['低', 'low'],
        _none: ['未知', 'unknown'],
    },
    confidence_note: {
        medium: ['推测', 'estimated'],
        low: ['不确定', 'uncertain'],
    },
    position_method: {
        lane_nw: ['分路与经济', 'lane and net worth'],
        archetype_nw: ['英雄定位与经济', 'hero archetype and net worth'],
        nw_only: ['仅按经济排序', 'net worth only'],
        _none: ['未知', 'unknown'],
    },
    // knowledge.mode of a report (the start page uses knowledge_mode above)
    knowledge_mode_short: {
        hybrid: ['混合检索', 'Hybrid retrieval'],
        lexical: ['关键词检索', 'Keyword retrieval'],
    },
    knowledge_mode_title: {
        hybrid: ['知识条目同时按关键词匹配和语义向量相似度检索，两路结果融合后排序。', 'Knowledge entries are found by keyword match and by semantic vector similarity; the two rankings are fused.'],
        lexical: ['知识条目只按关键词匹配检索（服务器上没有可用的向量索引）。', 'Knowledge entries are found by keyword match only (no vector index is available on the server).'],
    },
    // sources[].source_type of an evaluation
    source_type: {
        builtin: ['教练原则', 'Coaching principle'],
        editorial: ['编辑整理', 'Curated note'],
        operator: ['运营补充', 'Operator note'],
        opendota_stats: ['OpenDota 统计', 'OpenDota statistics'],
        match_mining: ['职业比赛统计', 'Pro-match statistics'],
        llm_distilled: ['AI 归纳', 'AI summary'],
        video: ['视频经验', 'Video tip'],
        _none: ['其他来源', 'Other source'],
    },
    source_type_title: {
        builtin: ['内置的通用教练原则（视野、站位、团战等）', 'General coaching principles built into the coach (vision, positioning, teamfights and so on)'],
        editorial: ['随项目维护的英雄、对位和物品要点，取自游戏数据并经人工整理', 'Hero, matchup and item notes kept with the project: taken from the game data and curated by hand'],
        operator: ['本服务器的运营者添加的知识文件', 'Knowledge files added by the operator of this server'],
        opendota_stats: ['自动从 OpenDota 公开统计中学到的数据（全分段玩家）', 'Learned automatically from the public OpenDota statistics (players of all ranks)'],
        match_mining: ['自动从近期职业比赛中统计出的规律', 'Patterns mined automatically from recent professional matches'],
        llm_distilled: ['大模型根据职业比赛统计归纳出的文字，未经人工校对', 'Text an LLM distilled from pro-match statistics; not reviewed by a person'],
        video: ['整理自社区视频的字幕，未经人工校对，可能对应较早的版本', 'Taken from the transcripts of community videos; not reviewed, and possibly from an older patch'],
    },
};

/* ====================================================================
 * Runtime: window.I18N
 *   t(locale, key, params)        string for key, {name} placeholders
 *   tn(locale, key, n, params)    plural form: '<key>.one' / '<key>.other'
 *   has(locale, key)              whether the key exists
 *   code(locale, domain, value, params)   label for a backend code
 *   prettify(code)                'some_code' -> 'Some Code'
 *   detect() / store(locale)      language choice (localStorage)
 * In development (localhost, or localStorage dota_analyzer_debug = 1) a
 * missing key or an unlabelled code is reported once with console.warn,
 * and so is a legacy key that exists in only one language.
 * ==================================================================== */
(function () {
    'use strict';
    const MSG = window.I18N_MESSAGES;
    const LOCALES = ['zh', 'en'];
    const STORAGE_KEY = 'dota_analyzer_locale';

    Object.entries(window.I18N_PAIRS).forEach(([key, pair]) => {
        MSG.zh[key] = pair[0];
        MSG.en[key] = pair[1];
    });
    Object.entries(window.I18N_CODES).forEach(([domain, table]) => {
        Object.entries(table).forEach(([value, pair]) => {
            MSG.zh[`code.${domain}.${value}`] = pair[0];
            MSG.en[`code.${domain}.${value}`] = pair[1];
        });
    });

    function readStore(key) {
        try { return window.localStorage.getItem(key); } catch (e) { return null; }
    }

    const DEV = ['localhost', '127.0.0.1', '[::1]', ''].includes(window.location.hostname)
        || readStore('dota_analyzer_debug') === '1';
    const missing = new Set();

    function report(kind, id) {
        const tag = kind + ':' + id;
        if (missing.has(tag)) return;
        missing.add(tag);
        if (DEV) console.warn(`[i18n] ${kind}: ${id}`);
    }

    function prettify(code) {
        const text = String(code == null ? '' : code)
            .replace(/^npc_dota_hero_/, '')
            .replace(/^npc_dota_/, '')
            .replace(/^special_bonus_(unique_)?/, '')
            .replace(/[_.]+/g, ' ')
            .trim();
        return text.replace(/\b[a-z]/g, (c) => c.toUpperCase());
    }

    function format(text, params) {
        if (!params) return text;
        return text.replace(/\{(\w+)\}/g, (whole, name) => (params[name] == null ? whole : String(params[name])));
    }

    function has(locale, key) {
        return Object.prototype.hasOwnProperty.call(MSG[locale] || MSG.zh, key);
    }

    function t(locale, key, params) {
        const lang = MSG[locale] ? locale : 'zh';
        let text = has(lang, key) ? MSG[lang][key] : undefined;
        if (text === undefined) {
            report('missing key', `${key} (${lang})`);
            const other = lang === 'zh' ? 'en' : 'zh';
            text = has(other, key) ? MSG[other][key] : prettify(String(key).split('.').pop());
        }
        return format(text, params);
    }

    function tn(locale, key, n, params) {
        const form = Number(n) === 1 && has(locale, key + '.one') ? '.one' : '.other';
        return t(locale, key + form, Object.assign({ n }, params));
    }

    function code(locale, domain, value, params) {
        const base = `code.${domain}.`;
        if (value == null || value === '') {
            return has(locale, base + '_none') ? t(locale, base + '_none', params) : '-';
        }
        const raw = String(value);
        if (has(locale, base + raw)) return t(locale, base + raw, params);
        const numbered = raw.match(/^(.*?)(\d+)$/);
        if (numbered && has(locale, base + numbered[1] + '#')) {
            return t(locale, base + numbered[1] + '#', Object.assign({ n: numbered[2] }, params));
        }
        report('unlabelled code', `${domain}.${raw}`);
        return prettify(raw);
    }

    function detect() {
        const stored = readStore(STORAGE_KEY);
        if (LOCALES.includes(stored)) return stored;
        const nav = String(window.navigator.language || '').toLowerCase();
        return nav.startsWith('zh') ? 'zh' : 'en';
    }

    function store(locale) {
        try { window.localStorage.setItem(STORAGE_KEY, locale); } catch (e) { /* private mode */ }
    }

    if (DEV) {
        const onlyIn = (a, b) => Object.keys(MSG[a]).filter((k) => !has(b, k));
        const zhOnly = onlyIn('zh', 'en');
        const enOnly = onlyIn('en', 'zh');
        if (zhOnly.length) console.warn('[i18n] keys without an English text: ' + zhOnly.join(', '));
        if (enOnly.length) console.warn('[i18n] keys without a Chinese text: ' + enOnly.join(', '));
    }

    window.I18N = { LOCALES, DEV, missing, t, tn, has, code, prettify, format, detect, store };
})();
