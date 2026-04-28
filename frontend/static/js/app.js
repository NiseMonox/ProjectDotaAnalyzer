const { createApp, ref, computed, watch, onMounted, nextTick } = Vue;

const CDN = 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react';

createApp({
    setup() {
        // ===== i18n =====
        const locale = ref(localStorage.getItem('dota_analyzer_locale') || 'zh');
        function t(key) {
            const msgs = window.I18N_MESSAGES || {};
            const langMsgs = msgs[locale.value] || msgs['zh'] || {};
            return langMsgs[key] || key;
        }
        function onLocaleChange() {
            localStorage.setItem('dota_analyzer_locale', locale.value);
            document.title = t('app_title');
        }

        // Map backend-provided Chinese lane_name strings to i18n keys so they
        // render in the active locale. Falls back to lane_role/lane ids when
        // the string doesn't match a known label.
        const LANE_NAME_I18N = {
            '安全路': 'lane_safe',
            '中路': 'lane_mid',
            '劣势路': 'lane_off',
            // Backend used to emit "野区" for lane_role=4 (OpenDota's "Jungle"
            // bucket); now folded into "游走" since modern Dota has no
            // dedicated junglers. Keep the legacy mapping for cached payloads.
            '野区': 'lane_roaming',
            '游走': 'lane_roaming',
            '未知': 'lane_unknown',
        };
        function translateLaneName(player) {
            if (!player) return '';
            if (player.is_roaming) return t('lane_roaming');
            const raw = player.lane_name;
            if (raw && LANE_NAME_I18N[raw]) return t(LANE_NAME_I18N[raw]);
            // Derive from numeric lane id when lane_name is missing/unknown.
            // lane=4 (OpenDota "Jungle") collapsed into "Roaming" — see
            // LANE_NAME_I18N comment above.
            const laneMap = { 1: 'lane_safe', 2: 'lane_mid', 3: 'lane_off', 4: 'lane_roaming' };
            const key = laneMap[player.lane];
            if (key) return t(key);
            return t('lane_unknown');
        }

        // State
        const analysisData = ref(null);
        const matchKey = ref('');
        const parseSource = ref('');
        const uploading = ref(false);
        const uploadError = ref('');
        const activeTab = ref('overview');
        const showAbilityBuilds = ref(false);
        const selectedTeamfight = ref(1);

        // Hero/item lookup data (loaded from static JSON)
        const heroData = ref(null);
        const itemData = ref(null);

        const wardFilters = ref({
            observers: true, sentries: true,
            radiant: true, dire: true,
        });

        // Ward timeline mode state
        const wardTimelineEnabled = ref(false);
        const wardTimelineTime = ref(0);
        const wardShowVision = ref(false);
        // Vision radius in map percentage units
        // OpenDota: getWardSize returns DIAMETER = mapSize * range / 12000
        // Our ECharts needs RADIUS in 0-100 map%: range / 12000 / 2 * 100
        const OBS_VISION_RADIUS = 1600 / 12000 / 2 * 100;   // ~6.67 map units
        const SENTRY_VISION_RADIUS = 1000 / 12000 / 2 * 100; // ~4.17 map units

        // Smoke map state
        const smokeTimelineTime = ref(0);
        const smokeTimelineEnabled = ref(false);
        const smokeSelected = ref([]); // empty = show all; otherwise indices into timeline

        // Teamfight replay state
        const teamfightReplayTime = ref(0);
        const teamfightPlaying = ref(false);
        const teamfightPlaySpeed = ref(1); // 0.25, 0.5, 1, 2
        let teamfightPlayInterval = null;


        // AI Coach state
        const aiCoachData = ref(null);
        const aiCoachLoading = ref(false);
        const aiCoachError = ref('');
        const aiCoachPlayerSlot = ref(null);
        // Progress state for the SSE stream of /ai-coach-stream/{slot}
        const aiCoachProgress = ref({ stage: '', current: 0, total: 0, percent: 0 });
        let aiCoachEventSource = null;

        // Map a backend "stage" string to a localized loading label.
        function aiCoachStageLabel(stage) {
            if (!stage) return t('ai_coach_loading_detail');
            return t('coach_stage_' + stage) || t('ai_coach_loading_detail');
        }

        // Roll-up percent across all stages so the bar moves smoothly even when
        // an individual stage reports current=0/total=1.
        const STAGE_WEIGHTS = {
            starting: [0, 2],
            detect: [2, 4],
            causal_chain: [4, 8],
            context: [8, 12],
            evaluate: [12, 90],     // the slow LLM batch dominates the wait
            comparison: [90, 95],
            summary: [95, 99],
            done: [99, 100],
        };
        function computeOverallPercent(stage, current, total) {
            const span = STAGE_WEIGHTS[stage] || [0, 100];
            const [lo, hi] = span;
            const frac = total > 0 ? Math.min(1, current / total) : 0;
            return Math.round(lo + (hi - lo) * frac);
        }

        // New tab state
        const selectedLaningPlayer = ref(0);
        const combatMode = ref('kills');
        const farmMode = ref('gold');
        const objMode = ref('timeline');

        // New chart refs
        const laningHeatmap = ref(null);
        const laningCsTimeline = ref(null);
        const laningHpTimeline = ref(null);
        const farmReasonChart = ref(null);
        const farmLhChart = ref(null);

        // New chart instances
        let laningHeatmapInstance = null;
        let laningCsTimelineInstance = null;
        let laningHpTimelineInstance = null;
        let farmReasonChartInstance = null;
        let farmLhChartInstance = null;

        // Refs for chart containers
        const wardMap = ref(null);
        const smokeMapRef = ref(null);
        const smokeChart = ref(null);
        const teamfightMap = ref(null);
        const economyChart = ref(null);

        // Chart instances
        let wardChartInstance = null;
        let smokeMapInstance = null;
        let economyChartInstance = null;
        let smokeChartInstance = null;
        let teamfightChartInstance = null;

        const tabs = computed(() => [
            { id: 'overview', label: t('tab_overview') },
            { id: 'benchmarks', label: t('tab_benchmarks') },
            { id: 'laning', label: t('tab_laning') },
            { id: 'combat', label: t('tab_combat') },
            { id: 'farm', label: t('tab_farm') },
            { id: 'items', label: t('tab_items') },
            { id: 'purchases', label: t('tab_purchases') },
            { id: 'economy', label: t('tab_economy') },
            { id: 'skills', label: t('tab_skills') },
            { id: 'objectives', label: t('tab_objectives') },
            { id: 'wards', label: t('tab_wards') },
            { id: 'smokes', label: t('tab_smokes') },
            { id: 'teamfights', label: t('tab_teamfights') },
            { id: 'chat', label: t('tab_chat') },
            { id: 'ai_coach', label: t('tab_ai_coach') },
        ]);

        const phaseLabels = computed(() => ({
            starting: t('phase_starting'),
            early: t('phase_early'),
            mid: t('phase_mid'),
            late: t('phase_late'),
        }));

        // Placeholder SVG data URIs for missing icons
        const PLACEHOLDER_HERO = 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32"><rect width="32" height="32" rx="4" fill="#2d3748"/><text x="16" y="20" text-anchor="middle" fill="#4a5568" font-size="14" font-family="sans-serif">?</text></svg>');
        const PLACEHOLDER_ITEM = 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="36" height="26" viewBox="0 0 36 26"><rect width="36" height="26" rx="3" fill="#2d3748"/></svg>');

        // --- CDN Icon URL Helpers ---
        function heroIcon(slug) {
            if (!slug) return PLACEHOLDER_HERO;
            return `${CDN}/heroes/icons/${slug}.png`;
        }

        function heroImg(slug) {
            if (!slug) return PLACEHOLDER_HERO;
            return `${CDN}/heroes/${slug}.png`;
        }

        function itemIcon(key) {
            if (!key) return PLACEHOLDER_ITEM;
            if (key.startsWith('recipe_')) return `${CDN}/items/recipe.png`;
            return `${CDN}/items/${key}.png`;
        }

        function abilityIcon(key) {
            if (!key || key.startsWith('talent_')) return '';
            return `${CDN}/abilities/${key}.png`;
        }

        // Lookup hero slug by localized name (for data that only has hero_name)
        function heroSlugByName(name) {
            if (!heroData.value) return '';
            const id = heroData.value.name_to_id[name];
            if (id && heroData.value.by_id[String(id)]) {
                return heroData.value.by_id[String(id)].slug;
            }
            return '';
        }

        function heroIconByName(name) {
            const slug = heroSlugByName(name);
            return slug ? heroIcon(slug) : PLACEHOLDER_HERO;
        }

        // --- Computed ---
        const overview = computed(() => analysisData.value?.match_overview || {});

        const radiantPlayers = computed(() =>
            (overview.value.players || []).filter(p => p.team === 'radiant')
        );

        const direPlayers = computed(() =>
            (overview.value.players || []).filter(p => p.team === 'dire')
        );

        const smokeData = computed(() =>
            analysisData.value?.smokes || { radiant: { total: 0 }, dire: { total: 0 }, timeline: [] }
        );

        const teamfightData = computed(() =>
            analysisData.value?.teamfights || []
        );

        const currentTeamfight = computed(() =>
            teamfightData.value.find(tf => tf.index === selectedTeamfight.value)
        );

        const smokeMatchDuration = computed(() =>
            analysisData.value?.match_overview?.duration || 3600
        );

        const currentTeamfightDuration = computed(() => {
            const tf = currentTeamfight.value;
            return tf ? tf.end_time - tf.start_time : 0;
        });

        // Laning data
        const laningData = computed(() => analysisData.value?.laning || { players: [] });
        const laningRadiant = computed(() => (laningData.value.players || []).filter(p => p.team === 'radiant'));
        const laningDire = computed(() => (laningData.value.players || []).filter(p => p.team === 'dire'));

        // Combat data
        const combatData = computed(() => analysisData.value?.combat || { kill_matrix: null, damage_matrix: null });
        const combatMatrix = computed(() => {
            if (combatMode.value === 'kills') return combatData.value.kill_matrix;
            return combatData.value.damage_matrix;
        });

        // Farm data
        const farmData = computed(() => analysisData.value?.farm || { unit_kills: [], gold_reasons: [], xp_reasons: [], lh_timeline: [] });

        // Purchase data
        const purchaseData = computed(() => analysisData.value?.purchases?.players || []);

        // Benchmark data
        const benchmarkPlayers = computed(() => analysisData.value?.benchmarks_analysis?.players || []);
        const benchmarkColumns = computed(() => {
            const cols = [
                { key: 'gold_per_min', label: 'GPM' },
                { key: 'xp_per_min', label: 'XPM' },
                { key: 'kills_per_min', label: t('benchmarks_kills_per_min') },
                { key: 'last_hits_per_min', label: t('benchmarks_lh_per_min') },
                { key: 'hero_damage_per_min', label: t('benchmarks_dmg_per_min') },
                { key: 'hero_healing_per_min', label: t('benchmarks_healing_per_min') },
                { key: 'tower_damage', label: t('benchmarks_tower_dmg') },
            ];
            // Only show columns that have data
            if (benchmarkPlayers.value.length === 0) return cols;
            return cols.filter(c => benchmarkPlayers.value.some(p => p.benchmarks && p.benchmarks[c.key]));
        });

        // Objectives data
        const objectivesData = computed(() => analysisData.value?.objectives_analysis || { objectives: [], runes: [] });

        // Draft data
        const draftData = computed(() => analysisData.value?.draft || { picks_bans: [], has_draft: false });

        // Chat data
        const chatMessages = computed(() => analysisData.value?.chat_analysis?.messages || []);

        // --- Game mode mapping ---
        const GAME_MODE_NAMES = {
            0: 'Unknown', 1: 'All Pick', 2: 'Captain\'s Mode', 3: 'Random Draft',
            4: 'Single Draft', 5: 'All Random', 6: 'Intro', 7: 'Diretide',
            8: 'Reverse Captain\'s Mode', 9: 'Greeviling', 10: 'Tutorial',
            11: 'Mid Only', 12: 'Least Played', 13: 'Limited Heroes',
            14: 'Compendium', 15: 'Custom', 16: 'Captain\'s Draft',
            17: 'Balanced Draft', 18: 'Ability Draft', 19: 'Event',
            20: 'All Random Deathmatch', 21: '1v1 Mid', 22: 'All Draft',
            23: 'Turbo', 24: 'Mutation',
        };

        // --- Match Header computed ---
        const matchHeader = computed(() => {
            const ov = overview.value;
            if (!ov || !ov.match_id) return null;
            const radiantWin = ov.radiant_win;
            const dur = ov.duration || 0;
            const mins = Math.floor(dur / 60);
            const secs = dur % 60;
            // Determine game_mode: could be string name or numeric id
            let gameMode = ov.game_mode || '';
            if (typeof gameMode === 'number') {
                gameMode = GAME_MODE_NAMES[gameMode] || 'Unknown';
            }
            return {
                winner: radiantWin ? 'radiant' : 'dire',
                winnerText: radiantWin ? 'Radiant Victory' : 'Dire Victory',
                radiantScore: ov.radiant_score || 0,
                direScore: ov.dire_score || 0,
                duration: `${mins}:${String(secs).padStart(2, '0')}`,
                gameMode,
                matchId: ov.match_id,
                endedAgo: timeAgo(ov.start_time),
                region: regionName(ov.region),
            };
        });

        // --- Overview scoreboard helpers ---
        function buildOverviewRow(player) {
            const itemBuilds = analysisData.value?.item_builds || [];
            const playerBuild = itemBuilds.find(b => b.player_slot === player.player_slot);

            // Build final 6 items with purchase times
            const finalKeys = player.final_item_keys || [];
            const items = finalKeys.slice(0, 6).map(key => {
                let time_str = '';
                // Look up purchase time from item_builds phases or purchase_log
                if (playerBuild) {
                    // Search all phases for this item key
                    for (const phase of ['starting', 'early', 'mid', 'late']) {
                        const phaseItems = playerBuild.item_phases?.[phase] || [];
                        const found = phaseItems.find(i => i.item_key === key);
                        if (found) {
                            time_str = found.time_str || '';
                            break;
                        }
                    }
                }
                // Fallback: search purchase_log
                if (!time_str && player.purchase_log) {
                    const logEntry = player.purchase_log.find(e => e.key === key);
                    if (logEntry) {
                        time_str = formatGameTime(logEntry.time);
                    }
                }
                return { key, time_str };
            });

            // Pad to 6 slots
            while (items.length < 6) {
                items.push({ key: '', time_str: '' });
            }

            // Backpack items
            const backpack = (player.backpack || []).map(key => ({ key }));

            const nw = player.net_worth || 0;
            const hd = player.hero_damage || 0;
            const td = player.tower_damage || 0;
            const hh = player.hero_healing || 0;

            return {
                hero_slug: player.hero_slug || '',
                hero_name: player.hero_name || '',
                player_name: player.player_name || '',
                level: player.level || 0,
                kills: player.kills || 0,
                deaths: player.deaths || 0,
                assists: player.assists || 0,
                lh_dn: `${player.last_hits || 0}/${player.denies || 0}`,
                net_worth: nw,
                net_worth_k: formatNumber(nw),
                gpm_xpm: `${player.gold_per_min || 0}/${player.xp_per_min || 0}`,
                hero_damage: hd,
                hero_damage_k: formatNumber(hd),
                tower_damage: td,
                tower_damage_k: formatNumber(td),
                hero_healing: hh,
                hero_healing_k: formatNumber(hh),
                items,
                backpack,
                player_slot: player.player_slot,
            };
        }

        const overviewRadiant = computed(() => {
            return radiantPlayers.value.map(p => buildOverviewRow(p));
        });

        const overviewDire = computed(() => {
            return direPlayers.value.map(p => buildOverviewRow(p));
        });

        // --- Team Totals computed ---
        function computeTeamTotal(rows) {
            const totals = {
                kills: 0, deaths: 0, assists: 0,
                lh: 0, dn: 0,
                net_worth: 0,
                gpm: 0, xpm: 0,
                hd: 0, td: 0, hh: 0,
            };
            rows.forEach(r => {
                totals.kills += r.kills;
                totals.deaths += r.deaths;
                totals.assists += r.assists;
                const parts = r.lh_dn.split('/');
                totals.lh += parseInt(parts[0]) || 0;
                totals.dn += parseInt(parts[1]) || 0;
                totals.net_worth += r.net_worth;
                const gx = r.gpm_xpm.split('/');
                totals.gpm += parseInt(gx[0]) || 0;
                totals.xpm += parseInt(gx[1]) || 0;
                totals.hd += r.hero_damage;
                totals.td += r.tower_damage;
                totals.hh += r.hero_healing;
            });
            return {
                kills: totals.kills,
                deaths: totals.deaths,
                assists: totals.assists,
                lh_dn: `${totals.lh}/${totals.dn}`,
                net_worth_k: formatNumber(totals.net_worth),
                gpm_xpm: `${totals.gpm}/${totals.xpm}`,
                hd_k: formatNumber(totals.hd),
                td_k: formatNumber(totals.td),
                hh_k: formatNumber(totals.hh),
            };
        }

        const teamTotals = computed(() => {
            return {
                radiant: computeTeamTotal(overviewRadiant.value),
                dire: computeTeamTotal(overviewDire.value),
            };
        });

        // --- Data availability checks for graceful degradation ---
        const parseSourceLabel = computed(() => {
            const src = parseSource.value;
            if (src === 'clarity') return t('source_clarity_short');
            if (src === 'opendota') return t('source_opendota');
            if (src === 'mock') return t('source_mock');
            return src || '';
        });

        const hasSkillBuilds = computed(() => {
            const builds = analysisData.value?.skill_builds;
            return builds && builds.length > 0 && builds.some(p => p.skill_order && p.skill_order.length > 0);
        });

        const hasWardData = computed(() => {
            const wards = analysisData.value?.wards;
            return wards && wards.all_wards && wards.all_wards.length > 0;
        });

        const hasSmokeData = computed(() => {
            const smokes = analysisData.value?.smokes;
            return smokes && smokes.timeline && smokes.timeline.length > 0;
        });

        const hasBenchmarkData = computed(() => {
            const players = analysisData.value?.benchmarks_analysis?.players;
            return players && players.length > 0 && players.some(p => p.benchmarks && Object.keys(p.benchmarks).length > 0);
        });

        const hasLaningData = computed(() => {
            const laning = analysisData.value?.laning;
            return laning && laning.players && laning.players.length > 0;
        });

        const hasTeamfightData = computed(() => {
            const tf = analysisData.value?.teamfights;
            return tf && tf.length > 0;
        });

        const hasEconomyTimeline = computed(() => {
            return !!(analysisData.value?.gold_xp_timeline && analysisData.value.gold_xp_timeline.length > 0);
        });

        const hasItemBuilds = computed(() => {
            const builds = analysisData.value?.item_builds;
            return builds && builds.length > 0;
        });

        const hasCombatData = computed(() => {
            const combat = analysisData.value?.combat;
            return combat && (combat.kill_matrix || combat.damage_matrix);
        });

        const hasFarmData = computed(() => {
            const farm = analysisData.value?.farm;
            return farm && ((farm.unit_kills && farm.unit_kills.length > 0) || (farm.gold_reasons && farm.gold_reasons.length > 0));
        });

        const hasPurchaseData = computed(() => {
            const purchases = analysisData.value?.purchases?.players;
            return purchases && purchases.length > 0;
        });

        const hasObjectivesData = computed(() => {
            const obj = analysisData.value?.objectives_analysis;
            return obj && ((obj.objectives && obj.objectives.length > 0) || (obj.runes && obj.runes.length > 0));
        });

        const wardStats = computed(() => {
            const wards = analysisData.value?.wards || {};
            return {
                totalObs: (wards.radiant?.total_obs || 0) + (wards.dire?.total_obs || 0),
                totalSent: (wards.radiant?.total_sent || 0) + (wards.dire?.total_sent || 0),
            };
        });

        const filteredWards = computed(() => {
            const allWards = analysisData.value?.wards?.all_wards || [];
            return allWards.filter(w => {
                if (!wardFilters.value.observers && w.ward_type === 'observer') return false;
                if (!wardFilters.value.sentries && w.ward_type === 'sentry') return false;
                if (!wardFilters.value.radiant && w.team === 'radiant') return false;
                if (!wardFilters.value.dire && w.team === 'dire') return false;
                return true;
            });
        });

        const wardMatchDuration = computed(() =>
            analysisData.value?.wards?.duration || analysisData.value?.match_overview?.duration || 3600
        );

        const DEFAULT_WARD_LIFETIME = { observer: 360, sentry: 480 };

        const timelineWards = computed(() => {
            if (!wardTimelineEnabled.value) return filteredWards.value;
            const curTime = wardTimelineTime.value;
            return filteredWards.value.filter(w => {
                if (w.time > curTime) return false;
                if (w.destroyed_time != null) {
                    if (w.destroyed_time <= curTime) return false;
                } else {
                    const expiry = w.time + (DEFAULT_WARD_LIFETIME[w.ward_type] || 360);
                    if (expiry <= curTime) return false;
                }
                return true;
            });
        });

        // --- Utility ---
        function formatGameTime(seconds) {
            const sign = seconds < 0 ? '-' : '';
            const abs = Math.abs(Math.floor(seconds));
            return `${sign}${Math.floor(abs / 60)}:${(abs % 60).toString().padStart(2, '0')}`;
        }

        function toggleSmokeSelect(idx) {
            const arr = smokeSelected.value;
            const pos = arr.indexOf(idx);
            if (pos === -1) {
                smokeSelected.value = [...arr, idx];
            } else {
                smokeSelected.value = arr.filter(i => i !== idx);
            }
        }

        function clearSmokeSelection() {
            smokeSelected.value = [];
        }

        function interpolatePosition(positions, timeOffset) {
            if (!positions || positions.length === 0) return null;
            if (positions.length === 1) return { x: positions[0].x, y: positions[0].y };
            if (timeOffset <= positions[0].time_offset) return { x: positions[0].x, y: positions[0].y };
            const last = positions[positions.length - 1];
            if (timeOffset >= last.time_offset) return { x: last.x, y: last.y };
            for (let i = 0; i < positions.length - 1; i++) {
                const a = positions[i], b = positions[i + 1];
                if (timeOffset >= a.time_offset && timeOffset <= b.time_offset) {
                    const frac = (b.time_offset - a.time_offset) === 0 ? 0 : (timeOffset - a.time_offset) / (b.time_offset - a.time_offset);
                    return { x: a.x + (b.x - a.x) * frac, y: a.y + (b.y - a.y) * frac };
                }
            }
            return { x: last.x, y: last.y };
        }

        function formatNumber(n) {
            if (n == null || n === 0) return '-';
            if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
            return String(n);
        }

        function formatNetWorth(n) {
            if (n == null || n === 0) return '0';
            if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
            return String(n);
        }

        // Region ID to name mapping (OpenDota regions)
        const REGION_MAP = {
            0: '', 1: 'US West', 2: 'US East', 3: 'Europe', 5: 'Singapore',
            6: 'Dubai', 7: 'Australia', 8: 'Stockholm', 9: 'Austria',
            10: 'Brazil', 11: 'South Africa', 12: 'PW Telecom Shanghai',
            13: 'PW Unicom', 14: 'Chile', 15: 'Peru', 16: 'India',
            17: 'PW Telecom Guangdong', 18: 'PW Telecom Zhejiang',
            19: 'Japan', 20: 'PW Telecom Wuhan', 25: 'PW Unicom Tianjin',
            37: 'Taiwan', 38: 'Argentina',
        };

        function regionName(regionId) {
            if (!regionId) return '';
            return REGION_MAP[regionId] || ('Region ' + regionId);
        }

        function timeAgo(unixTs) {
            if (!unixTs) return '';
            const now = Date.now() / 1000;
            const diff = Math.floor(now - unixTs);
            if (diff < 60) return diff + ' ' + t('time_seconds_ago');
            if (diff < 3600) return Math.floor(diff / 60) + ' ' + t('time_minutes_ago');
            if (diff < 86400) return Math.floor(diff / 3600) + ' ' + t('time_hours_ago');
            const days = Math.floor(diff / 86400);
            if (days < 30) return days + ' ' + t('time_days_ago');
            const months = Math.floor(days / 30);
            return months + ' ' + t('time_months_ago');
        }

        function teamSum(players, field) {
            if (!players || !players.length) return 0;
            return players.reduce((sum, p) => sum + (p[field] || 0), 0);
        }

        function getItemPurchaseTime(player, itemKey) {
            if (!player.purchase_log || !itemKey) return '';
            const entry = player.purchase_log.find(e => e.key === itemKey);
            return entry ? entry.time_str : '';
        }

        function shortAbility(name) {
            if (name.startsWith('talent_')) {
                return name.includes('left') ? 'T-L' : 'T-R';
            }
            const words = name.split(' ');
            if (words.length === 1) return name.slice(0, 4);
            return words.map(w => w[0]).join('').toUpperCase();
        }

        function getSkillColor(ability, heroName) {
            const player = analysisData.value?.skill_builds?.find(p => p.hero_name === heroName);
            if (!player) return 'skill-q';
            const nonTalents = player.skill_order.filter(s => !s.is_talent).map(s => s.ability);
            const unique = [...new Set(nonTalents)];
            const idx = unique.indexOf(ability);
            return ['skill-q', 'skill-w', 'skill-e', 'skill-r'][idx] || 'skill-q';
        }

        // --- File Handling ---
        async function handleFileSelect(e) {
            const file = e.target.files[0];
            if (file) await uploadFile(file);
        }

        async function handleDrop(e) {
            const file = e.dataTransfer.files[0];
            if (file) await uploadFile(file);
        }

        async function uploadFile(file) {
            if (!file.name.endsWith('.dem')) {
                uploadError.value = t('error_invalid_format');
                return;
            }
            uploading.value = true;
            uploadError.value = '';
            const formData = new FormData();
            formData.append('file', file);
            try {
                const res = await fetch('/api/upload', { method: 'POST', body: formData });
                if (!res.ok) throw new Error((await res.json()).detail || t('error_upload_failed'));
                const data = await res.json();
                parseSource.value = data.parse_source;
                matchKey.value = data._match_key || data.file_key || '';
                const aRes = await fetch(`/api/match/${data.file_key}`);
                const aData = await aRes.json();
                analysisData.value = aData.analysis;
                if (aData._match_key) matchKey.value = aData._match_key;
            } catch (e) {
                uploadError.value = e.message;
            } finally {
                uploading.value = false;
            }
        }

        const matchIdInput = ref('');

        async function fetchMatchById() {
            const mid = matchIdInput.value.trim();
            if (!mid) return;
            uploading.value = true;
            uploadError.value = '';
            try {
                const res = await fetch(`/api/fetch/${mid}`);
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail || `HTTP ${res.status}`);
                }
                const data = await res.json();
                analysisData.value = data.analysis;
                parseSource.value = data.parse_source;
                matchKey.value = data._match_key || mid;
                if (!data.has_parsed_data) {
                    // Request parse for detailed data
                    fetch(`/api/request-parse/${mid}`, { method: 'POST' }).catch(() => {});
                    uploadError.value = t('error_basic_data_fetched');
                }
            } catch (e) {
                uploadError.value = t('error_fetch_failed') + ': ' + e.message;
            } finally {
                uploading.value = false;
            }
        }

        async function loadDemo() {
            uploading.value = true;
            uploadError.value = '';
            try {
                const res = await fetch('/api/demo');
                const data = await res.json();
                analysisData.value = data.analysis;
                parseSource.value = data.parse_source;
                matchKey.value = data._match_key || 'demo';
            } catch (e) {
                uploadError.value = t('error_demo_failed') + ': ' + e.message;
            } finally {
                uploading.value = false;
            }
        }

        function closeAiCoachStream() {
            if (aiCoachEventSource) {
                try { aiCoachEventSource.close(); } catch (e) { /* ignore */ }
                aiCoachEventSource = null;
            }
        }

        function requestAiCoach(playerSlot) {
            // Tear down any in-flight stream before starting a new one.
            closeAiCoachStream();
            aiCoachLoading.value = true;
            aiCoachError.value = '';
            aiCoachPlayerSlot.value = playerSlot;
            aiCoachData.value = null;
            aiCoachProgress.value = { stage: 'starting', current: 0, total: 1, percent: 0 };

            const mk = matchKey.value || 'demo';
            const url = `/api/match/${mk}/ai-coach-stream/${playerSlot}?lang=${locale.value}`;
            const es = new EventSource(url);
            aiCoachEventSource = es;

            es.onmessage = (event) => {
                let msg;
                try { msg = JSON.parse(event.data); } catch (e) { return; }

                if (msg.type === 'progress') {
                    const total = msg.total || 1;
                    const current = msg.current || 0;
                    aiCoachProgress.value = {
                        stage: msg.stage || '',
                        current,
                        total,
                        percent: computeOverallPercent(msg.stage, current, total),
                    };
                } else if (msg.type === 'done') {
                    aiCoachProgress.value = { stage: 'done', current: 1, total: 1, percent: 100 };
                    aiCoachData.value = msg.data;
                    aiCoachLoading.value = false;
                    closeAiCoachStream();
                } else if (msg.type === 'error') {
                    aiCoachError.value = msg.message || t('ai_coach_failed');
                    aiCoachLoading.value = false;
                    closeAiCoachStream();
                }
            };

            es.onerror = () => {
                // EventSource auto-reconnects on transient errors. If the
                // connection truly dies (server closes without 'done'), we
                // bail out so the user isn't stuck on the spinner.
                if (es.readyState === EventSource.CLOSED) {
                    if (aiCoachLoading.value && !aiCoachData.value) {
                        aiCoachError.value = t('ai_coach_failed');
                        aiCoachLoading.value = false;
                    }
                    closeAiCoachStream();
                }
            };
        }

        function resetAnalysis() {
            analysisData.value = null;
            parseSource.value = '';
            matchKey.value = '';
            wardTimelineEnabled.value = false;
            wardTimelineTime.value = 0;
            wardShowVision.value = false;
            smokeTimelineEnabled.value = false;
            smokeTimelineTime.value = 0;
            smokeSelected.value = [];
            clearInterval(teamfightPlayInterval);
            teamfightPlayInterval = null;
            teamfightPlaying.value = false;
            teamfightReplayTime.value = 0;
            [wardChartInstance, smokeMapInstance, smokeChartInstance, teamfightChartInstance, economyChartInstance,
             laningHeatmapInstance, laningCsTimelineInstance, laningHpTimelineInstance, farmReasonChartInstance, farmLhChartInstance]
                .forEach(c => c?.dispose());
            wardChartInstance = smokeMapInstance = smokeChartInstance = teamfightChartInstance = economyChartInstance = null;
            laningHeatmapInstance = laningCsTimelineInstance = laningHpTimelineInstance = farmReasonChartInstance = farmLhChartInstance = null;
            aiCoachData.value = null;
            aiCoachLoading.value = false;
            aiCoachError.value = '';
            aiCoachPlayerSlot.value = null;
            aiCoachProgress.value = { stage: '', current: 0, total: 0, percent: 0 };
            closeAiCoachStream();
        }

        function renderMarkdown(text) {
            if (!text) return '';
            return text
                .replace(/^### (.*$)/gm, '<h4 style="color: var(--accent); margin: 12px 0 6px;">$1</h4>')
                .replace(/^## (.*$)/gm, '<h3 style="color: var(--text-primary); margin: 16px 0 8px;">$1</h3>')
                .replace(/\*\*(.*?)\*\*/g, '<strong style="color: var(--text-primary);">$1</strong>')
                .replace(/^\- (.*$)/gm, '<li style="color: var(--text-secondary); margin: 2px 0;">$1</li>')
                .replace(/^\d+\. (.*$)/gm, '<li style="color: var(--text-secondary); margin: 2px 0;">$1</li>')
                .replace(/\n\n/g, '<br><br>')
                .replace(/\n/g, '<br>');
        }

        // --- Chart Rendering ---
        // Map coordinate conversion following OpenDota's approach:
        // OpenDota cells 64-190 map to the full minimap image (127 units).
        // Game coords: cell = (game + 8000) / 62.5
        // UV = (cell - 64) / 127 * 100
        // Simplified: UV = (game + 4000) / 7937.5 * 100
        function gameToMap(x, y) {
            const mx = (x + 4000) / 7937.5 * 100;
            const my = (y + 4000) / 7937.5 * 100;
            return [mx, my];
        }

        // Shared map chart config (OpenDota-style)
        const MAP_IMAGE = '/static/assets/dota2_map.jpg';
        const MAP_BG = '#1a1e2a';

        function mapGraphic(mapSize) {
            return [{
                type: 'image', left: 0, top: 0, z: -1,
                style: { image: MAP_IMAGE, width: mapSize, height: mapSize, opacity: 0.85 },
            }];
        }

        function mapAxes() {
            return {
                xAxis: { type: 'value', min: 0, max: 100, show: false },
                yAxis: { type: 'value', min: 0, max: 100, show: false },
            };
        }

        function renderWardMap() {
            if (!wardMap.value) return;
            if (!wardChartInstance) {
                wardChartInstance = echarts.init(wardMap.value);
            }
            const mapSize = wardMap.value.clientWidth;

            const wards = timelineWards.value;
            const obsData = [], sentData = [];
            const visionData = [];

            wards.forEach(w => {
                const [mx, my] = gameToMap(w.x, w.y);
                const isObs = w.ward_type === 'observer';
                const isRadiant = w.team === 'radiant';
                // OpenDota style: obs = yellow fill + team border, sentry = blue fill + team border
                const fillColor = isObs ? '#ecc94b' : '#63b3ed';
                const borderColor = isRadiant ? '#48bb78' : '#fc8181';
                const wasDestroyed = w.destroyed_time != null;

                const entry = {
                    value: [mx, my],
                    name: `${w.hero_name} @ ${w.time_str}`,
                    _ward: w,
                    itemStyle: {
                        color: fillColor,
                        borderColor: borderColor,
                        borderWidth: 2,
                        opacity: wasDestroyed ? 0.6 : 1,
                    },
                };
                (isObs ? obsData : sentData).push(entry);

                if (wardShowVision.value) {
                    const radius = isObs ? OBS_VISION_RADIUS : SENTRY_VISION_RADIUS;
                    const visionFill = isRadiant
                        ? (isObs ? 'rgba(236,201,75,0.12)' : 'rgba(99,179,237,0.10)')
                        : (isObs ? 'rgba(236,201,75,0.12)' : 'rgba(99,179,237,0.10)');
                    const visionStroke = isRadiant
                        ? (isObs ? 'rgba(72,187,120,0.5)' : 'rgba(99,179,237,0.4)')
                        : (isObs ? 'rgba(252,129,129,0.5)' : 'rgba(246,135,179,0.4)');
                    visionData.push({ value: [mx, my, radius], fill: visionFill, stroke: visionStroke });
                }
            });

            const series = [];

            // Vision circles (behind wards)
            if (wardShowVision.value && visionData.length > 0) {
                series.push({
                    name: t('wards_vision_series'), type: 'custom', data: visionData,
                    z: 5, silent: true,
                    renderItem(params, api) {
                        const center = api.coord([api.value(0), api.value(1)]);
                        const radiusVal = api.value(2);
                        const pixelSize = api.size([radiusVal, radiusVal]);
                        const d = visionData[params.dataIndex];
                        return {
                            type: 'circle',
                            shape: { cx: center[0], cy: center[1], r: pixelSize[0] },
                            style: { fill: d.fill, stroke: d.stroke, lineWidth: 1.5 },
                        };
                    },
                });
            }

            // Observer wards - circles (OpenDota style: yellow with team border)
            series.push({
                name: t('wards_observer'), type: 'scatter', data: obsData,
                symbolSize: 16, symbol: 'circle', z: 10,
            });

            // Sentry wards - diamonds
            series.push({
                name: t('wards_sentry'), type: 'scatter', data: sentData,
                symbolSize: 12, symbol: 'diamond', z: 10,
            });

            // Destroyed ward markers (X marks)
            const destroyedData = wards
                .filter(w => w.destroyed_time != null)
                .map(w => {
                    const [mx, my] = gameToMap(w.x, w.y);
                    return {
                        value: [mx, my],
                        name: `${t('wards_dewarded')} ${formatGameTime(w.destroyed_time)}`,
                        itemStyle: { color: 'rgba(239,68,68,0.7)', borderColor: '#ef4444', borderWidth: 1 },
                    };
                });
            if (destroyedData.length > 0) {
                series.push({
                    name: t('wards_destroyed'), type: 'scatter', data: destroyedData,
                    symbolSize: 8, symbol: 'path://M-1,-1L1,1M1,-1L-1,1', z: 12,
                    silent: true,
                });
            }

            wardChartInstance.setOption({
                backgroundColor: MAP_BG,
                grid: { left: 0, right: 0, top: 0, bottom: 0 },
                tooltip: {
                    trigger: 'item',
                    backgroundColor: 'rgba(26,32,44,0.95)',
                    borderColor: '#4a5568',
                    textStyle: { color: '#e2e8f0', fontSize: 12 },
                    formatter: p => {
                        if (!p.data?._ward) return p.name;
                        const w = p.data._ward;
                        const type = w.ward_type === 'observer' ? t('wards_observer') : t('wards_sentry');
                        const team = w.team === 'radiant' ? t('radiant') : t('dire');
                        let html = `<b>${w.hero_name}</b><br/>${type} (${team})<br/>${t('wards_placed')}: ${w.time_str}`;
                        if (w.destroyed_time != null) {
                            const life = w.destroyed_time - w.time;
                            html += `<br/><span style="color:#fc8181">${t('wards_dewarded')} ${formatGameTime(w.destroyed_time)} (${t('wards_alive')} ${life}s)</span>`;
                        }
                        return html;
                    },
                },
                graphic: mapGraphic(mapSize),
                ...mapAxes(),
                series,
            }, true);
        }

        function renderSmokeMap() {
            if (!smokeMapRef.value) return;
            if (!smokeMapInstance) {
                smokeMapInstance = echarts.init(smokeMapRef.value);
            }
            const mapSize = smokeMapRef.value.clientWidth;

            const timeline = smokeData.value.timeline || [];
            let smokes = timeline;
            if (smokeTimelineEnabled.value) {
                const curTime = smokeTimelineTime.value;
                smokes = timeline.filter(s => s.time <= curTime);
            }
            if (smokeSelected.value.length > 0) {
                smokes = smokes.filter(s => smokeSelected.value.includes(timeline.indexOf(s)));
            }

            const pathsData = [];
            const startData = [];
            const endData = [];
            const arrowData = [];

            smokes.forEach(s => {
                const isRadiant = s.team === 'radiant';
                const color = isRadiant ? '#48bb78' : '#fc8181';
                const pathColor = isRadiant ? 'rgba(72,187,120,0.7)' : 'rgba(252,129,129,0.7)';
                const resultIcon = s.result === 'kill' ? ' ⚔' : s.result === 'objective' ? ' 🏰' : '';

                const [sx, sy] = gameToMap(s.x, s.y);
                startData.push({
                    value: [sx, sy],
                    _smoke: s,
                    itemStyle: {
                        color,
                        borderColor: 'rgba(255,255,255,0.8)',
                        borderWidth: 2,
                        shadowColor: color,
                        shadowBlur: 6,
                    },
                });

                if (s.path && s.path.length > 0) {
                    const allPts = [[s.x, s.y], ...s.path];
                    const mapped = allPts.map(p => gameToMap(p[0], p[1]));
                    pathsData.push({ points: mapped, color: pathColor });

                    const last = s.path[s.path.length - 1];
                    const [ex, ey] = gameToMap(last[0], last[1]);
                    endData.push({
                        value: [ex, ey],
                        name: `${t('smokes_smoke_break')} ${s.end_time_str || ''}${resultIcon}`,
                        itemStyle: {
                            color,
                            borderColor: 'rgba(255,255,255,0.6)',
                            borderWidth: 1.5,
                        },
                    });

                    // Arrow from second-to-last point toward last point
                    if (mapped.length >= 2) {
                        const prev = mapped[mapped.length - 2];
                        arrowData.push({ from: prev, to: [ex, ey], color: pathColor });
                    }
                }
            });

            const series = [];

            // Movement paths with dashed lines
            if (pathsData.length > 0) {
                series.push({
                    name: t('smokes_movement_path'), type: 'custom',
                    data: pathsData.map((_, i) => ({ value: [i] })),
                    z: 3, silent: true,
                    renderItem(params, api) {
                        const d = pathsData[params.dataIndex];
                        const pixelPoints = d.points.map(pt => api.coord(pt));
                        return {
                            type: 'polyline',
                            shape: { points: pixelPoints },
                            style: { stroke: d.color, lineWidth: 2.5, lineDash: [8, 4], fill: 'none' },
                        };
                    },
                });
            }

            // Arrows at path ends
            if (arrowData.length > 0) {
                series.push({
                    name: t('smokes_direction'), type: 'custom',
                    data: arrowData.map((_, i) => ({ value: [i] })),
                    z: 4, silent: true,
                    renderItem(params, api) {
                        const d = arrowData[params.dataIndex];
                        const to = api.coord(d.to);
                        const from = api.coord(d.from);
                        const angle = Math.atan2(to[1] - from[1], to[0] - from[0]);
                        const sz = 8;
                        return {
                            type: 'polygon',
                            shape: {
                                points: [
                                    [to[0], to[1]],
                                    [to[0] - sz * Math.cos(angle - 0.4), to[1] - sz * Math.sin(angle - 0.4)],
                                    [to[0] - sz * Math.cos(angle + 0.4), to[1] - sz * Math.sin(angle + 0.4)],
                                ],
                            },
                            style: { fill: d.color },
                        };
                    },
                });
            }

            // Smoke start points
            series.push({
                name: t('smokes_start_pos'), type: 'scatter', data: startData,
                symbolSize: 16, symbol: 'circle', z: 10,
            });

            // Smoke break points (triangles)
            if (endData.length > 0) {
                series.push({
                    name: t('smokes_break_point'), type: 'scatter', data: endData,
                    symbolSize: 12, symbol: 'triangle', z: 10,
                });
            }

            smokeMapInstance.setOption({
                backgroundColor: MAP_BG,
                grid: { left: 0, right: 0, top: 0, bottom: 0 },
                tooltip: {
                    trigger: 'item',
                    backgroundColor: 'rgba(26,32,44,0.95)',
                    borderColor: '#4a5568',
                    textStyle: { color: '#e2e8f0', fontSize: 12 },
                    formatter: p => {
                        if (!p.data?._smoke) return p.name;
                        const s = p.data._smoke;
                        const team = s.team === 'radiant' ? t('radiant') : t('dire');
                        let html = `<b>${team} ${t('tab_smokes')}</b> ${s.time_str}<br/>${s.num_players} ${t('smokes_persons')}: ${s.participants.map(pp => pp.hero_name).join(', ')}`;
                        if (s.duration) html += `<br/>${t('smokes_duration_label')}: ${s.duration}${t('smokes_seconds_suffix')}`;
                        if (s.result) html += `<br/>${t('smokes_result_label')}: ${smokeResultLabel(s.result)}`;
                        if (s.kills_gained > 0) html += ` (${s.kills_gained} ${t('smokes_kills_suffix')})`;
                        return html;
                    },
                },
                graphic: mapGraphic(mapSize),
                ...mapAxes(),
                series,
            }, true);
        }

        function renderSmokeTimeline() {
            if (!smokeChart.value) return;
            if (smokeChartInstance) smokeChartInstance.dispose();
            smokeChartInstance = echarts.init(smokeChart.value);

            const timeline = smokeData.value.timeline || [];
            const duration = overview.value.duration || 3600;

            smokeChartInstance.setOption({
                backgroundColor: 'transparent',
                tooltip: {
                    trigger: 'item',
                    formatter: p => {
                        const d = timeline[p.dataIndex];
                        if (!d) return '';
                        return `${d.time_str} - ${d.team === 'radiant' ? t('radiant') : t('dire')}<br>${d.num_players} ${t('smokes_persons')}: ${d.participants.map(pp => pp.hero_name).join(', ')}${d.duration ? '<br>' + t('smokes_duration_label') + ': ' + d.duration + t('smokes_seconds_suffix') : ''}`;
                    },
                },
                grid: { left: 60, right: 20, top: 30, bottom: 40 },
                xAxis: {
                    type: 'value', name: t('chart_game_time'),
                    min: 0, max: Math.ceil(duration / 60),
                    nameTextStyle: { color: '#a0aec0' },
                    axisLabel: { color: '#a0aec0' },
                    splitLine: { lineStyle: { color: '#2d3748' } },
                },
                yAxis: { type: 'category', data: [t('radiant'), t('dire')], axisLabel: { color: '#a0aec0' }, splitLine: { show: false } },
                series: [{
                    type: 'scatter',
                    data: timeline.map(s => ({
                        value: [Math.round(s.time / 60 * 10) / 10, s.team === 'radiant' ? t('radiant') : t('dire')],
                        itemStyle: { color: s.team === 'radiant' ? '#48bb78' : '#fc8181' },
                    })),
                    symbolSize: (d) => {
                        const _radiantLabel = t('radiant');
                        const s = timeline.find(tt =>
                            Math.round(tt.time / 60 * 10) / 10 === d[0] &&
                            (d[1] === _radiantLabel ? 'radiant' : 'dire') === tt.team);
                        return s ? s.num_players * 5 + 5 : 10;
                    },
                }],
            });
        }

        function renderTeamfightMap() {
            if (!teamfightMap.value || !currentTeamfight.value) return;
            if (!teamfightChartInstance) {
                teamfightChartInstance = echarts.init(teamfightMap.value);
            }
            const mapSize = teamfightMap.value.clientWidth;

            const tf = currentTeamfight.value;
            const replayT = teamfightReplayTime.value;

            const heroPositions = [];
            const pathsData = [];
            const castData = [];
            const deathMarkers = [];

            tf.players.forEach(p => {
                const team = p.team || (p.player_slot < 128 ? 'radiant' : 'dire');
                const isRadiant = team === 'radiant';
                const color = isRadiant ? '#48bb78' : '#fc8181';
                const dimColor = isRadiant ? 'rgba(72,187,120,0.3)' : 'rgba(252,129,129,0.3)';
                const pathColor = isRadiant ? 'rgba(72,187,120,0.5)' : 'rgba(252,129,129,0.5)';

                let mx, my;
                let isDead = false;

                if (p.positions && p.positions.length > 0) {
                    const pos = interpolatePosition(p.positions, replayT);
                    if (pos) {
                        [mx, my] = gameToMap(pos.x, pos.y);
                    } else {
                        [mx, my] = gameToMap(p.x, p.y);
                    }

                    // Build trail path up to current time
                    const trailPoints = p.positions
                        .filter(pp => pp.time_offset <= replayT)
                        .map(pp => gameToMap(pp.x, pp.y));
                    trailPoints.push([mx, my]);
                    if (trailPoints.length >= 2) {
                        pathsData.push({ points: trailPoints, color: pathColor });
                    }

                    isDead = p.deaths > 0 && p.death_time_offset != null && replayT >= p.death_time_offset;
                } else {
                    [mx, my] = gameToMap(p.x, p.y);
                    isDead = p.deaths > 0 && p.death_time_offset != null;
                }

                // Hero position marker
                const slug = heroSlugByName(p.hero_name);
                const iconUrl = slug ? heroIcon(slug) : '';
                heroPositions.push({
                    value: [mx, my],
                    name: p.hero_name,
                    _player: p,
                    symbol: iconUrl ? `image://${iconUrl}` : 'circle',
                    symbolSize: isDead ? 14 : 20,
                    itemStyle: {
                        color,
                        opacity: isDead ? 0.25 : 1,
                        borderColor: isRadiant ? '#276749' : '#9b2c2c',
                        borderWidth: 2,
                    },
                    label: {
                        show: true, position: 'top', fontSize: 10,
                        formatter: isDead ? `☠` : '',
                        color: isDead ? '#ef4444' : color,
                        fontWeight: 'bold',
                        distance: 2,
                    },
                });

                // Death skull marker
                if (isDead) {
                    deathMarkers.push({
                        value: [mx, my],
                        name: `${p.hero_name} ${t('teamfights_died')}`,
                        itemStyle: { color: '#ef4444', borderColor: 'rgba(239,68,68,0.5)', borderWidth: 1 },
                    });
                }

                // Ability casts up to current time
                if (p.ability_casts) {
                    p.ability_casts.forEach(ac => {
                        if (ac.time_offset <= replayT) {
                            const [ax, ay] = gameToMap(ac.x, ac.y);
                            castData.push({
                                value: [ax, ay],
                                name: ac.ability_name,
                                itemStyle: {
                                    color: '#ecc94b',
                                    borderColor: color,
                                    borderWidth: 2,
                                    shadowColor: '#ecc94b',
                                    shadowBlur: 4,
                                },
                            });
                        }
                    });
                }
            });

            const series = [];

            // Movement trail paths
            if (pathsData.length > 0) {
                series.push({
                    name: t('smokes_movement_path'), type: 'custom',
                    data: pathsData.map((_, i) => ({ value: [i] })),
                    z: 3, silent: true,
                    renderItem(params, api) {
                        const d = pathsData[params.dataIndex];
                        const pixelPoints = d.points.map(pt => api.coord(pt));
                        return {
                            type: 'polyline',
                            shape: { points: pixelPoints },
                            style: { stroke: d.color, lineWidth: 2.5, fill: 'none' },
                        };
                    },
                });
            }

            // Ability cast markers (diamonds)
            if (castData.length > 0) {
                series.push({
                    name: t('teamfights_ability_cast'), type: 'scatter', data: castData,
                    symbolSize: 10, symbol: 'diamond', z: 8,
                });
            }

            // Hero positions (with hero icons)
            series.push({
                name: t('teamfights_hero_pos'), type: 'scatter', data: heroPositions,
                z: 10,
            });

            teamfightChartInstance.setOption({
                backgroundColor: MAP_BG,
                grid: { left: 0, right: 0, top: 0, bottom: 0 },
                tooltip: {
                    trigger: 'item',
                    backgroundColor: 'rgba(26,32,44,0.95)',
                    borderColor: '#4a5568',
                    textStyle: { color: '#e2e8f0', fontSize: 12 },
                    formatter: p => {
                        const pl = p.data?._player || tf.players.find(pp => pp.hero_name === p.name);
                        if (!pl) return p.name;
                        const team = (pl.team || (pl.player_slot < 128 ? 'radiant' : 'dire')) === 'radiant' ? t('radiant') : t('dire');
                        const isDead = pl.deaths > 0;
                        return `<b>${pl.hero_name}</b> (${team})${isDead ? ' ☠' : ''}<br/>K/D: ${pl.kills}/${pl.deaths}<br/>${t('teamfights_damage')}: ${formatNumber(pl.damage_dealt)}<br/>${t('teamfights_taken')}: ${formatNumber(pl.damage_taken)}<br/>${t('teamfights_abilities')}: ${pl.abilities_used.join(', ') || '-'}`;
                    },
                },
                graphic: mapGraphic(mapSize),
                ...mapAxes(),
                series,
            }, true);
        }

        function toggleTeamfightPlay() {
            if (teamfightPlaying.value) {
                clearInterval(teamfightPlayInterval);
                teamfightPlayInterval = null;
                teamfightPlaying.value = false;
                return;
            }
            const tf = currentTeamfight.value;
            if (!tf) return;
            const tfDur = tf.end_time - tf.start_time;
            if (teamfightReplayTime.value >= tfDur) {
                teamfightReplayTime.value = 0;
            }
            teamfightPlaying.value = true;
            const stepMs = Math.round(100 / teamfightPlaySpeed.value);
            teamfightPlayInterval = setInterval(() => {
                teamfightReplayTime.value += 0.5;
                if (teamfightReplayTime.value >= tfDur) {
                    teamfightReplayTime.value = tfDur;
                    clearInterval(teamfightPlayInterval);
                    teamfightPlayInterval = null;
                    teamfightPlaying.value = false;
                }
            }, stepMs);
        }

        // --- Economy Chart ---
        const economyMode = ref('gold_diff'); // 'gold_diff', 'gold_total', 'xp_diff', 'xp_total', 'player_gold', 'player_xp', 'player_lh'

        function renderEconomyChart() {
            if (!economyChart.value || !analysisData.value?.gold_xp_timeline) return;
            if (!economyChartInstance) {
                economyChartInstance = echarts.init(economyChart.value);
            }
            const timeline = analysisData.value.gold_xp_timeline;
            let times = timeline.map(s => s.time_str);
            const mode = economyMode.value;

            let series, yAxisConfig;
            if (mode === 'gold_diff') {
                const diffData = timeline.map(s => s.gold_diff);
                series = [{
                    name: t('economy_gold_diff'),
                    type: 'line',
                    data: diffData,
                    smooth: true,
                    lineStyle: { color: '#ecc94b', width: 2 },
                    areaStyle: {
                        color: {
                            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
                            colorStops: [
                                { offset: 0, color: 'rgba(72,187,120,0.3)' },
                                { offset: 0.5, color: 'rgba(236,201,75,0.05)' },
                                { offset: 1, color: 'rgba(252,129,129,0.3)' },
                            ],
                        },
                    },
                    itemStyle: { color: '#ecc94b' },
                    markLine: {
                        data: [{ yAxis: 0 }],
                        lineStyle: { color: '#4a5568', type: 'dashed' },
                        label: { show: false },
                        symbol: 'none',
                    },
                }];
                yAxisConfig = {
                    type: 'value',
                    axisLabel: {
                        color: '#a0aec0', fontSize: 11,
                        formatter: v => v >= 0 ? `+${(v/1000).toFixed(1)}k` : `${(v/1000).toFixed(1)}k`,
                    },
                    splitLine: { lineStyle: { color: '#2d3748' } },
                };
            } else if (mode === 'xp_diff') {
                const diffData = timeline.map(s => s.xp_diff);
                series = [{
                    name: t('economy_xp_diff'),
                    type: 'line',
                    data: diffData,
                    smooth: true,
                    lineStyle: { color: '#63b3ed', width: 2 },
                    areaStyle: {
                        color: {
                            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
                            colorStops: [
                                { offset: 0, color: 'rgba(72,187,120,0.3)' },
                                { offset: 0.5, color: 'rgba(99,179,237,0.05)' },
                                { offset: 1, color: 'rgba(252,129,129,0.3)' },
                            ],
                        },
                    },
                    itemStyle: { color: '#63b3ed' },
                    markLine: {
                        data: [{ yAxis: 0 }],
                        lineStyle: { color: '#4a5568', type: 'dashed' },
                        label: { show: false },
                        symbol: 'none',
                    },
                }];
                yAxisConfig = {
                    type: 'value',
                    axisLabel: {
                        color: '#a0aec0', fontSize: 11,
                        formatter: v => v >= 0 ? `+${(v/1000).toFixed(1)}k` : `${(v/1000).toFixed(1)}k`,
                    },
                    splitLine: { lineStyle: { color: '#2d3748' } },
                };
            } else {
                // Total gold or XP
                const isGold = mode === 'gold_total';
                const rKey = isGold ? 'radiant_gold' : 'radiant_xp';
                const dKey = isGold ? 'dire_gold' : 'dire_xp';
                series = [
                    {
                        name: t('radiant'),
                        type: 'line',
                        data: timeline.map(s => s[rKey]),
                        smooth: true,
                        lineStyle: { color: '#48bb78', width: 2 },
                        itemStyle: { color: '#48bb78' },
                    },
                    {
                        name: t('dire'),
                        type: 'line',
                        data: timeline.map(s => s[dKey]),
                        smooth: true,
                        lineStyle: { color: '#fc8181', width: 2 },
                        itemStyle: { color: '#fc8181' },
                    },
                ];
                yAxisConfig = {
                    type: 'value',
                    axisLabel: {
                        color: '#a0aec0', fontSize: 11,
                        formatter: v => `${(v/1000).toFixed(0)}k`,
                    },
                    splitLine: { lineStyle: { color: '#2d3748' } },
                };
            }

            if (mode === 'player_gold' || mode === 'player_xp' || mode === 'player_lh') {
                const graphData = analysisData.value?.graphs;
                if (!graphData) return;
                const playerList = mode === 'player_gold' ? graphData.player_gold
                    : mode === 'player_xp' ? graphData.player_xp
                    : graphData.player_lh;
                if (!playerList || playerList.length === 0) return;

                const playerColors = [
                    '#3b82f6', '#10b981', '#a855f7', '#ecc94b', '#f97316',
                    '#ef4444', '#ec4899', '#8b5cf6', '#06b6d4', '#84cc16',
                ];
                const dataKey = mode === 'player_gold' ? 'gold_t' : mode === 'player_xp' ? 'xp_t' : 'lh_t';
                const maxLen = Math.max(...playerList.map(p => (p[dataKey] || []).length));
                times = Array.from({ length: maxLen }, (_, i) => `${i}:00`);

                series = playerList.map((p, idx) => ({
                    name: p.hero_name,
                    type: 'line',
                    data: p[dataKey] || [],
                    smooth: true,
                    lineStyle: { color: playerColors[idx % 10], width: 2 },
                    itemStyle: { color: playerColors[idx % 10] },
                    symbol: 'none',
                }));
                yAxisConfig = {
                    type: 'value',
                    axisLabel: {
                        color: '#a0aec0', fontSize: 11,
                        formatter: v => mode === 'player_lh' ? String(v) : `${(v/1000).toFixed(0)}k`,
                    },
                    splitLine: { lineStyle: { color: '#2d3748' } },
                };
            }

            // Pre-build hero slug map outside formatter for performance
            const _heroSlugMap = {};
            if (mode.startsWith('player_')) {
                const graphData = analysisData.value?.graphs;
                const pList = mode === 'player_gold' ? graphData?.player_gold
                    : mode === 'player_xp' ? graphData?.player_xp
                    : graphData?.player_lh;
                if (pList) pList.forEach(p => { _heroSlugMap[p.hero_name] = p.hero_slug; });
            }

            economyChartInstance.setOption({
                backgroundColor: 'transparent',
                tooltip: {
                    trigger: 'axis',
                    enterable: false,
                    backgroundColor: '#1a202c',
                    borderColor: '#4a5568',
                    textStyle: { color: '#e2e8f0', fontSize: 12 },
                    formatter: params => {
                        if (!params || params.length === 0) return '';
                        const idx = params[0].dataIndex;
                        if (mode === 'gold_diff' || mode === 'xp_diff') {
                            const s = timeline[idx];
                            if (!s) return '';
                            let html = `<b>${s.time_str}</b><br/>`;
                            if (mode === 'gold_diff') {
                                const d = s.gold_diff;
                                const label = d >= 0 ? t('economy_radiant_lead') : t('economy_dire_lead');
                                html += `${t('economy_gold_diff')}: <span style="color:${d >= 0 ? '#48bb78' : '#fc8181'}">${d >= 0 ? '+' : ''}${(d/1000).toFixed(1)}k</span> (${label})`;
                            } else {
                                const d = s.xp_diff;
                                const label = d >= 0 ? t('economy_radiant_lead') : t('economy_dire_lead');
                                html += `${t('economy_xp_diff')}: <span style="color:${d >= 0 ? '#48bb78' : '#fc8181'}">${d >= 0 ? '+' : ''}${(d/1000).toFixed(1)}k</span> (${label})`;
                            }
                            return html;
                        } else {
                            const mm = Math.floor(idx);
                            let html = `<div style="min-width:240px"><b>${mm}:00</b>`;
                            const sorted = [...params].sort((a, b) => (b.value || 0) - (a.value || 0));
                            sorted.forEach(p => {
                                const val = mode === 'player_lh' ? p.value : `${(p.value/1000).toFixed(1)}k`;
                                const slug = _heroSlugMap[p.seriesName] || '';
                                const img = slug ? `<img src="${CDN}/heroes/icons/${slug}.png" style="width:18px;height:18px;border-radius:2px;">` : '';
                                html += `<div style="display:flex;align-items:center;gap:4px;margin-top:4px;">${img}${p.marker}<span>${p.seriesName}: ${val}</span></div>`;
                            });
                            html += '</div>';
                            return html;
                        }
                    },
                },
                legend: {
                    show: mode === 'gold_total' || mode === 'xp_total' || mode.startsWith('player_'),
                    textStyle: { color: '#a0aec0', fontSize: 10 },
                    top: 5,
                    type: mode.startsWith('player_') ? 'scroll' : 'plain',
                },
                grid: { left: 60, right: 20, top: 40, bottom: 40 },
                xAxis: {
                    type: 'category',
                    data: times,
                    axisLabel: { color: '#a0aec0', fontSize: 11 },
                    axisLine: { lineStyle: { color: '#4a5568' } },
                },
                yAxis: yAxisConfig,
                series,
            }, true);
        }

        // --- Laning Heatmap ---
        function renderLaningHeatmap() {
            if (!laningHeatmap.value) return;
            if (!laningHeatmapInstance) {
                laningHeatmapInstance = echarts.init(laningHeatmap.value);
            }
            const mapSize = laningHeatmap.value.clientWidth;
            const slot = selectedLaningPlayer.value;
            const player = analysisData.value?.match_overview?.players?.find(p => p.player_slot === slot);
            // Find raw lane_pos from the graphs data player list
            const graphPlayer = analysisData.value?.graphs?.player_gold?.find(p => p.player_slot === slot);
            // We need lane_pos from raw data - check if it exists in the laning data
            // For now, show hero position from laning data
            const laningPlayer = laningData.value.players?.find(p => p.player_slot === slot);

            laningHeatmapInstance.setOption({
                backgroundColor: MAP_BG,
                grid: { left: 0, right: 0, top: 0, bottom: 0 },
                title: {
                    text: laningPlayer ? laningPlayer.hero_name + ' - ' + translateLaneName(laningPlayer) : '',
                    left: 'center', top: 5,
                    textStyle: { color: '#a0aec0', fontSize: 12 },
                },
                graphic: mapGraphic(mapSize),
                ...mapAxes(),
                series: [],
            }, true);
        }

        // --- Laning CS Timeline Chart ---
        function renderLaningCsTimeline() {
            if (!laningCsTimeline.value) return;
            if (!laningCsTimelineInstance) {
                laningCsTimelineInstance = echarts.init(laningCsTimeline.value);
            }
            const laning = analysisData.value?.laning;
            if (!laning?.players || laning.players.length === 0) {
                laningCsTimelineInstance.setOption({ title: { text: t('laning_no_cs_data'), left: 'center', top: 'center', textStyle: { color: '#a0aec0' } } }, true);
                return;
            }

            // Check if any player has snapshot data
            const hasSnapshots = laning.players.some(p => p.laning_snapshots && p.laning_snapshots.length > 0);
            if (!hasSnapshots) {
                laningCsTimelineInstance.setOption({ title: { text: t('laning_no_cs_data'), left: 'center', top: 'center', textStyle: { color: '#a0aec0' } } }, true);
                return;
            }

            const radiantColors = ['#48bb78', '#38a169', '#68d391', '#2f855a', '#9ae6b4'];
            const direColors = ['#fc8181', '#f56565', '#feb2b2', '#e53e3e', '#fc9c9c'];

            const series = laning.players.map((p, idx) => {
                const snapshots = p.laning_snapshots || [];
                const isRadiant = p.team === 'radiant';
                const teamIdx = isRadiant ? laning.players.filter((pp, ii) => ii < idx && pp.team === 'radiant').length : laning.players.filter((pp, ii) => ii < idx && pp.team === 'dire').length;
                const color = isRadiant ? (radiantColors[teamIdx % 5]) : (direColors[teamIdx % 5]);
                return {
                    name: p.hero_name,
                    type: 'line',
                    smooth: true,
                    data: snapshots.map(s => [s.time / 60, s.last_hits]),
                    lineStyle: { width: 2, color },
                    itemStyle: { color },
                    symbol: 'none',
                };
            });

            laningCsTimelineInstance.setOption({
                title: { text: t('laning_cs_title'), textStyle: { color: '#ddd', fontSize: 14 }, left: 10, top: 5 },
                tooltip: {
                    trigger: 'axis',
                    backgroundColor: 'rgba(26,32,44,0.95)',
                    borderColor: '#4a5568',
                    textStyle: { color: '#e2e8f0', fontSize: 11 },
                    formatter: params => {
                        if (!params || params.length === 0) return '';
                        const min = params[0].value[0].toFixed(1);
                        let html = `<b>${min} ${t('laning_minutes')}</b>`;
                        const sorted = [...params].sort((a, b) => (b.value[1] || 0) - (a.value[1] || 0));
                        sorted.forEach(p => {
                            html += `<div style="margin-top:2px;">${p.marker} ${p.seriesName}: ${p.value[1]} ${t('laning_lh')}</div>`;
                        });
                        return html;
                    },
                },
                legend: {
                    data: laning.players.map(p => p.hero_name),
                    textStyle: { color: '#aaa', fontSize: 10 },
                    top: 5, right: 10, type: 'scroll',
                },
                xAxis: { type: 'value', name: t('laning_minutes'), min: 0, max: 10, axisLabel: { color: '#aaa' }, nameTextStyle: { color: '#a0aec0' }, splitLine: { lineStyle: { color: '#2d3748' } } },
                yAxis: { type: 'value', name: t('laning_last_hits'), axisLabel: { color: '#aaa' }, nameTextStyle: { color: '#a0aec0' }, splitLine: { lineStyle: { color: '#2d3748' } } },
                series: series,
                backgroundColor: 'transparent',
                grid: { left: 60, right: 20, top: 60, bottom: 40 },
            }, true);
        }

        // --- Laning HP% Timeline Chart ---
        function renderLaningHpTimeline() {
            if (!laningHpTimeline.value) return;
            if (!laningHpTimelineInstance) {
                laningHpTimelineInstance = echarts.init(laningHpTimeline.value);
            }
            const laning = analysisData.value?.laning;
            if (!laning?.players || laning.players.length === 0) {
                laningHpTimelineInstance.setOption({ title: { text: t('laning_no_hp_data'), left: 'center', top: 'center', textStyle: { color: '#a0aec0' } } }, true);
                return;
            }

            const hasSnapshots = laning.players.some(p => p.laning_snapshots && p.laning_snapshots.length > 0);
            if (!hasSnapshots) {
                laningHpTimelineInstance.setOption({ title: { text: t('laning_no_hp_data'), left: 'center', top: 'center', textStyle: { color: '#a0aec0' } } }, true);
                return;
            }

            const radiantColors = ['#48bb78', '#38a169', '#68d391', '#2f855a', '#9ae6b4'];
            const direColors = ['#fc8181', '#f56565', '#feb2b2', '#e53e3e', '#fc9c9c'];

            const series = laning.players.map((p, idx) => {
                const snapshots = p.laning_snapshots || [];
                const isRadiant = p.team === 'radiant';
                const teamIdx = isRadiant ? laning.players.filter((pp, ii) => ii < idx && pp.team === 'radiant').length : laning.players.filter((pp, ii) => ii < idx && pp.team === 'dire').length;
                const color = isRadiant ? (radiantColors[teamIdx % 5]) : (direColors[teamIdx % 5]);
                return {
                    name: p.hero_name,
                    type: 'line',
                    smooth: true,
                    data: snapshots.map(s => {
                        const hpPct = s.max_hp > 0 ? Math.round(s.hp / s.max_hp * 100) : 100;
                        return [s.time / 60, hpPct];
                    }),
                    lineStyle: { width: 1.5, color },
                    itemStyle: { color },
                    symbol: 'none',
                    areaStyle: { color: 'transparent' },
                };
            });

            laningHpTimelineInstance.setOption({
                title: { text: t('laning_hp_title'), textStyle: { color: '#ddd', fontSize: 14 }, left: 10, top: 5 },
                tooltip: {
                    trigger: 'axis',
                    backgroundColor: 'rgba(26,32,44,0.95)',
                    borderColor: '#4a5568',
                    textStyle: { color: '#e2e8f0', fontSize: 11 },
                    formatter: params => {
                        if (!params || params.length === 0) return '';
                        const min = params[0].value[0].toFixed(1);
                        let html = `<b>${min} ${t('laning_minutes')}</b>`;
                        const sorted = [...params].sort((a, b) => (b.value[1] || 0) - (a.value[1] || 0));
                        sorted.forEach(p => {
                            const pctColor = p.value[1] >= 70 ? '#48bb78' : p.value[1] >= 30 ? '#ecc94b' : '#fc8181';
                            html += `<div style="margin-top:2px;">${p.marker} ${p.seriesName}: <span style="color:${pctColor}">${p.value[1]}%</span></div>`;
                        });
                        return html;
                    },
                },
                legend: {
                    data: laning.players.map(p => p.hero_name),
                    textStyle: { color: '#aaa', fontSize: 10 },
                    top: 5, right: 10, type: 'scroll',
                },
                xAxis: { type: 'value', name: t('laning_minutes'), min: 0, max: 10, axisLabel: { color: '#aaa' }, nameTextStyle: { color: '#a0aec0' }, splitLine: { lineStyle: { color: '#2d3748' } } },
                yAxis: { type: 'value', name: t('laning_hp_percent'), min: 0, max: 100, axisLabel: { color: '#aaa', formatter: v => v + '%' }, nameTextStyle: { color: '#a0aec0' }, splitLine: { lineStyle: { color: '#2d3748' } } },
                series: series,
                backgroundColor: 'transparent',
                grid: { left: 60, right: 20, top: 60, bottom: 40 },
            }, true);
        }

        // --- Farm Charts ---
        function renderFarmReasonChart() {
            if (!farmReasonChart.value) return;
            if (!farmReasonChartInstance) {
                farmReasonChartInstance = echarts.init(farmReasonChart.value);
            }

            const isGold = farmMode.value === 'gold';
            const data = isGold ? farmData.value.gold_reasons : farmData.value.xp_reasons;
            if (!data || data.length === 0) {
                farmReasonChartInstance.setOption({ title: { text: t('farm_no_data'), left: 'center', top: 'center', textStyle: { color: '#a0aec0' } } }, true);
                return;
            }

            // Collect all reason keys
            const allReasons = new Set();
            data.forEach(p => Object.keys(p.reasons || {}).forEach(k => allReasons.add(k)));
            const reasons = [...allReasons];

            // Map backend Chinese keys to i18n keys for translation
            const reasonI18nMap = {
                '小兵': 'farm_reason_creeps', '英雄': 'farm_reason_heroes',
                '建筑': 'farm_reason_buildings', '肉山': 'farm_reason_roshan',
                '信使': 'farm_reason_courier', '其他': 'farm_reason_other',
                '出售': 'farm_reason_sell', '死亡': 'farm_reason_death',
                '买活': 'farm_reason_buyback',
            };
            const translateReason = (key) => {
                if (reasonI18nMap[key]) return t(reasonI18nMap[key]);
                const unknownMatch = key.match(/^未知\((\d+)\)$/);
                if (unknownMatch) return `${t('farm_reason_unknown')}(${unknownMatch[1]})`;
                return key;
            };

            const reasonColors = {
                '小兵': '#48bb78', '英雄': '#fc8181', '建筑': '#63b3ed',
                '肉山': '#ecc94b', '信使': '#f687b3', '其他': '#a0aec0',
                '出售': '#9f7aea', '死亡': '#718096', '买活': '#ed8936',
            };

            const heroes = data.map(p => p.hero_name);
            const translatedReasons = reasons.map(translateReason);
            const series = reasons.map((reason, i) => ({
                name: translatedReasons[i],
                type: 'bar',
                stack: 'total',
                data: data.map(p => (p.reasons || {})[reason] || 0),
                itemStyle: { color: reasonColors[reason] || '#a0aec0' },
            }));

            farmReasonChartInstance.setOption({
                backgroundColor: 'transparent',
                tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' },
                    backgroundColor: 'rgba(26,32,44,0.95)', borderColor: '#4a5568',
                    textStyle: { color: '#e2e8f0', fontSize: 12 },
                },
                legend: { data: translatedReasons, textStyle: { color: '#a0aec0', fontSize: 11 }, top: 5 },
                grid: { left: 60, right: 20, top: 40, bottom: 60 },
                xAxis: {
                    type: 'category', data: heroes,
                    axisLabel: { color: '#a0aec0', fontSize: 10, rotate: 30 },
                    axisLine: { lineStyle: { color: '#4a5568' } },
                },
                yAxis: {
                    type: 'value',
                    axisLabel: { color: '#a0aec0', fontSize: 11, formatter: v => (v/1000).toFixed(0) + 'k' },
                    splitLine: { lineStyle: { color: '#2d3748' } },
                },
                series,
            }, true);
        }

        function renderFarmLhChart() {
            if (!farmLhChart.value) return;
            if (!farmLhChartInstance) {
                farmLhChartInstance = echarts.init(farmLhChart.value);
            }

            const data = farmData.value.lh_timeline || [];
            if (data.length === 0) {
                farmLhChartInstance.setOption({ title: { text: t('farm_no_data'), left: 'center', top: 'center', textStyle: { color: '#a0aec0' } } }, true);
                return;
            }

            const playerColors = [
                '#3b82f6', '#10b981', '#a855f7', '#ecc94b', '#f97316',
                '#ef4444', '#ec4899', '#8b5cf6', '#06b6d4', '#84cc16',
            ];

            const maxLen = Math.max(...data.map(p => (p.lh_t || []).length));
            const times = Array.from({ length: maxLen }, (_, i) => i + ':00');

            const series = data.map((p, idx) => ({
                name: p.hero_name,
                type: 'line',
                data: p.lh_t || [],
                smooth: true,
                lineStyle: { color: playerColors[idx % 10], width: 2 },
                itemStyle: { color: playerColors[idx % 10] },
                symbol: 'none',
            }));

            farmLhChartInstance.setOption({
                backgroundColor: 'transparent',
                tooltip: {
                    trigger: 'axis',
                    backgroundColor: 'rgba(26,32,44,0.95)', borderColor: '#4a5568',
                    textStyle: { color: '#e2e8f0', fontSize: 11 },
                },
                legend: {
                    data: data.map(p => p.hero_name),
                    textStyle: { color: '#a0aec0', fontSize: 10 },
                    top: 5, type: 'scroll',
                },
                grid: { left: 50, right: 20, top: 40, bottom: 30 },
                xAxis: {
                    type: 'category', data: times,
                    axisLabel: { color: '#a0aec0', fontSize: 11 },
                    axisLine: { lineStyle: { color: '#4a5568' } },
                },
                yAxis: {
                    type: 'value',
                    axisLabel: { color: '#a0aec0', fontSize: 11 },
                    splitLine: { lineStyle: { color: '#2d3748' } },
                },
                series,
            }, true);
        }

        // --- Watchers ---
        watch(activeTab, async (tab, oldTab) => {
            // Dispose stale chart instances (v-if destroys DOM, old instances become invalid)
            if ((oldTab === 'economy' || oldTab === 'graphs') && economyChartInstance) { economyChartInstance.dispose(); economyChartInstance = null; }
            if ((oldTab === 'wards') && wardChartInstance) { wardChartInstance.dispose(); wardChartInstance = null; }
            if (oldTab === 'smokes' || oldTab === 'wards') {
                if (smokeMapInstance) { smokeMapInstance.dispose(); smokeMapInstance = null; }
                if (smokeChartInstance) { smokeChartInstance.dispose(); smokeChartInstance = null; }
            }
            if (oldTab === 'teamfights' && teamfightChartInstance) { teamfightChartInstance.dispose(); teamfightChartInstance = null; }
            if (oldTab === 'laning') {
                if (laningHeatmapInstance) { laningHeatmapInstance.dispose(); laningHeatmapInstance = null; }
                if (laningCsTimelineInstance) { laningCsTimelineInstance.dispose(); laningCsTimelineInstance = null; }
                if (laningHpTimelineInstance) { laningHpTimelineInstance.dispose(); laningHpTimelineInstance = null; }
            }
            if (oldTab === 'farm') {
                if (farmReasonChartInstance) { farmReasonChartInstance.dispose(); farmReasonChartInstance = null; }
                if (farmLhChartInstance) { farmLhChartInstance.dispose(); farmLhChartInstance = null; }
            }

            await nextTick();
            if (tab === 'economy' || tab === 'graphs') renderEconomyChart();
            if (tab === 'wards') { renderWardMap(); renderSmokeMap(); renderSmokeTimeline(); }
            if (tab === 'smokes') { renderSmokeMap(); renderSmokeTimeline(); }
            if (tab === 'teamfights') renderTeamfightMap();
            if (tab === 'laning') { renderLaningHeatmap(); renderLaningCsTimeline(); renderLaningHpTimeline(); }
            if (tab === 'farm') {
                if (farmMode.value === 'gold' || farmMode.value === 'xp') renderFarmReasonChart();
                if (farmMode.value === 'lh') renderFarmLhChart();
            }
        });

        let wardRenderTimer = null;
        watch([timelineWards, wardShowVision], () => {
            if (activeTab.value !== 'wards') return;
            clearTimeout(wardRenderTimer);
            wardRenderTimer = setTimeout(() => renderWardMap(), 16);
        });

        watch(wardTimelineEnabled, (enabled) => {
            if (enabled) wardTimelineTime.value = wardMatchDuration.value;
            if (activeTab.value === 'wards') renderWardMap();
        });

        // Smoke timeline debounced watcher
        let smokeRenderTimer = null;
        watch([smokeTimelineTime, smokeSelected], () => {
            if (activeTab.value !== 'smokes') return;
            clearTimeout(smokeRenderTimer);
            smokeRenderTimer = setTimeout(() => renderSmokeMap(), 16);
        });

        watch(smokeTimelineEnabled, (enabled) => {
            if (enabled) smokeTimelineTime.value = smokeMatchDuration.value;
            if (activeTab.value === 'smokes') renderSmokeMap();
        });

        // Economy chart mode watcher
        watch(economyMode, () => {
            if (activeTab.value === 'economy' || activeTab.value === 'graphs') renderEconomyChart();
        });

        // Teamfight replay debounced watcher
        let tfRenderTimer = null;
        watch(teamfightReplayTime, () => {
            if (activeTab.value !== 'teamfights') return;
            clearTimeout(tfRenderTimer);
            tfRenderTimer = setTimeout(() => renderTeamfightMap(), 16);
        });

        watch(selectedTeamfight, () => {
            // Stop playing and reset replay time
            clearInterval(teamfightPlayInterval);
            teamfightPlayInterval = null;
            teamfightPlaying.value = false;
            teamfightReplayTime.value = 0;
            if (teamfightChartInstance) { teamfightChartInstance.dispose(); teamfightChartInstance = null; }
            if (activeTab.value === 'teamfights') nextTick(() => renderTeamfightMap());
        });

        // Re-create interval when speed changes during playback
        watch(teamfightPlaySpeed, () => {
            if (!teamfightPlaying.value) return;
            clearInterval(teamfightPlayInterval);
            const tf = currentTeamfight.value;
            if (!tf) return;
            const tfDur = tf.end_time - tf.start_time;
            const stepMs = Math.round(100 / teamfightPlaySpeed.value);
            teamfightPlayInterval = setInterval(() => {
                teamfightReplayTime.value += 0.5;
                if (teamfightReplayTime.value >= tfDur) {
                    teamfightReplayTime.value = tfDur;
                    clearInterval(teamfightPlayInterval);
                    teamfightPlayInterval = null;
                    teamfightPlaying.value = false;
                }
            }, stepMs);
        });

        watch(farmMode, async () => {
            if (activeTab.value !== 'farm') return;
            if (farmReasonChartInstance) { farmReasonChartInstance.dispose(); farmReasonChartInstance = null; }
            if (farmLhChartInstance) { farmLhChartInstance.dispose(); farmLhChartInstance = null; }
            await nextTick();
            if (farmMode.value === 'gold' || farmMode.value === 'xp') renderFarmReasonChart();
            if (farmMode.value === 'lh') renderFarmLhChart();
        });

        watch(selectedLaningPlayer, () => {
            if (activeTab.value === 'laning') renderLaningHeatmap();
        });

        // --- Init ---
        onMounted(async () => {
            // Load hero/item data for icon lookups
            try {
                const [hRes, iRes] = await Promise.all([
                    fetch('/static/data/heroes.json'),
                    fetch('/static/data/items.json'),
                ]);
                heroData.value = await hRes.json();
                itemData.value = await iRes.json();
            } catch (e) {
                console.warn('Failed to load hero/item data:', e);
            }

            window.addEventListener('resize', () => {
                wardChartInstance?.resize();
                smokeMapInstance?.resize();
                smokeChartInstance?.resize();
                teamfightChartInstance?.resize();
                economyChartInstance?.resize();
                laningHeatmapInstance?.resize();
                laningCsTimelineInstance?.resize();
                laningHpTimelineInstance?.resize();
                farmReasonChartInstance?.resize();
                farmLhChartInstance?.resize();
                proLaningCsChart?.resize();
                proLaningGoldChart?.resize();
                proMidgameFarmChart?.resize();
                proLategameRadarChart?.resize();
            });
        });

        // Smoke result label helper
        function smokeResultLabel(result) {
            const labels = {
                kill: t('smoke_result_kill'),
                no_contact: t('smoke_result_no_contact'),
                traded: t('smoke_result_traded'),
                objective: t('smoke_result_objective'),
            };
            return labels[result] || result;
        }
        function smokeResultColor(result) {
            const colors = { kill: '#48bb78', no_contact: '#a0aec0', traded: '#ecc94b', objective: '#63b3ed' };
            return colors[result] || '#a0aec0';
        }

        function benchmarkColor(pct) {
            if (pct >= 0.8) return 'text-green-400';
            if (pct >= 0.6) return 'text-blue-400';
            if (pct >= 0.4) return 'text-yellow-400';
            if (pct >= 0.2) return 'text-orange-400';
            return 'text-red-400';
        }

        // Update document title and re-render charts on locale change
        watch(locale, async () => {
            document.title = t('app_title');
            // Re-render active charts with new locale strings
            await nextTick();
            const tab = activeTab.value;
            if (tab === 'economy') renderEconomyChart();
            if (tab === 'wards') renderWardMap();
            if (tab === 'smokes') { renderSmokeMap(); renderSmokeTimeline(); }
            if (tab === 'teamfights') renderTeamfightMap();
            if (tab === 'laning') { renderLaningHeatmap(); renderLaningCsTimeline(); renderLaningHpTimeline(); }
            if (tab === 'farm') {
                if (farmMode.value === 'gold' || farmMode.value === 'xp') renderFarmReasonChart();
                if (farmMode.value === 'lh') renderFarmLhChart();
            }
        });

        // Set initial document title
        document.title = t('app_title');

        return {
            locale, t, onLocaleChange, translateLaneName,
            analysisData, parseSource, parseSourceLabel, uploading, uploadError,
            activeTab, showAbilityBuilds, selectedTeamfight, wardFilters,
            wardTimelineEnabled, wardTimelineTime, wardShowVision,
            wardMatchDuration, timelineWards,
            smokeTimelineEnabled, smokeTimelineTime, smokeMatchDuration, smokeSelected,
            teamfightReplayTime, teamfightPlaying, teamfightPlaySpeed, currentTeamfightDuration,
            economyMode, economyChart,
            tabs, phaseLabels,
            wardMap, smokeMapRef, smokeChart, teamfightMap,
            overview, radiantPlayers, direPlayers,
            smokeData, teamfightData, currentTeamfight,
            wardStats, filteredWards,
            formatNumber, formatGameTime, shortAbility, getSkillColor,
            matchIdInput, fetchMatchById,
            handleFileSelect, handleDrop, loadDemo, resetAnalysis,
            toggleTeamfightPlay, toggleSmokeSelect, clearSmokeSelection,
            smokeResultLabel, smokeResultColor, benchmarkColor,
            heroIcon, heroImg, itemIcon, heroIconByName, abilityIcon, heroSlugByName,
            selectedLaningPlayer, combatMode, farmMode, objMode,
            laningHeatmap, laningCsTimeline, laningHpTimeline, farmReasonChart, farmLhChart,
            laningData, laningRadiant, laningDire,
            combatData, combatMatrix,
            farmData, purchaseData, benchmarkPlayers, benchmarkColumns,
            objectivesData, draftData, chatMessages,
            hasSkillBuilds, hasWardData, hasSmokeData, hasBenchmarkData,
            hasLaningData, hasTeamfightData, hasEconomyTimeline,
            hasItemBuilds, hasCombatData, hasFarmData, hasPurchaseData, hasObjectivesData,
            PLACEHOLDER_HERO, PLACEHOLDER_ITEM,
            renderMarkdown,
            aiCoachData, aiCoachLoading, aiCoachError, aiCoachPlayerSlot, requestAiCoach,
            aiCoachProgress, aiCoachStageLabel,
            // New overview/header properties
            matchHeader, overviewRadiant, overviewDire, teamTotals,
            timeAgo, regionName, formatNetWorth, teamSum, getItemPurchaseTime,
        };
    },
}).mount('#app');
