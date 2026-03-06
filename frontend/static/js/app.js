const { createApp, ref, computed, watch, onMounted, nextTick } = Vue;

const CDN = 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react';

createApp({
    setup() {
        // State
        const analysisData = ref(null);
        const parseSource = ref('');
        const uploading = ref(false);
        const uploadError = ref('');
        const activeTab = ref('items');
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
        const OBS_VISION_RADIUS = 10;       // 1600 / 16000 * 100
        const SENTRY_VISION_RADIUS = 5.625; // 900 / 16000 * 100

        // Smoke map state
        const smokeTimelineTime = ref(0);
        const smokeTimelineEnabled = ref(false);
        const smokeSelected = ref([]); // empty = show all; otherwise indices into timeline

        // Teamfight replay state
        const teamfightReplayTime = ref(0);
        const teamfightPlaying = ref(false);
        const teamfightPlaySpeed = ref(1); // 0.25, 0.5, 1, 2
        let teamfightPlayInterval = null;

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

        const tabs = [
            { key: 'economy', label: '经济曲线', icon: '📈' },
            { key: 'items', label: '出装分析', icon: '🗡' },
            { key: 'skills', label: '加点分析', icon: '📖' },
            { key: 'wards', label: '眼位分析', icon: '👁' },
            { key: 'smokes', label: '开雾分析', icon: '💨' },
            { key: 'teamfights', label: '团战分析', icon: '⚔' },
        ];

        const phaseLabels = {
            starting: '出门装',
            early: '前期 (0-10min)',
            mid: '中期 (10-25min)',
            late: '后期 (25min+)',
        };

        // --- CDN Icon URL Helpers ---
        function heroIcon(slug) {
            if (!slug) return '';
            return `${CDN}/heroes/icons/${slug}.png`;
        }

        function heroImg(slug) {
            if (!slug) return '';
            return `${CDN}/heroes/${slug}.png`;
        }

        function itemIcon(key) {
            if (!key) return '';
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
            return slug ? heroIcon(slug) : '';
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
            const t = wardTimelineTime.value;
            return filteredWards.value.filter(w => {
                if (w.time > t) return false;
                if (w.destroyed_time != null) {
                    if (w.destroyed_time <= t) return false;
                } else {
                    const expiry = w.time + (DEFAULT_WARD_LIFETIME[w.ward_type] || 360);
                    if (expiry <= t) return false;
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
            if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
            return String(n);
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
                uploadError.value = '请上传 .dem 格式的录像文件';
                return;
            }
            uploading.value = true;
            uploadError.value = '';
            const formData = new FormData();
            formData.append('file', file);
            try {
                const res = await fetch('/api/upload', { method: 'POST', body: formData });
                if (!res.ok) throw new Error((await res.json()).detail || '上传失败');
                const data = await res.json();
                parseSource.value = data.parse_source;
                const aRes = await fetch(`/api/match/${data.file_key}`);
                analysisData.value = (await aRes.json()).analysis;
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
                if (!data.has_parsed_data) {
                    // Request parse for detailed data
                    fetch(`/api/request-parse/${mid}`, { method: 'POST' }).catch(() => {});
                    uploadError.value = '已获取基础数据。详细数据（眼位/团战/经济曲线）需要等待 OpenDota 解析完成后重新获取。已自动请求解析。';
                }
            } catch (e) {
                uploadError.value = '获取比赛数据失败: ' + e.message;
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
            } catch (e) {
                uploadError.value = '加载演示数据失败: ' + e.message;
            } finally {
                uploading.value = false;
            }
        }

        function resetAnalysis() {
            analysisData.value = null;
            parseSource.value = '';
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
            [wardChartInstance, smokeMapInstance, smokeChartInstance, teamfightChartInstance, economyChartInstance]
                .forEach(c => c?.dispose());
            wardChartInstance = smokeMapInstance = smokeChartInstance = teamfightChartInstance = null;
        }

        // --- Chart Rendering ---
        function gameToMap(x, y) {
            const min = -8000, max = 8000, range = max - min;
            return [((x - min) / range) * 100, ((y - min) / range) * 100];
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
                const entry = {
                    value: [mx, my],
                    name: `${w.hero_name} @ ${w.time_str}`,
                    itemStyle: {
                        color: w.team === 'radiant'
                            ? (w.ward_type === 'observer' ? '#48bb78' : '#63b3ed')
                            : (w.ward_type === 'observer' ? '#fc8181' : '#f687b3'),
                    },
                };
                (w.ward_type === 'observer' ? obsData : sentData).push(entry);

                if (wardShowVision.value) {
                    const radius = w.ward_type === 'observer' ? OBS_VISION_RADIUS : SENTRY_VISION_RADIUS;
                    const fill = w.team === 'radiant'
                        ? (w.ward_type === 'observer' ? 'rgba(72,187,120,0.15)' : 'rgba(99,179,237,0.12)')
                        : (w.ward_type === 'observer' ? 'rgba(252,129,129,0.15)' : 'rgba(246,135,179,0.12)');
                    const stroke = w.team === 'radiant'
                        ? (w.ward_type === 'observer' ? 'rgba(72,187,120,0.4)' : 'rgba(99,179,237,0.3)')
                        : (w.ward_type === 'observer' ? 'rgba(252,129,129,0.4)' : 'rgba(246,135,179,0.3)');
                    visionData.push({ value: [mx, my, radius], fill, stroke });
                }
            });

            const series = [
                { name: '侦查', type: 'scatter', data: obsData, symbolSize: 14, symbol: 'circle', z: 10, itemStyle: { borderColor: '#ecc94b', borderWidth: 1.5 } },
                { name: '岗哨', type: 'scatter', data: sentData, symbolSize: 10, symbol: 'diamond', z: 10, itemStyle: { borderColor: '#63b3ed', borderWidth: 1.5 } },
            ];

            if (wardShowVision.value && visionData.length > 0) {
                series.unshift({
                    name: '视野范围', type: 'custom', data: visionData,
                    z: 5, silent: true,
                    renderItem(params, api) {
                        const center = api.coord([api.value(0), api.value(1)]);
                        const radiusVal = api.value(2);
                        const pixelSize = api.size([radiusVal, radiusVal]);
                        const d = visionData[params.dataIndex];
                        return {
                            type: 'circle',
                            shape: { cx: center[0], cy: center[1], r: pixelSize[0] },
                            style: { fill: d.fill, stroke: d.stroke, lineWidth: 1 },
                        };
                    },
                });
            }

            wardChartInstance.setOption({
                backgroundColor: '#0d1117',
                grid: { left: 0, right: 0, top: 0, bottom: 0 },
                tooltip: { trigger: 'item', formatter: p => p.name },
                graphic: [{
                    type: 'image', left: 0, top: 0, z: -1,
                    style: {
                        image: '/static/assets/minimap.png',
                        width: mapSize,
                        height: mapSize,
                        opacity: 0.5,
                    },
                }],
                xAxis: { type: 'value', min: 0, max: 100, show: false },
                yAxis: { type: 'value', min: 0, max: 100, show: false },
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
                const t = smokeTimelineTime.value;
                smokes = timeline.filter(s => s.time <= t);
            }
            if (smokeSelected.value.length > 0) {
                smokes = smokes.filter(s => smokeSelected.value.includes(timeline.indexOf(s)));
            }

            const pathsData = [];
            const startData = [];
            const endData = [];

            smokes.forEach(s => {
                const color = s.team === 'radiant' ? '#48bb78' : '#fc8181';
                const pathColor = s.team === 'radiant' ? 'rgba(72,187,120,0.6)' : 'rgba(252,129,129,0.6)';

                const [sx, sy] = gameToMap(s.x, s.y);
                startData.push({
                    value: [sx, sy],
                    name: `${s.team === 'radiant' ? '天辉' : '夜魇'} ${s.time_str}\n${s.num_players}人: ${s.participants.map(pp => pp.hero_name).join(', ')}`,
                    itemStyle: { color },
                });

                if (s.path && s.path.length > 0) {
                    const allPts = [[s.x, s.y], ...s.path];
                    const mapped = allPts.map(p => gameToMap(p[0], p[1]));
                    pathsData.push({ points: mapped, color: pathColor });

                    const last = s.path[s.path.length - 1];
                    const [ex, ey] = gameToMap(last[0], last[1]);
                    endData.push({
                        value: [ex, ey],
                        name: `烟破 ${s.end_time_str || ''}`,
                        itemStyle: { color },
                    });
                }
            });

            const series = [];

            if (pathsData.length > 0) {
                series.push({
                    name: '移动路径', type: 'custom',
                    data: pathsData.map((_, i) => ({ value: [i] })),
                    z: 3, silent: true,
                    renderItem(params, api) {
                        const d = pathsData[params.dataIndex];
                        const pixelPoints = d.points.map(pt => api.coord(pt));
                        return {
                            type: 'polyline',
                            shape: { points: pixelPoints },
                            style: { stroke: d.color, lineWidth: 2.5, lineDash: [6, 4], fill: 'none' },
                        };
                    },
                });
            }

            series.push({
                name: '开雾位置', type: 'scatter', data: startData,
                symbolSize: 14, symbol: 'circle', z: 10,
                itemStyle: { borderColor: '#fff', borderWidth: 1.5 },
            });

            if (endData.length > 0) {
                series.push({
                    name: '烟破位置', type: 'scatter', data: endData,
                    symbolSize: 10, symbol: 'triangle', z: 10,
                    itemStyle: { borderColor: '#fff', borderWidth: 1 },
                });
            }

            smokeMapInstance.setOption({
                backgroundColor: '#0d1117',
                grid: { left: 0, right: 0, top: 0, bottom: 0 },
                tooltip: { trigger: 'item', formatter: p => p.name },
                graphic: [{
                    type: 'image', left: 0, top: 0, z: -1,
                    style: {
                        image: '/static/assets/minimap.png',
                        width: mapSize,
                        height: mapSize,
                        opacity: 0.5,
                    },
                }],
                xAxis: { type: 'value', min: 0, max: 100, show: false },
                yAxis: { type: 'value', min: 0, max: 100, show: false },
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
                        return `${d.time_str} - ${d.team === 'radiant' ? '天辉' : '夜魇'}<br>${d.num_players}人: ${d.participants.map(pp => pp.hero_name).join(', ')}${d.duration ? '<br>持续: ' + d.duration + '秒' : ''}`;
                    },
                },
                grid: { left: 60, right: 20, top: 30, bottom: 40 },
                xAxis: {
                    type: 'value', name: '游戏时间 (分钟)',
                    min: 0, max: Math.ceil(duration / 60),
                    nameTextStyle: { color: '#a0aec0' },
                    axisLabel: { color: '#a0aec0' },
                    splitLine: { lineStyle: { color: '#2d3748' } },
                },
                yAxis: { type: 'category', data: ['天辉', '夜魇'], axisLabel: { color: '#a0aec0' }, splitLine: { show: false } },
                series: [{
                    type: 'scatter',
                    data: timeline.map(s => ({
                        value: [Math.round(s.time / 60 * 10) / 10, s.team === 'radiant' ? '天辉' : '夜魇'],
                        itemStyle: { color: s.team === 'radiant' ? '#48bb78' : '#fc8181' },
                    })),
                    symbolSize: (d) => {
                        const s = timeline.find(t =>
                            Math.round(t.time / 60 * 10) / 10 === d[0] &&
                            (d[1] === '天辉' ? 'radiant' : 'dire') === t.team);
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
            const t = teamfightReplayTime.value;
            const tfDuration = tf.end_time - tf.start_time;

            const heroPositions = [];
            const pathsData = [];
            const castData = [];

            tf.players.forEach(p => {
                const team = p.team || (p.player_slot < 128 ? 'radiant' : 'dire');
                const color = team === 'radiant' ? '#48bb78' : '#fc8181';
                const pathColor = team === 'radiant' ? 'rgba(72,187,120,0.4)' : 'rgba(252,129,129,0.4)';

                let mx, my;
                let isDead = false;

                if (p.positions && p.positions.length > 0) {
                    const pos = interpolatePosition(p.positions, t);
                    if (pos) {
                        [mx, my] = gameToMap(pos.x, pos.y);
                    } else {
                        [mx, my] = gameToMap(p.x, p.y);
                    }

                    // Build trail path up to current time
                    const trailPoints = p.positions
                        .filter(pp => pp.time_offset <= t)
                        .map(pp => gameToMap(pp.x, pp.y));
                    trailPoints.push([mx, my]);
                    if (trailPoints.length >= 2) {
                        pathsData.push({ points: trailPoints, color: pathColor });
                    }

                    isDead = p.deaths > 0 && p.death_time_offset != null && t >= p.death_time_offset;
                } else {
                    [mx, my] = gameToMap(p.x, p.y);
                    isDead = p.deaths > 0 && p.death_time_offset != null;
                }

                heroPositions.push({
                    value: [mx, my],
                    name: p.hero_name,
                    itemStyle: {
                        color,
                        opacity: isDead ? 0.3 : 1,
                        borderColor: team === 'radiant' ? '#276749' : '#9b2c2c',
                        borderWidth: 2,
                    },
                    label: {
                        show: true, position: 'right', fontSize: 10,
                        formatter: isDead ? `${p.hero_name} ☠` : p.hero_name,
                        color,
                    },
                });

                // Ability casts up to current time
                if (p.ability_casts) {
                    p.ability_casts.forEach(ac => {
                        if (ac.time_offset <= t) {
                            const [ax, ay] = gameToMap(ac.x, ac.y);
                            castData.push({
                                value: [ax, ay],
                                name: ac.ability_name,
                                itemStyle: { color: '#ecc94b', borderColor: color, borderWidth: 1.5 },
                            });
                        }
                    });
                }
            });

            const series = [];

            // Movement paths
            if (pathsData.length > 0) {
                series.push({
                    name: '移动路径', type: 'custom',
                    data: pathsData.map((_, i) => ({ value: [i] })),
                    z: 3, silent: true,
                    renderItem(params, api) {
                        const d = pathsData[params.dataIndex];
                        const pixelPoints = d.points.map(pt => api.coord(pt));
                        return {
                            type: 'polyline',
                            shape: { points: pixelPoints },
                            style: { stroke: d.color, lineWidth: 2, fill: 'none' },
                        };
                    },
                });
            }

            // Ability cast markers
            if (castData.length > 0) {
                series.push({
                    name: '技能释放', type: 'scatter', data: castData,
                    symbolSize: 10, symbol: 'diamond', z: 8,
                });
            }

            // Hero positions
            series.push({
                name: '英雄位置', type: 'scatter', data: heroPositions,
                symbolSize: 16, z: 10,
            });

            teamfightChartInstance.setOption({
                backgroundColor: '#0d1117',
                grid: { left: 0, right: 0, top: 0, bottom: 0 },
                tooltip: {
                    trigger: 'item',
                    formatter: p => {
                        const pl = tf.players.find(pp => pp.hero_name === p.name);
                        if (!pl) return p.name;
                        return `<b>${pl.hero_name}</b><br>K/D: ${pl.kills}/${pl.deaths}<br>输出: ${pl.damage_dealt}<br>承受: ${pl.damage_taken}<br>技能: ${pl.abilities_used.join(', ')}`;
                    },
                },
                graphic: [{
                    type: 'image', left: 0, top: 0, z: -1,
                    style: {
                        image: '/static/assets/minimap.png',
                        width: mapSize,
                        height: mapSize,
                        opacity: 0.5,
                    },
                }],
                xAxis: { type: 'value', min: 0, max: 100, show: false },
                yAxis: { type: 'value', min: 0, max: 100, show: false },
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
        const economyMode = ref('gold_diff'); // 'gold_diff', 'gold_total', 'xp_diff', 'xp_total'

        function renderEconomyChart() {
            if (!economyChart.value || !analysisData.value?.gold_xp_timeline) return;
            if (!economyChartInstance) {
                economyChartInstance = echarts.init(economyChart.value);
            }
            const timeline = analysisData.value.gold_xp_timeline;
            const times = timeline.map(s => s.time_str);
            const mode = economyMode.value;

            let series, yAxisConfig;
            if (mode === 'gold_diff') {
                const diffData = timeline.map(s => s.gold_diff);
                series = [{
                    name: '经济差',
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
                    name: '经验差',
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
                        name: '天辉',
                        type: 'line',
                        data: timeline.map(s => s[rKey]),
                        smooth: true,
                        lineStyle: { color: '#48bb78', width: 2 },
                        itemStyle: { color: '#48bb78' },
                    },
                    {
                        name: '夜魇',
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

            economyChartInstance.setOption({
                backgroundColor: 'transparent',
                tooltip: {
                    trigger: 'axis',
                    backgroundColor: '#1a202c',
                    borderColor: '#4a5568',
                    textStyle: { color: '#e2e8f0', fontSize: 12 },
                    formatter: params => {
                        const idx = params[0].dataIndex;
                        const s = timeline[idx];
                        let html = `<b>${s.time_str}</b><br/>`;
                        if (mode === 'gold_diff') {
                            const d = s.gold_diff;
                            const label = d >= 0 ? '天辉领先' : '夜魇领先';
                            html += `经济差: <span style="color:${d >= 0 ? '#48bb78' : '#fc8181'}">${d >= 0 ? '+' : ''}${(d/1000).toFixed(1)}k</span> (${label})`;
                        } else if (mode === 'xp_diff') {
                            const d = s.xp_diff;
                            const label = d >= 0 ? '天辉领先' : '夜魇领先';
                            html += `经验差: <span style="color:${d >= 0 ? '#48bb78' : '#fc8181'}">${d >= 0 ? '+' : ''}${(d/1000).toFixed(1)}k</span> (${label})`;
                        } else {
                            params.forEach(p => {
                                html += `${p.marker} ${p.seriesName}: ${(p.value/1000).toFixed(1)}k<br/>`;
                            });
                        }
                        return html;
                    },
                },
                legend: {
                    show: mode === 'gold_total' || mode === 'xp_total',
                    textStyle: { color: '#a0aec0' },
                    top: 5,
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

        // --- Watchers ---
        watch(activeTab, async (tab, oldTab) => {
            // Dispose stale chart instances (v-if destroys DOM, old instances become invalid)
            if (oldTab === 'economy' && economyChartInstance) { economyChartInstance.dispose(); economyChartInstance = null; }
            if (oldTab === 'wards' && wardChartInstance) { wardChartInstance.dispose(); wardChartInstance = null; }
            if (oldTab === 'smokes') {
                if (smokeMapInstance) { smokeMapInstance.dispose(); smokeMapInstance = null; }
                if (smokeChartInstance) { smokeChartInstance.dispose(); smokeChartInstance = null; }
            }
            if (oldTab === 'teamfights' && teamfightChartInstance) { teamfightChartInstance.dispose(); teamfightChartInstance = null; }

            await nextTick();
            if (tab === 'economy') renderEconomyChart();
            if (tab === 'wards') renderWardMap();
            if (tab === 'smokes') { renderSmokeMap(); renderSmokeTimeline(); }
            if (tab === 'teamfights') renderTeamfightMap();
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
            if (activeTab.value === 'economy') renderEconomyChart();
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
            });
        });

        // Smoke result label helper
        function smokeResultLabel(result) {
            const labels = { kill: '击杀', no_contact: '无接触', traded: '交换', objective: '目标' };
            return labels[result] || result;
        }
        function smokeResultColor(result) {
            const colors = { kill: '#48bb78', no_contact: '#a0aec0', traded: '#ecc94b', objective: '#63b3ed' };
            return colors[result] || '#a0aec0';
        }

        return {
            analysisData, parseSource, uploading, uploadError,
            activeTab, selectedTeamfight, wardFilters,
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
            smokeResultLabel, smokeResultColor,
            heroIcon, heroImg, itemIcon, heroIconByName, abilityIcon,
        };
    },
}).mount('#app');
