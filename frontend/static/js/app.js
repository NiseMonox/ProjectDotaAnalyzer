/**
 * Dota 2 Match Analyzer - the Vue app (one root component, in-DOM template
 * in index.html). No build step; loaded after:
 *   i18n.js     strings and code labels   -> t / tn / te / codeLabel below
 *   api.js      DotaApi: every /api call, errors as {code, detail, retryAt, ...}
 *   catalog.js  DotaCatalog: heroes / items / abilities names and icons
 *   game.js     DotaGame: map geometry, ward rules, unit names, skill build table
 *   coach.js    DotaCoach: AI coach progress, report shape, knowledge sources
 *   ui.js       DotaUI: <ui-banner> <ui-empty> <ui-skeleton> <ui-toasts> <game-icon> <hero-tag>
 */
const { createApp, ref, computed, watch, onMounted, nextTick, provide } = Vue;

const app = createApp({
    setup() {
        // ===== i18n =====
        // Strings live in i18n.js; t / tn / te / codeLabel read them in the
        // current language, so everything re-renders when `locale` changes.
        const locale = ref(I18N.detect());
        DotaApi.setLang(locale.value);
        function t(key, params) {
            return I18N.t(locale.value, key, params);
        }
        function tn(key, n, params) {
            return I18N.tn(locale.value, key, n, params);
        }
        function te(key) {
            return I18N.has(locale.value, key);
        }
        // Label of a backend code: codeLabel('game_mode', 'turbo'). Unknown
        // codes come back prettified, never raw or "undefined".
        function codeLabel(domain, value, params) {
            return I18N.code(locale.value, domain, value, params);
        }

        // State
        const analysisData = ref(null);
        const matchKey = ref('');
        // 'clarity' | 'opendota' | 'mock' | 'unknown' (see codeLabel('source', ...))
        const parseSource = ref('');
        // false = OpenDota only has the scoreboard of this match (not parsed yet)
        const hasParsedData = ref(true);
        // GET /api/capabilities; null until the first answer
        const capabilities = ref(null);
        const capabilitiesFailed = ref(false);
        // The one flow in progress on the start page, or null:
        // { kind: 'upload' | 'fetch' | 'demo' | 'link', id? }
        const busy = ref(null);
        const uploading = computed(() => !!busy.value && busy.value.kind === 'upload');
        // Errors are kept as objects ({ error, source, retry, hintKey }) and
        // turned into text by errorText(), so they follow the language.
        const homeError = ref(null);    // start page
        const matchError = ref(null);   // inside the match view
        // 1 Hz clock, read only by the countdown of a rate-limit error
        const clock = ref(Date.now());
        const uploadProgress = ref({
            phase: '',          // 'uploading' | 'parsing' | 'loading' | ''
            percent: 0,
            loaded: 0,
            total: 0,
            speed: 0,           // bytes/sec
            eta: 0,             // seconds remaining for upload
            indeterminate: false, // true when total size is unknown (no Content-Length)
        });
        // Below this upload speed (bytes/sec) the UI shows the slow-upload hint
        const SLOW_UPLOAD_SPEED = 200000; // 200 KB/s
        const uploadIsSlow = computed(() =>
            uploadProgress.value.speed > 0 && uploadProgress.value.speed < SLOW_UPLOAD_SPEED
        );
        const activeTab = ref('overview');
        const selectedTeamfight = ref(1);

        // Bumped whenever a static catalog (heroes / items / abilities) has
        // loaded; the name and icon helpers read it so they re-render.
        const catalogVersion = ref(0);
        DotaCatalog.onChange(() => { catalogVersion.value++; });

        // Shared toast stack. A toast keeps what to say (an i18n key and its
        // parameters, or an error object), not the sentence, so it follows a
        // language switch: toast('some.key', 'info' | 'success' | 'warn',
        // params or () => params) and toastError(error).
        const toasts = DotaUI.createToasts();
        function toast(key, kind, params) {
            toasts.push({ key, params: params || null }, kind, undefined, 'key:' + key);
        }
        function toastError(error) {
            if (error) toasts.push({ error }, 'error', undefined, 'error:' + error.code);
        }

        // ===== State of the analysis tabs (reset in clearMatch) =====
        // Vision
        const wardFilters = ref({ observers: true, sentries: true, radiant: true, dire: true });
        const wardTimelineEnabled = ref(false);
        const wardTimelineTime = ref(0);
        const wardShowVision = ref(false);
        // Smokes
        const smokeTimelineEnabled = ref(false);
        const smokeTimelineTime = ref(0);
        const smokeSelected = ref([]);   // indexes into smokes.timeline; empty = all
        // Teamfights: playback position in seconds from the fight's start
        const teamfightReplayTime = ref(0);
        const teamfightPlaying = ref(false);
        const teamfightPlaySpeed = ref(2);   // 0.5 | 1 | 2 | 4 (1 = real time)
        // Chat: chat-wheel lines (phrases, voice lines, sprays) are part of the log; the filter hides them
        const chatShowWheel = ref(true);

        // AI coach (logic under "AI coach" below; reset in clearMatch)
        const coachSlot = ref(null);        // player picked in the coach tab
        const coachReport = ref(null);      // report on screen (schema version 5)
        const coachRefreshAt = ref(0);      // epoch ms from which a degraded report can be generated again (0 = now)
        const coachLoading = ref(false);
        const coachError = ref(null);       // { error, retry } like homeError
        const coachCancelled = ref(false);  // the user stopped waiting
        const coachFallback = ref(false);   // waiting on the JSON route: no progress events
        const coachProgress = ref(DotaCoach.newProgress());
        const coachOpenSources = ref({});   // evaluation key -> its source list is expanded
        const coachOpenEntries = ref({});   // '<evaluation key>|<entry id>' -> the text of that source is shown
        const coachEntries = ref({});       // '<lang>|<entry id>' -> { state: loading | ready | error, text, error }

        // Sub-views of the tabs
        const selectedLaningPlayer = ref(0);
        const economyMode = ref('gold_diff');   // gold_diff | gold_total | xp_diff | xp_total | player_gold | player_xp | player_lh
        const combatMode = ref('kills');    // kills | damage | feed | stats
        const farmMode = ref('gold');       // gold | xp | units | lh
        const objMode = ref('timeline');    // timeline | map | runes

        // Chart containers (template refs)
        const wardMap = ref(null);
        const smokeMapRef = ref(null);
        const smokeChart = ref(null);
        const teamfightMap = ref(null);
        const economyChart = ref(null);
        const laningHeatmap = ref(null);
        const laningCsTimeline = ref(null);
        const laningHpTimeline = ref(null);
        const farmReasonChart = ref(null);
        const farmLhChart = ref(null);
        const buildingMap = ref(null);

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

        // ===== Static catalogs: names and icons =====
        // DotaCatalog (catalog.js) holds heroes / items / abilities; these
        // wrappers make its lookups reactive and add the localized fallbacks.
        // In templates prefer <game-icon kind k label>: it shows a fallback
        // tile for an empty or unknown key and for a picture that fails.

        // Fallback tile of a key: the initials of its name, or "?" when the
        // name itself is only a placeholder (no hero slug, 'item_<id>').
        function iconTile(kind, key, name) {
            const isHero = kind === 'hero' || kind === 'hero-portrait';
            const nameless = isHero ? !key : (kind === 'item' && /^item_\d+$/.test(key || '') && !DotaCatalog.itemName(key));
            return DotaCatalog.fallbackTile(kind, nameless ? '' : name);
        }

        // Hero icon for chart tooltips (HTML strings): a real picture, or a "?" tile.
        function heroIcon(slug) {
            void catalogVersion.value;
            return DotaCatalog.heroIconUrl(slug) || DotaCatalog.fallbackTile('hero', '');
        }

        // Hero slug by display name (for data that only has hero_name)
        function heroSlugByName(name) {
            void catalogVersion.value;
            return DotaCatalog.heroSlugByName(name);
        }

        // --- Display names ---
        // Payloads name heroes, items and abilities in English; the catalogs
        // have the game's Chinese names, shown when the UI is Chinese.

        // hero_name can be a name, or a code when the hero is not known:
        // 'hero_<id>', 'unknown_<n>', '' (see codeLabel('hero', ...)).
        function heroName(value) {
            void catalogVersion.value;
            const name = value == null ? '' : String(value);
            const byId = /^hero_(\d+)$/.exec(name);
            if (byId) {
                const hero = DotaCatalog.heroById(byId[1]);
                return hero ? DotaCatalog.localName(hero, locale.value) : codeLabel('hero', name);
            }
            if (!name || /^unknown_\d+$/.test(name)) return codeLabel('hero', name);
            if (name.startsWith('npc_dota_hero_')) {
                const hero = DotaCatalog.heroBySlug(name.slice(14));
                return hero ? DotaCatalog.localName(hero, locale.value) : I18N.prettify(name);
            }
            return DotaCatalog.localName(DotaCatalog.heroByName(name), locale.value) || name;
        }

        // Item key -> name. `given` is the name the payload carried, used
        // when the catalog does not know the key.
        function itemName(key, given) {
            void catalogVersion.value;
            if (!key) return '';
            const known = DotaCatalog.itemName(key, locale.value);
            if (known) return known;
            if (/^item_\d+$/.test(key)) return codeLabel('item', key);
            if (key.startsWith('recipe_')) return t('items.recipe_of', { name: itemName(key.slice(7)) });
            if (given && given !== key && !/^Item \d+$/.test(given)) return given;
            return I18N.prettify(key);
        }

        // Template leftovers ("{s:bonus_damage}") must never reach the screen.
        function cleanGameText(text) {
            return String(text == null ? '' : text).replace(/\{[^}]*\}/g, '').replace(/\s{2,}/g, ' ').trim();
        }

        function isRawKey(text) {
            return /^[a-z0-9]+(_[a-z0-9]+)+$/.test(text);
        }

        // Ability key -> name. Talents take the name from the payload (the
        // backend resolves it for the match's patch; in Chinese its numbers
        // go into the catalog's wording); they have no icon.
        function abilityName(key, given) {
            void catalogVersion.value;
            if (key === 'special_bonus_attributes') return codeLabel('ability', key);
            const fromPayload = cleanGameText(given);
            const usable = fromPayload && fromPayload !== key && !isRawKey(fromPayload) ? fromPayload : '';
            if (DotaCatalog.isTalentKey(key || '')) {
                const localized = locale.value === 'zh' ? DotaCatalog.talentNameZh(key, usable) : '';
                return localized || usable || cleanGameText(DotaCatalog.abilityName(key, locale.value)) || I18N.prettify(key);
            }
            return DotaCatalog.abilityName(key, locale.value) || usable || (key ? I18N.prettify(key) : codeLabel('ability', ''));
        }

        // Scoreboard number of a player: 1-5 Radiant, 6-10 Dire.
        function playerNumber(player) {
            const slot = Number(player && player.player_slot) || 0;
            return slot >= 128 ? slot - 128 + 6 : slot + 1;
        }

        // Steam names can carry private-use glyphs (icons of the Steam font)
        // that no web font has: drop them instead of showing empty boxes.
        function cleanPlayerName(player) {
            return String((player && player.player_name) || '').replace(/[\uE000-\uF8FF]/g, '').trim();
        }

        function isAnonymous(player) {
            return !cleanPlayerName(player);
        }

        // player_name is '' for anonymous players: "Player N" from the slot.
        function playerName(player) {
            return cleanPlayerName(player) || t('match.player_n', { n: playerNumber(player) });
        }

        // Lookups behind <game-icon> (ui.js)
        provide('gameIcons', {
            url(kind, key) {
                void catalogVersion.value;
                if (kind === 'hero') return DotaCatalog.heroIconUrl(key);
                if (kind === 'hero-portrait') return DotaCatalog.heroImgUrl(key);
                if (kind === 'item') return DotaCatalog.itemIconUrl(key);
                if (kind === 'ability') return DotaCatalog.abilityIconUrl(key);
                return '';
            },
            label(kind, key) {
                void catalogVersion.value;
                if (kind === 'hero' || kind === 'hero-portrait') {
                    return DotaCatalog.localName(DotaCatalog.heroBySlug(key), locale.value) || codeLabel('hero', '');
                }
                if (kind === 'item') return itemName(key);
                if (kind === 'ability') return abilityName(key);
                return '';
            },
            tile: iconTile,
        });

        // --- Computed ---
        const overview = computed(() => analysisData.value?.match_overview || {});

        const radiantPlayers = computed(() =>
            (overview.value.players || []).filter(p => p.team === 'radiant')
        );

        const direPlayers = computed(() =>
            (overview.value.players || []).filter(p => p.team === 'dire')
        );

        // --- Match header ---
        // 'radiant' | 'dire' | null. null = unknown (truncated replay): never
        // invent a winner from a falsy radiant_win.
        const matchWinner = computed(() => {
            const o = overview.value;
            if (o.winner === 'radiant' || o.winner === 'dire') return o.winner;
            if (o.winner === undefined && typeof o.radiant_win === 'boolean' && !o.partial) return o.radiant_win ? 'radiant' : 'dire';
            return null;
        });

        // kind: radiant | dire | unknown (drives the banner colour)
        const matchResult = computed(() => {
            if (overview.value.pre_game_only) return { kind: 'unknown', label: t('match.pregame_only') };
            if (matchWinner.value === 'radiant') return { kind: 'radiant', label: t('radiant_victory') };
            if (matchWinner.value === 'dire') return { kind: 'dire', label: t('dire_victory') };
            return { kind: 'unknown', label: t('match.result_unknown') };
        });

        const gameModeLabel = computed(() => codeLabel('game_mode', overview.value.game_mode));

        // Region id 0 = not known (hidden)
        const regionLabel = computed(() => (Number(overview.value.region) > 0 ? codeLabel('region', overview.value.region) : ''));

        // Match ids are 64-bit on the wire but below 2^53, so a JS number is
        // exact; 0 = unknown (hidden).
        const matchIdText = computed(() => {
            const id = overview.value.match_id;
            return id && Number(id) > 0 ? String(id) : '';
        });

        const openDotaUrl = computed(() =>
            matchIdText.value && (parseSource.value === 'opendota' || parseSource.value === 'clarity')
                ? `https://www.opendota.com/matches/${matchIdText.value}`
                : ''
        );

        // Kill score of the banner. A payload can carry 0 : 0 beside a
        // scoreboard with kills (matches from before the API had team
        // scores): the kills of each side are the score then.
        const teamScores = computed(() => {
            const o = overview.value;
            let radiant = o.radiant_score;
            let dire = o.dire_score;
            if (!radiant && !dire) {
                const kills = [teamSum(radiantPlayers.value, 'kills'), teamSum(direPlayers.value, 'kills')];
                if (kills[0] || kills[1]) [radiant, dire] = kills;
            }
            return { radiant: radiant ?? '-', dire: dire ?? '-' };
        });

        // A replay cut before the horn has no game clock worth showing.
        const durationText = computed(() =>
            overview.value.pre_game_only ? '-' : (overview.value.duration_str || formatGameTime(overview.value.duration, '-'))
        );

        // false for a replay that ends before minute 5 (the backend answers
        // 409 coach_no_game); payloads without the field count as true.
        const coachAvailable = computed(() => overview.value.coach_available !== false);

        const scoreboardTeams = computed(() => [
            { id: 'radiant', players: radiantPlayers.value, won: matchWinner.value === 'radiant' },
            { id: 'dire', players: direPlayers.value, won: matchWinner.value === 'dire' },
        ]);

        // --- Positions (1 = carry ... 5 = hard support) ---
        // match_overview.players[].position: null (not derived: a replay too
        // short for it), or {number 1-5, confidence: high | medium | low,
        // method}. The backend derives it once, so the scoreboard, the coach
        // picker and a coach report name the same position.

        // Short chip: "1号位" / "Pos 1"; null = nothing to show. A detection
        // that is less than sure is marked; the tooltip says what it rests on.
        function positionChip(position) {
            const number = position ? Number(position.number) : 0;
            if (!Number.isInteger(number) || number < 1 || number > 5) return null;
            return {
                label: codeLabel('position_short', number),
                tone: DotaCoach.confidenceTone(position.confidence),
                unsure: position.confidence === 'medium' || position.confidence === 'low',
                title: `${codeLabel('position', number)} · ` + t('coach.position.title', {
                    confidence: codeLabel('confidence', position.confidence),
                    method: codeLabel('position_method', position.method),
                }),
            };
        }

        // Chip of every player by slot
        const positionChips = computed(() => {
            const chips = {};
            (overview.value.players || []).forEach(p => { chips[p.player_slot] = positionChip(p.position); });
            return chips;
        });

        // --- Start-page status line (from /api/capabilities) ---
        const statusChips = computed(() => {
            const cap = capabilities.value;
            // The last answer says nothing about a server that stopped answering.
            if (capabilitiesFailed.value) return [{ id: 'offline', kind: 'warn', text: t('home.status.offline'), title: '' }];
            if (!cap) return [];
            const chips = [];
            // game_patch is null until the server has looked the version up: hide it then
            if (cap.game_patch) chips.push({ id: 'patch', kind: 'plain', text: t('home.status.patch', { patch: cap.game_patch }), title: '' });
            chips.push(cap.replay_parser !== false
                ? { id: 'parser', kind: 'ok', text: t('home.status.parser_on'), title: '' }
                : { id: 'parser', kind: 'warn', text: t('home.status.parser_off'), title: t('upload.parser_off_text') });
            if (cap.ai_coach_llm) {
                // ai_coach_daily_remaining: null = no daily cap
                const left = cap.ai_coach_daily_remaining;
                const quota = left == null ? '' : ' · ' + t('home.status.quota', { n: left });
                chips.push({ id: 'coach', kind: left === 0 ? 'warn' : 'ok', text: t('home.status.coach_llm') + quota, title: '' });
            } else {
                chips.push({ id: 'coach', kind: 'plain', text: t('home.status.coach_rules'), title: t('home.status.coach_rules_title') });
            }
            if (cap.knowledge_mode) chips.push({ id: 'knowledge', kind: 'plain', text: codeLabel('knowledge_mode', cap.knowledge_mode), title: '' });
            return chips;
        });

        // Line under the spinner while a match is being loaded
        const busyLabel = computed(() => {
            const flow = busy.value;
            if (!flow) return '';
            if (flow.kind === 'fetch') return t('upload.fetching', { id: flow.id });
            if (flow.kind === 'demo') return t('upload.loading_demo');
            return t('upload.loading_match');
        });

        // ===== Analysis tabs: shared helpers =====

        // --- Teams. The payload's team is 'radiant' | 'dire'; null (unknown
        // owner), 'neutral' (creeps) and 'draw' get the neutral style. ---
        const TEAM_COLOR = { radiant: '#48bb78', dire: '#fc8181', none: '#a0aec0' };

        function teamId(team) {
            return team === 'radiant' || team === 'dire' ? team : 'none';
        }

        function teamColor(team) {
            return TEAM_COLOR[teamId(team)];
        }

        function teamClass(team) {
            return 'team-' + teamId(team);
        }

        function teamLabel(team) {
            return codeLabel('team', team === 'radiant' || team === 'dire' || team === 'neutral' ? team : '');
        }

        // --- Players ---
        const playersBySlot = computed(() => {
            const map = {};
            (overview.value.players || []).forEach(p => { map[p.player_slot] = p; });
            return map;
        });

        function playerBySlot(slot) {
            return playersBySlot.value[slot] || null;
        }

        // Hero slug of an entry that may only carry a hero name or a slot.
        function slugOf(entry) {
            if (!entry) return '';
            if (entry.hero_slug) return entry.hero_slug;
            const player = playerBySlot(entry.player_slot);
            if (player && player.hero_slug && (!entry.hero_name || entry.hero_name === player.hero_name)) return player.hero_slug;
            return heroSlugByName(entry.hero_name);
        }

        // --- Numbers and times ---

        // Game clock of an event: the payload's time_str, else from `time`.
        function eventTime(entry) {
            if (!entry) return '';
            return entry.time_str || formatGameTime(entry.time, '-');
        }

        function isMissing(n) {
            return n == null || n === '' || !Number.isFinite(Number(n));
        }

        // A count: 0 is a real value, a missing one is a dash.
        function formatCount(n) {
            return isMissing(n) ? '-' : String(Math.round(Number(n)));
        }

        // 1234 -> 1.2k, -2403 -> -2.4k, 0 -> 0, missing -> '-'
        function formatK(n) {
            if (isMissing(n)) return '-';
            const v = Number(n);
            return Math.abs(v) >= 1000 ? (v / 1000).toFixed(1) + 'k' : String(Math.round(v));
        }

        function formatSigned(n) {
            if (isMissing(n)) return '-';
            const v = Number(n);
            return v > 0 ? '+' + formatK(v) : formatK(v);
        }

        function deltaClass(n) {
            return n > 0 ? 'delta-up' : n < 0 ? 'delta-down' : 'delta-zero';
        }

        // --- Units that are not heroes (kill feed, objectives) ---

        // "Dire mid tier 1 tower" from DotaGame.parseBuilding()
        function buildingLabel(info) {
            if (!info) return '';
            const team = codeLabel('team', info.side);
            const lane = info.lane ? codeLabel('map_lane', info.lane) : '';
            if (info.kind === 'tower' && info.tier === 4) return t('building.tower4', { team });
            if (info.kind === 'tower') return t('building.tower', { team, lane, tier: codeLabel('tower_tier', info.tier) });
            if (info.kind === 'barracks_melee') return t('building.barracks_melee', { team, lane });
            if (info.kind === 'barracks_ranged') return t('building.barracks_ranged', { team, lane });
            if (info.kind === 'ancient') return t('building.ancient', { team });
            return t('building.other', { team });
        }

        // `team` is used when the unit name does not carry a side (the fountain).
        function unitLabel(name, team) {
            const unit = DotaGame.classifyUnit(name);
            if (!unit) return '';
            const side = unit.side || (team === 'radiant' || team === 'dire' ? team : '');
            if (unit.kind === 'building') {
                return unit.building ? buildingLabel(unit.building) : t('building.other', { team: codeLabel('team', side) });
            }
            if (unit.kind === 'hero') {
                return DotaCatalog.localName(DotaCatalog.heroBySlug(unit.slug), locale.value) || I18N.prettify(name);
            }
            if (unit.kind === 'unknown') return I18N.prettify(name);
            const sided = `unit.${unit.kind}_of`;
            return side && te(sided) ? t(sided, { team: codeLabel('team', side) }) : t('unit.' + unit.kind);
        }

        // Who made a kill: the hero, else the unit (tower, creeps, fountain), else the side.
        function killerLabel(kill) {
            if (!kill) return '';
            if (kill.killer) return heroName(kill.killer);
            return unitLabel(kill.killer_unit, kill.killer_team)
                || (kill.killer_team ? teamLabel(kill.killer_team) : t('combat.unknown_killer'));
        }

        // A kill row has the colour of the side that scored; a deny and a
        // kill without a side (neutral creeps) stay neutral.
        function killRowClass(kill) {
            return 'kill-' + (kill && !kill.deny ? teamId(kill.killer_team) : 'none');
        }

        // counted false = a probable reincarnation (Wraith King, Aegis): shown,
        // but it is no death and moves no score. confirmed false with counted
        // true = a death the source could only infer.
        function killTitle(kill) {
            if (!kill) return null;
            if (kill.counted === false) return t('combat.uncounted_title');
            if (kill.confirmed === false) return t('combat.unconfirmed_title');
            return null;
        }

        function killAssists(kill) {
            return ((kill && kill.assist_slots) || []).map(slot => {
                const player = playerBySlot(slot);
                return player ? { slot, name: heroName(player.hero_name), slug: player.hero_slug || '' } : null;
            }).filter(Boolean);
        }

        // --- Why a section has nothing to show ---
        // { title, text }. The text names the real reason: the replay is not
        // parsed yet, the recording stops before the game, the source never
        // has this data, or nothing of the kind happened in the match.
        const OPENDOTA_MISSING = new Set(['smokes']);

        function emptyState(section) {
            let reason = 'none';
            if (overview.value.pre_game_only) reason = 'pregame';
            else if (parseSource.value === 'opendota' && !hasParsedData.value && section !== 'benchmarks') reason = 'unparsed';
            else if (parseSource.value === 'opendota' && OPENDOTA_MISSING.has(section)) reason = 'opendota';
            else if (section === 'benchmarks') reason = 'replay';
            const own = `empty.${section}.${reason}`;
            return {
                title: t(`empty.${section}.title`),
                text: te(own) ? t(own) : t('empty.reason.' + reason),
            };
        }

        // --- Sortable tables ---
        // sortState[table] = { key, dir }: first click sorts descending, the
        // second ascending, the third restores the payload order.
        const sortState = ref({});

        function toggleSort(table, key) {
            const current = sortState.value[table];
            let next = { key, dir: -1 };
            if (current && current.key === key) next = current.dir === -1 ? { key, dir: 1 } : null;
            sortState.value = { ...sortState.value, [table]: next };
        }

        function sortAria(table, key) {
            const current = sortState.value[table];
            if (!current || current.key !== key) return 'none';
            return current.dir === 1 ? 'ascending' : 'descending';
        }

        // valueOf(row, key) defaults to row[key]; missing values sort last.
        function sortRows(table, rows, valueOf) {
            const list = rows || [];
            const current = sortState.value[table];
            if (!current) return list;
            const read = valueOf || ((row, key) => row[key]);
            return [...list].sort((a, b) => {
                const x = read(a, current.key);
                const y = read(b, current.key);
                const xMissing = x == null || x === '' || (typeof x === 'number' && Number.isNaN(x));
                const yMissing = y == null || y === '' || (typeof y === 'number' && Number.isNaN(y));
                if (xMissing || yMissing) return xMissing === yMissing ? 0 : (xMissing ? 1 : -1);
                if (typeof x === 'string' || typeof y === 'string') return String(x).localeCompare(String(y)) * current.dir;
                return (x - y) * current.dir;
            });
        }

        // ===== Economy tab =====
        const hasEconomyTimeline = computed(() => (analysisData.value?.gold_xp_timeline || []).length > 0);

        const economyModes = computed(() => [
            { id: 'gold_diff', label: t('economy_gold_diff') },
            { id: 'gold_total', label: t('economy_gold_total') },
            { id: 'xp_diff', label: t('economy_xp_diff') },
            { id: 'xp_total', label: t('economy_xp_total') },
            { id: 'player_gold', label: t('economy_player_gold'), divider: true },
            { id: 'player_xp', label: t('economy_player_xp') },
            { id: 'player_lh', label: t('economy_player_lh') },
        ]);

        // ===== Items and purchases tabs =====
        const itemBuilds = computed(() => analysisData.value?.item_builds || []);
        const hasItemBuilds = computed(() => itemBuilds.value.length > 0);
        const ITEM_PHASES = ['starting', 'early', 'mid', 'late'];
        // The lists by phase come from the purchase log (none for an unparsed OpenDota match).
        const itemsHaveTimeline = computed(() =>
            itemBuilds.value.some(p => Object.values(p.item_phases || {}).some(items => (items || []).length > 0))
        );

        // The four phases in game order (then any other phase a backend adds).
        function itemPhases(player) {
            const phases = (player && player.item_phases) || {};
            const extra = Object.keys(phases).filter(id => !ITEM_PHASES.includes(id));
            return [...ITEM_PHASES, ...extra].map(id => ({ id, items: phases[id] || [] }));
        }

        // Name of a purchase line; an entry without a key is still a line.
        function itemLabel(key, given) {
            return itemName(key, given) || codeLabel('item', '');
        }

        const purchaseData = computed(() => analysisData.value?.purchases?.players || []);
        const hasPurchaseData = computed(() => purchaseData.value.some(p => (p.purchase_log || []).length > 0));

        // phases: 'starting' (time <= 0), then five-minute buckets 'N-M'.
        function purchaseBuckets(player) {
            return Object.entries((player && player.phases) || {}).map(([id, items]) => {
                const range = /^(\d+)-(\d+)$/.exec(id);
                return {
                    id,
                    label: range ? t('purchases.bucket', { from: range[1], to: range[2] }) : codeLabel('item_phase', id),
                    items: items || [],
                };
            });
        }

        function purchaseTip(item) {
            const name = itemLabel(item.key);
            const time = eventTime(item);
            return time && time !== '-' ? `${name} · ${time}` : name;
        }

        // ===== Skills tab =====
        const hasSkillBuilds = computed(() =>
            (analysisData.value?.skill_builds || []).some(p => p.skill_order && p.skill_order.length > 0)
        );

        // Casts per ability key of each player (abilities.by_player lists them by name).
        const castCounts = computed(() => {
            const bySlot = {};
            (analysisData.value?.abilities?.by_player || []).forEach(p => {
                const counts = {};
                Object.entries(p.ability_key_map || {}).forEach(([name, key]) => {
                    const n = (p.ability_counts || {})[name];
                    if (key && n != null) counts[key] = (counts[key] || 0) + n;
                });
                bySlot[p.player_slot] = counts;
            });
            return bySlot;
        });

        const SKILL_ROW_CLASSES = ['skill-q', 'skill-w', 'skill-e'];

        // One build table per player (see DotaGame.skillGrid).
        const skillBuilds = computed(() => {
            // Before 7.40 a talent used up the skill point of its level
            const talentsCostPoint = !DotaGame.patchAtLeast(overview.value.patch, 7, 40);
            return (analysisData.value?.skill_builds || []).map(player => {
                const grid = DotaGame.skillGrid(player.skill_order, { talentsCostPoint });
                const casts = castCounts.value[player.player_slot] || {};
                let basics = 0;
                grid.rows.forEach(row => {
                    if (row.kind === 'ultimate') row.cls = 'skill-r';
                    else if (row.kind === 'talent') row.cls = 'skill-talent';
                    else if (row.kind === 'attribute') row.cls = 'skill-attr';
                    else row.cls = SKILL_ROW_CLASSES[basics++] || 'skill-x';
                    row.casts = row.key && casts[row.key] != null ? casts[row.key] : null;
                });
                return {
                    ...player,
                    grid,
                    talents: grid.steps.filter(step => step.kind === 'talent'),
                    hasCasts: grid.rows.some(row => row.casts != null),
                };
            });
        });

        const skillsHint = computed(() => (parseSource.value === 'opendota'
            ? joinSentences(t('skills.hint'), t('skills.estimated_hint'))
            : t('skills.hint')));

        // OpenDota has no skill timestamps: the backend estimates them.
        function skillStepTime(step) {
            const time = step.time_str || formatGameTime(step.time, '');
            return time && parseSource.value === 'opendota' ? '≈' + time : time;
        }

        function skillRowLabel(row) {
            if (row.kind === 'talent') return t('skills.talents');
            if (row.kind === 'attribute') return codeLabel('ability', DotaGame.ATTRIBUTES_KEY);
            return abilityName(row.key, row.name);
        }

        function skillStepTitle(step) {
            const name = step.kind === 'attribute' ? codeLabel('ability', DotaGame.ATTRIBUTES_KEY) : abilityName(step.key, step.name);
            const key = step.kind === 'talent' ? 'skills.step_talent' : 'skills.step_title';
            return t(key, { n: step.n, name, level: step.level, time: skillStepTime(step) || '-' });
        }

        // ===== Vision tab =====
        const allWards = computed(() => analysisData.value?.wards?.all_wards || []);
        const hasWardData = computed(() => allWards.value.length > 0);

        // The payload counts placed and dewarded wards per team; how the
        // others ended (denied, expired, still standing) is counted here
        // from the list, by the same state the table shows.
        const wardTeams = computed(() => ['radiant', 'dire'].map(id => {
            const stats = (analysisData.value?.wards || {})[id] || {};
            const ended = {};
            allWards.value.forEach(w => {
                if (w.team !== id) return;
                const state = DotaGame.wardState(w);
                ended[state] = (ended[state] || 0) + 1;
            });
            return {
                id,
                obs: stats.total_obs || 0,
                sen: stats.total_sent || 0,
                dewardedObs: stats.dewarded_obs || 0,
                dewardedSen: stats.dewarded_sent || 0,
                // an average over no (finished) wards is no number
                avgObs: stats.avg_obs_lifetime > 0 ? stats.avg_obs_lifetime : null,
                avgSen: stats.avg_sent_lifetime > 0 ? stats.avg_sent_lifetime : null,
                // 'destroyed' (nobody named) is rare: listed only when it happened
                outcomes: DotaGame.WARD_STATES
                    .filter(state => state !== 'destroyed' || ended[state] > 0)
                    .map(state => ({ state, n: ended[state] || 0 })),
            };
        }));

        const wardStats = computed(() => ({
            totalObs: wardTeams.value.reduce((sum, team) => sum + team.obs, 0),
            totalSent: wardTeams.value.reduce((sum, team) => sum + team.sen, 0),
        }));

        // Radii of this match's patch, for the legend tooltip and the circles.
        const wardRadii = computed(() => ({
            observer: DotaGame.wardVision('observer', overview.value.patch),
            sentry: DotaGame.wardVision('sentry', overview.value.patch),
        }));

        const filteredWards = computed(() => allWards.value.filter(w => {
            if (!wardFilters.value.observers && w.ward_type === 'observer') return false;
            if (!wardFilters.value.sentries && w.ward_type !== 'observer') return false;
            if (!wardFilters.value.radiant && w.team === 'radiant') return false;
            if (!wardFilters.value.dire && w.team === 'dire') return false;
            return true;
        }));

        // Slider range: from the first ward (wards go down before the horn) to the end of the data.
        const wardTimeRange = computed(() => {
            let min = 0;
            let max = Number(analysisData.value?.wards?.duration || overview.value.duration) || 0;
            allWards.value.forEach(w => {
                if (w.time < min) min = w.time;
                if (w.time > max) max = w.time;
                if (w.destroyed_time != null && w.destroyed_time > max) max = w.destroyed_time;
            });
            return { min: Math.floor(min / 5) * 5, max: Math.ceil(max / 5) * 5 };
        });

        // Wards standing at the slider time. destroyed_time is the moment a
        // ward died or ran out; null = still standing when the data ends.
        const timelineWards = computed(() => {
            if (!wardTimelineEnabled.value) return filteredWards.value;
            const now = wardTimelineTime.value;
            return filteredWards.value.filter(w => w.time <= now && (w.destroyed_time == null || w.destroyed_time > now));
        });

        function wardStateOf(ward) {
            return DotaGame.wardState(ward);
        }

        // player_slot -1 / an empty hero name: the source does not know who placed it.
        function wardOwnerName(ward) {
            return ward && ward.hero_name ? heroName(ward.hero_name) : t('wards.unknown_owner');
        }

        // killer_hero_name: the enemy that dewarded it, or the ally that
        // denied it; '' when a unit other than a hero did it.
        function wardStateDetail(ward) {
            const state = DotaGame.wardState(ward);
            const time = formatGameTime(ward.destroyed_time, '-');
            if (state === 'dewarded' || state === 'denied') {
                return ward.killer_hero_name
                    ? t(`wards.${state}_by`, { time, hero: heroName(ward.killer_hero_name) })
                    : t(`wards.${state}_at`, { time });
            }
            if (state === 'destroyed') return t('wards.destroyed_at', { time });
            if (state === 'expired') return t('wards.expired_at', { time });
            return t('wards.standing');
        }

        // Tooltip of an outcome: what the state means.
        function wardStateTitle(ward) {
            return codeLabel('ward_state_title', DotaGame.wardState(ward));
        }

        // Icon of an outcome (an SVG path on a 24 x 24 grid, drawn as a
        // stroke): a cross, the game's "!" of a deny, a question mark (nobody
        // named), a clock, a tick.
        const WARD_STATE_ICON = {
            dewarded: 'M6 6l12 12M18 6L6 18',
            denied: 'M12 4v10M12 19v.5',
            destroyed: 'M8.5 9a3.5 3.5 0 116 2.4c-1.2 1.1-2.5 1.6-2.5 3.6M12 19v.5',
            expired: 'M12 3a9 9 0 100 18 9 9 0 000-18zM12 7.5V12l3 2',
            active: 'M5 12.5l4.5 4.5L19 7',
        };

        function wardStateIcon(state) {
            return WARD_STATE_ICON[state] || WARD_STATE_ICON.destroyed;
        }

        const wardHasDenied = computed(() => allWards.value.some(w => DotaGame.wardState(w) === 'denied'));

        function wardTypeName(ward) {
            return t(ward && ward.ward_type === 'observer' ? 'wards.observer_one' : 'wards.sentry_one');
        }

        // Seconds the ward stood (to the end of the data when it never fell).
        function wardLife(ward) {
            if (!ward) return null;
            if (ward.lifetime != null) return ward.lifetime;
            const end = ward.destroyed_time != null ? ward.destroyed_time : wardTimeRange.value.max;
            return Math.max(0, end - ward.time);
        }

        function wardMaxLife(ward) {
            return DotaGame.wardLifetime(ward && ward.ward_type);
        }

        function wardLifePct(ward) {
            const life = wardLife(ward);
            return life == null ? 0 : Math.max(0, Math.min(100, Math.round(life / wardMaxLife(ward) * 100)));
        }

        // ===== Smokes tab =====
        const smokeData = computed(() => analysisData.value?.smokes || {});
        const smokeTimeline = computed(() => smokeData.value.timeline || []);
        const hasSmokeData = computed(() => smokeTimeline.value.length > 0);

        // Result codes: kill | objective | traded | caught | no_contact
        const SMOKE_RESULT_COLOR = { kill: '#48bb78', objective: '#63b3ed', traded: '#ecc94b', caught: '#fc8181', no_contact: '#a0aec0' };
        const SMOKE_STATS = [['kill', 'kills'], ['objective', 'objectives'], ['traded', 'traded'], ['caught', 'caught'], ['no_contact', 'no_contact']];

        function smokeResultColor(result) {
            return SMOKE_RESULT_COLOR[result] || TEAM_COLOR.none;
        }

        const smokeTeams = computed(() => ['radiant', 'dire'].map(id => {
            const stats = smokeData.value[id] || {};
            return {
                id,
                total: stats.total || 0,
                successRate: Math.round(Number(stats.success_rate) || 0),
                parts: SMOKE_STATS.map(([code, field]) => ({ code, n: stats[field] || 0 })),
            };
        }));

        const smokeTimeRange = computed(() => {
            let min = 0;
            let max = Number(overview.value.duration) || 0;
            smokeTimeline.value.forEach(s => {
                if (s.time < min) min = s.time;
                const end = s.end_time != null ? s.end_time : s.time;
                if (end > max) max = end;
            });
            return { min: Math.floor(min / 5) * 5, max: Math.ceil(max / 5) * 5 };
        });

        // "Traded · 3 kills, 1 death · Tower". A plain "Kill" result is the
        // count itself ("2 kills"), not "Kill · 2 kills".
        function smokeResultText(smoke) {
            const trade = [];
            if (smoke.kills_gained > 0) trade.push(tn('smokes.kills', smoke.kills_gained));
            if (smoke.deaths_suffered > 0) trade.push(tn('smokes.deaths', smoke.deaths_suffered));
            const parts = smoke.result === 'kill' && trade.length ? [] : [codeLabel('smoke_result', smoke.result || 'unknown')];
            if (trade.length) parts.push(trade.join(t('common.list_sep')));
            if (smoke.objective_taken) parts.push(codeLabel('smoke_objective', smoke.objective_taken));
            return parts.join(' · ');
        }

        // Heroes under the smoke; `user` marks the one who used it.
        function smokeMembers(smoke) {
            return ((smoke && smoke.participants) || []).map((member, index) => ({
                slot: member.player_slot != null ? member.player_slot : 'i' + index,
                name: heroName(member.hero_name),
                slug: slugOf(member),
                user: smoke.player_slot != null && member.player_slot === smoke.player_slot,
            }));
        }

        function toggleSmokeSelect(idx) {
            const picked = smokeSelected.value;
            smokeSelected.value = picked.includes(idx) ? picked.filter(i => i !== idx) : [...picked, idx];
        }

        function clearSmokeSelection() {
            smokeSelected.value = [];
        }

        // ===== Teamfights tab =====
        const teamfightData = computed(() => analysisData.value?.teamfights || []);
        const hasTeamfightData = computed(() => teamfightData.value.length > 0);
        const currentTeamfight = computed(() => teamfightData.value.find(tf => tf.index === selectedTeamfight.value) || null);

        // Playback window in seconds from the fight's (padded) start: positions
        // of replay sources begin a little before it and run past its end.
        // hasPositions is false for OpenDota, which only records where heroes
        // died (and has no damage taken per fight either).
        const tfWindow = computed(() => {
            const tf = currentTeamfight.value;
            if (!tf) return { min: 0, max: 0, hasPositions: false, hasDamageTaken: false };
            let min = 0;
            let max = Math.max(0, (tf.end_time || 0) - (tf.start_time || 0));
            let hasPositions = false;
            (tf.players || []).forEach(p => (p.positions || []).forEach(pos => {
                hasPositions = true;
                if (pos.time_offset < min) min = pos.time_offset;
                if (pos.time_offset > max) max = pos.time_offset;
            }));
            (tf.kill_timeline || []).forEach(kill => {
                if (kill.time_offset < min) min = kill.time_offset;
                if (kill.time_offset > max) max = kill.time_offset;
            });
            const hasDamageTaken = (tf.players || []).some(p => p.damage_taken > 0);
            return { min: Math.floor(min), max: Math.ceil(max), hasPositions, hasDamageTaken };
        });

        // gold_delta / xp_delta of a fight are Radiant minus Dire.
        function swingTeam(delta) {
            return delta > 0 ? 'radiant' : delta < 0 ? 'dire' : '';
        }

        function swingText(delta) {
            if (isMissing(delta)) return '-';
            if (!Number(delta)) return '0';
            return `${teamLabel(swingTeam(delta))} +${formatK(Math.abs(delta))}`;
        }

        // item_uses: { item key: times used in the fight }
        function itemUses(player) {
            return Object.entries((player && player.item_uses) || {})
                .filter(([, count]) => count > 0)
                .map(([key, count]) => ({ key, count }));
        }

        function interpolatePosition(positions, timeOffset) {
            if (!positions || positions.length === 0) return null;
            const first = positions[0];
            const last = positions[positions.length - 1];
            if (timeOffset <= first.time_offset) return { x: first.x, y: first.y };
            if (timeOffset >= last.time_offset) return { x: last.x, y: last.y };
            for (let i = 0; i < positions.length - 1; i++) {
                const a = positions[i], b = positions[i + 1];
                if (timeOffset >= a.time_offset && timeOffset <= b.time_offset) {
                    const span = b.time_offset - a.time_offset;
                    // A teleport or a respawn is not walked: the hero stays, then is there.
                    if (DotaGame.isJump(a, b, span)) return timeOffset < b.time_offset ? { x: a.x, y: a.y } : { x: b.x, y: b.y };
                    const frac = span === 0 ? 0 : (timeOffset - a.time_offset) / span;
                    return { x: a.x + (b.x - a.x) * frac, y: a.y + (b.y - a.y) * frac };
                }
            }
            return { x: last.x, y: last.y };
        }

        // The death a hero is down from at `now` (seconds into the fight),
        // or null while it is alive: for sources without alive_intervals.
        // A hero is down from a death until it buys back: the buyback times
        // come from the scoreboard data. A source without them only counts
        // buybacks per fight: then the first `buybacks` deaths are taken as
        // bought back a few seconds later.
        function fallenAt(tf, player, deaths, now) {
            const last = deaths.length ? deaths[deaths.length - 1] : null;
            if (!last) return null;
            const row = playerBySlot(player.player_slot);
            const buybacks = ((row && row.buyback_log) || []).map(b => Number(b.time) - Number(tf.start_time));
            if (buybacks.length) return buybacks.some(at => at >= last.time_offset && at <= now) ? null : last;
            return player.buybacks >= deaths.length && now - last.time_offset > 5 ? null : last;
        }

        // Whether a hero is down at `now`: null while it is alive, else
        // { since } = when it fell, in seconds into the fight (the start of
        // the playback for a hero that was dead before the fight began).
        // `deaths` are its counted deaths up to `now`, in time order.
        // alive_intervals (replay sources) know respawns and buybacks; the
        // playback runs a little past the fight they cover, so a death after
        // its end comes from the kill order. null = the source cannot tell
        // (OpenDota): the deaths and buybacks of the fight decide.
        function downAt(tf, player, deaths, now) {
            if (!Array.isArray(player.alive_intervals)) {
                const death = fallenAt(tf, player, deaths, now);
                return death ? { since: death.time_offset } : null;
            }
            const duration = Number(tf.duration) || 0;
            const late = deaths.filter(k => k.time_offset > duration).pop();
            if (late) return { since: late.time_offset };
            const life = DotaGame.lifeAt(player.alive_intervals, now, duration);
            return life.alive ? null : { since: life.since == null ? tfWindow.value.min : life.since };
        }

        // Playback: real time at 1x, ten frames a second.
        const TF_TICK_MS = 100;
        let teamfightTimer = null;

        function stopTeamfightPlay() {
            clearInterval(teamfightTimer);
            teamfightTimer = null;
            teamfightPlaying.value = false;
        }

        function startTeamfightTimer() {
            clearInterval(teamfightTimer);
            teamfightTimer = setInterval(() => {
                const end = tfWindow.value.max;
                const next = teamfightReplayTime.value + TF_TICK_MS / 1000 * teamfightPlaySpeed.value;
                if (next >= end) {
                    teamfightReplayTime.value = end;
                    stopTeamfightPlay();
                } else {
                    teamfightReplayTime.value = Math.round(next * 100) / 100;
                }
            }, TF_TICK_MS);
        }

        function toggleTeamfightPlay() {
            if (teamfightPlaying.value) {
                stopTeamfightPlay();
                return;
            }
            if (!currentTeamfight.value || !tfWindow.value.hasPositions) return;
            if (teamfightReplayTime.value >= tfWindow.value.max) teamfightReplayTime.value = tfWindow.value.min;
            teamfightPlaying.value = true;
            startTeamfightTimer();
        }

        // Jump to a moment of the fight (a line of the kill order).
        function seekTeamfight(offset) {
            stopTeamfightPlay();
            const range = tfWindow.value;
            teamfightReplayTime.value = Math.max(range.min, Math.min(range.max, Number(offset) || 0));
        }

        // Without positions there is nothing to play: show the end state.
        function resetTeamfightReplay() {
            stopTeamfightPlay();
            teamfightReplayTime.value = tfWindow.value.hasPositions ? tfWindow.value.min : tfWindow.value.max;
        }

        // ===== Laning tab =====
        const laningData = computed(() => analysisData.value?.laning || { players: [] });
        // A replay cut before the horn has rows (zeros, pre-game positions) but no lane to show.
        const hasLaningData = computed(() => !overview.value.pre_game_only
            && (laningData.value.players || []).some(p => p.lh_ten != null || p.gold_ten != null || (p.lane_pos || []).length > 0)
        );
        const laningTeams = computed(() => ['radiant', 'dire'].map(id => ({
            id,
            players: (laningData.value.players || []).filter(p => p.team === id),
        })));
        const selectedLaning = computed(() =>
            (laningData.value.players || []).find(p => p.player_slot === selectedLaningPlayer.value) || null
        );
        // HP / position samples every five seconds: replay sources only.
        const laningHasSnapshots = computed(() =>
            (laningData.value.players || []).some(p => (p.laning_snapshots || []).length > 0)
        );
        // Damage traded and regen used in lane: replay sources only (OpenDota sends zeros).
        const laningHasDetails = computed(() =>
            (laningData.value.players || []).some(p =>
                (p.damage_events || []).length > 0 || (p.regen_events || []).length > 0
                || p.total_damage_dealt_laning > 0 || p.total_damage_taken_laning > 0 || p.regen_items_used > 0)
        );

        // lane_efficiency_applicable is false for a game mode the figure does
        // not describe (Turbo): the column then holds dashes, and says why.
        const laningEfficiencyNote = computed(() =>
            laningData.value.lane_efficiency_applicable === false ? t('laning.efficiency_na', { mode: gameModeLabel.value }) : ''
        );

        // Why the damage / regen / HP columns are missing: OpenDota never has
        // them, a truncated replay may stop before them.
        const laningDetailsNote = computed(() => {
            if (laningHasDetails.value) return '';
            if (parseSource.value === 'opendota') return t('laning.no_details');
            return t(overview.value.partial ? 'laning.no_details_partial' : 'laning.no_details_none');
        });

        // The selected hero's laning damage to heroes by source (damage_events:
        // inflictor is 'attack', an ability key or an item key) and the regen
        // it used (regen_events). Replay sources only.
        const laningBreakdown = computed(() => {
            void catalogVersion.value;
            const player = selectedLaning.value;
            const damage = {};
            ((player && player.damage_events) || []).forEach(e => {
                const key = e.inflictor || 'attack';
                damage[key] = (damage[key] || 0) + (Number(e.damage) || 0);
            });
            const sources = Object.entries(damage)
                .filter(([, total]) => total > 0)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 8)
                .map(([key, total]) => {
                    if (key === 'attack') return { key, kind: 'attack', name: t('laning.attack'), total };
                    const isItem = !DotaCatalog.ability(key) && !!DotaCatalog.item(key);
                    return { key, kind: isItem ? 'item' : 'ability', name: isItem ? itemName(key) : abilityName(key), total };
                });
            const used = {};
            ((player && player.regen_events) || []).forEach(e => {
                if (e.item) used[e.item] = (used[e.item] || 0) + 1;
            });
            const regen = Object.entries(used)
                .sort((a, b) => b[1] - a[1])
                .map(([key, count]) => ({ key, count, name: itemLabel(key) }));
            return { sources, regen };
        });

        // lane_name is relative to the team: safe | mid | off | roaming | unknown.
        function laneLabel(player) {
            if (!player) return '';
            return codeLabel('lane', player.is_roaming ? 'roaming' : (player.lane_name || 'unknown'));
        }

        // lane_side is the absolute side of the map; shown for the side lanes only.
        function laneSideLabel(player) {
            const side = player && !player.is_roaming ? player.lane_side : '';
            return side === 'top' || side === 'bottom' ? codeLabel('lane_side', side) : '';
        }

        // Lane efficiency needs the gold at 10:00; without it there is no number.
        function efficiencyText(player) {
            if (!player || player.gold_ten == null || isMissing(player.lane_efficiency_pct)) return '-';
            return Math.round(player.lane_efficiency_pct) + '%';
        }

        function efficiencyClass(player) {
            if (efficiencyText(player) === '-') return 'cell-zero';
            const pct = player.lane_efficiency_pct;
            return pct >= 60 ? 'text-green-400' : pct >= 40 ? 'text-yellow-400' : 'text-red-400';
        }

        // ===== Combat tab =====
        const combatData = computed(() => analysisData.value?.combat || {});
        const combatKills = computed(() => combatData.value.kills || []);

        // Per-player numbers of the scoreboard that no other tab shows.
        const combatStats = computed(() => {
            const best = (counts) => {
                const reached = Object.keys(counts || {}).map(Number).filter(n => Number.isFinite(n) && counts[n] > 0);
                return reached.length ? Math.max(...reached) : 0;
            };
            const rows = (overview.value.players || []).map(p => ({
                player_slot: p.player_slot,
                hero_name: p.hero_name,
                hero_slug: p.hero_slug,
                team: p.team,
                stuns: typeof p.stuns === 'number' ? Math.round(p.stuns * 10) / 10 : null,
                apm: typeof p.actions_per_min === 'number' ? p.actions_per_min : null,
                participation: typeof p.teamfight_participation === 'number' ? Math.round(p.teamfight_participation * 100) : null,
                buybacks: typeof p.buyback_count === 'number' ? p.buyback_count : null,
                multi: best(p.multi_kills),      // { "2": n, "3": n, ... } -> the largest
                streak: best(p.kill_streaks),    // { "3": n, ... } -> the longest
            }));
            // A source without these numbers sends zeros for everyone.
            return rows.some(r => r.stuns || r.apm || r.participation || r.buybacks || r.multi || r.streak) ? rows : [];
        });

        const combatStatColumns = computed(() => [
            { key: 'stuns', label: t('combat.col_stuns'), title: t('combat.col_stuns_title') },
            { key: 'apm', label: t('combat.col_apm'), title: t('combat.col_apm_title') },
            { key: 'participation', label: t('combat.col_participation'), title: t('combat.col_participation_title') },
            { key: 'buybacks', label: t('combat.col_buybacks') },
            { key: 'multi', label: t('combat.col_multi') },
            { key: 'streak', label: t('combat.col_streak') },
        ]);

        const sortedCombatStats = computed(() => sortRows('combat', combatStats.value));

        // The game announces every streak of ten or more as "Beyond godlike".
        function streakLabel(n) {
            if (!n) return '-';
            return n > 10 ? t('combat.streak_beyond', { label: codeLabel('kill_streak', 10), n }) : codeLabel('kill_streak', n);
        }

        function matrixHasData(matrix) {
            return !!matrix && Array.isArray(matrix.matrix) && matrix.matrix.some(row => (row || []).some(value => value > 0));
        }

        const hasCombatData = computed(() =>
            combatKills.value.length > 0 || matrixHasData(combatData.value.kill_matrix) || matrixHasData(combatData.value.damage_matrix)
        );

        const combatModes = computed(() => {
            const modes = [
                { id: 'kills', label: t('combat_kill_matrix') },
                { id: 'damage', label: t('combat_damage_matrix') },
            ];
            if (combatKills.value.length) modes.push({ id: 'feed', label: t('combat.feed') });
            if (combatStats.value.length) modes.push({ id: 'stats', label: t('combat.stats') });
            return modes;
        });

        // Rows deal, columns receive. heroes[] are display names in player order.
        const combatMatrix = computed(() => {
            const source = combatMode.value === 'damage' ? combatData.value.damage_matrix
                : combatMode.value === 'kills' ? combatData.value.kill_matrix : null;
            if (!source || !Array.isArray(source.heroes) || !source.heroes.length || !Array.isArray(source.matrix)) return null;
            const byName = {};
            (overview.value.players || []).forEach(p => { byName[p.hero_name] = p; });
            let previousTeam = null;
            const heroes = source.heroes.map((name, index) => {
                const player = byName[name];
                const team = player ? player.team : '';
                const teamStart = index > 0 && team !== previousTeam;
                previousTeam = team;
                return {
                    name: player ? heroName(name) : (unitLabel(name) || heroName(name)),
                    slug: player ? (player.hero_slug || '') : heroSlugByName(name),
                    team,
                    teamStart,
                };
            });
            const size = heroes.length;
            const columnTotals = new Array(size).fill(0);
            let max = 1;
            const rows = heroes.map((hero, i) => {
                const values = heroes.map((other, j) => Number((source.matrix[i] || [])[j]) || 0);
                values.forEach((value, j) => {
                    columnTotals[j] += value;
                    if (value > max) max = value;
                });
                return { values, total: values.reduce((sum, value) => sum + value, 0) };
            });
            return { heroes, rows, columnTotals, max, grandTotal: columnTotals.reduce((sum, value) => sum + value, 0) };
        });

        // ===== Farm tab =====
        const farmData = computed(() => analysisData.value?.farm || {});

        function farmHasReasons(mode) {
            const list = mode === 'xp' ? farmData.value.xp_reasons : farmData.value.gold_reasons;
            return (list || []).some(p => p.reasons && Object.keys(p.reasons).length > 0);
        }

        const hasFarmData = computed(() =>
            (farmData.value.unit_kills || []).length > 0 || farmHasReasons('gold') || farmHasReasons('xp')
            || (farmData.value.lh_timeline || []).some(p => (p.lh_t || []).length > 0)
        );

        const farmModes = computed(() => [
            { id: 'gold', label: t('farm_gold_sources') },
            { id: 'xp', label: t('farm_xp_sources') },
            { id: 'units', label: t('farm_unit_kills') },
            { id: 'lh', label: t('farm_lh_timeline') },
        ]);

        const UNIT_KILL_COLUMNS = [
            ['hero_kills', 'farm_heroes'], ['lane_kills', 'farm_creeps', 'farm.creeps_title'], ['neutral_kills', 'farm_neutrals'],
            ['ancient_kills', 'farm_ancients'], ['tower_kills', 'farm_towers'], ['roshans_killed', 'farm_roshan'],
            ['courier_kills', 'farm_courier'],
        ];

        // A count the source cannot provide is null for everyone (a
        // scoreboard-only match has hero kills and nothing else): no column.
        const unitKillColumns = computed(() => UNIT_KILL_COLUMNS
            .filter(([key]) => (farmData.value.unit_kills || []).some(row => row[key] != null))
            .map(([key, label, title]) => ({ key, label: t(label), title: title ? t(title) : '' })));

        const unitKillsNote = computed(() => {
            if (unitKillColumns.value.length === UNIT_KILL_COLUMNS.length) return '';
            return joinSentences(t('farm.no_breakdown'), emptyState('farm').text);
        });

        const sortedUnitKills = computed(() => sortRows('units', farmData.value.unit_kills || []));

        // gold_reasons / xp_reasons codes -> label. The Shrine of Wisdom
        // replaced the Wisdom Rune in 7.38; the code stayed the same.
        function reasonLabel(kind, code) {
            if (code === 'wisdom_rune' && !DotaGame.patchAtLeast(overview.value.patch, 7, 38)) {
                return codeLabel('xp_reason', 'wisdom_rune_legacy');
            }
            return codeLabel(kind === 'xp' ? 'xp_reason' : 'gold_reason', code);
        }

        // ===== Benchmarks tab =====
        const benchmarkPlayers = computed(() => analysisData.value?.benchmarks_analysis?.players || []);
        const hasBenchmarkData = computed(() =>
            benchmarkPlayers.value.some(p => p.benchmarks && Object.keys(p.benchmarks).length > 0)
        );
        const BENCHMARK_KEYS = [
            'gold_per_min', 'xp_per_min', 'kills_per_min', 'deaths_per_min', 'assists_per_min', 'last_hits_per_min',
            'denies_per_min', 'hero_damage_per_min', 'hero_healing_per_min', 'tower_damage',
        ];

        const benchmarkColumns = computed(() => BENCHMARK_KEYS
            .filter(key => benchmarkPlayers.value.some(p => p.benchmarks && p.benchmarks[key]))
            .map(key => ({ key, label: codeLabel('benchmark', key), title: codeLabel('benchmark_title', key) })));

        // Sorted by percentile, not by the raw value.
        const sortedBenchmarks = computed(() =>
            sortRows('bench', benchmarkPlayers.value, (row, key) => {
                const entry = row.benchmarks && row.benchmarks[key];
                return entry ? entry.pct : null;
            })
        );

        function benchmarkValue(raw) {
            if (isMissing(raw)) return '-';
            const value = Number(raw);
            return Math.abs(value) < 10 && !Number.isInteger(value) ? value.toFixed(2) : String(Math.round(value));
        }

        function benchmarkPct(pct) {
            return isMissing(pct) ? '-' : Math.round(Number(pct) * 100) + '%';
        }

        function benchmarkColor(pct) {
            if (pct >= 0.8) return 'text-green-400';
            if (pct >= 0.6) return 'text-blue-400';
            if (pct >= 0.4) return 'text-yellow-400';
            if (pct >= 0.2) return 'text-orange-400';
            return 'text-red-400';
        }

        // ===== Objectives tab =====
        const objectivesData = computed(() => analysisData.value?.objectives_analysis || { objectives: [], runes: [] });
        const buildings = computed(() => (overview.value.buildings || []).filter(b => DotaGame.onMap(b.x, b.y)));

        const hasObjectivesData = computed(() =>
            (objectivesData.value.objectives || []).length > 0 || (objectivesData.value.runes || []).length > 0
            || buildings.value.some(b => b.destroyed_time != null)
        );

        const objModes = computed(() => [
            { id: 'timeline', label: t('objectives_timeline') },
            { id: 'map', label: t('objectives.map') },
            { id: 'runes', label: t('objectives_rune_stats') },
        ]);

        const OBJECTIVE_ICON = { blood: '🩸', tower: '🏰', barracks: '⛺', ancient: '👑', roshan: '🐉', aegis: '🛡', courier: '📦', tormentor: '🧊', other: '⚑' };

        // One line of the objectives timeline. `team` is the side credited
        // with the event; the subject shown is the hero, else that side.
        function objectiveRow(obj) {
            const type = String(obj.type || '');
            const row = {
                time: eventTime(obj), icon: OBJECTIVE_ICON.other, team: obj.team || '',
                heroName: obj.hero_name ? heroName(obj.hero_name) : '', heroSlug: obj.hero_slug || '',
                text: '', victimName: '', victimSlug: '', victimTeam: '', extra: '',
            };
            if (type === 'building_kill') {
                const info = DotaGame.parseBuilding(obj.key);
                const target = info ? buildingLabel(info) : (unitLabel(obj.key) || t('building.other', { team: codeLabel('team', '') }));
                // A building that fell to its own side was denied.
                row.text = t(info && info.side === obj.team ? 'obj.denied' : 'obj.destroyed', { target });
                row.icon = info && info.kind === 'ancient' ? OBJECTIVE_ICON.ancient
                    : info && info.kind !== 'tower' ? OBJECTIVE_ICON.barracks : OBJECTIVE_ICON.tower;
            } else if (type === 'CHAT_MESSAGE_FIRSTBLOOD') {
                row.icon = OBJECTIVE_ICON.blood;
                row.text = obj.victim_hero_name ? t('obj.first_blood_on') : t('obj.first_blood');
                if (obj.victim_hero_name) {
                    row.victimName = heroName(obj.victim_hero_name);
                    row.victimSlug = obj.victim_hero_slug || '';
                    row.victimTeam = obj.team === 'radiant' ? 'dire' : obj.team === 'dire' ? 'radiant' : '';
                }
            } else if (type === 'CHAT_MESSAGE_ROSHAN_KILL') {
                row.icon = OBJECTIVE_ICON.roshan;
                row.text = t('obj.roshan');
            } else if (type === 'CHAT_MESSAGE_AEGIS' || type === 'CHAT_MESSAGE_AEGIS_STOLEN' || type === 'CHAT_MESSAGE_DENIED_AEGIS') {
                row.icon = OBJECTIVE_ICON.aegis;
                row.text = t(type === 'CHAT_MESSAGE_AEGIS' ? 'obj.aegis' : type === 'CHAT_MESSAGE_AEGIS_STOLEN' ? 'obj.aegis_stolen' : 'obj.aegis_denied');
            } else if (type === 'CHAT_MESSAGE_COURIER_LOST') {
                row.icon = OBJECTIVE_ICON.courier;
                row.text = obj.courier_team ? t('obj.courier', { team: codeLabel('team', obj.courier_team) }) : t('obj.courier_plain');
                if (obj.value > 0) row.extra = t('obj.bounty', { gold: obj.value });
            } else if (type === 'CHAT_MESSAGE_MINIBOSS_KILL') {
                row.icon = OBJECTIVE_ICON.tormentor;
                row.text = t('obj.tormentor');
            } else if (type === 'CHAT_MESSAGE_TOWER_KILL' || type === 'CHAT_MESSAGE_TOWER_DENY' || type === 'CHAT_MESSAGE_BARRACKS_KILL') {
                // Old OpenDota matches list buildings without their name.
                row.icon = type === 'CHAT_MESSAGE_BARRACKS_KILL' ? OBJECTIVE_ICON.barracks : OBJECTIVE_ICON.tower;
                row.text = t(type === 'CHAT_MESSAGE_TOWER_DENY' ? 'obj.tower_denied' : type === 'CHAT_MESSAGE_BARRACKS_KILL' ? 'obj.barracks_destroyed' : 'obj.tower_destroyed');
            } else {
                row.text = I18N.prettify(type.replace(/^CHAT_MESSAGE_/, '').toLowerCase()) || t('common.unknown');
            }
            return row;
        }

        const objectiveRows = computed(() => (objectivesData.value.objectives || []).map(objectiveRow));

        // Towers / barracks still standing per team (match_overview.buildings).
        const buildingTeams = computed(() => ['radiant', 'dire'].map(id => {
            const own = buildings.value.filter(b => b.team === id);
            const count = (type, standingOnly) => own.filter(b => b.type === type && (!standingOnly || b.destroyed_time == null)).length;
            return { id, towers: count('tower', true), towersTotal: count('tower'), barracks: count('barracks', true), barracksTotal: count('barracks') };
        }));

        // runes[].runes: { code: count }. Known codes in a fixed order, then any new one.
        const RUNE_ORDER = ['bounty', 'water', 'wisdom', 'double_damage', 'haste', 'illusion', 'invisibility', 'regeneration', 'arcane', 'shield'];

        function runeLabel(code) {
            // Since 7.38 the "wisdom" pickups are Shrines of Wisdom
            if (code === 'wisdom' && !DotaGame.patchAtLeast(overview.value.patch, 7, 38)) return codeLabel('rune', 'wisdom_legacy');
            return codeLabel('rune', code);
        }

        const runeColumns = computed(() => {
            const seen = new Set();
            (objectivesData.value.runes || []).forEach(r => Object.keys(r.runes || {}).forEach(code => seen.add(code)));
            const codes = [...RUNE_ORDER.filter(code => seen.has(code)), ...[...seen].filter(code => !RUNE_ORDER.includes(code)).sort()];
            return codes.map(code => ({ key: code, label: runeLabel(code), title: runeLabel(code) }));
        });

        const sortedRunes = computed(() =>
            sortRows('runes', objectivesData.value.runes || [], (row, key) => (key === 'total' ? row.total : (row.runes || {})[key] || 0))
        );

        // ===== Chat tab =====
        const chatMessages = computed(() => analysisData.value?.chat_analysis?.messages || []);
        const chatWheelCount = computed(() => chatMessages.value.filter(m => m.type === 'chatwheel').length);
        const visibleChat = computed(() =>
            chatShowWheel.value ? chatMessages.value : chatMessages.value.filter(m => m.type !== 'chatwheel')
        );

        // type 'chat': key is the text the player typed. type 'chatwheel': key
        // is 'chatwheel_<id>' and the payload says what the line is: phrase
        // {en, zh} (null for a spray and for an id its catalog does not
        // know), sound (a voice line) and spray. Shown as text, never as HTML.
        function chatText(msg) {
            if (msg.type !== 'chatwheel') return String(msg.key == null ? '' : msg.key);
            const line = DotaGame.chatWheelLine(msg, locale.value);
            if (line.kind === 'phrase') return line.text;
            if (line.kind === 'spray') return '';
            return line.id == null ? t('chat.wheel') : t('chat.wheel_n', { n: line.id });
        }

        // How a chat-wheel line is marked: 'voice' (heard in the game),
        // 'phrase' (text only), 'spray', or 'unknown' (only its number is known).
        function chatWheelKind(msg) {
            const line = DotaGame.chatWheelLine(msg, locale.value);
            return line.sound && line.kind !== 'spray' ? 'voice' : line.kind;
        }

        function chatPlayerName(msg) {
            const player = playerBySlot(msg.player_slot);
            return player ? playerName(player) : '';
        }

        // ===== Draft (overview tab) =====
        const draftData = computed(() => analysisData.value?.draft || { picks_bans: [], has_draft: false });

        // Modes with a draft: each team bans and picks in turn, in a fixed order.
        const DRAFT_MODES = new Set(['captains_mode', 'reverse_captains_mode', 'captains_draft']);

        // Picks per team in pick order. Bans belong to a team, and have a
        // place in the order, only in a draft mode. Everywhere else they are
        // one shared row without numbers: in all pick the bans come before
        // the picks and are nobody's (a replay says team null; OpenDota
        // splits them between the teams and numbers them after the picks).
        const draftView = computed(() => {
            const entries = (draftData.value.picks_bans || [])
                .map((entry, idx) => Object.assign({ number: (Number.isFinite(entry.order) ? entry.order : idx) + 1 }, entry))
                .sort((a, b) => a.number - b.number);
            const isTeam = (team) => team === 'radiant' || team === 'dire';
            const drafted = DRAFT_MODES.has(overview.value.game_mode);
            const lineup = (id) => (overview.value.players || []).filter(p => p.team === id).map(p => p.hero_id);
            const teams = ['radiant', 'dire'].map(id => {
                let picks = entries.filter(e => e.is_pick && e.team === id);
                // Valve's API lists more "picks" than players for some all-pick
                // matches (heroes nobody ended up playing): keep the lineup.
                const played = lineup(id);
                if (played.length && picks.length > played.length) picks = picks.filter(e => played.includes(e.hero_id));
                return { id, picks, bans: drafted ? entries.filter(e => !e.is_pick && e.team === id) : [] };
            });
            return {
                visible: !!draftData.value.has_draft && entries.length > 0,
                teams,
                sharedBans: entries.filter(e => !e.is_pick && !(drafted && isTeam(e.team))),
            };
        });

        // --- Utility ---
        // Game clock m:ss; before the horn it is negative ('-0:10'). A missing
        // time (null / undefined / NaN) gives `fallback`, never 'NaN:NaN'.
        function formatGameTime(seconds, fallback = '') {
            const value = Number(seconds);
            if (seconds == null || seconds === '' || !Number.isFinite(value)) return fallback;
            const abs = Math.floor(Math.abs(value));
            const sign = value < 0 && abs > 0 ? '-' : '';
            return `${sign}${Math.floor(abs / 60)}:${(abs % 60).toString().padStart(2, '0')}`;
        }

        // Scoreboard-style amount: nothing (null or 0) is a dash
        function formatNumber(n) {
            if (n == null || n === 0) return '-';
            if (Math.abs(n) >= 1000) return (n / 1000).toFixed(1) + 'k';
            return String(n);
        }

        function formatNetWorth(n) {
            if (n == null) return '-';
            if (n === 0) return '0';
            if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
            return String(n);
        }

        // "45 s", "4 min 12 s", "1 h 5 min" (rounded up: used for countdowns)
        function formatDuration(totalSeconds) {
            const total = Math.max(0, Math.ceil(Number(totalSeconds) || 0));
            if (total < 60) return t('time.sec', { s: total });
            if (total < 3600) {
                const m = Math.floor(total / 60);
                const sec = total % 60;
                return sec ? t('time.min_sec', { m, s: sec }) : t('time.min', { m });
            }
            let h = Math.floor(total / 3600);
            let m = Math.ceil((total % 3600) / 60);
            if (m === 60) { h += 1; m = 0; }
            return m ? t('time.hour_min', { h, m }) : t('time.hour', { h });
        }

        function timeAgo(unixTs) {
            const ts = Number(unixTs);
            if (!ts) return '';
            const diff = Math.max(0, Math.floor(Date.now() / 1000 - ts));
            if (diff < 60) return t('time.ago.now');
            if (diff < 3600) return tn('time.ago.minutes', Math.floor(diff / 60));
            if (diff < 86400) return tn('time.ago.hours', Math.floor(diff / 3600));
            const days = Math.floor(diff / 86400);
            if (days < 30) return tn('time.ago.days', days);
            if (days < 365) return tn('time.ago.months', Math.floor(days / 30));
            return tn('time.ago.years', Math.floor(days / 365));
        }

        // Absolute date and time of a unix timestamp in the UI language.
        function formatDateTime(unixTs) {
            const ts = Number(unixTs);
            if (!ts) return '';
            try {
                return new Date(ts * 1000).toLocaleString(locale.value === 'en' ? 'en-US' : 'zh-CN', { dateStyle: 'medium', timeStyle: 'short' });
            } catch (e) {
                return new Date(ts * 1000).toISOString().slice(0, 16).replace('T', ' ');
            }
        }

        function teamSum(players, field) {
            if (!players || !players.length) return 0;
            return players.reduce((sum, p) => sum + (p[field] || 0), 0);
        }

        // The six main inventory slots of a scoreboard row ('' = empty slot).
        function mainItemSlots(player) {
            const keys = ((player && player.final_item_keys) || []).slice(0, 6);
            while (keys.length < 6) keys.push('');
            return keys;
        }

        // When the item in the final inventory was bought (the last purchase
        // of that key). '' for starting items (bought before the horn) and
        // for purchases without a known time.
        function itemPurchaseTime(player, itemKey) {
            const log = (player && player.purchase_log) || [];
            if (!itemKey) return '';
            for (let i = log.length - 1; i >= 0; i--) {
                if (log[i].key === itemKey) return log[i].time > 0 ? (log[i].time_str || formatGameTime(log[i].time)) : '';
            }
            return '';
        }

        function itemTooltip(player, itemKey) {
            const name = itemName(itemKey);
            const time = itemPurchaseTime(player, itemKey);
            return time ? t('items.bought_at', { name, time }) : name;
        }

        // ===== Errors =====
        // Every failed request yields a DotaApi error object ({code, detail,
        // lang, status, retryAt, ...}); errorText() turns it into the line
        // shown to the user, in the current language.

        // Error raised in the browser. extra: { textKey, params } pick the
        // i18n text; without them 'error.<code>' is used.
        function clientError(code, extra) {
            return Object.assign(DotaApi.makeError(code, { client: true }), extra || {});
        }

        // Seconds left before a rate-limited request may be repeated (0 = now).
        function retrySeconds(error) {
            if (!error || !error.retryAt) return 0;
            // `clock` makes this re-evaluate every second; Date.now() is the exact time.
            const now = Math.max(clock.value, Date.now());
            return Math.max(0, Math.ceil((error.retryAt - now) / 1000));
        }

        function joinSentences(first, second) {
            return locale.value === 'en' ? `${first} ${second}` : first + second;
        }

        // The backend's `detail` is used as long as it is in the UI language;
        // after a language switch the local text of the code takes over.
        // Unknown codes always show the backend detail.
        function errorText(error) {
            if (!error) return '';
            let text;
            if (error.textKey) {
                text = t(error.textKey, error.params);
            } else if (error.code === 'rate_limited') {
                // The backend sentence carries a fixed number of seconds; ours counts down.
                text = t('error.rate_limited');
            } else if (error.detail && error.lang === locale.value) {
                text = error.detail;
            } else if (te('error.' + error.code)) {
                text = t('error.' + error.code, { status: error.status });
            } else {
                text = error.detail || t('error.generic');
            }
            if (error.retryAt) {
                const left = retrySeconds(error);
                text = joinSentences(text, left > 0 ? t('error.retry_in', { time: formatDuration(left) }) : t('error.retry_now'));
            }
            return text;
        }

        // Optional second line: 'error_hint.<code>'. entry.hintKey replaces
        // it when the match itself is gone (a dead link): a network error or
        // a rate limit behind the same link says nothing about the match.
        function errorHint(entry) {
            if (!entry || !entry.error) return '';
            const code = entry.error.code;
            const key = entry.hintKey && code === 'match_not_found' ? entry.hintKey : 'error_hint.' + code;
            return te(key) ? t(key) : '';
        }

        // Codes where doing the same thing again can work.
        const RETRYABLE_CODES = new Set([
            'network', 'timeout', 'server_error', 'bad_response', 'http_error', 'internal_error',
            'rate_limited', 'server_busy', 'upload_in_progress', 'upload_timeout', 'upload_failed',
            'opendota_unavailable', 'coach_failed', 'coach_timeout', 'daily_limit',
        ]);
        // Codes where the way out is fetching the match by its ID.
        const MATCH_ID_CODES = new Set(['parser_unavailable', 'upload_too_large', 'replay_parse_failed']);

        function canRetry(entry) {
            return !!(entry && entry.retry && RETRYABLE_CODES.has(entry.error.code));
        }

        function suggestsMatchId(entry) {
            return !!(entry && entry.source === 'upload' && MATCH_ID_CODES.has(entry.error.code));
        }

        // Until when an action of the start page is refused by its rate limit
        // ({ upload | fetch: epoch ms }). Kept apart from the banner: closing
        // the banner does not lift the limit.
        const flowBlocks = ref({});

        // source: 'upload' | 'fetch' | 'demo' | 'link'; retry: function or null.
        function showHomeError(error, source, retry, hintKey) {
            if (!error || error.code === 'aborted') return;
            homeError.value = { error, source: source || '', retry: retry || null, hintKey: hintKey || '' };
            // A link to an OpenDota match is a fetch: it shares that limit.
            const limited = source === 'link' ? 'fetch' : source;
            if (error.retryAt && (limited === 'upload' || limited === 'fetch')) {
                flowBlocks.value = { ...flowBlocks.value, [limited]: error.retryAt };
            }
            // The server did not answer: the status line should say so too.
            if (error.code === 'network' || error.code === 'timeout') loadCapabilities();
            nextTick(() => {
                const banner = document.querySelector('.home-card .ui-banner-error');
                if (banner && banner.scrollIntoView) banner.scrollIntoView({ block: 'nearest' });
            });
        }

        function retryHomeError() {
            const entry = homeError.value;
            if (!canRetry(entry) || retrySeconds(entry.error) > 0) return;
            homeError.value = null;
            entry.retry();
        }

        // Error banner inside the match view (shared with the tabs).
        function showMatchError(error, retry, hintKey) {
            if (!error || error.code === 'aborted') return;
            matchError.value = { error, source: 'match', retry: retry || null, hintKey: hintKey || '' };
        }

        function retryMatchError() {
            const entry = matchError.value;
            if (!canRetry(entry) || retrySeconds(entry.error) > 0) return;
            matchError.value = null;
            entry.retry();
        }

        // Seconds a start-page action stays blocked by its own rate limit.
        function waitSeconds(source) {
            return retrySeconds({ retryAt: flowBlocks.value[source] || 0 });
        }

        // "You can try again in 42 s." under a blocked action whose banner was closed.
        function waitNote(source) {
            const left = waitSeconds(source);
            const shown = homeError.value && retrySeconds(homeError.value.error) > 0;
            return left > 0 && !shown ? joinSentences(t('error.rate_limited'), t('error.retry_in', { time: formatDuration(left) })) : '';
        }

        // ===== Capabilities =====
        // GET /api/capabilities drives what the start page offers. It is read
        // again before every upload and on each return to the start page
        // (quota, knowledge mode and the parser state change while we run).
        const DEFAULT_UPLOAD_EXTENSIONS = ['.dem', '.dem.bz2', '.dem.zst'];

        let capabilitiesAskedAt = 0;

        async function loadCapabilities() {
            capabilitiesAskedAt = Date.now();
            const res = await DotaApi.get('/api/capabilities', { timeoutMs: 8000 });
            if (res.ok) {
                capabilities.value = res.data;
                capabilitiesFailed.value = false;
            } else if (res.error.code !== 'aborted') {
                capabilitiesFailed.value = true;   // keep the last known answer
            }
            return res.ok;
        }

        // While the server does not answer, the start page asks again now
        // and then: the "unreachable" chip goes away by itself.
        const CAPABILITIES_RETRY_MS = 15000;
        function retryCapabilities() {
            if (capabilitiesFailed.value && !analysisData.value && !busy.value
                && !document.hidden && Date.now() - capabilitiesAskedAt >= CAPABILITIES_RETRY_MS) loadCapabilities();
        }

        const uploadExtensions = computed(() => {
            const list = capabilities.value && capabilities.value.upload_extensions;
            return Array.isArray(list) && list.length ? list.map(String) : DEFAULT_UPLOAD_EXTENSIONS;
        });
        // <input accept> only looks at the last extension: .dem,.bz2,.zst
        const uploadAccept = computed(() =>
            [...new Set(uploadExtensions.value.map(ext => '.' + ext.split('.').pop()))].join(',')
        );
        const maxUploadMb = computed(() => Number(capabilities.value && capabilities.value.max_upload_mb) || 0);
        // Unknown (no answer yet) counts as available: the server decides.
        const parserAvailable = computed(() => !capabilities.value || capabilities.value.replay_parser !== false);
        const uploadFormatHint = computed(() => {
            const exts = uploadExtensions.value.join(' / ');
            return maxUploadMb.value
                ? t('upload.format_hint', { exts, limit: maxUploadMb.value })
                : t('upload.format_hint_nolimit', { exts });
        });

        // ===== Start-page flows: upload, fetch by ID, demo, deep link =====
        const MATCH_TIMEOUT_MS = 90000;     // GET of a match payload
        const DEMO_TIMEOUT_MS = 180000;     // the demo replay may be parsed on first use
        const PARSE_TIMEOUT_MS = 300000;    // wait for the answer after the upload was sent

        const fileInput = ref(null);
        const matchIdField = ref(null);
        const matchIdInput = ref('');
        const dragActive = ref(false);
        let dragDepth = 0;
        let uploadHandle = null;    // running DotaApi.upload()
        let flowAbort = null;       // AbortController of the running match request

        function _formatBytes(b) {
            if (b < 1024) return Math.round(b) + ' B';
            if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB';
            if (b < 1024 * 1024 * 1024) return (b / (1024 * 1024)).toFixed(1) + ' MB';
            return (b / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
        }
        function _resetUploadProgress() {
            uploadProgress.value = { phase: '', percent: 0, loaded: 0, total: 0, speed: 0, eta: 0, indeterminate: false };
        }

        // "27.3 MB / 65.3 MB · 2.9 MB/s · 13 s left"
        const uploadProgressLabel = computed(() => {
            const u = uploadProgress.value;
            if (u.phase !== 'uploading' || !u.loaded) return '';
            const parts = [u.total ? `${_formatBytes(u.loaded)} / ${_formatBytes(u.total)}` : _formatBytes(u.loaded)];
            if (u.speed > 0) {
                parts.push(`${_formatBytes(u.speed)}/s`);
                if (u.total) parts.push(t('upload.time_left', { time: formatDuration(u.eta) }));
            }
            return parts.join(' · ');
        });

        // Start one flow; null when another one is running. Flows are told
        // apart by `seq` (busy.value is a reactive proxy, not the object).
        let flowSeq = 0;
        function beginFlow(info) {
            if (busy.value) return null;
            homeError.value = null;
            const flow = Object.assign({ seq: ++flowSeq }, info);
            busy.value = flow;
            return flow;
        }

        // false once the flow was cancelled (or replaced by another one)
        function isCurrentFlow(flow) {
            return !!busy.value && busy.value.seq === flow.seq;
        }

        function endFlow(flow) {
            if (!isCurrentFlow(flow)) return;
            busy.value = null;
            _resetUploadProgress();
        }

        // Cancel button of every flow: nothing stays pending afterwards.
        function cancelFlow() {
            if (uploadHandle) { uploadHandle.abort(); uploadHandle = null; }
            if (flowAbort) { flowAbort.abort(); flowAbort = null; }
            busy.value = null;
            _resetUploadProgress();
        }

        function focusMatchId() {
            nextTick(() => { if (matchIdField.value) matchIdField.value.focus(); });
        }

        function isMatchPayload(data) {
            return !!(data && data.analysis && typeof data.analysis === 'object' && data.analysis.match_overview);
        }

        // GET a match payload ({analysis, parse_source, has_parsed_data,
        // _match_key}); a 2xx answer of another shape is a bad_response error.
        async function fetchMatchPayload(path, timeoutMs) {
            const controller = new AbortController();
            flowAbort = controller;
            const res = await DotaApi.get(path, { signal: controller.signal, timeoutMs: timeoutMs || MATCH_TIMEOUT_MS });
            if (flowAbort === controller) flowAbort = null;
            if (res.ok && !isMatchPayload(res.data)) return { ok: false, error: clientError('bad_response') };
            return res;
        }

        // Put a match payload on screen (replaces the one shown, if any).
        // coachPlayer: player slot whose coach report a link asks for.
        function showMatch(data, fallbackKey, tab, coachPlayer) {
            if (analysisData.value) clearMatch();
            analysisData.value = data.analysis;
            parseSource.value = data.parse_source || 'unknown';
            hasParsedData.value = data.has_parsed_data !== false;
            matchKey.value = data._match_key || fallbackKey || '';
            initTabState();
            activeTab.value = tabs.value.some(x => x.id === tab) ? tab : 'overview';
            const coachLink = activeTab.value === 'ai_coach' && coachAvailable.value && coachPlayer != null && !!playerBySlot(coachPlayer);
            if (coachLink) coachSlot.value = coachPlayer;   // part of the address written below
            // Ability names are only needed once a match is open (650 KB).
            DotaCatalog.load('abilities');
            syncRoute(true);
            updateDocumentTitle();
            window.scrollTo(0, 0);
            if (coachLink) requestCoach(coachPlayer);
            nextTick(() => {
                revealActiveTab();
                renderActiveTab();
                // The control that opened the match is gone: keyboard focus goes to the tab bar.
                if (!document.activeElement || document.activeElement === document.body) focusActiveTab();
            });
        }

        // --- Upload ---
        function pickFile() {
            if (busy.value || !parserAvailable.value || !fileInput.value) return;
            fileInput.value.click();
        }

        function handleFileSelect(e) {
            const file = e.target.files && e.target.files[0];
            // Clear the input so re-selecting the same file (e.g. retry after
            // cancel) still fires a change event.
            e.target.value = '';
            if (file) uploadFile(file);
        }

        function dragHasFiles(e) {
            return !!(e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files'));
        }

        // The whole start page is the drop target; a file dropped anywhere
        // else must not make the browser navigate to it.
        function onWindowDragEnter(e) {
            if (!dragHasFiles(e)) return;
            dragDepth++;
            dragActive.value = !analysisData.value && !busy.value && parserAvailable.value;
        }

        function onWindowDragLeave(e) {
            if (!dragHasFiles(e)) return;
            dragDepth = Math.max(0, dragDepth - 1);
            if (!dragDepth) dragActive.value = false;
        }

        function onWindowDragOver(e) {
            if (dragHasFiles(e)) e.preventDefault();
        }

        function onWindowDrop(e) {
            if (!dragHasFiles(e)) return;
            e.preventDefault();
            dragDepth = 0;
            dragActive.value = false;
            const file = e.dataTransfer.files && e.dataTransfer.files[0];
            if (file && !analysisData.value) uploadFile(file);
        }

        // Checks done before any byte is sent; null when the file is fine.
        function validateReplayFile(file) {
            const name = String(file.name || '').toLowerCase();
            const exts = uploadExtensions.value;
            if (!exts.some(ext => name.endsWith(ext.toLowerCase()))) {
                return clientError('upload_bad_extension', { textKey: 'error.upload_bad_extension', params: { exts: exts.join(' / ') } });
            }
            if (!file.size) return clientError('upload_empty', { textKey: 'error.upload_empty' });
            const limit = maxUploadMb.value;
            if (limit && file.size > limit * 1024 * 1024) {
                return clientError('upload_too_large', {
                    textKey: 'error.upload_too_large_local',
                    params: { size: Math.ceil(file.size / (1024 * 1024)), limit },
                });
            }
            return null;
        }

        async function uploadFile(file) {
            if (!file || busy.value) return;
            if (!parserAvailable.value) {
                showHomeError(clientError('parser_unavailable', { textKey: 'error.parser_unavailable' }), 'upload');
                return;
            }
            let problem = validateReplayFile(file);
            if (problem) {
                showHomeError(problem, 'upload');
                return;
            }
            const flow = beginFlow({ kind: 'upload', name: file.name });
            if (!flow) return;
            const retry = () => uploadFile(file);
            uploadProgress.value = { phase: 'uploading', percent: 0, loaded: 0, total: file.size, speed: 0, eta: 0, indeterminate: false };

            // The parser can have gone away and the limit can have changed
            // since the page loaded: ask again before sending 50-300 MB.
            await loadCapabilities();
            if (!isCurrentFlow(flow)) return;
            problem = parserAvailable.value
                ? validateReplayFile(file)
                : clientError('parser_unavailable', { textKey: 'error.parser_unavailable' });
            if (problem) {
                endFlow(flow);
                showHomeError(problem, 'upload');
                return;
            }

            let lastSampleAt = performance.now();
            let lastSampleLoaded = 0;
            let speed = 0;
            const handle = DotaApi.upload('/api/upload', file, {
                processTimeoutMs: PARSE_TIMEOUT_MS,
                onProgress({ loaded, total }) {
                    if (!isCurrentFlow(flow)) return;
                    const now = performance.now();
                    const dt = (now - lastSampleAt) / 1000;
                    if (dt > 0.4) {
                        speed = (loaded - lastSampleLoaded) / dt;
                        lastSampleAt = now;
                        lastSampleLoaded = loaded;
                    }
                    if (!total) {
                        // Proxies may strip Content-Length; show transferred
                        // bytes with an indeterminate bar instead of frozen 0%
                        uploadProgress.value = { phase: 'uploading', percent: 0, loaded, total: 0, speed, eta: 0, indeterminate: true };
                        return;
                    }
                    uploadProgress.value = {
                        phase: 'uploading',
                        percent: Math.min(100, Math.round(loaded / total * 100)),
                        loaded, total, speed,
                        eta: speed > 0 ? (total - loaded) / speed : 0,
                        indeterminate: false,
                    };
                },
                onSent() {
                    if (isCurrentFlow(flow)) uploadProgress.value = { ...uploadProgress.value, phase: 'parsing', percent: 100 };
                },
            });
            uploadHandle = handle;
            const res = await handle.promise;
            if (uploadHandle === handle) uploadHandle = null;
            if (!isCurrentFlow(flow)) return;   // cancelled

            if (!res.ok) {
                endFlow(flow);
                // A dropped connection mid-upload: the server restarted, or it
                // cut off a file far above the limit.
                if (res.error.code === 'network') res.error.textKey = 'error.network_upload';
                if (res.error.code === 'parser_unavailable') loadCapabilities();
                showHomeError(res.error, 'upload', retry);
                return;
            }

            const key = res.data.file_key || res.data._match_key;
            uploadProgress.value = { ...uploadProgress.value, phase: 'loading', percent: 100 };
            const loaded = key
                ? await fetchMatchPayload(`/api/match/${encodeURIComponent(key)}`)
                : { ok: false, error: clientError('bad_response') };
            if (!isCurrentFlow(flow)) return;
            endFlow(flow);
            if (!loaded.ok) {
                showHomeError(loaded.error, 'upload', retry);
                return;
            }
            showMatch(loaded.data, key);
            // Same replay as before (also as .dem.bz2 / .dem.zst): no new parse.
            if (res.data.cached) toast('upload.cached', 'success');
        }

        // --- Fetch by match ID / demo / deep link ---

        // Digits, or a link that contains /matches/<id> (OpenDota, Dotabuff, Stratz).
        function parseMatchId(input) {
            const text = String(input || '').trim();
            if (/^\d{1,19}$/.test(text)) return text.replace(/^0+(?=\d)/, '');
            const fromLink = /\/match(?:es)?\/(\d{1,19})(?!\d)/i.exec(text);
            return fromLink ? fromLink[1] : '';
        }

        // options: { fallbackKey, tab, timeoutMs, retry, hintKey }
        async function runMatchFlow(info, path, options) {
            const o = options || {};
            const flow = beginFlow(info);
            if (!flow) return false;
            // Hero and item catalogs load alongside (they never fail the
            // flow): the first paint of the match then has names and icons.
            const [res] = await Promise.all([
                fetchMatchPayload(path, o.timeoutMs),
                DotaCatalog.load('heroes'),
                DotaCatalog.load('items'),
            ]);
            if (!isCurrentFlow(flow)) return false;   // cancelled
            endFlow(flow);
            if (!res.ok) {
                showHomeError(res.error, info.kind, o.retry, o.hintKey);
                return false;
            }
            showMatch(res.data, o.fallbackKey, o.tab, o.coachSlot);
            return true;
        }

        async function fetchMatchById() {
            if (busy.value) return;
            const id = parseMatchId(matchIdInput.value);
            if (!id) {
                if (String(matchIdInput.value).trim()) {
                    showHomeError(clientError('invalid_match_id', { textKey: 'error.invalid_match_id' }), 'fetch');
                }
                focusMatchId();
                return;
            }
            matchIdInput.value = id;
            await runMatchFlow({ kind: 'fetch', id }, `/api/fetch/${id}`, {
                fallbackKey: `opendota_${id}`,
                retry: fetchMatchById,
            });
        }

        async function loadDemo() {
            await runMatchFlow({ kind: 'demo' }, '/api/demo', { fallbackKey: 'demo', timeoutMs: DEMO_TIMEOUT_MS, retry: loadDemo });
        }

        // ===== Deep links: #/match/<match key>[/<tab>[/<player slot>]] =====
        // The match key is the server's (_match_key: demo, replay_<hash>,
        // opendota_<id>). A reload or a shared link loads it again from the
        // server; nothing of a match is kept in the browser. The player
        // slot belongs to the AI coach tab: the report shown there.
        function parseRoute() {
            const m = /^#\/match\/([^/]+)(?:\/([a-z_]+)(?:\/(\d{1,3}))?)?\/?$/.exec(window.location.hash || '');
            if (!m) return null;
            try {
                return { key: decodeURIComponent(m[1]), tab: m[2] || '', slot: m[3] === undefined ? null : Number(m[3]) };
            } catch (e) {
                return null;
            }
        }

        function routeHash() {
            if (!analysisData.value || !matchKey.value) return '';
            let tab = activeTab.value !== 'overview' ? '/' + activeTab.value : '';
            if (activeTab.value === 'ai_coach' && coachSlot.value != null) tab += '/' + coachSlot.value;
            return `#/match/${encodeURIComponent(matchKey.value)}${tab}`;
        }

        // The coach report a link names: asked for once its match is on screen.
        function openCoachRoute(slot) {
            if (activeTab.value !== 'ai_coach' || slot == null || slot === coachSlot.value) return;
            if (playerBySlot(slot)) requestCoach(slot);
        }

        // Mirror the state into the address bar. push = new history entry
        // (opening / leaving a match); tab changes replace the entry.
        function syncRoute(push) {
            const target = routeHash();
            const current = window.location.hash || '';
            if (target === current || (!target && !current.startsWith('#/'))) return;
            const url = window.location.pathname + window.location.search + target;
            try {
                if (push) window.history.pushState(null, '', url);
                else window.history.replaceState(null, '', url);
            } catch (e) { /* history is unavailable (sandboxed frame): links just do not update */ }
        }

        function pathForMatchKey(key) {
            if (key === 'demo') return '/api/demo';
            // /api/fetch also restores an OpenDota match the server no longer
            // holds and picks up the parsed data once OpenDota has it.
            const openDota = /^opendota_(\d+)$/.exec(key);
            if (openDota) return `/api/fetch/${openDota[1]}`;
            return `/api/match/${encodeURIComponent(key)}`;
        }

        // The server can drop an opendota_<id> match (cache pruning, an
        // update): a later call for it answers match_not_found. This loads
        // the match on the server again; true when it is back.
        async function restoreMatchOnServer() {
            if (!matchKey.value) return false;
            const res = await DotaApi.get(pathForMatchKey(matchKey.value), { timeoutMs: MATCH_TIMEOUT_MS });
            return res.ok;
        }

        async function openRoute(route) {
            const ok = await runMatchFlow({ kind: 'link' }, pathForMatchKey(route.key), {
                fallbackKey: route.key,
                tab: route.tab,
                coachSlot: route.slot,
                timeoutMs: route.key === 'demo' ? DEMO_TIMEOUT_MS : MATCH_TIMEOUT_MS,
                retry: () => openRoute(route),
                hintKey: 'error_hint.link_not_found',
            });
            // A link that cannot be opened leaves the start page, not a dead address.
            if (!ok && !analysisData.value && !busy.value) syncRoute(false);
        }

        // An address that looks like a route but is none ('#/match/demo/a/1/x',
        // a key that cannot be decoded): say so and leave a clean address,
        // instead of the start page under a dead link.
        function rejectBadRoute() {
            const hash = window.location.hash || '';
            if (parseRoute() || !hash.startsWith('#/')) return false;
            syncRoute(false);
            // ('#/' alone is just the start page)
            if (!analysisData.value && hash !== '#/') showHomeError(clientError('bad_link', { textKey: 'error.bad_link' }), 'link');
            return true;
        }

        // Back / forward, or an address typed by hand.
        function onRouteChange() {
            const route = parseRoute();
            if (!route) {
                if (rejectBadRoute()) return;
                if (analysisData.value) {
                    clearMatch();
                    updateDocumentTitle();
                    loadCapabilities();
                }
                return;
            }
            if (analysisData.value && route.key === matchKey.value) {
                activeTab.value = tabs.value.some(x => x.id === route.tab) ? route.tab : 'overview';
                openCoachRoute(route.slot);
                return;
            }
            if (busy.value) cancelFlow();
            if (analysisData.value) clearMatch();
            openRoute(route);
        }

        // ===== OpenDota match without parsed data =====
        // The match stays usable with its scoreboard. Parsing is only asked
        // for when the user clicks; "Check again" then loads the parsed data.
        // state: idle | requesting | checking; errorFrom: which button failed ('request' | 'check')
        const parseRequest = ref({ state: 'idle', requested: false, error: null, errorFrom: '' });

        // A rate-limited button waits for its own limit only.
        function parseActionBlocked(action) {
            const r = parseRequest.value;
            return r.state !== 'idle' || (r.errorFrom === action && retrySeconds(r.error) > 0);
        }

        const openDotaMatchId = computed(() => {
            const fromKey = /^opendota_(\d+)$/.exec(matchKey.value || '');
            if (fromKey) return fromKey[1];
            const id = analysisData.value && analysisData.value.match_overview && analysisData.value.match_overview.match_id;
            return id && Number(id) > 0 ? String(id) : '';
        });

        const showUnparsedNotice = computed(() =>
            !!analysisData.value && parseSource.value === 'opendota' && !hasParsedData.value && !!openDotaMatchId.value
        );

        async function requestParse() {
            const id = openDotaMatchId.value;
            if (!id || parseRequest.value.state !== 'idle') return;
            const key = matchKey.value;
            parseRequest.value = { ...parseRequest.value, state: 'requesting', error: null, errorFrom: '' };
            const res = await DotaApi.post(`/api/request-parse/${id}`, { timeoutMs: 60000 });
            if (matchKey.value !== key) return;   // another match was opened meanwhile
            parseRequest.value = {
                state: 'idle',
                requested: res.ok || parseRequest.value.requested,
                error: res.ok ? null : res.error,
                errorFrom: res.ok ? '' : 'request',
            };
        }

        async function refetchParsed() {
            const id = openDotaMatchId.value;
            if (!id || parseRequest.value.state !== 'idle') return;
            const key = matchKey.value;
            parseRequest.value = { ...parseRequest.value, state: 'checking', error: null, errorFrom: '' };
            const res = await DotaApi.get(`/api/fetch/${id}`, { timeoutMs: MATCH_TIMEOUT_MS });
            if (matchKey.value !== key) return;
            if (!res.ok || !isMatchPayload(res.data)) {
                parseRequest.value = {
                    ...parseRequest.value, state: 'idle',
                    error: res.ok ? clientError('bad_response') : res.error, errorFrom: 'check',
                };
                return;
            }
            if (res.data.has_parsed_data === false) {
                parseRequest.value = { ...parseRequest.value, state: 'idle', error: null, errorFrom: '' };
                toast('notice.unparsed.still', 'info');
                return;
            }
            showMatch(res.data, key, activeTab.value, coachSlot.value);
            toast('notice.unparsed.ready', 'success');
        }

        // ===== AI coach =====
        // One report per player and language. It comes over the event stream:
        // progress while the server writes it, or a single event when it is
        // cached. When the stream cannot be used, the JSON route returns the
        // same report, or the error in the backend's words. The server shares
        // one generation between all requests for the same report and keeps
        // working when nobody listens, so reconnecting, cancelling or asking
        // again never starts a second one. The one exception is asked for
        // by name: "generate again" on a degraded report (refresh=1), which
        // the server allows once per cooldown.
        const COACH_IDLE_MS = 330000;           // the server itself gives up after about 240 s
        const COACH_JSON_TIMEOUT_MS = 300000;
        const COACH_SWAP_DELAY_MS = 250;        // a cached report replaces the one on screen without a flash
        let coachStream = null;
        let coachAbort = null;
        let coachSwapTimer = null;
        let coachSeq = 0;                       // tells the current request from the ones it replaced

        function coachPath(route, slot) {
            return `/api/match/${encodeURIComponent(matchKey.value)}/${route}/${encodeURIComponent(slot)}`;
        }

        // Stop listening for the report that is on its way.
        function stopCoachRequest() {
            coachSeq += 1;
            if (coachStream) { coachStream.close(); coachStream = null; }
            if (coachAbort) { coachAbort.abort(); coachAbort = null; }
            if (coachSwapTimer) { clearTimeout(coachSwapTimer); coachSwapTimer = null; }
        }

        // Ask for the report of a player in the current language.
        // options.restored: the match was already put back on the server once.
        // options.refresh: "generate again" on a degraded report. Only that
        // button sets it; every other request takes the report the server has.
        function requestCoach(slot, options) {
            if (slot == null || !matchKey.value || !coachAvailable.value) return;
            stopCoachRequest();
            const seq = coachSeq;
            const lang = locale.value;
            const ask = {
                lang,
                restored: !!(options && options.restored),
                refresh: !!(options && options.refresh),
                params: options && options.refresh ? { lang, refresh: 1 } : { lang },
            };
            coachSlot.value = slot;
            coachLoading.value = true;
            coachError.value = null;
            coachCancelled.value = false;
            coachFallback.value = false;
            coachProgress.value = DotaCoach.newProgress();
            syncRoute(false);
            // Another player's report stays for a moment: a cached answer
            // replaces it at once, a new one gets the progress panel.
            if (coachReport.value && coachReport.value.player_slot !== slot) {
                coachSwapTimer = setTimeout(() => {
                    if (seq === coachSeq && coachLoading.value) coachReport.value = null;
                }, COACH_SWAP_DELAY_MS);
            }
            coachStream = DotaApi.stream(coachPath('ai-coach-stream', slot), {
                params: ask.params,
                idleMs: COACH_IDLE_MS,
                onMessage(msg) {
                    if (seq !== coachSeq) return;
                    if (msg.type === 'progress') coachProgress.value = DotaCoach.advance(coachProgress.value, msg);
                    else if (msg.type === 'done') finishCoach(seq, msg.data, ask);
                    else if (msg.type === 'error') failCoach(seq, DotaApi.errorFromEvent(msg, lang), ask);
                },
                onFail(reason) {
                    if (seq !== coachSeq) return;
                    coachStream = null;
                    if (reason === 'idle') failCoach(seq, clientError('coach_timeout'), ask);
                    else fetchCoachJson(seq, ask);
                },
            });
        }

        // The stream gave out: the JSON route waits for the same report (a
        // run the stream started with refresh=1 is joined, not started again).
        async function fetchCoachJson(seq, ask) {
            coachFallback.value = true;
            const controller = new AbortController();
            coachAbort = controller;
            const res = await DotaApi.get(coachPath('ai-coach', coachSlot.value), {
                params: ask.params, signal: controller.signal, timeoutMs: COACH_JSON_TIMEOUT_MS,
            });
            if (seq !== coachSeq) return;
            if (res.ok) finishCoach(seq, res.data, ask);
            else failCoach(seq, res.error, ask);
        }

        function finishCoach(seq, report, ask) {
            if (seq !== coachSeq) return;
            if (!DotaCoach.isReport(report)) { failCoach(seq, clientError('bad_response'), { ...ask, restored: true }); return; }
            stopCoachRequest();
            const slot = coachSlot.value;
            // The same player again (another language, a second try): the report is replaced in place.
            const samePlayer = !!coachReport.value && coachReport.value.player_slot === slot;
            if (!samePlayer) coachOpenSources.value = {};
            coachOpenEntries.value = {};
            coachReport.value = report;
            const again = DotaCoach.refreshState(report);
            coachRefreshAt.value = again.wait > 0 ? Date.now() + again.wait * 1000 : 0;
            coachLoading.value = false;
            coachFallback.value = false;
            coachProgress.value = { ...coachProgress.value, percent: 100 };
            // "Generate again" came back degraded: the new run had gaps
            // too, or the server has not started one yet. Either way the
            // page waits for the next click; the banner counts down to it.
            if (ask.refresh && again.available) {
                toast(again.wait > 0 ? 'coach.degraded.still' : 'coach.degraded.still_now', 'info',
                    () => ({ time: formatDuration(again.wait) }));
            }
            if (activeTab.value !== 'ai_coach' && !samePlayer) {
                const hero = report.hero_name;
                toast('coach.toast.ready', 'success', () => ({ hero: heroName(hero) }));
            }
            // A report written by the LLM used up some of the daily quota.
            if (capabilities.value && capabilities.value.ai_coach_llm) loadCapabilities();
        }

        async function failCoach(seq, error, ask) {
            if (seq !== coachSeq || !error || error.code === 'aborted') return;
            stopCoachRequest();
            const slot = coachSlot.value;
            const refresh = !!(ask && ask.refresh);
            // The server can drop a match it holds (cache pruning, a
            // restart): put it back once, then ask again.
            if (error.code === 'match_not_found' && !(ask && ask.restored)) {
                const mine = coachSeq;
                const key = matchKey.value;
                const back = await restoreMatchOnServer();
                if (mine !== coachSeq || key !== matchKey.value) return;
                if (back) { requestCoach(slot, { restored: true, refresh }); return; }
            }
            coachLoading.value = false;
            coachFallback.value = false;
            // Never leave another player's report under this player's error.
            if (coachReport.value && coachReport.value.player_slot !== slot) coachReport.value = null;
            // "Retry" repeats what was asked for, a new run included (a click each time).
            coachError.value = { error, source: 'coach', retry: () => requestCoach(slot, { refresh }), hintKey: '' };
            // A new run refused by a limit: "generate again" stays off as long as the limit lasts.
            if (refresh && error.retryAt > coachRefreshAt.value) coachRefreshAt.value = error.retryAt;
            if (activeTab.value !== 'ai_coach') toastError(error);
            if (error.code === 'rate_limited' || error.code === 'daily_limit') loadCapabilities();
        }

        function retryCoach() {
            const entry = coachError.value;
            if (!canRetry(entry) || retrySeconds(entry.error) > 0) return;
            entry.retry();
        }

        // "Cancel": the page stops waiting; the server finishes the report
        // it started and keeps it for the next request.
        function cancelCoach() {
            if (!coachLoading.value) return;
            stopCoachRequest();
            coachLoading.value = false;
            coachFallback.value = false;
            const shown = coachReport.value;
            if (shown) {
                // Back to the report that is still on screen.
                coachSlot.value = shown.player_slot;
                syncRoute(false);
                toast('coach.cancelled', 'info');
            } else {
                coachCancelled.value = true;
            }
        }

        // --- "Generate again" on a degraded report ---
        // The report says whether the server offers a new run and how long
        // it still refuses one (refresh: {available, retry_after}): until
        // then the button is off and counts down. One click is one request
        // with refresh=1; its answer, whatever it is, ends it.
        const coachCanRefresh = computed(() => DotaCoach.refreshState(coachReport.value).available);

        // Seconds until the new run can be asked for (0 = now).
        const coachRefreshWait = computed(() => retrySeconds({ retryAt: coachRefreshAt.value }));

        // Line beside the button: what it does, or how long it stays off.
        const coachRefreshNote = computed(() => {
            if (!coachCanRefresh.value) return '';
            const left = coachRefreshWait.value;
            return left > 0 ? t('coach.degraded.wait', { time: formatDuration(left) }) : t('coach.degraded.hint');
        });

        function retryDegradedCoach() {
            if (coachSlot.value == null || coachLoading.value || !coachCanRefresh.value || coachRefreshWait.value > 0) return;
            requestCoach(coachSlot.value, { refresh: true });
        }

        function resetCoach() {
            stopCoachRequest();
            coachSlot.value = null;
            coachReport.value = null;
            coachRefreshAt.value = 0;
            coachLoading.value = false;
            coachError.value = null;
            coachCancelled.value = false;
            coachFallback.value = false;
            coachProgress.value = DotaCoach.newProgress();
            coachOpenSources.value = {};
            coachOpenEntries.value = {};
        }

        // --- Picker and report header ---
        // report.position: {} when it could not be derived, else {number 1-5,
        // label_zh, label_en, confidence: high | medium | low, method}: the
        // same derivation as the chips of the picker (positionChips above).
        function positionView(position) {
            if (!position || !position.number) return null;
            const given = locale.value === 'en' ? position.label_en : position.label_zh;
            return {
                label: given || codeLabel('position', position.number),
                // said in words when the detection is less than sure
                note: knownCodeLabel('confidence_note', position.confidence),
                tone: DotaCoach.confidenceTone(position.confidence),
                title: t('coach.position.title', {
                    confidence: codeLabel('confidence', position.confidence),
                    method: codeLabel('position_method', position.method),
                }),
            };
        }

        const coachTeams = computed(() => scoreboardTeams.value.map(team => ({
            id: team.id,
            players: team.players.map(p => ({
                slot: p.player_slot,
                slug: p.hero_slug || '',
                hero: heroName(p.hero_name),
                player: playerName(p),
                kda: `${formatCount(p.kills)}/${formatCount(p.deaths)}/${formatCount(p.assists)}`,
                position: positionChips.value[p.player_slot] || null,
            })),
        })));

        const coachPlayer = computed(() => (coachSlot.value == null ? null : playerBySlot(coachSlot.value)));
        const coachHeroName = computed(() => (coachPlayer.value ? heroName(coachPlayer.value.hero_name) : ''));

        // What a new report will be: written by the model, or by the rules.
        const coachMode = computed(() => {
            const cap = capabilities.value;
            if (!cap) return null;
            if (!cap.ai_coach_llm) return { kind: 'rules', text: t('coach.mode.rules') };
            const left = cap.ai_coach_daily_remaining;
            return {
                kind: left === 0 ? 'warn' : 'llm',
                text: t('coach.mode.llm') + (left == null ? '' : ' ' + t('coach.mode.quota', { n: left })),
            };
        });

        // --- Progress panel ---
        const coachStageRows = computed(() => DotaCoach.stageRows(coachProgress.value).map(row => ({
            id: row.id,
            state: row.state,
            label: codeLabel('coach_stage', row.id),
            count: row.state === 'active' && row.total > 1 ? `${row.current}/${row.total}` : '',
        })));

        const coachStageLabel = computed(() =>
            coachFallback.value ? t('coach.loading.fallback') : codeLabel('coach_stage', DotaCoach.currentStage(coachProgress.value))
        );

        // Language the report on screen is written in (its own `lang`);
        // '' without a report.
        const coachReportLang = computed(() => DotaCoach.reportLang(coachReport.value));

        // The report on screen is in another language than the page (the
        // new one is loading, failed, or was cancelled).
        const coachLangMismatch = computed(() => !!coachReportLang.value && coachReportLang.value !== locale.value);

        // --- Report text ---
        // Texts of the report are plain; the model sometimes adds **bold**,
        // bullet lines and line breaks. Escaped first, so the only markup
        // in the result is what is produced here.
        function richText(text) {
            const lines = escapeHtml(String(text == null ? '' : text)).split(/\r?\n/);
            const out = [];
            let bullets = [];
            const flush = () => {
                if (bullets.length) out.push('<ul>' + bullets.map(b => `<li>${b}</li>`).join('') + '</ul>');
                bullets = [];
            };
            lines.forEach(raw => {
                const line = raw.trim().replace(/^#{1,6}\s+/, '').replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
                const bullet = /^(?:[-*•]|\d+[.)])\s+(.*)$/.exec(line);
                if (bullet) { bullets.push(bullet[1]); return; }
                flush();
                if (line) out.push(`<p>${line}</p>`);
            });
            flush();
            return out.join('');
        }

        // Label of a code that only has one when it is known ('' otherwise).
        function knownCodeLabel(domain, value) {
            return value != null && value !== '' && te(`code.${domain}.${value}`) ? codeLabel(domain, value) : '';
        }

        // Who wrote an evaluation or a phase review. In a report without any
        // model text only the degraded ones are marked: the badge of the
        // report already says "rule-based".
        function authorMark(item, report) {
            if (item.degraded) return { kind: 'degraded', label: t('coach.mark.degraded'), title: t('coach.mark.degraded_title') };
            if (item.llm_used) return { kind: 'llm', label: t('coach.mark.llm'), title: t('coach.mark.llm_title') };
            if (report.llm_used) return { kind: 'rules', label: t('coach.mark.rules'), title: t('coach.mark.rules_title') };
            return null;
        }

        function gradeView(score) {
            const tone = DotaCoach.gradeTone(score);
            return { text: tone === 'none' ? '-' : String(score).toUpperCase(), tone };
        }

        // Knowledge sources of one evaluation: {id, title, source_type,
        // patch, cited}; cited ones come first in the payload. The id is
        // what the text of the entry is read by (toggleSource below).
        function sourceView(source, index) {
            const type = String(source.source_type || '');
            return {
                key: source.id || 'source' + index,
                id: source.id ? String(source.id) : '',
                title: String(source.title || ''),
                type: DotaCoach.SOURCE_TYPES.includes(type) ? type : 'other',
                typeLabel: codeLabel('source_type', type),
                typeTitle: knownCodeLabel('source_type_title', type),
                patch: source.patch ? String(source.patch) : '',
                cited: !!source.cited,
            };
        }

        // Who landed the kill, from the kill feed of the match.
        function coachKillView(slot, time) {
            const kill = DotaCoach.findKill(analysisData.value?.combat?.kills, slot, time);
            if (!kill) return null;
            return {
                name: killerLabel(kill),
                slug: kill.killer ? (kill.killer_slug || heroSlugByName(kill.killer)) : '',
                isHero: !!kill.killer,
                team: kill.killer_team || '',
                assists: killAssists(kill),
            };
        }

        const CHAIN_SIGNIFICANCE = ['root_cause', 'contributing', 'consequence', 'final'];

        // causal_chain: null or {root_cause_type, root_cause_summary, steps[]};
        // the death is the last step, the others lead up to it.
        function chainView(chain, deathTime) {
            const steps = chain && Array.isArray(chain.steps) ? chain.steps : [];
            if (!steps.length) return null;
            const end = isMissing(deathTime) ? null : Number(deathTime);
            return {
                rootType: chain.root_cause_type ? codeLabel('root_cause', chain.root_cause_type) : '',
                rootSummary: chain.root_cause_summary || '',
                // Only the death itself: where it happened, no chain to draw.
                single: steps.length === 1,
                steps: steps.map((step, index) => {
                    const known = !isMissing(step.time);
                    const gap = known && end != null ? Math.round(end - Number(step.time)) : 0;
                    const significance = CHAIN_SIGNIFICANCE.includes(step.significance) ? step.significance : 'contributing';
                    return {
                        key: index,
                        time: known ? formatGameTime(step.time) : '',
                        before: significance !== 'final' && gap > 0 ? t('coach.chain.before', { time: formatDuration(gap) }) : '',
                        type: codeLabel('chain_step', step.type),
                        significance,
                        tag: significance === 'root_cause' || significance === 'final' ? codeLabel('chain_significance', significance) : '',
                        title: codeLabel('chain_significance_title', significance),
                        text: step.description || '',
                    };
                }),
            };
        }

        // Whether `text` already says `sentence` (rule-based texts reuse the
        // root-cause summary of the chain as the "mistake" line).
        function saysAlready(text, sentence) {
            const plain = value => String(value || '').toLowerCase().replace(/[.。\s]+$/, '');
            return !!plain(sentence) && plain(text).includes(plain(sentence));
        }

        // One item of death_analyses[] / key_moments[].
        function evaluationView(item, index, kind, report) {
            const known = !isMissing(item.time);
            const positive = item.severity === 'positive';
            // analysis / mistake / better_play are '' where there is nothing to say
            const mistake = String(item.mistake || '').trim();
            const sources = (Array.isArray(item.sources) ? item.sources : []).filter(s => s && s.title).map(sourceView);
            const chain = kind === 'death' ? chainView(item.causal_chain, item.time) : null;
            // One sentence, one place: the card does not repeat it under the chain.
            if (chain && (saysAlready(mistake, chain.rootSummary) || saysAlready(item.analysis, chain.rootSummary))) chain.rootSummary = '';
            return {
                key: kind + index,
                time: known ? formatGameTime(item.time) : '',
                beforeHorn: known && Number(item.time) < 0,
                // Every death is a 'death' of severity 'critical': the section says so already.
                type: kind === 'death' && item.point_type === 'death' ? '' : codeLabel('point_type', item.point_type),
                typeTone: DotaCoach.pointTone(item.point_type),
                severity: kind === 'death' && item.severity === 'critical' ? '' : codeLabel('severity', item.severity),
                severityTone: DotaCoach.severityTone(item.severity),
                grade: gradeView(item.score),
                brief: item.brief || codeLabel('point_type', item.point_type),
                analysis: item.analysis || '',
                mistake,
                better: String(item.better_play || '').trim(),
                lesson: String(item.lesson || '').trim(),
                positive,
                mark: authorMark(item, report),
                killer: kind === 'death' ? coachKillView(report.player_slot, item.time) : null,
                chain,
                sources,
                cited: sources.filter(s => s.cited).length,
            };
        }

        // One item of phase_reviews[]. stats.assists is null for OpenDota
        // matches; kda_estimated marks an undercount (no kill log).
        // noKills: no phase of the report has a kill or a death although the
        // numbers are only estimates (no kill log): "0/0" would be no data
        // shown as a fact.
        function phaseView(phase, index, report, team, noKills) {
            const stats = phase.stats || {};
            const facts = [];
            const add = (id, value, title) => facts.push({ id, label: t('coach.phase.stat.' + id), value, title: title || null });
            if (!noKills && !isMissing(stats.kills) && !isMissing(stats.deaths)) {
                const kd = `${stats.kills}/${stats.deaths}`;
                if (stats.kda_estimated) add('kd_estimated', kd, t('coach.phase.kd_estimated_title'));
                else if (isMissing(stats.assists)) add('kd', kd, t('coach.phase.kd_title'));
                else add('kda', `${kd}/${stats.assists}`);
            }
            // A match without minute-by-minute series (OpenDota scoreboard
            // only) reports 0 for all three: that is "no data", not a zero.
            if (stats.gpm_in_phase > 0 || stats.xpm_in_phase > 0) {
                if (!isMissing(stats.last_hits_delta)) add('lh', formatSigned(stats.last_hits_delta));
                add('gpm', formatCount(stats.gpm_in_phase));
                add('xpm', formatCount(stats.xpm_in_phase));
            }
            if (stats.level_at_end > 0) add('level', formatCount(stats.level_at_end));
            if (stats.teamfight_count > 0) add('fights', `${formatCount(stats.teamfight_wins)}/${stats.teamfight_count}`, t('coach.phase.fights_title'));
            if (!isMissing(stats.net_worth_at_end)) add('net_worth', formatK(stats.net_worth_at_end));
            // Without a gold series the lead is a bare 0, not a tie.
            if (hasEconomyTimeline.value && !isMissing(stats.radiant_gold_lead) && (team === 'radiant' || team === 'dire')) {
                add('team_gold', formatSigned(team === 'radiant' ? stats.radiant_gold_lead : -stats.radiant_gold_lead), t('coach.phase.team_gold_title'));
            }
            const from = isMissing(phase.start_min) ? stats.start_min : phase.start_min;
            const to = phase.end_min === undefined ? stats.end_min : phase.end_min;
            let range = phase.time_range || stats.time_range || '';
            if (!range && !isMissing(from)) range = isMissing(to) ? t('coach.phase.from', { from }) : t('coach.phase.range', { from, to });
            const texts = value => (Array.isArray(value) ? value : []).map(x => String(x || '').trim()).filter(Boolean);
            return {
                key: (phase.phase || 'phase') + index,
                name: codeLabel('coach_phase', phase.phase),
                range,
                grade: gradeView(phase.score),
                summary: phase.summary || '',
                facts,
                lists: [
                    { id: 'highlights', items: texts(phase.highlights) },
                    { id: 'issues', items: texts(phase.issues) },
                    { id: 'suggestions', items: texts(phase.suggestions) },
                ].filter(group => group.items.length),
                mark: authorMark(phase, report),
            };
        }

        // laning_review: {} without 10-minute data, else the numbers at
        // 10:00. null values are unknown (0 is a real value).
        function laningView(review) {
            if (!review || !Object.keys(review).length) return null;
            const result = review.lane_result;
            return {
                matchup: review.lane_matchup || '',
                summary: review.summary || '',
                trades: (Array.isArray(review.key_trades) ? review.key_trades : []).filter(Boolean),
                facts: [
                    { id: 'cs', value: formatCount(review.cs_10min) },
                    { id: 'denies', value: formatCount(review.denies_10min) },
                    { id: 'lane_cs', value: formatCount(review.lane_cs_10min), note: review.cs_efficiency || '', title: t('coach.laning.lane_cs_title') },
                    { id: 'result', value: result ? codeLabel('lane_result', result) : '-', tone: DotaCoach.laneTone(result) },
                    { id: 'gold', value: formatSigned(review.lane_gold_diff), tone: deltaTone(review.lane_gold_diff), title: t('coach.laning.diff_title') },
                    { id: 'xp', value: formatSigned(review.lane_xp_diff), tone: deltaTone(review.lane_xp_diff), title: t('coach.laning.diff_title') },
                ].map(fact => ({ ...fact, label: t('coach.laning.' + fact.id) })),
            };
        }

        function deltaTone(n) {
            if (isMissing(n) || Number(n) === 0) return 'none';
            return n > 0 ? 'good' : 'bad';
        }

        // pro_comparison: benchmarks of the role. applicable false = no
        // comparison for this game mode; partial = the replay ends early and
        // only some rows could be compared (no overall rating then).
        function benchmarkView(comparison) {
            if (!comparison || !Object.keys(comparison).length) return null;
            const rows = (Array.isArray(comparison.comparisons) ? comparison.comparisons : []).map((row, index) => ({
                key: (row.metric || 'metric') + index,
                label: row.metric_label || I18N.prettify(row.metric),
                // value_text / benchmark_text are written for display (units, m:ss)
                value: row.value_text || (isMissing(row.actual_value) ? '-' : String(row.actual_value)),
                benchmark: row.benchmark_text || (isMissing(row.benchmark_avg) ? '-' : String(row.benchmark_avg)),
                rating: codeLabel('rating', row.rating),
                tone: DotaCoach.ratingTone(row.rating),
                source: codeLabel('benchmark_source', row.benchmark_source),
                sourceTitle: knownCodeLabel('benchmark_source_title', row.benchmark_source),
                sourceKind: row.benchmark_source === 'pro_reference' ? 'pro' : (row.benchmark_source === 'opendota_population' ? 'population' : 'guideline'),
                text: row.description || '',
            }));
            const applicable = comparison.applicable !== false;
            const partial = !!comparison.partial;
            const rated = applicable && !partial && !!comparison.overall_rating;
            return {
                applicable,
                partial,
                role: comparison.role_label || '',
                summary: comparison.summary || '',
                overall: rated ? codeLabel('rating', comparison.overall_rating) : '',
                overallTone: rated ? DotaCoach.ratingTone(comparison.overall_rating) : 'none',
                rows,
            };
        }

        // improvement_areas[] / strengths[]: the description is whole
        // sentences that already quote the examples.
        function areaView(area, index) {
            const occurrences = Number(area.occurrences) || 0;
            return {
                key: (area.category_code || 'area') + index,
                name: knownCodeLabel('coach_category', area.category_code) || area.category || codeLabel('coach_category', area.category_code),
                grade: gradeView(area.avg_score),
                count: occurrences > 0 ? tn('coach.areas.decisions', occurrences) : '',
                text: area.description || '',
            };
        }

        // Everything the report template shows, with codes turned into
        // labels and missing values into dashes.
        const coachView = computed(() => {
            const report = coachReport.value;
            if (!report) return null;
            const slot = report.player_slot;
            const player = playerBySlot(slot);
            const team = player ? player.team : '';
            const list = value => (Array.isArray(value) ? value.filter(x => x && typeof x === 'object') : []);
            const benchmarks = benchmarkView(report.pro_comparison);
            const phases = list(report.phase_reviews);
            const noKills = phases.length > 0 && phases.every(phase => {
                const stats = phase.stats || {};
                return stats.kda_estimated && !Number(stats.kills) && !Number(stats.deaths);
            });
            const knowledge = report.knowledge && report.knowledge.mode ? report.knowledge : null;
            const badges = [];
            badges.push(report.llm_used
                ? { id: 'engine', kind: 'llm', text: t('coach.badge.llm'), title: t('coach.badge.llm_title') }
                : { id: 'engine', kind: 'rules', text: t('coach.badge.rules'), title: t('coach.badge.rules_title') });
            if (report.degraded) badges.push({ id: 'degraded', kind: 'warn', text: t('coach.badge.degraded'), title: t('coach.degraded.text') });
            if (benchmarks && benchmarks.partial) badges.push({ id: 'partial', kind: 'warn', text: t('coach.badge.partial'), title: t('coach.badge.partial_title') });
            if (benchmarks && !benchmarks.applicable) {
                badges.push({ id: 'mode', kind: 'plain', text: t('coach.badge.no_benchmarks', { mode: gameModeLabel.value }), title: benchmarks.summary });
            }
            const sources = DotaCoach.sourceSummary(report);
            const group = (id, items) => ({ id, items });
            return {
                slot,
                lang: coachReportLang.value || locale.value,
                degraded: !!report.degraded,
                canRefresh: coachCanRefresh.value,
                hero: heroName(report.hero_name || (player && player.hero_name)),
                slug: (player && player.hero_slug) || heroSlugByName(report.hero_name),
                player: player ? playerName(player) : '',
                team,
                grade: gradeView(report.overall_grade),
                summary: report.summary || '',
                position: positionView(report.position),
                badges,
                version: isMissing(report.report_version) ? '' : t('coach.badge.version', { n: report.report_version }),
                knowledge: knowledge && {
                    label: codeLabel('knowledge_mode_short', knowledge.mode),
                    title: knownCodeLabel('knowledge_mode_title', knowledge.mode),
                    detail: [
                        isMissing(knowledge.chunks) ? '' : tn('coach.knowledge.chunks', knowledge.chunks),
                        knowledge.vectors > 0 ? tn('coach.knowledge.vectors', knowledge.vectors) : '',
                        knowledge.embedder ? t('coach.knowledge.embedder', { name: knowledge.embedder }) : '',
                    ].filter(Boolean).join(' · '),
                },
                sources: {
                    ...sources,
                    types: sources.types.map(row => ({
                        ...row,
                        kind: DotaCoach.SOURCE_TYPES.includes(row.type) ? row.type : 'other',
                        label: codeLabel('source_type', row.type),
                        title: knownCodeLabel('source_type_title', row.type),
                    })),
                },
                areas: [
                    group('improve', list(report.improvement_areas).map(areaView)),
                    group('strengths', list(report.strengths).map(areaView)),
                ].filter(g => g.items.length),
                phases: phases.map((phase, i) => phaseView(phase, i, report, team, noKills)),
                laning: laningView(report.laning_review),
                // Both lists are chronological; a moment without a time comes last.
                evaluations: [
                    group('deaths', list(report.death_analyses).map((item, i) => evaluationView(item, i, 'death', report))),
                    group('moments', list(report.key_moments).map((item, i) => evaluationView(item, i, 'moment', report))),
                ].filter(g => g.items.length),
                benchmarks,
                jsonUrl: DotaApi.url(coachPath('ai-coach', slot), { lang: coachReportLang.value || locale.value }),
            };
        });

        // Sections of the report, for the jump links above it.
        const coachSections = computed(() => {
            const view = coachView.value;
            if (!view) return [];
            return [
                { id: 'areas', show: view.areas.length > 0 },
                { id: 'phases', show: view.phases.length > 0 },
                { id: 'laning', show: !!view.laning },
                ...view.evaluations.map(g => ({ id: g.id, count: g.items.length, show: true })),
                { id: 'benchmarks', show: !!view.benchmarks },
            ].filter(section => section.show).map(section => ({ ...section, label: t('coach.nav.' + section.id) }));
        });

        function scrollToCoachSection(id) {
            const el = document.getElementById('coach-' + id);
            if (el && el.scrollIntoView) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        // --- Source lists (collapsed by default) ---
        const coachSourceKeys = computed(() => {
            const groups = coachView.value ? coachView.value.evaluations : [];
            return groups.flatMap(g => g.items).filter(item => item.sources.length).map(item => item.key);
        });
        const coachAllSourcesOpen = computed(() =>
            coachSourceKeys.value.length > 0 && coachSourceKeys.value.every(key => coachOpenSources.value[key])
        );

        function setSourcesOpen(key, open) {
            if (!!coachOpenSources.value[key] === !!open) return;
            coachOpenSources.value = { ...coachOpenSources.value, [key]: !!open };
        }

        function toggleAllSources() {
            const open = !coachAllSourcesOpen.value;
            const next = {};
            coachSourceKeys.value.forEach(key => { next[key] = open; });
            coachOpenSources.value = next;
        }

        // --- Text of a source (read on demand) ---
        // A source row opens to the entry itself: GET /api/knowledge/<id>
        // in the language of the report (its titles are in that language).
        // Texts are kept per language while the page is open.
        function sourceEntryKey(source) {
            return (coachReportLang.value || locale.value) + '|' + source.id;
        }

        function sourceEntry(source) {
            return coachEntries.value[sourceEntryKey(source)] || null;
        }

        function isSourceOpen(itemKey, source) {
            return !!source.id && !!coachOpenEntries.value[itemKey + '|' + source.id];
        }

        async function loadSourceEntry(source) {
            const lang = coachReportLang.value || locale.value;
            const key = lang + '|' + source.id;
            const have = coachEntries.value[key];
            if (have && have.state !== 'error') return;
            coachEntries.value = { ...coachEntries.value, [key]: { state: 'loading' } };
            const res = await DotaApi.get('/api/knowledge/' + encodeURIComponent(source.id), { params: { lang } });
            const entry = res.ok && typeof res.data.text === 'string'
                ? { state: 'ready', text: res.data.text }
                // the detail of a refusal is written in the language asked for
                : { state: 'error', error: res.ok ? clientError('bad_response') : { ...res.error, lang } };
            coachEntries.value = { ...coachEntries.value, [key]: entry };
        }

        function toggleSource(itemKey, source) {
            if (!source.id) return;
            const key = itemKey + '|' + source.id;
            const open = !coachOpenEntries.value[key];
            coachOpenEntries.value = { ...coachOpenEntries.value, [key]: open };
            if (open) loadSourceEntry(source);
        }

        // Printing (or "save as PDF") shows every source list: the ones
        // that were closed are opened for the print and closed again after.
        let sourcesBeforePrint = null;

        function onBeforePrint() {
            if (activeTab.value !== 'ai_coach' || !coachView.value || sourcesBeforePrint) return;
            sourcesBeforePrint = coachOpenSources.value;
            const open = {};
            coachSourceKeys.value.forEach(key => { open[key] = true; });
            coachOpenSources.value = open;
        }

        function onAfterPrint() {
            if (!sourcesBeforePrint) return;
            coachOpenSources.value = sourcesBeforePrint;
            sourcesBeforePrint = null;
        }

        async function printCoachReport() {
            onBeforePrint();
            await nextTick();
            window.print();
        }

        // Drop the match on screen and everything that belongs to it.
        function clearMatch() {
            analysisData.value = null;
            parseSource.value = '';
            matchKey.value = '';
            hasParsedData.value = true;
            matchError.value = null;
            parseRequest.value = { state: 'idle', requested: false, error: null, errorFrom: '' };
            // Tabs: stop the playback, free the charts, back to the default sub-views
            stopTeamfightPlay();
            disposeCharts();
            sortState.value = {};
            selectedTeamfight.value = 1;
            selectedLaningPlayer.value = 0;
            economyMode.value = 'gold_diff';
            combatMode.value = 'kills';
            farmMode.value = 'gold';
            objMode.value = 'timeline';
            wardFilters.value = { observers: true, sentries: true, radiant: true, dire: true };
            chatShowWheel.value = true;
            wardTimelineEnabled.value = false;
            wardTimelineTime.value = 0;
            wardShowVision.value = false;
            smokeTimelineEnabled.value = false;
            smokeTimelineTime.value = 0;
            smokeSelected.value = [];
            teamfightReplayTime.value = 0;
            resetCoach();
        }

        // Defaults of the tab sub-views for the match just put on screen.
        function initTabState() {
            const fights = teamfightData.value;
            selectedTeamfight.value = fights.length ? fights[0].index : 1;
            const laners = laningData.value.players || [];
            selectedLaningPlayer.value = laners.length ? laners[0].player_slot : 0;
            // Sources without an income breakdown (the demo data) open on the unit table
            farmMode.value = farmHasReasons('gold') || !(farmData.value.unit_kills || []).length ? 'gold' : 'units';
            resetTeamfightReplay();
        }

        // "Back": from the match view to the start page.
        function resetAnalysis() {
            clearMatch();
            syncRoute(true);
            updateDocumentTitle();
            loadCapabilities();
            window.scrollTo(0, 0);
        }

        // Escape text before it is put into HTML (chart tooltips, richText).
        function escapeHtml(text) {
            return text
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }

        // ===== Charts =====
        // One ECharts instance per chart name. The containers live inside
        // v-if blocks: an instance is re-created when its element changed,
        // and all of them are disposed when the tab or the match changes.
        const chartInstances = {};

        function useChart(name, el) {
            let chart = chartInstances[name];
            if (chart && (chart.isDisposed() || chart.getDom() !== el)) {
                if (!chart.isDisposed()) chart.dispose();
                chart = null;
            }
            if (!chart) {
                chart = echarts.init(el);
                chartInstances[name] = chart;
            }
            return chart;
        }

        function disposeCharts() {
            Object.keys(chartInstances).forEach(name => {
                if (!chartInstances[name].isDisposed()) chartInstances[name].dispose();
                delete chartInstances[name];
            });
        }

        function resizeCharts() {
            Object.values(chartInstances).forEach(chart => {
                if (!chart.isDisposed()) chart.resize();
            });
        }

        // Several state changes in one frame draw a chart once.
        const pendingRenders = new Set();
        function scheduleRender(render) {
            if (pendingRenders.has(render)) return;
            pendingRenders.add(render);
            requestAnimationFrame(() => {
                pendingRenders.delete(render);
                render();
            });
        }

        // --- Shared pieces of the chart options ---
        const AXIS_TEXT = '#a0aec0';
        const GRID_LINE = '#2d3748';
        const PLAYER_COLORS = [
            '#3b82f6', '#10b981', '#a855f7', '#ecc94b', '#f97316',
            '#ef4444', '#ec4899', '#8b5cf6', '#06b6d4', '#84cc16',
        ];

        // Payload text inside an HTML tooltip
        function esc(value) {
            return escapeHtml(String(value == null ? '' : value));
        }

        // confine: a tooltip never widens the page on a phone
        function tooltipBase(trigger) {
            return {
                trigger, confine: true,
                backgroundColor: 'rgba(26,32,44,0.96)', borderColor: '#4a5568',
                textStyle: { color: '#e2e8f0', fontSize: 12 },
            };
        }

        function chartMessage(chart, text) {
            chart.setOption({
                backgroundColor: 'transparent',
                title: { text, left: 'center', top: 'center', textStyle: { color: AXIS_TEXT, fontSize: 13, fontWeight: 'normal' } },
            }, true);
        }

        function valueAxis(formatter, extra) {
            return Object.assign({
                type: 'value',
                axisLabel: { color: AXIS_TEXT, fontSize: 11, formatter },
                nameTextStyle: { color: AXIS_TEXT },
                splitLine: { lineStyle: { color: GRID_LINE } },
            }, extra);
        }

        function categoryAxis(data, extra) {
            return Object.assign({
                type: 'category', data,
                axisLabel: { color: AXIS_TEXT, fontSize: 11 },
                axisLine: { lineStyle: { color: '#4a5568' } },
            }, extra);
        }

        function lineSeries(name, data, color, extra) {
            return Object.assign({
                name, type: 'line', data, smooth: true, symbol: 'none',
                lineStyle: { color, width: 2 }, itemStyle: { color },
            }, extra);
        }

        const kAxis = (v) => (Math.abs(v) >= 1000 ? (v / 1000).toFixed(0) + 'k' : String(v));
        const signedK = (v) => (v > 0 ? '+' : '') + (v / 1000).toFixed(1) + 'k';

        // Tooltip of a per-player line chart: heroes sorted by value, with icons.
        function playerTooltip(title, params, slugByName, format) {
            const valueOf = (p) => (Array.isArray(p.value) ? p.value[1] : p.value);
            const rows = params.filter(p => valueOf(p) != null).sort((a, b) => (Number(valueOf(b)) || 0) - (Number(valueOf(a)) || 0));
            let html = `<div style="min-width:180px"><b>${esc(title)}</b>`;
            rows.forEach(p => {
                const icon = heroIcon(slugByName[p.seriesName] || '');
                html += '<div style="display:flex;align-items:center;gap:5px;margin-top:3px;">'
                    + `<img src="${esc(icon)}" alt="" style="width:16px;height:16px;border-radius:2px;">${p.marker}`
                    + `<span>${esc(p.seriesName)}: ${esc(format(valueOf(p)))}</span></div>`;
            });
            return html + '</div>';
        }

        // --- Maps ---
        // The map picture is the CSS background of the container (.dota-map,
        // stretched over the square), so the chart only draws the markers on
        // a 0..100 grid that fills it and a resize needs no redraw.
        function gameToMap(x, y) {
            return DotaGame.toMap(x, y);
        }

        function worldToMapLength(distance) {
            return DotaGame.mapLength(distance);
        }

        function mapOption(series, formatter, extra) {
            return Object.assign({
                backgroundColor: 'transparent',
                grid: { left: 0, right: 0, top: 0, bottom: 0 },
                tooltip: Object.assign(tooltipBase('item'), { formatter }),
                xAxis: { type: 'value', min: 0, max: 100, show: false },
                yAxis: { type: 'value', min: 0, max: 100, show: false },
                series,
            }, extra);
        }

        // --- Vision map ---
        const WARD_FILL = { observer: '#ecc94b', sentry: '#63b3ed' };
        // Same colours as the outcome chips of the ward list (.state-* in app.css)
        const WARD_STATE_COLOR = { dewarded: '#fc8181', denied: '#d6bcfa', destroyed: '#f6ad55', expired: '#a0aec0', active: '#68d391' };

        function renderWardMap() {
            if (!wardMap.value || !hasWardData.value) return;
            const chart = useChart('wards', wardMap.value);
            const liveOnly = wardTimelineEnabled.value;   // the slider shows the wards standing at that time
            const observers = [], sentries = [], dewards = [], denies = [], vision = [];

            timelineWards.value.forEach(w => {
                if (!DotaGame.onMap(w.x, w.y)) return;
                const point = gameToMap(w.x, w.y);
                const type = w.ward_type === 'observer' ? 'observer' : 'sentry';
                const state = DotaGame.wardState(w);
                const entry = {
                    value: point, _ward: w,
                    itemStyle: {
                        color: WARD_FILL[type], borderColor: teamColor(w.team), borderWidth: 2,
                        opacity: !liveOnly && state !== 'active' ? 0.6 : 1,
                    },
                };
                (type === 'observer' ? observers : sentries).push(entry);
                if (!liveOnly && state === 'dewarded') dewards.push({ value: point });
                if (!liveOnly && state === 'denied') denies.push({ value: point });
                if (wardShowVision.value) {
                    vision.push({ value: [point[0], point[1], worldToMapLength(wardRadii.value[type])], type, team: w.team });
                }
            });

            const series = [];
            if (vision.length) {
                series.push({
                    name: t('wards_vision_range'), type: 'custom', data: vision, z: 5, silent: true,
                    renderItem(params, api) {
                        const d = vision[params.dataIndex];
                        const center = api.coord([api.value(0), api.value(1)]);
                        const radius = api.size([api.value(2), 0])[0];
                        const observer = d.type === 'observer';
                        return {
                            type: 'circle',
                            shape: { cx: center[0], cy: center[1], r: radius },
                            style: {
                                fill: observer ? 'rgba(236,201,75,0.13)' : 'rgba(99,179,237,0.10)',
                                stroke: teamColor(d.team), opacity: 0.9,
                                lineWidth: 1.5, lineDash: observer ? null : [5, 4],
                            },
                        };
                    },
                });
            }
            series.push({ name: t('wards_observer'), type: 'scatter', data: observers, symbol: 'circle', symbolSize: 15, z: 10 });
            series.push({ name: t('wards_sentry'), type: 'scatter', data: sentries, symbol: 'diamond', symbolSize: 13, z: 10 });
            if (dewards.length) {
                series.push({
                    name: codeLabel('ward_state', 'dewarded'), type: 'scatter', data: dewards, silent: true, z: 12,
                    symbol: 'path://M-1,-1L1,1M1,-1L-1,1', symbolSize: 9,
                    itemStyle: { color: 'transparent', borderColor: '#ef4444', borderWidth: 2 },
                });
            }
            if (denies.length) {
                // The game's own sign for a deny: an exclamation mark
                series.push({
                    name: codeLabel('ward_state', 'denied'), type: 'scatter', data: denies, silent: true, z: 12,
                    symbol: 'path://M-1,-5L1,-5L1,1.5L-1,1.5ZM-1,3L1,3L1,5L-1,5Z', symbolSize: [4, 13],
                    itemStyle: { color: '#fff', borderColor: '#6b46c1', borderWidth: 1 },
                });
            }

            chart.setOption(mapOption(series, p => {
                const w = p.data && p.data._ward;
                if (!w) return '';
                const stateColor = WARD_STATE_COLOR[DotaGame.wardState(w)];
                return `<b>${esc(wardOwnerName(w))}</b> · ${esc(teamLabel(w.team))}<br/>${esc(wardTypeName(w))}`
                    + `<br/>${esc(t('wards_placed'))}: ${esc(eventTime(w))}`
                    + `<br/><span style="color:${stateColor}">${esc(wardStateDetail(w))}</span>`
                    + `<br/>${esc(t('wards.col_life'))}: ${esc(formatGameTime(wardLife(w), '-'))}`;
            }), true);
        }

        // --- Smoke map and timeline ---
        function smokeTooltip(s) {
            let html = `<b>${esc(teamLabel(s.team))}</b> · ${esc(eventTime(s))}`;
            if (s.hero_name) html += `<br/>${esc(t('smokes.used_by', { hero: heroName(s.hero_name) }))}`;
            const members = (s.participants || []).map(m => heroName(m.hero_name));
            if (members.length) html += `<br/>${esc(tn('smokes.heroes', members.length))}: ${esc(members.join(', '))}`;
            if (s.duration != null) html += `<br/>${esc(t('smokes_duration_label'))}: ${esc(t('time.sec', { s: s.duration }))}`;
            html += `<br/>${esc(t('smokes_result_label'))}: <span style="color:${smokeResultColor(s.result)}">${esc(smokeResultText(s))}</span>`;
            return html;
        }

        function renderSmokeMap() {
            if (!smokeMapRef.value || !hasSmokeData.value) return;
            const chart = useChart('smokeMap', smokeMapRef.value);
            const starts = [], ends = [], paths = [], arrows = [];

            smokeTimeline.value.forEach((s, index) => {
                if (smokeTimelineEnabled.value && s.time > smokeTimelineTime.value) return;
                if (smokeSelected.value.length && !smokeSelected.value.includes(index)) return;
                if (!DotaGame.onMap(s.x, s.y)) return;
                const color = teamColor(s.team);
                const start = gameToMap(s.x, s.y);
                starts.push({
                    value: start, _smoke: s,
                    itemStyle: { color, borderColor: 'rgba(255,255,255,0.85)', borderWidth: 2, shadowColor: color, shadowBlur: 6 },
                });
                const route = (s.path || []).filter(pt => Array.isArray(pt) && DotaGame.onMap(pt[0], pt[1]));
                if (!route.length) return;
                const points = [start, ...route.map(pt => gameToMap(pt[0], pt[1]))];
                const end = points[points.length - 1];
                paths.push({ points, color });
                arrows.push({ from: points[points.length - 2], to: end, color });
                // Where the smoke ended, coloured by what came of it
                ends.push({
                    value: end, _smoke: s,
                    itemStyle: { color: smokeResultColor(s.result), borderColor: color, borderWidth: 1.5 },
                });
            });

            const series = [];
            if (paths.length) {
                series.push({
                    name: t('smokes_movement_path'), type: 'custom', z: 3, silent: true,
                    data: paths.map((_, i) => ({ value: [i] })),
                    renderItem(params, api) {
                        const d = paths[params.dataIndex];
                        return {
                            type: 'polyline',
                            shape: { points: d.points.map(pt => api.coord(pt)) },
                            style: { stroke: d.color, opacity: 0.75, lineWidth: 2.5, lineDash: [8, 4], fill: 'none' },
                        };
                    },
                });
                series.push({
                    name: t('smokes_direction'), type: 'custom', z: 4, silent: true,
                    data: arrows.map((_, i) => ({ value: [i] })),
                    renderItem(params, api) {
                        const d = arrows[params.dataIndex];
                        const to = api.coord(d.to), from = api.coord(d.from);
                        const angle = Math.atan2(to[1] - from[1], to[0] - from[0]);
                        const size = 9;
                        return {
                            type: 'polygon',
                            shape: {
                                points: [
                                    to,
                                    [to[0] - size * Math.cos(angle - 0.4), to[1] - size * Math.sin(angle - 0.4)],
                                    [to[0] - size * Math.cos(angle + 0.4), to[1] - size * Math.sin(angle + 0.4)],
                                ],
                            },
                            style: { fill: d.color, opacity: 0.85 },
                        };
                    },
                });
            }
            series.push({ name: t('smokes_start_pos'), type: 'scatter', data: starts, symbol: 'circle', symbolSize: 16, z: 10 });
            if (ends.length) series.push({ name: t('smokes_break_point'), type: 'scatter', data: ends, symbol: 'triangle', symbolSize: 13, z: 11 });

            chart.setOption(mapOption(series, p => (p.data && p.data._smoke ? smokeTooltip(p.data._smoke) : '')), true);
        }

        // One dot per smoke on a time axis: size = heroes, fill = result, ring = team.
        function renderSmokeTimeline() {
            if (!smokeChart.value || !hasSmokeData.value) return;
            const chart = useChart('smokeTimeline', smokeChart.value);
            const timeline = smokeTimeline.value;
            const rows = ['radiant', 'dire'];
            if (timeline.some(s => teamId(s.team) === 'none')) rows.push('none');
            const range = smokeTimeRange.value;

            chart.setOption({
                backgroundColor: 'transparent',
                tooltip: Object.assign(tooltipBase('item'), { formatter: p => (timeline[p.data._index] ? smokeTooltip(timeline[p.data._index]) : '') }),
                grid: { left: 64, right: 20, top: 16, bottom: 44 },
                xAxis: valueAxis(v => v + ':00', {
                    name: t('chart_game_time'), nameLocation: 'middle', nameGap: 28,
                    // one spare minute on the left keeps a pre-horn smoke clear of the labels
                    min: Math.floor(range.min / 60) - (range.min < 0 ? 1 : 0), max: Math.ceil(range.max / 60), minInterval: 1,
                }),
                yAxis: {
                    type: 'category', inverse: true, data: rows.map(id => (id === 'none' ? codeLabel('team', '') : codeLabel('team', id))),
                    axisLabel: { color: AXIS_TEXT }, axisLine: { lineStyle: { color: '#4a5568' } }, splitLine: { show: false },
                },
                series: [{
                    type: 'scatter',
                    data: timeline.map((s, index) => ({
                        value: [Math.round(s.time / 6) / 10, rows.indexOf(teamId(s.team))],
                        _index: index,
                        symbolSize: 8 + 3 * Math.min(5, Math.max(1, Number(s.num_players) || 1)),
                        itemStyle: {
                            color: smokeResultColor(s.result), borderColor: teamColor(s.team), borderWidth: 2,
                            opacity: smokeSelected.value.length && !smokeSelected.value.includes(index) ? 0.3 : 1,
                        },
                    })),
                }],
            }, true);
            // A dot selects its smoke, like its line in the list.
            chart.off('click');
            chart.on('click', p => { if (p.data && p.data._index != null) toggleSmokeSelect(p.data._index); });
        }

        // --- Teamfight map ---
        // Positions are replayed over tfWindow; without positions (OpenDota)
        // only the places where heroes died are known.
        function renderTeamfightMap() {
            const tf = currentTeamfight.value;
            if (!teamfightMap.value || !tf) return;
            const chart = useChart('teamfight', teamfightMap.value);
            const now = teamfightReplayTime.value;
            const range = tfWindow.value;
            const kills = tf.kill_timeline || [];
            const rings = [], icons = [], trails = [], casts = [], marks = [];
            // A fallen hero fades during playback; the death-spot view keeps it readable
            const fade = range.hasPositions ? 0.35 : 0.85;

            (tf.players || []).forEach(p => {
                const team = p.team || (p.player_slot >= 128 ? 'dire' : 'radiant');
                const color = teamColor(team);
                const track = p.positions || [];
                const ownKills = kills.filter(k => k.victim_slot === p.player_slot && k.time_offset <= now);
                // A probable reincarnation (counted false) is not a death.
                const down = downAt(tf, p, ownKills.filter(k => k.counted !== false), now);
                const isDead = !!down;
                // A fallen hero stays where it died (its track goes on to the
                // fountain) and fades until it is alive again.
                const shownAt = isDead ? Math.min(now, down.since) : now;

                let pos = null;
                if (track.length) pos = interpolatePosition(track, shownAt);
                else if (DotaGame.onMap(p.x, p.y) && (range.hasPositions || p.deaths > 0)) pos = { x: p.x, y: p.y };
                if (!pos || !DotaGame.onMap(pos.x, pos.y)) return;
                const point = gameToMap(pos.x, pos.y);
                const name = heroName(p.hero_name);
                const key = 'p' + p.player_slot;

                if (track.length) {
                    // One line per stretch that was walked: a teleport starts a new one
                    // instead of drawing a straight line across the map.
                    let walked = [];
                    let before = null;
                    [...track.filter(q => q.time_offset <= shownAt), { x: pos.x, y: pos.y, time_offset: shownAt }].forEach(q => {
                        if (before && DotaGame.isJump(before, q, q.time_offset - before.time_offset)) {
                            if (walked.length >= 2) trails.push({ points: walked, color });
                            walked = [];
                        }
                        walked.push(gameToMap(q.x, q.y));
                        before = q;
                    });
                    if (walked.length >= 2) trails.push({ points: walked, color });
                }
                rings.push({
                    name: key, value: point,
                    itemStyle: { color: 'rgba(15,23,42,0.75)', borderColor: color, borderWidth: 2, opacity: isDead ? fade + 0.1 : 1 },
                });
                const iconUrl = DotaCatalog.heroIconUrl(p.hero_slug || heroSlugByName(p.hero_name));
                icons.push({
                    name: key, value: point, _player: p,
                    symbol: iconUrl ? `image://${iconUrl}` : 'circle',
                    symbolSize: iconUrl ? 20 : 12,
                    itemStyle: { color, opacity: isDead ? fade : 1 },
                    label: { show: !iconUrl, formatter: DotaCatalog.initials(name), color: '#fff', fontSize: 9, fontWeight: 'bold' },
                });

                ownKills.forEach((kill, i) => {
                    const at = track.length ? interpolatePosition(track, kill.time_offset) : pos;
                    if (!at || !DotaGame.onMap(at.x, at.y)) return;
                    marks.push({
                        name: `${key}k${i}`, value: gameToMap(at.x, at.y), _kill: kill,
                        itemStyle: { color: 'transparent', borderColor: kill.counted === false ? '#ecc94b' : '#ef4444', borderWidth: 2.5 },
                    });
                });

                (p.ability_casts || []).forEach((cast, i) => {
                    if (cast.time_offset > now || !DotaGame.onMap(cast.x, cast.y)) return;
                    casts.push({
                        name: `${key}c${i}`, value: gameToMap(cast.x, cast.y), _cast: cast, _hero: name,
                        itemStyle: {
                            color: '#ecc94b', borderColor: color, borderWidth: 1.5,
                            opacity: now - cast.time_offset <= 4 ? 1 : 0.35,   // fresh casts stand out
                        },
                    });
                });
            });

            const series = [
                {
                    id: 'trails', name: t('smokes_movement_path'), type: 'custom', z: 3, silent: true, animation: false,
                    data: trails.map((_, i) => ({ value: [i] })),
                    renderItem(params, api) {
                        const d = trails[params.dataIndex];
                        if (!d) return null;
                        return {
                            type: 'polyline',
                            shape: { points: d.points.map(pt => api.coord(pt)) },
                            style: { stroke: d.color, opacity: 0.45, lineWidth: 2, fill: 'none' },
                        };
                    },
                },
                { id: 'casts', name: t('teamfights_ability_cast'), type: 'scatter', data: casts, symbol: 'diamond', symbolSize: 10, z: 8, animation: false },
                { id: 'marks', name: t('teamfights.death_marker'), type: 'scatter', data: marks, symbol: 'path://M-1,-1L1,1M1,-1L-1,1', symbolSize: 14, z: 12, animation: false },
                { id: 'rings', type: 'scatter', data: rings, symbol: 'circle', symbolSize: 26, z: 10, silent: true },
                { id: 'heroes', name: t('teamfights_hero_pos'), type: 'scatter', data: icons, z: 11 },
            ];

            const option = mapOption(series, p => {
                const d = p.data || {};
                if (d._cast) return `<b>${esc(abilityName(d._cast.ability_key, d._cast.ability_name))}</b><br/>${esc(d._hero)} · ${esc(formatGameTime(tf.start_time + d._cast.time_offset))}`;
                if (d._kill) {
                    const k = d._kill;
                    const what = k.counted === false ? t('combat.uncounted') : t('teamfights_died');
                    return `<b>${esc(heroName(k.victim))}</b> ${esc(what)} · ${esc(eventTime(k))}<br/>← ${esc(killerLabel(k))}`;
                }
                const pl = d._player;
                if (!pl) return '';
                const used = (pl.abilities_used || []).map((key, i) => abilityName(key, (pl.abilities_used_names || [])[i]));
                return `<b>${esc(heroName(pl.hero_name))}</b> · ${esc(teamLabel(pl.team))}`
                    + `<br/>K/D: ${esc(formatCount(pl.kills))}/${esc(formatCount(pl.deaths))}`
                    + `<br/>${esc(t('teamfights_damage'))}: ${esc(formatNumber(pl.damage_dealt))}`
                    + `<br/>${esc(t('teamfights_taken'))}: ${esc(formatNumber(pl.damage_taken))}`
                    + (used.length ? `<br/>${esc(t('teamfights_abilities'))}: ${esc(used.join(', '))}` : '');
            }, {
                // Heroes glide from frame to frame while the fight plays
                animationDurationUpdate: teamfightPlaying.value ? TF_TICK_MS : 0,
                animationEasingUpdate: 'linear',
            });
            // Keep the series (and their points, matched by name) between frames
            chart.setOption(option, { replaceMerge: ['series'] });
        }

        // --- Economy chart ---
        function renderEconomyChart() {
            if (!economyChart.value || !hasEconomyTimeline.value) return;
            const chart = useChart('economy', economyChart.value);
            const timeline = analysisData.value.gold_xp_timeline;
            const mode = economyMode.value;
            let labels = timeline.map(s => s.time_str);
            let series, yAxis, formatter, visualMap;
            const legend = { show: false, textStyle: { color: AXIS_TEXT, fontSize: 10 }, top: 0, type: 'scroll' };

            if (mode === 'gold_diff' || mode === 'xp_diff') {
                const gold = mode === 'gold_diff';
                const name = gold ? t('economy_gold_diff') : t('economy_xp_diff');
                // Line and area take the colour of the side that leads: the
                // pieces split at y = 0, wherever the zero line is in the chart.
                const values = timeline.map(s => Number(gold ? s.gold_diff : s.xp_diff) || 0);
                series = [{
                    name, type: 'line', data: values,
                    symbol: 'none', lineStyle: { width: 2 }, areaStyle: { opacity: 0.22 },
                    markLine: { silent: true, symbol: 'none', data: [{ yAxis: 0 }], lineStyle: { color: '#718096', type: 'dashed' }, label: { show: false } },
                }];
                // (ECharts needs closed pieces to colour a line: the bound lies beyond every value)
                const bound = 2 * Math.max(1, ...values.map(Math.abs));
                visualMap = {
                    show: false, type: 'piecewise', dimension: 1, seriesIndex: 0,
                    pieces: [{ gt: 0, lte: bound, color: TEAM_COLOR.radiant }, { gte: -bound, lte: 0, color: TEAM_COLOR.dire }],
                };
                yAxis = valueAxis(signedK);
                formatter = params => {
                    const point = params && params[0];
                    if (!point || point.value == null) return '';
                    const lead = point.value > 0 ? t('economy_radiant_lead') : point.value < 0 ? t('economy_dire_lead') : '';
                    return `<b>${esc(point.axisValueLabel)}</b><br/>${esc(name)}: `
                        + `<span style="color:${teamColor(swingTeam(point.value))}">${esc(signedK(point.value))}</span>`
                        + (lead ? ` (${esc(lead)})` : '');
                };
            } else if (mode === 'gold_total' || mode === 'xp_total') {
                const gold = mode === 'gold_total';
                series = [
                    lineSeries(codeLabel('team', 'radiant'), timeline.map(s => (gold ? s.radiant_gold : s.radiant_xp)), TEAM_COLOR.radiant),
                    lineSeries(codeLabel('team', 'dire'), timeline.map(s => (gold ? s.dire_gold : s.dire_xp)), TEAM_COLOR.dire),
                ];
                yAxis = valueAxis(kAxis);
                legend.show = true;
                formatter = params => {
                    if (!params || !params.length) return '';
                    return `<b>${esc(params[0].axisValueLabel)}</b>` + params.map(p =>
                        `<br/>${p.marker}${esc(p.seriesName)}: ${esc(formatK(p.value))}`).join('');
                };
            } else {
                // Per-player series of graphs.player_gold / player_xp / player_lh: one value a minute
                const field = mode === 'player_gold' ? 'gold_t' : mode === 'player_xp' ? 'xp_t' : 'lh_t';
                const list = (analysisData.value.graphs || {})[mode] || [];
                const length = list.reduce((max, p) => Math.max(max, (p[field] || []).length), 0);
                if (!length) {
                    chartMessage(chart, t('farm_no_data'));
                    return;
                }
                labels = Array.from({ length }, (_, minute) => `${minute}:00`);
                const slugByName = {};
                series = list.map((p, index) => {
                    const name = heroName(p.hero_name);
                    slugByName[name] = p.hero_slug || '';
                    return lineSeries(name, p[field] || [], PLAYER_COLORS[index % PLAYER_COLORS.length]);
                });
                yAxis = valueAxis(mode === 'player_lh' ? String : kAxis);
                legend.show = true;
                formatter = params => (params && params.length
                    ? playerTooltip(params[0].axisValueLabel, params, slugByName, v => (mode === 'player_lh' ? formatCount(v) : formatK(v)))
                    : '');
            }

            const option = {
                backgroundColor: 'transparent',
                tooltip: Object.assign(tooltipBase('axis'), { formatter }),
                legend,
                grid: { left: 56, right: 16, top: legend.show ? 36 : 16, bottom: 32 },
                xAxis: categoryAxis(labels, { boundaryGap: false }),
                yAxis,
                series,
            };
            if (visualMap) option.visualMap = visualMap;
            chart.setOption(option, true);
        }

        // --- Laning charts ---
        // Last hits over the first ten minutes: the five-second samples of a
        // replay, else the per-minute series every source has.
        function renderLaningCsTimeline() {
            if (!laningCsTimeline.value) return;
            const chart = useChart('laningCs', laningCsTimeline.value);
            const players = laningData.value.players || [];
            const perMinute = {};
            (farmData.value.lh_timeline || []).forEach(p => { perMinute[p.player_slot] = p.lh_t || []; });
            const slugByName = {};
            const series = players.map((p, index) => {
                const name = heroName(p.hero_name);
                slugByName[name] = p.hero_slug || '';
                const samples = (p.laning_snapshots || []).filter(s => s.time >= 0 && s.time <= 600);
                const data = samples.length
                    ? samples.map(s => [s.time / 60, s.last_hits])
                    : (perMinute[p.player_slot] || []).slice(0, 11).map((lh, minute) => [minute, lh]);
                return lineSeries(name, data, PLAYER_COLORS[index % PLAYER_COLORS.length]);
            }).filter(s => s.data.length > 0);
            if (!series.length) {
                chartMessage(chart, t('laning_no_cs_data'));
                return;
            }
            const option = laningLineOption(t('laning_cs_title'), series, valueAxis(String));
            option.tooltip.formatter = params => (params && params.length
                ? playerTooltip(formatGameTime(Math.round(params[0].value[0] * 60)), params, slugByName, v => `${formatCount(v)} ${t('laning_lh')}`)
                : '');
            chart.setOption(option, true);
        }

        // Health and mana of the selected hero (five-second samples of a replay).
        function renderLaningHpTimeline() {
            const player = selectedLaning.value;
            if (!laningHpTimeline.value) return;
            const chart = useChart('laningHp', laningHpTimeline.value);
            const samples = ((player && player.laning_snapshots) || []).filter(s => s.time >= 0 && s.time <= 600);
            const share = (value, max) => (max > 0 ? Math.round(value / max * 100) : null);
            const hp = samples.filter(s => s.max_hp > 0).map(s => [s.time / 60, share(s.hp, s.max_hp)]);
            const mana = samples.filter(s => s.max_mana > 0).map(s => [s.time / 60, share(s.mana, s.max_mana)]);
            if (!hp.length) {
                chartMessage(chart, t('laning_no_hp_data'));
                return;
            }
            const series = [
                lineSeries(t('laning.hp'), hp, '#48bb78', { smooth: false, areaStyle: { color: 'rgba(72,187,120,0.12)' } }),
                lineSeries(t('laning.mana'), mana, '#63b3ed', { smooth: false, lineStyle: { color: '#63b3ed', width: 1.5 } }),
            ];
            const option = laningLineOption(t('laning.vitals_title'), series, valueAxis(v => v + '%', { min: 0, max: 100 }));
            option.tooltip.formatter = params => {
                if (!params || !params.length) return '';
                return `<b>${esc(formatGameTime(Math.round(params[0].value[0] * 60)))}</b>`
                    + params.map(p => `<br/>${p.marker}${esc(p.seriesName)}: ${esc(formatCount(p.value[1]))}%`).join('');
            };
            chart.setOption(option, true);
        }

        function laningLineOption(title, series, yAxis) {
            return {
                backgroundColor: 'transparent',
                title: { text: title, left: 4, top: 0, textStyle: { color: '#e2e8f0', fontSize: 13, fontWeight: 500 } },
                tooltip: tooltipBase('axis'),
                legend: { textStyle: { color: AXIS_TEXT, fontSize: 10 }, top: 26, left: 4, right: 4, type: 'scroll' },
                grid: { left: 44, right: 16, top: 62, bottom: 44 },
                xAxis: valueAxis(v => v + ':00', { name: t('chart_game_time'), nameLocation: 'middle', nameGap: 28, min: 0, max: 10, interval: 2 }),
                yAxis,
                series,
            };
        }

        // Where the selected hero stood in the first ten minutes: lane_pos is
        // [[x, y, samples], ...] (128-unit cells) from every source.
        function renderLaningHeatmap() {
            const player = selectedLaning.value;
            if (!laningHeatmap.value || !player) return;
            const chart = useChart('laningHeat', laningHeatmap.value);
            const cells = (player.lane_pos || []).filter(c => Array.isArray(c) && DotaGame.onMap(c[0], c[1]) && c[2] > 0);
            // The fountain cell holds most samples: a log scale keeps the lane readable
            const top = Math.log(1 + cells.reduce((max, c) => Math.max(max, c[2]), 1));
            const data = cells.map(c => {
                const weight = Math.log(1 + c[2]) / top;
                return {
                    value: gameToMap(c[0], c[1]),
                    symbolSize: 7 + 11 * weight,
                    itemStyle: { color: `hsl(${Math.round(210 - 210 * weight)}, 90%, 58%)`, opacity: 0.35 + 0.5 * weight },
                };
            }).sort((a, b) => a.symbolSize - b.symbolSize);
            chart.setOption(mapOption([{ type: 'scatter', data, symbol: 'circle', silent: true, z: 5, animation: false }], () => '', { tooltip: { show: false } }), true);
        }

        // --- Farm charts ---
        // gold_reasons / xp_reasons: { code: amount } per player. Colours
        // belong to the codes; an unknown code takes one of the spare colours.
        const REASON_ORDER = [
            'creep', 'neutral', 'hero', 'building', 'roshan', 'bounty_rune', 'wisdom_rune', 'outpost', 'ward', 'courier',
            'ability', 'illusion', 'passive', 'shared', 'catch_up', 'tome', 'other',
            'sell', 'abandon', 'cheat', 'selection_penalty', 'purchase', 'ability_cost', 'buyback', 'death',
        ];
        const REASON_COLOR = {
            creep: '#48bb78', neutral: '#2f855a', hero: '#fc8181', building: '#63b3ed', roshan: '#ecc94b',
            bounty_rune: '#f6ad55', wisdom_rune: '#d6bcfa', outpost: '#81e6d9', ward: '#b794f4', courier: '#f687b3',
            ability: '#4fd1c5', illusion: '#9ae6b4', passive: '#cbd5e0', shared: '#90cdf4', catch_up: '#90cdf4', tome: '#d6bcfa',
            other: '#a0aec0', sell: '#9f7aea', abandon: '#4a5568', cheat: '#4a5568', selection_penalty: '#4a5568',
            purchase: '#805ad5', ability_cost: '#319795', buyback: '#ed8936', death: '#718096',
        };
        const SPARE_COLORS = ['#f6e05e', '#fbb6ce', '#c3dafe', '#feb2b2', '#b2f5ea'];

        function renderFarmReasonChart() {
            if (!farmReasonChart.value) return;
            const chart = useChart('farmReasons', farmReasonChart.value);
            const kind = farmMode.value === 'xp' ? 'xp' : 'gold';
            const data = (kind === 'xp' ? farmData.value.xp_reasons : farmData.value.gold_reasons) || [];
            const seen = new Set();
            data.forEach(p => Object.keys(p.reasons || {}).forEach(code => seen.add(code)));
            const codes = [...REASON_ORDER.filter(code => seen.has(code)), ...[...seen].filter(code => !REASON_ORDER.includes(code)).sort()];
            if (!codes.length) {
                chartMessage(chart, t('farm_no_data'));
                return;
            }
            let spare = 0;
            const series = codes.map(code => ({
                name: reasonLabel(kind, code), type: 'bar', stack: 'total', barMaxWidth: 46,
                data: data.map(p => (p.reasons || {})[code] || 0),
                itemStyle: { color: REASON_COLOR[code] || SPARE_COLORS[spare++ % SPARE_COLORS.length] },
            }));
            chart.setOption({
                backgroundColor: 'transparent',
                tooltip: Object.assign(tooltipBase('axis'), {
                    axisPointer: { type: 'shadow' },
                    formatter: params => {
                        if (!params || !params.length) return '';
                        const rows = params.filter(p => p.value);
                        const total = rows.reduce((sum, p) => sum + p.value, 0);
                        return `<b>${esc(params[0].axisValueLabel)}</b>`
                            + rows.sort((a, b) => b.value - a.value).map(p => `<br/>${p.marker}${esc(p.seriesName)}: ${esc(formatSigned(p.value))}`).join('')
                            + `<br/><b>${esc(t('total'))}: ${esc(formatK(total))}</b>`;
                    },
                }),
                legend: { textStyle: { color: AXIS_TEXT, fontSize: 11 }, top: 0, type: 'scroll' },
                grid: { left: 52, right: 16, top: 40, bottom: 70 },
                xAxis: categoryAxis(data.map(p => heroName(p.hero_name)), { axisLabel: { color: AXIS_TEXT, fontSize: 10, rotate: 35, interval: 0 } }),
                yAxis: valueAxis(kAxis),
                series,
            }, true);
        }

        function renderFarmLhChart() {
            if (!farmLhChart.value) return;
            const chart = useChart('farmLh', farmLhChart.value);
            const data = (farmData.value.lh_timeline || []).filter(p => (p.lh_t || []).length > 0);
            if (!data.length) {
                chartMessage(chart, t('farm_no_data'));
                return;
            }
            const length = data.reduce((max, p) => Math.max(max, p.lh_t.length), 0);
            const slugByName = {};
            const series = data.map((p, index) => {
                const name = heroName(p.hero_name);
                slugByName[name] = p.hero_slug || '';
                return lineSeries(name, p.lh_t, PLAYER_COLORS[index % PLAYER_COLORS.length]);
            });
            chart.setOption({
                backgroundColor: 'transparent',
                tooltip: Object.assign(tooltipBase('axis'), {
                    formatter: params => (params && params.length ? playerTooltip(params[0].axisValueLabel, params, slugByName, formatCount) : ''),
                }),
                legend: { textStyle: { color: AXIS_TEXT, fontSize: 10 }, top: 0, type: 'scroll' },
                grid: { left: 48, right: 16, top: 36, bottom: 32 },
                xAxis: categoryAxis(Array.from({ length }, (_, minute) => `${minute}:00`), { boundaryGap: false }),
                yAxis: valueAxis(String),
                series,
            }, true);
        }

        function renderFarmCharts() {
            if (farmMode.value === 'gold' || farmMode.value === 'xp') renderFarmReasonChart();
            if (farmMode.value === 'lh') renderFarmLhChart();
        }

        // --- Building map (objectives tab) ---
        // match_overview.buildings: { name, team, type: tower | barracks | fort, x, y, destroyed_time }
        const BUILDING_SHAPE = { tower: ['roundRect', 13], barracks: ['rect', 9], fort: ['diamond', 22] };

        function renderBuildingMap() {
            if (!buildingMap.value || !buildings.value.length) return;
            const chart = useChart('buildings', buildingMap.value);
            const standing = [], fallen = [], crosses = [];
            buildings.value.forEach(b => {
                const point = gameToMap(b.x, b.y);
                const [symbol, size] = BUILDING_SHAPE[b.type] || BUILDING_SHAPE.tower;
                const color = teamColor(b.team);
                const gone = b.destroyed_time != null;
                (gone ? fallen : standing).push({
                    value: point, _building: b, symbol, symbolSize: size,
                    itemStyle: gone
                        ? { color: 'rgba(15,23,42,0.7)', borderColor: color, borderWidth: 1.5, opacity: 0.8 }
                        : { color, borderColor: 'rgba(255,255,255,0.9)', borderWidth: 1.5 },
                });
                if (gone) crosses.push({ value: point });
            });
            const series = [
                { name: t('objectives.destroyed'), type: 'scatter', data: fallen, z: 8 },
                { type: 'scatter', data: crosses, silent: true, z: 9, symbol: 'path://M-1,-1L1,1M1,-1L-1,1', symbolSize: 8, itemStyle: { color: 'transparent', borderColor: '#ef4444', borderWidth: 2 } },
                { name: t('objectives.standing'), type: 'scatter', data: standing, z: 10 },
            ];
            chart.setOption(mapOption(series, p => {
                const b = p.data && p.data._building;
                if (!b) return '';
                const info = DotaGame.parseBuilding(b.name);
                const name = info ? buildingLabel(info) : (unitLabel(b.name, b.team) || t('building.other', { team: teamLabel(b.team) }));
                const state = b.destroyed_time != null
                    ? `<span style="color:#fc8181">${esc(t('objectives.destroyed_at', { time: formatGameTime(b.destroyed_time, '-') }))}</span>`
                    : `<span style="color:#68d391">${esc(t('objectives.standing'))}</span>`;
                return `<b>${esc(name)}</b><br/>${state}`;
            }), true);
        }

        // Tab bar keyboard pattern: the arrow keys, Home and End move between
        // the tabs (only the active one is in the Tab order).
        function focusActiveTab() {
            const tab = document.getElementById('tab-' + activeTab.value);
            if (tab) tab.focus({ preventScroll: true });
        }

        function onTabKeydown(event) {
            const ids = tabs.value.map(tab => tab.id);
            const at = ids.indexOf(activeTab.value);
            const to = { ArrowRight: at + 1, ArrowLeft: at - 1, Home: 0, End: ids.length - 1 }[event.key];
            if (to === undefined || at < 0) return;
            event.preventDefault();
            activeTab.value = ids[(to + ids.length) % ids.length];
            nextTick(focusActiveTab);
        }

        // Escape closes a tooltip that opened on keyboard focus.
        function closeTip(event) {
            const tip = event.target && event.target.closest ? event.target.closest('.coach-tip') : null;
            if (tip && tip.blur) tip.blur();
        }

        // The tab row scrolls sideways on narrow screens: keep the active
        // tab visible (after a deep link or a back / forward step).
        function revealActiveTab() {
            const tab = document.querySelector('.od-tab-active');
            const row = tab && tab.parentElement;
            if (!row || row.scrollWidth <= row.clientWidth) return;
            const left = tab.offsetLeft - row.offsetLeft;
            if (left < row.scrollLeft || left + tab.offsetWidth > row.scrollLeft + row.clientWidth) {
                row.scrollLeft = Math.max(0, left - (row.clientWidth - tab.offsetWidth) / 2);
            }
        }

        // Draw the charts of the tab on screen. Called when the tab changes,
        // when a match is put on screen (the tab may be the same as before,
        // e.g. a deep link) and when the language changes.
        function renderActiveTab() {
            if (!analysisData.value) return;
            const tab = activeTab.value;
            if (tab === 'economy') renderEconomyChart();
            else if (tab === 'wards') renderWardMap();
            else if (tab === 'smokes') { renderSmokeMap(); renderSmokeTimeline(); }
            else if (tab === 'teamfights') renderTeamfightMap();
            else if (tab === 'laning') { renderLaningCsTimeline(); renderLaningHpTimeline(); renderLaningHeatmap(); }
            else if (tab === 'farm') renderFarmCharts();
            else if (tab === 'objectives') renderBuildingMap();
        }

        // --- Watchers ---
        watch(activeTab, async () => {
            // The old tab's chart containers are gone (v-if): free the charts.
            stopTeamfightPlay();
            disposeCharts();
            syncRoute(false);
            // The coach tab shows the server's mode and quota: both change while we run.
            if (activeTab.value === 'ai_coach') loadCapabilities();
            await nextTick();
            revealActiveTab();
            renderActiveTab();
        });

        // Vision: filters, slider and the vision toggle redraw the map
        watch([timelineWards, wardShowVision], () => {
            if (activeTab.value === 'wards') scheduleRender(renderWardMap);
        });
        watch(wardTimelineEnabled, (enabled) => {
            if (enabled) wardTimelineTime.value = wardTimeRange.value.max;
        });

        // Smokes
        watch([smokeTimelineTime, smokeTimelineEnabled], () => {
            if (activeTab.value === 'smokes') scheduleRender(renderSmokeMap);
        });
        watch(smokeSelected, () => {
            if (activeTab.value !== 'smokes') return;
            scheduleRender(renderSmokeMap);
            scheduleRender(renderSmokeTimeline);
        });
        watch(smokeTimelineEnabled, (enabled) => {
            if (enabled) smokeTimelineTime.value = smokeTimeRange.value.max;
        });

        watch(economyMode, () => {
            if (activeTab.value === 'economy') renderEconomyChart();
        });

        // Teamfights
        watch(teamfightReplayTime, () => {
            if (activeTab.value === 'teamfights') scheduleRender(renderTeamfightMap);
        });
        watch(selectedTeamfight, () => {
            resetTeamfightReplay();
            if (activeTab.value === 'teamfights') nextTick(renderTeamfightMap);
        });

        // Sub-views that swap their chart container: draw once the new one exists
        watch(farmMode, () => {
            if (activeTab.value === 'farm') nextTick(renderFarmCharts);
        });
        watch(objMode, () => {
            if (activeTab.value === 'objectives') nextTick(renderBuildingMap);
        });
        watch(selectedLaningPlayer, () => {
            if (activeTab.value !== 'laning') return;
            nextTick(() => {
                renderLaningHeatmap();
                renderLaningHpTimeline();
            });
        });

        // Names and icons inside the charts come from the catalogs
        watch(catalogVersion, () => nextTick(renderActiveTab));

        // --- Init ---
        onMounted(() => {
            setInterval(() => {
                clock.value = Date.now();
                retryCapabilities();
            }, 1000);

            loadCapabilities();
            // Names and icons: heroes and items now, abilities with the first match.
            DotaCatalog.load('heroes');
            DotaCatalog.load('items');

            window.addEventListener('popstate', onRouteChange);
            window.addEventListener('dragenter', onWindowDragEnter);
            window.addEventListener('dragleave', onWindowDragLeave);
            window.addEventListener('dragover', onWindowDragOver);
            window.addEventListener('drop', onWindowDrop);

            window.addEventListener('resize', resizeCharts);
            window.addEventListener('beforeprint', onBeforePrint);
            window.addEventListener('afterprint', onAfterPrint);
        });

        // Tab title: the match on screen, or the app name.
        function updateDocumentTitle() {
            const id = analysisData.value && analysisData.value.match_overview && analysisData.value.match_overview.match_id;
            const prefix = id && Number(id) > 0 ? `${t('match.id_label')} ${id} · ` : '';
            document.title = prefix + t('app_title');
        }

        // Language switch: remember it, send it with every request from now
        // on, and redo what is not plain reactive text.
        watch(locale, async () => {
            I18N.store(locale.value);
            DotaApi.setLang(locale.value);
            updateDocumentTitle();
            document.documentElement.lang = locale.value === 'en' ? 'en' : 'zh-CN';
            // The coach report is written by the backend in one language:
            // ask for it again (a cached report answers at once). The one
            // on screen stays until the new one is there.
            // An error whose code has no text here only exists in the
            // backend's words, in the language it was asked in: ask again.
            const foreignError = !!coachError.value && !te('error.' + coachError.value.error.code)
                && coachError.value.error.lang !== locale.value;
            if (coachSlot.value != null && (coachLoading.value || coachReport.value || foreignError)) requestCoach(coachSlot.value);
            // Charts carry translated labels: draw them again.
            await nextTick();
            renderActiveTab();
        });

        // Initial document title and language
        updateDocumentTitle();
        document.documentElement.lang = locale.value === 'en' ? 'en' : 'zh-CN';

        // A reload or a shared link (#/match/<key>/<tab>) opens the match
        // right away; started here so the first paint is the loading state.
        const initialRoute = parseRoute();
        if (initialRoute) openRoute(initialRoute);
        else rejectBadRoute();

        // Toasts on screen, in the current language.
        const toastItems = computed(() => toasts.items.value.map(({ id, kind, content }) => {
            const params = typeof content.params === 'function' ? content.params() : content.params;
            return { id, kind, text: content.error ? errorText(content.error) : t(content.key, params) };
        }));

        return {
            // i18n
            locale, t, tn, codeLabel,
            // shell state, errors, toasts
            analysisData, parseSource, capabilities,
            busy, busyLabel, uploading, homeError, matchError,
            errorText, errorHint, retrySeconds, canRetry, suggestsMatchId, waitSeconds, waitNote,
            showMatchError, retryHomeError, retryMatchError,
            toast, toastItems, closeToast: toasts.close,
            // start page
            fileInput, matchIdField, dragActive, pickFile, focusMatchId, cancelFlow,
            uploadAccept, uploadFormatHint, parserAvailable, statusChips,
            matchIdInput, fetchMatchById, handleFileSelect, loadDemo, resetAnalysis,
            uploadProgress, uploadProgressLabel, uploadIsSlow,
            // match header, scoreboard, draft
            activeTab, tabs, onTabKeydown, overview, teamScores,
            matchResult, gameModeLabel, regionLabel, matchIdText, openDotaUrl,
            durationText, coachAvailable, scoreboardTeams, positionChips, draftView,
            showUnparsedNotice, parseRequest, parseActionBlocked, requestParse, refetchParsed,
            // names, icons, formatting
            heroName, itemName, abilityName, playerName, isAnonymous,
            mainItemSlots, itemPurchaseTime, itemTooltip,
            formatDateTime, formatGameTime, formatNumber, formatNetWorth,
            formatCount, formatK, formatSigned, deltaClass, eventTime, timeAgo, teamSum,
            // shared by the analysis tabs
            teamId, teamClass, teamLabel, slugOf, emptyState,
            killerLabel, killRowClass, killTitle, killAssists,
            toggleSort, sortAria,
            // economy
            hasEconomyTimeline, economyModes, economyMode, economyChart,
            // items, purchases
            hasItemBuilds, itemBuilds, itemsHaveTimeline, itemPhases, itemLabel, hasPurchaseData, purchaseData, purchaseBuckets, purchaseTip,
            // skills
            hasSkillBuilds, skillBuilds, skillsHint, skillRowLabel, skillStepTitle, skillStepTime,
            // vision
            hasWardData, wardFilters, wardStats, wardTeams, wardRadii, filteredWards, timelineWards,
            wardTimelineEnabled, wardTimelineTime, wardTimeRange, wardShowVision, wardMap,
            wardStateOf, wardOwnerName, wardStateDetail, wardStateTitle, wardStateIcon, wardHasDenied, wardLife, wardMaxLife, wardLifePct,
            // smokes
            hasSmokeData, smokeTimeline, smokeTeams, smokeResultColor, smokeResultText, smokeMembers,
            smokeTimelineEnabled, smokeTimelineTime, smokeTimeRange, smokeSelected,
            toggleSmokeSelect, clearSmokeSelection, smokeMapRef, smokeChart,
            // teamfights
            hasTeamfightData, teamfightData, selectedTeamfight, currentTeamfight, tfWindow, teamfightMap,
            teamfightReplayTime, teamfightPlaying, teamfightPlaySpeed, toggleTeamfightPlay, seekTeamfight,
            swingTeam, swingText, itemUses,
            // laning
            hasLaningData, laningData, laningTeams, laningHasDetails, laningDetailsNote, laningEfficiencyNote, laningHasSnapshots,
            selectedLaningPlayer, selectedLaning, laningBreakdown, laneLabel, laneSideLabel, efficiencyText, efficiencyClass,
            laningHeatmap, laningCsTimeline, laningHpTimeline,
            // combat
            hasCombatData, combatModes, combatMode, combatMatrix, combatKills, combatStatColumns, sortedCombatStats, streakLabel,
            // farm
            hasFarmData, farmData, farmModes, farmMode, farmHasReasons, unitKillColumns, unitKillsNote, sortedUnitKills,
            farmReasonChart, farmLhChart,
            // benchmarks
            hasBenchmarkData, benchmarkColumns, sortedBenchmarks, benchmarkValue, benchmarkPct, benchmarkColor,
            // objectives
            hasObjectivesData, objectivesData, objModes, objMode, objectiveRows,
            buildings, buildingTeams, buildingMap, runeColumns, sortedRunes,
            // chat
            chatMessages, chatShowWheel, chatWheelCount, visibleChat, chatText, chatWheelKind, chatPlayerName,
            // AI coach
            coachSlot, coachLoading, coachError, coachCancelled, coachFallback, coachProgress,
            coachTeams, coachHeroName, coachMode, coachStageRows, coachStageLabel, coachLangMismatch,
            coachView, coachSections, coachOpenSources, coachAllSourcesOpen,
            requestCoach, retryCoach, cancelCoach, retryDegradedCoach, coachRefreshWait, coachRefreshNote,
            richText, scrollToCoachSection, setSourcesOpen, toggleAllSources, printCoachReport, closeTip,
            isSourceOpen, sourceEntry, toggleSource, loadSourceEntry,
        };
    },
});

DotaUI.install(app);

// A rendering error must not go unnoticed: log it for the developer.
app.config.errorHandler = (err, instance, info) => {
    console.error('[app] ' + info, err);
};

app.mount('#app');
