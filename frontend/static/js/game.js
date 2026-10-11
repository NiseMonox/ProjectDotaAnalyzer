/**
 * Dota 2 Match Analyzer - game rules and payload decoding (window.DotaGame)
 *
 * Pure functions shared by the analysis tabs: no DOM, no Vue, no texts.
 * app.js turns the codes returned here into localized labels.
 *
 *   Map          toMap(x, y), mapLength(distance), onMap(x, y), isJump(a, b, seconds)
 *   Patch rules  patchAtLeast(patch, major, minor), wardVision(type, patch),
 *                wardLifetime(type), wardState(ward)
 *   Teamfights   lifeAt(alive_intervals, seconds, duration)
 *   Units        parseBuilding(name), classifyUnit(name)
 *   Skills       skillGrid(skill_order)
 *   Chat         chatWheelId(key), chatWheelLine(message, lang)
 */
window.DotaGame = (function () {
    'use strict';

    // ---- Map geometry ----
    // Same convention as backend/app/geo.py: the payload is in world units
    // for every source; the map image spans -8192 .. 8064 on both axes and
    // the origin is bottom-left (y grows upward, like a chart axis).
    const WORLD_MIN = -8192;
    const WORLD_SPAN = 16256;

    // Markers keep this far (in map units) from the edge of the picture: a
    // smoke used inside the fountain would otherwise be drawn half off the map.
    const EDGE_INSET = 1.5;

    function pct(v) {
        return Math.min(100 - EDGE_INSET, Math.max(EDGE_INSET, (v - WORLD_MIN) / WORLD_SPAN * 100));
    }

    /** World (x, y) -> [0..100, 0..100] on the map; heroes can stand a little outside, so clamped. */
    function toMap(x, y) {
        return [pct(Number(x)), pct(Number(y))];
    }

    // Position samples of a fight come every two seconds while a hero lives.
    const SAMPLE_GAP_S = 5;

    /**
     * Whether a hero got from sample a to sample b ({x, y}, world units),
     * `seconds` apart, by something other than walking: a Town Portal
     * Scroll (too far for the time), or a death and respawn (a gap in the
     * samples: nothing is known in between). A Blink Dagger (1200 units)
     * still counts as moving.
     */
    function isJump(a, b, seconds) {
        const time = Math.max(0, Number(seconds) || 0);
        const distance = Math.hypot(Number(b.x) - Number(a.x), Number(b.y) - Number(a.y));
        return time > SAMPLE_GAP_S || distance > 1500 + 600 * time;
    }

    /** A world distance (vision radius) in map units. */
    function mapLength(distance) {
        return distance / WORLD_SPAN * 100;
    }

    /** false for a missing position (null / NaN): do not draw it at the origin. */
    function onMap(x, y) {
        return x != null && y != null && Number.isFinite(Number(x)) && Number.isFinite(Number(y));
    }

    // ---- Patch-dependent rules ----

    /** '7.41f' -> [7, 41]; null for '' / unknown. */
    function patchParts(patch) {
        const m = /^(\d+)\.(\d+)/.exec(String(patch || ''));
        return m ? [Number(m[1]), Number(m[2])] : null;
    }

    /** Whether the match patch is at least major.minor; an unknown patch counts as current. */
    function patchAtLeast(patch, major, minor) {
        const parts = patchParts(patch);
        if (!parts) return true;
        return parts[0] > major || (parts[0] === major && parts[1] >= minor);
    }

    // Observer: 1600 vision, 6 minutes. Sentry: true sight 1050 since 7.38
    // (1000 before), 7 minutes.
    const WARD_LIFETIME = { observer: 360, sentry: 420 };

    /** Vision (observer) or true-sight (sentry) radius in world units. */
    function wardVision(type, patch) {
        if (type === 'observer') return 1600;
        return patchAtLeast(patch, 7, 38) ? 1050 : 1000;
    }

    /** Full lifetime of a ward type in seconds. */
    function wardLifetime(type) {
        return WARD_LIFETIME[type] || WARD_LIFETIME.observer;
    }

    // In the order the Vision tab lists them.
    const WARD_STATES = ['dewarded', 'denied', 'destroyed', 'expired', 'active'];
    // A ward that fell this many seconds before its full lifetime did not expire.
    const WARD_EARLY_S = 5;

    /**
     * How a ward ended (the payload's `state`): 'dewarded' (the enemy removed
     * it), 'denied' (a hero of its own team destroyed it), 'expired' (it ran
     * its full time), 'destroyed' (it fell clearly before its time and the
     * data names nobody) or 'active' (still standing when the data ends). A
     * payload without `state` is read the same way from its other fields.
     */
    function wardState(ward) {
        if (!ward) return 'active';
        if (WARD_STATES.includes(ward.state)) return ward.state;
        if (ward.destroyed_time == null) return 'active';
        if (ward.denied) return 'denied';
        if (ward.dewarded) return 'dewarded';
        const life = ward.lifetime != null ? ward.lifetime : ward.destroyed_time - ward.time;
        return life < wardLifetime(ward.ward_type) - WARD_EARLY_S ? 'destroyed' : 'expired';
    }

    // ---- Teamfights ----

    /**
     * Whether a hero is alive `seconds` into a fight, from the payload's
     * alive_intervals: [[from, to], ...] in seconds from the fight start,
     * inside 0..duration ([] = dead for the whole fight). A death ends an
     * interval; a respawn or a buyback starts the next one; a hero still
     * dead from before the fight starts late.
     *
     * Returns { alive, since }: for a dead hero `since` is the moment it
     * fell, or null when it has been dead since before the fight. The
     * intervals cover 0..duration only: a moment before the fight reads as
     * its start, and an interval that reaches `duration` stays open
     * (nothing in them says the hero died there). A malformed pair is
     * skipped.
     */
    function lifeAt(intervals, seconds, duration) {
        const end = Number(duration);
        const closes = Number.isFinite(end) && end > 0 ? end : Infinity;
        const now = Math.max(0, Number(seconds) || 0);
        let since = null;
        for (const pair of Array.isArray(intervals) ? intervals : []) {
            if (!Array.isArray(pair)) continue;
            const from = Number(pair[0]);
            const to = Number(pair[1]);
            if (!Number.isFinite(from) || !Number.isFinite(to) || from > now) continue;
            if (now < to || to >= closes) return { alive: true, since: null };
            if (since == null || to > since) since = to;
        }
        return { alive: false, since };
    }

    // ---- Units ----
    const SIDES = { goodguys: 'radiant', badguys: 'dire' };
    const BUILDING_RE = /^npc_dota_(goodguys|badguys)_(?:tower([1-4])(?:_(top|mid|bot))?|(melee|range)_rax_(top|mid|bot)|(fort))$/;

    /**
     * 'npc_dota_badguys_tower1_mid' -> { side: 'dire', kind: 'tower', tier: 1, lane: 'mid' }.
     * kind: tower | barracks_melee | barracks_ranged | ancient. lane is ''
     * for the Ancient and for tier-4 towers. null when it is no such name.
     */
    function parseBuilding(name) {
        const m = BUILDING_RE.exec(String(name || ''));
        if (!m) return null;
        const side = SIDES[m[1]];
        if (m[6]) return { side, kind: 'ancient', tier: 0, lane: '' };
        if (m[4]) return { side, kind: m[4] === 'melee' ? 'barracks_melee' : 'barracks_ranged', tier: 0, lane: m[5] };
        const tier = Number(m[2]);
        return { side, kind: 'tower', tier, lane: tier === 4 ? '' : (m[3] || '') };
    }

    /**
     * What a non-hero unit name is: { kind, side, building?, slug? } with
     * kind: building | lane_creep | neutral | roshan | tormentor | fountain |
     * courier | hero | unknown. side is 'radiant' / 'dire' / '' (not in the name).
     */
    function classifyUnit(name) {
        const unit = String(name || '');
        if (!unit) return null;
        const building = parseBuilding(unit);
        if (building) return { kind: 'building', side: building.side, building };
        const sideMatch = /(goodguys|badguys)/.exec(unit);
        const side = sideMatch ? SIDES[sideMatch[1]] : '';
        if (unit.includes('fountain')) return { kind: 'fountain', side };
        if (unit.startsWith('npc_dota_hero_')) return { kind: 'hero', side: '', slug: unit.slice(14) };
        if (unit.startsWith('npc_dota_roshan')) return { kind: 'roshan', side: '' };
        if (unit.startsWith('npc_dota_miniboss')) return { kind: 'tormentor', side: '' };
        if (unit.startsWith('npc_dota_neutral_')) return { kind: 'neutral', side: '' };
        if (unit.includes('courier')) return { kind: 'courier', side };
        if (/^npc_dota_creep_(goodguys|badguys)/.test(unit) || /^npc_dota_(goodguys|badguys)_siege/.test(unit)) {
            return { kind: 'lane_creep', side };
        }
        if (/^npc_dota_(goodguys|badguys)_/.test(unit)) return { kind: 'building', side, building: null };
        return { kind: 'unknown', side };
    }

    // ---- Skill builds ----
    const ATTRIBUTES_KEY = 'special_bonus_attributes';

    function isTalentEntry(skill) {
        const key = skill.ability_key || '';
        if (key === ATTRIBUTES_KEY) return false;
        return !!skill.is_talent || key.startsWith('special_bonus_') || key.startsWith('talent_');
    }

    /**
     * skill_order[] -> the classic build table.
     *
     *   steps  one per entry, in order: { n (1-based), row, kind, key, name, level, time, time_str }
     *   rows   one per ability, then the attribute bonus, then the talents:
     *          { id, kind: basic | ultimate | attribute | talent, key, name, cells[] (step or null), points }
     *
     * The ultimate is the ability the payload flags (is_ultimate true: the
     * game data's word). Where the game data does not know the abilities
     * (is_ultimate null or absent) it is the earliest one that looks like an
     * ultimate: at most three levels, first point with the fifth skill point
     * or later, and its points spread out the way an ultimate's are (it
     * opens at hero levels 6 / 12 / 18; players bank points, mostly in
     * Turbo, so the limits are loose). An ability flagged false is never
     * taken for one, so heroes like Invoker get no ultimate row.
     * options.talentsCostPoint: true for matches before 7.40, where a talent
     * used up the skill point of its level.
     */
    function skillGrid(order, options) {
        const list = Array.isArray(order) ? order : [];
        const talentsCostPoint = !!(options && options.talentsCostPoint);
        const byId = new Map();
        const steps = [];
        let points = 0;   // skill points spent so far = hero level reached
        list.forEach((skill, index) => {
            const key = skill.ability_key || '';
            const talent = isTalentEntry(skill);
            const kind = talent ? 'talent' : key === ATTRIBUTES_KEY ? 'attribute' : 'basic';
            if (!talent || talentsCostPoint) points += 1;
            const id = kind === 'basic' ? 'a:' + (key || skill.ability || index) : kind;
            let row = byId.get(id);
            if (!row) {
                row = { id, kind, key: kind === 'talent' ? '' : key, name: kind === 'talent' ? '' : (skill.ability || ''), at: [], flagged: false, basic: false, points: 0, maxLevel: 0 };
                byId.set(id, row);
            }
            row.points += 1;
            row.at.push(points);
            row.maxLevel = Math.max(row.maxLevel, Number(skill.level) || row.points);
            if (skill.is_ultimate === true) row.flagged = true;
            if (skill.is_ultimate === false) row.basic = true;
            steps.push({
                n: index + 1, row: id, kind, key, name: skill.ability || '',
                level: Number(skill.level) || row.points,
                time: skill.time, time_str: skill.time_str || '',
            });
        });

        const abilities = [...byId.values()].filter(row => row.kind === 'basic');
        const ultimateLike = (row) => row.points <= 3 && row.maxLevel <= 3 && row.at[0] >= 5
            && (row.at.length < 2 || row.at[1] - row.at[0] >= 4)
            && (row.at.length < 3 || row.at[2] - row.at[1] >= 3);
        const ultimate = abilities.find(row => row.flagged) || abilities.find(row => !row.basic && ultimateLike(row));
        if (ultimate) ultimate.kind = 'ultimate';

        const rank = { basic: 0, ultimate: 1, attribute: 2, talent: 3 };
        const rows = [...byId.values()].sort((a, b) => rank[a.kind] - rank[b.kind]);
        rows.forEach(row => {
            row.cells = steps.map(step => (step.row === row.id ? step : null));
        });
        steps.forEach(step => {
            if (step.kind === 'basic' && ultimate && step.row === ultimate.id) step.kind = 'ultimate';
        });
        return { steps, rows };
    }

    // ---- Chat ----

    /** 'chatwheel_71' -> 71; null when the key is not a chat wheel id. */
    function chatWheelId(key) {
        const m = /^(?:chatwheel_)?(\d+)$/.exec(String(key == null ? '' : key));
        return m ? Number(m[1]) : null;
    }

    /**
     * What a chat-wheel message (type 'chatwheel') says: { kind, text, id, sound }.
     *   kind 'phrase'   text is the line in `lang` ('zh' | 'en'); the payload's
     *                   `phrase` is {en, zh}, and the other language is used
     *                   where one is empty
     *   kind 'spray'    a spray: a picture, no text
     *   kind 'unknown'  the payload has no phrase for this id (text '')
     * sound: the line is a voice line, heard in the game. id is the chat-wheel
     * number (null when the key carries none).
     */
    function chatWheelLine(msg, lang) {
        const m = msg || {};
        const id = chatWheelId(m.key);
        const phrase = m.phrase && typeof m.phrase === 'object' ? m.phrase : {};
        const own = lang === 'en' ? phrase.en : phrase.zh;
        const other = lang === 'en' ? phrase.zh : phrase.en;
        const text = String(own || other || '').trim();
        if (text) return { kind: 'phrase', text, id, sound: !!m.sound };
        return { kind: m.spray ? 'spray' : 'unknown', text: '', id, sound: !!m.sound };
    }

    return {
        WORLD_MIN, WORLD_SPAN, toMap, mapLength, onMap, isJump,
        patchParts, patchAtLeast, wardVision, wardLifetime, wardState, WARD_STATES,
        lifeAt,
        parseBuilding, classifyUnit,
        ATTRIBUTES_KEY, isTalentEntry, skillGrid,
        chatWheelId, chatWheelLine,
    };
})();
