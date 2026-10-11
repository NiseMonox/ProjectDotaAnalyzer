/**
 * Dota 2 Match Analyzer - AI coach helpers (window.DotaCoach)
 *
 * Pure functions for the coach tab: no DOM, no strings for the user (labels
 * of every code below come from i18n.js through codeLabel in app.js).
 *
 *   newProgress() / advance(progress, event) / stageRows(progress)
 *       progress of a report from the stream's {type:'progress', stage,
 *       current, total} events; the percentage never goes backwards
 *   isReport(data)               shape check of a report (schema version 5)
 *   reportLang(report)           language the report is written in
 *   refreshState(report)         whether a degraded report can be generated again, and when
 *   findKill(kills, slot, time)  kill-feed entry of a death in the report
 *   sourceSummary(report)        knowledge sources of the whole report
 *   tone maps                    code -> colour class suffix
 */
window.DotaCoach = (function () {
    'use strict';

    // ---- Progress ----
    // Stages in the order the backend runs them. `evaluate` and `phases`
    // run side by side when an LLM writes the report: their events
    // interleave, so they share one step of the sequence (same `group`).
    const STAGES = [
        { id: 'starting', group: 0, weight: 2 },
        { id: 'detect', group: 1, weight: 3 },
        { id: 'causal_chain', group: 2, weight: 3 },
        { id: 'context', group: 3, weight: 4 },
        { id: 'evaluate', group: 4, weight: 60 },   // one LLM answer per decision: most of the wait
        { id: 'phases', group: 4, weight: 24 },
        { id: 'summary', group: 5, weight: 3 },
        { id: 'done', group: 6, weight: 1 },
    ];
    const STAGE_BY_ID = {};
    STAGES.forEach((stage) => { STAGE_BY_ID[stage.id] = stage; });

    function newProgress() {
        // steps[stage] = { current, total } of the latest event of that stage
        return { steps: {}, group: -1, percent: 0 };
    }

    function stageState(progress, stage) {
        const step = progress.steps[stage.id];
        if (progress.group > stage.group || (stage.id === 'done' && step)) return 'done';
        return step ? 'active' : 'pending';
    }

    /** New progress after one event. Unknown stages and repeats are harmless. */
    function advance(progress, event) {
        const stage = STAGE_BY_ID[event && event.stage];
        if (!stage) return progress;
        const total = Math.max(1, Math.floor(Number(event.total)) || 1);
        const current = Math.min(total, Math.max(0, Math.floor(Number(event.current)) || 0));
        const before = progress.steps[stage.id];
        const steps = Object.assign({}, progress.steps);
        steps[stage.id] = before && before.total === total
            ? { current: Math.max(before.current, current), total }
            : { current, total };
        const next = { steps, group: Math.max(progress.group, stage.group), percent: 0 };
        let sum = 0;
        STAGES.forEach((s) => {
            const state = stageState(next, s);
            const step = steps[s.id];
            if (state === 'done') sum += s.weight;
            else if (state === 'active') sum += s.weight * (step.current / step.total);
        });
        // 100 is set when the report itself has arrived.
        next.percent = Math.max(progress.percent, Math.min(99, Math.round(sum)));
        return next;
    }

    /** The stages worth showing, with their state and count. */
    function stageRows(progress) {
        return STAGES.filter((s) => s.id !== 'starting' && s.id !== 'done').map((s) => {
            const step = progress.steps[s.id];
            return {
                id: s.id,
                state: stageState(progress, s),
                current: step ? step.current : 0,
                total: step ? step.total : 0,
            };
        });
    }

    /** Stage named in the status line: the first one still running. */
    function currentStage(progress) {
        const active = STAGES.find((s) => stageState(progress, s) === 'active');
        return active ? active.id : (progress.group < 0 ? 'starting' : 'summary');
    }

    // ---- Report ----
    function isObject(value) {
        return value !== null && typeof value === 'object' && !Array.isArray(value);
    }

    function isReport(data) {
        return isObject(data) && typeof data.overall_grade === 'string'
            && Array.isArray(data.death_analyses) && Array.isArray(data.key_moments);
    }

    function list(value) {
        return Array.isArray(value) ? value : [];
    }

    /** 'zh' | 'en': the report's own `lang`; '' when it does not say. */
    function reportLang(report) {
        const lang = report && report.lang;
        return lang === 'zh' || lang === 'en' ? lang : '';
    }

    /**
     * "Generate again" of a report: { available, wait }. The server offers
     * it for a degraded report only (refresh: {available, retry_after}) and
     * starts the new run when asked with ?refresh=1; `wait` is the number of
     * seconds until it will (0 = now). Asking earlier returns the same
     * report with the time that is left.
     */
    function refreshState(report) {
        const refresh = report && isObject(report.refresh) ? report.refresh : null;
        if (!refresh || refresh.available !== true || !report.degraded) return { available: false, wait: 0 };
        return { available: true, wait: Math.max(0, Math.ceil(Number(refresh.retry_after) || 0)) };
    }

    /**
     * The kill-feed entry (combat.kills[]) of a death analysed in the report.
     * The report has whole seconds (truncated), the feed has fractions.
     */
    function findKill(kills, slot, time) {
        if (time == null || !Number.isFinite(Number(time))) return null;
        let best = null;
        list(kills).forEach((kill) => {
            if (!kill || kill.victim_slot !== slot || kill.counted === false) return;
            const gap = Math.abs(Number(kill.time) - Number(time));
            if (gap <= 1 && (!best || gap < best.gap)) best = { gap, kill };
        });
        return best ? best.kill : null;
    }

    // Fixed order of the source types (most trusted first).
    const SOURCE_TYPES = ['builtin', 'editorial', 'operator', 'opendota_stats', 'match_mining', 'llm_distilled', 'video'];

    function sourceRank(type) {
        const index = SOURCE_TYPES.indexOf(type);
        return index < 0 ? SOURCE_TYPES.length : index;
    }

    /**
     * Knowledge sources behind the whole report: distinct entries (by id)
     * over all evaluations, counted by type. An entry counts as cited when
     * any evaluation cited it.
     */
    function sourceSummary(report) {
        const byId = new Map();
        let evaluations = 0;
        const items = list(report && report.death_analyses).concat(list(report && report.key_moments));
        items.forEach((item) => {
            const sources = list(item && item.sources);
            if (sources.length) evaluations += 1;
            sources.forEach((source) => {
                if (!isObject(source)) return;
                const id = String(source.id || source.title || '');
                const seen = byId.get(id);
                if (seen) seen.cited = seen.cited || !!source.cited;
                else byId.set(id, { type: String(source.source_type || ''), cited: !!source.cited });
            });
        });
        const counts = new Map();
        let cited = 0;
        byId.forEach((entry) => {
            const row = counts.get(entry.type) || { type: entry.type, count: 0, cited: 0 };
            row.count += 1;
            if (entry.cited) { row.cited += 1; cited += 1; }
            counts.set(entry.type, row);
        });
        return {
            total: byId.size,
            cited,
            evaluations,
            types: Array.from(counts.values()).sort((a, b) => sourceRank(a.type) - sourceRank(b.type)),
        };
    }

    // ---- Colours: code -> class suffix (see "AI coach" in app.css) ----
    const GRADES = ['S', 'A', 'B', 'C', 'D'];
    const RATING_TONE = { excellent: 'great', good: 'good', average: 'mid', below_average: 'low', poor: 'bad' };
    const SEVERITY_TONE = { critical: 'bad', major: 'low', minor: 'mid', positive: 'good' };
    const POINT_TONE = { death: 'bad', bad_trade: 'low', missed_cs: 'mid', overextend: 'warn', fight_engage: 'info', good_play: 'good' };
    const LANE_TONE = { won: 'good', even: 'mid', lost: 'bad' };
    const CONFIDENCE_TONE = { high: 'good', medium: 'mid', low: 'low' };

    /** 's' | 'a' | 'b' | 'c' | 'd' | 'none' for the grade classes. */
    function gradeTone(score) {
        const grade = String(score || '').toUpperCase();
        return GRADES.includes(grade) ? grade.toLowerCase() : 'none';
    }

    function tone(table, code) {
        return table[code] || 'none';
    }

    return {
        newProgress, advance, stageRows, currentStage,
        isReport, reportLang, refreshState, findKill, sourceSummary, SOURCE_TYPES,
        gradeTone,
        ratingTone: (code) => tone(RATING_TONE, code),
        severityTone: (code) => tone(SEVERITY_TONE, code),
        pointTone: (code) => tone(POINT_TONE, code),
        laneTone: (code) => tone(LANE_TONE, code),
        confidenceTone: (code) => tone(CONFIDENCE_TONE, code),
    };
})();
