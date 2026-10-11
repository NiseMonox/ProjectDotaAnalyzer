// Regression tests for the DOM-free frontend modules (game.js, coach.js, i18n.js).
// Run: node frontend_logic.test.js <path to frontend/static/js>
// No dependencies: the modules attach to `window`, which is this object.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const dir = path.resolve(process.argv[2] || path.join(__dirname, '../../frontend/static/js'));
const window = { location: { hostname: 'test.invalid' }, navigator: { language: 'en' }, localStorage: { getItem: () => null, setItem() {} } };
window.window = window;
const context = vm.createContext(window);
for (const file of ['i18n.js', 'game.js', 'coach.js']) {
    vm.runInContext(fs.readFileSync(path.join(dir, file), 'utf8'), context, { filename: file });
}
const { DotaGame, DotaCoach, I18N } = window;
// Results come from another realm: compare by value, not by prototype.
const plain = (value) => JSON.parse(JSON.stringify(value));
const same = (actual, expected, message) => assert.deepStrictEqual(plain(actual), expected, message);

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

// ---- Vision: ward states ----
test('ward state is the payload state, "denied" included', () => {
    for (const state of ['active', 'dewarded', 'denied', 'destroyed', 'expired']) {
        assert.strictEqual(DotaGame.wardState({ state, time: 0, destroyed_time: 10 }), state);
    }
    same(DotaGame.WARD_STATES.slice().sort(), ['active', 'denied', 'destroyed', 'dewarded', 'expired']);
});

test('a ward without a state is read from its fields', () => {
    const ward = (extra) => Object.assign({ ward_type: 'observer', time: 100 }, extra);
    assert.strictEqual(DotaGame.wardState(ward({ destroyed_time: null })), 'active');
    assert.strictEqual(DotaGame.wardState(ward({ destroyed_time: 130, denied: true })), 'denied');
    assert.strictEqual(DotaGame.wardState(ward({ destroyed_time: 130, dewarded: true })), 'dewarded');
    assert.strictEqual(DotaGame.wardState(ward({ destroyed_time: 130 })), 'destroyed');
    assert.strictEqual(DotaGame.wardState(ward({ destroyed_time: 460 })), 'expired');
    assert.strictEqual(DotaGame.wardState(ward({ state: 'vanished', destroyed_time: 130, denied: true })), 'denied', 'an unknown state code falls back to the fields');
    assert.strictEqual(DotaGame.wardState(null), 'active');
});

test('every ward state has a label and an explanation in both languages', () => {
    for (const state of DotaGame.WARD_STATES) {
        for (const lang of ['zh', 'en']) {
            assert.ok(I18N.has(lang, `code.ward_state.${state}`), `label ${state} ${lang}`);
            assert.ok(I18N.has(lang, `code.ward_state_title.${state}`), `explanation ${state} ${lang}`);
        }
    }
    for (const key of ['wards.dewarded_by', 'wards.dewarded_at', 'wards.denied_by', 'wards.denied_at', 'wards.expired_at', 'wards.destroyed_at', 'wards.standing']) {
        assert.ok(I18N.has('zh', key) && I18N.has('en', key), key);
    }
    assert.strictEqual(I18N.t('en', 'wards.denied_by', { time: '7:11', hero: 'Snapfire' }), 'Denied by Snapfire at 7:11');
    assert.strictEqual(I18N.t('zh', 'wards.denied_by', { time: '7:11', hero: '电炎绝手' }), '7:11 被己方 电炎绝手 反补');
});

// ---- Teamfights: alive intervals ----
test('a hero is alive inside an interval and down between two', () => {
    const intervals = [[0.0, 14.7], [27.0, 45.0]];      // died at 14.7, respawned at 27.0
    same(DotaGame.lifeAt(intervals, 0, 45), { alive: true, since: null });
    same(DotaGame.lifeAt(intervals, 14.6, 45), { alive: true, since: null });
    same(DotaGame.lifeAt(intervals, 14.7, 45), { alive: false, since: 14.7 }, 'the death itself ends the interval');
    same(DotaGame.lifeAt(intervals, 26.9, 45), { alive: false, since: 14.7 });
    same(DotaGame.lifeAt(intervals, 27.0, 45), { alive: true, since: null }, 'alive again at the respawn');
    same(DotaGame.lifeAt(intervals, 44, 45), { alive: true, since: null });
});

test('the end of the fight window is not a death', () => {
    same(DotaGame.lifeAt([[0, 45]], 45, 45), { alive: true, since: null });
    same(DotaGame.lifeAt([[0, 45]], 45.9, 45), { alive: true, since: null }, 'the playback runs past the fight');
    same(DotaGame.lifeAt([[0, 30.4]], 45.9, 45), { alive: false, since: 30.4 }, 'a hero that died stays down past the end');
    same(DotaGame.lifeAt([[0, 0.7], [25, 42]], 43, 42), { alive: true, since: null });
});

test('a hero dead from before the fight, or for all of it', () => {
    same(DotaGame.lifeAt([[14.1, 43.0]], 3, 43), { alive: false, since: null });
    same(DotaGame.lifeAt([[14.1, 43.0]], 14.1, 43), { alive: true, since: null });
    same(DotaGame.lifeAt([], 10, 50), { alive: false, since: null });
    same(DotaGame.lifeAt([[0, 20]], -1, 50), { alive: true, since: null }, 'before the fight = its start');
    same(DotaGame.lifeAt([[5, 20]], -1, 50), { alive: false, since: null });
});

test('two deaths with a buyback in between', () => {
    const intervals = [[10.0, 50.9], [54.0, 63.6]];
    same(DotaGame.lifeAt(intervals, 52, 78), { alive: false, since: 50.9 });
    same(DotaGame.lifeAt(intervals, 54, 78), { alive: true, since: null });
    same(DotaGame.lifeAt(intervals, 70, 78), { alive: false, since: 63.6 }, 'the latest death counts');
});

test('malformed intervals never throw', () => {
    same(DotaGame.lifeAt(null, 5, 40), { alive: false, since: null });
    same(DotaGame.lifeAt([null, [1], ['a', 'b'], [0, 10]], 5, 40), { alive: true, since: null });
    same(DotaGame.lifeAt([[0, 10]], 5, undefined), { alive: true, since: null });
    same(DotaGame.lifeAt([[0, 10]], 12, 0), { alive: false, since: 10 }, 'without a duration nothing stays open');
});

// ---- Chat wheel ----
test('a chat-wheel phrase is read in the UI language', () => {
    const msg = { type: 'chatwheel', key: 'chatwheel_71', phrase: { en: 'Axe is missing!', zh: '斧王不见了！' }, sound: false, spray: false };
    same(DotaGame.chatWheelLine(msg, 'en'), { kind: 'phrase', text: 'Axe is missing!', id: 71, sound: false });
    same(DotaGame.chatWheelLine(msg, 'zh'), { kind: 'phrase', text: '斧王不见了！', id: 71, sound: false });
});

test('a voice line keeps its text and is flagged', () => {
    const msg = { key: 'chatwheel_47001', phrase: { en: 'Tehehehehar.', zh: '呵呵呵呵呵。' }, sound: true, spray: false };
    same(DotaGame.chatWheelLine(msg, 'zh'), { kind: 'phrase', text: '呵呵呵呵呵。', id: 47001, sound: true });
});

test('the other language stands in for a missing one', () => {
    same(DotaGame.chatWheelLine({ key: 'chatwheel_9', phrase: { en: 'Missing top!', zh: '' } }, 'zh').text, 'Missing top!');
    same(DotaGame.chatWheelLine({ key: 'chatwheel_9', phrase: { zh: '上路不见了！' } }, 'en').text, '上路不见了！');
});

test('sprays and unknown ids have no text: the caller shows a label or "Chat wheel #id"', () => {
    same(DotaGame.chatWheelLine({ key: 'chatwheel_120001', phrase: null, sound: false, spray: true }, 'en'), { kind: 'spray', text: '', id: 120001, sound: false });
    same(DotaGame.chatWheelLine({ key: 'chatwheel_999999', phrase: null, sound: false, spray: false }, 'en'), { kind: 'unknown', text: '', id: 999999, sound: false });
    same(DotaGame.chatWheelLine({ key: 'chatwheel_5' }, 'zh'), { kind: 'unknown', text: '', id: 5, sound: false }, 'a payload without the new fields');
    same(DotaGame.chatWheelLine({ key: 'hello', phrase: 'not an object' }, 'en'), { kind: 'unknown', text: '', id: null, sound: false });
    same(DotaGame.chatWheelLine(null, 'en'), { kind: 'unknown', text: '', id: null, sound: false });
    for (const key of ['chat.wheel', 'chat.wheel_n', 'chat.tag.phrase', 'chat.tag.voice', 'chat.tag.spray', 'chat.title.phrase', 'chat.title.voice', 'chat.title.spray', 'chat.title.unknown']) {
        assert.ok(I18N.has('zh', key) && I18N.has('en', key), key);
    }
    assert.strictEqual(I18N.t('en', 'chat.wheel_n', { n: 71 }), 'Chat wheel #71');
});

// ---- Coach report ----
const report = (extra) => Object.assign({ overall_grade: 'B', death_analyses: [], key_moments: [] }, extra);

test('the language of a report is its own lang field', () => {
    assert.strictEqual(DotaCoach.reportLang(report({ lang: 'zh' })), 'zh');
    assert.strictEqual(DotaCoach.reportLang(report({ lang: 'en' })), 'en');
    assert.strictEqual(DotaCoach.reportLang(report({})), '');
    assert.strictEqual(DotaCoach.reportLang(report({ lang: 'fr' })), '');
    assert.strictEqual(DotaCoach.reportLang(null), '');
});

test('"generate again" is offered for a degraded report only, with the time left', () => {
    same(DotaCoach.refreshState(report({ degraded: true, refresh: { available: true, retry_after: 0 } })), { available: true, wait: 0 });
    same(DotaCoach.refreshState(report({ degraded: true, refresh: { available: true, retry_after: 41.2 } })), { available: true, wait: 42 });
    same(DotaCoach.refreshState(report({ degraded: true, refresh: { available: true, retry_after: null } })), { available: true, wait: 0 });
    same(DotaCoach.refreshState(report({ degraded: false, refresh: { available: false, retry_after: null } })), { available: false, wait: 0 });
    same(DotaCoach.refreshState(report({ degraded: true })), { available: false, wait: 0 }, 'a server that does not offer it');
    same(DotaCoach.refreshState(report({ degraded: false, refresh: { available: true, retry_after: 5 } })), { available: false, wait: 0 }, 'never for a complete report');
    same(DotaCoach.refreshState(report({ degraded: true, refresh: { available: 'yes', retry_after: 5 } })), { available: false, wait: 0 });
    same(DotaCoach.refreshState(null), { available: false, wait: 0 });
    for (const key of ['coach.degraded.title', 'coach.degraded.text', 'coach.degraded.hint', 'coach.degraded.wait', 'coach.degraded.retry', 'coach.degraded.still', 'coach.degraded.still_now']) {
        assert.ok(I18N.has('zh', key) && I18N.has('en', key), key);
    }
});

// ---- Positions ----
test('positions are 1号位…5号位 / Pos 1…5', () => {
    for (let n = 1; n <= 5; n++) {
        assert.strictEqual(I18N.code('zh', 'position_short', n), `${n}号位`);
        assert.strictEqual(I18N.code('en', 'position_short', n), `Pos ${n}`);
        assert.ok(I18N.code('zh', 'position', n).startsWith(`${n}号位`));
    }
    for (const code of ['high', 'medium', 'low']) assert.ok(I18N.has('zh', `code.confidence.${code}`) && I18N.has('en', `code.confidence.${code}`));
    for (const code of ['lane_nw', 'archetype_nw', 'nw_only']) assert.ok(I18N.has('zh', `code.position_method.${code}`) && I18N.has('en', `code.position_method.${code}`));
});

test('position labels name the number: never a bare 辅助, never 野区 / Jungle', () => {
    for (const domain of ['position', 'position_short']) {
        for (const [code, pair] of Object.entries(window.I18N_CODES[domain])) {
            assert.ok(/^(\d|\{n\})号位/.test(pair[0]), `${domain}.${code}: ${pair[0]}`);
            assert.ok(!/辅助|野/.test(pair[0]), pair[0]);
            assert.ok(/^(Position|Pos) (\d|\{n\})/.test(pair[1]) && !/jungle/i.test(pair[1]), pair[1]);
        }
    }
});

test('every string exists in both languages', () => {
    for (const [key, pair] of Object.entries(window.I18N_PAIRS)) {
        assert.ok(Array.isArray(pair) && pair.length === 2 && pair.every((text) => typeof text === 'string' && text), key);
    }
    for (const [domain, table] of Object.entries(window.I18N_CODES)) {
        for (const [code, pair] of Object.entries(table)) {
            assert.ok(Array.isArray(pair) && pair.length === 2 && pair.every((text) => typeof text === 'string' && text), `${domain}.${code}`);
        }
    }
});

let failed = 0;
for (const [name, fn] of tests) {
    try { fn(); console.log('ok   ' + name); } catch (error) { failed += 1; console.log('FAIL ' + name + '\n     ' + String(error && error.message).split('\n').join('\n     ')); }
}
console.log(`${tests.length - failed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
