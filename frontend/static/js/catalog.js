/**
 * Dota 2 Match Analyzer - static game catalogs (window.DotaCatalog)
 *
 * Loads /static/data/{heroes,items,abilities}.json (generated from the game
 * data, see their `meta`) and answers name / icon lookups for every tab:
 *
 *   heroes.json     { by_id: {id: {id, name, name_zh?, slug, img, icon, ...}}, name_to_id, meta }
 *   items.json      { by_key: {key: {id, key, name, name_zh?, img, cost, tier?, qual?, obsolete?, legacy?}}, meta }
 *   abilities.json  { by_key: {key: {key, name, name_zh?, img (null: no picture), id?, ids?, legacy?}}, meta }
 *
 * `name` is the English name (what every match payload carries, and what
 * lookups by name match on); `name_zh` the game's Simplified Chinese one,
 * where Valve has it. The name helpers take the UI language ('zh' | 'en')
 * and fall back to `name`.
 *
 * Nothing here is reactive: app.js wraps these lookups (it re-renders on
 * onChange and adds the i18n fallbacks for unknown keys). Icon URL helpers
 * return '' when there is no picture to request, so the caller shows a
 * fallback tile instead of a broken image.
 */
window.DotaCatalog = (function () {
    'use strict';

    const CDN = 'https://cdn.akamai.steamstatic.com/apps/dota2/images/dota_react';
    // ?v= is bumped when the shape of the catalogs changes (2: name_zh), so a
    // browser does not keep reading a cached copy without the new field.
    const FILES = { heroes: '/static/data/heroes.json?v=2', items: '/static/data/items.json?v=2', abilities: '/static/data/abilities.json?v=2' };

    const data = { heroes: null, items: null, abilities: null };
    const index = { heroBySlug: {}, itemById: {} };
    const pending = {};
    const listeners = [];

    function onChange(fn) {
        listeners.push(fn);
    }

    function buildIndex(name) {
        if (name === 'heroes') {
            index.heroBySlug = {};
            Object.values(data.heroes.by_id || {}).forEach((hero) => { index.heroBySlug[hero.slug] = hero; });
        } else if (name === 'items') {
            index.itemById = {};
            Object.values(data.items.by_key || {}).forEach((item) => {
                if (item.id != null && !(item.id in index.itemById)) index.itemById[item.id] = item;
            });
        }
    }

    /**
     * Load one catalog once (concurrent callers share the request). Resolves
     * to true/false and never rejects; a failed load can be retried later.
     */
    function load(name) {
        if (data[name]) return Promise.resolve(true);
        if (pending[name]) return pending[name];
        pending[name] = fetch(FILES[name])
            .then((res) => (res.ok ? res.json() : Promise.reject(new Error('HTTP ' + res.status))))
            .then((json) => {
                data[name] = json;
                buildIndex(name);
                listeners.forEach((fn) => fn(name));
                return true;
            })
            .catch((err) => {
                console.warn(`[catalog] ${name} not loaded: ${err.message}`);
                return false;
            })
            .finally(() => { delete pending[name]; });
        return pending[name];
    }

    function loaded(name) {
        return !!data[name];
    }

    function meta(name) {
        return (data[name] && data[name].meta) || null;
    }

    /** Display name of a catalog entry in the UI language ('' without an entry). */
    function localName(entry, locale) {
        if (!entry) return '';
        return (locale === 'zh' && entry.name_zh) || entry.name || '';
    }

    // ---- Heroes ----

    function heroBySlug(slug) {
        return (slug && index.heroBySlug[slug]) || null;
    }

    function heroById(id) {
        return (data.heroes && data.heroes.by_id[String(id)]) || null;
    }

    /** By display name; both spellings of renamed heroes resolve (name_to_id). */
    function heroByName(name) {
        if (!data.heroes || !name) return null;
        const id = data.heroes.name_to_id[name];
        return id ? heroById(id) : null;
    }

    function heroSlugByName(name) {
        const hero = heroByName(name);
        return hero ? hero.slug : '';
    }

    // '' for an empty slug and for one the loaded catalog does not know.
    function hasHeroPicture(slug) {
        return !!slug && (!data.heroes || !!index.heroBySlug[slug]);
    }

    /** Small square icon. '' when there is no picture to request. */
    function heroIconUrl(slug) {
        return hasHeroPicture(slug) ? `${CDN}/heroes/icons/${slug}.png` : '';
    }

    /** Wide portrait. '' when there is no picture to request. */
    function heroImgUrl(slug) {
        return hasHeroPicture(slug) ? `${CDN}/heroes/${slug}.png` : '';
    }

    // ---- Items ----

    function item(key) {
        return (key && data.items && data.items.by_key[key]) || null;
    }

    function itemById(id) {
        return index.itemById[id] || null;
    }

    /** Catalog entry for a payload key, including the 'item_<id>' form. */
    function resolveItem(key) {
        const direct = item(key);
        if (direct || !key) return direct;
        const byId = /^item_(\d+)$/.exec(key);
        return byId ? itemById(Number(byId[1])) : null;
    }

    function itemName(key, locale) {
        return localName(resolveItem(key), locale);
    }

    /**
     * Icon of an item key. Recipes share one picture. '' for an empty key and
     * for a key the loaded catalog does not know (no picture exists for it).
     */
    function itemIconUrl(key) {
        if (!key) return '';
        const entry = resolveItem(key);
        if (entry) return entry.img || `${CDN}/items/${entry.key}.png`;
        if (key.startsWith('recipe_')) return `${CDN}/items/recipe.png`;
        return data.items ? '' : `${CDN}/items/${key}.png`;
    }

    // ---- Abilities and talents ----

    function ability(key) {
        return (key && data.abilities && data.abilities.by_key[key]) || null;
    }

    /** Talents are special_bonus_* keys (legacy data: talent_*); the attribute bonus is not one. */
    function isTalentKey(key) {
        if (!key || key === 'special_bonus_attributes') return false;
        return key.startsWith('special_bonus_') || key.startsWith('talent_');
    }

    function abilityName(key, locale) {
        return localName(ability(key), locale);
    }

    const NAME_NUMBER = /\d+(?:\.\d+)?/g;

    /**
     * Chinese name of a talent as the payload names it. Talent keys are
     * reused between patches with other numbers, so `given` (the English
     * name the backend resolved for the match's patch) decides the numbers:
     * "+30 Counter Helix Damage" with the catalog's "+40 Counter Helix
     * Damage" / "+40 反击螺旋伤害" gives "+30 反击螺旋伤害". '' when the
     * catalog has no Chinese name or it is another talent by now (the
     * caller then shows `given`).
     */
    function talentNameZh(key, given) {
        const entry = ability(key);
        if (!entry || !entry.name_zh) return '';
        if (!given || given === entry.name) return entry.name_zh;
        const shape = (text) => text.replace(NAME_NUMBER, '#');
        if (shape(given) !== shape(entry.name)) return '';
        const theirs = given.match(NAME_NUMBER) || [];
        const ours = entry.name.match(NAME_NUMBER) || [];
        // The Chinese wording can state the numbers in another order: match them by value.
        const left = ours.map((value, index) => ({ value, to: theirs[index] }));
        let complete = true;
        const text = entry.name_zh.replace(NAME_NUMBER, (number) => {
            const at = left.findIndex((pair) => pair.value === number);
            if (at < 0) { complete = false; return number; }
            return left.splice(at, 1)[0].to;
        });
        return complete && !left.length ? text : '';
    }

    /** Icon of an ability key. '' for talents, the attribute bonus, abilities without a picture and unknown keys. */
    function abilityIconUrl(key) {
        if (!key || key.startsWith('special_bonus_') || key.startsWith('talent_')) return '';
        const entry = ability(key);
        if (entry) return entry.img || '';
        return data.abilities ? '' : `${CDN}/abilities/${key}.png`;
    }

    // ---- Fallback tile ----

    const TILE = {
        hero: [64, 64, 26],
        'hero-portrait': [128, 72, 30],
        item: [88, 64, 26],
        ability: [64, 64, 26],
    };

    /** Up to two letters for a tile: 'Consecrated Wraps' -> 'CW', '' -> '?'. */
    function initials(label) {
        const words = String(label || '').replace(/[^\p{L}\p{N} ]+/gu, ' ').trim().split(/\s+/).filter(Boolean);
        if (!words.length) return '?';
        const first = Array.from(words[0])[0];
        const second = words.length > 1 ? Array.from(words[1])[0] : '';
        return (first + second).toUpperCase();
    }

    const tileCache = {};

    /** Visible placeholder (SVG data URI) for a picture that does not exist. */
    function fallbackTile(kind, label) {
        const [w, h, font] = TILE[kind] || TILE.item;
        const text = initials(label);
        const id = `${kind}|${text}`;
        if (!tileCache[id]) {
            const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">`
                + `<rect width="${w}" height="${h}" fill="#243044"/>`
                + `<rect x="1.5" y="1.5" width="${w - 3}" height="${h - 3}" fill="none" stroke="#4a5b74" stroke-width="3" stroke-dasharray="6 5"/>`
                + `<text x="${w / 2}" y="${h / 2}" dy="0.36em" text-anchor="middle" fill="#9fb0c8" font-size="${font}" font-weight="700" font-family="Roboto,Arial,sans-serif">${text.replace(/[<>&]/g, '')}</text>`
                + '</svg>';
            tileCache[id] = 'data:image/svg+xml,' + encodeURIComponent(svg);
        }
        return tileCache[id];
    }

    return {
        CDN, load, loaded, meta, onChange, localName,
        heroBySlug, heroById, heroByName, heroSlugByName, heroIconUrl, heroImgUrl,
        item, itemById, itemName, itemIconUrl,
        ability, abilityName, abilityIconUrl, isTalentKey, talentNameZh,
        fallbackTile, initials,
    };
})();
