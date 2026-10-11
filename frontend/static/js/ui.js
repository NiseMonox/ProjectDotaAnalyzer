/**
 * Dota 2 Match Analyzer - shared UI building blocks (window.DotaUI)
 *
 * Global components, registered with DotaUI.install(app). They hold no text
 * of their own: every string comes in through props or slots, already
 * translated by the caller.
 *
 *   <ui-banner kind="info|success|warn|error" title text dismissible dismiss-label @dismiss>
 *       default slot = extra lines, #actions = buttons
 *   <ui-empty icon title text>            empty / "no data" state of a panel, #actions
 *   <ui-skeleton variant="lines|match" :rows label>   loading placeholder
 *   <ui-toasts :items label close-label @close>       stack of { id, kind, text }, see DotaUI.createToasts()
 *   <game-icon kind="hero|hero-portrait|item|ability" k="<slug or key>" label tip>
 *       picture of a hero / item / ability with a visible fallback tile when
 *       the key is empty, unknown, or the picture fails to load. Lookups come
 *       from the root app: provide('gameIcons', { url(kind, key),
 *       label(kind, key), tile(kind, key, name) }).
 *   <hero-tag slug name team sub strong>
 *       hero icon + name in its team colour ('radiant' | 'dire', anything
 *       else is neutral), with optional secondary text (`sub`). The caller
 *       passes the display name; `strong` makes it bold.
 *
 * In-DOM templates need the kebab-case tags and explicit closing tags.
 */
window.DotaUI = (function () {
    'use strict';

    const BANNER_ICONS = {
        info: 'M12 2a10 10 0 100 20 10 10 0 000-20zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z',
        success: 'M12 2a10 10 0 100 20 10 10 0 000-20zm-1.2 14.2l-4-4 1.4-1.4 2.6 2.6 5.6-5.6 1.4 1.4-7 7z',
        warn: 'M12 2.5L1.5 21h21L12 2.5zm1 15.5h-2v-2h2v2zm0-4h-2V9h2v5z',
        error: 'M12 2a10 10 0 100 20 10 10 0 000-20zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z',
    };

    const Banner = {
        props: {
            kind: { type: String, default: 'info' },
            title: { type: String, default: '' },
            text: { type: String, default: '' },
            dismissible: { type: Boolean, default: false },
            dismissLabel: { type: String, default: '' },
        },
        emits: ['dismiss'],
        computed: {
            iconPath() { return BANNER_ICONS[this.kind] || BANNER_ICONS.info; },
        },
        template: `
            <div class="ui-banner" :class="'ui-banner-' + kind" :role="kind === 'error' ? 'alert' : 'status'">
                <svg class="ui-banner-icon" viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true"><path :d="iconPath"/></svg>
                <div class="ui-banner-body">
                    <p v-if="title" class="ui-banner-title">{{ title }}</p>
                    <p v-if="text" class="ui-banner-text">{{ text }}</p>
                    <slot></slot>
                    <div v-if="$slots.actions" class="ui-banner-actions"><slot name="actions"></slot></div>
                </div>
                <button v-if="dismissible" type="button" class="ui-banner-close" :aria-label="dismissLabel" :title="dismissLabel" @click="$emit('dismiss')">
                    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>
                </button>
            </div>`,
    };

    const Empty = {
        props: {
            icon: { type: String, default: '' },
            title: { type: String, default: '' },
            text: { type: String, default: '' },
        },
        template: `
            <div class="empty-state">
                <div v-if="icon" class="empty-state-icon" aria-hidden="true">{{ icon }}</div>
                <p v-if="title" class="empty-state-title">{{ title }}</p>
                <p v-if="text" class="empty-state-desc">{{ text }}</p>
                <slot></slot>
                <div v-if="$slots.actions" class="empty-state-actions"><slot name="actions"></slot></div>
            </div>`,
    };

    const Skeleton = {
        props: {
            variant: { type: String, default: 'lines' },
            rows: { type: Number, default: 4 },
            label: { type: String, default: '' },
        },
        template: `
            <div class="ui-skeleton" :class="'ui-skeleton-' + variant" role="status" aria-busy="true" :aria-label="label">
                <template v-if="variant === 'match'">
                    <div class="ui-skel ui-skel-banner"></div>
                    <div class="ui-skel ui-skel-tabs"></div>
                    <div class="ui-skel-pad">
                        <div v-for="block in 2" :key="block" class="ui-skel-table">
                            <div class="ui-skel ui-skel-head"></div>
                            <div v-for="row in 5" :key="row" class="ui-skel ui-skel-row"></div>
                        </div>
                    </div>
                </template>
                <template v-else>
                    <div v-for="row in rows" :key="row" class="ui-skel ui-skel-line" :style="{ width: (92 - ((row * 17) % 38)) + '%' }"></div>
                </template>
            </div>`,
    };

    const Toasts = {
        props: {
            items: { type: Array, default: () => [] },
            label: { type: String, default: '' },
            closeLabel: { type: String, default: '' },
        },
        emits: ['close'],
        template: `
            <div class="ui-toasts" role="region" aria-live="polite" :aria-label="label">
                <div v-for="toast in items" :key="toast.id" class="ui-toast" :class="'ui-toast-' + toast.kind">
                    <span class="ui-toast-text">{{ toast.text }}</span>
                    <button type="button" class="ui-toast-close" :aria-label="closeLabel" :title="closeLabel" @click="$emit('close', toast.id)">
                        <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>
                    </button>
                </div>
            </div>`,
    };

    const GameIcon = {
        inject: ['gameIcons'],
        props: {
            kind: { type: String, required: true },
            k: { type: String, default: '' },
            // Alt text / tooltip; defaults to the display name of the key.
            label: { type: String, default: '' },
            // true = tooltip is the name, a string = that tooltip, false = none.
            tip: { type: [Boolean, String], default: true },
        },
        data() {
            return { failedUrl: '' };
        },
        computed: {
            name() { return this.label || this.gameIcons.label(this.kind, this.k); },
            url() { return this.gameIcons.url(this.kind, this.k); },
            // Compared by URL, so the flag clears itself when the key changes.
            broken() { return !this.url || this.failedUrl === this.url; },
            src() { return this.broken ? this.gameIcons.tile(this.kind, this.k, this.name) : this.url; },
            tooltip() { return this.tip === true ? (this.name || null) : (this.tip || null); },
        },
        methods: {
            onError() {
                if (!this.broken) this.failedUrl = this.url;
            },
        },
        template: '<img :src="src" :alt="name" :title="tooltip" :class="{ \'icon-fallback\': broken }" @error="onError">',
    };

    const HeroTag = {
        props: {
            slug: { type: String, default: '' },
            name: { type: String, default: '' },
            team: { type: String, default: '' },
            sub: { type: String, default: '' },
            strong: { type: Boolean, default: false },
        },
        computed: {
            teamClass() { return this.team === 'radiant' || this.team === 'dire' ? 'team-' + this.team : 'team-none'; },
        },
        template: `
            <span class="hero-tag" :class="{ 'hero-tag-strong': strong }">
                <game-icon kind="hero" :k="slug" :label="name" :tip="false" class="hero-icon-sm"></game-icon>
                <span class="hero-tag-name" :class="teamClass" :title="name">{{ name }}</span>
                <span v-if="sub" class="hero-tag-sub">{{ sub }}</span>
            </span>`,
    };

    /**
     * Toast store: { items (ref), push(content, kind = 'info', ms = 5000), close(id) }.
     * `content` is what to say, not the sentence: the caller turns each
     * { id, kind, content } into the text shown (so a toast follows a
     * language switch); `sameAs` names it, for the rule below.
     * Errors stay twice as long; ms = 0 keeps a toast until it is closed.
     */
    function createToasts() {
        const items = Vue.ref([]);
        let nextId = 1;
        function close(id) {
            items.value = items.value.filter((toast) => toast.id !== id);
        }
        function push(content, kind, ms, sameAs) {
            if (!content) return 0;
            const id = nextId++;
            const life = ms == null ? (kind === 'error' ? 10000 : 5000) : ms;
            const name = sameAs == null ? content : sameAs;
            // The same message again replaces the one on screen instead of stacking.
            const others = items.value.filter((toast) => toast.name !== name);
            items.value = [...others.slice(-3), { id, content, name, kind: kind || 'info' }];
            if (life > 0) setTimeout(() => close(id), life);
            return id;
        }
        return { items, push, close };
    }

    function install(app) {
        app.component('ui-banner', Banner);
        app.component('ui-empty', Empty);
        app.component('ui-skeleton', Skeleton);
        app.component('ui-toasts', Toasts);
        app.component('game-icon', GameIcon);
        app.component('hero-tag', HeroTag);
    }

    return { install, createToasts };
})();
