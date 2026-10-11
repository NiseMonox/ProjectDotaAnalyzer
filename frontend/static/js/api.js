/**
 * Dota 2 Match Analyzer - API layer (window.DotaApi)
 *
 * One request helper for every call to /api:
 *   - adds ?lang=<current language>, so the backend's `detail` arrives localized
 *   - never throws: resolves to { ok: true, data, status } or { ok: false, error }
 *   - turns every failure into the same error object:
 *       { code, status, detail, lang, retryAfter, retryAt, body, client }
 *     `code` is the backend's code ({detail, code, retry_after?} bodies) or one
 *     raised here: network, timeout, aborted, bad_response (2xx that is not a
 *     JSON object), server_error (5xx without a JSON body, e.g. the server is
 *     restarting), http_error (other statuses without a JSON body).
 *     `retryAt` is an epoch-ms timestamp when the body or the Retry-After
 *     header gave a wait time (rate limits), else 0.
 *   - stream() reads the AI-coach event stream and ends EventSource's
 *     endless reconnecting; errorFromEvent() turns its error events into
 *     the same error object.
 */
window.DotaApi = (function () {
    'use strict';

    const DEFAULT_TIMEOUT_MS = 60000;
    let currentLang = 'zh';

    function setLang(lang) {
        currentLang = lang === 'en' ? 'en' : 'zh';
    }

    /** Path + query for an API call; `lang` is added unless already present. */
    function url(path, params) {
        const u = new URL(path, window.location.href);
        if (!u.searchParams.has('lang')) u.searchParams.set('lang', currentLang);
        Object.entries(params || {}).forEach(([key, value]) => {
            if (value != null) u.searchParams.set(key, value);
        });
        return u.pathname + u.search;
    }

    function parseJson(text) {
        try { return JSON.parse(text); } catch (e) { return undefined; }
    }

    function isObject(value) {
        return value !== null && typeof value === 'object' && !Array.isArray(value);
    }

    function makeError(code, fields) {
        const f = fields || {};
        const retryAfter = Number(f.retryAfter) > 0 ? Math.ceil(Number(f.retryAfter)) : 0;
        return {
            code,
            status: f.status || 0,
            detail: typeof f.detail === 'string' ? f.detail : '',
            lang: f.lang || currentLang,
            retryAfter,
            retryAt: retryAfter ? Date.now() + retryAfter * 1000 : 0,
            body: isObject(f.body) ? f.body : null,
            client: !!f.client,
        };
    }

    /** Error object for a non-2xx response (JSON error body or not). */
    function errorFromResponse(status, body, retryAfterHeader, lang) {
        const obj = isObject(body) ? body : null;
        const code = obj && typeof obj.code === 'string' && obj.code
            ? obj.code
            : (status >= 500 ? 'server_error' : 'http_error');
        return makeError(code, {
            status,
            detail: obj ? obj.detail : '',
            retryAfter: (obj && obj.retry_after) || retryAfterHeader,
            body: obj,
            lang,
            client: !obj,
        });
    }

    /**
     * Error object for an AI-coach SSE event
     * {type: 'error', message, code, status, retry_after?} (sent with HTTP 200).
     */
    function errorFromEvent(msg, lang) {
        const m = isObject(msg) ? msg : {};
        return makeError(typeof m.code === 'string' && m.code ? m.code : 'server_error', {
            status: m.status,
            detail: m.message,
            retryAfter: m.retry_after,
            body: m,
            lang: lang || currentLang,
        });
    }

    /**
     * JSON request. opts: { method, params, timeoutMs (0 = none), signal }.
     * Resolves, never rejects.
     */
    async function request(path, opts) {
        const o = opts || {};
        const lang = currentLang;
        const controller = new AbortController();
        const timeoutMs = o.timeoutMs == null ? DEFAULT_TIMEOUT_MS : o.timeoutMs;
        let timedOut = false;
        const timer = timeoutMs > 0
            ? setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs)
            : null;
        const onOuterAbort = () => controller.abort();
        if (o.signal) {
            if (o.signal.aborted) controller.abort();
            else o.signal.addEventListener('abort', onOuterAbort, { once: true });
        }
        try {
            const res = await fetch(url(path, o.params), {
                method: o.method || 'GET',
                headers: { Accept: 'application/json' },
                signal: controller.signal,
            });
            const body = parseJson(await res.text());
            if (!res.ok) {
                return { ok: false, error: errorFromResponse(res.status, body, res.headers.get('Retry-After'), lang) };
            }
            if (!isObject(body)) {
                return { ok: false, error: makeError('bad_response', { status: res.status, lang, client: true }) };
            }
            return { ok: true, data: body, status: res.status };
        } catch (e) {
            const code = timedOut ? 'timeout' : (controller.signal.aborted ? 'aborted' : 'network');
            return { ok: false, error: makeError(code, { lang, client: true }) };
        } finally {
            if (timer) clearTimeout(timer);
            if (o.signal) o.signal.removeEventListener('abort', onOuterAbort);
        }
    }

    function get(path, opts) {
        return request(path, Object.assign({}, opts, { method: 'GET' }));
    }

    function post(path, opts) {
        return request(path, Object.assign({}, opts, { method: 'POST' }));
    }

    /**
     * Multipart upload with progress (XMLHttpRequest: fetch has no upload
     * progress). opts: { field, onProgress({loaded, total}), onSent(),
     * processTimeoutMs }. `total` is 0 when the size is not computable.
     * processTimeoutMs bounds the wait for the answer after the body was
     * sent (the server parses the replay then), so the UI cannot wait forever.
     * Returns { promise, abort }; the promise resolves like request().
     */
    function upload(path, file, opts) {
        const o = opts || {};
        const lang = currentLang;
        const xhr = new XMLHttpRequest();
        let settled = false;
        let timedOut = false;
        let processTimer = null;

        const promise = new Promise((resolve) => {
            const finish = (result) => {
                if (settled) return;
                settled = true;
                if (processTimer) clearTimeout(processTimer);
                resolve(result);
            };
            xhr.upload.onprogress = (ev) => {
                if (o.onProgress) o.onProgress({ loaded: ev.loaded, total: ev.lengthComputable ? ev.total : 0 });
            };
            xhr.upload.onload = () => {
                if (o.onSent) o.onSent();
                if (o.processTimeoutMs > 0) {
                    processTimer = setTimeout(() => { timedOut = true; xhr.abort(); }, o.processTimeoutMs);
                }
            };
            xhr.onload = () => {
                const body = parseJson(xhr.responseText);
                if (xhr.status < 200 || xhr.status >= 300) {
                    finish({ ok: false, error: errorFromResponse(xhr.status, body, xhr.getResponseHeader('Retry-After'), lang) });
                } else if (!isObject(body)) {
                    finish({ ok: false, error: makeError('bad_response', { status: xhr.status, lang, client: true }) });
                } else {
                    finish({ ok: true, data: body, status: xhr.status });
                }
            };
            xhr.onerror = () => finish({ ok: false, error: makeError('network', { lang, client: true }) });
            xhr.onabort = () => finish({ ok: false, error: makeError(timedOut ? 'timeout' : 'aborted', { lang, client: true }) });

            const form = new FormData();
            form.append(o.field || 'file', file, file.name);
            xhr.open('POST', url(path));
            xhr.setRequestHeader('Accept', 'application/json');
            xhr.send(form);
        });

        return {
            promise,
            abort() {
                if (!settled) {
                    try { xhr.abort(); } catch (e) { /* already finished */ }
                }
            },
        };
    }

    /**
     * Server-sent events over EventSource (the AI-coach stream). Every
     * `data:` line is parsed as JSON and handed to onMessage; ': ping'
     * heartbeat comments never reach the page.
     *
     * EventSource reconnects by itself after a dropped connection (the
     * server then re-attaches to the running job), and it would do so
     * forever against a server that is down. The caps below end that:
     * onFail(reason) is called once and the stream is closed.
     *   'closed'       the browser gave up (answer was not a 200 event stream)
     *   'unreachable'  more than maxRetries connection attempts failed in a row
     *   'unstable'     more than maxReconnects reconnects in total
     *   'idle'         connected, but no event for idleMs
     * opts: { params, onMessage(obj), onFail(reason), maxRetries = 2,
     *         maxReconnects = 5, idleMs = 0 (no limit) }
     * Returns { close() }. After close() or onFail() no handler runs again.
     */
    function stream(path, opts) {
        const o = opts || {};
        const maxRetries = o.maxRetries == null ? 2 : o.maxRetries;
        const maxReconnects = o.maxReconnects == null ? 5 : o.maxReconnects;
        let source = null;
        let over = false;
        let failuresInARow = 0;
        let reconnects = 0;
        let idleTimer = null;

        function close() {
            over = true;
            if (idleTimer) clearTimeout(idleTimer);
            if (source) {
                try { source.close(); } catch (e) { /* already closed */ }
            }
        }
        function fail(reason) {
            if (over) return;
            close();
            if (o.onFail) o.onFail(reason);
        }
        function touch() {
            if (!(o.idleMs > 0)) return;
            if (idleTimer) clearTimeout(idleTimer);
            idleTimer = setTimeout(() => fail('idle'), o.idleMs);
        }

        try {
            source = new EventSource(url(path, o.params));
        } catch (e) {
            // No EventSource (or a malformed URL): report it like a refused stream.
            setTimeout(() => fail('closed'), 0);
            return { close };
        }
        touch();
        source.onopen = () => {
            failuresInARow = 0;
            touch();
        };
        source.onmessage = (event) => {
            if (over) return;
            failuresInARow = 0;
            touch();
            const msg = parseJson(event.data);
            if (isObject(msg) && o.onMessage) o.onMessage(msg);
        };
        source.onerror = () => {
            if (over) return;
            if (source.readyState === EventSource.CLOSED) return fail('closed');
            // readyState CONNECTING: the browser is about to reconnect.
            failuresInARow += 1;
            reconnects += 1;
            if (failuresInARow > maxRetries) return fail('unreachable');
            if (reconnects > maxReconnects) return fail('unstable');
            touch();
        };
        return { close };
    }

    return { setLang, url, request, get, post, upload, stream, makeError, errorFromEvent };
})();
