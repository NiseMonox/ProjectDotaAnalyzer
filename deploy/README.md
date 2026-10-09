# Deployment Run-Book

Example target: an Ubuntu 22.04 server (login user `ubuntu`, Python 3.10), written `<server-ip>` below.
Install path: `/opt/dota-analyzer`.
URL without a reverse proxy: `http://<server-ip>:8001`.

## First-time install

```bash
# System deps (Java 17 runs the replay parser; the JDK can also rebuild the jar)
sudo apt-get update
sudo apt-get install -y openjdk-17-jdk-headless python3.10-venv python3-pip git

# App
sudo mkdir -p /opt/dota-analyzer && sudo chown ubuntu:ubuntu /opt/dota-analyzer
git clone https://github.com/NiseMonox/ProjectDotaAnalyzer.git /opt/dota-analyzer
cd /opt/dota-analyzer
python3 -m venv backend/venv
backend/venv/bin/pip install -U pip
backend/venv/bin/pip install -r backend/requirements.txt   # includes openai + python-dotenv

# Replay parser jar (tools/clarity-analyzer.jar). It is committed; rebuild it
# after changing tools/clarity-analyzer/ or if it is missing:
bash tools/setup_parser.sh

# Configuration + secrets (see .env.example for every option)
umask 077
cp .env.example .env
# edit .env: set at least DEEPSEEK_API_KEY and ADMIN_TOKEN, e.g.
#   DEEPSEEK_API_KEY=sk-...
#   ADMIN_TOKEN=$(openssl rand -hex 32)
chmod 600 .env

# Systemd
sudo cp deploy/dota-analyzer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dota-analyzer
systemctl is-active dota-analyzer
```

Open TCP 8001 in the cloud provider's firewall (security group) before expecting the app to answer from the public internet.

Run exactly **one** uvicorn worker (the unit does): rate limits, the AI-coach
daily quota, caches and in-flight de-duplication are kept per process.

## Configuration that matters on a public server

All settings are environment variables, read from `/opt/dota-analyzer/.env`
(documented in `.env.example`). The important ones:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | empty | LLM for the AI coach; empty = rule-based reports only |
| `ADMIN_TOKEN` | empty | Required (`X-Admin-Token` header) for `/api/admin/*` and `/api/parse-local/*`; empty = those endpoints return 403 |
| `AI_COACH_DAILY_LIMIT` | `200` | Max cold LLM-backed coach reports per UTC day (0 = unlimited); cached reports are free |
| `RATE_LIMIT_UPLOAD` | `10/3600` | Per-IP uploads per window (`count/seconds`, `0` disables) |
| `RATE_LIMIT_FETCH` | `30/600` | Per-IP OpenDota fetches (cache hits are free) |
| `RATE_LIMIT_REQUEST_PARSE` | `10/600` | Per-IP OpenDota parse requests |
| `RATE_LIMIT_AI_COACH` | `10/3600` | Per-IP cold AI-coach generations |
| `MAX_UPLOAD_MB` | `300` | Upload size limit (HTTP 413 beyond it) |
| `REPLAY_MAX_FILES` | `50` | Uploaded replays kept on disk (oldest deleted) |
| `MATCH_CACHE_MAX_FILES` | `200` | Parsed matches kept in `cache/matches/` |
| `TRUST_PROXY_HEADERS` | `0` | Use `X-Real-IP` / `X-Forwarded-For` as the client IP (only behind nginx) |
| `ALLOWED_ORIGINS` | empty | CORS allowlist; empty = same-origin only (the bundled frontend) |
| `CLARITY_MAX_CONCURRENCY` / `CLARITY_JVM_XMX` | `1` / `1536m` | Parallel replay parses and JVM heap per parse — size to the server's RAM |

Rate-limited requests get HTTP 429 with `{"detail", "code", "retry_after"}`
and a `Retry-After` header.

Admin calls:

```bash
TOKEN=$(grep ^ADMIN_TOKEN= /opt/dota-analyzer/.env | cut -d= -f2-)
curl -sS -H "X-Admin-Token: $TOKEN" http://127.0.0.1:8001/api/admin/status
curl -sS -H "X-Admin-Token: $TOKEN" http://127.0.0.1:8001/api/admin/gamedata-status
curl -sS -X POST -H "X-Admin-Token: $TOKEN" "http://127.0.0.1:8001/api/admin/update-gamedata?force=true"
```

## Redeploy after a `git push`

```bash
cd /opt/dota-analyzer
git pull
backend/venv/bin/pip install -r backend/requirements.txt   # no-op when unchanged
sudo systemctl restart dota-analyzer
journalctl -u dota-analyzer -n 50 --no-pager
```

Check `.env.example` for new settings after pulling. If
`tools/clarity-analyzer/` changed, rebuild the jar with `bash tools/setup_parser.sh`.

## Diagnostics

```bash
systemctl status dota-analyzer
journalctl -u dota-analyzer -f             # follow logs (LOG_LEVEL=DEBUG for more)
journalctl -u dota-analyzer --since "10 min ago"
ss -tlnp | grep 8001                       # is it listening?
curl -sS http://127.0.0.1:8001/api/demo | head -c 200
```

On startup the app refreshes game data in the background first and only then
runs the AI-coach self-learning job (set `GAMEDATA_AUTO_UPDATE=0` /
`AI_COACH_AUTO_LEARN=0` to skip either). Neither blocks serving requests.

## Rollback

```bash
sudo systemctl stop dota-analyzer
sudo systemctl disable dota-analyzer
# Optional: sudo rm -rf /opt/dota-analyzer
```

Nothing else on this host is affected — the app is self-contained.

## Later: add a domain + TLS (nginx)

When a domain is ready (e.g. `dota.example.com` → `<server-ip>`):

1. In the service unit change `--host 0.0.0.0` to `--host 127.0.0.1`, then
   `sudo systemctl daemon-reload && sudo systemctl restart dota-analyzer`.
2. `sudo apt-get install -y nginx` and install `deploy/nginx-dota-analyzer.conf`
   (instructions inside; set `server_name`). It already has the important bits:
   `client_max_body_size` above `MAX_UPLOAD_MB`, `proxy_buffering off` and a
   long `proxy_read_timeout` for the AI-coach SSE stream, long timeouts for
   uploads, and `X-Real-IP` / `X-Forwarded-For` headers.
3. Set `TRUST_PROXY_HEADERS=1` in `.env` so per-IP rate limits see real client
   IPs. Only do this when uvicorn listens on 127.0.0.1 — otherwise clients
   could reach port 8001 directly and spoof those headers.
4. `sudo certbot --nginx -d dota.example.com -m you@example.com --agree-tos --redirect`.
5. Close TCP 8001 in the security group; only 80/443 should stay open publicly.
6. CORS is off by default (same-origin). Only if another origin must call the
   API, set `ALLOWED_ORIGINS=https://other.example.com` in `.env`.
