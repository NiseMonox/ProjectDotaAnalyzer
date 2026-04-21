# Deployment Run-Book

Example target: an Ubuntu 22.04 server (login user `ubuntu`), written `<server-ip>` below.
Install path: `/opt/dota-analyzer`.
URL without a reverse proxy: `http://<server-ip>:8001`.

## First-time install

```bash
# System deps
sudo apt-get update
sudo apt-get install -y openjdk-17-jdk-headless python3.10-venv python3-pip git

# App
sudo mkdir -p /opt/dota-analyzer && sudo chown ubuntu:ubuntu /opt/dota-analyzer
git clone https://github.com/NiseMonox/ProjectDotaAnalyzer.git /opt/dota-analyzer
cd /opt/dota-analyzer
python3 -m venv backend/venv
backend/venv/bin/pip install -U pip
backend/venv/bin/pip install -r backend/requirements.txt
backend/venv/bin/pip install "openai>=1.30" "python-dotenv>=1.0"

# Secret
umask 077
printf 'DEEPSEEK_API_KEY=%s\n' "$YOUR_KEY" > /opt/dota-analyzer/.env
chmod 600 /opt/dota-analyzer/.env

# Systemd
sudo cp deploy/dota-analyzer.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dota-analyzer
systemctl is-active dota-analyzer
```

Open TCP 8001 in the cloud provider's firewall (security group) before expecting the app to answer from the public internet.

## Redeploy after a `git push`

```bash
cd /opt/dota-analyzer
git pull
# Only if requirements.txt changed:
backend/venv/bin/pip install -r backend/requirements.txt
sudo systemctl restart dota-analyzer
journalctl -u dota-analyzer -n 50 --no-pager
```

## Diagnostics

```bash
systemctl status dota-analyzer
journalctl -u dota-analyzer -f             # follow logs
journalctl -u dota-analyzer --since "10 min ago"
ss -tlnp | grep 8001                       # is it listening?
curl -sS http://127.0.0.1:8001/api/demo | head -c 200
```

## Rollback

```bash
sudo systemctl stop dota-analyzer
sudo systemctl disable dota-analyzer
# Optional: sudo rm -rf /opt/dota-analyzer
```

Nothing else on this host is affected — the app is self-contained.

## Later: add a domain + TLS

When a domain is ready (e.g. `dota.example.com` → `<server-ip>`):

1. Flip uvicorn to `--host 127.0.0.1 --port 8001` in the service unit.
2. Add an nginx vhost that `proxy_pass`es to `http://127.0.0.1:8001`.
3. `sudo certbot --nginx -d dota.example.com -m you@example.com --agree-tos --redirect`.
4. Close TCP 8001 in the security group; only 80/443 should stay open publicly.
5. Tighten CORS in `backend/app/main.py` to allowlist the domain.
