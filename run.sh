#!/usr/bin/env bash
# Run the Dota 2 Match Analyzer.
#
#   ./run.sh                 # production-style (no auto-reload)
#   DEV=1 ./run.sh           # auto-reload on code changes
#   HOST=127.0.0.1 PORT=8001 ./run.sh
#
# Creates backend/venv on first run (Python 3.10-3.13; PYTHON=... selects the
# interpreter) and reinstalls dependencies whenever backend/requirements.txt
# changes. Settings come from the environment or .env (see .env.example); the
# environment wins. Optional extras (backend/requirements-rag.txt) are never
# installed here.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
VENV_DIR="$BACKEND_DIR/venv"
REQUIREMENTS="$BACKEND_DIR/requirements.txt"
STAMP="$VENV_DIR/.requirements.sha256"

# Value of KEY from .env (last assignment, quotes stripped); the app itself
# loads .env too, this is only for the few settings run.sh needs.
dotenv_value() {
    [ -f "$PROJECT_DIR/.env" ] || return 0
    sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*//p" "$PROJECT_DIR/.env" \
        | tail -n 1 | sed -e 's/[[:space:]]*$//' -e 's/^["'"'"']//' -e 's/["'"'"']$//'
}

HOST="${HOST:-$(dotenv_value HOST)}"; HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-$(dotenv_value PORT)}"; PORT="${PORT:-8000}"
DEV="${DEV:-$(dotenv_value DEV)}"; DEV="${DEV:-0}"
PYTHON="${PYTHON:-python3}"
JAVA_PATH="${JAVA_PATH:-$(dotenv_value JAVA_PATH)}"
CLARITY_JAR_PATH="${CLARITY_JAR_PATH:-$(dotenv_value CLARITY_JAR_PATH)}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    # The pinned dependencies have wheels for these versions only; with
    # another one pip fails halfway with an error about a missing package.
    PY_VERSION="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    case "$PY_VERSION" in
        3.10|3.11|3.12|3.13) ;;
        "") echo "ERROR: cannot run '$PYTHON'. Set PYTHON to a Python 3.10-3.13 interpreter." >&2; exit 1 ;;
        *)  echo "ERROR: $PYTHON is Python $PY_VERSION, this project needs Python 3.10-3.13" >&2
            echo "       (for example: PYTHON=python3.12 ./run.sh)." >&2; exit 1 ;;
    esac
    echo "Creating virtual environment in $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
    rm -f "$STAMP"
fi

install_requirements() {
    local py="$VENV_DIR/bin/python"
    if "$py" -m pip --version >/dev/null 2>&1 || "$py" -m ensurepip --upgrade >/dev/null 2>&1; then
        "$py" -m pip install --upgrade pip >/dev/null 2>&1 || true
        "$py" -m pip install -r "$REQUIREMENTS"
    elif command -v uv >/dev/null 2>&1; then
        # venvs created by uv have no pip
        uv pip install --python "$py" -r "$REQUIREMENTS"
    else
        echo "ERROR: pip is not available in $VENV_DIR (install python3-venv / python3-pip)." >&2
        exit 1
    fi
}

# Portable hash (no sha256sum on macOS): use the venv's Python.
REQ_HASH="$("$VENV_DIR/bin/python" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$REQUIREMENTS")"
if [ ! -f "$STAMP" ] || [ "$(cat "$STAMP")" != "$REQ_HASH" ]; then
    echo "Installing dependencies from backend/requirements.txt ..."
    install_requirements
    echo "$REQ_HASH" > "$STAMP"
fi

# Relative paths in settings are relative to the project root.
abs_path() {
    case "$1" in
        /*) echo "$1" ;;
        *) echo "$PROJECT_DIR/$1" ;;
    esac
}

# Replay parsing needs Java 17+ and the clarity jar; without them uploads
# return 503 and only "fetch by match ID" (OpenDota) works.
JAVA_BIN="${JAVA_PATH:-}"
case "$JAVA_BIN" in
    "") ;;
    */*) JAVA_BIN="$(abs_path "$JAVA_BIN")"; [ -x "$JAVA_BIN" ] || JAVA_BIN="" ;;
    *) JAVA_BIN="$(command -v "$JAVA_BIN" || true)" ;;
esac
if [ -z "$JAVA_BIN" ]; then
    for candidate in \
        "$PROJECT_DIR/tools/jdk17/Contents/Home/bin/java" \
        "$PROJECT_DIR/tools/jdk17/bin/java"; do
        if [ -x "$candidate" ]; then JAVA_BIN="$candidate"; break; fi
    done
fi
if [ -z "$JAVA_BIN" ] && [ -n "${JAVA_HOME:-}" ] && [ -x "$JAVA_HOME/bin/java" ]; then
    JAVA_BIN="$JAVA_HOME/bin/java"
fi
if [ -z "$JAVA_BIN" ]; then
    JAVA_BIN="$(command -v java || true)"
fi
if [ -z "$JAVA_BIN" ]; then
    echo "WARNING: Java 17+ not found - replay upload/parsing is disabled (match-ID fetch still works)."
    echo "         Install openjdk-17 or run tools/setup_parser.sh."
fi
JAR_PATH="$(abs_path "${CLARITY_JAR_PATH:-tools/clarity-analyzer.jar}")"
if [ ! -f "$JAR_PATH" ]; then
    echo "WARNING: $JAR_PATH not found - build it with: bash tools/setup_parser.sh"
fi

RELOAD_ARGS=()
if [ "$DEV" = "1" ]; then
    RELOAD_ARGS=(--reload --reload-dir "$BACKEND_DIR/app")
fi

DISPLAY_HOST="$HOST"
if [ "$HOST" = "0.0.0.0" ]; then DISPLAY_HOST="localhost"; fi

echo ""
echo "=== Dota 2 Match Analyzer ==="
echo "  URL: http://$DISPLAY_HOST:$PORT"
echo "  API: http://$DISPLAY_HOST:$PORT/api/demo"
if [ "$DEV" = "1" ]; then echo "  Auto-reload: on (DEV=1)"; fi
echo ""
echo "  Press Ctrl+C to stop"
echo ""

cd "$PROJECT_DIR"
# ${arr[@]+...} keeps `set -u` happy with an empty array on bash 3.2 (macOS).
exec "$VENV_DIR/bin/uvicorn" backend.app.main:app --host "$HOST" --port "$PORT" ${RELOAD_ARGS[@]+"${RELOAD_ARGS[@]}"}
