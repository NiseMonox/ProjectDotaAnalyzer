#!/bin/bash
# Run the Dota 2 Match Analyzer
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"

# Check if virtual environment exists
if [ ! -d "$BACKEND_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$BACKEND_DIR/venv"
    echo "Installing dependencies..."
    "$BACKEND_DIR/venv/bin/pip" install -r "$BACKEND_DIR/requirements.txt"
fi

echo ""
echo "=== Dota 2 Match Analyzer ==="
echo "  URL: http://localhost:8000"
echo "  API: http://localhost:8000/api/demo"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

cd "$PROJECT_DIR"
"$BACKEND_DIR/venv/bin/uvicorn" backend.app.main:app --host 0.0.0.0 --port 8000 --reload
