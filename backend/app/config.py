from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPLAY_DIR = BASE_DIR / "replays"
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

# Parser settings
CLARITY_JAR_PATH = BASE_DIR / "tools" / "clarity-analyzer.jar"
JAVA_PATH = "java"

# Default test replay
DEFAULT_REPLAY = REPLAY_DIR / "8713267082.dem"

# Server settings
HOST = "0.0.0.0"
PORT = 8000
DEBUG = True

# Ensure directories exist
REPLAY_DIR.mkdir(parents=True, exist_ok=True)
