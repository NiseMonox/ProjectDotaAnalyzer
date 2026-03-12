import shutil
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPLAY_DIR = BASE_DIR / "replays"
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

# Parser settings
CLARITY_JAR_PATH = BASE_DIR / "tools" / "clarity-analyzer.jar"

# Java path: prefer project-local JDK 17 (required for clarity 3.x),
# fall back to system java if the local JDK is missing.
_LOCAL_JDK17 = BASE_DIR / "tools" / "jdk17" / "Contents" / "Home" / "bin" / "java"
if _LOCAL_JDK17.exists():
    JAVA_PATH = str(_LOCAL_JDK17)
elif shutil.which("java"):
    JAVA_PATH = shutil.which("java")
else:
    JAVA_PATH = "java"

# Default test replay
DEFAULT_REPLAY = REPLAY_DIR / "8713267082.dem"

# Server settings
HOST = "0.0.0.0"
PORT = 8000
DEBUG = True

# Ensure directories exist
REPLAY_DIR.mkdir(parents=True, exist_ok=True)
