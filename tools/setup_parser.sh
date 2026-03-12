#!/bin/bash
# Setup script for the Clarity replay parser
# Clarity is a Java library for parsing Dota 2 replay files
#
# REQUIREMENTS:
#   - Java 17+ (clarity 3.x requires Java 17; Java 11 is NOT sufficient)
#   - A pre-built JDK 17 is included at tools/jdk17/ for convenience
#
# This script can either:
#   1. Download a pre-built clarity-analyzer.jar (if available)
#   2. Build from source using the Gradle project in tools/clarity-analyzer/

set -e

TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$TOOLS_DIR")"

echo "=== Dota 2 Replay Parser Setup ==="
echo ""

# --- Locate Java 17+ ---

# Prefer the project-local JDK 17 bundled in tools/jdk17
LOCAL_JDK="${TOOLS_DIR}/jdk17/Contents/Home/bin/java"
if [ -x "$LOCAL_JDK" ]; then
    JAVA_CMD="$LOCAL_JDK"
    echo "Using project-local JDK 17: $LOCAL_JDK"
elif command -v java &> /dev/null; then
    JAVA_CMD="java"
    echo "Using system Java: $(command -v java)"
else
    echo "ERROR: Java not found. Java 17+ is required for clarity 3.x."
    echo ""
    echo "  Option 1: Use the bundled JDK 17 (should be at tools/jdk17/)"
    echo "  Option 2: Install Java 17+:"
    echo "    macOS:   brew install openjdk@17"
    echo "    Ubuntu:  sudo apt install openjdk-17-jdk"
    echo "    Windows: Download from https://adoptium.net/"
    echo ""
    exit 1
fi

JAVA_VERSION=$("$JAVA_CMD" -version 2>&1 | head -n1)
echo "Java version: $JAVA_VERSION"

# Verify Java version is 17+
JAVA_MAJOR=$("$JAVA_CMD" -version 2>&1 | head -n1 | sed -E 's/.*"([0-9]+).*/\1/')
if [ "$JAVA_MAJOR" -lt 17 ] 2>/dev/null; then
    echo ""
    echo "WARNING: Java $JAVA_MAJOR detected, but Java 17+ is required."
    echo "  Clarity 3.x will NOT work with Java $JAVA_MAJOR."
    echo "  A JDK 17 is bundled at: tools/jdk17/"
    echo ""
    exit 1
fi
echo ""

# --- Build or download clarity-analyzer ---

CLARITY_JAR="${TOOLS_DIR}/clarity-analyzer.jar"
CLARITY_SRC="${TOOLS_DIR}/clarity-analyzer"

# Option A: Build from source if the Gradle project exists
if [ -d "$CLARITY_SRC" ] && [ -f "$CLARITY_SRC/build.gradle" ]; then
    echo "Found clarity-analyzer source at: $CLARITY_SRC"
    echo "Building from source with Gradle..."
    echo ""

    export JAVA_HOME="${TOOLS_DIR}/jdk17/Contents/Home"

    cd "$CLARITY_SRC"
    if [ -x "./gradlew" ]; then
        ./gradlew shadowJar --no-daemon
    else
        echo "ERROR: gradlew not found or not executable in $CLARITY_SRC"
        echo "  Try: chmod +x ${CLARITY_SRC}/gradlew"
        exit 1
    fi

    # Find the built shadow/fat JAR
    BUILT_JAR=$(find "${CLARITY_SRC}/build/libs" -name "*-all.jar" -o -name "*shadow*.jar" | head -n1)
    if [ -z "$BUILT_JAR" ]; then
        # Fall back to any JAR in build/libs
        BUILT_JAR=$(find "${CLARITY_SRC}/build/libs" -name "*.jar" | head -n1)
    fi

    if [ -n "$BUILT_JAR" ]; then
        cp "$BUILT_JAR" "$CLARITY_JAR"
        echo ""
        echo "Built and copied JAR to: $CLARITY_JAR"
    else
        echo "ERROR: Build succeeded but no JAR found in ${CLARITY_SRC}/build/libs/"
        exit 1
    fi

# Option B: JAR already exists
elif [ -f "$CLARITY_JAR" ]; then
    echo "clarity-analyzer.jar already exists at: $CLARITY_JAR"
    echo "Skipping download. Delete the JAR and re-run to rebuild/re-download."

# Option C: No source, no JAR - show instructions
else
    echo "No clarity-analyzer source or JAR found."
    echo ""
    echo "To build from source:"
    echo "  1. Place the clarity-analyzer Gradle project in: ${CLARITY_SRC}/"
    echo "  2. Re-run this script"
    echo ""
    echo "The source project should have this structure:"
    echo "  tools/clarity-analyzer/"
    echo "    build.gradle"
    echo "    gradlew"
    echo "    src/"
    echo "      main/java/..."
    echo ""
    exit 1
fi

echo ""
echo "=== Setup Complete ==="
echo "  Parser JAR: ${CLARITY_JAR}"
echo "  Java:       $JAVA_CMD ($JAVA_VERSION)"
echo ""
echo "To test, run:"
echo "  $JAVA_CMD -jar $CLARITY_JAR <path-to-replay.dem>"
echo ""
echo "See also: https://github.com/skadistats/clarity"
