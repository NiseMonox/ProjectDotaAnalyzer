#!/usr/bin/env bash
# Setup script for the Clarity replay parser (tools/clarity-analyzer.jar).
#
# REQUIREMENTS:
#   - Java 17+ (clarity 4.x requires Java 17; Java 11 is NOT sufficient)
#   - A JDK (javac) when building from source
#
# Java is looked up in this order:
#   1. A project-local JDK in tools/jdk17/ (macOS layout tools/jdk17/Contents/Home
#      or Linux layout tools/jdk17/bin/java) — optional, gitignored
#   2. $JAVA_HOME
#   3. `java` on PATH
#
# The committed JAR is used as it is: this script only checks Java and that
# the JAR starts. It builds from source when the JAR is missing, or when asked
# to with REBUILD=1 (after changing tools/clarity-analyzer/).
#
# Why not always: the committed JAR was built with Eclipse Temurin 17.0.13+11,
# and the build is reproducible for that JDK build only. Another JDK 17
# (Ubuntu's openjdk-17, Homebrew) gives a JAR that behaves the same but is not
# byte-identical: git then shows the tracked tools/clarity-analyzer.jar as
# modified, and the next `git pull` that brings a new JAR is refused. If that
# happened on a server: `git checkout -- tools/clarity-analyzer.jar`.
#
# Environment:
#   REBUILD        1 = build from source even though the JAR exists
#   MAVEN_MIRROR   optional extra Maven repository tried before Maven Central,
#                  e.g. https://maven.aliyun.com/repository/public (China).
#                  The Gradle wrapper itself is downloaded from
#                  services.gradle.org, which this does not cover: use
#                  GRADLE_CMD where that host cannot be reached.
#   GRADLE_CMD     build with this Gradle instead of ./gradlew (e.g. "gradle")
#
# Usage: bash tools/setup_parser.sh            # check Java + the committed JAR
#        REBUILD=1 bash tools/setup_parser.sh  # rebuild the JAR from source

set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
CLARITY_JAR="${TOOLS_DIR}/clarity-analyzer.jar"
CLARITY_SRC="${TOOLS_DIR}/clarity-analyzer"

echo "=== Dota 2 Replay Parser Setup ==="
echo ""

# --- Locate Java 17+ ---------------------------------------------------------

JAVA_HOME_DIR=""
for candidate in "${TOOLS_DIR}/jdk17/Contents/Home" "${TOOLS_DIR}/jdk17"; do
    if [ -x "${candidate}/bin/java" ]; then
        JAVA_HOME_DIR="$candidate"
        echo "Using project-local JDK: ${JAVA_HOME_DIR}"
        break
    fi
done

if [ -z "$JAVA_HOME_DIR" ] && [ -n "${JAVA_HOME:-}" ] && [ -x "${JAVA_HOME}/bin/java" ]; then
    JAVA_HOME_DIR="$JAVA_HOME"
    echo "Using JAVA_HOME: ${JAVA_HOME_DIR}"
fi

if [ -z "$JAVA_HOME_DIR" ] && command -v java > /dev/null 2>&1; then
    if [ "$(uname -s)" = "Darwin" ] && [ -x /usr/libexec/java_home ]; then
        # java_home answers with the newest registered JDK even when none
        # satisfies "17+", so check what it returned before trusting it.
        CANDIDATE="$(/usr/libexec/java_home -v 17+ 2> /dev/null || true)"
        if [ -n "$CANDIDATE" ] && [ -x "${CANDIDATE}/bin/java" ]; then
            CANDIDATE_MAJOR="$("${CANDIDATE}/bin/java" -version 2>&1 | sed -n 's/.*version "\([0-9][0-9]*\).*/\1/p' | head -n 1)"
            if [ "${CANDIDATE_MAJOR:-0}" -ge 17 ] 2> /dev/null; then
                JAVA_HOME_DIR="$CANDIDATE"
            fi
        fi
    fi
    if [ -z "$JAVA_HOME_DIR" ]; then
        JAVA_BIN="$(command -v java)"
        # Resolve symlinks (/usr/bin/java -> /usr/lib/jvm/.../bin/java) portably.
        while [ -L "$JAVA_BIN" ]; do
            LINK="$(readlink "$JAVA_BIN")"
            case "$LINK" in
                /*) JAVA_BIN="$LINK" ;;
                *) JAVA_BIN="$(dirname "$JAVA_BIN")/$LINK" ;;
            esac
        done
        JAVA_HOME_DIR="$(cd "$(dirname "$JAVA_BIN")/.." && pwd)"
    fi
    echo "Using system Java: ${JAVA_HOME_DIR}"
fi

if [ -z "$JAVA_HOME_DIR" ] || [ ! -x "${JAVA_HOME_DIR}/bin/java" ]; then
    echo "ERROR: Java not found. Java 17+ is required for clarity 4.x."
    echo ""
    echo "  Install Java 17+:"
    echo "    Ubuntu:  sudo apt install openjdk-17-jdk"
    echo "    macOS:   brew install openjdk@17"
    echo "    Windows: https://adoptium.net/"
    echo "  or unpack a JDK 17 into tools/jdk17/"
    echo ""
    exit 1
fi

JAVA_CMD="${JAVA_HOME_DIR}/bin/java"
JAVA_VERSION="$("$JAVA_CMD" -version 2>&1 | grep -E '(java|openjdk) version' | head -n1 || true)"
echo "Java version: ${JAVA_VERSION}"

JAVA_MAJOR="$(echo "$JAVA_VERSION" | sed -E 's/.*version "([0-9]+)(\.([0-9]+))?.*/\1 \3/')"
read -r JAVA_MAJOR JAVA_MINOR <<< "$JAVA_MAJOR"
if [ "${JAVA_MAJOR:-0}" = "1" ]; then
    JAVA_MAJOR="${JAVA_MINOR:-0}"  # "1.8.0" style versions
fi
if ! [ "${JAVA_MAJOR:-0}" -ge 17 ] 2> /dev/null; then
    echo ""
    echo "ERROR: Java ${JAVA_MAJOR:-?} detected, but Java 17+ is required (clarity 4.x)."
    echo ""
    exit 1
fi
echo ""

# --- Build or reuse clarity-analyzer.jar --------------------------------------

case "${REBUILD:-0}" in
    1 | true | yes | on) REBUILD=1 ;;
    *) REBUILD=0 ;;
esac

if [ -f "$CLARITY_JAR" ] && [ "$REBUILD" != "1" ]; then
    echo "Using the existing JAR: $CLARITY_JAR"
    echo "(REBUILD=1 builds it from tools/clarity-analyzer/ instead; see the note at the top of this script.)"

elif [ -d "$CLARITY_SRC" ] && [ -f "$CLARITY_SRC/build.gradle" ]; then
    echo "Found clarity-analyzer source at: $CLARITY_SRC"

    if [ ! -x "${JAVA_HOME_DIR}/bin/javac" ]; then
        echo "ERROR: ${JAVA_HOME_DIR} is a JRE; building the parser needs a JDK (javac)."
        if [ -f "$CLARITY_JAR" ]; then
            echo "Keeping the existing prebuilt JAR: $CLARITY_JAR"
            exit 0
        fi
        exit 1
    fi

    echo "Building from source with Gradle..."
    if [ -n "${MAVEN_MIRROR:-}" ]; then
        echo "Using Maven mirror: ${MAVEN_MIRROR}"
    fi
    echo ""

    export JAVA_HOME="$JAVA_HOME_DIR"
    BUILD_OK=1
    (
        cd "$CLARITY_SRC"
        if [ -n "${GRADLE_CMD:-}" ]; then
            $GRADLE_CMD shadowJar --no-daemon
        elif [ -f "./gradlew" ]; then
            sh ./gradlew shadowJar --no-daemon
        else
            echo "ERROR: gradlew not found in $CLARITY_SRC (set GRADLE_CMD=gradle to use a system Gradle)"
            exit 1
        fi
    ) || BUILD_OK=0

    BUILT_JAR="${CLARITY_SRC}/build/libs/clarity-analyzer.jar"
    if [ "$BUILD_OK" = "1" ] && [ -f "$BUILT_JAR" ]; then
        # Replace atomically: a running backend may be starting a parse from the old JAR.
        cp "$BUILT_JAR" "${CLARITY_JAR}.new"
        mv -f "${CLARITY_JAR}.new" "$CLARITY_JAR"
        echo ""
        echo "Built and copied JAR to: $CLARITY_JAR"
    elif [ -f "$CLARITY_JAR" ]; then
        echo ""
        echo "WARNING: build failed; keeping the existing prebuilt JAR: $CLARITY_JAR"
    else
        echo ""
        echo "ERROR: build failed and no prebuilt JAR exists (expected ${BUILT_JAR})"
        exit 1
    fi

elif [ -f "$CLARITY_JAR" ]; then
    echo "No source to rebuild from; keeping the existing JAR: $CLARITY_JAR"

else
    echo "No clarity-analyzer source or JAR found."
    echo "Expected the Gradle project at: ${CLARITY_SRC}/ (build.gradle, gradlew, src/main/java/...)"
    exit 1
fi

# --- Verify the JAR starts (no arguments -> usage, exit code 64) --------------

set +e
USAGE="$("$JAVA_CMD" -jar "$CLARITY_JAR" 2>&1)"
STATUS=$?
set -e
if [ "$STATUS" != "64" ]; then
    echo "WARNING: the parser JAR did not start as expected (exit code ${STATUS})."
fi
# First line of the usage text: "clarity-analyzer <version>" (since 1.4.0).
JAR_VERSION="$(printf '%s\n' "$USAGE" | sed -n '1s/^clarity-analyzer \([0-9][0-9.]*\)$/\1/p')"

echo ""
echo "=== Setup Complete ==="
echo "  Parser JAR: ${CLARITY_JAR} (analyzer ${JAR_VERSION:-version unknown})"
echo "  Java:       ${JAVA_CMD} (${JAVA_VERSION})"
echo ""
echo "To test, run:"
echo "  \"${JAVA_CMD}\" -jar \"${CLARITY_JAR}\" <path-to-replay.dem> > raw.json"
echo "To check a parse against OpenDota's data for the same match:"
echo "  python3 \"${TOOLS_DIR}/compare_with_opendota.py\" raw.json <opendota-match.json>"
echo ""
echo "Backend knobs: CLARITY_JVM_XMX (JVM heap, default 1536m), CLARITY_TIMEOUT_S (default 300),"
echo "CLARITY_MAX_CONCURRENCY (parallel parses, default 1)."
