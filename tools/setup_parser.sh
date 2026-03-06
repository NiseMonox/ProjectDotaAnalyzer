#!/bin/bash
# Setup script for the Clarity replay parser
# Clarity is a Java library for parsing Dota 2 replay files

set -e

TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$TOOLS_DIR")"

echo "=== Dota 2 Replay Parser Setup ==="
echo ""

# Check Java
if ! command -v java &> /dev/null; then
    echo "ERROR: Java not found. Please install Java 11+."
    echo ""
    echo "  macOS:   brew install openjdk@17"
    echo "  Ubuntu:  sudo apt install openjdk-17-jdk"
    echo "  Windows: Download from https://adoptium.net/"
    echo ""
    exit 1
fi

JAVA_VERSION=$(java -version 2>&1 | head -n1)
echo "Found Java: $JAVA_VERSION"
echo ""

# Download clarity
CLARITY_VERSION="2.7.9"
CLARITY_JAR="clarity-analyzer.jar"
CLARITY_URL="https://repo1.maven.org/maven2/com/skadistats/clarity-examples/${CLARITY_VERSION}/clarity-examples-${CLARITY_VERSION}.jar"

echo "Downloading clarity v${CLARITY_VERSION}..."

if command -v curl &> /dev/null; then
    curl -L -o "${TOOLS_DIR}/${CLARITY_JAR}" "$CLARITY_URL"
elif command -v wget &> /dev/null; then
    wget -O "${TOOLS_DIR}/${CLARITY_JAR}" "$CLARITY_URL"
else
    echo "ERROR: Neither curl nor wget found. Please download manually:"
    echo "  URL: $CLARITY_URL"
    echo "  Save to: ${TOOLS_DIR}/${CLARITY_JAR}"
    exit 1
fi

echo ""
echo "Setup complete!"
echo "  Parser JAR: ${TOOLS_DIR}/${CLARITY_JAR}"
echo ""
echo "NOTE: The clarity-examples JAR includes built-in processors."
echo "For full custom analysis, you may need to build a custom Java extractor."
echo "See: https://github.com/skadistats/clarity"
