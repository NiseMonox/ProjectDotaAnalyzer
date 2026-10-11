<#
.SYNOPSIS
    Checks the Clarity replay parser (tools\clarity-analyzer.jar) on Windows.

.DESCRIPTION
    Windows counterpart of tools/setup_parser.sh: finds Java, checks that it is
    Java 17 or newer and that the committed JAR starts. Replay uploads need
    both; without them the server still fetches matches by ID.

        powershell -ExecutionPolicy Bypass -File tools\setup_parser.ps1
        powershell -ExecutionPolicy Bypass -File tools\setup_parser.ps1 -Rebuild

    Java is looked up in this order (the same as the server does):
      1. a project-local JDK unpacked into tools\jdk17 (tools\jdk17\bin\java.exe),
         optional and ignored by git
      2. %JAVA_HOME%
      3. java on PATH

    The committed JAR is used as it is: it is the same file on every operating
    system. -Rebuild (or REBUILD=1 in the environment) builds it from
    tools\clarity-analyzer with the Gradle wrapper instead, which is only needed
    after changing the Java source and needs a JDK (javac) and a download of
    Gradle and the libraries. The committed JAR was built with Eclipse Temurin
    17.0.13+11; another JDK gives a JAR that behaves the same but is not
    byte-identical, and git then shows tools\clarity-analyzer.jar as modified
    (undo with: git checkout -- tools/clarity-analyzer.jar).

    Environment for -Rebuild: MAVEN_MIRROR (an extra Maven repository tried
    first, e.g. https://maven.aliyun.com/repository/public), GRADLE_CMD (a
    Gradle to use instead of the wrapper, e.g. "gradle").

    Exit code 0 when the parser is ready, 1 otherwise.

    This file is plain ASCII on purpose: Windows PowerShell 5.1 reads a script
    without a byte order mark in the system code page.
#>
[CmdletBinding()]
param(
    [switch]$Rebuild
)

# "Continue", not "Stop": Windows PowerShell 5.1 turns every line a program
# writes to stderr into an error record once that stream is redirected, and
# "java -version" writes to stderr. Failures are checked by exit code below.
$ErrorActionPreference = "Continue"

$ToolsDir = $PSScriptRoot
$Jar = Join-Path $ToolsDir "clarity-analyzer.jar"
$Source = Join-Path $ToolsDir "clarity-analyzer"
$MinJava = 17

if (-not $Rebuild -and (@("1", "true", "yes", "on") -contains "$env:REBUILD".ToLower())) {
    $Rebuild = $true
}

function Invoke-Captured {
    # Runs a program; returns its exit code and its output (stdout and stderr) as lines.
    param([string]$Exe, [string[]]$Arguments = @())
    $lines = @()
    $code = -1
    try {
        $lines = @(& $Exe @Arguments 2>&1 | ForEach-Object { "$_" })
        $code = $LASTEXITCODE
    } catch {
        $lines = @("$_")
    }
    return [pscustomobject]@{ ExitCode = $code; Lines = $lines }
}

Write-Host "=== Dota 2 Replay Parser Setup ==="
Write-Host ""

# --- Locate Java 17+ ----------------------------------------------------------

$JavaExe = ""
$JavaFrom = ""
$bundled = Join-Path (Join-Path (Join-Path $ToolsDir "jdk17") "bin") "java.exe"
if (Test-Path -LiteralPath $bundled -PathType Leaf) {
    $JavaExe = $bundled
    $JavaFrom = "project-local JDK"
}
if (-not $JavaExe -and $env:JAVA_HOME) {
    $candidate = Join-Path (Join-Path $env:JAVA_HOME.Trim('"') "bin") "java.exe"
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $JavaExe = $candidate
        $JavaFrom = "JAVA_HOME"
    }
}
if (-not $JavaExe) {
    $command = Get-Command "java" -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) {
        $JavaExe = $command.Path
        $JavaFrom = "PATH"
    }
}

if (-not $JavaExe) {
    Write-Host "ERROR: Java not found. Java $MinJava or newer is required for the replay parser."
    Write-Host ""
    Write-Host "  Install a JDK 17, for example Eclipse Temurin: https://adoptium.net/"
    Write-Host "  (let the installer set JAVA_HOME and PATH, then open a new terminal),"
    Write-Host "  or unpack a JDK 17 zip so that tools\jdk17\bin\java.exe exists."
    Write-Host ""
    exit 1
}
Write-Host "Using Java from ${JavaFrom}: $JavaExe"

$probe = Invoke-Captured $JavaExe @("-version")
$versionLine = $probe.Lines | Where-Object { $_ -match 'version "' } | Select-Object -First 1
Write-Host "Java version: $versionLine"
$major = 0
if ("$versionLine" -match 'version "(\d+)(?:\.(\d+))?') {
    $major = [int]$Matches[1]
    if ($major -eq 1 -and $Matches[2]) {
        $major = [int]$Matches[2]    # "1.8.0" style versions
    }
}
if ($probe.ExitCode -ne 0 -or $major -eq 0) {
    Write-Host ""
    Write-Host "ERROR: '$JavaExe -version' did not answer with a version (exit code $($probe.ExitCode))."
    Write-Host ""
    exit 1
}
if ($major -lt $MinJava) {
    Write-Host ""
    Write-Host "ERROR: Java $major detected, but Java $MinJava or newer is required."
    Write-Host ""
    exit 1
}
Write-Host ""

# --- Build or reuse clarity-analyzer.jar ---------------------------------------

$hasJar = Test-Path -LiteralPath $Jar -PathType Leaf
if ($hasJar -and -not $Rebuild) {
    Write-Host "Using the existing JAR: $Jar"
    Write-Host "(-Rebuild builds it from tools\clarity-analyzer instead; see the note at the top of this script.)"
} elseif (Test-Path -LiteralPath (Join-Path $Source "build.gradle") -PathType Leaf) {
    Write-Host "Found clarity-analyzer source at: $Source"
    $javaHome = Split-Path -Parent (Split-Path -Parent $JavaExe)
    if (-not (Test-Path -LiteralPath (Join-Path (Join-Path $javaHome "bin") "javac.exe") -PathType Leaf)) {
        Write-Host "ERROR: $javaHome has no javac.exe; building the parser needs a JDK, not a JRE."
        if ($hasJar) {
            Write-Host "Keeping the existing prebuilt JAR: $Jar"
        }
        exit 1
    }
    Write-Host "Building from source with Gradle..."
    if ($env:MAVEN_MIRROR) {
        Write-Host "Using Maven mirror: $env:MAVEN_MIRROR"
    }
    Write-Host ""
    $env:JAVA_HOME = $javaHome
    $gradle = Join-Path $Source "gradlew.bat"
    if ($env:GRADLE_CMD) {
        $gradle = $env:GRADLE_CMD
    }
    Push-Location -LiteralPath $Source
    try {
        & $gradle shadowJar --no-daemon
        $buildCode = $LASTEXITCODE
    } catch {
        Write-Host "ERROR: could not run ${gradle}: $_"
        $buildCode = 1
    } finally {
        Pop-Location
    }
    $built = Join-Path (Join-Path (Join-Path $Source "build") "libs") "clarity-analyzer.jar"
    if ($buildCode -eq 0 -and (Test-Path -LiteralPath $built -PathType Leaf)) {
        # A running server has the old JAR open while it parses, and Windows
        # does not replace an open file: stop the server first if this fails.
        Copy-Item -LiteralPath $built -Destination $Jar -Force -ErrorAction SilentlyContinue
        if (-not $?) {
            Write-Host "ERROR: could not replace $Jar (is the server parsing a replay right now?)"
            exit 1
        }
        Write-Host ""
        Write-Host "Built and copied JAR to: $Jar"
    } elseif ($hasJar) {
        Write-Host ""
        Write-Host "WARNING: build failed; keeping the existing prebuilt JAR: $Jar"
    } else {
        Write-Host ""
        Write-Host "ERROR: build failed and no prebuilt JAR exists (expected $built)"
        exit 1
    }
} elseif ($hasJar) {
    Write-Host "No source to rebuild from; keeping the existing JAR: $Jar"
} else {
    Write-Host "No clarity-analyzer source or JAR found."
    Write-Host "Expected the Gradle project at: $Source (build.gradle, gradlew.bat, src\main\java\...)"
    exit 1
}

# --- Verify the JAR starts (no arguments -> usage, exit code 64) ----------------

$usage = Invoke-Captured $JavaExe @("-jar", $Jar)
$ready = $usage.ExitCode -eq 64
if (-not $ready) {
    Write-Host "WARNING: the parser JAR did not start as expected (exit code $($usage.ExitCode))."
    $usage.Lines | Select-Object -First 5 | ForEach-Object { Write-Host "  $_" }
}
# The usage text starts with "clarity-analyzer <version>" (since 1.4.0).
$jarVersion = "version unknown"
foreach ($line in $usage.Lines) {
    if ($line -match '^clarity-analyzer ([0-9][0-9.]*)\s*$') {
        $jarVersion = $Matches[1]
        break
    }
}

Write-Host ""
Write-Host "=== Setup Complete ==="
Write-Host "  Parser JAR: $Jar (analyzer $jarVersion)"
Write-Host "  Java:       $JavaExe ($versionLine)"
Write-Host ""
Write-Host "To test, run this in cmd.exe (not in Windows PowerShell 5.1: its > re-encodes the JSON as UTF-16):"
Write-Host "  `"$JavaExe`" -jar `"$Jar`" <path-to-replay.dem> > raw.json"
Write-Host "To check a parse against OpenDota's data for the same match:"
Write-Host "  python `"$(Join-Path $ToolsDir 'compare_with_opendota.py')`" raw.json <opendota-match.json>"
Write-Host ""
Write-Host "Backend knobs: CLARITY_JVM_XMX (JVM heap, default 1536m), CLARITY_TIMEOUT_S (default 300),"
Write-Host "CLARITY_MAX_CONCURRENCY (parallel parses, default 1)."
if (-not $ready) {
    exit 1
}
exit 0
