<#
.SYNOPSIS
    Runs the Dota 2 Match Analyzer on Windows (counterpart of run.sh).

.DESCRIPTION
        .\run.ps1                                              # http://localhost:8000
        $env:DEV = "1"; .\run.ps1                              # auto-reload on code changes
        $env:HOST = "127.0.0.1"; $env:PORT = "8001"; .\run.ps1

    When Windows answers "running scripts is disabled on this system", start
    it this way (nothing is changed on the computer):

        powershell -ExecutionPolicy Bypass -File run.ps1

    or allow local scripts for your account once and use .\run.ps1 from then on:

        Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

    Creates backend\venv on the first run (Python 3.10-3.13, found through the
    py launcher or as python; $env:PYTHON selects another interpreter) and
    reinstalls the dependencies whenever backend\requirements.txt changes.
    Settings come from the environment or .env (see .env.example); the
    environment wins. This script reads no setting itself: it asks the
    application, which loads .env, so HOST, PORT, DEV, JAVA_PATH and
    CLARITY_JAR_PATH mean here what they mean to the server. Optional extras
    (backend\requirements-rag.txt) are never installed here.

    Works with Windows PowerShell 5.1 and PowerShell 7. This file is plain
    ASCII on purpose: Windows PowerShell 5.1 reads a script without a byte
    order mark in the system code page.
#>

# "Continue", not "Stop": Windows PowerShell 5.1 turns every line a program
# writes to stderr into an error record once that stream is redirected or
# captured by the host, and pip and the server (all of its log lines) write
# there. Failures are checked by exit code below.
$ErrorActionPreference = "Continue"

$ProjectDir = $PSScriptRoot
$BackendDir = Join-Path $ProjectDir "backend"
$VenvDir = Join-Path $BackendDir "venv"
$VenvPython = Join-Path (Join-Path $VenvDir "Scripts") "python.exe"
$Requirements = Join-Path $BackendDir "requirements.txt"
$Stamp = Join-Path $VenvDir ".requirements.sha256"
# The pinned dependencies have wheels for these versions only; with another
# one pip fails halfway with an error about a missing package.
$SupportedPython = @("3.10", "3.11", "3.12", "3.13")

function Get-PythonVersion {
    # "3.12" for an interpreter that runs, "" otherwise: not installed, or the
    # Microsoft Store placeholder named python.exe that only prints a hint.
    param([string]$Exe, [string[]]$Prefix = @())
    try {
        $out = & $Exe @Prefix -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) {
            return "$(@($out)[-1])".Trim()
        }
    } catch {
    }
    return ""
}

function New-PythonCandidate {
    param([string]$Exe, [string[]]$Prefix = @())
    return [pscustomobject]@{ Exe = $Exe; Prefix = $Prefix }
}

if (-not (Test-Path -LiteralPath $Requirements -PathType Leaf)) {
    Write-Host "ERROR: $Requirements not found. Run this script from the project it belongs to."
    exit 1
}

# --- Virtual environment ------------------------------------------------------

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    $candidates = @()
    if ($env:PYTHON) {
        $candidates += New-PythonCandidate $env:PYTHON
    } else {
        # 3.12 first: the version the project is developed and tested on.
        foreach ($version in @("3.12", "3.13", "3.11", "3.10")) {
            $candidates += New-PythonCandidate "py" @("-$version")
        }
        $candidates += New-PythonCandidate "python"
        $candidates += New-PythonCandidate "python3"
    }
    $python = $null
    $rejected = @()
    foreach ($candidate in $candidates) {
        $found = Get-PythonVersion $candidate.Exe $candidate.Prefix
        if (-not $found) {
            continue
        }
        if ($SupportedPython -contains $found) {
            $python = $candidate
            break
        }
        $rejected += "$($candidate.Exe) $($candidate.Prefix -join ' ')".Trim() + " is Python $found"
    }
    if (-not $python) {
        if ($env:PYTHON -and -not $rejected) {
            Write-Host "ERROR: cannot run '$env:PYTHON'. Set PYTHON to a Python 3.10-3.13 interpreter."
        } else {
            Write-Host "ERROR: no Python 3.10-3.13 found (looked for the py launcher, python and python3)."
        }
        foreach ($line in $rejected) {
            Write-Host "       $line, this project needs Python 3.10-3.13."
        }
        Write-Host "       Install Python 3.12 from https://www.python.org/downloads/windows/ or the"
        Write-Host "       Microsoft Store, open a new terminal and run this script again. To use an"
        Write-Host "       interpreter in another place: `$env:PYTHON = 'C:\path\to\python.exe'"
        exit 1
    }
    Write-Host "Creating virtual environment in $VenvDir ..."
    $prefix = $python.Prefix
    & $python.Exe @prefix -m venv $VenvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        Write-Host "ERROR: could not create the virtual environment in $VenvDir."
        exit 1
    }
    if (Test-Path -LiteralPath $Stamp) {
        Remove-Item -LiteralPath $Stamp -Force
    }
} elseif (-not (Get-PythonVersion $VenvPython)) {
    Write-Host "ERROR: $VenvPython does not run (was the Python it was created with removed or updated?)."
    Write-Host "       Delete the folder $VenvDir and run this script again."
    exit 1
}

# --- Dependencies -------------------------------------------------------------

$requirementsHash = (Get-FileHash -LiteralPath $Requirements -Algorithm SHA256).Hash.ToLower()
$installedHash = ""
if (Test-Path -LiteralPath $Stamp -PathType Leaf) {
    $installedHash = "$(Get-Content -LiteralPath $Stamp -TotalCount 1)".Trim()
}
if ($installedHash -ne $requirementsHash) {
    Write-Host "Installing dependencies from backend\requirements.txt ..."
    & $VenvPython -m pip --version *> $null
    if ($LASTEXITCODE -ne 0) {
        & $VenvPython -m ensurepip --upgrade *> $null
    }
    & $VenvPython -m pip install --upgrade pip *> $null
    & $VenvPython -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: installing the dependencies failed (see the pip output above)."
        exit 1
    }
    Set-Content -LiteralPath $Stamp -Value $requirementsHash -Encoding ASCII
}

# --- Settings, as the application sees them -------------------------------------

Push-Location -LiteralPath $ProjectDir
$exitCode = 1
try {
    # One JSON line, ASCII only (json.dumps escapes everything else), so the
    # console code page cannot damage it on the way here.
    $probe = "import json, shutil; from backend.app import config; print(json.dumps(dict(host=config.HOST, port=config.PORT, dev=bool(config.DEV), java=bool(shutil.which(config.JAVA_PATH)), jar=config.CLARITY_JAR_PATH.is_file(), jar_path=str(config.CLARITY_JAR_PATH))))"
    $settings = $null
    try {
        $answer = & $VenvPython -c $probe 2>$null
        if ($LASTEXITCODE -eq 0 -and $answer) {
            $settings = "$(@($answer)[-1])" | ConvertFrom-Json
        }
    } catch {
        $settings = $null
    }
    if (-not $settings) {
        Write-Host "ERROR: the application's settings could not be read. This shows why:"
        Write-Host "       `"$VenvPython`" -c `"from backend.app import config`""
        exit 1
    }

    # Replay parsing needs Java 17+ and the clarity jar; without them uploads
    # return 503 and only "fetch by match ID" (OpenDota) works.
    if (-not $settings.java) {
        Write-Host "WARNING: Java 17+ not found - replay upload/parsing is disabled (match-ID fetch still works)."
        Write-Host "         Install a JDK 17 (https://adoptium.net/) or unpack one into tools\jdk17, then check"
        Write-Host "         with: powershell -ExecutionPolicy Bypass -File tools\setup_parser.ps1"
    }
    if (-not $settings.jar) {
        Write-Host "WARNING: $($settings.jar_path) not found - replay upload/parsing is disabled."
        Write-Host "         It is part of the repository: git checkout -- tools/clarity-analyzer.jar"
    }

    $serverArgs = @("-m", "uvicorn", "backend.app.main:app", "--host", "$($settings.host)", "--port", "$($settings.port)")
    if ($settings.dev) {
        $serverArgs += @("--reload", "--reload-dir", (Join-Path $BackendDir "app"))
    }

    $displayHost = "$($settings.host)"
    if ($displayHost -eq "0.0.0.0") {
        $displayHost = "localhost"
    }
    Write-Host ""
    Write-Host "=== Dota 2 Match Analyzer ==="
    Write-Host "  URL: http://${displayHost}:$($settings.port)"
    Write-Host "  API: http://${displayHost}:$($settings.port)/api/demo"
    if ($settings.dev) {
        Write-Host "  Auto-reload: on (DEV=1)"
    }
    Write-Host ""
    Write-Host "  Press Ctrl+C to stop"
    Write-Host ""

    & $VenvPython @serverArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $exitCode
