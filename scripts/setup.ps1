[CmdletBinding()]
param(
    [switch]$SkipFrontend
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    $PyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    $Python = Get-Command python.exe -ErrorAction SilentlyContinue

    if ($PyLauncher) {
        & $PyLauncher.Source -3 -m venv (Join-Path $ProjectRoot ".venv")
    }
    elseif ($Python) {
        & $Python.Source -m venv (Join-Path $ProjectRoot ".venv")
    }
    else {
        throw "Python 3 was not found. Install Python 3.11 or newer and rerun this script."
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the Python virtual environment."
    }
}

& $VenvPython -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "MathRAG requires Python 3.11 or newer. Delete .venv and recreate it with a supported Python version."
}

& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip."
}

& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements-dev.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install Python dependencies."
}

if (-not $SkipFrontend) {
    $Npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    $Node = Get-Command node.exe -ErrorAction SilentlyContinue
    if (-not $Npm -or -not $Node) {
        throw "npm.cmd was not found. Install Node.js 22 or newer and rerun this script."
    }

    $NodeMajor = & $Node.Source -p "Number(process.versions.node.split('.')[0])"
    if ($LASTEXITCODE -ne 0 -or [int]$NodeMajor -lt 22) {
        throw "MathRAG requires Node.js 22 or newer."
    }

    Push-Location (Join-Path $ProjectRoot "frontend")
    try {
        & $Npm.Source ci
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install frontend dependencies."
        }
    }
    finally {
        Pop-Location
    }
}

$EnvPath = Join-Path $ProjectRoot ".env"
if (-not (Test-Path $EnvPath)) {
    Copy-Item (Join-Path $ProjectRoot ".env.example") $EnvPath
    Write-Warning "Created .env from .env.example. Replace the placeholder DeepSeek API key before asking questions."
}

Write-Host "MathRAG setup completed." -ForegroundColor Green
Write-Host "Next: powershell -ExecutionPolicy Bypass -File .\scripts\start-dev.ps1"
