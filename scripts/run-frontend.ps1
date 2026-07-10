[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$BackendPort = 8000,

    [ValidateRange(1, 65535)]
    [int]$Port = 5173
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$FrontendRoot = Join-Path $ProjectRoot "frontend"
$Npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $Npm) {
    throw "npm.cmd was not found. Install Node.js 22 or newer."
}
if (-not (Test-Path (Join-Path $FrontendRoot "node_modules"))) {
    throw "Frontend dependencies not found. Run scripts\setup.ps1 first."
}

$env:VITE_API_BASE = "http://127.0.0.1:$BackendPort"
Set-Location $FrontendRoot
& $Npm.Source run dev -- --port $Port
