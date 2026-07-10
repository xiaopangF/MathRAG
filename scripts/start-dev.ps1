[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$BackendPort = 8000,

    [ValidateRange(1, 65535)]
    [int]$FrontendPort = 5173
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-PortAvailable {
    param([int]$Port)

    $Listener = [System.Net.Sockets.TcpListener]::new(
        [System.Net.IPAddress]::Loopback,
        $Port
    )
    try {
        $Listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        $Listener.Stop()
    }
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run scripts\setup.ps1 first."
}
if (-not (Test-Path (Join-Path $ProjectRoot "frontend\node_modules"))) {
    throw "Frontend dependencies not found. Run scripts\setup.ps1 first."
}
if (-not (Test-PortAvailable $BackendPort)) {
    throw "Backend port $BackendPort is already in use. Pass -BackendPort with another port."
}
if (-not (Test-PortAvailable $FrontendPort)) {
    throw "Frontend port $FrontendPort is already in use. Pass -FrontendPort with another port."
}

$PowerShell = (Get-Command powershell.exe).Source
$BackendScript = Join-Path $PSScriptRoot "run-backend.ps1"
$FrontendScript = Join-Path $PSScriptRoot "run-frontend.ps1"

$BackendProcess = Start-Process -FilePath $PowerShell -ArgumentList @(
    "-NoExit",
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $BackendScript,
    "-Port", $BackendPort
) -WorkingDirectory $ProjectRoot -PassThru

$FrontendProcess = Start-Process -FilePath $PowerShell -ArgumentList @(
    "-NoExit",
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $FrontendScript,
    "-BackendPort", $BackendPort,
    "-Port", $FrontendPort
) -WorkingDirectory $ProjectRoot -PassThru

Write-Host "MathRAG development services started." -ForegroundColor Green
Write-Host "Frontend: http://127.0.0.1:$FrontendPort"
Write-Host "Backend:  http://127.0.0.1:$BackendPort/docs"
Write-Host "Backend PID: $($BackendProcess.Id); frontend PID: $($FrontendProcess.Id)"
Write-Host "Close the two service windows to stop the development servers."
