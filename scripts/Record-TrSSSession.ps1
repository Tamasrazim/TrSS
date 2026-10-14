[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Game,

    [ValidateRange(0, 255)]
    [int]$Adapter = 0,

    [ValidateRange(0, 255)]
    [int]$Monitor = 0,

    [ValidateRange(1, 600)]
    [int]$Seconds = 60,

    [ValidateRange(1, 60)]
    [int]$SamplesPerSecond = 2,

    [ValidateRange(1, 30)]
    [int]$CountdownSeconds = 8,

    [string]$OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Find the packaged capture tool first, then the usual build output.
$searchPaths = @(
    (Join-Path $PSScriptRoot "trss_desktop_capture.exe"),
    (Join-Path $PSScriptRoot "..\build\directx12\Release\trss_desktop_capture.exe"),
    (Join-Path (Get-Location).Path "trss_desktop_capture.exe"),
    (Join-Path (Get-Location).Path "build\directx12\Release\trss_desktop_capture.exe")
)
$CaptureExe = $searchPaths |
    Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
    Select-Object -First 1

if (-not $CaptureExe) {
    throw "trss_desktop_capture.exe was not found. Extract the TrSS Windows capture package and keep this script beside the EXE."
}
if ($Game.Length -gt 100) {
    throw "Game labels must be no longer than 100 characters."
}

# Show available outputs, because Windows adapter and monitor indexes can change.
Write-Host ""
Write-Host "Available adapters and monitors:"
& $CaptureExe --list-devices
if ($LASTEXITCODE -ne 0) {
    throw "Could not list desktop capture devices."
}

if (-not $OutputDirectory) {
    $safeGame = ($Game -replace '[^a-zA-Z0-9]+', '-').Trim('-').ToLowerInvariant()
    if (-not $safeGame) {
        $safeGame = "game"
    }
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputDirectory = Join-Path (Get-Location).Path "captures\$safeGame-$stamp"
}

$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $OutputDirectory) {
    throw "Output already exists and will not be overwritten: $OutputDirectory"
}

$maxFrames = $Seconds * $SamplesPerSecond
Write-Host ""
Write-Host "Game label : $Game (typed by you; not detected from the game process)"
Write-Host "Adapter    : $Adapter"
Write-Host "Monitor    : $Monitor"
Write-Host "Duration   : $Seconds seconds"
Write-Host "Sample rate: up to $SamplesPerSecond saved frames per second"
Write-Host "Frame cap  : $maxFrames"
Write-Host "Output     : $OutputDirectory"
Write-Host ""
Write-Host "Start the game and make sure its picture is visible on the selected monitor."
[void](Read-Host "Press Enter, then switch back to gameplay during the countdown")
for ($left = $CountdownSeconds; $left -gt 0; $left--) {
    Write-Host "Capture starts in $left..."
    Start-Sleep -Seconds 1
}

$captureArgs = @(
    "--adapter", [string]$Adapter,
    "--monitor", [string]$Monitor,
    "--seconds", [string]$Seconds,
    "--fps", [string]$SamplesPerSecond,
    "--max-frames", [string]$maxFrames,
    "--game", $Game,
    "--output-dir", $OutputDirectory
)
& $CaptureExe @captureArgs
if ($LASTEXITCODE -ne 0) {
    throw "Capture failed with exit code $LASTEXITCODE. See the error above."
}

$sessionPath = Join-Path $OutputDirectory "session.json"
$session = Get-Content -LiteralPath $sessionPath -Raw | ConvertFrom-Json
Write-Host ""
Write-Host "Capture complete."
Write-Host "Frames saved : $($session.frames_saved)"
Write-Host "Actual output: $([math]::Round($session.measured_output_fps, 2)) saved frames/sec"
Write-Host "Session JSON : $sessionPath"
Write-Host "Frame index  : $(Join-Path $OutputDirectory 'capture-manifest.jsonl')"
Write-Host "Frame images : $(Join-Path $OutputDirectory 'frames')"
Write-Host ""
Write-Host "Run this script again from another stretch of gameplay to make a separate validation session."
