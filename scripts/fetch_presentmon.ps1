$ErrorActionPreference = "Stop"

# Download only the pinned official Windows x64 release and verify its published digest.
$version = "2.6.0"
$fileName = "PresentMon-$version-x64.exe"
$url = "https://github.com/GameTechDev/PresentMon/releases/download/v$version/$fileName"
$expectedSha256 = "b2a706bc6ad475749e3b7e3409263aa1e6906d45bdcf993f6dbc0f660188f1af"

$repoRoot = Split-Path -Parent $PSScriptRoot
$destinationDirectory = Join-Path $repoRoot "vendor\presentmon"
$destination = Join-Path $destinationDirectory $fileName
New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null

Write-Host "Downloading the pinned PresentMon $version frame telemetry runtime..."
Invoke-WebRequest -Uri $url -OutFile $destination

$actualSha256 = (Get-FileHash -Path $destination -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualSha256 -ne $expectedSha256) {
    Remove-Item -LiteralPath $destination -Force -ErrorAction SilentlyContinue
    throw "PresentMon SHA-256 verification failed. The downloaded file has been removed."
}

Write-Host "PresentMon $version is ready. SHA-256 verified."
