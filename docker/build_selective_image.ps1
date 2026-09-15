[CmdletBinding()]
param(
    [string]$Tag = 'dcdlp:selective-2.3.1-cu121',
    [string]$Archive = 'offline/dcdlp-selective-image.tar'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'docker.exe is not available. Install Docker Desktop before building.'
}

Write-Host "[build] Building $Tag from the selective Docker context..."
& docker build --pull=false --file docker/Dockerfile.selective --tag $Tag .
if ($LASTEXITCODE -ne 0) { throw 'Docker image build failed.' }

$archivePath = Join-Path $root $Archive
$archiveDir = Split-Path -Parent $archivePath
New-Item -ItemType Directory -Force -Path $archiveDir | Out-Null
if (Test-Path -LiteralPath $archivePath) {
    Write-Host "[build] Refusing to overwrite existing archive: $archivePath"
    throw 'Choose a new -Archive path or move the existing archive first.'
}

Write-Host "[build] Saving image to $archivePath ..."
& docker save --output $archivePath $Tag
if ($LASTEXITCODE -ne 0) { throw 'Could not export the Docker image.' }

$shaPath = "$archivePath.sha256"
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
"$hash  $(Split-Path -Leaf $archivePath)" | Set-Content -Encoding ASCII -LiteralPath $shaPath
Write-Host "[build] SHA256: $hash"
Write-Host '[build] Upload the .tar, .sha256, docker/start_selective_server.sh, and this repository (or the generated server bundle) to the Linux server.'
