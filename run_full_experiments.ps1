param(
    [int]$Gpu = 0,
    [bool]$Resume = $true,
    [switch]$SkipDataPreparation,
    [switch]$SkipGpuVerification,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Python = if (Test-Path (Join-Path $Root ".venv\Scripts\python.exe")) {
    Join-Path $Root ".venv\Scripts\python.exe"
} else {
    "python"
}
$env:CUDA_VISIBLE_DEVICES = "$Gpu"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Invoke-Required([string]$Step, [scriptblock]$Command) {
    Write-Host "`n== $Step ==" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

Write-Host "DCDLP full experiment launcher" -ForegroundColor Green
Write-Host "Root: $Root"
Write-Host "Python: $Python"
Write-Host "Physical GPU: $Gpu"
Write-Host "Resume completed jobs: $Resume"

Invoke-Required "Checking Python dependencies" { & $Python -m pip check }
if (-not $SkipGpuVerification) {
    Invoke-Required "Checking CUDA and PyG GPU extensions" { & $Python scripts/verify_gpu.py }
}
Invoke-Required "Running unit and regression tests" { & $Python -m pytest -q }

$DataSets = @(
    @{ Name = "cora"; Protocol = "heart" },
    @{ Name = "citeseer"; Protocol = "heart" },
    @{ Name = "pubmed"; Protocol = "heart" },
    @{ Name = "ogbl-collab"; Protocol = "ogb" },
    @{ Name = "ogbl-ddi"; Protocol = "ogb" }
)
$MissingData = @()
foreach ($DataSet in $DataSets) {
    $Prepared = Join-Path $Root "data\processed\$($DataSet.Name)_$($DataSet.Protocol)_seed0.npz"
    if (Test-Path -LiteralPath $Prepared) {
        Write-Host "[DATA READY] $($DataSet.Name) / $($DataSet.Protocol)"
        continue
    }
    $MissingData += $DataSet.Name
    Write-Warning "[DATA MISSING] $Prepared"
    if (-not $PreflightOnly -and -not $SkipDataPreparation) {
        Invoke-Required "Preparing $($DataSet.Name) ($($DataSet.Protocol))" {
            & $Python -m dcdlp.cli prepare-data `
                --dataset $DataSet.Name `
                --protocol $DataSet.Protocol `
                --seed 0
        }
        if (-not (Test-Path -LiteralPath $Prepared)) {
            throw "Data preparation returned successfully but did not create $Prepared."
        }
    }
}

$Suites = @("paper_main", "paper_ablation", "paper_analysis")
if ($PreflightOnly) {
    Write-Host "`n== Planned suite jobs ==" -ForegroundColor Cyan
    foreach ($Suite in $Suites) {
        $Arguments = @("scripts/run_suite.py", "--suite", $Suite, "--gpus", "$Gpu", "--dry-run")
        if ($Resume) { $Arguments += "--resume" }
        & $Python @Arguments
        if ($LASTEXITCODE -ne 0) { throw "Dry-run failed for $Suite." }
    }
    if ($MissingData.Count -gt 0) {
        Write-Host "Preflight completed. A real run will prepare: $($MissingData -join ', ')." -ForegroundColor Yellow
    } else {
        Write-Host "Preflight passed; all required prepared datasets are present." -ForegroundColor Green
    }
    exit 0
}

if ($SkipDataPreparation -and $MissingData.Count -gt 0) {
    throw "Prepared data is missing for: $($MissingData -join ', '). Remove -SkipDataPreparation to build it automatically."
}

$FailedSuites = @()
foreach ($Suite in $Suites) {
    Write-Host "`n== Running suite: $Suite ==" -ForegroundColor Cyan
    $Arguments = @("scripts/run_suite.py", "--suite", $Suite, "--gpus", "$Gpu", "--continue-on-error")
    if ($Resume) { $Arguments += "--resume" }
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        $FailedSuites += $Suite
        Write-Warning "$Suite contains failed jobs. Remaining suites will still run; inspect results/logs."
    }
}

Invoke-Required "Aggregating completed results" { & $Python scripts/aggregate_results.py }
Invoke-Required "Generating paper figures" { & $Python scripts/make_paper_figures.py }

Write-Host "`nFull launcher finished." -ForegroundColor Green
Write-Host "Manifest: $Root\results\manifest.csv"
Write-Host "Per-job logs: $Root\results\logs"
Write-Host "Aggregates: $Root\results\aggregate"
Write-Host "Figures: $Root\results\figures"
if ($FailedSuites.Count -gt 0) {
    throw "Some jobs failed in: $($FailedSuites -join ', '). Completed jobs and logs were preserved; rerun the same command to resume."
}
