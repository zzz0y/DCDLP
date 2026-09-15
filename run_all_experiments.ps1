param(
    [string]$Gpus = "0",
    [switch]$SkipDownload
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Python = if (Test-Path (Join-Path $Root ".venv\Scripts\python.exe")) { Join-Path $Root ".venv\Scripts\python.exe" } else { "python" }
if (-not $SkipDownload) {
    & $Python scripts/download_data.py --third-party --datasets cora citeseer pubmed ogbl-collab ogbl-ddi
}
& $Python -m pytest -q
& $Python scripts/run_suite.py --suite paper_main --gpus $Gpus --continue-on-error
& $Python scripts/run_suite.py --suite paper_ablation --gpus $Gpus --continue-on-error
& $Python scripts/run_suite.py --suite paper_analysis --gpus $Gpus --continue-on-error
& $Python scripts/aggregate_results.py
& $Python scripts/make_paper_figures.py
Write-Host "Experiment launcher finished. Inspect results/manifest.csv; NOT_RUN/FAILED entries are never converted to scores."
