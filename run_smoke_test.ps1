$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Python = if (Test-Path (Join-Path $Root ".venv\Scripts\python.exe")) { Join-Path $Root ".venv\Scripts\python.exe" } else { "python" }
& $Python -m pytest -q
& $Python scripts/run_suite.py --suite smoke
& $Python scripts/aggregate_results.py
& $Python scripts/make_paper_figures.py
Write-Host "Smoke suite completed. See results/manifest.csv and results/raw/."
