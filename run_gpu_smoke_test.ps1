param(
    [int]$Gpu = 0,
    [int]$Seed = 2026
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

function Assert-LastExitCode([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

Write-Host "[1/3] Verifying CUDA, autograd, and PyG CUDA extensions..."
& $Python scripts/verify_gpu.py
Assert-LastExitCode "GPU environment verification"

Write-Host "[2/3] Running the unit and regression tests..."
& $Python -m pytest -q
Assert-LastExitCode "Test suite"

Write-Host "[3/3] Running a two-stage DCDLP smoke training job on the GPU..."
& $Python -m dcdlp.cli train `
    --dataset smoke `
    --protocol standard `
    --seed $Seed `
    --protocol-train uniform `
    --pretrain-epochs 1 `
    --disentangle-epochs 1 `
    --hidden-dim 16 `
    --branch-dim 8 `
    --batch-size 128 `
    --intervention-ratio 0.05 `
    --ablation A14 `
    --device cuda
Assert-LastExitCode "DCDLP GPU smoke training"

Write-Host "GPU verification passed. Check results/raw and results/checkpoints for the generated artifacts."
