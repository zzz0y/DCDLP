#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${DCDLP_PYTHON:-python}"
GPUS="${DCDLP_GPUS:-0}"
RESUME="${DCDLP_RESUME:-true}"
PREFLIGHT_ONLY="${DCDLP_PREFLIGHT_ONLY:-false}"
SKIP_GPU_VERIFY="${DCDLP_SKIP_GPU_VERIFY:-false}"
JOBS="${DCDLP_JOBS:-4}"
LARGE_JOBS="${DCDLP_LARGE_JOBS:-1}"
THREADS_PER_JOB="${DCDLP_THREADS_PER_JOB:-8}"
LARGE_BATCH_SIZE="${DCDLP_LARGE_BATCH_SIZE:-8192}"
LARGE_MAX_INTERVENTIONS_PER_BATCH="${DCDLP_LARGE_MAX_INTERVENTIONS_PER_BATCH:-1}"
LARGE_MAX_INTERVENTIONS_PER_EPOCH="${DCDLP_LARGE_MAX_INTERVENTIONS_PER_EPOCH:-256}"
RUN_MODE="${DCDLP_RUN_MODE:-full}"

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export CUDA_VISIBLE_DEVICES="$GPUS"

mkdir -p results/logs results/raw results/checkpoints results/aggregate results/figures results/tables
LOG_FILE="results/logs/container_launcher.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "DCDLP offline full experiment launcher"
echo "Root: $ROOT"
echo "Python: $PYTHON_BIN"
echo "Visible GPUs: $GPUS"
echo "Resume completed jobs: $RESUME"
echo "Concurrent small/medium jobs: $JOBS"
echo "Concurrent OGB jobs: $LARGE_JOBS"
echo "CPU math threads per job: $THREADS_PER_JOB"
echo "OGB batch size: $LARGE_BATCH_SIZE"
echo "OGB max interventions per batch: $LARGE_MAX_INTERVENTIONS_PER_BATCH"
echo "OGB max interventions per epoch: $LARGE_MAX_INTERVENTIONS_PER_EPOCH"
echo "Run mode: $RUN_MODE"

"$PYTHON_BIN" -m pip check
if [[ "$SKIP_GPU_VERIFY" != "true" ]]; then
  "$PYTHON_BIN" scripts/verify_gpu.py
fi
"$PYTHON_BIN" -m pytest -q

if [[ "$RUN_MODE" == "pilot" || "$RUN_MODE" == "phase2" ]]; then
  required_data=(
    "data/processed/cora_heart_seed0.npz"
  )
else
  required_data=(
    "data/processed/cora_heart_seed0.npz"
    "data/processed/citeseer_heart_seed0.npz"
    "data/processed/pubmed_heart_seed0.npz"
    "data/processed/ogbl-collab_ogb_seed0.npz"
    "data/processed/ogbl-ddi_ogb_seed0.npz"
  )
fi
missing_data=()
for path in "${required_data[@]}"; do
  if [[ -f "$path" ]]; then
    echo "[DATA READY] $path"
  else
    missing_data+=("$path")
  fi
done

if (( ${#missing_data[@]} > 0 )); then
  printf 'Offline deployment is missing required data:\n'
  printf '  %s\n' "${missing_data[@]}"
  exit 2
fi

resume_args=()
if [[ "$RESUME" == "true" ]]; then
  resume_args+=(--resume)
fi

suites=(paper_main paper_ablation paper_analysis)
suite_filter_args=()
if [[ "$RUN_MODE" == "pilot" ]]; then
  suites=(paper_ablation)
  suite_filter_args=(
    --datasets cora,citeseer
    --models A0,A5,A14
    --seeds 0,1,2
    --protocols uniform
  )
elif [[ "$RUN_MODE" == "phase2" ]]; then
  # Keep the raw, residual and no-interaction runs as separate suite
  # identities so they cannot collide with legacy manifest entries.
  suites=(phase2_a5_raw phase2_a5_residual phase2_a5_no_interaction)
  suite_filter_args=()
elif [[ "$RUN_MODE" != "full" ]]; then
  echo "DCDLP_RUN_MODE must be 'pilot', 'phase2' or 'full', got: $RUN_MODE" >&2
  exit 2
fi
if [[ "$PREFLIGHT_ONLY" == "true" ]]; then
  for suite in "${suites[@]}"; do
    "$PYTHON_BIN" scripts/run_suite.py --suite "$suite" --gpus "$GPUS" \
      --dry-run "${suite_filter_args[@]}" "${resume_args[@]}"
  done
  echo "Preflight passed; no experiment jobs were launched."
  exit 0
fi

failed_suites=()
for suite in "${suites[@]}"; do
  echo "== Running suite: $suite =="
  if ! "$PYTHON_BIN" scripts/run_suite.py \
      --suite "$suite" --gpus "$GPUS" --jobs "$JOBS" --large-jobs "$LARGE_JOBS" \
      --threads-per-job "$THREADS_PER_JOB" --large-batch-size "$LARGE_BATCH_SIZE" \
      --large-max-interventions-per-batch "$LARGE_MAX_INTERVENTIONS_PER_BATCH" \
      --large-max-interventions-per-epoch "$LARGE_MAX_INTERVENTIONS_PER_EPOCH" \
      --continue-on-error "${suite_filter_args[@]}" "${resume_args[@]}"; then
    failed_suites+=("$suite")
    echo "[WARNING] $suite contains failed jobs; continuing with the remaining suites."
  fi
done

"$PYTHON_BIN" scripts/aggregate_results.py
if [[ "$RUN_MODE" == "pilot" ]]; then
  "$PYTHON_BIN" scripts/summarize_pilot.py
elif [[ "$RUN_MODE" == "phase2" ]]; then
  "$PYTHON_BIN" scripts/summarize_phase2.py
else
  "$PYTHON_BIN" scripts/make_paper_figures.py
fi

echo "Full launcher finished."
echo "Manifest: $ROOT/results/manifest.csv"
echo "Per-job logs: $ROOT/results/logs"
echo "Aggregates: $ROOT/results/aggregate"
echo "Figures: $ROOT/results/figures"
if [[ "$RUN_MODE" == "pilot" ]]; then
  echo "Pilot summary: $ROOT/results/aggregate/pilot_summary.csv"
  echo "To continue with every experiment, restart with DCDLP_RUN_MODE=full."
elif [[ "$RUN_MODE" == "phase2" ]]; then
  echo "Phase-2 summary: $ROOT/results/aggregate/phase2_summary.csv"
fi

if (( ${#failed_suites[@]} > 0 )); then
  printf 'Some suites contain failed jobs: %s\n' "${failed_suites[*]}"
  echo "Completed jobs were preserved; restart the same container command to resume."
  exit 1
fi
