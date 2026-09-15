#!/usr/bin/env bash
set -Eeuo pipefail

# Resolve the mounted project from the script location so a phase-2 run can
# use a fresh remote directory instead of mixing with the legacy tree.
ROOT="${DCDLP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="dcdlp:2.3.1-cu121"
CONTAINER="${DCDLP_CONTAINER:-dcdlp-experiment}"
GPUS="${DCDLP_GPUS:-0}"
JOBS="${DCDLP_JOBS:-4}"
LARGE_JOBS="${DCDLP_LARGE_JOBS:-1}"
THREADS_PER_JOB="${DCDLP_THREADS_PER_JOB:-8}"
LARGE_BATCH_SIZE="${DCDLP_LARGE_BATCH_SIZE:-8192}"
LARGE_MAX_INTERVENTIONS_PER_BATCH="${DCDLP_LARGE_MAX_INTERVENTIONS_PER_BATCH:-1}"
LARGE_MAX_INTERVENTIONS_PER_EPOCH="${DCDLP_LARGE_MAX_INTERVENTIONS_PER_EPOCH:-256}"
RUN_MODE="${DCDLP_RUN_MODE:-full}"
PREFLIGHT_ONLY="${DCDLP_PREFLIGHT_ONLY:-false}"

cd "$ROOT"
mkdir -p results/logs results/raw results/checkpoints results/aggregate results/figures results/tables

if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  current_status="$(docker inspect --format '{{.State.Status}}' "$CONTAINER")"
  if [[ "$current_status" == "running" ]]; then
    echo "Container $CONTAINER is already running."
    docker logs --tail 30 "$CONTAINER"
    exit 0
  fi
  docker rm "$CONTAINER" >/dev/null
fi

docker run --detach \
  --name "$CONTAINER" \
  --gpus all \
  --network none \
  --shm-size 16g \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --env PYTHONPATH=/workspace/src \
  --env DCDLP_GPUS="$GPUS" \
  --env DCDLP_JOBS="$JOBS" \
  --env DCDLP_LARGE_JOBS="$LARGE_JOBS" \
  --env DCDLP_THREADS_PER_JOB="$THREADS_PER_JOB" \
  --env DCDLP_LARGE_BATCH_SIZE="$LARGE_BATCH_SIZE" \
  --env DCDLP_LARGE_MAX_INTERVENTIONS_PER_BATCH="$LARGE_MAX_INTERVENTIONS_PER_BATCH" \
  --env DCDLP_LARGE_MAX_INTERVENTIONS_PER_EPOCH="$LARGE_MAX_INTERVENTIONS_PER_EPOCH" \
  --env DCDLP_RUN_MODE="$RUN_MODE" \
  --env DCDLP_RESUME=true \
  --env DCDLP_PREFLIGHT_ONLY="$PREFLIGHT_ONLY" \
  --volume "$ROOT:/workspace" \
  --workdir /workspace \
  "$IMAGE" \
  bash scripts/run_full_experiments.sh

echo "Started $CONTAINER."
echo "Follow progress with: docker logs -f $CONTAINER"
echo "Persistent log: $ROOT/results/logs/container_launcher.log"
