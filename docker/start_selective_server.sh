#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${DCDLP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${DCDLP_IMAGE:-dcdlp:selective-2.3.1-cu121}"
CONTAINER="${DCDLP_CONTAINER:-dcdlp-selective-server}"
ARCHIVE="${DCDLP_IMAGE_ARCHIVE:-$ROOT/offline/dcdlp-selective-image.tar}"
MODE="${DCDLP_RUN_MODE:-cora5}"
NETWORK="${DCDLP_NETWORK:-bridge}"
GPUS="${DCDLP_GPUS:-all}"
SHM_SIZE="${DCDLP_SHM_SIZE:-16g}"

cd "$ROOT"
mkdir -p results/phase2_selective_server

if ! command -v docker >/dev/null 2>&1; then
  echo 'docker is not installed or not on PATH.' >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo 'Docker daemon is unavailable. Start Docker and retry.' >&2
  exit 1
fi
if [[ "$GPUS" != "none" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo 'nvidia-smi is unavailable. Set DCDLP_GPUS=none for a CPU smoke run.' >&2
  exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  if [[ ! -f "$ARCHIVE" ]]; then
    echo "Image $IMAGE is not loaded and archive is missing: $ARCHIVE" >&2
    exit 1
  fi
  echo "[server] Loading $ARCHIVE ..."
  docker load --input "$ARCHIVE"
fi
docker image inspect "$IMAGE" >/dev/null

if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  status="$(docker inspect --format '{{.State.Status}}' "$CONTAINER")"
  if [[ "$status" == running ]]; then
    echo "[server] $CONTAINER is already running."
    docker logs --tail 30 "$CONTAINER"
    exit 0
  fi
  # Only the named launcher container is removed; result files remain on host.
  docker rm "$CONTAINER" >/dev/null
fi

gpu_args=()
if [[ "$GPUS" != "none" ]]; then
  gpu_args=(--gpus "$GPUS")
fi
env_args=(
  --env HOME=/tmp
  --env PYTHONPATH=/workspace/src
  --env DCDLP_RUN_MODE="$MODE"
  --env DCDLP_DEVICE="${DCDLP_DEVICE:-auto}"
  --env DCDLP_SEEDS="${DCDLP_SEEDS:-0,1,2,3,4}"
  --env DCDLP_MODELS="${DCDLP_MODELS:-M0,M1,M2,M3}"
  --env DCDLP_PRETRAIN_EPOCHS="${DCDLP_PRETRAIN_EPOCHS:-5}"
  --env DCDLP_ROUTING_EPOCHS="${DCDLP_ROUTING_EPOCHS:-5}"
  --env DCDLP_MAX_PROBE_PAIRS="${DCDLP_MAX_PROBE_PAIRS:-512}"
  --env DCDLP_OUTPUT_DIR="${DCDLP_OUTPUT_DIR:-results/phase2_selective_server/${MODE}}"
)
for proxy_name in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  if [[ -n "${!proxy_name:-}" ]]; then
    env_args+=(--env "$proxy_name=${!proxy_name}")
  fi
done

echo "[server] Starting $CONTAINER (mode=$MODE, network=$NETWORK, gpus=$GPUS) ..."
docker run --detach \
  --name "$CONTAINER" \
  "${gpu_args[@]}" \
  --network "$NETWORK" \
  --shm-size "$SHM_SIZE" \
  --user "$(id -u):$(id -g)" \
  "${env_args[@]}" \
  --volume "$ROOT/results:/workspace/results" \
  --workdir /workspace \
  "$IMAGE" \
  bash /workspace/docker/server_selective_entrypoint.sh

echo "[server] Started $CONTAINER."
echo "[server] Live log: docker logs -f $CONTAINER"
echo "[server] Output: $ROOT/results/phase2_selective_server/$MODE"
echo "[server] Stop safely: docker stop $CONTAINER"
