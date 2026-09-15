#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace

MODE="${DCDLP_RUN_MODE:-cora5}"
OUTPUT_DIR="${DCDLP_OUTPUT_DIR:-results/phase2_selective_server/${MODE}}"
DATA_ROOT="${DCDLP_DATA_ROOT:-data}"
DEVICE="${DCDLP_DEVICE:-auto}"
SEEDS="${DCDLP_SEEDS:-0,1,2,3,4}"
MODELS="${DCDLP_MODELS:-M0,M1,M2,M3}"
PRETRAIN_EPOCHS="${DCDLP_PRETRAIN_EPOCHS:-5}"
ROUTING_EPOCHS="${DCDLP_ROUTING_EPOCHS:-5}"
MAX_PROBE_PAIRS="${DCDLP_MAX_PROBE_PAIRS:-512}"

log() { printf '[dcdlp] %s\n' "$*"; }

if [[ "$MODE" == "shell" ]]; then
  exec bash
fi

if [[ "$MODE" == "preflight" || "${DCDLP_PREFLIGHT_ONLY:-false}" == "true" ]]; then
  log "running container preflight only"
  python - <<'PY'
from pathlib import Path
import torch
from dcdlp.data.loaders import load_dataset

root = Path('/workspace')
required = [
    root / 'data/processed/cora_heart_seed0.npz',
    root / 'third_party/HeaRT/dataset/cora/train_pos.txt',
    root / 'third_party/HeaRT/dataset/cora/valid_pos.txt',
    root / 'third_party/HeaRT/dataset/cora/test_pos.txt',
    root / 'third_party/HeaRT/dataset/cora/gnn_feature',
    root / 'third_party/HeaRT/dataset/cora/heart_valid_samples.npy',
    root / 'third_party/HeaRT/dataset/cora/heart_test_samples.npy',
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit('Missing packaged Cora/HeaRT files:\n' + '\n'.join(missing))
dataset = load_dataset('cora', root / 'data', 'heart', 0)
print(f'preflight: Cora nodes={dataset.num_nodes}, train={len(dataset.train_pos)}, '
      f'valid={len(dataset.valid_pos)}, test={len(dataset.test_pos)}')
print(f'preflight: torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, '
      f'cuda_device_count={torch.cuda.device_count()}')
PY
  exit 0
fi

case "$MODE" in
  smoke)
    DATASET=smoke
    SEEDS="${DCDLP_SEEDS:-0}"
    MODELS="${DCDLP_MODELS:-M0,M1,M3}"
    PRETRAIN_EPOCHS="${DCDLP_PRETRAIN_EPOCHS:-1}"
    ROUTING_EPOCHS="${DCDLP_ROUTING_EPOCHS:-1}"
    MAX_PROBE_PAIRS="${DCDLP_MAX_PROBE_PAIRS:-32}"
    ;;
  cora5|cora)
    DATASET=cora
    ;;
  *)
    echo "Unsupported DCDLP_RUN_MODE=$MODE (use preflight, smoke, cora5, or shell)." >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_DIR"
log "dataset=$DATASET seeds=$SEEDS models=$MODELS"
log "epochs=$PRETRAIN_EPOCHS+$ROUTING_EPOCHS device=$DEVICE output=$OUTPUT_DIR"
log "resume is enabled; existing files are never overwritten"

exec python scripts/run_phase2_selective_pipeline.py \
  --dataset "$DATASET" \
  --seeds "$SEEDS" \
  --models "$MODELS" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --pretrain-epochs "$PRETRAIN_EPOCHS" \
  --routing-epochs "$ROUTING_EPOCHS" \
  --device "$DEVICE" \
  --max-probe-pairs "$MAX_PROBE_PAIRS" \
  --resume
