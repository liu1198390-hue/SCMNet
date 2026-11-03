#!/usr/bin/env bash
# Example: DATA_ROOT=/path/to/data PRETRAINED=/path/to/vmamba_tiny_e292.pth bash scripts/run.sh glas
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
DATASET="${1:-glas}"
DATA_ROOT="${DATA_ROOT:-data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-0,1,2,3,4}"
EPOCHS="${EPOCHS:-30}"
GPU_ID="${GPU_ID:-0}"
: "${PRETRAINED:?Set PRETRAINED to the VMamba Tiny encoder checkpoint}"

case "$DATASET" in
  glas) PREFIX=GlaS; EVAL_PREFIX=GlaS_68_17; BATCH_SIZE=2; SPLITS=(testA testB test) ;;
  crag) PREFIX=CRAG; EVAL_PREFIX=CRAG; BATCH_SIZE=2; SPLITS=(test) ;;
  gzmh) PREFIX=GZMH_GLAND; EVAL_PREFIX=GZMH_GLAND; BATCH_SIZE=4; SPLITS=(test) ;;
  *) echo "Usage: $0 {glas|crag|gzmh}" >&2; exit 2 ;;
esac

TRAIN_DATA="${TRAIN_DATA:-$DATA_ROOT/${PREFIX}_coverage500_stride250_patches}"
EVAL_DATA="${EVAL_DATA:-$DATA_ROOT/${EVAL_PREFIX}_fg_aux}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
IFS=',' read -r -a SEED_LIST <<< "$SEEDS"
for seed in "${SEED_LIST[@]}"; do
  export PYTHONHASHSEED="$seed"
  RUN="$OUTPUT_ROOT/$DATASET/seed$seed"
  RESUME=()
  if [[ -f "$RUN/checkpoints/latest.pth" ]]; then
    RESUME=(--resume "$RUN/checkpoints/latest.pth")
  fi
  "$PYTHON" train.py --train_data_path "$TRAIN_DATA" --val_data_path "$EVAL_DATA" \
    --pretrained "$PRETRAINED" --output_dir "$RUN" --seed "$seed" \
    --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --amp --deterministic "${RESUME[@]}"
  for split in "${SPLITS[@]}"; do
    "$PYTHON" test.py --ckpt "$RUN/checkpoints/best_obj_F1.pth" \
      --data_path "$EVAL_DATA" --split "$split" --save_dir "$RUN/test/$split" --save_predictions
  done
done
