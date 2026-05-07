#!/usr/bin/env bash
# run_stage2_wan_api.sh
# Sequential pipeline:
#   1. Wan2.1 I2V   – GPU6 only (single-GPU + offload)
#   2. Wan2.2 I2V   – GPU6+7   (2-GPU FSDP, needs ~80 GB total)
#   3. API models   – Happy Horse + Wanx2.1 I2V Plus (no GPU needed)
#
# Usage:
#   bash run_stage2_wan_api.sh [--overwrite] [--subset gen|gr1|all]
#
# Expects to run ON lyg0270 (or ssh into it).
# Set DASHSCOPE_API_KEY if not hard-coded.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/usr/bin/python3   # system python3 on lyg0270
DATA_ROOT="/mnt/users/zirui/mizirui_benchmark/Mirabench_exp/video_batch_final"
OVERWRITE=""
SUBSET="all"

# ── parse args ──────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --overwrite) OVERWRITE="--overwrite" ;;
    --subset=*)  SUBSET="${arg#*=}" ;;
    --subset)    shift; SUBSET="$1" ;;
  esac
done

export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-sk-c354af57d6e54f7191e7c255cc54ab57}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 1. Wan2.1 I2V  (GPU6, single-GPU) ──────────────────────────
log "=== Stage 2-1: Wan2.1 I2V (GPU6) ==="
CUDA_VISIBLE_DEVICES=6 $PYTHON "$SCRIPT_DIR/run_wan21_mirabench.py" \
    --subset "$SUBSET" \
    --gpu 6 \
    --data-root "$DATA_ROOT" \
    $OVERWRITE
log "=== Wan2.1 DONE ==="

# ── 2. Wan2.2 I2V  (GPU6+7, 2-GPU FSDP) ──────────────────────
log "=== Stage 2-2: Wan2.2 I2V (GPU6+7) ==="
$PYTHON "$SCRIPT_DIR/run_wan22_mirabench.py" \
    --subset "$SUBSET" \
    --gpus 6,7 \
    --data-root "$DATA_ROOT" \
    $OVERWRITE
log "=== Wan2.2 DONE ==="

# ── 3. API models (no GPU) ──────────────────────────────────────
log "=== Stage 2-3: API models (Happy Horse + Wanx2.1 I2V Plus) ==="
$PYTHON "$SCRIPT_DIR/run_api_i2v.py" \
    --model all \
    --subset "$SUBSET" \
    --data-root "$DATA_ROOT" \
    $OVERWRITE
log "=== API models DONE ==="

log "Stage 2 complete."
