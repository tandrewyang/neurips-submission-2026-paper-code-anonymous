#!/bin/bash
# Stage 1: DreamDojo 2B (lyg0241 GPU 0) → DreamDojo 14B (lyg0270 GPU 6,7)
# Run from anywhere; logs go to Mirabench_exp/logs/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"
INFER_SCRIPT="${SCRIPT_DIR}/run_dreamdojo_mirabench.py"
DATA_ROOT="/mnt/users/zirui/mizirui_benchmark/World_Fufu/ActionFollowing/video_batch_final"
OUT_ROOT="/mnt/users/zirui/mizirui_benchmark/Mirabench_exp"

mkdir -p "${LOG_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: DreamDojo 2B on lyg0241 GPU 0
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "[Stage 1] DreamDojo 2B — lyg0241 GPU 0"
echo "============================================================"

# gen subset (15 episodes)
echo "[2B] Running gen subset..."
CUDA_VISIBLE_DEVICES=0 python3 "${INFER_SCRIPT}" \
    --model-size 2b \
    --subset gen \
    --data-root "${DATA_ROOT}" \
    --out-root "${OUT_ROOT}" \
    2>&1 | tee "${LOG_DIR}/dreamdojo_2b_gen.log"

echo "[2B] gen done. Starting gr1 subset..."

# gr1 subset (50 episodes)
CUDA_VISIBLE_DEVICES=0 python3 "${INFER_SCRIPT}" \
    --model-size 2b \
    --subset gr1 \
    --data-root "${DATA_ROOT}" \
    --out-root "${OUT_ROOT}" \
    2>&1 | tee "${LOG_DIR}/dreamdojo_2b_gr1.log"

echo "[2B] ALL DONE at $(date)"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: DreamDojo 14B on lyg0270 GPU 6,7
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "[Stage 1] DreamDojo 14B — lyg0270 GPU 6,7"
echo "============================================================"

ssh lyg0270 bash <<'REMOTE_EOF'
set -euo pipefail

INFER_SCRIPT="/mnt/users/zirui/mizirui_benchmark/Mirabench_exp/scripts/run_dreamdojo_mirabench.py"
DATA_ROOT="/mnt/users/zirui/mizirui_benchmark/World_Fufu/ActionFollowing/video_batch_final"
OUT_ROOT="/mnt/users/zirui/mizirui_benchmark/Mirabench_exp"
LOG_DIR="${OUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

echo "[14B] Running gen subset on GPU 6..."
CUDA_VISIBLE_DEVICES=6 python3 "${INFER_SCRIPT}" \
    --model-size 14b \
    --subset gen \
    --data-root "${DATA_ROOT}" \
    --out-root "${OUT_ROOT}" \
    2>&1 | tee "${LOG_DIR}/dreamdojo_14b_gen.log"

echo "[14B] gen done. Starting gr1 subset..."

CUDA_VISIBLE_DEVICES=6 python3 "${INFER_SCRIPT}" \
    --model-size 14b \
    --subset gr1 \
    --data-root "${DATA_ROOT}" \
    --out-root "${OUT_ROOT}" \
    2>&1 | tee "${LOG_DIR}/dreamdojo_14b_gr1.log"

echo "[14B] ALL DONE at $(date)"
REMOTE_EOF

echo ""
echo "============================================================"
echo "[Stage 1 COMPLETE] DreamDojo 2B + 14B finished at $(date)"
echo "Results: ${OUT_ROOT}/dreamdojo_2b/ and ${OUT_ROOT}/dreamdojo_14b/"
echo "============================================================"
