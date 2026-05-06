# MiraBench Action Following Evaluation

Evaluation scripts for the MiraBench action following benchmark, measuring **TCR** (Task Completion Rate), **OPS** (Object Preservation Score), and **GEN** (Generalizability).

## Metrics

### TCR — Task Completion Rate

Binary per-episode metric: does the predicted video complete the task?

- **Method C**: 16 uniformly sampled frames from the predicted video (no GT reference) are presented to InternVL3-78B in a single call. The VLM outputs 0 or 1 for the whole video.
- No per-frame voting, no tail-exponential weighting, no GT frames shown to the judge.

### OPS — Object Preservation Score

Binary per-episode metric: are objects visually coherent throughout the video?

- 16 frame pairs (pred + GT) are independently judged by InternVL3-78B.
- confidence = mean(votes); OPS = high (≥0.70) / medium (≥0.40) / low (<0.40).

### GEN — Generalizability

Derived from TCR scores across splits:

```
GEN = min(100, 100 * exp(-(TCR_GR1 - TCR_Gen) / 100))
```

- GEN = 100%: no overfitting (GR1 ≤ Gen)
- GEN < 100%: exponential penalty for the GR1-Gen gap

## Setup

```bash
pip install torch torchvision transformers numpy pillow
```

InternVL3-78B model weights must be available. Set the path via environment variable:

```bash
export INTERNVL_MODEL_PATH=/path/to/InternVL3-78B
```

Distributed across 3× 80GB GPUs (e.g., `CUDA_VISIBLE_DEVICES=0,1,2`).

## Usage

### TCR + GEN (single model)

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python3 eval_tcr.py \
    --model_dir  ../dreamdojo_2b/gr1 \
    --gt_dir     ../video_batch_final/gr1 \
    --pred_filename pred_2b.mp4 \
    --output     results/dreamdojo_2b_gr1_tcr.json
```

### TCR + GEN (all models at once)

Loads InternVL3-78B once and iterates over all models defined in `MODEL_PRED_TABLE`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python3 eval_tcr.py --all --output_dir results/tcr
```

Automatically prints a GEN summary table when all jobs complete.

### GEN only (from existing TCR results)

```bash
python3 eval_tcr.py --compute_gen --gen_dir results/tcr
```

### OPS (per model+subset)

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python3 eval_ops.py \
    --model_dir  ../dreamdojo_2b/gr1 \
    --gt_dir     ../video_batch_final/gr1 \
    --pred_filename pred_2b.mp4 \
    --output     results/dreamdojo_2b_gr1_ops.json
```

## Data Layout

```
video_batch_final/
├── gr1/                          # GR1 split (50 episodes)
│   ├── episode_0005/
│   │   ├── gt_0005.mp4           # Ground-truth video
│   │   └── instruct_0005.txt     # Task instruction
│   └── ...
└── generalizability/             # Gen split (15 episodes)
    ├── episode_0000/
    └── ...

{model_name}/
├── gr1/
│   └── episode_0005/
│       └── pred.mp4              # Predicted video (or pred_2b.mp4, etc.)
└── generalizability/
    └── ...
```

## Output Format

### TCR JSON

```json
{
  "summary": {
    "model": "dreamdojo_2b",
    "split": "gr1",
    "method": "C-whole16",
    "n_episodes": 50,
    "n_valid": 50,
    "n_tcr1": 46,
    "tcr_rate": 0.92
  },
  "results": [
    {"episode": 5, "TCR": 1, "raw_response": "1", "parse_ok": true},
    ...
  ]
}
```

### OPS JSON

```json
{
  "summary": {
    "n_episodes": 50,
    "ops_distribution": {"high": 48, "medium": 2, "low": 0},
    "mean_confidence": 0.96,
    "mean_ops_score": 33.6
  },
  "results": [
    {"episode": 5, "OPS": "high", "confidence": 1.0, "votes": [1, ...]},
    ...
  ]
}
```
