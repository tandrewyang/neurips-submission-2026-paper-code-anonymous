# Part 3: Action-Conditioned Video Generation

## Overview

This module generates benchmark videos by feeding **perturbed robot actions** to a world model.
The core pipeline uses DreamDojo's native `action_conditioned.py` script, which:

1. Loads actions from a **GR1 dataset** (LeRobot parquet format, 384D action space)
2. Applies perturbations **online** via `module_a_ewmbench/perturbations.py`
3. Generates videos conditioned on the perturbed actions

## Data Flow (Native Pipeline)

```
┌─────────────────────────────────────────────────────────────────────┐
│  run_native_s0.sh / run_native_s1.sh                                │
│                                                                     │
│  For each (task, episode):                                          │
│    action_conditioned.py                                            │
│      ├── --dataset-path <GR1_FILTERED_DATASET>                      │
│      │     └── MultiVideoActionDataset loads episode from parquet   │
│      │         → returns 384D normalized delta actions              │
│      ├── --batch-perturbations-schedule-file <SCHEDULE.json>        │
│      │     └── defines 4 conditions to run per episode              │
│      ├── --optimism-bias-repo <REPO_ROOT>                           │
│      │     └── imports module_a_ewmbench/perturbations.py           │
│      │         perturb_actions(actions[1,T,384], type, severity)    │
│      └── generates video for each condition                         │
│            → saves to test_data/<model>/<task>/ep_idx<N>/<cond>/    │
└─────────────────────────────────────────────────────────────────────┘
```

**Key point**: The pipeline does NOT read from `tasks/*/actions_per_condition/*.npy`.
Those `.npy` files are pre-computed references for the adapter-based pipeline (`run_model.py`)
only. The native pipeline loads actions directly from the dataset at runtime.

## Dataset Requirements

The GR1 filtered datasets must be available. Each is a directory containing:
- `data/` — parquet shards with columns: `video` (frames), `action` (384D float32)
- `meta/` — dataset metadata

Expected locations (configurable in `run_native_*.sh`):

| Task | Dataset Path |
|------|-------------|
| gr1_pnp_apple | `data/gr1_filtered/gr1_pnp_apple` |
| gr1_pnp_mango | `data/gr1_filtered/gr1_pnp_mango` |
| gr1_pnp_pear | `data/gr1_filtered/gr1_pnp_pear` |
| gr1_egodex_simple | `data/gr1_filtered/gr1_egodex_simple` |
| pnp_corn | `<INLAB_EVAL>/gr1_unified.pnp_corn_robot` |
| pnp_cucumber | `<INLAB_EVAL>/gr1_unified.pnp_cucumber_robot` |
| pnp_dragonfruit | `<INLAB_EVAL>/gr1_unified.pnp_dragonfruit_robot` |
| pour_items | `<INLAB_EVAL>/gr1_unified.pour_items_into_basket_robot` |
| fold_cloth | `<INLAB_EVAL>/gr1_unified.fold_cloth_robot` |

Where `<INLAB_EVAL>` defaults to the GR00T Teleop In-lab_Eval directory.

## Action Format (384D)

The dataset stores **normalized chunk-based delta actions** in 384 dimensions:
- Dims 0–28: active action channels (29D = left_arm[7] + right_arm[7] + left_hand[6] + right_hand[6] + waist[3])
- Dims 29–383: zero-padded (unused)

Normalization: min-max to [-1, 1] using q01/q99 from `GR1_unified_stats.json`, then chunk-based deltas (`action[t] - action[chunk_start]`).

## Perturbation Schedule

Each task has a schedule JSON in `schedules/<task>.json`:
```json
[
  {"perturbation_type": null, "perturbation_severity": 1.0, "relative_subdir": "baseline"},
  {"perturbation_type": "implicit_grip_force_weak", "perturbation_severity": 0.5, "relative_subdir": "implicit_grip_force_weak_s0.5"},
  {"perturbation_type": "implicit_premature_release", "perturbation_severity": 0.5, "relative_subdir": "implicit_premature_release_s0.5"},
  {"perturbation_type": "<random_from_pool>", "perturbation_severity": 0.5, "relative_subdir": "<type>_s0.5"}
]
```

- 2 mandatory perturbations: `implicit_grip_force_weak` + `implicit_premature_release`
- 1 random perturbation sampled from pool (varies per task)
- 1 baseline (no perturbation)
- Total: 4 conditions per episode

## Benchmark Design

| Category | Tasks | Episodes/task | Conditions/episode | Videos |
|----------|-------|---------------|-------------------|--------|
| PNP (pick-and-place) | 6 | 2 | 4 | 48 |
| Other (pour, fold, egodex) | 3 | 1 | 4 | 12 |
| **Total** | **9** | — | — | **60** |

## Directory Structure

```
data_generation_part3_action/
├── README.md                   ← this file
├── run_native.sh               ← single-GPU full run (sequential, 60 videos)
├── run_native_s0.sh            ← shard 0: ep0 of all tasks (36 videos)
├── run_native_s1.sh            ← shard 1: ep1 of PNP tasks (24 videos)
├── schedules/                  ← per-task perturbation schedules (JSON)
│   ├── gr1_pnp_apple.json
│   └── ...
├── setup_data.py               ← one-time: extract GT clips, first frames
├── run_model.py                ← adapter-based runner (alternative to native)
└── tasks/
    ├── gr1_pnp_apple/
    │   ├── task_info.json
    │   ├── ep000_idx137/
    │   │   ├── first_frame.jpg
    │   │   ├── gt.mp4
    │   │   └── actions_per_condition/   ← used by run_model.py only (NOT native)
    │   │       ├── baseline.npy
    │   │       └── ...
    │   └── ep001_idx8483/
    └── ...
```

## Quick Start

### Native Pipeline (DreamDojo 14B GR1) — Recommended

```bash
# Single GPU:
CUDA_VISIBLE_DEVICES=0 bash data_generation_part3_action/run_native.sh

# Two GPUs in parallel (split by episodes):
CUDA_VISIBLE_DEVICES=0 bash data_generation_part3_action/run_native_s0.sh &
CUDA_VISIBLE_DEVICES=1 bash data_generation_part3_action/run_native_s1.sh &
```

Output: `test_data/dreamdojo_14b_gr1/<task>/ep_idx<N>/<condition>/0000_pred.mp4`

### Adapter Pipeline (other models)

```bash
CUDA_VISIBLE_DEVICES=5 python3 data_generation_part3_action/run_model.py \
    --model cosmos_predict2_2b_gr1 \
    --out-dir test_data/cosmos_predict2_2b_gr1
```

Output: `test_data/<model_name>/<task>/ep<N>/<condition>_pred.mp4`

## Environment Requirements

- Python 3.10+ with DreamDojo venv (PyTorch 2.7+, cosmos_predict2, transformers)
- DreamDojo checkpoint: `14B_GR1_post-train/iter_000050000/model_ema_bf16.pt`
- Cosmos tokenizer: `Cosmos-Predict2.5-2B/tokenizer.pth`
- Cosmos-Reason1-7B (for guardrail): cached in HuggingFace hub
- 80GB GPU (A100/H100) per instance

## Available Models (--model for run_model.py)

Action-conditioned models with GR1 support:
- `cosmos_predict2_2b_gr1` — Cosmos 2B fine-tuned on GR1
- `cosmos_predict2_14b_gr1` — Cosmos 14B fine-tuned (needs 80GB GPU)
- `irasim_gr1` — IRASim fine-tuned
- `ctrlworld_gr1` — CtrlWorld fine-tuned
- `enerverse_ac` — EnerVerse action-conditioned (run LAST due to monkey-patching)
- `dreamdojo_2b_gr1` — DreamDojo 2B
- `dreamdojo_14b_gr1` — DreamDojo 14B (native pipeline preferred)

## Porting to a New Environment

To run this on a different machine:

1. Install DreamDojo and its dependencies (cosmos_predict2, cosmos_oss)
2. Place GR1 datasets in `data/gr1_filtered/` (or update paths in `run_native_*.sh`)
3. Download model checkpoints and update paths in `run_native_*.sh`:
   - `CHECKPOINT=<path_to_14B_GR1_post-train_model_ema_bf16.pt>`
   - `CHECKPOINTS_DIR=<path_to_dreamdojo_checkpoints_dir>`
   - Set `COSMOS_CHECKPOINT_OVERRIDE_*` env vars to local paths
4. Run `python setup_data.py` to extract first frames and GT videos (for evaluation)
5. Launch generation with `run_native.sh`
