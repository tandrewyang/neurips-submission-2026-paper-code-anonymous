# Part 3: Text-Instruction-Conditioned Video Generation

## Directory structure

```
data_generation_part3_instruction/
├── README.md               ← this file
├── setup_data.py           ← one-time: extract GT clips, first frames, prompts per task/episode
├── run_model.py            ← universal runner: --model MODEL_NAME --tasks ... --gpu N
└── tasks/
    ├── gr1_pnp_apple/
    │   ├── task_info.json              # dataset path, schedule (conditions), episode indices
    │   ├── ep000_idx137/
    │   │   ├── first_frame.jpg         # first frame of episode 137
    │   │   ├── gt.mp4                  # ground truth clip (77 frames)
    │   │   └── prompts.json            # {condition: prompt string}
    │   ├── ep001_idx3022/
    │   └── ep002_idx5843/
    ├── gr1_pnp_mango/
    ...
    └── fold_cloth/
```

## Quick start

### Step 1: Extract data (one-time)
```bash
cd /path/to/OptimismBias-WorldModel-Benchmark
/path/to/your_env/DreamDojo/.venv/bin/python3 \
    data_generation_part3_instruction/setup_data.py
```

### Step 2: Generate videos for a model
```bash
CUDA_VISIBLE_DEVICES=4 \
/path/to/your_env/DreamDojo/.venv/bin/python3 \
    data_generation_part3_instruction/run_model.py \
    --model cosmos14b_gr1 \
    --out-dir test_data/cosmos14b_gr1
```

Output: `test_data/<model_name>/<task>/ep<N>/<condition>_pred.mp4`

### Step 3: Assemble comparison videos
```bash
python3 scripts/assemble_test_data.py --model cosmos14b_gr1
```

Output: `test_data/<model_name>/<task>/ep<N>/compare_<condition>.mp4`
         `test_data/<model_name>/vlm_eval_input.jsonl`  ← feed to eval_vlm_threeway.py

## Available models (--model argument)
See `adapters/__init__.py` for the full list. Common choices:
- `cosmos14b_gr1`   — Cosmos-Predict2-14B text+image → video
- `wan14b`         — Wan2.1-14B
- `wan1b`          — Wan2.1-1B
- `cogvideox`      — CogVideoX-5B
