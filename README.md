# MiraBench: Evaluating Action-Conditioned Reliability in Robotic World Models

Anonymous code and data release for the NeurIPS 2026 submission.

## Repository layout

```
├── MiraBench_dataset/             # Released human-annotation corpus (906 video–annotation pairs)
├── level1_data_generation/        # Level-1: Phys-Cons + Phys-Law evaluator scripts
├── level2_data_generation/        # Level-2: Action-Following generation scripts (Wan / Cosmos / DreamDojo / ...)
├── level3_data_generation/        # Level-3: Optimism-Bias data generation (action- and instruction-conditioned)
├── evaluation/                    # Level-3 VLM-based evaluation pipeline (InternVL3-78B)
├── video_batch_final/             # Generated videos used by the Level-3 evaluator
├── .gitignore
└── README.md
```

Each `level{N}_data_generation/` subfolder ships its own README with usage details.

For the released human-annotation dataset (videos + annotations + parsed CSVs),
see [`MiraBench_dataset/README.md`](MiraBench_dataset/README.md).

---

## Level 3 — Optimism Bias Benchmark

This module evaluates whether world models exhibit **optimism bias** — ignoring perturbations applied to robot actions and still predicting normal/successful outcomes.

### Evaluation Method

**VLM Judge**: InternVL3-78B with 7-frame majority voting.

For each baseline-perturbed pair:
1. Extract 7 frames at [81%, 83%, 85%, 87%, 90%, 95%, 97%] of video progress
2. Each frame: concatenate baseline (LEFT) and perturbed (RIGHT) side-by-side
3. Ask VLM: "Same or Different?" with dynamic resolution (no distortion)
4. Majority vote → Y (Same = optimism bias) or N (Different = no bias)

**Scoring**: `Score = (1 - Y_rate) × 100`
- Higher = better (model correctly detects perturbation effects)
- 100 = model detects ALL perturbations
- 0 = model ignores ALL perturbations (maximum optimism bias)

### Perturbation Types

Each episode has 3 perturbations:
- **2 mandatory**: `implicit_grip_force_weak`, `implicit_premature_release`
- **1 optional** (varies per episode): from `contact_oscillation`, `approach_overshoot`, `wrist_tilt_grasp`, `carry_inertial_jerk`, `grip_carry_slip`

Total: 13 episodes × 3 perturbations = **39 evaluation pairs per model**

### Two Prompt Versions

| Prompt | Used for | Tolerance |
|--------|----------|-----------|
| Standard | Action-conditioned models (DreamDojo, Cosmos-Predict2) | Strict: object must be in same location |
| Lenient | Text/instruction-conditioned models (HappyHorse, Wan2.1) | Lenient: only fundamentally different actions count |

### Usage

```bash
# Generate videos for a model
python level3_data_generation/action_conditioned/run_model.py \
    --model cosmos_predict2_2b_gr1 \
    --out-dir test_data/cosmos_predict2_2b_gr1

# Evaluate a specific model
python evaluation/eval_optimism.py --batch dreamdojo_14b_gr1

# Score all models in test_data/
python evaluation/score_test_data.py
```

### Requirements

- InternVL3-78B (4× A100 80GB)
- Python 3.10+
- PyTorch 2.7+, transformers 4.51.3, timm, einops, opencv-python
