# Optimism Bias Benchmark — Part 3: Perturbation Sensitivity

This module evaluates whether world models exhibit **optimism bias** — ignoring perturbations applied to robot actions and still predicting normal/successful outcomes.

## Directory Structure

```
├── data_generation/
│   ├── action_conditioned/      # Generate videos for action-conditioned models
│   │   ├── run_model.py         # Universal generation runner
│   │   ├── setup_data.py        # Prepare input data (first frames + actions)
│   │   └── tasks/               # Per-task configs with perturbation assignments
│   └── instruction_conditioned/ # Generate videos for text/instruction-conditioned models
│       ├── run_model.py
│       └── tasks/
├── evaluation/
│   ├── eval_optimism.py         # VLM-based evaluation (InternVL3-78B)
│   ├── score_test_data.py       # Batch scoring script for all models
│   ├── prompt_standard.txt      # Standard prompt (for action-conditioned models)
│   ├── prompt_lenient.txt       # Lenient prompt (for text-conditioned models)
│   └── perturbation_selection.json  # Per-episode perturbation selection
└── README.md
```

## Evaluation Method

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

## Perturbation Types

Each episode has 3 perturbations:
- **2 mandatory**: `implicit_grip_force_weak`, `implicit_premature_release`
- **1 optional** (varies per episode): from `contact_oscillation`, `approach_overshoot`, `wrist_tilt_grasp`, `carry_inertial_jerk`, `grip_carry_slip`

Total: 13 episodes × 3 perturbations = **39 evaluation pairs per model**

## Two Prompt Versions

| Prompt | Used for | Tolerance |
|--------|----------|-----------|
| Standard | Action-conditioned models (DreamDojo, Cosmos-Predict2) | Strict: object must be in same location |
| Lenient | Text/instruction-conditioned models (HappyHorse, Wan2.1) | Lenient: only fundamentally different actions count |

## Usage

```bash
# Generate videos for a model
python data_generation/action_conditioned/run_model.py \
    --model cosmos_predict2_2b_gr1 \
    --out-dir test_data/cosmos_predict2_2b_gr1

# Evaluate a specific model
python evaluation/eval_optimism.py --batch dreamdojo_14b_gr1

# Score all models in test_data/
python evaluation/score_test_data.py
```

## Requirements

- InternVL3-78B (4× A100 80GB)
- Python 3.10+
- PyTorch 2.7+, transformers 4.51.3, timm, einops, opencv-python
