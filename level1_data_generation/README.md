# MiraBench — Level 1 Physics Evaluators

Reference-free physics evaluators for action-conditioned video world
models, paired with the MiraBench paper (Level 1 — Physics Adherence).
Two independent modules:

```
level1_data_generation/
├── physics_law/                # Level 1b — quantitative kinematic compliance
│   ├── physics_analyzer.py         # Free-fall pipeline (SAM2 + polyfit + VLM gate)
│   ├── horizontal_push_analyzer.py # Horizontal push / slide pipeline
│   ├── unified_motion_analyzer.py  # Single-video entry point (axis dispatch)
│   ├── run_batch.py                # Batch driver over a directory of videos
│   ├── requirements.txt
│   └── README.md
│
└── physical_consistency/       # Level 1a — behavioural-level consistency (PCS)
    ├── evaluate.py                 # PCS evaluator (D1 Object + D2 Occlusion)
    ├── requirements.txt
    └── README.md
```

---

## Level 1b — Physics Law Compliance

Explicit kinematic computation on a SAM2-tracked object trajectory,
gated by a 10 % VLM Video Quality Score (VQS). Produces a 0-100
`PhysLawScore` on the kinematic component plus VQS contribution.

```
PhysLawScore = 0.9 * effective_physics + VQS         # 0-100 scale
effective_physics = 100 * kinematic_score   if has_motion else 0
VQS ∈ {0, 5, 10}
```

`kinematic_score` is the gated mix of the curve branch
(quadratic-fit three-factor product `sign · magnitude · uniformity`,
length-weighted across segments) and the event branch
(velocity drop, post-landing drift, has\_impact, bounce decay).
See [physics_law/README.md](physics_law/README.md) for the full
formula and hyperparameters.

### Quick start

```bash
cd physics_law
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/sam2.git"

# SAM2 weights (Hiera-Large)
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# OpenAI-compatible VLM endpoint (used as the Stage-0 quality gate
# and for first-frame object localization)
export PHYSBENCH_VLM_API_KEY="<your key>"
export PHYSBENCH_VLM_BASE_URL="https://api.openai.com/v1"   # or any compatible gateway
export PHYSBENCH_VLM_MODEL="gpt-4.1-mini"                   # or gemini-2.5-pro / qwen3-vl, ...

# Single video
python unified_motion_analyzer.py path/to/video.mp4

# Batch over a directory of MP4s
python run_batch.py --input_dir path/to/videos --output_dir path/to/results
```

---

## Level 1a — Physical Consistency (PCS)

Two binary indicators on **InternVL3-78B**, aggregated by equal-weight
mean. 20 frames → 10 midcut pairs → vertical-stacked image → binary
A / B verdict.

```
PCS = (s_obj + s_occ) / 2 ∈ [0, 100]    # percentage scale
s_d = (1 - bad_count / 10) * 100         # per-indicator score ∈ [0, 100]
```

| Indicator | Asks the VLM |
|-----------|---|
| **D1 Object Consistency** (`SC_A2`) | shape / material / size / colour stability |
| **D2 Occlusion Consistency** (`SC_O`) | object identity preserved through occlusion |

### Quick start

```bash
cd physical_consistency
pip install -r requirements.txt

# Path or HF model id of InternVL3-78B
# (defaults to "OpenGVLab/InternVL3-78B" — pulled from HF if not local)
export PHYSCONS_MODEL_PATH=/local/path/to/InternVL3-78B   # optional

python evaluate.py --video path/to/video.mp4 --gpus 4,5
python evaluate.py --video_list videos.txt --gpus 4,5 --out results.jsonl
```

InternVL3-78B needs ≈80 GB GPU memory; recommend 2× A100/H100-80GB
with `device_map="auto"`.

---

## Cite

```bibtex
@misc{mirabench2026,
  title  = {MiraBench: Evaluating Action-Conditioned Reliability in Robotic World Models},
  note   = {Manuscript},
  year   = {2026}
}
```
