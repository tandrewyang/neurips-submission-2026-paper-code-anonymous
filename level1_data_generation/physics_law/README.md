# Physics Law Compliance Evaluator (Level 1b)

Explicit kinematic-fit evaluator for free-fall and horizontal
push / slide videos. The kinematic score is purely deterministic
(SAM2 mask centroid + polynomial fit + event features); a single VLM
call serves only as a 10 % video-quality gate.

## Files

| File | Role |
|---|---|
| `physics_analyzer.py` | Vertical free-fall pipeline (SAM2 propagation, segmentation, polyfit, event features, VQS gate, final-score combiner). |
| `horizontal_push_analyzer.py` | Horizontal push / slide pipeline (push / slide / rest segmentation; same three-factor scoring adapted to friction). |
| `unified_motion_analyzer.py` | Top-level single-video entry. Runs SAM2 once, classifies the dominant axis, dispatches to the matching pipeline. |
| `run_batch.py` | Batch driver over a directory of MP4s. |

## Install

```bash
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/sam2.git"

# SAM2.1 Hiera-Large weights
mkdir -p checkpoints
wget -O checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

## VLM configuration

The pipeline calls a multimodal model in two places:

1. **First-frame object localization** — used to bootstrap SAM2 (you
   can replace this with any open-vocabulary detector by passing
   `first_bbox=(x1,y1,x2,y2)` directly to `analyze_motion`).
2. **Stage-0 Video Quality Score (VQS)** — single call returning
   `{video_ok, has_motion}` in JSON.

Configure via environment variables:

```bash
export PHYSBENCH_VLM_API_KEY="..."
export PHYSBENCH_VLM_BASE_URL="https://api.openai.com/v1"   # any OpenAI-compatible
export PHYSBENCH_VLM_MODEL="gpt-4.1-mini"                   # or gemini-2.5-pro / qwen3-vl, ...
```

## Usage

### Single video

```bash
python unified_motion_analyzer.py path/to/video.mp4
```

Writes `<name>_unified.json` containing `score_pct`, per-segment
diagnostics (R², `sign_ok`, `magnitude_ok`, `uniformity_ok`,
`seg_score`), event features (`velocity_drop`, `post_landing_drift`,
`bounce_decay`), and the SAM2 centroid trajectory.

### Batch over a directory

```bash
export PHYSBENCH_VLM_API_KEY="..."
python run_batch.py \
  --input_dir  path/to/videos \
  --output_dir path/to/results
```

Each video produces `<name>_unified.json` plus an annotated
`<name>_annotated.mp4` overlay. Add `--no-annotate` to skip the
overlay or `--skip-existing` for incremental runs.

## Scoring formula

The final score is computed on the 0–100 scale:

```
PhysLawScore = 0.9 * effective_physics + VQS
effective_physics = 100 * kinematic_score   if has_motion else 0
VQS ∈ {0, 5, 10}
```

`kinematic_score ∈ [0, 1]` is a gated mix of two branches:

```
kinematic_score = 0.7 * curve + 0.3 * event   if curve <  0.3
                = 0.3 * curve + 0.7 * event   if curve >= 0.3
                = curve  or  event             if only one is usable
```

### Curve branch

```
seg_score   = sign_ok * magnitude_ok * uniformity_ok
curve_score = Agg(seg_score) * min(1, (n_valid/n_ref) / 0.3)
```

| Factor | Physical principle |
|---|---|
| `sign_ok` | Acceleration in the physically expected direction (gravity / friction). Wrong sign on a fall/rise segment forces `curve_score = 0`. A `rise` segment before any `fall` is also a hard ordering violation. |
| `magnitude_ok` | Vertical fall: `\|a\| / a_exp ∈ [0.3, 3]` → 1, linear decay outside. Horizontal slide: velocity decay `d ≥ 0.3` → 1; `d ∈ [0.05, 0.3]` linear to 0.4-1.0; `d < 0.05` collapses. |
| `uniformity_ok` | Half-split each segment, fit each half. `half_cv ≤ 0.15` → 1, `half_cv ≥ 0.80` → 0, linear in between. |

Aggregation: vertical uses a sign-gated **length-weighted mean** with a
hard `sign_ok=0 → 0` override; horizontal uses a plain mean over slide
segments, with a stricter coverage threshold (0.6) and a slide-coverage
factor `min(1, n_slide/n_valid/0.5)`.

The polynomial `R²` is recorded per segment but **does not** enter
`seg_score`.

### Event branch (vertical only)

Four discrete features combined with weights (0.30, 0.20, 0.30, 0.20):

| Feature | Mapping |
|---|---|
| `velocity_drop` | `(v_before - v_after) / v_before` clipped to [0,1] |
| `post_landing_drift` | `1 - 0.5 * Δy_post / 0.10`  |
| `has_impact` | binary 0 / 1 |
| `bounce_decay = h₂/h₁` | 1 if ≤ 0.7; linearly → 0 at 1.5; 0 above |

`event_score` is declared usable only when an impact was detected and
a fall phase exists.
