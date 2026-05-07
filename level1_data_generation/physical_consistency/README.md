# MiraBench — Physical Consistency Evaluator

A drop-in evaluator that scores robot-manipulation videos for physical
consistency. Outputs a single `PCS` (Physical Consistency Score) in
`[0, 100]` (percentage scale, matching `PhysLawScore`), computed from
two binary VLM indicators run on `InternVL3-78B`.

## Quick start

```bash
pip install -r requirements.txt

# Single video (both metrics, default)
python evaluate.py --video path/to/video.mp4 --gpus 4,5

# Single metric only
python evaluate.py --video path/to/video.mp4 --gpus 4,5 --metric sc_a2
python evaluate.py --video path/to/video.mp4 --gpus 4,5 --metric sc_o

# Batch
python evaluate.py --video_list videos.txt --gpus 4,5 --out results.jsonl
```

The InternVL3-78B model is resolved from the `PHYSCONS_MODEL_PATH`
environment variable (default: `OpenGVLab/InternVL3-78B` — pulled from
HuggingFace Hub on first run). Set it to a local snapshot if needed:

```bash
export PHYSCONS_MODEL_PATH=/local/path/to/InternVL3-78B
```

## Scoring

```
PCS = (SC_A2 + SC_O) / 2            # equal-weight mean ∈ [0, 100]

SC_A2  Subject Physical Consistency  binary VLM judgement
SC_O   Occlusion Consistency         binary VLM judgement

For each indicator:
  1. Extract 20 evenly-spaced frames from the video
  2. Pair them via "midcut" (frame i ↔ frame i+10) → 10 pairs
  3. For each pair, ask InternVL3-78B  → A (consistent) or B (inconsistent)
  4. bad_count = number of B answers (out of 10)
  5. score = (1 − bad_count / 10) × 100             ∈ [0, 100]
  6. label = "B" if bad_count ≥ 1 else "A"           (strict threshold)
```

Both metrics share the same frame-extraction and pair-selection logic, so
their `na` flags are always synchronised; if one is computable, so is the
other.

| Indicator | What it asks the VLM |
|-----------|----------------------|
| **SC_A2** | Compare two frames of the manipulated object — does its **shape, material, and size** stay physically plausible? |
| **SC_O**  | Compare two frames where the object may be **partially or fully occluded** — does it keep the same identity / colour / shape? |

The exact prompts are hard-coded as `PROMPT_SC_A2` and `PROMPT_SC_O_V5` in
`evaluate.py`. Both default to "answer A or B on the first line". The
judgement is binary; we deliberately do not use a 4-tier rubric or sampling
multiple completions, because the threshold-based aggregation already
provides graded scores via `bad_count / n_pairs`.

## Output format

```jsonc
{
    "video_path": "path/to/video.mp4",
    "sc_a2": {
        "score":        90.0,                 // (1 - bad_count/10) * 100
        "label":        "A",                  // "B" if bad_count >= 1
        "bad_count":    1,
        "n_pairs":      10,
        "pair_results": ["A","A","B","A","A","A","A","A","A","A"],
        "na":           false
    },
    "sc_o": {
        "score":        80.0,
        "label":        "B",
        "bad_count":    2,
        "n_pairs":      10,
        "pair_results": ["A","A","A","B","A","A","B","A","A","A"],
        "na":           false
    },
    "pcs":  85.0                              // (sc_a2.score + sc_o.score) / 2  ∈ [0, 100]
}
```

If frame extraction fails (corrupted file, < 2 frames), `na: true` is set
on each metric, `score`/`label` become `null`, and `pcs` is `null`.

## Dependencies

- `transformers` + `torch` for InternVL3-78B
- `av` (PyAV) for frame decoding
- `Pillow` for image stacking

InternVL3-78B requires ~80 GB of GPU memory. The default config in
`evaluate.py` is `--gpus 4,5` (two NVIDIA A100/L40S-class cards).

## Citation

```bibtex
@misc{mirabench2026,
    title  = {MiraBench: A Physical Consistency Benchmark for
              Action-Driven Video Generation},
    note   = {Manuscript in preparation},
    year   = {2026}
}
```
