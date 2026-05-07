# MiraBench: Evaluating Action-Conditioned Reliability in Robotic World Models

> Anonymous code and data release accompanying the NeurIPS 2026 submission.

MiraBench introduces a three-level evaluation protocol that asks whether
an action-conditioned world model's predictions actually match reality.
Each level targets a distinct class of failure modes and produces an
independent score; in addition, the repository ships a human-annotation
corpus that lets one verify how closely each algorithmic metric tracks
human judgement.

## Repository layout

```
├── MiraBench_dataset/            # Released human-annotation corpus (906 mp4 + 906 json + parsed CSVs)
├── level1_data_generation/       # Level 1: Phys-Cons + Phys-Law evaluator source
│   ├── physical_consistency/         16-dim physical-consistency scoring (occ68, 4-model head-to-head)
│   └── physics_law/                  Free-fall physics-law scoring (multi-object anomaly analysis)
├── level2_data_generation/       # Level 2: Action-Following video-generation scripts
│                                     DreamDojo / Cosmos / Wan / Kling / ... runners
├── level3_data_generation/       # Level 3: Optimism-Bias data generation
│   ├── action_conditioned/           Action-conditioned models (DreamDojo, Cosmos-Predict2, ...)
│   └── instruction_conditioned/      Instruction-conditioned models (HappyHorse, Wan2.1, ...)
├── evaluation/                   # Level 3 VLM evaluation pipeline (InternVL3-78B judge)
├── video_batch_final/            # Generated videos used by the Level 3 evaluator
├── .gitignore
└── README.md
```

## The three evaluation levels

| Level | What it measures | Output | Subfolder |
|---|---|---|---|
| **Level 1a** | Physical consistency: two binary VLM indicators — **object stability** (`obj`, `SC_A2`) + **occlusion consistency** (`occ`, `SC_O`) — on InternVL3-78B, evaluated on 10 frame pairs per video | `PCS ∈ [0, 100]` (percentage) | `level1_data_generation/physical_consistency/` |
| **Level 1b** | Physics-law compliance on free-fall / horizontal-push videos: SAM2 mask trajectory + polynomial kinematic fit, gated by a 10 % VLM video-quality score | `PhysLawScore ∈ [0, 100]` (percentage) | `level1_data_generation/physics_law/` |
| **Level 2** | Action-following fidelity (task-completion rate + 5 visual-quality dimensions) | TCR; PP / MQ / TC / VS / OS scores | `level2_data_generation/` |
| **Level 3** | Optimism bias (does the model ignore action perturbations and still predict success?) | `Score = (1 − Y_rate) × 100` | `level3_data_generation/` + `evaluation/` |

Each `level{N}_data_generation/` subfolder ships its own `README.md`
covering dependencies, configuration and run commands. Please consult
the per-level README before running anything.

## Human-annotation corpus

`MiraBench_dataset/` is the dataset described in Section 4.6 / Appendix H
of the paper:

- **906** world-model-generated videos paired with **906** PII-stripped
  human-annotation JSONs.
- Four evaluation levels: `physical_consistency` (186),
  `physics_law_compliance` (90), `action_following_fidelity` (210),
  `optimism_bias_detection` (420).
- A derivative `parsed/` folder with four long-format CSVs that
  reproduces every headline number in the paper without re-parsing the
  raw JSONs.
- Annotation JSONs have been stripped of internal IDs / URLs /
  timestamps; see [`MiraBench_dataset/README.md`](MiraBench_dataset/README.md)
  for the full schema.

## Licence

The human annotations and parsed scores are released under **CC BY 4.0**
for academic use. The mp4 videos are derivative outputs of third-party
world-model checkpoints; please consult the original model licences
(DreamDojo, Wan2.1, HappyHorse, …) before redistributing the videos.
Code is released under the licence specified inside each subdirectory.

## Citation

If this repository helps your research, please cite the MiraBench
paper. The full BibTeX entry will be added once the paper is publicly
available.
