# MiraBench Human Annotation Dataset

Human annotations released alongside the paper *"MiraBench: Evaluating
Action-Conditioned Reliability in Robotic World Models"* (NeurIPS 2026
submission).

This release contains **906 video–annotation pairs** covering four
evaluation dimensions of robotic world-model output. Each pair is a
short prediction video produced by a world model plus a structured human
annotation describing what the annotator observed.

## Contents

```
MiraBench_dataset/
├── README.md                              ← this file
├── physical_consistency/                  ← 186 videos (4 models)
│   ├── 0421/{g1,occ68,pnp5,pour20}/
│   ├── dreamdojo_2b/dreamdojo_2b/
│   ├── happy_horse/happy_horse/
│   └── wan2.1_14B/wan2.1_14B/
├── physics_law_compliance/                ← 90 videos (free-fall scenes)
│   ├── 0421/{banana,bottle}/
│   └── physics_law_2/
├── action_following_fidelity/             ← 210 videos
│   ├── flat/                                  # short flat-scene tasks
│   └── video_batch_gr1/episode_*/             # GR1 episode clips
├── optimism_bias_detection/               ← 420 videos
│   ├── human_annotation_dreamdojo_2b_gr1/videos/task_*/
│   ├── human_annotation_happyhorse_i2v/videos/task_*/
│   ├── human_annotation_wan21_i2v_14b/videos/task_*/
│   ├── optimism-{1,2}/videos/task_*/
│   └── optimism-3/task_*/
└── parsed/
    ├── physical_consistency.csv           ← long-format human scores
    ├── physics_law_compliance.csv
    ├── action_following_fidelity.csv
    ├── optimism_bias_detection.csv
    └── summary.txt
```

Every leaf folder contains pairs of `<stem>.mp4` (the world-model output)
and `<stem>.json` (the human annotation).

## The four dimensions

| Subfolder | Dimension | # videos | What annotators score |
|---|---|---|---|
| `physical_consistency` | Physical Consistency (PC) | 186 | 18 indicators across appearance / motion / occlusion / interaction / environment + a holistic 1–5 rating. Severity scale `A=1.00 / B=0.67 / C=0.33 / D=0.00`. |
| `physics_law_compliance` | Physics-Law Compliance (PL) | 90 | 18 free-fall sub-anomalies (release, free-fall, landing, bounce, appearance) + a 4-tier overall score (A/B/C/D = 4/3/2/1). |
| `action_following_fidelity` | Action-Following Fidelity (AF) | 210 | Task completion (TCR), per-object recognition (OPS), and 5-tier visual-quality scores (PP/MQ/TC/VS/OS). |
| `optimism_bias_detection` | Optimism-Bias Detection (OB) | 420 | 18 dimensions across two modules (MA: model awareness of perturbation; MB: task-success prediction reliability) — captures whether the world model "optimistically" predicts success that ground truth doesn't show. |

The four CSVs in `parsed/` are derivative long-format tables produced by
`parse.py` from the raw JSON; one row per `(video, indicator)`. They
include the score-mapping (raw answer text → grade letter → numeric
score) so you can reproduce the headline statistics without re-parsing.

## Annotation JSON schema

Every JSON looks like this (only the answer-bearing fields are shown):

```jsonc
{
  "markData": {
    "annotations": [],
    "notes": [],
    "tagsItems": [],
    "videoQuality": {
      "name": "物理一致性",          // dimension label
      "status": 1,                    // 1 = annotation completed
      "items": [                      // questionnaire definition
        {
          "id": 19872,                // indicator key
          "type": 1,                  // 1=single-select, 4=multi-select
          "must": 1,                  // 1=required
          "title": "颜色稳定性",
          "desc":  "主要物体的颜色在整个视频中是否保持稳定？",
          "option": "[{\"value\":\"是（无违规）...\"}, ...]"  // JSON-string
        },
        ...
      ],
      "question": {                   // annotator's chosen answer per item
        "19872": "轻微违规：1-2 帧有轻微色偏...",
        ...
      }
    }
  }
}
```

The `option` field is a JSON-encoded string (legacy of the labelling
platform). Decode it once with `json.loads(item["option"])` to get the
list of `{value: <option text>}` dicts.

The `question` map is keyed by `item.id` (string). Multi-select answers
use `/` as a delimiter — `"a/b/c/"` means three boxes were ticked.

For the full indicator → score mapping, see the `parsed/*.csv` columns
`code`, `family`, `scheme`, `grade`, `score`.

## Models

The `physical_consistency` and `optimism_bias_detection` folders each
contain videos from multiple world models in parallel sub-trees. The
mapping is:

| Folder prefix | Model |
|---|---|
| `0421/` (in PC, PL, AF) | DreamDojo-14B |
| `dreamdojo_2b/`, `human_annotation_dreamdojo_2b_gr1/` | DreamDojo-2B |
| `happy_horse/`, `human_annotation_happyhorse_i2v/` | Happy Horse (i2v) |
| `wan2.1_14B/`, `human_annotation_wan21_i2v_14b/` | Wan2.1-14B (i2v) |
| `optimism-1/`, `optimism-2/`, `optimism-3/` | mixed / unspecified |

## Privacy

The released JSONs have been stripped of:
- internal record IDs (`markData.lid`, `videoQuality.id`)
- internal source URLs (`markData.url`)
- annotator and project UIDs (`videoQuality.uid`, `items[*].uid`)
- annotation timestamps (`videoQuality.ctime`, `items[*].ctime`)
- questionnaire group IDs (`items[*].ng_id`)

Only questionnaire structure (titles, descriptions, options, required
flags) and annotator answers are retained. No annotator-identifying
information is present.

## License

The annotations and parsed scores are released for research use under
**CC BY 4.0**. The mp4 videos are derivative outputs of third-party
world-model checkpoints; please consult the original model licenses
(DreamDojo, HappyHorse, Wan2.1) before redistribution of the videos.

## Citation

If you use this dataset, please cite the MiraBench paper. The full
citation will be added once the paper is publicly available.
