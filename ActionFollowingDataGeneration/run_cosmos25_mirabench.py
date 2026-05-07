#!/usr/bin/env python3
"""
Cosmos-Predict2.5-14B I2V inference for Mirabench gr1 subset.
Adapted from teammate's run_cosmos14b.py — uses high-level Inference API
with Cosmos-Reason1-7B text encoder (public path).
"""
import os, sys, argparse, tempfile, shutil
from pathlib import Path

DREAMDOJO_DIR = "/mnt/users/zirui/mizirui_benchmark/DreamDojo"
os.chdir(DREAMDOJO_DIR)
sys.path.insert(0, DREAMDOJO_DIR)
sys.path.insert(0, os.path.join(DREAMDOJO_DIR, "packages/cosmos-oss"))

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── config ────────────────────────────────────────────────────────────────────
DATA_ROOT  = "/mnt/users/zirui/mizirui_benchmark/Mirabench_exp/video_batch_final"
OUT_ROOT   = "/mnt/users/zirui/mizirui_benchmark/Mirabench_exp/cosmos_14b"
HEIGHT, WIDTH = 480, 640
NUM_FRAMES = 97
GUIDANCE   = 1.0

# ── imports ─────────────────────────────────────────────���─────────────────────
from cosmos_oss.init import init_environment, cleanup_environment
init_environment()

# Patch CheckpointConfig.path to return local files instead of downloading from HF/S3.
# This preserves all other checkpoint metadata (experiment, s3.uri, etc.) while
# redirecting .path calls to local paths for all checkpoints we have locally.
_14B_CKPT    = "/mnt/public/models/nvidia/Cosmos-Predict2.5-14B/base/pre-trained/54937b8c-29de-4f04-862c-e67b04ec41e8_ema_bf16.pt"
_2B_CKPT     = "/mnt/public/models/nvidia/DreamDojo/2B_pretrain/iter_000140000/model_ema_bf16.pt"
_TOKENIZER   = "/mnt/public/models/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth"
_REASON1_DIR = "/mnt/public/models/nvidia/Cosmos-Reason1-7B"

_UUID_TO_LOCAL = {
    "54937b8c-29de-4f04-862c-e67b04ec41e8": _14B_CKPT,
    "e21d2a49-4747-44c8-ba44-9f6f9243715f": _14B_CKPT,
    "685afcaa-4de2-42fe-b7b9-69f7a2dee4d8": _TOKENIZER,
    "9c7b7da4-2d95-45bb-9cb8-2eed954e9736": "/tmp",          # guardrail, disabled
    "7219c6c7-f878-4137-bbdb-76842ea85e70": _REASON1_DIR,
    "cb3e3ffa-7b08-4c34-822d-61c7aa31a14f": _REASON1_DIR,
    "81edfebe-bd6a-4039-8c1d-737df1a790bf": _2B_CKPT,
    "d20b7120-df3e-4911-919d-db6e08bad31c": _2B_CKPT,
    "575edf0f-d973-4c74-b52c-69929a08d0a5": _2B_CKPT,
    "38c6c645-7d41-4560-8eeb-6f4ddc0e6574": _2B_CKPT,
    "524af350-2e43-496c-8590-3646ae1325da": _2B_CKPT,
    "f740321e-2cd6-4370-bbfe-545f4eca2065": _2B_CKPT,
}

from cosmos_predict2._src.imaginaire.utils.checkpoint_db import CheckpointConfig
# path is cached_property on lyg0270 — use .func not .fget
_cp = CheckpointConfig.path
_orig_path_func = _cp.func if hasattr(_cp, 'func') else _cp.fget

def _local_path(self):
    local = _UUID_TO_LOCAL.get(self.uuid)
    if local is not None:
        return local
    return _orig_path_func(self)

CheckpointConfig.path = property(_local_path)

from cosmos_predict2.config import InferenceArguments, InferenceType, SetupArguments
from cosmos_predict2.inference import Inference

import mediapy
from PIL import Image

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data-root",  default=DATA_ROOT)
parser.add_argument("--out-root",   default=OUT_ROOT)
parser.add_argument("--guidance",   type=float, default=GUIDANCE)
parser.add_argument("--num-frames", type=int,   default=NUM_FRAMES)
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--rank",       type=int,   default=0)
parser.add_argument("--world-size", type=int,   default=1)
parser.add_argument("--overwrite",  action="store_true")
parser.add_argument("--episode",    default=None, help="Run only e.g. 0034")
parser.add_argument("--subset",     choices=["gr1", "gen", "all"], default="gr1")
args = parser.parse_args()

data_root = Path(args.data_root)
out_root  = Path(args.out_root)

SUBSET_DIRS = {"gr1": "gr1", "gen": "generalizability"}
subsets = list(SUBSET_DIRS.keys()) if args.subset == "all" else [args.subset]

_EXPERIMENT_14B = "Stage-c_pt_4-reason_embeddings-v1p1-Index-43-Size-14B-Res-720-Fps-16_resume_from_reason1p1_rectified_flow_shift5_high_sigma"

# ── load model ────────────────────────────────────────────────────────────────
_tmpdir = tempfile.mkdtemp(prefix="cosmos25_frames_")
setup_args = SetupArguments(
    checkpoint_path=_14B_CKPT,
    experiment=_EXPERIMENT_14B,
    output_dir=out_root / "_tmp_cosmos_output",
    model="14B/pre-trained",
    context_parallel_size=1,
    disable_guardrails=True,
    offload_text_encoder=False,
    offload_tokenizer=False,
    offload_diffusion_model=False,
    offload_guardrail_models=True,
)
print("[Cosmos-Predict2.5-14B] loading model...")
inference = Inference(setup_args)
print("[Cosmos-Predict2.5-14B] model loaded.\n")

# ── episode loop ──────────────────────────────────────────────────────────────
for subset_key in subsets:
    subset_dir = SUBSET_DIRS[subset_key]
    episode_dirs = sorted((data_root / subset_dir).glob("episode_*"))
    episode_dirs = episode_dirs[args.rank::args.world_size]
    print(f"[{subset_key}] rank={args.rank}/{args.world_size}  {len(episode_dirs)} episodes assigned")

    for ep_dir in episode_dirs:
        ep_id = ep_dir.name.split("_")[-1]

        if args.episode and ep_id != args.episode:
            continue

        out_dir = out_root / subset_dir / f"episode_{ep_id}"
        out_mp4 = out_dir / "pred.mp4"
        if out_mp4.exists() and not args.overwrite:
            print(f"[skip] {out_mp4}")
            continue

        mp4_path = ep_dir / f"gt_{ep_id}.mp4"
        txt_path = ep_dir / f"instruct_{ep_id}.txt"
        if not mp4_path.exists():
            print(f"[warn] missing {mp4_path}, skip")
            continue

        prompt = txt_path.read_text().strip() if txt_path.is_file() else ""
        out_dir.mkdir(parents=True, exist_ok=True)

        # extract first frame
        frames = mediapy.read_video(str(mp4_path))
        first_frame = Image.fromarray(frames[0]).convert("RGB").resize((WIDTH, HEIGHT))
        tmp_png = os.path.join(_tmpdir, f"ep{ep_id}_frame0.png")
        first_frame.save(tmp_png)

        print(f"[run] {subset_key}/episode_{ep_id}  prompt: '{prompt[:60]}...'")

        inf_args = InferenceArguments(
            name=f"episode_{ep_id}_pred",
            inference_type=InferenceType.IMAGE2WORLD,
            input_path=Path(tmp_png),
            prompt=prompt,
            negative_prompt="",
            seed=args.seed,
            guidance=args.guidance,
            num_output_frames=args.num_frames,
            num_steps=35,
            resolution=f"{HEIGHT},{WIDTH}",
        )

        result_path = inference._generate_sample(inf_args, out_dir)
        if result_path:
            result_path = Path(result_path)
            if result_path != out_mp4:
                result_path.rename(out_mp4)
            print(f"[done] saved {out_mp4}")
        else:
            print(f"[error] generation failed for {subset_key}/episode_{ep_id}")

shutil.rmtree(_tmpdir, ignore_errors=True)
cleanup_environment()
print("\n[ALL DONE]")
