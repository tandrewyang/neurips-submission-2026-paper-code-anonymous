#!/usr/bin/env python3
"""
Cosmos-Predict2-14B I2V inference on Mirabench gr1 subset.

Model: /mnt/public/models/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1/model-480p-16fps.pt
Pure I2V — no action conditioning. First frame + text prompt → video.
Chunked autoregressive: last generated frame of chunk N conditions chunk N+1.
n_chunks determined by action sequence length (ra_XXXX.npy) to match DreamDojo video length.

Output: {out_root}/cosmos_14b/gr1/episode_XXXX/pred.mp4
Must be run from: /mnt/users/zirui/mizirui_benchmark/DreamDojo/
"""
import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("COSMOS_FORCE_OFFLINE", "1")

DREAMDOJO_DIR = Path("/mnt/users/zirui/mizirui_benchmark/DreamDojo")
sys.path.insert(0, str(DREAMDOJO_DIR))

import numpy as np
import torch
import torchvision
import mediapy

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

# video2world imports get_text_embedding into its own namespace at import time.
# Patch that reference directly so no T5 model is loaded (zeros of correct 1024-dim shape).
import cosmos_predict2._src.predict2.inference.video2world as _v2w
def _zero_text_embedding(prompts, **kwargs):
    if isinstance(prompts, str):
        prompts = [prompts]
    return torch.zeros(len(prompts), 512, 1024, device="cuda", dtype=torch.bfloat16)
_v2w.get_text_embedding = _zero_text_embedding

CHECKPOINT   = Path("/mnt/public/models/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1/model-480p-16fps.pt")
GR1_VAE_PATH = Path("/mnt/public/models/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1/tokenizer/tokenizer.pth")
EXPERIMENT   = "Stage-c_pt_4-Index-3-Size-14B-Res-480-Fps-16-Note-video_data1pt3_augmentor"
CONFIG_FILE  = "cosmos_predict2/_src/predict2/configs/video2world/config.py"

CHUNK_SIZE             = 12   # new frames generated per chunk
NUM_LATENT_COND_FRAMES = 1    # pure I2V: only first frame conditioned
RESOLUTION             = "480,640"
SAVE_FPS               = 10
ACTION_DIM             = 384  # unused for inference, only to determine n_chunks


def load_gr1_vae_weights(model: Video2WorldInference, vae_path: Path) -> None:
    if not vae_path.is_file():
        raise FileNotFoundError(f"GR1 VAE not found: {vae_path}")
    print(f"[VAE] loading GR1 tokenizer from {vae_path}")
    state = torch.load(str(vae_path), map_location="cpu", weights_only=True)
    tok   = model.model.tokenizer.model
    inner = tok.model
    missing, unexpected = inner.load_state_dict(state, strict=False)
    print(f"[VAE] missing={len(missing)} unexpected={len(unexpected)}")
    device = next(model.model.net.parameters()).device
    dtype  = next(model.model.net.parameters()).dtype
    inner.to(device=device, dtype=dtype)
    print(f"[VAE] moved to device={device} dtype={dtype}")


def get_first_frame(mp4_path: Path, resize_hw=(480, 640)) -> np.ndarray:
    video = mediapy.read_video(str(mp4_path))
    if len(video) == 0:
        raise RuntimeError(f"empty video: {mp4_path}")
    frame = video[0]
    if frame.shape[:2] != resize_hw:
        frame = mediapy.resize_image(frame, resize_hw)
    return frame


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    """(H,W,C) uint8 → (1,C,H,W) uint8"""
    return (torchvision.transforms.functional.to_tensor(frame).unsqueeze(0) * 255.0).to(torch.uint8)


def build_vid_input(cond_frame_t: torch.Tensor, num_frames: int) -> torch.Tensor:
    """(1,C,H,W) → (1,C,T,H,W) uint8: first frame real, rest zeros."""
    zeros = torch.zeros_like(cond_frame_t).repeat(num_frames - 1, 1, 1, 1)
    vid   = torch.cat([cond_frame_t, zeros], dim=0)
    return vid.unsqueeze(0).permute(0, 2, 1, 3, 4)


def run_episode(episode_dir: Path, model: Video2WorldInference, args, ep_idx: str, out_root: Path) -> None:
    print(f"\n{'='*60}")
    print(f"[{ep_idx}] {episode_dir}")

    mp4_path = episode_dir / f"gt_{ep_idx}.mp4"
    npy_path = episode_dir / f"ra_{ep_idx}.npy"
    txt_path = episode_dir / f"instruct_{ep_idx}.txt"

    if not mp4_path.exists():
        print(f"  SKIP: {mp4_path} not found")
        return

    out_dir  = out_root / episode_dir.parent.name / episode_dir.name
    out_path = out_dir / "pred.mp4"
    if out_path.exists() and not args.overwrite:
        print(f"  SKIP (exists): {out_path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # first frame
    res_hw = tuple(int(x) for x in args.resolution.split(","))
    init_frame = get_first_frame(mp4_path, resize_hw=res_hw)
    print(f"  first frame: {init_frame.shape}")

    # prompt
    prompt = txt_path.read_text().strip() if txt_path.is_file() else ""
    print(f"  prompt: '{prompt[:80]}...'")

    # n_chunks from action sequence length (for fair length comparison with DreamDojo)
    if npy_path.exists():
        T       = np.load(str(npy_path)).shape[0]
        n_chunks = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    else:
        n_chunks = 5  # fallback: ~60 frames
    if args.max_chunks > 0:
        n_chunks = min(n_chunks, args.max_chunks)
    print(f"  {n_chunks} chunks × {CHUNK_SIZE} frames")

    all_frames = []
    prev_frame = init_frame

    for ci in range(n_chunks):
        cond_t    = frame_to_tensor(prev_frame)
        num_frames = CHUNK_SIZE + 1          # 1 cond + 12 generated
        vid_input = build_vid_input(cond_t, num_frames)

        print(f"  chunk {ci+1}/{n_chunks}")

        out = model.generate_vid2world(
            prompt=prompt,
            input_path=vid_input,
            guidance=args.guidance,
            num_video_frames=num_frames,
            num_latent_conditional_frames=NUM_LATENT_COND_FRAMES,
            resolution=args.resolution,
            seed=args.seed + ci,
            negative_prompt="",
            lam_video=None,
        )

        # out: (1, C, T, H, W) float in [-1, 1]
        out_norm = (out - (-1.0)) / 2.0
        out_u8   = (torch.clamp(out_norm[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        prev_frame = out_u8[-1]   # last generated frame → next chunk's conditioning
        all_frames.append(out_u8)
        print(f"    chunk {ci+1} done: {out_u8.shape}  mean={out_u8.mean():.1f}")

    if not all_frames:
        return

    # stitch: keep all of chunk 0, drop first (conditioning) frame from chunks 1+
    full_video = all_frames[0] if len(all_frames) == 1 else np.concatenate(
        [all_frames[0]] + [f[1:] for f in all_frames[1:]], axis=0
    )

    mediapy.write_video(str(out_path), full_video, fps=SAVE_FPS)
    print(f"  saved {full_video.shape[0]} frames → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Cosmos-Predict2-14B I2V on Mirabench gr1")
    parser.add_argument("--data-root", default="/mnt/users/zirui/mizirui_benchmark/Mirabench_exp/video_batch_final")
    parser.add_argument("--out-root",  default="/mnt/users/zirui/mizirui_benchmark/Mirabench_exp")
    parser.add_argument("--max-chunks", type=int, default=0, help="0 = all")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--guidance",   type=float, default=1.0)
    parser.add_argument("--resolution", default=RESOLUTION, help="H,W e.g. 480,640 or 800,1280")
    parser.add_argument("--overwrite",  action="store_true")
    parser.add_argument("--rank",       type=int, default=0, help="Worker rank for parallel runs")
    parser.add_argument("--world-size", type=int, default=1, help="Total workers for parallel runs")
    parser.add_argument("--episode",    default=None, help="Run only this episode number, e.g. 0034")
    args = parser.parse_args()

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not GR1_VAE_PATH.is_file():
        raise FileNotFoundError(f"GR1 VAE not found: {GR1_VAE_PATH}")

    os.chdir(str(DREAMDOJO_DIR))

    out_root = Path(args.out_root) / "cosmos_14b"
    data_root = Path(args.data_root)

    print(f"[Cosmos-14B I2V] checkpoint: {CHECKPOINT}")
    print("[Cosmos-14B I2V] loading model...")
    model = Video2WorldInference(
        experiment_name=EXPERIMENT,
        ckpt_path=str(CHECKPOINT),
        s3_credential_path="",
        context_parallel_size=1,
        config_file=CONFIG_FILE,
    )
    print("[Cosmos-14B I2V] model loaded. Injecting GR1 VAE weights...")
    load_gr1_vae_weights(model, GR1_VAE_PATH)

    # gr1 subset only
    subset_dir   = data_root / "gr1"
    episode_dirs = sorted(ep for ep in subset_dir.iterdir() if ep.is_dir()) if subset_dir.is_dir() else []

    # interleaved sharding for parallel workers
    episode_dirs = episode_dirs[args.rank::args.world_size]
    print(f"\n[gr1] rank={args.rank}/{args.world_size}  {len(episode_dirs)} episodes assigned")

    for ep_dir in episode_dirs:
        ep_num = ep_dir.name.split("_")[-1]
        if args.episode and ep_num != args.episode:
            continue
        run_episode(ep_dir, model, args, ep_num, out_root)

    model.cleanup()
    print("\n[DONE]")


if __name__ == "__main__":
    main()
