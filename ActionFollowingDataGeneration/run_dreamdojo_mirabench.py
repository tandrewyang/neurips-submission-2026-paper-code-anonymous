#!/usr/bin/env python3
"""
DreamDojo inference on video_batch_final dataset (Mirabench format).

Input per episode (video_batch_final/{subset}/episode_XXXX/):
  - First frame:  gt_XXXX.mp4        (first frame extracted)
  - Actions:      ra_XXXX.npy        (variable shape, see mapping below)
  - Text prompt:  instruct_XXXX.txt  (expanded instruction)

Action mapping (gen subset, built by extract_gen_npy.py):
  ep0000-0004  DROID real          shape=(T, 30)  copied from droid_real_5/video_batch
  ep0005-0009  GR1 robot           shape=(T, 29)  copied from video_batch_gr1
  ep0010-0014  Lingchu glove-orig  shape=(T, 74)  left_wrist(7)+right_wrist(7)+left_qpos(30)+right_qpos(30)

Action mapping (gr1 subset, 50 episodes):
  all episodes  GR1 robot          shape=(T, 29)

All actions are zero-padded to 384D (first N dims filled, rest = 0),
matching the embed_nd_to_384d convention used in run_g1_galbot_batch.py.

Must be run from: /mnt/users/zirui/mizirui_benchmark/DreamDojo/
(config_file paths are relative to that directory)
"""

import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("COSMOS_FORCE_OFFLINE", "1")

DREAMDOJO_DIR = Path("/mnt/users/zirui/mizirui_benchmark/DreamDojo")
sys.path.insert(0, str(DREAMDOJO_DIR))

import numpy as np
import torch
import torchvision
import mediapy

from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference

WAN_VAE_PATH = Path("/mnt/public/models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth")

MODEL_CONFIGS = {
    "2b": {
        "checkpoint": DREAMDOJO_DIR / "checkpoints/DreamDojo/2B_GR1_post-train/iter_000050000/model_ema_bf16.pt",
        "experiment": "dreamdojo_2b_480_640_gr1",
        "config_file": "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        "output_suffix": "2b",
    },
    "14b": {
        "checkpoint": DREAMDOJO_DIR / "checkpoints/DreamDojo/14B_GR1_post-train/iter_000050000/model_ema_bf16.pt",
        "experiment": "dreamdojo_14b_480_640_gr1",
        "config_file": "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        "output_suffix": "14b",
    },
    "2b_pretrain": {
        "checkpoint": DREAMDOJO_DIR / "checkpoints/DreamDojo/2B_pretrain/iter_000140000/model_ema_bf16.pt",
        "experiment": "dreamdojo_2b_480_640_gr1",
        "config_file": "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        "output_suffix": "2b_pretrain",
    },
    "14b_pretrain": {
        "checkpoint": DREAMDOJO_DIR / "checkpoints/DreamDojo/14B_pretrain/iter_000140000/model_ema_bf16.pt",
        "experiment": "dreamdojo_14b_480_640_gr1",
        "config_file": "cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py",
        "output_suffix": "14b_pretrain",
    },
}

CHUNK_SIZE = 12
ACTION_DIM = 384
SAVE_FPS = 10
NUM_LATENT_COND_FRAMES = 1
RESOLUTION = "480,640"


def load_wan_vae_weights(model, vae_path: Path) -> None:
    if not vae_path.is_file():
        raise FileNotFoundError(f"WAN VAE checkpoint not found: {vae_path}")
    print(f"[VAE] Loading from: {vae_path}")
    state = torch.load(str(vae_path), map_location="cpu", weights_only=True)
    tok = model.model.tokenizer.model
    inner = tok.model
    missing, unexpected = inner.load_state_dict(state, strict=False)
    print(f"[VAE] missing={len(missing)} unexpected={len(unexpected)}")
    device = next(model.model.net.parameters()).device
    dtype = next(model.model.net.parameters()).dtype
    inner.to(device=device, dtype=dtype)
    print(f"[VAE] Moved to device={device} dtype={dtype}")


def embed_nd_to_384d(actions: np.ndarray, action_scale: float = 1.0) -> np.ndarray:
    """Zero-pad (T, D) actions to (T, 384) — first D dims filled, rest = 0.

    Handles all three gen action types:
      DROID (30D), GR1 (29D), Lingchu (74D)
    and the gr1 subset (29D).
    Matches embed_nd_to_384d convention from run_g1_galbot_batch.py.
    """
    T, D = actions.shape
    if D > ACTION_DIM:
        raise ValueError(f"Action dim {D} exceeds ACTION_DIM={ACTION_DIM}")
    padded = np.zeros((T, ACTION_DIM), dtype=np.float32)
    padded[:, :D] = (actions * action_scale).astype(np.float32)
    return padded


def get_first_frame(mp4_path: Path, resize_hw=(480, 640)) -> np.ndarray:
    video = mediapy.read_video(str(mp4_path))
    if len(video) == 0:
        raise RuntimeError(f"Empty video: {mp4_path}")
    frame = video[0]
    if frame.shape[:2] != resize_hw:
        frame = mediapy.resize_image(frame, resize_hw)
    return frame


def frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
    t = torchvision.transforms.functional.to_tensor(frame).unsqueeze(0) * 255.0
    return t.to(torch.uint8)


def build_vid_input(cond_frame_t: torch.Tensor, num_frames: int) -> torch.Tensor:
    zeros = torch.zeros_like(cond_frame_t).repeat(num_frames - 1, 1, 1, 1)
    vid = torch.cat([cond_frame_t, zeros], dim=0)
    return vid.unsqueeze(0).permute(0, 2, 1, 3, 4)


def run_episode(episode_dir: Path, model, args, episode_idx: str, out_root: Path) -> None:
    print(f"\n{'='*60}")
    print(f"[{episode_idx}] {episode_dir}")

    # --- Input files ---
    mp4_path = episode_dir / f"gt_{episode_idx}.mp4"
    npy_path = episode_dir / f"ra_{episode_idx}.npy"
    txt_path = episode_dir / f"instruct_{episode_idx}.txt"

    if not mp4_path.exists():
        print(f"  WARNING: {mp4_path} not found, skipping.")
        return

    # --- Output path ---
    out_dir = out_root / episode_dir.parent.name / episode_dir.name
    out_path = out_dir / f"pred_{args.model_size}.mp4"
    if out_path.exists() and not args.overwrite:
        print(f"  SKIP (already exists): {out_path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- First frame ---
    init_frame = get_first_frame(mp4_path)
    print(f"  first frame: {init_frame.shape}")

    # --- Actions ---
    actions_raw = np.load(str(npy_path))
    print(f"  actions raw: {actions_raw.shape} range=[{actions_raw.min():.4f}, {actions_raw.max():.4f}]")
    actions_384 = embed_nd_to_384d(actions_raw)

    # --- Text prompt ---
    prompt = txt_path.read_text().strip() if txt_path.is_file() else ""
    print(f"  prompt: '{prompt[:80]}...'")

    # --- Chunked autoregressive generation ---
    T = actions_384.shape[0]
    n_chunks = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    if args.max_chunks > 0:
        n_chunks = min(n_chunks, args.max_chunks)
    print(f"  {n_chunks} chunks × {CHUNK_SIZE} frames")

    all_frames = []
    prev_frame = init_frame

    for ci in range(n_chunks):
        t0 = ci * CHUNK_SIZE
        t1 = t0 + CHUNK_SIZE
        chunk_actions = actions_384[t0:t1]

        if chunk_actions.shape[0] < CHUNK_SIZE:
            pad = np.zeros((CHUNK_SIZE - chunk_actions.shape[0], ACTION_DIM), dtype=np.float32)
            chunk_actions = np.concatenate([chunk_actions, pad], axis=0)

        cond_t = frame_to_tensor(prev_frame)
        num_frames = chunk_actions.shape[0] + 1
        vid_input = build_vid_input(cond_t, num_frames)
        action_t = torch.from_numpy(chunk_actions).float()

        print(f"  chunk {ci+1}/{n_chunks}: actions[{t0}:{t1}]")

        out = model.generate_vid2world(
            prompt=prompt,
            input_path=vid_input,
            action=action_t,
            guidance=args.guidance,
            num_video_frames=num_frames,
            num_latent_conditional_frames=NUM_LATENT_COND_FRAMES,
            resolution=RESOLUTION,
            seed=args.seed + ci,
            negative_prompt="",
            lam_video=None,
        )

        out_norm = (out - (-1.0)) / 2.0
        out_u8 = (torch.clamp(out_norm[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        prev_frame = out_u8[-1]
        all_frames.append(out_u8)
        print(f"    chunk {ci+1} done: {out_u8.shape} mean={out_u8.mean():.1f}")

    if len(all_frames) == 0:
        return
    full_video = all_frames[0] if len(all_frames) == 1 else np.concatenate(
        [all_frames[0]] + [f[1:] for f in all_frames[1:]], axis=0
    )

    mediapy.write_video(str(out_path), full_video, fps=SAVE_FPS)
    print(f"  saved {full_video.shape[0]} frames → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="DreamDojo inference on Mirabench video_batch_final")
    parser.add_argument("--model-size", choices=["2b", "14b", "2b_pretrain", "14b_pretrain"], required=True)
    parser.add_argument("--subset", choices=["gen", "gr1", "all"], default="all",
                        help="Which subset to process: gen (15 eps), gr1 (50 eps), or all (65 eps)")
    parser.add_argument("--data-root", type=str,
                        default="/mnt/users/zirui/mizirui_benchmark/World_Fufu/ActionFollowing/video_batch_final")
    parser.add_argument("--out-root", type=str,
                        default="/mnt/users/zirui/mizirui_benchmark/Mirabench_exp")
    parser.add_argument("--max-chunks", type=int, default=0, help="0 = all chunks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_size]
    checkpoint = cfg["checkpoint"]
    out_root = Path(args.out_root) / f"dreamdojo_{args.model_size}"
    data_root = Path(args.data_root)

    # Switch to DreamDojo dir so relative paths (config_file) resolve correctly
    os.chdir(str(DREAMDOJO_DIR))

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"[DreamDojo {args.model_size.upper()}] Loading model from {checkpoint}")
    model = Video2WorldInference(
        experiment_name=cfg["experiment"],
        ckpt_path=str(checkpoint),
        s3_credential_path="",
        context_parallel_size=1,
        config_file=cfg["config_file"],
    )
    print("[DreamDojo] Model loaded. Injecting VAE weights...")
    load_wan_vae_weights(model, WAN_VAE_PATH)

    # Collect episode dirs
    subset_map = {
        "gen": ["generalizability"],
        "gr1": ["gr1"],
        "all": ["generalizability", "gr1"],
    }
    subsets = subset_map[args.subset]

    for subset_name in subsets:
        subset_dir = data_root / subset_name
        episode_dirs = sorted(subset_dir.iterdir()) if subset_dir.is_dir() else []
        print(f"\n[{subset_name}] {len(episode_dirs)} episodes")

        for ep_dir in episode_dirs:
            if not ep_dir.is_dir():
                continue
            # Extract numeric index from dirname like "episode_0000"
            ep_num = ep_dir.name.split("_")[-1]  # "0000"
            run_episode(ep_dir, model, args, ep_num, out_root)

    model.cleanup()
    print("\n[DONE] All episodes processed.")


if __name__ == "__main__":
    main()
