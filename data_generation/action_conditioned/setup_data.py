#!/usr/bin/env python3
"""
One-time setup: extract first frames, GT clips, and per-condition perturbed action arrays
for every task/episode in the action-conditioned benchmark.

Uses DreamDojo's native VideoActionDataset to get 384D normalized delta actions,
identical to the g9 human-annotation pipeline.

Output layout:
  tasks/<task>/
      task_info.json
      ep<N>_idx<M>/
          first_frame.jpg
          gt.mp4
          actions_per_condition/
              baseline.npy                          # (T-1, 384) DreamDojo-format actions
              implicit_grip_force_weak_s0.5.npy     # perturbed
              ...

Run from repo root:
  /path/to/your_env/DreamDojo/.venv/bin/python3 \
      data_generation_part3_action/setup_data.py
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DREAMDOJO_PATH = "/path/to/your_env/DreamDojo"

from module_a_ewmbench.bench_schedule import (
    TASK_REGISTRY, get_episode_indices, episodes_for_task,
    build_schedule, task_rng_seed,
)
from module_a_ewmbench.perturbations import perturb_actions

TASKS_DIR  = Path(__file__).resolve().parent / "tasks"
SEED       = 42
NUM_FRAMES = 77
FPS        = 20


# ── DreamDojo Native Dataloader ───────────────────────────────────────────────

_dataset_cache: dict = {}


def _get_dreamdojo_dataset(dataset_path: str, num_frames: int):
    """Cached DreamDojo VideoActionDataset — same pipeline as g9 generation."""
    key = (dataset_path, num_frames)
    if key in _dataset_cache:
        return _dataset_cache[key]

    if DREAMDOJO_PATH not in sys.path:
        sys.path.insert(0, DREAMDOJO_PATH)

    from groot_dreams.dataloader import VideoActionDataset

    print(f"  Loading DreamDojo dataset: {Path(dataset_path).name} …", flush=True)
    old_cwd = os.getcwd()
    os.chdir(DREAMDOJO_PATH)
    try:
        ds = VideoActionDataset(
            dataset_path=dataset_path,
            num_frames=num_frames,
            data_split="full",
            embodiment="gr1",
            single_base_index=True,
        )
    finally:
        os.chdir(old_cwd)
    _dataset_cache[key] = ds
    print(f"  Loaded: {len(ds)} episodes", flush=True)
    return ds


def get_actions(dataset_path: str, episode_idx: int, num_frames: int) -> np.ndarray:
    """Return (num_frames-1, 384) float32 via DreamDojo native pipeline."""
    ds = _get_dreamdojo_dataset(dataset_path, num_frames)
    traj_ids = ds.lerobot_dataset._trajectory_ids
    matches = np.where(traj_ids == episode_idx)[0]
    if len(matches) == 0:
        raise RuntimeError(f"Episode {episode_idx} not found in {Path(dataset_path).name}")
    data = ds[int(matches[0])]
    actions = data["action"].numpy().astype(np.float32)  # (num_frames-1, 384)
    n_need = num_frames - 1
    if len(actions) < n_need:
        pad = np.zeros((n_need - len(actions), actions.shape[1]), dtype=np.float32)
        actions = np.concatenate([actions, pad], axis=0)
    return actions[:n_need]


# ── Utilities ────────────────────────────────────────────────────────────────

def extract_first_frame(dataset_path: str, episode_idx: int, video_key: str | None) -> np.ndarray:
    ds = Path(dataset_path)
    if video_key is None:
        info_p = ds / "meta" / "info.json"
        if info_p.exists():
            for k, v in json.loads(info_p.read_text()).get("features", {}).items():
                if v.get("dtype") == "video":
                    video_key = k; break
    video_key = video_key or "observation.images.ego_view_freq20"
    chunk = episode_idx // 1000
    vp = ds / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_idx:06d}.mp4"
    if not vp.exists():
        vp = ds / "videos" / f"chunk-{chunk:03d}" / f"episode_{episode_idx:06d}.mp4"
    if not vp.exists():
        return np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        import av
        with av.open(str(vp)) as c:
            for frame in c.decode(video=0):
                return frame.to_ndarray(format="rgb24")
    except Exception:
        return np.zeros((480, 640, 3), dtype=np.uint8)


def extract_gt_clip(dataset_path: str, episode_idx: int, video_key: str | None) -> np.ndarray | None:
    """Return (T, H, W, 3) uint8 full-episode GT clip, or None on failure."""
    ds = Path(dataset_path)
    if video_key is None:
        info_p = ds / "meta" / "info.json"
        if info_p.exists():
            for k, v in json.loads(info_p.read_text()).get("features", {}).items():
                if v.get("dtype") == "video":
                    video_key = k; break
    video_key = video_key or "observation.images.ego_view_freq20"
    chunk = episode_idx // 1000
    vp = ds / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_idx:06d}.mp4"
    if not vp.exists():
        vp = ds / "videos" / f"chunk-{chunk:03d}" / f"episode_{episode_idx:06d}.mp4"
    if not vp.exists():
        return None
    try:
        import av
        frames = []
        with av.open(str(vp)) as c:
            for f in c.decode(video=0):
                frames.append(f.to_ndarray(format="rgb24"))
        return np.stack(frames) if frames else None
    except Exception:
        return None


def save_jpg(frame: np.ndarray, path: Path):
    from PIL import Image as PILImage
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(frame).save(str(path), quality=92)


def save_mp4(frames: np.ndarray, path: Path, fps: int = FPS):
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    H, W = frames.shape[1], frames.shape[2]
    with av.open(str(path), mode="w") as c:
        stream = c.add_stream("h264", rate=fps)
        stream.width = W; stream.height = H
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}
        for f in frames:
            vf = av.VideoFrame.from_ndarray(f, format="rgb24")
            for pkt in stream.encode(vf): c.mux(pkt)
        for pkt in stream.encode(): c.mux(pkt)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import torch

    total_eps = 0
    for task_label, task_info in TASK_REGISTRY.items():
        dataset_path = task_info["dataset"]
        video_key    = task_info.get("video_key")
        num_frames   = task_info.get("num_frames", NUM_FRAMES)
        n_ep        = episodes_for_task(task_label)
        ep_indices  = get_episode_indices(task_label)[:n_ep]
        sched       = build_schedule(task_label, n_ep, task_rng_seed(SEED, task_label))

        task_dir = TASKS_DIR / task_label
        task_dir.mkdir(parents=True, exist_ok=True)

        conditions = [(s[0], s[1], s[2]) for s in sched]  # (subdir, pert_type, severity)

        (task_dir / "task_info.json").write_text(json.dumps({
            "task_label":     task_label,
            "dataset_path":   dataset_path,
            "video_key":      video_key,
            "num_frames":     num_frames,
            "fps":            FPS,
            "episode_indices": ep_indices,
            "conditions":     [c[0] for c in conditions],
        }, indent=2))

        print(f"\n{'='*60}")
        print(f"Task: {task_label}  |  episodes: {ep_indices}")

        for ep_slot, ep_idx in enumerate(ep_indices):
            ep_dir     = task_dir / f"ep{ep_slot:03d}_idx{ep_idx}"
            acts_dir   = ep_dir / "actions_per_condition"
            ep_dir.mkdir(parents=True, exist_ok=True)
            acts_dir.mkdir(parents=True, exist_ok=True)

            print(f"  ep{ep_slot:03d} (idx={ep_idx}):", end="", flush=True)

            # First frame
            jpg_p = ep_dir / "first_frame.jpg"
            if not jpg_p.exists():
                frame = extract_first_frame(dataset_path, ep_idx, video_key)
                save_jpg(frame, jpg_p)
                print(" frame✓", end="", flush=True)
            else:
                print(" frame(skip)", end="", flush=True)

            # GT clip
            gt_p = ep_dir / "gt.mp4"
            if not gt_p.exists():
                gt_frames = extract_gt_clip(dataset_path, ep_idx, video_key)
                if gt_frames is not None:
                    save_mp4(gt_frames, gt_p, fps=FPS)
                    print(f" gt({len(gt_frames)}f)✓", end="", flush=True)
                else:
                    print(" gt(FAIL)", end="", flush=True)
            else:
                print(" gt(skip)", end="", flush=True)

            # Get baseline actions via DreamDojo native pipeline (same as g9 generation)
            baseline_384 = get_actions(dataset_path, ep_idx, num_frames)
            print(f" actions({baseline_384.shape[0]}×384)✓", end="", flush=True)

            # Save perturbed action arrays per condition — all in 384D DreamDojo format.
            for subdir_name, pert_type, severity in conditions:
                npy_p = acts_dir / f"{subdir_name}.npy"
                if npy_p.exists():
                    print(f" {subdir_name[:20]}(skip)", end="", flush=True)
                    continue
                if pert_type is None:
                    acts_out = baseline_384.copy()
                else:
                    acts_t = torch.from_numpy(baseline_384).unsqueeze(0).float()
                    acts_t = perturb_actions(acts_t, pert_type, severity=severity or 0.5)
                    acts_out = acts_t.squeeze(0).numpy()
                np.save(str(npy_p), acts_out)
                print("✓", end="", flush=True)
            print()

            total_eps += 1

    print(f"\n{'='*60}")
    print(f"Setup complete: {total_eps} episodes across {len(TASK_REGISTRY)} tasks")
    print(f"Data in: {TASKS_DIR}")


if __name__ == "__main__":
    main()
