#!/usr/bin/env python3
"""
One-time setup: extract first frames, GT clips, and per-episode prompts
for every task/episode in the text-instruction benchmark.

Output layout:
  tasks/<task>/
      task_info.json
      ep<N>_idx<M>/
          first_frame.jpg
          gt.mp4
          prompts.json   # {condition: prompt}

Run from repo root:
  /path/to/your_env/DreamDojo/.venv/bin/python3 \
      data_generation_part3_instruction/setup_data.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from module_a_ewmbench.bench_schedule import (
    TASK_REGISTRY, get_episode_indices, episodes_for_task,
    build_schedule, task_rng_seed,
)

TASKS_DIR   = Path(__file__).resolve().parent / "tasks"
PROMPTS_SRC = REPO_ROOT / "data" / "benchmark_prompts.json"
SEED        = 42
NUM_FRAMES  = 77   # GR1 clip length
FPS         = 20


# ── Utilities ────────────────────────────────────────────────────────────────

def extract_first_frame(dataset_path: str, episode_idx: int, video_key: str | None) -> "np.ndarray":
    import numpy as np
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
        print(f"  [WARN] video not found: {vp}")
        return np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        import av
        with av.open(str(vp)) as c:
            for frame in c.decode(video=0):
                return frame.to_ndarray(format="rgb24")
    except Exception as e:
        print(f"  [WARN] {e}")
        return np.zeros((480, 640, 3), dtype=np.uint8)


def extract_gt_clip(dataset_path: str, episode_idx: int, video_key: str | None) -> "np.ndarray | None":
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
        print(f"  [WARN] GT video not found: {vp}")
        return None
    try:
        import av, numpy as np
        frames = []
        with av.open(str(vp)) as c:
            for f in c.decode(video=0):
                frames.append(f.to_ndarray(format="rgb24"))
        if not frames:
            return None
        arr = np.stack(frames)  # (T, H, W, 3) — full episode
        return arr
    except Exception as e:
        print(f"  [WARN] cannot read GT clip {vp}: {e}")
        return None


def save_jpg(frame: "np.ndarray", path: Path):
    from PIL import Image as PILImage
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(frame).save(str(path), quality=92)


def save_mp4(frames: "np.ndarray", path: Path, fps: int = FPS):
    """Save (T, H, W, 3) uint8 → mp4 using av."""
    import av, numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    H, W = frames.shape[1], frames.shape[2]
    with av.open(str(path), mode="w") as c:
        stream = c.add_stream("h264", rate=fps)
        stream.width  = W
        stream.height = H
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}
        for f in frames:
            vf = av.VideoFrame.from_ndarray(f, format="rgb24")
            for pkt in stream.encode(vf):
                c.mux(pkt)
        for pkt in stream.encode():
            c.mux(pkt)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not PROMPTS_SRC.exists():
        print(f"ERROR: {PROMPTS_SRC} not found. Run scripts/build_benchmark_prompts.py first.")
        sys.exit(1)

    all_prompts = json.loads(PROMPTS_SRC.read_text())

    total_eps = 0
    for task_label, task_info in TASK_REGISTRY.items():
        dataset_path = task_info["dataset"]
        video_key    = task_info.get("video_key")
        num_frames   = task_info.get("num_frames", NUM_FRAMES)
        n_ep  = episodes_for_task(task_label)
        ep_indices  = get_episode_indices(task_label)[:n_ep]
        sched = build_schedule(task_label, n_ep, task_rng_seed(SEED, task_label))
        conditions  = [s[0] for s in sched]

        task_dir = TASKS_DIR / task_label
        task_dir.mkdir(parents=True, exist_ok=True)

        # task_info.json
        (task_dir / "task_info.json").write_text(json.dumps({
            "task_label":    task_label,
            "dataset_path":  dataset_path,
            "video_key":     video_key,
            "num_frames":    num_frames,
            "fps":           FPS,
            "episode_indices": ep_indices,
            "conditions":    conditions,
        }, indent=2))

        print(f"\n{'='*60}")
        print(f"Task: {task_label}  |  episodes: {ep_indices}")
        print(f"  Conditions: {conditions}")

        for ep_slot, ep_idx in enumerate(ep_indices):
            ep_dir = task_dir / f"ep{ep_slot:03d}_idx{ep_idx}"
            ep_dir.mkdir(parents=True, exist_ok=True)

            jpg_path = ep_dir / "first_frame.jpg"
            gt_path  = ep_dir / "gt.mp4"
            pr_path  = ep_dir / "prompts.json"

            print(f"  ep{ep_slot:03d} (idx={ep_idx}):", end="", flush=True)

            # First frame
            if not jpg_path.exists():
                frame = extract_first_frame(dataset_path, ep_idx, video_key)
                save_jpg(frame, jpg_path)
                print(" frame✓", end="", flush=True)
            else:
                print(" frame(skip)", end="", flush=True)

            # GT clip
            if not gt_path.exists():
                gt_frames = extract_gt_clip(dataset_path, ep_idx, video_key)
                if gt_frames is not None:
                    save_mp4(gt_frames, gt_path, fps=FPS)
                    print(f" gt({len(gt_frames)}f)✓", end="", flush=True)
                else:
                    print(" gt(FAIL)", end="", flush=True)
            else:
                print(" gt(skip)", end="", flush=True)

            # Prompts
            ep_prompts = all_prompts.get(task_label, {}).get(str(ep_idx), {})
            # Keep only conditions in the schedule
            ep_prompts_filtered = {c: ep_prompts.get(c, "") for c in conditions}
            pr_path.write_text(json.dumps(ep_prompts_filtered, indent=2))
            n_filled = sum(1 for v in ep_prompts_filtered.values() if v)
            print(f" prompts({n_filled}/{len(conditions)})✓")

            total_eps += 1

    print(f"\n{'='*60}")
    print(f"Setup complete: {total_eps} episodes across {len(TASK_REGISTRY)} tasks")
    print(f"Data in: {TASKS_DIR}")


if __name__ == "__main__":
    main()
