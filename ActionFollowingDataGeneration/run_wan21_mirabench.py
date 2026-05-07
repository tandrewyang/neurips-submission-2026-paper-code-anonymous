#!/usr/bin/env python3
"""
Wan2.1 I2V inference for Mirabench.
Clones the Wan2.1 repo if needed, then runs generate.py per episode.
"""
import os
import sys
import subprocess
import argparse
import shutil
import cv2
from pathlib import Path

DATA_ROOT   = "/mnt/users/zirui/mizirui_benchmark/Mirabench_exp/video_batch_final"
OUT_ROOT    = "/mnt/users/zirui/mizirui_benchmark/Mirabench_exp/wan2.1"
WAN21_REPO  = "/mnt/users/zirui/mizirui_benchmark/Wan2.1"
CKPT_DIR    = "/mnt/public/models/Wan-AI/Wan2.1-I2V-14B-720P"
SUBSET_MAP  = {"gen": "generalizability", "gr1": "gr1"}
# Portrait 480×640 – matches ground-truth resolution
VIDEO_SIZE  = "480*832"
NUM_FRAMES  = 81   # ~8 s @ ~10 fps (Wan default sampling rate)


def clone_wan21_repo():
    if Path(WAN21_REPO).exists() and Path(WAN21_REPO, "generate.py").exists():
        return
    print("[setup] Cloning Wan2.1 repo ...")
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/Wan-Video/Wan2.1.git", WAN21_REPO],
        check=True
    )
    print("[setup] Installing Wan2.1 requirements ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r",
         str(Path(WAN21_REPO, "requirements.txt")), "-q"],
        check=True
    )


def get_first_frame_jpg(mp4_path: str, out_jpg: str) -> bool:
    cap = cv2.VideoCapture(mp4_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    cv2.imwrite(out_jpg, frame)
    return True


def get_episodes(subset: str):
    folder = Path(DATA_ROOT) / SUBSET_MAP[subset]
    eps = sorted(folder.glob("episode_*"))
    return eps


def run_episode(ep_dir: Path, subset: str, gpu: int, overwrite: bool):
    ep_id   = ep_dir.name.split("_")[-1]
    sub_out = SUBSET_MAP[subset]
    out_dir = Path(OUT_ROOT) / sub_out / f"episode_{ep_id}"
    out_mp4 = out_dir / "pred.mp4"

    if out_mp4.exists() and not overwrite:
        print(f"[skip] {out_mp4} exists")
        return

    # find source files
    mp4_path     = ep_dir / f"gt_{ep_id}.mp4"
    instruct_txt = ep_dir / f"instruct_{ep_id}.txt"

    if not mp4_path.exists():
        print(f"[warn] missing {mp4_path}, skip")
        return
    if not instruct_txt.exists():
        print(f"[warn] missing {instruct_txt}, skip")
        return

    prompt = instruct_txt.read_text().strip()
    out_dir.mkdir(parents=True, exist_ok=True)

    # extract first frame
    tmp_jpg = str(out_dir / "first_frame.jpg")
    if not get_first_frame_jpg(str(mp4_path), tmp_jpg):
        print(f"[warn] cannot read first frame from {mp4_path}, skip")
        return

    # call generate.py
    generate_py = str(Path(WAN21_REPO, "generate.py"))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = [
        sys.executable, generate_py,
        "--task",         "i2v-14B",
        "--size",         VIDEO_SIZE,
        "--ckpt_dir",     CKPT_DIR,
        "--image",        tmp_jpg,
        "--prompt",       prompt,
        "--offload_model","True",
        "--frame_num",    str(NUM_FRAMES),
        "--save_file",    str(out_mp4),
    ]

    print(f"[run] GPU{gpu}  {sub_out}/episode_{ep_id}")
    result = subprocess.run(cmd, env=env, cwd=WAN21_REPO)
    if result.returncode != 0:
        print(f"[error] generate.py failed for {ep_dir.name}")
    else:
        print(f"[done] saved {out_mp4}")

    # keep first frame for debugging, remove to save space if desired
    # os.remove(tmp_jpg)


def main():
    global DATA_ROOT, OUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset",    choices=["gen","gr1","all"], default="all")
    parser.add_argument("--gpu",       type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--out-root",  default=OUT_ROOT)
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    OUT_ROOT  = args.out_root

    clone_wan21_repo()

    subsets = ["gen","gr1"] if args.subset == "all" else [args.subset]
    for subset in subsets:
        episodes = get_episodes(subset)
        print(f"\n=== Wan2.1 I2V  subset={subset}  {len(episodes)} episodes ===")
        for ep_dir in episodes:
            run_episode(ep_dir, subset, args.gpu, args.overwrite)


if __name__ == "__main__":
    main()
