#!/usr/bin/env python3
"""
Wan2.2 I2V-A14B inference for Mirabench.
Uses 2 GPUs (default 6,7) via torchrun + FSDP (requires 80GB total VRAM).
"""
import os
import sys
import subprocess
import argparse
import cv2
from pathlib import Path

DATA_ROOT   = "/mnt/users/zirui/mizirui_benchmark/Mirabench_exp/video_batch_final"
OUT_ROOT    = "/mnt/users/zirui/mizirui_benchmark/Mirabench_exp/wan2.2"
WAN22_REPO  = "/mnt/users/zirui/mizirui_benchmark/Wan2.2"
CKPT_DIR    = "/mnt/public/models/Wan-AI/Wan2.2-I2V-A14B"
SUBSET_MAP  = {"gen": "generalizability", "gr1": "gr1"}
VIDEO_SIZE  = "480*832"
NUM_FRAMES  = 81


def clone_wan22_repo():
    if Path(WAN22_REPO).exists() and Path(WAN22_REPO, "generate.py").exists():
        return
    print("[setup] Cloning Wan2.2 repo ...")
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/Wan-Video/Wan2.2.git", WAN22_REPO],
        check=True
    )
    print("[setup] Installing Wan2.2 requirements ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r",
         str(Path(WAN22_REPO, "requirements.txt")), "-q"],
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
    return sorted(folder.glob("episode_*"))


def run_episode(ep_dir: Path, subset: str, gpus: str, overwrite: bool):
    ep_id   = ep_dir.name.split("_")[-1]
    sub_out = SUBSET_MAP[subset]
    out_dir = Path(OUT_ROOT) / sub_out / f"episode_{ep_id}"
    out_mp4 = out_dir / "pred.mp4"

    if out_mp4.exists() and not overwrite:
        print(f"[skip] {out_mp4} exists")
        return

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

    tmp_jpg = str(out_dir / "first_frame.jpg")
    if not get_first_frame_jpg(str(mp4_path), tmp_jpg):
        print(f"[warn] cannot read first frame from {mp4_path}, skip")
        return

    generate_py = str(Path(WAN22_REPO, "generate.py"))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus

    n_gpus = len(gpus.split(","))
    if n_gpus > 1:
        cmd = [
            "torchrun", f"--nproc_per_node={n_gpus}", generate_py,
            "--task",               "i2v-A14B",
            "--size",               VIDEO_SIZE,
            "--ckpt_dir",           CKPT_DIR,
            "--image",              tmp_jpg,
            "--prompt",             prompt,
            "--dit_fsdp",
            "--t5_fsdp",
            f"--ulysses_size",      str(n_gpus),
            "--frame_num",          str(NUM_FRAMES),
            "--save_file",          str(out_mp4),
        ]
    else:
        # single GPU: needs offload + dtype conversion
        cmd = [
            sys.executable, generate_py,
            "--task",               "i2v-A14B",
            "--size",               VIDEO_SIZE,
            "--ckpt_dir",           CKPT_DIR,
            "--image",              tmp_jpg,
            "--prompt",             prompt,
            "--offload_model",      "True",
            "--convert_model_dtype",
            "--frame_num",          str(NUM_FRAMES),
            "--save_file",          str(out_mp4),
        ]

    print(f"[run] GPU{gpus}  {sub_out}/episode_{ep_id}  (n_gpus={n_gpus})")
    result = subprocess.run(cmd, env=env, cwd=WAN22_REPO)
    if result.returncode != 0:
        print(f"[error] generate.py failed for {ep_dir.name}")
    else:
        print(f"[done] saved {out_mp4}")


def main():
    global DATA_ROOT, OUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset",    choices=["gen","gr1","all"], default="all")
    parser.add_argument("--gpus",      default="6,7",
                        help="Comma-separated GPU IDs, e.g. 6,7")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--out-root",  default=OUT_ROOT)
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    OUT_ROOT  = args.out_root

    clone_wan22_repo()

    subsets = ["gen","gr1"] if args.subset == "all" else [args.subset]
    for subset in subsets:
        episodes = get_episodes(subset)
        print(f"\n=== Wan2.2 I2V-A14B  subset={subset}  {len(episodes)} episodes ===")
        for ep_dir in episodes:
            run_episode(ep_dir, subset, args.gpus, args.overwrite)


if __name__ == "__main__":
    main()
