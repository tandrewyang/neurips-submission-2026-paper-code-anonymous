#!/usr/bin/env python3
"""Download cosmos A2V 2B and 14B part2_action_following directly by known paths."""
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil, sys

REPO = "morinoppp/cosmos_results"
BASE = Path("/mnt/users/zirui/mizirui_benchmark/Mirabench_exp")
DST = {
    "2B":  BASE / "cosmos_a2v_2b",
    "14B": BASE / "cosmos_a2v_14b",
}

GEN_EPISODES = [f"episode_{i:04d}" for i in range(15)]
GR1_EPISODES = [
    "episode_0005","episode_0006","episode_0007","episode_0008","episode_0009",
    "episode_0012","episode_0015","episode_0020","episode_0021","episode_0022",
    "episode_0024","episode_0031","episode_0034","episode_0037","episode_0043",
    "episode_0046","episode_0047","episode_0050","episode_0051","episode_0058",
    "episode_0063","episode_0071","episode_0072","episode_0074","episode_0082",
    "episode_0084","episode_0090","episode_0098","episode_0099","episode_0102",
    "episode_0110","episode_0111","episode_0113","episode_0114","episode_0118",
    "episode_0123","episode_0127","episode_0129","episode_0132","episode_0138",
    "episode_0144","episode_0157","episode_0168","episode_0171","episode_0172",
    "episode_0185","episode_0188","episode_0191","episode_0193","episode_0197",
]

SUBSETS = {
    "generalizability": GEN_EPISODES,
    "gr1": GR1_EPISODES,
}

models = sys.argv[1:] if len(sys.argv) > 1 else ["2B", "14B"]

total = 0
for model in models:
    for subset, episodes in SUBSETS.items():
        for ep in episodes:
            hf_path = f"{model}/part2_action_following/{subset}/{ep}/pred.mp4"
            out_dir = DST[model] / subset / ep
            out_mp4 = out_dir / "pred.mp4"
            if out_mp4.exists():
                print(f"[skip] {model}/{subset}/{ep}")
                continue
            print(f"[dl]  {hf_path}", flush=True)
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                tmp = hf_hub_download(REPO, filename=hf_path, repo_type="dataset")
                shutil.copy2(tmp, out_mp4)
                print(f"[ok]  {out_mp4}", flush=True)
                total += 1
            except Exception as e:
                print(f"[err] {hf_path}: {e}", flush=True)

print(f"\nALL DONE — downloaded {total} files")
