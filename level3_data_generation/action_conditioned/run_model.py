#!/usr/bin/env python3
"""
Universal action-conditioned generation runner.

For each task × episode × condition in the benchmark:
  - Load first_frame.jpg + actions_per_condition/<condition>.npy from tasks/<task>/ep<N>_idx<M>/
  - Call the model adapter to generate a video
  - Save to test_data/<model_name>/<task>/ep<N>_idx<M>/<condition>_pred.mp4

Usage:
  CUDA_VISIBLE_DEVICES=4 \
  /path/to/your_env/DreamDojo/.venv/bin/python3 \
      data_generation_part3_action/run_model.py \
      --model cosmos_predict2_2b_gr1 \
      --out-dir test_data/cosmos_predict2_2b_gr1

  # Single task only:
  ... --tasks gr1_pnp_apple gr1_pnp_mango

  # Resume (skip already-generated):
  ... --resume

Available action-conditioned models:
  dreamdojo_2b_gr1, cosmos_predict2_2b_gr1 (cosmos25_2b_gr1),
  cosmos_predict2_14b_gr1 (cosmos14b_gr1), irasim_gr1 (irasim),
  ctrlworld_gr1 (ctrlworld), enerverse_ac
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = Path(__file__).resolve().parent / "tasks"
sys.path.insert(0, str(REPO_ROOT))

from module_a_ewmbench.bench_schedule import TASK_REGISTRY, ALL_TASKS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def save_video(frames: np.ndarray, path: Path, fps: int = 16):
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    H, W = frames.shape[1], frames.shape[2]
    with av.open(str(path), mode="w") as c:
        s = c.add_stream("h264", rate=fps)
        s.width = W; s.height = H; s.pix_fmt = "yuv420p"
        s.options = {"crf": "20", "preset": "fast"}
        for f in frames:
            vf = av.VideoFrame.from_ndarray(f, format="rgb24")
            for pkt in s.encode(vf): c.mux(pkt)
        for pkt in s.encode(): c.mux(pkt)
    log.info("  saved: %s (%d frames)", path, len(frames))


def run(model_name: str, tasks: list[str], out_dir: Path,
        device: str = "cuda:0", resume: bool = True):
    # Load adapter
    from adapters import ADAPTER_REGISTRY, load_adapter_class
    if model_name not in ADAPTER_REGISTRY:
        log.error("Adapter '%s' not found. Available: %s", model_name, list(ADAPTER_REGISTRY))
        sys.exit(1)
    cls = load_adapter_class(model_name)
    # Some adapters accept a config dict; others accept device= as a keyword.
    try:
        adapter = cls({"model_device": device, "model_name": model_name, "use_base_only": model_name.endswith("_base")})
    except TypeError:
        adapter = cls(device=device)
    log.info("Loaded adapter: %s", model_name)

    manifest: dict = {}
    for task_label in tasks:
        task_dir = TASKS_DIR / task_label
        if not task_dir.exists():
            log.warning("Task data not found: %s — run setup_data.py first", task_dir)
            continue
        task_info = json.loads((task_dir / "task_info.json").read_text())
        ep_indices = task_info["episode_indices"]
        conditions = task_info["conditions"]
        num_frames = task_info["num_frames"]

        log.info("="*60)
        log.info("Task: %s  |  %d episodes  |  %d conditions",
                 task_label, len(ep_indices), len(conditions))

        for ep_slot, ep_idx in enumerate(ep_indices):
            ep_dir   = task_dir / f"ep{ep_slot:03d}_idx{ep_idx}"
            jpg_p    = ep_dir / "first_frame.jpg"
            acts_dir = ep_dir / "actions_per_condition"
            if not jpg_p.exists():
                log.warning("Missing first_frame for %s/ep%03d — skipping", task_label, ep_slot)
                continue
            first_frame = np.array(PILImage.open(jpg_p).convert("RGB"))

            for condition in conditions:
                out_path = (out_dir / task_label
                            / f"ep{ep_slot:03d}_idx{ep_idx}"
                            / f"{condition}_pred.mp4")
                if resume and out_path.exists():
                    log.info("  %s/ep%03d/%s: skip (exists)", task_label, ep_slot, condition)
                    manifest.setdefault(f"{task_label}/ep{ep_slot:03d}/{condition}", {})[
                        "video"] = str(out_path)
                    continue

                npy_p = acts_dir / f"{condition}.npy"
                if not npy_p.exists():
                    log.warning("  no action array for %s/ep%03d/%s — skipping",
                                task_label, ep_slot, condition)
                    continue
                actions = np.load(str(npy_p))  # (T-1, 29) float32

                log.info("  ep%03d (idx=%d) [%s]  actions=%s",
                         ep_slot, ep_idx, condition, actions.shape)
                try:
                    frames = adapter.predict(
                        initial_frame=first_frame,
                        actions=actions,
                        num_frames=num_frames,
                    )
                    save_video(np.array(frames), out_path)
                    manifest.setdefault(
                        f"{task_label}/ep{ep_slot:03d}/{condition}", {}).update({
                        "video": str(out_path),
                        "task": task_label, "ep_slot": ep_slot, "ep_idx": ep_idx,
                        "condition": condition,
                    })
                except Exception as e:
                    log.error("  FAILED: %s", e, exc_info=True)
                    manifest.setdefault(
                        f"{task_label}/ep{ep_slot:03d}/{condition}", {})["error"] = str(e)

    mf_path = out_dir / "generation_manifest.json"
    mf_path.parent.mkdir(parents=True, exist_ok=True)
    mf_path.write_text(json.dumps(manifest, indent=2))
    n_ok  = sum(1 for v in manifest.values() if "video" in v)
    n_err = sum(1 for v in manifest.values() if "error" in v)
    log.info("Done. Videos: %d  Errors: %d  Manifest: %s", n_ok, n_err, mf_path)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     required=True, help="Adapter name (e.g. dreamdojo_2b_gr1, irasim)")
    p.add_argument("--tasks",     nargs="+", default=ALL_TASKS)
    p.add_argument("--out-dir",   default="", help="Default: test_data/<model>")
    p.add_argument("--device",    default="cuda:0")
    p.add_argument("--no-resume", action="store_true", help="Regenerate even if file exists")
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "test_data" / args.model
    run(
        model_name=args.model,
        tasks=[t for t in args.tasks if t in TASK_REGISTRY],
        out_dir=out_dir,
        device=args.device,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
