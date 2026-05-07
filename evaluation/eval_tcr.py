#!/usr/bin/env python3
"""
eval_tcr.py — TCR (Task Completion Rate) evaluation using Method C.

Method C: 16 uniformly sampled frames from the predicted video (no GT reference)
are presented to InternVL3-78B in a single call. The VLM outputs a binary 0/1
judgment for the whole video → that is TCR.

GEN (Generalizability) is automatically computed:
  GEN = min(100, 100 * exp(-(TCR_GR1 - TCR_Gen) / 100))

Usage:
  # Single model + subset
  CUDA_VISIBLE_DEVICES=0,1,2 python3 eval_tcr.py \
      --model_dir  ../dreamdojo_2b/gr1 \
      --gt_dir     ../video_batch_final/gr1 \
      --pred_filename pred_2b.mp4 \
      --output     results/dreamdojo_2b_gr1_tcr.json

  # All models at once (loads InternVL3 once)
  CUDA_VISIBLE_DEVICES=0,1,2 python3 eval_tcr.py --all

Requirements:
  pip install torch torchvision transformers numpy pillow
  InternVL3-78B model weights (set INTERNVL_MODEL_PATH or uses default)
"""

import argparse, json, math, os, re, sys, time, traceback
from pathlib import Path

import numpy as np
import torch
import torchvision.io as tvio
import torchvision.transforms as T
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode

INTERNVL_MODEL_PATH = os.environ.get(
    "INTERNVL_MODEL_PATH"
)
IMAGE_SIZE = 448
NUM_FRAMES = 16

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

WHOLE_PROMPT = (
    "You are evaluating a robot manipulation video.\n"
    "Here are 16 frames sampled uniformly from the predicted video, in chronological order.\n\n"
    'Task instruction: "{instruction}"\n\n'
    "Looking at the entire video: did the robot complete the task instruction?\n"
    "  • Track object positions across the full sequence to confirm goal achievement.\n"
    "  • Do NOT require any specific arm pose — focus on whether the target object reaches the correct final state.\n\n"
    "Respond ONLY with 0 or 1.\n"
    "1 = task completed in this video\n"
    "0 = task not completed"
)

# Default model table for --all mode. Edit as needed.
MODEL_PRED_TABLE = []
SUBSETS = ["gr1", "gen"]

# Default data root (parent of this script's directory)
DATA_ROOT = Path(__file__).resolve().parent.parent
GT_DIRS = {
    "gr1": DATA_ROOT / "video_batch_final" / "gr1",
    "gen": DATA_ROOT / "video_batch_final" / "generalizability",
}
SUBSET_TO_DIRNAME = {"gr1": "gr1", "gen": "generalizability"}
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "tcr"


# ---------------------------------------------------------------------------
# Model / inference helpers
# ---------------------------------------------------------------------------
def split_model_tp(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    nlay = math.ceil(num_layers / (world_size - 0.5))
    nlay = [nlay] * world_size
    nlay[0] = math.ceil(nlay[0] * 0.5)
    cnt = 0
    for i, n in enumerate(nlay):
        for _ in range(n):
            device_map[f'language_model.model.layers.{cnt}'] = i
            cnt += 1
    for k in ['vision_model', 'mlp1', 'language_model.model.tok_embeddings',
              'language_model.model.embed_tokens', 'language_model.output',
              'language_model.model.norm', 'language_model.model.rotary_emb',
              'language_model.lm_head', f'language_model.model.layers.{num_layers - 1}']:
        device_map[k] = 0
    return device_map


def build_transform():
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def frame_to_pixel_values(img):
    pv = torch.stack([build_transform()(img.resize((IMAGE_SIZE, IMAGE_SIZE)))])
    return pv, pv.shape[0]


def extract_frames(video_path, n_frames):
    try:
        vframes, _, _ = tvio.read_video(str(video_path), pts_unit="sec", output_format="THWC")
    except Exception:
        vframes, _, _ = tvio.read_video(str(video_path), pts_unit="pts", output_format="THWC")
    total = vframes.shape[0]
    if total == 0:
        raise ValueError(f"Empty video: {video_path}")
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    return [Image.fromarray(vframes[i].numpy()) for i in indices]


def parse_vote(text):
    t = (text or "").strip()
    if t in ("0", "1"):
        return int(t)
    m = re.search(r"\b([01])\b", t)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Episode discovery
# ---------------------------------------------------------------------------
def discover_episodes(model_dir, gt_dir, pred_filename):
    episodes = []
    model_dir = Path(model_dir)
    gt_dir = Path(gt_dir)
    if not model_dir.exists():
        print(f"  [warn] model_dir does not exist: {model_dir}", flush=True)
        return episodes
    for ep_dir in sorted(model_dir.iterdir()):
        if not (ep_dir.is_dir() and ep_dir.name.startswith("episode_")):
            continue
        tag = ep_dir.name.split("_")[1]
        pred = ep_dir / pred_filename
        instr = gt_dir / f"episode_{tag}" / f"instruct_{tag}.txt"
        if not pred.exists():
            print(f"  [warn] missing pred: {pred}", flush=True)
            continue
        instruction = (instr.read_text().strip()
                       if instr.exists() else "robot manipulation task")
        episodes.append({
            "episode": int(tag),
            "tag": tag,
            "pred": pred,
            "instruction": instruction,
        })
    return episodes


# ---------------------------------------------------------------------------
# Single-episode inference
# ---------------------------------------------------------------------------
def evaluate_one(model, tokenizer, pred_path, instruction):
    frames = extract_frames(pred_path, NUM_FRAMES)
    pvs, nps = [], []
    for f in frames:
        pv, np_ = frame_to_pixel_values(f)
        pvs.append(pv)
        nps.append(np_)
    pixel_values = torch.cat(pvs, dim=0).to(torch.bfloat16).cuda()

    frame_tokens = "".join([f"Frame{i+1}: <image>\n" for i in range(NUM_FRAMES)])
    question = frame_tokens + WHOLE_PROMPT.format(instruction=instruction)

    with torch.no_grad():
        response = model.chat(
            tokenizer, pixel_values, question,
            dict(max_new_tokens=8, do_sample=False),
            num_patches_list=nps,
        )
    vote = parse_vote(response)
    return response, vote, vote is not None


# ---------------------------------------------------------------------------
# Per-job runner
# ---------------------------------------------------------------------------
def run_job(model, tokenizer, *, model_name, subset_tag, pred_filename,
            model_dir, gt_dir, out_path):
    out_path = Path(out_path)
    if out_path.exists():
        print(f"[skip] {model_name}/{subset_tag}: {out_path} already exists", flush=True)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Job: model={model_name}  subset={subset_tag}  pred={pred_filename} ===",
          flush=True)
    episodes = discover_episodes(model_dir, gt_dir, pred_filename)
    print(f"  episodes discovered: {len(episodes)}", flush=True)

    results = []
    t0 = time.time()
    for item in episodes:
        ep = item["episode"]
        entry = {
            "episode": ep, "tag": item["tag"],
            "TCR": None, "raw_response": None, "parse_ok": False, "error": None,
        }
        try:
            t1 = time.time()
            raw, vote, ok = evaluate_one(model, tokenizer, item["pred"], item["instruction"])
            dt = time.time() - t1
            entry["raw_response"] = raw
            if ok:
                entry["TCR"] = vote
                entry["parse_ok"] = True
            print(f"[{model_name}/{subset_tag} ep{ep:04d}]  TCR={entry['TCR']} ok={ok} "
                  f"raw={raw!r} ({dt:.1f}s)", flush=True)
        except Exception as e:
            entry["error"] = str(e)
            print(f"[{model_name}/{subset_tag} ep{ep:04d}]  FAIL: {e}", flush=True)
            traceback.print_exc()
        results.append(entry)

        # Incremental save
        valid = [r for r in results if r["parse_ok"]]
        n_tcr1 = sum(r["TCR"] for r in valid) if valid else 0
        summary = {
            "model": model_name, "split": subset_tag, "method": "C-whole16",
            "pred_filename": pred_filename, "num_frames": NUM_FRAMES,
            "n_episodes": len(results), "n_valid": len(valid),
            "n_tcr1": n_tcr1 if valid else None,
            "tcr_rate": (n_tcr1 / len(valid)) if valid else None,
        }
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "results": results}, f,
                      indent=2, ensure_ascii=False)

    dt = time.time() - t0
    print(f"=== {model_name}/{subset_tag} done in {dt:.1f}s, saved -> {out_path} ===",
          flush=True)


# ---------------------------------------------------------------------------
# GEN computation
# ---------------------------------------------------------------------------
def compute_gen(results_dir):
    """Compute GEN scores from TCR results and print a summary table."""
    results_dir = Path(results_dir)
    tcr_gr1, tcr_gen = {}, {}
    for f in sorted(results_dir.glob("*.json")):
        d = json.load(open(f))
        s = d["summary"]
        model = s["model"]
        if s["split"] == "gr1":
            tcr_gr1[model] = s["tcr_rate"]
        else:
            tcr_gen[model] = s["tcr_rate"]

    all_models = sorted(tcr_gr1.keys(), key=lambda m: -tcr_gr1[m])

    print(f"\n{'='*70}")
    print(f"{'Model':<25} {'TCR GR1':>10} {'TCR Gen':>10} {'GEN':>10}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    for m in all_models:
        g1 = tcr_gr1[m] * 100
        gv = tcr_gen.get(m, 0) * 100
        gen_val = g1 - gv
        gen_score = min(100, 100 * math.exp(-gen_val / 100))
        print(f"{m:<25} {g1:>9.1f}% {gv:>9.1f}% {gen_score:>9.1f}%")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_internvl():
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}", flush=True)
    print(f"GPU count visible: {torch.cuda.device_count()}", flush=True)
    print(f"Loading InternVL3-78B from {INTERNVL_MODEL_PATH} ...", flush=True)
    t0 = time.time()
    device_map = split_model_tp(INTERNVL_MODEL_PATH)
    model = AutoModel.from_pretrained(
        INTERNVL_MODEL_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True, device_map=device_map,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(INTERNVL_MODEL_PATH, trust_remote_code=True, use_fast=False)
    print(f"Model ready in {time.time()-t0:.1f}s.  num_frames={NUM_FRAMES}", flush=True)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TCR evaluation (Method C) with GEN computation")
    parser.add_argument("--all", action="store_true",
                        help="Iterate over all (model, subset) pairs with one model load.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory for output JSONs (--all mode).")
    parser.add_argument("--model_dir", help="Model directory (single-job mode).")
    parser.add_argument("--gt_dir", help="GT directory (single-job mode).")
    parser.add_argument("--pred_filename", default="pred.mp4",
                        help="Pred video filename (default: pred.mp4).")
    parser.add_argument("--output", help="Output JSON path (single-job mode).")
    parser.add_argument("--model_name", default=None,
                        help="Override model name in output JSON.")
    parser.add_argument("--subset_tag", default=None, choices=[None, "gr1", "gen"],
                        help="Override split tag in output JSON.")
    parser.add_argument("--compute_gen", action="store_true",
                        help="Only compute GEN from existing TCR results (no inference).")
    parser.add_argument("--gen_dir", default=None,
                        help="Directory containing TCR result JSONs for GEN computation.")
    args = parser.parse_args()

    if args.compute_gen:
        gen_dir = args.gen_dir or args.output_dir
        compute_gen(gen_dir)
        return

    if args.all:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pending = []
        for model_name, pred_filename in MODEL_PRED_TABLE:
            for subset_tag in SUBSETS:
                out_path = out_dir / f"{model_name}_{subset_tag}_tcr.json"
                if out_path.exists():
                    print(f"[pre-skip] {out_path}", flush=True)
                    continue
                pending.append((model_name, pred_filename, subset_tag, out_path))
        if not pending:
            print("All output JSONs already present — nothing to do.", flush=True)
            compute_gen(out_dir)
            return

        model, tokenizer = load_internvl()
        for model_name, pred_filename, subset_tag, out_path in pending:
            subset_dir = SUBSET_TO_DIRNAME[subset_tag]
            model_dir = DATA_ROOT / model_name / subset_dir
            gt_dir = GT_DIRS[subset_tag]
            try:
                run_job(model, tokenizer,
                        model_name=model_name, subset_tag=subset_tag,
                        pred_filename=pred_filename,
                        model_dir=model_dir, gt_dir=gt_dir, out_path=out_path)
            except Exception as e:
                print(f"[fatal] {model_name}/{subset_tag}: {e}", flush=True)
                traceback.print_exc()
        print("\nALL JOBS DONE.", flush=True)
        compute_gen(out_dir)
        return

    # Single-job mode
    if not (args.model_dir and args.gt_dir and args.output):
        parser.error("Either pass --all, or provide --model_dir, --gt_dir, --output.")

    model_name = args.model_name or Path(args.model_dir).parent.name
    if args.subset_tag:
        subset_tag = args.subset_tag
    else:
        sub = Path(args.model_dir).name
        subset_tag = "gr1" if sub == "gr1" else "gen"

    model, tokenizer = load_internvl()
    run_job(model, tokenizer,
            model_name=model_name, subset_tag=subset_tag,
            pred_filename=args.pred_filename,
            model_dir=args.model_dir, gt_dir=args.gt_dir, out_path=args.output)


if __name__ == "__main__":
    main()
