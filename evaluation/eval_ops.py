#!/usr/bin/env python3
"""
eval_mirabench_ops.py — OPS evaluation for any model × any subset.

For each episode:
  - 16 uniformly sampled frames from pred + GT
  - InternVL3-78B judges each frame pair: object quality pass/fail (0/1)
  - confidence = mean(votes)
  - OPS bin: conf >= 0.70 → high; >= 0.40 → medium; else → low
"""

import argparse, json, math, os, re, traceback
from pathlib import Path

import numpy as np
import torch
import torchvision.io as tvio
import torchvision.transforms as T
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoTokenizer
from torchvision.transforms.functional import InterpolationMode

BASE_MODEL = ""
IMAGE_SIZE = 448
MAX_NUM_TILES = 1

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

HIGH_THRESH   = 0.70
MEDIUM_THRESH = 0.40

FRAME_PROMPT = (
    "You are evaluating the visual quality of a single predicted robot video frame.\n\n"
    "  • Frame1 = frame from the PREDICTED video (world model output — evaluate this)\n"
    "  • Frame2 = frame from the GROUND TRUTH video at the same timestamp (reference only)\n\n"
    'Task instruction: "{instruction}"\n\n'
    "Check Frame1 against Frame2 for the following issues:\n"
    "  1. Is the target object (the object being manipulated) clearly visible in Frame1, "
    "without unexpected blurring, occlusion, or disappearance?\n"
    "  2. Are all objects in Frame1 free of distortion or unnatural deformation?\n"
    "  3. Are there no objects that pop in or pop out unnaturally between frames "
    "(appearing or vanishing without physical cause)?\n\n"
    "Respond ONLY with 0 or 1. No explanation.\n"
    "1 = Frame1 passes all checks (high quality, matches GT object presence)\n"
    "0 = Frame1 fails at least one check (object issue, distortion, or pop artifact)"
)


def split_model(model_path):
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


def dynamic_preprocess(image):
    w, h = image.size
    aspect = w / h
    ratios = sorted(
        {(i, j) for n in range(1, MAX_NUM_TILES + 1)
         for i in range(1, n + 1) for j in range(1, n + 1) if 1 <= i * j <= MAX_NUM_TILES},
        key=lambda x: x[0] * x[1],
    )
    best = min(ratios, key=lambda r: abs(aspect - r[0] / r[1]))
    tw, th = IMAGE_SIZE * best[0], IMAGE_SIZE * best[1]
    resized = image.resize((tw, th))
    tiles = []
    for i in range(best[0] * best[1]):
        x0 = (i % best[0]) * IMAGE_SIZE
        y0 = (i // best[0]) * IMAGE_SIZE
        tiles.append(resized.crop((x0, y0, x0 + IMAGE_SIZE, y0 + IMAGE_SIZE)))
    if len(tiles) > 1:
        tiles.append(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
    return tiles


def frame_to_pixel_values(img):
    transform = build_transform()
    tiles = dynamic_preprocess(img)
    pv = torch.stack([transform(t) for t in tiles])
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
    return [Image.fromarray(vframes[i].numpy(), mode="RGB") for i in indices]


def parse_vote(text):
    t = text.strip()
    if t in ("0", "1"):
        return int(t)
    m = re.search(r"\b([01])\b", t)
    return int(m.group(1)) if m else None


def conf_to_ops(conf):
    if conf >= HIGH_THRESH:
        return "high"
    elif conf >= MEDIUM_THRESH:
        return "medium"
    return "low"


def discover_episodes(model_dir, gt_dir, pred_filename):
    """Find episodes present in both model_dir and gt_dir."""
    episodes = []
    for ep_dir in sorted(Path(model_dir).iterdir()):
        if not (ep_dir.is_dir() and ep_dir.name.startswith("episode_")):
            continue
        tag = ep_dir.name.split("_")[1]
        ep  = int(tag)
        pred = ep_dir / pred_filename
        gt   = Path(gt_dir) / f"episode_{tag}" / f"gt_{tag}.mp4"
        instr = Path(gt_dir) / f"episode_{tag}" / f"instruct_{tag}.txt"
        if not pred.exists():
            print(f"  [warn] missing pred: {pred}")
            continue
        if not gt.exists():
            print(f"  [warn] missing gt: {gt}")
            continue
        instruction = instr.read_text().strip() if instr.exists() else "robot manipulation task"
        episodes.append({"episode": ep, "tag": tag, "pred": pred, "gt": gt,
                         "instruction": instruction})
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",      required=True,
                        help="Dir containing episode_XXXX/pred*.mp4 for this model+subset")
    parser.add_argument("--gt_dir",         required=True,
                        help="Dir containing episode_XXXX/gt_XXXX.mp4 (video_batch_final subset)")
    parser.add_argument("--pred_filename",  default="pred.mp4",
                        help="Pred video filename inside each episode dir")
    parser.add_argument("--num_frames",     type=int, default=16)
    parser.add_argument("--output",         required=True,
                        help="Output JSON path")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading InternVL3-78B...")
    print(f"GPUs: {torch.cuda.device_count()} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','all')})")
    device_map = split_model(BASE_MODEL)
    model = AutoModel.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True, device_map=device_map,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=False)
    print(f"Model ready. num_frames={args.num_frames}\n")

    episodes = discover_episodes(args.model_dir, args.gt_dir, args.pred_filename)
    print(f"Episodes found: {len(episodes)}\n")

    results = []
    for item in episodes:
        ep  = item["episode"]
        entry = {
            "episode": ep, "tag": item["tag"],
            "OPS": None, "OPS_score": None, "confidence": None,
            "votes": [], "parse_ok": False, "error": None,
        }
        try:
            pred_frames = extract_frames(item["pred"], args.num_frames)
            gt_frames   = extract_frames(item["gt"],   args.num_frames)

            votes = []
            for pf, gf in zip(pred_frames, gt_frames):
                pv_p, np_p = frame_to_pixel_values(pf)
                pv_g, np_g = frame_to_pixel_values(gf)
                pixel_values = torch.cat([pv_p, pv_g], dim=0).to(torch.bfloat16).cuda()
                question = ("Frame1: <image>\nFrame2: <image>\n" +
                            FRAME_PROMPT.format(instruction=item["instruction"]))
                with torch.no_grad():
                    response = model.chat(
                        tokenizer, pixel_values, question,
                        dict(max_new_tokens=8, do_sample=False),
                        num_patches_list=[np_p, np_g],
                    )
                votes.append(parse_vote(response))

            valid = [v for v in votes if v is not None]
            if valid:
                conf    = sum(valid) / len(valid)
                ops_cat = conf_to_ops(conf)
                entry.update({
                    "OPS": ops_cat,
                    "OPS_score": OPS_SCORE[ops_cat],
                    "confidence": conf,
                    "votes": votes,
                    "parse_ok": True,
                })
        except Exception as e:
            entry["error"] = str(e)
            traceback.print_exc()

        if entry["parse_ok"]:
            print(f"[ep{ep:04d}]  OPS={entry['OPS']} conf={entry['confidence']:.2f}  votes={votes}")
        else:
            print(f"[ep{ep:04d}]  FAIL: {entry['error']}")
        results.append(entry)

        with open(out_path, "w") as f:
            json.dump({
                "model_dir": args.model_dir,
                "gt_dir": args.gt_dir,
                "pred_filename": args.pred_filename,
                "num_frames": args.num_frames,
                "results": results,
            }, f, indent=2, ensure_ascii=False)

    # Summary
    valid = [r for r in results if r["parse_ok"]]
    print(f"\n=== OPS Summary | {len(valid)}/{len(results)} valid ===")
    if valid:
        from collections import Counter
        dist = Counter(r["OPS"] for r in valid)
        mean_conf = sum(r["confidence"] for r in valid) / len(valid)
        mean_score = sum(r["OPS_score"] for r in valid) / len(valid)
        print(f"  Distribution: {dict(dist)}")
        print(f"  Mean confidence: {mean_conf:.3f}")
        print(f"  Mean OPS score:  {mean_score:.1f} / 35")

    summary = {
        "model_dir": args.model_dir,
        "pred_filename": args.pred_filename,
        "num_frames": args.num_frames,
        "n_episodes": len(results),
        "n_valid": len(valid),
        "ops_distribution": {k: v for k, v in Counter(r["OPS"] for r in valid).items()} if valid else {},
        "mean_confidence": sum(r["confidence"] for r in valid) / len(valid) if valid else None,
        "mean_ops_score": sum(r["OPS_score"] for r in valid) / len(valid) if valid else None,
    }
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
