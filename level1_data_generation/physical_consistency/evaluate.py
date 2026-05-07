"""
evaluate.py — Physical consistency evaluator (SC_A2 + SC_O)

Uses InternVL3-78B pairframe binary judgement (midcut strategy,
n_pairs=10, threshold=1).

Usage (single video):
    python evaluate.py --video path/to/video.mp4 --gpus 4,5
    python evaluate.py --video path/to/video.mp4 --gpus 4,5 --metric sc_o

Usage (batch):
    python evaluate.py --video_list videos.txt --gpus 4,5 --metric sc_a2 --out results.jsonl
    python evaluate.py --video_list videos.txt --gpus 4,5 --metric sc_o --out results.jsonl
    python evaluate.py --video_list videos.txt --gpus 4,5 --metric both --out results.jsonl
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

# ── Patch: torch.linspace on meta tensors (ViT init) ─────────────────────────
_orig_linspace = torch.linspace
def _safe_linspace(*args, **kwargs):
    if str(kwargs.get("device", "")) == "meta":
        kwargs["device"] = "cpu"
    try:
        return _orig_linspace(*args, **kwargs)
    except NotImplementedError:
        kwargs.pop("device", None)
        return _orig_linspace(*args, device="cpu", **kwargs)
torch.linspace = _safe_linspace

# ── Patch: transformers 5.2.0 compat ─────────────────────────────────────────
import transformers.integrations.accelerate as _ta
_orig_init_infer = _ta._init_infer_auto_device_map
def _patched_init_infer(model, *args, **kwargs):
    if not hasattr(model, "all_tied_weights_keys"):
        model.all_tied_weights_keys = {}
    return _orig_init_infer(model, *args, **kwargs)
_ta._init_infer_auto_device_map = _patched_init_infer

BASE_MODEL_PATH = os.environ.get(
    "PHYSCONS_MODEL_PATH", "OpenGVLab/InternVL3-78B"
)
N_PAIRS         = 10
THRESHOLD       = 1   # bad_count >= THRESHOLD → inconsistent (B)
PAIR_MODE       = "midcut"
IMAGENET_MEAN   = (0.485, 0.456, 0.406)
IMAGENET_STD    = (0.229, 0.224, 0.225)

PROMPT_SC_A2 = (
    "These two frames are from a robot manipulation video. "
    "Frame 1 is on top, Frame 2 is on bottom.\n\n"
    "Identify the MAIN OBJECT — the item being grasped or moved by the robot arm "
    "(not the arm itself).\n\n"
    "Carefully compare the main object between the two frames across ALL of the following:\n"
    "  • SHAPE: Does it keep the same overall form? "
    "Any unexpected deformation, collapse, bending, or structural change → inconsistent.\n"
    "  • MATERIAL: Does it behave like the same material type? "
    "A rigid object that looks soft/rubbery, or a flexible object that looks stiff → inconsistent. "
    "Even subtle changes in how the object deforms or holds its shape count.\n"
    "  • SIZE: Is it approximately the same size? "
    "Any unexplained sudden change in scale → inconsistent.\n"
    "  • COLOR: Is the dominant colour stable? Sudden hue change unexplained by lighting → inconsistent.\n\n"
    "Judge whether the object remains PHYSICALLY PLAUSIBLE across the two frames:\n"
    "  A. Consistent: shape, material, size, and colour are stable — differences are clearly "
    "explained by viewpoint, grip angle, or distance only.\n"
    "  B. Inconsistent: something physically implausible changed. "
    "Do NOT default to A — if you notice any suspicious change that cannot be explained "
    "by viewpoint alone, choose B.\n\n"
    "Output only A or B on the first line, then one sentence describing what you observed."
)

PROMPT = PROMPT_SC_A2  # backward compat

PROMPT_SC_O_V5 = (
    "These two frames are from a robot manipulation video. "
    "Frame 1 is on top, Frame 2 is on bottom.\n\n"
    "Context: one or both frames may involve occlusion — the object may "
    "be partially or fully hidden by the robot arm or another object.\n\n"
    "Focus on the main object (the item being grasped or moved, not the "
    "arm itself). If one frame shows it partially hidden, judge only the "
    "visible portions.\n\n"
    "Judge whether the object looks like the SAME object in both frames — "
    "same identity, colour, shape, and appearance:\n\n"
    "  A. Consistent: the visible portions look the same — any differences "
    "are explained by occlusion angle, viewpoint, or grip only.\n"
    "  B. Inconsistent: the visible portions look genuinely different in a "
    "way that occlusion alone cannot explain — changed colour, shape, "
    "texture, or appears to be a different object.\n\n"
    "- Do NOT default to A. If you are unsure, choose B.\n"
    "- Err on the side of flagging inconsistency.\n"
    "Output only A or B on the first line, then one sentence about what you observed."
)


# ── Frame utilities ───────────────────────────────────────────────────────────

def extract_frames(video_path: str, n: int) -> list[str]:
    """Extract n evenly-spaced frames to a temp dir, return sorted paths."""
    import av
    from PIL import Image as PILImage

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    tmp = tempfile.mkdtemp(prefix="sc_a2_frames_")
    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames or 0

    # Fallback: decode all and sample
    all_frames = []
    for frame in container.decode(video=0):
        all_frames.append(frame.to_image())
    container.close()

    if len(all_frames) < 2:
        raise ValueError(f"Too few frames ({len(all_frames)}) in {video_path}")

    total = len(all_frames)
    indices = [int(i * (total - 1) / (n - 1)) for i in range(n)]
    saved = []
    for i, idx in enumerate(indices):
        p = str(Path(tmp) / f"frame_{i:03d}.jpg")
        all_frames[idx].save(p)
        saved.append(p)
    return saved


def select_pairs_midcut(frame_paths: list[str], n_pairs: int) -> list[tuple]:
    total = 2 * n_pairs
    indices = [int(i * (len(frame_paths) - 1) / (total - 1)) for i in range(total)]
    frames = [frame_paths[i] for i in indices]
    first_half  = frames[:n_pairs]
    second_half = frames[n_pairs:]
    return list(zip(first_half, second_half))


def make_stacked_image(path_a: str, path_b: str) -> Image.Image:
    """Vertical stack: Frame 1 (before) on top, Frame 2 (after) on bottom."""
    ia = Image.open(path_a).convert("RGB")
    ib = Image.open(path_b).convert("RGB")
    w = min(ia.width, ib.width, 640)
    ha = int(ia.height * w / ia.width)
    hb = int(ib.height * w / ib.width)
    ia = ia.resize((w, ha), Image.BICUBIC)
    ib = ib.resize((w, hb), Image.BICUBIC)
    canvas = Image.new("RGB", (w, ha + hb))
    canvas.paste(ia, (0, 0))
    canvas.paste(ib, (0, ha))
    return canvas


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(gpus: str):
    from transformers import AutoModel, AutoTokenizer

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    print(f"Loading InternVL3-78B on GPUs {gpus} ...", flush=True)
    model = AutoModel.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True, use_fast=False
    )
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    print("Model loaded.", flush=True)
    return model, tokenizer, transform


# ── Inference ─────────────────────────────────────────────────────────────────

def parse_answer(text: str) -> str:
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    if first.upper().startswith("A"):
        return "A"
    if first.upper().startswith("B"):
        return "B"
    for ch in text:
        if ch.upper() in ("A", "B"):
            return ch.upper()
    return "?"


def infer_pair(model, tokenizer, transform, path_a: str, path_b: str,
               prompt: str = None) -> str:
    try:
        img = make_stacked_image(path_a, path_b)
        pixel_values = transform(img).unsqueeze(0).to(torch.bfloat16).cuda()
        gen_cfg = dict(max_new_tokens=128, do_sample=False)
        used_prompt = prompt if prompt is not None else PROMPT_SC_A2
        response = model.chat(
            tokenizer, pixel_values, "<image>\n" + used_prompt,
            gen_cfg, num_patches_list=[1]
        )
        return parse_answer(response.strip())
    except Exception as e:
        print(f"  [WARN] infer_pair: {e}", file=sys.stderr)
        return "?"


def _evaluate_metric(model, tokenizer, transform, video_path: str, prompt: str) -> dict:
    try:
        frames = extract_frames(video_path, n=N_PAIRS * 2)
    except Exception as e:
        return {"score": None, "label": None, "bad_count": -1,
                "n_pairs": N_PAIRS, "pair_results": [], "na": True,
                "error": str(e)}

    if len(frames) < 2:
        return {"score": None, "label": None, "bad_count": -1,
                "n_pairs": N_PAIRS, "pair_results": [], "na": True,
                "error": "too few frames"}

    pairs = select_pairs_midcut(frames, N_PAIRS)
    results = []
    for fa, fb in pairs:
        ans = infer_pair(model, tokenizer, transform, fa, fb, prompt=prompt)
        results.append(ans)

    bad_count = sum(1 for r in results if r == "B")
    label = "B" if bad_count >= THRESHOLD else "A"
    score = round((1.0 - bad_count / len(pairs)) * 100, 1)

    return {
        "score":        score,
        "label":        label,
        "bad_count":    bad_count,
        "n_pairs":      N_PAIRS,
        "pair_results": results,
        "na":           False,
    }


def evaluate_sc_a2(model, tokenizer, transform, video_path: str) -> dict:
    return _evaluate_metric(model, tokenizer, transform, video_path, PROMPT_SC_A2)


def evaluate_sc_o(model, tokenizer, transform, video_path: str) -> dict:
    return _evaluate_metric(model, tokenizer, transform, video_path, PROMPT_SC_O_V5)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Physical consistency evaluator (SC_A2 + SC_O, InternVL3-78B)")
    parser.add_argument("--video",      default=None, help="single video path")
    parser.add_argument("--video_list", default=None, help="text file with one video path per line")
    parser.add_argument("--gpus",       default="4,5", help="comma-separated GPU IDs")
    parser.add_argument("--metric",     default="both", choices=["sc_a2", "sc_o", "both"],
                        help="which metric to run (default: both)")
    parser.add_argument("--out",        default=None,  help="output JSONL path (batch mode)")
    args = parser.parse_args()

    if not args.video and not args.video_list:
        parser.error("Provide --video or --video_list")

    model, tokenizer, transform = load_model(args.gpus)

    videos = []
    if args.video:
        videos = [args.video]
    else:
        videos = [l.strip() for l in open(args.video_list) if l.strip()]

    out_f = open(args.out, "w") if args.out else None

    for i, vp in enumerate(videos):
        print(f"[{i+1}/{len(videos)}] {vp}", flush=True)
        out = {"video_path": vp}

        if args.metric in ("sc_a2", "both"):
            result = evaluate_sc_a2(model, tokenizer, transform, vp)
            out["sc_a2"] = result
            print(f"    sc_a2  score={result['score']}  bad={result['bad_count']}/{result['n_pairs']}  "
                  f"label={result['label']}", flush=True)

        if args.metric in ("sc_o", "both"):
            result = evaluate_sc_o(model, tokenizer, transform, vp)
            out["sc_o"] = result
            print(f"    sc_o   score={result['score']}  bad={result['bad_count']}/{result['n_pairs']}  "
                  f"label={result['label']}", flush=True)

        # PCS = equal-weight mean of the two indicators, on the [0, 100]
        # percentage scale (matches PhysLawScore for cross-level comparison).
        # Since both sc_a2 and sc_o call _evaluate_metric on the same video
        # with identical frame-extraction parameters, their `na` flags are
        # always synchronised: either both compute or both are N/A. We
        # therefore do not need a graceful-degradation branch.
        if args.metric == "both":
            a, o = out["sc_a2"], out["sc_o"]
            if a.get("na") or o.get("na"):
                out["pcs"] = None
            else:
                out["pcs"] = round((a["score"] + o["score"]) / 2.0, 2)
            print(f"    PCS    {out['pcs']}", flush=True)

        if out_f:
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            out_f.flush()
        else:
            print(json.dumps(out, indent=2, ensure_ascii=False))

    if out_f:
        out_f.close()
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
