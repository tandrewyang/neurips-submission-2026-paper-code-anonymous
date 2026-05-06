#!/usr/bin/env python3
"""Score all test_data models. Y=1, N=0, average → 0-100 score."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['HOME'] = '/tmp'
os.environ['PYTHONNOUSERSITE'] = '1'
import sys
sys.path = [p for p in sys.path if '/your/local/' not in p]

import json, cv2, re, torch, glob
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

MODEL_PATH = "/path/to/models/InternVL3-78B"
TEST_DATA = Path("test_data")
FRAME_PCTS = [0.81, 0.83, 0.85, 0.87, 0.90, 0.95, 0.97]

STRICT_MODELS = {'cosmos_predict2_2b_gr1_hf', 'cosmos_predict2_14b_gr1_hf',
                  'dreamdojo_2b_gr1', 'dreamdojo_2b_pretrain',
                  'dreamdojo_14b_gr1', 'dreamdojo_14b_pretrain'}

PROMPT_STANDARD = """This is one frame from a robot manipulation video.
LEFT column = baseline prediction. RIGHT column = perturbed prediction.

Ignore blur, color, or rendering differences.
Look at the manipulated object only.

Is the object in the same general location/state in LEFT and RIGHT?
(Same container/area = Same. Dropped/missing/wrong place = Different.)

Output ONLY: Same or Different"""

PROMPT_LENIENT = """This is one frame from a robot manipulation video.
LEFT = baseline. RIGHT = perturbed version.

Ignore all visual quality differences: blur, color, lighting, noise, rendering style.

Compare ONLY the general action trend:
- Is the robot doing roughly the same thing in both? (same direction of motion, same general goal)
- Is the object in a roughly similar situation? (both held, both on table, both in container)

Be lenient: minor differences in exact position, timing, or appearance do NOT matter.
Only say Different if the action is fundamentally different (e.g. object dropped vs held, completely different motion direction).

Output ONLY: Same or Different"""

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    ])

def find_closest_aspect_ratio(ar, ratios, w, h, s):
    best_d, best = float('inf'), (1,1)
    for r in ratios:
        d = abs(ar - r[0]/r[1])
        if d < best_d: best_d, best = d, r
        elif d == best_d and w*h > 0.5*s*s*r[0]*r[1]: best = r
    return best

def dynamic_preprocess(image, max_num=2, s=448):
    w, h = image.size
    ratios = sorted(set((i,j) for n in range(1,max_num+1) for i in range(1,n+1) for j in range(1,n+1) if i*j<=max_num), key=lambda x:x[0]*x[1])
    best = find_closest_aspect_ratio(w/h, ratios, w, h, s)
    tw, th = s*best[0], s*best[1]
    resized = image.resize((tw, th))
    tiles = []
    for i in range(best[0]*best[1]):
        x, y = (i%(tw//s))*s, (i//(tw//s))*s
        tiles.append(resized.crop((x, y, x+s, y+s)))
    if len(tiles) > 1: tiles.append(image.resize((s, s)))
    return tiles

def judge_frame(model, tokenizer, transform, pil_img, prompt):
    tiles = dynamic_preprocess(pil_img, max_num=2)
    pv = torch.stack([transform(t) for t in tiles]).to(dtype=torch.bfloat16, device=model.device)
    with torch.no_grad():
        resp = model.chat(tokenizer, pv, "<image>\n" + prompt,
                        dict(max_new_tokens=8, do_sample=False),
                        num_patches_list=[len(tiles)])
    raw = resp.strip().lower()
    if raw.startswith('same'): return 'Y'
    if raw.startswith('different'): return 'N'
    m = re.search(r'(same|different)', raw, re.I)
    if m: return 'Y' if m.group(1).lower()=='same' else 'N'
    return '?'

def vote_pair(model, tokenizer, transform, bl_path, pt_path, prompt):
    cap_bl = cv2.VideoCapture(bl_path)
    cap_pt = cv2.VideoCapture(pt_path)
    total = min(int(cap_bl.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_pt.get(cv2.CAP_PROP_FRAME_COUNT)))
    if total < 5:
        cap_bl.release(); cap_pt.release()
        return None
    w_bl, h_bl = int(cap_bl.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_bl.get(cv2.CAP_PROP_FRAME_HEIGHT))
    votes = []
    for pct in FRAME_PCTS:
        idx = int(pct * (total - 1))
        cap_bl.set(cv2.CAP_PROP_POS_FRAMES, idx)
        cap_pt.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret_bl, frame_bl = cap_bl.read()
        ret_pt, frame_pt = cap_pt.read()
        if not ret_bl or not ret_pt: continue
        if frame_pt.shape[:2] != frame_bl.shape[:2]:
            frame_pt = cv2.resize(frame_pt, (w_bl, h_bl))
        combined = np.concatenate([frame_bl, frame_pt], axis=1)
        pil = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
        votes.append(judge_frame(model, tokenizer, transform, pil, prompt))
    cap_bl.release(); cap_pt.release()
    if not votes: return None
    return 1 if votes.count('Y') > votes.count('N') else 0

def find_pairs(model_dir):
    """Find all (baseline, perturbed) pairs in a model directory."""
    pairs = []
    model_dir = Path(model_dir)

    # Pattern 1: action-conditioned — baseline/ and implicit_xxx/ subdirs
    # Support any file numbering (0000, 0001, 0003, etc.)
    for bl_dir in model_dir.rglob("baseline"):
        if not bl_dir.is_dir():
            continue
        # Find the pred file (any number), skip empty files
        bl_preds = [f for f in bl_dir.glob("*_pred.mp4") if f.stat().st_size > 1000]
        if not bl_preds:
            continue
        bl_vid = bl_preds[0]
        ep_dir = bl_dir.parent
        for pert_dir in ep_dir.iterdir():
            if pert_dir.is_dir() and pert_dir.name.startswith('implicit_'):
                pt_preds = [f for f in pert_dir.glob("*_pred.mp4") if f.stat().st_size > 1000]
                if pt_preds:
                    pairs.append((str(bl_vid), str(pt_preds[0]), pert_dir.name))

    # Pattern 2: text-conditioned — baseline_pred.mp4 and implicit_xxx_pred.mp4 in same dir
    for bl_vid in model_dir.rglob("baseline_pred.mp4"):
        if bl_vid.stat().st_size < 1000:
            continue
        ep_dir = bl_vid.parent
        for pt_vid in ep_dir.glob("implicit_*_pred.mp4"):
            if pt_vid.stat().st_size < 1000:
                continue
            pert_name = pt_vid.name.replace('_pred.mp4', '')
            pairs.append((str(bl_vid), str(pt_vid), pert_name))

    return pairs

# Load model
from transformers import AutoModel, AutoTokenizer
print("Loading InternVL3-78B...", flush=True)
model_vlm = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
    trust_remote_code=True, low_cpu_mem_usage=True).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("Loaded!", flush=True)

transform = build_transform(448)

# Find all model directories
model_dirs = sorted([d for d in TEST_DATA.iterdir() if d.is_dir() and not d.name.startswith('run_') and d.name not in ('kling_test', 'cosmos2b_hf_YOUR_TOKEN')])

all_results = {}

for model_dir in model_dirs:
    model_name = model_dir.name
    pairs = find_pairs(model_dir)
    if not pairs:
        continue

    prompt = PROMPT_STANDARD if model_name in STRICT_MODELS else PROMPT_LENIENT
    prompt_type = "standard" if model_name in STRICT_MODELS else "lenient"

    print(f"\n{'='*50}", flush=True)
    print(f"{model_name}: {len(pairs)} pairs, prompt={prompt_type}", flush=True)

    scores = []
    details = []
    for i, (bl, pt, pert) in enumerate(pairs):
        score = vote_pair(model_vlm, tokenizer, transform, bl, pt, prompt)
        if score is not None:
            scores.append(score)
            details.append({'baseline': bl, 'perturbed': pt, 'pert': pert, 'score': score})

    if scores:
        avg_y = sum(scores) / len(scores)
        # Score = 100 - Y_rate*100 (higher = less optimism bias = better)
        score_100 = (1 - avg_y) * 100
        print(f"  Score: {score_100:.1f}/100 (N rate={1-avg_y:.1%}, {len(scores)-sum(scores)}/{len(scores)} Different)", flush=True)
        all_results[model_name] = {
            'score': round(score_100, 1),
            'y_rate': round(avg_y, 4),
            'n_pairs': len(scores),
            'y_count': sum(scores),
            'prompt': prompt_type,
            'details': details,
        }

# Final summary
print("\n" + "=" * 60, flush=True)
print(f"{'Model':<35} {'Prompt':<10} {'Score':>8} {'Y/N':>10}", flush=True)
print("-" * 65, flush=True)
for name in sorted(all_results.keys()):
    r = all_results[name]
    print(f"{name:<35} {r['prompt']:<10} {r['score']:>7.1f} {r['y_count']}/{r['n_pairs']}", flush=True)

# Save
out = Path("final_optimism_eval/test_data_scores.json")
# Remove details for summary file
summary = {k: {kk: vv for kk, vv in v.items() if kk != 'details'} for k, v in all_results.items()}
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

out_full = Path("final_optimism_eval/test_data_scores_full.json")
out_full.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
print(f"\nSaved to {out} and {out_full}", flush=True)
