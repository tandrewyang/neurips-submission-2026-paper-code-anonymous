#!/usr/bin/env python3
"""
Optimism Bias VLM Evaluation — InternVL3-78B 7-Frame Voting
Usage:
  PYTHONNOUSERSITE=1 /path/to/python/bin/python eval_optimism.py --batch dreamdojo_14b
  PYTHONNOUSERSITE=1 /path/to/python/bin/python eval_optimism.py --batch happyhorse_i2v --lenient
  PYTHONNOUSERSITE=1 /path/to/python/bin/python eval_optimism.py  # run all
"""
import os
os.environ.setdefault('HOME', '/tmp')
os.environ.setdefault('PYTHONNOUSERSITE', '1')

import argparse, json, cv2, re, torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from collections import Counter

MODEL_PATH = "/path/to/models/InternVL3-78B"
FRAME_PCTS = [0.81, 0.83, 0.85, 0.87, 0.90, 0.95, 0.97]

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

# Batches that use separate baseline+perturbed videos (hi-res)
HIRES_BATCHES = {'happyhorse_i2v', 'wan21_i2v_14b'}

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

def evaluate_sample_combined(model, tokenizer, transform, video_path, prompt):
    """For 3-panel combined videos: crop out GT (left 1/3), judge right 2/3."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    votes = []
    for pct in FRAME_PCTS:
        idx = int(pct * (total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        cropped = frame[:, fw//3:, :]
        pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        votes.append(judge_frame(model, tokenizer, transform, pil, prompt))
    cap.release()
    return votes

def evaluate_sample_hires(model, tokenizer, transform, bl_path, pt_path, prompt):
    """For separate baseline+perturbed videos: concatenate side by side."""
    cap_bl = cv2.VideoCapture(bl_path)
    cap_pt = cv2.VideoCapture(pt_path)
    total = min(int(cap_bl.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_pt.get(cv2.CAP_PROP_FRAME_COUNT)))
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
    cap_bl.release()
    cap_pt.release()
    return votes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=str, default=None, help='Specific batch to evaluate')
    parser.add_argument('--lenient', action='store_true', help='Use lenient prompt')
    parser.add_argument('--data', type=str, default='final_optimism_eval/test_samples.json')
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    samples = json.load(open(args.data))
    if args.batch:
        samples = [s for s in samples if s['batch'] == args.batch]
    samples = [s for s in samples if s['MA9'] in ('Y', 'Y?', 'N')]
    print(f"Samples: {len(samples)}")

    prompt = PROMPT_LENIENT if args.lenient else PROMPT_STANDARD

    from transformers import AutoModel, AutoTokenizer
    print("Loading InternVL3-78B...")
    model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, low_cpu_mem_usage=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("Loaded!")

    transform = build_transform(448)
    results = []

    for i, s in enumerate(samples):
        batch, tid = s['batch'], s['task_id']
        if batch in HIRES_BATCHES:
            bl = f"results/human_annotation_{batch}/videos/{tid}/baseline.mp4"
            pt = f"results/human_annotation_{batch}/videos/{tid}/perturbed.mp4"
            votes = evaluate_sample_hires(model, tokenizer, transform, bl, pt, prompt)
        else:
            vpath = f"Human_annotation/{batch}/{tid}.mp4"
            votes = evaluate_sample_combined(model, tokenizer, transform, vpath, prompt)

        y_c, n_c = votes.count('Y'), votes.count('N')
        pred = 'Y' if y_c > n_c else 'N'
        results.append({'task_id': tid, 'batch': batch, 'gt_ma9': s['MA9'],
                        'pred': pred, 'votes': votes})

        if (i+1) % 20 == 0:
            c = sum(1 for r in results if (r['pred']=='Y' and r['gt_ma9'] in ('Y','Y?')) or (r['pred']=='N' and r['gt_ma9']=='N'))
            print(f"  [{i+1}/{len(samples)}] acc={c}/{len(results)}={c/len(results):.1%}")

    # Metrics
    print("\n" + "=" * 60)
    correct = sum(1 for r in results if (r['pred']=='Y' and r['gt_ma9'] in ('Y','Y?')) or (r['pred']=='N' and r['gt_ma9']=='N'))
    print(f"Accuracy (Y+Y?+N): {correct}/{len(results)} = {correct/len(results):.1%}")

    yn = [r for r in results if r['gt_ma9'] in ('Y','N')]
    if yn:
        c_yn = sum(1 for r in yn if (r['pred']=='Y' and r['gt_ma9']=='Y') or (r['pred']=='N' and r['gt_ma9']=='N'))
        print(f"Accuracy (Y+N only): {c_yn}/{len(yn)} = {c_yn/len(yn):.1%}")

    out = Path(f"final_optimism_eval/eval_results{'_'+args.batch if args.batch else '_all'}{'_lenient' if args.lenient else ''}.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()
