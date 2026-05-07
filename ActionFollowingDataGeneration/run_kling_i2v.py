#!/usr/bin/env python3
"""
DashScope Kling I2V inference for Mirabench.
Supports: kling-v3-video-generation  and  kling-v3-omni-video-generation

- kling_v3 uses REST API with base64 image + first_frame_url (works)
- kling_v3_omni uses dashscope SDK VideoSynthesis with img_url (omni doesn't accept base64/first_frame_url)
"""
import os
import sys
import time
import json
import base64
import argparse
import requests
import cv2
from pathlib import Path

DATA_ROOT  = "/mnt/users/zirui/mizirui_benchmark/Mirabench_exp/video_batch_final"
OUT_ROOTS  = {
    "kling_v3":      "/mnt/users/zirui/mizirui_benchmark/Mirabench_exp/kling_v3",
    "kling_v3_omni": "/mnt/users/zirui/mizirui_benchmark/Mirabench_exp/kling_v3_omni",
}
MODEL_IDS  = {
    "kling_v3":      "kling/kling-v3-video-generation",
    "kling_v3_omni": "kling/kling-v3-omni-video-generation",
}
SUBSET_MAP = {"gen": "generalizability", "gr1": "gr1"}
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-47a3100d0ee242e1b55b1110d1503bd7")
SYNTHESIS_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
TASK_URL      = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
POLL_INTERVAL = 15
MAX_WAIT      = 600


def image_to_base64_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"


def get_first_frame_jpg(mp4_path: str, out_jpg: str) -> bool:
    cap = cv2.VideoCapture(mp4_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    cv2.imwrite(out_jpg, frame)
    return True


# ---------------------------------------------------------------------------
# kling_v3: REST API with base64 + first_frame_url
# ---------------------------------------------------------------------------
def submit_kling_v3(model_id: str, prompt: str, img_url: str) -> str | None:
    headers = {
        "Authorization":      f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type":       "application/json",
        "X-DashScope-Async":  "enable",
    }
    body = {
        "model": model_id,
        "input": {"prompt": prompt, "first_frame_url": img_url},
        "parameters": {"size": "480*640"},
    }
    try:
        rsp = requests.post(SYNTHESIS_URL, headers=headers, json=body, timeout=30)
        data = rsp.json()
        task_id = (data.get("output") or {}).get("task_id")
        if not task_id:
            print(f"[error] submit failed: {data}")
        return task_id
    except Exception as e:
        print(f"[error] submit exception: {e}")
        return None


# ---------------------------------------------------------------------------
# kling_v3_omni: dashscope SDK with img_url (accepts base64 data URI)
# ---------------------------------------------------------------------------
def submit_kling_omni(model_id: str, prompt: str, img_url: str) -> str | None:
    headers = {
        "Authorization":      f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type":       "application/json",
        "X-DashScope-Async":  "enable",
    }
    body = {
        "model": model_id,
        "input": {
            "prompt": prompt,
            "media": [{"type": "first_frame", "url": img_url}],
        },
        "parameters": {"mode": "std", "duration": 5, "audio": False},
    }
    try:
        rsp = requests.post(SYNTHESIS_URL, headers=headers, json=body, timeout=60)
        data = rsp.json()
        task_id = (data.get("output") or {}).get("task_id")
        if not task_id:
            print(f"[error] omni submit failed: {data}")
        return task_id
    except Exception as e:
        print(f"[error] omni submit exception: {e}")
        return None


# ---------------------------------------------------------------------------
# Polling + download (shared)
# ---------------------------------------------------------------------------
def poll_task(task_id: str) -> str | None:
    headers = {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"}
    url = TASK_URL.format(task_id=task_id)
    waited = 0
    while waited < MAX_WAIT:
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
        try:
            rsp = requests.get(url, headers=headers, timeout=30)
            data = rsp.json()
            output = data.get("output") or {}
            status = output.get("task_status", "")
            if status == "SUCCEEDED":
                vid_url = output.get("video_url") or ""
                if not vid_url:
                    results = output.get("results") or []
                    if results:
                        vid_url = results[0].get("url", "")
                return vid_url if vid_url else None
            elif status in ("FAILED", "CANCELED"):
                print(f"[error] task {task_id} status={status}  msg={output.get('message','')}")
                return None
            else:
                print(f"  polling ... status={status}  waited={waited}s")
        except Exception as e:
            print(f"[warn] poll error: {e}")
    print(f"[error] timed out after {MAX_WAIT}s waiting for task {task_id}")
    return None


def download_video(url: str, out_path: str) -> bool:
    try:
        rsp = requests.get(url, timeout=120, stream=True)
        with open(out_path, "wb") as f:
            for chunk in rsp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"[error] download failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Per-episode logic
# ---------------------------------------------------------------------------
def get_episodes(subset: str):
    folder = Path(DATA_ROOT) / SUBSET_MAP[subset]
    return sorted(folder.glob("episode_*"))


def run_episode(ep_dir: Path, subset: str, model_key: str, overwrite: bool):
    ep_id   = ep_dir.name.split("_")[-1]
    sub_out = SUBSET_MAP[subset]
    out_dir = Path(OUT_ROOTS[model_key]) / sub_out / f"episode_{ep_id}"
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

    print(f"[run] {model_key}  {sub_out}/episode_{ep_id}")

    img_url = image_to_base64_url(tmp_jpg)

    # Submit using the appropriate method
    model_id = MODEL_IDS[model_key]
    if "omni" in model_key:
        task_id = submit_kling_omni(model_id, prompt, img_url)
    else:
        task_id = submit_kling_v3(model_id, prompt, img_url)

    if not task_id:
        return

    print(f"  task_id={task_id}")

    vid_url = poll_task(task_id)
    if not vid_url:
        return

    if download_video(vid_url, str(out_mp4)):
        print(f"[done] saved {out_mp4}")
    else:
        print(f"[error] download failed for {ep_dir.name}")


# ---------------------------------------------------------------------------
def main():
    global DATA_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     choices=list(MODEL_IDS.keys()) + ["all"],
                        default="all", help="Which Kling model to run")
    parser.add_argument("--subset",    choices=["gen","gr1","all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data-root", default=DATA_ROOT)
    args = parser.parse_args()

    DATA_ROOT = args.data_root

    models  = list(MODEL_IDS.keys()) if args.model == "all" else [args.model]
    subsets = ["gen","gr1"]          if args.subset == "all"  else [args.subset]

    for model_key in models:
        for subset in subsets:
            episodes = get_episodes(subset)
            print(f"\n=== {model_key}  subset={subset}  {len(episodes)} episodes ===")
            for ep_dir in episodes:
                run_episode(ep_dir, subset, model_key, args.overwrite)


if __name__ == "__main__":
    main()
