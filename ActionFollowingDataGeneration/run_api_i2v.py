#!/usr/bin/env python3
"""
DashScope API-based I2V inference for Mirabench.
Supports: happyhorse-1.0-i2v  and  wanx2.1-i2v-plus

Images are uploaded to DashScope OSS via their file-upload endpoint,
then the returned URL is passed to the video-synthesis API.
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
    "happyhorse":      "/mnt/users/zirui/mizirui_benchmark/Mirabench_exp/happyhorse",
    "wanx21_i2v_plus": "/mnt/users/zirui/mizirui_benchmark/Mirabench_exp/wanx21_i2v_plus",
}
MODEL_IDS  = {
    "happyhorse":      "happyhorse-1.0-i2v",
    "wanx21_i2v_plus": "wanx2.1-i2v-plus",
}
SUBSET_MAP = {"gen": "generalizability", "gr1": "gr1"}
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-c354af57d6e54f7191e7c255cc54ab57")
SYNTHESIS_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
TASK_URL      = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
POLL_INTERVAL = 10   # seconds between status polls
MAX_WAIT      = 600  # 10 minutes max per video


# ──────────────────────────────────────────────────────────────────
# Image upload helpers
# ──────────────────────────────────────────────────────────────────

def upload_image_dashscope(image_path: str) -> str:
    """
    Upload image to DashScope OSS and return a usable HTTPS URL.

    DashScope provides a two-step upload:
    1. POST /api/v1/uploads  →  get upload_url + object_key
    2. PUT upload_url with file content
    3. Use  oss://<object_key>  as img_url  (DashScope resolves this internally)

    If this fails, we fall back to a base64 data-URI (works for some models).
    """
    headers_auth = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    # Step 1: request upload slot
    try:
        rsp = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/uploads",
            headers=headers_auth,
            json={"filename": Path(image_path).name, "purpose": "assistants"},
            timeout=30,
        )
        data = rsp.json()
        upload_url  = data.get("upload_url") or data.get("data", {}).get("upload_url")
        object_key  = data.get("object_key") or data.get("data", {}).get("object_key")

        if upload_url and object_key:
            # Step 2: PUT the file
            with open(image_path, "rb") as f:
                put_rsp = requests.put(upload_url, data=f,
                                       headers={"Content-Type": "image/jpeg"},
                                       timeout=60)
            if put_rsp.status_code in (200, 204):
                return f"oss://{object_key}"
    except Exception as e:
        print(f"[warn] OSS upload failed ({e}), trying base64 fallback")

    # Fallback: base64 data URI (accepted by some DashScope I2V endpoints)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:image/jpeg;base64,{b64}"


def get_first_frame_jpg(mp4_path: str, out_jpg: str, min_side: int = 0) -> bool:
    cap = cv2.VideoCapture(mp4_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False
    if min_side > 0:
        h, w = frame.shape[:2]
        if h < min_side or w < min_side:
            scale = max(min_side / h, min_side / w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(out_jpg, frame)
    return True


# ──────────────────────────────────────────────────────────────────
# DashScope video-synthesis helpers
# ──────────────────────────────────────────────────────────────────

def submit_video_job(model_id: str, prompt: str, img_url: str) -> str | None:
    headers = {
        "Authorization":      f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type":       "application/json",
        "X-DashScope-Async":  "enable",
    }
    # happyhorse uses "media" (list); wanx models use "img_url" (string)
    if "happyhorse" in model_id:
        input_data = {"prompt": prompt, "media": [{"url": img_url}]}
    else:
        input_data = {"prompt": prompt, "img_url": img_url}
    body = {
        "model": model_id,
        "input": input_data,
        "parameters": {
            "size": "480*640",
        },
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


def poll_task(task_id: str) -> str | None:
    """Poll until done; return video URL or None."""
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
                # video URL is in output.video_url or output.results[0].url
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
                print(f"  polling … status={status}  waited={waited}s")
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


# ──────────────────────────────────────────────────────────────────
# Per-episode logic
# ──────────────────────────────────────────────────────────────────

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
    min_side = 300 if model_key == "happyhorse" else 0
    if not get_first_frame_jpg(str(mp4_path), tmp_jpg, min_side=min_side):
        print(f"[warn] cannot read first frame from {mp4_path}, skip")
        return

    print(f"[run] {model_key}  {sub_out}/episode_{ep_id}")

    # upload image → get URL
    # happyhorse only accepts https:// URLs in the url field; base64 data URIs work reliably
    if model_key == "happyhorse":
        with open(tmp_jpg, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        img_url = f"data:image/jpeg;base64,{b64}"
    else:
        img_url = upload_image_dashscope(tmp_jpg)

    # submit
    task_id = submit_video_job(MODEL_IDS[model_key], prompt, img_url)
    if not task_id:
        return

    print(f"  task_id={task_id}")

    # poll
    vid_url = poll_task(task_id)
    if not vid_url:
        return

    # download
    if download_video(vid_url, str(out_mp4)):
        print(f"[done] saved {out_mp4}")
    else:
        print(f"[error] download failed for {ep_dir.name}")


# ──────────────────────────────────────────────────────────────────

def main():
    global DATA_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     choices=list(MODEL_IDS.keys()) + ["all"],
                        default="all", help="Which API model to run")
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
