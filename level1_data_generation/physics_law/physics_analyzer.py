"""
Free Fall Physics Analyzer (SAM2 + polyfit)

Pipeline:
  1. VLM locates the target object center (x, y) in frame 0 (single API call).
  2. SAM2.1-Hiera-Large video propagation -> per-frame mask centroid trajectory.
  3. Velocity sign flips segment the trajectory (pure geometry).
  4. Each segment is fit with polyfit(t, y, deg=2) to y = 1/2 g t^2 + v0 t + y0.
     R^2 measures fit quality; g_fit = 2|a| is the normalized acceleration.
"""
import os
import cv2
import json
import time
import base64
import shutil
import tempfile
import numpy as np
from openai import OpenAI

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# VLM client: configured entirely via environment variables.
#   PHYSBENCH_VLM_MODEL     model id (default: gpt-4.1-mini)
#   PHYSBENCH_VLM_API_KEY   API key (or OPENAI_API_KEY)
#   PHYSBENCH_VLM_BASE_URL  OpenAI-compatible endpoint (default: api.openai.com)
VLM_MODEL = os.environ.get("PHYSBENCH_VLM_MODEL", "gpt-4.1-mini")
VLM_RETRIES = 2  # retries on network errors
_API_KEY = os.environ.get("PHYSBENCH_VLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
_BASE_URL = os.environ.get("PHYSBENCH_VLM_BASE_URL", "https://api.openai.com/v1")
if not _API_KEY:
    raise RuntimeError(
        "Missing VLM API key. Set environment variable PHYSBENCH_VLM_API_KEY or OPENAI_API_KEY."
    )
client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)


def _vlm_call_with_retry(messages, max_tokens=2000, timeout=60):
    # Note: Gemini 2.5 Pro/Flash spends "thinking" tokens before output;
    # max_tokens >= 2000 is needed to avoid truncated JSON.
    """Wrapper around client.chat.completions.create with retries."""
    last_err = None
    for attempt in range(VLM_RETRIES + 1):
        try:
            return client.chat.completions.create(
                model=VLM_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as e:
            last_err = e
            if attempt < VLM_RETRIES:
                print(f"    VLM failed ({type(e).__name__}), retry {attempt + 1}/{VLM_RETRIES}...")
                time.sleep(1.5 * (attempt + 1))
    raise last_err

# SAM2 checkpoint path
_PARTA_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_CKPT = os.path.join(_PARTA_DIR, "checkpoints", "sam2.1_hiera_large.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

_SAM2_PREDICTOR = None  # lazy init


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────
def encode_frame_base64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def _get_sam2_predictor():
    """Lazy load SAM2 video predictor on CUDA/MPS/CPU."""
    global _SAM2_PREDICTOR
    if _SAM2_PREDICTOR is not None:
        return _SAM2_PREDICTOR
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    _SAM2_PREDICTOR = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device=dev)
    _SAM2_PREDICTOR._dev = dev
    return _SAM2_PREDICTOR


def _dump_frames_to_dir(video_path: str, out_dir: str) -> tuple[int, float, int, int]:
    """Dump each frame as JPEG (00000.jpg ...) for SAM2. Returns (n_frames, fps, h, w)."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(out_dir, f"{i:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        i += 1
    cap.release()
    return i, fps, h, w


# ─────────────────────────────────────────────────────────────────────────────
# 1. VLM frame-0 object localization (single API call)
# ─────────────────────────────────────────────────────────────────────────────
def vlm_locate_first_frame(frame: np.ndarray) -> dict:
    """Have the VLM return the target object's bbox in frame 0, plus negative points (gripper / arm) for SAM2.

    Returns:
      {
        "bbox": [x1, y1, x2, y2] normalized to 0-1 (object only, no gripper),
        "x_norm", "y_norm": object center (also exposed as a convenience field),
        "gripper": [x, y] normalized, or None,
        "object": object name string,
      }
    """
    import re
    h, w = frame.shape[:2]
    target_w = 640
    scale = target_w / w
    small = cv2.resize(frame, (target_w, int(h * scale)))
    b64 = encode_frame_base64(small)

    prompt = """This is the first frame of a free-fall video. A robot end-effector (gripper / arm / chain) may be holding an object that will be released and fall; or the object may already be in free flight.

Output four fields:

1. `is_released_at_f0`: bool — whether the object is already fully detached from the gripper in frame 0.

2. `bbox: [x1, y1, x2, y2]` — tight bounding box of the object (normalized 0-1, object body only — exclude the gripper/arm/background).

3. `negative_points`: list of [x, y] — 2 to 4 normalized points telling the segmenter "these are NOT the object".
   Pick visible candidates from:
     - upper arm / manipulator segments (far from the object)
     - gripper / clamp base (not touching the object)
     - cables, chains, background structures
     - tabletop / floor
   Each negative point must be at least 5% (normalized) away from the bbox edge.
   If no obvious mechanical structure, return an empty list [].

4. `object`: str — name of the SPECIFIC object being held / falling.
   - Must be a single object, not a phrase like "fruit on a plate".
   - Prefer the object the gripper is currently grabbing (even if it sits in a plate on the table).
   - Use common nouns: apple, banana, dragon fruit, cucumber, glass bottle, can, cup, ball, ...
   - Do NOT use container names (plate, tray, tabletop) as the answer.

Output JSON only (object must be the specific held object):
{"is_released_at_f0": false, "bbox": [0.62, 0.38, 0.80, 0.55], "negative_points": [[0.50, 0.20], [0.80, 0.10], [0.25, 0.45]], "object": "<object_name>"}

If no object is found: {"is_released_at_f0": null, "bbox": null, "negative_points": [], "object": "none"}"""

    try:
        res = _vlm_call_with_retry(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=2000,
            timeout=60,
        )
        raw = res.choices[0].message.content.strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        bbox = data.get("bbox")
        neg_pts = data.get("negative_points") or []
        # Filter out negative points too close to the bbox (<5%)
        if bbox and len(bbox) == 4:
            pad = 0.05
            x1, y1, x2, y2 = bbox
            filtered_neg = []
            for pt in neg_pts:
                if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                    continue
                px, py = float(pt[0]), float(pt[1])
                if (x1 - pad) <= px <= (x2 + pad) and (y1 - pad) <= py <= (y2 + pad):
                    continue  # too close to bbox, drop
                filtered_neg.append([round(px, 4), round(py, 4)])
            neg_pts = filtered_neg[:4]  # at most 4
        else:
            neg_pts = []
        obj_name = data.get("object", "")
        is_released = data.get("is_released_at_f0")

        if bbox and len(bbox) == 4:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            return {
                "bbox": [round(float(x), 4) for x in bbox],
                "x_norm": round(float(cx), 4),
                "y_norm": round(float(cy), 4),
                "negative_points": neg_pts,
                "gripper": neg_pts[0] if neg_pts else None,  # primary negative point (gripper)
                "object": obj_name,
                "is_released_at_f0": bool(is_released) if is_released is not None else None,
            }
        return {"bbox": None, "x_norm": None, "y_norm": None, "negative_points": [],
                "gripper": None, "object": obj_name or "none",
                "is_released_at_f0": is_released}
    except Exception as e:
        return {"bbox": None, "x_norm": None, "y_norm": None, "negative_points": [],
                "gripper": None, "object": "", "is_released_at_f0": None,
                "error": str(e)}


def vlm_find_release_frame(video_path: str, n_samples: int = 8) -> dict:
    """Sample n_samples evenly-spaced frames and ask the VLM which is the first frame where the object is fully detached from the gripper (free flight),
    then return the object bbox at that frame.

    Returns:
      {
        "release_frame_idx": int — absolute frame index (0-based),
        "release_sample_idx": int — index within sampled frames (1-based, diagnostic),
        "bbox": [x1, y1, x2, y2] normalized (at release frame),
        "object": str,
        "sample_indices": list[int] sampled raw frame indices,
        "reason": str (if release frame was not found, explanation)
      }
    On failure release_frame_idx = None.
    """
    import re
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < n_samples:
        n_samples = max(2, total)
    step = total / n_samples
    sample_indices = [int(round(i * step)) for i in range(n_samples)]
    sample_indices = sorted(set(min(total - 1, x) for x in sample_indices))

    frames = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if not ret:
            continue
        frames.append((idx, f))
    cap.release()

    if len(frames) < 2:
        return {"release_frame_idx": None, "reason": "Could not read enough sampled frames"}

    # Tile frames horizontally (each scaled to 320 wide), with a frame-index label on top
    tile_w = 320
    tiles = []
    for pos, (fidx, f) in enumerate(frames, start=1):
        h0, w0 = f.shape[:2]
        sc = tile_w / w0
        t = cv2.resize(f, (tile_w, int(h0 * sc)))
        label = f"F{pos} (frame {fidx})"
        cv2.rectangle(t, (0, 0), (tile_w, 24), (0, 0, 0), -1)
        cv2.putText(t, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    grid = np.hstack(tiles)
    # Save the sampled grid to /tmp for inspection (the image the VLM saw)
    try:
        sample_grid_path = os.path.join(tempfile.gettempdir(),
                                       f"vlm_release_grid_{os.path.basename(video_path)}.jpg")
        cv2.imwrite(sample_grid_path, grid)
        _GRID_PATH_LAST = sample_grid_path
    except Exception:
        sample_grid_path = None
    b64 = encode_frame_base64(grid)

    prompt = f"""Below is a horizontal grid of {len(frames)} evenly-sampled frames from a "robot grabs object -> releases -> object falls" video (F1 earliest, F{len(frames)} latest).

Label each frame independently (look at the frame in isolation). Pick one:
- "held"     — the object is still in contact with / held by the gripper
- "released" — the object is fully detached and in free flight (airborne, unsupported)
- "landed"   — the object has landed / become static
- "unclear"  — the frame is blurry or the object is not visible

Also report what the object is.

Output JSON only (labels length must equal {len(frames)}). For `object` give the specific object (apple/banana/dragon fruit/cucumber/glass bottle/ball/...) — do NOT answer with a container (plate/tray):
{{"labels": ["held", "held", "released", "released", "released", "landed", "landed", "landed"],
  "object": "<object_name>"}}"""

    try:
        res = _vlm_call_with_retry(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=2000,
            timeout=60,
        )
        raw = res.choices[0].message.content.strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        labels = data.get("labels") or []
        obj = data.get("object", "")
        sample_idx_list = [fi for fi, _ in frames]

        first_rel_pos = None
        for i, lbl in enumerate(labels):
            if isinstance(lbl, str) and lbl.lower() == "released":
                first_rel_pos = i
                break

        if first_rel_pos is None or first_rel_pos >= len(sample_idx_list):
            return {
                "release_frame_idx": None, "release_sample_idx": None,
                "bbox": None, "negative_points": [], "object": obj or "",
                "sample_indices": sample_idx_list, "labels": labels,
                "sample_grid_path": sample_grid_path,
                "reason": "VLM did not label any frame as released",
            }

        # Take the first released sample; if the previous frame is "held",
        # bias the release toward "released" (safer to start SAM2 a frame or two later than to drag the held tail in).
        released_sample = sample_idx_list[first_rel_pos]
        if first_rel_pos > 0 and str(labels[first_rel_pos - 1]).lower() == "held":
            held_sample = sample_idx_list[first_rel_pos - 1]
            # 2/3 weight on released
            release_frame_idx = int(round(held_sample * (1/3) + released_sample * (2/3)))
        else:
            release_frame_idx = released_sample

        # Step 2: re-read the release frame at full resolution and ask the VLM for a precise bbox + negative points.
        cap2 = cv2.VideoCapture(video_path)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, release_frame_idx)
        ret2, release_frame = cap2.read()
        cap2.release()

        bbox = None
        neg_pts = []
        if ret2:
            bbox_info = vlm_locate_first_frame(release_frame)
            bbox = bbox_info.get("bbox")
            neg_pts = bbox_info.get("negative_points") or []
            if not obj:
                obj = bbox_info.get("object", "")

        if bbox is None:
            return {
                "release_frame_idx": None, "release_sample_idx": None,
                "bbox": None, "negative_points": [], "object": obj,
                "sample_indices": sample_idx_list, "labels": labels,
                "sample_grid_path": sample_grid_path,
                "reason": "VLM could not locate object bbox at the release frame",
            }

        return {
            "release_frame_idx": int(release_frame_idx),
            "release_sample_idx": first_rel_pos + 1,
            "bbox": bbox,
            "negative_points": neg_pts,
            "gripper": neg_pts[0] if neg_pts else None,
            "object": obj,
            "labels": labels,
            "sample_indices": sample_idx_list,
            "sample_grid_path": sample_grid_path,
            "reason": "ok",
        }
    except Exception as e:
        return {"release_frame_idx": None, "release_sample_idx": None,
                "bbox": None, "negative_points": [], "gripper": None,
                "object": "", "labels": [],
                "sample_indices": [fi for fi, _ in frames],
                "reason": f"VLM error: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# 2. SAM2 video propagation -> per-frame mask centroid
# ─────────────────────────────────────────────────────────────────────────────
def sam2_track_centroids(
    video_path: str,
    positive_point: tuple[float, float] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    negative_point: tuple[float, float] | None = None,
    negative_points: list | None = None,
    prompt_frame_idx: int = 0,
    keep_masks: bool = False,
    drift_area_ratio: float = 2.5,
) -> dict:
    """SAM2.1 video propagation. Supports starting from any frame (prompt_frame_idx), multiple negative points, and mask-drift detection.

    Args:
        prompt_frame_idx: frame at which the prompt is provided and forward propagation begins (0 = start of video).
        negative_points: list of [x, y] negative points (merged with the single negative_point arg).
        drift_area_ratio: if mask area exceeds drift_area_ratio * initial, the centroid is marked invalid.

    """
    import torch
    predictor = _get_sam2_predictor()

    work_dir = tempfile.mkdtemp(prefix="sam2_track_")
    frame_dir = os.path.join(work_dir, "frames")
    try:
        n, fps, h, w = _dump_frames_to_dir(video_path, frame_dir)
        if n < 4:
            raise RuntimeError(f"too few frames in video ({n} < 4)")

        prompt_frame_idx = max(0, min(prompt_frame_idx, n - 1))

        # Construct SAM2 prompt
        prompt_kwargs = {"frame_idx": prompt_frame_idx, "obj_id": 1}
        prompt_desc = [f"frame_idx={prompt_frame_idx}"]

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1 = float(np.clip(x1, 0, 1)) * (w - 1)
            y1 = float(np.clip(y1, 0, 1)) * (h - 1)
            x2 = float(np.clip(x2, 0, 1)) * (w - 1)
            y2 = float(np.clip(y2, 0, 1)) * (h - 1)
            prompt_kwargs["box"] = np.array([x1, y1, x2, y2], dtype=np.float32)
            prompt_desc.append(f"bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

        pts, lbls = [], []
        if positive_point is not None:
            px = float(np.clip(positive_point[0], 0.01, 0.99)) * (w - 1)
            py = float(np.clip(positive_point[1], 0.01, 0.99)) * (h - 1)
            pts.append([px, py]); lbls.append(1)
            prompt_desc.append(f"pos=({px:.0f},{py:.0f})")
        # Merge the single negative_point and the list form
        all_neg = []
        if negative_point is not None:
            all_neg.append(negative_point)
        if negative_points:
            all_neg.extend(negative_points)
        for npt in all_neg:
            nx = float(np.clip(npt[0], 0.01, 0.99)) * (w - 1)
            ny = float(np.clip(npt[1], 0.01, 0.99)) * (h - 1)
            pts.append([nx, ny]); lbls.append(0)
            prompt_desc.append(f"neg=({nx:.0f},{ny:.0f})")

        if pts:
            prompt_kwargs["points"] = np.array(pts, dtype=np.float32)
            prompt_kwargs["labels"] = np.array(lbls, dtype=np.int32)

        if "box" not in prompt_kwargs and not pts:
            raise ValueError("Either bbox or positive_point must be provided")

        print(f"  SAM2 prompt: {' + '.join(prompt_desc)}")

        # Pre-fill placeholders aligned with video frames (pre-prompt frames are marked invalid)
        centroids_by_fi: dict[int, dict] = {
            fi: {"frame_idx": fi, "x_px": None, "y_px": None,
                 "x_norm": None, "y_norm": None,
                 "mask_area": 0, "valid": False,
                 "pre_release": fi < prompt_frame_idx}
            for fi in range(n)
        }
        masks_by_fi: dict[int, np.ndarray] = {} if keep_masks else None

        # Track initial mask areas as baseline for drift detection
        initial_areas: list[int] = []

        with torch.inference_mode():
            state = predictor.init_state(video_path=frame_dir)
            predictor.add_new_points_or_box(state, **prompt_kwargs)
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                m = (mask_logits[0, 0] > 0.0).cpu().numpy()
                area = int(m.sum())
                if keep_masks:
                    masks_by_fi[frame_idx] = m.astype(np.bool_)
                if area < 30:  # mask lost
                    continue

                # Drift detection: only flag when the mask grows far beyond
                # the initial baseline (swallowed background). The lower bound
                # is loose (/6) because early frames often momentarily include the gripper.
                if len(initial_areas) < 3:
                    initial_areas.append(area)
                    is_drift = False
                else:
                    baseline = float(np.median(initial_areas))
                    is_drift = (
                        area > baseline * drift_area_ratio
                        or area < baseline / 6.0
                    )

                if is_drift:
                    # mark invalid but keep area for diagnostics
                    centroids_by_fi[frame_idx] = {
                        "frame_idx": frame_idx, "x_px": None, "y_px": None,
                        "x_norm": None, "y_norm": None,
                        "mask_area": area, "valid": False,
                        "pre_release": frame_idx < prompt_frame_idx,
                        "drift": True,
                    }
                    continue

                ys, xs = np.where(m)
                cx = float(xs.mean()); cy = float(ys.mean())
                centroids_by_fi[frame_idx] = {
                    "frame_idx": frame_idx,
                    "x_px": round(cx, 2), "y_px": round(cy, 2),
                    "x_norm": round(cx / (w - 1), 4), "y_norm": round(cy / (h - 1), 4),
                    "mask_area": area, "valid": True,
                    "pre_release": frame_idx < prompt_frame_idx,
                }

        # Re-align centroids to video frame order
        centroids = [centroids_by_fi[fi] for fi in range(n)]
        result = {
            "fps": fps, "h": h, "w": w, "n_frames": n,
            "centroids": centroids,
            "prompt_frame_idx": prompt_frame_idx,
        }
        if keep_masks:
            empty = np.zeros((h, w), dtype=np.bool_)
            result["masks"] = [masks_by_fi.get(fi, empty) for fi in range(n)]
        return result
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Trajectory segmentation: rest / fall / rise (rebound)
# ─────────────────────────────────────────────────────────────────────────────
def segment_trajectory(centroids: list[dict], fps: float) -> dict:
    """Segment the trajectory by the sign of y'(t), tolerating a static middle phase (e.g. "lift -> pause -> release -> fall").

    Segment types:
      - "rest": stationary (|v| under threshold persistently)
      - "lift": upward motion BEFORE the first fall (object held aloft); NOT scored.
      - "fall": downward motion (v_y > 0)
      - "rise": upward motion AFTER the first fall (rebound)

    Returns {segments, n_bounces, has_bounce, n_valid}.
    """
    valid = [(c["frame_idx"], c["y_norm"]) for c in centroids if c["valid"]]
    n = len(valid)
    if n < 6:
        return {"error": f"too few valid trajectory points ({n} < 6)"}

    valid.sort(key=lambda x: x[0])
    frames = np.array([v[0] for v in valid])
    y = np.array([v[1] for v in valid])
    t = frames / fps

    # Frame-to-frame velocity (y increases downward, so v > 0 means falling)
    dt = np.diff(t)
    dy = np.diff(y)
    v = np.where(dt > 0, dy / dt, 0.0)

    # Use 95th percentile instead of max to be robust to single-frame outliers
    v_ref = float(np.percentile(np.abs(v), 95)) if len(v) else 0.0
    move_thresh = max(0.08, v_ref * 0.15)  # y_norm/s
    min_v = max(0.05, v_ref * 0.12)

    # 1. Tag each v[i] as rest/move
    lbl_v = ["move" if abs(vi) > move_thresh else "rest" for vi in v]

    # 2. Compress consecutive identical labels into raw segments
    raw = []  # (pt_start, pt_end, label)
    i = 0
    while i < len(lbl_v):
        j = i
        while j < len(lbl_v) and lbl_v[j] == lbl_v[i]:
            j += 1
        raw.append((i, j, lbl_v[i]))
        i = j

    # 3. Drop very short rest segments (< 2 intervals) as noise -> merge into adjacent move
    merged = []
    for (a, b, lbl) in raw:
        # Short rest sandwiched between moves: absorb into the preceding move
        if lbl == "rest" and (b - a) < 2 and merged and merged[-1][2] == "move":
            prev_a, _, _ = merged.pop()
            merged.append((prev_a, b, "move"))
        # Coalesce two adjacent moves (after absorbing the intervening rest)
        elif merged and merged[-1][2] == "move" and lbl == "move":
            prev_a, _, _ = merged.pop()
            merged.append((prev_a, b, "move"))
        else:
            merged.append((a, b, lbl))

    # 4. Split each move by velocity sign flips (bounces); require k consecutive same-sign frames after flip
    FLIP_CONFIRM = 2  # need this many same-sign frames after flip
    final = []
    bounce_count = 0
    for (a, b, lbl) in merged:
        if lbl == "rest":
            final.append(("rest", a, b))
            continue
        flips = [a]
        k = a + 1
        while k < b:
            if (v[k - 1] * v[k] < -1e-4
                    and abs(v[k - 1]) > min_v and abs(v[k]) > min_v):
                # Verify: the next FLIP_CONFIRM frames must keep the new sign
                sign_new = 1 if v[k] > 0 else -1
                confirmed = True
                for kk in range(k + 1, min(k + FLIP_CONFIRM, b)):
                    s = 1 if v[kk] > 0 else (-1 if v[kk] < 0 else 0)
                    if s == 0 or s != sign_new:
                        confirmed = False
                        break
                if confirmed:
                    flips.append(k)
                    k += FLIP_CONFIRM  # skip frames already verified
                    continue
            k += 1
        flips.append(b)
        # Drop flips too close together (keep each sub-segment >= 2 velocity intervals)
        cleaned = [flips[0]]
        for f in flips[1:]:
            if f - cleaned[-1] >= 2:
                cleaned.append(f)
        if cleaned[-1] != b:
            cleaned[-1] = b
        flips = cleaned
        for kk in range(len(flips) - 1):
            aa, bb = flips[kk], flips[kk + 1]
            if bb - aa < 2:
                continue
            avg_v = float(np.mean(v[aa:bb]))
            sub_type = "fall" if avg_v > 0 else "rise"
            final.append((sub_type, aa, bb))
            if kk > 0:
                bounce_count += 1

    # 5. Mark "lift": rise segments occurring before the first fall
    first_fall_pos = None
    for idx, (tp, _, _) in enumerate(final):
        if tp == "fall":
            first_fall_pos = idx
            break
    segments = []
    for idx, (tp, a, b) in enumerate(final):
        cur_type = tp
        if tp == "rise" and (first_fall_pos is None or idx < first_fall_pos):
            cur_type = "lift"  # lifting phase before release
        segments.append({
            "type": cur_type,
            "pt_start": a, "pt_end": b,
            "frame_start": int(frames[a]), "frame_end": int(frames[b]),
            "t_start": round(float(t[a]), 3), "t_end": round(float(t[b]), 3),
            "n_points": b - a + 1,
        })

    return {
        "segments": segments,
        "n_bounces": bounce_count,
        "has_bounce": bounce_count > 0,
        "n_valid": n,
        "has_first_fall": first_fall_pos is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.5. Event-level features: impact, landing drift, bounce decay
# Discrete physical events; independent of per-segment polyfit quality.
# ─────────────────────────────────────────────────────────────────────────────
def extract_event_features(centroids: list[dict], fps: float) -> dict:
    """Compute three event-level features from the trajectory.
      - velocity_drop: ratio of speed change at impact (1.0 = perfect impact, 0.0 = gradual stop = floating anomaly).
      - post_landing_drift: post-impact position spread / 0.10 (typical object scale); > 1 indicates hovering / drifting after landing.
      - bounce_decay: ratio of second peak height to the first (should be < 1 for energy dissipation; > 1 is unphysical).
      - has_impact: whether a clean stop frame was found.
    """
    pts = [(c["frame_idx"], c.get("x_norm"), c.get("y_norm"))
           for c in centroids if c.get("valid")
           and c.get("x_norm") is not None and c.get("y_norm") is not None]
    out = {"velocity_drop": None, "post_landing_drift": None,
           "bounce_decay": None, "has_impact": False, "n_bounces_detected": 0}
    if len(pts) < 8:
        return out
    pts.sort(key=lambda v: v[0])
    fr = np.array([p[0] for p in pts])
    x = np.array([p[1] for p in pts]); y = np.array([p[2] for p in pts])
    t = fr / fps
    dt = np.diff(t); dy = np.diff(y); dx = np.diff(x)
    vy = np.where(dt > 0, dy / dt, 0.0)
    vx = np.where(dt > 0, dx / dt, 0.0)
    speed = np.sqrt(vy ** 2 + vx ** 2)

    # Impact: first index where |speed| stays near zero for 4+ consecutive frames
    rest_thresh = max(0.05, float(np.percentile(np.abs(speed), 95) * 0.10))
    impact = None
    for i in range(len(speed) - 4):
        if all(s < rest_thresh for s in speed[i:i + 4]):
            impact = i; break
    out["has_impact"] = impact is not None

    if impact is not None and impact >= 3:
        v_b = float(np.mean(speed[max(0, impact - 3):impact]))
        v_a = float(np.mean(speed[impact:impact + 4]))
        out["velocity_drop"] = (v_b - v_a) / max(v_b, 1e-6)
    if impact is not None and impact + 8 < len(y):
        y_post = y[impact:impact + 8]
        out["post_landing_drift"] = float(np.max(y_post) - np.min(y_post)) / 0.10

    # Rebound peaks: positions where vy flips from positive (down) to negative (up)
    sign_changes = [i for i in range(1, len(vy)) if vy[i - 1] > 0 and vy[i] < 0]
    peaks = []
    for sc in sign_changes:
        win = y[max(0, sc - 2):min(len(y), sc + 3)]
        if len(win) > 0:
            peaks.append(float(np.min(win)))
    out["n_bounces_detected"] = len(peaks)
    if len(peaks) >= 2:
        ground = float(np.max(y))
        h1 = ground - peaks[0]; h2 = ground - peaks[1]
        if h1 > 0.02:
            out["bounce_decay"] = h2 / h1
    return out


def compute_event_score(features: dict) -> dict:
    """Map each event feature to a [0, 1] sub-score and combine them.

    Mapping follows physical intuition:
      - high velocity_drop (sharp impact) -> high score
      - low post_landing_drift (stable landing) -> high score
      - has_impact True (clean stop) -> high score
      - low bounce_decay (< 0.7, strong decay) -> high score
    """
    vd = features.get("velocity_drop")
    pld = features.get("post_landing_drift")
    has_imp = bool(features.get("has_impact", False))
    bd = features.get("bounce_decay")

    vd_sub = 0.5 if vd is None else max(0.0, min(1.0, float(vd)))
    pld_sub = 0.5 if pld is None else max(0.0, min(1.0, 1.0 - float(pld) * 0.5))
    imp_sub = 1.0 if has_imp else 0.0
    if bd is None:
        bd_sub = 0.5
    elif bd <= 0.7:
        bd_sub = 1.0
    elif bd <= 1.5:
        bd_sub = 1.0 - (bd - 0.7) / 0.8
    else:
        bd_sub = 0.0

    weights = {"velocity_drop": 0.30, "post_landing_drift": 0.20,
               "has_impact": 0.30, "bounce_decay": 0.20}
    total = (weights["velocity_drop"] * vd_sub
             + weights["post_landing_drift"] * pld_sub
             + weights["has_impact"] * imp_sub
             + weights["bounce_decay"] * bd_sub)
    return {
        "event_score": round(float(total), 4),
        "subs": {
            "velocity_drop_sub": round(vd_sub, 3),
            "post_landing_drift_sub": round(pld_sub, 3),
            "has_impact_sub": round(imp_sub, 3),
            "bounce_decay_sub": round(bd_sub, 3),
        },
        "raw": features,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-segment polyfit y = 1/2 g t^2 + v0 t + y0 -> R^2 + g_fit
# ─────────────────────────────────────────────────────────────────────────────
def polyfit_scoring(centroids: list[dict], seg_info: dict, fps: float) -> dict:
    """Fit y = a t^2 + b t + c on each fall/rise segment.

    Per-segment physics compliance = sign_ok * magnitude_ok * uniformity_ok
    (R^2 is recorded per segment as a diagnostic but does not enter the score):
      - sign_ok: fall requires a > 0 (downward acceleration); rise is legal only after the first fall
                 and still requires a > 0 (gravity decelerates upward motion). Wrong sign -> anti-gravity violation.
      - magnitude_ok: ratio of fitted |a| to the kinematic 2|Dy|/Dt^2 expectation within [0.3, 3] -> 1.0;
                 outside that range it decays linearly. Constant velocity (|a| ~= 0) gets 0.
    """
    if "error" in seg_info:
        return {"error": seg_info["error"]}

    valid = [(c["frame_idx"], c["y_norm"]) for c in centroids if c["valid"]]
    valid.sort(key=lambda x: x[0])
    frames = np.array([v[0] for v in valid])
    y = np.array([v[1] for v in valid])
    t = frames / fps

    seg_details = []
    r2_vals = []
    g_fit_vals = []
    seg_scores = []       # sign_ok * magnitude_ok * uniformity_ok (R^2 kept as diagnostic only)
    has_prior_fall = False

    for s in seg_info["segments"]:
        a, b = s["pt_start"], s["pt_end"]
        n_pts = b - a + 1
        seg = {**s}  # copy

        if s["type"] in ("rest", "lift") or n_pts < 4:
            seg["R2"] = None
            seg["g_fit_norm"] = None
            if s["type"] == "lift":
                seg["note"] = "lift phase (object lifted before release; not scored)"
            elif s["type"] == "rest":
                seg["note"] = "rest segment (not fitted)"
            else:
                seg["note"] = "too few points (<4) to fit"
            seg_details.append(seg)
            continue

        tt = t[a:b + 1]
        yy = y[a:b + 1]

        # Trim impact-spike frames at the tail of a fall segment.
        # The transition frame between fall and rest often contains the touchdown
        # event, where dy/dt jumps to many times the segment's median velocity.
        # Including it in polyfit makes the second-half fit blow up (large spurious
        # |a|), which destroys uniformity_ok on otherwise-perfect free fall.
        impact_trimmed = 0
        if s["type"] == "fall" and n_pts >= 6:
            while n_pts - impact_trimmed >= 6:
                yy_sub = yy[:n_pts - impact_trimmed]
                dy_seq = np.abs(np.diff(yy_sub))
                if len(dy_seq) < 4:
                    break
                med_inner = float(np.median(dy_seq[:-1]))
                if med_inner > 1e-6 and dy_seq[-1] > 3.0 * med_inner:
                    impact_trimmed += 1
                else:
                    break
            if impact_trimmed > 0:
                tt = tt[:n_pts - impact_trimmed]
                yy = yy[:n_pts - impact_trimmed]
                n_pts = n_pts - impact_trimmed
        seg["impact_trimmed"] = impact_trimmed

        # If y-span is tiny the segment is effectively static; polyfit would produce a spurious g, so demote to rest.
        y_span = float(yy.max() - yy.min())
        if y_span < 0.03:
            seg["R2"] = None
            seg["g_fit_norm"] = None
            seg["type_raw"] = s["type"]
            seg["type"] = "rest"
            seg["note"] = f"y-span too small (Dy={y_span:.3f} < 0.03), treated as rest"
            seg_details.append(seg)
            continue

        # Re-origin t at the segment start for numerical stability
        tt0 = tt - tt[0]
        coeffs = np.polyfit(tt0, yy, deg=2)
        a2, a1, a0 = coeffs
        y_pred = np.polyval(coeffs, tt0)
        ss_res = float(np.sum((yy - y_pred) ** 2))
        ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        r2 = max(0.0, min(1.0, r2))

        # Uniformity: split the segment in half, fit each half, and compare the two halves' acceleration.
        # True uniformly-accelerated motion => nearly equal a values. Drifting a (sine/ramp/step) => large half-to-half cv.
        a_first = None
        a_second = None
        half_cv = 0.0
        if n_pts >= 8:
            mid = n_pts // 2
            tt_a = tt0[:mid]; yy_a = yy[:mid]
            tt_b = tt0[mid:]; yy_b = yy[mid:]
            if len(tt_a) >= 4 and len(tt_b) >= 4:
                c_a = np.polyfit(tt_a, yy_a, deg=2)
                c_b = np.polyfit(tt_b, yy_b, deg=2)
                a_first = float(2.0 * c_a[0])
                a_second = float(2.0 * c_b[0])
                a_ref = max(abs(a_first), abs(a_second), 1e-6)
                half_cv = float(abs(a_first - a_second) / a_ref)

        g_fit = 2.0 * abs(a2)  # acceleration magnitude in normalized units
        t_span = float(tt[-1] - tt[0])
        expected_a = (2.0 * y_span / (t_span ** 2)) if t_span > 1e-6 else 0.0

        # Magnitude check: |a| should match the 2|Dy|/Dt^2 scale.
        # The formula assumes "starting from rest" -- true for fall. Rise has a high
        # initial speed and the same g produces smaller Dy; rise skips magnitude.
        if s["type"] == "rise" and has_prior_fall:
            magnitude_ok = 1.0
            mag_note = ""
            ratio = (g_fit / expected_a) if expected_a > 1e-6 else 0.0
        elif expected_a < 1e-6:
            magnitude_ok = 0.0
            mag_note = "near-zero displacement/time"
            ratio = 0.0
        else:
            ratio = g_fit / expected_a
            if 0.3 <= ratio <= 3.0:
                magnitude_ok = 1.0
                mag_note = ""
            elif ratio < 0.3:
                magnitude_ok = max(0.0, ratio / 0.3)
                mag_note = f"|a| too small (ratio={ratio:.2f}); nearly constant velocity"
            else:
                magnitude_ok = max(0.0, 1.0 - (ratio - 3.0) / 3.0)
                mag_note = f"|a| too large (ratio={ratio:.2f}); not gravity-scale"

        # Sign check: polyfit a2 must point in the physically correct direction.
        # Confidence weight = min(1, ratio/0.3); when |a| is small (noise-dominated)
        # the sign is unreliable, so a wrong-sign penalty softens
        # from hard 0 toward (1 - confidence).
        a_down = (a2 > 0)
        sign_confident = min(1.0, ratio / 0.3)
        sign_note = ""
        if s["type"] == "fall":
            sign_base = 1.0 if a_down else 0.0
            if not a_down and sign_confident > 0.5:
                sign_note = "fall segment has upward acceleration (anti-gravity violation)"
        else:  # "rise"
            if not has_prior_fall:
                sign_base = 0.0
                sign_confident = 1.0  # ordering violation, full confidence
                sign_note = "rise appears before any fall (ordering violation)"
            else:
                sign_base = 1.0 if a_down else 0.0
                if not a_down and sign_confident > 0.5:
                    sign_note = "rise segment has upward acceleration (anti-gravity violation)"
        sign_ok = 1.0 - sign_confident * (1.0 - sign_base)

        # Uniformity check: half-to-half cv of fitted a.
        # half_cv = |a_first - a_second| / max(|a_first|, |a_second|)
        # <= 0.15 -> 1.0 (truly uniform); >= 0.80 -> 0.0 (strong drift); linearly interpolated in between.
        if a_first is None or a_second is None:
            uniformity_ok = 1.0  # too few points to judge
            unif_note = ""
        elif half_cv <= 0.15:
            uniformity_ok = 1.0
            unif_note = ""
        elif half_cv >= 0.80:
            uniformity_ok = 0.0
            unif_note = f"half-to-half a difference {half_cv:.2f}; a(t) is not constant"
        else:
            uniformity_ok = 1.0 - (half_cv - 0.15) / (0.80 - 0.15)
            unif_note = f"half-to-half a difference {half_cv:.2f}; acceleration drift"

        seg_score = sign_ok * magnitude_ok * uniformity_ok

        seg["R2"] = round(r2, 4)
        seg["a_half_first"] = round(a_first, 4) if a_first is not None else None
        seg["a_half_second"] = round(a_second, 4) if a_second is not None else None
        seg["half_cv"] = round(half_cv, 3)
        seg["g_fit_norm"] = round(g_fit, 4)
        seg["a_sign"] = "down" if a_down else "up"
        seg["y_span"] = round(y_span, 4)
        seg["expected_a"] = round(expected_a, 4)
        seg["a_ratio"] = round(ratio, 3)
        seg["sign_ok"] = round(sign_ok, 3)
        seg["magnitude_ok"] = round(magnitude_ok, 3)
        seg["uniformity_ok"] = round(uniformity_ok, 3)
        seg["seg_score"] = round(seg_score, 4)
        seg["note"] = " | ".join([n for n in (sign_note, mag_note, unif_note) if n])
        seg_details.append(seg)

        if s["type"] == "fall":
            has_prior_fall = True

        r2_vals.append(r2)
        g_fit_vals.append(g_fit)
        seg_scores.append(seg_score)

    # Cross-segment g consistency (only meaningful when there are >= 2 segments)
    if len(g_fit_vals) >= 2:
        g_mean = float(np.mean(g_fit_vals))
        g_cv = float(np.std(g_fit_vals) / g_mean) if g_mean > 1e-6 else 1.0
        cross_consistency = max(0.0, 1.0 - g_cv)
    else:
        g_mean = float(g_fit_vals[0]) if g_fit_vals else None
        cross_consistency = None

    # Per-trajectory aggregation:
    #   - any fitted fall/rise segment with sign_ok == 0 is a hard physics
    #     violation (anti-gravity) and forces the trajectory score to 0.
    #   - otherwise take the length-weighted mean of seg_scores so that a
    #     short, noise-sensitive rebound segment cannot single-handedly
    #     drag the whole trajectory to 0 the way `min` does.
    fitted_segs = [sd for sd in seg_details if sd.get("seg_score") is not None]
    hard_violation = any(
        sd.get("type") in ("fall", "rise")
        and sd.get("sign_ok") is not None
        and sd["sign_ok"] == 0.0
        for sd in fitted_segs
    )
    if hard_violation:
        mean_seg = 0.0
    elif fitted_segs:
        weights = [max(1, int(sd.get("n_points", 1))) for sd in fitted_segs]
        scores = [sd["seg_score"] for sd in fitted_segs]
        mean_seg = float(np.average(scores, weights=weights))
    else:
        mean_seg = 0.0

    return {
        "method": "sign_x_magnitude_x_uniformity_lenweighted",
        "segments": seg_details,
        "mean_R2": round(float(np.mean(r2_vals)), 4) if r2_vals else 0.0,
        "mean_seg_score": round(mean_seg, 4),
        "g_fit_norm_mean": round(g_mean, 4) if g_mean is not None else None,
        "g_cross_consistency": round(cross_consistency, 4) if cross_consistency is not None else None,
        "n_fit_segments": len(r2_vals),
        "n_valid": seg_info["n_valid"],
        "has_bounce": seg_info["has_bounce"],
        "n_bounces": seg_info["n_bounces"],
        "has_first_fall": seg_info.get("has_first_fall", False),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Final score
# ─────────────────────────────────────────────────────────────────────────────
def compute_final_score(analysis: dict, n_frames_total: int,
                        window_len: int | None = None,
                        event_score: dict | None = None) -> dict:
    """Combine curve-fit score and event score into the final score.

      curve_score = mean(seg_score) * coverage_factor
      event_score = see compute_event_score
      final = gated mix of curve and event (curve dominates when curve is low,
              flagging a clear violation; event dominates when curve is high).

    If the curve side is unevaluable (n_fit_segs = 0 / coverage < 0.3) but the
    event signal is usable, fall back to 100% event_score instead of None.
    """
    if "error" in analysis:
        return {"score": None, "score_pct": None, "issues": [analysis["error"]],
                "unevaluable": True}

    r2 = analysis.get("mean_R2", 0.0)
    mean_seg = analysis.get("mean_seg_score", 0.0)
    n_valid = analysis.get("n_valid", 0)
    n_fit_segs = analysis.get("n_fit_segments", 0)
    denom = window_len if window_len and window_len >= 6 else max(8, n_frames_total * 0.5)
    coverage = min(1.0, n_valid / denom)

    cc = analysis.get("g_cross_consistency")
    ev = (event_score or {}).get("event_score")
    ev_subs = (event_score or {}).get("subs", {})

    curve_evaluable = not (coverage < 0.3 or n_valid < 6 or n_fit_segs == 0)

    if curve_evaluable:
        coverage_factor = min(1.0, coverage / 0.3)
        cc_factor = 1.0
        curve_score = float(max(0.0, min(1.0, mean_seg * coverage_factor * cc_factor)))
    else:
        coverage_factor = round(min(1.0, coverage / 0.6), 3)
        cc_factor = None
        curve_score = None

    # The event signal is only "usable" when an impact was detected AND a fall
    # phase actually exists. Without a fall phase the physics premise (free fall +
    # landing) does not apply, so an "anti-gravity" video where the object only
    # rises must NOT be rescued by event_score's neutral 0.5 defaults.
    has_first_fall = bool(analysis.get("has_first_fall", False))
    event_usable = (ev is not None
                    and ev_subs.get("has_impact_sub", 0.0) > 0.5
                    and has_first_fall)

    if curve_score is not None and event_usable:
        # Gated fusion: when curve_score < 0.3 (a clear violation signal) let
        # the curve term dominate; otherwise let the event term dominate.
        if curve_score < 0.3:
            final = 0.70 * curve_score + 0.30 * ev
        else:
            final = 0.30 * curve_score + 0.70 * ev
    elif curve_score is not None:
        final = curve_score   # no usable event signal -> fall back to curve
    elif event_usable:
        final = ev            # curve unevaluable but event usable AND has_first_fall
    else:
        # neither is usable -> truly unevaluable
        return {
            "score": None, "score_pct": None,
            "mean_R2": round(r2, 4),
            "mean_seg_score": round(mean_seg, 4),
            "coverage": round(coverage, 3),
            "g_cross_consistency": round(cc, 3) if cc is not None else None,
            "event_score": ev,
            "unevaluable": True,
            "issues": [
                f"unevaluable: n_valid={n_valid}, coverage={coverage:.2f}, "
                f"n_fit_segs={n_fit_segs}, event_usable={event_usable}"
            ],
        }
    score = float(max(0.0, min(1.0, final)))

    issues = []
    if not analysis.get("has_first_fall", False):
        issues.append("no fall phase detected (object may be held throughout, or video has no free fall)")
    if r2 < 0.85:
        issues.append(f"info: polyfit R^2={r2:.2f} (diagnostic only, not used in scoring)")
    # Per-segment diagnostics
    for s in analysis.get("segments", []):
        if s.get("note") and s.get("seg_score") is not None and s["seg_score"] < 0.5:
            issues.append(f"segment[{s['type']}]: {s['note']}")
    if cc is not None and cc < 0.6:
        issues.append(f"cross-segment a consistency={cc:.2f}; segments disagree on acceleration")
    if coverage < 0.6:
        issues.append(f"valid tracking coverage={coverage:.2f}; SAM2 lost the object too often")
    if analysis.get("has_bounce"):
        issues.append(f"info: detected {analysis['n_bounces']} bounce(s); scored normally if sign/magnitude pass")

    if event_usable and ev is not None:
        if ev < 0.4:
            issues.append(f"event score {ev:.2f} is low (floating / through-ground / unphysical rebound)")

    return {
        "score": round(score, 4),
        "score_pct": round(score * 100, 1),
        "curve_score": round(curve_score, 4) if curve_score is not None else None,
        "event_score": round(ev, 4) if ev is not None else None,
        "event_subs": ev_subs if event_usable else None,
        "mean_R2": round(r2, 4),
        "mean_seg_score": round(mean_seg, 4),
        "coverage": round(coverage, 3),
        "coverage_factor": coverage_factor if isinstance(coverage_factor, (int, float)) else round(coverage_factor, 3),
        "cc_factor": round(cc_factor, 3) if isinstance(cc_factor, (int, float)) else None,
        "g_cross_consistency": round(cc, 3) if cc is not None else None,
        "unevaluable": False,
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main entry
# ─────────────────────────────────────────────────────────────────────────────
def analyze_video(
    video_path: str,
    output_dir: str | None = None,
    first_point: tuple[float, float] | None = None,
    first_bbox: tuple[float, float, float, float] | None = None,
    release_frame: int | None = None,
    use_vlm: bool = True,
    auto_release_detect: bool = True,    # try to start tracking at the release frame; fall back to frame 0 if bbox sanity-check fails
    use_label_window: bool = True,       # if True, score only the released-label window
    n_release_samples: int = 16,         # number of frames sampled for release detection
    release_bias_early: bool = True,     # if True, shift window start earlier to catch the release moment
    drift_area_ratio: float = 2.5,       # mask growth threshold (times the initial area) before flagging drift
    save_mask_video: bool = False,
) -> dict:
    """Main entry point.

    Default pipeline (robust):
      1. VLM sees frame 0 -> object bbox + gripper negative points.
      2. SAM2 propagates forward from frame 0 (bbox + negative points).
      3. segment_trajectory detects lift / rest / fall / rise (lift is not scored).
      4. polyfit-based scoring.

    Optional auto_release_detect=True:
      Sample several frames with the VLM to locate the release moment, then
      track only frames after release. Sensitive to VLM stability.

    Args:
        first_point / first_bbox: manually specify the object prompt (bypass VLM).
        release_frame: manually specify the release frame index.
        use_vlm: whether to call the VLM.
        auto_release_detect: whether to let the VLM locate the release frame.
        n_release_samples: number of frames the VLM samples for release detection.
        save_mask_video: write a video with the SAM2 mask overlay (for visualization).
    """
    if output_dir is None:
        output_dir = os.path.dirname(video_path)

    # Step 1: Locate the release frame + object bbox.
    # Strategy:
    #   A. Manual override -> use directly
    #   B. VLM looks at frame 0 first:
    #      - if the object is already detached, use the frame-0 bbox (fastest path)
    #      - else, run vlm_find_release_frame to sample multiple frames and locate release
    print(f"[1/4] Locating object and release frame...")
    release_info = {"release_frame_idx": 0, "bbox": None, "object": "",
                    "source": "", "reason": ""}

    if first_bbox is not None or first_point is not None:
        release_info["source"] = "manual"
        release_info["release_frame_idx"] = int(release_frame or 0)
        if first_bbox is not None:
            release_info["bbox"] = list(first_bbox)
        if first_point is not None:
            release_info["manual_point"] = list(first_point)
        release_info["object"] = "manual"
        print(f"  manual: release_frame={release_info['release_frame_idx']}, "
              f"bbox={release_info.get('bbox')}, point={release_info.get('manual_point')}")
    elif use_vlm:
        cap = cv2.VideoCapture(video_path)
        _, first_frame = cap.read(); cap.release()
        loc_f0 = vlm_locate_first_frame(first_frame)

        if auto_release_detect:
            # Sample multiple frames with the VLM to locate release; fall back to f0 on failure.
            print(f"  [auto_release_detect=True] sampling {n_release_samples} frames to locate release")
            vlm_result = vlm_find_release_frame(video_path, n_samples=n_release_samples)
            total_frames_est = (max(vlm_result["sample_indices"]) + 1
                                if vlm_result.get("sample_indices") else None)
            too_late = (
                vlm_result["release_frame_idx"] is not None and total_frames_est
                and vlm_result["release_frame_idx"] > total_frames_est * 0.55
            )
            # bbox sanity check: the release-frame bbox should be horizontally close to
            # the frame-0 bbox and vertically lower (gravity, no upward drift).
            inconsistent = False
            if vlm_result.get("bbox") and loc_f0.get("bbox"):
                f0b = loc_f0["bbox"]; rb = vlm_result["bbox"]
                f0_cx = (f0b[0] + f0b[2]) / 2; f0_cy = (f0b[1] + f0b[3]) / 2
                r_cx = (rb[0] + rb[2]) / 2; r_cy = (rb[1] + rb[3]) / 2
                dx = abs(r_cx - f0_cx); dy = r_cy - f0_cy
                # If dx > 0.15 or dy is significantly negative, mark inconsistent.
                if dx > 0.15 or dy < -0.05:
                    inconsistent = True
                    print(f"  WARNING: release bbox center ({r_cx:.2f},{r_cy:.2f}) "
                          f"inconsistent with f0 ({f0_cx:.2f},{f0_cy:.2f}) "
                          f"(dx={dx:.2f}, dy={dy:+.2f}); VLM may have switched objects.")

            if vlm_result["release_frame_idx"] is not None and not too_late and not inconsistent:
                release_info.update(vlm_result)
                release_info["source"] = "vlm_release_detect"
                print(f"  VLM detected: release_frame={vlm_result['release_frame_idx']} "
                      f"(F{vlm_result['release_sample_idx']}/{len(vlm_result['sample_indices'])}), "
                      f"obj={vlm_result['object']}, bbox={vlm_result['bbox']}")
            else:
                reason = ("release_frame is in the last >55% of the video" if too_late else
                          "release bbox inconsistent with f0" if inconsistent else
                          vlm_result.get("reason", ""))
                print(f"  WARNING: release-frame detection rejected ({reason}); falling back to frame-0 bbox + label-window filter")
                release_info.update({
                    "release_frame_idx": 0,
                    "bbox": loc_f0.get("bbox"),
                    "gripper": loc_f0.get("gripper"),
                    "negative_points": loc_f0.get("negative_points"),
                    "object": loc_f0.get("object", ""),
                    # keep labels so that label_window can still take effect
                    "labels": vlm_result.get("labels", []),
                    "sample_indices": vlm_result.get("sample_indices", []),
                    "source": "vlm_release_fallback_f0",
                    "reason": reason,
                })
        else:
            # Default: VLM provides a frame-0 bbox; segment_trajectory detects the lift phase automatically.
            if loc_f0.get("bbox") is not None:
                release_info.update({
                    "release_frame_idx": 0,
                    "bbox": loc_f0["bbox"],
                    "gripper": loc_f0.get("gripper"),
                    "object": loc_f0.get("object", ""),
                    "source": "vlm_f0",
                    "reason": "frame 0 bbox (lift phase excluded automatically by segmentation)",
                })
                print(f"  VLM frame 0: obj={loc_f0['object']}, "
                      f"bbox={loc_f0['bbox']}, gripper={loc_f0.get('gripper')}")
            elif loc_f0.get("x_norm") is not None:
                release_info.update({
                    "release_frame_idx": 0,
                    "manual_point": [loc_f0["x_norm"], loc_f0["y_norm"]],
                    "object": loc_f0.get("object", ""),
                    "source": "vlm_f0_point",
                })
                print(f"  VLM frame 0 (point): ({loc_f0['x_norm']:.2f}, {loc_f0['y_norm']:.2f})")
            else:
                print(f"  WARNING: VLM did not locate any object: {loc_f0.get('error', '')}; falling back to image center")
                release_info.update({
                    "release_frame_idx": 0,
                    "manual_point": [0.5, 0.5],
                    "source": "center_fallback",
                })
    else:
        release_info["source"] = "fallback"
        release_info["manual_point"] = [0.5, 0.5]
        print(f"  use_vlm=False; starting from image center at frame 0")

    # Step 2: SAM2 propagates from the release frame.
    print(f"[2/4] SAM2.1-Hiera-Large propagation from frame {release_info['release_frame_idx']} (MPS)...")
    t0 = time.time()
    pos_pt = release_info.get("manual_point")
    if pos_pt is None and release_info.get("bbox"):
        bb = release_info["bbox"]
        pos_pt = [(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2]
    track = sam2_track_centroids(
        video_path,
        positive_point=tuple(pos_pt) if pos_pt else None,
        bbox=tuple(release_info["bbox"]) if release_info.get("bbox") else None,
        negative_points=release_info.get("negative_points"),
        negative_point=tuple(release_info["gripper"])
            if release_info.get("gripper") and not release_info.get("negative_points")
            else None,
        prompt_frame_idx=release_info["release_frame_idx"],
        drift_area_ratio=drift_area_ratio,
        keep_masks=save_mask_video,
    )
    print(f"  elapsed {time.time()-t0:.1f}s, frames={track['n_frames']}, "
          f"valid centroids={sum(1 for c in track['centroids'] if c['valid'])}")

    # Step 2.5: Determine the scoring window (prefer labels already in release_info).
    label_window: tuple[int, int] | None = None
    labels_info = None
    labels = []
    sample_idx = []
    if use_vlm and release_info.get("labels"):
        labels = release_info["labels"]
        sample_idx = release_info.get("sample_indices") or []
    elif use_label_window and use_vlm \
            and release_info.get("source") in ("vlm_f0", "vlm_f0_point"):
        print(f"[2.5/4] VLM multi-frame labeling to determine the released window...")
        labels_info = vlm_find_release_frame(video_path, n_samples=n_release_samples)
        labels = labels_info.get("labels") or []
        sample_idx = labels_info.get("sample_indices") or []

    if use_label_window and labels and len(labels) == len(sample_idx):
        released_pos = [i for i, lbl in enumerate(labels)
                        if str(lbl).lower() == "released"]
        landed_pos = [i for i, lbl in enumerate(labels)
                      if str(lbl).lower() == "landed"]
        if released_pos:
            first_rel = released_pos[0]
            last_rel = released_pos[-1]
            if landed_pos:
                first_land = landed_pos[0]
                last_rel = max(last_rel, first_land - 1) if first_land > first_rel else last_rel

            # Bias the start toward "released" (2/3 weight) to avoid including the held tail.
            if release_bias_early and first_rel > 0 \
                    and str(labels[first_rel - 1]).lower() == "held":
                start_frame = int(round(sample_idx[first_rel - 1] * (1/3)
                                        + sample_idx[first_rel] * (2/3)))
            else:
                start_frame = sample_idx[first_rel]
            # Push the end as late as possible: up to the frame just before the next non-released sample.
            if last_rel + 1 < len(sample_idx):
                end_frame = sample_idx[last_rel + 1] - 1
            else:
                end_frame = sample_idx[-1]
            label_window = (start_frame, end_frame)
            print(f"  Labels ({len(labels)} samples): {labels}")
            print(f"  scoring window: frames [{start_frame}, {end_frame}]")
        else:
            print(f"  WARNING: no released labels: {labels}; scoring window covers the whole video")

    print(f"[3/4] Trajectory segmentation + per-segment polyfit(deg=2) fit...")
    centroids_for_scoring = track["centroids"]
    if label_window is not None:
        # Mark frames outside the window invalid (excluded from segmentation/scoring).
        w_start, w_end = label_window
        centroids_for_scoring = [
            c if (w_start <= c["frame_idx"] <= w_end) else
            {**c, "valid": False, "x_norm": None, "y_norm": None,
             "outside_label_window": True}
            for c in track["centroids"]
        ]
    seg_info = segment_trajectory(centroids_for_scoring, track["fps"])
    analysis = polyfit_scoring(centroids_for_scoring, seg_info, track["fps"])

    # Event features (independent of segmentation; usable when curve is unevaluable).
    event_features = extract_event_features(centroids_for_scoring, track["fps"])
    event_score = compute_event_score(event_features)

    print(f"[4/4] Final scoring...")
    window_len = (label_window[1] - label_window[0] + 1) if label_window else None
    score_result = compute_final_score(analysis,
                                       n_frames_total=track["n_frames"],
                                       window_len=window_len,
                                       event_score=event_score)

    result = {
        "video_path": video_path,
        "fps": track["fps"],
        "total_frames": track["n_frames"],
        "frame_hw": [track["h"], track["w"]],
        "release_info": release_info,
        "label_window": list(label_window) if label_window else None,
        "labels_info": labels_info,
        "first_frame_loc": release_info,  # field alias
        # Per-frame SAM2 tracking results (for visualization / diagnostics)
        "positions": [
            {
                "frame_idx": c["frame_idx"],
                "x_norm": c["x_norm"],
                "y_norm": c["y_norm"],
                "mask_area": c["mask_area"],
                "confidence": "high" if c["valid"] else "none",
                "pre_release": c.get("pre_release", False),
                "drift": c.get("drift", False),
                "in_freefall": None,
            }
            for c in track["centroids"]
        ],
        "analysis": analysis,
        "event_features": event_features,
        "event_score": event_score,
        "score": score_result,
    }

    basename = os.path.splitext(os.path.basename(video_path))[0]
    json_path = os.path.join(output_dir, f"{basename}_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Result saved: {json_path}")

    # Optional: write a video with mask overlay for visual inspection.
    if save_mask_video and "masks" in track:
        overlay_path = os.path.join(output_dir, f"{basename}_mask_overlay.mp4")
        annotate_video(video_path, result, overlay_path, masks=track["masks"])
        print(f"Mask-overlay video saved: {overlay_path}")
        result["mask_overlay_video"] = overlay_path

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 7. Annotated video (visualization)
# ─────────────────────────────────────────────────────────────────────────────
def annotate_video(
    video_path: str,
    result: dict,
    output_path: str,
    masks: list | None = None,
) -> str:
    """Render an annotated video: per-frame SAM2 mask (optional) + centroid dot + phase label + score overlay.

    Args:
        masks: optional list of HxW bool ndarrays (length = n_frames); if provided, renders a translucent green mask overlay.
    """
    if "error" in result:
        return video_path

    positions = {p["frame_idx"]: p for p in result["positions"]}
    segments = result.get("analysis", {}).get("segments", [])

    # Build per-frame phase info for the overlay
    phase_pri = {"rest": 0, "lift": 1, "fall": 2, "rise": 2}
    frame_phase: dict[int, dict] = {}
    for s in segments:
        for fi in range(s["frame_start"], s["frame_end"] + 1):
            cur = frame_phase.get(fi)
            r2 = s.get("R2")
            seg_sc = s.get("seg_score")
            is_ok = (seg_sc is None) or (seg_sc >= 0.5)
            pri = phase_pri.get(s["type"], 1) + (0 if is_ok else 3)
            if cur is None or pri > cur["pri"]:
                frame_phase[fi] = {
                    "type": s["type"], "R2": r2,
                    "ok": is_ok, "pri": pri,
                }

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not out.isOpened():
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    score_pct = result.get("score", {}).get("score_pct", 0)
    n_bounces = result.get("analysis", {}).get("n_bounces", 0)

    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Keep the original frame brightness; only the label text differs in/out of window.
        window = result.get("label_window")
        in_window = (window is None) or (window[0] <= fi <= window[1])

        if in_window:
            # Translucent SAM2 mask overlay (only inside the label window)
            if masks is not None and fi < len(masks):
                m = masks[fi]
                if m is not None and m.any():
                    overlay = frame.copy()
                    overlay[m] = (0, 255, 100)  # bright green
                    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
                    contours, _ = cv2.findContours(
                        m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(frame, contours, -1, (0, 255, 255), 2)

            p = positions.get(fi)
            if p is not None and p.get("x_norm") is not None and p.get("y_norm") is not None:
                x_px = int(p["x_norm"] * (w - 1))
                y_px = int(p["y_norm"] * (h - 1))
                cv2.circle(frame, (x_px, y_px), 6, (0, 0, 255), -1)
                cv2.circle(frame, (x_px, y_px), 6, (255, 255, 255), 2)
        else:
            tag = "PRE-RELEASE (held)" if window and fi < window[0] else "POST-LANDED"
            cv2.rectangle(frame, (0, 0), (360, 50), (60, 60, 60), -1)
            cv2.putText(frame, tag, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)

        info = frame_phase.get(fi) if in_window else None
        if info is not None:
            t = info["type"]
            r2 = info.get("R2")
            if not info["ok"]:
                color = (30, 30, 200); label = f"ANOM  R2={r2:.2f}" if r2 is not None else "ANOM"
            elif t == "fall":
                color = (0, 160, 0); label = f"FALL  R2={r2:.2f}" if r2 is not None else "FALL"
            elif t == "rise":
                color = (0, 165, 255); label = f"RISE  R2={r2:.2f}" if r2 is not None else "RISE"
            elif t == "lift":
                color = (180, 100, 40); label = "LIFT (not scored)"
            else:
                color = (120, 120, 120); label = "REST"
            cv2.rectangle(frame, (0, 0), (360, 50), color, -1)
            cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.rectangle(frame, (w - 240, 0), (w, 70), (30, 30, 30), -1)
        score_text = "Score: N/A" if score_pct is None else f"Score: {score_pct:.1f}%"
        cv2.putText(frame, score_text, (w - 230, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 215, 0), 2)
        if n_bounces > 0:
            cv2.putText(frame, f"Bounces: {n_bounces}", (w - 230, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 1)

        out.write(frame)
        fi += 1

    cap.release()
    out.release()
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# 8. VLM video quality check (for final_score weighting)
# ─────────────────────────────────────────────────────────────────────────────
def video_quality_check(video_path: str, n_samples: int = 6) -> dict:
    """Sample frames from video and check (a) the video is valid and
    (b) any object exhibits scorable motion (free-fall, slide, push, ...).
    Returns a Video Quality Score (VQS) used as the 10% video-quality term
    in the final score.

    Returns:
      {
        "video_ok": bool,
        "has_motion": bool,
        "reason": str,
        "vqs": float,   # 0=bad video, 5=valid no scorable motion, 10=has motion
      }
    """
    import re
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return {"video_ok": False, "has_motion": False, "reason": "Cannot open video", "vqs": 0.0}

    n_samples = min(n_samples, total)
    step = total / n_samples
    indices = [int(round(i * step)) for i in range(n_samples)]
    indices = sorted(set(min(total - 1, x) for x in indices))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret:
            frames.append(f)
    cap.release()

    if not frames:
        return {"video_ok": False, "has_motion": False, "reason": "No readable frames", "vqs": 0.0}

    # Tile frames horizontally
    tile_w = 320
    tiles = []
    for f in frames:
        h0, w0 = f.shape[:2]
        sc = tile_w / w0
        tiles.append(cv2.resize(f, (tile_w, int(h0 * sc))))
    max_h = max(t.shape[0] for t in tiles)
    padded = []
    for t in tiles:
        dh = max_h - t.shape[0]
        if dh > 0:
            t = np.vstack([t, np.zeros((dh, t.shape[1], 3), dtype=np.uint8)])
        padded.append(t)
    grid = np.hstack(padded)
    b64 = encode_frame_base64(grid)

    prompt = """You are evaluating a generated video that should show a physics-scorable scene
(an object undergoing free-fall, a slide / push along a surface, or any
other clear, sustained translational motion under physics).
The image shows several equally-spaced frames from the video tiled left to right.
Answer ONLY with valid JSON (no markdown) containing exactly these fields:
{
  "video_ok": true/false,
  "has_motion": true/false,
  "reason": "one short sentence"
}

Definitions:
- video_ok: true if the video shows a recognizable robot-arm scene (not all-black, not corrupted, not a static image).
- has_motion: true if ANY object visible in the scene undergoes a clear translational motion at some point in the video — free-fall, sliding, being pushed, rolling, or otherwise traversing space. Be LENIENT: a brief but unambiguous displacement is enough. Set to false ONLY when the entire clip is essentially static (object held in place / no scene motion)."""

    try:
        resp = _vlm_call_with_retry(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            max_tokens=300,
            timeout=60,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("`").strip()
        data = json.loads(raw)
        video_ok = bool(data.get("video_ok", False))
        has_motion = bool(data.get("has_motion", False))
        reason = str(data.get("reason", ""))
        if not video_ok:
            vqs = 0.0
        elif not has_motion:
            vqs = 5.0
        else:
            vqs = 10.0
        return {"video_ok": video_ok, "has_motion": has_motion, "reason": reason, "vqs": vqs}
    except Exception as e:
        print(f"    video_quality_check failed: {e}")
        return {"video_ok": None, "has_motion": None, "reason": str(e), "vqs": None}
