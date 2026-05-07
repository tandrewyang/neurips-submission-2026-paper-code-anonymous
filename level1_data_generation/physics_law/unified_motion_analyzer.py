"""
Unified physics-analysis entry point. Runs SAM2 once and auto-detects the dominant motion axis:

  - |Dy| >> |Dx| -> vertical-dominant: route to the free-fall pipeline (gravity).
  - |Dx| >> |Dy| -> horizontal-dominant: route to the push pipeline (friction).
  - |Dx| ~= |Dy|: mixed -> run both pipelines and report each score.

Reuses segmentation + scoring from physics_analyzer / horizontal_push_analyzer.
"""
import os
import json
import time
import numpy as np
import cv2

from physics_analyzer import (
    sam2_track_centroids,
    vlm_locate_first_frame,
    vlm_find_release_frame,
    segment_trajectory,
    polyfit_scoring,
    compute_final_score,
)
from horizontal_push_analyzer import (
    segment_horizontal_trajectory,
    polyfit_horizontal_scoring,
    compute_horizontal_score,
)


def _classify_axis(centroids: list[dict]) -> tuple[str, float, float]:
    """Classify dominant motion direction from x/y spans of valid centroids."""
    xs, ys = [], []
    for c in centroids:
        if c.get("valid") and c.get("x_norm") is not None and c.get("y_norm") is not None:
            xs.append(c["x_norm"]); ys.append(c["y_norm"])
    if len(xs) < 4:
        return "unknown", 0.0, 0.0
    dx = float(max(xs) - min(xs))
    dy = float(max(ys) - min(ys))
    if dy > 1.5 * dx and dy > 0.05:
        return "vertical", dx, dy
    if dx > 1.5 * dy and dx > 0.05:
        return "horizontal", dx, dy
    if max(dx, dy) < 0.05:
        return "static", dx, dy
    return "mixed", dx, dy


def analyze_motion(
    video_path: str,
    output_dir: str | None = None,
    first_bbox: tuple[float, float, float, float] | None = None,
    first_point: tuple[float, float] | None = None,
    start_frame: int = 0,
    use_vlm: bool = True,
    auto_release_detect: bool = True,
    n_release_samples: int = 16,
) -> dict:
    """Unified entry point: run SAM2, classify dominant axis, dispatch to the right scorer."""
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Prompt (VLM or manual)
    print(f"[1/5] Locating object...")
    pos_pt = None; bbox = None; obj_name = "manual"; neg_pts = []
    label_window = None; labels = []; sample_idx = []

    if first_bbox is not None:
        bbox = list(first_bbox)
        pos_pt = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        print(f"  manual bbox: {bbox}, start={start_frame}")
    elif first_point is not None:
        pos_pt = list(first_point)
        print(f"  manual point: {pos_pt}, start={start_frame}")
    elif use_vlm:
        if auto_release_detect:
            # Free-fall release-frame detection; if it fails we still have a frame-0 bbox.
            vlm_res = vlm_find_release_frame(video_path, n_samples=n_release_samples)
            labels = vlm_res.get("labels") or []
            sample_idx = vlm_res.get("sample_indices") or []
            if vlm_res.get("release_frame_idx") is not None and vlm_res.get("bbox"):
                start_frame = int(vlm_res["release_frame_idx"])
                bbox = list(vlm_res["bbox"])
                neg_pts = vlm_res.get("negative_points") or []
                obj_name = vlm_res.get("object", "")
                pos_pt = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                print(f"  VLM release: frame={start_frame} obj={obj_name} bbox={bbox}")
        if bbox is None and pos_pt is None:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            _, fr = cap.read(); cap.release()
            loc = vlm_locate_first_frame(fr)
            bbox = loc.get("bbox")
            if bbox is None and loc.get("x_norm") is not None:
                pos_pt = [loc["x_norm"], loc["y_norm"]]
            elif bbox is not None:
                pos_pt = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            neg_pts = loc.get("negative_points") or []
            obj_name = loc.get("object", "")
            print(f"  VLM frame {start_frame}: obj={obj_name} bbox={bbox}")
    else:
        raise ValueError("Provide one of: first_bbox / first_point / use_vlm=True")

    # Step 2: SAM2 tracking
    print(f"[2/5] SAM2 forward propagation from frame {start_frame}...")
    t0 = time.time()
    track = sam2_track_centroids(
        video_path,
        positive_point=tuple(pos_pt) if pos_pt else None,
        bbox=tuple(bbox) if bbox else None,
        negative_points=neg_pts or None,
        prompt_frame_idx=start_frame,
    )
    print(f"  elapsed {time.time()-t0:.1f}s, frames={track['n_frames']}, "
          f"valid={sum(1 for c in track['centroids'] if c['valid'])}")

    # Step 2.5: label_window (only used by the free-fall pipeline; horizontal-push uses the whole sequence)
    if use_vlm and labels and len(labels) == len(sample_idx):
        rel_pos = [i for i, l in enumerate(labels) if str(l).lower() == "released"]
        if rel_pos:
            first_rel, last_rel = rel_pos[0], rel_pos[-1]
            ws = sample_idx[first_rel]
            we = sample_idx[last_rel + 1] - 1 if last_rel + 1 < len(sample_idx) else sample_idx[-1]
            label_window = (ws, we)
            print(f"  label_window: [{ws}, {we}]")

    # Step 3: Dominant-axis classification
    centroids = track["centroids"]
    if label_window is not None:
        ws, we = label_window
        centroids_for_axis = [c for c in centroids if ws <= c["frame_idx"] <= we]
    else:
        centroids_for_axis = centroids
    axis, dx, dy = _classify_axis(centroids_for_axis)
    print(f"[3/5] dominant axis: {axis}  (dx={dx:.3f}  dy={dy:.3f})")

    # Same as free fall: mark frames outside label_window as invalid
    if label_window is not None:
        ws, we = label_window
        scored_centroids = [
            c if (ws <= c["frame_idx"] <= we) else
            {**c, "valid": False, "x_norm": None, "y_norm": None}
            for c in centroids
        ]
    else:
        scored_centroids = centroids

    # Step 4: Dispatch by axis
    print(f"[4/5] Segmentation + fitting...")
    result_v = result_h = None
    if axis in ("vertical", "mixed", "unknown"):
        seg_v = segment_trajectory(scored_centroids, track["fps"])
        ana_v = polyfit_scoring(scored_centroids, seg_v, track["fps"])
        win_len = (label_window[1] - label_window[0] + 1) if label_window else None
        sc_v = compute_final_score(ana_v, n_frames_total=track["n_frames"], window_len=win_len)
        result_v = {"analysis": ana_v, "score": sc_v}
    if axis in ("horizontal", "mixed"):
        seg_h = segment_horizontal_trajectory(scored_centroids, track["fps"])
        ana_h = polyfit_horizontal_scoring(scored_centroids, seg_h, track["fps"])
        sc_h = compute_horizontal_score(ana_h, n_frames_total=track["n_frames"])
        result_h = {"analysis": ana_h, "score": sc_h}

    # Step 5: Final primary score
    primary = None; primary_score = None
    if axis == "vertical":
        primary = "vertical"; primary_score = result_v["score"] if result_v else None
    elif axis == "horizontal":
        primary = "horizontal"; primary_score = result_h["score"] if result_h else None
    elif axis == "mixed":
        # Pick the higher-scoring evaluable side
        sv = result_v["score"].get("score_pct") if result_v and result_v["score"] else None
        sh = result_h["score"].get("score_pct") if result_h and result_h["score"] else None
        if sv is None and sh is None:
            primary = "mixed_unevaluable"
        elif sv is None:
            primary = "horizontal"; primary_score = result_h["score"]
        elif sh is None:
            primary = "vertical"; primary_score = result_v["score"]
        else:
            primary = "vertical" if sv >= sh else "horizontal"
            primary_score = result_v["score"] if primary == "vertical" else result_h["score"]
    else:
        primary = axis  # static / unknown

    print(f"[5/5] primary score: mode={primary}  "
          f"score={primary_score.get('score_pct') if primary_score else 'N/A'}")

    out = {
        "video_path": video_path,
        "fps": track["fps"],
        "total_frames": track["n_frames"],
        "frame_hw": [track["h"], track["w"]],
        "object": obj_name,
        "start_frame": start_frame,
        "label_window": list(label_window) if label_window else None,
        "axis_classification": {
            "axis": axis, "dx_span": round(dx, 4), "dy_span": round(dy, 4),
        },
        "primary_mode": primary,
        "primary_score": primary_score,
        "vertical": result_v,
        "horizontal": result_h,
        "positions": [
            {"frame_idx": c["frame_idx"], "x_norm": c.get("x_norm"),
             "y_norm": c.get("y_norm"), "mask_area": c.get("mask_area"),
             "confidence": "high" if c["valid"] else "none"}
            for c in track["centroids"]
        ],
    }

    basename = os.path.splitext(os.path.basename(video_path))[0]
    json_path = os.path.join(output_dir, f"{basename}_unified.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Result saved: {json_path}")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python unified_motion_analyzer.py <video_path>")
        sys.exit(1)
    r = analyze_motion(sys.argv[1])
    print(f"\nmode={r['primary_mode']}")
    if r.get("primary_score"):
        ps = r["primary_score"]
        print(f"score={ps.get('score_pct')}%  R²={ps.get('mean_R2')}")
