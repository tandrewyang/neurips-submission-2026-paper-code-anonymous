"""
Horizontal Push Physics Analyzer (SAM2 + polyfit on x(t))

Physics: object is pushed -> uniformly decelerates under friction -> stops. Throughout the slide phase a = -mu*g is constant.

Pipeline (mirrors physics_analyzer.py):
  1. VLM (or manual override) provides the pushed-object bbox at the moment motion begins.
  2. SAM2.1 video propagation; per-frame mask centroid yields x(t).
  3. segment_horizontal_trajectory splits the trajectory into push / slide / rest based on |v_x|;
  4. polyfit(deg=2) on each slide segment yields seg_score = sign * magnitude * uniformity (R^2 kept as diagnostic only):
       - sign:      a opposite to v (friction decelerates)
       - magnitude: |2*a2| matches the kinematic 2|Dx|/Dt^2 scale.
       - uniformity: half-split a comparison verifies a constant friction coefficient.
  5. compute_horizontal_score: mean(seg_score) × coverage × cc
"""
import os
import json
import time
import numpy as np
import cv2

from physics_analyzer import (
    sam2_track_centroids,
    vlm_locate_first_frame,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Segmentation (based on |v_x|)
# ─────────────────────────────────────────────────────────────────────────────
def segment_horizontal_trajectory(centroids: list[dict], fps: float) -> dict:
    """Segment x(t) into push / slide / rest.

    Segment types:
      - "rest":  |v_x| persistently near zero
      - "push":  |v_x| monotonically increasing (external force) -- NOT scored
      - "slide": |v_x| monotonically decreasing (friction only) -- scored
    """
    valid = [(c["frame_idx"], c["x_norm"]) for c in centroids
             if c["valid"] and c.get("x_norm") is not None]
    n = len(valid)
    if n < 6:
        return {"error": f"too few valid trajectory points ({n} < 6)"}

    valid.sort(key=lambda x: x[0])
    frames = np.array([v[0] for v in valid])
    x = np.array([v[1] for v in valid])
    t = frames / fps

    # Signed velocity
    dt = np.diff(t)
    dx = np.diff(x)
    v = np.where(dt > 0, dx / dt, 0.0)

    v_ref = float(np.percentile(np.abs(v), 95)) if len(v) else 0.0
    move_thresh = max(0.08, v_ref * 0.15)
    min_v = max(0.05, v_ref * 0.12)

    # 1. Tag each v[i] as rest/move
    lbl = ["move" if abs(vi) > move_thresh else "rest" for vi in v]

    # 2. Compress consecutive same-label intervals into raw segments (point indices a..b)
    raw = []; i = 0
    while i < len(lbl):
        j = i
        while j < len(lbl) and lbl[j] == lbl[i]:
            j += 1
        raw.append((i, j, lbl[i]))
        i = j

    # 3. Merge very short rest segments (< 2 intervals) into the neighboring move
    merged = []
    for (a, b, l) in raw:
        if l == "rest" and (b - a) < 2 and merged and merged[-1][2] == "move":
            pa, _, _ = merged.pop()
            merged.append((pa, b, "move"))
        elif merged and merged[-1][2] == "move" and l == "move":
            pa, _, _ = merged.pop()
            merged.append((pa, b, "move"))
        else:
            merged.append((a, b, l))

    # 4. Within each move: first split by velocity sign flips (each flip = new sub-segment),
    #    then in each unidirectional sub-segment split push/slide by where |v| peaks.
    final = []
    for (a, b, l) in merged:
        if l == "rest":
            final.append(("rest", a, b))
            continue

        # 4a. Split by velocity sign: consecutive same-sign frames form a unidirectional sub-segment
        sub_segs = []
        i = a
        while i < b:
            j = i + 1
            sign_i = 1 if v[i] > 0 else (-1 if v[i] < 0 else 0)
            while j < b:
                sign_j = 1 if v[j] > 0 else (-1 if v[j] < 0 else 0)
                # Allow short zero-velocity gaps (do not force a split); only sign flip splits.
                if sign_j != 0 and sign_i != 0 and sign_j != sign_i:
                    break
                j += 1
            if j - i >= 2:  # require at least 2 velocity intervals
                sub_segs.append((i, j))
            i = j

        # 4b. Within each unidirectional sub-segment, split push/slide at the |v| peak.
        for (sa, sb) in sub_segs:
            abs_v = np.abs(v[sa:sb])
            if len(abs_v) == 0:
                continue
            peak_off = int(np.argmax(abs_v))
            if peak_off <= 1:
                final.append(("slide", sa, sb))
            elif peak_off >= len(abs_v) - 2:
                final.append(("push", sa, sb))
            else:
                push_end = sa + peak_off + 1
                if push_end - sa >= 2:
                    final.append(("push", sa, push_end))
                slide_start = sa + peak_off
                if sb - slide_start >= 2:
                    final.append(("slide", slide_start, sb))

    segments = []
    for (tp, a, b) in final:
        segments.append({
            "type": tp,
            "pt_start": a, "pt_end": b,
            "frame_start": int(frames[a]), "frame_end": int(frames[b]),
            "t_start": round(float(t[a]), 3), "t_end": round(float(t[b]), 3),
            "n_points": b - a + 1,
        })

    has_slide = any(s["type"] == "slide" for s in segments)
    return {
        "segments": segments,
        "n_valid": n,
        "has_slide": has_slide,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-segment polyfit + physics-compliance scoring
# ─────────────────────────────────────────────────────────────────────────────
def polyfit_horizontal_scoring(centroids: list[dict], seg_info: dict, fps: float) -> dict:
    """Fit x = 1/2 a t^2 + v0 t + x0 on each slide segment, then run sign/magnitude/uniformity checks."""
    if "error" in seg_info:
        return {"error": seg_info["error"]}

    valid = [(c["frame_idx"], c["x_norm"]) for c in centroids
             if c["valid"] and c.get("x_norm") is not None]
    valid.sort(key=lambda v: v[0])
    frames = np.array([v[0] for v in valid])
    x = np.array([v[1] for v in valid])
    t = frames / fps

    seg_details = []
    r2_vals = []
    g_fit_vals = []
    seg_scores = []

    for s in seg_info["segments"]:
        a, b = s["pt_start"], s["pt_end"]
        n_pts = b - a + 1
        seg = {**s}

        if s["type"] in ("rest", "push") or n_pts < 4:
            seg["R2"] = None
            seg["a_fit_norm"] = None
            if s["type"] == "push":
                seg["note"] = "push segment (external acceleration phase; not scored)"
            elif s["type"] == "rest":
                seg["note"] = "rest segment (not fitted)"
            else:
                seg["note"] = "too few points (<4) to fit"
            seg_details.append(seg)
            continue

        tt = t[a:b + 1]
        xx = x[a:b + 1]
        x_span = float(xx.max() - xx.min())
        if x_span < 0.03:
            seg["R2"] = None
            seg["a_fit_norm"] = None
            seg["type_raw"] = s["type"]
            seg["type"] = "rest"
            seg["note"] = f"x-span too small (Dx={x_span:.3f} < 0.03); treated as rest"
            seg_details.append(seg)
            continue

        tt0 = tt - tt[0]
        coeffs = np.polyfit(tt0, xx, deg=2)
        a2, a1, a0 = coeffs
        x_pred = np.polyval(coeffs, tt0)
        ss_res = float(np.sum((xx - x_pred) ** 2))
        ss_tot = float(np.sum((xx - np.mean(xx)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        r2 = max(0.0, min(1.0, r2))

        # Uniformity: half-split, compare a from each half
        a_first = None; a_second = None; half_cv = 0.0
        if n_pts >= 8:
            mid = n_pts // 2
            tt_a = tt0[:mid]; xx_a = xx[:mid]
            tt_b = tt0[mid:]; xx_b = xx[mid:]
            if len(tt_a) >= 4 and len(tt_b) >= 4:
                ca = np.polyfit(tt_a, xx_a, deg=2)
                cb = np.polyfit(tt_b, xx_b, deg=2)
                a_first = float(2.0 * ca[0])
                a_second = float(2.0 * cb[0])
                a_ref = max(abs(a_first), abs(a_second), 1e-6)
                half_cv = float(abs(a_first - a_second) / a_ref)

        a_fit = 2.0 * abs(a2)  # deceleration magnitude
        t_span = float(tt[-1] - tt[0])
        # Polyfit-derived v_start (=a1) and v_end, used for the v_decay check
        v_start = float(a1)
        v_end = float(2.0 * a2 * t_span + a1)
        if abs(v_start) > 1e-6:
            v_decay = max(0.0, (abs(v_start) - abs(v_end)) / abs(v_start))
        else:
            v_decay = 0.0
        # Auxiliary fields kept for downstream tooling:
        expected_a = (2.0 * x_span / (t_span ** 2)) if t_span > 1e-6 else 0.0
        ratio = (a_fit / expected_a) if expected_a > 1e-6 else 0.0

        # Magnitude check: v_decay = 1 - |v_end|/|v_start|
        # A real slide should show measurable deceleration; only frictionless / constant-velocity motion scores 0.
        if abs(v_start) < 1e-6:
            magnitude_ok = 0.0
            mag_note = "initial velocity ~ 0; no slide signal"
        elif v_decay >= 0.30:
            magnitude_ok = 1.0; mag_note = ""
        elif v_decay >= 0.05:
            # Linear: 0.05 -> 0.4, 0.30 -> 1.0
            magnitude_ok = 0.4 + 0.6 * (v_decay - 0.05) / 0.25
            mag_note = f"v_decay={v_decay:.2f}; slide did not decelerate fully (partial credit)"
        else:
            # v_decay < 0.05 is essentially constant velocity (no friction); decay rapidly to 0.
            magnitude_ok = max(0.0, v_decay / 0.05) * 0.4
            mag_note = f"v hardly changed ({v_decay:.2f}); essentially frictionless constant velocity"

        # Sign check: a opposite to v (friction decelerates).
        # v0 = a1 (fitted initial velocity); a = 2*a2.
        # Deceleration => a opposite to v0 => a2 and a1 have opposite signs => a1*a2 < 0.
        sign_correct = (a1 * a2 < 0)
        sign_confident = min(1.0, ratio / 0.3)
        sign_note = ""
        if s["type"] == "slide":
            sign_base = 1.0 if sign_correct else 0.0
            if not sign_correct and sign_confident > 0.5:
                sign_note = "slide segment has acceleration in the same direction as velocity (no friction or anti-friction violation)"
        else:
            sign_base = 1.0
        sign_ok = 1.0 - sign_confident * (1.0 - sign_base)

        # Uniformity check
        if a_first is None or a_second is None:
            uniformity_ok = 1.0; unif_note = ""
        elif half_cv <= 0.15:
            uniformity_ok = 1.0; unif_note = ""
        elif half_cv >= 0.80:
            uniformity_ok = 0.0
            unif_note = f"half-to-half a difference {half_cv:.2f}; mu*g is not constant (unstable friction)"
        else:
            uniformity_ok = 1.0 - (half_cv - 0.15) / (0.80 - 0.15)
            unif_note = f"half-to-half a difference {half_cv:.2f}; friction coefficient drifts"
        # Short segments (< 8 points per half) are noise-dominated by SAM2; floor uniformity at 0.5
        # to prevent a real ~13-frame slide being driven to 0 by tracking jitter.
        if n_pts < 16 and uniformity_ok < 0.5:
            uniformity_ok = 0.5
            if unif_note:
                unif_note += " (few points; floored to 0.5)"

        seg_score = sign_ok * magnitude_ok * uniformity_ok

        seg["R2"] = round(r2, 4)
        seg["a_fit_norm"] = round(a_fit, 4)
        seg["v0_norm"] = round(v_start, 4)
        seg["v_end_norm"] = round(v_end, 4)
        seg["v_decay"] = round(v_decay, 3)
        seg["x_span"] = round(x_span, 4)
        seg["expected_a"] = round(expected_a, 4)
        seg["a_ratio"] = round(ratio, 3)
        seg["a_half_first"] = round(a_first, 4) if a_first is not None else None
        seg["a_half_second"] = round(a_second, 4) if a_second is not None else None
        seg["half_cv"] = round(half_cv, 3)
        seg["sign_ok"] = round(sign_ok, 3)
        seg["magnitude_ok"] = round(magnitude_ok, 3)
        seg["uniformity_ok"] = round(uniformity_ok, 3)
        seg["seg_score"] = round(seg_score, 4)
        seg["note"] = " | ".join([n for n in (sign_note, mag_note, unif_note) if n])
        seg_details.append(seg)

        r2_vals.append(r2)
        g_fit_vals.append(a_fit)
        seg_scores.append(seg_score)

    # Cross-segment a consistency when there are multiple slide segments
    if len(g_fit_vals) >= 2:
        a_mean = float(np.mean(g_fit_vals))
        a_cv = float(np.std(g_fit_vals) / a_mean) if a_mean > 1e-6 else 1.0
        cross_consistency = max(0.0, 1.0 - a_cv)
    else:
        a_mean = float(g_fit_vals[0]) if g_fit_vals else None
        cross_consistency = None

    return {
        "method": "horizontal_polyfit_r2_x_sign_x_mag_x_unif",
        "segments": seg_details,
        "mean_R2": round(float(np.mean(r2_vals)), 4) if r2_vals else 0.0,
        "mean_seg_score": round(float(np.mean(seg_scores)), 4) if seg_scores else 0.0,
        "a_fit_norm_mean": round(a_mean, 4) if a_mean is not None else None,
        "a_cross_consistency": round(cross_consistency, 4) if cross_consistency is not None else None,
        "n_fit_segments": len(r2_vals),
        "n_valid": seg_info["n_valid"],
        "has_slide": seg_info["has_slide"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Final score
# ─────────────────────────────────────────────────────────────────────────────
def compute_horizontal_score(analysis: dict, n_frames_total: int,
                              window_len: int | None = None) -> dict:
    """Same form as the free-fall final score: mean(seg_score) * coverage_factor * cc_factor."""
    if "error" in analysis:
        return {"score": None, "score_pct": None, "issues": [analysis["error"]],
                "unevaluable": True}

    r2 = analysis.get("mean_R2", 0.0)
    mean_seg = analysis.get("mean_seg_score", 0.0)
    n_valid = analysis.get("n_valid", 0)
    n_fit_segs = analysis.get("n_fit_segments", 0)
    denom = window_len if window_len and window_len >= 6 else max(8, n_frames_total * 0.5)
    coverage = min(1.0, n_valid / denom)
    cc = analysis.get("a_cross_consistency")

    if coverage < 0.3 or n_valid < 6:
        return {
            "score": None, "score_pct": None,
            "mean_R2": round(r2, 4), "mean_seg_score": round(mean_seg, 4),
            "coverage": round(coverage, 3),
            "a_cross_consistency": round(cc, 3) if cc is not None else None,
            "unevaluable": True,
            "issues": [f"unevaluable: n_valid={n_valid}, coverage={coverage:.2f}"],
        }
    if n_fit_segs == 0:
        # Valid frames but no fitted slide: likely all push (anti-friction / continuous acceleration); score 0
        segs = analysis.get("segments", []) or []
        push_pts = sum(s.get("n_points", 0) for s in segs if s.get("type") == "push")
        if push_pts >= 6:
            return {
                "score": 0.0, "score_pct": 0.0,
                "mean_R2": round(r2, 4), "mean_seg_score": 0.0,
                "coverage": round(coverage, 3),
                "a_cross_consistency": None,
                "unevaluable": False,
                "issues": ["accelerating throughout with no deceleration segment (anti-friction or continuous external force)"],
            }
        return {
            "score": None, "score_pct": None,
            "mean_R2": round(r2, 4), "mean_seg_score": 0.0,
            "coverage": round(coverage, 3),
            "a_cross_consistency": None,
            "unevaluable": True,
            "issues": ["no scoreable slide segment"],
        }

    # slide_coverage = fitted slide frames / valid tracked frames
    # Prevents the false-positive "one short uniform-deceleration window in a long video gets the whole video a high score".
    fit_slide_frames = sum(
        s.get("n_points", 0)
        for s in (analysis.get("segments") or [])
        if s.get("type") == "slide" and s.get("R2") is not None
    )
    slide_coverage = fit_slide_frames / max(n_valid, 1)
    if slide_coverage < 0.20:
        # slide ratio < 20% -> unevaluable
        return {
            "score": None, "score_pct": None,
            "mean_R2": round(r2, 4), "mean_seg_score": round(mean_seg, 4),
            "coverage": round(coverage, 3),
            "slide_coverage": round(slide_coverage, 3),
            "a_cross_consistency": round(cc, 3) if cc is not None else None,
            "unevaluable": True,
            "issues": [f"slide-frame ratio={slide_coverage:.2f}<0.20; "
                       f"video is mostly static/held; not a scoreable horizontal-push event"],
        }
    # Linear ramp on [0.20, 0.50]; full weight when >= 0.50
    slide_factor = min(1.0, slide_coverage / 0.50)

    coverage_factor = min(1.0, coverage / 0.6)
    cc_factor = 1.0 if cc is None else max(0.3, cc)
    score = float(max(0.0, min(1.0, mean_seg * coverage_factor * cc_factor * slide_factor)))

    issues = []
    if not analysis.get("has_slide", False):
        issues.append("no slide (friction-deceleration) phase detected")
    if r2 < 0.85:
        issues.append(f"info: polyfit R^2={r2:.2f} (diagnostic only, not used in scoring)")
    for s in analysis.get("segments", []):
        if s.get("note") and s.get("seg_score") is not None and s["seg_score"] < 0.5:
            issues.append(f"segment[{s['type']}]: {s['note']}")
    if cc is not None and cc < 0.6:
        issues.append(f"cross-segment a consistency={cc:.2f}; segments disagree on friction")
    if coverage < 0.6:
        issues.append(f"valid tracking coverage={coverage:.2f}; SAM2 lost the object")

    return {
        "score": round(score, 4),
        "score_pct": round(score * 100, 1),
        "mean_R2": round(r2, 4),
        "mean_seg_score": round(mean_seg, 4),
        "coverage": round(coverage, 3),
        "coverage_factor": round(coverage_factor, 3),
        "slide_coverage": round(slide_coverage, 3),
        "slide_factor": round(slide_factor, 3),
        "cc_factor": round(cc_factor, 3),
        "a_cross_consistency": round(cc, 3) if cc is not None else None,
        "unevaluable": False,
        "issues": issues,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main entry
# ─────────────────────────────────────────────────────────────────────────────
def analyze_horizontal_video(
    video_path: str,
    output_dir: str | None = None,
    first_bbox: tuple[float, float, float, float] | None = None,
    first_point: tuple[float, float] | None = None,
    start_frame: int = 0,
    use_vlm: bool = True,
    save_mask_video: bool = False,
) -> dict:
    """Main entry (horizontal-push variant).

    Args:
        first_bbox: manually specify the pushed-object bbox (normalized), typically at the moment motion begins.
        first_point: manual positive point (alternative to first_bbox).
        start_frame: frame at which SAM2 begins forward propagation.
        use_vlm: if True and no manual prompt is given, the VLM locates the object at frame=start_frame.
    """
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/4] Locating pushed object...")
    pos_pt = None; bbox = None; obj_name = "manual"; neg_pts = []

    if first_bbox is not None:
        bbox = list(first_bbox)
        pos_pt = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        print(f"  manual bbox: {bbox}, start_frame={start_frame}")
    elif first_point is not None:
        pos_pt = list(first_point)
        print(f"  manual point: {pos_pt}, start_frame={start_frame}")
    elif use_vlm:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        _, f = cap.read(); cap.release()
        loc = vlm_locate_first_frame(f)
        bbox = loc.get("bbox")
        pos_pt = [loc.get("x_norm"), loc.get("y_norm")] if loc.get("x_norm") is not None else None
        neg_pts = loc.get("negative_points") or []
        obj_name = loc.get("object", "")
        print(f"  VLM: obj={obj_name}, bbox={bbox}")
    else:
        raise ValueError("Provide first_bbox or first_point, or enable use_vlm")

    print(f"[2/4] SAM2 forward propagation from frame {start_frame}...")
    t0 = time.time()
    track = sam2_track_centroids(
        video_path,
        positive_point=tuple(pos_pt) if pos_pt else None,
        bbox=tuple(bbox) if bbox else None,
        negative_points=neg_pts or None,
        prompt_frame_idx=start_frame,
        keep_masks=save_mask_video,
    )
    print(f"  elapsed {time.time()-t0:.1f}s, frames={track['n_frames']}, "
          f"valid centroids={sum(1 for c in track['centroids'] if c['valid'])}")

    print(f"[3/4] Segmentation (push/slide/rest) + per-segment polyfit...")
    seg_info = segment_horizontal_trajectory(track["centroids"], track["fps"])
    analysis = polyfit_horizontal_scoring(track["centroids"], seg_info, track["fps"])

    print(f"[4/4] Final scoring...")
    score_result = compute_horizontal_score(analysis, n_frames_total=track["n_frames"])

    result = {
        "video_path": video_path,
        "fps": track["fps"],
        "total_frames": track["n_frames"],
        "frame_hw": [track["h"], track["w"]],
        "object": obj_name,
        "start_frame": start_frame,
        "positions": [
            {"frame_idx": c["frame_idx"], "x_norm": c.get("x_norm"),
             "y_norm": c.get("y_norm"), "mask_area": c.get("mask_area"),
             "confidence": "high" if c["valid"] else "none"}
            for c in track["centroids"]
        ],
        "analysis": analysis,
        "score": score_result,
    }

    basename = os.path.splitext(os.path.basename(video_path))[0]
    json_path = os.path.join(output_dir, f"{basename}_h_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Result saved: {json_path}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Annotated video
# ─────────────────────────────────────────────────────────────────────────────
def annotate_horizontal_video(video_path: str, result: dict, output_path: str) -> str:
    """Render an annotated video: centroid dot + per-frame segment label + score overlay."""
    if "error" in result:
        return video_path
    positions = {p["frame_idx"]: p for p in result["positions"]}
    segments = result.get("analysis", {}).get("segments", [])

    frame_phase = {}
    for s in segments:
        for fi in range(s["frame_start"], s["frame_end"] + 1):
            frame_phase[fi] = {"type": s["type"], "R2": s.get("R2"),
                               "seg_score": s.get("seg_score")}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not out.isOpened():
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    score_pct = result.get("score", {}).get("score_pct", 0)
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        p = positions.get(fi)
        if p and p.get("x_norm") is not None and p.get("y_norm") is not None:
            x_px = int(p["x_norm"] * (w - 1))
            y_px = int(p["y_norm"] * (h - 1))
            cv2.circle(frame, (x_px, y_px), 6, (0, 100, 255), -1)
            cv2.circle(frame, (x_px, y_px), 6, (255, 255, 255), 2)
        info = frame_phase.get(fi)
        if info is not None:
            t_ = info["type"]; r2 = info.get("R2")
            seg_sc = info.get("seg_score")
            if t_ == "slide":
                color = (0, 160, 0)
                lbl = f"SLIDE  R2={r2:.2f}" if r2 is not None else "SLIDE"
                if seg_sc is not None and seg_sc < 0.5:
                    color = (30, 30, 200); lbl = f"SLIDE-ANOM s={seg_sc:.2f}"
            elif t_ == "push":
                color = (180, 100, 40); lbl = "PUSH (not scored)"
            else:
                color = (120, 120, 120); lbl = "REST"
            cv2.rectangle(frame, (0, 0), (380, 50), color, -1)
            cv2.putText(frame, lbl, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (255, 255, 255), 2)
        cv2.rectangle(frame, (w - 240, 0), (w, 70), (30, 30, 30), -1)
        score_text = "Score: N/A" if score_pct is None else f"Score: {score_pct:.1f}%"
        cv2.putText(frame, score_text, (w - 230, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 215, 0), 2)
        out.write(frame); fi += 1
    cap.release(); out.release()
    return output_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python horizontal_push_analyzer.py <video_path>")
        sys.exit(1)
    res = analyze_horizontal_video(sys.argv[1])
    sc = res["score"]
    print(f"\n  score: {sc.get('score_pct')}%  R²: {sc.get('mean_R2')}  "
          f"cc: {sc.get('a_cross_consistency')}")
