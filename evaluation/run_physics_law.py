"""
Batch driver for the physics-law evaluator.

Recursively scans an input directory for *.mp4 files, runs the unified
motion analyzer on each, writes per-video JSON results and an annotated
overlay video, and combines the explicit kinematic score with the VLM
Video Quality Score (VQS) into a final 0-100 PhysLaw score.

Usage:
    export PHYSBENCH_VLM_API_KEY="your-key"
    python run_batch.py --input_dir path/to/videos --output_dir path/to/results

Optional flags:
    --no-annotate           Skip writing the annotated MP4 (faster, less disk).
    --skip-existing         Skip videos whose result JSON already exists.
    --pattern "*.mp4"       Glob pattern to match (default: *.mp4).

Environment variables (forwarded to physics_analyzer):
    PHYSBENCH_VLM_API_KEY   (required)
    PHYSBENCH_VLM_BASE_URL  OpenAI-compatible endpoint
    PHYSBENCH_VLM_MODEL     model id (default: gpt-4.1-mini)
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_motion_analyzer import analyze_motion          # noqa: E402
from physics_analyzer import annotate_video, video_quality_check  # noqa: E402


def evaluate_one(video_path: Path, output_dir: Path, annotate: bool) -> dict:
    """Run the full pipeline on one video; return the result dict."""
    out_json = output_dir / f"{video_path.stem}_unified.json"
    out_mp4  = output_dir / f"{video_path.stem}_annotated.mp4"

    r = analyze_motion(str(video_path), output_dir=str(output_dir),
                       use_vlm=True, auto_release_detect=False)

    if annotate:
        mode = r.get("primary_mode")
        scored = r.get(mode) if mode in ("vertical", "horizontal") else None
        if scored and scored.get("analysis"):
            view = {
                "positions":    r["positions"],
                "analysis":     scored["analysis"],
                "score":        scored["score"],
                "label_window": r.get("label_window"),
            }
            try:
                annotate_video(str(video_path), view, str(out_mp4))
            except Exception as e:
                print(f"      [warn] annotate_video failed: {e}", flush=True)

    qc = video_quality_check(str(video_path))
    r["video_quality"] = qc
    physics_score = (r.get("primary_score") or {}).get("score_pct")
    vqs           = qc.get("vqs")
    has_motion    = qc.get("has_motion")
    if vqs is not None:
        effective_physics = (physics_score or 0.0) if has_motion else 0.0
        r["final_score"] = round(0.9 * effective_physics + vqs, 2)
    else:
        r["final_score"] = None

    out_json.write_text(json.dumps(r, ensure_ascii=False))
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir",  required=True, help="Directory of input videos")
    parser.add_argument("--output_dir", required=True, help="Where to write results")
    parser.add_argument("--pattern",    default="*.mp4", help="Glob pattern (default: *.mp4)")
    parser.add_argument("--no-annotate", action="store_true", help="Skip annotated MP4 output")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip videos whose result JSON already exists")
    args = parser.parse_args()

    if not os.environ.get("PHYSBENCH_VLM_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("error: set PHYSBENCH_VLM_API_KEY (or OPENAI_API_KEY)")

    src = Path(args.input_dir).expanduser().resolve()
    dst = Path(args.output_dir).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)

    videos = sorted(src.rglob(args.pattern))
    print(f"Found {len(videos)} videos under {src}")
    print(f"Writing results to {dst}\n")

    done = skip = fail = 0
    for i, vp in enumerate(videos, 1):
        out_json = dst / f"{vp.stem}_unified.json"
        if args.skip_existing and out_json.exists():
            print(f"[{i:03d}/{len(videos)}] SKIP {vp.name}", flush=True)
            skip += 1
            continue

        print(f"[{i:03d}/{len(videos)}] {vp.name}", flush=True)
        try:
            r = evaluate_one(vp, dst, annotate=not args.no_annotate)
            ps = (r.get("primary_score") or {}).get("score_pct")
            vqs = (r.get("video_quality") or {}).get("vqs")
            print(f"      mode={r.get('primary_mode')}  "
                  f"phys={ps}  vqs={vqs}/10  final={r.get('final_score')}",
                  flush=True)
            done += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"      FAILED: {e}", flush=True)
            fail += 1

    print(f"\nDONE  done={done}  skip={skip}  fail={fail}")


if __name__ == "__main__":
    main()
