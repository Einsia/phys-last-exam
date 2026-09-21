#!/usr/bin/env python3
"""Run the frozen P6 evaluator over all 24 formal videos."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEOS = SCRIPT_DIR.parent / "minimax_h3" / "prompt_v1_20260825" / "videos"


def evaluate_one(video: Path, output_root: Path, config: Path) -> tuple[str, int, str]:
    sample_dir = output_root / video.stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate.py"),
        "--video",
        str(video),
        "--task_id",
        "P6",
        "--output",
        str(sample_dir / "result.json"),
        "--config",
        str(config),
        "--artifacts-dir",
        str(sample_dir),
    ]
    process = subprocess.run(command, text=True, capture_output=True)
    log = process.stdout + process.stderr
    (sample_dir / "evaluator.log").write_text(log, encoding="utf-8")
    return video.stem, process.returncode, log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "results_v1")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config_v1.yaml")
    parser.add_argument("--pattern", default="P6_*_seed*.mp4")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    videos = sorted(args.videos.glob(args.pattern))
    if args.limit:
        videos = videos[: args.limit]
    if args.limit is None and args.pattern == "P6_*_seed*.mp4" and len(videos) != 24:
        raise SystemExit(f"expected exactly 24 formal P6 videos, found {len(videos)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(evaluate_one, video, args.output_root, args.config): video
            for video in videos
        }
        for future in as_completed(futures):
            sample_id, return_code, log = future.result()
            print(f"[{sample_id}] return_code={return_code}")
            if return_code:
                failures.append((sample_id, return_code, log[-2000:]))

    rows = []
    for video in videos:
        result_path = args.output_root / video.stem / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = result.get("metrics", {})
        scores = result.get("scores", {})
        rows.append(
            {
                "task_id": "P6",
                "sample_id": video.stem,
                "source_id": result.get("source_id"),
                "extract_success": result.get("extract_success"),
                "structural_ok": result.get("structural_ok"),
                "physics_pass": result.get("physics_pass"),
                "overall_score": scores.get("overall"),
                "score_task_structure": scores.get("task_structure"),
                "score_center_contact_geometry": scores.get("center_contact_geometry"),
                "score_marker_rotation_quality": scores.get("marker_rotation_quality"),
                "score_pure_rolling_ratio": scores.get("pure_rolling_ratio"),
                "score_contact_point_velocity": scores.get("contact_point_velocity"),
                "M1_rolling_ratio_abs": metrics.get("M1_rolling_ratio_abs"),
                "M2_contact_velocity_nmae": metrics.get("M2_contact_velocity_nmae"),
                "track_coverage": metrics.get("track_coverage"),
                "pattern_coverage": metrics.get("pattern_coverage"),
                "hough_track_coverage": metrics.get("hough_track_coverage"),
                "max_hough_dropout_run_frames": metrics.get("max_hough_dropout_run_frames"),
                "failure_reason": json.dumps(result.get("failure_reason"), ensure_ascii=False),
            }
        )
    if rows:
        with (args.output_root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "formal_video_count": len(videos),
        "result_count": len(rows),
        "evaluator_process_failures": failures,
        "extract_success_count": sum(row["extract_success"] is True for row in rows),
        "structural_ok_count": sum(row["structural_ok"] is True for row in rows),
        "physics_pass_count": sum(row["physics_pass"] is True for row in rows),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
