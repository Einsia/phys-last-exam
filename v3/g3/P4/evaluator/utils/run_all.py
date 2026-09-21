#!/usr/bin/env python3
"""Run the P4 evaluator over the frozen 24-video MiniMax-H3 manifest."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_BATCH = (HERE.parent / "minimax_h3" / "prompt_v2_20260826" / "videos")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--videos", type=Path, default=DEFAULT_BATCH)
    p.add_argument("--outdir", type=Path, default=HERE / "results")
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--no-cotracker", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def flatten(result: dict) -> dict:
    m = result.get("metrics", {})
    s = result.get("scores", {})
    legacy = result.get("legacy_scores_0_100", {})
    q = result.get("qc", {})
    meas = result.get("measurements", {})
    return {
        "sample_id": result.get("sample_id"),
        "video": result.get("video"),
        "extract_success": result.get("extract_success"),
        "structural_ok": result.get("structural_ok"),
        "physics_pass": result.get("physics_pass"),
        "measurement_valid": result.get("measurement_valid"),
        "score_version": result.get("score_version"),
        "complete_rebound_arc_count": meas.get("complete_rebound_arc_count"),
        "M1_height_time_consistency": m.get("M1_height_time_restitution_consistency"),
        "M2_restitution_cv": m.get("M2_restitution_coefficient_cv"),
        "M2_monotonic_height_error": m.get("M2_monotonic_height_error"),
        "M2_near_nondecreasing_height_fraction": m.get("M2_near_nondecreasing_height_fraction"),
        "arc_shape_rmse": m.get("arc_shape_normalized_rmse"),
        "impact_x_spread_radii": m.get("impact_x_robust_spread_radii"),
        "score_structure": s.get("structure"),
        "score_monotonic": s.get("monotonic_energy_loss"),
        "score_ballistic": s.get("ballistic_arc_quality"),
        "score_restitution": s.get("restitution_consistency"),
        "score_spatial": s.get("spatial_stability"),
        "score_overall": s.get("overall"),
        "legacy_score_structure_0_100": legacy.get("structure"),
        "legacy_score_monotonic_0_100": legacy.get("monotonic_energy_loss"),
        "legacy_score_ballistic_0_100": legacy.get("ballistic_arc_quality"),
        "legacy_score_restitution_0_100": legacy.get("restitution_consistency"),
        "legacy_score_spatial_0_100": legacy.get("spatial_stability"),
        "legacy_score_overall_0_100": legacy.get("overall"),
        "fused_track_coverage": q.get("tracking", {}).get("fused_coverage"),
        "failure_reason": result.get("failure_reason"),
        "runtime_seconds": result.get("runtime_seconds"),
    }


def main() -> int:
    args = parse_args()
    videos = sorted(args.videos.glob("*.mp4"))
    if args.limit is not None:
        videos = videos[:args.limit]
    if not videos:
        raise SystemExit(f"no mp4 files under {args.videos}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for i, video in enumerate(videos, 1):
        sample_out = args.outdir / "json" / f"{video.stem}.json"
        debug = args.outdir / "debug" / video.stem
        sample_out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(HERE / "evaluate.py"),
               "--video", str(video), "--task_id", "P4",
               "--output", str(sample_out), "--config", str(args.config),
               "--debug-dir", str(debug)]
        if args.no_cotracker:
            cmd.append("--no-cotracker")
        print(f"[{i}/{len(videos)}] {video.name}", flush=True)
        proc = subprocess.run(cmd, text=True)
        if sample_out.exists():
            results.append(json.loads(sample_out.read_text()))
        else:
            print(f"  evaluator produced no JSON (rc={proc.returncode})", flush=True)

    rows = [flatten(r) for r in results]
    if rows:
        with (args.outdir / "results.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    n = len(results)
    current_values = [float(r.get("scores", {}).get("overall", 0.0)) for r in results]
    valid_values = [float(r.get("scores", {}).get("overall", 0.0)) for r in results
                    if r.get("measurement_valid")]
    legacy_values = [float(r.get("legacy_scores_0_100", {}).get("overall", 0.0))
                     for r in results]
    summary = {
        "task_id": "P4", "n_planned": len(videos), "n_results": n,
        "score_version": "continuous-0-1-v1",
        "score_range": [0.0, 1.0],
        "zero_score_policy": "measurement_invalid_only",
        "llm_or_vlm_used": False,
        "extract_success_rate": sum(bool(r.get("extract_success")) for r in results) / max(n, 1),
        "structural_success_rate": sum(bool(r.get("structural_ok")) for r in results) / max(n, 1),
        "measurement_valid_rate": sum(bool(r.get("measurement_valid")) for r in results) / max(n, 1),
        "end_to_end_pass_rate": sum(bool(r.get("physics_pass")) for r in results) / max(n, 1),
        "end_to_end_mean_score": statistics.fmean(current_values) if current_values else 0.0,
        "legacy_end_to_end_mean_score_0_100": statistics.fmean(legacy_values) if legacy_values else 0.0,
        "continuous_score_mean_all": statistics.fmean(current_values) if current_values else 0.0,
        "continuous_score_mean_valid": statistics.fmean(valid_values) if valid_values else 0.0,
        "continuous_score_median_valid": statistics.median(valid_values) if valid_values else 0.0,
        "continuous_score_min_valid": min(valid_values) if valid_values else 0.0,
        "continuous_score_max_valid": max(valid_values) if valid_values else 0.0,
        "continuous_score_zero_count": sum(value == 0.0 for value in current_values),
        "failure_counts": {},
        "results_csv": str((args.outdir / "results.csv").resolve()),
    }
    for r in results:
        reason = r.get("failure_reason") or "none"
        summary["failure_counts"][reason] = summary["failure_counts"].get(reason, 0) + 1
    if any(r.get("score_version") != "continuous-0-1-v1" for r in results):
        raise AssertionError("batch contains a non-continuous score schema")
    if any(not 0.0 <= value <= 1.0 for value in current_values):
        raise AssertionError("current P4 score outside [0,1]")
    (args.outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
