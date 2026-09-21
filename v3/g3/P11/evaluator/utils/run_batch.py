#!/usr/bin/env python3
"""Run the deterministic P11 evaluator over every MP4 in one prompt batch."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from evaluate import evaluate_video, hash_file


def nested(record: Dict[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def worker(video: str, config: str, json_path: str, debug_dir: str) -> Dict[str, Any]:
    return evaluate_video(
        Path(video),
        "P11",
        Path(config),
        output_path=Path(json_path),
        debug_dir=Path(debug_dir),
        write_debug=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-name", required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    frozen_path = args.config.with_name("FROZEN_CONFIG.sha256")
    if frozen_path.exists():
        expected_hash = frozen_path.read_text(encoding="utf-8").strip().split()[0]
        actual_hash = hash_file(args.config)
        if actual_hash != expected_hash:
            raise SystemExit(
                f"refusing batch run: config hash {actual_hash} differs from frozen {expected_hash}"
            )
    videos = sorted(args.videos_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no MP4 files found in {args.videos_dir}")
    json_dir = args.output_dir / "json"
    debug_root = args.output_dir / "debug"
    json_dir.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                worker,
                str(video.resolve()),
                str(args.config.resolve()),
                str((json_dir / f"{video.stem}.json").resolve()),
                str((debug_root / video.stem).resolve()),
            ): video
            for video in videos
        }
        for future in as_completed(futures):
            video = futures[future]
            try:
                result = future.result()
                results[video.stem] = result
                print(
                    f"DONE {video.stem} valid={nested(result, 'statuses', 'measurement_valid')} "
                    f"pass={nested(result, 'statuses', 'physics_pass')} "
                    f"score={nested(result, 'scores', 'overall')}"
                )
            except Exception as exc:
                failures[video.stem] = f"{type(exc).__name__}:{exc}"
                print(f"FAIL {video.stem} {failures[video.stem]}")
    fields = [
        "batch_name",
        "video_id",
        "sample_id",
        "extract_success",
        "structural_ok",
        "measurement_valid",
        "M1_valid",
        "M2_valid",
        "physics_pass",
        "theta_i_deg",
        "theta_t_deg",
        "snell_ratio",
        "M1_signed",
        "M1_abs",
        "M2_normalized",
        "M1_score",
        "M2_score",
        "metric_score",
        "overall_gated_score",
        "legacy_M1_score_0_100",
        "legacy_M2_score_0_100",
        "legacy_metric_score_0_100",
        "legacy_overall_gated_score_0_100",
        "scoring_version",
        "continuous_M1_snell_quality",
        "continuous_M2_intersection_quality",
        "continuous_fit_quality",
        "continuous_temporal_stability",
        "continuous_camera_stability",
        "continuous_physics_pair_quality",
        "continuous_overall",
        "invalid_reasons",
        "json_path",
        "overlay_path",
    ]
    rows = []
    for video in videos:
        result = results.get(video.stem)
        if result is None:
            rows.append(
                {
                    "batch_name": args.batch_name,
                    "video_id": video.stem,
                    "invalid_reasons": failures.get(video.stem, "missing_result"),
                    "json_path": str((json_dir / f"{video.stem}.json").resolve()),
                }
            )
            continue
        rows.append(
            {
                "batch_name": args.batch_name,
                "video_id": video.stem,
                "sample_id": result.get("sample_id"),
                "extract_success": nested(result, "statuses", "extract_success"),
                "structural_ok": nested(result, "statuses", "structural_ok"),
                "measurement_valid": nested(result, "statuses", "measurement_valid"),
                "M1_valid": nested(result, "statuses", "metric_validity", "M1_snell_residual"),
                "M2_valid": nested(result, "statuses", "metric_validity", "M2_intersection_consistency"),
                "physics_pass": nested(result, "statuses", "physics_pass"),
                "theta_i_deg": nested(result, "geometry", "angles", "measured_incidence_from_normal_deg"),
                "theta_t_deg": nested(result, "geometry", "angles", "measured_refraction_from_normal_deg"),
                "snell_ratio": nested(result, "metrics", "M1_snell_ratio"),
                "M1_signed": nested(result, "metrics", "M1_snell_residual_signed"),
                "M1_abs": nested(result, "metrics", "M1_snell_residual_abs"),
                "M2_normalized": nested(result, "metrics", "M2_intersection_disagreement_normalized"),
                "M1_score": nested(result, "scores", "dimensions", "M1_snell_quality"),
                "M2_score": nested(result, "scores", "dimensions", "M2_intersection_quality"),
                "metric_score": nested(result, "scores", "derived", "physics_pair_quality"),
                "overall_gated_score": nested(result, "scores", "overall"),
                "legacy_M1_score_0_100": nested(result, "legacy_scores_0_100", "M1_score_0_100"),
                "legacy_M2_score_0_100": nested(result, "legacy_scores_0_100", "M2_score_0_100"),
                "legacy_metric_score_0_100": nested(result, "legacy_scores_0_100", "geometric_metric_score_0_100"),
                "legacy_overall_gated_score_0_100": nested(result, "legacy_scores_0_100", "overall_gated_score_0_100"),
                "scoring_version": nested(result, "scores", "scoring_version"),
                "continuous_M1_snell_quality": nested(result, "scores", "dimensions", "M1_snell_quality"),
                "continuous_M2_intersection_quality": nested(result, "scores", "dimensions", "M2_intersection_quality"),
                "continuous_fit_quality": nested(result, "scores", "dimensions", "fit_quality"),
                "continuous_temporal_stability": nested(result, "scores", "dimensions", "temporal_stability"),
                "continuous_camera_stability": nested(result, "scores", "dimensions", "camera_stability"),
                "continuous_physics_pair_quality": nested(result, "scores", "derived", "physics_pair_quality"),
                "continuous_overall": nested(result, "scores", "overall"),
                "invalid_reasons": ";".join(nested(result, "statuses", "measurement_invalid_reasons") or []),
                "json_path": str((json_dir / f"{video.stem}.json").resolve()),
                "overlay_path": nested(result, "debug", "geometry_overlay"),
            }
        )
    csv_path = args.output_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    validity_count = sum(bool(row.get("measurement_valid")) for row in rows)
    structural_count = sum(bool(row.get("structural_ok")) for row in rows)
    pass_count = sum(row.get("physics_pass") is True for row in rows)
    valid_scores = [float(row["continuous_overall"]) for row in rows if row.get("measurement_valid")]
    legacy_scores = [
        float(row["legacy_overall_gated_score_0_100"])
        for row in rows
        if row.get("measurement_valid")
    ]
    report = {
        "batch_name": args.batch_name,
        "videos_dir": str(args.videos_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "evaluator_config": str(args.config.resolve()),
        "config_sha256": hash_file(args.config),
        "planned_video_count": len(videos),
        "completed_result_count": len(results),
        "fatal_failure_count": len(failures),
        "structural_ok_count": structural_count,
        "measurement_valid_count": validity_count,
        "physics_pass_count": pass_count,
        "mean_valid_overall_score": (sum(valid_scores) / len(valid_scores)) if valid_scores else None,
        "legacy_mean_valid_overall_score_0_100": (
            (sum(legacy_scores) / len(legacy_scores)) if legacy_scores else None
        ),
        "score_version": "continuous-0-1-v1",
        "score_range": [0.0, 1.0],
        "zero_score_policy": (
            "overall score is 0 only when measurement_valid is false; "
            "every measurement-valid result has overall score > 0"
        ),
        "scoring_version": "continuous-0-1-v1",
        "legacy_score_summary_0_100": {
            "mean_valid_overall_score": (sum(legacy_scores) / len(legacy_scores)) if legacy_scores else None
        },
        "failures": failures,
        "selection_policy": "all videos retained; no seed filtering; no best-of selection",
        "pooling_policy": "this report contains exactly one prompt batch",
        "model_policy": "no LLM and no VLM",
    }
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    sanity_path = args.output_dir / "human_sanity_review.csv"
    with sanity_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["video_id", "overlay_path", "review_status", "review_notes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "video_id": row["video_id"],
                    "overlay_path": row.get("overlay_path"),
                    "review_status": "pending",
                    "review_notes": "",
                }
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures and len(results) == len(videos) else 2


if __name__ == "__main__":
    raise SystemExit(main())
