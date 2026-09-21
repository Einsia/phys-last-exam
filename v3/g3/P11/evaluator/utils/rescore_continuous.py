#!/usr/bin/env python3
"""Migrate existing P11 JSON/CSV results to continuous-0-1-v1.

This script never opens a video and never reruns interface or light extraction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from continuous_scoring import SCORING_VERSION, migrate_record


DEFAULT_BATCHES = ["prompt_v2_20260826", "prompt_v3_20260826", "prompt_v4_20260826"]
CONTINUOUS_COLUMNS = [
    "scoring_version",
    "continuous_M1_snell_quality",
    "continuous_M2_intersection_quality",
    "continuous_fit_quality",
    "continuous_temporal_stability",
    "continuous_camera_stability",
    "continuous_physics_pair_quality",
    "continuous_overall",
]
CURRENT_SCORE_COLUMNS = ["M1_score", "M2_score", "metric_score", "overall_gated_score"]
LEGACY_SCORE_COLUMNS = [
    "legacy_M1_score_0_100",
    "legacy_M2_score_0_100",
    "legacy_metric_score_0_100",
    "legacy_overall_gated_score_0_100",
]
CURRENT_TO_LEGACY = dict(zip(CURRENT_SCORE_COLUMNS, LEGACY_SCORE_COLUMNS))


def canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def atomic_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def describe(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def continuous_row(result: Dict[str, Any]) -> Dict[str, Any]:
    scores = result["scores"]
    dimensions = scores["dimensions"]
    return {
        "scoring_version": scores["scoring_version"],
        "continuous_M1_snell_quality": dimensions["M1_snell_quality"],
        "continuous_M2_intersection_quality": dimensions["M2_intersection_quality"],
        "continuous_fit_quality": dimensions["fit_quality"],
        "continuous_temporal_stability": dimensions["temporal_stability"],
        "continuous_camera_stability": dimensions["camera_stability"],
        "continuous_physics_pair_quality": scores["derived"]["physics_pair_quality"],
        "continuous_overall": scores["overall"],
    }


def validate_score(result: Dict[str, Any]) -> None:
    scores = result["scores"]
    assert result["score_version"] == SCORING_VERSION
    assert scores["scoring_version"] == SCORING_VERSION
    numbers = list(scores["dimensions"].values()) + [scores["derived"]["physics_pair_quality"], scores["overall"]]
    assert all(0.0 <= float(value) <= 1.0 for value in numbers), numbers
    valid = result["statuses"]["measurement_valid"] is True
    if valid:
        assert float(scores["overall"]) > 0.0
    else:
        assert float(scores["overall"]) == 0.0
    assert "legacy_scores_0_100" in result


def rescore_batch(batch_dir: Path, dry_run: bool) -> Dict[str, Any]:
    json_dir = batch_dir / "json"
    json_paths = sorted(json_dir.glob("*.json"))
    if not json_paths:
        raise RuntimeError(f"no result JSON files under {json_dir}")
    results: Dict[str, Dict[str, Any]] = {}
    unchanged_measurement_digests: Dict[str, str] = {}
    for path in json_paths:
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        before = canonical_digest({"statuses": result.get("statuses"), "metrics": result.get("metrics"), "geometry": result.get("geometry")})
        result = migrate_record(result)
        validate_score(result)
        after = canonical_digest({"statuses": result.get("statuses"), "metrics": result.get("metrics"), "geometry": result.get("geometry")})
        if before != after:
            raise AssertionError(f"rescoring modified status/metric/geometry in {path}")
        unchanged_measurement_digests[result["video_id"]] = before
        results[result["video_id"]] = result
        if not dry_run:
            atomic_json(path, result)
    summary_path = batch_dir / "summary.csv"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        original_fields = list(reader.fieldnames or [])
        rows = list(reader)
    fields = [
        field
        for field in original_fields
        if field not in CONTINUOUS_COLUMNS and field not in LEGACY_SCORE_COLUMNS
    ]
    fields += LEGACY_SCORE_COLUMNS + CONTINUOUS_COLUMNS
    for row in rows:
        video_id = row["video_id"]
        if video_id not in results:
            raise KeyError(f"summary row has no JSON result: {video_id}")
        # Preserve the pre-migration 0--100 CSV values exactly once.  On
        # subsequent runs, retain the explicit legacy columns unchanged.
        for current_name, legacy_name in CURRENT_TO_LEGACY.items():
            if row.get(legacy_name) in (None, ""):
                row[legacy_name] = row.get(current_name)
        continuous = continuous_row(results[video_id])
        row.update(continuous)
        row["M1_score"] = continuous["continuous_M1_snell_quality"]
        row["M2_score"] = continuous["continuous_M2_intersection_quality"]
        row["metric_score"] = continuous["continuous_physics_pair_quality"]
        row["overall_gated_score"] = continuous["continuous_overall"]
    if not dry_run:
        atomic_csv(summary_path, fields, rows)
    valid_results = [result for result in results.values() if result["statuses"]["measurement_valid"] is True]
    overall_values = [float(result["scores"]["overall"]) for result in valid_results]
    dimension_names = list(next(iter(results.values()))["scores"]["dimensions"].keys())
    dimension_stats = {
        name: describe([float(result["scores"]["dimensions"][name]) for result in valid_results])
        for name in dimension_names
    }
    batch_scoring = {
        "scoring_version": SCORING_VERSION,
        "rescored_without_video_or_extraction": True,
        "json_count": len(results),
        "measurement_valid_count": len(valid_results),
        "overall": describe(overall_values),
        "dimensions": dimension_stats,
        "measurement_digest_sha256": canonical_digest(unchanged_measurement_digests),
    }
    report_path = batch_dir / "report.json"
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if "legacy_score_summary_0_100" not in report:
        report["legacy_score_summary_0_100"] = {
            "mean_valid_overall_score": report.get("mean_valid_overall_score"),
        }
    if "legacy_mean_valid_overall_score_0_100" not in report:
        report["legacy_mean_valid_overall_score_0_100"] = report[
            "legacy_score_summary_0_100"
        ].get("mean_valid_overall_score")
    report["score_version"] = SCORING_VERSION
    report["score_range"] = [0.0, 1.0]
    report["zero_score_policy"] = (
        "overall score is 0 only when measurement_valid is false; "
        "every measurement-valid result has overall score > 0"
    )
    report["mean_valid_overall_score"] = batch_scoring["overall"]["mean"]
    report["continuous_scoring"] = batch_scoring
    if not dry_run:
        atomic_json(report_path, report)
    return batch_scoring


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--batches", nargs="*", default=DEFAULT_BATCHES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    batch_results = {}
    for batch in args.batches:
        batch_results[batch] = rescore_batch(args.evaluation_root / batch, args.dry_run)
    manifest = {
        "scoring_version": SCORING_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "operation": "score-only migration; videos were not opened and extraction was not rerun",
        "batches": batch_results,
    }
    if not args.dry_run:
        atomic_json(args.evaluation_root / "continuous_rescore_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
