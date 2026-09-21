#!/usr/bin/env python3
"""Rescore frozen P9 measurements with continuous 0–1 quality functions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from continuous_scoring import (
    SCORE_FLOOR,
    SCORE_VERSION,
    assert_score_contract,
    evidence_fraction,
    high_quality,
    measurable_floor,
    residual_quality,
    weighted_geometric,
)


DIMENSION_WEIGHTS = {"setup": .15, "structural": .20, "periodicity": .25, "pendulum_law": .40}
SCALES = {
    "initial_length_ratio_error": .12,
    "release_time_difference_s": .50,
    "pivot_drift_fraction": .045,
    "string_length_change_fraction": .15,
    "pivot_separation_change_fraction": .035,
    "sinusoid_r2_shortfall": .30,
    "period_cv": .20,
    "m1_pendulum_law": .20,
}


def flatten(value: dict, prefix: str = "") -> dict:
    out: dict[str, object] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            out.update(flatten(item, name))
        elif isinstance(item, list):
            out[name] = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        else:
            out[name] = item
    return out


def stable_payload(record: dict) -> str:
    payload = {
        "status": record.get("status"),
        "metrics": record.get("metrics"),
        "pendulums": record.get("pendulums"),
        "camera_audit": record.get("camera_audit"),
        "errors": record.get("errors"),
        "warnings": record.get("warnings"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def score_record(record: dict) -> dict:
    before = stable_payload(record)
    if "legacy_dimension_scores_0_100" not in record:
        record["legacy_dimension_scores_0_100"] = record.get("dimension_scores", {})
    if "legacy_overall_score_0_100" not in record:
        record["legacy_overall_score_0_100"] = record.get("overall_score", 0.0)

    status = record.get("status", {})
    valid = bool(status.get("measurement_valid"))
    metrics = record.get("metrics", {})
    pendulums = record.get("pendulums", {})
    short = pendulums.get("short", {})
    long = pendulums.get("long", {})
    short_period = short.get("period") or {}
    long_period = long.get("period") or {}

    length_ratio = metrics.get("initial_length_ratio_short_over_long")
    length_error = abs(float(length_ratio) - .5) if length_ratio is not None else None
    coverages = [short.get("coverage"), long.get("coverage")]
    coverage = min(float(x) for x in coverages if x is not None) if all(x is not None for x in coverages) else 0.0
    fit_values = [short_period.get("fit_r2"), long_period.get("fit_r2")]
    fit_r2 = min(float(x) for x in fit_values if x is not None) if all(x is not None for x in fit_values) else None
    cycles = min(float(short_period.get("observed_cycles") or 0.0), float(long_period.get("observed_cycles") or 0.0))

    components = {
        "setup": {
            "length_ratio": residual_quality(length_error, SCALES["initial_length_ratio_error"]),
            "release_sync": residual_quality(metrics.get("release_time_difference_s"), SCALES["release_time_difference_s"]),
            "tracking_coverage": measurable_floor(coverage),
        },
        "structural": {
            "pivot_stability": residual_quality(metrics.get("pivot_drift_short_length_fraction_p95"), SCALES["pivot_drift_fraction"]),
            "string_length_stability": residual_quality(metrics.get("string_length_relative_change_p95_max"), SCALES["string_length_change_fraction"]),
            "pivot_separation_stability": residual_quality(metrics.get("pivot_separation_relative_change_p95"), SCALES["pivot_separation_change_fraction"]),
        },
        "periodicity": {
            "sinusoid_fit": high_quality(fit_r2, 1.0, SCALES["sinusoid_r2_shortfall"]),
            "period_consistency": residual_quality(metrics.get("maximum_period_cv"), SCALES["period_cv"]),
            "observed_cycles": evidence_fraction(cycles, 1.0),
        },
        "pendulum_law": {
            "period_length_relation": residual_quality(metrics.get("m1_abs_residual"), SCALES["m1_pendulum_law"]),
        },
    }
    component_weights = {
        "setup": {"length_ratio": .55, "release_sync": .25, "tracking_coverage": .20},
        "structural": {"pivot_stability": .35, "string_length_stability": .40, "pivot_separation_stability": .25},
        "periodicity": {"sinusoid_fit": .45, "period_consistency": .35, "observed_cycles": .20},
        "pendulum_law": {"period_length_relation": 1.0},
    }
    dimensions = {key: weighted_geometric(values, component_weights[key]) for key, values in components.items()}
    overall = weighted_geometric(dimensions, DIMENSION_WEIGHTS) if valid else 0.0
    if not valid:
        components = {key: {name: 0.0 for name in values} for key, values in components.items()}
        dimensions = {key: 0.0 for key in dimensions}
    assert_score_contract([*dimensions.values(), overall], valid, overall)

    record["score_version"] = SCORE_VERSION
    record["score_floor_for_measurable"] = SCORE_FLOOR
    record["score_scales"] = SCALES
    record["score_components"] = components
    record["dimension_scores"] = dimensions
    record["overall_score"] = overall
    if stable_payload(record) != before:
        raise AssertionError(f"rescoring changed metrics/status for {record.get('sample_id')}")
    return record


def self_test() -> None:
    assert residual_quality(0.0, .2) == 1.0
    assert residual_quality(.1, .2) > residual_quality(.2, .2) > residual_quality(.5, .2) > 0
    assert abs(residual_quality(.2 - 1e-9, .2) - residual_quality(.2 + 1e-9, .2)) < 1e-7
    assert_score_contract([.01, .5], True, .2)
    assert_score_contract([0.0], False, 0.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    self_test()
    paths = sorted((args.results / "json").glob("P9_*_seed*.json"))
    if len(paths) != 24:
        raise SystemExit(f"expected 24 P9 records, found {len(paths)}")
    records = []
    for path in paths:
        record = score_record(json.loads(path.read_text(encoding="utf-8")))
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)

    rows = [flatten(record) for record in records]
    fields = sorted({key for row in rows for key in row})
    with (args.results / "results_continuous.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

    values = [record["overall_score"] for record in records]
    valid_values = [record["overall_score"] for record in records if record.get("status", {}).get("measurement_valid")]
    report_path = args.results / "batch_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    legacy_mean = report.get("legacy_overall_mean_0_100")
    if legacy_mean is None:
        legacy_mean = report.get("overall_mean")
    current_mean = statistics.fmean(values)
    report.update({
        "score_version": SCORE_VERSION,
        "score_range": [0.0, 1.0],
        "zero_score_policy": "measurement_invalid_only",
        "overall_mean": current_mean,
        "legacy_overall_mean_0_100": legacy_mean,
        "continuous_score_mean_all": current_mean,
        "continuous_score_mean_valid": statistics.fmean(valid_values) if valid_values else 0.0,
        "continuous_score_median_valid": statistics.median(valid_values) if valid_values else 0.0,
        "continuous_score_min_valid": min(valid_values) if valid_values else 0.0,
        "continuous_score_max_valid": max(valid_values) if valid_values else 0.0,
        "continuous_score_zero_count": sum(value == 0 for value in values),
        "status_and_measurements_preserved": True,
    })
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key.startswith("continuous_") or key in {"score_version", "continuous_score_zero_count"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
