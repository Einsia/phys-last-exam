#!/usr/bin/env python3
"""Create a non-destructive P4 continuous-0-1-v2-geometric result bundle."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

VERSION = "continuous-0-1-v2-geometric"


def finite(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def clamp01(value: Any) -> float:
    value = finite(value)
    return 0.0 if value is None else max(0.0, min(1.0, value))


def residual_quality(error: Any, tolerance: float) -> float:
    value = finite(error)
    if value is None or tolerance <= 0 or not math.isfinite(tolerance):
        return 0.0
    ratio = abs(value) / tolerance
    return 1.0 / (1.0 + ratio * ratio)


def weighted_geometric(values: Mapping[str, Any], weights: Mapping[str, float], floor: float) -> float:
    pairs: list[tuple[float, float]] = []
    for key, weight in weights.items():
        weight = float(weight)
        if weight <= 0:
            continue
        pairs.append((max(floor, clamp01(values.get(key))), weight))
    if not pairs:
        raise ValueError("weighted geometric mean has no positively weighted components")
    total = sum(weight for _, weight in pairs)
    return max(floor, min(1.0, math.exp(sum(weight * math.log(value) for value, weight in pairs) / total)))


def flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            output.update(flatten(item, name))
        elif isinstance(item, list):
            output[name] = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        else:
            output[name] = item
    return output


def score_record(source: Mapping[str, Any], config: Mapping[str, Any], source_path: Path) -> dict[str, Any]:
    record = copy.deepcopy(dict(source))
    old_scores = source.get("scores")
    if not isinstance(old_scores, Mapping) or finite(old_scores.get("overall")) is None:
        raise ValueError(f"{source_path.name}: missing v1 scores")
    record["score_v1_snapshot"] = {
        "score_version": str(source.get("score_version") or "continuous-0-1-v1"),
        "overall": float(old_scores["overall"]),
        "dimensions": {key: float(value) for key, value in old_scores.items() if key != "overall"},
        "source_json": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }

    sample_id = str(source.get("sample_id") or source_path.stem)
    original_results = source_path.parent.parent.resolve()
    debug_dir = original_results / "debug" / sample_id
    record["v2_artifact_paths"] = {
        "debug_dir": str(debug_dir),
        "overlay": str(debug_dir / "overlay.mp4"),
        "plot": str(debug_dir / "trajectory_plot.png"),
        "extra": str(debug_dir / "keyframes.jpg"),
        "track_csv": str(debug_dir / "fused_track.csv"),
    }

    floor = float(config["epsilon_floor"])
    tolerance = config["tolerances"]
    component_weights = config["component_weights"]
    metrics = source.get("metrics") if isinstance(source.get("metrics"), Mapping) else {}
    measurements = source.get("measurements") if isinstance(source.get("measurements"), Mapping) else {}
    qc = source.get("qc") if isinstance(source.get("qc"), Mapping) else {}
    tracking = qc.get("tracking") if isinstance(qc.get("tracking"), Mapping) else {}

    arcs = finite(measurements.get("complete_rebound_arc_count")) or 0.0
    required = finite(measurements.get("required_rebound_arc_count")) or 4.0
    arc_fraction = clamp01(arcs / required) if required > 0 else 0.0
    components = {
        "structure": {
            "complete_arc_fraction": arc_fraction,
            "track_coverage": clamp01(tracking.get("fused_coverage")),
            "track_confidence": clamp01(tracking.get("confident_coverage")),
        },
        "monotonic_energy_loss": {
            "height_error": residual_quality(metrics.get("M2_monotonic_height_error"), tolerance["height_monotonic_error"]),
            "near_nondecreasing_fraction": residual_quality(metrics.get("M2_near_nondecreasing_height_fraction"), tolerance["near_nondecreasing_fraction"]),
            "time_error": residual_quality(metrics.get("M2_monotonic_time_excess"), tolerance["time_monotonic_excess"]),
        },
        "ballistic_arc_quality": {
            "arc_shape": residual_quality(metrics.get("arc_shape_normalized_rmse"), tolerance["arc_shape_rmse"]),
            "apex_symmetry": residual_quality(metrics.get("apex_time_asymmetry"), tolerance["apex_time_asymmetry"]),
        },
        "restitution_consistency": {
            "height_time_agreement": residual_quality(metrics.get("M1_height_time_restitution_consistency"), tolerance["m1_height_time_consistency"]),
            "coefficient_stability": residual_quality(metrics.get("M2_restitution_coefficient_cv"), tolerance["restitution_cv"]),
        },
        "spatial_stability": {
            "impact_position": residual_quality(metrics.get("impact_x_robust_spread_radii"), tolerance["impact_x_spread_radii"]),
            "arc_horizontal_drift": residual_quality(metrics.get("mean_arc_horizontal_drift_radii"), tolerance["arc_x_drift_radii"]),
            "camera": residual_quality(metrics.get("camera_drift_fraction_of_diagonal"), tolerance["camera_drift_fraction"]),
            "backend_agreement": residual_quality(metrics.get("best_backend_pair_disagreement_radii"), tolerance["backend_disagreement_radii"]),
        },
    }
    dimensions = {
        name: weighted_geometric(values, component_weights[name], floor)
        for name, values in components.items()
    }
    valid = bool(source.get("measurement_valid", source.get("extract_success")))
    overall_candidate = weighted_geometric(dimensions, config["dimension_weights"], floor)
    overall = overall_candidate if valid else 0.0
    scores = {**dimensions, "overall": overall}
    if any(finite(value) is None or not 0.0 <= float(value) <= 1.0 for value in scores.values()):
        raise AssertionError(f"{source_path.name}: invalid score range")
    if not valid and overall != 0.0:
        raise AssertionError(f"{source_path.name}: invalid measurement must have overall=0")

    record["score_version"] = VERSION
    record["scores"] = scores
    record["overall_score"] = overall
    record["score_floor_for_measurable"] = floor
    record["score_scales"] = tolerance
    record["score_components"] = components
    record["score_aggregation"] = {
        "aggregation_method": "weighted_geometric_mean",
        "weights": config["dimension_weights"],
        "epsilon_floor": floor,
        "overall_candidate_before_measurement_gate": overall_candidate,
    }
    return record


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    output = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for index in order[cursor:end]:
            output[index] = rank
        cursor = end
    return output


def correlation(left: list[float], right: list[float]) -> float:
    lm, rm = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((x - lm) * (y - rm) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - lm) ** 2 for x in left) * sum((y - rm) ** 2 for y in right))
    return numerator / denominator if denominator else 0.0


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "population_std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("score_version") != VERSION:
        raise SystemExit("configuration score_version mismatch")
    paths = sorted(args.input.glob("P4_*_seed*.json"))
    if len(paths) != 24:
        raise SystemExit(f"expected 24 P4 source records, found {len(paths)}")
    output_json = args.output / "json"
    output_json.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for path in paths:
        source = json.loads(path.read_text(encoding="utf-8"))
        record = score_record(source, config, path)
        (output_json / path.name).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)

    rows = [flatten(record) for record in records]
    fields = sorted({key for row in rows for key in row})
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    v1 = [float(record["score_v1_snapshot"]["overall"]) for record in records]
    v2 = [float(record["scores"]["overall"]) for record in records]
    names = [path.stem for path in paths]
    rank_v1, rank_v2 = ranks(v1), ranks(v2)
    shifts = [rank_v2[i] - rank_v1[i] for i in range(len(records))]
    dimension_names = [key for key in records[0]["scores"] if key != "overall"]
    comparison = {
        "task_id": "P4",
        "score_version": VERSION,
        "source_score_version": "continuous-0-1-v1",
        "count": len(records),
        "v1": distribution(v1),
        "v2": distribution(v2),
        "mean_delta_v2_minus_v1": statistics.fmean(v2) - statistics.fmean(v1),
        "std_ratio_v2_over_v1": statistics.pstdev(v2) / statistics.pstdev(v1) if statistics.pstdev(v1) else None,
        "pearson_v1_v2": correlation(v1, v2),
        "spearman_v1_v2": correlation(rank_v1, rank_v2),
        "dimension_means_v1": {key: statistics.fmean(float(r["score_v1_snapshot"]["dimensions"][key]) for r in records) for key in dimension_names},
        "dimension_means_v2": {key: statistics.fmean(float(r["scores"][key]) for r in records) for key in dimension_names},
        "largest_absolute_rank_shifts": sorted(
            ({"sample_id": names[i], "v1": v1[i], "v2": v2[i], "rank_shift": shifts[i]} for i in range(len(records))),
            key=lambda row: abs(row["rank_shift"]), reverse=True,
        )[:10],
        "top5_v2": [{"sample_id": names[i], "score": v2[i]} for i in sorted(range(len(v2)), key=lambda i: v2[i], reverse=True)[:5]],
        "bottom5_v2": [{"sample_id": names[i], "score": v2[i]} for i in sorted(range(len(v2)), key=lambda i: v2[i])[:5]],
    }
    (args.output / "comparison_stats.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "scoring_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    artifact_paths_exist = all(
        all(Path(record["v2_artifact_paths"][key]).exists() for key in ("overlay", "plot", "extra", "track_csv"))
        for record in records
    )
    checks = {
        "expected_count": 24,
        "actual_count": len(records),
        "json_count": len(list(output_json.glob("P4_*_seed*.json"))),
        "all_scores_finite": all(finite(value) is not None for record in records for value in record["scores"].values()),
        "all_scores_in_0_1": all(0.0 <= float(value) <= 1.0 for record in records for value in record["scores"].values()),
        "all_snapshots_present": all(record.get("score_v1_snapshot", {}).get("source_json") for record in records),
        "all_artifact_paths_exist": artifact_paths_exist,
        "invalid_overall_zero": all(bool(record.get("measurement_valid", record.get("extract_success"))) or record["scores"]["overall"] == 0.0 for record in records),
        "nan_count": sum(finite(value) is None for record in records for value in record["scores"].values()),
    }
    checks["passed"] = all(value is True or key in {"expected_count", "actual_count", "json_count", "nan_count"} for key, value in checks.items()) and checks["actual_count"] == checks["expected_count"] == checks["json_count"] and checks["nan_count"] == 0
    (args.output / "validation_report.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "task_id": "P4",
        "score_version": VERSION,
        "score_range": [0.0, 1.0],
        "record_count": len(records),
        "measurement_valid_count": sum(bool(record.get("measurement_valid", record.get("extract_success"))) for record in records),
        "overall": comparison["v2"],
        "validation_passed": checks["passed"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "summary": summary, "comparison": {k: comparison[k] for k in ("v1", "v2", "mean_delta_v2_minus_v1", "std_ratio_v2_over_v1", "spearman_v1_v2")}}, indent=2))
    return 0 if checks["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
