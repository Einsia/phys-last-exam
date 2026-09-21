#!/usr/bin/env python3
"""Create a non-destructive P3 continuous-0-1-v2-geometric result bundle."""

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
    """Half-quality-at-tolerance mapping requested for v2."""
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
        value = max(floor, clamp01(values.get(key)))
        pairs.append((value, weight))
    if not pairs:
        raise ValueError("weighted geometric mean has no positively weighted components")
    total = sum(weight for _, weight in pairs)
    return max(floor, min(1.0, math.exp(sum(weight * math.log(value) for value, weight in pairs) / total)))


def get(mapping: Mapping[str, Any], *path: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


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

    floor = float(config["epsilon_floor"])
    tolerances = config["tolerances"]
    component_weights = config["component_weights"]
    metrics = source.get("metrics") if isinstance(source.get("metrics"), Mapping) else {}
    measurements = source.get("measurements") if isinstance(source.get("measurements"), Mapping) else {}
    qc = source.get("qc") if isinstance(source.get("qc"), Mapping) else {}

    integrity_components: dict[str, float] = {}
    for slot in ("upper", "lower"):
        radius_cv = get(measurements, "balls", slot, "radius_robust_cv")
        disagreement = get(qc, "balls", slot, "backend_disagreement_median_radii")
        coverage = finite(get(qc, "balls", slot, "fused_coverage"))
        integrity_components[f"{slot}.radius_cv"] = residual_quality(radius_cv, tolerances["radius_cv"])
        integrity_components[f"{slot}.backend_disagreement"] = residual_quality(disagreement, tolerances["backend_disagreement_radii"])
        integrity_components[f"{slot}.coverage_error"] = residual_quality(
            None if coverage is None else 1.0 - coverage, tolerances["coverage_error"]
        )
    integrity = weighted_geometric(
        integrity_components, {key: 1.0 for key in integrity_components}, floor
    )

    task_components = {
        "release_sync": residual_quality(metrics.get("release_sync_frames"), tolerances["sync_frames"]),
        "launch_angle": residual_quality(metrics.get("max_angle_error_deg"), tolerances["angle_error_deg"]),
        "landing_level": residual_quality(metrics.get("max_landing_level_error_d"), tolerances["landing_level_d"]),
        "camera_translation": residual_quality(get(qc, "camera", "max_translation_diag"), tolerances["camera_translation_diag"]),
        "track_integrity": integrity,
    }
    task_score = weighted_geometric(task_components, component_weights["task"], floor)

    per_ball_metrics = get(metrics, "M2", "per_ball")
    if not isinstance(per_ball_metrics, Mapping):
        per_ball_metrics = {}
    per_ball: dict[str, Any] = {}
    dimension_ball_scores: dict[str, list[float]] = {"parabola": [], "horizontal": [], "vertical": []}
    for slot in ("upper", "lower"):
        ball = per_ball_metrics.get(slot) if isinstance(per_ball_metrics.get(slot), Mapping) else {}
        components = {
            "parabola": {
                "rmse": residual_quality(ball.get("parabola_rmse_d"), tolerances["parabola_rmse_d"]),
                "p95": residual_quality(ball.get("parabola_p95_d"), tolerances["parabola_p95_d"]),
            },
            "horizontal": {
                "x_fit_rmse": residual_quality(ball.get("x_fit_rmse_d"), tolerances["x_fit_rmse_d"]),
                "vx_robust_cv": residual_quality(ball.get("vx_robust_cv"), tolerances["vx_robust_cv"]),
            },
            "vertical": {
                "y_fit_rmse": residual_quality(ball.get("y_fit_rmse_d"), tolerances["y_fit_rmse_d"]),
                "vy_line_residual": residual_quality(ball.get("vy_line_residual"), tolerances["vy_line_residual"]),
                "ay_robust_cv": residual_quality(ball.get("ay_robust_cv"), tolerances["ay_robust_cv"]),
            },
        }
        scores = {
            "parabola": weighted_geometric(components["parabola"], component_weights["parabola_per_ball"], floor),
            "horizontal": weighted_geometric(components["horizontal"], component_weights["horizontal_per_ball"], floor),
            "vertical": weighted_geometric(components["vertical"], component_weights["vertical_per_ball"], floor),
        }
        for key, value in scores.items():
            dimension_ball_scores[key].append(value)
        per_ball[slot] = {"components": components, "scores": scores}

    parabola = weighted_geometric(
        {"upper": dimension_ball_scores["parabola"][0], "lower": dimension_ball_scores["parabola"][1]},
        {"upper": 0.5, "lower": 0.5}, floor,
    )
    horizontal = weighted_geometric(
        {"upper": dimension_ball_scores["horizontal"][0], "lower": dimension_ball_scores["horizontal"][1]},
        {"upper": 0.5, "lower": 0.5}, floor,
    )
    vertical = weighted_geometric(
        {"upper": dimension_ball_scores["vertical"][0], "lower": dimension_ball_scores["vertical"][1]},
        {"upper": 0.5, "lower": 0.5}, floor,
    )

    pair_components = {
        "equal_range": residual_quality(metrics.get("M1_abs"), tolerances["range_ratio_error"]),
        "equal_initial_speed": residual_quality(metrics.get("initial_speed_error"), tolerances["initial_speed_error"]),
        "gravity_symmetry": residual_quality(metrics.get("gravity_symmetric_error"), tolerances["gravity_error"]),
    }
    pair_consistency = weighted_geometric(pair_components, component_weights["pair_consistency"], floor)
    dimensions = {
        "task": task_score,
        "parabola": parabola,
        "horizontal": horizontal,
        "vertical": vertical,
        "pair_consistency": pair_consistency,
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
    record["score_profile"] = VERSION
    record["scores"] = scores
    record["overall_score"] = overall
    record["score_aggregation"] = {
        "aggregation_method": "weighted_geometric_mean",
        "weights": config["dimension_weights"],
        "epsilon_floor": floor,
    }
    record["score_details"] = {
        "profile": VERSION,
        "residual_quality_formula": config["residual_mapping"],
        "tolerances": tolerances,
        "aggregation_method": "weighted_geometric_mean",
        "weights": config["dimension_weights"],
        "epsilon_floor": floor,
        "measurement_valid_gate": config["invalid_policy"],
        "task_components": task_components,
        "track_integrity_components": integrity_components,
        "per_ball": per_ball,
        "pair_components": pair_components,
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
    paths = sorted(args.input.glob("P3_*_seed*.json"))
    if len(paths) != 24:
        raise SystemExit(f"expected 24 P3 source records, found {len(paths)}")
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
        "task_id": "P3",
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

    checks = {
        "expected_count": 24,
        "actual_count": len(records),
        "json_count": len(list(output_json.glob("P3_*_seed*.json"))),
        "all_scores_finite": all(finite(value) is not None for record in records for value in record["scores"].values()),
        "all_scores_in_0_1": all(0.0 <= float(value) <= 1.0 for record in records for value in record["scores"].values()),
        "all_snapshots_present": all(record.get("score_v1_snapshot", {}).get("source_json") for record in records),
        "all_debug_paths_preserved": all(record.get("debug") == json.loads(path.read_text(encoding="utf-8")).get("debug") for record, path in zip(records, paths)),
        "invalid_overall_zero": all(bool(record.get("measurement_valid", record.get("extract_success"))) or record["scores"]["overall"] == 0.0 for record in records),
        "nan_count": sum(finite(value) is None for record in records for value in record["scores"].values()),
    }
    checks["passed"] = all(value is True or (key in {"expected_count", "actual_count", "json_count", "nan_count"}) for key, value in checks.items()) and checks["actual_count"] == checks["expected_count"] == checks["json_count"] and checks["nan_count"] == 0
    (args.output / "validation_report.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "task_id": "P3",
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
