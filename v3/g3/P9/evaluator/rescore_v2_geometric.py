#!/usr/bin/env python3
"""Create versioned P9 continuous-0-1-v2-geometric results.

Only frozen JSON measurements are read.  The source result directory and its
debug artifacts remain untouched; no video decoding or learned model is used.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parent / "evaluation" / "minimax_h3_prompt_v1_20260825"
DEFAULT_OUTPUT = HERE / "results_v2_geometric"
DEFAULT_CONFIG = HERE / "continuous_score_config_v2_geometric.json"


def clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def residual_quality(value: Any, tolerance: float) -> float:
    error = finite_number(value)
    if error is None or tolerance <= 0:
        return 0.0
    ratio = max(0.0, error) / tolerance
    return clip01(1.0 / (1.0 + ratio * ratio))


def weighted_geometric(values: Mapping[str, float], weights: Mapping[str, float], floor: float) -> float:
    if not values or set(values) != set(weights):
        raise ValueError("values and weights must have identical non-empty keys")
    total = sum(float(weight) for weight in weights.values())
    if total <= 0:
        raise ValueError("weight sum must be positive")
    return clip01(math.exp(sum(
        float(weights[name]) / total * math.log(max(floor, clip01(value)))
        for name, value in values.items()
    )))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    data = sorted(float(value) for value in values)
    if not data:
        return {"count": 0}

    def quantile(fraction: float) -> float:
        position = fraction * (len(data) - 1)
        lo, hi = math.floor(position), math.ceil(position)
        if lo == hi:
            return data[lo]
        alpha = position - lo
        return data[lo] * (1 - alpha) + data[hi] * alpha

    return {
        "count": len(data),
        "min": data[0],
        "p10": quantile(0.10),
        "p25": quantile(0.25),
        "median": quantile(0.50),
        "mean": statistics.fmean(data),
        "stdev_population": statistics.pstdev(data),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "max": data[-1],
    }


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        for position in range(index, end):
            output[order[position]] = rank
        index = end
    return output


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_l, mean_r = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - mean_l) * (b - mean_r) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - mean_l) ** 2 for a in left) * sum((b - mean_r) ** 2 for b in right))
    return numerator / denominator if denominator > 0 else None


def derived_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(record.get("metrics") or {})
    pendulums = record.get("pendulums") or {}
    short, long = pendulums.get("short") or {}, pendulums.get("long") or {}
    short_period, long_period = short.get("period") or {}, long.get("period") or {}
    length_ratio = finite_number(metrics.get("initial_length_ratio_short_over_long"))
    metrics["initial_length_ratio_error"] = None if length_ratio is None else abs(length_ratio - 0.5)
    coverages = [finite_number(short.get("coverage")), finite_number(long.get("coverage"))]
    metrics["minimum_tracking_coverage"] = min(coverages) if all(value is not None for value in coverages) else None
    fit_values = [finite_number(short_period.get("fit_r2")), finite_number(long_period.get("fit_r2"))]
    metrics["minimum_sinusoid_fit_r2"] = min(fit_values) if all(value is not None for value in fit_values) else None
    cycle_values = [finite_number(short_period.get("observed_cycles")), finite_number(long_period.get("observed_cycles"))]
    metrics["minimum_observed_cycles"] = min(cycle_values) if all(value is not None for value in cycle_values) else None
    return metrics


def component_quality(value: Any, spec: Mapping[str, Any]) -> float:
    kind = str(spec["kind"])
    if kind == "residual":
        return residual_quality(value, float(spec["scale"]))
    if kind == "high_target":
        number = finite_number(value)
        shortfall = None if number is None else max(0.0, float(spec["target"]) - number)
        return residual_quality(shortfall, float(spec["scale"]))
    if kind == "direct_fraction":
        number = finite_number(value)
        return 0.0 if number is None else clip01(number)
    if kind == "evidence_fraction":
        number = finite_number(value)
        return 0.0 if number is None else clip01(number / float(spec["required"]))
    raise ValueError(f"unknown component kind: {kind}")


def score_dimensions(record: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, Any]]:
    metrics = derived_metrics(record)
    floor = float(config["input_floor"])
    dimensions: dict[str, float] = {}
    component_output: dict[str, dict[str, float]] = {}
    for dimension, dimension_spec in config["dimensions"].items():
        components = {
            name: component_quality(metrics.get(spec["metric"]), spec)
            for name, spec in dimension_spec["components"].items()
        }
        weights = {
            name: float(spec["weight"])
            for name, spec in dimension_spec["components"].items()
        }
        dimensions[dimension] = weighted_geometric(components, weights, floor)
        component_output[dimension] = components
    return dimensions, component_output, metrics


def validate_scores(values: Iterable[float]) -> None:
    for value in values:
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise AssertionError(f"score outside [0,1] or non-finite: {value}")


def rescore(source: dict[str, Any], config: Mapping[str, Any], config_hash: str, source_path: Path, source_root: Path) -> dict[str, Any]:
    result = copy.deepcopy(source)
    old_dimensions = {key: float(value) for key, value in (source.get("dimension_scores") or {}).items()}
    old_overall = float(source.get("overall_score", 0.0))
    result["score_v1_snapshot"] = {
        "score_version": "continuous-0-1-v1",
        "overall": old_overall,
        "dimensions": old_dimensions,
        "source_json": str(source_path.resolve()),
    }
    debug_dir = source_root.resolve() / "debug" / source_path.stem
    result["v2_artifact_paths"] = {
        "overlay": str(debug_dir / "overlay.mp4"),
        "plot": str(debug_dir / "diagnostics.png"),
    }

    status = source.get("status") or {}
    measurement_valid = bool(status.get("measurement_valid"))
    dimensions, components, derived = score_dimensions(source, config)
    overall_weights = {name: float(spec["weight"]) for name, spec in config["dimensions"].items()}
    overall = weighted_geometric(dimensions, overall_weights, float(config["input_floor"])) if measurement_valid else 0.0
    validate_scores([*dimensions.values(), overall, *[v for group in components.values() for v in group.values()]])

    result["score_version"] = config["schema_version"]
    result["score_floor_for_measurable"] = float(config["input_floor"])
    result["dimension_scores"] = dimensions
    result["overall_score"] = overall
    result["score_components"] = components
    result["score_derived_metrics"] = {
        key: derived[key]
        for key in ("initial_length_ratio_error", "minimum_tracking_coverage", "minimum_sinusoid_fit_r2", "minimum_observed_cycles")
    }
    result["score_aggregation"] = {
        "aggregation_method": "weighted_geometric_mean",
        "weights": overall_weights,
        "epsilon_floor": float(config["input_floor"]),
    }
    result["continuous_scoring"] = {
        "version": config["schema_version"],
        "config_sha256": config_hash,
        "residual_formula": config["residual_formula"],
        "aggregation_method": "weighted_geometric_mean",
        "weights": overall_weights,
        "epsilon_floor": float(config["input_floor"]),
        "zero_score_policy": config["zero_score_policy"],
        "uses_llm": False,
        "uses_vlm": False,
        "retracked": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    source_root, output_root, config_path = args.source_root.resolve(), args.output_root.resolve(), args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_hash = sha256(config_path)
    paths = sorted((source_root / "json").glob("P9_*_seed*.json"))
    if len(paths) != 24:
        raise SystemExit(f"expected 24 P9 source JSON files, found {len(paths)}")
    if output_root == source_root:
        raise SystemExit("output root must differ from source root")
    (output_root / "json").mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for path in paths:
        source_hashes[path.stem] = sha256(path)
        record = rescore(json.loads(path.read_text(encoding="utf-8")), config, config_hash, path, source_root)
        destination = output_root / "json" / path.name
        destination.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        records.append(record)
        dimensions = record["dimension_scores"]
        rows.append({
            "task_id": "P9",
            "result_id": path.stem,
            "sample_id": record.get("sample_id"),
            "score_version": record["score_version"],
            "measurement_valid": bool((record.get("status") or {}).get("measurement_valid")),
            "v1_overall": record["score_v1_snapshot"]["overall"],
            "v2_overall": record["overall_score"],
            "v2_minus_v1": record["overall_score"] - record["score_v1_snapshot"]["overall"],
            **{f"v2_{name}": dimensions[name] for name in config["dimensions"]},
            "m1_abs_residual": record.get("metrics", {}).get("m1_abs_residual"),
            "period_ratio_short_over_long": record.get("metrics", {}).get("period_ratio_short_over_long"),
            "initial_length_ratio_short_over_long": record.get("metrics", {}).get("initial_length_ratio_short_over_long"),
        })

    with (output_root / "results_v2_geometric.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "score_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    valid_flags = [bool((record.get("status") or {}).get("measurement_valid")) for record in records]
    v1 = [record["score_v1_snapshot"]["overall"] for record in records]
    v2 = [record["overall_score"] for record in records]
    valid_v1 = [value for value, valid in zip(v1, valid_flags) if valid]
    valid_v2 = [value for value, valid in zip(v2, valid_flags) if valid]
    overall_weights = {name: float(spec["weight"]) for name, spec in config["dimensions"].items()}
    comparison = {
        "task_id": "P9",
        "score_version": config["schema_version"],
        "source_root": str(source_root),
        "output_root": str(output_root),
        "sample_count": len(records),
        "measurement_valid_count": sum(valid_flags),
        "measurement_invalid_count": sum(not valid for valid in valid_flags),
        "v1_all": distribution(v1),
        "v2_all": distribution(v2),
        "v1_valid_only": distribution(valid_v1),
        "v2_valid_only": distribution(valid_v2),
        "pearson_v1_v2": correlation(valid_v1, valid_v2),
        "spearman_v1_v2": correlation(ranks(valid_v1), ranks(valid_v2)),
        "mean_delta_valid": statistics.fmean(b - a for a, b in zip(valid_v1, valid_v2)),
        "formula": config["residual_formula"],
        "aggregation_method": "weighted_geometric_mean",
        "weights": overall_weights,
        "epsilon_floor": config["input_floor"],
    }
    (output_root / "comparison_v1_v2.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")

    output_paths = sorted((output_root / "json").glob("P9_*_seed*.json"))
    rescored = [json.loads(path.read_text(encoding="utf-8")) for path in output_paths]
    all_numeric_scores = [
        float(value)
        for record in rescored
        for value in [record["overall_score"], *record["dimension_scores"].values(), *[v for group in record["score_components"].values() for v in group.values()]]
    ]
    validate_scores(all_numeric_scores)
    source_unchanged = all(sha256(path) == source_hashes[path.stem] for path in paths)
    manifest = {
        "score_version": config["schema_version"],
        "expected_count": 24,
        "json_count": len(output_paths),
        "csv_row_count": len(rows),
        "all_scores_finite": all(math.isfinite(value) for value in all_numeric_scores),
        "all_scores_in_0_1": all(0.0 <= value <= 1.0 for value in all_numeric_scores),
        "invalid_overall_zero": all(record["overall_score"] == 0.0 for record in rescored if not (record.get("status") or {}).get("measurement_valid")),
        "valid_overall_positive": all(record["overall_score"] > 0.0 for record in rescored if (record.get("status") or {}).get("measurement_valid")),
        "v1_snapshot_complete": all("score_v1_snapshot" in record for record in rescored),
        "source_json_sha256_unchanged": source_unchanged,
        "source_json_sha256": source_hashes,
        "config_sha256": config_hash,
        "tracking_rerun": False,
    }
    (output_root / "validation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not all([len(output_paths) == 24, len(rows) == 24, manifest["all_scores_finite"], manifest["all_scores_in_0_1"], manifest["invalid_overall_zero"], manifest["valid_overall_positive"], source_unchanged]):
        raise AssertionError(json.dumps(manifest, indent=2))
    print(json.dumps({"comparison": comparison, "validation": {k: v for k, v in manifest.items() if k != "source_json_sha256"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
