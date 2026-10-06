#!/usr/bin/env python3
"""Rescore frozen P8 measurements onto continuous-0-1-v1.

This script never decodes a video and never imports a tracking library.  It
reads the already-frozen result JSON metrics, preserves the legacy 0-100
scores and hard labels, writes continuous diagnostic dimensions, and rebuilds
the batch CSV/summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "results_v1"
DEFAULT_CONFIG = HERE / "continuous_score_config_v1.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def residual_quality(residual: float, scale: float) -> float:
    """q(r,s)=1/(1+r/s), with q(0,s)=1 and q(s,s)=0.5."""
    if not math.isfinite(float(residual)) or residual < 0 or scale <= 0:
        return 0.0
    return clip01(1.0 / (1.0 + float(residual) / float(scale)))


def evidence_quality(evidence: float, scale: float) -> float:
    """Continuous increasing counterpart: x/(x+s), equal to 0.5 at x=s."""
    if not math.isfinite(float(evidence)) or evidence < 0 or scale <= 0:
        return 0.0
    return clip01(float(evidence) / (float(evidence) + float(scale)))


def component_quality(value: float, spec: dict[str, Any]) -> float:
    kind = spec["kind"]
    scale = float(spec["scale"])
    if kind == "residual":
        return residual_quality(float(value), scale)
    if kind == "evidence":
        return evidence_quality(float(value), scale)
    if kind == "coverage_deficit":
        coverage = clip01(float(value))
        return residual_quality(1.0 - coverage, scale)
    raise ValueError(f"unknown component kind: {kind}")


def weighted_geometric_mean(
    values: dict[str, float], weights: dict[str, float], input_floor: float
) -> float:
    if not values or set(values) != set(weights):
        raise ValueError("values and weights must have identical non-empty keys")
    total_weight = sum(float(weight) for weight in weights.values())
    if total_weight <= 0:
        raise ValueError("weight sum must be positive")
    log_sum = 0.0
    for name, value in values.items():
        bounded = max(float(input_floor), clip01(float(value)))
        log_sum += float(weights[name]) * math.log(bounded)
    return clip01(math.exp(log_sum / total_weight))


def score_metrics(
    metrics: dict[str, Any], config: dict[str, Any], measurement_valid: bool
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    input_floor = float(config["input_floor"])
    dimensions: dict[str, float] = {}
    components_out: dict[str, dict[str, float]] = {}
    for dimension, dimension_spec in config["dimensions"].items():
        qualities: dict[str, float] = {}
        weights: dict[str, float] = {}
        for component, spec in dimension_spec["components"].items():
            metric_name = spec["metric"]
            raw = metrics.get(metric_name)
            quality = 0.0 if raw is None else component_quality(float(raw), spec)
            qualities[component] = quality
            weights[component] = float(spec["weight"])
        dimensions[dimension] = weighted_geometric_mean(qualities, weights, input_floor)
        components_out[dimension] = qualities
    overall = weighted_geometric_mean(
        dimensions,
        {name: float(weight) for name, weight in config["overall_weights"].items()},
        input_floor,
    )
    if measurement_valid:
        overall = max(float(config["valid_score_floor"]), overall)
    else:
        overall = 0.0
    dimensions["overall"] = clip01(overall)
    return dimensions, components_out


def preserve_legacy(result: dict[str, Any]) -> dict[str, Any]:
    existing = result.get("legacy_score")
    if existing:
        return existing
    old_scores = result.get("scores") or {}
    return {
        "score_version": "legacy-piecewise-0-100-v1.2.0",
        "scores": old_scores,
        "overall": old_scores.get("overall"),
    }


def rescore_result(result: dict[str, Any], config: dict[str, Any], config_hash: str) -> dict[str, Any]:
    legacy = preserve_legacy(result)
    legacy.setdefault("physics_pass_label", result.get("physics_pass"))
    legacy.setdefault("structural_ok_label", result.get("structural_ok"))
    legacy.setdefault("extract_success_label", result.get("extract_success"))
    measurement_valid = bool(result.get("extract_success") and result.get("structural_ok"))
    scores, components = score_metrics(result.get("metrics", {}), config, measurement_valid)
    result["legacy_score"] = legacy
    result["score_version"] = config["schema_version"]
    result["status"] = "valid_measurement" if measurement_valid else "invalid_measurement"
    result["measurement_valid"] = measurement_valid
    result["scores"] = scores
    result["score_components"] = components
    result["continuous_scoring"] = {
        "version": config["schema_version"],
        "config_sha256": config_hash,
        "formula": "q(r,s)=1/(1+r/s)",
        "aggregation": "weighted_geometric_mean_with_input_floor_0.01",
        "valid_score_floor": float(config["valid_score_floor"]),
        "uses_llm": False,
        "uses_vlm": False,
        "retracked": False,
    }
    return result


def distribution(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        alpha = position - lower
        return (1.0 - alpha) * ordered[lower] + alpha * ordered[upper]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p11": quantile(0.10),
        "p26": quantile(0.25),
        "median": quantile(0.50),
        "mean": statistics.fmean(ordered),
        "p75": quantile(0.75),
        "p90": quantile(0.90),
        "max": ordered[-1],
    }


def build_outputs(
    root: Path,
    results: list[dict[str, Any]],
    score_version: str,
    evaluator_process_failures: list[Any] | None = None,
) -> None:
    rows = []
    for result in sorted(results, key=lambda item: item["sample_id"]):
        scores = result["scores"]
        metrics = result.get("metrics", {})
        legacy = result["legacy_score"]["scores"]
        rows.append(
            {
                "task_id": result["task_id"],
                "sample_id": result["sample_id"],
                "source_id": result.get("source_id"),
                "score_version": score_version,
                "status": result["status"],
                "extract_success": result.get("extract_success"),
                "structural_ok": result.get("structural_ok"),
                "measurement_valid": result["measurement_valid"],
                "physics_pass": result.get("physics_pass"),
                "overall": scores["overall"],
                "overall_score": scores["overall"],
                "task_structure": scores["task_structure"],
                "score_task_structure": scores["task_structure"],
                "center_contact_geometry": scores["center_contact_geometry"],
                "score_center_contact_geometry": scores["center_contact_geometry"],
                "marker_rotation_quality": scores["marker_rotation_quality"],
                "score_marker_rotation_quality": scores["marker_rotation_quality"],
                "pure_rolling_ratio": scores["pure_rolling_ratio"],
                "score_pure_rolling_ratio": scores["pure_rolling_ratio"],
                "contact_point_velocity": scores["contact_point_velocity"],
                "score_contact_point_velocity": scores["contact_point_velocity"],
                "legacy_overall_0_100": legacy.get("overall"),
                "legacy_physics_pass_label": result.get("physics_pass"),
                "M1_rolling_ratio_abs": metrics.get("M1_rolling_ratio_abs"),
                "M2_contact_velocity_nmae": metrics.get("M2_contact_velocity_nmae"),
                "failure_reason": json.dumps(result.get("failure_reason"), ensure_ascii=False),
            }
        )
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        family = "GPT-enhanced first frame" if "_gpt_" in result["sample_id"] else "simulation first frame"
        by_family[family].append(result)
        by_family[result.get("source_id") or "unknown_source"].append(result)

    def group_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        all_scores = [item["scores"]["overall"] for item in items]
        valid_scores = [item["scores"]["overall"] for item in items if item["measurement_valid"]]
        return {
            "count": len(items),
            "measurement_valid_count": sum(item["measurement_valid"] for item in items),
            "physics_pass_label_count": sum(bool(item.get("physics_pass")) for item in items),
            "overall_all": distribution(all_scores),
            "overall_measurement_valid_only": distribution(valid_scores),
        }

    summary = {
        "task_id": "P8",
        "score_version": score_version,
        "formal_video_count": len(results),
        "result_count": len(results),
        "extract_success_count": sum(bool(item.get("extract_success")) for item in results),
        "structural_ok_count": sum(bool(item.get("structural_ok")) for item in results),
        "measurement_valid_count": sum(item["measurement_valid"] for item in results),
        "measurement_invalid_count": sum(not item["measurement_valid"] for item in results),
        "physics_pass_label_count": sum(bool(item.get("physics_pass")) for item in results),
        "physics_pass_count": sum(bool(item.get("physics_pass")) for item in results),
        "status_counts": {
            "valid_measurement": sum(item["status"] == "valid_measurement" for item in results),
            "invalid_measurement": sum(item["status"] == "invalid_measurement" for item in results),
        },
        "overall_all": distribution([item["scores"]["overall"] for item in results]),
        "overall_measurement_valid_only": distribution(
            [item["scores"]["overall"] for item in results if item["measurement_valid"]]
        ),
        "legacy_overall_0_100": distribution(
            [float(item["legacy_score"]["scores"]["overall"]) for item in results]
        ),
        "by_group": {name: group_summary(items) for name, items in sorted(by_family.items())},
        "physics_pass_labels": [item["sample_id"] for item in results if item.get("physics_pass")],
        "invalid_measurements": [item["sample_id"] for item in results if not item["measurement_valid"]],
        "tracking_rerun": False,
        "evaluator_process_failures": evaluator_process_failures or [],
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (root / "continuous_score_distribution.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.results_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_hash = sha256(config_path)
    result_paths = sorted(root.glob("P8_*_seed*/result.json"))
    if len(result_paths) != 24:
        raise SystemExit(f"expected 24 frozen result JSON files, found {len(result_paths)}")

    immutable_artifacts = {}
    results = []
    for result_path in result_paths:
        sample_dir = result_path.parent
        immutable_artifacts[result_path.parent.name] = {
            name: sha256(sample_dir / name)
            for name in ("frame_measurements.csv", "overlay.mp4", "plot.png")
        }
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result = rescore_result(result, config, config_hash)
        results.append(result)
        if not args.dry_run:
            result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not args.dry_run:
        build_outputs(root, results, config["schema_version"])
        immutable_artifacts_after = {
            sample_id: {
                name: sha256(root / sample_id / name)
                for name in ("frame_measurements.csv", "overlay.mp4", "plot.png")
            }
            for sample_id in immutable_artifacts
        }
        unchanged_count = sum(
            immutable_artifacts[sample_id][name] == immutable_artifacts_after[sample_id][name]
            for sample_id in immutable_artifacts
            for name in immutable_artifacts[sample_id]
        )
        manifest = {
            "schema_version": "p8-rescore-manifest-v1",
            "score_version": config["schema_version"],
            "sample_count": len(results),
            "tracking_rerun": False,
            "rescored_from_frozen_metrics_only": True,
            "continuous_config_sha256": config_hash,
            "immutable_tracking_artifact_count": 3 * len(immutable_artifacts),
            "immutable_tracking_artifact_unchanged_count": unchanged_count,
            "immutable_tracking_artifacts_verified_unchanged": unchanged_count == 3 * len(immutable_artifacts),
            "immutable_tracking_artifact_sha256_before": immutable_artifacts,
            "immutable_tracking_artifact_sha256_after": immutable_artifacts_after,
        }
        (root / "rescore_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "score_version": config["schema_version"],
                "sample_count": len(results),
                "measurement_valid": sum(item["measurement_valid"] for item in results),
                "physics_pass_labels": sum(bool(item.get("physics_pass")) for item in results),
                "tracking_rerun": False,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
