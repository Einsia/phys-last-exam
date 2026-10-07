#!/usr/bin/env python3
"""Continuous 0-1 scoring for P8 measurements."""

from __future__ import annotations

import math
from typing import Any


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
