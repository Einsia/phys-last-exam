#!/usr/bin/env python3
"""Continuous 0-1 scoring for P3 measurements."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

PROFILE = "continuous-0-1-v1"
SCORE_FLOOR = 0.01


def residual_quality(residual: Any, anchor: float) -> float:
    """Return q(r,s)=1/(1+|r|/s), or zero for unusable input."""
    try:
        value = abs(float(residual))
        scale = float(anchor)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or not math.isfinite(scale) or scale <= 0.0:
        return 0.0
    return float(1.0 / (1.0 + value / scale))


def weighted_geometric_mean(
    values: Sequence[Any], weights: Sequence[float], floor: float = SCORE_FLOOR
) -> float:
    """Weighted GM with every input clamped to [floor, 1]."""
    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must be non-empty and equal length")
    clean_weights = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) or weight < 0.0 for weight in clean_weights):
        raise ValueError("weights must be finite and non-negative")
    weight_sum = sum(clean_weights)
    if weight_sum <= 0.0:
        raise ValueError("at least one weight must be positive")
    clean_values: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if not math.isfinite(numeric):
            numeric = 0.0
        clean_values.append(max(floor, min(numeric, 1.0)))
    log_mean = sum(
        weight * math.log(value)
        for value, weight in zip(clean_values, clean_weights)
    ) / weight_sum
    return float(max(floor, min(math.exp(log_mean), 1.0)))


def _get(mapping: Mapping[str, Any], *path: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _bad_anchor(config: Mapping[str, Any], key: str) -> float:
    value = _get(config, "thresholds", key, "bad")
    anchor = float(value)
    if not math.isfinite(anchor) or anchor <= 0.0:
        raise ValueError(f"invalid bad/half-quality anchor for {key}: {value!r}")
    return anchor


def _metric_quality(value: Any, config: Mapping[str, Any], threshold_key: str) -> float:
    return residual_quality(value, _bad_anchor(config, threshold_key))


def score_continuous(
    result: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute all five dimensions and the overall continuous score."""
    metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
    qc = result.get("qc") if isinstance(result.get("qc"), Mapping) else {}
    measurements = (
        result.get("measurements")
        if isinstance(result.get("measurements"), Mapping)
        else {}
    )

    integrity_values: list[float] = []
    integrity_labels: list[str] = []
    per_ball_metrics = _get(metrics, "M2", "per_ball") or {}
    ball_measurements = measurements.get("balls") or {}
    qc_balls = qc.get("balls") or {}
    for slot in ("upper", "lower"):
        radius_cv = _get(ball_measurements, slot, "radius_robust_cv")
        disagreement = _get(qc_balls, slot, "backend_disagreement_median_radii")
        coverage = _get(qc_balls, slot, "fused_coverage")
        try:
            coverage_error = abs(1.0 - float(coverage))
        except (TypeError, ValueError):
            coverage_error = None
        integrity_values.extend(
            [
                residual_quality(radius_cv, 0.45),
                residual_quality(disagreement, 0.75),
                residual_quality(coverage_error, 0.15),
            ]
        )
        integrity_labels.extend(
            [f"{slot}.radius_cv", f"{slot}.backend_disagreement", f"{slot}.coverage_error"]
        )
    integrity = weighted_geometric_mean(integrity_values, [1.0] * len(integrity_values))

    task_components = {
        "release_sync": _metric_quality(metrics.get("release_sync_frames"), config, "sync_frames"),
        "launch_angle": _metric_quality(metrics.get("max_angle_error_deg"), config, "angle_error_deg"),
        "landing_level": _metric_quality(
            metrics.get("max_landing_level_error_d"), config, "landing_level_d"
        ),
        "camera_translation": _metric_quality(
            _get(qc, "camera", "max_translation_diag"), config, "camera_translation_diag"
        ),
        "track_integrity": integrity,
    }
    task = weighted_geometric_mean(
        list(task_components.values()), [0.25, 0.25, 0.20, 0.15, 0.15]
    )

    per_ball_details: dict[str, Any] = {}
    parabola_balls: list[float] = []
    horizontal_balls: list[float] = []
    vertical_balls: list[float] = []
    for slot in ("upper", "lower"):
        values = per_ball_metrics.get(slot) if isinstance(per_ball_metrics, Mapping) else {}
        if not isinstance(values, Mapping):
            values = {}
        para_components = {
            "rmse": _metric_quality(values.get("parabola_rmse_d"), config, "parabola_rmse_d"),
            "p95": _metric_quality(values.get("parabola_p95_d"), config, "parabola_p95_d"),
        }
        horizontal_components = {
            "x_fit_rmse": _metric_quality(values.get("x_fit_rmse_d"), config, "x_fit_rmse_d"),
            "vx_robust_cv": _metric_quality(values.get("vx_robust_cv"), config, "vx_robust_cv"),
        }
        vertical_components = {
            "y_fit_rmse": _metric_quality(values.get("y_fit_rmse_d"), config, "y_fit_rmse_d"),
            "vy_line_residual": _metric_quality(
                values.get("vy_line_residual"), config, "vy_line_residual"
            ),
            "ay_robust_cv": _metric_quality(values.get("ay_robust_cv"), config, "ay_robust_cv"),
        }
        para_score = weighted_geometric_mean(list(para_components.values()), [0.65, 0.35])
        horizontal_score = weighted_geometric_mean(
            list(horizontal_components.values()), [0.60, 0.40]
        )
        vertical_score = weighted_geometric_mean(
            list(vertical_components.values()), [0.65, 0.20, 0.15]
        )
        parabola_balls.append(para_score)
        horizontal_balls.append(horizontal_score)
        vertical_balls.append(vertical_score)
        per_ball_details[slot] = {
            "parabola_components": para_components,
            "parabola": para_score,
            "horizontal_components": horizontal_components,
            "horizontal": horizontal_score,
            "vertical_components": vertical_components,
            "vertical": vertical_score,
        }

    # Each ball is an independent observation; neither is fit to or scored against
    # the other, and the dimension aggregate gives them equal weight.
    parabola = weighted_geometric_mean(parabola_balls, [0.5, 0.5])
    horizontal = weighted_geometric_mean(horizontal_balls, [0.5, 0.5])
    vertical = weighted_geometric_mean(vertical_balls, [0.5, 0.5])

    pair_components = {
        "equal_range": _metric_quality(metrics.get("M1_abs"), config, "range_ratio_error"),
        "equal_initial_speed": _metric_quality(
            metrics.get("initial_speed_error"), config, "initial_speed_error"
        ),
        "gravity_symmetry": _metric_quality(
            metrics.get("gravity_symmetric_error"), config, "gravity_error"
        ),
    }
    pair_consistency = weighted_geometric_mean(
        list(pair_components.values()), [0.45, 0.40, 0.15]
    )

    dimensions = {
        "task": task,
        "parabola": parabola,
        "horizontal": horizontal,
        "vertical": vertical,
        "pair_consistency": pair_consistency,
    }
    overall_weights = config.get("score_weights") or {}
    overall_candidate = weighted_geometric_mean(
        list(dimensions.values()),
        [float(overall_weights.get(name, 0.0)) for name in dimensions],
    )
    overall = overall_candidate if bool(result.get("measurement_valid")) else 0.0
    scores = {**dimensions, "overall": float(overall)}
    details = {
        "profile": PROFILE,
        "residual_quality_formula": "q(r,s)=1/(1+abs(r)/s)",
        "anchor_policy": "s is the legacy bad threshold (the q=0.5 half-quality anchor)",
        "count_quality_formula": "max(0.01,min(max(observed,0)/required,1))",
        "aggregation": "weighted geometric mean; every input is clamped to [0.01,1]",
        "measurement_valid_gate": "overall=0 iff measurement_valid=false; hard failures do not zero it",
        "task_components": task_components,
        "track_integrity_components": dict(zip(integrity_labels, integrity_values)),
        "per_ball": per_ball_details,
        "pair_components": pair_components,
        "overall_candidate_before_measurement_gate": overall_candidate,
    }
    return scores, details


def rescore_result(result: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    rescored = copy.deepcopy(dict(result))
    if "scores_legacy" in rescored:
        legacy = copy.deepcopy(rescored["scores_legacy"])
    else:
        legacy = copy.deepcopy(rescored.get("scores") or {})
    scores, details = score_continuous(rescored, config)
    rescored["scores_legacy"] = legacy
    rescored["scores"] = scores
    rescored["score_profile"] = PROFILE
    rescored["score_version"] = PROFILE
    rescored["score_details"] = details
    rescored["score_status_semantics"] = {
        "extract_success_preserved": bool(result.get("extract_success")),
        "structural_ok_preserved": bool(result.get("structural_ok")),
        "measurement_valid_preserved": bool(result.get("measurement_valid")),
        "physics_pass_preserved": bool(result.get("physics_pass")),
    }
    for name, value in scores.items():
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise AssertionError(f"continuous score out of range: {name}={value!r}")
    if bool(result.get("measurement_valid")) and not scores["overall"] > 0.0:
        raise AssertionError("measurement_valid=true must have overall>0")
    if not bool(result.get("measurement_valid")) and scores["overall"] != 0.0:
        raise AssertionError("measurement_valid=false must have overall=0")
    if rescored.get("physics_pass") != result.get("physics_pass"):
        raise AssertionError("physics_pass changed during rescore")
    return rescored
