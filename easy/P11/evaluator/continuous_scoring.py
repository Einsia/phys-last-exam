"""Continuous 0--1 scoring for already-extracted P11 measurements."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping, Optional


SCORING_VERSION = "continuous-0-1-v1"
INPUT_FLOOR = 0.01
DIMENSION_WEIGHTS = {
    "M1_snell_quality": 0.60,
    "M2_intersection_quality": 0.20,
    "fit_quality": 0.08,
    "temporal_stability": 0.07,
    "camera_stability": 0.05,
}


def clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


def finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def q_residual(residual: Any, half_quality_anchor: float) -> float:
    """Smooth residual quality: q(0)=1 and q(anchor)=0.5."""

    value = finite(residual)
    anchor = finite(half_quality_anchor)
    if value is None or anchor is None or anchor <= 0:
        return 0.0
    value = max(0.0, value)
    return clamp01(1.0 / (1.0 + value / anchor))


def support_quality(value: Any, full_support_anchor: float) -> float:
    """Continuous linear support map, saturating at full support."""

    number = finite(value)
    anchor = finite(full_support_anchor)
    if number is None or anchor is None or anchor <= 0:
        return 0.0
    return clamp01(max(0.0, number) / anchor)


def weighted_geometric_mean(
    values: Mapping[str, float],
    weights: Mapping[str, float],
    floor: float = INPUT_FLOOR,
) -> float:
    weight_sum = sum(float(weights[key]) for key in weights)
    if weight_sum <= 0:
        raise ValueError("weighted geometric mean requires positive total weight")
    log_sum = 0.0
    for key, weight in weights.items():
        value = clamp01(float(values.get(key, 0.0)))
        log_sum += float(weight) * math.log(max(float(floor), value))
    return clamp01(math.exp(log_sum / weight_sum))


def nested(record: Mapping[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def line_fit_quality(line: Optional[Mapping[str, Any]], rmse_anchor: float, span_anchor: float) -> float:
    if not line:
        return 0.0
    components = {
        "rmse": q_residual(line.get("rmse_px"), rmse_anchor),
        "support": support_quality(line.get("span_tank_fraction"), span_anchor),
    }
    return weighted_geometric_mean(components, {"rmse": 0.70, "support": 0.30})


def continuous_dimensions(record: Mapping[str, Any]) -> Dict[str, float]:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
    geometry = record.get("geometry") if isinstance(record.get("geometry"), Mapping) else {}
    surface = geometry.get("surface") if isinstance(geometry.get("surface"), Mapping) else None
    incident = geometry.get("incident_ray") if isinstance(geometry.get("incident_ray"), Mapping) else None
    refracted = geometry.get("refracted_ray") if isinstance(geometry.get("refracted_ray"), Mapping) else None
    m1 = q_residual(metrics.get("M1_snell_residual_abs"), 0.12)
    m2 = q_residual(metrics.get("M2_intersection_disagreement_normalized"), 0.025)
    fit_components = {
        "surface": line_fit_quality(surface, rmse_anchor=3.0, span_anchor=0.42),
        "incident": line_fit_quality(incident, rmse_anchor=7.0, span_anchor=0.09),
        "refracted": line_fit_quality(refracted, rmse_anchor=7.0, span_anchor=0.10),
    }
    fit_quality = weighted_geometric_mean(
        fit_components,
        {"surface": 0.25, "incident": 0.375, "refracted": 0.375},
    )
    stability = nested(record, "temporal", "ray_stability")
    stability = stability if isinstance(stability, Mapping) else {}
    stable_on = nested(record, "temporal", "stable_on_frame_indices")
    stable_on_count = len(stable_on) if isinstance(stable_on, list) else 0
    temporal_components = {
        "incident_angle": q_residual(stability.get("incident_angle_mad_deg"), 4.0),
        "refracted_angle": q_residual(stability.get("refracted_angle_mad_deg"), 4.0),
        "sample_support": support_quality(stability.get("sample_count"), 7.0),
        "stable_on_support": support_quality(stable_on_count, 24.0),
    }
    temporal_quality = weighted_geometric_mean(
        temporal_components,
        {
            "incident_angle": 0.35,
            "refracted_angle": 0.35,
            "sample_support": 0.15,
            "stable_on_support": 0.15,
        },
    )
    camera = nested(record, "structure", "camera")
    camera = camera if isinstance(camera, Mapping) else {}
    camera_components = {
        "translation": q_residual(camera.get("translation_diagonal_fraction"), 0.02),
        "scale": q_residual(camera.get("scale_error"), 0.02),
        "rotation": q_residual(abs(finite(camera.get("rotation_deg")) or 0.0), 1.0)
        if finite(camera.get("rotation_deg")) is not None
        else 0.0,
    }
    camera_quality = weighted_geometric_mean(
        camera_components,
        {"translation": 1.0 / 3.0, "scale": 1.0 / 3.0, "rotation": 1.0 / 3.0},
    )
    return {
        "M1_snell_quality": m1,
        "M2_intersection_quality": m2,
        "fit_quality": fit_quality,
        "temporal_stability": temporal_quality,
        "camera_stability": camera_quality,
    }


def build_continuous_scores(record: Mapping[str, Any]) -> Dict[str, Any]:
    dimensions = continuous_dimensions(record)
    measurement_valid = nested(record, "statuses", "measurement_valid") is True
    overall = weighted_geometric_mean(dimensions, DIMENSION_WEIGHTS, INPUT_FLOOR) if measurement_valid else 0.0
    physics_pair = weighted_geometric_mean(
        dimensions,
        {"M1_snell_quality": 0.75, "M2_intersection_quality": 0.25},
        INPUT_FLOOR,
    )
    return {
        "scoring_version": SCORING_VERSION,
        "dimensions": dimensions,
        "derived": {
            "physics_pair_quality": physics_pair,
        },
        "overall": overall,
        "aggregation": {
            "method": "weighted_geometric_mean",
            "input_floor": INPUT_FLOOR,
            "weights": copy.deepcopy(DIMENSION_WEIGHTS),
            "validity_gate": "measurement_valid=false => overall=0; otherwise floor guarantees overall>0",
        },
        "quality_anchors": {
            "M1_abs_residual_half_quality": 0.12,
            "M2_normalized_residual_half_quality": 0.025,
            "surface_rmse_px_half_quality": 3.0,
            "ray_rmse_px_half_quality": 7.0,
            "angle_mad_deg_half_quality": 4.0,
            "camera_translation_fraction_half_quality": 0.02,
            "camera_scale_error_half_quality": 0.02,
            "camera_rotation_deg_half_quality": 1.0,
        },
        "hard_labels_not_used_as_score_steps": {
            "physics_pass": nested(record, "statuses", "physics_pass"),
            "structural_ok": nested(record, "statuses", "structural_ok"),
            "measurement_valid": measurement_valid,
        },
    }


def migrate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate in memory while preserving metrics, statuses and legacy score."""

    if "legacy_scores_0_100" not in record:
        current = record.get("scores")
        if isinstance(current, Mapping) and current.get("scoring_version") != SCORING_VERSION:
            record["legacy_scores_0_100"] = copy.deepcopy(current)
    record["scores"] = build_continuous_scores(record)
    # Keep a task-independent top-level version marker in addition to the
    # detailed marker nested under ``scores``.  Consumers can therefore
    # validate the score contract without knowing the P11 score layout.
    record["score_version"] = SCORING_VERSION
    return record
