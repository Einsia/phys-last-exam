"""Shared deterministic physics acceptance rules for Group 7.

The task evaluators measure pixels and expose measurements/metrics; this module
is the single place that turns a *complete* extraction into the declared
strict-physics decision.  Keeping the policy here makes the task-local CLI,
the batch runner, and independent re-runs use exactly the same rules.

No learned model or VLM is used.  A failed extraction is always a failed
physics decision, even when a partial measurement happens to look plausible.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


# P20's M1 combines the three air-to-water Snell estimates with a critical/TIR
# estimate.  A low CV is therefore not sufficient if the video omitted the
# TIR branch (or turned it into an ordinary refracted branch).  These limits
# mirror the explicit pixel-geometry checks in ``tasks/p20_optics.py``.
P20_MIN_SNELL_PAIRS = 3
P20_CRITICAL_REFLECTION_MAX_RESIDUAL_DEG = 10.0
P20_CRITICAL_REFLECTION_MAX_MISMATCH_RATIO = 0.020


def finite(value: object) -> float | None:
    """Return a finite float, or ``None`` for missing/non-finite values."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _p20_single_reflection_evidence(
    measurements: Mapping[str, Any], *, width: float | None = None
) -> bool:
    """Validate one frame's serialized water-side reflection geometry."""

    critical = measurements.get("critical_ray")
    if not isinstance(critical, Mapping):
        return False
    if critical.get("mode") != "water_side_reflection_pair":
        return False
    indices = critical.get("water_indices")
    water_rays = measurements.get("water_rays")
    if (
        not isinstance(indices, (list, tuple))
        or len(indices) != 2
        or not isinstance(water_rays, list)
    ):
        return False
    try:
        idx_a, idx_b = int(indices[0]), int(indices[1])
    except (TypeError, ValueError):
        return False
    if idx_a == idx_b or not (0 <= idx_a < len(water_rays) and 0 <= idx_b < len(water_rays)):
        return False
    ray_a, ray_b = water_rays[idx_a], water_rays[idx_b]
    if not isinstance(ray_a, Mapping) or not isinstance(ray_b, Mapping):
        return False
    slope_a = finite(ray_a.get("tangent_per_normal"))
    slope_b = finite(ray_b.get("tangent_per_normal"))
    if slope_a is None or slope_b is None or slope_a * slope_b >= 0.0:
        return False

    residual = finite(critical.get("reflection_residual_deg"))
    mismatch = finite(critical.get("intersection_mismatch_px"))
    frame_width = width if width is not None else finite(measurements.get("image_width"))
    if residual is None or residual > P20_CRITICAL_REFLECTION_MAX_RESIDUAL_DEG:
        return False
    if mismatch is None:
        return False
    max_mismatch = max(
        12.0,
        P20_CRITICAL_REFLECTION_MAX_MISMATCH_RATIO * frame_width if frame_width is not None else 12.0,
    )
    if mismatch > max_mismatch:
        return False
    angle = finite(critical.get("critical_angle_deg"))
    n_critical = finite(critical.get("n_critical"))
    if angle is None or not (20.0 <= angle <= 80.0) or n_critical is None or n_critical <= 1.0:
        return False
    return True


def _p20_reflection_evidence(measurements: Mapping[str, Any]) -> tuple[bool, str]:
    """Return whether rich P20 measurements prove a TIR branch.

    This is intentionally stricter than the extractor's diagnostic fallback:
    a single water-side incident ray has an angle but no reflected outgoing
    branch, and must not be promoted to a physics pass.  The check re-validates
    serialized ray geometry rather than trusting the summary boolean alone. If
    the best selected frame is not the frame containing the clearest branch,
    complete temporal frame records are accepted as evidence instead.
    """

    width = finite(measurements.get("image_width"))
    # New evaluator outputs explicitly bind the selected metric frame to the
    # complete reflection evidence.  Do not combine a pair count from one
    # frame with a TIR candidate from another frame.
    explicit_complete = measurements.get("critical_reflection_complete_frame_indices")
    if isinstance(explicit_complete, list):
        selected = measurements.get("selected_frame_index")
        try:
            selected_in_complete = selected is not None and int(selected) in {
                int(value) for value in explicit_complete
            }
        except (TypeError, ValueError):
            selected_in_complete = False
        if not selected_in_complete or not _p20_single_reflection_evidence(
            measurements, width=width
        ):
            return False, "critical_total_reflection_not_verified"
        return True, ""
    if _p20_single_reflection_evidence(measurements, width=width):
        return True, ""
    temporal = measurements.get("temporal_frame_measurements")
    if isinstance(temporal, list):
        for frame in temporal:
            if not isinstance(frame, Mapping):
                continue
            pair_count = finite(frame.get("snell_pair_count"))
            if pair_count is None or pair_count < P20_MIN_SNELL_PAIRS or frame.get("extract_success") is not True:
                continue
            if _p20_single_reflection_evidence(frame, width=width):
                return True, ""
    return False, "critical_total_reflection_not_verified"


def physics_decision(task: str, result: dict[str, Any]) -> tuple[bool, list[str]]:
    """Apply the v3 task-specific deterministic acceptance bands.

    The function intentionally checks ``extract_success`` first.  Metrics in a
    failed/partial extraction are diagnostic evidence only and must never be
    promoted to a physics pass by a downstream caller.  The public
    ``sample_xx.json`` envelope exposes only the Feishu-declared M1/M2 values;
    supplementary premise/quality gates below continue to consume the rich
    evaluator result and are preserved under ``verbose.status``.
    """

    if not result.get("extract_success"):
        return False, ["extract_failed"]

    metrics = result.get("metrics") or {}
    measurements = result.get("measurements") or {}
    reasons: list[str] = []
    m1 = finite(metrics.get("M1"))
    if m1 is None:
        return False, ["M1_missing"]

    if task == "P9":
        if m1 > 0.10:
            reasons.append("time_ratio_relative_error_gt_10pct")
        # The revised task defines only same-distance timing. Old angular
        # diagnostics cannot reintroduce the explicitly removed M2 gate.
    elif task == "P13":
        if m1 > 0.05:
            reasons.append("period_ratio_error_gt_0.05s")
        if measurements.get("T30_greater_than_T15") is not True:
            reasons.append("T30_not_greater_than_T15")
        residual = finite(metrics.get("M2_sinusoid_fit_residual"))
        if residual is not None and residual > 0.25:
            reasons.append("sinusoid_residual_gt_0.25")
        # Equal length is controlled by the first-frame/video specification.
        # Pixel circle fits remain useful diagnostics, but an apparent length
        # mismatch must not reject an otherwise measurable period comparison.
        # Likewise, tracking-quality flags stay in verbose evidence rather
        # than becoming undeclared benchmark violations.
        for item in result.get("metric_failures") or []:
            label = str(item)
            if label in {
                "string_length_measurement_unavailable",
                "string_length_relative_difference_gt_0.10",
            }:
                continue
            if label not in reasons:
                reasons.append(f"metric:{label}")
    elif task == "P11":
        if m1 > 0.10:
            reasons.append("acceleration_ratio_relative_error_gt_10pct")
        if finite(metrics.get("M2_t_down_gt_t_up")) != 1.0:
            reasons.append("return_time_not_longer_than_up_time")
        # A nominal 30-degree prompt is not reliable evidence: generated
        # first frames often render a visibly different incline.  P11's
        # evaluator therefore scores M1 against the observed ramp angle.  If
        # geometry was unavailable and the evaluator had to use the task prior
        # as a fallback, retain the raw fit but do not call it a strict physics
        # pass; the reason is explicit for human sanity review.
        if measurements.get("ramp_angle_fallback") is True:
            reasons.append("ramp_angle_estimation_fallback")
    elif task == "P20":
        if m1 > 0.05:
            reasons.append("optical_index_cv_gt_0.05")
        # Extraction intentionally keeps partial ray measurements, but a
        # *strict physics pass* represents the complete P20 scene: three
        # air-to-water Snell pairs plus an independently visible water-side
        # total-reflection branch.  Without this gate a clip containing only
        # ordinary refraction (or a lone water incident line) could obtain a
        # deceptively low CV and be labelled correct.
        pair_count = finite(measurements.get("snell_pair_count"))
        if pair_count is None or pair_count < P20_MIN_SNELL_PAIRS:
            reasons.append("snell_pair_count_lt_3")
        verified, _ = _p20_reflection_evidence(measurements)
        if verified is not True:
            reasons.append("critical_total_reflection_not_verified")
    elif task == "P28":
        faster = measurements.get("crushed_water_surface_rise_faster")
        if faster is None:
            raw = metrics.get("M1_crushed_water_surface_rise_faster", metrics.get("M1"))
            if isinstance(raw, bool):
                faster = raw
            elif finite(raw) is not None:
                faster = bool(finite(raw))
        if faster is not True:
            reasons.append("crushed_ice_not_faster")
    else:
        reasons.append("unknown_task")
    return not reasons, reasons


__all__ = ["finite", "physics_decision"]
