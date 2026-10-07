"""Deterministic evaluator for P11: a block reversing on a rough ramp.

The evaluator deliberately uses only pixels, frame timestamps, and the explicit
mechanics model.  It never calls a VLM.  The moving block is segmented against
a temporal-median background, its trajectory is reduced to the dominant ramp
axis, the rendered ramp angle is estimated from that axis/static edges, and
independent robust quadratics are fitted before and after the turn.  The
mechanics reference ratio is computed from that observed angle rather than
assuming the nominal prompt angle.

Public entry points
-------------------
``evaluate(frames, fps, debug_dir, sample_id=0)`` is the integration API.
``evaluate_video(video_path, ...)`` is a convenience wrapper for file-based evaluation.
``evaluate_p11`` is an alias for dispatchers that prefer task-qualified names.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


TASK_ID = "P11"
# ``RAMP_ANGLE_DEG`` is retained as the task-design prior and as a backwards
# compatible fallback for old callers.  It must not be used as the primary
# scoring angle: GPT/Image-to-video can render a visibly different slope than
# the nominal prompt angle.  The evaluator estimates the angle from the
# rendered ramp/trajectory for every usable video.
RAMP_ANGLE_DEG = 30.0
KINETIC_FRICTION_COEFFICIENT = 0.20


def expected_acceleration_ratio(
    angle_deg: float,
    mu_k: float = KINETIC_FRICTION_COEFFICIENT,
) -> float | None:
    """Return ``a_up/a_down`` for a ramp at ``angle_deg``.

    The upward and downward acceleration magnitudes for a block sliding on a
    rough incline are ``g(sin(theta)+mu*cos(theta))`` and
    ``g(sin(theta)-mu*cos(theta))``.  A non-positive denominator means that a
    block cannot slide down under this model (the angle is at/below the
    friction threshold), so ``None`` is returned instead of manufacturing a
    score.
    """

    try:
        theta = math.radians(float(angle_deg))
        mu = float(mu_k)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(theta) and math.isfinite(mu) and mu >= 0.0):
        return None
    numerator = math.sin(theta) + mu * math.cos(theta)
    denominator = math.sin(theta) - mu * math.cos(theta)
    if denominator <= 1e-9 or not math.isfinite(numerator):
        return None
    ratio = numerator / denominator
    return float(ratio) if math.isfinite(ratio) and ratio > 0.0 else None


# Public compatibility constant.  New evaluations use the observed angle and
# ``expected_acceleration_ratio`` below; direct integrations can
# import the nominal 30-degree value.
EXPECTED_ACCELERATION_RATIO = expected_acceleration_ratio(RAMP_ANGLE_DEG)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _result(
    sample_id: int | str,
    success: bool,
    measurements: dict[str, Any],
    metrics: dict[str, float] | None = None,
    failure_reasons: Sequence[str] = (),
    debug_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    reasons = list(dict.fromkeys(str(x) for x in failure_reasons if x))
    if metrics is None:
        metrics = {"M1": None}
    if not success:
        # A partial trajectory can still expose useful measurements, but it
        # is not a valid score.  Use JSON null rather than a synthetic zero so
        # CSV/JSON consumers cannot mistake an extraction failure for a
        # perfect M1.
        metrics = {str(k): None for k in metrics} or {"M1": None}
    try:
        serialized_sample_id: int | str = int(sample_id)
    except (TypeError, ValueError):
        serialized_sample_id = str(sample_id)
    return _jsonable(
        {
            "task_id": TASK_ID,
            "sample_id": serialized_sample_id,
            "extract_success": bool(success),
            "measurements": measurements,
            "metrics": metrics,
            "failure_reason": reasons[0] if reasons else None,
            "failure_reasons": reasons,
            "debug_artifacts": debug_artifacts or {},
        }
    )


def _validate_frames(frames: Sequence[np.ndarray], fps: float) -> tuple[list[np.ndarray], list[str]]:
    reasons: list[str] = []
    materialized = [np.asarray(frame) for frame in frames if frame is not None]
    if not materialized:
        return [], ["video_decode_failed"]
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        reasons.append("invalid_fps")
    shape = materialized[0].shape
    if len(shape) != 3 or shape[2] != 3:
        reasons.append("frames_are_not_bgr_color")
    if any(frame.shape != shape for frame in materialized):
        reasons.append("inconsistent_frame_dimensions")
    if len(materialized) < 12:
        reasons.append("too_few_frames")
    return materialized, reasons


def _temporal_background(frames: Sequence[np.ndarray], max_samples: int = 121) -> np.ndarray:
    indices = np.linspace(0, len(frames) - 1, min(len(frames), max_samples), dtype=int)
    stack = np.stack([frames[int(i)] for i in indices], axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def _extract_track(
    frames: Sequence[np.ndarray], background: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], dict[str, float]]:
    """Return frame indices, centroids, masks, and segmentation diagnostics."""

    height, width = frames[0].shape[:2]
    frame_area = float(height * width)
    min_area = max(16, int(frame_area * 0.00012))
    max_area = int(frame_area * 0.10)
    kernel3 = np.ones((3, 3), np.uint8)
    kernel5 = np.ones((5, 5), np.uint8)
    selected_indices: list[int] = []
    selected_points: list[tuple[float, float]] = []
    masks: list[np.ndarray] = []
    candidate_counts: list[int] = []

    for frame_index, frame in enumerate(frames):
        difference = cv2.absdiff(frame, background)
        scalar = np.max(difference, axis=2).astype(np.uint8)
        # Compression noise and mild illumination breathing normally remain below
        # 18 levels.  A robust per-frame floor handles noisier generated videos.
        median = float(np.median(scalar))
        mad = float(np.median(np.abs(scalar.astype(np.float32) - median)))
        threshold = int(np.clip(max(18.0, median + 6.0 * 1.4826 * mad), 18, 64))
        mask = np.where(scalar >= threshold, 255, 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5)
        masks.append(mask)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates: list[tuple[float, int]] = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            if w > 0.55 * width or h > 0.55 * height:
                continue
            component = labels == label
            mean_difference = float(np.mean(scalar[component]))
            mean_saturation = float(np.mean(hsv[:, :, 1][component]))
            # Shadows can be larger than the block.  Difference contrast and
            # saturation provide a small, appearance-agnostic preference for the
            # actual object without assuming a particular block colour.
            score = area * mean_difference * (1.0 + 0.35 * mean_saturation / 255.0)
            candidates.append((score, label))

        candidate_counts.append(len(candidates))
        if not candidates:
            continue
        _, best_label = max(candidates)
        selected_indices.append(frame_index)
        selected_points.append(tuple(float(v) for v in centroids[best_label]))

    diagnostics = {
        "minimum_component_area_px": float(min_area),
        "maximum_component_area_px": float(max_area),
        "frames_with_candidates": float(len(selected_indices)),
        "median_candidates_per_frame": float(np.median(candidate_counts)) if candidate_counts else 0.0,
    }
    return (
        np.asarray(selected_indices, dtype=int),
        np.asarray(selected_points, dtype=np.float64),
        masks,
        diagnostics,
    )


def _robust_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Fit the dominant straight trajectory with iterative MAD rejection."""

    keep = np.ones(len(points), dtype=bool)
    center = np.median(points, axis=0)
    axis = np.array([1.0, 0.0], dtype=np.float64)
    for _ in range(5):
        current = points[keep]
        center = np.mean(current, axis=0)
        _, _, vh = np.linalg.svd(current - center, full_matrices=False)
        axis = vh[0]
        normal = np.array([-axis[1], axis[0]])
        residual = np.abs((points - center) @ normal)
        med = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - med)))
        limit = max(2.5, med + 4.0 * 1.4826 * mad)
        new_keep = residual <= limit
        if np.count_nonzero(new_keep) < max(8, int(0.55 * len(points))):
            break
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    normal = np.array([-axis[1], axis[0]])
    parallel = (points[keep] - center) @ axis
    perpendicular = (points[keep] - center) @ normal
    span = float(np.ptp(parallel)) if len(parallel) else 0.0
    perpendicular_rmse = float(np.sqrt(np.mean(perpendicular**2))) if len(perpendicular) else math.inf
    diagnostics = {
        "axis_x": float(axis[0]),
        "axis_y": float(axis[1]),
        "axis_center_x": float(center[0]),
        "axis_center_y": float(center[1]),
        "axis_span_px": span,
        "axis_perpendicular_rmse_px": perpendicular_rmse,
        "axis_relative_rmse": perpendicular_rmse / max(span, 1e-9),
        "axis_inlier_fraction": float(np.mean(keep)),
    }
    return center, axis, keep, diagnostics


def _acute_line_angle_deg(axis: np.ndarray) -> float | None:
    """Return a line's acute image-plane angle above horizontal.

    Image ``y`` increases downwards and an unoriented line has two equivalent
    directions.  Taking absolute components therefore gives the physically
    useful incline angle in ``[0, 90]`` independent of motion direction.
    """

    vector = np.asarray(axis, dtype=np.float64).reshape(-1)
    if vector.size < 2:
        return None
    x, y = float(vector[0]), float(vector[1])
    norm = math.hypot(x, y)
    if not math.isfinite(norm) or norm <= 1e-9:
        return None
    angle = math.degrees(math.atan2(abs(y), abs(x)))
    return float(angle) if math.isfinite(angle) else None


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float | None:
    """Small dependency-free weighted median helper for Hough line angles."""

    if not values or len(values) != len(weights):
        return None
    pairs = sorted(
        (float(value), max(0.0, float(weight)))
        for value, weight in zip(values, weights)
        if math.isfinite(float(value)) and math.isfinite(float(weight))
    )
    if not pairs:
        return None
    total = sum(weight for _, weight in pairs)
    if total <= 1e-12:
        return float(np.median([value for value, _ in pairs]))
    halfway = 0.5 * total
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= halfway:
            return float(value)
    return float(pairs[-1][0])


def _estimate_ramp_angle(
    axis: np.ndarray,
    axis_diagnostics: dict[str, float],
    background: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate the rendered ramp angle from observable image geometry.

    The fitted block trajectory is already constrained to the ramp and is the
    most stable geometric cue.  When a temporal background is available we
    additionally inspect long, straight Hough edges near that direction (the
    static ramp board/edge), using them to suppress a trajectory outlier.  No
    nominal prompt angle is substituted unless both cues are unavailable.

    The returned dictionary is intentionally verbose: angle source, candidate
    lines, dispersion, confidence, and fallback state are preserved for human
    sanity checks.  A fallback still permits extraction of the raw motion
    measurements, but the shared physics policy can decline a strict pass.
    """

    trajectory_angle = _acute_line_angle_deg(axis)
    axis_span = float(axis_diagnostics.get("axis_span_px", 0.0) or 0.0)
    axis_rmse = float(axis_diagnostics.get("axis_perpendicular_rmse_px", math.inf) or math.inf)
    image_diagonal = 1.0
    if background is not None and getattr(background, "ndim", 0) >= 2:
        image_diagonal = max(1.0, math.hypot(float(background.shape[1]), float(background.shape[0])))

    hough_angles: list[float] = []
    hough_weights: list[float] = []
    if background is not None and trajectory_angle is not None:
        try:
            gray = cv2.cvtColor(np.asarray(background), cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            height, width = gray.shape[:2]
            min_length = max(40.0, 0.18 * float(min(width, height)))
            max_gap = max(6.0, 0.035 * float(max(width, height)))
            threshold = max(24, int(round(0.06 * float(min(width, height)))))
            lines = cv2.HoughLinesP(
                edges,
                1.0,
                np.pi / 180.0,
                threshold=threshold,
                minLineLength=int(round(min_length)),
                maxLineGap=int(round(max_gap)),
            )
            for line in lines if lines is not None else []:
                x1, y1, x2, y2 = [float(v) for v in line[0]]
                length = math.hypot(x2 - x1, y2 - y1)
                angle = _acute_line_angle_deg(np.asarray([x2 - x1, y2 - y1]))
                if angle is None or length < min_length:
                    continue
                difference = abs(angle - trajectory_angle)
                # A static ramp edge should agree with the tracked trajectory;
                # reject unrelated table/floor edges while allowing mild
                # perspective and segmentation drift.
                if difference > 12.0:
                    continue
                weight = length * max(0.05, 1.0 - difference / 12.0)
                hough_angles.append(float(angle))
                hough_weights.append(float(weight))
        except (cv2.error, ValueError, TypeError):
            # Geometry extraction must never turn a valid trajectory into a
            # hard failure merely because an image has no usable edge map.
            hough_angles = []
            hough_weights = []

    hough_angle = _weighted_median(hough_angles, hough_weights)
    if hough_angle is not None:
        chosen_angle = hough_angle
        source = "background_hough_near_trajectory"
        deviations = [abs(value - hough_angle) for value in hough_angles]
        dispersion = float(np.median(deviations)) if deviations else None
    elif trajectory_angle is not None:
        chosen_angle = trajectory_angle
        source = "trajectory_axis"
        dispersion = None
    else:
        chosen_angle = float(RAMP_ANGLE_DEG)
        source = "task_nominal_fallback"
        dispersion = None

    warnings: list[str] = []
    fallback = source == "task_nominal_fallback"
    if chosen_angle is None or not math.isfinite(float(chosen_angle)):
        chosen_angle = float(RAMP_ANGLE_DEG)
        source = "task_nominal_fallback"
        fallback = True
        warnings.append("ramp_angle_non_finite")
    chosen_angle = float(np.clip(float(chosen_angle), 0.0, 90.0))
    expected = expected_acceleration_ratio(chosen_angle, KINETIC_FRICTION_COEFFICIENT)
    if expected is None:
        # At/below the friction threshold the requested down-ramp motion is
        # physically undefined.  Keep a finite diagnostic baseline so M1 can
        # still expose the observed trajectory, while marking the estimate as
        # a physics-policy fallback.
        warnings.append("ramp_angle_at_or_below_friction_threshold")
        expected = expected_acceleration_ratio(RAMP_ANGLE_DEG, KINETIC_FRICTION_COEFFICIENT)
        fallback = True

    # Confidence is diagnostic only.  Long, straight tracks and mutually
    # consistent static edges receive higher confidence; it is not an opaque
    # extraction gate.
    span_score = float(np.clip(axis_span / max(0.20 * image_diagonal, 1e-9), 0.0, 1.0))
    straight_score = float(np.clip(1.0 - axis_rmse / max(8.0, 0.025 * max(axis_span, 1.0)), 0.0, 1.0))
    if hough_angle is not None and dispersion is not None:
        edge_score = float(np.clip(1.0 - dispersion / 6.0, 0.0, 1.0))
    else:
        edge_score = 0.55 if trajectory_angle is not None else 0.0
    confidence = float(np.clip(0.45 * span_score + 0.35 * straight_score + 0.20 * edge_score, 0.0, 1.0))
    if fallback:
        confidence *= 0.25

    return {
        "ramp_angle_deg": chosen_angle,
        "observed_ramp_angle_deg": chosen_angle,
        "ramp_angle_source": source,
        "ramp_angle_fallback": bool(fallback),
        "ramp_angle_confidence": confidence,
        "trajectory_axis_angle_deg": trajectory_angle,
        "background_hough_angle_deg": hough_angle,
        "background_hough_candidate_count": len(hough_angles),
        "background_hough_angle_median_abs_deviation_deg": dispersion,
        "expected_acceleration_ratio_from_observed_angle": expected,
        "ramp_angle_warnings": warnings,
    }


def _moving_median(values: np.ndarray, radius: int = 2) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    result = np.empty_like(values, dtype=np.float64)
    for index in range(len(values)):
        lo = max(0, index - radius)
        hi = min(len(values), index + radius + 1)
        result[index] = float(np.median(values[lo:hi]))
    return result


def _find_turn(projection: np.ndarray) -> tuple[int | None, float, dict[str, float]]:
    """Orient the coordinate uphill and locate the interior extremum."""

    smooth = _moving_median(projection, radius=max(1, min(3, len(projection) // 30)))
    edge_count = max(2, min(7, len(smooth) // 10))
    best: tuple[float, int, float, float, float] | None = None
    lower = max(3, int(0.12 * len(smooth)))
    upper = min(len(smooth) - 3, int(0.88 * len(smooth)))
    for orientation in (1.0, -1.0):
        oriented = orientation * smooth
        turn = int(np.argmax(oriented))
        if turn < lower or turn > upper:
            continue
        start = float(np.median(oriented[:edge_count]))
        end = float(np.median(oriented[-edge_count:]))
        peak = float(oriented[turn])
        before = peak - start
        after = peak - end
        score = min(before, after)
        if best is None or score > best[0]:
            best = (score, turn, orientation, before, after)
    if best is None:
        return None, 1.0, {"turn_score_px": 0.0}
    score, turn, orientation, before, after = best
    total_span = float(np.ptp(projection))
    diagnostics = {
        "turn_score_px": float(score),
        "turn_before_displacement_px": float(before),
        "turn_after_displacement_px": float(after),
        "turn_score_fraction_of_span": float(score / max(total_span, 1e-9)),
    }
    if score < max(4.0, 0.18 * total_span):
        return None, orientation, diagnostics
    return turn, orientation, diagnostics


def _robust_quadratic(times: np.ndarray, positions: np.ndarray) -> dict[str, Any]:
    shifted = times - float(times[0])
    keep = np.ones(len(times), dtype=bool)
    coefficients = np.polyfit(shifted, positions, 2)
    for _ in range(5):
        coefficients = np.polyfit(shifted[keep], positions[keep], 2)
        prediction = np.polyval(coefficients, shifted)
        residual = positions - prediction
        med = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - med)))
        limit = max(1.25, 4.0 * 1.4826 * mad)
        new_keep = np.abs(residual - med) <= limit
        if np.count_nonzero(new_keep) < max(5, int(0.65 * len(times))):
            break
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep
    coefficients = np.polyfit(shifted[keep], positions[keep], 2)
    prediction = np.polyval(coefficients, shifted)
    residual = positions - prediction
    rmse = float(np.sqrt(np.mean(residual[keep] ** 2)))
    centered = positions[keep] - float(np.mean(positions[keep]))
    total = float(np.sum(centered**2))
    r_squared = 1.0 - float(np.sum(residual[keep] ** 2)) / max(total, 1e-12)
    position_span = float(np.ptp(positions[keep]))
    return {
        "coefficients": coefficients,
        "prediction": prediction,
        "inliers": keep,
        "acceleration_px_s2": float(abs(2.0 * coefficients[0])),
        "signed_acceleration_px_s2": float(2.0 * coefficients[0]),
        "rmse_px": rmse,
        "relative_rmse": rmse / max(position_span, 1e-9),
        "r_squared": r_squared,
        "position_span_px": position_span,
        "inlier_fraction": float(np.mean(keep)),
    }


def _fit_shared_vertex(
    times: np.ndarray,
    positions: np.ndarray,
    rough_turn_index: int,
) -> dict[str, Any] | None:
    """Fit the two physical parabolas with one zero-velocity turning vertex.

    A free quadratic is biased when the tracked centroid forms a few-pixel
    plateau near zero speed.  Searching the shared vertex time makes the
    acceleration ratio invariant to that plateau and to the arbitrary pixel
    origin/scale.
    """

    if len(times) < 14:
        return None
    duration = float(times[-1] - times[0])
    rough_time = float(times[rough_turn_index])
    lower = max(float(times[6]), rough_time - 0.18 * duration)
    upper = min(float(times[-7]), rough_time + 0.18 * duration)
    if upper <= lower:
        return None

    best: tuple[float, float, np.ndarray, np.ndarray] | None = None
    for turn_time in np.linspace(lower, upper, 181):
        before = times <= turn_time
        after = times >= turn_time
        if np.count_nonzero(before) < 7 or np.count_nonzero(after) < 7:
            continue
        design = np.column_stack(
            [
                np.ones(len(times)),
                np.where(before, -0.5 * (turn_time - times) ** 2, 0.0),
                np.where(after, -0.5 * (times - turn_time) ** 2, 0.0),
            ]
        )
        coefficients, *_ = np.linalg.lstsq(design, positions, rcond=None)
        if coefficients[1] <= 0 or coefficients[2] <= 0:
            continue
        residual = positions - design @ coefficients
        # A trimmed objective prevents one bad segmentation frame from moving
        # the physically important zero-velocity vertex.
        squared = np.sort(residual**2)
        kept = squared[: max(10, int(math.ceil(0.90 * len(squared))))]
        objective = float(np.mean(kept))
        if best is None or objective < best[0]:
            best = (objective, float(turn_time), coefficients, design)
    if best is None:
        return None

    _, turn_time, coefficients, design = best
    keep = np.ones(len(times), dtype=bool)
    for _ in range(4):
        coefficients, *_ = np.linalg.lstsq(design[keep], positions[keep], rcond=None)
        residual = positions - design @ coefficients
        med = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - med)))
        limit = max(1.25, 4.0 * 1.4826 * mad)
        new_keep = np.abs(residual - med) <= limit
        if np.count_nonzero(new_keep) < max(10, int(0.70 * len(times))) or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    prediction = design @ coefficients
    turn_index = int(np.argmin(np.abs(times - turn_time)))
    before = np.arange(len(times)) <= turn_index
    after = np.arange(len(times)) >= turn_index

    def phase(mask: np.ndarray, acceleration: float) -> dict[str, Any]:
        residual = positions[mask] - prediction[mask]
        phase_keep = keep[mask]
        if not np.any(phase_keep):
            phase_keep = np.ones(np.count_nonzero(mask), dtype=bool)
        rmse = float(np.sqrt(np.mean(residual[phase_keep] ** 2)))
        span = float(np.ptp(positions[mask]))
        centered = positions[mask][phase_keep] - float(np.mean(positions[mask][phase_keep]))
        total = float(np.sum(centered**2))
        r_squared = 1.0 - float(np.sum(residual[phase_keep] ** 2)) / max(total, 1e-12)
        return {
            "acceleration_px_s2": float(acceleration),
            "signed_acceleration_px_s2": float(-acceleration),
            "rmse_px": rmse,
            "relative_rmse": rmse / max(span, 1e-9),
            "r_squared": r_squared,
            "position_span_px": span,
            "inlier_fraction": float(np.mean(phase_keep)),
            "prediction": prediction[mask],
            "coefficients": np.asarray([coefficients[0], turn_time, acceleration]),
        }

    return {
        "turn_time_s": turn_time,
        "turn_index": turn_index,
        "turn_coordinate_px": float(coefficients[0]),
        "up": phase(before, float(coefficients[1])),
        "down": phase(after, float(coefficients[2])),
        "inlier_fraction": float(np.mean(keep)),
    }


def _line_endpoints(center: np.ndarray, axis: np.ndarray, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    reach = math.hypot(width, height)
    a = center - axis * reach
    b = center + axis * reach
    return tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int))


def _write_tracking_debug(
    debug_dir: Path,
    frames: Sequence[np.ndarray],
    background: np.ndarray,
    frame_indices: np.ndarray,
    points: np.ndarray,
    center: np.ndarray | None,
    axis: np.ndarray | None,
    turn_frame: int | None,
) -> dict[str, str]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    background_path = debug_dir / "temporal_background.png"
    cv2.imwrite(str(background_path), background)
    artifacts["temporal_background"] = str(background_path)

    csv_path = debug_dir / "trajectory.csv"
    lookup = {int(i): point for i, point in zip(frame_indices, points)}
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "x_px", "y_px"])
        for frame_index, point in zip(frame_indices, points):
            writer.writerow([int(frame_index), float(point[0]), float(point[1])])
    artifacts["trajectory_csv"] = str(csv_path)

    chosen = np.linspace(0, len(frames) - 1, min(9, len(frames)), dtype=int)
    tiles: list[np.ndarray] = []
    height, width = frames[0].shape[:2]
    for frame_index in chosen:
        overlay = frames[int(frame_index)].copy()
        if center is not None and axis is not None:
            cv2.line(overlay, *_line_endpoints(center, axis, width, height), (0, 220, 255), 2, cv2.LINE_AA)
        for old_index in frame_indices[frame_indices <= frame_index]:
            point = lookup[int(old_index)]
            colour = (40, 210, 40) if turn_frame is None or old_index <= turn_frame else (255, 120, 30)
            cv2.circle(overlay, tuple(np.rint(point).astype(int)), 3, colour, -1, cv2.LINE_AA)
        point = lookup.get(int(frame_index))
        if point is not None:
            cv2.circle(overlay, tuple(np.rint(point).astype(int)), 8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"frame {int(frame_index)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tile_width = 360
        tile_height = max(1, int(round(height * tile_width / width)))
        tiles.append(cv2.resize(overlay, (tile_width, tile_height), interpolation=cv2.INTER_AREA))
    columns = 3
    rows = math.ceil(len(tiles) / columns)
    tile_h, tile_w = tiles[0].shape[:2]
    sheet = np.zeros((rows * tile_h, columns * tile_w, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[row * tile_h : (row + 1) * tile_h, column * tile_w : (column + 1) * tile_w] = tile
    contact_path = debug_dir / "tracking_contact_sheet.png"
    cv2.imwrite(str(contact_path), sheet)
    artifacts["tracking_contact_sheet"] = str(contact_path)
    return artifacts


def _write_fit_plot(
    debug_dir: Path,
    times: np.ndarray,
    positions: np.ndarray,
    turn_index: int,
    up_fit: dict[str, Any],
    down_fit: dict[str, Any],
) -> str:
    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 72, 24, 30, 62
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    t_min, t_max = float(times[0]), float(times[-1])
    p_min, p_max = float(np.min(positions)), float(np.max(positions))
    p_pad = max(1.0, 0.08 * (p_max - p_min))
    p_min -= p_pad
    p_max += p_pad

    def map_point(t: float, p: float) -> tuple[int, int]:
        x = margin_left + (t - t_min) / max(t_max - t_min, 1e-9) * (width - margin_left - margin_right)
        y = margin_top + (p_max - p) / max(p_max - p_min, 1e-9) * (height - margin_top - margin_bottom)
        return int(round(x)), int(round(y))

    cv2.line(canvas, (margin_left, margin_top), (margin_left, height - margin_bottom), (30, 30, 30), 2)
    cv2.line(canvas, (margin_left, height - margin_bottom), (width - margin_right, height - margin_bottom), (30, 30, 30), 2)
    for index, (time, position) in enumerate(zip(times, positions)):
        colour = (40, 160, 40) if index <= turn_index else (220, 110, 20)
        cv2.circle(canvas, map_point(float(time), float(position)), 3, colour, -1, cv2.LINE_AA)
    up_times = times[: turn_index + 1]
    up_pred = up_fit["prediction"]
    down_times = times[turn_index:]
    down_pred = down_fit["prediction"]
    for fit_times, fit_positions, colour in ((up_times, up_pred, (0, 100, 0)), (down_times, down_pred, (160, 50, 0))):
        mapped = np.asarray([map_point(float(t), float(p)) for t, p in zip(fit_times, fit_positions)], dtype=np.int32)
        if len(mapped) >= 2:
            cv2.polylines(canvas, [mapped], False, colour, 2, cv2.LINE_AA)
    cv2.putText(canvas, "time (s)", (width // 2 - 35, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.putText(canvas, "up-ramp coordinate (px)", (86, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    path = debug_dir / "ramp_axis_quadratic_fit.png"
    cv2.imwrite(str(path), canvas)
    return str(path)


def _write_summary(debug_dir: Path, result: dict[str, Any]) -> str:
    path = debug_dir / "p11_summary.json"
    path.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def evaluate(
    frames: Sequence[np.ndarray],
    fps: float,
    debug_dir: str | Path,
    sample_id: int = 0,
) -> dict[str, Any]:
    """Evaluate decoded BGR frames for P11."""

    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    materialized, failures = _validate_frames(frames, fps)
    if failures:
        result = _result(sample_id, False, {"frame_count": len(materialized), "fps": float(fps)}, failure_reasons=failures)
        result["debug_artifacts"]["summary"] = _write_summary(debug_path, result)
        return result

    fps = float(fps)
    background = _temporal_background(materialized)
    frame_indices, points, _, segmentation = _extract_track(materialized, background)
    minimum_track_points = max(10, int(math.ceil(0.25 * len(materialized))))
    if len(points) < minimum_track_points:
        artifacts = _write_tracking_debug(debug_path, materialized, background, frame_indices, points, None, None, None)
        result = _result(
            sample_id,
            False,
            {
                "fps": fps,
                "frame_count": len(materialized),
                "observed_track_points": len(points),
                "required_track_points": minimum_track_points,
                "segmentation": segmentation,
            },
            failure_reasons=["insufficient_block_track_points"],
            debug_artifacts=artifacts,
        )
        result["debug_artifacts"]["summary"] = _write_summary(debug_path, result)
        return result

    center, axis, axis_keep, axis_diagnostics = _robust_axis(points)
    # Estimate the *rendered* incline from observable geometry before fitting
    # accelerations.  The prompt's nominal 30 degrees is only a fallback: a
    # generated first frame can visibly render another angle.
    ramp_angle = _estimate_ramp_angle(axis, axis_diagnostics, background)
    frame_indices = frame_indices[axis_keep]
    points = points[axis_keep]
    order = np.argsort(frame_indices)
    frame_indices = frame_indices[order]
    points = points[order]
    diagonal = math.hypot(materialized[0].shape[1], materialized[0].shape[0])
    failures = []
    if len(points) < minimum_track_points:
        failures.append("too_many_off_axis_track_outliers")
    if axis_diagnostics["axis_span_px"] < max(8.0, 0.025 * diagonal):
        failures.append("block_displacement_too_small")
    if axis_diagnostics["axis_relative_rmse"] > 0.18:
        failures.append("trajectory_is_not_aligned_with_one_ramp_axis")

    raw_projection = (points - center) @ axis
    turn_index, orientation, turn_diagnostics = _find_turn(raw_projection)
    oriented_projection = raw_projection * orientation
    if turn_index is None:
        failures.append("turning_point_not_found")
        turn_frame = None
    else:
        turn_frame = int(frame_indices[turn_index])
        if turn_index + 1 < 7:
            failures.append("insufficient_ascent_observations")
        if len(points) - turn_index < 7:
            failures.append("insufficient_descent_observations")

    artifacts = _write_tracking_debug(
        debug_path,
        materialized,
        background,
        frame_indices,
        points,
        center,
        axis * orientation,
        turn_frame,
    )
    base_measurements = {
        "fps": fps,
        "frame_count": len(materialized),
        "observed_track_points": len(points),
        "segmentation": segmentation,
        "ramp_axis": axis_diagnostics | {"orientation_sign": float(orientation)},
        # Keep both flat fields (easy for CSV consumers) and the full nested
        # diagnostic object (useful for audit/review).
        "ramp_angle_deg": ramp_angle["ramp_angle_deg"],
        "observed_ramp_angle_deg": ramp_angle["observed_ramp_angle_deg"],
        "ramp_angle_source": ramp_angle["ramp_angle_source"],
        "ramp_angle_fallback": ramp_angle["ramp_angle_fallback"],
        "ramp_angle_confidence": ramp_angle["ramp_angle_confidence"],
        "ramp_angle_estimation": ramp_angle,
        "turn_detection": turn_diagnostics,
    }
    # Resolve upward-only motion from observed image coordinates; an unknown
    # turn caused by tracking loss is not assigned a physical score.
    upward=bool(len(points)>=minimum_track_points and points[0,1]-points[-1,1]>max(6.,.015*diagonal)
                and np.mean(np.diff(points[:,1])<=1.)>=.80
                and frame_indices[-1]>=len(materialized)-max(3,int(.1*len(materialized))))
    if turn_index is None and upward and failures==['turning_point_not_found']:
        base_measurements.update(observed_outcome='ascent_only',observed_outcome_physics_score=.1)
        result=_result(sample_id,True,base_measurements,metrics={'M1':None},debug_artifacts=artifacts)
        result['debug_artifacts']['summary']=_write_summary(debug_path,result)
        return result
    if failures or turn_index is None:
        result = _result(sample_id, False, base_measurements, failure_reasons=failures, debug_artifacts=artifacts)
        result["debug_artifacts"]["summary"] = _write_summary(debug_path, result)
        return result

    times = frame_indices.astype(np.float64) / fps
    shared_fit = _fit_shared_vertex(times, oriented_projection, turn_index)
    if shared_fit is None:
        failures.append("shared_turning_vertex_fit_failed")
        result = _result(sample_id, False, base_measurements, failure_reasons=failures, debug_artifacts=artifacts)
        result["debug_artifacts"]["summary"] = _write_summary(debug_path, result)
        return result
    turn_index = int(shared_fit["turn_index"])
    up_times = times[: turn_index + 1]
    down_times = times[turn_index:]
    up_fit = shared_fit["up"]
    down_fit = shared_fit["down"]
    if not math.isfinite(up_fit["acceleration_px_s2"]) or up_fit["acceleration_px_s2"] <= 1e-9:
        failures.append("ascent_acceleration_fit_is_degenerate")
    if not math.isfinite(down_fit["acceleration_px_s2"]) or down_fit["acceleration_px_s2"] <= 1e-9:
        failures.append("descent_acceleration_fit_is_degenerate")
    if up_fit["relative_rmse"] > 0.35:
        failures.append("ascent_quadratic_fit_unusable")
    if down_fit["relative_rmse"] > 0.35:
        failures.append("descent_quadratic_fit_unusable")

    acceleration_ratio = up_fit["acceleration_px_s2"] / max(down_fit["acceleration_px_s2"], 1e-12)
    if not math.isfinite(acceleration_ratio):
        failures.append("acceleration_ratio_is_not_finite")
    expected_ratio = ramp_angle.get("expected_acceleration_ratio_from_observed_angle")
    if expected_ratio is None or not math.isfinite(float(expected_ratio)):
        # This should only be reachable for an unexpectedly malformed angle
        # estimate.  Keep the raw fit and return a null score rather than
        # dividing by a fabricated denominator.
        failures.append("expected_acceleration_ratio_unavailable")
        expected_ratio = EXPECTED_ACCELERATION_RATIO
    turn_time = float(shared_fit["turn_time_s"])
    measurements = base_measurements | {
        "turn_frame": int(round(turn_time * fps)),
        "turn_time_s": turn_time,
        "t_up_s": turn_time - float(times[0]),
        "t_down_s": float(times[-1]) - turn_time,
        "a_up_px_s2": float(up_fit["acceleration_px_s2"]),
        "a_down_px_s2": float(down_fit["acceleration_px_s2"]),
        "acceleration_ratio": float(acceleration_ratio),
        "expected_acceleration_ratio": float(expected_ratio),
        "expected_acceleration_ratio_source": (
            "task_nominal_fallback" if ramp_angle["ramp_angle_fallback"] else "observed_ramp_angle"
        ),
        "shared_turn_coordinate_px": float(shared_fit["turn_coordinate_px"]),
        "shared_fit_inlier_fraction": float(shared_fit["inlier_fraction"]),
        "up_fit": {k: v for k, v in up_fit.items() if k not in {"prediction", "inliers", "coefficients"}}
        | {"coefficients": up_fit["coefficients"].tolist()},
        "down_fit": {k: v for k, v in down_fit.items() if k not in {"prediction", "inliers", "coefficients"}}
        | {"coefficients": down_fit["coefficients"].tolist()},
        "trajectory": [
            {
                "frame": int(frame_index),
                "time_s": float(time),
                "x_px": float(point[0]),
                "y_px": float(point[1]),
                "up_ramp_coordinate_px": float(position),
            }
            for frame_index, time, point, position in zip(frame_indices, times, points, oriented_projection)
        ],
    }
    artifacts["quadratic_fit_plot"] = _write_fit_plot(
        debug_path, times, oriented_projection, turn_index, up_fit, down_fit
    )
    metrics = {
        "M1": float(abs(acceleration_ratio / float(expected_ratio) - 1.0)),
        "M2_up_fit_relative_rmse": float(up_fit["relative_rmse"]),
        "M2_down_fit_relative_rmse": float(down_fit["relative_rmse"]),
        "M2_t_down_gt_t_up": float((times[-1] - turn_time) > (turn_time - times[0])),
    }
    result = _result(
        sample_id,
        not failures,
        measurements,
        metrics=metrics,
        failure_reasons=failures,
        debug_artifacts=artifacts,
    )
    result["debug_artifacts"]["summary"] = _write_summary(debug_path, result)
    return result


evaluate_p11 = evaluate


def evaluate_video(
    video_path: str | Path,
    debug_dir: str | Path,
    sample_id: int = 0,
) -> dict[str, Any]:
    """Decode a video and call :func:`evaluate`."""

    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return evaluate(frames, fps, debug_dir, sample_id=sample_id)


__all__ = [
    "evaluate",
    "evaluate_p11",
    "evaluate_video",
    "expected_acceleration_ratio",
]
