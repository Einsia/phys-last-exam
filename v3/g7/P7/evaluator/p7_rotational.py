"""Deterministic evaluator for P7 (solid sphere versus thin rolling ring).

Primary metric M1 is extracted from two independent image tracks.  Optional
rolling-without-slip metrics use a visible color fiducial; when no stable
fiducial exists the evaluator reports that metric as unavailable rather than
inventing an angular velocity.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Sequence

import cv2
import numpy as np

V3_ROOT = Path(__file__).resolve().parents[3]
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from shared.g7_cv_common import (
        MeasurementFailure, annularity_score, detect_circle_candidates,
        draw_track_overlay, estimate_camera_motion, result,
        select_circle_pair, smooth_1d, track_template_circle,
        validate_and_resize_frames, write_json, write_track_csv,
    )


TASK_ID = "P7"
EXPECTED_TIME_RATIO = math.sqrt(10.0 / 7.0)

# A rolling-time ratio is identifiable from a short visible segment; it does
# not require that either object reach the end of the ramp.  The old evaluator
# required four radii of travel, which turned otherwise measurable clips into
# extraction failures.  Keep only a small signal-to-noise floor here and leave
# the physics acceptance band to ``evaluator.physics``.
MIN_MEASUREMENT_SPAN_RADII = 0.75
MIN_TRACK_VALID_POINTS = 12
MIN_TRACK_COVERAGE = 0.35


def _feature_track_circle(
    frames: Sequence[np.ndarray], circle: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Track object texture with sparse forward/backward optical flow.

    GPT-generated clips often make a template match lock onto the static ramp
    once an object moves a few radii.  Features seeded inside the initial
    circle follow the object instead and naturally become invalid when it
    leaves the frame.  We intentionally retain the raw NaN mask; callers may
    interpolate only short *internal* gaps and must not extrapolate a vanished
    object through the rest of the clip.
    """

    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    radius = float(circle["r"])
    cx, cy = float(circle["x"]), float(circle["y"])
    mask = np.zeros_like(gray[0], dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), max(4, int(round(0.95 * radius))), 255, -1)
    initial = cv2.goodFeaturesToTrack(
        gray[0], mask=mask, maxCorners=80, qualityLevel=0.01,
        minDistance=3, blockSize=5,
    )
    centers = np.full((len(frames), 2), np.nan, dtype=np.float64)
    confidence = np.full(len(frames), np.nan, dtype=np.float64)
    centers[0] = (cx, cy)
    confidence[0] = 1.0
    diagnostics: dict[str, Any] = {
        "track_method": "lk_feature",
        "feature_count_initial": 0 if initial is None else int(len(initial)),
    }
    if initial is None or len(initial) < 4:
        diagnostics.update(
            track_coverage=1.0 / max(1, len(frames)),
            valid_points=1,
            longest_missing_run=max(0, len(frames) - 1),
            feature_tracking_failure="too_few_initial_features",
        )
        return centers, confidence, diagnostics

    points = initial.reshape(-1, 2).astype(np.float32)
    previous = gray[0]
    valid = np.zeros(len(frames), dtype=bool)
    valid[0] = True
    flow_errors: list[float] = []
    for index in range(1, len(frames)):
        next_points, status, forward_error = cv2.calcOpticalFlowPyrLK(
            previous, gray[index], points.reshape(-1, 1, 2), None,
            winSize=(31, 31), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_points is None:
            break
        next_points = next_points.reshape(-1, 2)
        keep = status.reshape(-1).astype(bool)
        # Forward/backward consistency rejects points that have fallen onto a
        # visually similar ramp edge or a newly exposed background region.
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray[index], previous, next_points.reshape(-1, 1, 2), None,
            winSize=(31, 31), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if back_points is not None:
            backward_error = np.linalg.norm(back_points.reshape(-1, 2) - points, axis=1)
            keep &= back_status.reshape(-1).astype(bool) & (backward_error < 2.5)
            if keep.any():
                flow_errors.extend(backward_error[keep].astype(float).tolist())
        if keep.sum() >= 4:
            displacement = next_points - points
            median_displacement = np.median(displacement[keep], axis=0)
            deviation = np.linalg.norm(displacement - median_displacement, axis=1)
            robust_scale = float(np.median(deviation[keep]))
            keep &= deviation < max(4.0, 2.5 * robust_scale + 2.0)
        if keep.sum() < 4:
            # Tracking is a contiguous prefix.  Do not turn a disappeared
            # object into a stationary extrapolated object.
            break
        points = next_points[keep].astype(np.float32)
        centers[index] = np.median(points, axis=0)
        forward_values = np.asarray(forward_error).reshape(-1) if forward_error is not None else np.zeros(len(next_points))
        confidence[index] = 1.0 / (1.0 + float(np.median(forward_values[keep])))
        valid[index] = True
        previous = gray[index]

    valid_indices = np.flatnonzero(valid)
    diagnostics.update(
        track_coverage=float(valid.mean()),
        valid_points=int(valid.sum()),
        valid_frame_start=int(valid_indices[0]) if len(valid_indices) else None,
        valid_frame_end=int(valid_indices[-1]) if len(valid_indices) else None,
        longest_missing_run=int(len(frames) - valid_indices[-1] - 1) if len(valid_indices) else len(frames),
        median_backward_error=float(np.median(flow_errors)) if flow_errors else None,
        feature_count_final=int(len(points)),
    )
    return centers, confidence, diagnostics


def _interpolate_internal_track(track: np.ndarray, *, max_gap: int | None = None) -> np.ndarray:
    """Fill only gaps bracketed by observations, preserving leading/trailing NaNs."""

    track = np.asarray(track, dtype=np.float64).copy()
    valid = np.isfinite(track[:, 0]) & np.isfinite(track[:, 1])
    if int(valid.sum()) < MIN_TRACK_VALID_POINTS:
        raise MeasurementFailure("track_has_too_few_valid_points", valid_points=int(valid.sum()))
    indices = np.arange(len(track), dtype=np.float64)
    valid_indices = np.flatnonzero(valid)
    first, last = int(valid_indices[0]), int(valid_indices[-1])
    if max_gap is not None:
        gaps = np.diff(valid_indices) - 1
        if len(gaps) and int(gaps.max()) > int(max_gap):
            raise MeasurementFailure(
                "track_internal_gap_too_long", longest_gap=int(gaps.max()), allowed_gap=int(max_gap)
            )
    for axis in range(2):
        track[first:last + 1, axis] = np.interp(
            indices[first:last + 1], valid_indices.astype(np.float64), track[valid_indices, axis]
        )
    track[:first] = np.nan
    track[last + 1:] = np.nan
    return track


def _track_span_radii(track: np.ndarray, radius: float) -> float:
    valid = np.isfinite(track[:, 0]) & np.isfinite(track[:, 1])
    if int(valid.sum()) < 2:
        return 0.0
    points = track[valid]
    return float(np.linalg.norm(points[-1] - points[0]) / max(radius, 1e-6))


def _crossing_time(values: np.ndarray, level: float, fps: float) -> float:
    monotone = np.maximum.accumulate(values)
    hits = np.flatnonzero(monotone >= level)
    if not len(hits):
        raise MeasurementFailure("object_did_not_reach_common_measurement_distance", level=level)
    index = int(hits[0])
    if index == 0:
        return 0.0
    y0, y1 = float(monotone[index - 1]), float(monotone[index])
    fraction = 0.0 if y1 <= y0 else float(np.clip((level - y0) / (y1 - y0), 0.0, 1.0))
    return (index - 1 + fraction) / fps


def _robust_quadratic_release_time(t: np.ndarray, s: np.ndarray, cutoff: float) -> float | None:
    use = (s >= -0.05 * cutoff) & (s <= cutoff)
    if use.sum() < 8:
        return None
    keep = use.copy()
    coefficients = None
    for _ in range(5):
        coefficients = np.polyfit(t[keep], s[keep], 2)
        residual = s - np.polyval(coefficients, t)
        med = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - med))) + 1e-6
        new_keep = use & (np.abs(residual - med) < 4.0 * 1.4826 * mad)
        if new_keep.sum() < 8 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if coefficients is None or coefficients[0] <= 1e-8:
        return None
    value = -coefficients[1] / (2.0 * coefficients[0])
    return float(value) if -0.5 <= value <= t[-1] else None


def _translation_measurements(
    tracks: Sequence[np.ndarray], radii: Sequence[float], fps: float,
    *, raw_valid_masks: Sequence[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    smoothed_tracks = []
    displacements = []
    directions = []
    observed_valid_masks: list[np.ndarray] = []
    valid_ranges: list[tuple[int, int]] = []
    for track_index, track in enumerate(tracks):
        raw_valid = (
            np.asarray(raw_valid_masks[track_index], dtype=bool)
            if raw_valid_masks is not None
            else np.isfinite(track[:, 0]) & np.isfinite(track[:, 1])
        )
        # The coordinates may have had short internal gaps filled by the
        # caller, so use the supplied mask for provenance but the finite
        # coordinates for numerical preparation.
        coordinate_valid = np.isfinite(track[:, 0]) & np.isfinite(track[:, 1])
        if int(raw_valid.sum()) < MIN_TRACK_VALID_POINTS:
            raise MeasurementFailure("track_has_too_few_valid_points", valid_points=int(raw_valid.sum()))
        valid_indices = np.flatnonzero(coordinate_valid)
        first, last = int(valid_indices[0]), int(valid_indices[-1])
        # Smooth only the observed interval.  In particular, do not smooth or
        # extrapolate a trailing NaN tail after an object leaves the frame.
        smoothed = np.full_like(track, np.nan, dtype=np.float64)
        for axis in range(2):
            segment = track[first:last + 1, axis].astype(np.float64)
            finite = np.isfinite(segment)
            if not finite.all():
                segment_indices = np.flatnonzero(finite)
                if len(segment_indices) < MIN_TRACK_VALID_POINTS:
                    raise MeasurementFailure("track_has_too_few_valid_points", valid_points=int(finite.sum()))
                segment = np.interp(np.arange(len(segment)), segment_indices, segment[finite])
            smoothed[first:last + 1, axis] = smooth_1d(segment, 7)
        smoothed_tracks.append(smoothed)
        observed_valid_masks.append(raw_valid)
        valid_ranges.append((first, last))
        observed = smoothed[first:last + 1]
        head = np.median(observed[: max(3, len(observed) // 20)], axis=0)
        tail = np.median(observed[-max(3, len(observed) // 10) :], axis=0)
        directions.append(tail - head)
    direction = np.sum([v / max(np.linalg.norm(v), 1e-9) for v in directions], axis=0)
    if np.linalg.norm(direction) < 0.5:
        direction = directions[int(np.argmax([np.linalg.norm(v) for v in directions]))]
    direction = direction / max(np.linalg.norm(direction), 1e-9)
    if np.mean([np.dot(v, direction) for v in directions]) < 0:
        direction *= -1
    # The task asserts equal outer radii.  A shared pixel scale keeps the time
    # gate at the same image-space distance and prevents independent noisy
    # Hough radius estimates from biasing M1.
    common_radius = float(np.median(radii))
    for track, (first, last) in zip(smoothed_tracks, valid_ranges):
        longitudinal = np.full(len(track), np.nan, dtype=np.float64)
        longitudinal[first:last + 1] = track[first:last + 1] @ direction
        start = float(np.median(longitudinal[first:first + max(3, (last - first + 1) // 30)]))
        displacement = (longitudinal - start) / max(common_radius, 1e-6)
        displacement[first:last + 1] = smooth_1d(displacement[first:last + 1], 7)
        displacements.append(displacement)
    spans = []
    violation_rates = []
    for displacement, (first, last) in zip(displacements, valid_ranges):
        observed = displacement[first:last + 1]
        baseline_count = max(3, len(observed) // 20)
        spans.append(float(np.percentile(observed, 96) - np.percentile(observed[:baseline_count], 20)))
        violation_rates.append(float(np.mean(np.diff(observed) < -0.035)) if len(observed) > 1 else 1.0)
    if min(spans) < MIN_MEASUREMENT_SPAN_RADII:
        raise MeasurementFailure(
            "rolling_motion_not_resolved", travel_span_radii=spans,
            required=MIN_MEASUREMENT_SPAN_RADII,
        )
    # Non-monotonicity is useful evidence about a generated clip, but it must
    # not erase a measurable time series.  Report it as a quality flag; the
    # strict physics layer consumes metric_failures separately.
    quality_flags = []
    if max(violation_rates) > 0.28:
        quality_flags.append("nonmonotonic_rolling_tracks")
    common_target = 0.78 * min(spans)
    start_level = 0.07 * common_target
    times = []
    releases = []
    for displacement, (first, last) in zip(displacements, valid_ranges):
        observed = displacement[first:last + 1]
        begin = _crossing_time(observed, start_level, fps)
        end = _crossing_time(observed, common_target, fps)
        if end <= begin:
            raise MeasurementFailure("nonpositive_rolling_duration", begin=begin, end=end)
        times.append(end - begin)
        t = np.arange(first, last + 1, dtype=np.float64) / fps
        releases.append(_robust_quadratic_release_time(t, observed, common_target))
    return smoothed_tracks, displacements, {
        "motion_direction_xy": direction,
        "travel_span_radii": spans,
        "monotonicity_violation_rate": violation_rates,
        "quality_flags": quality_flags,
        "raw_valid_masks": [mask.astype(np.uint8) for mask in observed_valid_masks],
        "valid_frame_ranges": [[start, end] for start, end in valid_ranges],
        "common_target_radii": common_target,
        "start_threshold_radii": start_level,
        "shared_radius_scale_px": common_radius,
        "measured_durations_s": times,
        "quadratic_release_times_s": releases,
    }


def _marker_reference(frame: np.ndarray, center: np.ndarray, radius: float) -> tuple[np.ndarray, dict[str, Any]] | None:
    h, w = frame.shape[:2]
    half = int(math.ceil(1.18 * radius))
    cx, cy = np.rint(center).astype(int)
    x0, x1 = max(0, cx - half), min(w, cx + half + 1)
    y0, y1 = max(0, cy - half), min(h, cy + half + 1)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    radial = np.hypot(xx - center[0], yy - center[1])
    mask = (radial >= 0.18 * radius) & (radial <= 1.12 * radius)
    pixels = lab[mask]
    if len(pixels) < 40:
        return None
    cv2.setRNGSeed(7007)
    compactness, labels, centers = cv2.kmeans(
        pixels, min(5, max(2, len(pixels) // 30)), None,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2), 1,
        cv2.KMEANS_PP_CENTERS,
    )
    del compactness
    coords = np.column_stack(np.nonzero(mask))[:, ::-1].astype(np.float64)
    best = None
    for cluster_index, color in enumerate(centers):
        selected = labels.reshape(-1) == cluster_index
        fraction = float(selected.mean())
        if not 0.003 <= fraction <= 0.18 or selected.sum() < 3:
            continue
        centroid_local = coords[selected].mean(axis=0)
        centroid = centroid_local + np.array([x0, y0])
        vector = centroid - center
        radial_fraction = float(np.linalg.norm(vector) / max(radius, 1e-6))
        spread = float(np.sqrt(np.mean(np.sum((coords[selected] - centroid_local) ** 2, axis=1))) / radius)
        if not 0.25 <= radial_fraction <= 1.18 or spread > 0.52:
            continue
        color_distances = np.linalg.norm(centers - color, axis=1)
        color_distances[cluster_index] = np.inf
        contrast = float(np.min(color_distances))
        sat = float(np.mean(hsv[mask][selected, 1]))
        score = contrast / 28.0 + sat / 150.0 + radial_fraction - spread - 1.5 * fraction
        if best is None or score > best[0]:
            best = (score, color.astype(np.float32), fraction, radial_fraction, spread, contrast, sat)
    if best is None or best[0] < 1.0:
        return None
    return best[1], {
        "initial_marker_score": best[0], "initial_marker_fraction": best[2],
        "initial_marker_radius_fraction": best[3], "initial_marker_spread": best[4],
        "initial_marker_color_contrast": best[5], "initial_marker_saturation": best[6],
    }


def _marker_phase(frames: Sequence[np.ndarray], track: np.ndarray, radius: float) -> tuple[np.ndarray, dict[str, Any]]:
    reference = _marker_reference(frames[0], track[0], radius)
    if reference is None:
        raise MeasurementFailure("rotation_fiducial_not_detected")
    color, diagnostics = reference
    angles = np.full(len(frames), np.nan, dtype=np.float64)
    weights_out = []
    for index, (frame, center) in enumerate(zip(frames, track)):
        if not np.isfinite(center).all():
            # A trailing NaN means that the object left the field of view (or
            # that optical flow lost it).  It is an honest missing sample, not
            # evidence for a stationary object.
            continue
        h, w = frame.shape[:2]
        half = int(math.ceil(1.18 * radius)); cx, cy = np.rint(center).astype(int)
        x0, x1 = max(0, cx - half), min(w, cx + half + 1)
        y0, y1 = max(0, cy - half), min(h, cy + half + 1)
        patch = frame[y0:y1, x0:x1]
        if patch.size == 0 or patch.shape[0] < 3 or patch.shape[1] < 3:
            continue
        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        radial = np.hypot(xx - center[0], yy - center[1])
        valid = (radial >= 0.16 * radius) & (radial <= 1.15 * radius)
        distance = np.linalg.norm(lab - color.reshape(1, 1, 3), axis=2)
        sigma = 16.0
        weights = np.exp(-0.5 * (distance / sigma) ** 2) * valid
        threshold = float(np.percentile(weights[valid], 91)) if valid.any() else 1.0
        weights *= weights >= max(0.18, threshold)
        total = float(weights.sum())
        if total < 1.2:
            continue
        marker = np.array([(weights * xx).sum() / total, (weights * yy).sum() / total])
        vector = marker - center
        radial_fraction = float(np.linalg.norm(vector) / radius)
        if 0.20 <= radial_fraction <= 1.20:
            angles[index] = math.atan2(vector[1], vector[0])
            weights_out.append(total)
    valid = np.isfinite(angles)
    if valid.mean() < 0.55:
        raise MeasurementFailure("rotation_fiducial_track_coverage_too_low", coverage=float(valid.mean()))
    indices = np.arange(len(angles))
    unwrapped_valid = np.unwrap(angles[valid])
    phase = np.interp(indices, indices[valid], unwrapped_valid)
    increments = np.diff(phase)
    if np.percentile(np.abs(increments), 95) > 1.8:
        raise MeasurementFailure("rotation_fiducial_phase_has_discontinuities")
    diagnostics.update(
        marker_track_coverage=float(valid.mean()),
        marker_phase_span_rad=float(np.ptp(phase)),
        median_marker_weight=float(np.median(weights_out)) if weights_out else 0.0,
    )
    return phase, diagnostics


def _rolling_ratio(displacement_px: np.ndarray, phase: np.ndarray, radius: float) -> tuple[float, dict[str, Any]]:
    x = displacement_px.astype(np.float64)
    y = phase.astype(np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 12 or np.ptp(y[keep]) < 0.8:
        raise MeasurementFailure("rotation_phase_span_too_small")
    for _ in range(5):
        matrix = np.column_stack([x[keep], np.ones(keep.sum())])
        slope, intercept = np.linalg.lstsq(matrix, y[keep], rcond=None)[0]
        residual = y - (slope * x + intercept)
        med = float(np.median(residual[keep])); mad = float(np.median(np.abs(residual[keep] - med))) + 1e-6
        new_keep = np.isfinite(residual) & (np.abs(residual - med) < 4.0 * 1.4826 * mad)
        if new_keep.sum() < 12 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if abs(slope) * radius < 0.08:
        raise MeasurementFailure("angular_motion_not_resolved", phase_per_radius=float(abs(slope) * radius))
    correlation = float(abs(np.corrcoef(x[keep], y[keep])[0, 1]))
    if correlation < 0.45:
        raise MeasurementFailure("rotation_translation_correlation_too_low", correlation=correlation)
    ratio = 1.0 / (abs(float(slope)) * radius)
    return ratio, {"phase_per_pixel": float(slope), "fit_inlier_fraction": float(keep.mean()), "correlation": correlation}


def evaluate(
    frames: Sequence[np.ndarray], fps: float, debug_dir: str | Path | None, sample_id: int | str = 0
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    measurements: dict[str, Any] = {}
    artifacts: list[Path] = []
    debug = Path(debug_dir) if debug_dir is not None else None
    try:
        work_frames, scale = validate_and_resize_frames(frames, fps, min_frames=30)
        diagnostics.update(frame_count=len(work_frames), fps=float(fps), processing_scale=scale)
        camera = estimate_camera_motion(work_frames)
        diagnostics["camera_motion"] = camera
        if camera.get("available") and camera.get("median_ransac_inlier_fraction", 0.0) >= 0.60 and (
            camera["max_translation_fraction"] > 0.045 or camera["max_rotation_deg"] > 2.0
            or camera["max_scale_change"] > 0.045
        ):
            raise MeasurementFailure("camera_motion_too_large", **camera)
        candidates = detect_circle_candidates(work_frames[0])
        first, second, selection = select_circle_pair(work_frames, candidates, task=TASK_ID)
        diagnostics["object_selection"] = selection
        raw_tracks = []
        confidences = []
        track_methods: list[str] = []
        track_quality_failures: list[str] = []
        radii = [first["r"], second["r"]]
        for circle in (first, second):
            template_track, template_confidence, template_diag = track_template_circle(
                work_frames, circle, search_radius_factor=4.0
            )
            feature_track, feature_confidence, feature_diag = _feature_track_circle(work_frames, circle)
            feature_span = _track_span_radii(feature_track, circle["r"])
            feature_points = int(feature_diag.get("valid_points", 0))
            feature_coverage = float(feature_diag.get("track_coverage", 0.0))
            template_points = int(np.isfinite(template_track[:, 0]).sum())
            template_coverage = float(template_diag.get("track_coverage", 0.0))

            # Prefer LK when it has a reasonably long, self-consistent track.
            # Template matching remains the fallback for textureless clips
            # (notably some synthetic renders).  This choice is made per
            # object, and both raw tracks are retained in diagnostics.
            if feature_points >= MIN_TRACK_VALID_POINTS and feature_coverage >= 0.75 and feature_span >= MIN_MEASUREMENT_SPAN_RADII:
                track, confidence, selected_method = feature_track, feature_confidence, "lk_feature"
                selected_diag = feature_diag
            elif template_points >= MIN_TRACK_VALID_POINTS and template_coverage >= MIN_TRACK_COVERAGE:
                track, confidence, selected_method = template_track, template_confidence, "template"
                selected_diag = template_diag
            elif feature_points >= MIN_TRACK_VALID_POINTS:
                track, confidence, selected_method = feature_track, feature_confidence, "lk_feature_partial"
                selected_diag = feature_diag
            elif template_points >= MIN_TRACK_VALID_POINTS:
                track, confidence, selected_method = template_track, template_confidence, "template_partial"
                selected_diag = template_diag
            else:
                raise MeasurementFailure(
                    "rolling_object_track_coverage_too_low",
                    template_track_coverage=template_coverage,
                    feature_track_coverage=feature_coverage,
                    template_valid_points=template_points,
                    feature_valid_points=feature_points,
                )
            raw_tracks.append(track)
            confidences.append(confidence)
            track_methods.append(selected_method)
            diagnostics.setdefault("tracks", []).append(
                {
                    "selected_method": selected_method,
                    "selected_span_radii": _track_span_radii(track, circle["r"]),
                    "track_coverage": float(selected_diag.get("track_coverage", 0.0)),
                    "valid_points": int(selected_diag.get("valid_points", np.isfinite(track[:, 0]).sum())),
                    "selected": selected_diag,
                    "template": template_diag,
                    "lk_feature": feature_diag,
                    "quality_flags": (
                        ["track_coverage_below_preferred"]
                        if float(selected_diag.get("track_coverage", 0.0)) < 0.75 else []
                    ),
                    "raw_valid_mask": np.isfinite(track[:, 0]) & np.isfinite(track[:, 1]),
                }
            )
            if float(selected_diag.get("track_coverage", 0.0)) < 0.75:
                track_quality_failures.append(
                    f"track_{len(raw_tracks)-1}_coverage_below_preferred"
                )
        tracks = [
            _interpolate_internal_track(track, max_gap=max(10, len(work_frames) // 7))
            for track in raw_tracks
        ]
        raw_valid_masks = [
            np.isfinite(track[:, 0]) & np.isfinite(track[:, 1]) for track in raw_tracks
        ]
        smoothed, displacement_radii, translation = _translation_measurements(
            tracks, radii, fps, raw_valid_masks=raw_valid_masks
        )
        diagnostics["translation"] = translation
        ring_scores = [annularity_score(work_frames[0], first), annularity_score(work_frames[0], second)]
        identity_gap = abs(ring_scores[0] - ring_scores[1])
        if identity_gap < 0.055:
            raise MeasurementFailure("sphere_ring_identity_ambiguous", annularity_scores=ring_scores, required_gap=0.055)
        ring_index = int(np.argmax(ring_scores)); sphere_index = 1 - ring_index
        durations = translation["measured_durations_s"]
        time_ratio = float(durations[ring_index] / durations[sphere_index])
        radius_ratio = float(radii[ring_index] / radii[sphere_index])
        release_times = translation["quadratic_release_times_s"]
        measurements.update(
            fps=float(fps), frame_count=len(work_frames), identity_method="inner_rim_and_hole_topology",
            annularity_scores=ring_scores, identity_confidence_gap=identity_gap,
            sphere_track_index=sphere_index, ring_track_index=ring_index,
            sphere_radius_px=float(radii[sphere_index]), ring_radius_px=float(radii[ring_index]),
            outer_radius_ratio=radius_ratio,
            t_sphere_s=float(durations[sphere_index]), t_ring_s=float(durations[ring_index]),
            time_ratio=time_ratio, expected_time_ratio=EXPECTED_TIME_RATIO,
            release_time_sphere_s=release_times[sphere_index], release_time_ring_s=release_times[ring_index],
            common_measurement_distance_radii=translation["common_target_radii"],
            sphere_track_method=track_methods[sphere_index], ring_track_method=track_methods[ring_index],
            sphere_raw_track_coverage=float(np.mean(np.isfinite(raw_tracks[sphere_index]).all(axis=1))),
            ring_raw_track_coverage=float(np.mean(np.isfinite(raw_tracks[ring_index]).all(axis=1))),
            sphere_raw_valid_frame_range=translation["valid_frame_ranges"][sphere_index],
            ring_raw_valid_frame_range=translation["valid_frame_ranges"][ring_index],
        )
        metric_failures: list[str] = list(track_quality_failures)
        metric_failures.extend(translation.get("quality_flags", []))
        measurements["measurement_quality"] = {
            "extract_semantics": "finite_common_distance_timing_available",
            "strict_physics_ready": not bool(track_quality_failures or translation.get("quality_flags")),
            "quality_flags": list(metric_failures),
            "raw_observations_preserved": True,
            "raw_track_artifact": "p7_raw_tracks.csv",
        }
        rolling_values: dict[int, float | None] = {sphere_index: None, ring_index: None}
        marker_diagnostics: dict[str, Any] = {}
        for index, label in ((sphere_index, "sphere"), (ring_index, "ring")):
            try:
                phase, marker_diag = _marker_phase(work_frames, smoothed[index], radii[index])
                longitudinal_px = displacement_radii[index] * radii[index]
                rolling, fit_diag = _rolling_ratio(longitudinal_px, phase, radii[index])
                rolling_values[index] = rolling
                marker_diagnostics[label] = {**marker_diag, **fit_diag}
            except MeasurementFailure as marker_failure:
                metric_failures.append(f"{label}:{marker_failure.code}")
                marker_diagnostics[label] = {"failure_reason": marker_failure.code, **marker_failure.details}
        measurements["rolling_ratio_sphere"] = rolling_values[sphere_index]
        measurements["rolling_ratio_ring"] = rolling_values[ring_index]
        measurements["measurement_quality"]["quality_flags"] = list(metric_failures)
        measurements["measurement_quality"]["strict_physics_ready"] = not bool(metric_failures)
        diagnostics["rotation_tracking"] = marker_diagnostics
        m1 = abs(time_ratio / EXPECTED_TIME_RATIO - 1.0)
        sphere_m2 = None if rolling_values[sphere_index] is None else abs(rolling_values[sphere_index] - 1.0)
        ring_m2 = None if rolling_values[ring_index] is None else abs(rolling_values[ring_index] - 1.0)
        available_m2 = [x for x in (sphere_m2, ring_m2) if x is not None]
        metrics = {
            "M1": m1,
            "time_ratio_relative_error": m1,
            "M2_sphere": sphere_m2,
            "M2_ring": ring_m2,
            "M2": float(np.mean(available_m2)) if len(available_m2) == 2 else None,
            "outer_radius_relative_difference": abs(radius_ratio - 1.0),
            "release_time_difference_s": None if None in release_times else abs(release_times[0] - release_times[1]),
        }
        if debug is not None:
            debug.mkdir(parents=True, exist_ok=True)
            overlay = draw_track_overlay(
                work_frames[0], [smoothed[sphere_index], smoothed[ring_index]],
                [(255, 160, 0), (0, 220, 255)], labels=["sphere", "ring"],
            )
            overlay_path = debug / "p7_tracks_overlay.png"; cv2.imwrite(str(overlay_path), overlay); artifacts.append(overlay_path)
            raw_overlay = draw_track_overlay(
                work_frames[0], [raw_tracks[sphere_index], raw_tracks[ring_index]],
                [(180, 60, 255), (60, 220, 60)], labels=["sphere raw", "ring raw"],
            )
            raw_overlay_path = debug / "p7_raw_tracks_overlay.png"
            cv2.imwrite(str(raw_overlay_path), raw_overlay); artifacts.append(raw_overlay_path)
            csv_path = write_track_csv(
                debug / "p7_tracks.csv",
                [
                    "frame", "time_s", "sphere_x", "sphere_y", "ring_x", "ring_y",
                    "sphere_s_over_R", "ring_s_over_R", "sphere_raw_valid", "ring_raw_valid",
                ],
                [np.arange(len(work_frames)), np.arange(len(work_frames)) / fps,
                 smoothed[sphere_index][:, 0], smoothed[sphere_index][:, 1],
                 smoothed[ring_index][:, 0], smoothed[ring_index][:, 1],
                 displacement_radii[sphere_index], displacement_radii[ring_index],
                 translation["raw_valid_masks"][sphere_index], translation["raw_valid_masks"][ring_index]],
            ); artifacts.append(csv_path)
            # Keep the un-interpolated observations as a separate audit trail.
            # ``p7_tracks.csv`` contains the smoothed series used for timing;
            # this file retains NaNs, raw coordinates, and validity bits so a
            # reviewer can distinguish observed pixels from interpolated data.
            raw_csv_path = write_track_csv(
                debug / "p7_raw_tracks.csv",
                [
                    "frame", "time_s", "sphere_raw_x", "sphere_raw_y",
                    "ring_raw_x", "ring_raw_y", "sphere_raw_valid", "ring_raw_valid",
                ],
                [
                    np.arange(len(work_frames)), np.arange(len(work_frames)) / fps,
                    raw_tracks[sphere_index][:, 0], raw_tracks[sphere_index][:, 1],
                    raw_tracks[ring_index][:, 0], raw_tracks[ring_index][:, 1],
                    np.isfinite(raw_tracks[sphere_index][:, 0]).astype(np.float64),
                    np.isfinite(raw_tracks[ring_index][:, 0]).astype(np.float64),
                ],
            ); artifacts.append(raw_csv_path)
            diagnostic_path = write_json(debug / "p7_diagnostics.json", {"measurements": measurements, "metrics": metrics, "diagnostics": diagnostics})
            artifacts.append(diagnostic_path)
        return result(TASK_ID, sample_id, True, measurements, metrics, metric_failures=metric_failures, diagnostics=diagnostics, debug_artifacts=artifacts)
    except MeasurementFailure as failure:
        if debug is not None:
            debug.mkdir(parents=True, exist_ok=True)
            failure_path = write_json(debug / "p7_failure.json", {"failure_reason": failure.code, "failure_details": failure.details, "diagnostics": diagnostics})
            artifacts.append(failure_path)
            if frames:
                preview_path = debug / "p7_failure_frame.png"; cv2.imwrite(str(preview_path), frames[0]); artifacts.append(preview_path)
        return result(TASK_ID, sample_id, False, measurements, {"M1": None, "M2": None}, failure=failure, diagnostics=diagnostics, debug_artifacts=artifacts)


evaluate_p7 = evaluate
