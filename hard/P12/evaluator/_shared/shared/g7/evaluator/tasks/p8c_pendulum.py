"""Deterministic evaluator for P8c, the finite-amplitude pendulum task."""
from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

try:
    from .g7_cv_common import (
        MeasurementFailure, detect_circle_candidates, draw_track_overlay,
        estimate_camera_motion, interpolate_track, result, robust_circle_fit,
        select_circle_pair, smooth_1d, track_template_circle,
        validate_and_resize_frames, write_json, write_track_csv,
    )
except ImportError:  # pragma: no cover
    from g7_cv_common import (
        MeasurementFailure, detect_circle_candidates, draw_track_overlay,
        estimate_camera_motion, interpolate_track, result, robust_circle_fit,
        select_circle_pair, smooth_1d, track_template_circle,
        validate_and_resize_frames, write_json, write_track_csv,
    )


TASK_ID = "P8c"


def _load_task_parameters() -> tuple[float, float, float]:
    """Read the two release angles and expected ratio from tasks.json.

    Keeping these values in the task manifest (rather than duplicating a
    stale amplitude constant in the evaluator) makes a revised amplitude pair
    auditable and prevents the first-frame generator and scorer drifting apart.
    The fallback is only for importing this module in isolation.
    """

    default = (15.0, 30.0, 1.0130520869)
    try:
        task_path = Path(__file__).resolve().parents[2] / "tasks.json"
        payload = json.loads(task_path.read_text(encoding="utf-8"))
        spec = next(item for item in payload["tasks"] if item["task_id"] == TASK_ID)
        angles = spec.get("release_angles_deg", [15.0, 30.0])
        if len(angles) != 2:
            raise ValueError("release_angles_deg must contain exactly two angles")
        small, large = sorted(float(value) for value in angles)
        expected = float(spec["expected"]["period_ratio"])
        return small, large, expected
    except (OSError, KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError):
        return default


SMALL_ANGLE_DEG, LARGE_ANGLE_DEG, TASK_SPEC_EXPECTED_RATIO = _load_task_parameters()


def _angle_label(angle_deg: float) -> str:
    """Return a stable schema label such as ``15deg`` or ``30deg``."""

    return f"{angle_deg:g}deg"


SMALL_LABEL = _angle_label(SMALL_ANGLE_DEG)
LARGE_LABEL = _angle_label(LARGE_ANGLE_DEG)

# These values describe measurement quality, not the declared physics law.
# They are intentionally kept separate from extraction success: a clip may
# contain enough visible motion to estimate a period even when its bob
# identity, coverage, or number of cycles is not strong enough for a strict
# physics pass.
TRACK_COVERAGE_QUALITY_FLOOR = 0.72
# Do not encode "two complete cycles" as an extraction prerequisite.  A clip
# with one visible cycle can still yield a period estimate; the established
# two-cycle criterion is retained only as a quality flag below.
MIN_SPECTRAL_CYCLES_FOR_MEASUREMENT = 0.90


def complete_elliptic_k_agm(modulus: float) -> float:
    """K(k) from the arithmetic-geometric mean, without SciPy."""

    if not 0.0 <= modulus < 1.0:
        raise ValueError("elliptic modulus must satisfy 0 <= k < 1")
    a, b = 1.0, math.sqrt(1.0 - modulus * modulus)
    for _ in range(64):
        next_a = 0.5 * (a + b)
        next_b = math.sqrt(a * b)
        if abs(next_a - next_b) < 1e-15 * next_a:
            a = next_a
            break
        a, b = next_a, next_b
    return math.pi / (2.0 * a)


def exact_period_factor(angle_deg: float) -> float:
    return complete_elliptic_k_agm(math.sin(math.radians(angle_deg) / 2.0))


EXPECTED_PERIOD_RATIO = exact_period_factor(LARGE_ANGLE_DEG) / exact_period_factor(SMALL_ANGLE_DEG)


def _principal_signal(track: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = track - np.mean(track, axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    signal = centered @ axis
    signal = smooth_1d(signal, 7)
    if signal[0] < 0:
        signal *= -1; axis *= -1
    return signal, axis


def _fit_sinusoid(t: np.ndarray, y: np.ndarray, frequency: float) -> tuple[float, np.ndarray]:
    omega = 2.0 * math.pi * frequency
    matrix = np.column_stack([np.ones(len(t)), t, np.cos(omega * t), np.sin(omega * t)])
    coefficients, _, _, _ = np.linalg.lstsq(matrix, y, rcond=None)
    fitted = matrix @ coefficients
    scale = float(np.sqrt(np.mean((y - np.mean(y)) ** 2))) + 1e-9
    return float(np.sqrt(np.mean((y - fitted) ** 2)) / scale), coefficients


def _period_from_signal(signal: np.ndarray, fps: float) -> dict[str, Any]:
    n = len(signal); t = np.arange(n, dtype=np.float64) / fps
    centered = signal - np.mean(signal)
    amplitude = 0.5 * float(np.percentile(signal, 98) - np.percentile(signal, 2))
    if amplitude < 2.5:
        raise MeasurementFailure("pendulum_angular_excursion_too_small", amplitude_px=amplitude)
    quality_flags: list[str] = []
    windowed = centered * np.hanning(n)
    frequencies = np.fft.rfftfreq(n, 1.0 / fps)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    duration = (n - 1) / fps
    valid = (frequencies >= MIN_SPECTRAL_CYCLES_FOR_MEASUREMENT / duration) & (
        frequencies <= min(0.22 * fps, 8.0)
    )
    if not valid.any() or float(power[valid].max()) <= 1e-9:
        raise MeasurementFailure("pendulum_period_spectrum_unresolved")
    initial = float(frequencies[np.flatnonzero(valid)[np.argmax(power[valid])]])
    low = max(0.80 / duration, 0.68 * initial)
    high = min(0.24 * fps, 1.34 * initial)
    best_frequency, best_residual, best_coefficients = initial, float("inf"), None
    for grid_size in (401, 301, 201):
        grid = np.linspace(low, high, grid_size)
        errors = []
        for frequency in grid:
            error, coefficients = _fit_sinusoid(t, signal, float(frequency))
            errors.append(error)
            if error < best_residual:
                best_frequency, best_residual, best_coefficients = float(frequency), error, coefficients
        spacing = float(grid[1] - grid[0])
        low = max(0.80 / duration, best_frequency - 3.0 * spacing)
        high = min(0.24 * fps, best_frequency + 3.0 * spacing)
    period = 1.0 / best_frequency
    cycles = duration / period
    # A short clip can still contain a usable period estimate.  Keep the
    # estimate and expose the limitation as a quality flag; strict physics
    # policy may reject it, but extraction must not discard the raw period.
    if cycles < 2.05:
        quality_flags.append("fewer_than_two_complete_pendulum_periods")
    # Independent extrema check catches a harmonic or a drifting template.
    minimum_distance = max(2, int(round(0.58 * period * fps)))
    maxima = []
    for index in range(1, n - 1):
        if signal[index] >= signal[index - 1] and signal[index] > signal[index + 1]:
            left = max(0, index - minimum_distance // 2); right = min(n, index + minimum_distance // 2 + 1)
            if signal[index] >= np.max(signal[left:right]): maxima.append(index)
    extrema_period = float(np.median(np.diff(maxima)) / fps) if len(maxima) >= 3 else None
    if extrema_period is not None and abs(extrema_period / period - 1.0) > 0.18:
        # A slow illumination/template drift can dominate the raw FFT.  Same-
        # side extrema give an independent fundamental-period seed; refine it
        # with the same least-squares fit instead of accepting a subharmonic.
        seed_frequency = 1.0 / extrema_period
        low2, high2 = 0.84 * seed_frequency, 1.16 * seed_frequency
        best_residual = float("inf")
        for grid_size in (401, 301):
            grid = np.linspace(low2, high2, grid_size)
            for frequency in grid:
                error, coefficients = _fit_sinusoid(t, signal, float(frequency))
                if error < best_residual:
                    best_frequency, best_residual, best_coefficients = float(frequency), error, coefficients
            spacing = float(grid[1] - grid[0])
            low2, high2 = best_frequency - 3.0 * spacing, best_frequency + 3.0 * spacing
        period = 1.0 / best_frequency
        cycles = duration / period
        if abs(extrema_period / period - 1.0) > 0.18:
            quality_flags.append("pendulum_period_harmonic_ambiguity")
    initial_velocity_normalized = float(abs(np.gradient(signal, 1.0 / fps)[0]) / max(amplitude * 2.0 * math.pi / period, 1e-9))
    return {
        "period_s": period, "frequency_hz": best_frequency, "cycles_observed": cycles,
        "sinusoid_fit_residual": best_residual, "amplitude_px": amplitude,
        "principal_fit_coefficients": best_coefficients,
        "extrema_period_s": extrema_period, "initial_velocity_normalized": initial_velocity_normalized,
        "quality_flags": quality_flags,
    }


def _angle_measurements(track: np.ndarray) -> tuple[dict[str, Any], np.ndarray | None]:
    try:
        circle = robust_circle_fit(track)
    except MeasurementFailure as failure:
        return {"available": False, "failure_reason": failure.code, **failure.details}, None
    center = circle["center"]
    angles = np.arctan2(track[:, 0] - center[0], track[:, 1] - center[1])
    equilibrium = float(np.median(angles))
    centered = angles - equilibrium
    amplitude = 0.5 * float(np.percentile(centered, 98) - np.percentile(centered, 2))
    output = {
        "available": True, "pivot_xy": center, "string_length_px": circle["radius"],
        "circle_fit_normalized_rms": circle["normalized_rms_residual"],
        "circle_fit_angular_span_deg": circle["angular_span_deg"],
        "oscillation_amplitude_deg": math.degrees(amplitude),
        "initial_angle_from_vertical_deg": math.degrees(float(angles[0])),
    }
    return output, angles


def _deduplicate_hough_circles(circles: Sequence[Sequence[float]], width: int, height: int) -> list[dict[str, float]]:
    """Normalize and NMS a frame's Hough proposals.

    HoughCircles reports the same bob at several ``param2`` values.  Keeping
    one proposal per spatial neighbourhood makes the subsequent nearest-
    trajectory assignment deterministic and prevents a duplicate of the
    current bob from winning over the other bob.
    """

    proposals: list[dict[str, float]] = []
    for raw in circles:
        if len(raw) < 3:
            continue
        x, y, radius = (float(raw[0]), float(raw[1]), float(raw[2]))
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(radius)):
            continue
        if radius <= 0.0 or x - radius < 1.0 or y - radius < 1.0 or x + radius >= width - 1 or y + radius >= height - 1:
            continue
        duplicate = False
        for prior in proposals:
            if math.hypot(x - prior["x"], y - prior["y"]) < 0.45 * min(radius, prior["r"]):
                duplicate = True
                # The lower-param2 proposals are more permissive; retain the
                # larger circle only when it is genuinely a distinct object.
                if radius > prior["r"] and radius / prior["r"] < 1.22:
                    prior.update(x=x, y=y, r=radius)
                break
        if not duplicate:
            proposals.append({"x": x, "y": y, "r": radius})
    return proposals


def _hough_candidates_per_frame(frames: Sequence[np.ndarray]) -> list[list[dict[str, float]]]:
    """Detect broad bob proposals once per frame for the P8c fallback.

    The normal template tracker is fast and remains the first choice.  This
    fallback is used only when a template is lost or an Hough radius was
    badly biased by a highlight.  It deliberately restricts proposals to the
    lower 30--96% of the image, where the hanging bobs are expected; support
    hardware near the top and floor reflections at the bottom are thereby
    excluded without using a learned detector.
    """

    output: list[list[dict[str, float]]] = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        height, width = gray.shape
        dim = min(height, width)
        found: list[Sequence[float]] = []
        for param2 in (25, 21, 18, 15):
            circles = cv2.HoughCircles(
                gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(10, int(round(dim * 0.02))),
                param1=120,
                param2=param2,
                minRadius=max(5, int(round(dim * 0.012))),
                maxRadius=max(8, int(round(dim * 0.11))),
            )
            if circles is not None:
                found.extend(circles[0].tolist())
            # Once a moderately strict pass has returned enough proposals,
            # avoid spending extra time on permissive duplicate passes.
            if len(found) >= 8:
                break
        proposals = _deduplicate_hough_circles(found, width, height)
        output.append([item for item in proposals if 0.30 * height < item["y"] < 0.96 * height])
    return output


def _hough_track_from_candidates(
    candidates_per_frame: Sequence[Sequence[dict[str, float]]],
    circle: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Track one bob by nearest predicted Hough proposal.

    Radius is treated as a soft prior (rather than an identity gate), because
    highlights and anti-aliasing can change the apparent Hough radius by a
    large factor while the center remains stable.  The returned raw track
    retains NaNs for genuinely missing frames so downstream debug CSVs expose
    the evidence that was actually available.
    """

    if not candidates_per_frame:
        raise MeasurementFailure("pendulum_hough_no_frames")
    radius0 = max(1.0, float(circle["r"]))
    centers = np.full((len(candidates_per_frame), 2), np.nan, dtype=np.float64)
    radii = np.full(len(candidates_per_frame), np.nan, dtype=np.float64)
    centers[0] = (float(circle["x"]), float(circle["y"]))
    radii[0] = radius0
    velocity = np.zeros(2, dtype=np.float64)
    misses = 0
    # The frame dimensions are only needed for a stable search scale; use the
    # same radius-relative scale as the template tracker.
    base_search = max(18.0, 3.5 * radius0)
    for index in range(1, len(candidates_per_frame)):
        valid_indices = np.flatnonzero(np.isfinite(centers[:index, 0]))
        if not len(valid_indices):
            misses += 1
            continue
        last_index = int(valid_indices[-1])
        predicted = centers[last_index] + velocity * max(1, index - last_index)
        search_radius = base_search * (1.0 + 0.35 * misses)
        proposals: list[tuple[float, dict[str, float], float]] = []
        for candidate in candidates_per_frame[index]:
            candidate_radius = max(1e-6, float(candidate["r"]))
            if not 0.45 * radius0 <= candidate_radius <= 2.10 * radius0:
                continue
            distance = math.hypot(float(candidate["x"]) - predicted[0], float(candidate["y"]) - predicted[1])
            if distance > 1.20 * search_radius:
                continue
            cost = (
                distance / search_radius
                + 0.35 * abs(math.log(candidate_radius / radius0))
                - 0.04 * min(3.0, candidate_radius / 10.0)
            )
            proposals.append((cost, candidate, distance))
        if not proposals:
            misses += 1
            continue
        _, selected, distance = min(proposals, key=lambda item: item[0])
        center = np.array([float(selected["x"]), float(selected["y"])], dtype=np.float64)
        centers[index] = center
        radii[index] = float(selected["r"])
        delta = center - centers[last_index]
        velocity = 0.65 * velocity + 0.35 * delta
        misses = 0
    valid = np.isfinite(centers[:, 0]) & np.isfinite(centers[:, 1])
    diagnostics = {
        "tracker_method": "per_frame_hough",
        "track_coverage": float(valid.mean()),
        "valid_points": int(valid.sum()),
        "longest_missing_run": int(_longest_false_run(valid)),
        "raw_valid_mask": valid.tolist(),
        "median_radius_px": None if not np.isfinite(radii).any() else float(np.nanmedian(radii)),
    }
    return centers, diagnostics


def _longest_false_run(valid: np.ndarray) -> int:
    """Local version avoiding an additional public helper dependency."""

    longest = current = 0
    for value in np.asarray(valid, dtype=bool):
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def _circular_hue_center(hues: np.ndarray, weights: np.ndarray) -> float:
    """Return an OpenCV-HSV hue centre on the circular 0..179 domain."""

    angles = np.asarray(hues, dtype=np.float64) * (2.0 * math.pi / 180.0)
    weights = np.asarray(weights, dtype=np.float64)
    sine = float(np.sum(np.sin(angles) * weights))
    cosine = float(np.sum(np.cos(angles) * weights))
    return float((math.atan2(sine, cosine) % (2.0 * math.pi)) * 180.0 / (2.0 * math.pi))


def _track_motion_span(track: np.ndarray) -> float:
    """Robust 2-D motion span used only for tracker diagnostics."""

    points = np.asarray(track, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return 0.0
    centered = points - np.mean(points, axis=0)
    if float(np.max(np.linalg.norm(centered, axis=1))) < 1e-9:
        return 0.0
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    signal = centered @ vt[0]
    return float(np.percentile(signal, 98) - np.percentile(signal, 2))


def _track_color_identity(
    frames: Sequence[np.ndarray],
    circle: dict[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Track one explicitly coloured bob by HSV identity.

    The refined P8c protocol deliberately gives the two bobs different,
    saturated colours.  A grayscale normalized-correlation tracker can swap
    those identities when the bobs approach one another, and on a nearly
    uniform wall it can also lock onto the empty first-frame location with a
    deceptively high correlation.  This deterministic colour-centroid track
    uses the protocol's identity cue directly; it falls back to the legacy
    template/Hough path whenever the initial bob is not chromatically
    distinctive.
    """

    if not frames:
        raise MeasurementFailure("pendulum_color_tracker_no_frames")
    radius = max(1.0, float(circle["r"]))
    cx, cy = float(circle["x"]), float(circle["y"])
    first_hsv = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)
    height, width = first_hsv.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.82 * radius) ** 2
    pixels = first_hsv[disk]
    if len(pixels) < 12:
        raise MeasurementFailure("pendulum_color_template_too_small", pixels=int(len(pixels)))
    saturation = pixels[:, 1].astype(np.float64)
    saturated = saturation >= max(45.0, float(np.percentile(saturation, 55)))
    if int(np.count_nonzero(saturated)) >= 12:
        pixels = pixels[saturated]
    target_saturation = float(np.median(pixels[:, 1]))
    target_value = float(np.median(pixels[:, 2]))
    if target_saturation < 65.0:
        raise MeasurementFailure(
            "pendulum_color_identity_not_distinctive",
            median_saturation=target_saturation,
        )
    target_hue = _circular_hue_center(
        pixels[:, 0], np.maximum(pixels[:, 1].astype(np.float64), 1.0)
    )
    hue_tolerance = float(np.clip(18.0 - 0.03 * target_saturation, 9.0, 14.0))
    saturation_floor = max(38.0, 0.38 * target_saturation)
    value_floor = max(20.0, 0.20 * target_value)
    expected_area = max(float(len(pixels)), 0.22 * math.pi * radius * radius)

    centers = np.full((len(frames), 2), np.nan, dtype=np.float64)
    centers[0] = (cx, cy)
    component_areas = np.full(len(frames), np.nan, dtype=np.float64)
    component_areas[0] = expected_area
    velocity = np.zeros(2, dtype=np.float64)
    misses = 0
    base_search = max(4.5 * radius, 0.065 * min(height, width))
    kernel = np.ones((3, 3), np.uint8)
    minimum_area = max(8, int(round(0.045 * math.pi * radius * radius)))
    maximum_area = 4.5 * math.pi * radius * radius

    for index, frame in enumerate(frames[1:], start=1):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(np.float32)
        hue_distance = np.abs(hue - target_hue)
        hue_distance = np.minimum(hue_distance, 180.0 - hue_distance)
        colour_mask = (
            (hue_distance <= hue_tolerance)
            & (hsv[:, :, 1] >= saturation_floor)
            & (hsv[:, :, 2] >= value_floor)
        )

        valid_indices = np.flatnonzero(np.isfinite(centers[:index, 0]))
        if not len(valid_indices):
            break
        last_index = int(valid_indices[-1])
        predicted = centers[last_index] + velocity * max(1, index - last_index)
        search_radius = base_search * (1.0 + min(1.4, 0.35 * misses))
        x0 = max(0, int(math.floor(predicted[0] - search_radius)))
        x1 = min(width, int(math.ceil(predicted[0] + search_radius + 1)))
        y0 = max(0, int(math.floor(predicted[1] - search_radius)))
        y1 = min(height, int(math.ceil(predicted[1] + search_radius + 1)))
        if x0 >= x1 or y0 >= y1:
            # A colour distractor can briefly send the constant-velocity
            # prediction outside the image.  Record the miss and let the
            # coverage check fall back to template/Hough; never pass an empty
            # ROI to OpenCV morphology.
            misses += 1
            continue
        local = colour_mask[y0:y1, x0:x1].astype(np.uint8) * 255
        local = cv2.morphologyEx(local, cv2.MORPH_OPEN, kernel)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(local, 8)
        candidates: list[tuple[float, float, np.ndarray]] = []
        for label in range(1, count):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if not minimum_area <= area <= maximum_area:
                continue
            center = np.asarray(centroids[label], dtype=np.float64) + np.array([x0, y0])
            distance = float(np.linalg.norm(center - predicted))
            if distance > search_radius:
                continue
            cost = distance / max(search_radius, 1e-9) + 0.30 * abs(
                math.log(max(area, 1.0) / max(expected_area, 1.0))
            )
            candidates.append((cost, area, center))
        if not candidates:
            misses += 1
            continue
        _, area, center = min(candidates, key=lambda item: item[0])
        centers[index] = center
        component_areas[index] = area
        velocity = 0.70 * velocity + 0.30 * (center - centers[last_index])
        misses = 0

    valid = np.isfinite(centers[:, 0]) & np.isfinite(centers[:, 1])
    diagnostics = {
        "tracker_method": "hsv_color_centroid",
        "track_coverage": float(valid.mean()),
        "valid_points": int(valid.sum()),
        "longest_missing_run": int(_longest_false_run(valid)),
        "raw_valid_mask": valid.tolist(),
        "target_hue_opencv": target_hue,
        "target_saturation": target_saturation,
        "target_value": target_value,
        "hue_tolerance": hue_tolerance,
        "median_component_area_px": (
            None if not np.isfinite(component_areas).any() else float(np.nanmedian(component_areas))
        ),
        "motion_span_px": _track_motion_span(centers),
    }
    return centers, diagnostics


def _track_with_fallback(
    frames: Sequence[np.ndarray],
    circle: dict[str, float],
    *,
    candidates_per_frame: Sequence[Sequence[dict[str, float]]] | None = None,
    candidate_cache: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run template tracking, then recover with Hough when needed."""

    template_track: np.ndarray | None = None
    template_diag: dict[str, Any]
    try:
        template_track, _, template_diag = track_template_circle(
            frames, circle, search_radius_factor=3.2
        )
    except MeasurementFailure as failure:
        template_diag = {"tracker_method": "template", "failure_reason": failure.code, **failure.details}

    if template_track is not None:
        template_valid_mask = np.isfinite(template_track[:, 0]) & np.isfinite(template_track[:, 1])
        template_coverage = float(template_valid_mask.mean())
        template_gap = int(_longest_false_run(template_valid_mask))
    else:
        template_valid_mask = np.zeros(len(frames), dtype=bool)
        template_coverage = 0.0
        template_gap = len(frames)
    template_diag = {
        **template_diag,
        "tracker_method": "template",
        "track_coverage": template_coverage,
        "valid_points": int(template_valid_mask.sum()),
        "longest_missing_run": template_gap,
        "raw_valid_mask": template_valid_mask.tolist(),
    }
    # The task's two explicit colours are the strongest identity cue.  Prefer
    # them whenever they yield a sufficiently supported trajectory; this also
    # prevents grayscale identity swaps when the bobs pass close together.
    try:
        colour_track, colour_diag = _track_color_identity(frames, circle)
    except MeasurementFailure as failure:
        colour_track = None
        colour_diag = {
            "tracker_method": "hsv_color_centroid",
            "failure_reason": failure.code,
            **failure.details,
        }
    if colour_track is not None:
        colour_valid = np.isfinite(colour_track[:, 0]) & np.isfinite(colour_track[:, 1])
        colour_coverage = float(colour_valid.mean())
        colour_gap = int(_longest_false_run(colour_valid))
        if (
            colour_coverage >= 0.65
            and int(colour_valid.sum()) >= max(12, int(round(0.30 * len(frames))))
            and colour_gap <= max(14, len(frames) // 5)
        ):
            return colour_track, {
                **colour_diag,
                "fallback_used": False,
                "color_identity_used": True,
                "template_diagnostics": template_diag,
            }
    # A nearly complete template track is preferable because it preserves the
    # original appearance-based measurements.  Otherwise use the explicit
    # per-frame Hough fallback.
    if template_track is not None and template_coverage >= 0.90 and template_gap <= max(8, len(frames) // 20):
        return template_track, {
            **template_diag,
            "fallback_used": False,
            "template_quality": "usable",
        }
    if candidates_per_frame is None:
        if candidate_cache is not None and candidate_cache.get("candidates") is not None:
            candidates_per_frame = candidate_cache["candidates"]
        else:
            candidates_per_frame = _hough_candidates_per_frame(frames)
            if candidate_cache is not None:
                candidate_cache["candidates"] = candidates_per_frame
    hough_track, hough_diag = _hough_track_from_candidates(candidates_per_frame, circle)
    hough_valid = np.isfinite(hough_track[:, 0]) & np.isfinite(hough_track[:, 1])
    # If Hough cannot improve the evidence, retain the template track so the
    # caller can report its raw coverage and make the final extraction decision
    # based on period fit rather than silently replacing it.
    if template_track is not None and hough_valid.mean() < template_coverage:
        return template_track, {
            **template_diag,
            "fallback_used": False,
            "fallback_attempted": True,
            "fallback_reason": "hough_not_better",
            "hough_diagnostics": hough_diag,
        }
    return hough_track, {
        **hough_diag,
        "fallback_used": True,
        "fallback_reason": "template_coverage_or_gap",
        "template_diagnostics": template_diag,
    }


def evaluate(
    frames: Sequence[np.ndarray], fps: float, debug_dir: str | Path | None, sample_id: int | str = 0
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}; measurements: dict[str, Any] = {}; artifacts: list[Path] = []
    debug = Path(debug_dir) if debug_dir is not None else None
    try:
        work_frames, scale = validate_and_resize_frames(frames, fps, min_frames=48)
        diagnostics.update(frame_count=len(work_frames), fps=float(fps), processing_scale=scale)
        camera = estimate_camera_motion(work_frames); diagnostics["camera_motion"] = camera
        if camera.get("available") and camera.get("median_ransac_inlier_fraction", 0.0) >= 0.60 and (
            camera["max_translation_fraction"] > 0.04 or camera["max_rotation_deg"] > 1.5
            or camera["max_scale_change"] > 0.04
        ):
            raise MeasurementFailure("camera_motion_too_large", **camera)
        candidates = detect_circle_candidates(work_frames[0])
        first, second, selection = select_circle_pair(work_frames, candidates, task=TASK_ID)
        diagnostics["object_selection"] = selection
        raw_tracks = []; track_diags = []
        hough_cache: dict[str, Any] = {}
        for circle in (first, second):
            # A local template is preferred, but Hough radius estimates can be
            # badly biased by highlights.  If it loses a bob, recover the
            # center trajectory frame-by-frame and retain both quality records.
            track, track_diag = _track_with_fallback(
                work_frames, circle, candidate_cache=hough_cache
            )
            raw_tracks.append(track)
            track_diags.append(track_diag)
        diagnostics["tracks"] = track_diags
        diagnostics["raw_track_valid_masks"] = [
            (np.isfinite(track[:, 0]) & np.isfinite(track[:, 1])).tolist()
            for track in raw_tracks
        ]
        tracks = []
        interpolation_quality_flags: list[str] = []
        for index, track in enumerate(raw_tracks):
            try:
                tracks.append(interpolate_track(track, max_gap=max(10, len(work_frames) // 8)))
            except MeasurementFailure as failure:
                # Preserve a period estimate when there are enough valid
                # samples but a long unsupported gap makes the conservative
                # interpolation guard fire.  The quality flag is propagated to
                # physics_decision; it is not silently treated as a pass.
                valid = np.isfinite(track[:, 0]) & np.isfinite(track[:, 1])
                if int(valid.sum()) < max(12, int(round(0.30 * len(track)))):
                    raise
                tracks.append(interpolate_track(track, max_gap=None))
                interpolation_quality_flags.append(f"track_{index}:large_interpolation_gap")
                track_diags[index]["interpolation_fallback"] = failure.code
        if interpolation_quality_flags:
            diagnostics["interpolation_quality_flags"] = interpolation_quality_flags
        signals_axes = [_principal_signal(track) for track in tracks]
        signals = [item[0] for item in signals_axes]
        axes = [item[1] for item in signals_axes]
        period_fits = [_period_from_signal(signal, fps) for signal in signals]
        angle_fits = [_angle_measurements(track) for track in tracks]
        amplitudes = [fit["amplitude_px"] for fit in period_fits]
        large_index = int(np.argmax(amplitudes)); small_index = 1 - large_index
        amplitude_identity_ratio = amplitudes[large_index] / max(amplitudes[small_index], 1e-9)
        amplitude_identity_ambiguous = amplitude_identity_ratio < 1.65
        t_small = float(period_fits[small_index]["period_s"]); t_large = float(period_fits[large_index]["period_s"])
        ratio = t_large / t_small
        angle_small = angle_fits[small_index][0]; angle_large = angle_fits[large_index][0]
        lengths = [angle_small.get("string_length_px"), angle_large.get("string_length_px")]
        length_ratio = None if any(value is None for value in lengths) else float(lengths[1] / lengths[0])
        small_number = f"{SMALL_ANGLE_DEG:g}"
        large_number = f"{LARGE_ANGLE_DEG:g}"
        small_suffix = _angle_label(SMALL_ANGLE_DEG)
        large_suffix = _angle_label(LARGE_ANGLE_DEG)
        measurements.update({
            "fps": float(fps), "frame_count": len(work_frames),
            f"pendulum_{small_number}_track_index": small_index,
            f"pendulum_{large_number}_track_index": large_index,
            f"T{small_number}_s": t_small, f"T{large_number}_s": t_large,
            "period_ratio": ratio,
            "release_angles_deg": [SMALL_ANGLE_DEG, LARGE_ANGLE_DEG],
            "expected_period_ratio_exact": EXPECTED_PERIOD_RATIO,
            "expected_period_ratio_from_tasks_json": TASK_SPEC_EXPECTED_RATIO,
            "task_spec_expected_mismatch": abs(EXPECTED_PERIOD_RATIO - TASK_SPEC_EXPECTED_RATIO),
            f"T{large_number}_greater_than_T{small_number}": bool(t_large > t_small),
            f"sinusoid_fit_residual_{small_number}": period_fits[small_index]["sinusoid_fit_residual"],
            f"sinusoid_fit_residual_{large_number}": period_fits[large_index]["sinusoid_fit_residual"],
            f"initial_velocity_normalized_{small_number}": period_fits[small_index]["initial_velocity_normalized"],
            f"initial_velocity_normalized_{large_number}": period_fits[large_index]["initial_velocity_normalized"],
            f"angle_fit_{small_number}": angle_small, f"angle_fit_{large_number}": angle_large,
            f"string_length_ratio_{large_number}_over_{small_number}": length_ratio,
            "amplitude_identity_ratio": amplitude_identity_ratio,
            "amplitude_identity_ambiguous": amplitude_identity_ambiguous,
            "track_quality": [
                {
                    "coverage": track_diag.get("track_coverage"),
                    "valid_points": track_diag.get("valid_points"),
                    "longest_missing_run": track_diag.get("longest_missing_run"),
                    "tracker_method": track_diag.get("tracker_method"),
                    "fallback_used": bool(track_diag.get("fallback_used", False)),
                }
                for track_diag in track_diags
            ],
        })
        metric_failures = []
        if not angle_small["available"]: metric_failures.append(f"{small_suffix}:pivot_circle_fit_failed")
        if not angle_large["available"]: metric_failures.append(f"{large_suffix}:pivot_circle_fit_failed")
        if amplitude_identity_ambiguous:
            metric_failures.append("pendulum_amplitude_identity_ambiguous")
        for period_fit in period_fits:
            metric_failures.extend(str(flag) for flag in period_fit.get("quality_flags", []))
        for track_index, track_diag in enumerate(track_diags):
            if float(track_diag.get("track_coverage", 0.0)) < TRACK_COVERAGE_QUALITY_FLOOR:
                metric_failures.append(f"track_{track_index}:coverage_below_{TRACK_COVERAGE_QUALITY_FLOOR:g}")
        metric_failures.extend(interpolation_quality_flags)
        # Apparent string length is retained only as a generation/first-frame
        # QA diagnostic.  The user explicitly controls equal length before
        # generation; a noisy circle/pivot fit must not reject an otherwise
        # measurable period comparison or become an undeclared physics gate.
        setup_diagnostics: list[str] = []
        if length_ratio is None:
            setup_diagnostics.append("string_length_measurement_unavailable")
        elif abs(length_ratio - 1.0) > 0.10:
            setup_diagnostics.append("string_length_relative_difference_gt_0.10")
        # A quality issue can affect both pendulums; retain one stable label
        # per issue rather than duplicating it in CSV/physics reason fields.
        metric_failures = list(dict.fromkeys(metric_failures))
        measurements["measurement_quality"] = {
            "extract_semantics": "finite_period_estimates_available",
            "strict_physics_ready": not bool(metric_failures),
            "quality_flags": list(metric_failures),
            "raw_observations_preserved": True,
            "raw_track_artifact": "p8c_raw_tracks.csv",
            "identity_assignment": "larger_observed_amplitude_as_30deg",
            "setup_diagnostics": setup_diagnostics,
            "string_length_is_not_an_extraction_or_physics_gate": True,
        }
        m1 = abs(ratio - EXPECTED_PERIOD_RATIO)
        metrics = {
            "M1": m1, "period_ratio_absolute_error": m1,
            "period_ratio_error_against_tasks_json": abs(ratio - TASK_SPEC_EXPECTED_RATIO),
            f"M2_T{large_number}_not_greater_penalty": 0.0 if t_large > t_small else 1.0,
            "M2_sinusoid_fit_residual": 0.5 * (period_fits[small_index]["sinusoid_fit_residual"] + period_fits[large_index]["sinusoid_fit_residual"]),
            "M2": (0.0 if t_large > t_small else 1.0) + 0.5 * (period_fits[small_index]["sinusoid_fit_residual"] + period_fits[large_index]["sinusoid_fit_residual"]),
            "string_length_relative_difference": None if length_ratio is None else abs(length_ratio - 1.0),
        }
        diagnostics["period_fits"] = period_fits; diagnostics["principal_axes"] = axes
        if debug is not None:
            debug.mkdir(parents=True, exist_ok=True)
            circle_overlays = []
            for angle_fit in angle_fits:
                if angle_fit[0]["available"]:
                    circle_overlays.append((np.asarray(angle_fit[0]["pivot_xy"]), float(angle_fit[0]["string_length_px"])))
            overlay = draw_track_overlay(
                work_frames[0], [tracks[small_index], tracks[large_index]], [(255, 120, 0), (0, 200, 255)],
                labels=[small_suffix, large_suffix], circles=circle_overlays,
            )
            overlay_path = debug / "p8c_tracks_and_pivots.png"; cv2.imwrite(str(overlay_path), overlay); artifacts.append(overlay_path)
            csv_path = write_track_csv(
                debug / "p8c_angles.csv",
                ["frame", "time_s", f"bob{small_suffix[:-3]}_x", f"bob{small_suffix[:-3]}_y",
                 f"bob{large_suffix[:-3]}_x", f"bob{large_suffix[:-3]}_y",
                 f"signal{small_suffix[:-3]}_px", f"signal{large_suffix[:-3]}_px"],
                [np.arange(len(work_frames)), np.arange(len(work_frames)) / fps,
                 tracks[small_index][:, 0], tracks[small_index][:, 1], tracks[large_index][:, 0], tracks[large_index][:, 1],
                 signals[small_index], signals[large_index]],
            ); artifacts.append(csv_path)
            # Keep the un-interpolated observations as a separate artifact.
            # ``p8c_angles.csv`` is convenient for fitting, whereas this file
            # is the audit trail: NaNs and validity bits show exactly which
            # frames were observed versus filled for measurement.
            raw_csv_path = write_track_csv(
                debug / "p8c_raw_tracks.csv",
                ["frame", "time_s",
                 f"bob{small_suffix[:-3]}_raw_x", f"bob{small_suffix[:-3]}_raw_y",
                 f"bob{large_suffix[:-3]}_raw_x", f"bob{large_suffix[:-3]}_raw_y",
                 f"bob{small_suffix[:-3]}_valid", f"bob{large_suffix[:-3]}_valid"],
                [np.arange(len(work_frames)), np.arange(len(work_frames)) / fps,
                 raw_tracks[small_index][:, 0], raw_tracks[small_index][:, 1],
                 raw_tracks[large_index][:, 0], raw_tracks[large_index][:, 1],
                 np.isfinite(raw_tracks[small_index][:, 0]).astype(np.float64),
                 np.isfinite(raw_tracks[large_index][:, 0]).astype(np.float64)],
            ); artifacts.append(raw_csv_path)
            diagnostic_path = write_json(debug / "p8c_diagnostics.json", {"measurements": measurements, "metrics": metrics, "diagnostics": diagnostics})
            artifacts.append(diagnostic_path)
        return result(TASK_ID, sample_id, True, measurements, metrics, metric_failures=metric_failures, diagnostics=diagnostics, debug_artifacts=artifacts)
    except MeasurementFailure as failure:
        if debug is not None:
            debug.mkdir(parents=True, exist_ok=True)
            failure_path = write_json(debug / "p8c_failure.json", {"failure_reason": failure.code, "failure_details": failure.details, "diagnostics": diagnostics})
            artifacts.append(failure_path)
            if frames:
                preview_path = debug / "p8c_failure_frame.png"; cv2.imwrite(str(preview_path), frames[0]); artifacts.append(preview_path)
        return result(TASK_ID, sample_id, False, measurements, {"M1": None, "M2": None}, failure=failure, diagnostics=diagnostics, debug_artifacts=artifacts)


evaluate_p8c = evaluate
