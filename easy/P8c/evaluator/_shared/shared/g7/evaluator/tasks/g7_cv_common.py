"""Shared deterministic computer-vision helpers for Group 7 evaluators.

This module deliberately contains no learned detector and never calls a VLM.
All image evidence comes from OpenCV/Numpy operations whose intermediate
results can be written to the evaluator debug directory.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


class MeasurementFailure(RuntimeError):
    """A named, machine-readable extraction failure."""

    def __init__(self, code: str, **details: Any) -> None:
        super().__init__(code)
        self.code = code
        self.details = details


def jsonable(value: Any) -> Any:
    """Convert Numpy values and non-finite floats to strict JSON values."""

    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def result(
    task_id: str,
    sample_id: int | str,
    success: bool,
    measurements: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    *,
    failure: MeasurementFailure | str | None = None,
    metric_failures: Sequence[str] = (),
    diagnostics: dict[str, Any] | None = None,
    debug_artifacts: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Build the common evaluator result without encoding failure as score 0."""

    if isinstance(failure, MeasurementFailure):
        failure_reason = failure.code
        failure_details = failure.details
    else:
        failure_reason = failure
        failure_details = {}
    failure_reasons = [failure_reason] if failure_reason else []
    return jsonable(
        {
            "task_id": task_id,
            "sample_id": sample_id,
            "extract_success": bool(success),
            "measurements": measurements or {},
            "metrics": metrics or {},
            "failure_reason": failure_reason,
            "failure_reasons": failure_reasons,
            "failure_details": failure_details,
            "metric_failures": list(metric_failures),
            "diagnostics": diagnostics or {},
            "debug_artifacts": list(debug_artifacts),
        }
    )


def validate_and_resize_frames(
    frames: Sequence[np.ndarray], fps: float, *, min_frames: int = 24, max_side: int = 960
) -> tuple[list[np.ndarray], float]:
    """Validate decoded frames and downscale deterministically for CV work."""

    if not frames:
        raise MeasurementFailure("video_decode_failed")
    if not math.isfinite(float(fps)) or fps <= 0:
        raise MeasurementFailure("invalid_fps", fps=fps)
    if len(frames) < min_frames:
        raise MeasurementFailure(
            "insufficient_frame_count", frame_count=len(frames), required=min_frames
        )
    first = np.asarray(frames[0])
    if first.ndim != 3 or first.shape[2] not in (3, 4):
        raise MeasurementFailure("unsupported_frame_format", shape=first.shape)
    h, w = first.shape[:2]
    if h < 96 or w < 96:
        raise MeasurementFailure("frame_resolution_too_small", width=w, height=h)
    for index, frame in enumerate(frames):
        if frame is None or frame.shape[:2] != (h, w):
            raise MeasurementFailure(
                "inconsistent_frame_geometry", frame_index=index, shape=getattr(frame, "shape", None)
            )
    scale = min(1.0, float(max_side) / max(h, w))
    if scale == 1.0:
        return [np.ascontiguousarray(f[:, :, :3]) for f in frames], 1.0
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = [cv2.resize(f[:, :, :3], size, interpolation=cv2.INTER_AREA) for f in frames]
    return resized, scale


def _disk_mask(shape: tuple[int, int], center: tuple[float, float], radius: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2


def _circle_edge_score(gray: np.ndarray, x: float, y: float, radius: float) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    h, w = gray.shape
    x0, x1 = max(0, int(x - 1.5 * radius)), min(w, int(x + 1.5 * radius + 1))
    y0, y1 = max(0, int(y - 1.5 * radius)), min(h, int(y + 1.5 * radius + 1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y)
    edge = (rr >= 0.85 * radius) & (rr <= 1.15 * radius)
    local = rr <= 1.4 * radius
    if edge.sum() < 8:
        return 0.0
    denom = float(np.median(mag[y0:y1, x0:x1][local])) + 1.0
    return float(np.median(mag[y0:y1, x0:x1][edge]) / denom)


def detect_circle_candidates(
    frame: np.ndarray,
    *,
    min_radius_fraction: float = 0.012,
    max_radius_fraction: float = 0.11,
    max_candidates: int = 32,
) -> list[dict[str, float]]:
    """Return NMS-filtered Hough circle proposals with deterministic scores."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    dim = min(h, w)
    min_radius = max(6, int(round(dim * min_radius_fraction)))
    max_radius = max(min_radius + 2, int(round(dim * max_radius_fraction)))
    found: list[np.ndarray] = []
    for param2 in (25, 21, 18):
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(12, int(dim * 0.025)),
            param1=120,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is not None:
            found.extend(circles[0])
        if len(found) >= 8:
            break
    proposals: list[dict[str, float]] = []
    for raw in found:
        x, y, radius = (float(raw[0]), float(raw[1]), float(raw[2]))
        if x - radius < 1 or y - radius < 1 or x + radius >= w - 1 or y + radius >= h - 1:
            continue
        score = _circle_edge_score(gray, x, y, radius)
        duplicate = False
        for prior in proposals:
            distance = math.hypot(x - prior["x"], y - prior["y"])
            if distance < 0.42 * min(radius, prior["r"]):
                duplicate = True
                if score > prior["edge_score"]:
                    prior.update(x=x, y=y, r=radius, edge_score=score)
                break
        if not duplicate:
            proposals.append({"x": x, "y": y, "r": radius, "edge_score": score})
    proposals.sort(key=lambda item: item["edge_score"], reverse=True)
    return proposals[:max_candidates]


def annularity_score(frame: np.ndarray, circle: dict[str, float]) -> float:
    """Measure evidence for an inner rim, which distinguishes a ring from a ball."""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    x, y, radius = circle["x"], circle["y"], circle["r"]
    h, w = gray.shape
    x0, x1 = max(0, int(x - 1.4 * radius)), min(w, int(x + 1.4 * radius + 1))
    y0, y1 = max(0, int(y - 1.4 * radius)), min(h, int(y + 1.4 * radius + 1))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    rr = np.hypot(xx - x, yy - y)
    local = mag[y0:y1, x0:x1]
    # The Hough outer-radius estimate can be biased by highlights, so include
    # the full plausible inner-rim band instead of sampling one narrow radius.
    inner = (rr >= 0.42 * radius) & (rr <= 0.93 * radius)
    outer = (rr >= 0.86 * radius) & (rr <= 1.14 * radius)
    center = rr <= 0.35 * radius
    shell = (rr >= 1.15 * radius) & (rr <= 1.38 * radius)
    if min(inner.sum(), outer.sum(), center.sum(), shell.sum()) < 8:
        return 0.0
    inner_edge = float(np.percentile(local[inner], 82))
    outer_edge = float(np.percentile(local[outer], 75)) + 1.0
    lab = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).astype(np.float32)
    center_color = np.median(lab[center], axis=0)
    shell_color = np.median(lab[shell], axis=0)
    color_distance = float(np.linalg.norm(center_color - shell_color))
    center_background_similarity = math.exp(-color_distance / 24.0)
    return float(inner_edge / outer_edge + 0.75 * center_background_similarity)


def select_circle_pair(
    frames: Sequence[np.ndarray], candidates: Sequence[dict[str, float]], *, task: str
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    """Select the moving, equally sized pair appropriate for P7 or P8c."""

    if task=='P7':
        # Small Hough circles contained in another resolved circular body are
        # surface marks or its inner rim, not separate rolling objects.
        candidates=[c for c in candidates if not any(
            p['r']>c['r']/0.78 and math.hypot(c['x']-p['x'],c['y']-p['y'])<1.05*p['r']
            and p['edge_score']>1.5 for p in candidates)]
    if len(candidates) < 2:
        raise MeasurementFailure("fewer_than_two_circular_objects_detected", candidates=len(candidates))
    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    sample_indices = sorted(
        set(min(len(frames) - 1, max(1, int(round(q * (len(frames) - 1))))) for q in (0.18, 0.38, 0.62))
    )
    later = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32) for i in sample_indices]
    h, w = gray0.shape
    scored: list[dict[str, float]] = []
    for candidate in candidates:
        x, y, radius = candidate["x"], candidate["y"], candidate["r"]
        mask = _disk_mask((h, w), (x, y), max(4.0, 0.9 * radius))
        diffs = [float(np.mean(np.abs(image[mask] - gray0[mask]))) / 64.0 for image in later]
        item = dict(candidate)
        item["temporal_change"] = float(np.clip(max(diffs, default=0.0), 0.0, 2.0))
        item["annularity"] = annularity_score(frames[0], item)
        scored.append(item)

    best: tuple[float, dict[str, float], dict[str, float]] | None = None
    considered = 0
    for i, first in enumerate(scored):
        for second in scored[i + 1 :]:
            mean_radius = 0.5 * (first["r"] + second["r"])
            ratio = max(first["r"], second["r"]) / max(1e-6, min(first["r"], second["r"]))
            distance = math.hypot(first["x"] - second["x"], first["y"] - second["y"])
            # Hough's radius estimate is often biased by specular highlights on
            # the pendulum bobs.  In particular, the same visible bob can be
            # reported at roughly 9 px in one clip and 15 px in another.  The
            # old global 1.48 limit therefore discarded the true P8c pair and
            # paired one bob with a static support/floor circle.  Keep the
            # stricter tolerance for P7 (where equal outer radius is part of
            # the object identity), but allow a wider proposal tolerance for
            # P8c; motion and distance terms still rank the pair.
            radius_ratio_limit = 2.05 if task == "P8c" else 1.48
            if ratio > radius_ratio_limit or distance < 1.28 * min(first["r"], second["r"]):
                continue
            if task == "P7" and distance > 9.0 * mean_radius:
                continue
            if task == 'P7' and distance < first['r']+second['r']:
                continue
            if task == "P8c" and distance < 3.0 * mean_radius:
                continue
            considered += 1
            radius_term = math.exp(-3.0 * abs(math.log(ratio)))
            motion_term = first["temporal_change"] + second["temporal_change"]
            edge_term = 0.08 * min(12.0, first["edge_score"] + second["edge_score"])
            task_term = 0.0
            if task == "P7":
                task_term = 0.35 * abs(first["annularity"] - second["annularity"])
            score = motion_term + radius_term + edge_term + task_term
            if best is None or score > best[0]:
                best = (score, first, second)
    if best is None:
        raise MeasurementFailure(
            "equal_radius_object_pair_not_found", candidates=len(candidates), pairs_considered=considered
        )
    _, first, second = best
    diagnostics = {
        "circle_candidates": scored,
        "selected_pair_score": best[0],
        "sampled_change_frames": sample_indices,
    }
    return dict(first), dict(second), diagnostics


def _tracking_representation(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = np.clip(mag * 0.7, 0, 255).astype(np.uint8)
    return cv2.addWeighted(gray, 0.42, mag, 0.58, 0)


def track_template_circle(
    frames: Sequence[np.ndarray], circle: dict[str, float], *, search_radius_factor: float = 3.5
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Track a circular object with a fixed masked edge/appearance template."""

    radius = float(circle["r"])
    half = max(9, int(round(radius * 1.23)))
    reps = [_tracking_representation(frame) for frame in frames]
    h, w = reps[0].shape
    cx0, cy0 = float(circle["x"]), float(circle["y"])
    x0, y0 = int(round(cx0)) - half, int(round(cy0)) - half
    x1, y1 = x0 + 2 * half + 1, y0 + 2 * half + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        raise MeasurementFailure("initial_object_template_out_of_bounds", x=cx0, y=cy0, radius=radius)
    template = reps[0][y0:y1, x0:x1].copy()
    yy, xx = np.ogrid[: template.shape[0], : template.shape[1]]
    mask = (((xx - half) ** 2 + (yy - half) ** 2) <= (1.16 * radius) ** 2).astype(np.uint8) * 255
    centers = np.full((len(frames), 2), np.nan, dtype=np.float64)
    confidence = np.full(len(frames), np.nan, dtype=np.float64)
    centers[0] = (cx0, cy0)
    confidence[0] = 1.0
    velocity = np.zeros(2, dtype=np.float64)
    misses = 0
    base_search = max(int(round(search_radius_factor * radius)), int(round(0.035 * min(h, w))))
    for index in range(1, len(frames)):
        last_indices = np.flatnonzero(np.isfinite(centers[:index, 0]))
        if not len(last_indices):
            break
        last_index = int(last_indices[-1])
        last = centers[last_index]
        if len(last_indices) >= 3:
            recent = centers[last_indices[-3:]]
            velocity = np.median(np.diff(recent, axis=0), axis=0)
        predicted = last + velocity * max(1, index - last_index)
        search_radius = int(base_search * (1.0 + min(1.5, 0.45 * misses)))
        sx0 = max(0, int(math.floor(predicted[0] - search_radius - half)))
        sy0 = max(0, int(math.floor(predicted[1] - search_radius - half)))
        sx1 = min(w, int(math.ceil(predicted[0] + search_radius + half + 1)))
        sy1 = min(h, int(math.ceil(predicted[1] + search_radius + half + 1)))
        search = reps[index][sy0:sy1, sx0:sx1]
        if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
            misses += 1
            continue
        response = cv2.matchTemplate(search, template, cv2.TM_CCORR_NORMED, mask=mask)
        response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, max_value, _, max_location = cv2.minMaxLoc(response)
        candidate = np.array(
            [sx0 + max_location[0] + half, sy0 + max_location[1] + half], dtype=np.float64
        )
        jump = float(np.linalg.norm(candidate - predicted))
        minimum_confidence = 0.30 if misses < 2 else 0.24
        if max_value < minimum_confidence or jump > 1.15 * search_radius:
            misses += 1
            continue
        centers[index] = candidate
        confidence[index] = max_value
        misses = 0
    valid = np.isfinite(centers[:, 0])
    diagnostics = {
        "radius_px": radius,
        "template_half_size": half,
        "track_coverage": float(valid.mean()),
        "median_match_confidence": float(np.nanmedian(confidence)),
        "longest_missing_run": longest_false_run(valid),
    }
    return centers, confidence, diagnostics


def longest_false_run(valid: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(valid, dtype=bool):
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def interpolate_track(track: np.ndarray, *, max_gap: int | None = None) -> np.ndarray:
    """Interpolate a 2-D track, rejecting tracks with large unsupported gaps."""

    track = np.asarray(track, dtype=np.float64)
    valid = np.isfinite(track[:, 0]) & np.isfinite(track[:, 1])
    if valid.sum() < 4:
        raise MeasurementFailure("track_has_too_few_valid_points", valid_points=int(valid.sum()))
    if max_gap is not None and longest_false_run(valid) > max_gap:
        raise MeasurementFailure(
            "track_gap_too_long", longest_gap=longest_false_run(valid), allowed_gap=max_gap
        )
    indices = np.arange(len(track), dtype=np.float64)
    result = track.copy()
    for axis in range(2):
        result[:, axis] = np.interp(indices, indices[valid], track[valid, axis])
    return result


def smooth_1d(values: Iterable[float], window: int = 7) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64)
    if len(values) < 3:
        return values.copy()
    window = max(3, int(window) | 1)
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if window < 3:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(padded, window), axis=1)
    sigma = max(0.8, window / 5.0)
    return cv2.GaussianBlur(med.reshape(-1, 1), (1, window), sigmaX=0.0, sigmaY=sigma).reshape(-1).astype(np.float64)


def robust_circle_fit(points: np.ndarray) -> dict[str, Any]:
    """Algebraic initialization followed by robust geometric Gauss-Newton."""

    points = np.asarray(points, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < 12:
        raise MeasurementFailure("circle_fit_has_too_few_points", valid_points=len(points))
    keep = np.ones(len(points), dtype=bool)
    cx = cy = radius = float("nan")
    for _ in range(6):
        fit_points = points[keep]
        mean = fit_points.mean(axis=0)
        scale = float(np.std(fit_points)) or 1.0
        normalized = (fit_points - mean) / scale
        x, y = normalized[:, 0], normalized[:, 1]
        matrix = np.column_stack([x, y, np.ones_like(x)])
        rhs = -(x * x + y * y)
        params, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
        if rank < 3:
            raise MeasurementFailure("circle_fit_degenerate_trajectory")
        ncx, ncy = -0.5 * params[0], -0.5 * params[1]
        radius_sq = ncx * ncx + ncy * ncy - params[2]
        if radius_sq <= 0:
            raise MeasurementFailure("circle_fit_nonpositive_radius")
        cx, cy = mean + scale * np.array([ncx, ncy])
        radius = scale * math.sqrt(radius_sq)
        for _inner in range(8):
            delta = fit_points - np.array([cx, cy])
            distances = np.linalg.norm(delta, axis=1)
            distances = np.maximum(distances, 1e-6)
            residual = distances - radius
            jacobian = np.column_stack([-delta[:, 0] / distances, -delta[:, 1] / distances, -np.ones(len(delta))])
            step, _, _, _ = np.linalg.lstsq(jacobian, -residual, rcond=None)
            cx += float(step[0]); cy += float(step[1]); radius += float(step[2])
            if np.linalg.norm(step) < 1e-5:
                break
        all_residuals = np.abs(np.linalg.norm(points - np.array([cx, cy]), axis=1) - radius)
        median = float(np.median(all_residuals))
        mad = float(np.median(np.abs(all_residuals - median))) + 1e-6
        new_keep = all_residuals <= median + 4.5 * 1.4826 * mad
        if new_keep.sum() < 12 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    residuals = np.linalg.norm(points[keep] - np.array([cx, cy]), axis=1) - radius
    angles = np.unwrap(np.arctan2(points[keep, 0] - cx, points[keep, 1] - cy))
    return {
        "center": np.array([cx, cy]),
        "radius": float(radius),
        "rms_residual": float(np.sqrt(np.mean(residuals**2))),
        "normalized_rms_residual": float(np.sqrt(np.mean(residuals**2)) / max(radius, 1e-6)),
        "inlier_fraction": float(keep.mean()),
        "angular_span_deg": float(np.degrees(np.percentile(angles, 99) - np.percentile(angles, 1))),
    }


def estimate_camera_motion(frames: Sequence[np.ndarray]) -> dict[str, Any]:
    """Estimate global first-to-frame affine drift from background features."""

    first = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(first, maxCorners=500, qualityLevel=0.015, minDistance=8, blockSize=7)
    if points is None or len(points) < 24:
        return {"available": False, "reason": "insufficient_background_features"}
    h, w = first.shape
    sample_indices = sorted(set(np.linspace(1, len(frames) - 1, min(8, len(frames) - 1), dtype=int).tolist()))
    translations: list[float] = []
    rotations: list[float] = []
    scales: list[float] = []
    inlier_fractions: list[float] = []
    for index in sample_indices:
        target = cv2.cvtColor(frames[index], cv2.COLOR_BGR2GRAY)
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            first, target, points, None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if moved is None or status is None:
            continue
        good = status.reshape(-1).astype(bool)
        if good.sum() < 12:
            continue
        affine, inliers = cv2.estimateAffinePartial2D(
            points.reshape(-1, 2)[good], moved.reshape(-1, 2)[good], method=cv2.RANSAC, ransacReprojThreshold=2.0
        )
        if affine is None:
            continue
        a, b, tx = affine[0]
        c, d, ty = affine[1]
        scale = math.sqrt(max(1e-12, a * a + c * c))
        rotation = math.degrees(math.atan2(c, a))
        translations.append(math.hypot(tx, ty))
        rotations.append(abs(rotation))
        scales.append(abs(scale - 1.0))
        inlier_fractions.append(float(inliers.mean()) if inliers is not None else 0.0)
    if not translations:
        return {"available": False, "reason": "global_affine_estimation_failed"}
    return {
        "available": True,
        "sample_indices": sample_indices,
        "max_translation_px": float(max(translations)),
        "max_translation_fraction": float(max(translations) / min(h, w)),
        "max_rotation_deg": float(max(rotations)),
        "max_scale_change": float(max(scales)),
        "median_ransac_inlier_fraction": float(np.median(inlier_fractions)),
    }


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path


def write_track_csv(path: Path, header: Sequence[str], columns: Sequence[np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.column_stack(columns)
    np.savetxt(path, matrix, delimiter=",", header=",".join(header), comments="", fmt="%.9g")
    return path


def draw_track_overlay(
    frame: np.ndarray,
    tracks: Sequence[np.ndarray],
    colors: Sequence[tuple[int, int, int]],
    *,
    labels: Sequence[str] | None = None,
    circles: Sequence[tuple[np.ndarray, float]] = (),
) -> np.ndarray:
    overlay = frame.copy()
    for index, (track, color) in enumerate(zip(tracks, colors)):
        valid = np.isfinite(track).all(axis=1)
        points = np.rint(track[valid]).astype(np.int32)
        if len(points) >= 2:
            cv2.polylines(overlay, [points], False, color, 2, cv2.LINE_AA)
        if len(points):
            cv2.circle(overlay, tuple(points[0]), 5, color, -1, cv2.LINE_AA)
            if labels and index < len(labels):
                cv2.putText(
                    overlay, labels[index], tuple(points[0] + np.array([7, -7])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
                )
    for center, radius in circles:
        if np.isfinite(center).all() and math.isfinite(radius):
            cv2.circle(overlay, tuple(np.rint(center).astype(int)), int(round(radius)), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.drawMarker(overlay, tuple(np.rint(center).astype(int)), (255, 255, 0), cv2.MARKER_CROSS, 12, 1)
    return overlay
