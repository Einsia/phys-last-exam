#!/usr/bin/env python3
"""Deterministic P11 geometric-optics evaluator.

The evaluator intentionally uses no language or vision-language model.  It
extracts the water interface from the laser-off background and the red optical
path from temporal/color differences, then fits the incident and transmitted
rays independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from continuous_scoring import migrate_record


EPS = 1.0e-9


def finite_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def robust_mad(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0
    med = float(np.median(array))
    return float(1.4826 * np.median(np.abs(array - med)))


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id_from_stem(stem: str) -> str:
    marker = "_seed"
    return stem.split(marker, 1)[0] if marker in stem else stem


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def decode_video(path: Path) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    frames: List[np.ndarray] = []
    timestamps_ms: List[float] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
        timestamps_ms.append(float(capture.get(cv2.CAP_PROP_POS_MSEC)))
    capture.release()
    if not frames:
        raise RuntimeError(f"video contains no decodable frames: {path}")
    height, width = frames[0].shape[:2]
    shape_consistent = all(frame.shape[:2] == (height, width) for frame in frames)
    if not shape_consistent:
        raise RuntimeError("video frame dimensions change during decoding")
    duration = ((len(frames) - 1) / fps) if fps > 0 else None
    return frames, {
        "width": width,
        "height": height,
        "fps": finite_or_none(fps),
        "decoded_frame_count": len(frames),
        "declared_frame_count": declared_count,
        "duration_seconds": finite_or_none(duration),
        "timestamps_monotonic": bool(
            all(b + 1.0e-6 >= a for a, b in zip(timestamps_ms, timestamps_ms[1:]))
        ),
    }


def weighted_line_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    iterations: int = 8,
    orthogonal: bool = False,
) -> Optional[Dict[str, Any]]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if weights is None:
        base_weights = np.ones_like(x)
    else:
        base_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        valid &= np.isfinite(base_weights) & (base_weights > 0)
    x, y, base_weights = x[valid], y[valid], base_weights[valid]
    if x.size < 8 or np.ptp(x) < 3.0:
        return None
    design = np.column_stack([x, np.ones_like(x)])
    robust_weights = np.ones_like(x)
    beta = np.array([0.0, float(np.median(y))])
    for _ in range(iterations):
        total = np.clip(base_weights * robust_weights, 1.0e-6, None)
        root = np.sqrt(total)
        if orthogonal:
            points=np.column_stack([x,y]);center=np.average(points,axis=0,weights=total)
            centered=points-center;cov=(centered*total[:,None]).T@centered/total.sum()
            _,vectors=np.linalg.eigh(cov);direction=vectors[:,-1]
            if abs(direction[0])<1e-6:return None
            slope=direction[1]/direction[0];beta=np.array([slope,center[1]-slope*center[0]])
        else:
            beta, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
        residual = (y - design @ beta)/(math.sqrt(1+beta[0]**2) if orthogonal else 1.)
        scale = max(0.35, robust_mad(residual))
        cutoff = 1.5 * scale
        robust_weights = np.ones_like(residual)
        large = np.abs(residual) > cutoff
        robust_weights[large] = cutoff / np.maximum(np.abs(residual[large]), EPS)
    residual = y - design @ beta
    scale = max(0.35, robust_mad(residual))
    inlier = np.abs(residual) <= max(2.5, 3.0 * scale)
    if int(inlier.sum()) < 8:
        return None
    rmse = float(np.sqrt(np.mean(np.square(residual[inlier]))))
    return {
        "slope": float(beta[0]),
        "intercept": float(beta[1]),
        "rmse_px": rmse,
        "inlier_count": int(inlier.sum()),
        "point_count": int(x.size),
        "inlier_fraction": float(inlier.mean()),
        "x_min": float(np.min(x[inlier])),
        "x_max": float(np.max(x[inlier])),
        "x_span_px": float(np.ptp(x[inlier])),
    }


def red_change_map(frame: np.ndarray, off_reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Color-vector temporal contrast detects visible beams of any hue. The
    # legacy name is retained for callers; red-channel preference is removed.
    delta=frame.astype(np.float32)-off_reference.astype(np.float32)
    contrast=np.linalg.norm(delta,axis=2)
    background=cv2.GaussianBlur(contrast,(0,0),sigmaX=11.,sigmaY=11.)
    local=np.maximum(contrast-background,0.)
    return local,local


def select_temporal_references(
    frames: Sequence[np.ndarray],
    tank_bounds: Tuple[float, float],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    temporal = config["temporal"]
    beam_cfg = config["beam"]
    count = len(frames)
    off_count = int(round(count * float(temporal["off_frame_fraction"])))
    off_count = max(int(temporal["off_frame_min"]), off_count)
    off_count = min(int(temporal["off_frame_max"]), off_count, max(1, count // 3))
    off_reference = np.median(np.stack(frames[:off_count], axis=0), axis=0).astype(np.uint8)
    height, width = frames[0].shape[:2]
    left = int(max(0, round(tank_bounds[0])))
    right = int(min(width, round(tank_bounds[1])))
    roi = np.zeros((height, width), dtype=bool)
    roi[max(0, int(0.02 * height)) : min(height, int(0.97 * height)), left:right] = True
    pixel_counts: List[int] = []
    strengths: List[float] = []
    for frame in frames:
        score, evidence = red_change_map(frame, off_reference)
        candidate = roi & (score >= float(beam_cfg["score_floor"])) & (
            evidence >= float(beam_cfg.get("local_red_dominance_min", beam_cfg["min_red_dominance"]))
        )
        pixel_counts.append(int(candidate.sum()))
        active_scores = score[candidate]
        strengths.append(float(np.median(active_scores)) if active_scores.size else 0.0)
    baseline = np.asarray(pixel_counts[:off_count], dtype=np.float64)
    baseline_med = float(np.median(baseline)) if baseline.size else 0.0
    baseline_mad = robust_mad(baseline)
    min_area = float(temporal["onset_min_red_pixels_norm"]) * height * width
    threshold = max(
        baseline_med + float(temporal["onset_sigma"]) * max(1.0, baseline_mad),
        baseline_med + min_area,
    )
    candidates = [index for index in range(off_count, count) if pixel_counts[index] >= threshold]
    stable_count = int(temporal["stable_on_frame_count"])
    selected = candidates[-stable_count:]
    if len(selected) < int(temporal["min_on_frames"]):
        order = sorted(range(off_count, count), key=lambda i: pixel_counts[i], reverse=True)
        selected = sorted(order[: min(stable_count, max(0, count - off_count))])
    laser_on = bool(
        len(selected) >= int(temporal["min_on_frames"])
        and selected
        and float(np.median([pixel_counts[i] for i in selected])) >= threshold
    )
    if selected:
        on_reference = np.median(np.stack([frames[i] for i in selected], axis=0), axis=0).astype(np.uint8)
        representative_index = int(selected[len(selected) // 2])
    else:
        on_reference = frames[-1].copy()
        representative_index = count - 1
    return {
        "off_reference": off_reference,
        "on_reference": on_reference,
        "off_frame_indices": list(range(off_count)),
        "on_frame_indices": selected,
        "representative_on_frame_index": representative_index,
        "red_pixel_counts": pixel_counts,
        "red_strengths": strengths,
        "onset_pixel_threshold": float(threshold),
        "laser_on_detected": laser_on,
    }


def fit_surface(
    off_reference: np.ndarray,
    tank_bounds: Tuple[float, float],
    expected_y: float,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    surface_cfg = config["surface"]
    height, width = off_reference.shape[:2]
    left = max(0, int(round(tank_bounds[0])))
    right = min(width, int(round(tank_bounds[1])))
    lab = cv2.cvtColor(off_reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_chan, a_chan, b_chan = cv2.split(lab)
    gy = np.abs(cv2.Sobel(l_chan, cv2.CV_32F, 0, 1, ksize=3))
    gy += 0.35 * np.abs(cv2.Sobel(a_chan, cv2.CV_32F, 0, 1, ksize=3))
    gy += 0.35 * np.abs(cv2.Sobel(b_chan, cv2.CV_32F, 0, 1, ksize=3))
    half = int(round(float(surface_cfg["search_half_height_norm"]) * height))
    y0 = max(2, int(round(expected_y)) - half)
    y1 = min(height - 2, int(round(expected_y)) + half + 1)
    if right - left < 30 or y1 - y0 < 3:
        return None
    region = gy[y0:y1, left:right]
    row_median = np.median(region, axis=1)
    row_upper = np.percentile(region, 75.0, axis=1)
    row_score = row_median + 0.35 * row_upper
    row_score = cv2.GaussianBlur(row_score.reshape(-1, 1), (1, 5), 0).reshape(-1)
    best_row = int(y0 + int(np.argmax(row_score)))
    fit_half = int(surface_cfg["fit_half_height_px"])
    yy0, yy1 = max(0, best_row - fit_half), min(height, best_row + fit_half + 1)
    patch = gy[yy0:yy1, left:right]
    threshold = float(np.percentile(patch, 65.0))
    ys, xs = np.nonzero(patch >= threshold)
    if xs.size < 16:
        return None
    xs = xs.astype(np.float64) + left
    ys = ys.astype(np.float64) + yy0
    weights = patch[(ys - yy0).astype(int), (xs - left).astype(int)].astype(np.float64)
    # Prefer edge pixels close to the strongest row; this rejects nearby tank rims.
    weights *= np.exp(-0.5 * np.square((ys - best_row) / max(1.5, fit_half / 2.0)))
    result = weighted_line_fit(xs, ys, weights)
    if result is None or abs(float(result["slope"])) > float(surface_cfg["max_abs_slope"]):
        return None
    band = np.asarray(row_score, dtype=np.float64)
    peak_z = (float(np.max(band)) - float(np.median(band))) / max(1.0, robust_mad(band))
    result.update(
        {
            "kind": "air_water_interface",
            "fit_source": "laser_off_background_gradient_irls",
            "expected_y_px": float(expected_y),
            "detected_row_px": float(best_row),
            "row_peak_robust_z": float(peak_z),
            "span_tank_fraction": float(result["x_span_px"] / max(1.0, tank_bounds[1] - tank_bounds[0])),
        }
    )
    return result


def make_beam_mask(
    frame: np.ndarray,
    off_reference: np.ndarray,
    tank_bounds: Tuple[float, float],
    config: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    beam_cfg = config["beam"]
    score, evidence = red_change_map(frame, off_reference)
    height, width = score.shape
    left, right = int(round(tank_bounds[0])), int(round(tank_bounds[1]))
    roi = np.zeros_like(score, dtype=bool)
    roi[max(0, int(0.015 * height)) : min(height, int(0.975 * height)), max(0, left) : min(width, right)] = True
    values = score[roi]
    median = float(np.median(values)) if values.size else 0.0
    sigma = robust_mad(values)
    threshold = max(float(beam_cfg["score_floor"]), median + float(beam_cfg["score_sigma"]) * max(0.5, sigma))
    mask = roi & (score >= threshold) & (
        evidence >= float(beam_cfg.get("local_red_dominance_min", beam_cfg["min_red_dominance"]))
    )
    binary = (mask.astype(np.uint8) * 255)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return binary, score, {
        "adaptive_score_threshold": float(threshold),
        "beam_mask_pixel_count": int(np.count_nonzero(binary)),
        "beam_mask_fraction": float(np.count_nonzero(binary) / max(1, binary.size)),
    }


def line_surface_intersection(line: Dict[str, Any], surface: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    denominator = float(line["slope"]) - float(surface["slope"])
    if abs(denominator) < 1.0e-5:
        return None
    x = (float(surface["intercept"]) - float(line["intercept"])) / denominator
    y = float(surface["slope"]) * x + float(surface["intercept"])
    return float(x), float(y)


def fit_ray(
    mask: np.ndarray,
    score_map: np.ndarray,
    surface: Dict[str, Any],
    tank_bounds: Tuple[float, float],
    medium: str,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    assert medium in {"air", "water"}
    beam_cfg = config["beam"]
    height, width = mask.shape
    tank_width = max(1.0, tank_bounds[1] - tank_bounds[0])
    candidate_mask = mask.copy()
    ys_grid, xs_grid = np.indices(mask.shape)
    surface_y = float(surface["slope"]) * xs_grid + float(surface["intercept"])
    signed = ys_grid - surface_y
    exclusion = float(beam_cfg["surface_exclusion_px"])
    if medium == "air":
        candidate_mask[signed >= -exclusion] = 0
    else:
        candidate_mask[signed <= exclusion] = 0
    candidate_mask[:, : max(0, int(tank_bounds[0]))] = 0
    candidate_mask[:, min(width, int(tank_bounds[1]) + 1) :] = 0
    hough_input = cv2.dilate(candidate_mask, np.ones((3, 3), np.uint8), iterations=1)
    lines = cv2.HoughLinesP(
        hough_input,
        1,
        np.pi / 720.0,
        threshold=int(beam_cfg["hough_threshold"]),
        minLineLength=max(10, int(float(beam_cfg["hough_min_length_tank_fraction"]) * tank_width)),
        maxLineGap=max(4, int(float(beam_cfg["hough_max_gap_tank_fraction"]) * tank_width)),
    )
    candidates: List[Tuple[float, float, float, float, float, float]] = []
    if lines is not None:
        for raw in lines[:, 0, :]:
            x1, y1, x2, y2 = map(float, raw)
            if abs(x2 - x1) < 2.0:
                continue
            if x2 < x1:
                x1, x2, y1, y2 = x2, x1, y2, y1
            slope = (y2 - y1) / (x2 - x1)
            if not (float(beam_cfg["min_positive_slope"]) <= slope <= float(beam_cfg["max_positive_slope"])):
                continue
            intercept = y1 - slope * x1
            probe = {"slope": slope, "intercept": intercept}
            intersection = line_surface_intersection(probe, surface)
            if intersection is None:
                continue
            xi, _ = intersection
            margin = 0.12 * tank_width
            if not (tank_bounds[0] - margin <= xi <= tank_bounds[1] + margin):
                continue
            length = math.hypot(x2 - x1, y2 - y1)
            endpoint_near_surface = min(
                abs(y1 - (float(surface["slope"]) * x1 + float(surface["intercept"]))),
                abs(y2 - (float(surface["slope"]) * x2 + float(surface["intercept"]))),
            )
            side_ok = (x2 <= xi + 0.10 * tank_width) if medium == "air" else (x1 >= xi - 0.10 * tank_width)
            ranking = length * (1.25 if side_ok else 0.55) / (1.0 + 0.015 * endpoint_near_surface)
            candidates.append((ranking, slope, intercept, x1, x2, length))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, slope0, intercept0, _, _, _ = candidates[0]
    ys, xs = np.nonzero(candidate_mask > 0)
    if xs.size < int(beam_cfg["min_inliers"]):
        return None
    distance = np.abs(ys - (slope0 * xs + intercept0)) / math.sqrt(1.0 + slope0 * slope0)
    close = distance <= float(beam_cfg["refine_half_width_px"])
    xs_close, ys_close = xs[close].astype(np.float64), ys[close].astype(np.float64)
    if xs_close.size < int(beam_cfg["min_inliers"]):
        return None
    weights = np.maximum(1.0, score_map[ys[close], xs[close]].astype(np.float64))
    fit = weighted_line_fit(xs_close, ys_close, weights, orthogonal=True)
    if fit is None:
        return None
    # One refinement pass around the robust estimate.
    distance2 = np.abs(ys - (float(fit["slope"]) * xs + float(fit["intercept"]))) / math.sqrt(
        1.0 + float(fit["slope"]) ** 2
    )
    close2 = distance2 <= float(beam_cfg["refine_half_width_px"])
    fit2 = weighted_line_fit(
        xs[close2].astype(np.float64),
        ys[close2].astype(np.float64),
        np.maximum(1.0, score_map[ys[close2], xs[close2]].astype(np.float64)),
        orthogonal=True,
    )
    if fit2 is not None:
        fit = fit2
    fit.update(
        {
            "kind": "incident_ray" if medium == "air" else "refracted_ray",
            "medium": medium,
            "fit_source": "temporal_color_change_hough_orthogonal_irls",
            "span_tank_fraction": float(fit["x_span_px"] / tank_width),
            "hough_candidate_count": len(candidates),
        }
    )
    return fit


def acute_angle_between_lines_deg(slope_a: float, slope_b: float) -> float:
    vector_a = np.array([1.0, float(slope_a)], dtype=np.float64)
    vector_b = np.array([1.0, float(slope_b)], dtype=np.float64)
    cosine = abs(float(np.dot(vector_a, vector_b))) / max(EPS, float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))
    return float(math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0)))))


def angle_from_normal_deg(ray: Dict[str, Any], surface: Dict[str, Any]) -> float:
    angle_from_surface = acute_angle_between_lines_deg(float(ray["slope"]), float(surface["slope"]))
    return float(90.0 - angle_from_surface)


def camera_motion(
    off_reference: np.ndarray,
    on_reference: np.ndarray,
    exclusion_mask: Optional[np.ndarray],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    gray0 = cv2.cvtColor(off_reference, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(on_reference, cv2.COLOR_BGR2GRAY)
    mask = np.full(gray0.shape, 255, dtype=np.uint8)
    if exclusion_mask is not None:
        expanded = cv2.dilate(exclusion_mask, np.ones((17, 17), np.uint8), iterations=1)
        mask[expanded > 0] = 0
    try:
        detector = cv2.SIFT_create(nfeatures=1500)
        norm = cv2.NORM_L2
        ratio = 0.75
        method = "SIFT_affine_ransac"
    except Exception:
        detector = cv2.ORB_create(nfeatures=1800)
        norm = cv2.NORM_HAMMING
        ratio = 0.80
        method = "ORB_affine_ransac"
    kp0, des0 = detector.detectAndCompute(gray0, mask)
    kp1, des1 = detector.detectAndCompute(gray1, mask)
    good = []
    if des0 is not None and des1 is not None and len(des0) >= 2 and len(des1) >= 2:
        matcher = cv2.BFMatcher(norm)
        for pair in matcher.knnMatch(des0, des1, k=2):
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                good.append(pair[0])
    affine = None
    inliers = None
    if len(good) >= 6:
        src = np.float32([kp0[m.queryIdx].pt for m in good])
        dst = np.float32([kp1[m.trainIdx].pt for m in good])
        affine, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=3000, confidence=0.995
        )
    if affine is None:
        shift, response = cv2.phaseCorrelate(gray0.astype(np.float32), gray1.astype(np.float32))
        translation = float(math.hypot(shift[0], shift[1]))
        scale = 1.0
        rotation = 0.0
        inlier_count = 0
        method = "phase_correlation_fallback"
        response_value = float(response)
    else:
        a, b = float(affine[0, 0]), float(affine[0, 1])
        scale = float(math.sqrt(a * a + b * b))
        rotation = float(math.degrees(math.atan2(b, a)))
        translation = float(math.hypot(float(affine[0, 2]), float(affine[1, 2])))
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        response_value = None
    diagonal = math.hypot(gray0.shape[1], gray0.shape[0])
    camera_cfg = config["camera"]
    enough_matches = inlier_count >= int(camera_cfg["min_inlier_matches"]) or method == "phase_correlation_fallback"
    static = bool(
        enough_matches
        and translation / max(1.0, diagonal) <= float(camera_cfg["max_translation_diagonal_fraction"])
        and abs(scale - 1.0) <= float(camera_cfg["max_scale_error"])
        and abs(rotation) <= float(camera_cfg["max_rotation_deg"])
    )
    return {
        "method": method,
        "matched_features": len(good),
        "inlier_matches": inlier_count,
        "translation_px": translation,
        "translation_diagonal_fraction": float(translation / max(1.0, diagonal)),
        "scale": scale,
        "scale_error": abs(scale - 1.0),
        "rotation_deg": rotation,
        "phase_response": response_value,
        "static": static,
    }


def line_quality_ok(line: Dict[str, Any], medium: str, config: Dict[str, Any]) -> bool:
    beam_cfg = config["beam"]
    min_span_key = "min_air_span_tank_fraction" if medium == "air" else "min_water_span_tank_fraction"
    return bool(
        int(line.get("inlier_count", 0)) >= int(beam_cfg["min_inliers"])
        and float(line.get("span_tank_fraction", 0.0)) >= float(beam_cfg[min_span_key])
        and float(line.get("rmse_px", float("inf"))) <= float(beam_cfg["max_ray_rmse_px"])
        and float(beam_cfg["min_positive_slope"]) <= float(line.get("slope", -1.0)) <= float(beam_cfg["max_positive_slope"])
    )


def surface_quality_ok(surface: Dict[str, Any], config: Dict[str, Any]) -> bool:
    cfg = config["surface"]
    return bool(
        float(surface.get("span_tank_fraction", 0.0)) >= float(cfg["min_span_tank_fraction"])
        and float(surface.get("rmse_px", float("inf"))) <= float(cfg["max_rmse_px"])
        and abs(float(surface.get("slope", float("inf")))) <= float(cfg["max_abs_slope"])
    )


def piecewise_score(value: Optional[float], good: float, bad: float) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    if value <= good:
        return 100.0
    if value >= bad:
        return 0.0
    return float(100.0 * (bad - value) / max(EPS, bad - good))


def temporal_ray_stability(
    frames: Sequence[np.ndarray],
    selected_indices: Sequence[int],
    off_reference: np.ndarray,
    surface: Dict[str, Any],
    tank_bounds: Tuple[float, float],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    if not selected_indices:
        return {"sample_count": 0, "incident_angle_mad_deg": None, "refracted_angle_mad_deg": None}
    positions = np.linspace(0, len(selected_indices) - 1, num=min(7, len(selected_indices)), dtype=int)
    incident_angles: List[float] = []
    refracted_angles: List[float] = []
    sampled_indices: List[int] = []
    for position in sorted(set(map(int, positions.tolist()))):
        index = int(selected_indices[position])
        mask, score, _ = make_beam_mask(frames[index], off_reference, tank_bounds, config)
        air = fit_ray(mask, score, surface, tank_bounds, "air", config)
        water = fit_ray(mask, score, surface, tank_bounds, "water", config)
        if air is not None and water is not None:
            sampled_indices.append(index)
            incident_angles.append(angle_from_normal_deg(air, surface))
            refracted_angles.append(angle_from_normal_deg(water, surface))
    return {
        "sampled_frame_indices": sampled_indices,
        "sample_count": len(sampled_indices),
        "incident_angles_from_normal_deg": incident_angles,
        "refracted_angles_from_normal_deg": refracted_angles,
        "incident_angle_median_deg": finite_or_none(np.median(incident_angles)) if incident_angles else None,
        "refracted_angle_median_deg": finite_or_none(np.median(refracted_angles)) if refracted_angles else None,
        "incident_angle_mad_deg": robust_mad(incident_angles) if len(incident_angles) >= 3 else None,
        "refracted_angle_mad_deg": robust_mad(refracted_angles) if len(refracted_angles) >= 3 else None,
    }


def draw_parametric_line(
    image: np.ndarray,
    line: Dict[str, Any],
    x_start: float,
    x_end: float,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    height, width = image.shape[:2]
    x1, x2 = float(x_start), float(x_end)
    y1 = float(line["slope"]) * x1 + float(line["intercept"])
    y2 = float(line["slope"]) * x2 + float(line["intercept"])
    points = [(int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))]
    cv2.line(image, points[0], points[1], color, thickness, cv2.LINE_AA)


def annotate_frame(frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    image = frame.copy()
    geometry = result.get("geometry", {})
    tank = geometry.get("tank") or {}
    left, right = tank.get("left_px"), tank.get("right_px")
    surface = geometry.get("surface")
    air = geometry.get("incident_ray")
    water = geometry.get("refracted_ray")
    intersections = geometry.get("intersections") or {}
    if left is not None and right is not None:
        cv2.rectangle(image, (int(left), 1), (int(right), image.shape[0] - 2), (160, 160, 160), 1)
    if surface is not None and left is not None and right is not None:
        draw_parametric_line(image, surface, left, right, (255, 220, 0), 2)
    air_point = intersections.get("incident_with_surface")
    water_point = intersections.get("refracted_with_surface")
    if air is not None and air_point is not None and left is not None:
        draw_parametric_line(image, air, left, float(air_point[0]), (0, 165, 255), 3)
        cv2.circle(image, tuple(map(lambda v: int(round(v)), air_point)), 7, (0, 165, 255), 2, cv2.LINE_AA)
    if water is not None and water_point is not None and right is not None:
        draw_parametric_line(image, water, float(water_point[0]), right, (0, 255, 255), 3)
        cv2.circle(image, tuple(map(lambda v: int(round(v)), water_point)), 7, (0, 255, 255), 2, cv2.LINE_AA)
    metrics = result.get("metrics", {})
    statuses = result.get("statuses", {})
    m1 = metrics.get("M1_snell_residual_abs")
    m2 = metrics.get("M2_intersection_disagreement_normalized")
    text_lines = [
        f"P11 extract={statuses.get('extract_success')} structure={statuses.get('structural_ok')}",
        f"valid={statuses.get('measurement_valid')} physics_pass={statuses.get('physics_pass')}",
        f"M1 |Snell residual|={m1:.4f}" if isinstance(m1, (int, float)) else "M1 |Snell residual|=null",
        f"M2 intersection/tank={m2:.4f}" if isinstance(m2, (int, float)) else "M2 intersection/tank=null",
    ]
    y = 30
    for text_line in text_lines:
        cv2.putText(image, text_line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(image, text_line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
        y += 27
    return image


def write_debug_artifacts(
    frames: Sequence[np.ndarray],
    temporal: Dict[str, Any],
    beam_mask: np.ndarray,
    result: Dict[str, Any],
    debug_dir: Path,
    fps: float,
) -> Dict[str, Any]:
    debug_dir.mkdir(parents=True, exist_ok=True)
    off_path = debug_dir / "keyframe_off.png"
    on_path = debug_dir / "keyframe_on.png"
    mask_path = debug_dir / "beam_mask.png"
    overlay_path = debug_dir / "geometry_overlay.png"
    video_path = debug_dir / "overlay.mp4"
    plot_path = debug_dir / "temporal_diagnostic.png"
    cv2.imwrite(str(off_path), temporal["off_reference"])
    cv2.imwrite(str(on_path), temporal["on_reference"])
    cv2.imwrite(str(mask_path), beam_mask)
    overlay = annotate_frame(temporal["on_reference"], result)
    cv2.imwrite(str(overlay_path), overlay)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, float(fps)), (width, height))
    overlay_written = bool(writer.isOpened())
    if overlay_written:
        for frame in frames:
            writer.write(annotate_frame(frame, result))
    writer.release()
    plot_written = False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        counts = temporal["red_pixel_counts"]
        figure, axis = plt.subplots(figsize=(9, 3.2), constrained_layout=True)
        axis.plot(np.arange(len(counts)), counts, color="#c62828", lw=1.8, label="red-change pixels")
        axis.axhline(float(temporal["onset_pixel_threshold"]), color="#1565c0", ls="--", label="onset threshold")
        for index in temporal["on_frame_indices"]:
            axis.axvline(index, color="#ff8f00", alpha=0.08)
        axis.set_xlabel("decoded frame")
        axis.set_ylabel("pixels")
        axis.set_title("Laser-on detection (no semantic model)")
        axis.grid(alpha=0.2)
        axis.legend(loc="best")
        figure.savefig(plot_path, dpi=150)
        plt.close(figure)
        plot_written = True
    except Exception as exc:
        result.setdefault("warnings", []).append(f"plot_generation_failed:{type(exc).__name__}:{exc}")
    return {
        "directory": str(debug_dir.resolve()),
        "keyframe_off": str(off_path.resolve()),
        "keyframe_on": str(on_path.resolve()),
        "beam_mask": str(mask_path.resolve()),
        "geometry_overlay": str(overlay_path.resolve()),
        "overlay_video": str(video_path.resolve()) if overlay_written else None,
        "temporal_plot": str(plot_path.resolve()) if plot_written else None,
    }


def evaluate_video(
    video_path: Path,
    task_id: str,
    config_path: Path,
    output_path: Optional[Path] = None,
    debug_dir: Optional[Path] = None,
    write_debug: bool = True,
) -> Dict[str, Any]:
    config = load_config(config_path)
    if task_id != "P11":
        raise ValueError(f"this evaluator only supports P11, received {task_id}")
    cv2.setNumThreads(int(config.get("runtime", {}).get("opencv_threads", 1)))
    frames, metadata = decode_video(video_path)
    height, width = frames[0].shape[:2]
    stem = video_path.stem
    sample_id = sample_id_from_stem(stem)
    sample_cfg = config["samples"].get(sample_id, config["generic_sample"])
    tank_left = float(sample_cfg["tank_left_norm"]) * width
    tank_right = float(sample_cfg["tank_right_norm"]) * width
    tank_width = tank_right - tank_left
    tank_bounds = (tank_left, tank_right)
    temporal = select_temporal_references(frames, tank_bounds, config)
    expected_surface_y = float(sample_cfg["surface_y_norm"]) * height
    surface = fit_surface(temporal["off_reference"], tank_bounds, expected_surface_y, config)
    beam_mask, beam_score, beam_diagnostics = make_beam_mask(
        temporal["on_reference"], temporal["off_reference"], tank_bounds, config
    )
    incident = fit_ray(beam_mask, beam_score, surface, tank_bounds, "air", config) if surface is not None else None
    refracted = fit_ray(beam_mask, beam_score, surface, tank_bounds, "water", config) if surface is not None else None
    camera = camera_motion(temporal["off_reference"], temporal["on_reference"], beam_mask, config)
    extract_reasons: List[str] = []
    if surface is None:
        extract_reasons.append("water_surface_not_extracted")
    if incident is None:
        extract_reasons.append("incident_ray_not_extracted")
    if refracted is None:
        extract_reasons.append("refracted_ray_not_extracted")
    extract_success = not extract_reasons
    surface_ok = surface_quality_ok(surface, config) if surface is not None else False
    incident_ok = line_quality_ok(incident, "air", config) if incident is not None else False
    refracted_ok = line_quality_ok(refracted, "water", config) if refracted is not None else False
    structural_reasons: List[str] = []
    if not temporal["laser_on_detected"]:
        structural_reasons.append("laser_never_reliably_switches_on")
    if not surface_ok:
        structural_reasons.append("water_surface_missing_or_unstable")
    if incident is None:
        structural_reasons.append("main_incident_path_missing")
    if refracted is None:
        structural_reasons.append("main_refracted_path_missing")
    if not camera["static"]:
        structural_reasons.append("camera_or_scene_motion_exceeds_gate")
    structural_ok = not structural_reasons
    air_intersection = line_surface_intersection(incident, surface) if incident is not None and surface is not None else None
    water_intersection = line_surface_intersection(refracted, surface) if refracted is not None and surface is not None else None
    theta_i = angle_from_normal_deg(incident, surface) if incident is not None and surface is not None else None
    theta_t = angle_from_normal_deg(refracted, surface) if refracted is not None and surface is not None else None
    temporal_stability = (
        temporal_ray_stability(
            frames,
            temporal["on_frame_indices"],
            temporal["off_reference"],
            surface,
            tank_bounds,
            config,
        )
        if surface is not None
        else {"sample_count": 0, "incident_angle_mad_deg": None, "refracted_angle_mad_deg": None}
    )
    validity_reasons: List[str] = []
    if not extract_success:
        validity_reasons.extend(extract_reasons)
    if surface is not None and not surface_ok:
        validity_reasons.append("surface_fit_quality_gate_failed")
    if incident is not None and not incident_ok:
        validity_reasons.append("incident_fit_quality_gate_failed")
    if refracted is not None and not refracted_ok:
        validity_reasons.append("refracted_fit_quality_gate_failed")
    margin = float(config["validity"]["intersection_margin_tank_fraction"]) * tank_width
    for label, point in (("incident", air_intersection), ("refracted", water_intersection)):
        if point is None:
            validity_reasons.append(f"{label}_surface_intersection_undefined")
        elif not (tank_left - margin <= point[0] <= tank_right + margin):
            validity_reasons.append(f"{label}_surface_intersection_outside_tank")
    min_angle = float(config["validity"]["min_angle_from_normal_deg"])
    max_angle = float(config["validity"]["max_angle_from_normal_deg"])
    for label, angle in (("incident", theta_i), ("refracted", theta_t)):
        if angle is None or not (min_angle <= angle <= max_angle):
            validity_reasons.append(f"{label}_angle_outside_valid_range")
    max_mad = float(config["validity"]["max_temporal_angle_mad_deg"])
    for label in ("incident", "refracted"):
        mad = temporal_stability.get(f"{label}_angle_mad_deg")
        if mad is not None and float(mad) > max_mad:
            validity_reasons.append(f"{label}_angle_temporally_unstable")
    measurement_valid = bool(extract_success and surface_ok and incident_ok and refracted_ok and not validity_reasons)
    expected_n = float(config["expected_refractive_index_ratio"])
    ratio = None
    m1_signed = None
    m1_abs = None
    m2 = None
    if measurement_valid and theta_i is not None and theta_t is not None:
        denominator = math.sin(math.radians(theta_t))
        if abs(denominator) > 1.0e-6:
            ratio = math.sin(math.radians(theta_i)) / denominator
            m1_signed = ratio - expected_n
            m1_abs = abs(m1_signed)
    if measurement_valid and air_intersection is not None and water_intersection is not None:
        m2 = abs(float(air_intersection[0]) - float(water_intersection[0])) / max(1.0, tank_width)
    metric_validity = {
        "M1_snell_residual": bool(measurement_valid and m1_abs is not None and math.isfinite(m1_abs)),
        "M2_intersection_consistency": bool(measurement_valid and m2 is not None and math.isfinite(m2)),
    }
    score_cfg = config["score"]
    m1_score = piecewise_score(m1_abs, float(score_cfg["m1_good"]), float(score_cfg["m1_bad"]))
    m2_score = piecewise_score(m2, float(score_cfg["m2_good"]), float(score_cfg["m2_bad"]))
    metric_score = None
    if m1_score is not None and m2_score is not None:
        metric_score = float(float(score_cfg["m1_weight"]) * m1_score + float(score_cfg["m2_weight"]) * m2_score)
    overall_score = metric_score if structural_ok and measurement_valid and metric_score is not None else 0.0
    physics_pass = None
    if all(metric_validity.values()):
        physics_pass = bool(m1_abs <= float(score_cfg["m1_pass"]) and m2 <= float(score_cfg["m2_pass"]))
    result: Dict[str, Any] = {
        "schema_version": "physical-bench-evaluator-1.0",
        "task_id": "P11",
        "sample_id": sample_id,
        "video_id": stem,
        "video_path": str(video_path.resolve()),
        "evaluator_version": config["evaluator_version"],
        "config_path": str(config_path.resolve()),
        "config_sha256": hash_file(config_path),
        "model_policy": {
            "uses_llm": False,
            "uses_vlm": False,
            "coordinate_extractors": ["OpenCV temporal difference", "Lab/HSV-style red evidence", "Hough", "IRLS"],
        },
        "video_metadata": metadata,
        "statuses": {
            "extract_success": extract_success,
            "structural_ok": structural_ok,
            "measurement_valid": measurement_valid,
            "metric_validity": metric_validity,
            "physics_pass": physics_pass,
            "extract_failure_reasons": extract_reasons,
            "structural_failure_reasons": structural_reasons,
            "measurement_invalid_reasons": sorted(set(validity_reasons)),
        },
        "temporal": {
            "off_frame_indices": temporal["off_frame_indices"],
            "stable_on_frame_indices": temporal["on_frame_indices"],
            "representative_on_frame_index": temporal["representative_on_frame_index"],
            "laser_on_detected": temporal["laser_on_detected"],
            "onset_pixel_threshold": temporal["onset_pixel_threshold"],
            "red_pixel_counts": temporal["red_pixel_counts"],
            "ray_stability": temporal_stability,
        },
        "structure": {
            "camera": camera,
            "surface_fit_ok": surface_ok,
            "incident_path_present": incident is not None,
            "refracted_path_present": refracted is not None,
            "beam_diagnostics": beam_diagnostics,
        },
        "geometry": {
            "tank": {
                "left_px": tank_left,
                "right_px": tank_right,
                "frozen_width_px": tank_width,
                "source": "sample_first-frame apparatus calibration in frozen config",
            },
            "surface": surface,
            "incident_ray": incident,
            "refracted_ray": refracted,
            "intersections": {
                "incident_with_surface": list(air_intersection) if air_intersection is not None else None,
                "refracted_with_surface": list(water_intersection) if water_intersection is not None else None,
                "x_disagreement_px": (
                    abs(float(air_intersection[0]) - float(water_intersection[0]))
                    if air_intersection is not None and water_intersection is not None
                    else None
                ),
            },
            "angles": {
                "nominal_incidence_from_normal_deg": sample_cfg.get("nominal_incidence_deg"),
                "measured_incidence_from_normal_deg": theta_i,
                "measured_refraction_from_normal_deg": theta_t,
                "surface_normal_source": "computed as perpendicular to independently fitted water interface",
                "drawn_normal_used_as_truth": False,
            },
        },
        "metrics": {
            "M1_snell_ratio": ratio,
            "M1_expected_ratio": expected_n,
            "M1_snell_residual_signed": m1_signed,
            "M1_snell_residual_abs": m1_abs,
            "M2_intersection_disagreement_normalized": m2,
            "M2_normalizer": "frozen first-frame tank width",
        },
        "scores": {
            "M1_score_0_100": m1_score,
            "M2_score_0_100": m2_score,
            "geometric_metric_score_0_100": metric_score,
            "overall_gated_score_0_100": overall_score,
            "weights": {"M1": score_cfg["m1_weight"], "M2": score_cfg["m2_weight"]},
        },
        "warnings": [],
    }
    # Keep the original 0--100 score as an explicit legacy field, while making
    # the smooth 0--1 score the primary schema for all future evaluations.
    result = migrate_record(result)
    if write_debug:
        if debug_dir is None:
            base = output_path.parent if output_path is not None else video_path.parent
            debug_dir = base / f"{stem}_debug"
        result["debug"] = write_debug_artifacts(
            frames,
            temporal,
            beam_mask,
            result,
            debug_dir,
            float(metadata.get("fps") or 24.0),
        )
    else:
        result["debug"] = None
    result = to_builtin(result)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--no-debug", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = evaluate_video(
            args.video,
            args.task_id,
            args.config,
            output_path=args.output,
            debug_dir=args.debug_dir,
            write_debug=not args.no_debug,
        )
    except Exception as exc:
        failure = {
            "schema_version": "physical-bench-evaluator-1.0",
            "task_id": args.task_id,
            "video_path": str(args.video),
            "statuses": {
                "extract_success": False,
                "structural_ok": False,
                "measurement_valid": False,
                "metric_validity": {"M1_snell_residual": False, "M2_intersection_consistency": False},
                "physics_pass": None,
                "extract_failure_reasons": [f"fatal:{type(exc).__name__}:{exc}"],
            },
            "metrics": {
                "M1_snell_ratio": None,
                "M1_snell_residual_signed": None,
                "M1_snell_residual_abs": None,
                "M2_intersection_disagreement_normalized": None,
            },
            "scores": {"geometric_metric_score_0_100": None, "overall_gated_score_0_100": 0.0},
        }
        failure = migrate_record(failure)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(failure, handle, ensure_ascii=False, indent=2)
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "video_id": result["video_id"],
                "statuses": result["statuses"],
                "metrics": result["metrics"],
                "scores": result["scores"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
