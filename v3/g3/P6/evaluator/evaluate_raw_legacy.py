#!/usr/bin/env python3
"""Deterministic P6 pure-rolling evaluator (no LLM/VLM).

The evaluator uses only classical computer vision and robust geometry:

* background KLT for camera-motion diagnostics;
* local circle Hough + ball-interior KLT for ball centre/radius tracking;
* colour/edge line estimators for the asymmetric surface band;
* robust temporal unwrapping and Savitzky-Golay derivatives;
* the rolling constraints v = omega R and v_contact = v - omega R.

All fitted tracks are independently visualised.  Robust fitting never removes
frames from the exported diagnostics; inlier masks and confidences are saved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.signal import medfilt, savgol_filter

from rescore_continuous import rescore_result


EPS = 1e-9


def apply_continuous_scoring(
    result: dict[str, Any], continuous_config_path: Path
) -> dict[str, Any]:
    """Attach the public continuous score while preserving frozen decisions.

    The evaluator intentionally computes the legacy 0-100 dimensions and the
    legacy ``structural_ok``/``physics_pass`` labels first. Only after those
    decisions are frozen do we translate the already-computed measurements to
    the public continuous-0-1-v1 score contract. This ordering makes the
    migration score-only: it cannot change the historical pass labels.
    """
    continuous_config_path = continuous_config_path.resolve()
    with continuous_config_path.open("r", encoding="utf-8") as handle:
        continuous_config = yaml.safe_load(handle)
    return rescore_result(
        result,
        continuous_config,
        sha256_file(continuous_config_path),
    )


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    if isinstance(value, tuple):
        return [_finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def robust_mad(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    median = np.median(values)
    return float(1.4826 * np.median(np.abs(values - median)))


def longest_true_run(mask: np.ndarray) -> tuple[int, int | None, int | None]:
    """Return length and inclusive endpoints of the longest True run."""
    best_length = 0
    best_start = best_end = None
    start = None
    for index, value in enumerate(np.asarray(mask, dtype=bool).tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            length = index - start
            if length > best_length:
                best_length = length
                best_start, best_end = start, index - 1
            start = None
    return best_length, best_start, best_end


def q_score(value: float, spec: dict[str, Any]) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    good = float(spec["good"])
    bad = float(spec["bad"])
    higher = bool(spec.get("higher_is_better", False))
    if higher:
        if value >= good:
            return 100.0
        if value <= bad:
            return 0.0
        return 100.0 * (value - bad) / (good - bad)
    if value <= good:
        return 100.0
    if value >= bad:
        return 0.0
    return 100.0 * (bad - value) / (bad - good)


def weighted_geometric_mean(scores: dict[str, float], weights: dict[str, float]) -> float:
    terms = []
    total = 0.0
    for key, weight in weights.items():
        score = max(0.0, min(100.0, float(scores.get(key, 0.0))))
        terms.append(float(weight) * math.log(max(score / 100.0, 1e-6)))
        total += float(weight)
    return float(100.0 * math.exp(sum(terms) / max(total, EPS)))


def infer_source_id(video: Path, sources: dict[str, Any]) -> str:
    stem = video.stem
    matches = [source for source in sources if stem.startswith(source + "_seed")]
    if len(matches) != 1:
        raise ValueError(f"cannot infer one configured source from {video.name}: {matches}")
    return matches[0]


def read_video(path: Path) -> tuple[list[np.ndarray], float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames or fps <= 0:
        raise RuntimeError(f"video decode failed: {path}")
    height, width = frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise RuntimeError("variable frame size is unsupported")
    return frames, fps, (width, height)


def circle_edge_support(gray: np.ndarray, circle: tuple[float, float, float]) -> float:
    cx, cy, radius = circle
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    values = []
    for theta in np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False):
        x = int(round(cx + radius * math.cos(theta)))
        y = int(round(cy + radius * math.sin(theta)))
        if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]:
            values.append(float(magnitude[y, x]))
    return float(np.median(values)) if values else 0.0


def local_hough_circle(
    frame: np.ndarray,
    prediction: np.ndarray,
    radius: float,
) -> tuple[np.ndarray | None, float | None, float]:
    height, width = frame.shape[:2]
    half = int(max(85, round(2.0 * radius)))
    x0 = max(0, int(round(prediction[0])) - half)
    x1 = min(width, int(round(prediction[0])) + half + 1)
    y0 = max(0, int(round(prediction[1])) - half)
    y1 = min(height, int(round(prediction[1])) + half + 1)
    crop = frame[y0:y1, x0:x1]
    if min(crop.shape[:2]) < 40:
        return None, None, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 1.4)
    found = None
    used_threshold = None
    for threshold in (27, 23, 20, 17):
        result = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.15,
            minDist=max(20.0, 0.65 * radius),
            param1=105,
            param2=threshold,
            minRadius=max(10, int(round(0.66 * radius))),
            maxRadius=max(15, int(round(1.34 * radius))),
        )
        if result is not None:
            found = result[0]
            used_threshold = threshold
            break
    if found is None:
        return None, None, 0.0
    candidates = []
    for local_x, local_y, candidate_radius in found:
        center = np.array([float(local_x + x0), float(local_y + y0)])
        distance = float(np.linalg.norm(center - prediction) / max(radius, EPS))
        radius_error = abs(float(candidate_radius) / max(radius, EPS) - 1.0)
        support = circle_edge_support(gray, (float(local_x), float(local_y), float(candidate_radius)))
        cost = 1.25 * distance + 0.65 * radius_error - 0.003 * min(support, 150.0)
        candidates.append((cost, center, float(candidate_radius), support))
    candidates.sort(key=lambda item: item[0])
    cost, center, candidate_radius, support = candidates[0]
    confidence = float(np.clip(1.0 - max(cost, 0.0) / 1.5, 0.0, 1.0))
    confidence *= 0.75 + 0.25 * float(used_threshold >= 23)
    confidence *= float(np.clip(support / 35.0, 0.45, 1.0))
    return center, candidate_radius, confidence


def ball_optical_flow(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> tuple[np.ndarray | None, float, float, int]:
    mask = np.zeros_like(previous_gray)
    cv2.circle(mask, tuple(np.round(center).astype(int)), max(8, int(round(0.78 * radius))), 255, -1)
    points = cv2.goodFeaturesToTrack(
        previous_gray,
        mask=mask,
        maxCorners=120,
        qualityLevel=0.008,
        minDistance=3,
        blockSize=5,
    )
    if points is None or len(points) < 6:
        return None, 0.0, 0.0, 0
    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        gray,
        points,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        previous_gray,
        forward,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if forward is None or backward is None:
        return None, 0.0, 0.0, 0
    valid = (status[:, 0] > 0) & (back_status[:, 0] > 0)
    fb_error = np.linalg.norm(points[:, 0] - backward[:, 0], axis=1)
    valid &= fb_error < 1.8
    source = points[valid, 0]
    target = forward[valid, 0]
    if len(source) < 6:
        return None, 0.0, 0.0, int(len(source))
    transform, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.5,
        maxIters=1000,
        confidence=0.99,
        refineIters=10,
    )
    if transform is None:
        displacement = np.median(target - source, axis=0)
        return center + displacement, 0.0, 0.25, int(len(source))
    homogeneous = np.array([center[0], center[1], 1.0])
    predicted = transform @ homogeneous
    rotation = math.atan2(float(transform[1, 0]), float(transform[0, 0]))
    inlier_fraction = float(np.mean(inliers[:, 0] > 0)) if inliers is not None else 0.0
    confidence = float(np.clip((len(source) / 25.0) * inlier_fraction, 0.0, 1.0))
    return predicted, rotation, confidence, int(len(source))


def background_shift(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, float]:
    scale = 0.5
    previous = cv2.resize(previous_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    current = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = np.full_like(previous, 255)
    exclusion_center = tuple(np.round(center * scale).astype(int))
    cv2.circle(mask, exclusion_center, max(15, int(round(2.0 * radius * scale))), 0, -1)
    points = cv2.goodFeaturesToTrack(
        previous,
        mask=mask,
        maxCorners=300,
        qualityLevel=0.015,
        minDistance=8,
        blockSize=7,
    )
    if points is None or len(points) < 20:
        return np.zeros(2), 0.0
    target, status, _ = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
    )
    valid = status[:, 0] > 0
    displacement = target[valid, 0] - points[valid, 0]
    if len(displacement) < 15:
        return np.zeros(2), 0.0
    median = np.median(displacement, axis=0)
    distances = np.linalg.norm(displacement - median, axis=1)
    cutoff = max(0.35, 3.5 * robust_mad(distances))
    inliers = distances <= cutoff
    if int(np.sum(inliers)) < 12:
        return np.zeros(2), 0.0
    shift = np.median(displacement[inliers], axis=0) / scale
    confidence = float(np.clip(np.sum(inliers) / 80.0, 0.0, 1.0))
    return shift.astype(float), confidence


def track_ball(
    frames: list[np.ndarray],
    initial_center: tuple[float, float],
    initial_radius: float,
) -> dict[str, np.ndarray]:
    count = len(frames)
    centers = np.full((count, 2), np.nan, dtype=float)
    stabilized = np.full((count, 2), np.nan, dtype=float)
    radii = np.full(count, np.nan, dtype=float)
    confidences = np.zeros(count, dtype=float)
    hough_confidences = np.zeros(count, dtype=float)
    flow_confidences = np.zeros(count, dtype=float)
    flow_rotations = np.zeros(count, dtype=float)
    camera_cumulative = np.zeros((count, 2), dtype=float)
    camera_confidences = np.zeros(count, dtype=float)

    center = np.asarray(initial_center, dtype=float)
    radius = float(initial_radius)
    refined, refined_radius, refined_confidence = local_hough_circle(frames[0], center, radius)
    if refined is not None and np.linalg.norm(refined - center) < 0.55 * radius:
        center = 0.35 * center + 0.65 * refined
        radius = 0.65 * radius + 0.35 * float(refined_radius)
    centers[0] = center
    stabilized[0] = center
    radii[0] = radius
    confidences[0] = max(0.8, refined_confidence)
    velocity = np.zeros(2, dtype=float)
    previous_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for index in range(1, count):
        gray = cv2.cvtColor(frames[index], cv2.COLOR_BGR2GRAY)
        camera_step, camera_confidence = background_shift(previous_gray, gray, center, radius)
        camera_cumulative[index] = camera_cumulative[index - 1] + camera_step
        camera_confidences[index] = camera_confidence

        flow_center, flow_rotation, flow_confidence, _ = ball_optical_flow(
            previous_gray, gray, center, radius
        )
        motion_prediction = center + velocity + camera_step
        if flow_center is not None and np.linalg.norm(flow_center - motion_prediction) < 1.1 * radius:
            prediction = 0.65 * flow_center + 0.35 * motion_prediction
        else:
            prediction = motion_prediction
        hough_center, hough_radius, hough_confidence = local_hough_circle(
            frames[index], prediction, radius
        )

        chosen = None
        confidence = 0.0
        if hough_center is not None and flow_center is not None:
            agreement = float(np.linalg.norm(hough_center - flow_center) / max(radius, EPS))
            if agreement < 0.70:
                hough_weight = 0.55 + 0.25 * hough_confidence
                chosen = hough_weight * hough_center + (1.0 - hough_weight) * flow_center
                confidence = (0.55 * hough_confidence + 0.45 * flow_confidence) * math.exp(-agreement)
            else:
                hough_distance = np.linalg.norm(hough_center - motion_prediction)
                flow_distance = np.linalg.norm(flow_center - motion_prediction)
                if hough_distance <= flow_distance and hough_confidence >= 0.25:
                    chosen, confidence = hough_center, 0.65 * hough_confidence
                else:
                    chosen, confidence = flow_center, 0.60 * flow_confidence
        elif hough_center is not None:
            chosen, confidence = hough_center, 0.70 * hough_confidence
        elif flow_center is not None:
            chosen, confidence = flow_center, 0.65 * flow_confidence
        else:
            chosen, confidence = motion_prediction, 0.05

        step = chosen - center - camera_step
        maximum_step = max(8.0, 0.75 * radius)
        step_norm = float(np.linalg.norm(step))
        if step_norm > maximum_step:
            step *= maximum_step / step_norm
            chosen = center + camera_step + step
            confidence *= 0.25
        previous_stabilized = center - camera_cumulative[index - 1]
        current_stabilized = chosen - camera_cumulative[index]
        measured_velocity = current_stabilized - previous_stabilized
        velocity = 0.72 * velocity + 0.28 * measured_velocity
        center = chosen
        if hough_radius is not None and 0.70 * radius <= hough_radius <= 1.30 * radius:
            radius = 0.94 * radius + 0.06 * float(hough_radius)

        centers[index] = center
        stabilized[index] = current_stabilized
        radii[index] = radius
        confidences[index] = float(np.clip(confidence, 0.0, 1.0))
        hough_confidences[index] = hough_confidence
        flow_confidences[index] = flow_confidence
        flow_rotations[index] = flow_rotation if math.isfinite(flow_rotation) else 0.0
        previous_gray = gray

    # A light robust smoothing removes sub-pixel Hough jitter without changing
    # the measured trajectory on the scale of one ball radius.
    for axis in range(2):
        stabilized[:, axis] = medfilt(stabilized[:, axis], kernel_size=3)
        if count >= 9:
            stabilized[:, axis] = savgol_filter(stabilized[:, axis], 9, 2, mode="interp")
    return {
        "centers": centers,
        "stabilized_centers": stabilized,
        "radii": radii,
        "confidence": confidences,
        "hough_confidence": hough_confidences,
        "flow_confidence": flow_confidences,
        "flow_rotation": flow_rotations,
        "camera_cumulative": camera_cumulative,
        "camera_confidence": camera_confidences,
    }


def line_angle_distance(a: float, b: float) -> float:
    return abs(((a - b + 0.5 * np.pi) % np.pi) - 0.5 * np.pi)


def pca_line(points_xy: np.ndarray) -> tuple[float, float, float]:
    if len(points_xy) < 12:
        return float("nan"), 0.0, float("inf")
    mean = np.mean(points_xy, axis=0)
    covariance = np.cov((points_xy - mean).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major = eigenvectors[:, order[0]]
    major_value = max(float(eigenvalues[order[0]]), EPS)
    minor_value = max(float(eigenvalues[order[1]]), EPS)
    angle = math.atan2(float(major[1]), float(major[0])) % np.pi
    elongation = math.sqrt(major_value / minor_value)
    normal = np.array([-major[1], major[0]])
    distance = abs(float(np.dot(mean, normal)))
    return angle, elongation, distance


def estimate_pattern_orientation(
    frame: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> tuple[float, float, list[tuple[float, float, str]]]:
    cx, cy = map(float, center)
    half = int(max(20, round(0.92 * radius)))
    x0 = max(0, int(round(cx)) - half)
    x1 = min(frame.shape[1], int(round(cx)) + half + 1)
    y0 = max(0, int(round(cy)) - half)
    y1 = min(frame.shape[0], int(round(cy)) + half + 1)
    crop = frame[y0:y1, x0:x1]
    if min(crop.shape[:2]) < 25:
        return float("nan"), 0.0, []
    yy, xx = np.indices(crop.shape[:2])
    local_x = xx.astype(float) + x0 - cx
    local_y = yy.astype(float) + y0 - cy
    inner = local_x**2 + local_y**2 <= (0.76 * radius) ** 2
    points = np.column_stack([local_x[inner], local_y[inner]])
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[inner]
    candidates: list[tuple[float, float, str]] = []
    if len(pixels) >= 200:
        stride = max(1, len(pixels) // 3500)
        sample = pixels[::stride]
        cv2.setRNGSeed(1701)
        _, labels, centers_lab = cv2.kmeans(
            sample,
            4,
            None,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5),
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        # Assign every pixel to the deterministic colour centres.
        distances = np.linalg.norm(pixels[:, None, :] - centers_lab[None, :, :], axis=2)
        full_labels = np.argmin(distances, axis=1)
        dominant = int(np.argmax(np.bincount(full_labels, minlength=4)))
        dominant_colour = centers_lab[dominant]
        for cluster in range(4):
            selected = full_labels == cluster
            fraction = float(np.mean(selected))
            if not (0.035 <= fraction <= 0.48) or int(np.sum(selected)) < 50:
                continue
            angle, elongation, line_distance = pca_line(points[selected])
            if not math.isfinite(angle):
                continue
            projection_span = float(np.ptp(points[selected] @ np.array([math.cos(angle), math.sin(angle)])))
            colour_contrast = float(np.linalg.norm(centers_lab[cluster] - dominant_colour))
            centrality = math.exp(-((line_distance / max(0.28 * radius, EPS)) ** 2))
            score = (
                min(1.0, max(0.0, (elongation - 1.15) / 3.0))
                * min(1.0, projection_span / max(1.15 * radius, EPS))
                * min(1.0, colour_contrast / 32.0)
                * centrality
            )
            if score >= 0.08:
                candidates.append((angle, score, "lab-cluster"))

        median_colour = np.median(pixels, axis=0)
        colour_distance = np.linalg.norm(pixels - median_colour, axis=1)
        cutoff = np.percentile(colour_distance, 79)
        selected = colour_distance >= cutoff
        angle, elongation, line_distance = pca_line(points[selected])
        if math.isfinite(angle):
            score = min(1.0, max(0.0, (elongation - 1.0) / 3.5))
            score *= math.exp(-((line_distance / max(0.30 * radius, EPS)) ** 2))
            if score >= 0.08:
                candidates.append((angle, 0.75 * score, "colour-tail"))

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0.8), 35, 100)
    edges[~inner] = 0
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(10, int(round(0.25 * radius))),
        minLineLength=max(10, int(round(0.42 * radius))),
        maxLineGap=max(4, int(round(0.22 * radius))),
    )
    if lines is not None:
        for line in np.asarray(lines).reshape(-1, 4):
            x_a, y_a, x_b, y_b = map(float, line)
            point_a = np.array([x_a + x0 - cx, y_a + y0 - cy])
            point_b = np.array([x_b + x0 - cx, y_b + y0 - cy])
            vector = point_b - point_a
            length = float(np.linalg.norm(vector))
            if length < EPS:
                continue
            normal = np.array([-vector[1], vector[0]]) / length
            distance = abs(float(np.dot(point_a, normal)))
            angle = math.atan2(float(vector[1]), float(vector[0])) % np.pi
            score = min(1.0, length / max(1.2 * radius, EPS))
            score *= math.exp(-((distance / max(0.30 * radius, EPS)) ** 2))
            if score >= 0.10:
                candidates.append((angle, 0.70 * score, "edge-line"))

    if not candidates:
        return float("nan"), 0.0, []
    candidates.sort(key=lambda item: item[1], reverse=True)
    anchor = candidates[0][0]
    compatible = [item for item in candidates if line_angle_distance(item[0], anchor) <= math.radians(28)]
    weights = np.asarray([item[1] for item in compatible], dtype=float)
    doubled = np.asarray([2.0 * item[0] for item in compatible])
    sine = float(np.sum(weights * np.sin(doubled)))
    cosine = float(np.sum(weights * np.cos(doubled)))
    angle = (0.5 * math.atan2(sine, cosine)) % np.pi
    resultant = math.hypot(sine, cosine) / max(float(np.sum(weights)), EPS)
    confidence = float(np.clip(candidates[0][1] * (0.55 + 0.45 * resultant), 0.0, 1.0))
    if len(compatible) >= 2:
        confidence = min(1.0, confidence + 0.08)
    return angle, confidence, candidates[:10]


def estimate_offset_marker_candidates(
    frame: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> list[dict[str, Any]]:
    """Find compact, off-centre colour blobs rigidly printed on the ball.

    The task's accepted first frames deliberately contain a red, brown, or
    white offset dot in addition to the pi-periodic band.  Tracking this dot
    resolves whole-turn ambiguity without imposing the rolling equation.
    """
    cx, cy = map(float, center)
    half = int(max(20, round(0.90 * radius)))
    x0 = max(0, int(round(cx)) - half)
    x1 = min(frame.shape[1], int(round(cx)) + half + 1)
    y0 = max(0, int(round(cy)) - half)
    y1 = min(frame.shape[0], int(round(cy)) + half + 1)
    crop = frame[y0:y1, x0:x1]
    if min(crop.shape[:2]) < 25:
        return []
    yy, xx = np.indices(crop.shape[:2])
    local_x = xx.astype(float) + x0 - cx
    local_y = yy.astype(float) + y0 - cy
    inner = local_x**2 + local_y**2 <= (0.80 * radius) ** 2
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[inner]
    if len(pixels) < 200:
        return []
    stride = max(1, len(pixels) // 4000)
    cv2.setRNGSeed(2701)
    _, _, centres_lab = cv2.kmeans(
        pixels[::stride],
        5,
        None,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.4),
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    distance = np.linalg.norm(lab[:, :, None, :] - centres_lab[None, None, :, :], axis=3)
    labels = np.argmin(distance, axis=2)
    dominant_labels = labels[inner]
    dominant = int(np.argmax(np.bincount(dominant_labels, minlength=5)))
    dominant_colour = centres_lab[dominant]
    disk_area = math.pi * radius * radius
    candidates: list[dict[str, Any]] = []
    kernel = np.ones((3, 3), np.uint8)
    for cluster in range(5):
        if cluster == dominant:
            continue
        mask = ((labels == cluster) & inner).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        component_count, component_labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        colour_contrast = float(np.linalg.norm(centres_lab[cluster] - dominant_colour))
        for component in range(1, component_count):
            area = float(stats[component, cv2.CC_STAT_AREA])
            area_fraction = area / max(disk_area, EPS)
            if not (0.0035 <= area_fraction <= 0.105):
                continue
            px, py = map(float, centroids[component])
            dx = px + x0 - cx
            dy = py + y0 - cy
            radial = math.hypot(dx, dy)
            radial_fraction = radial / max(radius, EPS)
            if not (0.20 <= radial_fraction <= 0.78):
                continue
            component_y, component_x = np.nonzero(component_labels == component)
            coords = np.column_stack([component_x.astype(float) - px, component_y.astype(float) - py])
            if len(coords) < 12:
                continue
            covariance = np.cov(coords.T)
            eigenvalues = np.linalg.eigvalsh(covariance)
            compactness = float(max(EPS, eigenvalues[0]) / max(eigenvalues[-1], EPS))
            contour_mask = (component_labels == component).astype(np.uint8) * 255
            contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = max((cv2.arcLength(contour, True) for contour in contours), default=0.0)
            circularity = float(np.clip(4.0 * np.pi * area / max(perimeter * perimeter, EPS), 0.0, 1.0))
            equivalent_radius = math.sqrt(area / np.pi)
            size_score = math.exp(-((equivalent_radius / max(0.15 * radius, EPS) - 1.0) / 0.75) ** 2)
            radial_score = math.exp(-((radial_fraction - 0.52) / 0.30) ** 2)
            score = (
                min(1.0, colour_contrast / 34.0)
                * (0.45 * min(1.0, compactness / 0.45) + 0.55 * circularity)
                * size_score
                * radial_score
            )
            if score < 0.075:
                continue
            candidates.append(
                {
                    "angle": math.atan2(dy, dx),
                    "radial": radial_fraction,
                    "score": float(np.clip(score, 0.0, 1.0)),
                    "center": [px + x0, py + y0],
                    "area_fraction": area_fraction,
                    "colour_lab": [float(value) for value in centres_lab[cluster]],
                }
            )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    # Merge near-duplicate colour clusters/components.
    unique: list[dict[str, Any]] = []
    for candidate in candidates:
        if all(
            abs(((candidate["angle"] - prior["angle"] + np.pi) % (2 * np.pi)) - np.pi) > 0.12
            or abs(candidate["radial"] - prior["radial"]) > 0.12
            for prior in unique
        ):
            unique.append(candidate)
    return unique[:8]


def track_offset_marker(
    candidates: list[list[dict[str, Any]]],
    line_observed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(candidates)
    angle = np.full(count, np.nan, dtype=float)
    confidence = np.zeros(count, dtype=float)
    used = np.zeros(count, dtype=bool)
    xy = np.full((count, 2), np.nan, dtype=float)
    first = next((index for index, items in enumerate(candidates) if items), None)
    if first is None:
        return angle, confidence, used, xy
    selected = candidates[first][0]
    angle[first] = float(selected["angle"])
    confidence[first] = float(selected["score"])
    used[first] = True
    xy[first] = selected["center"]
    radial_previous = float(selected["radial"])
    colour_reference = np.asarray(selected["colour_lab"], dtype=float)
    area_reference = float(selected["area_fraction"])
    relative_offset = (
        ((angle[first] - line_observed[first] + 0.5 * np.pi) % np.pi) - 0.5 * np.pi
        if math.isfinite(line_observed[first])
        else 0.0
    )
    recent_steps: list[float] = []
    for index in range(first + 1, count):
        predicted_step = float(np.median(recent_steps[-5:])) if recent_steps else 0.0
        prediction = angle[index - 1] + np.clip(predicted_step, -0.55, 0.55)
        best = None
        for item in candidates[index]:
            base = float(item["angle"])
            branch = base + round((prediction - base) / (2 * np.pi)) * (2 * np.pi)
            temporal = abs(branch - prediction)
            radial_cost = abs(float(item["radial"]) - radial_previous)
            colour_cost = float(
                np.linalg.norm(np.asarray(item["colour_lab"], dtype=float) - colour_reference)
            )
            area_cost = abs(float(item["area_fraction"]) - area_reference) / max(area_reference, 0.01)
            rigid_cost = 0.0
            if math.isfinite(line_observed[index]):
                candidate_offset = ((branch - line_observed[index] + 0.5 * np.pi) % np.pi) - 0.5 * np.pi
                rigid_cost = line_angle_distance(candidate_offset, relative_offset)
            cost = (
                temporal / 0.35
                + radial_cost / 0.18
                + colour_cost / 28.0
                + 0.25 * min(area_cost, 3.0)
                + rigid_cost / 0.75
                - 1.8 * float(item["score"])
            )
            if best is None or cost < best[0]:
                best = (cost, branch, item)
        if best is not None and abs(best[1] - prediction) <= 0.80:
            _, branch, item = best
            angle[index] = branch
            confidence[index] = float(item["score"])
            used[index] = True
            xy[index] = item["center"]
            step = branch - angle[index - 1]
            recent_steps.append(float(step))
            radial_previous = 0.8 * radial_previous + 0.2 * float(item["radial"])
            colour_reference = 0.96 * colour_reference + 0.04 * np.asarray(
                item["colour_lab"], dtype=float
            )
            area_reference = 0.92 * area_reference + 0.08 * float(item["area_fraction"])
            if math.isfinite(line_observed[index]):
                new_offset = ((branch - line_observed[index] + 0.5 * np.pi) % np.pi) - 0.5 * np.pi
                relative_offset = 0.9 * relative_offset + 0.1 * new_offset
        else:
            angle[index] = prediction
    for index in range(first - 1, -1, -1):
        angle[index] = angle[index + 1]
    reliable = np.flatnonzero(used)
    if len(reliable) >= 2:
        repaired = np.interp(np.arange(count), reliable, angle[reliable])
        if count >= 9:
            repaired = savgol_filter(medfilt(repaired, 3), 9, 2, mode="interp")
        angle = repaired
    return angle, confidence, used, xy


def unwrap_line_with_offset_marker(
    observed: np.ndarray,
    line_confidence: np.ndarray,
    flow_rotation: np.ndarray,
    dot_angle: np.ndarray,
    dot_used: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pair = np.isfinite(observed) & np.isfinite(dot_angle) & dot_used
    if int(np.sum(pair)) < 4:
        return unwrap_pattern_angles(observed, line_confidence, flow_rotation)
    differences = dot_angle[pair] - observed[pair]
    sine = np.median(np.sin(2.0 * differences))
    cosine = np.median(np.cos(2.0 * differences))
    offset = 0.5 * math.atan2(float(sine), float(cosine))
    count = len(observed)
    output = np.full(count, np.nan, dtype=float)
    used = np.zeros(count, dtype=bool)
    first = int(np.flatnonzero(pair)[0])
    output[first] = observed[first] + round((dot_angle[first] - offset - observed[first]) / np.pi) * np.pi
    used[first] = True
    for index in range(first + 1, count):
        prediction = output[index - 1] + float(np.clip(flow_rotation[index], -0.50, 0.50))
        target = dot_angle[index] - offset if dot_used[index] else prediction
        if math.isfinite(observed[index]) and line_confidence[index] >= 0.10:
            candidate = observed[index] + round((target - observed[index]) / np.pi) * np.pi
            if abs(candidate - target) <= 0.90:
                output[index] = candidate
                used[index] = True
            else:
                output[index] = target
        else:
            output[index] = target
    for index in range(first - 1, -1, -1):
        target = dot_angle[index] - offset if dot_used[index] else output[index + 1]
        if math.isfinite(observed[index]):
            output[index] = observed[index] + round((target - observed[index]) / np.pi) * np.pi
        else:
            output[index] = target
    reliable = np.flatnonzero(used)
    if len(reliable) >= 2:
        output = np.interp(np.arange(count), reliable, output[reliable])
    if count >= 9:
        output = savgol_filter(medfilt(output, 3), 9, 2, mode="interp")
    return output, used


def unwrap_pattern_angles(
    observed: np.ndarray,
    confidence: np.ndarray,
    flow_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(observed)
    unwrapped = np.full(count, np.nan, dtype=float)
    used = np.zeros(count, dtype=bool)
    valid_indices = np.flatnonzero(np.isfinite(observed) & (confidence >= 0.10))
    if len(valid_indices) == 0:
        return unwrapped, used
    first = int(valid_indices[0])
    unwrapped[first] = observed[first]
    used[first] = True
    for index in range(first + 1, count):
        # The visible band is a pi-periodic line.  Ball-interior KLT supplies
        # only the branch prediction; accepted image observations are never
        # blended toward flow, avoiding a systematic angular-speed bias.
        flow_step = float(np.clip(flow_rotation[index], -0.50, 0.50))
        prediction = unwrapped[index - 1] + flow_step
        if math.isfinite(observed[index]) and confidence[index] >= 0.10:
            base = observed[index]
            k = round((prediction - base) / np.pi)
            candidate = base + k * np.pi
            discrepancy = abs(candidate - prediction)
            limit = 0.78 + 0.25 * (1.0 - confidence[index])
            if discrepancy <= limit:
                unwrapped[index] = candidate
                used[index] = True
            else:
                unwrapped[index] = prediction
        else:
            unwrapped[index] = prediction
    for index in range(first - 1, -1, -1):
        unwrapped[index] = unwrapped[index + 1] - float(np.clip(flow_rotation[index + 1], -0.65, 0.65))

    # Reject isolated orientation glitches using a Hampel-style derivative test.
    increments = np.diff(unwrapped)
    if len(increments) >= 5:
        median_increment = medfilt(increments, kernel_size=5)
        residual = increments - median_increment
        scale = max(0.04, robust_mad(residual))
        bad = np.abs(residual) > max(0.45, 4.5 * scale)
        for index in np.flatnonzero(bad) + 1:
            used[index] = False
    reliable = np.flatnonzero(used & np.isfinite(unwrapped))
    if len(reliable) >= 2:
        repaired = np.interp(np.arange(count), reliable, unwrapped[reliable])
    else:
        repaired = unwrapped.copy()
    if count >= 9 and np.all(np.isfinite(repaired)):
        repaired = savgol_filter(medfilt(repaired, kernel_size=3), 9, 2, mode="interp")
    return repaired, used


def robust_linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) < EPS:
        return float("nan"), float("nan"), np.full_like(y, np.nan)
    initial = np.polyfit(x, y, 1)
    scale = max(robust_mad(y - np.polyval(initial, x)), 1e-3)

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (parameters[0] * x + parameters[1] - y) / scale

    fitted = least_squares(residual, initial, loss="soft_l1", f_scale=1.0).x
    prediction = fitted[0] * x + fitted[1]
    return float(fitted[0]), float(fitted[1]), prediction


def determine_motion_interval(
    centers: np.ndarray,
    radius: float,
    confidence: np.ndarray,
    threshold_radius_per_frame: float,
) -> np.ndarray:
    displacement = np.linalg.norm(np.diff(centers, axis=0, prepend=centers[[0]]), axis=1)
    speed = medfilt(displacement, kernel_size=5)
    moving = speed > threshold_radius_per_frame * radius
    sustained = np.convolve(moving.astype(int), np.ones(5, dtype=int), mode="same") >= 3
    candidate = sustained & (confidence >= 0.08)
    indices = np.flatnonzero(candidate)
    if len(indices) == 0:
        return np.zeros(len(centers), dtype=bool)
    start = max(0, int(indices[0]) - 2)
    end = min(len(centers) - 1, int(indices[-1]) + 2)
    interval = np.zeros(len(centers), dtype=bool)
    interval[start : end + 1] = True
    return interval


def path_geometry(centers: np.ndarray, motion: np.ndarray, radius: float) -> dict[str, Any]:
    indices = np.flatnonzero(motion)
    if len(indices) < 4:
        return {
            "cumulative_distance": np.zeros(len(centers)),
            "path_normal_p95_r": float("nan"),
            "path_direction_changes": float("nan"),
        }
    differences = np.diff(centers, axis=0, prepend=centers[[0]])
    cumulative = np.cumsum(np.linalg.norm(differences, axis=1))
    # A two-segment robust diagnostic captures the incline and level runout.
    points = centers[indices]
    best_residual = np.full(len(points), np.inf)
    best_split = None
    minimum_segment = max(5, len(points) // 8)
    for split in range(minimum_segment, len(points) - minimum_segment + 1):
        residual_parts = []
        for part in (points[:split], points[split:]):
            mean = np.mean(part, axis=0)
            _, _, vt = np.linalg.svd(part - mean, full_matrices=False)
            normal = vt[-1]
            residual_parts.append(np.abs((part - mean) @ normal))
        residual = np.concatenate(residual_parts)
        loss = float(np.median(residual) + 0.20 * np.percentile(residual, 90))
        if best_split is None or loss < best_split[0]:
            best_split = (loss, split)
            best_residual = residual
    tangent = differences[indices]
    angles = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    direction_changes = float(np.percentile(np.abs(np.diff(angles)), 95)) if len(angles) > 2 else 0.0
    return {
        "cumulative_distance": cumulative,
        "path_normal_p95_r": float(np.percentile(best_residual, 95) / max(radius, EPS)),
        "path_direction_changes": direction_changes,
    }


def compute_metrics(
    track: dict[str, np.ndarray],
    pattern_angles: np.ndarray,
    pattern_confidence: np.ndarray,
    pattern_used: np.ndarray,
    offset_confidence: np.ndarray,
    offset_used: np.ndarray,
    fps: float,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float], np.ndarray, dict[str, np.ndarray]]:
    centers = track["stabilized_centers"]
    radii = track["radii"]
    radius = float(np.median(radii[np.isfinite(radii)]))
    tracking = config["tracking"]
    motion = determine_motion_interval(
        centers,
        radius,
        track["confidence"],
        float(tracking["motion_speed_radius_per_frame"]),
    )
    indices = np.flatnonzero(motion)
    geometry = path_geometry(centers, motion, radius)
    cumulative = geometry.pop("cumulative_distance")

    time = np.arange(len(centers), dtype=float) / fps
    x = centers[:, 0]
    y = centers[:, 1]
    window = min(15, len(centers) if len(centers) % 2 == 1 else len(centers) - 1)
    window = max(5, window)
    if window % 2 == 0:
        window -= 1
    x_smooth = savgol_filter(x, window, 2, mode="interp")
    y_smooth = savgol_filter(y, window, 2, mode="interp")
    velocity_x = savgol_filter(x, window, 2, deriv=1, delta=1.0 / fps, mode="interp")
    velocity_y = savgol_filter(y, window, 2, deriv=1, delta=1.0 / fps, mode="interp")
    speed = np.hypot(velocity_x, velocity_y)
    omega_signed = savgol_filter(
        pattern_angles, window, 2, deriv=1, delta=1.0 / fps, mode="interp"
    )
    if len(indices):
        direction = float(np.sign(np.median(omega_signed[indices])))
        if direction == 0:
            direction = 1.0
    else:
        direction = 1.0
    omega = direction * omega_signed
    rolling_speed = radius * omega
    contact_velocity = speed - rolling_speed

    valid = motion & np.isfinite(speed) & np.isfinite(rolling_speed)
    valid &= track["confidence"] >= 0.08
    valid &= pattern_confidence >= 0.07
    valid &= omega > max(0.02, 0.015 * fps)
    valid &= speed > 0.02 * radius * fps
    valid_indices = np.flatnonzero(valid)
    ratios = speed[valid] / np.maximum(rolling_speed[valid], EPS)
    ratio_median = float(np.median(ratios)) if len(ratios) else float("nan")
    ratio_error = abs(ratio_median - 1.0) if math.isfinite(ratio_median) else float("nan")

    normalized_contact = np.abs(contact_velocity[valid]) / np.maximum(speed[valid], EPS)
    contact_nmae = float(np.median(normalized_contact)) if len(normalized_contact) else float("nan")
    contact_p95 = float(np.percentile(normalized_contact, 95)) if len(normalized_contact) else float("nan")

    angular_span = (
        abs(float(pattern_angles[indices[-1]] - pattern_angles[indices[0]])) if len(indices) else 0.0
    )
    if len(indices) >= 4:
        slope, intercept, theta_prediction = robust_linear_fit(
            cumulative[indices], pattern_angles[indices]
        )
        angular_residual = pattern_angles[indices] - theta_prediction
        angular_nrmse = float(
            np.sqrt(np.mean(angular_residual**2)) / max(angular_span, 0.25)
        )
        distance_slope, _, _ = robust_linear_fit(pattern_angles[indices], cumulative[indices])
        integrated_ratio = abs(distance_slope) / max(radius, EPS) if math.isfinite(distance_slope) else float("nan")
    else:
        slope = intercept = integrated_ratio = angular_nrmse = float("nan")

    increments = direction * np.diff(pattern_angles)
    motion_pairs = motion[1:] & motion[:-1]
    reversal_fraction = (
        float(np.mean(increments[motion_pairs] < -0.025)) if np.any(motion_pairs) else float("nan")
    )
    track_coverage = float(np.mean(track["confidence"] >= 0.08))
    pattern_coverage = (
        float(np.mean(pattern_confidence[motion] >= 0.10)) if np.any(motion) else 0.0
    )
    pattern_temporal_used_fraction = (
        float(np.mean(pattern_used[motion])) if np.any(motion) else 0.0
    )
    hough_dropout_threshold = float(tracking["hough_dropout_confidence"])
    if len(indices):
        motion_slice = slice(int(indices[0]), int(indices[-1]) + 1)
        hough_dropout_length, hough_dropout_local_start, hough_dropout_local_end = longest_true_run(
            track["hough_confidence"][motion_slice] < hough_dropout_threshold
        )
        hough_dropout_start = (
            int(indices[0]) + int(hough_dropout_local_start)
            if hough_dropout_local_start is not None
            else None
        )
        hough_dropout_end = (
            int(indices[0]) + int(hough_dropout_local_end)
            if hough_dropout_local_end is not None
            else None
        )
    else:
        hough_dropout_length = 0
        hough_dropout_start = hough_dropout_end = None
    offset_marker_coverage = float(np.mean(offset_used[motion])) if np.any(motion) else 0.0
    radius_cv = robust_mad(radii) / max(float(np.median(radii)), EPS)
    camera_displacement = np.linalg.norm(track["camera_cumulative"], axis=1)
    camera_drift_width = float(np.percentile(camera_displacement, 95) / 1344.0)

    integrated_error = abs(integrated_ratio - 1.0) if math.isfinite(integrated_ratio) else float("nan")
    metrics = {
        "M1_rolling_ratio_signed": integrated_ratio - 1.0 if math.isfinite(integrated_ratio) else None,
        "M1_rolling_ratio_abs": integrated_error,
        "M1_frame_ratio_signed": ratio_median - 1.0 if math.isfinite(ratio_median) else None,
        "M1_frame_ratio_abs": ratio_error,
        "M2_contact_velocity_nmae": contact_nmae,
        "M2_contact_velocity_p95": contact_p95,
        "track_coverage": track_coverage,
        "pattern_coverage": pattern_coverage,
        "pattern_temporal_used_fraction": pattern_temporal_used_fraction,
        "hough_track_coverage": float(
            np.mean(track["hough_confidence"][motion] >= hough_dropout_threshold)
        ) if np.any(motion) else 0.0,
        "max_hough_dropout_run_frames": int(hough_dropout_length),
        "max_hough_dropout_run_range": [hough_dropout_start, hough_dropout_end],
        "offset_marker_coverage": offset_marker_coverage,
        "radius_cv": radius_cv,
        "path_normal_p95_r": geometry["path_normal_p95_r"],
        "path_direction_change_p95_rad": geometry["path_direction_changes"],
        "camera_drift_width": camera_drift_width,
        "angular_reversal_fraction": reversal_fraction,
        "angular_fit_nrmse": angular_nrmse,
        "angular_span_rad": angular_span,
        "motion_frame_count": int(np.sum(motion)),
        "valid_ratio_frame_count": int(np.sum(valid)),
    }

    thresholds = config["thresholds"]
    scores = {
        "task_structure": float(
            0.50 * q_score(track_coverage, thresholds["track_coverage"])
            + 0.25 * q_score(camera_drift_width, thresholds["camera_drift_width"])
            + 0.25 * q_score(radius_cv, thresholds["radius_cv"])
        ),
        "center_contact_geometry": float(
            0.65 * q_score(geometry["path_normal_p95_r"], thresholds["path_normal_p95_r"])
            + 0.35 * q_score(radius_cv, thresholds["radius_cv"])
        ),
        "marker_rotation_quality": float(
            0.40 * q_score(pattern_coverage, thresholds["pattern_coverage"])
            + 0.30 * q_score(reversal_fraction, thresholds["angular_reversal_fraction"])
            + 0.30 * q_score(angular_nrmse, thresholds["angular_fit_nrmse"])
        ),
        "pure_rolling_ratio": q_score(integrated_error, thresholds["rolling_ratio_error"]),
        "contact_point_velocity": float(
            0.65 * q_score(contact_nmae, thresholds["contact_velocity_nmae"])
            + 0.35 * q_score(contact_p95, thresholds["contact_velocity_p95"])
        ),
    }
    scores["overall"] = weighted_geometric_mean(scores, config["weights"])
    arrays = {
        "time": time,
        "x_smooth": x_smooth,
        "y_smooth": y_smooth,
        "speed": speed,
        "omega": omega,
        "rolling_speed": rolling_speed,
        "contact_velocity": contact_velocity,
        "ratio": np.where(rolling_speed > EPS, speed / rolling_speed, np.nan),
        "motion": motion,
        "valid": valid,
        "cumulative_distance": cumulative,
    }
    return metrics, scores, motion, arrays


def structural_checks(
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> tuple[bool, list[str]]:
    tracking = config["tracking"]
    reasons = []
    if metrics["track_coverage"] < float(tracking["min_track_coverage"]):
        reasons.append("insufficient_ball_track_coverage")
    if metrics["pattern_coverage"] < float(tracking["min_pattern_coverage"]):
        reasons.append("insufficient_pattern_orientation_coverage")
    if metrics["motion_frame_count"] < int(tracking["min_motion_frames"]):
        reasons.append("insufficient_continuous_motion")
    if metrics["angular_span_rad"] < float(tracking["min_angular_span_rad"]):
        reasons.append("insufficient_visible_rotation")
    if metrics["radius_cv"] > float(tracking["max_radius_cv"]):
        reasons.append("ball_radius_or_identity_unstable")
    if metrics["camera_drift_width"] > float(tracking["max_camera_drift_width"]):
        reasons.append("camera_not_locked")
    if metrics["max_hough_dropout_run_frames"] > int(
        tracking["max_hough_dropout_run_frames"]
    ):
        reasons.append("ball_visual_identity_lost")
    if metrics["valid_ratio_frame_count"] < 12:
        reasons.append("insufficient_joint_translation_rotation_evidence")
    return len(reasons) == 0, reasons


def render_plot(
    output: Path,
    track: dict[str, np.ndarray],
    angles: np.ndarray,
    pattern_confidence: np.ndarray,
    arrays: dict[str, np.ndarray],
    metrics: dict[str, Any],
) -> None:
    centers = track["stabilized_centers"]
    time = arrays["time"]
    motion = arrays["motion"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axis = axes[0, 0]
    axis.plot(centers[:, 0], centers[:, 1], color="0.65", marker=".", ms=2, label="all")
    axis.plot(centers[motion, 0], centers[motion, 1], color="#1677ff", lw=2, label="motion")
    axis.invert_yaxis()
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("stabilized ball-centre track")
    axis.legend()

    axis = axes[0, 1]
    distance = arrays["cumulative_distance"]
    radius = float(np.median(track["radii"]))
    angle_zero = angles - angles[np.flatnonzero(motion)[0]] if np.any(motion) else angles
    distance_zero = distance - distance[np.flatnonzero(motion)[0]] if np.any(motion) else distance
    axis.plot(time, distance_zero, label="travel distance s (px)")
    axis.plot(time, radius * np.abs(angle_zero), label="R |delta theta| (px)")
    axis.set_title("integrated rolling relation")
    axis.set_xlabel("time (s)")
    axis.legend()

    axis = axes[1, 0]
    axis.plot(time, arrays["speed"], label="v")
    axis.plot(time, arrays["rolling_speed"], label="omega R")
    axis.fill_between(time, 0, np.nanmax(arrays["speed"]) if len(time) else 1, where=motion, alpha=0.08)
    axis.set_title("translation and rotation speed")
    axis.set_xlabel("time (s)")
    axis.set_ylabel("px/s")
    axis.legend()

    axis = axes[1, 1]
    axis.plot(time, np.abs(arrays["contact_velocity"]) / np.maximum(arrays["speed"], EPS), label="|v-omega R|/v")
    axis.plot(time, pattern_confidence, label="pattern confidence", alpha=0.8)
    axis.plot(time, track["confidence"], label="centre confidence", alpha=0.8)
    axis.set_ylim(0, min(2.0, max(1.05, float(np.nanpercentile(np.abs(arrays["contact_velocity"]) / np.maximum(arrays["speed"], EPS), 98)))))
    m1 = metrics["M1_rolling_ratio_abs"]
    m2 = metrics["M2_contact_velocity_nmae"]
    axis.set_title(
        f"M1={float(m1):.3f}  M2={float(m2):.3f}"
        if isinstance(m1, (float, int)) and isinstance(m2, (float, int))
        else "residual and extraction confidence"
    )
    axis.set_xlabel("time (s)")
    axis.legend()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def render_overlay(
    output: Path,
    frames: list[np.ndarray],
    fps: float,
    track: dict[str, np.ndarray],
    angles: np.ndarray,
    pattern_confidence: np.ndarray,
    dot_xy: np.ndarray,
    dot_used: np.ndarray,
    arrays: dict[str, np.ndarray],
    scores: dict[str, float],
) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create overlay video: {output}")
    trail: list[tuple[int, int]] = []
    for index, source in enumerate(frames):
        frame = source.copy()
        center = track["centers"][index]
        radius = float(track["radii"][index])
        point = tuple(np.round(center).astype(int))
        trail.append(point)
        for a, b in zip(trail[:-1], trail[1:]):
            cv2.line(frame, a, b, (255, 150, 0), 2, cv2.LINE_AA)
        colour = (40, 220, 40) if arrays["motion"][index] else (170, 170, 170)
        cv2.circle(frame, point, int(round(radius)), colour, 2, cv2.LINE_AA)
        cv2.circle(frame, point, 3, (0, 0, 255), -1, cv2.LINE_AA)
        if math.isfinite(angles[index]):
            vector = np.array([math.cos(angles[index]), math.sin(angles[index])]) * 0.80 * radius
            a = tuple(np.round(center - vector).astype(int))
            b = tuple(np.round(center + vector).astype(int))
            cv2.line(frame, a, b, (255, 0, 255), 3, cv2.LINE_AA)
        ratio = arrays["ratio"][index]
        lines = [
            f"P6 deterministic evaluator | frame {index}/{len(frames)-1}",
            f"centre_conf={track['confidence'][index]:.2f} pattern_conf={pattern_confidence[index]:.2f}",
            f"v/(omega R)={ratio:.3f}" if math.isfinite(ratio) else "v/(omega R)=N/A",
            f"overall={scores['overall']:.1f}",
        ]
        for row, text in enumerate(lines):
            y = 30 + 30 * row
            cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()


def write_frame_csv(
    output: Path,
    track: dict[str, np.ndarray],
    angles: np.ndarray,
    pattern_confidence: np.ndarray,
    pattern_used: np.ndarray,
    dot_confidence: np.ndarray,
    dot_used: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> None:
    fields = [
        "frame",
        "time_s",
        "center_x",
        "center_y",
        "radius",
        "center_confidence",
        "hough_confidence",
        "flow_confidence",
        "pattern_angle_rad",
        "pattern_confidence",
        "pattern_observation_used",
        "offset_marker_confidence",
        "offset_marker_used",
        "motion_interval",
        "measurement_valid",
        "speed_px_s",
        "omega_rad_s",
        "rolling_speed_px_s",
        "contact_velocity_px_s",
        "rolling_ratio",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(angles)):
            writer.writerow(
                _finite(
                    {
                        "frame": index,
                        "time_s": arrays["time"][index],
                        "center_x": track["stabilized_centers"][index, 0],
                        "center_y": track["stabilized_centers"][index, 1],
                        "radius": track["radii"][index],
                        "center_confidence": track["confidence"][index],
                        "hough_confidence": track["hough_confidence"][index],
                        "flow_confidence": track["flow_confidence"][index],
                        "pattern_angle_rad": angles[index],
                        "pattern_confidence": pattern_confidence[index],
                        "pattern_observation_used": pattern_used[index],
                        "offset_marker_confidence": dot_confidence[index],
                        "offset_marker_used": dot_used[index],
                        "motion_interval": arrays["motion"][index],
                        "measurement_valid": arrays["valid"][index],
                        "speed_px_s": arrays["speed"][index],
                        "omega_rad_s": arrays["omega"][index],
                        "rolling_speed_px_s": arrays["rolling_speed"][index],
                        "contact_velocity_px_s": arrays["contact_velocity"][index],
                        "rolling_ratio": arrays["ratio"][index],
                    }
                )
            )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    video = Path(args.video).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    artifact_dir = Path(args.artifacts_dir).resolve() if args.artifacts_dir else output.parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.task_id != config["task_id"]:
        raise ValueError(f"task mismatch: requested {args.task_id}, config is {config['task_id']}")
    source_id = infer_source_id(video, config["sources"])
    frames, fps, (width, height) = read_video(video)
    expected_width, expected_height = config["video_size"]
    scale_x = width / expected_width
    scale_y = height / expected_height
    if abs(scale_x - scale_y) > 0.02:
        raise ValueError("video aspect/scale differs from frozen first-frame configuration")
    source = config["sources"][source_id]
    center = (float(source["center"][0]) * scale_x, float(source["center"][1]) * scale_y)
    radius = float(source["radius"]) * 0.5 * (scale_x + scale_y)
    track = track_ball(frames, center, radius)

    observed = np.full(len(frames), np.nan, dtype=float)
    pattern_confidence = np.zeros(len(frames), dtype=float)
    for index, frame in enumerate(frames):
        angle, confidence, _ = estimate_pattern_orientation(
            frame, track["centers"][index], track["radii"][index]
        )
        observed[index] = angle
        pattern_confidence[index] = confidence
    # Offset-dot candidates are intentionally not used in the formal score:
    # on a projected 3-D sphere their apparent polar phase need not be linear,
    # while the full asymmetric band supplies dense orientation evidence.
    dot_confidence = np.zeros(len(frames), dtype=float)
    dot_used = np.zeros(len(frames), dtype=bool)
    dot_xy = np.full((len(frames), 2), np.nan, dtype=float)
    # The band itself gives a dense per-frame angle sequence.  The offset dot
    # is retained as an independent rigidity/visibility diagnostic; it is not
    # allowed to force the rolling measurement when the generated dot swims or
    # changes appearance relative to the band.
    angles, pattern_used = unwrap_pattern_angles(
        observed, pattern_confidence, track["flow_rotation"]
    )
    metrics, scores, motion, arrays = compute_metrics(
        track,
        angles,
        pattern_confidence,
        pattern_used,
        dot_confidence,
        dot_used,
        fps,
        config,
    )
    structural_ok, structural_reasons = structural_checks(metrics, config)
    pass_rules = config["pass_rules"]
    dimension_scores = [scores[key] for key in config["weights"] if key != "task_structure"]
    physics_pass = bool(
        structural_ok
        and scores["overall"] >= float(pass_rules["overall_min"])
        and min(dimension_scores) >= float(pass_rules["dimension_floor"])
        and metrics["M1_rolling_ratio_abs"] is not None
        and metrics["M1_rolling_ratio_abs"] <= float(pass_rules["rolling_ratio_error_max"])
        and metrics["M2_contact_velocity_nmae"] is not None
        and metrics["M2_contact_velocity_nmae"] <= float(pass_rules["contact_velocity_nmae_max"])
    )

    render_plot(
        artifact_dir / "plot.png",
        track,
        angles,
        pattern_confidence,
        arrays,
        metrics,
    )
    render_overlay(
        artifact_dir / "overlay.mp4",
        frames,
        fps,
        track,
        angles,
        pattern_confidence,
        dot_xy,
        dot_used,
        arrays,
        scores,
    )
    write_frame_csv(
        artifact_dir / "frame_measurements.csv",
        track,
        angles,
        pattern_confidence,
        pattern_used,
        dot_confidence,
        dot_used,
        arrays,
    )

    result = {
        "schema_version": "physical-bench-evaluator-result-v1",
        "evaluator": {
            "name": "p6_pure_rolling_classical_cv",
            "version": "1.2.0",
            "uses_llm": False,
            "uses_vlm": False,
            "learned_models": [],
            "config_sha256": sha256_file(config_path),
        },
        "task_id": args.task_id,
        "sample_id": video.stem,
        "source_id": source_id,
        "video": {
            "path": str(video),
            "sha256": sha256_file(video),
            "frame_count": len(frames),
            "fps": fps,
            "width": width,
            "height": height,
        },
        "extract_success": True,
        "structural_ok": structural_ok,
        "measurements": {
            "median_ball_radius_px": float(np.median(track["radii"])),
            "motion_start_frame": int(np.flatnonzero(motion)[0]) if np.any(motion) else None,
            "motion_end_frame": int(np.flatnonzero(motion)[-1]) if np.any(motion) else None,
            "travel_distance_px": (
                float(arrays["cumulative_distance"][np.flatnonzero(motion)[-1]] - arrays["cumulative_distance"][np.flatnonzero(motion)[0]])
                if np.any(motion)
                else None
            ),
            "visible_angular_span_rad": metrics["angular_span_rad"],
        },
        "metrics": metrics,
        "scores": scores,
        "physics_pass": physics_pass,
        "failure_reason": structural_reasons if structural_reasons else None,
        "artifacts": {
            "overlay_video": str(artifact_dir / "overlay.mp4"),
            "plot": str(artifact_dir / "plot.png"),
            "frame_measurements": str(artifact_dir / "frame_measurements.csv"),
        },
        "notes": [
            "M1 is a robust integrated estimate of v/(omega*R)-1 from travel distance versus unwrapped marker angle; the derivative-based frame estimate is also reported.",
            "M2 is the robust normalized contact-point tangential speed |v-omega*R|/v.",
            "No LLM, VLM, semantic detector, or learned vision model is used.",
            "The offset-dot candidate track is diagnostic-only and does not affect the rolling score.",
        ],
    }
    # Keep the just-computed legacy labels immutable, then expose only the
    # continuous 0-1 scores at the unprefixed public ``scores`` key.
    result = apply_continuous_scoring(
        result, Path(args.continuous_score_config)
    )
    with output.open("w", encoding="utf-8") as handle:
        json.dump(_finite(result), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    default_config = Path(__file__).with_name("config_v1.yaml")
    default_continuous_config = Path(__file__).with_name(
        "continuous_score_config_v1.yaml"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--task_id", default="P6")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument(
        "--continuous-score-config",
        default=str(default_continuous_config),
        help="continuous-0-1 score mapping (applied after legacy pass labels)",
    )
    parser.add_argument("--artifacts-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = evaluate(args)
    except Exception as error:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_version": "physical-bench-evaluator-result-v1",
            "task_id": args.task_id,
            "sample_id": Path(args.video).stem,
            "extract_success": False,
            "structural_ok": None,
            "measurements": {},
            "metrics": {},
            "scores": {},
            "physics_pass": False,
            "failure_reason": [f"evaluator_exception: {type(error).__name__}: {error}"],
        }
        # Evaluator exceptions are unmeasurable by definition. They still use
        # the same public schema, with continuous overall=0 and the empty
        # legacy score retained under ``legacy_score``.
        failure = apply_continuous_scoring(
            failure, Path(args.continuous_score_config)
        )
        with output.open("w", encoding="utf-8") as handle:
            json.dump(failure, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        raise
    print(
        json.dumps(
            {
                "sample_id": result["sample_id"],
                "structural_ok": result["structural_ok"],
                "physics_pass": result["physics_pass"],
                "overall": result["scores"]["overall"],
                "M1": result["metrics"]["M1_rolling_ratio_abs"],
                "M2": result["metrics"]["M2_contact_velocity_nmae"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
