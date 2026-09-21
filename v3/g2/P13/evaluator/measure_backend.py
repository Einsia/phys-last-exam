#!/usr/bin/env python3
"""Self-contained P13 plane-mirror evaluator.

Only NumPy/OpenCV/Matplotlib are required.  This file deliberately does not import
any module from the surrounding vdmbench checkout, so the whole P13 directory can
be copied and evaluated on its own.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

TASK_ID = "P13"
TASK_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vdmbench-matplotlib")


def finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def residual_physics_score(error: float, scale: float = 1.0) -> float:
    """V2 residual normalization q(e; a) = 1 / (1 + |e| / a)."""
    error, scale = float(error), float(scale)
    if not math.isfinite(error) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("residual error must be finite and scale must be finite and positive")
    return 1.0 / (1.0 + abs(error) / scale)


def recognized_score(physics_score: float) -> float:
    """Add the V2 recognition share once to a measurable M metric."""
    return 0.15 + 0.85 * float(np.clip(physics_score, 0.0, 1.0))


def relative_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    try:
        return path.resolve().relative_to(TASK_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def metadata(video: str, image_path: str | None, seed: int | None) -> tuple[str, str | None, int | None]:
    stem = Path(video).stem
    match = re.search(r"_seed(\d+)$", stem, flags=re.IGNORECASE)
    actual_seed = seed if seed is not None else (int(match.group(1)) if match else None)
    base = re.sub(r"_seed\d+$", "", stem, flags=re.IGNORECASE)
    if image_path is None:
        simulation = base.startswith(("sim_", "simulation_"))
        candidates = ([TASK_ROOT / "sim_first_frame.png",
                       TASK_ROOT / "first_frames" / "simulation" / f"{base}.png",
                       TASK_ROOT / "first_frame.png"]
                      if simulation else
                      [TASK_ROOT / "first_frame.png",
                       TASK_ROOT / "first_frames" / "gpt" / f"{base}.png"])
        image_path = str(next((candidate for candidate in candidates if candidate.is_file()), "")) or None
        if image_path is None:
            group = "simulation" if simulation else "gpt"
            images = sorted((TASK_ROOT / "first_frames" / group).glob("*.png"))
            image_path = str(images[0]) if len(images) == 1 else None
    return relative_path(video) or str(video), relative_path(image_path), actual_seed


def infer_model(video: str, supplied: str) -> str:
    if supplied and supplied != "unknown":
        return supplied
    for part in Path(video).parts:
        lower = part.lower()
        if "minimax" in lower and "h3" in lower:
            return part
    return supplied or "unknown"


def result_payload(video: str, image_path: str | None, seed: int | None, model: str,
                   success: bool, measurement: dict[str, Any] | None,
                   m1: float | None, m2: float | None, failure_reason: str | None,
                   debug: dict[str, Any], sample_id: str,
                   warnings: list[str] | None = None) -> dict[str, Any]:
    video_value, image_value, seed_value = metadata(video, image_path, seed)
    return finite({
        "task_id": TASK_ID,
        "video_path": video_value,
        "image_path": image_value,
        "seed": seed_value,
        "model": infer_model(video, model),
        "metrics": {
            "M1": {"extract_success": success, "metric": m1 if success else None},
            "M2": {"extract_success": success, "metric": m2 if success else None},
            "M3": None,
        },
        "verbose": {
            "sample_id": sample_id,
            "failure_reason": failure_reason,
            "quality_warnings": warnings or [],
            "measurements": measurement or {},
            "debug": debug,
        },
    })


def read_video(path: str, max_frames: int | None = None) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"video contains no frames: {path}")
    return np.stack(frames), fps


def hough_lines(frame: np.ndarray, min_length: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                          minLineLength=min_length, maxLineGap=8)
    if raw is None:
        return []
    out = []
    # OpenCV has returned both (N, 1, 4) and (N, 4) layouts across builds.
    # Normalizing here prevents a valid control video from becoming a runtime
    # error merely because the installed OpenCV uses the latter layout.
    for x1, y1, x2, y2 in np.asarray(raw).reshape(-1, 4):
        a = np.array([float(x1), float(y1)])
        b = np.array([float(x2), float(y2)])
        out.append((a, b, float(np.linalg.norm(b - a))))
    return sorted(out, key=lambda item: item[2], reverse=True)


def direction(line: tuple[np.ndarray, np.ndarray, float]) -> np.ndarray:
    d = line[1] - line[0]
    norm = np.linalg.norm(d)
    return d / norm if norm else np.zeros(2)


def intersection(x: tuple[np.ndarray, np.ndarray, float],
                 y: tuple[np.ndarray, np.ndarray, float]) -> np.ndarray | None:
    p, r = x[0], x[1] - x[0]
    q, s = y[0], y[1] - y[0]
    cross = float(r[0] * s[1] - r[1] * s[0])
    if abs(cross) < 1e-8:
        return None
    t = float(((q - p)[0] * s[1] - (q - p)[1] * s[0]) / cross)
    return p + t * r


def angle_to_normal(line: tuple[np.ndarray, np.ndarray, float], normal: np.ndarray) -> float:
    d = direction(line)
    n = np.linalg.norm(normal)
    if not n or not np.linalg.norm(d):
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(d, normal / n)), 0.0, 1.0))))


def load_rois() -> tuple[dict[str, tuple[float, float, float, float]], str]:
    path = TASK_ROOT / "evaluator" / "roi.json"
    if not path.is_file():
        return {}, "auto"
    try:
        data = json.loads(path.read_text())
        raw = data.get("rois", data)
        rois = {}
        for key, value in raw.items():
            if key in {"task_id", "frame_size", "source_video", "source_frame"}:
                continue
            if isinstance(value, (list, tuple)) and len(value) == 4:
                vals = tuple(float(item) for item in value)
                if all(0.0 <= item <= 1.0 for item in vals) and vals[2] > 0 and vals[3] > 0:
                    if vals[0] + vals[2] <= 1.0 and vals[1] + vals[3] <= 1.0:
                        rois[key] = vals
        return rois, "annotated"
    except (OSError, ValueError, TypeError):
        return {}, "auto_invalid"


def roi_pixels(box: tuple[float, float, float, float] | None,
               width: int, height: int) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    x, y, w, h = box
    return (max(0, int(round(x * width))), max(0, int(round(y * height))),
            min(width, int(round((x + w) * width))), min(height, int(round((y + h) * height))))


def color_mask(frame: np.ndarray, kind: str) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if kind == "red":
        mask = (((h <= 12) | (h >= 168)) & (s >= 90) & (v >= 100))
    else:
        # Cyan/blue emergent ray used by the generated videos.
        mask = ((h >= 72) & (h <= 112) & (s >= 70) & (v >= 90))
    mask = mask.astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def fit_segment(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    if points is None or len(points) < 8:
        return None
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).reshape(-1)
    direction = np.array([float(vx), float(vy)], dtype=float)
    direction /= max(np.linalg.norm(direction), 1e-9)
    center = np.array([float(x0), float(y0)], dtype=float)
    values = (np.asarray(points, dtype=float) - center) @ direction
    p0 = center + float(values.min()) * direction
    p1 = center + float(values.max()) * direction
    return p0, p1, float(np.linalg.norm(p1 - p0))


def points_in_roi(mask: np.ndarray, box: tuple[float, float, float, float] | None) -> np.ndarray:
    h, w = mask.shape[:2]
    px = roi_pixels(box, w, h)
    if px is not None:
        x0, y0, x1, y1 = px
        clipped = np.zeros_like(mask)
        clipped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        mask = clipped
    ys, xs = np.where(mask > 0)
    return np.column_stack([xs, ys]).astype(float) if len(xs) else np.empty((0, 2), dtype=float)


def mask_in_roi(mask: np.ndarray, box: tuple[float, float, float, float] | None) -> np.ndarray:
    """Return a copy of a mask clipped to the optional annotated search ROI."""
    px = roi_pixels(box, mask.shape[1], mask.shape[0])
    if px is None:
        return mask
    x0, y0, x1, y1 = px
    clipped = np.zeros_like(mask)
    clipped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return clipped


def select_reflected_line(frame: np.ndarray, contact: np.ndarray,
                          predicted: np.ndarray,
                          box: tuple[float, float, float, float] | None
                          ) -> dict[str, Any] | None:
    """Find one non-vertical colored ray that leaves the incidence point.

    Generated clips sometimes draw the reflected ray red and sometimes cyan,
    while the normal is cyan in both cases.  A single fit over all cyan pixels
    therefore locks onto the normal.  Hough candidates are evaluated after
    excluding the incident side, the vertical neighborhood of the normal, and
    downward rays; the longest remaining segment is the visible outgoing ray.
    """
    height, width = frame.shape[:2]
    predicted = np.asarray(predicted, dtype=float)
    predicted /= max(float(np.linalg.norm(predicted)), 1e-9)
    contact = np.asarray(contact, dtype=float)
    candidates: list[dict[str, Any]] = []
    for kind in ("red", "cyan"):
        mask = mask_in_roi(color_mask(frame, kind), box)
        ys, xs = np.where(mask > 0)
        if len(xs) < 12:
            continue
        points = np.column_stack([xs, ys]).astype(float)
        relative = points - contact
        distance = np.linalg.norm(relative, axis=1)
        # The reflected ray lies on the predicted side of the normal.  This
        # removes the red incident beam without requiring a fixed left/right
        # layout, and keeps the search above the plane where possible.
        keep = ((distance >= 8.0)
                & (relative @ predicted >= 0.04 * distance)
                & (relative[:, 0] * predicted[0] >= 0.04 * distance)
                & (relative[:, 1] <= 0.18 * height)
                & (np.abs(relative[:, 0]) >= max(8.0, 0.012 * width)))
        selected = points[keep]
        if len(selected) < 12:
            continue
        candidate_mask = np.zeros(mask.shape, dtype=np.uint8)
        candidate_mask[selected[:, 1].astype(int), selected[:, 0].astype(int)] = 255
        raw = cv2.HoughLinesP(candidate_mask, 1, np.pi / 180,
                              threshold=max(12, int(width * 0.012)),
                              minLineLength=max(18, int(width * 0.035)),
                              maxLineGap=12)
        if raw is None:
            continue
        for x1, y1, x2, y2 in np.asarray(raw).reshape(-1, 4):
            a = np.array([float(x1), float(y1)])
            b = np.array([float(x2), float(y2)])
            segment = b - a
            length = float(np.linalg.norm(segment))
            if length < max(18.0, 0.035 * width):
                continue
            near, far = (a, b) if np.linalg.norm(a - contact) <= np.linalg.norm(b - contact) else (b, a)
            near_distance = float(np.linalg.norm(near - contact))
            ray = (far - contact) / max(float(np.linalg.norm(far - contact)), 1e-9)
            alignment = float(np.dot(ray, predicted))
            # The normal is vertical; reject it explicitly instead of allowing
            # a long normal segment to dominate the score.
            if abs(float(ray[0])) < 0.12 or ray[1] > 0.08 or alignment < 0.0:
                continue
            cross = abs(float(segment[0] * (contact - near)[1]
                            - segment[1] * (contact - near)[0])) / max(length, 1e-9)
            if near_distance > 0.20 * height or cross > max(16.0, 0.045 * height):
                continue
            score = length * (0.65 + 0.35 * max(0.0, alignment))
            candidates.append({"a": near, "b": far, "length": float(np.linalg.norm(far - contact)),
                               "score": score, "confidence": min(1.0, len(selected) / 120.0)
                               * min(1.0, length / max(0.12 * width, 1.0)),
                               "color": kind, "contact_error_px": cross})
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item["score"]))


def select_mirror(frame: np.ndarray, rois: dict[str, tuple[float, float, float, float]]) -> tuple[np.ndarray, np.ndarray, float] | None:
    height, width = frame.shape[:2]
    hough = hough_lines(frame, max(30, int(width * 0.035)))
    box = roi_pixels(rois.get("mirror"), width, height)
    candidates = []
    for a, b, length in hough:
        if box is not None:
            x0, y0, x1, y1 = box
            if not (x0 <= a[0] <= x1 or x0 <= b[0] <= x1):
                continue
            if not (y0 <= a[1] <= y1 or y0 <= b[1] <= y1):
                continue
        direction = b - a
        if abs(direction[0]) < 1 or abs(direction[1] / direction[0]) > 0.35:
            continue
        center_y = float((a[1] + b[1]) / 2.0)
        if not 0.08 * height <= center_y <= 0.75 * height:
            continue
        candidates.append((length, a, b))
    if not candidates:
        return None
    # Hough splits long mirror edges into several segments. Merge segments
    # with nearly equal y before selecting the physical edge.
    clusters: list[list[tuple[float, np.ndarray, np.ndarray]]] = []
    for length, a, b in sorted(candidates, key=lambda item: float((item[1][1] + item[2][1]) / 2)):
        cy = float((a[1] + b[1]) / 2)
        cluster = next((c for c in clusters if abs(cy - c[0][0]) <= 10), None)
        if cluster is None:
            clusters.append([(cy, a, b)])
        else:
            cluster.append((cy, a, b))
    merged = []
    for cluster in clusters:
        points = np.vstack([np.asarray(item[1:3]) for item in cluster]); y = float(np.median(points[:, 1]))
        a = np.array([float(points[:, 0].min()), y]); b = np.array([float(points[:, 0].max()), y])
        merged.append((float(np.linalg.norm(b - a)), a, b))
    _, a, b = max(merged, key=lambda item: item[0])
    return a, b, float(np.linalg.norm(b - a))


def reflection_direction(incident: np.ndarray, normal: np.ndarray) -> np.ndarray:
    d = incident / max(np.linalg.norm(incident), 1e-9)
    n = normal / max(np.linalg.norm(normal), 1e-9)
    out = d - 2.0 * float(np.dot(d, n)) * n
    return out / max(np.linalg.norm(out), 1e-9)


def extract(frames: np.ndarray) -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
    height, width = frames.shape[1:3]
    rois, roi_source = load_rois()
    mirror = select_mirror(frames[0], rois)
    debug: dict[str, Any] = {"roi_source": roi_source, "roi": rois,
                             "frame_lines": [], "reflected_lines": [],
                             "reflected_lengths_px": [], "reflected_angles_deg": [],
                             "reflected_confidence": []}
    if mirror is None:
        return None, debug, "P13_mirror_missing"
    mirror_line = (mirror[0], mirror[1], mirror[2])
    # The task defines the mirror normal as the image-vertical line.  Using a
    # Hough-derived mirror edge here lets a slightly slanted board edge bias
    # every incidence/reflection angle, so keep this reference exactly vertical.
    normal = np.array([0.0, -1.0])
    first_count = max(3, min(8, len(frames)))
    incident_points = np.concatenate([points_in_roi(color_mask(frame, "red"), rois.get("incident_ray"))
                                      for frame in frames[:first_count]], axis=0)
    incident = fit_segment(incident_points)
    if incident is None:
        return None, debug, "P13_incident_ray_missing"
    incident_line = (incident[0], incident[1], incident[2])
    contact = None
    # The visible incident beam endpoint is the physical hit point. Prefer it
    # when it lies inside the merged mirror span; using an edge intersection
    # would move the point outside the mirror for a rectangular slab.
    mirror_x0, mirror_x1 = sorted((float(mirror[0][0]), float(mirror[1][0])))
    endpoints = [incident[0], incident[1]]
    inside = [point for point in endpoints if mirror_x0 - .04 * width <= point[0] <= mirror_x1 + .04 * width]
    if inside:
        contact = max(inside, key=lambda point: float(np.linalg.norm(point - np.mean(incident_points, axis=0))))
    if contact is None:
        contact = intersection(mirror_line, incident_line)
    if contact is None or not np.all(np.isfinite(contact)):
        return None, debug, "P13_incidence_intersection_missing"
    incident_angle = angle_to_normal(incident_line, normal)
    if not np.isfinite(incident_angle) or incident_angle <= 1:
        return None, debug, "P13_degenerate_incident_angle"
    predicted = reflection_direction(direction(incident_line), normal)
    reflected_lines = []
    lengths = []
    angles = []
    confidences = []
    intersection_errors = []
    for frame in frames:
        selected = select_reflected_line(frame, contact, predicted, rois.get("reflected_ray"))
        if selected is None:
            reflected_lines.append(None); lengths.append(float("nan")); angles.append(float("nan")); confidences.append(0.0); continue
        near = np.asarray(selected["a"], dtype=float)
        far = np.asarray(selected["b"], dtype=float)
        ray = (far - contact) / max(float(np.linalg.norm(far - contact)), 1e-9)
        visible_length = float(max(np.linalg.norm(far - contact), 0.0))
        reflected_lines.append({"a": near, "b": far, "length": visible_length,
                                "color": selected["color"]})
        lengths.append(visible_length)
        angle = angle_to_normal((contact, contact + ray, 1.0), normal)
        angles.append(angle)
        confidence = float(selected["confidence"])
        confidences.append(confidence)
        intersection_errors.append(float(selected["contact_error_px"]))
    valid = np.isfinite(lengths) & (np.asarray(confidences) >= 0.25)
    if int(valid.sum()) < max(3, len(frames) // 12):
        return None, {**debug, "mirror": {"a": mirror[0], "b": mirror[1]},
                      "incident": {"a": incident[0], "b": incident[1]},
                      "incidence_point": contact, "reflected_lines": reflected_lines,
                      "reflected_lengths_px": lengths, "reflected_angles_deg": angles,
                      "reflected_confidence": confidences,
                      "normal_direction": normal}, "P13_reflected_ray_missing"
    median_angle = float(np.nanmedian(np.asarray(angles)[valid]))
    error = float(np.median(intersection_errors) / max(np.hypot(width, height), 1.0)) if intersection_errors else float("nan")
    measurement = {
        "incident_angle_deg": float(incident_angle),
        "reflected_angle_deg": median_angle,
        "angle_difference_deg": abs(float(incident_angle) - median_angle),
        "normal_direction": "vertical",
        "normal_angle_deg_from_vertical": 0.0,
        "incidence_point_error_norm": error,
        "stable_frame_count": int(valid.sum()),
        "rays_visible": True,
        "first_frame_contact": True,
        "incident_static": True,
        "reflected_visible_frame_count": int(valid.sum()),
        "reflected_propagation_frame_range": [int(np.flatnonzero(valid)[0]), int(np.flatnonzero(valid)[-1])],
        "reflected_length_px": float(np.nanmedian(np.asarray(lengths)[valid])),
        "ray_confidence": float(np.median(np.asarray(confidences)[valid])),
        "mirror_confidence": 1.0,
    }
    debug.update({"mirror": {"a": mirror[0], "b": mirror[1]},
                  "incident": {"a": incident[0], "b": incident[1]},
                  "incidence_point": contact, "predicted_reflection_direction": predicted,
                  "reflected_lines": reflected_lines, "reflected_lengths_px": lengths,
                  "reflected_angles_deg": angles, "reflected_confidence": confidences,
                  "normal_direction": normal})
    return measurement, debug, None


def draw_debug(frames: np.ndarray, debug: dict[str, Any], root: Path,
               sample_id: str, fps: float,
               measurement: dict[str, Any] | None = None) -> dict[str, str | None]:
    folder = root / sample_id
    folder.mkdir(parents=True, exist_ok=True)
    raw_path = folder / "measurements.json"
    raw_path.write_text(json.dumps(finite(debug), indent=2, allow_nan=False))
    plot_path = folder / "plot.png"
    plt = __import__("matplotlib.pyplot", fromlist=["subplots"])
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(debug.get("reflected_lengths_px", []), label="reflected length (px)")
    axes[1].plot(debug.get("reflected_angles_deg", []), label="measured reflected angle (deg)")
    measurement = measurement or {}
    axes[1].axhline(float(measurement.get("incident_angle_deg", np.nan)), color="black", linestyle="--", label="incident angle")
    axes[0].set_ylabel("length"); axes[1].set_ylabel("angle"); axes[1].set_xlabel("frame")
    for axis in axes: axis.legend(loc="best")
    fig.suptitle(f"{sample_id}: dynamic reflected-ray tracking"); fig.tight_layout(); fig.savefig(plot_path, dpi=120); plt.close(fig)
    overlay_path = folder / "overlay.mp4"
    writer = cv2.VideoWriter(str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps or 24.0, (frames.shape[2], frames.shape[1]))
    if not writer.isOpened():
        return {"directory": str(folder), "plot": str(plot_path),
                "measurements": str(raw_path), "overlay_video": None}
    mirror_item = debug.get("mirror")
    incident_item = debug.get("incident")
    contact = np.asarray(debug.get("incidence_point", [np.nan, np.nan]), dtype=float)
    predicted = np.asarray(debug.get("predicted_reflection_direction", [np.nan, np.nan]), dtype=float)
    normal = np.asarray(debug.get("normal_direction", [0.0, -1.0]), dtype=float)
    for index, frame in enumerate(frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if mirror_item:
            a, b = np.asarray(mirror_item["a"], int), np.asarray(mirror_item["b"], int)
            cv2.line(bgr, tuple(a), tuple(b), (0, 220, 220), 2)
        if incident_item:
            a, b = np.asarray(incident_item["a"], int), np.asarray(incident_item["b"], int)
            cv2.line(bgr, tuple(a), tuple(b), (0, 0, 255), 2)
        if np.all(np.isfinite(contact)) and np.all(np.isfinite(predicted)):
            p = contact.astype(int); q = (contact + predicted * max(frame.shape[:2]) * 1.2).astype(int)
            cv2.arrowedLine(bgr, tuple(p), tuple(q), (0, 220, 0), 2, tipLength=0.04)
            cv2.circle(bgr, tuple(p), 6, (255, 0, 255), -1)
            if np.all(np.isfinite(normal)) and np.linalg.norm(normal) > 0:
                normal_unit = normal / np.linalg.norm(normal)
                n0 = (contact - normal_unit * max(frame.shape[:2]) * 0.45).astype(int)
                n1 = (contact + normal_unit * max(frame.shape[:2]) * 0.45).astype(int)
                cv2.line(bgr, tuple(n0), tuple(n1), (255, 255, 0), 2)
                cv2.putText(bgr, "normal", (int(contact[0]) + 8, int(contact[1]) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1)
        lines = debug.get("reflected_lines", [])
        if index < len(lines) and lines[index]:
            item = lines[index]; a, b = np.asarray(item["a"], int), np.asarray(item["b"], int)
            cv2.line(bgr, tuple(a), tuple(b), (255, 80, 0), 3)
            cv2.putText(bgr, f"ray {item['length']:.0f}px", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 80, 0), 2)
        writer.write(bgr)
    writer.release()
    return {"directory": str(folder), "plot": str(plot_path),
            "measurements": str(raw_path), "overlay_video": str(overlay_path)}


def evaluate_video(video: str, output: Path, sample_id: str, max_frames: int | None,
                   image_path: str | None, seed: int | None, model: str) -> int:
    try:
        frames, fps = read_video(video, max_frames)
        measurement, debug_data, reason = extract(frames)
        debug = draw_debug(frames, debug_data, output.parent.parent / "debug"
                            if output.parent.name == "json" else output.parent / "debug",
                            sample_id, fps, measurement)
        if measurement is None:
            payload = result_payload(video, image_path, seed, model, False, None,
                                     None, None, reason, debug, sample_id)
        else:
            q1 = residual_physics_score(measurement["angle_difference_deg"], 90.0)
            q2 = residual_physics_score(measurement["incidence_point_error_norm"])
            m1, m2 = recognized_score(q1), recognized_score(q2)
            measurement["score_normalization"] = {
                "score_range": [0.0, 1.0], "higher_is_better": True,
                "M1": {"raw_measurement": "angle_difference_deg",
                       "scale": 90.0, "physics_score": q1, "recognition_score": 0.15,
                       "formula": "0.15 + 0.85 / (1 + abs(angle_difference_deg) / 90)", "score": m1},
                "M2": {"raw_measurement": "incidence_point_error_norm",
                       "scale": 1.0, "physics_score": q2, "recognition_score": 0.15,
                       "formula": "0.15 + 0.85 / (1 + abs(incidence_point_error_norm))", "score": m2},
            }
            warnings = []
            if measurement["angle_difference_deg"] > 15.0:
                warnings.append("P13_reflection_angle_mismatch")
            if measurement["ray_confidence"] < 0.45:
                warnings.append("P13_reflected_ray_low_confidence")
            payload = result_payload(video, image_path, seed, model, True, measurement,
                                     m1, m2, None, debug, sample_id, warnings)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        debug_root = output.parent.parent / "debug" if output.parent.name == "json" else output.parent / "debug"
        folder = debug_root / sample_id; folder.mkdir(parents=True, exist_ok=True)
        error_path = folder / "error.txt"; error_path.write_text(error + "\n")
        payload = result_payload(video, image_path, seed, model, False, None, None, None,
                                 error, {"directory": str(folder), "error": str(error_path)},
                                 sample_id)
        print(error, file=sys.stderr)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, allow_nan=False))
        return 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(finite(payload), indent=2, allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-contained P13 evaluator")
    parser.add_argument("--video")
    parser.add_argument("--task_id", default=TASK_ID)
    parser.add_argument("--output")
    parser.add_argument("--videos")
    parser.add_argument("--outdir")
    parser.add_argument("--model", default="unknown")
    parser.add_argument("--image-path", "--image_path", dest="image_path")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sample-id")
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.task_id.upper() != TASK_ID:
        parser.error(f"this evaluator only supports {TASK_ID}")
    if args.video:
        if not args.output:
            parser.error("--video requires --output")
        return evaluate_video(args.video, Path(args.output), args.sample_id or Path(args.video).stem,
                              args.max_frames, args.image_path, args.seed, args.model)
    if not args.videos:
        parser.error("provide --video or --videos")
    video_root = Path(args.videos)
    files = [video_root] if video_root.is_file() else sorted(video_root.glob("*.mp4"))
    if not files:
        print(f"no MP4 videos found under {video_root}", file=sys.stderr); return 1
    outdir = Path(args.outdir or "eval_results")
    json_dir = outdir / "json"; json_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(files, 1):
        rc = evaluate_video(str(path), json_dir / f"{path.stem}.json", path.stem,
                            args.max_frames, args.image_path, args.seed, args.model)
        print(f"[{index}/{len(files)}] {path.stem} {'OK' if rc == 0 else 'ERROR'}")
    import csv
    with (outdir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "video_path", "image_path", "seed", "model", "M1_extract_success", "M1", "M2_extract_success", "M2", "M3", "failure_reason"])
        writer.writeheader()
        for path in sorted(json_dir.glob("*.json")):
            payload = json.loads(path.read_text())
            writer.writerow({"task_id": payload["task_id"], "video_path": payload["video_path"],
                             "image_path": payload["image_path"], "seed": payload["seed"],
                             "model": payload["model"],
                             "M1_extract_success": payload["metrics"]["M1"]["extract_success"],
                             "M1": payload["metrics"]["M1"]["metric"],
                             "M2_extract_success": payload["metrics"]["M2"]["extract_success"],
                             "M2": payload["metrics"]["M2"]["metric"], "M3": None,
                             "failure_reason": payload["verbose"]["failure_reason"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
