"""Deterministic, VLM-free evaluator for Group 7 task P12 (optics).

The evaluator measures ray geometry directly from pixels:

* a long, nearly horizontal Hough line establishes the flat interface;
* a saturated thin-line colour is selected as the laser mask;
* air-side and water-side ray segments are fitted independently;
* segments sharing an interface intersection are paired for Snell estimates;
* an unpaired water-side reflected pair supplies the critical-angle estimate.

No learned model, OCR, prompt text, or task answer is used for extraction.  The
known water index and critical angle are reported only as optional diagnostic
errors; P12's primary M1 is the coefficient of variation of independently
measured index estimates.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


TASK_ID = "P12"
EXPECTED_N_WATER = 1.333
EXPECTED_CRITICAL_DEG = 48.75
MIN_SNELL_PAIRS = 3
# A strict P12 pass must show a *pair* of water-side branches that meet at
# one interface point and are mirror-symmetric about the surface normal.  The
# looser single-water-ray fallback below remains useful diagnostic evidence
# (and keeps extraction from throwing away a measurable angle), but it is not
# evidence of total internal reflection.
CRITICAL_REFLECTION_MAX_RESIDUAL_DEG = 10.0
CRITICAL_REFLECTION_MAX_MISMATCH_RATIO = 0.020


@dataclass(frozen=True)
class Interface:
    """Straight interface in point/tangent/normal form."""

    point: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    angle_deg: float
    residual_px: float
    support_length_px: float
    laser_crossings: int


@dataclass(frozen=True)
class RayLine:
    """One fitted laser segment on one side of the interface."""

    side: str
    intersection: np.ndarray
    interface_coordinate: float
    angle_normal_deg: float
    tangent_per_normal: float
    direction: np.ndarray
    length_px: float
    support_count: int
    endpoints: tuple[int, int, int, int]


@dataclass
class FrameAnalysis:
    success: bool
    failure_reasons: list[dict[str, Any]]
    measurements: dict[str, Any]
    overlay: np.ndarray
    laser_mask: np.ndarray
    quality_score: float


def _failure(code: str, detail: str, **evidence: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "detail": detail}
    if evidence:
        item["evidence"] = evidence
    return item


def _json_float(value: float | np.floating[Any] | None) -> float | None:
    """Return a finite JSON number, preserving unavailable values as null."""

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _circular_hue_distance(hue: np.ndarray, center: int) -> np.ndarray:
    delta = np.abs(hue.astype(np.int16) - int(center))
    return np.minimum(delta, 180 - delta)


def _line_length(line: Sequence[int | float]) -> float:
    x1, y1, x2, y2 = map(float, line)
    return math.hypot(x2 - x1, y2 - y1)


def _line_angle_horizontal_deg(line: Sequence[int | float]) -> float:
    x1, y1, x2, y2 = map(float, line)
    angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180.0
    return min(angle, 180.0 - angle)


def _hough_segments(
    mask: np.ndarray,
    *,
    threshold: int,
    min_length: float,
    max_gap: float,
) -> list[tuple[int, int, int, int]]:
    raw = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 360.0,
        threshold=max(8, int(threshold)),
        minLineLength=max(8, int(min_length)),
        maxLineGap=max(2, int(max_gap)),
    )
    if raw is None:
        return []
    # OpenCV has returned both ``(N, 1, 4)`` and ``(N, 4)`` here across
    # versions (and some builds return a single ``(4,)`` row).  Normalising
    # before iterating avoids treating the first scalar of a flat row as an
    # iterable, which caused ``numpy.int32 is not iterable`` under OpenCV 5.
    arr = np.asarray(raw)
    if arr.size == 0:
        return []
    arr = arr.reshape(-1, 4)
    return [tuple(int(value) for value in row[:4]) for row in arr]


def _candidate_laser_mask(frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose the saturated colour band with the strongest thin oblique lines."""

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    high_sv = (sat >= 72) & (val >= 105)
    kernel = np.ones((3, 3), np.uint8)

    best_score = -1.0
    best_mask = np.zeros((h, w), np.uint8)
    best_meta: dict[str, Any] = {}
    # A water/glass body can occupy a large saturated region and produce many
    # spurious Hough edges.  Real laser beams are thin, so retain a second
    # shortlist of low-coverage candidates and prefer it whenever it has
    # meaningful oblique support.  This is especially important for the P12
    # incident-only first-frame protocol, where there are no outgoing rays to
    # provide extra geometric evidence.
    thin_candidates: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    # A broad cyan/blue hue may cover most of a tank, but its ray is usually a
    # separate elongated connected component.  Keep those components in a
    # second pool so the water body cannot suppress an otherwise measurable ray.
    component_candidates: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    max_component_area = max(60000, int(0.04 * h * w))
    # Fixed circular bins ensure a small red laser is still considered even when
    # a large cyan water region dominates the global hue histogram.
    for center in range(0, 180, 10):
        selected = high_sv & (_circular_hue_distance(hue, center) <= 9)
        mask = (selected.astype(np.uint8) * 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        coverage = float(np.count_nonzero(mask)) / float(h * w)
        if coverage < 0.00008:
            continue
        segments = _hough_segments(
            mask,
            threshold=max(12, int(min(h, w) * 0.018)),
            min_length=max(18.0, min(h, w) * 0.045),
            max_gap=max(6.0, min(h, w) * 0.018),
        )
        oblique = [
            seg
            for seg in segments
            if 8.0 <= _line_angle_horizontal_deg(seg) <= 88.5
        ]
        lengths = sorted((_line_length(seg) for seg in oblique), reverse=True)
        if not lengths:
            continue
        # Cap each segment so duplicate Hough edges cannot dominate indefinitely.
        capped = sum(min(length, 0.30 * math.hypot(w, h)) for length in lengths[:24])
        coverage_penalty = 1.0 if coverage <= 0.025 else 0.025 / coverage
        score = capped * coverage_penalty
        if coverage <= 0.035 and capped > 0.0:
            thin_candidates.append((score, mask, {
                "hue_center": center,
                "coverage": coverage,
                "oblique_hough_segments": len(oblique),
                "selection_score": score,
                "selection_pool": "thin",
            }))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            bw = int(stats[label, cv2.CC_STAT_WIDTH]); bh = int(stats[label, cv2.CC_STAT_HEIGHT])
            longest = max(bw, bh); shortest = max(1, min(bw, bh))
            if area < 24 or area > max_component_area or longest < max(18.0, min(h, w) * 0.045) or longest / shortest < 1.45:
                continue
            component = np.where(labels == label, 255, 0).astype(np.uint8)
            component_segments = _hough_segments(
                component,
                threshold=max(12, int(min(h, w) * 0.018)),
                min_length=max(18.0, min(h, w) * 0.045),
                max_gap=max(6.0, min(h, w) * 0.018),
            )
            component_oblique = [
                seg for seg in component_segments
                if 8.0 <= _line_angle_horizontal_deg(seg) <= 88.5
            ]
            component_lengths = sorted((_line_length(seg) for seg in component_oblique), reverse=True)
            if not component_lengths:
                continue
            component_score = sum(min(length, 0.30 * math.hypot(w, h)) for length in component_lengths[:24])
            component_candidates.append((component_score, component, {
                "hue_center": center,
                "coverage": float(np.count_nonzero(component)) / float(h * w),
                "oblique_hough_segments": len(component_oblique),
                "selection_score": component_score,
                "selection_pool": "thin_component",
                "component_area": area,
            }))
        # Keep the legacy winner only from a sparse hue band.  A broad cyan
        # tank surface can have a large aggregate Hough score but is not a
        # valid single-colour laser mask; its isolated ray is handled by the
        # component pool below when multi-colour mode is warranted.
        if coverage <= 0.035 and score > best_score:
            best_score = score
            best_mask = mask
            best_meta = {
                "hue_center": center,
                "coverage": coverage,
                "oblique_hough_segments": len(oblique),
                "selection_score": score,
            }

    # Prefer thin candidates over a broad cyan/blue body.  P12 first frames may
    # deliberately use one distinct colour per ray to make manual tracking
    # easier, so retain the strongest candidate from each separated hue cluster
    # and analyse their union.  Adjacent 10-degree bins from the same beam are
    # merged; broad saturated tank surfaces are excluded by the coverage cap.
    candidate_pool = thin_candidates + component_candidates
    # Only enter multi-colour mode when at least three independent hue groups
    # have substantial oblique support.  This preserves the legacy single-red
    # mask for ordinary H3 clips, where weak blue/green tank reflections can
    # otherwise create dozens of false ray fragments.
    hue_scores: dict[int, float] = {}
    for score, _, meta in candidate_pool:
        center = int(meta.get("hue_center", 0))
        hue_scores[center] = max(hue_scores.get(center, 0.0), float(score))
    hue_groups: list[dict[str, Any]] = []
    for center, score in sorted(hue_scores.items(), key=lambda item: item[1], reverse=True):
        group = next(
            (item for item in hue_groups if float(_circular_hue_distance(np.asarray([center], dtype=np.uint8), int(item["center"]))[0]) <= 15.0),
            None,
        )
        if group is None:
            hue_groups.append({"center": center, "score": score})
        else:
            group["score"] = max(float(group["score"]), score)
    top_hue_score = max(hue_scores.values(), default=0.0)
    robust_hue_groups = [group for group in hue_groups if float(group["score"]) >= max(100.0, 0.25 * top_hue_score)]
    # A generated clip can develop several broad coloured reflections after
    # frame 0.  Estimate the union footprint before enabling multi-colour mode;
    # genuine thin rays occupy only a small fraction of the image, whereas a
    # reflection/water mask quickly exceeds this conservative 4% budget.
    quick_union = np.zeros((h, w), np.uint8)
    quick_chosen: list[np.ndarray] = []
    for score, mask, _ in sorted(candidate_pool, key=lambda item: item[0], reverse=True):
        if score < max(100.0, 0.25 * top_hue_score):
            continue
        pixels = int(np.count_nonzero(mask))
        if pixels <= 0:
            continue
        if any(
            int(np.count_nonzero((mask > 0) & (prior > 0))) / float(min(pixels, max(1, int(np.count_nonzero(prior))))) > 0.42
            for prior in quick_chosen
        ):
            continue
        prospective = cv2.bitwise_or(quick_union, mask)
        quick_chosen.append(mask)
        quick_union = prospective
        if len(quick_chosen) >= 12:
            break
    quick_union_coverage = float(np.count_nonzero(quick_union)) / float(h * w)
    multi_color_enabled = len(robust_hue_groups) >= 3 and quick_union_coverage <= 0.04

    if candidate_pool and multi_color_enabled:
        ordered = sorted(candidate_pool, key=lambda item: item[0], reverse=True)
        best_thin_score = ordered[0][0]
        chosen: list[tuple[float, np.ndarray, dict[str, Any]]] = []
        chosen_centers: list[int] = []
        union = np.zeros((h, w), np.uint8)
        for candidate in ordered:
            score, mask, meta = candidate
            center = int(meta.get("hue_center", 0))
            # Keep weaker hues (notably cyan against a blue tank) as long as
            # they have real oblique support.  A low floor prevents isolated
            # antialias noise from entering the union.
            if score < max(100.0, 0.08 * best_thin_score):
                continue
            pixels = int(np.count_nonzero(mask))
            if pixels <= 0:
                continue
            # Adjacent hue bins for one antialiased stripe overlap strongly;
            # deduplicate by pixels instead of hue distance so nearby but
            # distinct green/amber rays are retained.
            duplicate = False
            for _, prior_mask, _ in chosen:
                overlap = int(np.count_nonzero((mask > 0) & (prior_mask > 0)))
                prior_pixels = max(1, int(np.count_nonzero(prior_mask)))
                if overlap / float(min(pixels, prior_pixels)) > 0.42:
                    duplicate = True
                    break
            if duplicate:
                continue
            prospective = cv2.bitwise_or(union, mask)
            prospective_coverage = float(np.count_nonzero(prospective)) / float(h * w)
            if prospective_coverage > 0.20:
                continue
            chosen.append(candidate)
            chosen_centers.append(center)
            union = prospective
            if len(chosen) >= 8:
                break
        if chosen:
            best_score = float(sum(item[0] for item in chosen))
            best_mask = union
            best_meta = {
                "hue_centers": chosen_centers,
                "coverage": float(np.count_nonzero(union)) / float(h * w),
                "oblique_hough_segments": int(sum(item[2].get("oblique_hough_segments", 0) for item in chosen)),
                "selection_score": best_score,
                "selection_pool": "thin_union",
                "candidate_count": len(chosen),
                "component_candidates": int(sum(item[2].get("selection_pool") == "thin_component" for item in chosen)),
                "color_group_count": len(robust_hue_groups),
                "candidate_union_coverage": quick_union_coverage,
            }
    elif candidate_pool and not multi_color_enabled and thin_candidates:
        # Legacy compatibility path: when one colour dominates, retain the
        # historical hue-band union (adjacent bins within 20° are one stripe).
        # This keeps previously scored H3 videos numerically stable while the
        # component-aware path above is reserved for true multi-colour frames.
        ordered = sorted(thin_candidates, key=lambda item: item[0], reverse=True)
        legacy_best = ordered[0][0]
        legacy_union = np.zeros((h, w), np.uint8)
        legacy_chosen: list[tuple[float, np.ndarray, dict[str, Any]]] = []
        legacy_centers: list[int] = []
        for candidate in ordered:
            score, mask, meta = candidate
            center = int(meta.get("hue_center", 0))
            if score < 0.30 * legacy_best:
                break
            if any(float(_circular_hue_distance(np.asarray([center], dtype=np.uint8), prior)[0]) <= 20.0 for prior in legacy_centers):
                continue
            prospective = cv2.bitwise_or(legacy_union, mask)
            if float(np.count_nonzero(prospective)) / float(h * w) > 0.10:
                continue
            legacy_chosen.append(candidate)
            legacy_centers.append(center)
            legacy_union = prospective
        if legacy_chosen:
            best_score = float(sum(item[0] for item in legacy_chosen))
            best_mask = legacy_union
            best_meta = {
                "hue_centers": legacy_centers,
                "coverage": float(np.count_nonzero(legacy_union)) / float(h * w),
                "oblique_hough_segments": int(sum(item[2].get("oblique_hough_segments", 0) for item in legacy_chosen)),
                "selection_score": best_score,
                "selection_pool": "thin_union_legacy",
                "candidate_count": len(legacy_chosen),
                "color_group_count": len(robust_hue_groups),
                "candidate_union_coverage": quick_union_coverage,
            }
        else:
            best_meta = {**best_meta, "selection_pool": "single_hue_compat", "color_group_count": len(robust_hue_groups)}
    elif candidate_pool and best_score < 0.0:
        # A single-colour frame with no sparse whole-band winner can still use
        # its strongest isolated component, without enabling a broad union.
        fallback_score, fallback_mask, fallback_meta = max(candidate_pool, key=lambda item: item[0])
        best_score = fallback_score
        best_mask = fallback_mask
        best_meta = {**fallback_meta, "selection_pool": "single_hue_component_compat", "color_group_count": len(robust_hue_groups)}
    elif candidate_pool and not multi_color_enabled:
        best_meta = {**best_meta, "selection_pool": "single_hue_compat", "color_group_count": len(robust_hue_groups), "candidate_union_coverage": quick_union_coverage}

    # Red/orange fallback for antialiased rays whose saturated core is very thin.
    if best_score < 0.0:
        red = ((_circular_hue_distance(hue, 0) <= 18) & (sat >= 52) & (val >= 90))
        best_mask = red.astype(np.uint8) * 255
        best_meta = {
            "hue_center": 0,
            "coverage": float(np.count_nonzero(best_mask)) / float(h * w),
            "oblique_hough_segments": 0,
            "selection_score": 0.0,
            "fallback": "red_orange",
        }

    # Join antialiased beam edges, while retaining enough separation at vertices.
    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return best_mask, best_meta


def _ray_crossings_for_line(mask: np.ndarray, line: Sequence[int], band: int) -> int:
    """Count separated laser intersections along a nearly horizontal line."""

    h, w = mask.shape
    x1, y1, x2, y2 = map(float, line)
    if abs(x2 - x1) < 1.0:
        return 0
    left = max(0, int(math.floor(min(x1, x2))))
    right = min(w - 1, int(math.ceil(max(x1, x2))))
    if right <= left:
        return 0
    xs = np.arange(left, right + 1)
    ys = y1 + (xs - x1) * (y2 - y1) / (x2 - x1)
    active = np.zeros(xs.shape[0], np.uint8)
    for offset in range(-band, band + 1):
        yy = np.clip(np.rint(ys + offset).astype(np.int32), 0, h - 1)
        active |= (mask[yy, xs] > 0).astype(np.uint8)
    # Merge the few-pixel gap produced by a bright interface crossing a beam.
    active = cv2.morphologyEx(active.reshape(1, -1) * 255, cv2.MORPH_CLOSE, np.ones((1, 7), np.uint8))[0]
    count, _, stats, _ = cv2.connectedComponentsWithStats((active > 0).astype(np.uint8).reshape(1, -1))
    crossings = 0
    for idx in range(1, count):
        width = int(stats[idx, cv2.CC_STAT_WIDTH])
        if 2 <= width <= max(50, int(0.055 * w)):
            crossings += 1
    return crossings


def _interface_transition(frame: np.ndarray, line: Sequence[int], offset: int = 7) -> float:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(float, line)
    if abs(x2 - x1) < 1.0:
        return 0.0
    left = max(int(0.12 * w), int(math.floor(min(x1, x2))))
    right = min(int(0.88 * w), int(math.ceil(max(x1, x2))))
    if right - left < 20:
        return 0.0
    xs = np.arange(left, right + 1, max(1, w // 500))
    ys = y1 + (xs - x1) * (y2 - y1) / (x2 - x1)
    ya = np.clip(np.rint(ys - offset).astype(np.int32), 0, h - 1)
    yb = np.clip(np.rint(ys + offset).astype(np.int32), 0, h - 1)
    a = frame[ya, xs].astype(np.float32)
    b = frame[yb, xs].astype(np.float32)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def _detect_interface(frame: np.ndarray, laser_mask: np.ndarray) -> tuple[Interface | None, dict[str, Any]]:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 130)
    segments = _hough_segments(
        edges,
        threshold=max(30, int(w * 0.035)),
        min_length=max(80.0, w * 0.22),
        max_gap=max(12.0, w * 0.045),
    )
    candidates: list[tuple[float, tuple[int, int, int, int], int, float]] = []
    for seg in segments:
        angle = _line_angle_horizontal_deg(seg)
        y_mid = 0.5 * (seg[1] + seg[3])
        if angle > 8.0 or not (0.12 * h <= y_mid <= 0.88 * h):
            continue
        crossings = _ray_crossings_for_line(laser_mask, seg, band=max(3, int(0.006 * h)))
        transition = _interface_transition(frame, seg)
        length = _line_length(seg)
        score = length / w + 0.90 * min(crossings, 6) + 0.010 * min(transition, 100.0)
        candidates.append((score, seg, crossings, transition))

    if not candidates:
        return None, {"horizontal_hough_candidates": 0}

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, seed, seed_crossings, _ = candidates[0]
    seed_angle = math.atan2(seed[3] - seed[1], seed[2] - seed[0])
    seed_mid_y = 0.5 * (seed[1] + seed[3])
    points: list[tuple[float, float]] = []
    support_length = 0.0
    for _, seg, _, _ in candidates:
        angle = math.atan2(seg[3] - seg[1], seg[2] - seg[0])
        while angle - seed_angle > math.pi / 2:
            angle -= math.pi
        while seed_angle - angle > math.pi / 2:
            angle += math.pi
        mid_y = 0.5 * (seg[1] + seg[3])
        if abs(math.degrees(angle - seed_angle)) <= 2.5 and abs(mid_y - seed_mid_y) <= max(7.0, 0.012 * h):
            points.extend([(float(seg[0]), float(seg[1])), (float(seg[2]), float(seg[3]))])
            support_length += _line_length(seg)

    if len(points) < 2:
        return None, {"horizontal_hough_candidates": len(candidates)}

    pts = np.asarray(points, dtype=np.float32)
    vx, vy, x0, y0 = map(float, cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1))
    tangent = np.array([vx, vy], dtype=np.float64)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
    if tangent[0] < 0:
        tangent = -tangent
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    if normal[1] < 0:
        normal = -normal
    point = np.array([x0, y0], dtype=np.float64)
    residuals = np.abs((pts.astype(np.float64) - point) @ normal)
    angle_deg = math.degrees(math.atan2(tangent[1], tangent[0]))
    interface = Interface(
        point=point,
        tangent=tangent,
        normal=normal,
        angle_deg=angle_deg,
        residual_px=float(np.sqrt(np.mean(residuals ** 2))),
        support_length_px=support_length,
        laser_crossings=seed_crossings,
    )
    meta = {
        "horizontal_hough_candidates": len(candidates),
        "angle_deg": angle_deg,
        "residual_px": interface.residual_px,
        "support_length_px": support_length,
        "laser_crossings": seed_crossings,
    }
    return interface, meta


def _signed_distance_grid(shape: tuple[int, int], interface: Interface) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    return (
        (xx.astype(np.float32) - float(interface.point[0])) * float(interface.normal[0])
        + (yy.astype(np.float32) - float(interface.point[1])) * float(interface.normal[1])
    )


def _extract_zone_segments(
    mask: np.ndarray,
    interface: Interface,
    side: str,
) -> list[dict[str, Any]]:
    h, w = mask.shape
    distance = _signed_distance_grid((h, w), interface)
    gap = max(5.0, 0.008 * min(h, w))
    if side == "air":
        zone = distance < -gap
    elif side == "water":
        zone = distance > gap
    else:
        raise ValueError(f"unknown side: {side}")
    zone_mask = np.where(zone, mask, 0).astype(np.uint8)
    segments = _hough_segments(
        zone_mask,
        threshold=max(12, int(min(h, w) * 0.016)),
        min_length=max(24.0, min(h, w) * 0.055),
        max_gap=max(7.0, min(h, w) * 0.022),
    )
    out: list[dict[str, Any]] = []
    for endpoints in segments:
        x1, y1, x2, y2 = map(float, endpoints)
        p = np.array([x1, y1], dtype=np.float64)
        v = np.array([x2 - x1, y2 - y1], dtype=np.float64)
        length = float(np.linalg.norm(v))
        if length < 1.0:
            continue
        denom = float(v @ interface.normal)
        if abs(denom) < 0.12 * length:
            # Nearly parallel to the interface: not an incident/refracted ray.
            continue
        t = -float((p - interface.point) @ interface.normal) / denom
        intersection = p + t * v
        coord = float((intersection - interface.point) @ interface.tangent)
        # Do not admit lines whose extrapolated crossing is far outside frame.
        if not (-0.12 * w <= intersection[0] <= 1.12 * w and -0.12 * h <= intersection[1] <= 1.12 * h):
            continue
        unit = v / length
        normal_projection = abs(float(unit @ interface.normal))
        angle_normal = math.degrees(math.acos(float(np.clip(normal_projection, 0.0, 1.0))))
        tangent_per_normal = float(v @ interface.tangent) / denom
        if not (1.5 <= angle_normal <= 84.0):
            continue
        out.append(
            {
                "side": side,
                "intersection": intersection,
                "coord": coord,
                "angle": angle_normal,
                "tpn": tangent_per_normal,
                "direction": unit,
                "length": length,
                "endpoints": tuple(map(int, endpoints)),
            }
        )
    return out


def _cluster_ray_segments(
    segments: Sequence[dict[str, Any]],
    *,
    width: int,
    angle_tol: float = 4.0,
) -> list[RayLine]:
    """Merge duplicate Hough edges/fragments using crossing and orientation."""

    # A thick beam produces two Hough edges whose extrapolated interface
    # intercepts separate by thickness/cos(angle); allow that without merging
    # distinct P12 beams, which are deliberately spaced far apart.
    x_tol = max(12.0, 0.018 * width)
    # Thick antialiased multi-colour beams may request a slightly wider
    # tolerance from the caller; legacy single-colour clips retain 4° here so
    # their historical segment counts/metrics remain stable.
    clusters: list[list[dict[str, Any]]] = []
    # Long candidates seed clusters first, making the result insensitive to Hough order.
    for seg in sorted(segments, key=lambda item: item["length"], reverse=True):
        assigned = False
        for cluster in clusters:
            weights = np.asarray([item["length"] for item in cluster], dtype=float)
            coord = float(np.average([item["coord"] for item in cluster], weights=weights))
            angle = float(np.average([item["angle"] for item in cluster], weights=weights))
            tpn = float(np.average([item["tpn"] for item in cluster], weights=weights))
            sign_ok = (seg["tpn"] == 0.0 or tpn == 0.0 or math.copysign(1.0, seg["tpn"]) == math.copysign(1.0, tpn))
            if sign_ok and abs(seg["coord"] - coord) <= x_tol and abs(seg["angle"] - angle) <= angle_tol:
                cluster.append(seg)
                assigned = True
                break
        if not assigned:
            clusters.append([seg])

    rays: list[RayLine] = []
    for cluster in clusters:
        weights = np.asarray([item["length"] for item in cluster], dtype=float)
        weight_sum = float(np.sum(weights))
        intersection = np.average(np.stack([item["intersection"] for item in cluster]), axis=0, weights=weights)
        direction = np.average(np.stack([item["direction"] for item in cluster]), axis=0, weights=weights)
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        representative = max(cluster, key=lambda item: item["length"])
        rays.append(
            RayLine(
                side=str(representative["side"]),
                intersection=np.asarray(intersection, dtype=np.float64),
                interface_coordinate=float(np.average([item["coord"] for item in cluster], weights=weights)),
                angle_normal_deg=float(np.average([item["angle"] for item in cluster], weights=weights)),
                tangent_per_normal=float(np.average([item["tpn"] for item in cluster], weights=weights)),
                direction=direction,
                length_px=max(float(item["length"]) for item in cluster),
                support_count=len(cluster),
                endpoints=tuple(representative["endpoints"]),
            )
        )
    return sorted(rays, key=lambda ray: ray.interface_coordinate)


def _refine_ray_lines(
    rays: Sequence[RayLine],
    mask: np.ndarray,
    interface: Interface,
) -> list[RayLine]:
    """Refit Hough clusters to beam centre pixels instead of either beam edge."""

    h, w = mask.shape
    yy, xx = np.nonzero(mask)
    if xx.size == 0:
        return list(rays)
    points = np.column_stack((xx, yy)).astype(np.float64)
    signed = (points - interface.point) @ interface.normal
    gap = max(5.0, 0.008 * min(h, w))
    band = max(7.0, 0.011 * min(h, w))
    refined: list[RayLine] = []
    for ray in rays:
        side_ok = signed < -gap if ray.side == "air" else signed > gap
        perpendicular = np.array([-ray.direction[1], ray.direction[0]], dtype=np.float64)
        near = np.abs((points - ray.intersection) @ perpendicular) <= band
        support = points[side_ok & near]
        if support.shape[0] < 20:
            refined.append(ray)
            continue
        vx, vy, x0, y0 = map(
            float,
            cv2.fitLine(support.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1),
        )
        direction = np.array([vx, vy], dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        point = np.array([x0, y0], dtype=np.float64)
        denom = float(direction @ interface.normal)
        if abs(denom) < 0.12:
            refined.append(ray)
            continue
        crossing = point - direction * (float((point - interface.point) @ interface.normal) / denom)
        coord = float((crossing - interface.point) @ interface.tangent)
        angle = math.degrees(math.acos(float(np.clip(abs(denom), 0.0, 1.0))))
        tpn = float(direction @ interface.tangent) / denom
        # Draw the measured centre line across the actual support extent.
        along = (support - point) @ direction
        p1 = point + direction * float(np.min(along))
        p2 = point + direction * float(np.max(along))
        refined.append(
            RayLine(
                side=ray.side,
                intersection=crossing,
                interface_coordinate=coord,
                angle_normal_deg=angle,
                tangent_per_normal=tpn,
                direction=direction,
                length_px=float(np.max(along) - np.min(along)),
                support_count=int(support.shape[0]),
                endpoints=(
                    int(round(p1[0])),
                    int(round(p1[1])),
                    int(round(p2[0])),
                    int(round(p2[1])),
                ),
            )
        )
    return sorted(refined, key=lambda item: item.interface_coordinate)


def _pair_snell_rays(
    air: Sequence[RayLine],
    water: Sequence[RayLine],
    *,
    width: int,
) -> tuple[list[dict[str, Any]], set[int]]:
    tolerance = max(16.0, 0.022 * width)
    edges: list[tuple[float, int, int]] = []
    for ai, incident in enumerate(air):
        for wi, refracted in enumerate(water):
            same_lateral_side = (
                incident.tangent_per_normal == 0.0
                or refracted.tangent_per_normal == 0.0
                or math.copysign(1.0, incident.tangent_per_normal)
                == math.copysign(1.0, refracted.tangent_per_normal)
            )
            delta = abs(incident.interface_coordinate - refracted.interface_coordinate)
            if delta <= tolerance:
                # Crossing agreement determines measurability.  A generated
                # outgoing branch can land on the physically wrong lateral
                # side of the normal; that is still an angle measurement, not
                # an extraction failure.  Prefer the physically usual side
                # only as a tie-breaker and expose the relation in the pair's
                # diagnostics so a bad video cannot be silently "fixed" by
                # the extractor.
                side_penalty = 0.0 if same_lateral_side else 0.30 * tolerance
                cost = (
                    delta
                    + 0.08 * abs(incident.angle_normal_deg - refracted.angle_normal_deg)
                    + side_penalty
                )
                edges.append((cost, ai, wi))

    used_air: set[int] = set()
    used_water: set[int] = set()
    pairs: list[dict[str, Any]] = []
    for _, ai, wi in sorted(edges):
        if ai in used_air or wi in used_water:
            continue
        incident, refracted = air[ai], water[wi]
        same_lateral_side = (
            incident.tangent_per_normal == 0.0
            or refracted.tangent_per_normal == 0.0
            or math.copysign(1.0, incident.tangent_per_normal)
            == math.copysign(1.0, refracted.tangent_per_normal)
        )
        sin_water = math.sin(math.radians(refracted.angle_normal_deg))
        if abs(sin_water) < 1e-5:
            continue
        n_est = math.sin(math.radians(incident.angle_normal_deg)) / sin_water
        pairs.append(
            {
                "air_index": ai,
                "water_index": wi,
                "interface_x_px": float(0.5 * (incident.intersection[0] + refracted.intersection[0])),
                "interface_y_px": float(0.5 * (incident.intersection[1] + refracted.intersection[1])),
                "intersection_mismatch_px": abs(incident.interface_coordinate - refracted.interface_coordinate),
                "incidence_angle_deg": incident.angle_normal_deg,
                "refraction_angle_deg": refracted.angle_normal_deg,
                "refracted_branch_same_lateral_side": bool(same_lateral_side),
                "n_snell": n_est,
            }
        )
        used_air.add(ai)
        used_water.add(wi)
    pairs.sort(key=lambda item: item["interface_x_px"])
    return pairs, used_water


def _find_critical_ray(
    air: Sequence[RayLine],
    water: Sequence[RayLine],
    used_water: set[int],
    *,
    width: int,
) -> dict[str, Any] | None:
    tolerance = max(18.0, 0.025 * width)
    remaining = [(idx, ray) for idx, ray in enumerate(water) if idx not in used_water]
    candidates: list[tuple[float, dict[str, Any]]] = []
    for pos, (idx_a, ray_a) in enumerate(remaining):
        for idx_b, ray_b in remaining[pos + 1 :]:
            opposite = ray_a.tangent_per_normal * ray_b.tangent_per_normal < 0.0
            separation = abs(ray_a.interface_coordinate - ray_b.interface_coordinate)
            if not opposite or separation > tolerance:
                continue
            angle = 0.5 * (ray_a.angle_normal_deg + ray_b.angle_normal_deg)
            reflection_residual = abs(ray_a.angle_normal_deg - ray_b.angle_normal_deg)
            if not (20.0 <= angle <= 80.0):
                continue
            # Prefer mirror symmetry and a well-isolated water-only interface event.
            nearest_air = min(
                (abs(ray_a.interface_coordinate - incident.interface_coordinate) for incident in air),
                default=float("inf"),
            )
            isolation_penalty = max(0.0, tolerance - nearest_air) / tolerance
            score = reflection_residual + 0.05 * separation + 8.0 * isolation_penalty
            n_critical = 1.0 / math.sin(math.radians(angle))
            candidates.append(
                (
                    score,
                    {
                        "mode": "water_side_reflection_pair",
                        "water_indices": [idx_a, idx_b],
                        "interface_x_px": float(0.5 * (ray_a.intersection[0] + ray_b.intersection[0])),
                        "interface_y_px": float(0.5 * (ray_a.intersection[1] + ray_b.intersection[1])),
                        "critical_angle_deg": angle,
                        "reflection_residual_deg": reflection_residual,
                        "intersection_mismatch_px": separation,
                        "n_critical": n_critical,
                    },
                )
            )
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]

    # A grazing critical beam may have only its incident water branch visible.
    # This fallback is measurement-capable but explicitly lower-confidence.
    isolated: list[tuple[float, int, RayLine]] = []
    for idx, ray in remaining:
        nearest_air = min(
            (abs(ray.interface_coordinate - incident.interface_coordinate) for incident in air),
            default=float("inf"),
        )
        if nearest_air > 1.35 * tolerance and 20.0 <= ray.angle_normal_deg <= 80.0:
            # Select by visual isolation and support, never by closeness to the
            # expected answer (which would make extraction circular).
            isolated.append((-(nearest_air + 0.02 * ray.length_px), idx, ray))
    if not isolated:
        return None
    _, idx, ray = min(isolated)
    angle = ray.angle_normal_deg
    return {
        "mode": "single_water_incident_lower_confidence",
        "water_indices": [idx],
        "interface_x_px": float(ray.intersection[0]),
        "interface_y_px": float(ray.intersection[1]),
        "critical_angle_deg": angle,
        "reflection_residual_deg": None,
        "intersection_mismatch_px": None,
        "n_critical": 1.0 / math.sin(math.radians(angle)),
    }


def _verify_critical_reflection(
    critical: dict[str, Any] | None,
    water: Sequence[RayLine],
    *,
    width: int,
) -> tuple[bool, dict[str, Any]]:
    """Verify that a critical-ray candidate is actual reflected geometry.

    ``_find_critical_ray`` deliberately has a low-confidence fallback for a
    lone water-side incident line.  That fallback is important for preserving
    raw angle measurements, but it must never satisfy the benchmark's strict
    TIR premise.  This helper keeps the distinction explicit and auditable:
    two distinct water rays, opposite lateral slopes, a finite mirror-angle
    residual, and a common interface crossing are all required.
    """

    evidence: dict[str, Any] = {
        "required_mode": "water_side_reflection_pair",
        "max_reflection_residual_deg": CRITICAL_REFLECTION_MAX_RESIDUAL_DEG,
        "max_intersection_mismatch_px": max(
            12.0, CRITICAL_REFLECTION_MAX_MISMATCH_RATIO * float(width)
        ),
    }
    if not isinstance(critical, dict):
        evidence["reason"] = "critical_candidate_missing"
        return False, evidence
    evidence["mode"] = critical.get("mode")
    if critical.get("mode") != "water_side_reflection_pair":
        evidence["reason"] = "candidate_is_not_a_reflection_pair"
        return False, evidence

    indices = critical.get("water_indices")
    if not isinstance(indices, (list, tuple)) or len(indices) != 2:
        evidence["reason"] = "reflection_pair_does_not_have_two_water_rays"
        return False, evidence
    try:
        idx_a, idx_b = (int(indices[0]), int(indices[1]))
    except (TypeError, ValueError):
        evidence["reason"] = "reflection_pair_indices_invalid"
        return False, evidence
    if idx_a == idx_b or not (0 <= idx_a < len(water) and 0 <= idx_b < len(water)):
        evidence["reason"] = "reflection_pair_indices_out_of_range"
        return False, evidence
    ray_a, ray_b = water[idx_a], water[idx_b]
    opposite = (
        ray_a.tangent_per_normal * ray_b.tangent_per_normal < 0.0
    )
    evidence["water_indices"] = [idx_a, idx_b]
    evidence["opposite_lateral_slopes"] = bool(opposite)
    if not opposite:
        evidence["reason"] = "reflection_pair_slopes_not_opposite"
        return False, evidence

    residual = _json_float(critical.get("reflection_residual_deg"))
    mismatch = _json_float(critical.get("intersection_mismatch_px"))
    angle = _json_float(critical.get("critical_angle_deg"))
    n_critical = _json_float(critical.get("n_critical"))
    evidence.update(
        {
            "reflection_residual_deg": residual,
            "intersection_mismatch_px": mismatch,
            "critical_angle_deg": angle,
            "n_critical": n_critical,
        }
    )
    if residual is None:
        evidence["reason"] = "reflection_residual_missing"
        return False, evidence
    if residual > CRITICAL_REFLECTION_MAX_RESIDUAL_DEG:
        evidence["reason"] = "reflection_residual_too_large"
        return False, evidence
    if mismatch is None:
        evidence["reason"] = "reflection_intersection_mismatch_missing"
        return False, evidence
    if mismatch > evidence["max_intersection_mismatch_px"]:
        evidence["reason"] = "reflection_intersection_mismatch_too_large"
        return False, evidence
    if angle is None or not (20.0 <= angle <= 80.0):
        evidence["reason"] = "critical_angle_out_of_measurable_range"
        return False, evidence
    if n_critical is None or n_critical <= 1.0:
        evidence["reason"] = "critical_index_missing_or_unphysical"
        return False, evidence
    evidence["reason"] = "verified"
    return True, evidence


def _draw_infinite_segment(
    image: np.ndarray,
    ray: RayLine,
    interface: Interface,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = ray.endpoints
    cv2.line(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    p = tuple(np.rint(ray.intersection).astype(int))
    if 0 <= p[0] < w and 0 <= p[1] < h:
        cv2.circle(image, p, max(4, thickness + 2), color, 1, cv2.LINE_AA)


def _make_overlay(
    frame: np.ndarray,
    interface: Interface | None,
    air: Sequence[RayLine],
    water: Sequence[RayLine],
    pairs: Sequence[dict[str, Any]],
    critical: dict[str, Any] | None,
    failures: Sequence[dict[str, Any]],
) -> np.ndarray:
    overlay = frame.copy()
    h, w = overlay.shape[:2]
    scale = max(0.55, min(w, h) / 1024.0)
    thickness = max(2, int(round(3 * scale)))
    if interface is not None:
        for x in (0, w - 1):
            if abs(interface.tangent[0]) < 1e-9:
                continue
            y = interface.point[1] + (x - interface.point[0]) * interface.tangent[1] / interface.tangent[0]
            if x == 0:
                p0 = (x, int(round(y)))
            else:
                p1 = (x, int(round(y)))
        if "p0" in locals() and "p1" in locals():
            cv2.line(overlay, p0, p1, (80, 255, 80), thickness, cv2.LINE_AA)

    critical_indices = set(critical.get("water_indices", [])) if critical else set()
    paired_air = {int(pair["air_index"]) for pair in pairs}
    paired_water = {int(pair["water_index"]) for pair in pairs}
    for idx, ray in enumerate(air):
        _draw_infinite_segment(overlay, ray, interface, (0, 220, 255) if idx in paired_air else (0, 128, 255), thickness)
    for idx, ray in enumerate(water):
        if idx in critical_indices:
            color = (255, 80, 255)
        elif idx in paired_water:
            color = (255, 220, 0)
        else:
            color = (255, 128, 0)
        _draw_infinite_segment(overlay, ray, interface, color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y_text = max(22, int(28 * scale))
    if failures:
        text = "P12 extraction: " + ", ".join(str(item["code"]) for item in failures)
        cv2.putText(overlay, text, (12, y_text), font, 0.62 * scale, (40, 40, 255), max(1, thickness - 1), cv2.LINE_AA)
    else:
        estimates = [float(pair["n_snell"]) for pair in pairs]
        if critical:
            estimates.append(float(critical["n_critical"]))
        cv = float(np.std(estimates) / np.mean(estimates)) if estimates and np.mean(estimates) else 0.0
        critical_text = f", n-critical={critical['n_critical']:.3f}" if critical else ", n-critical=unavailable"
        text = f"P12: {len(pairs)} Snell pairs{critical_text}, CV={cv:.4f}"
    cv2.putText(overlay, text, (12, y_text), font, 0.62 * scale, (80, 255, 80), max(1, thickness - 1), cv2.LINE_AA)
    return overlay


def analyze_frame(frame: np.ndarray) -> FrameAnalysis:
    """Analyze one BGR frame and return geometry plus a deterministic overlay."""

    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        blank = np.zeros((64, 64, 3), np.uint8)
        reason = _failure("invalid_frame", "Expected one non-empty BGR image.")
        return FrameAnalysis(False, [reason], {}, blank, blank[:, :, 0], 0.0)
    h, w = frame.shape[:2]
    if h < 160 or w < 240:
        reason = _failure("frame_too_small", "Ray angles cannot be resolved reliably at this size.", width=w, height=h)
        return FrameAnalysis(False, [reason], {}, frame.copy(), np.zeros((h, w), np.uint8), 0.0)

    laser_mask, mask_meta = _candidate_laser_mask(frame)
    failures: list[dict[str, Any]] = []
    diagnostic_failures: list[dict[str, Any]] = []
    if np.count_nonzero(laser_mask) < max(30, int(0.00005 * h * w)):
        failures.append(_failure("laser_mask_not_detected", "No saturated thin-line laser colour was found.", **mask_meta))
        overlay = _make_overlay(frame, None, [], [], [], None, failures)
        return FrameAnalysis(False, failures, {"laser_mask": mask_meta}, overlay, laser_mask, 0.0)

    interface, interface_meta = _detect_interface(frame, laser_mask)
    if interface is None:
        failures.append(
            _failure(
                "interface_not_detected",
                "No long flat line with laser crossings could be fitted.",
                **interface_meta,
            )
        )
        overlay = _make_overlay(frame, None, [], [], [], None, failures)
        return FrameAnalysis(False, failures, {"laser_mask": mask_meta}, overlay, laser_mask, 0.0)

    if abs(interface.angle_deg) > 10.0:
        diagnostic_failures.append(
            _failure(
                "interface_tilt_out_of_range",
                "Detected interface is too steep for the locked frontal P12 view.",
                angle_deg=interface.angle_deg,
            )
        )
    if interface.residual_px > max(5.0, 0.009 * h):
        diagnostic_failures.append(
            _failure(
                "interface_not_flat",
                "Interface line-fit residual is too large.",
                residual_px=interface.residual_px,
            )
        )

    air_raw = _extract_zone_segments(laser_mask, interface, "air")
    water_raw = _extract_zone_segments(laser_mask, interface, "water")
    # Multi-colour Blender/GPT first frames often expose two edge fits of one
    # thick stripe with a ~5° angular spread.  Use the wider merge only for the
    # explicit union mask; legacy single-colour video masks retain 4° behavior.
    cluster_angle_tol = 6.0 if mask_meta.get("selection_pool") == "thin_union" else 4.0
    air = _cluster_ray_segments(air_raw, width=w, angle_tol=cluster_angle_tol)
    water = _cluster_ray_segments(water_raw, width=w, angle_tol=cluster_angle_tol)
    air = _refine_ray_lines(air, laser_mask, interface)
    water = _refine_ray_lines(water, laser_mask, interface)
    if len(air) < MIN_SNELL_PAIRS or len(water) < MIN_SNELL_PAIRS + 1:
        diagnostic_failures.append(
            _failure(
                "insufficient_ray_segments",
                "Not enough independently fitted ray segments exist on both sides of the interface.",
                air_segments=len(air),
                water_segments=len(water),
                required_air=MIN_SNELL_PAIRS,
                required_water=MIN_SNELL_PAIRS + 1,
            )
        )
    if not air or not water:
        failures.append(
            _failure(
                "no_ray_segments",
                "No fitted ray segment was found on one side of the interface.",
                air_segments=len(air), water_segments=len(water),
            )
        )

    pairs, used_water = _pair_snell_rays(air, water, width=w)
    if not pairs:
        failures.append(
            _failure(
                "no_snell_pair",
                "No incident/outgoing ray pair shares a measurable interface endpoint.",
                air_segments=len(air), water_segments=len(water),
            )
        )
        # Keep the historical, human-readable diagnostic as well.  It is
        # useful when inspecting incident-only first frames, while the new
        # extraction policy still treats it as a hard failure only because no
        # pair at all exists.
        failures.append(
            _failure(
                "insufficient_snell_pairs",
                "No Snell pair was available (minimum-count warning).",
                pair_count=0,
                required=MIN_SNELL_PAIRS,
            )
        )
    if len(pairs) < MIN_SNELL_PAIRS:
        diagnostic_failures.append(
            _failure(
                "insufficient_snell_pairs",
                "Fewer than three incident/refracted pairs share interface intersections.",
                pair_count=len(pairs),
                required=MIN_SNELL_PAIRS,
            )
        )

    critical = _find_critical_ray(air, water, used_water, width=w)
    if critical is None:
        diagnostic_failures.append(
            _failure(
                "critical_ray_not_detected",
                "No isolated water-side critical/TIR ray could be measured.",
                unpaired_water_segments=max(0, len(water) - len(used_water)),
            )
        )
    critical_reflection_verified, critical_reflection_evidence = _verify_critical_reflection(
        critical, water, width=w
    )
    if critical is not None and not critical_reflection_verified:
        diagnostic_failures.append(
            _failure(
                "critical_reflection_not_verified",
                "A critical-ray candidate was found, but it is not a verified water-side reflection pair.",
                **critical_reflection_evidence,
            )
        )

    # Deduplicate codes when a low segment count necessarily causes low pair count.
    seen: set[str] = set()
    failures = [item for item in failures if not (item["code"] in seen or seen.add(item["code"]))]
    overlay = _make_overlay(frame, interface, air, water, pairs, critical, failures)

    n_snell = [float(pair["n_snell"]) for pair in pairs if math.isfinite(float(pair["n_snell"]))]
    all_estimates = list(n_snell)
    if critical is not None and math.isfinite(float(critical["n_critical"])):
        all_estimates.append(float(critical["n_critical"]))
    mean_n = float(np.mean(all_estimates)) if all_estimates else float("nan")
    cv = float(np.std(all_estimates) / mean_n) if len(all_estimates) >= 2 and abs(mean_n) > 1e-9 else float("nan")
    snell_mean = float(np.mean(n_snell)) if n_snell else float("nan")
    residuals = [abs(value - snell_mean) for value in n_snell] if n_snell else []

    measurements: dict[str, Any] = {
        "image_width": w,
        "image_height": h,
        "laser_mask": mask_meta,
        "interface": {
            **interface_meta,
            "point_px": interface.point.tolist(),
            "tangent_xy": interface.tangent.tolist(),
            "normal_xy": interface.normal.tolist(),
        },
        "air_ray_count": len(air),
        "water_ray_count": len(water),
        # Preserve every fitted centre line in verbose evidence.  The public
        # schema promotes only declared M1/M2, but these records make endpoint
        # pairing and the second-half angle sweep independently auditable.
        "air_rays": [
            {
                "interface_x_px": float(ray.intersection[0]),
                "interface_y_px": float(ray.intersection[1]),
                "interface_coordinate_px": float(ray.interface_coordinate),
                "angle_normal_deg": float(ray.angle_normal_deg),
                "tangent_per_normal": float(ray.tangent_per_normal),
                "length_px": float(ray.length_px),
                "support_count": int(ray.support_count),
                "endpoints": list(map(int, ray.endpoints)),
            }
            for ray in air
        ],
        "water_rays": [
            {
                "interface_x_px": float(ray.intersection[0]),
                "interface_y_px": float(ray.intersection[1]),
                "interface_coordinate_px": float(ray.interface_coordinate),
                "angle_normal_deg": float(ray.angle_normal_deg),
                "tangent_per_normal": float(ray.tangent_per_normal),
                "length_px": float(ray.length_px),
                "support_count": int(ray.support_count),
                "endpoints": list(map(int, ray.endpoints)),
            }
            for ray in water
        ],
        "snell_pair_count": len(pairs),
        "snell_pairs": pairs,
        "n_snell_estimates": n_snell,
        "n_snell_mean": _json_float(snell_mean),
        "per_ray_snell_residual": residuals,
        "critical_ray": critical,
        "critical_reflection_verified": bool(critical_reflection_verified),
        "critical_reflection_evidence": critical_reflection_evidence,
        "critical_angle_deg": _json_float(float(critical["critical_angle_deg"])) if critical else None,
        "n_critical_estimate": _json_float(float(critical["n_critical"])) if critical else None,
        "n_estimates_combined": all_estimates,
        "n_combined_mean": _json_float(mean_n),
        "coefficient_of_variation": _json_float(cv),
        "quality_warnings": diagnostic_failures,
        "measurement_semantics": "retain_any_measurable_snell_pair; strict_physics_requires_three_snell_pairs_and_verified_total_reflection",
    }

    quality = (
        5.0 * min(len(pairs), 4)
        + (5.0 if critical_reflection_verified else 0.0)
        + min(interface.laser_crossings, 6)
        - 1.0 * len(diagnostic_failures)
        - 4.0 * len(failures)
        - min(interface.residual_px, 20.0)
    )
    return FrameAnalysis(not failures, failures, measurements, overlay, laser_mask, quality)


def _sample_indices(frame_count: int, maximum: int = 9) -> list[int]:
    if frame_count <= maximum:
        return list(range(frame_count))
    return sorted(set(int(round(v)) for v in np.linspace(0, frame_count - 1, maximum)))


def _result(
    sample_id: int | str,
    success: bool,
    measurements: dict[str, Any],
    metrics: dict[str, float],
    failure_reasons: Sequence[dict[str, Any]],
    debug_artifacts: Sequence[str],
) -> dict[str, Any]:
    primary = None if success else (failure_reasons[0]["code"] if failure_reasons else "unknown_extraction_failure")
    return {
        "task_id": TASK_ID,
        "sample_id": sample_id,
        "extract_success": bool(success),
        "measurements": measurements,
        # Partial frame geometry is diagnostic evidence, not a score.  Null
        # metrics make that distinction explicit to downstream CSV readers.
        "metrics": metrics if success else {key: None for key in metrics},
        "failure_reason": primary,
        "failure_reasons": list(failure_reasons),
        "debug_artifacts": list(debug_artifacts),
    }


def evaluate_frames(
    frames: Sequence[np.ndarray],
    fps: float,
    debug_dir: str | Path | None = None,
    sample_id: int | str = 0,
) -> dict[str, Any]:
    """Evaluate decoded BGR frames.

    Multiple frames are inspected, then the clearest complete geometry is used.
    Temporal consistency is still reported across all independently successful
    frames, so a video cannot hide changing optical laws behind one good frame.
    """

    metric_template = {
        "M1": 0.0,
        "coefficient_of_variation": 0.0,
        "per_ray_snell_residual_rmse": 0.0,
        "interface_line_fit_residual": 0.0,
        "n_water_relative_error": 0.0,
        "critical_angle_absolute_error_deg": 0.0,
        "temporal_n_coefficient_of_variation": 0.0,
    }
    if not frames:
        reasons = [_failure("empty_video", "The video decoded to zero frames.")]
        return _result(sample_id, False, {"fps": float(fps), "frame_count": 0}, metric_template, reasons, [])

    indices = _sample_indices(len(frames))
    analyses: list[tuple[int, FrameAnalysis]] = []
    artifacts: list[str] = []
    out_dir = Path(debug_dir) if debug_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        analysis = analyze_frame(frames[index])
        analyses.append((index, analysis))
        if out_dir is not None:
            overlay_path = out_dir / f"frame_{index:04d}_p12_overlay.png"
            mask_path = out_dir / f"frame_{index:04d}_p12_laser_mask.png"
            cv2.imwrite(str(overlay_path), analysis.overlay)
            cv2.imwrite(str(mask_path), analysis.laser_mask)
            artifacts.extend([str(overlay_path), str(mask_path)])

    stage_boundary = len(frames) / 2.0
    temporal_frame_measurements: list[dict[str, Any]] = []
    for index, analysis in analyses:
        frame_measurements = analysis.measurements
        temporal_frame_measurements.append(
            {
                "frame_index": int(index),
                "time_s": float(index / fps) if fps > 0 else None,
                "prompt_stage": "stage_one_complete_branches" if index < stage_boundary else "stage_two_angle_sweep",
                "extract_success": bool(analysis.success),
                "failure_codes": [item["code"] for item in analysis.failure_reasons],
                "interface": frame_measurements.get("interface"),
                "air_ray_count": frame_measurements.get("air_ray_count"),
                "water_ray_count": frame_measurements.get("water_ray_count"),
                "air_rays": frame_measurements.get("air_rays", []),
                "water_rays": frame_measurements.get("water_rays", []),
                "snell_pair_count": frame_measurements.get("snell_pair_count"),
                "snell_pairs": frame_measurements.get("snell_pairs", []),
                "n_snell_estimates": frame_measurements.get("n_snell_estimates", []),
                "n_critical_estimate": frame_measurements.get("n_critical_estimate"),
                "critical_ray": frame_measurements.get("critical_ray"),
                "critical_reflection_verified": bool(
                    frame_measurements.get("critical_reflection_verified", False)
                ),
                "critical_reflection_evidence": frame_measurements.get(
                    "critical_reflection_evidence", {}
                ),
                "coefficient_of_variation": frame_measurements.get("coefficient_of_variation"),
            }
        )
    critical_verified_indices = [
        int(item["frame_index"])
        for item in temporal_frame_measurements
        if item.get("critical_reflection_verified")
    ]
    critical_complete_indices = [
        int(item["frame_index"])
        for item in temporal_frame_measurements
        if item.get("critical_reflection_verified")
        and int(item.get("snell_pair_count") or 0) >= MIN_SNELL_PAIRS
        and item.get("extract_success")
    ]
    critical_stage_counts = {
        "stage_one_complete_branches": sum(
            item.get("prompt_stage") == "stage_one_complete_branches"
            and item.get("critical_reflection_verified")
            for item in temporal_frame_measurements
        ),
        "stage_two_angle_sweep": sum(
            item.get("prompt_stage") == "stage_two_angle_sweep"
            and item.get("critical_reflection_verified")
            for item in temporal_frame_measurements
        ),
    }
    stage_measurement_summary = {
        label: {
            "sampled_frames": sum(item["prompt_stage"] == label for item in temporal_frame_measurements),
            "extractable_frames": sum(
                item["prompt_stage"] == label and item["extract_success"]
                for item in temporal_frame_measurements
            ),
            "measured_snell_pairs": sum(
                int(item.get("snell_pair_count") or 0)
                for item in temporal_frame_measurements
                if item["prompt_stage"] == label
            ),
        }
        for label in ("stage_one_complete_branches", "stage_two_angle_sweep")
    }

    successful = [(index, analysis) for index, analysis in analyses if analysis.success]
    if not successful:
        # Use the best partial analysis to expose the most specific evidence.
        index, best = max(analyses, key=lambda item: item[1].quality_score)
        measurements = dict(best.measurements)
        measurements.update(
            {
                "fps": float(fps),
                "frame_count": len(frames),
                "analyzed_frame_indices": indices,
                "selected_frame_index": index,
                "video_prompt_protocol": "two_stage_complete_then_sweep",
                "stage_measurement_summary": stage_measurement_summary,
                "temporal_frame_measurements": temporal_frame_measurements,
                "critical_reflection_verified": bool(critical_complete_indices),
                "critical_reflection_verified_frame_indices": critical_verified_indices,
                "critical_reflection_complete_frame_indices": critical_complete_indices,
                "critical_reflection_verified_stage_counts": critical_stage_counts,
            }
        )
        return _result(sample_id, False, measurements, metric_template, best.failure_reasons, artifacts)

    # Prefer a frame that contains the complete declared optical scene (three
    # Snell pairs *and* a verified TIR pair) whenever one was sampled.  This
    # keeps the M1 estimate and the strict-premise evidence on the same frame;
    # otherwise a visually clean refraction-only frame could be selected while
    # a different transient frame supplied an unrelated reflection candidate.
    complete_reflection_successful = [
        (index, analysis)
        for index, analysis in successful
        if analysis.measurements.get("critical_reflection_verified")
        and int(analysis.measurements.get("snell_pair_count") or 0) >= MIN_SNELL_PAIRS
    ]
    selection_pool = complete_reflection_successful or successful
    selected_index, best = max(selection_pool, key=lambda item: item[1].quality_score)
    temporal_n = [
        float(item.measurements["n_combined_mean"])
        for _, item in successful
        if item.measurements.get("n_combined_mean") is not None
        and math.isfinite(float(item.measurements["n_combined_mean"]))
    ]
    temporal_mean = float(np.mean(temporal_n)) if temporal_n else float("nan")
    temporal_cv = float(np.std(temporal_n) / temporal_mean) if len(temporal_n) >= 2 and temporal_mean else None
    cv_raw = best.measurements.get("coefficient_of_variation")
    cv = float(cv_raw) if cv_raw is not None and math.isfinite(float(cv_raw)) else None
    residuals = np.asarray(best.measurements["per_ray_snell_residual"], dtype=float)
    residual_rmse = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0
    interface_residual = float(best.measurements["interface"]["residual_px"])
    n_mean_raw = best.measurements.get("n_combined_mean")
    n_mean = float(n_mean_raw) if n_mean_raw is not None and math.isfinite(float(n_mean_raw)) else float("nan")
    critical_angle_raw = best.measurements.get("critical_angle_deg")
    critical_angle = float(critical_angle_raw) if critical_angle_raw is not None and math.isfinite(float(critical_angle_raw)) else None
    measurements = dict(best.measurements)
    measurements.update(
        {
            "fps": float(fps),
            "frame_count": len(frames),
            "analyzed_frame_indices": indices,
            "successful_frame_indices": [index for index, _ in successful],
            "selected_frame_index": selected_index,
            "selection_pool": (
                "complete_snell_and_tir"
                if complete_reflection_successful
                else "best_extractable_frame"
            ),
            "temporal_n_combined_means": temporal_n,
            "video_prompt_protocol": "two_stage_complete_then_sweep",
            "stage_measurement_summary": stage_measurement_summary,
            "temporal_frame_measurements": temporal_frame_measurements,
            "critical_reflection_verified": bool(critical_complete_indices),
            "critical_reflection_verified_frame_indices": critical_verified_indices,
            "critical_reflection_complete_frame_indices": critical_complete_indices,
            "critical_reflection_verified_stage_counts": critical_stage_counts,
        }
    )
    metrics = {
        "M1": cv,
        "coefficient_of_variation": cv,
        "per_ray_snell_residual_rmse": residual_rmse,
        "interface_line_fit_residual": interface_residual / max(1.0, float(frames[selected_index].shape[0])),
        "n_water_relative_error": abs(n_mean / EXPECTED_N_WATER - 1.0) if math.isfinite(n_mean) else None,
        "critical_angle_absolute_error_deg": abs(critical_angle - EXPECTED_CRITICAL_DEG) if critical_angle is not None else None,
        "temporal_n_coefficient_of_variation": temporal_cv,
    }
    if out_dir is not None:
        summary_path = out_dir / "p12_summary_overlay.png"
        cv2.imwrite(str(summary_path), best.overlay)
        artifacts.append(str(summary_path))
    return _result(sample_id, True, measurements, metrics, [], artifacts)


def _read_video(video_path: str | Path) -> tuple[float, list[np.ndarray]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0, []
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return fps, frames


def evaluate(
    frames: Sequence[np.ndarray],
    fps: float,
    debug_dir: str | Path | None,
    sample_id: int | str = 0,
) -> dict[str, Any]:
    """Shared task-module contract used by the Group 7 dispatcher."""

    return evaluate_frames(frames, fps, debug_dir, sample_id)


def evaluate_p12(
    frames: Sequence[np.ndarray],
    fps: float,
    debug_dir: str | Path | None,
    sample_id: int | str = 0,
) -> dict[str, Any]:
    return evaluate_frames(frames, fps, debug_dir, sample_id)


def evaluate_video(
    video_path: str | Path,
    sample_id: int | str = 0,
    debug_dir: str | Path | None = None,
) -> dict[str, Any]:
    fps, frames = _read_video(video_path)
    return evaluate_frames(frames, fps, debug_dir, sample_id)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-id", default=0)
    parser.add_argument("--debug-dir", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = evaluate_video(args.video, args.sample_id, args.debug_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if result["extract_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
