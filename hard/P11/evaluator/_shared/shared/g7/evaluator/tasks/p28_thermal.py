"""Deterministic evaluator for P28: crushed ice versus one compact block.

Two independent vessels are localized from their first-frame walls and base.
Water-interface observations supply the same-frame normalized phase-lead
comparison; half-rise and colour-footprint curves remain diagnostics only. A complete disappearance of
every crystal is *not* required:
generated clips often end while a few translucent pieces remain, even though
the water signal is already measurable.  All classification, water-surface,
and optional disappearance events are retained as deterministic diagnostics;
Both sides must independently show visible ice shrinkage before water-level
comparisons can be scored. Translucent ice uses the local DINO/SAM2 identity tracker;
no VLM, final-frame prototype, or liquid timing supplies that ice identity.

Public entry points follow the same contract as the other task modules:
``evaluate(frames, fps, debug_dir, sample_id=0)`` and ``evaluate_video(...)``.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .p28_melting import observe_pair_melting
from .p28_surface import observe_surfaces, reliable_ice_masks, rise_event as observed_surface_rise_event
from .p28_surface_lead import compare_surface_lead, RULE_VERSION as SURFACE_LEAD_RULE
from .p28_vessels import locate_vessels


TASK_ID = "P28"
SIDES = ("left", "right")
DISAPPEARANCE_FRACTION = 0.08
WATER_MATCH_MARGIN = 5.0
WATER_MIN_COLOUR_DISTANCE = 9.0

# The revised P28 first frame uses tall, straight-walled beakers.  A liquid
# surface is therefore a horizontal interface rather than a broad tray
# footprint.  These bounds keep the deterministic line detector away from
# the rim/floor while still allowing a low initial level to rise through most
# of the lower beaker.  The area/height signals remain available as a fallback
# for clips in which a generated model does not render a clean interface.
SURFACE_SEARCH_TOP_FRACTION = 0.08
SURFACE_SEARCH_BOTTOM_FRACTION = 0.80
SURFACE_FLOOR_FRACTION = 0.82
SURFACE_EDGE_THRESHOLD = 9.0
SURFACE_MIN_SCORE = 9.0
SURFACE_INITIAL_MAX_JUMP_FRACTION = 0.28
SURFACE_TRACK_MAX_JUMP_FRACTION = 0.10
SURFACE_EARLY_WINDOW_SECONDS = 1.0


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
        # Preserve partial measurements under ``measurements`` while making
        # invalid score fields unambiguously non-numeric.  In particular, a
        # failed endpoint check must not look like M1 == 0 (a perfect ratio).
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
    materialized = [np.asarray(frame) for frame in frames if frame is not None]
    failures: list[str] = []
    if not materialized:
        return [], ["video_decode_failed"]
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        failures.append("invalid_fps")
    shape = materialized[0].shape
    if len(shape) != 3 or shape[2] != 3:
        failures.append("frames_are_not_bgr_color")
    if any(frame.shape != shape for frame in materialized):
        failures.append("inconsistent_frame_dimensions")
    # Water-surface growth is a curve measurement; it does not require two
    # complete melt/disappearance events. Keep only a small minimum so short
    # clips can still yield a partial raw curve and quality diagnostics.
    if len(materialized) < 8:
        failures.append("too_few_frames_for_surface_rise")
    return materialized, failures


def _median_frame(frames: Sequence[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def _difference_scalar(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.max(cv2.absdiff(frame, reference), axis=2).astype(np.uint8)


def _tail_noise_threshold(tail_frames: Sequence[np.ndarray], tail_reference: np.ndarray) -> tuple[int, dict[str, float]]:
    if not tail_frames:
        return 14, {"tail_noise_median": 0.0, "tail_noise_mad": 0.0, "difference_threshold": 14.0}
    values = np.concatenate([_difference_scalar(frame, tail_reference).ravel() for frame in tail_frames])
    median = float(np.median(values))
    mad = float(np.median(np.abs(values.astype(np.float32) - median)))
    p995 = float(np.percentile(values, 99.5))
    threshold = int(np.clip(max(14.0, median + 7.0 * 1.4826 * mad, 1.25 * p995), 14, 72))
    return threshold, {
        "tail_noise_median": median,
        "tail_noise_mad": mad,
        "tail_noise_p99_5": p995,
        "difference_threshold": float(threshold),
    }


def _clean_mask(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = np.zeros_like(mask)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            result[labels == label] = 255
    return result


def _localize_rois(
    initial: np.ndarray,
    tail_reference: np.ndarray,
    threshold: int,
) -> tuple[dict[str, tuple[int, int, int, int]], dict[str, np.ndarray], list[str]]:
    height, width = initial.shape[:2]
    difference = _difference_scalar(initial, tail_reference)
    midpoint = width // 2
    center_gap = max(2, int(0.025 * width))
    half_bounds = {
        "left": (int(0.02 * width), midpoint - center_gap),
        "right": (midpoint + center_gap, int(0.98 * width)),
    }
    minimum_component = max(8, int(height * width * 0.00005))
    minimum_dynamic_area = max(28, int(height * width * 0.00035))
    rois: dict[str, tuple[int, int, int, int]] = {}
    initial_masks: dict[str, np.ndarray] = {}
    failures: list[str] = []

    for side in SIDES:
        x0, x1 = half_bounds[side]
        half_mask = np.zeros((height, width), dtype=np.uint8)
        half_mask[:, x0:x1] = np.where(difference[:, x0:x1] >= threshold, 255, 0).astype(np.uint8)
        half_mask[: int(0.04 * height)] = 0
        half_mask[int(0.98 * height) :] = 0
        half_mask = _clean_mask(half_mask, minimum_component)
        ys, xs = np.nonzero(half_mask)
        if len(xs) < minimum_dynamic_area:
            failures.append(f"ice_region_not_localized_{side}")
            continue
        raw_x0, raw_x1 = int(xs.min()), int(xs.max()) + 1
        raw_y0, raw_y1 = int(ys.min()), int(ys.max()) + 1
        raw_w, raw_h = raw_x1 - raw_x0, raw_y1 - raw_y0
        # A valid measurement surface may deliberately occupy almost the
        # entire left/right half of the frame: the revised tall beaker can
        # expose a broad liquid band while side splitting keeps the samples
        # independent.
        # The old 88%-of-half width gate therefore rejected visible melting
        # as "dynamic_region_not_localized".  Side splitting already prevents
        # the two samples from merging, so only a near-full-height change is
        # evidence of unlocalized global/camera motion.  Wide water growth is
        # measurement evidence, not an extraction failure.
        if raw_h > 0.82 * height:
            failures.append(f"dynamic_region_not_localized_{side}")
            continue
        pad_x = max(8, int(0.18 * raw_w))
        pad_y = max(8, int(0.24 * raw_h))
        roi_x0 = max(x0, raw_x0 - pad_x)
        roi_x1 = min(x1, raw_x1 + pad_x)
        roi_y0 = max(0, raw_y0 - pad_y)
        roi_y1 = min(height, raw_y1 + pad_y)
        rois[side] = (roi_x0, roi_y0, roi_x1 - roi_x0, roi_y1 - roi_y0)
        initial_masks[side] = half_mask[roi_y0:roi_y1, roi_x0:roi_x1].copy()
    return rois, initial_masks, failures


def _looks_like_tall_beakers(frame: np.ndarray) -> bool:
    """Detect a tall-beaker layout from long vertical wall edges.

    The bundled P28 first frame places the intact block on the left and
    crushed ice on the right. Two long, high-contrast wall edges in each image
    half distinguish tall beakers from shallow trays, allowing role selection
    from the visible scene without task text or a learned model.
    """

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[0] < 32 or array.shape[1] < 32:
        return False
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gradient_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    # Long vertical walls produce a high fraction of horizontal-gradient
    # pixels in a single column.  Ignore a small top/bottom border where rims
    # and the bench can otherwise imitate a wall.
    height, width = gray.shape
    y0, y1 = int(round(0.08 * height)), int(round(0.94 * height))
    coverage = np.mean(gradient_x[y0:y1] >= 22.0, axis=0)
    upper_y0, upper_y1 = int(round(0.10 * height)), int(round(0.45 * height))
    upper_coverage = np.mean(gradient_x[upper_y0:upper_y1] >= 22.0, axis=0)
    midpoint = width // 2
    left_max = float(np.max(coverage[:midpoint])) if midpoint else 0.0
    right_max = float(np.max(coverage[midpoint:])) if midpoint < width else 0.0
    left_upper_max = float(np.max(upper_coverage[:midpoint])) if midpoint else 0.0
    right_upper_max = float(np.max(upper_coverage[midpoint:])) if midpoint < width else 0.0
    # The threshold is well below the ~0.5--0.7 coverage of the revised
    # renders and above the ~0.1--0.2 coverage of the former shallow trays.
    return bool(
        min(left_max, right_max) >= 0.30
        and min(left_upper_max, right_upper_max) >= 0.55
    )


def _fragmentation_features(frame_roi: np.ndarray, initial_mask: np.ndarray) -> dict[str, float]:
    area = int(np.count_nonzero(initial_mask))
    count, _, stats, _ = cv2.connectedComponentsWithStats(initial_mask, 8)
    components = [int(stats[label, cv2.CC_STAT_AREA]) for label in range(1, count) if stats[label, cv2.CC_STAT_AREA] >= 5]
    component_count = len(components)

    contours, _ = cv2.findContours(initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    compactness = perimeter * perimeter / max(4.0 * math.pi * area, 1e-9)

    distance = cv2.distanceTransform(initial_mask, cv2.DIST_L2, 5)
    if float(distance.max()) > 0:
        dilated = cv2.dilate(distance, np.ones((9, 9), np.uint8))
        peaks = ((distance >= dilated - 1e-6) & (distance >= max(1.5, 0.20 * float(distance.max())))).astype(np.uint8) * 255
        peaks = cv2.dilate(peaks, np.ones((3, 3), np.uint8))
        peak_count = max(0, cv2.connectedComponents(peaks, 8)[0] - 1)
    else:
        peak_count = 0

    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 125)
    support = cv2.dilate(initial_mask, np.ones((5, 5), np.uint8)) > 0
    edge_density = float(np.count_nonzero((edges > 0) & support) / max(area, 1))
    # Counts dominate; compactness and internal edge density break ties when
    # touching pieces form one connected silhouette.
    score = float(component_count + 0.75 * peak_count + 0.22 * compactness + 7.0 * edge_density)
    return {
        "initial_mask_area_px": float(area),
        "component_count": float(component_count),
        "distance_peak_count": float(peak_count),
        "compactness": compactness,
        "edge_density": edge_density,
        "fragmentation_score": score,
    }


def _isotonic_decreasing(values: np.ndarray) -> np.ndarray:
    """Least-squares non-increasing fit using the pool-adjacent-violators algorithm."""

    blocks: list[list[float]] = []  # [mean, weight, start, end]
    for index, value in enumerate(values.astype(float)):
        blocks.append([float(value), 1.0, float(index), float(index)])
        while len(blocks) >= 2 and blocks[-2][0] < blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            weight = left[1] + right[1]
            mean = (left[0] * left[1] + right[0] * right[1]) / weight
            blocks.append([mean, weight, left[2], right[3]])
    result = np.empty(len(values), dtype=float)
    for mean, _, start, end in blocks:
        result[int(start) : int(end) + 1] = mean
    return result


def _occupancy_curves(
    frames: Sequence[np.ndarray],
    tail_reference: np.ndarray,
    rois: dict[str, tuple[int, int, int, int]],
    threshold: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    curves: dict[str, list[float]] = {side: [] for side in SIDES}
    masks: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
    frame_area = frames[0].shape[0] * frames[0].shape[1]
    minimum_component = max(5, int(frame_area * 0.000025))
    for frame in frames:
        scalar = _difference_scalar(frame, tail_reference)
        for side in SIDES:
            x, y, w, h = rois[side]
            roi_mask = np.where(scalar[y : y + h, x : x + w] >= threshold, 255, 0).astype(np.uint8)
            roi_mask = _clean_mask(roi_mask, minimum_component)
            masks[side].append(roi_mask)
            curves[side].append(float(np.count_nonzero(roi_mask)))
    return {side: np.asarray(curves[side], dtype=float) for side in SIDES}, masks


def _isotonic_increasing(values: np.ndarray) -> np.ndarray:
    """Least-squares non-decreasing fit (pool-adjacent-violators)."""

    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()
    # The existing decreasing implementation is numerically stable and keeps
    # the provenance of every frame.  Negating gives the increasing variant.
    return -_isotonic_decreasing(-values)


def _surface_line_candidates(frame_roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate horizontal-interface rows and their edge strengths.

    The detector is deliberately low-level: it only uses a blurred grayscale
    vertical gradient and the fraction of the interior that contains an edge.
    Requiring a broad horizontal response suppresses the many short edges made
    by individual crushed-ice pieces, while excluding the side walls and the
    bottom/rim avoids treating the beaker geometry as a liquid surface.
    Rows are local to ``frame_roi`` and refer to the upper edge of the
    interface.  No semantic/object detector is involved.
    """

    array = np.asarray(frame_roi)
    if array.ndim != 3 or array.shape[0] < 12 or array.shape[1] < 12:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=float)
    height, width = array.shape[:2]
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x0 = max(2, int(round(0.08 * width)))
    x1 = min(width - 2, int(round(0.92 * width)))
    if x1 <= x0 + 4:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=float)
    # A small blur makes the score stable under compression noise and tiny
    # ripples but keeps a one-to-two-pixel water line visible.
    smoothed = cv2.GaussianBlur(gray[:, x0:x1], (0, 0), 1.2)
    gradient = np.abs(np.diff(smoothed, axis=0))
    median_gradient = np.median(gradient, axis=1)
    coverage = np.mean(gradient >= SURFACE_EDGE_THRESHOLD, axis=1)
    score = median_gradient + 20.0 * coverage
    score = np.convolve(score, np.ones(5, dtype=float) / 5.0, mode="same")

    low = max(3, int(round(SURFACE_SEARCH_TOP_FRACTION * height)))
    high = min(len(score) - 3, int(round(SURFACE_SEARCH_BOTTOM_FRACTION * height)))
    if high <= low:
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=float)
    valid = np.zeros(len(score), dtype=bool)
    valid[low:high] = True
    left = np.r_[score[0], score[:-1]]
    right = np.r_[score[1:], score[-1]]
    local_max = valid & (score >= left) & (score >= right) & (score >= SURFACE_MIN_SCORE)
    rows = np.flatnonzero(local_max).astype(np.int32)
    return rows, score[rows].astype(float)


def _track_surface_lines(
    frames: Sequence[np.ndarray],
    rois: dict[str, tuple[int, int, int, int]],
    fps: float,
) -> dict[str, dict[str, np.ndarray]]:
    """Track a plausible liquid level in each beaker using edge continuity.

    New P28 clips start dry, so a surface is initialized at a conservative
    floor row and remains there until a broad horizontal edge appears nearby.
    During the first second a larger jump is allowed because the first visible
    water band can appear below the ice silhouette; afterwards the row is
    constrained to short frame-to-frame moves.  A candidate above the floor is
    never accepted solely because it is a strong ice top edge.  The returned
    validity/strength arrays preserve exactly which frames supplied evidence.
    """

    tracked: dict[str, dict[str, np.ndarray]] = {}
    early_window = max(1, int(round(max(float(fps), 1.0) * SURFACE_EARLY_WINDOW_SECONDS)))
    for side in SIDES:
        if side not in rois:
            tracked[side] = {
                "surface_y_px": np.asarray([], dtype=float),
                "surface_edge_strength": np.asarray([], dtype=float),
                "surface_valid": np.asarray([], dtype=bool),
                "surface_floor_px": np.asarray([], dtype=float),
            }
            continue
        x, y, width, height = rois[side]
        floor = float(np.clip(round(SURFACE_FLOOR_FRACTION * height), 1, max(height - 2, 1)))
        previous = floor
        rows: list[float] = []
        strengths: list[float] = []
        valid_flags: list[bool] = []
        floor_values: list[float] = []
        for frame_index, frame in enumerate(frames):
            roi = np.asarray(frame)[y : y + height, x : x + width]
            candidates, scores = _surface_line_candidates(roi)
            chosen = previous
            strength = 0.0
            accepted = False
            if len(candidates):
                # The liquid cannot be below the beaker floor.  Prefer a
                # nearby line, with a small score reward to break ties between
                # texture edges at almost the same row.
                candidates_float = candidates.astype(float)
                valid_candidates = candidates_float <= floor
                candidates_float = candidates_float[valid_candidates]
                candidate_scores = scores[valid_candidates]
                if len(candidates_float):
                    # Acquisition is an observation state, not a one-second
                    # timer. Late-appearing water must still be detectable.
                    early = not any(valid_flags) or (len(valid_flags)>=early_window and not any(valid_flags[-early_window:]))
                    max_jump = (
                        SURFACE_INITIAL_MAX_JUMP_FRACTION * height
                        if early
                        else SURFACE_TRACK_MAX_JUMP_FRACTION * height
                    )
                    allowed=(np.abs(candidates_float-previous)<=max_jump)&(candidate_scores>=SURFACE_MIN_SCORE)
                    if early:allowed &= candidates_float>=.45*height
                    candidates_float=candidates_float[allowed];candidate_scores=candidate_scores[allowed]
                    if not len(candidates_float):
                        rows.append(float(previous));strengths.append(0.);valid_flags.append(False);floor_values.append(floor)
                        continue
                    costs = np.abs(candidates_float - previous)
                    # A downward jump is possible while a dry beaker is first
                    # acquiring a visible pool, but should not dominate later.
                    costs += np.maximum(0.0, candidates_float - previous - 0.04 * height) * 1.6
                    costs += np.maximum(0.0, previous - candidates_float - 0.14 * height) * 2.0
                    costs -= np.clip(candidate_scores - SURFACE_MIN_SCORE, 0.0, 25.0) * 1.8
                    index = int(np.argmin(costs))
                    candidate = float(candidates_float[index])
                    candidate_strength = float(candidate_scores[index])
                    # During the initial dry period reject a high ice-top edge
                    # that is implausibly far above the floor.  Once a line has
                    # been acquired, continuity is the stronger safeguard.
                    too_high_initial = early and candidate < 0.45 * height
                    if (
                        not too_high_initial
                        and abs(candidate - previous) <= max_jump
                        and candidate_strength >= SURFACE_MIN_SCORE
                    ):
                        chosen = candidate
                        strength = candidate_strength
                        accepted = True
            rows.append(float(chosen))
            strengths.append(float(strength))
            valid_flags.append(bool(accepted))
            floor_values.append(floor)
            previous = chosen
        tracked[side] = {
            "surface_y_px": np.asarray(rows, dtype=float),
            "surface_edge_strength": np.asarray(strengths, dtype=float),
            "surface_valid": np.asarray(valid_flags, dtype=bool),
            "surface_floor_px": np.asarray(floor_values, dtype=float),
        }
    return tracked


def _water_surface_curves(
    frames: Sequence[np.ndarray],
    initial_reference: np.ndarray,
    tail_reference: np.ndarray,
    rois: dict[str, tuple[int, int, int, int]],
    fps: float = 24.0,
    ice_masks: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[np.ndarray]]]:
    """Measure growth of the liquid surface using initial/tail colour anchors.

    The final reference contains the liquid pool (and the empty beaker),
    whereas the initial reference contains the ice.  A pixel is considered water-like
    when it is materially closer in Lab colour to the tail than to the initial
    frame.  This is deliberately a soft, appearance-agnostic proxy: it works
    for clear water, tinted water, and wet reflections without assuming a
    particular RGB value.  Only the lower surface-bearing part of each ROI is
    used, avoiding the beaker rim/floor and most of the original ice silhouette.
    """

    curves: dict[str, dict[str, list[float]]] = {
        side: {
            "area_px": [],
            "height_px": [],
            "centroid_y": [],
        }
        for side in SIDES
    }
    masks: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
    for frame in frames:
        for side in SIDES:
            x, y, w, h = rois[side]
            current = frame[y : y + h, x : x + w]
            initial = initial_reference[y : y + h, x : x + w]
            tail = tail_reference[y : y + h, x : x + w]
            lab_current = cv2.cvtColor(current, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab_initial = cv2.cvtColor(initial, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab_tail = cv2.cvtColor(tail, cv2.COLOR_BGR2LAB).astype(np.float32)
            d_initial = np.linalg.norm(lab_current - lab_initial, axis=2)
            d_tail = np.linalg.norm(lab_current - lab_tail, axis=2)
            yy = np.arange(h, dtype=np.int32)[:, None]
            surface_band = yy >= int(round(0.25 * h))
            candidate = (
                surface_band
                & (d_initial >= WATER_MIN_COLOUR_DISTANCE)
                & (d_tail + WATER_MATCH_MARGIN < d_initial)
                & (d_tail <= 58.0)
            )
            mask = _clean_mask(candidate.astype(np.uint8) * 255, max(4, int(0.00001 * w * h)))
            masks[side].append(mask)
            active = mask > 0
            area = float(np.count_nonzero(active))
            curves[side]["area_px"].append(area)
            if area:
                rows = np.flatnonzero(np.any(active, axis=1))
                curves[side]["height_px"].append(float(rows[-1] - rows[0] + 1) if len(rows) else 0.0)
                # ``flatnonzero`` is row-major with a stride of ``w``; using
                # modulo ``h`` would accidentally report an x-like coordinate.
                # ``where`` keeps the actual vertical row coordinate.
                curves[side]["centroid_y"].append(float(np.mean(np.where(active)[0])))
            else:
                curves[side]["height_px"].append(0.0)
                curves[side]["centroid_y"].append(float("nan"))
    # For tall straight-walled beakers, also retain a direct horizontal-level
    # signal.  It is computed independently of the colour proxy above and is
    # used as the preferred M1 cue when both sides expose a stable line.
    line_curves = observe_surfaces(frames, rois, float(fps), ice_masks=ice_masks)
    result: dict[str, dict[str, np.ndarray]] = {}
    for side, series in curves.items():
        item = {name: np.asarray(values, dtype=float) for name, values in series.items()}
        item.update(line_curves.get(side, {}))
        result[side] = item
    return result, masks


def _water_rise_event(series: dict[str, np.ndarray], fps: float, signal_family: str | None = None) -> dict[str, Any]:
    """Return a soft half-rise time/rate without a complete-melt gate."""

    if series.get('independent_surface_observation'):
        return observed_surface_rise_event(series,fps)

    area = np.asarray(series.get("area_px", []), dtype=float)
    height = np.asarray(series.get("height_px", []), dtype=float)
    surface_y = np.asarray(series.get("surface_y_px", []), dtype=float)
    surface_valid = np.asarray(series.get("surface_valid", []), dtype=bool)
    n = len(area)
    if n == 0:
        return {
            "usable": False,
            "signal_name": None,
            "raw_area_px": area,
            "raw_height_px": height,
            "raw_surface_y_px": surface_y,
            "surface_valid": surface_valid,
            "surface_rise_px": np.asarray([], dtype=float),
            "surface_line_usable": False,
            "surface_line_valid_fraction": 0.0,
            "normalized_surface_rise": np.asarray([], dtype=float),
            "isotonic_surface_rise": np.asarray([], dtype=float),
            "dynamic_range_px": 0.0,
            "half_rise_frame": None,
            "half_rise_time_s": None,
            "rise_rate_per_s": None,
            "quality_flags": ["water_surface_curve_empty"],
        }
    start_count = max(3, min(n // 8, int(math.ceil(0.35 * fps))))
    tail_count = max(3, min(n // 8, int(math.ceil(0.35 * fps))))
    candidates: list[tuple[str, np.ndarray, float]] = []
    for name, values in (("area", area), ("height", height)):
        baseline = float(np.median(values[:start_count]))
        tail = float(np.median(values[-tail_count:]))
        dynamic = tail - baseline
        candidates.append((name, values, dynamic))

    # A direct tracked interface is more physically interpretable for the new
    # tall-beaker frame than the broad colour footprint.  Require evidence on a
    # meaningful fraction of frames and a positive upward movement; otherwise
    # retain the colour-area/extent fallback used by older tray-style clips.
    surface_line_usable = False
    surface_line_valid_fraction = 0.0
    surface_rise = np.asarray([], dtype=float)
    surface_baseline = None
    surface_tail = None
    surface_dynamic = 0.0
    if len(surface_y) == n:
        if len(surface_valid) != n:
            surface_valid = np.isfinite(surface_y)
        # Carried floor/last-known coordinates are not observations.
        finite = np.isfinite(surface_y) & surface_valid
        surface_line_valid_fraction = float(np.mean(surface_valid & finite)) if n else 0.0
        if np.any(finite):
            # The tracker normally supplies a finite floor value for every
            # frame.  Interpolate defensively so a future detector revision can
            # leave genuinely missing rows without breaking the raw curve.
            indexes = np.arange(n)
            filled = np.where(finite,surface_y,np.nan)
            if not np.all(finite):
                good = indexes[finite]
                if len(good) >= 2:
                    internal=(indexes>=good[0])&(indexes<=good[-1])&~finite
                    filled[internal] = np.interp(indexes[internal], good, surface_y[finite])
            good=np.flatnonzero(finite)
            surface_baseline = float(np.median(surface_y[good[:start_count]]))
            surface_tail = float(np.median(surface_y[good[-tail_count:]]))
            surface_dynamic = float(surface_baseline - surface_tail)
            surface_rise = surface_baseline - filled
            # Eight pixels is a conservative line displacement at the native
            # 768/1024-pixel render sizes; the relative term scales to resized
            # clips.  No disappearance/completion requirement is imposed.
            surface_line_usable = bool(
                surface_line_valid_fraction >= 0.50 and good[-1]>=n-tail_count
                and surface_dynamic >= 8.0
            )

    if surface_line_usable and signal_family in (None,"surface_line"):
        signal_name = "surface_line"
        signal = surface_rise
        dynamic = surface_dynamic
    else:
        # Prefer projected area (the most reliable cue on a flat tray), then
        # use vertical extent if perspective makes the area nearly constant.
        eligible=[c for c in candidates if signal_family in (None,c[0])]
        if signal_family=="surface_line":eligible=[("surface_line",surface_rise,surface_dynamic)]
        # Compare signal-to-resolution, not pixel areas directly to lengths.
        signal_name, signal, dynamic = max(eligible, key=lambda item: item[2]/max(12.0,.06*abs(float(np.median(item[1][-tail_count:])))))
    if signal_name=="surface_line" and len(signal)!=n:signal=np.full(n,np.nan)
    finite_signal=np.isfinite(signal)
    observed_values=signal[finite_signal]
    baseline = float(np.median(observed_values[:start_count])) if len(observed_values) else 0.
    tail = float(np.median(observed_values[-tail_count:])) if len(observed_values) else 0.
    dynamic = float(tail - baseline)
    flags: list[str] = []
    minimum_dynamic=8.0 if signal_name=="surface_line" else max(12.0,0.06*max(abs(tail),1.0))
    usable = bool(dynamic >= minimum_dynamic and (signal_name!="surface_line" or surface_line_usable))
    if not usable:
        flags.append("water_surface_dynamic_range_small")
    normalized = np.clip((signal - baseline) / max(dynamic, 1e-9), -1.0, 2.0)
    isotonic=np.full(n,np.nan)
    if finite_signal.any():isotonic[finite_signal]=_isotonic_increasing(normalized[finite_signal])
    half_frame: int | None = None
    if usable:
        threshold = 0.50
        for index in range(max(1, int(0.04 * n)), n):
            if isotonic[index] >= threshold:
                half_frame = index
                break
    if half_frame is None and usable:
        flags.append("water_surface_half_rise_not_observed")
    duration = max((n - 1) / max(float(fps), 1e-9), 1e-9)
    return {
        "usable": usable,
        "signal_name": signal_name,
        "raw_area_px": area,
        "raw_height_px": height,
        "raw_surface_y_px": surface_y,
        "surface_valid": surface_valid,
        "surface_rise_px": surface_rise,
        "surface_line_usable": surface_line_usable,
        "surface_line_valid_fraction": surface_line_valid_fraction,
        "surface_line_baseline_y_px": surface_baseline,
        "surface_line_tail_y_px": surface_tail,
        "surface_line_dynamic_range_px": surface_dynamic,
        "baseline_px": baseline,
        "tail_px": tail,
        "dynamic_range_px": dynamic,
        "minimum_dynamic_range":minimum_dynamic,
        "half_rise_definition":"half of this observed signal change; not half melted mass",
        "normalized_surface_rise": normalized,
        "isotonic_surface_rise": isotonic,
        "half_rise_frame": half_frame,
        "half_rise_time_s": None if half_frame is None else float(half_frame / fps),
        "rise_rate_per_s": float(dynamic / duration) if usable else None,
        "quality_flags": flags,
    }


def _curve_event(curve: np.ndarray, fps: float) -> tuple[dict[str, Any], list[str]]:
    start_count = max(3, min(len(curve) // 8, int(math.ceil(0.35 * fps))))
    tail_count = max(3, min(len(curve) // 8, int(math.ceil(0.35 * fps))))
    initial_level = float(np.percentile(curve[:start_count], 75))
    tail_level = float(np.median(curve[-tail_count:]))
    dynamic_range = initial_level - tail_level
    failures: list[str] = []
    if initial_level <= 0 or dynamic_range < max(20.0, 0.20 * initial_level):
        failures.append("ice_occupancy_dynamic_range_too_small")
    normalized = np.clip((curve - tail_level) / max(dynamic_range, 1e-9), 0.0, 2.0)
    isotonic = _isotonic_decreasing(normalized)
    persistence = max(3, int(math.ceil(0.30 * fps)))
    event_index: int | None = None
    for index in range(max(1, int(0.05 * len(curve))), len(curve) - persistence + 1):
        window = isotonic[index : index + persistence]
        if np.all(window <= DISAPPEARANCE_FRACTION):
            event_index = index
            break
    if event_index is None:
        failures.append("persistent_ice_disappearance_not_observed")
    elif event_index >= len(curve) - tail_count:
        # A crossing that appears only inside the reference tail is weak evidence:
        # it may merely be the median-reference construction rather than melting.
        failures.append("disappearance_only_observed_in_reference_tail")
    upward_steps = float(np.mean(np.diff(normalized) > 0.08)) if len(normalized) > 1 else 1.0
    return {
        "raw_area_px": curve,
        "normalized_area": normalized,
        "isotonic_normalized_area": isotonic,
        "initial_level_px": initial_level,
        "tail_level_px": tail_level,
        "dynamic_range_px": dynamic_range,
        "persistence_frames": persistence,
        "disappearance_frame": event_index,
        "disappearance_time_s": None if event_index is None else float(event_index / fps),
        "large_upward_step_fraction": upward_steps,
    }, failures


def _prototype_residual_fraction(
    initial_roi: np.ndarray,
    tail_roi: np.ndarray,
    initial_mask: np.ndarray,
) -> dict[str, float]:
    """Detect a coherent late component that still resembles initial ice."""

    positive = cv2.cvtColor(initial_roi, cv2.COLOR_BGR2LAB)[initial_mask > 0]
    negative = cv2.cvtColor(tail_roi, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    if len(positive) < 12 or len(negative) < 12:
        return {"prototype_separation": 0.0, "late_ice_like_fraction": 0.0, "classifier_usable": 0.0}
    positive_center = np.median(positive.astype(float), axis=0)
    negative_center = np.median(negative.astype(float), axis=0)
    separation = float(np.linalg.norm(positive_center - negative_center))
    tail_lab = cv2.cvtColor(tail_roi, cv2.COLOR_BGR2LAB).astype(float)
    d_positive = np.linalg.norm(tail_lab - positive_center, axis=2)
    d_negative = np.linalg.norm(tail_lab - negative_center, axis=2)
    similar = ((d_positive + 4.0 < 0.72 * d_negative) & (d_positive < 34.0)).astype(np.uint8) * 255
    similar = _clean_mask(similar, max(5, int(0.015 * np.count_nonzero(initial_mask))))
    fraction = float(np.count_nonzero(similar) / max(np.count_nonzero(initial_mask), 1))
    return {
        "prototype_separation": separation,
        "late_ice_like_fraction": fraction,
        "classifier_usable": float(separation >= 10.0),
    }


def _write_area_csv(
    path: Path,
    fps: float,
    events: dict[str, dict[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "time_s",
                "left_area_px",
                "left_normalized",
                "left_isotonic",
                "right_area_px",
                "right_normalized",
                "right_isotonic",
            ]
        )
        length = len(events["left"]["raw_area_px"])
        for index in range(length):
            writer.writerow(
                [
                    index,
                    index / fps,
                    events["left"]["raw_area_px"][index],
                    events["left"]["normalized_area"][index],
                    events["left"]["isotonic_normalized_area"][index],
                    events["right"]["raw_area_px"][index],
                    events["right"]["normalized_area"][index],
                    events["right"]["isotonic_normalized_area"][index],
                ]
            )


def _write_curve_plot(path: Path, fps: float, events: dict[str, dict[str, Any]]) -> None:
    width, height = 900, 510
    left_margin, right_margin, top_margin, bottom_margin = 70, 25, 30, 60
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    length = len(events["left"]["normalized_area"])

    def point(index: int, value: float) -> tuple[int, int]:
        x = left_margin + index / max(length - 1, 1) * (width - left_margin - right_margin)
        y = top_margin + (1.25 - float(np.clip(value, 0, 1.25))) / 1.25 * (height - top_margin - bottom_margin)
        return int(round(x)), int(round(y))

    cv2.line(canvas, (left_margin, top_margin), (left_margin, height - bottom_margin), (30, 30, 30), 2)
    cv2.line(canvas, (left_margin, height - bottom_margin), (width - right_margin, height - bottom_margin), (30, 30, 30), 2)
    threshold_y = point(0, DISAPPEARANCE_FRACTION)[1]
    cv2.line(canvas, (left_margin, threshold_y), (width - right_margin, threshold_y), (120, 120, 120), 1)
    colours = {"left": (210, 90, 30), "right": (40, 150, 40)}
    for side in SIDES:
        mapped = np.asarray(
            [point(index, value) for index, value in enumerate(events[side]["isotonic_normalized_area"])],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [mapped], False, colours[side], 3, cv2.LINE_AA)
        event = events[side]["disappearance_frame"]
        if event is not None:
            cv2.circle(canvas, point(int(event), float(events[side]["isotonic_normalized_area"][event])), 7, colours[side], -1)
    duration = (length - 1) / fps
    cv2.putText(canvas, f"time (s), duration={duration:.2f}", (width // 2 - 115, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.putText(canvas, "normalized visible-ice occupancy", (82, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.putText(canvas, "left", (width - 175, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colours["left"], 2)
    cv2.putText(canvas, "right", (width - 95, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colours["right"], 2)
    cv2.imwrite(str(path), canvas)


def _write_water_plot(path: Path, fps: float, water_events: dict[str, dict[str, Any]]) -> None:
    """Write the primary water-surface comparison used by P28 M1."""

    width, height = 900, 510
    left_margin, right_margin, top_margin, bottom_margin = 70, 25, 30, 60
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    length = max((len(item.get("isotonic_surface_rise", [])) for item in water_events.values()), default=1)

    def point(index: int, value: float) -> tuple[int, int]:
        x = left_margin + index / max(length - 1, 1) * (width - left_margin - right_margin)
        y = top_margin + (1.10 - float(np.clip(value, -0.05, 1.10))) / 1.15 * (height - top_margin - bottom_margin)
        return int(round(x)), int(round(y))

    cv2.line(canvas, (left_margin, top_margin), (left_margin, height - bottom_margin), (30, 30, 30), 2)
    cv2.line(canvas, (left_margin, height - bottom_margin), (width - right_margin, height - bottom_margin), (30, 30, 30), 2)
    cv2.line(canvas, point(0, 0.5), point(length - 1, 0.5), (150, 150, 150), 1)
    colours = {"left": (210, 90, 30), "right": (40, 150, 40)}
    for side in SIDES:
        values = np.asarray(water_events.get(side, {}).get("isotonic_surface_rise", []), dtype=float)
        if len(values):
            good=np.flatnonzero(np.isfinite(values))
            for run in np.split(good,np.flatnonzero(np.diff(good)>1)+1):
                if len(run)<2:continue
                mapped=np.asarray([point(int(index),values[index]) for index in run],dtype=np.int32)
                cv2.polylines(canvas,[mapped],False,colours[side],3,cv2.LINE_AA)
        event = water_events.get(side, {}).get("half_rise_frame")
        if event is not None and len(values):
            cv2.circle(canvas, point(int(event), float(values[int(event)])), 7, colours[side], -1)
    duration = (length - 1) / max(float(fps), 1e-9)
    cv2.putText(canvas, f"time (s), duration={duration:.2f}", (width // 2 - 115, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.putText(canvas, "normalized water-surface rise (proxy)", (82, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    cv2.putText(canvas, "left", (width - 175, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colours["left"], 2)
    cv2.putText(canvas, "right", (width - 95, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colours["right"], 2)
    cv2.imwrite(str(path), canvas)


def _write_water_csv(path: Path, fps: float, water_events: dict[str, dict[str, Any]]) -> None:
    """Persist raw and monotonic water-surface signals for audit/review."""

    length = max((len(item.get("raw_area_px", [])) for item in water_events.values()), default=0)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "frame", "time_s", "left_water_area_px", "left_water_height_px",
            "left_surface_rise", "left_surface_y_px", "left_surface_valid",
            "right_water_area_px", "right_water_height_px", "right_surface_rise",
            "right_surface_y_px", "right_surface_valid",
        ])
        for index in range(length):
            row = [index, index / max(float(fps), 1e-9)]
            for side in SIDES:
                item = water_events.get(side, {})
                for key in (
                    "raw_area_px",
                    "raw_height_px",
                    "normalized_surface_rise",
                    "raw_surface_y_px",
                    "surface_valid",
                ):
                    values = item.get(key, [])
                    row.append(values[index] if index < len(values) else None)
            # The loop writes left area/height/rise/line then right area/height/
            # rise/line, matching the header above.
            writer.writerow(row)


def _write_segmentation_debug(
    debug_dir: Path,
    frames: Sequence[np.ndarray],
    rois: dict[str, tuple[int, int, int, int]],
    masks: dict[str, list[np.ndarray]],
    events: dict[str, dict[str, Any]],
    classification: dict[str, Any],
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    initial_overlay = frames[0].copy()
    colours = {"left": (255, 100, 20), "right": (30, 200, 50)}
    for side in SIDES:
        x, y, w, h = rois[side]
        mask = masks[side][0]
        colour_layer = np.zeros_like(initial_overlay[y : y + h, x : x + w])
        colour_layer[:] = colours[side]
        active = mask > 0
        roi = initial_overlay[y : y + h, x : x + w]
        roi[active] = cv2.addWeighted(roi[active], 0.45, colour_layer[active], 0.55, 0)
        cv2.rectangle(initial_overlay, (x, y), (x + w, y + h), colours[side], 2)
        label = f"{side}: {classification.get(side + '_role', '?')}"
        cv2.putText(initial_overlay, label, (x + 4, max(22, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, colours[side], 2, cv2.LINE_AA)
    initial_path = debug_dir / "initial_roi_segmentation.png"
    cv2.imwrite(str(initial_path), initial_overlay)
    artifacts["initial_roi_segmentation"] = str(initial_path)

    chosen = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
    tiles: list[np.ndarray] = []
    source_h, source_w = frames[0].shape[:2]
    for frame_index in chosen:
        overlay = frames[int(frame_index)].copy()
        for side in SIDES:
            x, y, w, h = rois[side]
            mask = masks[side][int(frame_index)]
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            shifted = [contour + np.array([[[x, y]]], dtype=np.int32) for contour in contours]
            cv2.drawContours(overlay, shifted, -1, colours[side], 2)
            normalized = events[side]["normalized_area"][int(frame_index)]
            cv2.putText(overlay, f"{side} {normalized:.2f}", (x + 3, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colours[side], 2, cv2.LINE_AA)
        cv2.putText(overlay, f"frame {int(frame_index)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tile_w = 420
        tile_h = max(1, int(round(source_h * tile_w / source_w)))
        tiles.append(cv2.resize(overlay, (tile_w, tile_h), interpolation=cv2.INTER_AREA))
    columns = 2
    rows = math.ceil(len(tiles) / columns)
    tile_h, tile_w = tiles[0].shape[:2]
    sheet = np.zeros((rows * tile_h, columns * tile_w, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[row * tile_h : (row + 1) * tile_h, column * tile_w : (column + 1) * tile_w] = tile
    contact_path = debug_dir / "melting_contact_sheet.png"
    cv2.imwrite(str(contact_path), sheet)
    artifacts["melting_contact_sheet"] = str(contact_path)
    return artifacts


def _write_water_line_debug(
    debug_dir: Path,
    frames: Sequence[np.ndarray],
    rois: dict[str, tuple[int, int, int, int]],
    water_events: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Render a compact audit sheet showing the tracked water interfaces."""

    if not frames or any(side not in rois for side in SIDES):
        return {}
    chosen = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
    colours = {"left": (255, 90, 20), "right": (40, 220, 70)}
    tiles: list[np.ndarray] = []
    source_h, source_w = frames[0].shape[:2]
    for frame_index in chosen:
        overlay = np.asarray(frames[int(frame_index)]).copy()
        for side in SIDES:
            x, y, w, h = rois[side]
            event = water_events.get(side, {})
            rows = np.asarray(event.get("raw_surface_y_px", []), dtype=float)
            if int(frame_index) >= len(rows) or not math.isfinite(float(rows[int(frame_index)])):
                continue
            yy = int(round(y + float(rows[int(frame_index)])))
            xx0 = x + int(round(0.08 * w))
            xx1 = x + int(round(0.92 * w))
            valid = np.asarray(event.get("surface_valid", []), dtype=bool)
            is_valid = int(frame_index) < len(valid) and bool(valid[int(frame_index)])
            colour = colours[side] if is_valid else (130, 130, 130)
            cv2.line(overlay, (xx0, yy), (xx1, yy), colour, 3, cv2.LINE_AA)
            cv2.putText(
                overlay,
                f"{side} y={yy}{'' if is_valid else ' (held)'}",
                (xx0 + 3, max(20, yy - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                colour,
                2,
                cv2.LINE_AA,
            )
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
        tile_w = 420
        tile_h = max(1, int(round(source_h * tile_w / source_w)))
        tiles.append(cv2.resize(overlay, (tile_w, tile_h), interpolation=cv2.INTER_AREA))
    if not tiles:
        return {}
    columns = 2
    rows = math.ceil(len(tiles) / columns)
    tile_h, tile_w = tiles[0].shape[:2]
    sheet = np.zeros((rows * tile_h, columns * tile_w, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[row * tile_h : (row + 1) * tile_h, column * tile_w : (column + 1) * tile_w] = tile
    path = debug_dir / "water_surface_contact_sheet.png"
    cv2.imwrite(str(path), sheet)
    return {"water_surface_contact_sheet": str(path)}


def _write_summary(debug_dir: Path, result: dict[str, Any]) -> str:
    path = debug_dir / "p28_summary.json"
    path.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def evaluate(
    frames: Sequence[np.ndarray],
    fps: float,
    debug_dir: str | Path,
    sample_id: int = 0,
) -> dict[str, Any]:
    """Evaluate decoded BGR frames for P28."""

    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)
    materialized, failures = _validate_frames(frames, fps)
    if failures:
        result = _result(sample_id, False, {"frame_count": len(materialized), "fps": float(fps)}, failure_reasons=failures)
        result["debug_artifacts"]["summary"] = _write_summary(debug_path, result)
        return result

    # A changing liquid level is not evidence that ice melted. Establish both
    # initial ice identities and observed shrinkage independently BEFORE the
    # tail-based water localization can turn filling/lighting into a score.
    melting, melting_failures = observe_pair_melting(materialized, debug_path)
    if melting_failures:
        artifacts = {side + "_ice_masks": observation["mask_artifact"]
                     for side, observation in melting.items() if observation.get("mask_artifact")}
        result = _result(sample_id, False,
                         {"frame_count": len(materialized), "fps": float(fps),
                          "melting_observation": melting,
                          "melting_prerequisite_passed": False},
                         failure_reasons=melting_failures, debug_artifacts=artifacts)
        result["debug_artifacts"]["summary"] = _write_summary(debug_path, result)
        return result

    fps = float(fps)
    tail_count = max(5, min(len(materialized) // 6, int(math.ceil(0.75 * fps))))
    start_count = max(3, min(len(materialized) // 10, int(math.ceil(0.35 * fps))))
    initial_reference = _median_frame(materialized[:start_count])
    tail_reference = _median_frame(materialized[-tail_count:])
    threshold, noise_diagnostics = _tail_noise_threshold(materialized[-tail_count:], tail_reference)
    # Apparatus coordinates come exclusively from observed first-frame walls
    # and their base. Later reflections cannot expand the localization region.
    rois, vessel_diagnostics = locate_vessels(materialized[0])
    localization_failures=['initial_vessel_geometry_unresolved_'+side for side in SIDES if side not in rois]
    ice_masks={side:np.load(melting[side]['mask_artifact'])['masks'] for side in SIDES}
    ice_masks,ice_mask_reliability=reliable_ice_masks(ice_masks,melting)
    initial_masks={side:ice_masks[side][0][y:y+h,x:x+w].astype(np.uint8)*255
                   for side,(x,y,w,h) in rois.items()}
    failures.extend(localization_failures)
    base_measurements: dict[str, Any] = {
        "fps": fps,
        "frame_count": len(materialized),
        "tail_reference_frames": tail_count,
        "initial_reference_frames": start_count,
        "noise_model": noise_diagnostics,
        "rois": {side: list(rois[side]) for side in rois},
        "vessel_geometry":vessel_diagnostics,
        "ice_mask_reliability":ice_mask_reliability,
        "melting_observation": melting,
        "melting_prerequisite_passed": True,
    }
    tail_path = debug_path / "tail_reference.png"
    initial_path = debug_path / "initial_reference.png"
    cv2.imwrite(str(tail_path), tail_reference)
    cv2.imwrite(str(initial_path), initial_reference)
    artifacts = {"tail_reference": str(tail_path), "initial_reference": str(initial_path)}
    if failures or any(side not in rois for side in SIDES):
        result = _result(sample_id, False, base_measurements, failure_reasons=failures, debug_artifacts=artifacts)
        result["debug_artifacts"]["summary"] = _write_summary(debug_path, result)
        return result

    features: dict[str, dict[str, float]] = {}
    residual_checks: dict[str, dict[str, float]] = {}
    for side in SIDES:
        x, y, w, h = rois[side]
        features[side] = _fragmentation_features(initial_reference[y : y + h, x : x + w], initial_masks[side])
        residual_checks[side] = _prototype_residual_fraction(
            initial_reference[y : y + h, x : x + w],
            tail_reference[y : y + h, x : x + w],
            initial_masks[side],
        )
        # Residual translucent ice is expected in short generated clips.  Keep
        # this classifier as an auditable diagnostic only; it is never an
        # extraction gate and never means that the water measurement failed.

    score_difference = features["left"]["fragmentation_score"] - features["right"]["fragmentation_score"]
    scale = max(features["left"]["fragmentation_score"], features["right"]["fragmentation_score"], 1.0)
    classification_margin = abs(score_difference) / scale
    # The revised first-frame protocol declares the spatial roles (left block,
    # right crushed).  Use a clear morphology signal when it agrees, but fall
    # back to the declared layout instead of rejecting a clip for a small
    # fragmentation margin.  This keeps role ambiguity visible in diagnostics.
    inferred_crushed_side = "left" if score_difference > 0 else "right"
    declared_crushed_side = "right"
    tall_beaker_layout = _looks_like_tall_beakers(initial_reference)
    # Spatial roles are part of the fixed P28 input protocol for every
    # vessel shape. A tail-difference mask includes rising water and cannot
    # identify ice fragmentation; keep that diagnostic without swapping roles.
    crushed_side = declared_crushed_side
    role_source = "task_declared_left_block_right_crushed"
    block_side = "right" if crushed_side == "left" else "left"
    classification = {
        "crushed_side": crushed_side,
        "block_side": block_side,
        "inferred_crushed_side": inferred_crushed_side,
        "role_inference_disagrees": bool(inferred_crushed_side != crushed_side),
        "declared_layout": "left_block_right_crushed",
        "tall_beaker_layout_detected": tall_beaker_layout,
        "role_source": role_source,
        "classification_margin": float(classification_margin),
        "left_role": "crushed" if crushed_side == "left" else "block",
        "right_role": "crushed" if crushed_side == "right" else "block",
        "features": features,
    }

    curves, masks = _occupancy_curves(materialized, tail_reference, rois, threshold)
    events: dict[str, dict[str, Any]] = {}
    event_quality_flags: dict[str, list[str]] = {}
    for side in SIDES:
        events[side], event_failures = _curve_event(curves[side], fps)
        # Missing complete disappearance is not a failure under the revised
        # protocol.  Preserve the reasons next to each curve for sanity check.
        event_quality_flags[side] = list(event_failures)

    water_curves, water_masks = _water_surface_curves(
        materialized, initial_reference, tail_reference, rois, fps=fps, ice_masks=ice_masks
    )
    water_events = {
        side: _water_rise_event(water_curves[side], fps) for side in SIDES
    }
    # Colour-mask area/extent remains diagnostic. Its first activation is not
    # an observed water-level rise and cannot supply a fallback half-rise time.
    observation_path=debug_path/'water_interface_observations.json'
    observation_path.write_text(json.dumps(_jsonable({side:{key:water_curves[side][key]
        for key in ('surface_observation_kind','surface_candidates','initial_level_method')}
        for side in SIDES}),ensure_ascii=False,indent=2))
    artifacts['water_interface_observations']=str(observation_path)

    artifacts.update(_write_segmentation_debug(debug_path, materialized, rois, masks, events, classification))
    artifacts.update(_write_water_line_debug(debug_path, materialized, rois, water_events))
    area_csv = debug_path / "ice_area_curves.csv"
    _write_area_csv(area_csv, fps, events)
    artifacts["ice_area_curves_csv"] = str(area_csv)
    plot_path = debug_path / "ice_area_curves.png"
    _write_curve_plot(plot_path, fps, events)
    artifacts["ice_area_curves_plot"] = str(plot_path)
    water_plot_path = debug_path / "water_surface_curves.png"
    _write_water_plot(water_plot_path, fps, water_events)
    artifacts["water_surface_curves_plot"] = str(water_plot_path)
    water_csv = debug_path / "water_surface_curves.csv"
    _write_water_csv(water_csv, fps, water_events)
    artifacts["water_surface_curves_csv"] = str(water_csv)

    crushed_time = events[crushed_side]["disappearance_time_s"]
    block_time = events[block_side]["disappearance_time_s"]
    if crushed_time is None or block_time is None or block_time <= 0:
        ratio = math.nan
    else:
        ratio = float(crushed_time / block_time)

    # Primary M1: a sustained same-frame normalized liquid-level lead.
    # Half-rise values below are retained as measurement diagnostics.
    crushed_water = water_events[crushed_side]
    block_water = water_events[block_side]
    signal_pair = (
        crushed_water.get("signal_name"),
        block_water.get("signal_name"),
    )
    # Comparing a direct interface displacement with a colour-area proxy would
    # mix units and can manufacture an ordering.  Only use the water half-rise
    # decision when both beakers were measured by the same signal family.
    same_water_signal = bool(signal_pair[0] and signal_pair[0] == signal_pair[1])
    water_times_available = (
        same_water_signal
        and crushed_water.get("usable")
        and block_water.get("usable")
        and crushed_water.get("half_rise_time_s") is not None
        and block_water.get("half_rise_time_s") is not None
    )
    if water_times_available:
        crushed_rise_time = float(crushed_water["half_rise_time_s"])
        block_rise_time = float(block_water["half_rise_time_s"])
        crushed_water_rise_faster = bool(crushed_rise_time < block_rise_time)
        water_comparison_source = "water_surface_half_rise_time"
    else:
        crushed_rise_time = None
        block_rise_time = None
        crushed_water_rise_faster = None
        water_comparison_source = "unavailable"
    surface_lead = compare_surface_lead(water_curves, fps, melting)
    lead_curves = surface_lead.pop('_curves', {})
    lead_path = debug_path / 'surface_lead_observations.csv'
    with lead_path.open('w', newline='') as stream:
        import csv
        writer = csv.writer(stream)
        fields = list(lead_curves)
        writer.writerow(['frame', 'time_s', *fields])
        for i in range(len(materialized)) if fields else []:
            row = [lead_curves[key][i] for key in fields]
            writer.writerow([i, i / fps, *[bool(v) if isinstance(v, (bool, np.bool_)) else float(v) if np.isfinite(v) else '' for v in row]])
    lead_json = debug_path / 'surface_lead_decision.json'
    lead_json.write_text(json.dumps(_jsonable(surface_lead), ensure_ascii=False, indent=2))
    artifacts.update(surface_lead_observations_csv=str(lead_path), surface_lead_decision=str(lead_json))
    if not surface_lead['usable']:
        failures.append('liquid_surface_lead_unobservable')
    initial_area_crushed = features[crushed_side]["initial_mask_area_px"]
    initial_area_block = features[block_side]["initial_mask_area_px"]
    initial_area_ratio = float(initial_area_crushed / max(initial_area_block, 1e-9))
    measurements = base_measurements | {
        'evaluation_rule': SURFACE_LEAD_RULE,
        'surface_lead': surface_lead,
        "classification": classification,
        "late_residual_checks": residual_checks,
        "event_quality_flags": event_quality_flags,
        "water_surface": {
            side: {
                key: value
                for key, value in water_event.items()
                if key not in {
                    "raw_area_px",
                    "raw_height_px",
                    "raw_surface_y_px",
                    "surface_valid",
                    "surface_rise_px",
                    "normalized_surface_rise",
                    "isotonic_surface_rise",
                }
            }
            for side, water_event in water_events.items()
        },
        "events": {
            side: {
                k: v
                for k, v in event.items()
                if k not in {"raw_area_px", "normalized_area", "isotonic_normalized_area"}
            }
            for side, event in events.items()
        },
        "t_crushed_s": crushed_time,
        "t_block_s": block_time,
        "t_crushed_over_t_block": ratio,
        "crushed_water_surface_half_rise_time_s": crushed_rise_time,
        "block_water_surface_half_rise_time_s": block_rise_time,
        "crushed_water_surface_rise_faster": crushed_water_rise_faster,
        "water_surface_comparison_source": water_comparison_source,
        "water_surface_signal_pair": list(signal_pair),
        "water_surface_same_signal": same_water_signal,
        "initial_projected_area_ratio_crushed_over_block": initial_area_ratio,
        "disappearance_threshold_fraction": DISAPPEARANCE_FRACTION,
    }
    metrics = {
        # Current M1 is a qualitative phase-lead observation. A measured
        # absence is false; an unobservable comparison is null, never false.
        "M1": surface_lead['lead_observed'],
        "M1_crushed_liquid_level_lead": surface_lead['lead_observed'],
        "M2_t_crushed_lt_t_block": float(math.isfinite(ratio) and ratio < 1.0),
        "M2_initial_projected_area_log_error": float(abs(math.log(max(initial_area_ratio, 1e-9)))),
        "M2_left_curve_upward_step_fraction": float(events["left"]["large_upward_step_fraction"]),
        "M2_right_curve_upward_step_fraction": float(events["right"]["large_upward_step_fraction"]),
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


evaluate_p28 = evaluate


def evaluate_video(
    video_path: str | Path,
    debug_dir: str | Path,
    sample_id: int = 0,
) -> dict[str, Any]:
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


__all__ = ["evaluate", "evaluate_p28", "evaluate_video"]
