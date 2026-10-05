#!/usr/bin/env python3
"""Deterministic P4 repeated-bounce evaluator.

No LLM or VLM is imported or called.  The optional learned component is CoTracker,
a locally deployed point-tracking model.  It is ensembled with two independent
classical CV trackers (seed-colour segmentation and temporal-background subtraction).

SOP CLI:
    python evaluate.py --video clip.mp4 --task_id P4 --output result.json

The evaluator intentionally separates extraction failure from an observable model
failure.  A well-tracked clip with fewer than four complete rebound arcs has
``extract_success=true`` and ``structural_ok=false``; it is not dropped from the
benchmark denominator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks, savgol_filter

from rescore_continuous import score_record as continuous_score_record


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"


def finite(v: Any) -> Any:
    """Convert numpy values and non-finite floats into strict JSON values."""
    if isinstance(v, dict):
        return {str(k): finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [finite(x) for x in v]
    if isinstance(v, np.ndarray):
        return finite(v.tolist())
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def qscore(error: float | None, good: float, bad: float) -> float:
    if error is None or not math.isfinite(float(error)):
        return 0.0
    e = float(error)
    if e <= good:
        return 100.0
    if e >= bad:
        return 0.0
    return 100.0 * (bad - e) / (bad - good)


def geometric_score(parts: dict[str, float], weights: dict[str, float]) -> float:
    if not parts:
        return 0.0
    eps = 1e-4
    total = sum(float(weights[k]) for k in parts)
    return float(100.0 * math.exp(sum(
        float(weights[k]) / total * math.log(max(eps, parts[k] / 100.0))
        for k in parts
    )))


def physics_decision(structural_ok: bool, scores: dict[str, float], metrics: dict,
                     cfg: dict) -> tuple[bool, list[str]]:
    """Apply the frozen physics gates without averaging away a bad rebound."""
    rules = cfg["pass_rules"]
    reasons: list[str] = []
    ratios = np.asarray(metrics.get("height_ratios") or [], dtype=float)
    height_limit = float(rules.get("maximum_successive_height_ratio", 0.995))
    monotonic_ok = bool(len(ratios) >= 3 and np.isfinite(ratios).all()
                        and np.all(ratios < height_limit))
    m1 = metrics.get("M1_height_time_restitution_consistency")
    if not monotonic_ok:
        reasons.append("rebound_heights_not_strictly_decreasing")
    if m1 is None or float(m1) > float(rules["m1_max"]):
        reasons.append("height_time_restitution_consistency_failed")
    low = [key for key, value in scores.items()
           if key not in {"soft_diagnostic_overall", "overall"}
           and float(value) < float(rules["dimension_min"])]
    if low:
        reasons.append("low_dimensions:" + ",".join(low))
    overall = float(scores.get("overall", 0.0))
    if overall < float(rules["overall_min"]):
        reasons.append("overall_score_below_threshold")
    passed = bool(structural_ok and monotonic_ok and m1 is not None
                  and float(m1) <= float(rules["m1_max"])
                  and not low and overall >= float(rules["overall_min"]))
    return passed, reasons


def robust_scale(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return 0.0
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


@dataclass
class Seed:
    cx: float
    cy: float
    radius: float
    rgb: tuple[int, int, int]
    confidence: float
    source: str


@dataclass
class TrackData:
    xy: np.ndarray
    radius: np.ndarray
    found: np.ndarray
    backend: str


@dataclass
class Arc:
    number: int
    impact_start: int
    apex: int
    impact_end: int
    height_px: float
    duration_frames: int
    impact_start_xy: tuple[float, float]
    apex_xy: tuple[float, float]
    impact_end_xy: tuple[float, float]
    shape_rmse: float
    apex_time_asymmetry: float
    x_drift_radii: float


def read_video(path: Path) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError("no frames decoded")
    return np.stack(frames), fps


def resize_frames(frames: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = frames.shape[1:3]
    if w <= max_width:
        return frames, 1.0
    scale = max_width / float(w)
    size = (int(round(w * scale)), int(round(h * scale)))
    out = np.stack([cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames])
    return out, scale


def _component_candidates(mask: np.ndarray, diff: np.ndarray, hsv: np.ndarray,
                          h: int, w: int) -> list[dict[str, float]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[dict[str, float]] = []
    min_area = max(12.0, 0.00004 * h * w)
    max_area = 0.025 * h * w
    for c in contours:
        area = float(cv2.contourArea(c))
        if not min_area <= area <= max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 3 or bh < 3:
            continue
        peri = float(cv2.arcLength(c, True))
        circularity = min(1.0, 4.0 * math.pi * area / max(peri * peri, 1e-6))
        aspect = min(bw, bh) / max(bw, bh)
        fill = area / max(float(bw * bh), 1.0)
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        pix = m > 0
        dyn = float(np.median(diff[pix])) if pix.any() else 0.0
        sat = float(np.median(hsv[..., 1][pix])) / 255.0 if pix.any() else 0.0
        moments = cv2.moments(c)
        if abs(moments["m00"]) < 1e-6:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        radius = math.sqrt(max(area, 1.0) / math.pi)
        # A weak upper-scene prior is legitimate task geometry, not content recognition:
        # the initially suspended ball must be above its impact plate.
        upper = max(0.0, 1.0 - max(0.0, cy / h - 0.70) / 0.30)
        edge_penalty = 0.4 if cx < radius or cx > w - radius else 1.0
        score = edge_penalty * (
            2.5 * circularity + 1.5 * aspect + 0.7 * fill +
            1.8 * min(dyn / 35.0, 2.0) + 0.35 * sat + 0.35 * upper
        )
        out.append({"score": score, "cx": cx, "cy": cy, "radius": radius,
                    "area": area, "circularity": circularity, "aspect": aspect,
                    "dynamic": dyn, "saturation": sat, "x": x, "y": y,
                    "w": bw, "h": bh})
    return out


def detect_seed(frames: np.ndarray) -> Seed:
    """Locate the initially suspended ball using temporal evidence plus shape.

    The detector never assumes an orange ball.  Frame 0 is compared with a robust
    temporal median made from later frames; compact circular regions that disappear
    from their initial position are preferred.  A high-chroma mask only supplies
    additional candidate boundaries and is not required.
    """
    n, h, w = frames.shape[:3]
    idx = np.unique(np.linspace(max(3, n // 8), n - 1, min(25, max(5, n - 3))).astype(int))
    later = frames[idx]
    med = np.median(later.astype(np.float32), axis=0).astype(np.uint8)
    lab0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2LAB).astype(np.float32)
    labm = cv2.cvtColor(med, cv2.COLOR_RGB2LAB).astype(np.float32)
    diff = np.linalg.norm(lab0 - labm, axis=2)
    hsv = cv2.cvtColor(frames[0], cv2.COLOR_RGB2HSV)

    # Camera shimmer can make every edge nonzero, hence a high percentile plus floor.
    tau = max(13.0, float(np.percentile(diff, 98.8)))
    dynamic = (diff >= tau).astype(np.uint8) * 255
    dynamic = cv2.morphologyEx(dynamic, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dynamic = cv2.morphologyEx(dynamic, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    chroma = ((hsv[..., 1] > 70) & (hsv[..., 2] > 65)).astype(np.uint8) * 255
    chroma = cv2.morphologyEx(chroma, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    candidates = _component_candidates(dynamic, diff, hsv, h, w)
    # Chroma candidates are scored by the same temporal-difference term, so a static
    # colourful cabinet or support does not beat the moving ball.
    candidates += _component_candidates(chroma, diff, hsv, h, w)
    if not candidates:
        raise RuntimeError("automatic ball seed detector found no compact candidate")

    # Deduplicate near-identical candidates from the two masks.
    candidates.sort(key=lambda z: z["score"], reverse=True)
    best = candidates[0]
    if best["dynamic"] < 8.0 or best["circularity"] < 0.20:
        raise RuntimeError(
            f"ball seed confidence too low (dynamic={best['dynamic']:.1f}, "
            f"circularity={best['circularity']:.2f})")

    cx, cy, r = best["cx"], best["cy"], max(3.0, best["radius"])
    rr = max(2, int(round(0.25 * r)))
    x0, x1 = max(0, int(cx) - rr), min(w, int(cx) + rr + 1)
    y0, y1 = max(0, int(cy) - rr), min(h, int(cy) + rr + 1)
    rgb = tuple(np.median(frames[0, y0:y1, x0:x1].reshape(-1, 3), axis=0)
                .round().astype(int).tolist())
    confidence = float(np.clip(
        0.35 * best["circularity"] + 0.25 * best["aspect"] +
        0.40 * min(best["dynamic"] / 35.0, 1.0), 0.0, 1.0))
    return Seed(float(cx), float(cy), float(r), rgb, confidence,
                "temporal_median_shape")


def run_trackers(frames: np.ndarray, seed: Seed, cfg: dict,
                 scratch_dir: Path, no_cotracker: bool = False
                 ) -> tuple[dict[str, TrackData], np.ndarray | None, float]:
    project = Path(cfg["physbench_project"])
    if not project.is_absolute():
        project = (HERE / project).resolve()
    if not project.exists():
        raise RuntimeError(f"physbench tracker project missing: {project}")
    sys.path.insert(0, str(project))
    from physbench.first_frame import BallSeed  # type: ignore
    from physbench.neural_track import TrackerBackendConfig  # type: ignore
    from physbench.tracking import run_tracks  # type: ignore

    backends = tuple(cfg["tracker_backends"])
    if no_cotracker:
        backends = tuple(b for b in backends if b != "cotracker")
    if len(backends) < 2:
        raise RuntimeError("at least two independent tracking backends are required")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    bseed = BallSeed(seed.cx, seed.cy, seed.radius, seed.rgb)
    bcfg = TrackerBackendConfig(
        devices=str(cfg["tracker_gpu"]), scratch_dir=str(scratch_dir), bg_grid=True)
    t0 = time.time()
    result = run_tracks(frames, bseed, backends=backends, backend_cfg=bcfg)
    tracks = {
        name: TrackData(np.asarray(tr.xy, float), np.asarray(tr.radius, float),
                        np.asarray(tr.found, bool), name)
        for name, tr in result.tracks.items()
    }
    shift = None if result.camera_shift is None else np.asarray(result.camera_shift, float)
    return tracks, shift, time.time() - t0


def _pair_disagreement(a: TrackData, b: TrackData, radius: float) -> float:
    both = a.found & b.found
    if both.sum() < 5:
        return float("inf")
    return float(np.median(np.linalg.norm(a.xy[both] - b.xy[both], axis=1)) /
                 max(radius, 1e-6))


def fuse_tracks(tracks: dict[str, TrackData], seed: Seed,
                max_agree_radii: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    names = list(tracks)
    n = len(next(iter(tracks.values())).xy)
    pairwise: dict[str, float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairwise[f"{a}__{b}"] = _pair_disagreement(tracks[a], tracks[b], seed.radius)

    # Rank the global fallback by agreement, coverage, a valid start near the seed,
    # and non-static vertical motion.  This does not fit a bounce model.
    quality: dict[str, float] = {}
    for name, tr in tracks.items():
        found = tr.found
        coverage = float(found.mean())
        start_d = (float(np.linalg.norm(tr.xy[np.flatnonzero(found)[0]] -
                                       np.array([seed.cx, seed.cy]))) / seed.radius
                   if found.any() else 99.0)
        yspan = (float(np.nanpercentile(tr.xy[:, 1], 95) -
                       np.nanpercentile(tr.xy[:, 1], 5)) / seed.radius
                 if found.sum() >= 5 else 0.0)
        agrees = [v for k, v in pairwise.items() if name in k and math.isfinite(v)]
        agreement = min(agrees) if agrees else 9.0
        neural_bonus = 0.12 if name == "cotracker" else 0.0
        quality[name] = (1.5 * coverage + min(yspan / 8.0, 1.0) +
                         math.exp(-0.5 * start_d) + math.exp(-agreement) + neural_bonus)
    fallback = max(quality, key=quality.get)

    xy = np.full((n, 2), np.nan)
    confident = np.zeros(n, bool)
    agree_px = max_agree_radii * seed.radius
    for t in range(n):
        available = [(name, tracks[name].xy[t]) for name in names if tracks[name].found[t]]
        if not available:
            continue
        best_cluster: list[tuple[str, np.ndarray]] = []
        for name, p in available:
            cluster = [(n2, p2) for n2, p2 in available
                       if np.linalg.norm(p2 - p) <= agree_px]
            if len(cluster) > len(best_cluster):
                best_cluster = cluster
            elif len(cluster) == len(best_cluster) and name == fallback:
                best_cluster = cluster
        if len(best_cluster) >= 2:
            xy[t] = np.median(np.stack([p for _, p in best_cluster]), axis=0)
            confident[t] = True
        elif tracks[fallback].found[t]:
            xy[t] = tracks[fallback].xy[t]
        else:
            xy[t] = available[0][1]

    # Remove isolated catastrophic jumps.  The threshold is deliberately large so
    # true impact reversals survive; it only catches tracker swaps to another object.
    valid = np.isfinite(xy[:, 0])
    for _ in range(2):
        drop: list[int] = []
        for t in np.flatnonzero(valid):
            lo, hi = max(0, t - 3), min(n, t + 4)
            nb = np.flatnonzero(valid[lo:hi]) + lo
            nb = nb[nb != t]
            if len(nb) < 3:
                continue
            med = np.median(xy[nb], axis=0)
            if np.linalg.norm(xy[t] - med) > max(5.0 * seed.radius, 18.0):
                drop.append(int(t))
        if not drop:
            break
        xy[drop] = np.nan
        confident[drop] = False
        valid[drop] = False

    diag = {"pairwise_median_disagreement_radii": pairwise,
            "backend_quality": quality, "fallback_backend": fallback,
            "backend_coverage": {k: float(v.found.mean()) for k, v in tracks.items()},
            "fused_coverage": float(np.isfinite(xy[:, 0]).mean()),
            "confident_coverage": float(confident.mean())}
    return xy, confident, diag


def interpolate_short_gaps(xy: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    out = xy.copy()
    n = len(out)
    measured = np.isfinite(out[:, 0])
    for dim in range(2):
        valid = np.isfinite(out[:, dim])
        ids = np.flatnonzero(valid)
        for a, b in zip(ids[:-1], ids[1:]):
            if 1 < b - a <= max_gap + 1:
                out[a:b + 1, dim] = np.linspace(out[a, dim], out[b, dim], b - a + 1)
    return out, measured


def estimate_camera_shift_classic(frames: np.ndarray, xy: np.ndarray,
                                  radius: float) -> np.ndarray:
    """Low-resolution phase-correlation fallback when CoTracker is disabled."""
    n, h, w = frames.shape[:3]
    small_w = min(384, w)
    s = small_w / w
    grays = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY),
                        (small_w, int(round(h * s))), interpolation=cv2.INTER_AREA)
             .astype(np.float32) for f in frames]
    hh, ww = grays[0].shape
    win = cv2.createHanningWindow((ww, hh), cv2.CV_32F)
    shift = np.zeros((n, 2), float)
    for i in range(1, n):
        a, b = grays[i - 1].copy(), grays[i].copy()
        for frame_index, image in ((i - 1, a), (i, b)):
            if np.isfinite(xy[frame_index, 0]):
                cx, cy = xy[frame_index] * s
                pad = int(max(8, 2.5 * radius * s))
                x0, x1 = max(0, int(cx) - pad), min(ww, int(cx) + pad)
                y0, y1 = max(0, int(cy) - pad), min(hh, int(cy) + pad)
                image[y0:y1, x0:x1] = np.median(image)
        (dx, dy), response = cv2.phaseCorrelate(a, b, win)
        if response < 0.05 or math.hypot(dx, dy) > 0.05 * math.hypot(ww, hh):
            dx = dy = 0.0
        shift[i] = shift[i - 1] + np.array([dx / s, dy / s])
    return shift


def _smooth_coordinate(v: np.ndarray) -> np.ndarray:
    x = np.asarray(v, float).copy()
    valid = np.isfinite(x)
    if valid.sum() < 5:
        return x
    ids = np.arange(len(x))
    x[~valid] = np.interp(ids[~valid], ids[valid], x[valid])
    x = median_filter(x, size=3, mode="nearest")
    # Five frames preserves short rebound cusps much better than a long spline.
    if len(x) >= 5:
        x = savgol_filter(x, 5, 2, mode="interp")
    return x


def _release_frame(y: np.ndarray, radius: float) -> int:
    dy = np.diff(y)
    noise = max(0.03 * radius, robust_scale(dy[:max(5, len(dy) // 8)]))
    threshold = max(0.10 * radius, 3.0 * noise)
    for i in range(0, max(1, len(dy) - 3)):
        # Positive image-y velocity is downward; require sustained departure.
        if np.sum(dy[i:i + 3] > threshold) >= 2:
            return i
    return 0


def detect_arcs(xy: np.ndarray, radius: float, cfg: dict
                ) -> tuple[list[Arc], dict[str, Any]]:
    x = _smooth_coordinate(xy[:, 0])
    y = _smooth_coordinate(xy[:, 1])
    if np.isfinite(y).sum() < 8:
        return [], {"reason": "too_few_finite_points"}
    release = _release_frame(y, radius)
    # Estimate *measurement* noise, not physical velocity.  Using the scale of
    # diff(y) would incorrectly call fast free fall "noise" and erase later small
    # but valid rebounds.  Residuals to a short local polynomial isolate tracker
    # scatter while preserving impact cusps.
    raw_y = np.asarray(xy[:, 1], float).copy()
    valid_raw = np.isfinite(raw_y)
    if valid_raw.sum() >= 5:
        ids = np.arange(len(raw_y))
        raw_y[~valid_raw] = np.interp(ids[~valid_raw], ids[valid_raw], raw_y[valid_raw])
        noise = robust_scale(raw_y - y)
    else:
        noise = 0.0
    min_h = max(float(cfg["minimum_arc_height_radii"]) * radius, 4.0 * noise, 1.5)
    min_dist = max(2, int(cfg["minimum_arc_duration_frames"]) - 1)

    impact_candidates, props = find_peaks(
        y, prominence=max(0.25 * min_h, 0.7), distance=min_dist,
        plateau_size=(1, max(1, len(y) // 5)))
    impact_candidates = impact_candidates[impact_candidates > release + 1]

    # Include derivative sign-change impacts that may have modest prominence after
    # the later rebound heights become small.  These still must form a valid arc.
    dy = np.gradient(y)
    turns = np.flatnonzero((dy[:-1] > 0) & (dy[1:] <= 0)) + 1
    turns = turns[turns > release + 1]
    candidates = np.unique(np.r_[impact_candidates, turns]).astype(int)
    # Merge candidates on the same contact plateau, keeping the lowest point.
    merged: list[int] = []
    for p in candidates:
        if merged and p - merged[-1] <= 2:
            lo, hi = merged[-1], p
            merged[-1] = int(lo + np.argmax(y[lo:hi + 1]))
        else:
            merged.append(int(p))

    # A noisy flat apex can produce a tiny positive-y turning point. It is
    # not a collision unless it reaches the observed contact plane shared by
    # the lower turning points. No decay law enters this validity check.
    if len(merged)>=3:
        levels=y[np.asarray(merged,int)]
        lower_turns=levels[levels>=np.percentile(levels,65)-max(3.0*radius,6.0*noise)]
        floor_level=float(np.median(lower_turns))
        contact_tol=max(.65*radius,3.0*robust_scale(lower_turns),4.0*noise,2.0)
        contacts=[i for i in merged if abs(y[i]-floor_level)<=contact_tol]
        if len(contacts)>=2:merged=contacts

    # `find_peaks` cannot mark an endpoint.  A generated clip is allowed to end
    # shortly after its fourth return, so recover a final contact only when the tail
    # actually reaches the established impact level.  A still-descending ball well
    # above that level remains an incomplete arc.
    if merged:
        impact_level = float(np.median(y[np.asarray(merged, int)]))
        contact_tol = max(0.55 * radius, 4.0 * noise, 2.0)
        start = merged[-1] + int(cfg["minimum_arc_duration_frames"])
        if start < len(y) and y[-1] >= impact_level - contact_tol:
            tail = np.arange(start, len(y))
            reached = tail[y[tail] >= impact_level - contact_tol]
            if len(reached):
                end_contact = int(reached[0])
                if (end_contact - merged[-1] >= int(cfg["minimum_arc_duration_frames"]) and
                        impact_level - float(np.min(y[merged[-1]:end_contact + 1])) >= min_h):
                    merged.append(end_contact)

    arcs: list[Arc] = []
    for p0, p1 in zip(merged[:-1], merged[1:]):
        duration = p1 - p0
        if duration < int(cfg["minimum_arc_duration_frames"]):
            continue
        seg = y[p0:p1 + 1]
        # Apex is the highest position in the image: minimum y.
        rel_apex = int(np.argmin(seg[1:-1])) + 1 if len(seg) > 2 else 0
        apex = p0 + rel_apex
        if apex <= p0 or apex >= p1:
            continue
        frac = (apex - p0) / duration
        baseline_apex = y[p0] + frac * (y[p1] - y[p0])
        height = float(baseline_apex - y[apex])
        if height < min_h:
            continue

        ids = np.arange(p0, p1 + 1)
        u = (ids - p0) / duration
        baseline = y[p0] + u * (y[p1] - y[p0])
        observed_h = baseline - y[ids]
        ideal = 4.0 * height * u * (1.0 - u)
        shape_rmse = float(np.sqrt(np.mean((observed_h - ideal) ** 2)) / max(height, 1e-6))
        asym = float(abs(frac - 0.5) / 0.5)
        x_drift = float((np.nanmax(x[ids]) - np.nanmin(x[ids])) / max(radius, 1e-6))
        arcs.append(Arc(
            number=len(arcs) + 1, impact_start=p0, apex=apex, impact_end=p1,
            height_px=height, duration_frames=duration,
            impact_start_xy=(float(x[p0]), float(y[p0])),
            apex_xy=(float(x[apex]), float(y[apex])),
            impact_end_xy=(float(x[p1]), float(y[p1])),
            shape_rmse=shape_rmse, apex_time_asymmetry=asym,
            x_drift_radii=x_drift))

    # The required bounces must be consecutive. Select the longest chain sharing
    # impact endpoints; isolated false peaks cannot be stitched across a gap.
    chains: list[list[Arc]] = []
    for arc in arcs:
        if chains and chains[-1][-1].impact_end == arc.impact_start:
            chains[-1].append(arc)
        else:
            chains.append([arc])
    best = max(chains, key=lambda c: (len(c), sum(a.height_px for a in c)), default=[])
    for i, arc in enumerate(best, 1):
        arc.number = i
    debug = {"release_frame": release, "track_noise_px": noise,
             "minimum_arc_height_px": min_h, "impact_candidates": merged,
             "all_valid_arc_count": len(arcs), "chain_lengths": [len(c) for c in chains]}
    return best, debug


def duplicate_ball_evidence(frames: np.ndarray, seed: Seed, sample_stride: int = 3) -> dict[str, Any]:
    """High-precision duplicate detector based on seed-colour components.

    It is intentionally conservative: a positive finding can hard-fail a clip, but
    a negative finding is not presented as proof that no differently coloured clone
    exists.
    """
    ref = np.asarray(seed.rgb, np.uint8).reshape(1, 1, 3)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(3)
    expected = math.pi * seed.radius * seed.radius
    take = np.unique(np.linspace(0, len(frames) - 1, min(31, len(frames))).astype(int))
    bg_rgb = np.median(frames[take].astype(np.float32), axis=0).astype(np.uint8)
    bg_lab = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    flagged: list[int] = []
    max_count = 0
    for i in range(0, len(frames), sample_stride):
        lab = cv2.cvtColor(frames[i], cv2.COLOR_RGB2LAB).astype(np.float32)
        d = np.linalg.norm(lab - ref_lab, axis=2)
        dynamic = np.linalg.norm(lab - bg_lab, axis=2)
        # Static surfaces that merely share the ball colour are not duplicate
        # evidence.  Requiring temporal novelty makes this deliberately one-sided:
        # positives are strong, negatives are only "not observed".
        mask = ((d < 45.0) & (dynamic > 13.0)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        centres: list[tuple[float, float]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, bw, bh = cv2.boundingRect(contour)
            peri = float(cv2.arcLength(contour, True))
            circ = 4.0 * math.pi * area / max(peri * peri, 1e-6)
            if (0.25 * expected <= area <= 2.6 * expected and
                    min(bw, bh) / max(bw, bh) > 0.50 and circ > 0.38):
                count += 1
                moments = cv2.moments(contour)
                if abs(moments["m00"]) > 1e-6:
                    centres.append((moments["m10"] / moments["m00"],
                                    moments["m01"] / moments["m00"]))
        # A highlight splitting one sphere into adjacent components is not a clone.
        if len(centres) >= 2:
            separated = any(math.dist(a, b) > 2.2 * seed.radius
                            for j, a in enumerate(centres) for b in centres[j + 1:])
            if not separated:
                count = 1
        max_count = max(max_count, count)
        if count >= 2:
            flagged.append(i)
    return {"positive": len(flagged) >= 2, "flagged_frames": flagged,
            "maximum_similar_blob_count": max_count,
            "method": "conservative_seed_colour_components"}


def calculate_metrics(arcs: list[Arc], xy: np.ndarray, radius: float,
                      camera_shift: np.ndarray, track_diag: dict, cfg: dict
                      ) -> tuple[dict[str, Any], dict[str, float]]:
    use = arcs
    heights = np.array([a.height_px for a in use], float)
    durations = np.array([a.duration_frames for a in use], float)
    h_ratio = heights[1:] / heights[:-1] if len(heights) >= 2 else np.array([])
    t_ratio = durations[1:] / durations[:-1] if len(durations) >= 2 else np.array([])
    cor_h = np.sqrt(np.clip(h_ratio, 0.0, None))
    cor_t = t_ratio
    m1_each = np.abs(cor_h - cor_t)
    m1 = float(np.mean(m1_each)) if len(m1_each) else None
    cor_combined = 0.5 * (cor_h + cor_t) if len(cor_h) else np.array([])
    cor_cv = (float(np.std(cor_combined, ddof=1) /
                    max(abs(float(np.mean(cor_combined))), 1e-6))
              if len(cor_combined) >= 2 else None)
    h_excess = float(np.mean(np.maximum(0.0, h_ratio - 1.0))) if len(h_ratio) else None
    # A perfectly equal sequence is not "monotonically decreasing".  Demand a
    # small 2% resolved drop, while allowing ratios in [0.98, 0.995) to pass as
    # lower within practical tracking uncertainty.  Exact/near equality receives
    # a separate categorical penalty so all-equal bounces cannot score perfectly.
    h_monotonic_error = (float(np.mean(np.maximum(0.0, h_ratio - 0.98)))
                         if len(h_ratio) else None)
    h_near_nondecrease_fraction = (float(np.mean(h_ratio >= 0.98))
                                   if len(h_ratio) else None)
    t_excess = float(np.mean(np.maximum(0.0, t_ratio - 1.0))) if len(t_ratio) else None
    h_viol = int(np.sum(h_ratio >= 1.0)) if len(h_ratio) else None
    h_hard_viol = (int(np.sum(h_ratio >= float(
        cfg["pass_rules"].get("maximum_successive_height_ratio", 0.995))))
                   if len(h_ratio) else None)
    t_viol = int(np.sum(t_ratio >= 1.0)) if len(t_ratio) else None
    shape = float(np.mean([a.shape_rmse for a in use])) if use else None
    asym = float(np.mean([a.apex_time_asymmetry for a in use])) if use else None
    xdrift = float(np.mean([a.x_drift_radii for a in use])) if use else None
    impact_x = np.array([use[0].impact_start_xy[0]] +
                        [a.impact_end_xy[0] for a in use]) if use else np.array([])
    impact_spread = (robust_scale(impact_x) / max(radius, 1e-6)
                     if len(impact_x) >= 2 else None)
    cam_fraction = (float(np.max(np.linalg.norm(camera_shift - camera_shift[0], axis=1)) /
                          math.hypot(xy.shape[0], 1)) if False else None)
    # Above placeholder intentionally replaced below using the frame diagonal passed
    # through track_diag; keeping camera units explicit prevents frame-count mistakes.
    frame_diag = float(track_diag["frame_diagonal_px"])
    cam_fraction = float(np.max(np.linalg.norm(camera_shift - camera_shift[0], axis=1)) /
                         max(frame_diag, 1e-6))
    finite_dis = [v for v in track_diag["pairwise_median_disagreement_radii"].values()
                  if v is not None and math.isfinite(float(v))]
    disagreement = min(finite_dis) if finite_dis else None

    metrics = {
        "M1_height_time_restitution_consistency": m1,
        "M1_per_successive_pair": m1_each,
        "M2_restitution_coefficient_cv": cor_cv,
        "M2_monotonic_height_error": h_monotonic_error,
        "M2_strict_height_increase_excess": h_excess,
        "M2_near_nondecreasing_height_fraction": h_near_nondecrease_fraction,
        "M2_monotonic_time_excess": t_excess,
        "height_monotonic_violation_count": h_viol,
        "height_monotonic_hard_violation_count": h_hard_viol,
        "time_monotonic_violation_count": t_viol,
        "height_ratios": h_ratio,
        "time_interval_ratios": t_ratio,
        "restitution_from_height": cor_h,
        "restitution_from_time": cor_t,
        "arc_shape_normalized_rmse": shape,
        "apex_time_asymmetry": asym,
        "mean_arc_horizontal_drift_radii": xdrift,
        "impact_x_robust_spread_radii": impact_spread,
        "camera_drift_fraction_of_diagonal": cam_fraction,
        "best_backend_pair_disagreement_radii": disagreement,
    }

    th = cfg["score_thresholds"]
    completeness = min(100.0, 100.0 * len(use)/max(2,int(cfg["minimum_complete_arcs"])))
    coverage = 100.0 * min(1.0, track_diag["fused_coverage"] /
                           float(cfg["minimum_track_coverage"]))
    confidence = 100.0 * min(1.0, track_diag["confident_coverage"] /
                             float(cfg["minimum_confident_coverage"]))
    structure = 0.60 * completeness + 0.20 * coverage + 0.20 * confidence
    height_continuous = qscore(h_monotonic_error, *th["monotonic_height_excess"])
    height_categorical = (100.0 * (1.0 - h_near_nondecrease_fraction)
                          if h_near_nondecrease_fraction is not None else 0.0)
    height_monotonic_score = 0.50 * height_continuous + 0.50 * height_categorical
    monotonic = 0.80 * height_monotonic_score + \
                0.20 * qscore(t_excess, *th["monotonic_time_excess"])
    ballistic = 0.70 * qscore(shape, *th["arc_shape_rmse"]) + \
                0.30 * qscore(asym, *th["apex_time_asymmetry"])
    restitution = 0.65 * qscore(m1, *th["m1_height_time_consistency"]) + \
                  0.35 * qscore(cor_cv, *th["restitution_cv"])
    stability = (0.45 * qscore(impact_spread, *th["impact_x_spread_radii"]) +
                 0.25 * qscore(xdrift, *th["arc_x_drift_radii"]) +
                 0.15 * qscore(cam_fraction, *th["camera_drift_fraction"]) +
                 0.15 * qscore(disagreement, *th["backend_disagreement_radii"]))
    scores = {"structure": structure, "monotonic_energy_loss": monotonic,
              "ballistic_arc_quality": ballistic,
              "restitution_consistency": restitution,
              "spatial_stability": stability}
    return finite(metrics), {k: float(v) for k, v in scores.items()}


def save_plot(path: Path, xy: np.ndarray, arcs: list[Arc], fps: float,
              result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(xy)) / (fps if fps > 0 else 1.0)
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    ax[0].plot(t, xy[:, 1], color="0.65", lw=1, label="fused y (image coordinates)")
    for a in arcs:
        ax[0].axvline(t[a.impact_start], color="tab:red", alpha=0.45)
        ax[0].scatter(t[a.apex], xy[a.apex, 1], color="tab:blue", s=35)
        ax[0].text(t[a.apex], xy[a.apex, 1], f" A{a.number}", fontsize=8)
    if arcs:
        ax[0].axvline(t[arcs[-1].impact_end], color="tab:red", alpha=0.45)
    ax[0].invert_yaxis()
    ax[0].set(xlabel="time (s)", ylabel="vertical position (px)",
              title=f"P4 tracked motion — {result['sample_id']}")
    ax[0].grid(alpha=0.25)

    use = arcs[:4]
    nums = np.arange(1, len(use) + 1)
    if use:
        ax[1].bar(nums - 0.15, [a.height_px for a in use], width=0.3,
                  label="rebound height (px)")
        ax2 = ax[1].twinx()
        ax2.bar(nums + 0.15, [a.duration_frames for a in use], width=0.3,
                color="tab:orange", label="impact interval (frames)")
        ax2.set_ylabel("frames")
    ax[1].set(xlabel="complete rebound arc", ylabel="height (px)")
    ax[1].set_xticks(nums)
    ax[1].grid(axis="y", alpha=0.25)
    m1 = result["metrics"].get("M1_height_time_restitution_consistency")
    ax[1].set_title(f"M1={m1 if m1 is not None else 'N/A'}; continuous overall={result['scores']['overall']:.3f}")
    fig.savefig(path, dpi=145)
    plt.close(fig)


def save_keyframes(path: Path, frames: np.ndarray, arcs: list[Arc], release: int) -> None:
    ids = [release]
    for a in arcs[:4]:
        ids.extend([a.impact_start, a.apex])
    if arcs:
        ids.append(arcs[min(3, len(arcs) - 1)].impact_end)
    ids = list(dict.fromkeys(int(np.clip(i, 0, len(frames) - 1)) for i in ids))
    thumbs = []
    for i in ids:
        bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        tw = 320
        th = int(round(h * tw / w))
        thumb = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"frame {i}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (0, 0, 255), 2, cv2.LINE_AA)
        thumbs.append(thumb)
    if not thumbs:
        return
    per_row = 5
    rows = []
    for s in range(0, len(thumbs), per_row):
        row = thumbs[s:s + per_row]
        while len(row) < per_row:
            row.append(np.zeros_like(thumbs[0]))
        rows.append(cv2.hconcat(row))
    cv2.imwrite(str(path), cv2.vconcat(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def save_overlay(path: Path, frames: np.ndarray, scale: float, fused: np.ndarray,
                 tracks: dict[str, TrackData], arcs: list[Arc], result: dict,
                 fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    raw_path = path.with_name(path.stem + ".mpeg4-tmp.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps if fps > 0 else 24.0, (w, h))
    impact_map: dict[int, str] = {}
    apex_map: dict[int, str] = {}
    for a in arcs:
        impact_map[a.impact_start] = f"I{a.number}"
        apex_map[a.apex] = f"A{a.number}"
    if arcs:
        impact_map[arcs[-1].impact_end] = f"I{len(arcs) + 1}"
    colors = {"cotracker": (255, 0, 255), "color": (0, 180, 255),
              "bgsub": (255, 200, 0)}
    trail: list[tuple[int, int]] = []
    for i, rgb in enumerate(frames):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for name, tr in tracks.items():
            if tr.found[i]:
                p = tuple(np.round(tr.xy[i] / scale).astype(int))
                cv2.circle(bgr, p, 5, colors.get(name, (255, 255, 0)), 2)
        if np.isfinite(fused[i, 0]):
            p = tuple(np.round(fused[i] / scale).astype(int))
            trail.append(p)
            trail = trail[-30:]
            cv2.circle(bgr, p, 8, (0, 255, 0), 2)
        if len(trail) > 1:
            cv2.polylines(bgr, [np.asarray(trail, np.int32)], False, (0, 255, 0), 2)
        label = impact_map.get(i) or apex_map.get(i)
        if label and np.isfinite(fused[i, 0]):
            p = tuple(np.round(fused[i] / scale).astype(int))
            cv2.putText(bgr, label, (p[0] + 12, p[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 255), 2, cv2.LINE_AA)
        cv2.rectangle(bgr, (0, 0), (w, 62), (0, 0, 0), -1)
        text = (f"P4 | complete arcs={len(arcs)} | structure={'OK' if result['structural_ok'] else 'FAIL'} "
                f"| continuous score={result['scores']['overall']:.3f} | frame={i}")
        cv2.putText(bgr, text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(bgr)
    writer.release()
    # Browser-compatible H.264 makes the mandatory debug overlay directly
    # inspectable in the same gallery.  Fall back to the lossless operation of
    # keeping the MPEG-4 file if this deployment lacks ffmpeg/libx264.
    ffmpeg = Path(sys.executable).with_name("ffmpeg")
    try:
        if not ffmpeg.exists():
            raise FileNotFoundError(ffmpeg)
        subprocess.run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
                        "-i", str(raw_path), "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        "-an", str(path)], check=True)
        raw_path.unlink()
    except Exception:
        os.replace(raw_path, path)


def failure_result(video: Path, task_id: str, reason: str, elapsed: float,
                   evaluator_version: str) -> dict:
    return {
        "task_id": task_id, "sample_id": video.stem,
        "video": str(video.resolve()), "video_sha256": sha256_file(video),
        "evaluator_version": evaluator_version,
        "llm_or_vlm_used": False, "extract_success": False,
        "structural_ok": None, "measurement_valid": False,
        "metric_validity": {"M1": False, "M2": False, "M3": False},
        "physics_pass": False,
        "measurements": {}, "metrics": {
            "M1_height_time_restitution_consistency": None,
            "M2_restitution_coefficient_cv": None,
            "M3": None,
        },
        "scores": {"structure": 0.0, "monotonic_energy_loss": 0.0,
                   "ballistic_arc_quality": 0.0, "restitution_consistency": 0.0,
                   "spatial_stability": 0.0, "overall": 0.0},
        "qc": {}, "failure_reason": reason,
        "runtime_seconds": elapsed,
    }


def evaluate(video: Path, task_id: str, cfg: dict, debug_dir: Path | None,
             no_cotracker: bool = False) -> dict:
    started = time.time()
    frames_full, fps = read_video(video)
    frames, scale = resize_frames(frames_full, int(cfg["max_tracking_width"]))
    seed = detect_seed(frames)
    scratch = (debug_dir or HERE / "scratch") / "tracker_scratch"
    tracks, camera_shift, tracking_seconds = run_trackers(
        frames, seed, cfg, scratch, no_cotracker=no_cotracker)
    fused, confident, track_diag = fuse_tracks(
        tracks, seed, float(cfg["maximum_backend_disagreement_radii"]))
    fused, directly_measured = interpolate_short_gaps(
        fused, int(cfg["maximum_track_gap_frames"]))
    coverage = float(np.isfinite(fused[:, 0]).mean())
    track_diag["fused_coverage"] = coverage
    track_diag["directly_measured_coverage"] = float(directly_measured.mean())
    track_diag["frame_diagonal_px"] = float(math.hypot(frames.shape[2], frames.shape[1]))
    track_diag["tracking_runtime_seconds"] = tracking_seconds

    if camera_shift is None or len(camera_shift) != len(frames):
        camera_shift = estimate_camera_shift_classic(frames, fused, seed.radius)
        track_diag["camera_shift_source"] = "phase_correlation"
    else:
        track_diag["camera_shift_source"] = "cotracker_background_grid"
    corrected = fused - camera_shift

    if coverage < float(cfg["minimum_track_coverage"]):
        raise RuntimeError(f"insufficient fused track coverage: {coverage:.3f}")
    arcs, event_diag = detect_arcs(corrected, seed.radius, cfg)
    duplicate = duplicate_ball_evidence(frames, seed)
    min_arcs = int(cfg["minimum_complete_arcs"])
    enough_arcs = len(arcs) >= min_arcs
    camera_drift = float(np.max(np.linalg.norm(camera_shift - camera_shift[0], axis=1)) /
                         max(track_diag["frame_diagonal_px"], 1e-6))
    camera_ok = camera_drift <= float(cfg["maximum_camera_drift_fraction"])
    structural_ok = bool(enough_arcs and camera_ok and not duplicate["positive"])

    metrics, dim_scores = calculate_metrics(
        arcs, corrected, seed.radius, camera_shift, track_diag, cfg)
    soft_overall = geometric_score(dim_scores, cfg["dimension_weights"])
    candidate_scores = {**dim_scores, "soft_diagnostic_overall": soft_overall,
                        "overall": soft_overall if structural_ok else 0.0}
    physics_pass, physical_reasons = physics_decision(
        structural_ok, candidate_scores, metrics, cfg)
    # End-to-end score is zero for every hard or physics failure.  The conditional
    # score remains available separately for diagnosing how close the motion was.
    scores = {**dim_scores, "soft_diagnostic_overall": soft_overall,
              "overall": soft_overall if physics_pass else 0.0}

    reasons: list[str] = []
    if not enough_arcs:
        reasons.append(f"only_{len(arcs)}_complete_rebound_arcs_detected;_need_{min_arcs}")
    if not camera_ok:
        reasons.append(f"camera_drift_{camera_drift:.4f}_exceeds_limit")
    if duplicate["positive"]:
        reasons.append("multiple_ball_like_instances_detected")
    if structural_ok and not physics_pass:
        reasons.extend(physical_reasons)

    use = arcs
    measurements = {
        "fps": fps, "frame_count": len(frames), "width": frames_full.shape[2],
        "height": frames_full.shape[1], "tracking_scale": scale,
        "seed": asdict(seed), "release_frame": event_diag.get("release_frame"),
        "complete_rebound_arc_count": len(arcs),
        "required_rebound_arc_count": min_arcs,
        "single_motion_no_repeated_bounce": bool(len(arcs)<2 and coverage>=float(cfg['minimum_track_coverage'])
            and np.nanmax(corrected[:,1])-corrected[0,1] > max(4.,seed.radius)),
        "impact_frames": ([use[0].impact_start] + [a.impact_end for a in use]
                          if use else []),
        "apex_frames": [a.apex for a in use],
        "rebound_heights_px": [a.height_px for a in use],
        "impact_intervals_frames": [a.duration_frames for a in use],
        "arcs": [asdict(a) for a in arcs],
    }
    result = {
        "task_id": task_id, "sample_id": video.stem,
        "video": str(video.resolve()), "video_sha256": sha256_file(video),
        "evaluator_version": cfg["evaluator_version"],
        "llm_or_vlm_used": False,
        "tracking_methods": list(tracks),
        "extract_success": True, "structural_ok": structural_ok,
        "measurement_valid": True,
        "metric_validity": {"M1": bool(len(arcs) >= 2),
                            "M2": bool(len(arcs) >= 2), "M3": False},
        "physics_pass": physics_pass,
        "measurements": finite(measurements), "metrics": finite(metrics),
        "scores": finite(scores),
        "qc": finite({"tracking": track_diag, "events": event_diag,
                      "duplicate_ball_evidence": duplicate,
                      "camera_drift_fraction": camera_drift,
                      "directly_measured_track_fraction": float(directly_measured.mean()),
                      "confident_track_fraction": float(confident.mean())}),
        "failure_reason": "; ".join(reasons) if reasons else None,
        "runtime_seconds": time.time() - started,
        "environment": {"python": sys.version.split()[0], "opencv": cv2.__version__,
                        "numpy": np.__version__, "platform": platform.platform()},
    }
    # Keep categorical benchmark decisions separate from the continuous score.
    # A measurable physics/structure failure therefore retains a non-zero score.
    result = continuous_score_record(result)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        save_plot(debug_dir / "trajectory_plot.png", corrected, arcs, fps, result)
        save_keyframes(debug_dir / "keyframes.jpg", frames_full, arcs,
                       int(event_diag.get("release_frame", 0)))
        save_overlay(debug_dir / "overlay.mp4", frames_full, scale, fused,
                     tracks, arcs, result, fps)
        np.savetxt(debug_dir / "fused_track.csv",
                   np.c_[np.arange(len(corrected)), corrected, directly_measured, confident],
                   delimiter=",", header="frame,x_stabilized,y_stabilized,direct,confident",
                   comments="")
    return finite(result)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--task_id", default="P4")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--debug-dir", type=Path, default=None)
    p.add_argument("--no-cotracker", action="store_true",
                   help="classical deterministic fallback; formal run uses CoTracker")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        result = evaluate(args.video, args.task_id, cfg, args.debug_dir,
                          no_cotracker=args.no_cotracker)
    except Exception as exc:  # Always emit SOP JSON, including extraction failures.
        result = failure_result(args.video, args.task_id,
                                f"{type(exc).__name__}: {exc}",
                                time.time() - started, cfg["evaluator_version"])
    # The second call is idempotent and also covers extraction-failure records.
    result = continuous_score_record(result)
    args.output.write_text(json.dumps(finite(result), ensure_ascii=False, indent=2))
    print(json.dumps({"sample_id": result["sample_id"],
                      "extract_success": result["extract_success"],
                      "structural_ok": result["structural_ok"],
                      "physics_pass": result["physics_pass"],
                      "score": result["scores"]["overall"],
                      "failure_reason": result["failure_reason"]},
                     ensure_ascii=False))
    return 0 if result["extract_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
