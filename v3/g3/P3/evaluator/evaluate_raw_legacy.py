#!/usr/bin/env python3
"""Deterministic, no-LLM/VLM evaluator for P3 complementary projectiles.

Neural components are restricted to SAM2 segmentation propagation and CoTracker point
tracking.  They only return image coordinates.  Events, fits, residuals, gates and
scores are fixed numerical procedures defined by the versioned YAML configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import yaml

from rescore_continuous import rescore_result
from scipy.signal import savgol_filter


HERE = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    # bool subclasses int in Python, so this must precede the integer branch.
    # Keeping status fields as JSON booleans is part of the public schema.
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def load_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()


def read_video(path: Path) -> tuple[np.ndarray, np.ndarray, float, dict]:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 24.0)
        frames, times = [], []
        for index, frame in enumerate(container.decode(stream)):
            frames.append(frame.to_ndarray(format="rgb24"))
            if frame.pts is not None and frame.time_base is not None:
                times.append(float(frame.pts * frame.time_base))
            else:
                times.append(index / fps)
        audio_streams = len(container.streams.audio)
    if not frames:
        raise RuntimeError("video has no decoded frames")
    arr = np.stack(frames).astype(np.uint8, copy=False)
    ts = np.asarray(times, np.float64)
    if len(ts) > 1 and np.any(np.diff(ts) <= 0):
        ts = np.arange(len(arr), dtype=np.float64) / fps
    duplicate = []
    for i in range(1, len(arr)):
        if np.mean(np.abs(arr[i].astype(np.int16) - arr[i - 1].astype(np.int16))) < 0.08:
            duplicate.append(i)
    meta = {
        "n_frames": len(arr), "width": int(arr.shape[2]), "height": int(arr.shape[1]),
        "fps": fps, "duration_s": float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
        "audio_streams": audio_streams, "duplicate_frame_count": len(duplicate),
        "duplicate_frames": duplicate,
    }
    return arr, ts, fps, meta


def sample_id_from_video(video: Path) -> tuple[str, int | None]:
    match = re.match(r"(.+)_seed(\d+)$", video.stem)
    return (match.group(1), int(match.group(2))) if match else (video.stem, None)


@dataclass
class Track:
    xy: np.ndarray
    radius: np.ndarray
    score: np.ndarray
    name: str

    @property
    def found(self) -> np.ndarray:
        return np.isfinite(self.xy[:, 0]) & np.isfinite(self.xy[:, 1])

    @property
    def coverage(self) -> float:
        return float(self.found.mean())


def run_neural_tracking(frames: np.ndarray, sample_cfg: dict, cfg: dict,
                        debug_dir: Path, video_hash: str, config_hash: str,
                        device: str, force: bool = False) -> tuple[dict[str, dict[str, Track]], np.ndarray, dict]:
    cache = debug_dir / "tracks_raw.npz"
    cache_meta = debug_dir / "tracks_raw.meta.json"
    expected = {"video_sha256": video_hash, "config_sha256": config_hash,
                "evaluator_version": cfg["evaluator_version"]}
    if cache.exists() and cache_meta.exists() and not force:
        try:
            if json.loads(cache_meta.read_text()) == expected:
                data = np.load(cache, allow_pickle=False)
                return _tracks_from_npz(data, sample_cfg)
        except Exception:
            pass

    seeds = [{"slot": slot, **ball} for slot, ball in sample_cfg["balls"].items()]
    runtime = cfg["runtime"]
    debug_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="p3-track-", dir=str(debug_dir)) as temp:
        temp_path = Path(temp)
        frame_path = temp_path / "frames.npy"
        job_path = temp_path / "job.json"
        np.save(frame_path, np.ascontiguousarray(frames))
        job = {
            "frames": str(frame_path), "seeds": seeds, "output": str(cache),
            "device": "cuda", "dtype": runtime.get("dtype", "bfloat16"),
            "sam2_path": runtime["sam2_snapshot"],
            "cotracker_source": runtime["cotracker_source"],
            "cotracker_checkpoint": runtime["cotracker_checkpoint"],
        }
        job_path.write_text(json.dumps(job))
        env = dict(os.environ)
        env.update({
            "CUDA_VISIBLE_DEVICES": str(device),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
        })
        cmd = [runtime["tracking_python"], str(HERE / "track_worker_p3.py"), str(job_path)]
        started = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        seconds = time.time() - started
        if not cache.exists():
            raise RuntimeError(
                f"tracking worker failed rc={proc.returncode}\n"
                f"{(proc.stdout + proc.stderr)[-6000:]}"
            )
        cache_meta.write_text(json.dumps(expected, indent=2))
    data = np.load(cache, allow_pickle=False)
    tracks, shift, meta = _tracks_from_npz(data, sample_cfg)
    meta["worker_seconds"] = seconds
    meta["worker_returncode"] = proc.returncode
    meta["worker_tail"] = (proc.stdout + proc.stderr)[-2000:]
    return tracks, shift, meta


def _tracks_from_npz(data, sample_cfg):
    tracks: dict[str, dict[str, Track]] = {}
    for slot in sample_cfg["balls"]:
        tracks[slot] = {}
        for backend in ("sam2", "cotracker"):
            tracks[slot][backend] = Track(
                xy=np.asarray(data[f"{backend}_{slot}_xy"], float),
                radius=np.asarray(data[f"{backend}_{slot}_radius"], float),
                score=np.asarray(data[f"{backend}_{slot}_score"], float),
                name=backend,
            )
    shift = np.asarray(data["camera_shift"], float)
    raw_meta = str(np.asarray(data["meta_json"]).item())
    return tracks, shift, json.loads(raw_meta)


def estimate_camera_similarity(frames: np.ndarray, cotracker_shift: np.ndarray,
                               stride: int = 4) -> tuple[np.ndarray, dict]:
    """Estimate frame-zero -> frame similarity without consulting the ball motion."""
    n, h, w = frames.shape[:3]
    resize = 0.5 if w > 900 else 1.0
    ref = cv2.cvtColor(cv2.resize(frames[0], None, fx=resize, fy=resize), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=1800, contrastThreshold=0.025)
    kp0, des0 = sift.detectAndCompute(ref, None)
    indices = sorted(set(list(range(0, n, stride)) + [n - 1]))
    observations = []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    if des0 is not None and len(kp0) >= 16:
        for index in indices:
            if index == 0:
                observations.append((0, 0.0, 0.0, 1.0, 0.0, len(kp0), len(kp0)))
                continue
            gray = cv2.cvtColor(
                cv2.resize(frames[index], None, fx=resize, fy=resize), cv2.COLOR_RGB2GRAY
            )
            kp, des = sift.detectAndCompute(gray, None)
            if des is None or len(kp) < 12:
                continue
            pairs = matcher.knnMatch(des0, des, k=2)
            good = [a for a, b in pairs if a.distance < 0.72 * b.distance]
            if len(good) < 12:
                continue
            src = np.float32([kp0[m.queryIdx].pt for m in good])
            dst = np.float32([kp[m.trainIdx].pt for m in good])
            matrix, inliers = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.5,
                maxIters=3000, confidence=0.995
            )
            if matrix is None or inliers is None or int(inliers.sum()) < 10:
                continue
            a, b = float(matrix[0, 0]), float(matrix[1, 0])
            scale = math.hypot(a, b)
            angle = math.degrees(math.atan2(b, a))
            tx, ty = float(matrix[0, 2] / resize), float(matrix[1, 2] / resize)
            observations.append((index, tx, ty, scale, angle, len(good), int(inliers.sum())))

    transform = np.zeros((n, 2, 3), float)
    if len(observations) >= max(4, len(indices) // 3):
        oi = np.asarray([x[0] for x in observations], float)
        tx = np.interp(np.arange(n), oi, [x[1] for x in observations])
        ty = np.interp(np.arange(n), oi, [x[2] for x in observations])
        scale = np.interp(np.arange(n), oi, [x[3] for x in observations])
        angle = np.radians(np.interp(np.arange(n), oi, [x[4] for x in observations]))
        source = "sift_ransac_similarity"
    else:
        tx = np.asarray(cotracker_shift[:, 0], float)
        ty = np.asarray(cotracker_shift[:, 1], float)
        scale = np.ones(n)
        angle = np.zeros(n)
        source = "cotracker_background_translation_fallback"
    for index in range(n):
        c, s = math.cos(angle[index]) * scale[index], math.sin(angle[index]) * scale[index]
        transform[index] = ((c, -s, tx[index]), (s, c, ty[index]))
    diag = math.hypot(w, h)
    metrics = {
        "source": source,
        "sampled_frames": len(indices), "valid_similarity_frames": len(observations),
        "max_translation_px": float(np.max(np.hypot(tx, ty))),
        "max_translation_diag": float(np.max(np.hypot(tx, ty)) / diag),
        "max_scale_error": float(np.max(np.abs(scale - 1.0))),
        "max_rotation_deg": float(np.max(np.abs(np.degrees(angle)))),
        "observations": [
            {"frame": int(o[0]), "tx_px": o[1], "ty_px": o[2], "scale": o[3],
             "rotation_deg": o[4], "matches": o[5], "inliers": o[6]}
            for o in observations
        ],
    }
    return transform, metrics


def compensate_track(track: Track, transform: np.ndarray) -> Track:
    xy = track.xy.copy()
    for index in np.flatnonzero(track.found):
        matrix = np.vstack([transform[index], [0.0, 0.0, 1.0]])
        inverse = np.linalg.inv(matrix)
        xy[index] = (inverse @ np.r_[xy[index], 1.0])[:2]
    scale = np.sqrt(transform[:, 0, 0] ** 2 + transform[:, 1, 0] ** 2)
    radius = track.radius / np.where(scale > 1e-8, scale, 1.0)
    return Track(xy=xy, radius=radius, score=track.score.copy(), name=track.name)


def fuse_tracks(sam: Track, cot: Track, seed: dict) -> tuple[Track, dict]:
    n = len(sam.xy)
    xy = np.full((n, 2), np.nan)
    radius = np.full(n, np.nan)
    score = np.full(n, np.nan)
    source = np.full(n, "missing", dtype="U24")
    distances = []
    overlap_indices = []
    for index in range(n):
        has_s, has_c = sam.found[index], cot.found[index]
        if has_s and has_c:
            disagreement = float(np.linalg.norm(sam.xy[index] - cot.xy[index]) / seed["radius"])
            distances.append(disagreement)
            overlap_indices.append(index)
            if disagreement <= 1.25:
                xy[index] = 0.65 * sam.xy[index] + 0.35 * cot.xy[index]
                source[index] = "fused"
            else:
                # SAM2's mask identity survives the documented same-colour near-overlap
                # case better than CoTracker.  The disagreement is retained as QC and
                # can invalidate the sample; it is never silently discarded.
                xy[index] = sam.xy[index]
                source[index] = "sam2_disagreement"
            radius[index] = sam.radius[index] if np.isfinite(sam.radius[index]) else cot.radius[index]
            score[index] = min(sam.score[index], cot.score[index])
        elif has_s:
            xy[index], radius[index], score[index], source[index] = (
                sam.xy[index], sam.radius[index], sam.score[index], "sam2_only"
            )
        elif has_c:
            xy[index], radius[index], score[index], source[index] = (
                cot.xy[index], cot.radius[index], cot.score[index], "cotracker_only"
            )
    overlap = np.asarray(overlap_indices, int)
    coverage_sections = []
    if len(overlap):
        thirds = np.array_split(np.arange(n), 3)
        for section in thirds:
            coverage_sections.append(bool(np.intersect1d(overlap, section).size))
    meta = {
        "sam2_coverage": sam.coverage, "cotracker_coverage": cot.coverage,
        "fused_coverage": float(np.isfinite(xy[:, 0]).mean()),
        "backend_overlap_frames": int(len(overlap)),
        "backend_overlap_sections": coverage_sections,
        "backend_disagreement_median_radii": float(np.median(distances)) if distances else None,
        "backend_disagreement_p95_radii": float(np.percentile(distances, 95)) if distances else None,
        "source_counts": {name: int(np.sum(source == name)) for name in np.unique(source)},
    }
    return Track(xy=xy, radius=radius, score=score, name="fused"), {**meta, "source": source}


def local_colour_validation(frames: np.ndarray, guide: Track, seed: dict) -> dict:
    """Local colour evidence only; it never changes the fused coordinates."""
    h, w = frames.shape[1:3]
    yy, xx = np.ogrid[:h, :w]
    disc = (xx - seed["cx"]) ** 2 + (yy - seed["cy"]) ** 2 <= (0.70 * seed["radius"]) ** 2
    lab0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2LAB).astype(np.float32)
    target = np.median(lab0[disc], axis=0)
    centres, support = [], np.zeros(len(frames), bool)
    errors = []
    search = int(max(16, round(4.0 * seed["radius"])))
    expected = math.pi * seed["radius"] ** 2
    for index in np.flatnonzero(guide.found):
        gx, gy = guide.xy[index]
        x0, x1 = max(0, int(gx) - search), min(w, int(gx) + search + 1)
        y0, y1 = max(0, int(gy) - search), min(h, int(gy) + search + 1)
        lab = cv2.cvtColor(frames[index, y0:y1, x0:x1], cv2.COLOR_RGB2LAB).astype(np.float32)
        distance = np.linalg.norm(lab - target, axis=2)
        best = None
        for threshold in (28.0, 40.0, 55.0, 75.0):
            mask = (distance < threshold).astype(np.uint8)
            count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
            for component in range(1, count):
                area = float(stats[component, cv2.CC_STAT_AREA])
                if not 0.06 * expected <= area <= 3.5 * expected:
                    continue
                cx, cy = centroids[component] + np.array([x0, y0])
                err = float(np.hypot(cx - gx, cy - gy) / seed["radius"])
                area_err = abs(math.log(max(area, 1.0) / expected))
                cost = err + 0.25 * area_err
                if best is None or cost < best[0]:
                    best = (cost, cx, cy, err)
            if best is not None and best[0] < 1.8:
                break
        if best is not None:
            support[index] = True
            centres.append((index, best[1], best[2]))
            errors.append(best[3])
    return {
        "coverage_on_guide": float(support[guide.found].mean()) if guide.found.any() else 0.0,
        "median_disagreement_radii": float(np.median(errors)) if errors else None,
        "p95_disagreement_radii": float(np.percentile(errors, 95)) if errors else None,
        "support_frames": int(support.sum()),
    }


def _max_gap(found: np.ndarray, lo: int = 0, hi: int | None = None) -> int:
    hi = len(found) - 1 if hi is None else hi
    best = run = 0
    for value in found[lo:hi + 1]:
        run = 0 if value else run + 1
        best = max(best, run)
    return int(best)


def interpolate_short_gaps(xy: np.ndarray, max_gap: int = 2) -> tuple[np.ndarray, np.ndarray]:
    out = xy.copy()
    filled = np.zeros(len(xy), bool)
    found = np.isfinite(xy[:, 0])
    index = 0
    while index < len(xy):
        if found[index]:
            index += 1
            continue
        start = index
        while index < len(xy) and not found[index]:
            index += 1
        end = index - 1
        length = end - start + 1
        if start > 0 and index < len(xy) and length <= max_gap:
            for k in range(start, index):
                fraction = (k - start + 1) / (length + 1)
                out[k] = (1 - fraction) * out[start - 1] + fraction * out[index]
                filled[k] = True
    return out, filled


def extract_flight(track: Track, seed: dict, timestamps: np.ndarray, cfg: dict) -> tuple[dict | None, str]:
    gates = cfg["gates"]
    xy_event, imputed = interpolate_short_gaps(track.xy, max_gap=2)
    found = np.isfinite(xy_event[:, 0])
    if int(found.sum()) < gates["min_frames"]:
        return None, "too_few_track_points"
    centre = np.array([seed["cx"], seed["cy"]], float)
    displacement = np.linalg.norm(xy_event - centre, axis=1)
    confirmation = int(gates["release_confirmation_frames"])
    threshold = float(gates["release_displacement_radii"] * seed["radius"])
    release = None
    for index in range(1, len(track.xy) - confirmation + 1):
        window = slice(index, index + confirmation)
        if found[window].all() and np.all(displacement[window] > threshold):
            if np.nanmedian(np.diff(xy_event[index - 1:index + confirmation, 0])) > 0:
                release = index
                break
    if release is None:
        span = np.nanmax(displacement) if np.isfinite(displacement).any() else 0.0
        return None, "ball_never_released" if span < seed["radius"] else "release_not_resolved"

    valid_after = np.flatnonzero(found & (np.arange(len(found)) >= release))
    if len(valid_after) < gates["min_frames"]:
        return None, "too_few_post_release_points"
    apex = int(valid_after[np.argmin(xy_event[valid_after, 1])])
    if apex <= release:
        return None, "no_ascent"

    y_ref = float(seed["cy"])
    min_x = float(seed["cx"] + 4.0 * seed["radius"])
    landing = None
    t_land = None
    x_land = None
    # First descending crossing of the frozen launch-centre height.
    for a in range(apex, len(xy_event) - 1):
        b = a + 1
        if not (found[a] and found[b]) or xy_event[b, 0] < min_x:
            continue
        ya, yb = xy_event[a, 1], xy_event[b, 1]
        if ya <= y_ref <= yb and yb > ya + 1e-6:
            fraction = float((y_ref - ya) / (yb - ya))
            t_land = float(timestamps[a] + fraction * (timestamps[b] - timestamps[a]))
            x_land = float(xy_event[a, 0] + fraction * (xy_event[b, 0] - xy_event[a, 0]))
            landing = b
            break
    # Contact without a literal crossing: enter a stable band around the lane.
    if landing is None:
        # A rigid lane may stop the centre slightly above its release centre, so no
        # literal level crossing occurs.  Detect the first post-apex local lowest point
        # (image y maximum) followed by a clear rebound, independently of any fit.
        lowest = apex
        contact = None
        for index in range(apex + 1, len(xy_event)):
            if not found[index]:
                continue
            if xy_event[index, 1] > xy_event[lowest, 1]:
                lowest = index
            elif (xy_event[index, 1] < xy_event[lowest, 1] - 0.08 * seed["radius"]
                  and xy_event[lowest, 0] >= min_x):
                contact = lowest
                break
        if contact is not None:
            level_error_d = abs(xy_event[contact, 1] - y_ref) / (2 * seed["radius"])
            if level_error_d <= gates["max_landing_level_error_diameters"]:
                landing = contact
                t_land = float(timestamps[contact])
                x_land = float(xy_event[contact, 0])
    if landing is None:
        band = 0.50 * seed["radius"]
        for index in range(apex + 1, len(xy_event) - 2):
            if not found[index:index + 3].all() or xy_event[index, 0] < min_x:
                continue
            if abs(xy_event[index, 1] - y_ref) <= band:
                dy = np.diff(xy_event[index:index + 3, 1])
                if np.nanmax(np.abs(dy)) <= 0.35 * seed["radius"]:
                    landing = index
                    t_land = float(timestamps[index])
                    x_land = float(xy_event[index, 0])
                    break
    if landing is None:
        return None, "first_landing_not_observed"
    if landing - release < gates["min_frames"]:
        return None, f"flight_too_short({landing - release})"
    duration = float(t_land - timestamps[max(0, release - 1)])
    if duration < gates["min_duration_s"]:
        return None, f"flight_duration_too_short({duration:.3f}s)"
    inner_coverage = float(track.found[release:landing + 1].mean())
    gap = _max_gap(track.found, release, landing)

    # Local sub-frame apex for debug and height; event identity itself stays data-driven.
    around = np.arange(max(release, apex - 3), min(landing + 1, apex + 4))
    around = around[track.found[around]]
    t_apex, y_apex = float(timestamps[apex]), float(track.xy[apex, 1])
    if len(around) >= 3:
        tau = timestamps[around] - timestamps[around].mean()
        coeff = np.polyfit(tau, track.xy[around, 1], 2)
        if coeff[0] > 1e-9:
            vertex = -coeff[1] / (2 * coeff[0])
            if tau.min() <= vertex <= tau.max():
                t_apex = float(vertex + timestamps[around].mean())
                y_apex = float(np.polyval(coeff, vertex))
    launch_index = max(0, release - 1)
    return {
        "release_frame": int(release), "launch_frame": int(launch_index),
        "apex_frame": int(apex), "landing_frame": int(landing),
        "t_launch_s": float(timestamps[launch_index]), "t_apex_s": t_apex,
        "t_land_s": float(t_land), "x_launch_px": float(seed["cx"]),
        "x_land_px": float(x_land), "y_ref_px": y_ref, "y_apex_px": y_apex,
        "inner_coverage": inner_coverage, "max_gap": gap,
        "event_imputed_frames": np.flatnonzero(imputed[release:landing + 1]).astype(int) + release,
        "landing_level_error_d": float(abs(xy_event[landing, 1] - y_ref) / (2 * seed["radius"])),
    }, "ok"


def robust_fit(design: np.ndarray, values: np.ndarray, iterations: int = 8) -> np.ndarray:
    weights = np.ones(len(values), float)
    coef = np.linalg.lstsq(design, values, rcond=None)[0]
    for _ in range(iterations):
        residual = values - design @ coef
        centre = np.median(residual)
        scale = 1.4826 * np.median(np.abs(residual - centre)) + 1e-8
        u = np.abs(residual - centre) / (1.5 * scale)
        weights = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
        coef = np.linalg.lstsq(design * weights[:, None], values * weights, rcond=None)[0]
    return coef


def robust_cv(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return float("nan")
    denominator = abs(float(np.median(values)))
    if denominator < 1e-8:
        return float("inf")
    return float(1.4826 * np.median(np.abs(values - np.median(values))) / denominator)


def measure_ball(track: Track, seed: dict, flight: dict, timestamps: np.ndarray) -> tuple[dict, dict]:
    lo, hi = flight["launch_frame"], flight["landing_frame"]
    valid = track.found & (np.arange(len(track.xy)) >= lo) & (np.arange(len(track.xy)) <= hi)
    indices = np.flatnonzero(valid)
    t = timestamps[indices] - flight["t_launch_s"]
    diameter = 2.0 * float(seed["radius"])
    x = (track.xy[indices, 0] - float(seed["cx"])) / diameter
    y = (float(seed["cy"]) - track.xy[indices, 1]) / diameter

    dx = np.column_stack([np.ones(len(t)), t])
    dy = np.column_stack([np.ones(len(t)), t, t ** 2])
    cx = robust_fit(dx, x)
    cy = robust_fit(dy, y)
    pred_x, pred_y = dx @ cx, dy @ cy
    rx, ry = x - pred_x, y - pred_y
    vx0, vy0, gravity = float(cx[1]), float(cy[1]), float(-2.0 * cy[2])

    spatial = np.column_stack([np.ones(len(x)), x, x ** 2])
    cp = robust_fit(spatial, y)
    r_para = y - spatial @ cp

    dense_xy, _ = interpolate_short_gaps(track.xy[lo:hi + 1], max_gap=2)
    dense_ok = np.isfinite(dense_xy[:, 0])
    vx_cv = vy_residual = ay_cv = float("nan")
    series: dict[str, np.ndarray] = {}
    if dense_ok.all() and len(dense_xy) >= 9:
        td = timestamps[lo:hi + 1] - flight["t_launch_s"]
        xd = (dense_xy[:, 0] - seed["cx"]) / diameter
        yd = (seed["cy"] - dense_xy[:, 1]) / diameter
        window = min(11, len(xd) if len(xd) % 2 else len(xd) - 1)
        if window >= 7:
            delta = float(np.median(np.diff(td)))
            vx = savgol_filter(xd, window, 3, deriv=1, delta=delta, mode="interp")
            vy = savgol_filter(yd, window, 3, deriv=1, delta=delta, mode="interp")
            ay = savgol_filter(yd, window, 3, deriv=2, delta=delta, mode="interp")
            edge = window // 2
            core = slice(edge, len(xd) - edge)
            vx_cv = robust_cv(vx[core])
            scale_vy = abs(gravity) * max(float(td[-1] - td[0]), 1e-6)
            vy_residual = float(np.sqrt(np.mean(
                (vy[core] - (vy0 - gravity * td[core])) ** 2
            )) / max(scale_vy, 1e-8))
            ay_cv = robust_cv(ay[core])
            series = {"t": td, "x": xd, "y": yd, "vx": vx, "vy": vy, "ay": ay}

    radius_segment = track.radius[lo:hi + 1]
    radius_segment = radius_segment[np.isfinite(radius_segment)]
    radius_cv = robust_cv(radius_segment) if len(radius_segment) >= 5 else float("nan")
    measured = {
        "angle_target_deg": float(seed["angle_deg"]),
        "angle_measured_deg": float(math.degrees(math.atan2(vy0, vx0))) if vx0 > 0 else None,
        "v0_diameters_per_s": float(math.hypot(vx0, vy0)),
        "vx0_diameters_per_s": vx0, "vy0_diameters_per_s": vy0,
        "g_diameters_per_s2": gravity,
        "range_diameters": float((flight["x_land_px"] - seed["cx"]) / diameter),
        "height_diameters": float((seed["cy"] - flight["y_apex_px"]) / diameter),
        "fit_points": int(len(indices)), "radius_robust_cv": radius_cv,
    }
    metrics = {
        "parabola_rmse_d": float(np.sqrt(np.mean(r_para ** 2))),
        "parabola_p95_d": float(np.percentile(np.abs(r_para), 95)),
        "parabola_quadratic": float(cp[2]),
        "x_fit_rmse_d": float(np.sqrt(np.mean(rx ** 2))),
        "y_fit_rmse_d": float(np.sqrt(np.mean(ry ** 2))),
        "x_fit_p95_d": float(np.percentile(np.abs(rx), 95)),
        "y_fit_p95_d": float(np.percentile(np.abs(ry), 95)),
        "vx_robust_cv": vx_cv, "vy_line_residual": vy_residual,
        "ay_robust_cv": ay_cv,
    }
    debug = {
        "indices": indices, "t": t, "x": x, "y": y,
        "pred_x": pred_x, "pred_y": pred_y,
        "spatial_coeff": cp, "time_x_coeff": cx, "time_y_coeff": cy,
        "series": series,
    }
    return {"measurements": measured, "metrics": metrics}, debug


def q_score(error: float | None, threshold: dict) -> float:
    if error is None or not math.isfinite(float(error)):
        return 0.0
    value = abs(float(error))
    good, bad = float(threshold["good"]), float(threshold["bad"])
    if value <= good:
        return 100.0
    if value >= bad:
        return 0.0
    return float(100.0 * (bad - value) / (bad - good))


def aggregate_balls(a: float, b: float) -> float:
    return float(0.7 * min(a, b) + 0.3 * np.mean([a, b]))


def score_results(ball_results: dict, flights: dict, tracking_qc: dict,
                  camera: dict, cfg: dict) -> tuple[dict, dict]:
    thresholds = cfg["thresholds"]
    by_angle = {int(round(v["measurements"]["angle_target_deg"])): v for v in ball_results.values()}
    b30, b60 = by_angle[30], by_angle[60]
    f30 = flights[next(slot for slot, r in ball_results.items()
                       if int(round(r["measurements"]["angle_target_deg"])) == 30)]
    f60 = flights[next(slot for slot, r in ball_results.items()
                       if int(round(r["measurements"]["angle_target_deg"])) == 60)]

    sync_frames = abs(f30["t_launch_s"] - f60["t_launch_s"]) * tracking_qc["fps"]
    angle30 = b30["measurements"]["angle_measured_deg"]
    angle60 = b60["measurements"]["angle_measured_deg"]
    # Non-positive fitted horizontal velocity has no measured launch angle.
    # Preserve that missing value for the existing q_score(None) policy.
    angle_error = (max(abs(angle30 - 30.0), abs(angle60 - 60.0))
                   if angle30 is not None and angle60 is not None else None)
    level_error = max(f30["landing_level_error_d"], f60["landing_level_error_d"])
    integrity_parts = []
    for slot, result in ball_results.items():
        radius_cv = result["measurements"]["radius_robust_cv"]
        disagreement = tracking_qc["balls"][slot]["backend_disagreement_median_radii"]
        coverage_error = 1.0 - tracking_qc["balls"][slot]["fused_coverage"]
        integrity_parts.extend([
            q_score(radius_cv, {"good": 0.10, "bad": 0.45}),
            q_score(disagreement, {"good": 0.15, "bad": 0.75}),
            q_score(coverage_error, {"good": 0.02, "bad": 0.15}),
        ])
    integrity_score = float(np.mean(integrity_parts))
    task = (
        0.25 * q_score(sync_frames, thresholds["sync_frames"])
        + 0.25 * q_score(angle_error, thresholds["angle_error_deg"])
        + 0.20 * q_score(level_error, thresholds["landing_level_d"])
        + 0.15 * q_score(camera["max_translation_diag"], thresholds["camera_translation_diag"])
        + 0.15 * integrity_score
    )

    para_scores, horizontal_scores, vertical_scores = [], [], []
    for result in ball_results.values():
        metric = result["metrics"]
        para_scores.append(
            0.65 * q_score(metric["parabola_rmse_d"], thresholds["parabola_rmse_d"])
            + 0.35 * q_score(metric["parabola_p95_d"], thresholds["parabola_p95_d"])
        )
        horizontal_scores.append(
            0.60 * q_score(metric["x_fit_rmse_d"], thresholds["x_fit_rmse_d"])
            + 0.40 * q_score(metric["vx_robust_cv"], thresholds["vx_robust_cv"])
        )
        vertical_scores.append(
            0.65 * q_score(metric["y_fit_rmse_d"], thresholds["y_fit_rmse_d"])
            + 0.20 * q_score(metric["vy_line_residual"], thresholds["vy_line_residual"])
            + 0.15 * q_score(metric["ay_robust_cv"], thresholds["ay_robust_cv"])
        )
    parabola = aggregate_balls(*para_scores)
    horizontal = aggregate_balls(*horizontal_scores)
    vertical = aggregate_balls(*vertical_scores)

    r30, r60 = b30["measurements"]["range_diameters"], b60["measurements"]["range_diameters"]
    v30, v60 = b30["measurements"]["v0_diameters_per_s"], b60["measurements"]["v0_diameters_per_s"]
    g30, g60 = b30["measurements"]["g_diameters_per_s2"], b60["measurements"]["g_diameters_per_s2"]
    m1_signed = r30 / r60 - 1.0 if abs(r60) > 1e-9 else float("inf")
    speed_ratio_signed = v30 / v60 - 1.0 if abs(v60) > 1e-9 else float("inf")
    gravity_ratio_signed = g30 / g60 - 1.0 if abs(g60) > 1e-9 else float("inf")
    gravity_symmetric = 2.0 * abs(g30 - g60) / max(abs(g30) + abs(g60), 1e-9)
    pair = (
        0.45 * q_score(abs(m1_signed), thresholds["range_ratio_error"])
        + 0.40 * q_score(abs(speed_ratio_signed), thresholds["initial_speed_error"])
        + 0.15 * q_score(gravity_symmetric, thresholds["gravity_error"])
    )
    scores = {"task": task, "parabola": parabola, "horizontal": horizontal,
              "vertical": vertical, "pair_consistency": pair}
    weights = cfg["score_weights"]
    overall = 100.0
    for name, score in scores.items():
        if score <= 0:
            overall = 0.0
            break
        overall *= (score / 100.0) ** float(weights[name])
    metrics = {
        "M1_signed": m1_signed, "M1_abs": abs(m1_signed),
        "initial_speed_ratio_signed": speed_ratio_signed,
        "initial_speed_error": abs(speed_ratio_signed),
        "gravity_ratio_signed": gravity_ratio_signed,
        "gravity_symmetric_error": gravity_symmetric,
        "release_sync_frames": sync_frames, "max_angle_error_deg": angle_error,
        "max_landing_level_error_d": level_error,
    }
    return {**scores, "overall_conditional": overall}, metrics


def render_plot(path: Path, ball_results: dict, debug_fit: dict, scores: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {30: "#19a974", 60: "#ff8c42"}
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    for column, (slot, result) in enumerate(ball_results.items()):
        angle = int(round(result["measurements"]["angle_target_deg"]))
        dbg = debug_fit[slot]
        colour = colours[angle]
        axes[0, column].scatter(dbg["x"], dbg["y"], s=14, color=colour, label="observed")
        order = np.argsort(dbg["x"])
        axes[0, column].plot(dbg["x"][order], dbg["pred_y"][order], color="black", lw=1.5,
                             label="ballistic time fit")
        coeff = dbg["spatial_coeff"]
        grid = np.linspace(dbg["x"].min(), dbg["x"].max(), 200)
        axes[0, column].plot(grid, coeff[0] + coeff[1] * grid + coeff[2] * grid ** 2,
                             "--", color="#6c5ce7", label="free parabola")
        axes[0, column].set(title=f"{slot}: target {angle}°", xlabel="x / ball diameter",
                            ylabel="height / ball diameter")
        axes[0, column].legend(fontsize=8)
        axes[1, column].plot(dbg["t"], dbg["x"], ".", color="#0984e3", label="x")
        axes[1, column].plot(dbg["t"], dbg["pred_x"], "-", color="#2d3436", label="x fit")
        axes[1, column].plot(dbg["t"], dbg["y"], ".", color=colour, label="y")
        axes[1, column].plot(dbg["t"], dbg["pred_y"], "-", color="#636e72", label="y fit")
        axes[1, column].set(xlabel="time after launch (s)", ylabel="diameters")
        axes[1, column].legend(fontsize=8)
        series = dbg.get("series", {})
        if series:
            axes[2, column].plot(series["t"], series["vx"], label="vx", color="#0984e3")
            axes[2, column].plot(series["t"], series["vy"], label="vy", color=colour)
            axes[2, column].plot(series["t"], series["ay"], label="ay", color="#6c5ce7")
        axes[2, column].axhline(0, color="#999", lw=0.7)
        axes[2, column].set(xlabel="time after launch (s)", ylabel="diameters / s or s²")
        axes[2, column].legend(fontsize=8)
    overall = scores.get("overall", scores.get("overall_conditional", 0.0))
    fig.suptitle(
        f"P3 continuous scores: task={scores['task']:.3f}, "
        f"parabola={scores['parabola']:.3f}, "
        f"horizontal={scores['horizontal']:.3f}, "
        f"vertical={scores['vertical']:.3f}, "
        f"pair={scores['pair_consistency']:.3f}, overall={float(overall):.3f}"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_overlay(path: Path, frames: np.ndarray, fps: float, raw_tracks: dict,
                   fused_tracks: dict, flights: dict, ball_results: dict,
                   scores: dict, hard_fails: list[str]) -> None:
    scale = 0.5
    h, w = frames.shape[1:3]
    out_w, out_h = int(round(w * scale)), int(round(h * scale))
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=max(1, int(round(fps))))
    stream.width, stream.height = out_w, out_h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "24", "preset": "veryfast"}
    colours = {30: (60, 220, 120), 60: (30, 150, 255)}
    for index, rgb in enumerate(frames):
        image = cv2.cvtColor(cv2.resize(rgb, (out_w, out_h)), cv2.COLOR_RGB2BGR)
        for slot, fused in fused_tracks.items():
            angle = int(round(ball_results.get(slot, {"measurements": {"angle_target_deg": 0}})
                              ["measurements"]["angle_target_deg"]))
            colour = colours.get(angle, (255, 255, 255))
            for backend, backend_colour in (("sam2", (255, 220, 0)), ("cotracker", (255, 0, 220))):
                track = raw_tracks[slot][backend]
                if track.found[index]:
                    p = tuple(np.round(track.xy[index] * scale).astype(int))
                    cv2.circle(image, p, 3, backend_colour, 1, cv2.LINE_AA)
            if fused.found[index]:
                p = tuple(np.round(fused.xy[index] * scale).astype(int))
                r = max(3, int(round(np.nan_to_num(fused.radius[index], nan=10.0) * scale)))
                cv2.circle(image, p, r, colour, 2, cv2.LINE_AA)
            history = fused.xy[:index + 1]
            good = np.flatnonzero(np.isfinite(history[:, 0]))
            if len(good) > 1:
                pts = np.round(history[good] * scale).astype(np.int32)
                cv2.polylines(image, [pts], False, colour, 1, cv2.LINE_AA)
            if slot in flights:
                for label, key in (("L", "launch_frame"), ("A", "apex_frame"), ("D", "landing_frame")):
                    event = flights[slot][key]
                    if event <= index and fused.found[event]:
                        p = tuple(np.round(fused.xy[event] * scale).astype(int))
                        cv2.putText(image, label, p, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    colour, 1, cv2.LINE_AA)
        overall_text = scores.get("overall", scores.get("overall_conditional"))
        overall_text = 0.0 if overall_text is None else float(overall_text)
        lines = [
            f"P3 frame {index:03d}",
            f"task {scores.get('task', 0):.3f} para {scores.get('parabola', 0):.3f} "
            f"vx {scores.get('horizontal', 0):.3f} ay {scores.get('vertical', 0):.3f}",
            f"pair {scores.get('pair_consistency', 0):.3f} overall {overall_text:.3f}",
        ]
        if hard_fails:
            lines.append("HARD FAIL: " + ", ".join(hard_fails[:2]))
        y = 22
        for line in lines:
            cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (10, 10, 10), 1, cv2.LINE_AA)
            y += 20
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def evaluate(video: Path, output: Path, config_path: Path, debug_dir: Path,
             device: str | None = None, force_tracking: bool = False) -> dict:
    cfg, config_hash = load_config(config_path)
    sample_id, seed_number = sample_id_from_video(video)
    if sample_id not in cfg["samples"]:
        raise KeyError(f"no fixed first-frame annotation for {sample_id}")
    sample_cfg = cfg["samples"][sample_id]
    video_hash = _sha256(video)
    frames, timestamps, fps, video_meta = read_video(video)
    video_meta["sha256"] = video_hash
    debug_dir.mkdir(parents=True, exist_ok=True)

    tracks_raw, cot_shift, worker_meta = run_neural_tracking(
        frames, sample_cfg, cfg, debug_dir, video_hash, config_hash,
        device or cfg["runtime"]["default_device"], force_tracking
    )
    transform, camera = estimate_camera_similarity(frames, cot_shift)
    tracks = {
        slot: {backend: compensate_track(track, transform)
               for backend, track in by_backend.items()}
        for slot, by_backend in tracks_raw.items()
    }
    fused, tracking_ball_qc, colour_qc = {}, {}, {}
    for slot, seed in sample_cfg["balls"].items():
        fused[slot], fusion = fuse_tracks(tracks[slot]["sam2"], tracks[slot]["cotracker"], seed)
        source = fusion.pop("source")
        tracking_ball_qc[slot] = fusion
        colour_qc[slot] = local_colour_validation(frames, fused[slot], seed)
        # source is debug-only and not serialized into the public JSON.
        np.save(debug_dir / f"{slot}_fusion_source.npy", source)

    extract_success = True
    measurement_valid = True
    structural_ok = True
    hard_fails: list[str] = []
    warnings: list[str] = []
    gates = cfg["gates"]
    if camera["max_translation_diag"] > gates["max_camera_translation_diag"]:
        structural_ok = False; hard_fails.append("CAMERA_TRANSLATION")
    if camera["max_scale_error"] > gates["max_camera_scale_error"]:
        structural_ok = False; hard_fails.append("CAMERA_SCALE_CHANGE")
    if camera["max_rotation_deg"] > gates["max_camera_rotation_deg"]:
        structural_ok = False; hard_fails.append("CAMERA_ROTATION")

    flights, flight_reasons = {}, {}
    for slot, seed in sample_cfg["balls"].items():
        qc = tracking_ball_qc[slot]
        if max(qc["sam2_coverage"], qc["cotracker_coverage"]) < gates["min_primary_coverage_overall"]:
            extract_success = False
            hard_fails.append(f"{slot.upper()}_NOT_OBSERVABLE")
        if qc["fused_coverage"] < gates["min_fused_coverage_overall"]:
            measurement_valid = False
            warnings.append(f"{slot}: low fused coverage {qc['fused_coverage']:.3f}")
        overlap = qc["backend_overlap_frames"]
        disagreement = qc["backend_disagreement_median_radii"]
        colour_support = colour_qc[slot]["coverage_on_guide"]
        if overlap < gates["min_backend_overlap_frames"] and colour_support < 0.60:
            measurement_valid = False
            warnings.append(f"{slot}: insufficient independent tracker support")
        if disagreement is not None and disagreement > gates["max_backend_disagreement_radii"]:
            measurement_valid = False
            warnings.append(f"{slot}: tracker disagreement {disagreement:.3f} radii")
        flight, reason = extract_flight(fused[slot], seed, timestamps, cfg)
        flight_reasons[slot] = reason
        if flight is None:
            if reason == "ball_never_released":
                structural_ok = False; hard_fails.append(f"{slot.upper()}_NO_RELEASE")
            elif extract_success:
                structural_ok = False; hard_fails.append(f"{slot.upper()}_NO_COMPLETE_LANDING")
            continue
        flights[slot] = flight
        if flight["inner_coverage"] < gates["min_fused_coverage_flight"]:
            measurement_valid = False
            warnings.append(f"{slot}: low flight coverage {flight['inner_coverage']:.3f}")
        if flight["max_gap"] > gates["max_internal_gap_frames"]:
            measurement_valid = False
            warnings.append(f"{slot}: internal gap {flight['max_gap']} frames")
        range_d = (flight["x_land_px"] - seed["cx"]) / (2 * seed["radius"])
        if range_d < gates["min_horizontal_span_diameters"]:
            structural_ok = False; hard_fails.append(f"{slot.upper()}_INSUFFICIENT_RANGE")
        if flight["landing_level_error_d"] > gates["max_landing_level_error_diameters"]:
            structural_ok = False; hard_fails.append(f"{slot.upper()}_WRONG_LANDING_LEVEL")
        margin = gates["border_margin_radii"] * seed["radius"]
        segment = fused[slot].xy[flight["launch_frame"]:flight["landing_frame"] + 1]
        observed = segment[np.isfinite(segment[:, 0])]
        if len(observed) and np.any((observed[:, 0] < margin) | (observed[:, 0] > frames.shape[2] - margin)
                                    | (observed[:, 1] < margin) | (observed[:, 1] > frames.shape[1] - margin)):
            structural_ok = False; hard_fails.append(f"{slot.upper()}_BORDER_CROP")

    ball_results, debug_fit = {}, {}
    if len(flights) == 2:
        for slot, seed in sample_cfg["balls"].items():
            ball_results[slot], debug_fit[slot] = measure_ball(
                fused[slot], seed, flights[slot], timestamps
            )
            if ball_results[slot]["metrics"]["parabola_quadratic"] >= 0:
                hard_fails.append(f"{slot.upper()}_NON_DOWNWARD_PARABOLA")
            if ball_results[slot]["measurements"]["g_diameters_per_s2"] <= 0:
                hard_fails.append(f"{slot.upper()}_NO_DOWNWARD_ACCELERATION")
        tracking_qc = {"fps": fps, "balls": tracking_ball_qc}
        scores, pair_metrics = score_results(ball_results, flights, tracking_qc, camera, cfg)
    else:
        measurement_valid = False
        scores = {"task": 0.0, "parabola": 0.0, "horizontal": 0.0,
                  "vertical": 0.0, "pair_consistency": 0.0,
                  "overall_conditional": None}
        pair_metrics = {}

    if any(reason.endswith("NON_DOWNWARD_PARABOLA") or reason.endswith("NO_DOWNWARD_ACCELERATION")
           for reason in hard_fails):
        structural_ok = False
    hard_fails = sorted(set(hard_fails))
    warnings = sorted(set(warnings))
    conditional = scores.get("overall_conditional")
    end_to_end = float(conditional) if (
        extract_success and measurement_valid and structural_ok and conditional is not None
    ) else 0.0
    rules = cfg["pass_rules"]
    core = [scores.get(k, 0.0) or 0.0 for k in
            ("parabola", "horizontal", "vertical", "pair_consistency")]
    physics_pass = bool(
        extract_success and measurement_valid and structural_ok
        and conditional is not None and conditional >= rules["overall_min"]
        and scores["task"] >= rules["task_min"]
        and min(core) >= rules["core_dimension_min"]
        and pair_metrics.get("M1_abs", float("inf")) <= rules["range_error_max"]
        and pair_metrics.get("initial_speed_error", float("inf")) <= rules["initial_speed_error_max"]
    )

    first_frame = (config_path.parent / sample_cfg["first_frame"]).resolve()
    first_frame_psnr = None
    if first_frame.exists():
        expected = cv2.cvtColor(cv2.imread(str(first_frame)), cv2.COLOR_BGR2RGB)
        if expected.shape == frames[0].shape:
            mse = float(np.mean((expected.astype(np.float32) - frames[0].astype(np.float32)) ** 2))
            first_frame_psnr = float("inf") if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))
    result = {
        "schema_version": cfg["schema_version"],
        "evaluator_version": cfg["evaluator_version"], "config_sha256": config_hash,
        "task_id": "P3", "sample_id": sample_id, "seed": seed_number,
        "first_frame_method": sample_cfg["method"], "video": str(video),
        "video_sha256": video_hash, "extract_success": extract_success,
        "structural_ok": structural_ok, "measurement_valid": measurement_valid,
        "physics_pass": physics_pass,
        "metric_validity": {
            "M1": bool(measurement_valid and len(ball_results) == 2),
            "M2": bool(measurement_valid and len(ball_results) == 2),
            "M3": False,
        },
        "measurements": {
            "balls": {slot: value["measurements"] for slot, value in ball_results.items()},
            "events": flights, "first_frame_psnr_db": first_frame_psnr,
        },
        "metrics": {
            "M1": pair_metrics.get("M1_signed"), "M2": {
                "per_ball": {slot: value["metrics"] for slot, value in ball_results.items()},
                "initial_speed_error": pair_metrics.get("initial_speed_error"),
            }, "M3": None, **pair_metrics,
        },
        "scores": {**scores, "end_to_end": end_to_end},
        "qc": {
            "video": video_meta, "camera": camera, "tracking_worker": worker_meta,
            "balls": tracking_ball_qc, "colour_validation": colour_qc,
            "flight_extract_reasons": flight_reasons,
        },
        "hard_fail_reasons": hard_fails, "warnings": warnings,
        "failure_reason": "; ".join(hard_fails or warnings) or None,
        "debug": {
            "overlay_video": str(debug_dir / "overlay.mp4"),
            "plot": str(debug_dir / "plot.png"),
            "raw_tracks": str(debug_dir / "tracks_raw.npz"),
        },
    }
    # Keep the legacy 0--100 decision procedure above intact so historical
    # physics/structure labels remain comparable, then expose the graded
    # continuous 0--1 score profile.  Only measurement-invalid samples receive
    # an overall score of exactly zero; measurable hard failures retain a
    # positive, continuously varying quality score.
    result = rescore_result(result, cfg)
    scores = result["scores"]
    safe = _json_safe(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(safe, ensure_ascii=False, indent=2))
    if len(ball_results) == 2:
        render_plot(debug_dir / "plot.png", ball_results, debug_fit, scores)
    render_overlay(debug_dir / "overlay.mp4", frames, fps, tracks, fused, flights,
                   ball_results, scores, hard_fails)
    return safe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--task_id", default="P3")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--force-tracking", action="store_true")
    args = parser.parse_args()
    if args.task_id != "P3":
        raise ValueError(f"this evaluator only supports P3, got {args.task_id}")
    debug = args.debug_dir or args.output.parent.parent / "debug" / args.video.stem
    result = evaluate(args.video.resolve(), args.output.resolve(), args.config.resolve(),
                      debug.resolve(), args.device, args.force_tracking)
    print(json.dumps({
        "sample_id": result["sample_id"], "extract_success": result["extract_success"],
        "structural_ok": result["structural_ok"], "measurement_valid": result["measurement_valid"],
        "overall": result["scores"]["end_to_end"], "pass": result["physics_pass"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
