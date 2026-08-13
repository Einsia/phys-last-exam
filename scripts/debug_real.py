#!/usr/bin/env python
"""Dump the track and flight-extraction internals for one real video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physbench.first_frame import SampleSpec  # noqa: E402
from physbench.io_video import read_clip  # noqa: E402
from physbench.config import TrackingConfig  # noqa: E402
from physbench.kinematics import _moving_run, compensate, extract_flight  # noqa: E402
from physbench.tracking import (estimate_camera_shift, run_tracks,  # noqa: E402
                               track_disagreement)

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--spec", required=True)
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--backends", default=",".join(TrackingConfig().backends),
                help="comma-separated pair; defaults to the configured default")
args = ap.parse_args()

spec = SampleSpec.load(args.spec)
clip = read_clip(args.video)
seed = spec.ball
print(f"{Path(args.video).name}: {clip.n} frames {clip.width}x{clip.height} fps={clip.fps}")
print(f"seed ball: ({seed.cx:.1f},{seed.cy:.1f}) r={seed.radius:.1f} rgb={seed.rgb}")

ts = run_tracks(clip.frames, seed, backends=tuple(args.backends.split(",")))
ta, tb = ts.pair
print("coverage " + "  ".join(f"{b}={t.coverage:.3f}" for b, t in ts.tracks.items())
      + f"  disagree={track_disagreement(ta, tb, seed.radius):.3f} radii")
primary = ts.primary
if ts.camera_shift is not None:
    shift, src = ts.camera_shift, "cotracker bg grid"
else:
    shift, src = estimate_camera_shift(clip.frames, primary, seed), "phase correlation"
print(f"camera drift max = {np.max(np.linalg.norm(shift, axis=1)):.2f} px "
      f"({np.max(np.linalg.norm(shift, axis=1)) / clip.diag:.4f} of diagonal) via {src}")

xy = compensate(primary.xy, shift)
idx = np.flatnonzero(np.isfinite(xy[:, 0]))
if len(idx) >= 3:
    tf = idx.astype(float)
    x, y = xy[idx, 0], xy[idx, 1]
    sp = np.hypot(np.gradient(x, tf), np.gradient(y, tf))
    ka = int(np.argmin(y))
    peak = float(np.nanmax(sp))
    thr = max(0.18 * peak, 0.015 * seed.radius)
    k0, k1 = _moving_run(sp, ka, thr)
    print(f"peak_speed={peak:.2f} thr={thr:.2f} apex at frame {idx[ka]} (y={y[ka]:.1f})")
    print(f"moving run frames {idx[k0]}..{idx[k1]}")

f, reason = extract_flight(xy, seed.radius)
print(f"extract_flight -> {reason}")
if f:
    print(f"  t_launch={f.t_launch:.2f} t_apex={f.t_apex:.2f} t_land={f.t_land:.2f}")
    print(f"  y_ref={f.y_ref:.2f} y_apex={f.y_apex:.2f} H={f.H:.1f} R={f.R:.1f} H/R={f.H / f.R:.4f}")
    print(f"  launch_extrap={f.launch_extrap_frames:.2f} land_extrap={f.land_extrap_frames:.2f}")

print(f"\n{'n':>4} {'x':>8} {'y':>8} {'rad':>6} {'sc':>6} {'speed':>7} {'bx':>8} {'by':>8}")
prev = None
for i in range(0, clip.n, args.every):
    cx, cy = xy[i]
    v = np.hypot(cx - prev[0], cy - prev[1]) if (prev and np.isfinite(cx)) else float("nan")
    if np.isfinite(cx):
        prev = (cx, cy)
    print(f"{i:>4} {cx:>8.2f} {cy:>8.2f} {primary.radius[i]:>6.2f} {primary.score[i]:>6.3f} "
          f"{v:>7.2f} {tb.xy[i, 0]:>8.2f} {tb.xy[i, 1]:>8.2f}")
