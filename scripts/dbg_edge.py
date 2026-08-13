#!/usr/bin/env python
"""Inspect why a specific edge case classified the way it did."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physbench.first_frame import BALL_RGB, BallSeed  # noqa: E402
from physbench.kinematics import extract_flight  # noqa: E402
from physbench.render_synth import build, ideal, render  # noqa: E402
from physbench.tracking import CLASSIC_BACKENDS, run_tracks  # noqa: E402

W, H, GF = 1344, 768, 0.86
RAD = max(6.0, 0.022 * H)

print("=== apex_above_canvas (theta=80) ===")
traj = ideal(R=0.62 * W, theta_deg=80.0, n_flight=60, x0=0.10 * W, y_ref=H * GF - RAD, mode="gt")
frames = render(traj, W, H, RAD, GF)
seed = BallSeed(cx=traj.x[0], cy=traj.y[0], radius=RAD, rgb=BALL_RGB)
print(f"constructed y range: {traj.y.min():.1f} .. {traj.y.max():.1f}  (canvas 0..{H})")
ts = run_tracks(frames, seed, backends=CLASSIC_BACKENDS)
ta, tb = ts.pair
print("coverage " + "  ".join(f"{b}={t.coverage:.3f}" for b, t in ts.tracks.items()))
found = np.flatnonzero(ta.found)
print(f"found frames: {found.min()}..{found.max()}, gaps at "
      f"{sorted(set(range(found.min(), found.max() + 1)) - set(found.tolist()))[:12]}")
print(f"last found position: {ta.xy[found[-1]]}")
print(f"min tracked y: {np.nanmin(ta.xy[:, 1]):.1f}")
f, r = extract_flight(ta.xy, seed.radius)
print(f"extract_flight -> {r}")

print("\n=== flight_8_frames ===")
frames2, seed2, traj2 = build("gt", theta_deg=45.0, n_flight=8)
t2 = run_tracks(frames2, seed2, backends=CLASSIC_BACKENDS).tracks["color"]
f2, r2 = extract_flight(t2.xy, seed2.radius)
print(f"coverage={t2.coverage:.3f}  extract_flight -> {r2}")
fnd = np.flatnonzero(t2.found)
print(f"last found position: {t2.xy[fnd[-1]]}  (canvas {W}x{H}, margin={1.5 * seed2.radius:.1f})")
