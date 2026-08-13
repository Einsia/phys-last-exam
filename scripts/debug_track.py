#!/usr/bin/env python
"""Per-frame track vs constructed truth, for diagnosing a synthetic mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physbench.render_synth import build  # noqa: E402
from physbench.tracking import CLASSIC_BACKENDS, run_tracks  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="gt")
ap.add_argument("--blur", type=int, default=1)
ap.add_argument("--theta", type=float, default=45.0)
ap.add_argument("--n-flight", type=int, default=60)
# Classic by default: this dump is a per-frame comparison against constructed truth,
# and it should not need a GPU to answer "is the tracker where the ball is".
ap.add_argument("--backends", default=",".join(CLASSIC_BACKENDS))
args = ap.parse_args()

frames, seed, traj = build(args.mode, theta_deg=args.theta, n_flight=args.n_flight,
                           blur_substeps=args.blur)
ts = run_tracks(frames, seed, backends=tuple(args.backends.split(",")))
ta, tb = ts.pair
print(f"mode={args.mode} blur={args.blur} frames={len(frames)} radius={seed.radius:.1f}")
print("coverage " + "  ".join(f"{b}={t.coverage:.3f}" for b, t in ts.tracks.items()))
print(f"launch={traj.launch} land={traj.land}")
print(f"\n{'n':>4} {'true_x':>8} {'true_y':>8} {'cx':>8} {'cy':>8} {'err':>7} "
      f"{'rad':>6} {'score':>6} {'bx':>8} {'by':>8} {'step':>7}")
prev = None
for i in range(len(frames)):
    tx, ty = traj.x[i], traj.y[i]
    cx, cy = ta.xy[i]
    bx, by = tb.xy[i]
    err = np.hypot(cx - tx, cy - ty) if np.isfinite(cx) else float("nan")
    step = np.hypot(tx - prev[0], ty - prev[1]) if prev else 0.0
    prev = (tx, ty)
    print(f"{i:>4} {tx:>8.2f} {ty:>8.2f} {cx:>8.2f} {cy:>8.2f} {err:>7.2f} "
          f"{ta.radius[i]:>6.2f} {ta.score[i]:>6.3f} {bx:>8.2f} {by:>8.2f} {step:>7.2f}")
