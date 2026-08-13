#!/usr/bin/env python
"""Verify the chain across the angle range, not just at 45 degrees.

Two independent checks per angle. First an analytic one, straight off the constructed
arrays with no tracking involved: does the renderer's parameterisation actually satisfy
H/R = tan(theta)/4? If it did not, every synthetic validation would be resting on a
wrong fixture and would prove nothing. Second the full render -> track -> measure chain,
so the noise floor is known across the range rather than at one convenient angle.

Usage:
  envs/physbench/bin/python scripts/sweep_angles.py
  envs/physbench/bin/python scripts/sweep_angles.py --backends color,bgsub
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physbench.config import TaskConfig  # noqa: E402
from physbench.io_video import Clip  # noqa: E402
from physbench.render_synth import build  # noqa: E402
from physbench.tasks.projectile import evaluate_projectile  # noqa: E402

KEYS = ("M1_HR_nominal", "M2_time_symmetry", "M4_parabolicity", "M5_vx_conservation")

ap = argparse.ArgumentParser()
ap.add_argument("--backends", default=None, help="comma-separated pair, e.g. color,bgsub")
ap.add_argument("--device", default=None, help="CUDA_VISIBLE_DEVICES for the worker")
args = ap.parse_args()

cfg = TaskConfig()
if args.backends:
    cfg.tracking.backends = tuple(args.backends.split(","))
if args.device:
    cfg.tracking.device = args.device

failures: list[str] = []
print(f"tracker backends: {cfg.tracking.backends}\n")
print(f"{'theta':>6}  {'analytic H/R':>13}{'tan/4':>9}{'rel err':>10}   "
      f"{'measured H/R':>13}{'M1':>8}{'M2':>8}{'M4':>8}{'M5':>8}  verdict")
worst = 0.0
for th in (20, 30, 45, 60, 70):
    frames, seed, traj = build("gt", theta_deg=float(th), n_flight=60)
    lo, hi = traj.launch, traj.land
    H = traj.y[lo] - traj.y[lo:hi + 1].min()
    R = abs(traj.x[hi] - traj.x[lo])
    theory = float(np.tan(np.radians(th)) / 4)

    res, _ = evaluate_projectile(Clip(frames=frames, fps=24.0, path=f"<gt{th}>"),
                                seed, float(th), sample_id=f"gt_theta{th}",
                                model="synthetic", cfg=cfg)
    e = {r.name: r.residual for r in res.residuals}
    worst = max(worst, max(e.get(k, 0.0) for k in KEYS))
    print(f"{th:>6}  {H / R:>13.5f}{theory:>9.5f}{abs(H / R / theory - 1):>10.2e}   "
          f"{res.measured['H_over_R']:>13.5f}"
          + "".join(f"{e.get(k, float('nan')):>8.4f}" for k in KEYS)
          + f"  {'PASS' if res.passed_all else 'FAIL'}")
    # The fixture must be exact, not merely close: everything synthetic rests on it.
    if abs(H / R / theory - 1) > 1e-12:
        failures.append(f"theta={th}: renderer H/R off theory by {abs(H / R / theory - 1):.2e}")
    if not res.passed_all:
        failures.append(f"theta={th}: exact-physics render did not pass every primary")

print(f"\nworst primary residual across the range: {worst:.4f}")
print(f"tolerances: hr={cfg.tol.hr_nominal} tsym={cfg.tol.time_symmetry} "
      f"para={cfg.tol.parabolicity} vx={cfg.tol.vx_conservation}")
if failures:
    print(f"\n{len(failures)} problem(s):")
    for f in failures:
        print(f"  {f}")
    raise SystemExit(1)
print("fixture exact and chain unbiased across the range")
