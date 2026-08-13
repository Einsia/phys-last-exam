#!/usr/bin/env python
"""Probe the boundaries where the chain is most likely to misbehave.

A real clip can easily land on any of these, and each one has a different correct
answer. What is being checked is not that everything passes -- it is that each case is
classified into the right bucket, because the split between "unmeasurable" (lowers PMR)
and "wrong" (lowers PPR, raises HVR) is what stops a model from scoring well by
generating clips that cannot be measured.

Runs against the default (learned) tracker pair, because these verdicts have to hold
for the configuration that actually scores real video. `--backends color,bgsub` checks
the classic pair, which is ~5x faster and needs no GPU.

Usage:
  envs/physbench/bin/python scripts/edge_cases.py
  envs/physbench/bin/python scripts/edge_cases.py --backends color,bgsub
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physbench.config import TaskConfig  # noqa: E402
from physbench.first_frame import BALL_RGB, BallSeed, draw_ball, render_scene  # noqa: E402
from physbench.io_video import Clip  # noqa: E402
from physbench.render_synth import build, ideal, render  # noqa: E402
from physbench.tasks.projectile import evaluate_projectile  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backends", default=None, help="comma-separated pair, e.g. color,bgsub")
ap.add_argument("--device", default=None, help="CUDA_VISIBLE_DEVICES for the worker")
args = ap.parse_args()

cfg = TaskConfig()
if args.backends:
    cfg.tracking.backends = tuple(args.backends.split(","))
if args.device:
    cfg.tracking.device = args.device
print(f"tracker backends: {cfg.tracking.backends}\n")
W, H, GF = 1344, 768, 0.86
RAD = max(6.0, 0.022 * H)
FAILURES: list[str] = []


def run(label: str, frames, seed, theta=45.0, measurable: bool | None = None,
        contains: str = ""):
    """Assert one clip's bucket.

    `measurable` is checked against the boolean, not against a substring of the
    description: "measurable" is a substring of "not measurable", so a substring test
    passes on the exact regression it is supposed to catch.
    """
    res, _ = evaluate_projectile(Clip(frames=frames, fps=24.0, path=f"<{label}>"),
                                seed, theta, sample_id=label, model="edge", cfg=cfg)
    prim = [r for r in res.residuals if r.primary]
    got = ("not measurable: " + ", ".join(res.gate_reasons)) if not res.measurable else (
        f"measurable, {len(prim)} primary, "
        + ("PASS" if res.passed_all else "FAIL")
        + (f", viol={','.join(v.name for v in res.violations)}" if res.violations else ""))

    bad = []
    if measurable is not None and res.measurable != measurable:
        bad.append(f"measurable={measurable}")
    if contains and contains not in got:
        bad.append(f"containing {contains!r}")
    print(f"{'!! ' if bad else 'OK '}{label:<26} {got}")
    if bad:
        print(f"     expected {' and '.join(bad)}")
        FAILURES.append(f"{label}: expected {' and '.join(bad)}, got {got!r}")
    return res


print("short flights (gate is min_flight_frames=12)\n")
for nf in (8, 12, 14, 20):
    frames, seed, _ = build("gt", theta_deg=45.0, n_flight=nf)
    run(f"flight_{nf}_frames", frames, seed, measurable=nf >= 12)

print("\nball never lands (clip ends mid-flight)\n")
# Truncate a gt clip just after the apex: the ball is still airborne and still in frame,
# so this is a physics-shaped failure, not an observability one -- unless it left the view.
frames, seed, traj = build("gt", theta_deg=45.0, n_flight=60)
run("truncated_after_apex", frames[:46], seed, measurable=False,
    contains="flight_not_contained")

print("\nball leaves the frame (observability, must lower PMR not PPR)\n")
# Launch steeply enough that the apex is outside the canvas.
traj2 = ideal(R=0.62 * W, theta_deg=80.0, n_flight=60, x0=0.10 * W,
              y_ref=H * GF - RAD, mode="gt")
f2 = render(traj2, W, H, RAD, GF)
# The gate that fires here differs by backend and both are observability failures: the
# classic pair reports detection_gap(6) -- at 80 deg the ball crosses the border band in
# one step, so it is never seen near an edge -- while SAM2 simply stops returning the
# object, which shows up as low_coverage. Only the bucket is asserted.
run("apex_above_canvas", f2, BallSeed(cx=traj2.x[0], cy=traj2.y[0], radius=RAD, rgb=BALL_RGB),
    theta=80.0, measurable=False)

print("\nball absent entirely (nothing to track)\n")
empty, _ = render_scene(W, H, ground_frac=GF, seed=0)
run("no_ball_at_all", np.repeat(empty[None], 40, axis=0),
    BallSeed(cx=0.1 * W, cy=H * GF - RAD, radius=RAD, rgb=BALL_RGB),
    measurable=False, contains="tracker_disagreement")

print("\nball at rest for the whole clip (no launch)\n")
# The one case whose bucket depends on the tracker, and the difference is instructive.
#
# The classic pair reports tracker_disagreement(inf) and the clip is not measurable --
# but only because bgsub is blind to a stationary ball by construction: it becomes its
# own temporal-median background, so coverage goes to zero. The verdict is right for a
# reason that has nothing to do with observability.
#
# The learned pair tracks the static ball at full coverage with both backends agreeing,
# so the clip is measurable and carries a hard violation. That is where this repo's own
# rule puts it (README, "the measurability gate, and why it is split"): tracked
# perfectly, in frame throughout, and still not doing what was asked. It also gives the
# right incentive -- a model that emits static clips must not be able to lower PMR
# instead of PPR, which is exactly what proposal.md 3.3 warns about.
still, _ = render_scene(W, H, ground_frac=GF, seed=0)
draw_ball(still, 0.1 * W, H * GF - RAD, RAD, BALL_RGB)
run("never_launches", np.repeat(still[None], 40, axis=0),
    BallSeed(cx=0.1 * W, cy=H * GF - RAD, radius=RAD, rgb=BALL_RGB),
    measurable=cfg.tracking.is_neural,
    contains="ball_never_moved" if cfg.tracking.is_neural else "tracker_disagreement")

print("\npanning camera (drift must be caught, not absorbed)\n")
frames, seed, _ = build("gt", theta_deg=45.0, n_flight=60)
rolled = np.stack([np.roll(f, int(round(0.55 * i)), axis=1) for i, f in enumerate(frames)])
run("camera_pan_0.55px_per_frame", rolled, seed, measurable=False,
    contains="camera_drift")

if FAILURES:
    print(f"\n{len(FAILURES)} edge case(s) landed in the wrong bucket:")
    for f in FAILURES:
        print(f"  {f}")
    raise SystemExit(1)
print("\nall edge cases classified as intended")
