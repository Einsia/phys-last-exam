#!/usr/bin/env python
"""Per-frame centroid error of every backend against exact synthetic truth.

This is the evidence behind the default backend choice, and it is worth having as a
script rather than a claim in a docstring: the synthetic renderer knows where it put
the ball to sub-pixel accuracy, so tracker error can be measured directly instead of
being inferred from the residuals it eventually produces.

What to read, and in what order:

  bias     constant offset. Nearly irrelevant here -- every invariant in the task is a
           ratio of pixel differences, so a fixed offset cancels. Reported because a
           LARGE bias still says the backend is not centring on the ball.
  sd       frame-to-frame scatter with the bias removed. This is what reaches the
           residuals and therefore what sets the noise floor.
  corr     correlation of vertical error with speed. The one bias that does not
           cancel: an error that grows with speed is asymmetric between ascent and
           descent, which is exactly what M5 and M2 measure.

Run it on the clean render AND the blurred one. The ordering between backends is not
the same in both, and the blurred case is the one that resembles real video.

Usage:
  envs/physbench/bin/python scripts/tracker_precision.py
  envs/physbench/bin/python scripts/tracker_precision.py --blur 3 --theta 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physbench.config import TrackingConfig  # noqa: E402
from physbench.neural_track import TrackerBackendConfig  # noqa: E402
from physbench.render_synth import build  # noqa: E402
from physbench.tracking import run_tracks, track_bgsub, track_color  # noqa: E402

ALL = ("sam2", "cotracker", "color", "bgsub")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=45.0)
    ap.add_argument("--n-flight", type=int, default=60)
    ap.add_argument("--blur", type=int, default=1, help="motion-blur substeps")
    ap.add_argument("--device", default=None, help="CUDA_VISIBLE_DEVICES for the worker")
    ap.add_argument("--backends", default=",".join(ALL))
    ap.add_argument("--out", default=None, help="write the table as JSON too")
    args = ap.parse_args()

    want = tuple(b.strip() for b in args.backends.split(",") if b.strip())
    frames, seed, traj = build("gt", theta_deg=args.theta, n_flight=args.n_flight,
                               blur_substeps=args.blur)
    truth = np.stack([traj.x, traj.y], axis=1)
    lo, hi = traj.launch, traj.land
    dev = args.device or TrackingConfig().device
    print(f"gt render: theta={args.theta:g} blur={args.blur} frames={len(frames)} "
          f"radius={seed.radius:.2f}px  flight {lo}..{hi}")

    tracks: dict[str, np.ndarray] = {}
    timing: dict[str, float] = {}
    # Neural backends share one subprocess; classic ones are measured individually so
    # the reported seconds are per-backend in both cases.
    neural = tuple(b for b in want if b in ("sam2", "sam3", "cotracker"))
    classic = tuple(b for b in want if b not in ("sam2", "sam3", "cotracker"))
    if len(neural) == 2:
        t0 = time.time()
        ts = run_tracks(frames, seed, backends=neural,
                        backend_cfg=TrackerBackendConfig(devices=dev))
        dt = time.time() - t0
        for b, t in ts.tracks.items():
            tracks[b], timing[b] = t.xy, dt / 2
    elif neural:
        raise SystemExit("pass two neural backends (e.g. sam3,cotracker), or neither")
    # Called directly rather than through run_tracks, which always runs a pair: timing
    # the pair and attributing it to one backend would overstate each by ~2x.
    for b in classic:
        fn = {"color": track_color, "bgsub": track_bgsub}.get(b)
        if fn is None:
            raise SystemExit(f"unknown backend {b!r}")
        t0 = time.time()
        tracks[b] = fn(frames, seed).xy
        timing[b] = time.time() - t0

    rows = []
    for b in want:
        xy = tracks[b]
        seg, tr = xy[lo:hi + 1], truth[lo:hi + 1]
        ok = np.isfinite(seg[:, 0])
        err = np.linalg.norm(seg[ok] - tr[ok], axis=1)
        e = seg[ok] - tr[ok]
        bias = np.median(e, axis=0)
        e0 = e - bias
        speed = np.concatenate([[0.0], np.linalg.norm(np.diff(truth, axis=0), axis=1)])
        sp = speed[lo:hi + 1][ok]
        corr = float(np.corrcoef(e0[:, 1], sp)[0, 1]) if len(sp) > 3 else float("nan")
        rows.append({"backend": b, "coverage": float(ok.mean()),
                     "med_err": float(np.median(err)), "p90_err": float(np.percentile(err, 90)),
                     "bias_x": float(bias[0]), "bias_y": float(bias[1]),
                     "sd_x": float(e0[:, 0].std()), "sd_y": float(e0[:, 1].std()),
                     "corr_err_speed": corr, "seconds": timing.get(b, float("nan"))})

    print(f"\n{'backend':<11}{'cov':>5}{'med|e|':>8}{'p90':>7}"
          f"{'bias_x':>8}{'bias_y':>8}{'sd_x':>7}{'sd_y':>7}{'corr':>7}{'sec':>7}")
    print("-" * 72)
    for r in rows:
        print(f"{r['backend']:<11}{r['coverage']:>5.2f}{r['med_err']:>8.3f}{r['p90_err']:>7.3f}"
              f"{r['bias_x']:>+8.3f}{r['bias_y']:>+8.3f}{r['sd_x']:>7.3f}{r['sd_y']:>7.3f}"
              f"{r['corr_err_speed']:>7.2f}{r['seconds']:>7.1f}")
    print("\nsd (bias removed) is the number that reaches the residuals; bias cancels\n"
          "in every ratio invariant. Compare --blur 1 against --blur 3 before drawing\n"
          "any conclusion about which backend is more precise.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"theta": args.theta, "blur": args.blur, "radius_px": seed.radius,
             "rows": rows}, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
