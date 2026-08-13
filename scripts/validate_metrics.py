#!/usr/bin/env python
"""Score synthetic clips whose physics is known by construction.

This is the calibration step for the whole chain. `gt` is correct by construction, so
whatever residual it reports is the noise floor of tracking plus fitting, and the
tolerances have to sit clearly above it. The other modes each break one thing, and
the table shows whether the intended invariant fires while the others stay quiet.

  gt            everything passes, residuals at the noise floor
  no_gravity    triangular path: H/R doubles, timing still symmetric
  time_warp     same arc, ascent stretched: timing breaks, shape holds
  wrong_angle   perfect physics at 30 deg when asked for theta: only M1_nominal fires
  const_speed   correct parabolic path at uniform speed: only parabolicity-in-time sees it

The reported noise floor belongs to whichever tracker pair ran, which is the point of
`--backends`: the tolerances have to be checked against the tracker actually in use, not
against a number baked into a docstring.

Usage:
  envs/physbench/bin/python scripts/validate_metrics.py [--theta 45]
  envs/physbench/bin/python scripts/validate_metrics.py --backends color,bgsub
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path



sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physbench.config import TaskConfig  # noqa: E402
from physbench.io_video import Clip, read_clip, write_clip  # noqa: E402
from physbench.metrics import aggregate  # noqa: E402
from physbench.render_synth import build  # noqa: E402
from physbench.tasks.projectile import evaluate_projectile  # noqa: E402

MODES = ["gt", "no_gravity", "time_warp", "wrong_angle", "const_speed"]
# What each mode should do to each primary invariant.
#   pass / fail  asserted against the configured tolerance
#   detect       asserted only to sit well clear of the noise floor, not against the
#                tolerance. const_speed is the one case that needs this: a correct
#                parabolic PATH traversed at uniform speed is only ~2% from parabolic
#                in normalised RMS at 45 deg, because speed varies by just 1/cos over
#                the arc. The residual sees it at ~14x the floor, but whether it should
#                cross the threshold is a policy choice to calibrate on real video --
#                not something to tune until this ablation goes red.
#   absent       the invariant must be skipped entirely (its precondition fails)
EXPECT = {
    "gt":          {"M1_HR_nominal": "pass", "M2_time_symmetry": "pass",
                    "M4_parabolicity": "pass", "M5_vx_conservation": "pass",
                    "M7_accel_geometry": "pass"},
    # Triangular path: v_x is untouched, so M5 must stay quiet. Each leg is linear, so
    # per-half a_y is ~0 and M6 has nothing to compare -- it must be skipped, not
    # allowed to report a ratio of two zeros. M7 does NOT fire here and is asserted to
    # pass on purpose: the tent biases a_y low and (via apex rounding) H low too, and
    # they cancel to 7.97 vs 8. That measured insensitivity is why M7 is a diagnostic
    # and not a gate. M1 and M4 are what catch this clip.
    "no_gravity":  {"M1_HR_nominal": "fail", "M2_time_symmetry": "pass",
                    "M4_parabolicity": "fail", "M5_vx_conservation": "pass",
                    "M6_gravity_symmetry": "absent", "M7_accel_geometry": "pass"},
    "time_warp":   {"M1_HR_nominal": "pass", "M2_time_symmetry": "fail",
                    "M4_parabolicity": "fail", "M5_vx_conservation": "fail"},
    # Perfect physics, wrong angle: every invariant that does not reference the prompt
    # must pass. This is what separates instruction-following from physical validity.
    "wrong_angle": {"M1_HR_nominal": "fail", "M2_time_symmetry": "pass",
                    "M4_parabolicity": "pass", "M5_vx_conservation": "pass",
                    "M7_accel_geometry": "pass"},
    "const_speed": {"M1_HR_nominal": "pass", "M2_time_symmetry": "pass",
                    "M4_parabolicity": "detect", "M5_vx_conservation": "pass"},
}
DETECT_FLOOR_MULT = 5.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=45.0)
    ap.add_argument("--n-flight", type=int, default=60)
    ap.add_argument("--config", default=None)
    ap.add_argument("--outdir", default="data/synthetic")
    ap.add_argument("--blur", type=int, default=1, help="motion-blur substeps for the gt+blur run")
    ap.add_argument("--round-trip", action="store_true",
                    help="write mp4 and re-decode, so codec loss is included")
    ap.add_argument("--backends", default=None,
                    help="comma-separated tracker pair, e.g. color,bgsub")
    ap.add_argument("--device", default=None, help="CUDA_VISIBLE_DEVICES for the worker")
    args = ap.parse_args()

    cfg = TaskConfig.load(args.config)
    if args.backends:
        cfg.tracking.backends = tuple(args.backends.split(","))
    if args.device:
        cfg.tracking.device = args.device
    print(f"tracker backends: {cfg.tracking.backends}\n")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = [(m, 1) for m in MODES] + [("gt", max(2, args.blur))]
    results, rows = [], []

    for mode, blur in runs:
        label = mode if blur == 1 else f"{mode}+blur{blur}"
        frames, seed, traj = build(mode, theta_deg=args.theta, n_flight=args.n_flight,
                                   blur_substeps=blur)
        if args.round_trip:
            path = write_clip(outdir / f"{label}.mp4", frames, fps=24.0)
            clip = read_clip(path)
        else:
            clip = Clip(frames=frames, fps=24.0, path=f"<synthetic:{label}>")

        res, _ = evaluate_projectile(clip, seed, args.theta, sample_id=label,
                                     model="synthetic", cfg=cfg)
        results.append(res)
        by = {r.name: r for r in res.residuals}
        rows.append((label, res, by, traj))

        # Truth check on the raw geometry, independent of pass/fail bookkeeping.
        if res.measurable and res.measured:
            t_hr = traj.truth["H_over_R"]
            m_hr = res.measured["H_over_R"]
            print(f"[{label}] H/R measured {m_hr:.4f} vs constructed {t_hr:.4f} "
                  f"(rel {abs(m_hr / t_hr - 1) * 100:.2f}%)  "
                  f"n_up {res.measured['n_up']:.2f} n_down {res.measured['n_down']:.2f} "
                  f"(constructed {traj.truth['n_up']:.1f}/{traj.truth['n_down']:.1f})")
        else:
            print(f"[{label}] not measurable: {res.gate_reasons or res.violations}")

    names = ["M1_HR_nominal", "M2_time_symmetry", "M4_parabolicity",
             "M5_vx_conservation", "M7_accel_geometry",
             "M1_HR_selfconsistent", "M3_space_symmetry", "M6_gravity_symmetry"]
    short = {n: n.split("_")[0] for n in names}
    print(f"\n{'clip':<15}{'meas':<6}" + "".join(f"{short[n]:<13}" for n in names) + "viol")
    print("-" * 130)
    for label, res, by, _ in rows:
        cells = ""
        for n in names:
            r = by.get(n)
            cells += f"{'skip':<13}" if r is None else \
                f"{('%.4f' % r.residual) + ('' if r.passed else '!'):<13}"
        print(f"{label:<15}{str(res.measurable):<6}{cells}"
              f"{','.join(v.name for v in res.violations) or '-'}")

    # Per-invariant noise floor, taken as the worst exact-physics render. "detect"
    # expectations are judged against this rather than against the tolerance.
    floors: dict[str, float] = {}
    for label, _, by, _ in rows:
        if label.startswith("gt"):
            for n, r in by.items():
                floors[n] = max(floors.get(n, 0.0), r.residual)

    print("\nexpectation check")
    ok = True
    for label, res, by, _ in rows:
        mode = label.split("+")[0]
        for name, want in EXPECT[mode].items():
            r = by.get(name)
            if want == "detect":
                fl = floors.get(name, 0.0)
                seen = r is not None and r.residual > DETECT_FLOOR_MULT * max(fl, 1e-9)
                mult = (r.residual / max(fl, 1e-9)) if r else float("nan")
                if not seen:
                    ok = False
                    print(f"  MISMATCH {label:<14} {name:<22} want >{DETECT_FLOOR_MULT}x floor,"
                          f" got {mult:.1f}x")
                else:
                    print(f"  note     {label:<14} {name:<22} e={r.residual:.4f} = {mult:.0f}x floor"
                          f", tol={r.tol} -> below tolerance, flagged as sensitivity limit")
                continue
            if want == "absent":
                if r is not None:
                    ok = False
                    print(f"  MISMATCH {label:<14} {name:<22} want skipped, got "
                          f"e={r.residual:.4f} (precondition should have excluded it)")
                continue
            got = "pass" if (r is not None and r.passed) else "fail"
            if got != want:
                ok = False
                print(f"  MISMATCH {label:<14} {name:<22} want {want}, got {got}"
                      + (f" (e={r.residual:.4f}, tol={r.tol})" if r else " (missing)"))
    print("  all invariants behaved as intended" if ok else "  SOME INVARIANTS MISBEHAVED")

    print("\nnoise floor on exact-physics renders")
    for label, res, by, _ in rows:
        if not label.startswith("gt"):
            continue
        print(f"  {label:<12} " + "  ".join(
            f"{n.split('_', 1)[0]}={by[n].residual:.4f}" for n in names if n in by))
    worst = max((by[n].residual for label, _, by, _ in rows if label.startswith("gt")
                 for n in names if n in by), default=float("nan"))
    print(f"  worst gt residual {worst:.4f}; tolerances in use: "
          f"hr={cfg.tol.hr_nominal} tsym={cfg.tol.time_symmetry} para={cfg.tol.parabolicity}")

    agg = aggregate(results)
    (outdir / "validation_summary.json").write_text(json.dumps(
        {"aggregate": agg, "per_sample": [r.to_dict() for r in results],
         "config": cfg.to_dict()}, indent=2, default=str))
    print(f"\nwrote {outdir / 'validation_summary.json'}")
    print(f"aggregate over synthetic set: PMR={agg['PMR']:.2f} PE={agg['PE']:.4f} "
          f"PPR={agg['PPR']:.2f} HVR={agg['HVR']:.2f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
