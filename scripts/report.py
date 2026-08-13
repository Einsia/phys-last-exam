#!/usr/bin/env python
"""Cross-angle report: instruction-following separated from physical validity.

The single most diagnostic number here is the slope of measured launch angle against
prompted launch angle. A model that follows the instruction gives slope 1; one that
ignores it and always launches at its favourite angle gives slope 0. That distinction is
invisible in a scalar "physics score", and it is exactly what M1-nominal vs
M1-selfconsistent is built to separate.

Usage:
  envs/physbench/bin/python scripts/report.py --summary data/results/minimax-h3/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--summary", required=True)
args = ap.parse_args()

d = json.loads(Path(args.summary).read_text())
rows = d["per_sample"]
agg = d["aggregate"]
print(f"{d['model']} on {d['task']}   n={agg['n_videos']}")
print(f"PMR {agg['PMR']:.3f}   PE {agg['PE']:.4f}   PPR {agg['PPR']:.3f}   HVR {agg['HVR']:.3f}\n")

print("angle following")
print(f"  {'sample':<34}{'prompt':>7}{'launch':>8}{'H/R->th':>9}{'H/R':>8}{'theory':>8}"
      f"{'M2':>7}{'M5':>7}")
pts = []
for r in sorted(rows, key=lambda r: (r["measured"].get("theta_nominal_deg", 0), r["sample_id"])):
    m = r.get("measured") or {}
    if not r["measurable"] or "H_over_R" not in m:
        reason = ", ".join(r["gate_reasons"]) or ", ".join(v["name"] for v in r["violations"])
        print(f"  {r['sample_id']:<34}{'':>7}  not measurable: {reason}")
        continue
    res = {x["name"]: x for x in r["residuals"]}
    th_nom = m["theta_nominal_deg"]
    th_launch = m.get("theta_measured_deg", float("nan"))
    th_hr = m["theta_from_HR_deg"]
    pts.append((th_nom, th_launch, th_hr))
    print(f"  {r['sample_id']:<34}{th_nom:>7.0f}{th_launch:>8.1f}{th_hr:>9.1f}"
          f"{m['H_over_R']:>8.4f}{res['M1_HR_nominal']['theory']:>8.4f}"
          f"{res['M2_time_symmetry']['residual']:>7.3f}"
          f"{res.get('M5_vx_conservation', {}).get('residual', float('nan')):>7.3f}")

if len(pts) >= 3:
    a = np.array(pts, float)
    uniq = np.unique(a[:, 0])
    if len(uniq) >= 2:
        for col, lab in ((1, "launch-direction"), (2, "H/R-implied")):
            ok = np.isfinite(a[:, col])
            slope, icpt = np.polyfit(a[ok, 0], a[ok, col], 1)
            print(f"\n  measured({lab}) = {slope:.3f} * prompted + {icpt:.1f}"
                  f"    (1.0 = follows the instruction, 0.0 = ignores it)")
        print(f"  mean measured launch angle {np.nanmean(a[:, 1]):.1f} deg "
              f"over prompts {sorted(int(u) for u in uniq)}")

print("\nper invariant")
print(f"  {'name':<24}{'mean e':>9}{'median e':>10}{'pass':>7}{'mean meas':>11}{'theory':>9}")
for k, v in agg.get("per_invariant", {}).items():
    print(f"  {k:<24}{v['mean_residual']:>9.4f}{v['median_residual']:>10.4f}"
          f"{v['pass_rate']:>7.2f}{v['mean_measured']:>11.4f}{v['mean_theory']:>9.4f}"
          f"{'' if v['primary'] else '  (diag)'}")

if agg.get("violation_counts"):
    print("\nhard violations: " + ", ".join(f"{k}={v}" for k, v in agg["violation_counts"].items()))
if agg.get("gate_failure_counts"):
    print("gate failures:   " + ", ".join(f"{k}={v}" for k, v in agg["gate_failure_counts"].items()))

# Prompt phrasing effect: does asking for slow motion change the physics, or only the
# frame budget? Every invariant here is invariant to the time scale, so a real shift
# means the phrasing changed the generated dynamics, not just their sampling rate.
by_variant: dict[str, list] = {}
for r in rows:
    m = r.get("measured") or {}
    if r["measurable"] and "H_over_R" in m:
        v = "slowmo" if "slowmo" in r["sample_id"] else "plain"
        by_variant.setdefault(v, []).append(
            (r["scene_residual"], m.get("flight_frames"), m.get("theta_measured_deg")))
if len(by_variant) > 1:
    print("\nprompt phrasing")
    for v, xs in sorted(by_variant.items()):
        e = [x[0] for x in xs if x[0] is not None]
        fr = [x[1] for x in xs if x[1]]
        th = [x[2] for x in xs if x[2]]
        print(f"  {v:<8} n={len(xs)}  mean e_i={np.mean(e):.4f}  "
              f"mean flight frames={np.mean(fr):.1f}  mean launch angle={np.nanmean(th):.1f} deg")
