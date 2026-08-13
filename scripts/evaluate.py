#!/usr/bin/env python
"""Evaluate generated videos and report PMR / PE / PPR / HVR.

Reads a generation manifest (or a bare directory of mp4s plus specs), tracks the ball,
measures the calibration-invariant quantities, and writes per-sample JSON plus an
aggregate summary. Nothing here needs a reference video, a simulator, or a judge.

Usage:
  envs/physbench/bin/python scripts/evaluate.py --videos data/videos/minimax-h3 \
      --config configs/p2_projectile.yaml --viz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physbench.config import TaskConfig  # noqa: E402
from physbench.first_frame import SampleSpec  # noqa: E402
from physbench.io_video import read_clip  # noqa: E402
from physbench.metrics import aggregate  # noqa: E402
from physbench.tasks.projectile import evaluate_projectile  # noqa: E402
from physbench.viz import save_overlay  # noqa: E402


def collect(videos: Path, specs: Path) -> list[dict]:
    """Pair every mp4 with its spec, from the manifest when there is one."""
    manifest = videos / "manifest.json"
    if manifest.exists():
        entries = []
        for key, m in json.loads(manifest.read_text()).items():
            if "video" in m and Path(m["video"]).exists():
                entries.append({"key": key, **m})
        if entries:
            return sorted(entries, key=lambda e: e["key"])

    out = []
    for mp4 in sorted(videos.glob("*.mp4")):
        stem = mp4.stem
        cand = [p for p in specs.glob("*.spec.json") if stem.startswith(SampleSpec.load(p).sample_id)]
        if not cand:
            print(f"  skip {mp4.name}: no matching spec in {specs}")
            continue
        spec = SampleSpec.load(cand[0])
        out.append({"key": stem, "video": str(mp4), "spec": str(cand[0]),
                    "sample_id": spec.sample_id, "theta_deg": spec.params.get("theta_deg"),
                    "variant": spec.params.get("variant")})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--specs", default="data/first_frames")
    ap.add_argument("--config", default="configs/p2_projectile.yaml")
    ap.add_argument("--model", default=None, help="label for the report; defaults to dir name")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    videos = Path(args.videos)
    cfg = TaskConfig.load(args.config)
    model = args.model or videos.name
    outdir = Path(args.outdir or f"data/results/{model}")
    outdir.mkdir(parents=True, exist_ok=True)

    entries = collect(videos, Path(args.specs))
    if not entries:
        print(f"no videos found under {videos}")
        return 1
    print(f"evaluating {len(entries)} videos from {videos}\n")

    results = []
    for i, e in enumerate(entries, 1):
        spec = SampleSpec.load(e["spec"])
        theta = float(e.get("theta_deg") or spec.params["theta_deg"])
        clip = read_clip(e["video"], max_frames=args.max_frames)
        res, dbg = evaluate_projectile(clip, spec.ball, theta, sample_id=e["key"],
                                      model=model, cfg=cfg)
        results.append(res)

        (outdir / f"{e['key']}.json").write_text(json.dumps(res.to_dict(), indent=2, default=str))
        if args.viz:
            save_overlay(outdir / f"{e['key']}.png", clip, spec.ball, res, dbg, theta)

        head = f"[{i}/{len(entries)}] {e['key']}  ({clip.n} frames)"
        if not res.measurable:
            print(f"{head}\n    NOT MEASURABLE: {', '.join(res.gate_reasons)}")
            continue
        sr = res.scene_residual
        print(f"{head}   e_i={'n/a' if sr is None else f'{sr:.4f}'}"
              f"   {'PASS' if res.passed_all else 'FAIL'}")
        for r in res.residuals:
            print(f"    {r.name:<24} {r.measured:>9.4f} vs {r.theory:>8.4f}   "
                  f"e={r.residual:>7.4f} tol={r.tol:<6.3f} {'ok' if r.passed else 'FAIL'}"
                  f"{'' if r.primary else '  (diag)'}")
        if res.measured.get("theta_measured_deg") is not None:
            print(f"    theta: prompt {theta:g}deg, measured at launch "
                  f"{res.measured['theta_measured_deg']:.1f}deg, "
                  f"implied by H/R {res.measured['theta_from_HR_deg']:.1f}deg")
        for v in res.violations:
            print(f"    HARD VIOLATION {v.name}: {v.detail}")

    agg = aggregate(results)
    summary = {"model": model, "task": cfg.task, "videos": str(videos),
               "config": cfg.to_dict(), "aggregate": agg,
               "per_sample": [r.to_dict() for r in results]}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'=' * 72}\n{model} on {cfg.task}   n={agg['n_videos']}")
    print(f"  PMR  {agg['PMR']:.3f}   ({agg['n_measurable']}/{agg['n_videos']} measurable)")
    print(f"  PE   {agg['PE']:.4f}   (mean scene residual over {agg['n_with_residuals']} samples;"
          f" median {agg.get('PE_median', float('nan')):.4f})")
    print(f"  PPR  {agg['PPR']:.3f}")
    print(f"  HVR  {agg['HVR']:.3f}")
    if agg.get("per_invariant"):
        print("\n  per invariant:")
        print(f"    {'name':<24}{'mean e':>9}{'median e':>10}{'pass':>7}   {'mean measured':>14}"
              f"{'mean theory':>13}")
        for k, v in agg["per_invariant"].items():
            print(f"    {k:<24}{v['mean_residual']:>9.4f}{v['median_residual']:>10.4f}"
                  f"{v['pass_rate']:>7.2f}   {v['mean_measured']:>14.4f}{v['mean_theory']:>13.4f}"
                  f"{'' if v['primary'] else '   (diag)'}")
    if agg.get("violation_counts"):
        print("\n  hard violations: " + ", ".join(f"{k}={v}" for k, v in agg["violation_counts"].items()))
    if agg.get("gate_failure_counts"):
        print("  gate failures:   " + ", ".join(f"{k}={v}" for k, v in agg["gate_failure_counts"].items()))
    print(f"\nwrote {outdir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
