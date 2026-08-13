#!/usr/bin/env python
"""Write the synthetic clips as a fake model output directory.

Same layout a real VDM run produces (mp4s + manifest + spec sidecars), so the eval
driver can be exercised on clips whose physics is known. Doubles as a regression
fixture: `evaluate.py` over this directory should always reproduce the table in
validate_metrics.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physbench.first_frame import SampleSpec  # noqa: E402
from physbench.io_video import write_clip  # noqa: E402
from physbench.render_synth import build  # noqa: E402

MODES = ["gt", "no_gravity", "time_warp", "wrong_angle", "const_speed"]

ap = argparse.ArgumentParser()
ap.add_argument("--theta", type=float, default=45.0)
ap.add_argument("--n-flight", type=int, default=60)
ap.add_argument("--videos", default="data/videos/synthetic")
ap.add_argument("--specs", default="data/first_frames_synthetic")
args = ap.parse_args()

vdir, sdir = Path(args.videos), Path(args.specs)
vdir.mkdir(parents=True, exist_ok=True)
sdir.mkdir(parents=True, exist_ok=True)

manifest = {}
for mode, blur in [(m, 1) for m in MODES] + [("gt", 3)]:
    label = mode if blur == 1 else f"{mode}_blur{blur}"
    frames, seed, traj = build(mode, theta_deg=args.theta, n_flight=args.n_flight,
                              blur_substeps=blur)
    mp4 = write_clip(vdir / f"{label}.mp4", frames, fps=24.0)
    ff = sdir / f"{label}.png"
    Image.fromarray(frames[0]).save(ff)
    spec = SampleSpec(sample_id=label, task="P2_projectile",
                      width=frames.shape[2], height=frames.shape[1],
                      prompt=f"synthetic {mode}, theta={args.theta}",
                      first_frame=str(ff), ball=seed,
                      params={"theta_deg": args.theta, "variant": mode},
                      notes="physics known by construction; "
                            f"constructed H/R={traj.truth['H_over_R']:.4f}")
    spec.save(sdir / f"{label}.spec.json")
    manifest[label] = {"video": mp4, "spec": str(sdir / f"{label}.spec.json"),
                       "sample_id": label, "theta_deg": args.theta, "variant": mode,
                       "model": "synthetic", "constructed": traj.truth}
    print(f"{label:<14} -> {mp4}")

(vdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
print(f"\nmanifest -> {vdir / 'manifest.json'}")
