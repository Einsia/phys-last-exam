#!/usr/bin/env python
"""Draw the P2 first frames and write a spec sidecar for each.

The sidecar is what carries the tracker seed (exact ball colour, radius, centre) and
the nominal angle forward to generation and evaluation, so no later stage has to
re-estimate anything it could have been told.

Usage:
  envs/physbench/bin/python scripts/make_first_frames.py --config configs/p2_projectile.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physbench.first_frame import SampleSpec, make_projectile_first_frame  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/p2_projectile.yaml")
    ap.add_argument("--outdir", default="data/first_frames")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    w, h = cfg["canvas"]["width"], cfg["canvas"]["height"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    n = 0
    for theta in cfg["angles_deg"]:
        for variant, template in cfg["prompts"].items():
            sample_id = f"P2_theta{theta}_{variant}"
            img, ball, ground_y = make_projectile_first_frame(width=w, height=h)
            img_path = outdir / f"{sample_id}.png"
            Image.fromarray(img).save(img_path)

            spec = SampleSpec(
                sample_id=sample_id,
                task=cfg["task"],
                width=w, height=h,
                prompt=" ".join(template.format(theta=theta).split()),
                first_frame=str(img_path),
                ball=ball,
                params={"theta_deg": float(theta), "variant": variant},
                ground_y=ground_y,
                notes="angle stated only in the prompt, never drawn into the frame, "
                      "so M1 against the nominal angle is a real instruction-following test",
            )
            spec.save(outdir / f"{sample_id}.spec.json")
            n += 1
            print(f"{sample_id}: {img_path}  ball=({ball.cx:.1f},{ball.cy:.1f}) r={ball.radius:.1f}")

    print(f"\nwrote {n} first frames + specs to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
