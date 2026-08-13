#!/usr/bin/env python
"""Generate videos from the P2 first frames with a chosen VDM.

Each sample is one subprocess into the model's own conda env (see physbench/vdm.py),
so a crash or an OOM costs one sample rather than the run. Finished samples are
skipped on re-invocation, which makes this safe to restart.

Usage:
  envs/physbench/bin/python scripts/generate.py --model minimax-h3 \
      --specs data/first_frames --outdir data/videos/minimax-h3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physbench.first_frame import SampleSpec  # noqa: E402
from physbench.vdm import build_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minimax-h3")
    ap.add_argument("--config", default="configs/p2_projectile.yaml")
    ap.add_argument("--specs", default="data/first_frames")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--devices", default="4,5,6,7")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these sample ids")
    ap.add_argument("--num-frames", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--no-flash3", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--retries", type=int, default=3, help="retry on host-OOM SIGKILL")
    ap.add_argument("--retry-wait", type=int, default=90)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    outdir = Path(args.outdir or f"data/videos/{args.model}")
    outdir.mkdir(parents=True, exist_ok=True)
    logdir = Path("logs") / args.model
    logdir.mkdir(parents=True, exist_ok=True)

    specs = sorted(Path(args.specs).glob("*.spec.json"))
    if args.only:
        specs = [p for p in specs if SampleSpec.load(p).sample_id in set(args.only)]
    if not specs:
        print("no specs matched")
        return 1

    seeds = args.seeds if args.seeds is not None else cfg.get("seeds", [42])
    num_frames = args.num_frames or cfg.get("num_frames", 124)
    model = build_model(args.model, devices=args.devices, flash3=not args.no_flash3,
                        height=cfg["canvas"]["height"], width=cfg["canvas"]["width"],
                        steps=args.steps)

    manifest_path = outdir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    jobs = [(p, s) for p in specs for s in seeds]
    print(f"{len(jobs)} jobs ({len(specs)} specs x {len(seeds)} seeds) -> {outdir}")

    done = fail = 0
    for i, (spec_path, seed) in enumerate(jobs, 1):
        spec = SampleSpec.load(spec_path)
        key = f"{spec.sample_id}_seed{seed}"
        out = outdir / f"{key}.mp4"
        if out.exists() and not args.overwrite:
            print(f"[{i}/{len(jobs)}] {key}: exists, skipping")
            manifest.setdefault(key, {}).update({"video": str(out), "spec": str(spec_path)})
            continue

        print(f"[{i}/{len(jobs)}] {key}: generating ...", flush=True)
        t0 = time.time()
        gen = None
        for attempt in range(1, args.retries + 1):
            try:
                gen = model.generate(spec.first_frame, spec.prompt, str(out),
                                     num_frames=num_frames, seed=seed,
                                     log_path=str(logdir / f"{key}.log"))
                break
            except Exception as exc:  # noqa: BLE001 - one bad sample must not end the run
                msg = str(exc)
                (logdir / f"{key}.err.log").write_text(msg)
                # rc=-9 is SIGKILL from the host OOM killer, which on a shared box is
                # usually somebody else's memory spike rather than anything about this
                # sample. Worth retrying; a real failure will reproduce.
                transient = "rc=-9" in msg or "rc=-15" in msg
                if attempt < args.retries and transient:
                    print(f"    attempt {attempt} killed (likely host OOM); "
                          f"retrying in {args.retry_wait}s")
                    time.sleep(args.retry_wait)
                    continue
                fail += 1
                print(f"    FAILED after {time.time() - t0:.0f}s: {type(exc).__name__}"
                      f"{' (transient, retries exhausted)' if transient else ''}; "
                      f"see {logdir / (key + '.err.log')}")
                break
        if gen is None:
            continue

        (logdir / f"{key}.log").write_text(gen.log)
        manifest[key] = {"video": gen.path, "spec": str(spec_path), "seed": seed,
                         "sample_id": spec.sample_id, "prompt": spec.prompt,
                         "theta_deg": spec.params.get("theta_deg"),
                         "variant": spec.params.get("variant"),
                         "model": args.model, "num_frames": num_frames,
                         "gen_seconds": round(gen.seconds, 1), "cmd": gen.cmd}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        done += 1
        print(f"    ok in {gen.seconds:.0f}s -> {out.name} "
              f"({Path(gen.path).stat().st_size / 1024**2:.1f} MB)")

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\ngenerated {done}, failed {fail}, manifest -> {manifest_path}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
