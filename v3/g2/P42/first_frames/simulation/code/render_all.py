#!/usr/bin/env python3
"""Render or reuse one procedural first frame for every task."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import bpy

from .common import HEIGHT, WIDTH, finish_default_camera, lab_shell, palette, reset_scene


PILOT_SLUGS = {
    "P2": "P2_projectile",
    "P4": "P4_bounce",
    "P7": "P7_rolling",
    "P12": "P12_refraction",
    "P42": "P42_static_friction",
}


def load_builders() -> dict[str, tuple[object, bool]]:
    builders: dict[str, tuple[object, bool]] = {}
    for module_name in (
        "simulation.builders_mech_optics",
        "simulation.builders_thermal_em",
        "simulation.builders_induction_surface",
    ):
        try:
            module = __import__(module_name, fromlist=["BUILDERS"])
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                continue
            raise
        overlap = set(builders) & set(module.BUILDERS)
        if overlap:
            raise RuntimeError(f"duplicate builders in {module_name}: {sorted(overlap)}")
        builders.update(module.BUILDERS)
    return builders


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_for(path: Path, task_id: str, *, renderer: str, elapsed: float, reused_from: str | None = None) -> dict[str, object]:
    return {
        "task_id": task_id,
        "method": "simulation",
        "candidate": 1,
        "file": str(path),
        "width": WIDTH,
        "height": HEIGHT,
        "sha256": sha256_file(path),
        "renderer": renderer,
        "reused_from": reused_from,
        "elapsed_seconds": round(elapsed, 3),
        "external_assets": [],
    }


def reuse_pilot(task_id: str, pilot_root: Path, output_root: Path) -> dict[str, object]:
    started = time.time()
    source = pilot_root / PILOT_SLUGS[task_id] / "candidate_01.png"
    if not source.exists():
        raise FileNotFoundError(f"pilot simulation is missing: {source}")
    out_dir = output_root / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "candidate_01.png"
    shutil.copy2(source, target)
    record = record_for(target, task_id, renderer="Blender Python / pilot reuse", elapsed=time.time()-started, reused_from=str(source))
    (out_dir / "meta.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def render_one(task_id: str, builder_spec: tuple[object, bool], output_root: Path, *, engine: str, samples: int) -> dict[str, object]:
    started = time.time()
    reset_scene(engine, samples=samples)
    mats = palette()
    builder, dark = builder_spec
    lab_shell(mats, dark=dark)
    builder(mats)
    finish_default_camera(dark=dark)
    out_dir = output_root / task_id
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "candidate_01.png"
    bpy.context.scene.render.filepath = str(output.resolve())
    bpy.ops.render.render(write_still=True)
    record = record_for(
        output,
        task_id,
        renderer=f"Blender Python {'.'.join(map(str, bpy.app.version))} / {engine}",
        elapsed=time.time()-started,
    )
    (out_dir / "meta.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=Path("tasks_all.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/simulation"))
    parser.add_argument("--pilot-root", type=Path, default=Path("../physical-bench-first-frame-study/outputs/simulation"))
    parser.add_argument("--task", action="append", help="Render only this task ID; repeatable")
    parser.add_argument("--engine", choices=("cycles", "eevee", "workbench"), default="cycles")
    parser.add_argument("--samples", type=int, default=36)
    parser.add_argument("--no-reuse-pilot", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    task_payload = json.loads(args.tasks.read_text(encoding="utf-8"))
    tasks = task_payload["tasks"] if isinstance(task_payload, dict) else task_payload
    task_ids = [task["id"] for task in tasks]
    if args.task:
        requested = set(args.task)
        unknown = requested - set(task_ids)
        if unknown:
            parser.error(f"unknown task IDs: {sorted(unknown)}")
        task_ids = [task_id for task_id in task_ids if task_id in requested]

    builders = load_builders()
    records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for index, task_id in enumerate(task_ids, start=1):
        print(f"[{index}/{len(task_ids)}] simulation {task_id}", flush=True)
        try:
            if task_id in PILOT_SLUGS and not args.no_reuse_pilot:
                record = reuse_pilot(task_id, args.pilot_root, args.output_root)
            else:
                if task_id not in builders:
                    raise KeyError(f"no simulation builder registered for {task_id}")
                record = render_one(task_id, builders[task_id], args.output_root, engine=args.engine, samples=args.samples)
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
        except Exception as exc:
            error = {"task_id": task_id, "error": str(exc), "traceback": traceback.format_exc()}
            errors.append(error)
            print(json.dumps(error, ensure_ascii=False), file=sys.stderr, flush=True)
            if not args.continue_on_error:
                break

    manifest = {
        "method": "simulation",
        "requested": len(task_ids),
        "completed": len(records),
        "failed": len(errors),
        "records": records,
        "errors": errors,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("requested", "completed", "failed")}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
