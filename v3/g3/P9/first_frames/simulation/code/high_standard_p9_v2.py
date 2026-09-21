#!/usr/bin/env python3
"""Render three corrected high-standard P9 structural first frames.

Every variant contains only a freestanding support, one rigid top beam, two
pendulum strings, two identical bobs, and their pivot bearings.  No visible
release line, catch, black marker, guide arc, label, or outcome hint exists.
The short and long pendulum lengths are exactly 1:2; both start motionless at
the same 12 degree angle to vertical and lean to the same side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _find_project_root(script: Path) -> Path:
    for parent in (script.parent, *script.parents):
        if (parent / "simulation" / "common.py").is_file() and (parent / "tasks_all.json").is_file():
            return parent
    raise RuntimeError("could not locate project root")


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = _find_project_root(SCRIPT_PATH)
sys.path.insert(0, str(PROJECT_ROOT))

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

from simulation.common import (  # noqa: E402
    add_box,
    add_camera,
    add_cylinder,
    add_sphere,
    area_light,
    cylinder_between,
    material,
    palette,
    reset_scene,
)


RENDER_WIDTH = 2048
RENDER_HEIGHT = 1152
INITIAL_ANGLE_DEGREES = 12.0
SHORT_LENGTH_M = 1.75
LONG_LENGTH_M = 3.50
LENGTH_RATIO = LONG_LENGTH_M / SHORT_LENGTH_M
BOB_RADIUS_M = 0.29
PIVOT_Z_M = 5.46
PIVOT_X_M = (-2.40, 1.10)
GRAVITY_MPS2 = 9.81

VARIANTS: dict[int, dict[str, str]] = {
    1: {
        "name": "modern_blue_steel_lab",
        "apparatus": "blue powder-coated rectangular steel test frame with machined steel pivots",
        "environment": "bright contemporary university dynamics laboratory with cool softboxes",
        "camera": "exact-front orthographic documentary view",
    },
    2: {
        "name": "classic_hardwood_teaching_room",
        "apparatus": "dark hardwood post-and-lintel teaching stand with brass pivots",
        "environment": "warm older teaching laboratory with cream wall and window-like side light",
        "camera": "long-lens perspective view with slight lateral and downward offset",
    },
    3: {
        "name": "industrial_black_metrology_bay",
        "apparatus": "black T-slot style industrial frame with stainless pivots and wide machine feet",
        "environment": "dark industrial metrology bay with cool rim lighting",
        "camera": "elevated orthographic inspection view",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_render(engine: str, samples: int, resolution_scale: float) -> dict[str, Any]:
    if not 0.25 <= resolution_scale <= 1.0:
        raise ValueError("resolution_scale must be within 0.25--1.0")
    reset_scene(engine, samples=samples)
    scene = bpy.context.scene
    if engine == "cycles":
        # Avoid multi-worker VRAM contention on the shared server.
        scene.cycles.device = "CPU"
        cycles_addon = bpy.context.preferences.addons.get("cycles")
        if cycles_addon is not None:
            try:
                prefs = cycles_addon.preferences
                prefs.get_devices()
                for device in prefs.devices:
                    device.use = device.type == "CPU"
            except Exception:
                pass
        try:
            scene.cycles.use_gpu_denoising = False
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
        except (AttributeError, TypeError, ValueError):
            pass
    scene.render.resolution_x = int(round(RENDER_WIDTH * resolution_scale))
    scene.render.resolution_y = int(round(RENDER_HEIGHT * resolution_scale))
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 10
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    return palette()


def _environment(mats: dict[str, Any], variant: int) -> None:
    if variant == 1:
        floor = material("P9v2_modern_floor", (0.30, 0.33, 0.36), roughness=0.72)
        wall = material("P9v2_modern_wall", (0.76, 0.79, 0.80), roughness=0.88)
        add_box("P9_environment_floor", (0.0, 0.8, -0.15), (14.0, 8.0, 0.30), floor, bevel=0.0)
        add_box("P9_environment_wall", (0.0, 3.5, 3.2), (14.0, 0.18, 6.6), wall, bevel=0.0)
        area_light("P9_modern_key", (-3.8, -3.0, 7.4), (-1.0, 0.0, 3.1), 1120, 4.0)
        area_light("P9_modern_fill", (4.2, -1.8, 6.1), (1.2, 0.0, 2.8), 680, 3.2, (0.78, 0.90, 1.0))
    elif variant == 2:
        floor = material("P9v2_old_floor", (0.22, 0.13, 0.065), roughness=0.76)
        wall = material("P9v2_cream_wall", (0.67, 0.58, 0.43), roughness=0.91)
        add_box("P9_environment_floor", (0.0, 0.8, -0.15), (14.0, 8.0, 0.30), floor, bevel=0.0)
        add_box("P9_environment_wall", (0.0, 3.5, 3.2), (14.0, 0.18, 6.6), wall, bevel=0.0)
        # Broad warm side light evokes a real classroom window without adding
        # visible objects that could distract from the pendulums.
        area_light("P9_classic_window", (-5.4, -2.5, 6.5), (-1.3, 0.0, 3.0), 980, 4.8, (1.0, 0.72, 0.46))
        area_light("P9_classic_fill", (4.1, -0.8, 5.3), (1.2, 0.0, 2.5), 390, 3.0, (0.72, 0.82, 1.0))
    else:
        floor = material("P9v2_industrial_floor", (0.025, 0.032, 0.038), roughness=0.62)
        wall = material("P9v2_industrial_wall", (0.045, 0.055, 0.066), roughness=0.78)
        add_box("P9_environment_floor", (0.0, 0.8, -0.15), (14.0, 8.0, 0.30), floor, bevel=0.0)
        add_box("P9_environment_wall", (0.0, 3.5, 3.2), (14.0, 0.18, 6.6), wall, bevel=0.0)
        area_light("P9_industrial_top", (0.0, -0.5, 8.2), (0.0, 0.0, 3.0), 850, 3.4, (0.70, 0.84, 1.0))
        area_light("P9_industrial_rim", (-5.0, 0.8, 5.8), (-1.0, 0.0, 3.0), 560, 2.0, (0.35, 0.65, 1.0))
        area_light("P9_industrial_warm_fill", (4.6, -2.4, 4.0), (1.0, 0.0, 2.5), 330, 2.2, (1.0, 0.55, 0.30))


def _frame_materials(mats: dict[str, Any], variant: int) -> tuple[Any, Any, Any]:
    if variant == 1:
        return (
            material("P9v2_blue_powdercoat", (0.012, 0.075, 0.18), roughness=0.32, metallic=0.65),
            mats["steel"],
            material("P9v2_red_ceramic_bob", (0.72, 0.035, 0.014), roughness=0.30, metallic=0.03),
        )
    if variant == 2:
        return (
            mats["wood_dark"],
            mats["brass"],
            material("P9v2_ivory_ceramic_bob", (0.83, 0.72, 0.50), roughness=0.38, metallic=0.02),
        )
    return (
        material("P9v2_black_tslot", (0.008, 0.012, 0.016), roughness=0.24, metallic=0.78),
        mats["aluminum"],
        material("P9v2_safety_yellow_bob", (0.91, 0.43, 0.012), roughness=0.34, metallic=0.04),
    )


def _build_frame(mats: dict[str, Any], variant: int, frame_mat: Any, pivot_mat: Any) -> None:
    """Build three materially and structurally distinct freestanding stands."""
    beam_z = 5.70
    if variant == 1:
        add_box("P9_rigid_common_top_beam", (0.0, 0.0, beam_z), (10.72, 0.44, 0.34), frame_mat, bevel=0.055)
        for side, x in (("left", -5.18), ("right", 5.18)):
            add_box(f"P9_{side}_stand_upright", (x, 0.0, 2.88), (0.32, 0.54, 5.52), frame_mat, bevel=0.045)
            add_box(f"P9_{side}_stand_foot", (x, 0.0, 0.15), (1.30, 1.20, 0.23), pivot_mat, bevel=0.045)
    elif variant == 2:
        # Chunkier wood joinery and wide timber sills are visibly unlike the
        # modern steel rectangle while preserving the same clear swing bay.
        add_box("P9_rigid_common_top_beam", (0.0, 0.0, beam_z), (10.86, 0.70, 0.46), frame_mat, bevel=0.075)
        for side, x in (("left", -5.18), ("right", 5.18)):
            add_box(f"P9_{side}_stand_upright", (x, 0.0, 2.88), (0.46, 0.72, 5.44), frame_mat, bevel=0.065)
            add_box(f"P9_{side}_stand_foot", (x, 0.0, 0.18), (1.62, 1.42, 0.29), frame_mat, bevel=0.070)
            add_box(f"P9_{side}_joinery_block", (x, 0.0, 5.39), (0.76, 0.80, 0.58), pivot_mat, bevel=0.055)
    else:
        add_box("P9_rigid_common_top_beam", (0.0, 0.0, beam_z), (10.84, 0.38, 0.38), frame_mat, bevel=0.025)
        # Thin aluminium slot rails and broad machine bases make this a true
        # industrial extrusion stand, not just a recoloured version 1.
        add_box("P9_top_beam_slot_rail", (0.0, -0.205, beam_z), (10.60, 0.035, 0.060), pivot_mat, bevel=0.010)
        for side, x in (("left", -5.20), ("right", 5.20)):
            add_box(f"P9_{side}_stand_upright", (x, 0.0, 2.86), (0.38, 0.38, 5.56), frame_mat, bevel=0.020)
            add_box(f"P9_{side}_upright_slot_rail", (x, -0.205, 2.86), (0.055, 0.035, 5.34), pivot_mat, bevel=0.008)
            add_box(f"P9_{side}_stand_foot", (x, 0.0, 0.14), (1.75, 1.34, 0.25), pivot_mat, bevel=0.035)


def _build_pendulum(
    mats: dict[str, Any],
    *,
    index: int,
    pivot_x: float,
    length: float,
    bob_material: Any,
    pivot_material: Any,
) -> dict[str, Any]:
    angle = math.radians(INITIAL_ANGLE_DEGREES)
    pivot = Vector((pivot_x, 0.0, PIVOT_Z_M))
    bob = pivot + Vector((length * math.sin(angle), 0.0, -length * math.cos(angle)))
    cylinder_between(f"P9_pendulum_{index}_string", tuple(pivot), tuple(bob), 0.016, mats["black"])
    add_sphere(f"P9_pendulum_{index}_bob", tuple(bob), BOB_RADIUS_M, bob_material, segments=64)
    add_cylinder(
        f"P9_pendulum_{index}_pivot_bearing",
        tuple(pivot),
        0.105,
        0.42,
        pivot_material,
        rotation=(math.pi / 2.0, 0.0, 0.0),
    )
    return {
        "index": index,
        "pivot_m": [round(float(v), 6) for v in pivot],
        "bob_center_m": [round(float(v), 6) for v in bob],
        "string_length_m": length,
        "initial_angle_to_vertical_degrees": INITIAL_ANGLE_DEGREES,
        "initial_side": "right",
        "initial_angular_velocity_rad_s": 0.0,
        "bob_radius_m": BOB_RADIUS_M,
        "state": "stationary_pre_release_first_frame",
        "small_angle_period_seconds": round(2.0 * math.pi * math.sqrt(length / GRAVITY_MPS2), 6),
    }


def _camera(variant: int) -> bpy.types.Object:
    target = (0.0, 0.0, 3.18)
    if variant == 1:
        camera = add_camera((0.0, -16.0, 3.70), target, lens=72.0)
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = 12.25
    elif variant == 2:
        camera = add_camera((0.62, -19.0, 4.15), target, lens=56.0)
        camera.data.type = "PERSP"
        camera.data.lens = 56.0
    else:
        camera = add_camera((0.0, -16.0, 4.82), target, lens=70.0)
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = 12.55
    camera.data.dof.use_dof = False
    bpy.context.scene.camera = camera
    return camera


def _build_scene(mats: dict[str, Any], variant: int) -> tuple[list[dict[str, Any]], bpy.types.Object]:
    _environment(mats, variant)
    frame_mat, pivot_mat, bob_mat = _frame_materials(mats, variant)
    _build_frame(mats, variant, frame_mat, pivot_mat)
    pendulums = [
        _build_pendulum(
            mats,
            index=1,
            pivot_x=PIVOT_X_M[0],
            length=SHORT_LENGTH_M,
            bob_material=bob_mat,
            pivot_material=pivot_mat,
        ),
        _build_pendulum(
            mats,
            index=2,
            pivot_x=PIVOT_X_M[1],
            length=LONG_LENGTH_M,
            bob_material=bob_mat,
            pivot_material=pivot_mat,
        ),
    ]
    return pendulums, _camera(variant)


def _validate_scene(pendulums: list[dict[str, Any]], camera: bpy.types.Object) -> None:
    if len(pendulums) != 2:
        raise AssertionError("P9 must contain exactly two pendulums")
    angles = [float(item["initial_angle_to_vertical_degrees"]) for item in pendulums]
    if max(angles) - min(angles) >= 0.5 or any(abs(value - 12.0) > 1e-8 for value in angles):
        raise AssertionError(f"initial angles are not matched at 12 degrees: {angles}")
    lengths = [float(item["string_length_m"]) for item in pendulums]
    if abs(max(lengths) / min(lengths) - 2.0) > 1e-8:
        raise AssertionError(f"pendulum length ratio is not exactly 1:2: {lengths}")
    if len({float(item["bob_radius_m"]) for item in pendulums}) != 1:
        raise AssertionError("bob radii differ")
    if max(item["pivot_m"][2] for item in pendulums) - min(item["pivot_m"][2] for item in pendulums) > 1e-8:
        raise AssertionError("pivots are not at the same height")
    if any(item["initial_side"] != "right" for item in pendulums):
        raise AssertionError("both pendulums must start on the same side")
    if any(float(item["initial_angular_velocity_rad_s"]) != 0.0 for item in pendulums):
        raise AssertionError("both pendulums must be motionless")
    bobs = [obj for obj in bpy.data.objects if obj.name.endswith("_bob")]
    strings = [obj for obj in bpy.data.objects if obj.name.endswith("_string")]
    if len(bobs) != 2 or len(strings) != 2:
        raise AssertionError(f"unexpected object counts: bobs={len(bobs)}, strings={len(strings)}")
    forbidden = ("release", "catch", "carrier", "guide", "arc", "arrow", "marker")
    offending = [obj.name for obj in bpy.data.objects if any(term in obj.name.lower() for term in forbidden)]
    if offending:
        raise AssertionError(f"forbidden visible helper objects exist: {offending}")
    if any(obj.type == "FONT" for obj in bpy.data.objects):
        raise AssertionError("text is forbidden in first frames")
    if camera.data.dof.use_dof:
        raise AssertionError("depth of field must remain disabled so both strings are sharp")


def _render_variant(
    variant: int,
    output_root: Path,
    *,
    engine: str,
    samples: int,
    resolution_scale: float,
) -> dict[str, Any]:
    started = time.time()
    mats = _configure_render(engine, samples, resolution_scale)
    pendulums, camera = _build_scene(mats, variant)
    bpy.context.view_layer.update()
    _validate_scene(pendulums, camera)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / f"sim_{variant:02d}.png"
    bpy.context.scene.render.filepath = str(output.resolve())
    bpy.ops.render.render(write_still=True)
    optical_axis = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    record: dict[str, Any] = {
        "task_id": "P9",
        "candidate": variant,
        "variant": VARIANTS[variant],
        "method": "high_standard_procedural_simulation_v2",
        "file": str(output.resolve()),
        "width": bpy.context.scene.render.resolution_x,
        "height": bpy.context.scene.render.resolution_y,
        "sha256": _sha256(output),
        "renderer": f"Blender Python {bpy.app.version_string} / {engine}",
        "samples": samples,
        "elapsed_seconds": round(time.time() - started, 3),
        "camera": {
            "type": camera.data.type,
            "location": [round(float(v), 6) for v in camera.location],
            "optical_axis": [round(float(v), 6) for v in optical_axis],
            "orthographic_scale": camera.data.ortho_scale if camera.data.type == "ORTHO" else None,
            "lens_mm": camera.data.lens,
            "dof_enabled": False,
        },
        "physical_parameters": {
            "gravity_mps2": GRAVITY_MPS2,
            "initial_angle_to_vertical_degrees": INITIAL_ANGLE_DEGREES,
            "maximum_angle_mismatch_degrees": 0.0,
            "short_length_m": SHORT_LENGTH_M,
            "long_length_m": LONG_LENGTH_M,
            "length_ratio_short_to_long": "1:2",
            "length_ratio_long_to_short": LENGTH_RATIO,
            "pendulums": pendulums,
        },
        "hard_constraints": {
            "exactly_two_strings": True,
            "exactly_two_identical_bobs": True,
            "shared_rigid_beam": True,
            "pivots_same_height": True,
            "length_ratio_exactly_1_to_2": True,
            "same_initial_angle_12_degrees": True,
            "same_initial_side": True,
            "both_stationary": True,
            "visible_release_lines_catches_markers_absent": True,
            "full_future_swing_space_in_frame": True,
            "text_arrows_formulas_guide_arcs_absent": True,
        },
        "external_assets": [],
        "code": str(SCRIPT_PATH),
        "code_sha256": _sha256(SCRIPT_PATH),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_root / f"sim_{variant:02d}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("sop_delivery/P9/high_standard_v2/simulation"),
    )
    parser.add_argument("--variant", type=int, action="append", choices=sorted(VARIANTS))
    parser.add_argument("--engine", choices=("cycles", "eevee", "workbench"), default="cycles")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--resolution-scale", type=float, default=1.0)
    args = parser.parse_args()
    selected = args.variant or sorted(VARIANTS)
    records = [
        _render_variant(
            variant,
            args.output_root,
            engine=args.engine,
            samples=args.samples,
            resolution_scale=args.resolution_scale,
        )
        for variant in selected
    ]
    manifest = {
        "task_id": "P9",
        "version": "high_standard_v2",
        "completed": len(records),
        "physical_invariant": {
            "short_length_m": SHORT_LENGTH_M,
            "long_length_m": LONG_LENGTH_M,
            "ratio_short_to_long": "1:2",
            "both_initial_angles_degrees": [INITIAL_ANGLE_DEGREES, INITIAL_ANGLE_DEGREES],
            "same_side": "right",
            "same_pivot_height_m": PIVOT_Z_M,
            "identical_bob_radius_m": BOB_RADIUS_M,
            "visible_release_helpers": 0,
        },
        "records": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
