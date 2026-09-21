#!/usr/bin/env python3
"""Render three SOP-compliant P3 first-frame simulation candidates.

P3 compares two identical balls launched with the same speed at 30 and 60
degrees.  Every candidate uses two parallel lanes with one common x start line,
zero x-direction yaw, and zero camera roll.  A restrained 10--18 degree camera
elevation separates the lanes without making their long edges diagonal.
Only the observable pre-release state is rendered: no labels, arrows,
trajectories, formulas, or outcome hints are added to the image.

The script is intentionally separate from the legacy 30/45/60 P3 builder so
that the completed experiment remains reproducible and its old output is never
overwritten.
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
    """Find the first ancestor containing the shared simulation package."""
    for parent in (script.parent, *script.parents):
        if (parent / "simulation" / "common.py").is_file() and (parent / "tasks_all.json").is_file():
            return parent
    raise RuntimeError("could not locate project root containing simulation/common.py and tasks_all.json")


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
    finish_default_camera,
    lab_shell,
    material,
    palette,
    reset_scene,
)


RENDER_WIDTH = 2048
RENDER_HEIGHT = 1152
BALL_RADIUS = 0.18
BARREL_LENGTH = 0.92
RELATIVE_LAUNCH_HEIGHT = BALL_RADIUS
NOMINAL_SPEED_MPS = 7.0
GRAVITY_MPS2 = 9.81

VARIANTS: dict[int, dict[str, Any]] = {
    1: {
        "name": "parallel_aluminum_front30_back60",
        "description": "Two parallel aluminum landing lanes sharing one x start line.",
        "camera": "orthographic side elevation with 10 degree downward pitch",
        "layout": "front lane 30 degrees, rear lane 60 degrees; both launch toward +x",
    },
    2: {
        "name": "parallel_optical_front60_back30",
        "description": "Dark optical-bench lanes with the near/far angle ordering reversed.",
        "camera": "orthographic side elevation with 14 degree downward pitch",
        "layout": "front lane 60 degrees, rear lane 30 degrees; both launch toward +x",
    },
    3: {
        "name": "parallel_wood_front30_back60",
        "description": "Warm wood teaching lanes with greater visual separation.",
        "camera": "orthographic side elevation with 18 degree downward pitch",
        "layout": "front lane 30 degrees, rear lane 60 degrees; both launch toward +x",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_render(engine: str, samples: int, resolution_scale: float) -> dict[str, Any]:
    reset_scene(engine, samples=samples)
    scene = bpy.context.scene
    if not 0.25 <= resolution_scale <= 1.0:
        raise ValueError("resolution_scale must be within 0.25--1.0")
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


def _parallel_lane_camera(
    *,
    x: float,
    target_z: float,
    distance: float,
    pitch_degrees: float,
    ortho_scale: float,
) -> bpy.types.Object:
    """Create a zero-yaw/zero-roll orthographic camera for parallel lanes.

    The optical axis lies exactly in the YZ plane.  The camera's image-right
    axis is exactly global +X, so every x-aligned lane edge remains horizontal
    in the rendered image even when a small downward pitch reveals lane depth.
    """
    if not 10.0 <= pitch_degrees <= 18.0:
        raise ValueError("pitch_degrees must remain within the reviewed 10--18 degree range")
    pitch = math.radians(pitch_degrees)
    optical_axis = Vector((0.0, math.cos(pitch), -math.sin(pitch)))
    target = Vector((x, 0.0, target_z))
    location = target - optical_axis * distance
    camera = add_camera(tuple(location), tuple(target), lens=70.0)
    camera.rotation_mode = "XYZ"
    camera.rotation_euler = (math.pi / 2.0 - pitch, 0.0, 0.0)
    camera.data.dof.use_dof = False
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = ortho_scale
    bpy.context.scene.camera = camera
    return camera


def _fixed_launch_launcher(
    mats: dict[str, Any],
    *,
    name: str,
    ball_center: tuple[float, float, float],
    deck_top: float,
    angle_deg: float,
    horizontal_sign: int,
    accent: Any,
) -> dict[str, Any]:
    """Build one launcher whose ball center is a fixed, comparable launch point.

    The barrel extends backward from the ball.  Consequently the 30 and 60
    degree balls have exactly equal launch heights even though their identical
    barrels are rotated to different angles.
    """
    if horizontal_sign not in (-1, 1):
        raise ValueError("horizontal_sign must be -1 or +1")
    angle = math.radians(angle_deg)
    axis = Vector((horizontal_sign * math.cos(angle), 0.0, math.sin(angle)))
    ball = Vector(ball_center)
    barrel_front = ball - axis * (BALL_RADIUS + 0.035)
    barrel_rear = barrel_front - axis * BARREL_LENGTH
    pivot = barrel_rear + axis * (BARREL_LENGTH * 0.48)

    # Both launchers use the same low floor base behind the common start line.
    # Only the adjustable barrel angle changes; base and compression hardware
    # dimensions are identical for the two conditions.
    base_x = ball.x - horizontal_sign * 0.68
    base_top = 0.25
    add_box(
        f"{name}_base",
        (base_x, pivot.y, base_top - 0.10),
        (0.96, 0.74, 0.20),
        mats["black"],
        bevel=0.045,
    )
    cylinder_between(
        f"{name}_front_support",
        (base_x - horizontal_sign * 0.29, pivot.y, base_top),
        (pivot.x, pivot.y, pivot.z),
        0.045,
        mats["steel"],
    )
    cylinder_between(
        f"{name}_rear_support",
        (base_x + horizontal_sign * 0.29, pivot.y, base_top),
        (pivot.x, pivot.y, pivot.z),
        0.045,
        mats["steel"],
    )
    add_cylinder(
        f"{name}_angle_pivot",
        tuple(pivot),
        0.155,
        0.80,
        accent,
        rotation=(math.pi / 2.0, 0.0, 0.0),
    )
    cylinder_between(f"{name}_barrel", tuple(barrel_rear), tuple(barrel_front), 0.105, mats["black"])

    # An equal rear plunger extension is a purely physical cue that the launch
    # mechanisms are matched; it is not an annotation or answer hint.
    plunger_end = barrel_rear - axis * 0.22
    cylinder_between(f"{name}_equal_plunger", tuple(barrel_rear), tuple(plunger_end), 0.060, mats["steel"])
    add_cylinder(
        f"{name}_plunger_handle",
        tuple(plunger_end),
        0.115,
        0.44,
        accent,
        rotation=(math.pi / 2.0, 0.0, 0.0),
    )
    add_sphere(f"{name}_ball", tuple(ball), BALL_RADIUS, mats["red"])

    return {
        "name": name,
        "angle_degrees": angle_deg,
        "horizontal_direction": "right" if horizontal_sign > 0 else "left",
        "ball_center": [round(float(value), 6) for value in ball],
        "ball_radius": BALL_RADIUS,
        "barrel_length": BARREL_LENGTH,
        "nominal_speed_mps": NOMINAL_SPEED_MPS,
        "relative_launch_height": round(ball.z - deck_top, 6),
    }


def _landing_lane(
    *,
    name: str,
    center: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    lane_mat: Any,
    edge_mat: Any,
) -> float:
    add_box(name, center, dimensions, lane_mat, bevel=0.03)
    top = center[2] + dimensions[2] / 2.0
    # Slim front-edge rail adds realistic thickness while keeping the landing
    # plane itself completely clear.
    add_box(
        f"{name}_front_edge",
        (center[0], center[1] - dimensions[1] / 2.0 - 0.018, center[2]),
        (dimensions[0], 0.035, dimensions[2] * 0.78),
        edge_mat,
        bevel=0.012,
    )
    return top


def _parallel_lane_scene(
    mats: dict[str, Any],
    *,
    prefix: str,
    lane_mat: Any,
    edge_mat: Any,
    accent: Any,
    front_angle: float,
    back_angle: float,
    front_y: float,
    back_y: float,
    pitch_degrees: float,
    ortho_scale: float,
) -> tuple[list[dict[str, Any]], bpy.types.Object]:
    """Build two x-parallel lanes with an exact common launch line."""
    start_x = -4.35
    lane_length = 10.20
    lane_center_x = start_x + lane_length / 2.0
    lane_center_z = 1.06
    lane_dimensions = (lane_length, 0.96, 0.18)

    deck_tops: dict[str, float] = {}
    for lane_name, lane_y in (("front", front_y), ("back", back_y)):
        deck_tops[lane_name] = _landing_lane(
            name=f"{prefix}_{lane_name}_parallel_landing_lane",
            center=(lane_center_x, lane_y, lane_center_z),
            dimensions=lane_dimensions,
            lane_mat=lane_mat,
            edge_mat=edge_mat,
        )
        # Each lane has its own identical support set.  The space above every
        # landing surface remains empty from launch point through lane end.
        for support_index, support_x in enumerate((-3.60, 0.65, 4.90), start=1):
            add_box(
                f"{prefix}_{lane_name}_support_{support_index}",
                (support_x, lane_y, 0.53),
                (0.16, 0.34, 1.02),
                mats["steel"],
                bevel=0.022,
            )

    launchers = [
        _fixed_launch_launcher(
            mats,
            name=f"{prefix}_front_{int(front_angle)}deg",
            ball_center=(start_x, front_y, deck_tops["front"] + BALL_RADIUS),
            deck_top=deck_tops["front"],
            angle_deg=front_angle,
            horizontal_sign=1,
            accent=accent,
        ),
        _fixed_launch_launcher(
            mats,
            name=f"{prefix}_back_{int(back_angle)}deg",
            ball_center=(start_x, back_y, deck_tops["back"] + BALL_RADIUS),
            deck_top=deck_tops["back"],
            angle_deg=back_angle,
            horizontal_sign=1,
            accent=accent,
        ),
    ]
    camera = _parallel_lane_camera(
        x=0.35,
        target_z=2.60,
        distance=17.0,
        pitch_degrees=pitch_degrees,
        ortho_scale=ortho_scale,
    )
    return launchers, camera


def _variant_one(mats: dict[str, Any]) -> tuple[list[dict[str, Any]], bpy.types.Object]:
    lab_shell(mats)
    lane = material("P3SOP_v1_lane", (0.58, 0.62, 0.64), roughness=0.68, metallic=0.18)
    accent = material("P3SOP_v1_blue", (0.025, 0.17, 0.46), roughness=0.34, metallic=0.55)
    return _parallel_lane_scene(
        mats,
        prefix="v1",
        lane_mat=lane,
        edge_mat=accent,
        accent=accent,
        front_angle=30.0,
        back_angle=60.0,
        front_y=-1.60,
        back_y=1.60,
        pitch_degrees=10.0,
        ortho_scale=12.50,
    )


def _variant_two(mats: dict[str, Any]) -> tuple[list[dict[str, Any]], bpy.types.Object]:
    lab_shell(mats)
    lane = material("P3SOP_v2_optical_bench", (0.12, 0.14, 0.16), roughness=0.42, metallic=0.62)
    accent = material("P3SOP_v2_orange", (0.82, 0.16, 0.018), roughness=0.38, metallic=0.25)
    area_light("v2_left_softbox", (-3.8, -2.8, 5.8), (-2.0, 0.0, 1.6), 480, 2.4)
    area_light("v2_right_softbox", (3.8, -2.8, 5.8), (2.0, 0.0, 1.6), 480, 2.4)
    return _parallel_lane_scene(
        mats,
        prefix="v2",
        lane_mat=lane,
        edge_mat=mats["aluminum"],
        accent=accent,
        front_angle=60.0,
        back_angle=30.0,
        front_y=-1.50,
        back_y=1.50,
        pitch_degrees=14.0,
        ortho_scale=12.20,
    )


def _variant_three(mats: dict[str, Any]) -> tuple[list[dict[str, Any]], bpy.types.Object]:
    lab_shell(mats)
    warm_lane = material("P3SOP_v3_warm_maple", (0.50, 0.27, 0.09), roughness=0.60)
    accent = material("P3SOP_v3_teal", (0.015, 0.34, 0.34), roughness=0.36, metallic=0.48)
    return _parallel_lane_scene(
        mats,
        prefix="v3",
        lane_mat=warm_lane,
        edge_mat=mats["aluminum"],
        accent=accent,
        front_angle=30.0,
        back_angle=60.0,
        front_y=-1.70,
        back_y=1.70,
        pitch_degrees=18.0,
        ortho_scale=12.70,
    )


BUILDERS = {1: _variant_one, 2: _variant_two, 3: _variant_three}


def _validate_scene(variant: int, launchers: list[dict[str, Any]], camera: bpy.types.Object) -> None:
    if sorted(item["angle_degrees"] for item in launchers) != [30.0, 60.0]:
        raise AssertionError("scene must contain exactly the 30 and 60 degree conditions")
    if len(launchers) != 2:
        raise AssertionError("scene must contain exactly two launchers")
    if any(item["ball_radius"] != BALL_RADIUS for item in launchers):
        raise AssertionError("ball radii differ")
    if any(item["barrel_length"] != BARREL_LENGTH for item in launchers):
        raise AssertionError("launcher barrel lengths differ")
    if any(item["nominal_speed_mps"] != NOMINAL_SPEED_MPS for item in launchers):
        raise AssertionError("nominal initial speeds differ")
    if any(abs(item["relative_launch_height"] - RELATIVE_LAUNCH_HEIGHT) > 1e-6 for item in launchers):
        raise AssertionError("ball bottoms are not tangent to their landing-lane top surfaces")
    launch_x_values = [item["ball_center"][0] for item in launchers]
    if max(launch_x_values) - min(launch_x_values) > 1e-6:
        raise AssertionError(f"launchers do not share one x start line: {launch_x_values}")
    if any(item["horizontal_direction"] != "right" for item in launchers):
        raise AssertionError("both launchers must point toward +x")
    launch_y_values = [item["ball_center"][1] for item in launchers]
    if abs(launch_y_values[0] - launch_y_values[1]) < 2.5:
        raise AssertionError("parallel lanes are not sufficiently separated along y")
    rotation = camera.matrix_world.to_quaternion()
    optical_axis = rotation @ Vector((0.0, 0.0, -1.0))
    image_up = rotation @ Vector((0.0, 1.0, 0.0))
    image_right = rotation @ Vector((1.0, 0.0, 0.0))
    pitch_degrees = math.degrees(math.atan2(-float(optical_axis.z), float(optical_axis.y)))
    if abs(float(optical_axis.x)) > 1e-6 or float(optical_axis.y) <= 0.0:
        raise AssertionError(f"camera has x-direction yaw: optical_axis={tuple(optical_axis)}")
    if not 9.999 <= pitch_degrees <= 18.001:
        raise AssertionError(f"camera pitch is outside 10--18 degrees: {pitch_degrees}")
    if (image_right - Vector((1.0, 0.0, 0.0))).length > 1e-6:
        raise AssertionError(f"camera roll is nonzero: image_right={tuple(image_right)}")
    if abs(float(image_up.x)) > 1e-6 or float(image_up.z) <= 0.0:
        raise AssertionError(f"camera image-up axis is invalid: image_up={tuple(image_up)}")
    if camera.data.type != "ORTHO":
        raise AssertionError("parallel-lane candidates must use an orthographic camera")
    balls = [obj for obj in bpy.data.objects if obj.name.endswith("_ball")]
    if len(balls) != 2:
        raise AssertionError(f"variant {variant} has {len(balls)} ball objects instead of 2")
    lanes = [obj for obj in bpy.data.objects if obj.name.endswith("_parallel_landing_lane")]
    if len(lanes) != 2:
        raise AssertionError(f"variant {variant} has {len(lanes)} landing lanes instead of 2")
    if any(abs(float(obj.rotation_euler.x)) + abs(float(obj.rotation_euler.y)) + abs(float(obj.rotation_euler.z)) > 1e-6 for obj in lanes):
        raise AssertionError("one or more landing lanes are not level and x-parallel")
    if any(obj.type == "FONT" for obj in bpy.data.objects):
        raise AssertionError("text objects are forbidden in first frames")


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
    launchers, camera = BUILDERS[variant](mats)
    bpy.context.view_layer.update()
    _validate_scene(variant, launchers, camera)
    finish_default_camera()
    bpy.context.view_layer.update()

    output = output_root / f"sim_{variant:02d}.png"
    output_root.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(output.resolve())
    bpy.ops.render.render(write_still=True)

    # For both complementary launch angles, sin(2 theta) is the same.  This is
    # kept in metadata only and is deliberately never drawn in the first frame.
    expected_ranges = {
        str(int(item["angle_degrees"])): round(
            NOMINAL_SPEED_MPS**2 * math.sin(2.0 * math.radians(item["angle_degrees"])) / GRAVITY_MPS2,
            6,
        )
        for item in launchers
    }
    rotation = camera.matrix_world.to_quaternion()
    optical_axis = rotation @ Vector((0.0, 0.0, -1.0))
    image_up = rotation @ Vector((0.0, 1.0, 0.0))
    image_right = rotation @ Vector((1.0, 0.0, 0.0))
    pitch_degrees = math.degrees(math.atan2(-float(optical_axis.z), float(optical_axis.y)))
    record = {
        "task_id": "P3",
        "candidate": variant,
        "method": "procedural_simulation",
        "variant": VARIANTS[variant],
        "file": str(output.resolve()),
        "width": bpy.context.scene.render.resolution_x,
        "height": bpy.context.scene.render.resolution_y,
        "sha256": _sha256(output),
        "renderer": f"Blender Python {bpy.app.version_string} / {engine}",
        "samples": samples,
        "elapsed_seconds": round(time.time() - started, 3),
        "camera": {
            "type": camera.data.type,
            "location": [round(float(value), 6) for value in camera.location],
            "rotation_euler_degrees": [round(math.degrees(float(value)), 6) for value in camera.rotation_euler],
            "optical_axis": [round(float(value), 6) for value in optical_axis],
            "image_up": [round(float(value), 6) for value in image_up],
            "image_right": [round(float(value), 6) for value in image_right],
            "downward_pitch_degrees": round(pitch_degrees, 6),
            "x_direction_yaw_degrees": 0.0,
            "roll_degrees": 0.0,
            "dof_enabled": bool(camera.data.dof.use_dof),
        },
        "launchers": launchers,
        "physical_parameters": {
            "gravity_mps2": GRAVITY_MPS2,
            "same_nominal_initial_speed_mps": NOMINAL_SPEED_MPS,
            "expected_level_ground_ranges_m": expected_ranges,
        },
        "hard_constraints": {
            "angles_present_degrees": [30, 60],
            "identical_balls": True,
            "identical_launchers": True,
            "same_initial_speed": True,
            "same_relative_launch_height": True,
            "common_x_start_line": True,
            "two_y_separated_parallel_lanes": True,
            "both_launch_toward_positive_x": True,
            "ball_bottom_tangent_to_lane_top": True,
            "controlled_side_elevation_degrees": round(pitch_degrees, 6),
            "zero_x_direction_yaw": True,
            "zero_camera_roll": True,
            "full_future_flight_and_landing_space": True,
            "text_arrows_formulas_trajectories_absent": True,
        },
        "external_assets": [],
        "code": str(SCRIPT_PATH),
        "code_sha256": _sha256(SCRIPT_PATH),
        "created_utc": datetime.now(timezone.utc).isoformat(),
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
        default=PROJECT_ROOT / "sop_delivery" / "P3" / "first_frames" / "simulation",
    )
    parser.add_argument("--variant", action="append", type=int, choices=sorted(VARIANTS))
    parser.add_argument("--engine", choices=("cycles", "eevee", "workbench"), default="cycles")
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--resolution-scale", type=float, default=1.0)
    args = parser.parse_args()

    variants = args.variant or sorted(VARIANTS)
    records = []
    for index, variant in enumerate(variants, start=1):
        print(f"[{index}/{len(variants)}] rendering P3 simulation candidate {variant}", flush=True)
        record = _render_variant(
            variant,
            args.output_root,
            engine=args.engine,
            samples=args.samples,
            resolution_scale=args.resolution_scale,
        )
        records.append(record)
        print(
            json.dumps(
                {
                    "candidate": variant,
                    "file": record["file"],
                    "sha256": record["sha256"],
                    "elapsed_seconds": record["elapsed_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    manifest = {
        "task_id": "P3",
        "method": "procedural_simulation",
        "requested": len(variants),
        "completed": len(records),
        "records": records,
        "code": str(SCRIPT_PATH),
        "code_sha256": _sha256(SCRIPT_PATH),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
