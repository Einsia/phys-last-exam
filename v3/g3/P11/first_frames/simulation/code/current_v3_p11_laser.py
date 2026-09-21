#!/usr/bin/env python3
"""Render the current V3 P11 air-to-water laser-refraction first frames.

The three variants deliberately share one exact optical geometry.  Only the
laboratory finish, materials, and harmless background appearance change.  The
ray geometry is generated from Snell's law rather than drawn by eye.
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
        if (parent / "simulation" / "common.py").is_file():
            return parent
    raise RuntimeError("could not locate project root containing simulation/common.py")


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
    reset_scene,
)


RENDER_WIDTH = 2048
RENDER_HEIGHT = 1152
VERSION = "current_v3_laser_preaction_v5"
N_AIR = 1.0
N_WATER = 1.333

# Keep every powered-off flashlight low and at the same far-left location.
# The incidence point is then derived from the requested angle, so the future
# beam necessarily meets the water surface safely inside the tank.
LASER_APERTURE_X = -0.425
LASER_APERTURE_Z = 0.135
OPTICAL_PLANE_Y = -0.151
WATER_DROP = 0.265625
CAMERA_TARGET = Vector((0.0275, 0.0, 0.005625))
# Blender's orthographic scale is the horizontal camera width.  At 16:9,
# 1.28 world units horizontally gives exactly 0.72 vertically.
ORTHO_SCALE = 1.28


VARIANTS: dict[int, dict[str, Any]] = {
    1: {
        "id": "modern_university_optics_lab",
        "incident_angle_deg": 30.0,
        "wall": (0.055, 0.068, 0.080),
        "bench": (0.018, 0.024, 0.032),
        "bench_metallic": 0.34,
        "hardware": (0.10, 0.12, 0.14),
        "hardware_metallic": 0.64,
        "glass": (0.22, 0.52, 0.62),
        "water": (0.015, 0.16, 0.23),
        "accent": (0.20, 0.44, 0.70),
        "warm": False,
        "dark": False,
    },
    2: {
        "id": "classic_teaching_optics_lab",
        "incident_angle_deg": 45.0,
        "wall": (0.095, 0.070, 0.045),
        "bench": (0.17, 0.060, 0.018),
        "bench_metallic": 0.0,
        "hardware": (0.16, 0.20, 0.12),
        "hardware_metallic": 0.54,
        "glass": (0.24, 0.50, 0.53),
        "water": (0.018, 0.14, 0.19),
        "accent": (0.48, 0.31, 0.08),
        "warm": True,
        "dark": False,
    },
    3: {
        "id": "industrial_fluid_optics_metrology_bay",
        "incident_angle_deg": 60.0,
        "wall": (0.018, 0.026, 0.036),
        "bench": (0.018, 0.022, 0.028),
        "bench_metallic": 0.72,
        "hardware": (0.22, 0.25, 0.28),
        "hardware_metallic": 0.86,
        "glass": (0.18, 0.45, 0.58),
        "water": (0.006, 0.11, 0.19),
        "accent": (0.035, 0.20, 0.36),
        "warm": False,
        "dark": True,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_ior(mat: bpy.types.Material, value: float) -> None:
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None and "IOR" in bsdf.inputs:
        bsdf.inputs["IOR"].default_value = value


def _optical_points(incident_angle_deg: float) -> tuple[float, float, Vector, Vector, Vector]:
    theta_i = math.radians(incident_angle_deg)
    theta_t = math.asin(N_AIR * math.sin(theta_i) / N_WATER)
    source = Vector((LASER_APERTURE_X, OPTICAL_PLANE_Y, LASER_APERTURE_Z))
    interface = Vector(
        (LASER_APERTURE_X + math.tan(theta_i) * LASER_APERTURE_Z, OPTICAL_PLANE_Y, 0.0)
    )
    transmitted_end = Vector(
        (interface.x + math.tan(theta_t) * WATER_DROP, OPTICAL_PLANE_Y, -WATER_DROP)
    )
    return theta_i, theta_t, source, interface, transmitted_end


def _world_to_pixel(point: Vector) -> list[int]:
    pixels_per_unit = RENDER_WIDTH / ORTHO_SCALE
    x = RENDER_WIDTH / 2 + (point.x - CAMERA_TARGET.x) * pixels_per_unit
    y = RENDER_HEIGHT / 2 - (point.z - CAMERA_TARGET.z) * pixels_per_unit
    return [round(x), round(y)]


def _make_materials(spec: dict[str, Any]) -> dict[str, bpy.types.Material]:
    wall = material("P11V3_wall", spec["wall"], roughness=0.82)
    bench = material(
        "P11V3_bench",
        spec["bench"],
        roughness=0.43 if spec["bench_metallic"] else 0.60,
        metallic=spec["bench_metallic"],
    )
    hardware = material(
        "P11V3_hardware",
        spec["hardware"],
        roughness=0.28,
        metallic=spec["hardware_metallic"],
    )
    glass = material(
        "P11V3_borosilicate_glass",
        spec["glass"],
        roughness=0.035,
        transmission=0.88,
        alpha=0.14,
    )
    water = material(
        "P11V3_clear_fresh_water",
        spec["water"],
        roughness=0.025,
        transmission=0.62,
        alpha=0.45,
    )
    interface = material(
        "P11V3_flat_water_interface",
        tuple(min(1.0, c * 1.45 + 0.08) for c in spec["water"]),
        roughness=0.12,
        metallic=0.02,
        transmission=0.32,
        alpha=0.72,
    )
    glass_edge = material(
        "P11V3_visible_glass_edges",
        tuple(min(1.0, c * 1.30 + 0.18) for c in spec["glass"]),
        roughness=0.30,
        metallic=0.03,
        transmission=0.0,
        alpha=1.0,
    )
    accent = material("P11V3_mount_accent", spec["accent"], roughness=0.40, metallic=0.32)
    black = material("P11V3_laser_black", (0.075, 0.088, 0.105), roughness=0.34, metallic=0.68)
    normal = material("P11V3_neutral_nonluminous_normal", (0.46, 0.48, 0.50), roughness=0.78)
    laser_core = material(
        "P11V3_635nm_laser_core",
        (0.92, 0.001, 0.001),
        roughness=0.08,
        emission=(1.0, 0.0, 0.0),
        emission_strength=2.6,
    )
    laser_halo = material(
        "P11V3_635nm_laser_scattering",
        (0.45, 0.001, 0.001),
        roughness=0.12,
        alpha=0.10,
        emission=(1.0, 0.0, 0.0),
        emission_strength=0.45,
    )
    aperture_off = material(
        "P11V3_laser_aperture_off",
        (0.14, 0.003, 0.004),
        roughness=0.18,
        metallic=0.18,
    )
    _set_ior(glass, 1.46)
    _set_ior(water, N_WATER)
    return {
        "wall": wall,
        "bench": bench,
        "hardware": hardware,
        "glass": glass,
        "water": water,
        "interface": interface,
        "glass_edge": glass_edge,
        "accent": accent,
        "black": black,
        "normal": normal,
        "laser_core": laser_core,
        "laser_halo": laser_halo,
        "aperture_off": aperture_off,
    }


def _build_shell(mats: dict[str, bpy.types.Material], spec: dict[str, Any]) -> None:
    # Only broad, non-linear background shapes are used so nothing can be
    # mistaken for an extra ray.
    add_box("P11V3_background_wall", (0.0275, 0.31, 0.03), (1.42, 0.030, 0.82), mats["wall"], bevel=0.004)
    add_box("P11V3_bench_top", (0.0275, 0.0, -0.326), (1.32, 0.56, 0.040), mats["bench"], bevel=0.008)


def _build_tank(mats: dict[str, bpy.types.Material]) -> None:
    inner_left = -0.5000
    inner_right = 0.5550
    inner_bottom = -0.303125
    outer_top = 0.2940
    wall_t = 0.0080
    depth = 0.280
    center_x = (inner_left + inner_right) / 2.0
    inner_width = inner_right - inner_left
    center_z = (inner_bottom + outer_top) / 2.0
    height = outer_top - inner_bottom

    add_box(
        "P11V3_clear_fresh_water_body",
        (center_x, 0.0, inner_bottom / 2.0),
        (inner_width, depth - 2 * wall_t, -inner_bottom),
        mats["water"],
        bevel=0.002,
    )
    add_box("P11V3_tank_bottom", (center_x, 0.0, inner_bottom - wall_t / 2), (inner_width + 2 * wall_t, depth, wall_t), mats["glass"], bevel=0.0015)
    add_box("P11V3_tank_left_wall", (inner_left - wall_t / 2, 0.0, center_z), (wall_t, depth, height), mats["glass"], bevel=0.0015)
    add_box("P11V3_tank_right_wall", (inner_right + wall_t / 2, 0.0, center_z), (wall_t, depth, height), mats["glass"], bevel=0.0015)
    # Side-view cutaway: do not place a broad front pane over the laser body.
    # The three front-panel edges below still make the transparent tank outline
    # explicit while preserving visibility of the pre-action apparatus.
    add_box("P11V3_tank_back_panel", (center_x, depth / 2, center_z), (inner_width + 2 * wall_t, wall_t, height), mats["glass"], bevel=0.0015)
    add_box("P11V3_calm_horizontal_air_water_interface", (center_x, -0.003, 0.0), (inner_width, depth - 0.016, 0.0022), mats["interface"], bevel=0.0004)
    add_box("P11V3_front_meniscus_line", (center_x, -depth / 2 - 0.0045, 0.0), (inner_width, 0.0022, 0.0016), mats["interface"], bevel=0.0003)
    # Crisp front-panel edges keep the transparent vessel measurable against a
    # dark background without turning it into an opaque frame.
    front_y = -depth / 2 - 0.0048
    add_box("P11V3_visible_left_glass_edge", (inner_left - wall_t / 2, front_y, center_z), (wall_t * 1.15, 0.0025, height), mats["glass_edge"], bevel=0.0010)
    add_box("P11V3_visible_right_glass_edge", (inner_right + wall_t / 2, front_y, center_z), (wall_t * 1.15, 0.0025, height), mats["glass_edge"], bevel=0.0010)
    add_box("P11V3_visible_bottom_glass_edge", (center_x, front_y, inner_bottom - wall_t / 2), (inner_width + 2 * wall_t, 0.0025, wall_t * 1.15), mats["glass_edge"], bevel=0.0010)


def _build_laser_and_mount(
    mats: dict[str, bpy.types.Material], source: Vector, interface: Vector
) -> None:
    direction = (interface - source).normalized()
    barrel_back = source - direction * 0.078
    cylinder_between("P11V3_single_laser_source", tuple(barrel_back), tuple(source), 0.018, mats["black"])
    cylinder_between("P11V3_laser_source_accent", tuple(barrel_back + direction * 0.010), tuple(barrel_back + direction * 0.030), 0.019, mats["accent"])
    cylinder_between("P11V3_laser_source_front_ring", tuple(source - direction * 0.014), tuple(source - direction * 0.004), 0.0195, mats["hardware"])
    add_sphere("P11V3_single_laser_aperture_off", tuple(source), 0.0060, mats["aperture_off"], segments=32)

    bench_top = -0.306
    barrel_mid = (barrel_back + source) / 2.0
    post_x = barrel_mid.x - 0.034
    post_y = -0.165
    arm_z = barrel_mid.z + 0.048
    post_depth = arm_z - bench_top
    add_cylinder("P11V3_mount_post", (post_x, post_y, bench_top + post_depth / 2.0), 0.0085, post_depth, mats["hardware"])
    cylinder_between("P11V3_mount_top_arm", (post_x, post_y, arm_z), (barrel_mid.x, post_y, arm_z), 0.0070, mats["hardware"])
    cylinder_between("P11V3_mount_short_clamp", (barrel_mid.x, post_y, arm_z), (barrel_mid.x, post_y, barrel_mid.z), 0.0060, mats["hardware"])
    add_box("P11V3_mount_base", (post_x, post_y, bench_top + 0.009), (0.075, 0.090, 0.018), mats["black"], bevel=0.005)


def _add_beam_segment(name: str, start: Vector, end: Vector, mats: dict[str, bpy.types.Material]) -> None:
    cylinder_between(f"{name}_halo", tuple(start), tuple(end), 0.0021, mats["laser_halo"])
    cylinder_between(f"{name}_core", tuple(start), tuple(end), 0.00110, mats["laser_core"])


def _build_future_optical_geometry_debug(
    mats: dict[str, bpy.types.Material],
    source: Vector,
    interface: Vector,
    transmitted_end: Vector,
) -> None:
    _add_beam_segment("P11V3_incident_ray_in_air", source, interface, mats)
    _add_beam_segment("P11V3_refracted_ray_in_water", interface, transmitted_end, mats)
    add_sphere("P11V3_single_incidence_spot", tuple(interface), 0.0028, mats["laser_core"], segments=32)


def _build_interface_normal(mats: dict[str, bpy.types.Material], interface: Vector) -> None:
    # A short neutral-gray dashed calibration normal on the front panel.  It is
    # intentionally non-luminous and does not reveal the future refracted ray.
    dash = 0.0075
    gap = 0.0050
    for side in (-1, 1):
        cursor = 0.0060
        index = 0
        while cursor + dash <= 0.0601:
            z0 = side * cursor
            z1 = side * (cursor + dash)
            cylinder_between(
                f"P11V3_normal_{'air' if side > 0 else 'water'}_{index:02d}",
                (interface.x, -0.1545, z0),
                (interface.x, -0.1545, z1),
                0.00072,
                mats["normal"],
            )
            cursor += dash + gap
            index += 1


def _configure_camera_and_light(spec: dict[str, Any]) -> None:
    camera = add_camera((CAMERA_TARGET.x, -2.2, CAMERA_TARGET.z), tuple(CAMERA_TARGET), lens=70.0)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = ORTHO_SCALE
    camera.data.dof.use_dof = False

    warm = bool(spec["warm"])
    area_light(
        "P11V3_key_light",
        (-0.36, -0.65, 0.58),
        (0.0, 0.0, -0.02),
        25.0 if not spec["dark"] else 20.0,
        0.42,
        (1.0, 0.86, 0.68) if warm else (0.82, 0.91, 1.0),
    )
    area_light(
        "P11V3_fill_light",
        (0.50, -0.28, 0.34),
        (0.12, 0.0, -0.08),
        12.0 if not spec["dark"] else 10.0,
        0.32,
        (0.86, 0.91, 1.0),
    )


def _render_variant(
    variant: int,
    output_dir: Path,
    engine: str,
    samples: int,
    debug_rays: bool,
) -> dict[str, Any]:
    started = time.time()
    spec = VARIANTS[variant]
    theta_i, theta_t, source, interface, transmitted_end = _optical_points(spec["incident_angle_deg"])
    reset_scene(engine, samples=samples)
    scene = bpy.context.scene
    scene.render.resolution_x = RENDER_WIDTH
    scene.render.resolution_y = RENDER_HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 8
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    if hasattr(scene.render, "use_motion_blur"):
        scene.render.use_motion_blur = False
    if scene.world is not None:
        scene.world.color = spec["wall"]
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        if bg is not None:
            bg.inputs["Color"].default_value = (*spec["wall"], 1.0)
            bg.inputs["Strength"].default_value = 0.10 if spec["dark"] else 0.14
    scene.view_settings.exposure = -0.25 if spec["dark"] else -0.40

    mats = _make_materials(spec)
    _build_shell(mats, spec)
    _build_tank(mats)
    _build_laser_and_mount(mats, source, interface)
    if debug_rays:
        _build_interface_normal(mats, interface)
        _build_future_optical_geometry_debug(mats, source, interface, transmitted_end)
    _configure_camera_and_light(spec)
    bpy.context.view_layer.update()

    output = output_dir / f"sim_{variant:02d}.png"
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)

    snell_residual = abs(N_AIR * math.sin(theta_i) - N_WATER * math.sin(theta_t))
    record = {
        "task_id": "P11",
        "task_definition_zh": "一束细激光从空气斜射入透明水槽中的清水，入射光、折射光和界面均清晰可见。",
        "method": "procedural_blender_simulation_pre_action",
        "version": VERSION,
        "candidate": variant,
        "variant": spec["id"],
        "frame_state": "pre_action_laser_power_off",
        "file": str(output),
        "width": RENDER_WIDTH,
        "height": RENDER_HEIGHT,
        "sha256": _sha256(output),
        "renderer": f"Blender Python {'.'.join(map(str, bpy.app.version))} / {engine}",
        "samples": samples,
        "elapsed_seconds": round(time.time() - started, 3),
        "optics": {
            "medium_from": "air",
            "medium_to": "clear_fresh_water",
            "n_air": N_AIR,
            "n_water": N_WATER,
            "incident_angle_from_normal_deg": spec["incident_angle_deg"],
            "refracted_angle_from_normal_deg": math.degrees(theta_t),
            "snell_residual": snell_residual,
            "source_world_xyz": [round(v, 9) for v in source],
            "incidence_world_xyz": [round(v, 9) for v in interface],
            "transmitted_end_world_xyz": [round(v, 9) for v in transmitted_end],
            "future_expected_pixels_after_laser_turns_on": {
                "laser_aperture": _world_to_pixel(source),
                "incidence": _world_to_pixel(interface),
                "transmitted_end": _world_to_pixel(transmitted_end),
                "waterline_y": 585,
            },
        },
        "hard_constraints": {
            "exactly_one_laser_source": True,
            "laser_initially_power_off": not debug_rays,
            "visible_incident_segment_absent_in_first_frame": not debug_rays,
            "visible_refracted_segment_absent_in_first_frame": not debug_rays,
            "outcome_not_revealed_in_first_frame": not debug_rays,
            "laser_axis_aims_at_future_interface_point": True,
            "laser_position_shared_far_left_across_variants": True,
            "laser_lowered_with_clear_top_margin": True,
            "future_interface_hit_has_side_wall_clearance": True,
            "future_refracted_ray_bends_toward_normal": True,
            "flat_horizontal_interface": True,
            "interface_normal_absent_in_first_frame": not debug_rays,
            "measurement_overlays_absent_in_first_frame": not debug_rays,
            "camera_locked_orthographic_side_view": True,
            "people_text_logos_absent": True,
        },
        "code": str(SCRIPT_PATH),
        "code_sha256": _sha256(SCRIPT_PATH),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata = output_dir / f"sim_{variant:02d}.json"
    metadata.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"sop_delivery/P11/{VERSION}/simulation"),
    )
    parser.add_argument("--variant", type=int, action="append", choices=tuple(VARIANTS))
    parser.add_argument("--engine", choices=("eevee", "cycles"), default="eevee")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument(
        "--debug-rays",
        action="store_true",
        help="Render future Snell rays for geometry debugging only; never use as a formal first frame.",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = list(dict.fromkeys(args.variant or sorted(VARIANTS)))
    records: list[dict[str, Any]] = []
    for variant in requested:
        record = _render_variant(variant, output_dir, args.engine, args.samples, args.debug_rays)
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    manifest = {
        "task_id": "P11",
        "version": VERSION,
        "method": "procedural_blender_simulation_pre_action",
        "frame_state": "debug_future_rays" if args.debug_rays else "pre_action_laser_power_off",
        "requested": len(requested),
        "completed": len(records),
        "failed": 0,
        "shared_optics": records[0]["optics"] if records else {},
        "records": records,
        "code": str(SCRIPT_PATH),
        "code_sha256": _sha256(SCRIPT_PATH),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"requested": len(requested), "completed": len(records), "failed": 0}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
