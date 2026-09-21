#!/usr/bin/env python3
"""High-standard, auditable simulation sources for Physical-Bench P4 and P6.

The images contain only the initial state.  P4 shows one rubber ball still
held by an electromagnetic release above a hard impact plate.  P6 shows one
asymmetrically marked ball still held against an incline by a thin gate.  No
motion trails, outcome cues, arrows, labels, formulas, or text are rendered.

This module is deliberately separate from ``builders_mech_optics.py`` so the
legacy full-batch builders and their already published output remain intact.
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
    add_torus,
    add_tube,
    area_light,
    cylinder_between,
    finish_default_camera,
    lab_shell,
    material,
    palette,
    ramp_point,
    reset_scene,
)


RENDER_WIDTH = 2048
RENDER_HEIGHT = 1152

VARIANTS: dict[int, dict[str, str]] = {
    1: {
        "theme": "modern",
        "name": "modern_university_lab",
        "description": "Contemporary university laboratory, cool daylight and a right three-quarter camera.",
    },
    2: {
        "theme": "classic",
        "name": "classic_teaching_lab",
        "description": "Older teaching laboratory, warm window light and an opposite-side camera.",
    },
    3: {
        "theme": "industrial",
        "name": "industrial_test_bay",
        "description": "Full-scale industrial test bay, practical work lights and a near-frontal elevated camera.",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_render(engine: str, samples: int, resolution_scale: float, device: str) -> dict[str, Any]:
    if not 0.25 <= resolution_scale <= 1.0:
        raise ValueError("resolution_scale must be within 0.25--1.0")
    reset_scene(engine, samples=samples)
    scene = bpy.context.scene
    if engine == "cycles" and device == "cpu":
        # Explicit fallback for shared servers where another render may fill
        # VRAM between two task renders.  Geometry and pixels stay reproducible.
        scene.cycles.device = "CPU"
    scene.render.resolution_x = int(round(RENDER_WIDTH * resolution_scale))
    scene.render.resolution_y = int(round(RENDER_HEIGHT * resolution_scale))
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 10
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    # Modest path depth is enough for these opaque scenes and materially cuts
    # Cycles startup/render cost without changing the physical geometry.
    if engine == "cycles":
        scene.cycles.max_bounces = 5
        scene.cycles.diffuse_bounces = 3
        scene.cycles.glossy_bounces = 3
    return palette()


def _noise_pbr_material(
    name: str,
    color: tuple[float, float, float],
    *,
    roughness: float,
    metallic: float = 0.0,
    noise_scale: float = 7.0,
    bump_strength: float = 0.12,
) -> bpy.types.Material:
    """Small procedural PBR material used to avoid sterile toy surfaces."""
    mat = material(name, color, roughness=roughness, metallic=metallic)
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = noise_scale
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.72
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = bump_strength
    bump.inputs["Distance"].default_value = 0.075
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _asymmetric_band_material(
    name: str,
    base_color: tuple[float, float, float],
    band_color: tuple[float, float, float],
) -> bpy.types.Material:
    """Smooth shader-space diagonal band, with no polygon-staircase edge."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.61

    texcoord = nodes.new("ShaderNodeTexCoord")
    subtract = nodes.new("ShaderNodeVectorMath")
    subtract.operation = "SUBTRACT"
    subtract.inputs[1].default_value = (0.5, 0.5, 0.5)
    normalize = nodes.new("ShaderNodeVectorMath")
    normalize.operation = "NORMALIZE"
    dot = nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    dot.inputs[1].default_value = Vector((0.58, 0.24, 0.78)).normalized()
    absolute = nodes.new("ShaderNodeMath")
    absolute.operation = "ABSOLUTE"
    threshold = nodes.new("ShaderNodeMath")
    threshold.name = "P6_smooth_asymmetric_band_mask"
    threshold.operation = "LESS_THAN"
    threshold.inputs[1].default_value = 0.23
    color_mix = nodes.new("ShaderNodeMixRGB")
    color_mix.blend_type = "MIX"
    color_mix.inputs[1].default_value = (*base_color, 1.0)
    color_mix.inputs[2].default_value = (*band_color, 1.0)
    links.new(texcoord.outputs["Generated"], subtract.inputs[0])
    links.new(subtract.outputs["Vector"], normalize.inputs[0])
    links.new(normalize.outputs["Vector"], dot.inputs[0])
    links.new(dot.outputs["Value"], absolute.inputs[0])
    links.new(absolute.outputs[0], threshold.inputs[0])
    links.new(threshold.outputs[0], color_mix.inputs[0])
    links.new(color_mix.outputs["Color"], bsdf.inputs["Base Color"])

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 23.0
    noise.inputs["Detail"].default_value = 4.0
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.13
    bump.inputs["Distance"].default_value = 0.055
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def _studio_environment(mats: dict[str, Any], *, theme: str) -> None:
    if theme not in {"modern", "classic", "industrial"}:
        raise ValueError(f"unknown environment theme: {theme}")
    lab_shell(mats, dark=theme == "industrial")
    # The shared shell defaults to a deliberately bright generic exposure.
    # Four soft sources at that setting bleach dark PBR finishes; a neutral
    # exposure keeps rubber, stone and powder coat materially distinguishable.
    bpy.context.scene.view_settings.exposure = 0.35 if theme == "industrial" else 0.0
    if theme == "modern":
        wall = material("modern_pale_lab_wall", (0.69, 0.72, 0.73), roughness=0.88)
        floor = material("modern_resin_floor", (0.25, 0.28, 0.30), roughness=0.70)
        trim = material("modern_wall_trim", (0.12, 0.14, 0.16), roughness=0.70, metallic=0.18)
        light_color = (0.80, 0.91, 1.0)
        front_light = (-4.3, -4.7, 7.2)
        rim_light = (4.8, 1.4, 6.0)
    elif theme == "classic":
        wall = _noise_pbr_material(
            "classic_worn_plaster",
            (0.52, 0.43, 0.31),
            roughness=0.94,
            noise_scale=5.0,
            bump_strength=0.08,
        )
        floor = mats["wood_dark"]
        trim = material("classic_dark_brass_trim", (0.25, 0.15, 0.045), roughness=0.48, metallic=0.54)
        light_color = (1.0, 0.73, 0.48)
        front_light = (4.8, -4.2, 6.7)
        rim_light = (-4.5, 1.2, 5.4)
        # One distant, non-overlapping cabinet makes the teaching-room scale
        # legible without inserting any object into the task's motion region.
        add_box("classic_background_cabinet", (5.2, 3.12, 1.55), (1.55, 0.32, 2.65), mats["wood_dark"], bevel=0.045)
    else:
        wall = _noise_pbr_material(
            "industrial_concrete_wall",
            (0.105, 0.12, 0.135),
            roughness=0.91,
            noise_scale=4.0,
            bump_strength=0.15,
        )
        floor = _noise_pbr_material(
            "industrial_rubber_floor",
            (0.035, 0.045, 0.052),
            roughness=0.84,
            noise_scale=14.0,
            bump_strength=0.09,
        )
        trim = material("industrial_safety_blue_trim", (0.015, 0.10, 0.24), roughness=0.40, metallic=0.50)
        light_color = (0.68, 0.82, 1.0)
        front_light = (-1.0, -5.5, 7.8)
        rim_light = (5.5, 0.4, 6.6)
    bpy.data.objects["rear_wall"].data.materials[0] = wall
    bpy.data.objects["lab_floor"].data.materials[0] = floor
    # A restrained wall/floor seam and a low equipment plinth give real scale
    # without introducing labels or task-irrelevant apparatus.
    add_box("rear_wall_trim", (0.0, 3.37, 0.86), (13.7, 0.06, 0.12), trim, bevel=0.018)
    area_light("front_softbox", front_light, (-0.8, 0.0, 2.4), 620, 3.1, light_color)
    area_light("rim_softbox", rim_light, (0.5, 0.0, 2.5), 390, 2.5, (0.72, 0.84, 1.0))


def _p4_scene(mats: dict[str, Any], variant: int) -> dict[str, Any]:
    """One orange ball held high above a hard plate, before release."""
    theme = VARIANTS[variant]["theme"]
    _studio_environment(mats, theme=theme)
    # Keep the entire future vertical corridor visually empty.  The generic
    # rear-wall trim is useful for scale in P6 but would cross this corridor.
    bpy.data.objects["rear_wall_trim"].hide_render = True
    orange_colors = {
        1: (0.88, 0.115, 0.012),
        2: (0.70, 0.075, 0.012),
        3: (0.96, 0.17, 0.006),
    }
    plate_specs = {
        1: ((0.17, 0.19, 0.21), 0.73, 0.04),
        2: ((0.075, 0.065, 0.052), 0.61, 0.52),
        3: ((0.025, 0.032, 0.041), 0.36, 0.82),
    }
    frame_specs = {
        1: ((0.075, 0.10, 0.13), 0.38, 0.58),
        2: ((0.105, 0.17, 0.055), 0.57, 0.42),
        3: ((0.012, 0.045, 0.145), 0.32, 0.72),
    }
    orange_rubber = _noise_pbr_material(
        "P4_orange_rubber",
        orange_colors[variant],
        roughness=0.64,
        noise_scale=19.0,
        bump_strength=0.16,
    )
    plate_color, plate_roughness, plate_metallic = plate_specs[variant]
    hard_plate = _noise_pbr_material(
        "P4_hard_impact_stone",
        plate_color,
        roughness=plate_roughness,
        metallic=plate_metallic,
        noise_scale=8.0,
        bump_strength=0.11,
    )
    frame_color, frame_roughness, frame_metallic = frame_specs[variant]
    painted_steel = _noise_pbr_material(
        "P4_painted_release_frame",
        frame_color,
        roughness=frame_roughness,
        metallic=frame_metallic,
        noise_scale=22.0,
        bump_strength=0.055,
    )
    coil_colors = {1: (0.53, 0.16, 0.045), 2: (0.55, 0.32, 0.075), 3: (0.66, 0.19, 0.035)}
    copper = material("P4_visible_copper_coil", coil_colors[variant], roughness=0.27, metallic=0.82)

    plate_center = (-0.75, 0.0, 0.50)
    plate_size = (6.80, 2.55, 0.28)
    plate_top = plate_center[2] + plate_size[2] / 2.0
    add_box("P4_full_hard_impact_plate", plate_center, plate_size, hard_plate, bevel=0.045)
    # Four compact rubber feet make the rigid plate mechanically plausible.
    for index, (x, y) in enumerate(((-3.55, -0.82), (-3.55, 0.82), (2.05, -0.82), (2.05, 0.82)), start=1):
        add_cylinder(f"P4_plate_foot_{index}", (x, y, 0.26), 0.11, 0.30, mats["rubber"])

    ball_radius = 0.34
    ball_center = Vector((-0.75, -0.04, 4.92))
    add_sphere("P4_single_orange_rubber_ball", tuple(ball_center), ball_radius, orange_rubber, segments=64)

    # A small vertical electromagnet touches the ball at its top.  The rigid
    # column and cantilever are outside the fall corridor, leaving the full
    # future vertical path and the impact plate plainly visible.
    magnet_radius = 0.19
    magnet_depth = 0.36
    magnet_center = Vector((ball_center.x, ball_center.y, ball_center.z + ball_radius + magnet_depth / 2.0))
    add_cylinder("P4_small_electromagnetic_release", tuple(magnet_center), magnet_radius, magnet_depth, painted_steel)
    add_torus(
        "P4_release_copper_coil",
        (magnet_center.x, magnet_center.y, magnet_center.z + 0.02),
        major_radius=0.205,
        minor_radius=0.030,
        mat=copper,
        rotation=(0.0, 0.0, 0.0),
    )
    column_x = {1: 3.25, 2: -4.70, 3: 3.55}[variant]
    column_z = 2.85
    add_box("P4_release_column", (column_x, 0.18, column_z), (0.18, 0.66, 5.55), painted_steel, bevel=0.035)
    add_box("P4_release_column_base", (column_x, 0.18, 0.30), (1.05, 1.20, 0.22), painted_steel, bevel=0.055)
    arm_z = magnet_center.z + magnet_depth / 2.0 + 0.09
    if variant == 2:
        add_box(
            "P4_release_cantilever",
            ((column_x + ball_center.x) / 2.0, ball_center.y, arm_z),
            (abs(column_x - ball_center.x), 0.19, 0.15),
            painted_steel,
            bevel=0.028,
        )
    elif variant == 3:
        for rail_y in (-0.14, 0.14):
            cylinder_between(
                f"P4_release_cantilever_{rail_y:+.2f}",
                (column_x, rail_y, arm_z),
                (ball_center.x, rail_y, arm_z),
                0.060,
                mats["steel"],
            )
        cylinder_between(
            "P4_industrial_column_brace",
            (column_x - 0.46, 0.18, 0.43),
            (column_x, 0.18, 4.15),
            0.065,
            painted_steel,
        )
    else:
        cylinder_between(
            "P4_release_cantilever",
            (column_x, ball_center.y, arm_z),
            (ball_center.x, ball_center.y, arm_z),
            0.075,
            mats["steel"],
        )
    cylinder_between(
        "P4_magnet_hanger",
        (ball_center.x, ball_center.y, arm_z),
        (ball_center.x, ball_center.y, magnet_center.z + magnet_depth / 2.0),
        0.055,
        mats["steel"],
    )
    add_tube(
        "P4_release_power_cable",
        (
            (ball_center.x + 0.08, ball_center.y + 0.04, arm_z + 0.03),
            (0.55, ball_center.y + 0.04, arm_z + 0.08),
            (column_x + 0.14, 0.05, arm_z - 0.04),
            (column_x + 0.22, 0.05, 1.25),
        ),
        0.026,
        mats["rubber"],
        resolution=2,
    )

    # A long-lens, slight three-quarter camera reads as a laboratory photo but
    # keeps the release head, entire clear fall corridor, and full plate in one
    # uncropped frame.
    camera_specs = {
        1: ((7.40, -18.30, 6.10), (-0.25, 0.0, 2.78), 58.0),
        2: ((-8.20, -18.50, 5.65), (-0.30, 0.0, 2.72), 56.0),
        3: ((0.10, -21.00, 7.00), (-0.40, 0.0, 2.60), 55.0),
    }
    camera_location, camera_target, camera_lens = camera_specs[variant]
    camera = add_camera(camera_location, camera_target, lens=camera_lens)
    camera.data.dof.use_dof = False

    corridor_height = float(ball_center.z - ball_radius - plate_top)
    magnet_ball_contact_error = abs(
        float(magnet_center.z - magnet_depth / 2.0 - (ball_center.z + ball_radius))
    )
    return {
        "variant": {**VARIANTS[variant], "index": variant},
        "camera": camera,
        "physical_parameters": {
            "ball_center_m": [round(float(value), 6) for value in ball_center],
            "ball_radius_m": ball_radius,
            "impact_plate_top_z_m": round(plate_top, 6),
            "clear_vertical_corridor_m": round(corridor_height, 6),
            "magnet_center_m": [round(float(value), 6) for value in magnet_center],
            "magnet_depth_m": magnet_depth,
            "magnet_ball_contact_error_m": round(magnet_ball_contact_error, 9),
        },
        "hard_constraints": {
            "exactly_one_orange_rubber_ball": True,
            "ball_held_by_small_electromagnetic_release": True,
            "magnet_ball_contact_exact": True,
            "ball_not_yet_released": True,
            "complete_clear_fall_and_rebound_corridor_visible": True,
            "complete_hard_impact_plate_visible": True,
        },
    }


def _painted_ball_patch(
    name: str,
    ball_center: Vector,
    ball_radius: float,
    outward_normal: Vector,
    patch_radius: float,
    patch_mat: Any,
) -> None:
    """Place one thin, tangent paint disk on the camera-facing hemisphere."""
    normal = outward_normal.normalized()
    center = ball_center + normal * (ball_radius + 0.006)
    half_depth = 0.009
    cylinder_between(name, tuple(center - normal * half_depth), tuple(center + normal * half_depth), patch_radius, patch_mat)


def _add_asymmetric_banded_ball(
    name: str,
    center: Vector,
    radius: float,
    patterned_mat: Any,
    accent_mat: Any,
) -> None:
    """Create one sphere with a diagonal painted band and one offset marker.

    The band is evaluated in shader space, rather than built as a raised ring,
    so the ball remains exactly spherical and tangent geometry is unchanged.
    Its diagonal orientation plus one low/right marker is useful for rotation
    tracking but cannot read as two eyes and a mouth.
    """
    add_sphere(name, tuple(center), radius, patterned_mat, segments=96)
    _painted_ball_patch(
        "P6_single_offset_accent_patch",
        center,
        radius,
        Vector((0.38, -0.88, -0.28)),
        radius * 0.18,
        accent_mat,
    )


def _p6_scene(mats: dict[str, Any], variant: int) -> dict[str, Any]:
    """One asymmetrically marked ball tangent to a ramp and held by a gate."""
    theme = VARIANTS[variant]["theme"]
    _studio_environment(mats, theme=theme)
    ramp_angle_deg = 20.0
    angle = math.radians(ramp_angle_deg)
    ramp_length = 8.60
    ramp_width = 1.88
    ramp_thickness = 0.22
    desired_low_surface_z = 0.68
    center_z = desired_low_surface_z + math.sin(angle) * ramp_length / 2.0 - math.cos(angle) * ramp_thickness / 2.0
    ramp_center = (0.0, 0.0, center_z)

    if variant == 2:
        rough_ramp = mats["wood"]
    else:
        ramp_specs = {
            1: ((0.075, 0.095, 0.115), 0.77, 0.08),
            3: ((0.012, 0.050, 0.145), 0.48, 0.48),
        }
        ramp_color, ramp_roughness, ramp_metallic = ramp_specs[variant]
        rough_ramp = _noise_pbr_material(
            "P6_rough_ramp_surface",
            ramp_color,
            roughness=ramp_roughness,
            metallic=ramp_metallic,
            noise_scale=17.0,
            bump_strength=0.19,
        )
    frame_specs = {
        1: ((0.025, 0.09, 0.18), 0.38, 0.54),
        2: ((0.075, 0.055, 0.035), 0.55, 0.48),
        3: ((0.015, 0.018, 0.024), 0.31, 0.78),
    }
    frame_color, frame_roughness, frame_metallic = frame_specs[variant]
    frame_mat = _noise_pbr_material(
        "P6_support_frame",
        frame_color,
        roughness=frame_roughness,
        metallic=frame_metallic,
        noise_scale=25.0,
        bump_strength=0.05,
    )
    gate_specs = {
        1: ((0.76, 0.018, 0.008), 0.38, 0.48),
        2: ((0.012, 0.015, 0.020), 0.45, 0.64),
        3: ((0.95, 0.29, 0.008), 0.34, 0.42),
    }
    gate_color, gate_roughness, gate_metallic = gate_specs[variant]
    gate_mat = material("P6_high_contrast_opaque_stop_gate", gate_color, roughness=gate_roughness, metallic=gate_metallic)
    ball_specs = {
        1: ((0.76, 0.60, 0.27), (0.012, 0.025, 0.045), (0.58, 0.02, 0.01)),
        2: ((0.72, 0.66, 0.50), (0.025, 0.12, 0.045), (0.48, 0.018, 0.012)),
        3: ((0.90, 0.49, 0.018), (0.008, 0.010, 0.014), (0.88, 0.88, 0.82)),
    }
    ball_color, band_color, accent_color = ball_specs[variant]
    ball_mat = _asymmetric_band_material(
        "P6_patterned_ball_surface_with_smooth_band",
        ball_color,
        band_color,
    )
    accent_mat = material("P6_single_asymmetric_accent", accent_color, roughness=0.56)

    add_box(
        "P6_complete_rough_incline",
        ramp_center,
        (ramp_length, ramp_width, ramp_thickness),
        rough_ramp,
        rotation=(0.0, angle, 0.0),
        bevel=0.045,
    )
    # Side rails remain lower than the ball, so they reveal the full slope but
    # cannot be mistaken for the thin start gate.
    for side_y in (-ramp_width / 2.0 - 0.025, ramp_width / 2.0 + 0.025):
        side_center = ramp_point(ramp_center, angle, 0.0, side_y, ramp_thickness / 2.0 + 0.055)
        add_box(
            f"P6_low_side_rail_{side_y:+.3f}",
            tuple(side_center),
            (ramp_length, 0.055, 0.11),
            frame_mat,
            rotation=(0.0, angle, 0.0),
            bevel=0.018,
        )

    # Structurally plausible supports under three ramp stations.
    for index, u in enumerate((-3.35, 0.0, 3.30), start=1):
        underside = ramp_point(ramp_center, angle, u, 0.0, -ramp_thickness / 2.0)
        cylinder_between(
            f"P6_support_leg_{index}",
            (underside.x, -0.58, 0.25),
            (underside.x, -0.58, underside.z),
            0.065,
            mats["steel"],
        )
        cylinder_between(
            f"P6_support_crossbar_{index}",
            (underside.x, -0.72, underside.z),
            (underside.x, 0.72, underside.z),
            0.052,
            frame_mat,
        )
        add_box(f"P6_support_foot_{index}", (underside.x, -0.58, 0.18), (0.64, 0.38, 0.10), mats["rubber"], bevel=0.03)

    # A larger but still apparatus-scale ball makes both contact relations
    # auditable while the complete incline and runout remain in frame.
    ball_radius = 0.55
    ball_u = -2.55
    surface_normal_offset = ramp_thickness / 2.0 + ball_radius
    ball_center = ramp_point(ramp_center, angle, ball_u, 0.0, surface_normal_offset)
    _add_asymmetric_banded_ball(
        "P6_single_asymmetrically_patterned_ball",
        ball_center,
        ball_radius,
        ball_mat,
        accent_mat,
    )

    # The gate's upstream face is tangent to the ball along the ramp tangent.
    # It rotates with the ramp so it remains perpendicular to the surface.
    gate_thickness = 0.10
    gate_height = 0.72
    gate_u = ball_u + ball_radius + gate_thickness / 2.0
    gate_y = -0.12
    gate_width = 0.50
    gate_center = ramp_point(
        ramp_center,
        angle,
        gate_u,
        gate_y,
        ramp_thickness / 2.0 + gate_height / 2.0,
    )
    add_box(
        "P6_thin_release_gate",
        tuple(gate_center),
        (gate_thickness, gate_width, gate_height),
        gate_mat,
        rotation=(0.0, angle, 0.0),
        bevel=0.012,
    )
    # The blade slots directly into the two side rails.  No near-side actuator
    # is drawn: it would occlude the two contact boundaries being evaluated.

    low_end_surface = ramp_point(ramp_center, angle, ramp_length / 2.0, 0.0, ramp_thickness / 2.0)
    runout_start_x = float(low_end_surface.x) - 0.03
    runout_length = 2.55
    runout_thickness = 0.18
    runout_center = (runout_start_x + runout_length / 2.0, 0.0, desired_low_surface_z - runout_thickness / 2.0)
    add_box(
        "P6_complete_level_runout",
        runout_center,
        (runout_length, ramp_width, runout_thickness),
        rough_ramp,
        bevel=0.045,
    )
    for side_y in (-ramp_width / 2.0 - 0.025, ramp_width / 2.0 + 0.025):
        add_box(
            f"P6_runout_side_rail_{side_y:+.3f}",
            (runout_center[0], side_y, desired_low_surface_z + 0.055),
            (runout_length, 0.055, 0.11),
            frame_mat,
            bevel=0.018,
        )

    # Long-lens, lightly elevated cameras greatly reduce the previous skew and
    # see over the near rail, exposing both the ramp contact and stop-gate
    # contact.  Small left/right differences still make three real viewpoints.
    camera_specs = {
        1: ((3.0, -28.0, 9.2), (1.0, 0.0, 1.85), 76.0),
        2: ((-3.0, -27.0, 8.4), (1.0, 0.0, 1.90), 72.0),
        3: ((0.0, -29.0, 10.0), (1.0, 0.0, 1.80), 78.0),
    }
    camera_location, camera_target, camera_lens = camera_specs[variant]
    camera = add_camera(camera_location, camera_target, lens=camera_lens)
    camera.data.dof.use_dof = False

    tangent = Vector((math.cos(angle), 0.0, -math.sin(angle)))
    normal = Vector((math.sin(angle), 0.0, math.cos(angle)))
    contact_point = ball_center - normal * ball_radius
    ball_downhill_point = ball_center + tangent * ball_radius
    gate_upstream_center = gate_center - tangent * (gate_thickness / 2.0)
    tangency_error = abs(float((gate_upstream_center - ball_downhill_point).dot(tangent)))
    if tangency_error > 1e-6:
        raise AssertionError(f"P6 gate is not tangent to the ball along the ramp: error={tangency_error}")
    if abs(float(ball_center.y - gate_y)) > gate_width / 2.0:
        raise AssertionError("P6 stop gate does not laterally cover the ball contact point")

    return {
        "variant": {**VARIANTS[variant], "index": variant},
        "camera": camera,
        "physical_parameters": {
            "ramp_angle_degrees": ramp_angle_deg,
            "ramp_length_m": ramp_length,
            "ramp_width_m": ramp_width,
            "ball_center_m": [round(float(value), 6) for value in ball_center],
            "ball_radius_m": ball_radius,
            "ball_ramp_contact_point_m": [round(float(value), 6) for value in contact_point],
            "gate_center_m": [round(float(value), 6) for value in gate_center],
            "gate_thickness_m": gate_thickness,
            "gate_width_m": gate_width,
            "gate_lateral_center_m": gate_y,
            "gate_ball_tangency_error_m": round(tangency_error, 9),
            "level_runout_length_m": runout_length,
        },
        "hard_constraints": {
            "exactly_one_ball": True,
            "visibly_asymmetric_non_text_pattern": True,
            "pattern_is_offset_band_plus_single_patch_not_face": True,
            "ball_tangent_to_incline": True,
            "thin_gate_tangent_and_holding_ball": True,
            "gate_is_opaque_high_contrast_solid": True,
            "ball_not_yet_rolling": True,
            "complete_incline_visible": True,
            "complete_level_runout_visible": True,
        },
    }


BUILDERS = {"P4": _p4_scene, "P6": _p6_scene}


def _validate_scene(task_id: str, scene_info: dict[str, Any]) -> None:
    if any(obj.type == "FONT" for obj in bpy.data.objects):
        raise AssertionError("text objects are forbidden in first frames")
    if task_id == "P4":
        balls = [obj for obj in bpy.data.objects if obj.name == "P4_single_orange_rubber_ball"]
        if len(balls) != 1:
            raise AssertionError(f"P4 must contain exactly one task ball, found {len(balls)}")
        p = scene_info["physical_parameters"]
        ball_bottom = p["ball_center_m"][2] - p["ball_radius_m"]
        if ball_bottom <= p["impact_plate_top_z_m"]:
            raise AssertionError("P4 ball is not held above the impact plate")
        if p["magnet_ball_contact_error_m"] > 1e-6:
            raise AssertionError("P4 electromagnet is not in exact contact with the held ball")
    elif task_id == "P6":
        balls = [obj for obj in bpy.data.objects if obj.name == "P6_single_asymmetrically_patterned_ball"]
        if len(balls) != 1:
            raise AssertionError(f"P6 must contain exactly one task ball, found {len(balls)}")
        patches = [obj for obj in bpy.data.objects if obj.name == "P6_single_offset_accent_patch"]
        if len(patches) != 1:
            raise AssertionError(f"P6 must contain one offset accent patch, found {len(patches)}")
        ball_material = balls[0].data.materials[0]
        if ball_material.node_tree.nodes.get("P6_smooth_asymmetric_band_mask") is None:
            raise AssertionError("P6 ball is missing its smooth asymmetric band shader")
        if scene_info["physical_parameters"]["gate_ball_tangency_error_m"] > 1e-6:
            raise AssertionError("P6 gate tangency metadata exceeds tolerance")
    if scene_info["camera"] is not bpy.context.scene.camera:
        raise AssertionError("scene camera is not the reviewed locked camera")


def _render_task(
    task_id: str,
    variant: int,
    delivery_root: Path,
    *,
    engine: str,
    samples: int,
    resolution_scale: float,
    device: str,
) -> dict[str, Any]:
    if task_id == "P6" and engine == "cycles" and resolution_scale >= 0.999 and samples < 128:
        raise ValueError("formal P6 Cycles renders require at least 128 samples")
    started = time.time()
    mats = _configure_render(engine, samples, resolution_scale, device)
    if engine == "cycles" and task_id == "P6":
        # Formal P6 uses conservative adaptive sampling and OIDN's normal /
        # albedo guidance so flat regions clean up without erasing the two
        # contact edges or asymmetric paint patches.
        cycles = bpy.context.scene.cycles
        cycles.use_adaptive_sampling = True
        cycles.adaptive_threshold = 0.005
        cycles.use_denoising = True
        if hasattr(cycles, "denoiser"):
            cycles.denoiser = "OPENIMAGEDENOISE"
        if hasattr(cycles, "denoising_input_passes"):
            cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
    scene_info = BUILDERS[task_id](mats, variant)
    finish_default_camera()
    bpy.context.view_layer.update()
    _validate_scene(task_id, scene_info)

    output_dir = delivery_root / task_id / "high_standard_v1" / "simulation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"sim_{variant:02d}.png"
    bpy.context.scene.render.filepath = str(output.resolve())
    bpy.ops.render.render(write_still=True)

    camera = scene_info["camera"]
    record = {
        "task_id": task_id,
        "candidate": variant,
        "variant": scene_info["variant"],
        "method": "procedural_simulation_source",
        "version": "high_standard_v1",
        "file": str(output.resolve()),
        "width": bpy.context.scene.render.resolution_x,
        "height": bpy.context.scene.render.resolution_y,
        "sha256": _sha256(output),
        "renderer": f"Blender Python {bpy.app.version_string} / {engine}",
        "render_device": str(bpy.context.scene.cycles.device) if engine == "cycles" else "N/A",
        "samples": samples,
        "denoising_enabled": bool(bpy.context.scene.cycles.use_denoising) if engine == "cycles" else False,
        "adaptive_threshold": (
            round(float(bpy.context.scene.cycles.adaptive_threshold), 6) if engine == "cycles" else None
        ),
        "denoiser": (
            str(getattr(bpy.context.scene.cycles, "denoiser", "N/A")) if engine == "cycles" else "N/A"
        ),
        "denoising_input_passes": (
            str(getattr(bpy.context.scene.cycles, "denoising_input_passes", "N/A"))
            if engine == "cycles"
            else "N/A"
        ),
        "elapsed_seconds": round(time.time() - started, 3),
        "initial_state_only": True,
        "camera": {
            "type": camera.data.type,
            "lens_mm": round(float(camera.data.lens), 3),
            "location": [round(float(value), 6) for value in camera.location],
            "rotation_euler_degrees": [round(math.degrees(float(value)), 6) for value in camera.rotation_euler],
            "dof_enabled": bool(camera.data.dof.use_dof),
        },
        "physical_parameters": scene_info["physical_parameters"],
        "hard_constraints": {
            **scene_info["hard_constraints"],
            "text_arrows_formulas_trajectories_absent": True,
            "motion_blur_absent": True,
        },
        "external_assets": [],
        "code": str(SCRIPT_PATH),
        "code_sha256": _sha256(SCRIPT_PATH),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / f"sim_{variant:02d}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delivery-root", type=Path, default=PROJECT_ROOT / "sop_delivery")
    parser.add_argument("--task", action="append", choices=sorted(BUILDERS))
    parser.add_argument("--variant", action="append", type=int, choices=sorted(VARIANTS))
    parser.add_argument("--engine", choices=("cycles", "eevee", "workbench"), default="cycles")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--resolution-scale", type=float, default=1.0)
    args = parser.parse_args()

    task_ids = args.task or sorted(BUILDERS)
    variants = args.variant or sorted(VARIANTS)
    records = []
    total = len(task_ids) * len(variants)
    current = 0
    for task_id in task_ids:
        for variant in variants:
            current += 1
            print(
                f"[{current}/{total}] rendering {task_id} high-standard simulation candidate {variant}",
                flush=True,
            )
            record = _render_task(
                task_id,
                variant,
                args.delivery_root,
                engine=args.engine,
                samples=args.samples,
                resolution_scale=args.resolution_scale,
                device=args.device,
            )
            records.append(record)
            print(
                json.dumps(
                    {
                        "task_id": task_id,
                        "candidate": variant,
                        "file": record["file"],
                        "sha256": record["sha256"],
                        "elapsed_seconds": record["elapsed_seconds"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    # All visual variants of one task must encode exactly the same physical
    # initial state.  Fail the batch if a material/camera edit changed it.
    for task_id in task_ids:
        task_records = [record for record in records if record["task_id"] == task_id]
        physics_payloads = {
            json.dumps(record["physical_parameters"], sort_keys=True, separators=(",", ":"))
            for record in task_records
        }
        if len(physics_payloads) != 1:
            raise AssertionError(f"{task_id} visual variants do not share one physical initial state")
        task_manifest = {
            "task_id": task_id,
            "method": "procedural_simulation_source",
            "version": "high_standard_v1",
            "requested": len(task_records),
            "completed": len(task_records),
            "shared_physical_initial_state": True,
            "records": task_records,
            "code": str(SCRIPT_PATH),
            "code_sha256": _sha256(SCRIPT_PATH),
        }
        task_output_dir = args.delivery_root / task_id / "high_standard_v1" / "simulation"
        (task_output_dir / "manifest.json").write_text(
            json.dumps(task_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    manifest = {
        "method": "procedural_simulation_source",
        "version": "high_standard_v1",
        "requested": total,
        "completed": len(records),
        "records": records,
        "code": str(SCRIPT_PATH),
        "code_sha256": _sha256(SCRIPT_PATH),
    }
    manifest_path = args.delivery_root / "high_standard_p4_p6_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
