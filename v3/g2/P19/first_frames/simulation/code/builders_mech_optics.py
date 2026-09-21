#!/usr/bin/env python3
"""Stable procedural first-frame builders for Physical-Bench P1--P19.

The scenes intentionally depict only the observable initial state.  Geometry
is static, legible and inexpensive to render; there are no captions, arrows,
trajectories or outcome cues.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import bpy
from mathutils import Vector

from simulation.common import (
    add_beaker,
    add_box,
    add_camera,
    add_cylinder,
    add_glass_tank,
    add_lab_table,
    add_mesh,
    add_pendulum,
    add_sphere,
    add_torus,
    add_tube,
    cylinder_between,
    material,
    point_light,
    ramp_point,
)


Mats = dict[str, Any]
Builder = Callable[[Mats], None]


def _side_camera(*, x: float = 0.0, z: float = 2.6, distance: float = 13.0, lens: float = 52.0) -> None:
    add_camera((x, -distance, z), (x, 0.0, z), lens=lens)


def _dim_shell_lights(factor: float) -> None:
    """Retain gentle fill while allowing one experimental light to dominate."""
    for name in ("key", "fill"):
        obj = bpy.data.objects.get(name)
        if obj is not None and getattr(obj, "data", None) is not None:
            obj.data.energy *= factor


def _launcher(
    mats: Mats,
    *,
    name: str,
    origin: tuple[float, float, float],
    angle: float,
    length: float = 0.92,
    ball_radius: float = 0.18,
) -> Vector:
    """Compact barrel, rigid base and a ball held at the muzzle."""
    p0 = Vector(origin)
    direction = Vector((math.cos(angle), 0.0, math.sin(angle)))
    p1 = p0 + direction * length
    add_box(f"{name}_base", (p0.x - 0.18, p0.y, p0.z - 0.24), (0.75, 0.66, 0.18), mats["black"], bevel=0.055)
    add_cylinder(f"{name}_pivot", tuple(p0), 0.18, 0.76, mats["steel"], rotation=(math.pi / 2, 0.0, 0.0))
    cylinder_between(f"{name}_barrel", tuple(p0), tuple(p1), 0.105, mats["black"])
    ball = p1 + direction * (ball_radius * 0.92)
    add_sphere(f"{name}_ball", tuple(ball), ball_radius, mats["red"])
    return ball


def _ramp(mats: Mats, *, name: str, center: tuple[float, float, float], angle: float, size=(7.0, 1.8, 0.18)) -> None:
    add_box(name, center, size, mats["wood"], rotation=(0.0, angle, 0.0), bevel=0.055)
    # Two restrained supports make the initial state look load bearing.
    add_box(f"{name}_low_support", (center[0] + 2.72, center[1], 0.48), (0.24, size[1] * 0.75, 0.9), mats["steel"], bevel=0.03)
    add_box(f"{name}_high_support", (center[0] - 2.72, center[1], 1.28), (0.24, size[1] * 0.75, 2.5), mats["steel"], bevel=0.03)


def scene_p1(mats: Mats) -> None:
    """Dense sphere held above a long, plain free-fall corridor."""
    plain = material("P1_plain_backdrop", (0.55, 0.59, 0.61), roughness=0.82)
    add_box("P1_backdrop", (0.0, 0.8, 2.8), (7.0, 0.12, 5.7), plain, bevel=0.0)
    add_sphere("P1_dense_ball", (0.0, -0.05, 4.75), 0.28, mats["steel"])
    add_cylinder("P1_electromagnet", (0.0, -0.05, 5.18), 0.17, 0.55, mats["black"])
    cylinder_between("P1_release_arm", (0.0, -0.05, 5.42), (1.25, -0.05, 5.42), 0.065, mats["steel"])
    add_box("P1_catch_tray", (0.0, -0.05, 0.34), (2.1, 1.15, 0.16), mats["rubber"], bevel=0.05)
    # The release head and catch tray are both measurement-critical.  A lower,
    # wider framing keeps them comfortably inside the image instead of grazing
    # the top and bottom crop.
    _side_camera(z=2.65, distance=14.5, lens=48.0)


def scene_p2(mats: Mats) -> None:
    """One angled projectile launcher before release."""
    top = add_lab_table(mats, z=0.72, top=(11.4, 2.9, 0.18))
    add_box("P2_level_landing_lane", (0.65, 0.0, top + 0.055), (9.7, 1.15, 0.11), mats["white"], bevel=0.02)
    # Move the rig inward so the base, pivot, barrel and ball all have visible
    # breathing room on the launch side.
    _launcher(mats, name="P2_launcher", origin=(-3.75, 0.0, top + 0.40), angle=math.radians(45), length=0.92)
    _side_camera(x=0.15, z=2.45, distance=14.8, lens=47.0)


def scene_p3(mats: Mats) -> None:
    """Three identical launch lanes at 30, 45 and 60 degrees in one view."""
    top = add_lab_table(mats, z=0.68, top=(12.2, 4.8, 0.18))
    lane_mat = material("P3_lane_surface", (0.52, 0.55, 0.56), roughness=0.7)
    for index, (y, degrees) in enumerate(((-1.35, 30), (0.0, 45), (1.35, 60)), start=1):
        add_box(f"P3_lane_{index}", (0.45, y, top + 0.055), (10.2, 0.88, 0.11), lane_mat, bevel=0.02)
        _launcher(
            mats,
            name=f"P3_launcher_{index}",
            origin=(-4.35, y, top + 0.39),
            angle=math.radians(degrees),
            length=0.78,
            ball_radius=0.16,
        )
    add_camera((7.0, -14.0, 6.8), (0.0, 0.0, 2.0), lens=53.0)


def scene_p4(mats: Mats) -> None:
    """Rubber ball above a hard plate, before the first of many bounces."""
    plate = material("P4_hard_stone", (0.38, 0.41, 0.43), roughness=0.65)
    add_box("P4_impact_plate", (0.0, 0.0, 0.42), (5.7, 2.2, 0.28), plate, bevel=0.035)
    add_sphere("P4_rubber_ball", (0.0, -0.05, 4.65), 0.34, mats["orange"])
    add_cylinder("P4_electromagnet", (0.0, -0.05, 5.15), 0.18, 0.58, mats["black"])
    cylinder_between("P4_release_arm", (0.0, -0.05, 5.38), (1.5, -0.05, 5.38), 0.065, mats["steel"])
    add_box("P4_release_post", (1.52, 0.0, 2.85), (0.14, 0.55, 5.15), mats["steel"], bevel=0.025)
    _side_camera(z=2.9, distance=12.6, lens=56.0)


def scene_p5(mats: Mats) -> None:
    """Two identical spheres aligned for a head-on elastic collision."""
    top = add_lab_table(mats, z=0.72, top=(11.2, 2.8, 0.18))
    add_box("P5_track_bed", (0.0, 0.0, top + 0.10), (9.6, 0.74, 0.14), mats["aluminum"], bevel=0.035)
    for y in (-0.43, 0.43):
        add_box(f"P5_track_rail_{y}", (0.0, y, top + 0.24), (9.6, 0.075, 0.25), mats["steel"], bevel=0.018)
    ball_z = top + 0.42
    add_sphere("P5_incoming_equal_ball", (-3.0, 0.0, ball_z), 0.27, mats["steel"])
    add_sphere("P5_stationary_equal_ball", (0.0, 0.0, ball_z), 0.27, mats["steel"])
    add_cylinder("P5_spring_plunger", (-3.64, 0.0, ball_z), 0.19, 0.75, mats["black"], rotation=(0.0, math.pi / 2, 0.0))
    add_box("P5_plunger_base", (-4.15, 0.0, top + 0.30), (0.45, 1.0, 0.58), mats["black"], bevel=0.045)
    _side_camera(z=2.15, distance=13.0, lens=54.0)


def scene_p6(mats: Mats) -> None:
    """Asymmetrically marked sphere held on a rough incline."""
    angle = math.radians(18)
    center = (0.0, 0.0, 2.0)
    _ramp(mats, name="P6_rough_ramp", center=center, angle=angle, size=(7.4, 1.55, 0.18))
    ball = ramp_point(center, angle, -2.4, 0.0, 0.49)
    add_sphere("P6_patterned_sphere", tuple(ball), 0.40, mats["blue"])
    # Two small raised color patches give optical-flow features without text.
    add_sphere("P6_front_patch", (ball.x, ball.y - 0.377, ball.z + 0.02), 0.105, mats["yellow"], segments=32)
    add_sphere("P6_offset_patch", (ball.x + 0.19, ball.y - 0.32, ball.z + 0.19), 0.075, mats["red"], segments=32)
    # A full-depth vertical blade sits on the downhill side of the sphere.  Its
    # upstream face is tangent to the ball, so the support relation is legible.
    gate_base = ramp_point(center, angle, -1.78, 0.0, 0.10)
    add_box(
        "P6_retractable_gate",
        (gate_base.x, 0.0, gate_base.z + 0.34),
        (0.14, 1.48, 0.68),
        mats["steel"],
        bevel=0.025,
    )
    add_box(
        "P6_gate_dark_face",
        (gate_base.x - 0.075, -0.01, gate_base.z + 0.34),
        (0.025, 1.36, 0.56),
        mats["black"],
        bevel=0.01,
    )
    add_box("P6_level_runout", (3.8, 0.0, 0.37), (2.0, 1.55, 0.14), mats["wood"], bevel=0.045)
    _side_camera(z=2.35, distance=12.8, lens=52.0)


def scene_p7(mats: Mats) -> None:
    """Equal-diameter solid sphere and thin ring behind one start gate."""
    angle = math.radians(17)
    center = (0.0, 0.0, 2.15)
    _ramp(mats, name="P7_common_ramp", center=center, angle=angle, size=(7.4, 2.6, 0.18))
    radius = 0.39
    sphere = ramp_point(center, angle, -2.3, -0.63, 0.49)
    ring = ramp_point(center, angle, -2.3, 0.63, 0.49)
    add_sphere("P7_solid_sphere", tuple(sphere), radius, mats["steel"])
    add_torus(
        "P7_thin_ring",
        tuple(ring),
        major_radius=0.305,
        minor_radius=0.085,
        mat=mats["brass"],
        rotation=(math.pi / 2, 0.0, 0.0),
    )
    gate = ramp_point(center, angle, -1.94, 0.0, 0.46)
    cylinder_between("P7_shared_start_gate", (gate.x, -1.2, gate.z), (gate.x, 1.2, gate.z), 0.052, mats["black"])
    add_box("P7_level_runout", (3.75, 0.0, 0.43), (2.1, 2.6, 0.16), mats["wood"], bevel=0.045)
    add_camera((4.0, -13.4, 5.0), (0.0, 0.0, 2.0), lens=55.0)


def scene_p8(mats: Mats) -> None:
    """Three equal pendulums held at visibly different amplitudes."""
    add_box("P8_rigid_top_beam", (0.0, 0.0, 5.25), (9.2, 0.42, 0.34), mats["steel"], bevel=0.05)
    for index, (x, degrees) in enumerate(((-3.0, 5), (0.0, 15), (3.0, 30)), start=1):
        bob = add_pendulum(
            mats,
            pivot=(x, 0.0, 5.05),
            length=2.65,
            angle=math.radians(degrees),
            plane_y=0.0,
            bob_radius=0.25,
            name=f"P8_pendulum_{index}",
        )
        add_cylinder(
            f"P8_release_catch_{index}",
            (bob.x, -0.27, bob.z),
            0.09,
            0.35,
            mats["black"],
            rotation=(math.pi / 2, 0.0, 0.0),
        )
    # Includes the complete support beam, all pivots, strings and the displaced
    # 30-degree bob with a generous margin.
    _side_camera(z=3.10, distance=15.0, lens=50.0)


def scene_p9(mats: Mats) -> None:
    """Two identical bobs on different string lengths at one small angle."""
    add_box("P9_rigid_top_beam", (0.0, 0.0, 5.35), (8.0, 0.45, 0.34), mats["steel"], bevel=0.05)
    for index, (x, length) in enumerate(((-2.0, 2.15), (2.0, 3.35)), start=1):
        bob = add_pendulum(
            mats,
            pivot=(x, 0.0, 5.15),
            length=length,
            angle=math.radians(12),
            plane_y=0.0,
            bob_radius=0.27,
            name=f"P9_pendulum_{index}",
        )
        add_cylinder(
            f"P9_release_catch_{index}",
            (bob.x, -0.27, bob.z),
            0.09,
            0.35,
            mats["black"],
            rotation=(math.pi / 2, 0.0, 0.0),
        )
    _side_camera(z=3.1, distance=12.8, lens=55.0)


def scene_p10(mats: Mats) -> None:
    """Block at rest against a spring plunger at the foot of a rough ramp."""
    angle = math.radians(22)
    center = (0.0, 0.0, 2.0)
    _ramp(mats, name="P10_friction_ramp", center=center, angle=angle, size=(8.2, 1.75, 0.20))
    pos = ramp_point(center, angle, 2.45, 0.0, 0.34)
    add_box("P10_sliding_block", tuple(pos), (0.78, 0.96, 0.48), mats["wood_dark"], rotation=(0.0, angle, 0.0), bevel=0.045)
    plunger_base = ramp_point(center, angle, 3.02, 0.0, 0.27)
    plunger_tip = ramp_point(center, angle, 2.78, 0.0, 0.27)
    cylinder_between("P10_spring_plunger", tuple(plunger_base), tuple(plunger_tip), 0.11, mats["steel"])
    add_box("P10_plunger_housing", tuple(ramp_point(center, angle, 3.30, 0.0, 0.30)), (0.55, 1.28, 0.55), mats["black"], rotation=(0.0, angle, 0.0), bevel=0.05)
    _side_camera(z=2.5, distance=13.0, lens=52.0)


def scene_p11(mats: Mats) -> None:
    """Straight straw crossing a flat air-water interface."""
    top = add_lab_table(mats, z=0.72, top=(8.2, 3.0, 0.18))
    tank = add_glass_tank(
        mats,
        center=(0.0, 0.0, top + 1.45),
        size=(4.4, 1.8, 2.65),
        bottom_z=top,
        fill=0.58,
        name="P11_tank",
    )
    straw_mat = material("P11_matte_straw", (0.9, 0.12, 0.025), roughness=0.55)
    cylinder_between(
        "P11_straight_straw",
        (-1.55, -0.22, tank["bottom"] + 0.32),
        (1.55, -0.22, tank["top"] + 0.85),
        0.075,
        straw_mat,
    )
    _side_camera(z=2.35, distance=10.8, lens=61.0)


def scene_p12(mats: Mats) -> None:
    """Multiple refraction angles and a near-critical internal ray in one tank."""
    top = add_lab_table(mats, z=0.68, top=(12.6, 3.6, 0.18))
    # Use two physically separate bays: three air-to-water measurements on the
    # left and an uncluttered total-internal-reflection measurement on the right.
    refract_tank = add_glass_tank(
        mats,
        center=(-2.10, 0.0, top + 1.55),
        size=(6.40, 1.85, 3.10),
        bottom_z=top,
        fill=0.55,
        name="P12_refraction_bay",
    )
    tir_tank = add_glass_tank(
        mats,
        center=(3.40, 0.0, top + 1.55),
        size=(4.00, 1.85, 3.10),
        bottom_z=top,
        fill=0.55,
        name="P12_TIR_bay",
    )
    waterline = refract_tank["liquid_top"]
    laser = mats["laser"]
    source_z = refract_tank["top"] + 0.38
    below_z = refract_tank["bottom"] + 0.23
    # Incidence angles are approximately 11, 37 and 46 degrees from the normal.
    # The interface hits are two metres apart, preventing any visual merging.
    air_specs = (
        ((-5.00, -0.33, source_z), (-4.65, -0.33, waterline), (-4.48, -0.33, below_z)),
        ((-4.00, -0.33, source_z), (-2.65, -0.33, waterline), (-2.20, -0.33, below_z)),
        ((-2.40, -0.33, source_z), (-0.55, -0.33, waterline), (0.05, -0.33, below_z)),
    )
    for index, (source, hit, below) in enumerate(air_specs, start=1):
        add_box(f"P12_raybox_{index}", source, (0.34, 0.48, 0.25), mats["black"], bevel=0.035)
        cylinder_between(f"P12_incident_ray_{index}", source, hit, 0.030, laser)
        cylinder_between(f"P12_refracted_ray_{index}", hit, below, 0.030, laser)
        add_sphere(f"P12_interface_spot_{index}", hit, 0.060, laser, segments=24)
    # The right bay has a >critical-angle incident branch and a symmetric
    # reflected branch, both fully contained in that bay.
    critical_source = (4.95, -0.33, tir_tank["bottom"] + 0.23)
    critical_hit = (3.25, -0.33, tir_tank["liquid_top"])
    reflected_end = (1.55, -0.33, tir_tank["bottom"] + 0.23)
    add_box("P12_underwater_laser", critical_source, (0.52, 0.44, 0.28), mats["black"], rotation=(0.0, -0.86, 0.0), bevel=0.035)
    cylinder_between("P12_near_critical_incident", critical_source, critical_hit, 0.034, laser)
    cylinder_between("P12_internal_reflection", critical_hit, reflected_end, 0.034, laser)
    add_sphere("P12_critical_surface_spot", critical_hit, 0.065, laser, segments=24)
    _side_camera(z=2.65, distance=15.0, lens=48.0)


def scene_p13(mats: Mats) -> None:
    """Horizontal plane mirror with a vertical incidence/reflection section.

    The ray and the normal are both in the XZ plane, while the mirror surface
    is horizontal.  A straight front camera looks along Y, so that section is
    parallel to the camera plane and all three measurable lines stay visible.
    """
    floor = material("P13_matte_dark_floor", (0.018, 0.022, 0.028), roughness=0.72)
    add_box("P13_dark_horizontal_floor", (0.0, 0.6, 0.08), (11.2, 5.2, 0.16), floor, bevel=0.0)
    mirror = material("P13_horizontal_plane_mirror", (0.30, 0.34, 0.39), roughness=0.06, metallic=0.94)
    mirror_center = (0.0, 0.45, 0.58)
    add_box("P13_flat_mirror_surface", mirror_center, (7.6, 3.9, 0.14), mirror, bevel=0.012)
    add_box("P13_flat_mirror_underlay", (0.0, 0.45, 0.47), (7.82, 4.1, 0.10), mats["black"], bevel=0.025)
    for y in (-1.35, 2.25):
        add_box(f"P13_mirror_support_{y:+.2f}", (0.0, y, 0.32), (0.34, 0.34, 0.30), mats["black"], bevel=0.02)

    ray_y = -0.55
    mirror_hit = (-0.15, ray_y, 0.67)
    laser_source = (-3.75, ray_y, 4.95)
    add_box("P13_overhead_laser_source", laser_source, (0.34, 0.34, 0.62), mats["black"], rotation=(0.0, -0.64, 0.0), bevel=0.04)
    cylinder_between("P13_downward_incident_laser", laser_source, mirror_hit, 0.028, mats["laser"])
    add_sphere("P13_mirror_spot", mirror_hit, 0.060, mats["laser"], segments=24)

    normal_mat = material(
        "P13_vertical_mirror_normal",
        (0.48, 0.62, 0.70),
        roughness=0.32,
        emission=(0.10, 0.20, 0.28),
        emission_strength=1.3,
    )
    normal_end = (mirror_hit[0], ray_y, 4.42)
    cylinder_between("P13_vertical_mirror_normal", mirror_hit, normal_end, 0.018, normal_mat)
    add_camera((0.0, -14.0, 2.55), (0.0, 0.0, 2.55), lens=54.0)


def scene_p14(mats: Mats) -> None:
    """One point light casting four distinct radial shadows."""
    _dim_shell_lights(0.025)
    world_bg = bpy.context.scene.world.node_tree.nodes.get("Background")
    if world_bg is not None:
        world_bg.inputs["Strength"].default_value = 0.060
    floor = material("P14_shadow_floor", (0.36, 0.37, 0.38), roughness=0.94)
    add_box("P14_receiving_plane", (0.0, 0.0, 0.36), (9.6, 7.6, 0.16), floor, bevel=0.025)
    led = material("P14_led_emitter", (1.0, 0.77, 0.42), roughness=0.1, emission=(1.0, 0.48, 0.12), emission_strength=9.0)
    add_sphere("P14_visible_point_lamp", (0.0, 0.0, 1.95), 0.105, led, segments=32)
    add_cylinder("P14_lamp_stand", (0.0, 0.0, 1.14), 0.050, 1.58, mats["black"])
    # Put the actual point source just below the visible emissive bead.  A
    # point light placed inside an opaque sphere cannot illuminate the rods.
    point_light("P14_only_point_light", (0.16, 0.0, 1.95), 2600.0, color=(1.0, 0.74, 0.46), radius=0.015)
    # Shorter rods and moderate radial offsets keep every shadow endpoint well
    # inside the receiving plane while preserving four distinct directions.
    rods = ((-1.45, -0.75, 0.82), (-0.65, 1.35, 0.72), (1.15, 1.15, 0.90), (1.55, -0.80, 0.78))
    colors = (mats["red"], mats["blue"], mats["yellow"], mats["green"])
    for index, ((x, y, height), rod_mat) in enumerate(zip(rods, colors), start=1):
        add_cylinder(f"P14_upright_rod_{index}", (x, y, 0.44 + height / 2), 0.105, height, rod_mat)
    add_camera((0.0, -9.5, 11.5), (0.0, 0.0, 0.55), lens=52.0)


def scene_p15(mats: Mats) -> None:
    """One glossy sphere with a single highlight and one cast shadow."""
    _dim_shell_lights(0.015)
    world_bg = bpy.context.scene.world.node_tree.nodes.get("Background")
    if world_bg is not None:
        world_bg.inputs["Strength"].default_value = 0.050
    floor = material("P15_matte_floor", (0.36, 0.37, 0.38), roughness=0.96)
    glossy = material("P15_glossy_ball", (0.055, 0.085, 0.13), roughness=0.115, metallic=0.06)
    bulb = material("P15_visible_bulb", (1.0, 0.72, 0.28), roughness=0.06, emission=(1.0, 0.36, 0.05), emission_strength=8.0)
    add_box("P15_receiving_floor", (0.0, 0.0, 0.38), (10.5, 7.5, 0.18), floor, bevel=0.025)
    add_sphere("P15_glossy_sphere", (0.65, 0.20, 1.34), 0.88, glossy)
    lamp_pos = (-3.0, -1.55, 4.15)
    add_sphere("P15_visible_lamp", lamp_pos, 0.17, bulb, segments=32)
    cylinder_between("P15_lamp_arm", (-4.05, -1.55, 0.50), (-4.05, -1.55, 4.15), 0.075, mats["black"])
    cylinder_between("P15_lamp_head_arm", (-4.05, -1.55, 4.15), lamp_pos, 0.075, mats["black"])
    point_light("P15_single_experimental_light", (-3.0, -1.55, 3.92), 2300.0, color=(1.0, 0.68, 0.38), radius=0.045)
    add_camera((7.6, -13.2, 6.3), (-0.15, 0.0, 1.45), lens=50.0)


def scene_p16(mats: Mats) -> None:
    """Rigid ladder with four collinear markers before release."""
    add_box("P16_floor", (0.0, 0.0, 0.16), (10.5, 3.8, 0.28), mats["white"], bevel=0.02)
    add_box("P16_vertical_wall", (3.0, 0.72, 3.0), (0.18, 3.0, 6.0), mats["white"], bevel=0.02)
    bottom = Vector((-2.55, 0.0, 0.38))
    top = Vector((2.83, 0.0, 5.55))
    axis = (top - bottom).normalized()
    across = Vector((-axis.z, 0.0, axis.x))
    rail_a0, rail_a1 = bottom + across * 0.34, top + across * 0.34
    rail_b0, rail_b1 = bottom - across * 0.34, top - across * 0.34
    cylinder_between("P16_ladder_rail_a", tuple(rail_a0), tuple(rail_a1), 0.095, mats["wood_dark"])
    cylinder_between("P16_ladder_rail_b", tuple(rail_b0), tuple(rail_b1), 0.095, mats["wood_dark"])
    for index, t in enumerate((0.08, 0.24, 0.40, 0.56, 0.72, 0.88), start=1):
        center = bottom.lerp(top, t)
        cylinder_between(
            f"P16_ladder_rung_{index}",
            tuple(center - across * 0.34),
            tuple(center + across * 0.34),
            0.055,
            mats["wood"],
        )
    marker_mats = (mats["red"], mats["yellow"], mats["blue"], mats["green"])
    for index, (t, marker_mat) in enumerate(zip((0.17, 0.38, 0.62, 0.84), marker_mats), start=1):
        center = bottom.lerp(top, t)
        add_cylinder(
            f"P16_collinear_marker_{index}",
            (center.x, -0.115, center.z),
            0.13,
            0.055,
            marker_mat,
            rotation=(math.pi / 2, 0.0, 0.0),
            vertices=48,
        )
    add_box("P16_remote_release", (2.79, -0.10, 5.72), (0.34, 0.38, 0.28), mats["black"], bevel=0.035)
    # Wide enough for both ladder ends, all four markers, the wall/floor corner
    # and the entire empty floor region into which the rigid body can fall.
    _side_camera(z=2.85, distance=15.6, lens=48.0)


def scene_p17(mats: Mats) -> None:
    """Pebble held above the calm center of a wide ripple tray."""
    _dim_shell_lights(0.22)
    water = material("P17_dark_water", (0.018, 0.19, 0.27), roughness=0.24, transmission=0.08, alpha=0.92)
    rim = material("P17_matte_rim", (0.32, 0.35, 0.37), roughness=0.72, metallic=0.30)
    pebble = material("P17_pebble", (0.16, 0.13, 0.105), roughness=0.78)
    add_cylinder("P17_tray_base", (0.0, 0.0, 0.42), 4.25, 0.22, mats["black"], vertices=96)
    add_cylinder("P17_still_water", (0.0, 0.0, 0.57), 4.02, 0.10, water, vertices=96)
    add_torus("P17_tray_rim", (0.0, 0.0, 0.63), 4.12, 0.115, rim, rotation=(0.0, 0.0, 0.0))
    add_sphere("P17_center_pebble", (0.0, 0.0, 1.05), 0.20, pebble, segments=40)
    add_cylinder("P17_release_tip", (0.0, 0.0, 1.55), 0.055, 0.80, mats["black"])
    cylinder_between("P17_release_arm", (0.0, 0.0, 1.92), (4.70, 0.0, 1.92), 0.048, mats["steel"])
    add_box("P17_release_post", (4.76, 0.0, 1.05), (0.14, 0.42, 1.9), mats["steel"], bevel=0.025)
    # The original tight overhead view cropped the circular boundary.  This
    # wider true top view also places the support post outside the tray.
    add_camera((0.0, 0.0, 14.5), (0.0, 0.0, 0.45), lens=48.0)


def _rotate_xz(local_x: float, local_z: float, angle: float, center: tuple[float, float, float]) -> tuple[float, float]:
    """Match Blender's positive Y-axis rotation for an x-z cross-section."""
    x = center[0] + math.cos(angle) * local_x + math.sin(angle) * local_z
    z = center[2] - math.sin(angle) * local_x + math.cos(angle) * local_z
    return x, z


def _tilted_liquid_prism(
    name: str,
    *,
    center: tuple[float, float, float],
    width: float,
    depth: float,
    height: float,
    angle: float,
    waterline: float,
    mat: Any,
) -> None:
    """Fill the part of a rotated rectangle below a horizontal waterline."""
    polygon = [
        _rotate_xz(-width / 2, -height / 2, angle, center),
        _rotate_xz(width / 2, -height / 2, angle, center),
        _rotate_xz(width / 2, height / 2, angle, center),
        _rotate_xz(-width / 2, height / 2, angle, center),
    ]

    clipped: list[tuple[float, float]] = []
    for current, following in zip(polygon, polygon[1:] + polygon[:1]):
        current_inside = current[1] <= waterline
        next_inside = following[1] <= waterline
        if current_inside:
            clipped.append(current)
        if current_inside != next_inside:
            ratio = (waterline - current[1]) / (following[1] - current[1])
            clipped.append((current[0] + ratio * (following[0] - current[0]), waterline))

    y0, y1 = center[1] - depth / 2, center[1] + depth / 2
    vertices = [(x, y0, z) for x, z in clipped] + [(x, y1, z) for x, z in clipped]
    n = len(clipped)
    faces: list[tuple[int, ...]] = [tuple(range(n)), tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j, n + i))
    add_mesh(name, vertices, faces, mat)


def scene_p18(mats: Mats) -> None:
    """Tilted straight-sided tank with a horizontal free surface and plumb ball."""
    top = add_lab_table(mats, z=0.62, top=(11.0, 3.7, 0.18))
    angle = math.radians(20)
    center = (-1.15, 0.0, top + 1.65)
    width, depth, height = 4.35, 1.72, 2.55
    wall_t = 0.065
    add_box("P18_tank_front", (center[0], -depth / 2, center[2]), (width, wall_t, height), mats["glass"], rotation=(0.0, angle, 0.0), bevel=0.008)
    add_box("P18_tank_back", (center[0], depth / 2, center[2]), (width, wall_t, height), mats["glass"], rotation=(0.0, angle, 0.0), bevel=0.008)
    for side_name, local_x in (("left", -width / 2), ("right", width / 2)):
        x, z = _rotate_xz(local_x, 0.0, angle, center)
        add_box(f"P18_tank_{side_name}", (x, 0.0, z), (wall_t, depth, height), mats["glass"], rotation=(0.0, angle, 0.0), bevel=0.008)
    x, z = _rotate_xz(0.0, -height / 2, angle, center)
    add_box("P18_tank_bottom", (x, 0.0, z), (width, depth, wall_t), mats["glass"], rotation=(0.0, angle, 0.0), bevel=0.008)
    _tilted_liquid_prism(
        "P18_horizontal_water",
        center=center,
        width=width - 0.14,
        depth=depth - 0.14,
        height=height - 0.14,
        angle=angle,
        waterline=center[2] + 0.16,
        mat=mats["water"],
    )
    # Triangular-looking support pair, built from straight steel members.
    cylinder_between("P18_frame_left", (-3.25, 0.55, top), (-2.65, 0.55, top + 2.75), 0.07, mats["steel"])
    cylinder_between("P18_frame_right", (1.15, 0.55, top), (0.75, 0.55, top + 1.75), 0.07, mats["steel"])
    ball_x = 3.15
    add_sphere("P18_plumb_ball", (ball_x, -0.1, top + 3.45), 0.25, mats["red"])
    add_cylinder("P18_ball_electromagnet", (ball_x, -0.1, top + 3.84), 0.15, 0.48, mats["black"])
    cylinder_between("P18_release_arm", (ball_x, -0.1, top + 4.03), (4.28, -0.1, top + 4.03), 0.055, mats["steel"])
    add_box("P18_release_post", (4.30, 0.0, top + 2.02), (0.13, 0.48, 4.10), mats["steel"], bevel=0.02)
    add_cylinder("P18_ball_catch", (ball_x, -0.1, top + 0.10), 0.52, 0.16, mats["rubber"])
    _side_camera(z=2.65, distance=13.0, lens=53.0)


def scene_p19(mats: Mats) -> None:
    """Transparent U-tube with unequal arm bores and a continuous liquid.

    Separate curve sections are intentional: Blender's curve bevel is uniform
    along a spline, while this experiment needs visibly different vertical arm
    diameters.  The sections overlap at the two shoulders and therefore read as
    one continuous glass tube and one continuous liquid column in the render.
    """
    top = add_lab_table(mats, z=0.58, top=(8.7, 3.4, 0.18))
    glass = material("P19_U_tube_glass", (0.22, 0.46, 0.54), roughness=0.035, transmission=0.88, alpha=0.25)
    colored_water = material("P19_continuous_water", (0.015, 0.30, 0.55), roughness=0.055, transmission=0.34, alpha=0.74)
    water_edge = material("P19_liquid_surface", (0.005, 0.10, 0.20), roughness=0.18, metallic=0.08)

    left_x, right_x = -1.62, 1.62
    # Lift the bend above the support base so the complete U silhouette remains
    # visible instead of being hidden behind the base's front edge.
    bend_center_z, bend_radius = 2.65, 1.62
    arm_top = 4.85
    bend_points = []
    for index in range(49):
        theta = math.pi + math.pi * index / 48.0
        bend_points.append((bend_radius * math.cos(theta), 0.0, bend_center_z + bend_radius * math.sin(theta)))
    add_tube("P19_small_glass_bottom_connector", bend_points, 0.16, glass, resolution=3)
    add_tube("P19_glass_left_arm", [(left_x, 0.0, bend_center_z), (left_x, 0.0, arm_top)], 0.40, glass, resolution=3)
    add_tube("P19_glass_right_arm", [(right_x, 0.0, bend_center_z), (right_x, 0.0, arm_top)], 0.28, glass, resolution=3)

    # The initial disturbance is represented by unequal free-surface heights;
    # both branches remain connected through the filled bottom bend.
    left_level, right_level = 3.92, 3.02
    liquid_bend_points = []
    for index in range(49):
        theta = math.pi + math.pi * index / 48.0
        liquid_bend_points.append((bend_radius * math.cos(theta), -0.01, bend_center_z + bend_radius * math.sin(theta)))
    add_tube("P19_small_liquid_bottom_connector", liquid_bend_points, 0.105, colored_water, resolution=3)
    add_tube("P19_liquid_left_arm", [(left_x, -0.01, bend_center_z), (left_x, -0.01, left_level)], 0.29, colored_water, resolution=3)
    add_tube("P19_liquid_right_arm", [(right_x, -0.01, bend_center_z), (right_x, -0.01, right_level)], 0.17, colored_water, resolution=3)
    add_cylinder("P19_left_surface", (left_x, -0.01, left_level), 0.29, 0.024, water_edge, vertices=48)
    add_cylinder("P19_right_surface", (right_x, -0.01, right_level), 0.17, 0.024, water_edge, vertices=48)

    # Minimal support: a white base, two narrow clamps and one top rail.  It
    # keeps the U silhouette legible without adding unrelated apparatus.
    add_box("P19_support_base", (0.0, 0.0, top - 0.06), (6.8, 1.55, 0.12), mats["white"], bevel=0.025)
    add_box("P19_support_crossbar", (0.0, 0.42, arm_top + 0.25), (4.15, 0.12, 0.12), mats["steel"], bevel=0.02)
    for x in (left_x, right_x):
        add_box(f"P19_clamp_{x:+.2f}", (x, 0.0, bend_center_z + 0.12), (0.16, 0.82, 0.16), mats["steel"], bevel=0.02)
    add_camera((0.0, -13.5, 2.55), (0.0, 0.0, 2.55), lens=57.0)


BUILDERS: dict[str, tuple[Builder, bool]] = {
    "P1": (scene_p1, False),
    "P2": (scene_p2, False),
    "P3": (scene_p3, False),
    "P4": (scene_p4, False),
    "P5": (scene_p5, False),
    "P6": (scene_p6, False),
    "P7": (scene_p7, False),
    "P8": (scene_p8, False),
    "P9": (scene_p9, False),
    "P10": (scene_p10, False),
    "P11": (scene_p11, False),
    "P12": (scene_p12, True),
    "P13": (scene_p13, True),
    "P14": (scene_p14, True),
    "P15": (scene_p15, True),
    "P16": (scene_p16, False),
    "P17": (scene_p17, False),
    "P18": (scene_p18, False),
    "P19": (scene_p19, False),
}
