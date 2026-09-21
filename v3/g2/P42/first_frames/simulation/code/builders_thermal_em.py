#!/usr/bin/env python3
"""Procedural first-frame builders for Physical-Bench P20--P35.

The scenes intentionally depict only a clean observable initial state.  They
use inexpensive rigid geometry so the full batch is stable to render on a
headless Blender worker; later video generation remains responsible for the
motion or phase evolution.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from simulation.common import (
    add_bar_magnet,
    add_beaker,
    add_box,
    add_camera,
    add_candle,
    add_coil,
    add_compass,
    add_cone,
    add_cylinder,
    add_glass_tank,
    add_lab_table,
    add_sphere,
    add_torus,
    add_tube,
    cylinder_between,
    material,
)


def _front_camera(target=(0.0, 0.0, 2.1), *, distance: float = 11.5, height: float = 3.6, lens: float = 55.0) -> None:
    add_camera((0.0, -distance, height), target, lens=lens)


def _overhead_camera(target=(0.0, 0.0, 1.0), *, height: float = 10.5, lens: float = 54.0) -> None:
    add_camera((0.0, 0.0, height), target, lens=lens)


def _tray(mats, name: str, x: float, y: float, z: float, *, width: float = 2.4, depth: float = 1.65) -> None:
    add_box(f"{name}_base", (x, y, z), (width, depth, 0.10), mats["black"], bevel=0.06)
    rim_z = z + 0.12
    add_box(f"{name}_rim_front", (x, y - depth / 2, rim_z), (width, 0.08, 0.24), mats["steel"], bevel=0.025)
    add_box(f"{name}_rim_back", (x, y + depth / 2, rim_z), (width, 0.08, 0.24), mats["steel"], bevel=0.025)
    add_box(f"{name}_rim_left", (x - width / 2, y, rim_z), (0.08, depth, 0.24), mats["steel"], bevel=0.025)
    add_box(f"{name}_rim_right", (x + width / 2, y, rim_z), (0.08, depth, 0.24), mats["steel"], bevel=0.025)


def _electroscope(mats, name: str, x: float, y: float, base_z: float, *, spread: float = 0.035) -> None:
    add_cylinder(f"{name}_base", (x, y, base_z + 0.08), 0.34, 0.16, mats["black"])
    cylinder_between(f"{name}_stem", (x, y, base_z + 0.15), (x, y, base_z + 1.45), 0.035, mats["brass"])
    add_sphere(f"{name}_knob", (x, y, base_z + 1.58), 0.22, mats["brass"])
    leaf_z = base_z + 0.67
    add_box(
        f"{name}_leaf_left",
        (x - 0.08, y, leaf_z),
        (0.10, 0.025, 0.72),
        mats["brass"],
        rotation=(0.0, -spread, 0.0),
        bevel=0.008,
    )
    add_box(
        f"{name}_leaf_right",
        (x + 0.08, y, leaf_z),
        (0.10, 0.025, 0.72),
        mats["brass"],
        rotation=(0.0, spread, 0.0),
        bevel=0.008,
    )


def _wire_cage(mats, center=(0.0, 0.15, 2.0), size=(3.1, 2.1, 2.7)) -> None:
    cx, cy, cz = center
    sx, sy, sz = size
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = cz - sz / 2, cz + sz / 2
    for x in (x0, x1):
        for y in (y0, y1):
            cylinder_between(f"cage_vertical_{x:.2f}_{y:.2f}", (x, y, z0), (x, y, z1), 0.035, mats["steel"])
    for z in (z0, z1):
        for y in (y0, y1):
            cylinder_between(f"cage_x_{z:.2f}_{y:.2f}", (x0, y, z), (x1, y, z), 0.035, mats["steel"])
        for x in (x0, x1):
            cylinder_between(f"cage_y_{z:.2f}_{x:.2f}", (x, y0, z), (x, y1, z), 0.035, mats["steel"])
    for i in range(1, 6):
        x = x0 + sx * i / 6
        for y in (y0, y1):
            cylinder_between(f"cage_frontback_v_{i}_{y:.2f}", (x, y, z0), (x, y, z1), 0.012, mats["steel"])
    for i in range(1, 5):
        z = z0 + sz * i / 5
        for y in (y0, y1):
            cylinder_between(f"cage_frontback_h_{i}_{y:.2f}", (x0, y, z), (x1, y, z), 0.012, mats["steel"])
        for x in (x0, x1):
            cylinder_between(f"cage_side_h_{i}_{x:.2f}", (x, y0, z), (x, y1, z), 0.012, mats["steel"])


def _scatter_filings(mats, prefix: str, *, centers: tuple[float, ...], z: float, count_per_center: int = 28) -> None:
    for group, cx in enumerate(centers):
        for i in range(count_per_center):
            angle = (i * 2.399963 + group * 0.61) % (2 * math.pi)
            radial = 0.42 + 1.42 * ((i * 17 % count_per_center) / max(1, count_per_center - 1))
            x = cx + radial * math.cos(angle)
            y = radial * 0.62 * math.sin(angle)
            heading = (i * 1.73 + group) % math.pi
            dx, dy = 0.075 * math.cos(heading), 0.075 * math.sin(heading)
            cylinder_between(
                f"{prefix}_{group}_{i}",
                (x - dx, y - dy, z),
                (x + dx, y + dy, z),
                0.012,
                mats["steel"],
            )


def scene_p20(mats) -> None:
    top = add_lab_table(mats)
    tank = add_glass_tank(mats, center=(0.0, 0.0, 2.15), size=(5.7, 1.75, 2.5), bottom_z=top + 0.05, fill=0.68, name="buoyancy_tank")
    water_z = tank["liquid_top"]
    height = 0.72
    specimens = (
        ("wood", -1.55, 0.58, mats["wood"]),
        ("plastic", 0.0, 0.82, mats["green"]),
        ("ice", 1.55, 0.91, mats["ice"]),
    )
    for name, x, fraction, mat in specimens:
        center_z = water_z - fraction * height + height / 2
        add_box(f"floating_{name}", (x, -0.05, center_z), (0.90, 0.68, height), mat, bevel=0.08)
    _front_camera((0.0, 0.0, 2.15), height=3.25, lens=58)


def scene_p21(mats) -> None:
    top = add_lab_table(mats)
    vessel = add_beaker(mats, location=(0.0, 0.0, top + 0.02), radius=0.78, height=3.15, fill=0.58, name="melting_ice_cup")
    h = 0.72
    add_box("floating_ice", (0.0, -0.05, vessel["liquid_top"] + 0.08), (0.82, 0.64, h), mats["ice"], rotation=(0.02, 0.08, -0.04), bevel=0.10)
    _front_camera((0.0, 0.0, 2.35), height=3.35, lens=68)


def scene_p21b(mats) -> None:
    top = add_lab_table(mats)
    vessel = add_beaker(mats, location=(0.0, 0.0, top + 0.02), radius=0.82, height=3.25, fill=0.60, name="stone_ice_cup")
    ice_z = vessel["liquid_top"] + 0.10
    add_box("floating_ice_with_stone", (0.0, 0.0, ice_z), (0.94, 0.72, 0.82), mats["ice"], rotation=(0.01, 0.08, -0.03), bevel=0.12)
    add_sphere("embedded_dark_pebble", (0.05, -0.25, ice_z - 0.06), 0.22, mats["black"])
    _front_camera((0.0, 0.0, 2.35), height=3.35, lens=68)


def scene_p21c(mats) -> None:
    top = add_lab_table(mats)
    vessel = add_beaker(mats, location=(0.0, 0.0, top + 0.02), radius=0.82, height=3.25, fill=0.66, liquid=mats["oil"], name="phase_change_cup")
    add_box("sunken_frozen_block", (0.0, -0.03, vessel["bottom"] + 0.38), (0.88, 0.68, 0.68), mats["ice"], bevel=0.11)
    _front_camera((0.0, 0.0, 2.35), height=3.35, lens=68)


def scene_p22(mats) -> None:
    top = add_lab_table(mats, wood=False)
    puddle = material("thin_meltwater", (0.06, 0.24, 0.31), roughness=0.05, transmission=0.7, alpha=0.30)
    add_cylinder("initial_wet_meniscus", (0.0, 0.0, top + 0.025), 0.72, 0.025, puddle, vertices=72)
    add_box("single_melting_ice_block", (0.0, 0.0, top + 0.43), (1.22, 0.98, 0.82), mats["ice"], rotation=(0.015, 0.035, -0.04), bevel=0.14)
    add_camera((4.8, -8.8, 5.2), (0.0, 0.0, 1.15), lens=60)


def scene_p23(mats) -> None:
    top = add_lab_table(mats)
    add_beaker(mats, location=(0.0, 0.0, top + 0.02), radius=0.74, height=3.45, fill=0.67, name="freezing_cylinder")
    for z in (top + 0.58, top + 1.08, top + 1.58, top + 2.08, top + 2.58, top + 3.08):
        cylinder_between("unnumbered_tick", (-0.78, -0.76, z), (-0.62, -0.76, z), 0.012, mats["white"])
    _front_camera((0.0, 0.0, 2.45), height=3.55, lens=70)


def scene_p24(mats) -> None:
    top = add_lab_table(mats)
    add_candle(mats, location=(-1.25, 0.0, top), height=1.25, radius=0.23, name="gravity_candle")
    plume_mat = material("faint_thermal_plume", (0.55, 0.59, 0.62), roughness=0.9, transmission=0.25, alpha=0.16)
    for i, xoff in enumerate((-0.08, 0.0, 0.08)):
        pts = []
        for j in range(9):
            z = top + 1.82 + j * 0.22
            x = -1.25 + xoff + 0.025 * math.sin(j * 1.4 + i)
            pts.append((x, 0.04 + i * 0.025, z))
        add_tube(f"thermal_plume_{i}", pts, 0.035 - i * 0.006, plume_mat, resolution=2)
    add_beaker(mats, location=(1.25, 0.0, top + 0.02), radius=0.68, height=1.85, fill=0.64, name="still_water_glass")
    _front_camera((0.0, 0.0, 2.15), height=3.05, lens=60)


def scene_p25(mats) -> None:
    top = add_lab_table(mats)
    rod_z = top + 1.55
    add_box("rod_support", (2.85, 0.0, top + 0.78), (0.36, 0.62, 1.55), mats["steel"], bevel=0.04)
    colors = (
        (1.00, 0.18, 0.015),
        (0.90, 0.075, 0.012),
        (0.68, 0.035, 0.012),
        (0.45, 0.025, 0.015),
        (0.28, 0.025, 0.022),
        (0.18, 0.025, 0.025),
        (0.12, 0.03, 0.035),
        (0.09, 0.045, 0.05),
        (0.16, 0.18, 0.19),
        (0.28, 0.30, 0.31),
        (0.36, 0.39, 0.41),
        (0.40, 0.43, 0.45),
    )
    x0, segment = -3.0, 0.50
    for i, color in enumerate(colors):
        glow = max(0.0, 4.5 - i * 0.85)
        mat = material(
            f"iron_temperature_{i}",
            color,
            roughness=0.30,
            metallic=0.68,
            emission=color if glow > 0 else None,
            emission_strength=glow,
        )
        cylinder_between(
            f"heated_rod_segment_{i}",
            (x0 + i * segment, 0.0, rod_z),
            (x0 + (i + 1) * segment + 0.006, 0.0, rod_z),
            0.14,
            mat,
        )
    add_coil(mats, center=(-2.72, 0.0, rod_z), radius=0.34, length=0.62, turns=9, axis="x", name="induction_heater")
    _front_camera((0.0, 0.0, 2.05), height=3.15, lens=58)


def scene_p26(mats) -> None:
    top = add_lab_table(mats)
    tank = add_glass_tank(mats, center=(0.0, 0.0, 2.65), size=(2.25, 1.55, 3.65), bottom_z=top + 0.04, fill=0.91, name="deep_bubble_tank")
    bubble_glass = material(
        "refractive_air_bubble",
        (0.70, 0.88, 0.98),
        roughness=0.012,
        transmission=0.96,
        alpha=0.20,
    )
    bubble_rim = material(
        "bubble_specular_rim",
        (0.72, 0.91, 1.0),
        roughness=0.06,
        emission=(0.42, 0.72, 1.0),
        emission_strength=1.25,
    )
    bubble_center = (0.0, -0.50, tank["bottom"] + 0.54)
    add_sphere("single_refractive_bubble", bubble_center, 0.235, bubble_glass)
    # A very thin physical highlight ring makes the gas/liquid interface read
    # clearly through two glass walls without turning it into a solid ball.
    add_torus("single_bubble_bright_rim", bubble_center, 0.215, 0.014, bubble_rim)
    add_sphere("single_bubble_specular_glint", (-0.075, -0.705, bubble_center[2] + 0.085), 0.035, bubble_rim, segments=24)
    add_cylinder("bubble_release_nozzle", (0.0, 0.0, tank["bottom"] + 0.10), 0.09, 0.20, mats["steel"])
    # Accentuate the free surface as an optical boundary and frame the entire
    # water column, including the surface and tank rim.
    cylinder_between(
        "visible_free_surface_front_edge",
        (-1.03, -0.78, tank["liquid_top"] + 0.006),
        (1.03, -0.78, tank["liquid_top"] + 0.006),
        0.012,
        bubble_rim,
    )
    _front_camera((0.0, 0.0, 2.70), distance=12.8, height=3.75, lens=58)


def scene_p27(mats) -> None:
    top = add_lab_table(mats)
    tray_z = top + 0.08
    _tray(mats, "whole_tray", -1.55, 0.0, tray_z)
    _tray(mats, "crushed_tray", 1.55, 0.0, tray_z)
    add_box("whole_ice_sample", (-1.55, 0.0, tray_z + 0.46), (1.15, 0.85, 0.72), mats["ice"], bevel=0.12)
    piece_positions = (
        (-0.50, -0.42, 0.00), (-0.15, -0.42, 0.03), (0.20, -0.42, -0.02), (0.53, -0.42, 0.01),
        (-0.48, -0.04, 0.02), (-0.12, -0.04, -0.02), (0.24, -0.04, 0.03), (0.52, -0.04, -0.01),
        (-0.45, 0.34, -0.01), (-0.10, 0.34, 0.03), (0.25, 0.34, 0.00), (0.50, 0.34, 0.02),
    )
    for i, (dx, dy, dz) in enumerate(piece_positions):
        add_box(
            f"crushed_ice_{i}",
            (1.55 + dx, dy, tray_z + 0.36 + dz),
            (0.32, 0.31, 0.42),
            mats["ice"],
            rotation=(0.04 * math.sin(i), 0.05 * math.cos(i * 0.7), 0.12 * math.sin(i * 1.3)),
            bevel=0.07,
        )
    add_camera((4.8, -9.2, 6.6), (0.0, 0.0, 1.15), lens=58)


def scene_p28(mats) -> None:
    # The two equal-length black threads hang from nearby points on one rigid
    # support.  Their centers are separated by exactly two radii, so the balls
    # begin touching and can separate symmetrically in the generated video.
    top = add_lab_table(mats, z=0.64, top=(8.8, 3.2, 0.18))
    attach_z = top + 3.45
    add_box("suspension_crossbar", (0.0, 0.0, attach_z + 0.12), (2.25, 0.38, 0.24), mats["steel"], bevel=0.04)
    add_sphere("support_center_bolt", (0.0, 0.0, attach_z + 0.12), 0.11, mats["black"])
    radius = 0.29
    ball_x = radius
    ball_z = attach_z - 2.42
    for side, sign in (("left", -1), ("right", 1)):
        attach = (sign * 0.42, 0.0, attach_z)
        bob = (sign * ball_x, 0.0, ball_z)
        cylinder_between(f"{side}_black_equal_thread", attach, bob, 0.018, mats["black"])
        add_sphere(f"{side}_charged_pith_ball", bob, radius, mats["aluminum"])
    add_camera((0.0, -13.5, 2.45), (0.0, 0.0, 2.45), lens=53)


def scene_p29(mats) -> None:
    platform_z = 0.12
    person_x = -0.82
    add_box("insulating_platform", (person_x, 0.0, platform_z), (1.75, 1.12, 0.24), mats["rubber"], bevel=0.08)
    # Generator, complete from base to dome.
    add_box("generator_base", (1.65, 0.0, 0.22), (1.05, 0.90, 0.34), mats["black"], bevel=0.09)
    add_cylinder("generator_column", (1.65, 0.0, 1.55), 0.27, 2.45, mats["blue"])
    add_sphere("generator_dome", (1.65, 0.0, 3.18), 0.75, mats["aluminum"])
    # A seven-head-tall, rear-facing adult mannequin.  Rounded joints, tapered
    # clothing and articulated limbs avoid the previous block-figure look.
    skin = material("neutral_anonymous_skin", (0.48, 0.24, 0.13), roughness=0.68)
    shirt = material("matte_lab_shirt", (0.46, 0.55, 0.62), roughness=0.82)
    pants = material("dark_fabric_trousers", (0.035, 0.045, 0.065), roughness=0.88)
    shoe_z = 0.34
    for side, x in (("left", person_x - 0.17), ("right", person_x + 0.17)):
        add_box(f"{side}_shoe", (x, -0.08, shoe_z), (0.24, 0.48, 0.17), mats["rubber"], bevel=0.07)
        cylinder_between(f"{side}_adult_leg", (x, 0.0, 0.48), (x, 0.0, 1.64), 0.115, pants)
        add_sphere(f"{side}_knee", (x, 0.0, 1.06), 0.125, pants, segments=32)
    add_cone("adult_tapered_torso", (person_x, 0.0, 2.15), 0.31, 0.43, 1.10, shirt)
    add_cylinder("adult_neck", (person_x, 0.0, 2.78), 0.105, 0.20, skin)
    add_sphere("anonymous_rear_head", (person_x, 0.0, 3.02), 0.265, skin)
    # Relaxed left arm and articulated right arm reaching to the dome.
    left_shoulder = (person_x - 0.38, 0.0, 2.54)
    left_elbow = (person_x - 0.49, 0.0, 1.98)
    left_hand = (person_x - 0.43, 0.0, 1.50)
    cylinder_between("left_upper_arm", left_shoulder, left_elbow, 0.095, shirt)
    add_sphere("left_elbow_joint", left_elbow, 0.105, shirt, segments=32)
    cylinder_between("left_forearm", left_elbow, left_hand, 0.082, skin)
    add_sphere("left_relaxed_hand", left_hand, 0.11, skin, segments=32)
    right_shoulder = (person_x + 0.38, 0.0, 2.54)
    right_elbow = (0.10, 0.0, 2.78)
    dome_hand = (0.94, 0.0, 3.15)
    cylinder_between("right_upper_arm", right_shoulder, right_elbow, 0.095, shirt)
    add_sphere("right_elbow_joint", right_elbow, 0.105, shirt, segments=32)
    cylinder_between("right_forearm_to_dome", right_elbow, dome_hand, 0.082, skin)
    add_sphere("contacting_hand", dome_hand, 0.12, skin, segments=32)
    hair = material("anonymous_dark_hair", (0.035, 0.018, 0.012), roughness=0.76)
    add_sphere("rear_hair_cap", (person_x, -0.075, 3.06), 0.285, hair)
    # Long strands first follow the scalp, then bow outward under charge and
    # finally sag under gravity.  Six control points per strand read as soft
    # bundles rather than rigid radial spikes.
    for i in range(31):
        u = -1.0 + 2.0 * i / 30
        layer = (i % 4 - 1.5) * 0.018
        start = (person_x + 0.205 * u, -0.265 + layer, 3.12 + 0.10 * (1.0 - u * u))
        points = (
            start,
            (person_x + 0.28 * u, -0.30 + layer, 3.20 + 0.09 * (1.0 - abs(u))),
            (person_x + 0.47 * u, -0.32 + layer, 3.22 + 0.04 * (1.0 - abs(u))),
            (person_x + 0.70 * u, -0.31 + layer, 3.13 - 0.09 * abs(u)),
            (person_x + 0.88 * u, -0.28 + layer, 2.91 - 0.18 * abs(u)),
            (person_x + 1.00 * u, -0.24 + layer, 2.55 - 0.28 * (1.0 - abs(u))),
        )
        add_tube(f"flexible_charged_hair_{i}", points, 0.012 + 0.002 * (i % 3), hair, resolution=3)
    _front_camera((0.22, 0.0, 1.95), distance=13.2, height=3.15, lens=54)


def scene_p30(mats) -> None:
    top = add_lab_table(mats)
    water_stream = material("clear_thin_stream", (0.04, 0.30, 0.42), roughness=0.04, transmission=0.7, alpha=0.55)
    for side, x, outer_x, rod_mat in (("positive", -1.45, -3.7, mats["red"]), ("negative", 1.45, 3.7, mats["blue"])):
        add_box(f"{side}_nozzle_support", (x, 0.2, top + 2.65), (0.65, 0.60, 0.32), mats["steel"], bevel=0.05)
        add_cylinder(f"{side}_nozzle", (x, 0.0, top + 2.36), 0.10, 0.36, mats["steel"])
        add_tube(f"{side}_straight_water_stream", ((x, 0.0, top + 2.18), (x, 0.0, top + 0.38)), 0.055, water_stream, resolution=2)
        _tray(mats, f"{side}_catch_basin", x, 0.0, top + 0.07, width=1.42, depth=1.05)
        inner_end = x - 0.62 if x < 0 else x + 0.62
        cylinder_between(f"{side}_charged_rod", (outer_x, -0.12, top + 1.38), (inner_end, -0.12, top + 1.38), 0.13, rod_mat)
        add_sphere(f"{side}_rod_tip", (inner_end, -0.12, top + 1.38), 0.14, rod_mat)
    _front_camera((0.0, 0.0, 2.05), height=3.25, lens=58)


def scene_p31(mats) -> None:
    top = add_lab_table(mats)
    _wire_cage(mats, center=(-0.85, 0.20, top + 1.58), size=(3.0, 2.05, 2.85))
    _electroscope(mats, "inside_electroscope", -0.85, 0.18, top + 0.02, spread=0.02)
    _electroscope(mats, "outside_electroscope", 2.35, 0.0, top + 0.02, spread=0.02)
    add_sphere("external_charged_sphere", (3.85, 0.0, top + 2.20), 0.36, mats["red"])
    cylinder_between("insulating_sphere_support", (3.85, 0.0, top + 0.10), (3.85, 0.0, top + 1.90), 0.055, mats["rubber"])
    add_camera((5.9, -11.8, 4.8), (0.5, 0.0, 2.15), lens=58)


def scene_p32(mats) -> None:
    top = add_lab_table(mats, top=(8.8, 5.7, 0.18))
    add_bar_magnet(mats, location=(0.0, 0.0, top + 0.18), length=2.6, width=0.64, height=0.30, name="filings_bar_magnet")
    add_box("transparent_filing_sheet", (0.0, 0.0, top + 0.38), (7.0, 4.35, 0.08), mats["glass"], bevel=0.04)
    _scatter_filings(mats, "random_filing", centers=(0.0,), z=top + 0.46, count_per_center=62)
    _overhead_camera((0.0, 0.0, top + 0.1), height=10.8, lens=56)


def scene_p33(mats) -> None:
    top = add_lab_table(mats, top=(9.3, 6.0, 0.18))
    magnet_z = top + 0.20
    add_bar_magnet(mats, location=(0.0, 0.0, magnet_z), length=2.45, width=0.62, height=0.34, name="compass_bar_magnet")
    radius = 2.15
    for i in range(14):
        phi = 2 * math.pi * i / 14
        x, y = radius * math.cos(phi), radius * math.sin(phi)
        bx = 3 * math.cos(phi) ** 2 - 1
        by = 3 * math.cos(phi) * math.sin(phi)
        needle_angle = math.atan2(by, bx)
        add_compass(mats, location=(x, y, top + 0.16), angle=needle_angle, radius=0.25, name=f"dipole_compass_{i}")
    _overhead_camera((0.0, 0.0, top + 0.1), height=16.0, lens=50)


def scene_p33b(mats) -> None:
    top = add_lab_table(mats, top=(8.8, 5.8, 0.18))
    add_bar_magnet(mats, location=(0.0, -0.45, top + 0.18), length=2.25, width=0.62, height=0.32, name="single_line_magnet")
    add_box("clear_tracer_plate", (0.0, 0.0, top + 0.38), (7.0, 4.6, 0.08), mats["glass"], bevel=0.04)
    tracer_mat = material("ferromagnetic_tracer", (0.08, 0.09, 0.10), roughness=0.34, metallic=0.78)
    constant = 3.15
    for i in range(43):
        theta = 0.18 + (math.pi - 0.36) * i / 42
        radius = constant * math.sin(theta) ** 2
        x = radius * math.cos(theta)
        y = -0.45 + radius * math.sin(theta)
        add_sphere(f"single_fieldline_tracer_{i}", (x, y, top + 0.50), 0.045, tracer_mat, segments=24)
    _overhead_camera((0.0, 0.35, top + 0.1), height=10.8, lens=56)


def scene_p34(mats) -> None:
    top = add_lab_table(mats, top=(8.8, 5.8, 0.18))
    add_box("transparent_compass_board", (0.0, 0.0, top + 0.27), (6.8, 4.9, 0.10), mats["glass"], bevel=0.04)
    cylinder_between("straight_current_wire", (0.0, 0.0, top - 0.30), (0.0, 0.0, top + 1.10), 0.075, mats["copper"])
    add_cylinder("wire_insulating_bushing", (0.0, 0.0, top + 0.34), 0.17, 0.18, mats["rubber"])
    radius = 1.95
    for i in range(12):
        phi = 2 * math.pi * i / 12
        x, y = radius * math.cos(phi), radius * math.sin(phi)
        add_compass(mats, location=(x, y, top + 0.40), angle=phi + math.pi / 2, radius=0.25, name=f"wire_compass_{i}")
    _overhead_camera((0.0, 0.0, top + 0.15), height=16.0, lens=50)


def scene_p35(mats) -> None:
    top = add_lab_table(mats, top=(9.4, 5.8, 0.18))
    add_bar_magnet(mats, location=(-1.65, 0.0, top + 0.18), length=2.05, width=0.64, height=0.33, name="left_broken_fragment")
    add_bar_magnet(mats, location=(1.65, 0.0, top + 0.18), length=2.05, width=0.64, height=0.33, name="right_broken_fragment")
    add_box("transparent_fragment_sheet", (0.0, 0.0, top + 0.39), (8.0, 4.6, 0.08), mats["glass"], bevel=0.04)
    # A few small matching chips make the two inward faces read as one fresh break.
    add_cone("fracture_chip_left", (-0.48, 0.08, top + 0.47), 0.10, 0.025, 0.10, mats["steel"], rotation=(math.pi / 2, 0.0, 0.2), vertices=5)
    add_cone("fracture_chip_right", (0.48, -0.08, top + 0.47), 0.10, 0.025, 0.10, mats["steel"], rotation=(math.pi / 2, 0.0, -0.2), vertices=5)
    _scatter_filings(mats, "fragment_filing", centers=(-1.65, 1.65), z=top + 0.49, count_per_center=33)
    _overhead_camera((0.0, 0.0, top + 0.1), height=11.2, lens=55)


Builder = Callable[[dict], None]

BUILDERS: dict[str, tuple[Builder, bool]] = {
    "P20": (scene_p20, False),
    "P21": (scene_p21, False),
    "P21b": (scene_p21b, False),
    "P21c": (scene_p21c, False),
    "P22": (scene_p22, False),
    "P23": (scene_p23, False),
    "P24": (scene_p24, True),
    "P25": (scene_p25, True),
    "P26": (scene_p26, False),
    "P27": (scene_p27, False),
    "P28": (scene_p28, False),
    "P29": (scene_p29, False),
    "P30": (scene_p30, False),
    "P31": (scene_p31, False),
    "P32": (scene_p32, False),
    "P33": (scene_p33, False),
    "P33b": (scene_p33b, False),
    "P34": (scene_p34, False),
    "P35": (scene_p35, False),
}
