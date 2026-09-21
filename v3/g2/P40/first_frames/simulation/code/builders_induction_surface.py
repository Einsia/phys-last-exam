#!/usr/bin/env python3
"""Procedural first-frame builders for Physical-Bench P36--P50.

The scenes are deliberately stable, static initial configurations.  They use
only observable apparatus geometry: no captions, arrows, paths, equations, or
other hints about the expected outcome are rendered into a frame.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from simulation.common import (
    add_bar_magnet,
    add_box,
    add_camera,
    add_coil,
    add_cone,
    add_cylinder,
    add_glass_tank,
    add_lab_table,
    add_mesh,
    add_sphere,
    add_torus,
    add_tube,
    cylinder_between,
    material,
    ramp_point,
)


Builder = Callable[[dict], None]


def _front_camera(target=(0.0, 0.0, 2.2), *, distance: float = 12.0, height: float = 3.5, lens: float = 56.0) -> None:
    add_camera((0.0, -distance, height), target, lens=lens)


def _tray(mats, name: str, center: Sequence[float], size=(2.2, 1.45)) -> None:
    x, y, z = center
    sx, sy = size
    add_box(f"{name}_base", (x, y, z), (sx, sy, 0.10), mats["black"], bevel=0.035)
    for side, dx, dy, dims in (
        ("front", 0.0, -sy / 2, (sx, 0.07, 0.22)),
        ("back", 0.0, sy / 2, (sx, 0.07, 0.22)),
        ("left", -sx / 2, 0.0, (0.07, sy, 0.22)),
        ("right", sx / 2, 0.0, (0.07, sy, 0.22)),
    ):
        add_box(f"{name}_{side}_rim", (x + dx, y + dy, z + 0.11), dims, mats["steel"], bevel=0.018)


def _arc_tube(
    name: str,
    center: Sequence[float],
    radius: float,
    tube_radius: float,
    mat,
    *,
    start: float,
    end: float,
    samples: int = 72,
) -> None:
    cx, cy, cz = center
    points = []
    for index in range(samples + 1):
        angle = start + (end - start) * index / samples
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle), cz))
    add_tube(name, points, tube_radius, mat, resolution=2)


def _local_xz(center: Sequence[float], angle: float, x: float, z: float) -> tuple[float, float, float]:
    """Transform local x/z coordinates by a positive Blender Y rotation."""
    cx, cy, cz = center
    return (
        cx + math.cos(angle) * x + math.sin(angle) * z,
        cy,
        cz - math.sin(angle) * x + math.cos(angle) * z,
    )


def _spherical_cap_z(name: str, center_x: float, center_y: float, base_z: float, base_radius: float, height: float, mat) -> None:
    """Create a closed spherical-cap sessile droplet with a horizontal base."""
    sphere_radius = (base_radius * base_radius + height * height) / (2.0 * height)
    sphere_center_z = base_z + height - sphere_radius
    theta_max = math.acos(max(-1.0, min(1.0, (base_z - sphere_center_z) / sphere_radius)))
    segments, rings = 56, 18
    vertices: list[tuple[float, float, float]] = [(center_x, center_y, sphere_center_z + sphere_radius)]
    for ring in range(1, rings + 1):
        theta = theta_max * ring / rings
        ring_radius = sphere_radius * math.sin(theta)
        z = sphere_center_z + sphere_radius * math.cos(theta)
        for segment in range(segments):
            phi = 2.0 * math.pi * segment / segments
            vertices.append((center_x + ring_radius * math.cos(phi), center_y + ring_radius * math.sin(phi), z))

    faces: list[tuple[int, ...]] = []
    for segment in range(segments):
        faces.append((0, 1 + segment, 1 + (segment + 1) % segments))
    for ring in range(rings - 1):
        first = 1 + ring * segments
        second = first + segments
        for segment in range(segments):
            nxt = (segment + 1) % segments
            faces.append((first + segment, second + segment, second + nxt, first + nxt))
    base_first = 1 + (rings - 1) * segments
    faces.append(tuple(base_first + index for index in reversed(range(segments))))
    add_mesh(name, vertices, faces, mat, smooth=True)


def _spherical_cap_x(
    name: str,
    base_x: float,
    center_y: float,
    center_z: float,
    base_radius: float,
    height: float,
    mat,
    *,
    direction: float = 1.0,
) -> None:
    """Create a spherical cap that bulges along either X direction."""
    direction = 1.0 if direction >= 0.0 else -1.0
    sphere_radius = (base_radius * base_radius + height * height) / (2.0 * height)
    signed_center_offset = direction * (height - sphere_radius)
    sphere_center_x = base_x + signed_center_offset
    theta_max = math.acos(max(-1.0, min(1.0, (base_x - sphere_center_x) / (direction * sphere_radius))))
    segments, rings = 56, 16
    vertices: list[tuple[float, float, float]] = [(sphere_center_x + direction * sphere_radius, center_y, center_z)]
    for ring in range(1, rings + 1):
        theta = theta_max * ring / rings
        ring_radius = sphere_radius * math.sin(theta)
        x = sphere_center_x + direction * sphere_radius * math.cos(theta)
        for segment in range(segments):
            phi = 2.0 * math.pi * segment / segments
            vertices.append((x, center_y + ring_radius * math.sin(phi), center_z + ring_radius * math.cos(phi)))

    faces: list[tuple[int, ...]] = []
    for segment in range(segments):
        faces.append((0, 1 + (segment + 1) % segments, 1 + segment))
    for ring in range(rings - 1):
        first = 1 + ring * segments
        second = first + segments
        for segment in range(segments):
            nxt = (segment + 1) % segments
            faces.append((first + segment, first + nxt, second + nxt, second + segment))
    if direction < 0.0:
        faces = [tuple(reversed(face)) for face in faces]
    add_mesh(name, vertices, faces, mat, smooth=True)


def _concave_meniscus(
    name: str,
    center_x: float,
    center_y: float,
    rim_z: float,
    bore_radius: float,
    sag: float,
    mat,
) -> None:
    """Create a shallow wetting meniscus constrained to one circular bore."""
    segments, rings = 48, 6
    vertices: list[tuple[float, float, float]] = [(center_x, center_y, rim_z - sag)]
    for ring in range(1, rings + 1):
        radial_fraction = ring / rings
        radius = bore_radius * radial_fraction
        z = rim_z - sag * (1.0 - radial_fraction * radial_fraction)
        for segment in range(segments):
            phi = 2.0 * math.pi * segment / segments
            vertices.append((center_x + radius * math.cos(phi), center_y + radius * math.sin(phi), z))
    faces: list[tuple[int, ...]] = []
    for segment in range(segments):
        faces.append((0, 1 + segment, 1 + (segment + 1) % segments))
    for ring in range(rings - 1):
        first = 1 + ring * segments
        second = first + segments
        for segment in range(segments):
            nxt = (segment + 1) % segments
            faces.append((first + segment, second + segment, second + nxt, first + nxt))
    add_mesh(name, vertices, faces, mat, smooth=True)

    # A real meniscus catches light along its front meridian.  This curved,
    # material-bearing line is part of the liquid surface, not an annotation.
    profile = []
    for index in range(17):
        local_x = -bore_radius + 2.0 * bore_radius * index / 16
        radial_fraction = abs(local_x) / bore_radius
        z = rim_z - sag * (1.0 - radial_fraction * radial_fraction)
        profile.append((center_x + local_x, center_y - bore_radius * 0.10, z))
    add_tube(f"{name}_front_profile", profile, max(0.008, bore_radius * 0.055), mat, resolution=2)


def _scene_p36_shared(mats) -> None:
    """Matched magnetic and plastic pucks held on one copper incline."""
    top = add_lab_table(mats, z=0.62, top=(10.5, 4.0, 0.18))
    angle = math.radians(18.0)
    center = (0.0, 0.0, top + 1.25)
    add_box(
        "P36_thick_copper_ramp",
        center,
        (8.0, 2.55, 0.18),
        mats["copper"],
        rotation=(0.0, angle, 0.0),
        bevel=0.045,
    )
    # Rigid triangular-looking support members keep the conductor visibly inclined.
    cylinder_between("P36_high_support", (-3.45, 0.82, top), (-3.45, 0.82, top + 2.45), 0.075, mats["steel"])
    cylinder_between("P36_high_support_brace", (-3.45, 0.82, top + 2.42), (-1.75, 0.82, top), 0.065, mats["steel"])
    cylinder_between("P36_hinge", (3.65, -1.25, top + 0.16), (3.65, 1.25, top + 0.16), 0.105, mats["steel"])

    puck_u = -2.45
    puck_depth = 0.24
    for name, lane, puck_mat in (
        ("magnet", -0.58, mats["steel"]),
        ("plastic_control", 0.58, mats["blue"]),
    ):
        point = ramp_point(center, angle, puck_u, lane, 0.18 + puck_depth / 2)
        add_cylinder(
            f"P36_{name}_puck",
            tuple(point),
            0.34,
            puck_depth,
            puck_mat,
            rotation=(0.0, angle, 0.0),
            vertices=64,
        )

    gate_point = ramp_point(center, angle, -2.83, 0.0, 0.30)
    cylinder_between(
        "P36_synchronized_release_bar",
        (gate_point.x, -1.03, gate_point.z),
        (gate_point.x, 1.03, gate_point.z),
        0.075,
        mats["black"],
    )
    for lane in (-1.03, 1.03):
        foot = ramp_point(center, angle, -2.83, lane, 0.11)
        cylinder_between(
            f"P36_gate_post_{lane:+.2f}",
            tuple(foot),
            (gate_point.x, lane, gate_point.z),
            0.045,
            mats["steel"],
        )
    add_camera((7.2, -11.8, 5.3), (0.0, 0.0, top + 1.0), lens=55.0)


def scene_p36(mats) -> None:
    _scene_p36_shared(mats)


def scene_p36b(mats) -> None:
    _scene_p36_shared(mats)


def scene_p37(mats) -> None:
    """Closed and split jumping rings on matched solenoid cores."""
    top = add_lab_table(mats, z=0.62, top=(9.8, 3.7, 0.18))
    ring_z = top + 1.55
    for side, x in (("closed", -2.0), ("open", 2.0)):
        add_cylinder(f"P37_{side}_coil_form", (x, 0.0, top + 0.66), 0.61, 0.92, mats["black"])
        add_coil(mats, center=(x, 0.0, top + 0.69), radius=0.66, length=0.80, turns=13, axis="z", name=f"P37_{side}_solenoid")
        add_cylinder(f"P37_{side}_iron_core", (x, 0.0, top + 2.12), 0.25, 3.35, mats["steel"])
        add_cylinder(f"P37_{side}_ring_rest", (x, 0.0, ring_z - 0.08), 0.39, 0.11, mats["black"])
    add_torus("P37_closed_aluminum_ring", (-2.0, 0.0, ring_z), 0.58, 0.075, mats["aluminum"], rotation=(0.0, 0.0, 0.0))
    gap = math.radians(25.0)
    _arc_tube(
        "P37_open_aluminum_ring",
        (2.0, 0.0, ring_z),
        0.58,
        0.075,
        mats["aluminum"],
        start=-math.pi / 2 + gap / 2,
        end=3 * math.pi / 2 - gap / 2,
    )
    add_box("P37_shared_rigid_backplate", (0.0, 0.78, top + 2.05), (6.2, 0.16, 4.05), mats["white"], bevel=0.035)
    _front_camera((0.0, 0.0, top + 2.15), distance=12.8, height=5.20, lens=57.0)


def _pendulum_plate(mats, *, name: str, pivot_x: float, pivot_z: float, angle: float, slotted: bool) -> None:
    arm_length = 2.05
    plate_center = (
        pivot_x + arm_length * math.sin(angle),
        0.0,
        pivot_z - arm_length * math.cos(angle),
    )
    plate_top = _local_xz(plate_center, angle, 0.0, 0.64)
    cylinder_between(f"{name}_arm", (pivot_x, 0.0, pivot_z), plate_top, 0.055, mats["steel"])
    add_cylinder(f"{name}_pivot", (pivot_x, 0.0, pivot_z), 0.11, 0.46, mats["steel"], rotation=(math.pi / 2, 0.0, 0.0))
    if not slotted:
        add_box(name, plate_center, (1.08, 0.12, 1.22), mats["aluminum"], rotation=(0.0, angle, 0.0), bevel=0.025)
        return

    # An equal-outline comb plate whose long cuts interrupt circulating currents.
    spine = _local_xz(plate_center, angle, 0.0, 0.48)
    add_box(f"{name}_top_spine", spine, (1.08, 0.12, 0.25), mats["aluminum"], rotation=(0.0, angle, 0.0), bevel=0.018)
    for index in range(6):
        local_x = -0.45 + index * 0.18
        finger = _local_xz(plate_center, angle, local_x, -0.12)
        add_box(f"{name}_finger_{index}", finger, (0.11, 0.12, 0.94), mats["aluminum"], rotation=(0.0, angle, 0.0), bevel=0.014)


def scene_p38(mats) -> None:
    """Matched solid and slotted eddy-current pendulums before release."""
    top = add_lab_table(mats, z=0.52, top=(10.4, 3.7, 0.18))
    pivot_z = top + 4.25
    angle = math.radians(-25.0)
    add_box("P38_crossbar", (0.0, 0.45, pivot_z + 0.14), (7.2, 0.28, 0.28), mats["steel"], bevel=0.04)
    for x in (-3.35, 3.35):
        add_box(f"P38_frame_post_{x:+.2f}", (x, 0.45, top + 2.05), (0.20, 0.25, 4.1), mats["steel"], bevel=0.025)
    _pendulum_plate(mats, name="P38_solid_plate", pivot_x=-1.85, pivot_z=pivot_z, angle=angle, slotted=False)
    _pendulum_plate(mats, name="P38_slotted_plate", pivot_x=1.85, pivot_z=pivot_z, angle=angle, slotted=True)
    # Identical pole pairs wait at the bottom of the two future swing paths.
    for x in (-1.85, 1.85):
        magnet_z = pivot_z - 2.70
        add_box(f"P38_front_pole_{x:+.2f}", (x, -0.48, magnet_z), (0.92, 0.38, 0.82), mats["magnet_red"], bevel=0.05)
        add_box(f"P38_back_pole_{x:+.2f}", (x, 0.48, magnet_z), (0.92, 0.38, 0.82), mats["magnet_blue"], bevel=0.05)
    # One rigid cross-shaft carries two linked catches at the exact same height.
    # Each catch visibly overlaps one pendulum arm, so neither plate appears to
    # hang unsupported at its initial angle.
    latch_drop = 0.86
    latch_z = pivot_z - latch_drop
    arm_fraction = latch_drop / (2.05 * math.cos(angle))
    catch_xs = tuple(pivot_x + 2.05 * math.sin(angle) * arm_fraction for pivot_x in (-1.85, 1.85))
    cylinder_between(
        "P38_synchronized_latch_shaft",
        (catch_xs[0] - 0.42, -0.16, latch_z),
        (catch_xs[1] + 0.42, -0.16, latch_z),
        0.060,
        mats["black"],
    )
    for side, catch_x in zip(("solid", "slotted"), catch_xs):
        add_box(
            f"P38_{side}_linked_catch",
            (catch_x, -0.13, latch_z),
            (0.28, 0.28, 0.22),
            mats["black"],
            bevel=0.035,
        )
    add_box(
        "P38_common_latch_actuator",
        (catch_xs[1] + 0.58, -0.16, latch_z),
        (0.34, 0.40, 0.42),
        mats["steel"],
        bevel=0.05,
    )
    _front_camera((0.0, 0.0, top + 2.45), distance=13.8, height=3.7, lens=55.0)


def scene_p39(mats) -> None:
    """Stationary bar magnet, closed coil circuit, and unlit lamp."""
    top = add_lab_table(mats, z=0.62, top=(10.2, 3.6, 0.18))
    axis_z = top + 1.65
    add_coil(mats, center=(0.0, 0.0, axis_z), radius=0.76, length=1.55, turns=19, axis="x", name="P39_pickup_coil")
    add_cylinder("P39_coil_left_support", (-0.62, 0.0, top + 0.67), 0.075, 1.35, mats["steel"])
    add_cylinder("P39_coil_right_support", (0.62, 0.0, top + 0.67), 0.075, 1.35, mats["steel"])
    add_bar_magnet(mats, location=(-1.72, 0.0, axis_z), length=2.0, width=0.48, height=0.42, name="P39_stationary_bar_magnet")
    add_box("P39_linear_rail", (-2.25, 0.32, top + 0.35), (4.2, 0.36, 0.18), mats["black"], bevel=0.035)
    add_box("P39_magnet_carriage", (-1.72, 0.32, top + 0.55), (0.72, 0.54, 0.28), mats["steel"], bevel=0.04)

    bulb_x = 3.15
    add_cylinder("P39_lamp_base", (bulb_x, 0.0, top + 0.54), 0.32, 0.52, mats["brass"])
    add_sphere("P39_clear_unlit_bulb", (bulb_x, 0.0, top + 1.18), 0.48, mats["glass"])
    filament = (
        (bulb_x - 0.16, -0.06, top + 0.82),
        (bulb_x - 0.12, -0.06, top + 1.17),
        (bulb_x, -0.06, top + 1.31),
        (bulb_x + 0.12, -0.06, top + 1.17),
        (bulb_x + 0.16, -0.06, top + 0.82),
    )
    add_tube("P39_dark_filament", filament, 0.018, mats["black"], resolution=2)
    add_tube(
        "P39_upper_circuit_wire",
        ((0.78, 0.18, axis_z + 0.55), (1.45, 0.18, top + 2.45), (3.15, 0.18, top + 2.45), (3.15, 0.18, top + 1.62)),
        0.032,
        mats["wire"],
        resolution=2,
    )
    add_tube(
        "P39_lower_circuit_wire",
        ((-0.78, 0.18, axis_z - 0.55), (-0.20, 0.18, top + 0.26), (3.15, 0.18, top + 0.26)),
        0.032,
        mats["wire"],
        resolution=2,
    )
    _front_camera((0.25, 0.0, top + 1.55), distance=12.4, height=3.15, lens=58.0)


def scene_p40(mats) -> None:
    """Two equal-bore vertical glass tubes with unequal sand charges."""
    ground = material("P40_level_ground", (0.30, 0.32, 0.34), roughness=0.88)
    add_box("P40_shared_horizontal_surface", (0.0, 0.45, 0.15), (10.8, 4.2, 0.30), ground, bevel=0.025)
    tube_glass = material("P40_transparent_equal_bore_glass", (0.32, 0.58, 0.66), roughness=0.035, transmission=0.90, alpha=0.24)
    sand = material("P40_same_dry_noncohesive_sand", (0.54, 0.34, 0.13), roughness=0.97)
    valve_mat = material("P40_valve_black", (0.012, 0.016, 0.020), roughness=0.28, metallic=0.55)
    tube_radius, tube_height, tube_bottom = 0.78, 3.40, 3.30
    for side, x, sand_height in (("left", -2.05, 1.95), ("right", 2.05, 1.18)):
        add_cylinder(
            f"P40_{side}_vertical_glass_tube",
            (x, 0.0, tube_bottom + tube_height / 2),
            tube_radius,
            tube_height,
            tube_glass,
            vertices=72,
        )
        add_torus(f"P40_{side}_glass_top_rim", (x, 0.0, tube_bottom + tube_height), tube_radius * 0.98, 0.035, tube_glass)
        add_cylinder(
            f"P40_{side}_sand_column",
            (x, -0.015, tube_bottom + sand_height / 2),
            tube_radius * 0.78,
            sand_height,
            sand,
            vertices=72,
        )
        # Narrow outlet and a closed butterfly-style gate beneath each tube.
        add_cylinder(f"P40_{side}_bottom_outlet", (x, 0.0, tube_bottom - 0.20), 0.22, 0.42, tube_glass, vertices=48)
        add_cylinder(f"P40_{side}_valve_body", (x, 0.0, tube_bottom - 0.47), 0.28, 0.18, valve_mat, vertices=48)
        add_box(f"P40_{side}_valve_handle", (x, -0.02, tube_bottom - 0.47), (0.72, 0.10, 0.10), mats["steel"], bevel=0.02)

    add_box("P40_shared_support_crossbar", (0.0, 0.52, tube_bottom + tube_height - 0.18), (5.8, 0.14, 0.16), mats["steel"], bevel=0.02)
    for x in (-3.10, 3.10):
        add_box(f"P40_support_post_{x:+.2f}", (x, 0.52, 2.35), (0.16, 0.16, 4.7), mats["steel"], bevel=0.02)
    add_camera((0.0, -15.5, 3.75), (0.0, 0.0, 3.75), lens=48.0)


def scene_p41(mats) -> None:
    """Matched sand and water vessels immediately before simultaneous outflow."""
    sand_center, water_center = (-1.85, 0.0, 3.12), (1.85, 0.0, 3.12)
    for name, center, liquid in (
        ("sand", sand_center, mats["sand"]),
        ("water", water_center, mats["water"]),
    ):
        tank = add_glass_tank(mats, center=center, size=(2.10, 1.35, 2.65), bottom_z=1.77, fill=0.78, liquid=liquid, name=f"P41_{name}_vessel")
        add_cylinder(f"P41_{name}_outlet", (center[0], 0.0, tank["bottom"] - 0.18), 0.14, 0.42, mats["steel"])
        add_box(f"P41_{name}_closed_gate", (center[0], -0.02, tank["bottom"] - 0.02), (0.58, 0.30, 0.09), mats["black"], bevel=0.018)
        _tray(mats, f"P41_{name}_collector", (center[0], 0.0, 0.24), size=(2.2, 1.35))
    add_box("P41_frame_crossbar", (0.0, 0.72, 4.68), (6.4, 0.15, 0.15), mats["steel"], bevel=0.02)
    for x in (-3.0, 3.0):
        add_box(f"P41_frame_post_{x:+.2f}", (x, 0.72, 2.45), (0.15, 0.15, 4.5), mats["steel"], bevel=0.02)
    _front_camera((0.0, 0.0, 2.45), distance=12.8, height=3.45, lens=58.0)


def scene_p42(mats) -> None:
    """Different-volume blocks on one slowly tilting board."""
    # Use a dark floor and one clearly isolated board so the downhill path is
    # not mistaken for a second platform in the side view.
    add_box("P42_floor_base", (0.0, 0.55, 0.16), (11.6, 4.0, 0.32), mats["dark"], bevel=0.02)
    top = 0.32
    angle = math.radians(4.5)
    center = (0.0, 0.0, top + 0.80)
    add_box("P42_shared_hinged_board", center, (8.6, 2.65, 0.20), mats["wood"], rotation=(0.0, angle, 0.0), bevel=0.045)
    # Mass contrast comes only from volume: both blocks use the same wood
    # material and contact treatment, while the heavy block is physically
    # larger.  They are arranged side by side along the downhill axis so a
    # straight front camera can see both silhouettes without perspective yaw.
    block_specs = (
        ("light", -2.05, (0.98, 0.78, 0.58)),
        ("heavy", -0.35, (1.38, 0.92, 0.82)),
    )
    for name, block_u, block_dims in block_specs:
        point = ramp_point(center, angle, block_u, 0.0, 0.10 + block_dims[2] / 2)
        add_box(
            f"P42_{name}_wood_block",
            tuple(point),
            block_dims,
            mats["wood_dark"],
            rotation=(0.0, angle, 0.0),
            bevel=0.055,
        )
    hinge = ramp_point(center, angle, 4.12, 0.0, -0.02)
    cylinder_between("P42_board_hinge", (hinge.x, -1.38, hinge.z), (hinge.x, 1.38, hinge.z), 0.12, mats["steel"])
    screw_x = -3.88
    cylinder_between("P42_slow_lifting_screw", (screw_x, 0.92, top + 0.08), (screw_x, 0.92, center[2] + 0.46), 0.085, mats["steel"])
    add_cylinder("P42_lift_motor", (screw_x, 0.92, top + 0.18), 0.28, 0.52, mats["black"])
    # Straight front view: the board and both differently sized blocks remain
    # unrotated in the image while the full downhill path stays visible.
    add_camera((0.0, -13.8, 1.25), (0.0, 0.0, 1.25), lens=54.0)


def scene_p43(mats) -> None:
    """Uniform cube upright against a slow motorized pusher."""
    top = add_lab_table(mats, z=0.62, top=(10.4, 3.8, 0.18))
    add_box("P43_high_friction_strip", (0.55, 0.0, top + 0.06), (4.4, 1.35, 0.12), mats["rubber"], bevel=0.025)
    cube_size = 1.62
    cube_x = 0.55
    add_box("P43_homogeneous_cube", (cube_x, 0.0, top + 0.12 + cube_size / 2), (cube_size, cube_size, cube_size), mats["wood"], bevel=0.045)
    contact_x = cube_x - cube_size / 2 - 0.07
    contact_z = top + 0.12 + cube_size * 0.76
    add_box("P43_pusher_contact_pad", (contact_x, 0.0, contact_z), (0.14, 0.72, 0.52), mats["black"], bevel=0.025)
    cylinder_between("P43_actuator_rod", (-3.0, 0.0, contact_z), (contact_x - 0.08, 0.0, contact_z), 0.10, mats["steel"])
    add_box("P43_motorized_actuator", (-3.45, 0.0, contact_z), (0.88, 0.92, 0.82), mats["black"], bevel=0.08)
    add_box("P43_actuator_rail", (-1.55, 0.45, top + 0.24), (4.55, 0.25, 0.20), mats["steel"], bevel=0.025)
    _front_camera((0.25, 0.0, top + 1.35), distance=11.8, height=3.05, lens=62.0)


def scene_p44(mats) -> None:
    """Uniform-link chain settled into a deep catenary between equal anchors."""
    add_box("P44_high_contrast_backdrop", (0.0, 0.85, 2.75), (9.2, 0.16, 5.5), mats["white"], bevel=0.0)
    anchor_x = 3.42
    anchor_z = 4.52
    for x in (-anchor_x, anchor_x):
        add_box(f"P44_support_post_{x:+.2f}", (x, 0.55, 2.3), (0.22, 0.30, 4.6), mats["steel"], bevel=0.03)
        add_cylinder(f"P44_anchor_{x:+.2f}", (x, 0.0, anchor_z), 0.15, 0.75, mats["brass"], rotation=(math.pi / 2, 0.0, 0.0))

    a = 2.2
    bottom_z = 1.32
    count = 35
    for index in range(count):
        x = -anchor_x + 2.0 * anchor_x * index / (count - 1)
        z = bottom_z + a * (math.cosh(x / a) - 1.0)
        rotation = (math.pi / 2, 0.0, 0.0) if index % 2 == 0 else (0.0, math.pi / 2, 0.0)
        add_torus(f"P44_chain_link_{index}", (x, 0.0, z), 0.125, 0.036, mats["steel"], rotation=rotation)
    _front_camera((0.0, 0.0, 2.8), distance=13.5, height=3.0, lens=66.0)


def scene_p45(mats) -> None:
    """Two different-bore capillaries sharing one water reservoir before rise."""
    top = add_lab_table(mats, z=0.52, top=(9.8, 3.8, 0.18))
    backdrop = material("P45_optical_backdrop", (0.11, 0.14, 0.16), roughness=0.82)
    capillary_water = material(
        "P45_visible_capillary_water",
        (0.025, 0.34, 0.52),
        roughness=0.045,
        transmission=0.42,
        alpha=0.68,
    )
    add_box("P45_dark_optical_backdrop", (0.0, 0.92, top + 2.28), (6.25, 0.12, 4.55), backdrop, bevel=0.025)
    tank = add_glass_tank(mats, center=(0.0, 0.0, top + 0.90), size=(5.2, 1.6, 1.65), bottom_z=top + 0.05, fill=0.58, name="P45_common_water_reservoir")
    lower_z = tank["bottom"] + 0.28
    upper_z = tank["top"] + 2.60
    meniscus_level = tank["liquid_top"] + 0.030
    for name, x, outer_radius, bore_radius in (
        ("narrow", -1.10, 0.17, 0.065),
        ("wide", 1.10, 0.36, 0.245),
    ):
        add_cylinder(
            f"P45_{name}_glass_capillary",
            (x, 0.0, (lower_z + upper_z) / 2),
            outer_radius,
            upper_z - lower_z,
            mats["glass"],
            vertices=72,
        )
        sag = min(0.058, bore_radius * 0.30)
        column_top = meniscus_level - sag * 0.72
        add_cylinder(
            f"P45_{name}_initial_water_column",
            (x, -0.01, (lower_z + column_top) / 2),
            bore_radius,
            column_top - lower_z,
            capillary_water,
            vertices=64,
        )
        _concave_meniscus(
            f"P45_{name}_initial_meniscus",
            x,
            -0.01,
            meniscus_level,
            bore_radius * 0.94,
            sag,
            capillary_water,
        )
    add_box("P45_upper_clamp_bar", (0.0, 0.44, upper_z - 0.38), (4.4, 0.20, 0.24), mats["steel"], bevel=0.035)
    for x in (-1.10, 1.10):
        add_torus(f"P45_clamp_ring_{x:+.2f}", (x, 0.0, upper_z - 0.38), 0.34, 0.055, mats["black"], rotation=(0.0, 0.0, 0.0))
    add_box("P45_clamp_post", (3.0, 0.44, top + 2.15), (0.18, 0.22, 4.3), mats["steel"], bevel=0.025)
    _front_camera((0.0, 0.0, top + 2.15), distance=12.6, height=3.5, lens=62.0)


def scene_p46(mats) -> None:
    """Irregular polydisperse quasi-2D foam with many three-film junctions."""
    top = add_lab_table(mats, z=0.50, top=(9.5, 3.2, 0.18))
    plate_center_z = top + 2.35
    film = material(
        "P46_visible_soap_border",
        (0.20, 0.64, 0.92),
        roughness=0.06,
        transmission=0.30,
        alpha=0.78,
        emission=(0.08, 0.32, 0.58),
        emission_strength=0.65,
    )
    add_box("P46_back_glass_plate", (0.0, 0.22, plate_center_z), (7.0, 0.055, 4.45), mats["glass"], bevel=0.025)
    add_box("P46_front_glass_plate", (0.0, -0.22, plate_center_z), (7.0, 0.055, 4.45), mats["glass"], bevel=0.025)
    # A visible perimeter makes the two-plate apparatus legible even against a
    # dark shell; corner cylinders are the uniform gap spacers.
    add_box("P46_plate_top_edge", (0.0, -0.25, plate_center_z + 2.23), (7.18, 0.12, 0.12), mats["aluminum"], bevel=0.022)
    add_box("P46_plate_bottom_edge", (0.0, -0.25, plate_center_z - 2.23), (7.18, 0.12, 0.12), mats["aluminum"], bevel=0.022)
    add_box("P46_plate_left_edge", (-3.56, -0.25, plate_center_z), (0.12, 0.12, 4.52), mats["aluminum"], bevel=0.022)
    add_box("P46_plate_right_edge", (3.56, -0.25, plate_center_z), (0.12, 0.12, 4.52), mats["aluminum"], bevel=0.022)
    for x in (-3.35, 3.35):
        for z in (plate_center_z - 2.08, plate_center_z + 2.08):
            add_cylinder(
                f"P46_plate_spacer_{x:+.2f}_{z:.2f}",
                (x, 0.0, z),
                0.12,
                0.52,
                mats["brass"],
                rotation=(math.pi / 2, 0.0, 0.0),
            )

    # Begin with a finite honeycomb topology so all shared boundaries and
    # three-way nodes are watertight.  A deterministic spatial warp then makes
    # cell areas and film curvature visibly nonuniform without creating planar
    # four-way crossings.
    cell_radius = 0.62
    raw_edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for column in range(-2, 3):
        for row in range(-1, 2):
            center_x = 1.5 * cell_radius * column
            center_z = math.sqrt(3.0) * cell_radius * (row + 0.5 * (column % 2))
            keys = []
            for corner in range(6):
                angle = corner * math.pi / 3.0
                keys.append(
                    (
                        round(center_x + cell_radius * math.cos(angle), 4),
                        round(center_z + cell_radius * math.sin(angle), 4),
                    )
                )
            for corner in range(6):
                edge = tuple(sorted((keys[corner], keys[(corner + 1) % 6])))
                raw_edges.add(edge)

    raw_vertices = sorted({key for edge in raw_edges for key in edge})
    warped: dict[tuple[float, float], tuple[float, float]] = {}
    for index, (x, z) in enumerate(raw_vertices):
        warped_x = x + 0.12 * math.sin(1.37 * z + index * 0.41) + 0.035 * z
        warped_z = z + 0.10 * math.cos(1.11 * x - index * 0.33) - 0.025 * x
        warped[(x, z)] = (
            max(-3.22, min(3.22, warped_x)),
            max(-1.96, min(1.96, warped_z)),
        )

    degree = {key: 0 for key in raw_vertices}
    for edge in raw_edges:
        degree[edge[0]] += 1
        degree[edge[1]] += 1
    for index, edge in enumerate(sorted(raw_edges)):
        x0, z0 = warped[edge[0]]
        x1, z1 = warped[edge[1]]
        dx, dz = x1 - x0, z1 - z0
        length = max(0.001, math.hypot(dx, dz))
        bow = 0.035 * math.sin(index * 1.71) + 0.018 * math.cos(index * 0.93)
        midpoint = (
            (x0 + x1) / 2.0 - bow * dz / length,
            -0.015,
            plate_center_z + (z0 + z1) / 2.0 + bow * dx / length,
        )
        add_tube(
            f"P46_irregular_film_{index}",
            ((x0, -0.015, plate_center_z + z0), midpoint, (x1, -0.015, plate_center_z + z1)),
            0.024,
            film,
            resolution=3,
        )
    node_index = 0
    for key in raw_vertices:
        if degree[key] != 3:
            continue
        x, z = warped[key]
        add_sphere(
            f"P46_three_film_node_{node_index}",
            (x, -0.015, plate_center_z + z),
            0.050,
            film,
            segments=24,
        )
        node_index += 1
    add_box("P46_plate_base", (0.0, 0.0, top + 0.08), (7.6, 1.05, 0.16), mats["black"], bevel=0.04)
    _front_camera((0.0, 0.0, plate_center_z), distance=13.4, height=plate_center_z + 0.12, lens=58.0)


def scene_p47(mats) -> None:
    """Unequal connected soap bubbles with one curved separator."""
    top = add_lab_table(mats, z=0.50, top=(9.5, 3.4, 0.18))
    soap_shell = material("P47_thin_soap_shell", (0.18, 0.48, 0.76), roughness=0.035, transmission=0.78, alpha=0.20)
    separator = material(
        "P47_high_contrast_shared_film",
        (0.16, 0.72, 0.96),
        roughness=0.045,
        transmission=0.34,
        alpha=0.68,
        emission=(0.04, 0.32, 0.62),
        emission_strength=0.75,
    )
    center_z = top + 2.35
    # Two truncated spherical shells share one circular Plateau border; unlike
    # overlapping full spheres, this keeps the double-bubble geometry credible.
    _spherical_cap_x("P47_smaller_soap_bubble", 0.02, 0.0, center_z, 0.83, 2.05, soap_shell, direction=-1.0)
    _spherical_cap_x("P47_larger_soap_bubble", 0.02, 0.0, center_z, 0.83, 2.65, soap_shell, direction=1.0)
    membrane_base_x = 0.02
    membrane_radius = 0.83
    membrane_height = 0.58
    _spherical_cap_x(
        "P47_curved_shared_membrane",
        membrane_base_x,
        -0.035,
        center_z,
        membrane_radius,
        membrane_height,
        separator,
        direction=1.0,
    )
    add_torus(
        "P47_shared_plateau_border",
        (membrane_base_x, 0.0, center_z),
        membrane_radius,
        0.014,
        separator,
        rotation=(0.0, math.pi / 2, 0.0),
    )
    # Front-facing meridian highlight makes the physical separator curvature
    # legible in the locked camera while remaining part of the soap film.
    sphere_radius = (membrane_radius * membrane_radius + membrane_height * membrane_height) / (2.0 * membrane_height)
    sphere_center_x = membrane_base_x + membrane_height - sphere_radius
    profile = []
    for index in range(25):
        local_z = -membrane_radius + 2.0 * membrane_radius * index / 24
        x = sphere_center_x + math.sqrt(max(0.0, sphere_radius * sphere_radius - local_z * local_z))
        profile.append((x, -0.09, center_z + local_z))
    add_tube("P47_curved_membrane_meridian", profile, 0.028, separator, resolution=3)
    for name, x, width in (("small", -0.98, 0.18), ("large", 1.18, 0.22)):
        add_cone(f"P47_{name}_support_nozzle", (x, 0.0, top + 0.62), width, width * 0.55, 1.12, mats["steel"])
        add_cylinder(f"P47_{name}_nozzle_base", (x, 0.0, top + 0.13), 0.32, 0.22, mats["black"])
    _front_camera((0.18, 0.0, center_z), distance=12.6, height=center_z + 0.10, lens=60.0)


def scene_p48(mats) -> None:
    """Two fully detached airborne droplets immediately before contact."""
    droplet = material("P48_clean_water", (0.04, 0.30, 0.50), roughness=0.035, transmission=0.78, alpha=0.46)
    center_z = 2.55
    left_radius, right_radius = 0.48, 0.68
    gap = 0.14
    left_x = -(gap + left_radius + right_radius) / 2
    right_x = (gap + left_radius + right_radius) / 2
    add_sphere("P48_small_free_droplet", (left_x, 0.0, center_z), left_radius, droplet, segments=72)
    add_sphere("P48_large_free_droplet", (right_x, 0.0, center_z), right_radius, droplet, segments=72)
    add_box("P48_dark_strobe_backdrop", (0.0, 1.15, center_z), (6.5, 0.18, 4.3), mats["black"], bevel=0.0)
    add_camera((0.0, -8.4, center_z + 0.03), (0.0, 0.0, center_z), lens=76.0)


def scene_p49(mats) -> None:
    """Two same-material spheres held at equal depth in one glycerin column."""
    top = add_lab_table(mats, z=0.48, top=(8.4, 3.6, 0.18))
    glycerin = material("P49_clear_glycerin", (0.38, 0.27, 0.055), roughness=0.055, transmission=0.68, alpha=0.34)
    tank = add_glass_tank(mats, center=(0.0, 0.0, top + 2.45), size=(4.15, 1.75, 4.65), bottom_z=top + 0.05, fill=0.92, liquid=glycerin, name="P49_tall_glycerin_tank")
    release_z = tank["liquid_top"] - 0.50
    add_sphere("P49_small_steel_sphere", (-0.78, -0.18, release_z), 0.24, mats["steel"], segments=48)
    add_sphere("P49_large_steel_sphere", (0.78, -0.18, release_z), 0.38, mats["steel"], segments=56)
    add_box("P49_submerged_release_bar", (0.0, 0.27, release_z + 0.34), (2.55, 0.18, 0.14), mats["black"], bevel=0.025)
    for x in (-0.78, 0.78):
        cylinder_between(f"P49_release_pin_{x:+.2f}", (x, 0.18, release_z + 0.32), (x, -0.02, release_z + 0.12), 0.035, mats["black"])
    add_box("P49_release_post", (1.72, 0.52, tank["top"] + 0.35), (0.18, 0.18, 1.10), mats["steel"], bevel=0.02)
    _front_camera((0.0, 0.0, top + 2.45), distance=12.6, height=3.45, lens=68.0)


def scene_p50(mats) -> None:
    """Different-volume sessile droplets with one common contact angle."""
    top = add_lab_table(mats, z=0.55, top=(10.7, 3.6, 0.18))
    substrate = material("P50_homogeneous_substrate", (0.16, 0.19, 0.22), roughness=0.32, metallic=0.12)
    droplet = material("P50_same_liquid", (0.025, 0.31, 0.53), roughness=0.035, transmission=0.74, alpha=0.48)
    surface_z = top + 0.17
    add_box("P50_single_level_substrate", (0.0, 0.0, top + 0.08), (8.2, 2.20, 0.18), substrate, bevel=0.025)
    # Identical height/base-radius ratio gives every spherical cap the same Young angle.
    for index, (x, base_radius) in enumerate(((-2.65, 0.42), (-1.15, 0.58), (0.72, 0.76), (2.72, 0.96)), start=1):
        _spherical_cap_z(
            f"P50_equilibrium_droplet_{index}",
            x,
            -0.03,
            surface_z,
            base_radius,
            base_radius * 0.68,
            droplet,
        )
    add_box("P50_diffuse_backlight_panel", (0.0, 1.20, top + 1.42), (9.2, 0.15, 2.8), mats["white"], bevel=0.0)
    # Center the asymmetric four-drop extent and widen the field so both
    # contact lines of the largest right-hand cap retain a clean margin.
    add_camera(
        (0.30, -14.5, top + 0.95),
        (0.30, 0.0, top + 0.70),
        lens=60.0,
    )


BUILDERS: dict[str, tuple[Builder, bool]] = {
    "P36": (scene_p36, False),
    "P36b": (scene_p36b, False),
    "P37": (scene_p37, False),
    "P38": (scene_p38, False),
    "P39": (scene_p39, False),
    "P40": (scene_p40, False),
    "P41": (scene_p41, False),
    "P42": (scene_p42, False),
    "P43": (scene_p43, False),
    "P44": (scene_p44, False),
    "P45": (scene_p45, False),
    "P46": (scene_p46, True),
    "P47": (scene_p47, True),
    "P48": (scene_p48, True),
    "P49": (scene_p49, False),
    "P50": (scene_p50, True),
}
