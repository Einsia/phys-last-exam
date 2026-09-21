#!/usr/bin/env python3
"""Shared Blender primitives for the Physical-Bench full first-frame study.

The renderer deliberately builds only observable initial states.  It avoids
captions, arrows, trajectories and outcome hints so that each image can be
used directly as the first frame of a video-generation task.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import bpy
from mathutils import Vector


WIDTH = 1344
HEIGHT = 768


def reset_scene(engine: str = "cycles", *, samples: int = 36) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    if engine == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.max_bounces = 7
        scene.cycles.transparent_max_bounces = 6
        scene.cycles.device = "CPU"
        cycles_addon = bpy.context.preferences.addons.get("cycles")
        if cycles_addon is not None:
            try:
                prefs = cycles_addon.preferences
                prefs.compute_device_type = "OPTIX"
                prefs.get_devices()
                selected = False
                for device in prefs.devices:
                    use = device.type == "OPTIX" and not selected
                    device.use = use
                    selected = selected or use
                if selected:
                    scene.cycles.device = "GPU"
            except Exception:
                pass
    elif engine == "workbench":
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
        scene.display.shading.show_shadows = True
        scene.display.shading.show_cavity = True
        scene.display.shading.cavity_type = "BOTH"
        scene.display.shading.background_type = "WORLD"
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"

    scene.render.resolution_x = WIDTH
    scene.render.resolution_y = HEIGHT
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 12
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.fps = 30
    if scene.world is None:
        scene.world = bpy.data.worlds.new("laboratory_world")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.028, 0.036, 0.045, 1.0)
    bg.inputs["Strength"].default_value = 0.32
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except (TypeError, ValueError):
        try:
            scene.view_settings.look = "Medium High Contrast"
        except (TypeError, ValueError):
            pass
    scene.view_settings.exposure = 0.8


def material(
    name: str,
    color: tuple[float, float, float],
    *,
    roughness: float = 0.45,
    metallic: float = 0.0,
    transmission: float = 0.0,
    alpha: float = 1.0,
    emission: tuple[float, float, float] | None = None,
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if "Transmission Weight" in bsdf.inputs:
        bsdf.inputs["Transmission Weight"].default_value = transmission
    elif "Transmission" in bsdf.inputs:
        bsdf.inputs["Transmission"].default_value = transmission
    if "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = alpha
    if emission is not None:
        key = "Emission Color" if "Emission Color" in bsdf.inputs else "Emission"
        bsdf.inputs[key].default_value = (*emission, 1.0)
        if "Emission Strength" in bsdf.inputs:
            bsdf.inputs["Emission Strength"].default_value = emission_strength
    mat.diffuse_color = (*color, alpha)
    if alpha < 1.0:
        try:
            mat.surface_render_method = "DITHERED"
        except Exception:
            pass
    return mat


def wood_material(name: str, base: tuple[float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.58
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 5.5
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.72
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (0.45, 7.0, 2.0)
    texcoord = nodes.new("ShaderNodeTexCoord")
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*tuple(max(0.0, c * 0.52) for c in base), 1.0)
    ramp.color_ramp.elements[1].color = (*tuple(min(1.0, c * 1.28) for c in base), 1.0)
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def palette() -> dict[str, bpy.types.Material]:
    return {
        "white": material("warm_off_white", (0.75, 0.78, 0.79), roughness=0.75),
        "dark": material("charcoal", (0.025, 0.032, 0.04), roughness=0.48),
        "black": material("black_anodized", (0.015, 0.02, 0.025), roughness=0.27, metallic=0.58),
        "steel": material("brushed_steel", (0.38, 0.42, 0.46), roughness=0.22, metallic=0.88),
        "aluminum": material("aluminum", (0.62, 0.65, 0.67), roughness=0.27, metallic=0.9),
        "copper": material("copper", (0.62, 0.22, 0.08), roughness=0.25, metallic=0.86),
        "brass": material("brass", (0.67, 0.42, 0.08), roughness=0.28, metallic=0.82),
        "wood": wood_material("unfinished_maple", (0.6, 0.34, 0.15)),
        "wood_dark": wood_material("dark_walnut", (0.28, 0.11, 0.045)),
        "red": material("red_rubber", (0.62, 0.018, 0.012), roughness=0.5),
        "blue": material("blue_polymer", (0.015, 0.16, 0.55), roughness=0.42),
        "green": material("green_polymer", (0.015, 0.38, 0.16), roughness=0.43),
        "yellow": material("yellow_polymer", (0.82, 0.53, 0.015), roughness=0.44),
        "orange": material("orange_polymer", (0.9, 0.2, 0.012), roughness=0.43),
        "rubber": material("dark_rubber", (0.025, 0.027, 0.03), roughness=0.72),
        "sand": material("dry_sand", (0.55, 0.35, 0.14), roughness=0.94),
        "ice": material("clear_ice", (0.5, 0.76, 0.86), roughness=0.12, transmission=0.72, alpha=0.7),
        "water": material("clear_water", (0.035, 0.22, 0.32), roughness=0.06, transmission=0.68, alpha=0.38),
        "oil": material("clear_oil", (0.46, 0.34, 0.05), roughness=0.08, transmission=0.55, alpha=0.42),
        "glass": material("laboratory_glass", (0.38, 0.56, 0.62), roughness=0.045, transmission=0.82, alpha=0.2),
        "wax": material("ivory_wax", (0.88, 0.78, 0.52), roughness=0.7),
        "flame": material("flame_emission", (1.0, 0.22, 0.01), roughness=0.1, emission=(1.0, 0.08, 0.0), emission_strength=9.0),
        "laser": material("red_laser", (0.8, 0.002, 0.001), roughness=0.1, emission=(1.0, 0.0, 0.0), emission_strength=16.0),
        "magnet_red": material("magnet_red", (0.58, 0.015, 0.01), roughness=0.35, metallic=0.35),
        "magnet_blue": material("magnet_blue", (0.015, 0.08, 0.5), roughness=0.35, metallic=0.35),
        "wire": material("enameled_copper", (0.5, 0.07, 0.025), roughness=0.3, metallic=0.72),
        "soap": material("soap_film", (0.24, 0.48, 0.74), roughness=0.04, transmission=0.82, alpha=0.22),
    }


def assign(obj: bpy.types.Object, mat: bpy.types.Material) -> bpy.types.Object:
    obj.data.materials.append(mat)
    return obj


def add_box(
    name: str,
    location: Sequence[float],
    dimensions: Sequence[float],
    mat: bpy.types.Material,
    *,
    rotation: Sequence[float] = (0.0, 0.0, 0.0),
    bevel: float = 0.035,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    assign(obj, mat)
    if bevel > 0:
        mod = obj.modifiers.new("real_edge", "BEVEL")
        mod.width = bevel
        mod.segments = 3
    return obj


def add_sphere(name: str, location: Sequence[float], radius: float, mat: bpy.types.Material, *, segments: int = 48) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=max(24, segments // 2), radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    assign(obj, mat)
    bpy.ops.object.shade_smooth()
    return obj


def add_cylinder(
    name: str,
    location: Sequence[float],
    radius: float,
    depth: float,
    mat: bpy.types.Material,
    *,
    rotation: Sequence[float] = (0.0, 0.0, 0.0),
    vertices: int = 64,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    assign(obj, mat)
    mod = obj.modifiers.new("machined_edge", "BEVEL")
    mod.width = min(radius, depth) * 0.055
    mod.segments = 3
    return obj


def add_cone(
    name: str,
    location: Sequence[float],
    radius1: float,
    radius2: float,
    depth: float,
    mat: bpy.types.Material,
    *,
    rotation: Sequence[float] = (0.0, 0.0, 0.0),
    vertices: int = 64,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cone_add(
        vertices=vertices,
        radius1=radius1,
        radius2=radius2,
        depth=depth,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    assign(obj, mat)
    return obj


def add_torus(
    name: str,
    location: Sequence[float],
    major_radius: float,
    minor_radius: float,
    mat: bpy.types.Material,
    *,
    rotation: Sequence[float] = (math.pi / 2, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_segments=72,
        minor_segments=18,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    assign(obj, mat)
    bpy.ops.object.shade_smooth()
    return obj


def cylinder_between(
    name: str,
    start: Sequence[float],
    end: Sequence[float],
    radius: float,
    mat: bpy.types.Material,
) -> bpy.types.Object:
    p0, p1 = Vector(start), Vector(end)
    direction = p1 - p0
    obj = add_cylinder(name, tuple((p0 + p1) / 2.0), radius, max(direction.length, 1e-5), mat)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = direction.to_track_quat("Z", "Y")
    return obj


def add_tube(
    name: str,
    points: Iterable[Sequence[float]],
    radius: float,
    mat: bpy.types.Material,
    *,
    cyclic: bool = False,
    resolution: int = 3,
) -> bpy.types.Object:
    pts = [tuple(p) for p in points]
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = resolution
    curve.bevel_depth = radius
    curve.bevel_resolution = 4
    spline = curve.splines.new("POLY")
    spline.points.add(len(pts) - 1)
    for point, co in zip(spline.points, pts):
        point.co = (*co, 1.0)
    spline.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    assign(obj, mat)
    return obj


def add_disc(name: str, location: Sequence[float], radius: float, depth: float, mat: bpy.types.Material, *, rotation=(math.pi / 2, 0.0, 0.0)) -> bpy.types.Object:
    return add_cylinder(name, location, radius, depth, mat, rotation=rotation)


def add_mesh(
    name: str,
    vertices: list[Sequence[float]],
    faces: list[Sequence[int]],
    mat: bpy.types.Material,
    *,
    smooth: bool = False,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    assign(obj, mat)
    if smooth:
        for poly in mesh.polygons:
            poly.use_smooth = True
    return obj


def look_at(obj: bpy.types.Object, target: Sequence[float]) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_camera(location: Sequence[float], target: Sequence[float], *, lens: float = 52.0) -> bpy.types.Object:
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = "locked_camera"
    camera.data.lens = lens
    camera.data.sensor_width = 36.0
    look_at(camera, target)
    bpy.context.scene.camera = camera
    return camera


def area_light(
    name: str,
    location: Sequence[float],
    target: Sequence[float],
    energy: float,
    size: float,
    color: tuple[float, float, float] = (1.0, 0.97, 0.92),
) -> bpy.types.Object:
    bpy.ops.object.light_add(type="AREA", location=location)
    lamp = bpy.context.object
    lamp.name = name
    lamp.data.energy = energy
    lamp.data.shape = "DISK"
    lamp.data.size = size
    lamp.data.color = color
    look_at(lamp, target)
    return lamp


def point_light(name: str, location: Sequence[float], energy: float, *, color=(1.0, 0.72, 0.42), radius: float = 0.12) -> bpy.types.Object:
    bpy.ops.object.light_add(type="POINT", location=location)
    lamp = bpy.context.object
    lamp.name = name
    lamp.data.energy = energy
    lamp.data.color = color
    lamp.data.shadow_soft_size = radius
    return lamp


def lab_shell(mats: dict[str, bpy.types.Material], *, dark: bool = False, floor_z: float = 0.0) -> None:
    floor_mat = material("floor_surface", (0.035, 0.045, 0.055) if dark else (0.34, 0.36, 0.37), roughness=0.65)
    wall_mat = mats["dark"] if dark else mats["white"]
    add_box("lab_floor", (0, 0.8, floor_z - 0.15), (14.0, 8.0, 0.3), floor_mat, bevel=0.0)
    add_box("rear_wall", (0, 3.5, 3.2), (14.0, 0.18, 6.6), wall_mat, bevel=0.0)
    area_light("key", (-3.6, -3.2, 7.8), (0, 0, 1.4), 1100 if not dark else 560, 4.4)
    area_light("fill", (4.3, -0.4, 5.5), (0, 0, 1.4), 760 if not dark else 300, 3.6, (0.78, 0.89, 1.0))


def add_lab_table(mats: dict[str, bpy.types.Material], *, z: float = 0.78, top=(10.8, 3.8, 0.18), wood: bool = False) -> float:
    top_mat = mats["wood_dark"] if wood else mats["black"]
    add_box("lab_table", (0, 0, z), top, top_mat, bevel=0.045)
    for x in (-4.35, 4.35):
        for y in (-1.15, 1.15):
            add_cylinder(f"table_leg_{x}_{y}", (x, y, z / 2), 0.085, z, mats["steel"])
    return z + top[2] / 2


def add_glass_tank(
    mats: dict[str, bpy.types.Material],
    *,
    center=(0.0, 0.0, 1.8),
    size=(5.0, 2.0, 2.4),
    bottom_z: float | None = None,
    fill: float = 0.7,
    liquid: bpy.types.Material | None = None,
    name: str = "tank",
) -> dict[str, float]:
    cx, cy, cz = center
    sx, sy, sz = size
    if bottom_z is None:
        bottom_z = cz - sz / 2
    else:
        cz = bottom_z + sz / 2
    t = 0.055
    liquid = liquid or mats["water"]
    water_top = bottom_z + sz * fill
    add_box(f"{name}_liquid", (cx, cy, (bottom_z + water_top) / 2), (sx - 2*t, sy - 2*t, water_top - bottom_z), liquid, bevel=0.01)
    add_box(f"{name}_bottom", (cx, cy, bottom_z - t/2), (sx, sy, t), mats["glass"], bevel=0.008)
    add_box(f"{name}_left", (cx - sx/2, cy, cz), (t, sy, sz), mats["glass"], bevel=0.008)
    add_box(f"{name}_right", (cx + sx/2, cy, cz), (t, sy, sz), mats["glass"], bevel=0.008)
    add_box(f"{name}_front", (cx, cy - sy/2, cz), (sx, t, sz), mats["glass"], bevel=0.006)
    add_box(f"{name}_back", (cx, cy + sy/2, cz), (sx, t, sz), mats["glass"], bevel=0.006)
    return {"bottom": bottom_z, "top": bottom_z + sz, "liquid_top": water_top, "cx": cx, "cy": cy, "sx": sx, "sy": sy, "sz": sz}


def add_beaker(
    mats: dict[str, bpy.types.Material],
    *,
    location=(0.0, 0.0, 1.25),
    radius: float = 0.7,
    height: float = 1.6,
    fill: float = 0.65,
    liquid: bpy.types.Material | None = None,
    name: str = "beaker",
) -> dict[str, float]:
    x, y, z = location
    liquid = liquid or mats["water"]
    add_cylinder(f"{name}_glass", (x, y, z + height/2), radius, height, mats["glass"], vertices=72)
    add_cylinder(f"{name}_liquid", (x, y, z + height*fill/2 + 0.04), radius*0.91, height*fill, liquid, vertices=72)
    add_torus(f"{name}_rim", (x, y, z + height), radius*0.98, 0.035, mats["glass"], rotation=(0,0,0))
    return {"bottom": z, "top": z + height, "liquid_top": z + height*fill, "radius": radius}


def add_bar_magnet(
    mats: dict[str, bpy.types.Material],
    *,
    location=(0.0, 0.0, 1.2),
    length: float = 2.2,
    width: float = 0.65,
    height: float = 0.42,
    rotation=(0.0, 0.0, 0.0),
    name: str = "bar_magnet",
) -> tuple[bpy.types.Object, bpy.types.Object]:
    x, y, z = location
    # The two halves touch exactly; color is the only pole cue.
    a = add_box(f"{name}_north", (x-length/4, y, z), (length/2, width, height), mats["magnet_red"], rotation=rotation, bevel=0.05)
    b = add_box(f"{name}_south", (x+length/4, y, z), (length/2, width, height), mats["magnet_blue"], rotation=rotation, bevel=0.05)
    return a, b


def add_compass(mats: dict[str, bpy.types.Material], *, location=(0.0, 0.0, 1.0), angle: float = 0.0, radius: float = 0.28, name: str = "compass") -> None:
    x, y, z = location
    add_cylinder(f"{name}_case", (x, y, z), radius, 0.07, mats["brass"], rotation=(0,0,0))
    d = Vector((math.cos(angle), math.sin(angle), 0.0)) * radius * 0.72
    cylinder_between(f"{name}_red_needle", (x, y, z+0.055), (x+d.x, y+d.y, z+0.055), 0.025, mats["red"])
    cylinder_between(f"{name}_dark_needle", (x, y, z+0.055), (x-d.x, y-d.y, z+0.055), 0.025, mats["black"])


def add_pendulum(
    mats: dict[str, bpy.types.Material],
    *,
    pivot=(0.0, 0.0, 4.0),
    length: float = 2.4,
    angle: float = 0.0,
    plane_y: float = 0.0,
    bob_radius: float = 0.25,
    name: str = "pendulum",
) -> Vector:
    px, _, pz = pivot
    bob = Vector((px + length*math.sin(angle), plane_y, pz - length*math.cos(angle)))
    cylinder_between(f"{name}_string", (px, plane_y, pz), tuple(bob), 0.018, mats["black"])
    add_sphere(f"{name}_bob", tuple(bob), bob_radius, mats["brass"])
    add_cylinder(f"{name}_pivot", (px, plane_y, pz), 0.09, 0.3, mats["steel"], rotation=(math.pi/2,0,0))
    return bob


def add_candle(mats: dict[str, bpy.types.Material], *, location=(0.0, 0.0, 1.0), height=1.3, radius=0.22, name="candle") -> None:
    x, y, z = location
    add_cylinder(f"{name}_wax", (x, y, z+height/2), radius, height, mats["wax"])
    cylinder_between(f"{name}_wick", (x,y,z+height), (x,y,z+height+0.16), 0.018, mats["black"])
    add_cone(f"{name}_flame", (x,y,z+height+0.35), 0.14, 0.015, 0.45, mats["flame"])
    point_light(f"{name}_light", (x,y,z+height+0.33), 150, radius=0.22)


def add_coil(mats: dict[str, bpy.types.Material], *, center=(0.0,0.0,1.5), radius=0.75, length=1.6, turns=18, axis="x", name="coil") -> None:
    cx, cy, cz = center
    points=[]
    for i in range(turns*14+1):
        t = i/(turns*14)*turns*2*math.pi
        axial = -length/2 + length*i/(turns*14)
        if axis == "x":
            points.append((cx+axial, cy+radius*math.cos(t), cz+radius*math.sin(t)))
        elif axis == "y":
            points.append((cx+radius*math.cos(t), cy+axial, cz+radius*math.sin(t)))
        else:
            points.append((cx+radius*math.cos(t), cy+radius*math.sin(t), cz+axial))
    add_tube(name, points, 0.035, mats["wire"])


def add_drop(mats: dict[str, bpy.types.Material], *, location=(0.0,0.0,1.0), radius=0.45, flatten=0.65, name="droplet", liquid=None) -> bpy.types.Object:
    obj = add_sphere(name, location, radius, liquid or mats["water"])
    obj.scale.z = flatten
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def ramp_point(center: Sequence[float], angle: float, u: float, v: float, normal_offset: float) -> Vector:
    c = Vector(center)
    tangent = Vector((math.cos(angle), 0.0, -math.sin(angle)))
    normal = Vector((math.sin(angle), 0.0, math.cos(angle)))
    return c + tangent*u + Vector((0.0, v, 0.0)) + normal*normal_offset


def finish_default_camera(*, dark: bool = False) -> None:
    if bpy.context.scene.camera is None:
        add_camera((7.8, -11.8, 5.6), (0, 0, 1.65), lens=54)
    if dark:
        bpy.context.scene.view_settings.exposure = 1.1
    bpy.context.view_layer.update()
