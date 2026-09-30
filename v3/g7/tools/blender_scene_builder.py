"""Blender 3.6 scene builder for Group 7 physical VDM first frames.

Usage (on the render host):
  blender -b --python blender_scene_builder.py -- --task P7 --variant 1 --out /path/P7.png
"""
import argparse
import math
import os
import random

import bpy
from mathutils import Vector, Matrix

def mat(name, color, metallic=0.0, rough=.4, transmission=0.0):
    m=bpy.data.materials.new(name); m.diffuse_color=(*color,1); m.use_nodes=True
    bs=m.node_tree.nodes.get('Principled BSDF'); bs.inputs['Base Color'].default_value=(*color,1); bs.inputs['Metallic'].default_value=metallic; bs.inputs['Roughness'].default_value=rough
    if transmission:
        bs.inputs['Transmission'].default_value=transmission; bs.inputs['IOR'].default_value=1.45
        try:
            m.blend_method='BLEND'; m.use_screen_refraction=True; bs.inputs['Alpha'].default_value=.42
            m.diffuse_color=(*color,.42)
        except Exception: pass
    if name == 'laser red':
        try:
            bs.inputs['Emission'].default_value=(*color,1); bs.inputs['Emission Strength'].default_value=4.0
        except Exception: pass
    return m
WOOD=mat('warm maple',(0.55,.28,.10),0,.3); DARKWOOD=mat('dark wood',(.16,.08,.035),0,.36); METAL=mat('brushed aluminium',(.32,.38,.43),.8,.24); RING_METAL=mat('ring brass',(.72,.34,.06),.72,.22); BLACK=mat('black rubber',(.015,.018,.022),0,.5); RED=mat('laser red',(1,.015,.008),0,.18); BLUE=mat('water',(0.04,.25,.38),0,.12,.15); ICE=mat('ice',(.24,.72,.86),0,.24,.12); GLASS=mat('glass',(.55,.75,.82),0,.08,.7); RUBBER=mat('red rubber',(.55,.025,.015),0,.3); PEND_RED=mat('pendulum saturated red',(.86,.012,.018),.03,.24); PEND_BLUE=mat('pendulum saturated blue',(.012,.08,.90),.04,.22); WATER_POOL=mat('shallow water pool',(.08,.42,.58),0,.08,.32); FLOOR=mat('lab floor',(.055,.065,.08),0,.45)
# P12 uses separate saturated colours rather than one red mask.  Keeping the
# four materials explicit makes each incident segment independently
# segmentable in a generated video and in a pixel-level sanity check.
def ray_mat(name, color):
    m = mat(name, color, 0.0, .16)
    bs = m.node_tree.nodes.get('Principled BSDF')
    if bs is not None:
        try:
            bs.inputs['Emission'].default_value = (*color, 1)
            bs.inputs['Emission Strength'].default_value = 3.5
        except Exception:
            pass
    return m

def add_emission(material, color, strength):
    """Give a measurement aid a restrained, camera-stable luminance."""
    bs = material.node_tree.nodes.get('Principled BSDF')
    if bs is not None:
        try:
            bs.inputs['Emission'].default_value = (*color, 1)
            bs.inputs['Emission Strength'].default_value = strength
        except Exception:
            pass
    return material
RAY_CYAN=ray_mat('incident cyan',(0.0, .85, 1.0))
RAY_GREEN=ray_mat('incident green',(.25, 1.0, .03))
RAY_AMBER=ray_mat('incident amber',(1.0, .53, .015))
RAY_MAGENTA=ray_mat('incident magenta',(1.0, .02, .72))
# Small, high-contrast fiducials are visual aids only.  They are attached to
# the moving bodies and are deliberately much smaller than the bodies, so a
# tracker can estimate translation/rotation without changing the experiment.
MARKER_BLUE=mat('fiducial blue',(.02,.35,1.0),.15,.22)
MARKER_YELLOW=mat('fiducial yellow',(1.0,.65,.02),.05,.24)
NORMAL=mat('surface normal',(.72,.86,.92),.15,.28)
INTERFACE=mat('interface marker',(.72,.92,1.0),.05,.2)
TRAY_GLASS=mat('clear tray glass',(.70,.88,.95),0,.12,.45)
def cube(n, loc, scale, material, bevel=.03):
    bpy.ops.mesh.primitive_cube_add(location=loc); o=bpy.context.object; o.name=n; o.scale=scale; bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    if bevel:
        b=o.modifiers.new('soft edges','BEVEL'); b.width=bevel; b.segments=3
    o.data.materials.append(material); return o
def cyl(n, loc, radius, depth, material, rot=None, verts=48):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts,radius=radius,depth=depth,location=loc,rotation=rot or (0,0,0)); o=bpy.context.object; o.name=n; o.data.materials.append(material); b=o.modifiers.new('soft edges','BEVEL'); b.width=.02; b.segments=2; return o
def beaker_wall(n, x, radius, height, material, container_id):
    """Straight transparent beaker shell with open top and measurable walls."""
    # ``primitive_cylinder_add`` is capped by default.  An NGON cap would make
    # this look like a closed glass drum and could hide both the ice and the
    # liquid surface generated later, so explicitly build only the side wall.
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=96,
        radius=radius,
        depth=height,
        end_fill_type='NOTHING',
        location=(x,0,height/2+.02),
    )
    o=bpy.context.object; o.name=n; o.data.materials.append(material)
    solid=o.modifiers.new('thin glass wall','SOLIDIFY'); solid.thickness=.045
    bevel=o.modifiers.new('rounded glass edges','BEVEL'); bevel.width=.035; bevel.segments=3
    o['straight_walled']=True; o['open_top']=True; o['container_id']=container_id; o['role']='beaker_wall'; return o
def sphere(n, loc, radius, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=radius, location=loc); o=bpy.context.object; o.name=n; o.data.materials.append(material); bpy.ops.object.shade_smooth(); return o
def torus(n, loc, major, minor, material, rot=(0,0,0)):
    bpy.ops.mesh.primitive_torus_add(major_radius=major,minor_radius=minor,major_segments=64,minor_segments=20,location=loc,rotation=rot); o=bpy.context.object; o.name=n; o.data.materials.append(material); return o
def torus_quat(n, loc, major, minor, material, q):
    bpy.ops.mesh.primitive_torus_add(major_radius=major,minor_radius=minor,major_segments=64,minor_segments=20,location=loc); o=bpy.context.object; o.name=n; o.rotation_mode='QUATERNION'; o.rotation_quaternion=q; o.data.materials.append(material); return o
def beam(a,b,material,thick=.018,name='laser'):
    """Create a measurable straight segment between two 3-D points.

    ``name`` is optional for backwards compatibility with the other task
    builders.  P12 uses semantic names and custom properties so a Blender
    scene audit can distinguish incident beams from normals/interface aids
    without relying on material colour alone.
    """
    a,b=Vector(a),Vector(b); d=b-a; mid=(a+b)/2; o=cyl(name,mid,thick,d.length,material); o.rotation_mode='QUATERNION'; o.rotation_quaternion=d.to_track_quat('Z','Y'); return o
def look_at(o, target): o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler()
def ramp_basis(theta):
    """Return the local downhill tangent and outward surface normal.

    A board created along local +X and rotated about Y by ``theta`` has
    tangent ``(cos(theta), 0, -sin(theta))`` and normal
    ``(sin(theta), 0, cos(theta))``.  Keeping all object positions in this
    frame prevents gaps/penetration caused by rotating the board after
    placing an object in world coordinates.
    """
    s,c=math.sin(theta),math.cos(theta)
    return Vector((c,0,-s)), Vector((s,0,c))
def ramp_point(theta, center_z, u, y=0.0, normal_offset=0.0):
    t,n=ramp_basis(theta)
    return Vector((0,0,center_z)) + t*u + Vector((0,y,0)) + n*normal_offset
def ramp_surface(theta, center_z, u, y=0.0, half_thickness=.12):
    return ramp_point(theta, center_z, u, y, half_thickness)
def ramp_body_center(theta, center_z, u, y, normal_offset, half_thickness=.12):
    return ramp_surface(theta, center_z, u, y, half_thickness) + ramp_basis(theta)[1]*normal_offset
def box_normal_extent(theta, half_x, half_z):
    """Half extent of a box rotated by ``theta`` along the ramp normal.

    The P10 block/plunger are rotated by the same angle as the board, so their
    local Z axis is already the surface normal.  Returning ``half_z`` avoids
    double-projecting the local dimensions and guarantees exact contact.
    """
    return half_z
def wheel_quaternion(theta):
    """Orient a hoop in the ramp tangent-normal plane; axle is along Y."""
    t,n=ramp_basis(theta)
    axle=Vector((0,-1,0))
    # The columns are the world directions of the torus' local X/Y/Z axes.
    return Matrix((t,n,axle)).transposed().to_quaternion()
def spring_quaternion(theta):
    t,_=ramp_basis(theta)
    return Vector((0,0,1)).rotation_difference(t)
def ramp_leg(name, theta, center_z, u, half_x=.18, half_y=.8, material=None):
    """Vertical support whose top just reaches the underside of the board."""
    if material is None: material=METAL
    floor_top=-.03
    p=ramp_point(theta, center_z, u, 0.0, -.12)
    h=max(.10,p.z-floor_top)
    return cube(name,(p.x,0.0,floor_top+h/2),(half_x,half_y,h/2),material)
def ramp_crossbar(name, theta, center_z, u, material=None, width=1.05):
    if material is None: material=BLACK
    p=ramp_surface(theta,center_z,u,0.0,.012)
    o=cube(name,p,(.035,width,.012),material,.005); o.rotation_euler[1]=theta; return o
def ramp_finish_mark(name, theta, center_z, u, material=None, width=.42):
    """A flush transverse finish fiducial, not a physical obstacle."""
    if material is None: material=MARKER_YELLOW
    p=ramp_surface(theta,center_z,u,0.0,.014)
    o=cube(name,p,(.045,width,.010),material,.006); o.rotation_euler[1]=theta; return o
def attach_marker(n, loc, material, radius=.055):
    return sphere(n, loc, radius, material)
def body_fiducial(name, center, radius, theta, material, side=-1.0, parent=None):
    """Small attached dot for robust rotation/translation tracking."""
    _,n=ramp_basis(theta)
    v=(n*.78+Vector((0,side*.63,0))).normalized()
    m=sphere(name,Vector(center)+v*(radius+.012),.055,material)
    if parent is not None:
        m.parent=parent; m.matrix_parent_inverse=parent.matrix_world.inverted()
    return m
def surface_marker(name, parent, local_offset, material, radius=.055):
    """Make a small coloured dot part of a moving body's local geometry."""
    p=Vector(parent.location)+Vector(local_offset)
    m=sphere(name,p,radius,material); m.parent=parent; m.location=Vector(local_offset)
    return m
def make_open_tray(x):
    """Identical low-rim trays; the ice silhouettes stay unobstructed."""
    base_top=.30
    cube('tray_base',(x,0,.18),(1.70,1.25,.12),METAL,.08)
    wall_h=.16; wall_t=.045; z=base_top+wall_h/2
    cube('tray_left_wall',(x-1.63,0,z),(wall_t,1.20,wall_h/2),TRAY_GLASS,.02)
    cube('tray_right_wall',(x+1.63,0,z),(wall_t,1.20,wall_h/2),TRAY_GLASS,.02)
    cube('tray_front_wall',(x,-1.18,z),(1.60,wall_t,wall_h/2),TRAY_GLASS,.02)
    cube('tray_back_wall',(x,1.18,z),(1.60,wall_t,wall_h/2),TRAY_GLASS,.02)
def setup(task, variant, out):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    global WOOD,DARKWOOD,METAL,RING_METAL,BLACK,RED,BLUE,GLASS,ICE,RUBBER,PEND_RED,PEND_BLUE,WATER_POOL,FLOOR,MARKER_BLUE,MARKER_YELLOW,NORMAL,INTERFACE,TRAY_GLASS,RAY_CYAN,RAY_GREEN,RAY_AMBER,RAY_MAGENTA
    WOOD=mat('warm maple',(.55,.28,.10),0,.3); DARKWOOD=mat('dark wood',(.16,.08,.035),0,.36); METAL=mat('brushed aluminium',(.32,.38,.43),.8,.24); RING_METAL=mat('ring brass',(.72,.34,.06),.72,.22); BLACK=mat('black rubber',(.015,.018,.022),0,.5); RED=mat('laser red',(1,.015,.008),0,.18); BLUE=mat('water',(.04,.25,.38),0,.12,.15); ICE=mat('ice',(.65,.86,.94),0,.20,.12); GLASS=mat('glass',(.55,.75,.82),0,.08,.7); RUBBER=mat('red rubber',(.55,.025,.015),0,.3); PEND_RED=mat('pendulum saturated red',(.86,.012,.018),.03,.24); PEND_BLUE=mat('pendulum saturated blue',(.012,.08,.90),.04,.22); WATER_POOL=mat('shallow water pool',(.08,.42,.58),0,.08,.32); FLOOR=mat('lab floor',(.055,.065,.08),0,.45)
    RAY_CYAN=ray_mat('incident cyan',(0.0, .85, 1.0)); RAY_GREEN=ray_mat('incident green',(.25, 1.0, .03)); RAY_AMBER=ray_mat('incident amber',(1.0, .53, .015)); RAY_MAGENTA=ray_mat('incident magenta',(1.0, .02, .72))
    MARKER_BLUE=mat('fiducial blue',(.02,.35,1.0),.15,.22); MARKER_YELLOW=mat('fiducial yellow',(1.0,.65,.02),.05,.24); NORMAL=mat('surface normal',(.72,.86,.92),.15,.28); INTERFACE=mat('interface marker',(.72,.92,1.0),.05,.2); TRAY_GLASS=mat('clear tray glass',(.70,.88,.95),0,.12,.45)
    # The optics scene is intentionally dark.  Restrained emission keeps the
    # dashed normals and, especially, the single water-air boundary visible
    # without turning either into a second laser beam.
    add_emission(NORMAL,(.72,.86,.92),.45); add_emission(INTERFACE,(.72,.92,1.0),.85)
    scene=bpy.context.scene; scene.render.engine='BLENDER_EEVEE'; scene.eevee.use_gtao=True; scene.eevee.gtao_distance=3; scene.eevee.gtao_factor=1.35; scene.render.resolution_x=1536; scene.render.resolution_y=1024; scene.render.resolution_percentage=100; scene.render.image_settings.file_format='PNG'; scene.render.filepath=out
    world=scene.world or bpy.data.worlds.new('World'); scene.world=world; world.use_nodes=True; world.node_tree.nodes['Background'].inputs['Color'].default_value=(.025,.035,.05,1) if task=='P12' else (.12,.14,.17,1); world.node_tree.nodes['Background'].inputs['Strength'].default_value=.25
    cube('floor',(0,0,-.18),(8,6,.15),FLOOR,0)
    bpy.ops.object.light_add(type='AREA', location=(2,-3,6)); key=bpy.context.object; key.data.energy=1100; key.data.shape='DISK'; key.data.size=5; look_at(key,(0,0,0))
    bpy.ops.object.light_add(type='AREA', location=(-4,2,3)); fill=bpy.context.object; fill.data.energy=700; fill.data.size=4; look_at(fill,(0,0,0))
    bpy.ops.object.camera_add(location=(7,-10,5.2)); cam=bpy.context.object; cam.data.lens=52; scene.camera=cam
    if task=='P7':
        # The ramp rises toward +X and rolls downhill toward -X.  Positions
        # are computed from the rotated surface so both bodies truly contact
        # the board and start at the same local height.
        theta=-math.radians(14); ramp_z=1.30
        # Use a wider board and separated lanes so the sphere and hoop remain
        # distinct in the locked oblique side view used by the video tracker.
        ramp=cube('incline',(0,0,ramp_z),(4.8,1.45,.12),WOOD); ramp.rotation_euler[1]=theta
        ramp['incline_deg']=14.0; ramp['downhill_axis']='negative_X'; ramp['outer_radius']=.42
        ramp_leg('low_leg',theta,ramp_z,-3.35,.18,.82); ramp_leg('high_leg',theta,ramp_z,3.35,.18,.82)
        for y in (-1.36,1.36):
            rail_p=ramp_surface(theta,ramp_z,0,y,.075)
            rail=cube('side_rail',rail_p,(4.72,.035,.075),BLACK,.012); rail.rotation_euler[1]=theta
        start_u=2.05; gate_u=2.52
        # Two narrow release tabs leave both bodies completely visible in the
        # locked-off view; a broad plate here would hide the hoop and spoil
        # pixel-level tracking of its marker.
        for y in (-.82,.82):
            gate_p=ramp_surface(theta,ramp_z,gate_u,y,.40)
            gate=cube('release_tab',gate_p,(.055,.075,.40),BLACK,.012); gate.rotation_euler[1]=theta; gate['release_u']=gate_u
        # Put the hoop in the near lane so its full circular silhouette and
        # fiducial remain unobstructed in the locked-off view.
        sphere_c=ramp_body_center(theta,ramp_z,start_u,.82,.42)
        solid=sphere('solid_sphere',sphere_c,.42,METAL); solid['outer_radius']=.42; solid['start_u']=start_u; solid['rolling_axis']='Y'
        body_fiducial('sphere_fiducial',sphere_c,.42,theta,MARKER_BLUE,-1,solid)
        # Correct ring orientation: hoop in the ramp tangent-normal plane,
        # axle along Y across the ramp width, so it rolls downhill along X.
        ring_c=ramp_body_center(theta,ramp_z,start_u,-.82,.42)
        ring=torus_quat('thin_ring',ring_c,.34,.08,RING_METAL,wheel_quaternion(theta)); ring['outer_radius']=.42; ring['start_u']=start_u; ring['rolling_axis']='Y'
        t,n=ramp_basis(theta); ring_marker_world=ring_c+t*.29+n*.29+Vector((0,-.085,0))
        ring_marker=sphere('ring_fiducial',ring_marker_world,.065,MARKER_YELLOW); ring_marker.parent=ring; ring_marker.matrix_parent_inverse=ring.matrix_world.inverted()
        ramp_finish_mark('finish_fiducial',theta,ramp_z,-2.65,MARKER_YELLOW,.42)
        cam.location=(-7.4,-13.8,4.8); cam.data.lens=48; look_at(cam,(0,0,1.45))
    elif task=='P8c':
        bar_z=3.9; cube('pendulum_bar',(0,0,bar_z),(4.3,.3,.14),METAL)
        # The benchmark revision uses 15° and 30° release amplitudes.  Keep
        # the two pivot locations and equal string length fixed so frame-0
        # geometry remains directly measurable by the video evaluator.
        # Keep pivot-to-bob length exactly equal and make the release angles
        # explicit in world geometry.  A front orthographic camera removes
        # perspective scaling, so pixel distances and angles stay measurable.
        for idx,(x,ang,side,bob_mat) in enumerate([(-1.65,15,-1,PEND_RED),(1.65,30,1,PEND_BLUE)]):
            L=2.35; a=math.radians(ang); p=(x+side*L*math.sin(a),0,bar_z-.12-L*math.cos(a))
            pendulum_id='red_left_15deg' if idx==0 else 'blue_right_30deg'
            string=beam((x,0,bar_z-.12),p,METAL,.022,f'pendulum_string_{pendulum_id}')
            string['string_length']=L; string['release_angle_deg']=ang; string['pendulum_id']=pendulum_id
            bob=sphere(f'pendulum_bob_{pendulum_id}',p,.25,bob_mat); bob['string_length']=L; bob['release_angle_deg']=ang; bob['pendulum_id']=pendulum_id
            # Do not add a contrasting dot or a vertical guide: the saturated
            # red/blue bodies already provide identity and an uncluttered
            # spherical silhouette gives the most stable centre measurement.
        cam.location=(0,-15.0,2.25); cam.data.type='ORTHO'; cam.data.ortho_scale=7.05; look_at(cam,(0,0,2.25))
    elif task=='P10':
        theta=-math.radians(30); ramp_z=2.05
        ramp=cube('rough_ramp',(0,0,ramp_z),(3.9,1.05,.12),DARKWOOD); ramp.rotation_euler[1]=theta
        ramp['incline_deg']=30.0; ramp['mu_k']=.20; ramp['motion_axis']='ramp_tangent'
        ramp_leg('low_leg',theta,ramp_z,-2.75,.20,.80); ramp_leg('high_leg',theta,ramp_z,2.75,.20,.80)
        for y in (-.96,.96):
            rail_p=ramp_surface(theta,ramp_z,0,y,.07); rail=cube('side_rail',rail_p,(3.78,.035,.07),BLACK,.012); rail.rotation_euler[1]=theta
        block_u=-1.90; block_clearance=box_normal_extent(theta,.42,.35); block_c=ramp_body_center(theta,ramp_z,block_u,0,block_clearance)
        block=cube('block',block_c,(.42,.42,.35),RUBBER); block.rotation_euler[1]=theta; block['mu_k']=.20; block['start_u']=block_u
        body_fiducial('block_fiducial',block_c,.47,theta,MARKER_YELLOW,-1,block)
        # A compact spring launcher sits downhill of the block and points up
        # the ramp (+u), making the intended initial impulse legible.
        plunger_u=-2.48; plunger_clearance=box_normal_extent(theta,.22,.30); plunger_p=ramp_body_center(theta,ramp_z,plunger_u,0,plunger_clearance)
        plunger=cube('plunger',plunger_p,(.22,.5,.30),BLACK); plunger.rotation_euler[1]=theta
        for u in (-2.48,-2.33,-2.18,-2.03):
            p=ramp_surface(theta,ramp_z,u,0,.23); torus_quat('launcher_spring',p,.10,.025,METAL,spring_quaternion(theta))
        ramp_finish_mark('stop_reference',theta,ramp_z,1.85,MARKER_YELLOW,.42)
        cam.location=(0,-15.5,5.0); cam.data.lens=50; look_at(cam,(0,0,1.90))
    elif task=='P12':
        # P12's first-frame contract is incident-only: four coplanar rays end
        # exactly on one flat interface.  The outgoing Snell/TIR segments are
        # intentionally absent so an image-to-video model has to generate them.
        # The ray plane is just inside the front tank face (all four beams and
        # every normal share this y coordinate); this avoids glass refraction
        # while preserving subtle tank/glass highlights in the render.
        water_top=2.70; ray_y=-1.08
        cube('tank_bottom',(0,0,1.15),(3.5,1.2,.08),GLASS); cube('tank_back',(0,1.15,2.6),(3.5,.06,1.45),GLASS); cube('tank_left',(-3.45,0,2.6),(.06,1.2,1.45),GLASS); cube('tank_right',(3.45,0,2.6),(.06,1.2,1.45),GLASS); cube('water',(0,0,1.95),(3.35,1.05,.75),BLUE,0)
        interface=beam((-3.2,ray_y,water_top),(3.2,ray_y,water_top),INTERFACE,.018,'water_air_interface')
        interface['role']='interface'; interface['plane_y']=ray_y; interface['z']=water_top; interface['normal_axis']='Z'; interface['water_index']=1.333

        def dashed_normal(idx, x):
            # Four short dashes leave a small, measurable gap at the interface
            # while the endpoint fiducial marks the exact shared intersection.
            for part,(za,zb) in enumerate(((water_top-.46,water_top-.28),(water_top-.16,water_top-.045),(water_top+.045,water_top+.16),(water_top+.28,water_top+.46)),1):
                normal=beam((x,ray_y,za),(x,ray_y,zb),NORMAL,.010,f'surface_normal_{idx:02d}_{part:02d}')
                normal['role']='surface_normal'; normal['ray_id']=idx; normal['plane_y']=ray_y; normal['interface_z']=water_top

        incident_geometry=[]

        def add_incident(idx, material, hit_x, theta_i, source_z, medium_from, source_side):
            """Add one ray that terminates exactly at the shared interface.

            ``source_side`` is -1 for an upper-left air source and +1 for the
            lower-right water source.  Angles are measured from the vertical
            surface normal, so |dx|/|dz| = tan(theta_i) by construction.
            """
            dz=abs(source_z-water_top)
            source_x=hit_x+source_side*math.tan(math.radians(theta_i))*dz
            source=(source_x,ray_y,source_z); hit=(hit_x,ray_y,water_top)
            measured=math.degrees(math.atan2(abs(source_x-hit_x),dz))
            if abs(measured-theta_i)>1e-7:
                raise RuntimeError(f'P12 ray {idx}: requested {theta_i} deg, built {measured} deg')
            if not (-3.35 <= source_x <= 3.35):
                raise RuntimeError(f'P12 ray {idx}: source x={source_x:.4f} leaves the tank interior')
            ray=beam(source,hit,material,.026,f'incident_ray_{idx:02d}')
            ray['role']='incident'; ray['medium_from']=medium_from; ray['ray_id']=idx; ray['incidence_angle_deg']=theta_i; ray['plane_y']=ray_y; ray['interface_z']=water_top; ray['incident_only']=True
            dashed_normal(idx,hit_x)
            marker=sphere(f'interface_endpoint_{idx:02d}',hit,.042 if idx<4 else .045,INTERFACE); marker['role']='ray_endpoint'; marker['ray_id']=idx; marker['plane_y']=ray_y; marker['interface_z']=water_top
            incident_geometry.append((idx,source,hit,theta_i,medium_from))

        # Exact 20/40/60-degree air incidents.  Their four surface hit points
        # are separated by at least 1.1 scene units for unambiguous pairing.
        source_z=4.12
        air_defs=((1,RAY_CYAN,-2.55,20.0),(2,RAY_GREEN,-.85,40.0),(3,RAY_AMBER,.75,60.0))
        for idx,material,hit_x,theta_i in air_defs:
            add_incident(idx,material,hit_x,theta_i,source_z,'air',-1.0)

        # The 49-degree magenta ray originates at the lower right and travels
        # up-left to the interface.  It is slightly above water's 48.75-degree
        # critical angle, but this first frame contains no outgoing/TIR branch.
        critical_deg=49.0; hit_x=1.85; under_z=1.48
        add_incident(4,RAY_MAGENTA,hit_x,critical_deg,under_z,'water',1.0)
        bpy.data.objects['incident_ray_04']['water_air_critical_reference_deg']=48.75

        # Scene-level invariants make accidental future geometry drift fail
        # loudly during a Blender build instead of entering a rendered batch.
        if len(incident_geometry)!=4:
            raise RuntimeError('P12 must contain exactly four incident rays')
        hit_points=[item[2] for item in incident_geometry]
        if any(abs(hit[1]-ray_y)>1e-9 or abs(hit[2]-water_top)>1e-9 for hit in hit_points):
            raise RuntimeError('P12 incident endpoints must share one plane and one flat interface')
        if min(abs(hit_points[i][0]-hit_points[j][0]) for i in range(4) for j in range(i)) < 1.0:
            raise RuntimeError('P12 incident endpoints are not sufficiently separated')

        # Exact front-on orthographic view: no parallax, camera tilt, pan, or
        # depth-dependent perspective can distort the measured angles.
        cam.location=(0,-15.0,2.65); cam.data.type='ORTHO'; cam.data.ortho_scale=7.25; look_at(cam,(0,0,2.65))
    elif task=='P27':
        # Refined task: two identical dry, straight-walled beakers.  No liquid
        # plane exists in frame zero.  The low ice fill leaves room for the
        # video model to expose a rising water surface without losing the rim.
        left_x,right_x=-1.85,1.85; beaker_r=1.12; beaker_h=3.25; inner_floor=.20
        for container_id,x in (('left',left_x),('right',right_x)):
            wall=beaker_wall(f'straight_glass_beaker_{container_id}',x,beaker_r,beaker_h,TRAY_GLASS,container_id)
            base=cyl(f'thick_beaker_base_{container_id}',(x,0,.08),beaker_r,.16,TRAY_GLASS); base['container_id']=container_id; base['role']='beaker_base'
            rim=torus(f'open_beaker_rim_{container_id}',(x,0,beaker_h+.02),beaker_r,.055,TRAY_GLASS)
            rim['container_id']=container_id; rim['role']='open_rim'
        half=(.67,.62,.38); whole_volume=(2*half[0])*(2*half[1])*(2*half[2])
        whole=cube('compact_ice_block',(left_x,0,inner_floor+half[2]),half,ICE,.10); whole['ice_volume']=whole_volume; whole['mass_reference']='equal_to_crushed_total'; whole['container']='left'; whole['ice_form']='one_compact_block'
        random.seed(2701); raw=[]; total_raw=0.0
        for i in range(46):
            r=random.uniform(.10,.18); raw.append(r); total_raw += 4.0/3.0*math.pi*r**3
        scale=(whole_volume/total_raw)**(1.0/3.0)
        crushed_volume=0.0
        for i,r in enumerate(raw):
            r*=scale
            # Irregular shapes aid segmentation, but normalise the three axis
            # scales to determinant 1 so deformation does not silently change
            # the equal-mass volume established above.
            shape=[random.uniform(.75,1.20),random.uniform(.70,1.16),random.uniform(.68,1.12)]
            determinant=shape[0]*shape[1]*shape[2]
            shape=[value/(determinant**(1.0/3.0)) for value in shape]
            max_horizontal_extent=r*max(shape[0],shape[1]); rr=beaker_r-max_horizontal_extent-.10
            angle=random.uniform(0,2*math.pi); radial=rr*math.sqrt(random.random())
            x=right_x+radial*math.cos(angle); y=radial*math.sin(angle); z=inner_floor+r*shape[2]+random.uniform(0,.32)
            chunk=sphere(f'crushed_ice_{i:02d}',(x,y,z),r,ICE); chunk.scale=tuple(shape)
            volume=4.0/3.0*math.pi*r**3; crushed_volume+=volume
            chunk['volume']=volume; chunk['container']='right'; chunk['ice_form']='crushed_piece'; chunk['volume_preserving_shape_scale']=True
        if abs(crushed_volume/whole_volume-1.0)>1e-7:
            raise RuntimeError(f'P27 equal-mass geometry drifted: crushed/block volume={crushed_volume/whole_volume}')
        whole['crushed_total_volume']=crushed_volume; whole['equal_mass_geometry_validated']=True
        cam.location=(0,-15.0,1.72); cam.data.type='ORTHO'; cam.data.ortho_scale=6.15; look_at(cam,(0,0,1.72))
    # Keep the three delivered variants physically identical while providing
    # small, reproducible camera/light alternatives for first-frame selection.
    if variant == 2:
        # P12 variants alter illumination only; moving the camera would break
        # its exact common-plane/orthographic measurement contract.
        if task not in ('P12','P8c','P27'):
            cam.location.x += .28; cam.location.y += .10
            look_at(cam,(0,0,1.0 if task != 'P12' else 2.4))
        key.data.energy *= 1.08
    elif variant == 3:
        if task not in ('P12','P8c','P27'):
            cam.location.x -= .22; cam.location.y -= .08
            look_at(cam,(0,0,1.0 if task != 'P12' else 2.4))
        fill.data.energy *= 1.12
    scene.view_settings.look='Medium High Contrast'; bpy.ops.wm.save_as_mainfile(filepath=out.replace('.png','.blend')); bpy.ops.render.render(write_still=True)
if __name__=='__main__':
    import os
    # Environment variables are convenient on remote shells where Blender's
    # own argument parser can consume the double-dash separator.
    if os.environ.get('VDM_TASK') and os.environ.get('VDM_OUT'):
        setup(os.environ['VDM_TASK'],int(os.environ.get('VDM_VARIANT','1')),os.environ['VDM_OUT'])
    else:
        a=argparse.ArgumentParser(); a.add_argument('--task',required=True); a.add_argument('--variant',type=int,default=1); a.add_argument('--out',required=True); args=a.parse_args(); setup(args.task,args.variant,args.out)
