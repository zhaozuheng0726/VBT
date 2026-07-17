import argparse
import math
import os
from mathutils import Vector

import bpy


def parse_args():
    argv = []
    if "--" in os.sys.argv:
        argv = os.sys.argv[os.sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description="Render water-like original/recon frames in Blender.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-abc")
    source.add_argument("--input-original")
    parser.add_argument("--input-recon", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--orig-out", required=True)
    parser.add_argument("--recon-out", required=True)
    parser.add_argument("--resolution", type=int, default=1400)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--transmission-weight", type=float, default=1.0)
    parser.add_argument("--water-roughness", type=float, default=0.008)
    parser.add_argument("--absorption-density", type=float, default=0.0)
    parser.add_argument("--background-preset", choices=["studio", "grass"], default="studio")
    parser.add_argument("--camera-preset", choices=["hero", "side"], default="hero")
    parser.add_argument("--skip-original-render", action="store_true")
    return parser.parse_args(argv)


def reset_scene(background_preset):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 192
    scene.cycles.use_adaptive_sampling = True
    scene.render.resolution_x = 1400
    scene.render.resolution_y = 1400
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.exposure = 1.25
    scene.view_settings.gamma = 1.0
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes["Background"]
    if background_preset == "grass":
        bg.inputs[0].default_value = (0.50, 0.66, 0.88, 1.0)
        bg.inputs[1].default_value = 0.55
    else:
        bg.inputs[0].default_value = (0.055, 0.065, 0.085, 1.0)
        bg.inputs[1].default_value = 1.1
    return scene


def choose_largest_mesh():
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects found")
    return max(meshes, key=lambda o: len(o.data.vertices) if o.data else 0)


def bbox_world(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    bmin = Vector((min(p.x for p in corners), min(p.y for p in corners), min(p.z for p in corners)))
    bmax = Vector((max(p.x for p in corners), max(p.y for p in corners), max(p.z for p in corners)))
    center = (bmin + bmax) * 0.5
    size = bmax - bmin
    radius = max(size.x, size.y, size.z) * 0.5
    return bmin, bmax, center, radius


def add_camera_and_lights(center, radius, background_preset, camera_preset):
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    if camera_preset == "side":
        cam.location = center + Vector((radius * 3.15, -radius * 0.18, radius * 0.32))
        cam.data.lens = 78
    else:
        cam.location = center + Vector((radius * 2.6, -radius * 0.65, radius * 0.55))
        cam.data.lens = 68
    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    bpy.ops.mesh.primitive_plane_add(size=radius * 8.0, location=(center.x, center.y, center.z - radius * 1.15))
    plane = bpy.context.object
    plane.name = "Ground"
    ground_mat = bpy.data.materials.new("GroundMat")
    ground_mat.use_nodes = True
    nt = ground_mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    if background_preset == "grass":
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.location = (-520, 40)
        tex.inputs["Scale"].default_value = 22.0
        tex.inputs["Detail"].default_value = 6.0
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.location = (-300, 40)
        ramp.color_ramp.elements[0].color = (0.06, 0.16, 0.04, 1.0)
        ramp.color_ramp.elements[1].color = (0.20, 0.36, 0.10, 1.0)
        nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        bsdf.inputs["Roughness"].default_value = 0.88
        bsdf.inputs["Specular IOR Level"].default_value = 0.18
    else:
        bsdf.inputs["Base Color"].default_value = (0.16, 0.17, 0.19, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.14
        bsdf.inputs["Specular IOR Level"].default_value = 0.55
    plane.data.materials.append(ground_mat)

    key_data = bpy.data.lights.new(name="Key", type="AREA")
    key_data.energy = 135000.0 if background_preset == "grass" else 165000.0
    key_data.shape = "RECTANGLE"
    key_data.size = radius * 2.4
    key_data.size_y = radius * 1.2
    key = bpy.data.objects.new("Key", key_data)
    bpy.context.scene.collection.objects.link(key)
    key.location = center + Vector((radius * 1.6, -radius * 2.0, radius * 2.7))
    key.rotation_euler = (math.radians(56), 0.0, math.radians(34))

    rim_data = bpy.data.lights.new(name="Rim", type="AREA")
    rim_data.energy = 60000.0 if background_preset == "grass" else 70000.0
    rim_data.shape = "RECTANGLE"
    rim_data.size = radius * 2.2
    rim_data.size_y = radius * 0.8
    rim = bpy.data.objects.new("Rim", rim_data)
    bpy.context.scene.collection.objects.link(rim)
    rim.location = center + Vector((-radius * 1.8, radius * 1.3, radius * 1.9))
    rim.rotation_euler = (math.radians(35), 0.0, math.radians(-140))

    fill_data = bpy.data.lights.new(name="Fill", type="AREA")
    fill_data.energy = 52000.0 if background_preset == "grass" else 42000.0
    fill_data.shape = "RECTANGLE"
    fill_data.size = radius * 2.6
    fill_data.size_y = radius * 1.6
    fill = bpy.data.objects.new("Fill", fill_data)
    bpy.context.scene.collection.objects.link(fill)
    fill.location = center + Vector((radius * 0.2, radius * 2.0, radius * 1.25))
    fill.rotation_euler = (math.radians(68), 0.0, math.radians(180))

    sun_data = bpy.data.lights.new(name="Sun", type="SUN")
    sun_data.energy = 3.8 if background_preset == "grass" else 2.8
    sun = bpy.data.objects.new("Sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.location = center + Vector((radius * 3.0, -radius * 3.0, radius * 5.0))
    sun.rotation_euler = (math.radians(50), math.radians(10), math.radians(28))


def make_water_material(name, transmission_weight, water_roughness, absorption_density):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for node in list(nt.nodes):
        if node.name not in {"Material Output"}:
            nt.nodes.remove(node)
    out = nt.nodes["Material Output"]
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (-220, 0)
    bsdf.inputs["Base Color"].default_value = (0.985, 0.99, 1.0, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = water_roughness
    bsdf.inputs["IOR"].default_value = 1.333
    bsdf.inputs["Transmission Weight"].default_value = transmission_weight
    bsdf.inputs["Specular IOR Level"].default_value = 0.95
    bsdf.inputs["Coat Weight"].default_value = 0.1
    bsdf.inputs["Coat Roughness"].default_value = 0.01
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    if absorption_density > 0.0:
        absorption = nt.nodes.new("ShaderNodeVolumeAbsorption")
        absorption.location = (-220, -180)
        absorption.inputs["Color"].default_value = (0.93, 0.98, 1.0, 1.0)
        absorption.inputs["Density"].default_value = absorption_density
        nt.links.new(absorption.outputs["Volume"], out.inputs["Volume"])
    return mat


def apply_water_material(obj, transmission_weight, water_roughness, absorption_density):
    mat = make_water_material(f"{obj.name}_Water", transmission_weight, water_roughness, absorption_density)
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    if hasattr(obj.data, "polygons"):
        for poly in obj.data.polygons:
            poly.use_smooth = True


def ensure_object_mode():
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def render_mesh_object(
    obj,
    output_path,
    resolution,
    samples,
    transmission_weight,
    water_roughness,
    absorption_density,
    background_preset,
    camera_preset,
    camera_center=None,
    camera_radius=None,
):
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.cycles.samples = samples
    if camera_center is None or camera_radius is None:
        _, _, center, radius = bbox_world(obj)
    else:
        center, radius = camera_center, camera_radius
    add_camera_and_lights(center, radius, background_preset, camera_preset)
    apply_water_material(obj, transmission_weight, water_roughness, absorption_density)
    scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def render_original_abc(
    input_abc,
    frame,
    output_path,
    resolution,
    samples,
    transmission_weight,
    water_roughness,
    absorption_density,
    background_preset,
    camera_preset,
    skip_render=False,
):
    reset_scene(background_preset)
    bpy.ops.wm.alembic_import(filepath=input_abc)
    scene = bpy.context.scene
    scene.frame_set(frame)
    obj = choose_largest_mesh()
    _, _, center, radius = bbox_world(obj)
    if not skip_render:
        render_mesh_object(
            obj,
            output_path,
            resolution,
            samples,
            transmission_weight,
            water_roughness,
            absorption_density,
            background_preset,
            camera_preset,
            center,
            radius,
        )
    return center, radius


def render_original_asset(
    input_path,
    output_path,
    resolution,
    samples,
    transmission_weight,
    water_roughness,
    absorption_density,
    background_preset,
    camera_preset,
    skip_render=False,
):
    reset_scene(background_preset)
    bpy.ops.wm.obj_import(filepath=input_path)
    obj = choose_largest_mesh()
    _, _, center, radius = bbox_world(obj)
    if not skip_render:
        render_mesh_object(
            obj,
            output_path,
            resolution,
            samples,
            transmission_weight,
            water_roughness,
            absorption_density,
            background_preset,
            camera_preset,
            center,
            radius,
        )
    return center, radius


def convert_volume_to_mesh(volume_obj):
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    volume_obj.select_set(True)
    bpy.context.view_layer.objects.active = volume_obj
    result = bpy.ops.object.volume_to_mesh(resolution_mode='VOXEL_SIZE', voxel_size=0.6, threshold=0.08, adaptivity=0.0)
    if "FINISHED" not in result:
        raise RuntimeError(f"volume_to_mesh failed: {result}")
    mesh_objs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not mesh_objs:
        mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError("No mesh object produced by volume_to_mesh")
    return max(mesh_objs, key=lambda o: len(o.data.vertices) if o.data else 0)


def render_recon_asset(
    input_path,
    output_path,
    resolution,
    samples,
    transmission_weight,
    water_roughness,
    absorption_density,
    background_preset,
    camera_preset,
    camera_center=None,
    camera_radius=None,
):
    reset_scene(background_preset)
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=input_path)
        mesh_obj = choose_largest_mesh()
    elif ext == ".vdb":
        bpy.ops.object.volume_import(filepath=input_path)
        volume_obj = bpy.context.selected_objects[0]
        mesh_obj = convert_volume_to_mesh(volume_obj)
    else:
        raise RuntimeError(f"Unsupported recon asset format: {input_path}")
    render_mesh_object(
        mesh_obj,
        output_path,
        resolution,
        samples,
        transmission_weight,
        water_roughness,
        absorption_density,
        background_preset,
        camera_preset,
        camera_center,
        camera_radius,
    )


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.orig_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.recon_out), exist_ok=True)
    common = (
        args.orig_out,
        args.resolution,
        args.samples,
        args.transmission_weight,
        args.water_roughness,
        args.absorption_density,
        args.background_preset,
        args.camera_preset,
        args.skip_original_render,
    )
    if args.input_abc:
        center, radius = render_original_abc(args.input_abc, args.frame, *common)
    else:
        center, radius = render_original_asset(args.input_original, *common)
    render_recon_asset(
        args.input_recon,
        args.recon_out,
        args.resolution,
        args.samples,
        args.transmission_weight,
        args.water_roughness,
        args.absorption_density,
        args.background_preset,
        args.camera_preset,
        center,
        radius,
    )


if __name__ == "__main__":
    main()
