from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def project_root_from_here() -> Path | None:
    env_root = os.getenv("VBT_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "VBT").exists() and (parent / "3D").exists():
            return parent
        if parent.name == "VBT" and (parent.parent / "3D").exists():
            return parent.parent
    return None


def default_3d_path(*parts: str) -> str:
    root = project_root_from_here()
    if root is None:
        return str(Path(*parts))
    return str(root / "3D" / Path(*parts))


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Headless smoke test for the VBT Blender bridge addon.")
    parser.add_argument("--addon-dir", required=True)
    parser.add_argument("--input-vbt", required=True)
    parser.add_argument("--metadata", default="")
    parser.add_argument("--converter", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--reload-frame", type=int, default=None)
    parser.add_argument("--auto-frame", type=int, default=None)
    parser.add_argument(
        "--direct-preview-exe",
        default=default_3d_path("vdb_tools", "build", "Release", "render_temporal_vbt_direct_preview.exe"),
    )
    parser.add_argument("--direct-preview-output", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--showcase-script",
        default=default_3d_path("blender_render_smoke_showcase.py"),
        help="Optional existing smoke showcase script used for a visible validation render.",
    )
    parser.add_argument("--resolution-x", type=int, default=512)
    parser.add_argument("--resolution-y", type=int, default=288)
    parser.add_argument("--samples", type=int, default=24)
    return parser.parse_args(argv)


def look_at(obj, target: Vector):
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def make_volume_material(obj):
    mat = bpy.data.materials.new("VBT_Bridge_Test_Volume")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (520, 0)
    volume_info = nt.nodes.new("ShaderNodeVolumeInfo")
    volume_info.location = (-520, 0)
    density_mul = nt.nodes.new("ShaderNodeMath")
    density_mul.operation = "MULTIPLY"
    density_mul.location = (-260, 0)
    density_mul.inputs[1].default_value = 13.0
    volume = nt.nodes.new("ShaderNodeVolumePrincipled")
    volume.location = (0, 0)
    volume.inputs["Color"].default_value = (0.68, 0.72, 0.78, 1.0)
    volume.inputs["Anisotropy"].default_value = 0.2
    nt.links.new(volume_info.outputs["Density"], density_mul.inputs[0])
    nt.links.new(density_mul.outputs["Value"], volume.inputs["Density"])
    nt.links.new(volume.outputs["Volume"], out.inputs["Volume"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def bbox_world(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_v = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    max_v = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    center = (min_v + max_v) * 0.5
    radius = max((v - center).length for v in corners)
    return center, max(radius, 1.0)


def setup_scene(resolution_x: int, resolution_y: int, samples: int):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = False
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = -0.25
    scene.view_settings.gamma = 1.0

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.05, 0.06, 0.07, 1.0)
        bg.inputs["Strength"].default_value = 0.6
    return scene


def load_showcase_module(path: str):
    script = Path(path)
    if not script.exists():
        return None
    spec = importlib.util.spec_from_file_location("vbt_bridge_smoke_showcase", script)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_showcase_render_setup(obj, showcase):
    preset = dict(showcase.CASE_PRESETS["industrial_chimney"])
    showcase.configure_world(preset["world"])
    center, radius = showcase.bbox_world(obj)
    showcase.add_camera(center, radius, preset)
    if preset["ground"]:
        showcase.add_ground(center, radius, preset["world"] != "sky_day")
    showcase.add_light_rig(center, radius, preset["light_rig"])
    showcase.apply_volume_material(obj, "industrial_chimney", preset)


def main():
    args = parse_args()
    sys.path.insert(0, args.addon_dir)
    import vbt_blender_bridge

    setup_scene(args.resolution_x, args.resolution_y, args.samples)
    vbt_blender_bridge.register()
    bpy.ops.vbt.import_frame(
        filepath=args.input_vbt,
        metadata_path=args.metadata,
        converter_path=args.converter,
        cache_dir=args.cache_dir,
        frame=args.frame,
    )
    obj = bpy.context.object
    if args.reload_frame is not None:
        bpy.ops.vbt.reload_selected_frame(frame=args.reload_frame)
        obj = bpy.context.object
    if args.auto_frame is not None:
        bpy.ops.vbt.toggle_auto_reload(enabled=True)
        bpy.context.scene.frame_set(args.auto_frame)
        obj = bpy.context.object
    if args.direct_preview_output:
        bpy.ops.vbt.direct_preview_selected(
            preview_exe=args.direct_preview_exe,
            output_dir=str(Path(args.direct_preview_output).parent),
            frame=int(obj.get("vbt_frame", args.frame)),
            width=256,
            height=144,
            steps=96,
        )
        generated = Path(obj.get("vbt_direct_preview", ""))
        expected = Path(args.direct_preview_output)
        if generated and generated != expected:
            expected.parent.mkdir(parents=True, exist_ok=True)
            expected.write_bytes(generated.read_bytes())
    showcase = load_showcase_module(args.showcase_script)
    if showcase is not None:
        apply_showcase_render_setup(obj, showcase)
    else:
        make_volume_material(obj)
        center, radius = bbox_world(obj)
        camera_data = bpy.data.cameras.new("Camera")
        camera = bpy.data.objects.new("Camera", camera_data)
        bpy.context.collection.objects.link(camera)
        camera.location = center + Vector((0.35 * radius, -3.2 * radius, 0.8 * radius))
        look_at(camera, center)
        camera.data.lens = 55.0
        bpy.context.scene.camera = camera

        light_data = bpy.data.lights.new("Key_Area", "AREA")
        light = bpy.data.objects.new("Key_Area", light_data)
        bpy.context.collection.objects.link(light)
        light.location = center + Vector((-1.4 * radius, -2.0 * radius, 2.0 * radius))
        look_at(light, center)
        light.data.energy = 450.0
        light.data.size = radius * 1.2

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = args.output
    bpy.ops.render.render(write_still=True)
    print(f"[vbt-bridge-test] rendered {args.output}")
    print(f"[vbt-bridge-test] proxy {obj.get('vbt_proxy_vdb')}")


if __name__ == "__main__":
    main()
