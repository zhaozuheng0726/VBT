"""Render one OpenVDB smoke frame with stock Blender Cycles."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import bpy
from mathutils import Vector


MATERIAL_PRESETS = {
    "legacy": {
        "density_scale": 8.0,
        "volume_color": (0.62, 0.66, 0.72),
        "anisotropy": 0.2,
        "world_color": (0.055, 0.065, 0.08),
        "world_strength": 0.5,
        "light_energy": 5000.0,
    },
    "paper_gray": {
        "density_scale": 8.0,
        "volume_color": (0.68, 0.72, 0.78),
        "anisotropy": 0.2,
        "world_color": (0.80, 0.81, 0.82),
        "world_strength": 1.0,
        "light_energy": 5000.0,
    },
    "charcoal": {
        "density_scale": 12.0,
        "volume_color": (0.18, 0.20, 0.24),
        "anisotropy": 0.12,
        "world_color": (0.80, 0.81, 0.82),
        "world_strength": 1.0,
        "light_energy": 5000.0,
    },
    "fire": {
        "density_scale": 3.0,
        "volume_color": (0.12, 0.13, 0.15),
        "anisotropy": 0.1,
        "world_color": (0.008, 0.009, 0.012),
        "world_strength": 0.12,
        "light_energy": 700.0,
    },
}


def _argv_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _configure_cycles(device: str) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    if device == "CPU":
        scene.cycles.device = "CPU"
        return

    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = device
        prefs.get_devices()
        enabled = []
        for item in prefs.devices:
            item.use = item.type == device
            if item.use:
                enabled.append(item.name)
        if not enabled:
            raise RuntimeError(f"no {device} Cycles device found")
        scene.cycles.device = "GPU"
        print("CYCLES_DEVICE", device, enabled, flush=True)
    except Exception as exc:
        print("CYCLES_DEVICE_FALLBACK", str(exc), flush=True)
        scene.cycles.device = "CPU"


def _make_volume_material(
    density_scale: float,
    volume_color: tuple[float, float, float],
    anisotropy: float,
    material_preset: str,
    include_fire_emission: bool = True,
) -> bpy.types.Material:
    material = bpy.data.materials.new("VBT_Proxy_Smoke")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    info = nodes.new("ShaderNodeVolumeInfo")
    multiply = nodes.new("ShaderNodeMath")
    multiply.operation = "MULTIPLY"
    multiply.inputs[1].default_value = density_scale
    principled = nodes.new("ShaderNodeVolumePrincipled")
    principled.inputs["Color"].default_value = (*volume_color, 1.0)
    principled.inputs["Anisotropy"].default_value = anisotropy
    output = nodes.new("ShaderNodeOutputMaterial")

    info.location = (-500, 0)
    multiply.location = (-280, 0)
    principled.location = (-40, 0)
    output.location = (240, 0)
    links.new(info.outputs["Density"], multiply.inputs[0])
    links.new(multiply.outputs["Value"], principled.inputs["Density"])
    if material_preset == "fire" and include_fire_emission:
        flame_attribute = nodes.new("ShaderNodeAttribute")
        flame_attribute.attribute_name = "flames"
        flame_attribute.location = (-720, -260)
        flame_scale = nodes.new("ShaderNodeMath")
        flame_scale.operation = "MULTIPLY"
        flame_scale.inputs[1].default_value = 0.18
        flame_scale.location = (-500, -220)
        emission_strength = nodes.new("ShaderNodeMath")
        emission_strength.operation = "MULTIPLY"
        emission_strength.inputs[1].default_value = 0.9
        emission_strength.location = (-260, -220)
        fire_color = nodes.new("ShaderNodeValToRGB")
        fire_color.location = (-250, -390)
        ramp = fire_color.color_ramp
        ramp.elements.remove(ramp.elements[1])
        ramp.elements[0].position = 0.0
        ramp.elements[0].color = (0.35, 0.005, 0.0, 1.0)
        for position, color in (
            (0.12, (1.0, 0.03, 0.0, 1.0)),
            (0.35, (1.0, 0.22, 0.005, 1.0)),
            (0.62, (1.0, 0.48, 0.02, 1.0)),
            (1.0, (1.0, 0.82, 0.28, 1.0)),
        ):
            element = ramp.elements.new(position)
            element.color = color
        links.new(flame_attribute.outputs["Fac"], flame_scale.inputs[0])
        links.new(flame_attribute.outputs["Fac"], emission_strength.inputs[0])
        links.new(flame_scale.outputs["Value"], fire_color.inputs["Fac"])
        links.new(emission_strength.outputs["Value"], principled.inputs["Emission Strength"])
        links.new(fire_color.outputs["Color"], principled.inputs["Emission Color"])
    links.new(principled.outputs["Volume"], output.inputs["Volume"])
    return material


def _volume_bounds(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    bpy.context.view_layer.update()
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minimum = Vector(tuple(min(p[i] for p in corners) for i in range(3)))
    maximum = Vector(tuple(max(p[i] for p in corners) for i in range(3)))
    return minimum, maximum


def _reference_bounds(path: Path) -> tuple[Vector, Vector]:
    volume = bpy.data.volumes.new("VBT_Camera_Reference")
    volume.filepath = str(path.resolve())
    volume.grids.load()
    obj = bpy.data.objects.new("VBT_Camera_Reference", volume)
    bpy.context.collection.objects.link(obj)
    bounds = _volume_bounds(obj)
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.volumes.remove(volume)
    return bounds


def render(args: argparse.Namespace) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    scene = bpy.context.scene
    _configure_cycles(args.device)
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.volume_bounces = 1
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(args.output.resolve())

    preset = MATERIAL_PRESETS[args.material_preset]
    density_scale = args.density_scale
    if density_scale is None:
        density_scale = preset["density_scale"]

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (*preset["world_color"], 1.0)
    background.inputs["Strength"].default_value = preset["world_strength"]

    volume = bpy.data.volumes.new("VBT_Proxy_Volume")
    volume.filepath = str(args.input_vdb.resolve())
    volume.grids.load()
    volume.render.space = "WORLD"
    volume.render.step_size = args.step_size
    obj = bpy.data.objects.new("VBT_Proxy_Volume", volume)
    bpy.context.collection.objects.link(obj)
    volume.materials.append(
        _make_volume_material(
            density_scale,
            preset["volume_color"],
            preset["anisotropy"],
            args.material_preset,
            args.flame_vdb is None,
        )
    )

    flame_volume = None
    if args.flame_vdb:
        flame_volume = bpy.data.volumes.new("VBT_Proxy_Flame_Volume")
        flame_volume.filepath = str(args.flame_vdb.resolve())
        flame_volume.grids.load()
        flame_volume.render.space = "WORLD"
        flame_volume.render.step_size = args.step_size
        flame_obj = bpy.data.objects.new("VBT_Proxy_Flame_Volume", flame_volume)
        bpy.context.collection.objects.link(flame_obj)
        flame_volume.materials.append(
            _make_volume_material(
                0.0,
                preset["volume_color"],
                preset["anisotropy"],
                "fire",
                True,
            )
        )

    volume_minimum, volume_maximum = _volume_bounds(obj)
    if args.camera_reference_vdb:
        minimum, maximum = _reference_bounds(args.camera_reference_vdb)
    else:
        minimum, maximum = volume_minimum, volume_maximum
    center = (minimum + maximum) * 0.5
    size = maximum - minimum
    radius = max(size) * 0.5

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + Vector((radius * 1.45, -radius * 2.75, radius * 0.25))
    _look_at(camera, center + Vector((0.0, 0.0, radius * 0.05)))
    camera_data.lens = 52.0
    camera_data.clip_end = radius * 20.0
    scene.camera = camera

    light_data = bpy.data.lights.new("Key_Area", type="AREA")
    light_data.energy = preset["light_energy"]
    light_data.size = radius * 2.2
    light = bpy.data.objects.new("Key_Area", light_data)
    bpy.context.collection.objects.link(light)
    light.location = center + Vector((radius * 1.3, -radius * 1.8, radius * 1.4))
    _look_at(light, center)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output_blend:
        args.output_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend.resolve()))

    print(
        "VDB_RENDER",
        {
            "grids": [grid.name for grid in volume.grids],
            "flame_grids": [grid.name for grid in flame_volume.grids] if flame_volume else [],
            "bbox_min": tuple(round(v, 4) for v in minimum),
            "bbox_max": tuple(round(v, 4) for v in maximum),
            "volume_bbox_min": tuple(round(v, 4) for v in volume_minimum),
            "volume_bbox_max": tuple(round(v, 4) for v in volume_maximum),
            "camera_reference": str(args.camera_reference_vdb.resolve()) if args.camera_reference_vdb else None,
            "flame_vdb": str(args.flame_vdb.resolve()) if args.flame_vdb else None,
            "material_preset": args.material_preset,
            "density_scale": density_scale,
            "step_size": args.step_size,
            "samples": args.samples,
            "resolution": (args.width, args.height),
            "output": str(args.output.resolve()),
        },
        flush=True,
    )
    bpy.ops.render.render(write_still=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-vdb", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-blend", type=Path)
    parser.add_argument("--camera-reference-vdb", type=Path)
    parser.add_argument("--flame-vdb", type=Path)
    parser.add_argument("--device", choices=("CPU", "CUDA", "OPTIX"), default="CUDA")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--density-scale", type=float)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument(
        "--material-preset",
        choices=tuple(MATERIAL_PRESETS),
        default="legacy",
    )
    args = parser.parse_args(_argv_after_double_dash())
    if not args.input_vdb.is_file():
        raise FileNotFoundError(args.input_vdb)
    if args.camera_reference_vdb and not args.camera_reference_vdb.is_file():
        raise FileNotFoundError(args.camera_reference_vdb)
    if args.flame_vdb and not args.flame_vdb.is_file():
        raise FileNotFoundError(args.flame_vdb)
    render(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
