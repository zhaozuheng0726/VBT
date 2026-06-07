"""Create a minimal Blender scene for the native VBT Cycles prototype.

Run from a Blender build that contains the native VBT patch:

    blender --background --python cycles_native_vbt/create_native_vbt_scene.py -- \
      --input-vbt outputs/render_sequence_20260407/industrial_chimney_mainline.vbtp \
      --frame 100 \
      --output-blend build/native_vbt_test.blend

The script intentionally does not call the OpenVDB proxy bridge. It creates a
Volume datablock whose filepath points directly at the `.vbtp` file. The patched
Cycles sync path detects that extension and uploads the VBT resident blob.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import bpy


def _argv_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _make_volume_material(density_scale: float) -> bpy.types.Material:
    material = bpy.data.materials.new("Native_VBT_Density")
    material.use_nodes = True
    tree = material.node_tree
    nodes = tree.nodes
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (320, 0)

    try:
        principled = nodes.new("ShaderNodeVolumePrincipled")
        principled.location = (0, 0)
        if "Density" in principled.inputs:
            principled.inputs["Density"].default_value = density_scale
        if "Density Attribute" in principled.inputs:
            principled.inputs["Density Attribute"].default_value = "density"
        tree.links.new(principled.outputs["Volume"], output.inputs["Volume"])
    except Exception:
        info = nodes.new("ShaderNodeVolumeInfo")
        multiply = nodes.new("ShaderNodeMath")
        multiply.operation = "MULTIPLY"
        multiply.inputs[1].default_value = density_scale
        scatter = nodes.new("ShaderNodeVolumeScatter")
        info.location = (-420, 0)
        multiply.location = (-180, 0)
        scatter.location = (40, 0)
        tree.links.new(info.outputs["Density"], multiply.inputs[0])
        tree.links.new(multiply.outputs[0], scatter.inputs["Density"])
        tree.links.new(scatter.outputs["Volume"], output.inputs["Volume"])

    return material


def create_scene(input_vbt: Path, frame: int, density_scale: float, output_blend: Path | None) -> None:
    _clear_scene()

    scene = bpy.context.scene
    scene.frame_set(frame)
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720

    volume_data = bpy.data.volumes.new("Native_VBT_Volume")
    volume_data.filepath = str(input_vbt.resolve())
    volume_data.render.space = "OBJECT"
    volume_data.render.step_size = 0.002

    volume_object = bpy.data.objects.new("Native_VBT_Volume", volume_data)
    bpy.context.collection.objects.link(volume_object)
    volume_object.location = (-0.5, -0.5, -0.5)
    volume_object.scale = (3.0, 3.0, 3.0)
    volume_object.data.materials.append(_make_volume_material(density_scale))

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (2.4, -4.0, 2.1)
    camera.rotation_euler = (1.1, 0.0, 0.48)
    camera_data.lens = 45
    scene.camera = camera

    light_data = bpy.data.lights.new("Key_Area", type="AREA")
    light = bpy.data.objects.new("Key_Area", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (1.5, -2.5, 3.5)
    light_data.energy = 450
    light_data.size = 4.0

    if output_blend:
        output_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(output_blend.resolve()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-vbt", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--density-scale", type=float, default=1.0)
    parser.add_argument("--output-blend", type=Path, default=None)
    args = parser.parse_args(_argv_after_double_dash())

    if not args.input_vbt.exists():
        raise FileNotFoundError(args.input_vbt)

    create_scene(args.input_vbt, args.frame, args.density_scale, args.output_blend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
