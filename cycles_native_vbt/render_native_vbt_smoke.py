"""Render a low-cost native VBT smoke-test scene from Blender.

This script assumes the .blend already contains a Volume datablock pointing at
a .vbtp file, as created by create_native_vbt_scene.py.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct
import sys
import warnings

import bpy
from mathutils import Vector

warnings.filterwarnings("ignore", category=DeprecationWarning)


INDUSTRIAL_CHIMNEY_PRESET = {
    "camera_azimuth_deg": 270.0,
    "camera_elevation_deg": -5.0,
    "camera_distance_scale": 2.9,
    "target_offset": (0.0, 0.0, -0.35),
    "lens": 35.0,
    "world_kind": "studio_gray",
    "density_scale": 13.0,
    "density_gamma": 0.80,
    "density_threshold": 0.018,
    "volume_color": (0.66, 0.68, 0.72),
    "anisotropy": 0.28,
}

INDUSTRIAL_CHIMNEY_PAPER_PRESET = {
    "camera_azimuth_deg": 270.0,
    "camera_elevation_deg": -5.0,
    "camera_distance_scale": 2.5,
    "target_offset": (0.0, 0.0, -0.20),
    "lens": 35.0,
    "world_kind": "studio_gray",
    "density_scale": 768.0,
    "density_gamma": 1.0,
    "density_threshold": 0.0,
    "volume_color": (0.68, 0.72, 0.78),
    "anisotropy": 0.2,
}

MATERIAL_PRESETS = {
    "paper_gray": {
        "density_scale": 768.0,
        "volume_color": (0.68, 0.72, 0.78),
        "anisotropy": 0.2,
    },
    "charcoal": {
        "density_scale": 1152.0,
        "volume_color": (0.18, 0.20, 0.24),
        "anisotropy": 0.12,
    },
    "soft_ash": {
        "density_scale": 640.0,
        "volume_color": (0.52, 0.56, 0.62),
        "anisotropy": 0.38,
    },
    "cool_steel": {
        "density_scale": 900.0,
        "volume_color": (0.30, 0.40, 0.58),
        "anisotropy": 0.24,
    },
}


def _argv_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def _configure_cycles_device(device: str) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    if device == "CPU":
        scene.cycles.device = "CPU"
        return

    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = device
    prefs.get_devices()

    for cycles_device in prefs.devices:
        cycles_device.use = cycles_device.type == device

    print(
        "CYCLES_DEVICES",
        [(d.name, d.type, d.use) for d in prefs.devices],
        flush=True,
    )
    scene.cycles.device = "GPU"


def _parse_rgb(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected R,G,B color, got: {value}")
    return (parts[0], parts[1], parts[2])


def _parse_xyz(value: str) -> tuple[float, float, float]:
    return _parse_rgb(value)


def _read_vbt_dimensions(filepath: str) -> tuple[int, int, int, int] | None:
    path = Path(bpy.path.abspath(filepath))
    if not path.exists() or path.suffix.lower() != ".vbtp":
        return None
    with path.open("rb") as handle:
        raw = handle.read(struct.calcsize("<8s11I"))
    if len(raw) != struct.calcsize("<8s11I"):
        return None
    magic, version, width, height, depth, frames, *_ = struct.unpack("<8s11I", raw)
    if magic != b"VBTPACK4":
        return None
    if version == 0 or width == 0 or height == 0 or depth == 0 or frames == 0:
        return None
    return width, height, depth, frames


def _primary_volume_object() -> bpy.types.Object | None:
    for obj in bpy.context.scene.objects:
        if obj.type == "VOLUME":
            return obj
    return None


def _configure_volume_aspect(fit_scale: float) -> None:
    if fit_scale <= 0.0:
        return
    obj = _primary_volume_object()
    if obj is None:
        return
    dims = _read_vbt_dimensions(obj.data.filepath)
    if dims is None:
        return
    width, height, depth, frames = dims
    max_dim = float(max(width, height, depth))
    scale = Vector((fit_scale * width / max_dim, fit_scale * height / max_dim, fit_scale * depth / max_dim))
    obj.scale = scale
    obj.location = -0.5 * scale
    bpy.context.view_layer.update()
    print(
        "VBT_VOLUME_DIMS",
        {"width": width, "height": height, "depth": depth, "frames": frames, "world_scale": tuple(round(v, 6) for v in scale)},
        flush=True,
    )


def _configure_world(color: tuple[float, float, float], strength: float) -> None:
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is None:
        return
    background.inputs["Color"].default_value = (*color, 1.0)
    background.inputs["Strength"].default_value = strength


def _configure_world_kind(world_kind: str | None) -> None:
    if not world_kind:
        return

    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()

    output = tree.nodes.new("ShaderNodeOutputWorld")
    output.location = (300, 0)
    background = tree.nodes.new("ShaderNodeBackground")
    background.location = (70, 0)

    if world_kind == "studio_gray":
        background.inputs["Color"].default_value = (0.80, 0.81, 0.82, 1.0)
        background.inputs["Strength"].default_value = 1.0
        scene.view_settings.exposure = -0.15
    elif world_kind == "dark_stage":
        background.inputs["Color"].default_value = (0.02, 0.025, 0.03, 1.0)
        background.inputs["Strength"].default_value = 0.06
        scene.view_settings.exposure = -0.05
    else:
        background.inputs["Color"].default_value = (0.15, 0.17, 0.20, 1.0)
        background.inputs["Strength"].default_value = 0.3

    tree.links.new(background.outputs["Background"], output.inputs["Surface"])


def _make_showcase_volume_material(
    density_scale: float,
    density_gamma: float,
    density_threshold: float,
    volume_color: tuple[float, float, float],
    anisotropy: float,
) -> bpy.types.Material:
    material = bpy.data.materials.new("Native_VBT_Showcase_Volume")
    material.use_nodes = True
    tree = material.node_tree
    nodes = tree.nodes
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (300, 0)

    volume_info = nodes.new("ShaderNodeVolumeInfo")
    volume_info.location = (-900, 0)
    density_socket = volume_info.outputs["Density"]

    if density_threshold > 0.0:
        subtract = nodes.new("ShaderNodeMath")
        subtract.operation = "SUBTRACT"
        subtract.inputs[1].default_value = density_threshold
        subtract.location = (-700, 80)
        maximum = nodes.new("ShaderNodeMath")
        maximum.operation = "MAXIMUM"
        maximum.inputs[1].default_value = 0.0
        maximum.location = (-530, 80)
        tree.links.new(density_socket, subtract.inputs[0])
        tree.links.new(subtract.outputs["Value"], maximum.inputs[0])
        density_socket = maximum.outputs["Value"]

    power = nodes.new("ShaderNodeMath")
    power.operation = "POWER"
    power.inputs[1].default_value = density_gamma
    power.location = (-360, 60)
    multiply = nodes.new("ShaderNodeMath")
    multiply.operation = "MULTIPLY"
    multiply.inputs[1].default_value = density_scale
    multiply.location = (-180, 60)
    principled = nodes.new("ShaderNodeVolumePrincipled")
    principled.location = (20, 0)
    if "Color" in principled.inputs:
        principled.inputs["Color"].default_value = (*volume_color, 1.0)
    if "Anisotropy" in principled.inputs:
        principled.inputs["Anisotropy"].default_value = anisotropy
    if "Density Attribute" in principled.inputs:
        principled.inputs["Density Attribute"].default_value = "density"

    tree.links.new(density_socket, power.inputs[0])
    tree.links.new(power.outputs["Value"], multiply.inputs[0])
    tree.links.new(multiply.outputs["Value"], principled.inputs["Density"])
    tree.links.new(principled.outputs["Volume"], output.inputs["Volume"])
    return material


def _replace_volume_material(
    density_scale: float,
    density_gamma: float,
    density_threshold: float,
    volume_color: tuple[float, float, float],
    anisotropy: float,
) -> None:
    material = _make_showcase_volume_material(
        density_scale,
        density_gamma,
        density_threshold,
        volume_color,
        anisotropy,
    )
    for obj in bpy.context.scene.objects:
        if obj.type != "VOLUME":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(material)


def _configure_volume_material(
    density_scale: float | None,
    volume_color: tuple[float, float, float],
    anisotropy: float,
) -> None:
    for obj in bpy.context.scene.objects:
        if obj.type != "VOLUME":
            continue
        for material in obj.data.materials:
            if material is None or material.node_tree is None:
                continue
            for node in material.node_tree.nodes:
                if node.bl_idname == "ShaderNodeVolumePrincipled":
                    if density_scale is not None and "Density" in node.inputs:
                        node.inputs["Density"].default_value = density_scale
                    if "Color" in node.inputs:
                        node.inputs["Color"].default_value = (*volume_color, 1.0)
                    if "Anisotropy" in node.inputs:
                        node.inputs["Anisotropy"].default_value = anisotropy
                elif node.bl_idname == "ShaderNodeVolumeScatter":
                    if "Color" in node.inputs:
                        node.inputs["Color"].default_value = (*volume_color, 1.0)
                    if "Anisotropy" in node.inputs:
                        node.inputs["Anisotropy"].default_value = anisotropy
                elif node.bl_idname == "ShaderNodeMath" and density_scale is not None:
                    if getattr(node, "operation", "") == "MULTIPLY" and len(node.inputs) > 1:
                        node.inputs[1].default_value = density_scale


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _volume_bbox_world() -> tuple[Vector, float] | None:
    volume_objects = [obj for obj in bpy.context.scene.objects if obj.type == "VOLUME"]
    if not volume_objects:
        return None
    obj = volume_objects[0]
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_v = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    max_v = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    if max((max_v - min_v).x, (max_v - min_v).y, (max_v - min_v).z) <= 1.0e-8:
        # VBT volumes generate the unit cube mesh inside Cycles, but Blender's
        # Python bound_box can remain degenerate because there is no OpenVDB
        # tree loaded on the host side.
        unit_corners = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ]
        corners = [obj.matrix_world @ Vector(corner) for corner in unit_corners]
        min_v = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
        max_v = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    center = (min_v + max_v) * 0.5
    size = max_v - min_v
    radius = 0.5 * max(size.x, size.y, size.z)
    print(
        "VBT_VOLUME_BBOX",
        {
            "min": tuple(round(v, 6) for v in min_v),
            "max": tuple(round(v, 6) for v in max_v),
            "center": tuple(round(v, 6) for v in center),
            "radius": round(radius, 6),
        },
        flush=True,
    )
    return center, max(radius, 1.0)


def _ensure_camera() -> bpy.types.Object:
    camera = bpy.context.scene.camera
    if camera is None:
        camera_data = bpy.data.cameras.new("Camera")
        camera = bpy.data.objects.new("Camera", camera_data)
        bpy.context.collection.objects.link(camera)
        bpy.context.scene.camera = camera
    return camera


def _configure_orbit_camera(
    lens: float,
    camera_shift: tuple[float, float] | None,
    azimuth_deg: float,
    elevation_deg: float,
    distance_scale: float,
    target_offset: tuple[float, float, float],
) -> None:
    bbox = _volume_bbox_world()
    if bbox is None:
        return
    center, radius = bbox
    camera = _ensure_camera()
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    horizontal = math.cos(elevation) * distance_scale
    camera.location = center + Vector(
        (
            radius * horizontal * math.cos(azimuth),
            radius * horizontal * math.sin(azimuth),
            radius * math.sin(elevation) * distance_scale,
        )
    )
    target = center + Vector((radius * target_offset[0], radius * target_offset[1], radius * target_offset[2]))
    _look_at(camera, target)
    camera.data.lens = lens
    camera.data.clip_end = max(camera.data.clip_end, radius * 20.0)
    if camera_shift is not None:
        camera.data.shift_x = camera_shift[0]
        camera.data.shift_y = camera_shift[1]
    print(
        "VBT_CAMERA",
        {
            "location": tuple(round(v, 6) for v in camera.location),
            "target": tuple(round(v, 6) for v in target),
            "lens": lens,
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
            "distance_scale": distance_scale,
        },
        flush=True,
    )


def _configure_camera(
    auto_camera: bool,
    lens: float,
    camera_shift: tuple[float, float] | None,
    orbit: tuple[float, float, float] | None,
    target_offset: tuple[float, float, float],
) -> None:
    if orbit is not None:
        _configure_orbit_camera(lens, camera_shift, orbit[0], orbit[1], orbit[2], target_offset)
        return

    camera = _ensure_camera()
    camera.data.lens = lens
    if camera_shift is not None:
        camera.data.shift_x = camera_shift[0]
        camera.data.shift_y = camera_shift[1]

    if not auto_camera:
        return

    bbox = _volume_bbox_world()
    if bbox is None:
        return
    center, radius = bbox
    camera.location = center + Vector((0.32 * radius, -3.35 * radius, 0.72 * radius))
    _look_at(camera, center + Vector((0.0, 0.0, 0.10 * radius)))
    camera.data.lens = lens


def _apply_case_preset(args: argparse.Namespace) -> None:
    if args.case_preset == "industrial_chimney":
        preset = INDUSTRIAL_CHIMNEY_PRESET
    elif args.case_preset == "industrial_chimney_paper":
        preset = INDUSTRIAL_CHIMNEY_PAPER_PRESET
    else:
        return
    if args.density_scale is None:
        args.density_scale = preset["density_scale"]
    if args.density_gamma is None:
        args.density_gamma = preset["density_gamma"]
    if args.density_threshold is None:
        args.density_threshold = preset["density_threshold"]
    if args.volume_color is None:
        args.volume_color = ",".join(str(v) for v in preset["volume_color"])
    if args.anisotropy is None:
        args.anisotropy = preset["anisotropy"]
    if args.world_kind is None:
        args.world_kind = preset["world_kind"]
    if args.lens is None:
        args.lens = preset["lens"]
    if args.camera_azimuth_deg is None:
        args.camera_azimuth_deg = preset["camera_azimuth_deg"]
    if args.camera_elevation_deg is None:
        args.camera_elevation_deg = preset["camera_elevation_deg"]
    if args.camera_distance_scale is None:
        args.camera_distance_scale = preset["camera_distance_scale"]
    if args.target_offset is None:
        args.target_offset = ",".join(str(v) for v in preset["target_offset"])


def _apply_material_preset(args: argparse.Namespace) -> None:
    if args.material_preset == "custom":
        return
    preset = MATERIAL_PRESETS[args.material_preset]
    if args.density_scale is None:
        args.density_scale = preset["density_scale"]
    if args.volume_color is None:
        args.volume_color = ",".join(str(v) for v in preset["volume_color"])
    if args.anisotropy is None:
        args.anisotropy = preset["anisotropy"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-preset",
        choices=("none", "industrial_chimney", "industrial_chimney_paper"),
        default="none",
    )
    parser.add_argument(
        "--material-preset",
        choices=("custom", *MATERIAL_PRESETS.keys()),
        default="custom",
    )
    parser.add_argument("--device", choices=("CPU", "CUDA"), default="CPU")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--density-scale", type=float, default=None)
    parser.add_argument("--density-gamma", type=float, default=None)
    parser.add_argument("--density-threshold", type=float, default=None)
    parser.add_argument("--volume-color", default=None)
    parser.add_argument("--anisotropy", type=float, default=None)
    parser.add_argument("--world-color", default="0.035,0.040,0.045")
    parser.add_argument("--world-strength", type=float, default=0.35)
    parser.add_argument(
        "--world-kind",
        choices=("studio_gray", "dark_stage", "default"),
        default=None,
    )
    parser.add_argument("--exposure", type=float, default=-0.1)
    parser.add_argument("--look", default="Medium High Contrast")
    parser.add_argument("--auto-camera", action="store_true")
    parser.add_argument("--lens", type=float, default=None)
    parser.add_argument("--camera-azimuth-deg", type=float, default=None)
    parser.add_argument("--camera-elevation-deg", type=float, default=None)
    parser.add_argument("--camera-distance-scale", type=float, default=None)
    parser.add_argument("--target-offset", default=None)
    parser.add_argument("--camera-shift", default=None)
    parser.add_argument("--aspect-correct-volume", action="store_true")
    parser.add_argument("--volume-fit-scale", type=float, default=3.0)
    parser.add_argument("--showcase-material", action="store_true")
    parser.add_argument("--denoise", action="store_true")
    args = parser.parse_args(_argv_after_double_dash())
    _apply_material_preset(args)
    _apply_case_preset(args)
    if (
        args.case_preset == "industrial_chimney_paper" or args.material_preset != "custom"
    ) and args.showcase_material:
        raise ValueError(
            "paper and material presets require the direct Principled density-attribute path; "
            "do not pass --showcase-material"
        )

    _configure_cycles_device(args.device)

    scene = bpy.context.scene
    scene.frame_set(args.frame)
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = args.denoise
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.exposure = args.exposure
    scene.view_settings.gamma = 1.0
    if args.look:
        scene.view_settings.look = args.look

    if args.aspect_correct_volume:
        _configure_volume_aspect(args.volume_fit_scale)

    if args.world_kind is not None:
        _configure_world_kind(args.world_kind)
    else:
        _configure_world(_parse_rgb(args.world_color), args.world_strength)

    volume_color = _parse_rgb(args.volume_color or "0.68,0.72,0.78")
    anisotropy = 0.2 if args.anisotropy is None else args.anisotropy
    density_gamma = 1.0 if args.density_gamma is None else args.density_gamma
    density_threshold = 0.0 if args.density_threshold is None else args.density_threshold
    if args.showcase_material:
        if args.density_scale is None:
            raise ValueError("--showcase-material requires --density-scale or a case preset")
        _replace_volume_material(
            args.density_scale,
            density_gamma,
            density_threshold,
            volume_color,
            anisotropy,
        )
    else:
        _configure_volume_material(args.density_scale, volume_color, anisotropy)
    print(
        "VBT_MATERIAL",
        {
            "preset": args.material_preset,
            "path": "showcase" if args.showcase_material else "direct_density_attribute",
            "density_scale": args.density_scale,
            "volume_color": volume_color,
            "anisotropy": anisotropy,
        },
        flush=True,
    )

    orbit = None
    if (
        args.camera_azimuth_deg is not None
        and args.camera_elevation_deg is not None
        and args.camera_distance_scale is not None
    ):
        orbit = (args.camera_azimuth_deg, args.camera_elevation_deg, args.camera_distance_scale)
    lens = 50.0 if args.lens is None else args.lens
    target_offset = _parse_xyz(args.target_offset or "0.0,0.0,0.0")
    camera_shift = None
    if args.camera_shift is not None:
        parts = [float(part.strip()) for part in args.camera_shift.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Expected X,Y camera shift, got: {args.camera_shift}")
        camera_shift = (parts[0], parts[1])
    _configure_camera(args.auto_camera, lens, camera_shift, orbit, target_offset)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.output_prefix.resolve())
    bpy.ops.render.render(write_still=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
