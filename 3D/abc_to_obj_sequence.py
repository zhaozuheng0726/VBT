import argparse
import json
import os

import bpy
from mathutils import Vector


def parse_args():
    argv = []
    if "--" in os.sys.argv:
        argv = os.sys.argv[os.sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description="Export an Alembic mesh cache to an OBJ sequence.")
    parser.add_argument("--input-abc", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--object-name", default="")
    parser.add_argument("--frame-start", type=int, default=-1)
    parser.add_argument("--frame-end", type=int, default=-1)
    parser.add_argument("--frame-step", type=int, default=1)
    return parser.parse_args(argv)


def choose_mesh_object(requested_name):
    if requested_name:
        obj = bpy.data.objects.get(requested_name)
        if obj is None:
            raise RuntimeError(f"Requested object not found: {requested_name}")
        if obj.type != "MESH":
            raise RuntimeError(f"Requested object is not a mesh: {requested_name} ({obj.type})")
        return obj

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects found after Alembic import")
    return max(meshes, key=lambda o: len(o.data.vertices) if o.data else 0)


def export_sequence(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.alembic_import(filepath=args.input_abc)

    scene = bpy.context.scene
    obj = choose_mesh_object(args.object_name)

    frame_start = scene.frame_start if args.frame_start < 0 else args.frame_start
    frame_end = scene.frame_end if args.frame_end < 0 else args.frame_end
    frame_step = max(1, args.frame_step)

    os.makedirs(args.output_dir, exist_ok=True)

    bbox_min = [1.0e18, 1.0e18, 1.0e18]
    bbox_max = [-1.0e18, -1.0e18, -1.0e18]
    exported = []

    for frame in range(frame_start, frame_end + 1, frame_step):
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(dg)

        for corner in eval_obj.bound_box:
            p = eval_obj.matrix_world @ Vector(corner)
            for axis in range(3):
                bbox_min[axis] = min(bbox_min[axis], p[axis])
                bbox_max[axis] = max(bbox_max[axis], p[axis])

        for other in bpy.data.objects:
            other.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        out_name = f"{obj.name}_{frame:04d}.obj"
        out_path = os.path.join(args.output_dir, out_name)
        bpy.ops.wm.obj_export(
            filepath=out_path,
            export_selected_objects=True,
            export_animation=False,
            export_triangulated_mesh=True,
        )
        exported.append(out_name)
        print(f"EXPORTED {frame} {out_path}")

    summary = {
        "input_abc": args.input_abc,
        "object_name": obj.name,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_step": frame_step,
        "frame_count": len(exported),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "exported_files": exported,
    }
    with open(os.path.join(args.output_dir, "sequence_info.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    export_sequence(parse_args())
