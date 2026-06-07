from __future__ import annotations

bl_info = {
    "name": "VBT Blender Bridge",
    "author": "VBT",
    "version": (0, 1, 0),
    "blender": (5, 1, 0),
    "location": "File > Import > VBT Frame (.vbtp)",
    "description": "Import a VBT frame into Blender through a lazy OpenVDB proxy.",
    "category": "Import-Export",
}

import os
import re
import shutil
import subprocess
from pathlib import Path

import bpy
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)
from bpy_extras.io_utils import ImportHelper


def _project_root_from_here() -> Path | None:
    env_root = os.getenv("VBT_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src").exists() and (parent / "render").exists() and (parent / "cycles_native_vbt").exists():
            return parent
        if (parent / "VBT").exists() and (parent / "3D").exists():
            return parent
        if parent.name == "VBT" and (parent.parent / "3D").exists():
            return parent.parent
    return None


def _default_tool_path(exe_name: str) -> Path:
    candidates = []
    tools_dir = os.getenv("VBT_TOOLS_DIR")
    if tools_dir:
        candidates.append(Path(tools_dir).expanduser() / exe_name)
    project_root = _project_root_from_here()
    if project_root is not None:
        candidates.append(project_root / "build" / "Release" / exe_name)
        candidates.append(project_root / "build" / exe_name)
        candidates.append(project_root / "build" / "blender_bridge_tools" / "Release" / exe_name)
        candidates.append(project_root / "build" / "blender_bridge_tools" / exe_name)
        candidates.append(project_root / "3D" / "vdb_tools" / "build" / "Release" / exe_name)
    found = shutil.which(exe_name)
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else Path(exe_name)


DEFAULT_CONVERTER = _default_tool_path("render_temporal_vbt_to_vdb.exe")
DEFAULT_DIRECT_PREVIEW = _default_tool_path("render_temporal_vbt_direct_preview.exe")

_VBT_RELOAD_GUARD = False


def _safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def _resolve_blender_path(path: str) -> Path:
    if not path:
        return Path()
    return Path(bpy.path.abspath(path)).expanduser()


def _default_cache_dir() -> Path:
    blend_dir = Path(bpy.path.abspath("//"))
    if str(blend_dir) and blend_dir.exists():
        return blend_dir / "vbt_bridge_cache"
    return Path(os.getenv("TEMP", ".")) / "vbt_bridge_cache"


def _resolve_executable(path: Path) -> str | None:
    if path.exists():
        return str(path)
    found = shutil.which(str(path)) or shutil.which(path.name)
    return found


def _metadata_candidates(input_vbt: Path) -> list[Path]:
    stem = input_vbt.stem
    parent = input_vbt.parent
    candidates = [
        parent / f"{stem}.metadata.json",
        parent / f"{stem}_density.metadata.json",
        parent / f"{stem.replace('_mainline', '')}.metadata.json",
        parent / f"{stem.replace('_mainline', '')}_density.metadata.json",
    ]
    candidates.extend(sorted(parent.glob("*.metadata.json")))
    metadata_dir = os.getenv("VBT_METADATA_DIR")
    if metadata_dir:
        candidates.extend(sorted(Path(metadata_dir).expanduser().glob("*.metadata.json")))
    project_root = _project_root_from_here()
    if project_root is not None:
        converted = project_root / "3D" / "smokeDate" / "converted"
        if converted.exists():
            candidates.extend(sorted(converted.glob("*.metadata.json")))
    seen = set()
    unique = []
    for path in candidates:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def find_metadata_for_vbt(input_vbt: Path) -> Path | None:
    stem_tokens = set(re.split(r"[_\-.]+", input_vbt.stem.lower()))
    best_path = None
    best_score = -1
    for path in _metadata_candidates(input_vbt):
        if not path.exists():
            continue
        name_tokens = set(re.split(r"[_\-.]+", path.stem.lower()))
        score = len(stem_tokens & name_tokens)
        if input_vbt.parent == path.parent:
            score += 4
        if "density" in path.stem.lower():
            score += 1
        if score > best_score:
            best_score = score
            best_path = path
    return best_path


def _run_converter(
    converter: Path,
    input_vbt: Path,
    metadata_path: Path,
    output_vdb: Path,
    frame: int,
    grid_name: str,
    background: float,
    sparse_threshold: float,
    use_clamp_min: bool,
    clamp_min: float,
    use_clamp_max: bool,
    clamp_max: float,
    shell_empty_scale: float,
    use_shell_empty_max: bool,
    shell_empty_max: float,
) -> str:
    converter_cmd = _resolve_executable(converter)
    if converter_cmd is None:
        raise RuntimeError(f"VBT converter not found: {converter}")
    if not input_vbt.exists():
        raise RuntimeError(f"VBT file not found: {input_vbt}")
    if not metadata_path.exists():
        raise RuntimeError(f"Metadata file not found: {metadata_path}")

    output_vdb.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        converter_cmd,
        "--input-vbt",
        str(input_vbt),
        "--metadata",
        str(metadata_path),
        "--output-vdb",
        str(output_vdb),
        "--frame",
        str(frame),
        "--grid-name",
        grid_name,
        "--background",
        str(background),
        "--sparse-threshold",
        str(sparse_threshold),
        "--shell-empty-scale",
        str(shell_empty_scale),
    ]
    if use_clamp_min:
        cmd.extend(["--clamp-min", str(clamp_min)])
    if use_clamp_max:
        cmd.extend(["--clamp-max", str(clamp_max)])
    if use_shell_empty_max:
        cmd.extend(["--shell-empty-max", str(shell_empty_max)])

    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        details = (proc.stdout + "\n" + proc.stderr).strip()
        raise RuntimeError(f"VBT to VDB conversion failed:\n{details}")
    return (proc.stdout + "\n" + proc.stderr).strip()


def import_vbt_frame_as_volume(
    *,
    input_vbt: Path,
    metadata_path: Path,
    frame: int,
    cache_dir: Path,
    converter: Path = DEFAULT_CONVERTER,
    grid_name: str = "density",
    background: float = 0.0,
    sparse_threshold: float = 0.0,
    use_clamp_min: bool = False,
    clamp_min: float = 0.0,
    use_clamp_max: bool = False,
    clamp_max: float = 1.0,
    shell_empty_scale: float = 1.0,
    use_shell_empty_max: bool = False,
    shell_empty_max: float = 0.2,
) -> bpy.types.Object:
    output_vdb = cache_dir / f"{_safe_stem(input_vbt)}_f{frame:04d}.vdb"
    log = _run_converter(
        converter,
        input_vbt,
        metadata_path,
        output_vdb,
        frame,
        grid_name,
        background,
        sparse_threshold,
        use_clamp_min,
        clamp_min,
        use_clamp_max,
        clamp_max,
        shell_empty_scale,
        use_shell_empty_max,
        shell_empty_max,
    )

    before = set(bpy.context.scene.objects)
    bpy.ops.object.volume_import(filepath=str(output_vdb))
    after = set(bpy.context.scene.objects)
    new_objects = list(after - before)
    if new_objects:
        obj = new_objects[0]
    elif bpy.context.object is not None:
        obj = bpy.context.object
    else:
        raise RuntimeError(f"Blender failed to import generated VDB: {output_vdb}")

    obj.name = f"VBT_{input_vbt.stem}_f{frame:04d}"
    obj.data.name = f"{obj.name}_Volume"
    obj["vbt_source"] = str(input_vbt)
    obj["vbt_metadata"] = str(metadata_path)
    obj["vbt_frame"] = frame
    obj["vbt_proxy_vdb"] = str(output_vdb)
    obj["vbt_cache_dir"] = str(cache_dir)
    obj["vbt_converter"] = str(converter)
    obj["vbt_grid_name"] = grid_name
    obj["vbt_background"] = background
    obj["vbt_sparse_threshold"] = sparse_threshold
    obj["vbt_shell_empty_scale"] = shell_empty_scale
    obj["vbt_use_shell_empty_max"] = use_shell_empty_max
    obj["vbt_shell_empty_max"] = shell_empty_max
    obj["vbt_use_clamp_min"] = use_clamp_min
    obj["vbt_clamp_min"] = clamp_min
    obj["vbt_use_clamp_max"] = use_clamp_max
    obj["vbt_clamp_max"] = clamp_max
    obj["vbt_converter_log"] = log[-4096:]
    obj["vbt_auto_reload"] = False
    obj["vbt_timeline_offset"] = 0
    return obj


def reload_vbt_object_frame(obj: bpy.types.Object, frame: int, *, force: bool = False) -> Path | None:
    if not force and int(obj.get("vbt_frame", -1)) == frame:
        return Path(obj.get("vbt_proxy_vdb", ""))
    input_vbt = Path(obj["vbt_source"])
    metadata_path = Path(obj["vbt_metadata"])
    cache_dir = Path(obj.get("vbt_cache_dir", str(_default_cache_dir())))
    converter = Path(obj.get("vbt_converter", str(DEFAULT_CONVERTER)))
    grid_name = str(obj.get("vbt_grid_name", "density"))
    output_vdb = cache_dir / f"{_safe_stem(input_vbt)}_f{frame:04d}.vdb"

    log = _run_converter(
        converter,
        input_vbt,
        metadata_path,
        output_vdb,
        frame,
        grid_name,
        float(obj.get("vbt_background", 0.0)),
        float(obj.get("vbt_sparse_threshold", 0.0)),
        bool(obj.get("vbt_use_clamp_min", False)),
        float(obj.get("vbt_clamp_min", 0.0)),
        bool(obj.get("vbt_use_clamp_max", False)),
        float(obj.get("vbt_clamp_max", 1.0)),
        float(obj.get("vbt_shell_empty_scale", 1.0)),
        bool(obj.get("vbt_use_shell_empty_max", False)),
        float(obj.get("vbt_shell_empty_max", 0.2)),
    )

    if not hasattr(obj.data, "filepath"):
        raise RuntimeError("Selected object data is not a file-backed Blender Volume")
    obj.data.filepath = str(output_vdb)
    if hasattr(obj.data, "reload"):
        obj.data.reload()
    obj.name = f"VBT_{input_vbt.stem}_f{frame:04d}"
    obj["vbt_frame"] = frame
    obj["vbt_proxy_vdb"] = str(output_vdb)
    obj["vbt_converter_log"] = log[-4096:]
    return output_vdb


class VBT_OT_import_frame(bpy.types.Operator, ImportHelper):
    bl_idname = "vbt.import_frame"
    bl_label = "Import VBT Frame"
    bl_description = "Decode one VBT frame to an OpenVDB proxy and import it as a Blender Volume"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".vbtp"
    filter_glob: StringProperty(default="*.vbtp", options={"HIDDEN"})

    metadata_path: StringProperty(
        name="Metadata JSON",
        description="Frame metadata JSON used by the VBT to VDB bridge",
        subtype="FILE_PATH",
    )
    frame: IntProperty(name="Frame", default=0, min=0)
    cache_dir: StringProperty(
        name="Cache Directory",
        description="Directory for generated OpenVDB proxy files",
        subtype="DIR_PATH",
        default="",
    )
    converter_path: StringProperty(
        name="VBT to VDB Converter",
        description="Path to render_temporal_vbt_to_vdb.exe",
        subtype="FILE_PATH",
        default=str(DEFAULT_CONVERTER),
    )
    grid_name: StringProperty(name="Grid Name", default="density")
    background: FloatProperty(name="Background", default=0.0)
    sparse_threshold: FloatProperty(name="Sparse Threshold", default=0.0, min=0.0)
    shell_empty_scale: FloatProperty(name="Shell Empty Scale", default=1.0)
    use_shell_empty_max: BoolProperty(name="Use Shell Empty Max", default=False)
    shell_empty_max: FloatProperty(name="Shell Empty Max", default=0.2)
    use_clamp_min: BoolProperty(name="Use Clamp Min", default=False)
    clamp_min: FloatProperty(name="Clamp Min", default=0.0)
    use_clamp_max: BoolProperty(name="Use Clamp Max", default=False)
    clamp_max: FloatProperty(name="Clamp Max", default=1.0)

    def check(self, context):
        if self.filepath and not self.metadata_path:
            metadata = find_metadata_for_vbt(_resolve_blender_path(self.filepath))
            if metadata is not None:
                self.metadata_path = str(metadata)
                return True
        return False

    def execute(self, context):
        try:
            if not self.metadata_path:
                metadata = find_metadata_for_vbt(_resolve_blender_path(self.filepath))
                if metadata is not None:
                    self.metadata_path = str(metadata)
            cache_dir = _resolve_blender_path(self.cache_dir) if self.cache_dir else _default_cache_dir()
            obj = import_vbt_frame_as_volume(
                input_vbt=_resolve_blender_path(self.filepath),
                metadata_path=_resolve_blender_path(self.metadata_path),
                frame=self.frame,
                cache_dir=cache_dir,
                converter=_resolve_blender_path(self.converter_path),
                grid_name=self.grid_name,
                background=self.background,
                sparse_threshold=self.sparse_threshold,
                use_clamp_min=self.use_clamp_min,
                clamp_min=self.clamp_min,
                use_clamp_max=self.use_clamp_max,
                clamp_max=self.clamp_max,
                shell_empty_scale=self.shell_empty_scale,
                use_shell_empty_max=self.use_shell_empty_max,
                shell_empty_max=self.shell_empty_max,
            )
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        context.view_layer.objects.active = obj
        obj.select_set(True)
        self.report({"INFO"}, f"Imported VBT frame {self.frame}: {obj.name}")
        return {"FINISHED"}


class VBT_OT_reload_selected_frame(bpy.types.Operator):
    bl_idname = "vbt.reload_selected_frame"
    bl_label = "Reload VBT Frame"
    bl_description = "Regenerate the selected VBT Volume proxy for a target frame"
    bl_options = {"REGISTER", "UNDO"}

    frame: IntProperty(
        name="Frame",
        description="Frame to decode; use -1 for the current Blender scene frame",
        default=-1,
        min=-1,
    )

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and "vbt_source" in obj and "vbt_metadata" in obj

    def execute(self, context):
        obj = context.object
        offset = int(obj.get("vbt_timeline_offset", 0))
        frame = context.scene.frame_current + offset if self.frame < 0 else self.frame
        try:
            output_vdb = reload_vbt_object_frame(obj, frame, force=True)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Reloaded VBT frame {frame}: {output_vdb.name if output_vdb else 'unchanged'}")
        return {"FINISHED"}


class VBT_OT_toggle_auto_reload(bpy.types.Operator):
    bl_idname = "vbt.toggle_auto_reload"
    bl_label = "Toggle VBT Timeline Reload"
    bl_description = "Enable or disable automatic VBT proxy reload for the selected Volume on frame changes"
    bl_options = {"REGISTER", "UNDO"}

    enabled: BoolProperty(name="Enabled", default=True)

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and "vbt_source" in obj and "vbt_metadata" in obj

    def execute(self, context):
        obj = context.object
        obj["vbt_auto_reload"] = bool(self.enabled)
        self.report({"INFO"}, f"VBT timeline reload {'enabled' if self.enabled else 'disabled'} for {obj.name}")
        return {"FINISHED"}


class VBT_OT_set_timeline_offset(bpy.types.Operator):
    bl_idname = "vbt.set_timeline_offset"
    bl_label = "Set VBT Timeline Offset"
    bl_description = "Set data_frame = Blender_scene_frame + offset for automatic reload"
    bl_options = {"REGISTER", "UNDO"}

    offset: IntProperty(name="Offset", default=0)

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and "vbt_source" in obj and "vbt_metadata" in obj

    def execute(self, context):
        context.object["vbt_timeline_offset"] = int(self.offset)
        self.report({"INFO"}, f"VBT timeline offset set to {self.offset}")
        return {"FINISHED"}


class VBT_OT_direct_preview_selected(bpy.types.Operator):
    bl_idname = "vbt.direct_preview_selected"
    bl_label = "Direct VBT Preview"
    bl_description = "Render a preview image by directly sampling the selected compressed VBT payload"
    bl_options = {"REGISTER"}

    preview_exe: StringProperty(
        name="Direct Preview EXE",
        description="Path to render_temporal_vbt_direct_preview.exe",
        subtype="FILE_PATH",
        default=str(DEFAULT_DIRECT_PREVIEW),
    )
    output_dir: StringProperty(
        name="Output Directory",
        description="Directory for direct preview BMP files",
        subtype="DIR_PATH",
        default="",
    )
    frame: IntProperty(
        name="Frame",
        description="Frame to preview; use -1 for scene frame plus VBT timeline offset",
        default=-1,
        min=-1,
    )
    width: IntProperty(name="Width", default=512, min=32, max=4096)
    height: IntProperty(name="Height", default=288, min=32, max=4096)
    steps: IntProperty(name="Ray Steps", default=96, min=8, max=1024)
    exposure: FloatProperty(name="Exposure", default=0.045, min=0.0)
    density_scale: FloatProperty(name="Density Scale", default=1.0, min=0.0)

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and "vbt_source" in obj and "vbt_metadata" in obj

    def execute(self, context):
        obj = context.object
        preview_exe = _resolve_blender_path(self.preview_exe)
        preview_cmd = _resolve_executable(preview_exe)
        if preview_cmd is None:
            self.report({"ERROR"}, f"Direct preview EXE not found: {preview_exe}")
            return {"CANCELLED"}

        frame = context.scene.frame_current + int(obj.get("vbt_timeline_offset", 0)) if self.frame < 0 else self.frame
        if frame < 0:
            self.report({"ERROR"}, f"Invalid VBT frame: {frame}")
            return {"CANCELLED"}

        output_dir = _resolve_blender_path(self.output_dir) if self.output_dir else Path(obj.get("vbt_cache_dir", str(_default_cache_dir())))
        output_dir.mkdir(parents=True, exist_ok=True)
        input_vbt = Path(obj["vbt_source"])
        output_bmp = output_dir / f"{_safe_stem(input_vbt)}_direct_f{frame:04d}.bmp"
        cmd = [
            preview_cmd,
            "--input-vbt",
            str(input_vbt),
            "--metadata",
            str(Path(obj["vbt_metadata"])),
            "--output-bmp",
            str(output_bmp),
            "--frame",
            str(frame),
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--steps",
            str(self.steps),
            "--exposure",
            str(self.exposure),
            "--density-scale",
            str(self.density_scale),
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode != 0:
            details = (proc.stdout + "\n" + proc.stderr).strip()
            self.report({"ERROR"}, f"Direct VBT preview failed: {details[-512:]}")
            return {"CANCELLED"}

        image = bpy.data.images.load(str(output_bmp), check_existing=False)
        obj["vbt_direct_preview"] = str(output_bmp)
        obj["vbt_direct_preview_frame"] = frame
        obj["vbt_direct_preview_log"] = (proc.stdout + "\n" + proc.stderr)[-4096:]
        self.report({"INFO"}, f"Direct VBT preview rendered: {output_bmp.name} ({image.size[0]}x{image.size[1]})")
        return {"FINISHED"}


class VBT_PT_bridge_panel(bpy.types.Panel):
    bl_label = "VBT Bridge"
    bl_idname = "VBT_PT_bridge_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "VBT"

    def draw(self, context):
        layout = self.layout
        obj = context.object
        if obj is None or "vbt_source" not in obj:
            layout.label(text="Select a VBT-imported Volume.")
            layout.operator(VBT_OT_import_frame.bl_idname, text="Import VBT Frame")
            return

        layout.label(text=f"Source: {Path(obj['vbt_source']).name}")
        layout.label(text=f"Frame: {obj.get('vbt_frame', 'unknown')}")
        layout.label(text=f"Proxy: {Path(obj.get('vbt_proxy_vdb', '')).name}")
        layout.label(text=f"Auto reload: {'on' if obj.get('vbt_auto_reload', False) else 'off'}")
        layout.label(text=f"Timeline offset: {obj.get('vbt_timeline_offset', 0)}")
        op = layout.operator(VBT_OT_reload_selected_frame.bl_idname, text="Reload Current Scene Frame")
        op.frame = -1
        op = layout.operator(VBT_OT_reload_selected_frame.bl_idname, text="Reload Stored Frame")
        op.frame = int(obj.get("vbt_frame", context.scene.frame_current))
        if obj.get("vbt_auto_reload", False):
            op = layout.operator(VBT_OT_toggle_auto_reload.bl_idname, text="Disable Timeline Auto Reload")
            op.enabled = False
        else:
            op = layout.operator(VBT_OT_toggle_auto_reload.bl_idname, text="Enable Timeline Auto Reload")
            op.enabled = True
        op = layout.operator(VBT_OT_set_timeline_offset.bl_idname, text="Use Zero-Based Timeline")
        op.offset = -1
        op = layout.operator(VBT_OT_set_timeline_offset.bl_idname, text="Use Direct Timeline")
        op.offset = 0
        layout.separator()
        layout.operator(VBT_OT_direct_preview_selected.bl_idname, text="Render Direct VBT Preview")
        if obj.get("vbt_direct_preview"):
            layout.label(text=f"Direct preview: {Path(obj['vbt_direct_preview']).name}")


@persistent
def vbt_frame_change_handler(scene):
    global _VBT_RELOAD_GUARD
    if _VBT_RELOAD_GUARD:
        return
    _VBT_RELOAD_GUARD = True
    try:
        for obj in scene.objects:
            if not obj.get("vbt_auto_reload", False):
                continue
            if "vbt_source" not in obj or "vbt_metadata" not in obj:
                continue
            frame = scene.frame_current + int(obj.get("vbt_timeline_offset", 0))
            if frame < 0:
                continue
            try:
                reload_vbt_object_frame(obj, frame)
            except Exception as exc:
                print(f"[VBT Blender Bridge] auto reload failed for {obj.name}: {exc}")
    finally:
        _VBT_RELOAD_GUARD = False


def menu_func_import(self, context):
    self.layout.operator(VBT_OT_import_frame.bl_idname, text="VBT Frame (.vbtp)")


classes = (
    VBT_OT_import_frame,
    VBT_OT_reload_selected_frame,
    VBT_OT_toggle_auto_reload,
    VBT_OT_set_timeline_offset,
    VBT_OT_direct_preview_selected,
    VBT_PT_bridge_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    if vbt_frame_change_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(vbt_frame_change_handler)


def unregister():
    if vbt_frame_change_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(vbt_frame_change_handler)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
