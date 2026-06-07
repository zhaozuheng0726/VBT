# VBT Blender Bridge

Minimal Blender addon for loading one `.vbtp` frame through an OpenVDB proxy.

This path does not modify Blender or Cycles. It invokes:

- `render_temporal_vbt_to_vdb.exe` for proxy import
- `render_temporal_vbt_direct_preview.exe` for direct compressed preview

The addon resolves helper binaries in this order:

- `VBT_TOOLS_DIR`
- repo-local `build/`
- legacy workspace layout
- `PATH`

Useful environment variables:

- `VBT_PROJECT_ROOT`: repository root
- `VBT_TOOLS_DIR`: directory containing the helper executables
- `VBT_METADATA_DIR`: optional fallback directory for `*.metadata.json`

Current scope:

- import one `.vbtp` frame as a Blender Volume
- auto-fill likely metadata JSON paths when possible
- store the VBT source, metadata, frame, converter, and generated proxy path on the object
- reload the selected VBT Volume to the current scene frame from the `VBT` viewport panel
- optionally auto-refresh selected/imported VBT Volumes on Blender timeline frame changes
- keep material/camera setup in Blender, while only swapping the cached VDB proxy
- render an optional direct compressed VBT preview image without generating a VDB proxy

## Manual Use

1. Install `vbt_blender_bridge.py` from Blender's Add-ons panel.
2. Enable `VBT Blender Bridge`.
3. If needed, build the helper binaries from `../blender_bridge_tools/`.
4. Use `File > Import > VBT Frame (.vbtp)`.
5. Set:
   - `Metadata JSON`
   - `Frame`
   - optional cache directory
   - optional `render_temporal_vbt_to_vdb.exe` path
6. Select the imported Volume and use `Viewport > Sidebar > VBT > Reload Current Scene Frame` to refresh the proxy for the current timeline frame.
7. Use `Enable Timeline Auto Reload` if the VBT proxy should update when the Blender timeline frame changes.
8. Use `Render Direct VBT Preview` to generate a BMP preview by ray marching the compressed `.vbtp` payload directly.

Timeline mapping is:

```text
VBT data frame = Blender scene frame + timeline offset
```

Use `Use Zero-Based Timeline` when Blender frame 1 should map to VBT frame 0.

Imported objects store these custom properties:

- `vbt_source`
- `vbt_metadata`
- `vbt_frame`
- `vbt_proxy_vdb`

## Headless Test

```powershell
& "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" -b -P .\test_vbt_bridge_render.py -- `
  --addon-dir . `
  --input-vbt C:\path\to\asset.vbtp `
  --metadata C:\path\to\asset.metadata.json `
  --converter %VBT_PROJECT_ROOT%\3D\vdb_tools\build\Release\render_temporal_vbt_to_vdb.exe `
  --cache-dir C:\tmp\vbt_bridge_cache `
  --frame 100 `
  --reload-frame 101 `
  --auto-frame 102 `
  --direct-preview-output C:\tmp\vbt_direct_preview.bmp `
  --output C:\tmp\vbt_bridge_test.png
```

## Direct Preview Scope

`Render Direct VBT Preview` does not use Cycles and does not generate an OpenVDB proxy.
It calls `render_temporal_vbt_direct_preview.exe`, which samples the compressed VBT
payload directly and writes a lightweight BMP preview. This is intended as a
compressed-state preview path, not a replacement for production Cycles rendering.
