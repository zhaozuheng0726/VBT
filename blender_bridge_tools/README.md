# Blender Bridge Tools

This directory contains the helper binaries used by `blender_bridge/`.

- `render_temporal_vbt_direct_preview`: direct compressed preview path, no OpenVDB required
- `render_temporal_vbt_to_vdb`: one-frame `.vbtp -> .vdb` conversion for Blender proxy import

`render_temporal_vbt_to_vdb` is optional and only builds when `OpenVDB` is found.
