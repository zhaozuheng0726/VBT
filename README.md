# VBT Mainline

This repository is a trimmed VBT code package that keeps only the main workflow:

- scientific field compression to `VBTPACK4`
- render-oriented temporal-first compression for smoke/fire volumes
- GPU-query and render-side probe tools
- Blender end-to-end integration code, including a native Cycles staging path

The current paper narrative is render-first. The scientific frontend is still
kept here because it exercises the same packed 4D backend and file layout.

## Repository Layout

- `src/`: main encoder/decoder backend and the unified CLI entry
- `render/`: GPU query benchmark, scientific frame export, and smoke-ray probe tools
- `cycles_native_vbt/`: native Cycles staging code and Blender helper scripts
- `blender_bridge/`: Blender addon that loads one VBT frame through an OpenVDB proxy
- `blender_bridge_tools/`: optional helper tools for proxy-VDB export and direct preview
- `blender_patch/`: patch material for the Blender/Cycles fork

## Build

The minimal build only needs a C++17 compiler:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

Optional dependencies:

- `OpenMP`: enabled automatically when available
- `Vulkan SDK + glslc`: required for `vbt_query_bench` and `vbt_smoke_vbt_bench`
- `OpenVDB`: only required for `render_temporal_vbt_to_vdb`

If you only want the encoder and native Cycles staging probe:

```powershell
cmake -S . -B build -DVBT_BUILD_RENDER_TOOLS=OFF -DVBT_BUILD_BLENDER_BRIDGE_TOOLS=OFF
cmake --build build --config Release
```

## Main Workflows

### 1. Scientific Compression

```powershell
.\build\Release\vbt_spatialfirst_probe.exe `
  --input-raw C:\path\to\J2.raw `
  --profile generic `
  --sample-step 8 `
  --generic-adaptive-coarse-keep `
  --generic-dense-crossover `
  --generic-dense-grid-res 4 `
  --generic-dense-bits 4 `
  --full-eval `
  --save-vbt C:\path\to\J2.vbtp
```

Export one reconstructed frame:

```powershell
.\build\Release\vbt_export_scientific_frame.exe `
  --input-vbt C:\path\to\J2.vbtp `
  --frame 64 `
  --output-raw C:\path\to\J2_frame64.raw
```

### 2. Render-Oriented Compression

```powershell
.\build\Release\vbt_spatialfirst_probe.exe `
  --input-raw C:\path\to\smoke.raw `
  --metadata C:\path\to\smoke.metadata.json `
  --profile density `
  --sample-step 8 `
  --save-vbt C:\path\to\smoke.vbtp
```

Generate a smoke-ray packet summary for downstream rendering tests:

```powershell
.\build\Release\vbt_render_smoke_probe.exe `
  --input-vbt C:\path\to\smoke.vbtp `
  --metadata C:\path\to\smoke.metadata.json `
  --frame 100 `
  --output-rays C:\path\to\rays.bin `
  --output-summary C:\path\to\probe_summary.json
```

### 3. GPU Query / Render Bench

If Vulkan is available:

```powershell
.\build\Release\vbt_query_bench.exe `
  --input-vbt C:\path\to\J2.vbtp `
  --batch-size 65536 `
  --pattern random `
  --bench-mode compare
```

## Blender Integration

### Native Cycles Staging

`cycles_native_vbt/` contains the standalone sampler, blob packer, loader, and
Blender scene scripts for the native `.vbtp` path. The intended contract is:

```text
.vbtp asset -> Blender Volume object -> Cycles scene sync -> device buffers -> Cycles volume kernel sampler
```

Build the host-side sampler probe:

```powershell
cmake --build build --config Release --target vbt_sampler_cpu_probe
```

Create a minimal native-VBT Blender scene:

```powershell
blender --background --python cycles_native_vbt/create_native_vbt_scene.py -- `
  --input-vbt C:\path\to\smoke.vbtp `
  --frame 100 `
  --output-blend C:\path\to\native_vbt_test.blend
```

### Blender Proxy Bridge

`blender_bridge/` is a fallback workflow. It imports one frame through a cached
OpenVDB proxy and can also call the direct compressed preview tool.

If `OpenVDB` is available, build the helper tools with:

```powershell
cmake -S . -B build -DVBT_BUILD_BLENDER_BRIDGE_TOOLS=ON
cmake --build build --config Release --target render_temporal_vbt_to_vdb render_temporal_vbt_direct_preview
```

## Blender Patch

`blender_patch/native_cycles_vbt.patch` is the render-path patch for the Blender
fork used during development. Apply it to the recorded Blender commit before
building the native path.
