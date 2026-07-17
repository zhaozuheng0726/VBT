# Native Cycles VBT Integration Staging

This directory is the staging area for the native `.vbtp` path inside Cycles.

It is intentionally separate from `blender_bridge/`. The existing bridge converts
VBT frames to cached OpenVDB proxies and lets Blender/Cycles render those proxies.
That is useful as a DCC workflow bridge, but it is not a native Cycles backend.

The native target is different: Cycles keeps a VBT payload resident on the
device and samples it directly from the volume kernel, without generating
OpenVDB or NanoVDB intermediates.

Current status:

- The Blender/Cycles patch targets commit
  `940972dfaa708826f3d16709d8d54ae61d32f5a5` and is stored in
  `../blender_patch/native_cycles_vbt.patch`.
- The patch routes `.vbtp` Volume paths to a resident VBT density image blob and
  samples it in `kernel/util/vbt.h` from `kernel/util/image_3d.h`.
- `vbt_cycles_sampler.h`: first standalone device-style sampler for render VBT
  payloads. It covers the main-paper render modes: Empty, TemporalGrid4, and
  TemporalFine6. Tag 2 remains reserved for the compact shell route and is not
  part of the main Cycles MVP.
- `vbt_cycles_loader.*`: standalone VBTPACK4 host loader intended to move into a
  Blender/Cycles source checkout.
- `vbt_cycles_blob.*`: resident blob packer matching the native Cycles image
  layout: header, offset table, and payload pool in one uchar buffer.
- `vbt_sampler_cpu_probe.cpp`: CPU probe that loads a `.vbtp` and samples index
  coordinates through the packed blob and Cycles-style sampler.
- `create_native_vbt_scene.py`: Blender script for creating a minimal test scene
  that points a Volume datablock directly at a `.vbtp` file.
- `TODO.md`: implementation plan and current progress for the Blender/Cycles fork.

The current native patch is a CPU/CUDA prototype. It avoids `.vdb` proxy
generation, but final image parity and performance validation remain open.

## Local Probe

Build from the repository root:

```powershell
cmake -S . -B build
cmake --build build --config Release --target vbt_sampler_cpu_probe
```

## Native Scene Smoke Test

After applying `../blender_patch/native_cycles_vbt.patch` to the recorded Blender
commit and building Blender:

```powershell
blender --background --python cycles_native_vbt/create_native_vbt_scene.py -- `
  --input-vbt outputs/render_sequence_20260407/industrial_chimney_mainline.vbtp `
  --frame 100 `
  --output-blend build/native_vbt_test.blend
```

The generated scene stores the `.vbtp` path directly in a Blender Volume
datablock. It should not generate or reference `.vdb` proxy files.

Run:

```powershell
build\vbt_sampler_cpu_probe.exe `
  --input-vbt outputs\render_sequence_20260407\industrial_chimney_mainline.vbtp `
  --frame 100 `
  --probe 64,64,64 `
  --probe 100,100,100 `
  --probe 128,128,128
```
