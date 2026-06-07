# Native Cycles VBT Backend TODO

## Goal

Make Cycles sample VBT density directly from resident VBT payload buffers:

```text
.vbtp asset -> Blender volume object -> Cycles scene sync -> device buffers -> Cycles volume kernel sampler
```

No cached `.vdb`, no OpenVDB proxy, no NanoVDB conversion in the native path.

## Non-goals for the first MVP

- Do not support the shell-residual route/tag 2 in the main path.
- Do not support all Cycles devices at once. Start with CPU + CUDA.
- Do not support motion blur, velocity grids, multi-grid VBT assets, or adaptive step-size policy changes.
- Do not replace Blender's existing OpenVDB/NanoVDB path.

## Phase 0: Baseline And Build Setup

- [x] Pick a fixed Blender source revision and record it in this directory.
- [ ] Build vanilla Blender/Cycles once before any VBT changes.
- [ ] Enable CPU Cycles render first; enable CUDA after CPU correctness passes.
- [ ] Add a minimal test `.blend` with one VBT volume object, camera, material, and one fixed frame.
- [x] Add a script that creates a minimal VBT volume scene without using the OpenVDB proxy bridge.
- [ ] Keep the current `blender_bridge/` path as a correctness fallback for image comparisons.

Current build blocker on this machine:

- Blender commit `940972dfaa708826f3d16709d8d54ae61d32f5a5` requires MSVC
  2022 17.14.14+ for `WITH_BLENDER=ON`; installed VS 2022 reports 17.14.9.
- `WITH_BLENDER=OFF` avoids that version check but still needs full Blender
  release manifest files and precompiled `lib/windows_x64` dependencies, which
  are absent from the sparse checkout.

Acceptance:

- Vanilla Blender build succeeds.
- Existing OpenVDB volume render still works.
- One VBT asset can be located by an absolute path in a local test scene.

## Phase 1: Shared VBT Sampler Extraction

- [x] Create a standalone render-VBT device sampler skeleton in `vbt_cycles_sampler.h`.
- [x] Create a standalone VBTPACK4 host loader in `vbt_cycles_loader.*` so the staging code can move into a Blender source checkout without depending on `render/src`.
- [x] Add a host-side probe that loads a `.vbtp` and samples fixed points through the Cycles-style sampler.
- [x] Route the host-side probe through the same packed resident blob layout used by the native Cycles image loader.
- [ ] Extend the host-side probe into a comparison test against `render/shaders/vbt_smoke_sample.comp` or `render_temporal_vbt_direct_preview.cpp`.
- [ ] Add bounds tests for empty leaves, frame clamping, edge leaves, and TemporalFine6 residuals.
- [x] Freeze the render-mode subset for the Cycles MVP: tags 0, 1, and 3 only.

Acceptance:

- CPU host test returns values matching the existing render sampler within `1e-4` for representative points.
- Tag 2 returns coarse-only or zero with an explicit unsupported-mode marker in debug builds.

## Phase 2: Blender Data Model

- Current prototype shortcut: use existing `Volume.filepath` with a `.vbtp`
  extension. This avoids DNA/RNA changes but is not a polished UI/data model.
- [ ] Add a VBT volume source option to Blender volume data or create a minimal VBT-specific ID/property wrapper.
- [ ] Store source path, metadata path, frame index, bbox min, voxel size, grid name, and density scale.
- [ ] Add file reload invalidation when path or frame changes.
- [ ] Keep UI minimal: path field, frame field, reload button.

Likely Blender-side files to inspect in a source checkout:

- `source/blender/makesdna/`
- `source/blender/makesrna/`
- `source/blender/blenkernel/`
- `source/blender/editors/io/`
- `source/blender/io/`

Acceptance:

- Blender can open a scene containing a VBT-backed volume reference.
- Changing frame invalidates the Cycles scene.
- Existing Volume/OpenVDB UI remains unaffected.

## Phase 3: Cycles Scene Sync

- [x] In Cycles Blender sync, detect VBT-backed volume objects before the OpenVDB/NanoVDB path.
- [x] Load the `.vbtp` header, offset table, and payload into host-side Cycles scene storage.
- [x] Create a Cycles-side VBT volume descriptor with dimensions, frame count, leaf size, leaf counts, bbox min, and current frame.
- [x] Attach the descriptor to the volume shader/attribute path without disturbing existing OpenVDB volumes.

Likely Cycles files to inspect:

- `intern/cycles/blender/volume.cpp`
- `intern/cycles/scene/volume.*`
- `intern/cycles/scene/image.*`
- `intern/cycles/scene/scene.*`

Acceptance:

- Cycles scene sync prints one VBT volume descriptor for the test asset.
- No `.vdb` file is generated.
- Existing OpenVDB volume render still works.

## Phase 4: Device Buffers

- [x] Add a compact descriptor for VBT volumes.
- [x] Upload descriptor, offset table, and payload as one resident uchar image blob through Cycles `ImageManager`.
- [ ] Split descriptor/frame updates from immutable payload upload.
- [ ] Track dirty flags so frame-only changes do not re-upload payload unless the file changes.
- [x] For CUDA, route VBT blobs as linear resident image buffers rather than texture objects.
- [x] For CPU, route VBT blobs through `KernelImageInfo::data` host pointers.

Acceptance:

- First render uploads VBT payload once.
- Frame change only updates the small descriptor if the same file remains loaded.
- Device memory usage reports include VBT payload bytes.

## Phase 5: Kernel Sampling

- [x] Include/adapt `vbt_cycles_sampler.h` into Cycles kernel code.
- [x] Add a VBT volume sampling branch in the volume attribute lookup path.
- [x] Map Cycles object position to VBT index position using image metadata transform.
- [x] Sample density for the current frame using tags 0, 1, and 3.
- [x] Return density to the existing volume attribute shader path.
- [x] Keep the initial sampler scalar-only: one density grid.

Likely kernel files to inspect:

- `intern/cycles/kernel/`
- `intern/cycles/kernel/svm/`
- `intern/cycles/kernel/osl/` only after SVM path works

Acceptance:

- CPU Cycles render produces a visible VBT volume without OpenVDB proxy.
- CUDA Cycles render matches CPU within visual tolerance.
- Rendered still image matches the proxy bridge baseline within expected differences.

## Phase 6: Validation

- [ ] Compare direct native Cycles render vs current OpenVDB proxy render for one frame.
- [ ] Compare fixed point samples against existing CPU direct preview sampler.
- [ ] Profile payload upload time, first render time, steady-state render time, and frame-change time.
- [ ] Record whether the native path improves frame-change latency over proxy conversion.

Acceptance:

- Native path has zero proxy generation time.
- Frame changes do not write `.vdb` files.
- Visual quality remains consistent with current VBT reconstruction.

## Phase 7: Paper/Artifact Wording

- [ ] If only Phases 1-3 are done: call it a native-integration prototype, not a native backend.
- [ ] If Phases 1-5 are done on CPU/CUDA: call it a CUDA/CPU native Cycles prototype.
- [ ] If all device backends are done: call it a native Cycles backend.

Suggested paper-safe wording for a CPU/CUDA-only prototype:

```text
We additionally prototype a native Cycles path that uploads the VBT offset table
and payload as resident device buffers and evaluates density directly in the
Cycles volume kernel for CPU/CUDA backends. This prototype avoids OpenVDB proxy
generation, but support for all Cycles device backends remains future work.
```
