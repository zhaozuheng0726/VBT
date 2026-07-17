# 3D Data And OpenVDB Bridge

This directory is the 3D support workspace for VBT. It was merged into the
main repository on 2026-07-11.

## Current Data

- `data/test_scenarios`: corrected Scientific J1-J8 RAW files and metadata
- `data/fluid/water_flow_levelset.raw`: retained 1.0-voxel legacy baseline
- `data/fluid/water_flow_levelset.metadata.json`: legacy baseline layout
- `data/fluid/source`: original Alembic and Blender source assets
- `data/smoke`: six complete OpenVDB smoke/fire sequences used as local source data
- `../outputs/fluid_final_highres_20260715`: final 0.5 quality and 0.75
  performance liquid profiles

## Current Tools

- `vdb_tools`: OpenVDB conversion and metadata/RAW regression tests
- `render_temporal_vbt_to_vdb.cpp`: production VBTPACK4 to OpenVDB bridge
- `render_temporal_leaf_diagnose.cpp`: surface-band and leaf diagnostics
- `raw_to_vdb.cpp`: strict dense RAW to OpenVDB conversion
- `vdb_to_obj.cpp`: OpenVDB level-set mesh extraction
- `make_blender_water_animation_pairwise.py`: pairwise Cycles render pipeline

For `water_flow.abc`, VBT frame indices are zero-based while Alembic scene
frames are one-based. The pairwise driver therefore defaults to
`--abc-frame-offset 1`; VBT index 72 must be compared with Alembic frame 73.

Fluid evaluation uses two separate contracts:

- compression-only: original dense level set and Mode2 output pass through the
  same VDB, `phi=0`, smoothing, material, camera, and render pipeline
- end-to-end: Alembic source mesh is compared with the final Mode2 surface and
  includes mesh-to-level-set representation loss

Do not use an Alembic-versus-Mode2 image as evidence of compression-only error.
Gaussian level-set smoothing is disabled by default because the frame-72 sweep
showed that the former `mix=0.3` contract removed thin sheets and droplets.
Mode2 surface, coarse-guard, and temporal bands are converted from voxel units
to world distance using metadata `voxel_size`; this is required for resolution-
independent behavior.

`compare/openvdb-master` and `compare/zfp-develop` are local source
dependencies and are intentionally excluded from Git. Historical baseline
trees and generated results are also excluded.

## Build

From the VBT repository root:

```powershell
cmake -S 3D/vdb_tools -B 3D/vdb_tools/build
cmake --build 3D/vdb_tools/build --config Release
ctest --test-dir 3D/vdb_tools/build -C Release --output-on-failure
```

Large source volumes and generated outputs remain local and are not versioned.
