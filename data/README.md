# VBT data library

This is the single data entry point for VBT Studio. Final compressed assets are
grouped by rendering purpose:

```text
data/
  smoke/   seven density-only dataset entries
  fire/    density + flames (+ temperature) grouped by dataset
  fluid/   quality 0.5 and performance 0.75 Mode2 level sets
```

The files under `data/` are the canonical final compressed assets. Runtime use
does not depend on dated campaign directories under `outputs/`. Density assets
shared by smoke and fire may still be NTFS hard links to avoid duplicate bytes,
but every canonical path is inside this directory.

## VBT Studio

- Smoke: open the dataset's `*_density.vbtp`.
- Fire: open `*_density.vbtp`, then use **Add Field** for `*_flames.vbtp` and,
  when present, `*_temperature.vbtp`.
- Fluid: open either profile's `water_flow_levelset_mode2.vbtp`.

VBT Studio opens this directory by default. Each VBT has a matching-stem
`*.metadata.json` in the same folder, so bounding boxes, field roles, voxel
scale, and level-set background values load automatically.

## Source data

Source data remains separate from compressed runtime assets:

```text
3D/data/smoke/         smoke and fire VDB sequences
3D/data/fluid/source/  water Alembic and Blender source assets
```

Rebuild or verify `catalog.json` against the canonical files with:

```powershell
python scripts/build_data_library.py
python scripts/build_data_library.py --verify-only
```
