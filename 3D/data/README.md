# Source volume data

This directory contains source assets used to build and validate VBT data.
Runtime-ready compressed files are organized separately under `../../data` for
VBT Studio.

```text
3D/data/
  smoke/          source smoke/fire VDB sequences
  fluid/          retained level-set source and metadata
    source/       original water Alembic and Blender assets
  test_scenarios/ small conversion and regression fixtures
```

Do not move these source directories without updating the campaign scripts.
They are referenced by reproducible smoke, fire, and fluid conversion commands.
Use `python scripts/build_data_library.py` from the VBT root to rebuild the
categorized Studio entry points without copying source or compressed payloads.
