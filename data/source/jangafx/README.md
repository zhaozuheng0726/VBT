# JangaFX EmberGen VDB baselines

Official source:

`https://jangafx.com/software/embergen/download/free-vdb-animations`

The downloaded assets include an explicit JangaFX CC0 Public Domain license.
The immutable source URLs, expected archive bytes, and page-advertised frame
counts are recorded in `manifests/sources.json`. Archive inventories are the
authority when a download page and its actual file list disagree.

## Layout

```text
data/source/jangafx/
  archives/         downloaded ZIP/RAR files
  openvdb/          extracted OpenVDB sequences when extraction is required
  vbt_source_subset/ unified entry to the exact VDB frames used by VBT
  matched_openvdb/  temporary density-only OpenVDB baselines
  nanovdb/          temporary NanoVDB baselines and retained probes
  previews/         MP4/WEBM files shipped in archives
  presets/          EmberGen .ember presets shipped in archives
  manifests/        provenance, hashes, inventories, and measured CSV data
```

Large source and generated baseline files are excluded from Git. CSV manifests
and scripts are retained for reproducibility.

## Reproduction

```powershell
./scripts/download_jangafx_vdb.ps1 -Dataset all
./scripts/inventory_jangafx_archives.ps1 -Dataset all
./scripts/measure_jangafx_vdb_baselines.ps1 `
  -SourceRoot ./data/source/jangafx/vbt_source_subset -Dataset all
```

Full archive extraction is optional and is provided by
`scripts/extract_jangafx_vdb.ps1`. It is not required for storage inventory
when the matched VBT source frames already exist.

The downloader resolves MediaFire's temporary direct URL on every attempt,
supports byte-range resume, validates the exact server-reported byte count, and
writes SHA256 hashes. The extractor validates official frame counts, separates
previews and presets, and checks the exact frame subset used by each canonical
VBT asset.

## Comparison contracts

The baseline table keeps four byte counts separate:

- `official_source_openvdb_bytes`: every VDB frame and every field as shipped.
- `matched_sequence_source_openvdb_bytes`: source files for exactly the frames
  listed in the canonical VBT metadata; these files can still contain extra
  fields such as flames.
- `matched_density_openvdb_bytes`: those same density grids rewritten as
  density-only OpenVDB files with the standard OpenVDB writer.
- `nanovdb_none_bytes` and `nanovdb_zip_bytes`: the same density grids converted
  with OpenVDB 13.0.1 `openToNanoVDB`, using NanoVDB `Codec::NONE` and
  `Codec::ZIP`, respectively.

`Codec::NONE` measures the directly accessible NanoVDB layout and is relevant
to mapped/GPU-resident footprint. `Codec::ZIP` measures compressed disk storage.
They are not interchangeable. VBT is also density-only in this table, and every
ratio uses the exact VBT file bytes from `data/catalog.json`.

The per-frame conversion CSV also records read, conversion, and write call
durations as campaign diagnostics. These are sequential warm-cache runs and do
not flush the Windows file cache, so they are not a cold-load or interactive
playback benchmark and must not be presented as one.

Tornado Looping and Smoke Plume require explicit frame-subset handling: the
canonical VBT assets contain frames 0--127 and 0--165, while the official
downloads contain 200 and 167 frames. The metadata filename list, not a guessed
numeric range, defines the matched comparison.
