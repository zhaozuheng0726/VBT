#!/usr/bin/env python3
"""Build the VBT Studio data catalog without duplicating large payloads."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "data"


ASSETS = [
    # Smoke density fields.
    *( 
        (
            "smoke",
            dataset,
            "density",
            f"outputs/thesis_render_extension_20260711/{dataset}/{dataset}_density.vbtp",
            f"outputs/thesis_render_extension_20260711/{dataset}/{dataset}_density.metadata.json",
        )
        for dataset in (
            "industrial_chimney",
            "tornado_looping",
            "dust_devil",
            "ground_explosion",
            "smoke_plume",
            "dust_shockwave",
            "aerial_explosion",
        )
    ),
    # Fire datasets keep all fields together for VBT Studio's Add Field flow.
    (
        "fire",
        "ground_explosion",
        "density",
        "outputs/thesis_render_extension_20260711/ground_explosion/ground_explosion_density.vbtp",
        "outputs/thesis_render_extension_20260711/ground_explosion/ground_explosion_density.metadata.json",
    ),
    (
        "fire",
        "ground_explosion",
        "flames",
        "outputs/native_fire_campaign_20260714/ground_explosion/compressed/ground_explosion_flames.vbtp",
        "outputs/native_fire_campaign_20260714/ground_explosion/compressed/ground_explosion_flames.metadata.json",
    ),
    (
        "fire",
        "aerial_explosion",
        "density",
        "outputs/thesis_render_extension_20260711/aerial_explosion/aerial_explosion_density.vbtp",
        "outputs/thesis_render_extension_20260711/aerial_explosion/aerial_explosion_density.metadata.json",
    ),
    (
        "fire",
        "aerial_explosion",
        "flames",
        "outputs/native_fire_campaign_20260714/aerial_explosion/compressed/aerial_explosion_flames.vbtp",
        "outputs/native_fire_campaign_20260714/aerial_explosion/compressed/aerial_explosion_flames.metadata.json",
    ),
    (
        "fire",
        "aerial_explosion",
        "temperature",
        "outputs/native_fire_campaign_20260714/aerial_explosion/compressed/aerial_explosion_temperature.vbtp",
        "outputs/native_fire_campaign_20260714/aerial_explosion/compressed/aerial_explosion_temperature.metadata.json",
    ),
    (
        "fluid",
        "quality_0p5",
        "levelset",
        "outputs/fluid_final_highres_20260715/quality_0p5/water_flow_levelset_mode2.vbtp",
        "outputs/fluid_final_highres_20260715/quality_0p5/water_flow_levelset.metadata.json",
    ),
    (
        "fluid",
        "performance_0p75",
        "levelset",
        "outputs/fluid_final_highres_20260715/performance_0p75/water_flow_levelset_mode2.vbtp",
        "outputs/fluid_final_highres_20260715/performance_0p75/water_flow_levelset.metadata.json",
    ),
]


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def link_file(source: Path, destination: Path, verify_only: bool) -> str:
    if not source.is_file():
        raise FileNotFoundError(f"Missing canonical asset: {source}")
    if destination.exists():
        if not destination.is_file() or not os.path.samefile(source, destination):
            raise RuntimeError(f"Catalog path is not the expected hard link: {destination}")
        return "verified"
    if verify_only:
        raise FileNotFoundError(f"Missing catalog link: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)
    if not os.path.samefile(source, destination):
        raise RuntimeError(f"Hard-link verification failed: {destination}")
    return "created"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    records = []
    for category, dataset, field, vbt_relative, metadata_relative in ASSETS:
        source_vbt = ROOT / vbt_relative
        source_metadata = ROOT / metadata_relative
        destination_dir = LIBRARY / category / dataset
        destination_vbt = destination_dir / source_vbt.name
        destination_metadata = destination_dir / f"{destination_vbt.stem}.metadata.json"
        vbt_status = link_file(source_vbt, destination_vbt, args.verify_only)
        metadata_status = link_file(source_metadata, destination_metadata, args.verify_only)
        records.append(
            {
                "category": category,
                "dataset": dataset,
                "field": field,
                "vbt": relative(destination_vbt),
                "metadata": relative(destination_metadata),
                "canonical_vbt": relative(source_vbt),
                "canonical_metadata": relative(source_metadata),
                "bytes": source_vbt.stat().st_size,
                "hard_link_verified": os.path.samefile(source_vbt, destination_vbt),
                "status": f"{vbt_status}/{metadata_status}",
            }
        )

    manifest = {
        "format": "vbt_data_library_v1",
        "asset_count": len(records),
        "categories": ["smoke", "fire", "fluid"],
        "storage": "NTFS hard links; no VBT payload bytes are duplicated",
        "source_data": {
            "smoke_and_fire_vdb": "3D/data/smoke",
            "fluid_source": "3D/data/fluid/source",
        },
        "assets": records,
    }
    if not args.verify_only:
        LIBRARY.mkdir(parents=True, exist_ok=True)
        (LIBRARY / "catalog.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"VBT data library {'verified' if args.verify_only else 'built'}: "
        f"{len(records)} assets in {LIBRARY}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
