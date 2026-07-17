#!/usr/bin/env python3
"""Build and verify the canonical VBT Studio runtime data catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "data"
CATALOG = LIBRARY / "catalog.json"


ASSETS = [
    *(
        (
            "smoke",
            dataset,
            "density",
            f"data/smoke/{dataset}/{dataset}_density.vbtp",
            f"data/smoke/{dataset}/{dataset}_density.metadata.json",
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
    (
        "fire",
        "ground_explosion",
        "density",
        "data/fire/ground_explosion/ground_explosion_density.vbtp",
        "data/fire/ground_explosion/ground_explosion_density.metadata.json",
    ),
    (
        "fire",
        "ground_explosion",
        "flames",
        "data/fire/ground_explosion/ground_explosion_flames.vbtp",
        "data/fire/ground_explosion/ground_explosion_flames.metadata.json",
    ),
    (
        "fire",
        "aerial_explosion",
        "density",
        "data/fire/aerial_explosion/aerial_explosion_density.vbtp",
        "data/fire/aerial_explosion/aerial_explosion_density.metadata.json",
    ),
    (
        "fire",
        "aerial_explosion",
        "flames",
        "data/fire/aerial_explosion/aerial_explosion_flames.vbtp",
        "data/fire/aerial_explosion/aerial_explosion_flames.metadata.json",
    ),
    (
        "fire",
        "aerial_explosion",
        "temperature",
        "data/fire/aerial_explosion/aerial_explosion_temperature.vbtp",
        "data/fire/aerial_explosion/aerial_explosion_temperature.metadata.json",
    ),
    (
        "fluid",
        "quality_0p5",
        "levelset",
        "data/fluid/quality_0p5/water_flow_levelset_mode2.vbtp",
        "data/fluid/quality_0p5/water_flow_levelset_mode2.metadata.json",
    ),
    (
        "fluid",
        "performance_0p75",
        "levelset",
        "data/fluid/performance_0p75/water_flow_levelset_mode2.vbtp",
        "data/fluid/performance_0p75/water_flow_levelset_mode2.metadata.json",
    ),
]


def verify_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing metadata: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    required = ("width", "height", "depth", "frames")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Metadata {path} is missing {missing}")
    dimensions = [int(metadata[key]) for key in required]
    if any(value <= 0 for value in dimensions):
        raise ValueError(f"Metadata {path} has invalid dimensions {dimensions}")
    return metadata


def build_manifest() -> dict[str, Any]:
    records = []
    for category, dataset, field, vbt_relative, metadata_relative in ASSETS:
        vbt = ROOT / vbt_relative
        metadata_path = ROOT / metadata_relative
        if not vbt.is_file():
            raise FileNotFoundError(f"Missing canonical VBT asset: {vbt}")
        metadata = verify_metadata(metadata_path)
        records.append(
            {
                "category": category,
                "dataset": dataset,
                "field": field,
                "vbt": vbt_relative,
                "metadata": metadata_relative,
                "bytes": vbt.stat().st_size,
                "dimensions": [
                    int(metadata["width"]),
                    int(metadata["height"]),
                    int(metadata["depth"]),
                    int(metadata["frames"]),
                ],
                "status": "verified",
            }
        )
    return {
        "format": "vbt_data_library_v2",
        "asset_count": len(records),
        "categories": ["smoke", "fire", "fluid"],
        "storage": "canonical final compressed assets under data/",
        "source_data": {
            "smoke_and_fire_vdb": "3D/data/smoke",
            "fluid_source": "3D/data/fluid/source",
        },
        "assets": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest()
    if args.verify_only:
        if not CATALOG.is_file():
            raise FileNotFoundError(f"Missing catalog: {CATALOG}")
        current = json.loads(CATALOG.read_text(encoding="utf-8"))
        if current != manifest:
            raise RuntimeError("data/catalog.json is out of date; rebuild it first")
    else:
        CATALOG.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"VBT data library {'verified' if args.verify_only else 'built'}: "
        f"{manifest['asset_count']} canonical assets in {LIBRARY}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
