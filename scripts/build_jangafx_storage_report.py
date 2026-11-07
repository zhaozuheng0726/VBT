#!/usr/bin/env python3
"""Join verified archive, OpenVDB, NanoVDB, and VBT storage measurements."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "source" / "jangafx" / "manifests"
ARCHIVE_CSV = MANIFESTS / "archive_inventory.csv"
BASELINE_CSV = MANIFESTS / "vdb_nanovdb_summary.csv"
OUTPUT_CSV = MANIFESTS / "storage_baseline_paper.csv"
OUTPUT_TEX = MANIFESTS / "storage_baseline_paper_rows.tex"

LABELS = {
    "industrial_chimney": "Industrial Chimney",
    "tornado_looping": "Tornado Looping",
    "dust_devil": "Dust Devil",
    "ground_explosion": "Ground Explosion",
    "smoke_plume": "Smoke Plume",
    "dust_shockwave": "Dust Shockwave",
    "aerial_explosion": "Aerial Explosion",
}


def read_index(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {row["dataset"]: row for row in rows}


def integer(row: dict[str, str], key: str) -> int:
    return int(row[key])


def mib(value: int) -> float:
    return value / (1024.0 * 1024.0)


def main() -> None:
    archives = read_index(ARCHIVE_CSV)
    baselines = read_index(BASELINE_CSV)
    expected = set(LABELS)
    if set(archives) != expected:
        raise RuntimeError(f"Archive inventory datasets differ: {sorted(set(archives) ^ expected)}")
    if set(baselines) != expected:
        raise RuntimeError(f"Baseline datasets differ: {sorted(set(baselines) ^ expected)}")

    output_rows: list[dict[str, object]] = []
    tex_rows: list[str] = []
    for dataset, label in LABELS.items():
        archive = archives[dataset]
        baseline = baselines[dataset]
        matched_frames = integer(baseline, "matched_frames")
        if matched_frames != integer(archive, "vbt_matched_frames"):
            raise RuntimeError(f"Matched frame count differs for {dataset}")
        matched_source = integer(baseline, "matched_sequence_source_openvdb_bytes")
        if matched_source != integer(archive, "matched_source_vdb_bytes"):
            raise RuntimeError(f"Matched source VDB bytes differ for {dataset}")

        openvdb_bytes = integer(baseline, "matched_density_openvdb_bytes")
        nano_zip_bytes = integer(baseline, "nanovdb_zip_bytes")
        nano_none_bytes = integer(baseline, "nanovdb_none_bytes")
        vbt_bytes = integer(baseline, "vbt_density_bytes")
        row = {
            "dataset": dataset,
            "label": label,
            "matched_frames": matched_frames,
            "page_advertised_frames": integer(archive, "page_advertised_frames"),
            "archive_vdb_frames": integer(archive, "archive_vdb_frames"),
            "archive_frame_count_matches_page": archive["frame_count_matches_page"],
            "official_archive_bytes": integer(archive, "archive_bytes"),
            "official_archive_vdb_bytes": integer(archive, "archive_vdb_bytes"),
            "matched_source_vdb_bytes": matched_source,
            "matched_density_openvdb_bytes": openvdb_bytes,
            "nanovdb_zip_bytes": nano_zip_bytes,
            "nanovdb_none_bytes": nano_none_bytes,
            "vbt_density_bytes": vbt_bytes,
            "matched_density_openvdb_mib": round(mib(openvdb_bytes), 3),
            "nanovdb_zip_mib": round(mib(nano_zip_bytes), 3),
            "nanovdb_none_mib": round(mib(nano_none_bytes), 3),
            "vbt_density_mib": round(mib(vbt_bytes), 3),
            "openvdb_over_vbt": round(openvdb_bytes / vbt_bytes, 3),
            "nanovdb_zip_over_vbt": round(nano_zip_bytes / vbt_bytes, 3),
            "nanovdb_none_over_vbt": round(nano_none_bytes / vbt_bytes, 3),
        }
        output_rows.append(row)
        tex_rows.append(
            f"{label} & {matched_frames} & {mib(openvdb_bytes):.1f} & "
            f"{mib(nano_zip_bytes):.1f} & {mib(nano_none_bytes):.1f} & "
            f"{mib(vbt_bytes):.1f} & {openvdb_bytes / vbt_bytes:.2f}$\\times$ & "
            f"{nano_zip_bytes / vbt_bytes:.2f}$\\times$ & "
            f"{nano_none_bytes / vbt_bytes:.2f}$\\times$ \\\\"
        )

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    OUTPUT_TEX.write_text("\n".join(tex_rows) + "\n", encoding="ascii")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
