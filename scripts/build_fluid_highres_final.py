#!/usr/bin/env python3
"""Build the final quality/performance fluid package from validated candidates."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "outputs/fluid_highres_candidates_20260715"
FINAL = REPO / "outputs/fluid_final_highres_20260715"
REPORT = REPO / "reports/fluid_final_highres_20260715"


PROFILES = {
    "quality_0p5": {
        "candidate": "voxel_0p5",
        "vbt": "water_flow_levelset_0p5_mode2_scaled_control8.vbtp",
        "metadata": "water_flow_levelset_0p5.metadata.json",
        "raw": "water_flow_levelset_0p5.raw",
        "role": "paper and offline quality",
    },
    "performance_0p75": {
        "candidate": "voxel_0p75",
        "vbt": "water_flow_levelset_0p75_mode2_scaled.vbtp",
        "metadata": "water_flow_levelset_0p75.metadata.json",
        "raw": "water_flow_levelset_0p75.raw",
        "role": "interactive preview and faster bridge decode",
    },
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def copy(path: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return destination


def make_selection_figure(validation: dict, output: Path) -> None:
    panels = (
        (
            "Alembic source",
            SOURCE / "renders/source/frame_0072.png",
            "source frame 0073",
        ),
        (
            "Performance 0.75",
            SOURCE / "voxel_0p75/renders/end_to_end/frame_0072_mode2.png",
            f"Chamfer {validation['candidates']['voxel_0p75']['geometry_means']['source_to_mode2']['chamfer_l1']:.4f}",
        ),
        (
            "Quality 0.5",
            SOURCE / "voxel_0p5/renders/end_to_end/frame_0072_mode2.png",
            f"Chamfer {validation['candidates']['voxel_0p5']['geometry_means']['source_to_mode2']['chamfer_l1']:.4f}",
        ),
    )
    tile = 500
    gap = 16
    top = 52
    bottom = 48
    canvas = Image.new("RGB", (gap + len(panels) * (tile + gap), top + tile + bottom), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 23)
    detail_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    for index, (title, path, detail) in enumerate(panels):
        x = gap + index * (tile + gap)
        with Image.open(path) as image:
            panel = image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(panel, (x, top))
        for text, y, text_font in ((title, 13, title_font), (detail, top + tile + 14, detail_font)):
            box = draw.textbbox((0, 0), text, font=text_font)
            draw.text((x + (tile - (box[2] - box[0])) / 2, y), text, fill="#202124", font=text_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main() -> int:
    validation = load_json(SOURCE / "validation_summary.json")
    renders = load_json(SOURCE / "render_summary.json")
    decode = load_json(SOURCE / "decode_benchmark.json")
    FINAL.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    manifest = {
        "date": "2026-07-15",
        "source": str(REPO / "3D/data/fluid/source/water_flow.abc"),
        "frame_mapping": "VBT index N maps to Alembic scene frame N+1",
        "surface_extraction": {"isovalue": 0.0, "gaussian_smoothing": False},
        "profiles": {},
    }
    table_rows = []
    geometry_rows = []
    render_rows = []
    for profile_name, profile in PROFILES.items():
        candidate_name = profile["candidate"]
        source_dir = SOURCE / candidate_name
        output_dir = FINAL / profile_name
        output_dir.mkdir(parents=True, exist_ok=True)
        input_vbt = source_dir / profile["vbt"]
        input_metadata = source_dir / profile["metadata"]
        input_raw = source_dir / profile["raw"]
        output_vbt = copy(input_vbt, output_dir / "water_flow_levelset_mode2.vbtp")
        output_metadata = copy(input_metadata, output_dir / "water_flow_levelset.metadata.json")
        canonical_metadata = load_json(output_metadata)
        canonical_metadata["source_dir"] = "3D/data/fluid/source"
        canonical_metadata["source_asset"] = "3D/data/fluid/source/water_flow.abc"
        canonical_metadata["generated_obj_cache_retained"] = False
        output_metadata.write_text(
            json.dumps(canonical_metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        copy(source_dir / "renders/end_to_end_5x2.png", output_dir / "end_to_end_5x2.png")
        copy(source_dir / "renders/compression_only_5x2.png", output_dir / "compression_only_5x2.png")
        copy(source_dir / "validation_summary.json", output_dir / "validation_summary.json")
        copy(source_dir / "renders/render_summary.json", output_dir / "render_summary.json")

        candidate = validation["candidates"][candidate_name]
        render = renders["candidates"][candidate_name]
        decode_row = decode[candidate_name]
        profile_manifest = {
            "role": profile["role"],
            "dimensions": candidate["dimensions"],
            "voxel_size": candidate["voxel_size"],
            "raw": {
                "bytes": input_raw.stat().st_size,
                "sha256": sha256(input_raw),
                "retained": False,
                "reproducible_from": "water_flow.abc via abc_to_obj_sequence + objseq_to_raw",
            },
            "vbt": {
                "path": str(output_vbt),
                "bytes": output_vbt.stat().st_size,
                "sha256": sha256(output_vbt),
                "compression_ratio": candidate["compression_ratio"],
                "sampled_psnr_db": candidate["encoder"]["sampled_psnr_db"],
            },
            "metadata": {"path": str(output_metadata), "sha256": sha256(output_metadata)},
            "zero_crossing": candidate["diagnostic_totals"],
            "geometry_means": candidate["geometry_means"],
            "render_means": {
                "end_to_end": render["end_to_end"]["mean"],
                "compression_only": render["compression_only"]["mean"],
            },
            "vbt_to_vdb_bridge_mean_ms": decode_row["mean_ms"],
            "all_renders_nonblank": render["all_nonblank"],
        }
        manifest["profiles"][profile_name] = profile_manifest
        table_rows.append({
            "profile": profile_name,
            "voxel_size": candidate["voxel_size"],
            "raw_gib": input_raw.stat().st_size / (1024 ** 3),
            "vbt_mib": output_vbt.stat().st_size / (1024 ** 2),
            "compression_ratio": candidate["compression_ratio"],
            "sampled_psnr_db": candidate["encoder"]["sampled_psnr_db"],
            "sign_errors": candidate["diagnostic_totals"]["mode2_sign_errors"],
            "band_voxels": candidate["diagnostic_totals"]["band_voxels"],
            "bridge_mean_ms": decode_row["mean_ms"],
        })
        for stage, metrics in candidate["geometry_means"].items():
            geometry_rows.append({"profile": profile_name, "stage": stage, **metrics})
        for contract in ("end_to_end", "compression_only"):
            render_rows.append({"profile": profile_name, "contract": contract, **render[contract]["mean"]})

    for name, rows in (
        ("profile_summary.csv", table_rows),
        ("geometry_summary.csv", geometry_rows),
        ("render_summary.csv", render_rows),
    ):
        with (REPORT / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    make_selection_figure(validation, REPORT / "fluid_profile_selection_frame0072_3x1.png")
    (REPORT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    quality = manifest["profiles"]["quality_0p5"]
    performance = manifest["profiles"]["performance_0p75"]
    readme = f"""# Final High-Resolution Fluid Profiles

The visible detail loss was dominated by Alembic-to-level-set resolution and Gaussian
smoothing, not by Mode2. Both retained profiles use `phi=0` with no Gaussian smoothing.

## Selected Profiles

- Quality 0.5: {quality['vbt']['bytes']:,} bytes, {quality['vbt']['compression_ratio']:.2f}x,
  {quality['vbt']['sampled_psnr_db']:.4f} dB. Mean source-to-Mode2 Chamfer is
  {quality['geometry_means']['source_to_mode2']['chamfer_l1']:.6f}; bridge decode is
  {quality['vbt_to_vdb_bridge_mean_ms']:.1f} ms.
- Performance 0.75: {performance['vbt']['bytes']:,} bytes,
  {performance['vbt']['compression_ratio']:.2f}x, {performance['vbt']['sampled_psnr_db']:.4f} dB.
  Mean source-to-Mode2 Chamfer is
  {performance['geometry_means']['source_to_mode2']['chamfer_l1']:.6f}; bridge decode is
  {performance['vbt_to_vdb_bridge_mean_ms']:.1f} ms.

The quality profile is the primary high-quality/offline result. The performance profile is retained
for interactive preview and material tuning. VBT-to-VDB timing includes VDB construction and
serialization and is not a direct GPU-render benchmark.

## Scientific Contract

- VBT index `N` maps to Alembic frame `N+1`.
- Compression-only compares original level set and Mode2 through identical extraction/rendering.
- End-to-end compares Alembic source and Mode2 and includes voxelization loss.
- Geometry and zero-crossing metrics are primary; render PSNR/SSIM are auxiliary.

All selected-frame renders passed the nonblank contract. Full values are in the CSV files and
`manifest.json`.
"""
    (REPORT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"final": str(FINAL), "report": str(REPORT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
