#!/usr/bin/env python3
"""Validate full-sequence high-resolution fluid Mode2 candidates."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "outputs" / "fluid_highres_candidates_20260715"
FRAMES = (0, 36, 72, 108, 140)
RAW_TO_VDB = REPO / "3D/vdb_tools/build/Release/raw_to_vdb.exe"
VBT_TO_VDB = REPO / "3D/vdb_tools/build/Release/render_temporal_vbt_to_vdb.exe"
VDB_TO_OBJ = REPO / "3D/vdb_tools/build/Release/vdb_to_obj.exe"
DIAGNOSE = REPO / "3D/vdb_tools/build/Release/render_temporal_leaf_diagnose.exe"
COMPARE = REPO / "scripts/compare_surface_meshes.py"


CONFIGS = {
    "voxel_0p75": {
        "stem": "water_flow_levelset_0p75",
        "vbt": "water_flow_levelset_0p75_mode2_scaled.vbtp",
        "encoder_report": "encoder_report_scaled.md",
        "control_eps_scale": 4.0,
    },
    "voxel_0p5": {
        "stem": "water_flow_levelset_0p5",
        "vbt": "water_flow_levelset_0p5_mode2_scaled_control8.vbtp",
        "encoder_report": "encoder_report_scaled_control8.md",
        "control_eps_scale": 8.0,
    },
}


def run(command: list[str], capture: bool = False) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout if capture else ""


def parse_markdown_row(text: str, name: str) -> list[str]:
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        fields = [field.strip() for field in line.strip().strip("|").split("|")]
        if fields and fields[0].strip("`") == name:
            return fields
    raise ValueError(f"missing Markdown row: {name}")


def parse_diagnostic(text: str) -> dict[str, float | int]:
    row = parse_markdown_row(text, "TOTAL")
    return {
        "leaf_count": int(row[1]),
        "active_voxels": int(row[3]),
        "surface_active_voxels": int(row[4]),
        "surface_keyframes": int(row[5]),
        "band_voxels": int(row[7]),
        "coarse_sign_errors": int(row[8]),
        "coarse_sign_error_rate": float(row[9]),
        "mode2_sign_errors": int(row[10]),
        "mode2_sign_error_rate": float(row[11]),
        "coarse_band_rmse": float(row[12]),
        "mode2_band_rmse": float(row[13]),
    }


def parse_encoder_report(path: Path) -> dict[str, float | int]:
    row = parse_markdown_row(path.read_text(encoding="utf-8"), "levelset_surface_packed")
    return {
        "estimated_megabytes": float(row[1].replace("`", "").removesuffix(" MB")),
        "sampled_rmse": float(row[2].replace("`", "")),
        "sampled_psnr_db": float(row[3].replace("`", "").removesuffix(" dB")),
        "coarse_keyframes": int(row[4].replace("`", "")),
        "surface_keyframes": int(row[5].replace("`", "")),
        "fine_leaves": int(row[7].replace("`", "")),
    }


def average_metrics(rows: list[dict], section: str) -> dict[str, float]:
    keys = (
        "chamfer_l1",
        "sampled_hausdorff",
        "mean_unoriented_normal_angle_degrees",
    )
    return {
        key: sum(row[section]["symmetric"][key] for row in rows) / len(rows)
        for key in keys
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--samples", type=int, default=100_000)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.samples <= 0:
        parser.error("--samples must be positive")

    summary = {"frames": list(FRAMES), "samples_per_mesh": args.samples, "candidates": {}}
    for config_name, config in CONFIGS.items():
        candidate = root / config_name
        stem = candidate / str(config["stem"])
        raw = stem.with_suffix(".raw")
        metadata_path = stem.with_suffix(".metadata.json")
        vbt = candidate / str(config["vbt"])
        encoder_report_path = candidate / str(config["encoder_report"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        background = float(metadata["half_width_voxels"]) * float(metadata["voxel_size"])
        band_width = float(metadata["shell_width_voxels"]) * float(metadata["voxel_size"])
        raw_bytes = raw.stat().st_size
        vbt_bytes = vbt.stat().st_size
        validation_root = candidate / "validation"
        validation_root.mkdir(parents=True, exist_ok=True)

        diagnostics = []
        geometry_rows = []
        for frame in FRAMES:
            frame_root = validation_root / f"frame_{frame:04d}"
            frame_root.mkdir(parents=True, exist_ok=True)
            diagnostic_text = run(
                [
                    str(DIAGNOSE),
                    "--input-vbt", str(vbt),
                    "--input-raw", str(raw),
                    "--raw-metadata", str(metadata_path),
                    "--frame", str(frame),
                    "--iso-value", "0",
                    "--band-width", str(band_width),
                    "--top", "10",
                ],
                capture=True,
            )
            (frame_root / "leaf_diagnostic.md").write_text(diagnostic_text, encoding="utf-8")
            diagnostic = {"frame": frame, **parse_diagnostic(diagnostic_text)}
            diagnostics.append(diagnostic)

            original_vdb = frame_root / "original_levelset.vdb"
            mode2_vdb = frame_root / "mode2_levelset.vdb"
            original_obj = frame_root / "original_levelset.obj"
            mode2_obj = frame_root / "mode2_levelset.obj"
            source_obj = root / "obj_sequence" / f"water_{frame + 1:04d}.obj"

            run([
                str(RAW_TO_VDB), "--input-raw", str(raw), "--metadata", str(metadata_path),
                "--output-vdb", str(original_vdb), "--frame", str(frame),
                "--background", str(background),
            ])
            run([
                str(VBT_TO_VDB), "--input-vbt", str(vbt), "--metadata", str(metadata_path),
                "--output-vdb", str(mode2_vdb), "--frame", str(frame),
                "--background", str(background),
            ])
            for vdb, obj in ((original_vdb, original_obj), (mode2_vdb, mode2_obj)):
                run([
                    str(VDB_TO_OBJ), "--input-vdb", str(vdb), "--output-obj", str(obj),
                    "--grid-name", "density", "--isovalue", "0", "--adaptivity", "0",
                    "--smooth-gaussian-width", "0", "--smooth-gaussian-iterations", "1",
                    "--smooth-mix", "0",
                ])
            original_vdb.unlink()
            mode2_vdb.unlink()

            comparisons = {}
            comparison_specs = (
                ("source_to_original_levelset", source_obj, original_obj),
                ("original_levelset_to_mode2", original_obj, mode2_obj),
                ("source_to_mode2", source_obj, mode2_obj),
            )
            for name, reference, comparison in comparison_specs:
                output_json = frame_root / f"{name}.json"
                run([
                    "python", str(COMPARE), "--reference", str(reference),
                    "--candidate", str(comparison), "--samples", str(args.samples),
                    "--seed", "20260715", "--output-json", str(output_json),
                ], capture=True)
                comparisons[name] = json.loads(output_json.read_text(encoding="utf-8"))
            geometry_rows.append({"frame": frame, **comparisons})
            print(f"validated {config_name} frame {frame}", flush=True)

        candidate_summary = {
            "dimensions": [metadata["width"], metadata["height"], metadata["depth"], metadata["frames"]],
            "voxel_size": metadata["voxel_size"],
            "band_width_world": band_width,
            "control_eps_scale": config["control_eps_scale"],
            "raw_bytes": raw_bytes,
            "vbt_bytes": vbt_bytes,
            "compression_ratio": raw_bytes / vbt_bytes,
            "encoder": parse_encoder_report(encoder_report_path),
            "diagnostics": diagnostics,
            "diagnostic_totals": {
                "band_voxels": sum(row["band_voxels"] for row in diagnostics),
                "coarse_sign_errors": sum(row["coarse_sign_errors"] for row in diagnostics),
                "mode2_sign_errors": sum(row["mode2_sign_errors"] for row in diagnostics),
                "mean_coarse_band_rmse": sum(row["coarse_band_rmse"] for row in diagnostics) / len(diagnostics),
                "mean_mode2_band_rmse": sum(row["mode2_band_rmse"] for row in diagnostics) / len(diagnostics),
            },
            "geometry": geometry_rows,
            "geometry_means": {
                section: average_metrics(geometry_rows, section)
                for section in (
                    "source_to_original_levelset",
                    "original_levelset_to_mode2",
                    "source_to_mode2",
                )
            },
        }
        summary["candidates"][config_name] = candidate_summary
        (candidate / "validation_summary.json").write_text(
            json.dumps(candidate_summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )

    output = root / "validation_summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "candidates": list(CONFIGS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
