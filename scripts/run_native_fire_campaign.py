#!/usr/bin/env python3
"""Compress flame fields and render aligned smoke/fire comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

from audit_smoke_render_pairs import build_contact_sheet, compare_images
from check_render_nonblank import inspect_image
from verify_vdb_frame_alignment import verify_alignment


DATASETS = (
    {
        "key": "ground_explosion",
        "label": "Ground Explosion",
        "frame": 64,
        "source_dir": Path(
            "3D/data/smoke/GroundExplosionVDB/ground_explosion/ground_explosion_VDB"
        ),
        "source_file": "ground_explosion_0064.vdb",
    },
    {
        "key": "aerial_explosion",
        "label": "Aerial Explosion",
        "frame": 60,
        "source_dir": Path(
            "3D/data/smoke/AerialExplosionVDB/AerialExplosion/AerialExplosionVDB"
        ),
        "source_file": "AerialExplosion_0060.vdb",
    },
)


def run_logged(command: list[str], stdout_path: Path, stderr_path: Path, cwd: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        subprocess.run(command, cwd=cwd, check=True, stdout=stdout, stderr=stderr)
    if stderr_path.stat().st_size == 0:
        stderr_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--blender",
        type=Path,
        default=Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"),
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--size", type=int, default=640)
    args = parser.parse_args()

    root = args.root.resolve()
    blender = args.blender.resolve()
    vdb_to_raw = root / "3D/vdb_tools/build/Release/vdb_to_raw.exe"
    encoder = root / "build/Release/vbt_render_temporal_kf_probe.exe"
    decoder = root / "3D/vdb_tools/build/Release/render_temporal_vbt_to_vdb.exe"
    render_script = root / "blender_bridge_tools/render_vdb_smoke.py"
    density_root = root / "outputs/thesis_render_extension_20260711"
    output_root = root / "outputs/native_fire_campaign_20260714"
    report_root = root / "reports/native_fire_campaign_20260714"
    working_root = output_root / "_working"
    verify_root = output_root / "_verify_temp"
    report_root.mkdir(parents=True, exist_ok=True)
    working_root.mkdir(parents=True, exist_ok=True)

    required = (blender, vdb_to_raw, encoder, decoder, render_script)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    results = []
    contact_rows = []
    for spec in DATASETS:
        key = spec["key"]
        label = spec["label"]
        frame = spec["frame"]
        source_dir = root / spec["source_dir"]
        source_vdb = source_dir / spec["source_file"]
        dataset_root = output_root / key
        compressed_root = dataset_root / "compressed"
        reconstructed_root = dataset_root / "reconstructed"
        render_root = dataset_root / "comparison/fire"
        logs_root = dataset_root / "logs"
        for directory in (compressed_root, reconstructed_root, render_root, logs_root):
            directory.mkdir(parents=True, exist_ok=True)

        work_prefix = working_root / f"{key}_flames"
        work_raw = work_prefix.with_suffix(".raw")
        work_metadata = work_prefix.with_suffix(".metadata.json")
        flame_vbt = compressed_root / f"{key}_flames.vbtp"
        flame_metadata = compressed_root / f"{key}_flames.metadata.json"
        if not flame_vbt.is_file() or not flame_metadata.is_file():
            run_logged(
                [
                    str(vdb_to_raw),
                    "--input-dir",
                    str(source_dir),
                    "--output-prefix",
                    str(work_prefix),
                    "--grid",
                    "flames",
                    "--time-fastest",
                ],
                logs_root / "flames_to_raw.stdout.log",
                logs_root / "flames_to_raw.stderr.log",
                root,
            )
            flame_raw_bytes = work_raw.stat().st_size
            run_logged(
                [
                    str(encoder),
                    "--input-raw",
                    str(work_raw),
                    "--metadata",
                    str(work_metadata),
                    "--sample-step",
                    "8",
                    "--control-eps-scale",
                    "2",
                    "--packed-keyframe-codec",
                    "fp16",
                    "--final-only",
                    "--route-empty-visible-thr",
                    "0.001",
                    "--route-fine-visible-thr",
                    "0.005",
                    "--route-fine-band-thr",
                    "0.005",
                    "--route-coarse-rmse-thr",
                    "0.005",
                    "--route-coarse-peak-thr",
                    "0.02",
                    "--route-fine-gain-thr",
                    "0.03",
                    "--omp-threads",
                    "16",
                    "--save-vbt",
                    str(flame_vbt),
                    "--report",
                    str(report_root / f"{key}_flames_compression.md"),
                ],
                logs_root / "flames_encode.stdout.log",
                logs_root / "flames_encode.stderr.log",
                root,
            )
            shutil.copy2(work_metadata, flame_metadata)
        else:
            metadata = json.loads(flame_metadata.read_text(encoding="utf-8"))
            flame_raw_bytes = (
                metadata["width"]
                * metadata["height"]
                * metadata["depth"]
                * metadata["frames"]
                * 4
            )

        flame_meta = json.loads(flame_metadata.read_text(encoding="utf-8"))
        if flame_meta["grid_name"] != "flames":
            raise ValueError(f"wrong compressed grid for {key}: {flame_meta['grid_name']}")
        if flame_meta["frame_files"][frame] != source_vdb.name:
            raise ValueError(f"frame mapping mismatch for {key}")

        density_vbt = density_root / key / f"{key}_density.vbtp"
        density_metadata = density_root / key / f"{key}_density.metadata.json"
        density_reconstructed = reconstructed_root / f"frame{frame:04d}_density.vdb"
        flame_reconstructed = reconstructed_root / f"frame{frame:04d}_flames.vdb"
        run_logged(
            [
                str(decoder),
                "--input-vbt",
                str(density_vbt),
                "--metadata",
                str(density_metadata),
                "--output-vdb",
                str(density_reconstructed),
                "--frame",
                str(frame),
                "--grid-name",
                "density",
            ],
            logs_root / "density_decode.stdout.log",
            logs_root / "density_decode.stderr.log",
            root,
        )
        run_logged(
            [
                str(decoder),
                "--input-vbt",
                str(flame_vbt),
                "--metadata",
                str(flame_metadata),
                "--output-vdb",
                str(flame_reconstructed),
                "--frame",
                str(frame),
                "--grid-name",
                "flames",
            ],
            logs_root / "flames_decode.stdout.log",
            logs_root / "flames_decode.stderr.log",
            root,
        )

        previous_source = source_dir / flame_meta["frame_files"][frame - 1]
        next_source = source_dir / flame_meta["frame_files"][frame + 1]
        density_alignment = verify_alignment(
            previous_source,
            source_vdb,
            next_source,
            density_reconstructed,
            vdb_to_raw,
            verify_root,
            "density",
        )
        flame_alignment = verify_alignment(
            previous_source,
            source_vdb,
            next_source,
            flame_reconstructed,
            vdb_to_raw,
            verify_root,
            "flames",
        )

        original_output = render_root / f"{key}_original_fire_f{frame:04d}_{args.size}x{args.size}_s{args.samples}.png"
        reconstructed_output = render_root / f"{key}_reconstructed_fire_f{frame:04d}_{args.size}x{args.size}_s{args.samples}.png"
        common_render = [
            str(blender),
            "--background",
            "--python",
            str(render_script),
            "--",
            "--camera-reference-vdb",
            str(source_vdb),
            "--material-preset",
            "fire",
            "--width",
            str(args.size),
            "--height",
            str(args.size),
            "--samples",
            str(args.samples),
            "--step-size",
            "1.0",
            "--device",
            "CUDA",
        ]
        run_logged(
            common_render
            + [
                "--input-vdb",
                str(source_vdb),
                "--flame-vdb",
                str(source_vdb),
                "--output",
                str(original_output),
            ],
            logs_root / "render_original.stdout.log",
            logs_root / "render_original.stderr.log",
            root,
        )
        run_logged(
            common_render
            + [
                "--input-vdb",
                str(density_reconstructed),
                "--flame-vdb",
                str(flame_reconstructed),
                "--output",
                str(reconstructed_output),
            ],
            logs_root / "render_reconstructed.stdout.log",
            logs_root / "render_reconstructed.stderr.log",
            root,
        )

        render_metrics = compare_images(original_output, reconstructed_output)
        original_nonblank = inspect_image(original_output, "presentation")
        reconstructed_nonblank = inspect_image(reconstructed_output, "presentation")
        result = {
            "dataset": key,
            "label": label,
            "frame": frame,
            "source_vdb": str(source_vdb.relative_to(root)),
            "flame_raw_bytes": flame_raw_bytes,
            "flame_vbt": str(flame_vbt.relative_to(root)),
            "flame_vbt_bytes": flame_vbt.stat().st_size,
            "flame_raw_to_vbt": flame_raw_bytes / flame_vbt.stat().st_size,
            "density_reconstructed_vdb": str(density_reconstructed.relative_to(root)),
            "flame_reconstructed_vdb": str(flame_reconstructed.relative_to(root)),
            "density_alignment": density_alignment,
            "flame_alignment": flame_alignment,
            "original_render": str(original_output.relative_to(root)),
            "reconstructed_render": str(reconstructed_output.relative_to(root)),
            "render_metrics": render_metrics,
            "original_nonblank": original_nonblank,
            "reconstructed_nonblank": reconstructed_nonblank,
        }
        result_path = report_root / f"{key}_fire_result.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        results.append(result)
        contact_rows.append(
            {
                "label": label,
                "frame": frame,
                "reference": original_output,
                "candidate": reconstructed_output,
            }
        )

        for path in (work_raw, work_metadata, Path(str(work_raw) + ".frame_major.tmp")):
            if path.exists():
                path.unlink()

    contact_sheet = output_root / "all_fire_original_reconstructed_2x2.png"
    build_contact_sheet(contact_rows, contact_sheet)
    payload = {
        "contract": {
            "fields": ["density", "flames"],
            "renderer": "Blender 5.1 Cycles CUDA",
            "material": "fire",
            "resolution": [args.size, args.size],
            "samples": args.samples,
            "camera": "same selected source VDB reference for original and reconstructed",
        },
        "all_nonblank": all(
            item["original_nonblank"]["nonblank"]
            and item["reconstructed_nonblank"]["nonblank"]
            for item in results
        ),
        "all_density_frames_closest": all(
            item["density_alignment"]["selected_is_closest"] for item in results
        ),
        "all_flame_frames_closest": all(
            item["flame_alignment"]["selected_is_closest"] for item in results
        ),
        "contact_sheet": str(contact_sheet.relative_to(root)),
        "datasets": results,
    }
    report = report_root / "fire_campaign_summary.json"
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for directory in (working_root, verify_root):
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
    print(
        json.dumps(
            {
                "report": str(report.relative_to(root)),
                "contact_sheet": str(contact_sheet.relative_to(root)),
                "all_nonblank": payload["all_nonblank"],
                "all_density_frames_closest": payload["all_density_frames_closest"],
                "all_flame_frames_closest": payload["all_flame_frames_closest"],
            },
            indent=2,
        )
    )
    return 0 if payload["all_nonblank"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
