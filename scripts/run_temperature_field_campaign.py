#!/usr/bin/env python3
"""Build and validate the Aerial Explosion temperature VBT field."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from verify_vdb_frame_alignment import verify_alignment


def run_logged(command: list[str], stdout_path: Path, stderr_path: Path, root: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        subprocess.run(command, cwd=root, check=True, stdout=stdout, stderr=stderr)
    if stderr_path.stat().st_size == 0:
        stderr_path.unlink()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source_dir = root / "3D/data/smoke/AerialExplosionVDB/AerialExplosion/AerialExplosionVDB"
    output_root = root / "outputs/native_fire_campaign_20260714/aerial_explosion"
    compressed = output_root / "compressed"
    reconstructed = output_root / "reconstructed"
    logs = output_root / "logs"
    reports = root / "reports/native_fire_campaign_20260714"
    working = root / "outputs/native_fire_campaign_20260714/_working"
    for directory in (compressed, reconstructed, logs, reports, working):
        directory.mkdir(parents=True, exist_ok=True)

    vdb_to_raw = root / "3D/vdb_tools/build/Release/vdb_to_raw.exe"
    encoder = root / "build/Release/vbt_render_temporal_kf_probe.exe"
    decoder = root / "3D/vdb_tools/build/Release/render_temporal_vbt_to_vdb.exe"
    prefix = working / "aerial_explosion_temperature"
    raw = prefix.with_suffix(".raw")
    working_metadata = prefix.with_suffix(".metadata.json")
    vbt = compressed / "aerial_explosion_temperature.vbtp"
    metadata = compressed / "aerial_explosion_temperature.metadata.json"
    compression_report = reports / "aerial_explosion_temperature_compression.md"

    if not vbt.is_file() or not metadata.is_file():
        run_logged(
            [
                str(vdb_to_raw),
                "--input-dir", str(source_dir),
                "--output-prefix", str(prefix),
                "--grid", "temperature",
                "--time-fastest",
            ],
            logs / "temperature_to_raw.stdout.log",
            logs / "temperature_to_raw.stderr.log",
            root,
        )
        run_logged(
            [
                str(encoder),
                "--input-raw", str(raw),
                "--metadata", str(working_metadata),
                "--sample-step", "8",
                "--control-eps-scale", "2",
                "--packed-keyframe-codec", "fp16",
                "--final-only",
                "--route-empty-visible-thr", "1",
                "--route-fine-visible-thr", "20",
                "--route-fine-band-thr", "20",
                "--route-coarse-rmse-thr", "20",
                "--route-coarse-peak-thr", "100",
                "--route-fine-gain-thr", "0.03",
                "--omp-threads", "16",
                "--save-vbt", str(vbt),
                "--report", str(compression_report),
            ],
            logs / "temperature_encode.stdout.log",
            logs / "temperature_encode.stderr.log",
            root,
        )
        shutil.copy2(working_metadata, metadata)

    document = json.loads(metadata.read_text(encoding="utf-8"))
    if document["grid_name"] != "temperature" or document["frames"] != 120:
        raise ValueError("Unexpected Aerial Explosion temperature metadata")
    raw_bytes = (
        document["width"] * document["height"] * document["depth"] * document["frames"] * 4
    )

    frame = 60
    rebuilt = reconstructed / "frame0060_temperature.vdb"
    if not rebuilt.is_file():
        run_logged(
            [
                str(decoder),
                "--input-vbt", str(vbt),
                "--metadata", str(metadata),
                "--output-vdb", str(rebuilt),
                "--frame", str(frame),
                "--grid-name", "temperature",
            ],
            logs / "temperature_decode.stdout.log",
            logs / "temperature_decode.stderr.log",
            root,
        )

    alignment_path = reports / "aerial_explosion_temperature_alignment.json"
    if alignment_path.is_file():
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    else:
        alignment = verify_alignment(
            source_dir / "AerialExplosion_0059.vdb",
            source_dir / "AerialExplosion_0060.vdb",
            source_dir / "AerialExplosion_0061.vdb",
            rebuilt,
            vdb_to_raw,
            root / "outputs/native_fire_campaign_20260714/_verify_temperature",
            "temperature",
        )
        alignment_path.write_text(json.dumps(alignment, indent=2) + "\n", encoding="utf-8")

    summary = {
        "field": "temperature",
        "frame": frame,
        "data_range": [document["data_min"], document["data_max"]],
        "raw_bytes": raw_bytes,
        "vbt_bytes": vbt.stat().st_size,
        "raw_to_vbt": raw_bytes / vbt.stat().st_size,
        "compression_report": str(compression_report.relative_to(root)),
        "alignment": alignment,
    }
    summary_path = reports / "aerial_explosion_temperature_result.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    for temporary in (raw, working_metadata, Path(str(raw) + ".frame_major.tmp")):
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"summary": str(summary_path), "raw_to_vbt": summary["raw_to_vbt"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
