#!/usr/bin/env python3
"""Verify a reconstructed VDB against its selected and neighboring source frames."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _difference_metrics(raw: np.memmap, source_index: int, reconstructed_index: int) -> dict[str, float]:
    count = raw.shape[0]
    chunk_size = 1_000_000
    sum_squared = 0.0
    sum_absolute = 0.0
    max_absolute = 0.0
    source_minimum = math.inf
    source_maximum = -math.inf
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        source = raw[start:stop, source_index].astype(np.float64)
        reconstructed = raw[start:stop, reconstructed_index].astype(np.float64)
        difference = source - reconstructed
        absolute = np.abs(difference)
        sum_squared += float(np.sum(np.square(difference), dtype=np.float64))
        sum_absolute += float(np.sum(absolute, dtype=np.float64))
        max_absolute = max(max_absolute, float(np.max(absolute)))
        source_minimum = min(source_minimum, float(np.min(source)))
        source_maximum = max(source_maximum, float(np.max(source)))
    mse = sum_squared / count
    data_range = source_maximum - source_minimum
    unit_range_psnr = math.inf if mse == 0.0 else -10.0 * math.log10(mse)
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "source_min": source_minimum,
        "source_max": source_maximum,
        "source_data_range": data_range,
        "psnr_db": (
            math.inf
            if mse == 0.0
            else 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)
        ),
        "unit_range_psnr_db": unit_range_psnr,
        "mean_absolute_error": sum_absolute / count,
        "max_absolute_error": max_absolute,
    }


def verify_alignment(
    previous: Path,
    selected: Path,
    next_frame: Path,
    reconstructed: Path,
    vdb_to_raw: Path,
    temp_root: Path,
    grid_name: str = "density",
) -> dict[str, object]:
    inputs = (previous, selected, next_frame, reconstructed, vdb_to_raw)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vbt_frame_alignment_", dir=temp_root) as directory:
        workspace = Path(directory)
        pair_dir = workspace / "frames"
        pair_dir.mkdir()
        for name, source in zip(
            ("00_previous.vdb", "01_selected.vdb", "02_next.vdb", "03_reconstructed.vdb"),
            (previous, selected, next_frame, reconstructed),
            strict=True,
        ):
            _link_or_copy(source, pair_dir / name)

        output_prefix = workspace / "alignment"
        command = (
            str(vdb_to_raw),
            "--input-dir",
            str(pair_dir),
            "--output-prefix",
            str(output_prefix),
            "--grid",
            grid_name,
            "--time-fastest",
        )
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        metadata_path = output_prefix.with_suffix(".metadata.json")
        raw_path = output_prefix.with_suffix(".raw")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["frames"] != 4 or not metadata["time_is_fastest_dimension"]:
            raise ValueError("unexpected VDB-to-raw layout")

        voxel_count = metadata["width"] * metadata["height"] * metadata["depth"]
        raw = np.memmap(raw_path, dtype=np.float32, mode="r", shape=(voxel_count, 4))
        comparisons = {
            "previous_vs_reconstructed": _difference_metrics(raw, 0, 3),
            "selected_vs_reconstructed": _difference_metrics(raw, 1, 3),
            "next_vs_reconstructed": _difference_metrics(raw, 2, 3),
        }
        del raw

    selected_mse = comparisons["selected_vs_reconstructed"]["mse"]
    previous_mse = comparisons["previous_vs_reconstructed"]["mse"]
    next_mse = comparisons["next_vs_reconstructed"]["mse"]
    return {
        "previous": str(previous),
        "selected": str(selected),
        "next": str(next_frame),
        "reconstructed": str(reconstructed),
        "grid": grid_name,
        "dimensions": [metadata["width"], metadata["height"], metadata["depth"]],
        "bbox_min": metadata["bbox_min"],
        "bbox_max": metadata["bbox_max"],
        "comparisons": comparisons,
        "selected_is_closest": selected_mse < previous_mse and selected_mse < next_mse,
        "selected_vs_previous_mse_ratio": selected_mse / previous_mse if previous_mse else 0.0,
        "selected_vs_next_mse_ratio": selected_mse / next_mse if next_mse else 0.0,
        "converter_output": completed.stdout.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--selected", required=True, type=Path)
    parser.add_argument("--next", required=True, type=Path, dest="next_frame")
    parser.add_argument("--reconstructed", required=True, type=Path)
    parser.add_argument("--vdb-to-raw", required=True, type=Path)
    parser.add_argument("--temp-root", required=True, type=Path)
    parser.add_argument("--grid", default="density")
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    result = verify_alignment(
        args.previous,
        args.selected,
        args.next_frame,
        args.reconstructed,
        args.vdb_to_raw,
        args.temp_root,
        args.grid,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["selected_is_closest"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
