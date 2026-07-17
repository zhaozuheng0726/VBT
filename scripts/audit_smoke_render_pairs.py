#!/usr/bin/env python3
"""Audit aligned smoke renders and rebuild their quality reports."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

from check_render_nonblank import inspect_image


DATASETS = (
    ("industrial_chimney", 125, "Industrial Chimney"),
    ("tornado_looping", 64, "Tornado Looping"),
    ("dust_devil", 50, "Dust Devil"),
    ("ground_explosion", 100, "Ground Explosion"),
    ("smoke_plume", 83, "Smoke Plume"),
    ("dust_shockwave", 77, "Dust Shockwave"),
    ("aerial_explosion", 60, "Aerial Explosion"),
)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def compare_images(reference_path: Path, candidate_path: Path) -> dict[str, object]:
    reference = load_rgb(reference_path)
    candidate = load_rgb(candidate_path)
    if reference.shape != candidate.shape:
        raise ValueError(f"image shape mismatch: {reference.shape} != {candidate.shape}")

    difference = reference - candidate
    mse = float(np.mean(np.square(difference), dtype=np.float64))
    return {
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "width": int(reference.shape[1]),
        "height": int(reference.shape[0]),
        "mse": mse,
        "psnr_db": math.inf if mse == 0.0 else -10.0 * math.log10(mse),
        "ssim": float(
            structural_similarity(reference, candidate, channel_axis=2, data_range=1.0)
        ),
        "mean_absolute_error": float(np.mean(np.abs(difference), dtype=np.float64)),
        "max_absolute_error": float(np.max(np.abs(difference))),
    }


def read_render_log(path: Path) -> dict[str, object]:
    contents = path.read_bytes()
    encoding = "utf-16" if contents.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    lines = contents.decode(encoding, errors="replace").splitlines()
    device = next((line for line in lines if line.startswith("CYCLES_DEVICE ")), None)
    render_line = next((line for line in lines if line.startswith("VDB_RENDER ")), None)
    if device is None or render_line is None:
        raise ValueError(f"missing render metadata in {path}")
    metadata = ast.literal_eval(render_line.removeprefix("VDB_RENDER "))
    metadata["cycles_device"] = device.removeprefix("CYCLES_DEVICE ")
    return metadata


def alignment_checks(original: dict[str, object], reconstructed: dict[str, object]) -> dict[str, bool]:
    return {
        "camera_reference": original["camera_reference"] == reconstructed["camera_reference"],
        "camera_bbox_min": original["bbox_min"] == reconstructed["bbox_min"],
        "camera_bbox_max": original["bbox_max"] == reconstructed["bbox_max"],
        "cycles_device": original["cycles_device"] == reconstructed["cycles_device"],
    }


def build_contact_sheet(
    rows: list[dict[str, object]],
    output_path: Path,
    headers: tuple[str, str] = ("Original", "Reconstructed"),
) -> None:
    tile_size = 384
    title_height = 34
    header_height = 42
    sheet = Image.new("RGB", (tile_size * 2, header_height + len(rows) * (tile_size + title_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    header_font = ImageFont.load_default(size=22)
    draw.text((tile_size // 2, 10), headers[0], fill="black", font=header_font, anchor="ma")
    draw.text((tile_size + tile_size // 2, 10), headers[1], fill="black", font=header_font, anchor="ma")

    for index, row in enumerate(rows):
        top = header_height + index * (tile_size + title_height)
        label = f"{row['label']} | frame {row['frame']:04d}"
        for column, key in enumerate(("reference", "candidate")):
            with Image.open(row[key]) as source:
                tile = source.convert("RGB").resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            sheet.paste(tile, (column * tile_size, top))
            draw.text(
                (column * tile_size + tile_size // 2, top + tile_size + 7),
                label,
                fill="black",
                font=font,
                anchor="ma",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--renders-root",
        type=Path,
        default=Path("outputs/thesis_render_extension_20260711"),
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=Path("reports/native_smoke_campaign_20260714"),
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=Path(
            "outputs/native_smoke_campaign_20260714/"
            "all_datasets_original_reconstructed_7x2.png"
        ),
    )
    args = parser.parse_args()

    results = []
    contact_rows = []
    for key, frame, label in DATASETS:
        dataset_dir = args.renders_root / key
        prefix = f"frame{frame:04d}"
        reference_path = dataset_dir / f"{prefix}_original_cycles.png"
        candidate_path = dataset_dir / f"{prefix}_reconstructed_cycles.png"
        paths = (
            reference_path,
            candidate_path,
            dataset_dir / "render_original.log",
            dataset_dir / "render_reconstructed.log",
        )
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(", ".join(missing))

        metrics = compare_images(reference_path, candidate_path)
        original_nonblank = inspect_image(reference_path, "presentation")
        reconstructed_nonblank = inspect_image(candidate_path, "presentation")
        original_log = read_render_log(dataset_dir / "render_original.log")
        reconstructed_log = read_render_log(dataset_dir / "render_reconstructed.log")
        checks = alignment_checks(original_log, reconstructed_log)
        checks["image_dimensions"] = (
            original_nonblank["width"] == reconstructed_nonblank["width"]
            and original_nonblank["height"] == reconstructed_nonblank["height"]
        )
        checks["original_nonblank"] = bool(original_nonblank["nonblank"])
        checks["reconstructed_nonblank"] = bool(reconstructed_nonblank["nonblank"])

        metric_path = dataset_dir / f"{prefix}_cycles_metrics.json"
        metric_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        results.append(
            {
                "dataset": key,
                "label": label,
                "frame": frame,
                "reference": str(reference_path),
                "candidate": str(candidate_path),
                "camera_reference": original_log["camera_reference"],
                "camera_bbox_min": original_log["bbox_min"],
                "camera_bbox_max": original_log["bbox_max"],
                "cycles_device": original_log["cycles_device"],
                "checks": checks,
                "aligned_and_nonblank": all(checks.values()),
                "original_nonblank": original_nonblank,
                "reconstructed_nonblank": reconstructed_nonblank,
                "metrics": metrics,
            }
        )
        contact_rows.append(
            {
                "label": label,
                "frame": frame,
                "reference": reference_path,
                "candidate": candidate_path,
            }
        )

    payload = {
        "contract": {
            "renderer": "Blender 5.1 Cycles CUDA",
            "resolution": [512, 512],
            "samples": 32,
            "density_scale": 8.0,
            "step_size": 1.0,
            "nonblank_profile": "presentation",
            "alignment": "same camera reference and reference bounds",
        },
        "all_aligned_and_nonblank": all(item["aligned_and_nonblank"] for item in results),
        "datasets": results,
    }
    args.report_root.mkdir(parents=True, exist_ok=True)
    json_path = args.report_root / "aligned_pairs_audit.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    csv_path = args.report_root / "aligned_pairs_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "dataset",
                "frame",
                "width",
                "height",
                "psnr_db",
                "ssim",
                "mean_absolute_error",
                "max_absolute_error",
                "aligned_and_nonblank",
            ),
        )
        writer.writeheader()
        for item in results:
            metrics = item["metrics"]
            writer.writerow(
                {
                    "dataset": item["dataset"],
                    "frame": item["frame"],
                    "width": metrics["width"],
                    "height": metrics["height"],
                    "psnr_db": metrics["psnr_db"],
                    "ssim": metrics["ssim"],
                    "mean_absolute_error": metrics["mean_absolute_error"],
                    "max_absolute_error": metrics["max_absolute_error"],
                    "aligned_and_nonblank": item["aligned_and_nonblank"],
                }
            )

    build_contact_sheet(contact_rows, args.contact_sheet)
    print(json.dumps({"report": str(json_path), "csv": str(csv_path), "contact_sheet": str(args.contact_sheet), "all_aligned_and_nonblank": payload["all_aligned_and_nonblank"]}, indent=2))
    return 0 if payload["all_aligned_and_nonblank"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
