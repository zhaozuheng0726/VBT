#!/usr/bin/env python3
"""Render aligned end-to-end and compression-only fluid candidate comparisons."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "outputs/fluid_highres_candidates_20260715"
BLENDER = Path(r"C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
BLENDER_SCRIPT = REPO / "3D/blender_render_water_pair.py"
COMPARE_IMAGES = REPO / "scripts/compare_render_images.py"
ABC = REPO / "3D/data/fluid/source/water_flow.abc"
FRAMES = (0, 36, 72, 108, 140)
CONFIGS = ("voxel_0p75", "voxel_0p5")


def run(command: list[str], capture: bool = False) -> str:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout if capture else ""


def blender_command(
    original_flag: str,
    original: Path,
    reconstructed: Path,
    source_frame: int,
    original_output: Path,
    reconstructed_output: Path,
    skip_original: bool,
    resolution: int,
    samples: int,
) -> list[str]:
    command = [
        str(BLENDER), "-b", "-P", str(BLENDER_SCRIPT), "--",
        original_flag, str(original),
        "--input-recon", str(reconstructed),
        "--frame", str(source_frame),
        "--orig-out", str(original_output),
        "--recon-out", str(reconstructed_output),
        "--resolution", str(resolution),
        "--samples", str(samples),
        "--transmission-weight", "1",
        "--water-roughness", "0.008",
        "--absorption-density", "0",
        "--background-preset", "studio",
        "--camera-preset", "hero",
    ]
    if skip_original:
        command.append("--skip-original-render")
    return command


def compare(reference: Path, candidate: Path, output: Path) -> dict:
    run([
        "python", str(COMPARE_IMAGES), "--reference", str(reference),
        "--candidate", str(candidate), "--output-json", str(output),
    ], capture=True)
    return json.loads(output.read_text(encoding="utf-8"))


def image_contract(path: Path) -> dict[str, float | int | bool]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    std = float(np.std(rgb))
    return {
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "mean": float(np.mean(rgb)),
        "stddev": std,
        "minimum": float(np.min(rgb)),
        "maximum": float(np.max(rgb)),
        "nonblank": std > 0.01 and float(np.max(rgb) - np.min(rgb)) > 0.1,
    }


def make_gallery(
    top_paths: list[Path],
    bottom_paths: list[Path],
    top_label: str,
    bottom_label: str,
    output: Path,
) -> None:
    tile = 300
    label_width = 150
    header = 42
    canvas = Image.new("RGB", (label_width + tile * len(top_paths), header + tile * 2), "white")
    draw = ImageDraw.Draw(canvas)
    label_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 18)
    frame_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    draw.text((18, header + tile // 2 - 10), top_label, fill="#202124", font=label_font)
    draw.text((18, header + tile + tile // 2 - 10), bottom_label, fill="#202124", font=label_font)
    for index, frame in enumerate(FRAMES):
        x = label_width + index * tile
        text = f"VBT {frame:04d} / ABC {frame + 1:04d}"
        box = draw.textbbox((0, 0), text, font=frame_font)
        draw.text((x + (tile - (box[2] - box[0])) / 2, 12), text, fill="#3c4043", font=frame_font)
        for row, path in enumerate((top_paths[index], bottom_paths[index])):
            with Image.open(path) as image:
                panel = image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(panel, (x, header + row * tile))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def mean_metrics(rows: list[dict]) -> dict[str, float]:
    return {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in ("psnr_db", "ssim", "mean_absolute_error")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    root = args.root.resolve()
    shared_source = root / "renders/source"
    shared_source.mkdir(parents=True, exist_ok=True)
    summary = {"frames": list(FRAMES), "resolution": args.resolution, "samples": args.samples, "candidates": {}}

    for config_index, config_name in enumerate(CONFIGS):
        validation = root / config_name / "validation"
        render_root = root / config_name / "renders"
        end_root = render_root / "end_to_end"
        compression_root = render_root / "compression_only"
        end_root.mkdir(parents=True, exist_ok=True)
        compression_root.mkdir(parents=True, exist_ok=True)
        end_metrics = []
        compression_metrics = []
        contracts = []

        source_paths = []
        end_mode2_paths = []
        levelset_paths = []
        compression_mode2_paths = []
        for frame in FRAMES:
            source = shared_source / f"frame_{frame:04d}.png"
            mode2_obj = validation / f"frame_{frame:04d}/mode2_levelset.obj"
            original_obj = validation / f"frame_{frame:04d}/original_levelset.obj"
            end_mode2 = end_root / f"frame_{frame:04d}_mode2.png"
            levelset = compression_root / f"frame_{frame:04d}_original.png"
            compression_mode2 = compression_root / f"frame_{frame:04d}_mode2.png"

            run(blender_command(
                "--input-abc", ABC, mode2_obj, frame + 1, source, end_mode2,
                skip_original=config_index > 0, resolution=args.resolution, samples=args.samples,
            ))
            run(blender_command(
                "--input-original", original_obj, mode2_obj, frame + 1, levelset, compression_mode2,
                skip_original=False, resolution=args.resolution, samples=args.samples,
            ))
            end_metrics.append(compare(
                source, end_mode2, end_root / f"frame_{frame:04d}_metrics.json"
            ))
            compression_metrics.append(compare(
                levelset, compression_mode2, compression_root / f"frame_{frame:04d}_metrics.json"
            ))
            for path in (source, end_mode2, levelset, compression_mode2):
                contract = image_contract(path)
                contract["path"] = str(path)
                if not contract["nonblank"]:
                    raise RuntimeError(f"blank render: {path}")
                contracts.append(contract)
            source_paths.append(source)
            end_mode2_paths.append(end_mode2)
            levelset_paths.append(levelset)
            compression_mode2_paths.append(compression_mode2)
            print(f"rendered {config_name} frame {frame}", flush=True)

        make_gallery(
            source_paths, end_mode2_paths, "Alembic", "Mode2",
            render_root / "end_to_end_5x2.png",
        )
        make_gallery(
            levelset_paths, compression_mode2_paths, "Level set", "Mode2",
            render_root / "compression_only_5x2.png",
        )
        candidate_summary = {
            "end_to_end": {"per_frame": end_metrics, "mean": mean_metrics(end_metrics)},
            "compression_only": {
                "per_frame": compression_metrics,
                "mean": mean_metrics(compression_metrics),
            },
            "image_contracts": contracts,
            "all_nonblank": all(row["nonblank"] for row in contracts),
        }
        summary["candidates"][config_name] = candidate_summary
        (render_root / "render_summary.json").write_text(
            json.dumps(candidate_summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )

    output = root / "render_summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "candidates": list(CONFIGS)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
