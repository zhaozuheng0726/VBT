#!/usr/bin/env python3
"""Compare two rendered images with a fixed, reproducible metric contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    reference = load_rgb(args.reference)
    candidate = load_rgb(args.candidate)
    if reference.shape != candidate.shape:
        raise ValueError(f"image shape mismatch: {reference.shape} != {candidate.shape}")

    difference = reference - candidate
    mse = float(np.mean(np.square(difference), dtype=np.float64))
    metrics = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "width": int(reference.shape[1]),
        "height": int(reference.shape[0]),
        "mse": mse,
        "psnr_db": math.inf if mse == 0.0 else -10.0 * math.log10(mse),
        "ssim": float(structural_similarity(reference, candidate, channel_axis=2, data_range=1.0)),
        "mean_absolute_error": float(np.mean(np.abs(difference), dtype=np.float64)),
        "max_absolute_error": float(np.max(np.abs(difference))),
    }

    output = json.dumps(metrics, indent=2, ensure_ascii=True)
    print(output)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
