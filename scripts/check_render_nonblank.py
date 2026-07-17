#!/usr/bin/env python3
"""Check whether rendered images contain meaningful visible foreground."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


THRESHOLDS = {
    "numeric": {
        "stddev": 0.001,
        "dynamic_range": 0.01,
        "foreground_fraction": 0.0001,
    },
    "presentation": {
        "stddev": 0.01,
        "dynamic_range": 0.08,
        "foreground_fraction": 0.01,
    },
}


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0


def corner_background(image: np.ndarray) -> np.ndarray:
    height, width, _ = image.shape
    patch = max(4, min(height, width) // 20)
    corners = np.concatenate(
        (
            image[:patch, :patch].reshape(-1, 3),
            image[:patch, -patch:].reshape(-1, 3),
            image[-patch:, :patch].reshape(-1, 3),
            image[-patch:, -patch:].reshape(-1, 3),
        ),
        axis=0,
    )
    return np.median(corners, axis=0)


def inspect_image(path: Path, profile: str) -> dict[str, object]:
    image = load_rgb(path)
    grayscale = np.mean(image, axis=2)
    background = corner_background(image)
    foreground_distance = np.linalg.norm(image - background, axis=2) / np.sqrt(3.0)

    stddev = float(np.std(grayscale, dtype=np.float64))
    low = float(np.percentile(grayscale, 0.5))
    high = float(np.percentile(grayscale, 99.5))
    dynamic_range = high - low
    foreground_fraction = float(np.mean(foreground_distance > 0.03))
    thresholds = THRESHOLDS[profile]
    checks = {
        "stddev": stddev >= thresholds["stddev"],
        "dynamic_range": dynamic_range >= thresholds["dynamic_range"],
        "foreground_fraction": foreground_fraction >= thresholds["foreground_fraction"],
    }

    return {
        "path": str(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "profile": profile,
        "mean": float(np.mean(grayscale, dtype=np.float64)),
        "stddev": stddev,
        "percentile_0_5": low,
        "percentile_99_5": high,
        "dynamic_range": dynamic_range,
        "corner_background_rgb": [float(value) for value in background],
        "foreground_fraction": foreground_fraction,
        "thresholds": thresholds,
        "checks": checks,
        "nonblank": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--profile", choices=tuple(THRESHOLDS), default="presentation")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    missing = [path for path in args.images if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))

    results = [inspect_image(path, args.profile) for path in args.images]
    payload = {
        "profile": args.profile,
        "all_nonblank": all(bool(result["nonblank"]) for result in results),
        "images": results,
    }
    output = json.dumps(payload, indent=2, ensure_ascii=True)
    print(output)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(output + "\n", encoding="utf-8")
    return 0 if payload["all_nonblank"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
