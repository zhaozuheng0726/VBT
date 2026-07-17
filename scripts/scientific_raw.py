from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_metadata(path: Path) -> dict:
    meta = json.loads(path.read_text(encoding="utf-8-sig"))
    dimensions(meta)
    return meta


def dimensions(meta: dict) -> tuple[int, int, int, int]:
    dims = meta.get("dimensions")
    if not isinstance(dims, list) or len(dims) != 4:
        raise ValueError("Scientific metadata requires dimensions=[X,Y,Z,T]")
    w, h, d, frames = (int(value) for value in dims)
    if min(w, h, d, frames) <= 0:
        raise ValueError(f"Scientific dimensions must be positive: {dims}")

    axis_order = [str(axis).upper() for axis in meta.get("axis_order", ["X", "Y", "Z", "T"])]
    if axis_order != ["X", "Y", "Z", "T"]:
        raise ValueError(f"Unsupported scientific axis_order={axis_order}")
    return w, h, d, frames


def is_time_fastest(meta: dict) -> bool:
    dimensions(meta)
    return bool(meta.get("time_is_fastest_dimension", False))


def raw_layout_name(meta: dict) -> str:
    return "time-fastest" if is_time_fastest(meta) else "frame-major"


def codec_dimensions(meta: dict) -> tuple[int, int, int, int]:
    w, h, d, frames = dimensions(meta)
    return (frames, w, h, d) if is_time_fastest(meta) else (w, h, d, frames)


def canonical_4d_view(path: Path, meta: dict, mode: str = "r") -> np.ndarray:
    w, h, d, frames = dimensions(meta)
    raw = np.memmap(path, dtype=np.float32, mode=mode)
    expected = w * h * d * frames
    if raw.size != expected:
        raise ValueError(f"Raw size mismatch for {path}: expected {expected} float32 values, got {raw.size}")
    if is_time_fastest(meta):
        return raw.reshape((d, h, w, frames), order="C").transpose(3, 0, 1, 2)
    return raw.reshape((frames, d, h, w), order="C")


def load_frame(path: Path, meta: dict, frame: int, copy: bool = False) -> np.ndarray:
    _, _, _, frames = dimensions(meta)
    if frame < 0 or frame >= frames:
        raise IndexError(f"Frame {frame} is outside [0,{frames})")
    view = canonical_4d_view(path, meta)[frame]
    return np.array(view, copy=copy)
