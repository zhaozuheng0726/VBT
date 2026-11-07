#!/usr/bin/env python3
"""Extract exact VBTPACK4 mode and payload statistics from canonical assets."""

from __future__ import annotations

import csv
import json
import mmap
import struct
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data" / "catalog.json"
PACKAGE = ROOT / "reports" / "thesis_package_20260717"
TABLES = PACKAGE / "tables" / "modes"
FIGURES = PACKAGE / "figures" / "modes"

HEADER = struct.Struct("<8s11I")
UINT32 = struct.Struct("<I")
PREFIX = struct.Struct("<HH")
HEADER_BYTES = HEADER.size
MODE_NAMES = {
    0: "Empty",
    1: "TemporalGrid4",
    2: "TemporalShellVoxel (Mode2)",
    3: "TemporalFine6",
}
MODE_COLORS = {
    0: "#d8d8d8",
    1: "#1f6f78",
    2: "#3d7d4f",
    3: "#d97724",
}
DATASET_LABELS = {
    "industrial_chimney": "Industrial Chimney",
    "tornado_looping": "Tornado Looping",
    "dust_devil": "Dust Devil",
    "ground_explosion": "Ground Explosion",
    "smoke_plume": "Smoke Plume",
    "dust_shockwave": "Dust Shockwave",
    "aerial_explosion": "Aerial Explosion",
    "quality_0p5": "Fluid Quality 0.5",
    "performance_0p75": "Fluid Performance 0.75",
}


def align4(value: int) -> int:
    return (value + 3) & ~3


def percentile(values: list[int], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def selected_assets() -> list[dict[str, object]]:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    assets = []
    for asset in catalog["assets"]:
        # Fire density entries are hard links to the smoke density assets.
        if asset["category"] == "fire" and asset["field"] == "density":
            continue
        assets.append(asset)
    return assets


def display_label(asset: dict[str, object]) -> str:
    dataset = DATASET_LABELS[str(asset["dataset"])]
    field = str(asset["field"])
    if field == "density" or field == "levelset":
        return dataset
    return f"{dataset} / {field}"


def parse_mode2_leaf(data: mmap.mmap, leaf_start: int, leaf_bytes: int) -> dict[str, int]:
    if leaf_bytes < 88:
        raise ValueError("Mode2 leaf is too short")
    packed_header = UINT32.unpack_from(data, leaf_start)[0]
    coarse_codec = (packed_header >> 2) & 0x3
    fine_codec = (packed_header >> 4) & 0x3
    coarse_resolution = (packed_header >> 6) & 0x7
    coarse_keys, fine_keys = PREFIX.unpack_from(data, leaf_start + 4)
    if coarse_codec not in (0, 1) or fine_codec not in (0, 1):
        raise ValueError("Unsupported Mode2 sequence codec")
    if coarse_resolution == 0:
        raise ValueError("Mode2 coarse resolution is zero")

    coarse_controls = coarse_resolution ** 3
    coarse_descriptor_bytes = coarse_controls * 4
    coarse_bin_bytes = align4(coarse_controls * 8)
    coarse_key_bytes = align4(coarse_keys * (3 if coarse_codec == 0 else 8))
    occupancy_offset = 8 + coarse_descriptor_bytes + coarse_bin_bytes + coarse_key_bytes
    if occupancy_offset + 80 > leaf_bytes:
        raise ValueError("Mode2 occupancy section is out of bounds")

    masks = struct.unpack_from("<8Q", data, leaf_start + occupancy_offset)
    prefixes = struct.unpack_from("<8H", data, leaf_start + occupancy_offset + 64)
    active_voxels = 0
    expected_prefix = 0
    for mask, prefix in zip(masks, prefixes):
        if prefix != expected_prefix:
            raise ValueError("Mode2 occupancy prefix is inconsistent")
        active = int(mask).bit_count()
        active_voxels += active
        expected_prefix += active
    if active_voxels == 0:
        raise ValueError("Mode2 leaf has no active shell voxels")

    residual_offset = align4(occupancy_offset + 80)
    descriptor_bytes = active_voxels * 4
    bin_bytes = align4(active_voxels * 4)
    key_bytes = align4(fine_keys * (3 if fine_codec == 0 else 8))
    expected_bytes = residual_offset + descriptor_bytes + bin_bytes + key_bytes
    if expected_bytes != leaf_bytes:
        raise ValueError(
            f"Mode2 leaf size mismatch: expected {expected_bytes}, found {leaf_bytes}"
        )
    return {
        "active_voxels": active_voxels,
        "coarse_keys": coarse_keys,
        "fine_keys": fine_keys,
        "occupancy_bytes": 80,
        "residual_bytes": descriptor_bytes + bin_bytes + key_bytes,
    }


def analyze_asset(asset: dict[str, object]) -> dict[str, object]:
    path = ROOT / str(asset["vbt"])
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            file_bytes = len(data)
            values = HEADER.unpack_from(data, 0)
            magic = values[0]
            (
                version,
                width,
                height,
                depth,
                frames,
                leaf_size,
                coarse_resolution,
                max_coarse_keep,
                leaf_count,
                profile_type,
                coarse_scale_count,
            ) = values[1:]
            if magic != b"VBTPACK4" or version != 4 or leaf_size != 8:
                raise ValueError(f"Unsupported VBT header: {path}")
            expected_leaves = (
                ((width + leaf_size - 1) // leaf_size)
                * ((height + leaf_size - 1) // leaf_size)
                * ((depth + leaf_size - 1) // leaf_size)
            )
            if leaf_count != expected_leaves:
                raise ValueError(f"Leaf count mismatch: {path}")

            offset_position = HEADER_BYTES + coarse_scale_count * 4
            offsets = np.frombuffer(
                data,
                dtype="<u4",
                count=leaf_count + 1,
                offset=offset_position,
            ).copy()
            if offsets[0] != 0 or np.any(offsets[1:] <= offsets[:-1]):
                raise ValueError(f"Invalid offset table: {path}")
            payload_position = offset_position + (leaf_count + 1) * 4
            expected_file_bytes = payload_position + int(offsets[-1]) * 4
            if expected_file_bytes != len(data):
                raise ValueError(f"File size mismatch: {path}")

            modes = np.empty(leaf_count, dtype=np.uint8)
            mode_counts: Counter[int] = Counter()
            mode_bytes: Counter[int] = Counter()
            mode2_active_voxels: list[int] = []
            mode2_fine_keys: list[int] = []
            mode2_coarse_keys: list[int] = []
            mode2_residual_bytes = 0
            mode2_occupancy_bytes = 0
            for leaf_index in range(leaf_count):
                start_word = int(offsets[leaf_index])
                end_word = int(offsets[leaf_index + 1])
                leaf_bytes = (end_word - start_word) * 4
                leaf_start = payload_position + start_word * 4
                packed_header = UINT32.unpack_from(data, leaf_start)[0]
                mode = packed_header & 0x3
                modes[leaf_index] = mode
                mode_counts[mode] += 1
                mode_bytes[mode] += leaf_bytes
                if mode == 0 and leaf_bytes != 4:
                    raise ValueError(f"Empty leaf is not 4 bytes: {path} leaf {leaf_index}")
                if mode == 2:
                    stats = parse_mode2_leaf(data, leaf_start, leaf_bytes)
                    mode2_active_voxels.append(stats["active_voxels"])
                    mode2_fine_keys.append(stats["fine_keys"])
                    mode2_coarse_keys.append(stats["coarse_keys"])
                    mode2_residual_bytes += stats["residual_bytes"]
                    mode2_occupancy_bytes += stats["occupancy_bytes"]

    leaf_grid = (
        (width + leaf_size - 1) // leaf_size,
        (height + leaf_size - 1) // leaf_size,
        (depth + leaf_size - 1) // leaf_size,
    )
    payload_bytes = int(offsets[-1]) * 4
    overhead_bytes = payload_position
    return {
        "asset": asset,
        "path": path,
        "label": display_label(asset),
        "width": width,
        "height": height,
        "depth": depth,
        "frames": frames,
        "leaf_count": leaf_count,
        "leaf_grid": leaf_grid,
        "profile_type": profile_type,
        "coarse_resolution": coarse_resolution,
        "max_coarse_keep": max_coarse_keep,
        "file_bytes": file_bytes,
        "payload_bytes": payload_bytes,
        "overhead_bytes": overhead_bytes,
        "mode_counts": mode_counts,
        "mode_bytes": mode_bytes,
        "modes": modes,
        "mode2_active_voxels": mode2_active_voxels,
        "mode2_fine_keys": mode2_fine_keys,
        "mode2_coarse_keys": mode2_coarse_keys,
        "mode2_residual_bytes": mode2_residual_bytes,
        "mode2_occupancy_bytes": mode2_occupancy_bytes,
    }


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(results: list[dict[str, object]]) -> None:
    detail_rows = []
    summary_rows = []
    mode2_rows = []
    for result in results:
        asset = result["asset"]
        leaf_count = int(result["leaf_count"])
        payload_bytes = int(result["payload_bytes"])
        active_count = leaf_count - int(result["mode_counts"][0])
        summary_rows.append({
            "category": asset["category"],
            "dataset": asset["dataset"],
            "field": asset["field"],
            "label": result["label"],
            "dimensions": f"{result['width']}x{result['height']}x{result['depth']}x{result['frames']}",
            "leaf_count": leaf_count,
            "active_leaf_count": active_count,
            "active_leaf_percent": 100.0 * active_count / leaf_count,
            "file_bytes": result["file_bytes"],
            "payload_bytes": payload_bytes,
            "overhead_bytes": result["overhead_bytes"],
            "overhead_percent": 100.0 * int(result["overhead_bytes"]) / int(result["file_bytes"]),
        })
        for mode in range(4):
            count = int(result["mode_counts"][mode])
            byte_count = int(result["mode_bytes"][mode])
            detail_rows.append({
                "category": asset["category"],
                "dataset": asset["dataset"],
                "field": asset["field"],
                "label": result["label"],
                "mode_id": mode,
                "mode": MODE_NAMES[mode],
                "leaf_count": count,
                "leaf_percent": 100.0 * count / leaf_count,
                "payload_bytes": byte_count,
                "payload_percent": 100.0 * byte_count / payload_bytes,
                "mean_leaf_bytes": byte_count / count if count else 0.0,
            })
        if result["mode2_active_voxels"]:
            active = result["mode2_active_voxels"]
            fine_keys = result["mode2_fine_keys"]
            coarse_keys = result["mode2_coarse_keys"]
            mode2_bytes = int(result["mode_bytes"][2])
            mode2_rows.append({
                "profile": asset["dataset"],
                "label": result["label"],
                "mode2_leaves": len(active),
                "shell_voxels_mean": np.mean(active),
                "shell_voxels_p50": percentile(active, 50),
                "shell_voxels_p90": percentile(active, 90),
                "shell_voxels_p99": percentile(active, 99),
                "shell_voxels_max": max(active),
                "shell_occupancy_percent_mean": 100.0 * np.mean(active) / 512.0,
                "coarse_keys_mean": np.mean(coarse_keys),
                "fine_keys_mean": np.mean(fine_keys),
                "fine_keys_per_shell_voxel_mean": np.mean(
                    np.asarray(fine_keys, dtype=np.float64) / np.asarray(active, dtype=np.float64)
                ),
                "mode2_payload_mib": mode2_bytes / (1024.0 * 1024.0),
                "occupancy_table_mib": int(result["mode2_occupancy_bytes"]) / (1024.0 * 1024.0),
                "residual_stream_mib": int(result["mode2_residual_bytes"]) / (1024.0 * 1024.0),
            })

    write_csv(TABLES / "render_mode_distribution.csv", list(detail_rows[0]), detail_rows)
    write_csv(TABLES / "render_asset_mode_summary.csv", list(summary_rows[0]), summary_rows)
    write_csv(TABLES / "mode2_internal_stats.csv", list(mode2_rows[0]), mode2_rows)

    compact_latex = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Asset & Active leaves & High mode & High-mode leaves & High-mode bytes \\\\",
        "\\midrule",
    ]
    for result in results:
        asset = result["asset"]
        leaf_count = int(result["leaf_count"])
        active = leaf_count - int(result["mode_counts"][0])
        high_mode = 2 if asset["category"] == "fluid" else 3
        high_label = "Mode2" if high_mode == 2 else "Fine6"
        high_count = int(result["mode_counts"][high_mode])
        high_bytes = int(result["mode_bytes"][high_mode])
        compact_latex.append(
            f"{result['label']} & {100.0 * active / leaf_count:.1f}\\% & {high_label} & "
            f"{100.0 * high_count / leaf_count:.1f}\\% & "
            f"{100.0 * high_bytes / int(result['payload_bytes']):.1f}\\% \\\\"
        )
    compact_latex.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "render_mode_summary_compact.tex").write_text(
        "\n".join(compact_latex), encoding="utf-8"
    )

    latex = [
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Asset & Empty & Grid4 & Mode2 & Fine6 & Empty bytes & Grid4 bytes & Mode2 bytes & Fine6 bytes \\\\",
        "\\midrule",
    ]
    for result in results:
        leaf_count = int(result["leaf_count"])
        payload_bytes = int(result["payload_bytes"])
        leaf = [100.0 * int(result["mode_counts"][mode]) / leaf_count for mode in range(4)]
        byte = [100.0 * int(result["mode_bytes"][mode]) / payload_bytes for mode in range(4)]
        latex.append(
            f"{result['label']} & "
            + " & ".join(f"{value:.1f}\\%" for value in leaf)
            + " & "
            + " & ".join(f"{value:.1f}\\%" for value in byte)
            + " \\\\"
        )
    latex.extend(["\\bottomrule", "\\end{tabular}", ""])
    (TABLES / "render_mode_distribution.tex").write_text("\n".join(latex), encoding="utf-8")


def validate_cache_counts(results: list[dict[str, object]]) -> None:
    by_key = {
        (str(result["asset"]["dataset"]), str(result["asset"]["field"])): result
        for result in results
    }
    contracts = {
        "industrial_chimney": [("industrial_chimney", "density")],
        "ground_physical_fire": [("ground_explosion", "density"), ("ground_explosion", "flames")],
        "aerial_physical_fire": [
            ("aerial_explosion", "density"),
            ("aerial_explosion", "flames"),
            ("aerial_explosion", "temperature"),
        ],
        "fluid_performance_fast": [("performance_0p75", "levelset")],
        "fluid_quality_fast": [("quality_0p5", "levelset")],
    }
    measured = {}
    source = PACKAGE / "tables" / "studio" / "cache_memory.csv"
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            measured[row["contract"]] = row
    rows = []
    for contract, keys in contracts.items():
        expected_total = sum(int(by_key[key]["leaf_count"]) for key in keys)
        expected_active = sum(
            int(by_key[key]["leaf_count"]) - int(by_key[key]["mode_counts"][0])
            for key in keys
        )
        actual = measured[contract]
        actual_total = int(actual["total_leaves"])
        actual_active = int(actual["active_leaves"])
        exact = expected_total == actual_total and expected_active == actual_active
        rows.append({
            "contract": contract,
            "mode_parser_total_leaves": expected_total,
            "studio_total_leaves": actual_total,
            "mode_parser_active_leaves": expected_active,
            "studio_active_leaves": actual_active,
            "exact_match": exact,
        })
        if not exact:
            raise ValueError(f"Mode counts do not match VBTStudio cache contract: {contract}")
    write_csv(TABLES / "mode_cache_consistency.csv", list(rows[0]), rows)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": "#555555",
        "axes.titleweight": "bold",
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def plot_distribution(results: list[dict[str, object]]) -> None:
    labels = [str(result["label"]) for result in results]
    y = np.arange(len(results))
    leaf_values = np.array([
        [100.0 * int(result["mode_counts"][mode]) / int(result["leaf_count"]) for mode in range(4)]
        for result in results
    ])
    byte_values = np.array([
        [100.0 * int(result["mode_bytes"][mode]) / int(result["payload_bytes"]) for mode in range(4)]
        for result in results
    ])
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.4), sharey=True)
    titles = ["Leaf-count share", "Payload-byte share"]
    for ax, values, title in zip(axes, (leaf_values, byte_values), titles):
        left = np.zeros(len(results))
        for mode in range(4):
            ax.barh(
                y,
                values[:, mode],
                left=left,
                color=MODE_COLORS[mode],
                edgecolor="white",
                linewidth=0.5,
                label=MODE_NAMES[mode],
            )
            for row, value in enumerate(values[:, mode]):
                if value >= 8.0:
                    color = "#222222" if mode == 0 else "white"
                    ax.text(left[row] + value / 2.0, row, f"{value:.0f}%", ha="center", va="center", color=color, fontsize=8)
            left += values[:, mode]
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share (%)")
        ax.set_title(title)
        ax.xaxis.grid(True, color="#e3e3e3", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.suptitle("VBTPACK4 render and Mode2 routing distribution", fontsize=16, fontweight="bold", y=0.985)
    fig.text(0.5, 0.052, "Fire density hard links are counted once through the smoke density assets.", ha="center", fontsize=9, color="#555555")
    fig.subplots_adjust(left=0.24, right=0.985, top=0.92, bottom=0.12, wspace=0.08)
    fig.savefig(FIGURES / "render_mode_distribution.png", dpi=220, facecolor="white")
    plt.close(fig)


def best_mode_slice(result: dict[str, object]) -> tuple[np.ndarray, int]:
    nx, ny, nz = result["leaf_grid"]
    grid = np.asarray(result["modes"], dtype=np.uint8).reshape(nz, ny, nx)
    active_by_slice = np.count_nonzero(grid, axis=(1, 2))
    best_z = int(np.argmax(active_by_slice))
    return grid[best_z], best_z


def plot_spatial_slices(results: list[dict[str, object]]) -> None:
    wanted = [
        ("ground_explosion", "density"),
        ("smoke_plume", "density"),
        ("aerial_explosion", "flames"),
    ]
    selected = []
    for dataset, field in wanted:
        selected.append(next(
            result for result in results
            if result["asset"]["dataset"] == dataset and result["asset"]["field"] == field
        ))
    cmap = ListedColormap([MODE_COLORS[mode] for mode in range(4)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.0))
    for ax, result in zip(axes, selected):
        mode_slice, z_index = best_mode_slice(result)
        ax.imshow(mode_slice, origin="lower", interpolation="nearest", cmap=cmap, norm=norm, aspect="equal")
        ax.set_title(f"{result['label']}\nleaf Z slice {z_index}", fontsize=11)
        ax.set_xlabel("Leaf X")
        ax.set_ylabel("Leaf Y")
        ax.tick_params(labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODE_COLORS[mode]) for mode in (0, 1, 3)]
    labels = [MODE_NAMES[mode] for mode in (0, 1, 3)]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Spatial routing slices at the most active leaf plane", fontsize=16, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.84, bottom=0.16, wspace=0.24)
    fig.savefig(FIGURES / "render_mode_spatial_slices.png", dpi=220, facecolor="white")
    plt.close(fig)


def plot_mode2_utilization(results: list[dict[str, object]]) -> None:
    mode2_results = [result for result in results if result["mode2_active_voxels"]]
    bins = np.array([0, 32, 64, 128, 256, 384, 512])
    bin_labels = ["1-32", "33-64", "65-128", "129-256", "257-384", "385-512"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = np.arange(len(bin_labels))
    width = 0.36
    colors = ["#1f6f78", "#d97724"]
    for index, result in enumerate(mode2_results):
        values = np.asarray(result["mode2_active_voxels"], dtype=np.int32)
        counts, _ = np.histogram(values, bins=bins)
        shares = counts * 100.0 / len(values)
        axes[0].bar(x + (index - 0.5) * width, shares, width, label=result["label"], color=colors[index])
    axes[0].set_xticks(x, bin_labels)
    axes[0].set_xlabel("Active shell voxels per Mode2 leaf")
    axes[0].set_ylabel("Mode2 leaves (%)")
    axes[0].set_title("Shell occupancy distribution")
    axes[0].yaxis.grid(True, color="#e3e3e3", linewidth=0.7)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False)

    labels = [result["label"].replace("Fluid ", "") for result in mode2_results]
    payload = [int(result["mode_bytes"][2]) / (1024.0 * 1024.0) for result in mode2_results]
    occupancy = [int(result["mode2_occupancy_bytes"]) / (1024.0 * 1024.0) for result in mode2_results]
    residual = [int(result["mode2_residual_bytes"]) / (1024.0 * 1024.0) for result in mode2_results]
    base = [total - event - mask for total, event, mask in zip(payload, residual, occupancy)]
    axes[1].bar(labels, residual, color="#d97724", label="Shell residual stream")
    axes[1].bar(labels, occupancy, bottom=residual, color="#1f6f78", label="Occupancy table")
    axes[1].bar(
        labels,
        base,
        bottom=np.asarray(residual) + np.asarray(occupancy),
        color="#555555",
        label="Header + coarse stream",
    )
    for i, value in enumerate(payload):
        axes[1].text(i, value, f"{value:.1f} MiB", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("Payload (MiB)")
    axes[1].set_title("Mode2 payload components")
    axes[1].yaxis.grid(True, color="#e3e3e3", linewidth=0.7)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Mode2 shell utilization and storage", fontsize=16, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.16, wspace=0.25)
    fig.savefig(FIGURES / "mode2_internal_utilization.png", dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_style()
    results = []
    for asset in selected_assets():
        print(f"Analyzing {asset['vbt']} ...", flush=True)
        results.append(analyze_asset(asset))
    write_tables(results)
    validate_cache_counts(results)
    plot_distribution(results)
    plot_spatial_slices(results)
    plot_mode2_utilization(results)
    print(f"Analyzed {len(results)} unique assets")
    print(f"Tables: {TABLES}")
    print(f"Figures: {FIGURES}")


if __name__ == "__main__":
    main()
