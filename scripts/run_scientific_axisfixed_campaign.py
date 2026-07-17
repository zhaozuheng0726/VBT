#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from scientific_raw import dimensions, load_metadata, raw_layout_name


VBT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = VBT_ROOT / "3D" / "data" / "test_scenarios"
VBT_EXE = VBT_ROOT / "build" / "Release" / "vbt_spatialfirst_probe.exe"
VALIDATE_EXE = (
    VBT_ROOT
    / "build"
    / "render"
    / "Release"
    / "vbt_query_bench.exe"
)

CAMPAIGN = "scientific_axisfixed_20260710"
OUTPUT_DIR = VBT_ROOT / "outputs" / CAMPAIGN
REPORT_DIR = VBT_ROOT / "reports" / CAMPAIGN
MANIFEST_DIR = REPORT_DIR / "manifests"
LOG_DIR = REPORT_DIR / "logs"


@dataclass(frozen=True)
class Dataset:
    case_id: str
    name: str
    dct_keep: int
    event_top_k: int
    event_threshold: float
    norm_threshold: float
    peak_threshold: float
    dense4_gate: float
    stratified: bool = False


DATASETS = [
    Dataset("J1", "J1_isotropic1024coarse_pressure_256x256x256_128f", 6, 256, 0.005, 0.0002, 0.001, 0.30),
    Dataset("J2", "J2_isotropic1024coarse_ux_256x256x256_128f", 12, 768, 0.0015, 0.00015, 0.001, 0.25, True),
    Dataset("J3", "J3_mixing_density_256x256x256_128f", 15, 1024, 0.001, 0.0001, 0.001, 0.30),
    Dataset("J4", "J4_mixing_pressure_256x256x256_128f", 6, 256, 0.005, 0.0002, 0.001, 0.30),
    Dataset("J5", "J5_mhd1024_bx_256x256x256_128f", 14, 1024, 0.001, 0.0001, 0.001, 0.40, True),
    Dataset("J6", "J6_channel_pressure_256x256x256_128f", 10, 768, 0.0015, 0.00015, 0.001, 0.30),
    Dataset("J7", "J7_channel_ux_256x256x256_128f", 10, 768, 0.0015, 0.00015, 0.001, 0.30, True),
    Dataset("J8", "J8_transition_bl_pressure_256x224x256_128f", 16, 1024, 0.001, 0.0001, 0.001, 0.30),
]


REPORT_FIELDS = {
    "saved_bytes": "saved probe file bytes",
    "evaluated_samples": "evaluated samples",
    "rmse": "voxel RMSE",
    "psnr_db": "voxel PSNR",
    "leaf_count": "leaf count",
    "mode0": "mode0 blocks",
    "mode1": "mode1 blocks",
    "mode2": "mode2 blocks",
    "mode3": "mode3 blocks",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_report(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    metrics: dict[str, int | float] = {}
    for output_key, label in REPORT_FIELDS.items():
        match = re.search(rf"- {re.escape(label)}: `([^`]+)`", text)
        if not match:
            raise RuntimeError(f"Report is missing '{label}': {path}")
        raw = match.group(1).replace(" dB", "").strip()
        metrics[output_key] = float(raw) if any(c in raw for c in ".eE") else int(raw)
    return metrics


def write_json_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(temporary, path)


def build_command(dataset: Dataset, raw: Path, metadata: Path, output: Path, report: Path, threads: int) -> list[str]:
    command = [
        str(VBT_EXE),
        "--input-raw", str(raw),
        "--metadata", str(metadata),
        "--profile", "generic",
        "--dct-keep", str(dataset.dct_keep),
        "--event-top-k", str(dataset.event_top_k),
        "--event-threshold", str(dataset.event_threshold),
        "--event-min-count", "3",
        "--split-scientific-render-modes",
        "--generic-adaptive-coarse-keep",
        "--generic-dense-crossover",
        "--generic-dense-grid-res", "3",
        "--generic-densegrid4-candidate",
        "--generic-densegrid4-dist-threshold", str(dataset.dense4_gate),
        "--generic-dense-bits", "4",
        "--generic-rdo-lambda", "1e-05",
        "--generic-rdo-spatial-stride", "2",
        "--generic-rdo-time-stride", "4",
        "--generic-rdo-p99-weight", "2",
        "--generic-rdo-peak-weight", "4",
        "--generic-rdo-max-envelope",
        "--generic-norm-thr", str(dataset.norm_threshold),
        "--generic-peak-thr", str(dataset.peak_threshold),
        "--full-eval",
        "--omp-threads", str(threads),
        "--save-vbt", str(output),
        "--report", str(report),
    ]
    if dataset.stratified:
        command.append("--generic-stratified-events")
    return command


def validate_output(path: Path) -> None:
    result = subprocess.run(
        [str(VALIDATE_EXE), "--input-vbt", str(path), "--batch-size", "1", "--pattern", "same-t"],
        cwd=VBT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"VBTPACK4 validation failed for {path}:\n{result.stdout}\n{result.stderr}")


def run_dataset(dataset: Dataset, threads: int, force: bool, hash_raw: bool) -> dict:
    raw = DATA_ROOT / f"{dataset.name}.raw"
    metadata_path = DATA_ROOT / f"{dataset.name}.metadata.json"
    output = OUTPUT_DIR / f"{dataset.name}.vbtp"
    report = REPORT_DIR / f"{dataset.name}.md"
    manifest_path = MANIFEST_DIR / f"{dataset.case_id}.json"
    log_path = LOG_DIR / f"{dataset.case_id}.log"

    if not raw.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing input for {dataset.case_id}: {raw} / {metadata_path}")
    metadata = load_metadata(metadata_path)
    w, h, d, frames = dimensions(metadata)
    expected_bytes = w * h * d * frames * 4
    if raw.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"{dataset.case_id} raw size mismatch: expected {expected_bytes}, got {raw.stat().st_size}"
        )
    if raw_layout_name(metadata) != "time-fastest":
        raise RuntimeError(f"{dataset.case_id} must declare time-fastest raw storage")

    if not force and output.exists() and report.exists() and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            validate_output(output)
            print(f"[skip] {dataset.case_id}: complete manifest already exists", flush=True)
            return existing

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    partial_output = output.with_suffix(".partial.vbtp")
    partial_report = report.with_suffix(".partial.md")
    for partial in (partial_output, partial_report):
        if partial.exists():
            partial.unlink()

    command = build_command(dataset, raw, metadata_path, partial_output, partial_report, threads)
    manifest = {
        "campaign": CAMPAIGN,
        "case_id": dataset.case_id,
        "dataset": dataset.name,
        "status": "running",
        "started_utc": utc_now(),
        "raw_path": str(raw),
        "raw_bytes": raw.stat().st_size,
        "raw_sha256": sha256(raw) if hash_raw else None,
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "raw_layout": raw_layout_name(metadata),
        "dimensions_xyzt": [w, h, d, frames],
        "encoder_path": str(VBT_EXE),
        "encoder_sha256": sha256(VBT_EXE),
        "command": command,
        "output_path": str(output),
        "report_path": str(report),
        "log_path": str(log_path),
    }
    write_json_atomic(manifest_path, manifest)

    print(f"[run] {dataset.case_id}: {dataset.name}", flush=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(subprocess.list2cmdline(command) + "\n\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=VBT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        manifest.update(
            status="failed",
            finished_utc=utc_now(),
            elapsed_seconds=elapsed,
            return_code=result.returncode,
        )
        write_json_atomic(manifest_path, manifest)
        raise RuntimeError(f"{dataset.case_id} encoder failed; see {log_path}")

    validate_output(partial_output)
    metrics = parse_report(partial_report)
    os.replace(partial_output, output)
    os.replace(partial_report, report)
    manifest.update(
        status="complete",
        finished_utc=utc_now(),
        elapsed_seconds=elapsed,
        return_code=0,
        output_bytes=output.stat().st_size,
        output_sha256=sha256(output),
        metrics=metrics,
    )
    write_json_atomic(manifest_path, manifest)
    print(
        f"[done] {dataset.case_id}: {elapsed / 60.0:.2f} min, "
        f"{metrics['saved_bytes'] / 1.0e6:.3f} MB, {metrics['psnr_db']:.4f} dB",
        flush=True,
    )
    return manifest


def write_summary(manifests: list[dict]) -> None:
    complete = [item for item in manifests if item.get("status") == "complete"]
    summary = {
        "campaign": CAMPAIGN,
        "generated_utc": utc_now(),
        "complete_cases": [item["case_id"] for item in complete],
        "cases": complete,
    }
    write_json_atomic(REPORT_DIR / "campaign_manifest.json", summary)

    lines = [
        f"# {CAMPAIGN}",
        "",
        "| Case | Output MB | PSNR dB | RMSE | Elapsed min |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in complete:
        metrics = item["metrics"]
        lines.append(
            f"| {item['case_id']} | {item['output_bytes'] / 1.0e6:.3f} | "
            f"{metrics['psnr_db']:.4f} | {metrics['rmse']:.8g} | "
            f"{item['elapsed_seconds'] / 60.0:.2f} |"
        )
    lines.extend(
        [
            "",
            "- Raw contract: axis_order=[X,Y,Z,T], time_is_fastest_dimension=true.",
            "- Encoder internal layout: canonical frame-major [T][Z][Y][X].",
            "- Every output passed the strict VBTPACK4 loader before finalization.",
        ]
    )
    (REPORT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", default=[], help="Case id, e.g. --only J1")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-raw-hash", action="store_true")
    args = parser.parse_args()

    if not VBT_EXE.exists() or not VALIDATE_EXE.exists():
        raise FileNotFoundError("Build the encoder and strict validator before running the campaign")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    selected = {value.upper() for value in args.only}
    datasets = [dataset for dataset in DATASETS if not selected or dataset.case_id in selected]
    unknown = selected - {dataset.case_id for dataset in DATASETS}
    if unknown:
        raise ValueError(f"Unknown cases: {sorted(unknown)}")

    free_bytes = shutil.disk_usage(VBT_ROOT).free
    if free_bytes < 10 * 1024**3:
        raise RuntimeError(f"Insufficient free disk space: {free_bytes / 1024**3:.2f} GiB")

    manifests = []
    for dataset in datasets:
        manifests.append(
            run_dataset(
                dataset,
                threads=args.threads,
                force=args.force,
                hash_raw=not args.skip_raw_hash,
            )
        )
        write_summary(manifests)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        raise
