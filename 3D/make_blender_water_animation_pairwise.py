import argparse
import subprocess
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
BLENDER = Path(r"C:\Program Files\Blender Foundation\Blender 5.1\blender.exe")
RAW_TO_VDB = ROOT / "vdb_tools" / "build" / "Release" / "raw_to_vdb.exe"
VBT_TO_VDB = ROOT / "vdb_tools" / "build" / "Release" / "render_temporal_vbt_to_vdb.exe"
VDB_TO_OBJ = ROOT / "vdb_tools" / "build" / "Release" / "vdb_to_obj.exe"
PROBE_EXE = ROOT.parent / "build" / "Release" / "vbt_render_temporal_kf_probe.exe"
BLENDER_PAIR = ROOT / "blender_render_water_pair.py"


def run(cmd):
    subprocess.run(cmd, check=True)


def make_gif_from_dir(input_dir: Path, output_gif: Path, duration_ms: int):
    frames = [imageio.imread(p) for p in sorted(input_dir.glob("frame_*.png"))]
    imageio.mimsave(output_gif, frames, duration=duration_ms / 1000.0)


def make_compare_gif(orig_dir: Path, recon_dir: Path, output_gif: Path, duration_ms: int):
    orig_paths = sorted(orig_dir.glob("frame_*.png"))
    recon_paths = sorted(recon_dir.glob("frame_*.png"))
    frames = []
    for op, rp in zip(orig_paths, recon_paths):
        orig = Image.open(op).convert("RGBA")
        recon = Image.open(rp).convert("RGBA")
        canvas = Image.new("RGBA", (orig.width * 2, orig.height), (0, 0, 0, 255))
        canvas.paste(orig, (0, 0))
        canvas.paste(recon, (orig.width, 0))
        frames.append(np.array(ImageOps.autocontrast(canvas.convert("RGB"))))
    imageio.mimsave(output_gif, frames, duration=duration_ms / 1000.0)


def main():
    parser = argparse.ArgumentParser(description="Render liquid animation by invoking the single-frame hero-shot pipeline for every frame.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--input-abc", required=True)
    parser.add_argument(
        "--abc-frame-offset",
        type=int,
        default=1,
        help="Alembic scene frame corresponding to VBT index 0 (water_flow.abc starts at frame 1).",
    )
    parser.add_argument("--input-vbt", default="")
    parser.add_argument("--input-raw", default="")
    parser.add_argument("--probe-exe", default=str(PROBE_EXE))
    parser.add_argument("--probe-variant", default="")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--frames", default="")
    parser.add_argument("--isovalue", type=float, required=True)
    parser.add_argument("--vdb-background", type=float)
    parser.add_argument("--smooth-gaussian-width", type=int, default=0)
    parser.add_argument("--smooth-gaussian-iterations", type=int, default=1)
    parser.add_argument("--smooth-mix", type=float, default=0.0)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--transmission-weight", type=float, default=1.0)
    parser.add_argument("--water-roughness", type=float, default=0.008)
    parser.add_argument("--absorption-density", type=float, default=0.0)
    parser.add_argument("--background-preset", choices=["studio", "grass"], default="studio")
    parser.add_argument("--camera-preset", choices=["hero", "side"], default="hero")
    parser.add_argument("--probe-sample-step", type=int, default=2)
    parser.add_argument("--probe-cutoff", default="0.0003")
    parser.add_argument("--probe-cutoff-band", default="0.0001")
    parser.add_argument("--probe-temporal-eps-abs", default="1e-5")
    parser.add_argument("--probe-temporal-eps-rel", default="0.02")
    parser.add_argument("--probe-control-eps-scale", default="8")
    parser.add_argument("--probe-temporal-gamma-delta", default="0.2")
    parser.add_argument("--probe-bg-zero-ratio", default="0.30")
    parser.add_argument("--probe-packed-keyframe-codec", default="fp16")
    parser.add_argument("--gif-duration-ms", type=int, default=60)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    use_probe = bool(args.probe_variant)
    if not use_probe and not args.input_vbt:
        parser.error("Either --input-vbt or --probe-variant is required")
    if use_probe and not args.input_raw:
        parser.error("--input-raw is required when using --probe-variant")

    if args.frames:
        frames = [int(text.strip()) for text in args.frames.split(",") if text.strip()]
        if not frames:
            parser.error("--frames must contain at least one frame index")
    else:
        frames = list(range(args.frame_count))
    if len(set(frames)) != len(frames):
        parser.error("--frames must not contain duplicate frame indices")
    if any(frame < 0 or frame >= args.frame_count for frame in frames):
        parser.error("--frames contains an index outside [0, frame-count)")

    input_abc = Path(args.input_abc).resolve()
    output_root = Path(args.output_root).resolve()
    orig_dir = output_root / "orig_png"
    recon_dir = output_root / "recon_png"
    tmp_dir = output_root / "_tmp"
    orig_dir.mkdir(parents=True, exist_ok=True)
    recon_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        source_frame = frame + args.abc_frame_offset
        tmp_vdb = tmp_dir / f"frame_{frame:04d}.vdb"
        tmp_obj = tmp_dir / f"frame_{frame:04d}.obj"
        orig_out = orig_dir / f"frame_{frame:04d}.png"
        recon_out = recon_dir / f"frame_{frame:04d}.png"

        if use_probe:
            run(
                [
                    str(args.probe_exe),
                    "--input-raw",
                    str(Path(args.input_raw)),
                    "--metadata",
                    str(Path(args.metadata)),
                    "--sample-step",
                    str(args.probe_sample_step),
                    "--cutoff",
                    str(args.probe_cutoff),
                    "--cutoff-band",
                    str(args.probe_cutoff_band),
                    "--temporal-eps-abs",
                    str(args.probe_temporal_eps_abs),
                    "--temporal-eps-rel",
                    str(args.probe_temporal_eps_rel),
                    "--control-eps-scale",
                    str(args.probe_control_eps_scale),
                    "--temporal-gamma-delta",
                    str(args.probe_temporal_gamma_delta),
                    "--bg-zero-ratio",
                    str(args.probe_bg_zero_ratio),
                    "--packed-keyframe-codec",
                    str(args.probe_packed_keyframe_codec),
                    "--route-empty-visible-thr",
                    "0",
                    "--route-fine-visible-thr",
                    "0",
                    "--route-fine-band-thr",
                    "0",
                    "--route-coarse-rmse-thr",
                    "0",
                    "--route-coarse-peak-thr",
                    "0",
                    "--route-fine-gain-thr",
                    "0",
                    "--cutoff-protect",
                    "--export-variant",
                    str(args.probe_variant),
                    "--export-frame",
                    str(frame),
                    "--export-frame-raw",
                    str(tmp_dir / f"frame_{frame:04d}.raw"),
                    "--export-frame-metadata",
                    str(tmp_dir / f"frame_{frame:04d}.metadata.json"),
                    "--report",
                    str(tmp_dir / f"frame_{frame:04d}_probe.md"),
                ]
            )
            run(
                [
                    str(RAW_TO_VDB),
                    "--input-raw",
                    str(tmp_dir / f"frame_{frame:04d}.raw"),
                    "--metadata",
                    str(tmp_dir / f"frame_{frame:04d}.metadata.json"),
                    "--output-vdb",
                    str(tmp_vdb),
                    "--frame",
                    "0",
                ]
            )
        else:
            vbt_to_vdb_cmd = [
                str(VBT_TO_VDB),
                "--input-vbt",
                str(Path(args.input_vbt)),
                "--metadata",
                str(Path(args.metadata)),
                "--output-vdb",
                str(tmp_vdb),
                "--frame",
                str(frame),
            ]
            if args.vdb_background is not None:
                vbt_to_vdb_cmd.extend(["--background", str(args.vdb_background)])
            run(vbt_to_vdb_cmd)
        run(
            [
                str(VDB_TO_OBJ),
                "--input-vdb",
                str(tmp_vdb),
                "--output-obj",
                str(tmp_obj),
                "--grid-name",
                "density",
                "--isovalue",
                str(args.isovalue),
                "--adaptivity",
                "0.0",
                "--smooth-gaussian-width",
                str(args.smooth_gaussian_width),
                "--smooth-gaussian-iterations",
                str(args.smooth_gaussian_iterations),
                "--smooth-mix",
                str(args.smooth_mix),
            ]
        )
        run(
            [
                str(BLENDER),
                "-b",
                "-P",
                str(BLENDER_PAIR),
                "--",
                "--input-abc",
                str(input_abc),
                "--input-recon",
                str(tmp_obj),
                "--frame",
                str(source_frame),
                "--orig-out",
                str(orig_out),
                "--recon-out",
                str(recon_out),
                "--resolution",
                str(args.resolution),
                "--samples",
                str(args.samples),
                "--transmission-weight",
                str(args.transmission_weight),
                "--water-roughness",
                str(args.water_roughness),
                "--absorption-density",
                str(args.absorption_density),
                "--background-preset",
                str(args.background_preset),
                "--camera-preset",
                str(args.camera_preset),
            ]
        )

        tmp_vdb.unlink(missing_ok=True)
        tmp_obj.unlink(missing_ok=True)
        if use_probe:
            (tmp_dir / f"frame_{frame:04d}.raw").unlink(missing_ok=True)
            (tmp_dir / f"frame_{frame:04d}.metadata.json").unlink(missing_ok=True)
            (tmp_dir / f"frame_{frame:04d}_probe.md").unlink(missing_ok=True)

    make_gif_from_dir(orig_dir, output_root / f"{args.name}_orig_blender.gif", args.gif_duration_ms)
    make_gif_from_dir(recon_dir, output_root / f"{args.name}_recon_blender.gif", args.gif_duration_ms)
    make_compare_gif(orig_dir, recon_dir, output_root / f"{args.name}_compare_blender.gif", args.gif_duration_ms)

    try:
        tmp_dir.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()
