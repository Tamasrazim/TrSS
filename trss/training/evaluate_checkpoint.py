"""Evaluate a trained TrSS checkpoint on a held-out paired-render manifest.

Writes full-size model predictions, bicubic-baseline comparisons, per-frame/per-scene
RGB metrics, and a local HTML gallery. These metrics describe the supplied dataset;
synthetic-scene results are not evidence of game capture or Arc GPU performance.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from .data import TrSSFrameDataset
from .infer import _load_checkpoint, _resolve_device, _synchronize
from .manifest import load_manifest
from .model import MODEL_FORMAT, compose_prediction

# Accumulate metrics in display-encoded RGB space, matching the supplied PNG targets.
_METRIC_KEYS = ("model_abs", "model_sq", "bicubic_abs", "bicubic_sq", "elements")


def _sha256(path: Path) -> str:
    """Hash the exact checkpoint bytes so an evaluation can be reproduced."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_metadata(manifest_path: Path) -> dict[int, dict[str, Any]]:
    """Keep scene/frame annotations that the tensor loader intentionally ignores."""
    metadata: dict[int, dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                metadata[line_number] = row
    return metadata


def _select_indices(total: int, max_frames: int | None) -> list[int]:
    """Choose evenly spaced frames when a quick, explicitly sampled run is requested."""
    if total < 1:
        raise ValueError("The validation manifest contains no frames.")
    if max_frames is None or max_frames >= total:
        return list(range(total))
    if max_frames < 1:
        raise ValueError("max_frames must be at least one.")
    if max_frames == 1:
        return [(total - 1) // 2]
    return sorted({
        int(round(index * (total - 1) / (max_frames - 1)))
        for index in range(max_frames)
    })


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Cover an axis from end to end, including its far edge."""
    if length <= tile_size:
        return [0]
    if overlap == 0:
        return list(range(0, length, tile_size))
    stride = tile_size - 2 * overlap
    if stride < 1:
        raise ValueError("tile overlap must be less than half the tile size.")
    starts = list(range(0, length - tile_size + 1, stride))
    far_edge = length - tile_size
    if starts[-1] != far_edge:
        starts.append(far_edge)
    return starts


def _axis_weights(
    length: int,
    overlap: int,
    *,
    has_previous_tile: bool,
    has_next_tile: bool,
) -> torch.Tensor:
    """Down-weight artificial tile borders while leaving real image borders intact."""
    values = np.ones(length, dtype=np.float32)
    ramp = min(2 * overlap, length)
    if ramp:
        if has_previous_tile:
            values[:ramp] = np.linspace(1.0 / (ramp + 1), 1.0, ramp, dtype=np.float32)
        if has_next_tile:
            values[-ramp:] = np.minimum(
                values[-ramp:],
                np.linspace(1.0, 1.0 / (ramp + 1), ramp, dtype=np.float32),
            )
    return torch.from_numpy(values)


def _predict_tiled(
    model: torch.nn.Module,
    sample: dict[str, torch.Tensor],
    device: torch.device,
    *,
    tile_size: int,
    overlap: int,
) -> torch.Tensor:
    """Reconstruct a complete frame in overlapping tiles to bound accelerator memory."""
    features = sample["features"]
    _, height, width = features.shape
    if tile_size < 2:
        raise ValueError("tile_size must be at least 2 low-resolution pixels.")
    if overlap < 0 or overlap * 2 >= tile_size:
        raise ValueError("tile_overlap must be non-negative and less than half tile_size.")

    rows = _tile_starts(height, tile_size, overlap)
    columns = _tile_starts(width, tile_size, overlap)
    full_height, full_width = height * 2, width * 2
    accumulated = torch.zeros((3, full_height, full_width), dtype=torch.float32)
    total_weight = torch.zeros((1, full_height, full_width), dtype=torch.float32)

    with torch.inference_mode():
        for row_index, top in enumerate(rows):
            tile_height = min(tile_size, height - top)
            for column_index, left in enumerate(columns):
                tile_width = min(tile_size, width - left)
                bottom, right = top + tile_height, left + tile_width
                high_top, high_left = top * 2, left * 2
                high_bottom, high_right = bottom * 2, right * 2

                tile_features = features[:, top:bottom, left:right].unsqueeze(0).to(device)
                tile_current = sample["low_res"][
                    :, top:bottom, left:right
                ].unsqueeze(0).to(device)
                tile_history = sample["history"][
                    :, high_top:high_bottom, high_left:high_right
                ].unsqueeze(0).to(device)
                tile_valid = sample["history_valid"][
                    :, high_top:high_bottom, high_left:high_right
                ].unsqueeze(0).to(device)

                raw_head = model(tile_features)
                tile_prediction = compose_prediction(
                    raw_head.float(),
                    tile_current.float(),
                    tile_history.float(),
                    tile_valid.float(),
                )[0].detach().float().cpu()

                tile_overlap_y = min(overlap * 2, tile_height * 2)
                tile_overlap_x = min(overlap * 2, tile_width * 2)
                weights_y = _axis_weights(
                    tile_height * 2,
                    tile_overlap_y // 2,
                    has_previous_tile=row_index > 0,
                    has_next_tile=row_index < len(rows) - 1,
                )
                weights_x = _axis_weights(
                    tile_width * 2,
                    tile_overlap_x // 2,
                    has_previous_tile=column_index > 0,
                    has_next_tile=column_index < len(columns) - 1,
                )
                weights = torch.outer(weights_y, weights_x).unsqueeze(0)

                accumulated[:, high_top:high_bottom, high_left:high_right] += (
                    tile_prediction * weights
                )
                total_weight[:, high_top:high_bottom, high_left:high_right] += weights

    if not bool((total_weight > 0).all()):
        raise RuntimeError("Internal tiled evaluation error: some output pixels were not covered.")
    return (accumulated / total_weight).clamp(0.0, 1.0)


def _empty_accumulator() -> dict[str, float]:
    return {key: 0.0 for key in _METRIC_KEYS}


def _add_errors(
    accumulator: dict[str, float],
    prediction: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Accumulate pixel-weighted error sums instead of averaging frame averages."""
    model_error = prediction.float() - target.float()
    baseline_error = baseline.float() - target.float()
    accumulator["model_abs"] += float(model_error.abs().sum())
    accumulator["model_sq"] += float(model_error.square().sum())
    accumulator["bicubic_abs"] += float(baseline_error.abs().sum())
    accumulator["bicubic_sq"] += float(baseline_error.square().sum())
    accumulator["elements"] += float(target.numel())


def _finish_metrics(
    accumulator: dict[str, float],
    *,
    frames: int | None = None,
) -> dict[str, Any]:
    """Derive MAE, RMSE, and PSNR from accumulated display-RGB errors."""
    count = accumulator["elements"]
    if count <= 0:
        raise ValueError("Cannot report metrics for an empty evaluation.")
    model_mae = accumulator["model_abs"] / count
    model_mse = accumulator["model_sq"] / count
    base_mae = accumulator["bicubic_abs"] / count
    base_mse = accumulator["bicubic_sq"] / count
    values: dict[str, Any] = {
        "mae": model_mae,
        "rmse": math.sqrt(model_mse),
        "psnr_db": -10.0 * math.log10(max(model_mse, 1e-12)),
        "bicubic_mae": base_mae,
        "bicubic_rmse": math.sqrt(base_mse),
        "bicubic_psnr_db": -10.0 * math.log10(max(base_mse, 1e-12)),
        "mae_improvement_percent": (
            (base_mae - model_mae) / base_mae * 100.0 if base_mae > 0 else 0.0
        ),
        "psnr_delta_db": (
            -10.0 * math.log10(max(model_mse, 1e-12))
            + 10.0 * math.log10(max(base_mse, 1e-12))
        ),
    }
    if frames is not None:
        values["frames"] = frames
    return values


def _uint8_chw(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized CPU CHW RGB tensor to a contiguous uint8 image."""
    return np.rint(
        tensor.detach().float().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
    ).astype(np.uint8)


def _comparison_image(
    destination: Path,
    panels: list[tuple[str, np.ndarray]],
    *,
    panel_width: int = 480,
    panel_height: int = 270,
) -> None:
    """Save a compact, labelled four-way preview without resizing full-res outputs."""
    label_height = 32
    canvas = Image.new(
        "RGB",
        (panel_width * len(panels), panel_height + label_height),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, pixels) in enumerate(panels):
        x0 = index * panel_width
        draw.text((x0 + 10, 9), label, fill=(245, 245, 245))
        with Image.fromarray(pixels, mode="RGB") as source:
            available = (panel_width - 16, panel_height - 16)
            scale = min(available[0] / source.width, available[1] / source.height)
            size = (
                max(1, int(round(source.width * scale))),
                max(1, int(round(source.height * scale))),
            )
            preview = source.resize(size, Image.Resampling.LANCZOS)
            canvas.paste(
                preview,
                (
                    x0 + (panel_width - size[0]) // 2,
                    label_height + (panel_height - size[1]) // 2,
                ),
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=90, optimize=True)


def _write_gallery(
    root: Path,
    frame_rows: list[dict[str, Any]],
    global_metrics: dict[str, Any],
) -> None:
    """Build an offline page for visual inspection of model and baseline outputs."""
    cards: list[str] = []
    for frame in frame_rows:
        metrics = frame["metrics"]
        cards.append(
            "<article class=\"card\">"
            f"<a href=\"{html.escape(frame['comparison'], quote=True)}\">"
            f"<img loading=\"lazy\" src=\"{html.escape(frame['comparison'], quote=True)}\" "
            f"alt=\"Comparison for {html.escape(frame['scene_id'])} frame "
            f"{frame['frame_index']}\"></a>"
            f"<h2>{html.escape(frame['scene_id'])} · frame {frame['frame_index']}</h2>"
            f"<p>TrSS MAE {metrics['mae']:.5f} · PSNR {metrics['psnr_db']:.2f} dB"
            f"<br>Bicubic MAE {metrics['bicubic_mae']:.5f} · "
            f"PSNR {metrics['bicubic_psnr_db']:.2f} dB</p>"
            f"<p><a href=\"{html.escape(frame['prediction'], quote=True)}\">Full-size TrSS prediction</a></p>"
            "</article>"
        )
    direction = (
        "TrSS beats bicubic on aggregate MAE."
        if global_metrics["mae"] < global_metrics["bicubic_mae"]
        else "TrSS does not beat bicubic on aggregate MAE."
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TrSS checkpoint evaluation</title>
<style>
:root{{color-scheme:dark;font-family:system-ui,sans-serif;background:#101010;color:#f5f5f5}}
body{{margin:0;padding:24px;max-width:1800px;margin-inline:auto}}
h1{{margin-bottom:6px}}.notice{{color:#c8c8c8;max-width:1000px;line-height:1.5}}
.summary{{display:flex;flex-wrap:wrap;gap:12px;margin:22px 0}}
.metric{{border:1px solid #555;padding:12px 16px;min-width:160px}}
.metric strong{{display:block;font-size:1.2rem;margin-top:5px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:18px}}
.card{{border:1px solid #444;padding:12px;min-width:0;overflow:hidden}}
.card img{{display:block;width:100%;height:auto;background:#1b1b1b}}
.card h2{{font-size:1rem;overflow-wrap:anywhere}}a{{color:#ddd}}
</style>
</head>
<body>
<h1>TrSS checkpoint evaluation</h1>
<p class="notice">Metrics compare normalized display-encoded RGB pixels against the supplied target images.
They measure this manifest only, not live-game rendering, photorealism, or GPU frame rate.
{html.escape(direction)}</p>
<section class="summary">
<div class="metric">Frames<strong>{global_metrics.get('frames', len(frame_rows))}</strong></div>
<div class="metric">TrSS MAE<strong>{global_metrics['mae']:.6f}</strong></div>
<div class="metric">Bicubic MAE<strong>{global_metrics['bicubic_mae']:.6f}</strong></div>
<div class="metric">TrSS PSNR<strong>{global_metrics['psnr_db']:.3f} dB</strong></div>
<div class="metric">Bicubic PSNR<strong>{global_metrics['bicubic_psnr_db']:.3f} dB</strong></div>
</section>
<div class="grid">{''.join(cards)}</div>
</body></html>
"""
    (root / "index.html").write_text(document, encoding="utf-8", newline="\n")


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    motion_scale_px: float = 64.0,
    tile_size: int = 128,
    tile_overlap: int = 16,
    max_frames: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate a checkpoint, saving predictions, comparison previews, and metrics."""
    started = perf_counter()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()

    if output == output.parent or output.name in ("", ".", ".."):
        raise ValueError("output_dir must name a dedicated output folder.")
    if output.exists() and not force:
        raise FileExistsError(f"Output path already exists: {output}. Pass --force to replace it.")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path must be a directory: {output}")
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_file}")
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_file}")
    if not math.isfinite(motion_scale_px) or motion_scale_px <= 0:
        raise ValueError("motion_scale_px must be finite and greater than zero.")
    if tile_size < 2 or tile_overlap < 0 or tile_overlap * 2 >= tile_size:
        raise ValueError("tile_size must be >= 2 and tile_overlap less than half tile_size.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be at least one.")
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved_manifest = manifest_file
    rows = load_manifest(resolved_manifest)
    metadata = _read_source_metadata(resolved_manifest)
    selected_indices = _select_indices(len(rows), max_frames)
    device = _resolve_device(device_name)
    model, checkpoint = _load_checkpoint(checkpoint_file)
    model = model.to(device).eval()
    dataset = TrSSFrameDataset(resolved_manifest, motion_scale_px=motion_scale_px)

    transaction_root = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.evaluate-", dir=output.parent
    ))
    staging = transaction_root / "payload"
    staging.mkdir()
    (staging / "renders").mkdir()
    (staging / "comparisons").mkdir()
    backup: Path | None = None

    global_acc = _empty_accumulator()
    scene_accumulators: dict[str, dict[str, float]] = {}
    frame_results: list[dict[str, Any]] = []
    try:
        for sample_index in selected_indices:
            manifest_row = rows[sample_index]
            source = metadata.get(int(manifest_row.get("line", 0)), {})
            scene_id = str(source.get("scene_id", "scene-unknown"))
            frame_index = int(source.get("frame_index", sample_index))
            sample = dataset[sample_index]
            height, width = sample["low_res"].shape[-2:]
            target = sample["target"].float()
            prediction = _predict_tiled(
                model,
                sample,
                device,
                tile_size=tile_size,
                overlap=tile_overlap,
            )

            # Baseline interpolation stays on CPU so a full-frame baseline does not
            # compete with tiled activations for scarce accelerator memory.
            bicubic = F.interpolate(
                sample["low_res"].unsqueeze(0).float(),
                scale_factor=2,
                mode="bicubic",
                align_corners=False,
            ).clamp(0.0, 1.0)[0]

            _synchronize(device)
            _add_errors(global_acc, prediction, bicubic, target)
            scene_acc = scene_accumulators.setdefault(scene_id, _empty_accumulator())
            _add_errors(scene_acc, prediction, bicubic, target)
            frame_acc = _empty_accumulator()
            _add_errors(frame_acc, prediction, bicubic, target)
            frame_metrics = _finish_metrics(frame_acc)

            line_number = int(manifest_row.get("line", sample_index + 1))
            filename = f"frame-line-{line_number:06d}"
            prediction_name = f"renders/{filename}.png"
            comparison_name = f"comparisons/{filename}.jpg"
            Image.fromarray(_uint8_chw(prediction), mode="RGB").save(
                staging / prediction_name, format="PNG", optimize=True
            )

            low_rgb = _uint8_chw(sample["low_res"])
            # Nearest-neighbour expansion intentionally exposes the original input pixels.
            low_preview = Image.fromarray(low_rgb, mode="RGB").resize(
                (width * 2, height * 2), Image.Resampling.NEAREST
            )
            target_rgb = _uint8_chw(target)
            _comparison_image(
                staging / comparison_name,
                [
                    ("Input ×2 nearest", np.asarray(low_preview, dtype=np.uint8)),
                    ("Bicubic baseline", _uint8_chw(bicubic)),
                    ("TrSS prediction", _uint8_chw(prediction)),
                    ("Target reference", target_rgb),
                ],
            )

            frame_results.append({
                "scene_id": scene_id,
                "frame_index": frame_index,
                "manifest_line": line_number,
                "input_size": [width, height],
                "target_size": [width * 2, height * 2],
                "prediction": prediction_name,
                "comparison": comparison_name,
                "metrics": frame_metrics,
            })

        global_metrics = _finish_metrics(global_acc, frames=len(frame_results))
        scene_metrics = {
            scene_id: _finish_metrics(accumulator, frames=sum(
                1 for frame in frame_results if frame["scene_id"] == scene_id
            ))
            for scene_id, accumulator in sorted(scene_accumulators.items())
        }
        selected_source_types = [
            metadata.get(int(rows[index].get("line", 0)), {}).get("source_type")
            for index in selected_indices
        ]
        synthetic = bool(selected_source_types) and all(
            source_type == "synthetic_controlled_3d_scene"
            for source_type in selected_source_types
        )
        report: dict[str, Any] = {
            "format": "trss-checkpoint-evaluation-v1",
            "evaluation_completed": True,
            "output_dir": str(output),
            "checkpoint": str(checkpoint_file),
            "checkpoint_sha256": _sha256(checkpoint_file),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "model_format": MODEL_FORMAT,
            "model_config": model.config,
            "manifest": str(resolved_manifest),
            "manifest_frame_count": len(rows),
            "evaluated_frame_count": len(frame_results),
            "sampling": (
                "all manifest frames"
                if len(frame_results) == len(rows)
                else "deterministic evenly-spaced subset; not a full validation score"
            ),
            "device": str(device),
            "tile_size_low_resolution": tile_size,
            "tile_overlap_low_resolution": tile_overlap,
            "motion_scale_px": motion_scale_px,
            "source_type": (
                "synthetic_controlled_3d_scene" if synthetic else "provided_manifest"
            ),
            "actual_game_capture": False if synthetic else None,
            "metrics_domain": "normalized display-encoded RGB target pixels",
            "metrics": global_metrics,
            "beats_bicubic_on_mae": global_metrics["mae"] < global_metrics["bicubic_mae"],
            "beats_bicubic_on_psnr": global_metrics["psnr_db"] > global_metrics["bicubic_psnr_db"],
            "per_scene_metrics": scene_metrics,
            "frames": frame_results,
            "elapsed_seconds": perf_counter() - started,
            "limits": [
                "This report measures image errors against the supplied target frames only.",
                "A synthetic-scene result is not evidence of generalization to a real game.",
                "This tool does not capture a live game or measure game frame rate.",
                "Tiled inference blends overlaps; record tile settings when comparing runs.",
            ],
        }
        (staging / "evaluation-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _write_gallery(staging, frame_results, global_metrics)

        if output.exists():
            backup = transaction_root / "previous-output"
            output.rename(backup)
        try:
            staging.rename(output)
        except Exception:
            if backup is not None and backup.exists() and not output.exists():
                backup.rename(output)
                backup = None
            raise

        # The generated report keeps relative render/gallery links, and its absolute
        # output_dir is already the final destination rather than the staging path.
        return report
    finally:
        shutil.rmtree(transaction_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a TrSS checkpoint against bicubic interpolation on paired "
            "validation images and build an offline visual review gallery."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path,
                        help="held-out validation JSONL manifest")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", help="auto, xpu, cuda, or cpu")
    parser.add_argument("--motion-scale-px", type=float, default=64.0)
    parser.add_argument("--tile-size", type=int, default=128,
                        help="input-resolution tile edge in pixels")
    parser.add_argument("--tile-overlap", type=int, default=16,
                        help="input-resolution overlap blended between tiles")
    parser.add_argument("--max-frames", type=int,
                        help="evaluate an evenly spaced subset instead of the full manifest")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing output directory after evaluation succeeds")
    args = parser.parse_args()
    try:
        report = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            device_name=args.device,
            motion_scale_px=args.motion_scale_px,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            max_frames=args.max_frames,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, NotADirectoryError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "evaluation_report": str(args.output_dir.expanduser().resolve() / "evaluation-report.json"),
        "gallery": str(args.output_dir.expanduser().resolve() / "index.html"),
        "evaluated_frames": report["evaluated_frame_count"],
        "metrics": report["metrics"],
        "beats_bicubic_on_mae": report["beats_bicubic_on_mae"],
        "beats_bicubic_on_psnr": report["beats_bicubic_on_psnr"],
        "source_type": report["source_type"],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
