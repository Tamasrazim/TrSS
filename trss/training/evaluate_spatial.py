"""Evaluate an RGB-only spatial checkpoint and write a local visual comparison gallery."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .spatial_data import TrSSSpatialFrameDataset, load_spatial_manifest
from .spatial_model import SPATIAL_MODEL_FORMAT, TrSSSpatialSR
from .train_spatial import _frame_metrics, _predict_tiled, _resolve_device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evenly_spaced(length: int, maximum: int | None) -> list[int]:
    if maximum is None or maximum >= length:
        return list(range(length))
    if maximum < 1:
        raise ValueError("max_frames must be a positive integer when supplied.")
    if maximum == 1:
        return [length // 2]
    return sorted({round(i * (length - 1) / (maximum - 1)) for i in range(maximum)})


def _tensor_image(tensor: torch.Tensor) -> Image.Image:
    """Convert one RGB CHW tensor in [0,1] to a displayable RGB image."""
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Preview conversion expects one image per tensor.")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Preview conversion expects a CHW RGB tensor.")
    pixels = (
        tensor.detach().to("cpu").float().clamp(0.0, 1.0)
        .permute(1, 2, 0).numpy() * 255.0 + 0.5
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _write_gallery(path: Path, report: dict[str, Any]) -> None:
    cards: list[str] = []
    for frame in report["previews"]:
        title = html.escape(frame["title"])
        sections = []
        for key, label in (
            ("input", "Input (nearest enlarged)"),
            ("bicubic", "Bicubic baseline"),
            ("prediction", "TrSS spatial prediction"),
            ("target", "Captured target"),
        ):
            image_path = html.escape(frame[key], quote=True)
            sections.append(
                f'<figure><figcaption>{label}</figcaption>'
                f'<a href="{image_path}"><img loading="lazy" src="{image_path}" alt="{label}"></a>'
                f'</figure>'
            )
        metrics = frame["metrics"]
        cards.append(
            '<section class="sample"><h2>' + title + '</h2>'
            + '<p class="metrics">TrSS MAE ' + f'{metrics["mae"]:.6f}'
            + ' · PSNR ' + f'{metrics["psnr_db"]:.3f} dB</p>'
            + '<div class="images">' + "".join(sections) + '</div></section>'
        )
    text = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TrSS spatial checkpoint evaluation</title>
<style>
:root{color-scheme:dark}body{margin:0;background:#101216;color:#eef1f5;font:15px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:1500px;margin:auto;padding:28px}h1{font-size:clamp(24px,4vw,42px);margin:0}
.summary{color:#aab3bf;margin:8px 0 28px}.sample{padding:20px 0;border-top:1px solid #39414e}
.images{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
figure{margin:0;min-width:0}figcaption{color:#b8c1ce;font-size:13px;margin-bottom:6px}
img{display:block;width:100%;height:auto;background:#060708;border:1px solid #303744}
.metrics{font:13px ui-monospace,Consolas,monospace;color:#bbc4d0}
@media(max-width:900px){.images{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:500px){main{padding:16px}.images{grid-template-columns:1fr}}
</style></head><body><main>
<h1>TrSS spatial checkpoint evaluation</h1>
<p class="summary">Offline RGB-only 2x reconstruction. The target is the recorded frame; input is derived by downsampling that same frame. This gallery is not a live-game or temporal-rendering test.</p>
""" + "".join(cards) + "
</main></body></html>
"
    path.write_text(text, encoding="utf-8", newline="\n")


@torch.no_grad()
def evaluate_spatial_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    tile_size: int = 256,
    tile_overlap: int = 16,
    max_frames: int | None = None,
    max_previews: int = 12,
    force: bool = False,
) -> dict[str, Any]:
    """Compare a spatial checkpoint against bicubic interpolation and save examples."""
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    rows = load_spatial_manifest(manifest_file)
    if max_frames is not None and (
        not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 1
    ):
        raise ValueError("max_frames must be a positive integer when supplied.")
    if not isinstance(max_previews, int) or isinstance(max_previews, bool) or max_previews < 1:
        raise ValueError("max_previews must be a positive integer.")
    if tile_size < 2 or tile_overlap < 0 or tile_overlap * 2 >= tile_size:
        raise ValueError("tile_size must be >= 2 and tile_overlap less than half tile_size.")

    try:
        checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not load checkpoint '{checkpoint_file}': {exc}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != SPATIAL_MODEL_FORMAT:
        raise ValueError(
            f"Checkpoint format must be '{SPATIAL_MODEL_FORMAT}'; temporal checkpoints are incompatible."
        )
    config = checkpoint.get("model_config")
    state = checkpoint.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("Spatial checkpoint is missing model_config or model_state_dict.")

    device = _resolve_device(device_name)
    model = TrSSSpatialSR(
        channels=int(config.get("channels", 32)),
        blocks=int(config.get("blocks", 6)),
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Checkpoint weights do not match the recorded model config: {exc}") from exc
    model.to(device).eval()

    output = Path(output_dir).expanduser().resolve()
    if output == output.parent or not output.name:
        raise ValueError("output_dir must name a dedicated output folder.")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output}")
    if output.exists() and not force:
        raise FileExistsError(
            f"Output path already exists: {output}. Pass --force to replace it."
        )

    dataset = TrSSSpatialFrameDataset(manifest_file)
    selected = _evenly_spaced(len(dataset), max_frames)
    preview_positions = set(_evenly_spaced(len(selected), max_previews))
    output.parent.mkdir(parents=True, exist_ok=True)
    transaction_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.spatial-eval-", dir=output.parent)
    )
    staging = transaction_root / "payload"
    staging.mkdir()
    backup: Path | None = None
    try:
        preview_dir = staging / "previews"
        preview_dir.mkdir()
        aggregate_abs = aggregate_sq = bicubic_abs = bicubic_sq = 0.0
        component_count = 0
        per_frame: list[dict[str, Any]] = []
        previews: list[dict[str, Any]] = []

        for position, row_index in enumerate(selected):
            sample = dataset[row_index]
            low = sample["low_res"].unsqueeze(0)
            target = sample["target"].unsqueeze(0)
            prediction = _predict_tiled(
                model, low, device, tile_size=tile_size, tile_overlap=tile_overlap
            )
            baseline = F.interpolate(
                low, scale_factor=2, mode="bicubic", align_corners=False
            ).clamp(0.0, 1.0)
            metrics = _frame_metrics(prediction, target)
            baseline_metrics = _frame_metrics(baseline, target)
            difference = prediction - target
            baseline_difference = baseline - target
            aggregate_abs += float(difference.abs().sum().item())
            aggregate_sq += float(difference.square().sum().item())
            bicubic_abs += float(baseline_difference.abs().sum().item())
            bicubic_sq += float(baseline_difference.square().sum().item())
            component_count += target.numel()

            per_frame.append({
                "manifest_index": row_index,
                "scene_id": row["scene_id"],
                "source_name": row.get("source_name"),
                "frame_index": row.get("frame_index"),
                "width": int(target.shape[-1]),
                "height": int(target.shape[-2]),
                "metrics": metrics,
                "bicubic_metrics": baseline_metrics,
            })
            if position in preview_positions:
                prefix = f"frame-{row_index:06d}"
                nearest = F.interpolate(low, scale_factor=2, mode="nearest")
                assets = {
                    "input": nearest,
                    "bicubic": baseline,
                    "prediction": prediction,
                    "target": target,
                }
                paths: dict[str, str] = {}
                for key, tensor in assets.items():
                    relative = Path("previews") / f"{prefix}-{key}.png"
                    _tensor_image(tensor).save(staging / relative, format="PNG")
                    paths[key] = relative.as_posix()
                title = row.get("source_name") or f"Frame {row_index}"
                if row.get("frame_index") is not None:
                    title += f" · source frame {row['frame_index']}"
                previews.append({
                    "title": title,
                    **paths,
                    "metrics": metrics,
                    "bicubic_metrics": baseline_metrics,
                })

        if component_count == 0:
            raise ValueError("Manifest selection produced no validation pixels.")
        mae = aggregate_abs / component_count
        mse = aggregate_sq / component_count
        baseline_mae = bicubic_abs / component_count
        baseline_mse = bicubic_sq / component_count
        report = {
            "schema_version": 1,
            "completed": True,
            "checkpoint_format": SPATIAL_MODEL_FORMAT,
            "checkpoint": str(checkpoint_file),
            "checkpoint_sha256": _sha256(checkpoint_file),
            "manifest": str(manifest_file),
            "manifest_sha256": _sha256(manifest_file),
            "device": str(device),
            "model_config": dict(model.config),
            "manifest_frames": len(dataset),
            "evaluated_frames": len(selected),
            "subset_evaluation": len(selected) != len(dataset),
            "tile_size_low_resolution_pixels": tile_size,
            "tile_overlap_low_resolution_pixels": tile_overlap,
            "metrics": {
                "mae": mae,
                "rmse": math.sqrt(mse),
                "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
            },
            "bicubic_metrics": {
                "mae": baseline_mae,
                "rmse": math.sqrt(baseline_mse),
                "psnr_db": -10.0 * math.log10(max(baseline_mse, 1e-12)),
            },
            "beats_bicubic_on_mae": mae < baseline_mae,
            "beats_bicubic_on_psnr": mse < baseline_mse,
            "per_frame": per_frame,
            "previews": previews,
            "limits": [
                "RGB-only spatial reconstruction; no real depth, motion, or temporal history.",
                "Input images are downsampled versions of recorded targets, not the game's original internal low-resolution render.",
                "Metrics only characterize the selected local clips and do not measure in-game FPS or live rendering latency.",
            ],
        }
        (staging / "evaluation-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _write_gallery(staging / "index.html", report)
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
        return report
    finally:
        shutil.rmtree(transaction_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a TrSS RGB-only checkpoint and generate an offline visual gallery."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, xpu, or cuda")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=16)
    parser.add_argument("--max-frames", type=int, help="Evenly sample this many frames for metrics.")
    parser.add_argument("--max-previews", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        report = evaluate_spatial_checkpoint(
            args.checkpoint,
            args.manifest,
            args.output_dir,
            device_name=args.device,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            max_frames=args.max_frames,
            max_previews=args.max_previews,
            force=args.force,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    root = args.output_dir.expanduser().resolve()
    print(json.dumps({
        "report": str(root / "evaluation-report.json"),
        "gallery": str(root / "index.html"),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "evaluated_frames": report["evaluated_frames"],
        "beats_bicubic_on_mae": report["beats_bicubic_on_mae"],
        "beats_bicubic_on_psnr": report["beats_bicubic_on_psnr"],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
