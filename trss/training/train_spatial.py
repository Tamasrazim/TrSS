"""Train and validate the RGB-only TrSS spatial baseline on local image pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .spatial_data import TrSSSpatialFrameDataset, load_spatial_manifest
from .spatial_model import SPATIAL_MODEL_FORMAT, TrSSSpatialSR, compose_spatial_prediction


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "xpu" and (
        not hasattr(torch, "xpu") or not torch.xpu.is_available()
    ):
        raise RuntimeError(
            "PyTorch cannot see an XPU. Install Intel-GPU PyTorch wheels and check the driver."
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type not in ("cpu", "xpu", "cuda"):
        raise ValueError("device must be cpu, xpu, cuda, or auto.")
    return device


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_splits(
    training_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> None:
    overlapping_scenes = {
        row["scene_id"] for row in training_rows
    } & {row["scene_id"] for row in validation_rows}
    if overlapping_scenes:
        raise ValueError(
            "Train and validation manifests share scene_id values: "
            + ", ".join(sorted(overlapping_scenes))
        )
    train_paths = {
        row[name] for row in training_rows for name in ("low_res", "target")
    }
    validation_paths = {
        row[name] for row in validation_rows for name in ("low_res", "target")
    }
    if train_paths & validation_paths:
        raise ValueError("Train and validation manifests reuse at least one exact RGB frame file.")


def _random_patch(
    low: torch.Tensor, target: torch.Tensor, patch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop one aligned input/target patch while both tensors are still on CPU."""
    _, _, height, width = low.shape
    patch_h = min(height, patch_size)
    patch_w = min(width, patch_size)
    y = random.randint(0, height - patch_h)
    x = random.randint(0, width - patch_w)
    low = low[:, :, y : y + patch_h, x : x + patch_w]
    target = target[:, :, 2 * y : 2 * (y + patch_h), 2 * x : 2 * (x + patch_w)]
    if random.random() < 0.5:
        low = low.flip(-1)
        target = target.flip(-1)
    return low, target


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - 2 * overlap
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def _predict_tiled(
    model: TrSSSpatialSR,
    low_cpu: torch.Tensor,
    device: torch.device,
    *,
    tile_size: int,
    tile_overlap: int,
) -> torch.Tensor:
    """Run large frames in overlapping tiles and return the result on CPU."""
    if low_cpu.ndim != 4 or low_cpu.shape[0] != 1:
        raise ValueError("Tiled prediction expects one NCHW frame at a time.")
    _, _, height, width = low_cpu.shape
    accumulator = torch.zeros((1, 3, height * 2, width * 2), device=device)
    weights = torch.zeros((1, 1, height * 2, width * 2), device=device)

    for top in _tile_starts(height, tile_size, tile_overlap):
        bottom = min(top + tile_size, height)
        for left in _tile_starts(width, tile_size, tile_overlap):
            right = min(left + tile_size, width)
            tile = low_cpu[:, :, top:bottom, left:right].to(device)
            prediction = compose_spatial_prediction(model(tile), tile)
            tile_h, tile_w = bottom - top, right - left
            weight_y = torch.ones((1, 1, tile_h * 2, 1), device=device)
            weight_x = torch.ones((1, 1, 1, tile_w * 2), device=device)
            ramp_y = min(tile_overlap * 2, tile_h * 2)
            ramp_x = min(tile_overlap * 2, tile_w * 2)
            if top > 0 and ramp_y:
                weight_y[:, :, :ramp_y, :] = torch.linspace(
                    0.05, 1.0, ramp_y, device=device
                ).view(1, 1, ramp_y, 1)
            if bottom < height and ramp_y:
                weight_y[:, :, -ramp_y:, :] = torch.linspace(
                    1.0, 0.05, ramp_y, device=device
                ).view(1, 1, ramp_y, 1)
            if left > 0 and ramp_x:
                weight_x[:, :, :, :ramp_x] = torch.linspace(
                    0.05, 1.0, ramp_x, device=device
                ).view(1, 1, 1, ramp_x)
            if right < width and ramp_x:
                weight_x[:, :, :, -ramp_x:] = torch.linspace(
                    1.0, 0.05, ramp_x, device=device
                ).view(1, 1, 1, ramp_x)
            window = weight_y * weight_x
            y0, x0 = top * 2, left * 2
            y1, x1 = bottom * 2, right * 2
            accumulator[:, :, y0:y1, x0:x1] += prediction * window
            weights[:, :, y0:y1, x0:x1] += window

    if torch.any(weights <= 0):
        raise RuntimeError("Internal tiled predictor left uncovered output pixels.")
    return (accumulator / weights).clamp(0.0, 1.0).cpu()


def _frame_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    difference = prediction.float() - target.float()
    mse = float(difference.square().mean().item())
    return {
        "mae": float(difference.abs().mean().item()),
        "rmse": math.sqrt(mse),
        "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
    }


@torch.no_grad()
def _evaluate(
    model: TrSSSpatialSR,
    loader: DataLoader,
    device: torch.device,
    *,
    tile_size: int,
    tile_overlap: int,
) -> dict[str, Any]:
    model.eval()
    pixel_count = 0
    absolute_error = squared_error = 0.0
    bicubic_absolute_error = bicubic_squared_error = 0.0
    per_frame: list[dict[str, Any]] = []

    for index, batch in enumerate(loader):
        low = batch["low_res"]
        target = batch["target"]
        prediction = _predict_tiled(
            model, low, device, tile_size=tile_size, tile_overlap=tile_overlap
        )
        bicubic = F.interpolate(
            low, scale_factor=2, mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)
        diff = prediction - target
        baseline_diff = bicubic - target
        pixel_count += target.numel()
        absolute_error += float(diff.abs().sum().item())
        squared_error += float(diff.square().sum().item())
        bicubic_absolute_error += float(baseline_diff.abs().sum().item())
        bicubic_squared_error += float(baseline_diff.square().sum().item())
        per_frame.append(
            {
                "index": index,
                "width": int(target.shape[-1]),
                "height": int(target.shape[-2]),
                "metrics": _frame_metrics(prediction, target),
                "bicubic_metrics": _frame_metrics(bicubic, target),
            }
        )

    if pixel_count == 0:
        raise ValueError("Validation dataset contains no pixels.")
    mae = absolute_error / pixel_count
    mse = squared_error / pixel_count
    baseline_mae = bicubic_absolute_error / pixel_count
    baseline_mse = bicubic_squared_error / pixel_count
    return {
        "frames": len(per_frame),
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
    }


def train_spatial(
    train_manifest: str | Path,
    validation_manifest: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    epochs: int = 5,
    channels: int = 32,
    blocks: int = 6,
    patch_size: int = 96,
    learning_rate: float = 2e-4,
    tile_size: int = 256,
    tile_overlap: int = 16,
    seed: int = 20261014,
    force: bool = False,
) -> dict[str, Any]:
    """Train a local RGB-only model and publish artifacts only after success."""
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if not isinstance(patch_size, int) or isinstance(patch_size, bool) or patch_size < 2:
        raise ValueError("patch_size must be an integer of at least 2.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and greater than zero.")
    if tile_size < 2 or tile_overlap < 0 or tile_overlap * 2 >= tile_size:
        raise ValueError("tile_size must be >= 2 and tile_overlap less than half tile_size.")
    if channels < 8 or blocks < 1:
        raise ValueError("channels must be >= 8 and blocks must be >= 1.")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    train_path = Path(train_manifest).expanduser().resolve()
    validation_path = Path(validation_manifest).expanduser().resolve()
    train_rows = load_spatial_manifest(train_path)
    validation_rows = load_spatial_manifest(validation_path)
    _check_splits(train_rows, validation_rows)

    output = Path(output_dir).expanduser().resolve()
    if output == output.parent or not output.name:
        raise ValueError("output_dir must name a dedicated output folder.")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output}")
    if output.exists() and not force:
        raise FileExistsError(
            f"Output path already exists: {output}. Pass --force to replace it."
        )

    device = _resolve_device(device_name)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "xpu":
        torch.xpu.manual_seed(seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_data = TrSSSpatialFrameDataset(train_path)
    validation_data = TrSSSpatialFrameDataset(validation_path)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=1, shuffle=False)
    model = TrSSSpatialSR(channels=channels, blocks=blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    output.parent.mkdir(parents=True, exist_ok=True)
    transaction_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.spatial-train-", dir=output.parent)
    )
    staging = transaction_root / "payload"
    staging.mkdir()
    backup: Path | None = None
    history: list[dict[str, Any]] = []
    best_mae = float("inf")
    start = perf_counter()
    try:
        for epoch in range(1, epochs + 1):
            epoch_start = perf_counter()
            model.train()
            loss_sum = 0.0
            sample_count = 0
            for batch in train_loader:
                # Crop on CPU first to keep high-resolution PNGs off accelerator memory.
                low, target = _random_patch(batch["low_res"], batch["target"], patch_size)
                low, target = low.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = compose_spatial_prediction(model(low), low)
                loss = F.l1_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach().cpu().item())
                sample_count += 1
            if sample_count == 0:
                raise ValueError("Training manifest has no frames.")

            validation = _evaluate(
                model, validation_loader, device,
                tile_size=tile_size, tile_overlap=tile_overlap,
            )
            row = {
                "epoch": epoch,
                "training_l1": loss_sum / sample_count,
                "validation": validation["metrics"],
                "bicubic": validation["bicubic_metrics"],
                "validation_frames": validation["frames"],
                "beats_bicubic_on_mae": validation["beats_bicubic_on_mae"],
                "beats_bicubic_on_psnr": validation["beats_bicubic_on_psnr"],
                "epoch_seconds": perf_counter() - epoch_start,
                "per_frame_validation": validation["per_frame"],
            }
            history.append(row)
            checkpoint = {
                "format": SPATIAL_MODEL_FORMAT,
                "model_config": dict(model.config),
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in model.state_dict().items()
                },
                "epoch": epoch,
                "validation_metrics": validation["metrics"],
                "source_type": "user-recorded-gameplay-video-rgb-only",
                "limitations": [
                    "No game-renderer depth or motion vectors are consumed.",
                    "This spatial model does not learn temporal stability.",
                    "Metrics describe only the supplied held-out capture clips.",
                ],
            }
            torch.save(checkpoint, staging / "checkpoint-last.pt")
            if validation["metrics"]["mae"] < best_mae:
                best_mae = validation["metrics"]["mae"]
                torch.save(checkpoint, staging / "checkpoint-best.pt")

        report = {
            "schema_version": 1,
            "completed": True,
            "model_format": SPATIAL_MODEL_FORMAT,
            "output_dir": str(output),
            "device": str(device),
            "model_config": dict(model.config),
            "train_manifest": str(train_path),
            "train_manifest_sha256": _manifest_hash(train_path),
            "validation_manifest": str(validation_path),
            "validation_manifest_sha256": _manifest_hash(validation_path),
            "training_frames": len(train_data),
            "validation_frames": len(validation_data),
            "training_scenes": sorted({row["scene_id"] for row in train_rows}),
            "validation_scenes": sorted({row["scene_id"] for row in validation_rows}),
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "patch_size_low_resolution_pixels": patch_size,
            "tile_size_low_resolution_pixels": tile_size,
            "tile_overlap_low_resolution_pixels": tile_overlap,
            "history": history,
            "best_validation_mae": best_mae,
            "elapsed_seconds": perf_counter() - start,
            "limits": [
                "RGB-only 2x spatial reconstruction; no depth, motion, visibility mask, or temporal history.",
                "Input is formed by downsampling each recorded target frame, not by replaying the game's native low-resolution renderer.",
                "No live game hook, overlay, in-game FPS measurement, or Arc B580 performance claim is included.",
                "Keep footage and extracted frames local unless the game/content licence permits redistribution.",
            ],
        }
        (staging / "training-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
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
        description="Train a separate RGB-only TrSS 2x model from aligned image pairs or game video."
    )
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, xpu, or cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        report = train_spatial(
            args.train_manifest,
            args.validation_manifest,
            args.output_dir,
            device_name=args.device,
            epochs=args.epochs,
            channels=args.channels,
            blocks=args.blocks,
            patch_size=args.patch_size,
            learning_rate=args.learning_rate,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            seed=args.seed,
            force=args.force,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    root = args.output_dir.expanduser().resolve()
    print(json.dumps({
        "training_report": str(root / "training-report.json"),
        "best_checkpoint": str(root / "checkpoint-best.pt"),
        "device": report["device"],
        "epochs_completed": report["epochs_completed"],
        "best_validation_mae": report["best_validation_mae"],
        "history": report["history"],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
