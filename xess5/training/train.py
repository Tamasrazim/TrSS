"""Train the original XESS5 model with local paired-render data."""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import XESS5FrameDataset
from .model import MODEL_FORMAT, XESS5TemporalSR, compose_prediction, model_size_report


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
            "PyTorch cannot see an XPU. Install the Intel-GPU PyTorch wheels and "
            "confirm the graphics driver before training on the Arc B580."
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch install.")
    if device.type not in ("cpu", "cuda", "xpu"):
        raise ValueError("device must be auto, cpu, cuda, xpu, or an explicit device index.")
    return device


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if device.type == "xpu":
        supported = torch.xpu.is_bf16_supported()
    elif device.type == "cuda":
        supported = torch.cuda.is_bf16_supported()
    else:
        supported = False
    if not supported:
        raise RuntimeError(
            "--amp currently requires BF16 support on the selected XPU or CUDA device. "
            "Run without --amp rather than silently changing precision."
        )
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True)


def _crop_batch(
    batch: dict[str, torch.Tensor],
    patch_size: int,
    *,
    random_crop: bool,
) -> dict[str, torch.Tensor]:
    """Crop on CPU before transfer so full-resolution targets stay out of GPU memory."""
    height, width = batch["features"].shape[-2:]
    patch = min(int(patch_size), height, width)
    if patch < 2:
        raise ValueError("Training patches must be at least 2x2 input pixels.")
    if random_crop:
        top = random.randint(0, height - patch)
        left = random.randint(0, width - patch)
    else:
        top = (height - patch) // 2
        left = (width - patch) // 2

    scale = 2
    cropped = {
        "features": batch["features"][..., top : top + patch, left : left + patch],
        "low_res": batch["low_res"][..., top : top + patch, left : left + patch],
    }
    high_top, high_left = top * scale, left * scale
    high_bottom, high_right = high_top + patch * scale, high_left + patch * scale
    for key in ("target", "history", "history_valid"):
        cropped[key] = batch[key][..., high_top:high_bottom, high_left:high_right]
    return cropped


def _training_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Charbonnier reconstruction plus a modest edge term helps retain crisp boundaries.
    difference = prediction.float() - target.float()
    pixel = torch.sqrt(difference.square() + 1e-6).mean()
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    edge = F.l1_loss(pred_x.float(), target_x.float())
    edge = edge + F.l1_loss(pred_y.float(), target_y.float())
    return pixel + edge * 0.1


def _run_epoch(
    model: XESS5TemporalSR,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    patch_size: int,
    amp: bool,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    samples = 0

    for batch in loader:
        # Keep crop selection and image decoding on the CPU.
        batch = _crop_batch(batch, patch_size, random_crop=training)
        features = batch["features"].to(device)
        current = batch["low_res"].to(device)
        target = batch["target"].to(device)
        history = batch["history"].to(device)
        history_valid = batch["history_valid"].to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with _autocast_context(device, amp):
                raw_head = model(features)
            # Composition and loss stay FP32 to reduce numerical instability.
            prediction = compose_prediction(
                raw_head.float(), current.float(), history.float(), history_valid.float()
            )
            loss = _training_loss(prediction, target)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        samples += 1

    if samples == 0:
        raise RuntimeError("The selected dataset has no training batches.")
    return total_loss / samples


def _cpu_state_dict(model: XESS5TemporalSR) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the original XESS5 2x temporal reconstruction model."
    )
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", help="auto, xpu, cuda, or cpu")
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--context-blocks", type=int, default=4)
    parser.add_argument("--motion-scale-px", type=float, default=64.0)
    parser.add_argument("--patch-size", type=int, default=128, help="crop size in low-res pixels")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--amp", action="store_true", help="use BF16 autocast when supported")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patch_size < 2:
        parser.error("epochs/batch-size must be positive and patch-size must be at least 2.")
    if not (0.0 < args.learning_rate < float("inf")):
        parser.error("learning-rate must be finite and greater than zero.")
    if not (0.0 < args.motion_scale_px < float("inf")):
        parser.error("motion-scale-px must be finite and greater than zero.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(args.seed)

    device = _resolve_device(args.device)
    if args.amp:
        # Fail before loading the dataset if the chosen device cannot run BF16.
        _autocast_context(device, True)

    train_set = XESS5FrameDataset(
        args.train_manifest, motion_scale_px=args.motion_scale_px
    )
    validation_set = (
        XESS5FrameDataset(
            args.validation_manifest, motion_scale_px=args.motion_scale_px
        )
        if args.validation_manifest
        else None
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    validation_loader = (
        DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=0)
        if validation_set is not None
        else None
    )

    model = XESS5TemporalSR(
        channels=args.channels,
        blocks=args.blocks,
        context_blocks=args.context_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config: dict[str, Any] = {
        "format": MODEL_FORMAT,
        "device": str(device),
        "config": model.config,
        "model_size": model_size_report(model),
        "motion_scale_px": args.motion_scale_px,
        "patch_size_low_resolution": args.patch_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "amp_bfloat16": args.amp,
        "train_manifest": str(args.train_manifest.resolve()),
        "validation_manifest": (
            str(args.validation_manifest.resolve()) if args.validation_manifest else None
        ),
        "history": [],
    }
    best_score = float("inf")
    run_path = args.output_dir / "training-report.json"
    started = perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_started = perf_counter()
        train_loss = _run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            patch_size=args.patch_size,
            amp=args.amp,
        )
        validation_loss = None
        if validation_loader is not None:
            validation_loss = _run_epoch(
                model,
                validation_loader,
                device,
                optimizer=None,
                patch_size=args.patch_size,
                amp=args.amp,
            )
        scheduler.step()

        # A separate validation manifest is preferred. Without one, best is train-loss best.
        score = validation_loss if validation_loss is not None else train_loss
        checkpoint = {
            "format": MODEL_FORMAT,
            "config": model.config,
            "state_dict": _cpu_state_dict(model),
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        }
        torch.save(checkpoint, args.output_dir / "checkpoint-last.pt")
        if score < best_score:
            best_score = score
            torch.save(checkpoint, args.output_dir / "checkpoint-best.pt")

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": perf_counter() - epoch_started,
        }
        run_config["history"].append(epoch_record)
        run_config["elapsed_seconds"] = perf_counter() - started
        run_path.write_text(
            json.dumps(run_config, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(epoch_record, allow_nan=False), flush=True)

    print(f"Training report: {run_path}")
    print(f"Best checkpoint: {args.output_dir / 'checkpoint-best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
