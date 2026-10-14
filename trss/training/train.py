"""Train and resume the original TrSS model from local paired-render data."""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import TrSSFrameDataset
from .model import MODEL_FORMAT, TrSSTemporalSR, compose_prediction, model_size_report


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
        supported = (
            hasattr(torch.xpu, "is_bf16_supported") and torch.xpu.is_bf16_supported()
        )
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


def _tile_starts(length: int, patch: int) -> list[int]:
    """Choose patch starts that cover the full axis, including its far edge."""
    if patch >= length:
        return [0]
    starts = list(range(0, length - patch + 1, patch))
    far_edge = length - patch
    if starts[-1] != far_edge:
        starts.append(far_edge)
    return starts


def _crop_batch(
    batch: dict[str, torch.Tensor],
    patch_size: int,
    *,
    random_crop: bool,
    top: int | None = None,
    left: int | None = None,
) -> dict[str, torch.Tensor]:
    """Crop each sample independently, then apply flow-aware flips for training."""
    height, width = batch["features"].shape[-2:]
    patch = min(int(patch_size), height, width)
    if patch < 2:
        raise ValueError("Training patches must be at least 2x2 input pixels.")

    high_patch = patch * 2
    output: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("features", "low_res", "target", "history", "history_valid")
    }

    for index in range(batch["features"].shape[0]):
        if random_crop:
            sample_top = random.randint(0, height - patch)
            sample_left = random.randint(0, width - patch)
        else:
            sample_top = (height - patch) // 2 if top is None else top
            sample_left = (width - patch) // 2 if left is None else left

        if not (0 <= sample_top <= height - patch):
            raise ValueError("Crop top is outside the input frame.")
        if not (0 <= sample_left <= width - patch):
            raise ValueError("Crop left is outside the input frame.")

        high_top, high_left = sample_top * 2, sample_left * 2
        high_bottom, high_right = high_top + high_patch, high_left + high_patch
        sample = {
            "features": batch["features"][
                index : index + 1, :, sample_top : sample_top + patch,
                sample_left : sample_left + patch
            ],
            "low_res": batch["low_res"][
                index : index + 1, :, sample_top : sample_top + patch,
                sample_left : sample_left + patch
            ],
            "target": batch["target"][
                index : index + 1, :, high_top:high_bottom, high_left:high_right
            ],
            "history": batch["history"][
                index : index + 1, :, high_top:high_bottom, high_left:high_right
            ],
            "history_valid": batch["history_valid"][
                index : index + 1, :, high_top:high_bottom, high_left:high_right
            ],
        }

        if random_crop:
            # Flip every aligned image and mask together. Motion is a vector, so
            # reflecting the image also reflects the corresponding vector component.
            flip_x = random.random() < 0.5
            flip_y = random.random() < 0.5
            flip_dims = tuple(
                dimension
                for condition, dimension in ((flip_y, -2), (flip_x, -1))
                if condition
            )
            if flip_dims:
                sample = {
                    key: torch.flip(value, dims=flip_dims)
                    for key, value in sample.items()
                }
                if flip_x:
                    sample["features"][:, 4:5].neg_()
                if flip_y:
                    sample["features"][:, 5:6].neg_()

        for key, value in sample.items():
            output[key].append(value)

    return {key: torch.cat(values, dim=0) for key, values in output.items()}


def _validation_crops(
    batch: dict[str, torch.Tensor], patch_size: int
):
    """Yield a deterministic tiling so validation scores the whole frame, not its center."""
    height, width = batch["features"].shape[-2:]
    patch = min(int(patch_size), height, width)
    for top in _tile_starts(height, patch):
        for left in _tile_starts(width, patch):
            yield _crop_batch(
                batch, patch, random_crop=False, top=top, left=left
            )


def _training_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Charbonnier reconstruction plus an edge term helps retain crisp boundaries.
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
    model: TrSSTemporalSR,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    patch_size: int,
    amp: bool,
) -> dict[str, float]:
    """Run one pass and compare the network against the bicubic baseline."""
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "mae": 0.0,
        "mse": 0.0,
        "bicubic_mae": 0.0,
        "bicubic_mse": 0.0,
    }
    sample_count = 0

    for batch in loader:
        # Random crops and augmentation are used only for training. Validation
        # deterministically visits every region of each frame in bounded patches.
        # Stream validation tiles one at a time to keep host RAM bounded.
        crops = (
            (_crop_batch(batch, patch_size, random_crop=True),)
            if training
            else _validation_crops(batch, patch_size)
        )

        for cropped in crops:
            features = cropped["features"].to(device)
            current = cropped["low_res"].to(device)
            target = cropped["target"].to(device)
            history = cropped["history"].to(device)
            history_valid = cropped["history_valid"].to(device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):
                with _autocast_context(device, amp):
                    raw_head = model(features)
                # Composition and losses stay FP32 to reduce numerical instability.
                prediction = compose_prediction(
                    raw_head.float(), current.float(), history.float(), history_valid.float()
                )
                loss = _training_loss(prediction, target)

                if not bool(torch.isfinite(loss).all()):
                    raise RuntimeError(
                        "Training produced a non-finite loss. Stop the run and inspect "
                        "the input data, learning rate, and precision settings."
                    )

                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0, error_if_nonfinite=True
                    )
                    optimizer.step()

            # Report direct image metrics as well as the training objective. These
            # expose when a lower objective still fails to beat simple interpolation.
            with torch.no_grad():
                baseline = F.interpolate(
                    current.float(),
                    scale_factor=2,
                    mode="bicubic",
                    align_corners=False,
                ).clamp(0.0, 1.0)
                prediction_error = prediction.detach() - target.float()
                baseline_error = baseline - target.float()
                batch_metrics = {
                    "loss": loss.detach(),
                    "mae": prediction_error.abs().mean(),
                    "mse": prediction_error.square().mean(),
                    "bicubic_mae": baseline_error.abs().mean(),
                    "bicubic_mse": baseline_error.square().mean(),
                }

            batch_samples = int(features.shape[0])
            for name, value in batch_metrics.items():
                totals[name] += float(value.float().cpu()) * batch_samples
            sample_count += batch_samples

    if sample_count == 0:
        raise RuntimeError("The selected dataset has no training batches.")

    mean = {name: value / sample_count for name, value in totals.items()}
    # RGB targets are normalized to [0, 1], so peak signal is 1.0.
    mean["psnr_db"] = -10.0 * math.log10(max(mean["mse"], 1e-12))
    mean["bicubic_psnr_db"] = -10.0 * math.log10(max(mean["bicubic_mse"], 1e-12))
    del mean["mse"]
    del mean["bicubic_mse"]
    return mean


def _check_split_overlap(
    train_set: TrSSFrameDataset, validation_set: TrSSFrameDataset | None
) -> None:
    """Reject exact input/target frame reuse across training and validation splits."""
    if validation_set is None:
        return

    # Reusing the same input with a different target still leaks the scene/frame
    # into validation. Compare resolved paths from the manifest loader so aliases
    # such as relative paths and absolute paths resolve to the same file.
    for field, label in (
        ("target", "target"),
        ("low_res", "low-resolution input"),
    ):
        train_frames = {row[field] for row in train_set.rows}
        validation_frames = {row[field] for row in validation_set.rows}
        overlap = train_frames.intersection(validation_frames)
        if overlap:
            example = str(sorted(overlap)[0])
            raise ValueError(
                "Training and validation manifests reuse at least one exact "
                f"{label} frame. Move it to only one split; example: {example}"
            )


def _preflight_dataset(
    name: str, dataset: TrSSFrameDataset, *, require_uniform_dimensions: bool
) -> dict[str, Any]:
    """Decode every frame pair before training and summarize the actual data."""
    dimensions: dict[str, int] = {}
    frames_without_valid_history = 0
    depth_min = float("inf")
    depth_max = float("-inf")
    max_normalized_motion = 0.0

    for index in range(len(dataset)):
        sample = dataset[index]
        for tensor_name, tensor in sample.items():
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    f"{name} sample {index + 1} contains non-finite values in {tensor_name}."
                )

        _, height, width = sample["low_res"].shape
        if height < 2 or width < 2:
            raise ValueError(
                f"{name} sample {index + 1} is {width}x{height}; each input dimension "
                "must be at least 2 pixels."
            )
        size_key = f"{width}x{height}"
        dimensions[size_key] = dimensions.get(size_key, 0) + 1
        if not bool((sample["history_valid"] > 0).any()):
            frames_without_valid_history += 1

        depth = sample["features"][3]
        depth_min = min(depth_min, float(depth.min()))
        depth_max = max(depth_max, float(depth.max()))
        max_normalized_motion = max(
            max_normalized_motion, float(sample["features"][4:6].abs().max())
        )

    if require_uniform_dimensions and len(dimensions) > 1:
        raise ValueError(
            f"{name} contains mixed input resolutions {sorted(dimensions)}. "
            "Use --batch-size 1 or group each resolution into its own manifest."
        )

    return {
        "frames": len(dataset),
        "low_resolution_dimensions": dict(sorted(dimensions.items())),
        "frames_without_valid_history_pixels": frames_without_valid_history,
        "normalized_depth_range": [depth_min, depth_max],
        "max_abs_normalized_motion": max_normalized_motion,
        "all_tensors_finite": True,
    }


def _cpu_tree(value: Any) -> Any:
    """Move tensors nested in optimizer state to CPU for portable checkpoints."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    """Write to a temporary file first so interruption cannot corrupt the last checkpoint."""
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _atomic_json_write(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def _set_learning_rate(
    optimizer: torch.optim.Optimizer, base_rate: float, epoch: int, total_epochs: int
) -> float:
    """Use an epoch-indexed cosine curve, which can safely continue after resume."""
    progress = min(max(epoch - 1, 0), max(total_epochs - 1, 0))
    rate = base_rate * 0.5 * (1.0 + math.cos(math.pi * progress / max(total_epochs, 1)))
    for group in optimizer.param_groups:
        group["lr"] = rate
    return rate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train or resume the original TrSS 2x temporal reconstruction model."
    )
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate every dataset item and exit without training")
    parser.add_argument("--resume", type=Path, help="resume from checkpoint-last.pt")
    parser.add_argument("--device", default="auto", help="auto, xpu, cuda, or cpu")
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--context-blocks", type=int, default=4)
    parser.add_argument("--motion-scale-px", type=float, default=64.0)
    parser.add_argument("--patch-size", type=int, default=128, help="crop size in low-res pixels")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100, help="total target epoch, including resume")
    parser.add_argument("--learning-rate", type=float, help="initial learning rate; defaults to 2e-4")
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--amp", action="store_true", help="use BF16 autocast when supported")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patch_size < 2:
        parser.error("epochs/batch-size must be positive and patch-size must be at least 2.")
    if args.learning_rate is not None and not (
        math.isfinite(args.learning_rate) and args.learning_rate > 0
    ):
        parser.error("learning-rate must be finite and greater than zero.")
    if not (math.isfinite(args.motion_scale_px) and args.motion_scale_px > 0):
        parser.error("motion-scale-px must be finite and greater than zero.")
    if args.preflight_only and args.resume:
        parser.error("--resume cannot be combined with --preflight-only.")
    if not args.preflight_only and args.output_dir is None:
        parser.error("--output-dir is required unless --preflight-only is used.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(args.seed)

    train_path = args.train_manifest.expanduser().resolve()
    validation_path = (
        args.validation_manifest.expanduser().resolve()
        if args.validation_manifest
        else None
    )
    train_set = TrSSFrameDataset(train_path, motion_scale_px=args.motion_scale_px)
    validation_set = (
        TrSSFrameDataset(validation_path, motion_scale_px=args.motion_scale_px)
        if validation_path is not None
        else None
    )
    try:
        _check_split_overlap(train_set, validation_set)
    except ValueError as exc:
        parser.error(str(exc))

    if args.preflight_only:
        try:
            train_summary = _preflight_dataset(
                "training", train_set,
                require_uniform_dimensions=args.batch_size > 1,
            )
            validation_summary = (
                _preflight_dataset(
                    "validation", validation_set, require_uniform_dimensions=False
                )
                if validation_set is not None else None
            )
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(f"dataset preflight failed: {exc}")
        print(json.dumps({
            "preflight": "passed",
            "training": train_summary,
            "validation": validation_summary,
            "notice": (
                "Preflight validates the files and tensor contracts; it does not "
                "establish that the target renders are high quality or correctly paired."
            ),
        }, indent=2, allow_nan=False))
        return 0

    device = _resolve_device(args.device)
    if args.amp:
        # Fail before model creation if the chosen device cannot run BF16.
        _autocast_context(device, True)

    train_generator = torch.Generator().manual_seed(args.seed)
    validation_generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
    )
    validation_loader = (
        DataLoader(
            validation_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            generator=validation_generator,
        )
        if validation_set is not None
        else None
    )

    model = TrSSTemporalSR(
        channels=args.channels,
        blocks=args.blocks,
        context_blocks=args.context_blocks,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate or 2e-4)

    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.output_dir / "training-report.json"
    started = perf_counter()
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_score = float("inf")
    saved_run_config: dict[str, Any] = {}
    elapsed_before_resume = 0.0
    checkpoint: dict[str, Any] | None = None

    if args.resume:
        resume_path = args.resume.expanduser().resolve()
        if not resume_path.is_file():
            parser.error(f"resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != MODEL_FORMAT:
            parser.error(
                f"checkpoint format {checkpoint.get('format')!r} is not supported by "
                f"this model ({MODEL_FORMAT!r})."
            )
        if checkpoint.get("config") != model.config:
            parser.error(
                "checkpoint model settings differ from --channels/--blocks/"
                "--context-blocks; pass the exact original settings."
            )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        completed_epoch = int(checkpoint.get("epoch", 0))
        start_epoch = completed_epoch + 1
        if start_epoch > args.epochs:
            parser.error(
                f"checkpoint already reached epoch {completed_epoch}; increase --epochs "
                "to continue beyond it."
            )
        best_score = float(checkpoint.get("best_score", float("inf")))
        saved_run_config = checkpoint.get("run_config", {})
        elapsed_before_resume = float(saved_run_config.get("elapsed_seconds", 0.0))

        # Do not accidentally continue on a different dataset or input encoding.
        if saved_run_config:
            if saved_run_config.get("train_manifest") != str(train_path):
                parser.error("resume checkpoint uses a different training manifest.")
            saved_validation = saved_run_config.get("validation_manifest")
            current_validation = str(validation_path) if validation_path else None
            if saved_validation != current_validation:
                parser.error("resume checkpoint uses a different validation manifest.")
            if saved_run_config.get("motion_scale_px") != args.motion_scale_px:
                parser.error("resume checkpoint uses a different motion scale.")

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            print(
                "Notice: this older checkpoint has no optimizer state; resuming weights "
                "with a fresh AdamW optimizer.",
                flush=True,
            )

        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "train_loader_rng_state" in checkpoint:
            train_generator.set_state(checkpoint["train_loader_rng_state"])

        history = list(checkpoint.get("history", []))
        if not history and run_path.is_file():
            # Preserve reports made by the earlier checkpoint format when possible.
            try:
                old_report = json.loads(run_path.read_text(encoding="utf-8"))
                history = [
                    row for row in old_report.get("history", [])
                    if int(row.get("epoch", 0)) <= completed_epoch
                ]
                elapsed_before_resume = float(
                    old_report.get("elapsed_seconds", elapsed_before_resume)
                )
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                history = []

        old_base_rate = checkpoint.get("base_learning_rate")
        base_learning_rate = (
            args.learning_rate
            if args.learning_rate is not None
            else float(old_base_rate if old_base_rate is not None else 2e-4)
        )
        print(
            f"Resuming from epoch {completed_epoch}; target is epoch {args.epochs}.",
            flush=True,
        )
    else:
        base_learning_rate = args.learning_rate or 2e-4

    run_config: dict[str, Any] = {
        "format": MODEL_FORMAT,
        "device": str(device),
        "config": model.config,
        "model_size": model_size_report(model),
        "motion_scale_px": args.motion_scale_px,
        "patch_size_low_resolution": args.patch_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "base_learning_rate": base_learning_rate,
        "amp_bfloat16": args.amp,
        "train_manifest": str(train_path),
        "validation_manifest": str(validation_path) if validation_path else None,
        "history": history,
        "elapsed_seconds_before_resume": elapsed_before_resume,
        "resumed_from": str(args.resume.expanduser().resolve()) if args.resume else None,
    }

    # Restore RNGs after model construction; otherwise random initialization would
    # consume the state intended for the next crop and sample shuffle.
    if checkpoint is not None:
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "train_loader_rng_state" in checkpoint:
            train_generator.set_state(checkpoint["train_loader_rng_state"])

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = perf_counter()
        used_learning_rate = _set_learning_rate(
            optimizer, base_learning_rate, epoch, args.epochs
        )
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            patch_size=args.patch_size,
            amp=args.amp,
        )
        validation_metrics = None
        if validation_loader is not None:
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                device,
                optimizer=None,
                patch_size=args.patch_size,
                amp=False,
            )
        train_loss = train_metrics["loss"]
        validation_loss = (
            validation_metrics["loss"] if validation_metrics is not None else None
        )

        # A separate validation manifest is preferred; without one, best is based
        # on augmented training patches and is not a reliable generalization score.
        score = validation_loss if validation_loss is not None else train_loss
        improved = score < best_score
        if improved:
            best_score = score

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mae": train_metrics["mae"],
            "train_psnr_db": train_metrics["psnr_db"],
            "validation_loss": validation_loss,
            "validation_mae": (
                validation_metrics["mae"] if validation_metrics is not None else None
            ),
            "validation_psnr_db": (
                validation_metrics["psnr_db"] if validation_metrics is not None else None
            ),
            "validation_bicubic_mae": (
                validation_metrics["bicubic_mae"]
                if validation_metrics is not None else None
            ),
            "validation_bicubic_psnr_db": (
                validation_metrics["bicubic_psnr_db"]
                if validation_metrics is not None else None
            ),
            "learning_rate": used_learning_rate,
            "epoch_seconds": perf_counter() - epoch_started,
            "validation_tiled_full_frames": validation_loader is not None,
        }
        history.append(epoch_record)
        run_config["history"] = history
        run_config["elapsed_seconds_before_resume"] = run_config.get(
            "elapsed_seconds_before_resume", 0.0
        )
        run_config["elapsed_seconds"] = (
            float(run_config["elapsed_seconds_before_resume"])
            + perf_counter() - started
        )

        # Include optimizer and RNG state so a stop/restart continues the same run
        # instead of silently resetting AdamW or repeating the original shuffle.
        saved_checkpoint = {
            "format": MODEL_FORMAT,
            "config": model.config,
            "state_dict": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "best_score": best_score,
            "base_learning_rate": base_learning_rate,
            "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "train_loader_rng_state": train_generator.get_state(),
            "history": history,
            "run_config": run_config,
        }
        _atomic_torch_save(saved_checkpoint, args.output_dir / "checkpoint-last.pt")
        if improved:
            _atomic_torch_save(saved_checkpoint, args.output_dir / "checkpoint-best.pt")
        _atomic_json_write(run_config, run_path)
        print(json.dumps(epoch_record, allow_nan=False), flush=True)

    print(f"Training report: {run_path}")
    print(f"Last checkpoint: {args.output_dir / 'checkpoint-last.pt'}")
    print(f"Best checkpoint: {args.output_dir / 'checkpoint-best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
