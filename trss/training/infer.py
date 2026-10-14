"""Run a trained first-party TrSS checkpoint on one offline frame."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .data import _read_depth, _read_mask, _read_motion, _read_rgb
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
        raise RuntimeError("PyTorch cannot see an XPU; confirm the Intel GPU driver and XPU wheels.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch install.")
    if device.type not in ("cpu", "cuda", "xpu"):
        raise ValueError("device must be auto, cpu, cuda, xpu, or an explicit device index.")
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> tuple[TrSSTemporalSR, dict[str, Any]]:
    # The checkpoint is a local tensor dictionary written by the TrSS trainer.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != MODEL_FORMAT:
        raise ValueError(
            f"Unsupported checkpoint format. Expected '{MODEL_FORMAT}' from the TrSS trainer."
        )
    config = checkpoint.get("config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing the model config or tensor state dictionary.")
    model = TrSSTemporalSR(
        channels=int(config["channels"]),
        blocks=int(config["blocks"]),
        context_blocks=int(config["context_blocks"]),
    )
    model.load_state_dict(state_dict, strict=True)
    return model, checkpoint


def run_frame(
    *,
    checkpoint_path: str | Path,
    color_path: str | Path,
    depth_path: str | Path,
    motion_path: str | Path,
    output_path: str | Path,
    device_name: str = "xpu",
    history_path: str | Path | None = None,
    history_valid_path: str | Path | None = None,
    motion_scale_px: float = 64.0,
    warmup: int = 2,
    iterations: int = 5,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the trained network; history must already be reprojected to this frame."""
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    color_file = Path(color_path).expanduser().resolve()
    depth_file = Path(depth_path).expanduser().resolve()
    motion_file = Path(motion_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    for label, path in (
        ("checkpoint", checkpoint_file),
        ("color", color_file),
        ("depth", depth_file),
        ("motion", motion_file),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
    if (history_path is None) != (history_valid_path is None):
        raise ValueError("--history and --history-valid must be supplied together.")
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be at least one.")
    if not np.isfinite(motion_scale_px) or motion_scale_px <= 0:
        raise ValueError("motion_scale_px must be finite and greater than zero.")

    color = _read_rgb(color_file)
    height, width, _ = color.shape
    depth = _read_depth(depth_file, height, width)
    motion = _read_motion(motion_file, height, width, float(motion_scale_px))
    output_height, output_width = height * 2, width * 2

    if history_path is None:
        # No supplied history means an explicit first frame, camera cut, or reset.
        history = np.zeros((3, output_height, output_width), dtype=np.float32)
        history_valid = np.zeros((1, output_height, output_width), dtype=np.float32)
    else:
        history_file = Path(history_path).expanduser().resolve()
        valid_file = Path(history_valid_path).expanduser().resolve()
        if not history_file.is_file() or not valid_file.is_file():
            raise FileNotFoundError("History image and validity mask must both exist.")
        history_rgb = _read_rgb(history_file)
        if history_rgb.shape[:2] != (output_height, output_width):
            raise ValueError("History must be 2x input size and already warped to the current view.")
        history = np.ascontiguousarray(history_rgb.transpose(2, 0, 1))
        history_valid = _read_mask(valid_file, output_height, output_width)

    current_tensor = torch.from_numpy(np.ascontiguousarray(color.transpose(2, 0, 1)))
    depth_tensor = torch.from_numpy(depth)
    motion_tensor = torch.from_numpy(motion)
    history_tensor = torch.from_numpy(history)
    valid_tensor = torch.from_numpy(history_valid)
    history_lr = F.interpolate(
        history_tensor.unsqueeze(0), size=(height, width), mode="area"
    ).squeeze(0)
    valid_lr = F.interpolate(
        valid_tensor.unsqueeze(0), size=(height, width), mode="area"
    ).squeeze(0)
    features = torch.cat(
        (current_tensor, depth_tensor, motion_tensor, history_lr, valid_lr), dim=0
    ).unsqueeze(0)

    device = _resolve_device(device_name)
    model, checkpoint = _load_checkpoint(checkpoint_file)
    model = model.to(device).eval()
    features = features.to(device)
    current_tensor = current_tensor.unsqueeze(0).to(device)
    history_tensor = history_tensor.unsqueeze(0).to(device)
    valid_tensor = valid_tensor.unsqueeze(0).to(device)

    # Separate neural inference time from the complete output-composition path.
    inference_times: list[float] = []
    frame_times: list[float] = []
    output: torch.Tensor | None = None
    with torch.inference_mode():
        for index in range(warmup + iterations):
            _synchronize(device)
            started = perf_counter()
            raw_head = model(features)
            _synchronize(device)
            inference_ms = (perf_counter() - started) * 1000.0
            output = compose_prediction(
                raw_head.float(),
                current_tensor.float(),
                history_tensor.float(),
                valid_tensor.float(),
            )
            _synchronize(device)
            frame_ms = (perf_counter() - started) * 1000.0
            if index >= warmup:
                inference_times.append(inference_ms)
                frame_times.append(frame_ms)

    assert output is not None
    output_array = (
        output[0].detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
    ).round().astype(np.uint8)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_array, mode="RGB").save(output_file)

    sorted_frame_times = sorted(frame_times)
    p95_index = min(len(sorted_frame_times) - 1, int(np.ceil(0.95 * len(sorted_frame_times))) - 1)
    info: dict[str, Any] = {
        "format": MODEL_FORMAT,
        "device": str(device),
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": _sha256(checkpoint_file),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_config": model.config,
        "model_size": model_size_report(model),
        "input_size": [width, height],
        "output_size": [output_width, output_height],
        "history_supplied": history_path is not None,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "mean_inference_ms": mean(inference_times),
        "mean_frame_path_ms": mean(frame_times),
        "median_frame_path_ms": float(np.median(frame_times)),
        "p95_frame_path_ms": float(sorted_frame_times[p95_index]),
        "offline_fps_equivalent": 1000.0 / mean(frame_times),
        "output": str(output_file),
        "note": "Offline frame timing; not in-game FPS or proof of Arc performance.",
    }
    if report_path is not None:
        report_file = Path(report_path).expanduser().resolve()
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(info, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        info["report"] = str(report_file)
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an TrSS checkpoint on a local frame.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--color", required=True, type=Path)
    parser.add_argument("--depth", required=True, type=Path)
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--history", type=Path, help="already-reprojected previous output at 2x size")
    parser.add_argument("--history-valid", type=Path, help="2x resolution validity mask")
    parser.add_argument("--motion-scale-px", type=float, default=64.0)
    parser.add_argument("--device", default="xpu", help="xpu (default), cuda, cpu, or auto")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    info = run_frame(
        checkpoint_path=args.checkpoint,
        color_path=args.color,
        depth_path=args.depth,
        motion_path=args.motion,
        output_path=args.output,
        device_name=args.device,
        history_path=args.history,
        history_valid_path=args.history_valid,
        motion_scale_px=args.motion_scale_px,
        warmup=args.warmup,
        iterations=args.iterations,
        report_path=args.report,
    )
    print(json.dumps(info, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
