"""Run a trained first-party TrSS checkpoint on one offline frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
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


def _device_name(device: torch.device) -> str | None:
    """Return the runtime's device name when that backend exposes one."""
    try:
        if device.type == "xpu":
            return str(torch.xpu.get_device_name(device))
        if device.type == "cuda":
            return str(torch.cuda.get_device_name(device))
    except (AttributeError, RuntimeError, TypeError):
        return None
    return "CPU" if device.type == "cpu" else None


def _reset_peak_device_memory(device: torch.device) -> None:
    """Reset the optional device allocator peak counter without making it required."""
    try:
        if device.type == "xpu":
            torch.xpu.reset_peak_memory_stats(device)
        elif device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
    except (AttributeError, RuntimeError, TypeError):
        pass


def _device_memory_snapshot(device: torch.device) -> dict[str, int | None]:
    """Read model-process allocator data; native GPU-wide VRAM is a different metric."""
    result: dict[str, int | None] = {
        "allocated_bytes": None,
        "reserved_bytes": None,
        "peak_allocated_bytes": None,
    }
    try:
        if device.type == "xpu":
            result["allocated_bytes"] = int(torch.xpu.memory_allocated(device))
            result["reserved_bytes"] = int(torch.xpu.memory_reserved(device))
            result["peak_allocated_bytes"] = int(torch.xpu.max_memory_allocated(device))
        elif device.type == "cuda":
            result["allocated_bytes"] = int(torch.cuda.memory_allocated(device))
            result["reserved_bytes"] = int(torch.cuda.memory_reserved(device))
            result["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    except (AttributeError, RuntimeError, TypeError):
        # Some PyTorch accelerator runtimes do not implement every allocator API.
        # Leave unsupported values null rather than substituting a different measure.
        pass
    return result


def _timing_summary(values: list[float]) -> dict[str, float]:
    """Summarize individual synchronized samples instead of timing the whole loop only."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": round(float(np.min(array)), 6),
        "mean": round(float(np.mean(array)), 6),
        "median": round(float(np.median(array)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "p99": round(float(np.percentile(array, 99)), 6),
        "max": round(float(np.max(array)), 6),
    }


def _append_telemetry_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append a complete run as one JSONL record for longitudinal comparisons."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\\n") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\\n")
        stream.flush()


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
    telemetry_jsonl_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run the trained network; history must already be reprojected to this frame."""
    call_started = perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
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

    input_setup_started = perf_counter()
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
    input_setup_ms = (perf_counter() - input_setup_started) * 1000.0

    # Keep steady-state device work separate from input preparation and file I/O.
    # The peak counter starts after model/input allocation, so the final total reports
    # baseline allocated memory separately from additional inference-time peak memory.
    memory_before = _device_memory_snapshot(device)
    _reset_peak_device_memory(device)
    inference_times: list[float] = []
    composition_times: list[float] = []
    model_composition_times: list[float] = []
    iteration_samples: list[dict[str, float | int]] = []
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
                composition_ms = max(0.0, frame_ms - inference_ms)
                inference_times.append(inference_ms)
                composition_times.append(composition_ms)
                model_composition_times.append(frame_ms)
                iteration_samples.append({
                    "iteration": index - warmup + 1,
                    "model_inference_ms": round(inference_ms, 6),
                    "composition_ms": round(composition_ms, 6),
                    "model_plus_composition_ms": round(frame_ms, 6),
                    "fps_equivalent": round(1000.0 / frame_ms, 6) if frame_ms > 0 else 0.0,
                })

    assert output is not None
    output_write_started = perf_counter()
    output_array = (
        output[0].detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
    ).round().astype(np.uint8)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_array, mode="RGB").save(output_file)
    output_write_ms = (perf_counter() - output_write_started) * 1000.0

    memory_after = _device_memory_snapshot(device)
    info: dict[str, Any] = {
        "schema_version": "trss.model-performance/1.0",
        "started_utc": started_utc,
        "format": MODEL_FORMAT,
        "device": str(device),
        "device_name": _device_name(device),
        "runtime": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "platform": platform.platform(),
            "cpu_threads": int(torch.get_num_threads()),
        },
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
        "input_setup_ms": input_setup_ms,
        "timing_ms": {
            "model_inference": _timing_summary(inference_times),
            "composition": _timing_summary(composition_times),
            "model_plus_composition": _timing_summary(model_composition_times),
        },
        "mean_inference_ms": mean(inference_times),
        "mean_composition_ms": mean(composition_times),
        "mean_model_composition_ms": mean(model_composition_times),
        "median_model_composition_ms": float(np.median(model_composition_times)),
        "p95_model_composition_ms": float(np.percentile(model_composition_times, 95)),
        "p99_model_composition_ms": float(np.percentile(model_composition_times, 99)),
        "model_composition_fps_equivalent": 1000.0 / mean(model_composition_times),
        "theoretical_60_fps_budget_ms": 1000.0 / 60.0,
        "iterations_measured": iteration_samples,
        "device_memory": {
            "allocated_before_benchmark_bytes": memory_before["allocated_bytes"],
            "reserved_before_benchmark_bytes": memory_before["reserved_bytes"],
            "allocated_after_benchmark_bytes": memory_after["allocated_bytes"],
            "reserved_after_benchmark_bytes": memory_after["reserved_bytes"],
            "peak_allocated_during_benchmark_bytes": memory_after["peak_allocated_bytes"],
            "scope": "PyTorch allocator used by this process; not total adapter VRAM or another game's memory.",
        },
        "output_encode_write_ms": output_write_ms,
        "benchmark_call_elapsed_ms": (perf_counter() - call_started) * 1000.0,
        "output": str(output_file),
        "note": (
            "This is a repeated offline single-frame model microbenchmark. Timings include synchronized "
            "model inference plus the Python/Torch output compositor, but exclude live capture, file decoding, "
            "input setup, initial device transfer, and output encoding. The FPS-equivalent is model-path "
            "capacity only, not in-game FPS or proof of live TrSS integration. GPU allocator fields are "
            "process allocation data, not total adapter VRAM; unavailable counters remain null."
        ),
    }
    if telemetry_jsonl_path is not None:
        telemetry_file = Path(telemetry_jsonl_path).expanduser().resolve()
        _append_telemetry_jsonl(telemetry_file, info)
        info["telemetry_jsonl"] = str(telemetry_file)
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
    parser.add_argument("--report", type=Path, help="optional detailed JSON report")
    parser.add_argument(
        "--telemetry-jsonl", type=Path,
        help="optional append-only log for comparing repeated TrSS model runs",
    )
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
        telemetry_jsonl_path=args.telemetry_jsonl,
    )
    print(json.dumps(info, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
