"""Command-line interface for the XESS5 Intel Arc neural inference engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import __version__
from .backends.openvino_backend import BackendUnavailable
from .contracts import parse_value_range
from .engine import ImageInferenceEngine, inspect_model, available_devices


def _range_argument(value: str) -> tuple[float, float]:
    try:
        return parse_value_range(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xess5",
        description="Local neural inference engine targeting Intel Arc GPUs.",
    )
    parser.add_argument("--version", action="version", version=f"XESS5 {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "devices",
        help="show the GPU devices visible to OpenVINO",
    )

    inspect = commands.add_parser(
        "inspect",
        help="inspect model input/output shapes without running inference",
    )
    inspect.add_argument("--model", required=True, type=Path, help="local OpenVINO/ONNX model")

    infer = commands.add_parser(
        "infer",
        help="run and benchmark a one-input, one-output image model locally",
    )
    infer.add_argument("--model", required=True, type=Path, help="OpenVINO-supported model file")
    infer.add_argument("--input", required=True, type=Path, help="input image path")
    infer.add_argument("--output", required=True, type=Path, help="where to save the final result")
    infer.add_argument(
        "--device",
        default="ARC",
        help="ARC (default) or an exact OpenVINO device ID such as GPU.0",
    )
    infer.add_argument("--layout", choices=("NCHW", "NHWC"), default="NCHW")
    infer.add_argument("--input-channel-order", choices=("RGB", "BGR"), default="RGB")
    infer.add_argument("--output-channel-order", choices=("RGB", "BGR"), default="RGB")
    infer.add_argument(
        "--input-range", type=_range_argument, default=(0.0, 1.0),
        metavar="MIN:MAX", help="model input value range (default: 0:1)",
    )
    infer.add_argument(
        "--output-range", type=_range_argument, default=(0.0, 1.0),
        metavar="MIN:MAX", help="model output value range before conversion to pixels (default: 0:1)",
    )
    infer.add_argument(
        "--warmup", type=int, default=3,
        help="inference calls to discard before measurements (default: 3)",
    )
    infer.add_argument(
        "--iterations", type=int, default=10,
        help="measured inference calls (default: 10)",
    )
    infer.add_argument(
        "--report", type=Path,
        help="optional path for a JSON report of timing and model information",
    )
    return parser


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a report atomically so an interrupted write doesn't leave half a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run_devices() -> int:
    devices = available_devices()
    print(json.dumps({"devices": devices}, indent=2))
    if not any(item["device"].upper().startswith("GPU") for item in devices):
        print(
            "No OpenVINO GPU device was reported. Check the Intel graphics driver "
            "and OpenVINO GPU plugin before attempting GPU inference.",
            file=sys.stderr,
        )
        return 2
    if not any("arc" in item["name"].casefold() for item in devices):
        print(
            "OpenVINO lists GPU hardware, but no device name contains 'Arc'. "
            "XESS5 will not guess which adapter is the Arc GPU. Inspect the device "
            "list and pass its exact GPU.X ID if the driver reports a different name.",
            file=sys.stderr,
        )
        return 2
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    report = inspect_model(args.model)
    print(json.dumps(report, indent=2))
    if not report["supported_by_current_image_engine"]:
        print(
            "This model does not match the current single-image contract. "
            "A dedicated adapter is required for models with multiple inputs/outputs "
            "or non-image tensor shapes.",
            file=sys.stderr,
        )
        return 3
    return 0


def _ensure_safe_paths(
    input_path: Path,
    model_path: Path,
    output_path: Path,
    report_path: Path | None,
) -> None:
    """Do not let output or diagnostics overwrite any source or model file."""
    protected = {
        input_path.resolve(),
        model_path.resolve(),
    }
    # OpenVINO IR often pairs an XML graph with a BIN weight file.
    if model_path.suffix.casefold() == ".xml":
        paired_weights = model_path.with_suffix(".bin")
        if paired_weights.exists():
            protected.add(paired_weights.resolve())

    if output_path.resolve() in protected:
        raise ValueError(
            "Output path must differ from the source image and every model file."
        )
    if report_path is not None:
        resolved_report = report_path.resolve()
        if resolved_report in protected or resolved_report == output_path.resolve():
            raise ValueError(
                "Report path must differ from the source image, model files, and output image."
            )


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _run_inference(args: argparse.Namespace) -> int:
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input image does not exist: {args.input}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model file does not exist: {args.model}")

    _ensure_safe_paths(args.input, args.model, args.output, args.report)

    # Decode once; every run receives the same pixels so comparisons are repeatable.
    try:
        with Image.open(args.input) as opened:
            original = opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not decode input image {args.input}: {exc}") from exc

    model_load_start = perf_counter()
    engine = ImageInferenceEngine(
        args.model,
        device=args.device,
        layout=args.layout,
        input_range=args.input_range,
        output_range=args.output_range,
        input_channel_order=args.input_channel_order,
        output_channel_order=args.output_channel_order,
    )
    model_load_ms = (perf_counter() - model_load_start) * 1000.0

    cold_start = perf_counter()
    last_result = engine.infer(original)
    first_call_wall_ms = (perf_counter() - cold_start) * 1000.0

    # The first call includes compilation on many devices. Treat it as the first
    # warm-up by default, but preserve its timing in the report for transparency.
    warmup_performed = 1
    for _ in range(max(0, args.warmup - 1)):
        engine.infer(original)
        warmup_performed += 1

    measured_inference_ms: list[float] = []
    measured_wall_ms: list[float] = []
    for _ in range(args.iterations):
        call_start = perf_counter()
        last_result = engine.infer(original)
        measured_wall_ms.append((perf_counter() - call_start) * 1000.0)
        measured_inference_ms.append(last_result.inference_ms)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    last_result.image.save(args.output)

    inference_median = float(np.median(measured_inference_ms))
    wall_median = float(np.median(measured_wall_ms))
    report: dict[str, object] = {
        "engine": "XESS5",
        "engine_version": __version__,
        "operation": "single_frame_model_inference",
        "model": last_result.model_path,
        "device_requested": args.device,
        "device_selected": last_result.device,
        "device_name": last_result.device_name,
        "layout": args.layout,
        "input_channel_order": args.input_channel_order,
        "output_channel_order": args.output_channel_order,
        "input_range": list(args.input_range),
        "output_range": list(args.output_range),
        "input_path": str(args.input.resolve()),
        "output_path": str(args.output.resolve()),
        "input_size": list(original.size),
        "model_input_size": list(last_result.input_size),
        "output_size": list(last_result.output_size),
        "model_load_ms": round(model_load_ms, 3),
        "first_call_wall_ms": round(first_call_wall_ms, 3),
        "warmup_requested": args.warmup,
        "warmup_performed": warmup_performed,
        "iterations": args.iterations,
        "inference_ms": {
            "min": round(min(measured_inference_ms), 3),
            "median": round(inference_median, 3),
            "p95": round(_percentile(measured_inference_ms, 95), 3),
            "max": round(max(measured_inference_ms), 3),
        },
        "end_to_end_call_ms": {
            "min": round(min(measured_wall_ms), 3),
            "median": round(wall_median, 3),
            "p95": round(_percentile(measured_wall_ms, 95), 3),
            "max": round(max(measured_wall_ms), 3),
        },
        "inference_fps_equivalent": (
            round(1000.0 / inference_median, 2) if inference_median > 0 else None
        ),
        "note": (
            "Isolated image-model inference only. This does not establish live-game "
            "integration, temporal neural rendering, DLSS 5 compatibility, or any "
            "FPS improvement in a game. GPU execution is requested explicitly; CPU "
            "fallback is refused."
        ),
    }

    print(json.dumps(report, indent=2))
    if args.report:
        _write_json(args.report, report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "devices":
            return _run_devices()
        if args.command == "inspect":
            return _run_inspect(args)
        if args.command == "infer":
            return _run_inference(args)
        parser.error("unknown command")
    except (BackendUnavailable, FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"XESS5 error: {exc}", file=sys.stderr)
        return 2
    return 0
