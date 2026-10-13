"""Command-line interface for the XESS5 inference engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from PIL import Image

from . import __version__
from .backends.openvino_backend import BackendUnavailable
from .contracts import parse_value_range
from .engine import ImageInferenceEngine


def _range_argument(value: str) -> tuple[float, float]:
    try:
        return parse_value_range(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xess5",
        description="Local image inference engine targeting Intel Arc GPUs.",
    )
    parser.add_argument("--version", action="version", version=f"XESS5 {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "devices",
        help="list devices exposed by the OpenVINO runtime",
    )

    infer = commands.add_parser(
        "infer",
        help="run a one-input, one-output image model on a local device",
    )
    infer.add_argument("--model", required=True, type=Path, help="OpenVINO-supported model file")
    infer.add_argument("--input", required=True, type=Path, help="input image path")
    infer.add_argument("--output", required=True, type=Path, help="where to save the result")
    infer.add_argument("--device", default="GPU", help="OpenVINO device name (default: GPU)")
    infer.add_argument("--layout", choices=("NCHW", "NHWC"), default="NCHW")
    infer.add_argument(
        "--input-range", type=_range_argument, default=(0.0, 1.0),
        metavar="MIN:MAX", help="model input value range (default: 0:1)",
    )
    infer.add_argument(
        "--output-range", type=_range_argument, default=(0.0, 1.0),
        metavar="MIN:MAX", help="model output value range before conversion to pixels (default: 0:1)",
    )
    infer.add_argument(
        "--report", type=Path,
        help="optional path for a JSON report of the inference run",
    )
    return parser


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_devices() -> int:
    from .engine import available_devices

    devices = available_devices()
    print(json.dumps({"devices": devices}, indent=2))
    if not any(item["device"].upper().startswith("GPU") for item in devices):
        print(
            "No OpenVINO GPU device was reported. Check the Intel graphics driver "
            "and OpenVINO GPU plugin before attempting GPU inference.",
            file=sys.stderr,
        )
        return 2
    return 0


def _run_inference(args: argparse.Namespace) -> int:
    if not args.input.is_file():
        raise FileNotFoundError(f"Input image does not exist: {args.input}")
    if args.output.resolve() == args.input.resolve():
        raise ValueError("Output path must differ from input path; the source image is never overwritten.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(args.input) as opened:
        original = opened.convert("RGB")

    engine = ImageInferenceEngine(
        args.model,
        device=args.device,
        layout=args.layout,
        input_range=args.input_range,
        output_range=args.output_range,
    )
    whole_run_start = perf_counter()
    result = engine.infer(original)
    elapsed_ms = (perf_counter() - whole_run_start) * 1000.0
    result.image.save(args.output)

    report: dict[str, object] = {
        "engine": "XESS5",
        "engine_version": __version__,
        "operation": "single_frame_image_inference",
        "model": result.model_path,
        "device": result.device,
        "device_name": result.device_name,
        "layout": args.layout,
        "input_range": list(args.input_range),
        "output_range": list(args.output_range),
        "input_path": str(args.input.resolve()),
        "output_path": str(args.output.resolve()),
        "input_size": list(original.size),
        "model_input_size": list(result.input_size),
        "output_size": list(result.output_size),
        "inference_ms": round(result.inference_ms, 3),
        "total_ms": round(elapsed_ms, 3),
        "inference_fps_equivalent": (
            round(1000.0 / result.inference_ms, 2) if result.inference_ms > 0 else None
        ),
        "note": (
            "Single-frame model inference only. This does not establish live-game integration, "
            "temporal neural rendering, DLSS 5 compatibility, or an FPS improvement in a game."
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
        if args.command == "infer":
            return _run_inference(args)
        parser.error("unknown command")
    except (BackendUnavailable, FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"XESS5 error: {exc}", file=sys.stderr)
        return 2
    return 0
