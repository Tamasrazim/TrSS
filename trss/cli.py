"""Command-line interface for the TrSS Intel Arc neural inference engine."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import __version__
from .backends.directx12_backend import run_dx12_probe
from .backends.openvino_backend import BackendUnavailable
from .backends.xmx_engine import XMXBackendUnavailable, XMXFrameEngine
from .contracts import parse_value_range
from .engine import ImageInferenceEngine, inspect_model, available_devices
from .tensor_engine import TensorInferenceEngine
from .rendering import FramePacket
from .rendering.adapter import (
    AdapterSpecError,
    ModelAdapterSpec,
    adapt_frame,
    preview_from_outputs,
)
from .rendering.hnet import HNetContractError, HNetControls, HNetFrameEngine
from .rendering.sequence import run_sequence


def _range_argument(value: str) -> tuple[float, float]:
    try:
        return parse_value_range(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trss",
        description="Local neural inference engine targeting Intel Arc GPUs.",
    )
    parser.add_argument("--version", action="version", version=f"TrSS {__version__}")
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

    tensor_infer = commands.add_parser(
        "infer-tensors",
        help="run a model with multiple named tensor inputs and outputs",
    )
    tensor_infer.add_argument("--model", required=True, type=Path, help="local OpenVINO/ONNX model")
    tensor_infer.add_argument(
        "--input", dest="tensor_inputs", action="append", required=True,
        metavar="NAME=FILE.npy",
        help="model input name and a local NumPy .npy tensor; repeat for every model input",
    )
    tensor_infer.add_argument("--output-dir", required=True, type=Path, help="directory for output .npy tensors")
    tensor_infer.add_argument("--device", default="ARC", help="ARC (default) or exact GPU.X ID")
    tensor_infer.add_argument("--warmup", type=int, default=2, help="warm-up calls after model compilation")
    tensor_infer.add_argument("--iterations", type=int, default=10, help="measured calls (default: 10)")
    tensor_infer.add_argument("--report", type=Path, help="optional JSON timing report")

    render_frame = commands.add_parser(
        "render-frame",
        help="adapt a captured frame and real auxiliary buffers to a multi-input neural model",
    )
    render_frame.add_argument("--model", required=True, type=Path, help="local OpenVINO/ONNX model")
    render_frame.add_argument("--spec", required=True, type=Path, help="explicit JSON model-input adapter spec")
    render_frame.add_argument("--color", required=True, type=Path, help="captured RGB frame image")
    render_frame.add_argument("--depth", type=Path, help="optional depth array in .npy format")
    render_frame.add_argument("--motion", type=Path, help="optional HxWx2 motion-vector .npy")
    render_frame.add_argument("--noise", type=Path, help="optional HxWx3 noise tensor in .npy")
    render_frame.add_argument("--history", type=Path, help="optional previous output frame in HxWx3 .npy")
    render_frame.add_argument("--conditioning", type=Path, help="optional conditioning vector in .npy")
    render_frame.add_argument("--frame-id", type=int, default=0)
    render_frame.add_argument("--timestamp", type=float, default=0.0, help="source frame timestamp in seconds")
    render_frame.add_argument("--scene-id", help="optional stable ID for the current scene")
    render_frame.add_argument("--camera-cut", action="store_true", help="mark this frame as a temporal discontinuity")
    render_frame.add_argument("--color-space", choices=("srgb", "linear-srgb", "pq", "hlg", "unknown"), default="unknown")
    render_frame.add_argument("--depth-encoding", choices=("linear", "device-z", "normalized", "unknown"), default="unknown")
    render_frame.add_argument("--motion-units", choices=("pixel-per-frame", "normalized", "unknown"), default="unknown")
    render_frame.add_argument("--device", default="ARC", help="ARC (default) or exact GPU.X ID")
    render_frame.add_argument("--warmup", type=int, default=2)
    render_frame.add_argument("--iterations", type=int, default=10)
    render_frame.add_argument("--output-dir", required=True, type=Path, help="directory for raw model output tensors")
    render_frame.add_argument("--preview-output", type=Path, help="optional PNG/JPEG preview; requires an explicit RGB output declaration in the adapter spec")
    render_frame.add_argument("--history-output", type=Path, help="optional .npy normalized RGB HWC output to feed as the next frame's history")
    render_frame.add_argument("--report", type=Path, help="optional JSON report of preprocessing and inference timings")

    render_sequence = commands.add_parser(
        "render-sequence",
        help="process recorded frames in order with explicit per-frame inputs and temporal history",
    )
    render_sequence.add_argument("--model", required=True, type=Path, help="local OpenVINO/ONNX model")
    render_sequence.add_argument("--spec", required=True, type=Path, help="explicit JSON model-input adapter spec")
    render_sequence.add_argument("--manifest", required=True, type=Path, help="JSONL frame manifest; paths are relative to this file")
    render_sequence.add_argument("--output-dir", required=True, type=Path, help="directory for sequential RGB PNG outputs")
    render_sequence.add_argument("--device", default="ARC", help="ARC (default) or exact GPU.X ID")
    render_sequence.add_argument("--report", type=Path, help="optional JSON sequence/timing report")

    hnet_frame = commands.add_parser(
        "hnet-frame",
        help="run a supplied 16-channel HNet-style model on one local frame",
    )
    hnet_frame.add_argument("--model", required=True, type=Path, help="user-supplied compatible model; weights are not bundled")
    hnet_frame.add_argument("--color", required=True, type=Path, help="source screenshot/frame")
    hnet_frame.add_argument("--output", required=True, type=Path, help="final RGB image path")
    hnet_frame.add_argument("--history", type=Path, help="optional previous RGB HWC float32 .npy in 0..1")
    hnet_frame.add_argument("--history-output", type=Path, help="optional next RGB HWC float32 .npy in 0..1")
    hnet_frame.add_argument("--head-output", type=Path, help="optional raw four-channel neural head .npy")
    hnet_frame.add_argument("--device", default="ARC", help="ARC (default) or exact GPU.X ID")
    hnet_frame.add_argument("--output-layout", choices=("NCHW", "NHWC"), default="NCHW")
    hnet_frame.add_argument("--frame-index", type=int, default=0, help="deterministic noise frame index")
    hnet_frame.add_argument("--style-index", type=int, default=0, help="documented style index from 0 through 127")
    hnet_frame.add_argument("--tone-strength", type=float, default=1.0)
    hnet_frame.add_argument("--structure-strength", type=float, default=1.0)
    hnet_frame.add_argument("--warmup", type=int, default=1, help="extra warm-up calls after the priming call")
    hnet_frame.add_argument("--iterations", type=int, default=5, help="measured full-frame calls (default: 5)")
    hnet_frame.add_argument("--report", type=Path, help="optional JSON timing and model-contract report")

    xmx_frame = commands.add_parser(
        "xmx-frame",
        help="run the experimental native Intel Arc XMX frame graph with local logical weights",
    )
    xmx_frame.add_argument(
        "--weights", required=True, type=Path,
        help="compatible fully-logical safetensors weights; no weights are bundled",
    )
    xmx_frame.add_argument("--color", required=True, type=Path, help="source RGB frame image")
    xmx_frame.add_argument("--output", required=True, type=Path, help="final RGB image path")
    xmx_frame.add_argument("--history", type=Path, help="optional previous RGB HWC float32 .npy in 0..1")
    xmx_frame.add_argument("--history-output", type=Path, help="optional next RGB HWC float32 .npy in 0..1")
    xmx_frame.add_argument("--head-output", type=Path, help="optional raw four-channel neural head .npy")
    xmx_frame.add_argument(
        "--vulkan-device",
        help="optional Vulkan device-name substring; it must uniquely identify one Intel Arc adapter",
    )
    xmx_frame.add_argument("--frame-index", type=int, default=0, help="deterministic noise frame index")
    xmx_frame.add_argument("--style-index", type=int, default=0, help="documented style index from 0 through 127")
    xmx_frame.add_argument("--tone-strength", type=float, default=1.0)
    xmx_frame.add_argument("--structure-strength", type=float, default=1.0)
    xmx_frame.add_argument("--warmup", type=int, default=1, help="extra warm-up calls after the priming call")
    xmx_frame.add_argument("--iterations", type=int, default=5, help="measured full-frame calls (default: 5)")
    xmx_frame.add_argument("--report", type=Path, help="optional JSON timing and native-device report")

    dx12_probe = commands.add_parser(
        "graphics-probe",
        aliases=["dx12-probe"],
        help="inspect DirectX 9/11/12 adapters or run an API-specific native smoke test",
    )
    dx12_probe.add_argument(
        "--api", choices=("d3d9", "d3d11", "d3d12"), default="d3d12",
        help="graphics API to test (default: d3d12)",
    )
    dx12_modes = dx12_probe.add_mutually_exclusive_group()
    dx12_modes.add_argument(
        "--list-devices", action="store_true",
        help="list the devices reported by the selected DirectX API",
    )
    dx12_modes.add_argument(
        "--self-test", action="store_true",
        help="run the selected API's hardware smoke test (D3D9 render path or D3D11/12 compute)",
    )
    dx12_modes.add_argument(
        "--warp-self-test", action="store_true",
        help="run the API's WARP smoke test where supported (CI only; not GPU performance)",
    )
    dx12_probe.add_argument(
        "--device-substring",
        help="optional unique hardware adapter-name substring for --self-test",
    )
    dx12_probe.add_argument(
        "--executable", type=Path,
        help="optional path to the native executable for the selected API",
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
            "TrSS will not guess which adapter is the Arc GPU. Inspect the device "
            "list and pass its exact GPU.X ID if the driver reports a different name.",
            file=sys.stderr,
        )
        return 2
    return 0


def _run_inspect(args: argparse.Namespace) -> int:
    report = inspect_model(args.model)
    print(json.dumps(report, indent=2))
    if not report["rank_four_single_input_output_candidate"]:
        print(
            "This model is not a one-input/one-output rank-four candidate for the "
            "convenience image command. Inspect its port definitions and use "
            "infer-tensors or a model-specific adapter.",
            file=sys.stderr,
        )
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

    # Keep the first/compilation call separate from the requested warm-up count.
    # This makes --warmup 0 mean zero additional discarded calls, not one.
    for _ in range(args.warmup):
        engine.infer(original)
    warmup_performed = args.warmup

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
        "engine": "TrSS",
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
        "priming_call_excluded_from_samples": True,
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


def _parse_named_tensor(value: str) -> tuple[str, Path]:
    """Parse NAME=PATH while allowing equals signs in the path."""
    if "=" not in value:
        raise ValueError(f"Tensor input {value!r} must be written as NAME=FILE.npy.")
    name, raw_path = value.split("=", maxsplit=1)
    name = name.strip()
    path = Path(raw_path.strip()).expanduser()
    if not name or not raw_path.strip():
        raise ValueError(f"Tensor input {value!r} must include both a name and a file path.")
    return name, path


def _safe_tensor_filename(name: str) -> str:
    """Keep model-provided port names from escaping the output directory."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "tensor"


def _run_tensor_inference(args: argparse.Namespace) -> int:
    """Run a model-specific multi-input graph on the selected local GPU."""
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1.")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model file does not exist: {args.model}")

    parsed_inputs = [_parse_named_tensor(value) for value in args.tensor_inputs]
    names = [name for name, _ in parsed_inputs]
    if len(set(names)) != len(names):
        raise ValueError("Each model input name may be supplied only once.")

    tensor_arrays: dict[str, np.ndarray] = {}
    input_paths: dict[str, Path] = {}
    for name, path in parsed_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Tensor input file does not exist: {path}")
        if path.suffix.casefold() != ".npy":
            raise ValueError(f"Tensor input {path} must be a .npy file.")
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read numeric tensor {path}: {exc}") from exc
        if not isinstance(array, np.ndarray) or array.dtype.hasobject:
            raise ValueError(f"Tensor input {path} is not a numeric NumPy array.")
        tensor_arrays[name] = array
        input_paths[name] = path.resolve()

    resolved_model = args.model.expanduser().resolve()
    resolved_report = args.report.resolve() if args.report else None
    protected = {resolved_model, *input_paths.values()}
    if resolved_report in protected:
        raise ValueError("The JSON report path must not overwrite a model or input tensor.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    engine_load_started = perf_counter()
    engine = TensorInferenceEngine(args.model, device=args.device)
    model_load_ms = (perf_counter() - engine_load_started) * 1000.0

    missing = sorted(set(engine.input_names) - set(tensor_arrays))
    extra = sorted(set(tensor_arrays) - set(engine.input_names))
    if missing or extra:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unknown: " + ", ".join(extra))
        raise ValueError(
            "Tensor files do not match this model's ports (" + "; ".join(details) + "). "
            "Run trss inspect --model MODEL first to see the exact input names."
        )

    prime_started = perf_counter()
    last = engine.infer(tensor_arrays)
    prime_call_ms = (perf_counter() - prime_started) * 1000.0

    priming_compile_ms = last.compile_ms
    for _ in range(args.warmup):
        last = engine.infer(tensor_arrays)

    measured_inference_ms: list[float] = []
    measured_total_ms: list[float] = []
    for _ in range(args.iterations):
        last = engine.infer(tensor_arrays)
        measured_inference_ms.append(last.inference_ms)
        measured_total_ms.append(last.total_ms)

    output_paths: dict[str, str] = {}
    used_names: set[str] = set()
    for index, (name, output) in enumerate(last.outputs.items()):
        stem = _safe_tensor_filename(name)
        if stem in used_names:
            stem = f"{stem}_{index}"
        used_names.add(stem)
        destination = (args.output_dir / f"{stem}.npy").resolve()
        if destination in protected or destination == resolved_report:
            raise ValueError(
                f"Output tensor path {destination} would overwrite an input or report."
            )
        np.save(destination, output, allow_pickle=False)
        output_paths[name] = str(destination)

    inference_median = float(np.median(measured_inference_ms))
    total_median = float(np.median(measured_total_ms))
    report: dict[str, object] = {
        "engine": "TrSS",
        "engine_version": __version__,
        "operation": "multi_input_tensor_inference",
        "model": last.model_path,
        "device_requested": args.device,
        "device_selected": last.device,
        "device_name": last.device_name,
        "model_load_ms": round(model_load_ms, 3),
        "prime_call_ms": round(prime_call_ms, 3),
        "prime_compile_ms": round(priming_compile_ms, 3),
        "warmup_calls": args.warmup,
        "iterations": args.iterations,
        "inputs": {
            name: {
                "path": str(input_paths[name]),
                "shape": list(last.input_shapes[name]),
                "dtype": str(tensor_arrays[name].dtype),
            }
            for name in engine.input_names
        },
        "outputs": {
            name: {
                "path": output_paths[name],
                "shape": list(last.output_shapes[name]),
                "dtype": str(array.dtype),
            }
            for name, array in last.outputs.items()
        },
        "inference_ms": {
            "min": round(min(measured_inference_ms), 3),
            "median": round(inference_median, 3),
            "p95": round(_percentile(measured_inference_ms, 95), 3),
            "max": round(max(measured_inference_ms), 3),
        },
        "engine_call_ms": {
            "min": round(min(measured_total_ms), 3),
            "median": round(total_median, 3),
            "p95": round(_percentile(measured_total_ms, 95), 3),
            "max": round(max(measured_total_ms), 3),
        },
        "note": (
            "This runs the selected model with supplied tensors; TrSS does not guess "
            "preprocessing for color, depth, motion vectors, noise, or temporal history. "
            "Only a verified model-specific adapter can turn these outputs into neural rendering."
        ),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        _write_json(args.report, report)
    return 0


def _load_npy(path: Path, label: str) -> np.ndarray:
    """Load a numeric array without enabling object deserialization."""
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load {label} tensor {path}: {exc}") from exc
    if not isinstance(value, np.ndarray) or value.dtype.hasobject:
        raise ValueError(f"{label} input {path} is not a numeric NumPy array.")
    return value


def _run_render_frame(args: argparse.Namespace) -> int:
    """Run one frame through the explicit preprocessing spec and multi-input engine."""
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1.")
    if args.frame_id < 0:
        raise ValueError("--frame-id must be non-negative.")
    if not np.isfinite(args.timestamp) or args.timestamp < 0:
        raise ValueError("--timestamp must be finite and non-negative.")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model file does not exist: {args.model}")
    if not args.spec.is_file():
        raise FileNotFoundError(f"Adapter spec does not exist: {args.spec}")
    if not args.color.is_file():
        raise FileNotFoundError(f"Color frame does not exist: {args.color}")

    named_auxiliary = {
        "depth": args.depth,
        "motion_vectors": args.motion,
        "noise": args.noise,
        "history": args.history,
        "conditioning": args.conditioning,
    }
    for label, path in named_auxiliary.items():
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label} input does not exist: {path}")

    # Treat every source file and the model/spec as read-only. Output names
    # are not known until model ports are inspected, so validate again below.
    protected = {
        args.model.expanduser().resolve(),
        args.spec.expanduser().resolve(),
        args.color.expanduser().resolve(),
    }
    protected.update(
        path.expanduser().resolve()
        for path in named_auxiliary.values()
        if path is not None
    )
    resolved_report = args.report.expanduser().resolve() if args.report else None
    resolved_preview = args.preview_output.expanduser().resolve() if args.preview_output else None
    resolved_history_output = (
        args.history_output.expanduser().resolve() if args.history_output else None
    )
    if resolved_report in protected:
        raise ValueError("Report path must differ from every model/spec/input file.")
    if resolved_preview in protected or resolved_history_output in protected:
        raise ValueError("Preview/history output paths must differ from every model/spec/input file.")
    if resolved_preview is not None and resolved_history_output == resolved_preview:
        raise ValueError("Preview output and history output must be different files.")
    if resolved_preview is not None and resolved_preview.suffix.casefold() not in (
        ".png", ".jpg", ".jpeg", ".webp", ".bmp"
    ):
        raise ValueError("Preview output must use PNG, JPEG, WebP, or BMP.")
    if resolved_history_output is not None and resolved_history_output.suffix.casefold() != ".npy":
        raise ValueError("History output must use the .npy extension.")

    try:
        with Image.open(args.color) as opened:
            color = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Could not decode color frame {args.color}: {exc}") from exc

    auxiliary: dict[str, np.ndarray | None] = {}
    for label, path in named_auxiliary.items():
        auxiliary[label] = _load_npy(path, label) if path is not None and label != "history" else None
    history = _load_npy(args.history, "history") if args.history is not None else None

    frame = FramePacket(
        color=color,
        timestamp=args.timestamp,
        frame_id=args.frame_id,
        scene_id=args.scene_id,
        depth=auxiliary["depth"],
        motion_vectors=auxiliary["motion_vectors"],
        noise=auxiliary["noise"],
        conditioning=auxiliary["conditioning"],
        camera_cut=args.camera_cut,
        color_space=args.color_space,
        depth_encoding=args.depth_encoding,
        motion_units=args.motion_units,
    )
    spec = ModelAdapterSpec.load(args.spec)
    if (resolved_preview is not None or resolved_history_output is not None) and spec.preview_output is None:
        raise AdapterSpecError(
            "The adapter spec must declare preview_output before TrSS can export a display RGB frame or temporal history."
        )

    engine_started = perf_counter()
    engine = TensorInferenceEngine(args.model, device=args.device)
    engine_load_ms = (perf_counter() - engine_started) * 1000.0

    # A mismatched spec is rejected before running any model inference.
    model_shapes = engine.input_shapes
    inputs = adapt_frame(frame, spec, model_shapes, history=history)
    priming_started = perf_counter()
    last = engine.infer(inputs)
    priming_wall_ms = (perf_counter() - priming_started) * 1000.0
    priming_compile_ms = last.compile_ms

    for _ in range(args.warmup):
        inputs = adapt_frame(frame, spec, model_shapes, history=history)
        last = engine.infer(inputs)

    preprocessing_ms: list[float] = []
    inference_ms: list[float] = []
    end_to_end_ms: list[float] = []
    for _ in range(args.iterations):
        call_started = perf_counter()
        preprocessing_started = perf_counter()
        inputs = adapt_frame(frame, spec, model_shapes, history=history)
        preprocessing_ms.append((perf_counter() - preprocessing_started) * 1000.0)
        last = engine.infer(inputs)
        inference_ms.append(last.inference_ms)
        end_to_end_ms.append((perf_counter() - call_started) * 1000.0)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    seen_stems: set[str] = set()
    destinations: list[tuple[str, Path, np.ndarray]] = []
    for index, (name, output) in enumerate(last.outputs.items()):
        stem = _safe_tensor_filename(name)
        if stem in seen_stems:
            stem = f"{stem}_{index}"
        seen_stems.add(stem)
        destination = (output_dir / f"{stem}.npy").resolve()
        if destination in protected or destination == resolved_report:
            raise ValueError(f"Output tensor path {destination} would overwrite a source or report.")
        destinations.append((name, destination, output))

    preview = None
    if resolved_preview is not None or resolved_history_output is not None:
        preview = preview_from_outputs(last.outputs, spec, target_size=frame.size)

    output_destinations = {destination for _, destination, _ in destinations}
    sidecar_paths = {
        path for path in (resolved_preview, resolved_history_output) if path is not None
    }
    if len(sidecar_paths) != sum(
        path is not None for path in (resolved_preview, resolved_history_output)
    ):
        raise ValueError("Preview and history output paths must be unique.")
    if sidecar_paths & output_destinations:
        raise ValueError("Preview/history path would collide with a model output tensor path.")
    if sidecar_paths & protected:
        raise ValueError("Preview/history path would overwrite a source file.")
    if resolved_report in output_destinations or resolved_report in sidecar_paths:
        raise ValueError("Report path must differ from every output file.")

    if resolved_preview is not None and preview is not None:
        from PIL import Image as PILImage
        pixels = np.rint(np.clip(preview, 0.0, 1.0) * 255.0).astype(np.uint8)
        PILImage.fromarray(pixels).save(resolved_preview)
    if resolved_history_output is not None and preview is not None:
        np.save(resolved_history_output, preview, allow_pickle=False)

    for name, destination, output in destinations:
        np.save(destination, output, allow_pickle=False)
        output_paths[name] = str(destination)

    report: dict[str, object] = {
        "engine": "TrSS",
        "engine_version": __version__,
        "operation": "adapter_driven_single_frame_model_inference",
        "model": last.model_path,
        "adapter_spec": str(args.spec.resolve()),
        "device_requested": args.device,
        "device_selected": last.device,
        "device_name": last.device_name,
        "frame": {
            "color_path": str(args.color.resolve()),
            "width": frame.width,
            "height": frame.height,
            "frame_id": frame.frame_id,
            "timestamp": frame.timestamp,
            "scene_id": frame.scene_id,
            "camera_cut": frame.camera_cut,
            "color_space": frame.color_space,
            "depth_encoding": frame.depth_encoding,
            "motion_units": frame.motion_units,
            "available_sources": sorted(frame.model_inputs().keys()),
            "has_previous_history": history is not None,
        },
        "engine_load_ms": round(engine_load_ms, 3),
        "priming_call_wall_ms": round(priming_wall_ms, 3),
        "priming_compile_ms": round(priming_compile_ms, 3),
        "warmup_calls": args.warmup,
        "iterations": args.iterations,
        "preprocessing_ms": {
            "median": round(float(np.median(preprocessing_ms)), 3),
            "p95": round(_percentile(preprocessing_ms, 95), 3),
        },
        "inference_ms": {
            "median": round(float(np.median(inference_ms)), 3),
            "p95": round(_percentile(inference_ms, 95), 3),
        },
        "end_to_end_ms": {
            "median": round(float(np.median(end_to_end_ms)), 3),
            "p95": round(_percentile(end_to_end_ms, 95), 3),
        },
        "input_tensors": {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "source": spec.inputs[name].source,
                "layout": spec.inputs[name].layout,
            }
            for name, array in inputs.items()
        },
        "output_tensors": {
            name: {
                "path": output_paths[name],
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            for name, array in last.outputs.items()
        },
        "preview_output": (
            {
                "path": str(resolved_preview),
                "port": spec.preview_output.port if spec.preview_output else None,
                "size": [frame.width, frame.height],
                "dtype": "uint8 RGB PNG/JPEG/WebP/BMP on disk",
            }
            if resolved_preview is not None and spec.preview_output is not None
            else None
        ),
        "history_output": (
            {
                "path": str(resolved_history_output),
                "shape": [frame.height, frame.width, 3],
                "dtype": "float32",
                "range": [0, 1],
                "use_as_next_frame_history": True,
            }
            if resolved_history_output is not None
            else None
        ),
        "note": (
            "This runs the selected model using a declared preprocessing contract and supplied "
            "frame buffers. It is not live-game capture, does not synthesize missing depth/motion, "
            "does not automatically compose arbitrary model outputs into RGB, and does not imply "
            "DLSS 5 compatibility or an in-game FPS gain."
        ),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        _write_json(args.report, report)
    return 0


def _safe_hnet_paths(
    model_path: Path,
    color_path: Path,
    output_path: Path,
    history_input: Path | None,
    history_output: Path | None,
    head_output: Path | None,
    report_path: Path | None,
) -> None:
    """Refuse path collisions that could destroy the model, input, or another output."""
    model = model_path.expanduser().resolve()
    protected = {model, color_path.expanduser().resolve()}
    if model.suffix.casefold() == ".xml":
        weights = model.with_suffix(".bin")
        if weights.exists():
            protected.add(weights.resolve())
    if history_input is not None:
        protected.add(history_input.expanduser().resolve())

    output_image = output_path.expanduser().resolve()
    optional_outputs = [
        path.expanduser().resolve()
        for path in (history_output, head_output, report_path)
        if path is not None
    ]
    all_destinations = [output_image, *optional_outputs]
    if len(set(all_destinations)) != len(all_destinations):
        raise ValueError("The image, history, neural-head, and report outputs must use different paths.")
    conflicts = sorted(str(path) for path in all_destinations if path in protected)
    if conflicts:
        raise ValueError(
            "Output path would overwrite the model or an input file: " + ", ".join(conflicts)
        )

    if output_image.suffix.casefold() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        raise ValueError("Image output must use PNG, JPEG, WebP, or BMP.")
    for path, label in ((history_output, "history"), (head_output, "neural-head")):
        if path is not None and path.suffix.casefold() != ".npy":
            raise ValueError(f"{label} output must use the .npy extension.")


def _load_hwc_history(path: Path) -> np.ndarray:
    """Load a caller-provided prior RGB output without enabling Python object loading."""
    try:
        history = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load history tensor {path}: {exc}") from exc
    if (
        not isinstance(history, np.ndarray)
        or history.dtype.hasobject
        or history.ndim != 3
        or history.shape[2] != 3
        or not np.issubdtype(history.dtype, np.number)
        or np.issubdtype(history.dtype, np.complexfloating)
    ):
        raise ValueError("History must be a numeric HxWx3 RGB .npy tensor.")
    history = np.asarray(history, dtype=np.float32)
    if not np.isfinite(history).all() or np.any(history < 0.0) or np.any(history > 1.0):
        raise ValueError("History must contain finite RGB values in the 0..1 range.")
    return history


def _run_render_sequence(args: argparse.Namespace) -> int:
    """Run a validated offline frame manifest and maintain safe temporal history."""
    report = run_sequence(
        model_path=args.model,
        spec_path=args.spec,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        device=args.device,
        report_path=args.report,
        engine_factory=TensorInferenceEngine,
    )
    print(json.dumps(report, indent=2))
    if args.report:
        _write_json(args.report, report)
    return 0


def _run_hnet_frame(args: argparse.Namespace) -> int:
    """Run the specific 16-input-channel/4-output-channel frame adapter."""
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1.")
    if args.frame_index < 0:
        raise ValueError("--frame-index cannot be negative.")
    if not 0 <= args.style_index <= 127:
        raise ValueError("--style-index must be between 0 and 127.")
    if not np.isfinite(args.tone_strength) or not np.isfinite(args.structure_strength):
        raise ValueError("Tone and structure strengths must be finite.")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model file does not exist: {args.model}")
    if not args.color.is_file():
        raise FileNotFoundError(f"Source image does not exist: {args.color}")
    if args.history is not None and not args.history.is_file():
        raise FileNotFoundError(f"History tensor does not exist: {args.history}")

    _safe_hnet_paths(
        args.model,
        args.color,
        args.output,
        args.history,
        args.history_output,
        args.head_output,
        args.report,
    )

    try:
        with Image.open(args.color) as opened:
            # The model contract is float RGB in 0..1, not display bytes.
            color = np.asarray(opened.convert("RGB"), dtype=np.float32) / np.float32(255.0)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not decode source frame {args.color}: {exc}") from exc

    previous = _load_hwc_history(args.history) if args.history is not None else None
    if previous is not None and previous.shape != color.shape:
        raise ValueError(
            f"History shape {previous.shape!r} must match the current frame {color.shape!r}."
        )

    controls = HNetControls(
        style_index=args.style_index,
        tone_strength=args.tone_strength,
        structure_strength=args.structure_strength,
    )

    engine_load_started = perf_counter()
    tensor_engine = TensorInferenceEngine(args.model, device=args.device)
    engine = HNetFrameEngine(tensor_engine, output_layout=args.output_layout)
    engine_load_ms = (perf_counter() - engine_load_started) * 1000.0

    first_started = perf_counter()
    last_result = engine.process_frame(
        color, history=previous, frame_index=args.frame_index, controls=controls
    )
    priming_wall_ms = (perf_counter() - first_started) * 1000.0
    priming_compile_ms = last_result.compile_ms

    for _ in range(args.warmup):
        last_result = engine.process_frame(
            color, history=previous, frame_index=args.frame_index, controls=controls
        )

    neural_ms: list[float] = []
    total_ms: list[float] = []
    for _ in range(args.iterations):
        started = perf_counter()
        last_result = engine.process_frame(
            color, history=previous, frame_index=args.frame_index, controls=controls
        )
        total_ms.append((perf_counter() - started) * 1000.0)
        neural_ms.append(last_result.inference_ms)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.rint(np.clip(last_result.image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(np.ascontiguousarray(rgb)).save(output_path)

    history_path = args.history_output.expanduser().resolve() if args.history_output else None
    head_path = args.head_output.expanduser().resolve() if args.head_output else None
    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(history_path, last_result.history.astype(np.float32), allow_pickle=False)
    if head_path is not None:
        head_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(head_path, last_result.head.astype(np.float32), allow_pickle=False)

    report: dict[str, object] = {
        "engine": "TrSS",
        "engine_version": __version__,
        "operation": "hnet_contract_frame_inference",
        "model": str(args.model.resolve()),
        "device_requested": args.device,
        "device_selected": last_result.device,
        "device_name": last_result.device_name,
        "input_frame": str(args.color.resolve()),
        "input_size": list(color.shape[1::-1]),
        "network_size": [
            last_result.geometry.network_width,
            last_result.geometry.network_height,
        ],
        "feature_contract": {
            "input_layout": "NCHW",
            "input_shape": [
                1, 16,
                last_result.geometry.network_height,
                last_result.geometry.network_width,
            ],
            "output_layout": args.output_layout,
            "output_channels": 4,
            "frame_index": args.frame_index,
            "style_index": args.style_index,
            "tone_strength": args.tone_strength,
            "structure_strength": args.structure_strength,
            "history_supplied": previous is not None,
            "motion_reprojection": "not performed; caller supplied no scene motion field",
        },
        "engine_load_ms": round(engine_load_ms, 3),
        "priming_wall_ms": round(priming_wall_ms, 3),
        "priming_compile_ms": round(priming_compile_ms, 3),
        "warmup_calls": args.warmup,
        "iterations": args.iterations,
        "neural_inference_ms": {
            "median": round(float(np.median(neural_ms)), 3),
            "p95": round(_percentile(neural_ms, 95), 3),
        },
        "complete_frame_ms": {
            "median": round(float(np.median(total_ms)), 3),
            "p95": round(_percentile(total_ms, 95), 3),
        },
        "inference_fps_equivalent": (
            round(1000.0 / float(np.median(neural_ms)), 2)
            if float(np.median(neural_ms)) > 0 else None
        ),
        "outputs": {
            "image": str(output_path),
            "history": str(history_path) if history_path is not None else None,
            "raw_head": str(head_path) if head_path is not None else None,
        },
        "note": (
            "HNet-style contract path only. This does not bundle or identify any trained model "
            "weights, does not prove vendor-compatible model inference, does not capture a live "
            "game, and does not establish DLSS 5 compatibility or in-game FPS."
        ),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        _write_json(args.report, report)
    return 0


def _run_xmx_frame(args: argparse.Namespace) -> int:
    """Run the vendored Vulkan/XMX graph against a user-supplied logical model."""
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")
    if args.iterations < 1:
        raise ValueError("--iterations must be at least 1.")
    if args.frame_index < 0:
        raise ValueError("--frame-index cannot be negative.")
    if not 0 <= args.style_index <= 127:
        raise ValueError("--style-index must be between 0 and 127.")
    if not np.isfinite(args.tone_strength) or not np.isfinite(args.structure_strength):
        raise ValueError("Tone and structure strengths must be finite.")
    if not args.weights.is_file():
        raise FileNotFoundError(f"Logical model weights do not exist: {args.weights}")
    if not args.color.is_file():
        raise FileNotFoundError(f"Source image does not exist: {args.color}")
    if args.history is not None and not args.history.is_file():
        raise FileNotFoundError(f"History tensor does not exist: {args.history}")

    _safe_hnet_paths(
        args.weights,
        args.color,
        args.output,
        args.history,
        args.history_output,
        args.head_output,
        args.report,
    )

    try:
        with Image.open(args.color) as opened:
            color = np.asarray(opened.convert("RGB"), dtype=np.float32) / np.float32(255.0)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not decode source frame {args.color}: {exc}") from exc

    previous = _load_hwc_history(args.history) if args.history is not None else None
    if previous is not None and previous.shape != color.shape:
        raise ValueError(
            f"History shape {previous.shape!r} must match the current frame {color.shape!r}."
        )

    controls = HNetControls(
        style_index=args.style_index,
        tone_strength=args.tone_strength,
        structure_strength=args.structure_strength,
    )

    initialization_started = perf_counter()
    engine = XMXFrameEngine(args.weights, device_substring=args.vulkan_device)
    engine_initialization_ms = (perf_counter() - initialization_started) * 1000.0
    try:
        # The first call prepares scratch storage, lazily creates native pipelines,
        # records the graph, and warms the device. Do not mix that cost into steady-state
        # frame latency measurements.
        priming_started = perf_counter()
        last_result = engine.process_frame(
            color,
            history=previous,
            frame_index=args.frame_index,
            controls=controls,
        )
        priming_wall_ms = (perf_counter() - priming_started) * 1000.0

        for _ in range(args.warmup):
            last_result = engine.process_frame(
                color,
                history=previous,
                frame_index=args.frame_index,
                controls=controls,
            )

        measured_results = []
        full_pipeline_ms = []
        for _ in range(args.iterations):
            started = perf_counter()
            last_result = engine.process_frame(
                color,
                history=previous,
                frame_index=args.frame_index,
                controls=controls,
            )
            full_pipeline_ms.append((perf_counter() - started) * 1000.0)
            measured_results.append(last_result)

        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rgb = np.rint(np.clip(last_result.image, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(np.ascontiguousarray(rgb)).save(output_path)

        history_path = args.history_output.expanduser().resolve() if args.history_output else None
        head_path = args.head_output.expanduser().resolve() if args.head_output else None
        if history_path is not None:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(history_path, last_result.history.astype(np.float32), allow_pickle=False)
        if head_path is not None:
            head_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(head_path, last_result.head.astype(np.float32), allow_pickle=False)

        runner_times = [item.frame_runner_ms for item in measured_results]
        preprocess_times = [item.preprocess_ms for item in measured_results]
        composition_times = [item.composition_ms for item in measured_results]
        report: dict[str, object] = {
            "engine": "TrSS",
            "engine_version": __version__,
            "operation": "native_xmx_frame_inference",
            "logical_weights": str(args.weights.expanduser().resolve()),
            "weights_metadata": engine.metadata,
            "device_requested": args.vulkan_device or "unique Vulkan adapter containing 'Arc'",
            "device_selected": last_result.device_name,
            "input_frame": str(args.color.expanduser().resolve()),
            "input_size": list(color.shape[1::-1]),
            "network_size": list(last_result.network_size),
            "feature_contract": {
                "input_layout": "HWC",
                "input_channels": 16,
                "output_channels": 4,
                "frame_index": args.frame_index,
                "style_index": args.style_index,
                "tone_strength": args.tone_strength,
                "structure_strength": args.structure_strength,
                "history_supplied": previous is not None,
                "motion_reprojection": "not performed; no scene motion field was supplied",
            },
            "engine_initialization_ms": round(engine_initialization_ms, 3),
            "priming_wall_ms": round(priming_wall_ms, 3),
            "warmup_calls_after_priming": args.warmup,
            "iterations": args.iterations,
            "native_frame_execution_ms": {
                "median": round(float(np.median(runner_times)), 3),
                "p95": round(_percentile(runner_times, 95), 3),
            },
            "preprocessing_ms": {
                "median": round(float(np.median(preprocess_times)), 3),
                "p95": round(_percentile(preprocess_times, 95), 3),
            },
            "composition_ms": {
                "median": round(float(np.median(composition_times)), 3),
                "p95": round(_percentile(composition_times, 95), 3),
            },
            "complete_local_frame_ms": {
                "median": round(float(np.median(full_pipeline_ms)), 3),
                "p95": round(_percentile(full_pipeline_ms, 95), 3),
            },
            "native_frame_fps_equivalent": (
                round(1000.0 / float(np.median(runner_times)), 2)
                if float(np.median(runner_times)) > 0 else None
            ),
            "allocated_native_buffer_bytes": last_result.allocated_buffer_bytes,
            "outputs": {
                "image": str(output_path),
                "history": str(history_path) if history_path is not None else None,
                "raw_head": str(head_path) if head_path is not None else None,
            },
            "note": (
                "Experimental offline native XMX frame path. It requires a matching "
                "fully-logical model file and a compatible Intel Arc driver. It does not "
                "capture a live game, perform motion-vector reprojection, or establish "
                "DLSS 5 compatibility or in-game performance."
            ),
        }
        print(json.dumps(report, indent=2))
        if args.report:
            _write_json(args.report, report)
    finally:
        engine.close()
    return 0


def _run_dx12_probe(args: argparse.Namespace) -> int:
    """Launch the selected first-party DirectX API probe."""
    if args.device_substring is not None and not args.device_substring.strip():
        raise ValueError("--device-substring cannot be blank.")

    if args.api == "d3d9" and args.warp_self_test:
        raise ValueError("Direct3D 9 has no WARP compute test; use an actual D3D9-capable adapter.")

    if args.list_devices:
        operation = "list-devices"
    elif args.warp_self_test:
        operation = "warp-self-test"
    else:
        # With no explicit mode, use the real hardware self-test and never silently
        # substitute WARP or CPU execution.
        operation = "self-test"

    return run_dx12_probe(
        operation=operation,
        executable=args.executable,
        device_substring=args.device_substring,
        api=args.api,
    )


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
        if args.command == "infer-tensors":
            return _run_tensor_inference(args)
        if args.command == "render-frame":
            return _run_render_frame(args)
        if args.command == "render-sequence":
            return _run_render_sequence(args)
        if args.command == "hnet-frame":
            return _run_hnet_frame(args)
        if args.command == "xmx-frame":
            return _run_xmx_frame(args)
        if args.command in ("graphics-probe", "dx12-probe"):
            return _run_dx12_probe(args)
        parser.error("unknown command")
    except (BackendUnavailable, AdapterSpecError, HNetContractError, FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"TrSS error: {exc}", file=sys.stderr)
        return 2
    return 0
