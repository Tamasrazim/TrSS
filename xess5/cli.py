"""Command-line interface for the XESS5 Intel Arc neural inference engine."""

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
from .backends.openvino_backend import BackendUnavailable
from .contracts import parse_value_range
from .engine import ImageInferenceEngine, inspect_model, available_devices
from .tensor_engine import TensorInferenceEngine
from .rendering import FramePacket
from .rendering.adapter import AdapterSpecError, ModelAdapterSpec, adapt_frame


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
    render_frame.add_argument("--report", type=Path, help="optional JSON report of preprocessing and inference timings")
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
            "Run xess5 inspect --model MODEL first to see the exact input names."
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
        "engine": "XESS5",
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
            "This runs the selected model with supplied tensors; XESS5 does not guess "
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
    if resolved_report in protected:
        raise ValueError("Report path must differ from every model/spec/input file.")

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

    if resolved_report in {destination for _, destination, _ in destinations}:
        raise ValueError("The JSON report path must differ from every output tensor path.")

    for name, destination, output in destinations:
        np.save(destination, output, allow_pickle=False)
        output_paths[name] = str(destination)

    report: dict[str, object] = {
        "engine": "XESS5",
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
        parser.error("unknown command")
    except (BackendUnavailable, AdapterSpecError, FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"XESS5 error: {exc}", file=sys.stderr)
        return 2
    return 0
