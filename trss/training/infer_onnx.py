"""Run the exported TrSS ONNX graph through OpenVINO on one manifest frame.

This offline adapter prepares the exact ten-channel feature layout used by
training, compiles the exported graph on an explicitly selected OpenVINO device,
and saves the composed RGB output. It does not capture live game buffers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..backends.openvino_backend import describe_devices, load_openvino, resolve_device
from .manifest import load_manifest


_EXPECTED_INPUTS = {"features", "low_res_rgb", "history_rgb", "history_valid"}


def _sha256(path: Path) -> str:
    """Return the checksum of the model that was actually loaded."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rgb(path: Path) -> np.ndarray:
    """Load display RGB as contiguous float32 CHW data in the 0–1 range."""
    try:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Could not read RGB image {path}: {exc}") from exc
    return np.ascontiguousarray(pixels.transpose(2, 0, 1), dtype=np.float32)


def _read_npy(path: Path, name: str) -> np.ndarray:
    """Read a numeric array without allowing pickled Python objects."""
    if path.suffix.lower() != ".npy":
        raise ValueError(f"{name} must be stored as an explicit .npy array.")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load {name} array {path}: {exc}") from exc
    if array.dtype.kind not in "uif":
        raise ValueError(f"{name} must contain real numeric values.")
    result = np.asarray(array, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return result


def _read_mask(path: Path, height: int, width: int) -> np.ndarray:
    """Read a mask and enforce finite normalized values at 2x output resolution."""
    if path.suffix.lower() == ".npy":
        mask = _read_npy(path, "history_valid")
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
    else:
        try:
            with Image.open(path) as image:
                mask = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not read history-validity mask {path}: {exc}") from exc
    if mask.shape != (height, width):
        raise ValueError(
            f"history_valid shape {mask.shape} must match the high-resolution target {(height, width)}."
        )
    mask = np.asarray(mask, dtype=np.float32)
    if not np.isfinite(mask).all() or mask.min(initial=0.0) < 0.0 or mask.max(initial=0.0) > 1.0:
        raise ValueError("history_valid must contain only finite values in [0, 1].")
    return np.ascontiguousarray(mask)


def _prepare_inputs(row: dict[str, Any], *, motion_scale_px: float) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    """Create the same feature ordering and exact 2x area reductions as training."""
    current = _read_rgb(Path(row["low_res"]))
    height, width = current.shape[-2:]
    target = _read_rgb(Path(row["target"]))
    if target.shape[-2:] != (height * 2, width * 2):
        raise ValueError(
            f"target must be 2x low_res; got input {(width, height)} and "
            f"target {(target.shape[-1], target.shape[-2])}."
        )

    depth = _read_npy(Path(row["depth"]), "depth")
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.shape != (height, width):
        raise ValueError(f"depth shape {depth.shape} must match {(height, width)}.")
    if depth.min(initial=0.0) < 0.0 or depth.max(initial=0.0) > 1.0:
        raise ValueError("depth must be normalized to [0, 1].")
    depth = depth[None, ...]

    motion = _read_npy(Path(row["motion"]), "motion")
    if motion.shape != (height, width, 2):
        raise ValueError(f"motion shape {motion.shape} must be {(height, width, 2)}.")
    motion = np.clip(motion / motion_scale_px, -1.0, 1.0).transpose(2, 0, 1)

    output_height, output_width = height * 2, width * 2
    if row["history"] is None:
        if row["history_valid"] is not None:
            raise ValueError("history and history_valid must both be null for a reset frame.")
        history = np.zeros((3, output_height, output_width), dtype=np.float32)
        history_valid = np.zeros((1, output_height, output_width), dtype=np.float32)
    else:
        history = _read_rgb(Path(row["history"]))
        if history.shape != (3, output_height, output_width):
            raise ValueError(
                f"history shape {history.shape} must be {(3, output_height, output_width)}."
            )
        mask = _read_mask(Path(row["history_valid"]), output_height, output_width)
        history_valid = mask[None, ...]

    # A 2x box average matches torch's area interpolation for these integer ratios.
    history_low = history.reshape(3, height, 2, width, 2).mean(axis=(2, 4))
    valid_low = history_valid.reshape(1, height, 2, width, 2).mean(axis=(2, 4))
    features = np.concatenate(
        (current, depth, motion, history_low, valid_low),
        axis=0,
    ).astype(np.float32, copy=False)
    if features.shape != (10, height, width):
        raise RuntimeError(f"Internal feature assembly error: got {features.shape}.")

    inputs = {
        "features": np.ascontiguousarray(features[None]),
        "low_res_rgb": np.ascontiguousarray(current[None]),
        "history_rgb": np.ascontiguousarray(history[None]),
        "history_valid": np.ascontiguousarray(history_valid[None]),
    }
    return inputs, (width, height)


def run_onnx_frame(
    *,
    model_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    row_index: int = 0,
    device_name: str = "ARC",
    motion_scale_px: float = 64.0,
    warmup: int = 1,
    iterations: int = 5,
    report_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run one paired-manifest input through the exported model using OpenVINO."""
    started = perf_counter()
    model_file = Path(model_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    output_file = Path(output_path).expanduser().resolve()
    report_file = (
        Path(report_path).expanduser().resolve()
        if report_path is not None
        else Path(str(output_file) + ".json")
    )

    if model_file.suffix.lower() != ".onnx" or not model_file.is_file():
        raise FileNotFoundError(f"An existing .onnx model is required: {model_file}")
    if output_file.suffix.lower() != ".png":
        raise ValueError("output_path must end in .png.")
    if output_file == report_file:
        raise ValueError("The PNG output and JSON report must be different files.")
    if not math.isfinite(motion_scale_px) or motion_scale_px <= 0:
        raise ValueError("motion_scale_px must be finite and greater than zero.")
    if not isinstance(row_index, int) or isinstance(row_index, bool) or row_index < 0:
        raise ValueError("row_index must be a zero-based non-negative integer.")
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be at least one.")
    for path in (output_file, report_file):
        if path.exists() and not path.is_file():
            raise IsADirectoryError(f"Output path must be a file: {path}")
        if path.exists() and not force:
            raise FileExistsError(f"Output already exists: {path}. Pass --force to replace it.")
    if model_file in (output_file, report_file) or manifest_file in (output_file, report_file):
        raise ValueError("Output and report paths must not overwrite the model or manifest.")

    rows = load_manifest(manifest_file)
    if row_index >= len(rows):
        raise ValueError(
            f"row_index {row_index} is outside this manifest, which contains {len(rows)} rows."
        )
    row = rows[row_index]
    source_files = {
        Path(value).resolve()
        for key, value in row.items()
        if key != "line" and value is not None
    }
    if output_file in source_files or report_file in source_files:
        raise ValueError("Output and report paths must not overwrite any frame input or target.")

    preprocess_started = perf_counter()
    inputs, input_size = _prepare_inputs(row, motion_scale_px=motion_scale_px)
    preprocess_ms = (perf_counter() - preprocess_started) * 1000.0

    try:
        ov = load_openvino()
        core = ov.Core()
        # CPU is an explicit offline-debug choice. All GPU choices use the project's
        # strict device resolver; the ARC alias never silently falls back to CPU.
        resolved_device = (
            "CPU" if device_name.strip().upper() == "CPU"
            else resolve_device(core, device_name)
        )
        device_details = next(
            (item for item in describe_devices(core) if item["device"] == resolved_device),
            {"device": resolved_device, "name": resolved_device, "type": "unknown"},
        )
        compile_started = perf_counter()
        model = core.read_model(str(model_file))
        model_inputs = {port.get_any_name() for port in model.inputs}
        if model_inputs != _EXPECTED_INPUTS:
            raise ValueError(
                f"ONNX input names must be {_EXPECTED_INPUTS}; found {sorted(model_inputs)}."
            )
        if len(model.outputs) != 1:
            raise ValueError(f"ONNX model must have one composed output; found {len(model.outputs)}.")
        compiled = core.compile_model(model, resolved_device)
        compile_ms = (perf_counter() - compile_started) * 1000.0
    except Exception as exc:
        # Keep the model/device failure actionable without hiding its cause.
        if isinstance(exc, (FileNotFoundError, ValueError)):
            raise
        raise RuntimeError(f"OpenVINO could not prepare the ONNX model on {device_name!r}: {exc}") from exc

    feed = {name: np.asarray(value, dtype=np.float32) for name, value in inputs.items()}
    for _ in range(warmup):
        compiled(feed)

    samples: list[float] = []
    result: np.ndarray | None = None
    for _ in range(iterations):
        run_started = perf_counter()
        outputs = compiled(feed)
        samples.append((perf_counter() - run_started) * 1000.0)
        result = np.asarray(outputs[compiled.output(0)], dtype=np.float32)

    if result is None or result.shape != (1, 3, input_size[1] * 2, input_size[0] * 2):
        actual = None if result is None else result.shape
        raise RuntimeError(
            f"The ONNX graph returned shape {actual}; expected "
            f"(1, 3, {input_size[1] * 2}, {input_size[0] * 2})."
        )
    if not np.isfinite(result).all():
        raise RuntimeError("ONNX inference returned NaN or infinite values.")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    temp_image = None
    temp_report = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_file.name}.", suffix=".png",
            dir=output_file.parent, delete=False,
        ) as stream:
            temp_image = Path(stream.name)
        pixels = np.rint(
            np.clip(result[0].transpose(1, 2, 0), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        Image.fromarray(pixels, mode="RGB").save(temp_image, format="PNG")

        report: dict[str, Any] = {
            "format": "trss-onnx-frame-run-v1",
            "model": str(model_file),
            "model_sha256": _sha256(model_file),
            "manifest": str(manifest_file),
            "manifest_row_index": row_index,
            "manifest_line": row["line"],
            "output": str(output_file),
            "report": str(report_file),
            "device": resolved_device,
            "device_name": device_details["name"],
            "input_size": list(input_size),
            "output_size": [input_size[0] * 2, input_size[1] * 2],
            "preprocess_ms": preprocess_ms,
            "compile_ms": compile_ms,
            "warmup_calls": warmup,
            "measured_iterations": iterations,
            "mean_inference_ms": float(np.mean(samples)),
            "min_inference_ms": float(np.min(samples)),
            "max_inference_ms": float(np.max(samples)),
            "output_encode_write_ms": None,
            "elapsed_seconds_before_report_write": perf_counter() - started,
            "live_capture_used": False,
            "note": (
                "Offline manifest-frame inference only. This command does not acquire live game "
                "buffers, does not validate perceptual quality, and does not report in-game FPS. "
                "CPU is supported only when explicitly selected for debugging."
            ),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            prefix=f".{report_file.name}.", suffix=".json",
            dir=report_file.parent, delete=False,
        ) as stream:
            temp_report = Path(stream.name)
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")

        # Each file is first fully written to a same-directory temporary path.
        # A failed encode never damages an existing user file.
        os.replace(temp_image, output_file)
        temp_image = None
        os.replace(temp_report, report_file)
        temp_report = None
        return report
    finally:
        for temporary in (temp_image, temp_report):
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one paired-manifest frame through an exported TrSS ONNX graph with OpenVINO. "
            "This is offline inference, not live game capture."
        )
    )
    parser.add_argument("--model", required=True, type=Path, help="TrSS composed .onnx graph")
    parser.add_argument("--manifest", required=True, type=Path, help="paired-frame JSONL manifest")
    parser.add_argument("--row", type=int, default=0, help="zero-based index among manifest records")
    parser.add_argument("--output", required=True, type=Path, help="composed RGB PNG output")
    parser.add_argument("--device", default="ARC", help="ARC, exact GPU.X, or explicit CPU for debugging")
    parser.add_argument("--motion-scale-px", type=float, default=64.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--report", type=Path, help="JSON report; defaults to OUTPUT.png.json")
    parser.add_argument("--force", action="store_true", help="replace existing output/report files")
    args = parser.parse_args()
    try:
        report = run_onnx_frame(
            model_path=args.model,
            manifest_path=args.manifest,
            output_path=args.output,
            row_index=args.row,
            device_name=args.device,
            motion_scale_px=args.motion_scale_px,
            warmup=args.warmup,
            iterations=args.iterations,
            report_path=args.report,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, IsADirectoryError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "output": report["output"],
        "report": report["report"],
        "device": report["device"],
        "device_name": report["device_name"],
        "input_size": report["input_size"],
        "output_size": report["output_size"],
        "compile_ms": report["compile_ms"],
        "mean_inference_ms": report["mean_inference_ms"],
        "live_capture_used": report["live_capture_used"],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
