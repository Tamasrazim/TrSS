"""Manifest-driven offline sequence inference with explicit temporal history.

This path processes one recorded frame at a time. It is not a live-game
capture hook, and it never invents missing depth, motion, or history buffers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..tensor_engine import TensorInferenceEngine
from .adapter import (
    AdapterSpecError,
    ModelAdapterSpec,
    adapt_frame,
    preview_from_outputs,
)
from .frame import FramePacket, TemporalHistory


@dataclass(frozen=True, slots=True)
class SequenceEntry:
    """One validated JSONL row, with paths resolved against the manifest."""

    line_number: int
    frame_id: int
    timestamp: float
    color_path: Path
    depth_path: Path | None
    motion_path: Path | None
    noise_path: Path | None
    conditioning_path: Path | None
    initial_history_path: Path | None
    scene_id: str | None
    camera_cut: bool
    color_space: str
    depth_encoding: str
    motion_units: str

    @property
    def source_paths(self) -> tuple[Path, ...]:
        return tuple(
            path
            for path in (
                self.color_path,
                self.depth_path,
                self.motion_path,
                self.noise_path,
                self.conditioning_path,
                self.initial_history_path,
            )
            if path is not None
        )


_ALLOWED_FIELDS = {
    "frame_id",
    "timestamp",
    "color",
    "depth",
    "motion",
    "noise",
    "conditioning",
    "initial_history",
    "scene_id",
    "camera_cut",
    "color_space",
    "depth_encoding",
    "motion_units",
}


def _resolve_manifest_path(
    raw: Any,
    *,
    key: str,
    manifest_dir: Path,
    line_number: int,
    required: bool = False,
) -> Path | None:
    """Resolve one path but do not silently treat malformed values as absent."""
    if raw is None:
        if required:
            raise ValueError(
                f"Sequence manifest line {line_number}: {key!r} is required."
            )
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            f"Sequence manifest line {line_number}: {key!r} must be a non-empty path string."
        )

    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _parse_entry(
    value: Any, *, manifest_dir: Path, line_number: int
) -> SequenceEntry:
    """Validate a manifest row before any model execution starts."""
    if not isinstance(value, dict):
        raise ValueError(
            f"Sequence manifest line {line_number}: each row must be a JSON object."
        )
    unknown = sorted(set(value) - _ALLOWED_FIELDS)
    if unknown:
        raise ValueError(
            f"Sequence manifest line {line_number}: unknown fields: {', '.join(unknown)}."
        )

    frame_id = value.get("frame_id")
    if (
        isinstance(frame_id, bool)
        or not isinstance(frame_id, int)
        or frame_id < 0
    ):
        raise ValueError(
            f"Sequence manifest line {line_number}: frame_id must be a non-negative integer."
        )

    timestamp = value.get("timestamp")
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not np.isfinite(timestamp)
        or timestamp < 0
    ):
        raise ValueError(
            f"Sequence manifest line {line_number}: timestamp must be finite and non-negative."
        )

    color_path = _resolve_manifest_path(
        value.get("color"),
        key="color",
        manifest_dir=manifest_dir,
        line_number=line_number,
        required=True,
    )
    assert color_path is not None

    optional_paths = {
        key: _resolve_manifest_path(
            value.get(key),
            key=key,
            manifest_dir=manifest_dir,
            line_number=line_number,
        )
        for key in ("depth", "motion", "noise", "conditioning", "initial_history")
    }

    scene_id = value.get("scene_id")
    if scene_id is not None and not isinstance(scene_id, str):
        raise ValueError(
            f"Sequence manifest line {line_number}: scene_id must be a string or null."
        )
    camera_cut = value.get("camera_cut", False)
    if not isinstance(camera_cut, bool):
        raise ValueError(
            f"Sequence manifest line {line_number}: camera_cut must be true or false."
        )

    defaults = {
        "color_space": "unknown",
        "depth_encoding": "unknown",
        "motion_units": "unknown",
    }
    conventions: dict[str, str] = {}
    for key, default in defaults.items():
        raw = value.get(key, default)
        if not isinstance(raw, str):
            raise ValueError(
                f"Sequence manifest line {line_number}: {key} must be a string."
            )
        conventions[key] = raw

    entry = SequenceEntry(
        line_number=line_number,
        frame_id=frame_id,
        timestamp=float(timestamp),
        color_path=color_path,
        depth_path=optional_paths["depth"],
        motion_path=optional_paths["motion"],
        noise_path=optional_paths["noise"],
        conditioning_path=optional_paths["conditioning"],
        initial_history_path=optional_paths["initial_history"],
        scene_id=scene_id,
        camera_cut=camera_cut,
        color_space=conventions["color_space"],
        depth_encoding=conventions["depth_encoding"],
        motion_units=conventions["motion_units"],
    )
    for path in entry.source_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Sequence manifest line {line_number}: input file does not exist: {path}"
            )
    return entry


def _load_manifest(path: Path) -> list[SequenceEntry]:
    """Read JSONL rows and resolve relative inputs beside the manifest file."""
    try:
        lines = path.expanduser().resolve().read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FileNotFoundError(f"Could not read sequence manifest {path}: {exc}") from exc

    entries: list[SequenceEntry] = []
    manifest_path = path.expanduser().resolve()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Sequence manifest line {line_number} is not valid JSON: {exc.msg}."
            ) from exc
        entries.append(_parse_entry(row, manifest_dir=manifest_path.parent, line_number=line_number))

    if not entries:
        raise ValueError(f"Sequence manifest {manifest_path} contains no frame records.")
    return entries


def _load_array(path: Path, label: str) -> np.ndarray:
    """Load a numeric .npy tensor without enabling object deserialization."""
    if path.suffix.casefold() != ".npy":
        raise ValueError(f"{label} input must use the .npy extension: {path}")
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load {label} tensor {path}: {exc}") from exc
    if (
        not isinstance(value, np.ndarray)
        or value.dtype.hasobject
        or not np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.complexfloating)
    ):
        raise ValueError(f"{label} input must be a real-valued numeric NumPy array: {path}")
    if not np.isfinite(value).all():
        raise ValueError(f"{label} input contains NaN or infinite values: {path}")
    return value


def _load_seed_history(path: Path, frame: FramePacket) -> np.ndarray:
    """Load an explicit RGB history seed after a start/reset, validating its size/range."""
    if path.suffix.casefold() != ".npy":
        raise ValueError(f"initial_history must use the .npy extension: {path}")
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load initial history tensor {path}: {exc}") from exc
    if (
        not isinstance(value, np.ndarray)
        or value.dtype.hasobject
        or value.ndim != 3
        or value.shape != (frame.height, frame.width, 3)
        or not np.issubdtype(value.dtype, np.number)
        or np.issubdtype(value.dtype, np.complexfloating)
    ):
        raise ValueError(
            "initial_history must be a numeric HxWx3 RGB .npy matching the current frame."
        )
    history = np.asarray(value, dtype=np.float32)
    if not np.isfinite(history).all() or np.any(history < 0.0) or np.any(history > 1.0):
        raise ValueError("initial_history must contain finite normalized RGB values in [0, 1].")
    return history


def _load_frame(entry: SequenceEntry) -> FramePacket:
    """Decode a row's actual source files into the shared frame contract."""
    try:
        with Image.open(entry.color_path) as opened:
            color = np.asarray(opened.convert("RGB"), dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(
            f"Sequence manifest line {entry.line_number}: could not decode "
            f"color frame {entry.color_path}: {exc}"
        ) from exc

    depth = _load_array(entry.depth_path, "depth") if entry.depth_path else None
    motion = _load_array(entry.motion_path, "motion") if entry.motion_path else None
    noise = _load_array(entry.noise_path, "noise") if entry.noise_path else None
    conditioning = (
        _load_array(entry.conditioning_path, "conditioning")
        if entry.conditioning_path
        else None
    )
    return FramePacket(
        color=color,
        timestamp=entry.timestamp,
        frame_id=entry.frame_id,
        scene_id=entry.scene_id,
        depth=depth,
        motion_vectors=motion,
        noise=noise,
        conditioning=conditioning,
        camera_cut=entry.camera_cut,
        color_space=entry.color_space,  # FramePacket validates supported values.
        depth_encoding=entry.depth_encoding,
        motion_units=entry.motion_units,
    )


def run_sequence(
    *,
    model_path: Path,
    spec_path: Path,
    manifest_path: Path,
    output_dir: Path,
    device: str = "ARC",
    report_path: Path | None = None,
    engine_factory: Callable[..., Any] | None = None,
) -> dict[str, object]:
    """Run a JSONL frame sequence and carry only valid, explicit RGB history."""
    for path, label in (
        (model_path, "Model"),
        (spec_path, "Adapter spec"),
        (manifest_path, "Sequence manifest"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file does not exist: {path}")

    entries = _load_manifest(manifest_path)
    spec = ModelAdapterSpec.load(spec_path)
    if spec.preview_output is None:
        raise AdapterSpecError(
            "render-sequence requires preview_output so each output can be validated "
            "as RGB and safely reused as normalized temporal history."
        )

    resolved_model = model_path.expanduser().resolve()
    protected = {
        resolved_model,
        spec_path.expanduser().resolve(),
        manifest_path.expanduser().resolve(),
    }
    if resolved_model.suffix.casefold() == ".xml":
        paired_weights = resolved_model.with_suffix(".bin")
        if paired_weights.is_file():
            protected.add(paired_weights.resolve())
    for entry in entries:
        protected.update(entry.source_paths)

    destination_dir = output_dir.expanduser().resolve()
    destinations = [
        (destination_dir / f"frame_{index:06d}.png").resolve()
        for index in range(len(entries))
    ]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Generated sequence output paths are not unique.")
    conflicts = sorted(str(path) for path in destinations if path in protected)
    if conflicts:
        raise ValueError(
            "Sequence output would overwrite an input file: " + ", ".join(conflicts)
        )

    resolved_report = report_path.expanduser().resolve() if report_path else None
    if resolved_report is not None:
        if resolved_report in protected:
            raise ValueError("Report path must differ from every model, spec, manifest, and input file.")
        if resolved_report in set(destinations):
            raise ValueError("Report path must differ from every generated sequence frame.")
        if resolved_report.suffix.casefold() != ".json":
            raise ValueError("Sequence report must use the .json extension.")

    factory = engine_factory or TensorInferenceEngine
    engine = factory(resolved_model, device=device)
    needs_history = any(binding.source == "history" for binding in spec.inputs.values())
    temporal = TemporalHistory()
    frame_reports: list[dict[str, object]] = []
    inference_times: list[float] = []
    complete_times: list[float] = []
    total_compile_ms = 0.0
    try:
        for index, entry in enumerate(entries):
            frame_started = perf_counter()
            frame = _load_frame(entry)

            previous = temporal.history_for(frame)
            reset_reason = temporal.last_reset_reason
            history_source = "previous_output" if previous is not None else None
            if previous is None and needs_history and entry.initial_history_path is not None:
                previous = _load_seed_history(entry.initial_history_path, frame)
                history_source = "manifest_initial_history"

            if needs_history and previous is None:
                raise AdapterSpecError(
                    f"Frame {entry.frame_id} on manifest line {entry.line_number} has no valid "
                    "previous RGB output; supply initial_history for the first frame and after "
                    "every scene/resolution/camera-cut/frame-gap reset that needs a seed."
                )

            preprocess_started = perf_counter()
            inputs = adapt_frame(frame, spec, engine.input_shapes, history=previous)
            preprocess_ms = (perf_counter() - preprocess_started) * 1000.0

            inference_started = perf_counter()
            result = engine.infer(inputs)
            call_wall_ms = (perf_counter() - inference_started) * 1000.0
            inference_times.append(float(result.inference_ms))
            complete_times.append((perf_counter() - frame_started) * 1000.0)
            total_compile_ms += float(result.compile_ms)

            preview = preview_from_outputs(
                result.outputs, spec, target_size=frame.size
            )
            destination = destinations[index]
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".tmp")
            Image.fromarray(
                np.rint(np.clip(preview, 0.0, 1.0) * 255.0).astype(np.uint8)
            ).save(temporary, format="PNG")
            temporary.replace(destination)

            temporal.store(frame, preview)
            frame_reports.append(
                {
                    "index": index,
                    "manifest_line": entry.line_number,
                    "frame_id": frame.frame_id,
                    "timestamp": frame.timestamp,
                    "scene_id": frame.scene_id,
                    "camera_cut": frame.camera_cut,
                    "input": str(entry.color_path),
                    "output": str(destination),
                    "size": [frame.width, frame.height],
                    "available_sources": sorted(frame.model_inputs()),
                    "history_input_used": needs_history and previous is not None,
                    "history_source": history_source,
                    "history_reset_reason": reset_reason,
                    "preprocessing_ms": round(preprocess_ms, 3),
                    "compile_ms": round(float(result.compile_ms), 3),
                    "inference_ms": round(float(result.inference_ms), 3),
                    "model_call_wall_ms": round(call_wall_ms, 3),
                    "frame_wall_ms": round(float(complete_times[-1]), 3),
                    "device_selected": result.device,
                    "device_name": result.device_name,
                }
            )
    finally:
        close = getattr(engine, "close", None)
        if callable(close):
            close()

    report: dict[str, object] = {
        "engine": "TrSS",
        "operation": "offline_frame_sequence_inference",
        "model": str(resolved_model),
        "adapter_spec": str(spec_path.expanduser().resolve()),
        "manifest": str(manifest_path.expanduser().resolve()),
        "device_requested": device,
        "device_selected": frame_reports[-1]["device_selected"],
        "device_name": frame_reports[-1]["device_name"],
        "frames_processed": len(frame_reports),
        "output_directory": str(destination_dir),
        "temporal_history": {
            "enabled_for_model": needs_history,
            "seed_rule": (
                "Use previous normalized RGB output when continuous; initial_history is "
                "required on startup or after a temporal discontinuity when the model needs history."
            ),
            "motion_reprojection": "not performed; only the declared model adapter can implement it",
        },
        "aggregate": {
            "total_compile_ms": round(total_compile_ms, 3),
            "inference_ms_median": round(float(np.median(inference_times)), 3),
            "inference_ms_p95": round(float(np.percentile(inference_times, 95)), 3),
            "frame_wall_ms_median": round(float(np.median(complete_times)), 3),
            "frame_wall_ms_p95": round(float(np.percentile(complete_times, 95)), 3),
        },
        "frames": frame_reports,
        "note": (
            "Offline recorded-frame processing only. This command does not capture a live game, "
            "invent missing depth or motion, perform motion-vector reprojection, or establish "
            "real-time performance on a physical Intel Arc GPU."
        ),
    }
    return report
