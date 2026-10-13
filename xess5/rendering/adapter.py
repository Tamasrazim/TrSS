"""Turn real frame buffers into the tensors a documented model expects.

The adapter intentionally refuses undocumented guesses: each model port must
name its source, layout, and normalization contract. It does not create fake
depth, motion vectors, noise, or temporal history. A model with additional
non-image-specific preprocessing needs its own adapter rather than a relaxed
generic conversion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .frame import FramePacket

SourceName = Literal[
    "color", "depth", "motion_vectors", "noise", "history", "conditioning"
]
LayoutName = Literal["NCHW", "NHWC", "CHW", "HWC", "HW", "VECTOR", "NC"]
ResizeMode = Literal["nearest", "bilinear"]


class AdapterSpecError(ValueError):
    """A model adapter specification is missing or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class InputBinding:
    """Preprocessing instructions for one named model input port."""

    source: SourceName
    layout: LayoutName
    source_range: tuple[float, float]
    target_range: tuple[float, float]
    resize: ResizeMode = "bilinear"
    channel_order: Literal["RGB", "BGR"] = "RGB"

    @classmethod
    def from_dict(cls, value: Any, *, port_name: str) -> "InputBinding":
        if not isinstance(value, dict):
            raise AdapterSpecError(f"Binding for {port_name!r} must be a JSON object.")

        allowed_keys = {
            "source", "layout", "source_range", "target_range", "resize", "channel_order"
        }
        unknown = sorted(set(value) - allowed_keys)
        if unknown:
            raise AdapterSpecError(
                f"Binding for {port_name!r} contains unknown fields: {', '.join(unknown)}."
            )

        source = value.get("source")
        if source not in (
            "color", "depth", "motion_vectors", "noise", "history", "conditioning"
        ):
            raise AdapterSpecError(
                f"Binding for {port_name!r} needs a supported source "
                "(color/depth/motion_vectors/noise/history/conditioning)."
            )

        layout = value.get("layout")
        if layout not in ("NCHW", "NHWC", "CHW", "HWC", "HW", "VECTOR", "NC"):
            raise AdapterSpecError(
                f"Binding for {port_name!r} has unsupported layout {layout!r}."
            )

        def parse_range(key: str) -> tuple[float, float]:
            raw = value.get(key)
            if (
                not isinstance(raw, list)
                or len(raw) != 2
                or isinstance(raw[0], bool)
                or isinstance(raw[1], bool)
            ):
                raise AdapterSpecError(
                    f"Binding for {port_name!r} must declare {key} as [MIN, MAX]."
                )
            try:
                low, high = float(raw[0]), float(raw[1])
            except (TypeError, ValueError) as exc:
                raise AdapterSpecError(
                    f"Binding for {port_name!r} has an invalid {key}."
                ) from exc
            if not np.isfinite(low) or not np.isfinite(high) or low >= high:
                raise AdapterSpecError(
                    f"Binding for {port_name!r}: {key} values must be finite and ascending."
                )
            return low, high

        resize = value.get("resize", "bilinear")
        if resize not in ("nearest", "bilinear"):
            raise AdapterSpecError(
                f"Binding for {port_name!r} has unsupported resize mode {resize!r}."
            )

        channel_order = value.get("channel_order", "RGB")
        if channel_order not in ("RGB", "BGR"):
            raise AdapterSpecError(
                f"Binding for {port_name!r} channel_order must be RGB or BGR."
            )
        if channel_order != "RGB" and source not in ("color", "history", "noise"):
            raise AdapterSpecError(
                f"Binding for {port_name!r} may use BGR only for RGB image-like sources."
            )

        return cls(
            source=source,
            layout=layout,
            source_range=parse_range("source_range"),
            target_range=parse_range("target_range"),
            resize=resize,
            channel_order=channel_order,
        )


@dataclass(frozen=True, slots=True)
class PreviewBinding:
    """Explicit contract for interpreting one model output as an RGB image."""

    port: str
    layout: Literal["NCHW", "NHWC"]
    source_range: tuple[float, float]
    channel_order: Literal["RGB", "BGR"] = "RGB"

    @classmethod
    def from_dict(cls, value: Any) -> "PreviewBinding":
        if not isinstance(value, dict):
            raise AdapterSpecError("preview_output must be a JSON object.")
        unknown = sorted(set(value) - {"port", "layout", "source_range", "channel_order"})
        if unknown:
            raise AdapterSpecError(
                "preview_output contains unknown fields: " + ", ".join(unknown)
            )
        port = value.get("port")
        if not isinstance(port, str) or not port.strip():
            raise AdapterSpecError("preview_output.port must be a non-empty model output name.")
        layout = value.get("layout")
        if layout not in ("NCHW", "NHWC"):
            raise AdapterSpecError("preview_output.layout must be NCHW or NHWC.")
        raw_range = value.get("source_range")
        if (
            not isinstance(raw_range, list)
            or len(raw_range) != 2
            or isinstance(raw_range[0], bool)
            or isinstance(raw_range[1], bool)
        ):
            raise AdapterSpecError("preview_output.source_range must be [MIN, MAX].")
        try:
            low, high = float(raw_range[0]), float(raw_range[1])
        except (TypeError, ValueError) as exc:
            raise AdapterSpecError("preview_output.source_range is invalid.") from exc
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise AdapterSpecError(
                "preview_output.source_range values must be finite and ascending."
            )
        channel_order = value.get("channel_order", "RGB")
        if channel_order not in ("RGB", "BGR"):
            raise AdapterSpecError("preview_output.channel_order must be RGB or BGR.")
        return cls(
            port=port.strip(),
            layout=layout,
            source_range=(low, high),
            channel_order=channel_order,
        )


@dataclass(frozen=True)
class ModelAdapterSpec:
    """Versioned mapping between model input ports and optional display output."""

    inputs: dict[str, InputBinding]
    preview_output: PreviewBinding | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "ModelAdapterSpec":
        if not isinstance(value, dict):
            raise AdapterSpecError("The adapter spec root must be a JSON object.")
        if value.get("version") != 1:
            raise AdapterSpecError("Adapter spec version must be 1.")
        if set(value) - {"version", "description", "inputs", "preview_output"}:
            unknown = sorted(set(value) - {"version", "description", "inputs", "preview_output"})
            raise AdapterSpecError(
                "Adapter spec has unknown root fields: " + ", ".join(unknown)
            )

        raw_inputs = value.get("inputs")
        if not isinstance(raw_inputs, dict) or not raw_inputs:
            raise AdapterSpecError("Adapter spec needs a non-empty inputs object.")

        bindings: dict[str, InputBinding] = {}
        for raw_name, raw_binding in raw_inputs.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise AdapterSpecError("Model input names must be non-empty strings.")
            name = raw_name.strip()
            if name in bindings:
                raise AdapterSpecError(f"Duplicate model input name {name!r}.")
            bindings[name] = InputBinding.from_dict(raw_binding, port_name=name)

        preview_raw = value.get("preview_output")
        preview = (
            PreviewBinding.from_dict(preview_raw)
            if preview_raw is not None
            else None
        )
        return cls(inputs=bindings, preview_output=preview)

    @classmethod
    def load(cls, path: str | Path) -> "ModelAdapterSpec":
        spec_path = Path(path).expanduser().resolve()
        try:
            data = json.loads(spec_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise FileNotFoundError(f"Could not read adapter spec {spec_path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise AdapterSpecError(
                f"Adapter spec {spec_path} is not valid JSON: {exc}"
            ) from exc
        return cls.from_dict(data)


def _resize_image(array: np.ndarray, width: int, height: int, mode: ResizeMode) -> np.ndarray:
    """Resize H×W or H×W×C data without adding another imaging dependency."""
    if array.shape[:2] == (height, width):
        return array

    was_2d = array.ndim == 2
    pixels = array[:, :, None] if was_2d else array
    source_height, source_width, channels = pixels.shape

    if mode == "nearest":
        ys = np.minimum(
            (np.arange(height, dtype=np.float64) * source_height / height).astype(np.int64),
            source_height - 1,
        )
        xs = np.minimum(
            (np.arange(width, dtype=np.float64) * source_width / width).astype(np.int64),
            source_width - 1,
        )
        result = pixels[ys[:, None], xs[None, :], :]
    else:
        # Pixel-center bilinear interpolation. Keep the calculations explicit
        # so auxiliary float buffers don't get clipped like display images.
        y = np.clip(
            (np.arange(height, dtype=np.float64) + 0.5) * source_height / height - 0.5,
            0.0, source_height - 1.0,
        )
        x = np.clip(
            (np.arange(width, dtype=np.float64) + 0.5) * source_width / width - 0.5,
            0.0, source_width - 1.0,
        )
        y0 = np.floor(y).astype(np.int64)
        x0 = np.floor(x).astype(np.int64)
        y1 = np.minimum(y0 + 1, source_height - 1)
        x1 = np.minimum(x0 + 1, source_width - 1)
        fy = (y - y0).astype(np.float32)[:, None, None]
        fx = (x - x0).astype(np.float32)[None, :, None]
        values = pixels.astype(np.float32, copy=False)
        top_left = values[y0[:, None], x0[None, :], :]
        top_right = values[y0[:, None], x1[None, :], :]
        bottom_left = values[y1[:, None], x0[None, :], :]
        bottom_right = values[y1[:, None], x1[None, :], :]
        top = top_left * (1.0 - fx) + top_right * fx
        bottom = bottom_left * (1.0 - fx) + bottom_right * fx
        result = top * (1.0 - fy) + bottom * fy

    if was_2d:
        return result[:, :, 0]
    return result


def _source_for(binding: InputBinding, frame: FramePacket, history: np.ndarray | None) -> np.ndarray:
    """Find the exact named source; missing frame data is never fabricated."""
    if binding.source == "history":
        if history is None:
            raise AdapterSpecError(
                "The model requires temporal history, but none is available for this frame. "
                "Reset/prime the history or use a model spec without a history input."
            )
        value = history
    elif binding.source == "conditioning":
        value = frame.conditioning
    else:
        value = getattr(frame, binding.source)

    if value is None:
        raise AdapterSpecError(
            f"The model needs source {binding.source!r}, but the frame did not supply it."
        )
    result = np.asarray(value)
    if result.dtype.hasobject or not np.issubdtype(result.dtype, np.number):
        raise AdapterSpecError(f"Source {binding.source!r} must be a numeric array.")
    if np.issubdtype(result.dtype, np.complexfloating):
        raise AdapterSpecError(f"Source {binding.source!r} must not contain complex values.")
    if not np.isfinite(result).all():
        raise AdapterSpecError(f"Source {binding.source!r} contains NaN or infinite values.")

    if binding.source in ("color", "history", "noise") and result.ndim != 3:
        raise AdapterSpecError(
            f"Source {binding.source!r} must be an H×W×C array; got {result.shape!r}."
        )
    if binding.source == "depth":
        if result.ndim == 3 and result.shape[2] == 1:
            result = result[:, :, 0]
        if result.ndim != 2:
            raise AdapterSpecError("Depth must be an H×W array or an H×W×1 array.")
    if binding.source == "motion_vectors" and (
        result.ndim != 3 or result.shape[2] != 2
    ):
        raise AdapterSpecError("Motion vectors must be H×W×2 with their source's documented units.")
    if binding.source == "conditioning" and result.ndim != 1:
        raise AdapterSpecError("Conditioning must be a one-dimensional feature vector.")

    if binding.channel_order == "BGR":
        if result.ndim != 3 or result.shape[2] < 3:
            raise AdapterSpecError(
                f"Source {binding.source!r} cannot be converted to BGR because it has fewer than 3 channels."
            )
        result = result[:, :, :3][:, :, ::-1]

    return result


def _target_spatial_size(
    layout: LayoutName,
    model_shape: tuple[int | None, ...],
    source: np.ndarray,
    port_name: str,
) -> tuple[int, int] | None:
    """Resolve target (width, height) for a spatial model input."""
    if layout in ("NCHW", "NHWC", "CHW", "HWC"):
        expected_rank = 4 if layout in ("NCHW", "NHWC") else 3
        if len(model_shape) != expected_rank:
            raise AdapterSpecError(
                f"Model input {port_name!r} uses {layout} but shape is {model_shape!r}."
            )
        if layout == "NCHW":
            batch, channels, height, width = model_shape
        elif layout == "NHWC":
            batch, height, width, channels = model_shape
        elif layout == "CHW":
            channels, height, width = model_shape
        else:
            height, width, channels = model_shape

        expected_channels = channels
        if expected_channels is not None and source.ndim == 3 and source.shape[2] != expected_channels:
            raise AdapterSpecError(
                f"Model input {port_name!r} expects {expected_channels} channels, "
                f"but source has {source.shape[2]}. No implicit channel synthesis is performed."
            )
        if source.ndim == 2 and expected_channels not in (None, 1):
            raise AdapterSpecError(
                f"Model input {port_name!r} expects {expected_channels} channels, "
                "but source is a single-channel 2D buffer."
            )
        source_h, source_w = source.shape[:2]
        target_h = source_h if height is None else height
        target_w = source_w if width is None else width
        if batch not in (None, 1):
            raise AdapterSpecError(
                f"Model input {port_name!r} uses batch size {batch}; frame adapter supports batch 1."
            )
        return int(target_w), int(target_h)

    if layout == "HW":
        if len(model_shape) != 2 or source.ndim != 2:
            raise AdapterSpecError(
                f"Model input {port_name!r} uses HW and requires a rank-two source and model shape."
            )
        height = source.shape[0] if model_shape[0] is None else model_shape[0]
        width = source.shape[1] if model_shape[1] is None else model_shape[1]
        return int(width), int(height)

    if layout == "NC":
        if len(model_shape) != 2 or source.ndim != 1:
            raise AdapterSpecError(
                f"Model input {port_name!r} uses NC and requires a vector and rank-two model shape."
            )
        if model_shape[0] not in (None, 1):
            raise AdapterSpecError(
                f"Model input {port_name!r} uses batch size {model_shape[0]}; only batch 1 is supported."
            )
        if model_shape[1] is not None and model_shape[1] != source.shape[0]:
            raise AdapterSpecError(
                f"Conditioning vector for {port_name!r} has {source.shape[0]} values, "
                f"but model expects {model_shape[1]}."
            )
        return None

    if layout == "VECTOR":
        if len(model_shape) != 1 or source.ndim != 1:
            raise AdapterSpecError(
                f"Model input {port_name!r} uses VECTOR and requires rank-one model and source shapes."
            )
        if model_shape[0] is not None and model_shape[0] != source.shape[0]:
            raise AdapterSpecError(
                f"Vector input {port_name!r} has {source.shape[0]} values, "
                f"but model expects {model_shape[0]}."
            )
        return None

    raise AdapterSpecError(f"Unsupported layout {layout!r} for input {port_name!r}.")


def _format_tensor(
    source: np.ndarray,
    binding: InputBinding,
    model_shape: tuple[int | None, ...],
    port_name: str,
) -> np.ndarray:
    """Resample, normalize and arrange one source exactly as the spec declares."""
    target_size = _target_spatial_size(binding.layout, model_shape, source, port_name)

    if target_size is not None:
        target_width, target_height = target_size
        source = _resize_image(source, target_width, target_height, binding.resize)

    if source.ndim == 2:
        channel_count = 1
        image = source[:, :, None]
    elif source.ndim == 3:
        channel_count = source.shape[2]
        image = source
    else:
        channel_count = 0
        image = source

    if binding.layout == "NCHW":
        if image.ndim != 3:
            raise AdapterSpecError(f"Input {port_name!r} needs image data for NCHW layout.")
        tensor = np.transpose(image, (2, 0, 1))[None, ...]
    elif binding.layout == "NHWC":
        if image.ndim != 3:
            raise AdapterSpecError(f"Input {port_name!r} needs image data for NHWC layout.")
        tensor = image[None, ...]
    elif binding.layout == "CHW":
        if image.ndim != 3:
            raise AdapterSpecError(f"Input {port_name!r} needs image data for CHW layout.")
        tensor = np.transpose(image, (2, 0, 1))
    elif binding.layout == "HWC":
        if image.ndim != 3:
            raise AdapterSpecError(f"Input {port_name!r} needs image data for HWC layout.")
        tensor = image
    elif binding.layout == "HW":
        if source.ndim != 2:
            if channel_count == 1:
                tensor = image[:, :, 0]
            else:
                raise AdapterSpecError(f"Input {port_name!r} needs a single-channel HW source.")
        else:
            tensor = source
    elif binding.layout == "VECTOR":
        tensor = source
    elif binding.layout == "NC":
        tensor = source[None, :]
    else:
        raise AdapterSpecError(f"Unsupported layout {binding.layout!r}.")

    low, high = binding.source_range
    target_low, target_high = binding.target_range
    tensor = np.asarray(tensor, dtype=np.float32)
    tensor = ((tensor - low) / (high - low)) * (target_high - target_low) + target_low
    tensor = np.ascontiguousarray(tensor)

    if len(model_shape) != tensor.ndim:
        raise AdapterSpecError(
            f"Input {port_name!r} format produced rank {tensor.ndim}, "
            f"but model shape {model_shape!r} has rank {len(model_shape)}."
        )
    for axis, (actual, expected) in enumerate(zip(tensor.shape, model_shape)):
        if expected is not None and actual != expected:
            raise AdapterSpecError(
                f"Input {port_name!r} axis {axis}: adapter produced {actual}, model expects {expected}."
            )

    return tensor


def adapt_frame(
    frame: FramePacket,
    spec: ModelAdapterSpec,
    model_shapes: dict[str, tuple[int | None, ...]],
    *,
    history: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Translate a validated frame into one tensor per declared model input."""
    spec_names = set(spec.inputs)
    model_names = set(model_shapes)
    missing = sorted(model_names - spec_names)
    extra = sorted(spec_names - model_names)
    if missing or extra:
        details = []
        if missing:
            details.append("spec missing model ports: " + ", ".join(missing))
        if extra:
            details.append("spec contains unknown model ports: " + ", ".join(extra))
        raise AdapterSpecError("; ".join(details))

    result: dict[str, np.ndarray] = {}
    for port_name, binding in spec.inputs.items():
        source = _source_for(binding, frame, history)
        result[port_name] = _format_tensor(
            source, binding, model_shapes[port_name], port_name
        )
    return result



def preview_from_outputs(
    outputs: dict[str, np.ndarray],
    spec: ModelAdapterSpec,
    *,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Decode a declared RGB image output to HWC float32 in the 0..1 display range.

    This is intentionally optional: a model that emits feature planes or a
    packed/learned head must provide its own output composition function instead
    of labelling the raw tensor as RGB.
    """
    binding = spec.preview_output
    if binding is None:
        raise AdapterSpecError(
            "No preview_output is declared in the model adapter spec. "
            "Only direct RGB/RGBA image outputs can use the generic preview path."
        )
    if binding.port not in outputs:
        raise AdapterSpecError(
            f"preview_output refers to {binding.port!r}, but the model produced: "
            + ", ".join(sorted(outputs))
        )

    raw = np.asarray(outputs[binding.port])
    if raw.ndim != 4 or raw.shape[0] != 1:
        raise AdapterSpecError(
            f"Preview port {binding.port!r} must be a batch-one rank-four tensor; "
            f"received {raw.shape!r}."
        )

    if binding.layout == "NCHW":
        channels = int(raw.shape[1])
        image = np.transpose(raw[0], (1, 2, 0))
    else:
        channels = int(raw.shape[3])
        image = raw[0]

    if channels not in (3, 4):
        raise AdapterSpecError(
            f"Preview port {binding.port!r} must contain 3 or 4 channels, got {channels}. "
            "Feature tensors and packed heads require model-specific output composition."
        )
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise AdapterSpecError(
            f"Preview port {binding.port!r} has incompatible image shape {image.shape!r}."
        )

    image = np.asarray(image, dtype=np.float32)
    if not np.isfinite(image).all():
        raise AdapterSpecError(f"Preview port {binding.port!r} contains invalid values.")

    low, high = binding.source_range
    image = np.clip((image - low) / (high - low), 0.0, 1.0)
    image = image[:, :, :3]
    if binding.channel_order == "BGR":
        image = image[:, :, ::-1]

    target_width, target_height = target_size
    if (image.shape[1], image.shape[0]) != (target_width, target_height):
        image = _resize_image(image, target_width, target_height, "bilinear")
    return np.ascontiguousarray(image, dtype=np.float32)
