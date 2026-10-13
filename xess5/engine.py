"""A small, explicit image-to-image inference engine.

This is the first inference contract for XESS5, not a claim of DLSS 5
compatibility. Neural-rendering models that require depth, motion vectors,
temporal state, noise, or extra conditioning need a dedicated adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from .backends.openvino_backend import BackendUnavailable, load_openvino, resolve_device
from .contracts import ChannelOrder, Layout, image_to_tensor, tensor_to_image


class ModelContractError(RuntimeError):
    """The selected model does not match the engine's current image contract."""


@dataclass(frozen=True)
class InferenceResult:
    """Processed image and measurements from one model call."""

    image: Image.Image
    inference_ms: float
    input_size: tuple[int, int]
    output_size: tuple[int, int]
    device: str
    device_name: str
    model_path: str


def _dimension_length(dimension: Any) -> int | None:
    """Return a fixed OpenVINO dimension length, or None for dynamic dimensions."""
    if dimension.is_static:
        return int(dimension.get_length())
    return None


def _numpy_dtype(element_type: Any) -> np.dtype:
    """Map model input precision to a NumPy type supported by this adapter."""
    label = str(element_type).lower()
    if "f16" in label or "float16" in label:
        return np.dtype(np.float16)
    if "f32" in label or "float32" in label:
        return np.dtype(np.float32)
    raise ModelContractError(
        f"Model input precision {element_type!s} is not supported by the initial "
        "RGB float engine. Use an FP16 or FP32 image input, or add a model-specific adapter."
    )


def available_devices() -> list[dict[str, str]]:
    """Report the inference devices exposed by OpenVINO."""
    from .backends.openvino_backend import list_devices

    return list_devices()


def inspect_model(model_path: str | Path) -> dict[str, object]:
    """Read a local model and describe its ports without compiling or running it."""
    resolved_path = Path(model_path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {resolved_path}")

    ov = load_openvino()
    try:
        core = ov.Core()
        model = core.read_model(str(resolved_path))
    except Exception as exc:
        raise ModelContractError(
            f"OpenVINO could not inspect model {resolved_path}: {exc}"
        ) from exc

    def describe_port(port: Any, index: int) -> dict[str, object]:
        shape = []
        for dimension in port.partial_shape:
            length = _dimension_length(dimension)
            shape.append(length)
        try:
            name = port.get_any_name()
        except Exception:
            name = f"input_{index}"
        return {
            "index": index,
            "name": str(name),
            "shape": shape,
            "shape_text": str(port.partial_shape),
            "precision": str(port.element_type),
        }

    inputs = [describe_port(port, i) for i, port in enumerate(model.inputs)]
    outputs = [describe_port(port, i) for i, port in enumerate(model.outputs)]
    return {
        "model": str(resolved_path),
        "input_count": len(inputs),
        "output_count": len(outputs),
        "inputs": inputs,
        "outputs": outputs,
        "supported_by_current_image_engine": (
            len(inputs) == 1
            and len(outputs) == 1
            and len(inputs[0]["shape"]) == 4
            and len(outputs[0]["shape"]) == 4
        ),
        "note": (
            "The current generic image engine requires one rank-four RGB input and "
            "one rank-four image output. Multi-input models for depth, motion vectors, "
            "noise, or temporal history need a specialized adapter."
        ),
    }


class ImageInferenceEngine:
    """Run one-input, one-output RGB image models on an OpenVINO device."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "ARC",
        layout: Layout = "NCHW",
        input_range: tuple[float, float] = (0.0, 1.0),
        output_range: tuple[float, float] = (0.0, 1.0),
        input_channel_order: ChannelOrder = "RGB",
        output_channel_order: ChannelOrder = "RGB",
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = device
        self.layout = layout
        self.input_range = input_range
        self.output_range = output_range
        self.input_channel_order = input_channel_order
        self.output_channel_order = output_channel_order

        if layout not in ("NCHW", "NHWC"):
            raise ModelContractError(f"Unsupported tensor layout: {layout!r}.")
        if input_channel_order not in ("RGB", "BGR"):
            raise ModelContractError(f"Unsupported input channel order: {input_channel_order!r}.")
        if output_channel_order not in ("RGB", "BGR"):
            raise ModelContractError(f"Unsupported output channel order: {output_channel_order!r}.")
        for label, value_range in (
            ("input", input_range),
            ("output", output_range),
        ):
            if (
                len(value_range) != 2
                or not np.isfinite(value_range[0])
                or not np.isfinite(value_range[1])
                or value_range[0] >= value_range[1]
            ):
                raise ModelContractError(
                    f"The {label} range must contain two finite values in ascending order."
                )
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")

        ov = load_openvino()
        try:
            self.core = ov.Core()
        except Exception as exc:
            raise BackendUnavailable(f"OpenVINO could not initialize: {exc}") from exc

        self.device = resolve_device(self.core, device)

        try:
            model = self.core.read_model(str(self.model_path))
        except Exception as exc:
            raise ModelContractError(
                f"OpenVINO could not read model {self.model_path}: {exc}"
            ) from exc

        if len(model.inputs) != 1 or len(model.outputs) != 1:
            raise ModelContractError(
                "The initial engine requires exactly one model input and one model output. "
                "Models with multiple inputs (for example depth, motion vectors, or history) "
                "need a model-specific adapter."
            )

        input_port = model.inputs[0]
        output_port = model.outputs[0]
        input_shape = list(input_port.partial_shape)

        if len(input_shape) != 4:
            raise ModelContractError(
                f"Expected a rank-four image input, got {input_port.partial_shape}."
            )

        dimensions = [_dimension_length(item) for item in input_shape]
        if layout == "NCHW":
            batch, channels, model_height, model_width = dimensions
        else:
            batch, model_height, model_width, channels = dimensions

        if batch not in (None, 1):
            raise ModelContractError(
                f"Expected model batch size 1 or dynamic, got {batch}."
            )
        if channels not in (None, 3):
            raise ModelContractError(
                f"Expected three RGB input channels or a dynamic channel dimension, got {channels}."
            )

        # Static spatial dimensions are respected. Dynamic spatial dimensions are
        # specialized to the incoming image size for a predictable first engine.
        self._height = model_height
        self._width = model_width
        self._input_dtype = _numpy_dtype(input_port.element_type)

        # Keep the original shape contract so dynamic models can be safely
        # re-specialized when a later frame uses a different resolution.
        self._base_model = model.clone()
        self._model = self._base_model.clone()

        # The actual dimensions are known when infer() is called. Leave truly
        # dynamic spatial dimensions dynamic until then; the model is reshaped
        # to the actual image in _prepare_model().
        self._device_name = self._read_device_name(self.device)
        self._compiled = None
        self._prepared_shape: tuple[int, ...] | None = None

        if output_shape := list(output_port.partial_shape):
            if len(output_shape) != 4:
                raise ModelContractError(
                    f"Expected a rank-four image output, got {output_port.partial_shape}."
                )
        else:
            raise ModelContractError("The model has no usable output shape.")

    def _read_device_name(self, device: str) -> str:
        try:
            return str(self.core.get_property(device, "FULL_DEVICE_NAME"))
        except Exception:
            return device

    def _prepare_model(self, width: int, height: int) -> None:
        if self.layout == "NCHW":
            target_shape = [1, 3, height, width]
        else:
            target_shape = [1, height, width, 3]

        shape_key = tuple(target_shape)
        if self._compiled is not None and self._prepared_shape == shape_key:
            return

        # Reshape a fresh model clone; never mutate the original contract.
        self._model = self._base_model.clone()
        input_port = self._model.inputs[0]
        current_shape = [_dimension_length(item) for item in input_port.partial_shape]
        resolved_shape = []
        for index, current_dimension in enumerate(current_shape):
            requested = target_shape[index]
            if current_dimension is not None and current_dimension != requested:
                # The caller has already chosen the model's fixed spatial size.
                # Any remaining mismatch means its static contract is incompatible.
                if index in ((2, 3) if self.layout == "NCHW" else (1, 2)):
                    raise ModelContractError(
                        f"Model input has a static shape {input_port.partial_shape}; "
                        "the requested dimensions do not match."
                    )
                raise ModelContractError(
                    f"Model input dimension {index} is fixed at {current_dimension}, "
                    f"but the image contract requires {requested}."
                )
            resolved_shape.append(requested if current_dimension is None else current_dimension)

        if any(value is None for value in current_shape):
            self._model.reshape({input_port: resolved_shape})

        try:
            self._compiled = self.core.compile_model(self._model, self.device)
        except Exception as exc:
            raise BackendUnavailable(
                f"OpenVINO could not compile the model for {self.device} "
                f"({self._device_name}): {exc}"
            ) from exc
        self._prepared_shape = shape_key

    def infer(self, image: Image.Image) -> InferenceResult:
        """Run one inference and return its output, timing, and effective dimensions."""
        if image.width < 1 or image.height < 1:
            raise ValueError("Input image dimensions must be positive.")

        width = self._width or image.width
        height = self._height or image.height
        self._prepare_model(width, height)

        tensor = image_to_tensor(
            image,
            layout=self.layout,
            width=width,
            height=height,
            value_range=self.input_range,
            channel_order=self.input_channel_order,
            dtype=self._input_dtype,
        )

        request = self._compiled.create_infer_request()
        started = perf_counter()
        request.infer({0: tensor})
        inference_ms = (perf_counter() - started) * 1000.0

        output = np.array(request.get_output_tensor(0).data, copy=True)
        output_image = tensor_to_image(
            output,
            layout=self.layout,
            value_range=self.output_range,
            channel_order=self.output_channel_order,
        )

        return InferenceResult(
            image=output_image,
            inference_ms=inference_ms,
            input_size=(width, height),
            output_size=output_image.size,
            device=self.device,
            device_name=self._device_name,
            model_path=str(self.model_path),
        )
