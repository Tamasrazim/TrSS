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

from .backends.openvino_backend import BackendUnavailable, load_openvino
from .contracts import Layout, image_to_tensor, tensor_to_image


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


class ImageInferenceEngine:
    """Run one-input, one-output RGB image models on an OpenVINO device."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "GPU",
        layout: Layout = "NCHW",
        input_range: tuple[float, float] = (0.0, 1.0),
        output_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = device
        self.layout = layout
        self.input_range = input_range
        self.output_range = output_range

        if layout not in ("NCHW", "NHWC"):
            raise ModelContractError(f"Unsupported tensor layout: {layout!r}.")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")

        ov = load_openvino()
        try:
            self.core = ov.Core()
        except Exception as exc:
            raise BackendUnavailable(f"OpenVINO could not initialize: {exc}") from exc

        exposed_devices = [str(item) for item in self.core.available_devices]
        if device not in exposed_devices:
            available = ", ".join(exposed_devices) if exposed_devices else "none"
            raise BackendUnavailable(
                f"Requested device {device!r} is not available. OpenVINO reports: {available}. "
                "For Intel Arc, install a compatible Intel graphics driver and the optional "
                'XESS5 GPU dependencies with: python -m pip install -e ".[gpu]"'
            )

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

        target_height = model_height or 0
        target_width = model_width or 0
        # Keep the original shape contract so dynamic models can be safely
        # re-specialized when a later frame uses a different resolution.
        self._base_model = model.clone()
        self._model = self._base_model.clone()

        # The actual dimensions are known when infer() is called. Leave truly
        # dynamic spatial dimensions dynamic until then; the model is reshaped
        # to the actual image in _prepare_model().
        self._input_name = input_port.get_any_name()
        self._output_name = output_port.get_any_name()
        self._device_name = self._read_device_name(device)
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
