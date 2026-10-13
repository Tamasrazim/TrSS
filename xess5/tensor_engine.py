"""Multi-input OpenVINO inference core for graphics and temporal models.

Unlike ImageInferenceEngine, this class does not decide how scene color,
depth, motion vectors, noise, history, or conditioning are constructed. A
model-specific adapter prepares those tensors and binds them by model input
name. This keeps unknown model preprocessing from being guessed silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any

import numpy as np

from .backends.openvino_backend import (
    BackendUnavailable,
    load_openvino,
    resolve_device,
)
from .engine import ModelContractError, _dimension_length


def _port_name(port: Any, index: int, prefix: str) -> str:
    """Return a stable public binding name for one model port."""
    try:
        name = str(port.get_any_name())
    except Exception:
        name = f"{prefix}_{index}"
    return name or f"{prefix}_{index}"


def _port_numpy_dtype(port: Any) -> np.dtype:
    """Map common OpenVINO tensor element types to NumPy dtypes."""
    label = str(port.element_type).casefold()
    mappings = (
        (("f16", "float16"), np.float16),
        (("f32", "float32"), np.float32),
        (("f64", "float64"), np.float64),
        (("u8", "uint8"), np.uint8),
        (("i8", "int8"), np.int8),
        (("u16", "uint16"), np.uint16),
        (("i16", "int16"), np.int16),
        (("i32", "int32"), np.int32),
        (("i64", "int64"), np.int64),
    )
    for aliases, dtype in mappings:
        if any(alias in label for alias in aliases):
            return np.dtype(dtype)
    raise ModelContractError(
        f"Model tensor precision {port.element_type!s} is not supported by the "
        "generic adapter. Add an explicit conversion for this model."
    )


@dataclass(frozen=True)
class TensorInferenceResult:
    """Outputs and timings for one multi-input model invocation."""

    outputs: dict[str, np.ndarray]
    input_shapes: dict[str, tuple[int, ...]]
    output_shapes: dict[str, tuple[int, ...]]
    compile_ms: float
    inference_ms: float
    total_ms: float
    device: str
    device_name: str
    model_path: str


class TensorInferenceEngine:
    """Execute a model with explicit named inputs and multiple outputs.

    One engine instance is serialized by an internal lock, so the persistent
    inference request cannot be used concurrently. Create one instance per
    concurrent render stream when parallel inference is genuinely required.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "ARC",
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")

        ov = load_openvino()
        try:
            self.core = ov.Core()
        except Exception as exc:
            raise BackendUnavailable(f"OpenVINO could not initialize: {exc}") from exc

        self.device = resolve_device(self.core, device)
        try:
            self._base_model = self.core.read_model(str(self.model_path))
        except Exception as exc:
            raise ModelContractError(
                f"OpenVINO could not read model {self.model_path}: {exc}"
            ) from exc

        if not self._base_model.inputs:
            raise ModelContractError("The model has no inputs.")
        if not self._base_model.outputs:
            raise ModelContractError("The model has no outputs.")

        self._input_names = [
            _port_name(port, index, "input")
            for index, port in enumerate(self._base_model.inputs)
        ]
        self._output_names = [
            _port_name(port, index, "output")
            for index, port in enumerate(self._base_model.outputs)
        ]
        if len(set(self._input_names)) != len(self._input_names):
            raise ModelContractError(
                "Model input names are not unique. Export the model with unique names "
                "before binding frame tensors."
            )
        if len(set(self._output_names)) != len(self._output_names):
            raise ModelContractError(
                "Model output names are not unique. Export the model with unique names "
                "before binding output tensors."
            )

        self._input_dtypes = [
            _port_numpy_dtype(port) for port in self._base_model.inputs
        ]
        self._device_name = self._read_device_name()
        self._compiled_shape: tuple[tuple[int, ...], ...] | None = None
        self._compiled_model: Any | None = None
        self._request: Any | None = None
        self._lock = Lock()

    @property
    def input_names(self) -> tuple[str, ...]:
        """Input names the frame adapter must provide."""
        return tuple(self._input_names)

    @property
    def output_names(self) -> tuple[str, ...]:
        """Output names available from each inference."""
        return tuple(self._output_names)

    def _read_device_name(self) -> str:
        try:
            return str(self.core.get_property(self.device, "FULL_DEVICE_NAME"))
        except Exception:
            return self.device

    def _prepare_inputs(
        self,
        inputs: dict[str, np.ndarray],
    ) -> tuple[list[np.ndarray], tuple[tuple[int, ...], ...]]:
        expected = set(self._input_names)
        supplied = set(inputs)
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)
        if missing or extra:
            details = []
            if missing:
                details.append("missing inputs: " + ", ".join(missing))
            if extra:
                details.append("unexpected inputs: " + ", ".join(extra))
            raise ModelContractError("; ".join(details))

        arrays: list[np.ndarray] = []
        shapes: list[tuple[int, ...]] = []

        for index, (name, port, expected_dtype) in enumerate(
            zip(self._input_names, self._base_model.inputs, self._input_dtypes)
        ):
            array = np.asarray(inputs[name])
            if array.ndim != len(port.partial_shape):
                raise ModelContractError(
                    f"Input {name!r} expects rank {len(port.partial_shape)}, "
                    f"got shape {array.shape!r}."
                )
            if not np.issubdtype(array.dtype, np.number):
                raise ModelContractError(f"Input {name!r} must be numeric.")
            if not np.isfinite(array).all():
                raise ModelContractError(
                    f"Input {name!r} contains NaN or infinite values."
                )

            model_dimensions = [
                _dimension_length(dimension) for dimension in port.partial_shape
            ]
            for axis, (actual, expected) in enumerate(
                zip(array.shape, model_dimensions)
            ):
                if expected is not None and actual != expected:
                    raise ModelContractError(
                        f"Input {name!r} axis {axis} is fixed at {expected}, "
                        f"but received {actual}."
                    )
                if actual < 1:
                    raise ModelContractError(
                        f"Input {name!r} has an empty dimension at axis {axis}."
                    )

            # Convert only to the type declared by the model, and make the
            # array contiguous so the runtime does not need a hidden copy.
            array = np.ascontiguousarray(array, dtype=expected_dtype)
            arrays.append(array)
            shapes.append(tuple(int(value) for value in array.shape))

        return arrays, tuple(shapes)

    def _ensure_compiled(
        self,
        shapes: tuple[tuple[int, ...], ...],
    ) -> float:
        if self._compiled_model is not None and self._compiled_shape == shapes:
            return 0.0

        started = perf_counter()
        model = self._base_model.clone()
        reshape: dict[Any, list[int]] = {}

        for index, (port, actual_shape) in enumerate(
            zip(model.inputs, shapes)
        ):
            model_shape = [
                _dimension_length(dimension) for dimension in port.partial_shape
            ]
            if len(model_shape) != len(actual_shape):
                raise ModelContractError(
                    f"Input {self._input_names[index]!r} changed rank."
                )
            if any(dimension is None for dimension in model_shape):
                reshape[port] = list(actual_shape)

        if reshape:
            try:
                model.reshape(reshape)
            except Exception as exc:
                raise ModelContractError(
                    "The model does not accept the provided dynamic tensor shapes: "
                    f"{exc}"
                ) from exc

        try:
            compiled = self.core.compile_model(model, self.device)
            request = compiled.create_infer_request()
        except Exception as exc:
            raise BackendUnavailable(
                f"OpenVINO could not compile the model for {self.device} "
                f"({self._device_name}): {exc}"
            ) from exc

        self._compiled_model = compiled
        self._request = request
        self._compiled_shape = shapes
        return (perf_counter() - started) * 1000.0

    def infer(self, inputs: dict[str, np.ndarray]) -> TensorInferenceResult:
        """Run the model with one array per declared input and copy all outputs."""
        total_started = perf_counter()
        with self._lock:
            arrays, shapes = self._prepare_inputs(inputs)
            compile_ms = self._ensure_compiled(shapes)

            # OpenVINO accepts input indexes in the infer() mapping. Using
            # indexes avoids ambiguity when a model stores multiple tensor aliases.
            request_inputs = {index: array for index, array in enumerate(arrays)}
            inference_started = perf_counter()
            try:
                self._request.infer(request_inputs)
            except Exception as exc:
                raise RuntimeError(
                    f"Inference failed on {self.device} ({self._device_name}): {exc}"
                ) from exc
            inference_ms = (perf_counter() - inference_started) * 1000.0

            outputs: dict[str, np.ndarray] = {}
            output_shapes: dict[str, tuple[int, ...]] = {}
            for index, name in enumerate(self._output_names):
                output = np.array(
                    self._request.get_output_tensor(index).data,
                    copy=True,
                )
                if not np.isfinite(output).all():
                    raise RuntimeError(
                        f"Model output {name!r} contains NaN or infinite values."
                    )
                outputs[name] = output
                output_shapes[name] = tuple(int(value) for value in output.shape)

            total_ms = (perf_counter() - total_started) * 1000.0
            return TensorInferenceResult(
                outputs=outputs,
                input_shapes={
                    name: shape for name, shape in zip(self._input_names, shapes)
                },
                output_shapes=output_shapes,
                compile_ms=compile_ms,
                inference_ms=inference_ms,
                total_ms=total_ms,
                device=self.device,
                device_name=self._device_name,
                model_path=str(self.model_path),
            )
