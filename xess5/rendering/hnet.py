"""Specialized adapter for the documented 16-channel HNet-style frame contract.

This implements the publicly documented frame contract as an independent
pre/post-processing component. It contains no trained weights, vendor DLL,
network graph, Vulkan hook, or proprietary material. A matching model must
still be supplied by the user and must expose one [1,16,H,W] input and one
[1,4,H,W] output (or the configured output layout).

Research references and licenses are recorded in docs/THIRD-PARTY-RESEARCH.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


class HNetContractError(ValueError):
    """The color, model shapes, or neural head does not match this adapter."""


OutputLayout = Literal["NCHW", "NHWC"]


def _half(value: np.ndarray | float) -> np.ndarray:
    """Round through IEEE FP16 and return the rounded value widened to FP32."""
    return np.asarray(value, dtype=np.float32).astype(np.float16).astype(np.float32)


@dataclass(frozen=True, slots=True)
class NetworkGeometry:
    """Original frame dimensions and the padded network's working dimensions."""

    output_width: int
    output_height: int
    network_width: int
    network_height: int

    def __post_init__(self) -> None:
        if min(
            self.output_width,
            self.output_height,
            self.network_width,
            self.network_height,
        ) <= 0:
            raise HNetContractError("All frame and network dimensions must be positive.")
        if (
            self.network_width < self.output_width
            or self.network_height < self.output_height
        ):
            raise HNetContractError("Network dimensions must cover the source frame.")

    @property
    def is_identity(self) -> bool:
        return (
            self.output_width == self.network_width
            and self.output_height == self.network_height
        )

    @staticmethod
    def _mirror_indices(source_extent: int, target_extent: int) -> np.ndarray:
        if source_extent <= 0 or target_extent < source_extent:
            raise HNetContractError("Invalid mirror-padding dimensions.")
        index = np.arange(target_extent, dtype=np.int64)
        # Reflect across the last valid pixel without repeating the edge:
        # 0, 1, 2, 3, 2, 1, ... for a source extent of four.
        return np.where(
            index < source_extent,
            index,
            np.maximum(0, 2 * source_extent - 2 - index),
        )

    def source_rows(self) -> np.ndarray:
        return self._mirror_indices(self.output_height, self.network_height)

    def source_columns(self) -> np.ndarray:
        return self._mirror_indices(self.output_width, self.network_width)

    def extend(self, image: np.ndarray) -> np.ndarray:
        """Mirror-pad an HWC image using the model contract's edge convention."""
        value = np.asarray(image)
        if value.ndim != 3:
            raise HNetContractError("Only HWC images can be extended by this adapter.")
        if value.shape[:2] != (self.output_height, self.output_width):
            raise HNetContractError(
                f"Expected frame {(self.output_height, self.output_width)}, "
                f"received {value.shape[:2]}."
            )
        return value[self.source_rows()[:, None], self.source_columns()[None, :], :]

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crop a network-sized HWC output back to its original dimensions."""
        value = np.asarray(image)
        if value.ndim != 3 or value.shape[:2] != (
            self.network_height,
            self.network_width,
        ):
            raise HNetContractError(
                "The neural output must match the network's padded spatial dimensions."
            )
        return value[: self.output_height, : self.output_width, :]


def _field_alignment(extent: int) -> int:
    """Find the graph-specific multiple needed for exact staged reductions."""
    size = int(extent)
    if size <= 0:
        raise HNetContractError("Frame extents must be positive.")
    reductions = 0
    for level in range(6):
        half_extent = -(-((size + 1) // 2) // 4) * 4
        reductions += int(half_extent < size)
        reductions += int(level == 0 and half_extent % 8 != 0)
        size = half_extent
    return 1 << reductions


def network_geometry(width: int, height: int, *, minimum: int = 320) -> NetworkGeometry:
    """Resolve the HNet frame field while respecting its minimum and graph alignment.

    The padding convention follows the published DLSS-NR-on-Intel research notes.
    The returned geometry is deterministic; it does not scale or crop the source
    before feature assembly.
    """
    width, height, minimum = int(width), int(height), int(minimum)
    if width <= 0 or height <= 0:
        raise HNetContractError("Frame width and height must be positive.")
    floor = max(128, minimum)
    align_width = _field_alignment(width)
    align_height = _field_alignment(height)
    field_width = max(floor, -(-width // align_width) * align_width)
    field_height = max(floor, -(-height // align_height) * align_height)

    # Avoid the fully aligned bottleneck field described in the recovered graph
    # analysis; for those dimensions the vendor allocates one more width alignment.
    if (
        field_width % (4 * align_width) == 0
        and field_height % (4 * align_height) == 0
    ):
        field_width += align_width

    network_width = -(-field_width // 64) * 64
    network_height = -(-field_height // 64) * 64
    return NetworkGeometry(width, height, network_width, network_height)


def _dynamic_shift_mix(value: np.ndarray) -> np.ndarray:
    value = value.astype(np.uint32, copy=False)
    shift = (value >> np.uint32(28)) + np.uint32(4)
    mixed = value ^ (value >> shift)
    return (mixed * np.uint32(0x108EF2D9)).astype(np.uint32)


def _uniform24(value: np.ndarray) -> np.ndarray:
    mixed = _dynamic_shift_mix(value)
    bits = (mixed >> np.uint32(30)) ^ (mixed >> np.uint32(8))
    # The increment avoids a zero radius for the Box-Muller transform.
    return (bits + np.uint32(1)).astype(np.float32) * np.float32(
        5.960464477539063e-8
    )


def deterministic_noise(height: int, width: int, frame_index: int = 0) -> np.ndarray:
    """Generate the three deterministic, half-rounded Gaussian channels."""
    height, width, frame_index = int(height), int(width), int(frame_index)
    if height <= 0 or width <= 0 or frame_index < 0:
        raise HNetContractError("Noise dimensions must be positive and frame index non-negative.")

    yy, xx = np.indices((height, width), dtype=np.uint32)
    seed = yy * np.uint32(0xD8163841)
    seed ^= xx * np.uint32(0x8DA6B343)
    seed ^= np.uint32((frame_index * 0x9E3779B9) & 0xFFFFFFFF)
    seed ^= np.uint32(0x243F6A88)
    multiplied = _dynamic_shift_mix(seed)
    mixed = multiplied ^ (multiplied >> np.uint32(22))

    radius_a_uniform = _uniform24(
        mixed * np.uint32(0xCAA5B80D) + np.uint32(0x21DD796B)
    )
    angle_b_uniform = _uniform24(
        mixed * np.uint32(0x83232C31) + np.uint32(0x3463E0AC)
    )
    radius_b_uniform = _uniform24(
        mixed * np.uint32(0x2C9277B5) + np.uint32(0xAC564B05)
    )
    angle_a_uniform = _uniform24(
        mixed * np.uint32(0xFA6DC5F9) + np.uint32(0x4712A88E)
    )

    radius_a = np.sqrt(
        np.float32(-2.0) * np.log(radius_a_uniform)
    ).astype(np.float32)
    radius_b = np.sqrt(
        np.float32(-2.0) * np.log(radius_b_uniform)
    ).astype(np.float32)
    tau = np.float32(6.2831854820251465)
    angle_a = (tau * angle_a_uniform).astype(np.float32)
    angle_b = (tau * angle_b_uniform).astype(np.float32)
    return np.stack(
        (
            _half(radius_b * np.cos(angle_a)),
            _half(radius_b * np.sin(angle_a)),
            _half(radius_a * np.cos(angle_b)),
        ),
        axis=-1,
    ).astype(np.float32)


def _validate_rgb01(value: np.ndarray, *, name: str) -> np.ndarray:
    """Require HWC RGB float-like pixels in the explicitly declared 0..1 range."""
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3:
        raise HNetContractError(f"{name} must be an HxWx3 RGB image.")
    if not np.issubdtype(image.dtype, np.number) or np.issubdtype(
        image.dtype, np.complexfloating
    ):
        raise HNetContractError(f"{name} must contain real numeric pixels.")
    image = np.asarray(image, dtype=np.float32)
    if not np.isfinite(image).all():
        raise HNetContractError(f"{name} contains NaN or infinite pixels.")
    if np.any(image < 0.0) or np.any(image > 1.0):
        raise HNetContractError(
            f"{name} must already be normalized to 0..1 RGB. Do not pass uint8 pixels directly."
        )
    return image


def scaled_color(color: np.ndarray) -> np.ndarray:
    """Apply the model's three distinct FP16 rounding points to RGB pixels."""
    sampled = _half(color)
    centered = _half(sampled - np.float32(0.5))
    return _half(centered * np.float32(0.125))


@dataclass(frozen=True, slots=True)
class HNetControls:
    """The three documented conditioning scalars supplied to the network."""

    style_index: int = 0
    tone_strength: float = 1.0
    structure_strength: float = 1.0

    def validate(self) -> None:
        if (
            isinstance(self.style_index, bool)
            or not isinstance(self.style_index, (int, np.integer))
            or not 0 <= self.style_index <= 127
        ):
            raise HNetContractError("style_index must be an integer from 0 through 127.")
        if not np.isfinite(self.tone_strength) or not np.isfinite(self.structure_strength):
            raise HNetContractError("Tone and structure strengths must be finite.")


def make_features(
    color: np.ndarray,
    *,
    frame_index: int = 0,
    history: np.ndarray | None = None,
    controls: HNetControls = HNetControls(),
    geometry: NetworkGeometry | None = None,
) -> tuple[np.ndarray, NetworkGeometry]:
    """Assemble one padded [H,W,16] feature input in the documented channel order.

    Color and history are RGB float32 in 0..1. On the first frame, history is the
    current color. This function has no real engine motion vectors, so history
    reprojection is identity; do not treat this as motion-compensated temporal NR.
    """
    color = _validate_rgb01(color, name="color")
    height, width = color.shape[:2]
    controls.validate()

    if geometry is None:
        geometry = network_geometry(width, height)
    if (geometry.output_width, geometry.output_height) != (width, height):
        raise HNetContractError("Geometry output size must match the current frame.")

    if history is None:
        history_color = color
    else:
        history_color = _validate_rgb01(history, name="history")
        if history_color.shape != color.shape:
            raise HNetContractError("History dimensions must match the current color frame.")

    extended_color = geometry.extend(color)
    extended_history = geometry.extend(history_color)
    features = np.zeros(
        (geometry.network_height, geometry.network_width, 16),
        dtype=np.float32,
    )

    features[..., 0:3] = deterministic_noise(
        geometry.network_height, geometry.network_width, frame_index
    )
    features[..., 3] = np.float32(1.0)
    features[..., 4:7] = scaled_color(extended_color)
    features[..., 7:10] = scaled_color(extended_history)
    features[..., 10] = _half(np.float32(controls.style_index / 128.0))
    features[..., 11] = _half(np.float32(controls.tone_strength))
    features[..., 12] = _half(np.float32(controls.structure_strength))
    features[..., 13] = _half(np.float32(-1.0))
    features[..., 14] = _half(np.float32(-1.0))
    # Channel 15 deliberately remains zero.
    return features, geometry


def compose_head(
    head: np.ndarray,
    color: np.ndarray,
    *,
    history: np.ndarray | None = None,
    blend_scale: float = 0.73974609375,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose the 4-channel model head into the image and next temporal state.

    The first three output channels are a residual; channel four is the temporal
    gate logit. Returns (current display RGB, next ungraded history), both HWC
    float32 in 0..1. Style grades, masks, motion reprojection, and post-processing
    beyond the published head/temporal blend are deliberately separate.
    """
    color = _validate_rgb01(color, name="color")
    raw = np.asarray(head)
    if raw.ndim != 3 or raw.shape[:2] != color.shape[:2] or raw.shape[2] != 4:
        raise HNetContractError(
            f"head must have shape {(color.shape[0], color.shape[1], 4)}, "
            f"received {raw.shape!r}."
        )
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise HNetContractError("head must contain real numeric values.")
    head = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(head).all():
        raise HNetContractError("head contains NaN or infinite values.")
    if not np.isfinite(blend_scale) or blend_scale <= 0:
        raise HNetContractError("blend_scale must be a positive finite value.")

    current_history = color if history is None else _validate_rgb01(history, name="history")
    if current_history.shape != color.shape:
        raise HNetContractError("History dimensions must match the current color frame.")

    residual = _half(head[..., :3]) * np.float32(0.25)
    predicted = np.clip(color + residual, 0.0, 1.0).astype(np.float32)

    # The gate logit is rounded to FP16 before sigmoid; this matches the documented
    # contract, and keeps the gate independent of the input precision used by NumPy.
    logit = _half(head[..., 3:4])
    with np.errstate(over="ignore", invalid="ignore"):
        sigmoid = np.float32(1.0) / (
            np.float32(1.0) + np.exp(-logit, dtype=np.float32)
        )
    scale_half = np.float32(_half(np.float32(blend_scale)))
    alpha = np.clip(sigmoid * scale_half, 0.0, 1.0)
    composed = np.clip(
        predicted + alpha * (current_history - predicted),
        0.0,
        1.0,
    ).astype(np.float32)
    return composed, composed.copy()


@dataclass(frozen=True, slots=True)
class HNetFrameResult:
    """A processed RGB frame, temporal state, raw head, and measured GPU timings."""

    image: np.ndarray
    history: np.ndarray
    head: np.ndarray
    geometry: NetworkGeometry
    device: str
    device_name: str
    compile_ms: float
    inference_ms: float
    total_ms: float


class HNetFrameEngine:
    """Run a supplied 16-channel-input/4-channel-output graph on the selected GPU."""

    def __init__(self, tensor_engine: Any, *, output_layout: OutputLayout = "NCHW") -> None:
        if output_layout not in ("NCHW", "NHWC"):
            raise HNetContractError("output_layout must be NCHW or NHWC.")
        if len(tensor_engine.input_names) != 1 or len(tensor_engine.output_names) != 1:
            raise HNetContractError("HNet frame contract requires exactly one input and one output.")
        if not hasattr(tensor_engine, "input_shapes"):
            raise HNetContractError("The selected tensor engine does not expose input shapes.")
        self.engine = tensor_engine
        self.input_name = tensor_engine.input_names[0]
        self.output_name = tensor_engine.output_names[0]
        self.input_shape = tensor_engine.input_shapes[self.input_name]
        self.output_layout = output_layout

        shape = self.input_shape
        if len(shape) != 4 or shape[0] not in (None, 1) or shape[1] not in (None, 16):
            raise HNetContractError(
                f"Expected NCHW input [1,16,H,W], got {shape!r}. "
                "Export or select a graph with the documented HNet input contract."
            )

    def process_frame(
        self,
        color: np.ndarray,
        *,
        history: np.ndarray | None = None,
        frame_index: int = 0,
        controls: HNetControls = HNetControls(),
    ) -> HNetFrameResult:
        """Prepare features, execute the supplied model, compose its head, and crop."""
        import time

        started = time.perf_counter()
        color = _validate_rgb01(color, name="color")
        features, geometry = make_features(
            color,
            frame_index=frame_index,
            history=history,
            controls=controls,
        )
        expected = self.input_shape
        actual = (1, 16, geometry.network_height, geometry.network_width)
        for axis, (required, got) in enumerate(zip(expected, actual)):
            if required is not None and required != got:
                raise HNetContractError(
                    f"Model input axis {axis} is fixed at {required}, "
                    f"but this frame requires {got}. A dynamic spatial model or matching field is needed."
                )

        input_tensor = np.ascontiguousarray(
            np.transpose(features, (2, 0, 1))[None, ...],
            dtype=np.float32,
        )
        result = self.engine.infer({self.input_name: input_tensor})
        raw = np.asarray(result.outputs[self.output_name], dtype=np.float32)

        if raw.ndim != 4 or raw.shape[0] != 1:
            raise HNetContractError(
                f"Expected rank-four batch-one model output, got {raw.shape!r}."
            )
        if self.output_layout == "NCHW":
            if raw.shape[1] != 4:
                raise HNetContractError(
                    f"Expected four output channels in NCHW layout, got {raw.shape!r}."
                )
            head = np.transpose(raw[0], (1, 2, 0))
        else:
            if raw.shape[3] != 4:
                raise HNetContractError(
                    f"Expected four output channels in NHWC layout, got {raw.shape!r}."
                )
            head = raw[0]

        if head.shape[:2] != (geometry.network_height, geometry.network_width):
            raise HNetContractError(
                f"Model output spatial size {head.shape[:2]} does not match network field "
                f"{(geometry.network_height, geometry.network_width)}."
            )

        history_field = geometry.extend(color if history is None else _validate_rgb01(history, name="history"))
        if history_field.shape[:2] != color_field_shape(geometry):
            raise HNetContractError("Padded history has an unexpected geometry.")
        rendered_field, next_history_field = compose_head(
            head,
            geometry.extend(color),
            history=history_field,
        )
        image = geometry.crop(rendered_field)
        next_history = geometry.crop(next_history_field)
        total_ms = (time.perf_counter() - started) * 1000.0

        return HNetFrameResult(
            image=image,
            history=next_history,
            head=head,
            geometry=geometry,
            device=result.device,
            device_name=result.device_name,
            compile_ms=result.compile_ms,
            inference_ms=result.inference_ms,
            total_ms=total_ms,
        )


def color_field_shape(geometry: NetworkGeometry) -> tuple[int, int]:
    """Return the expected height/width shape of a padded RGB field."""
    return (geometry.network_height, geometry.network_width)
