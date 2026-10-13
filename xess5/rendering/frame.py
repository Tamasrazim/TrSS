"""Validated game-frame inputs and safe temporal history management.

This file defines the data contract only. It does not pretend to reconstruct
depth or motion data when a capture path does not provide them, and it leaves
motion-vector reprojection to a model-specific GPU adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np
from numpy.typing import NDArray


ColorSpace = Literal["srgb", "linear-srgb", "pq", "hlg", "unknown"]
DepthEncoding = Literal["linear", "device-z", "normalized", "unknown"]
MotionUnits = Literal["pixel-per-frame", "normalized", "unknown"]


class FrameValidationError(ValueError):
    """A frame or one of its optional rendering inputs is invalid."""


def _array(
    value: np.ndarray,
    *,
    name: str,
    ndim: int,
    finite: bool = True,
) -> np.ndarray:
    """Validate an array without silently changing its numerical meaning."""
    result = np.asarray(value)
    if result.ndim != ndim:
        raise FrameValidationError(
            f"{name} must be rank {ndim}; received shape {result.shape!r}."
        )
    if not np.issubdtype(result.dtype, np.number):
        raise FrameValidationError(f"{name} must contain numeric values.")
    if finite and not np.isfinite(result).all():
        raise FrameValidationError(f"{name} contains NaN or infinite values.")
    return result


@dataclass(frozen=True, slots=True)
class FramePacket:
    """One rendered frame and any real auxiliary buffers supplied by its source.

    Color and optional image buffers use HWC layout. Values are not implicitly
    normalized: the source adapter must document whether color is sRGB-encoded,
    linear-light, HDR, or another representation before a model consumes it.
    Motion vectors are in pixels per frame unless the source adapter documents
    another convention. The packet preserves these values without guessing.
    """

    color: np.ndarray
    timestamp: float
    frame_id: int
    scene_id: str | None = None
    depth: np.ndarray | None = None
    motion_vectors: np.ndarray | None = None
    noise: np.ndarray | None = None
    conditioning: np.ndarray | None = None
    camera_cut: bool = False
    color_space: ColorSpace = "unknown"
    depth_encoding: DepthEncoding = "unknown"
    motion_units: MotionUnits = "unknown"

    def __post_init__(self) -> None:
        color = _array(self.color, name="color", ndim=3)
        if color.shape[0] < 1 or color.shape[1] < 1:
            raise FrameValidationError("color must have a non-empty width and height.")
        if color.shape[2] not in (3, 4):
            raise FrameValidationError(
                f"color must have RGB or RGBA channels, got {color.shape[2]}."
            )

        height, width = color.shape[:2]
        object.__setattr__(self, "color", color)

        try:
            timestamp = float(self.timestamp)
        except (TypeError, ValueError) as exc:
            raise FrameValidationError("timestamp must be a finite non-negative number.") from exc
        if not np.isfinite(timestamp) or timestamp < 0:
            raise FrameValidationError("timestamp must be a finite non-negative number.")
        object.__setattr__(self, "timestamp", timestamp)

        if (
            isinstance(self.frame_id, (bool, np.bool_))
            or not isinstance(self.frame_id, (int, np.integer))
            or self.frame_id < 0
        ):
            raise FrameValidationError("frame_id must be a non-negative integer.")
        if self.scene_id is not None and not isinstance(self.scene_id, str):
            raise FrameValidationError("scene_id must be a string or None.")
        if self.color_space not in ("srgb", "linear-srgb", "pq", "hlg", "unknown"):
            raise FrameValidationError(f"Unsupported color_space: {self.color_space!r}.")
        if self.depth_encoding not in ("linear", "device-z", "normalized", "unknown"):
            raise FrameValidationError(
                f"Unsupported depth_encoding: {self.depth_encoding!r}."
            )
        if self.motion_units not in (
            "pixel-per-frame", "normalized", "unknown"
        ):
            raise FrameValidationError(f"Unsupported motion_units: {self.motion_units!r}.")

        if self.depth is not None:
            depth = np.asarray(self.depth)
            if depth.ndim == 3 and depth.shape[2] == 1:
                depth = depth[:, :, 0]
            depth = _array(depth, name="depth", ndim=2)
            if depth.shape != (height, width):
                raise FrameValidationError(
                    f"depth must match color dimensions {(height, width)}, "
                    f"got {depth.shape!r}."
                )
            object.__setattr__(self, "depth", depth)

        if self.motion_vectors is not None:
            motion = _array(self.motion_vectors, name="motion_vectors", ndim=3)
            if motion.shape != (height, width, 2):
                raise FrameValidationError(
                    "motion_vectors must have shape "
                    f"{(height, width, 2)}, got {motion.shape!r}."
                )
            object.__setattr__(self, "motion_vectors", motion)

        if self.noise is not None:
            noise = _array(self.noise, name="noise", ndim=3)
            if noise.shape != (height, width, 3):
                raise FrameValidationError(
                    f"noise must have shape {(height, width, 3)}, got {noise.shape!r}."
                )
            object.__setattr__(self, "noise", noise)

        if self.conditioning is not None:
            conditioning = _array(
                self.conditioning, name="conditioning", ndim=1
            )
            if conditioning.size < 1:
                raise FrameValidationError("conditioning cannot be empty.")
            object.__setattr__(self, "conditioning", conditioning)

    @property
    def width(self) -> int:
        return int(self.color.shape[1])

    @property
    def height(self) -> int:
        return int(self.color.shape[0])

    @property
    def size(self) -> tuple[int, int]:
        """Return (width, height), the same order used by image APIs."""
        return (self.width, self.height)

    def model_inputs(self) -> Mapping[str, np.ndarray]:
        """Return only auxiliary inputs that the capture source really supplied."""
        result: dict[str, np.ndarray] = {"color": self.color}
        for name in ("depth", "motion_vectors", "noise", "conditioning"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


class TemporalHistory:
    """Keep the previous output and invalidate it across discontinuities.

    The renderer calls history_for() before inference, then store() after it
    has a valid output. The returned previous output is not motion-reprojected;
    the model adapter must use the current frame's motion vectors to reproject
    history if its model requires that.
    """

    def __init__(self, *, max_gap_seconds: float = 0.5) -> None:
        if not np.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be a positive finite number.")
        self.max_gap_seconds = float(max_gap_seconds)
        self._output: np.ndarray | None = None
        self._timestamp: float | None = None
        self._frame_id: int | None = None
        self._scene_id: str | None = None
        self._size: tuple[int, int] | None = None
        self.last_reset_reason: str = "not_initialized"

    @property
    def valid(self) -> bool:
        return self._output is not None

    def reset(self, reason: str = "explicit_reset") -> None:
        """Drop history and record why it was invalidated."""
        self._output = None
        self._timestamp = None
        self._frame_id = None
        self._scene_id = None
        self._size = None
        self.last_reset_reason = reason

    def history_for(self, frame: FramePacket) -> np.ndarray | None:
        """Return a copy of history only if it belongs to a continuous sequence."""
        if self._output is None:
            return None

        if frame.camera_cut:
            self.reset("camera_cut")
            return None
        if frame.size != self._size:
            self.reset("resolution_changed")
            return None
        if frame.scene_id != self._scene_id:
            self.reset("scene_changed")
            return None
        if self._frame_id is not None and frame.frame_id <= self._frame_id:
            self.reset("frame_order_changed")
            return None
        if self._frame_id is not None and frame.frame_id != self._frame_id + 1:
            self.reset("frame_sequence_gap")
            return None
        if self._timestamp is not None:
            delta = frame.timestamp - self._timestamp
            if delta <= 0:
                self.reset("timestamp_not_monotonic")
                return None
            if delta > self.max_gap_seconds:
                self.reset("frame_gap")
                return None

        self.last_reset_reason = ""
        return self._output.copy()

    def store(self, frame: FramePacket, output: np.ndarray) -> None:
        """Commit a processed frame only after validating its dimensions and values."""
        if frame.camera_cut:
            # The current frame starts the new temporal segment; do not keep the
            # prior scene's output, but allow this output to become future history.
            self.reset("camera_cut")

        # Keep the reset reason available for diagnostics until the next frame
        # successfully consumes history.
        image = _array(output, name="rendered_output", ndim=3)
        if image.shape != (frame.height, frame.width, 3):
            raise FrameValidationError(
                "rendered_output must be RGB HWC and match the source dimensions; "
                f"expected {(frame.height, frame.width, 3)}, got {image.shape!r}."
            )

        self._output = np.ascontiguousarray(image, dtype=np.float32).copy()
        self._timestamp = float(frame.timestamp)
        self._frame_id = int(frame.frame_id)
        self._scene_id = frame.scene_id
        self._size = frame.size
