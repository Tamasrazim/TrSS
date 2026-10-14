"""Image/tensor conversions shared by the TrSS inference engine."""

from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image

Layout = Literal["NCHW", "NHWC"]
ChannelOrder = Literal["RGB", "BGR"]


def parse_value_range(value: str) -> tuple[float, float]:
    """Parse an inclusive numeric range such as '0:1' or '-1:1'."""
    try:
        low_text, high_text = value.split(":", maxsplit=1)
        low, high = float(low_text), float(high_text)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid range {value!r}; expected MIN:MAX, for example 0:1."
        ) from exc

    if not np.isfinite(low) or not np.isfinite(high) or not low < high:
        raise ValueError(f"Invalid range {value!r}; MIN must be finite and less than MAX.")
    return low, high


def image_to_tensor(
    image: Image.Image,
    *,
    layout: Layout,
    width: int,
    height: int,
    value_range: tuple[float, float] = (0.0, 1.0),
    channel_order: ChannelOrder = "RGB",
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    """Convert an image into a batch-one RGB tensor with a declared value range."""
    if layout not in ("NCHW", "NHWC"):
        raise ValueError(f"Unsupported image layout: {layout!r}")
    if channel_order not in ("RGB", "BGR"):
        raise ValueError(f"Unsupported channel order: {channel_order!r}")
    if width < 1 or height < 1:
        raise ValueError("Tensor width and height must be positive.")
    if (
        len(value_range) != 2
        or not np.isfinite(value_range[0])
        or not np.isfinite(value_range[1])
        or value_range[0] >= value_range[1]
    ):
        raise ValueError("The value range must contain two finite values in ascending order.")

    low, high = value_range
    rgb = image.convert("RGB")
    if rgb.size != (width, height):
        rgb = rgb.resize((width, height), Image.Resampling.LANCZOS)

    pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    if channel_order == "BGR":
        pixels = pixels[:, :, ::-1]
    pixels = pixels * (high - low) + low

    if layout == "NCHW":
        pixels = np.transpose(pixels, (2, 0, 1))
    return np.ascontiguousarray(pixels[np.newaxis, ...], dtype=dtype)


def tensor_to_image(
    tensor: np.ndarray,
    *,
    layout: Layout,
    value_range: tuple[float, float] = (0.0, 1.0),
    channel_order: ChannelOrder = "RGB",
) -> Image.Image:
    """Convert a batch-one model output into an RGB image."""
    output = np.asarray(tensor)

    if np.issubdtype(output.dtype, np.complexfloating):
        raise ValueError("Complex-valued model outputs require an explicit conversion.")
    if not np.issubdtype(output.dtype, np.number):
        raise ValueError("Model output must contain numeric values.")
    if output.ndim != 4 or output.shape[0] != 1:
        raise ValueError(
            f"Expected one rank-four output tensor, got shape {output.shape!r}."
        )

    if layout == "NCHW":
        output = np.transpose(output[0], (1, 2, 0))
    elif layout == "NHWC":
        output = output[0]
    else:
        raise ValueError(f"Unsupported image layout: {layout!r}")

    if output.ndim != 3 or output.shape[2] not in (1, 3, 4):
        raise ValueError(
            "Expected a one-, three-, or four-channel image output, "
            f"got shape {output.shape!r}."
        )

    if channel_order not in ("RGB", "BGR"):
        raise ValueError(f"Unsupported channel order: {channel_order!r}")
    low, high = value_range
    if (
        not np.isfinite(low)
        or not np.isfinite(high)
        or low >= high
    ):
        raise ValueError("The value range must contain two finite values in ascending order.")

    # NaN and infinities must not turn into random pixels in the saved image.
    output = np.nan_to_num(output.astype(np.float32), nan=low, posinf=high, neginf=low)
    output = np.clip((output - low) / (high - low), 0.0, 1.0)
    output = np.rint(output * 255.0).astype(np.uint8)

    if output.shape[2] == 1:
        output = np.repeat(output, 3, axis=2)
    elif output.shape[2] == 4:
        output = output[:, :, :3]

    if channel_order == "BGR":
        output = output[:, :, ::-1]
    return Image.fromarray(np.ascontiguousarray(output))
