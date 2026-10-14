"""Small RGB-only 2x super-resolution model, separate from temporal TrSS."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import EfficientResidualBlock

SPATIAL_MODEL_FORMAT = "trss-spatial-sr-v1"


class TrSSSpatialSR(nn.Module):
    """Learn an RGB residual over bicubic interpolation, with no auxiliary inputs."""

    def __init__(self, *, channels: int = 32, blocks: int = 6) -> None:
        super().__init__()
        if channels < 8:
            raise ValueError("channels must be at least 8.")
        if blocks < 1:
            raise ValueError("blocks must be at least 1.")
        self.config = {
            "format": SPATIAL_MODEL_FORMAT,
            "input_channels": 3,
            "output_channels": 3,
            "upscale_factor": 2,
            "channels": int(channels),
            "blocks": int(blocks),
        }
        self.stem = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(*(EfficientResidualBlock(channels) for _ in range(blocks)))
        self.fuse = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.head = nn.Conv2d(channels, 3 * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        # Start at the bicubic baseline; early outputs should not contain random noise.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, low_res_rgb: torch.Tensor) -> torch.Tensor:
        if low_res_rgb.ndim != 4 or low_res_rgb.shape[1] != 3:
            raise ValueError("Spatial model input must be RGB NCHW.")
        if low_res_rgb.shape[-2] < 2 or low_res_rgb.shape[-1] < 2:
            raise ValueError("Input height and width must each be at least 2.")
        base = self.stem(low_res_rgb)
        features = self.fuse(self.body(base) + base)
        return self.shuffle(self.head(F.gelu(features)))


def compose_spatial_prediction(
    raw_residual: torch.Tensor,
    low_res_rgb: torch.Tensor,
    *,
    residual_limit: float = 0.25,
) -> torch.Tensor:
    """Add a bounded learned residual to the deterministic bicubic baseline."""
    if raw_residual.ndim != 4 or raw_residual.shape[1] != 3:
        raise ValueError("raw_residual must be a rank-four RGB tensor.")
    if low_res_rgb.ndim != 4 or low_res_rgb.shape[1] != 3:
        raise ValueError("low_res_rgb must be a rank-four RGB tensor.")
    expected = (
        low_res_rgb.shape[0],
        3,
        low_res_rgb.shape[-2] * 2,
        low_res_rgb.shape[-1] * 2,
    )
    if tuple(raw_residual.shape) != expected:
        raise ValueError(f"raw_residual must have shape {expected}.")
    if not 0.0 < residual_limit <= 1.0:
        raise ValueError("residual_limit must be in the range (0, 1].")
    base = F.interpolate(
        low_res_rgb.float(), scale_factor=2, mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)
    residual = torch.tanh(raw_residual.float()) * float(residual_limit)
    return (base + residual).clamp(0.0, 1.0)
