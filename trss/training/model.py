"""Original, scale-2 TrSS temporal reconstruction model.

This is a first-party trainable reference architecture, not the vendored research
graph used by the experimental xmx-frame command. Its weights must be trained.
"""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn
from torch.nn import functional as F

INPUT_CHANNELS = 10
OUTPUT_CHANNELS = 4
UPSCALE_FACTOR = 2
MODEL_FORMAT = "trss-temporal-sr-v1"


class EfficientResidualBlock(nn.Module):
    """Mix nearby features with separable convolutions to limit per-frame work."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        expanded = channels * expansion
        self.expand = nn.Conv2d(channels, expanded, kernel_size=1)
        self.spatial = nn.Conv2d(
            expanded, expanded, kernel_size=3, padding=1, groups=expanded
        )
        self.project = nn.Conv2d(expanded, channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Keep the skip path simple so the block can learn a small correction.
        correction = self.expand(value)
        correction = self.activation(correction)
        correction = self.spatial(correction)
        correction = self.activation(correction)
        correction = self.project(correction)
        return value + correction * 0.2


class TrSSTemporalSR(nn.Module):
    """Reconstruct a 2x RGB frame from real color/depth/motion/history inputs.

    Input channel contract, all NCHW and spatially aligned at low resolution:
      0..2  current display-encoded RGB, normalized to [0, 1]
      3     depth normalized to [0, 1] using the dataset's declared convention
      4..5  horizontal/vertical motion in pixels per frame, divided by motion scale
      6..8  previous output warped into the current view, downsampled to input size
      9     history-validity mask, where 0 means do not trust prior history

    The model predicts three spatial residual channels and one temporal blend gate
    at twice the input width and height. Motion reprojection and the validity mask
    must come from the caller/capture pipeline; the network does not invent them.
    """

    def __init__(
        self,
        *,
        channels: int = 48,
        blocks: int = 10,
        context_blocks: int = 4,
    ) -> None:
        super().__init__()
        if channels < 8:
            raise ValueError("channels must be at least 8.")
        if blocks < 1:
            raise ValueError("blocks must be at least 1.")
        if context_blocks < 0:
            raise ValueError("context_blocks cannot be negative.")

        self.config = {
            "channels": int(channels),
            "blocks": int(blocks),
            "context_blocks": int(context_blocks),
            "input_channels": INPUT_CHANNELS,
            "output_channels": OUTPUT_CHANNELS,
            "upscale_factor": UPSCALE_FACTOR,
            "format": MODEL_FORMAT,
        }

        self.stem = nn.Conv2d(INPUT_CHANNELS, channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *(EfficientResidualBlock(channels) for _ in range(blocks))
        )

        # The low-resolution context branch sees a wider area at half the cost.
        self.context_down = nn.Conv2d(
            channels, channels * 2, kernel_size=3, stride=2, padding=1
        )
        self.context_body = nn.Sequential(
            *(EfficientResidualBlock(channels * 2) for _ in range(context_blocks))
        )
        self.context_project = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.fuse = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # PixelShuffle converts the feature map to a 2x, four-channel output head.
        self.head = nn.Conv2d(
            channels,
            OUTPUT_CHANNELS * (UPSCALE_FACTOR ** 2),
            kernel_size=3,
            padding=1,
        )
        self.shuffle = nn.PixelShuffle(UPSCALE_FACTOR)
        self._initialize_output_head()

    def _initialize_output_head(self) -> None:
        # Start as a neutral upsampler rather than adding random image noise.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        gate_start = OUTPUT_CHANNELS - 1
        subpixel_count = UPSCALE_FACTOR ** 2
        with torch.no_grad():
            self.head.bias[
                gate_start * subpixel_count : (gate_start + 1) * subpixel_count
            ].fill_(-2.0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("Model input must be an NCHW rank-four tensor.")
        if features.shape[1] != INPUT_CHANNELS:
            raise ValueError(
                f"Expected {INPUT_CHANNELS} feature channels; got {features.shape[1]}."
            )
        if features.shape[-2] < 2 or features.shape[-1] < 2:
            raise ValueError("Input height and width must each be at least 2.")

        base = self.stem(features)
        local = self.body(base)

        context = self.context_down(local)
        context = self.context_body(context)
        context = F.interpolate(
            context, size=local.shape[-2:], mode="bilinear", align_corners=False
        )
        local = self.fuse(local + self.context_project(context) + base)
        return self.shuffle(self.head(F.gelu(local)))


def compose_prediction(
    raw_head: torch.Tensor,
    low_res_rgb: torch.Tensor,
    history_rgb: torch.Tensor,
    history_valid: torch.Tensor,
    *,
    residual_limit: float = 0.25,
) -> torch.Tensor:
    """Compose the network head into an RGB frame using explicit valid history."""
    if raw_head.ndim != 4 or raw_head.shape[1] != OUTPUT_CHANNELS:
        raise ValueError("raw_head must have shape [N, 4, 2H, 2W].")
    if low_res_rgb.ndim != 4 or low_res_rgb.shape[1] != 3:
        raise ValueError("low_res_rgb must have shape [N, 3, H, W].")

    batch, _, height, width = low_res_rgb.shape
    expected_head = (batch, OUTPUT_CHANNELS, height * 2, width * 2)
    expected_rgb = (batch, 3, height * 2, width * 2)
    expected_mask = (batch, 1, height * 2, width * 2)
    if tuple(raw_head.shape) != expected_head:
        raise ValueError(f"raw_head must have shape {expected_head}.")
    if tuple(history_rgb.shape) != expected_rgb:
        raise ValueError(f"history_rgb must have shape {expected_rgb}.")
    if tuple(history_valid.shape) != expected_mask:
        raise ValueError(f"history_valid must have shape {expected_mask}.")

    # Bicubic is only the base image; the learned residual supplies reconstruction.
    base = F.interpolate(
        low_res_rgb.float(),
        scale_factor=UPSCALE_FACTOR,
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 1.0)
    residual = torch.tanh(raw_head[:, :3].float()) * float(residual_limit)
    spatial = (base + residual).clamp(0.0, 1.0)

    # A disocclusion/reset mask can veto temporal blending even when the gate is high.
    gate = torch.sigmoid(raw_head[:, 3:4].float())
    gate = gate * history_valid.float().clamp(0.0, 1.0)
    previous = history_rgb.float().clamp(0.0, 1.0)
    return (spatial * (1.0 - gate) + previous * gate).clamp(0.0, 1.0)


def model_size_report(model: nn.Module) -> dict[str, int]:
    """Report raw tensor storage estimates; checkpoint files add small metadata overhead."""
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "parameters": int(parameter_count),
        "fp16_weight_bytes": int(parameter_count * 2),
        "fp32_weight_bytes": int(parameter_count * 4),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an TrSS model configuration.")
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--context-blocks", type=int, default=4)
    args = parser.parse_args()
    model = TrSSTemporalSR(
        channels=args.channels,
        blocks=args.blocks,
        context_blocks=args.context_blocks,
    )
    print(json.dumps({"config": model.config, **model_size_report(model)}, indent=2))


if __name__ == "__main__":
    _main()
