"""Frame contracts, temporal state, and model-specific neural-rendering adapters."""

from .frame import FramePacket, FrameValidationError, TemporalHistory
from .hnet import (
    HNetContractError,
    HNetControls,
    HNetFrameEngine,
    HNetFrameResult,
    NetworkGeometry,
    compose_head,
    deterministic_noise,
    make_features,
    network_geometry,
    scaled_color,
)

__all__ = [
    "FramePacket",
    "FrameValidationError",
    "TemporalHistory",
    "HNetContractError",
    "HNetControls",
    "HNetFrameEngine",
    "HNetFrameResult",
    "NetworkGeometry",
    "compose_head",
    "deterministic_noise",
    "make_features",
    "network_geometry",
    "scaled_color",
]
