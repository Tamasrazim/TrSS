"""Paired-frame dataset for the first-party TRSS training architecture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.nn import functional as F
from torch.utils.data import Dataset

from .manifest import load_manifest


def _read_rgb(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            result = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Could not read RGB image '{path}': {exc}") from exc
    return np.ascontiguousarray(result)


def _read_numeric_npy(path: Path, field: str) -> np.ndarray:
    if path.suffix.lower() != ".npy":
        raise ValueError(f"'{field}' must be a .npy array so its values remain explicit.")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load '{field}' array '{path}': {exc}") from exc
    if array.dtype.kind not in "uif":
        raise ValueError(f"'{field}' must contain real numeric values.")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"'{field}' contains NaN or infinite values.")
    return array


def _read_depth(path: Path, height: int, width: int) -> np.ndarray:
    depth = _read_numeric_npy(path, "depth")
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.shape != (height, width):
        raise ValueError(
            f"Depth shape {depth.shape} must match low-resolution color {(height, width)}."
        )
    if depth.min(initial=0.0) < 0.0 or depth.max(initial=0.0) > 1.0:
        raise ValueError("Depth must be pre-normalized to [0, 1] using a documented encoding.")
    return depth[None, ...]


def _read_motion(
    path: Path,
    height: int,
    width: int,
    motion_scale_px: float,
) -> np.ndarray:
    motion = _read_numeric_npy(path, "motion")
    if motion.shape != (height, width, 2):
        raise ValueError(
            f"Motion shape {motion.shape} must be {(height, width, 2)} in HWC order."
        )
    # Motion is stored in pixels per frame. Bound the model input without changing sign.
    normalized = np.clip(motion / motion_scale_px, -1.0, 1.0)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1))


def _read_mask(path: Path, height: int, width: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = _read_numeric_npy(path, "history_valid")
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
    else:
        try:
            with Image.open(path) as image:
                mask = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not read history-validity mask '{path}': {exc}") from exc
    if mask.shape != (height, width):
        raise ValueError(
            f"History-validity mask shape {mask.shape} must match target {(height, width)}."
        )
    if (
        not np.isfinite(mask).all()
        or mask.min(initial=0.0) < 0.0
        or mask.max(initial=0.0) > 1.0
    ):
        raise ValueError("History-validity mask must contain finite values in [0, 1].")
    return np.ascontiguousarray(mask[None, ...], dtype=np.float32)


class TRSSFrameDataset(Dataset[dict[str, torch.Tensor]]):
    """Load aligned low/high-resolution frames and real renderer auxiliary buffers."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        motion_scale_px: float = 64.0,
    ) -> None:
        if not np.isfinite(motion_scale_px) or motion_scale_px <= 0:
            raise ValueError("motion_scale_px must be a finite number greater than zero.")
        self.rows = load_manifest(manifest_path)
        self.motion_scale_px = float(motion_scale_px)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row: dict[str, Any] = self.rows[index]
        low_rgb = _read_rgb(row["low_res"])
        target_rgb = _read_rgb(row["target"])
        height, width, _ = low_rgb.shape
        target_height, target_width, _ = target_rgb.shape
        if (target_height, target_width) != (height * 2, width * 2):
            raise ValueError(
                "Every target must be exactly 2x the low-resolution input: "
                f"input={(height, width)}, target={(target_height, target_width)}."
            )

        depth = _read_depth(row["depth"], height, width)
        motion = _read_motion(
            row["motion"], height, width, self.motion_scale_px
        )

        if row["history"] is None:
            # Explicit nulls mean first frame/camera cut: history is not valid.
            history_hr = np.zeros(
                (3, target_height, target_width), dtype=np.float32
            )
            history_valid_hr = np.zeros(
                (1, target_height, target_width), dtype=np.float32
            )
        else:
            history_rgb = _read_rgb(row["history"])
            if history_rgb.shape[:2] != (target_height, target_width):
                raise ValueError("Warped history must align with the high-resolution target.")
            valid = _read_mask(
                row["history_valid"], target_height, target_width
            )
            history_hr = np.ascontiguousarray(history_rgb.transpose(2, 0, 1))
            history_valid_hr = valid

        current = torch.from_numpy(
            np.ascontiguousarray(low_rgb.transpose(2, 0, 1))
        )
        depth_tensor = torch.from_numpy(depth)
        motion_tensor = torch.from_numpy(motion)
        history_tensor = torch.from_numpy(history_hr)
        valid_tensor = torch.from_numpy(history_valid_hr)

        # The network sees reduced history features; final blending uses full-size history.
        history_lr = F.interpolate(
            history_tensor.unsqueeze(0), size=(height, width), mode="area"
        ).squeeze(0)
        valid_lr = F.interpolate(
            valid_tensor.unsqueeze(0), size=(height, width), mode="area"
        ).squeeze(0)
        features = torch.cat(
            (current, depth_tensor, motion_tensor, history_lr, valid_lr), dim=0
        )
        if features.shape[0] != 10:
            raise RuntimeError(
                f"Internal feature assembly error: got {features.shape[0]} channels."
            )

        target = torch.from_numpy(
            np.ascontiguousarray(target_rgb.transpose(2, 0, 1))
        )
        return {
            "features": features,
            "low_res": current,
            "target": target,
            "history": history_tensor,
            "history_valid": valid_tensor,
        }
