"""RGB-only frame dataset for the spatial TrSS baseline.

This data contract intentionally does not ask for fake depth, motion, or history.
It is separate from TrSSFrameDataset, which requires genuine renderer buffers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


def load_spatial_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL manifest and resolve image paths relative to that manifest."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Spatial manifest not found: {manifest_path}")
    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{manifest_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{manifest_path}:{line_number}: each line must be an object.")
            for name in ("low_res", "target", "scene_id"):
                if not isinstance(row.get(name), str) or not row[name].strip():
                    raise ValueError(
                        f"{manifest_path}:{line_number}: '{name}' must be a non-empty string."
                    )
            resolved = dict(row)
            for name in ("low_res", "target"):
                candidate = Path(row[name]).expanduser()
                if not candidate.is_absolute():
                    candidate = manifest_path.parent / candidate
                candidate = candidate.resolve()
                if not candidate.is_file():
                    raise FileNotFoundError(
                        f"{manifest_path}:{line_number}: '{name}' file does not exist: {candidate}"
                    )
                resolved[name] = candidate
            resolved["scene_id"] = row["scene_id"]
            resolved["line"] = line_number
            rows.append(resolved)
    if not rows:
        raise ValueError(f"Spatial manifest contains no usable rows: {manifest_path}")
    return rows


def _read_rgb(path: Path) -> torch.Tensor:
    try:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Could not read RGB image '{path}': {exc}") from exc
    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))


class TrSSSpatialFrameDataset(Dataset[dict[str, torch.Tensor]]):
    """Read half-resolution RGB input and its aligned 2x RGB target."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.rows = load_spatial_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        low = _read_rgb(row["low_res"])
        target = _read_rgb(row["target"])
        _, height, width = low.shape
        if tuple(target.shape[1:]) != (height * 2, width * 2):
            raise ValueError(
                "Spatial target must be exactly twice the input dimensions: "
                f"input={(height, width)}, target={tuple(target.shape[1:])}."
            )
        return {"low_res": low, "target": target}
