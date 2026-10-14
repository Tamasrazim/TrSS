"""Tests for the separate RGB-only spatial model and gameplay-pair trainer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from trss.training.spatial_data import TrSSSpatialFrameDataset
from trss.training.spatial_model import (
    SPATIAL_MODEL_FORMAT,
    TrSSSpatialSR,
    compose_spatial_prediction,
)
from trss.training.train_spatial import train_spatial


def _write_rgb(path: Path, width: int, height: int, value: int) -> None:
    Image.fromarray(
        np.full((height, width, 3), value, dtype=np.uint8), mode="RGB"
    ).save(path)


def _write_manifest(root: Path, name: str, scene: str, offset: int) -> Path:
    low_name = f"{scene}-low.png"
    target_name = f"{scene}-target.png"
    _write_rgb(root / low_name, 8, 6, 70 + offset)
    _write_rgb(root / target_name, 16, 12, 90 + offset)
    manifest = root / name
    manifest.write_text(
        json.dumps({
            "low_res": low_name,
            "target": target_name,
            "scene_id": scene,
            "capture_id": scene,
            "auxiliary_buffers": "not captured; use spatial-only model",
        }) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_spatial_model_starts_at_bicubic_baseline_and_backpropagates():
    model = TrSSSpatialSR(channels=8, blocks=1)
    low = torch.rand(1, 3, 6, 8)
    raw = model(low)
    output = compose_spatial_prediction(raw, low)
    baseline = torch.nn.functional.interpolate(
        low, scale_factor=2, mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)
    assert raw.shape == (1, 3, 12, 16)
    assert torch.allclose(output, baseline, atol=1e-6)
    output.mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_spatial_dataset_checks_exact_two_x_pair_contract(tmp_path):
    manifest = _write_manifest(tmp_path, "frames.jsonl", "road-a", 0)
    sample = TrSSSpatialFrameDataset(manifest)[0]
    assert sample["low_res"].shape == (3, 6, 8)
    assert sample["target"].shape == (3, 12, 16)


def test_spatial_training_writes_checkpoint_and_compares_bicubic(tmp_path):
    train_manifest = _write_manifest(tmp_path, "train.jsonl", "capture-a", 0)
    validation_manifest = _write_manifest(tmp_path, "validation.jsonl", "capture-b", 20)
    output = tmp_path / "run"
    report = train_spatial(
        train_manifest, validation_manifest, output,
        device_name="cpu", epochs=1, channels=8, blocks=1,
        patch_size=4, tile_size=8, tile_overlap=1, seed=7,
    )
    assert report["completed"] is True
    assert report["model_format"] == SPATIAL_MODEL_FORMAT
    assert report["training_scenes"] == ["capture-a"]
    assert report["validation_scenes"] == ["capture-b"]
    assert (output / "checkpoint-best.pt").is_file()
    assert (output / "checkpoint-last.pt").is_file()
    saved = torch.load(
        output / "checkpoint-best.pt", map_location="cpu", weights_only=True
    )
    assert saved["format"] == SPATIAL_MODEL_FORMAT
    assert saved["source_type"] == "user-recorded-gameplay-video-rgb-only"
    assert (output / "training-report.json").is_file()


def test_spatial_training_rejects_scene_leakage(tmp_path):
    manifest = _write_manifest(tmp_path, "shared.jsonl", "same-capture", 0)
    with pytest.raises(ValueError, match="share scene_id"):
        train_spatial(
            manifest, manifest, tmp_path / "never-created",
            device_name="cpu", epochs=1, channels=8, blocks=1,
            patch_size=4, tile_size=8, tile_overlap=1,
        )
