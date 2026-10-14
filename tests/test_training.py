"""CPU tests for the first-party TRSS training model and dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from trss.training.data import TRSSFrameDataset
from trss.training.model import TRSSTemporalSR, compose_prediction, model_size_report


def _write_rgb(path: Path, height: int, width: int, value: int) -> None:
    Image.fromarray(
        np.full((height, width, 3), value, dtype=np.uint8), mode="RGB"
    ).save(path)


def _make_dataset_files(tmp_path: Path) -> Path:
    _write_rgb(tmp_path / "low.png", 4, 6, 90)
    _write_rgb(tmp_path / "target.png", 8, 12, 100)
    np.save(tmp_path / "depth.npy", np.full((4, 6), 0.5, dtype=np.float32))
    np.save(tmp_path / "motion.npy", np.zeros((4, 6, 2), dtype=np.float32))
    row = {
        "low_res": "low.png",
        "target": "target.png",
        "depth": "depth.npy",
        "motion": "motion.npy",
        "history": None,
        "history_valid": None,
    }
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


def test_model_output_is_double_size_and_starts_as_neutral_upscale():
    model = TRSSTemporalSR(channels=16, blocks=2, context_blocks=1)
    features = torch.rand(1, 10, 8, 10)
    current = torch.rand(1, 3, 8, 10)
    history = torch.rand(1, 3, 16, 20)
    valid = torch.zeros(1, 1, 16, 20)
    raw = model(features)
    output = compose_prediction(raw, current, history, valid)
    baseline = torch.nn.functional.interpolate(
        current, scale_factor=2, mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)

    assert raw.shape == (1, 4, 16, 20)
    assert output.shape == (1, 3, 16, 20)
    assert torch.allclose(output, baseline, atol=1e-6)


def test_model_rejects_wrong_feature_count_and_backpropagates():
    model = TRSSTemporalSR(channels=16, blocks=2, context_blocks=0)
    with pytest.raises(ValueError, match="feature channels"):
        model(torch.rand(1, 9, 8, 8))

    raw = model(torch.rand(1, 10, 8, 8))
    raw.square().mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    report = model_size_report(model)
    assert report["fp16_weight_bytes"] == report["parameters"] * 2
    assert report["fp32_weight_bytes"] == report["parameters"] * 4


def test_frame_dataset_builds_ten_channels_without_faking_temporal_history(tmp_path):
    manifest = _make_dataset_files(tmp_path)
    sample = TRSSFrameDataset(manifest)[0]
    assert sample["features"].shape == (10, 4, 6)
    assert sample["target"].shape == (3, 8, 12)
    assert sample["history"].shape == (3, 8, 12)
    assert sample["history_valid"].shape == (1, 8, 12)
    assert torch.count_nonzero(sample["history"]) == 0
    assert torch.count_nonzero(sample["history_valid"]) == 0
