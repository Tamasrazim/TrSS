"""Tests for spatial checkpoint evaluation and the offline image-comparison gallery."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from trss.training.evaluate_spatial import evaluate_spatial_checkpoint
from trss.training.spatial_model import SPATIAL_MODEL_FORMAT, TrSSSpatialSR


def _image(path: Path, width: int, height: int, value: int) -> None:
    Image.fromarray(
        np.full((height, width, 3), value, dtype=np.uint8), mode="RGB"
    ).save(path)


def test_evaluator_saves_prediction_gallery_and_metric_report(tmp_path):
    _image(tmp_path / "low.png", 8, 6, 90)
    _image(tmp_path / "target.png", 16, 12, 110)
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text(
        json.dumps({
            "low_res": "low.png",
            "target": "target.png",
            "scene_id": "recording-b",
            "source_name": "ets2-road-b.mkv",
            "frame_index": 120,
        }) + "\n",
        encoding="utf-8",
    )
    model = TrSSSpatialSR(channels=8, blocks=1)
    checkpoint_path = tmp_path / "spatial.pt"
    torch.save({
        "format": SPATIAL_MODEL_FORMAT,
        "model_config": dict(model.config),
        "model_state_dict": model.state_dict(),
        "source_type": "user-recorded-gameplay-video-rgb-only",
    }, checkpoint_path)

    output = tmp_path / "evaluation"
    report = evaluate_spatial_checkpoint(
        checkpoint_path, manifest, output, device_name="cpu",
        tile_size=8, tile_overlap=1, max_previews=1,
    )
    assert report["completed"] is True
    assert report["evaluated_frames"] == 1
    assert report["metrics"]["mae"] == pytest.approx(
        report["bicubic_metrics"]["mae"], abs=1e-6
    )
    assert len(report["previews"]) == 1
    assert (output / "evaluation-report.json").is_file()
    assert (output / "index.html").is_file()
    for key in ("input", "bicubic", "prediction", "target"):
        assert (output / report["previews"][0][key]).is_file()
    saved = json.loads(
        (output / "evaluation-report.json").read_text(encoding="utf-8")
    )
    assert saved["checkpoint_sha256"] == report["checkpoint_sha256"]
    assert "Captured target" in (output / "index.html").read_text(encoding="utf-8")


def test_evaluator_rejects_temporal_checkpoint_format(tmp_path):
    _image(tmp_path / "low.png", 8, 6, 90)
    _image(tmp_path / "target.png", 16, 12, 110)
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text(
        json.dumps({"low_res": "low.png", "target": "target.png", "scene_id": "scene"}) + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "temporal.pt"
    torch.save({"format": "trss-temporal-sr-v1"}, checkpoint)
    with pytest.raises(ValueError, match="temporal checkpoints are incompatible"):
        evaluate_spatial_checkpoint(
            checkpoint, manifest, tmp_path / "out", device_name="cpu",
            tile_size=8, tile_overlap=1,
        )
