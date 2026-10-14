"""Tests for full-frame checkpoint evaluation and baseline comparison."""

from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")

from trss.training.evaluate_checkpoint import evaluate_checkpoint
from trss.training.model import MODEL_FORMAT, TrSSTemporalSR
from trss.training.prepare_pilot import prepare_pilot_dataset


def test_evaluator_writes_full_size_predictions_metrics_and_offline_gallery(tmp_path):
    pilot = tmp_path / "pilot"
    prepared = prepare_pilot_dataset(
        pilot,
        scenes=4,
        frames_per_scene=2,
        width=32,
        height=20,
        seed=7021,
        validation_fraction=0.5,
        max_review_frames=8,
    )
    assert prepared["passed"] is True, prepared

    model = TrSSTemporalSR(channels=8, blocks=1, context_blocks=0)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "format": MODEL_FORMAT,
            "config": model.config,
            "state_dict": model.state_dict(),
            "epoch": 1,
        },
        checkpoint,
    )

    output = tmp_path / "evaluation"
    report = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        manifest_path=pilot / "validation.jsonl",
        output_dir=output,
        device_name="cpu",
        tile_size=16,
        tile_overlap=2,
        max_frames=3,
    )

    assert report["evaluation_completed"] is True
    assert report["source_type"] == "synthetic_controlled_3d_scene"
    assert report["actual_game_capture"] is False
    assert report["manifest_frame_count"] == 4
    assert report["evaluated_frame_count"] == 3
    assert report["sampling"].startswith("deterministic evenly-spaced subset")
    assert report["checkpoint_epoch"] == 1
    assert len(report["checkpoint_sha256"]) == 64
    assert math.isfinite(report["metrics"]["mae"])
    assert math.isfinite(report["metrics"]["bicubic_psnr_db"])
    assert report["metrics"]["frames"] == 3
    assert set(report["per_scene_metrics"])

    saved = json.loads(
        (output / "evaluation-report.json").read_text(encoding="utf-8")
    )
    assert saved["format"] == "trss-checkpoint-evaluation-v1"
    assert saved["output_dir"] == str(output.resolve())
    assert len(saved["frames"]) == 3
    assert (output / "index.html").is_file()
    gallery = (output / "index.html").read_text(encoding="utf-8")
    assert "Target reference" not in gallery  # The labels live in each comparison image.
    assert "synthetic-scene result" in gallery

    for frame in report["frames"]:
        from PIL import Image

        prediction_path = output / frame["prediction"]
        comparison_path = output / frame["comparison"]
        assert prediction_path.is_file()
        assert comparison_path.is_file()
        with Image.open(prediction_path) as prediction:
            assert prediction.size == (64, 40)
            assert prediction.mode == "RGB"
        with Image.open(comparison_path) as comparison:
            assert comparison.width == 4 * 480
            assert comparison.height > 270


def test_evaluator_refuses_to_replace_existing_output_without_force(tmp_path):
    output = tmp_path / "already-there"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Pass --force"):
        evaluate_checkpoint(
            checkpoint_path=tmp_path / "missing-checkpoint.pt",
            manifest_path=tmp_path / "missing-manifest.jsonl",
            output_dir=output,
            device_name="cpu",
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_evaluator_rejects_invalid_tiling_before_creating_output(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    model = TrSSTemporalSR(channels=8, blocks=1, context_blocks=0)
    torch.save(
        {"format": MODEL_FORMAT, "config": model.config, "state_dict": model.state_dict()},
        checkpoint,
    )
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text("", encoding="utf-8")
    output = tmp_path / "evaluation"

    with pytest.raises(ValueError, match="tile_size"):
        evaluate_checkpoint(
            checkpoint_path=checkpoint,
            manifest_path=manifest,
            output_dir=output,
            tile_size=16,
            tile_overlap=8,
        )
    assert not output.exists()
