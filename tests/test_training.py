"""CPU tests for the first-party TrSS training model and dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from trss.training import train as training
from trss.training.data import TrSSFrameDataset
from trss.training.model import TrSSTemporalSR, compose_prediction, model_size_report


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
    model = TrSSTemporalSR(channels=16, blocks=2, context_blocks=1)
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
    model = TrSSTemporalSR(channels=16, blocks=2, context_blocks=0)
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
    sample = TrSSFrameDataset(manifest)[0]
    assert sample["features"].shape == (10, 4, 6)
    assert sample["target"].shape == (3, 8, 12)
    assert sample["history"].shape == (3, 8, 12)
    assert sample["history_valid"].shape == (1, 8, 12)
    assert torch.count_nonzero(sample["history"]) == 0
    assert torch.count_nonzero(sample["history_valid"]) == 0


def test_random_flip_reflects_motion_vectors_with_the_images(monkeypatch):
    batch = {
        "features": torch.zeros(2, 10, 4, 6),
        "low_res": torch.zeros(2, 3, 4, 6),
        "target": torch.zeros(2, 3, 8, 12),
        "history": torch.zeros(2, 3, 8, 12),
        "history_valid": torch.ones(2, 1, 8, 12),
    }
    batch["features"][:, 4] = 1.0
    batch["features"][:, 5] = 2.0
    randint_values = iter((0, 0, 1, 2))
    random_values = iter((0.0, 0.0, 1.0, 1.0))
    monkeypatch.setattr(training.random, "randint", lambda low, high: next(randint_values))
    monkeypatch.setattr(training.random, "random", lambda: next(random_values))

    cropped = training._crop_batch(batch, 2, random_crop=True)

    assert cropped["features"].shape == (2, 10, 2, 2)
    assert torch.all(cropped["features"][0, 4] == -1.0)
    assert torch.all(cropped["features"][0, 5] == -2.0)
    assert torch.all(cropped["features"][1, 4] == 1.0)
    assert torch.all(cropped["features"][1, 5] == 2.0)


def test_validation_tiles_cover_full_frame_edges():
    assert training._tile_starts(4, 4) == [0]
    assert training._tile_starts(10, 4) == [0, 4, 6]
    assert training._tile_starts(11, 4) == [0, 4, 7]


def test_split_overlap_rejects_the_same_reference_frame(tmp_path):
    manifest = _make_dataset_files(tmp_path)
    train_set = TrSSFrameDataset(manifest)
    validation_set = TrSSFrameDataset(manifest)
    with pytest.raises(ValueError, match="reuse at least one exact target frame"):
        training._check_split_overlap(train_set, validation_set)


def test_preflight_validates_dataset_without_gpu_or_output_directory(
    tmp_path, monkeypatch, capsys
):
    manifest = _make_dataset_files(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--train-manifest", str(manifest),
            "--preflight-only",
            "--device", "xpu",
            "--batch-size", "1",
        ],
    )

    assert training.main() == 0
    # Preflight is read-only: it should not demand an XPU or create model files.
    summary = json.loads(capsys.readouterr().out)
    assert summary["preflight"] == "passed"
    assert summary["training"]["frames"] == 1
    assert summary["training"]["low_resolution_dimensions"] == {"6x4": 1}
    assert summary["training"]["all_tensors_finite"] is True


def test_validation_reports_neural_and_bicubic_image_metrics(tmp_path):
    manifest = _make_dataset_files(tmp_path)
    dataset = TrSSFrameDataset(manifest)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = TrSSTemporalSR(channels=8, blocks=1, context_blocks=0)

    metrics = training._run_epoch(
        model,
        loader,
        torch.device("cpu"),
        optimizer=None,
        patch_size=4,
        amp=False,
    )

    assert set(metrics) == {
        "loss", "mae", "psnr_db", "bicubic_mae", "bicubic_psnr_db"
    }
    # The fresh output head is deliberately neutral, so the model initially is
    # exactly the bicubic baseline on reset frames.
    assert metrics["mae"] == pytest.approx(metrics["bicubic_mae"], abs=1e-6)
    assert metrics["psnr_db"] == pytest.approx(metrics["bicubic_psnr_db"], abs=1e-6)


def test_training_can_resume_from_last_checkpoint(tmp_path, monkeypatch):
    manifest = _make_dataset_files(tmp_path)
    output = tmp_path / "weights"
    args = [
        "train",
        "--train-manifest", str(manifest),
        "--output-dir", str(output),
        "--device", "cpu",
        "--channels", "8",
        "--blocks", "1",
        "--context-blocks", "0",
        "--patch-size", "4",
        "--epochs", "2",
        "--seed", "9",
    ]
    monkeypatch.setattr("sys.argv", args)

    # Simulate a stop after epoch one has safely written checkpoint-last.pt.
    real_run_epoch = training._run_epoch
    calls = {"count": 0}

    def interrupt_before_second_epoch(*run_args, **run_kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated interruption")
        return real_run_epoch(*run_args, **run_kwargs)

    monkeypatch.setattr(training, "_run_epoch", interrupt_before_second_epoch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        training.main()

    checkpoint_path = output / "checkpoint-last.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert "optimizer_state_dict" in checkpoint

    monkeypatch.setattr(training, "_run_epoch", real_run_epoch)
    monkeypatch.setattr(
        "sys.argv",
        args + ["--resume", str(checkpoint_path)],
    )
    assert training.main() == 0

    report = json.loads((output / "training-report.json").read_text(encoding="utf-8"))
    assert [row["epoch"] for row in report["history"]] == [1, 2]
    assert (output / "checkpoint-best.pt").is_file()
