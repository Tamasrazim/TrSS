"""End-to-end test for a small recorded-gameplay experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from trss.training.gameplay_experiment import run_gameplay_experiment


def _write_clip(path: Path, offset: int) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (32, 24)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build has no MJPG video writer.")
    try:
        for index in range(8):
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            frame[:, :, 0] = (offset + index * 13) % 255
            frame[:, :, 1] = (index * 19) % 255
            frame[:, :, 2] = (offset * 2 + index * 5) % 255
            writer.write(frame)
    finally:
        writer.release()


def test_gameplay_experiment_publishes_dataset_checkpoint_metrics_and_gallery(tmp_path):
    first = tmp_path / "ets2-route-a.avi"
    second = tmp_path / "ets2-route-b.avi"
    _write_clip(first, 11)
    _write_clip(second, 143)

    output = tmp_path / "ets2-experiment"
    report = run_gameplay_experiment(
        [first, second],
        output,
        game="Euro Truck Simulator 2",
        sample_every=2,
        max_frames_per_video=3,
        validation_fraction=0.5,
        seed=17,
        device="cpu",
        epochs=1,
        channels=8,
        blocks=1,
        patch_size=4,
        tile_size=8,
        tile_overlap=1,
        max_previews=2,
    )

    assert report["completed"] is True
    assert report["game"] == "Euro Truck Simulator 2"
    assert report["dataset"]["training_frames"] == 3
    assert report["dataset"]["validation_frames"] == 3
    assert report["training"]["epochs_completed"] == 1
    assert (output / "training/checkpoint-best.pt").is_file()
    assert (output / "dataset/capture-report.json").is_file()
    assert (output / "evaluation/evaluation-report.json").is_file()
    assert (output / "evaluation/index.html").is_file()

    training_report = json.loads(
        (output / "training/training-report.json").read_text(encoding="utf-8")
    )
    evaluation_report = json.loads(
        (output / "evaluation/evaluation-report.json").read_text(encoding="utf-8")
    )
    assert training_report["output_dir"] == str(output / "training")
    assert training_report["train_manifest"] == str(output / "dataset/train.jsonl")
    assert evaluation_report["manifest"] == str(output / "dataset/validation.jsonl")
    assert report["evaluation"]["evaluated_frames"] == 3
    assert report["evaluation"]["checkpoint_sha256"] == evaluation_report["checkpoint_sha256"]
    assert len(evaluation_report["previews"]) == 2


def test_gameplay_experiment_preserves_existing_output_without_force(tmp_path):
    first = tmp_path / "route-a.avi"
    second = tmp_path / "route-b.avi"
    _write_clip(first, 11)
    _write_clip(second, 143)
    output = tmp_path / "keep"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("must survive", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Pass --force"):
        run_gameplay_experiment([first, second], output, device="cpu")
    assert sentinel.read_text(encoding="utf-8") == "must survive"
