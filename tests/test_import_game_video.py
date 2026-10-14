"""Tests for importing ordinary video files into the local TrSS dataset."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2")

from trss.training.import_game_video import import_game_videos


def _write_clip(path: Path, offset: int) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (32, 24)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build has no MJPG video writer.")
    try:
        for index in range(8):
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            frame[:, :, 0] = (offset + index * 9) % 255
            frame[:, :, 1] = (index * 17) % 255
            writer.write(frame)
    finally:
        writer.release()


def test_import_builds_aligned_pairs_and_splits_whole_recordings(tmp_path):
    first = tmp_path / "ets2-road-a.avi"
    second = tmp_path / "ets2-road-b.avi"
    _write_clip(first, 10)
    _write_clip(second, 120)

    output = tmp_path / "imported"
    report = import_game_videos(
        [first, second], output, sample_every=2, max_frames_per_video=3, seed=12
    )

    train_rows = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validation_rows = [
        json.loads(line)
        for line in (output / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert report["completed"] is True
    assert report["training_frames"] == 3
    assert report["validation_frames"] == 3
    assert {row["scene_id"] for row in train_rows}.isdisjoint(
        {row["scene_id"] for row in validation_rows}
    )
    for row in train_rows + validation_rows:
        with Image.open(output / row["low_res"]) as low, Image.open(output / row["target"]) as target:
            assert target.size == (low.width * 2, low.height * 2)
        assert row["auxiliary_buffers"] == "not captured; use spatial-only model"
    assert (output / "capture-report.json").is_file()
    assert (output / "provenance.json").is_file()


def test_import_refuses_to_replace_output_without_force(tmp_path):
    first = tmp_path / "first.avi"
    second = tmp_path / "second.avi"
    _write_clip(first, 1)
    _write_clip(second, 2)
    output = tmp_path / "dataset"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Pass --force"):
        import_game_videos([first, second], output)
    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"


def test_import_rejects_one_clip_and_identical_recordings(tmp_path):
    first = tmp_path / "first.avi"
    duplicate = tmp_path / "duplicate.avi"
    _write_clip(first, 1)
    duplicate.write_bytes(first.read_bytes())

    with pytest.raises(ValueError, match="at least two separate"):
        import_game_videos([first], tmp_path / "one")
    with pytest.raises(ValueError, match="identical file content"):
        import_game_videos([first, duplicate], tmp_path / "duplicate-out")
