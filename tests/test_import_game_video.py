"""Tests for importing gameplay videos and native TrSS desktop capture sessions."""
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



def _write_desktop_session(root: Path, offset: int, game_label: str) -> None:
    root.mkdir(parents=True)
    (root / ".trss-desktop-capture-session").write_text(
        "Created by TrSS Desktop Capture. Safe for --force replacement.\\n",
        encoding="utf-8",
    )
    frame_dir = root / "frames"
    frame_dir.mkdir()
    manifest_rows = []
    for index in range(4):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = (offset + index * 11) % 255
        frame[:, :, 1] = (index * 23 + offset) % 255
        frame[:, :, 2] = (index * 7) % 255
        relative = f"frames/frame_{index:06d}.png"
        Image.fromarray(frame, mode="RGB").save(root / relative)
        manifest_rows.append({
            "frame_id": index,
            "timestamp": index / 2,
            "color": relative,
            "scene_id": "desktop-capture-session",
            "camera_cut": index == 0,
            "color_space": "unknown",
            "depth_encoding": "unknown",
            "motion_units": "unknown",
        })
    (root / "capture-manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\\n" for row in manifest_rows), encoding="utf-8"
    )
    (root / "session.json").write_text(
        json.dumps({
            "format": "trss-desktop-capture-session-v1",
            "game_label": game_label,
            "game_label_source": "user-supplied; no game-process detection",
            "width": 32,
            "height": 24,
            "requested_fps": 2,
            "measured_output_fps": 2.0,
            "frames_saved": len(manifest_rows),
            "elapsed_seconds": 1.5,
            "has_depth": False,
            "has_motion_vectors": False,
            "has_high_resolution_reference": False,
            "training_ready": False,
        }),
        encoding="utf-8",
    )


def test_imports_native_capture_sessions_and_keeps_sessions_disjoint(tmp_path):
    first = tmp_path / "ets2-road-a"
    second = tmp_path / "ets2-road-b"
    _write_desktop_session(first, 10, "Euro Truck Simulator 2")
    _write_desktop_session(second, 120, "Euro Truck Simulator 2")

    output = tmp_path / "native-import"
    report = import_game_videos(
        [first, second], output, sample_every=15, max_frames_per_video=3, seed=9
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
    assert report["dataset_type"].endswith("or-desktop-capture")
    assert report["training_frames"] == 3
    assert report["validation_frames"] == 3
    assert {row["scene_id"] for row in train_rows}.isdisjoint(
        {row["scene_id"] for row in validation_rows}
    )
    for row in train_rows + validation_rows:
        with Image.open(output / row["low_res"]) as low, Image.open(output / row["target"]) as target:
            assert target.size == (low.width * 2, low.height * 2)
        assert row["source_type"] == "trss-desktop-capture-session"
        assert row["game_label"] == "Euro Truck Simulator 2"
        assert row["timestamp_seconds"] in (0, 0.5, 1.0)
        assert row["auxiliary_buffers"] == "not captured; use spatial-only model"


def test_import_rejects_capture_manifest_path_traversal(tmp_path):
    first = tmp_path / "valid"
    second = tmp_path / "invalid"
    _write_desktop_session(first, 10, "Euro Truck Simulator 2")
    _write_desktop_session(second, 120, "Euro Truck Simulator 2")
    manifest = second / "capture-manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    rows[1]["color"] = "../outside.png"
    manifest.write_text("".join(json.dumps(row) + "\\n" for row in rows), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the capture folder"):
        import_game_videos([first, second], tmp_path / "invalid-import")
