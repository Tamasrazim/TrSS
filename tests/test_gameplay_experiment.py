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



def _write_desktop_session(root: Path, offset: int, game_label: str) -> None:
    root.mkdir(parents=True)
    (root / ".trss-desktop-capture-session").write_text(
        "Test capture marker", encoding="utf-8"
    )
    (root / "frames").mkdir()
    manifest_rows = []
    for index in range(4):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = (offset + index * 13) % 255
        frame[:, :, 1] = (offset + index * 17) % 255
        frame[:, :, 2] = (index * 29) % 255
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
        "".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8"
    )
    (root / "session.json").write_text(
        json.dumps({
            "format": "trss-desktop-capture-session-v1",
            "game_label": game_label,
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


def test_gameplay_experiment_trains_from_native_capture_sessions(tmp_path):
    first = tmp_path / "ets2-road-a"
    second = tmp_path / "ets2-road-b"
    _write_desktop_session(first, 19, "Euro Truck Simulator 2")
    _write_desktop_session(second, 137, "Euro Truck Simulator 2")

    output = tmp_path / "native-capture-experiment"
    report = run_gameplay_experiment(
        [first, second],
        output,
        game="Euro Truck Simulator 2",
        max_frames_per_video=3,
        validation_fraction=0.5,
        seed=21,
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
    assert report["dataset"]["training_frames"] == 3
    assert report["dataset"]["validation_frames"] == 3
    assert report["training"]["epochs_completed"] == 1
    assert (output / "training/checkpoint-best.pt").is_file()
    assert (output / "evaluation/evaluation-report.json").is_file()
    imported = [
        json.loads(line)
        for line in (output / "dataset/train.jsonl").read_text(encoding="utf-8").splitlines()
    ] + [
        json.loads(line)
        for line in (output / "dataset/validation.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["source_type"] == "trss-desktop-capture-session" for row in imported)
    assert all(row["game_label"] == "Euro Truck Simulator 2" for row in imported)
