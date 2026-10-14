"""End-to-end verification of the complete synthetic TrSS pilot experiment."""

from __future__ import annotations

import json

import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("openvino")

from trss.training.pilot_experiment import run_pilot_experiment


def test_complete_pilot_runs_train_evaluate_export_and_openvino(tmp_path, monkeypatch):
    # Small CPU-only tensors are faster and stable when worker libraries use one thread.
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")

    output = tmp_path / "trss-pilot-experiment"
    report = run_pilot_experiment(
        output,
        scenes=4,
        frames_per_scene=2,
        width=32,
        height=20,
        seed=53031,
        validation_fraction=0.25,
        epochs=1,
        train_device="cpu",
        evaluation_device="cpu",
        inference_device="CPU",
        channels=8,
        blocks=1,
        context_blocks=0,
        patch_size=8,
        batch_size=1,
        max_review_frames=4,
        tile_size=16,
        tile_overlap=2,
    )

    assert report["experiment_completed"] is True
    assert report["source_type"] == "synthetic_controlled_3d_scene"
    assert report["actual_game_capture"] is False
    assert report["stages"]["dataset"]["passed"] is True
    assert report["stages"]["training"]["completed_epochs"] == 1
    assert report["stages"]["quality_evaluation"]["passed"] is True
    assert report["stages"]["onnx_export"]["passed"] is True
    assert report["stages"]["openvino_smoke"]["passed"] is True
    assert report["stages"]["openvino_smoke"]["device"] == "CPU"
    assert report["stages"]["openvino_smoke"]["live_capture_used"] is False
    assert len(report["stages"]["openvino_smoke"]["output_sha256"]) == 64
    assert report["stages"]["openvino_smoke"]["target_metrics"]["mae"] >= 0
    assert report["stages"]["quality_evaluation"]["metrics"]["frames"] == 2

    for relative in (
        "dataset/pipeline-report.json",
        "dataset/train.jsonl",
        "dataset/validation.jsonl",
        "dataset/review-gallery/index.html",
        "training/training-report.json",
        "training/checkpoint-best.pt",
        "training/checkpoint-last.pt",
        "evaluation/evaluation-report.json",
        "evaluation/index.html",
        "artifacts/trss-composed.onnx",
        "artifacts/trss-composed.onnx.json",
        "openvino-smoke/frame.png",
        "openvino-smoke/frame.png.json",
        "experiment-report.json",
        "training.log",
    ):
        assert (output / relative).is_file(), relative

    with Image.open(output / "openvino-smoke/frame.png") as image:
        assert image.size == (64, 40)
        assert image.mode == "RGB"

    report_path = output / "experiment-report.json"
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["output_dir"] == str(output.resolve())
    training_log = (output / "training.log").read_text(encoding="utf-8")
    assert str(output.resolve()) in training_log
    assert ".experiment-" not in training_log
    assert saved["stages"]["training"]["checkpoint"] == "training/checkpoint-best.pt"
    assert saved["stages"]["onnx_export"]["artifact"] == "artifacts/trss-composed.onnx"
    evaluation = json.loads(
        (output / "evaluation/evaluation-report.json").read_text(encoding="utf-8")
    )
    assert evaluation["checkpoint"] == str((output / "training/checkpoint-best.pt").resolve())
    assert evaluation["evaluation_completed"] is True


def test_pilot_experiment_refuses_to_replace_existing_output_without_force(tmp_path):
    output = tmp_path / "safe-output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("safe", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Pass --force"):
        run_pilot_experiment(output)

    assert marker.read_text(encoding="utf-8") == "safe"


def test_pilot_experiment_validates_arguments_before_touching_output(tmp_path):
    output = tmp_path / "no-output"
    with pytest.raises(ValueError, match="epochs"):
        run_pilot_experiment(output, epochs=0)

    assert not output.exists()
