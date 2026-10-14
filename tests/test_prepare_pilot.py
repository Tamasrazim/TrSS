"""End-to-end tests for the one-command controlled-scene dataset pipeline."""

from __future__ import annotations

import json
import sys

import pytest
import torch

from trss.training.prepare_pilot import prepare_pilot_dataset
from trss.training.train import main as train_main


def test_prepare_pilot_runs_generation_split_audit_and_gallery_atomically(tmp_path):
    output = tmp_path / "pilot"
    report = prepare_pilot_dataset(
        output,
        scenes=4,
        frames_per_scene=2,
        width=40,
        height=24,
        seed=9281,
        validation_fraction=0.25,
        max_review_frames=8,
    )

    assert report["passed"] is True, report
    assert report["actual_game_capture"] is False
    assert report["generation"]["frame_count"] == 8
    assert report["split"]["scene_split_is_disjoint"] is True
    assert set(report["split"]["training_scenes"]).isdisjoint(
        set(report["split"]["validation_scenes"])
    )
    assert report["audit"]["passed"] is True
    assert report["review"]["passed"] is True

    for relative in (
        "all-frames.jsonl",
        "train.jsonl",
        "validation.jsonl",
        "provenance.json",
        "generation-report.json",
        "dataset-audit.json",
        "pipeline-report.json",
        "review-gallery/index.html",
        "review-gallery/review-report.json",
    ):
        assert (output / relative).is_file(), relative

    saved = json.loads((output / "pipeline-report.json").read_text(encoding="utf-8"))
    assert saved["output_dir"] == str(output.resolve()) or "output_dir" not in saved
    audit = json.loads((output / "dataset-audit.json").read_text(encoding="utf-8"))
    assert audit["report_path"] == str((output / "dataset-audit.json").resolve())
    assert audit["splits"]["training"]["manifest"] == str((output / "train.jsonl").resolve())
    review = json.loads((output / "review-gallery/review-report.json").read_text(encoding="utf-8"))
    assert review["gallery_path"] == str((output / "review-gallery/index.html").resolve())


def test_prepare_pilot_protects_existing_output_unless_force_is_set(tmp_path):
    output = tmp_path / "pilot"
    output.mkdir()
    marker = output / "do-not-delete.txt"
    marker.write_text("safe", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Pass --force"):
        prepare_pilot_dataset(
            output, scenes=2, frames_per_scene=2, width=32, height=20
        )
    assert marker.read_text(encoding="utf-8") == "safe"


def test_prepare_pilot_rejects_a_single_scene_before_touching_output(tmp_path):
    output = tmp_path / "pilot"
    with pytest.raises(ValueError, match="at least 2"):
        prepare_pilot_dataset(output, scenes=1)
    assert not output.exists()


def test_generated_pilot_can_complete_one_cpu_training_epoch(tmp_path, monkeypatch):
    """A tiny epoch proves the paired data reaches the actual optimizer/checkpoint path."""
    output = tmp_path / "trainable-pilot"
    prepared = prepare_pilot_dataset(
        output,
        scenes=4,
        frames_per_scene=2,
        width=32,
        height=20,
        seed=331,
        validation_fraction=0.25,
        max_review_frames=4,
    )
    assert prepared["passed"] is True, prepared

    training_output = tmp_path / "one-epoch"
    args = [
        "trss-training-smoke",
        "--train-manifest", str(output / "train.jsonl"),
        "--validation-manifest", str(output / "validation.jsonl"),
        "--output-dir", str(training_output),
        "--device", "cpu",
        "--channels", "8",
        "--blocks", "1",
        "--context-blocks", "0",
        "--patch-size", "8",
        "--batch-size", "1",
        "--epochs", "1",
        "--seed", "331",
    ]
    previous_threads = torch.get_num_threads()
    try:
        # Tiny tensors run much faster and more predictably with one CPU worker thread.
        torch.set_num_threads(1)
        monkeypatch.setattr(sys, "argv", args)
        assert train_main() == 0
    finally:
        torch.set_num_threads(previous_threads)

    report = json.loads((training_output / "training-report.json").read_text(encoding="utf-8"))
    assert len(report["history"]) == 1
    epoch = report["history"][0]
    assert epoch["epoch"] == 1
    assert epoch["validation_loss"] is not None
    assert epoch["train_loss"] >= 0
    assert (training_output / "checkpoint-last.pt").is_file()
    assert (training_output / "checkpoint-best.pt").is_file()
