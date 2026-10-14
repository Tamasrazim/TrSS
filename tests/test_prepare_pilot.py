"""End-to-end tests for the one-command controlled-scene dataset pipeline."""

from __future__ import annotations

import json

import pytest

from trss.training.prepare_pilot import prepare_pilot_dataset


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
