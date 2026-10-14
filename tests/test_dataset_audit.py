"""Tests for the TrSS dataset audit command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from trss.training.audit_dataset import audit_manifests


def _save_rgb(path: Path, height: int, width: int, value: int) -> None:
    pixels = np.full((height, width, 3), value, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _make_row(folder: Path, scene: str, index: int, value: int) -> dict[str, object]:
    prefix = f"{scene}-{index}"
    low_path = folder / f"{prefix}-low.png"
    target_path = folder / f"{prefix}-target.png"
    depth_path = folder / f"{prefix}-depth.npy"
    motion_path = folder / f"{prefix}-motion.npy"
    _save_rgb(low_path, 3, 4, value)
    _save_rgb(target_path, 6, 8, min(value + 1, 255))
    np.save(depth_path, np.full((3, 4), 0.5, dtype=np.float32))
    np.save(motion_path, np.zeros((3, 4, 2), dtype=np.float32))
    return {
        "low_res": low_path.name,
        "target": target_path.name,
        "depth": depth_path.name,
        "motion": motion_path.name,
        "history": None,
        "history_valid": None,
        "scene_id": scene,
    }


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _provenance(path: Path) -> Path:
    path.write_text(
        json.dumps({
            "dataset_name": "unit-test-renders",
            "source": "generated test fixtures",
            "license": "test-only",
        }),
        encoding="utf-8",
    )
    return path


def test_audit_passes_valid_separated_manifests_and_reports_statistics(tmp_path):
    train_rows = [
        _make_row(tmp_path, "scene-a", 0, 40),
        _make_row(tmp_path, "scene-a", 1, 60),
    ]
    validation_rows = [_make_row(tmp_path, "scene-b", 0, 90)]
    train = _write_manifest(tmp_path / "train.jsonl", train_rows)
    validation = _write_manifest(tmp_path / "validation.jsonl", validation_rows)
    provenance = _provenance(tmp_path / "provenance.json")

    report = audit_manifests(
        train, validation, tmp_path / "audit.json", provenance_json=provenance
    )

    assert report["passed"] is True
    assert report["cross_split_duplicate_pixel_groups"] == {"low_res": 0, "target": 0}
    assert report["splits"]["training"]["frames"] == 2
    assert report["splits"]["validation"]["input_dimensions_and_counts"] == {"4x3": 1}
    assert report["splits"]["training"]["history_reset_frames"] == 2
    assert report["provenance"]["license"] == "test-only"
    saved = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert saved["passed"] is True
    assert saved["report_path"] == str((tmp_path / "audit.json").resolve())


def test_audit_catches_identical_decoded_frames_saved_under_different_names(tmp_path):
    train = _write_manifest(
        tmp_path / "train.jsonl", [_make_row(tmp_path, "scene-a", 0, 40)]
    )
    validation = _write_manifest(
        tmp_path / "validation.jsonl", [_make_row(tmp_path, "scene-b", 0, 40)]
    )

    report = audit_manifests(train, validation, tmp_path / "audit.json")

    assert report["passed"] is False
    assert report["cross_split_duplicate_pixel_groups"]["low_res"] == 1
    assert report["cross_split_near_duplicate_warning_counts"] == {
        "low_res": 0, "target": 0
    }
    assert any(
        "identical decoded frame pixels occur in both" in error["error"]
        for error in report["errors"]
    )


def test_audit_catches_scene_overlap_even_when_pixels_differ(tmp_path):
    train = _write_manifest(
        tmp_path / "train.jsonl", [_make_row(tmp_path, "shared-scene", 0, 40)]
    )
    validation = _write_manifest(
        tmp_path / "validation.jsonl", [_make_row(tmp_path, "shared-scene", 1, 90)]
    )

    report = audit_manifests(train, validation, tmp_path / "audit.json")

    assert report["passed"] is False
    assert any(
        error.get("overlapping_scenes") == ["shared-scene"]
        for error in report["errors"]
    )


def test_audit_reports_invalid_auxiliary_tensor_shapes(tmp_path):
    train_row = _make_row(tmp_path, "scene-a", 0, 40)
    np.save(tmp_path / "bad-motion.npy", np.zeros((3, 4), dtype=np.float32))
    train_row["motion"] = "bad-motion.npy"
    train = _write_manifest(tmp_path / "train.jsonl", [train_row])
    validation = _write_manifest(
        tmp_path / "validation.jsonl", [_make_row(tmp_path, "scene-b", 0, 90)]
    )

    report = audit_manifests(train, validation, tmp_path / "audit.json")

    assert report["passed"] is False
    assert any("motion shape" in error["error"] for error in report["errors"])


def test_audit_warns_when_provenance_is_not_supplied(tmp_path):
    train = _write_manifest(
        tmp_path / "train.jsonl", [_make_row(tmp_path, "scene-a", 0, 40)]
    )
    validation = _write_manifest(
        tmp_path / "validation.jsonl", [_make_row(tmp_path, "scene-b", 0, 90)]
    )

    report = audit_manifests(train, validation, tmp_path / "audit.json")

    assert report["passed"] is True
    assert any("No provenance JSON" in warning["warning"] for warning in report["warnings"])


def test_audit_rejects_incomplete_provenance(tmp_path):
    train = _write_manifest(
        tmp_path / "train.jsonl", [_make_row(tmp_path, "scene-a", 0, 40)]
    )
    validation = _write_manifest(
        tmp_path / "validation.jsonl", [_make_row(tmp_path, "scene-b", 0, 90)]
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps({"dataset_name": "missing licence"}), encoding="utf-8"
    )

    report = audit_manifests(
        train, validation, tmp_path / "audit.json", provenance_json=provenance
    )

    assert report["passed"] is False
    assert any("source, license" in error["error"] for error in report["errors"])



def test_audit_warns_about_near_duplicates_across_splits_without_failing(tmp_path):
    train = _write_manifest(
        tmp_path / "train.jsonl", [_make_row(tmp_path, "scene-a", 0, 40)]
    )
    validation = _write_manifest(
        tmp_path / "validation.jsonl", [_make_row(tmp_path, "scene-b", 0, 45)]
    )

    report = audit_manifests(train, validation, tmp_path / "audit.json")

    assert report["passed"] is True
    assert report["cross_split_near_duplicate_warning_counts"] == {
        "low_res": 1, "target": 1
    }
    near_warnings = [
        warning for warning in report["warnings"]
        if "possible near-duplicate" in warning.get("warning", "")
    ]
    assert len(near_warnings) == 2
    assert all(warning["hamming_distance"] == 0 for warning in near_warnings)
    assert all(warning["mean_rgb_distance"] < 10 for warning in near_warnings)


def test_audit_rejects_invalid_near_duplicate_thresholds(tmp_path):
    train = _write_manifest(
        tmp_path / "train.jsonl", [_make_row(tmp_path, "scene-a", 0, 40)]
    )
    validation = _write_manifest(
        tmp_path / "validation.jsonl", [_make_row(tmp_path, "scene-b", 0, 90)]
    )

    try:
        audit_manifests(
            train, validation, tmp_path / "audit.json",
            near_duplicate_hamming_distance=17,
        )
    except ValueError as error:
        assert "from 0 to 16" in str(error)
    else:
        raise AssertionError("expected invalid near-duplicate threshold to be rejected")
