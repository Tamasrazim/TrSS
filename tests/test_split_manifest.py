"""Tests for deterministic, scene-separated dataset manifest generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trss.training.manifest import load_manifest
from trss.training.split_manifest import split_manifest


def _make_source(folder: Path, scene_counts: dict[str, int]) -> Path:
    rows = []
    for scene, count in scene_counts.items():
        for index in range(count):
            prefix = f"{scene}-{index}"
            paths = {
                "low_res": f"{prefix}-low.png",
                "target": f"{prefix}-target.png",
                "depth": f"{prefix}-depth.npy",
                "motion": f"{prefix}-motion.npy",
            }
            for name in paths.values():
                (folder / name).write_bytes(b"fixture data")
            rows.append({
                **paths,
                "history": None,
                "history_valid": None,
                "scene_id": scene,
                "capture_note": "keep provenance fields",
            })
    manifest = folder / "all-frames.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def _read_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_split_keeps_each_scene_whole_and_rebases_paths(tmp_path):
    source = _make_source(tmp_path, {"scene-a": 4, "scene-b": 2, "scene-c": 1})
    train_path = tmp_path / "generated" / "train.jsonl"
    validation_path = tmp_path / "generated" / "validation.jsonl"

    report = split_manifest(source, train_path, validation_path, seed=19)

    train_rows = _read_rows(train_path)
    validation_rows = _read_rows(validation_path)
    train_scenes = {str(row["scene_id"]) for row in train_rows}
    validation_scenes = {str(row["scene_id"]) for row in validation_rows}

    assert train_scenes
    assert validation_scenes
    assert train_scenes.isdisjoint(validation_scenes)
    assert train_scenes | validation_scenes == {"scene-a", "scene-b", "scene-c"}
    assert len(train_rows) + len(validation_rows) == 7
    assert all(row["capture_note"] == "keep provenance fields" for row in train_rows)
    assert all(row["capture_note"] == "keep provenance fields" for row in validation_rows)
    assert len(load_manifest(train_path)) == len(train_rows)
    assert len(load_manifest(validation_path)) == len(validation_rows)
    assert report["scene_split_is_disjoint"] is True
    assert report["total_frames"] == 7
    assert report["training_frames"] == len(train_rows)
    assert report["validation_frames"] == len(validation_rows)


def test_split_is_deterministic_for_the_same_seed(tmp_path):
    source = _make_source(tmp_path, {"scene-a": 3, "scene-b": 2, "scene-c": 2, "scene-d": 1})
    first = split_manifest(
        source, tmp_path / "first" / "train.jsonl", tmp_path / "first" / "val.jsonl",
        seed=73,
    )
    second = split_manifest(
        source, tmp_path / "second" / "train.jsonl", tmp_path / "second" / "val.jsonl",
        seed=73,
    )
    assert first["training_scenes"] == second["training_scenes"]
    assert first["validation_scenes"] == second["validation_scenes"]


def test_split_rejects_shared_color_frame_with_different_scene_labels(tmp_path):
    source = _make_source(tmp_path, {"scene-a": 1, "scene-b": 1})
    rows = _read_rows(source)
    rows[1]["low_res"] = rows[0]["low_res"]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="assigned to multiple scene IDs"):
        split_manifest(source, tmp_path / "train.jsonl", tmp_path / "val.jsonl")


def test_split_requires_two_distinct_scenes(tmp_path):
    source = _make_source(tmp_path, {"only-scene": 4})
    with pytest.raises(ValueError, match="At least two distinct scene IDs"):
        split_manifest(source, tmp_path / "train.jsonl", tmp_path / "val.jsonl")


def test_split_requires_explicit_nonempty_scene_metadata(tmp_path):
    source = _make_source(tmp_path, {"scene-a": 1, "scene-b": 1})
    rows = _read_rows(source)
    del rows[0]["scene_id"]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-empty 'scene_id'"):
        split_manifest(source, tmp_path / "train.jsonl", tmp_path / "val.jsonl")


def test_split_will_not_overwrite_existing_outputs_without_force(tmp_path):
    source = _make_source(tmp_path, {"scene-a": 1, "scene-b": 1})
    train_path = tmp_path / "train.jsonl"
    train_path.write_text("preserve this file", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Pass --force"):
        split_manifest(source, train_path, tmp_path / "val.jsonl")
    assert train_path.read_text(encoding="utf-8") == "preserve this file"


def test_split_refuses_to_overwrite_the_input_manifest(tmp_path):
    source = _make_source(tmp_path, {"scene-a": 1, "scene-b": 1})
    original = source.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="must not overwrite the input manifest"):
        split_manifest(source, source, tmp_path / "val.jsonl")
    assert source.read_text(encoding="utf-8") == original
