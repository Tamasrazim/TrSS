"""Tests for the deterministic controlled-scene TrSS data generator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from trss.training.audit_dataset import audit_manifests
from trss.training.generate_scene_dataset import generate_scene_dataset
from trss.training.manifest import load_manifest
from trss.training.split_manifest import split_manifest


def test_generator_writes_complete_paired_sequences_and_geometry_buffers(tmp_path):
    output = tmp_path / "pilot"
    report = generate_scene_dataset(
        output,
        scenes=2,
        frames_per_scene=3,
        width=40,
        height=24,
        seed=1234,
    )

    assert report["frame_count"] == 6
    assert report["actual_game_capture"] is False
    rows = load_manifest(output / "all-frames.jsonl")
    assert len(rows) == 6

    by_scene: dict[str, list[dict]] = {}
    for row in rows:
        raw_line = (output / "all-frames.jsonl").read_text(encoding="utf-8").splitlines()[
            row["line"] - 1
        ]
        raw = json.loads(raw_line)
        by_scene.setdefault(raw["scene_id"], []).append((raw, row))

        with Image.open(row["low_res"]) as image:
            assert image.size == (40, 24)
        with Image.open(row["target"]) as image:
            assert image.size == (80, 48)

        depth = np.load(row["depth"], allow_pickle=False)
        motion = np.load(row["motion"], allow_pickle=False)
        assert depth.shape == (24, 40)
        assert depth.dtype == np.float32
        assert np.isfinite(depth).all()
        assert float(depth.min()) >= 0
        assert float(depth.max()) <= 1
        assert motion.shape == (24, 40, 2)
        assert motion.dtype == np.float32
        assert np.isfinite(motion).all()

    for scene_rows in by_scene.values():
        scene_rows.sort(key=lambda item: item[0]["frame_index"])
        assert scene_rows[0][0]["history"] is None
        assert scene_rows[0][0]["history_valid"] is None
        assert np.count_nonzero(
            np.load(scene_rows[0][1]["motion"], allow_pickle=False)
        ) == 0
        for raw, resolved in scene_rows[1:]:
            assert raw["history"] is not None
            assert raw["history_valid"] is not None
            with Image.open(resolved["history"]) as history:
                assert history.size == (80, 48)
            with Image.open(resolved["history_valid"]) as mask:
                assert mask.size == (80, 48)
                values = np.asarray(mask, dtype=np.uint8)
                assert set(np.unique(values)).issubset({0, 255})
        # A moving camera and moving props should produce non-zero flow.
        assert any(
            np.any(np.abs(np.load(resolved["motion"], allow_pickle=False)) > 1e-4)
            for _, resolved in scene_rows[1:]
        )

    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert "synthetic" in provenance["data_kind"]
    generation = json.loads((output / "generation-report.json").read_text(encoding="utf-8"))
    assert generation["actual_game_capture"] is False
    assert "not a capture hook" in generation["limitations"][0]


def test_generator_is_deterministic_for_same_seed_and_settings(tmp_path):
    settings = {
        "scenes": 2,
        "frames_per_scene": 2,
        "width": 32,
        "height": 20,
        "seed": 87,
    }
    left = tmp_path / "left"
    right = tmp_path / "right"
    generate_scene_dataset(left, **settings)
    generate_scene_dataset(right, **settings)

    relative_files = [
        "all-frames.jsonl",
        "provenance.json",
        "generation-report.json",
        "input/scene-000/frame-0000.png",
        "target/scene-001/frame-0001.png",
        "depth/scene-001/frame-0001.npy",
        "motion/scene-000/frame-0001.npy",
        "history/scene-000/frame-0001.png",
        "history_valid/scene-000/frame-0001.png",
    ]
    for relative in relative_files:
        assert (left / relative).read_bytes() == (right / relative).read_bytes()


def test_generator_protects_existing_output_unless_force_is_set(tmp_path):
    output = tmp_path / "pilot"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Use --force"):
        generate_scene_dataset(
            output,
            scenes=2,
            frames_per_scene=2,
            width=32,
            height=20,
        )
    assert marker.read_text(encoding="utf-8") == "keep me"

    generate_scene_dataset(
        output,
        scenes=2,
        frames_per_scene=2,
        width=32,
        height=20,
        force=True,
    )
    assert not marker.exists()
    assert (output / "all-frames.jsonl").is_file()


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("scenes", 1, "at least 2"),
        ("frames_per_scene", 1, "at least 2"),
        ("width", 12, "at least 24"),
        ("height", 8, "at least 16"),
    ],
)
def test_generator_rejects_invalid_dimensions_and_counts(
    tmp_path, setting, value, message
):
    settings = {
        "scenes": 2,
        "frames_per_scene": 2,
        "width": 32,
        "height": 20,
    }
    settings[setting] = value
    with pytest.raises(ValueError, match=message):
        generate_scene_dataset(tmp_path / "bad", **settings)



def test_generated_data_splits_by_scene_and_passes_structural_audit(tmp_path):
    output = tmp_path / "pipeline"
    generate_scene_dataset(
        output,
        scenes=4,
        frames_per_scene=2,
        width=64,
        height=36,
        seed=991,
    )
    train = output / "train.jsonl"
    validation = output / "validation.jsonl"
    split = split_manifest(
        output / "all-frames.jsonl",
        train,
        validation,
        validation_fraction=0.25,
        seed=17,
        scene_field="scene_id",
    )
    assert set(split["train_scenes"]).isdisjoint(set(split["validation_scenes"]))

    report = audit_manifests(
        train,
        validation,
        tmp_path / "audit.json",
        provenance_json=output / "provenance.json",
    )
    assert report["passed"] is True, report["errors"]
    assert report["splits"]["training"]["frames"] + report["splits"]["validation"]["frames"] == 8
    assert report["cross_split_duplicate_pixel_groups"] == {
        "low_res": 0,
        "target": 0,
    }
