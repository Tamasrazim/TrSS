"""Tests for the offline paired-frame review gallery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from trss.training.review_dataset import build_review_gallery


def _save_rgb(path: Path, height: int, width: int, value: int) -> None:
    pixels = np.full((height, width, 3), value, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def _make_manifest(folder: Path, rows_per_scene: dict[str, int]) -> Path:
    rows = []
    for scene, count in rows_per_scene.items():
        for index in range(count):
            prefix = f"{scene}-{index}"
            low = folder / f"{prefix}-low.png"
            target = folder / f"{prefix}-target.png"
            depth = folder / f"{prefix}-depth.npy"
            motion = folder / f"{prefix}-motion.npy"
            _save_rgb(low, 6, 8, 30 + index)
            _save_rgb(target, 12, 16, 50 + index)
            np.save(depth, np.full((6, 8), 0.5, dtype=np.float32))
            np.save(motion, np.zeros((6, 8, 2), dtype=np.float32))
            rows.append({
                "low_res": low.name,
                "target": target.name,
                "depth": depth.name,
                "motion": motion.name,
                "history": None,
                "history_valid": None,
                "scene_id": scene,
            })
    manifest = folder / "frames.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return manifest


def test_gallery_is_offline_and_contains_both_images_and_scene_metadata(tmp_path):
    source = _make_manifest(tmp_path, {"scene-a": 3, "scene-b": 2, "scene-c": 1})
    out = tmp_path / "review"

    report = build_review_gallery(source, out, max_frames=4, seed=11)

    html = (out / "index.html").read_text(encoding="utf-8")
    saved = json.loads((out / "review-report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["source_frames"] == 6
    assert report["selected_count"] == 4
    assert report["preview_errors"] == 0
    assert report["gallery_path"] == str((out / "index.html").resolve())
    assert saved["gallery_path"] == report["gallery_path"]
    assert "INPUT" in html and "REFERENCE TARGET" in html
    assert any(scene in html for scene in ("scene-a", "scene-b", "scene-c"))
    assert "https://" not in html and "<script>" in html
    for row in report["selected_frames"]:
        assert row["input_thumbnail"]
        assert row["target_thumbnail"]
        assert (out / row["input_thumbnail"]).is_file()
        assert (out / row["target_thumbnail"]).is_file()
        assert row["target_width"] == row["input_width"] * 2
        assert row["target_height"] == row["input_height"] * 2


def test_gallery_sampling_is_repeatable_for_same_seed(tmp_path):
    source = _make_manifest(tmp_path, {"scene-a": 4, "scene-b": 3, "scene-c": 2})
    first = build_review_gallery(source, tmp_path / "one", max_frames=5, seed=123)
    second = build_review_gallery(source, tmp_path / "two", max_frames=5, seed=123)

    first_lines = [row["manifest_line"] for row in first["selected_frames"]]
    second_lines = [row["manifest_line"] for row in second["selected_frames"]]
    assert first_lines == second_lines
    assert first["selected_scene_counts"] == second["selected_scene_counts"]


def test_gallery_spreads_selection_across_scenes_when_sample_allows(tmp_path):
    source = _make_manifest(tmp_path, {"scene-a": 8, "scene-b": 2, "scene-c": 1})
    report = build_review_gallery(source, tmp_path / "review", max_frames=3, seed=44)
    assert len(report["selected_scene_counts"]) == 3


def test_gallery_records_bad_image_but_still_writes_review_page(tmp_path):
    source = _make_manifest(tmp_path, {"scene-a": 1, "scene-b": 1})
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
    ]
    (tmp_path / rows[0]["low_res"]).write_bytes(b"not a valid image")
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = build_review_gallery(source, tmp_path / "review", max_frames=2, seed=1)

    assert report["passed"] is False
    assert report["preview_errors"] == 1
    assert (tmp_path / "review" / "index.html").is_file()
    assert "PREVIEW ERROR" in (tmp_path / "review" / "index.html").read_text(encoding="utf-8")


def test_gallery_validates_sampling_and_thumbnail_sizes(tmp_path):
    source = _make_manifest(tmp_path, {"scene-a": 1, "scene-b": 1})
    for options, expected in (
        ({"max_frames": 0}, "max_frames"),
        ({"thumbnail_width": 30}, "thumbnail dimensions"),
    ):
        try:
            build_review_gallery(source, tmp_path / "review", **options)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"{expected} must be rejected")
