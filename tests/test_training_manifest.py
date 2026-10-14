"""Manifest tests do not require the optional PyTorch dependency."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trss.training.manifest import ManifestError, load_manifest


def _make_row_files(folder: Path) -> Path:
    for name in ("low.png", "target.png", "depth.npy", "motion.npy"):
        (folder / name).write_bytes(b"test fixture")
    manifest = folder / "train.jsonl"
    manifest.write_text(
        json.dumps({
            "low_res": "low.png",
            "target": "target.png",
            "depth": "depth.npy",
            "motion": "motion.npy",
            "history": None,
            "history_valid": None,
        }) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_manifest_resolves_paths_and_requires_explicit_history_reset(tmp_path):
    manifest = _make_row_files(tmp_path)
    rows = load_manifest(manifest)
    assert len(rows) == 1
    assert rows[0]["low_res"] == (tmp_path / "low.png").resolve()
    assert rows[0]["history"] is None
    assert rows[0]["history_valid"] is None


def test_manifest_rejects_missing_auxiliary_buffers(tmp_path):
    manifest = _make_row_files(tmp_path)
    manifest.write_text(
        json.dumps({"low_res": "low.png", "target": "target.png"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="depth, motion, history"):
        load_manifest(manifest)


def test_manifest_rejects_history_without_a_validity_mask(tmp_path):
    manifest = _make_row_files(tmp_path)
    row = json.loads(manifest.read_text(encoding="utf-8"))
    row["history"] = "target.png"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="must either both name files or both be null"):
        load_manifest(manifest)
