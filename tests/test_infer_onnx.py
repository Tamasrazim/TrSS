"""Tests for offline OpenVINO execution of the exported TrSS ONNX graph."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("openvino")

from trss.training.export_onnx import export_checkpoint_to_onnx
from trss.training.infer_onnx import run_onnx_frame
from trss.training.model import MODEL_FORMAT, TrSSTemporalSR


def _make_manifest(root):
    width, height = 24, 16
    Image.fromarray(np.full((height, width, 3), 90, dtype=np.uint8), mode="RGB").save(
        root / "low.png"
    )
    Image.fromarray(np.full((height * 2, width * 2, 3), 100, dtype=np.uint8), mode="RGB").save(
        root / "target.png"
    )
    np.save(root / "depth.npy", np.full((height, width), 0.5, dtype=np.float32))
    np.save(root / "motion.npy", np.zeros((height, width, 2), dtype=np.float32))
    row = {
        "low_res": "low.png",
        "target": "target.png",
        "depth": "depth.npy",
        "motion": "motion.npy",
        "history": None,
        "history_valid": None,
    }
    manifest = root / "frames.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


def test_openvino_runs_exported_graph_from_manifest_and_saves_png(tmp_path):
    torch.manual_seed(88)
    model = TrSSTemporalSR(channels=8, blocks=1, context_blocks=0).eval()
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({
        "format": MODEL_FORMAT,
        "config": model.config,
        "state_dict": model.state_dict(),
        "epoch": 1,
    }, checkpoint)
    artifact = tmp_path / "composed.onnx"
    export_checkpoint_to_onnx(checkpoint_path=checkpoint, output_path=artifact)
    manifest = _make_manifest(tmp_path)
    output = tmp_path / "render.png"

    report = run_onnx_frame(
        model_path=artifact,
        manifest_path=manifest,
        output_path=output,
        row_index=0,
        device_name="CPU",
        warmup=0,
        iterations=1,
    )

    assert report["format"] == "trss-onnx-frame-run-v1"
    assert report["device"] == "CPU"
    assert report["live_capture_used"] is False
    assert report["input_size"] == [24, 16]
    assert report["output_size"] == [48, 32]
    assert report["mean_inference_ms"] >= 0
    assert output.is_file()
    assert output.with_name("render.png.json").is_file()
    with Image.open(output) as image:
        assert image.size == (48, 32)
        assert image.mode == "RGB"


def test_openvino_runner_rejects_overwrite_without_force(tmp_path):
    manifest = _make_manifest(tmp_path)
    model = tmp_path / "dummy.onnx"
    model.write_bytes(b"not a real model")
    output = tmp_path / "preserve.png"
    report = output.with_name(output.name + ".json")
    output.write_bytes(b"keep image")
    report.write_text('{"keep": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="Pass --force"):
        run_onnx_frame(
            model_path=model,
            manifest_path=manifest,
            output_path=output,
            device_name="CPU",
        )

    assert output.read_bytes() == b"keep image"
    assert report.read_text(encoding="utf-8") == '{"keep": true}\n'
