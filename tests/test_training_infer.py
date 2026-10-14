"""CPU tests for trained-checkpoint offline inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from trss.training.infer import run_frame
from trss.training.model import MODEL_FORMAT, TrSSTemporalSR


def _save_rgb(path: Path, height: int, width: int, value: int) -> None:
    Image.fromarray(
        np.full((height, width, 3), value, dtype=np.uint8), mode="RGB"
    ).save(path)


def test_offline_inference_loads_checkpoint_writes_double_size_and_reports(tmp_path):
    model = TrSSTemporalSR(channels=16, blocks=2, context_blocks=1)
    checkpoint = tmp_path / "test-checkpoint.pt"
    torch.save(
        {
            "format": MODEL_FORMAT,
            "config": model.config,
            "state_dict": model.state_dict(),
            "epoch": 3,
            "train_loss": 0.1,
            "validation_loss": 0.2,
        },
        checkpoint,
    )
    _save_rgb(tmp_path / "color.png", 4, 6, 90)
    np.save(tmp_path / "depth.npy", np.full((4, 6), 0.5, dtype=np.float32))
    np.save(tmp_path / "motion.npy", np.zeros((4, 6, 2), dtype=np.float32))
    output = tmp_path / "out" / "render.png"
    report = tmp_path / "logs" / "frame.json"

    info = run_frame(
        checkpoint_path=checkpoint,
        color_path=tmp_path / "color.png",
        depth_path=tmp_path / "depth.npy",
        motion_path=tmp_path / "motion.npy",
        output_path=output,
        device_name="cpu",
        warmup=0,
        iterations=1,
        report_path=report,
    )

    with Image.open(output) as rendered:
        assert rendered.size == (12, 8)
        assert rendered.mode == "RGB"
    assert info["checkpoint_epoch"] == 3
    assert info["input_size"] == [6, 4]
    assert info["output_size"] == [12, 8]
    assert info["history_supplied"] is False
    assert info["measured_iterations"] == 1
    assert len(info["checkpoint_sha256"]) == 64
    assert info["input_setup_ms"] >= 0
    assert info["mean_inference_ms"] >= 0
    assert info["mean_model_composition_ms"] >= 0
    assert info["output_encode_write_ms"] >= 0
    assert info["benchmark_call_elapsed_ms"] >= info["mean_model_composition_ms"]
    assert "not in-game FPS" in info["note"]
    assert "mean_frame_path_ms" not in info
    assert json.loads(report.read_text(encoding="utf-8"))["format"] == MODEL_FORMAT
