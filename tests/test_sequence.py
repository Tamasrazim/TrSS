"""Tests for manifest-driven frame-sequence processing without a GPU."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from trss.rendering.adapter import AdapterSpecError
from trss.rendering.sequence import run_sequence


def _write_spec(path: Path, *, needs_history: bool = True) -> None:
    inputs = {
        "color": {
            "source": "color",
            "layout": "NCHW",
            "source_range": [0, 255],
            "target_range": [0, 1],
        }
    }
    if needs_history:
        inputs["previous"] = {
            "source": "history",
            "layout": "NCHW",
            "source_range": [0, 1],
            "target_range": [0, 1],
        }
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "inputs": inputs,
                "preview_output": {
                    "port": "rgb",
                    "layout": "NCHW",
                    "source_range": [0, 1],
                    "channel_order": "RGB",
                },
            }
        ),
        encoding="utf-8",
    )


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


class FakeSequenceEngine:
    """Make two distinct outputs and record the history tensor given to each call."""

    def __init__(self, model_path, *, device):
        self.model_path = str(Path(model_path).resolve())
        self.input_shapes = {
            "color": (1, 3, 1, 1),
            "previous": (1, 3, 1, 1),
        }
        self.device = "GPU.1"
        self.device_name = "Intel Arc B580 Graphics"
        self.calls = []
        self.closed = False

    def infer(self, inputs):
        self.calls.append(
            {
                "color": inputs["color"].copy(),
                "previous": inputs["previous"].copy(),
            }
        )
        value = 0.25 * len(self.calls)
        return SimpleNamespace(
            outputs={"rgb": np.full((1, 3, 1, 1), value, dtype=np.float32)},
            input_shapes={key: tuple(array.shape) for key, array in inputs.items()},
            output_shapes={"rgb": (1, 3, 1, 1)},
            compile_ms=2.0 if len(self.calls) == 1 else 0.0,
            inference_ms=1.5,
            total_ms=1.7,
            device=self.device,
            device_name=self.device_name,
            model_path=self.model_path,
        )

    def close(self):
        self.closed = True


def _create_sources(tmp_path: Path):
    model = tmp_path / "model.onnx"
    spec = tmp_path / "adapter.json"
    manifest = tmp_path / "frames.jsonl"
    first = tmp_path / "input-000.png"
    second = tmp_path / "input-001.png"
    seed = tmp_path / "history-seed.npy"
    model.write_bytes(b"fake model")
    _write_spec(spec)
    Image.new("RGB", (1, 1), (255, 0, 0)).save(first)
    Image.new("RGB", (1, 1), (0, 255, 0)).save(second)
    np.save(seed, np.full((1, 1, 3), 0.5, dtype=np.float32), allow_pickle=False)
    _write_manifest(
        manifest,
        [
            {
                "frame_id": 10,
                "timestamp": 1.0,
                "color": first.name,
                "initial_history": seed.name,
                "scene_id": "scene-a",
            },
            {
                "frame_id": 11,
                "timestamp": 1.1,
                "color": second.name,
                "scene_id": "scene-a",
            },
        ],
    )
    return model, spec, manifest, first, second, seed


def test_sequence_carries_previous_output_as_history_and_writes_rgb_frames(tmp_path):
    model, spec, manifest, _, _, _ = _create_sources(tmp_path)
    engines = []

    def engine_factory(model_path, *, device):
        engine = FakeSequenceEngine(model_path, device=device)
        engines.append(engine)
        return engine

    report = run_sequence(
        model_path=model,
        spec_path=spec,
        manifest_path=manifest,
        output_dir=tmp_path / "renders",
        device="ARC",
        report_path=tmp_path / "report.json",
        engine_factory=engine_factory,
    )

    engine = engines[0]
    assert engine.closed is True
    assert len(engine.calls) == 2
    assert np.all(engine.calls[0]["previous"] == 0.5)  # explicit first-frame seed
    assert np.all(engine.calls[1]["previous"] == 0.25)  # first rendered output
    assert report["frames_processed"] == 2
    assert report["frames"][0]["history_source"] == "manifest_initial_history"
    assert report["frames"][1]["history_source"] == "previous_output"
    assert report["frames"][1]["history_reset_reason"] == ""
    assert report["aggregate"]["total_compile_ms"] == 2.0
    assert Image.open(tmp_path / "renders" / "frame_000000.png").getpixel((0, 0)) == (64, 64, 64)
    assert Image.open(tmp_path / "renders" / "frame_000001.png").getpixel((0, 0)) == (128, 128, 128)


def test_sequence_refuses_to_guess_history_after_a_frame_gap(tmp_path):
    model, spec, manifest, first, second, seed = _create_sources(tmp_path)
    _write_manifest(
        manifest,
        [
            {
                "frame_id": 10,
                "timestamp": 1.0,
                "color": first.name,
                "initial_history": seed.name,
                "scene_id": "scene-a",
            },
            {
                "frame_id": 12,
                "timestamp": 1.2,
                "color": second.name,
                "scene_id": "scene-a",
            },
        ],
    )

    engine = FakeSequenceEngine(model, device="ARC")
    with pytest.raises(AdapterSpecError, match="supply initial_history"):
        run_sequence(
            model_path=model,
            spec_path=spec,
            manifest_path=manifest,
            output_dir=tmp_path / "renders",
            engine_factory=lambda *args, **kwargs: engine,
        )
    assert engine.closed is True


def test_sequence_rejects_output_paths_that_would_replace_a_source(tmp_path):
    model = tmp_path / "model.onnx"
    spec = tmp_path / "adapter.json"
    manifest = tmp_path / "frames.jsonl"
    source = tmp_path / "frame_000000.png"
    model.write_bytes(b"fake model")
    _write_spec(spec, needs_history=False)
    Image.new("RGB", (1, 1), (1, 2, 3)).save(source)
    _write_manifest(
        manifest,
        [{"frame_id": 0, "timestamp": 0.0, "color": source.name}],
    )

    def should_not_create_engine(*args, **kwargs):
        pytest.fail("path validation must happen before model initialization")

    with pytest.raises(ValueError, match="overwrite an input file"):
        run_sequence(
            model_path=model,
            spec_path=spec,
            manifest_path=manifest,
            output_dir=tmp_path,
            engine_factory=should_not_create_engine,
        )


def test_sequence_requires_a_declared_rgb_preview_contract(tmp_path):
    model, spec, manifest, first, second, seed = _create_sources(tmp_path)
    spec.write_text(
        json.dumps({
            "version": 1,
            "inputs": {
                "color": {
                    "source": "color",
                    "layout": "NCHW",
                    "source_range": [0, 255],
                    "target_range": [0, 1],
                }
            },
        }),
        encoding="utf-8",
    )

    def should_not_create_engine(*args, **kwargs):
        pytest.fail("the RGB output contract must be checked before model initialization")

    with pytest.raises(AdapterSpecError, match="requires preview_output"):
        run_sequence(
            model_path=model,
            spec_path=spec,
            manifest_path=manifest,
            output_dir=tmp_path / "renders",
            engine_factory=should_not_create_engine,
        )
