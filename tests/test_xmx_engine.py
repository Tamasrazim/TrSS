"""CPU-only tests for the native XMX bridge using a small fake device runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import xess5.backends.xmx_engine as xmx_module
from xess5.backends.xmx_engine import XMXBackendUnavailable, XMXFrameEngine


def fake_xmx_modules(device_name: str = "Intel Arc B580 Graphics"):
    class FakeRuntime:
        def __init__(self):
            self.lib = SimpleNamespace(
                xmx_device=lambda: device_name.encode("utf-8")
            )
            self.buffer_bytes = 4096

    class FakeDeviceWeights:
        def __init__(self, _runtime, weights):
            self.weights = weights
            self.closed = False

        def close(self):
            self.closed = True

    class FakeResidentFrame:
        instances = []

        def __init__(self, _runtime, _weights, height, width):
            self.height = height
            self.width = width
            self.closed = False
            self.run_count = 0
            self.last_execution = None
            self.instances.append(self)

        def run(self, features, *, execution):
            assert features.shape == (self.height, self.width, 16)
            self.run_count += 1
            self.last_execution = execution
            return np.zeros((self.height, self.width, 4), dtype=np.float32)

        def close(self):
            self.closed = True

    model_module = SimpleNamespace(
        load_logical=lambda _path: (
            {"test.weight": np.ones((1,), dtype=np.float32)},
            {"fully_logical": "true", "format": "safetensors"},
        )
    )
    runner = SimpleNamespace(
        DeviceWeights=FakeDeviceWeights,
        ResidentFrame=FakeResidentFrame,
    )
    modules = SimpleNamespace(
        runtime_module=SimpleNamespace(Runtime=FakeRuntime),
        model_module=model_module,
        frame_runner=runner,
    )
    return modules, FakeResidentFrame


def make_engine(monkeypatch, tmp_path: Path, device_name: str = "Intel Arc B580 Graphics"):
    weights = tmp_path / "logical-weights.safetensors"
    weights.write_bytes(b"fake test file")
    modules, fake_frame = fake_xmx_modules(device_name)
    monkeypatch.setattr(xmx_module, "_load_vendor_modules", lambda: modules)
    return XMXFrameEngine(weights), fake_frame


def test_xmx_frame_engine_runs_and_crops_the_four_channel_head(monkeypatch, tmp_path):
    engine, fake_frame = make_engine(monkeypatch, tmp_path)
    color = np.full((3, 4, 3), 0.25, dtype=np.float32)

    try:
        result = engine.process_frame(color, frame_index=7)

        assert result.image.shape == (3, 4, 3)
        assert result.history.shape == (3, 4, 3)
        assert result.head.shape == (3, 4, 4)
        assert np.allclose(result.image, color)
        assert np.allclose(result.history, color)
        assert result.device_name == "Intel Arc B580 Graphics"
        assert result.network_size[0] >= color.shape[1]
        assert result.network_size[1] >= color.shape[0]
        assert result.allocated_buffer_bytes == 4096
        assert result.total_ms >= result.frame_runner_ms >= 0
        assert fake_frame.instances[0].last_execution == "replay"
    finally:
        engine.close()


def test_xmx_frame_engine_reuses_the_frame_graph_for_the_same_network_size(monkeypatch, tmp_path):
    engine, fake_frame = make_engine(monkeypatch, tmp_path)
    color = np.full((3, 4, 3), 0.4, dtype=np.float32)

    try:
        engine.process_frame(color, frame_index=1)
        engine.process_frame(color, frame_index=2)

        assert len(fake_frame.instances) == 1
        assert fake_frame.instances[0].run_count == 2
    finally:
        engine.close()

    assert fake_frame.instances[0].closed is True


def test_xmx_frame_engine_rejects_a_non_arc_adapter(monkeypatch, tmp_path):
    weights = tmp_path / "logical-weights.safetensors"
    weights.write_bytes(b"fake test file")
    modules, _ = fake_xmx_modules("Intel Graphics")
    monkeypatch.setattr(xmx_module, "_load_vendor_modules", lambda: modules)

    with pytest.raises(XMXBackendUnavailable, match="non-Arc device"):
        XMXFrameEngine(weights)


def test_xmx_frame_engine_rejects_missing_weights_before_loading_runtime(tmp_path):
    with pytest.raises(FileNotFoundError, match="Logical model weights"):
        XMXFrameEngine(tmp_path / "missing.safetensors")
