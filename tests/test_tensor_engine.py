"""Tests for multi-input model inference without requiring a physical GPU."""

from __future__ import annotations

import numpy as np
import pytest

from xess5.engine import ModelContractError
from xess5.tensor_engine import TensorInferenceEngine
from test_engine import (
    FakeCore,
    FakeModel,
    FakeOpenVINO,
    FakeOutputTensor,
    install_fake_openvino,
)
import xess5.tensor_engine as tensor_engine_module


def install_tensor_openvino(monkeypatch, tmp_path, core):
    # Patch the backend symbol imported by TensorInferenceEngine itself.
    model_file = install_fake_openvino(monkeypatch, tmp_path, core)
    monkeypatch.setattr(
        tensor_engine_module, "load_openvino", lambda: FakeOpenVINO(core)
    )
    return model_file


class SumInferRequest:
    def __init__(self):
        self.outputs: list[np.ndarray] = []

    def infer(self, inputs: dict[int, np.ndarray]) -> None:
        # Broadcast a one-channel depth tensor into an RGB color tensor.
        self.outputs = [np.array(inputs[0] + inputs[1], copy=True)]

    def get_output_tensor(self, index: int) -> FakeOutputTensor:
        return FakeOutputTensor(self.outputs[index])


class SumCompiledModel:
    def create_infer_request(self) -> SumInferRequest:
        return SumInferRequest()


class MultiInputCore(FakeCore):
    def compile_model(self, model: FakeModel, device: str) -> SumCompiledModel:
        self.compiled_devices.append(device)
        self.compiled_models.append(model.clone())
        return SumCompiledModel()


def test_multi_input_model_binds_named_tensors_and_returns_outputs(monkeypatch, tmp_path):
    core = MultiInputCore(model=FakeModel(
        inputs=[
            [1, 3, None, None],
            [1, 1, None, None],
        ],
        outputs=[[1, 3, None, None]],
    ))
    model_file = install_tensor_openvino(monkeypatch, tmp_path, core)
    engine = TensorInferenceEngine(model_file, device="ARC")

    assert engine.input_names == ("input_0", "input_1")
    assert engine.output_names == ("output_0",)

    color = np.full((1, 3, 4, 6), 0.25, dtype=np.float32)
    depth = np.full((1, 1, 4, 6), 0.50, dtype=np.float32)
    result = engine.infer({"input_0": color, "input_1": depth})

    assert result.device == "GPU.1"
    assert result.input_shapes == {
        "input_0": (1, 3, 4, 6),
        "input_1": (1, 1, 4, 6),
    }
    assert result.output_shapes == {"output_0": (1, 3, 4, 6)}
    assert result.outputs["output_0"][0, 0, 0, 0] == pytest.approx(0.75)
    assert result.inference_ms >= 0
    assert result.total_ms >= result.inference_ms


def test_multi_input_engine_reports_missing_and_extra_inputs(monkeypatch, tmp_path):
    core = MultiInputCore(model=FakeModel(inputs=[
        [1, 3, None, None],
        [1, 1, None, None],
    ]))
    model_file = install_tensor_openvino(monkeypatch, tmp_path, core)
    engine = TensorInferenceEngine(model_file, device="GPU.1")

    with pytest.raises(ModelContractError, match="missing inputs"):
        engine.infer({"input_0": np.zeros((1, 3, 4, 4), dtype=np.float32)})

    with pytest.raises(ModelContractError, match="unexpected inputs"):
        engine.infer({
            "input_0": np.zeros((1, 3, 4, 4), dtype=np.float32),
            "input_1": np.zeros((1, 1, 4, 4), dtype=np.float32),
            "depth_extra": np.zeros((1, 1, 4, 4), dtype=np.float32),
        })


def test_multi_input_engine_reuses_compilation_for_same_shapes(monkeypatch, tmp_path):
    core = MultiInputCore(model=FakeModel(inputs=[
        [1, 3, None, None],
        [1, 1, None, None],
    ]))
    model_file = install_tensor_openvino(monkeypatch, tmp_path, core)
    engine = TensorInferenceEngine(model_file, device="GPU.1")

    inputs = {
        "input_0": np.ones((1, 3, 4, 4), dtype=np.float32),
        "input_1": np.zeros((1, 1, 4, 4), dtype=np.float32),
    }
    first = engine.infer(inputs)
    second = engine.infer(inputs)

    assert first.compile_ms > 0
    assert second.compile_ms == 0
    assert core.compiled_devices == ["GPU.1"]


def test_multi_input_engine_recompiles_after_resolution_change(monkeypatch, tmp_path):
    core = MultiInputCore(model=FakeModel(inputs=[
        [1, 3, None, None],
        [1, 1, None, None],
    ]))
    model_file = install_tensor_openvino(monkeypatch, tmp_path, core)
    engine = TensorInferenceEngine(model_file, device="GPU.1")

    engine.infer({
        "input_0": np.ones((1, 3, 4, 4), dtype=np.float32),
        "input_1": np.zeros((1, 1, 4, 4), dtype=np.float32),
    })
    engine.infer({
        "input_0": np.ones((1, 3, 6, 8), dtype=np.float32),
        "input_1": np.zeros((1, 1, 6, 8), dtype=np.float32),
    })

    assert len(core.compiled_models) == 2



def test_multi_input_engine_rejects_complex_input_without_coercing(monkeypatch, tmp_path):
    core = MultiInputCore(model=FakeModel(inputs=[
        [1, 3, None, None],
        [1, 1, None, None],
    ]))
    model_file = install_tensor_openvino(monkeypatch, tmp_path, core)
    engine = TensorInferenceEngine(model_file, device="GPU.1")

    with pytest.raises(ModelContractError, match="real-valued numeric"):
        engine.infer({
            "input_0": np.ones((1, 3, 4, 4), dtype=np.complex64),
            "input_1": np.zeros((1, 1, 4, 4), dtype=np.float32),
        })
