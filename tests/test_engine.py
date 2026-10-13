"""Hardware-independent tests for the model contract and inference lifecycle."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import xess5.engine as engine_module
from xess5.backends.openvino_backend import BackendUnavailable, resolve_device
from xess5.engine import ImageInferenceEngine, ModelContractError, inspect_model


class FakeDimension:
    """Small stand-in for an OpenVINO dimension."""

    def __init__(self, value: int | None):
        self.value = value

    @property
    def is_static(self) -> bool:
        return self.value is not None

    @property
    def is_dynamic(self) -> bool:
        return self.value is None

    def get_length(self) -> int:
        assert self.value is not None
        return self.value

    def __str__(self) -> str:
        return str(self.value) if self.value is not None else "?"


class FakeShape(list):
    """OpenVINO-like partial shape with readable output."""

    def __str__(self) -> str:
        return "[" + ", ".join(str(item) for item in self) + "]"


class FakePort:
    def __init__(self, shape: list[int | None], name: str, precision: str = "f32"):
        self._dims = [FakeDimension(value) for value in shape]
        self._name = name
        self.element_type = precision

    @property
    def partial_shape(self) -> FakeShape:
        return FakeShape(self._dims)

    def get_any_name(self) -> str:
        return self._name


class FakeModel:
    def __init__(
        self,
        inputs: list[list[int | None]] | None = None,
        outputs: list[list[int | None]] | None = None,
    ):
        inputs = inputs or [[1, 3, None, None]]
        outputs = outputs or [[1, 3, None, None]]
        self.inputs = [FakePort(shape, f"input_{index}") for index, shape in enumerate(inputs)]
        self.outputs = [FakePort(shape, f"output_{index}") for index, shape in enumerate(outputs)]

    def clone(self) -> "FakeModel":
        return copy.deepcopy(self)

    def reshape(self, port_shapes: dict[FakePort, list[int]]) -> None:
        for port, shape in port_shapes.items():
            port._dims = [FakeDimension(value) for value in shape]


class FakeOutputTensor:
    def __init__(self, data: np.ndarray):
        self.data = data


class FakeInferRequest:
    def __init__(self):
        self.output: np.ndarray | None = None

    def infer(self, inputs: dict[int, np.ndarray]) -> None:
        # Identity inference is enough to verify tensor layouts and engine plumbing.
        self.output = np.array(inputs[0], copy=True)

    def get_output_tensor(self, _index: int) -> FakeOutputTensor:
        assert self.output is not None
        return FakeOutputTensor(self.output)


class FakeCompiledModel:
    def create_infer_request(self) -> FakeInferRequest:
        return FakeInferRequest()


class FakeCore:
    def __init__(
        self,
        names: dict[str, str] | None = None,
        model: FakeModel | None = None,
    ):
        self.names = names or {
            "CPU": "Intel CPU",
            "GPU.0": "Intel Graphics",
            "GPU.1": "Intel Arc B580 Graphics",
        }
        self.model = model or FakeModel()
        self.available_devices = list(self.names)
        self.compiled_devices: list[str] = []
        self.compiled_models: list[FakeModel] = []

    def get_property(self, device: str, property_name: str) -> str:
        if property_name == "FULL_DEVICE_NAME":
            return self.names[device]
        if property_name == "DEVICE_TYPE":
            return "GPU" if device.startswith("GPU") else "CPU"
        raise KeyError(property_name)

    def read_model(self, _path: str) -> FakeModel:
        return self.model.clone()

    def compile_model(self, model: FakeModel, device: str) -> FakeCompiledModel:
        self.compiled_devices.append(device)
        self.compiled_models.append(model.clone())
        return FakeCompiledModel()


class FakeOpenVINO:
    def __init__(self, core: FakeCore):
        self._core = core

    def Core(self) -> FakeCore:
        return self._core


def install_fake_openvino(monkeypatch, tmp_path: Path, core: FakeCore) -> Path:
    model_file = tmp_path / "model.xml"
    model_file.write_text("<test-model />", encoding="utf-8")
    monkeypatch.setattr(engine_module, "load_openvino", lambda: FakeOpenVINO(core))
    return model_file


def test_arc_alias_selects_arc_not_integrated_gpu():
    core = FakeCore()
    assert resolve_device(core, "ARC") == "GPU.1"


def test_arc_alias_refuses_ambiguous_arc_adapters():
    core = FakeCore(names={
        "GPU.0": "Intel Arc A770",
        "GPU.1": "Intel Arc B580",
    })
    with pytest.raises(BackendUnavailable, match="More than one Intel Arc"):
        resolve_device(core, "ARC")


def test_arc_alias_fails_instead_of_selecting_a_non_arc_gpu():
    core = FakeCore(names={
        "GPU.0": "Intel Graphics",
        "GPU.1": "NVIDIA GeForce",
    })
    with pytest.raises(BackendUnavailable, match="No Intel Arc"):
        resolve_device(core, "ARC")


def test_explicit_gpu_id_can_be_selected():
    assert resolve_device(FakeCore(), "GPU.0") == "GPU.0"


def test_cpu_fallback_is_refused():
    with pytest.raises(BackendUnavailable, match="refuses CPU fallback"):
        resolve_device(FakeCore(), "CPU")


def test_inspect_model_returns_names_shapes_and_precision(monkeypatch, tmp_path):
    core = FakeCore(model=FakeModel(
        inputs=[[1, 3, 224, 320]],
        outputs=[[1, 3, 448, 640]],
    ))
    model_file = install_fake_openvino(monkeypatch, tmp_path, core)

    summary = inspect_model(model_file)

    assert summary["input_count"] == 1
    assert summary["output_count"] == 1
    assert summary["inputs"][0]["shape"] == [1, 3, 224, 320]
    assert summary["outputs"][0]["shape"] == [1, 3, 448, 640]
    assert summary["inputs"][0]["precision"] == "f32"
    assert summary["rank_four_single_input_output_candidate"] is True


def test_engine_runs_identity_model_and_reports_actual_arc_device(monkeypatch, tmp_path):
    core = FakeCore()
    model_file = install_fake_openvino(monkeypatch, tmp_path, core)
    engine = ImageInferenceEngine(model_file, device="ARC")

    source = Image.new("RGB", (6, 4), (12, 34, 56))
    result = engine.infer(source)

    assert core.compiled_devices == ["GPU.1"]
    assert result.device == "GPU.1"
    assert result.device_name == "Intel Arc B580 Graphics"
    assert result.input_size == (6, 4)
    assert result.output_size == (6, 4)
    assert result.image.getpixel((0, 0)) == (12, 34, 56)
    assert result.inference_ms >= 0
    # Repeated frames should reuse the compiled request instead of recreating it.
    first_request = engine._request
    engine.infer(source)
    assert engine._request is first_request


def test_engine_resizes_to_fixed_model_input_shape(monkeypatch, tmp_path):
    core = FakeCore(model=FakeModel(
        inputs=[[1, 3, 2, 2]],
        outputs=[[1, 3, 2, 2]],
    ))
    model_file = install_fake_openvino(monkeypatch, tmp_path, core)
    engine = ImageInferenceEngine(model_file, device="GPU.1")

    result = engine.infer(Image.new("RGB", (8, 6), (100, 80, 60)))

    assert result.input_size == (2, 2)
    assert result.output_size == (2, 2)


def test_engine_recompiles_dynamic_model_for_new_resolution(monkeypatch, tmp_path):
    core = FakeCore(model=FakeModel())
    model_file = install_fake_openvino(monkeypatch, tmp_path, core)
    engine = ImageInferenceEngine(model_file, device="GPU.1")

    engine.infer(Image.new("RGB", (8, 6), (20, 30, 40)))
    first_request = engine._request
    engine.infer(Image.new("RGB", (10, 7), (20, 30, 40)))

    assert engine._request is not first_request
    assert core.compiled_devices == ["GPU.1", "GPU.1"]
    assert core.compiled_models[0].inputs[0].partial_shape[2].get_length() == 6
    assert core.compiled_models[1].inputs[0].partial_shape[2].get_length() == 7


def test_engine_rejects_multi_input_model_until_adapter_exists(monkeypatch, tmp_path):
    core = FakeCore(model=FakeModel(inputs=[
        [1, 3, 224, 224],
        [1, 2, 224, 224],
    ]))
    model_file = install_fake_openvino(monkeypatch, tmp_path, core)

    with pytest.raises(ModelContractError, match="exactly one model input"):
        ImageInferenceEngine(model_file, device="GPU.1")


def test_engine_validates_numeric_ranges_before_loading_backend(monkeypatch, tmp_path):
    model_file = tmp_path / "model.xml"
    model_file.write_text("<test-model />", encoding="utf-8")

    with pytest.raises(ModelContractError, match="input range"):
        ImageInferenceEngine(model_file, input_range=(1.0, 1.0))



def test_inspect_model_does_not_call_non_rank_four_models_image_candidates(monkeypatch, tmp_path):
    core = FakeCore(model=FakeModel(
        inputs=[[1, 3, 224, 224], [1, 1, 224, 224]],
        outputs=[[1, 3, 224, 224]],
    ))
    model_file = install_fake_openvino(monkeypatch, tmp_path, core)

    summary = inspect_model(model_file)

    assert summary["rank_four_single_input_output_candidate"] is False
    assert "does not prove" in summary["note"]
