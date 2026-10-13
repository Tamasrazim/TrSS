"""CLI tests run without OpenVINO or a physical GPU."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import xess5.cli as cli


def test_parse_named_tensor_preserves_equals_in_file_path():
    name, path = cli._parse_named_tensor(r"depth=C:\capture\frame=01.npy")
    assert name == "depth"
    assert str(path) == r"C:\capture\frame=01.npy"


@pytest.mark.parametrize("argument", ["depth", "=file.npy", "depth="])
def test_parse_named_tensor_requires_name_and_file(argument):
    with pytest.raises(ValueError, match="NAME=FILE.npy|both a name"):
        cli._parse_named_tensor(argument)


def test_safe_tensor_filename_cannot_create_directories():
    assert cli._safe_tensor_filename("../../depth map") == "depth_map"
    assert cli._safe_tensor_filename("***") == "tensor"


def test_output_path_cannot_replace_image_or_model(tmp_path):
    image = tmp_path / "input.png"
    model = tmp_path / "model.onnx"
    output = tmp_path / "result.png"
    image.touch()
    model.touch()

    with pytest.raises(ValueError, match="must differ"):
        cli._ensure_safe_paths(image, model, image, None)
    with pytest.raises(ValueError, match="must differ"):
        cli._ensure_safe_paths(image, model, model, None)
    with pytest.raises(ValueError, match="Report path"):
        cli._ensure_safe_paths(image, model, output, image)


def test_output_path_cannot_replace_openvino_ir_weights(tmp_path):
    image = tmp_path / "input.png"
    model = tmp_path / "model.xml"
    weights = tmp_path / "model.bin"
    image.touch()
    model.touch()
    weights.touch()

    with pytest.raises(ValueError, match="must differ"):
        cli._ensure_safe_paths(image, model, weights, None)


def test_image_inference_obeys_exact_warmup_count_and_writes_report(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    input_image = tmp_path / "input.png"
    output_image = tmp_path / "out" / "output.png"
    report_path = tmp_path / "reports" / "run.json"
    model.write_bytes(b"dummy model file for mocked engine")
    Image.new("RGB", (2, 2), (10, 20, 30)).save(input_image)

    class FakeEngine:
        calls = 0

        def __init__(self, model_path, **kwargs):
            self.model_path = str(Path(model_path).resolve())
            assert kwargs["device"] == "ARC"

        def infer(self, source):
            type(self).calls += 1
            return SimpleNamespace(
                image=Image.new("RGB", source.size, (50, 60, 70)),
                inference_ms=5.0,
                input_size=source.size,
                output_size=source.size,
                device="GPU.1",
                device_name="Intel Arc B580 Graphics",
                model_path=self.model_path,
            )

    monkeypatch.setattr(cli, "ImageInferenceEngine", FakeEngine)
    args = cli._build_parser().parse_args([
        "infer",
        "--model", str(model),
        "--input", str(input_image),
        "--output", str(output_image),
        "--device", "ARC",
        "--warmup", "0",
        "--iterations", "2",
        "--report", str(report_path),
    ])

    assert cli._run_inference(args) == 0
    # One cold/priming call plus exactly two measured calls: no hidden warm-up.
    assert FakeEngine.calls == 3
    assert output_image.is_file()
    assert Image.open(output_image).getpixel((0, 0)) == (50, 60, 70)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["warmup_requested"] == 0
    assert report["warmup_performed"] == 0
    assert report["priming_call_excluded_from_samples"] is True
    assert report["iterations"] == 2
    assert report["device_name"] == "Intel Arc B580 Graphics"


def test_tensor_inference_writes_separate_numpy_outputs_and_report(monkeypatch, tmp_path):
    model = tmp_path / "render.onnx"
    color_path = tmp_path / "color.npy"
    depth_path = tmp_path / "depth.npy"
    output_dir = tmp_path / "outputs"
    report_path = tmp_path / "reports" / "tensor.json"
    model.write_bytes(b"mock model")
    np.save(color_path, np.ones((1, 3, 2, 2), dtype=np.float32), allow_pickle=False)
    np.save(depth_path, np.ones((1, 1, 2, 2), dtype=np.float32), allow_pickle=False)

    class FakeTensorEngine:
        def __init__(self, model_path, *, device):
            self.model_path = str(Path(model_path).resolve())
            assert device == "ARC"
            self.input_names = ("color", "depth")

        def infer(self, inputs):
            return SimpleNamespace(
                outputs={
                    "render/rgb": np.full((1, 3, 2, 2), 0.5, dtype=np.float32),
                    "history blend": np.full((1, 1, 2, 2), 0.75, dtype=np.float32),
                },
                input_shapes={name: tuple(value.shape) for name, value in inputs.items()},
                output_shapes={
                    "render/rgb": (1, 3, 2, 2),
                    "history blend": (1, 1, 2, 2),
                },
                compile_ms=12.0,
                inference_ms=4.0,
                total_ms=4.5,
                device="GPU.1",
                device_name="Intel Arc B580 Graphics",
                model_path=self.model_path,
            )

    monkeypatch.setattr(cli, "TensorInferenceEngine", FakeTensorEngine)
    args = cli._build_parser().parse_args([
        "infer-tensors",
        "--model", str(model),
        "--input", f"color={color_path}",
        "--input", f"depth={depth_path}",
        "--output-dir", str(output_dir),
        "--device", "ARC",
        "--warmup", "1",
        "--iterations", "2",
        "--report", str(report_path),
    ])

    assert cli._run_tensor_inference(args) == 0
    rgb = np.load(output_dir / "render_rgb.npy", allow_pickle=False)
    blend = np.load(output_dir / "history_blend.npy", allow_pickle=False)
    assert np.all(rgb == 0.5)
    assert np.all(blend == 0.75)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["device_name"] == "Intel Arc B580 Graphics"
    assert report["prime_compile_ms"] == 12.0
    assert report["iterations"] == 2
    assert report["outputs"]["render/rgb"]["path"].endswith("render_rgb.npy")


def test_tensor_output_cannot_overwrite_an_input_tensor(monkeypatch, tmp_path):
    model = tmp_path / "render.onnx"
    color_path = tmp_path / "color.npy"
    model.write_bytes(b"mock model")
    np.save(color_path, np.zeros((1, 3, 2, 2), dtype=np.float32), allow_pickle=False)

    class FakeTensorEngine:
        def __init__(self, model_path, *, device):
            self.model_path = str(Path(model_path).resolve())
            self.input_names = ("color",)

        def infer(self, inputs):
            return SimpleNamespace(
                outputs={"color": np.zeros((1, 3, 2, 2), dtype=np.float32)},
                input_shapes={"color": tuple(inputs["color"].shape)},
                output_shapes={"color": (1, 3, 2, 2)},
                compile_ms=0.0,
                inference_ms=1.0,
                total_ms=1.0,
                device="GPU.1",
                device_name="Intel Arc B580 Graphics",
                model_path=self.model_path,
            )

    monkeypatch.setattr(cli, "TensorInferenceEngine", FakeTensorEngine)
    args = cli._build_parser().parse_args([
        "infer-tensors",
        "--model", str(model),
        "--input", f"color={color_path}",
        "--output-dir", str(tmp_path),
        "--warmup", "0",
        "--iterations", "1",
    ])
    with pytest.raises(ValueError, match="would overwrite"):
        cli._run_tensor_inference(args)
