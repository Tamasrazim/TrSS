"""CLI tests run without OpenVINO or a physical GPU."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import trss.cli as cli


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



def test_render_frame_adapts_color_and_depth_and_reports_timings(monkeypatch, tmp_path):
    model = tmp_path / "render.onnx"
    spec = tmp_path / "adapter.json"
    color_path = tmp_path / "frame.png"
    depth_path = tmp_path / "depth.npy"
    output_dir = tmp_path / "output"
    preview_path = tmp_path / "rendered.png"
    history_path = tmp_path / "next-history.npy"
    report_path = tmp_path / "report" / "frame.json"

    model.write_bytes(b"mock model for this CLI test")
    Image.new("RGB", (2, 2), (255, 0, 0)).save(color_path)
    np.save(depth_path, np.full((2, 2), 0.75, dtype=np.float32), allow_pickle=False)
    spec.write_text(json.dumps({
        "version": 1,
        "inputs": {
            "color": {
                "source": "color",
                "layout": "NCHW",
                "source_range": [0, 255],
                "target_range": [0, 1],
            },
            "depth": {
                "source": "depth",
                "layout": "NCHW",
                "source_range": [0, 1],
                "target_range": [0, 1],
                "resize": "nearest",
            },
        },
        "preview_output": {
            "port": "render/rgb",
            "layout": "NCHW",
            "source_range": [0, 1],
            "channel_order": "RGB",
        },
    }), encoding="utf-8")

    class FakeTensorEngine:
        calls = 0

        def __init__(self, model_path, *, device):
            self.model_path = str(Path(model_path).resolve())
            assert device == "ARC"
            self.input_shapes = {
                "color": (1, 3, 2, 2),
                "depth": (1, 1, 2, 2),
            }

        def infer(self, inputs):
            type(self).calls += 1
            assert inputs["color"].shape == (1, 3, 2, 2)
            assert inputs["color"][0, :, 0, 0].tolist() == pytest.approx([1, 0, 0])
            assert np.all(inputs["depth"] == 0.75)
            compile_ms = 9.0 if type(self).calls == 1 else 0.0
            return SimpleNamespace(
                outputs={"render/rgb": np.full((1, 3, 2, 2), 0.3, dtype=np.float32)},
                input_shapes={name: tuple(value.shape) for name, value in inputs.items()},
                output_shapes={"render/rgb": (1, 3, 2, 2)},
                compile_ms=compile_ms,
                inference_ms=4.0,
                total_ms=4.5,
                device="GPU.1",
                device_name="Intel Arc B580 Graphics",
                model_path=self.model_path,
            )

    monkeypatch.setattr(cli, "TensorInferenceEngine", FakeTensorEngine)
    args = cli._build_parser().parse_args([
        "render-frame",
        "--model", str(model),
        "--spec", str(spec),
        "--color", str(color_path),
        "--depth", str(depth_path),
        "--output-dir", str(output_dir),
        "--preview-output", str(preview_path),
        "--history-output", str(history_path),
        "--device", "ARC",
        "--warmup", "1",
        "--iterations", "2",
        "--frame-id", "12",
        "--timestamp", "1.25",
        "--scene-id", "test-scene",
        "--report", str(report_path),
    ])

    assert cli._run_render_frame(args) == 0
    assert FakeTensorEngine.calls == 4  # prime + 1 warm-up + 2 measurements
    saved = np.load(output_dir / "render_rgb.npy", allow_pickle=False)
    assert np.all(saved == 0.3)
    assert Image.open(preview_path).getpixel((0, 0)) == (76, 76, 76)
    history = np.load(history_path, allow_pickle=False)
    assert history.shape == (2, 2, 3)
    assert history.dtype == np.float32
    assert np.all(history == 0.3)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["device_name"] == "Intel Arc B580 Graphics"
    assert report["priming_compile_ms"] == 9.0
    assert report["frame"]["frame_id"] == 12
    assert report["frame"]["scene_id"] == "test-scene"
    assert report["frame"]["available_sources"] == ["color", "depth"]
    assert report["iterations"] == 2
    assert report["output_tensors"]["render/rgb"]["path"].endswith("render_rgb.npy")
    assert report["preview_output"]["path"] == str(preview_path.resolve())
    assert report["history_output"]["range"] == [0, 1]


def test_render_frame_does_not_overwrite_an_input_frame(monkeypatch, tmp_path):
    model = tmp_path / "render.onnx"
    spec = tmp_path / "adapter.json"
    color_path = tmp_path / "frame.png"
    depth_path = tmp_path / "depth.npy"
    model.write_bytes(b"mock model")
    Image.new("RGB", (1, 1), (20, 30, 40)).save(color_path)
    np.save(depth_path, np.ones((1, 1), dtype=np.float32), allow_pickle=False)
    spec.write_text(json.dumps({
        "version": 1,
        "inputs": {
            "depth": {
                "source": "depth",
                "layout": "NCHW",
                "source_range": [0, 1],
                "target_range": [0, 1],
            },
        },
    }), encoding="utf-8")

    class FakeTensorEngine:
        def __init__(self, model_path, *, device):
            self.model_path = str(Path(model_path).resolve())
            self.input_shapes = {"depth": (1, 1, 1, 1)}

        def infer(self, inputs):
            return SimpleNamespace(
                outputs={"depth": np.zeros((1, 1, 1, 1), dtype=np.float32)},
                input_shapes={"depth": (1, 1, 1, 1)},
                output_shapes={"depth": (1, 1, 1, 1)},
                compile_ms=0.0,
                inference_ms=1.0,
                total_ms=1.0,
                device="GPU.1",
                device_name="Intel Arc B580",
                model_path=self.model_path,
            )

    monkeypatch.setattr(cli, "TensorInferenceEngine", FakeTensorEngine)
    args = cli._build_parser().parse_args([
        "render-frame",
        "--model", str(model),
        "--spec", str(spec),
        "--color", str(color_path),
        "--depth", str(depth_path),
        "--output-dir", str(tmp_path),
        "--warmup", "0",
        "--iterations", "1",
    ])
    # The model port is named "depth", so the generated depth.npy output would
    # otherwise replace the exact .npy file that supplied the source depth data.
    with pytest.raises(ValueError, match="would overwrite"):
        cli._run_render_frame(args)



def test_hnet_frame_command_runs_one_model_contract_and_writes_outputs(monkeypatch, tmp_path):
    model = tmp_path / "hnet.onnx"
    color_path = tmp_path / "frame.png"
    history_input = tmp_path / "previous.npy"
    output_path = tmp_path / "output" / "frame.png"
    history_output = tmp_path / "output" / "history.npy"
    head_output = tmp_path / "output" / "head.npy"
    report_path = tmp_path / "report" / "run.json"

    model.write_bytes(b"mock model")
    Image.new("RGB", (2, 2), (255, 128, 0)).save(color_path)
    np.save(history_input, np.full((2, 2, 3), 0.6, dtype=np.float32), allow_pickle=False)

    class FakeTensorEngine:
        def __init__(self, model_path, *, device):
            self.model_path = str(Path(model_path).resolve())
            assert device == "ARC"

    class FakeHNetEngine:
        calls = 0

        def __init__(self, tensor_engine, *, output_layout):
            assert output_layout == "NCHW"
            self.tensor_engine = tensor_engine

        def process_frame(self, color, *, history, frame_index, controls):
            type(self).calls += 1
            assert color.shape == (2, 2, 3)
            assert color[0, 0].tolist() == pytest.approx([1.0, 128 / 255, 0.0])
            assert history.shape == (2, 2, 3)
            assert np.all(history == 0.6)
            assert frame_index == 7
            assert controls.style_index == 2
            assert controls.tone_strength == pytest.approx(0.75)
            assert controls.structure_strength == pytest.approx(1.25)
            image = np.full((2, 2, 3), 0.25, dtype=np.float32)
            next_history = np.full((2, 2, 3), 0.35, dtype=np.float32)
            head = np.full((320, 320, 4), 0.1, dtype=np.float32)
            return SimpleNamespace(
                image=image,
                history=next_history,
                head=head,
                geometry=SimpleNamespace(network_width=320, network_height=320),
                device="GPU.1",
                device_name="Intel Arc B580 Graphics",
                compile_ms=8.0 if type(self).calls == 1 else 0.0,
                inference_ms=3.0,
                total_ms=3.5,
            )

    monkeypatch.setattr(cli, "TensorInferenceEngine", FakeTensorEngine)
    monkeypatch.setattr(cli, "HNetFrameEngine", FakeHNetEngine)
    args = cli._build_parser().parse_args([
        "hnet-frame",
        "--model", str(model),
        "--color", str(color_path),
        "--output", str(output_path),
        "--history", str(history_input),
        "--history-output", str(history_output),
        "--head-output", str(head_output),
        "--device", "ARC",
        "--output-layout", "NCHW",
        "--frame-index", "7",
        "--style-index", "2",
        "--tone-strength", "0.75",
        "--structure-strength", "1.25",
        "--warmup", "1",
        "--iterations", "2",
        "--report", str(report_path),
    ])

    assert cli._run_hnet_frame(args) == 0
    assert FakeHNetEngine.calls == 4  # priming + warm-up + two measured frames
    assert Image.open(output_path).getpixel((0, 0)) == (64, 64, 64)
    saved_history = np.load(history_output, allow_pickle=False)
    assert saved_history.shape == (2, 2, 3)
    assert saved_history.dtype == np.float32
    assert np.all(saved_history == 0.35)
    saved_head = np.load(head_output, allow_pickle=False)
    assert saved_head.shape == (320, 320, 4)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["device_name"] == "Intel Arc B580 Graphics"
    assert report["priming_compile_ms"] == 8.0
    assert report["feature_contract"]["frame_index"] == 7
    assert report["feature_contract"]["style_index"] == 2
    assert report["feature_contract"]["history_supplied"] is True
    assert report["iterations"] == 2
    assert report["outputs"]["history"] == str(history_output.resolve())
    assert report["outputs"]["raw_head"] == str(head_output.resolve())


def test_hnet_frame_refuses_output_overwriting_source_or_model(tmp_path):
    model = tmp_path / "hnet.onnx"
    color = tmp_path / "frame.png"
    model.write_bytes(b"mock model")
    Image.new("RGB", (1, 1)).save(color)

    with pytest.raises(ValueError, match="overwrite"):
        cli._safe_hnet_paths(model, color, color, None, None, None, None)
    with pytest.raises(ValueError, match="overwrite"):
        cli._safe_hnet_paths(model, color, model, None, None, None, None)
    with pytest.raises(ValueError, match="must use the .npy"):
        cli._safe_hnet_paths(
            model, color, tmp_path / "out.png", None,
            tmp_path / "history.png", None, None
        )


def test_hnet_frame_parser_rejects_invalid_style_index(monkeypatch, tmp_path):
    model = tmp_path / "hnet.onnx"
    color = tmp_path / "frame.png"
    model.write_bytes(b"model")
    Image.new("RGB", (1, 1)).save(color)
    args = cli._build_parser().parse_args([
        "hnet-frame",
        "--model", str(model),
        "--color", str(color),
        "--output", str(tmp_path / "out.png"),
        "--style-index", "128",
    ])
    with pytest.raises(ValueError, match="between 0 and 127"):
        cli._run_hnet_frame(args)
