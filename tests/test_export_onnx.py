"""CPU tests for export to portable ONNX with the exact TrSS compositor."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")

from trss.training.export_onnx import export_checkpoint_to_onnx
from trss.training.model import MODEL_FORMAT, TrSSTemporalSR, compose_prediction


def _checkpoint(path):
    torch.manual_seed(41)
    model = TrSSTemporalSR(channels=8, blocks=1, context_blocks=0).eval()
    torch.save(
        {
            "format": MODEL_FORMAT,
            "config": model.config,
            "state_dict": model.state_dict(),
            "epoch": 3,
        },
        path,
    )
    return model


def test_onnx_export_runs_dynamic_shapes_and_matches_torch_composition(tmp_path):
    checkpoint = tmp_path / "checkpoint-best.pt"
    model = _checkpoint(checkpoint)
    artifact = tmp_path / "trss-composed.onnx"

    report = export_checkpoint_to_onnx(
        checkpoint_path=checkpoint,
        output_path=artifact,
        opset=17,
        dynamic_spatial=True,
    )

    assert report["format"] == "trss-onnx-artifact-v1"
    assert report["checkpoint_epoch"] == 3
    assert report["dynamic_spatial_axes"] is True
    assert len(report["checkpoint_sha256"]) == 64
    assert len(report["onnx_sha256"]) == 64
    assert artifact.is_file()
    sidecar = artifact.with_name(artifact.name + ".json")
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text(encoding="utf-8"))["output"]["name"] == "output_rgb"

    graph = onnx.load(str(artifact))
    onnx.checker.check_model(graph, full_check=True)
    session = onnxruntime.InferenceSession(
        str(artifact),
        providers=["CPUExecutionProvider"],
    )

    # The second size catches accidental fixed spatial dimensions in the exported graph.
    for height, width in ((16, 24), (20, 14)):
        torch.manual_seed(height * 100 + width)
        features = torch.rand(1, 10, height, width, dtype=torch.float32)
        low_res = torch.rand(1, 3, height, width, dtype=torch.float32)
        history = torch.rand(1, 3, height * 2, width * 2, dtype=torch.float32)
        history_valid = torch.rand(1, 1, height * 2, width * 2, dtype=torch.float32)

        with torch.inference_mode():
            expected = compose_prediction(
                model(features), low_res, history, history_valid
            ).cpu().numpy()
        actual = session.run(
            ["output_rgb"],
            {
                "features": features.numpy(),
                "low_res_rgb": low_res.numpy(),
                "history_rgb": history.numpy(),
                "history_valid": history_valid.numpy(),
            },
        )[0]
        assert actual.shape == (1, 3, height * 2, width * 2)
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


def test_onnx_export_protects_existing_artifact_and_sidecar(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint)
    artifact = tmp_path / "already-there.onnx"
    sidecar = artifact.with_name(artifact.name + ".json")
    artifact.write_bytes(b"keep model")
    sidecar.write_text('{"keep": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="Pass --force"):
        export_checkpoint_to_onnx(
            checkpoint_path=checkpoint,
            output_path=artifact,
        )

    assert artifact.read_bytes() == b"keep model"
    assert sidecar.read_text(encoding="utf-8") == '{"keep": true}\n'


def test_onnx_export_rejects_wrong_extension_before_writing(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    _checkpoint(checkpoint)
    output = tmp_path / "model.bin"

    with pytest.raises(ValueError, match="end in .onnx"):
        export_checkpoint_to_onnx(
            checkpoint_path=checkpoint,
            output_path=output,
        )

    assert not output.exists()
