"""Export a trained TrSS checkpoint as a portable ONNX image-composition graph.

The exported graph accepts the ten-channel TrSS feature tensor plus its full-resolution
RGB/history inputs and returns the composed 2x RGB image. It does not capture live
graphics buffers; the caller remains responsible for supplying correctly aligned data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import nn

from .infer import _load_checkpoint
from .model import MODEL_FORMAT, compose_prediction


class _ComposedTrSSExport(nn.Module):
    """Keep model inference and the official TrSS frame compositor in one graph."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        features: torch.Tensor,
        low_res_rgb: torch.Tensor,
        history_rgb: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> torch.Tensor:
        raw_head = self.model(features)
        return compose_prediction(raw_head, low_res_rgb, history_rgb, history_valid)


def _sha256(path: Path) -> str:
    """Hash an artifact in bounded memory so its identity can be recorded."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write readable, finite JSON with stable UTF-8 line endings."""
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def export_checkpoint_to_onnx(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    opset: int = 17,
    dynamic_spatial: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Export a local TrSS checkpoint plus its exact output composition to ONNX."""
    started = perf_counter()
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    metadata_output = Path(str(output) + ".json")

    if output.suffix.lower() != ".onnx":
        raise ValueError("output_path must end in .onnx.")
    if output == checkpoint_file or metadata_output == checkpoint_file:
        raise ValueError("The ONNX output must not overwrite the source checkpoint.")
    if not isinstance(opset, int) or isinstance(opset, bool) or opset < 13:
        raise ValueError("opset must be an integer greater than or equal to 13.")
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_file}")
    if output.exists() and not output.is_file():
        raise IsADirectoryError(f"ONNX output path must be a file: {output}")
    if metadata_output.exists() and not metadata_output.is_file():
        raise IsADirectoryError(f"ONNX metadata path must be a file: {metadata_output}")
    if not force and (output.exists() or metadata_output.exists()):
        raise FileExistsError(
            f"The ONNX artifact or its sidecar already exists for {output}. "
            "Pass --force to replace the pair."
        )

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export dependencies are missing. Install TrSS with "
            "'python -m pip install -e ".[export]" and retry."
        ) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    transaction_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.export-", dir=output.parent)
    )
    staged_onnx = transaction_root / output.name
    staged_metadata = transaction_root / f"{output.name}.json"
    backup_onnx = transaction_root / "previous.onnx"
    backup_metadata = transaction_root / "previous-metadata.json"
    saved_onnx = False
    saved_metadata = False
    installed_onnx = False
    installed_metadata = False

    try:
        model, checkpoint = _load_checkpoint(checkpoint_file)
        model = model.cpu().float().eval()
        exported_model = _ComposedTrSSExport(model).eval()

        # Symbolic spatial axes let one graph handle different input resolutions.
        sample_height, sample_width = 16, 24
        examples = (
            torch.zeros((1, 10, sample_height, sample_width), dtype=torch.float32),
            torch.zeros((1, 3, sample_height, sample_width), dtype=torch.float32),
            torch.zeros((1, 3, sample_height * 2, sample_width * 2), dtype=torch.float32),
            torch.zeros((1, 1, sample_height * 2, sample_width * 2), dtype=torch.float32),
        )
        dynamic_axes = None
        if dynamic_spatial:
            dynamic_axes = {
                "features": {0: "batch", 2: "input_height", 3: "input_width"},
                "low_res_rgb": {0: "batch", 2: "input_height", 3: "input_width"},
                "history_rgb": {0: "batch", 2: "output_height", 3: "output_width"},
                "history_valid": {0: "batch", 2: "output_height", 3: "output_width"},
                "output_rgb": {0: "batch", 2: "output_height", 3: "output_width"},
            }

        torch.onnx.export(
            exported_model,
            examples,
            str(staged_onnx),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["features", "low_res_rgb", "history_rgb", "history_valid"],
            output_names=["output_rgb"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
        graph = onnx.load(str(staged_onnx), load_external_data=True)
        onnx.checker.check_model(graph, full_check=True)

        report: dict[str, Any] = {
            "format": "trss-onnx-artifact-v1",
            "artifact": str(output),
            "metadata": str(metadata_output),
            "checkpoint": str(checkpoint_file),
            "checkpoint_sha256": _sha256(checkpoint_file),
            "onnx_sha256": _sha256(staged_onnx),
            "source_model_format": MODEL_FORMAT,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "model_config": model.config,
            "opset": opset,
            "dynamic_spatial_axes": bool(dynamic_spatial),
            "precision": "float32",
            "scale_factor": 2,
            "inputs": [
                {
                    "name": "features",
                    "shape": ["N", 10, "H", "W"] if dynamic_spatial else [1, 10, sample_height, sample_width],
                    "meaning": (
                        "10 channels: display-RGB, normalized depth, normalized XY motion, "
                        "downsampled reprojected RGB history, and downsampled history-validity."
                    ),
                },
                {
                    "name": "low_res_rgb",
                    "shape": ["N", 3, "H", "W"] if dynamic_spatial else [1, 3, sample_height, sample_width],
                    "meaning": "Current display-encoded RGB in NCHW order and [0, 1].",
                },
                {
                    "name": "history_rgb",
                    "shape": ["N", 3, "2H", "2W"] if dynamic_spatial else [1, 3, sample_height * 2, sample_width * 2],
                    "meaning": "History already reprojected into the current view at output resolution.",
                },
                {
                    "name": "history_valid",
                    "shape": ["N", 1, "2H", "2W"] if dynamic_spatial else [1, 1, sample_height * 2, sample_width * 2],
                    "meaning": "Output-resolution validity mask; zero prevents temporal blending.",
                },
            ],
            "output": {
                "name": "output_rgb",
                "shape": ["N", 3, "2H", "2W"] if dynamic_spatial else [1, 3, sample_height * 2, sample_width * 2],
                "meaning": "Composed RGB frame: bicubic base, network residual, and validity-gated history blend.",
            },
            "input_consistency": (
                "The low_res_rgb values must match features[:, 0:3]; feature channels 6:9 "
                "and 9 must encode the downsampled versions of history_rgb and history_valid."
            ),
            "limits": [
                "This artifact exports the first-party trained model and output compositor only.",
                "It does not acquire live game/desktop color, depth, motion, or temporal-history buffers.",
                "The caller must preserve the documented feature ordering, units, sign conventions, and reprojection.",
                "Export success and ONNX graph validation do not prove image quality, real-time speed, or Arc B580 execution.",
            ],
            "elapsed_seconds": perf_counter() - started,
        }
        _write_json(staged_metadata, report)

        # Keep the previous pair until the complete new graph and sidecar are ready.
        try:
            if output.exists():
                os.replace(output, backup_onnx)
                saved_onnx = True
            if metadata_output.exists():
                os.replace(metadata_output, backup_metadata)
                saved_metadata = True
            os.replace(staged_onnx, output)
            installed_onnx = True
            os.replace(staged_metadata, metadata_output)
            installed_metadata = True
        except Exception:
            if installed_metadata and metadata_output.exists():
                metadata_output.unlink()
            if installed_onnx and output.exists():
                output.unlink()
            if saved_onnx and backup_onnx.exists():
                os.replace(backup_onnx, output)
            if saved_metadata and backup_metadata.exists():
                os.replace(backup_metadata, metadata_output)
            raise

        return report
    finally:
        shutil.rmtree(transaction_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained TrSS checkpoint to ONNX, including scale-2 output "
            "composition. Live game capture is not included."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="destination .onnx file")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--static-shapes",
        action="store_true",
        help="export the example 16x24 input shape only instead of dynamic spatial axes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing ONNX/JSON artifact pair after export succeeds",
    )
    args = parser.parse_args()
    try:
        report = export_checkpoint_to_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            opset=args.opset,
            dynamic_spatial=not args.static_shapes,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, IsADirectoryError, ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "output": report["artifact"],
        "metadata": report["metadata"],
        "opset": report["opset"],
        "dynamic_spatial_axes": report["dynamic_spatial_axes"],
        "onnx_sha256": report["onnx_sha256"],
        "limits": report["limits"],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
