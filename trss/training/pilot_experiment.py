"""Run a complete, explicitly synthetic TrSS pilot experiment in one command.

This path proves that controlled data can travel through training, held-out
quality evaluation, ONNX export, and OpenVINO inference. The pilot uses a CPU
box-scene renderer and is not evidence of live-game quality or Arc performance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

from .evaluate_checkpoint import evaluate_checkpoint
from .export_onnx import export_checkpoint_to_onnx
from .infer_onnx import run_onnx_frame
from .prepare_pilot import prepare_pilot_dataset, _rewrite_staging_paths


def _write_json(path: Path, report: dict[str, Any]) -> None:
    """Save a report without NaN/Infinity or platform-specific text encoding."""
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _last_lines(text: str, limit: int = 30) -> str:
    """Keep a useful but bounded error excerpt if the training subprocess fails."""
    return "\n".join(text.splitlines()[-limit:])


def run_pilot_experiment(
    output_dir: str | Path,
    *,
    scenes: int = 4,
    frames_per_scene: int = 3,
    width: int = 64,
    height: int = 36,
    seed: int = 20261014,
    validation_fraction: float = 0.25,
    epochs: int = 2,
    train_device: str = "cpu",
    evaluation_device: str = "cpu",
    inference_device: str = "CPU",
    channels: int = 16,
    blocks: int = 2,
    context_blocks: int = 1,
    patch_size: int = 16,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    motion_scale_px: float = 64.0,
    amp: bool = False,
    max_review_frames: int = 12,
    tile_size: int = 32,
    tile_overlap: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    """Generate, train, evaluate, export, and execute one local TrSS pilot."""
    output = Path(output_dir).expanduser().resolve()
    if output == output.parent or output.name in ("", ".", ".."):
        raise ValueError("output_dir must name a dedicated output folder.")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path must be a directory: {output}")
    if output.exists() and not force:
        raise FileExistsError(
            f"Output path already exists: {output}. Pass --force to replace it."
        )
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if patch_size < 2:
        raise ValueError("patch_size must be at least 2.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and greater than zero.")
    if not math.isfinite(motion_scale_px) or motion_scale_px <= 0:
        raise ValueError("motion_scale_px must be finite and greater than zero.")
    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    if tile_size < 2 or tile_overlap < 0 or tile_overlap * 2 >= tile_size:
        raise ValueError("tile_size must be >= 2 and tile_overlap less than half tile_size.")
    if max_review_frames < 1:
        raise ValueError("max_review_frames must be at least 1.")
    if not train_device.strip() or not evaluation_device.strip() or not inference_device.strip():
        raise ValueError("Device selections cannot be empty.")

    output.parent.mkdir(parents=True, exist_ok=True)
    transaction_root = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.experiment-", dir=output.parent
    ))
    staging = transaction_root / "payload"
    staging.mkdir()
    backup: Path | None = None
    promoted = False
    stage_seconds: dict[str, float] = {}

    try:
        dataset_dir = staging / "dataset"
        training_dir = staging / "training"
        evaluation_dir = staging / "evaluation"
        artifacts_dir = staging / "artifacts"
        inference_dir = staging / "openvino-smoke"
        artifacts_dir.mkdir()

        stage_started = perf_counter()
        dataset_report = prepare_pilot_dataset(
            dataset_dir,
            scenes=scenes,
            frames_per_scene=frames_per_scene,
            width=width,
            height=height,
            seed=seed,
            validation_fraction=validation_fraction,
            max_review_frames=max_review_frames,
            motion_scale_px=motion_scale_px,
        )
        stage_seconds["dataset_generation_split_audit_review"] = perf_counter() - stage_started
        if not dataset_report.get("passed"):
            raise RuntimeError("The pilot dataset failed its audit; see dataset/pipeline-report.json.")

        # Call the exact trainer CLI the user can run independently. Preserve its
        # stdout/stderr in the output so this one-command path is diagnosable.
        train_command = [
            sys.executable, "-m", "trss.training.train",
            "--train-manifest", str(dataset_dir / "train.jsonl"),
            "--validation-manifest", str(dataset_dir / "validation.jsonl"),
            "--output-dir", str(training_dir),
            "--device", train_device,
            "--channels", str(channels),
            "--blocks", str(blocks),
            "--context-blocks", str(context_blocks),
            "--patch-size", str(patch_size),
            "--batch-size", str(batch_size),
            "--epochs", str(epochs),
            "--seed", str(seed),
            "--motion-scale-px", str(motion_scale_px),
            "--learning-rate", str(learning_rate),
        ]
        if amp:
            train_command.append("--amp")

        stage_started = perf_counter()
        training_process = subprocess.run(
            train_command,
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        stage_seconds["training"] = perf_counter() - stage_started
        training_log = (
            "COMMAND: " + " ".join(train_command) + "\n\n"
            + "===== STDOUT =====\n" + training_process.stdout
            + "\n===== STDERR =====\n" + training_process.stderr
        )
        (staging / "training.log").write_text(training_log, encoding="utf-8", newline="\n")
        if training_process.returncode != 0:
            excerpt = _last_lines(training_process.stdout + "\n" + training_process.stderr)
            raise RuntimeError(
                f"Training failed with exit code {training_process.returncode}. "
                f"Last output lines:\n{excerpt}"
            )

        checkpoint = training_dir / "checkpoint-best.pt"
        training_report_path = training_dir / "training-report.json"
        if not checkpoint.is_file() or not training_report_path.is_file():
            raise RuntimeError("Training exited successfully but its report/checkpoint is missing.")
        training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
        if not training_report.get("history"):
            raise RuntimeError("Training completed without recording any epoch metrics.")

        stage_started = perf_counter()
        quality_report = evaluate_checkpoint(
            checkpoint_path=checkpoint,
            manifest_path=dataset_dir / "validation.jsonl",
            output_dir=evaluation_dir,
            device_name=evaluation_device,
            motion_scale_px=motion_scale_px,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
        )
        stage_seconds["held_out_quality_evaluation"] = perf_counter() - stage_started

        stage_started = perf_counter()
        onnx_path = artifacts_dir / "trss-composed.onnx"
        export_report = export_checkpoint_to_onnx(
            checkpoint_path=checkpoint,
            output_path=onnx_path,
        )
        stage_seconds["onnx_export_and_validation"] = perf_counter() - stage_started

        # Execute the exported graph on an actual validation row. CPU is allowed
        # only when explicitly selected; ARC goes through the strict device resolver.
        stage_started = perf_counter()
        inference_report = run_onnx_frame(
            model_path=onnx_path,
            manifest_path=dataset_dir / "validation.jsonl",
            row_index=0,
            output_path=inference_dir / "frame.png",
            device_name=inference_device,
            motion_scale_px=motion_scale_px,
            warmup=1,
            iterations=3,
        )
        stage_seconds["openvino_exported_graph_smoke"] = perf_counter() - stage_started

        report: dict[str, Any] = {
            "format": "trss-pilot-experiment-v1",
            "experiment_completed": True,
            "output_dir": str(output),
            "experiment_type": "controlled_synthetic_3d_scene",
            "actual_game_capture": False,
            "source_type": "synthetic_controlled_3d_scene",
            "seed": seed,
            "configuration": {
                "scenes": scenes,
                "frames_per_scene": frames_per_scene,
                "input_size": [width, height],
                "target_scale": 2,
                "epochs": epochs,
                "train_device": train_device,
                "evaluation_device": evaluation_device,
                "inference_device": inference_report["device"],
                "channels": channels,
                "blocks": blocks,
                "context_blocks": context_blocks,
                "patch_size": patch_size,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "motion_scale_px": motion_scale_px,
                "amp_bfloat16": amp,
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
            },
            "stages": {
                "dataset": {
                    "passed": dataset_report["passed"],
                    "pipeline_report": "dataset/pipeline-report.json",
                    "training_frames": dataset_report["split"]["training_frames"],
                    "validation_frames": dataset_report["split"]["validation_frames"],
                    "training_scenes": dataset_report["split"]["training_scenes"],
                    "validation_scenes": dataset_report["split"]["validation_scenes"],
                },
                "training": {
                    "passed": True,
                    "report": "training/training-report.json",
                    "checkpoint": "training/checkpoint-best.pt",
                    "completed_epochs": len(training_report["history"]),
                    "last_epoch": training_report["history"][-1],
                },
                "quality_evaluation": {
                    "passed": quality_report["evaluation_completed"],
                    "report": "evaluation/evaluation-report.json",
                    "gallery": "evaluation/index.html",
                    "metrics": quality_report["metrics"],
                    "beats_bicubic_on_mae": quality_report["beats_bicubic_on_mae"],
                    "beats_bicubic_on_psnr": quality_report["beats_bicubic_on_psnr"],
                    "checkpoint_sha256": quality_report["checkpoint_sha256"],
                },
                "onnx_export": {
                    "passed": True,
                    "artifact": "artifacts/trss-composed.onnx",
                    "metadata": "artifacts/trss-composed.onnx.json",
                    "onnx_sha256": export_report["onnx_sha256"],
                    "opset": export_report["opset"],
                    "dynamic_spatial_axes": export_report["dynamic_spatial_axes"],
                },
                "openvino_smoke": {
                    "passed": True,
                    "output": "openvino-smoke/frame.png",
                    "report": "openvino-smoke/frame.png.json",
                    "device": inference_report["device"],
                    "device_name": inference_report["device_name"],
                    "compile_ms": inference_report["compile_ms"],
                    "mean_inference_ms": inference_report["mean_inference_ms"],
                    "output_size": inference_report["output_size"],
                    "live_capture_used": inference_report["live_capture_used"],
                },
            },
            "stage_seconds": stage_seconds,
            "limits": [
                "The generated images are synthetic box scenes, not photorealistic game capture.",
                "The training run proves the optimizer/checkpoint path, not generalization.",
                "Quality metrics apply only to held-out scenes from this synthetic pilot.",
                "ONNX/OpenVINO success does not prove Arc B580 execution unless the report names that device.",
                "This is offline inference and does not hook a live game or measure in-game FPS.",
            ],
        }
        _write_json(staging / "experiment-report.json", report)

        # Rewrite staged absolute references before promoting the complete run.
        _rewrite_staging_paths(staging, output)
        if output.exists():
            backup = transaction_root / "previous-output"
            output.rename(backup)
        try:
            staging.rename(output)
            promoted = True
        except Exception:
            if backup is not None and backup.exists() and not output.exists():
                backup.rename(output)
                backup = None
            raise

        final_report = json.loads(
            (output / "experiment-report.json").read_text(encoding="utf-8")
        )
        final_report["output_dir"] = str(output)
        return final_report
    finally:
        # Existing user output is restored by the promotion rollback above; on a
        # failed new run, only temporary files are discarded.
        shutil.rmtree(transaction_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded synthetic TrSS experiment: generate, train, evaluate, export ONNX, "
            "then execute one offline frame through OpenVINO. Not live-game capture."
        )
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--frames-per-scene", type=int, default=3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-device", default="cpu", help="cpu, xpu, cuda, or auto")
    parser.add_argument("--evaluation-device", default="cpu", help="cpu, xpu, cuda, or auto")
    parser.add_argument("--inference-device", default="CPU", help="CPU for debugging, ARC, or GPU.X")
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--context-blocks", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--motion-scale-px", type=float, default=64.0)
    parser.add_argument("--amp", action="store_true", help="use BF16 autocast when supported")
    parser.add_argument("--max-review-frames", type=int, default=12)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--tile-overlap", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="replace an existing output after success")
    args = parser.parse_args()

    try:
        report = run_pilot_experiment(
            args.output_dir,
            scenes=args.scenes,
            frames_per_scene=args.frames_per_scene,
            width=args.width,
            height=args.height,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            epochs=args.epochs,
            train_device=args.train_device,
            evaluation_device=args.evaluation_device,
            inference_device=args.inference_device,
            channels=args.channels,
            blocks=args.blocks,
            context_blocks=args.context_blocks,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            motion_scale_px=args.motion_scale_px,
            amp=args.amp,
            max_review_frames=args.max_review_frames,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            force=args.force,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(json.dumps({
        "experiment_report": str(args.output_dir.expanduser().resolve() / "experiment-report.json"),
        "dataset_gallery": str(args.output_dir.expanduser().resolve() / "dataset/review-gallery/index.html"),
        "quality_gallery": str(args.output_dir.expanduser().resolve() / "evaluation/index.html"),
        "onnx_artifact": str(args.output_dir.expanduser().resolve() / "artifacts/trss-composed.onnx"),
        "openvino_output": str(args.output_dir.expanduser().resolve() / "openvino-smoke/frame.png"),
        "experiment_completed": report["experiment_completed"],
        "quality_metrics": report["stages"]["quality_evaluation"]["metrics"],
        "inference_device": report["stages"]["openvino_smoke"]["device_name"],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
