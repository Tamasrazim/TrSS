"""Run an end-to-end RGB-only experiment using local gameplay recordings.

The experiment imports video files or native TrSS desktop-capture sessions, trains
a separate spatial model, evaluates it against bicubic interpolation, and publishes
reports/gallery together. It never attaches to a game process and does not fabricate
renderer auxiliary buffers.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .evaluate_spatial import evaluate_spatial_checkpoint
from .import_game_video import import_game_videos
from .train_spatial import train_spatial


def _replace_paths(value: Any, old: str, new: str) -> Any:
    """Rebase temporary absolute paths after the staged result is promoted."""
    if isinstance(value, str):
        return value.replace(old, new).replace(old.replace("\\", "/"), new.replace("\\", "/"))
    if isinstance(value, list):
        return [_replace_paths(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_paths(item, old, new) for key, item in value.items()}
    return value


def _rebase_json_artifacts(root: Path, staging: Path, output: Path) -> None:
    """Repair paths embedded in machine-readable reports without touching images."""
    old = str(staging)
    new = str(output)
    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not rebase generated JSON report '{path}': {exc}") from exc
        data = _replace_paths(data, old, new)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    for path in root.rglob("*.jsonl"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            data = [json.loads(line) for line in lines if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not rebase generated JSONL manifest '{path}': {exc}") from exc
        data = _replace_paths(data, old, new)
        path.write_text(
            "".join(json.dumps(item, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
                    for item in data),
            encoding="utf-8",
            newline="\n",
        )


def run_gameplay_experiment(
    videos: Iterable[str | Path],
    output_dir: str | Path,
    *,
    game: str = "unspecified",
    sample_every: int = 15,
    max_frames_per_video: int = 120,
    validation_fraction: float = 0.2,
    seed: int = 20261014,
    device: str = "auto",
    epochs: int = 5,
    channels: int = 32,
    blocks: int = 6,
    patch_size: int = 96,
    tile_size: int = 256,
    tile_overlap: int = 16,
    max_frames: int | None = None,
    max_previews: int = 12,
    force: bool = False,
) -> dict[str, Any]:
    """Import clips, train, evaluate, and publish one self-contained experiment folder."""
    if not isinstance(game, str) or not game.strip():
        raise ValueError("game label must be a non-empty string.")
    game = game.strip()
    if len(game) > 100:
        raise ValueError("game label must be at most 100 characters.")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if max_frames is not None and (
        not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames < 1
    ):
        raise ValueError("max_frames must be a positive integer when supplied.")
    if not isinstance(max_previews, int) or isinstance(max_previews, bool) or max_previews < 1:
        raise ValueError("max_previews must be a positive integer.")

    output = Path(output_dir).expanduser().resolve()
    if output == output.parent or not output.name:
        raise ValueError("output_dir must name a dedicated output folder.")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output}")
    if output.exists() and not force:
        raise FileExistsError(
            f"Output path already exists: {output}. Pass --force to replace it."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    transaction = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.gameplay-experiment-", dir=output.parent)
    )
    staging = transaction / "payload"
    staging.mkdir()
    backup: Path | None = None

    try:
        dataset_dir = staging / "dataset"
        training_dir = staging / "training"
        evaluation_dir = staging / "evaluation"

        capture_report = import_game_videos(
            videos,
            dataset_dir,
            sample_every=sample_every,
            max_frames_per_video=max_frames_per_video,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        training_report = train_spatial(
            dataset_dir / "train.jsonl",
            dataset_dir / "validation.jsonl",
            training_dir,
            device_name=device,
            epochs=epochs,
            channels=channels,
            blocks=blocks,
            patch_size=patch_size,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            seed=seed,
        )
        evaluation_report = evaluate_spatial_checkpoint(
            training_dir / "checkpoint-best.pt",
            dataset_dir / "validation.jsonl",
            evaluation_dir,
            device_name=device,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            max_frames=max_frames,
            max_previews=max_previews,
        )

        report = {
            "schema_version": 1,
            "completed": True,
            "experiment_type": "recorded-gameplay-rgb-only-spatial-upscaling",
            "game": game,
            "output_dir": str(output),
            "dataset": {
                "report": "dataset/capture-report.json",
                "train_manifest": "dataset/train.jsonl",
                "validation_manifest": "dataset/validation.jsonl",
                "training_frames": capture_report["training_frames"],
                "validation_frames": capture_report["validation_frames"],
                "source_count": capture_report["source_count"],
            },
            "training": {
                "report": "training/training-report.json",
                "best_checkpoint": "training/checkpoint-best.pt",
                "device": training_report["device"],
                "epochs_completed": training_report["epochs_completed"],
                "best_validation_mae": training_report["best_validation_mae"],
                "last_epoch": training_report["history"][-1],
            },
            "evaluation": {
                "report": "evaluation/evaluation-report.json",
                "gallery": "evaluation/index.html",
                "evaluated_frames": evaluation_report["evaluated_frames"],
                "subset_evaluation": evaluation_report["subset_evaluation"],
                "metrics": evaluation_report["metrics"],
                "bicubic_metrics": evaluation_report["bicubic_metrics"],
                "beats_bicubic_on_mae": evaluation_report["beats_bicubic_on_mae"],
                "beats_bicubic_on_psnr": evaluation_report["beats_bicubic_on_psnr"],
                "checkpoint_sha256": evaluation_report["checkpoint_sha256"],
            },
            "limits": [
                "Local gameplay videos and native TrSS desktop-capture sessions are read; no game process access or injection occurs.",
                "The RGB target is a recorded video frame, and its input is made by downsampling that frame.",
                "This is a spatial 2x baseline, not temporal reconstruction with game depth, motion, or history.",
                "The metrics describe the supplied recordings and do not prove live-game quality, latency, FPS, or Arc B580 performance.",
                "Keep recordings, extracted frames, and model checkpoints local unless licences allow redistribution.",
            ],
        }

        # Reports inside the staged tree must continue to point to correct paths afterward.
        _rebase_json_artifacts(staging, staging, output)
        (staging / "experiment-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        if output.exists():
            backup = transaction / "previous-output"
            output.rename(backup)
        try:
            staging.rename(output)
        except Exception:
            if backup is not None and backup.exists() and not output.exists():
                backup.rename(output)
                backup = None
            raise

        final_report = json.loads(
            (output / "experiment-report.json").read_text(encoding="utf-8")
        )
        return final_report
    finally:
        shutil.rmtree(transaction, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import game recordings, train TrSS spatial SR, and write a local comparison gallery."
    )
    parser.add_argument(
        "videos", nargs="+",
        help="At least two video recordings or TrSS desktop-capture session folders."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--game", default="unspecified", help="Label recorded footage (for the report).")
    parser.add_argument("--sample-every", type=int, default=15)
    parser.add_argument("--max-frames-per-video", type=int, default=120)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--device", default="auto", help="auto, cpu, xpu, or cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=int, default=16)
    parser.add_argument("--max-frames", type=int, help="Evaluate an evenly spaced subset, if supplied.")
    parser.add_argument("--max-previews", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        report = run_gameplay_experiment(
            args.videos, args.output_dir, game=args.game,
            sample_every=args.sample_every,
            max_frames_per_video=args.max_frames_per_video,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            device=args.device,
            epochs=args.epochs,
            channels=args.channels,
            blocks=args.blocks,
            patch_size=args.patch_size,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            max_frames=args.max_frames,
            max_previews=args.max_previews,
            force=args.force,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    root = args.output_dir.expanduser().resolve()
    print(json.dumps({
        "experiment_report": str(root / "experiment-report.json"),
        "dataset_gallery": str(root / "dataset/review-gallery/index.html"),
        "quality_gallery": str(root / "evaluation/index.html"),
        "best_checkpoint": str(root / "training/checkpoint-best.pt"),
        "game": report["game"],
        "epochs_completed": report["training"]["epochs_completed"],
        "evaluation_metrics": report["evaluation"]["metrics"],
        "beats_bicubic_on_mae": report["evaluation"]["beats_bicubic_on_mae"],
        "beats_bicubic_on_psnr": report["evaluation"]["beats_bicubic_on_psnr"],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
