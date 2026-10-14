"""Run the controlled TrSS pilot dataset pipeline with one command.

The pipeline generates explicitly synthetic 3D scene data, creates scene-disjoint
training/validation manifests, audits the resulting split, and builds a local
HTML gallery. It does not capture a running game or claim photorealistic quality.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .audit_dataset import audit_manifests
from .generate_scene_dataset import generate_scene_dataset
from .review_dataset import build_review_gallery
from .split_manifest import split_manifest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON report with finite numbers and predictable UTF-8 encoding."""
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _rewrite_staging_paths(root: Path, final_root: Path) -> None:
    """Fix absolute report/gallery references after the staging folder is renamed."""
    old_prefix = str(root.resolve())
    new_prefix = str(final_root.resolve())
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in (".json", ".jsonl", ".html"):
            continue
        content = path.read_text(encoding="utf-8")
        updated = content.replace(old_prefix, new_prefix)
        if updated != content:
            path.write_text(updated, encoding="utf-8", newline="\n")


def prepare_pilot_dataset(
    output_dir: str | Path,
    *,
    scenes: int = 6,
    frames_per_scene: int = 6,
    width: int = 160,
    height: int = 90,
    seed: int = 20261014,
    validation_fraction: float = 0.2,
    max_review_frames: int = 36,
    motion_scale_px: float = 64.0,
    scene_field: str = "scene_id",
    force: bool = False,
) -> dict[str, Any]:
    """Generate, scene-split, audit, and visually index a pilot dataset directory."""
    output = Path(output_dir).expanduser().resolve()
    if output == output.parent or output.name in ("", ".", ".."):
        raise ValueError("output_dir must name a dedicated output folder.")
    if not scene_field.strip():
        raise ValueError("scene_field must be a non-empty metadata field.")
    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    if not math.isfinite(motion_scale_px) or motion_scale_px <= 0:
        raise ValueError("motion_scale_px must be finite and greater than zero.")
    if not isinstance(max_review_frames, int) or isinstance(max_review_frames, bool) or max_review_frames < 1:
        raise ValueError("max_review_frames must be a positive integer.")
    if output.exists() and not force:
        raise FileExistsError(
            f"Output path already exists: {output}. Pass --force to replace it."
        )
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path must be a directory: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    transaction_root = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.prepare-", dir=output.parent
    ))
    staging = transaction_root / "payload"
    backup: Path | None = None
    moved_into_place = False

    try:
        generation = generate_scene_dataset(
            staging,
            scenes=scenes,
            frames_per_scene=frames_per_scene,
            width=width,
            height=height,
            seed=seed,
        )
        split = split_manifest(
            staging / "all-frames.jsonl",
            staging / "train.jsonl",
            staging / "validation.jsonl",
            validation_fraction=validation_fraction,
            seed=seed,
            scene_field=scene_field,
        )
        audit = audit_manifests(
            staging / "train.jsonl",
            staging / "validation.jsonl",
            staging / "dataset-audit.json",
            scene_field=scene_field,
            motion_scale_px=motion_scale_px,
            provenance_json=staging / "provenance.json",
        )
        gallery = build_review_gallery(
            staging / "all-frames.jsonl",
            staging / "review-gallery",
            max_frames=max_review_frames,
            seed=seed,
            scene_field=scene_field,
        )
        passed = bool(audit["passed"] and gallery["passed"])
        report: dict[str, Any] = {
            "format": "trss-pilot-pipeline-v1",
            "passed": passed,
            "source_type": "synthetic_controlled_3d_scene",
            "actual_game_capture": False,
            "note": (
                "Pipeline validation only. Passing structural checks does not prove "
                "visual alignment, photorealistic quality, game capture support, or "
                "Arc GPU performance. Inspect review-gallery/index.html manually."
            ),
            "generation": {
                "frame_count": generation["frame_count"],
                "scenes": generation["scenes"],
                "frames_per_scene": generation["frames_per_scene"],
                "input_size": generation["input_size"],
                "target_size": generation["target_size"],
                "seed": generation["seed"],
            },
            "split": {
                "training_frames": split["training_frames"],
                "validation_frames": split["validation_frames"],
                "training_scenes": split["training_scenes"],
                "validation_scenes": split["validation_scenes"],
                "scene_split_is_disjoint": split["scene_split_is_disjoint"],
            },
            "audit": {
                "passed": audit["passed"],
                "report": "dataset-audit.json",
                "errors": len(audit["errors"]),
                "warnings": len(audit["warnings"]),
                "cross_split_duplicate_pixel_groups": audit["cross_split_duplicate_pixel_groups"],
                "cross_split_near_duplicate_warning_counts": audit["cross_split_near_duplicate_warning_counts"],
            },
            "review": {
                "passed": gallery["passed"],
                "gallery": "review-gallery/index.html",
                "selected_frames": gallery["selected_count"],
                "preview_errors": gallery["preview_errors"],
            },
            "commands": {
                "train_manifest": "train.jsonl",
                "validation_manifest": "validation.jsonl",
                "audit_report": "dataset-audit.json",
                "review_gallery": "review-gallery/index.html",
            },
        }
        _write_json(staging / "pipeline-report.json", report)
        _rewrite_staging_paths(staging, output)

        if output.exists():
            backup_parent = Path(tempfile.mkdtemp(
                prefix=f".{output.name}.backup-", dir=output.parent
            ))
            backup_parent.rmdir()
            backup = backup_parent
            output.rename(backup)
        try:
            staging.rename(output)
            moved_into_place = True
        except Exception:
            if backup is not None and backup.exists() and not output.exists():
                backup.rename(output)
                backup = None
            raise

        if backup is not None and backup.exists():
            shutil.rmtree(backup)
            backup = None

        final_report = json.loads(
            (output / "pipeline-report.json").read_text(encoding="utf-8")
        )
        final_report["output_dir"] = str(output)
        return final_report
    finally:
        if transaction_root.exists():
            shutil.rmtree(transaction_root, ignore_errors=True)
        if backup is not None and backup.exists():
            if moved_into_place and output.exists():
                shutil.rmtree(backup, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-command controlled synthetic data pipeline: render, scene-split, "
            "audit, and create an offline gallery. Not game capture."
        )
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument("--frames-per-scene", type=int, default=6)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-review-frames", type=int, default=36)
    parser.add_argument("--motion-scale-px", type=float, default=64.0)
    parser.add_argument("--scene-field", default="scene_id")
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory.")
    args = parser.parse_args()

    try:
        report = prepare_pilot_dataset(
            args.output_dir,
            scenes=args.scenes,
            frames_per_scene=args.frames_per_scene,
            width=args.width,
            height=args.height,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            max_review_frames=args.max_review_frames,
            motion_scale_px=args.motion_scale_px,
            scene_field=args.scene_field,
            force=args.force,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
