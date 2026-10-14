"""Create deterministic scene-separated train/validation manifests.

This utility uses only the standard library plus TrSS's path-only manifest
validator. It never creates frames or invents depth, motion, or history data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .manifest import load_manifest

_PATH_FIELDS = ("low_res", "target", "depth", "motion", "history", "history_valid")
_FRAME_ID_FIELDS = ("low_res", "target")


def _read_source_rows(
    source: Path, scene_field: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate every referenced file and preserve optional provenance fields."""
    resolved_rows = load_manifest(source)
    raw_rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            # load_manifest has already checked JSON syntax and row contracts.
            row = json.loads(line)
            scene = row.get(scene_field)
            if not isinstance(scene, str) or not scene.strip():
                raise ValueError(
                    f"{source}:{line_number}: each frame needs a non-empty "
                    f"'{scene_field}' string identifying its capture scene/sequence."
                )
            row[scene_field] = scene.strip()
            raw_rows.append(row)

    if len(raw_rows) != len(resolved_rows):
        raise RuntimeError("Manifest row count changed while it was being read.")
    return raw_rows, resolved_rows


def _check_scene_leakage(
    rows: list[dict[str, Any]], resolved_rows: list[dict[str, Any]], scene_field: str
) -> None:
    """Refuse exact color/target file reuse under different scene labels."""
    owners: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row, resolved in zip(rows, resolved_rows):
        scene = row[scene_field]
        for field in _FRAME_ID_FIELDS:
            owners[(field, str(resolved[field]))].add(scene)

    for (field, path), scenes in owners.items():
        if len(scenes) > 1:
            names = ", ".join(sorted(scenes))
            raise ValueError(
                f"Exact {field} frame '{path}' is assigned to multiple scene IDs "
                f"({names}); correct the scene labels before splitting."
            )


def _portable_path(path: Path, manifest_path: Path) -> str:
    """Store paths relative to each output manifest where the filesystem allows it."""
    resolved = path.resolve()
    try:
        relative = os.path.relpath(resolved, start=manifest_path.parent.resolve())
    except ValueError:
        # On Windows, inputs and output may be on different drive letters.
        return str(resolved)
    return Path(relative).as_posix()


def _render_manifest(
    rows: list[dict[str, Any]],
    resolved_rows: list[dict[str, Any]],
    output_path: Path,
) -> str:
    """Rebase file paths so the generated manifest works from its new directory."""
    lines: list[str] = []
    for row, resolved in zip(rows, resolved_rows):
        output_row = dict(row)
        for field in _PATH_FIELDS:
            value = resolved[field]
            output_row[field] = (
                None if value is None else _portable_path(Path(value), output_path)
            )
        lines.append(json.dumps(output_row, ensure_ascii=False, allow_nan=False))
    return "\n".join(lines) + "\n"


def _write_pair(
    train_path: Path,
    train_text: str,
    validation_path: Path,
    validation_text: str,
    *,
    force: bool,
) -> None:
    """Prepare both manifests completely before replacing output files."""
    temporary_paths: list[Path] = []
    destinations = (train_path, validation_path)
    payloads = (train_text, validation_text)

    if train_path == validation_path:
        raise ValueError("Training and validation output paths must be different.")
    for destination in destinations:
        if destination.exists() and not force:
            raise FileExistsError(
                f"Output already exists: {destination}. Pass --force to replace it."
            )

    try:
        for destination, payload in zip(destinations, payloads):
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as stream:
                stream.write(payload)
                temporary_paths.append(Path(stream.name))

        # Both complete temporary files exist before either destination is replaced.
        for temporary, destination in zip(temporary_paths, destinations):
            os.replace(temporary, destination)
    finally:
        for temporary in temporary_paths:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                # Preserve the original write/replace error if cleanup also fails.
                pass


def split_manifest(
    input_manifest: str | Path,
    train_output: str | Path,
    validation_output: str | Path,
    *,
    validation_fraction: float = 0.2,
    seed: int = 20261014,
    scene_field: str = "scene_id",
    force: bool = False,
) -> dict[str, Any]:
    """Split complete scenes into two manifests and summarize the resulting split."""
    source = Path(input_manifest).expanduser().resolve()
    train_path = Path(train_output).expanduser().resolve()
    validation_path = Path(validation_output).expanduser().resolve()

    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be finite and strictly between 0 and 1.")
    if not scene_field.strip() or scene_field in _PATH_FIELDS:
        raise ValueError("scene_field must name a separate metadata field, such as scene_id.")
    if train_path == validation_path:
        raise ValueError("Training and validation output paths must be different.")
    if source in (train_path, validation_path):
        raise ValueError("Output manifests must not overwrite the input manifest.")
    if not force:
        for output in (train_path, validation_path):
            if output.exists():
                raise FileExistsError(
                    f"Output already exists: {output}. Pass --force to replace it."
                )

    rows, resolved_rows = _read_source_rows(source, scene_field)
    _check_scene_leakage(rows, resolved_rows, scene_field)

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row[scene_field]].append(index)
    if len(groups) < 2:
        raise ValueError(
            "At least two distinct scene IDs are required; frames from one scene "
            "must not be randomly split between training and validation."
        )

    # Whole scenes are indivisible, so the final fraction may differ from the target.
    target_validation_frames = max(1, math.ceil(len(rows) * validation_fraction))
    candidates = sorted(groups)
    random.Random(seed).shuffle(candidates)
    validation_scenes: set[str] = set()
    validation_frame_count = 0
    while candidates and len(candidates) > 1 and validation_frame_count < target_validation_frames:
        chosen = min(
            candidates,
            key=lambda scene: abs(
                validation_frame_count + len(groups[scene]) - target_validation_frames
            ),
        )
        candidates.remove(chosen)
        validation_scenes.add(chosen)
        validation_frame_count += len(groups[chosen])

    if not validation_scenes:
        raise ValueError("Could not allocate a validation scene while leaving training data.")

    train_indexes = [i for i, row in enumerate(rows) if row[scene_field] not in validation_scenes]
    validation_indexes = [i for i, row in enumerate(rows) if row[scene_field] in validation_scenes]
    train_rows = [rows[i] for i in train_indexes]
    train_resolved = [resolved_rows[i] for i in train_indexes]
    validation_rows = [rows[i] for i in validation_indexes]
    validation_resolved = [resolved_rows[i] for i in validation_indexes]

    train_text = _render_manifest(train_rows, train_resolved, train_path)
    validation_text = _render_manifest(
        validation_rows, validation_resolved, validation_path
    )
    _write_pair(
        train_path,
        train_text,
        validation_path,
        validation_text,
        force=force,
    )

    training_scenes = sorted(set(groups) - validation_scenes)
    return {
        "format": "trss-scene-split-v1",
        "input_manifest": str(source),
        "training_manifest": str(train_path),
        "validation_manifest": str(validation_path),
        "scene_field": scene_field,
        "seed": seed,
        "requested_validation_fraction": validation_fraction,
        "training_frames": len(train_rows),
        "validation_frames": len(validation_rows),
        "total_frames": len(rows),
        "actual_validation_fraction": len(validation_rows) / len(rows),
        "training_scenes": training_scenes,
        "validation_scenes": sorted(validation_scenes),
        "scene_split_is_disjoint": set(training_scenes).isdisjoint(validation_scenes),
        "notice": (
            "This only organizes existing files. It does not generate renders, "
            "reproject history, verify visual pairing, or train model weights."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split a TrSS JSONL frame manifest by whole scene/capture sequence."
    )
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--train-output", required=True, type=Path)
    parser.add_argument("--validation-output", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument(
        "--scene-field",
        default="scene_id",
        help="JSON field naming a complete capture scene or sequence (default: scene_id)",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace output manifests if they already exist"
    )
    args = parser.parse_args()

    try:
        report = split_manifest(
            args.input_manifest,
            args.train_output,
            args.validation_output,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            scene_field=args.scene_field,
            force=args.force,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
