"""Audit TrSS paired-render manifests before training.

Checks file decoding, 2x target dimensions, auxiliary tensor contracts, scene
separation, and exact pixel duplicates across training/validation manifests.
It does not judge whether two images are semantically aligned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .manifest import load_manifest

_FRAME_FIELDS = ("low_res", "target")
_PROVENANCE_FIELDS = ("dataset_name", "source", "license")


def _read_raw_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate manifest paths once, then preserve any extra per-frame metadata."""
    resolved_rows = load_manifest(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            rows.append(json.loads(line))
    if len(rows) != len(resolved_rows):
        raise RuntimeError("Manifest row count changed while it was being read.")
    return rows, resolved_rows


def _read_rgb(path: Path) -> np.ndarray:
    """Read any supported image as explicit 8-bit RGB pixels."""
    try:
        with Image.open(path) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(f"cannot decode RGB image '{path}': {exc}") from exc


def _pixel_hash(pixels: np.ndarray) -> str:
    """Hash decoded pixels, so re-encoded copies are still detected."""
    digest = hashlib.sha256()
    digest.update(str(pixels.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(pixels.tobytes(order="C"))
    return digest.hexdigest()


def _read_npy(path: Path, field: str) -> np.ndarray:
    """Load a numeric array without allowing embedded Python objects."""
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load {field} array '{path}': {exc}") from exc
    if array.dtype.kind not in "uif":
        raise ValueError(f"{field} must be a real numeric NumPy array.")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{field} contains NaN or infinite values.")
    return array


def _read_depth(path: Path, height: int, width: int) -> dict[str, float]:
    depth = _read_npy(path, "depth")
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.shape != (height, width):
        raise ValueError(
            f"depth shape {depth.shape} must match input image {(height, width)}."
        )
    minimum, maximum = float(depth.min()), float(depth.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError("depth values must be normalized to [0, 1].")
    return {"min": minimum, "max": maximum, "mean": float(depth.mean())}


def _read_motion(
    path: Path, height: int, width: int, motion_scale_px: float
) -> dict[str, float]:
    motion = _read_npy(path, "motion")
    if motion.shape != (height, width, 2):
        raise ValueError(
            f"motion shape {motion.shape} must be {(height, width, 2)} in HWC order."
        )
    absolute = np.abs(motion)
    return {
        "max_abs_pixels_per_frame": float(absolute.max()),
        "p99_abs_pixels_per_frame": float(np.percentile(absolute, 99)),
        "component_fraction_clipped_by_training_scale": float(
            np.mean(absolute > motion_scale_px)
        ),
    }


def _read_history_mask(path: Path, height: int, width: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        mask = _read_npy(path, "history-validity mask")
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
    else:
        try:
            with Image.open(path) as image:
                mask = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise ValueError(f"cannot decode history-validity mask '{path}': {exc}") from exc

    if mask.shape != (height, width):
        raise ValueError(
            f"history-validity mask shape {mask.shape} must match output {(height, width)}."
        )
    if not np.isfinite(mask).all() or mask.min() < 0.0 or mask.max() > 1.0:
        raise ValueError("history-validity mask values must be finite and in [0, 1].")
    return mask


def _audit_row(
    raw: dict[str, Any],
    resolved: dict[str, Any],
    *,
    split_name: str,
    scene_field: str,
    motion_scale_px: float,
    hashes: dict[str, dict[str, list[dict[str, Any]]]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate one paired frame and collect hashes for exact-leak detection."""
    line = int(resolved["line"])
    scene = raw.get(scene_field)
    if not isinstance(scene, str) or not scene.strip():
        errors.append({
            "split": split_name, "line": line,
            "error": f"missing non-empty scene identifier field '{scene_field}'",
        })
        scene = None
    elif isinstance(scene, str):
        scene = scene.strip()

    detail: dict[str, Any] = {
        "line": line,
        "scene_id": scene,
        "input_size": None,
        "target_size": None,
        "depth": None,
        "motion": None,
        "history_valid_fraction": None,
    }

    try:
        low_path = Path(resolved["low_res"])
        target_path = Path(resolved["target"])
        low_rgb = _read_rgb(low_path)
        target_rgb = _read_rgb(target_path)
        low_height, low_width = low_rgb.shape[:2]
        target_height, target_width = target_rgb.shape[:2]
        detail["input_size"] = [low_width, low_height]
        detail["target_size"] = [target_width, target_height]

        for field, pixels, path in (
            ("low_res", low_rgb, low_path),
            ("target", target_rgb, target_path),
        ):
            digest = _pixel_hash(pixels)
            hashes[field].setdefault(digest, []).append({
                "scene_id": scene, "line": line, "path": str(path),
            })

        if (target_height, target_width) != (low_height * 2, low_width * 2):
            raise ValueError(
                f"target size {(target_width, target_height)} must be exactly 2x "
                f"the input size {(low_width, low_height)}."
            )

        detail["depth"] = _read_depth(Path(resolved["depth"]), low_height, low_width)
        detail["motion"] = _read_motion(
            Path(resolved["motion"]), low_height, low_width, motion_scale_px
        )

        history_path = resolved["history"]
        validity_path = resolved["history_valid"]
        if history_path is None:
            detail["history_state"] = "reset_or_first_frame"
            detail["history_valid_fraction"] = 0.0
        else:
            history_rgb = _read_rgb(Path(history_path))
            if history_rgb.shape[:2] != (target_height, target_width):
                raise ValueError(
                    "warped history image must match the high-resolution target dimensions."
                )
            mask = _read_history_mask(
                Path(validity_path), target_height, target_width
            )
            detail["history_state"] = "provided"
            detail["history_valid_fraction"] = float(np.mean(mask))

    except (OSError, ValueError) as exc:
        errors.append({
            "split": split_name, "line": line, "scene_id": scene, "error": str(exc)
        })

    return detail


def _index_duplicates(
    hashes: dict[str, dict[str, list[dict[str, Any]]]],
    split_name: str,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> dict[str, int]:
    """Flag identical decoded pixels that were assigned to multiple scene IDs."""
    counts: dict[str, int] = {}
    for field, entries in hashes.items():
        duplicate_groups = 0
        for digest, records in entries.items():
            if len(records) < 2:
                continue
            duplicate_groups += 1
            scenes = {record["scene_id"] for record in records}
            if len(scenes) > 1:
                errors.append({
                    "split": split_name,
                    "field": field,
                    "pixel_sha256": digest,
                    "error": "identical decoded frame pixels are assigned to multiple scene IDs",
                    "records": records,
                })
            else:
                warnings.append({
                    "split": split_name,
                    "field": field,
                    "pixel_sha256": digest,
                    "warning": "identical decoded frame pixels occur multiple times within one scene",
                    "occurrences": len(records),
                })
        counts[field] = duplicate_groups
    return counts


def _read_provenance(path: Path) -> dict[str, Any]:
    """Require dataset identity, origin, and licence to make the audit traceable."""
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("provenance JSON must contain an object.")
    missing = [
        name for name in _PROVENANCE_FIELDS
        if not isinstance(value.get(name), str) or not value[name].strip()
    ]
    if missing:
        raise ValueError("provenance JSON needs non-empty fields: " + ", ".join(missing))
    return {name: value[name].strip() for name in _PROVENANCE_FIELDS}


def _write_report(path: Path, report: dict[str, Any]) -> None:
    """Write a complete report to a temporary file, then atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def audit_manifests(
    train_manifest: str | Path,
    validation_manifest: str | Path,
    report_path: str | Path,
    *,
    scene_field: str = "scene_id",
    motion_scale_px: float = 64.0,
    provenance_json: str | Path | None = None,
) -> dict[str, Any]:
    """Audit frame contracts and look for exact decoded-pixel leakage."""
    train_path = Path(train_manifest).expanduser().resolve()
    validation_path = Path(validation_manifest).expanduser().resolve()
    report_file = Path(report_path).expanduser().resolve()
    provenance_path = (
        Path(provenance_json).expanduser().resolve()
        if provenance_json is not None else None
    )

    if not math.isfinite(motion_scale_px) or motion_scale_px <= 0:
        raise ValueError("motion_scale_px must be finite and greater than zero.")
    if not scene_field.strip():
        raise ValueError("scene_field must be a non-empty metadata field name.")
    protected_paths = {train_path, validation_path}
    if provenance_path is not None:
        protected_paths.add(provenance_path)
    if report_file in protected_paths:
        raise ValueError("report output must not overwrite a manifest or provenance JSON.")

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    split_details: dict[str, Any] = {}
    hashes: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    scene_sets: dict[str, set[str]] = {}

    for split_name, path in (("training", train_path), ("validation", validation_path)):
        rows, resolved_rows = _read_raw_rows(path)
        split_hashes = {
            "low_res": defaultdict(list),
            "target": defaultdict(list),
        }
        row_details = [
            _audit_row(
                raw, resolved,
                split_name=split_name,
                scene_field=scene_field,
                motion_scale_px=motion_scale_px,
                hashes=split_hashes,
                errors=errors,
            )
            for raw, resolved in zip(rows, resolved_rows)
        ]
        scene_ids = {
            row["scene_id"] for row in row_details
            if isinstance(row["scene_id"], str) and row["scene_id"]
        }
        scene_sets[split_name] = scene_ids
        valid_input_sizes = Counter(
            tuple(row["input_size"]) for row in row_details
            if row["input_size"] is not None
        )
        reset_frames = sum(
            row.get("history_state") == "reset_or_first_frame" for row in row_details
        )
        clipping_frames = sum(
            isinstance(row.get("motion"), dict)
            and row["motion"]["component_fraction_clipped_by_training_scale"] > 0
            for row in row_details
        )
        hashes[split_name] = split_hashes
        duplicates = _index_duplicates(split_hashes, split_name, errors, warnings)
        split_details[split_name] = {
            "manifest": str(path),
            "frames": len(rows),
            "scenes": sorted(scene_ids),
            "scene_count": len(scene_ids),
            "input_dimensions_and_counts": {
                f"{size[0]}x{size[1]}": count
                for size, count in sorted(valid_input_sizes.items())
            },
            "history_reset_frames": reset_frames,
            "frames_with_motion_components_clipped_by_training_scale": clipping_frames,
            "duplicate_pixel_groups_by_field": duplicates,
            "frame_details": row_details,
        }

    overlap = sorted(scene_sets["training"].intersection(scene_sets["validation"]))
    if overlap:
        errors.append({
            "error": "scene IDs appear in both training and validation manifests",
            "overlapping_scenes": overlap,
        })

    cross_split_duplicates: dict[str, int] = {}
    for field in _FRAME_FIELDS:
        shared = sorted(
            set(hashes["training"][field]).intersection(hashes["validation"][field])
        )
        cross_split_duplicates[field] = len(shared)
        for digest in shared:
            errors.append({
                "field": field,
                "pixel_sha256": digest,
                "error": "identical decoded frame pixels occur in both training and validation",
                "training_records": hashes["training"][field][digest],
                "validation_records": hashes["validation"][field][digest],
            })

    provenance = None
    if provenance_path is None:
        warnings.append({
            "warning": "No provenance JSON supplied; source and licence traceability remain unverified."
        })
    else:
        try:
            provenance = _read_provenance(provenance_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"error": f"invalid provenance JSON: {exc}"})

    report: dict[str, Any] = {
        "format": "trss-dataset-audit-v1",
        "passed": not errors,
        "motion_scale_px": motion_scale_px,
        "scene_field": scene_field,
        "provenance": provenance,
        "cross_split_duplicate_pixel_groups": cross_split_duplicates,
        "splits": split_details,
        "errors": errors,
        "warnings": warnings,
        "notice": (
            "A structural audit cannot prove that low-resolution inputs and reference "
            "renders are visually aligned, legally usable, or representative of a game. "
            "Review sample images and capture/asset licences before training."
        ),
    }
    report["report_path"] = str(report_file)
    _write_report(report_file, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit paired TrSS frame manifests before starting model training."
    )
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--scene-field", default="scene_id")
    parser.add_argument("--motion-scale-px", type=float, default=64.0)
    parser.add_argument(
        "--provenance-json",
        type=Path,
        help="JSON object with non-empty dataset_name, source, and license fields",
    )
    args = parser.parse_args()

    try:
        report = audit_manifests(
            args.train_manifest,
            args.validation_manifest,
            args.report,
            scene_field=args.scene_field,
            motion_scale_px=args.motion_scale_px,
            provenance_json=args.provenance_json,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
