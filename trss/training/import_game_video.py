"""Import user-recorded gameplay clips into a local RGB-only 2x dataset.

This tool decodes ordinary files; it does not attach to, inject into, or modify a game.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

_VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _opencv_module():
    """Load the optional decoder only when the user runs this command."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Video import needs OpenCV. Install it with: python -m pip install -e '.[video]'"
        ) from exc
    return cv2


def _expand_video_inputs(values: Iterable[str | Path]) -> list[Path]:
    candidates: list[Path] = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Gameplay recording does not exist: {path}")
        if path.is_dir():
            candidates.extend(
                item.resolve()
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in _VIDEO_EXTENSIONS
            )
        elif path.is_file():
            if path.suffix.lower() not in _VIDEO_EXTENSIONS:
                raise ValueError(
                    f"Unsupported video extension '{path.suffix}' for {path.name}. "
                    f"Supported: {', '.join(sorted(_VIDEO_EXTENSIONS))}."
                )
            candidates.append(path)
        else:
            raise ValueError(f"Input is not a regular file or directory: {path}")

    unique: dict[str, Path] = {}
    for item in sorted(candidates, key=lambda part: (part.name.casefold(), str(part))):
        unique[str(item).casefold()] = item
    paths = list(unique.values())
    if len(paths) < 2:
        raise ValueError(
            "Provide at least two separate gameplay recordings. Whole clips stay on "
            "one side of the train/validation split to reduce frame leakage."
        )
    return paths


def import_game_videos(
    videos: Iterable[str | Path],
    output_dir: str | Path,
    *,
    sample_every: int = 15,
    max_frames_per_video: int = 240,
    validation_fraction: float = 0.2,
    seed: int = 20261014,
    force: bool = False,
) -> dict[str, Any]:
    """Create matched half-resolution inputs and captured-resolution RGB targets.

    Each target is one decoded video frame. The low-resolution input is made by
    downsampling the same frame, so the pair is spatially aligned. No depth,
    motion vectors, or reprojected temporal history are invented.
    """
    if not isinstance(sample_every, int) or isinstance(sample_every, bool) or sample_every < 1:
        raise ValueError("sample_every must be a positive integer.")
    if (
        not isinstance(max_frames_per_video, int)
        or isinstance(max_frames_per_video, bool)
        or max_frames_per_video < 1
    ):
        raise ValueError("max_frames_per_video must be a positive integer.")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not math.isfinite(validation_fraction) or not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")

    sources = _expand_video_inputs(videos)
    source_hashes = {path: _sha256(path) for path in sources}
    if len(set(source_hashes.values())) != len(source_hashes):
        raise ValueError("Two or more input recordings have identical file content.")

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
    transaction_root = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.game-import-", dir=output.parent)
    )
    staging = transaction_root / "payload"
    backup: Path | None = None
    try:
        staging.mkdir()
        cv2 = _opencv_module()

        # Split by complete video file, never by adjacent frames from one recording.
        shuffled = list(sources)
        random.Random(seed).shuffle(shuffled)
        validation_count = max(
            1, min(len(shuffled) - 1, round(len(shuffled) * validation_fraction))
        )
        validation_sources = set(shuffled[:validation_count])

        train_rows: list[dict[str, Any]] = []
        validation_rows: list[dict[str, Any]] = []
        clip_reports: list[dict[str, Any]] = []

        for source in sources:
            digest = source_hashes[source]
            capture_id = f"capture-{digest[:12]}"
            split = "validation" if source in validation_sources else "train"
            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                capture.release()
                raise ValueError(
                    f"Could not open '{source.name}'. Check the file and OpenCV decoder support."
                )
            reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
            fps = reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else None
            decoded = 0
            sampled = 0
            dimensions: set[tuple[int, int]] = set()
            rows_for_clip: list[dict[str, Any]] = []
            frame_root = staging / "frames" / capture_id
            (frame_root / "input").mkdir(parents=True)
            (frame_root / "target").mkdir(parents=True)

            try:
                while sampled < max_frames_per_video:
                    ok, bgr = capture.read()
                    if not ok:
                        break
                    frame_index = decoded
                    decoded += 1
                    if frame_index % sample_every != 0:
                        continue

                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    height, width = rgb.shape[:2]
                    # Cropping an odd final row or column keeps every pair exactly 2x.
                    width -= width % 2
                    height -= height % 2
                    if width < 4 or height < 4:
                        raise ValueError(f"'{source.name}' contains a frame smaller than 4x4.")
                    target = Image.fromarray(rgb[:height, :width, :], mode="RGB")
                    low = target.resize(
                        (width // 2, height // 2), resample=Image.Resampling.LANCZOS
                    )
                    filename = f"frame-{frame_index:07d}.png"
                    target_path = frame_root / "target" / filename
                    low_path = frame_root / "input" / filename
                    target.save(target_path, format="PNG", optimize=False)
                    low.save(low_path, format="PNG", optimize=False)

                    rows_for_clip.append(
                        {
                            "low_res": low_path.relative_to(staging).as_posix(),
                            "target": target_path.relative_to(staging).as_posix(),
                            "scene_id": capture_id,
                            "capture_id": capture_id,
                            "source_name": source.name,
                            "source_sha256": digest,
                            "frame_index": frame_index,
                            "timestamp_seconds": frame_index / fps if fps is not None else None,
                            "width": width,
                            "height": height,
                            "pairing": "same-decoded-frame; input downsampled from target",
                            "auxiliary_buffers": "not captured; use spatial-only model",
                        }
                    )
                    dimensions.add((width, height))
                    sampled += 1
            finally:
                capture.release()

            if not rows_for_clip:
                raise ValueError(f"No frames were sampled from '{source.name}'.")
            (train_rows if split == "train" else validation_rows).extend(rows_for_clip)
            clip_reports.append(
                {
                    "capture_id": capture_id,
                    "source_name": source.name,
                    "source_sha256": digest,
                    "split": split,
                    "fps": fps,
                    "frames_decoded": decoded,
                    "frames_sampled": sampled,
                    "dimensions": [
                        {"width": width, "height": height}
                        for width, height in sorted(dimensions)
                    ],
                }
            )

        for filename, rows in (
            ("train.jsonl", train_rows),
            ("validation.jsonl", validation_rows),
        ):
            with (staging / filename).open("w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(
                        json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
                    )

        report = {
            "schema_version": 1,
            "dataset_type": "rgb-only-spatial-super-resolution-from-gameplay-video",
            "completed": True,
            "output_dir": str(output),
            "source_count": len(sources),
            "training_captures": sorted({row["capture_id"] for row in train_rows}),
            "validation_captures": sorted({row["capture_id"] for row in validation_rows}),
            "training_frames": len(train_rows),
            "validation_frames": len(validation_rows),
            "sample_every_source_frames": sample_every,
            "max_frames_per_video": max_frames_per_video,
            "validation_fraction_requested": validation_fraction,
            "seed": seed,
            "clips": clip_reports,
            "limits": [
                "Decodes local video files only; no game-process access or injection is used.",
                "Each RGB target is a captured frame and its input is derived by downsampling that frame.",
                "Gameplay video does not provide renderer depth, motion vectors, visibility masks, or reprojected history.",
                "This dataset is for the RGB-only spatial model, not the ten-channel temporal model.",
                "Keep footage and derived frames local unless the content licence allows redistribution.",
            ],
        }
        _write_json(staging / "capture-report.json", report)
        _write_json(staging / "provenance.json", {
            "schema_version": 1,
            "dataset_type": report["dataset_type"],
            "seed": seed,
            "clips": clip_reports,
            "note": "Source names and hashes are recorded; source videos are not copied.",
        })

        if output.exists():
            backup = transaction_root / "previous-output"
            output.rename(backup)
        try:
            staging.rename(output)
        except Exception:
            if backup is not None and backup.exists() and not output.exists():
                backup.rename(output)
                backup = None
            raise
        return report
    finally:
        shutil.rmtree(transaction_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build RGB-only 2x pairs from ordinary game recordings, without game hooks."
    )
    parser.add_argument("videos", nargs="+", help="At least two video files or folders.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-every", type=int, default=15)
    parser.add_argument("--max-frames-per-video", type=int, default=240)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        report = import_game_videos(
            args.videos, args.output_dir,
            sample_every=args.sample_every,
            max_frames_per_video=args.max_frames_per_video,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            force=args.force,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    root = args.output_dir.expanduser().resolve()
    print(json.dumps({
        "capture_report": str(root / "capture-report.json"),
        "training_manifest": str(root / "train.jsonl"),
        "validation_manifest": str(root / "validation.jsonl"),
        "training_frames": report["training_frames"],
        "validation_frames": report["validation_frames"],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
