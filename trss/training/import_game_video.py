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
    """Collect video files and complete TrSS desktop-capture folders."""
    candidates: list[Path] = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Gameplay recording does not exist: {path}")
        if path.is_dir():
            if _has_capture_session_artifacts(path):
                candidates.append(path)
                continue

            # A parent folder may contain multiple capture sessions and/or video clips.
            session_roots = {
                item.parent.resolve()
                for item in path.rglob("capture-manifest.jsonl")
                if (item.parent / "session.json").is_file()
                and (item.parent / ".trss-desktop-capture-session").is_file()
            }
            candidates.extend(session_roots)
            for item in path.rglob("*"):
                if not item.is_file() or item.suffix.lower() not in _VIDEO_EXTENSIONS:
                    continue
                resolved = item.resolve()
                if any(root == resolved or root in resolved.parents for root in session_roots):
                    continue
                candidates.append(resolved)
        elif path.is_file():
            if path.suffix.lower() not in _VIDEO_EXTENSIONS:
                raise ValueError(
                    f"Unsupported recording extension '{path.suffix}' for {path.name}. "
                    f"Supported videos: {', '.join(sorted(_VIDEO_EXTENSIONS))}; "
                    "TrSS desktop-capture session folders are also supported."
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
            "Provide at least two separate gameplay recordings or TrSS capture sessions. "
            "Complete recordings stay on one side of the train/validation split."
        )
    return paths

def _has_capture_session_artifacts(path: Path) -> bool:
    """Return whether a folder looks like a TrSS capture, including an incomplete one."""
    names = (".trss-desktop-capture-session", "session.json", "capture-manifest.jsonl")
    return any((path / name).exists() for name in names)


def _read_capture_session(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate a TrSS desktop capture and resolve each manifest frame safely."""
    root = source.resolve()
    required = (
        root / ".trss-desktop-capture-session",
        root / "session.json",
        root / "capture-manifest.jsonl",
    )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            f"'{root}' is not a complete TrSS capture session; missing {', '.join(missing)}."
        )

    metadata = json.loads(required[1].read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("format") != "trss-desktop-capture-session-v1":
        raise ValueError(f"'{root / 'session.json'}' has an unsupported capture format.")
    width, height = metadata.get("width"), metadata.get("height")
    if (
        not isinstance(width, int) or isinstance(width, bool) or width < 4
        or not isinstance(height, int) or isinstance(height, bool) or height < 4
    ):
        raise ValueError(f"'{root / 'session.json'}' has invalid frame dimensions.")

    rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    previous_timestamp = -1.0
    with required[2].open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of '{required[2]}'."
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest line {line_number} must contain a JSON object.")
            frame_id = row.get("frame_id")
            timestamp = row.get("timestamp")
            color = row.get("color")
            if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 0:
                raise ValueError(f"Manifest line {line_number} has an invalid frame_id.")
            if frame_id in seen_ids:
                raise ValueError(f"Manifest contains duplicate frame_id {frame_id}.")
            if (
                not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool)
                or not math.isfinite(float(timestamp)) or float(timestamp) < 0
                or float(timestamp) <= previous_timestamp
            ):
                raise ValueError(f"Manifest line {line_number} has a non-increasing timestamp.")
            if not isinstance(color, str) or not color.strip():
                raise ValueError(f"Manifest line {line_number} has no color path.")
            relative = Path(color)
            if relative.is_absolute():
                raise ValueError(f"Manifest line {line_number} must use a relative color path.")
            resolved = (root / relative).resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"Manifest line {line_number} points outside the capture folder."
                ) from exc
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"Capture frame referenced on line {line_number} is missing: {resolved}"
                )
            row["_resolved_color_path"] = resolved
            rows.append(row)
            seen_ids.add(frame_id)
            previous_timestamp = float(timestamp)

    if not rows:
        raise ValueError(f"Capture manifest contains no frames: {required[2]}")
    reported_frames = metadata.get("frames_saved")
    if reported_frames is not None and reported_frames != len(rows):
        raise ValueError(
            f"Capture manifest has {len(rows)} rows but session.json reports {reported_frames}."
        )
    return metadata, rows


def _source_sha256(source: Path) -> str:
    """Hash a video file or every byte of a captured desktop session."""
    if source.is_file():
        return _sha256(source)

    _, rows = _read_capture_session(source)
    digest = hashlib.sha256()
    for name in (
        ".trss-desktop-capture-session",
        "session.json",
        "capture-manifest.jsonl",
    ):
        digest.update(name.encode("utf-8") + b"\0")
        with (source / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    for row in rows:
        digest.update(str(row["color"]).encode("utf-8") + b"\0")
        with row["_resolved_color_path"].open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _save_rgb_pair(
    target: Image.Image, staging: Path, frame_root: Path, frame_index: int
) -> tuple[Path, Path, int, int]:
    """Save one captured RGB reference and its exactly half-size synthetic input."""
    target = target.convert("RGB")
    width, height = target.size
    width -= width % 2
    height -= height % 2
    if width < 4 or height < 4:
        raise ValueError("A captured frame is smaller than 4x4 after even-dimension cropping.")
    target = target.crop((0, 0, width, height))
    low = target.resize((width // 2, height // 2), resample=Image.Resampling.LANCZOS)
    filename = f"frame-{frame_index:07d}.png"
    target_path = frame_root / "target" / filename
    low_path = frame_root / "input" / filename
    target.save(target_path, format="PNG", optimize=False)
    low.save(low_path, format="PNG", optimize=False)
    return low_path, target_path, width, height



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
    """Create aligned RGB pairs from video clips or native TrSS capture folders.

    Video files are sampled by source-frame index. Desktop-capture sessions are
    already sampled by the native recorder, so their saved rows are consumed in
    order. Inputs are downsampled from their own targets; auxiliary renderer
    buffers are never invented.
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
    source_hashes = {path: _source_sha256(path) for path in sources}
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
        cv2 = _opencv_module() if any(source.is_file() for source in sources) else None

        # Split by complete recording/session, never by adjacent frames from one source.
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
            dimensions: set[tuple[int, int]] = set()
            rows_for_clip: list[dict[str, Any]] = []
            frame_root = staging / "frames" / capture_id
            (frame_root / "input").mkdir(parents=True)
            (frame_root / "target").mkdir(parents=True)

            if source.is_dir():
                # The native sampler already chooses when to save; do not apply the
                # video-only source-frame interval to those sampled rows.
                session, manifest_rows = _read_capture_session(source)
                source_name = source.name
                game_label = str(session.get("game_label", "unspecified")).strip() or "unspecified"
                selected_rows = manifest_rows[:max_frames_per_video]
                for frame_row in selected_rows:
                    frame_index = int(frame_row["frame_id"])
                    frame_path = frame_row["_resolved_color_path"]
                    with Image.open(frame_path) as decoded_image:
                        expected_size = (session["width"], session["height"])
                        if decoded_image.size != expected_size:
                            raise ValueError(
                                f"Frame {frame_index} in '{source.name}' is "
                                f"{decoded_image.width}x{decoded_image.height}; "
                                f"session.json says {expected_size[0]}x{expected_size[1]}."
                            )
                        target = decoded_image.convert("RGB")
                    low_path, target_path, width, height = _save_rgb_pair(
                        target, staging, frame_root, frame_index
                    )
                    rows_for_clip.append(
                        {
                            "low_res": low_path.relative_to(staging).as_posix(),
                            "target": target_path.relative_to(staging).as_posix(),
                            "scene_id": capture_id,
                            "capture_id": capture_id,
                            "source_name": source_name,
                            "source_type": "trss-desktop-capture-session",
                            "source_sha256": digest,
                            "game_label": game_label,
                            "frame_index": frame_index,
                            "timestamp_seconds": float(frame_row["timestamp"]),
                            "width": width,
                            "height": height,
                            "pairing": "same-captured-frame; input downsampled from target",
                            "auxiliary_buffers": "not captured; use spatial-only model",
                        }
                    )
                    dimensions.add((width, height))

                decoded = len(manifest_rows)
                sampled = len(rows_for_clip)
                sample_fps = session.get("measured_output_fps")
                if (
                    not isinstance(sample_fps, (int, float)) or isinstance(sample_fps, bool)
                    or not math.isfinite(float(sample_fps)) or float(sample_fps) <= 0
                ):
                    sample_fps = None
                if not rows_for_clip:
                    raise ValueError(f"No frames were sampled from capture session '{source.name}'.")
                (train_rows if split == "train" else validation_rows).extend(rows_for_clip)
                clip_reports.append(
                    {
                        "capture_id": capture_id,
                        "source_name": source_name,
                        "source_type": "trss-desktop-capture-session",
                        "game_label": game_label,
                        "source_sha256": digest,
                        "split": split,
                        "capture_sample_fps": sample_fps,
                        "duration_seconds": session.get("elapsed_seconds"),
                        "frames_decoded": decoded,
                        "frames_sampled": sampled,
                        "dimensions": [
                            {"width": width, "height": height}
                            for width, height in sorted(dimensions)
                        ],
                    }
                )
                continue

            assert cv2 is not None
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
                    target = Image.fromarray(rgb)
                    low_path, target_path, width, height = _save_rgb_pair(
                        target, staging, frame_root, frame_index
                    )
                    rows_for_clip.append(
                        {
                            "low_res": low_path.relative_to(staging).as_posix(),
                            "target": target_path.relative_to(staging).as_posix(),
                            "scene_id": capture_id,
                            "capture_id": capture_id,
                            "source_name": source.name,
                            "source_type": "gameplay-video",
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
                    "source_type": "gameplay-video",
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
            "dataset_type": "rgb-only-spatial-super-resolution-from-gameplay-video-or-desktop-capture",
            "completed": True,
            "output_dir": str(output),
            "source_count": len(sources),
            "training_captures": sorted({row["capture_id"] for row in train_rows}),
            "validation_captures": sorted({row["capture_id"] for row in validation_rows}),
            "training_frames": len(train_rows),
            "validation_frames": len(validation_rows),
            "sample_every_source_frames_for_video_only": sample_every,
            "max_frames_per_video": max_frames_per_video,
            "validation_fraction_requested": validation_fraction,
            "seed": seed,
            "clips": clip_reports,
            "limits": [
                "Reads local video files and first-party TrSS desktop-capture folders; no game-process access or injection is used.",
                "Each RGB target is a captured frame and its input is derived by downsampling that frame.",
                "Video and visible-desktop capture do not provide renderer depth, motion vectors, visibility masks, or reprojected history.",
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
        description="Build RGB-only 2x pairs from gameplay videos or TrSS desktop capture sessions."
    )
    parser.add_argument(
        "videos", nargs="+",
        help="At least two video recordings or TrSS desktop-capture session folders."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--sample-every", type=int, default=15,
        help="Sample interval in source-video frames; desktop-capture sessions are already sampled."
    )
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
