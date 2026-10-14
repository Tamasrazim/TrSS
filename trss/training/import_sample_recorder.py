"""Decode TrSS Sample Recorder JSON chunks into sessions used by the RGB-only trainer.

The recorder embeds JPEGs in small, size-capped JSON files. This importer validates
those files and turns them into ordinary TrSS capture-session folders; it never
contacts a game or uploads captured content.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

_SCHEMA = "trss.sample-recorder.chunk.v1"
_CAPTURE_SCHEMA = "trss-desktop-capture-session-v1"
_MAX_CHUNK_BYTES = 23 * 1024 * 1024
_MAX_FRAMES_PER_SESSION = 50_000


@dataclass(frozen=True)
class ChunkInfo:
    """The small amount of data needed to group a chunk without retaining its images."""

    path: Path
    session_id: str
    chunk_index: int
    game_name: str
    game_source: str
    metadata: dict[str, Any]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _read_chunk_header(path: Path) -> ChunkInfo:
    size = path.stat().st_size
    if size > _MAX_CHUNK_BYTES:
        raise ValueError(
            f"'{path}' is {size:,} bytes. A recorder JSON chunk must not exceed "
            f"{_MAX_CHUNK_BYTES:,} bytes (23 MiB)."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid UTF-8 JSON from '{path}': {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != _SCHEMA:
        raise ValueError(f"'{path}' is not a supported TrSS Sample Recorder JSON chunk.")
    session_id = data.get("sessionId")
    chunk_index = data.get("chunkIndex")
    game = data.get("game")
    frames = data.get("frames")
    if not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 200:
        raise ValueError(f"'{path}' has an invalid sessionId.")
    if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 1:
        raise ValueError(f"'{path}' has an invalid chunkIndex.")
    if not isinstance(game, dict):
        raise ValueError(f"'{path}' is missing game metadata.")
    game_name = game.get("name")
    game_source = game.get("source", "unknown")
    if not isinstance(game_name, str) or not game_name.strip() or len(game_name) > 200:
        raise ValueError(f"'{path}' has an invalid game name.")
    if not isinstance(game_source, str) or len(game_source) > 100:
        raise ValueError(f"'{path}' has an invalid game source.")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"'{path}' contains no captured frames.")
    return ChunkInfo(
        path=path.resolve(),
        session_id=session_id,
        chunk_index=chunk_index,
        game_name=game_name.strip(),
        game_source=game_source,
        metadata={},
    )


def _collect_chunks(values: Iterable[str | Path]) -> list[ChunkInfo]:
    paths: list[Path] = []
    for raw in values:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Sample Recorder input does not exist: {path}")
        if path.is_dir():
            paths.extend(
                item for item in path.rglob("chunk-*.json")
                if item.is_file() and item.suffix.lower() == ".json"
            )
        elif path.is_file():
            if path.suffix.lower() != ".json":
                raise ValueError(f"Expected a completed .json chunk, got '{path.name}'.")
            paths.append(path)
        else:
            raise ValueError(f"Input is not a regular file or directory: {path}")

    unique = {str(path).casefold(): path for path in paths}
    if not unique:
        raise ValueError(
            "No completed chunk-*.json files were found. Finish a recording or switch "
            "away from the game to finalize its current chunk before importing."
        )
    chunks = [_read_chunk_header(path) for path in sorted(unique.values())]
    seen: set[tuple[str, int]] = set()
    for chunk in chunks:
        key = (chunk.session_id, chunk.chunk_index)
        if key in seen:
            raise ValueError(
                f"Session '{chunk.session_id}' contains duplicate chunk index {chunk.chunk_index}."
            )
        seen.add(key)
    return chunks


def _safe_slug(value: str) -> str:
    output = "".join(char.lower() if char.isalnum() else "-" for char in value)
    output = "-".join(part for part in output.split("-") if part)
    return (output or "game")[:60]


def _session_folder_name(game: str, session_id: str) -> str:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
    return f"{_safe_slug(game)}-{digest}"


def _decode_frame(frame: Any, *, path: Path, index: int) -> tuple[bytes, int, int, int, float]:
    if not isinstance(frame, dict):
        raise ValueError(f"Frame {index} in '{path.name}' must be a JSON object.")
    frame_index = frame.get("frameIndex")
    elapsed_ms = frame.get("elapsedMs")
    width = frame.get("width")
    height = frame.get("height")
    image_text = frame.get("imageBase64")
    image_format = frame.get("format")
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0:
        raise ValueError(f"Frame {index} in '{path.name}' has an invalid frameIndex.")
    if (
        not isinstance(elapsed_ms, (int, float))
        or isinstance(elapsed_ms, bool)
        or not math.isfinite(float(elapsed_ms))
        or elapsed_ms < 0
    ):
        raise ValueError(f"Frame {index} in '{path.name}' has an invalid elapsedMs.")
    for label, value in (("width", width), ("height", height)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 4 or value > 1920:
            raise ValueError(f"Frame {index} in '{path.name}' has an invalid {label}; captured pixels must be at most 1920 per dimension.")
    if image_format != "image/jpeg":
        raise ValueError(f"Frame {index} in '{path.name}' is not a JPEG image.")
    if not isinstance(image_text, str) or not image_text:
        raise ValueError(f"Frame {index} in '{path.name}' has no imageBase64 data.")
    try:
        image_bytes = base64.b64decode(image_text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Frame {index} in '{path.name}' has invalid base64 data.") from exc
    if not image_bytes:
        raise ValueError(f"Frame {index} in '{path.name}' contains an empty image.")
    return image_bytes, width, height, frame_index, float(elapsed_ms) / 1000.0


def _write_capture_session(
    chunks: list[ChunkInfo],
    destination: Path,
) -> dict[str, Any]:
    ordered_chunks = sorted(chunks, key=lambda item: item.chunk_index)
    first = ordered_chunks[0]
    for chunk in ordered_chunks:
        if chunk.game_name != first.game_name or chunk.session_id != first.session_id:
            raise ValueError("Recorder chunks from different sessions were grouped together.")
        if chunk.game_source != first.game_source:
            raise ValueError(f"Game source changes between chunks of session '{first.session_id}'.")

    destination.mkdir(parents=True)
    frame_root = destination / "frames"
    frame_root.mkdir()
    manifest_path = destination / "capture-manifest.jsonl"
    expected_size: tuple[int, int] | None = None
    first_elapsed: float | None = None
    previous_elapsed: float | None = None
    previous_frame_index = -1
    saved = 0
    chunk_reports: list[dict[str, Any]] = []

    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
        for chunk in ordered_chunks:
            try:
                data = json.loads(chunk.path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not reread chunk '{chunk.path}': {exc}") from exc
            frames = data.get("frames")
            if not isinstance(frames, list) or not frames:
                raise ValueError(f"Chunk '{chunk.path.name}' contains no frames.")
            if saved + len(frames) > _MAX_FRAMES_PER_SESSION:
                raise ValueError(
                    f"Session '{first.session_id}' exceeds the {_MAX_FRAMES_PER_SESSION:,} "
                    "frame safety limit. Import smaller recording sessions separately."
                )
            chunk_first_id = None
            chunk_last_id = None
            for row_index, row in enumerate(frames):
                image_bytes, declared_width, declared_height, frame_index, elapsed = _decode_frame(
                    row, path=chunk.path, index=row_index
                )
                if frame_index <= previous_frame_index:
                    raise ValueError(
                        f"Frame indexes in session '{first.session_id}' must increase "
                        "strictly across all chunks."
                    )
                if previous_elapsed is not None and elapsed <= previous_elapsed:
                    raise ValueError(
                        f"Elapsed times in session '{first.session_id}' must increase strictly."
                    )
                previous_frame_index = frame_index
                previous_elapsed = elapsed
                if first_elapsed is None:
                    first_elapsed = elapsed

                try:
                    with Image.open(BytesIO(image_bytes)) as decoded:
                        if decoded.format != "JPEG" or decoded.size != (declared_width, declared_height):
                            raise ValueError(
                                f"Frame {frame_index} in '{chunk.path.name}' has image dimensions "
                                "that do not match its JSON metadata."
                            )
                        decoded.load()
                        rgb = decoded.convert("RGB")
                        if expected_size is None:
                            expected_size = rgb.size
                        if rgb.size != expected_size:
                            # One session must have a stable tensor shape. The recorder usually
                            # captures a fixed game resolution; resize only after a mode change.
                            rgb = rgb.resize(expected_size, resample=Image.Resampling.LANCZOS)
                            output_buffer = BytesIO()
                            rgb.save(output_buffer, format="JPEG", quality=95, optimize=True)
                            image_bytes = output_buffer.getvalue()
                except (OSError, ValueError) as exc:
                    raise ValueError(
                        f"Could not decode frame {frame_index} in '{chunk.path.name}': {exc}"
                    ) from exc

                filename = f"frame_{saved:07d}.jpg"
                image_path = frame_root / filename
                image_path.write_bytes(image_bytes)
                relative = f"frames/{filename}"
                timestamp = elapsed - (first_elapsed or 0.0)
                if timestamp < 0:
                    raise ValueError("A frame timestamp precedes the first recorded frame.")
                manifest_row = {
                    "frame_id": saved,
                    "timestamp": timestamp,
                    "color": relative,
                    "source_frame_index": frame_index,
                    "chunk_index": chunk.chunk_index,
                }
                manifest.write(json.dumps(
                    manifest_row, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                ) + "\n")
                chunk_first_id = frame_index if chunk_first_id is None else chunk_first_id
                chunk_last_id = frame_index
                saved += 1
            chunk_reports.append({
                "file": chunk.path.name,
                "chunk_index": chunk.chunk_index,
                "frames": len(frames),
                "first_frame_index": chunk_first_id,
                "last_frame_index": chunk_last_id,
                "source_bytes": chunk.path.stat().st_size,
            })

    if saved == 0 or expected_size is None or first_elapsed is None or previous_elapsed is None:
        raise ValueError(f"Session '{first.session_id}' produced no usable images.")
    duration = max(0.0, previous_elapsed - first_elapsed)
    width, height = expected_size
    metadata = {
        "format": _CAPTURE_SCHEMA,
        "game_label": first.game_name,
        "source_type": "trss-sample-recorder-json",
        "source_session_id": first.session_id,
        "source_store": first.game_source,
        "width": width,
        "height": height,
        "frames_saved": saved,
        "elapsed_seconds": duration,
        "measured_output_fps": (saved - 1) / duration if duration > 0 and saved > 1 else None,
        "capture_type": "RGB-only visible game window",
        "limits": [
            "RGB color screenshots only; no depth, motion vectors, camera matrices, or renderer history.",
            "The later training input is synthetically downsampled from this captured target frame.",
            "The capture does not establish live-game reconstruction quality or GPU performance.",
        ],
    }
    _write_json(destination / "session.json", metadata)
    (destination / ".trss-desktop-capture-session").write_text(
        "TrSS Sample Recorder JSON import v1\n", encoding="utf-8", newline="\n"
    )
    return {
        "session_id": first.session_id,
        "game": first.game_name,
        "source": first.game_source,
        "directory": destination.name,
        "frames": saved,
        "width": width,
        "height": height,
        "chunks": chunk_reports,
    }


def import_sample_recorder_json(
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Validate JSON chunks, group all chunks by capture session, and extract frames."""
    chunks = _collect_chunks(inputs)
    grouped: dict[str, list[ChunkInfo]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.session_id, []).append(chunk)

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
    transaction = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.sample-recorder-import-", dir=output.parent
    ))
    staging = transaction / "payload"
    staging.mkdir()
    backup: Path | None = None
    try:
        reports: list[dict[str, Any]] = []
        used_destinations: set[str] = set()
        for session_id, session_chunks in sorted(grouped.items()):
            game_name = session_chunks[0].game_name
            folder_name = _session_folder_name(game_name, session_id)
            if folder_name.casefold() in used_destinations:
                raise ValueError("Two recorder sessions map to the same output directory.")
            used_destinations.add(folder_name.casefold())
            reports.append(_write_capture_session(session_chunks, staging / folder_name))

        report = {
            "schema_version": 1,
            "completed": True,
            "input_type": _SCHEMA,
            "output_dir": str(output),
            "source_chunk_count": len(chunks),
            "session_count": len(reports),
            "frame_count": sum(item["frames"] for item in reports),
            "maximum_chunk_bytes": _MAX_CHUNK_BYTES,
            "sessions": reports,
            "next_step": (
                "Pass this output folder to trss-gameplay-experiment with at least two "
                "independent capture sessions to keep train and validation data separated."
            ),
        }
        _write_json(staging / "sample-recorder-import-report.json", report)

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
        return report
    finally:
        shutil.rmtree(transaction, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract TrSS Sample Recorder JSON chunks into RGB capture sessions."
    )
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="Completed chunk JSON files, or folders containing chunk-*.json files.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        report = import_sample_recorder_json(args.inputs, args.output_dir, force=args.force)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "import_report": str(args.output_dir.expanduser().resolve() / "sample-recorder-import-report.json"),
        "sessions": report["session_count"],
        "frames": report["frame_count"],
        "output_dir": str(args.output_dir.expanduser().resolve()),
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
