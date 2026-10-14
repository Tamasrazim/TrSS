import base64
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from trss.training.import_game_video import import_game_videos
from trss.training.import_sample_recorder import (
    _MAX_CHUNK_BYTES,
    import_sample_recorder_json,
)


def _jpeg(color: tuple[int, int, int], size: tuple[int, int] = (24, 16)) -> bytes:
    image = Image.new("RGB", size, color)
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=90)
    return stream.getvalue()


def _write_chunk(
    path: Path,
    *,
    session_id: str,
    chunk_index: int,
    frames: list[dict],
    game: str = "Test Game",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "trss.sample-recorder.chunk.v1",
        "sessionId": session_id,
        "chunkIndex": chunk_index,
        "game": {
            "name": game,
            "source": "Steam",
            "executableName": "testgame.exe",
        },
        "processId": 1234,
        "capture": {
            "warmupSeconds": 120,
            "sampleIntervalMs": 500,
            "maxJsonBytes": _MAX_CHUNK_BYTES,
            "imageFormat": "image/jpeg",
            "imageEncoding": "base64",
        },
        "frames": frames,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _frame(index: int, elapsed_ms: int, color: tuple[int, int, int], size=(24, 16)) -> dict:
    data = _jpeg(color, size)
    return {
        "frameIndex": index,
        "capturedAtUtc": f"2026-10-14T00:00:{index:02d}.000Z",
        "elapsedMs": elapsed_ms,
        "sourceWidth": size[0],
        "sourceHeight": size[1],
        "width": size[0],
        "height": size[1],
        "format": "image/jpeg",
        "imageBase64": base64.b64encode(data).decode("ascii"),
        "processCpuPercent": 2.5,
        "workingSetBytes": 1024,
    }


def test_importer_groups_chunks_and_emits_compatible_capture_sessions(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    _write_chunk(samples / "game-a" / "chunk-0002.json", session_id="session-a",
        chunk_index=2, frames=[_frame(1, 121000, (30, 40, 50)), _frame(2, 121500, (40, 50, 60))])
    _write_chunk(samples / "game-a" / "chunk-0001.json", session_id="session-a",
        chunk_index=1, frames=[_frame(0, 120500, (20, 30, 40))])
    _write_chunk(samples / "game-b" / "chunk-0001.json", session_id="session-b",
        chunk_index=1, frames=[_frame(0, 120250, (60, 70, 80))], game="Other Game")
    output = tmp_path / "imported"

    report = import_sample_recorder_json([samples], output)

    assert report["session_count"] == 2
    assert report["source_chunk_count"] == 3
    assert report["frame_count"] == 4
    session_dirs = sorted(
        path.parent for path in output.rglob("capture-manifest.jsonl")
    )
    assert len(session_dirs) == 2
    for session_dir in session_dirs:
        marker = session_dir / ".trss-desktop-capture-session"
        metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in (session_dir / "capture-manifest.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert marker.is_file()
        assert metadata["format"] == "trss-desktop-capture-session-v1"
        assert metadata["frames_saved"] == len(rows)
        assert all((session_dir / row["color"]).is_file() for row in rows)

    # These extracted folders can be passed directly to the existing data import path.
    dataset = tmp_path / "dataset"
    dataset_report = import_game_videos([output], dataset, max_frames_per_video=10)
    assert dataset_report["source_count"] == 2
    assert dataset_report["training_frames"] + dataset_report["validation_frames"] == 4


def test_importer_resizes_mid_session_resolution_changes(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    _write_chunk(samples / "chunk-0001.json", session_id="resolution-change",
        chunk_index=1, frames=[
            _frame(0, 120000, (40, 80, 120), (24, 16)),
            _frame(1, 120500, (80, 120, 160), (32, 20)),
        ])
    report = import_sample_recorder_json([samples], tmp_path / "out")
    session = next((tmp_path / "out").glob("test-game-*"))
    metadata = json.loads((session / "session.json").read_text(encoding="utf-8"))
    rows = (session / "capture-manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert report["frame_count"] == 2
    assert metadata["width"] == 24 and metadata["height"] == 16
    assert len(rows) == 2
    for row in rows:
        record = json.loads(row)
        with Image.open(session / record["color"]) as image:
            assert image.size == (24, 16)


def test_importer_rejects_invalid_base64_and_duplicate_chunk_indices(tmp_path: Path) -> None:
    bad = tmp_path / "chunk-0001.json"
    row = _frame(0, 120000, (1, 2, 3))
    row["imageBase64"] = "not base64!"
    _write_chunk(bad, session_id="bad", chunk_index=1, frames=[row])
    with pytest.raises(ValueError, match="invalid base64"):
        import_sample_recorder_json([bad], tmp_path / "out")

    samples = tmp_path / "duplicates"
    _write_chunk(samples / "chunk-0001-one.json", session_id="same", chunk_index=1,
        frames=[_frame(0, 120000, (1, 2, 3))])
    _write_chunk(samples / "chunk-0001-two.json", session_id="same", chunk_index=1,
        frames=[_frame(1, 120500, (4, 5, 6))])
    with pytest.raises(ValueError, match="duplicate chunk index"):
        import_sample_recorder_json([samples], tmp_path / "other")


def test_importer_protects_existing_output_and_rejects_oversized_json(tmp_path: Path) -> None:
    samples = tmp_path / "samples"
    _write_chunk(samples / "chunk-0001.json", session_id="one", chunk_index=1,
        frames=[_frame(0, 120000, (4, 5, 6))])
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not replace", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        import_sample_recorder_json([samples], output)
    assert sentinel.read_text(encoding="utf-8") == "do not replace"

    oversized = tmp_path / "oversized.json"
    with oversized.open("wb") as stream:
        stream.truncate(_MAX_CHUNK_BYTES + 1)
    with pytest.raises(ValueError, match="23 MiB"):
        import_sample_recorder_json([oversized], tmp_path / "unused")
