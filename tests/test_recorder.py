"""Small tests for the recorder's portable data and bundle contract."""

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from trss.recorder import (
    CaptureError,
    _bundle_readme,
    _has_packaging_headroom,
    create_bundle,
    make_sequence_row,
    safe_label,
    verify_bundle_archive,
)


def test_safe_label_removes_path_and_control_characters() -> None:
    result = safe_label(r'  level: one\..\private/name <tag>  ', "fallback")
    assert "/" not in result
    assert "\\" not in result
    assert "<" not in result
    assert ">" not in result
    assert "\n" not in safe_label("scene\nlabel", "fallback")


def test_sequence_row_has_only_real_color_source_fields() -> None:
    row = make_sequence_row(
        frame_id=0,
        timestamp=0.01671,
        scene_id="trss-demo",
        color_path="frames/frame-000000.png",
    )
    assert row["frame_id"] == 0
    assert row["timestamp"] == 0.01671
    assert row["camera_cut"] is True
    assert row["color"] == "frames/frame-000000.png"
    assert row["color_space"] == "srgb"
    assert row["depth_encoding"] == "unknown"
    assert row["motion_units"] == "unknown"
    assert "depth" not in row
    assert "motion" not in row
    assert "initial_history" not in row


def test_bundle_is_valid_and_checksums_match(tmp_path: Path) -> None:
    session_dir = tmp_path / "session-test"
    (session_dir / "frames").mkdir(parents=True)
    frame_data = b"test-frame-payload"
    (session_dir / "frames" / "frame-000000.png").write_bytes(frame_data)

    rows = [
        make_sequence_row(
            frame_id=0,
            timestamp=0.0,
            scene_id="trss-test",
            color_path="frames/frame-000000.png",
        )
    ]
    zip_path = tmp_path / "session-test.zip"
    metadata = {
        "schema_version": "1.0",
        "session": {"id": "test-session", "status": "completed"},
        "privacy": {"automatic_upload": False},
    }
    system_metadata = {"operating_system": "TestOS"}

    result = create_bundle(
        session_dir=session_dir,
        zip_path=zip_path,
        metadata=metadata,
        system_metadata=system_metadata,
        sequence_rows=rows,
        keep_unpacked=True,
    )

    assert result == zip_path
    assert session_dir.is_dir()
    assert zipfile.is_zipfile(zip_path)
    verification = verify_bundle_archive(zip_path)
    assert verification["verified"] is True
    assert verification["files_verified"] >= 5
    with zipfile.ZipFile(zip_path, "r") as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        prefix = "session-test/"
        assert prefix + "metadata.json" in names
        assert prefix + "system.json" in names
        assert prefix + "sequence.jsonl" in names
        assert prefix + "frames/frame-000000.png" in names
        assert prefix + "SHA256SUMS.txt" in names

        saved_metadata = json.loads(archive.read(prefix + "metadata.json"))
        assert saved_metadata["privacy"]["automatic_upload"] is False

        manifest_line = archive.read(prefix + "sequence.jsonl").decode("utf-8").strip()
        parsed_row = json.loads(manifest_line)
        assert parsed_row["color"] == "frames/frame-000000.png"

        checksum_lines = archive.read(prefix + "SHA256SUMS.txt").decode("utf-8").splitlines()
        checksum_map = {
            line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in checksum_lines
        }
        expected = hashlib.sha256(frame_data).hexdigest()
        assert checksum_map["frames/frame-000000.png"] == expected


def test_bundle_only_removes_unpacked_copy_when_requested(tmp_path: Path) -> None:
    session_dir = tmp_path / "discard-after-zip"
    (session_dir / "frames").mkdir(parents=True)
    (session_dir / "frames" / "frame-000000.png").write_bytes(b"frame")
    zip_path = tmp_path / "discard-after-zip.zip"

    create_bundle(
        session_dir=session_dir,
        zip_path=zip_path,
        metadata={"session": {"id": "cleanup-test"}},
        system_metadata={},
        sequence_rows=[],
        keep_unpacked=False,
    )

    assert zip_path.is_file()
    assert not session_dir.exists()


def test_bundle_readme_is_honest_about_missing_render_buffers() -> None:
    text = _bundle_readme()
    assert "not a ready-to-train" in text.lower()
    assert "motion vectors" in text.lower()
    assert "does not upload" in text.lower()



def test_game_library_bundle_includes_and_verifies_performance_jsonl(tmp_path: Path) -> None:
    session_dir = tmp_path / "performance-session"
    (session_dir / "frames").mkdir(parents=True)
    (session_dir / "frames" / "frame-000000.png").write_bytes(b"frame")
    zip_path = tmp_path / "performance-session.zip"
    sample = {
        "timestamp_utc": "2026-10-15T00:00:00Z",
        "game_cpu_percent": 21.5,
        "game_gpu_3d_percent": None,
        "fps": None,
        "fps_source": "not implemented",
    }

    create_bundle(
        session_dir=session_dir,
        zip_path=zip_path,
        metadata={"session": {"id": "performance-session"}},
        system_metadata={},
        sequence_rows=[],
        keep_unpacked=True,
        performance_rows=[sample],
    )

    with zipfile.ZipFile(zip_path, "r") as archive:
        payload = archive.read(
            "performance-session/performance.jsonl"
        ).decode("utf-8").strip()
    assert json.loads(payload) == sample
    assert verify_bundle_archive(zip_path)["verified"] is True


def test_bundle_verification_detects_content_changed_after_packaging(tmp_path: Path) -> None:
    session_dir = tmp_path / "tamper-session"
    (session_dir / "frames").mkdir(parents=True)
    (session_dir / "frames" / "frame-000000.png").write_bytes(b"original")
    zip_path = tmp_path / "tamper-session.zip"
    create_bundle(
        session_dir=session_dir,
        zip_path=zip_path,
        metadata={"session": {"id": "tamper-session"}},
        system_metadata={},
        sequence_rows=[],
        keep_unpacked=True,
    )

    with zipfile.ZipFile(zip_path, "r") as original:
        entries = {
            name: original.read(name)
            for name in original.namelist()
        }
    entries["tamper-session/frames/frame-000000.png"] = b"changed bytes"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as edited:
        for name, data in entries.items():
            edited.writestr(name, data)

    with pytest.raises(CaptureError, match="SHA-256 verification failed"):
        verify_bundle_archive(zip_path)


def test_packaging_headroom_keeps_room_for_a_second_copy_of_the_frames() -> None:
    reserve = 512 * 1024 * 1024
    assert _has_packaging_headroom(reserve + 10_000, 10_000)
    assert not _has_packaging_headroom(reserve + 9_999, 10_000)
    assert _has_packaging_headroom(100, 50, reserve_bytes=50)
    assert not _has_packaging_headroom(99, 50, reserve_bytes=50)


def test_bundle_preserves_prebuilt_streamed_sequence_file(tmp_path: Path) -> None:
    session_dir = tmp_path / "streamed-session"
    (session_dir / "frames").mkdir(parents=True)
    (session_dir / "frames" / "frame-000000.png").write_bytes(b"frame")
    expected_row = {
        "frame_id": 0,
        "timestamp": 0.0,
        "color": "frames/frame-000000.png",
    }
    sequence_file = session_dir / "sequence.jsonl"
    sequence_file.write_text(
        json.dumps(expected_row, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    expected_sequence_bytes = sequence_file.read_bytes()
    zip_path = tmp_path / "streamed-session.zip"

    create_bundle(
        session_dir=session_dir,
        zip_path=zip_path,
        metadata={"session": {"captured_frames": 1}},
        system_metadata={},
        sequence_rows=None,
        keep_unpacked=True,
    )

    with zipfile.ZipFile(zip_path, "r") as archive:
        saved = archive.read("streamed-session/sequence.jsonl")
    # Preserve the original file bytes; newline style may be LF or CRLF on Windows.
    assert saved == expected_sequence_bytes
    assert verify_bundle_archive(zip_path)["verified"] is True
