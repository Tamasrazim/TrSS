"""Small tests for the recorder's portable data and bundle contract."""

import hashlib
import json
from pathlib import Path
import zipfile

from trss.recorder import (
    _bundle_readme,
    create_bundle,
    make_sequence_row,
    safe_label,
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
