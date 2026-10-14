"""Tests for TrSS's first-party graphics settings inventory and matrix export."""
import csv
import json
from pathlib import Path

from trss.game_matrix import (
    STANDARD_SETTINGS,
    append_game_matrix_row,
    build_matrix_row,
    parse_additional_settings,
    parse_graphics_config_file,
)


def test_xml_config_reader_extracts_only_allowlisted_graphics_values(tmp_path: Path) -> None:
    config = tmp_path / "system.xml"
    config.write_text(
        """<?xml version="1.0"?>
        <Settings>
          <API>Vulkan</API>
          <resolutionWidth>2560</resolutionWidth>
          <resolutionHeight>1440</resolutionHeight>
          <VSync value="On" />
          <textureQuality>Ultra</textureQuality>
          <motionBlur>false</motionBlur>
          <accountEmail>someone@example.invalid</accountEmail>
        </Settings>""",
        encoding="utf-8",
    )

    parsed = parse_graphics_config_file(config)

    assert parsed["values"]["graphics_api"] == "Vulkan"
    assert parsed["values"]["display_resolution"] == "2560x1440"
    assert parsed["values"]["vsync"] == "On"
    assert parsed["values"]["texture_quality"] == "Ultra"
    assert parsed["values"]["motion_blur"] == "false"
    assert "accountEmail" not in parsed["values"]
    assert all("someone" not in str(value) for value in parsed["values"].values())
    assert parsed["source_names"]["texture_quality"] == "system.xml"


def test_ini_reader_keeps_literal_values_and_ignores_unrelated_keys(tmp_path: Path) -> None:
    config = tmp_path / "GameUserSettings.ini"
    config.write_text(
        "[ScalabilityGroups]\n"
        "sg.TextureQuality=3\n"
        "sg.ShadowQuality=2\n"
        "bUseVSync=True\n"
        "UserToken=secret-value\n",
        encoding="utf-8",
    )

    parsed = parse_graphics_config_file(config)

    assert parsed["values"]["texture_quality"] == "3"
    assert parsed["values"]["shadow_quality"] == "2"
    assert parsed["values"]["vsync"] == "True"
    assert "UserToken" not in parsed["values"]


def test_json_reader_recurses_but_only_records_known_setting_names(tmp_path: Path) -> None:
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps({
            "video": {"resolution": "1920x1080", "upscalerMode": "Quality"},
            "privacy": {"accountName": "do-not-store"},
        }),
        encoding="utf-8",
    )

    parsed = parse_graphics_config_file(config)

    assert parsed["values"]["display_resolution"] == "1920x1080"
    assert parsed["values"]["upscaler_mode"] == "Quality"
    assert len(parsed["values"]) == 2


def test_profile_is_exact_only_when_confirmed_and_every_standard_field_is_filled() -> None:
    values = {field: "Not supported" for field in STANDARD_SETTINGS}
    values["graphics_api"] = "DirectX 12"
    values["display_resolution"] = "1920x1080"
    row = build_matrix_row(
        session_id="session-1",
        game_name="Sample Game",
        platform="Steam",
        started_utc="2026-10-14T00:00:00Z",
        elapsed_seconds=12.5,
        captured_frames=25,
        performance_rows=[{"fps": None, "game_cpu_percent": 25.0}],
        graphics_profile={
            "profile_name": "Baseline",
            "experiment_group": "smoke-test",
            "verified_in_game": True,
            "values": values,
            "sources": {"graphics_api": "participant_verified"},
            "additional_settings": {"GrassDistance": "High"},
        },
    )

    assert row["matrix_row_exact"] is True
    assert row["matrix_row_complete"] is True
    assert row["settings_coverage"]["coverage_percent"] == 100.0
    assert row["performance_summary"]["average_fps"] is None
    assert row["performance_summary"]["average_game_cpu_percent"] == 25.0
    assert len(row["settings_hash_sha256"]) == 64


def test_profile_with_missing_fields_is_not_marked_exact_even_if_checked() -> None:
    row = build_matrix_row(
        session_id="session-2",
        game_name="Sample Game",
        platform="Disk",
        started_utc="2026-10-14T00:00:00Z",
        elapsed_seconds=1.0,
        captured_frames=2,
        performance_rows=[],
        graphics_profile={"verified_in_game": True, "values": {"graphics_api": "Vulkan"}},
    )

    assert row["matrix_row_exact"] is False
    assert row["matrix_row_complete"] is False
    assert "display_resolution" in row["settings_coverage"]["missing_settings"]


def test_numeric_engine_enum_is_not_mislabelled_as_exact() -> None:
    values = {field: "Not supported" for field in STANDARD_SETTINGS}
    values["texture_quality"] = "3"
    row = build_matrix_row(
        session_id="enum-session",
        game_name="Sample Game",
        platform="Disk",
        started_utc="2026-10-14T00:00:00Z",
        elapsed_seconds=1.0,
        captured_frames=2,
        performance_rows=[],
        graphics_profile={
            "verified_in_game": True,
            "values": values,
            "sources": {"texture_quality": "local_game_config"},
        },
    )

    assert row["matrix_row_exact"] is False
    assert row["matrix_row_complete"] is False
    assert row["settings_coverage"]["settings_needing_value_decoding"] == ["texture_quality"]


def test_settings_hash_changes_when_a_setting_changes() -> None:
    values = {field: "Unknown" for field in STANDARD_SETTINGS}
    common = {
        "platform": "Disk",
        "started_utc": "2026-10-14T00:00:00Z",
        "elapsed_seconds": 1.0,
        "captured_frames": 1,
        "performance_rows": [],
        "graphics_profile": {"values": values, "verified_in_game": True},
    }
    first = build_matrix_row(session_id="a", game_name="Sample Game", **common)
    changed_values = dict(values, texture_quality="Ultra")
    second = build_matrix_row(
        session_id="b",
        game_name="Sample Game",
        **{**common, "graphics_profile": {"values": changed_values, "verified_in_game": True}},
    )

    assert first["settings_hash_sha256"] != second["settings_hash_sha256"]


def test_additional_settings_supports_game_specific_fields() -> None:
    settings = parse_additional_settings(
        "GrassDistance = 1.5\n"
        "VegetationDetail = Ultra\n"
        "not a setting\n"
        "= missing-name\n"
        "Empty =   \n"
    )

    assert settings == {"GrassDistance": "1.5", "VegetationDetail": "Ultra"}


def test_matrix_jsonl_and_csv_are_written_and_session_is_idempotent(tmp_path: Path) -> None:
    values = {field: "Not supported" for field in STANDARD_SETTINGS}
    row = build_matrix_row(
        session_id="stable-session",
        game_name="Sample Game",
        platform="Epic Games",
        started_utc="2026-10-14T00:00:00Z",
        elapsed_seconds=2.0,
        captured_frames=4,
        performance_rows=[],
        graphics_profile={"profile_name": "Preset A", "values": values, "verified_in_game": True},
    )

    jsonl_path, csv_path = append_game_matrix_row(tmp_path, row)
    append_game_matrix_row(tmp_path, row)

    stored = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))

    assert len(stored) == 1
    assert len(csv_rows) == 1
    assert csv_rows[0]["texture_quality"] == "Not supported"
    assert csv_rows[0]["matrix_row_exact"] == "True"
