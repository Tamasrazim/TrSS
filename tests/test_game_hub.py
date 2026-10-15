"""Portable tests for TrSS Game Library discovery helpers."""

import json
from pathlib import Path

from PIL import Image

from trss.game_hub import (
    GameEntry,
    _SequenceLog,
    _friendly_metric,
    _hex_rgb,
    _name_key,
    _parse_vdf_pairs,
    choose_game_executable,
    load_capture_root,
    parse_steam_manifest,
    parse_steam_libraries,
    save_capture_root,
)


def test_steam_vdf_parser_unescapes_windows_library_paths(tmp_path: Path) -> None:
    library = tmp_path / "SteamLibrary"
    library.mkdir()
    vdf_path = tmp_path / "libraryfolders.vdf"
    vdf_path.write_text(
        '"libraryfolders"\n{\n'
        '  "0" { "path" "' + str(library).replace("\\", "\\\\") + '" }\n'
        '  "1" { "path" "Z:\\\\Games\\\\SteamLibrary" }\n'
        '}\n',
        encoding="utf-8",
    )

    paths = parse_steam_libraries(vdf_path)

    assert library in paths
    assert all(path.is_dir() for path in paths)


def test_vdf_pair_parser_is_case_insensitive_for_keys() -> None:
    parsed = _parse_vdf_pairs('"AppID" "123" "Name" "Sample Game" "installdir" "Sample"')

    assert parsed["appid"] == "123"
    assert parsed["name"] == "Sample Game"
    assert parsed["installdir"] == "Sample"


def test_steam_manifest_requires_a_name_appid_and_install_directory(tmp_path: Path) -> None:
    manifest = tmp_path / "appmanifest_123.acf"
    manifest.write_text(
        '"AppState"\n{\n'
        ' "appid" "123"\n'
        ' "name" "Sample Racing Game"\n'
        ' "installdir" "Sample Racing Game"\n'
        '}\n',
        encoding="utf-8",
    )

    assert parse_steam_manifest(manifest) == {
        "app_id": "123",
        "name": "Sample Racing Game",
        "install_dir": "Sample Racing Game",
    }

    manifest.write_text('"AppState" { "appid" "123" }', encoding="utf-8")
    assert parse_steam_manifest(manifest) is None


def test_game_executable_selection_avoids_crash_handler(tmp_path: Path) -> None:
    game_folder = tmp_path / "Sample Racing Game"
    game_folder.mkdir()
    main_exe = game_folder / "SampleRacingGame.exe"
    main_exe.write_bytes(b"\0" * (2 * 1024 * 1024))
    crash_exe = game_folder / "UnityCrashHandler64.exe"
    crash_exe.write_bytes(b"\0" * (6 * 1024 * 1024))

    result = choose_game_executable(game_folder, "Sample Racing Game")

    assert result == main_exe


def test_game_entry_key_and_executable_name_are_stable(tmp_path: Path) -> None:
    exe = tmp_path / "A Game.exe"
    entry = GameEntry(name="A Game", exe_path=str(exe), install_dir=str(tmp_path))

    assert entry.exe_name == "A Game.exe"
    assert entry.key
    assert Path(entry.short_path).name == tmp_path.name


def test_icon_color_uses_the_executable_icon_png(tmp_path: Path) -> None:
    icon_path = tmp_path / "icon.png"
    Image.new("RGB", (64, 64), (35, 155, 215)).save(icon_path)

    assert _hex_rgb(str(icon_path)) == "#239bd7"
    # Games without an icon must use the fallback instead of crashing the UI.
    assert _hex_rgb(None) == "#aeb8c8"
    assert _hex_rgb("") == "#aeb8c8"
    assert _hex_rgb(str(tmp_path / "missing.png")) == "#aeb8c8"


def test_metric_display_never_fabricates_missing_values() -> None:
    assert _friendly_metric(None) == "—"
    assert _friendly_metric(52.25, "%") == "52.2%"
    assert _name_key("Red Dead Redemption 2") == "reddeadredemption2"



def test_capture_folder_setting_is_saved_and_invalid_paths_fall_back(tmp_path: Path) -> None:
    destination = tmp_path / "external-drive" / "TrSS"
    destination.mkdir(parents=True)
    settings_file = tmp_path / "app-data" / "capture-location.json"
    default_root = tmp_path / "default-captures"

    assert load_capture_root(settings_file, default_root) == default_root
    assert save_capture_root(destination, settings_file) == destination
    assert load_capture_root(settings_file, default_root) == destination

    settings_file.write_text(
        json.dumps({"schema_version": 1, "capture_root": "relative-folder"}),
        encoding="utf-8",
    )
    assert load_capture_root(settings_file, default_root) == default_root


def test_capture_folder_setting_does_not_accept_a_file_as_a_folder(tmp_path: Path) -> None:
    not_a_folder = tmp_path / "a-file.txt"
    not_a_folder.write_text("not a directory", encoding="utf-8")
    settings_file = tmp_path / "app-data" / "capture-location.json"

    try:
        save_capture_root(not_a_folder, settings_file)
    except ValueError as exc:
        assert "existing absolute capture folder" in str(exc)
    else:
        raise AssertionError("A file path must not be accepted as the capture directory")


def test_sequence_log_streams_frame_rows_to_disk(tmp_path: Path) -> None:
    sequence_path = tmp_path / "session" / "sequence.jsonl"
    rows = _SequenceLog(sequence_path)

    assert len(rows) == 0
    assert not rows
    rows.append({"frame_id": 0, "color": "frames/frame-000000.png"})
    rows.append({"frame_id": 1, "color": "frames/frame-000001.png"})

    saved = [json.loads(line) for line in sequence_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows
    assert saved == [
        {"frame_id": 0, "color": "frames/frame-000000.png"},
        {"frame_id": 1, "color": "frames/frame-000001.png"},
    ]
