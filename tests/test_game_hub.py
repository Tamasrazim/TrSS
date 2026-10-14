"""Portable tests for TrSS Game Library discovery and performance summaries."""

import json
from pathlib import Path

from PIL import Image

from trss.game_hub import (
    GameEntry,
    _friendly_metric,
    _hex_rgb,
    _name_key,
    _parse_vdf_pairs,
    choose_game_executable,
    parse_steam_manifest,
    parse_steam_libraries,
    summarize_presentmon_samples,
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
    assert _hex_rgb(str(tmp_path / "missing.png")) == "#aeb8c8"


def test_metric_display_never_fabricates_missing_values() -> None:
    assert _friendly_metric(None) == "—"
    assert _friendly_metric(52.25, "%") == "52.2%"
    assert _name_key("Red Dead Redemption 2") == "reddeadredemption2"


def test_presentmon_summarizer_calculates_fps_and_frametime_from_real_samples() -> None:
    summary = summarize_presentmon_samples(
        [(8.0, 10.0), (9.0, 10.0), (7.5, 10.0), (10.0, 10.0), (8.5, 10.0)],
        last_sample_age_seconds=0.05,
    )

    assert summary["fps"] == 100.0
    assert summary["application_fps"] == 116.3
    assert summary["frametime_ms"] == 10.0
    assert summary["application_frametime_ms"] == 8.6
    assert summary["fps_samples"] == 5


def test_presentmon_summarizer_drops_stale_or_missing_samples() -> None:
    missing = summarize_presentmon_samples([], last_sample_age_seconds=None)
    stale = summarize_presentmon_samples(
        [(8.0, 8.0), (9.0, 9.0), (10.0, 10.0)],
        last_sample_age_seconds=3.0,
    )

    assert missing["fps"] is None
    assert stale["fps"] is None
    assert stale["fps_samples"] == 0
