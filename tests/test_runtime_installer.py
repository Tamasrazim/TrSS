"""Safety and behavior tests for the TrSS runtime installer."""
import json
from pathlib import Path
import struct

import pytest

import trss.runtime_installer as installer


def _write_pe(
    path: Path,
    *,
    machine: int = 0x8664,
    optional_magic: int = 0x20B,
    payload: bytes = b"",
) -> Path:
    """Write a minimal architecture-identifiable PE header for installer tests."""
    data = bytearray(0x80 + 26 + len(payload))
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = bytes((0x50, 0x45, 0, 0))
    struct.pack_into("<H", data, 0x84, machine)
    struct.pack_into("<H", data, 0x80 + 20, 0xF0)
    struct.pack_into("<H", data, 0x80 + 24, optional_magic)
    data[0x80 + 26:] = payload
    path.write_bytes(data)
    return path


@pytest.fixture
def install_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(installer, "IS_WINDOWS", True)
    game_dir = tmp_path / "Game"
    game_dir.mkdir()
    exe = _write_pe(game_dir / "ExampleGame.exe", payload=b"test-game")
    dll = tmp_path / "build" / "dxgi.dll"
    dll.parent.mkdir()
    _write_pe(dll, payload=b"first-party-runtime-test-binary")
    return game_dir, exe, dll


def test_installs_first_party_dll_and_creates_settings_and_presets(install_tree) -> None:
    game_dir, exe, dll = install_tree

    result = installer.install_runtime(exe, dll)

    target = game_dir / "dxgi.dll"
    assert target.read_bytes() == dll.read_bytes()
    assert result["runtime_sha256"] == installer.sha256_file(target)
    config = (game_dir / "TrSS.ini").read_text(encoding="utf-8")
    assert "[TrSSRuntime]" in config
    assert "Product=TrSS" in config
    for preset in ("Neutral", "Cinematic", "Vibrant", "Clarity", "Night", "Film"):
        preset_path = game_dir / "TrSS" / "Presets" / f"{preset}.ini"
        assert preset_path.is_file()
        assert "Product=TrSS" in preset_path.read_text(encoding="utf-8")
    manifest = json.loads(
        (game_dir / "TrSS" / installer.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["product"] == "TrSS Runtime"
    assert manifest["runtime_sha256"] == installer.sha256_file(target)
    assert manifest["network_upload"] is False


def test_installer_requires_a_64_bit_game_executable(install_tree) -> None:
    game_dir, exe, dll = install_tree
    _write_pe(exe, machine=0x14C, optional_magic=0x10B, payload=b"x86-game")

    with pytest.raises(installer.RuntimeInstallError, match="x64-only"):
        installer.install_runtime(exe, dll)

    assert not (game_dir / "dxgi.dll").exists()
    assert not (game_dir / "TrSS.ini").exists()


def test_installer_rejects_a_non_pe_runtime_dll(install_tree) -> None:
    game_dir, exe, dll = install_tree
    dll.write_bytes(b"not-a-Windows-library")

    with pytest.raises(installer.RuntimeInstallError, match="not a valid Windows PE"):
        installer.install_runtime(exe, dll)

    assert not (game_dir / "dxgi.dll").exists()


def test_refuses_to_overwrite_existing_dxgi_proxy(install_tree) -> None:
    game_dir, exe, dll = install_tree
    existing = game_dir / "dxgi.dll"
    existing.write_bytes(b"some-other-graphics-layer")

    with pytest.raises(installer.RuntimeInstallError, match="will not overwrite"):
        installer.install_runtime(exe, dll)

    assert existing.read_bytes() == b"some-other-graphics-layer"
    assert not (game_dir / "TrSS.ini").exists()


def test_refuses_to_overwrite_unidentified_config(install_tree) -> None:
    game_dir, exe, dll = install_tree
    config = game_dir / "TrSS.ini"
    config.write_text("[OtherTool]\nValue=leave-me-alone\n", encoding="utf-8")

    with pytest.raises(installer.RuntimeInstallError, match="will leave it untouched"):
        installer.install_runtime(exe, dll)

    assert config.read_text(encoding="utf-8") == "[OtherTool]\nValue=leave-me-alone\n"
    assert not (game_dir / "dxgi.dll").exists()


def test_uninstall_removes_only_verified_dll_and_preserves_user_settings(install_tree) -> None:
    game_dir, exe, dll = install_tree
    installer.install_runtime(exe, dll)
    config = game_dir / "TrSS.ini"
    config.write_text(config.read_text(encoding="utf-8") + "\n[User]\nCustomField=keep\n", encoding="utf-8")
    preset = game_dir / "TrSS" / "Presets" / "Custom.ini"
    preset.write_text("[TrSSPreset]\nProduct=TrSS\n[Effects]\nSaturation=1.6\n", encoding="utf-8")

    result = installer.uninstall_runtime(exe)

    assert not (game_dir / "dxgi.dll").exists()
    assert not (game_dir / "TrSS" / installer.MANIFEST_NAME).exists()
    assert "CustomField=keep" in config.read_text(encoding="utf-8")
    assert preset.is_file()
    assert result["runtime_removed"].endswith("dxgi.dll")


def test_failed_manifest_write_rolls_back_the_partial_install(
    install_tree, monkeypatch: pytest.MonkeyPatch
) -> None:
    game_dir, exe, dll = install_tree
    original_write_text = Path.write_text
    manifest_path = game_dir / "TrSS" / installer.MANIFEST_NAME

    def fail_only_on_manifest(path: Path, *args, **kwargs):
        if path == manifest_path:
            raise OSError("simulated disk write failure")
        return original_write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_only_on_manifest)

    with pytest.raises(OSError, match="simulated disk write failure"):
        installer.install_runtime(exe, dll)

    assert not (game_dir / "dxgi.dll").exists()
    assert not (game_dir / "dxgi.dll.trss-installing").exists()
    assert not (game_dir / "TrSS.ini").exists()
    assert not manifest_path.exists()
    assert not list((game_dir / "TrSS" / "Presets").glob("*.ini"))


def test_uninstall_refuses_if_installed_dll_was_replaced(install_tree) -> None:
    game_dir, exe, dll = install_tree
    installer.install_runtime(exe, dll)
    target = game_dir / "dxgi.dll"
    target.write_bytes(b"changed-after-install")

    with pytest.raises(installer.RuntimeInstallError, match="differs from the TrSS build"):
        installer.uninstall_runtime(exe)

    assert target.read_bytes() == b"changed-after-install"
    assert (game_dir / "TrSS" / installer.MANIFEST_NAME).exists()


def test_custom_preset_writer_does_not_execute_or_embed_any_code() -> None:
    text = installer.preset_text("Custom", {"Saturation": 1.4, "Sharpen": 0.5})

    assert "[TrSSPreset]" in text
    assert "Product=TrSS" in text
    assert "Saturation=1.4" in text
    assert "Sharpen=0.5" in text
    assert "import " not in text.casefold()
    assert "dll" not in text.casefold()


def test_runtime_lookup_includes_source_build_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = tmp_path / "build" / "runtime" / "Release" / "dxgi.dll"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"native runtime")
    monkeypatch.setattr(installer, "__file__", str(tmp_path / "repo" / "trss" / "runtime_installer.py"))
    monkeypatch.setattr(installer.sys, "executable", str(tmp_path / "python.exe"))
    monkeypatch.setattr(installer.sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    # Source-root lookup is repository-relative, so mirror the expected build path.
    source_runtime = tmp_path / "repo" / "build" / "runtime" / "Release" / "dxgi.dll"
    source_runtime.parent.mkdir(parents=True)
    source_runtime.write_bytes(b"expected source runtime")

    assert installer.find_runtime_dll() == source_runtime.resolve()
