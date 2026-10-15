"""Tests for the optional DirectX API CLI bridge; no Windows GPU is required."""

from __future__ import annotations

import pytest

import trss.cli as cli
from trss.backends.directx12_backend import (
    build_probe_command,
    locate_probe_executable,
)


@pytest.mark.parametrize(
    ("api", "filename"),
    [
        ("d3d9", "trss_d3d9_probe.exe"),
        ("d3d11", "trss_d3d11_probe.exe"),
        ("d3d12", "trss_dx12_probe.exe"),
    ],
)
def test_api_probe_uses_the_matching_native_executable_name(tmp_path, api, filename):
    probe_dir = tmp_path / "build" / "Release"
    probe_dir.mkdir(parents=True)
    expected = probe_dir / filename
    expected.write_bytes(b"test executable")

    from trss.backends import directx12_backend as backend
    monkeypatch = pytest.MonkeyPatch()
    fake_root = tmp_path / "repo"
    fake_module = fake_root / "trss" / "backends" / "directx12_backend.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.write_text("# test path", encoding="utf-8")
    monkeypatch.setattr(backend, "__file__", str(fake_module))
    try:
        target = fake_root / "native" / "directx12" / "build" / "Release" / filename
        target.parent.mkdir(parents=True)
        target.write_bytes(b"test executable")
        assert locate_probe_executable(api=api) == target.resolve()
    finally:
        monkeypatch.undo()


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("list-devices", ["--list-devices"]),
        ("capability-check", ["--capability-check"]),
        ("self-test", ["--self-test"]),
        ("warp-self-test", ["--warp-self-test"]),
    ],
)
def test_probe_command_selects_one_operation(tmp_path, operation, expected):
    executable = tmp_path / "probe.exe"
    command = build_probe_command(executable, operation=operation)
    assert command == [str(executable), *expected]


@pytest.mark.parametrize("api, filename", [
    ("d3d9", "trss_d3d9_probe.exe"),
    ("d3d11", "trss_d3d11_probe.exe"),
    ("d3d12", "trss_dx12_probe.exe"),
])
def test_probe_command_api_is_validated_and_remains_argv_safe(tmp_path, api, filename):
    executable = tmp_path / filename
    command = build_probe_command(
        executable, operation="self-test", device_substring='Arc "B580"', api=api
    )
    assert command == [
        str(executable), "--self-test", "--device-substring", 'Arc "B580"'
    ]


@pytest.mark.parametrize("operation", ["list-devices", "warp-self-test"])
def test_device_substring_is_rejected_for_non_hardware_modes(tmp_path, operation):
    with pytest.raises(ValueError, match="only valid for the hardware self-test"):
        build_probe_command(
            tmp_path / "probe.exe",
            operation=operation,
            device_substring="B580",
        )


def test_d3d9_rejects_warp_test_because_it_has_no_compute_shader_path(tmp_path):
    with pytest.raises(ValueError, match="D3D9 has no WARP compute test"):
        build_probe_command(
            tmp_path / "trss_d3d9_probe.exe",
            operation="warp-self-test",
            api="d3d9",
        )


def test_blank_device_substring_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must not be blank"):
        build_probe_command(
            tmp_path / "probe.exe",
            operation="self-test",
            device_substring="   ",
        )


def test_unsupported_api_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported DirectX API"):
        build_probe_command(
            tmp_path / "probe.exe",
            operation="self-test",
            api="opengl",  # type: ignore[arg-type]
        )

def test_d3d12_capability_check_is_not_offered_for_other_apis(tmp_path):
    for api in ("d3d9", "d3d11"):
        with pytest.raises(ValueError, match="supports Direct3D 12 only"):
            build_probe_command(
                tmp_path / "probe.exe",
                operation="capability-check",
                api=api,
            )


def test_cli_d3d12_capability_check_passes_adapter_selection(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cli,
        "run_dx12_probe",
        lambda **kwargs: captured.update(kwargs) or 0,
    )
    parsed = cli._build_parser().parse_args([
        "graphics-probe",
        "--api", "d3d12",
        "--capability-check",
        "--device-substring", "Intel Arc B580",
    ])

    assert cli._run_dx12_probe(parsed) == 0
    assert captured == {
        "operation": "capability-check",
        "executable": None,
        "device_substring": "Intel Arc B580",
        "api": "d3d12",
    }


def test_cli_rejects_capability_check_for_d3d11():
    parsed = cli._build_parser().parse_args([
        "graphics-probe", "--api", "d3d11", "--capability-check"
    ])
    with pytest.raises(ValueError, match="supports Direct3D 12 only"):
        cli._run_dx12_probe(parsed)


def test_explicit_missing_executable_fails_instead_of_searching_other_paths(tmp_path):
    missing = tmp_path / "not-built.exe"
    with pytest.raises(FileNotFoundError, match="not-built.exe"):
        locate_probe_executable(missing)


def test_cli_graphics_probe_selects_api_and_never_silently_uses_warp(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cli,
        "run_dx12_probe",
        lambda **kwargs: captured.update(kwargs) or 0,
    )
    parsed = cli._build_parser().parse_args([
        "graphics-probe",
        "--api", "d3d11",
        "--self-test",
        "--device-substring", "Intel Arc B580",
    ])

    assert cli._run_dx12_probe(parsed) == 0
    assert captured == {
        "operation": "self-test",
        "executable": None,
        "device_substring": "Intel Arc B580",
        "api": "d3d11",
    }


def test_cli_rejects_d3d9_warp_self_test():
    parsed = cli._build_parser().parse_args([
        "graphics-probe", "--api", "d3d9", "--warp-self-test"
    ])
    with pytest.raises(ValueError, match="Direct3D 9 has no WARP compute test"):
        cli._run_dx12_probe(parsed)
