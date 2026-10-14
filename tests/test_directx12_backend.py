"""Tests for the optional Direct3D 12 CLI bridge; no Windows GPU is required."""

from __future__ import annotations

from pathlib import Path

import pytest

import trss.cli as cli
from trss.backends.directx12_backend import (
    DirectX12BackendUnavailable,
    build_probe_command,
    locate_probe_executable,
)


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("list-devices", ["--list-devices"]),
        ("self-test", ["--self-test"]),
        ("warp-self-test", ["--warp-self-test"]),
    ],
)
def test_probe_command_selects_one_operation(tmp_path, operation, expected):
    executable = tmp_path / "trss_dx12_probe.exe"
    command = build_probe_command(executable, operation=operation)
    assert command == [str(executable), *expected]


def test_hardware_device_substring_is_forwarded_without_being_rewritten(tmp_path):
    executable = tmp_path / "trss_dx12_probe.exe"
    command = build_probe_command(
        executable,
        operation="self-test",
        device_substring="Intel Arc B580",
    )
    assert command == [
        str(executable),
        "--self-test",
        "--device-substring",
        "Intel Arc B580",
    ]


@pytest.mark.parametrize("operation", ["list-devices", "warp-self-test"])
def test_device_substring_is_rejected_for_non_hardware_modes(tmp_path, operation):
    with pytest.raises(ValueError, match="only valid for the hardware self-test"):
        build_probe_command(
            tmp_path / "trss_dx12_probe.exe",
            operation=operation,
            device_substring="B580",
        )


def test_blank_device_substring_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must not be blank"):
        build_probe_command(
            tmp_path / "trss_dx12_probe.exe",
            operation="self-test",
            device_substring="   ",
        )


def test_explicit_missing_executable_fails_instead_of_searching_other_paths(tmp_path):
    missing = tmp_path / "not-built.exe"
    with pytest.raises(FileNotFoundError, match="not-built.exe"):
        locate_probe_executable(missing)


def test_located_executable_must_be_a_file(tmp_path):
    fake = tmp_path / "trss_dx12_probe.exe"
    fake.write_bytes(b"local test fixture")
    assert locate_probe_executable(fake) == fake.resolve()


@pytest.mark.parametrize(
    ("arguments", "expected_operation"),
    [
        (["--list-devices"], "list-devices"),
        (["--self-test"], "self-test"),
        (["--warp-self-test"], "warp-self-test"),
        ([], "self-test"),
    ],
)
def test_cli_dx12_probe_selects_requested_mode(monkeypatch, arguments, expected_operation):
    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "run_dx12_probe", fake_runner)
    parsed = cli._build_parser().parse_args(["dx12-probe", *arguments])

    assert cli._run_dx12_probe(parsed) == 0
    assert captured["operation"] == expected_operation
    assert captured["device_substring"] is None
    assert captured["executable"] is None


def test_cli_dx12_probe_never_silently_uses_warp_for_hardware_mode(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cli, "run_dx12_probe", lambda **kwargs: captured.update(kwargs) or 0
    )
    parsed = cli._build_parser().parse_args([
        "dx12-probe", "--device-substring", "Intel Arc B580"
    ])

    assert cli._run_dx12_probe(parsed) == 0
    assert captured["operation"] == "self-test"
    assert captured["device_substring"] == "Intel Arc B580"


def test_cli_dx12_probe_rejects_blank_device_substring():
    parsed = cli._build_parser().parse_args([
        "dx12-probe", "--device-substring", "   "
    ])
    with pytest.raises(ValueError, match="cannot be blank"):
        cli._run_dx12_probe(parsed)
