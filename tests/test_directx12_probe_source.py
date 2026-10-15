"""Regression checks for the native D3D12 diagnostic path.

These assertions run on any CI host; the Windows workflow separately compiles the
C++ probe and executes its compute roundtrip on WARP.
"""
from pathlib import Path


PROBE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "native"
    / "directx12"
    / "src"
    / "trss_dx12_probe.cpp"
)


def test_d3d12_has_a_standalone_capability_report() -> None:
    source = PROBE_SOURCE.read_text(encoding="utf-8")

    assert '"--capability-check"' in source
    assert "D3D12_FEATURE_DATA_D3D12_OPTIONS" in source
    assert "D3D12_FEATURE_DATA_D3D12_OPTIONS1" in source
    assert "D3D12_FEATURE_ROOT_SIGNATURE" in source
    assert "D3D12_FEATURE_ARCHITECTURE1" in source
    assert '"d3d12_capability_check"' in source
    assert "does not launch" in source
    assert "does not attach to or modify any game" in source


def test_d3d12_fence_wait_has_a_timeout() -> None:
    source = PROBE_SOURCE.read_text(encoding="utf-8")

    assert "kFenceWaitTimeoutMs = 30000" in source
    assert "WaitForSingleObject(\n            completion_event.get(), kFenceWaitTimeoutMs)" in source
    assert "WaitForSingleObject(completion_event.get(), INFINITE)" not in source
    assert "completed_value == UINT64_MAX" in source
    assert "GetDeviceRemovedReason()" in source


def test_native_d3d12_probe_requires_one_explicit_operation() -> None:
    source = PROBE_SOURCE.read_text(encoding="utf-8")

    assert "if (selected_modes != 1)" in source
    assert "Choose exactly one of --list-devices, --capability-check, --self-test" in source
