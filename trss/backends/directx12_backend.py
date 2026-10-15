"""Launch the optional native Direct3D 12 compute bring-up executable."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Literal


ProbeOperation = Literal["list-devices", "capability-check", "self-test", "warp-self-test"]
DirectXApi = Literal["d3d9", "d3d11", "d3d12"]

_DIRECTX_EXECUTABLES: dict[str, str] = {
    "d3d9": "trss_d3d9_probe.exe",
    "d3d11": "trss_d3d11_probe.exe",
    "d3d12": "trss_dx12_probe.exe",
}

_DIRECTX_ENVIRONMENT: dict[str, str] = {
    "d3d9": "TrSS_DX9_PROBE",
    "d3d11": "TrSS_DX11_PROBE",
    "d3d12": "TrSS_DX12_PROBE",
}


class DirectX12BackendUnavailable(RuntimeError):
    """The native Direct3D 12 executable is not built or cannot be launched."""


def locate_probe_executable(
    executable: str | Path | None = None, *, api: DirectXApi = "d3d12"
) -> Path:
    """Find the API-specific locally built probe; never download binaries."""
    if api not in _DIRECTX_EXECUTABLES:
        raise ValueError(f"Unsupported DirectX API: {api!r}")
    if executable is not None:
        candidate = Path(executable).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"DirectX API probe executable not found: {candidate}")
        return candidate

    root = Path(__file__).resolve().parents[2]
    configured = os.environ.get(_DIRECTX_ENVIRONMENT[api])
    executable_name = _DIRECTX_EXECUTABLES[api]
    candidates = (
        [Path(configured).expanduser().resolve()]
        if configured
        else [
            root / "native" / "directx12" / "build" / "Release" / executable_name,
            root / "native" / "directx12" / "build" / "Debug" / executable_name,
            root / "native" / "directx12" / "build" / executable_name,
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    expected = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise DirectX12BackendUnavailable(
        f"The native {api.upper()} probe ({executable_name}) is not built. "
        "Build all DirectX probes from the repository root with:\n"
        "  cmake -S native/directx12 -B native/directx12/build -A x64\n"
        "  cmake --build native/directx12/build --config Release --parallel\n"
        f"Checked paths:\n{expected}"
    )


def build_probe_command(
    executable: str | Path,
    *,
    operation: ProbeOperation,
    device_substring: str | None = None,
    api: DirectXApi = "d3d12",
) -> list[str]:
    """Build a predictable argv list; tests can verify it without Windows or a GPU."""
    if api not in _DIRECTX_EXECUTABLES:
        raise ValueError(f"Unsupported DirectX API: {api!r}")
    if operation not in ("list-devices", "self-test", "warp-self-test"):
        raise ValueError(f"Unsupported Direct3D 12 probe operation: {operation!r}")
    if api == "d3d9" and operation == "warp-self-test":
        raise ValueError("D3D9 has no WARP compute test; select a physical D3D9 adapter.")
    if device_substring is not None:
        if not device_substring.strip():
            raise ValueError("device_substring must not be blank.")
        if operation not in ("self-test", "capability-check"):
            raise ValueError(
                "device_substring is only valid for the hardware self-test or capability check."
            )

    command = [str(Path(executable))]
    if operation == "capability-check" and api != "d3d12":
        raise ValueError("The capability-check operation currently supports Direct3D 12 only.")
    command.append({
        "list-devices": "--list-devices",
        "capability-check": "--capability-check",
        "self-test": "--self-test",
        "warp-self-test": "--warp-self-test",
    }[operation])
    if device_substring is not None:
        command.extend(("--device-substring", device_substring.strip()))
    return command


def run_dx12_probe(
    *,
    operation: ProbeOperation,
    executable: str | Path | None = None,
    device_substring: str | None = None,
    api: DirectXApi = "d3d12",
) -> int:
    """Run the selected locally built DirectX API probe and return its exit code."""
    if os.name != "nt":
        raise DirectX12BackendUnavailable(
            "The native DirectX 9/11/12 API probes currently run on Windows only. "
            "The existing Vulkan/XMX research backend remains separate."
        )

    probe = locate_probe_executable(executable, api=api)
    command = build_probe_command(
        probe,
        operation=operation,
        device_substring=device_substring,
        api=api,
    )
    try:
        completed = subprocess.run(command, check=False)
    except OSError as exc:
        raise DirectX12BackendUnavailable(
            f"Could not launch the native {api.upper()} probe '{probe}': {exc}"
        ) from exc
    return int(completed.returncode)
