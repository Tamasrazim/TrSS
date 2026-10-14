"""Launch the optional native Direct3D 12 compute bring-up executable."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Literal


ProbeOperation = Literal["list-devices", "self-test", "warp-self-test"]


class DirectX12BackendUnavailable(RuntimeError):
    """The native Direct3D 12 executable is not built or cannot be launched."""


def locate_probe_executable(executable: str | Path | None = None) -> Path:
    """Find the locally built probe without downloading or installing binaries."""
    if executable is not None:
        candidate = Path(executable).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"Direct3D 12 probe executable not found: {candidate}")
        return candidate

    root = Path(__file__).resolve().parents[2]
    configured = os.environ.get("TrSS_DX12_PROBE")
    candidates = (
        [Path(configured).expanduser().resolve()]
        if configured
        else [
            root / "native" / "directx12" / "build" / "Release" / "trss_dx12_probe.exe",
            root / "native" / "directx12" / "build" / "Debug" / "trss_dx12_probe.exe",
            root / "native" / "directx12" / "build" / "trss_dx12_probe.exe",
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    expected = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise DirectX12BackendUnavailable(
        "The native Direct3D 12 probe is not built. Build it from the repository root with:\n"
        "  cmake -S native/directx12 -B native/directx12/build -A x64\n"
        "  cmake --build native/directx12/build --config Release --parallel\n"
        f"Checked paths:\n{expected}"
    )


def build_probe_command(
    executable: str | Path,
    *,
    operation: ProbeOperation,
    device_substring: str | None = None,
) -> list[str]:
    """Build a predictable argv list; tests can verify it without Windows or a GPU."""
    if operation not in ("list-devices", "self-test", "warp-self-test"):
        raise ValueError(f"Unsupported Direct3D 12 probe operation: {operation!r}")
    if device_substring is not None:
        if not device_substring.strip():
            raise ValueError("device_substring must not be blank.")
        if operation != "self-test":
            raise ValueError(
                "device_substring is only valid for the hardware self-test."
            )

    command = [str(Path(executable))]
    command.append({
        "list-devices": "--list-devices",
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
) -> int:
    """Run the built native probe and return its exit code."""
    if os.name != "nt":
        raise DirectX12BackendUnavailable(
            "The native Direct3D 12 backend currently runs on Windows only. "
            "The existing Vulkan/XMX research backend remains separate."
        )

    probe = locate_probe_executable(executable)
    command = build_probe_command(
        probe,
        operation=operation,
        device_substring=device_substring,
    )
    try:
        completed = subprocess.run(command, check=False)
    except OSError as exc:
        raise DirectX12BackendUnavailable(
            f"Could not launch the native Direct3D 12 probe '{probe}': {exc}"
        ) from exc
    return int(completed.returncode)
