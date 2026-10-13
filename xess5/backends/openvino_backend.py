"""OpenVINO runtime discovery for Intel Arc GPU inference.

OpenVINO is loaded lazily so CPU-only unit tests and CLI help do not require
the inference runtime to be installed.
"""

from __future__ import annotations

from typing import Any


class BackendUnavailable(RuntimeError):
    """Raised when an inference backend is not installed or usable."""


def load_openvino() -> Any:
    """Import OpenVINO or return an actionable installation error."""
    try:
        import openvino as ov
    except ImportError as exc:
        raise BackendUnavailable(
            "OpenVINO is not installed. Activate your environment and run: "
            'python -m pip install -e ".[gpu]"'
        ) from exc
    return ov


def list_devices() -> list[dict[str, str]]:
    """Return the devices exposed by the local OpenVINO installation."""
    ov = load_openvino()
    core = ov.Core()
    devices: list[dict[str, str]] = []

    for device in core.available_devices:
        try:
            full_name = str(core.get_property(device, "FULL_DEVICE_NAME"))
        except Exception:
            full_name = device

        devices.append({
            "device": str(device),
            "name": full_name,
        })

    return devices
