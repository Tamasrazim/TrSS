"""OpenVINO discovery and strict device selection for Intel Arc inference.

This module deliberately never falls back to CPU. A renderer needs to know
which GPU is actually doing the work, and hybrid systems can expose several
OpenVINO GPU devices (for example, an integrated GPU and an Arc dGPU).
"""

from __future__ import annotations

from typing import Any


class BackendUnavailable(RuntimeError):
    """Raised when an inference backend is not installed or usable."""


def load_openvino() -> Any:
    """Import OpenVINO only when a GPU feature is requested."""
    try:
        import openvino as ov
    except ImportError as exc:
        raise BackendUnavailable(
            "OpenVINO is not installed. Activate your environment and run: "
            'python -m pip install -e ".[gpu]"'
        ) from exc
    return ov


def _property(core: Any, device: str, name: str, fallback: str = "") -> str:
    """Read a device property without making older plugin builds unusable."""
    try:
        return str(core.get_property(device, name))
    except Exception:
        return fallback


def describe_devices(core: Any) -> list[dict[str, str]]:
    """Return device IDs and names without hiding integrated/discrete GPUs."""
    devices: list[dict[str, str]] = []
    for device_value in core.available_devices:
        device = str(device_value)
        devices.append({
            "device": device,
            "name": _property(core, device, "FULL_DEVICE_NAME", device),
            "type": _property(core, device, "DEVICE_TYPE", "unknown"),
        })
    return devices


def list_devices() -> list[dict[str, str]]:
    """Return the devices exposed by the local OpenVINO installation."""
    ov = load_openvino()
    return describe_devices(ov.Core())


def resolve_device(core: Any, requested: str) -> str:
    """Resolve an explicit GPU ID or the ARC alias to one real device ID.

    ARC deliberately requires an Intel Arc name. If more than one Arc adapter
    is attached, the caller must select a specific GPU.X to avoid selecting an
    integrated adapter or the wrong discrete adapter silently.
    """
    requested = requested.strip()
    if not requested:
        raise BackendUnavailable("The device name cannot be empty.")

    devices = describe_devices(core)
    by_id = {item["device"]: item for item in devices}
    gpu_devices = [
        item for item in devices
        if item["device"].upper().startswith("GPU")
    ]

    if requested.upper() == "ARC":
        arc_devices = [
            item for item in gpu_devices
            if "arc" in item["name"].casefold()
        ]
        if len(arc_devices) == 1:
            return arc_devices[0]["device"]
        if len(arc_devices) > 1:
            choices = ", ".join(
                f'{item["device"]} ({item["name"]})' for item in arc_devices
            )
            raise BackendUnavailable(
                f"More than one Intel Arc device is available: {choices}. "
                "Select the exact device ID, for example --device GPU.1."
            )

        seen = ", ".join(
            f'{item["device"]} ({item["name"]})' for item in gpu_devices
        ) or "no GPU devices"
        raise BackendUnavailable(
            f"No Intel Arc adapter was identified by OpenVINO. Detected: {seen}. "
            "Check the Intel graphics driver and OpenVINO GPU plugin. If your Arc "
            "device is listed under an unexpected name, select its exact GPU.X ID "
            "explicitly after inspecting the trss devices output."
        )

    if requested.upper() == "GPU":
        # OpenVINO documents GPU as an alias for GPU.0. Use that documented
        # choice only when it exists; do not arbitrarily choose a GPU.X entry.
        if "GPU.0" in by_id:
            return "GPU.0"
        if "GPU" in by_id:
            return "GPU"
        raise BackendUnavailable(
            "OpenVINO's GPU alias is unavailable. Run trss devices and select "
            "an explicit GPU.X device ID."
        )

    if requested not in by_id:
        choices = ", ".join(
            f'{item["device"]} ({item["name"]})' for item in devices
        ) or "none"
        raise BackendUnavailable(
            f"Device {requested!r} is not available. Detected devices: {choices}."
        )

    if not requested.upper().startswith("GPU"):
        raise BackendUnavailable(
            f"Device {requested!r} is not a GPU. TRSS refuses CPU fallback; "
            "choose ARC or an explicit GPU.X ID."
        )

    return requested
