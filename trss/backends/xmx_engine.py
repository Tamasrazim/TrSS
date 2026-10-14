"""Optional bridge from TrSS frame contracts to the vendored Intel Xe2 XMX runner.

This path needs a locally built Vulkan runtime, all required SPIR-V shaders, and a
compatible legal logical-weight safetensors file. It is intentionally separate from
the OpenVINO path: building this source is not proof of model or game compatibility.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np

from ..rendering.hnet import HNetControls, compose_head, make_features


class XMXBackendUnavailable(RuntimeError):
    """The native XMX runtime, device, or provided weight file cannot be used."""


@dataclass(frozen=True, slots=True)
class XMXFrameResult:
    """A native XMX frame plus measurements and the selected network field."""

    image: np.ndarray
    history: np.ndarray
    head: np.ndarray
    device_name: str
    network_size: tuple[int, int]
    preprocess_ms: float
    frame_runner_ms: float
    composition_ms: float
    total_ms: float
    allocated_buffer_bytes: int | None


def _module_is_from(module: ModuleType, root: Path) -> bool:
    """Avoid accidentally loading an unrelated module with the vendor's short name."""
    origin = getattr(module, "__file__", None)
    if not origin:
        return False
    try:
        return Path(origin).resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def _load_vendor_modules() -> SimpleNamespace:
    """Load the checked-in runner only after its native build artifacts exist."""
    repo_root = Path(__file__).resolve().parents[2]
    vendor_root = repo_root / "vendor" / "dlss-nr-intel"
    gpu_dir = vendor_root / "src" / "gpu"
    ref_dir = vendor_root / "src" / "ref"
    work_dir = vendor_root / "work"

    if not (gpu_dir / "nr_frame_resident.py").is_file():
        raise XMXBackendUnavailable(
            f"The vendored XMX frame runner is missing: {gpu_dir}"
        )

    library_name = "libxmx.dll" if os.name == "nt" else "libxmx.so"
    library_path = work_dir / library_name
    required_shaders = (
        "gemm_coopmat.spv",
        "gemm_resident.spv",
        "gemm_tiled.spv",
        "gemm_staged.spv",
        "gemm_staged32.spv",
        "gemm_staged32_deep.spv",
        "attention_rows.spv",
        "resident.spv",
        "attention.spv",
        "history.spv",
        "gemm_batched.spv",
        "gemm_f16acc.spv",
        "gemm_coopmat_int8.spv",
        "gemm_staged_int8.spv",
        "window_attention.spv",
        "window_block.spv",
        "global_attention.spv",
        "ffn_fused.spv",
    )
    missing = [str(path) for path in (library_path, *(work_dir / name for name in required_shaders))
               if not path.is_file()]
    if missing:
        examples = "\n".join(f"  - {item}" for item in missing[:6])
        more = f"\n  - ... and {len(missing) - 6} more" if len(missing) > 6 else ""
        raise XMXBackendUnavailable(
            "The native Vulkan/XMX build is incomplete. Missing artifacts:\n"
            f"{examples}{more}\n"
            f"Build the vendored project out of tree from {vendor_root} and keep its "
            "generated library and shaders in vendor/dlss-nr-intel/work/ as its CMake "
            "configuration expects."
        )

    # The upstream modules use short absolute imports (xmxres, nr_model and
    # nr_resident). Put only the checked-in source directories at the front.
    for directory in (str(gpu_dir), str(ref_dir)):
        while directory in sys.path:
            sys.path.remove(directory)
        sys.path.insert(0, directory)

    try:
        runner = importlib.import_module("nr_frame_resident")
        runtime_module = importlib.import_module("xmxres")
        model_module = importlib.import_module("nr_model")
        resident_module = importlib.import_module("nr_resident")
        xmx_module = importlib.import_module("xmx")
    except Exception as exc:
        raise XMXBackendUnavailable(
            f"Could not import the vendored XMX runner from {vendor_root}: {exc}"
        ) from exc

    modules = {
        "frame runner": runner,
        "runtime": runtime_module,
        "model loader": model_module,
        "resident kernels": resident_module,
        "XMX driver": xmx_module,
    }
    wrong = [
        f"{label}: {getattr(module, '__file__', '<unknown>')}"
        for label, module in modules.items()
        if not _module_is_from(module, vendor_root)
    ]
    if wrong:
        raise XMXBackendUnavailable(
            "An unrelated Python module shadows the vendored XMX source:\n"
            + "\n".join(wrong)
        )

    return SimpleNamespace(
        frame_runner=runner,
        runtime_module=runtime_module,
        model_module=model_module,
        resident_module=resident_module,
        xmx_module=xmx_module,
        vendor_root=vendor_root,
    )


class XMXFrameEngine:
    """Execute the vendored 16-channel neural-rendering graph on Intel Arc XMX."""

    def __init__(
        self,
        weights_path: str | Path,
        *,
        device_substring: str | None = None,
    ) -> None:
        self.weights_path = Path(weights_path).expanduser().resolve()
        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Logical model weights do not exist: {self.weights_path}")
        if device_substring is not None and not device_substring.strip():
            raise ValueError("device_substring must not be empty when provided.")

        self.runtime: Any | None = None
        self.device_name = ""
        self._modules: SimpleNamespace | None = None
        self._weights: dict[str, np.ndarray] = {}
        self._metadata: dict[str, Any] = {}
        self._device_weights: Any | None = None
        self._frame: Any | None = None
        self._frame_size: tuple[int, int] | None = None
        self._closed = False

        previous_device = os.environ.get("TrSS_VULKAN_DEVICE")
        if device_substring is not None:
            os.environ["TrSS_VULKAN_DEVICE"] = device_substring.strip()
        try:
            self._modules = _load_vendor_modules()
            self.runtime = self._modules.runtime_module.Runtime()
        except XMXBackendUnavailable:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise XMXBackendUnavailable(
                "Could not initialize the native Vulkan/XMX runtime. Check that the "
                "vendored library and shaders match the installed Vulkan driver and "
                f"toolchain: {exc}"
            ) from exc
        finally:
            # The C runtime reads this only when it first selects a Vulkan device.
            if device_substring is not None:
                if previous_device is None:
                    os.environ.pop("TrSS_VULKAN_DEVICE", None)
                else:
                    os.environ["TrSS_VULKAN_DEVICE"] = previous_device

        try:
            raw_name = self.runtime.lib.xmx_device()
            if isinstance(raw_name, bytes):
                self.device_name = raw_name.decode("utf-8", errors="replace")
            else:
                self.device_name = str(raw_name)
            if "arc" not in self.device_name.casefold():
                raise XMXBackendUnavailable(
                    "The native runtime selected a non-Arc device "
                    f"({self.device_name!r}); TrSS's XMX path requires Intel Arc."
                )

            self._weights, self._metadata = self._modules.model_module.load_logical(
                str(self.weights_path)
            )
            self._device_weights = self._modules.frame_runner.DeviceWeights(
                self.runtime, self._weights
            )
        except XMXBackendUnavailable:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise XMXBackendUnavailable(
                "Could not load the logical neural-rendering model or upload its weights. "
                "The file must be a compatible fully-logical safetensors model that is "
                f"licensed for this use: {exc}"
            ) from exc

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a copy of the logical model metadata."""
        return dict(self._metadata)

    def process_frame(
        self,
        color: np.ndarray,
        *,
        history: np.ndarray | None = None,
        frame_index: int = 0,
        controls: HNetControls = HNetControls(),
    ) -> XMXFrameResult:
        """Run one RGB float frame through XMX and compose its four-channel head."""
        if self._closed or self.runtime is None or self._device_weights is None:
            raise XMXBackendUnavailable("This XMXFrameEngine is closed or not initialized.")

        total_started = perf_counter()
        # Let the adapter reject complex or non-numeric input before any cast.
        source_color = np.asarray(color)
        features, geometry = make_features(
            source_color,
            frame_index=frame_index,
            history=history,
            controls=controls,
        )
        color = np.asarray(source_color, dtype=np.float32)
        preprocess_ms = (perf_counter() - total_started) * 1000.0

        network_height, network_width = features.shape[:2]
        network_size = (network_width, network_height)
        if self._frame is None or self._frame_size != network_size:
            if self._frame is not None:
                self._frame.close()
            try:
                self._frame = self._modules.frame_runner.ResidentFrame(
                    self.runtime,
                    self._device_weights,
                    network_height,
                    network_width,
                )
            except Exception as exc:
                raise XMXBackendUnavailable(
                    f"Could not allocate the native frame graph for {network_size}: {exc}"
                ) from exc
            self._frame_size = network_size

        runner_started = perf_counter()
        try:
            head = np.asarray(self._frame.run(features, execution="replay"), dtype=np.float32)
        except Exception as exc:
            raise XMXBackendUnavailable(
                "The native frame runner failed. Check that the logical weights match "
                f"the pinned 71-block graph and that the Arc driver supports its kernels: {exc}"
            ) from exc
        frame_runner_ms = (perf_counter() - runner_started) * 1000.0

        composition_started = perf_counter()
        source_history = color if history is None else np.asarray(history, dtype=np.float32)
        padded_color = geometry.extend(color)
        padded_history = geometry.extend(source_history)
        composed, next_history = compose_head(
            head,
            padded_color,
            history=padded_history,
        )
        cropped_image = geometry.crop(composed)
        cropped_history = geometry.crop(next_history)
        cropped_head = geometry.crop(head)
        composition_ms = (perf_counter() - composition_started) * 1000.0

        try:
            buffer_bytes = int(self.runtime.buffer_bytes)
        except Exception:
            buffer_bytes = None

        return XMXFrameResult(
            image=cropped_image,
            history=cropped_history,
            head=cropped_head,
            device_name=self.device_name,
            network_size=network_size,
            preprocess_ms=preprocess_ms,
            frame_runner_ms=frame_runner_ms,
            composition_ms=composition_ms,
            total_ms=(perf_counter() - total_started) * 1000.0,
            allocated_buffer_bytes=buffer_bytes,
        )

    def close(self) -> None:
        """Release frame and weight buffers; the process-owned Vulkan context is shared."""
        if self._closed:
            return
        self._closed = True
        if self._frame is not None:
            try:
                self._frame.close()
            except Exception:
                pass
            self._frame = None
        if self._device_weights is not None:
            try:
                self._device_weights.close()
            except Exception:
                pass
            self._device_weights = None
        self._weights.clear()
        self._metadata.clear()
