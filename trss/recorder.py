"""A local-first Windows capture tool for collecting opt-in TrSS research samples.

The recorder saves visible pixels from one selected application window. It does not
read game memory, attach to a process, record audio, upload files, or fabricate
depth/motion data. Captures are useful for visual review and compatibility research;
they are not automatically valid supervised-training pairs.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from typing import Any

try:
    from PIL import Image, ImageGrab
except ImportError as exc:  # Make a missing install helpful for people running source.
    raise SystemExit(
        "TrSS Data Recorder needs Pillow. Install the project with: python -m pip install -e ."
    ) from exc


APP_NAME = "TrSS Data Recorder"
APP_VERSION = "0.2.0"
CONSENT_POLICY_VERSION = "2026-10-15"
SAMPLE_RATE_OPTIONS = ("5", "10", "15", "30")
DURATION_OPTIONS = ("5", "10", "15", "30")
STOP_HOTKEY_TEXT = "Ctrl + Alt + F10"
HOTKEY_VK = 0x79  # F10. Checked while the capture window is minimized.
FRAME_DIRECTORY = "frames"
MIN_PACKAGING_FREE_BYTES = 512 * 1024 * 1024


def _has_packaging_headroom(
    free_bytes: int,
    captured_bytes: int,
    reserve_bytes: int = MIN_PACKAGING_FREE_BYTES,
) -> bool:
    """Return whether the disk can hold the ZIP while retaining the original PNGs."""
    return int(free_bytes) >= max(0, int(captured_bytes)) + max(0, int(reserve_bytes))


class CaptureError(RuntimeError):
    """Raised when a selected window can no longer be captured safely."""


@dataclass(frozen=True)
class WindowChoice:
    """A window handle and its title, shown only in the local UI."""

    hwnd: int
    title: str
    width: int
    height: int

    @property
    def label(self) -> str:
        # The title is for selection only; it is intentionally never written to a bundle.
        return f"{self.title[:90]}  ·  {self.width} × {self.height}  ·  #{self.hwnd:x}"


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


def _enable_dpi_awareness() -> None:
    """Ask Windows for physical-pixel coordinates so crops match the actual window."""
    if os.name != "nt":
        return
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        # Older Windows builds may not expose this call; capture still gets a chance.
        pass


def _window_client_bounds(hwnd: int) -> tuple[int, int, int, int]:
    """Return the visible client area in screen coordinates, excluding window borders."""
    if os.name != "nt":
        raise CaptureError("Privacy-safe application-window capture is currently Windows-only.")

    user32 = ctypes.windll.user32
    # Declare pointer-sized handle signatures so 64-bit HWND values are not truncated.
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(POINT)]
    user32.ClientToScreen.restype = wintypes.BOOL
    client = RECT()
    if not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
        raise CaptureError("The selected window was closed or minimized.")
    if not user32.GetClientRect(hwnd, ctypes.byref(client)):
        raise CaptureError("Windows could not read the selected window's bounds.")

    origin = POINT(client.left, client.top)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise CaptureError("Windows could not locate the selected window on screen.")

    width = client.right - client.left
    height = client.bottom - client.top
    if width < 64 or height < 64:
        raise CaptureError("The selected window is too small to record.")
    return origin.x, origin.y, origin.x + width, origin.y + height


def list_capture_windows(exclude_hwnd: int | None = None) -> list[WindowChoice]:
    """List visible, non-minimized application windows without saving their titles."""
    if os.name != "nt":
        return []

    user32 = ctypes.windll.user32
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    choices: list[WindowChoice] = []

    def visit(hwnd: int, _lparam: int) -> bool:
        if exclude_hwnd is not None and int(hwnd) == int(exclude_hwnd):
            return True
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length < 2:
            return True

        buffer = ctypes.create_unicode_buffer(min(length + 1, 512))
        user32.GetWindowTextW(hwnd, buffer, len(buffer))
        title = buffer.value.strip()
        if not title or title.casefold().startswith(APP_NAME.casefold()):
            return True

        try:
            left, top, right, bottom = _window_client_bounds(int(hwnd))
        except CaptureError:
            return True

        width, height = right - left, bottom - top
        if width < 240 or height < 160:
            return True

        choices.append(WindowChoice(int(hwnd), title, width, height))
        return True

    callback = callback_type(visit)
    user32.EnumWindows(callback, 0)
    return sorted(choices, key=lambda item: item.title.casefold())


def _foreground_window_handle() -> int:
    """Return the top-level window Windows currently considers active."""
    if os.name != "nt":
        return 0
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    active = user32.GetForegroundWindow()
    return int(active) if active else 0


def grab_window(hwnd: int, *, require_foreground: bool = False) -> Image.Image:
    """Capture the selected window's screen rectangle and reject accidental overlays."""
    bounds = _window_client_bounds(hwnd)
    if require_foreground and _foreground_window_handle() != int(hwnd):
        raise CaptureError(
            "The selected application is no longer the active window. The recorder stopped "
            "to avoid collecting another window by mistake. Re-select it and start again."
        )
    try:
        image = ImageGrab.grab(bbox=bounds, all_screens=True)
    except (OSError, RuntimeError) as exc:
        raise CaptureError(
            "Windows could not capture this window. Try borderless-windowed mode and "
            "make sure the window is visible."
        ) from exc

    expected = (bounds[2] - bounds[0], bounds[3] - bounds[1])
    if image.size != expected:
        raise CaptureError(
            "The captured image size did not match the window. Check Windows display scaling "
            "and try again."
        )
    # Focus can change while the screen crop is being read. Discard the frame rather than
    # include another app's pixels after an Alt+Tab or a pop-up takes over.
    if require_foreground and _foreground_window_handle() != int(hwnd):
        raise CaptureError(
            "Focus changed during capture. This frame was discarded and recording stopped "
            "to protect against capturing another window."
        )
    return image.convert("RGB")


def _total_memory_bytes() -> int | None:
    """Read installed RAM on Windows without collecting any device identifier."""
    if os.name != "nt":
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except (AttributeError, OSError):
        pass
    return None


def _windows_hardware_labels() -> dict[str, Any]:
    """Query only the CPU model and GPU names/driver versions; skip personal identifiers."""
    if os.name != "nt":
        return {}

    # This fixed PowerShell command reads local hardware labels only. It does not access
    # the network, list running processes, or read usernames, hostnames, or serial numbers.
    command = (
        "$cpu = (Get-CimInstance Win32_Processor | "
        "Select-Object -First 1 -ExpandProperty Name); "
        "$gpus = @(Get-CimInstance Win32_VideoController | "
        "Select-Object Name, DriverVersion, DriverDate); "
        "[ordered]@{ cpu = $cpu; gpus = $gpus } | ConvertTo-Json -Depth 4 -Compress"
    )
    no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            creationflags=no_window,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {}
        value = json.loads(result.stdout)
        if not isinstance(value, dict):
            return {}
        gpus = value.get("gpus", [])
        if isinstance(gpus, dict):
            gpus = [gpus]
        if not isinstance(gpus, list):
            gpus = []
        safe_gpus = []
        for gpu in gpus:
            if isinstance(gpu, dict):
                safe_gpus.append(
                    {
                        key: gpu.get(key)
                        for key in ("Name", "DriverVersion", "DriverDate")
                    }
                )
        return {
            "cpu_model": value.get("cpu"),
            "graphics_adapters": safe_gpus,
        }
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {}


def collect_system_metadata() -> dict[str, Any]:
    """Collect a small, explicitly documented hardware/software profile."""
    data: dict[str, Any] = {
        "operating_system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "total_physical_memory_bytes": _total_memory_bytes(),
        "hardware_labels": _windows_hardware_labels(),
        "privacy_note": (
            "No hostname, account name, IP address, device serial number, process list, "
            "clipboard, microphone, or audio is collected."
        ),
    }
    return data


def safe_label(value: str, fallback: str, limit: int = 80) -> str:
    """Keep user-provided labels short and safe for file names and JSON metadata."""
    cleaned = re.sub(r"[\x00-\x1f<>:\"/\\|?*]", " ", value)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned[:limit] or fallback


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def make_sequence_row(
    *,
    frame_id: int,
    timestamp: float,
    scene_id: str,
    color_path: str,
) -> dict[str, Any]:
    """Create a truthful TrSS color-sequence record with no invented auxiliary buffers."""
    return {
        "frame_id": int(frame_id),
        "timestamp": round(float(timestamp), 6),
        "scene_id": scene_id,
        "camera_cut": frame_id == 0,
        "color": color_path,
        "color_space": "srgb",
        "depth_encoding": "unknown",
        "motion_units": "unknown",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_checksums(session_dir: Path) -> None:
    """Create a deterministic integrity list for every bundle file except itself."""
    checksum_path = session_dir / "SHA256SUMS.txt"
    files = sorted(
        path for path in session_dir.rglob("*")
        if path.is_file() and path != checksum_path
    )
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(session_dir).as_posix()}"
        for path in files
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_bundle_archive(zip_path: Path) -> dict[str, Any]:
    """Check ZIP structure and verify every file against its SHA-256 manifest.

    The manifest is inside the archive, so verification does not depend on an unpacked
    session folder being kept. It also rejects duplicate and path-traversal archive names.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            damaged = archive.testzip()
            if damaged is not None:
                raise CaptureError(f"The finished ZIP failed its CRC check at {damaged}.")

            file_names = [
                info.filename for info in archive.infolist()
                if not info.is_dir()
            ]
            if len(file_names) != len(set(file_names)):
                raise CaptureError("The finished ZIP contains duplicate file entries.")
            if not file_names or any("/" not in name for name in file_names):
                raise CaptureError("The ZIP does not contain the expected session folder.")

            roots = {name.split("/", 1)[0] for name in file_names}
            if len(roots) != 1 or any(
                name.startswith("/") or "\\" in name
                or any(part in {"", ".", ".."} for part in name.split("/"))
                for name in file_names
            ):
                raise CaptureError("The ZIP contains an unsafe or unexpected file path.")

            prefix = next(iter(roots)) + "/"
            checksum_name = prefix + "SHA256SUMS.txt"
            if file_names.count(checksum_name) != 1:
                raise CaptureError("The ZIP does not contain one SHA256SUMS.txt manifest.")

            try:
                manifest_text = archive.read(checksum_name).decode("utf-8")
            except (UnicodeDecodeError, OSError) as exc:
                raise CaptureError("The checksum manifest could not be read.") from exc

            checksums: dict[str, str] = {}
            for line in manifest_text.splitlines():
                if not line:
                    continue
                match = re.fullmatch(r"([0-9a-fA-F]{64})  (.+)", line)
                if match is None:
                    raise CaptureError("The checksum manifest contains an invalid entry.")
                digest, relative_name = match.groups()
                if (
                    relative_name.startswith("/")
                    or "\\" in relative_name
                    or any(part in {"", ".", ".."} for part in relative_name.split("/"))
                    or relative_name in checksums
                ):
                    raise CaptureError("The checksum manifest contains an unsafe or duplicate path.")
                checksums[relative_name] = digest.lower()

            archive_files = {
                name[len(prefix):]
                for name in file_names
                if name.startswith(prefix) and name != checksum_name
            }
            if len(archive_files) != len(file_names) - 1 or set(checksums) != archive_files:
                raise CaptureError(
                    "The checksum manifest does not cover exactly the files in the ZIP."
                )

            for relative_name, expected_digest in checksums.items():
                digest = hashlib.sha256()
                with archive.open(prefix + relative_name, "r") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected_digest:
                    raise CaptureError(
                        f"SHA-256 verification failed for {relative_name}."
                    )

            return {
                "verified": True,
                "files_verified": len(checksums),
                "checksum_manifest": "SHA256SUMS.txt",
            }
    except zipfile.BadZipFile as exc:
        raise CaptureError("The finished ZIP is not a valid archive.") from exc


def _bundle_readme() -> str:
    return (
        "TrSS DATA RECORDER — CAPTURE BUNDLE\n"
        "===================================\n\n"
        "This bundle contains an opt-in sequence of visible pixels from a selected game "
        "window, basic hardware/driver labels, capture timing, and SHA-256 checksums. "
        "Some game-library captures also include performance.jsonl with measured Windows "
        "CPU, memory, and available per-process GPU counters. The recorder does not upload anything.\n\n"
        "IMPORTANT LIMITATION\n"
        "--------------------\n"
        "This is a visual capture sample for review and compatibility research. A screen "
        "capture does not contain renderer depth, motion vectors, reprojected history, or "
        "a matched high-resolution ground-truth target. It is NOT a ready-to-train TrSS "
        "supervised dataset by itself. Never invent missing buffers or label them as real.\n\n"
        "FILES\n"
        "-----\n"
        "metadata.json       Capture settings, timing, consent version, and privacy facts.\n"
        "system.json         OS, CPU/GPU model and driver labels, and installed RAM when available.\n"
        "sequence.jsonl      Per-frame color sequence with elapsed timestamps.\n"
        "frames/             Lossless PNG frames from the selected window client area.\n"
        "performance.jsonl   Present only when the game library collects performance samples.\n"
        "SHA256SUMS.txt      SHA-256 hashes for the other files in this bundle.\n\n"
        "PRIVACY\n"
        "-------\n"
        "No audio, microphone, clipboard, process list, host name, account name, IP address, "
        "or device serial number is collected. The selected window's title is used only in "
        "the local picker and is not saved. Frame images may still contain visible personal "
        "information inside the chosen application; review the capture before sharing it.\n\n"
        "CONTRIBUTION CHECK\n"
        "------------------\n"
        "Before sharing, confirm that you are allowed to capture and share the displayed "
        "content and that the bundle contains no private information. The consent record "
        "records checkbox confirmation, not a legal determination of ownership or permission.\n"
    )


def create_bundle(
    *,
    session_dir: Path,
    zip_path: Path,
    metadata: dict[str, Any],
    system_metadata: dict[str, Any],
    sequence_rows: list[dict[str, Any]] | None,
    keep_unpacked: bool,
    performance_rows: list[dict[str, Any]] | None = None,
) -> Path:
    """Write reports, hash the files, package them, and verify the ZIP before cleanup."""
    _write_json(session_dir / "metadata.json", metadata)
    _write_json(session_dir / "system.json", system_metadata)
    sequence_path = session_dir / "sequence.jsonl"
    if sequence_rows is None:
        # The Game Library streams sequence rows during capture to stay memory-safe.
        if not sequence_path.is_file():
            raise CaptureError("The streamed sequence.jsonl file is missing.")
    else:
        with sequence_path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in sequence_rows:
                stream.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
    if performance_rows is not None:
        with (session_dir / "performance.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            for row in performance_rows:
                stream.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
    (session_dir / "README.txt").write_text(_bundle_readme(), encoding="utf-8")
    _write_checksums(session_dir)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as bundle:
            for path in sorted(item for item in session_dir.rglob("*") if item.is_file()):
                archive_name = f"{session_dir.name}/{path.relative_to(session_dir).as_posix()}"
                bundle.write(path, archive_name)

        verify_bundle_archive(zip_path)
    except Exception:
        # Never leave a partial ZIP beside a recoverable source folder.
        zip_path.unlink(missing_ok=True)
        raise

    if not keep_unpacked:
        shutil.rmtree(session_dir)
    return zip_path


def _hotkey_is_down() -> bool:
    """Detect Ctrl+Alt+F10 while the UI is minimized, without installing a key-hook package."""
    if os.name != "nt":
        return False
    user32 = ctypes.windll.user32
    down = 0x8000
    return bool(
        user32.GetAsyncKeyState(0x11) & down
        and user32.GetAsyncKeyState(0x12) & down
        and user32.GetAsyncKeyState(HOTKEY_VK) & down
    )


def _wait_until(deadline: float, stop_event: threading.Event) -> bool:
    """Wait in short intervals so the global early-stop hotkey stays responsive."""
    while not stop_event.is_set():
        if _hotkey_is_down():
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        stop_event.wait(min(remaining, 0.05))
    return False


class TrSSRecorderApp:
    """Dark, minimal desktop UI and capture-session coordinator."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("900x800")
        self.root.minsize(760, 700)
        self.root.configure(bg="#0d0f12")

        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()
        self.is_recording = False
        self.window_choices: dict[str, WindowChoice] = {}
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.last_zip: Path | None = None

        self.game_var = tk.StringVar()
        self.scene_var = tk.StringVar()
        self.window_var = tk.StringVar()
        self.sample_rate_var = tk.StringVar(value="10")
        self.duration_var = tk.StringVar(value="10")
        self.output_var = tk.StringVar(
            value=str(Path.home() / "Documents" / "TrSS-Captures")
        )
        self.share_consent_var = tk.BooleanVar(value=False)
        self.rights_consent_var = tk.BooleanVar(value=False)
        self.keep_unpacked_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Choose an application window to begin.")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._configure_styles()
        self._build_ui()
        self.refresh_windows()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._drain_events)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", font=("Segoe UI", 10))
        style.configure("TFrame", background="#0d0f12")
        style.configure("Card.TFrame", background="#16191e")
        style.configure("TLabel", background="#0d0f12", foreground="#e7eaf0")
        style.configure("Muted.TLabel", background="#0d0f12", foreground="#9aa3af")
        style.configure("Card.TLabel", background="#16191e", foreground="#e7eaf0")
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 22), foreground="#f5f6f8")
        style.configure("Section.TLabel", background="#16191e", font=("Segoe UI Semibold", 11))
        style.configure(
            "TEntry",
            fieldbackground="#0f1216",
            foreground="#f0f2f5",
            insertcolor="#f0f2f5",
            padding=7,
        )
        style.configure(
            "TCombobox",
            fieldbackground="#0f1216",
            background="#242a32",
            foreground="#f0f2f5",
            arrowcolor="#f0f2f5",
            padding=6,
        )
        style.map("TCombobox", fieldbackground=[("readonly", "#0f1216")])
        style.configure("TButton", padding=(11, 8), background="#252a32", foreground="#f0f2f5")
        style.map("TButton", background=[("active", "#343b46"), ("disabled", "#191d23")])
        style.configure(
            "Accent.TButton",
            padding=(14, 10),
            background="#f0f2f5",
            foreground="#101216",
            font=("Segoe UI Semibold", 10),
        )
        style.map("Accent.TButton", background=[("active", "#d3d7de"), ("disabled", "#555b65")])
        style.configure("TCheckbutton", background="#16191e", foreground="#dfe3e9")
        style.map("TCheckbutton", background=[("active", "#16191e")])
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#252a32",
            background="#cfd5de",
            bordercolor="#252a32",
            lightcolor="#cfd5de",
            darkcolor="#cfd5de",
        )

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=(24, 20))
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 16))
        title_block = ttk.Frame(header)
        title_block.pack(side="left", fill="x", expand=True)
        ttk.Label(title_block, text="TrSS / DATA RECORDER", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            title_block,
            text="Opt-in visual samples for graphics research · Local-first · No uploads",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            header,
            text="WINDOW CAPTURE  /  PNG",
            padding=(10, 6),
            background="#1b2027",
            foreground="#c8ced8",
        ).pack(side="right", anchor="n")

        setup = ttk.Frame(outer, style="Card.TFrame", padding=16)
        setup.pack(fill="x")
        ttk.Label(setup, text="CAPTURE SETUP", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 12)
        )

        ttk.Label(setup, text="Game / application", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=4
        )
        self.game_entry = ttk.Entry(setup, textvariable=self.game_var)
        self.game_entry.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(setup, text="Scene label (optional)", style="Card.TLabel").grid(
            row=2, column=0, sticky="w", pady=4
        )
        ttk.Entry(setup, textvariable=self.scene_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=4
        )

        ttk.Label(setup, text="Application window", style="Card.TLabel").grid(
            row=3, column=0, sticky="w", pady=4
        )
        self.window_combo = ttk.Combobox(
            setup, textvariable=self.window_var, state="readonly", values=()
        )
        self.window_combo.grid(row=3, column=1, sticky="ew", pady=4)
        actions = ttk.Frame(setup, style="Card.TFrame")
        actions.grid(row=3, column=2, sticky="e", padx=(8, 0))
        ttk.Button(actions, text="Refresh", command=self.refresh_windows).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(actions, text="Preview", command=self.preview_window).pack(side="left")

        ttk.Label(setup, text="Sample rate", style="Card.TLabel").grid(
            row=4, column=0, sticky="w", pady=4
        )
        settings_row = ttk.Frame(setup, style="Card.TFrame")
        settings_row.grid(row=4, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Combobox(
            settings_row, textvariable=self.sample_rate_var, state="readonly", values=SAMPLE_RATE_OPTIONS, width=8
        ).pack(side="left")
        ttk.Label(settings_row, text="PNG samples / second", style="Card.TLabel").pack(
            side="left", padx=(8, 26)
        )
        ttk.Label(settings_row, text="Duration", style="Card.TLabel").pack(side="left", padx=(0, 8))
        ttk.Combobox(
            settings_row,
            textvariable=self.duration_var,
            state="readonly",
            values=DURATION_OPTIONS,
            width=8,
        ).pack(side="left")
        ttk.Label(settings_row, text="seconds", style="Card.TLabel").pack(
            side="left", padx=(8, 0)
        )

        ttk.Label(setup, text="Save to", style="Card.TLabel").grid(
            row=5, column=0, sticky="w", pady=4
        )
        ttk.Entry(setup, textvariable=self.output_var).grid(
            row=5, column=1, sticky="ew", pady=4
        )
        ttk.Button(setup, text="Browse…", command=self.choose_output).grid(
            row=5, column=2, sticky="e", padx=(8, 0), pady=4
        )
        setup.columnconfigure(1, weight=1)

        privacy = ttk.Frame(outer, style="Card.TFrame", padding=16)
        privacy.pack(fill="x", pady=(12, 0))
        ttk.Label(privacy, text="BEFORE YOU RECORD", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            privacy,
            text=(
                "The selected window's screen area is sampled; keep it active while recording. "
                "Recording stops if another window takes focus. Audio, clipboard, process lists, "
                "host/account names, and device serials are not collected. Use borderless/windowed "
                "mode; exclusive-fullscreen capture may be black. Disable notifications and remove "
                "private information before sharing."
            ),
            style="Card.TLabel",
            wraplength=820,
            justify="left",
        ).pack(anchor="w", fill="x", pady=(8, 8))
        ttk.Checkbutton(
            privacy,
            text="I have permission to capture and share the displayed content.",
            variable=self.rights_consent_var,
        ).pack(anchor="w", pady=3)
        ttk.Checkbutton(
            privacy,
            text="I agree to save this capture bundle locally and share it with the TrSS project.",
            variable=self.share_consent_var,
        ).pack(anchor="w", pady=3)
        ttk.Checkbutton(
            privacy,
            text="Keep the unpacked session folder after the ZIP has been verified.",
            variable=self.keep_unpacked_var,
        ).pack(anchor="w", pady=3)

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(14, 8))
        self.start_button = ttk.Button(
            controls, text="Start capture", style="Accent.TButton", command=self.start_capture
        )
        self.start_button.pack(side="left")
        ttk.Label(
            controls,
            text=f"To stop early: {STOP_HOTKEY_TEXT}",
            style="Muted.TLabel",
        ).pack(side="left", padx=(12, 0), pady=(2, 0))
        ttk.Button(controls, text="Open output folder", command=self.open_output).pack(side="right")

        self.progress = ttk.Progressbar(
            outer, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.progress.pack(fill="x", pady=(0, 8))
        ttk.Label(outer, textvariable=self.status_var, style="Muted.TLabel").pack(anchor="w")

        preview_card = ttk.Frame(outer, style="Card.TFrame", padding=10)
        preview_card.pack(fill="both", expand=True, pady=(12, 0))
        ttk.Label(preview_card, text="CAPTURE PREVIEW", style="Section.TLabel").pack(anchor="w")
        self.preview_label = ttk.Label(
            preview_card,
            text="Choose a window, then select Preview to check exactly what will be captured.",
            style="Card.TLabel",
            anchor="center",
            justify="center",
        )
        self.preview_label.pack(fill="both", expand=True, pady=(8, 0))

    def refresh_windows(self) -> None:
        if os.name != "nt":
            self.window_combo["values"] = ()
            self.window_var.set("")
            self.status_var.set("Capture is currently available on Windows 10/11 only.")
            self.start_button.state(["disabled"])
            return

        try:
            self.root.update_idletasks()
            choices = list_capture_windows(exclude_hwnd=self.root.winfo_id())
        except Exception as exc:
            self.status_var.set(f"Could not list windows: {exc}")
            return

        previous = self.window_var.get()
        self.window_choices = {choice.label: choice for choice in choices}
        labels = list(self.window_choices)
        self.window_combo["values"] = labels
        if previous in self.window_choices:
            self.window_var.set(previous)
        elif labels:
            self.window_var.set(labels[0])
        else:
            self.window_var.set("")
        self.status_var.set(
            f"{len(labels)} visible application window(s) available. Preview before recording."
        )

    def _selected_window(self) -> WindowChoice | None:
        return self.window_choices.get(self.window_var.get())

    def preview_window(self) -> None:
        choice = self._selected_window()
        if choice is None:
            messagebox.showwarning(APP_NAME, "Select a visible application window first.")
            return

        # A screen crop would show this UI if we captured before minimizing. Switch away,
        # focus the chosen app, then sample only after the recorder has disappeared.
        self.status_var.set("Switching to the selected window for a clean preview…")
        self.root.iconify()
        try:
            user32 = ctypes.windll.user32
            user32.SetForegroundWindow.argtypes = [wintypes.HWND]
            user32.SetForegroundWindow.restype = wintypes.BOOL
            user32.SetForegroundWindow(choice.hwnd)
        except (AttributeError, OSError):
            pass
        self.root.after(500, lambda: self._finish_preview(choice))

    def _finish_preview(self, choice: WindowChoice) -> None:
        error_message: str | None = None
        try:
            frame = grab_window(choice.hwnd, require_foreground=True)
            frame.thumbnail((760, 340), Image.Resampling.LANCZOS)
            self.preview_photo = ImageTk.PhotoImage(frame)
            self.preview_label.configure(image=self.preview_photo, text="")
            self.status_var.set(
                f"Preview verified: {choice.width} × {choice.height}. "
                "Only the active selected window will be sampled; its title is not saved."
            )
        except CaptureError as exc:
            self.status_var.set("Preview was not captured.")
            error_message = str(exc)
        finally:
            self.root.deiconify()
            self.root.state("normal")
            self.root.lift()

        if error_message:
            messagebox.showerror(APP_NAME, error_message)

    def choose_output(self) -> None:
        folder = filedialog.askdirectory(
            title="Choose where to save TrSS capture bundles",
            initialdir=self.output_var.get() or str(Path.home()),
            mustexist=False,
        )
        if folder:
            self.output_var.set(folder)

    def open_output(self) -> None:
        folder = Path(self.output_var.get()).expanduser()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            else:
                messagebox.showinfo(APP_NAME, str(folder))
        except OSError as exc:
            messagebox.showerror(APP_NAME, f"Could not open output folder: {exc}")

    def start_capture(self) -> None:
        if self.is_recording:
            return
        if os.name != "nt":
            messagebox.showerror(
                APP_NAME,
                "This build uses Windows application-window capture so it never needs to record "
                "the entire desktop. Windows 10/11 is required.",
            )
            return
        if not self.rights_consent_var.get() or not self.share_consent_var.get():
            messagebox.showwarning(
                APP_NAME,
                "Both permission and data-sharing confirmations must be checked before capture.",
            )
            return

        game = safe_label(self.game_var.get(), "", limit=80)
        if not game:
            messagebox.showwarning(APP_NAME, "Enter the game or application name.")
            self.game_entry.focus_set()
            return
        choice = self._selected_window()
        if choice is None:
            messagebox.showwarning(APP_NAME, "Select a visible application window first.")
            return

        output_root = Path(self.output_var.get()).expanduser()
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            if not output_root.is_dir():
                raise OSError("The selected output path is not a folder.")
        except OSError as exc:
            messagebox.showerror(APP_NAME, f"Could not use the output folder: {exc}")
            return

        sample_rate = int(self.sample_rate_var.get())
        duration = int(self.duration_var.get())
        max_frames = sample_rate * duration
        bytes_equivalent = choice.width * choice.height * 3 * max_frames
        try:
            free_bytes = shutil.disk_usage(output_root).free
            # PNG files are content-dependent; this is a conservative planning signal, not
            # a prediction of ZIP size.
            if free_bytes < bytes_equivalent // 4:
                proceed = messagebox.askyesno(
                    APP_NAME,
                    "The selected drive has less free space than one quarter of the raw RGB "
                    "equivalent for this session. PNG output varies greatly with scene content. "
                    "Continue anyway?",
                )
                if not proceed:
                    return
        except OSError:
            pass

        session_id = str(uuid.uuid4())
        session_name = f"trss-capture-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{session_id[:8]}"
        session_dir = output_root / session_name
        zip_path = output_root / f"{session_name}.zip"
        try:
            session_dir.mkdir(parents=False, exist_ok=False)
            (session_dir / FRAME_DIRECTORY).mkdir()
        except OSError as exc:
            messagebox.showerror(APP_NAME, f"Could not create the capture folder: {exc}")
            return

        game_label = game
        scene_label = safe_label(self.scene_var.get(), "", limit=80) or None
        self.is_recording = True
        self.stop_event.clear()
        self.progress_var.set(0)
        self.start_button.state(["disabled"])
        self.status_var.set(
            "Starting in 2 seconds. The recorder will minimize and focus the selected window."
        )
        self.root.after(
            600,
            lambda: self._minimize_then_record(
                choice=choice,
                session_id=session_id,
                session_name=session_name,
                session_dir=session_dir,
                zip_path=zip_path,
                output_root=output_root,
                game_label=game_label,
                scene_label=scene_label,
                sample_rate=sample_rate,
                duration=duration,
                keep_unpacked=self.keep_unpacked_var.get(),
            ),
        )

    def _minimize_then_record(
        self,
        *,
        choice: WindowChoice,
        session_id: str,
        session_name: str,
        session_dir: Path,
        zip_path: Path,
        output_root: Path,
        game_label: str,
        scene_label: str | None,
        sample_rate: int,
        duration: int,
        keep_unpacked: bool,
    ) -> None:
        # Minimize before the first frame so this tool never records its own interface.
        self.root.iconify()
        try:
            user32 = ctypes.windll.user32
            user32.SetForegroundWindow.argtypes = [wintypes.HWND]
            user32.SetForegroundWindow.restype = wintypes.BOOL
            user32.SetForegroundWindow(choice.hwnd)
        except (AttributeError, OSError):
            pass

        worker = threading.Thread(
            target=self._record_worker,
            kwargs={
                "choice": choice,
                "session_id": session_id,
                "session_name": session_name,
                "session_dir": session_dir,
                "zip_path": zip_path,
                "output_root": output_root,
                "game_label": game_label,
                "scene_label": scene_label,
                "sample_rate": sample_rate,
                "duration": duration,
                "keep_unpacked": keep_unpacked,
            },
            name="TrSS-Recorder-Capture",
            daemon=True,
        )
        worker.start()

    def _record_worker(
        self,
        *,
        choice: WindowChoice,
        session_id: str,
        session_name: str,
        session_dir: Path,
        zip_path: Path,
        output_root: Path,
        game_label: str,
        scene_label: str | None,
        sample_rate: int,
        duration: int,
        keep_unpacked: bool,
    ) -> None:
        del output_root  # The output path is represented by the session and ZIP locations.
        frames_dir = session_dir / FRAME_DIRECTORY
        rows: list[dict[str, Any]] = []
        started_utc = utc_now_text()
        started_at: float | None = None
        frame_interval = 1.0 / sample_rate
        next_deadline = time.monotonic() + 1.5  # Give focus/minimize time to settle.
        dropped_slots = 0
        captured_bytes = 0
        captured_width: int | None = None
        captured_height: int | None = None
        status = "completed"
        error_summary: str | None = None
        termination_reason = "duration_limit"
        last_elapsed = 0.0

        try:
            while True:
                if not _wait_until(next_deadline, self.stop_event):
                    termination_reason = (
                        "stop_event" if self.stop_event.is_set() else "early_stop_hotkey"
                    )
                    break
                if started_at is not None and time.monotonic() - started_at >= duration:
                    termination_reason = "duration_limit"
                    break

                frame = grab_window(choice.hwnd, require_foreground=True)
                now = time.monotonic()
                if started_at is None:
                    started_at = now
                elapsed = now - started_at

                if captured_width is None:
                    captured_width, captured_height = frame.size
                elif frame.size != (captured_width, captured_height):
                    raise CaptureError(
                        "The selected window changed resolution during recording. "
                        "This session was stopped to keep every frame consistently sized."
                    )

                frame_name = f"frame-{len(rows):06d}.png"
                frame_path = frames_dir / frame_name
                frame.save(frame_path, format="PNG", optimize=False)
                captured_bytes += frame_path.stat().st_size
                rows.append(
                    make_sequence_row(
                        frame_id=len(rows),
                        timestamp=elapsed,
                        scene_id=f"trss-{session_id[:8]}",
                        color_path=f"{FRAME_DIRECTORY}/{frame_name}",
                    )
                )
                last_elapsed = elapsed
                if len(rows) == 1 or len(rows) % 5 == 0:
                    self.events.put(
                        (
                            "progress",
                            {
                                "frames": len(rows),
                                "expected": sample_rate * duration,
                                "elapsed": elapsed,
                                "duration": duration,
                            },
                        )
                    )

                # A stored ZIP needs another copy of the captured PNGs. Stop
                # before the drive fills, leaving the originals and room for packaging.
                try:
                    free_bytes = shutil.disk_usage(session_dir).free
                    if not _has_packaging_headroom(free_bytes, captured_bytes):
                        status = "interrupted"
                        termination_reason = "low_disk_space"
                        error_summary = (
                            "Capture stopped early to preserve enough free disk space to "
                            "package the saved frames safely. Choose a larger or emptier "
                            "drive for longer sessions."
                        )
                        break
                except OSError:
                    # Keep the capture going if Windows temporarily cannot report free space;
                    # actual frame-write errors are still handled by the capture error path.
                    pass

                next_deadline += frame_interval
                now_after_write = time.monotonic()
                if now_after_write > next_deadline:
                    # Skip missed schedule slots; do not duplicate the previous image or
                    # falsify timestamps to make the capture appear to have reached its target sample rate.
                    missed = int((now_after_write - next_deadline) // frame_interval) + 1
                    dropped_slots += missed
                    next_deadline += missed * frame_interval

        except CaptureError as exc:
            status = "interrupted"
            termination_reason = "capture_error"
            error_summary = str(exc)
        except (OSError, ValueError, RuntimeError) as exc:
            status = "interrupted"
            termination_reason = "capture_error"
            error_summary = f"Capture failed ({type(exc).__name__}). Check the output drive and window."
        except Exception as exc:  # Preserve a usable partial session if an unexpected failure occurs.
            status = "interrupted"
            termination_reason = "unexpected_error"
            error_summary = f"Unexpected capture error ({type(exc).__name__})."

        if not rows:
            shutil.rmtree(session_dir, ignore_errors=True)
            self.events.put(("error", error_summary or "No frames were captured. Try Preview first."))
            return

        ended_utc = utc_now_text()
        elapsed_seconds = round(last_elapsed, 6)
        observed_sample_rate = (
            round((len(rows) - 1) / elapsed_seconds, 4)
            if len(rows) > 1 and elapsed_seconds > 0
            else None
        )
        metadata = {
            "schema_version": "1.1",
            "application": {"name": APP_NAME, "version": APP_VERSION},
            "session": {
                "id": session_id,
                "started_utc": started_utc,
                "ended_utc": ended_utc,
                "status": status,
                "termination_reason": termination_reason,
                "game_or_application": game_label,
                "scene_label": scene_label,
                "capture_method": "Windows visible application client-area screenshot",
                "capture_target_title_saved": False,
                "width": captured_width,
                "height": captured_height,
                "requested_sample_rate": sample_rate,
                "sample_rate_semantics": (
                    "PNG screenshot samples per second; not application FPS or displayed FPS."
                ),
                "duration_limit_seconds": duration,
                "elapsed_seconds": elapsed_seconds,
                "observed_sample_rate_from_saved_timestamps": observed_sample_rate,
                "captured_frames": len(rows),
                "missed_capture_slots": dropped_slots,
                "frame_encoding": "PNG RGB",
                "audio_recorded": False,
                "partial_error": error_summary,
            },
            "consent": {
                "policy_version": CONSENT_POLICY_VERSION,
                "confirmed_at_utc": ended_utc,
                "permission_to_capture_and_share_confirmed": True,
                "sharing_with_trss_project_confirmed": True,
                "participant_identity_collected": False,
            },
            "privacy": {
                "network_access": False,
                "automatic_upload": False,
                "microphone_or_audio_recorded": False,
                "clipboard_read": False,
                "running_process_list_collected": False,
                "hostname_collected": False,
                "account_name_collected": False,
                "ip_address_collected": False,
                "device_serial_numbers_collected": False,
                "raw_window_title_saved": False,
            },
        }
        try:
            system_metadata = collect_system_metadata()
            create_bundle(
                session_dir=session_dir,
                zip_path=zip_path,
                metadata=metadata,
                system_metadata=system_metadata,
                sequence_rows=rows,
                keep_unpacked=keep_unpacked,
            )
        except Exception as exc:
            # Keep a visible partial capture folder when packaging fails so the frames can
            # be recovered manually instead of silently discarding them.
            self.events.put(
                (
                    "error",
                    f"Captured {len(rows)} frame(s), but packaging failed "
                    f"({type(exc).__name__}). The folder was kept at: {session_dir}",
                )
            )
            return

        self.events.put(
            (
                "complete",
                {
                    "zip_path": str(zip_path),
                    "session_dir": str(session_dir) if session_dir.exists() else None,
                    "frames": len(rows),
                    "elapsed": elapsed_seconds,
                    "observed_sample_rate": observed_sample_rate,
                    "dropped_slots": dropped_slots,
                    "status": status,
                    "error": error_summary,
                },
            )
        )

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    frames = payload["frames"]
                    expected = max(1, payload["expected"])
                    self.progress_var.set(min(100.0, 100.0 * frames / expected))
                    self.status_var.set(
                        f"Capturing frame {frames:,} · "
                        f"{payload['elapsed']:.1f} / {payload['duration']} seconds · "
                        f"Early stop: {STOP_HOTKEY_TEXT}"
                    )
                elif kind == "complete":
                    self.is_recording = False
                    self.start_button.state(["!disabled"])
                    self.last_zip = Path(payload["zip_path"])
                    self.root.deiconify()
                    self.root.state("normal")
                    self.root.lift()
                    self.progress_var.set(100)
                    self.status_var.set(
                        f"Bundle verified · {payload['frames']} frames · "
                        f"{payload['zip_path']}"
                    )
                    summary = (
                        f"Capture bundle created and ZIP integrity checked.\n\n"
                        f"Frames: {payload['frames']}\n"
                        f"Observed screenshot rate: {payload['observed_sample_rate'] or 'not available'} samples/s\n"
                        f"Skipped capture slots: {payload['dropped_slots']}\n"
                        f"Status: {payload['status']}\n\n"
                        f"File: {payload['zip_path']}"
                    )
                    if payload["error"]:
                        summary += f"\n\nNote: {payload['error']}"
                    messagebox.showinfo(APP_NAME, summary)
                elif kind == "error":
                    self.is_recording = False
                    self.start_button.state(["!disabled"])
                    self.root.deiconify()
                    self.root.state("normal")
                    self.root.lift()
                    self.status_var.set(str(payload))
                    messagebox.showerror(APP_NAME, str(payload))
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(100, self._drain_events)

    def on_close(self) -> None:
        if self.is_recording:
            messagebox.showwarning(
                APP_NAME,
                f"Capture is active. Use {STOP_HOTKEY_TEXT} to stop early, then close the app "
                "after the ZIP has been verified.",
            )
            return
        self.root.destroy()


def main() -> None:
    """Launch the desktop recorder; all capture operations remain on the local machine."""
    global tk, ttk, filedialog, messagebox, ImageTk
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from PIL import ImageTk
    except ImportError as exc:
        raise SystemExit(
            "The desktop interface needs Python with Tkinter enabled. Reinstall Python with "
            "the Tcl/Tk feature, then run the recorder again."
        ) from exc
    _enable_dpi_awareness()
    root = tk.Tk()
    TrSSRecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
