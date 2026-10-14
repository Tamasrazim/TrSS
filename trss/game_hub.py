"""TrSS Game Library: discover and launch games with first-party performance monitoring.

Game FPS and frametime collection will be implemented in native TrSS code. Missing
measurements remain unknown instead of being guessed from monitor refresh or screenshot rate.
"""

from __future__ import annotations

import ctypes
from collections import Counter
from ctypes import wintypes
from dataclasses import asdict, dataclass, field as dataclass_field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import uuid
import zipfile
from typing import Any, Iterable

try:
    from PIL import Image, ImageGrab, ImageTk, ImageStat
except ImportError as exc:
    raise SystemExit(
        "TrSS Game Library needs Pillow. Install it with: python -m pip install -e ."
    ) from exc

from trss.game_matrix import (
    FIELD_GROUPS,
    STANDARD_SETTINGS,
    append_game_matrix_row,
    build_matrix_row,
    parse_additional_settings,
    parse_graphics_config_file,
    probe_game_settings,
)
from trss.recorder import (
    APP_NAME as RECORDER_NAME,
    CaptureError,
    _bundle_readme,
    _sha256_file,
    _window_client_bounds,
    collect_system_metadata,
    create_bundle,
    make_sequence_row,
    safe_label,
    utc_now_text,
)


APP_NAME = "TrSS Game Library"
APP_VERSION = "0.3.0"
STOP_CAPTURE_HOTKEY = "Ctrl + Alt + F8"
SHOW_LIBRARY_HOTKEY = "Ctrl + Alt + F9"
STOP_NOW_HOTKEY = "Ctrl + Alt + F10"
CAPTURE_RATES = ("1", "2", "5", "10")
BACKGROUND = "#0b0e12"
CARD_BACKGROUND = "#171b21"
CARD_HOVER = "#202630"
TEXT = "#edf0f4"
MUTED = "#9aa4b2"
ACCENT = "#dce2ea"
SKIP_DIR_NAMES = {
    "$recycle.bin", "system volume information", "windows", "appdata",
    "program files", "program files (x86)", "common files", "windowsapps",
    "windows defender", "internet explorer",
    "microsoft shared", "package cache", "installer", "softwaredistribution",
    "winsxs", "temp", "tmp", "node_modules", "__pycache__", ".git",
    "crashdumps", "shadercache", "webcache", "logs",
}
SKIP_EXE_WORDS = (
    "unins", "uninstall", "setup", "installer", "redist", "vcredist",
    "crashhandler", "crashreportclient", "unitycrash", "reportcrash",
    "easyanticheat", "eac_launcher", "battleye", "beservice", "cefprocess",
    "prereq", "dxsetup", "vc_redist", "anti-cheat", "anticheat",
)
SKIP_GAME_LABELS = (
    "microsoft visual c++", "directx setup", "easy anti-cheat", "battleye",
    "windows sdk", "nvidia physx", "openal installer", "uninstall",
    "update service", "crash reporter",
)
NON_GAME_APP_TERMS = (
    "microsoft visual c++", "directx", "windows sdk", "nvidia", "amd software",
    "intel driver", "intel graphics", "realtek", "logitech", "corsair", "razer",
    "steelseries", "msi center", "armoury crate", "asus", "gigabyte", "dragon center",
    "adobe", "google chrome", "microsoft edge", "mozilla firefox", "discord",
    "spotify", "obs studio", "vlc media", "7-zip", "winrar", "python", "node.js",
    "git version", "visual studio", "visual studio code", "microsoft 365", "office",
    "onedrive", "powershell", "winget", "teams", "zoom workplace", "anydesk",
    "openvpn", "tailscale", "battle.net", "epic games launcher", "steam",
    "universal c runtime", "webview2", "update service", "support assistant",
)
STEAM_PATHS = (
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Steam",
    Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Steam",
    Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
    / "Programs" / "Steam",
)
EPIC_MANIFEST_DIRS = (
    Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData"))
    / "Epic" / "EpicGamesLauncher" / "Data" / "Manifests",
)
KNOWN_GAME_ROOT_NAMES = (
    "Games", "Game Library", "SteamLibrary", "Epic Games", "EpicGames",
    "GOG Games", "GOG Galaxy", "EA Games", "Electronic Arts",
    "Ubisoft Game Launcher", "Ubisoft", "XboxGames", "Riot Games",
    "Battle.net", "Blizzard Entertainment", "Rockstar Games",
)
LOCAL_ROOT = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
APP_DATA = LOCAL_ROOT / "TrSS" / "GameLibrary"
ICON_CACHE = APP_DATA / "icons"
LIBRARY_FILE = APP_DATA / "library.json"


@dataclass
class GameEntry:
    """A discovered launchable game. Absolute paths are only stored locally."""

    name: str
    exe_path: str
    platform: str = "Disk"
    install_dir: str = ""
    app_id: str | None = None
    launch_uri: str | None = None
    icon_path: str | None = None
    source: str = "scan"

    @property
    def key(self) -> str:
        return os.path.normcase(os.path.abspath(self.exe_path))

    @property
    def exe_name(self) -> str:
        return Path(self.exe_path).name

    @property
    def short_path(self) -> str:
        try:
            path = Path(self.install_dir or self.exe_path).resolve()
            parts = path.parts
            return str(Path(*parts[-3:])) if len(parts) > 3 else str(path)
        except (OSError, ValueError):
            return Path(self.exe_path).parent.name


def _path_exists(path: str | Path | None) -> bool:
    try:
        return bool(path) and Path(path).is_file()
    except (OSError, TypeError, ValueError):
        return False


def _name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _parse_vdf_pairs(text: str) -> dict[str, str]:
    """Read simple key/value pairs from Steam's VDF text files."""
    return {
        key.casefold(): value.replace(r"\\", "\\")
        for key, value in re.findall(r'"([^"]+)"\s*"([^"]*)"', text)
    }


def parse_steam_libraries(vdf_path: str | Path) -> list[Path]:
    """Read the known library roots in a Steam libraryfolders.vdf file."""
    path = Path(vdf_path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    roots: list[Path] = []
    # Modern VDF nests library entries; this reads path fields without depending on
    # one particular indentation or key ordering.
    for value in re.findall(r'"path"\s*"([^"]+)"', text, flags=re.IGNORECASE):
        candidate = Path(value.replace(r"\\", "\\"))
        if candidate.is_dir() and candidate not in roots:
            roots.append(candidate)
    return roots


def parse_steam_manifest(path: str | Path) -> dict[str, str] | None:
    """Extract the small set of fields used from a Steam appmanifest."""
    manifest_path = Path(path)
    try:
        text = manifest_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    pairs = _parse_vdf_pairs(text)
    app_id = pairs.get("appid", "")
    name = pairs.get("name", "").strip()
    install_dir = pairs.get("installdir", "").strip()
    if not app_id or not name or not install_dir:
        return None
    if any(term in name.casefold() for term in SKIP_GAME_LABELS):
        return None
    return {"app_id": app_id, "name": name, "install_dir": install_dir}


def _ignored_executable(path: Path) -> bool:
    name = path.name.casefold()
    if not name.endswith(".exe"):
        return True
    return any(word in name for word in SKIP_EXE_WORDS)


def _walk_executables(
    root: Path,
    *,
    max_depth: int = 5,
    min_size: int = 256 * 1024,
    stop_event: threading.Event | None = None,
    visited_budget: int = 140_000,
) -> list[Path]:
    """Find executables below a game folder, pruning caches and support-tool folders."""
    if not root.is_dir():
        return []
    root_depth = len(root.parts)
    results: list[Path] = []
    visited = 0
    for current, dirnames, filenames in os.walk(root, topdown=True, onerror=lambda _err: None):
        if stop_event is not None and stop_event.is_set():
            break
        visited += len(filenames) + len(dirnames)
        if visited > visited_budget:
            break
        current_path = Path(current)
        depth = len(current_path.parts) - root_depth
        dirnames[:] = [
            folder for folder in dirnames
            if folder.casefold() not in SKIP_DIR_NAMES
            and not folder.casefold().startswith((".", "$"))
            and depth < max_depth
        ]
        for filename in filenames:
            candidate = current_path / filename
            if _ignored_executable(candidate):
                continue
            try:
                if candidate.stat().st_size < min_size:
                    continue
            except OSError:
                continue
            results.append(candidate)
    return results


def choose_game_executable(
    root: str | Path,
    game_name: str,
    *,
    preferred: str | Path | None = None,
    stop_event: threading.Event | None = None,
) -> Path | None:
    """Choose the likely game binary without mistaking common redistributables for a game."""
    root_path = Path(root)
    if preferred:
        preferred_path = Path(preferred)
        if not preferred_path.is_absolute():
            preferred_path = root_path / preferred_path
        if _path_exists(preferred_path) and not _ignored_executable(preferred_path):
            return preferred_path

    executables = _walk_executables(root_path, max_depth=6, min_size=256 * 1024, stop_event=stop_event)
    if not executables:
        return None

    name_key = _name_key(game_name)

    def score(path: Path) -> tuple[int, int, str]:
        stem = _name_key(path.stem)
        full = str(path).casefold()
        try:
            size = path.stat().st_size
        except OSError:
            size = 0

        points = 0
        if stem == name_key:
            points += 140
        elif name_key and (name_key in stem or stem in name_key):
            points += 75
        if "shipping" in stem:
            points += 25
        if "win64" in stem or "x64" in stem:
            points += 5
        if "binaries" in full and ("win64" in full or "win32" in full):
            points += 35
        if path.parent == root_path:
            points += 18
        if size >= 100 * 1024 * 1024:
            points += 45
        elif size >= 20 * 1024 * 1024:
            points += 30
        elif size >= 5 * 1024 * 1024:
            points += 15
        if "launcher" in stem and stem != name_key:
            points -= 35
        if "benchmark" in stem:
            points -= 20
        # Prefer the likely game executable over small support tools.
        return points, size, str(path).casefold()

    return max(executables, key=score)


def _is_reasonable_game_title(name: str) -> bool:
    cleaned = name.strip()
    return (
        len(cleaned) >= 2
        and not any(term in cleaned.casefold() for term in SKIP_GAME_LABELS)
        and not cleaned.casefold().endswith(("uninstaller", "setup", "redistributable"))
    )


def _new_entry(
    name: str,
    exe: Path,
    *,
    platform_name: str,
    app_id: str | None = None,
    launch_uri: str | None = None,
    source: str = "scan",
) -> GameEntry:
    return GameEntry(
        name=safe_label(name, exe.stem, 100),
        exe_path=str(exe.resolve()),
        platform=platform_name,
        install_dir=str(exe.parent.resolve()),
        app_id=app_id,
        launch_uri=launch_uri,
        source=source,
    )


def _steam_games(stop_event: threading.Event | None = None) -> list[GameEntry]:
    games: list[GameEntry] = []
    libraries: list[Path] = []
    for steam_root in STEAM_PATHS:
        if not steam_root.is_dir():
            continue
        libraries.append(steam_root)
        vdf = steam_root / "steamapps" / "libraryfolders.vdf"
        for library in parse_steam_libraries(vdf):
            if library not in libraries:
                libraries.append(library)

    for library in libraries:
        steamapps = library / "steamapps"
        if not steamapps.is_dir():
            continue
        for manifest_path in steamapps.glob("appmanifest_*.acf"):
            if stop_event is not None and stop_event.is_set():
                return games
            fields = parse_steam_manifest(manifest_path)
            if not fields:
                continue
            install = steamapps / "common" / fields["install_dir"]
            executable = choose_game_executable(
                install, fields["name"], stop_event=stop_event
            )
            if executable:
                games.append(
                    _new_entry(
                        fields["name"],
                        executable,
                        platform_name="Steam",
                        app_id=fields["app_id"],
                        launch_uri=f"steam://rungameid/{fields['app_id']}",
                        source="steam-manifest",
                    )
                )
    return games


def _epic_games(stop_event: threading.Event | None = None) -> list[GameEntry]:
    games: list[GameEntry] = []
    for manifest_dir in EPIC_MANIFEST_DIRS:
        if not manifest_dir.is_dir():
            continue
        for manifest_path in manifest_dir.glob("*.item"):
            if stop_event is not None and stop_event.is_set():
                return games
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            name = str(payload.get("DisplayName") or payload.get("AppName") or "").strip()
            location = Path(str(payload.get("InstallLocation") or ""))
            executable = str(payload.get("LaunchExecutable") or "").strip()
            app_id = str(payload.get("AppName") or payload.get("CatalogItemId") or "")
            if not name or not location.is_dir() or not _is_reasonable_game_title(name):
                continue
            chosen = choose_game_executable(
                location, name, preferred=executable or None, stop_event=stop_event
            )
            if chosen:
                games.append(
                    _new_entry(
                        name,
                        chosen,
                        platform_name="Epic Games",
                        app_id=app_id or None,
                        launch_uri=(
                            f"com.epicgames.launcher://apps/{app_id}?action=launch&silent=true"
                            if app_id else None
                        ),
                        source="epic-manifest",
                    )
                )
    return games


def _registry_install_locations() -> list[tuple[str, Path]]:
    """Read game names and install locations from the normal Windows uninstall keys."""
    if os.name != "nt":
        return []
    import winreg

    locations: list[tuple[str, Path]] = []
    roots = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]
    for hive, subkey in roots:
        try:
            parent = winreg.OpenKey(hive, subkey)
        except OSError:
            continue
        with parent:
            for index in range(winreg.QueryInfoKey(parent)[0]):
                try:
                    child_name = winreg.EnumKey(parent, index)
                    with winreg.OpenKey(parent, child_name) as child:
                        display_name = str(winreg.QueryValueEx(child, "DisplayName")[0])
                        install_dir = str(winreg.QueryValueEx(child, "InstallLocation")[0])
                except OSError:
                    continue
                location = Path(install_dir)
                if (
                    _is_reasonable_game_title(display_name)
                    and not any(term in display_name.casefold() for term in NON_GAME_APP_TERMS)
                    and location.is_dir()
                    and len(display_name.strip()) >= 3
                ):
                    locations.append((display_name, location))
    return locations


def _known_game_roots() -> list[Path]:
    candidates: list[Path] = []
    for base in (
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")),
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")),
        Path(os.environ.get("USERPROFILE", str(Path.home()))),
        Path(os.environ.get("PUBLIC", r"C:\Users\Public")),
        Path("C:/"),
        Path("D:/"),
        Path("E:/"),
    ):
        for name in KNOWN_GAME_ROOT_NAMES:
            root = base / name
            if root.is_dir() and root not in candidates:
                candidates.append(root)
    for variable in ("LOCALAPPDATA", "PROGRAMDATA"):
        base = Path(os.environ.get(variable, str(Path.home())))
        for rel in (
            "EA Games", "Ubisoft Game Launcher/games", "GOG Galaxy/Games",
            "Rockstar Games", "Battle.net", "Riot Games", "XboxGames",
        ):
            root = base / rel
            if root.is_dir() and root not in candidates:
                candidates.append(root)
    return candidates


def _scan_game_root(
    root: Path,
    *,
    platform_name: str = "Disk",
    stop_event: threading.Event | None = None,
) -> list[GameEntry]:
    """Scan one explicitly chosen game folder, making one card per discovered executable."""
    games: list[GameEntry] = []
    try:
        children = sorted(root.iterdir(), key=lambda p: p.name.casefold())
    except OSError:
        children = []
    folders = [child for child in children if child.is_dir()]
    # A directory picked by the participant may itself be the game root.
    if not folders:
        folders = [root]
    elif any(p.suffix.casefold() == ".exe" for p in children):
        folders.insert(0, root)

    for folder in folders:
        if stop_event is not None and stop_event.is_set():
            break
        if folder.name.casefold() in SKIP_DIR_NAMES:
            continue
        if folder.name.casefold() in {"steamapps", "redist", "redistributables", "support"}:
            continue
        exe = choose_game_executable(folder, folder.name, stop_event=stop_event)
        if not exe:
            continue
        if not _is_reasonable_game_title(folder.name):
            continue
        games.append(
            _new_entry(
                folder.name,
                exe,
                platform_name=platform_name,
                source="folder-scan",
            )
        )
    return games


def _looks_like_game_install(executable: Path) -> bool:
    """Reject generic tools found on a full-drive scan unless their folder looks game-like."""
    parts = [part.casefold() for part in executable.parts]
    known_roots = {
        "games", "game library", "steamlibrary", "steamapps", "epic games",
        "epicgames", "xboxgames", "gog games", "ea games", "ubisoft game launcher",
        "riot games", "battle.net", "blizzard entertainment", "rockstar games",
    }
    if any(part in known_roots for part in parts):
        return True

    game_markers = {"binaries", "win64", "win32", "paks", "content", "engine"}
    if any(part in game_markers for part in parts):
        return True

    stem = executable.stem
    for ancestor in list(executable.parents)[:5]:
        for folder in ("Content", "Paks", "Engine", "Data", "assets", "Assets"):
            try:
                if (ancestor / folder).is_dir():
                    return True
            except OSError:
                pass
        try:
            if (ancestor / f"{stem}_Data").is_dir():
                return True
        except OSError:
            pass
    return False


def _scan_all_fixed_drives(
    stop_event: threading.Event,
    progress: Any,
) -> list[GameEntry]:
    """Deep-scan fixed drives only when explicitly requested; allow cancellation."""
    if os.name != "nt":
        return []
    kernel32 = ctypes.windll.kernel32
    drive_mask = int(kernel32.GetLogicalDrives())
    roots: list[Path] = []
    for index in range(26):
        if not (drive_mask & (1 << index)):
            continue
        root = Path(f"{chr(ord('A') + index)}:\\")
        try:
            if kernel32.GetDriveTypeW(str(root)) == 3:  # DRIVE_FIXED
                roots.append(root)
        except (OSError, TypeError):
            continue

    games: list[GameEntry] = []
    exe_seen: set[str] = set()
    for drive in roots:
        if stop_event.is_set():
            break
        progress(f"Scanning {drive} for game executables…")
        for current, dirnames, filenames in os.walk(
            drive, topdown=True, onerror=lambda _err: None
        ):
            if stop_event.is_set():
                break
            current_path = Path(current)
            dirnames[:] = [
                name for name in dirnames
                if name.casefold() not in SKIP_DIR_NAMES
                and not name.casefold().startswith((".", "$"))
            ]
            for filename in filenames:
                if stop_event.is_set():
                    break
                path = current_path / filename
                if _ignored_executable(path):
                    continue
                key = os.path.normcase(str(path))
                if key in exe_seen:
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size < 3 * 1024 * 1024 or not _looks_like_game_install(path):
                    continue
                exe_seen.add(key)
                parent_name = path.parent.name
                title = path.stem
                if title.casefold().endswith(("-win64-shipping", "-shipping")):
                    title = re.sub(r"(?i)(-win64)?-shipping$", "", title)
                if not _is_reasonable_game_title(title):
                    continue
                # Ignore common helper names missed by the filename filter.
                if _name_key(title) in {"launcher", "updater", "patcher", "bootstrapper"}:
                    continue
                games.append(
                    _new_entry(
                        title,
                        path,
                        platform_name="Disk",
                        source="full-drive-scan",
                    )
                )
            # Full-drive scans can encounter enormous SDK and source trees; prune them
            # further once their first level is known.
            if current_path != drive and current_path.name.casefold() in {
                "sdk", "source", "samples", "docs", "documentation", "test",
                "tests", "tools", "toolchains",
            }:
                dirnames[:] = []
    return games


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _RGBQUAD(ctypes.Structure):
    _fields_ = [
        ("rgbBlue", ctypes.c_ubyte), ("rgbGreen", ctypes.c_ubyte),
        ("rgbRed", ctypes.c_ubyte), ("rgbReserved", ctypes.c_ubyte),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER), ("bmiColors", _RGBQUAD * 1)]


class ExeIconExtractor:
    """Use Windows Shell/GDI APIs to read an executable's associated icon into PNG."""

    @classmethod
    def save_icon(cls, exe_path: str | Path, output: Path, size: int = 64) -> str | None:
        if os.name != "nt":
            return None
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            if output.is_file() and output.stat().st_size > 100:
                return str(output)
        except OSError:
            pass

        shell32 = ctypes.windll.shell32
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        large_icon = wintypes.HICON()
        small_icon = wintypes.HICON()
        shell32.ExtractIconExW.argtypes = [
            wintypes.LPCWSTR, ctypes.c_int, ctypes.POINTER(wintypes.HICON),
            ctypes.POINTER(wintypes.HICON), wintypes.UINT,
        ]
        shell32.ExtractIconExW.restype = wintypes.UINT
        count = shell32.ExtractIconExW(str(exe_path), 0, ctypes.byref(large_icon), ctypes.byref(small_icon), 1)
        if not count or not large_icon:
            return None

        user32.GetDC.argtypes = [wintypes.HWND]
        user32.GetDC.restype = wintypes.HDC
        user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        user32.ReleaseDC.restype = ctypes.c_int
        user32.DestroyIcon.argtypes = [wintypes.HICON]
        user32.DestroyIcon.restype = wintypes.BOOL
        user32.DrawIconEx.argtypes = [
            wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.HICON,
            ctypes.c_int, ctypes.c_int, wintypes.UINT, wintypes.HBRUSH, wintypes.UINT,
        ]
        user32.DrawIconEx.restype = wintypes.BOOL
        gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
        gdi32.CreateCompatibleDC.restype = wintypes.HDC
        gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        gdi32.SelectObject.restype = wintypes.HGDIOBJ
        gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        gdi32.DeleteObject.restype = wintypes.BOOL
        gdi32.DeleteDC.argtypes = [wintypes.HDC]
        gdi32.DeleteDC.restype = wintypes.BOOL
        screen_dc = user32.GetDC(None)
        memory_dc = gdi32.CreateCompatibleDC(screen_dc)
        bitmap_info = _BITMAPINFO()
        bitmap_info.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bitmap_info.bmiHeader.biWidth = size
        bitmap_info.bmiHeader.biHeight = -size
        bitmap_info.bmiHeader.biPlanes = 1
        bitmap_info.bmiHeader.biBitCount = 32
        bitmap_info.bmiHeader.biCompression = 0
        bits = ctypes.c_void_p()
        gdi32.CreateDIBSection.argtypes = [
            wintypes.HDC, ctypes.POINTER(_BITMAPINFO), wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD,
        ]
        gdi32.CreateDIBSection.restype = wintypes.HBITMAP
        bitmap = gdi32.CreateDIBSection(
            screen_dc, ctypes.byref(bitmap_info), 0, ctypes.byref(bits), None, 0
        )
        old_object = gdi32.SelectObject(memory_dc, bitmap)
        try:
            user32.DrawIconEx(memory_dc, 0, 0, large_icon, size, size, 0, None, 0x0003)
            raw = ctypes.string_at(bits, size * size * 4)
            image = Image.frombytes("RGBA", (size, size), raw, "raw", "BGRA")
            alpha = image.getchannel("A")
            if alpha.getextrema() == (0, 0):
                image.putalpha(255)
            image.save(output, format="PNG", optimize=True)
            return str(output)
        except (OSError, ValueError):
            return None
        finally:
            try:
                gdi32.SelectObject(memory_dc, old_object)
                gdi32.DeleteObject(bitmap)
                gdi32.DeleteDC(memory_dc)
                user32.ReleaseDC(None, screen_dc)
                user32.DestroyIcon(large_icon)
                if small_icon:
                    user32.DestroyIcon(small_icon)
            except (OSError, AttributeError):
                pass


def load_library() -> tuple[list[GameEntry], list[str]]:
    """Load a previously scanned library and custom folders from the local profile."""
    try:
        payload = json.loads(LIBRARY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], []
    entries: list[GameEntry] = []
    for row in payload.get("games", []):
        try:
            entry = GameEntry(**row)
            if _path_exists(entry.exe_path):
                entries.append(entry)
        except (TypeError, ValueError):
            continue
    roots = [str(root) for root in payload.get("custom_roots", []) if Path(root).is_dir()]
    return entries, roots


def save_library(games: Iterable[GameEntry], custom_roots: Iterable[str]) -> None:
    """Persist the selected executable index and scan roots locally, never to a server."""
    APP_DATA.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "updated_utc": utc_now_text(),
        "custom_roots": sorted(set(custom_roots)),
        "games": [asdict(game) for game in games],
    }
    temporary = LIBRARY_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(LIBRARY_FILE)


def _foreground_hwnd() -> int:
    if os.name != "nt":
        return 0
    user32 = ctypes.windll.user32
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    result = user32.GetForegroundWindow()
    return int(result) if result else 0


def _window_pid(hwnd: int) -> int:
    if os.name != "nt" or not hwnd:
        return 0
    pid = wintypes.DWORD()
    user32 = ctypes.windll.user32
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
    return int(pid.value)


def _process_image_path(pid: int) -> str | None:
    if os.name != "nt" or not pid:
        return None
    kernel32 = ctypes.windll.kernel32
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        capacity = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(capacity.value)
        if kernel32.QueryFullProcessImageNameW(
            handle, 0, buffer, ctypes.byref(capacity)
        ):
            return buffer.value
        return None
    finally:
        kernel32.CloseHandle(handle)


def _find_game_window(game: GameEntry) -> tuple[int, int] | None:
    """Find a visible window owned by this game executable, without saving its title."""
    if os.name != "nt":
        return None
    user32 = ctypes.windll.user32
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    expected = os.path.normcase(os.path.abspath(game.exe_path))
    foreground = _foreground_hwnd()
    if foreground:
        pid = _window_pid(foreground)
        path = _process_image_path(pid)
        if path and os.path.normcase(os.path.abspath(path)) == expected:
            return foreground, pid

    matches: list[tuple[int, int]] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        pid = _window_pid(int(hwnd))
        path = _process_image_path(pid)
        if path and os.path.normcase(os.path.abspath(path)) == expected:
            matches.append((int(hwnd), pid))
        return True

    callback = callback_type(visit)
    user32.EnumWindows(callback, 0)
    return matches[0] if matches else None


class _PDHValueUnion(ctypes.Union):
    _fields_ = [
        ("longValue", ctypes.c_long), ("doubleValue", ctypes.c_double),
        ("largeValue", ctypes.c_longlong), ("ansiStringValue", ctypes.c_char_p),
        ("wideStringValue", ctypes.c_wchar_p),
    ]


class _PDHCounterValue(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("status", wintypes.DWORD), ("value", _PDHValueUnion)]


class _PDHValueItem(ctypes.Structure):
    _fields_ = [("name", wintypes.LPWSTR), ("formatted", _PDHCounterValue)]


class PDHReader:
    """Read per-process GPU engine utilization and dedicated-memory counters when exposed."""

    def __init__(self) -> None:
        self.query = ctypes.c_void_p()
        self.gpu_counter = ctypes.c_void_p()
        self.memory_counter = ctypes.c_void_p()
        self.enabled = False
        if os.name != "nt":
            return
        try:
            self.pdh = ctypes.WinDLL("pdh")
            self.pdh.PdhOpenQueryW.argtypes = [
                wintypes.LPCWSTR, ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_void_p)
            ]
            self.pdh.PdhAddEnglishCounterW.argtypes = [
                ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_ulonglong,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self.pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
            self.pdh.PdhGetFormattedCounterArrayW.argtypes = [
                ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
            ]
            if self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)) != 0:
                return
            self.pdh.PdhAddEnglishCounterW(
                self.query, r"\GPU Engine(*)\Utilization Percentage", 0,
                ctypes.byref(self.gpu_counter),
            )
            self.pdh.PdhAddEnglishCounterW(
                self.query, r"\GPU Process Memory(*)\Dedicated Usage", 0,
                ctypes.byref(self.memory_counter),
            )
            self.pdh.PdhCollectQueryData(self.query)
            self.enabled = bool(self.gpu_counter or self.memory_counter)
        except (AttributeError, OSError):
            self.enabled = False

    def _read_counter(self, counter: ctypes.c_void_p) -> list[tuple[str, float]] | None:
        if not self.enabled or not counter:
            return None
        buffer_bytes = wintypes.DWORD(0)
        item_count = wintypes.DWORD(0)
        result = self.pdh.PdhGetFormattedCounterArrayW(
            counter, 0x00000200, ctypes.byref(buffer_bytes), ctypes.byref(item_count), None
        )
        if not buffer_bytes.value or item_count.value == 0:
            return None
        buffer = ctypes.create_string_buffer(buffer_bytes.value)
        result = self.pdh.PdhGetFormattedCounterArrayW(
            counter, 0x00000200, ctypes.byref(buffer_bytes), ctypes.byref(item_count),
            ctypes.cast(buffer, ctypes.c_void_p),
        )
        if result != 0:
            return None
        items = ctypes.cast(
            buffer, ctypes.POINTER(_PDHValueItem * item_count.value)
        ).contents
        values: list[tuple[str, float]] = []
        for item in items:
            if item.formatted.status not in (0, 1):
                continue
            try:
                value = float(item.formatted.doubleValue)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                values.append((str(item.name or ""), value))
        return values

    def sample(self, pid: int) -> tuple[float | None, int | None]:
        if not self.enabled or not pid:
            return None, None
        try:
            self.pdh.PdhCollectQueryData(self.query)
            gpu_rows = self._read_counter(self.gpu_counter)
            mem_rows = self._read_counter(self.memory_counter)
        except (OSError, AttributeError):
            return None, None
        gpu = None
        if gpu_rows is not None:
            wanted = f"pid_{pid}_"
            relevant = [
                value for name, value in gpu_rows
                if wanted in name.casefold() and "engtype_3d" in name.casefold()
            ]
            if relevant:
                gpu = round(min(100.0, sum(relevant)), 1)
        memory = None
        if mem_rows is not None:
            wanted = f"pid_{pid}_"
            relevant_memory = [
                int(value) for name, value in mem_rows
                if wanted in name.casefold()
            ]
            if relevant_memory:
                memory = sum(relevant_memory)
        return gpu, memory

    def close(self) -> None:
        if self.query and os.name == "nt":
            try:
                self.pdh.PdhCloseQuery(self.query)
            except (AttributeError, OSError):
                pass
            self.query = ctypes.c_void_p()


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class PerformanceReader:
    """Read system/process telemetry without pretending it is an FPS measurement."""

    def __init__(self) -> None:
        self.prev_system: tuple[int, int, int] | None = None
        self.prev_process: dict[int, tuple[int, int, float]] = {}
        self.pdh = PDHReader()

    @staticmethod
    def _filetime_value(value: wintypes.FILETIME) -> int:
        return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)

    def _system_cpu(self) -> float | None:
        if os.name != "nt":
            return None
        kernel32 = ctypes.windll.kernel32
        kernel32.GetSystemTimes.argtypes = [
            ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ]
        kernel32.GetSystemTimes.restype = wintypes.BOOL
        idle = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        if not ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            return None
        current = (
            self._filetime_value(idle),
            self._filetime_value(kernel),
            self._filetime_value(user),
        )
        previous = self.prev_system
        self.prev_system = current
        if previous is None:
            return None
        delta_idle = current[0] - previous[0]
        delta_total = (current[1] - previous[1]) + (current[2] - previous[2])
        if delta_total <= 0:
            return None
        return round(max(0.0, min(100.0, 100.0 * (1.0 - delta_idle / delta_total))), 1)

    @staticmethod
    def _memory() -> dict[str, int | float | None]:
        if os.name != "nt":
            return {"ram_total_bytes": None, "ram_used_bytes": None, "ram_percent": None}
        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        kernel32 = ctypes.windll.kernel32
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MEMORYSTATUSEX)]
        kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {"ram_total_bytes": None, "ram_used_bytes": None, "ram_percent": None}
        total = int(status.ullTotalPhys)
        available = int(status.ullAvailPhys)
        return {
            "ram_total_bytes": total,
            "ram_used_bytes": max(0, total - available),
            "ram_percent": float(status.dwMemoryLoad),
        }

    def _process_stats(self, pid: int) -> dict[str, int | float | None]:
        empty: dict[str, int | float | None] = {
            "game_cpu_percent": None, "game_ram_bytes": None,
        }
        if os.name != "nt" or not pid:
            return empty
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        PROCESS_VM_READ = 0x0010
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
            ctypes.POINTER(wintypes.FILETIME),
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, False, pid
        )
        if not handle:
            return empty
        try:
            cpu_time: float | None = None
            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel_time = wintypes.FILETIME()
            user_time = wintypes.FILETIME()
            if kernel32.GetProcessTimes(
                handle, ctypes.byref(creation), ctypes.byref(exit_time),
                ctypes.byref(kernel_time), ctypes.byref(user_time),
            ):
                now = time.monotonic()
                ticks = self._filetime_value(kernel_time) + self._filetime_value(user_time)
                previous = self.prev_process.get(pid)
                self.prev_process[pid] = (ticks, 0, now)
                if previous:
                    delta_ticks = ticks - previous[0]
                    delta_wall = now - previous[2]
                    if delta_wall > 0:
                        # Process CPU can exceed 100% when several logical cores are busy.
                        logical_cpus = max(1, os.cpu_count() or 1)
                        cpu_time = round(
                            max(
                                0.0,
                                min(
                                    100.0,
                                    100.0 * delta_ticks / 10_000_000 / delta_wall / logical_cpus,
                                ),
                            ),
                            1,
                        )

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            memory = PROCESS_MEMORY_COUNTERS()
            memory.cb = ctypes.sizeof(memory)
            ram_bytes: int | None = None
            try:
                psapi = ctypes.WinDLL("psapi")
                if psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(memory), memory.cb
                ):
                    ram_bytes = int(memory.WorkingSetSize)
            except (OSError, AttributeError):
                pass
            return {"game_cpu_percent": cpu_time, "game_ram_bytes": ram_bytes}
        finally:
            kernel32.CloseHandle(handle)

    def sample(self, game: GameEntry | None) -> dict[str, Any]:
        """Read available system/process counters; native TrSS frame timing is in development."""
        pid = 0
        hwnd = 0
        if game:
            found = _find_game_window(game)
            if found:
                hwnd, pid = found
        memory = self._memory()
        process = self._process_stats(pid)
        gpu, vram = self.pdh.sample(pid) if pid else (None, None)
        return {
            "timestamp_utc": utc_now_text(),
            "game": game.name if game else None,
            "game_window_found": bool(hwnd),
            "fps": None,
            "application_fps": None,
            "frametime_ms": None,
            "application_frametime_ms": None,
            "fps_1_percent_low": None,
            "fps_samples": 0,
            "fps_source": "Native TrSS per-present telemetry is in development",
            "system_cpu_percent": self._system_cpu(),
            "game_cpu_percent": process["game_cpu_percent"],
            "game_ram_bytes": process["game_ram_bytes"],
            "ram_total_bytes": memory["ram_total_bytes"],
            "ram_used_bytes": memory["ram_used_bytes"],
            "ram_percent": memory["ram_percent"],
            "game_gpu_3d_percent": gpu,
            "game_dedicated_gpu_bytes": vram,
        }

    def close(self) -> None:
        self.pdh.close()

@dataclass
class CaptureSession:
    """Mutable state for an unlimited-duration, participant-started capture."""

    session_id: str
    game: GameEntry
    folder: Path
    zip_path: Path
    sample_rate: int
    stop_event: threading.Event
    started_utc: str
    started_monotonic: float
    frame_rows: list[dict[str, Any]]
    performance_rows: list[dict[str, Any]]
    lock: threading.Lock
    captured_bytes: int = 0
    closing: bool = False
    status: str = "recording"
    error: str | None = None
    stop_reason: str = "user_stop"
    # The UI asks for the in-game graphics profile after collection stops. The
    # worker waits here so the finished ZIP contains the reviewed settings.
    review_event: threading.Event = dataclass_field(default_factory=threading.Event)
    detected_settings: dict[str, Any] = dataclass_field(default_factory=dict)
    graphics_settings: dict[str, Any] = dataclass_field(default_factory=dict)
    matrix_csv_path: str | None = None
    matrix_index_error: str | None = None


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _friendly_metric(value: Any, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def _format_bytes(value: Any) -> str:
    if value is None:
        return "—"
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1024.0 or unit == "TB":
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return "—"


def _hex_rgb(image_path: str | None) -> str:
    try:
        with Image.open(image_path) as image:
            small = image.convert("RGBA").resize((24, 24), Image.Resampling.LANCZOS)
            visible: list[tuple[int, int, int]] = []
            saturated: list[tuple[int, int, int]] = []
            for red, green, blue, alpha in small.getdata():
                if alpha < 90:
                    continue
                rgb = (red, green, blue)
                if max(rgb) >= 52:
                    visible.append(rgb)
                    if max(rgb) - min(rgb) >= 25:
                        saturated.append(rgb)
            # Ignore transparent/black border pixels. Saturated icon colors give a more
            # faithful ambient glow than the black or grey areas common in EXE icons.
            palette = saturated or visible
            if not palette:
                return "#aeb8c8"
            color = Counter(palette).most_common(1)[0][0]
            r, g, b = color
            return f"#{r:02x}{g:02x}{b:02x}"
    except (OSError, ValueError, TypeError):
        return "#aeb8c8"


def _blend_hex(left: str, right: str, amount: float) -> str:
    amount = max(0.0, min(1.0, amount))
    try:
        a = tuple(int(left[i:i + 2], 16) for i in (1, 3, 5))
        b = tuple(int(right[i:i + 2], 16) for i in (1, 3, 5))
        mixed = tuple(round(x + (y - x) * amount) for x, y in zip(a, b))
        return "#" + "".join(f"{part:02x}" for part in mixed)
    except (ValueError, IndexError):
        return BACKGROUND


class GameCard:
    """One executable-backed box in the responsive game library."""

    def __init__(self, app: "GameHubApp", parent: tk.Widget, game: GameEntry) -> None:
        self.app = app
        self.game = game
        self.accent = _hex_rgb(game.icon_path)
        self.frame = tk.Frame(
            parent,
            bg=CARD_BACKGROUND,
            highlightbackground="#282e38",
            highlightthickness=1,
            bd=0,
            width=236,
            height=224,
            cursor="hand2",
        )
        self.frame.grid_propagate(False)

        top = tk.Frame(self.frame, bg=CARD_BACKGROUND)
        top.pack(fill="x", padx=13, pady=(13, 6))
        icon_image: ImageTk.PhotoImage | None = None
        try:
            if game.icon_path and Path(game.icon_path).is_file():
                with Image.open(game.icon_path) as icon:
                    icon_image = ImageTk.PhotoImage(icon.convert("RGBA").resize((64, 64), Image.Resampling.LANCZOS))
        except (OSError, ValueError):
            icon_image = None
        self.icon_image = icon_image
        self.icon_label = tk.Label(
            top,
            image=self.icon_image,
            text="GAME" if self.icon_image is None else "",
            bg=CARD_BACKGROUND,
            fg=MUTED,
            width=64,
            height=64,
            compound="center",
            font=("Segoe UI Semibold", 9),
            cursor="hand2",
        )
        self.icon_label.pack(anchor="w")

        self.name_label = tk.Label(
            self.frame,
            text=game.name,
            bg=CARD_BACKGROUND,
            fg=TEXT,
            anchor="w",
            justify="left",
            wraplength=205,
            font=("Segoe UI Semibold", 11),
            cursor="hand2",
        )
        self.name_label.pack(fill="x", padx=13, pady=(4, 2))
        self.platform_label = tk.Label(
            self.frame,
            text=f"{game.platform.upper()}  /  {game.exe_name[:24]}",
            bg=CARD_BACKGROUND,
            fg=MUTED,
            anchor="w",
            font=("Consolas", 8),
            cursor="hand2",
        )
        self.platform_label.pack(fill="x", padx=13, pady=(0, 4))
        self.path_label = tk.Label(
            self.frame,
            text=game.short_path,
            bg=CARD_BACKGROUND,
            fg="#788392",
            anchor="w",
            font=("Consolas", 8),
            cursor="hand2",
        )
        self.path_label.pack(fill="x", padx=13, pady=(0, 10))
        self.run_label = tk.Label(
            self.frame,
            text="▶   LAUNCH GAME",
            bg="#242b35",
            fg="#e8edf4",
            anchor="center",
            font=("Segoe UI Semibold", 9),
            cursor="hand2",
            padx=8,
            pady=7,
        )
        self.run_label.pack(fill="x", padx=12, side="bottom", pady=(0, 12))

        self._bind_clicks(self.frame)
        for child in self.frame.winfo_children():
            self._bind_clicks(child)
        for child in top.winfo_children():
            self._bind_clicks(child)
        self.frame.bind("<Enter>", self.on_enter)
        self.frame.bind("<Leave>", self.on_leave)
        for widget in (self.icon_label, self.name_label, self.platform_label, self.path_label, self.run_label, top):
            widget.bind("<Enter>", self.on_enter)
            widget.bind("<Leave>", self.on_leave)

    def _bind_clicks(self, widget: tk.Widget) -> None:
        widget.bind("<Button-1>", self.on_click)

    def on_click(self, _event: tk.Event | None = None) -> str:
        self.app.launch_game(self.game)
        return "break"

    def on_enter(self, _event: tk.Event | None = None) -> None:
        self.frame.configure(bg=CARD_HOVER)
        self.frame.configure(highlightbackground=self.accent, highlightthickness=2)
        self.app.set_ambient_target(self.accent)
        self.app.set_status(f"{self.game.name} · Click the card to launch")

    def on_leave(self, event: tk.Event | None = None) -> None:
        if event and event.widget is not self.frame:
            # Child-to-child motion should keep the card hovered. Only clear when the
            # pointer is truly outside the card's bounds.
            pointer_x = self.frame.winfo_pointerx()
            pointer_y = self.frame.winfo_pointery()
            left = self.frame.winfo_rootx()
            top = self.frame.winfo_rooty()
            if left <= pointer_x <= left + self.frame.winfo_width() and top <= pointer_y <= top + self.frame.winfo_height():
                return
        self.frame.configure(bg=CARD_BACKGROUND)
        self.frame.configure(highlightbackground="#282e38", highlightthickness=1)
        self.app.set_ambient_target("#7f8a9b")


class PerformanceOverlay:
    """Small always-on-top, click-through performance display for a running game."""

    def __init__(self, app: "GameHubApp") -> None:
        self.app = app
        self.window = tk.Toplevel(app.root)
        self.window.title("TrSS Performance")
        self.window.overrideredirect(True)
        self.window.configure(bg="#11151b")
        try:
            self.window.wm_attributes("-topmost", True)
            self.window.wm_attributes("-alpha", 0.92)
        except tk.TclError:
            pass
        self.window.geometry("342x294+0+38")
        self.window.withdraw()

        self.heading = tk.Label(
            self.window,
            text="TRSS  /  LIVE TELEMETRY",
            bg="#151a22",
            fg="#dce3ed",
            font=("Consolas", 10, "bold"),
            anchor="w",
            padx=12,
            pady=10,
        )
        self.heading.pack(fill="x")
        self.game_label = tk.Label(
            self.window, text="No game selected", bg="#11151b", fg=TEXT,
            font=("Segoe UI Semibold", 10), anchor="w", padx=12, pady=7,
        )
        self.game_label.pack(fill="x")
        self.metrics_label = tk.Label(
            self.window,
            text="FPS     —  | FT —\n1% LOW  —\nAPP FPS —\nCPU     —\nGPU 3D  —\nVRAM    —\nRAM     —\nGAME    waiting for a game window",
            bg="#11151b",
            fg="#c6ceda",
            font=("Consolas", 10),
            justify="left",
            anchor="w",
            padx=12,
            pady=3,
        )
        self.metrics_label.pack(fill="x")
        self.footer = tk.Label(
            self.window,
            text="F8 CAPTURE  ·  F9 LIBRARY  ·  F10 STOP",
            bg="#151a22",
            fg="#97a4b5",
            font=("Consolas", 8),
            anchor="w",
            padx=12,
            pady=8,
        )
        self.footer.pack(fill="x")
        self.visible = False
        self.last_metrics: dict[str, Any] = {}
        self.window.update_idletasks()
        self._set_click_through()
        self.move_to_corner()

    def _set_click_through(self) -> None:
        if os.name != "nt":
            return
        try:
            hwnd = int(self.window.winfo_id())
            user32 = ctypes.windll.user32
            get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
            set_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
            get_long.argtypes = [wintypes.HWND, ctypes.c_int]
            get_long.restype = ctypes.c_ssize_t
            set_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
            set_long.restype = ctypes.c_ssize_t
            GWL_EXSTYLE = -20
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_LAYERED = 0x00080000
            WS_EX_NOACTIVATE = 0x08000000
            current = get_long(hwnd, GWL_EXSTYLE)
            set_long(
                hwnd, GWL_EXSTYLE,
                current | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_NOACTIVATE,
            )
            # Ask Windows 10 2004+ to exclude the OSD from screen captures where supported.
            user32.SetWindowDisplayAffinity(wintypes.HWND(hwnd), 0x00000011)
        except (AttributeError, OSError, tk.TclError):
            pass

    def move_to_corner(self) -> None:
        try:
            self.window.update_idletasks()
            width = self.window.winfo_width()
            x = max(0, self.app.root.winfo_screenwidth() - width - 22)
            self.window.geometry(f"+{x}+48")
        except tk.TclError:
            pass

    def show(self) -> None:
        if self.visible:
            return
        self.window.deiconify()
        self.window.lift()
        try:
            self.window.attributes("-topmost", True)
        except tk.TclError:
            pass
        self.visible = True
        self.move_to_corner()

    def hide(self) -> None:
        if not self.visible:
            return
        self.window.withdraw()
        self.visible = False

    def update_metrics(self, metrics: dict[str, Any] | None, game: GameEntry | None, capture_active: bool) -> None:
        self.last_metrics = metrics or {}
        if game:
            self.game_label.configure(text=game.name[:34])
        else:
            self.game_label.configure(text="Select a game from the library")

        m = self.last_metrics
        game_state = "game window detected" if m.get("game_window_found") else "waiting for game window"
        fps_value = m.get("fps")
        fps_text = f"{float(fps_value):.1f}" if fps_value is not None else "—"
        low_value = m.get("fps_1_percent_low")
        low_text = f"{float(low_value):.1f}" if low_value is not None else "—"
        app_fps = m.get("application_fps")
        app_fps_text = f"{float(app_fps):.1f}" if app_fps is not None else "—"
        frametime = m.get("frametime_ms")
        frametime_text = f"{float(frametime):.2f} ms" if frametime is not None else "—"
        fps_source = str(m.get("fps_source") or "Waiting for frame telemetry")
        lines = [
            f"FPS     {fps_text}  | FT {frametime_text}",
            f"1% LOW  {low_text} FPS",
            f"APP FPS {app_fps_text}",
            f"CPU     {_friendly_metric(m.get('system_cpu_percent'), '%')} system / {_friendly_metric(m.get('game_cpu_percent'), '%')} game",
            f"GPU 3D  {_friendly_metric(m.get('game_gpu_3d_percent'), '%')}",
            f"VRAM    {_format_bytes(m.get('game_dedicated_gpu_bytes'))} game use",
            f"RAM     {_friendly_metric(m.get('ram_percent'), '%')} {_format_bytes(m.get('ram_used_bytes'))} / {_format_bytes(m.get('ram_total_bytes'))}",
            f"GAME    {game_state}",
            f"FRAME   {fps_source[:48]}",
            f"DATA    {'CAPTURING' if capture_active else 'monitoring only'}",
        ]
        self.metrics_label.configure(text="\n".join(lines))
        self.footer.configure(
            text=f"F8 {'STOP' if capture_active else 'CAPTURE'}  ·  F9 LIBRARY  ·  F10 STOP"
        )


class GameHubApp:
    """Main game-grid UI, discovery coordinator and opt-in local collector."""

    def __init__(self, root: tk.Tk) -> None:
        if os.name != "nt":
            raise RuntimeError("TrSS Game Library currently requires Windows 10/11.")
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1180x850")
        self.root.minsize(940, 680)
        self.root.configure(bg=BACKGROUND)
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.scan_stop = threading.Event()
        self.closing = False
        self.games, self.custom_roots = load_library()
        self.cards: list[GameCard] = []
        self.selected_game: GameEntry | None = None
        self.active_capture: CaptureSession | None = None
        self.capture_lock = threading.RLock()
        self.capture_rate = 2
        self.monitor_stop = threading.Event()
        self.monitor_thread: threading.Thread | None = None
        self.monitor_reader = PerformanceReader()
        self.last_metrics: dict[str, Any] | None = None
        self.ambient_target = "#7f8a9b"
        self.ambient_current = "#0b0e12"
        self.ambient_animating = False
        self.status_var = tk.StringVar(value="Loading local game library…")
        self.search_var = tk.StringVar()
        self.capture_rate_var = tk.StringVar(value="2")
        self.share_consent_var = tk.BooleanVar(value=False)
        self.rights_consent_var = tk.BooleanVar(value=False)
        self.search_filter = "All"
        self.scan_thread: threading.Thread | None = None
        self.hotkey_state: dict[int, bool] = {}
        self.overlay = PerformanceOverlay(self)

        self._configure_styles()
        self._build_ui()
        self.render_games()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Configure>", self._on_resize)
        self.root.after(100, self._drain_events)
        self.root.after(160, self._poll_hotkeys)
        self._start_monitor()
        self.scan_installed_games(silent=bool(self.games))
        if self.games:
            self.set_status(f"{len(self.games)} game(s) loaded from the local library. Checking installed stores…")
        else:
            self.set_status("Ready to scan Steam, Epic Games, common game folders, or your drives.")

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", padding=(10, 7), background="#252b34", foreground=TEXT)
        style.map("TButton", background=[("active", "#333c49"), ("disabled", "#191d23")])
        style.configure(
            "Accent.TButton",
            padding=(13, 8),
            background="#dce2ea",
            foreground="#101318",
            font=("Segoe UI Semibold", 9),
        )
        style.map("Accent.TButton", background=[("active", "#ffffff")])
        style.configure("TCheckbutton", background=BACKGROUND, foreground="#cbd3dd")
        style.map("TCheckbutton", background=[("active", BACKGROUND)])
        style.configure(
            "TCombobox", fieldbackground="#151a21", background="#252b34",
            foreground=TEXT, arrowcolor=TEXT, padding=5,
        )
        style.map("TCombobox", fieldbackground=[("readonly", "#151a21")])

    def _build_ui(self) -> None:
        self.shell = tk.Frame(self.root, bg=BACKGROUND)
        self.shell.pack(fill="both", expand=True, padx=24, pady=20)

        header = tk.Frame(self.shell, bg=BACKGROUND)
        header.pack(fill="x", pady=(0, 14))
        title_block = tk.Frame(header, bg=BACKGROUND)
        title_block.pack(side="left", fill="x", expand=True)
        tk.Label(
            title_block, text="TRSS  /  GAME LIBRARY", bg=BACKGROUND, fg=TEXT,
            font=("Segoe UI Semibold", 23), anchor="w",
        ).pack(anchor="w")
        tk.Label(
            title_block,
            text="Find games. Launch directly. Collect opt-in graphics telemetry while you play.",
            bg=BACKGROUND, fg=MUTED, font=("Segoe UI", 10), anchor="w",
        ).pack(anchor="w", pady=(4, 0))

        button_block = tk.Frame(header, bg=BACKGROUND)
        button_block.pack(side="right", anchor="n")
        ttk.Button(button_block, text="Rescan", command=self.scan_installed_games).pack(side="left", padx=(0, 7))
        ttk.Button(button_block, text="Add folder", command=self.add_folder).pack(side="left", padx=(0, 7))
        ttk.Button(button_block, text="Scan all drives", command=self.scan_all_drives).pack(side="left", padx=(0, 7))
        ttk.Button(button_block, text="Matrix folder", command=self.open_matrix_folder).pack(side="left")

        toolbar = tk.Frame(self.shell, bg="#12171e", highlightthickness=1, highlightbackground="#252c36")
        toolbar.pack(fill="x", pady=(0, 12))
        search = ttk.Entry(toolbar, textvariable=self.search_var, width=35)
        search.pack(side="left", padx=(12, 10), pady=10, ipady=3)
        self.search_var.trace_add("write", lambda *_args: self.render_games())
        tk.Label(toolbar, text="FILTER", bg="#12171e", fg="#8491a1", font=("Consolas", 9)).pack(side="left", padx=(3, 8))
        for label in ("All", "Steam", "Epic Games", "Disk"):
            ttk.Button(
                toolbar, text=label,
                command=lambda selected=label: self.set_filter(selected),
            ).pack(side="left", padx=3, pady=7)
        self.count_label = tk.Label(
            toolbar, text="0 GAMES", bg="#12171e", fg="#c5ceda",
            font=("Consolas", 9, "bold"),
        )
        self.count_label.pack(side="right", padx=13)

        self.grid_wrap = tk.Frame(self.shell, bg=BACKGROUND)
        self.grid_wrap.pack(fill="both", expand=True)
        self.game_canvas = tk.Canvas(
            self.grid_wrap, bg=BACKGROUND, highlightthickness=0, bd=0,
        )
        self.scrollbar = ttk.Scrollbar(
            self.grid_wrap, orient="vertical", command=self.game_canvas.yview
        )
        self.game_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.game_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.card_host = tk.Frame(self.game_canvas, bg=BACKGROUND)
        self.host_window = self.game_canvas.create_window((0, 0), window=self.card_host, anchor="nw")
        self.card_host.bind("<Configure>", self._update_scroll_region)
        self.game_canvas.bind("<Configure>", self._resize_card_host)
        self.game_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        footer = tk.Frame(
            self.shell, bg="#10141a", highlightthickness=1, highlightbackground="#252c36"
        )
        footer.pack(fill="x", pady=(12, 0))
        footer_left = tk.Frame(footer, bg="#10141a")
        footer_left.pack(side="left", fill="x", expand=True, padx=12, pady=10)
        tk.Label(
            footer_left, text="LIVE PERFORMANCE OVERLAY", bg="#10141a", fg=TEXT,
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w")
        tk.Label(
            footer_left,
            text=(
                f"{STOP_CAPTURE_HOTKEY} toggle collection  ·  {SHOW_LIBRARY_HOTKEY} show library  ·  "
                f"{STOP_NOW_HOTKEY} stop collection"
            ),
            bg="#10141a", fg=MUTED, font=("Consolas", 8),
        ).pack(anchor="w", pady=(3, 0))
        footer_controls = tk.Frame(footer, bg="#10141a")
        footer_controls.pack(side="right", padx=12, pady=8)
        tk.Label(footer_controls, text="Capture FPS", bg="#10141a", fg=MUTED).pack(side="left", padx=(0, 6))
        ttk.Combobox(
            footer_controls, textvariable=self.capture_rate_var,
            state="readonly", values=CAPTURE_RATES, width=4,
        ).pack(side="left", padx=(0, 12))
        ttk.Button(footer_controls, text="Show performance", command=self.show_overlay).pack(side="left")

        consent = tk.Frame(self.shell, bg=BACKGROUND)
        consent.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(
            consent,
            text="I have permission to capture and share the selected game's visible content.",
            variable=self.rights_consent_var,
        ).pack(anchor="w")
        ttk.Checkbutton(
            consent,
            text="I agree to save performance logs and optional screenshots locally for TrSS research.",
            variable=self.share_consent_var,
        ).pack(anchor="w", pady=(3, 0))
        tk.Label(
            self.shell,
            textvariable=self.status_var,
            bg=BACKGROUND, fg="#98a4b3",
            font=("Consolas", 9), anchor="w",
        ).pack(fill="x", pady=(8, 0))

    def _on_mousewheel(self, event: tk.Event) -> str:
        try:
            self.game_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except (tk.TclError, AttributeError):
            pass
        return "break"

    def _on_resize(self, _event: tk.Event | None = None) -> None:
        # The canvas configure callback keeps the card grid aligned; the ambient
        # transition is handled by a short, independent animation timer.
        pass

    def _resize_card_host(self, event: tk.Event) -> None:
        self.game_canvas.itemconfigure(self.host_window, width=event.width)

    def _update_scroll_region(self, _event: tk.Event | None = None) -> None:
        try:
            self.game_canvas.configure(scrollregion=self.game_canvas.bbox("all"))
        except tk.TclError:
            pass

    def set_status(self, value: str) -> None:
        self.status_var.set(value)

    def open_matrix_folder(self) -> None:
        """Open the local CSV/JSONL matrix without uploading any collected data."""
        matrix_dir = APP_DATA / "matrix"
        try:
            matrix_dir.mkdir(parents=True, exist_ok=True)
            os.startfile(str(matrix_dir))
        except (OSError, AttributeError) as exc:
            messagebox.showerror(APP_NAME, f"Cannot open the local matrix folder: {exc}")

    def set_filter(self, value: str) -> None:
        self.search_filter = value
        self.render_games()

    def render_games(self) -> None:
        if not hasattr(self, "card_host"):
            return
        for child in self.card_host.winfo_children():
            child.destroy()
        self.cards.clear()
        query = self.search_var.get().strip().casefold()
        filtered = [
            game for game in self.games
            if (self.search_filter == "All" or game.platform == self.search_filter)
            and (not query or query in game.name.casefold() or query in game.exe_name.casefold())
            and _path_exists(game.exe_path)
        ]
        filtered.sort(key=lambda game: game.name.casefold())
        self.count_label.configure(text=f"{len(filtered)} GAMES")
        width = max(400, self.game_canvas.winfo_width())
        columns = max(2, min(5, width // 245))
        for index, game in enumerate(filtered):
            card_holder = tk.Frame(self.card_host, bg=BACKGROUND, padx=6, pady=6)
            card_holder.grid(
                row=index // columns, column=index % columns, sticky="nsew"
            )
            card = GameCard(self, card_holder, game)
            card.frame.pack(fill="both", expand=True)
            self.cards.append(card)
            self.card_host.grid_columnconfigure(index % columns, weight=1, uniform="game-column")
        for column in range(columns):
            self.card_host.grid_columnconfigure(column, weight=1, uniform="game-column")
        if not filtered:
            message = (
                "No matching games. Select Add folder or Scan all drives to search a location."
                if self.games else
                "No games indexed yet. Rescan installed stores or choose Add folder."
            )
            tk.Label(
                self.card_host, text=message, bg=BACKGROUND, fg=MUTED,
                font=("Segoe UI", 12), pady=55,
            ).grid(row=0, column=0, columnspan=columns, sticky="ew")
        self._update_scroll_region()

    def set_ambient_target(self, color: str) -> None:
        self.ambient_target = color
        if not self.ambient_animating:
            self.ambient_animating = True
            self.root.after(16, self._animate_ambient)

    def _animate_ambient(self) -> None:
        if self.closing:
            return
        next_color = _blend_hex(self.ambient_current, self.ambient_target, 0.22)
        if next_color == self.ambient_target or sum(
            abs(int(next_color[i:i + 2], 16) - int(self.ambient_target[i:i + 2], 16))
            for i in (1, 3, 5)
        ) <= 3:
            next_color = self.ambient_target
        self.ambient_current = next_color
        ambient = _blend_hex(BACKGROUND, next_color, 0.105)
        try:
            self.root.configure(bg=ambient)
            self.shell.configure(bg=ambient)
            self.grid_wrap.configure(bg=ambient)
            self.game_canvas.configure(bg=ambient)
            self.card_host.configure(bg=ambient)
            self.count_label.configure(bg="#12171e")
            for child in self.shell.winfo_children():
                if isinstance(child, tk.Frame) and child not in (self.grid_wrap,):
                    # Keep structural frames in the same dark environment while the
                    # icon-derived tint shifts under the card grid.
                    if child.cget("bg") == BACKGROUND:
                        child.configure(bg=ambient)
            if _blend_hex(next_color, self.ambient_target, 1.0) != next_color:
                pass
        except tk.TclError:
            self.ambient_animating = False
            return
        if next_color != self.ambient_target:
            self.root.after(16, self._animate_ambient)
        else:
            self.ambient_animating = False

    def _prepare_icons(self, games: list[GameEntry]) -> None:
        for game in games:
            if not _path_exists(game.exe_path):
                continue
            try:
                stat = Path(game.exe_path).stat()
                revision = f"{stat.st_mtime_ns}:{stat.st_size}"
            except OSError:
                revision = "missing"
            token = hashlib.sha256(
                f"{os.path.normcase(game.exe_path)}:{revision}".encode("utf-8", "replace")
            ).hexdigest()[:24]
            game.icon_path = ExeIconExtractor.save_icon(game.exe_path, ICON_CACHE / f"{token}.png")

    def scan_installed_games(self, silent: bool = False) -> None:
        if self.scan_thread and self.scan_thread.is_alive():
            self.set_status("A scan is already running. Wait for it to finish or cancel it.")
            return
        self.scan_stop.clear()
        self.set_status("Scanning installed stores and common game folders…")

        def work() -> None:
            results: list[GameEntry] = []
            results.extend(_steam_games(self.scan_stop))
            results.extend(_epic_games(self.scan_stop))

            for name, install in _registry_install_locations():
                if self.scan_stop.is_set():
                    break
                if not _is_reasonable_game_title(name):
                    continue
                exe = choose_game_executable(install, name, stop_event=self.scan_stop)
                if exe:
                    results.append(_new_entry(name, exe, platform_name="Disk", source="uninstall-registry"))

            for root in _known_game_roots():
                if self.scan_stop.is_set():
                    break
                platform_name = "Epic Games" if "epic" in str(root).casefold() else "Disk"
                results.extend(_scan_game_root(root, platform_name=platform_name, stop_event=self.scan_stop))

            for root_string in list(self.custom_roots):
                if self.scan_stop.is_set():
                    break
                results.extend(_scan_game_root(Path(root_string), stop_event=self.scan_stop))

            by_exe: dict[str, GameEntry] = {}
            for game in results + list(self.games):
                if not _path_exists(game.exe_path):
                    continue
                key = game.key
                current = by_exe.get(key)
                if current is None or self._source_priority(game.source) > self._source_priority(current.source):
                    by_exe[key] = game
            merged = sorted(by_exe.values(), key=lambda item: item.name.casefold())
            self._prepare_icons(merged)
            self.events.put(("scan_done", {"games": merged, "silent": silent, "cancelled": self.scan_stop.is_set()}))

        self.scan_thread = threading.Thread(target=work, name="TrSS-Game-Scan", daemon=True)
        self.scan_thread.start()

    @staticmethod
    def _source_priority(source: str) -> int:
        return {
            "steam-manifest": 5,
            "epic-manifest": 5,
            "uninstall-registry": 4,
            "folder-scan": 3,
            "full-drive-scan": 2,
            "scan": 1,
        }.get(source, 0)

    def add_folder(self) -> None:
        folder = filedialog.askdirectory(
            title="Choose a folder containing installed games",
            initialdir=str(Path.home()),
            mustexist=True,
        )
        if not folder:
            return
        root = Path(folder)
        if str(root) not in self.custom_roots:
            self.custom_roots.append(str(root))
        self.set_status(f"Scanning {root} for game executables…")

        def work() -> None:
            added = _scan_game_root(root, stop_event=self.scan_stop)
            self._prepare_icons(added)
            self.events.put(("folder_done", {"root": str(root), "games": added}))

        threading.Thread(target=work, name="TrSS-Added-Folder-Scan", daemon=True).start()

    def scan_all_drives(self) -> None:
        if self.scan_thread and self.scan_thread.is_alive():
            messagebox.showwarning(APP_NAME, "A game scan is already running.")
            return
        confirm = messagebox.askyesno(
            APP_NAME,
            "Scan fixed drives for executable files that look like games?\n\n"
            "This can take a while on large disks. System folders are skipped. "
            "The scan stays on this PC and you can cancel it.",
        )
        if not confirm:
            return
        self.scan_stop.clear()
        self.set_status("Starting optional deep scan of local fixed drives…")

        def work() -> None:
            games = _scan_all_fixed_drives(
                self.scan_stop,
                lambda text_value: self.events.put(("status", text_value)),
            )
            self._prepare_icons(games)
            self.events.put(("full_scan_done", {"games": games, "cancelled": self.scan_stop.is_set()}))

        self.scan_thread = threading.Thread(target=work, name="TrSS-Full-Drive-Scan", daemon=True)
        self.scan_thread.start()

    def launch_game(self, game: GameEntry) -> None:
        if not _path_exists(game.exe_path):
            messagebox.showerror(APP_NAME, f"The game executable could not be found:\n{game.exe_path}")
            self.scan_installed_games(silent=True)
            return

        self.selected_game = game
        self.overlay.update_metrics(self.last_metrics, game, self.active_capture is not None)
        try:
            if game.launch_uri:
                try:
                    os.startfile(game.launch_uri)  # type: ignore[attr-defined]
                except OSError:
                    subprocess.Popen([game.exe_path], cwd=str(Path(game.exe_path).parent))
            else:
                subprocess.Popen([game.exe_path], cwd=str(Path(game.exe_path).parent))
        except (OSError, ValueError) as exc:
            messagebox.showerror(
                APP_NAME,
                f"Could not launch {game.name}.\n\n{exc}\n\n"
                "Some games must be started through their own launcher. Try the game's store client.",
            )
            return

        self.overlay.show()
        self.set_status(
            f"Launched {game.name}. OSD is active; press {STOP_CAPTURE_HOTKEY} while the game is focused to begin an unlimited capture."
        )
        # Leave a moment for the game window to take focus before hiding the library.
        self.root.after(900, self.root.iconify)

    def show_overlay(self) -> None:
        self.overlay.show()
        if not self.selected_game:
            self.set_status("Select and launch a game to attach performance readings to its process.")

    def show_library(self) -> None:
        try:
            self.root.deiconify()
            self.root.state("normal")
            self.root.lift()
            self.root.focus_force()
        except tk.TclError:
            pass

    def _start_monitor(self) -> None:
        if self.monitor_thread and self.monitor_thread.is_alive():
            return

        def monitor() -> None:
            while not self.monitor_stop.is_set():
                game = self.selected_game
                try:
                    metrics = self.monitor_reader.sample(game)
                except Exception as exc:
                    metrics = {
                        "timestamp_utc": utc_now_text(),
                        "game": game.name if game else None,
                        "game_window_found": False,
                        "fps": None,
                        "fps_source": "Native TrSS per-present telemetry is in development",
                        "monitor_error": type(exc).__name__,
                    }
                self.events.put(("metrics", metrics))
                self._append_metric_if_recording(metrics)
                self.monitor_stop.wait(1.0)
            self.monitor_reader.close()

        self.monitor_thread = threading.Thread(target=monitor, name="TrSS-Performance-Monitor", daemon=True)
        self.monitor_thread.start()

    def _append_metric_if_recording(self, metrics: dict[str, Any]) -> None:
        with self.capture_lock:
            session = self.active_capture
            if not session or session.closing:
                return
            with session.lock:
                if not session.closing:
                    session.performance_rows.append(dict(metrics))

    def start_capture(self) -> None:
        with self.capture_lock:
            if self.active_capture:
                self.set_status("A capture is already running. Press the capture hotkey again to stop.")
                return
        game = self.selected_game
        if not game:
            self.set_status("Launch a game first. The collector only records the selected game window.")
            return
        if not self.rights_consent_var.get() or not self.share_consent_var.get():
            self.show_library()
            messagebox.showwarning(
                APP_NAME,
                "Confirm both permission and local research-data collection in the library before recording.",
            )
            return
        target = _find_game_window(game)
        if not target:
            self.set_status(f"{game.name}: waiting for its game window. Focus the running game and retry.")
            return
        foreground = _foreground_hwnd()
        if foreground != target[0]:
            self.set_status("Focus the selected game first, then press Ctrl + Alt + F8 again.")
            return

        output_root = APP_DATA / "captures"
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(output_root).free
            if free < 500 * 1024 * 1024:
                messagebox.showwarning(APP_NAME, "Less than 500 MB is free in the capture location. Choose another drive in the library before recording.")
                return
        except OSError as exc:
            messagebox.showerror(APP_NAME, f"Cannot prepare local capture storage: {exc}")
            return

        session_id = str(uuid.uuid4())
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"trss-{safe_label(game.name, 'game', 48).replace(' ', '-')}-{stamp}-{session_id[:8]}"
        folder = output_root / folder_name
        zip_path = output_root / f"{folder_name}.zip"
        try:
            folder.mkdir(parents=True, exist_ok=False)
            (folder / "frames").mkdir()
        except OSError as exc:
            messagebox.showerror(APP_NAME, f"Cannot create capture session: {exc}")
            return

        session = CaptureSession(
            session_id=session_id,
            game=game,
            folder=folder,
            zip_path=zip_path,
            sample_rate=max(1, min(10, int(self.capture_rate_var.get()))),
            stop_event=threading.Event(),
            started_utc=utc_now_text(),
            started_monotonic=time.monotonic(),
            frame_rows=[],
            performance_rows=[],
            lock=threading.Lock(),
        )
        with self.capture_lock:
            self.active_capture = session
        worker = threading.Thread(
            target=self._capture_worker, args=(session, target[0]),
            name="TrSS-Continuous-Game-Capture", daemon=True,
        )
        worker.start()
        self.overlay.update_metrics(self.last_metrics, game, True)
        self.set_status(
            f"CAPTURING {game.name} at {session.sample_rate} FPS with no duration or frame-count limit. Press {STOP_CAPTURE_HOTKEY} to stop."
        )

    def stop_capture(self, reason: str = "user_stop") -> None:
        with self.capture_lock:
            session = self.active_capture
            if not session:
                return
            session.stop_reason = reason
            session.stop_event.set()
        self.set_status("Stopping collection and packaging the saved sample…")

    def _review_graphics_settings(self, session: CaptureSession) -> None:
        """Ask for the exact settings used in this run before its dataset is packaged."""
        detected = session.detected_settings or {}
        initial_values = dict(detected.get("values") or {})
        baseline_values = dict(initial_values)
        baseline_sources = dict(detected.get("sources") or {})
        baseline_names = dict(detected.get("source_names") or {})
        profile_values: dict[str, Any] = {}

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Graphics Settings Review — {session.game.name}")
        dialog.geometry("1020x850")
        dialog.minsize(860, 680)
        dialog.configure(bg=BACKGROUND)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.lift()

        outer = tk.Frame(dialog, bg=BACKGROUND)
        outer.pack(fill="both", expand=True, padx=18, pady=14)
        tk.Label(
            outer, text="GRAPHICS PROFILE / POST-CAPTURE REVIEW",
            bg=BACKGROUND, fg=TEXT, font=("Segoe UI Semibold", 17), anchor="w",
        ).pack(fill="x")
        tk.Label(
            outer,
            text=(
                f"{session.game.name} · {len(initial_values)} setting value(s) read from "
                f"{int(detected.get('matched_file_count') or 0)} local configuration file(s). "
                "Config values may be defaults or stale: compare them with the game's own menu."
            ),
            bg=BACKGROUND, fg=MUTED, font=("Segoe UI", 9), anchor="w", justify="left", wraplength=960,
        ).pack(fill="x", pady=(5, 10))

        identity = tk.Frame(outer, bg=BACKGROUND)
        identity.pack(fill="x", pady=(0, 8))
        tk.Label(identity, text="PROFILE LABEL", bg=BACKGROUND, fg=MUTED, font=("Consolas", 9)).grid(row=0, column=0, sticky="w")
        profile_var = tk.StringVar(
            master=dialog, value=f"Run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        ttk.Entry(identity, textvariable=profile_var, width=34).grid(row=1, column=0, sticky="ew", padx=(0, 12), pady=(3, 0))
        tk.Label(identity, text="EXPERIMENT / SWEEP GROUP", bg=BACKGROUND, fg=MUTED, font=("Consolas", 9)).grid(row=0, column=1, sticky="w")
        group_var = tk.StringVar(master=dialog, value=session.game.name)
        ttk.Entry(identity, textvariable=group_var, width=34).grid(row=1, column=1, sticky="ew", padx=(0, 12), pady=(3, 0))
        identity.columnconfigure(0, weight=1)
        identity.columnconfigure(1, weight=1)

        coverage_var = tk.StringVar(master=dialog)
        tk.Label(
            outer, textvariable=coverage_var, bg=BACKGROUND, fg="#dce2ea",
            font=("Consolas", 9, "bold"), anchor="w",
        ).pack(fill="x", pady=(0, 6))

        scroll_host = tk.Frame(outer, bg=BACKGROUND)
        scroll_host.pack(fill="both", expand=True)
        canvas = tk.Canvas(scroll_host, bg=BACKGROUND, highlightthickness=0, bd=0)
        scrollbar = ttk.Scrollbar(scroll_host, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        body = tk.Frame(canvas, bg=BACKGROUND)
        body_window = canvas.create_window((0, 0), window=body, anchor="nw")

        def update_scroll_region(_event: tk.Event | None = None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def resize_body(event: tk.Event) -> None:
            canvas.itemconfigure(body_window, width=event.width)

        body.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", resize_body)

        field_vars: dict[str, tk.StringVar] = {}
        source_labels: dict[str, tk.Label] = {}
        field_rows: dict[str, tk.Entry] = {}
        for group_name, field_names in FIELD_GROUPS:
            group_frame = tk.LabelFrame(
                body, text=group_name.upper(), bg=BACKGROUND, fg=TEXT,
                font=("Segoe UI Semibold", 9), bd=1, relief="solid",
                highlightbackground="#252c36", padx=10, pady=5,
            )
            group_frame.pack(fill="x", padx=(0, 4), pady=(0, 9))
            group_frame.columnconfigure(1, weight=1)
            for row_number, field_name in enumerate(field_names):
                current = initial_values.get(field_name)
                value_var = tk.StringVar(master=dialog, value="" if current is None else str(current))
                field_vars[field_name] = value_var
                tk.Label(
                    group_frame, text=STANDARD_SETTINGS[field_name], bg=BACKGROUND, fg="#d3d9e2",
                    anchor="w", width=27,
                ).grid(row=row_number, column=0, sticky="w", padx=(0, 10), pady=3)
                entry = ttk.Entry(group_frame, textvariable=value_var, width=31)
                entry.grid(row=row_number, column=1, sticky="ew", padx=(0, 10), pady=3)
                field_rows[field_name] = entry
                source_name = baseline_names.get(field_name)
                source_text = (
                    f"Detected: {source_name} — verify" if source_name
                    else "No config value detected"
                )
                source_label = tk.Label(
                    group_frame, text=source_text, bg=BACKGROUND,
                    fg="#8ea4bf" if source_name else "#687382", anchor="w",
                    font=("Segoe UI", 8), width=31,
                )
                source_label.grid(row=row_number, column=2, sticky="w", pady=3)
                source_labels[field_name] = source_label

        additional_box = tk.LabelFrame(
            body, text="ADDITIONAL GAME-SPECIFIC SETTINGS", bg=BACKGROUND, fg=TEXT,
            font=("Segoe UI Semibold", 9), padx=10, pady=7,
        )
        additional_box.pack(fill="x", padx=(0, 4), pady=(0, 9))
        tk.Label(
            additional_box,
            text="One exact setting per line, written as  Setting name = Value. Use this for options not in the standard matrix.",
            bg=BACKGROUND, fg=MUTED, anchor="w", justify="left", wraplength=900,
        ).pack(fill="x", pady=(0, 5))
        additional_text = tk.Text(
            additional_box, height=5, wrap="none", bg="#11161c", fg=TEXT,
            insertbackground=TEXT, relief="flat", font=("Consolas", 9),
        )
        additional_text.pack(fill="x")
        detected_additional = detected.get("additional_settings") or {}
        if detected_additional:
            additional_text.insert(
                "1.0",
                "\n".join(f"{name} = {value}" for name, value in detected_additional.items()),
            )

        notes_box = tk.LabelFrame(
            body, text="RUN NOTES (NO PERSONAL OR ACCOUNT DATA)", bg=BACKGROUND, fg=TEXT,
            font=("Segoe UI Semibold", 9), padx=10, pady=7,
        )
        notes_box.pack(fill="x", padx=(0, 4), pady=(0, 8))
        notes_text = tk.Text(
            notes_box, height=3, wrap="word", bg="#11161c", fg=TEXT,
            insertbackground=TEXT, relief="flat", font=("Segoe UI", 9),
        )
        notes_text.pack(fill="x")

        def update_coverage(*_args: Any) -> None:
            filled = sum(bool(variable.get().strip()) for variable in field_vars.values())
            coverage_var.set(
                f"STANDARD SETTINGS RECORDED: {filled}/{len(STANDARD_SETTINGS)}  ·  "
                "Use Unknown or Not supported explicitly; blank fields remain missing."
            )

        for variable in field_vars.values():
            variable.trace_add("write", update_coverage)
        update_coverage()

        verification_var = tk.BooleanVar(master=dialog, value=False)
        verify_check = ttk.Checkbutton(
            outer,
            text="I compared this profile with the game's in-game Graphics / Display menu for this exact run.",
            variable=verification_var,
        )
        verify_check.pack(anchor="w", pady=(8, 5))

        def import_config_file() -> None:
            selected = filedialog.askopenfilename(
                parent=dialog,
                title="Import a game graphics settings file",
                filetypes=[
                    ("Game settings", "*.xml *.ini *.cfg *.json *.toml *.txt"),
                    ("All files", "*.*"),
                ],
            )
            if not selected:
                return
            parsed = parse_graphics_config_file(selected)
            values = parsed.get("values") or {}
            if not values:
                messagebox.showwarning(
                    APP_NAME,
                    "No recognized graphics setting keys were found. TrSS does not import the whole file; only known graphics keys are read.",
                    parent=dialog,
                )
                return
            base_name = Path(selected).name
            for field_name, value in values.items():
                if field_name not in field_vars:
                    continue
                field_vars[field_name].set(str(value))
                baseline_values[field_name] = str(value)
                baseline_sources[field_name] = "imported_local_config"
                baseline_names[field_name] = base_name
                source_labels[field_name].configure(text=f"Imported: {base_name} — verify", fg="#8ea4bf")
            self.set_status(f"Imported {len(values)} recognized setting value(s) from {base_name}; verify them in-game.")

        footer = tk.Frame(outer, bg=BACKGROUND)
        footer.pack(fill="x", pady=(8, 0))
        ttk.Button(footer, text="Import config…", command=import_config_file).pack(side="left")

        def save_profile() -> None:
            verified = bool(verification_var.get())
            values: dict[str, str | None] = {
                field_name: variable.get().strip() or None
                for field_name, variable in field_vars.items()
            }
            sources: dict[str, str] = {}
            source_names: dict[str, str] = {}
            for field_name, value in values.items():
                if value is None:
                    continue
                if (
                    field_name in baseline_values
                    and value == str(baseline_values[field_name])
                    and field_name in baseline_sources
                ):
                    sources[field_name] = baseline_sources[field_name]
                    if baseline_names.get(field_name):
                        source_names[field_name] = baseline_names[field_name]
                else:
                    sources[field_name] = "participant_verified" if verified else "participant_reported"

            additional = parse_additional_settings(additional_text.get("1.0", "end-1c"))
            profile_values.update({
                "profile_name": profile_var.get().strip() or "Unlabelled run",
                "experiment_group": group_var.get().strip(),
                "verified_in_game": verified,
                "values": values,
                "sources": sources,
                "source_names": source_names,
                "additional_settings": additional,
                "notes": notes_text.get("1.0", "end-1c").strip(),
                "files_checked": int(detected.get("files_checked") or 0),
                "matched_file_count": int(detected.get("matched_file_count") or 0),
                "matched_file_names": list(detected.get("matched_file_names") or []),
            })
            session.graphics_settings = profile_values
            session.review_event.set()
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()

        save_button = ttk.Button(
            footer,
            text="Save verified profile" if verification_var.get() else "Save unverified profile",
            command=save_profile,
        )
        save_button.pack(side="right")
        verification_var.trace_add(
            "write",
            lambda *_args: save_button.configure(
                text="Save verified profile" if verification_var.get() else "Save unverified profile"
            ),
        )
        dialog.protocol("WM_DELETE_WINDOW", save_profile)
        dialog.wait_window()

        # If window destruction occurs unusually, do not strand the capture worker.
        if not session.review_event.is_set():
            session.graphics_settings = {
                "profile_name": "Unverified run",
                "experiment_group": "",
                "verified_in_game": False,
                "values": initial_values,
                "sources": baseline_sources,
                "source_names": baseline_names,
                "additional_settings": {},
                "notes": "",
                "files_checked": int(detected.get("files_checked") or 0),
                "matched_file_count": int(detected.get("matched_file_count") or 0),
                "matched_file_names": list(detected.get("matched_file_names") or []),
            }
            session.review_event.set()

    def _capture_worker(self, session: CaptureSession, hwnd: int) -> None:
        """Capture until stopped; the only limits are user stop, focus loss, or a real error."""
        frame_interval = 1.0 / session.sample_rate
        next_capture = time.monotonic()
        last_status_update = 0.0
        try:
            while not session.stop_event.is_set():
                now = time.monotonic()
                if now < next_capture:
                    session.stop_event.wait(min(0.02, next_capture - now))
                    continue
                if _foreground_hwnd() != hwnd:
                    raise CaptureError(
                        "The game lost focus. Collection stopped so unrelated windows were not captured."
                    )
                frame = self._capture_game_window(hwnd)
                elapsed = time.monotonic() - session.started_monotonic
                frame_name = f"frame-{len(session.frame_rows):08d}.png"
                frame_path = session.folder / "frames" / frame_name
                frame.save(frame_path, format="PNG", optimize=False)
                session.captured_bytes += frame_path.stat().st_size
                session.frame_rows.append(
                    make_sequence_row(
                        frame_id=len(session.frame_rows),
                        timestamp=elapsed,
                        scene_id=session.session_id[:12],
                        color_path=f"frames/{frame_name}",
                    )
                )
                next_capture += frame_interval
                now = time.monotonic()
                if now > next_capture:
                    # Skip missed slots; never duplicate frames or falsify timing.
                    missed = int((now - next_capture) // frame_interval) + 1
                    next_capture += missed * frame_interval
                if now - last_status_update >= 2.0:
                    last_status_update = now
                    self.events.put((
                        "capture_progress",
                        {
                            "frames": len(session.frame_rows),
                            "elapsed": now - session.started_monotonic,
                            "folder": str(session.folder),
                        },
                    ))
                try:
                    free = shutil.disk_usage(session.folder).free
                    # ZIP creation temporarily needs room for another copy of the bundle.
                    # Stop early enough to preserve both the PNGs and the finished archive.
                    if free < session.captured_bytes + 512 * 1024 * 1024:
                        session.status = "interrupted"
                        session.error = (
                            "Capture stopped to preserve enough free space to package the "
                            "collected files safely. Choose a larger or emptier drive for longer runs."
                        )
                        break
                except OSError:
                    pass
        except CaptureError as exc:
            session.status = "interrupted"
            session.error = str(exc)
        except (OSError, ValueError, RuntimeError) as exc:
            session.status = "interrupted"
            session.error = f"Capture failed ({type(exc).__name__})."
        except Exception as exc:
            session.status = "interrupted"
            session.error = f"Unexpected capture failure ({type(exc).__name__})."

        if session.status == "recording":
            session.status = "completed"
        with self.capture_lock:
            with session.lock:
                session.closing = True
                metrics_rows = list(session.performance_rows)
            if self.active_capture is session:
                self.active_capture = None

        if not session.frame_rows:
            shutil.rmtree(session.folder, ignore_errors=True)
            self.events.put(("capture_error", session.error or "No frames were captured. Focus the game before starting."))
            return

        # Read only allowlisted graphics keys from likely local settings files.
        # The participant still needs to confirm these against the live game menu:
        # a file can contain an old profile or an engine-specific numeric enum.
        try:
            session.detected_settings = probe_game_settings(
                session.game.name, session.game.install_dir
            )
        except Exception as exc:
            session.detected_settings = {
                "values": {}, "sources": {}, "source_names": {},
                "files_checked": 0, "matched_file_count": 0,
                "matched_file_names": [], "probe_error": type(exc).__name__,
            }

        # Pause packaging until the user labels and checks the captured configuration.
        self.events.put(("capture_review", session))
        session.review_event.wait()
        graphics_profile = session.graphics_settings or {
            "profile_name": "Unverified run",
            "experiment_group": "",
            "verified_in_game": False,
            "values": session.detected_settings.get("values", {}),
            "sources": session.detected_settings.get("sources", {}),
            "source_names": session.detected_settings.get("source_names", {}),
            "additional_settings": {},
            "notes": "",
            "files_checked": session.detected_settings.get("files_checked", 0),
            "matched_file_count": session.detected_settings.get("matched_file_count", 0),
            "matched_file_names": session.detected_settings.get("matched_file_names", []),
        }

        elapsed = round(time.monotonic() - session.started_monotonic, 4)
        matrix_row = build_matrix_row(
            session_id=session.session_id,
            game_name=session.game.name,
            platform=session.game.platform,
            started_utc=session.started_utc,
            elapsed_seconds=elapsed,
            captured_frames=len(session.frame_rows),
            performance_rows=metrics_rows,
            graphics_profile=graphics_profile,
        )
        graphics_payload = {
            "schema_version": "trss.graphics-settings/1.0",
            "game": {"name": session.game.name, "platform": session.game.platform},
            "profile": {
                "name": matrix_row["profile_name"],
                "experiment_group": matrix_row["experiment_group"],
                "verified_in_game": matrix_row["profile_verified_in_game"],
                "notes": matrix_row["participant_notes"],
            },
            "settings": matrix_row["settings"],
            "setting_sources": matrix_row["setting_sources"],
            "source_file_names": matrix_row["source_file_names"],
            "additional_game_settings": matrix_row["additional_game_settings"],
            "automatic_config_probe": {
                "files_checked": matrix_row["config_discovery"]["files_checked"],
                "files_with_graphics_settings": matrix_row["config_discovery"]["files_with_graphics_settings"],
                "matched_file_names": list(graphics_profile.get("matched_file_names") or []),
                "warning": (
                    "Detected config values can be stale or use game-specific numeric enums. "
                    "Only a participant check against the in-game menu marks a profile verified."
                ),
            },
        }
        (session.folder / "graphics_settings.json").write_text(
            json.dumps(graphics_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (session.folder / "game_matrix_row.json").write_text(
            json.dumps(matrix_row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        metadata = {
            "schema_version": "3.0",
            "application": {"name": APP_NAME, "version": APP_VERSION},
            "session": {
                "id": session.session_id,
                "game": session.game.name,
                "platform": session.game.platform,
                "started_utc": session.started_utc,
                "ended_utc": utc_now_text(),
                "status": session.status,
                "termination_reason": session.error or session.stop_reason,
                "elapsed_seconds": elapsed,
                "sample_rate_requested_fps": session.sample_rate,
                "captured_frames": len(session.frame_rows),
                "performance_samples": len(metrics_rows),
                "frame_encoding": "lossless RGB PNG",
                "capture_method": "visible active game-window screen sample",
                "audio_recorded": False,
                "automatic_upload": False,
                "depth_buffer_collected": False,
                "motion_vectors_collected": False,
                "fps_note": "Native TrSS per-present FPS and frametime telemetry is still in development. Values remain null rather than being estimated from refresh rate, screenshots, or GPU load.",
            },
            "graphics_matrix": {
                "settings_file": "graphics_settings.json",
                "row_file": "game_matrix_row.json",
                "profile_name": matrix_row["profile_name"],
                "experiment_group": matrix_row["experiment_group"],
                "profile_verified_in_game": matrix_row["profile_verified_in_game"],
                "matrix_row_complete": matrix_row["matrix_row_complete"],
                "matrix_row_exact": matrix_row["matrix_row_exact"],
                "settings_hash_sha256": matrix_row["settings_hash_sha256"],
                "known_standard_settings": matrix_row["settings_coverage"]["known_standard_settings"],
                "total_standard_settings": matrix_row["settings_coverage"]["total_standard_settings"],
                "coverage_percent": matrix_row["settings_coverage"]["coverage_percent"],
                "missing_settings": matrix_row["settings_coverage"]["missing_settings"],
                "note": "One matrix row represents this recording and its reviewed graphics profile. Repeat capture after changing the game settings to build a multi-profile matrix.",
            },
            "consent": {
                "policy_version": "2026-10-14",
                "permission_confirmed": True,
                "research_collection_confirmed": True,
                "identity_collected": False,
            },
            "privacy": {
                "network_upload": False,
                "game_install_path_in_bundle": False,
                "raw_window_title_in_bundle": False,
                "hostname_collected": False,
                "account_name_collected": False,
                "clipboard_read": False,
                "audio_recorded": False,
                "device_serial_numbers_collected": False,
                "game_process_path_in_bundle": False,
            },
        }
        try:
            create_bundle(
                session_dir=session.folder,
                zip_path=session.zip_path,
                metadata=metadata,
                system_metadata=collect_system_metadata(),
                sequence_rows=session.frame_rows,
                keep_unpacked=False,
            )
        except Exception as exc:
            self.events.put((
                "capture_error",
                f"Captured {len(session.frame_rows)} frame(s), but ZIP packaging failed ({type(exc).__name__}). Files kept at {session.folder}",
            ))
            return

        try:
            _jsonl_path, matrix_csv_path = append_game_matrix_row(APP_DATA / "matrix", matrix_row)
            session.matrix_csv_path = str(matrix_csv_path)
        except Exception as exc:
            # The per-session ZIP already contains the matrix row and setting evidence.
            session.matrix_index_error = type(exc).__name__
        self.events.put((
            "capture_complete",
            {
                "zip_path": str(session.zip_path),
                "frames": len(session.frame_rows),
                "elapsed": elapsed,
                "performance_samples": len(metrics_rows),
                "status": session.status,
                "error": session.error,
                "profile_name": matrix_row["profile_name"],
                "matrix_row_complete": matrix_row["matrix_row_complete"],
                "matrix_row_exact": matrix_row["matrix_row_exact"],
                "settings_coverage": matrix_row["settings_coverage"]["coverage_percent"],
                "matrix_csv_path": session.matrix_csv_path,
                "matrix_index_error": session.matrix_index_error,
            },
        ))

    @staticmethod
    def _capture_game_window(hwnd: int) -> Image.Image:
        if _foreground_hwnd() != hwnd:
            raise CaptureError("The game lost focus before the screenshot started.")
        bounds = _window_client_bounds(hwnd)
        try:
            image = ImageGrab.grab(bbox=bounds, all_screens=True)
        except (OSError, RuntimeError) as exc:
            raise CaptureError("Could not read the active game-window pixels. Try borderless-windowed mode.") from exc
        expected = (bounds[2] - bounds[0], bounds[3] - bounds[1])
        if image.size != expected:
            raise CaptureError("The game window changed size or Windows display scaling made the capture inconsistent.")
        if _foreground_hwnd() != hwnd:
            raise CaptureError("The game lost focus during screenshot capture; the frame was discarded.")
        return image.convert("RGB")

    def _poll_hotkeys(self) -> None:
        if self.closing:
            return
        try:
            if self.root.winfo_exists():
                down = 0x8000
                if os.name == "nt":
                    user32 = ctypes.windll.user32
                    ctrl_alt = bool(user32.GetAsyncKeyState(0x11) & down) and bool(user32.GetAsyncKeyState(0x12) & down)
                    for vk in (0x77, 0x78, 0x79):
                        is_down = bool(user32.GetAsyncKeyState(vk) & down)
                        just_pressed = is_down and not self.hotkey_state.get(vk, False)
                        self.hotkey_state[vk] = is_down
                        if not (ctrl_alt and just_pressed):
                            continue
                        if vk == 0x77:  # F8: toggle capture without repeat while held
                            self.toggle_capture()
                        elif vk == 0x78:  # F9: restore the library
                            self.show_library()
                        elif vk == 0x79:  # F10: stop capture
                            self.stop_capture("hotkey_stop")
                # The OSD is hidden when the library is visible so it never covers its controls.
                if self.root.state() == "normal" and self.root.winfo_viewable():
                    self.overlay.hide()
                elif self.selected_game:
                    self.overlay.show()
                self.overlay.update_metrics(
                    self.last_metrics, self.selected_game, self.active_capture is not None
                )
        except (tk.TclError, OSError):
            pass
        self.root.after(160, self._poll_hotkeys)


    def toggle_capture(self) -> None:
        if self.active_capture:
            self.stop_capture("hotkey_toggle")
        else:
            self.start_capture()

    def _drain_events(self) -> None:
        if self.closing:
            return
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "scan_done":
                    self.games = payload["games"]
                    save_library(self.games, self.custom_roots)
                    self.render_games()
                    self.set_status(
                        f"Found {len(self.games)} game executables. Hover a card for its icon tint; click one to launch."
                    )
                elif kind == "folder_done":
                    existing = {game.key for game in self.games}
                    self.games.extend(game for game in payload["games"] if game.key not in existing)
                    self.games.sort(key=lambda game: game.name.casefold())
                    save_library(self.games, self.custom_roots)
                    self.render_games()
                    self.set_status(
                        f"Folder scan complete: {len(payload['games'])} game(s) found in {payload['root']}."
                    )
                elif kind == "full_scan_done":
                    existing = {game.key for game in self.games}
                    self.games.extend(game for game in payload["games"] if game.key not in existing)
                    self.games.sort(key=lambda game: game.name.casefold())
                    save_library(self.games, self.custom_roots)
                    self.render_games()
                    self.set_status(f"Deep scan {'cancelled' if payload['cancelled'] else 'complete'}: {len(self.games)} game(s) indexed.")
                elif kind == "status":
                    self.set_status(str(payload))
                elif kind == "metrics":
                    self.last_metrics = payload
                    self.overlay.update_metrics(payload, self.selected_game, self.active_capture is not None)
                elif kind == "capture_progress":
                    self.set_status(
                        f"CAPTURING · {payload['frames']:,} frames · {payload['elapsed']:.0f}s elapsed · unlimited duration"
                    )
                elif kind == "capture_review":
                    try:
                        self._review_graphics_settings(payload)
                    except Exception as exc:
                        # Still package the frames if the review UI itself fails; mark the row unverified.
                        payload.graphics_settings = {
                            "profile_name": "Unverified run",
                            "experiment_group": payload.game.name,
                            "verified_in_game": False,
                            "values": payload.detected_settings.get("values", {}),
                            "sources": payload.detected_settings.get("sources", {}),
                            "source_names": payload.detected_settings.get("source_names", {}),
                            "additional_settings": {},
                            "notes": f"Review dialog failed: {type(exc).__name__}",
                            "files_checked": payload.detected_settings.get("files_checked", 0),
                            "matched_file_count": payload.detected_settings.get("matched_file_count", 0),
                            "matched_file_names": payload.detected_settings.get("matched_file_names", []),
                        }
                        payload.review_event.set()
                elif kind == "capture_complete":
                    self.overlay.update_metrics(self.last_metrics, self.selected_game, False)
                    completeness = (
                        "EXACT VERIFIED MATRIX ROW" if payload["matrix_row_exact"]
                        else f"matrix {payload['settings_coverage']:.1f}% · "
                             f"{'profile fields filled' if payload['matrix_row_complete'] else 'missing setting fields'}"
                    )
                    self.set_status(
                        f"Bundle verified · {payload['frames']:,} frames · "
                        f"{payload['performance_samples']} performance samples · {completeness} · "
                        f"{payload['zip_path']}"
                    )
                    if payload["matrix_index_error"]:
                        self.set_status(
                            f"Bundle contains the settings row, but the shared matrix index could not be updated "
                            f"({payload['matrix_index_error']}). ZIP: {payload['zip_path']}"
                        )
                    elif payload["matrix_csv_path"]:
                        self.set_status(
                            f"Bundle verified · {payload['frames']:,} frames · {completeness} · "
                            f"matrix: {payload['matrix_csv_path']} · ZIP: {payload['zip_path']}"
                        )
                elif kind == "capture_error":
                    self.overlay.update_metrics(self.last_metrics, self.selected_game, False)
                    self.set_status(str(payload))
                self._update_scroll_region()
        except queue.Empty:
            pass
        if not self.closing:
            self.root.after(120, self._drain_events)

    def on_close(self) -> None:
        if self.active_capture:
            if not messagebox.askyesno(
                APP_NAME,
                "A data capture is running. Stop it, package the files, and close the library?",
            ):
                return
            self.stop_capture("application_exit")
            self.root.after(700, self._close_after_capture)
            return
        self._shutdown()

    def _close_after_capture(self) -> None:
        if self.active_capture:
            self.root.after(400, self._close_after_capture)
        else:
            self._shutdown()

    def _shutdown(self) -> None:
        self.closing = True
        self.monitor_stop.set()
        self.scan_stop.set()
        try:
            self.overlay.window.destroy()
        except tk.TclError:
            pass
        self.root.destroy()


def main() -> None:
    """Launch the native-feeling Windows game library and performance overlay."""
    if os.name != "nt":
        raise SystemExit("TrSS Game Library currently requires Windows 10/11.")
    APP_DATA.mkdir(parents=True, exist_ok=True)
    ICON_CACHE.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    GameHubApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
