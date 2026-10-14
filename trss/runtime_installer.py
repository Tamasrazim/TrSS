"""Install or remove the original TrSS D3D11/DXGI runtime beside a selected game.

This installer intentionally refuses to overwrite any existing dxgi.dll. Such a
file may belong to another graphics layer, compatibility shim, or game mod.
It never downloads, embeds, or replaces another project's runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime, timezone
from typing import Any

APP_NAME = "TrSS Runtime Installer"
IS_WINDOWS = os.name == "nt"
APP_VERSION = "0.1.0"
RUNTIME_DLL_NAME = "dxgi.dll"
CONFIG_NAME = "TrSS.ini"
TRSS_FOLDER = "TrSS"
PRESET_FOLDER = "Presets"
MANIFEST_NAME = "install-manifest.json"

DEFAULT_CONFIG = """[TrSSRuntime]
Product=TrSS
Version=0.1.0
ActivePreset=Neutral

[Effects]
Enabled=1
Brightness=0
Contrast=1
Saturation=1
Gamma=1
Sharpen=0
Bloom=0
Vignette=0
Grain=0
Temperature=0
Chromatic=0
"""

PRESETS: dict[str, dict[str, float]] = {
    "Neutral": {
        "Brightness": 0, "Contrast": 1, "Saturation": 1, "Gamma": 1,
        "Sharpen": 0, "Bloom": 0, "Vignette": 0, "Grain": 0,
        "Temperature": 0, "Chromatic": 0,
    },
    "Cinematic": {
        "Brightness": 0, "Contrast": 1.10, "Saturation": 0.88, "Gamma": 1,
        "Sharpen": 0.18, "Bloom": 0.22, "Vignette": 0.18, "Grain": 0,
        "Temperature": 0.06, "Chromatic": 0,
    },
    "Vibrant": {
        "Brightness": 0, "Contrast": 1.08, "Saturation": 1.38, "Gamma": 1,
        "Sharpen": 0.12, "Bloom": 0.12, "Vignette": 0, "Grain": 0,
        "Temperature": 0, "Chromatic": 0,
    },
    "Clarity": {
        "Brightness": 0, "Contrast": 1.06, "Saturation": 1.04, "Gamma": 1,
        "Sharpen": 0.72, "Bloom": 0, "Vignette": 0, "Grain": 0,
        "Temperature": 0, "Chromatic": 0,
    },
    "Night": {
        "Brightness": 0.12, "Contrast": 0.94, "Saturation": 0.92, "Gamma": 1.12,
        "Sharpen": 0, "Bloom": 0.08, "Vignette": 0, "Grain": 0,
        "Temperature": 0, "Chromatic": 0,
    },
    "Film": {
        "Brightness": 0, "Contrast": 1.04, "Saturation": 0.84, "Gamma": 1,
        "Sharpen": 0, "Bloom": 0.10, "Vignette": 0.22, "Grain": 0.022,
        "Temperature": -0.035, "Chromatic": 0,
    },
}


class RuntimeInstallError(RuntimeError):
    """A guarded install/uninstall operation could not safely continue."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_runtime_dll() -> Path | None:
    """Find the CI-built native DLL or the one bundled into the installer."""
    roots: list[Path] = []
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        roots.append(Path(bundled))
    executable_dir = Path(sys.executable).resolve().parent
    roots.append(executable_dir)
    source_root = Path(__file__).resolve().parent.parent
    roots.extend([
        source_root / "build" / "runtime" / "Release",
        source_root / "native" / "runtime" / "build" / "Release",
        source_root / "dist",
    ])
    seen: set[str] = set()
    for root in roots:
        key = os.path.normcase(str(root))
        if key in seen:
            continue
        seen.add(key)
        candidate = root / RUNTIME_DLL_NAME
        if candidate.is_file():
            return candidate.resolve()
    return None


def default_config_text() -> str:
    return DEFAULT_CONFIG


def preset_text(name: str, settings: dict[str, float]) -> str:
    lines = ["[TrSSPreset]", "Product=TrSS", "", "[Effects]"]
    for key, value in settings.items():
        lines.append(f"{key}={value:g}")
    lines.append("")
    return "\n".join(lines)


def install_runtime(game_executable: str | Path, runtime_dll: str | Path) -> dict[str, Any]:
    """Install the compiled shim without overwriting existing graphics layers."""
    exe = Path(game_executable).expanduser().resolve()
    dll = Path(runtime_dll).expanduser().resolve()
    if not IS_WINDOWS:
        raise RuntimeInstallError("The live runtime installer only supports Windows.")
    if exe.suffix.casefold() != ".exe" or not exe.is_file():
        raise RuntimeInstallError("Choose an existing game .exe file.")
    if not dll.is_file() or dll.name.casefold() != RUNTIME_DLL_NAME:
        raise RuntimeInstallError("The compiled first-party dxgi.dll runtime was not found.")

    game_dir = exe.parent
    runtime_target = game_dir / RUNTIME_DLL_NAME
    config_path = game_dir / CONFIG_NAME
    trss_dir = game_dir / TRSS_FOLDER
    preset_dir = trss_dir / PRESET_FOLDER
    manifest_path = trss_dir / MANIFEST_NAME

    # Never stomp an unrelated proxy or compatibility DLL. Backing it up and
    # replacing it could silently break another tool or a game's launch chain.
    if runtime_target.exists():
        raise RuntimeInstallError(
            f"{runtime_target.name} already exists beside this game. TrSS will not overwrite it. "
            "Remove or manage the existing graphics shim separately first."
        )
    if manifest_path.exists():
        raise RuntimeInstallError(
            "A TrSS installation manifest already exists but the runtime DLL is missing. "
            "Check the game folder before attempting another installation."
        )
    if config_path.exists():
        try:
            current_config = config_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise RuntimeInstallError(f"Cannot read existing TrSS.ini: {exc}") from exc
        if not (
            "[TrSSRuntime]" in current_config
            and "Product=TrSS" in current_config
        ):
            raise RuntimeInstallError(
                "A TrSS.ini already exists but is not marked as a TrSS runtime configuration. "
                "TrSS will leave it untouched."
            )

    created: list[Path] = []
    try:
        trss_dir.mkdir(parents=True, exist_ok=True)
        preset_dir.mkdir(parents=True, exist_ok=True)
        if not config_path.exists():
            config_path.write_text(default_config_text(), encoding="utf-8")
            created.append(config_path)

        for name, settings in PRESETS.items():
            path = preset_dir / f"{name}.ini"
            if not path.exists():
                path.write_text(preset_text(name, settings), encoding="utf-8")
                created.append(path)

        dll_hash = sha256_file(dll)
        staged = game_dir / f"{RUNTIME_DLL_NAME}.trss-installing"
        shutil.copy2(dll, staged)
        if sha256_file(staged) != dll_hash:
            staged.unlink(missing_ok=True)
            raise RuntimeInstallError("Copied runtime DLL checksum did not match the build.")
        os.replace(staged, runtime_target)
        created.append(runtime_target)

        manifest = {
            "product": "TrSS Runtime",
            "installer_version": APP_VERSION,
            "runtime_filename": RUNTIME_DLL_NAME,
            "runtime_sha256": dll_hash,
            "game_executable_name": exe.name,
            "installed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "graphics_api_target": "Direct3D 11 via DXGI",
            "network_upload": False,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        # On an interrupted install, remove only files made by this attempt.
        for path in reversed(created):
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                pass
        raise

    return {
        "game_executable_name": exe.name,
        "runtime_path": str(runtime_target),
        "config_path": str(config_path),
        "preset_directory": str(preset_dir),
        "runtime_sha256": dll_hash,
    }


def uninstall_runtime(game_executable: str | Path) -> dict[str, Any]:
    """Remove only the exact DLL whose checksum was recorded by TrSS."""
    exe = Path(game_executable).expanduser().resolve()
    if os.name != "nt":
        raise RuntimeInstallError("The live runtime installer only supports Windows.")
    if exe.suffix.casefold() != ".exe" or not exe.is_file():
        raise RuntimeInstallError("Choose the existing game .exe where TrSS was installed.")

    game_dir = exe.parent
    runtime_target = game_dir / RUNTIME_DLL_NAME
    manifest_path = game_dir / TRSS_FOLDER / MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeInstallError(
            "No TrSS installation manifest was found. TrSS will not delete an unverified dxgi.dll."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeInstallError(f"The installation manifest cannot be verified: {exc}") from exc
    if manifest.get("product") != "TrSS Runtime":
        raise RuntimeInstallError("The manifest does not identify a TrSS runtime installation.")

    if runtime_target.exists():
        expected = str(manifest.get("runtime_sha256", ""))
        actual = sha256_file(runtime_target)
        if not expected or actual.casefold() != expected.casefold():
            raise RuntimeInstallError(
                "The installed dxgi.dll differs from the TrSS build recorded in the manifest. "
                "It may have been replaced or modified, so it was left untouched."
            )
        runtime_target.unlink()
    manifest_path.unlink()
    # Preserve TrSS.ini and presets so a later reinstall keeps the user's work.
    return {
        "game_executable_name": exe.name,
        "runtime_removed": str(runtime_target),
        "configuration_preserved": str(game_dir / CONFIG_NAME),
        "presets_preserved": str(game_dir / TRSS_FOLDER / PRESET_FOLDER),
    }


class RuntimeInstallerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("780x560")
        self.root.minsize(720, 520)
        self.root.configure(bg="#0b0e12")
        self.game_path = tk.StringVar(master=root)
        self.scope_confirm = tk.BooleanVar(master=root, value=False)
        self.status = tk.StringVar(master=root, value="Select a game executable to begin.")
        self._build_ui()

    def _build_ui(self) -> None:
        outer = tk.Frame(self.root, bg="#0b0e12")
        outer.pack(fill="both", expand=True, padx=24, pady=22)
        tk.Label(
            outer, text="TrSS  /  RUNTIME INSTALLER", bg="#0b0e12", fg="#f1f4f8",
            font=("Segoe UI Semibold", 22), anchor="w",
        ).pack(fill="x")
        tk.Label(
            outer,
            text="Install the first-party TrSS live post-processing runtime beside one selected game.",
            bg="#0b0e12", fg="#9aa4b2", font=("Segoe UI", 10), anchor="w",
        ).pack(fill="x", pady=(4, 18))

        target = tk.LabelFrame(
            outer, text="GAME EXECUTABLE", bg="#0b0e12", fg="#dce2ea",
            font=("Segoe UI Semibold", 9), padx=12, pady=12,
        )
        target.pack(fill="x")
        row = tk.Frame(target, bg="#0b0e12")
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.game_path).pack(side="left", fill="x", expand=True, ipady=5)
        ttk.Button(row, text="Browse…", command=self._browse).pack(side="left", padx=(8, 0))

        warning = tk.Frame(outer, bg="#151a21", highlightbackground="#303946", highlightthickness=1)
        warning.pack(fill="x", pady=(16, 12))
        tk.Label(
            warning, text="CURRENT SUPPORT / EXPERIMENTAL", bg="#151a21", fg="#88c7ff",
            font=("Consolas", 10, "bold"), anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(
            warning,
            text=(
                "The initial runtime targets compatible Direct3D 11 games that create their swap chain through DXGI. "
                "It is not a universal D3D12, D3D9, Vulkan or OpenGL injector. A game may use an unsupported "
                "path or swap-chain format, and the overlay may not appear. The current menu applies original "
                "TrSS image post-processing; trained neural upscaling is not part of this runtime yet."
            ),
            bg="#151a21", fg="#c4ccd7", font=("Segoe UI", 9),
            anchor="w", justify="left", wraplength=690,
        ).pack(fill="x", padx=12, pady=(0, 10))

        consent = ttk.Checkbutton(
            outer,
            text=(
                "I will use this only where local graphics injection is permitted, not to bypass anti-cheat "
                "or protection in multiplayer games."
            ),
            variable=self.scope_confirm,
        )
        consent.pack(anchor="w", pady=(2, 12))

        actions = tk.Frame(outer, bg="#0b0e12")
        actions.pack(fill="x")
        ttk.Button(actions, text="Install runtime", command=self._install).pack(side="left")
        ttk.Button(actions, text="Uninstall runtime", command=self._uninstall).pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="Open game folder", command=self._open_folder).pack(side="left", padx=(8, 0))

        tk.Label(
            outer, textvariable=self.status, bg="#0b0e12", fg="#b8c3d0",
            font=("Consolas", 9), anchor="w", justify="left", wraplength=720,
        ).pack(fill="x", pady=(16, 0))

        tk.Label(
            outer,
            text="TrSS is independent, first-party code. Existing dxgi.dll files are never overwritten.",
            bg="#0b0e12", fg="#788595", font=("Segoe UI", 8), anchor="w",
        ).pack(fill="x", side="bottom", pady=(10, 0))

    def _browse(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.root,
            title="Select the actual game executable",
            filetypes=[("Windows executable", "*.exe"), ("All files", "*.*")],
        )
        if selected:
            self.game_path.set(selected)
            self.status.set(f"Selected: {Path(selected).name}")

    def _verify_scope(self) -> bool:
        if not self.scope_confirm.get():
            messagebox.showwarning(
                APP_NAME,
                "Confirm that you are allowed to install a local graphics runtime and will not use it to bypass anti-cheat.",
                parent=self.root,
            )
            return False
        return True

    def _install(self) -> None:
        if not self._verify_scope():
            return
        runtime = find_runtime_dll()
        if runtime is None:
            messagebox.showerror(
                APP_NAME,
                "The compiled TrSS dxgi.dll was not found.\n\nBuild native/runtime with CMake first or use the installer artifact built by CI.",
                parent=self.root,
            )
            return
        try:
            result = install_runtime(self.game_path.get(), runtime)
        except (OSError, RuntimeInstallError, ValueError) as exc:
            messagebox.showerror(APP_NAME, str(exc), parent=self.root)
            self.status.set("Installation stopped without overwriting an existing graphics DLL.")
            return
        self.status.set(f"Installed beside {result['game_executable_name']} · SHA-256 {result['runtime_sha256'][:16]}…")
        messagebox.showinfo(
            APP_NAME,
            "TrSS runtime installed.\n\n"
            f"Game: {result['game_executable_name']}\n"
            "Launch the game, then press Ctrl + Alt + Page Up to open the TrSS menu.\n\n"
            "If the menu does not appear, the game may not use the supported D3D11/DXGI path. "
            "Remove the runtime before troubleshooting the game.",
            parent=self.root,
        )

    def _uninstall(self) -> None:
        try:
            result = uninstall_runtime(self.game_path.get())
        except (OSError, RuntimeInstallError, ValueError) as exc:
            messagebox.showerror(APP_NAME, str(exc), parent=self.root)
            return
        self.status.set(f"Removed runtime from {result['game_executable_name']}; settings and presets were preserved.")
        messagebox.showinfo(
            APP_NAME,
            "TrSS's verified runtime DLL was removed. TrSS.ini and user presets were preserved.",
            parent=self.root,
        )

    def _open_folder(self) -> None:
        raw = self.game_path.get().strip()
        if not raw:
            messagebox.showinfo(APP_NAME, "Select a game executable first.", parent=self.root)
            return
        game_dir = Path(raw).expanduser().resolve().parent
        if not game_dir.is_dir():
            messagebox.showerror(APP_NAME, "The selected game folder does not exist.", parent=self.root)
            return
        try:
            os.startfile(str(game_dir))  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror(APP_NAME, str(exc), parent=self.root)


def main() -> None:
    if os.name != "nt":
        raise SystemExit("TrSS Runtime Installer requires Windows.")
    root = tk.Tk()
    RuntimeInstallerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
