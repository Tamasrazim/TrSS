"""First-party TrSS DXGI runtime installer.

Install is explicit per-game and x64-only. Existing game-local dxgi.dll files are
never overwritten because they may belong to another mod or compatibility layer.
Uninstall verifies the file hashes before removing anything installed by TrSS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

RUNTIME_NAME = "dxgi.dll"
FORWARDER_NAME = "trss_original_dxgi.dll"
MANIFEST_NAME = "TrSS.runtime-install.json"
SETTINGS_NAME = "TrSS.ini"
SCHEMA_VERSION = "trss-runtime-install/1.0"


class InstallError(RuntimeError):
    """A safe install/uninstall cannot proceed without risking another file."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pe_architecture(path: str | Path) -> str:
    """Read PE headers only; do not execute an executable or dynamic library."""
    file_path = Path(path)
    try:
        with file_path.open("rb") as stream:
            header = stream.read(64)
            if len(header) < 64 or header[:2] != b"MZ":
                raise InstallError(f"{file_path.name} is not a Windows PE file.")
            pe_offset = struct.unpack_from("<I", header, 0x3C)[0]
            if pe_offset > 64 * 1024 * 1024:
                raise InstallError(f"{file_path.name} has an invalid PE header offset.")
            stream.seek(pe_offset)
            prefix = stream.read(26)
    except OSError as exc:
        raise InstallError(f"Cannot read {file_path}: {exc}") from exc
    if len(prefix) < 26 or prefix[:4] != b"PE\0\0":
        raise InstallError(f"{file_path.name} has an invalid PE signature.")
    machine = struct.unpack_from("<H", prefix, 4)[0]
    optional_magic = struct.unpack_from("<H", prefix, 24)[0]
    if machine == 0x8664 and optional_magic == 0x20B:
        return "x64"
    if machine == 0x14C and optional_magic == 0x10B:
        return "x86"
    if machine == 0xAA64 and optional_magic == 0x20B:
        return "arm64"
    return f"unsupported (machine=0x{machine:04X}, optional=0x{optional_magic:04X})"


def _system_dxgi_default() -> Path:
    windows = Path(os.environ.get("WINDIR", r"C:\Windows"))
    # A 32-bit Python process on 64-bit Windows sees Sysnative as the real
    # 64-bit system directory. The packaged manager is normally x64, but source
    # runs should still pick the correct OS DXGI file.
    if struct.calcsize("P") == 4 and os.environ.get("PROCESSOR_ARCHITEW6432"):
        return windows / "Sysnative" / "dxgi.dll"
    return windows / "System32" / "dxgi.dll"


def find_runtime_dll() -> Path | None:
    """Find the build artifact beside the manager or in a local source build."""
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).resolve().parent / RUNTIME_NAME)
    own_dir = Path(__file__).resolve().parent
    candidates.extend([
        own_dir / RUNTIME_NAME,
        own_dir.parent / "runtime" / "Release" / RUNTIME_NAME,
        own_dir.parent / "build" / "runtime" / "Release" / RUNTIME_NAME,
        own_dir.parent / "native" / "runtime" / "build" / "Release" / RUNTIME_NAME,
    ])
    for item in candidates:
        if item.is_file():
            return item.resolve()
    return None


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallError(f"The installation marker is unreadable: {path.name}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise InstallError(
            f"{path.name} exists but is not a valid TrSS runtime marker. "
            "TrSS will not overwrite or remove files based on an unknown marker."
        )
    return data


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.trss-",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        shutil.copystat(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_manifest(destination: Path, data: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.trss-",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _default_settings_text() -> str:
    return (
        "[TrSSRuntime]\n"
        "Product=TrSS\n"
        "Version=0.1.0\n"
        "ActivePreset=Neutral\n"
        "\n"
        "[Effects]\n"
        "Enabled=1\n"
        "Brightness=0.000000\n"
        "Contrast=1.000000\n"
        "Saturation=1.000000\n"
        "Gamma=1.000000\n"
        "Sharpen=0.000000\n"
        "Bloom=0.000000\n"
        "Vignette=0.000000\n"
        "Grain=0.000000\n"
        "Temperature=0.000000\n"
        "Chromatic=0.000000\n"
    )


def install_runtime(
    game_executable: str | Path,
    runtime_dll: str | Path | None = None,
    *,
    system_dxgi: str | Path | None = None,
) -> dict[str, Any]:
    """Install into one game directory without replacing an unrelated local DXGI shim."""
    game_exe = Path(game_executable).expanduser().resolve()
    if not game_exe.is_file() or game_exe.suffix.casefold() != ".exe":
        raise InstallError("Choose an existing game .exe file.")
    if pe_architecture(game_exe) != "x64":
        raise InstallError(
            "This first runtime build is x64-only. This game executable is not a compatible x64 PE."
        )

    runtime_source = Path(runtime_dll).expanduser().resolve() if runtime_dll else find_runtime_dll()
    if runtime_source is None or not runtime_source.is_file():
        raise InstallError(
            "Cannot find the built TrSS dxgi.dll. Select it from the TrSS Runtime build package."
        )
    if runtime_source.name.casefold() != RUNTIME_NAME:
        raise InstallError(f"The runtime file must be named {RUNTIME_NAME}.")
    if pe_architecture(runtime_source) != "x64":
        raise InstallError("The TrSS runtime DLL is not an x64 Windows library.")

    original_source = Path(system_dxgi).expanduser().resolve() if system_dxgi else _system_dxgi_default()
    if not original_source.is_file():
        raise InstallError(f"Windows' original DXGI library was not found: {original_source}")
    if pe_architecture(original_source) != "x64":
        raise InstallError("The system DXGI file is not x64; refusing to install the x64 runtime.")

    game_dir = game_exe.parent
    runtime_target = game_dir / RUNTIME_NAME
    forwarder_target = game_dir / FORWARDER_NAME
    manifest_target = game_dir / MANIFEST_NAME
    settings_target = game_dir / SETTINGS_NAME

    manifest = _read_manifest(manifest_target)
    upgrading = manifest is not None
    if upgrading:
        if Path(str(manifest.get("game_executable", ""))).resolve() != game_exe:
            raise InstallError("The existing TrSS marker belongs to a different game executable.")
        if not runtime_target.is_file() or sha256_file(runtime_target) != manifest.get("runtime_sha256"):
            raise InstallError(
                "The installed dxgi.dll was changed after TrSS installed it. "
                "TrSS will not overwrite the modified file."
            )
        if not forwarder_target.is_file() or sha256_file(forwarder_target) != manifest.get("forwarder_sha256"):
            raise InstallError("The saved Windows DXGI forwarder was changed; refusing an in-place update.")
    else:
        if runtime_target.exists():
            raise InstallError(
                f"{runtime_target} already exists. It may belong to the game or another mod. "
                "TrSS will not overwrite or chain an unreviewed DXGI proxy."
            )
        if forwarder_target.exists():
            raise InstallError(
                f"{forwarder_target} already exists but has no TrSS marker. Rename or inspect it manually first."
            )

    runtime_hash = sha256_file(runtime_source)
    source_original_hash = sha256_file(original_source)
    previous_manifest = manifest
    runtime_installed = False
    forwarder_created = False
    runtime_backup: Path | None = None

    try:
        if not upgrading:
            _atomic_copy(original_source, forwarder_target)
            forwarder_created = True
        else:
            # Preserve the current runtime so an interrupted upgrade can roll back.
            fd, backup_name = tempfile.mkstemp(
                prefix=".dxgi.trss-backup-", dir=str(game_dir)
            )
            os.close(fd)
            runtime_backup = Path(backup_name)
            shutil.copy2(runtime_target, runtime_backup)

        _atomic_copy(runtime_source, runtime_target)
        runtime_installed = True

        if not settings_target.exists():
            settings_target.write_text(_default_settings_text(), encoding="utf-8")

        new_manifest = {
            "schema_version": SCHEMA_VERSION,
            "game_executable": str(game_exe),
            "installed_at_utc": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "runtime_file": RUNTIME_NAME,
            "runtime_sha256": runtime_hash,
            "forwarder_file": FORWARDER_NAME,
            "forwarder_sha256": sha256_file(forwarder_target),
            "windows_dxgi_source": str(original_source),
            "windows_dxgi_sha256": source_original_hash,
            "architecture": "x64",
            "rendering_backend": "DXGI / Direct3D 11 initial target",
            "settings_file": SETTINGS_NAME,
            "settings_created_by_installer": not settings_target.exists(),
        }
        # Preserve whether TrSS created the settings file before writing the marker.
        new_manifest["settings_created_by_installer"] = not (
            upgrading and (previous_manifest or {}).get("settings_created_by_installer") is False
        )
        _write_manifest(manifest_target, new_manifest)
    except Exception as exc:
        if runtime_backup is not None and runtime_backup.exists():
            try:
                os.replace(runtime_backup, runtime_target)
                runtime_backup = None
            except OSError:
                pass
        elif runtime_installed:
            runtime_target.unlink(missing_ok=True)
        if forwarder_created:
            forwarder_target.unlink(missing_ok=True)
        if isinstance(exc, InstallError):
            raise
        raise InstallError(
            f"Install failed ({type(exc).__name__}). Check game-folder permissions and close the game before installing."
        ) from exc
    finally:
        if runtime_backup is not None:
            runtime_backup.unlink(missing_ok=True)

    return {
        "game_executable": str(game_exe),
        "runtime_dll": str(runtime_target),
        "forwarder_dll": str(forwarder_target),
        "settings_file": str(settings_target),
        "upgraded": upgrading,
        "runtime_sha256": runtime_hash,
    }


def uninstall_runtime(game_executable: str | Path) -> dict[str, Any]:
    """Remove only files whose hashes prove they still match TrSS's own install marker."""
    game_exe = Path(game_executable).expanduser().resolve()
    if not game_exe.is_file() or game_exe.suffix.casefold() != ".exe":
        raise InstallError("Choose the installed game .exe file.")
    game_dir = game_exe.parent
    manifest_target = game_dir / MANIFEST_NAME
    manifest = _read_manifest(manifest_target)
    if manifest is None:
        raise InstallError("No TrSS runtime installation marker was found beside this game.")
    if Path(str(manifest.get("game_executable", ""))).resolve() != game_exe:
        raise InstallError("The TrSS marker belongs to a different game executable.")

    runtime_target = game_dir / str(manifest.get("runtime_file", RUNTIME_NAME))
    forwarder_target = game_dir / str(manifest.get("forwarder_file", FORWARDER_NAME))
    if not runtime_target.is_file() or sha256_file(runtime_target) != manifest.get("runtime_sha256"):
        raise InstallError("dxgi.dll no longer matches TrSS's installed hash; nothing was removed.")
    if not forwarder_target.is_file() or sha256_file(forwarder_target) != manifest.get("forwarder_sha256"):
        raise InstallError("The original DXGI forwarder no longer matches its install hash; nothing was removed.")

    runtime_target.unlink()
    forwarder_target.unlink()
    manifest_target.unlink()
    # User settings and preset files deliberately remain so removal never erases
    # settings the user may want to keep or export.
    return {
        "game_executable": str(game_exe),
        "removed": [str(runtime_target), str(forwarder_target), str(manifest_target)],
        "kept": [str(game_dir / SETTINGS_NAME), str(game_dir / "TrSS" / "Presets")],
    }


class RuntimeManagerUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TrSS Runtime Manager")
        self.root.geometry("760x390")
        self.root.minsize(690, 360)
        self.root.configure(bg="#0b0e12")
        self.game_var = tk.StringVar()
        detected = find_runtime_dll()
        self.runtime_var = tk.StringVar(value=str(detected) if detected else "")
        self.status_var = tk.StringVar(value="Choose a 64-bit game executable to install or remove TrSS.")
        self._build()

    def _build(self) -> None:
        bg, fg, muted = "#0b0e12", "#edf0f4", "#9aa4b2"
        frame = tk.Frame(self.root, bg=bg)
        frame.pack(fill="both", expand=True, padx=22, pady=20)
        tk.Label(frame, text="TrSS / RUNTIME MANAGER", bg=bg, fg=fg,
                 font=("Segoe UI Semibold", 20)).pack(anchor="w")
        tk.Label(
            frame,
            text="Install the first-party D3D11 post-processing runtime into one game folder.",
            bg=bg, fg=muted, font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(4, 18))

        tk.Label(frame, text="GAME EXECUTABLE", bg=bg, fg=muted,
                 font=("Consolas", 9)).pack(anchor="w")
        game_row = tk.Frame(frame, bg=bg)
        game_row.pack(fill="x", pady=(4, 12))
        ttk.Entry(game_row, textvariable=self.game_var).pack(side="left", fill="x", expand=True)
        ttk.Button(game_row, text="Browse…", command=self._choose_game).pack(side="left", padx=(8, 0))

        tk.Label(frame, text="TRSS RUNTIME DLL (x64)", bg=bg, fg=muted,
                 font=("Consolas", 9)).pack(anchor="w")
        runtime_row = tk.Frame(frame, bg=bg)
        runtime_row.pack(fill="x", pady=(4, 12))
        ttk.Entry(runtime_row, textvariable=self.runtime_var).pack(side="left", fill="x", expand=True)
        ttk.Button(runtime_row, text="Browse…", command=self._choose_runtime).pack(side="left", padx=(8, 0))

        tk.Label(
            frame,
            text=(
                "Ctrl + Alt + Page Up opens the in-game menu. The first build targets compatible "
                "x64 D3D11/DXGI games only. Do not install in anti-cheat-protected multiplayer sessions. "
                "TrSS does not bypass anti-cheat protections. Existing dxgi.dll files are never overwritten."
            ),
            bg=bg, fg=muted, font=("Segoe UI", 9), justify="left", wraplength=700,
        ).pack(anchor="w", pady=(0, 12))

        buttons = tk.Frame(frame, bg=bg)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Install / Update", command=self._install).pack(side="left")
        ttk.Button(buttons, text="Uninstall TrSS", command=self._uninstall).pack(side="left", padx=8)
        ttk.Button(buttons, text="Open Selected Game Folder", command=self._open_folder).pack(side="left")
        tk.Label(frame, textvariable=self.status_var, bg=bg, fg="#c8d5e4",
                 font=("Segoe UI", 9), justify="left", wraplength=700).pack(anchor="w", pady=(12, 0))

    def _choose_game(self) -> None:
        chosen = filedialog.askopenfilename(
            parent=self.root, title="Choose a game executable",
            filetypes=[("Windows game executable", "*.exe"), ("All files", "*.*")],
        )
        if chosen:
            self.game_var.set(chosen)
            self.status_var.set(f"Selected: {chosen}")

    def _choose_runtime(self) -> None:
        chosen = filedialog.askopenfilename(
            parent=self.root, title="Choose the compiled TrSS runtime DLL",
            filetypes=[("TrSS runtime", "dxgi.dll"), ("DLL files", "*.dll"), ("All files", "*.*")],
        )
        if chosen:
            self.runtime_var.set(chosen)

    def _install(self) -> None:
        game = self.game_var.get().strip()
        if not game:
            messagebox.showwarning("TrSS Runtime", "Choose a game executable first.", parent=self.root)
            return
        approved = messagebox.askokcancel(
            "Install TrSS into this game?",
            "TrSS will add its own dxgi.dll proxy beside the selected executable and copy the "
            "current Windows DXGI library under a private forwarder name. It will not overwrite an "
            "existing game-local dxgi.dll. This can be incompatible with some games and anti-cheat "
            "systems. Close the game before proceeding. Continue only for a game you are authorized to modify.",
            icon="warning", parent=self.root,
        )
        if not approved:
            return
        try:
            result = install_runtime(game, self.runtime_var.get().strip() or None)
        except Exception as exc:
            self.status_var.set(str(exc))
            messagebox.showerror("TrSS Runtime", str(exc), parent=self.root)
            return
        action = "updated" if result["upgraded"] else "installed"
        self.status_var.set(f"TrSS runtime {action} for {result['game_executable']}.")
        messagebox.showinfo(
            "TrSS Runtime",
            f"TrSS runtime {action}.\n\nGame: {result['game_executable']}\n"
            f"Runtime: {result['runtime_dll']}\n\nOpen the game, then press Ctrl + Alt + Page Up.",
            parent=self.root,
        )

    def _uninstall(self) -> None:
        game = self.game_var.get().strip()
        if not game:
            messagebox.showwarning("TrSS Runtime", "Choose the installed game's .exe first.", parent=self.root)
            return
        try:
            result = uninstall_runtime(game)
        except Exception as exc:
            messagebox.showerror("TrSS Runtime", str(exc), parent=self.root)
            return
        self.status_var.set(f"TrSS runtime removed for {result['game_executable']}. Settings were kept.")
        messagebox.showinfo(
            "TrSS Runtime",
            "TrSS runtime and its copied Windows DXGI forwarder were removed.\n"
            f"Preserved user settings:\n{result['kept'][0]}\n{result['kept'][1]}",
            parent=self.root,
        )

    def _open_folder(self) -> None:
        raw = self.game_var.get().strip()
        if not raw:
            return
        path = Path(raw).expanduser()
        folder = path.parent if path.suffix.casefold() == ".exe" else path
        if not folder.is_dir():
            messagebox.showerror("TrSS Runtime", "The selected game folder does not exist.", parent=self.root)
            return
        try:
            os.startfile(str(folder))
        except (OSError, AttributeError) as exc:
            messagebox.showerror("TrSS Runtime", str(exc), parent=self.root)


def main() -> int:
    parser = argparse.ArgumentParser(description="Install or remove the first-party TrSS game runtime.")
    parser.add_argument("--install", type=Path, help="install next to this game executable")
    parser.add_argument("--uninstall", type=Path, help="remove TrSS from this game executable's folder")
    parser.add_argument("--runtime-dll", type=Path, help="built TrSS x64 dxgi.dll")
    args = parser.parse_args()
    try:
        if args.install:
            result = install_runtime(args.install, args.runtime_dll)
            print(json.dumps(result, indent=2))
            return 0
        if args.uninstall:
            result = uninstall_runtime(args.uninstall)
            print(json.dumps(result, indent=2))
            return 0
        root = tk.Tk()
        RuntimeManagerUI(root)
        root.mainloop()
        return 0
    except Exception as exc:
        if getattr(sys, "frozen", False):
            try:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("TrSS Runtime", str(exc), parent=root)
                root.destroy()
            except tk.TclError:
                pass
        print(f"TrSS runtime manager: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
