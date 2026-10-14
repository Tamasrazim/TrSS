# TrSS Live Runtime (first-party D3D11 prototype)

TrSS now has a **first-party, app-local DXGI runtime prototype**. It is inspired by the workflow of a per-game graphics injector: select a game's actual executable in the TrSS Runtime Installer, install TrSS beside that executable, launch the game normally, then press **Ctrl + Alt + Page Up** to open the in-game menu.

This is not ReShade code or a wrapper around another graphics tool. The runtime, shader, settings menu, presets, and installer are implemented in TrSS. It does not include or download third-party injector binaries.

## Current support boundary

The first native target is **compatible Direct3D 11 games that create a DXGI swap chain through the ordinary DXGI factory path**, using a single-sample 8-bit BGRA/RGBA back buffer. This is a narrow starting point, not universal game support. D3D9, D3D12, Vulkan, OpenGL, unusual/custom swap-chain formats, some fullscreen modes, and games whose graphics path bypasses the intercepted DXGI entry points are not supported by this build. A game may fail to show the menu even if it launches normally.

The current shader applies first-party **image post-processing** to the game's presented image. It is not yet using the PyTorch model for live inference. The following features are separate milestones: native D3D12/Vulkan integration, model checkpoint loading in the in-game renderer, real depth and motion-vector acquisition, temporal reprojection, more effects, and physical testing across game/API combinations. CI proves that the current C++ code compiles and the installer tests pass; it does not prove that a real game supports the hook or that the runtime is safe or visually correct on every driver.

## Build and install

The **Build first-party TrSS live runtime** workflow builds both the native dxgi.dll and a one-file Windows installer. Download the artifact named **TrSS-Live-Runtime-windows-x64** from the workflow run.

1. Launch TrSS-Runtime-Installer.exe.
2. Choose the actual game executable, not a launcher, shortcut, or setup program.
3. Confirm that local graphics injection is allowed for that game and that you are not trying to bypass anti-cheat or game protections.
4. Click **Install runtime**.
5. Start the game normally. On a supported D3D11/DXGI path, press **Ctrl + Alt + Page Up** to toggle the TrSS menu.

The installer places dxgi.dll beside the chosen executable, creates TrSS.ini, provides starter preset files, and writes a SHA-256 installation manifest. It does **not** overwrite an existing dxgi.dll, because that file may belong to another graphics layer, mod, compatibility shim, or the game itself. Resolve that conflict separately before installing TrSS; the installer will not silently chain-load or replace another tool.

Use **Uninstall runtime** to remove TrSS. It deletes only the dxgi.dll whose checksum matches the TrSS installation manifest, then preserves TrSS.ini and all presets. If the DLL has changed since installation, the uninstaller refuses to delete it automatically.

## In-game menu and controls

Press **Ctrl + Alt + Page Up** to show or hide the menu. The initial menu exposes these live controls:

| Control | Effect |
| --- | --- |
| Brightness | Lift or lower overall image brightness |
| Contrast | Change contrast around the midpoint |
| Saturation | Reduce or intensify color |
| Gamma | Adjust the output curve |
| Sharpen | Add a local high-frequency detail pass |
| Bloom | Add a basic highlight glow |
| Vignette | Darken image edges |
| Film grain | Add subtle screen-space noise |
| Color temperature | Adjust the red/blue balance |
| Chromatic aberration | Apply a small channel offset |

These are TrSS post-processing controls; they do not change the game's internal texture, shadow, ray-tracing or upscaling menu. The current effects are simple real-time shader operations, not a trained neural reconstruction model. Heavy effects and incorrect settings can reduce image quality, so start with **Neutral**.

Built-in looks are **Neutral**, **Cinematic**, **Vibrant**, **Clarity**, **Night**, and **Film**. **Save preset** writes the current values to an INI file; **Load preset** imports a TrSS preset file. The runtime reads settings rather than executing preset files as code. Presets are stored under TrSS/Presets beside the game.

If the game starts but the overlay never appears, return to the installer and remove TrSS before troubleshooting. Do not keep an unknown proxy DLL installed in a game folder as an experiment.

## Keep the recorder independent

The **TrSS Game Library recorder is separate from the live renderer**. It does not have to communicate with, load, or instrument the TrSS runtime. Use the recorder in two controlled passes:

1. **Baseline:** with TrSS runtime not installed for that game, launch the game from the library and record a run. In the post-capture review, set the profile label to Baseline.
2. **TrSS:** install the runtime, launch the same game and repeat the same scene and workload with TrSS active. In the review, use a specific profile label such as TrSS - Cinematic (rather than the generic TrSS label) when comparing a named preset.
3. Use the **same experiment group**, and verify the same game's graphics settings in both review dialogs. Use identical game build, resolution, scene, camera position, game settings, API and driver whenever possible. Each exact row is fingerprinted; comparison refuses to pair rows whose standard settings hashes differ.
4. Repeat both modes several times and alternate order when practical to reduce temperature, cache and run-order bias.

The recorder captures screenshots and available Windows counters such as CPU, memory, and per-process GPU utilization/VRAM. FPS/frametime remain unavailable until a real first-party per-present source is implemented; screenshot capture rate is never substituted for game FPS.

After both modes have been recorded, compare the local JSONL matrix:

~~~powershell
trss-compare-performance --matrix "$env:LOCALAPPDATA\TrSS\GameLibrary\matrix\game-matrix.jsonl" --output-dir "logs\performance-comparison" --baseline-profile "Baseline" --trss-profile "TrSS" --require-exact
~~~

The command writes performance-comparison.csv and performance-comparison.json. It only compares rows that share the same game, experiment group and settings fingerprint; --require-exact additionally requires both profiles to be marked exact and verified in-game. A percentage change is a descriptive observation, not proof that TrSS caused the change. A comparable workload, repeated runs, and real metrics matter.

## How this supports future PyTorch work

The recorder's performance comparison is **evaluation evidence**, not a direct supervised-learning target. It can identify the performance cost of a shader/model configuration and help prioritize later optimization. It cannot teach a reconstruction network how pixels should look by itself.

Training the first-party PyTorch reconstruction model still requires aligned low-resolution inputs and high-resolution reference frames, real depth/motion information, and correctly reprojected temporal history where required. The current runtime does not yet provide those renderer-native buffers, and the repo does not ship trained weights. The model's offline performance telemetry is also separate from the in-game recorder. Join those datasets only through explicit, versioned experiment metadata; do not fabricate missing frames, settings, or GPU measurements.

## Local files and privacy

- dxgi.dll: the first-party native runtime installed beside the game.
- TrSS.ini: global effects on/off and current numeric setting values for this installation.
- TrSS/Presets/*.ini: built-in or user-saved presets.
- TrSS/install-manifest.json: runtime checksum and install metadata used for safe removal.
- Game Library logs and performance matrix: remain in the local Windows profile unless you choose to share them.

TrSS does not upload game frames, configuration files, or performance logs. Do not record or share screen content you do not have permission to capture. This experimental hook may be incompatible with protected online games; it does not attempt to bypass protections.
