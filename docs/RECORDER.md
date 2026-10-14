# TrSS Game Library

TrSS Game Library is a Windows game launcher and opt-in data collection tool for TrSS research. It scans known game stores and common install folders, displays one card per detected executable using the icon embedded in that executable, and launches games when the card is clicked.

## Library discovery

On launch, the library looks for installed Steam games using Steam's library manifests, Epic Games installs using Epic's local item manifests, game-like uninstall registry entries, and common game directories. Use **Add folder** to scan a selected library directory. **Scan all drives** performs an optional deeper scan of fixed drives; it can be slow on large disks and can be cancelled by closing the app. The index and extracted icons are cached under the local Windows profile. Nothing is uploaded.

The launcher cannot identify every game perfectly. Some store-managed, encrypted, Microsoft Store, anti-cheat, or launcher-dependent titles may require manual folder selection or may not start correctly from their extracted executable. Steam and Epic manifest entries try to use the respective store's launch URI where available.

## Live on-screen telemetry

When a game is launched, the library minimizes and shows a small always-on-top, click-through overlay. It displays the metrics available from Windows on that machine:

- System CPU use and the selected game process CPU use.
- System memory use and the selected game's working-set memory.
- Per-game displayed FPS, application FPS, average frametime and a rolling 1% low estimate from per-present Windows ETW frame events, when those events are available.
- Per-process GPU 3D utilization and dedicated GPU memory, when the Windows GPU performance counters expose them.

The portable build includes the pinned PresentMon 2.6.0 console telemetry runtime as a hidden background component; participants do not need to install or open a separate performance-monitoring app. Its upstream MIT notice is included with the release. If Windows denies ETW trace access, the FPS line shows the reason and remains unavailable instead of estimating values from the monitor refresh rate, screenshot rate, or GPU load. Windows Performance Log permission may be required. Some exclusive-fullscreen games may also prevent an ordinary desktop overlay from appearing.

## Start an unlimited-duration capture

1. Run TrSS Game Library and scan your stores or game folders.
2. Check both permission and research-collection confirmations at the bottom of the window. Only collect content you are allowed to capture and share.
3. Click a game card to launch that title. Wait until the game's own window is open and focused.
4. Use **Ctrl + Alt + F8** to start collection. The selected capture rate (1, 2, 5, or 10 screenshots per second) applies to lossless PNG screenshots; performance metrics are sampled about once per second. There is no timer or fixed frame-count limit. The only automatic stops are focus loss, a capture error, or very low free disk space.
5. Use **Ctrl + Alt + F8** to stop and package the session, or **Ctrl + Alt + F10** to stop immediately. Use **Ctrl + Alt + F9** to bring the library back. Keep the game focused while collecting; losing focus stops the screenshot session so other windows are not recorded.
6. Inspect the ZIP and README before sharing it with the project maintainer. The collector does not upload the data.

## What a capture bundle contains

| File | Meaning |
| --- | --- |
| metadata.json | Game label, duration, requested capture rate, count, stop reason, consent flags and privacy facts. |
| system.json | Operating-system, CPU/GPU driver labels, and installed memory when available. |
| sequence.jsonl | Frame index, measured elapsed timestamp, session ID and relative PNG path. |
| frames/frame-XXXXXXXX.png | Lossless RGB screenshots from the active game window. |
| performance.jsonl | Timestamped CPU, game-memory, system-memory, available per-process GPU counters, per-game FPS, application FPS, frametime and 1% low. Values stay null when the metric is unavailable. |
| SHA256SUMS.txt | SHA-256 hashes of all other files in the bundle. |
| README.txt | Human-readable contents, privacy details and limitations. |

The recorder does not save the absolute game installation path or raw window title in the bundle. It never records audio, microphone input, clipboard contents, hostname, account name, IP address or device serial numbers. The game window itself can still display private messages, account details or third-party content, so preview the title and screen before collecting and sharing.

## Important model-training limitation

These bundles do not contain real depth buffers, motion vectors, reprojected temporal history or matched high-resolution reference renders. A screen capture and an FPS log are not a paired neural-rendering dataset. TrSS must not fabricate missing training buffers or label a simple screenshot as ground truth.

## Install from source

Requires Windows 10/11, Python 3.10+ with Tkinter enabled, and a writable local data folder:

~~~powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
.\scripts\fetch_presentmon.ps1
trss-recorder
~~~

The first-time source setup downloads the official PresentMon 2.6.0 Windows x64 collector and checks its pinned SHA-256 digest. That small open-source component is bundled into the app build and runs hidden only for per-frame telemetry. No telemetry is sent to PresentMon's maintainers or to TrSS servers.

To run the old single-window capture utility directly, use **trss-window-recorder**. The new game library is the recommended participant interface.

## Portable Windows build

The repository workflow builds a single-file Windows executable and a SHA-256 checksum. Open the GitHub Actions page and select **Build TrSS Game Library** to download the build artifact. A version tag named **game-library-v...** publishes the executable, checksum and PresentMon notice as a GitHub Release. The build is generated in CI; it still needs real-machine validation across different Windows versions, graphics drivers, game launchers and fullscreen modes.
