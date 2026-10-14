# TrSS Game Library

TrSS Game Library is a Windows game launcher and local, opt-in research recorder for TrSS. It discovers installed games, shows one card per executable with the executable's icon, launches games, displays available Windows performance counters, captures game-window samples, and builds a structured graphics-settings matrix across recording sessions.

## Game discovery

On launch, the library checks Steam manifests, Epic local manifests, game-like uninstall registry entries, and common installation directories. Use **Add folder** to index a selected game directory. **Scan all drives** optionally performs a deeper fixed-drive scan, which can be slow and can be cancelled. The game index and extracted icons stay in the local Windows profile.

Some store-managed, encrypted, Microsoft Store, anti-cheat, or launcher-dependent titles may need manual folder selection or may not start through their extracted executable. Store manifest entries use a store launch URI where one is available.

## Live monitoring: what is and is not measured

The click-through overlay displays measurements the local Windows system exposes: system CPU, the selected process's CPU and working-set memory, system memory, and process GPU utilization / dedicated GPU memory when Windows performance counters expose those counters.

**Native TrSS game FPS, application FPS, frametime, and 1% lows are not implemented yet.** They remain null/unavailable; TrSS does not infer them from screenshot rate, monitor refresh, or GPU load. The recorder also does not capture renderer-native depth, motion vectors, temporal history, or paired reference renders. A screenshot is not a substitute for those buffers.

## Capture a run

1. Start the library, select the installed game, and launch it from its game card.
2. Confirm both consent checkboxes. Only collect screenshots and telemetry you are allowed to capture and share.
3. Focus the game's own window and press **Ctrl + Alt + F8**. The selected rate (1, 2, 5, or 10 screenshots per second) controls lossless PNG sampling; available performance counters are sampled roughly once a second. Duration and frame count are not fixed.
4. Press **Ctrl + Alt + F8** again to stop and review the run, or **Ctrl + Alt + F10** to stop early. **Ctrl + Alt + F9** restores the library.
5. When the recording stops, TrSS restores the library and opens **Graphics Profile / Post-Capture Review**. Check the detected values against the game's own Graphics / Display menu, fill any missing values, add game-specific settings, select the in-game verification checkbox only when you actually checked the menu, and save the profile. The capture is not packaged until this review is saved, so the settings and image data stay attached to the same session.

The collector stops screenshot sampling when the game loses focus, when capture errors occur, or when it must stop to preserve enough free disk space to create a ZIP safely. It does not run in the background after the participant stops it, and it does not upload files.

## Graphics settings: detection and verification

After each run, TrSS searches a bounded set of conventional configuration filenames in the selected installation folder and title-matching user-profile folders. It reads only allowlisted graphics-related keys into the standardized settings model. It never copies an entire game configuration file or saves the full path of an imported file. You can also use **Import config…** in the review dialog to read recognized keys from an explicitly selected XML, INI, CFG, JSON, TOML, or text settings file.

This is deliberately conservative. Games have different settings menus and config formats; some settings are held in memory while the game runs, config files may be stale, and numeric config values may be engine-specific enum codes. Automatically reading arbitrary runtime settings from every game is not a universal Windows capability. TrSS labels file-derived values by source and leaves them for review rather than silently mapping an unexplained number to “Ultra” or “High”. An imported file is not itself proof of the current in-game configuration.

The review has standardized fields for game build, graphics API, display mode and resolution, refresh rate, frame limit, V-Sync, HDR, upscaling technology and mode, frame generation, anti-aliasing, render scale, ray/path tracing, lighting, textures, geometry/LOD, shadows, reflections, effects, post-processing, vegetation, water, FOV, and other common settings. Use **Additional game-specific settings** to record options that this common schema does not cover, one exact `Setting name = Value` per line.

Use explicit values. A blank is missing. **Unknown**, **Not captured**, and **Not verified** do not count as confirmed values. Use **Not supported** or **Not applicable** only when that is true for the game. Numeric values in enum-like config fields remain unresolved until you replace them with the exact value shown by the game. The matrix marks a row **exact** only when the participant confirms it against the game's menu, every standard setting has an explicit unambiguous value or applicability state, and no numeric enum remains unresolved.

## Build a controlled game matrix

One recording produces one matrix row for one settings profile. For a sweep:

1. Choose an experiment group name and keep it identical across the runs that belong together.
2. Record the baseline using a consistent game scene, camera position, resolution context, and workload.
3. Change the setting or setting combination inside the game's own menu. If the game requires a restart, restart it before the next run.
4. Record again and describe the new profile. Repeat until every requested profile has its own capture row.

TrSS does **not** change game settings for you, write into game config files, or claim to have run every possible combination automatically. The recorder captures the current run and asks you to label it; consistent test conditions and actual menu confirmation are what make the matrix meaningful. For a controlled performance comparison, change one variable per run whenever practical, and write down unavoidable changes (scene, API, driver, game build, resolution, or other settings) in the profile or experiment notes.

## What is saved

Each session ZIP contains:

| File | Contents |
| --- | --- |
| `metadata.json` | Session details, consent/privacy flags, graphics-matrix coverage, verification state, and limitations. |
| `system.json` | Available OS, CPU, GPU-driver labels, and installed-memory data. |
| `sequence.jsonl` | Frame order, measured elapsed timestamp, session ID, and relative PNG path. |
| `frames/frame-XXXXXXXX.png` | Lossless RGB screenshots from the focused game window. |
| `performance.jsonl` | Timestamped CPU, memory, and available per-process GPU measurements; FPS/frametime remain null until first-party TrSS frame telemetry exists. |
| `graphics_settings.json` | Standardized setting values, per-value source labels, imported/detected source filenames, the participant's verification state, and optional game-specific settings. |
| `game_matrix_row.json` | One normalized, hash-addressed matrix row with completeness flags and available metric summaries. |
| `SHA256SUMS.txt` | SHA-256 hashes of the packaged files. |
| `README.txt` | Human-readable bundle contents and privacy limitations. |

The shared local matrix is written to `%LOCALAPPDATA%\TrSS\GameLibrary\matrix\game-matrix.csv` and `game-matrix.jsonl`. The CSV is a flat one-row-per-recording table for spreadsheet and research analysis; JSONL retains each nested matrix row. Use **Matrix folder** in the app to open it. These files stay on the machine unless you choose to share them.

The bundle does not include the absolute game install path, an imported config file's full path, the raw window title, audio, microphone input, clipboard contents, hostname, account name, IP address, or device serial numbers. Screenshot contents can still show private messages, account details, or third-party content, so review them before sharing.

## Important research limitation

These screen samples and available performance counters are useful for exploratory visual/performance research, not a complete supervised neural-rendering dataset. They do not contain real depth buffers, motion vectors, reprojected history, or paired high-resolution reference frames. TrSS must not fabricate missing inputs or label a screenshot as renderer ground truth.

## Install from source

Requires Windows 10/11, Python 3.10+ with Tkinter enabled, and a writable local data folder:

~~~powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
trss-recorder
~~~

To launch the older single-window capture utility directly, use **trss-window-recorder**. The game library is the intended participant interface.

## Portable Windows build and verification

The **Build TrSS Game Library** workflow runs the project tests, builds a one-file Windows x64 executable, and creates a SHA-256 checksum. The workflow artifact is available on the GitHub Actions run; the tagged release path is optional and is not required to test the executable. A successful CI build does not replace real-machine validation across Windows versions, GPUs/drivers, store launchers, fullscreen modes, config formats, or long captures. FPS/frametime, renderer buffers, and a broad catalogue of game-specific config adapters are still engineering work.
