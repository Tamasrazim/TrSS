# TrSS Sample Recorder

**TrSS Sample Recorder** is a portable, native Windows GUI for collecting small RGB-only samples while playing a supported installed game. The app is written in C++/Win32 and uses Windows APIs; it does not need Python, a console window, PowerShell, a service, administrator rights, or a third-party runtime package.

## Use it

1. Download the **trss-sample-recorder-win-x64** artifact from the latest successful **TrSS Sample Recorder — native Windows build** workflow run.
2. Extract the ZIP and double-click **TrSSSampleRecorder.exe**. The program opens a normal Windows application window; it does not open a terminal.
3. Let it scan the local game library. Select a game and choose **Launch selected**, or launch the game from its regular launcher.
4. The recorder begins its **120-second warm-up** when it first detects a visible window belonging to an installed game. No screenshots are stored during the warm-up.
5. Keep the game in the foreground to collect samples. Switch away from the game or close the recorder to finalize the current JSON file. Open the samples folder from the app when you are ready to review the files.

The default output folder is **Documents\\TrSS Sample Recorder**. Use **Choose samples folder** to change it. The recorder never uploads data anywhere.

## Installed-game detection

The library scans Steam library manifests, Epic Games Launcher manifests, common Ubisoft Connect / EA / GOG / Xbox / local game folders, and Windows uninstall registry entries. This is best-effort detection rather than a promise that every third-party launcher or unusual custom installation will be recognized.

Games may need to be launched through their original store or launcher if their anti-cheat, DRM, or bootstrapper rejects direct launching. The recorder can still start its warm-up when it later detects the game's installed executable and visible game window.

## What each JSON file contains

The recorder samples at up to **2 frames per second**, only while the target game window is in the foreground. Each image is JPEG-encoded, embedded as Base64 in the JSON object, and capped at 1920 pixels on its longest side. Every completed **chunk-####.json** is measured in UTF-8 bytes and stays at or below **23 MiB (24,117,248 bytes)**. When a chunk reaches the limit, the recorder closes it and starts the next numbered chunk.

While a chunk is being written, it uses a **.json.part** temporary name. The file is renamed to **.json** only after the closing JSON syntax is written, so completed chunks are parseable and ready to upload. An empty or incomplete temporary chunk is not a valid upload.

Each chunk contains:

- Session ID and chunk number, game name / detected store, process ID, and executable name.
- Per-frame UTC time, monotonic elapsed time, image dimensions, image format and embedded pixels.
- Best-effort process CPU percentage and working-set memory. Missing measurements are written as null, not fabricated.
- Explicit metadata that the source contains visible RGB pixels only.

The recorder does **not** record the whole desktop in the background. It checks that the selected game's process owns the foreground window before saving a sample, and it does not store window titles or full local executable paths in the JSON.

## Privacy, compatibility, and limits

- Samples stay local until the user manually copies or uploads them. There is no network upload or GitHub token in the application.
- A screenshot can still include an in-game name, chat, or other information visible inside the game. Review your own captures before sharing them.
- The app does not inject into a game, read game memory for rendering data, install an overlay, or modify game files. Generic Windows screen capture can fail with exclusive-fullscreen or protected surfaces; **Borderless Windowed** mode may work better.
- This is a screen sampler, not a renderer hook. It cannot obtain the game's original high-resolution target, depth buffer, motion vectors, camera matrices, material data, or temporal history. It does not capture true in-game FPS or prove that neural rendering is running.
- Captured RGB frames can be used as source images for TrSS's **RGB-only spatial experiment**. The existing training path creates a smaller input by downsampling each captured frame; this is useful for a controlled super-resolution experiment, but it does not train a renderer with real ground-truth low/high resolution pairs or temporal game buffers.
- Only completed JSON chunks with the **trss.sample-recorder.chunk.v1** schema should be uploaded. Keep separate game sessions separate so validation can be held out by whole session rather than mixing adjacent frames across the train/validation split.

## Import chunks into the TrSS experiment pipeline

The repository includes **trss-import-sample-recorder**. It validates the 23 MiB limit and embedded image metadata, groups chunks by their original session ID, extracts the JPEG frames, and writes ordinary TrSS capture-session folders. Those sessions can be supplied to **trss-gameplay-experiment**.

For developer-side use after installing TrSS's Python training dependencies:

~~~text
trss-import-sample-recorder <folder-containing-chunks> --output-dir imported-samples
trss-gameplay-experiment imported-samples --output-dir gameplay-experiment --game "Recorded game"
~~~

Use at least two independent recorder sessions for a training experiment. The importer does not create depth or motion data, and the resulting model remains an RGB-only spatial baseline.
