# TrSS Data Recorder

TrSS Data Recorder is a small, opt-in Windows desktop app for collecting short, lossless visual samples and basic graphics-hardware labels from people who volunteer to help with TrSS research.

It is designed to be handed to testers who are not developers. It provides a preview before recording, clear consent steps, a compact capture duration, a shareable ZIP, a frame manifest, a hardware report, and SHA-256 checksums.

## What it records

- A screen crop matching the client area of one selected application window, saved as sequential RGB PNG images. The chosen window must remain active; recording stops if focus changes.
- Actual elapsed timestamps for each saved frame and the requested capture rate.
- The game/application label supplied by the participant, the capture size, the frame count, skipped capture slots, and whether the session completed.
- The operating-system version, CPU model, available graphics-adapter names and driver versions, and total installed RAM when available.

It does **not** record audio or the microphone, read clipboard contents, enumerate running processes, save the selected window title, collect usernames, hostnames, IP addresses or device serial numbers, attach to a game's process, or send data anywhere. The recorder has no upload function. The participant receives a local ZIP they can inspect and choose to send.

The selected window's pixels can still contain private information or third-party content visible inside that window. Participants must preview the capture, disable notifications, remove private information, and only share material they have permission to capture and share.

## Important research limitation

A window screenshot is not the same thing as a renderer capture. This utility collects visible color frames for visual review and compatibility research. It does not obtain real depth buffers, motion vectors, reprojected history, or a matched high-resolution ground-truth render.

**A recorder ZIP alone is not a ready-to-train TrSS supervised dataset.** The current first-party training pipeline requires aligned low-resolution/reference pairs and real auxiliary buffers under the contracts in [TRAINING.md](TRAINING.md). Never synthesize missing depth or motion data and label it as captured truth. A native capture path that safely obtains the required render buffers is a separate engineering task.

## Run from source

Requirements: Windows 10/11, Python 3.10 or newer, and a visible application window. Borderless-windowed mode is recommended; exclusive-fullscreen games may produce black screenshots.

From the repository directory, in PowerShell:

~~~powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
trss-recorder
~~~

You can also run **python run_recorder.py** from the repository directory after installing the project.

## Participant instructions

1. Open the target game/application and use borderless-windowed mode where possible.
2. Open TrSS Data Recorder. Select the target application window, enter the game/application name, and optionally add a neutral scene label such as **outdoor-driving** or **interior-pan**.
3. Select **Preview**. Confirm the preview contains only the intended application and no private information.
4. Choose 5, 10, 15, or 30 frames per second and a 5, 10, 15, or 30 second duration. Short, focused samples are preferred.
5. Choose an output folder. Check both permission and sharing confirmations only if they are true for your situation.
6. Select **Start capture**. The recorder minimizes and brings the selected window forward. Keep that application active throughout the sample. If another window takes focus, recording stops and the usable partial sample is packaged with an interruption reason. Audio is not recorded.
7. To stop early, press **Ctrl + Alt + F10**. The app returns when recording and ZIP verification finish.
8. Review the generated ZIP and its **README.txt** before sending it to the project maintainer. Send it through the channel the maintainer explicitly provides; the recorder itself will not upload it.

The recorder cannot verify ownership or permission for captured content. Checking the consent boxes records the participant's confirmation only.

## Files in the ZIP

Each bundle contains a single named session folder with:

| File | Purpose |
| --- | --- |
| **metadata.json** | Settings, start/end times, frame counts, elapsed timing, consent version, and privacy statements. |
| **system.json** | A limited OS/CPU/GPU-driver/RAM profile. |
| **sequence.jsonl** | One manifest row per frame with the frame number, measured timestamp, scene/session ID, and relative color path. |
| **frames/frame-XXXXXX.png** | Lossless captured color frame. |
| **SHA256SUMS.txt** | SHA-256 hashes of the other files, useful for detecting accidental changes. |
| **README.txt** | Plain-language contents, limitations, and privacy notes. |

The **sequence.jsonl** file uses TrSS's existing color-sequence manifest contract. Since the recorder does not capture depth or motion, those fields remain absent rather than being fabricated. Models that require those buffers will not run from this data alone.

## Build a portable Windows executable

The repository includes a Windows GitHub Actions workflow to package the app with PyInstaller. Run **Build TrSS Data Recorder** from the Actions tab to create a temporary build artifact. A tag named **recorder-v...** also publishes the compiled executable and its SHA-256 file as a GitHub Release.

To build locally on Windows:

~~~powershell
python -m pip install -e ".[test]"
python -m pip install pyinstaller
pytest -q
python -m PyInstaller --noconfirm --clean --onefile --windowed --name TrSS-Data-Recorder --collect-all PIL run_recorder.py
~~~

The executable will be written to **dist\TrSS-Data-Recorder.exe**. Build and test it on a real Windows machine before distributing it widely; screenshot behavior varies by game, graphics mode, Windows build, and driver.

## Troubleshooting

**Preview is black or fails.** Switch the application to borderless-windowed mode, make sure it is not minimized, refresh the window list, and preview again. The app must become the foreground window for the preview to proceed. Protected video surfaces and some exclusive-fullscreen paths may not be capturable.

**Recording stops after a resize.** Keep the selected application's render size fixed during a session. The recorder stops if the frame dimensions change so the manifest does not mix resolutions.

**Few frames were saved or the observed FPS is low.** Capture cost depends on resolution and image complexity. Use a lower frame rate or shorter sample. The metadata reports the actual saved timestamps and skipped schedule slots; it does not pretend the requested rate was achieved.

**Packaging fails because of disk space or access.** Choose a writable folder with enough free space. The uncompressed session folder is kept if ZIP creation fails so already captured frames can be recovered manually.

**The app does not launch.** Rebuild from a supported Windows Python installation. If running source, make sure the repository was installed with **python -m pip install -e .**.
