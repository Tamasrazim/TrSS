# Windows desktop capture sampler

TrSS now has a first-party Windows color sampler based on DXGI Desktop Duplication and Direct3D 11 readback. It records a selected monitor to numbered PNG frames and writes a timestamped JSONL manifest plus a session JSON report. You can add a user-supplied game label so captures are easier to identify; the tool does not inspect the game's process.

This is an acquisition step, not a game injection hook and not a completed neural-rendering integration. Desktop Duplication supplies visible color pixels only. This tool does not retrieve a game's private render targets, depth buffer, motion vectors, disocclusion masks, camera matrices, or a high-resolution reference. The generated manifest is not training-ready for the first-party temporal model. A model adapter must reject missing depth or motion when those inputs are required.

## Build

~~~powershell
cmake -S native/directx12 -B build/directx12 -A x64
cmake --build build/directx12 --config Release --parallel
~~~

The executable is build/directx12/Release/trss_desktop_capture.exe. It requires no Python package and uses Windows SDK components.

## Choose a monitor

~~~powershell
.\build\directx12\Release\trss_desktop_capture.exe --list-devices
~~~

The listing shows DXGI adapter indexes, attached outputs, dimensions, and desktop origins. Choose an adapter/output pair from the list. Indexes may change when Windows hardware or display configuration changes.

## Record a bounded sample

~~~powershell
.\build\directx12\Release\trss_desktop_capture.exe --adapter 0 --monitor 0 --seconds 10 --fps 2 --max-frames 20 --game "Euro Truck Simulator 2" --output-dir captures\desktop-test
~~~

The sampler saves the most recently acquired desktop pixels at the requested time interval. If the visible desktop does not change, it repeats the last acquired image to preserve a time sequence. Actual cadence may be lower than requested because of CPU readback, PNG encoding, desktop updates, or system load; check session.json rather than treating requested FPS as achieved performance.

Outputs:

- frames/frame_000000.png and one numbered PNG per saved sample.
- capture-manifest.jsonl with relative frame paths, monotonic timestamps, and an exact duplicate comparison against the previous saved image (null for the first frame). Its rows can be read by trss render-sequence for compatible color-only models.
- session.json with adapter/monitor, dimensions, timing, output rate, and explicit limitations.
- .trss-desktop-capture-session, a safety marker required for --force replacement.

Existing output paths are protected by default. The --force option can replace only a directory already marked as a TrSS capture session. Frames are first written into a sibling staging directory and promoted only after a successful run; a failed run leaves an earlier capture untouched. The tool refuses a drive root, the current folder, or any parent of the current folder as the output location.

## Limits

1. It captures visible desktop pixels through DXGI Desktop Duplication. It does not attach to or modify a game process.
2. It does not promise capture of protected video or protected game surfaces. Desktop Duplication can fail after display-mode changes or when desktop access is lost; restart the session after such a failure.
3. The sampler copies updated desktop textures into a staging resource, but maps pixels into CPU memory and encodes PNGs only at the requested sample cadence. Copy/readback still has some overhead; its achieved output rate is not in-game FPS, rendering latency, or an Arc B580 performance result.
4. Color-space metadata is set to unknown rather than guessing how Windows HDR, tone mapping, overlays, and display color management affected pixels.
5. The manifest deliberately omits depth, motion, history, and ground-truth references. TrSS does not fabricate them. A renderer-specific integration must provide real auxiliary buffers before training or running a model that requires them.
6. The Windows CI workflow builds this executable and uses --self-test-png to verify WIC PNG encoding. That test does not access the desktop or claim hardware validation.

## Train from captures made while playing

The capture manifest is an index, not a self-contained training dataset. The PNGs it references are the visual samples, so keep each complete session folder together.

Start with Euro Truck Simulator 2 and record two separate sessions from different stretches of road. Ten minutes sampled at 2 images per second yields at most 1,200 PNGs per session. Use separate output folders:

~~~powershell
.\build\directx12\Release\trss_desktop_capture.exe --adapter 0 --monitor 0 --seconds 600 --fps 2 --max-frames 1200 --game "Euro Truck Simulator 2" --output-dir captures\ets2-road-a
.\build\directx12\Release\trss_desktop_capture.exe --adapter 0 --monitor 0 --seconds 600 --fps 2 --max-frames 1200 --game "Euro Truck Simulator 2" --output-dir captures\ets2-road-b
~~~

Choose adapter/monitor indexes reported by --list-devices; they may vary on the PC. The game label is manual, not process-detected. The desktop sampler logs captured frame timestamps, capture statistics, and counts of identical versus changed adjacent samples; this helps spot a frozen menu/desktop capture. It does not measure the game's actual FPS, frame time, GPU utilization, or render latency.

Install Intel's PyTorch XPU build and then install TrSS:

~~~powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/xpu
python -m pip install -e ".[training,test]"
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"
~~~

Feed the two full capture folders—not only their JSON files—into the existing import, train, and evaluation workflow:

~~~powershell
python -m trss.training.gameplay_experiment "captures\ets2-road-a" "captures\ets2-road-b" --output-dir logs\ets2-first-run --game "Euro Truck Simulator 2" --device xpu --epochs 5 --max-frames-per-video 120 --validation-fraction 0.5
~~~

The importer validates each manifest path, creates a low-resolution input from each captured target, and keeps entire sessions separated between training and validation. The workflow then trains the RGB-only spatial baseline and creates an offline comparison gallery.

This is a real-game-image experiment, but the low-resolution input is synthetically downsampled from the captured target. The capture has no game depth buffer, motion vectors, original low-resolution render target, or temporal reprojection data; this does not establish a working live-game neural renderer or Arc B580 performance.

Keep game captures and extracted images local unless you have permission to redistribute them. For reviewing results, share the small experiment/evaluation JSON reports and a comparison screenshot; a manifest by itself does not contain visual training data.


## Quick capture → train test

The Windows artifact bundle includes Record-TrSSSession.ps1 beside the standalone EXE. For your first test, collect two short sessions so you can confirm that capture and training work before making large folders.

First, drive around a stretch of road and capture one minute:

~~~powershell
powershell -ExecutionPolicy Bypass -NoProfile -File .\Record-TrSSSession.ps1 -Game "Euro Truck Simulator 2" -Adapter 0 -Monitor 0 -Seconds 60 -SamplesPerSecond 2 -OutputDirectory "captures\ets2-road-a"
~~~

Then drive to a different stretch of road and run the same command with a different output folder:

~~~powershell
powershell -ExecutionPolicy Bypass -NoProfile -File .\Record-TrSSSession.ps1 -Game "Euro Truck Simulator 2" -Adapter 0 -Monitor 0 -Seconds 60 -SamplesPerSecond 2 -OutputDirectory "captures\ets2-road-b"
~~~

Both sessions must be captured in separate locations to give training a held-out recording. The helper lists monitor indexes and gives an 8-second countdown before it starts. The game must be visible on the selected monitor when that countdown ends. It creates each output folder with PNGs, capture-manifest.jsonl, and session.json.

Install Intel's PyTorch XPU build, then install TrSS and run a small training/evaluation experiment:

~~~powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/xpu
python -m pip install -e ".[training,test]"
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"

python -m trss.training.gameplay_experiment "captures\ets2-road-a" "captures\ets2-road-b" --output-dir logs\ets2-first-run --game "Euro Truck Simulator 2" --device xpu --epochs 3 --max-frames-per-video 120 --validation-fraction 0.5
~~~

Open logs\ets2-first-run\experiment-report.json and logs\ets2-first-run\evaluation\index.html. The workflow imports the two capture folders, creates half-size inputs from captured targets, trains the RGB-only spatial baseline, and compares predictions with the held-out recording. Start small; increase capture duration and epochs only after this test finishes cleanly.

**JSON alone is not the training data.** The manifest points to the PNGs; keep the entire capture folder together. You do not need to upload capture folders to GitHub or commit game frames. Keep them local unless you have permission to redistribute them. To get results reviewed, share experiment-report.json, evaluation-report.json, and a comparison screenshot. This pipeline uses captured RGB only; it does not read engine depth/motion buffers, and it is not yet a live-game temporal neural renderer.
