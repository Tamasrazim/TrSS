# Windows desktop capture sampler

TrSS now has a first-party Windows color sampler based on DXGI Desktop Duplication and Direct3D 11 readback. It records a selected monitor to numbered PNG frames and writes a JSONL manifest that follows the existing TrSS recorded-sequence contract.

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
.\build\directx12\Release\trss_desktop_capture.exe --adapter 0 --monitor 0 --seconds 10 --fps 30 --max-frames 300 --output-dir captures\desktop-test
~~~

The sampler saves the most recently acquired desktop pixels at the requested time interval. If the visible desktop does not change, it repeats the last acquired image to preserve a time sequence. Actual cadence may be lower than requested because of CPU readback, PNG encoding, desktop updates, or system load; check session.json rather than treating requested FPS as achieved performance.

Outputs:

- frames/frame_000000.png and one numbered PNG per saved sample.
- capture-manifest.jsonl with relative frame paths and monotonic seconds as timestamps. Its rows can be read by trss render-sequence for compatible color-only models.
- session.json with adapter/monitor, dimensions, timing, output rate, and explicit limitations.
- .trss-desktop-capture-session, a safety marker required for --force replacement.

Existing output paths are protected by default. The --force option can replace only a directory already marked as a TrSS capture session. Frames are first written into a sibling staging directory and promoted only after a successful run; a failed run leaves an earlier capture untouched. The tool refuses a drive root, the current folder, or any parent of the current folder as the output location.

## Limits

1. It captures visible desktop pixels through DXGI Desktop Duplication. It does not attach to or modify a game process.
2. It does not promise capture of protected video or protected game surfaces. Desktop Duplication can fail after display-mode changes or when desktop access is lost; restart the session after such a failure.
3. The sampler reads GPU textures back to CPU memory and encodes PNGs. Its achieved frame rate is not in-game FPS, rendering latency, or an Arc B580 performance result.
4. Color-space metadata is set to unknown rather than guessing how Windows HDR, tone mapping, overlays, and display color management affected pixels.
5. The manifest deliberately omits depth, motion, history, and ground-truth references. TrSS does not fabricate them. A renderer-specific integration must provide real auxiliary buffers before training or running a model that requires them.
6. The Windows CI workflow builds this executable and uses --self-test-png to verify WIC PNG encoding. That test does not access the desktop or claim hardware validation.
