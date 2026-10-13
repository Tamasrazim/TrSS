# XESS5 architecture

XESS5 is being built as an application we own and understand, rather than as a launcher around another application's UI. The current executable is a native Windows frame-inspection shell. The image baseline deliberately does not claim to be a neural model.

## Current executable

- **UI:** Win32 window and native controls.
- **Image decoding:** Windows Imaging Component (WIC).
- **Image storage:** top-down 8-bit BGRA pixels.
- **Comparison:** preserve the source, generate a separately stored reference image, and show either image or an A/B split.
- **Reference operation:** a basic unsharp-mask filter, used to verify that loading, processing, display, and comparison work together.
- **Processing responsiveness:** the reference operation runs on a worker thread; the UI receives its result through a window message.
- **External components:** none are bundled for this initial path. WIC and Win32 are part of Windows.

This conventional image operation is not evidence of neural rendering. It is a baseline for the user interface and pixel-handling path only.

## Planned rendering boundaries

Keep each responsibility separate so a missing model or unsupported game API cannot break the interface.

### 1. Frame source

Input adapters should produce an explicit frame description: width, height, pixel format, colour space, timestamp, and, where available, depth and motion vectors. The first source is a local image. Later sources may include a controlled test renderer, desktop/game capture, or a supported graphics-API integration.

### 2. GPU backend

Probe the GPU and driver at startup, record the available APIs and formats, and choose a backend only when every required capability is present. The initial hardware target is the Intel Arc B580. Use Vulkan or another verified Intel-supported compute path, measuring GPU time separately from capture, copies, and presentation.

A Vulkan-capable GPU does not imply that every feature or every injected game path is supported. In particular, any matrix instructions, precision formats, or shader extensions must be checked at runtime.

### 3. Model backend

The model loader must verify its manifest, tensor shapes, precision, memory requirements, file hashes, and license before allocating large buffers. Keep the model format replaceable.

A source-code implementation and its model parameters are separate things. The application must never silently treat NVIDIA's proprietary DLL or extracted weights as freely redistributable. Use a separately licensed model or material the user has legitimate permission to use.

### 4. Renderer and history

The future renderer owns intermediate GPU images, compute dispatches, frame history, resize handling, and device-loss recovery. History is invalidated on scene-size changes, source changes, discontinuities, and explicit reset. The first neural inference milestone should use deterministic still-frame input before temporal history is enabled.

### 5. Game integration

The game hook is a separate module from the application shell. Each supported API/architecture combination needs a reproducible test. Installation must show exactly which files would change, preserve originals safely, roll back on error, and provide a verified removal path. Never silently touch anti-cheat protected game installations or claim online-game safety.

## Build

Requires Windows 10/11 x64, Visual Studio 2022 with the Desktop development with C++ workload, and CMake 3.24 or newer.

From a Developer PowerShell in the repository:

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release --parallel
.\build\Release\XESS5.exe
```

The initial frame-inspection shell only uses Windows system libraries. Vulkan and model-runtime dependencies will not be added until a tested backend is ready.

## Acceptance gates

1. Build succeeds on a clean Windows x64 runner.
2. A local PNG/JPEG/BMP/TIFF image loads without changing the source file.
3. The source and processed images have matching dimensions and remain independently viewable.
4. Processing does not freeze the UI or pair a result with the wrong input.
5. Backend self-test identifies the selected GPU and required capabilities.
6. The neural path is only labelled neural after licensed model inference is actually executing.
7. Frame timings distinguish inference, copy/capture, and total end-to-end latency.
8. Install/remove tests prove the game directory is restored after success, cancellation, and failure.

Each gate should have an executable test or repeatable manual test case. Don't claim an acceptance gate passed until it has actually run.
