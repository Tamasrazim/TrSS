# TrSS Renderer — Developer README

This is my working map for TrSS: what the renderer currently does, how its parts fit together, what has actually been tested, and what I should build next. It is an engineering note, not a promise that unfinished features already work.

**Status date: 14 October 2026**

## The goal

TrSS is my first-party, local-first real-time scene-synthesis and neural-rendering project. The first hardware validation target is my Intel Arc B580 12 GB. The end goal is to reconstruct and improve a game frame using a trained model, with better detail and stable results across consecutive frames—not merely to add color filters or enlarge an image.

The renderer must eventually combine a real frame source, a compatible trained model, real auxiliary data where the model needs it, efficient GPU inference, temporal handling, and a way to measure both image quality and performance. These parts must work together before I call TrSS a live neural renderer.

## Current status — read this before making claims

| Area | Actual state |
| --- | --- |
| Python model / inference tools | Implemented as a research pipeline for offline images and tensors |
| First-party PyTorch model | Trainable 2× temporal reconstruction baseline; no trained checkpoint is bundled |
| Dataset tools | Manifest splitting, validation, consistency checks, and a local visual review gallery are implemented |
| Native Direct3D API probes | D3D9, D3D11, and D3D12 smoke-test paths exist; probes are not game renderers |
| Vulkan/XMX route | Experimental offline/native bridge; physical Arc B580 inference is not verified |
| In-game runtime | First-party Windows x64 DXGI proxy and D3D11 post-processing shader/menu compile in CI |
| Effects in the current runtime | Conventional screen-space post-processing, not neural reconstruction |
| Runtime + PyTorch integration | Not implemented |
| Live depth, motion vectors, and reprojected history | Not acquired by the current runtime |
| Actual game test | Not yet reported; compatibility is unknown |
| Arc B580 quality and speed measurements | Not yet completed |
| Trained weights and a paired training dataset | Not bundled |

**Overall roadmap estimate: 65%.** That number is a rough engineering estimate toward a complete, live, physically validated neural renderer. It is not a measured metric, percentage of supported games, or performance result. Recent swap-chain and installer hardening makes the current prototype more robust on paper, but it does not justify increasing the neural-renderer progress estimate before real integration and validation.

## How the parts fit together

TrSS currently contains several related but separate paths. Keep them separate when debugging and documenting results.

### 1. Offline model path

The Python engine accepts saved images or explicitly described tensors, selects a real device exposed by the installed backend, validates input/output shapes, and reports timing. Model adapters describe the meaning and format of each model input. The offline sequence runner can maintain history between recorded frames and reset it at scene changes, gaps, cuts, or invalid timing.

Relevant areas:
- `trss/engine.py` and `trss/tensor_engine.py`: model execution and named tensors.
- `trss/rendering/frame.py` and `trss/rendering/sequence.py`: frame contracts and offline sequence processing.
- `trss/training/`: the first-party PyTorch baseline, training loop, inference timing, manifest audit, splitting, and visual review.
- `docs/TRAINING.md` and `docs/MODEL-ADAPTERS.md`: dataset and model contracts.

This path can test a model on files. It does **not** capture frames from a live game.

### 2. Native live runtime

The live prototype is in `native/runtime/`. It builds an app-local `dxgi.dll` proxy. When a compatible game creates a DXGI swap chain through an intercepted factory path, TrSS attempts to hook Present and apply a first-party D3D11 shader to the image before presentation. The in-game menu is drawn by the same native runtime.

The current shader exposes these controls:

- Brightness, contrast, saturation, and gamma.
- Sharpening and a basic bloom pass.
- Vignette and film grain.
- Color temperature and chromatic aberration.
- Neutral, Cinematic, Vibrant, Clarity, Night, and Film presets.

The menu toggle is **Ctrl + Alt + Page Up**. These controls are intended to make the prototype visible and testable; they are not evidence that the neural model runs in a game.

### 3. Independent game recorder

The Windows TrSS Game Library/recorder is a separate tool. It does not need a renderer connection and must remain usable on its own. It records screenshots and available Windows process/system counters, asks me to verify graphics settings after a capture, and builds a local matrix for controlled comparisons.

Native FPS and frame-time values remain unavailable until TrSS's own per-present/frame-event collector is implemented. The recorder must never relabel its screenshot rate as game FPS. See [docs/RECORDER.md](RECORDER.md) and [docs/RUNTIME.md](RUNTIME.md).

### 4. Graphics API probes and XMX work

The D3D9, D3D11, and D3D12 probes verify basic graphics/API operations. They do not hook games. The experimental Vulkan/XMX frame path is separate from the D3D11 runtime. Compiling a probe or native library only tells me that a build path succeeded; it does not prove the model runs in a game or on the B580.

## Live runtime: exact current limits

The prototype targets **Windows x64 games using a compatible Direct3D 11/DXGI swap chain**, created through the ordinary intercepted DXGI factory path and using a single-sample 8-bit BGRA/RGBA back buffer.

It is a narrow starting point. D3D9, D3D12, Vulkan, OpenGL, unsupported back-buffer formats, alternate creation paths, some fullscreen arrangements, and games that bypass the intercepted path are outside this prototype's support boundary. Even a compatible-looking game may not show the menu. No anti-cheat or protection bypass is provided or intended.

Native work already added:
- Active back-buffer index selection for flip-model swap chains that expose `IDXGISwapChain3`.
- Per-buffer render-resource preparation and guards for unsupported buffer counts/resources.
- Windowed menu/input handling and resize-related resource reset logic.
- x64 executable architecture validation before installation.
- Refusal to overwrite an existing `dxgi.dll`.
- A SHA-256 install manifest; uninstall removes only the matching TrSS DLL and preserves settings/presets.
- Automated tests for installer validation, plus a successful Windows CI runtime build.

These are source/build checks. **They are not a substitute for trying the runtime in a real game.** The most recent successful native runtime build artifact is produced by the [first-party runtime workflow](https://github.com/Tamasrazim/TrSS/actions/workflows/build-trss-runtime.yml). Do not treat it as a production release.

## My next steps, in order

### Step 1 — verify the existing build in one game

This is the current external blocker: I need a real test result before I can responsibly claim the menu or effect path works in a game.

1. Download the `TrSS-Live-Runtime-windows-x64` artifact from the successful [runtime workflow run](https://github.com/Tamasrazim/TrSS/actions/runs/38590709910).
2. Pick one local/offline game that uses a compatible D3D11 path and does not require bypassing anti-cheat or other game protections. Back up the game's folder/configuration first.
3. Run `TrSS-Runtime-Installer.exe`, select the actual game executable, and install only if there is no conflicting `dxgi.dll`.
4. Launch the game normally. Try **Ctrl + Alt + Page Up**. Check if the menu opens, mouse interaction works, presets can be selected, and a clearly visible change such as saturation or brightness affects the frame.
5. If the game fails to start, renders incorrectly, or becomes unstable, close it and use the TrSS uninstaller. Do not leave a broken proxy installed while trying more games.

Record the exact game/version, API if known, resolution/window mode, whether the game launched, whether the menu appeared, which controls worked, and any crash/error. Do not use an anti-cheat-protected online session for this test.

### Step 2 — fix runtime bugs based on evidence

Use the result from Step 1 to reproduce one specific issue at a time. Check the Present hook, back-buffer index/resource lifetime, resize and alt-tab behavior, input capture, state restoration, and menu rendering. A CI pass is not enough to mark any of these game behaviors complete.

### Step 3 — establish a real, lawful frame-data plan

A presented color image alone is not enough for every neural renderer. Identify which API path can provide the actual color buffer and, for the chosen model, real depth, motion vectors, and temporal inputs. Those buffers are game/engine-dependent; TrSS must record them only when genuinely available and must not invent replacement data.

The existing screenshot recorder is for evaluation and review, not a source of true depth or motion vectors. For training, use legally usable, aligned input/reference frame pairs and document the capture method, color space, motion/depth convention, game settings, and scene identity.

### Step 4 — make model and native runtime contracts agree

Before connecting the model to the game, pin down the exact model graph, trained checkpoint format, tensor layout/range, color space, scale, output-head composition, and history semantics. The first-party PyTorch checkpoint is not automatically compatible with the experimental fixed-graph XMX runner. Add explicit weight mapping and parity tests instead of assuming one backend can load another backend's files.

Then implement and profile one native inference route. Define frame ownership, synchronization, GPU resource lifetime, device-loss handling, resize behavior, and frame pacing. Do not add more graphics APIs until one supported end-to-end path is verified.

### Step 5 — train and validate on the target GPU

First prepare an audited dataset with scene-separated train/validation splits, genuinely aligned reference frames, and any auxiliary buffers required by the model. Train a checkpoint, compare it against simple baselines such as bicubic reconstruction, review visual outputs, and measure synchronized model/composition latency.

Then run on the physical Arc B580 and report real image-quality and timing values. Do not convert isolated model latency into a claim about in-game FPS. Keep measurements separate for model inference, full processing path, and game-level performance.

### Step 6 — connect performance comparisons without making false training claims

Record matched Baseline and TrSS runs using the same game, experiment group, graphics settings, resolution, scene, and driver. The recorder's matrix can help estimate process/GPU overhead when the required counters are present. Repeat each run and label the renderer/preset accurately.

Performance CSV/JSON is evaluation evidence; it is not supervised pixel-training data. Join it to image datasets only by explicit experiment identifiers and metadata, never by guesswork.

## What counts as complete

Do not call TrSS a working live neural renderer until all of these are true:

- A trained, legally usable model and a documented model/checkpoint contract exist.
- A supported live graphics path delivers valid frames and all model-required auxiliary inputs.
- The native runtime actually runs that trained model; the post-processing shader is not standing in for it.
- Temporal handling/output composition match the verified model contract.
- The runtime has been exercised in real target games without breaking normal rendering on the declared support path.
- Visual outputs and end-to-end latency are recorded on the physical Arc B580.
- The documentation distinguishes measurements from estimates and lists untested APIs/games honestly.

## Build and test pointers

- Live runtime and installation behavior: [docs/RUNTIME.md](RUNTIME.md).
- Recorder and game-matrix workflow: [docs/RECORDER.md](RECORDER.md).
- Model training: [docs/TRAINING.md](TRAINING.md).
- Frame and model adapters: [docs/MODEL-ADAPTERS.md](MODEL-ADAPTERS.md).
- Engine architecture: [docs/ARCHITECTURE.md](ARCHITECTURE.md).
- Run Python checks from the repository root with `python -m pytest`.
- The live runtime must be built on Windows with MSVC and the Windows SDK by the native runtime workflow.

**Rule for future updates:** if code is only compiled or tested with mocks, call it compiled/tested—not game-verified. Keep the roadmap at 65% until there is meaningful progress on trained-model integration and actual hardware/game validation.
