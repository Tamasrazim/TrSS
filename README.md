# XESS5

**A Python-first neural graphics engine for Intel Arc.**

XESS5 is being developed around actual model inference, graphics-frame contracts, temporal state, and GPU diagnostics, with the Intel Arc B580 12 GB as the first hardware target. The aim is a local neural-rendering engine that we own and can integrate with a documented graphics path—not a wrapper around someone else's app.

This is independent research software, not an Intel or NVIDIA product. It does not claim to be NVIDIA DLSS 5, and the engine must not label a conventional filter or generic super-resolution model as neural rendering.

## What is implemented

- **OpenVINO device discovery:** lists the GPU devices actually exposed by the installed runtime, including distinct `GPU.X` adapters.
- **Strict Arc selection:** `ARC` resolves only when one device is identified as Intel Arc. Ambiguous Arc devices must be selected explicitly; a CPU fallback is rejected.
- **Single-image inference:** a convenience path for one rank-four RGB image input and one image output, with NCHW/NHWC layout, RGB/BGR channels, numeric range, FP16/FP32 input, warm-up calls, repeated measurements, and JSON reports.
- **Model inspection:** reads a local ONNX/OpenVINO-supported model and describes its input/output names, shapes, and precision without running inference.
- **Multi-input/multi-output inference:** named tensors are checked against the model's actual ports. Dynamic shapes are specialized and compiled for the selected GPU; outputs and timing details are returned explicitly.
- **Graphics-frame contract:** `FramePacket` carries source color plus only those depth, motion-vector, noise, and conditioning buffers actually supplied by the source, along with color/depth/motion conventions.
- **Spec-driven frame adapter:** an explicit JSON file maps model port names to frame sources, tensor layouts, resampling rules, channel order, and numeric ranges. Static model sizes are honored; missing depth/motion/history is an error, not fabricated input.
- **Offline frame inference command:** `render-frame` feeds a screenshot and real auxiliary `.npy` buffers through the adapter into a multi-input model, writes raw output tensors, and records preprocessing/compile/inference timings. It is a model-pipeline test, not a live game hook.
- **Specialized HNet-style frame contract:** `xess5/rendering/hnet.py` packs the documented 16-channel feature tensor, deterministic noise, exact staged FP16 color scaling, network field padding, and the four-channel residual/temporal-gate composition. `hnet-frame` connects that adapter to a supplied, shape-compatible model. The graph and trained weights are not bundled, so this does not by itself establish functional equivalence or vendor model compatibility.
- **Temporal history safety:** previous outputs can be kept between frames, but are invalidated after camera cuts, scene/resolution changes, frame-order gaps, or stale timestamps.
- **Automated CPU-side tests:** fake-runtime tests exercise model binding, shape changes, GPU-selection rules, data validation, and temporal history without pretending to test real GPU performance.

The frame and tensor layers are infrastructure for a neural renderer. They are not themselves a trained rendering model, a game capture hook, or proof of support on a physical Arc B580.

## Setup

Use Python 3.10 or newer. On Windows PowerShell, from the repository folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[gpu,test]"
```

OpenVINO GPU support also depends on a compatible Intel graphics driver and the installed runtime's GPU plugin.

## Commands

### Identify the actual adapter

```powershell
xess5 devices
```

XESS5 prints the device IDs and names it detects. It refuses to claim Arc selection when it can't distinguish an Arc GPU. On a machine with multiple Intel Arc GPUs, pass an explicit device such as `GPU.1`.

### Inspect a model before executing it

```powershell
xess5 inspect --model "models\your-model.onnx"
```

The command shows every named input and output, its shape, and its precision. Do this before preparing tensors; XESS5 intentionally does not invent preprocessing semantics for unknown models.

### Run a simple image-to-image model

```powershell
xess5 infer --model "models\your-model.onnx" --input "input.png" --output "output.png" --device ARC --layout NCHW --input-channel-order RGB --output-channel-order RGB --input-range 0:1 --output-range 0:1 --warmup 3 --iterations 10 --report "logs\run.json"
```

Use this only for a model that truly expects a single rank-four image tensor and returns a rank-four image tensor. Confirm channel order, layout, dimensions, and ranges from the model's documentation. The reported inference-equivalent FPS is only the reciprocal of the isolated model time; it is not game FPS.

### Adapt a captured frame to model inputs

For a model whose port semantics you have verified, create a model-specific JSON file using [the adapter template](docs/examples/adapter-template.json), then run a saved frame with the actual auxiliary buffers available to your capture workflow:

```powershell
xess5 render-frame --model "models\render-model.onnx" --spec "models\render-adapter.json" --color "frames\frame-0001.png" --depth "frames\depth-0001.npy" --motion "frames\motion-0001.npy" --history "frames\previous-output.npy" --output-dir "work\frame-output" --preview-output "work\frame.png" --history-output "work\frame-history.npy" --device ARC --warmup 2 --iterations 10 --report "logs\frame-report.json"
```

Only provide buffers that actually exist and that the adapter spec requires. Preview and history output require an explicit direct-RGB output declaration in the spec; feature planes and packed neural heads need a model-specific composition routine. History output is HWC float32 in the [0, 1] range. Depth and motion encoding, color space, model-specific noise, tensor normalization, temporal reprojection, and output-head composition must match the chosen model's documentation. See [MODEL-ADAPTERS.md](docs/MODEL-ADAPTERS.md). This command processes offline frame files; it does not capture a live game or claim real-time performance.

### Execute a model with multiple named tensors

For a model with color, depth, motion, history, or conditioning inputs, save the exact tensors expected by that model as NumPy `.npy` files and pass one `--input NAME=FILE.npy` argument per declared input:

```powershell
xess5 infer-tensors --model "models\render-model.onnx" --input "color=work\color.npy" --input "depth=work\depth.npy" --input "motion=work\motion.npy" --output-dir "work\outputs" --device ARC --warmup 2 --iterations 10 --report "logs\tensor-run.json"
```

Names must exactly match the model's input ports, and tensors must match the expected dimensions, precision, units, and preprocessing. The engine refuses missing/extra names, non-finite values, and complex inputs. It does not infer depth or motion vectors from color; those buffers must come from a real, documented source or a deliberate model-specific preprocessing stage.

### Run the HNet-style 16-to-4-channel frame adapter

The specialized adapter is for a compatible, user-supplied model that accepts the documented `[1,16,H,W]` feature tensor and emits a `[1,4,H,W]` head. It builds the frame conditioning and applies the documented residual/gate composition around the supplied model. It does not contain model weights or a neural graph.

```powershell
xess5 inspect --model "models\your-hnet-model.onnx"
xess5 hnet-frame --model "models\your-hnet-model.onnx" --color "frames\input.png" --output "work\rendered.png" --device ARC --output-layout NCHW --frame-index 0 --style-index 0 --tone-strength 1 --structure-strength 1 --warmup 1 --iterations 5 --history-output "work\next-history.npy" --head-output "work\raw-head.npy" --report "logs\hnet-frame.json"
```

The command refuses incompatible input/output shapes rather than pretending any rank-four network is the expected graph. A supplied model must also have matching weights, input preprocessing, head output semantics, and legal terms. The report's timing separates model inference from the whole offline frame path; neither is in-game FPS. Previous history is optional for the first frame and, when provided, must be an RGB HWC `.npy` array in the 0..1 range.

### Run the tests

```powershell
python -m pytest
```

CI checks syntax and the CPU-only test suite. CI does not have the target Arc B580, so it cannot prove GPU inference speed or visual results.

## Still missing for actual neural rendering

1. **A verified model-specific neural-rendering model.** The generic JSON adapter and a specialized HNet-style 16-to-4 frame contract are implemented, but the actual graph and legally usable parameters are not bundled. Exact tensor semantics, weights, precision, temporal behavior, and all model-specific post-processing still require verification against a compatible model.
2. **Model parameters.** Source code and trained weights are different things. No NVIDIA DLL, extracted proprietary weights, or trained model is bundled with XESS5.
3. **A real graphics-frame source.** No desktop/game capture hook is implemented yet, and the engine currently receives local images or explicit arrays instead of frames from a running game.
4. **Temporal reprojection and composition.** The current history store invalidates unsafe history but does not calculate motion-vector reprojection or the network's exact history blend.
5. **Live GPU integration.** There is no XESS5 Vulkan layer/game hook yet; `hnet-frame` only processes local frame files through a supplied model.
6. **Physical hardware validation.** No XESS5 run on the user's Arc B580 has yet established model compatibility, visual quality, latency, or in-game performance.

A generic image model may demonstrate the inference pipeline, but it is not a stand-in for a game-conditioned neural renderer. The next real milestone is to match an actually usable neural-rendering model's exact contract and parameters, add its preprocessing and output composition, and measure execution on the physical B580.

## Research references

- [DLSS-NR on Intel](https://github.com/Uzbekunknown/dlss-nr-on-intel) — experimental Intel Xe2 neural-rendering research with a Vulkan path. Its current code is a research reference; the published project requires separate compatible model material and should not be copied without retaining its Apache-2.0 license and notices.
- [OpenDLSS-NR](https://github.com/aloshdenny/open-dlss) — another model implementation whose documented GPU requirements are NVIDIA-specific; not a direct B580 backend.
- [OpenVINO](https://github.com/openvinotoolkit/openvino) — the current generic inference runtime.

These are technical references, not opaque third-party applications wrapped by XESS5. See [third-party research and licensing boundaries](docs/THIRD-PARTY-RESEARCH.md). Any adopted code, model, shader, or dataset must have its exact version and license recorded before it becomes a dependency.

## Development rules

- Main branch only; commits land directly on `main`.
- No releases or extra branches.
- Local-first: no account, cloud inference, or subscription is required.
- Model and graphics claims are only added after an executable test supports them.
