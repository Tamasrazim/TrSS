# XESS5 engine architecture

XESS5 is a Python-first neural inference engine for Intel Arc. The first target is the Arc B580 12 GB. Python owns model contracts, tensor preparation, execution orchestration, diagnostics, and tests. GPU work is delegated to a verified inference backend; performance-critical capture or graphics integration can be native code later, but it must serve the engine rather than become a separate demo application.

## Existing implementation

- `xess5/contracts.py`: explicit numeric-range parsing and RGB image/tensor conversion.
- `xess5/backends/openvino_backend.py`: lazy OpenVINO loading and local device discovery.
- `xess5/engine.py`: model validation, FP16/FP32 image input, static/dynamic shape preparation, GPU compilation, one inference request, output conversion, and timing.
- `xess5/cli.py`: device listing and model inference commands.
- `tests/test_contracts.py`: CPU-only tests for tensor layouts, ranges, input resizing, and output conversion.

The current contract expects one rank-four RGB image tensor and one rank-four image output. The same layout is assumed for the input and output. Input and output value ranges are explicit command-line options because models do not all use the same preprocessing.

## Why this isn't the full neural renderer yet

A general image-to-image model cannot automatically replace DLSS 5 neural rendering. A specialized rendering network may expect scene-specific conditioning, injected noise, depth, motion vectors, reprojected history, style controls, and other tensors. Those inputs must be derived from the actual graphics pipeline and passed with the model's documented preprocessing. XESS5 must add a model-specific adapter rather than guess these details.

The initial generic engine is a reusable inference layer. It does not contain proprietary NVIDIA weights, reconstruct those weights, or claim compatibility with DLSS-NR's model contract.

## Inference lifecycle

1. **Detect:** enumerate OpenVINO devices, and refuse an unavailable requested GPU instead of silently falling back to CPU.
2. **Validate:** read the local model, check its input/output count, rank, channel dimensions, and supported floating-point precision.
3. **Specialize:** resolve dynamic image dimensions to the input image and respect fixed dimensions declared by the model.
4. **Prepare:** decode a local image, convert it to RGB, resize only when the model declares a fixed input shape, apply the user-selected value range, and create a contiguous tensor.
5. **Execute:** compile for the selected device and measure the inference call separately.
6. **Convert:** normalize model output using its declared value range and write the output image to a separate path.
7. **Report:** emit device name, input/model/output dimensions, inference timing, total timing, and an FPS-equivalent for the isolated inference call. That number is not game FPS.

## Planned engine milestones

### 1. Verified GPU inference
Run a permissively licensed image-to-image model on the Arc B580 with a recorded device name, model hash, settings, output image, and reproducible timings. A CI runner without the target hardware cannot validate this.

### 2. Model-specific neural-rendering adapter
Document all inputs, tensor shapes, layouts, precision, normalization, conditioning, and output composition for a model whose license permits the planned use. If an adapter requires multiple inputs or temporal state, extend the contract deliberately instead of relaxing validation.

### 3. Graphics-frame source
Add a repeatable test renderer and explicit frame metadata: size, pixel format, colour space, timestamp, and optional depth/motion vectors. Check capture overhead separately from model execution.

### 4. Temporal state
Add previous-frame history, reprojection, reset on scene cuts/resize/source changes, and tests that detect stale history. Only use motion and depth data when the source provides valid buffers.

### 5. Low-overhead live integration
Investigate Vulkan integration for documented and tested paths. Installation must preview changed files, preserve existing files, roll back on failures, and offer a verified remove operation. Do not suggest modified game files are safe with anti-cheat protected online play.

## Build and run

Requires Python 3.10+, a compatible Intel graphics driver, and OpenVINO's GPU plugin for GPU inference.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[gpu,test]"
xess5 devices
xess5 infer --model "models\model.onnx" --input "input.png" --output "output.png" --device GPU --layout NCHW --input-range 0:1 --output-range 0:1 --report "logs\run.json"
python -m pytest
```

Use a local model that follows the declared image contract. ONNX and OpenVINO IR are supported only insofar as OpenVINO can read them and their tensor contracts match. Model weights are not supplied by this repository.

## Acceptance gates

- CPU-only unit tests pass on CI.
- The runtime reports the exact available GPU devices.
- The selected GPU is not replaced by silent CPU fallback.
- Invalid model contracts and unavailable devices report actionable errors.
- Source images are never overwritten by output.
- Inference and end-to-end timings are separated and accurately named.
- At least one compatible licensed model is run on the physical Arc B580 before claiming hardware support.
- Model-specific neural-rendering inputs are validated before calling a model a neural renderer.
- Live integration has a repeatable test and a demonstrated rollback before being called supported.
