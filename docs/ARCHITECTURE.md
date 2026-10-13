# XESS5 engine architecture

XESS5 is a Python-first neural graphics engine targeting Intel Arc. The Arc B580 12 GB is the first validation target. Python owns orchestration, explicit tensor contracts, model execution, input validation, temporal state, diagnostics, and tests. Low-level GPU kernels and a native graphics hook may be needed later, but they exist to power the engine—not a separate UI prototype.

## Components in the repository

- `xess5/backends/openvino_backend.py`: local device inventory and strict Arc/explicit GPU selection. No silent CPU fallback.
- `xess5/contracts.py`: normalized-range parsing and image-to-tensor/tensor-to-image conversion, with explicit NCHW/NHWC and RGB/BGR handling.
- `xess5/engine.py`: single-image convenience engine, model inspection, shape validation, GPU compilation, output conversion, and timings.
- `xess5/tensor_engine.py`: generic multi-input/multi-output tensor engine. Named inputs are validated, dynamic tensor shapes are specialized, compilation is reused for repeated shapes, and copied outputs are reported with compile/inference/total timings.
- `xess5/rendering/frame.py`: a validated `FramePacket` for color and optional real auxiliary buffers, plus temporal output history with reset rules.
- `xess5/cli.py`: `devices`, `inspect`, `infer`, and `infer-tensors`.
- `tests/`: CPU-only tests for conversions, model contracts, named multi-input execution using a fake runtime, adapter selection, and history invalidation.

## GPU selection

`xess5 devices` reports the actual IDs and names OpenVINO exposes. `--device ARC` is intentionally strict: it selects an Intel Arc-named GPU only if that is unambiguous. When multiple devices identify as Arc, pass an exact ID such as `GPU.1`. CPU devices are not accepted as substitutes for a GPU. This prevents a hybrid system from accidentally running on the integrated adapter or CPU while presenting the result as B580 performance.

The Windows/Linux driver and OpenVINO GPU plugin are system requirements. A successful CPU-only CI run is not hardware validation.

## Single-image engine versus tensor engine

`ImageInferenceEngine` is a convenience interface for simple models with exactly one rank-four image input and one rank-four image output. Its resize/layout/range options must match the selected model.

`TensorInferenceEngine` is the lower-level interface for rendering graphs that consume more than one tensor. It takes a map from model input names to arrays, verifies exact input-name coverage and shape constraints, and returns a dictionary of outputs. This allows the adapter layer to express color, depth, motion, noise, prior history, and conditioning without forcing them into an invented image-only interface.

The generic tensor engine does not know what a tensor means. It does not generate motion vectors, transform a depth buffer into the model's expected encoding, synthesize noise, or choose temporal blend weights. Those responsibilities belong to a model-specific adapter with a documented contract.

## Frame packet and history

`FramePacket` stores:

- RGB/RGBA color as an HWC numeric array, with explicit color-space metadata;
- optional H×W depth, H×W×2 motion vectors, H×W×3 noise, and one-dimensional conditioning;
- timestamps, frame IDs, scene identity, and camera-cut information;
- explicit depth and motion-vector conventions, where known.

Missing inputs remain missing. The package never fills a depth or motion tensor with made-up values.

`TemporalHistory` stores the last valid processed RGB output and can return a copy to the next frame. It resets on camera cuts, size/scene changes, non-increasing frame IDs, dropped frame sequence IDs, non-monotonic timestamps, and long gaps. This is history lifecycle management only: there is no reprojection, optical-flow estimate, or model-specific blend in this class.

## Inference lifecycle

1. **Discover:** list the devices OpenVINO exposes; resolve `ARC` or an explicit `GPU.X` ID.
2. **Inspect:** read the local model, and report the model's names, shapes, and element types.
3. **Validate:** check input count/names, ranks, static dimensions, numeric types, and finite values. Fail rather than guess.
4. **Prepare:** the adapter converts actual frame data to each model's documented layout, precision, units, and ranges.
5. **Compile:** specialize dynamic dimensions and compile the model for the selected GPU. Reuse the compiled request while the same shape remains active.
6. **Execute:** measure inference time separately from compile and total call time. Do not describe isolated inference throughput as game FPS.
7. **Compose:** a future model-specific adapter maps output tensors into the frame according to the selected model's real output contract.
8. **Report:** save device identity, model path, shapes, settings, timings, and errors.

## Next engineering milestones

### 1. Choose and verify the model contract
Select properly licensed parameters and an architecture relevant to neural graphics. Record exact file hashes, names, input meanings, tensor shapes, color space, normalization, precision, output composition, and the license. If it expects a proprietary runtime or weights that cannot be redistributed, document that boundary rather than bundling it.

### 2. Implement a model-specific adapter
Map `FramePacket` into the actual model's inputs. Implement exact preprocessing, deterministic noise if required, conditioning vectors, depth/motion conventions, and output composition. Check an offline set of frames against known reference outputs before integrating a game.

### 3. Validate on the physical Arc B580
Record driver, OpenVINO/runtime version, selected device name, resolution, model hash, compile time, inference time, total latency, output image, and VRAM if available. Include a warm-up phase and report median/p95 measurements. The result must be measured on the actual card.

### 4. Add a real frame source
Begin with a controlled renderer/capture test and clearly track color space, format, timestamp, depth and motion availability. Measure capture and readback overhead separately from the network. Never fabricate missing buffers.

### 5. Add temporal reprojection
Use the selected model's real motion convention and previous-frame contract. Invalidate history after scene cuts, camera changes, resolution changes, dropped frames, or invalid data. Test motion, ghosting, disocclusion, and scene resets.

### 6. Add live graphics integration
Only after offline output and performance are correct, build a documented Vulkan integration path. Preview every file change, preserve originals, restore files after cancellation/failure, and verify removal. Do not claim compatibility with anti-cheat protected online games.

## Build and run

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[gpu,test]"
xess5 devices
xess5 inspect --model "models\model.onnx"
python -m pytest
```

Simple image inference:

```powershell
xess5 infer --model "models\model.onnx" --input "input.png" --output "output.png" --device ARC --layout NCHW --input-range 0:1 --output-range 0:1 --warmup 3 --iterations 10 --report "logs\run.json"
```

Multiple inputs:

```powershell
xess5 infer-tensors --model "models\render-model.onnx" --input "color=work\color.npy" --input "depth=work\depth.npy" --input "motion=work\motion.npy" --output-dir "work\outputs" --device ARC --warmup 2 --iterations 10 --report "logs\tensor-run.json"
```

The tensor command runs the model with caller-supplied arrays. Correct preprocessing and the meaning of each input remain the adapter's responsibility.

## Acceptance gates

- CPU-only tests pass on CI.
- Arc selection is unambiguous and CPU fallback is refused.
- Models and tensors that do not match the declared contract fail with useful diagnostics.
- Input files and model files are never overwritten by output.
- The generic tensor engine accepts multiple inputs and outputs, reuses compiled shapes, and reports timings.
- At least one compatible and legally usable neural model is executed on the physical Arc B580.
- A model-specific adapter produces validated neural-rendering output from real frame inputs.
- Reprojection, frame timing, and visual quality have repeatable tests.
- A live graphics integration has a rollback test before it is described as supported.
