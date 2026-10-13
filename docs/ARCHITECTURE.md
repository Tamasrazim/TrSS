# XESS5 engine architecture

XESS5 is a Python-first neural graphics engine targeting Intel Arc. The Arc B580 12 GB is the first validation target. Python owns orchestration, explicit tensor contracts, model execution, input validation, temporal state, diagnostics, and tests. A pinned Intel Xe2 Vulkan/XMX research backend is vendored under `vendor/dlss-nr-intel/` and connected through the experimental offline `xmx-frame` command. The native route is not yet hardware-validated, and the first-party live game-capture hook remains unimplemented. This project is an engine, not a separate UI prototype.

## Components in the repository

- `xess5/backends/openvino_backend.py`: local device inventory and strict Arc/explicit GPU selection. No silent CPU fallback.
- `xess5/contracts.py`: normalized-range parsing and image-to-tensor/tensor-to-image conversion, with explicit NCHW/NHWC and RGB/BGR handling.
- `xess5/engine.py`: single-image convenience engine, model inspection, shape validation, GPU compilation, output conversion, and timings.
- `xess5/tensor_engine.py`: generic multi-input/multi-output tensor engine. Named inputs are validated, dynamic tensor shapes are specialized, compilation is reused for repeated shapes, and copied outputs are reported with compile/inference/total timings.
- `xess5/rendering/frame.py`: a validated `FramePacket` for color and optional real auxiliary buffers, plus temporal output history with reset rules.
- `xess5/cli.py`: `devices`, `inspect`, `infer`, `infer-tensors`, `render-frame`, `hnet-frame`, and experimental `xmx-frame`.
- `xess5/backends/xmx_engine.py`: optional bridge for loading local logical weights, selecting an Arc-named Vulkan device, retaining device-side weights/frame buffers, and running the native 16-to-4 frame graph.
- `vendor/dlss-nr-intel/`: pinned Apache-2.0 Vulkan/XMX runtime, shaders, and frame-runner source. The native C file has a fail-closed device-selection change and is compile-checked in CI.
- `tests/`: CPU-only tests for conversions, model contracts, named multi-input execution, adapter selection, history invalidation, and the XMX bridge with a fake native frame runner. CI does not build all SPIR-V shaders or execute the native backend on a GPU.

## GPU selection

`xess5 devices` reports the actual IDs and names OpenVINO exposes. `--device ARC` is intentionally strict: it selects an Intel Arc-named GPU only if that is unambiguous. When multiple devices identify as Arc, pass an exact ID such as `GPU.1`. CPU devices are not accepted as substitutes for a GPU. This prevents a hybrid system from accidentally running on the integrated adapter or CPU while presenting the result as B580 performance.

The separate vendored Vulkan runtime applies the same fail-closed principle. With no override, it requires exactly one Vulkan adapter name containing `Arc`. Set `XESS5_VULKAN_DEVICE` to a more specific name substring only when that substring uniquely identifies the intended adapter. This selection code is source-level hardening; physical Vulkan startup is not yet part of Python CI.

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


## Specialized HNet-style frame adapter

`xess5/rendering/hnet.py` adds a separate, explicit pre/post-processing path around a supplied model. It was implemented from the public frame contract documented by the Intel Xe2 research project and its cited Apache-2.0 feature-contract reference; the project records the research and license boundaries in [THIRD-PARTY-RESEARCH.md](THIRD-PARTY-RESEARCH.md).

The adapter performs:

1. Input validation for RGB float values in the range 0..1.
2. Deterministic three-channel noise generation from network pixel coordinates and a frame index.
3. Network-field alignment and mirror extension.
4. Three-step FP16 color scaling, preserving the separate rounding points.
5. A 16-channel feature tensor, in HWC before conversion to NCHW:
   - channels 0-2 deterministic noise;
   - channel 3 constant one;
   - channels 4-6 current color at the model's scaled representation;
   - channels 7-9 previous color/prediction at the same representation (current color on the first frame);
   - channel 10 normalized style index;
   - channel 11 local tone strength;
   - channel 12 local structure strength;
   - channels 13-14 set to -1 for the disabled optional masks;
   - channel 15 zero.
6. Strict model checks requiring one NCHW 16-channel input and one four-channel output head, NCHW or NHWC.
7. Four-channel composition: rounded RGB residual plus a sigmoid temporal gate blended with the supplied prior history; the adapter returns a normalized RGB output and the next history buffer.

### Important current limits

This is an independently implemented contract adapter, not the 71-block trained network. It does not embed or recover vendor weights, supply the actual model graph, reproduce every vendor style grade/mask or intensity control, or reproject history using engine motion vectors. The first implementation uses identity reprojection and only the explicitly implemented head composition. It must not be treated as a bit-exact vendor clone.

To test the adapter, supply an otherwise compatible, legally usable ONNX/OpenVINO model file with a one-input/one-output contract. The network-field size has to match the model's static dimensions when those are fixed. A mismatched graph stops with a shape error before the result is described as supported.

CLI:

```powershell
xess5 hnet-frame --model "models\model.onnx" --color "frames\frame.png" --output "work\rendered.png" --device ARC --output-layout NCHW --frame-index 0 --warmup 1 --iterations 5 --history-output "work\next-history.npy" --head-output "work\head.npy" --report "logs\hnet.json"
```

The command reports the requested and resolved GPU ID, device name, network dimensions, preprocessing/model timing, and output paths. All inference inputs and model files stay local. No actual Arc B580 run, live-game capture, Vulkan hook, or in-game performance result has been verified by CPU-only CI.
