# XESS5

**A Python-first neural inference engine targeting Intel Arc GPUs.**

XESS5 is the engine project: model loading, tensor preparation, GPU inference, timing, diagnostics, and eventually real-time frame processing. The first hardware target is the Intel Arc B580 12 GB, with support intended to extend to compatible Intel Arc GPUs.

This is an independent research project, not an Intel or NVIDIA product. It does not claim to be NVIDIA DLSS 5, and it must not label a filter or conventional upscaler as neural rendering.

## The actual objective

Build the neural-processing engine first, then connect it to real graphics frames. Do not spend the project on a mock UI or an image-viewer prototype.

The code currently provides a Python engine shell with an OpenVINO GPU backend, an explicit single-image model contract, input/output tensor conversion, device discovery, timing, and JSON reports. This is the first inference milestone; it is not yet a game renderer.

## Current engine

- **Python-first orchestration:** model contracts, input preparation, model invocation, error handling, measurements, and tests.
- **Intel GPU inference:** OpenVINO's local GPU plugin, with runtime device checks rather than an assumption that a GPU is present.
- **Local processing:** input files, models, and output images are read or written on the user's machine. No account or cloud inference is required.
- **Explicit model contract:** the first adapter accepts one rank-four RGB image input and one rank-four image output, in NCHW or NHWC layout. It supports FP16/FP32 input tensors and lets the operator set input/output value ranges.
- **No bundled weights:** the engine never pretends that model code and trained model parameters are the same thing.
- **Observable execution:** inference time, total call time, input/output dimensions, selected device, and a JSON report.
- **Main-only development:** work lands directly on main. No feature branches, pull requests, or releases.

OpenVINO documents Intel Arc GPUs as supported GPU-inference devices on Windows 10/11 and compatible Linux platforms. The graphics driver remains a separate system requirement: [OpenVINO system requirements](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html).

## Build and run on Windows

Use Python 3.10 or newer. In PowerShell from the repository folder:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[gpu,test]"
```

List the devices that OpenVINO can see:

```powershell
xess5 devices
```

Run an image-to-image model in OpenVINO-supported format (for example, ONNX or OpenVINO IR XML):

```powershell
xess5 infer --model "models\your-model.onnx" --input "input.png" --output "output.png" --device GPU --layout NCHW --input-range 0:1 --output-range 0:1 --report "logs\run.json"
```

This command expects a **compatible image-to-image model**. It is not a generic loader for every neural network: the model must follow the declared tensor contract, and the selected model's license must permit its use. Use the correct layout and numeric ranges for the model; incorrect preprocessing can produce meaningless images.

Run CPU-only tests:

```powershell
python -m pytest
```

No release build is published. The repository's CI checks source compilation and tests only.

## What is not implemented yet

1. A model-specific adapter for neural rendering that consumes scene conditioning, depth, motion vectors, noise, or temporal history.
2. A real-time capture/render path from live game frames.
3. A supported Vulkan layer or other low-overhead game integration.
4. Arc B580 hardware measurements from this repository's own inference run.
5. End-to-end latency and visual-quality comparisons against a baseline.

A still-image model invocation is useful engine infrastructure, but it is not evidence of live neural rendering or a guaranteed FPS increase.

## Research references

- [DLSS-NR on Intel](https://github.com/Uzbekunknown/dlss-nr-on-intel) — an experimental Intel Xe2 Vulkan-layer research project. Its README describes an Intel-oriented implementation, but its supplied release requires compatible model material from the user.
- [OpenDLSS-NR](https://github.com/aloshdenny/open-dlss) — a separate Vulkan reimplementation. Its documented GPU requirements are NVIDIA-specific, so it is not a direct B580 backend.
- [OpenVINO](https://github.com/openvinotoolkit/openvino) — an open-source inference runtime with an Intel GPU plugin.

These projects are references, not code or model weights silently copied into XESS5. Before adapting any component, record its exact license, preserve required notices, and validate that its hardware and model requirements fit the B580.

## Development rule

The next milestone is a reproducible neural model run on the Intel GPU, followed by a model-specific rendering adapter. Only after those work should we develop live frame capture and temporal rendering. Every feature must be tested and measured on the target hardware before it is described as supported.
