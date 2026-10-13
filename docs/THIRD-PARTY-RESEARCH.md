# Neural-rendering research references and licensing boundaries

XESS5 is an independent engine project. These references help document the public architecture and the hardware-inference options; they are not evidence that XESS5 includes a compatible production model or that every upstream component can be redistributed together.

## DLSS-NR on Intel Xe2

Repository: https://github.com/Uzbekunknown/dlss-nr-on-intel  
License: Apache-2.0, with third-party notices in the upstream repository's `LICENSE` and `NOTICE`.

The project describes itself as an independent reimplementation of a neural-rendering inference pass for Intel Xe2. Its published architecture document describes a 16-channel model input, a four-channel output head, deterministic noise, scaled current/previous frame data, control scalars, residual composition, and temporal blending. It also explicitly states that it does not distribute NVIDIA model weights or the vendor DLL.

XESS5 currently implements an independent frame-contract adapter for that documented 16-to-4 channel shape, plus deterministic noise, field alignment, FP16-rounded color scaling, residual/gate composition, and an OpenVINO inference wrapper. This is an adapter around a user-supplied compatible model file; it is not a copy of the upstream graph or weights, and it does not download or extract vendor files.

## MLX-DLSS

Repository: https://github.com/iamwavecut/MLX-DLSS  
License: Apache-2.0 in its repository.

Its published Python feature contract was consulted to check the shape and intent of the input feature assembly, including its 16-channel semantics, mirror extension, deterministic noise, RGB scaling, and four-channel composition. XESS5's `xess5/rendering/hnet.py` is written as a separate implementation and does not import this repository at runtime.

## OpenDLSS-NR

Repository: https://github.com/maanHimself/OpenDLSS-NR

The Intel-targeted research documentation cites this work for parts of the recovered graph and frame geometry. XESS5 does not vendor its code. Before copying or adapting any exact source, shader, test, or model artifact, record and comply with the exact upstream license and attribution requirements.

## OpenVINO

Repository: https://github.com/openvinotoolkit/openvino  
Documentation: https://docs.openvino.ai/

OpenVINO is the current model-inference runtime used by the generic engine. It supports model loading and GPU compilation where the local installed runtime, plugin, device, and driver support the requested graph. The generic runtime does not make a model's input semantics known, nor does it supply model weights.

## Important separation of concerns

- **Code is not a trained model.** None of the above repositories' existence means model parameters are included in XESS5.
- **Public graph documentation is not vendor source code.** XESS5 does not bundle or reconstruct a proprietary NVIDIA binary.
- **A compatible tensor shape is not proof of functional equivalence.** Exact feature assembly, parameter mapping, weight precision, head composition, temporal reprojection, and post-processing must match the specific usable model.
- **An OpenVINO GPU report is not proof of Arc B580 performance.** Hardware execution still needs to be measured on the physical GPU.
- **Offline frame processing is not game integration.** XESS5 currently receives local image and optional tensor files; it does not yet hook into a live graphics swapchain.
- **Compatibility is not implied.** This project does not claim affiliation with or official support from Intel or NVIDIA, and does not claim official DLSS 5 compatibility.

Before any upstream code is copied, make a per-file license inventory and preserve required attribution and notices. Model files must be obtained only from sources whose terms permit their intended use.
