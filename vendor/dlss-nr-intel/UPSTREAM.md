# Vendored Intel Xe2 XMX inference backend

This directory contains the runtime/build subset of
[Uzbekunknown/dlss-nr-on-intel](https://github.com/Uzbekunknown/dlss-nr-on-intel),
pinned to upstream commit `d29b6652940ae120c0152f0b7d6e50109846f4e3`
(2026-10-12).

The upstream project's Apache License 2.0 and NOTICE are retained here.
The vendored code is pinned to the commit above. Most upstream files are kept
unchanged; `src/gpu/libxmx.c` has one XESS5-specific change in Vulkan adapter
selection, marked by a nearby comment. That change requires one unique Arc-name
match by default, or one unique match for the optional
`XESS5_VULKAN_DEVICE` substring. It fails instead of silently selecting the
first available GPU when the choice is ambiguous or no Arc-named adapter exists.
First-party model and frame integration remains in the `xess5/` package.

## Why this source is present

These Python orchestrators, GLSL compute shaders, and C/Vulkan runtime are an
existing Intel Xe2 XMX execution path that XESS5 can integrate with instead of
trying to reproduce all of the hardware-specific GEMM and attention kernels.
This is source code, not a separately installed graphics application wrapper.

The native backend is intentionally isolated under this directory so it can be
updated and audited against a specific upstream commit. Keep required upstream
notices when modifying or distributing it.

## Not included

No NVIDIA DLL, model weights, extracted parameters, game files, prebuilt binary,
or compiled SPIR-V shader is included. A compatible logical-weight file must be
provided locally and the native runtime/shaders must be built for the installed
Vulkan toolchain. XESS5 must check the actual Vulkan device name before running
the model and report that selection transparently.

## Build boundary

The upstream CMake project compiles the native `libxmx` Vulkan runtime,
required compute shaders, and optional host image helpers. On Windows, use a
supported GCC/Clang toolchain with `_Float16`, Vulkan headers/loader, and
`glslangValidator`; upstream CMake rejects MSVC. The game layer is not included
in the XESS5 build path here; this vendored subset is for direct frame inference.
