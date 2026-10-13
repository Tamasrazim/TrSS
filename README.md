# XESS5

**An independent, local-first neural graphics renderer for Intel Arc.**

XESS5 is the working name for a native Windows graphics project aimed at testing and developing neural-rendering techniques on Intel Arc hardware, especially the Arc B580 12 GB. It is not an Intel product, is not affiliated with Intel or NVIDIA, and does not claim to implement NVIDIA DLSS 5.

## Goal

Build one application we control: a clear, measurable way to inspect and compare original and processed frames, develop GPU-accelerated image reconstruction and neural-rendering paths, and work toward supported live-game integration. No account, cloud inference, subscription, or separate third-party wrapper application.

The target is real rendered imagery, not a mock interface. Visual quality, frame time, stability, and reversibility must be measured.

## What matters first

1. **A real comparison path.** Load a local image or captured frame, compare original and processed output side-by-side and with a movable divider, and save both outputs for inspection.
2. **A GPU execution path.** Use hardware-supported graphics and compute APIs. The Arc B580 is the initial target; performance is measured on the hardware rather than guessed.
3. **A replaceable model backend.** Keep the renderer separate from model loading so a properly licensed model can be selected without rewriting the interface or graphics path.
4. **Evidence before claims.** Report input size, processing time, frame rate where measurable, model identity, settings, and errors. Never label a conventional shader as neural rendering.
5. **Safe game integration.** Add live integration only for a documented graphics path, with clear install/remove behavior and no silent replacement of original game files. Do not use this with anti-cheat protected online play.

## Reality checks

- DLSS Super Resolution, frame generation, ray reconstruction, and neural rendering are different workloads. One does not imply the others.
- An open-source implementation does not automatically include model weights or grant rights to proprietary NVIDIA files.
- XESS5 will use only source code, model weights, datasets, and assets whose licenses permit the intended use. Each dependency must retain its notices.
- A working first rendering path does not guarantee higher FPS. Neural processing consumes GPU time, memory bandwidth, and VRAM.
- Intel Arc B580 support must be verified by running and measuring the application on that card.

## Development direction

The main build is organized around these components:

- **Application shell:** native Windows UI, settings, logs, and original-versus-processed comparison.
- **Frame input:** local image first, followed by captured frames and selected live graphics paths.
- **Render backend:** GPU image processing and buffer management, with explicit capability checks.
- **Model backend:** loads a supported model only after its format, license, and hardware requirements have been verified.
- **Diagnostics:** frame timing, GPU/backend information, model status, logs, and exportable reports.
- **Integration layer:** isolated from the UI and reversible; each supported API gets a separate tested path.

## Current status

The repository has just been initialized. Core implementation, model selection, builds, and hardware validation are **not yet complete**. A feature is only marked supported after its code and a reproducible test exist.

## Links

- [Intel-specific DLSS-NR research](https://github.com/Uzbekunknown/dlss-nr-on-intel) — useful research reference; experimental and requires separately supplied model material.
- [OpenDLSS-NR](https://github.com/aloshdenny/open-dlss) — a separate implementation whose documented GPU requirements do not make it a direct Arc B580 backend.

These are research references, not dependencies automatically copied into XESS5.

## License and third-party notices

No project-wide reuse license has been granted yet. Until a license is added, do not assume the XESS5 source may be redistributed or reused. Any subsequently adopted third-party component must be recorded in a dependency/license inventory before it is integrated.
