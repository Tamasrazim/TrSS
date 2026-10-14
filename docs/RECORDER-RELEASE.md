# TrSS Game Library

An opt-in Windows game discovery, launcher and telemetry collector.

- Detects known Steam and Epic installs plus selected/common game directories.
- Extracts game artwork from the executable's embedded icon and launches games from individual cards.
- Shows supported CPU, memory and per-process GPU counters in an always-on-top click-through overlay. FPS/frametime fields are reserved for TrSS's first-party frame-event collector, which is still in development.
- Collects screenshots and timestamped telemetry indefinitely until stopped, with local-only ZIP output and SHA-256 checksums.
- Does not upload data, record audio, inspect game memory, or invent depth/motion buffers.
- Does not bundle or launch a third-party FPS telemetry tool.

Use only with games and displayed content you are allowed to capture and share. Review the ZIP before sending it to anyone.
