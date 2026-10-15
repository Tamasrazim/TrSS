# TrSS Game Library / Data Recorder

An opt-in Windows game discovery, launcher and telemetry collector.

- Detects known Steam and Epic installs plus selected/common game directories.
- Extracts game artwork from the executable's embedded icon and launches games from individual cards.
- Shows supported CPU, memory and per-process GPU counters in an always-on-top click-through overlay. FPS/frametime fields are reserved for TrSS's first-party frame-event collector, which is still in development.
- Collects visible-window PNG samples and timestamped telemetry until stopped, with local-only ZIP output. Game Library captures include `performance.jsonl` for available Windows counters; sample rate is not game FPS.
- Verifies archive CRCs, safe ZIP paths, complete SHA-256 manifest coverage, and every file digest before deleting a retained session folder. Packaging failures keep the capture folder for recovery.
- Does not upload data, record audio, inspect game memory, or invent depth/motion buffers.
- Does not bundle or launch a third-party FPS telemetry tool.

Use only with games and displayed content you are allowed to capture and share. Review the ZIP before sending it to anyone.
