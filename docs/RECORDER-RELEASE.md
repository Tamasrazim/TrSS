# TrSS Game Library / Data Recorder

An opt-in Windows game discovery, launcher and telemetry collector.

- Detects known Steam and Epic installs plus selected/common game directories.
- Extracts game artwork from the executable's embedded icon and launches games from individual cards.
- Shows supported CPU, memory and per-process GPU counters in an always-on-top click-through overlay. FPS/frametime fields are reserved for TrSS's first-party frame-event collector, which is still in development.
- The Game Library records visible-window PNG samples and available Windows counters until stopped, with automatic focus-loss and low-disk safety stops. The standalone Data Recorder offers preset sample-rate and duration controls.
- The Game Library writes its timestamped measurements to `performance.jsonl`; screenshot sample rate is not game FPS, and unsupported FPS/frametime values are never invented.
- Both capture tools create local ZIP bundles. Bundle creation verifies archive CRCs, safe ZIP paths, complete SHA-256 manifest coverage, and every file digest before deleting an unpacked session folder. Packaging failures keep the capture folder for recovery.
- Choose a different storage drive in the Game Library's **Capture folder** control; the chosen destination is saved locally between launches.
- Does not upload data, record audio, inspect game memory, or invent depth/motion buffers.
- Does not bundle or launch a third-party FPS telemetry tool.

Use only with games and displayed content you are allowed to capture and share. Review the ZIP before sending it to anyone.
