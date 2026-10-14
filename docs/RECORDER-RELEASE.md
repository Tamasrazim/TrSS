# TrSS Game Library

An opt-in Windows game discovery, launcher and telemetry collector.

- Detects known Steam and Epic installs plus selected/common game directories.
- Extracts game artwork from the executable's embedded icon and launches games from individual cards.
- Shows per-game FPS, application FPS, frametime and 1% low from bundled PresentMon ETW, plus supported CPU, memory and per-process GPU counters in an always-on-top click-through overlay.
- Collects screenshots and timestamped telemetry indefinitely until stopped, with local-only ZIP output and SHA-256 checksums.
- Does not upload data, record audio, inspect game memory, or invent depth/motion buffers.
- If Windows denies ETW access, FPS and frametime remain unavailable rather than being inferred from refresh rate or screenshot rate.

The portable build bundles PresentMon 2.6.0 as a hidden telemetry component. Its MIT license and pinned-binary checksum are included in PRESENTMON-NOTICE.txt. Users running from source should run scripts/fetch_presentmon.ps1 once to download and verify the official binary.

Use only with games and displayed content you are allowed to capture and share. Review the ZIP before sending it to anyone.
