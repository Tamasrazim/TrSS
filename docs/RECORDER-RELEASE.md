# TrSS Data Recorder

This is an experimental, opt-in Windows data collection utility for the TrSS research project.

- Captures short visible-window RGB sequences as PNG.
- Saves actual frame timestamps, requested frame rate, skipped capture slots, and a limited hardware/driver profile.
- Creates a local ZIP with a sequence manifest and SHA-256 checksums.
- Does not record audio, upload data, or collect host/account identifiers.
- Does not extract game depth, motion vectors, or ground-truth reference renders. These bundles are for visual review and compatibility research and are not standalone supervised-training datasets.

Use only with content you are allowed to capture and share. Preview the selected window and inspect the bundle before sending it.
