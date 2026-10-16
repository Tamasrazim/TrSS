# TrSS GPU performance corpus

This directory stores attributed measurements for a **separate GPU-performance estimation/runtime-policy track**.

## Corpus size

- `benchmarks.csv`: 132 native gaming FPS measurements across 22 GPUs, 2 workloads, and 3 resolutions.
- `passmark-g3dmark.csv`: 71 PassMark G3D Mark records across different GPU categories.
- `passmark-directcompute.csv`: 89 PassMark GPU compute records.
- **292 measurements** across three distinct schemas.

## Sources and measurements

### Gaming performance (FPS)

Tom's Hardware's 2026 GPU hierarchy provides geometric-mean native FPS for rasterization and ray tracing at 1080p, 1440p, and 4K. The source describes 11 raster games and 8 RT games, with upscaling and frame generation disabled. Exact APIs/settings vary by game.

Source: https://www.tomshardware.com/reviews/gpu-hierarchy,4388.html (source last updated 2026-06-24).

### G3D Mark score

PassMark's public GPU list contains aggregate G3D Mark scores and chart ranks. These are based on submitted PerformanceTest results; configurations are not controlled like a single lab's GPU review. They are not FPS.

Source: https://www.videocardbenchmark.net/gpu_list.php (snapshot date: 2026-10-16; the chart changes over time).

### GPU compute

PassMark's DirectCompute page publishes average operations/second. This is a separate compute benchmark and must not be treated as game FPS or a TrSS neural-reconstruction-kernel measurement.

Source: https://www.videocardbenchmark.net/directCompute.html (snapshot date: 2026-10-16; the chart changes over time).

## Validation

From the repository root, run:

```bash
python scripts/validate_gpu_performance.py
```

This checks schema, row counts, unique keys, positive finite measurements, HTTPS source URLs, and native-resolution metadata. It does not independently reproduce third-party benchmarks.

## How this helps TrSS

Keep these measurements in separate target tasks. FPS, G3D Mark and operations/second have different units and test methodologies; a model must not mix them as one target. A future performance predictor should use documented features such as GPU model/class, VRAM, architecture, bandwidth, TDP, driver, API, resolution and graphics settings, then be evaluated on held-out GPUs/test runs.

**These performance tables do not train the TrSS image-reconstruction model.** `TrSSTemporalSR` requires paired low-resolution and reference high-resolution images, documented depth, motion vectors and valid temporal history. Benchmark values contain none of those tensors. This data is for the separate performance-estimation track.

## Provenance

See `provenance.json`. Values remain attributed to their publishers. Confirm publisher terms before redistributing larger or automated transcriptions.
