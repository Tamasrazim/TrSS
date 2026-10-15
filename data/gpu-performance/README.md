# TrSS GPU-performance seed dataset

This directory contains a **curated seed of 132 aggregate benchmark measurements** (22 GPUs × 2 workloads × 3 output resolutions) transcribed from Tom's Hardware's 2026 GPU hierarchy.

## What this dataset is—and is not

- It is useful for a first GPU-performance profile, workload comparison, runtime policy research, and checking whether a performance-estimation baseline is plausible.
- FPS values are geometric means across a multi-game suite, **not per-game measurements**. Raster values aggregate 11 games; ray-tracing values aggregate 8 RT games.
- Results are native-resolution, with upscaling and frame generation off. The article uses an AMD Ryzen 7 9800X3D testbed and a mix of high/ultra settings. Exact API and per-game settings vary.
- The initial file contains a representative subset of GPUs, not the full source hierarchy. Values are transcribed from the tables and should be checked against the source before being treated as a release-grade corpus.
- **This is not a valid training dataset for TrSS's image-reconstruction neural network.** FPS numbers contain no paired image target, depth, motion vectors, or history buffers. Do not feed these rows to the super-resolution training loader. Keep this corpus separate for hardware/performance modeling and runtime decisions.

## Files

- `benchmarks.csv`: long-form measurements. Each row is one GPU × workload × resolution.
- `provenance.json`: source and collection notes.
- `../../scripts/validate_gpu_performance.py`: dependency-free integrity checker.

## Schema

| Field | Meaning |
|---|---|
| `gpu`, `vendor` | Product label as shown in the source and vendor |
| `workload` | `raster_native` or `ray_tracing_native` |
| `resolution`, `width`, `height` | Render resolution |
| `mean_fps` | Published suite geometric-mean FPS |
| `statistic`, `game_count` | Aggregation type and number of games |
| `upscaling`, `frame_generation` | Both disabled for these results |
| `test_platform` | Source testbed/context; API is not assumed to be uniform |
| `source_id`, `source_url`, `source_published`, `retrieved_on` | Traceability fields |

## Validate

Run from the repository root:

```bash
python scripts/validate_gpu_performance.py
```

The checker validates schema, numeric ranges, expected source metadata, and duplicate keys. It does not claim to reproduce the publisher's tests.

## Next expansion

1. Add individual per-game results only when the source publishes exact FPS and settings.
2. Add independent sources such as Phoronix, keeping each OS, driver, API, CPU, benchmark, and test date distinct—never merge incomparable numbers into one average.
3. Add driver versions, GPU power, VRAM usage, frame-time percentiles, and 1% lows when explicitly published.
4. Split evaluation by source/test run, not by adjacent rows from the same suite.
5. Only then train and evaluate a performance predictor against held-out measurements.

## Source attribution

- Tom's Hardware, *The GPU benchmarks hierarchy 2026*, last updated 2026-06-24: https://www.tomshardware.com/reviews/gpu-hierarchy,4388.html
- Phoronix, *Intel Arc Graphics B570 & B580 Gaming Performance On Linux For Mid 2025*: https://www.phoronix.com/review/intel-battlemage-linux-may2025 (research reference for future Linux-specific expansion; its values are not mixed into this Windows-oriented seed table).

The data are facts/measurements with source URLs retained. Refer to the publisher's terms before redistributing a larger transcription or automated copy of its tables.
