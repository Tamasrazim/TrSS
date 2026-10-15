#!/usr/bin/env python3
"""Validate the curated TrSS GPU performance benchmark CSV without third-party packages."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gpu-performance" / "benchmarks.csv"
EXPECTED_FIELDS = {
    "gpu", "vendor", "workload", "resolution", "width", "height", "mean_fps",
    "statistic", "game_count", "upscaling", "frame_generation", "test_platform",
    "source_id", "source_url", "source_published", "retrieved_on",
}
VALID_RESOLUTIONS = {"1920x1080": (1920, 1080), "2560x1440": (2560, 1440), "3840x2160": (3840, 2160)}
VALID_WORKLOADS = {"raster_native", "ray_tracing_native"}
EXPECTED_ROWS = 132


def validate(path: Path = DATA) -> list[str]:
    errors: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    count = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if set(reader.fieldnames or []) != EXPECTED_FIELDS:
                return [f"CSV columns do not match schema: {reader.fieldnames!r}"]
            for line, row in enumerate(reader, start=2):
                count += 1
                prefix = f"line {line}"
                for field in EXPECTED_FIELDS:
                    if not (row.get(field) or "").strip():
                        errors.append(f"{prefix}: missing {field}")
                key = (row["gpu"], row["workload"], row["resolution"])
                if key in seen:
                    errors.append(f"{prefix}: duplicate GPU/workload/resolution key {key!r}")
                seen.add(key)
                if row["workload"] not in VALID_WORKLOADS:
                    errors.append(f"{prefix}: unsupported workload {row['workload']!r}")
                if row["resolution"] not in VALID_RESOLUTIONS:
                    errors.append(f"{prefix}: unsupported resolution {row['resolution']!r}")
                else:
                    expected_width, expected_height = VALID_RESOLUTIONS[row["resolution"]]
                    try:
                        if int(row["width"]) != expected_width or int(row["height"]) != expected_height:
                            errors.append(f"{prefix}: resolution does not match width/height")
                    except ValueError:
                        errors.append(f"{prefix}: width/height must be integers")
                try:
                    fps = float(row["mean_fps"])
                    if not math.isfinite(fps) or fps <= 0 or fps > 1000:
                        errors.append(f"{prefix}: mean_fps must be finite and in (0, 1000]")
                except ValueError:
                    errors.append(f"{prefix}: mean_fps must be numeric")
                try:
                    games = int(row["game_count"])
                    expected_games = 11 if row["workload"] == "raster_native" else 8
                    if games != expected_games:
                        errors.append(f"{prefix}: game_count should be {expected_games} for this source suite")
                except ValueError:
                    errors.append(f"{prefix}: game_count must be an integer")
                if row["statistic"] != "geometric_mean_fps":
                    errors.append(f"{prefix}: unexpected statistic")
                if row["upscaling"] != "off" or row["frame_generation"] != "off":
                    errors.append(f"{prefix}: this seed corpus must represent native rendering only")
                if not row["source_url"].startswith("https://"):
                    errors.append(f"{prefix}: source_url must be HTTPS")
                if row["source_published"] != "2026-06-24" or row["retrieved_on"] != "2026-10-16":
                    errors.append(f"{prefix}: source dates do not match dataset provenance")
    except OSError as exc:
        return [f"cannot read {path}: {exc}"]
    if count != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} measurements, found {count}")
    if len(seen) != count:
        errors.append("duplicate keys detected")
    if not errors:
        print(f"PASS: {count} measurements, {len({key[0] for key in seen})} GPUs, {len(VALID_WORKLOADS)} workloads, {len(VALID_RESOLUTIONS)} resolutions")
    return errors


def main() -> int:
    errors = validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
