#!/usr/bin/env python3
"""Structural validation for the TrSS multi-source GPU performance corpus.

This tool checks data integrity only; it does not rerun publisher benchmarks.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gpu-performance"
SCHEMAS = {
    "benchmarks.csv": {
        "fields": {"gpu","vendor","workload","resolution","width","height","mean_fps","statistic","game_count","upscaling","frame_generation","test_platform","source_id","source_url","source_published","retrieved_on"},
        "minimum": 132, "exact": 132, "keys": ("gpu","workload","resolution"), "value": "mean_fps",
    },
    "passmark-g3dmark.csv": {
        "fields": {"gpu","vendor","device_class","g3d_mark_score","overall_rank","metric","source_url","source_snapshot_date","retrieved_on","notes"},
        "minimum": 60, "keys": ("gpu",), "value": "g3d_mark_score",
    },
    "passmark-directcompute.csv": {
        "fields": {"gpu","vendor","device_class","average_ops_per_second","metric","source_url","source_snapshot_date","retrieved_on","notes"},
        "minimum": 70, "keys": ("gpu",), "value": "average_ops_per_second",
    },
}
RESOLUTIONS = {"1920x1080":(1920,1080), "2560x1440":(2560,1440), "3840x2160":(3840,2160)}
WORKLOADS = {"raster_native","ray_tracing_native"}

def validate_file(name: str, spec: dict) -> tuple[list[str], int]:
    path = DATA / name
    errors = []
    try:
        with path.open("r",encoding="utf-8-sig",newline="") as stream:
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or []) != spec["fields"]:
                return [f"{name}: CSV fields do not match schema"], 0
            rows = list(reader)
    except OSError as exc:
        return [f"{name}: cannot read file: {exc}"], 0

    if len(rows) < spec["minimum"]:
        errors.append(f"{name}: expected at least {spec['minimum']} rows, found {len(rows)}")
    if "exact" in spec and len(rows) != spec["exact"]:
        errors.append(f"{name}: expected exactly {spec['exact']} rows, found {len(rows)}")
    seen = set()
    for line,row in enumerate(rows,start=2):
        prefix=f"{name}:{line}"
        for field in spec["fields"]:
            if not (row.get(field) or "").strip():
                errors.append(f"{prefix}: missing {field}")
        key=tuple(row[k].strip().casefold() for k in spec["keys"])
        if key in seen:
            errors.append(f"{prefix}: duplicate key {key!r}")
        seen.add(key)
        try:
            value=float(row[spec["value"]])
            if not math.isfinite(value) or value <= 0:
                errors.append(f"{prefix}: measurement must be positive and finite")
        except (ValueError,TypeError):
            errors.append(f"{prefix}: invalid numeric measurement")
        if not row.get("source_url","").startswith("https://"):
            errors.append(f"{prefix}: source URL must use HTTPS")

    if name == "benchmarks.csv":
        if len({r["gpu"] for r in rows}) != 22:
            errors.append(f"{name}: expected 22 unique GPU models")
        for line,row in enumerate(rows,start=2):
            prefix=f"{name}:{line}"
            if row["workload"] not in WORKLOADS:
                errors.append(f"{prefix}: unsupported workload")
            if row["resolution"] not in RESOLUTIONS:
                errors.append(f"{prefix}: unsupported resolution")
            else:
                w,h=RESOLUTIONS[row["resolution"]]
                try:
                    if int(row["width"]) != w or int(row["height"]) != h:
                        errors.append(f"{prefix}: dimensions conflict with resolution")
                except ValueError:
                    errors.append(f"{prefix}: dimensions must be integers")
            try:
                count=int(row["game_count"])
                expected=11 if row["workload"]=="raster_native" else 8
                if count != expected:
                    errors.append(f"{prefix}: unexpected source suite size")
            except ValueError:
                errors.append(f"{prefix}: game_count must be an integer")
            if row["statistic"] != "geometric_mean_fps":
                errors.append(f"{prefix}: unexpected statistic")
            if row["upscaling"] != "off" or row["frame_generation"] != "off":
                errors.append(f"{prefix}: expected native rendering")
            if row["source_published"] != "2026-06-24" or row["retrieved_on"] != "2026-10-16":
                errors.append(f"{prefix}: unexpected source/retrieval date")
    return errors,len(rows)

def main() -> int:
    problems=[]
    summary=[]
    for name,spec in SCHEMAS.items():
        errors,count=validate_file(name,spec)
        problems.extend(errors)
        if not errors:
            summary.append(f"{name}: PASS ({count} records)")
    if problems:
        for problem in problems:
            print("ERROR: "+problem,file=sys.stderr)
        return 1
    print("GPU performance corpus structural validation passed.")
    for line in summary:
        print(line)
    print("Note: validates structure only; no third-party benchmark was rerun.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
