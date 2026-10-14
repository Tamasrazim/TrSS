"""Compare ordinary-game recordings with TrSS-enabled recordings.

This is a research comparison tool, not an injector and not a PyTorch training
loop. It compares already-recorded metrics only when game, experiment group, and
graphics-settings fingerprint match. The recorder never needs to connect to the
runtime, and missing FPS values stay missing.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

SCHEMA = "trss.performance-comparison/1.0"
METRICS = (
    "average_fps",
    "average_frametime_ms",
    "average_game_cpu_percent",
    "average_system_cpu_percent",
    "average_game_gpu_3d_percent",
    "average_game_dedicated_gpu_bytes",
    "average_game_ram_bytes",
    "average_ram_used_bytes",
    "average_ram_total_bytes",
    "average_ram_percent",
)
DISPLAY_NAMES = {
    "average_fps": "Game FPS",
    "average_frametime_ms": "Game frametime (ms)",
    "average_game_cpu_percent": "Game CPU (%)",
    "average_system_cpu_percent": "System CPU (%)",
    "average_game_gpu_3d_percent": "Game GPU 3D (%)",
    "average_game_dedicated_gpu_bytes": "Game dedicated VRAM (bytes)",
    "average_game_ram_bytes": "Game RAM (bytes)",
    "average_ram_used_bytes": "System RAM used (bytes)",
    "average_ram_total_bytes": "System RAM total (bytes)",
    "average_ram_percent": "System RAM load (%)",
}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_matrix(path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read a JSONL game matrix; preserve valid rows if an unrelated line is damaged."""
    source = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                warnings.append(f"Line {line_number}: invalid JSON was skipped.")
                continue
            if isinstance(value, dict) and value.get("schema_version") == "trss.graphics-matrix/1.0":
                rows.append(value)
            else:
                warnings.append(f"Line {line_number}: not a TrSS graphics-matrix row; skipped.")
    if not rows:
        warnings.append("No valid TrSS game-matrix rows were found.")
    return rows, warnings


def _row_matches_profile(row: dict[str, Any], requested: str) -> bool:
    return _norm(row.get("profile_name")) == _norm(requested)


def _profile_exact(row: dict[str, Any]) -> bool:
    return bool(row.get("matrix_row_exact") and row.get("profile_verified_in_game"))


def _metric_values(rows: Iterable[dict[str, Any]], metric: str) -> tuple[float | None, int, int]:
    values: list[float] = []
    captured_samples = 0
    valid_rows = 0
    for row in rows:
        value = _finite_number((row.get("performance_summary") or {}).get(metric))
        if value is None:
            continue
        values.append(value)
        valid_rows += 1
        count_key = "sample_count_" + metric.removeprefix("average_")
        raw_count = _finite_number((row.get("performance_summary") or {}).get(count_key))
        if raw_count is not None and raw_count > 0:
            captured_samples += int(raw_count)
    return (mean(values) if values else None), len(values), captured_samples


def compare_matrix_rows(
    rows: list[dict[str, Any]],
    *,
    baseline_profile: str = "Baseline",
    trss_profile: str = "TrSS",
    require_exact: bool = False,
) -> dict[str, Any]:
    """Pair runs by game + experiment group + identical settings fingerprint."""
    buckets: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = {}
    label_map: dict[tuple[str, str], tuple[str, str]] = {}
    skipped_rows: list[dict[str, Any]] = []

    for row in rows:
        profile = row.get("profile_name")
        is_baseline = _row_matches_profile(row, baseline_profile)
        is_trss = _row_matches_profile(row, trss_profile)
        if not (is_baseline or is_trss):
            continue
        game = str(row.get("game") or "").strip()
        group = str(row.get("experiment_group") or "").strip()
        settings_hash = str(row.get("settings_hash_sha256") or "").strip().casefold()
        if not game or not group or not settings_hash:
            skipped_rows.append({
                "session_id": row.get("session_id"),
                "game": game,
                "experiment_group": group,
                "profile_name": profile,
                "reason": "game, experiment group, or settings fingerprint is missing",
            })
            continue
        if require_exact and not _profile_exact(row):
            skipped_rows.append({
                "session_id": row.get("session_id"),
                "game": game,
                "experiment_group": group,
                "profile_name": profile,
                "reason": "profile is not marked exact and verified in-game",
            })
            continue
        key = (_norm(game), _norm(group), settings_hash)
        mode = "baseline" if is_baseline else "trss"
        buckets.setdefault(key, {"baseline": [], "trss": []})[mode].append(row)
        label_map[(_norm(game), _norm(group))] = (game, group)

    comparisons: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = list(skipped_rows)

    for key, modes in sorted(buckets.items()):
        baseline_rows = modes["baseline"]
        trss_rows = modes["trss"]
        game, group = label_map[(key[0], key[1])]
        if not baseline_rows or not trss_rows:
            unmatched.append({
                "game": game,
                "experiment_group": group,
                "settings_hash_sha256": key[2],
                "baseline_runs": len(baseline_rows),
                "trss_runs": len(trss_rows),
                "reason": "no baseline row" if not baseline_rows else "no TrSS row with the same settings fingerprint",
            })
            continue

        exact_pair = all(_profile_exact(row) for row in (*baseline_rows, *trss_rows))
        for metric in METRICS:
            baseline_value, baseline_valid_rows, baseline_samples = _metric_values(baseline_rows, metric)
            trss_value, trss_valid_rows, trss_samples = _metric_values(trss_rows, metric)
            if baseline_value is None or trss_value is None:
                # FPS and frametime intentionally remain absent until TrSS has an
                # independent, first-party source for measuring the actual game.
                continue
            delta = trss_value - baseline_value
            percentage = (delta / abs(baseline_value) * 100.0) if baseline_value != 0 else None
            comparisons.append({
                "game": game,
                "experiment_group": group,
                "settings_hash_sha256": key[2],
                "metric": metric,
                "metric_label": DISPLAY_NAMES.get(metric, metric),
                "baseline_profile": baseline_profile,
                "trss_profile": trss_profile,
                "baseline_runs_total": len(baseline_rows),
                "trss_runs_total": len(trss_rows),
                "baseline_runs_with_metric": baseline_valid_rows,
                "trss_runs_with_metric": trss_valid_rows,
                "baseline_samples_total": baseline_samples,
                "trss_samples_total": trss_samples,
                "baseline_mean": baseline_value,
                "trss_mean": trss_value,
                "absolute_delta_trss_minus_baseline": delta,
                "percent_delta_trss_vs_baseline": percentage,
                "settings_match": True,
                "profiles_exact_and_verified": exact_pair,
                "interpretation": "descriptive change only; not causal proof",
            })

    return {
        "schema_version": SCHEMA,
        "comparison_kind": "independent baseline game recording vs TrSS-enabled game recording",
        "baseline_profile": baseline_profile,
        "trss_profile": trss_profile,
        "require_exact_profiles": require_exact,
        "paired_comparison_groups": len({
            (item["game"], item["experiment_group"], item["settings_hash_sha256"])
            for item in comparisons
        }),
        "comparison_rows": comparisons,
        "unmatched_or_skipped_rows": unmatched,
        "limitations": [
            "The game recorder is independent of the TrSS renderer; no live connection is required.",
            "A pair is made only when the game, experiment group, and standardized settings SHA-256 match.",
            "FPS and frametime are omitted when either mode lacks real measured values; screenshot rate is not game FPS.",
            "Each value is an average of per-run averages, giving each run equal weight.",
            "Differences describe recorded telemetry but do not prove TrSS caused the change; scene, driver, temperature, and run order still matter.",
            "Performance logs are evaluation evidence, not supervised image-training samples. Neural reconstruction training still requires properly aligned reference frames and real depth/motion/history inputs.",
        ],
    }


def write_comparison(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "performance-comparison.json"
    csv_path = directory / "performance-comparison.csv"
    temporary_json = json_path.with_suffix(".json.tmp")
    temporary_csv = csv_path.with_suffix(".csv.tmp")

    temporary_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    columns = [
        "game", "experiment_group", "settings_hash_sha256", "metric", "metric_label",
        "baseline_profile", "trss_profile", "baseline_runs_total", "trss_runs_total",
        "baseline_runs_with_metric", "trss_runs_with_metric", "baseline_samples_total",
        "trss_samples_total", "baseline_mean", "trss_mean",
        "absolute_delta_trss_minus_baseline", "percent_delta_trss_vs_baseline",
        "settings_match", "profiles_exact_and_verified", "interpretation",
    ]
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["comparison_rows"])
    temporary_json.replace(json_path)
    temporary_csv.replace(csv_path)
    return json_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare TrSS Game Library baseline and TrSS-enabled game recordings."
    )
    parser.add_argument("--matrix", required=True, type=Path, help="Path to game-matrix.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder for comparison JSON and CSV")
    parser.add_argument("--baseline-profile", default="Baseline", help='Exact profile label for the unmodified-game run (default: "Baseline")')
    parser.add_argument("--trss-profile", default="TrSS", help='Exact profile label for the TrSS-enabled run (default: "TrSS")')
    parser.add_argument("--require-exact", action="store_true", help="Ignore profiles that are not marked exact and verified in-game")
    args = parser.parse_args()

    rows, warnings = load_matrix(args.matrix)
    report = compare_matrix_rows(
        rows,
        baseline_profile=args.baseline_profile,
        trss_profile=args.trss_profile,
        require_exact=args.require_exact,
    )
    report["input_matrix"] = str(args.matrix.expanduser().resolve())
    report["warnings"] = warnings
    json_path, csv_path = write_comparison(report, args.output_dir)
    print(json.dumps({
        "schema_version": report["schema_version"],
        "paired_comparison_groups": report["paired_comparison_groups"],
        "comparison_rows": len(report["comparison_rows"]),
        "unmatched_or_skipped_rows": len(report["unmatched_or_skipped_rows"]),
        "json_report": str(json_path),
        "csv_report": str(csv_path),
        "warnings": warnings,
    }, indent=2, ensure_ascii=False))
    return 0 if report["comparison_rows"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
