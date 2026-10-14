"""Tests for matched baseline-versus-TrSS performance comparison."""
import csv
import json
from pathlib import Path

from trss.performance_compare import (
    compare_matrix_rows,
    load_matrix,
    write_comparison,
)


def _row(
    *,
    session: str,
    profile: str,
    settings_hash: str = "a" * 64,
    group: str = "same-scene",
    exact: bool = True,
    cpu: float | None = None,
    gpu: float | None = None,
    fps: float | None = None,
) -> dict:
    return {
        "schema_version": "trss.graphics-matrix/1.0",
        "session_id": session,
        "game": "Example Game",
        "platform": "Steam",
        "profile_name": profile,
        "experiment_group": group,
        "settings_hash_sha256": settings_hash,
        "matrix_row_exact": exact,
        "profile_verified_in_game": exact,
        "performance_summary": {
            "average_game_cpu_percent": cpu,
            "sample_count_game_cpu_percent": 4 if cpu is not None else 0,
            "average_game_gpu_3d_percent": gpu,
            "sample_count_game_gpu_3d_percent": 4 if gpu is not None else 0,
            "average_fps": fps,
            "sample_count_fps": 0 if fps is None else 4,
        },
    }


def test_comparison_pairs_only_when_game_group_and_settings_hash_match() -> None:
    rows = [
        _row(session="base-1", profile="Baseline", cpu=30.0, gpu=65.0),
        _row(session="trss-1", profile="TrSS", cpu=34.0, gpu=75.0),
        _row(
            session="trss-wrong-settings",
            profile="TrSS",
            settings_hash="b" * 64,
            cpu=39.0,
            gpu=80.0,
        ),
    ]

    report = compare_matrix_rows(rows)

    assert report["paired_comparison_groups"] == 1
    cpu = next(row for row in report["comparison_rows"] if row["metric"] == "average_game_cpu_percent")
    assert cpu["baseline_mean"] == 30.0
    assert cpu["trss_mean"] == 34.0
    assert cpu["absolute_delta_trss_minus_baseline"] == 4.0
    assert round(cpu["percent_delta_trss_vs_baseline"], 2) == 13.33
    assert cpu["settings_match"] is True
    assert any("no baseline row" in row.get("reason", "") for row in report["unmatched_or_skipped_rows"])


def test_repeated_runs_are_equally_weighted_and_missing_fps_is_not_fabricated() -> None:
    rows = [
        _row(session="base-1", profile="Baseline", cpu=20.0, fps=None),
        _row(session="base-2", profile="Baseline", cpu=40.0, fps=None),
        _row(session="trss-1", profile="TrSS", cpu=30.0, fps=None),
    ]

    report = compare_matrix_rows(rows)

    cpu = next(row for row in report["comparison_rows"] if row["metric"] == "average_game_cpu_percent")
    assert cpu["baseline_runs_total"] == 2
    assert cpu["baseline_mean"] == 30.0
    assert cpu["trss_mean"] == 30.0
    assert not any(row["metric"] == "average_fps" for row in report["comparison_rows"])
    assert any("FPS" in limitation for limitation in report["limitations"])


def test_group_and_profile_labels_must_match_exactly() -> None:
    rows = [
        _row(session="base", profile="Baseline", group="track-a", cpu=10.0),
        _row(session="trss", profile="TrSS", group="track-b", cpu=12.0),
    ]

    report = compare_matrix_rows(rows)

    assert report["comparison_rows"] == []
    assert len(report["unmatched_or_skipped_rows"]) == 2


def test_require_exact_excludes_unverified_profiles() -> None:
    rows = [
        _row(session="base", profile="Baseline", exact=False, cpu=12.0),
        _row(session="trss", profile="TrSS", exact=False, cpu=14.0),
    ]

    report = compare_matrix_rows(rows, require_exact=True)

    assert report["comparison_rows"] == []
    assert len(report["unmatched_or_skipped_rows"]) == 2
    assert all("not marked exact" in row["reason"] for row in report["unmatched_or_skipped_rows"])


def test_jsonl_loader_skips_damaged_lines_and_export_is_readable(tmp_path: Path) -> None:
    matrix = tmp_path / "game-matrix.jsonl"
    rows = [
        _row(session="base", profile="Baseline", cpu=20.0),
        _row(session="trss", profile="TrSS", cpu=22.0),
    ]
    matrix.write_text(
        "\n".join([json.dumps(rows[0]), "not-json", json.dumps(rows[1])]) + "\n",
        encoding="utf-8",
    )

    loaded, warnings = load_matrix(matrix)
    report = compare_matrix_rows(loaded)
    json_path, csv_path = write_comparison(report, tmp_path / "report")

    assert len(loaded) == 2
    assert len(warnings) == 1
    assert json.loads(json_path.read_text(encoding="utf-8"))["paired_comparison_groups"] == 1
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert exported
    assert exported[0]["settings_match"] == "True"
