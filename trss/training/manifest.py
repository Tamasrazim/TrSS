"""Load and validate the local JSONL manifest for TrSS training pairs.

This module intentionally uses only the Python standard library, so manifest
errors can be checked without installing PyTorch or loading image data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REQUIRED_FIELDS = (
    "low_res",
    "target",
    "depth",
    "motion",
    "history",
    "history_valid",
)


class ManifestError(ValueError):
    """A training manifest entry is incomplete, invalid, or points to no file."""


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL entries and resolve their file paths relative to the manifest."""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")

    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(
                    f"{manifest_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ManifestError(
                    f"{manifest_path}:{line_number}: each line must be a JSON object."
                )

            missing = [name for name in _REQUIRED_FIELDS if name not in row]
            if missing:
                raise ManifestError(
                    f"{manifest_path}:{line_number}: missing explicit fields: "
                    + ", ".join(missing)
                )

            resolved: dict[str, Any] = {"line": line_number}
            for name in _REQUIRED_FIELDS:
                value = row[name]
                if name in ("history", "history_valid") and value is None:
                    resolved[name] = None
                    continue
                if not isinstance(value, str) or not value.strip():
                    raise ManifestError(
                        f"{manifest_path}:{line_number}: '{name}' must be a non-empty "
                        "path, or null for both history fields on a reset/first frame."
                    )
                candidate = Path(value).expanduser()
                if not candidate.is_absolute():
                    candidate = manifest_path.parent / candidate
                candidate = candidate.resolve()
                if not candidate.is_file():
                    raise ManifestError(
                        f"{manifest_path}:{line_number}: '{name}' file does not exist: "
                        f"{candidate}"
                    )
                resolved[name] = candidate

            if (resolved["history"] is None) != (resolved["history_valid"] is None):
                raise ManifestError(
                    f"{manifest_path}:{line_number}: 'history' and 'history_valid' "
                    "must either both name files or both be null."
                )
            rows.append(resolved)

    if not rows:
        raise ManifestError(f"Training manifest contains no entries: {manifest_path}")
    return rows
