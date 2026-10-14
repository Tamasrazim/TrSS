"""Build an offline visual review gallery for real TrSS frame pairs.

The gallery helps a human inspect sample inputs and reference targets before
training. It does not repair pairs, synthesize auxiliary buffers, or train weights.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from .manifest import load_manifest


def _raw_rows(path: Path) -> list[dict[str, Any]]:
    """Load optional scene/provenance columns after the shared manifest validator."""
    resolved = load_manifest(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            rows.append(json.loads(line))
    if len(rows) != len(resolved):
        raise RuntimeError("Manifest row count changed while it was being read.")
    return rows


def _scene_name(row: dict[str, Any], scene_field: str) -> str:
    """Return the declared scene ID or a safe label when no ID was supplied."""
    value = row.get(scene_field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "(scene ID not supplied)"


def _select_indexes(
    rows: list[dict[str, Any]],
    *,
    max_frames: int,
    seed: int,
    scene_field: str,
) -> list[int]:
    """Pick a reproducible sample round-robin across scene IDs."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[_scene_name(row, scene_field)].append(index)

    generator = random.Random(seed)
    scenes = sorted(groups)
    generator.shuffle(scenes)
    for indexes in groups.values():
        generator.shuffle(indexes)

    selected: list[int] = []
    limit = min(max_frames, len(rows))
    while len(selected) < limit:
        made_progress = False
        for scene in scenes:
            if groups[scene] and len(selected) < limit:
                selected.append(groups[scene].pop())
                made_progress = True
        if not made_progress:
            break
    return selected


def _thumbnail(
    source: Path,
    destination: Path,
    *,
    width: int,
    height: int,
) -> tuple[int, int]:
    """Write a proportion-preserving local preview and return source dimensions."""
    try:
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            original_size = image.size
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise ValueError(f"cannot decode image '{source}': {exc}") from exc

    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (15, 17, 21))
    position = ((width - image.width) // 2, (height - image.height) // 2)
    canvas.paste(image, position)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=90, optimize=True)
    return original_size


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a complete report to a temporary file, then replace the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _gallery_html(
    *,
    source_manifest: Path,
    report: dict[str, Any],
    scene_field: str,
) -> str:
    """Render a single-file, offline HTML page that only references local thumbnails."""
    cards: list[str] = []
    for row in report["selected_frames"]:
        scene = html.escape(str(row["scene_id"]))
        frame_label = f'Manifest line {row["manifest_line"]}'
        image_cards: list[str] = []
        for field, label in (
            ("input_thumbnail", "INPUT"),
            ("target_thumbnail", "REFERENCE TARGET · 2×"),
        ):
            image_path = row.get(field)
            if image_path:
                image_cards.append(
                    f'<figure><img loading="lazy" src="{html.escape(image_path, quote=True)}" '
                    f'alt="{html.escape(label, quote=True)}">'
                    f"<figcaption>{html.escape(label)}</figcaption></figure>"
                )
            else:
                image_cards.append(
                    '<figure><div class="missing">Preview unavailable</div>'
                    f"<figcaption>{html.escape(label)}</figcaption></figure>"
                )

        error = row.get("preview_error")
        status = (
            '<span class="flag bad">PREVIEW ERROR</span>'
            if error
            else '<span class="flag">REVIEW PAIR</span>'
        )
        size_text = (
            f'{row["input_width"]}×{row["input_height"]} → '
            f'{row["target_width"]}×{row["target_height"]}'
            if row.get("input_width") and row.get("target_width")
            else "Image dimensions unavailable"
        )
        history = "reset / first frame" if row.get("history_is_reset") else "provided history"
        error_html = f'<p class="error">{html.escape(str(error))}</p>' if error else ""
        cards.append(
            f'<article class="card" data-scene="{scene}" '
            f'data-line="{row["manifest_line"]}">'
            f'<div class="card-head"><div><span class="eyebrow">{html.escape(frame_label)}</span>'
            f'<h2>{scene}</h2></div>{status}</div>'
            f'<div class="images">{"".join(image_cards)}</div>'
            f'<div class="metadata"><span>SIZE <b>{html.escape(size_text)}</b></span>'
            f'<span>HISTORY <b>{html.escape(history)}</b></span></div>'
            f'<details><summary>Source paths</summary>'
            f'<p><strong>Input</strong><br>{html.escape(str(row["low_res_path"]))}</p>'
            f'<p><strong>Reference</strong><br>{html.escape(str(row["target_path"]))}</p>'
            f'{error_html}</details></article>'
        )

    errors = int(report["preview_errors"])
    status = (
        f"{errors} selected pair(s) could not be fully previewed."
        if errors
        else "All selected image pairs decoded successfully."
    )
    scene_counts = report["selected_scene_counts"]
    scene_summary = ", ".join(
        f"{html.escape(name)}: {count}" for name, count in sorted(scene_counts.items())
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TrSS · Dataset Pair Review</title>
<style>
:root {{ color-scheme: dark; font-family: Inter, "Segoe UI", Arial, sans-serif; background:#090b0e; color:#edf0f4; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; }}
header {{ padding:28px clamp(18px,4vw,54px) 22px; border-bottom:1px solid #303640; background:linear-gradient(135deg,#151920,#090b0e); }}
.kicker,.eyebrow {{ color:#929baa; font:700 11px/1.4 ui-monospace,Consolas,monospace; letter-spacing:.12em; text-transform:uppercase; }}
h1 {{ margin:9px 0 7px; font-size:clamp(25px,4vw,38px); letter-spacing:-.04em; }}
p {{ line-height:1.55; }}
.sub {{ max-width:850px; margin:0; color:#adb5c1; }}
.stats {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:20px; }}
.stat {{ border:1px solid #353d48; border-radius:8px; padding:10px 13px; min-width:120px; background:#11151b; }}
.stat strong {{ display:block; font-size:20px; }}
.stat span {{ color:#9ba4b1; font:10px ui-monospace,Consolas,monospace; letter-spacing:.08em; }}
.toolbar {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; padding:18px clamp(18px,4vw,54px); border-bottom:1px solid #242a33; }}
input {{ background:#11151b; border:1px solid #414a57; color:#f4f6fa; border-radius:7px; padding:12px 14px; min-width:min(100%,340px); font:inherit; }}
.note {{ color:#a5aebb; font-size:13px; }}
main {{ padding:24px clamp(14px,3vw,38px) 54px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr)); gap:18px; }}
.card {{ overflow:hidden; border:1px solid #303742; border-radius:11px; background:#101319; }}
.card-head {{ padding:16px 17px 12px; display:flex; align-items:center; justify-content:space-between; gap:12px; }}
h2 {{ margin:3px 0 0; font-size:17px; overflow-wrap:anywhere; }}
.flag {{ flex-shrink:0; border:1px solid #626d7b; color:#d6dce5; border-radius:4px; padding:5px 7px; font:700 9px ui-monospace,Consolas,monospace; letter-spacing:.07em; }}
.bad {{ border-color:#a9b5c5; color:#fff; }}
.images {{ display:grid; grid-template-columns:1fr 1fr; gap:1px; background:#323945; }}
figure {{ margin:0; min-width:0; background:#090b0e; }}
img {{ display:block; width:100%; height:auto; aspect-ratio:{report["thumbnail_width"]}/{report["thumbnail_height"]}; object-fit:contain; background:#0f1115; }}
figcaption {{ padding:8px 10px; color:#a8b1be; font:700 10px ui-monospace,Consolas,monospace; letter-spacing:.08em; }}
.missing {{ display:grid; place-items:center; aspect-ratio:{report["thumbnail_width"]}/{report["thumbnail_height"]}; color:#a8b1be; font-size:12px; background:repeating-linear-gradient(135deg,#11151b,#11151b 8px,#171c24 8px,#171c24 16px); }}
.metadata {{ display:flex; flex-wrap:wrap; gap:12px; padding:12px 17px; border-bottom:1px solid #2b313a; }}
.metadata span {{ display:grid; gap:4px; color:#8f99a8; font:9px ui-monospace,Consolas,monospace; letter-spacing:.08em; }}
.metadata b {{ color:#dbe0e7; font-size:11px; letter-spacing:0; }}
details {{ padding:12px 17px 15px; color:#a7b0bd; font-size:12px; }}
summary {{ cursor:pointer; color:#e3e7ed; }}
details p {{ overflow-wrap:anywhere; font-size:11px; }}
.error {{ color:#fff; border-left:2px solid #c2cbd8; padding-left:10px; }}
footer {{ padding:0 clamp(18px,4vw,54px) 30px; color:#7f8998; font-size:11px; }}
.empty {{ color:#9ca6b4; }}
@media (max-width:560px) {{ .images {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <div class="kicker">TAMASRAZIM · REAL-TIME SCENE SYNTHESIS</div>
  <h1>Dataset pair review</h1>
  <p class="sub">Offline visual inspection of sampled low-resolution inputs and their reference targets.
  This page does not modify your dataset or prove that pairs are correctly aligned.</p>
  <div class="stats">
    <div class="stat"><strong>{report["source_frames"]}</strong><span>FRAMES IN MANIFEST</span></div>
    <div class="stat"><strong>{len(report["selected_frames"])}</strong><span>SELECTED FOR REVIEW</span></div>
    <div class="stat"><strong>{len(scene_counts)}</strong><span>SCENES REPRESENTED</span></div>
    <div class="stat"><strong>{errors}</strong><span>PREVIEW ERRORS</span></div>
  </div>
</header>
<div class="toolbar">
  <input id="search" type="search" placeholder="Filter by scene ID or manifest line…" aria-label="Filter frames">
  <span class="note">{html.escape(status)}</span>
  <span class="note">Sampling is deterministic (seed {report["seed"]}).</span>
</div>
<main><div class="cards" id="cards">
{"".join(cards) if cards else '<p class="empty">No pair previews were generated. Check the manifest and report errors.</p>'}
</div></main>
<footer>
  <div><strong>Source manifest:</strong> {html.escape(str(source_manifest))}</div>
  <div><strong>Scene field:</strong> {html.escape(scene_field)} · <strong>Seed:</strong> {report["seed"]}</div>
  <p>Selected scene coverage: {scene_summary or "none"}.</p>
  <p>Review visual alignment, exposure/color-space conventions, scene cuts, motion direction, and asset licences manually before training.</p>
</footer>
<script>
const search = document.getElementById('search');
const cards = Array.from(document.querySelectorAll('.card'));
search.addEventListener('input', () => {{
  const term = search.value.trim().toLowerCase();
  cards.forEach(card => {{
    const scene = (card.dataset.scene || '').toLowerCase();
    const line = (card.dataset.line || '').toLowerCase();
    card.hidden = !(scene.includes(term) || line.includes(term));
  }});
}});
</script>
</body>
</html>
"""


def build_review_gallery(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    max_frames: int = 36,
    seed: int = 20261014,
    scene_field: str = "scene_id",
    thumbnail_width: int = 480,
    thumbnail_height: int = 270,
) -> dict[str, Any]:
    """Generate an offline HTML gallery plus portable local thumbnail assets."""
    source = Path(manifest_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1.")
    if not scene_field.strip():
        raise ValueError("scene_field must be a non-empty metadata field.")
    if not (64 <= thumbnail_width <= 1600 and 64 <= thumbnail_height <= 1200):
        raise ValueError("thumbnail dimensions must be between 64x64 and 1600x1200.")

    raw_rows = _raw_rows(source)
    resolved_rows = load_manifest(source)
    selected = _select_indexes(
        raw_rows, max_frames=max_frames, seed=seed, scene_field=scene_field
    )
    destination.mkdir(parents=True, exist_ok=True)
    images_dir = destination / "images"
    selected_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    scene_counts: Counter[str] = Counter()

    for number, index in enumerate(selected, start=1):
        raw = raw_rows[index]
        resolved = resolved_rows[index]
        scene = _scene_name(raw, scene_field)
        scene_counts[scene] += 1
        input_path = Path(resolved["low_res"])
        target_path = Path(resolved["target"])
        detail: dict[str, Any] = {
            "manifest_line": int(resolved["line"]),
            "scene_id": scene,
            "low_res_path": str(input_path),
            "target_path": str(target_path),
            "input_thumbnail": None,
            "target_thumbnail": None,
            "input_width": None,
            "input_height": None,
            "target_width": None,
            "target_height": None,
            "history_is_reset": resolved["history"] is None,
            "preview_error": None,
        }
        failures: list[str] = []
        for field, source_path, label in (
            ("input_thumbnail", input_path, "input"),
            ("target_thumbnail", target_path, "reference target"),
        ):
            thumb_name = f"pair-{number:04d}-{label.replace(' ', '-')}.jpg"
            thumb_path = images_dir / thumb_name
            try:
                original_width, original_height = _thumbnail(
                    source_path,
                    thumb_path,
                    width=thumbnail_width,
                    height=thumbnail_height,
                )
                detail[field] = f"images/{thumb_name}"
                if label == "input":
                    detail["input_width"] = original_width
                    detail["input_height"] = original_height
                else:
                    detail["target_width"] = original_width
                    detail["target_height"] = original_height
            except (OSError, ValueError) as exc:
                failures.append(str(exc))
        if failures:
            detail["preview_error"] = "; ".join(failures)
            errors.append({
                "manifest_line": detail["manifest_line"],
                "scene_id": scene,
                "errors": failures,
            })
        selected_rows.append(detail)

    report: dict[str, Any] = {
        "format": "trss-pair-review-v1",
        "passed": not errors,
        "source_manifest": str(source),
        "source_frames": len(raw_rows),
        "selected_count": len(selected_rows),
        "seed": seed,
        "scene_field": scene_field,
        "thumbnail_width": thumbnail_width,
        "thumbnail_height": thumbnail_height,
        "selected_scene_counts": dict(sorted(scene_counts.items())),
        "selected_frames": selected_rows,
        "preview_errors": len(errors),
        "errors": errors,
        "notice": (
            "Visual inspection support only. The gallery does not alter source images, "
            "validate true scene alignment, verify licence terms, or train the model."
        ),
    }
    html_path = destination / "index.html"
    report_path = destination / "review-report.json"
    report["report_path"] = str(report_path)
    report["gallery_path"] = str(html_path)
    _write_json(report_path, report)
    html_path.write_text(
        _gallery_html(source_manifest=source, report=report, scene_field=scene_field),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an offline visual review gallery of real paired TrSS frames."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--scene-field", default="scene_id")
    parser.add_argument("--thumbnail-width", type=int, default=480)
    parser.add_argument("--thumbnail-height", type=int, default=270)
    args = parser.parse_args()

    try:
        report = build_review_gallery(
            args.manifest,
            args.output_dir,
            max_frames=args.max_frames,
            seed=args.seed,
            scene_field=args.scene_field,
            thumbnail_width=args.thumbnail_width,
            thumbnail_height=args.thumbnail_height,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
