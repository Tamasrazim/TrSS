"""First-party graphics-setting evidence and game-matrix records for TrSS.

This module never guesses a setting from FPS, GPU load, screenshot cadence, or a
preset label. It reads only allowlisted graphics keys from likely local game
configuration files; a participant must verify them against the game's own menu
before a run can be labelled an exact matrix row.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Iterable


# Every run uses the same columns, so rows from different games can be compared.
STANDARD_SETTINGS: dict[str, str] = {
    "game_build": "Game version / build",
    "graphics_api": "Graphics API / renderer",
    "graphics_preset": "Overall graphics preset",
    "display_mode": "Display mode",
    "display_resolution": "Display resolution",
    "render_resolution": "Render resolution",
    "refresh_rate_hz": "Display refresh rate (Hz)",
    "target_fps": "In-game frame-rate limit",
    "vsync": "V-Sync",
    "hdr": "HDR",
    "upscaler": "Upscaling technology",
    "upscaler_mode": "Upscaler mode",
    "frame_generation": "Frame generation",
    "anti_aliasing": "Anti-aliasing",
    "ray_tracing": "Ray tracing",
    "ray_tracing_quality": "Ray-tracing quality",
    "path_tracing": "Path tracing",
    "dynamic_resolution": "Dynamic resolution",
    "render_scale_percent": "Render scale (%)",
    "image_sharpening": "Image sharpening",
    "low_latency_mode": "Low-latency mode",
    "texture_quality": "Texture quality",
    "anisotropic_filtering": "Anisotropic filtering",
    "geometry_quality": "Geometry / LOD quality",
    "vegetation_quality": "Vegetation / grass quality",
    "crowd_density": "Crowd density",
    "effects_quality": "Effects quality",
    "shader_quality": "Shader quality",
    "terrain_quality": "Terrain quality",
    "shadow_quality": "Shadow quality",
    "far_shadow_quality": "Far-shadow quality",
    "reflection_quality": "Reflection quality",
    "ambient_occlusion": "Ambient occlusion",
    "global_illumination": "Global illumination",
    "volumetric_quality": "Volumetrics / fog",
    "post_processing": "Post-processing",
    "motion_blur": "Motion blur",
    "depth_of_field": "Depth of field",
    "bloom": "Bloom",
    "water_quality": "Water quality",
    "field_of_view_degrees": "Field of view (degrees)",
}

FIELD_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Build and display", (
        "game_build", "display_mode", "display_resolution", "render_resolution",
        "refresh_rate_hz", "target_fps", "vsync", "hdr", "field_of_view_degrees",
    )),
    ("Rendering and reconstruction", (
        "graphics_api", "graphics_preset", "anti_aliasing", "upscaler",
        "upscaler_mode", "frame_generation", "dynamic_resolution",
        "render_scale_percent", "image_sharpening", "low_latency_mode",
    )),
    ("Ray tracing and lighting", (
        "ray_tracing", "ray_tracing_quality", "path_tracing",
        "ambient_occlusion", "global_illumination", "shadow_quality",
        "far_shadow_quality", "reflection_quality", "volumetric_quality",
    )),
    ("Materials, geometry and effects", (
        "texture_quality", "anisotropic_filtering", "geometry_quality",
        "vegetation_quality", "crowd_density", "effects_quality",
        "shader_quality", "terrain_quality", "post_processing", "motion_blur",
        "depth_of_field", "bloom", "water_quality",
    )),
)

# These are real config-key aliases, not guessed interpretations of numeric enums.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "game_build": ("gameversion", "buildversion", "version", "buildnumber", "gamebuild"),
    "graphics_api": ("api", "graphicsapi", "renderingapi", "graphicsbackend", "renderer", "preferredgraphicsapi"),
    "graphics_preset": ("graphicspreset", "overallquality", "qualitypreset", "preset", "graphicsquality"),
    "display_mode": ("displaymode", "screenmode", "screentype", "windowmode", "fullscreen", "windowedfullscreen"),
    "display_resolution": ("resolution", "displayresolution", "screenresolution", "outputresolution", "resolutionmode"),
    "render_resolution": ("renderresolution", "internalresolution", "upscalerresolution"),
    "refresh_rate_hz": ("refreshrate", "refreshratehz", "displayrefreshrate", "monitorrefresh"),
    "target_fps": ("fpslimit", "framelimit", "maxfps", "targetfps", "maxframerate", "frameratecap"),
    "vsync": ("vsync", "verticalsync", "enablevsync", "usevsync", "busevsync"),
    "hdr": ("hdr", "enablehdr", "hdrenabled", "hdrmode"),
    "upscaler": ("upscaler", "upscaling", "upscalingmethod", "upscalertechnology", "dlss", "fsr", "xess", "tsr"),
    "upscaler_mode": ("upscalermode", "upscalingmode", "dlssmode", "fsrmode", "xessmode", "tsrquality"),
    "frame_generation": ("framegeneration", "framegen", "dlssframegeneration", "fsrframegeneration", "xessframegeneration"),
    "anti_aliasing": ("antialiasing", "antialiasingmethod", "aamode", "taa", "fxaa", "msaa", "smaa"),
    "ray_tracing": ("raytracing", "enableraytracing", "raytracingenabled", "rtenabled"),
    "ray_tracing_quality": ("raytracingquality", "rtquality", "raytracingshadows", "raytracingreflections"),
    "path_tracing": ("pathtracing", "enablepathtracing", "pathtracingenabled"),
    "dynamic_resolution": ("dynamicresolution", "dynamicresolutionenabled", "adaptiveresolution"),
    "render_scale_percent": ("renderscale", "renderscalepercent", "resolution_scale", "resolutionpercentage", "renderpercentage"),
    "image_sharpening": ("sharpening", "imagesharpening", "sharpness", "sharpen"),
    "low_latency_mode": ("lowlatency", "lowlatencymode", "reflex", "antilag", "latencyreduction"),
    "texture_quality": ("texturequality", "texturesquality", "texturequalitylevel", "texturestreaming"),
    "anisotropic_filtering": ("anisotropicfiltering", "anisotropy", "anisotropiclevel", "aflevel"),
    "geometry_quality": ("geometryquality", "lodquality", "lodscale", "geometrydetail", "objectdetail", "viewdistance"),
    "vegetation_quality": ("vegetationquality", "grassquality", "grasslod", "foliagequality", "vegetationdetail"),
    "crowd_density": ("crowddensity", "populationdensity", "crowdquality", "populationquality"),
    "effects_quality": ("effectsquality", "particlequality", "particlesquality", "effectdetail"),
    "shader_quality": ("shaderquality", "shadersquality", "shaderdetail"),
    "terrain_quality": ("terrainquality", "landscapequality", "grounddetail"),
    "shadow_quality": ("shadowquality", "shadowsquality", "shadowresolution", "softshadows"),
    "far_shadow_quality": ("farshadowquality", "distantshadowquality", "longshadows"),
    "reflection_quality": ("reflectionquality", "reflectionsquality", "screen spacereflections", "ssrquality", "mirrorquality"),
    "ambient_occlusion": ("ambientocclusion", "aoquality", "ssao", "hbao", "gtao", "rtao"),
    "global_illumination": ("globalillumination", "giquality", "illuminationquality", "indirectlighting"),
    "volumetric_quality": ("volumetricquality", "volumetricsquality", "fogquality", "volumetricfog", "cloudquality"),
    "post_processing": ("postprocessing", "postprocessingquality", "postfx", "postfxquality"),
    "motion_blur": ("motionblur", "motionblurenabled", "motionblurstrength"),
    "depth_of_field": ("depthoffield", "dof", "depthoffieldenabled"),
    "bloom": ("bloom", "bloomenabled", "bloomquality"),
    "water_quality": ("waterquality", "waterdetail", "watersimulationquality"),
    "field_of_view_degrees": ("fov", "fieldofview", "fieldofviewdegrees", "fovdegrees"),
}

_ALIAS_TO_FIELD = {
    re.sub(r"[^a-z0-9]", "", alias.casefold()): field
    for field, aliases in _FIELD_ALIASES.items()
    for alias in (field, *aliases)
}
_CONFIG_FILENAMES = {
    "system.xml", "graphicsconfig.xml", "hardware_settings_config.xml",
    "settings.xml", "graphics.ini", "graphics.cfg", "graphics.json",
    "settings.ini", "settings.cfg", "settings.json", "settings.toml",
    "usersettings.ini", "usersettings.cfg", "usersettings.json",
    "gameusersettings.ini", "video.ini", "video.cfg", "video.json",
    "user.cfg", "user.settings", "config.ini",
}
_MAX_CONFIG_BYTES = 2 * 1024 * 1024
_MAX_CONFIG_FILES = 24


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _safe_value(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple)):
        return None
    if isinstance(value, bool):
        return "On" if value else "Off"
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    text = text.strip().strip('"').strip("'").strip()
    return text[:160] or None


def _pairs_from_json(value: Any) -> Iterable[tuple[str, Any]]:
    """Walk JSON objects but emit only scalar name/value pairs for the allowlist."""
    if isinstance(value, dict):
        for name, item in value.items():
            if isinstance(item, (dict, list)):
                yield from _pairs_from_json(item)
            else:
                yield str(name), item
    elif isinstance(value, list):
        for item in value:
            yield from _pairs_from_json(item)


def _pairs_from_xml(root: ET.Element) -> Iterable[tuple[str, Any]]:
    for element in root.iter():
        # Handles both <TextureQuality>High</TextureQuality> and
        # <Option name="TextureQuality" value="High" /> configuration styles.
        if element.text and element.text.strip():
            yield element.tag.rsplit("}", 1)[-1], element.text.strip()
        attributes = {str(k).casefold(): v for k, v in element.attrib.items()}
        if "name" in attributes and ("value" in attributes or "setting" in attributes):
            yield str(attributes["name"]), attributes.get("value", attributes.get("setting"))
        elif "value" in attributes:
            # Configs often encode a setting as <VSync value="1" />.
            yield element.tag.rsplit("}", 1)[-1], attributes["value"]
        for name, value in element.attrib.items():
            yield str(name), value


def parse_graphics_config_file(path: str | Path) -> dict[str, Any]:
    """Read one config file; return only recognized graphics keys, never its full contents."""
    file_path = Path(path)
    try:
        if file_path.stat().st_size > _MAX_CONFIG_BYTES:
            return {"values": {}, "sources": {}, "source_names": {}, "error": "file_too_large"}
        content = file_path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return {"values": {}, "sources": {}, "source_names": {}, "error": "unreadable"}

    pairs: list[tuple[str, Any]] = []
    suffix = file_path.suffix.casefold()
    try:
        if suffix == ".json":
            pairs.extend(_pairs_from_json(json.loads(content)))
        elif suffix == ".xml":
            pairs.extend(_pairs_from_xml(ET.fromstring(content)))
        else:
            for line in content.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith(("#", ";", "//", "[")):
                    continue
                match = re.match(r"^([A-Za-z][A-Za-z0-9_. \-]{0,79})\s*[:=]\s*(.*?)\s*(?:[;#].*)?$", stripped)
                if match:
                    pairs.append((match.group(1).strip(), match.group(2).strip()))
    except (json.JSONDecodeError, ET.ParseError, ValueError):
        # A partly damaged config can still be parsed by the conservative line reader.
        for line in content.splitlines():
            match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_. \-]{0,79})\s*[:=]\s*(.*?)\s*$", line)
            if match:
                pairs.append((match.group(1).strip(), match.group(2).strip()))

    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    source_names: dict[str, str] = {}
    width: str | None = None
    height: str | None = None
    for name, raw_value in pairs:
        normalized = _key(name)
        value = _safe_value(raw_value)
        if not value:
            continue
        # Width and height are combined only when both are explicitly present.
        if normalized in {"resolutionwidth", "screenwidth", "displaywidth", "windowwidth", "width"}:
            width = value
            continue
        if normalized in {"resolutionheight", "screenheight", "displayheight", "windowheight", "height"}:
            height = value
            continue
        field = _ALIAS_TO_FIELD.get(normalized)
        if field is None:
            # Unreal-style keys often add a type/scalability prefix, e.g.
            # sg.TextureQuality, r.MotionBlur, or bUseVSync.
            for prefix in ("sg", "b", "r", "i", "f", "t"):
                if normalized.startswith(prefix):
                    field = _ALIAS_TO_FIELD.get(normalized[len(prefix):])
                    if field:
                        break
        if field:
            values[field] = value
            sources[field] = "local_game_config"
            source_names[field] = file_path.name
    if width and height and "display_resolution" not in values:
        values["display_resolution"] = f"{width}x{height}"
        sources["display_resolution"] = "local_game_config"
        source_names["display_resolution"] = file_path.name
    return {"values": values, "sources": sources, "source_names": source_names}


def _name_matches(game_name: str, folder_name: str) -> bool:
    wanted, found = _key(game_name), _key(folder_name)
    return bool(wanted and found and len(found) >= 4 and (wanted in found or found in wanted))


def discover_game_config_files(game_name: str, install_dir: str | Path) -> list[Path]:
    """Find a bounded set of conventional settings files without scanning whole drives."""
    install = Path(install_dir) if install_dir else Path()
    directories: list[Path] = []
    if install.is_dir():
        directories.extend([
            install,
            install / "Settings",
            install / "Config",
            install / "Saved" / "Config" / "Windows",
            install / "Saved" / "Config" / "WindowsNoEditor",
        ])

    home = Path.home()
    documents = Path(os.environ.get("USERPROFILE", str(home))) / "Documents"
    roots = [
        documents / "My Games",
        home / "Documents" / "My Games",
        Path(os.environ.get("USERPROFILE", str(home))) / "Saved Games",
    ]
    for variable in ("LOCALAPPDATA", "APPDATA"):
        raw_root = os.environ.get(variable)
        if raw_root:
            roots.append(Path(raw_root))

    for root in roots:
        if not root.is_dir():
            continue
        # Only descend into title-matching child folders; never walk all of AppData.
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children[:300]:
            try:
                if child.is_dir() and _name_matches(game_name, child.name):
                    directories.extend([
                        child,
                        child / "Settings",
                        child / "Config",
                        child / "Saved" / "Config" / "Windows",
                        child / "Saved" / "Config" / "WindowsNoEditor",
                    ])
            except OSError:
                continue

    unique_directories: list[Path] = []
    seen_directories: set[str] = set()
    for directory in directories:
        try:
            if directory.is_dir():
                key = os.path.normcase(str(directory.resolve()))
                if key not in seen_directories:
                    seen_directories.add(key)
                    unique_directories.append(directory)
        except OSError:
            continue

    discovered: list[Path] = []
    seen_files: set[str] = set()
    files_examined = 0
    for root in unique_directories:
        for current, folder_names, file_names in os.walk(root, topdown=True, onerror=lambda _error: None):
            current_path = Path(current)
            depth = len(current_path.parts) - len(root.parts)
            folder_names[:] = [
                name for name in folder_names
                if name.casefold() not in {"cache", "caches", "logs", "crashdumps", "shadercache", "screenshots", "movies"}
                and depth < 3
            ]
            files_examined += len(file_names)
            if files_examined > 20_000:
                return discovered[:_MAX_CONFIG_FILES]
            for filename in file_names:
                if filename.casefold() not in _CONFIG_FILENAMES:
                    continue
                candidate = current_path / filename
                try:
                    if candidate.stat().st_size > _MAX_CONFIG_BYTES:
                        continue
                    key = os.path.normcase(str(candidate.resolve()))
                except OSError:
                    continue
                if key not in seen_files:
                    seen_files.add(key)
                    discovered.append(candidate)
                    if len(discovered) >= _MAX_CONFIG_FILES:
                        return discovered
    return discovered


def probe_game_settings(game_name: str, install_dir: str | Path) -> dict[str, Any]:
    """Gather safe, allowlisted values from conventional local settings files."""
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    source_names: dict[str, str] = {}
    files = discover_game_config_files(game_name, install_dir)
    # Process install defaults first and user files later so a user profile can override them.
    install_text = os.path.normcase(str(Path(install_dir).resolve())) if install_dir else ""
    files.sort(key=lambda path: (
        0 if install_text and os.path.normcase(str(path.resolve())).startswith(install_text) else 1,
        path.name.casefold(),
        -path.stat().st_mtime_ns if path.exists() else 0,
    ))
    matched_files: list[str] = []
    for path in files:
        parsed = parse_graphics_config_file(path)
        if not parsed["values"]:
            continue
        matched_files.append(path.name)
        values.update(parsed["values"])
        sources.update(parsed["sources"])
        source_names.update(parsed["source_names"])
    return {
        "values": values,
        "sources": sources,
        "source_names": source_names,
        "files_checked": len(files),
        "matched_file_count": len(matched_files),
        "matched_file_names": list(dict.fromkeys(matched_files)),
    }


def parse_additional_settings(text: str) -> dict[str, str]:
    """Parse game-specific graphics settings written as 'Setting name = exact value' lines."""
    result: dict[str, str] = {}
    for line in text.splitlines()[:100]:
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if not name or not value:
            continue
        name = re.sub(r"[\r\n\t]", " ", name)[:80].strip()
        value = re.sub(r"[\r\n\t]", " ", value)[:160].strip()
        if name and value:
            result[name] = value
    return dict(sorted(result.items(), key=lambda item: item[0].casefold()))


def _canonical_settings(values: dict[str, Any] | None) -> dict[str, str | None]:
    raw = values or {}
    result: dict[str, str | None] = {}
    for field in STANDARD_SETTINGS:
        result[field] = _safe_value(raw.get(field))
    return result


def build_matrix_row(
    *,
    session_id: str,
    game_name: str,
    platform: str,
    started_utc: str,
    elapsed_seconds: float,
    captured_frames: int,
    performance_rows: list[dict[str, Any]],
    graphics_profile: dict[str, Any],
) -> dict[str, Any]:
    """Build one traceable row; an exact row requires in-menu confirmation and zero blanks."""
    values = _canonical_settings(graphics_profile.get("values"))
    additional = graphics_profile.get("additional_settings") or {}
    additional = {
        str(key)[:80]: _safe_value(value)
        for key, value in list(additional.items())[:100]
        if str(key).strip() and _safe_value(value)
    }
    sources = {
        field: str((graphics_profile.get("sources") or {}).get(field, "participant_reported"))[:48]
        for field, value in values.items() if value is not None
    }
    source_names = {
        field: str((graphics_profile.get("source_names") or {}).get(field, ""))[:120]
        for field in values if values[field] is not None
    }
    missing = [field for field, value in values.items() if value is None]
    not_confirmed = [
        field for field, value in values.items()
        if value is not None and value.strip().casefold()
        in {"unknown", "not captured", "not verified", "unavailable", "?"}
    ]
    # Several engines store enum indices (for example TextureQuality=3) instead
    # of names. Keep the raw number for traceability, but never call it exact until
    # the participant replaces it with the value shown by the game's own menu.
    literal_numeric_fields = {
        "game_build", "refresh_rate_hz", "target_fps",
        "render_scale_percent", "field_of_view_degrees",
    }
    unresolved_numeric = [
        field for field, value in values.items()
        if value is not None
        and field not in literal_numeric_fields
        and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value.strip())
        and sources.get(field) in {"local_game_config", "imported_local_config"}
    ]
    verified = bool(graphics_profile.get("verified_in_game"))
    settings_identity = {
        "game": _key(game_name),
        "values": values,
        "additional_settings": additional,
    }
    canonical_json = json.dumps(settings_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    settings_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    numeric_metrics = (
        "fps", "frametime_ms", "game_cpu_percent", "system_cpu_percent",
        "game_gpu_percent", "system_gpu_percent", "game_dedicated_gpu_bytes",
        "game_ram_bytes", "ram_percent",
    )
    summaries: dict[str, float | None] = {}
    for key in numeric_metrics:
        samples: list[float] = []
        for item in performance_rows:
            value = item.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                samples.append(float(value))
        summaries[f"average_{key}"] = round(sum(samples) / len(samples), 6) if samples else None
        summaries[f"sample_count_{key}"] = len(samples)

    incomplete_fields = sorted(set(missing) | set(not_confirmed) | set(unresolved_numeric))
    known = len(STANDARD_SETTINGS) - len(incomplete_fields)
    return {
        "schema_version": "trss.graphics-matrix/1.0",
        "session_id": session_id,
        "captured_at_utc": started_utc,
        "game": game_name,
        "platform": platform,
        "profile_name": str(graphics_profile.get("profile_name") or "Unlabelled run")[:120],
        "experiment_group": str(graphics_profile.get("experiment_group") or "")[:120],
        "profile_verified_in_game": verified,
        "matrix_row_exact": bool(verified and not incomplete_fields),
        "matrix_row_complete": not incomplete_fields,
        "settings_coverage": {
            "known_standard_settings": known,
            "total_standard_settings": len(STANDARD_SETTINGS),
            "coverage_percent": round(100 * known / len(STANDARD_SETTINGS), 2),
            "missing_settings": missing,
            "settings_not_confirmed": not_confirmed,
            "settings_needing_value_decoding": unresolved_numeric,
        },
        "settings_hash_sha256": settings_hash,
        "settings": values,
        "setting_sources": sources,
        "source_file_names": source_names,
        "additional_game_settings": additional,
        "capture": {
            "elapsed_seconds": round(float(elapsed_seconds), 4),
            "captured_frames": int(captured_frames),
            "performance_samples": len(performance_rows),
        },
        "performance_summary": summaries,
        "participant_notes": str(graphics_profile.get("notes") or "")[:2000],
        "config_discovery": {
            "files_checked": int(graphics_profile.get("files_checked") or 0),
            "files_with_graphics_settings": int(graphics_profile.get("matched_file_count") or 0),
        },
    }


def append_game_matrix_row(matrix_directory: str | Path, row: dict[str, Any]) -> tuple[Path, Path]:
    """Append the JSONL source of truth and rebuild a flat CSV suitable for analysis."""
    directory = Path(matrix_directory)
    directory.mkdir(parents=True, exist_ok=True)
    jsonl_path = directory / "game-matrix.jsonl"
    csv_path = directory / "game-matrix.csv"
    session_id = str(row.get("session_id") or "")
    # Do not create duplicate rows if export is retried after a successful append.
    already_written = False
    if jsonl_path.is_file():
        try:
            with jsonl_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    try:
                        if json.loads(line).get("session_id") == session_id:
                            already_written = True
                            break
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
    if not already_written:
        with jsonl_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)

    columns = [
        "session_id", "captured_at_utc", "game", "platform", "profile_name",
        "experiment_group", "profile_verified_in_game", "matrix_row_complete",
        "matrix_row_exact", "settings_hash_sha256", "settings_coverage_percent",
        "captured_frames", "elapsed_seconds", "performance_samples",
        "average_fps", "average_frametime_ms", "average_game_cpu_percent",
        "average_system_cpu_percent", "average_game_gpu_percent",
        "average_system_gpu_percent", "average_game_ram_bytes",
        "average_game_dedicated_gpu_bytes", *STANDARD_SETTINGS.keys(),
        "additional_game_settings_json", "missing_settings_json",
        "settings_needing_value_decoding_json", "participant_notes",
    ]
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for item in rows:
            capture = item.get("capture") or {}
            coverage = item.get("settings_coverage") or {}
            metrics = item.get("performance_summary") or {}
            flat: dict[str, Any] = {
                "session_id": item.get("session_id"),
                "captured_at_utc": item.get("captured_at_utc"),
                "game": item.get("game"),
                "platform": item.get("platform"),
                "profile_name": item.get("profile_name"),
                "experiment_group": item.get("experiment_group"),
                "profile_verified_in_game": item.get("profile_verified_in_game"),
                "matrix_row_complete": item.get("matrix_row_complete"),
                "matrix_row_exact": item.get("matrix_row_exact"),
                "settings_hash_sha256": item.get("settings_hash_sha256"),
                "settings_coverage_percent": coverage.get("coverage_percent"),
                "captured_frames": capture.get("captured_frames"),
                "elapsed_seconds": capture.get("elapsed_seconds"),
                "performance_samples": capture.get("performance_samples"),
                "additional_game_settings_json": json.dumps(item.get("additional_game_settings") or {}, ensure_ascii=False, sort_keys=True),
                "missing_settings_json": json.dumps(coverage.get("missing_settings") or [], ensure_ascii=False),
                "settings_needing_value_decoding_json": json.dumps(coverage.get("settings_needing_value_decoding") or [], ensure_ascii=False),
                "participant_notes": item.get("participant_notes"),
            }
            for setting in STANDARD_SETTINGS:
                flat[setting] = (item.get("settings") or {}).get(setting)
            for key in (
                "fps", "frametime_ms", "game_cpu_percent", "system_cpu_percent",
                "game_gpu_percent", "system_gpu_percent", "game_ram_bytes",
                "game_dedicated_gpu_bytes",
            ):
                flat[f"average_{key}"] = metrics.get(f"average_{key}")
            writer.writerow(flat)
    os.replace(temporary, csv_path)
    return jsonl_path, csv_path
