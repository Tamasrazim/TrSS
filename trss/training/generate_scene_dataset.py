"""Generate controlled, first-party 3D frame sequences for TrSS pipeline bring-up.

This small NumPy/Pillow rasterizer renders a known scene at two resolutions and
derives depth, frame-to-frame screen motion, and visibility-tested warped history
from the scene geometry. It is synthetic scene data, not a game/desktop capture,
and it is not intended to prove photorealistic quality or GPU performance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_NEAR_PLANE = 0.1
_FAR_PLANE = 30.0
_BACKGROUND = np.array([13, 17, 25], dtype=np.uint8)
_LIGHT_DIRECTION = np.array([0.35, 0.82, -0.45], dtype=np.float64)
_LIGHT_DIRECTION /= np.linalg.norm(_LIGHT_DIRECTION)

# Eight shared vertices of a unit cube. Each face is split into two triangles.
_CUBE_VERTICES = np.array([
    [-0.5, -0.5, -0.5],
    [ 0.5, -0.5, -0.5],
    [ 0.5,  0.5, -0.5],
    [-0.5,  0.5, -0.5],
    [-0.5, -0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5],
    [-0.5,  0.5,  0.5],
], dtype=np.float64)
_CUBE_TRIANGLES = (
    (0, 2, 1), (0, 3, 2),  # rear
    (4, 5, 6), (4, 6, 7),  # front
    (0, 1, 5), (0, 5, 4),  # bottom
    (3, 7, 6), (3, 6, 2),  # top
    (1, 2, 6), (1, 6, 5),  # right
    (0, 4, 7), (0, 7, 3),  # left
)
_PALETTE = (
    (45, 167, 181),
    (223, 118, 61),
    (145, 171, 92),
    (166, 111, 204),
    (220, 190, 83),
    (83, 127, 205),
    (204, 91, 119),
    (104, 177, 142),
)


def _camera_pose(time_index: int, scene_index: int) -> tuple[np.ndarray, ...]:
    """Return camera origin and right/up/forward unit vectors for one frame."""
    t = float(time_index)
    phase = scene_index * 0.71
    origin = np.array([
        0.13 * math.sin(t * 0.11 + phase),
        2.15 + 0.035 * math.sin(t * 0.17 + phase),
        -8.0,
    ], dtype=np.float64)
    target = np.array([
        0.08 * math.sin(t * 0.08 + phase),
        -0.15,
        3.2,
    ], dtype=np.float64)
    forward = target - origin
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)
    return origin, right, up, forward


def _make_objects(scene_index: int, seed: int) -> list[dict[str, Any]]:
    """Build repeatable boxes with different sizes, positions, colours, and motion."""
    rng = np.random.default_rng(seed + scene_index * 104729)
    objects: list[dict[str, Any]] = [
        {
            "name": "ground",
            "center": (0.0, -1.05, 5.5),
            "scale": (18.0, 0.22, 19.0),
            "color": (47, 55, 66),
            "yaw": 0.0,
            "phase": 0.0,
            "amplitude": 0.0,
            "spin": 0.0,
        },
        {
            "name": "rear-wall",
            "center": (0.0, 3.2, 15.1),
            "scale": (18.0, 9.0, 0.24),
            "color": (33, 42, 57),
            "yaw": 0.0,
            "phase": 0.0,
            "amplitude": 0.0,
            "spin": 0.0,
        },
    ]

    for index in range(12):
        width = float(rng.uniform(0.48, 1.15))
        height = float(rng.uniform(0.75, 2.25))
        depth = float(rng.uniform(0.50, 1.20))
        x = float(rng.uniform(-3.3, 3.3))
        z = float(rng.uniform(0.25, 11.3))
        # The props stand on the top face of the ground instead of being placed
        # at arbitrary heights; a few tall silhouettes help exercise occlusion.
        y = -0.94 + height / 2.0
        moving = index % 3 != 0
        objects.append({
            "name": f"prop-{index:02d}",
            "center": (x, y, z),
            "scale": (width, height, depth),
            "color": _PALETTE[(index + scene_index * 3) % len(_PALETTE)],
            "yaw": float(rng.uniform(-0.65, 0.65)),
            "phase": float(rng.uniform(0.0, math.tau)),
            "amplitude": float(rng.uniform(0.16, 0.43) if moving else 0.0),
            "spin": float(rng.uniform(0.035, 0.105) if moving else 0.0),
        })
    return objects


def _object_matrix(obj: dict[str, Any], time_index: int) -> np.ndarray:
    """Return the object's scale, rotation, and translation as a 4x4 transform."""
    x, y, z = obj["center"]
    sx, sy, sz = obj["scale"]
    t = float(time_index)
    phase = float(obj["phase"])
    amplitude = float(obj["amplitude"])
    x += amplitude * math.sin(t * 0.19 + phase)
    y += amplitude * 0.18 * math.sin(t * 0.13 + phase * 0.7)
    z += amplitude * 0.12 * math.cos(t * 0.15 + phase)
    yaw = float(obj["yaw"]) + float(obj["spin"]) * t
    pitch = 0.035 * math.sin(t * 0.12 + phase) if obj["spin"] else 0.0

    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    cos_x, sin_x = math.cos(pitch), math.sin(pitch)
    rotate_y = np.array([
        [cos_y, 0.0, sin_y],
        [0.0, 1.0, 0.0],
        [-sin_y, 0.0, cos_y],
    ], dtype=np.float64)
    rotate_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_x, -sin_x],
        [0.0, sin_x, cos_x],
    ], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotate_y @ rotate_x @ np.diag([sx, sy, sz])
    transform[:3, 3] = [x, y, z]
    return transform


def _project(
    points: np.ndarray,
    camera: tuple[np.ndarray, ...],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world-space points to pixel-edge coordinates and return view depth."""
    origin, right, up, forward = camera
    relative = points - origin
    view_x = relative @ right
    view_y = relative @ up
    view_z = relative @ forward
    focal = height / (2.0 * math.tan(math.radians(58.0) / 2.0))
    safe_z = np.where(np.abs(view_z) < 1e-9, 1e-9, view_z)
    screen = np.stack((
        width / 2.0 + focal * view_x / safe_z,
        height / 2.0 - focal * view_y / safe_z,
    ), axis=-1)
    return screen, view_z


def _render(
    objects: list[dict[str, Any]],
    matrices: list[np.ndarray],
    camera: tuple[np.ndarray, ...],
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    """Rasterize the visible faces and retain the geometry needed for motion."""
    color = np.empty((height, width, 3), dtype=np.uint8)
    color[...] = _BACKGROUND
    depth_buffer = np.full((height, width), _FAR_PLANE, dtype=np.float32)
    object_ids = np.full((height, width), -1, dtype=np.int16)
    local_positions = np.zeros((height, width, 3), dtype=np.float32)

    for object_id, (obj, matrix) in enumerate(zip(objects, matrices)):
        world_vertices = _CUBE_VERTICES @ matrix[:3, :3].T + matrix[:3, 3]
        screen_vertices, vertex_depth = _project(world_vertices, camera, width, height)

        for tri in _CUBE_TRIANGLES:
            indexes = np.asarray(tri, dtype=np.int64)
            points = screen_vertices[indexes]
            depths = vertex_depth[indexes]
            if (
                np.any(depths <= _NEAR_PLANE)
                or np.any(depths >= _FAR_PLANE)
                or not np.isfinite(points).all()
            ):
                continue

            x_min = max(0, int(math.floor(float(points[:, 0].min()))))
            x_max = min(width - 1, int(math.ceil(float(points[:, 0].max()))))
            y_min = max(0, int(math.floor(float(points[:, 1].min()))))
            y_max = min(height - 1, int(math.ceil(float(points[:, 1].max()))))
            if x_min > x_max or y_min > y_max:
                continue

            p0, p1, p2 = points
            denominator = (
                (p1[1] - p2[1]) * (p0[0] - p2[0])
                + (p2[0] - p1[0]) * (p0[1] - p2[1])
            )
            if abs(float(denominator)) < 1e-10:
                continue

            xs, ys = np.meshgrid(
                np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5,
                np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5,
            )
            w0 = (
                (p1[1] - p2[1]) * (xs - p2[0])
                + (p2[0] - p1[0]) * (ys - p2[1])
            ) / denominator
            w1 = (
                (p2[1] - p0[1]) * (xs - p2[0])
                + (p0[0] - p2[0]) * (ys - p2[1])
            ) / denominator
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
            if not inside.any():
                continue

            inverse_depth = w0 / depths[0] + w1 / depths[1] + w2 / depths[2]
            safe_inverse = np.where(inverse_depth > 1e-12, inverse_depth, 1e-12)
            fragment_depth = 1.0 / safe_inverse
            y_slice = slice(y_min, y_max + 1)
            x_slice = slice(x_min, x_max + 1)
            z_region = depth_buffer[y_slice, x_slice]
            write = inside & (fragment_depth < z_region) & (fragment_depth > _NEAR_PLANE)
            if not write.any():
                continue

            color_region = color[y_slice, x_slice]
            id_region = object_ids[y_slice, x_slice]
            local_region = local_positions[y_slice, x_slice]
            weights = np.stack((w0 / depths[0], w1 / depths[1], w2 / depths[2]), axis=-1)
            weights /= safe_inverse[..., None]
            local_points = weights @ _CUBE_VERTICES[indexes]

            world_tri = world_vertices[indexes]
            normal = np.cross(world_tri[1] - world_tri[0], world_tri[2] - world_tri[0])
            norm = float(np.linalg.norm(normal))
            if norm > 1e-12:
                normal /= norm
            illumination = 0.44 + 0.48 * abs(float(np.dot(normal, _LIGHT_DIRECTION)))
            face_color = np.clip(np.asarray(obj["color"], dtype=np.float64) * illumination, 0, 255)
            rounded_color = np.rint(face_color).astype(np.uint8)

            z_region[write] = fragment_depth[write].astype(np.float32)
            color_region[write] = rounded_color
            id_region[write] = object_id
            local_region[write] = local_points[write].astype(np.float32)

    normalized_depth = np.ones((height, width), dtype=np.float32)
    covered = object_ids >= 0
    normalized_depth[covered] = np.clip(
        (depth_buffer[covered] - _NEAR_PLANE) / (_FAR_PLANE - _NEAR_PLANE),
        0.0, 1.0
    )
    return {
        "color": color,
        "depth": normalized_depth,
        "view_depth": depth_buffer,
        "object_ids": object_ids,
        "local_positions": local_positions,
        "covered": covered,
    }


def _project_visible_points(
    render: dict[str, np.ndarray],
    previous_matrices: list[np.ndarray],
    previous_camera: tuple[np.ndarray, ...],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each current visible surface point back into the previous camera."""
    object_ids = render["object_ids"]
    previous_x = np.zeros((height, width), dtype=np.float32)
    previous_y = np.zeros((height, width), dtype=np.float32)
    previous_z = np.full((height, width), _FAR_PLANE, dtype=np.float32)
    for object_id, previous_matrix in enumerate(previous_matrices):
        visible = object_ids == object_id
        if not visible.any():
            continue
        local = render["local_positions"][visible].astype(np.float64)
        previous_world = local @ previous_matrix[:3, :3].T + previous_matrix[:3, 3]
        screen, z = _project(previous_world, previous_camera, width, height)
        previous_x[visible] = screen[:, 0].astype(np.float32)
        previous_y[visible] = screen[:, 1].astype(np.float32)
        previous_z[visible] = z.astype(np.float32)
    return previous_x, previous_y, previous_z


def _motion_vectors(
    render: dict[str, np.ndarray],
    previous_matrices: list[np.ndarray],
    previous_camera: tuple[np.ndarray, ...],
    width: int,
    height: int,
) -> np.ndarray:
    """Store current-minus-previous screen displacement in low-res pixels/frame."""
    flow = np.zeros((height, width, 2), dtype=np.float32)
    visible = render["covered"]
    if not visible.any():
        return flow
    prev_x, prev_y, prev_z = _project_visible_points(
        render, previous_matrices, previous_camera, width, height
    )
    y, x = np.indices((height, width), dtype=np.float32)
    current_x = x + 0.5
    current_y = y + 0.5
    valid_projection = visible & (prev_z > _NEAR_PLANE) & (prev_z < _FAR_PLANE)
    flow[..., 0][valid_projection] = current_x[valid_projection] - prev_x[valid_projection]
    flow[..., 1][valid_projection] = current_y[valid_projection] - prev_y[valid_projection]
    return flow


def _warp_previous_target(
    current: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
    previous_matrices: list[np.ndarray],
    previous_camera: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Gather previous-frame target pixels, rejecting occluded or out-of-view samples."""
    height, width = current["object_ids"].shape
    warped = np.zeros_like(current["color"])
    valid_mask = np.zeros((height, width), dtype=np.uint8)
    previous_x, previous_y, previous_z = _project_visible_points(
        current, previous_matrices, previous_camera, width, height
    )

    for object_id in range(len(previous_matrices)):
        visible = current["object_ids"] == object_id
        if not visible.any():
            continue
        screen_x = previous_x[visible]
        screen_y = previous_y[visible]
        projected_z = previous_z[visible]
        in_frame = (
            (screen_x >= 0.0) & (screen_x < width)
            & (screen_y >= 0.0) & (screen_y < height)
            & (projected_z > _NEAR_PLANE) & (projected_z < _FAR_PLANE)
        )
        if not in_frame.any():
            continue

        destination_y, destination_x = np.nonzero(visible)
        destination_y = destination_y[in_frame]
        destination_x = destination_x[in_frame]
        sample_x = np.floor(screen_x[in_frame]).astype(np.int64)
        sample_y = np.floor(screen_y[in_frame]).astype(np.int64)
        previous_ids = previous["object_ids"][sample_y, sample_x]
        observed_z = previous["view_depth"][sample_y, sample_x]
        depth_tolerance = np.maximum(0.035, projected_z[in_frame] * 0.012)
        visible_in_previous = (
            (previous_ids == object_id)
            & (np.abs(observed_z - projected_z[in_frame]) <= depth_tolerance)
        )
        if not visible_in_previous.any():
            continue
        dst_y = destination_y[visible_in_previous]
        dst_x = destination_x[visible_in_previous]
        src_y = sample_y[visible_in_previous]
        src_x = sample_x[visible_in_previous]
        warped[dst_y, dst_x] = previous["color"][src_y, src_x]
        valid_mask[dst_y, dst_x] = 255

    return warped, valid_mask


def _save_rgb(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG", optimize=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _create_dataset(
    root: Path,
    *,
    scenes: int,
    frames_per_scene: int,
    width: int,
    height: int,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for scene_index in range(scenes):
        scene_id = f"scene-{scene_index:03d}"
        objects = _make_objects(scene_index, seed)
        previous_target: dict[str, np.ndarray] | None = None
        previous_matrices: list[np.ndarray] | None = None
        previous_camera: tuple[np.ndarray, ...] | None = None

        for frame_index in range(frames_per_scene):
            current_matrices = [
                _object_matrix(obj, frame_index) for obj in objects
            ]
            current_camera = _camera_pose(frame_index, scene_index)
            low = _render(objects, current_matrices, current_camera, width, height)
            high = _render(
                objects, current_matrices, current_camera, width * 2, height * 2
            )

            relative_folder = Path(scene_id) / f"frame-{frame_index:04d}"
            low_path = Path("input") / relative_folder.with_suffix(".png")
            target_path = Path("target") / relative_folder.with_suffix(".png")
            depth_path = Path("depth") / relative_folder.with_suffix(".npy")
            motion_path = Path("motion") / relative_folder.with_suffix(".npy")

            _save_rgb(root / low_path, low["color"])
            _save_rgb(root / target_path, high["color"])
            (root / depth_path).parent.mkdir(parents=True, exist_ok=True)
            (root / motion_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(root / depth_path, low["depth"].astype(np.float32))
            if frame_index == 0:
                motion = np.zeros((height, width, 2), dtype=np.float32)
            else:
                assert previous_matrices is not None and previous_camera is not None
                motion = _motion_vectors(
                    low, previous_matrices, previous_camera, width, height
                )
            np.save(root / motion_path, motion)

            history_path: str | None = None
            history_valid_path: str | None = None
            if frame_index > 0:
                assert previous_target is not None
                assert previous_matrices is not None and previous_camera is not None
                history, valid_mask = _warp_previous_target(
                    high, previous_target, previous_matrices, previous_camera
                )
                history_file = Path("history") / relative_folder.with_suffix(".png")
                valid_file = Path("history_valid") / relative_folder.with_suffix(".png")
                _save_rgb(root / history_file, history)
                (root / valid_file).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(valid_mask, mode="L").save(valid_file, format="PNG", optimize=False)
                history_path = history_file.as_posix()
                history_valid_path = valid_file.as_posix()

            rows.append({
                "scene_id": scene_id,
                "frame_index": frame_index,
                "camera_id": f"camera-{scene_index:03d}",
                "generator": "trss-controlled-software-rasterizer-v1",
                "depth_encoding": "linear view-space depth normalized from near=0.1 to far=30.0; background=1.0",
                "motion_convention": "current pixel center minus the same surface point projected into the previous frame; x right-positive, y down-positive, low-res pixels per frame",
                "low_res": low_path.as_posix(),
                "target": target_path.as_posix(),
                "depth": depth_path.as_posix(),
                "motion": motion_path.as_posix(),
                "history": history_path,
                "history_valid": history_valid_path,
                "source_type": "synthetic_controlled_3d_scene",
                "seed": seed + scene_index * 104729,
            })

            previous_target = high
            previous_matrices = current_matrices
            previous_camera = current_camera

    manifest_path = root / "all-frames.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    provenance = {
        "dataset_name": "TrSS controlled 3D renderer pilot",
        "source": "First-party deterministic software rasterizer and generated box-scene geometry; no external assets",
        "license": "Synthetic first-party data generated for TrSS engineering and training-pipeline tests",
        "data_kind": "synthetic controlled-scene renders; not captured from a running game",
        "generator": "trss-controlled-software-rasterizer-v1",
        "seed": seed,
    }
    _write_json(root / "provenance.json", provenance)
    report = {
        "format": "trss-controlled-scene-dataset-v1",
        "source_type": "synthetic_controlled_3d_scene",
        "actual_game_capture": False,
        "scenes": scenes,
        "frames_per_scene": frames_per_scene,
        "frame_count": len(rows),
        "input_size": [width, height],
        "target_size": [width * 2, height * 2],
        "seed": seed,
        "manifest": "all-frames.jsonl",
        "provenance": "provenance.json",
        "depth": "Computed from rasterized view-space geometry; normalized linear view depth, not a vendor GPU depth buffer.",
        "motion": "Computed by projecting visible surface positions through the previous object transforms and camera; current-minus-previous screen displacement.",
        "history": "Previous target frames reprojected by geometry and filtered against previous object IDs and depth; reset frames use null paths.",
        "limitations": [
            "This is a controlled synthetic scene, not a capture hook for a running game or desktop.",
            "The renderer uses flat-shaded boxes and a simple CPU rasterizer; results are for pipeline bring-up, not photorealistic training claims.",
            "Temporal visibility uses nearest-pixel previous-frame depth/object-ID checks; review the generated masks before relying on them for model quality.",
        ],
    }
    _write_json(root / "generation-report.json", report)
    return report


def generate_scene_dataset(
    output_dir: str | Path,
    *,
    scenes: int = 4,
    frames_per_scene: int = 6,
    width: int = 160,
    height: int = 90,
    seed: int = 20261014,
    force: bool = False,
) -> dict[str, Any]:
    """Generate a deterministic pilot dataset into a replace-on-success directory."""
    if not isinstance(scenes, int) or isinstance(scenes, bool) or scenes < 2:
        raise ValueError("scenes must be an integer of at least 2 for separate train/validation scenes.")
    if not isinstance(frames_per_scene, int) or isinstance(frames_per_scene, bool) or frames_per_scene < 2:
        raise ValueError("frames_per_scene must be an integer of at least 2.")
    if not isinstance(width, int) or isinstance(width, bool) or width < 24:
        raise ValueError("width must be an integer of at least 24 pixels.")
    if not isinstance(height, int) or isinstance(height, bool) or height < 16:
        raise ValueError("height must be an integer of at least 16 pixels.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")

    output = Path(output_dir).expanduser().resolve()
    if output == output.parent or output.name in ("", ".", ".."):
        raise ValueError("output_dir must name a dedicated output folder.")
    if output.exists() and not force:
        raise FileExistsError(f"Output path already exists: {output}. Use --force to replace it.")
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path must be a directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    backup: Path | None = None
    try:
        report = _create_dataset(
            staging,
            scenes=scenes,
            frames_per_scene=frames_per_scene,
            width=width,
            height=height,
            seed=seed,
        )
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=output.parent))
            backup.rmdir()
            output.rename(backup)
        try:
            staging.rename(output)
        except Exception:
            if backup is not None and backup.exists() and not output.exists():
                backup.rename(output)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
        return {**report, "output_dir": str(output)}
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup is not None and backup.exists():
            # Only remove the old copy if the new one replaced it successfully.
            if output.exists():
                shutil.rmtree(backup, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate controlled synthetic 3D training pairs and geometry-derived "
            "depth, motion, and reprojected-history data. This is not game capture."
        )
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--frames-per-scene", type=int, default=6)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--force", action="store_true", help="Replace an existing output directory.")
    args = parser.parse_args()

    try:
        report = generate_scene_dataset(
            args.output_dir,
            scenes=args.scenes,
            frames_per_scene=args.frames_per_scene,
            width=args.width,
            height=args.height,
            seed=args.seed,
            force=args.force,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
