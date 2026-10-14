"""Frame-source validation and temporal history behavior."""

import numpy as np
import pytest

from trss.rendering import FramePacket, FrameValidationError, TemporalHistory


def make_frame(
    frame_id: int,
    timestamp: float,
    *,
    size: tuple[int, int] = (4, 3),
    scene_id: str | None = "scene-a",
    camera_cut: bool = False,
) -> FramePacket:
    width, height = size
    return FramePacket(
        color=np.zeros((height, width, 3), dtype=np.float32),
        timestamp=timestamp,
        frame_id=frame_id,
        scene_id=scene_id,
        depth=np.ones((height, width), dtype=np.float32),
        motion_vectors=np.zeros((height, width, 2), dtype=np.float32),
        noise=np.zeros((height, width, 3), dtype=np.float32),
        conditioning=np.asarray([0.4, 0.8], dtype=np.float32),
        camera_cut=camera_cut,
    )


def test_frame_packet_exposes_only_inputs_that_exist():
    frame = FramePacket(
        color=np.zeros((10, 20, 3), dtype=np.float16),
        timestamp=1.25,
        frame_id=7,
    )

    assert frame.size == (20, 10)
    assert set(frame.model_inputs()) == {"color"}


def test_frame_packet_preserves_real_depth_motion_noise_and_conditioning():
    frame = make_frame(1, 0.01)

    assert set(frame.model_inputs()) == {
        "color", "depth", "motion_vectors", "noise", "conditioning"
    }
    assert frame.motion_vectors.shape == (3, 4, 2)
    assert frame.conditioning.shape == (2,)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("color", np.zeros((4, 4), dtype=np.float32), "rank 3"),
        ("depth", np.zeros((2, 2), dtype=np.float32), "depth must match"),
        ("motion_vectors", np.zeros((3, 4, 3), dtype=np.float32), "motion_vectors must"),
        ("noise", np.zeros((3, 4, 1), dtype=np.float32), "noise must have shape"),
        ("conditioning", np.asarray([np.nan], dtype=np.float32), "conditioning contains"),
    ],
)
def test_frame_packet_rejects_malformed_auxiliary_data(field, value, match):
    kwargs = {
        "color": np.zeros((3, 4, 3), dtype=np.float32),
        "timestamp": 0.01,
        "frame_id": 1,
    }
    kwargs[field] = value
    with pytest.raises(FrameValidationError, match=match):
        FramePacket(**kwargs)


def test_temporal_history_is_empty_until_an_output_is_stored():
    history = TemporalHistory()
    assert history.history_for(make_frame(1, 0.01)) is None
    assert history.valid is False


def test_temporal_history_returns_copy_of_previous_output():
    history = TemporalHistory()
    first = make_frame(1, 0.01)
    output = np.full((first.height, first.width, 3), 0.25, dtype=np.float16)
    history.store(first, output)

    previous = history.history_for(make_frame(2, 0.02))
    assert previous is not None
    assert previous.dtype == np.float32
    assert previous[0, 0, 0] == pytest.approx(0.25)
    previous[0, 0, 0] = 99.0
    assert history.history_for(make_frame(2, 0.02))[0, 0, 0] == pytest.approx(0.25)


@pytest.mark.parametrize(
    "next_frame,reason",
    [
        (make_frame(2, 0.02, camera_cut=True), "camera_cut"),
        (make_frame(2, 0.02, size=(8, 3)), "resolution_changed"),
        (make_frame(2, 0.02, scene_id="scene-b"), "scene_changed"),
        (make_frame(1, 0.02), "frame_order_changed"),
        (make_frame(2, 0.00), "timestamp_not_monotonic"),
        (make_frame(2, 1.0), "frame_gap"),
    ],
)
def test_temporal_history_resets_on_discontinuities(next_frame, reason):
    history = TemporalHistory(max_gap_seconds=0.5)
    first = make_frame(1, 0.01)
    history.store(first, np.zeros((first.height, first.width, 3), dtype=np.float32))

    assert history.history_for(next_frame) is None
    assert history.valid is False
    assert history.last_reset_reason == reason


def test_temporal_history_refuses_values_outside_normalized_rgb_range():
    history = TemporalHistory()
    frame = make_frame(1, 0.01)

    with pytest.raises(FrameValidationError, match="normalized to the \\[0, 1\\] range"):
        history.store(frame, np.full((frame.height, frame.width, 3), 1.5, dtype=np.float32))

    assert history.valid is False


def test_temporal_history_refuses_wrong_output_shape():
    history = TemporalHistory()
    frame = make_frame(1, 0.01)
    with pytest.raises(FrameValidationError, match="rendered_output must be RGB HWC"):
        history.store(frame, np.zeros((2, 2, 3), dtype=np.float32))


def test_temporal_history_camera_cut_frame_can_become_new_history():
    history = TemporalHistory()
    frame = make_frame(3, 0.03, camera_cut=True)
    history.store(frame, np.ones((frame.height, frame.width, 3), dtype=np.float32))

    following = history.history_for(make_frame(4, 0.04))
    assert following is not None
    assert np.all(following == 1.0)



def test_frame_packet_rejects_complex_color_values():
    with pytest.raises(FrameValidationError, match="real-valued numeric"):
        FramePacket(
            color=np.ones((2, 2, 3), dtype=np.complex64),
            timestamp=0.0,
            frame_id=0,
        )
