"""Model adapter tests; no OpenVINO runtime or physical GPU required."""

import numpy as np
import pytest

from xess5.rendering import FramePacket
from xess5.rendering.adapter import (
    AdapterSpecError,
    ModelAdapterSpec,
    adapt_frame,
)


def binding(source, layout, source_range, target_range, resize="nearest", **extra):
    return {
        "source": source,
        "layout": layout,
        "source_range": list(source_range),
        "target_range": list(target_range),
        "resize": resize,
        **extra,
    }


def make_frame():
    return FramePacket(
        color=np.asarray(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 255]],
            ],
            dtype=np.uint8,
        ),
        depth=np.asarray([[0.0, 0.5], [0.75, 1.0]], dtype=np.float32),
        motion_vectors=np.asarray(
            [[[1.0, -1.0], [0.0, 0.5]], [[-0.5, 0.25], [0.75, -0.25]]],
            dtype=np.float32,
        ),
        conditioning=np.asarray([0.2, 0.8], dtype=np.float32),
        frame_id=10,
        timestamp=2.0,
        scene_id="scene-1",
    )


def test_spec_requires_explicit_version_and_input_bindings():
    with pytest.raises(AdapterSpecError, match="version must be 1"):
        ModelAdapterSpec.from_dict({"version": 2, "inputs": {"x": {}}})
    with pytest.raises(AdapterSpecError, match="non-empty inputs"):
        ModelAdapterSpec.from_dict({"version": 1, "inputs": {}})


def test_spec_rejects_unknown_fields_and_bad_ranges():
    with pytest.raises(AdapterSpecError, match="unknown fields"):
        ModelAdapterSpec.from_dict({
            "version": 1,
            "inputs": {
                "color": {
                    **binding("color", "NCHW", (0, 255), (0, 1)),
                    "magic_mode": True,
                }
            },
        })

    with pytest.raises(AdapterSpecError, match="finite and ascending"):
        ModelAdapterSpec.from_dict({
            "version": 1,
            "inputs": {
                "color": binding("color", "NCHW", (1, 1), (0, 1))
            },
        })


def test_color_becomes_normalized_nchw_and_respects_static_model_shape():
    frame = make_frame()
    spec = ModelAdapterSpec.from_dict({
        "version": 1,
        "inputs": {
            "image_input": binding("color", "NCHW", (0, 255), (0, 1))
        },
    })
    output = adapt_frame(
        frame, spec, {"image_input": (1, 3, 2, 2)}
    )["image_input"]

    assert output.shape == (1, 3, 2, 2)
    assert output.dtype == np.float32
    assert output[0, :, 0, 0].tolist() == pytest.approx([1, 0, 0])
    assert output[0, :, 1, 0].tolist() == pytest.approx([0, 0, 1])


def test_color_can_be_converted_to_bgr_and_nhwc():
    frame = make_frame()
    spec = ModelAdapterSpec.from_dict({
        "version": 1,
        "inputs": {
            "color": binding(
                "color", "NHWC", (0, 255), (-1, 1), channel_order="BGR"
            )
        },
    })
    output = adapt_frame(frame, spec, {"color": (1, 2, 2, 3)})["color"]

    assert output.shape == (1, 2, 2, 3)
    assert output[0, 0, 0].tolist() == pytest.approx([-1, -1, 1])


def test_bgr_conversion_preserves_rgba_alpha_as_bgra():
    frame = FramePacket(
        color=np.asarray([[[10, 20, 30, 40]]], dtype=np.uint8),
        timestamp=1.0,
        frame_id=0,
    )
    spec = ModelAdapterSpec.from_dict({
        "version": 1,
        "inputs": {
            "rgba": binding(
                "color", "NHWC", (0, 255), (0, 255), channel_order="BGR"
            )
        },
    })

    output = adapt_frame(frame, spec, {"rgba": (1, 1, 1, 4)})["rgba"]

    assert output[0, 0, 0].tolist() == pytest.approx([30, 20, 10, 40])


def test_depth_is_resized_and_expanded_to_a_single_channel():
    frame = make_frame()
    spec = ModelAdapterSpec.from_dict({
        "version": 1,
        "inputs": {
            "depth": binding("depth", "NCHW", (0, 1), (0, 1), resize="nearest")
        },
    })
    output = adapt_frame(frame, spec, {"depth": (1, 1, 4, 6)})["depth"]

    assert output.shape == (1, 1, 4, 6)
    assert np.all(output[0, 0, :2, :3] == 0.0)
    assert np.all(output[0, 0, 2:, 3:] == 1.0)


def test_motion_vectors_keep_two_channels_and_preserve_declared_units():
    frame = make_frame()
    spec = ModelAdapterSpec.from_dict({
        "version": 1,
        "inputs": {
            "motion": binding(
                "motion_vectors", "NCHW", (-1, 1), (-1, 1)
            )
        },
    })
    output = adapt_frame(frame, spec, {"motion": (1, 2, 2, 2)})["motion"]

    assert output.shape == (1, 2, 2, 2)
    assert output[0, :, 0, 0].tolist() == pytest.approx([1, -1])


def test_conditioning_vector_is_mapped_to_nc_without_image_resizing():
    frame = make_frame()
    spec = ModelAdapterSpec.from_dict({
        "version": 1,
        "inputs": {
            "condition": binding("conditioning", "NC", (0, 1), (-1, 1))
        },
    })
    output = adapt_frame(frame, spec, {"condition": (1, 2)})["condition"]

    assert output.shape == (1, 2)
    assert output[0].tolist() == pytest.approx([-0.6, 0.6])


def test_history_binding_requires_real_previous_output():
    frame = make_frame()
    spec = ModelAdapterSpec.from_dict({
        "version": 1,
        "inputs": {
            "previous_frame": binding("history", "NCHW", (0, 1), (0, 1))
        },
    })
    with pytest.raises(AdapterSpecError, match="requires temporal history"):
        adapt_frame(frame, spec, {"previous_frame": (1, 3, 2, 2)})


def test_adapter_refuses_unavailable_depth_instead_of_inventing_it():
    frame = FramePacket(
        color=np.zeros((2, 2, 3), dtype=np.uint8),
        timestamp=1.0,
        frame_id=1,
    )
    spec = ModelAdapterSpec.from_dict({
        "version": 1,
        "inputs": {"depth": binding("depth", "NCHW", (0, 1), (0, 1))},
    })
    with pytest.raises(AdapterSpecError, match="did not supply it"):
        adapt_frame(frame, spec, {"depth": (1, 1, 2, 2)})


def test_spec_must_match_every_input_name_exactly():
    frame = make_frame()
    spec = ModelAdapterSpec.from_dict({
        "version": 1,
        "inputs": {"unknown": binding("color", "NCHW", (0, 255), (0, 1))},
    })
    with pytest.raises(AdapterSpecError, match="spec missing model ports"):
        adapt_frame(frame, spec, {"color": (1, 3, 2, 2)})


def test_bilinear_resampling_has_expected_midpoint():
    frame = FramePacket(
        color=np.asarray([[[0, 0, 0], [100, 100, 100]]], dtype=np.float32),
        timestamp=0.0,
        frame_id=0,
    )
    spec = ModelAdapterSpec.from_dict({
        "version": 1,
        "inputs": {
            "color": binding("color", "NHWC", (0, 100), (0, 1), resize="bilinear")
        },
    })
    output = adapt_frame(frame, spec, {"color": (1, 1, 4, 3)})["color"]

    assert output.shape == (1, 1, 4, 3)
    assert output[0, 0, 1, 0] == pytest.approx(0.25)
    assert output[0, 0, 2, 0] == pytest.approx(0.75)



def test_preview_binding_converts_declared_nchw_output_to_rgb_float():
    from xess5.rendering.adapter import preview_from_outputs

    spec = ModelAdapterSpec.from_dict({
        "version": 1,
        "inputs": {"color": binding("color", "NCHW", (0, 255), (0, 1))},
        "preview_output": {
            "port": "rgb",
            "layout": "NCHW",
            "source_range": [-1, 1],
            "channel_order": "RGB",
        },
    })
    raw = np.asarray([[[[-1, 1]], [[0, 0]], [[1, -1]]]], dtype=np.float32)
    preview = preview_from_outputs({"rgb": raw}, spec, target_size=(2, 1))

    assert preview.shape == (1, 2, 3)
    assert preview.dtype == np.float32
    assert preview[0, 0].tolist() == pytest.approx([0, 0.5, 1])
    assert preview[0, 1].tolist() == pytest.approx([1, 0.5, 0])


def test_preview_binding_reorders_bgr_to_rgb():
    from xess5.rendering.adapter import preview_from_outputs

    spec = ModelAdapterSpec.from_dict({
        "version": 1,
        "inputs": {"color": binding("color", "NCHW", (0, 255), (0, 1))},
        "preview_output": {
            "port": "rgb",
            "layout": "NHWC",
            "source_range": [0, 1],
            "channel_order": "BGR",
        },
    })
    raw = np.asarray([[[[0.1, 0.2, 0.9]]]], dtype=np.float32)
    preview = preview_from_outputs({"rgb": raw}, spec, target_size=(1, 1))

    assert preview[0, 0].tolist() == pytest.approx([0.9, 0.2, 0.1])


def test_preview_binding_refuses_feature_tensors():
    from xess5.rendering.adapter import preview_from_outputs

    spec = ModelAdapterSpec.from_dict({
        "version": 1,
        "inputs": {"color": binding("color", "NCHW", (0, 255), (0, 1))},
        "preview_output": {
            "port": "features",
            "layout": "NCHW",
            "source_range": [0, 1],
        },
    })
    with pytest.raises(AdapterSpecError, match="Feature tensors"):
        preview_from_outputs(
            {"features": np.zeros((1, 16, 2, 2), dtype=np.float32)},
            spec,
            target_size=(2, 2),
        )
