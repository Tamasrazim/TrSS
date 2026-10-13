"""Tests for the HNet-style model-specific frame contract."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from xess5.rendering.hnet import (
    HNetContractError,
    HNetControls,
    HNetFrameEngine,
    NetworkGeometry,
    compose_head,
    deterministic_noise,
    make_features,
    network_geometry,
    scaled_color,
)


@pytest.mark.parametrize(
    "width,height,expected",
    [
        (1024, 576, (1024, 576)),
        (563, 317, (640, 320)),
        (1920, 1080, (1920, 1152)),
        (1280, 720, (1344, 768)),
        (64, 48, (384, 320)),
    ],
)
def test_network_geometry_matches_documented_graph_alignment(width, height, expected):
    geometry = network_geometry(width, height)
    assert (geometry.network_width, geometry.network_height) == expected
    assert geometry.network_width >= width
    assert geometry.network_height >= height
    assert geometry.network_width % 64 == 0
    assert geometry.network_height % 64 == 0


def test_geometry_mirrors_without_repeating_the_last_pixel():
    geometry = NetworkGeometry(
        output_width=4,
        output_height=3,
        network_width=7,
        network_height=5,
    )
    source = np.arange(3 * 4, dtype=np.float32).reshape(3, 4, 1)
    extended = geometry.extend(source)

    assert extended.shape == (5, 7, 1)
    assert extended[0, :, 0].tolist() == [0, 1, 2, 3, 2, 1, 0]
    assert extended[:, 0, 0].tolist() == [0, 4, 8, 4, 0]
    assert np.array_equal(geometry.crop(extended), source)


def test_mirror_padding_repeats_when_the_network_field_is_much_larger():
    # Tiny frames can be padded to a much larger minimum neural-network field.
    # Reflection must continue periodically instead of turning into a flat border.
    geometry = NetworkGeometry(
        output_width=4,
        output_height=3,
        network_width=10,
        network_height=7,
    )
    source = np.arange(3 * 4, dtype=np.float32).reshape(3, 4, 1)

    extended = geometry.extend(source)

    assert extended[0, :, 0].tolist() == [0, 1, 2, 3, 2, 1, 0, 1, 2, 3]
    assert extended[:, 0, 0].tolist() == [0, 4, 8, 4, 0, 4, 8]
    assert np.array_equal(geometry.crop(extended), source)


def test_single_pixel_source_can_be_mirror_padded_safely():
    geometry = NetworkGeometry(
        output_width=1,
        output_height=1,
        network_width=6,
        network_height=5,
    )
    source = np.asarray([[[0.375]]], dtype=np.float32)

    extended = geometry.extend(source)

    assert extended.shape == (5, 6, 1)
    assert np.all(extended == np.float32(0.375))
    assert np.array_equal(geometry.crop(extended), source)


def test_noise_is_repeatable_for_a_frame_and_changes_with_frame_index():
    first = deterministic_noise(9, 13, frame_index=4)
    again = deterministic_noise(9, 13, frame_index=4)
    later = deterministic_noise(9, 13, frame_index=5)

    assert first.shape == (9, 13, 3)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.array_equal(first, again)
    assert not np.array_equal(first, later)


def test_noise_rejects_invalid_frame_indices():
    with pytest.raises(HNetContractError, match="non-negative integer"):
        deterministic_noise(4, 4, -1)
    with pytest.raises(HNetContractError, match="non-negative integer"):
        deterministic_noise(4, 4, 1.5)


def test_scaled_color_keeps_three_distinct_half_rounding_points():
    color = np.asarray([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    value = scaled_color(color)

    expected = np.asarray(
        [[[-0.0625, 0.0, 0.0625]]], dtype=np.float32
    )
    assert value.dtype == np.float32
    assert np.array_equal(value, expected)


def test_feature_tensor_contains_documented_16_channels():
    color = np.asarray(
        [
            [[0.0, 0.25, 1.0], [0.5, 0.75, 0.0]],
            [[1.0, 0.5, 0.25], [0.25, 1.0, 0.5]],
        ],
        dtype=np.float32,
    )
    geometry = NetworkGeometry(2, 2, 2, 2)
    features, returned_geometry = make_features(
        color,
        geometry=geometry,
        frame_index=2,
        controls=HNetControls(
            style_index=2,
            tone_strength=0.5,
            structure_strength=1.5,
        ),
    )

    assert returned_geometry == geometry
    assert features.shape == (2, 2, 16)
    assert features.dtype == np.float32
    assert np.array_equal(features[..., 0:3], deterministic_noise(2, 2, 2))
    assert np.all(features[..., 3] == 1.0)
    assert np.array_equal(features[..., 4:7], scaled_color(color))
    # With no history supplied, the current color is copied into history channels.
    assert np.array_equal(features[..., 7:10], features[..., 4:7])
    assert np.all(features[..., 10] == np.float32(_half_scalar(2.0 / 128)))
    assert np.all(features[..., 11] == np.float32(0.5))
    assert np.all(features[..., 12] == np.float32(1.5))
    assert np.all(features[..., 13] == -1.0)
    assert np.all(features[..., 14] == -1.0)
    assert np.all(features[..., 15] == 0.0)


def _half_scalar(value: float) -> float:
    return np.asarray(value, dtype=np.float32).astype(np.float16).astype(np.float32).item()


def test_history_is_scaled_and_mirrored_separately_from_current_color():
    color = np.full((2, 2, 3), 0.2, dtype=np.float32)
    history = np.full((2, 2, 3), 0.8, dtype=np.float32)
    geometry = NetworkGeometry(2, 2, 4, 4)

    features, _ = make_features(color, history=history, geometry=geometry)
    assert np.all(features[..., 4:7] == scaled_color(geometry.extend(color)))
    assert np.all(features[..., 7:10] == scaled_color(geometry.extend(history)))
    assert not np.array_equal(features[..., 4:7], features[..., 7:10])


def test_feature_builder_rejects_non_normalized_or_wrong_shaped_color():
    with pytest.raises(HNetContractError, match="normalized to 0..1"):
        make_features(np.full((2, 2, 3), 255, dtype=np.uint8))
    with pytest.raises(HNetContractError, match="HxWx3"):
        make_features(np.zeros((2, 2), dtype=np.float32))


def test_head_composition_blends_to_identity_history_using_half_logit():
    color = np.full((2, 3, 3), 0.25, dtype=np.float32)
    history = np.full((2, 3, 3), 0.75, dtype=np.float32)
    head = np.zeros((2, 3, 4), dtype=np.float32)

    rendered, next_history = compose_head(head, color, history=history)
    # Zero RGB residual leaves the current image unchanged before temporal blending.
    # A zero logit yields a gate of half the published blend constant.
    alpha = 0.5 * float(np.asarray(0.73974609375, dtype=np.float32).astype(np.float16))
    expected = 0.25 + alpha * (0.75 - 0.25)
    assert np.allclose(rendered, expected, atol=1e-6)
    assert np.array_equal(next_history, rendered)
    assert rendered.dtype == np.float32


def test_head_composition_uses_current_color_as_first_history():
    color = np.full((1, 1, 3), 0.5, dtype=np.float32)
    head = np.zeros((1, 1, 4), dtype=np.float32)
    rendered, history = compose_head(head, color)

    assert np.array_equal(rendered, color)
    assert np.array_equal(history, color)


def test_head_composition_rejects_wrong_channels_and_nan():
    color = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(HNetContractError, match="head must have shape"):
        compose_head(np.zeros((2, 2, 3), dtype=np.float32), color)
    bad = np.zeros((2, 2, 4), dtype=np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(HNetContractError, match="NaN"):
        compose_head(bad, color)


class FakeHNetTensorEngine:
    def __init__(self, input_shape=(1, 16, None, None), output_layout="NCHW"):
        self.input_names = ("features",)
        self.output_names = ("head",)
        self.input_shapes = {"features": input_shape}
        self.output_layout = output_layout
        self.calls = 0

    def infer(self, inputs):
        self.calls += 1
        value = inputs["features"]
        assert value.ndim == 4 and value.shape[0] == 1 and value.shape[1] == 16
        height, width = value.shape[2:]
        head = np.zeros((1, 4, height, width), dtype=np.float32)
        return SimpleNamespace(
            outputs={"head": head},
            device="GPU.1",
            device_name="Intel Arc B580 Graphics",
            compile_ms=7.0 if self.calls == 1 else 0.0,
            inference_ms=3.0,
            total_ms=3.5,
        )


def test_hnet_engine_runs_the_16_to_4_frame_contract_and_crops_output():
    color = np.full((320, 320, 3), 0.4, dtype=np.float32)
    engine = HNetFrameEngine(FakeHNetTensorEngine())

    result = engine.process_frame(color, frame_index=3)

    assert result.image.shape == (320, 320, 3)
    assert result.history.shape == (320, 320, 3)
    assert result.head.shape == (320, 320, 4)
    assert result.geometry.is_identity
    assert np.allclose(result.image, color)
    assert result.device_name == "Intel Arc B580 Graphics"
    assert result.compile_ms == 7.0
    assert result.inference_ms == 3.0


def test_hnet_engine_rejects_the_generic_rgb_only_model_contract():
    with pytest.raises(HNetContractError, match=r"\[1,16,H,W\]"):
        HNetFrameEngine(FakeHNetTensorEngine(input_shape=(1, 3, None, None)))
