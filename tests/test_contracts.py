"""Fast CPU-only tests for the image contract used by the inference engine."""

import numpy as np
import pytest
from PIL import Image

from trss.contracts import image_to_tensor, parse_value_range, tensor_to_image


def test_parse_value_range_accepts_normalized_and_signed_ranges():
    assert parse_value_range("0:1") == (0.0, 1.0)
    assert parse_value_range("-1:1") == (-1.0, 1.0)


@pytest.mark.parametrize("text", ["", "0", "a:b", "2:1", "nan:1", "0:inf"])
def test_parse_value_range_rejects_invalid_ranges(text):
    with pytest.raises(ValueError):
        parse_value_range(text)


def test_image_to_tensor_nchw_has_batch_channel_height_width():
    image = Image.new("RGB", (4, 2), (255, 128, 0))
    tensor = image_to_tensor(image, layout="NCHW", width=4, height=2)
    assert tensor.shape == (1, 3, 2, 4)
    assert tensor.dtype == np.float32
    assert tensor[0, 0, 0, 0] == pytest.approx(1.0)
    assert tensor[0, 1, 0, 0] == pytest.approx(128 / 255)


def test_image_to_tensor_nhwc_has_batch_height_width_channel():
    image = Image.new("RGB", (3, 2), (10, 20, 30))
    tensor = image_to_tensor(image, layout="NHWC", width=3, height=2)
    assert tensor.shape == (1, 2, 3, 3)
    assert tensor[0, 0, 0].tolist() == pytest.approx([10 / 255, 20 / 255, 30 / 255])


def test_image_to_tensor_resizes_to_fixed_model_input():
    image = Image.new("RGB", (64, 32), (255, 255, 255))
    tensor = image_to_tensor(image, layout="NCHW", width=16, height=8)
    assert tensor.shape == (1, 3, 8, 16)


def test_tensor_to_image_round_trips_rgb_pixels():
    source = np.asarray([[[[1.0]], [[0.0]], [[0.5]]]], dtype=np.float32)
    image = tensor_to_image(source, layout="NCHW")
    assert image.size == (1, 1)
    assert image.getpixel((0, 0)) == (255, 0, 128)


def test_tensor_to_image_replaces_invalid_numbers_and_clamps_range():
    values = np.array([[[[np.nan, 2.0, -1.0]]]], dtype=np.float32)
    image = tensor_to_image(values, layout="NHWC")
    assert image.getpixel((0, 0)) == (0, 255, 0)


def test_tensor_to_image_rejects_non_batch_one_output():
    with pytest.raises(ValueError, match="rank-four"):
        tensor_to_image(np.zeros((3, 5, 5), dtype=np.float32), layout="NCHW")



def test_image_to_tensor_supports_bgr_models():
    image = Image.new("RGB", (1, 1), (10, 20, 30))
    tensor = image_to_tensor(
        image, layout="NCHW", width=1, height=1, channel_order="BGR"
    )
    assert tensor[0, :, 0, 0].tolist() == pytest.approx(
        [30 / 255, 20 / 255, 10 / 255]
    )


def test_tensor_to_image_supports_bgr_model_outputs():
    source = np.asarray([[[[30 / 255]], [[20 / 255]], [[10 / 255]]]], dtype=np.float32)
    image = tensor_to_image(source, layout="NCHW", channel_order="BGR")
    assert image.getpixel((0, 0)) == (10, 20, 30)


@pytest.mark.parametrize("value_range", [(1.0, 1.0), (2.0, 1.0), (float("nan"), 1.0)])
def test_image_to_tensor_rejects_invalid_value_ranges(value_range):
    with pytest.raises(ValueError, match="value range"):
        image_to_tensor(
            Image.new("RGB", (1, 1)),
            layout="NCHW",
            width=1,
            height=1,
            value_range=value_range,
        )


def test_tensor_to_image_rejects_invalid_value_ranges():
    with pytest.raises(ValueError, match="value range"):
        tensor_to_image(
            np.zeros((1, 3, 1, 1), dtype=np.float32),
            layout="NCHW",
            value_range=(0.5, 0.5),
        )



def test_tensor_to_image_rejects_complex_model_outputs():
    with pytest.raises(ValueError, match="Complex-valued"):
        tensor_to_image(
            np.ones((1, 3, 2, 2), dtype=np.complex64),
            layout="NCHW",
        )
