# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for ChannelSubset and the motion-infused augmentation presets.

The invariant under test throughout is that a motion channel encodes "no motion" as
exactly zero. Any augmentation that lifts a zero off zero invents motion the upstream
pipeline never measured, so the presets must confine additive/nonlinear intensity ops
to the appearance channels and use only multiplicative ops on the motion channels.
"""

from typing import Any, Dict, List

import numpy as np
import pytest

from rfdetr.datasets.aug_config import (
    AUG_MOTION_GREY,
    AUG_MOTION_ONLY,
    AUG_MOTION_RGB,
)
from rfdetr.datasets.transforms import _build_albu_transform, _is_geometric_transform

# preset, channel count, motion channel indices, appearance channel indices
PRESETS = [
    pytest.param(AUG_MOTION_RGB, 4, [3], [0, 1, 2], id="rgb_plus_flow"),
    pytest.param(AUG_MOTION_GREY, 3, [0, 2], [1], id="motion_grey_motion"),
    pytest.param(AUG_MOTION_ONLY, 3, [0, 1, 2], [], id="all_motion"),
]


def _build(preset: List[Dict[str, Any]]) -> list:
    """Build every entry of a list-form preset."""
    built = []
    for entry in preset:
        (name, params), = entry.items()
        built.append(_build_albu_transform(name, params))
    return built


def _pixel_transforms(built: list) -> list:
    """The non-geometric entries, i.e. the ones that may alter pixel values."""
    return [t for t in built if not _is_geometric_transform(t)]


def _motion_image(channels: int) -> np.ndarray:
    """An image whose left third is true-zero background, as a motion channel would be."""
    img = np.zeros((32, 32, channels), dtype=np.uint8)
    img[:, 11:22] = 110
    img[:, 22:] = 230
    return img


class TestChannelSubset:
    def test_leaves_unselected_channels_bit_identical(self) -> None:
        img = np.full((16, 16, 4), 120, dtype=np.uint8)
        transform = _build_albu_transform(
            "ChannelSubset",
            {
                "channels": [0, 1, 2],
                "transforms": [{"RandomBrightnessContrast": {"brightness_limit": 0.5, "contrast_limit": 0.5, "p": 1.0}}],
                "p": 1.0,
            },
        )
        for _ in range(25):
            out = transform(image=img)["image"]
            assert np.array_equal(out[:, :, 3], img[:, :, 3])

    def test_rejects_out_of_range_channels(self) -> None:
        transform = _build_albu_transform(
            "ChannelSubset",
            {"channels": [7], "transforms": [{"RandomBrightnessContrast": {"p": 1.0}}], "p": 1.0},
        )
        with pytest.raises(ValueError, match="out of range"):
            transform(image=np.zeros((8, 8, 3), dtype=np.uint8))

    def test_is_not_geometric_so_boxes_are_untouched(self) -> None:
        transform = _build_albu_transform(
            "ChannelSubset",
            {"channels": [0], "transforms": [{"RandomBrightnessContrast": {"p": 1.0}}], "p": 1.0},
        )
        assert _is_geometric_transform(transform) is False


class TestMotionPresets:
    @pytest.mark.parametrize("preset, channels, motion, appearance", PRESETS)
    def test_every_transform_builds(
        self, preset: List[Dict[str, Any]], channels: int, motion: List[int], appearance: List[int]
    ) -> None:
        # build_albumentations_transforms swallows failures with a warning, so a preset
        # that names an incompatible transform degrades silently to fewer augmentations.
        assert len(_build(preset)) == len(preset)

    @pytest.mark.parametrize("preset, channels, motion, appearance", PRESETS)
    def test_no_vertical_flip(
        self, preset: List[Dict[str, Any]], channels: int, motion: List[int], appearance: List[int]
    ) -> None:
        assert not any("VerticalFlip" in entry for entry in preset)

    @pytest.mark.parametrize("preset, channels, motion, appearance", PRESETS)
    def test_motion_background_stays_exactly_zero(
        self, preset: List[Dict[str, Any]], channels: int, motion: List[int], appearance: List[int]
    ) -> None:
        img = _motion_image(channels)
        transforms = _pixel_transforms(_build(preset))
        for _ in range(200):
            out = img.copy()
            for transform in transforms:
                out = transform(image=out)["image"]
            for channel in motion:
                plane = out[:, :, channel]
                if (plane == 0).all():
                    continue  # modality dropout fired; the whole channel is blank
                assert (plane[:, :11] == 0).all(), "an augmentation lifted still background off zero"

    @pytest.mark.parametrize("preset, channels, motion, appearance", PRESETS)
    def test_values_never_exceed_the_dtype_ceiling(
        self, preset: List[Dict[str, Any]], channels: int, motion: List[int], appearance: List[int]
    ) -> None:
        img = _motion_image(channels)
        transforms = _pixel_transforms(_build(preset))
        for _ in range(100):
            out = img.copy()
            for transform in transforms:
                out = transform(image=out)["image"]
            assert out.dtype == np.uint8
            assert out.max() <= 255

    @pytest.mark.parametrize("preset, channels, motion, appearance", PRESETS)
    def test_modality_dropout_fires_sometimes_but_not_always(
        self, preset: List[Dict[str, Any]], channels: int, motion: List[int], appearance: List[int]
    ) -> None:
        img = _motion_image(channels)
        transforms = _pixel_transforms(_build(preset))
        blanked = 0
        trials = 400
        for _ in range(trials):
            out = img.copy()
            for transform in transforms:
                out = transform(image=out)["image"]
            if any((out[:, :, c] == 0).all() for c in motion):
                blanked += 1
        assert 0 < blanked < trials, "dropout must fire on some frames and not on others"

    def test_all_motion_never_blanks_every_channel(self) -> None:
        # Blanking all three would leave an empty image that still carries boxes: that is
        # label noise, not augmentation. The OneOf must pick at most one channel.
        img = _motion_image(3)
        transforms = _pixel_transforms(_build(AUG_MOTION_ONLY))
        for _ in range(600):
            out = img.copy()
            for transform in transforms:
                out = transform(image=out)["image"]
            blank = [c for c in range(3) if (out[:, :, c] == 0).all()]
            assert len(blank) <= 1, f"channels {blank} blanked simultaneously"
