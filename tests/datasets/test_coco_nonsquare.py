# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Non-square ``(height, width)`` resolutions in the COCO transform builders."""

import warnings
from typing import Any

import pytest
import torch
from PIL import Image

from rfdetr.datasets.coco import (
    _build_train_resize_config,
    compute_multi_scale_scales,
    draft_size_for_transforms,
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)

_BACKENDS = [
    pytest.param(None, id="torchvision"),
    pytest.param({"HorizontalFlip": {"p": 0.5}}, id="albumentations"),
]


def _sample() -> tuple[Image.Image, dict[str, Any]]:
    """Return a 200x120 image with one box."""
    target = {
        "boxes": torch.tensor([[20.0, 30.0, 80.0, 90.0]]),
        "labels": torch.tensor([1]),
        "size": torch.tensor([120, 200]),
        "orig_size": torch.tensor([120, 200]),
    }
    return Image.new("RGB", (200, 120)), target


def _build(factory: Any, image_set: str, resolution: Any, **kwargs: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return factory(image_set, resolution, **kwargs)


class TestMultiScaleScales:
    def test_int_resolution_keeps_side_lengths(self) -> None:
        assert compute_multi_scale_scales(512) == [320, 384, 448, 512, 576, 640, 704, 768]

    @pytest.mark.parametrize("resolution", [(384, 768), (768, 384), (512, 512)])
    def test_pair_resolution_yields_block_aligned_pairs(self, resolution: tuple[int, int]) -> None:
        scales = compute_multi_scale_scales(resolution, patch_size=16, num_windows=4)

        assert resolution in scales
        for height, width in scales:
            assert height % 64 == 0 and width % 64 == 0
            # orientation and aspect follow the base resolution
            assert (height <= width) == (resolution[0] <= resolution[1])
            assert height / width == pytest.approx(resolution[0] / resolution[1], rel=0.2)

    def test_pair_resolution_drops_too_small_scales(self) -> None:
        scales = compute_multi_scale_scales((128, 256), patch_size=16, num_windows=4)
        assert min(min(scale) for scale in scales) == 128

    def test_draft_size_uses_longer_side(self) -> None:
        assert draft_size_for_transforms("train", (384, 768)) == 768
        assert draft_size_for_transforms("train", 512, multi_scale=True) == 768
        assert draft_size_for_transforms("val", (384, 768)) is None


class TestTrainResizeConfig:
    def test_square_branch_resizes_to_height_width(self) -> None:
        (entry,) = _build_train_resize_config([(64, 96)], square=True, scale_jitter=False)
        assert entry == {"OneOf": {"transforms": [{"Resize": {"height": 64, "width": 96}}]}}

    def test_aspect_preserving_branch_uses_short_side(self) -> None:
        (entry,) = _build_train_resize_config([(64, 96), (128, 192)], square=False, scale_jitter=False)
        assert entry["Sequential"]["transforms"][0] == {"SmallestMaxSize": {"max_size": [64, 128]}}


class TestNonSquareTransforms:
    @pytest.mark.parametrize("aug_config", _BACKENDS)
    @pytest.mark.parametrize("image_set", ["train", "val", "test", "val_speed"])
    @pytest.mark.parametrize("scale_jitter", [True, False])
    def test_square_div_64_resizes_to_exact_height_width(
        self, image_set: str, aug_config: Any, scale_jitter: bool
    ) -> None:
        transforms = _build(
            make_coco_transforms_square_div_64, image_set, (64, 96), aug_config=aug_config, scale_jitter=scale_jitter
        )
        image, target = transforms(*_sample())

        assert image.shape == (3, 64, 96)
        assert target["size"].tolist() == [64, 96]

    @pytest.mark.parametrize("aug_config", _BACKENDS)
    def test_multi_scale_square_div_64_draws_from_pair_ladder(self, aug_config: Any) -> None:
        transforms = _build(
            make_coco_transforms_square_div_64,
            "train",
            (128, 192),
            multi_scale=True,
            patch_size=16,
            num_windows=2,
            aug_config=aug_config,
        )
        ladder = set(compute_multi_scale_scales((128, 192), patch_size=16, num_windows=2))
        for _ in range(8):
            image, _ = transforms(*_sample())
            assert tuple(image.shape[-2:]) in ladder

    @pytest.mark.parametrize("aug_config", _BACKENDS)
    @pytest.mark.parametrize("image_set", ["train", "val", "test"])
    def test_aspect_preserving_pipeline_uses_short_side(self, image_set: str, aug_config: Any) -> None:
        transforms = _build(make_coco_transforms, image_set, (60, 96), aug_config=aug_config, scale_jitter=False)
        image, _ = transforms(*_sample())

        # the 200x120 source keeps its aspect ratio with the short side at min(60, 96)
        assert tuple(image.shape[-2:]) == (60, 100)

    @pytest.mark.parametrize("aug_config", _BACKENDS)
    def test_val_speed_resizes_to_exact_height_width(self, aug_config: Any) -> None:
        transforms = _build(make_coco_transforms, "val_speed", (64, 96), aug_config=aug_config)
        image, _ = transforms(*_sample())
        assert image.shape == (3, 64, 96)
