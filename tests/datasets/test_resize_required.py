# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""The resize stage must never be silently dropped.

The resize to the model's training resolution is built through the same Albumentations config path as the optional
augmentations, which logs-and-skips whatever it cannot construct. Losing the resize is not a degraded augmentation
policy: images reach the model at their native size while boxes are normalised against that size, so the model learns
a geometry that inference never reproduces. These tests pin the failure to a loud one.
"""

import pytest

from rfdetr.datasets import coco, transforms
from rfdetr.datasets.transforms import AlbumentationsWrapper


class TestRequireAlbumentations:
    """A missing albumentations install raises rather than yielding zero transforms."""

    def test_from_config_raises_when_albumentations_missing(self, monkeypatch):
        monkeypatch.setattr(transforms, "alb", None)
        with pytest.raises(ImportError, match="albumentations is required"):
            AlbumentationsWrapper.from_config([{"Resize": {"height": 640, "width": 640}}])

    def test_empty_config_still_returns_empty(self, monkeypatch):
        """An explicitly empty augmentation config is legitimate and must not raise."""
        monkeypatch.setattr(transforms, "alb", None)
        assert AlbumentationsWrapper.from_config([]) == []

    @pytest.mark.parametrize("image_set", ["train", "val", "test", "val_speed"])
    def test_make_coco_transforms_raises_when_albumentations_missing(self, monkeypatch, image_set):
        monkeypatch.setattr(transforms, "alb", None)
        with pytest.raises(ImportError, match="albumentations is required"):
            coco.make_coco_transforms(image_set, resolution=640)


class TestBuildResizeGuard:
    """Even with albumentations present, a resize that builds nothing is fatal."""

    @pytest.mark.parametrize("image_set", ["train", "val", "test", "val_speed"])
    def test_raises_when_resize_builds_no_transforms(self, monkeypatch, image_set):
        monkeypatch.setattr(AlbumentationsWrapper, "from_config", staticmethod(lambda config: []))
        with pytest.raises(RuntimeError, match="built zero transforms"):
            coco.make_coco_transforms(image_set, resolution=640)

    @pytest.mark.parametrize("image_set", ["train", "val", "test", "val_speed"])
    def test_square_div_64_raises_when_resize_builds_no_transforms(self, monkeypatch, image_set):
        monkeypatch.setattr(AlbumentationsWrapper, "from_config", staticmethod(lambda config: []))
        with pytest.raises(RuntimeError, match="built zero transforms"):
            coco.make_coco_transforms_square_div_64(image_set, resolution=640)
