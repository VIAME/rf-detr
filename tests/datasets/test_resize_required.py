# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""The resize stage must never be silently dropped.

The Albumentations pipeline (used for a non-empty ``aug_config``) builds its resize through the same config path as the
optional augmentations, which logs-and-skips whatever it cannot construct. Losing the resize is not a degraded
augmentation policy: images reach the model at their native size while boxes are normalised against that size, so the
model learns a geometry that inference never reproduces. These tests pin the failure to a loud one.
"""

import pytest

from rfdetr.datasets import coco, transforms
from rfdetr.datasets.transforms import AlbumentationsWrapper

_AUG_CONFIG = {"HorizontalFlip": {"p": 0.5}}
_FACTORIES = [
    pytest.param(coco.make_coco_transforms, id="standard"),
    pytest.param(coco.make_coco_transforms_square_div_64, id="square_div_64"),
]


class TestRequireAlbumentations:
    """A missing albumentations install raises rather than yielding zero transforms."""

    def test_from_config_raises_when_albumentations_missing(self, monkeypatch):
        monkeypatch.setattr(transforms, "alb", None)
        with pytest.raises(ImportError, match="require the optional augmentation extra"):
            AlbumentationsWrapper.from_config([{"Resize": {"height": 640, "width": 640}}])

    def test_empty_config_still_returns_empty(self, monkeypatch):
        """An explicitly empty augmentation config is legitimate and must not raise."""
        monkeypatch.setattr(transforms, "alb", None)
        assert AlbumentationsWrapper.from_config([]) == []

    @pytest.mark.parametrize("factory", _FACTORIES)
    def test_albumentations_pipeline_raises_when_albumentations_missing(self, monkeypatch, factory):
        monkeypatch.setattr(transforms, "alb", None)
        with pytest.raises(ImportError, match="require the optional augmentation extra"):
            factory("train", resolution=640, aug_config=_AUG_CONFIG)

    @pytest.mark.parametrize("image_set", ["val", "test", "val_speed"])
    def test_torchvision_pipeline_does_not_need_albumentations(self, monkeypatch, image_set):
        """The default pipeline resizes with torchvision, so it builds without albumentations."""
        monkeypatch.setattr(transforms, "alb", None)
        assert coco.make_coco_transforms(image_set, resolution=640) is not None


class TestBuildResizeGuard:
    """Even with albumentations present, a resize that builds nothing is fatal."""

    @pytest.mark.parametrize("factory", _FACTORIES)
    def test_raises_when_resize_builds_no_transforms(self, monkeypatch, factory):
        monkeypatch.setattr(AlbumentationsWrapper, "from_config", staticmethod(lambda config, **kwargs: []))
        with pytest.raises(RuntimeError, match="built zero transforms"):
            factory("train", resolution=640, aug_config=_AUG_CONFIG)

    @pytest.mark.parametrize("image_set", ["train", "val", "test", "val_speed"])
    def test_build_resize_raises_on_empty_stage(self, monkeypatch, image_set):
        monkeypatch.setattr(AlbumentationsWrapper, "from_config", staticmethod(lambda config, **kwargs: []))
        with pytest.raises(RuntimeError, match="built zero transforms"):
            coco._build_resize([{"Resize": {"height": 640, "width": 640}}], image_set)

    def test_build_resize_raises_on_unbuildable_transform(self):
        """A resize transform that fails to build raises instead of being logged and skipped."""
        with pytest.raises(RuntimeError, match="Failed to build required transform"):
            coco._build_resize([{"NoSuchResize": {"height": 640, "width": 640}}], "train")
