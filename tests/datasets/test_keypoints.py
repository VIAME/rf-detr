# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch
from PIL import Image

from rfdetr.datasets.coco import ConvertCoco, _require_cpu_backend_for_keypoints
from rfdetr.datasets.transforms import AlbumentationsWrapper, Normalize

alb = pytest.importorskip("albumentations")


class TestConvertCocoKeypoints:
    """COCO keypoint parsing into ``[N, K, 3]`` targets."""

    def _target(self) -> dict:
        return {
            "image_id": 1,
            "annotations": [
                {
                    "bbox": [10, 10, 20, 20],
                    "category_id": 0,
                    "area": 400,
                    "iscrowd": 0,
                    "keypoints": [15, 15, 2, 25, 25, 1],
                },
                {
                    "bbox": [40, 40, 10, 10],
                    "category_id": 0,
                    "area": 100,
                    "iscrowd": 0,
                    "keypoints": [45, 45, 2, 0, 0, 0],
                },
            ],
        }

    def test_parses_keypoints(self) -> None:
        convert = ConvertCoco(include_keypoints=True, num_keypoints=2)
        _, target = convert(Image.new("RGB", (100, 80)), self._target())
        assert target["keypoints"].shape == (2, 2, 3)
        assert target["keypoints"][0, 0].tolist() == [15.0, 15.0, 2.0]
        assert target["keypoints"][1, 1, 2].item() == 0.0  # absent tail stays absent

    def test_absent_when_not_requested(self) -> None:
        convert = ConvertCoco(include_keypoints=False)
        _, target = convert(Image.new("RGB", (100, 80)), self._target())
        assert "keypoints" not in target

    def test_pads_missing_keypoints_field(self) -> None:
        convert = ConvertCoco(include_keypoints=True, num_keypoints=2)
        target = {
            "image_id": 1,
            "annotations": [{"bbox": [10, 10, 20, 20], "category_id": 0, "area": 400, "iscrowd": 0}],
        }
        _, out = convert(Image.new("RGB", (100, 80)), target)
        assert out["keypoints"].shape == (1, 2, 3)
        assert out["keypoints"].abs().sum().item() == 0.0  # all zero / absent


class TestKeypointAugmentation:
    """Geometric augmentation keeps keypoints aligned and updates visibility."""

    def _target(self) -> dict:
        return {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
            "labels": torch.tensor([0]),
            "keypoints": torch.tensor([[[15.0, 20.0, 2.0], [25.0, 20.0, 1.0]]]),
            "area": torch.tensor([400.0]),
            "size": torch.tensor([80, 100]),
        }

    def test_horizontal_flip_moves_keypoints(self) -> None:
        wrapper = AlbumentationsWrapper(alb.HorizontalFlip(p=1.0))
        _, out = wrapper(Image.new("RGB", (100, 80)), self._target())
        kp = out["keypoints"]
        assert kp.shape == (1, 2, 3)
        # x mirrors about the 100px width; visibility is preserved.
        assert kp[0, 0, 0].item() == pytest.approx(84.0, abs=1.5)
        assert kp[0, 0, 2].item() == 2.0

    def test_crop_marks_out_of_frame_keypoint_absent(self) -> None:
        wrapper = AlbumentationsWrapper(alb.Crop(x_min=0, y_min=0, x_max=20, y_max=80, p=1.0))
        target = self._target()
        target["keypoints"] = torch.tensor([[[5.0, 20.0, 2.0], [25.0, 20.0, 2.0]]])  # tail outside x<20
        _, out = wrapper(Image.new("RGB", (100, 80)), target)
        kp = out["keypoints"]
        assert kp[0, 0, 2].item() == 2.0  # in-frame head stays visible
        assert kp[0, 1, 2].item() == 0.0  # out-of-frame tail marked absent

    def test_normalize_scales_keypoints(self) -> None:
        norm = Normalize()
        image = torch.rand(3, 80, 100)
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
            "keypoints": torch.tensor([[[50.0, 40.0, 2.0], [0.0, 0.0, 0.0]]]),
        }
        _, out = norm(image, target)
        assert out["keypoints"][0, 0, 0].item() == pytest.approx(0.5)
        assert out["keypoints"][0, 0, 1].item() == pytest.approx(0.5)


class TestKeypointBackendGuard:
    """Keypoint training must use the CPU augmentation backend."""

    def test_rejects_gpu_backend(self) -> None:
        with pytest.raises(ValueError, match="augmentation_backend='cpu'"):
            _require_cpu_backend_for_keypoints(include_keypoints=True, gpu_postprocess=True)

    def test_allows_cpu_backend(self) -> None:
        _require_cpu_backend_for_keypoints(include_keypoints=True, gpu_postprocess=False)
        _require_cpu_backend_for_keypoints(include_keypoints=False, gpu_postprocess=True)
