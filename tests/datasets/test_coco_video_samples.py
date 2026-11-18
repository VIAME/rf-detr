# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Video (``temporal_files``) and four-channel samples through ``CocoDetection`` and its transform pipelines.

A video sample is the reference frame plus its context frames stacked on the channel axis. PIL cannot hold more than
four channels, so the stack travels through the transforms as an HWC array and must come out as one CHW float tensor
with the target's boxes left as an ``(N, 4)`` tensor.
"""

import json
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from rfdetr.datasets.coco import (
    CocoDetection,
    build_roboflow_from_coco,
    make_coco_transforms,
    make_coco_transforms_square_div_64,
)

_WIDTH, _HEIGHT = 96, 64
_NUM_FRAMES = 3
# One box in the left half of the frame, COCO [x, y, w, h].
_BBOX = [8, 16, 24, 32]


@contextmanager
def _ignore_backend_warning() -> Iterator[None]:
    """Silence the one-off notice the default (torchvision) train pipeline emits."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def _frame(brightness: int) -> np.ndarray:
    """Return an RGB frame whose left half is bright and right half is black."""
    frame = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    frame[:, : _WIDTH // 2] = brightness
    return frame


def _write_video_dataset(root: Path, *, absolute_paths: bool = False, extension: str = "png") -> Path:
    """Write a Roboflow-layout train split of two video samples and return the dataset root."""
    split = root / "train"
    split.mkdir(parents=True)
    images = []
    annotations = []
    for image_id in range(2):
        names = [f"clip{image_id}_frame{index}.{extension}" for index in range(_NUM_FRAMES)]
        for index, name in enumerate(names):
            Image.fromarray(_frame(brightness=250 - 60 * index)).save(split / name)
        paths = [str(split / name) if absolute_paths else name for name in names]
        images.append(
            {
                "id": image_id,
                "file_name": paths[0],
                "temporal_files": paths[1:],
                "width": _WIDTH,
                "height": _HEIGHT,
            }
        )
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": _BBOX,
                "area": _BBOX[2] * _BBOX[3],
                "iscrowd": 0,
            }
        )
    coco = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "fish"}]}
    (split / "_annotations.coco.json").write_text(json.dumps(coco))
    return root


def _left_is_brighter(image: torch.Tensor) -> list[bool]:
    """Return, per stacked frame, whether its left half is brighter than its right half."""
    frames = image.reshape(_NUM_FRAMES, 3, *image.shape[-2:])
    half = image.shape[-1] // 2
    return [bool(frame[..., :half].mean() > frame[..., half:].mean()) for frame in frames]


def _assert_video_sample(image: Any, target: dict[str, Any]) -> None:
    """Check the image is a 9-channel float tensor and the target's boxes stayed an ``(N, 4)`` tensor."""
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.float32
    assert image.ndim == 3
    assert image.shape[0] == 3 * _NUM_FRAMES
    assert target["boxes"].shape == (1, 4)
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].shape == (1,)
    # normalized cxcywh
    assert target["boxes"].min() >= 0.0
    assert target["boxes"].max() <= 1.0


def _dataset(root: Path, transforms: Any) -> CocoDetection:
    return CocoDetection(root / "train", root / "train" / "_annotations.coco.json", transforms=transforms)


class TestVideoSampleLoading:
    """``temporal_files`` frames are stacked behind the reference frame on the channel axis."""

    def test_untransformed_sample_is_hwc_stack_in_file_order(self, tmp_path: Path) -> None:
        dataset = _dataset(_write_video_dataset(tmp_path), transforms=None)
        image, target = dataset[0]

        assert isinstance(image, np.ndarray)
        assert image.shape == (_HEIGHT, _WIDTH, 3 * _NUM_FRAMES)
        # reference first, then the context frames in listed order
        assert [int(image[0, 0, 3 * index]) for index in range(_NUM_FRAMES)] == [250, 190, 130]
        assert target["orig_size"].tolist() == [_HEIGHT, _WIDTH]
        assert target["boxes"].tolist() == [[8.0, 16.0, 32.0, 48.0]]

    def test_absolute_frame_paths_are_supported(self, tmp_path: Path) -> None:
        dataset = _dataset(_write_video_dataset(tmp_path, absolute_paths=True), transforms=None)
        image, _ = dataset[1]
        assert image.shape == (_HEIGHT, _WIDTH, 3 * _NUM_FRAMES)

    def test_context_frame_of_other_size_falls_back_to_reference(self, tmp_path: Path) -> None:
        root = _write_video_dataset(tmp_path)
        Image.fromarray(np.full((10, 10, 3), 7, dtype=np.uint8)).save(root / "train" / "clip0_frame1.png")
        image, _ = _dataset(root, transforms=None)[0]

        assert image.shape == (_HEIGHT, _WIDTH, 3 * _NUM_FRAMES)
        assert np.array_equal(image[..., 3:6], image[..., 0:3])

    def test_jpeg_frames_are_stacked_with_draft_decode(self, tmp_path: Path) -> None:
        """A drafted (reduced) JPEG decode keeps all frames the same size and scales the annotations."""
        root = _write_video_dataset(tmp_path, extension="jpg")
        split = root / "train"
        dataset = CocoDetection(split, split / "_annotations.coco.json", transforms=None, draft_size=32)
        image, target = dataset[0]

        assert image.shape == (_HEIGHT // 2, _WIDTH // 2, 3 * _NUM_FRAMES)
        assert target["boxes"].tolist() == [[4.0, 8.0, 16.0, 24.0]]


class TestVideoSampleTransforms:
    """Both transform backends turn the frame stack into a 9-channel tensor without touching the boxes."""

    @pytest.mark.parametrize("image_set", ["train", "val", "val_speed"])
    @pytest.mark.parametrize(
        "factory",
        [
            pytest.param(make_coco_transforms, id="standard"),
            pytest.param(make_coco_transforms_square_div_64, id="square_div_64"),
        ],
    )
    @pytest.mark.parametrize(
        "aug_config",
        [
            pytest.param(None, id="torchvision"),
            pytest.param({"HorizontalFlip": {"p": 0.5}}, id="albumentations"),
        ],
    )
    def test_sample_is_nine_channel_float_tensor(
        self, tmp_path: Path, image_set: str, factory: Any, aug_config: Any
    ) -> None:
        with _ignore_backend_warning():
            # no random crop, so the single box always survives
            transforms = factory(image_set, 64, aug_config=aug_config, scale_jitter=False)
        image, target = _dataset(_write_video_dataset(tmp_path), transforms)[0]

        _assert_video_sample(image, target)
        assert min(image.shape[-2:]) == 64

    def test_albumentations_flip_is_applied_to_every_frame(self, tmp_path: Path) -> None:
        transforms = make_coco_transforms_square_div_64(
            "train", 64, aug_config={"HorizontalFlip": {"p": 1.0}}, scale_jitter=False
        )
        image, target = _dataset(_write_video_dataset(tmp_path), transforms)[0]

        _assert_video_sample(image, target)
        assert _left_is_brighter(image) == [False] * _NUM_FRAMES
        # the box started in the left half, so its centre is now in the right half
        assert target["boxes"][0, 0] > 0.5

    def test_torchvision_flip_is_consistent_across_frames_and_boxes(self, tmp_path: Path) -> None:
        with _ignore_backend_warning():
            transforms = make_coco_transforms_square_div_64("train", 64, scale_jitter=False)
        dataset = _dataset(_write_video_dataset(tmp_path), transforms)

        orientations = set()
        for seed in range(12):
            torch.manual_seed(seed)
            image, target = dataset[0]
            _assert_video_sample(image, target)
            left_bright = _left_is_brighter(image)
            # every frame shares one orientation, and the box moved with the pixels
            assert len(set(left_bright)) == 1
            assert bool(target["boxes"][0, 0] < 0.5) == left_bright[0]
            orientations.add(left_bright[0])
        # both the flipped and the unflipped case were exercised
        assert orientations == {True, False}

    def test_builder_produces_video_train_sample(self, tmp_path: Path) -> None:
        args = SimpleNamespace(
            dataset_dir=str(_write_video_dataset(tmp_path)),
            square_resize_div_64=True,
            segmentation_head=False,
            multi_scale=False,
            expanded_scales=False,
            patch_size=16,
            num_windows=1,
            aug_config={"HorizontalFlip": {"p": 0.5}},
            augmentation_backend="cpu",
            scale_jitter=False,
        )
        dataset = build_roboflow_from_coco("train", args, resolution=(64, 96))
        image, target = dataset[0]

        _assert_video_sample(image, target)
        assert image.shape == (3 * _NUM_FRAMES, 64, 96)


class TestFourChannelSample:
    """An RGBA file (RGB plus an appended motion channel) keeps its fourth channel."""

    @pytest.mark.parametrize(
        "aug_config",
        [
            pytest.param(None, id="torchvision"),
            pytest.param({"HorizontalFlip": {"p": 0.5}}, id="albumentations"),
        ],
    )
    def test_rgba_image_yields_four_channel_tensor(self, tmp_path: Path, aug_config: Any) -> None:
        root = _write_video_dataset(tmp_path)
        rgba = np.concatenate([_frame(200), np.full((_HEIGHT, _WIDTH, 1), 40, dtype=np.uint8)], axis=2)
        Image.fromarray(rgba, mode="RGBA").save(root / "train" / "clip0_frame0.png")
        coco = json.loads((root / "train" / "_annotations.coco.json").read_text())
        for entry in coco["images"]:
            del entry["temporal_files"]
        (root / "train" / "_annotations.coco.json").write_text(json.dumps(coco))

        raw_image, _ = _dataset(root, transforms=None)[0]
        assert raw_image.mode == "RGBA"

        with _ignore_backend_warning():
            transforms = make_coco_transforms_square_div_64("train", 64, aug_config=aug_config, scale_jitter=False)
        image, target = _dataset(root, transforms)[0]
        assert image.shape == (4, 64, 64)
        assert image.dtype == torch.float32
        assert target["boxes"].shape == (1, 4)

        # the three-channel neighbour is still loaded as RGB
        assert _dataset(root, transforms=None)[1][0].mode == "RGB"
