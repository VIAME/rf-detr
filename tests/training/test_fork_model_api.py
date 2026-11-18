# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Non-square, video and multistep-schedule behaviour of the model build, public API and training modules."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rfdetr import RFDETRNano
from rfdetr.training.auto_batch import _make_synthetic_batch
from rfdetr.training.module_model import _build_managed_scheduler


@pytest.mark.parametrize(
    ("kwargs", "channels", "shape"),
    [
        pytest.param({"resolution": (320, 448)}, 3, (320, 448), id="non-square"),
        pytest.param({"resolution": 384, "temporal_frames": 3}, 9, (384, 384), id="video"),
        pytest.param({"resolution": (320, 448), "temporal_frames": 2, "num_channels": 4}, 8, (320, 448), id="both"),
    ],
)
def test_predict_accepts_non_square_and_stacked_frame_inputs(
    kwargs: dict[str, object], channels: int, shape: tuple[int, int]
) -> None:
    """predict() resizes to (height, width) and validates the stacked-frame channel count."""
    model = RFDETRNano(pretrain_weights=None, num_classes=1, device="cpu", **kwargs)

    assert len(model.means) == len(model.stds) == channels
    assert model.means[: model.model_config.num_channels] == model.means[-model.model_config.num_channels :]
    assert model._prediction_context().default_shape == shape
    assert model._runtime_info()["default_shape"] == shape

    image = (np.random.default_rng(0).random((90, 150, channels)) * 255).astype(np.uint8)
    detections = model.predict(image, threshold=0.0)
    assert detections.xyxy.shape[1] == 4

    with pytest.raises(ValueError, match="Invalid tensor image shape"):
        model.predict(torch.rand(channels + 1, 64, 64))


def test_multistep_schedule_drops_every_lr_drop_epochs() -> None:
    """The managed "multistep" preset decays 10x every lr_drop epochs, after warmup."""
    train_config = SimpleNamespace(lr_scheduler="multistep", lr_scheduler_kwargs={"lr_drop": 2})
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)

    scheduler = _build_managed_scheduler(optimizer, train_config, total_steps=100, steps_per_epoch=10, warmup_steps=5)
    factors = [scheduler.lr_lambdas[0](step) for step in (0, 4, 19, 20, 39, 40, 60)]

    assert factors == pytest.approx([0.0, 0.8, 1.0, 0.1, 0.1, 0.01, 0.001])


def test_synthetic_probe_batch_follows_height_width_resolution() -> None:
    """The auto-batch probe builds images, padding masks and targets at (height, width)."""
    samples, targets = _make_synthetic_batch(
        2, (64, 96), torch.device("cpu"), num_classes=3, segmentation_head=True, max_targets_per_image=4, num_channels=6
    )

    assert samples.tensors.shape == (2, 6, 64, 96)
    assert samples.mask.shape == (2, 64, 96)
    assert targets[0]["masks"].shape[-2:] == (64, 96)
    assert targets[0]["size"].tolist() == [64, 96]
