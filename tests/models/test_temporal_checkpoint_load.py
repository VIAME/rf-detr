# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""A single-frame checkpoint must load into a video (``temporal_frames > 1``) model and keep its outputs."""

from __future__ import annotations

import pytest
import torch

from rfdetr import RFDETRNano
from rfdetr.models.weights import load_pretrain_weights

RESOLUTION = 384
TEMPORAL_FRAMES = 3


@pytest.fixture(scope="module")
def single_and_video_models() -> tuple[RFDETRNano, RFDETRNano]:
    """Build a single-frame and a 3-frame nano detector on CPU without pretrained weights."""
    torch.manual_seed(0)
    single = RFDETRNano(pretrain_weights=None, num_classes=1, device="cpu", resolution=RESOLUTION)
    video = RFDETRNano(
        pretrain_weights=None, num_classes=1, device="cpu", resolution=RESOLUTION, temporal_frames=TEMPORAL_FRAMES
    )
    return single, video


def _assert_same_outputs(single_model: torch.nn.Module, video_model: torch.nn.Module) -> None:
    """Assert the video model reproduces the single-frame model on the reference frame."""
    single_model.eval()
    video_model.eval()
    torch.manual_seed(1)
    frames = torch.randn(1, 3 * TEMPORAL_FRAMES, RESOLUTION, RESOLUTION)
    with torch.no_grad():
        video_out = video_model(frames)
        single_out = single_model(frames[:, :3])
    for key in ("pred_logits", "pred_boxes"):
        assert torch.allclose(video_out[key], single_out[key], atol=1e-5), f"{key} differs from the single-frame model"


def test_single_frame_state_dict_loads_with_only_fusion_keys_missing(
    single_and_video_models: tuple[RFDETRNano, RFDETRNano],
) -> None:
    """strict=False load tolerates the absent fusion keys, and zero-initialised fusion keeps the outputs."""
    single, video = single_and_video_models
    single_model, video_model = single.model.model, video.model.model

    incompatible = video_model.load_state_dict(single_model.state_dict(), strict=False)

    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys, "the video model should own temporal fusion parameters"
    assert all(".temporal_fusion." in key for key in incompatible.missing_keys)
    _assert_same_outputs(single_model, video_model)


def test_load_pretrain_weights_accepts_single_frame_checkpoint(
    single_and_video_models: tuple[RFDETRNano, RFDETRNano],
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``load_pretrain_weights`` loads a checkpoint without fusion keys and does not report a partial load."""
    single, video = single_and_video_models
    single_model, video_model = single.model.model, video.model.model
    checkpoint_path = tmp_path / "single_frame.pth"
    torch.save({"model": single_model.state_dict()}, checkpoint_path)
    model_config = video.model_config.model_copy(update={"pretrain_weights": str(checkpoint_path)})

    warnings: list[str] = []
    monkeypatch.setattr(
        "rfdetr.models.weights.logger.warning", lambda msg, *args, **kwargs: warnings.append(str(msg) % args)
    )
    load_pretrain_weights(video_model, model_config)

    assert not any("loaded only partially" in message for message in warnings), warnings
    _assert_same_outputs(single_model, video_model)
