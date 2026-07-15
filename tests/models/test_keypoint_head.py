# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
import torch

from rfdetr.models.criterion import SetCriterion
from rfdetr.models.heads.keypoint import KeypointHead
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.models.postprocess import PostProcess


@pytest.fixture()
def criterion() -> SetCriterion:
    """A criterion instance usable purely for its ``loss_keypoints`` method."""
    return SetCriterion(
        num_classes=2,
        matcher=HungarianMatcher(),
        weight_dict={},
        focal_alpha=0.25,
        losses=["keypoints"],
    )


class TestKeypointHead:
    """Shape and initialisation behaviour of :class:`KeypointHead`."""

    def test_output_shapes(self) -> None:
        head = KeypointHead(hidden_dim=32, num_keypoints=2)
        hs = torch.randn(3, 2, 5, 32)  # [layers, batch, queries, channels]
        coord_delta, vis_logit = head(hs)
        assert coord_delta.shape == (3, 2, 5, 2, 2)
        assert vis_logit.shape == (3, 2, 5, 2, 1)

    def test_coord_head_zero_initialised(self) -> None:
        head = KeypointHead(hidden_dim=16, num_keypoints=2)
        coord_delta, _ = head(torch.randn(1, 1, 4, 16))
        # The final coord layer is zero-initialised so keypoints start at the box reference point.
        assert torch.allclose(coord_delta, torch.zeros_like(coord_delta))

    def test_rejects_zero_keypoints(self) -> None:
        with pytest.raises(ValueError):
            KeypointHead(hidden_dim=16, num_keypoints=0)


class TestLossKeypoints:
    """Behaviour of :meth:`SetCriterion.loss_keypoints`."""

    def test_supervises_only_visible_keypoints(self, criterion: SetCriterion) -> None:
        pred = torch.zeros(1, 2, 2, 3)
        pred[0, 0, 0, :2] = torch.tensor([0.4, 0.4])  # off from the visible target
        outputs = {"pred_keypoints": pred}
        target_kp = torch.tensor(
            [[[0.5, 0.5, 2.0], [0.0, 0.0, 0.0]]],  # head visible, tail absent
            dtype=torch.float32,
        )
        targets = [{"keypoints": target_kp, "labels": torch.tensor([0])}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        losses = criterion.loss_keypoints(outputs, targets, indices, num_boxes=1.0)

        # Only the visible head keypoint contributes: |0.4-0.5| + |0.4-0.5| = 0.2 over one visible point.
        assert losses["loss_keypoint"].item() == pytest.approx(0.2, abs=1e-5)
        assert torch.isfinite(losses["loss_keypoint_vis"])

    def test_empty_match_returns_zero(self, criterion: SetCriterion) -> None:
        outputs = {"pred_keypoints": torch.rand(1, 2, 2, 3)}
        targets = [{"keypoints": torch.zeros(0, 2, 3)}]
        indices = [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))]
        losses = criterion.loss_keypoints(outputs, targets, indices, num_boxes=1.0)
        assert losses["loss_keypoint"].item() == 0.0
        assert losses["loss_keypoint_vis"].item() == 0.0

    def test_skips_when_no_predicted_keypoints(self, criterion: SetCriterion) -> None:
        # Two-stage encoder outputs carry no keypoints; the loss must silently skip them.
        assert criterion.loss_keypoints({}, [{"keypoints": torch.zeros(1, 2, 3)}], [], num_boxes=1.0) == {}


class TestKeypointMatcherCost:
    """The keypoint term only participates when enabled and present in the targets."""

    def test_keypoint_cost_ignored_without_targets(self) -> None:
        matcher = HungarianMatcher(cost_keypoint=4.0)
        outputs = {
            "pred_logits": torch.randn(1, 5, 3),
            "pred_boxes": torch.rand(1, 5, 4),
            "pred_keypoints": torch.rand(1, 5, 2, 3),
        }
        targets = [{"labels": torch.tensor([0]), "boxes": torch.rand(1, 4)}]  # no "keypoints" key
        indices = matcher(outputs, targets, group_detr=1)
        assert len(indices) == 1 and len(indices[0][0]) == 1

    def test_keypoint_cost_pulls_matching(self) -> None:
        # Two queries; query 1's keypoints coincide with the single target, query 0's do not.
        matcher = HungarianMatcher(cost_class=0.0, cost_bbox=0.0, cost_giou=0.0, cost_keypoint=1.0)
        target_kp = torch.tensor([[[0.5, 0.5, 2.0], [0.6, 0.6, 2.0]]], dtype=torch.float32)
        pred_kp = torch.zeros(1, 2, 2, 3)
        pred_kp[0, 0, :, :2] = 0.1  # far
        pred_kp[0, 1, :, :2] = torch.tensor([[0.5, 0.5], [0.6, 0.6]])  # exact
        outputs = {
            "pred_logits": torch.zeros(1, 2, 3),
            "pred_boxes": torch.rand(1, 2, 4),
            "pred_keypoints": pred_kp,
        }
        targets = [{"labels": torch.tensor([0]), "boxes": torch.rand(1, 4), "keypoints": target_kp}]
        (src, tgt), = matcher(outputs, targets, group_detr=1)
        assert src.tolist() == [1]
        assert tgt.tolist() == [0]


class TestPostProcessKeypoints:
    """PostProcess attaches per-detection keypoints in absolute coordinates."""

    def test_keypoints_gathered_and_scaled(self) -> None:
        pp = PostProcess(num_select=3)
        outputs = {
            "pred_logits": torch.randn(1, 6, 3),
            "pred_boxes": torch.rand(1, 6, 4),
            "pred_keypoints": torch.rand(1, 6, 2, 3),
        }
        results = pp(outputs, torch.tensor([[80, 100]]))
        assert results[0]["keypoints"].shape == (3, 2, 3)
        # x, y scaled into image extent (100 wide, 80 tall); visibility in [0, 1] after sigmoid.
        kp = results[0]["keypoints"]
        assert (kp[..., 0] <= 100.0).all() and (kp[..., 1] <= 80.0).all()
        assert (kp[..., 2] >= 0.0).all() and (kp[..., 2] <= 1.0).all()
