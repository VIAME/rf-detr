# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Keypoint head: per-query keypoint coordinate + visibility regression."""

from typing import Tuple

import torch
from torch import nn

from rfdetr.models.math import MLP


class KeypointHead(nn.Module):
    """Predicts a fixed number of keypoints per query.

    Each object query emits ``num_keypoints`` keypoints.  For every keypoint the head produces a 2-D coordinate delta
    (combined with the box reference point by the caller, mirroring the box regression head) and a visibility logit.
    The absent/present decision for a keypoint is left to the visibility logit, which lets the same head cover objects
    with a variable number of visible keypoints (e.g. 0, 1, or 2) without changing its output shape.

    The coordinate MLP's final layer is zero-initialised so that, before training, every keypoint starts at its box
    reference point rather than at an arbitrary location.

    Args:
        hidden_dim: Feature dimension coming from the transformer decoder.
        num_keypoints: Number of keypoints predicted per query.
        num_layers: Number of layers in each MLP branch.
    """

    def __init__(self, hidden_dim: int, num_keypoints: int, num_layers: int = 3) -> None:
        super().__init__()
        if num_keypoints < 1:
            raise ValueError(f"num_keypoints must be >= 1, got {num_keypoints}")
        self.num_keypoints = num_keypoints
        self.coord_embed = MLP(hidden_dim, hidden_dim, num_keypoints * 2, num_layers)
        self.vis_embed = MLP(hidden_dim, hidden_dim, num_keypoints, num_layers)
        nn.init.constant_(self.coord_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.coord_embed.layers[-1].bias.data, 0)

    def forward(self, hs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project decoder hidden states to keypoint coordinate deltas and visibility logits.

        Args:
            hs: Decoder output of shape ``(..., num_queries, hidden_dim)``.  The leading dimensions (decoder layers,
                batch) are preserved.

        Returns:
            A tuple ``(coord_delta, vis_logit)`` where ``coord_delta`` has shape ``(..., num_queries, num_keypoints,
            2)`` and ``vis_logit`` has shape ``(..., num_queries, num_keypoints, 1)``.  The coordinate deltas are raw
            (pre-sigmoid / pre-reparametrisation) — the caller combines them with the box reference point exactly as
            it does for box regression.
        """
        coord_delta = self.coord_embed(hs)
        coord_delta = coord_delta.reshape(*coord_delta.shape[:-1], self.num_keypoints, 2)
        vis_logit = self.vis_embed(hs)
        vis_logit = vis_logit.reshape(*vis_logit.shape[:-1], self.num_keypoints, 1)
        return coord_delta, vis_logit
