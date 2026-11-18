# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Feature-level fusion of neighbouring video frames into a reference frame."""

from typing import List

import torch
from torch import nn


class _FusionBlock(nn.Module):
    """Residual fusion of ``num_frames`` feature maps into frame 0's map.

    The last conv is zero-initialised, so a freshly built block returns the
    reference features unchanged and a single-frame checkpoint keeps its outputs.
    """

    def __init__(self, channels: int, num_frames: int) -> None:
        super().__init__()
        # Reference, each context frame, and each context-minus-reference difference.
        in_channels = channels * (2 * num_frames - 1)
        self.reduce = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(32 if channels % 32 == 0 else 1, channels)
        self.act = nn.GELU()
        self.mix = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, T, C, H, W), frame 0 is the reference.
        ref = feats[:, 0]
        ctx = feats[:, 1:]
        diff = ctx - ref.unsqueeze(1)
        stacked = torch.cat([ref, ctx.flatten(1, 2), diff.flatten(1, 2)], dim=1)
        return ref + self.mix(self.act(self.norm(self.reduce(stacked))))


class TemporalFusion(nn.Module):
    """One :class:`_FusionBlock` per projector level.

    Args:
        channels: Channel count of every projector output level.
        num_frames: Frames per sample, reference first.
        num_levels: Number of projector output levels.
    """

    def __init__(self, channels: int, num_frames: int, num_levels: int) -> None:
        super().__init__()
        if num_frames < 2:
            raise ValueError("TemporalFusion needs at least 2 frames per sample")
        self.num_frames = num_frames
        self.blocks = nn.ModuleList([_FusionBlock(channels, num_frames) for _ in range(num_levels)])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Fuse a list of ``(B, T, C, H, W)`` maps into ``(B, C, H, W)`` maps."""
        return [block(feat) for block, feat in zip(self.blocks, feats)]
