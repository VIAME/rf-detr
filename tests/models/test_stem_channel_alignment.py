# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for patch-embedding stem alignment when loading a checkpoint.

``adapt_input_channels`` tiles a stock 3-channel stem out to ``num_channels`` and runs
*after* ``load_pretrain_weights``. That ordering fails when the checkpoint is itself a
multi-channel model -- fine-tuning a 4-channel run from its own checkpoint -- because
the stem is still 3 wide when ``load_state_dict`` runs.
"""

from __future__ import annotations

import pytest
import torch

from rfdetr.models.weights import (
    _align_stem_to_checkpoint,
    _checkpoint_stem_channels,
    adapt_input_channels,
)

PROJ_KEY = "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"


class _PatchEmbeddings(torch.nn.Module):
    """Minimal stand-in exposing the attribute path adapt_input_channels walks."""

    def __init__(self, num_channels: int, embed_dim: int = 8, patch: int = 2) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.projection = torch.nn.Conv2d(num_channels, embed_dim, kernel_size=patch, stride=patch)


class _Embeddings(torch.nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.patch_embeddings = _PatchEmbeddings(num_channels)


class _InnerEncoder(torch.nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.embeddings = _Embeddings(num_channels)


class _Encoder(torch.nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.encoder = _InnerEncoder(num_channels)


class _Outer(torch.nn.Module):
    """backbone[0]: the real path is backbone[0].encoder.encoder.embeddings."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.encoder = _Encoder(num_channels)


class _FakeModel(torch.nn.Module):
    """Model whose backbone[0].encoder.encoder.embeddings.patch_embeddings resolves."""

    def __init__(self, num_channels: int = 3) -> None:
        super().__init__()
        self.backbone = torch.nn.ModuleList([_Outer(num_channels)])

    @property
    def stem(self) -> torch.nn.Conv2d:
        return self.backbone[0].encoder.encoder.embeddings.patch_embeddings.projection

    @property
    def stem_prefix(self) -> str:
        return "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection"


def _stem_channels(model: _FakeModel) -> int:
    return model.stem.weight.shape[1]


class TestCheckpointStemChannels:
    """Verify the checkpoint stem width is read off the state dict."""

    def test_reads_channel_count_from_projection_weight(self) -> None:
        state = {PROJ_KEY: torch.zeros(8, 4, 2, 2)}
        assert _checkpoint_stem_channels(state) == 4

    def test_returns_none_when_no_projection_present(self) -> None:
        assert _checkpoint_stem_channels({"some.other.weight": torch.zeros(3)}) is None

    def test_ignores_non_4d_tensors_on_a_matching_key(self) -> None:
        assert _checkpoint_stem_channels({PROJ_KEY: torch.zeros(8)}) is None


class TestAdaptInputChannelsIdempotence:
    """adapt_input_channels must not re-tile an already-correct stem."""

    def test_widens_a_three_channel_stem(self) -> None:
        model = _FakeModel(3)
        adapt_input_channels(model, 4)
        assert _stem_channels(model) == 4

    def test_second_call_leaves_weights_untouched(self) -> None:
        """Re-tiling would rescale by 3/num_channels and corrupt loaded weights."""
        model = _FakeModel(3)
        adapt_input_channels(model, 4)
        after_first = model.stem.weight.detach().clone()

        adapt_input_channels(model, 4)

        assert torch.equal(model.stem.weight.detach(), after_first)

    def test_three_channel_target_is_still_a_noop(self) -> None:
        model = _FakeModel(3)
        before = model.stem.weight.detach().clone()
        adapt_input_channels(model, 3)
        assert torch.equal(model.stem.weight.detach(), before)


class TestAlignStemToCheckpoint:
    """Verify the pre-load alignment covers each checkpoint/model combination."""

    def test_four_channel_checkpoint_widens_the_stem_before_load(self) -> None:
        """The regression: a 4-channel checkpoint into a freshly built 3-channel stem."""
        model = _FakeModel(3)
        state = {PROJ_KEY: torch.zeros(8, 4, 2, 2)}

        _align_stem_to_checkpoint(model, state, num_channels=4)

        assert _stem_channels(model) == 4

    def test_three_channel_checkpoint_leaves_the_stem_alone(self) -> None:
        """Stock COCO weights still load 3-into-3, then widen after."""
        model = _FakeModel(3)
        state = {PROJ_KEY: torch.zeros(8, 3, 2, 2)}

        _align_stem_to_checkpoint(model, state, num_channels=4)

        assert _stem_channels(model) == 3

    def test_checkpoint_without_a_stem_is_ignored(self) -> None:
        model = _FakeModel(3)
        _align_stem_to_checkpoint(model, {"head.weight": torch.zeros(2)}, num_channels=4)
        assert _stem_channels(model) == 3

    def test_mismatched_non_rgb_checkpoint_raises(self) -> None:
        """A 4-channel checkpoint into a 3-channel model cannot be reconciled."""
        model = _FakeModel(3)
        state = {PROJ_KEY: torch.zeros(8, 4, 2, 2)}

        with pytest.raises(ValueError, match="4-channel input stem"):
            _align_stem_to_checkpoint(model, state, num_channels=3)

    def test_aligned_stem_then_adapt_is_a_noop(self) -> None:
        """The full call order: align, load, then adapt must not rescale."""
        model = _FakeModel(3)
        state = {PROJ_KEY: torch.zeros(8, 4, 2, 2)}
        _align_stem_to_checkpoint(model, state, num_channels=4)
        model.stem.weight.data.fill_(1.0)  # stand in for loaded checkpoint weights

        adapt_input_channels(model, 4)

        assert torch.equal(model.stem.weight.detach(), torch.ones_like(model.stem.weight))

    def test_state_dict_load_succeeds_after_alignment(self) -> None:
        """End to end: the load_state_dict call that used to raise now succeeds."""
        model = _FakeModel(3)
        source = _FakeModel(4)
        state = {f"backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.{k}": v
                 for k, v in source.stem.state_dict().items()}

        _align_stem_to_checkpoint(model, state, num_channels=4)
        incompatible = model.load_state_dict(state, strict=False)

        assert not incompatible.unexpected_keys
        assert torch.equal(model.stem.weight.detach(), source.stem.weight.detach())


class TestAdaptInputChannelsEncoderChecks:
    """Channel adaptation rewrites DINOv2's patch embedding and must reject other encoders."""

    def test_non_dinov2_encoder_name_raises(self) -> None:
        model = _FakeModel(3)
        with pytest.raises(ValueError, match="num_channels=4 is supported for DINOv2 encoders only, not encoder='toy'"):
            adapt_input_channels(model, 4, encoder_name="toy")
        assert _stem_channels(model) == 3

    def test_encoder_without_patch_embedding_raises(self) -> None:
        """Without an encoder name the stem is checked structurally."""
        model = torch.nn.Module()
        outer = torch.nn.Module()
        outer.encoder = torch.nn.Linear(2, 2)
        model.backbone = torch.nn.ModuleList([outer])

        with pytest.raises(ValueError, match="supported for DINOv2 encoders only, not encoder='Linear'"):
            adapt_input_channels(model, 4)

    def test_three_channels_skips_the_encoder_check(self) -> None:
        adapt_input_channels(_FakeModel(3), 3, encoder_name="toy")

    def test_dinov2_encoder_name_adapts(self) -> None:
        model = _FakeModel(3)
        adapt_input_channels(model, 5, encoder_name="dinov2_windowed_small")
        assert _stem_channels(model) == 5
        assert model.backbone[0].encoder.encoder.embeddings.patch_embeddings.num_channels == 5


class TestFourChannelCheckpointRoundTrip:
    """A real 4-channel model reloads its own checkpoint through the inference context builder."""

    def test_four_channel_checkpoint_reloads_unscaled(self, tmp_path) -> None:
        from rfdetr.config import RFDETRNanoConfig
        from rfdetr.inference import _build_model_context

        source = _build_model_context(RFDETRNanoConfig(pretrain_weights=None, num_channels=4, device="cpu")).model
        key = "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"
        source_state = source.state_dict()
        assert source_state[key].shape[1] == 4
        checkpoint_path = tmp_path / "four_channel.pth"
        torch.save({"model": source_state}, checkpoint_path)

        reloaded = _build_model_context(
            RFDETRNanoConfig(pretrain_weights=str(checkpoint_path), num_channels=4, device="cpu")
        ).model

        assert torch.equal(reloaded.state_dict()[key], source_state[key])

    def test_four_channel_checkpoint_into_rgb_model_raises(self, tmp_path) -> None:
        from rfdetr.config import RFDETRNanoConfig
        from rfdetr.inference import _build_model_context

        source = _build_model_context(RFDETRNanoConfig(pretrain_weights=None, num_channels=4, device="cpu")).model
        checkpoint_path = tmp_path / "four_channel.pth"
        torch.save({"model": source.state_dict()}, checkpoint_path)

        with pytest.raises(ValueError, match="4-channel input stem"):
            _build_model_context(RFDETRNanoConfig(pretrain_weights=str(checkpoint_path), device="cpu"))
