# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Augmentation presets and default configuration for RF-DETR training.

Import a preset and pass it as ``aug_config`` to your training call:

```python
from rfdetr.datasets.aug_config import AUG_CONSERVATIVE, AUG_AGGRESSIVE, AUG_AERIAL, AUG_INDUSTRIAL

model.train(dataset_dir="...", aug_config=AUG_CONSERVATIVE) model.train(dataset_dir="...", aug_config=AUG_AGGRESSIVE)

# Disable all augmentations
model.train(dataset_dir="...", aug_config={})

# Fully custom
model.train(dataset_dir="...", aug_config={"HorizontalFlip": {"p": 0.5}})
```

## Available presets

| Preset         | Best for                                         |
| -------------- | ------------------------------------------------ |
| ``AUG_CONSERVATIVE``  | Small datasets (under 500 images)             |
| ``AUG_AGGRESSIVE``    | Large datasets (2000+ images)                 |
| ``AUG_AERIAL``        | Satellite / overhead imagery                  |
| ``AUG_INDUSTRIAL``    | Manufacturing / inspection data               |

## Transform Categories

**Geometric transforms** (automatically transform bounding boxes):
- Flips: HorizontalFlip, VerticalFlip
- Rotations: Rotate, Affine, ShiftScaleRotate
- Crops: RandomCrop, CenterCrop, RandomResizedCrop
- Perspective: Perspective, ElasticTransform, GridDistortion

**Pixel-level transforms** (preserve bounding boxes):
- Color: ColorJitter, HueSaturationValue, RandomBrightnessContrast
- Blur/Noise: GaussianBlur, GaussNoise, Blur
- Enhancement: CLAHE, Sharpen, Equalize

## Best Practices

1. **Start conservative**: Use moderate probabilities (p=0.3-0.5) and small parameter ranges
2. **Geometric caution**: Extreme rotations (>45°) or crops may remove too many boxes
3. **Performance**: Fewer transforms = faster training; prioritize transforms that match your domain
4. **Validation**: Monitor validation mAP - excessive augmentation can hurt performance
5. **Domain-specific**: Enable augmentations that reflect real-world variations in your data

## Adding Custom Transforms

For geometric transforms not in GEOMETRIC_TRANSFORMS set, add them in transforms.py:

```python
GEOMETRIC_TRANSFORMS = {
    ...
    "YourCustomTransform",  # Add here
}
```

## Kornia GPU Backend

When ``augmentation_backend="auto"`` or ``"gpu"`` is set in ``TrainConfig``, augmentations run on the GPU via Kornia
instead of Albumentations.

**Supported transforms** (all presets):

| Preset key | Kornia equivalent | Notes |
|---|---|---|
| ``HorizontalFlip`` | ``K.RandomHorizontalFlip`` | Direct |
| ``VerticalFlip`` | ``K.RandomVerticalFlip`` | Direct |
| ``Rotate`` | ``K.RandomRotation`` | ``limit`` may be scalar or tuple |
| ``Affine`` | ``K.RandomAffine`` | ``translate_percent`` treated as fraction |
| ``ColorJitter`` | ``K.ColorJiggle`` | Same multiplicative semantics |
| ``RandomBrightnessContrast`` | ``K.ColorJiggle`` | ``brightness_limit`` / ``contrast_limit`` direct |
| ``GaussianBlur`` | ``K.RandomGaussianBlur`` | ``blur_limit`` rounded up to odd; ``sigma=(0.1, 2.0)`` |
| ``GaussNoise`` | ``K.RandomGaussianNoise`` | Upper bound of ``std_range`` used as fixed std |

**Phase 1 limitation**: Segmentation models (``segmentation_head=True``) skip GPU augmentation; CPU Albumentations are
used instead. Mask support is planned for Phase 2.
"""

# ---------------------------------------------------------------------------
# Default configuration (backward-compatible baseline)
# ---------------------------------------------------------------------------

AUG_CONFIG = {
    "HorizontalFlip": {"p": 0.5},
    # "VerticalFlip": {"p": 0.5},
    # "Rotate": {"limit": 15, "p": 0.5},  # Better keep small angles
}

# ---------------------------------------------------------------------------
# Named presets — import and pass directly as aug_config=<preset>
# ---------------------------------------------------------------------------

#: Minimal augmentations — safe for small datasets (under 500 images).
AUG_CONSERVATIVE = {
    "HorizontalFlip": {"p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.1,
        "contrast_limit": 0.1,
        "p": 0.3,
    },
}

#: Aggressive augmentations — for larger datasets (2000+ images).
AUG_AGGRESSIVE = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "Rotate": {"limit": 45, "p": 0.5},
    "Affine": {
        "scale": (0.8, 1.2),
        "translate_percent": (-0.1, 0.1),
        "rotate": (-15, 15),
        "shear": (-5, 5),
        "p": 0.5,
    },
    "ColorJitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.2,
        "hue": 0.1,
        "p": 0.5,
    },
}

#: Optimised for aerial / satellite imagery (overhead views, 90° rotations).
AUG_AERIAL = {
    "HorizontalFlip": {"p": 0.5},
    "VerticalFlip": {"p": 0.5},
    "Rotate": {"limit": (90, 90), "p": 0.5},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.15,
        "contrast_limit": 0.15,
        "p": 0.4,
    },
}

#: Optimised for industrial / manufacturing data (lighting & sensor noise).
AUG_INDUSTRIAL = {
    "HorizontalFlip": {"p": 0.3},
    "RandomBrightnessContrast": {
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "p": 0.5,
    },
    "GaussianBlur": {"blur_limit": 3, "p": 0.3},
    "GaussNoise": {"std_range": (0.01, 0.05), "p": 0.3},
}

# ---------------------------------------------------------------------------
# Motion-infused imagery
#
# These are for models whose input carries motion channels alongside (or instead
# of) appearance channels. Two rules shape all three:
#
#   1. No VerticalFlip. Every preset above flips vertically, which is fine for
#      overhead imagery but not for a scene with a real up direction -- it
#      manufactures upside-down subjects and a ceiling made of seabed, which the
#      model never sees at inference.
#
#   2. Photometric augmentation is confined to the appearance channels with
#      ChannelSubset. Brightness and contrast are meaningless on a motion channel
#      -- an offset lifts stationary background off true zero, inventing motion --
#      and saturation/hue are worse still, mixing unrelated motion operators
#      together as if they were colour. Geometric augmentation, by contrast, is
#      safe on every channel: it moves pixels without reinterpreting their values.
#      This only holds because the motion channels are undirected (flow is encoded
#      as magnitude, not as a (dx, dy) vector, which a flip would have to negate).
#
# The channel indices below are not generic -- they must match the layout the
# augmentation pipeline writes. See VIAME's train_aug_* pipelines.
# ---------------------------------------------------------------------------

#: Intensity ops for the appearance channels. Additive/nonlinear, so they must
#: never see a motion channel.
_MOTION_INTENSITY_OPS = [
    {"RandomBrightnessContrast": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.5}},
    {"RandomGamma": {"gamma_limit": (80, 120), "p": 0.3}},
]

#: Gain jitter for the motion channels: multiply by a single random factor in
#: [0.7, 1.3] and clip at the dtype ceiling.
#:
#: This is the one photometric-family op that IS safe on a motion channel, and the
#: reason is that it is purely multiplicative. Motion channels encode "no motion"
#: as exactly zero, and 0 * k == 0, so stationary background is preserved bit for
#: bit no matter what factor is drawn. Additive brightness is what breaks that
#: invariant -- it lifts still background off zero and invents motion where the
#: pipeline measured none.
#:
#: What it buys: the absolute scale of every motion channel is an artifact of
#: choices made upstream -- the optical-flow `scale` parameter, the byte
#: scale_factor after the variance and differencing filters, the frame rate, how
#: fast the fish happened to be swimming. A model trained on one fixed scale can
#: latch onto the absolute magnitude. Jittering the gain forces it to key on the
#: spatial structure of the motion instead.
#:
#: per_channel is False on purpose: one factor is shared across all the selected
#: channels, which models a global change in scene motion and preserves the ratios
#: between channels (those ratios are informative -- they are what distinguishes a
#: fast small target from a slow large one). elementwise is False because this is a
#: gain, not per-pixel noise. Clipping at the ceiling is albumentations' default for
#: integer dtypes and is what we want: motion that saturates simply reads as "very
#: fast".
_MOTION_GAIN = [
    {
        "MultiplicativeNoise": {
            "multiplier": (0.7, 1.3),
            "per_channel": False,
            "elementwise": False,
            "p": 0.5,
        }
    },
]

#: Geometric ops shared by the motion presets. Safe on every channel.
_MOTION_GEOMETRIC_OPS = [
    {"HorizontalFlip": {"p": 0.5}},
    {
        "Affine": {
            "scale": (0.9, 1.1),
            "translate_percent": (-0.05, 0.05),
            "rotate": (-10, 10),
            "p": 0.5,
        }
    },
]

# These are lists, not dicts, because a preset needs more than one ChannelSubset
# entry (one for the appearance channels, one for the motion channels) and a dict
# keyed by transform name cannot hold duplicates.

#: RGB + optical-flow magnitude, channel layout [ R, G, B, flow ].
AUG_MOTION_RGB = [
    *_MOTION_GEOMETRIC_OPS,
    {
        "ChannelSubset": {
            "channels": [0, 1, 2],
            "transforms": _MOTION_INTENSITY_OPS + [
                {"ColorJitter": {"brightness": 0.0, "contrast": 0.0, "saturation": 0.2, "hue": 0.05, "p": 0.3}},
            ],
            "p": 1.0,
        }
    },
    {"ChannelSubset": {"channels": [3], "transforms": _MOTION_GAIN, "p": 1.0}},
]

#: Two motion channels around a greyscale one, layout [ motion, grey, motion ].
#: No ColorJitter: saturation and hue are undefined on a single channel.
AUG_MOTION_GREY = [
    *_MOTION_GEOMETRIC_OPS,
    {"ChannelSubset": {"channels": [1], "transforms": _MOTION_INTENSITY_OPS, "p": 1.0}},
    {"ChannelSubset": {"channels": [0, 2], "transforms": _MOTION_GAIN, "p": 1.0}},
]

#: All-motion input with no appearance channel at all. Every channel is a motion
#: channel, so the gain applies to the whole image and no subsetting is needed.
AUG_MOTION_ONLY = [
    *_MOTION_GEOMETRIC_OPS,
    *_MOTION_GAIN,
]
